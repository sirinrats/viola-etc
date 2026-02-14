"""
viola_etc/runner.py

Main ETC orchestrator: run_etc()

This module stitches together:
- validation
- sky model loading
- signal + background + SNR
- magnitude sweep
- molecule templates (optional)

"""

from __future__ import annotations

import inspect
import math
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..constants import (
    ARCSEC2_TO_SR,
    MAG_SWEEP_MAX,
    MAG_SWEEP_MIN,
    MAG_SWEEP_STEP,
    UM_TO_M,
    C_M_S,
    H_J_S,
)

from ..models import (
    ETCResult,
    InstrumentConfig,
    SiteConfig,
    UserInputs,
    get_default_instrument_config,
    get_default_site_config,
)

from .validate import (
    validate_instrument_config,
    validate_mag_sweep,
    validate_site_config,
    validate_user_inputs,
)
from ..observing.sky_io import (
    check_sky_grid_vs_resolution,
    get_sky_fits_path,
    load_skytable,
)
from ..target.photometry import (
    fnu_w_m2_hz_to_mjy,
    mag_to_fnu_w_m2_hz,
    planck_bnu_W_m2_hz_sr,
    stellar_flambda_um_arr,
)
from ..target.molecules import (
    estimate_transits_for_molecules,
    load_molecule_templates,
)


# =========================
# Small internal helpers
# =========================

def compute_band_window_um(band: str, cfg: InstrumentConfig) -> Tuple[float, float]:
    b = band.strip().upper()
    if b not in cfg.band_edges_um:
        raise ValueError(f"band must be one of {list(cfg.band_edges_um.keys())}, got {band!r}")
    return cfg.band_edges_um[b]


def compute_total_exptime_s(t_exp_s: float, n_exp: int) -> float:
    return float(t_exp_s * n_exp)


def compute_telescope_area_m2(d_m: float) -> float:
    return float(np.pi * (d_m / 2.0) ** 2)


def compute_tau_optics(optics: Dict[str, float]) -> float:
    vals = np.array(list(optics.values()), dtype=float)
    if np.any(vals < 0.0) or np.any(vals > 1.0):
        raise ValueError(f"Optics transmissions must be in [0,1]. Got: {optics}")
    return float(np.prod(vals))


def variance_pair_factor(use_nod_subtraction: bool) -> float:
    # A–B subtraction: variances add -> factor ~2
    return 2.0 if use_nod_subtraction else 1.0


def build_snr_mag_sweep(
    s_res_arr: np.ndarray,
    m_mag: float,
    V_bg_base_arr: np.ndarray,
    V_read: float,
    mag_min: float,
    mag_max: float,
    mag_step: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute median SNR per resolution element as a function of magnitude.

    We assume background terms stay fixed (same sky/thermal/dark/read) and only
    scale the signal with magnitude.
    """
    # base signal at m=0
    scale_current = 10.0 ** (-0.4 * m_mag)
    s_res_base0 = s_res_arr / scale_current

    mag_grid = np.arange(mag_min, mag_max + 0.5 * mag_step, mag_step, dtype=float)
    snr_med_grid = np.zeros_like(mag_grid)

    for i, m in enumerate(mag_grid):
        s_res = s_res_base0 * 10.0 ** (-0.4 * m)
        V_tot = V_bg_base_arr + V_read + s_res
        snr = np.where(V_tot > 0, s_res / np.sqrt(V_tot), 0.0)
        snr_med_grid[i] = float(np.nanmedian(snr))

    return mag_grid, snr_med_grid


# =========================
# Main runner
# =========================

def run_etc(
    u: UserInputs,
    cfg: Optional[InstrumentConfig] = None,
    site: Optional[SiteConfig] = None,
    mag_sweep_min: float = MAG_SWEEP_MIN,
    mag_sweep_max: float = MAG_SWEEP_MAX,
    mag_sweep_step: float = MAG_SWEEP_STEP,
    molecule_dir: str = "./molecular_templates",
    enable_molecules: bool = True,
    # molecule knobs
    prom_abs: float = 0.05,
    snr_thresh: float = 100.0,
    n_frames_per_transit: int = 50,
    detrend_p: float = 0.5,
    detect_sig: float = 5.0,
    min_lines_for_calc: int = 50,
    find_valleys: bool = False,
) -> ETCResult:
    """
    Main ETC entry point.

    Physically ordered pipeline:
      Target (TOA) -> Atmosphere -> Telescope/Instrument -> Detector

    Returns ETCResult with stable fields for Streamlit UI.
    """
    # -------------------------
    # Defaults + validation
    # -------------------------
    cfg = get_default_instrument_config() if cfg is None else cfg
    site = get_default_site_config() if site is None else site

    validate_instrument_config(cfg)
    validate_site_config(site)
    validate_user_inputs(u, cfg)
    validate_mag_sweep(mag_sweep_min, mag_sweep_max, mag_sweep_step)

    # -------------------------
    # Derived scalars
    # -------------------------
    band = u.target.band.strip().upper()
    lo_um, hi_um = compute_band_window_um(band, cfg)

    area_m2 = compute_telescope_area_m2(cfg.d_m)
    tau_opt = compute_tau_optics(cfg.optics)
    t_total_s = compute_total_exptime_s(u.obs.t_exp_s, u.obs.n_exp)
    pair_factor = variance_pair_factor(u.obs.use_nod_subtraction)

    # -------------------------
    # Load atmosphere/sky model
    # -------------------------
    sky_fits_path = get_sky_fits_path(
    site,
    u.obs.sky_model_name,
    target_alt_deg=u.obs.target_alt_deg,
    pwv_mm=u.obs.pwv_mm,
    )

    lam_um_arr, trans_arr, zl_arr, oh_arr, sml_arr, sky_phi_um_arcsec2 = load_skytable(
        sky_fits_path=sky_fits_path,
        band=band,
        cfg=cfg,
        oh_scatter_frac=site.oh_scatter_frac,
    )
    grid_stats = check_sky_grid_vs_resolution(lam_um_arr, cfg.resolving_power)

    # -------------------------
    # Target photometry (sets the TOA flux scale)
    # -------------------------
    fnu = mag_to_fnu_w_m2_hz(u.target.m_mag, band, u.target.mag_system)
    f_mjy = fnu_w_m2_hz_to_mjy(fnu)

    lam_m_arr = lam_um_arr * UM_TO_M  # keep this (used later)

    F_lambda_um_arr, sed_meta = stellar_flambda_um_arr(
        lam_um_arr=lam_um_arr,
        band=band,
        mag_system=u.target.mag_system,
        m_mag=u.target.m_mag,
        source_sed=u.target.source_sed,
        T_star_K=u.target.T_star_K,
        phoenix_newera_dir=None,
    )


    # =========================
    # SIGNAL (Target -> Atmosphere -> Instrument -> Detector)
    # =========================
    # --- (1) photons from target above atmosphere (TOA) ---
    e_photon_arr = H_J_S * C_M_S / lam_m_arr
    ph_toa_rate_arr = (F_lambda_um_arr / e_photon_arr)  # ph s^-1 m^-2 um^-1

    # per-resolution bandwidth
    delta_lambda_um_arr = lam_um_arr / cfg.resolving_power

    # photons collected by telescope in one exposure set (still TOA, not yet attenuated)
    N_toa_res_arr = ph_toa_rate_arr * delta_lambda_um_arr * area_m2 * t_total_s  # photons / res

    # --- (2) atmosphere transmission (tellurics) ---
    N_ground_res_arr = N_toa_res_arr * trans_arr

    # --- (3) slit throughput (seeing -> slit loss) ---
    sqrt_ln2 = math.sqrt(math.log(2.0))
    tau_slit = max(
        0.0,
        min(1.0, math.erf(sqrt_ln2 * (cfg.slit_width_as / u.obs.seeing_fwhm_as))),
    )
    tau_slit = float(tau_slit)

    # --- (4) telescope + instrument optics ---
    # Final detected signal electrons per resolution element
    s_res_arr = N_ground_res_arr * (tau_opt * tau_slit)  # e- / res

    n_pix_per_res = int(cfg.nx * cfg.ny)
    s_pix_arr = s_res_arr / n_pix_per_res

    # =========================
    # BACKGROUND + NOISE (origins matter)
    # =========================
    # Extraction solid angle per resolution element:
    omega_res_arcsec2 = cfg.slit_width_as * (cfg.pix_scale_ny * cfg.ny)
    omega_res_sr = omega_res_arcsec2 * ARCSEC2_TO_SR

    # --- Sky emission background (origin: atmosphere) ---
    # IMPORTANT subtlety:
    # - If sky_phi_um_arcsec2 is already the sky brightness at the telescope (ground),
    #   you should NOT multiply by trans_arr again.
    # - Your current behavior multiplies by trans_arr (kept for backward compatibility).
    #
    # If later you confirm sky_phi is "at ground", change:
    #     tau_sky_path_arr = tau_opt
    # instead of:
    #     tau_sky_path_arr = tau_opt * trans_arr
    tau_sky_path_arr = tau_opt * trans_arr

    sky_res_rate_arr = (
        area_m2
        * omega_res_arcsec2
        * tau_sky_path_arr
        * delta_lambda_um_arr
        * sky_phi_um_arcsec2
    )
    N_sky_arr = sky_res_rate_arr * t_total_s

    # --- Instrument/telescope thermal background (origin: instrument) ---
    nu_hz_arr = C_M_S / (lam_um_arr * UM_TO_M)
    Bnu_arr = planck_bnu_W_m2_hz_sr(nu_hz_arr, cfg.t_ambient_k)

    # Thermal rate per res element:
    # (area * omega_sr / (h*R)) * epsilon * Bnu
    th_res_rate_arr = (area_m2 * omega_res_sr / (H_J_S * cfg.resolving_power)) * cfg.epsilon_eff * Bnu_arr
    N_th_arr = th_res_rate_arr * t_total_s

    # --- Detector backgrounds ---
    N_dark_arr = np.full_like(N_sky_arr, cfg.dark_rate * t_total_s * n_pix_per_res)

    # --- Variances ---
    V_bg_poisson_arr = pair_factor * (N_sky_arr + N_th_arr + N_dark_arr)
    V_read = pair_factor * (cfg.rn_e**2) * u.obs.n_exp * n_pix_per_res
    V_sig_arr = s_res_arr

    V_total_arr = V_sig_arr + V_bg_poisson_arr + V_read
    snr_res_arr = np.where(V_total_arr > 0, s_res_arr / np.sqrt(V_total_arr), 0.0)

    # =========================
    # Median SNR vs magnitude
    # =========================
    V_bg_base_arr = pair_factor * (N_sky_arr + N_th_arr + N_dark_arr)
    mag_grid, snr_med_grid = build_snr_mag_sweep(
        s_res_arr=s_res_arr,
        m_mag=u.target.m_mag,
        V_bg_base_arr=V_bg_base_arr,
        V_read=V_read,
        mag_min=mag_sweep_min,
        mag_max=mag_sweep_max,
        mag_step=mag_sweep_step,
    )

    # =========================
    # Molecules (optional)
    # =========================
    mol_templates: Dict[str, Dict[str, Any]] = {}
    mol_metrics: Dict[str, Dict[str, Any]] = {}

    if enable_molecules:
        # Signature-safe call (prevents passing unsupported kwargs like star_name)
        sig = inspect.signature(load_molecule_templates)
        lt_kwargs: Dict[str, Any] = {}

        # wavelength grid
        if "lam_um_arr" in sig.parameters:
            lt_kwargs["lam_um_arr"] = lam_um_arr
        elif "lam_um" in sig.parameters:
            lt_kwargs["lam_um"] = lam_um_arr

        # band / resolution
        if "band" in sig.parameters:
            lt_kwargs["band"] = band
        if "resolving_power" in sig.parameters:
            lt_kwargs["resolving_power"] = cfg.resolving_power
        elif "R" in sig.parameters:
            lt_kwargs["R"] = cfg.resolving_power

        # templates directory
        if "molecule_dir" in sig.parameters:
            lt_kwargs["molecule_dir"] = molecule_dir
        elif "template_dir" in sig.parameters:
            lt_kwargs["template_dir"] = molecule_dir

        # molecules list
        default_mols = ("CH4", "H2O", "CO2", "CO")
        if "molecules" in sig.parameters:
            lt_kwargs["molecules"] = default_mols
        elif "mols" in sig.parameters:
            lt_kwargs["mols"] = default_mols

        # star_name ONLY if supported
        if "star_name" in sig.parameters:
            lt_kwargs["star_name"] = getattr(u.target, "star_name", "")

        mol_templates = load_molecule_templates(**lt_kwargs)

        if mol_templates:
            mol_metrics = estimate_transits_for_molecules(
                templates=mol_templates,
                lam_um_arr=lam_um_arr,
                snr_res_arr=snr_res_arr,
                nx=cfg.nx,
                prom_abs=prom_abs,
                snr_thresh=snr_thresh,
                n_frames_per_transit=n_frames_per_transit,
                detrend_p=detrend_p,
                line_contrast=u.target.planet_line_contrast,
                detect_sig=detect_sig,
                min_lines_for_calc=min_lines_for_calc,
                find_valleys=find_valleys,
            )
        else:
            mol_metrics = {}

    # =========================
    # Summary lines
    # =========================
    pair_txt = "ON" if u.obs.use_nod_subtraction else "OFF"
    ratio = float(grid_stats.get("ratio_grid_to_res", np.nan))

    summary_lines: List[str] = [
        f"Band: {band}  [{lo_um:.3f}–{hi_um:.3f} μm]",
        f"Exposures: N = {u.obs.n_exp:d},   texp = {u.obs.t_exp_s:.0f} s,   Total = {t_total_s:.0f} s",
        f"A–B nod subtraction: {pair_txt}",
        f"Telescope diameter: {cfg.d_m:.1f} m",
        f"Optics throughput: {tau_opt:.2f}",
        f"Telluric transmission: {np.nanmedian(trans_arr):.3f}",
        f"Sky grid sampling: {grid_stats['dlam_grid_med_nm']:.3f} nm",
        f"Instrument resolution: {grid_stats['dlam_res_med_nm']:.3f} nm (R = {cfg.resolving_power:.0f})",
        f"Slit throughput (seeing → slit): {tau_slit:.2f}",
        f"Flux density: {f_mjy:.2f} mJy",
        f"Median SNR per resolution: {np.nanmedian(snr_res_arr):.0f}",
    ]

    if sed_meta.get("sed_fallback_reason"):
        summary_lines.append(
            f"SED: {sed_meta.get('sed_requested')} (fallback → {sed_meta.get('sed_used')}; {sed_meta.get('sed_fallback_reason')})"
        )
    else:
        summary_lines.append(f"SED: {sed_meta.get('sed_used')}")


    if np.isfinite(ratio) and ratio > 1.5:
        summary_lines.append("!!!! Grid is coarse vs instrument resolution; OH/telluric features may be mis-estimated.")

    if enable_molecules:
        if mol_templates:
            summary_lines.append(f"Molecule templates loaded: {', '.join([k.upper() for k in mol_templates.keys()])}")
        else:
            summary_lines.append("Molecule templates loaded: NONE")

    # =========================
    # Metadata (debug-friendly)
    # =========================
    meta: Dict[str, Any] = {
        "user_inputs": asdict(u),
        "instrument_config": asdict(cfg),
        "site_config": asdict(site),
        "band_window_um": (lo_um, hi_um),
        "t_total_s": float(t_total_s),
        "telescope_area_m2": float(area_m2),
        "tau_optics_scalar": float(tau_opt),
        "variance_pair_factor": float(pair_factor),
        "sky_model_name": u.obs.sky_model_name,
        "sky_fits_path": sky_fits_path,
        "sky_n_samples": int(lam_um_arr.size),
        "sky_lam_um_min": float(lam_um_arr.min()),
        "sky_lam_um_max": float(lam_um_arr.max()),
        "sky_trans_median": float(np.nanmedian(trans_arr)),
        "dlam_grid_med_nm": float(grid_stats["dlam_grid_med_nm"]),
        "dlam_res_med_nm": float(grid_stats["dlam_res_med_nm"]),
        "ratio_grid_to_res": float(grid_stats["ratio_grid_to_res"]),
        "fnu_w_m2_hz": float(fnu),
        "flux_mjy": float(f_mjy),
        "sed_requested": sed_meta.get("sed_requested"),
        "sed_used": sed_meta.get("sed_used"),
        "sed_fallback_reason": sed_meta.get("sed_fallback_reason"),
        "phoenix_file": sed_meta.get("phoenix_file"),
        "phoenix_teff_selected_k": sed_meta.get("phoenix_teff_selected_k"),
        "tau_slit": float(tau_slit),
        "omega_res_arcsec2": float(omega_res_arcsec2),
        "read_variance_e2": float(V_read),
        "median_signal_res_e": float(np.nanmedian(s_res_arr)),
        "median_snr_res": float(np.nanmedian(snr_res_arr)),
        "median_sky_res_e": float(np.nanmedian(N_sky_arr)),
        "median_thermal_res_e": float(np.nanmedian(N_th_arr)),
        "median_dark_res_e": float(np.nanmedian(N_dark_arr)),
        "mag_sweep_min": float(mag_sweep_min),
        "mag_sweep_max": float(mag_sweep_max),
        "mag_sweep_step": float(mag_sweep_step),
        "molecule_dir": str(molecule_dir),
        "enable_molecules": bool(enable_molecules),
        "prom_abs": float(prom_abs),
        "snr_thresh": float(snr_thresh),
        "n_frames_per_transit": int(n_frames_per_transit),
        "detrend_p": float(detrend_p),
        "detect_sig": float(detect_sig),
        "min_lines_for_calc": int(min_lines_for_calc),
        "find_valleys": bool(find_valleys),
    }

    return ETCResult(
        meta=meta,
        summary_lines=summary_lines,
        lam_um=lam_um_arr,
        trans=trans_arr,
        zl=zl_arr,
        oh=oh_arr,
        sml=sml_arr,
        sky_phi_um_arcsec2=sky_phi_um_arcsec2,
        signal_res_e=s_res_arr,
        signal_pix_e=s_pix_arr,
        sky_res_e=N_sky_arr,
        thermal_res_e=N_th_arr,
        dark_res_e=N_dark_arr,
        snr_res=snr_res_arr,
        mag_grid=mag_grid,
        snr_med_grid=snr_med_grid,
        molecule_templates=mol_templates,
        molecule_metrics=mol_metrics,
    )
