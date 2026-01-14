"""
viola_etc/runner.py

Main ETC orchestrator: run_etc()

This module stitches together:
- validation
- sky model loading
- signal + background + SNR
- magnitude sweep
- molecule templates (optional)

Goal: keep runner readable; push helpers to dedicated modules.
"""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from constants import (
    ARCSEC2_TO_SR,
    MAG_SWEEP_MAX,
    MAG_SWEEP_MIN,
    MAG_SWEEP_STEP,
    UM_TO_M,
    C_M_S,
    H_J_S,
)

from .models import (
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
from .sky_io import (
    check_sky_grid_vs_resolution,
    get_sky_fits_path,
    load_skytable,
)
from .photometry import (
    fnu_w_m2_hz_to_mjy,
    mag_to_fnu_w_m2_hz,
    planck_blambda_W_m2_um_sr,
    planck_bnu_W_m2_hz_sr,
)
from .molecules import (
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
    molecule_dir: str = "./molecular_lines",
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
    Main ETC entry point (v0.3).

    Returns ETCResult with stable fields for Streamlit UI.
    """
    cfg = get_default_instrument_config() if cfg is None else cfg
    site = get_default_site_config() if site is None else site

    validate_instrument_config(cfg)
    validate_site_config(site)
    validate_user_inputs(u, cfg)
    validate_mag_sweep(mag_sweep_min, mag_sweep_max, mag_sweep_step)

    # --- derived scalars ---
    band = u.target.band.strip().upper()
    lo_um, hi_um = compute_band_window_um(band, cfg)

    area_m2 = compute_telescope_area_m2(cfg.d_m)
    tau_opt = compute_tau_optics(cfg.optics)
    t_total_s = compute_total_exptime_s(u.obs.t_exp_s, u.obs.n_exp)
    pair_factor = variance_pair_factor(u.obs.use_nod_subtraction)

    # --- load sky model ---
    sky_fits_path = get_sky_fits_path(site, u.obs.sky_model_name)
    lam_um_arr, trans_arr, zl_arr, oh_arr, sml_arr, sky_phi_um_arcsec2 = load_skytable(
        sky_fits_path=sky_fits_path,
        band=band,
        cfg=cfg,
        oh_scatter_frac=site.oh_scatter_frac,
    )
    grid_stats = check_sky_grid_vs_resolution(lam_um_arr, cfg.resolving_power)

    # --- mag -> fnu ---
    fnu = mag_to_fnu_w_m2_hz(u.target.m_mag, band, u.target.mag_system)
    f_mjy = fnu_w_m2_hz_to_mjy(fnu)

    # =========================
    # SIGNAL
    # =========================
    lam_m_arr = lam_um_arr * UM_TO_M

    lam_ref_um = float(np.nanmedian(lam_um_arr))
    lam_ref_m = lam_ref_um * UM_TO_M

    # Convert Fnu at reference wavelength to Flambda at that wavelength:
    # F_lambda = (c / lambda^2) * F_nu   then per um => * 1e-6
    F_lambda_ref_um = (C_M_S / lam_ref_m**2) * fnu * 1e-6  # W m^-2 um^-1

    sed = u.target.source_sed.strip().lower()
    if sed == "blackbody":
        B_ref = float(planck_blambda_W_m2_um_sr(np.array([lam_ref_um]), u.target.T_star_K)[0])
        B_arr = planck_blambda_W_m2_um_sr(lam_um_arr, u.target.T_star_K)
        F_lambda_um_arr = F_lambda_ref_um * (B_arr / B_ref)
    else:
        # phoenix placeholder: flat Fnu
        F_lambda_um_arr = (C_M_S / lam_m_arr**2) * fnu * 1e-6

    e_photon_arr = H_J_S * C_M_S / lam_m_arr
    ph_source_arr = F_lambda_um_arr / e_photon_arr  # ph s^-1 m^-2 um^-1

    delta_lambda_um_arr = lam_um_arr / cfg.resolving_power

    # slit throughput (seeing -> slit loss)
    sqrt_ln2 = math.sqrt(math.log(2.0))
    tau_slit = max(
        0.0,
        min(1.0, math.erf(sqrt_ln2 * (cfg.slit_width_as / u.obs.seeing_fwhm_as))),
    )
    tau_slit = float(tau_slit)

    tau_point_arr = tau_opt * tau_slit * trans_arr
    s_res_arr = ph_source_arr * delta_lambda_um_arr * area_m2 * t_total_s * tau_point_arr  # e- / res

    n_pix_per_res = int(cfg.nx * cfg.ny)
    s_pix_arr = s_res_arr / n_pix_per_res

    # =========================
    # BACKGROUND + NOISE
    # =========================
    omega_res_arcsec2 = cfg.slit_width_as * (cfg.pix_scale_ny * cfg.ny)
    omega_res_sr = omega_res_arcsec2 * ARCSEC2_TO_SR

    tau_extend_arr = tau_opt * trans_arr

    eps_sky_res_rate_arr = (
        area_m2
        * omega_res_arcsec2
        * tau_extend_arr
        * delta_lambda_um_arr
        * sky_phi_um_arcsec2
    )
    N_sky_arr = eps_sky_res_rate_arr * t_total_s

    nu_hz_arr = C_M_S / (lam_um_arr * UM_TO_M)
    Bnu_arr = planck_bnu_W_m2_hz_sr(nu_hz_arr, cfg.t_ambient_k)

    # Thermal: (area * omega_sr / (h*R)) * epsilon * Bnu
    eps_th_res_rate_arr = (area_m2 * omega_res_sr / (H_J_S * cfg.resolving_power)) * cfg.epsilon_eff * Bnu_arr
    N_th_arr = eps_th_res_rate_arr * t_total_s

    N_dark_arr = np.full_like(N_sky_arr, cfg.dark_rate * t_total_s * n_pix_per_res)

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
        mol_templates = load_molecule_templates(
            lam_um_arr=lam_um_arr,
            star_name=u.target.star_name,
            band=band,
            resolving_power=cfg.resolving_power,
            molecule_dir=molecule_dir,
            molecules=("CH4", "H2O", "CO2", "CO"),
        )
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
        f"Optics throughput: {tau_opt:.2f}",
        f"Telluric transmission: {np.nanmedian(trans_arr):.3f}",
        f"Sky grid sampling: {grid_stats['dlam_grid_med_nm']:.3f} nm",
        f"Instrument resolution: {grid_stats['dlam_res_med_nm']:.3f} nm (R = {cfg.resolving_power:.0f})",
        f"Slit throughput (seeing→slit): {tau_slit:.2f}",
        f"Flux density: {f_mjy:.2f} mJy",
        f"Median signal per res: {np.nanmedian(s_res_arr):.1f} e-",
        f"Median SNR per res: {np.nanmedian(snr_res_arr):.2f}",
    ]

    if np.isfinite(ratio) and ratio > 1.5:
        summary_lines.append("!!!! Grid is coarse vs instrument resolution; OH/telluric features may be mis-estimated.")

    if enable_molecules:
        if mol_templates:
            summary_lines.append(f"Molecule templates loaded: {', '.join([k.upper() for k in mol_templates.keys()])}")
        else:
            summary_lines.append("Molecule templates loaded: NONE")

    # =========================
    # Meta (debug-friendly)
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

