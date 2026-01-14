"""
etc_core.py

Core ETC computations (early development).

Focus:
- structured inputs (TargetParams, ObservingCondition, UserInputs)
- hidden defaults (InstrumentConfig, SiteConfig)
- load precomputed SkyCalc FITS table (skytable.fits)
- compute signal / background / noise / SNR per resolution element
- optional molecule-template based detection estimator
- return notebook-style outputs as result.summary_lines
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from constants import (
    get_fnu0_w_m2_hz,
    C_M_S,
    H_J_S,
    K_B_J_K,
    UM_TO_M,
    ARCSEC2_TO_SR,
    MAG_SWEEP_MIN,
    MAG_SWEEP_MAX,
    MAG_SWEEP_STEP,
)

# =========================
# Target / planet assumptions
# =========================


@dataclass(frozen=True)
class TargetParams:
    """
    Target star / planet-signal assumptions.
    """
    band: str                 # 'J'/'H'/'K'
    mag_system: str           # 'Vega' or 'AB'
    m_mag: float              # magnitude in chosen band/system

    # Used for molecule template filenames
    star_name: str = "HD21520b"

    # SED selector:
    # - "blackbody": Planck shape (anchored to band magnitude)
    # - "phoenix"  : placeholder (treated as flat Fnu)
    source_sed: str = "blackbody"

    # used by blackbody shape
    T_star_K: float = 5800.0

    # detection knobs
    planet_line_contrast: float = 1e-4


# =========================
# Hidden defaults (instrument + site)
# =========================


@dataclass(frozen=True)
class InstrumentConfig:
    """
    Instrument parameters (hidden defaults).
    """

    # telescope / spectrograph
    d_m: float = 2.0
    resolving_power: float = 300_000.0   # R = lambda / d_lambda
    nx: int = 2                          # pixels per res el (dispersion)
    ny: int = 2                          # spatial pixels/rows summed

    # band windows [um]
    band_edges_um: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "J": (1.10, 1.30),
        "H": (1.50, 1.70),
        "K": (1.90, 2.50),
    })

    # slit / pixel geometry
    slit_width_as: float = 1.6           # arcsec
    pix_scale_ny: float = 0.8            # arcsec/pix (spatial)

    # optics transmissions (0..1 each) -> scalar throughput
    optics: Dict[str, float] = field(default_factory=lambda: {
        "tau_m_1": 0.95,
        "tau_window": 0.95,
        "tau_frd": 0.90,
        "doublet": 0.96,
        "collimator": 0.90,
        "echelle": 0.90,
        "x_disp": 0.80,
        "camera": 0.90,
        "detector": 0.80,
    })

    # thermal & detector
    t_ambient_k: float = 275.0
    epsilon_eff: float = 0.03
    rn_e: float = 20.0
    dark_rate: float = 0.01


@dataclass(frozen=True)
class SiteConfig:
    """
    Site / sky model defaults (hidden).
    """
    sky_model_base_dir: str = "./sky_models"
    oh_scatter_frac: float = 0.10


def get_default_instrument_config() -> InstrumentConfig:
    return InstrumentConfig()


def get_default_site_config() -> SiteConfig:
    return SiteConfig()


# =========================
# Observing condition
# =========================


@dataclass(frozen=True)
class ObservingCondition:
    """
    Observing / site conditions.
    """
    t_exp_s: float
    n_exp: int
    use_nod_subtraction: bool
    seeing_fwhm_as: float

    # sky model folder under ./sky_models
    sky_model_name: str = "skymodel_Paranal_hd21520_pwv1p0"


# =========================
# User-facing inputs
# =========================


@dataclass(frozen=True)
class UserInputs:
    """
    Inputs exposed in the Streamlit UI.
    """
    target: TargetParams
    obs: ObservingCondition


# =========================
# Validation
# =========================


def validate_user_inputs(u: UserInputs, cfg: InstrumentConfig) -> None:
    b = u.target.band.strip().upper()
    if b not in cfg.band_edges_um:
        raise ValueError(f"band must be one of {list(cfg.band_edges_um.keys())}, got {u.target.band!r}")

    sys = u.target.mag_system.strip().upper()
    if sys not in ("VEGA", "AB"):
        raise ValueError("mag_system must be 'Vega' or 'AB'")

    if not np.isfinite(u.target.m_mag):
        raise ValueError("m_mag must be finite")

    sed = u.target.source_sed.strip().lower()
    if sed not in ("blackbody", "phoenix"):
        raise ValueError("source_sed must be 'blackbody' or 'phoenix'")

    if u.target.T_star_K <= 0:
        raise ValueError("T_star_K must be > 0")

    if not np.isfinite(u.target.planet_line_contrast):
        raise ValueError("planet_line_contrast must be finite")
    if not (0.0 <= u.target.planet_line_contrast <= 1.0):
        raise ValueError("planet_line_contrast must be in [0, 1]")

    if u.obs.t_exp_s <= 0:
        raise ValueError("t_exp_s must be > 0")
    if u.obs.n_exp < 1:
        raise ValueError("n_exp must be >= 1")
    if u.obs.seeing_fwhm_as <= 0:
        raise ValueError("seeing_fwhm_as must be > 0")

    if not isinstance(u.obs.sky_model_name, str) or len(u.obs.sky_model_name.strip()) == 0:
        raise ValueError("sky_model_name must be a non-empty string")

    if not isinstance(u.target.star_name, str) or len(u.target.star_name.strip()) == 0:
        raise ValueError("star_name must be a non-empty string")


def validate_instrument_config(cfg: InstrumentConfig) -> None:
    if cfg.d_m <= 0:
        raise ValueError("d_m must be > 0")
    if cfg.resolving_power <= 0:
        raise ValueError("resolving_power must be > 0")
    if cfg.nx <= 0 or cfg.ny <= 0:
        raise ValueError("nx and ny must be > 0")
    if cfg.slit_width_as <= 0 or cfg.pix_scale_ny <= 0:
        raise ValueError("slit_width_as and pix_scale_ny must be > 0")

    for k, (lo, hi) in cfg.band_edges_um.items():
        if not np.isfinite(lo) or not np.isfinite(hi):
            raise ValueError(f"band_edges_um values must be finite. Got: {k}: {(lo, hi)}")
        if lo <= 0 or hi <= 0 or hi <= lo:
            raise ValueError(f"band_edges_um must satisfy 0 < lo < hi. Got: {k}: {(lo, hi)}")

    vals = np.array(list(cfg.optics.values()), dtype=float)
    if np.any(vals < 0.0) or np.any(vals > 1.0):
        raise ValueError(f"Optics transmissions must be in [0,1]. Got: {cfg.optics}")


def validate_site_config(site: SiteConfig) -> None:
    if not isinstance(site.sky_model_base_dir, str) or len(site.sky_model_base_dir.strip()) == 0:
        raise ValueError("sky_model_base_dir must be a non-empty string")
    if not (0.0 <= site.oh_scatter_frac <= 1.0):
        raise ValueError("oh_scatter_frac must be in [0, 1]")


def validate_mag_sweep(mag_min: float, mag_max: float, mag_step: float) -> None:
    if not (np.isfinite(mag_min) and np.isfinite(mag_max) and np.isfinite(mag_step)):
        raise ValueError("mag sweep inputs must be finite")
    if mag_step <= 0:
        raise ValueError("mag_sweep_step must be > 0")
    if mag_max <= mag_min:
        raise ValueError("mag_sweep_max must be > mag_sweep_min")


# =========================
# Derived quantities
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


def get_sky_fits_path(site: SiteConfig, sky_model_name: str) -> str:
    # expected: ./sky_models/<sky_model_name>/skytable.fits
    name = str(sky_model_name).strip()
    if len(name) == 0:
        raise ValueError("sky_model_name must be a non-empty string")
    return str(Path(site.sky_model_base_dir) / name / "skytable.fits")


# =========================
# Sky model I/O
# =========================


def load_skytable(
    sky_fits_path: str,
    band: str,
    cfg: InstrumentConfig,
    oh_scatter_frac: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Read SkyCalc FITS and trim to band.

    Returns
    -------
    lam_um_arr : wavelength [um]
    trans_arr  : transmission [0..1]
    zl_arr     : zodiacal emission [ph s^-1 m^-2 um^-1 arcsec^-2]
    oh_arr     : OH emission [same units]
    sml_arr    : scattered moonlight [same units]
    sky_phi_um_arcsec2 : zl + sml + oh_scatter_frac*oh
    """
    try:
        from astropy.io import fits
    except ImportError as e:
        raise ImportError("astropy required for SkyCalc FITS. pip install astropy") from e

    def _get_col(data, candidates, default=None):
        for name in candidates:
            if name in data.names:
                return np.array(data[name], dtype=float)
        if default is not None:
            return np.full(len(data), float(default))
        raise KeyError(f"None of {candidates} found. Available: {list(data.names)}")

    p = Path(sky_fits_path)
    if not p.exists():
        raise FileNotFoundError(f"Sky model FITS not found: {sky_fits_path}")

    with fits.open(str(p)) as hdul:
        t = hdul[1].data

        lam_nm = _get_col(t, ["lam", "wavelength", "lambda"])
        lam_um = lam_nm * 1e-3  # nm -> um

        trans = _get_col(t, ["trans"])

        zl = _get_col(t, ["flux_zl", "phi_zl", "zodiacal"], default=0.0)
        oh = _get_col(t, ["flux_ael", "phi_oh", "oh_emiss"], default=0.0)
        sml = _get_col(t, ["flux_sml", "moon_scattered", "moonlight"], default=0.0)

        sky_phi_um_arcsec2_full = zl + sml + oh_scatter_frac * oh

    b = band.strip().upper()
    if b not in cfg.band_edges_um:
        raise ValueError(f"band must be one of {list(cfg.band_edges_um.keys())}, got {band!r}")

    lam_min, lam_max = cfg.band_edges_um[b]
    m = (lam_um >= lam_min) & (lam_um <= lam_max)

    lam_um_arr = lam_um[m]
    trans_arr = trans[m]
    zl_arr = zl[m]
    oh_arr = oh[m]
    sml_arr = sml[m]
    sky_phi_um_arcsec2 = sky_phi_um_arcsec2_full[m]

    if lam_um_arr.size == 0:
        raise ValueError(f"No sky samples in {b} window {cfg.band_edges_um[b]} for file {sky_fits_path}")

    return lam_um_arr, trans_arr, zl_arr, oh_arr, sml_arr, sky_phi_um_arcsec2


# =========================
# Sky model validation
# =========================


def check_sky_grid_vs_resolution(lam_um_arr: np.ndarray, resolving_power: float) -> Dict[str, float]:
    # compare SkyCalc grid vs instrument dlam = lam/R
    if lam_um_arr.size < 2:
        return {"dlam_grid_med_nm": np.nan, "dlam_res_med_nm": np.nan, "ratio_grid_to_res": np.nan}

    dlam_grid_um_med = float(np.median(np.diff(lam_um_arr)))
    dlam_res_um_med = float(np.median(lam_um_arr / resolving_power))
    ratio = float(dlam_grid_um_med / dlam_res_um_med) if dlam_res_um_med > 0 else np.nan

    return {
        "dlam_grid_med_nm": dlam_grid_um_med * 1e3,
        "dlam_res_med_nm": dlam_res_um_med * 1e3,
        "ratio_grid_to_res": ratio,
    }


# =========================
# Photometry + Planck
# =========================


def mag_to_fnu_w_m2_hz(mag: float, band: str, mag_system: str) -> float:
    # Fnu = Fnu0 * 10^(-0.4*mag)
    fnu0 = get_fnu0_w_m2_hz(band=band, mag_system=mag_system)
    return float(fnu0 * 10.0 ** (-0.4 * mag))


def fnu_w_m2_hz_to_mjy(fnu: float) -> float:
    # 1 mJy = 1e-29 W m^-2 Hz^-1
    return float(fnu / 1e-29)


def planck_blambda_W_m2_um_sr(lam_um: np.ndarray, T_K: float) -> np.ndarray:
    # B_lambda [W m^-2 sr^-1 um^-1]
    lam_m = lam_um * UM_TO_M
    x = (H_J_S * C_M_S) / (lam_m * K_B_J_K * T_K)
    B_m = (2.0 * H_J_S * C_M_S**2) / (lam_m**5) / np.expm1(x)  # per meter
    return B_m * 1e-6  # per um


def planck_bnu_W_m2_hz_sr(nu_hz: np.ndarray, T_K: float) -> np.ndarray:
    # B_nu [W m^-2 sr^-1 Hz^-1]
    x = (H_J_S * nu_hz) / (K_B_J_K * T_K)
    return (2.0 * H_J_S * nu_hz**3 / C_M_S**2) / np.expm1(x)


# =========================
# SNR vs magnitude sweep
# =========================


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
# Molecule templates
# =========================


def _normalize_minmax_safe(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    m = np.isfinite(y)
    if not np.any(m):
        return np.full_like(y, np.nan, dtype=float)

    lo = float(np.nanmin(y[m]))
    hi = float(np.nanmax(y[m]))
    if (not np.isfinite(lo)) or (not np.isfinite(hi)) or (hi <= lo):
        out = np.zeros_like(y, dtype=float)
        out[~m] = np.nan
        return out

    out = (y - lo) / (hi - lo)
    out[~m] = np.nan
    return out


def _interp_to_grid(wl_src: np.ndarray, y_src: np.ndarray, wl_tgt: np.ndarray) -> np.ndarray:
    wl_src = np.asarray(wl_src, float)
    y_src = np.asarray(y_src, float)
    wl_tgt = np.asarray(wl_tgt, float)

    m = np.isfinite(wl_src) & np.isfinite(y_src)
    if not np.any(m):
        return np.full_like(wl_tgt, np.nan, dtype=float)

    wl = wl_src[m]
    yy = y_src[m]

    order = np.argsort(wl)
    wl = wl[order]
    yy = yy[order]

    # de-duplicate wl
    uniq, inv = np.unique(wl, return_inverse=True)
    if uniq.size != wl.size:
        yy2 = np.bincount(inv, weights=yy) / np.maximum(1, np.bincount(inv))
        wl = uniq
        yy = yy2

    return np.interp(wl_tgt, wl, yy, left=np.nan, right=np.nan)


def _try_find_peaks(y: np.ndarray, prominence: float, distance: int) -> np.ndarray:
    y = np.asarray(y, float)
    distance = int(max(1, distance))

    try:
        from scipy.signal import find_peaks  # type: ignore
        pks, _props = find_peaks(y, prominence=prominence, distance=distance)
        return np.asarray(pks, dtype=int)
    except Exception:
        # fallback: very simple local maxima
        m = np.isfinite(y)
        if np.sum(m) < 3:
            return np.array([], dtype=int)

        yy = np.copy(y)
        yy[~m] = -np.inf

        # local maxima
        left = yy[1:-1] > yy[:-2]
        right = yy[1:-1] > yy[2:]
        p = np.where(left & right)[0] + 1

        # crude prominence filter
        keep = []
        for idx in p:
            lo = max(0, idx - distance)
            hi = min(len(yy), idx + distance + 1)
            base = np.nanmin(yy[lo:hi])
            prom = yy[idx] - base
            if np.isfinite(prom) and prom >= prominence:
                keep.append(idx)

        if not keep:
            return np.array([], dtype=int)

        # enforce minimum spacing
        keep = np.array(sorted(keep), dtype=int)
        out = [int(keep[0])]
        for i in keep[1:]:
            if i - out[-1] >= distance:
                out.append(int(i))
        return np.asarray(out, dtype=int)


def load_molecule_templates(
    lam_um_arr: np.ndarray,
    star_name: str,
    band: str,
    resolving_power: float,
    molecule_dir: str = "./molecular_lines",
    molecules: Tuple[str, ...] = ("CH4", "H2O", "CO2", "CO"),
) -> Dict[str, Dict[str, Any]]:
    """
    Loads files like:
      {star_name}_{MOL}_{BAND}_band_R{R}.csv
    Example:
      HD21520b_CH4_H_band_R300000.csv

    Expected columns:
      wl_micron, transit_cm
    """
    out: Dict[str, Dict[str, Any]] = {}

    base = Path(molecule_dir)
    if not base.exists():
        return out

    band_u = band.strip().upper()
    star = star_name.strip()

    # match your filenames exactly
    R_int = int(round(float(resolving_power)))

    for mol in molecules:
        fn = f"{star}_{mol}_{band_u}_band_R{R_int}.csv"
        p = base / fn
        if not p.exists():
            continue

        # load (no pandas required)
        try:
            data = np.genfromtxt(str(p), delimiter=",", names=True, dtype=None, encoding=None)
            if "wl_micron" not in data.dtype.names or "transit_cm" not in data.dtype.names:
                continue

            wl = np.asarray(data["wl_micron"], float)
            y = np.asarray(data["transit_cm"], float)

        except Exception:
            continue

        y0 = _normalize_minmax_safe(y)
        y_res = _interp_to_grid(wl, y0, lam_um_arr)
        y_res = _normalize_minmax_safe(y_res)

        out[mol.lower()] = {
            "path": str(p),
            "wl_um": wl,
            "template_raw": y0,
            "template_resampled": y_res,
        }

    return out


def estimate_transits_for_molecules(
    templates: Dict[str, Dict[str, Any]],
    lam_um_arr: np.ndarray,
    snr_res_arr: np.ndarray,
    nx: int,
    prom_abs: float,
    snr_thresh: float,
    n_frames_per_transit: int,
    detrend_p: float,
    line_contrast: float,
    detect_sig: float,
    min_lines_for_calc: int,
    find_valleys: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    Your style: count lines from peak finding on normalized template.
    Then estimate:
      snr_transit ≈ line_contrast * median(snr/res) * sqrt(n_lines * n_frames_per_transit * detrend_p)
      n_transits  ≈ (detect_sig / snr_transit)^2
    """
    out: Dict[str, Dict[str, Any]] = {}

    snr = np.asarray(snr_res_arr, float)
    snr_med = float(np.nanmedian(snr)) if np.any(np.isfinite(snr)) else float("nan")

    dist = int(max(1, int(nx)))

    for mol, rec in templates.items():
        y = np.asarray(rec.get("template_resampled", None), float)
        if y.size == 0:
            continue

        y_in = -y if find_valleys else y
        pks = _try_find_peaks(y_in, prominence=float(prom_abs), distance=dist)

        # filter by local SNR
        if pks.size:
            s_at = snr[pks]
            keep = np.isfinite(s_at) & (s_at >= float(snr_thresh))
            pks_keep = pks[keep]
        else:
            pks_keep = np.array([], dtype=int)

        n_lines = int(pks_keep.size)

        if n_lines >= int(min_lines_for_calc) and np.isfinite(snr_med):
            spt = float(line_contrast) * snr_med * math.sqrt(float(n_lines) * float(n_frames_per_transit) * float(detrend_p))
            ntr = (float(detect_sig) / spt) ** 2 if spt > 0 else float("inf")
        else:
            spt = float("nan")
            ntr = float("nan")

        out[mol] = {
            "peaks_idx": pks,
            "peaks_idx_thr": pks_keep,
            "n_lines": n_lines,
            "snr_per_transit": spt,
            "n_transits_req": ntr,
        }

    return out


# =========================
# Result container
# =========================


@dataclass
class ETCResult:
    meta: Dict[str, Any]
    summary_lines: Optional[List[str]] = None

    lam_um: Optional[np.ndarray] = None
    trans: Optional[np.ndarray] = None
    zl: Optional[np.ndarray] = None
    oh: Optional[np.ndarray] = None
    sml: Optional[np.ndarray] = None
    sky_phi_um_arcsec2: Optional[np.ndarray] = None

    # signal/noise/SNR arrays (per res el unless stated)
    signal_res_e: Optional[np.ndarray] = None
    signal_pix_e: Optional[np.ndarray] = None
    sky_res_e: Optional[np.ndarray] = None
    thermal_res_e: Optional[np.ndarray] = None
    dark_res_e: Optional[np.ndarray] = None
    snr_res: Optional[np.ndarray] = None

    # median SNR vs magnitude
    mag_grid: Optional[np.ndarray] = None
    snr_med_grid: Optional[np.ndarray] = None

    # molecules
    molecule_templates: Optional[Dict[str, Dict[str, Any]]] = None
    molecule_metrics: Optional[Dict[str, Dict[str, Any]]] = None


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
    # molecule knobs (defaults = your notebook knobs)
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

    Steps:
    - validate inputs
    - load sky model arrays
    - compute signal / noise / SNR
    - build SNR vs mag sweep
    - optional molecule template estimator
    - build summary + meta
    """
    cfg = get_default_instrument_config() if cfg is None else cfg
    site = get_default_site_config() if site is None else site

    validate_instrument_config(cfg)
    validate_site_config(site)
    validate_user_inputs(u, cfg)
    validate_mag_sweep(mag_sweep_min, mag_sweep_max, mag_sweep_step)

    # --- basic derived scalars ---
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
    mol_templates: Optional[Dict[str, Dict[str, Any]]] = None
    mol_metrics: Optional[Dict[str, Dict[str, Any]]] = None

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
    ratio = grid_stats["ratio_grid_to_res"]

    summary_lines: List[str] = [
        f"Band: {band}  [{lo_um:.3f}–{hi_um:.3f} μm]",
        f"Exposures: N = {u.obs.n_exp:d},   texp = {u.obs.t_exp_s:.0f} s,   Total = {t_total_s:.0f} s",
        f"A–B nod subtraction: {pair_txt}",
        f"Optics throughput: {tau_opt:.2f}",
        f"Telluric transmission: {np.nanmedian(trans_arr):.3f}",
        f"Sky grid sampling: {grid_stats['dlam_grid_med_nm']:.3f} nm",
        f"Instrument resolution: {grid_stats['dlam_res_med_nm']:.3f} nm (R = {cfg.resolving_power:.0f})",
        f"Slit throughput (seeing→slit): {tau_slit:.2f}",
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
