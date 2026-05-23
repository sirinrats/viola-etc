"""
viola_etc/target/photometry.py

Photometry + Planck helper functions.

Notes:
- mag -> Fnu uses constants.get_fnu0_w_m2_hz
- Planck functions are pure physics helpers
"""

from __future__ import annotations
from pathlib import Path
import numpy as np

from ..constants import (
    C_M_S, H_J_S, K_B_J_K, UM_TO_M, get_fnu0_w_m2_hz,
    PHOENIX_NEWERA_TEFF_MIN, PHOENIX_NEWERA_TEFF_MAX,
    PHOENIX_NEWERA_GRID_DIR,
    PHOENIX_NEWERA_CACHE_DIR,
    PHOENIX_NEWERA_REMOTE_BASE_URL,
)

from .newera_io import ensure_newera_grid_file

try:
    import streamlit as st
except Exception:
    st = None


def mag_to_fnu_w_m2_hz(mag: float, band: str, mag_system: str) -> float:
    """
    Convert magnitude -> F_nu [W m^-2 Hz^-1]

    Fnu = Fnu0 * 10^(-0.4*mag)
    """
    fnu0 = get_fnu0_w_m2_hz(band=band, mag_system=mag_system)
    return float(fnu0 * 10.0 ** (-0.4 * float(mag)))


def fnu_w_m2_hz_to_mjy(fnu: float) -> float:
    """
    Convert F_nu [W m^-2 Hz^-1] -> mJy

    1 mJy = 1e-29 W m^-2 Hz^-1
    """
    return float(float(fnu) / 1e-29)


def planck_blambda_W_m2_um_sr(lam_um: np.ndarray, T_K: float) -> np.ndarray:
    """
    Planck function B_lambda in units:
      W m^-2 sr^-1 um^-1
    """
    lam_um = np.asarray(lam_um, dtype=float)
    lam_m = lam_um * UM_TO_M

    x = (H_J_S * C_M_S) / (lam_m * K_B_J_K * float(T_K))
    B_m = (2.0 * H_J_S * C_M_S**2) / (lam_m**5) / np.expm1(x)  # per meter
    return B_m * 1e-6  # per um


def planck_bnu_W_m2_hz_sr(nu_hz: np.ndarray, T_K: float) -> np.ndarray:
    """
    Planck function B_nu in units:
      W m^-2 sr^-1 Hz^-1
    """
    nu_hz = np.asarray(nu_hz, dtype=float)
    x = (H_J_S * nu_hz) / (K_B_J_K * float(T_K))
    return (2.0 * H_J_S * nu_hz**3 / C_M_S**2) / np.expm1(x)

def _read_newera_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    CSV columns: wl_A, flux
      wl_A  : Angstrom
      flux  : erg / s / cm^2 / cm  (per cm of wavelength)
    """
    arr = np.genfromtxt(path, delimiter=",", names=True, dtype=float)

    if arr.dtype.names is None:
        raise ValueError(f"Could not parse CSV header in {path.name}")

    need = {"wl_A", "flux"}
    if not need.issubset(set(arr.dtype.names)):
        raise ValueError(f"Expected columns wl_A, flux in {path.name}, got {arr.dtype.names}")

    wl_A = np.asarray(arr["wl_A"], float)
    flux = np.asarray(arr["flux"], float)

    m = np.isfinite(wl_A) & np.isfinite(flux) & (wl_A > 0)
    wl_A = wl_A[m]
    flux = flux[m]

    if wl_A.size < 10:
        raise ValueError(f"Too few valid samples in {path.name}")

    return wl_A, flux

def _read_newera_csv_cached(path_str: str) -> tuple[np.ndarray, np.ndarray]:
    from pathlib import Path
    return _read_newera_csv(Path(path_str))

if st is not None:
    _read_newera_csv_cached = st.cache_data(show_spinner=False)(_read_newera_csv_cached)

def stellar_flambda_um_arr(
    lam_um_arr: np.ndarray,
    band: str,
    mag_system: str,
    m_mag: float,
    source_sed: str,
    T_star_K: float,
    phoenix_newera_dir: str | None = None,
    v_rv_km_s: float = 0.0,
) -> tuple[np.ndarray, dict]:
    """
    Build stellar F_lambda(λ) on the ETC wavelength grid.

    Returns:
      F_lambda_um_arr : [W m^-2 um^-1] sampled on lam_um_arr
      sed_meta        : dict with keys:
        - sed_requested
        - sed_used
        - sed_fallback_reason (optional)
        - phoenix_teff_selected_k (optional; filled later when grids implemented)
    """
    lam_um_arr = np.asarray(lam_um_arr, dtype=float)

    # magnitude scale (TOA Fnu)
    fnu = mag_to_fnu_w_m2_hz(float(m_mag), band, mag_system)

    lam_m_arr = lam_um_arr * UM_TO_M
    lam_ref_um = float(np.nanmedian(lam_um_arr))
    lam_ref_m = lam_ref_um * UM_TO_M

    # Convert Fnu at reference wavelength to Flambda at that wavelength:
    # F_lambda = (c / lambda^2) * F_nu   then per um => * 1e-6
    F_lambda_ref_um = (C_M_S / lam_ref_m**2) * fnu * 1e-6  # W m^-2 um^-1

    sed_requested = str(source_sed).strip().lower()
    sed = sed_requested

    # Treat legacy "phoenix" alias as NewEra
    if sed == "phoenix":
        sed = "phoenix-newera"

    sed_meta: dict = {
        "sed_requested": sed_requested,
        "sed_used": sed,
    }

    # Enforce NewEra Teff range with fallback
    if sed == "phoenix-newera":
        t = float(T_star_K)
        if not (PHOENIX_NEWERA_TEFF_MIN <= t <= PHOENIX_NEWERA_TEFF_MAX):
            sed_meta["sed_fallback_reason"] = (
                f"Teff out of range for PHOENIX-NewEra "
                f"({PHOENIX_NEWERA_TEFF_MIN:.0f}–{PHOENIX_NEWERA_TEFF_MAX:.0f} K)"
            )
            sed = "blackbody"
            sed_meta["sed_used"] = sed

    # Blackbody
    if sed == "blackbody":
        if float(v_rv_km_s) != 0.0:
            beta = float(v_rv_km_s) / (C_M_S * 1e-3)
            D = float(np.sqrt((1.0 + beta) / (1.0 - beta)))
            lam_eval_um = lam_um_arr / D
            lam_ref_eval_um = lam_ref_um / D
        else:
            lam_eval_um = lam_um_arr
            lam_ref_eval_um = lam_ref_um
        B_ref = float(planck_blambda_W_m2_um_sr(np.array([lam_ref_eval_um]), float(T_star_K))[0])
        B_arr = planck_blambda_W_m2_um_sr(lam_eval_um, float(T_star_K))
        F_lambda_um_arr = F_lambda_ref_um * (B_arr / B_ref)
        return F_lambda_um_arr, sed_meta

    # PHOENIX-NewEra (local OR cache+download)
    if sed == "phoenix-newera":
        teff_list_json = Path(__file__).resolve().parents[1] / "assets" / "newera_teff_list.json"

        # Decide where to look locally:
        # - If caller passes phoenix_newera_dir, use that 
        # - Else use project constant PHOENIX_NEWERA_GRID_DIR
        local_dir = PHOENIX_NEWERA_GRID_DIR if not phoenix_newera_dir else phoenix_newera_dir

        remote_base = PHOENIX_NEWERA_REMOTE_BASE_URL
        if st is not None:
            remote_base = st.secrets.get("newera", {}).get("remote_base_url", remote_base)

        fpath, teff_sel, dl_url = ensure_newera_grid_file(
            float(T_star_K),
            local_grid_dir=local_dir,
            cache_dir=PHOENIX_NEWERA_CACHE_DIR,
            remote_base_url=remote_base,
            teff_list_json=teff_list_json,
        )

        wl_A, flux_cgs_per_cm = _read_newera_csv_cached(str(fpath))
        wl_um_grid = wl_A * 1e-4  # Angstrom -> um

        # Unit conversion: erg/s/cm^2/cm  ->  W/m^2/um
        F_lambda_um_grid = flux_cgs_per_cm * 1e-7

        # sort for interpolation safety
        idx = np.argsort(wl_um_grid)
        wl_um_grid = wl_um_grid[idx]
        F_lambda_um_grid = F_lambda_um_grid[idx]

        # Relativistic Doppler shift: λ_obs = λ_rest × sqrt((1 + β)/(1 - β))
        # Shifts stellar absorption lines relative to the telluric grid.
        # Only affects PHOENIX-NewEra (blackbody has no spectral features).
        if float(v_rv_km_s) != 0.0:
            beta = float(v_rv_km_s) / (C_M_S * 1e-3)
            wl_um_grid = wl_um_grid * np.sqrt((1.0 + beta) / (1.0 - beta))

        # interpolate onto ETC grid
        F_lambda_um_arr = np.interp(
            lam_um_arr,
            wl_um_grid,
            F_lambda_um_grid,
            left=np.nan,
            right=np.nan,
        )

        # fill outside-range with nearest edge value
        if np.any(~np.isfinite(F_lambda_um_arr)):
            good = np.isfinite(F_lambda_um_arr)
            if not np.any(good):
                # grid does not cover the band -> fallback to blackbody
                sed_meta["sed_fallback_reason"] = "NewEra grid does not cover band; falling back to Blackbody"
                B_ref = float(planck_blambda_W_m2_um_sr(np.array([lam_ref_um]), float(T_star_K))[0])
                B_arr = planck_blambda_W_m2_um_sr(lam_um_arr, float(T_star_K))
                F_lambda_um_arr = F_lambda_ref_um * (B_arr / B_ref)
                sed_meta["sed_used"] = "blackbody"
                return F_lambda_um_arr, sed_meta

            first = int(np.argmax(good))
            last = int(len(good) - 1 - np.argmax(good[::-1]))
            F_lambda_um_arr[:first] = F_lambda_um_arr[first]
            F_lambda_um_arr[last + 1:] = F_lambda_um_arr[last]

        # Normalize to match magnitude-derived Flambda at lam_ref
        F_ref_grid = float(np.interp(lam_ref_um, wl_um_grid, F_lambda_um_grid))
        if (not np.isfinite(F_ref_grid)) or (F_ref_grid <= 0):
            sed_meta["sed_fallback_reason"] = f"Invalid NewEra reference flux in {Path(fpath).name}; falling back to Blackbody"
            B_ref = float(planck_blambda_W_m2_um_sr(np.array([lam_ref_um]), float(T_star_K))[0])
            B_arr = planck_blambda_W_m2_um_sr(lam_um_arr, float(T_star_K))
            F_lambda_um_arr = F_lambda_ref_um * (B_arr / B_ref)
            sed_meta["sed_used"] = "blackbody"
            return F_lambda_um_arr, sed_meta

        scale = float(F_lambda_ref_um / F_ref_grid)
        F_lambda_um_arr = F_lambda_um_arr * scale

        sed_meta["phoenix_file"] = str(Path(fpath).name)
        sed_meta["phoenix_teff_selected_k"] = int(teff_sel)
        if dl_url is not None:
            sed_meta["phoenix_download_url"] = str(dl_url)

        return F_lambda_um_arr, sed_meta


    # safety fallback (should not reach here)
    F_lambda_um_arr = (C_M_S / lam_m_arr**2) * fnu * 1e-6
    return F_lambda_um_arr, sed_meta

