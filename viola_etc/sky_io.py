"""
viola_etc/sky_io.py

Sky model pathing + SkyCalc FITS I/O.

Expected sky model layout:
  ./sky_models/<sky_model_name>/skytable.fits
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from .models import InstrumentConfig, SiteConfig


def get_sky_fits_path(site: SiteConfig, sky_model_name: str) -> str:
    """
    Build expected path:
      <site.sky_model_base_dir>/<sky_model_name>/skytable.fits
    """
    name = str(sky_model_name).strip()
    if len(name) == 0:
        raise ValueError("sky_model_name must be a non-empty string")
    return str(Path(site.sky_model_base_dir) / name / "skytable.fits")


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


def check_sky_grid_vs_resolution(lam_um_arr: np.ndarray, resolving_power: float) -> Dict[str, float]:
    """
    Compare SkyCalc wavelength grid spacing vs instrument resolution element size.

    Returns dict with:
      dlam_grid_med_nm, dlam_res_med_nm, ratio_grid_to_res
    """
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

