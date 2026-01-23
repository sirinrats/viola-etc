"""
viola_etc/validate.py

Validation helpers for VIOLA ETC inputs and configs.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from ..models import InstrumentConfig, SiteConfig, UserInputs


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

    if not np.isfinite(u.target.T_star_K):
        raise ValueError("T_star_K must be finite")
    if not (2000.0 <= float(u.target.T_star_K) <= 40000.0):
        raise ValueError("T_star_K must be in [2000, 40000]")

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

    # Grid inputs (required)
    if not np.isfinite(u.obs.target_alt_deg):
        raise ValueError("target_alt_deg must be finite")
    if not (0.0 < float(u.obs.target_alt_deg) <= 90.0):
        raise ValueError("target_alt_deg must be in (0, 90] degrees")

    if not np.isfinite(u.obs.pwv_mm):
        raise ValueError("pwv_mm must be finite")
    if float(u.obs.pwv_mm) <= 0.0:
        raise ValueError("pwv_mm must be > 0")

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
    