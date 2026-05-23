"""
viola_etc/models.py

Dataclasses for VIOLA ETC.

This module defines:
- user inputs: TargetParams, ObservingCondition, UserInputs
- hidden defaults: InstrumentConfig, SiteConfig
- outputs container: ETCResult

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from .observing.sky_io import SkyData

import numpy as np


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

    # SED selector:
    # - "blackbody"      : Planck shape (anchored to band magnitude)
    # - "phoenix-newera" : PHOENIX-NewEra grid (Teff 2300–12000 K); falls back to blackbody if out of range
    # - "phoenix"        : legacy alias for "phoenix-newera"
    source_sed: str = "blackbody"

    # stellar effective temperature — used to shape the blackbody SED or select the PHOENIX-NewEra grid file
    T_star_K: float = 5800.0

    # stellar radial velocity — shifts PHOENIX-NewEra lines relative to tellurics;
    # also adjusts the blackbody continuum shape (rest-frame evaluation); positive = redshift
    v_rv_km_s: float = 0.0

    # detection knob
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
        "vipa": 0.90,
        "x_disp": 0.80,
        "camera": 0.90,
        "detector": 0.95,
    })

    # thermal & detector
    t_ambient_k: float = 275.0            # instrument ambient temperature [K]
    epsilon_eff: float = 0.10             # effective emissivity for thermal background calculation
    rn_e: float = 20.0                    # read noise [e- per pixel per read]
    dark_rate: float = 0.01               # dark current [e- per second per pixel]


@dataclass(frozen=True)
class SiteConfig:
    """
    Site / sky model defaults (hidden).
    """
    sky_model_base_dir: str = "./sky_models"


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
    target_alt_deg: float = 60.0
    pwv_mm: float = 1.0

    # sky model folder under ./sky_models
    sky_model_name: str = "SkyCalc_Grid"


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
# Result container
# =========================

@dataclass
class ETCResult:
    """
    Output container returned by run_etc().

    NOTE: molecule_* fields are dicts (not None) by default to avoid AttributeError in UI.
    """
    meta: Dict[str, Any] = field(default_factory=dict)
    summary_lines: List[str] = field(default_factory=list)

    lam_um: Optional[np.ndarray] = None
    trans: Optional[np.ndarray] = None
    sky: Optional["SkyData"] = None   # all sky arrays; see sky_io.SkyData

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
    molecule_templates: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    molecule_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)

