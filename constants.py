"""
constants.py

Project-wide physical constants and unit conversion helpers.

"""

from __future__ import annotations

from typing import Dict
import numpy as np


# =========================
# Physical constants (SI)
# =========================
C_M_S: float = 2.99792458e8        # speed of light (m/s)
H_J_S: float = 6.62607015e-34      # Planck constant (J s)
K_B_J_K: float = 1.380649e-23      # Boltzmann constant (J/K)


# =========================
# Unit helpers
# =========================
UM_TO_M: float = 1e-6              # micron -> meter

# Convert arcsec^2 to steradian:
# 1 arcsec = (pi / 180 / 3600) rad, so (arcsec)^2 -> (rad)^2 = steradian
ARCSEC2_TO_SR: float = (np.pi / (180.0 * 3600.0)) ** 2


# =========================
# Photometric zero-points (2MASS) in F_nu
# Units: W m^-2 Hz^-1
# https://irsa.ipac.caltech.edu/data/SPITZER/docs/dataanalysistools/tools/pet/magtojy/ref.html?
#
# These values correspond to 0-mag flux densities in each band.
# 1 Jy = 1e-26 W m^-2 Hz^-1
# =========================
FNU0_2MASS_W_M2_HZ: Dict[str, float] = {
    "J": 1.594e-23,
    "H": 1.024e-23,
    "K": 0.667e-23,
}

# =========================
# Photometric zero-point (AB)
# AB system is defined such that 0 mag = 3631 Jy at all frequencies.
# 1 Jy = 1e-26 W m^-2 Hz^-1
# =========================
FNU0_AB_W_M2_HZ: float = 3631.0e-26


# =========================
# snr vs magnitude sweep (defaults)
# =========================
MAG_SWEEP_MIN = 5.0
MAG_SWEEP_MAX = 16.0
MAG_SWEEP_STEP = 0.5

# =========================
# molecular detection (defaults)
# =========================
MOLECULE_LINES_DIR = "./molecular_lines"   # your note: /molecular_lines/*.csv
MOLECULE_TEMPLATE_COL_WL = "wl_micron"
MOLECULE_TEMPLATE_COL_Y  = "transit_cm"

# default list shown in app
DEFAULT_MOLECULES = ["ch4", "h2o", "co2", "co"]



def get_fnu0_w_m2_hz(band: str, mag_system: str) -> float:
    """
    Return F_nu zero-point (W m^-2 Hz^-1) for the chosen magnitude system.

    Parameters
    ----------
    band : str
        'J', 'H', 'K' (used only for Vega/2MASS here).
    mag_system : str
        'Vega' or 'AB' (case-insensitive).

    Returns
    -------
    fnu0 : float
        Zero-point flux density in W m^-2 Hz^-1.
    """
    sys = mag_system.strip().upper()
    b = band.strip().upper()

    if sys == "AB":
        return FNU0_AB_W_M2_HZ

    if sys == "VEGA":
        if b not in FNU0_2MASS_W_M2_HZ:
            raise ValueError(f"band must be one of {list(FNU0_2MASS_W_M2_HZ.keys())}, got {band!r}")
        return FNU0_2MASS_W_M2_HZ[b]

    raise ValueError(f"mag_system must be 'Vega' or 'AB', got {mag_system!r}")
