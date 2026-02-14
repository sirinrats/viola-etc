"""
viola_etc/constants.py

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
ARCSEC2_TO_SR: float = (np.pi / (180.0 * 3600.0)) ** 2

# =========================
# Photometric zero-points (2MASS) in F_nu
# Units: W m^-2 Hz^-1
# =========================
FNU0_2MASS_W_M2_HZ: Dict[str, float] = {
    "J": 1.594e-23,
    "H": 1.024e-23,
    "K": 0.667e-23,
}

# =========================
# Photometric zero-point (AB)
# =========================
FNU0_AB_W_M2_HZ: float = 3631.0e-26

# =========================
# snr vs magnitude sweep (defaults)
# =========================
MAG_SWEEP_MIN = 5.0
MAG_SWEEP_MAX = 16.0
MAG_SWEEP_STEP = 0.5

# =========================
# PHOENIX-NewEra grid valid range (Teff)
# =========================
PHOENIX_NEWERA_TEFF_MIN = 2300.0
PHOENIX_NEWERA_TEFF_MAX = 12000.0

# =========================
# PHOENIX-NewEra storage
# =========================
PHOENIX_NEWERA_CACHE_DIR = "./.cache/NewEra_grids_R300K"
# For Streamlit Cloud later (S3/R2/Drive/HF). Keep empty for local dev.
PHOENIX_NEWERA_REMOTE_BASE_URL = ""
PHOENIX_NEWERA_GRID_DIR = "./NewEra_grids_R300K"


def get_fnu0_w_m2_hz(band: str, mag_system: str) -> float:
    sys = mag_system.strip().upper()
    b = band.strip().upper()

    if sys == "AB":
        return FNU0_AB_W_M2_HZ

    if sys == "VEGA":
        if b not in FNU0_2MASS_W_M2_HZ:
            raise ValueError(f"band must be one of {list(FNU0_2MASS_W_M2_HZ.keys())}, got {band!r}")
        return FNU0_2MASS_W_M2_HZ[b]

    raise ValueError(f"mag_system must be 'Vega' or 'AB', got {mag_system!r}")

