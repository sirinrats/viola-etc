"""
viola_etc/photometry.py

Photometry + Planck helper functions.

Notes:
- mag -> Fnu uses constants.get_fnu0_w_m2_hz 
- Planck functions are pure physics helpers
"""

from __future__ import annotations

import numpy as np

from ..constants import C_M_S, H_J_S, K_B_J_K, UM_TO_M, get_fnu0_w_m2_hz

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

