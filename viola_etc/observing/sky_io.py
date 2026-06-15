"""
viola_etc/observing/sky_io.py

Sky model path resolution + SkyCalc FITS I/O for the 45-file JHK grid.

Grid layout:
  ./sky_models/skytable_alt{alt}_pwv{pwv}_JHK_R300k.fits
  5 altitudes (90/75/60/45/30 deg) × 9 PWV values (0.5–20.0 mm) = 45 files.

Each FITS binary table contains 14 columns (vacuum wavelengths, 900–2750 nm,
R = 300 000, observatory = Paranal):
  lam       vacuum wavelength [nm]
  trans     total atmospheric transmission [0–1]
  flux      total sky emission [ph s^-1 m^-2 um^-1 arcsec^-2]
  flux_zl   zodiacal light
  flux_ael  airglow emission lines (upper atmosphere)
  flux_tme  molecular emission (lower atmosphere)
  flux_sml  scattered moonlight
  flux_ssl  scattered starlight
  flux_arc  airglow / residual continuum
  flux_tie  telescope/instrument thermal emission
  trans_ma  molecular absorption transmission
  trans_o3  ozone absorption transmission
  trans_rs  Rayleigh scattering transmission
  trans_ms  Mie scattering transmission
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, NamedTuple, Tuple
import re

import numpy as np

from ..models import InstrumentConfig, SiteConfig


# ---------------------------------------------------------------------------
# Sky data container
# ---------------------------------------------------------------------------

class SkyData(NamedTuple):
    """
    All sky model arrays trimmed to a single band window.
    All flux columns in ph s^-1 m^-2 um^-1 arcsec^-2.

    sky_phi is the explicit sum of the 6 sky emission components
    (flux_zl + flux_ael + flux_tme + flux_sml + flux_ssl + flux_arc).
    flux_tie (SkyCalc telescope/instrument thermal) is excluded because
    the ETC computes its own instrument thermal via a Planck greybody
    (runner.py). 
    flux_tie is stored for diagnostics but not added to sky_phi.
    """
    lam_um   : np.ndarray   # vacuum wavelength [um]
    trans    : np.ndarray   # total atmospheric transmission [0–1]
    sky_phi  : np.ndarray   # sky emission = sum of 6 components (flux_tie excluded)
    flux_zl  : np.ndarray   # zodiacal light
    flux_ael : np.ndarray   # airglow emission lines (upper atmosphere)
    flux_tme : np.ndarray   # molecular emission (lower atmosphere)
    flux_sml : np.ndarray   # scattered moonlight
    flux_ssl : np.ndarray   # scattered starlight
    flux_arc : np.ndarray   # airglow / residual continuum
    flux_tie : np.ndarray   # telescope/instrument thermal
    trans_ma : np.ndarray   # molecular absorption transmission
    trans_o3 : np.ndarray   # ozone absorption transmission
    trans_rs : np.ndarray   # Rayleigh scattering transmission
    trans_ms : np.ndarray   # Mie scattering transmission


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_alt_pwv_from_filename(p: Path) -> Tuple[float, float] | None:
    """
    Parse (alt_deg, pwv_mm) from filenames like:
      skytable_alt30_pwv0p5_JHK_R300k.fits
    Returns None if the pattern is not matched.
    """
    name = p.stem.lower()
    m_alt = re.search(r"alt(?P<alt>\d+(?:p\d+)?)", name)
    m_pwv = re.search(r"pwv(?P<pwv>\d+(?:p\d+)?)", name)
    if m_alt is None or m_pwv is None:
        return None
    try:
        return float(m_alt.group("alt").replace("p", ".")), \
               float(m_pwv.group("pwv").replace("p", "."))
    except ValueError:
        return None


def _pick_closest_grid_fits(grid_dir: Path, alt_deg: float, pwv_mm: float) -> str:
    """
    Return path of the grid file whose (alt, pwv) is closest to the
    requested values under an L1 metric.
    """
    files = sorted(grid_dir.glob("*.fits"))
    if not files:
        raise FileNotFoundError(f"No .fits files found in: {grid_dir}")

    candidates = [
        (f, a, w)
        for f in files
        for a, w in [_parse_alt_pwv_from_filename(f) or (None, None)]
        if a is not None
    ]
    if not candidates:
        raise ValueError(
            f"Could not parse alt/pwv from any FITS filename in {grid_dir}. "
            "Expected pattern: skytable_alt30_pwv0p5_*.fits"
        )

    # L1 sum of (degrees) + (mm): dimensionally mixed, but correct for this grid.
    # The 5x9 altitude-PWV grid is a complete rectangular product, and altitude is
    # coarsely sampled (15 deg steps) relative to PWV, so the L1 nearest neighbour
    # reduces to the per-axis nearest node. Assumes a rectangular grid; revisit if
    # the grid axes or spacing change.
    best = min(candidates, key=lambda r: abs(r[1] - alt_deg) + abs(r[2] - pwv_mm))
    return str(best[0])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_sky_fits_path(
    site: SiteConfig,
    sky_model_name: str,
    *,
    target_alt_deg: float | None = None,
    pwv_mm: float | None = None,
) -> str:
    """
    Resolve the path to a SkyCalc FITS file.

    For sky_model_name == "SkyCalc_Grid":
      Picks the file in <site.sky_model_base_dir>/ whose
      (alt, pwv) is closest (L1) to (target_alt_deg, pwv_mm).

    For any other name (legacy single-file layout):
      Returns <site.sky_model_base_dir>/<sky_model_name>/skytable.fits
    """
    name = str(sky_model_name).strip()
    if not name:
        raise ValueError("sky_model_name must be a non-empty string")

    base = Path(site.sky_model_base_dir)

    if name == "SkyCalc_Grid":
        if target_alt_deg is None or pwv_mm is None:
            raise ValueError("SkyCalc_Grid requires target_alt_deg and pwv_mm")
        return _pick_closest_grid_fits(
            base, float(target_alt_deg), float(pwv_mm)
        )

    return str(base / name / "skytable.fits")


def load_skytable(
    sky_fits_path: str,
    band: str,
    cfg: InstrumentConfig,
) -> SkyData:
    """
    Read a SkyCalc FITS file and return all sky arrays trimmed to the
    requested band window.

    Parameters
    ----------
    sky_fits_path : str
        Path to the FITS file (from get_sky_fits_path).
    band : str
        Band key matching cfg.band_edges_um (e.g. "J", "H", "K").
    cfg : InstrumentConfig
        Instrument configuration supplying band_edges_um.

    Returns
    -------
    SkyData NamedTuple — see class docstring for field descriptions.

    Notes
    -----
    sky_phi is computed as:
        flux_zl + flux_ael + flux_tme + flux_sml + flux_ssl + flux_arc
    flux_tie (SkyCalc telescope/instrument thermal) is intentionally
    excluded: the ETC computes instrument thermal independently via a
    Planck greybody in runner.py.  flux_tie is stored in SkyData for
    diagnostic access but is not added to sky_phi.
    """
    try:
        from astropy.io import fits
    except ImportError as e:
        raise ImportError("astropy is required. pip install astropy") from e

    def _col(data, name, default=0.0):
        if name in data.names:
            return np.array(data[name], dtype=float)
        return np.full(len(data), float(default))

    p = Path(sky_fits_path)
    if not p.exists():
        raise FileNotFoundError(f"Sky FITS not found: {sky_fits_path}")

    with fits.open(str(p)) as hdul:
        t = hdul[1].data

        for required in ("lam", "trans"):
            if required not in t.names:
                raise ValueError(
                    f"Sky FITS {p.name} is missing required column {required!r}; "
                    f"available columns: {list(t.names)}"
                )

        lam_um  = _col(t, "lam") * 1e-3      # nm -> um
        trans   = _col(t, "trans")

        flux_zl  = _col(t, "flux_zl")
        flux_ael = _col(t, "flux_ael")
        flux_tme = _col(t, "flux_tme")
        flux_sml = _col(t, "flux_sml")
        flux_ssl = _col(t, "flux_ssl")
        flux_arc = _col(t, "flux_arc")
        flux_tie = _col(t, "flux_tie")        # zero in J/H/K for current grid

        trans_ma = _col(t, "trans_ma", default=1.0)
        trans_o3 = _col(t, "trans_o3", default=1.0)
        trans_rs = _col(t, "trans_rs", default=1.0)
        trans_ms = _col(t, "trans_ms", default=1.0)

    # flux_tie excluded: ETC computes instrument thermal separately via Planck greybody
    sky_phi = flux_zl + flux_ael + flux_tme + flux_sml + flux_ssl + flux_arc

    b = band.strip().upper()
    if b not in cfg.band_edges_um:
        raise ValueError(
            f"band must be one of {list(cfg.band_edges_um.keys())}, got {band!r}"
        )
    lam_min, lam_max = cfg.band_edges_um[b]
    m = (lam_um >= lam_min) & (lam_um <= lam_max)

    if not m.any():
        raise ValueError(
            f"No sky samples in {b} window {cfg.band_edges_um[b]} um "
            f"for file {sky_fits_path}"
        )

    return SkyData(
        lam_um   = lam_um[m],
        trans    = trans[m],
        sky_phi  = sky_phi[m],
        flux_zl  = flux_zl[m],
        flux_ael = flux_ael[m],
        flux_tme = flux_tme[m],
        flux_sml = flux_sml[m],
        flux_ssl = flux_ssl[m],
        flux_arc = flux_arc[m],
        flux_tie = flux_tie[m],
        trans_ma = trans_ma[m],
        trans_o3 = trans_o3[m],
        trans_rs = trans_rs[m],
        trans_ms = trans_ms[m],
    )


def check_sky_grid_vs_resolution(
    lam_um_arr: np.ndarray,
    resolving_power: float,
) -> Dict[str, float]:
    """
    Compare SkyCalc wavelength grid spacing against the instrument
    resolution element size.

    Returns dict with keys:
      dlam_grid_med_nm   median grid spacing [nm]
      dlam_res_med_nm    median resolution element width [nm]
      ratio_grid_to_res  dlam_grid / dlam_res  (< 1 means grid is finer than res element)
    """
    if lam_um_arr.size < 2:
        return {
            "dlam_grid_med_nm": np.nan,
            "dlam_res_med_nm": np.nan,
            "ratio_grid_to_res": np.nan,
        }

    dlam_grid = float(np.median(np.diff(lam_um_arr)))
    dlam_res  = float(np.median(lam_um_arr / resolving_power))
    ratio     = dlam_grid / dlam_res if dlam_res > 0 else np.nan

    return {
        "dlam_grid_med_nm":  dlam_grid * 1e3,
        "dlam_res_med_nm":   dlam_res  * 1e3,
        "ratio_grid_to_res": ratio,
    }
