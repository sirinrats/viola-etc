"""
viola_etc/molecules.py

Molecule template loading + simple detection estimator.

Two filename conventions supported:

  Legacy v0.4 (HD21520b grid):
    {target}_{MOL}_{BAND}_band_R{R}.csv         columns: wl_micron, transit_cm

  v0.5 grid (PSG + petitRADTRANS, peak-anchor counting):
    {class}_transmission_{MOL}_{BAND}_band.csv  columns: wl_micron, value

Pick the v0.5 grid by passing ``class_name='temperate_terrestrial'`` (etc.) and
``molecule_dir='molecular_templates'``.

Line counting recipe (v0.5):
  1.  Load raw transit-depth (ppm); interpolate to ETC wavelength grid.
  2.  Determine the per-template in-band dynamic range.
  3.  Set the absolute prominence threshold = ``prom_abs × in_band_range`` (ppm).
  4.  ``find_peaks(prominence=threshold_ppm, distance=nx)`` — scipy computes
      each peak's prominence locally; only peaks with local prominence above
      the absolute threshold are kept.
  5.  Filter by per-pixel SNR threshold (unchanged from v0.4).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


# =========================
# Helpers
# =========================

def _normalize_minmax_safe(y: np.ndarray) -> np.ndarray:
    """min-max → [0,1].  Used only for the plotting helper, not for counting."""
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
    """Find peaks in ``y`` with an *absolute* prominence threshold.  scipy
    computes each peak's prominence locally; the threshold is the minimum
    allowed local prominence."""
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

        # crude prominence filter (local, no global saddle search)
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

        keep = np.array(sorted(keep), dtype=int)
        out = [int(keep[0])]
        for i in keep[1:]:
            if i - out[-1] >= distance:
                out.append(int(i))
        return np.asarray(out, dtype=int)


# =========================
# Public API
# =========================

def load_molecule_templates(
    lam_um_arr: np.ndarray,
    band: str,
    resolving_power: float,
    molecule_dir: str = "./molecular_templates",
    molecules: Tuple[str, ...] = ("CH4", "CO", "CO2", "H2O", "NH3", "O2", "OH"),
    class_name: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """Load one transmission template per molecule for the chosen band.

    If ``class_name`` is provided, the v0.5 filename pattern is used:
        ``{class_name}_transmission_{MOL}_{BAND}_band.csv``
    Otherwise the legacy patterns are tried.

    For each molecule, the returned dict carries the raw transit depth in ppm
    on the ETC wavelength grid (``template_ppm``) and a min-max-normalised
    copy for plotting (``template_resampled``).
    """
    out: Dict[str, Dict[str, Any]] = {}

    base = Path(molecule_dir)
    if not base.exists():
        return out

    band_u = band.strip().upper()
    R_int = int(round(float(resolving_power)))

    for mol in molecules:
        mol_u = mol.strip().upper()

        if class_name:
            patterns = [
                f"{class_name}_transmission_{mol_u}_{band_u}_band.csv",
                f"{class_name}_transmission_{mol_u}_{band_u}_band_R{R_int}.csv",
            ]
        else:
            patterns = [
                f"*_{mol_u}_{band_u}_band_R{R_int}.csv",
                f"*_{mol_u}_{band_u}_band.csv",
                f"*_{mol_u}_{band_u}_*.csv",
            ]

        matches = []
        for pat in patterns:
            matches = sorted(base.glob(pat))
            if matches:
                break

        if not matches:
            continue

        p = matches[0]
        try:
            data = np.genfromtxt(str(p), delimiter=",", names=True, dtype=None, encoding=None)
            if "wl_micron" not in data.dtype.names:
                continue
            # Accept the v0.5 "value", or legacy "transit_cm" / "fp_fs_ppm".
            value_col = next((c for c in ("value", "value_ppm", "transit_cm", "fp_fs_ppm")
                              if c in data.dtype.names), None)
            if value_col is None:
                continue
            wl = np.asarray(data["wl_micron"], float)
            y  = np.asarray(data[value_col], float)
        except Exception:
            continue

        # Interpolate raw values onto the ETC wavelength grid (NO normalisation
        # — counting uses raw ppm so an absolute prominence threshold scales
        # correctly with each template's dynamic range).
        y_res_ppm = _interp_to_grid(wl, y, lam_um_arr)

        # Min-max copy retained for plotting backward compatibility.
        y_res_norm = _normalize_minmax_safe(y_res_ppm)

        out[mol_u.lower()] = {
            "path": str(p),
            "wl_um": wl,
            "template_raw": y.copy(),                # source data (ppm, source grid)
            "template_ppm": y_res_ppm,               # raw ppm on ETC grid → used for counting
            "template_resampled": y_res_norm,        # min-max [0,1] on ETC grid → used for plotting
            "n_matches": len(matches),
            "all_matches": [str(x) for x in matches],
        }

    return out


def estimate_transits_for_molecules(
    templates: Dict[str, Dict[str, Any]],
    lam_um_arr: np.ndarray,
    snr_res_arr: np.ndarray,
    nx: int,
    prom_abs: float,
    snr_thresh: float,
    detrend_p: float,
    line_contrast: float,
    detect_sig: float,
    min_lines_for_calc: int,
) -> Dict[str, Dict[str, Any]]:
    """Count lines using a per-template absolute prominence threshold scaled
    to the in-band dynamic range, then estimate detection significance:

        threshold_ppm     = prom_abs × (max(template_ppm) − min(template_ppm))
        N_lines           = #{ peaks with local prominence ≥ threshold_ppm }
        snr_per_night     = line_contrast × detrend_p × sqrt(Σ SNR_res,j²)
        n_nights_required = (detect_sig / snr_per_night)²

    The sum runs over the N_lines selected peak positions (j ∈ L).
    ``prom_abs`` is interpreted as a fraction (0.05 = 5 % of the in-band
    dynamic range).  ``snr_res_arr`` is the night-accumulated stellar SNR and
    is filtered against ``snr_thresh`` at the detected peak positions.
    """
    out: Dict[str, Dict[str, Any]] = {}

    snr = np.asarray(snr_res_arr, float)

    dist = int(max(1, int(nx)))

    for mol, rec in templates.items():
        # Prefer the raw-ppm template; fall back to the [0,1] one for legacy templates
        y = np.asarray(rec.get("template_ppm",
                       rec.get("template_resampled", None)), float)
        if y.size == 0:
            continue

        y_in = y

        # absolute prominence threshold = user fraction × in-band dynamic range
        finite = y_in[np.isfinite(y_in)]
        if finite.size:
            inband_range = float(finite.max() - finite.min())
        else:
            inband_range = 0.0
        threshold_abs = max(float(prom_abs) * inband_range, 0.0)

        pks = _try_find_peaks(y_in, prominence=threshold_abs, distance=dist)

        # filter by local SNR (unchanged)
        if pks.size:
            s_at = snr[pks]
            keep = np.isfinite(s_at) & (s_at >= float(snr_thresh))
            pks_keep = pks[keep]
        else:
            pks_keep = np.array([], dtype=int)

        n_lines = int(pks_keep.size)

        if n_lines >= int(min_lines_for_calc):
            snr_at_lines = snr[pks_keep]   # SNR at the N_lines selected peaks
            snr_quadrature = float(np.sqrt(np.sum(snr_at_lines ** 2)))
            spt = float(line_contrast) * float(detrend_p) * snr_quadrature
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
            "threshold_abs_ppm": threshold_abs,   # for diagnostic / display
            "inband_range_ppm": inband_range,
        }

    return out
