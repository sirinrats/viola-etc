"""
viola_etc/molecules.py

Molecule template loading + simple detection estimator.

Files expected in molecule_dir:
  {star_name}_{MOL}_{BAND}_band_R{R}.csv

Example:
  HD21520b_CH4_H_band_R300000.csv

Expected columns:
  wl_micron, transit_cm
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


# =========================
# Helpers
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


# =========================
# Public API
# =========================

def load_molecule_templates(
    lam_um_arr: np.ndarray,
    band: str,
    resolving_power: float,
    molecule_dir: str = "./molecular_templates",
    molecules: Tuple[str, ...] = ("CH4", "H2O", "CO2", "CO"),
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}

    base = Path(molecule_dir)
    if not base.exists():
        return out

    band_u = band.strip().upper()
    R_int = int(round(float(resolving_power)))

    for mol in molecules:
        mol_u = mol.strip().upper()

        # Your chosen convention: *_{MOL}_{BAND}_*.csv
        # Also include R in the pattern if you want tighter matching.
        patterns = [
            f"*_{mol_u}_{band_u}_band_R{R_int}.csv",
            f"*_{mol_u}_{band_u}_*.csv",  # fallback if R not in filename
        ]

        matches = []
        for pat in patterns:
            matches = sorted(base.glob(pat))
            if matches:
                break

        if not matches:
            continue

        # If multiple matches, pick deterministically:
        # 1) prefer the first after sort (stable), OR
        # 2) choose newest: max(matches, key=lambda p: p.stat().st_mtime)
        p = matches[0]

        try:
            data = np.genfromtxt(str(p), delimiter=",", names=True, dtype=None, encoding=None)
            if ("wl_micron" not in data.dtype.names) or ("transit_cm" not in data.dtype.names):
                continue

            wl = np.asarray(data["wl_micron"], float)
            y  = np.asarray(data["transit_cm"], float)
        except Exception:
            continue

        y0 = _normalize_minmax_safe(y)
        y_res = _interp_to_grid(wl, y0, lam_um_arr)
        y_res = _normalize_minmax_safe(y_res)

        out[mol_u.lower()] = {
            "path": str(p),
            "wl_um": wl,
            "template_raw": y0,
            "template_resampled": y_res,
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
    n_frames_per_transit: int,
    detrend_p: float,
    line_contrast: float,
    detect_sig: float,
    min_lines_for_calc: int,
    find_valleys: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    Count lines using peak finding on normalized template.
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
            spt = float(line_contrast) * snr_med * math.sqrt(
                float(n_lines) * float(n_frames_per_transit) * float(detrend_p)
            )
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

