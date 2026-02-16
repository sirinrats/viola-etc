"""
viola_etc/target/newera_io.py

PHOENIX-NewEra grid discovery + on-demand download.

Goal:
- Work locally when grids exist on disk
- Work on Streamlit Cloud by downloading missing files into a cache directory
"""

from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path
from typing import Optional


_TEFF_RE = re.compile(r"lte(\d{5})", re.IGNORECASE)


def parse_teff_from_filename(name: str) -> Optional[int]:
    m = _TEFF_RE.search(name)
    if not m:
        return None
    return int(m.group(1))


def load_newera_teff_list(teff_list_json: str | Path) -> list[int]:
    """
    Load available Teff grid values from a small JSON committed to the repo.
    Format: {"teff_list": [2300, 2400, ...]}
    """
    p = Path(teff_list_json)
    if not p.exists():
        raise FileNotFoundError(f"Teff list JSON not found: {p}")

    obj = json.loads(p.read_text())
    teffs = [int(x) for x in obj.get("teff_list", [])]
    teffs = sorted(set(teffs))
    if not teffs:
        raise ValueError(f"No teff_list found in {p}")
    return teffs


def nearest_teff(teff_k: float, teff_list: list[int]) -> int:
    t = float(teff_k)
    return min(teff_list, key=lambda x: abs(float(x) - t))


def newera_filename_for_teff(teff_int: int) -> str:
    """
    Map Teff -> expected filename.

    Adjust this function if your naming scheme changes.
    """
    return f"lte{teff_int:05d}-4.50-0.0.PHOENIX-NewEra-ACES-COND-2023.HSR.csv"


def download_file(url: str, dst: Path) -> None:
    """
    Download url -> dst (atomic-ish via .part).
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    try:
        urllib.request.urlretrieve(url, tmp)  # simple and OK for now
        tmp.replace(dst)
    finally:
        # cleanup on failure
        if tmp.exists() and not dst.exists():
            try:
                tmp.unlink()
            except Exception:
                pass


def ensure_newera_grid_file(
    teff_k: float,
    *,
    local_grid_dir: Optional[str | Path],
    cache_dir: str | Path,
    remote_base_url: str,
    teff_list_json: str | Path,
) -> tuple[Path, int, str | None]:
    """
    Ensure the required NewEra file exists locally.

    Returns:
      (path, teff_selected_k, download_url_or_none)

    Behavior:
    1) If local_grid_dir exists and contains file => use it
    2) Else use cache_dir:
       - if file exists => use it
       - else download from remote_base_url => cache_dir => use it

    Raises:
      FileNotFoundError if file not found and remote_base_url not set.
    """
    teffs = load_newera_teff_list(teff_list_json)
    tsel = nearest_teff(teff_k, teffs)
    fname = newera_filename_for_teff(tsel)

    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    # 1) Prefer local directory if present
    if local_grid_dir is not None:
        d = Path(local_grid_dir)
        if d.exists():
            f = d / fname
            if f.exists():
                return f, tsel, None

    # 2) Cache directory
    f_cache = cache / fname
    if f_cache.exists():
        return f_cache, tsel, None

    base = str(remote_base_url).strip()

    # Treat file:// (and local paths) as "not remote" — remote must be http(s)
    if base.startswith("file://") or (base and "://" not in base):
        base = ""

    if not base:
        raise FileNotFoundError(
            "PHOENIX-NewEra grid file not found locally or in cache, and remote_base_url is empty.\n"
            f"Missing: {fname}\n"
            f"Looked in:\n"
            f"  local_grid_dir: {local_grid_dir}\n"
            f"  cache_dir     : {cache}\n"
            "Set PHOENIX_NEWERA_REMOTE_BASE_URL to an http(s) URL to enable downloading."
        )

    url = base.rstrip("/") + "/" + fname
    download_file(url, f_cache)
    return f_cache, tsel, url

