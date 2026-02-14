"""
viola_etc/target/newera_io.py

PHOENIX-NewEra grid discovery + on-demand download.

Goal:
- Work locally when grids exist on disk
- Work on Streamlit Cloud by downloading missing files into a cache directory

Key behaviors:
- Prefer local_grid_dir if file exists there.
- Else use cache_dir:
    - if cached file exists -> use it (touch mtime for LRU)
    - else download -> cache_dir (with a per-file lock to avoid double-download)
- Optional cache eviction (LRU by mtime)
- Shows a Streamlit spinner/status for BOTH:
    - waiting for someone else to download
    - downloading ourselves
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
import urllib.request
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse, unquote


_TEFF_RE = re.compile(r"lte(\d{5})", re.IGNORECASE)


# ============================================================
# Basic helpers
# ============================================================

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


def _scheme(x: str) -> str:
    try:
        return urlparse(str(x)).scheme.lower()
    except Exception:
        return ""


def _join_base(base: str, fname: str) -> str:
    b = str(base).rstrip("/")
    return f"{b}/{fname}"


def _touch(path: Path) -> None:
    """
    Touch mtime so LRU eviction can treat it as recently used.
    """
    try:
        path.touch()
    except Exception:
        pass


# ============================================================
# Streamlit UI helpers (safe when Streamlit not installed)
# ============================================================

def _st_ctx(msg: str):
    """
    Return a context manager showing status/spinner if Streamlit is available.
    No-op otherwise (CLI/tests still work).

    We prefer st.status if available, else st.spinner.
    """
    try:
        import streamlit as st
        if hasattr(st, "status"):
            # Note: st.status returns a context manager
            return st.status(msg, expanded=False)
        return st.spinner(msg)
    except Exception:
        return nullcontext()


# ============================================================
# Download helper
# ============================================================

def download_file(url_or_s3: str, dst: Path) -> None:
    """
    Download remote -> dst (atomic-ish via .part)

    Supports:
      - s3://bucket/prefix/file.csv
      - https://.../file.csv
      - file:///abs/path/file.csv
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")

    try:
        sch = _scheme(url_or_s3)

        if sch == "s3":
            # Lazy import so local runs don't require aws libs
            import boto3
            try:
                import streamlit as st  # only available in Streamlit runtime
                aws_cfg = st.secrets.get("aws", st.secrets)
                session = boto3.Session(
                    aws_access_key_id=aws_cfg.get("access_key_id"),
                    aws_secret_access_key=aws_cfg.get("secret_access_key"),
                    region_name=aws_cfg.get("region", "us-east-1"),
                )
            except Exception:
                # On non-Streamlit runs, rely on env/instance role/default chain
                session = boto3.Session()

            u = urlparse(url_or_s3)
            bucket = u.netloc
            key = u.path.lstrip("/")
            session.client("s3").download_file(bucket, key, str(tmp))

        elif sch in ("http", "https"):
            urllib.request.urlretrieve(url_or_s3, tmp)

        elif sch == "file":
            # file:///... -> local path copy
            p = Path(unquote(urlparse(url_or_s3).path))
            if not p.exists():
                raise FileNotFoundError(f"Local file not found: {p}")
            shutil.copyfile(p, tmp)

        else:
            raise ValueError(f"Unsupported remote scheme for {url_or_s3!r}")

        tmp.replace(dst)

    finally:
        # cleanup on failure
        if tmp.exists() and not dst.exists():
            try:
                tmp.unlink()
            except Exception:
                pass


# ============================================================
# Concurrency + eviction
# ============================================================

@contextmanager
def _file_lock(lock_path: Path, poll_s: float = 0.5, timeout_s: float = 600.0):
    """
    Simple lock using exclusive file create.

    - If lock exists, wait until it disappears (another session is downloading).
    - Works across sessions sharing the same filesystem (Streamlit Cloud typical).

    NOTE: lock_path is a file (e.g., <fname>.lock).
    """
    start = time.time()
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break  # acquired
        except FileExistsError:
            if (time.time() - start) > timeout_s:
                raise TimeoutError(f"Timed out waiting for lock: {lock_path}")
            time.sleep(poll_s)

    try:
        yield
    finally:
        try:
            lock_path.unlink()
        except Exception:
            pass


def enforce_cache_limit(cache_dir: Path, max_bytes: int) -> None:
    """
    Evict oldest files until total cache size <= max_bytes.
    Only considers *.csv files in cache_dir.
    """
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        return

    files: list[Path] = [p for p in cache_dir.glob("*.csv") if p.is_file()]
    if not files:
        return

    def size(p: Path) -> int:
        try:
            return p.stat().st_size
        except Exception:
            return 0

    def mtime(p: Path) -> float:
        try:
            return p.stat().st_mtime
        except Exception:
            return 0.0

    total = sum(size(p) for p in files)
    if total <= max_bytes:
        return

    # Oldest first (LRU eviction if you touch files on access)
    files.sort(key=mtime)

    for p in files:
        if total <= max_bytes:
            break

        # Skip if a lock exists for this file (someone downloading it)
        lock_file = p.with_name(p.name + ".lock")
        if lock_file.exists():
            continue

        try:
            total -= size(p)
            p.unlink()
        except Exception:
            # If delete fails, skip it
            pass


# ============================================================
# Main entry: ensure file exists
# ============================================================

def ensure_newera_grid_file(
    teff_k: float,
    *,
    local_grid_dir: Optional[str | Path],
    cache_dir: str | Path,
    remote_base_url: str,
    teff_list_json: str | Path,
    # knobs
    lock_timeout_s: float = 180.0,
    max_cache_bytes: int = 5_000_000_000,
) -> tuple[Path, int, str | None]:
    """
    Ensure the required NewEra file exists locally.

    Returns:
      (path, teff_selected_k, download_url_or_none)

    Behavior:
    1) If local_grid_dir exists and contains file => use it
    2) Else use cache_dir:
       - if file exists => use it (touch mtime)
       - else download from remote_base_url => cache_dir => use it

    Concurrency-safe:
      - Uses a per-file lock (exclusive create) to avoid double-download.

    Cache-safe:
      - Enforces a cache size limit by evicting oldest CSVs.
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
                # Optional touch local file too (not required)
                _touch(f)
                return f, tsel, None

    # 2) Cache directory
    f_cache = cache / fname
    if f_cache.exists():
        _touch(f_cache)
        return f_cache, tsel, None

    base = str(remote_base_url).strip()
    if not base:
        raise FileNotFoundError(
            "PHOENIX-NewEra grid file not found locally or in cache, and remote_base_url is empty.\n"
            f"Missing: {fname}\n"
            f"Looked in:\n"
            f"  local_grid_dir: {local_grid_dir}\n"
            f"  cache_dir     : {cache}\n"
            "Set st.secrets['newera']['remote_base_url'] (or equivalent) to enable downloading."
        )

    # Validate scheme (allow s3/http/https/file)
    sch = urlparse(base).scheme.lower()
    if sch not in ("s3", "http", "https", "file"):
        raise ValueError(f"Unsupported remote_base_url scheme: {sch!r} (base={base!r})")

    url = _join_base(base, fname)

    # ---- Per-file lock ----
    lock_path = cache / (fname + ".lock")

    # If someone else is downloading, show "waiting"
    waiting = lock_path.exists()
    wait_msg = f"⏳ Waiting for PHOENIX-NewEra grid ({tsel} K) to be cached…"
    dl_msg = f"⬇️ Downloading PHOENIX-NewEra grid ({tsel} K)…"

    with _st_ctx(wait_msg if waiting else dl_msg):
        with _file_lock(lock_path, timeout_s=lock_timeout_s, poll_s=0.25):
            # Re-check after acquiring lock (another session may have downloaded it)
            if f_cache.exists():
                _touch(f_cache)
                return f_cache, tsel, None

            # We are the downloader
            download_file(url, f_cache)
            _touch(f_cache)

    # ---- Cache eviction (outside lock is OK; eviction skips locked files) ----
    enforce_cache_limit(cache, max_bytes=max_cache_bytes)

    return f_cache, tsel, url
