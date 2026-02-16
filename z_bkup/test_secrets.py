import streamlit as st
import boto3
import awswrangler as wr
import numpy as np
import pandas as pd
from pathlib import Path

st.title("Secrets + S3 read test")

# 1) show what Streamlit sees
st.write("Secrets keys:", list(st.secrets.keys()))

# 2) Support BOTH secrets layouts:
#    A) nested: st.secrets["aws"]["access_key_id"]
#    B) top-level: st.secrets["access_key_id"]
aws_cfg = st.secrets.get("aws")

# quick sanity check (don’t print secrets)
st.write(
    {
        "region": aws_cfg.get("region", "us-east-1"),
        "has_access_key_id": "access_key_id" in aws_cfg,
        "has_secret_access_key": "secret_access_key" in aws_cfg,
        "access_key_prefix": (str(aws_cfg.get("access_key_id", ""))[:4] + "****")
        if aws_cfg.get("access_key_id")
        else None,
    }
)

# 3) create boto3 session from secrets (once)
session = boto3.Session(
    aws_access_key_id=aws_cfg["access_key_id"],
    aws_secret_access_key=aws_cfg["secret_access_key"],
    region_name=aws_cfg.get("region", "us-east-1"),
)

# 4) read one CSV from S3
path = "s3://viola-etc-project-us-east-1/newera_grids_r300k/lte03000-4.50-0.0.PHOENIX-NewEra-ACES-COND-2023.HSR.csv"
st.write("Reading:", path)

def _read_from_s3_newera_csv(path: str) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Read NewEra CSV from S3.

    CSV columns: wl_A, flux
      wl_A  : Angstrom
      flux  : erg / s / cm^2 / cm  (per cm of wavelength)
    """
    df = wr.s3.read_csv(path=path, boto3_session=session)

    if "wl_A" not in df.columns or "flux" not in df.columns:
        raise ValueError(f"Expected columns wl_A, flux in {path}. Got: {list(df.columns)}")

    wl_A = df["wl_A"].to_numpy(dtype=float, copy=False)
    flux = df["flux"].to_numpy(dtype=float, copy=False)

    m = np.isfinite(wl_A) & np.isfinite(flux) & (wl_A > 0)
    wl_A = wl_A[m]
    flux = flux[m]

    if wl_A.size < 10:
        raise ValueError(f"Too few valid samples in {Path(path).name}")

    return wl_A, flux, int(len(df))

try:
    wl_A, flux, n_rows = _read_from_s3_newera_csv(path)

    st.success(f"Loaded {n_rows:,} rows")
    st.write("First 5 rows (like df.head()):")

    head_df = pd.DataFrame({"wl_A": wl_A[:5], "flux": flux[:5]})
    st.dataframe(head_df, use_container_width=True)

except Exception as e:
    st.error("Failed to read from S3")
    st.exception(e)
