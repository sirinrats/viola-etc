# VIOLA ETC v0.5

Exposure-time calculator for the VIOLA near-infrared high-resolution spectrograph (Wendelstein 2 m / VLT 8 m).

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Required data (not in repo)

The grids are too large for GitHub (~12 GB total) and are hosted on S3.  The app downloads on demand on first use, given AWS credentials in Streamlit secrets.  Local copies (e.g., for offline use) can be placed at:

- `stellar_models/`                — PHOENIX-NewEra stellar atmospheres (~73 CSVs)
- `molecular_templates/`           — molecular transmission templates (PSG + petitRADTRANS, ~69 CSVs)
- `sky_models/`                    — SkyCalc telluric / sky-emission grid (45 FITS files)

## Streamlit secrets

`.streamlit/secrets.toml`:

```toml
[aws]
access_key_id     = "..."
secret_access_key = "..."
region_name       = "..."

[newera]
remote_base_url   = "s3://<bucket>/stellar_models/"

[molecular_templates]
remote_base_url   = "s3://<bucket>/molecular_templates/"

[skycalc]
remote_base_url   = "s3://<bucket>/sky_models/"
```

## Citation

(See VIOLA ETC technical documentation, distributed separately.)
