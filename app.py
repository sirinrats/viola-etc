import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from constants import MAG_SWEEP_MIN, MAG_SWEEP_MAX, MAG_SWEEP_STEP

from viola_etc.models import (
    TargetParams,
    ObservingCondition,
    UserInputs,
    InstrumentConfig,
    SiteConfig,
)
from viola_etc.runner import run_etc


# -------------------------
# Session state
# -------------------------
if "etc_result" not in st.session_state:
    st.session_state.etc_result = None
if "etc_error" not in st.session_state:
    st.session_state.etc_error = None

if "mag_sweep_min" not in st.session_state:
    st.session_state.mag_sweep_min = float(MAG_SWEEP_MIN)
if "mag_sweep_max" not in st.session_state:
    st.session_state.mag_sweep_max = float(MAG_SWEEP_MAX)
if "mag_sweep_step" not in st.session_state:
    st.session_state.mag_sweep_step = float(MAG_SWEEP_STEP)

# molecule defaults
if "molecule_dir" not in st.session_state:
    st.session_state.molecule_dir = "./molecular_lines"
if "enable_molecules" not in st.session_state:
    st.session_state.enable_molecules = True
if "prom_abs" not in st.session_state:
    st.session_state.prom_abs = 0.05
if "snr_thresh" not in st.session_state:
    st.session_state.snr_thresh = 100.0
if "n_frames_per_transit" not in st.session_state:
    st.session_state.n_frames_per_transit = 50
if "detrend_p" not in st.session_state:
    st.session_state.detrend_p = 0.5
if "detect_sig" not in st.session_state:
    st.session_state.detect_sig = 5.0
if "min_lines_for_calc" not in st.session_state:
    st.session_state.min_lines_for_calc = 50
if "find_valleys" not in st.session_state:
    st.session_state.find_valleys = False

# -------------------------
# Page
# -------------------------
st.header("VIOLA ETC -- Ver 0.3 🌈 ", divider="rainbow")

st.markdown(
    """
<style>
div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input {
  background-color: #EEF5FF !important;
  color: #111111 !important;
}
div[data-testid="stTextInput"] input:focus,
div[data-testid="stTextInput"] input:focus-visible,
div[data-testid="stNumberInput"] input:focus,
div[data-testid="stNumberInput"] input:focus-visible {
  background-color: #EEF5FF !important;
  box-shadow: none !important;
  outline: none !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# =========================
# Target module
# =========================
st.subheader("🌟 Target")

c1, c2, c3 = st.columns(3)
with c1:
    band = st.radio("Observing band", ["J", "H", "K"], horizontal=True, key="band")
with c2:
    mag_system = st.radio("Magnitude system", ["Vega", "AB"], horizontal=True, key="mag_system")
with c3:
    m_mag = st.number_input("Stellar magnitude (mag)", value=8.0, step=0.1, key="m_mag")

with st.expander("Target (advanced)", expanded=False):
    star_name = st.text_input("Star name (for molecule templates)", value="HD21520b", key="star_name")
    source_sed = st.radio("Stellar SED model", ["Blackbody", "PHOENIX"], horizontal=True, key="source_sed")
    T_star_K = st.number_input(
        "Stellar effective temperature, Teff (K)",
        value=5800,
        step=100,
        min_value=1,
        format="%d",
        key="T_star_K",
    )

planet_line_contrast = st.number_input(
    "Planet-to-star line contrast",
    min_value=1e-14,
    max_value=1.0,
    value=1e-4,
    format="%.1e",
    key="planet_line_contrast",
)

# =========================
# Observing module
# =========================
st.subheader("☁️ Observing conditions")

c1, c2 = st.columns(2)
with c1:
    t_exp_s = st.number_input(
        "Exposure time per frame (s)",
        value=600,
        step=10,
        min_value=1,
        format="%d",
        key="t_exp_s",
    )
with c2:
    n_exp = st.number_input(
        "Number of frames",
        value=3,
        step=1,
        min_value=1,
        format="%d",
        key="n_exp",
    )

c3, c4 = st.columns(2)
with c3:
    seeing_fwhm_as = st.number_input(
        "Seeing FWHM (arcsec)",
        value=1.0,
        step=0.1,
        min_value=0.1,
        key="seeing_fwhm_as",
    )

with c4:
    left_pad, box = st.columns([0.25, 0.75])
    with left_pad:
        st.write("")
    with box:
        st.write("")
        st.write("")
        use_nod_subtraction = st.checkbox("A–B nod subtraction", value=True, key="use_nod_subtraction")

with st.expander("Observing conditions (advanced)", expanded=False):
    sky_model_name = st.text_input(
        "Precomputed sky model",
        value="skymodel_Paranal_hd21520_pwv1p0",
        key="sky_model_name",
        help="Folder name under ./sky_models (expects skytable.fits inside).",
    )

# =========================
# Outputs module
# =========================
st.subheader("📋 Outputs")

with st.expander("SNR vs magnitude (advanced)", expanded=False):
    st.session_state.mag_sweep_min = st.number_input("Mag sweep min", value=float(st.session_state.mag_sweep_min), step=0.5)
    st.session_state.mag_sweep_max = st.number_input("Mag sweep max", value=float(st.session_state.mag_sweep_max), step=0.5)
    st.session_state.mag_sweep_step = st.number_input("Mag sweep step", value=float(st.session_state.mag_sweep_step), step=0.1)

with st.expander("Molecules (advanced)", expanded=False):
    st.session_state.enable_molecules = st.checkbox("Enable molecule templates", value=bool(st.session_state.enable_molecules))
    st.session_state.molecule_dir = st.text_input("Molecule templates directory", value=str(st.session_state.molecule_dir))

    c1, c2, c3 = st.columns(3)
    with c1:
        st.session_state.prom_abs = st.number_input("prom_abs", value=float(st.session_state.prom_abs), step=0.01)
        st.session_state.snr_thresh = st.number_input("snr_thresh", value=float(st.session_state.snr_thresh), step=10.0)
    with c2:
        st.session_state.n_frames_per_transit = st.number_input("n_frames_per_transit", value=int(st.session_state.n_frames_per_transit), step=1)
        st.session_state.detrend_p = st.number_input("detrend_p", value=float(st.session_state.detrend_p), step=0.05)
    with c3:
        st.session_state.detect_sig = st.number_input("detect_sig", value=float(st.session_state.detect_sig), step=0.5)
        st.session_state.min_lines_for_calc = st.number_input("min_lines_for_calc", value=int(st.session_state.min_lines_for_calc), step=1)

    st.session_state.find_valleys = st.checkbox("Find valleys (absorption) instead of peaks", value=bool(st.session_state.find_valleys))

run_clicked = st.button("Run ETC", type="primary")

if run_clicked:
    target = TargetParams(
        band=band,
        mag_system=mag_system,
        m_mag=float(m_mag),
        star_name=str(star_name).strip(),
        source_sed=source_sed.strip().lower(),
        T_star_K=float(T_star_K),
        planet_line_contrast=float(planet_line_contrast),
    )

    obs = ObservingCondition(
        t_exp_s=float(t_exp_s),
        n_exp=int(n_exp),
        use_nod_subtraction=bool(use_nod_subtraction),
        seeing_fwhm_as=float(seeing_fwhm_as),
        sky_model_name=str(sky_model_name),
    )

    u = UserInputs(target=target, obs=obs)
    cfg = InstrumentConfig()
    site = SiteConfig()

    try:
        st.session_state.etc_result = run_etc(
            u,
            cfg=cfg,
            site=site,
            mag_sweep_min=float(st.session_state.mag_sweep_min),
            mag_sweep_max=float(st.session_state.mag_sweep_max),
            mag_sweep_step=float(st.session_state.mag_sweep_step),
            molecule_dir=str(st.session_state.molecule_dir),
            enable_molecules=bool(st.session_state.enable_molecules),
            prom_abs=float(st.session_state.prom_abs),
            snr_thresh=float(st.session_state.snr_thresh),
            n_frames_per_transit=int(st.session_state.n_frames_per_transit),
            detrend_p=float(st.session_state.detrend_p),
            detect_sig=float(st.session_state.detect_sig),
            min_lines_for_calc=int(st.session_state.min_lines_for_calc),
            find_valleys=bool(st.session_state.find_valleys),
        )
        st.session_state.etc_error = None
    except Exception as e:
        st.session_state.etc_result = None
        st.session_state.etc_error = str(e)

if st.session_state.etc_error:
    st.error(f"ETC failed: {st.session_state.etc_error}")
    st.stop()

result = st.session_state.etc_result
if result is None:
    st.info("Click Run ETC to generate outputs.")
    st.stop()

cfg = InstrumentConfig()

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Result", "Sky model plots", "Signal / Noise plots", "Molecules", "Debug (metadata)"]
)

# -------------------------
# Tab 1: Result
# -------------------------
with tab1:
    for line in (getattr(result, "summary_lines", None) or []):
        if ":" in line:
            label, value = line.split(":", 1)
            st.markdown(
                f"{label}: <span style='color:#2e7d32; font-weight:600'>{value.strip()}</span>",
                unsafe_allow_html=True,
            )
        else:
            st.write(line)

    lam = getattr(result, "lam_um", None)
    snr = getattr(result, "snr_res", None)
    if lam is not None and snr is not None:
        lam = np.asarray(lam, float)
        snr = np.asarray(snr, float)
        m = np.isfinite(lam) & np.isfinite(snr)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(lam[m], snr[m])
        ax.set_xlabel("Wavelength (μm)")
        ax.set_ylabel("SNR per resolution element")
        ax.set_title(f"SNR per res vs wavelength ({band}-band)")
        ax.grid(True, which="both", alpha=0.35)
        try:
            fig.tight_layout()
        except Exception:
            pass
        st.pyplot(fig, clear_figure=True)

    if getattr(result, "mag_grid", None) is not None and getattr(result, "snr_med_grid", None) is not None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.semilogy(result.mag_grid, result.snr_med_grid, label="Median SNR / res")
        ax.axvline(float(m_mag), linestyle="--", label=f"Target mag={float(m_mag):.1f}")
        ax.set_xlabel(f"{band}-band magnitude")
        ax.set_ylabel("Median SNR per resolution element")
        ax.set_title(f"Median SNR vs magnitude ({band}-band)")
        ax.grid(True, which="major", alpha=0.5)
        ax.grid(True, which="minor", alpha=0.25)
        ax.minorticks_on()
        ax.legend(loc="best")
        try:
            fig.tight_layout()
        except Exception:
            pass
        st.pyplot(fig, clear_figure=True)

# -------------------------
# Tab 2: Sky plots
# -------------------------
with tab2:
    lam = getattr(result, "lam_um", None)
    if lam is None:
        st.warning("No arrays to plot.")
        st.stop()
    lam = np.asarray(lam, float)

    for arr_name, title, ylabel in [
        ("trans", f"Atmospheric transmission ({band}-band)", "Atmospheric transmission"),
        ("zl", f"Zodiacal emission ({band}-band)", "Sky emission [ph s^-1 m^-2 μm^-1 arcsec^-2]"),
        ("oh", f"OH airglow ({band}-band)", "Emission [ph s^-1 m^-2 μm^-1 arcsec^-2]"),
        ("sml", f"Scattered moonlight ({band}-band)", "Emission [ph s^-1 m^-2 μm^-1 arcsec^-2]"),
    ]:
        arr = getattr(result, arr_name, None)
        if arr is None:
            continue
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(lam, arr)
        ax.set_xlabel("Wavelength (μm)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        try:
            fig.tight_layout()
        except Exception:
            pass
        st.pyplot(fig, clear_figure=True)

# -------------------------
# Tab 3: Signal/Noise plots
# -------------------------
with tab3:
    lam = getattr(result, "lam_um", None)
    if lam is None:
        st.warning("No arrays to plot.")
        st.stop()
    lam = np.asarray(lam, float)

    for arr_name, title, ylabel in [
        ("signal_res_e", f"Signal per res ({band}-band)", "Signal per res [e-]"),
        ("signal_pix_e", f"Signal per pixel ({band}-band, nx×ny = {cfg.nx}×{cfg.ny})", "Signal per pixel [e-]"),
        ("sky_res_e", f"Sky background ({band}-band)", "Sky background per res [e-]"),
        ("thermal_res_e", f"Thermal background ({band}-band)", "Thermal background per res [e-]"),
        ("dark_res_e", f"Dark current ({band}-band)", "Dark counts per res [e-]"),
    ]:
        arr = getattr(result, arr_name, None)
        if arr is None:
            continue
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(lam, arr)
        ax.set_xlabel("Wavelength (μm)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        try:
            fig.tight_layout()
        except Exception:
            pass
        st.pyplot(fig, clear_figure=True)

# -------------------------
# Tab 4: Molecules (ROBUST)
# -------------------------
with tab4:
    if not bool(st.session_state.enable_molecules):
        st.info("Molecules are disabled (enable in Outputs → Molecules).")
        st.stop()

    # ✅ key fix: use getattr to avoid AttributeError if etc_core is older
    templates = getattr(result, "molecule_templates", None) or {}
    metrics = getattr(result, "molecule_metrics", None) or {}

    if not templates:
        st.warning(
            "No molecule templates loaded OR your etc_core.py is missing molecule outputs.\n\n"
            "If you just updated etc_core.py, restart Streamlit to reload modules."
        )
        st.write("Expected files like:")
        st.code("HD21520b_CH4_H_band_R300000.csv   (columns: wl_micron, transit_cm)")
        st.write(f"Current molecule_dir: {st.session_state.molecule_dir}")
        st.stop()

    st.write("Loaded templates:")
    st.write(", ".join([k.upper() for k in templates.keys()]))

    lam = np.asarray(getattr(result, "lam_um", []), float)
    snr = np.asarray(getattr(result, "snr_res", []), float)

    for mol, rec in templates.items():
        tpl = np.asarray(rec.get("template_resampled", []), float)
        if tpl.size == 0 or lam.size == 0:
            continue

        fig, ax1 = plt.subplots(figsize=(8.5, 4.5))
        ax1.plot(lam, tpl, lw=1.8, label=f"{mol.upper()} template (norm)")
        ax1.set_ylim(-0.05, 1.05)
        ax1.set_xlabel("Wavelength (μm)")
        ax1.set_ylabel("Template (0..1)")
        ax1.grid(alpha=0.3)

        mm = metrics.get(mol, {})
        pks = np.asarray(mm.get("peaks_idx_thr", np.array([], dtype=int)), dtype=int)
        if pks.size and pks.max() < len(lam):
            ax1.plot(lam[pks], tpl[pks], linestyle="none", marker="o", ms=4, label="lines kept")

        ax2 = ax1.twinx()
        if snr.size == lam.size:
            ax2.plot(lam, snr, lw=1.2, alpha=0.25, label="SNR")
        ax2.set_ylabel("SNR per res")

        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="best", fontsize=9)
        ax1.set_title(f"{mol.upper()}")

        try:
            fig.tight_layout()
        except Exception:
            pass
        st.pyplot(fig, clear_figure=True)

        n_lines = mm.get("n_lines", None)
        spt = mm.get("snr_per_transit", None)
        ntr = mm.get("n_transits_req", None)

        if n_lines is not None:
            if (spt is not None) and np.isfinite(spt) and (ntr is not None) and np.isfinite(ntr):
                st.write(f"{mol.upper()}: n_lines={int(n_lines)} | snr/transit={spt:.2f} | n_transits={ntr:.2f}")
            else:
                st.write(f"{mol.upper()}: n_lines={int(n_lines)} | snr/transit=n/a | n_transits=n/a")

# -------------------------
# Tab 5: Debug
# -------------------------
with tab5:
    st.json(getattr(result, "meta", {}))
