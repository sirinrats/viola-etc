import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

from viola_etc.constants import MAG_SWEEP_MIN, MAG_SWEEP_MAX, MAG_SWEEP_STEP
from viola_etc.models import (
    TargetParams,
    ObservingCondition,
    UserInputs,
    InstrumentConfig,
    SiteConfig,
)
from viola_etc.core.runner import run_etc

from dataclasses import replace


# =========================
# App constants
# =========================
MOLECULE_TEMPLATE_DIR = "./molecular_templates"


# =========================
# Defaults
# =========================
BASE_DEFAULTS = {
    # core result state
    "etc_result": None,
    "etc_error": None,
    "last_run_inputs": None,
    # expander state
    "target_adv_open": False,
    # sweep
    "custom_mag_sweep": False,
    "mag_sweep_min": float(MAG_SWEEP_MIN),
    "mag_sweep_max": float(MAG_SWEEP_MAX),
    "mag_sweep_step": float(MAG_SWEEP_STEP),
    # target
    "band": "H",
    "mag_system": "Vega",
    "m_mag": 8.0,
    "T_star_K": 5800,
    "source_sed": "Blackbody",
    # toggles
    "include_planet_detection": False,
    # observing
    "t_exp_s": 120,
    "n_exp": 50,
    "seeing_fwhm_as": 1.0,
    "use_nod_subtraction": False,
    "sky_model_name": "SkyCalc_Grid",
    "target_alt_deg": 60.0,
    "pwv_mm": 1.0,
    "telescope_aperture_m": 2.0,
}

DET_DEFAULTS = {
    "planet_line_contrast": 1e-4,  # float
    "prom_abs": 0.05,              # float
    "snr_thresh": 100.0,           # float
    "detrend_p": 0.5,              # float
    "detect_sig": 5.0,             # float
    "min_lines_for_calc": 50,      # int
}

INPUT_KEYS = [
    # target
    "band",
    "mag_system",
    "m_mag",
    "T_star_K",
    "source_sed",
    "include_planet_detection",
    # detection
    "planet_line_contrast",
    "prom_abs",
    "snr_thresh",
    "detrend_p",
    "detect_sig",
    "min_lines_for_calc",
    # observing
    "t_exp_s",
    "n_exp",
    "seeing_fwhm_as",
    "use_nod_subtraction",
    "sky_model_name",
    "target_alt_deg",
    "pwv_mm",
    # telescope
    "telescope_aperture_m",
    # sweep
    "mag_sweep_min",
    "mag_sweep_max",
    "mag_sweep_step",
]


# =========================
# Session helpers
# =========================
def ss_default(key, value):
    if key not in st.session_state:
        st.session_state[key] = value


def _coerce_numeric(key: str, default_value):
    """
    Ensure session_state[key] has the same numeric type as default_value.

    """
    if key not in st.session_state:
        st.session_state[key] = default_value
        return

    v = st.session_state.get(key)
    if v is None:
        st.session_state[key] = default_value
        return

    target_type = type(default_value)

    # bool is subclass of int; avoid treating bool as int
    if target_type is int and isinstance(v, bool):
        st.session_state[key] = int(default_value)
        return

    if isinstance(v, target_type):
        return

    if isinstance(v, str):
        s = v.strip()
        try:
            st.session_state[key] = target_type(s)
            return
        except Exception:
            st.session_state[key] = default_value
            return

    if isinstance(v, (int, float, np.integer, np.floating)):
        try:
            st.session_state[key] = target_type(v)
        except Exception:
            st.session_state[key] = default_value
        return

    st.session_state[key] = default_value


def seed_detection_defaults(force_reset: bool = False):
    """
    Ensure detection controls have sensible defaults.

    If force_reset=True:
      - delete keys
      - set defaults (correct type)
    """
    for k, v in DET_DEFAULTS.items():
        if force_reset:
            st.session_state.pop(k, None)
            st.session_state[k] = v
        else:
            if k not in st.session_state:
                st.session_state[k] = v
            else:
                _coerce_numeric(k, v)
                cur = st.session_state.get(k)
                if isinstance(cur, (int, float)) and (cur == 0 or cur <= 1e-14):
                    st.session_state[k] = v


def normalize_sed_name(x: str) -> str:
    """
    Map UI labels to canonical internal SED keys.
    Canonical keys are stable across UI changes.
    """
    s = str(x).strip().lower()
    s = s.replace("_", "-").replace(" ", "")
    if "phoenix" in s and "newera" in s:
        return "phoenix-newera"
    if "phoenix" in s:
        return "phoenix-newera"  # treat any phoenix UI as NewEra in v0.4
    return "blackbody"


def on_toggle_planet_detection():
    """
    Callback for enabling molecular detection.

    When the user turns on molecular detection:
    - Keep the “Target (advanced)” expander open
    - Force the detection-related inputs to reset to defaults once, so you don’t get Streamlit’s weird first-time conditional-widget values (like 0/min_value)
    """
    if not st.session_state.get("include_planet_detection", False):
        return

    st.session_state["target_adv_open"] = True
    st.session_state["_det_force_reset"] = True


def init_session_state():
    """
    Ensures that all required session-state variables exist, have correct numeric types, 
    and that detection parameters are initialized safely so Streamlit widgets don’t crash 
    or display wrong fallback values.

    """
    for k, v in BASE_DEFAULTS.items():
        ss_default(k, v)

    # Coerce numeric base keys (prevents type drift)
    for k in ["m_mag", "mag_sweep_min", "mag_sweep_max", "mag_sweep_step", "seeing_fwhm_as", "target_alt_deg", "pwv_mm", "telescope_aperture_m"]:
        _coerce_numeric(k, BASE_DEFAULTS[k])
    for k in ["T_star_K", "t_exp_s", "n_exp"]:
        _coerce_numeric(k, BASE_DEFAULTS[k])

    seed_detection_defaults(force_reset=False)
    

def on_toggle_custom_mag_sweep():
    # Only act when turning ON
    if not st.session_state.get("custom_mag_sweep", False):
        return

    # Reset to defaults when enabled
    st.session_state["mag_sweep_min"] = float(MAG_SWEEP_MIN)
    st.session_state["mag_sweep_max"] = float(MAG_SWEEP_MAX)
    st.session_state["mag_sweep_step"] = float(MAG_SWEEP_STEP)

# =========================
# Small UI helpers
# =========================
def vspacer(px: int):
    st.markdown(f"<div style='height:{int(px)}px'></div>", unsafe_allow_html=True)


def render_line_green_value(line: str):
    if ":" not in line:
        st.write(line)
        return
    label, value = line.split(":", 1)
    st.markdown(
        f"{label}: <span style='color:#2e7d32; font-weight:600'>{value.strip()}</span>",
        unsafe_allow_html=True,
    )


def plot_simple_xy(x, y, xlabel, ylabel, title, semilogy=False):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    (ax.semilogy if semilogy else ax.plot)(x, y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.35)
    try:
        fig.tight_layout()
    except Exception:
        pass
    st.pyplot(fig, clear_figure=True)


def plot_simple_from_result(result, lam, arr_name: str, title: str, ylabel: str):
    arr = getattr(result, arr_name, None)
    if arr is None:
        return
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

def render_headline_metric(label: str, value: str):
    st.markdown(
        f"""
        <div style="
            padding: 0.75rem 1rem;
            border-radius: 0.8rem;
            background: rgba(46, 125, 50, 0.08);
            border: 1px solid rgba(46, 125, 50, 0.25);
            margin: 0.25rem 0 0.8rem 0;
        ">
          <div style="font-size: 0.9rem; opacity: 0.8; margin-bottom: 0.15rem;">{label}</div>
          <div style="font-size: 1.8rem; font-weight: 800; color: #2e7d32; line-height: 1.1;">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_molecule_metric_line(mol: str, n_lines: int, det_sig, ntr):
    det_txt = "n/a" if (det_sig is None or not np.isfinite(det_sig)) else f"{float(det_sig):.2f}σ"
    ntr_txt = "n/a" if (ntr is None or not np.isfinite(ntr)) else f"{int(np.ceil(float(ntr)))}"

    st.markdown(
        f"""
        <div style="
            padding: 0.55rem 0.8rem;
            border-radius: 0.8rem;
            border: 1px solid rgba(46,125,50,0.25);
            background: rgba(46,125,50,0.08);
            margin: 0.35rem 0 0.55rem 0;
        ">
          <span style="margin-left:0.6rem; opacity:0.85;">Usable {mol.upper()} lines:</span>
          <span style="font-weight:700; color:#2e7d32;"> {int(n_lines)}</span>
          <span style="margin-left:0.6rem; opacity:0.85;">Detection (per transit):</span>
          <span style="font-weight:900; color:#2e7d32;"> {det_txt}</span>
          <span style="margin-left:0.6rem; opacity:0.85;">Required transit(s):</span>
          <span style="font-weight:900; color:#2e7d32;"> {ntr_txt}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================
# Last-run snapshot helpers
# =========================
def get_current_inputs_snapshot() -> dict:
    return {k: st.session_state.get(k) for k in INPUT_KEYS}


def show_inputs_changed_warning():
    last = st.session_state.get("last_run_inputs")
    if not last:
        return
    now = get_current_inputs_snapshot()
    if now != last:
        st.warning("Inputs changed since last run — click **Run ETC** to see new results.")


def get_run_context():
    last = st.session_state.get("last_run_inputs") or {}
    band_used = last.get("band", st.session_state.band)
    mag_used = last.get("m_mag", float(st.session_state.m_mag))
    return str(band_used), float(mag_used)


def last_run_bool(key: str, default: bool = False) -> bool:
    last = st.session_state.get("last_run_inputs") or {}
    return bool(last.get(key, default))


# =========================
# Sections
# =========================
def render_target_section():
    st.subheader("🌟 Target")

    # To do: extended source, emission lines+doppler

    c1, c2, c3 = st.columns(3)
    with c1:
        st.radio("Observing band", ["J", "H", "K"], horizontal=True, key="band")
    with c2:
        st.radio("Magnitude system", ["Vega", "AB"], horizontal=True, key="mag_system")
    with c3:
        st.number_input(
            "Stellar magnitude (mag)",
            min_value=-28.0,
            max_value=40.0,
            step=0.1,
            format="%.2f",
            key="m_mag",
        )

    st.number_input(
        "Stellar effective temperature (K)",
        min_value=2000,
        max_value=40000,
        step=100,
        format="%d",
        key="T_star_K",
    )

    with st.expander("Target (advanced)", expanded=st.session_state.target_adv_open):
        st.session_state.target_adv_open = True

        st.radio(
            "Stellar SED model",
            ["Blackbody", "PHOENIX-NewEra"],
            horizontal=True,
            key="source_sed",
            help="PHOENIX-NewEra grids are available for Teff = 2300–12000 K. If selected with Teff outside this range, the ETC will fall back to the Blackbody model.",
        )

        st.markdown("<hr style='margin: 0.4rem 0 1.8rem 0;'>", unsafe_allow_html=True)

        st.checkbox(
            "Molecular detection",
            key="include_planet_detection",
            on_change=on_toggle_planet_detection,
            help="Enable to estimate molecular detectability. Additional parameters will appear below.",
        )

        if not st.session_state.include_planet_detection:
            return

        # Apply a one-time hard reset on first render after enabling
        force = bool(st.session_state.pop("_det_force_reset", False))
        seed_detection_defaults(force_reset=force)

        st.number_input(
            "Planet-to-star line contrast",
            min_value=1e-14,
            max_value=1.0,
            step=1e-5,
            format="%.1e",
            key="planet_line_contrast",
            help="Order-of-magnitude contrast used for detectability estimates.",
        )

        st.markdown("**Detection thresholds**")
        st.caption(
            "These parameters control how strictly template lines are selected and what detection significance is required."
        )

        c1, c2 = st.columns(2)
        with c1:
            st.number_input(
                "Minimum SNR per resolution",
                min_value=1.0,
                step=10.0,
                format="%.1f",
                key="snr_thresh",
                help="Only template lines in wavelength regions with SNR greater than this value are used in the detectability estimate."
            )

            st.number_input(
                "Minimum line depth (normalized)",
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                format="%.2f",
                key="prom_abs",
            )

            st.number_input(
                "Required detection significance (σ)",
                min_value=1.0,
                step=0.5,
                format="%.1f",
                key="detect_sig",
                help="Target detection significance used to estimate the number of transits (nights) required.",
            )

        with c2:
            st.number_input(
                "Minimum number of lines",
                min_value=1,
                step=1,
                format="%d",
                key="min_lines_for_calc",
                help="Require at least this many template lines to compute detection significance."
            )

            st.number_input(
                "Detrending efficiency (0–1)",
                min_value=0.0,
                max_value=1.0,
                step=0.05,
                format="%.2f",
                key="detrend_p",
            )

        # Hidden by design: valley/peak selection removed.
        # Engine is forced to peak-finding only.


def render_observing_section():
    st.subheader("🌦️ Observing conditions")

    c0, _ = st.columns([1, 1])
    with c0:
        st.radio(
            "Telescope aperture",
            options=[2.0, 8.0],
            format_func=lambda x: f"{x:.0f} m",
            horizontal=True,
            key="telescope_aperture_m",
        )

    c1, c2 = st.columns(2)
    with c1:
        st.number_input(
            "Exposure time per frame (s)",
            min_value=1,
            max_value=3600,
            step=10,
            format="%d",
            key="t_exp_s",
        )
    with c2:
        st.number_input(
            "Number of frames",
            min_value=1,
            step=1,
            format="%d",
            key="n_exp",
        )

    c5, c6 = st.columns(2)
    with c5:
        st.number_input(
            "Target altitude (deg)",
            min_value=1.0,
            max_value=90.0,
            step=1.0,
            format="%.0f",
            key="target_alt_deg",
        )
    with c6:
        st.number_input(
            "PWV (mm)",
            min_value=0.1,
            max_value=20.0,
            step=0.1,
            format="%.1f",
            key="pwv_mm",
        )

    c3, c4 = st.columns(2)
    with c3:
        st.number_input(
            "Seeing FWHM (arcsec)",
            min_value=0.1,
            step=0.1,
            format="%.1f",
            key="seeing_fwhm_as",
        )
    with c4:
        vspacer(28)
        st.checkbox("A–B nod subtraction", key="use_nod_subtraction")



def render_outputs_section():
    st.subheader("📋 Outputs")

    st.checkbox(
        "Customize SNR–magnitude plot range and step",
        key="custom_mag_sweep",
        on_change=on_toggle_custom_mag_sweep,
    )


    if st.session_state.get("custom_mag_sweep", False):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.number_input("Min mag", step=0.5, format="%.2f", key="mag_sweep_min")
        with c2:
            st.number_input("Max mag", step=0.5, format="%.2f", key="mag_sweep_max")
        with c3:
            st.number_input("Mag step", step=0.1, format="%.2f", key="mag_sweep_step")

    return st.button("Run ETC", type="primary")


# =========================
# Run ETC (compute only on button click)
# =========================
def run_etc_if_clicked(run_clicked: bool):

    """
    This function takes current UI values, converts them into structured input objects, 
    calls run_etc() only when the user clicks Run, and then saves either the result or 
    the error so the rest of the app can display it consistently.

    """

    if not run_clicked:
        return

    seed_detection_defaults(force_reset=False)

    target = TargetParams(
        band=str(st.session_state.band),
        mag_system=str(st.session_state.mag_system),
        m_mag=float(st.session_state.m_mag),
        source_sed=normalize_sed_name(st.session_state.source_sed),
        T_star_K=float(st.session_state.T_star_K),
        planet_line_contrast=float(st.session_state.planet_line_contrast),
    )

    obs = ObservingCondition(
        t_exp_s=float(st.session_state.t_exp_s),
        n_exp=int(st.session_state.n_exp),
        use_nod_subtraction=bool(st.session_state.use_nod_subtraction),
        seeing_fwhm_as=float(st.session_state.seeing_fwhm_as),
        sky_model_name=str(st.session_state.sky_model_name),
        target_alt_deg=float(st.session_state.target_alt_deg),
        pwv_mm=float(st.session_state.pwv_mm),
    )

    u = UserInputs(target=target, obs=obs)
    cfg0 = InstrumentConfig()
    cfg = replace(cfg0, d_m=float(st.session_state.telescope_aperture_m))
    site = SiteConfig()

    kwargs = dict(
        cfg=cfg,
        site=site,
        mag_sweep_min=float(st.session_state.mag_sweep_min),
        mag_sweep_max=float(st.session_state.mag_sweep_max),
        mag_sweep_step=float(st.session_state.mag_sweep_step),
        molecule_dir=MOLECULE_TEMPLATE_DIR,
        enable_molecules=bool(st.session_state.include_planet_detection),
        prom_abs=float(st.session_state.prom_abs),
        snr_thresh=float(st.session_state.snr_thresh),
        n_frames_per_transit=int(st.session_state.n_exp),
        detrend_p=float(st.session_state.detrend_p),
        detect_sig=float(st.session_state.detect_sig),
        min_lines_for_calc=int(st.session_state.min_lines_for_calc),
        # Forced behavior: peak finding only
        find_valleys=False,
    )

    try:
        st.session_state.etc_result = run_etc(u, **kwargs)
        st.session_state.etc_error = None
        st.session_state.last_run_inputs = get_current_inputs_snapshot()
    except Exception as e:
        st.session_state.etc_result = None
        st.session_state.etc_error = str(e)


# =========================
# Tabs
# =========================
def render_tab_result(result):
    band_used, mag_used = get_run_context()

    # Pull summary lines once
    lines = list(getattr(result, "summary_lines", None) or [])

    # Headline: Median SNR per resolution
    headline_key = "Median SNR per resolution:"

    headline = [ln for ln in lines if ln.startswith(headline_key)]
    rest = [ln for ln in lines if not ln.startswith(headline_key)]

    if headline:
        _, v = headline[0].split(":", 1)
        render_headline_metric("Median SNR per resolution", v.strip())

    # --- Plots
    lam = getattr(result, "lam_um", None)
    snr = getattr(result, "snr_res", None)
    if lam is not None and snr is not None:
        lam = np.asarray(lam, float)
        snr = np.asarray(snr, float)
        m = np.isfinite(lam) & np.isfinite(snr)
        if np.any(m):
            plot_simple_xy(
                lam[m],
                snr[m],
                xlabel="Wavelength (μm)",
                ylabel="SNR per resolution",
                title=f"SNR per resolution vs wavelength ({band_used}-band)",
            )

    if getattr(result, "mag_grid", None) is not None and getattr(result, "snr_med_grid", None) is not None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.semilogy(result.mag_grid, result.snr_med_grid, label="Median SNR / res")
        ax.axvline(mag_used, linestyle="--", label=f"Target mag={mag_used:.1f}")
        ax.set_xlabel(f"{band_used}-band magnitude")
        ax.set_ylabel("Median SNR per resolution")
        ax.set_title(f"Median SNR per resolution vs magnitude ({band_used}-band)")
        ax.grid(True, which="major", alpha=0.5)
        ax.grid(True, which="minor", alpha=0.25)
        ax.minorticks_on()
        ax.legend(loc="best")
        try:
            fig.tight_layout()
        except Exception:
            pass
        st.pyplot(fig, clear_figure=True)


def render_tab_sky(result):

    band_used, _ = get_run_context()

    lam = getattr(result, "lam_um", None)
    if lam is None:
        st.warning("No arrays to plot.")
        return
    lam = np.asarray(lam, float)

    for arr_name, title, ylabel in [
        ("trans", f"Atmospheric transmission ({band_used}-band)", "Atmospheric transmission"),
        ("zl", f"Zodiacal emission ({band_used}-band)", "Sky emission [ph s^-1 m^-2 μm^-1 arcsec^-2]"),
        ("oh", f"OH airglow ({band_used}-band)", "Emission [ph s^-1 m^-2 μm^-1 arcsec^-2]"),
        ("sml", f"Scattered moonlight ({band_used}-band)", "Emission [ph s^-1 m^-2 μm^-1 arcsec^-2]"),
    ]:
        plot_simple_from_result(result, lam, arr_name, title, ylabel)


def render_tab_signal_noise(result, cfg: InstrumentConfig):
    band_used, _ = get_run_context()

    lam = getattr(result, "lam_um", None)
    if lam is None:
        st.warning("No arrays to plot.")
        return
    lam = np.asarray(lam, float)

    for arr_name, title, ylabel in [
        ("signal_res_e", f"Signal per resolution ({band_used}-band)", "Signal per resolution [e-]"),
        ("signal_pix_e", f"Signal per pixel ({band_used}-band, nx×ny = {cfg.nx}×{cfg.ny})", "Signal per pixel [e-]"),
        ("sky_res_e", f"Sky background ({band_used}-band)", "Sky background per resolution [e-]"),
        ("thermal_res_e", f"Thermal background ({band_used}-band)", "Thermal background per resolution [e-]"),
        ("dark_res_e", f"Dark current ({band_used}-band)", "Dark counts per resolution [e-]"),
    ]:
        plot_simple_from_result(result, lam, arr_name, title, ylabel)


def render_tab_molecules(result):
    if not last_run_bool("include_planet_detection", False):
        st.info("Enable **Target (advanced) → Molecular detection**, then click **Run ETC**.")
        return

    templates = getattr(result, "molecule_templates", None) or {}
    metrics = getattr(result, "molecule_metrics", None) or {}

    if not templates:
        st.warning("No molecule templates were loaded.")
        st.write("Template directory:", MOLECULE_TEMPLATE_DIR)
        return

    mols = sorted([k.upper() for k in templates.keys()])
    mol_html = ", ".join(
        [f"<span style='color:#2e7d32; font-weight:600'>{m}</span>" for m in mols]
    )

    st.markdown(f"**Loaded templates:** {mol_html}", unsafe_allow_html=True)

    lam = np.asarray(getattr(result, "lam_um", []), float)
    snr = np.asarray(getattr(result, "snr_res", []), float)

    for mol, rec in templates.items():
        tpl = np.asarray(rec.get("template_resampled", []), float)
        if tpl.size == 0 or lam.size == 0:
            continue

        fig, ax1 = plt.subplots(figsize=(8.5, 4.5))
        ax1.plot(lam, tpl, lw=1.8, label=f"{mol.upper()} template")
        ax1.set_ylim(-0.05, 1.05)
        ax1.set_xlabel("Wavelength (μm)")
        ax1.set_ylabel("Template (normalized)")
        ax1.grid(alpha=0.3)

        mm = metrics.get(mol, {})
        pks = np.asarray(mm.get("peaks_idx_thr", np.array([], dtype=int)), dtype=int)
        if pks.size and pks.max() < len(lam):
            ax1.plot(lam[pks], tpl[pks], linestyle="none", marker="o", ms=4, label="Selected lines")

        ax2 = ax1.twinx()
        if snr.size == lam.size:
            ax2.plot(lam, snr, lw=1.2, alpha=0.25, label="Target spectrum")
        ax2.set_ylabel("SNR per resolution")

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
            render_molecule_metric_line(mol, n_lines, spt, ntr)


def render_tab_debug(result):
    st.subheader("Debug")

    st.markdown("**Session-state snapshot (values + types)**")
    rows = []
    for k in INPUT_KEYS:
        v = st.session_state.get(k, None)
        rows.append({"key": k, "value": repr(v), "type": type(v).__name__})
    st.dataframe(rows, width="stretch", hide_index=True)

    st.markdown("**Last-run inputs**")
    st.json(st.session_state.get("last_run_inputs", {}))

    st.markdown("**Result metadata**")
    st.json(getattr(result, "meta", {}))


# =========================
# Main
# =========================
def main():
    init_session_state()

    st.set_page_config(
    page_title="VIOLA ETC",
    page_icon=Image.open("assets/viola_bw_icon.png"),
    layout="centered", 
    )

    st.header("VIOLA Exposure Time Calculator -- Ver 0.4 🌈 ", divider="rainbow")

    render_target_section()
    render_observing_section()
    run_clicked = render_outputs_section()

    run_etc_if_clicked(run_clicked)

    if st.session_state.etc_error:
        st.error(f"ETC failed: {st.session_state.etc_error}")

    result = st.session_state.etc_result
    if result is None:
        st.info("Click **Run ETC** to generate outputs.")
        st.stop()

    show_inputs_changed_warning()

    cfg0 = InstrumentConfig()
    cfg = replace(cfg0, d_m=float(st.session_state.get("telescope_aperture_m", cfg0.d_m)))

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Results", "Sky model", "Signal & Noise", "Molecules", "Debug"]
    )

    with tab1:
        render_tab_result(result)
    with tab2:
        render_tab_sky(result)
    with tab3:
        render_tab_signal_noise(result, cfg)
    with tab4:
        render_tab_molecules(result)
    with tab5:
        render_tab_debug(result)


if __name__ == "__main__":
    main()
