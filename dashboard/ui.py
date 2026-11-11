"""Accessible shared visual system and navigation for the electricity platform."""
from __future__ import annotations

import os
import requests
from html import escape
import streamlit as st

API_DEFAULT = os.getenv("ELECTRICITY_API_URL", "http://127.0.0.1:8000")

PAGES = [
    ("Home", "app.py", "🏠"),
    ("Stations", "pages/station_management.py", "📡"),
    ("Models", "pages/model_management.py", "🧠"),
    ("Pipeline", "pages/pipeline_control.py", "⚙️"),
    ("Monitoring", "pages/monitoring.py", "📈"),
]

CSS = r"""
<style>
:root { color-scheme: dark; }
.stApp {
  background:
    radial-gradient(ellipse at 78% 0%, rgba(28,105,190,.20), transparent 38%),
    radial-gradient(ellipse at 5% 55%, rgba(28,75,145,.13), transparent 32%),
    linear-gradient(135deg, #061326 0%, #091a31 48%, #071326 100%);
  color: #edf4ff;
}
[data-testid="stHeader"] { background: rgba(5,16,33,.78); }
[data-testid="stSidebarNav"] { display:none; }
[data-testid="stToolbar"] { right: 1rem; }
.block-container { max-width: 1480px; padding-top: 1.3rem; padding-bottom: 3rem; }
[data-testid="stSidebar"] { background: linear-gradient(180deg,#08172c,#071326); border-right: 1px solid rgba(122,169,226,.16); }
[data-testid="stSidebar"] label, [data-testid="stSidebar"] p { color: #bac9df; }
.ef-brand-row { display:flex; align-items:center; justify-content:space-between; gap:16px; padding: 4px 0 16px; border-bottom:1px solid rgba(142,183,235,.20); margin-bottom:12px; }
.ef-brand { display:flex; align-items:center; gap:12px; color:#f3f7ff; font-weight:800; font-size:1.15rem; letter-spacing:.2px; }
.ef-bolt { display:grid; place-items:center; width:42px; height:42px; border-radius:14px; background:linear-gradient(140deg,#159ee9,#6555f4); box-shadow:0 7px 24px rgba(38,125,237,.30); font-size:1.45rem; }
.ef-brand small { display:block; color:#91a8c8; font-size:.72rem; font-weight:500; letter-spacing:.2px; margin-top:2px; }
.ef-health { border:1px solid rgba(53,211,157,.25); background:rgba(23,160,120,.12); color:#70e7b7; border-radius:999px; padding:7px 12px; font-size:.78rem; white-space:nowrap; }
[data-testid="stPageLink"] a { border:1px solid rgba(126,169,222,.18); background:rgba(14,35,62,.64); border-radius:12px; min-height:40px; transition:all .18s ease; }
[data-testid="stPageLink"] a:hover { border-color:rgba(64,174,255,.65); background:rgba(30,84,143,.40); transform:translateY(-1px); }
[data-testid="stPageLink"] p { color:#dce9fb; font-size:.88rem; font-weight:650; }
.ef-hero { position:relative; overflow:hidden; border:1px solid rgba(83,157,240,.28); border-radius:24px; padding:34px 34px 28px; background: radial-gradient(ellipse at 82% 44%,rgba(47,119,210,.35),transparent 36%), linear-gradient(112deg,rgba(5,24,48,.98),rgba(13,42,78,.92) 55%,rgba(30,38,91,.82)); box-shadow:0 18px 50px rgba(0,0,0,.22); }
.ef-hero:after { content:""; position:absolute; width:420px; height:420px; right:-130px; top:-230px; border:1px solid rgba(91,180,255,.25); border-radius:50%; box-shadow:0 0 0 34px rgba(91,180,255,.035),0 0 0 70px rgba(91,180,255,.025); }
.ef-eyebrow { display:inline-flex; align-items:center; gap:8px; border:1px solid rgba(106,181,255,.32); background:rgba(41,118,193,.16); border-radius:999px; color:#b8dcff; padding:7px 12px; font-size:.78rem; font-weight:650; margin-bottom:16px; }
.ef-hero h1 { color:#f6f9ff; font-size:clamp(2.2rem,4.5vw,3.8rem); line-height:1.06; letter-spacing:-1.5px; margin:0 0 14px 0; font-weight:850; }
.ef-gradient { background:linear-gradient(90deg,#44c5ff,#8886ff 75%); -webkit-background-clip:text; background-clip:text; color:transparent; }
.ef-hero p { color:#c2d3e9; font-size:1.03rem; line-height:1.65; max-width:690px; margin:0; }
.ef-hero-visual { min-height:230px; display:flex; flex-direction:column; justify-content:center; }
.ef-chart-panel { border:1px solid rgba(129,182,242,.28); background:linear-gradient(145deg,rgba(19,54,100,.72),rgba(14,28,61,.78)); border-radius:19px; padding:17px 18px; box-shadow:0 14px 36px rgba(0,0,0,.22); }
.ef-chart-panel .small { color:#a9c0df; font-size:.75rem; }
.ef-chart-panel .big { color:#f5f8ff; font-size:1.7rem; font-weight:800; margin-top:3px; }
.ef-wave { width:100%; height:104px; margin-top:8px; }
.ef-section-title { color:#f2f6ff; font-size:1.1rem; font-weight:750; margin:22px 0 10px; }
.ef-card { height:100%; min-height:175px; border:1px solid rgba(119,166,224,.23); border-radius:18px; background:linear-gradient(145deg,rgba(16,42,74,.82),rgba(10,26,49,.88)); padding:20px 20px 17px; box-shadow:0 9px 28px rgba(0,0,0,.14); }
.ef-card .icon { width:42px; height:42px; border-radius:13px; display:grid; place-items:center; background:rgba(34,139,226,.17); border:1px solid rgba(68,164,246,.30); font-size:1.35rem; margin-bottom:14px; }
.ef-card h3 { color:#f3f7ff; font-size:1.04rem; margin:0 0 7px; font-weight:750; }
.ef-card p { color:#aebfd6; font-size:.87rem; line-height:1.55; margin:0; }
.ef-panel { border:1px solid rgba(119,166,224,.20); border-radius:18px; background:rgba(11,29,53,.72); padding:17px 18px; }
.ef-kpi-label { color:#93abc9; font-size:.77rem; text-transform:uppercase; letter-spacing:.7px; }
.ef-kpi-value { color:#f2f7ff; font-size:1.55rem; font-weight:800; margin-top:4px; }
.ef-kpi-note { color:#8fa9c8; font-size:.75rem; margin-top:4px; }
.ef-flow { border:1px solid rgba(108,155,211,.22); border-radius:16px; background:linear-gradient(90deg,rgba(12,37,69,.9),rgba(17,35,68,.88)); padding:15px 12px; text-align:center; }
.ef-flow b { color:#eaf2ff; display:block; font-size:.86rem; }
.ef-flow span { color:#8fa8c8; font-size:.74rem; }
.ef-muted { color:#9bb0cb; font-size:.82rem; }
.stMarkdown, p, li { color:inherit; }
.stButton button, .stDownloadButton button { border-radius:12px; border:1px solid rgba(111,173,242,.35); background:linear-gradient(100deg,#148fe0,#5546e8); color:white; font-weight:700; min-height:42px; }
.stButton button:hover, .stDownloadButton button:hover { border-color:#7ac7ff; color:white; box-shadow:0 7px 24px rgba(27,127,225,.25); }
[data-testid="stMetric"] { border:1px solid rgba(119,166,224,.18); background:rgba(13,34,61,.70); border-radius:15px; padding:14px 16px; }
[data-testid="stMetricLabel"] { color:#9eb3d0 !important; }
[data-testid="stMetricValue"] { color:#f3f7ff !important; }
hr { border-color:rgba(119,166,224,.18); }
h1,h2,h3 { color:#f3f7ff; }
[data-testid="stDataFrame"] { border:1px solid rgba(119,166,224,.20); border-radius:12px; }
/* Accessibility: Streamlit may use its light theme while the app shell is dark.
   Force legible widget labels on the dark shell, but keep text DARK inside the
   light input fields. Shared across Home, Stations, Models, Pipeline and Monitoring. */
.stApp [data-testid="stWidgetLabel"],
.stApp [data-testid="stWidgetLabel"] p,
.stApp [data-testid="stWidgetLabel"] span,
.stApp [data-testid="stFileUploader"] label,
.stApp [data-testid="stFileUploader"] label p,
.stApp [data-testid="stCheckbox"] label p,
.stApp [data-testid="stRadio"] label p,
.stApp [data-testid="stSlider"] label p { color:#e8f2ff !important; opacity:1 !important; }
.stApp [data-testid="stCaptionContainer"],
.stApp [data-testid="stCaptionContainer"] p,
.stApp .stCaption { color:#b9cce4 !important; opacity:1 !important; }
.stApp [data-testid="stExpander"] details {
  border:1px solid rgba(125,179,238,.33) !important;
  border-radius:12px !important;
  background:rgba(9,30,56,.85) !important;
}
.stApp [data-testid="stExpander"] details > summary {
  background:#163354 !important;
  color:#f3f8ff !important;
  border-radius:11px !important;
}
.stApp [data-testid="stExpander"] details > summary p,
.stApp [data-testid="stExpander"] details > summary span,
.stApp [data-testid="stExpander"] details > summary svg { color:#f3f8ff !important; opacity:1 !important; }
.stApp [data-testid="stExpander"] details[open] > summary {
  border-bottom:1px solid rgba(125,179,238,.28);
  border-radius:11px 11px 0 0 !important;
}
/* Light-valued select, number and text inputs remain readable. */
.stApp [data-testid="stTextInput"] input,
.stApp [data-testid="stNumberInput"] input,
.stApp [data-testid="stTextArea"] textarea,
.stApp [data-testid="stSelectbox"] [data-baseweb="select"] [role="combobox"],
.stApp [data-testid="stSelectbox"] [data-baseweb="select"] input { color:#172e48 !important; }
.stApp [data-testid="stFileUploaderDropzone"] {
  background:rgba(20,49,85,.70) !important;
  border:1px dashed rgba(133,190,248,.55) !important;
}
.stApp [data-testid="stFileUploaderDropzone"] p,
.stApp [data-testid="stFileUploaderDropzone"] small { color:#e3eeff !important; }
.stApp [data-baseweb="tab-list"] [role="tab"] { color:#b9cce4 !important; }
.stApp [data-baseweb="tab-list"] [role="tab"][aria-selected="true"] { color:#ffffff !important; }
.stApp [data-testid="stAlert"] p { opacity:1; }

/* Customer-facing accessibility: minimum contrast for every page and nested widget. */
.stApp [data-testid="stWidgetLabel"] *,
.stApp [data-testid="stMarkdownContainer"] > p,
.stApp [data-testid="stMarkdownContainer"] > ul,
.stApp [data-testid="stMarkdownContainer"] > ol { color:#e7f0fc; }
.stApp [data-testid="stExpander"] details > summary,
.stApp [data-testid="stExpander"] details > summary * {color:#f2f8ff !important;}
.stApp [data-testid="stExpander"] details[open] {background:#102843 !important;}
.stApp [data-testid="stExpander"] details[open] div[data-testid="stMarkdownContainer"] > p {color:#e7f0fc !important;}
.stApp [data-baseweb="tab-list"] { gap:4px; border-bottom:1px solid #35577d; }
.stApp [data-baseweb="tab-list"] [role="tab"] { color:#dce9fa !important; font-weight:650; min-height:44px; border-radius:8px 8px 0 0; }
.stApp [data-baseweb="tab-list"] [role="tab"][aria-selected="true"] {color:#ffffff !important; background:#17456e !important; border-bottom:3px solid #5bc4ff;}
.stApp [data-testid="stAlert"] {border-radius:12px;}
.stApp [data-testid="stAlert"] p, .stApp [data-testid="stAlert"] span {color:inherit !important; opacity:1 !important;}
.stApp [data-testid="stSelectbox"] [data-baseweb="select"],
.stApp [data-testid="stMultiSelect"] [data-baseweb="select"],
.stApp [data-testid="stTextInput"] input,
.stApp [data-testid="stNumberInput"] input,
.stApp [data-testid="stTextArea"] textarea,
.stApp [data-testid="stDateInput"] input {background:#f5f8fc !important; color:#152b44 !important; border-radius:9px;}
.stApp [data-baseweb="select"] [role="combobox"] *,
.stApp [data-baseweb="select"] [role="combobox"] input {color:#152b44 !important;}
.stApp [data-testid="stWidgetLabel"] {font-weight:650; margin-bottom:2px;}
.stApp button:focus-visible, .stApp input:focus-visible, .stApp a:focus-visible {outline:3px solid #70c7ff !important; outline-offset:2px;}
.stApp [data-testid="stToast"] {color:#f6faff;}
.stApp [data-testid="stMetricLabel"] *, .stApp [data-testid="stMetricValue"] * {color:inherit !important;}
@media (max-width:768px) { .block-container {padding:1rem .7rem 2rem;} .ef-hero{padding:18px;} .ef-hero h1{font-size:2.2rem;} .ef-brand-row{margin-bottom:6px;} }
/* Alerts and help copy must be readable against this dark custom shell. */
.stApp [data-testid="stAlert"] { background:#14304d !important; border:1px solid #42688e !important; }
.stApp [data-testid="stAlert"] p,
.stApp [data-testid="stAlert"] span,
.stApp [data-testid="stAlert"] li,
.stApp [data-testid="stAlert"] div[data-testid="stMarkdownContainer"] { color:#eef6ff !important; }
.stApp [data-testid="stAlert"] svg { color:#f8cf77 !important; }
.stApp [data-testid="stSidebar"] [data-testid="stExpander"] summary {color:#f3f7ff !important;}

/* Streamlit info/warning banners use the dark shell color scheme too. */
.stApp [data-testid="stAlert"] {
  background:rgba(18,47,80,.95) !important;
  border:1px solid rgba(129,188,243,.35) !important;
  padding:9px 12px !important;
}
.stApp [data-testid="stAlert"] p,
.stApp [data-testid="stAlert"] span {color:#eaf4ff !important;}
/* Smaller headings keep the working chart visible above the fold. */
.stApp h1 {font-size:clamp(1.6rem,2.4vw,2.1rem) !important; line-height:1.2 !important;}
.stApp h2 {font-size:1.26rem !important; margin-top:.65rem !important;}
/* Compact operational layout: space goes to charts, not repeated state tiles. */
.stApp [data-testid="stMetric"] {
  min-height: 71px;
  border-radius: 10px;
  padding: 8px 12px !important;
}
.stApp [data-testid="stMetricLabel"] p {font-size:.75rem !important; line-height:1.18 !important;}
.stApp [data-testid="stMetricValue"] {line-height:1.18 !important;}
.stApp [data-testid="stMetricValue"] div {font-size:clamp(1.1rem,1.7vw,1.55rem) !important; line-height:1.22 !important;}
.ef-status-strip {display:flex; flex-wrap:wrap; align-items:center; gap:7px 10px; margin:6px 0 12px;}
.ef-status-chip {display:inline-flex; align-items:center; gap:6px; border:1px solid rgba(126,169,222,.26); background:rgba(12,33,57,.85); color:#e7f0fc; padding:5px 9px; border-radius:8px; font-size:.79rem; line-height:1.2;}
.ef-status-chip .label {color:#b9cce4;}
.ef-status-chip .value {font-weight:700; color:#edf6ff;}
.ef-status-chip.good .value {color:#7ce0ba;}
.ef-status-chip.bad .value {color:#ff9e9e;}
.ef-status-chip.warn .value {color:#ffd08a;}
.ef-status-chip .dot {font-size:.8rem;}
.ef-section-help {cursor:help; font-size:.72rem; border:1px solid #577395; color:#b3cae7; border-radius:999px; padding:0 6px; vertical-align:middle; margin-left:6px; font-weight:500;}
.ef-hero {padding:19px 22px 16px; border-radius:16px;}
.ef-hero h1 {font-size:clamp(1.75rem,3vw,2.6rem); margin-bottom:8px;}
.ef-hero p {font-size:.88rem; line-height:1.42;}
.ef-eyebrow {margin-bottom:7px; padding:4px 9px;}
.ef-hero-visual {min-height:140px;}
.ef-wave {height:62px;}
.ef-card {min-height:135px; padding:13px;}
.ef-card .icon {height:28px; width:28px; margin-bottom:8px; font-size:1rem;}
@media(max-width:768px) { .ef-status-chip {font-size:.75rem;} .ef-hero {padding:15px;} }


/* Global high-contrast labels and Plotly titles for the dark dashboard shell. */
.stApp [data-testid="stMarkdownContainer"] h1,
.stApp [data-testid="stMarkdownContainer"] h2,
.stApp [data-testid="stMarkdownContainer"] h3,
.stApp [data-testid="stMarkdownContainer"] h4,
.stApp [data-testid="stMarkdownContainer"] h5,
.stApp [data-testid="stMarkdownContainer"] h6,
.stApp [data-testid="stWidgetLabel"] p,
.stApp [data-testid="stMetricLabel"] p,
.stApp [data-testid="stCaptionContainer"] p {color:#e9f3ff !important; opacity:1 !important;}
.stApp [data-testid="stPlotlyChart"] .main-svg text {fill:#e8f2ff !important;}
.stApp [data-testid="stPlotlyChart"] .gtitle {fill:#f3f8ff !important;}
.stApp [data-testid="stPlotlyChart"] .xtitle,
.stApp [data-testid="stPlotlyChart"] .ytitle {fill:#d8e8ff !important;}
.stApp [data-testid="stForm"] {border-color:#35557a !important;}
.stApp [data-testid="stToggle"] label p,
.stApp [data-testid="stRadio"] label p,
.stApp [data-testid="stCheckbox"] label p {color:#e9f3ff !important;}
</style>
"""


def setup_page(page_title: str, active_page: str) -> str:
    """Configure a page, render the shared top navigation, return API base URL."""
    st.set_page_config(
        page_title=f"{page_title} · E-DF Platform",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(CSS, unsafe_allow_html=True)

    brand_col, health_col = st.columns([4.4, 1.1], vertical_alignment="center")
    with brand_col:
        st.markdown(
            '<div class="ef-brand-row"><div class="ef-brand">'
            '<span class="ef-bolt">ϟ</span><span>E-DF Platform'
            '<small>Electricity Demand Forecasting</small></span></div></div>',
            unsafe_allow_html=True,
        )
    with health_col:
        st.markdown('<div class="ef-health">●&nbsp; Analytics Workspace</div>', unsafe_allow_html=True)

    page_for_section = {
        "home": "app.py",
        "stations": "pages/station_management.py",
        "models": "pages/model_management.py",
        "pipeline": "pages/pipeline_control.py",
        "monitoring": "pages/monitoring.py",
    }
    current_page = page_for_section.get(active_page)
    nav_cols = st.columns([0.8, 1.35, 1.35, 1.15, 1.15], gap="small")
    for col, (label, page, icon) in zip(nav_cols, PAGES):
        with col:
            try:
                active_label = f"{label} ·" if page == current_page else label
                st.page_link(page, label=active_label, icon=icon, use_container_width=True)
            except Exception:
                # Keep the page usable on Streamlit versions without page_link.
                if st.button(f"{icon} {label}", key=f"ef_nav_{active_page}_{label}", use_container_width=True):
                    if hasattr(st, "switch_page"):
                        st.switch_page(page)

    # Keep operator settings out of the default dashboard view. Streamlit's
    # standard sidebar toggle exposes this area only when the user opens it.
    with st.sidebar:
        with st.expander("⚙️ Workspace settings", expanded=False):
            api_url = st.text_input("FastAPI URL", value=API_DEFAULT, key="ef_api_url")
            st.caption("Advanced setting · Backend address shared by all pages")

            if st.checkbox('Edit residual spike thresholds',value=False,
                           key='ef_edit_residual_thresholds'):
                _residual_spike_calibration((api_url or API_DEFAULT).rstrip('/'))

    return (api_url or API_DEFAULT).rstrip("/")


def section_title(title: str, subtitle: str | None = None) -> None:
    """Keep guidance accessible on hover without permanently consuming a full row."""
    help_icon = (
        f'<span class="ef-section-help" title="{escape(subtitle, quote=True)}" '
        f'aria-label="More about {escape(title, quote=True)}">?</span>'
        if subtitle else ""
    )
    st.markdown(
        f'<div class="ef-section-title">{escape(title)}{help_icon}</div>',
        unsafe_allow_html=True,
    )


def render_status_strip(items: list[tuple[str, str, str]]) -> None:
    """Small, readable status indicators. Tone: good, warn, bad or neutral."""
    allowed = {"good", "warn", "bad", "neutral"}
    parts = []
    for label, value, tone in items:
        safe_tone = tone if tone in allowed else "neutral"
        dot = "●" if safe_tone in {"good", "warn", "bad"} else "·"
        parts.append(
            f'<span class="ef-status-chip {safe_tone}">'
            f'<span class="dot" aria-hidden="true">{dot}</span>'
            f'<span class="label">{escape(str(label))}</span>'
            f'<span class="value">{escape(str(value))}</span>'
            f'</span>'
        )
    st.markdown('<div class="ef-status-strip">' + "".join(parts) + '</div>', unsafe_allow_html=True)


def _residual_spike_calibration(api_url: str) -> None:
    """Edit station-specific thresholds only on demand, not at every page refresh."""
    try:
        response=requests.get(f'{api_url}/stations',timeout=(2,8))
        response.raise_for_status()
        stations=response.json().get('stations') or {}
        if not stations:
            st.info('Register a station before editing residual thresholds.')
            return
        station=st.selectbox('Station for spike calibration',sorted(stations),
                             key='ef_residual_setting_station')
        result=requests.get(f'{api_url}/stations/{station}/residual-settings',timeout=(2,8))
        result.raise_for_status()
        current=result.json().get('settings') or {}
    except (requests.RequestException,ValueError) as exc:
        st.warning(f'Cannot load station spike thresholds: {exc}')
        return
    with st.form(f'ef_spike_thresholds_{station}'):
        rule=st.selectbox('Detector agreement', ['both','either'],
                          index=0 if current.get('decision_rule','both')=='both' else 1,
                          help='Both: statistical and autoencoder scores must exceed thresholds. Either: one score is enough.')
        stat=st.number_input('Statistical score threshold (0 = model calibration)',
                             min_value=0.,max_value=10000.,step=.1,
                             value=float(current.get('stat_threshold') or 0.))
        ae=st.number_input('Autoencoder score threshold (0 = model calibration)',
                           min_value=0.,max_value=10000.,step=.1,
                           value=float(current.get('ae_threshold') or 0.))
        up=st.checkbox('Highlight upward spikes only',
                       value=bool(current.get('upward_only',True)))
        save=st.form_submit_button('Save spike thresholds',use_container_width=True)
    if save:
        try:
            response=requests.put(f'{api_url}/stations/{station}/residual-settings',
                                  json={'decision_rule':rule,'stat_threshold':stat or None,
                                        'ae_threshold':ae or None,'upward_only':up},timeout=(2,12))
            response.raise_for_status()
            st.success(f'Spike thresholds saved for {station}. New forecasts use this calibration.')
        except requests.RequestException as exc:
            st.error(f'Could not save calibration: {exc}')
