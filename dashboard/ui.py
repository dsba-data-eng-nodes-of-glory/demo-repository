"""Shared visual system and navigation for the Electricity Forecasting POC."""
from __future__ import annotations

import streamlit as st

API_DEFAULT = "http://127.0.0.1:8000"

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
</style>
"""


def setup_page(page_title: str, active_page: str) -> str:
    """Configure a page, render the shared top navigation, return API base URL."""
    st.set_page_config(
        page_title=f"{page_title} · E-DF Platform",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
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

    nav_cols = st.columns([0.8, 1.35, 1.35, 1.15, 1.15], gap="small")
    for col, (label, page, icon) in zip(nav_cols, PAGES):
        with col:
            try:
                st.page_link(page, label=label, icon=icon, use_container_width=True)
            except Exception:
                # Keep the page usable on Streamlit versions without page_link.
                if st.button(f"{icon} {label}", key=f"ef_nav_{active_page}_{label}", use_container_width=True):
                    if hasattr(st, "switch_page"):
                        st.switch_page(page)

    with st.sidebar:
        st.markdown("### Workspace settings")
        api_url = st.text_input("FastAPI URL", value=API_DEFAULT, key="ef_api_url")
        st.caption("Backend connection used by all modules.")
        st.divider()
        st.markdown("**Platform modules**")
        st.caption("Home · Stations · Models · Pipeline · Monitoring")
        st.markdown("---")
        st.caption("Local POC · Kafka + Spark + LSTM")

    return (api_url or API_DEFAULT).rstrip("/")


def section_title(title: str, subtitle: str | None = None) -> None:
    st.markdown(f'<div class="ef-section-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="ef-muted">{subtitle}</div>', unsafe_allow_html=True)
