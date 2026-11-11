"""Professional landing page for the Electricity Demand Forecasting Platform."""
from __future__ import annotations

import html
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

try:
    from dashboard.ui import setup_page, section_title, render_status_strip
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from dashboard.ui import setup_page, section_title, render_status_strip

api = setup_page("Platform Overview", "home")


def api_get(path: str, params: dict | None = None, timeout: int = 10):
    response = requests.get(f"{api}{path}", params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


def api_post(path: str, timeout: int = 15):
    response = requests.post(f"{api}{path}", timeout=timeout)
    response.raise_for_status()
    return response.json()


# ---------------------------------------------------------------------
# Backend health and station/model inventory
# ---------------------------------------------------------------------
backend_ok = False
registry: dict = {}
pipeline: dict = {}
station_statuses: dict[str, dict] = {}
model_statuses: dict[str, dict] = {}
backend_error = None

try:
    # One batched endpoint replaces 2N+2 requests for N stations. Fall back to
    # the old endpoints for compatibility with users updating only the UI.
    try:
        overview = api_get("/stations/overview", timeout=12)
        registry = {"stations": overview.get("stations", {}), "active_station": overview.get("active_station")}
        station_statuses = overview.get("statuses", {})
        station_components = overview.get("components", {})
        pipeline = {"kafka_tcp_available": overview.get("kafka_tcp_available")}
        model_statuses = {
            key: {"checkpoint_exists": details.get("model_ready", False)}
            for key, details in station_statuses.items()
        }
    except requests.HTTPError as missing_endpoint:
        if missing_endpoint.response is None or missing_endpoint.response.status_code != 404:
            raise
        registry = api_get("/stations")
        pipeline = api_get("/pipeline/status")
        station_components = {}
        for station_id in registry.get("stations", {}):
            try:
                station_statuses[station_id] = api_get(f"/stations/{station_id}/status", timeout=5)
                model_statuses[station_id] = api_get("/models/status", {"station": station_id}, timeout=5)
            except requests.RequestException:
                station_statuses[station_id] = {}
                model_statuses[station_id] = {}
    backend_ok = True
except requests.RequestException as exc:
    backend_error = str(exc)
    station_components = {}

stations = registry.get("stations", {}) if backend_ok else {}
station_ids = list(stations.keys())
active_station = registry.get("active_station") if backend_ok else None
if active_station not in station_ids and station_ids:
    active_station = station_ids[0]

kafka_ok = bool(pipeline.get("kafka_tcp_available")) if backend_ok else False
running_services = sum(
    1
    for status in station_statuses.values()
    for service in status.get("services", {}).values()
    if service.get("running")
)
model_count = sum(
    sum(bool(component.get("checkpoint_exists")) for component in parts.values())
    for parts in station_components.values()
) if station_components else sum(1 for model in model_statuses.values() if model.get("checkpoint_exists"))
active_count = sum(
    1 for status in station_statuses.values()
    if status.get("lifecycle_status") == "active"
)

# ---------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------
hero_left, hero_right = st.columns([1.4, 1], gap="large", vertical_alignment="center")
with hero_left:
    st.markdown(
        '<div class="ef-hero">'
        '<div class="ef-eyebrow">● &nbsp; DATA ENGINEERING&nbsp; · &nbsp; STREAMING ANALYTICS</div>'
        '<h1>Electricity Demand<br><span class="ef-gradient">Forecasting Platform</span></h1>'
        '<p>Configure stations, compare trained models and monitor demand forecasts. '
        'Review forecast spike risk and renewable generation before operational decisions.</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    cta1, cta2 = st.columns([1, 1.2])
    with cta1:
        st.page_link("pages/monitoring.py", label="Explore monitoring", icon="📈", use_container_width=True)
    with cta2:
        st.page_link("pages/station_management.py", label="Manage stations", icon="⚡", use_container_width=True)

with hero_right:
    station_label = html.escape(str(stations.get(active_station, {}).get("display_name", active_station))) if active_station else "Awaiting connection"
    badge_text = "Kafka connected" if kafka_ok else "Kafka unavailable"
    badge_color = "#61e4b1" if kafka_ok else "#ffbd73"
    connection_text = f"● {badge_text}" if backend_ok else "● Backend not reachable"
    st.markdown(
        f'<div class="ef-chart-panel">'
        f'<div class="small">ACTIVE STATION</div>'
        f'<div class="big">{station_label}</div>'
        '<svg class="ef-wave" viewBox="0 0 520 112" preserveAspectRatio="none" aria-label="Electricity signal">'
        '<defs><linearGradient id="efg" x1="0" x2="1"><stop offset="0" stop-color="#25baff"/><stop offset="1" stop-color="#9b7cff"/></linearGradient></defs>'
        '<path d="M0 78 C45 70 55 80 85 63 S130 22 158 52 S200 91 232 70 S280 28 310 48 S355 94 392 63 S455 38 520 45" fill="none" stroke="url(#efg)" stroke-width="3.5" stroke-linecap="round"/>'
        '<path d="M0 95 C65 88 93 94 130 80 S190 64 220 82 S300 100 340 78 S430 65 520 82" fill="none" stroke="#4976ef" stroke-opacity=".45" stroke-width="2"/>'
        '<g fill="#54caff"><circle cx="158" cy="52" r="4"/><circle cx="310" cy="48" r="4"/><circle cx="392" cy="63" r="4"/></g>'
        '</svg>'
        f'<div class="small" style="color:{badge_color}">{connection_text}</div>'
        '</div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------
section_title("Live platform overview", "A quick view of registered stations, deployed models and running services.")
@st.fragment(run_every=2.0)
def render_live_overview_metrics() -> None:
    # One batched call per refresh; do not rerender station controls/cards.
    try:
        current = api_get("/stations/overview", timeout=5)
    except requests.RequestException:
        # Initial values remain useful if the endpoint temporarily stalls or
        # a legacy backend does not support the batched endpoint.
        current = None
    if current is None:
        counts = (len(station_ids) if backend_ok else "—",
                  active_count if backend_ok else "—",
                  model_count if backend_ok else "—",
                  running_services if backend_ok else "—")
    else:
        current_stations = current.get("stations") or {}
        current_statuses = current.get("statuses") or {}
        current_components = current.get("components") or {}
        counts = (
            len(current_stations),
            sum(1 for item in current_statuses.values() if item.get("lifecycle_status") == "active"),
            (sum(sum(bool(comp.get("checkpoint_exists")) for comp in comps.values())
                 for comps in current_components.values())
             if current_components else sum(bool(item.get("model_ready"))
                                            for item in current_statuses.values())),
            sum(1 for item in current_statuses.values()
                for service in (item.get("services") or {}).values() if service.get("running")),
        )
    k1, k2, k3, k4 = st.columns(4)
    for column, label, value in zip((k1, k2, k3, k4),
                                   ("Registered stations", "Active stations", "Deployed model components", "Running services"), counts):
        column.metric(label, value)


render_live_overview_metrics()

if not backend_ok:
    st.error(f"FastAPI is not reachable at {api}. Start the backend to load live platform information.\n\n{backend_error}")

with st.expander("New station? Quick setup", expanded=not bool(station_ids)):
    quick1, quick2, quick3 = st.columns(3, gap="small")
    with quick1:
        st.markdown("**1. Add station**")
        st.page_link("pages/station_management.py", label="Register", icon="📡", use_container_width=True)
    with quick2:
        st.markdown("**2. Train model**")
        st.page_link("pages/model_management.py", label="Model Lab", icon="🧠", use_container_width=True)
    with quick3:
        st.markdown("**3. Monitor**")
        st.page_link("pages/pipeline_control.py", label="Pipeline", icon="⚙️", use_container_width=True)

# ---------------------------------------------------------------------
# Registered station cards and independent lifecycle controls
# ---------------------------------------------------------------------
section_title(
    "Registered stations",
    "Each registered station has independent activation, stop/reset and monitoring controls. Kafka topics are created automatically during registration and retried during activation. Multiple stations can run simultaneously.",
)
if backend_ok and station_ids:
    card_columns = st.columns(2, gap="large")
    for index, station_id in enumerate(station_ids):
        entry = stations.get(station_id, {})
        status = station_statuses.get(station_id, {})
        services = status.get("services", {})
        is_active = status.get("lifecycle_status") == "active" or any(
            services.get(name, {}).get("running", False)
            for name in ("spark", "predictor", "producer")
        )
        model_ready = bool(model_statuses.get(station_id, {}).get("checkpoint_exists"))
        topics_ready = bool(entry.get("topics_ready"))
        with card_columns[index % 2]:
            with st.container(border=True):
                display_name = html.escape(str(entry.get("display_name", station_id)))
                training_running = bool(services.get("training", {}).get("running"))
                if is_active:
                    state_color, state_label = "#22c55e", "ACTIVE"
                elif training_running:
                    state_color, state_label = "#fbbf24", "TRAINING"
                else:
                    state_color, state_label = "#f87171", "STOPPED"
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:10px;margin:0 0 4px 0;">'
                    f'<h3 style="margin:0;color:{state_color};font-weight:750;">{display_name}</h3>'
                    f'<span style="font-size:10px;font-weight:700;letter-spacing:.08em;'
                    f'color:{state_color};border:1px solid {state_color}66;'
                    f'background:{state_color}18;border-radius:999px;padding:3px 8px;">{state_label}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                render_status_strip([
                    ("Demand", "Ready" if model_ready else "Needs training", "good" if model_ready else "warn"),
                    ("Kafka topics", "Ready" if topics_ready else "Check", "good" if topics_ready else "warn"),
                ])
                with st.expander("Station details", expanded=False):
                    st.code(station_id, language="text")
                    st.caption(
                        f"Raw: `{entry.get('raw_topic', station_id + '-raw')}` · "
                        f"Featured: `{entry.get('featured_topic', station_id + '-featured')}`"
                    )
                b1, b2, b3 = st.columns(3)
                with b1:
                    if st.button("Activate", key=f"home_activate_{station_id}", type="primary", disabled=(is_active or not model_ready), use_container_width=True, help="Train and deploy a model from Model Management before activation." if not model_ready else "Activate this station independently."):
                        try:
                            response = requests.post(f"{api}/stations/{station_id}/activate", timeout=120)
                            response.raise_for_status()
                            st.success("Station activated.")
                            st.rerun()
                        except requests.RequestException as exc:
                            detail = exc.response.text if getattr(exc, "response", None) is not None else str(exc)
                            st.error(f"Activation failed: {detail}")
                with b2:
                    with st.popover("Stop / reset", use_container_width=True, disabled=not is_active):
                        st.warning("This clears this station's Kafka replay records, Spark checkpoint and forecast results. Trained models and source data are retained.")
                        confirm_stop = st.checkbox("I understand the station's replay results will be cleared", key=f"home_confirm_stop_{station_id}")
                        stop_requested = st.button("Confirm stop and reset", key=f"home_stop_{station_id}", disabled=not confirm_stop, use_container_width=True)
                    if is_active and stop_requested:
                        try:
                            response = requests.post(f"{api}/stations/{station_id}/stop", timeout=300)
                            response.raise_for_status()
                            st.success("Station stopped and reset. Other active stations are unaffected.")
                            st.rerun()
                        except requests.RequestException as exc:
                            detail = exc.response.text if getattr(exc, "response", None) is not None else str(exc)
                            st.error(f"Stop/reset failed: {detail}")
                with b3:
                    st.page_link(
                        "pages/monitoring.py",
                        label="Monitor",
                        icon="📈",
                        use_container_width=True,
                        query_params={"station": station_id},
                    )
elif backend_ok:
    st.info("No stations are registered yet. Open Station Management to add your first station.")

# ---------------------------------------------------------------------
# Module cards
# ---------------------------------------------------------------------
section_title("Your workspace", "Each module is connected to the same station registry and backend API.")
modules = [
    ("◉", "Station Management", "Register stations, configure timestamp handling, inspect station data and manage station deletion.", "pages/station_management.py", "Open station management", "#168ee2"),
    ("▱", "Model Management", "Review model readiness, training metrics, model versions and checkpoint metadata.", "pages/model_management.py", "Open model management", "#9a5cf1"),
    ("▷", "Pipeline Control", "Start and stop station services independently and inspect Kafka, Spark, predictor and producer status.", "pages/pipeline_control.py", "Open pipeline control", "#18b88b"),
    ("⌁", "Live Monitoring", "Explore actual demand, the latest forecast from the selected model and forecast history for each station.", "pages/monitoring.py", "Open monitoring", "#e89a43"),
]
card_cols = st.columns(4, gap="medium")
for col, (icon, title, description, page, link_label, accent) in zip(card_cols, modules):
    with col:
        st.markdown(
            f'<div class="ef-card"><div class="icon" style="border-color:{accent}55;background:{accent}1a">{icon}</div>'
            f'<h3>{title}</h3><p>{description}</p></div>',
            unsafe_allow_html=True,
        )
        st.page_link(page, label=link_label, icon="➡️", use_container_width=True)

# ---------------------------------------------------------------------
# Current station and forecast preview
# ---------------------------------------------------------------------
section_title("Demand signal preview", "Shows saved actuals and the latest forecast run when those records are available.")
if backend_ok and active_station:
    try:
        active_entry = stations.get(active_station, {})
        active_horizon_steps = int(active_entry.get("horizon_steps", 96))
        actual_payload = api_get(
            "/results/actuals", {"station": active_station, "limit": active_horizon_steps}
        )
        forecast_payload = api_get("/results/forecasts", {"station": active_station, "limit": 2000})
        actuals = pd.DataFrame(actual_payload.get("records", []))
        forecasts = pd.DataFrame(forecast_payload.get("records", []))

        fig = go.Figure()
        if not actuals.empty and {"utc_timestamp", "actual_demand"}.issubset(actuals.columns):
            actuals["utc_timestamp"] = pd.to_datetime(actuals["utc_timestamp"], utc=True, errors="coerce")
            actuals["actual_demand"] = pd.to_numeric(actuals["actual_demand"], errors="coerce")
            actuals = actuals.dropna(subset=["utc_timestamp", "actual_demand"]).sort_values("utc_timestamp")
            fig.add_trace(go.Scatter(x=actuals["utc_timestamp"], y=actuals["actual_demand"], name="Actual demand", mode="lines", line=dict(width=2.5, color="#38bdf8")))

        if not forecasts.empty and {"forecast_timestamp_utc", "predicted_demand"}.issubset(forecasts.columns):
            forecasts["forecast_timestamp_utc"] = pd.to_datetime(forecasts["forecast_timestamp_utc"], utc=True, errors="coerce")
            forecasts["predicted_demand"] = pd.to_numeric(forecasts["predicted_demand"], errors="coerce")
            if "forecast_issued_at_utc" in forecasts.columns:
                forecasts["forecast_issued_at_utc"] = pd.to_datetime(forecasts["forecast_issued_at_utc"], utc=True, errors="coerce")
                latest_issue = forecasts["forecast_issued_at_utc"].max()
                forecasts = forecasts[forecasts["forecast_issued_at_utc"] == latest_issue]
            forecasts = forecasts.dropna(subset=["forecast_timestamp_utc", "predicted_demand"]).sort_values("forecast_timestamp_utc")
            fig.add_trace(go.Scatter(x=forecasts["forecast_timestamp_utc"], y=forecasts["predicted_demand"], name="Latest forecast", mode="lines", line=dict(width=2.5, dash="dash", color="#a78bfa")))

        if len(fig.data):
            fig.update_layout(
                template="plotly_dark", height=340, margin=dict(l=12, r=12, t=18, b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(7,20,39,.55)",
                legend=dict(orientation="h", y=1.12, x=0),
                xaxis_title="Timestamp (UTC)", yaxis_title="Demand (MW)",
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("No actual or forecast records are available yet for this station. Start the station pipeline to populate the chart.")
    except requests.RequestException as exc:
        st.warning(f"Forecast preview is temporarily unavailable: {exc}")
else:
    st.info("Register or connect a station to see live demand and forecast data here.")

# ---------------------------------------------------------------------
# Architecture strip
# ---------------------------------------------------------------------
with st.expander("How the platform works", expanded=False):
    flow = [
        ("▤", "Data ingestion", "CSV source"),
        ("◉", "Kafka", "Event streaming"),
        ("✳", "Spark", "Feature engineering"),
        ("⌁", "Models", "Forecasts & risk"),
        ("▣", "Dashboard", "Monitoring & analysis"),
    ]
    flow_cols = st.columns(5, gap="small")
    for col, (icon, title, sub) in zip(flow_cols, flow):
        with col:
            st.markdown(f'<div class="ef-flow"><div style="font-size:1.35rem;margin-bottom:5px">{icon}</div><b>{title}</b><span>{sub}</span></div>', unsafe_allow_html=True)

st.markdown('<div style="text-align:center;color:#7088a8;font-size:.74rem;padding:24px 0 0">E-DF Platform · Forecasting analytics</div>', unsafe_allow_html=True)
