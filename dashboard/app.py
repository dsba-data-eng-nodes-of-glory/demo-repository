"""Professional landing page for the Electricity Demand Forecasting Platform."""
from __future__ import annotations

import html
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

try:
    from dashboard.ui import setup_page, section_title
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from dashboard.ui import setup_page, section_title

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
    registry = api_get("/stations")
    pipeline = api_get("/pipeline/status")
    backend_ok = True
    for station_id in registry.get("stations", {}):
        try:
            station_statuses[station_id] = api_get(f"/stations/{station_id}/status", timeout=5)
        except requests.RequestException:
            station_statuses[station_id] = {}
        try:
            model_statuses[station_id] = api_get("/models/status", {"station": station_id}, timeout=5)
        except requests.RequestException:
            model_statuses[station_id] = {}
except requests.RequestException as exc:
    backend_error = str(exc)

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
model_count = sum(1 for model in model_statuses.values() if model.get("checkpoint_exists"))
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
        '<p>A scalable, end-to-end workspace for multi-station electricity demand forecasting. '
        'Connect data ingestion, Kafka streaming, Spark feature engineering and station-specific LSTM models.</p>'
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
k1, k2, k3, k4 = st.columns(4)
k1.metric("Registered stations", len(station_ids) if backend_ok else "—")
k2.metric("Active stations", active_count if backend_ok else "—")
k3.metric("Models available", model_count if backend_ok else "—")
k4.metric("Running services", running_services if backend_ok else "—")

if not backend_ok:
    st.error(f"FastAPI is not reachable at {api}. Start the backend to load live platform information.\n\n{backend_error}")

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
                st.caption(f"Station ID: `{station_id}`")
                m1, m2, m3 = st.columns(3)
                m1.markdown(
                    f'<div style="border:1px solid {state_color}55;background:{state_color}12;'
                    f'border-radius:12px;padding:10px 12px;min-height:86px;">'
                    f'<div style="font-size:13px;color:#9db3cd;">State</div>'
                    f'<div style="font-size:21px;line-height:1.35;font-weight:700;color:{state_color};">'
                    f'{state_label.title()}</div></div>',
                    unsafe_allow_html=True,
                )
                m2.metric("Model", "Ready" if model_ready else "Not trained")
                m3.metric("Kafka topics", "Ready" if topics_ready else "Check")
                st.caption(
                    f"Raw: `{entry.get('raw_topic', station_id + '-raw')}`  ·  "
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
                    if st.button("Stop + reset", key=f"home_stop_{station_id}", disabled=not is_active, use_container_width=True, help="Stops this station and automatically clears its Kafka records, Spark checkpoint and station result files. Source CSVs and trained model are retained."):
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
section_title("End-to-end workflow", "A single operational path from incoming records to decision-ready forecasts.")
flow = [
    ("▤", "Data ingestion", "CSV source"),
    ("◉", "Kafka", "Event streaming"),
    ("✳", "Spark", "Feature engineering"),
    ("⌁", "LSTM", "24-hour forecast"),
    ("▣", "Dashboard", "Monitoring & analysis"),
]
flow_cols = st.columns(5, gap="small")
for col, (icon, title, sub) in zip(flow_cols, flow):
    with col:
        st.markdown(f'<div class="ef-flow"><div style="font-size:1.35rem;margin-bottom:5px">{icon}</div><b>{title}</b><span>{sub}</span></div>', unsafe_allow_html=True)

st.markdown('<div style="text-align:center;color:#7088a8;font-size:.74rem;padding:24px 0 0">E-DF Platform · Local research and demonstration workspace</div>', unsafe_allow_html=True)
