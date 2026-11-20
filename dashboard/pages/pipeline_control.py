"""Station-scoped lifecycle controls for Kafka, Spark, predictor and producer."""
from __future__ import annotations

import requests
import streamlit as st

try:
    from dashboard.ui import setup_page, section_title, render_status_strip
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from dashboard.ui import setup_page, section_title, render_status_strip

from src.storage.station_hierarchy import station_label

api = setup_page("Pipeline Control", "pipeline")


def get_json(path: str, params: dict | None = None):
    response = requests.get(f"{api}{path}", params=params, timeout=(3, 30))
    response.raise_for_status()
    return response.json()


def post_json(path: str, payload: dict | None = None, timeout: int = 60):
    response = requests.post(f"{api}{path}", json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=5, show_spinner=False)
def probe_cassandra() -> dict:
    """Read the pipeline storage file when the API status has no Cassandra field.

    Streamlit may be started with STORAGE_BACKEND=csv. That override applies
    only to this dashboard process. Station predictors follow config/storage.yaml.
    """
    import socket
    from pathlib import Path

    import yaml

    path = Path(__file__).resolve().parents[2] / "config" / "storage.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    backend = str(cfg.get("backend") or "csv").strip().lower()
    status = {"backend": backend, "available": False, "host": None, "port": None}
    if backend != "cassandra":
        return status
    cass = cfg.get("cassandra") or {}
    hosts = cass.get("hosts") or ["127.0.0.1"]
    host = str(hosts[0])
    port = int(cass.get("port") or 9042)
    status["host"] = host
    status["port"] = port
    try:
        with socket.create_connection((host, port), timeout=1):
            status["available"] = True
    except OSError:
        status["available"] = False
    return status


try:
    registry = get_json("/stations")
except requests.RequestException as exc:
    st.error(f"Cannot connect to FastAPI at {api}: {exc}")
    st.info(
        "Check the FastAPI terminal and open http://127.0.0.1:8000/health/live. "
        "If that URL also times out, the API process is blocked or stopped; restarting Streamlit alone will not fix it."
    )
    st.stop()

stations = registry.get("stations", {})
station_ids = list(stations)
if not station_ids:
    st.info("No stations are registered. Register a station before starting pipeline services.")
    st.page_link("pages/station_management.py", label="Open Station Management", icon="📡")
    st.stop()

active = registry.get("active_station")
index = station_ids.index(active) if active in station_ids else 0
station = st.selectbox(
    "Station",
    station_ids,
    index=index,
    format_func=lambda key: station_label(key, stations[key]),
)
entry = stations[station]

try:
    status = get_json(f"/stations/{station}/status")
except requests.RequestException as exc:
    st.error(f"Could not load station status: {exc}")
    st.stop()

section_title(
    "Pipeline operations",
    "Check service readiness before starting. Use Stop to pause the station. Reset Replay clears its replay position.",
)
# Keep buttons outside the fragment; only the status strip polls FastAPI.
@st.fragment(run_every=2.0)
def render_live_pipeline_status(selected_station: str) -> None:
    try:
        live = get_json(f"/stations/{selected_station}/status")
    except requests.RequestException as exc:
        st.warning(f"Live pipeline status temporarily unavailable: {exc}")
        return
    live_services = live.get("services", {})
    live_kafka = bool(live.get("kafka_tcp_available", False))
    live_cassandra = live.get("cassandra")
    if not isinstance(live_cassandra, dict) or "backend" not in live_cassandra:
        live_cassandra = probe_cassandra()
    cassandra_backend = str(live_cassandra.get("backend") or "csv")
    cassandra_up = bool(live_cassandra.get("available"))
    if cassandra_backend == "cassandra":
        cassandra_chip = ("Cassandra", "Connected" if cassandra_up else "Offline", "good" if cassandra_up else "bad")
    else:
        cassandra_chip = ("Cassandra", "Not used", "neutral")
    live_model = bool(live.get("model_ready", False))
    render_status_strip([
        ("Station", str(live.get("lifecycle_status", "unknown")).replace("_", " ").title(),
         "good" if live.get("lifecycle_status") == "active" else "neutral"),
        ("Kafka", "Connected" if live_kafka else "Offline", "good" if live_kafka else "bad"),
        cassandra_chip,
        ("Demand model", "Ready" if live_model else "Missing", "good" if live_model else "warn"),
        *[(name.title(), item.get("state", "running" if item.get("running") else "stopped").title(),
           "good" if item.get("running") else {"restarting": "warn", "failed": "bad"}.get(item.get("state"), "neutral"))
          for name in ("training", "spark", "predictor", "producer")
          for item in [live_services.get(name, {})]],
    ])
    if not live_kafka:
        st.warning("Kafka is not reachable at 127.0.0.1:9092. Start Kafka before resetting or activating this station.")
    if cassandra_backend == "cassandra" and not cassandra_up:
        cassandra_host = live_cassandra.get("host") or "127.0.0.1"
        cassandra_port = live_cassandra.get("port") or 9042
        st.warning(
            f"Cassandra is not reachable at {cassandra_host}:{cassandra_port}. "
            "Forecasts and actuals stay unavailable until the node is running."
        )
    if live.get("initial_training_pending"):
        st.info("Initial model training is queued or running. Refresh the controls when it finishes.")
    elif not live_model:
        st.warning("The demand model artifact is missing. Reset is still available, but starting the current demand-driven pipeline requires a trained demand model.")


render_live_pipeline_status(station)
services = status.get("services", {})
kafka_ok = status.get("kafka_tcp_available", False)
model_ready = status.get("model_ready", False)

# Show the same model components as Model Management. Legacy station records
# receive inferred component statuses from /models/components.
if st.toggle("Station model components · details", value=False, key=f"pipeline_components_{station}"):
    try:
        component_data = get_json("/models/components", {"station": station})
        details = component_data.get("sources") or {}
        labels = {
            "demand_forecast": "Demand forecasting",
            "solar_forecast": "Solar forecasting",
            "wind_forecast": "Wind forecasting",
            "residual_anomaly": "Residual Load anomaly autoencoder",
        }
        rows = []
        for key, label in labels.items():
            component = details.get(key, {})
            rows.append({
                "Model component": label,
                "Selected": "Yes" if component.get("enabled") else "No",
                "Training dataset": "Available" if component.get("registered_file_exists") else "Not configured",
                "Deployment": "Artifact ready" if component.get("deployment_exists") else "Missing",
                "Checkpoint": component.get("deployment_file") or "—",
            })
        st.dataframe(rows, hide_index=True, use_container_width=True)
        st.caption("Solar/Wind checkpoint readiness is independent of demand. Live prediction is demand-driven; the existing station activation still requires a demand model.")
    except requests.RequestException as exc:
        st.warning(f"Model component details are unavailable: {exc}")

station_running = status.get("lifecycle_status") in {"active", "recovering", "failed"}
start_col, stop_col, refresh_col = st.columns([1.25, 1.1, .8])
with start_col:
    if st.button(
        "▶  Start selected station",
        type="primary",
        use_container_width=True,
        disabled=not model_ready or not kafka_ok or status.get("initial_training_pending", False),
    ):
        try:
            result = post_json(f"/stations/{station}/activate", timeout=90)
            st.success(f"Start request completed for {station}.")
            st.json(result)
        except requests.RequestException as exc:
            detail = exc.response.text if getattr(exc, "response", None) is not None else str(exc)
            st.error(f"Could not start station: {detail}")
with stop_col:
    if st.button("■  Stop selected station", use_container_width=True, disabled=not station_running):
        try:
            post_json(f"/stations/{station}/stop", timeout=120)
            st.success(f"{station} stopped. Activate it to resume.")
            st.rerun()
        except requests.RequestException as exc:
            detail = exc.response.text if getattr(exc, "response", None) is not None else str(exc)
            st.error(f"Could not stop station: {detail}")
with refresh_col:
    if st.button("↻ Refresh", use_container_width=True):
        st.rerun()

with st.expander("Reset replay"):
    confirm_reset = st.checkbox(
        "Clear this station's Kafka replay records, checkpoints and forecast outputs",
        key=f"confirm_pipeline_reset_{station}",
    )
    if st.button("Reset replay from the beginning", disabled=station_running or not confirm_reset):
        try:
            result = post_json("/pipeline/reset", {"station": station, "confirm": True}, timeout=300)
            st.success(f"{station} replay was reset. Start the station to replay from the beginning.")
            st.json(result)
        except requests.RequestException as exc:
            detail = exc.response.text if getattr(exc, "response", None) is not None else str(exc)
            st.error(f"Could not reset replay: {detail}")

with st.expander("Advanced: Kafka topics and service logs", expanded=False):
    section_title(
        "Station Kafka topics",
        "Kafka topics are provisioned automatically during station registration and retried during activation. Reset Replay clears retained replay records and keeps topic definitions.",
    )
    topics = status.get("topics", {})
    t1, t2 = st.columns(2)
    t1.markdown(f"**Raw topic**  \n`{topics.get('raw') or entry.get('raw_topic', '—')}`")
    t2.markdown(f"**Featured topic**  \n`{topics.get('featured') or entry.get('featured_topic', '—')}`")
    section_title("Service logs", "Recent output from the selected station's individual service processes.")
    log_service = st.selectbox("Log stream", ["spark", "predictor", "producer", "training", "live_actuals"], key="pipeline_log_service")
    if st.button("Load latest log lines"):
        try:
            result = get_json(f"/pipeline/logs/{log_service}", {"station": station})
            st.code(result.get("content") or "No log output is available yet.", language="text")
        except requests.RequestException as exc:
            detail = exc.response.text if getattr(exc, "response", None) is not None else str(exc)
            st.error(f"Could not retrieve log: {detail}")

