"""Station-scoped lifecycle controls for Kafka, Spark, predictor and producer."""
from __future__ import annotations

import requests
import streamlit as st

try:
    from dashboard.ui import setup_page, section_title
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from dashboard.ui import setup_page, section_title

api = setup_page("Pipeline Control", "pipeline")


def get_json(path: str, params: dict | None = None):
    response = requests.get(f"{api}{path}", params=params, timeout=12)
    response.raise_for_status()
    return response.json()


def post_json(path: str, payload: dict | None = None, timeout: int = 60):
    response = requests.post(f"{api}{path}", json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


try:
    registry = get_json("/stations")
except requests.RequestException as exc:
    st.error(f"Cannot connect to FastAPI: {exc}")
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
    format_func=lambda key: stations[key].get("display_name", key),
)
entry = stations[station]

try:
    status = get_json(f"/stations/{station}/status")
except requests.RequestException as exc:
    st.error(f"Could not load station status: {exc}")
    st.stop()

section_title(
    "Pipeline operations",
    "Start or stop services for the selected station without affecting other station processes.",
)
services = status.get("services", {})
cols = st.columns(4)
for col, name in zip(cols, ("training", "spark", "predictor", "producer")):
    details = services.get(name, {})
    col.metric(name.title(), "Running" if details.get("running") else "Stopped")

kafka_ok = status.get("kafka_tcp_available", False)
model_ready = status.get("model_ready", False)
a, b, c = st.columns(3)
a.metric("Kafka broker", "Reachable" if kafka_ok else "Unavailable")
b.metric("LSTM checkpoint", "Ready" if model_ready else "Missing")
c.metric("Station state", status.get("lifecycle_status", "unknown").replace("_", " ").title())

if not kafka_ok:
    st.warning("Kafka is not reachable at 127.0.0.1:9092. Start Kafka before resetting or activating this station.")
if not model_ready:
    st.warning("The model checkpoint is missing. Reset is still available, but starting the pipeline requires a trained model.")

station_services_running = any(
    services.get(name, {}).get("running", False)
    for name in ("training", "spark", "predictor", "producer")
)

start_col, stop_col, refresh_col = st.columns([1.25, 1.1, .8])
with start_col:
    if st.button(
        "▶  Start selected station",
        type="primary",
        use_container_width=True,
        disabled=not model_ready or not kafka_ok,
    ):
        try:
            result = post_json(f"/stations/{station}/activate", timeout=90)
            st.success(f"Start request completed for {station}.")
            st.json(result)
        except requests.RequestException as exc:
            detail = exc.response.text if getattr(exc, "response", None) is not None else str(exc)
            st.error(f"Could not start station: {detail}")
with stop_col:
    if st.button("■  Stop selected station", use_container_width=True):
        try:
            result = post_json(f"/stations/{station}/stop", timeout=45)
            st.success(f"Stop request completed for {station}.")
            st.json(result)
        except requests.RequestException as exc:
            detail = exc.response.text if getattr(exc, "response", None) is not None else str(exc)
            st.error(f"Could not stop station: {detail}")
with refresh_col:
    if st.button("↻ Refresh", use_container_width=True):
        st.rerun()

# ---------------------------------------------------------------------
# Station-scoped reset only (does not restart any process)
# ---------------------------------------------------------------------
section_title(
    "Reset selected station",
    "Clear this station's Kafka records, Spark checkpoint and saved result rows. "
    "The pipeline will remain stopped. Start it separately when you are ready.",
)

with st.expander("Reset details and confirmation", expanded=False):
    st.warning(
        f"Reset will clear old records from {station}'s raw and featured Kafka topics, "
        "remove this station's Spark checkpoint, and back up then remove only this station's "
        "rows from actuals.csv and forecasts.csv. It preserves the Kafka topics, trained "
        "model, source CSVs and other stations' output rows. It will NOT start Spark, "
        "predictor or producer."
    )
    st.caption(
        "Kafka must be running. Stop this station first. Reset is blocked if either Kafka "
        "topic is shared with another registered station, has more than one partition, "
        "or another station's predictor is writing to the shared result CSVs."
    )
    st.checkbox(
        f"I understand that {station}'s Kafka records, Spark checkpoint and saved results will be reset.",
        key=f"confirm_reset_ack_{station}",
    )
    typed_station = st.text_input(
        f"Type {station} to confirm",
        key=f"confirm_reset_station_{station}",
        placeholder=station,
    )
    if station_services_running:
        st.info("Stop training, Spark, predictor and producer for this station, then refresh status before resetting.")

    reset_confirmed = (
        st.session_state.get(f"confirm_reset_ack_{station}", False)
        and typed_station.strip() == station
    )
    if st.button(
        f"Reset {station} (do not start pipeline)",
        type="secondary",
        use_container_width=True,
        disabled=(not reset_confirmed or station_services_running or not kafka_ok),
    ):
        try:
            with st.spinner(f"Resetting {station}. Services will remain stopped…"):
                result = post_json(
                    "/pipeline/reset",
                    {"station": station, "confirm": True},
                    timeout=240,
                )
            st.success(f"{station} reset completed. No services were started.")
            st.info("When ready, use **Start selected station** above. The pipeline will then begin consuming the now-empty Kafka topics from the new beginning.")
            st.json(result)
        except requests.RequestException as exc:
            detail = exc.response.text if getattr(exc, "response", None) is not None else str(exc)
            st.error(f"Station reset failed: {detail}")

section_title(
    "Station Kafka topics",
    "Topic creation is additive. The manager never deletes topic definitions. Reset only clears retained records for the selected station.",
)
topics = status.get("topics", {})
t1, t2 = st.columns(2)
t1.markdown(f"**Raw topic**  \n`{topics.get('raw') or entry.get('raw_topic', '—')}`")
t2.markdown(f"**Featured topic**  \n`{topics.get('featured') or entry.get('featured_topic', '—')}`")
if st.button("Create / verify Kafka topics", use_container_width=False):
    try:
        result = post_json(f"/stations/{station}/topics", timeout=30)
        st.success("Kafka topics verified.")
        st.json(result)
    except requests.RequestException as exc:
        detail = exc.response.text if getattr(exc, "response", None) is not None else str(exc)
        st.error(detail)

section_title("Service logs", "Recent output from the selected station's individual service processes.")
log_service = st.selectbox("Log stream", ["spark", "predictor", "producer", "training"], key="pipeline_log_service")
if st.button("Load latest log lines"):
    try:
        result = get_json(f"/pipeline/logs/{log_service}", {"station": station})
        st.code(result.get("content") or "No log output is available yet.", language="text")
    except requests.RequestException as exc:
        detail = exc.response.text if getattr(exc, "response", None) is not None else str(exc)
        st.error(f"Could not retrieve log: {detail}")

st.caption(
    "Kafka must be started separately. Reset does not start or stop services; "
    "ordinary Start / Stop controls manage only the selected station."
)
