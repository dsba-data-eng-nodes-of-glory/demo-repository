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

def rerun_page():
    rerun = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
    if rerun:
        rerun()

api = setup_page("Station Management", "stations")
st.title("Station Management")
st.caption("Register station data, configure timestamp interpretation and manage registered station records.")


def api_get(path: str):
    response = requests.get(f"{api}{path}", timeout=15)
    response.raise_for_status()
    return response.json()


def api_post(path: str, *, json=None, data=None, files=None, timeout=60):
    response = requests.post(f"{api}{path}", json=json, data=data, files=files, timeout=timeout)
    response.raise_for_status()
    return response.json()


def api_delete(path: str, *, json=None, timeout=240):
    response = requests.delete(f"{api}{path}", json=json, timeout=timeout)
    response.raise_for_status()
    return response.json()


try:
    registry = api_get("/stations")
except requests.RequestException as exc:
    st.error(f"Cannot reach FastAPI at {api}: {exc}")
    st.stop()

stations = registry.get("stations", {})

# ---------------------------------------------------------------------
# Register new station
# ---------------------------------------------------------------------
st.subheader("Register a new station")
st.markdown(
    "Upload historical station data. Timestamp and demand/load columns are required and detected automatically. "
    "Temperature is optional; when absent, the platform adds a constant placeholder, so the model cannot learn temperature effects. "
    "Choose the forecast interval; observations are averaged into that interval, while empty periods remain separate gaps. "
    "Training windows never span a gap, and the latest continuous segment is reserved for the demo stream."
)
with st.form("station_onboarding_form", clear_on_submit=False):
    c1, c2 = st.columns(2)
    with c1:
        station_id = st.text_input("Station ID", placeholder="tennet").strip().lower()
        display_name = st.text_input("Display name", placeholder="TenneT")
        stream_fraction = st.slider("Reserve for demo stream (%)", min_value=10, max_value=50, value=20, step=5)
        uploaded = st.file_uploader("Historical station CSV", type=["csv"], key="station_csv_upload")
    with c2:
        st.markdown("**Timestamp configuration**")
        format_options = {
            "Auto detect (recommended)": "auto",
            "DD-MM-YYYY HH:MM": "%d-%m-%Y %H:%M",
            "DD-MM-YYYY HH:MM:SS": "%d-%m-%Y %H:%M:%S",
            "DD.MM.YYYY HH:MM": "%d.%m.%Y %H:%M",
            "DD.MM.YYYY HH:MM:SS": "%d.%m.%Y %H:%M:%S",
            "DD/MM/YYYY HH:MM": "%d/%m/%Y %H:%M",
            "MM/DD/YYYY HH:MM": "%m/%d/%Y %H:%M",
            "YYYY-MM-DD HH:MM": "%Y-%m-%d %H:%M",
            "YYYY-MM-DD HH:MM:SS": "%Y-%m-%d %H:%M:%S",
            "ISO 8601": "auto",
            "Custom format": "__custom__",
        }
        selected_format_label = st.selectbox(
            "Timestamp format",
            list(format_options),
            index=0,
            help="Auto detect supports common European, ISO 8601 and timezone-aware timestamp values.",
        )
        timestamp_format = format_options[selected_format_label]
        if timestamp_format == "__custom__":
            timestamp_format = st.text_input(
                "Python datetime format",
                value="%Y-%m-%d %H:%M:%S",
                help="Example: %d-%m-%Y %H:%M",
            ).strip()
        interval_minutes = int(st.number_input(
            "Forecast interval (minutes)",
            min_value=5,
            max_value=1440,
            value=15,
            step=1,
            help=(
                "Source observations are averaged into this interval. "
                "The forecasting contract keeps a 7-day history and a 24-hour forecast horizon; "
                "the number of rows is derived automatically. The interval must divide both durations exactly."
            ),
        ))
        if 10080 % interval_minutes != 0 or 1440 % interval_minutes != 0:
            st.warning(
                f"{interval_minutes} minutes does not divide the 7-day history and 24-hour forecast "
                "into whole observations. Choose a different interval."
            )
        else:
            st.caption(
                f"Timestamp and demand/load columns are required. Temperature is optional. "
                f"Source rows will be averaged into {interval_minutes}-minute bins; missing bins stay gaps. "
                f"The model will use {10080 // interval_minutes:,} history steps and "
                f"{1440 // interval_minutes:,} forecast steps for a 7-day lookback and 24-hour horizon. "
                "Offset-aware timestamps are converted to UTC; timezone-naive timestamps are assumed to be UTC."
            )
    st.markdown("**Optional Kafka topic names**")
    topic_col1, topic_col2 = st.columns(2)
    with topic_col1:
        raw_topic = st.text_input("Raw topic (leave blank for automatic name)", placeholder="tennet-raw")
    with topic_col2:
        featured_topic = st.text_input("Featured topic (leave blank for automatic name)", placeholder="tennet-featured")
    submitted = st.form_submit_button("Validate data and register station", type="primary")

if submitted:
    if not station_id or uploaded is None:
        st.error("Enter a station ID and upload a CSV file.")
    else:
        # The backend detects the required columns from the uploaded CSV header
        # and returns a validation error with sample values when parsing fails.
        try:
            payload = {
                "station_id": station_id,
                "display_name": display_name or station_id.upper(),
                "stream_fraction": stream_fraction / 100.0,
                "timestamp_format": timestamp_format,
                "interval_minutes": interval_minutes,
                "raw_topic": raw_topic.strip(),
                "featured_topic": featured_topic.strip(),
            }
            result = api_post(
                "/stations/onboard",
                data=payload,
                files={"file": (uploaded.name, uploaded.getvalue(), "text/csv")},
                timeout=240,
            )
            st.success(f"Station '{result['station']}' registered and data validated.")
            if result.get("topics_ready"):
                st.success("Kafka topics were created or already existed.")
            else:
                st.warning(f"Station registered, but Kafka topics are pending: {result.get('topic_warning')}. Topics will be retried automatically when you activate the station after Kafka is available.")
            if result.get("temperature_note"):
                st.warning(result["temperature_note"])
            if result.get("resampling_note"):
                st.info(result["resampling_note"])
            st.json(result)
            rerun_page()
        except requests.HTTPError as exc:
            detail = exc.response.text if exc.response is not None else str(exc)
            st.error(f"Station registration failed: {detail}")
        except requests.RequestException as exc:
            st.error(f"Station registration failed: {exc}")

st.markdown("---")

# ---------------------------------------------------------------------
# Existing station operations
# ---------------------------------------------------------------------
st.subheader("Registered stations")
try:
    registry = api_get("/stations")
    stations = registry.get("stations", {})
except requests.RequestException as exc:
    st.error(f"Could not refresh station registry: {exc}")
    st.stop()

if not stations:
    st.info("No stations are registered yet.")
    st.stop()

station_ids = list(stations.keys())
default_station = registry.get("active_station")
default_index = station_ids.index(default_station) if default_station in station_ids else 0
selected = st.selectbox(
    "Select station",
    station_ids,
    index=default_index,
    format_func=lambda key: stations[key].get("display_name", key),
)
entry = stations[selected]

try:
    status = api_get(f"/stations/{selected}/status")
    model = api_get(f"/models/status?station={selected}")
except requests.RequestException as exc:
    st.error(f"Could not load station status: {exc}")
    st.stop()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Registered rows", f"{entry.get('registered_rows', '—'):,}" if isinstance(entry.get("registered_rows"), int) else "—")
k2.metric("Model ready", "Yes" if model.get("checkpoint_exists") else "No")
k3.metric("Kafka", "Available" if status.get("kafka_tcp_available") else "Unavailable")
k4.metric("Station state", str(status.get("lifecycle_status", entry.get("status", "registered"))).replace("_", " ").title())

with st.expander("Station configuration", expanded=False):
    st.json({k: v for k, v in entry.items() if k not in {"registered_source_file"}})

# Lifecycle operations are controlled from Home so registered stations can be
# independently activated and stopped/reset without duplicating controls here.
st.info("Use the station cards on Home to activate a station, open its monitoring page, or stop + reset it. Kafka topics are provisioned automatically during registration and retried during activation. Model training is managed separately in Model Management.")

# ---------------------------------------------------------------------
# Service and model status
# ---------------------------------------------------------------------
st.markdown("#### Connected pipeline services")
services = status.get("services", {})
service_cols = st.columns(4)
for col, service_name in zip(service_cols, ("training", "spark", "predictor", "producer")):
    details = services.get(service_name, {})
    running = details.get("running", False)
    col.metric(service_name.title(), "Running" if running else "Stopped")

st.markdown("#### Model information")
if model.get("checkpoint_exists"):
    metadata = model.get("metadata") or {}
    m1, m2, m3 = st.columns(3)
    m1.metric("Model version", str(model.get("model_version") or "v1"))
    metric = metadata.get("best_validation_rmse_scaled")
    m2.metric("Validation RMSE (scaled)", f"{metric:.4f}" if isinstance(metric, (float, int)) else "Not available")
    m3.metric("Best epoch", str(metadata.get("best_epoch", "Not available")))
    st.caption(f"Checkpoint: {model.get('checkpoint_path')}")
else:
    st.info("No trained model exists yet. Train a model from Model Management before activating this station from Home.")

# ---------------------------------------------------------------------
# Logs
# ---------------------------------------------------------------------
st.markdown("#### Station process logs")
log_service = st.selectbox("Log source", ["training", "spark", "predictor", "producer"], key="station_log_service")
if st.button("Refresh selected log"):
    try:
        log_data = api_get(f"/pipeline/logs/{log_service}?station={selected}")
        st.code(log_data.get("content", "No log output yet."), language="text")
    except requests.RequestException as exc:
        st.error(f"Could not load log: {exc}")

if st.button("Refresh station status"):
    rerun_page()

st.caption("Kafka topics are created automatically during registration when the broker is available. If Kafka is temporarily unavailable, activation retries topic creation automatically.")


# ---------------------------------------------------------------------
# Permanent station deletion
# ---------------------------------------------------------------------
st.markdown("---")
st.subheader("Danger zone — Delete station")
st.warning(
    "This permanently removes the selected station from the registry, deletes its "
    "configured input/history/training CSVs, model checkpoint and metadata, station "
    "Spark checkpoint, station-specific logs, and its rows in actuals.csv and forecasts.csv. "
    "Its Kafka records are cleared, but the empty Kafka topic definitions are retained "
    "to avoid Windows KRaft topic-directory deletion errors. Other registered stations "
    "and their data are preserved."
)
with st.expander("Delete this station and all its data", expanded=False):
    st.write(f"Selected station: **{selected}** — {entry.get('display_name', selected)}")
    st.caption(
        "The operation is blocked while this station has running producer, Spark, predictor "
        "or training processes. It is also blocked if a topic or configured file is shared "
        "with another station, or if another station's predictor is writing to the shared result CSVs."
    )
    delete_ack = st.checkbox(
        "I understand this permanently deletes this station's local data, model and saved results.",
        key=f"delete_ack_{selected}",
    )
    typed_station = st.text_input(
        f"Type {selected} to confirm",
        key=f"delete_station_text_{selected}",
        placeholder=selected,
    )
    services_running = any(
        status.get("services", {}).get(name, {}).get("running", False)
        for name in ("training", "spark", "predictor", "producer")
    )
    if services_running:
        st.info("Stop all services for this station before deleting it.")
    if st.button(
        f"Delete {selected} permanently",
        type="primary",
        use_container_width=True,
        disabled=(not delete_ack or typed_station.strip().lower() != selected.lower() or services_running),
        key=f"delete_station_button_{selected}",
    ):
        try:
            with st.spinner(f"Deleting {selected} and clearing its records…"):
                deletion = api_delete(
                    f"/stations/{selected}",
                    json={"confirm_station_id": typed_station},
                    timeout=300,
                )
            st.success(deletion.get("message", f"Station {selected} deleted."))
            st.json(deletion)
            rerun_page()
        except requests.HTTPError as exc:
            detail = exc.response.text if exc.response is not None else str(exc)
            st.error(f"Station deletion failed: {detail}")
        except requests.RequestException as exc:
            st.error(f"Station deletion failed: {exc}")
