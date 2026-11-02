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
st.caption("Central control for station registration, data validation, LSTM training, Kafka topics and live streaming.")


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
    "Upload one continuous 15-minute historical CSV containing **utc_timestamp, demand, temperature**. "
    "The onboarding service reserves the final part as a simulated stream and uses the earlier data for training/validation."
)
with st.form("station_onboarding_form", clear_on_submit=False):
    c1, c2 = st.columns(2)
    with c1:
        station_id = st.text_input("Station ID", placeholder="tennet").strip().lower()
        display_name = st.text_input("Display name", placeholder="TenneT")
    with c2:
        stream_fraction = st.slider("Reserve for demo stream (%)", min_value=10, max_value=50, value=20, step=5)
        uploaded = st.file_uploader("Historical station CSV", type=["csv"], key="station_csv_upload")
    submitted = st.form_submit_button("Validate data and register station")

if submitted:
    if not station_id or uploaded is None:
        st.error("Enter a station ID and upload a CSV file.")
    else:
        try:
            payload = {
                "station_id": station_id,
                "display_name": display_name or station_id.upper(),
                "stream_fraction": stream_fraction / 100.0,
            }
            result = api_post(
                "/stations/onboard",
                data=payload,
                files={"file": (uploaded.name, uploaded.getvalue(), "text/csv")},
                timeout=180,
            )
            st.success(f"Station '{result['station']}' registered and data validated.")
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

# Lifecycle buttons: training, topics, activation and stop.
st.markdown("#### Station lifecycle")
col_train, col_topics, col_activate, col_stop = st.columns(4)
with col_train:
    if st.button("Train / retrain model", type="primary"):
        try:
            result = api_post(f"/stations/{selected}/train", timeout=30)
            st.success("Training process started.")
            st.json(result)
        except requests.HTTPError as exc:
            st.error(exc.response.text if exc.response is not None else str(exc))
        except requests.RequestException as exc:
            st.error(str(exc))
with col_topics:
    if st.button("Create / verify topics"):
        try:
            result = api_post(f"/stations/{selected}/topics", timeout=30)
            st.success("Kafka topics are ready.")
            st.json(result)
        except requests.HTTPError as exc:
            st.error(exc.response.text if exc.response is not None else str(exc))
        except requests.RequestException as exc:
            st.error(str(exc))
with col_activate:
    if st.button("Activate station"):
        try:
            result = api_post(f"/stations/{selected}/activate", timeout=60)
            st.success("Station pipeline start requested.")
            st.json(result)
        except requests.HTTPError as exc:
            st.error(exc.response.text if exc.response is not None else str(exc))
        except requests.RequestException as exc:
            st.error(str(exc))
with col_stop:
    if st.button("Stop station"):
        try:
            result = api_post(f"/stations/{selected}/stop", timeout=30)
            st.success("Station services stop requested.")
            st.json(result)
        except requests.HTTPError as exc:
            st.error(exc.response.text if exc.response is not None else str(exc))
        except requests.RequestException as exc:
            st.error(str(exc))

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
    st.info("No trained model exists yet. Train this station before activating its streaming pipeline.")

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

st.caption("Kafka is managed separately and must be running. Station activation creates missing topics but never deletes or truncates existing topics.")


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
