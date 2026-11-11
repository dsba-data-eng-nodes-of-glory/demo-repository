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

def rerun_page():
    rerun = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
    if rerun:
        rerun()

api = setup_page("Station Management", "stations")
st.title("Station Management")



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
st.subheader("Add a station")
with st.expander("Excel upload guidance", expanded=False):
    st.markdown(
        "Upload one workbook with **Demand**, **Solar** and **Wind** sheets. "
        "Demand also supplies anomaly training. All four models are selected by default."
    )
st.link_button("Download station training Excel template", f"{api}/stations/template", use_container_width=False)
model_labels = {
    "demand_forecast": "Demand forecasting",
    "solar_forecast": "Solar forecasting",
    "wind_forecast": "Wind forecasting",
    "residual_anomaly": "Residual Load anomaly autoencoder",
}
with st.form("station_onboarding_form", clear_on_submit=False):
    c1, c2 = st.columns(2)
    with c1:
        station_id = st.text_input("Station ID", placeholder="tennet").strip().lower()
        display_name = st.text_input("Display name", placeholder="TenneT")
        stream_fraction = st.slider("Reserve for demo stream (%)", min_value=10, max_value=50, value=20, step=5)
        uploaded = st.file_uploader("Historical station workbook", type=["xlsx", "xlsm"], key="station_workbook_upload")
    with c2:
        st.markdown("**Timestamp configuration**")
        format_options = {
            "Auto detect / ISO 8601 (recommended)": "auto",
            "DD-MM-YYYY HH:MM": "%d-%m-%Y %H:%M",
            "DD-MM-YYYY HH:MM:SS": "%d-%m-%Y %H:%M:%S",
            "DD.MM.YYYY HH:MM": "%d.%m.%Y %H:%M",
            "DD/MM/YYYY HH:MM": "%d/%m/%Y %H:%M",
            "YYYY-MM-DD HH:MM": "%Y-%m-%d %H:%M",
            "Custom format": "__custom__",
        }
        timestamp_format = format_options[st.selectbox("Timestamp format", list(format_options))]
        if timestamp_format == "__custom__":
            timestamp_format = st.text_input("Python datetime format", value="%Y-%m-%d %H:%M:%S").strip()
        source_timezone = st.text_input(
            "Timezone for timestamps without an offset", value="UTC",
            help="Example: Europe/Berlin; offset-aware timestamps are normalized to UTC.",
        ).strip()
        interval_minutes = int(st.number_input(
            "Source observation interval (minutes)", min_value=5, max_value=1440, value=15,
            help="The current solar/wind and anomaly trainers require 15-minute observations.",
        ))
        residual_demand_unit = st.selectbox(
            "Demand units for residual-load calculation",
            options=["mwh_per_interval", "mw"],
            format_func=lambda value: "MWh per 15-minute interval" if value == "mwh_per_interval" else "MW (average power)",
            help="Solar and Wind Energy are provided as MWh per 15-minute interval. MW demand is converted to MWh before subtraction.",
        )
        if 10080 % interval_minutes != 0 or 1440 % interval_minutes != 0:
            st.warning("Interval must divide both the 7-day lookback and 24-hour horizon.")
        elif interval_minutes != 15:
            st.warning("Solar, wind and anomaly components currently support only 15-minute data; deselect them for other intervals.")
        else:
            st.caption(f"7-day history: {10080 // interval_minutes} rows · 24-hour horizon: {1440 // interval_minutes} rows")
    st.markdown("**Select models to register**")
    selected_models = []
    mcols = st.columns(4)
    for idx, (model_id, label) in enumerate(model_labels.items()):
        with mcols[idx]:
            if st.checkbox(label, value=True, key=f"onboard_{model_id}"):
                selected_models.append(model_id)
    st.caption("Residual-load detection requires all three Demand, Solar and Wind Energy datasets, including installed-capacity columns.")
    st.markdown("**Optional Kafka topic names**")
    topic_col1, topic_col2 = st.columns(2)
    with topic_col1:
        raw_topic = st.text_input("Raw topic (automatic if empty)", placeholder="tennet-raw")
    with topic_col2:
        featured_topic = st.text_input("Featured topic (automatic if empty)", placeholder="tennet-featured")
    submitted = st.form_submit_button("Validate workbook and register station", type="primary")

if submitted:
    if not station_id or uploaded is None:
        st.error("Enter a station ID and upload the Excel workbook.")
    elif not selected_models:
        st.error("Select at least one model component.")
    elif "residual_anomaly" in selected_models and not {"demand_forecast", "solar_forecast", "wind_forecast"}.issubset(selected_models):
        st.error("Residual-load anomaly detection requires Demand, Solar and Wind Energy forecasting.")
    elif interval_minutes != 15 and any(m != "demand_forecast" for m in selected_models):
        st.error("Solar, wind and anomaly training currently require 15-minute data.")
    else:
        try:
            result = api_post(
                "/stations/onboard",
                data={
                    "station_id": station_id,
                    "display_name": display_name or station_id.upper(),
                    "stream_fraction": stream_fraction / 100.0,
                    "timestamp_format": timestamp_format,
                    "source_timezone": source_timezone or "UTC",
                    "interval_minutes": interval_minutes,
                    "selected_models": ",".join(selected_models),
                    "residual_demand_unit": residual_demand_unit,
                    "raw_topic": raw_topic.strip(),
                    "featured_topic": featured_topic.strip(),
                },
                files={"file": (uploaded.name, uploaded.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
                timeout=240,
            )
            st.success(f"Station '{result['station']}' registered; {len(selected_models)} components selected.")
            if result.get("topics_ready"):
                st.success("Kafka topics are ready.")
            elif result.get("topic_warning"):
                st.warning(f"Kafka setup pending: {result['topic_warning']}")
            with st.expander("Registration details", expanded=False):
                st.json(result)
            rerun_page()
        except requests.HTTPError as exc:
            try:
                detail = exc.response.json().get("detail", exc.response.text)
            except (ValueError, AttributeError):
                detail = str(exc)
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
selected_components = entry.get("model_components", {})
try:
    component_summary = api_get(f"/models/components?station={selected}")
    inferred_enabled = component_summary.get("enabled") or {}
    sources = component_summary.get("sources") or {}
    selected_count = sum(bool(value) for value in inferred_enabled.values())
    ready_count = sum(bool(source.get("deployment_exists")) for source in sources.values())
    render_status_strip([
        ("Model components", f"{selected_count} selected", "neutral"),
        ("Deployed", f"{ready_count} artifacts", "neutral"),
    ])
    with st.expander("Registered model components", expanded=False):
        if component_summary.get("legacy_station"):
            st.caption("Legacy station: components detected from existing datasets and checkpoints.")
        component_rows = []
        for model_id, label in model_labels.items():
            selected_for_station = bool(inferred_enabled.get(model_id, False))
            source = sources.get(model_id) or {}
            has_data = bool(source.get("registered_file_exists"))
            has_model = bool(source.get("deployment_exists"))
            if not selected_for_station:
                state = "Not selected"
            elif not has_data:
                state = "Needs training data"
            elif not has_model:
                state = "Needs model training"
            else:
                state = "Ready"
            component_rows.append({
                "Component": label,
                "Selected": "Yes" if selected_for_station else "No",
                "Training data": "Available" if has_data else "Missing",
                "Deployed model": "Available" if has_model else "Not deployed",
                "Next action": state,
            })
        st.dataframe(component_rows, hide_index=True, use_container_width=True)
        st.caption("Selected ≠ trained ≠ deployed. A station may retain a trained model while its streaming pipeline is stopped.")
        st.page_link("pages/model_management.py", label="Train or deploy a component", icon="🧠", use_container_width=False)
except requests.RequestException as exc:
    st.warning(f"Component status is unavailable: {exc}")

try:
    status = api_get(f"/stations/{selected}/status")
    model = api_get(f"/models/status?station={selected}")
except requests.RequestException as exc:
    st.error(f"Could not load station status: {exc}")
    st.stop()

@st.fragment(run_every=2.0)
def render_live_station_status(station_id: str, station_entry: dict, model_info: dict) -> None:
    """Poll status only; never refresh registration or deletion forms."""
    try:
        live_status = api_get(f"/stations/{station_id}/status")
    except requests.RequestException as exc:
        st.warning(f"Live station status temporarily unavailable: {exc}")
        return
    rows_registered = station_entry.get("registered_rows")
    render_status_strip([
        ("History", f"{rows_registered:,} rows" if isinstance(rows_registered, int) else "—", "neutral"),
        ("Demand", "Ready" if model_info.get("checkpoint_exists") else "Needs training",
         "good" if model_info.get("checkpoint_exists") else "warn"),
        ("Kafka", "Connected" if live_status.get("kafka_tcp_available") else "Offline",
         "good" if live_status.get("kafka_tcp_available") else "bad"),
        ("Station", str(live_status.get("lifecycle_status", station_entry.get("status", "registered"))).replace("_", " ").title(), "neutral"),
    ])
    with st.expander("Technical station details", expanded=False):
        st.json({k: v for k, v in station_entry.items() if k not in {"registered_source_file"}})
        services = live_status.get("services", {})
        render_status_strip([
            (name.title(), "Running" if services.get(name, {}).get("running") else "Stopped",
             "good" if services.get(name, {}).get("running") else "neutral")
            for name in ("training", "spark", "predictor", "producer")
        ])
        if model_info.get("checkpoint_exists"):
            metadata = model_info.get("metadata") or {}
            st.write(f"Model version: {model_info.get('model_version') or 'v1'}")
            rmse = metadata.get("best_validation_rmse_scaled")
            st.write(f"Validation RMSE (scaled): {rmse:.4f}" if isinstance(rmse, (int, float)) else "Validation RMSE: —")
            st.write(f"Checkpoint: `{model_info.get('checkpoint_path')}`")


render_live_station_status(selected, entry, model)

# ---------------------------------------------------------------------
# Logs
# ---------------------------------------------------------------------
with st.expander("Advanced: service logs", expanded=False):
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
        type="secondary",
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
