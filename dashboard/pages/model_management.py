"""Model Lab: configure, train, compare, register and deploy forecasting models.

The existing operational accuracy panels are retained in the Live Performance tab.
"""
from __future__ import annotations

import math
import requests
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from dashboard.ui import setup_page, section_title, render_status_strip
    from dashboard.model_metrics import (
        build_evaluation_frame, calculate_metrics, metrics_by_horizon, metrics_by_run,
    )
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from dashboard.ui import setup_page, section_title, render_status_strip
    from dashboard.model_metrics import (
        build_evaluation_frame, calculate_metrics, metrics_by_horizon, metrics_by_run,
    )

api = setup_page("Model Management", "models")


@st.cache_data(ttl=2, show_spinner=False)
def get_json(api_url: str, path: str, params: dict | None = None) -> dict:
    response = requests.get(f"{api_url.rstrip('/')}{path}", params=params, timeout=25)
    response.raise_for_status()
    return response.json()


def api_write(method: str, path: str, payload: dict) -> dict:
    response = requests.request(method, f"{api.rstrip('/')}{path}", json=payload, timeout=60)
    response.raise_for_status()
    return response.json()


def upload_dataset(station: str, component: str, uploaded, timestamp_format: str, source_timezone: str) -> dict:
    response = requests.post(
        f"{api.rstrip('/')}/models/training-data/upload",
        data={"station": station, "component": component, "timestamp_format": timestamp_format,
              "source_timezone": source_timezone},
        files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type or "application/octet-stream")},
        timeout=(10, 300),
    )
    response.raise_for_status()
    return response.json()


def readable_http_error(exc: requests.RequestException) -> str:
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            return str(response.json().get("detail", response.text))
        except (ValueError, AttributeError):
            return response.text[:800]
    return str(exc)


def fmt_mw(value) -> str:
    return "—" if value is None or pd.isna(value) else f"{float(value):,.2f} MW"


def fmt_pct(value) -> str:
    return "—" if value is None or pd.isna(value) else f"{float(value):.2f}%"


def fmt_utc(value) -> str:
    if value is None or pd.isna(value):
        return "—"
    return pd.to_datetime(value, utc=True).strftime("%Y-%m-%d %H:%M UTC")


def dark_layout(fig: go.Figure, height: int = 340) -> go.Figure:
    fig.update_layout(
        template="plotly_dark", height=height,
        margin=dict(l=12, r=12, t=55, b=12),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(7,20,39,.55)",
        hovermode="x unified", legend=dict(orientation="h", y=1.12, x=0),
    )
    return fig


def _parameter_widget(label: str, value, key: str, parameter_name: str):
    if isinstance(value, bool):
        return st.checkbox(label, value=value, key=key)
    if parameter_name in ("subsample", "colsample_bytree"):
        return st.slider(label, min_value=0.1, max_value=1.0, value=float(value), step=0.05, key=key)
    if isinstance(value, int):
        minimum = 0 if parameter_name == "random_state" else 1
        maximum = 4 if parameter_name == "n_jobs" else 2000
        return st.number_input(label, min_value=minimum, max_value=maximum, value=int(value), step=1, key=key)
    if isinstance(value, float):
        if parameter_name == "learning_rate":
            maximum = 0.3
            minimum = 0.0001
        else:
            maximum = 100.0
            minimum = 0.0
        step = 0.001 if abs(value) < 0.01 else 0.01
        return st.number_input(label, min_value=minimum, max_value=maximum, value=float(value), step=step, format="%.4f", key=key)
    return st.text_input(label, value=str(value), key=key)


COMPONENT_LABELS = {
    "demand_forecast": "Demand forecasting",
    "solar_forecast": "Solar forecasting",
    "wind_forecast": "Wind forecasting",
    "residual_anomaly": "Residual Load anomaly autoencoder",
}


def render_model_lab(station: str, station_label: str, selected_component: str,
                     component_payload: dict) -> None:
    section_title("Model training workspace", "Choose one component, its training dataset and algorithm. Training creates a candidate version; deployment remains a separate approval step.")
    try:
        station_status = get_json(api, f"/stations/{station}/status")
    except requests.RequestException as exc:
        st.error(f"Could not load training status: {readable_http_error(exc)}")
        return
    available = component_payload.get("component_algorithms", {})
    enabled = component_payload.get("enabled") or {}
    source_info = (component_payload.get("sources") or {}).get(selected_component, {})
    selected_enabled = bool(enabled.get(selected_component))
    algorithms = available.get(selected_component, [])
    if not selected_enabled:
        st.warning(
            f"{COMPONENT_LABELS[selected_component]} was deselected for this station. "
            "To preserve the customer's onboarding choice, it cannot be retrained "
            "until the station configuration is explicitly updated."
        )
    if not algorithms:
        st.error(f"No algorithms are configured for {selected_component}.")
        return
    st.markdown(f"**Station:** {station_label} (`{station}`) · **Component:** {COMPONENT_LABELS[selected_component]}")
    algorithm_labels = component_payload.get("labels", {})
    algo = st.selectbox(
        "Training algorithm", algorithms,
        format_func=lambda key: algorithm_labels.get(key, {}).get("label", key.upper()),
        key=f"training_algorithm_{station}_{selected_component}",
    )
    if selected_component == "demand_forecast":
        defaults = component_payload.get("demand_algorithms", {}).get(algo, {})
    elif selected_component == "residual_anomaly":
        defaults = component_payload.get("anomaly_algorithms", {}).get(algo, {})
    else:
        defaults = component_payload.get("renewable_algorithms", {}).get(algo, {})
    training = station_status.get("services", {}).get("training", {})
    status_value = "Running" if training.get("running") else "Idle"
    st.caption(f"Training process: **{status_value}** · One training job per station at a time.")
    if not defaults:
        st.error(f"No default parameters for {selected_component}/{algo}.")
        return
    if selected_component in ("solar_forecast", "wind_forecast"):
        st.info("Solar and Wind share algorithm parameter defaults across stations. Input forecast-weather columns must contain predictions available at the corresponding origin.")
    if selected_component == "residual_anomaly":
        st.info("Dense Autoencoder only · 96-step capacity-aware residual windows · needs >=2 years of aligned Demand/Solar/Wind for fitting, plus validation/calibration.")

    # Each upload is validated, normalized and stored as an immutable candidate
    # input. It does not modify live producer files or deployed model assignments.
    section_title("Training data", "Use this station's current historical data or upload a complete replacement training snapshot for just this model.")
    try:
        uploads = get_json(api, "/models/training-data", {"station": station, "component": selected_component}).get("datasets", [])
    except requests.RequestException as exc:
        st.warning(f"Uploaded dataset history is unavailable: {readable_http_error(exc)}")
        uploads = []
    registered_exists = bool(source_info.get("registered_file_exists"))
    registered_path = source_info.get("registered_file") or "Not configured"
    dataset_map = {item["source_id"]: item for item in uploads}
    source_choices = ["registered"] + list(dataset_map)
    preferred = st.session_state.pop(f"preferred_source_{station}_{selected_component}", None)
    source_key = f"training_source_{station}_{selected_component}"
    if preferred in source_choices:
        st.session_state[source_key] = preferred
    selected_source = st.selectbox(
        "Dataset for this training run", source_choices,
        format_func=lambda value: (
            f"Registered historical dataset{' (missing)' if not registered_exists else ''}"
            if value == "registered" else
            f"Uploaded · {dataset_map[value].get('original_filename')} · {dataset_map[value].get('rows', 0):,} rows · {dataset_map[value].get('uploaded_at', '')[:16]}"
        ),
        key=source_key,
    )
    if selected_source == "registered":
        st.caption(f"Registered file: `{registered_path}`")
        source_available = registered_exists
        if not registered_exists:
            st.warning("No existing training file was found for this component. Upload historical data below before starting training.")
    else:
        source_available = True
        item = dataset_map[selected_source]
        st.caption(
            f"Validated snapshot · {item['rows']:,} observations · "
            f"{item['start_utc']} to {item['end_utc']} · SHA-256 {item['sha256'][:12]}…"
        )

    with st.expander("Upload and validate new historical training data", expanded=not registered_exists):
        st.write("Upload a model CSV or the three-sheet Excel template. Residual Load requires all three sheets with aligned timestamps and capacity.")
        st.link_button("Download station Excel training template", f"{api.rstrip('/')}/stations/template")
        new_file = st.file_uploader(
            "New training dataset (CSV / Excel)", type=["csv", "xlsx", "xlsm"],
            key=f"new_training_file_{station}_{selected_component}",
        )
        s1, s2 = st.columns(2)
        with s1:
            upload_tz = st.text_input(
                "Timezone for timestamps without an offset", value="UTC",
                key=f"training_tz_{station}_{selected_component}",
            )
        with s2:
            upload_fmt = st.text_input(
                "Timestamp format", value="auto",
                help="Use auto for ISO-8601; otherwise specify a Python datetime format.",
                key=f"training_format_{station}_{selected_component}",
            )
        st.caption("A new upload does not overwrite existing station data or deployed models. Upload enough continuous history for the selected model; this is not a partial-data append operation.")
        if st.button(
            "Validate & save dataset", disabled=not selected_enabled or new_file is None,
            key=f"save_uploaded_dataset_{station}_{selected_component}",
        ):
            try:
                with st.spinner("Checking the complete dataset and storing a model-specific snapshot…"):
                    result = upload_dataset(station, selected_component, new_file, upload_fmt, upload_tz)
                st.session_state[f"preferred_source_{station}_{selected_component}"] = result["source_id"]
                get_json.clear()
                st.success(f"Validated {result['rows']:,} rows. The uploaded snapshot is ready for retraining.")
                st.rerun()
            except requests.RequestException as exc:
                st.error(f"Upload rejected: {readable_http_error(exc)}")

    with st.expander("⚙ Configure training parameters", expanded=True):
        edits = {}
        grid = st.columns(3)
        for idx, (param, value) in enumerate(defaults.items()):
            with grid[idx % 3]:
                edits[param] = _parameter_widget(
                    param.replace("_", " ").title(), value,
                    f"model_cfg_{station}_{selected_component}_{algo}_{param}", param,
                )
        if selected_component != "residual_anomaly":
            val_fraction = st.slider(
                "Chronological validation fraction", 0.05, 0.40,
                float(component_payload.get("training_options", {}).get("validation_fraction", 0.15)), 0.05,
                key=f"model_valid_{station}_{selected_component}",
            )
        else:
            val_fraction = 0.15
        if st.button("Save parameter defaults", key=f"save_component_cfg_{station}_{selected_component}_{algo}"):
            try:
                saved = api_write("PUT", f"/models/component-config/{selected_component}/{algo}", {
                    "parameters": edits,
                })
                get_json.clear()
                st.success("Saved model defaults." + (" Solar and wind share this profile." if saved.get("shared_across_solar_and_wind") else ""))
            except requests.RequestException as exc:
                st.error(f"Configuration could not be saved: {readable_http_error(exc)}")
    left, right = st.columns([1, 2], vertical_alignment="center")
    with left:
        if st.button(
            "▶ Run training", type="primary", use_container_width=True,
            disabled=bool(training.get("running")) or not source_available or not selected_enabled,
            key=f"train_component_{station}_{selected_component}_{algo}",
        ):
            try:
                result = api_write("POST", "/models/experiments/train", {
                    "station": station, "component": selected_component,
                    "algorithm": algo, "parameters": edits,
                    "training_options": {"validation_fraction": val_fraction},
                    "training_data_id": selected_source if selected_source != "registered" else None,
                })
                get_json.clear()
                st.success(f"Training started: {result.get('experiment_id')}")
            except requests.RequestException as exc:
                st.error(f"Training could not be started: {readable_http_error(exc)}")
    with right:
        st.caption("Each run creates a candidate model version. It does not change the current live model until you explicitly deploy it.")
    try:
        records = get_json(api, "/models/experiments", {"station": station, "component": selected_component}).get("experiments", [])
    except requests.RequestException as exc:
        st.warning(f"Experiment registry is unavailable: {readable_http_error(exc)}")
        return
    if records:
        st.markdown("**Recent experiments — selected component**")
        rows = []
        for item in records[:15]:
            metric = item.get("metrics") or {}
            rows.append({
                "Experiment": item.get("experiment_id"),
                "Algorithm": item.get("algorithm"), "Version": item.get("version", "—"),
                "Status": item.get("status"),
                "Dataset": (item.get("training_data_id") or "Registered data")[:32],
                "MAE": metric.get("mae_mwh", metric.get("mae_mw")),
                "RMSE": metric.get("rmse_mwh", metric.get("rmse_mw")),
                "WAPE (%)": metric.get("wape_pct"),
                "Validation loss": metric.get("best_validation_loss"),
                "Error": item.get("error", ""),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        metric_unit = "MWh per interval" if selected_component in ("solar_forecast", "wind_forecast") else "MW" if selected_component == "demand_forecast" else "weighted reconstruction loss"
        st.caption(f"Metric unit: {metric_unit}. Do not compare absolute errors across different components or training datasets.")
    else:
        st.caption("No experiments yet for this model component.")


def render_comparison(station: str, component: str = "demand_forecast") -> None:
    try:
        station_status = get_json(api, f"/stations/{station}/status")
        interval_minutes = int(station_status.get("interval_minutes", 15))
        history_steps = int(station_status.get("history_steps", 672))
        horizon_steps = int(station_status.get("horizon_steps", 96))
        records = get_json(api, "/models/experiments", {"station": station, "component": component}).get("experiments", [])
    except requests.RequestException as exc:
        st.error(f"Could not load experiment registry or station forecast contract: {exc}")
        return
    section_title(
        f"Compare experiments · {COMPONENT_LABELS.get(component, component)}",
        f"Compare 2–4 completed experiments using the same station, chronological holdout and forecast contract "
        f"({history_steps:,} history steps, {horizon_steps:,} forecast steps at {interval_minutes}-minute intervals).",
    )
    complete = [item for item in records if item.get("status") in ("candidate", "deployed", "previous")]
    if component == "residual_anomaly":
        st.info("Autoencoder quality is evaluated using reconstruction error, calibration and anomaly detection—not forecasting MAE. Compare training summaries in the registry.")
        return
    if len(complete) < 2:
        st.info("At least two completed model experiments are required for comparison. Train another algorithm first.")
        return
    labels = {
        item["experiment_id"]: f"{item.get('algorithm', '').upper()} {item.get('version', '')} · {item['experiment_id']}"
        for item in complete
    }
    selected = st.multiselect(
        "Select 2–4 experiments", list(labels),
        format_func=lambda value: labels.get(value, value),
        key=f"model_compare_selection_{station}_{component}",
    )
    if len(selected) > 4:
        st.warning("Select no more than four experiments at a time.")
    if st.button("Compare selected models", type="primary", disabled=not 2 <= len(selected) <= 4, key=f"compare_models_{station}_{component}"):
        try:
            result = api_write("POST", "/models/compare", {"experiment_ids": selected})
            st.session_state[f"model_compare_result_{station}_{component}"] = result
        except requests.RequestException as exc:
            st.error(f"Comparison failed: {exc}")

    result = st.session_state.get(f"model_compare_result_{station}_{component}")
    if not result:
        return
    if result.get("comparable"):
        st.success("Comparable experiment set: station, history, horizon, interval, validation period and evaluation protocol match.")
    else:
        st.warning("These experiments are not a fully controlled comparison.")
        for warning in result.get("warnings", []):
            st.write(f"• {warning}")

    results = result.get("results", [])
    if not results:
        st.info("No comparison results returned.")
        return
    frame = pd.DataFrame(results)
    metric_unit = "MWh" if component in ("solar_forecast", "wind_forecast") else "MW"
    display_columns = [
        "model", "status", f"MAE ({metric_unit})", f"RMSE ({metric_unit})", "WAPE (%)", f"Bias ({metric_unit})",
        "Training time (s)", "Inference latency (ms/origin)", "Model size (bytes)",
    ]
    visible = frame[[column for column in display_columns if column in frame.columns]].copy()
    st.dataframe(visible.round(4), use_container_width=True, hide_index=True)

    metric_choice = st.selectbox("Compare metric", [f"MAE ({metric_unit})", f"RMSE ({metric_unit})", "WAPE (%)", "Training time (s)", "Inference latency (ms/origin)"], key=f"comparison_metric_{station}_{component}")
    plot_frame = frame.dropna(subset=[metric_choice]) if metric_choice in frame else pd.DataFrame()
    if not plot_frame.empty:
        fig = go.Figure(go.Bar(x=plot_frame["model"], y=plot_frame[metric_choice], text=plot_frame[metric_choice].round(3), textposition="outside"))
        dark_layout(fig, 320).update_layout(title=f"Model comparison · {metric_choice}", xaxis_title="Experiment", yaxis_title=metric_choice)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with st.expander("Compare hyperparameters and training lineage"):
        for row in results:
            st.markdown(f"**{row['model']}**")
            st.json({
                "parameters": row.get("Parameters", {}),
                "feature_schema": row.get("feature_schema"),
                "validation_origin_count": row.get("validation_origin_count"),
                "training_data_sha256": row.get("training_data_sha256"),
                "training_data_id": row.get("training_data_id"),
            })

    horizon_rows = []
    for row in results:
        for point in row.get("horizon_metrics", []):
            horizon_rows.append({"model": row["model"], **point})
    if horizon_rows:
        horizon_frame = pd.DataFrame(horizon_rows)
        horizon_frame["lead_minutes"] = horizon_frame["horizon_step"].astype(float) * interval_minutes
        horizon_options = ["mae_mwh", "rmse_mwh", "wape_pct"] if metric_unit == "MWh" else ["mae_mw", "rmse_mw", "wape_pct"]
        horizon_options = [name for name in horizon_options if name in horizon_frame.columns]
        if not horizon_options:
            return
        horizon_metric = st.selectbox("Horizon metric", horizon_options, format_func=lambda value: value.replace("_", " ").upper(), key=f"horizon_compare_metric_{station}_{component}")
        fig = go.Figure()
        for model_name, group in horizon_frame.groupby("model"):
            fig.add_trace(go.Scatter(x=group["lead_minutes"], y=group[horizon_metric], mode="lines", name=model_name))
        dark_layout(fig, 340).update_layout(
            title=f"Forecast error by lead time · {horizon_metric}",
            xaxis_title="Forecast lead time (minutes)", yaxis_title=horizon_metric,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_registry_and_deployment(station: str, component: str = "demand_forecast") -> None:
    section_title("Model registry", "Trained artifacts, configuration, performance metrics and deployment lifecycle for the selected station.")
    try:
        payload = get_json(api, "/models/registry", {"station": station})
        status = get_json(api, "/models/status", {"station": station})
    except requests.RequestException as exc:
        st.error(f"Could not load model registry: {exc}")
        return
    records = [item for item in payload.get("experiments", [])
               if item.get("component", "demand_forecast") == component]
    active = status.get("active_model") or {}
    if component != "demand_forecast":
        active = next((item for item in records if item.get("status") == "deployed"), {})
    if active:
        st.success(f"Active deployment: {active.get('algorithm', '').upper()} {active.get('version', '')} · {active.get('experiment_id')}")
    elif component == "demand_forecast" and status.get("checkpoint_exists"):
        st.info(f"Legacy model currently configured: {status.get('model_checkpoint', 'LSTM checkpoint')}")
    else:
        try:
            source = get_json(api, "/models/components", {"station": station}).get("sources", {}).get(component, {})
            deployed_path = source.get("deployment_file")
            if source.get("deployment_exists"):
                st.info(f"Legacy/existing deployed {COMPONENT_LABELS.get(component, component)} artifact: `{deployed_path}`")
            else:
                st.warning("No model artifact is currently deployed for this station.")
        except requests.RequestException:
            st.warning("No registered deployment details are available for this component.")
    if records:
        rows = []
        for item in records:
            metrics = item.get("metrics") or {}
            rows.append({
                "Experiment ID": item.get("experiment_id"),
                "Algorithm": item.get("algorithm"),
                "Version": item.get("version", "—"),
                "Status": item.get("status"),
                "MAE": metrics.get("mae_mwh", metrics.get("mae_mw")),
                "RMSE": metrics.get("rmse_mwh", metrics.get("rmse_mw")),
                "WAPE (%)": metrics.get("wape_pct"),
                "Dataset": item.get("training_data_id") or "Registered",
                "Created": item.get("created_at"),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No registered candidate experiments yet.")

    deployable = [item for item in records if item.get("status") in ("candidate", "deployed", "previous")]
    if deployable:
        id_to_label = {item["experiment_id"]: f"{item.get('algorithm', '').upper()} {item.get('version', '')} · {item['experiment_id']}" for item in deployable}
        chosen = st.selectbox("Model version to deploy", list(id_to_label), format_func=lambda value: id_to_label[value], key=f"deploy_model_select_{station}_{component}")
        st.caption("Deployment updates the station's active model assignment. Stop its predictor first; restart the station pipeline after deployment.")
        if st.button("Deploy selected model", type="primary", key=f"deploy_model_{station}_{component}"):
            try:
                deployment = api_write("POST", "/models/deploy", {"station": station, "experiment_id": chosen})
                st.success(deployment.get("message", "Deployment assignment saved."))
                st.json(deployment)
                get_json.clear()
            except requests.RequestException as exc:
                st.error(f"Deployment was not applied: {exc}")





def _render_renewable_operational_performance(station: str, component: str) -> None:
    """Operational replay metrics; never treat static future test data as live."""
    kind = "solar" if component == "solar_forecast" else "wind"
    section_title("Live/replay forecasting accuracy", "Only matured forecasts are scored against matching generation observations up to the streaming replay watermark.")
    s1, s2 = st.columns([1, 3])
    with s1:
        lead = st.selectbox("Forecast lead time", [1, 4, 24, 96],
                            format_func=lambda n: {1: "15 minutes", 4: "1 hour", 24: "6 hours", 96: "24 hours"}[n],
                            key=f"renewable_lead_{station}_{kind}")
    try:
        payload = get_json(api, "/results/renewables/performance", {"station": station, "horizon_step": lead, "limit": 400})
    except requests.RequestException as exc:
        st.warning(f"Live model analysis unavailable: {readable_http_error(exc)}. Check that FastAPI has the updated results API.")
        return
    section = (payload.get("models") or {}).get(kind) or {}
    metrics = section.get("metrics") or {}
    count = int(metrics.get("matched_points") or 0)
    if count == 0:
        st.info(payload.get("message") or "No matured forecast/actual matches are available for this lead time yet.")
        st.caption("Training validation results above remain available; live MAE cannot be estimated from forecasts that have not reached their target timestamps.")
        return
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Matched points", f"{count:,}")
    m2.metric("MAE", f"{metrics['mae_mwh']:,.1f} MWh")
    m3.metric("RMSE", f"{metrics['rmse_mwh']:,.1f} MWh")
    wape = metrics.get("wape_pct")
    m4.metric("WAPE", f"{wape:,.1f}%" if wape is not None else "N/A")
    st.caption(f"Evaluation through {payload.get('watermark_utc', '—')} · {payload.get('message', '')}")
    frame = pd.DataFrame(section.get("records") or [])
    if frame.empty:
        return
    frame["forecast_timestamp_utc"] = pd.to_datetime(frame["forecast_timestamp_utc"], utc=True, errors="coerce")
    frame = frame.sort_values("forecast_timestamp_utc")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frame["forecast_timestamp_utc"], y=frame["actual_mwh"],
                             name="Observed generation", mode="lines", line=dict(color="#38bdf8", width=2.5)))
    fig.add_trace(go.Scatter(x=frame["forecast_timestamp_utc"], y=frame["predicted_mwh"],
                             name="Saved forecast", mode="lines", line=dict(color="#f5a65b", width=2, dash="dash")))
    dark_layout(fig, 380).update_layout(
        title=f"{kind.title()} · forecast vs observed generation",
        xaxis_title="Forecast target time (UTC)", yaxis_title="Generation per 15-minute interval (MWh)",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    err = frame["predicted_mwh"] - frame["actual_mwh"]
    efig = go.Figure()
    efig.add_trace(go.Scatter(x=frame["forecast_timestamp_utc"], y=err, mode="lines",
                              name="Forecast error", line=dict(color="#a78bfa", width=2)))
    efig.add_hline(y=0, line=dict(color="#a9bfd7", dash="dot"))
    dark_layout(efig, 285).update_layout(title="Forecast error over time", xaxis_title="Target time (UTC)", yaxis_title="Forecast − actual (MWh)")
    st.plotly_chart(efig, use_container_width=True, config={"displayModeBar": False})
    with st.expander("Inspect matched forecast samples", expanded=False):
        st.dataframe(frame.tail(120), hide_index=True, use_container_width=True)


def _render_residual_operational_performance(station: str) -> None:
    from dashboard.residual_plot import residual_figures, residual_summary
    # Align model analysis to the Demand replay origin rather than selecting an
    # unrelated persisted origin from a later historical replay session.
    # This is a read-only forecast request; no neural-network inference.
    current_origin = None
    try:
        forecast_rows = get_json(api, '/results/forecasts',
                                 {'station':station,'limit':96}).get('records', [])
        origins = pd.to_datetime([r.get('forecast_origin_utc') for r in forecast_rows],
                                 utc=True, errors='coerce')
        usable = origins[~pd.isna(origins)]
        if len(usable):
            current_origin = usable.max().isoformat()
    except (requests.RequestException, ValueError, TypeError):
        pass
    request_params = {'station':station,'history_runs':24,'cached_only':True}
    if current_origin:
        request_params.update({'origin_utc':current_origin,'lookback_minutes':60})
    try:
        payload = get_json(api, '/results/residual/forecast', request_params)
    except requests.RequestException as exc:
        st.warning(f'Residual-load performance unavailable: {readable_http_error(exc)}')
        return
    if not payload.get('available'):
        st.info(payload.get('message','Waiting for synchronized residual forecasts.'))
        return
    if payload.get('snapshot_lag_minutes') not in (None, 0):
        st.caption(f'Residual assessment origin: {payload.get("origin_utc")} · '
                   f'{payload.get("snapshot_lag_minutes")} minutes behind latest Demand origin.')
    summary=residual_summary(payload)
    c1,c2,c3,c4=st.columns(4)
    fmt=lambda value: '—' if value is None else f'{value:,.0f}'
    c1.metric('Next residual',fmt(summary['next']))
    c2.metric('24h peak',fmt(summary['peak']))
    c3.metric('Largest 30m ramp',fmt(summary['ramp']))
    c4.metric('Flagged intervals',str(summary['flagged']))
    fig1,fig2=residual_figures(payload)
    st.plotly_chart(fig1,use_container_width=True,config={'displayModeBar':False})
    st.plotly_chart(fig2,use_container_width=True,config={'displayModeBar':False})
    with st.expander('Saved residual-load forecast assessments', expanded=False):
        origins=[{'Forecast origin':item.get('origin_utc'),
                  'Flagged spike intervals':item.get('flagged_points',0)}
                 for item in payload.get('history',[])]
        if origins: st.dataframe(pd.DataFrame(origins), use_container_width=True, hide_index=True)
    st.caption('Forecast-only risk screening; statistical/AE threshold quantiles are not grid incident probabilities.')


def render_component_performance(station: str, component: str) -> None:
    """Report facts supported by the component registry; avoid inventing live accuracy."""
    section_title(
        f"Model analysis · {COMPONENT_LABELS.get(component, component)}",
        "Candidate validation metrics and deployment status for the selected model component.",
    )
    try:
        component_doc = get_json(api, "/models/components", {"station": station})
        info = (component_doc.get("sources") or {}).get(component, {})
        records = get_json(api, "/models/experiments", {"station": station, "component": component}).get("experiments", [])
    except requests.RequestException as exc:
        st.error(f"Could not load model component analysis: {readable_http_error(exc)}")
        return
    render_status_strip([
        ("Component", "Enabled" if info.get("enabled") else "Disabled",
         "good" if info.get("enabled") else "bad"),
        ("Training data", "Found" if info.get("registered_file_exists") else "Missing",
         "good" if info.get("registered_file_exists") else "warn"),
        ("Deployment", "Ready" if info.get("deployment_exists") else "Not ready",
         "good" if info.get("deployment_exists") else "warn"),
    ])
    if info.get("deployment_file"):
        with st.expander("Deployment details", expanded=False):
            st.code(info["deployment_file"], language="text")
    complete = [item for item in records if item.get("status") in ("candidate", "deployed", "previous")]
    if complete:
        rows = []
        for item in complete[:30]:
            scores = item.get("metrics") or {}
            rows.append({
                "Version": item.get("version"), "Algorithm": item.get("algorithm"),
                "Status": item.get("status"),
                "MAE (MWh)": scores.get("mae_mwh") if component != "residual_anomaly" else None,
                "RMSE (MWh)": scores.get("rmse_mwh") if component != "residual_anomaly" else None,
                "WAPE (%)": scores.get("wape_pct"),
                "Validation reconstruction loss": scores.get("best_validation_loss") if component == "residual_anomaly" else None,
                "Dataset": item.get("training_data_id") or "Registered",
                "Experiment": item.get("experiment_id"),
            })
        frame = pd.DataFrame(rows)
        if component == "residual_anomaly":
            frame = frame.drop(columns=["MAE (MWh)", "RMSE (MWh)", "WAPE (%)"])
        else:
            frame = frame.drop(columns=["Validation reconstruction loss"])
        st.dataframe(frame, hide_index=True, use_container_width=True)
    else:
        st.caption("No candidate experiments yet. Existing deployed checkpoints remain available.")
    if component == "residual_anomaly":
        with st.expander("About autoencoder metrics", expanded=False):
            st.caption("Reconstruction loss and thresholds are model diagnostics, not validated incident-detection precision or recall.")
        _render_residual_operational_performance(station)
    else:
        _render_renewable_operational_performance(station, component)


def render_live_performance(station: str, evaluation_window: int) -> None:
    try:
        model = get_json(api, "/models/status", {"station": station})
        status = get_json(api, f"/stations/{station}/status")
    except requests.RequestException as exc:
        st.error(f"Could not retrieve model details: {exc}")
        st.stop()

    metadata = model.get("metadata") or {}
    interval_minutes = int(status.get("interval_minutes", 15))
    horizon_steps = int(status.get("horizon_steps", metadata.get("horizon", metadata.get("horizon_steps", 96))))
    training = status.get("services", {}).get("training", {})
    ready = bool(model.get("checkpoint_exists"))
    if training.get("running"):
        lifecycle_label = "Training in progress"
    elif ready:
        lifecycle_label = "Model ready"
    else:
        lifecycle_label = "Model not trained"

    section_title("Model status", "Model assets and training information are read from the selected station's registry and metadata.")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Model artifact", "Available" if ready else "Missing")
    c2.metric("Algorithm", str(model.get("model_algorithm") or metadata.get("algorithm", "LSTM")).upper())
    c3.metric("Lifecycle", status.get("lifecycle_status", "unknown").replace("_", " ").title())
    c4.metric("Version", str(model.get("model_version") or "—"))
    c5.metric("Training process", "Running" if training.get("running") else ("Completed" if training.get("return_code") == 0 else "Stopped"))

    if ready:
        st.success(lifecycle_label)
    else:
        st.warning(lifecycle_label)
        st.page_link("pages/station_management.py", label="Go to Station Management to train this station", icon="📡")

    # Training-time validation metrics remain separate from operational accuracy.
    if metadata:
        section_title("Offline training validation", "Metrics recorded during model training. These are not live streaming accuracy metrics.")
        rmse_scaled = metadata.get("best_validation_rmse_scaled")
        mae_original = metadata.get("best_validation_mae_original")
        best_epoch = metadata.get("best_epoch", "—")
        t1, t2, t3 = st.columns(3)
        t1.metric("Best validation RMSE (scaled)", f"{rmse_scaled:.5f}" if isinstance(rmse_scaled, (int, float)) else "—")
        t2.metric("Validation MAE", fmt_mw(mae_original))
        t3.metric("Best epoch", str(best_epoch))

        history = metadata.get("training_history", [])
        if history:
            train_frame = pd.DataFrame(history)
            required = {"epoch", "train_rmse_scaled", "validation_rmse_scaled"}
            if required.issubset(train_frame.columns):
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=train_frame["epoch"], y=train_frame["train_rmse_scaled"], mode="lines+markers", name="Train RMSE"))
                fig.add_trace(go.Scatter(x=train_frame["epoch"], y=train_frame["validation_rmse_scaled"], mode="lines+markers", name="Validation RMSE"))
                dark_layout(fig, 300).update_layout(title="Training history", xaxis_title="Epoch", yaxis_title="Scaled RMSE")
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("Model metadata is not available yet. Train the station to create a checkpoint and metadata file.")

    # Load operational result data from the existing station-filtered result APIs.
    section_title(
        "Live model performance",
        f"Rolling {interval_minutes}-minute-ahead evaluation using horizon step 1. "
        "Each score is calculated only after the corresponding actual demand is available."
    )
    try:
        forecast_payload = get_json(api, "/results/forecasts", {"station": station, "limit": 20000})
        actual_payload = get_json(api, "/results/actuals", {"station": station, "limit": 20000})
        forecasts = pd.DataFrame(forecast_payload.get("records", []))
        actuals = pd.DataFrame(actual_payload.get("records", []))
    except requests.RequestException as exc:
        st.error(f"Could not load saved actuals/forecasts for performance evaluation: {exc}")
        forecasts = pd.DataFrame()
        actuals = pd.DataFrame()

    evaluated = build_evaluation_frame(forecasts, actuals)

    if evaluated.empty:
        st.info(
            "No forecast targets have matching realized actual demand yet. "
            "Keep the station pipeline running until later observations arrive; "
            "the performance panel will populate automatically."
        )
    else:
        # One-step forecasts represent the station's configured observation
        # interval. Keep only the latest saved prediction if a target is duplicated.
        one_step = evaluated[evaluated["horizon_step"] == 1].copy()
        if not one_step.empty:
            if "forecast_issued_at_utc" in one_step.columns:
                one_step = one_step.sort_values("forecast_issued_at_utc")
            one_step = one_step.drop_duplicates("forecast_timestamp_utc", keep="last")
            latest_actual_time = evaluated["actual_timestamp_utc"].max()
            if evaluation_window > 0:
                window_start = latest_actual_time - pd.Timedelta(hours=int(evaluation_window))
                live_sample = one_step[one_step["actual_timestamp_utc"] >= window_start]
            else:
                live_sample = one_step
            live_metrics = calculate_metrics(live_sample)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric(f"{interval_minutes}-minute MAE", fmt_mw(live_metrics["mae_mw"]))
            m2.metric(f"{interval_minutes}-minute RMSE", fmt_mw(live_metrics["rmse_mw"]))
            m3.metric(f"{interval_minutes}-minute WAPE", fmt_pct(live_metrics["wape_pct"]))
            m4.metric("Evaluated predictions", f"{live_metrics['matched_observations']:,}")
            st.caption(
                f"Actual-data watermark: {fmt_utc(latest_actual_time)} · "
                f"Window: {'all available data' if evaluation_window == 0 else f'last {evaluation_window} hours'} · "
                "Lower MAE/RMSE/WAPE is better; positive bias means over-forecasting."
            )

            if not live_sample.empty:
                live_chart = live_sample.sort_values("actual_timestamp_utc").tail(500)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=live_chart["actual_timestamp_utc"], y=live_chart["actual_demand"], mode="lines", name="Actual demand", line=dict(width=2.4, color="#38bdf8")))
                fig.add_trace(go.Scatter(x=live_chart["actual_timestamp_utc"], y=live_chart["predicted_demand"], mode="lines", name=f"{interval_minutes}-minute-ahead forecast", line=dict(width=2.2, dash="dash", color="#a78bfa")))
                dark_layout(fig, 340).update_layout(title=f"Recent {interval_minutes}-minute-ahead forecast vs actual", xaxis_title="Realized target timestamp (UTC)", yaxis_title="Demand (MW)")
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info(f"No realized horizon-step-1 predictions are available yet. The first {interval_minutes}-minute-ahead forecast will be scored when its target actual arrives.")

        # Historical full-horizon scores include only forecast runs with at least
        # 80% of their target actuals observed, avoiding misleading partial runs.
        model_horizon = int(metadata.get("horizon", metadata.get("horizon_steps", horizon_steps))) if metadata else horizon_steps
        run_summary = metrics_by_run(evaluated, model_horizon)
        matured_runs = run_summary[run_summary["matured"]].copy() if not run_summary.empty else pd.DataFrame()

        section_title("Historical forecasting accuracy", "Per-run scores are included only when at least 80% of the forecast horizon has matching actual demand. Forecasts whose target actuals have not arrived are excluded.")
        if matured_runs.empty:
            st.info(
                f"No forecast run has reached the {int(model_horizon * 0.80 + 0.999)}-of-{model_horizon} "
                "matched-observation threshold yet. Historical full-horizon accuracy will appear as replay/stream actuals catch up."
            )
        else:
            latest_complete = matured_runs.sort_values("run_origin_utc").iloc[-1]
            h1, h2, h3, h4 = st.columns(4)
            h1.metric("Latest matured-run MAE", fmt_mw(latest_complete["mae_mw"]))
            h2.metric("Latest matured-run RMSE", fmt_mw(latest_complete["rmse_mw"]))
            h3.metric("Latest matured-run WAPE", fmt_pct(latest_complete["wape_pct"]))
            h4.metric("Matched horizon steps", f"{int(latest_complete['matched_observations']):,}/{model_horizon}")

            trend = matured_runs.sort_values("run_origin_utc").tail(300)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=trend["run_origin_utc"], y=trend["mae_mw"], mode="lines+markers", name="Run MAE", line=dict(color="#38bdf8", width=2), marker=dict(size=4)))
            fig.add_trace(go.Scatter(x=trend["run_origin_utc"], y=trend["rmse_mw"], mode="lines", name="Run RMSE", line=dict(color="#a78bfa", width=2)))
            dark_layout(fig, 350).update_layout(title="Historical full-horizon error by forecast origin", xaxis_title="Forecast origin (UTC)", yaxis_title="Error (MW)")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            horizon_profile = metrics_by_horizon(evaluated, run_summary)
            if not horizon_profile.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=horizon_profile["horizon_step"], y=horizon_profile["mae_mw"], mode="lines+markers", name="MAE by lead time", line=dict(color="#f5a65b", width=2.5), marker=dict(size=5)))
                dark_layout(fig, 330).update_layout(title="Historical error by forecast lead time", xaxis_title=f"Forecast horizon step ({interval_minutes}-minute intervals)", yaxis_title="MAE (MW)")
                fig.update_xaxes(dtick=8)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            # Compare an actual-vs-predicted trajectory for a completed saved run.
            origins = sorted(matured_runs["run_origin_utc"].dropna().unique(), reverse=True)
            origin_labels = [pd.Timestamp(value).strftime("%Y-%m-%d %H:%M UTC") for value in origins]
            selected_label = st.selectbox("Inspect a matured forecast run", origin_labels, key=f"model_run_{station}")
            selected_origin = pd.Timestamp(origins[origin_labels.index(selected_label)])
            selected_run = evaluated[evaluated["run_origin_utc"] == selected_origin].sort_values("forecast_timestamp_utc")
            if not selected_run.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=selected_run["forecast_timestamp_utc"], y=selected_run["actual_demand"], mode="lines+markers", name="Actual demand", line=dict(color="#38bdf8", width=2.2)))
                fig.add_trace(go.Scatter(x=selected_run["forecast_timestamp_utc"], y=selected_run["predicted_demand"], mode="lines+markers", name="Predicted demand", line=dict(color="#a78bfa", width=2.2, dash="dash")))
                dark_layout(fig, 350).update_layout(title=f"Actual vs predicted — origin {selected_label}", xaxis_title="Forecast target timestamp (UTC)", yaxis_title="Demand (MW)")
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            display = matured_runs.sort_values("run_origin_utc", ascending=False).head(50).copy()
            display["Forecast origin (UTC)"] = display["run_origin_utc"].dt.strftime("%Y-%m-%d %H:%M")
            display = display.rename(columns={
                "matched_observations": "Matched steps",
                "coverage_pct": "Coverage %",
                "mae_mw": "MAE (MW)",
                "rmse_mw": "RMSE (MW)",
                "wape_pct": "WAPE %",
                "bias_mw": "Bias (MW)",
            })
            cols = ["Forecast origin (UTC)", "Matched steps", "Coverage %", "MAE (MW)", "RMSE (MW)", "WAPE %", "Bias (MW)"]
            st.dataframe(display[cols].round(3), use_container_width=True, hide_index=True)
            st.caption("History shown is based on the latest 20,000 forecast records and latest 20,000 actual records exposed by the results API. This may be a rolling subset, not the entire lifetime archive.")

    # Model assets and process details.
    section_title("Model artifacts", "File locations and process status are resolved by FastAPI from the selected station configuration.")
    st.code(
        f"Checkpoint\n{model.get('checkpoint_path', 'Not configured')}\n\n"
        f"Metadata\n{model.get('metadata_path', 'Not configured')}",
        language="text",
    )
    with st.expander("Training process details"):
        st.json(model.get("training_process", {}))
    with st.expander("Model metadata", expanded=False):
        if metadata:
            st.json(metadata)
        else:
            st.info("No model metadata is available yet.")

    st.caption(
        "Operational evaluation compares saved forecast targets with realized actual demand at matching UTC timestamps. "
        f"The {interval_minutes}-minute metric uses horizon step 1; historical full-horizon charts use runs with at least 80% realized targets."
    )


def render_model_analysis_panel(station: str, component: str, evaluation_window: int) -> None:
    """Live-only panel: fragment refresh must not rerun training/registry forms."""
    if component == "demand_forecast":
        render_live_performance(station, evaluation_window)
    else:
        render_component_performance(station, component)


st.title("🧠 Model Management & Model Lab")


try:
    station_registry = get_json(api, "/stations")
except requests.RequestException as exc:
    st.error(f"Cannot load station registry from FastAPI: {exc}")
    st.stop()
stations = station_registry.get("stations", {})
station_ids = list(stations)
if not station_ids:
    st.info("No stations are registered yet. Start at Station Management to add one.")
    st.page_link("pages/station_management.py", label="Open Station Management", icon="📡")
    st.stop()
active_station = station_registry.get("active_station")
station_index = station_ids.index(active_station) if active_station in station_ids else 0
station = st.selectbox(
    "Station workspace", station_ids, index=station_index,
    format_func=lambda key: f"{stations[key].get('display_name', key)} ({key})",
)
station_label = stations[station].get("display_name", station)
try:
    component_payload = get_json(api, "/models/components", {"station": station})
except requests.RequestException as exc:
    st.error(f"Could not load component definitions: {readable_http_error(exc)}")
    st.stop()

# One common component selector drives all four tabs, including comparison,
# registry/deployment and component analysis (not demand-only by default).
enabled_components = component_payload.get("enabled") or {}
selected_component = st.selectbox(
    "Model component to manage / retrain", list(COMPONENT_LABELS),
    format_func=lambda comp: (
        COMPONENT_LABELS[comp] + (" · disabled" if not enabled_components.get(comp, False) else "")
    ),
    key=f"training_component_{station}",
)

# Streamlit executes *every* st.tabs body even if hidden. The old UI fetched
# large monitoring histories and experiment registries while users were simply
# editing training parameters. Render only the section the customer selected.
sections = [
    "Configure & Retrain", "Compare Models", "Registry & Deployment", "Model Analysis",
]
active_section = st.radio(
    "Model management workspace", sections, horizontal=True,
    key=f"model_workspace_section_{station}",
    help="Only the selected section loads data. This keeps the training screen responsive.",
)
if active_section == "Configure & Retrain":
    render_model_lab(station, station_label, selected_component, component_payload)
elif active_section == "Compare Models":
    render_comparison(station, selected_component)
elif active_section == "Registry & Deployment":
    render_registry_and_deployment(station, selected_component)
else:
    refresh_col, interval_col, button_col = st.columns([1.7, 2.0, 1.2])
    with refresh_col:
        # Versioned keys ensure existing sessions pick up the new defaults once.
        auto_refresh = st.checkbox("Auto-refresh model analysis", value=True,
                                   key=f"model_analysis_auto_2s_{station}")
    with interval_col:
        refresh_seconds = st.selectbox("Refresh every", [2, 5, 10, 15, 30], index=0,
                                       format_func=lambda n: f"{n} seconds", disabled=not auto_refresh,
                                       key=f"model_analysis_interval_2s_{station}")
    with button_col:
        if st.button("Refresh analysis", key=f"model_analysis_refresh_{station}"):
            get_json.clear()
            st.rerun()
    if selected_component == "demand_forecast":
        evaluation_window = st.selectbox("Live evaluation window", [24, 72, 168, 0], index=0,
                                          format_func=lambda hours: "All available matched data" if hours == 0 else f"Last {hours} hours")
    else:
        evaluation_window = 24
    # Refresh just the analysis panel, not training controls, station selection,
    # configuration forms or the application shell. A slow request may delay a tick.
    analysis_panel = st.fragment(run_every=float(refresh_seconds) if auto_refresh else None)(
        render_model_analysis_panel
    )
    analysis_panel(station, selected_component, evaluation_window)
