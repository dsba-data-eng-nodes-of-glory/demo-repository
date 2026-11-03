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
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None

try:
    from dashboard.ui import setup_page, section_title
    from dashboard.model_metrics import (
        build_evaluation_frame, calculate_metrics, metrics_by_horizon, metrics_by_run,
    )
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from dashboard.ui import setup_page, section_title
    from dashboard.model_metrics import (
        build_evaluation_frame, calculate_metrics, metrics_by_horizon, metrics_by_run,
    )

api = setup_page("Model Management", "models")


@st.cache_data(ttl=3, show_spinner=False)
def get_json(api_url: str, path: str, params: dict | None = None) -> dict:
    response = requests.get(f"{api_url.rstrip('/')}{path}", params=params, timeout=25)
    response.raise_for_status()
    return response.json()


def api_write(method: str, path: str, payload: dict) -> dict:
    response = requests.request(method, f"{api.rstrip('/')}{path}", json=payload, timeout=60)
    response.raise_for_status()
    return response.json()


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


def render_model_lab(station: str, station_label: str) -> None:
    section_title("Model training workspace", "Choose an algorithm, edit its parameters in the compact configuration window, then launch a versioned experiment.")
    try:
        config_payload = get_json(api, "/models/config")
        algorithms_payload = get_json(api, "/models/algorithms")
        station_status = get_json(api, f"/stations/{station}/status")
    except requests.RequestException as exc:
        st.error(f"Could not load model configuration or station status: {exc}")
        return

    algorithms = config_payload.get("algorithms", {})
    saved_training_options_from_api = config_payload.get("training_options", {"validation_fraction": 0.15})
    algorithm_info = {item["key"]: item for item in algorithms_payload.get("algorithms", [])}
    # Hydrate the page from server-persisted YAML so a new browser session can
    # use the saved configuration without relying on prior Streamlit state.
    active_model = station_status.get("active_model") or {}
    initial_algorithm = active_model.get("algorithm", "lstm")
    if initial_algorithm not in algorithms:
        initial_algorithm = "lstm" if "lstm" in algorithms else next(iter(algorithms), None)
    if "saved_model_algorithm" not in st.session_state and initial_algorithm:
        st.session_state["saved_model_algorithm"] = initial_algorithm
        st.session_state["saved_model_parameters"] = dict(algorithms.get(initial_algorithm, {}))
        st.session_state["saved_training_options"] = dict(saved_training_options_from_api)
    if "model_lab_algorithm" not in st.session_state and initial_algorithm:
        st.session_state["model_lab_algorithm"] = initial_algorithm
    label_to_key = {item.get("label", key): key for key, item in algorithm_info.items()}
    labels = list(label_to_key)
    if not labels:
        st.error("No model algorithms are configured.")
        return

    with st.container(border=True):
        left, right = st.columns([3, 1], vertical_alignment="center")
        with left:
            st.markdown(f"**Training station:** {station_label} (`{station}`)")
            interval_minutes = int(station_status.get("interval_minutes", 15))
            history_steps = int(station_status.get("history_steps", 672))
            horizon_steps = int(station_status.get("horizon_steps", 96))
            st.caption(
                f"Forecast contract: {history_steps:,} history steps → {horizon_steps:,} future steps "
                f"· {interval_minutes}-minute interval · 7-day lookback / 24-hour horizon"
            )
        with right:
            selected_key = st.session_state.get("model_lab_algorithm", "lstm")
            st.metric("Training status", "Running" if station_status.get("services", {}).get("training", {}).get("running") else "Idle")

        # Compact popover: model dropdown and editable algorithm-specific parameters.
        with st.popover("⚙ Configure model", use_container_width=True):
            current_label = algorithm_info.get(selected_key, {}).get("label", labels[0])
            label_index = labels.index(current_label) if current_label in labels else 0
            selected_label = st.selectbox("Machine learning model", labels, index=label_index, key="model_lab_algorithm_select")
            algorithm = label_to_key[selected_label]
            st.session_state["model_lab_algorithm"] = algorithm
            st.caption(algorithm_info.get(algorithm, {}).get("description", ""))
            defaults = algorithms.get(algorithm, {})
            edited = {}
            for parameter_name, default_value in defaults.items():
                pretty = parameter_name.replace("_", " ").title()
                edited[parameter_name] = _parameter_widget(
                    pretty, default_value,
                    key=f"model_cfg_{station}_{algorithm}_{parameter_name}",
                    parameter_name=parameter_name,
                )
            st.divider()
            validation_fraction = st.slider(
                "Chronological validation fraction", 0.05, 0.40, 0.15, 0.05,
                key=f"model_validation_fraction_{station}",
            )
            if st.button("Save configuration", type="primary", use_container_width=True, key=f"save_model_config_{station}"):
                try:
                    saved = api_write("PUT", f"/models/config/{algorithm}", {
                        "parameters": edited,
                        "training_options": {"validation_fraction": validation_fraction},
                    })
                    st.session_state["saved_model_algorithm"] = algorithm
                    st.session_state["saved_model_parameters"] = saved.get("parameters", edited)
                    st.session_state["saved_training_options"] = saved.get(
                        "training_options", {"validation_fraction": validation_fraction}
                    )
                    get_json.clear()
                    st.success(f"{selected_label} configuration saved.")
                except requests.RequestException as exc:
                    st.error(f"Could not save configuration: {exc}")

        saved_algorithm = st.session_state.get("saved_model_algorithm")
        saved_params = st.session_state.get("saved_model_parameters")
        if saved_algorithm and saved_params:
            saved_label = algorithm_info.get(saved_algorithm, {}).get("label", saved_algorithm)
            st.caption(f"Saved configuration: **{saved_label}** · {len(saved_params)} parameters")
        else:
            st.info("Open Configure model, choose an algorithm, edit its parameters and save the configuration.")

        training_running = bool(station_status.get("services", {}).get("training", {}).get("running"))
        editor_algorithm = st.session_state.get("model_lab_algorithm", "lstm")
        config_matches_editor = saved_algorithm == editor_algorithm
        if saved_algorithm and not config_matches_editor:
            st.warning("The selected algorithm has changed. Save its edited configuration before starting training.")
        run_col, info_col = st.columns([1, 2], vertical_alignment="center")
        with run_col:
            run_disabled = training_running or not bool(saved_algorithm and saved_params and config_matches_editor)
            if st.button("▶ Run training experiment", type="primary", disabled=run_disabled, use_container_width=True, key=f"run_model_experiment_{station}"):
                try:
                    result = api_write("POST", "/models/experiments/train", {
                        "station": station,
                        "algorithm": saved_algorithm,
                        "parameters": saved_params,
                        "training_options": st.session_state.get("saved_training_options", saved_training_options_from_api),
                    })
                    st.success(f"Training started: {result.get('experiment_id')}")
                    st.session_state["last_model_experiment_id"] = result.get("experiment_id")
                    get_json.clear()
                except requests.RequestException as exc:
                    st.error(f"Training could not be started: {exc}")
        with info_col:
            st.caption("Training creates a candidate model version. It does not replace the model currently serving forecasts.")

    try:
        experiments = get_json(api, "/models/experiments", {"station": station}).get("experiments", [])
    except requests.RequestException as exc:
        st.warning(f"Experiment list is temporarily unavailable: {exc}")
        return
    if experiments:
        st.markdown("**Recent experiments**")
        rows = []
        for item in experiments[:8]:
            metrics = item.get("metrics") or {}
            rows.append({
                "Experiment": item.get("experiment_id"),
                "Algorithm": item.get("algorithm"),
                "Version": item.get("version", "—"),
                "Status": item.get("status"),
                "MAE (MW)": metrics.get("mae_mw"),
                "RMSE (MW)": metrics.get("rmse_mw"),
                "WAPE (%)": metrics.get("wape_pct"),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.caption("No registered experiments yet. Your existing deployed LSTM remains unchanged.")


def render_comparison(station: str) -> None:
    try:
        station_status = get_json(api, f"/stations/{station}/status")
        interval_minutes = int(station_status.get("interval_minutes", 15))
        history_steps = int(station_status.get("history_steps", 672))
        horizon_steps = int(station_status.get("horizon_steps", 96))
        records = get_json(api, "/models/experiments", {"station": station}).get("experiments", [])
    except requests.RequestException as exc:
        st.error(f"Could not load experiment registry or station forecast contract: {exc}")
        return
    section_title(
        "Compare experiments",
        f"Compare 2–4 completed experiments using the same station, chronological holdout and forecast contract "
        f"({history_steps:,} history steps, {horizon_steps:,} forecast steps at {interval_minutes}-minute intervals).",
    )
    complete = [item for item in records if item.get("status") in ("candidate", "deployed", "previous")]
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
        key=f"model_compare_selection_{station}",
    )
    if len(selected) > 4:
        st.warning("Select no more than four experiments at a time.")
    if st.button("Compare selected models", type="primary", disabled=not 2 <= len(selected) <= 4, key=f"compare_models_{station}"):
        try:
            result = api_write("POST", "/models/compare", {"experiment_ids": selected})
            st.session_state[f"model_compare_result_{station}"] = result
        except requests.RequestException as exc:
            st.error(f"Comparison failed: {exc}")

    result = st.session_state.get(f"model_compare_result_{station}")
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
    display_columns = [
        "model", "status", "MAE (MW)", "RMSE (MW)", "WAPE (%)", "Bias (MW)",
        "Training time (s)", "Inference latency (ms/origin)", "Model size (bytes)",
    ]
    visible = frame[[column for column in display_columns if column in frame.columns]].copy()
    st.dataframe(visible.round(4), use_container_width=True, hide_index=True)

    metric_choice = st.selectbox("Compare metric", ["MAE (MW)", "RMSE (MW)", "WAPE (%)", "Training time (s)", "Inference latency (ms/origin)"], key=f"comparison_metric_{station}")
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
            })

    horizon_rows = []
    for row in results:
        for point in row.get("horizon_metrics", []):
            horizon_rows.append({"model": row["model"], **point})
    if horizon_rows:
        horizon_frame = pd.DataFrame(horizon_rows)
        horizon_frame["lead_minutes"] = horizon_frame["horizon_step"].astype(float) * interval_minutes
        horizon_metric = st.selectbox("Horizon metric", ["mae_mw", "rmse_mw", "wape_pct"], format_func=lambda value: value.replace("_", " ").upper(), key=f"horizon_compare_metric_{station}")
        fig = go.Figure()
        for model_name, group in horizon_frame.groupby("model"):
            fig.add_trace(go.Scatter(x=group["lead_minutes"], y=group[horizon_metric], mode="lines", name=model_name))
        dark_layout(fig, 340).update_layout(
            title=f"Forecast error by lead time · {horizon_metric}",
            xaxis_title="Forecast lead time (minutes)", yaxis_title=horizon_metric,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_registry_and_deployment(station: str) -> None:
    section_title("Model registry", "Trained artifacts, configuration, performance metrics and deployment lifecycle for the selected station.")
    try:
        payload = get_json(api, "/models/registry", {"station": station})
        status = get_json(api, "/models/status", {"station": station})
    except requests.RequestException as exc:
        st.error(f"Could not load model registry: {exc}")
        return
    records = payload.get("experiments", [])
    active = status.get("active_model") or {}
    if active:
        st.success(f"Active deployment: {active.get('algorithm', '').upper()} {active.get('version', '')} · {active.get('experiment_id')}")
    elif status.get("checkpoint_exists"):
        st.info(f"Legacy model currently configured: {status.get('model_checkpoint', 'LSTM checkpoint')}")
    else:
        st.warning("No model artifact is currently deployed for this station.")
    if records:
        rows = []
        for item in records:
            metrics = item.get("metrics") or {}
            rows.append({
                "Experiment ID": item.get("experiment_id"),
                "Algorithm": item.get("algorithm"),
                "Version": item.get("version", "—"),
                "Status": item.get("status"),
                "MAE (MW)": metrics.get("mae_mw"),
                "RMSE (MW)": metrics.get("rmse_mw"),
                "WAPE (%)": metrics.get("wape_pct"),
                "Created": item.get("created_at"),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No registered candidate experiments yet.")

    deployable = [item for item in records if item.get("status") in ("candidate", "deployed", "previous")]
    if deployable:
        id_to_label = {item["experiment_id"]: f"{item.get('algorithm', '').upper()} {item.get('version', '')} · {item['experiment_id']}" for item in deployable}
        chosen = st.selectbox("Model version to deploy", list(id_to_label), format_func=lambda value: id_to_label[value], key=f"deploy_model_select_{station}")
        st.caption("Deployment updates the station's active model assignment. Stop its predictor first; restart the station pipeline after deployment.")
        if st.button("Deploy selected model", type="primary", key=f"deploy_model_{station}"):
            try:
                deployment = api_write("POST", "/models/deploy", {"station": station, "experiment_id": chosen})
                st.success(deployment.get("message", "Deployment assignment saved."))
                st.json(deployment)
                get_json.clear()
            except requests.RequestException as exc:
                st.error(f"Deployment was not applied: {exc}")





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


st.title("🧠 Model Management & Model Lab")
st.caption("Configure algorithms, run forecasting experiments, compare results, register versions and deploy an approved model.")

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

train_tab, compare_tab, registry_tab, live_tab = st.tabs([
    "⚙ Configure & Train", "⇄ Compare Models", "▣ Model Registry & Deployment", "◉ Live Performance",
])
with train_tab:
    render_model_lab(station, station_label)
with compare_tab:
    render_comparison(station)
with registry_tab:
    render_registry_and_deployment(station)
with live_tab:
    with st.sidebar:
        st.subheader("Live model analytics")
        auto_refresh = st.checkbox("Auto-refresh live performance", value=True)
        refresh_seconds = st.selectbox("Refresh interval", [2, 5, 10, 15], index=1, format_func=lambda seconds: f"Every {seconds} seconds", disabled=not auto_refresh)
        if auto_refresh and st_autorefresh is not None:
            st_autorefresh(interval=refresh_seconds * 1000, key="model_performance_refresh")
        elif auto_refresh:
            st.caption("Install streamlit-autorefresh for automatic refresh.")
        if st.button("Refresh analytics", use_container_width=True):
            get_json.clear()
            st.rerun()
        evaluation_window = st.selectbox("Live evaluation window", [24, 72, 168, 0], index=0, format_func=lambda hours: "All available matched data" if hours == 0 else f"Last {hours} hours")
    render_live_performance(station, evaluation_window)

