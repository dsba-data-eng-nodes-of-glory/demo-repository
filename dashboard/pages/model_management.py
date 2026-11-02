"""Station-specific model readiness, live evaluation and forecast accuracy analytics."""
from __future__ import annotations

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
        build_evaluation_frame,
        calculate_metrics,
        metrics_by_horizon,
        metrics_by_run,
    )
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from dashboard.ui import setup_page, section_title
    from dashboard.model_metrics import (
        build_evaluation_frame,
        calculate_metrics,
        metrics_by_horizon,
        metrics_by_run,
    )


api = setup_page("Model Management", "models")


@st.cache_data(ttl=5, show_spinner=False)
def get_json(api_url: str, path: str, params: dict | None = None) -> dict:
    response = requests.get(f"{api_url.rstrip('/')}{path}", params=params, timeout=25)
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
        template="plotly_dark",
        height=height,
        margin=dict(l=12, r=12, t=55, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(7,20,39,.55)",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.12, x=0),
    )
    return fig


st.title("🧠 Model Management")
st.caption(
    "Model readiness, training validation and rolling forecast evaluation. "
    "Operational accuracy is computed only where forecast targets have matching realized demand."
)

with st.sidebar:
    st.subheader("Model analytics")
    auto_refresh = st.checkbox("Auto-refresh live performance", value=True)
    refresh_seconds = st.selectbox(
        "Refresh interval", [2, 5, 10, 15], index=1,
        format_func=lambda seconds: f"Every {seconds} seconds",
        disabled=not auto_refresh,
    )
    if auto_refresh:
        if st_autorefresh is not None:
            st_autorefresh(interval=refresh_seconds * 1000, key="model_performance_refresh")
        else:
            st.caption("For automatic refresh, install `streamlit-autorefresh`. Use Refresh now otherwise.")
    if st.button("Refresh now", use_container_width=True):
        get_json.clear()
        st.rerun()
    st.divider()
    evaluation_window = st.selectbox(
        "Live evaluation window", [24, 72, 168, 0], index=0,
        format_func=lambda hours: "All available matched data" if hours == 0 else f"Last {hours} hours",
    )

try:
    registry = get_json(api, "/stations")
except requests.RequestException as exc:
    st.error(f"Cannot load station registry from FastAPI: {exc}")
    st.stop()

stations = registry.get("stations", {})
station_ids = list(stations)
if not station_ids:
    st.info("No stations are registered yet. Start at Station Management to add one.")
    st.page_link("pages/station_management.py", label="Open Station Management", icon="📡")
    st.stop()

active = registry.get("active_station")
index = station_ids.index(active) if active in station_ids else 0
station = st.selectbox(
    "Select station", station_ids, index=index,
    format_func=lambda key: stations[key].get("display_name", key),
)

try:
    model = get_json(api, "/models/status", {"station": station})
    status = get_json(api, f"/stations/{station}/status")
except requests.RequestException as exc:
    st.error(f"Could not retrieve model details: {exc}")
    st.stop()

metadata = model.get("metadata") or {}
training = status.get("services", {}).get("training", {})
ready = bool(model.get("checkpoint_exists"))
if training.get("running"):
    lifecycle_label = "Training in progress"
elif ready:
    lifecycle_label = "Model ready"
else:
    lifecycle_label = "Model not trained"

section_title("Model status", "Model assets and training information are read from the selected station's registry and metadata.")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Model checkpoint", "Available" if ready else "Missing")
c2.metric("Lifecycle", status.get("lifecycle_status", "unknown").replace("_", " ").title())
c3.metric("Version", str(model.get("model_version") or "—"))
c4.metric("Training process", "Running" if training.get("running") else ("Completed" if training.get("return_code") == 0 else "Stopped"))

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
section_title("Live model performance", "Rolling 15-minute-ahead evaluation using horizon step 1. Each score is calculated only after the corresponding actual demand is available.")
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
    # One-step forecasts are a clean operational metric for a 15-minute data
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
        m1.metric("15-minute MAE", fmt_mw(live_metrics["mae_mw"]))
        m2.metric("15-minute RMSE", fmt_mw(live_metrics["rmse_mw"]))
        m3.metric("15-minute WAPE", fmt_pct(live_metrics["wape_pct"]))
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
            fig.add_trace(go.Scatter(x=live_chart["actual_timestamp_utc"], y=live_chart["predicted_demand"], mode="lines", name="15-minute-ahead forecast", line=dict(width=2.2, dash="dash", color="#a78bfa")))
            dark_layout(fig, 340).update_layout(title="Recent 15-minute-ahead forecast vs actual", xaxis_title="Realized target timestamp (UTC)", yaxis_title="Demand (MW)")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("No realized horizon-step-1 predictions are available yet. The first 15-minute-ahead forecast will be scored when its target actual arrives.")

    # Historical full-horizon scores include only forecast runs with at least
    # 80% of their target actuals observed, avoiding misleading partial runs.
    model_horizon = int(metadata.get("horizon", 96)) if metadata else 96
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
            dark_layout(fig, 330).update_layout(title="Historical error by forecast lead time", xaxis_title="Forecast horizon step (15-minute intervals)", yaxis_title="MAE (MW)")
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
    "The 15-minute metric uses horizon step 1; historical full-horizon charts use runs with at least 80% realized targets."
)
