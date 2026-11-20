"""Electricity demand monitoring: actual history connected to the selected model forecast."""

from datetime import date, timedelta
from pathlib import Path
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

try:
    from dashboard.ui import setup_page, render_status_strip
    from dashboard.hierarchy_monitoring import METRICS, render_group
    from src.storage.hierarchy_rollup import demand_to_mw_factor
    from dashboard.station_tree import select_monitoring_node
    from dashboard.renewable_plot import build_renewable_figure
    from dashboard.renewable_summary import build_renewable_summaries
    from dashboard.forecast_selection import select_latest_complete_forecast_run
    from dashboard.residual_plot import residual_figures, residual_summary
    from dashboard.replay_display import filter_actuals_to_cutoff, read_last_acked_timestamp, resolve_actuals_display_cutoff
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from dashboard.ui import setup_page, render_status_strip
    from dashboard.hierarchy_monitoring import METRICS, render_group
    from src.storage.hierarchy_rollup import demand_to_mw_factor
    from dashboard.station_tree import select_monitoring_node
    from dashboard.renewable_plot import build_renewable_figure
    from dashboard.renewable_summary import build_renewable_summaries
    from dashboard.forecast_selection import select_latest_complete_forecast_run
    from dashboard.residual_plot import residual_figures, residual_summary
    from dashboard.replay_display import filter_actuals_to_cutoff, read_last_acked_timestamp, resolve_actuals_display_cutoff

from dashboard.monitoring_chart import ACTUAL_COLOR, FORECAST_COLOR, apply_monitoring_axes, mark_now, render_initializing
from dashboard.monitoring_progress import forecast_cursor, timeline_is_live
from dashboard.monitoring_units import interval_unit_label, unit_factor

@st.cache_data(ttl=1, show_spinner=False)
def api_get(api_url: str, endpoint: str, params: dict | None = None) -> dict:
    response = requests.get(
        f"{api_url.rstrip('/')}{endpoint}", params=params, timeout=(1.5, 12 if endpoint == "/results/renewables" else 5)
    )
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=60, show_spinner=False)
def api_get_station_registry(api_url: str) -> dict:
    response = requests.get(f"{api_url.rstrip('/')}/stations", timeout=(1.5, 5))
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=15, show_spinner=False)
def api_get_readiness(api_url: str) -> dict:
    response = requests.get(f"{api_url.rstrip('/')}/health/ready", timeout=(1.5, 5))
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=30, show_spinner=False)
def api_get_quality_events(api_url: str, station: str, limit: int = 100) -> dict:
    response = requests.get(
        f"{api_url.rstrip('/')}/results/quality/events",
        params={"station": station, "limit": limit},
        timeout=(1.5, 5),
    )
    response.raise_for_status()
    return response.json()



def as_utc(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    if frame.empty or column not in frame.columns:
        return pd.DataFrame()
    result = frame.copy()
    result[column] = pd.to_datetime(result[column], utc=True, errors="coerce")
    return (
        result.dropna(subset=[column])
        .sort_values(column)
        .reset_index(drop=True)
    )


def number_column(frame: pd.DataFrame, column: str, factor: float = 1.0) -> pd.DataFrame:
    if frame.empty or column not in frame.columns:
        return pd.DataFrame()
    result = frame.copy()
    result[column] = pd.to_numeric(result[column], errors="coerce") * factor
    return result.dropna(subset=[column])


def fmt_utc(value) -> str:
    if value is None or pd.isna(value):
        return "—"
    return pd.to_datetime(value, utc=True).strftime("%Y-%m-%d %H:%M UTC")


# ---------------------------------------------------------------------
# Shared application shell
# ---------------------------------------------------------------------
api = setup_page("Live Monitoring", "monitoring")

monitoring_title = st.empty()
monitoring_title.title("⚡ Live Monitoring")

# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("Monitoring Controls")
    display_unit = st.radio(
        "Display unit", ("MWh", "MW"), horizontal=True, key="monitoring_display_unit",
        help="MWh shows energy per observation interval. MW shows average power over that interval.",
    )
    auto_refresh = st.checkbox("Enable auto-refresh", value=True)
    # Migrate one existing Streamlit session to the new default. Subsequent user
    # choices are kept, so an explicit 5/10/15-second selection is respected.
    if not st.session_state.get("_monitoring_default_2s_applied"):
        st.session_state["electricity_poc_monitoring_refresh_v4"] = 2
        st.session_state["_monitoring_default_2s_applied"] = True
    refresh_seconds = st.selectbox(
        "Refresh interval",
        [2, 5, 10, 15],
        index=0,
        key="electricity_poc_monitoring_refresh_v4",
        format_func=lambda seconds: f"Every {seconds} seconds",
    )

    if st.button("Refresh now", use_container_width=True):
        if hasattr(st, "rerun"):
            st.rerun()
        else:
            st.experimental_rerun()

    st.divider()
    history_hours = st.selectbox(
        "Actual history shown",
        [6, 12, 24, 48, 72],
        index=4,  # Default to 72 hours (3 days) before the forecast origin.
        format_func=lambda hours: f"Previous {hours} hours",
    )
    show_forecast_table = st.checkbox("Show forecast table", value=False)
    show_diagnostics = st.checkbox(
        "Show technical details", value=False, key="monitor_tech_details",
        help="Show API row counts, storage, quality events and replay diagnostics.",
    )

# ---------------------------------------------------------------------
# Station selection
# ---------------------------------------------------------------------
try:
    registry = api_get_station_registry(api)
except requests.RequestException as exc:
    st.error(f"Cannot load station configuration from FastAPI: {exc}")
    st.stop()

stations = registry.get("stations", {})
station_ids = list(stations.keys())

if not station_ids:
    st.info("No stations are configured in config/stations.yaml.")
    st.stop()

active_station = registry.get("active_station")
try:
    requested_station = st.query_params.get("station")
except Exception:
    requested_station = None
preferred_station = requested_station or active_station or station_ids[0]
try:
    tree = api_get(api, "/stations/hierarchy").get("nodes", [])
except requests.RequestException:
    tree = []
if tree:
    with st.sidebar:
        selected_node = select_monitoring_node(tree, preferred_station)
else:
    station = st.selectbox("Station", station_ids, index=station_ids.index(preferred_station)
                           if preferred_station in station_ids else 0)
    selected_node = {"id": station, "name": stations[station].get("display_name") or station, "children": []}
station = selected_node["id"]
monitoring_title.title(f"⚡ Live Monitoring · {selected_node['name']}")

use_hierarchy_view = bool(selected_node["children"])
group_metric = None
with st.sidebar:
    if not selected_node["children"] and stations.get(station, {}).get("hierarchy_path"):
        use_hierarchy_view = not st.checkbox(
            "Show individual model and anomaly panels", value=True, key=f"individual_panels_{station}",
        )
    if use_hierarchy_view:
        include_forecasts = st.checkbox("Show forecasts", value=True)
        if not selected_node["children"]:
            group_metric = st.selectbox("Show", list(METRICS), format_func=lambda metric: METRICS[metric][0])


def render_live_monitoring_panel(
    api: str,
    station: str,
    stations: dict,
    history_hours: int,
    show_forecast_table: bool,
    auto_refresh: bool,
    refresh_seconds: int,
    show_diagnostics: bool = False,
    display_unit: str = "MWh",
) -> None:
    """Render the live monitoring panel in isolation from the app shell.

    Streamlit's fragment scheduler reruns only this function, not the complete
    page, navigation, sidebar controls, and unrelated UI.
    """
    # Health and data-quality details are operator diagnostics, not recurring
    # customer-facing KPIs. Avoid two extra API calls on every chart refresh.
    readiness: dict = {}
    quality_records: list[dict] = []
    if show_diagnostics:
        with st.expander("Advanced: API health and data quality", expanded=False):
            try:
                readiness = api_get_readiness(api)
                quality_payload = api_get_quality_events(api, station, 100)
                quality_records = quality_payload.get("records", [])
                health_text = "Ready" if readiness.get("ready") else "Check backend"
                st.caption(
                    f"API: {health_text} · Storage: {(readiness.get('storage') or {}).get('backend', 'unknown')} "
                    f"· Data-quality events: {len(quality_records)}"
                )
                if quality_records:
                    st.dataframe(pd.DataFrame(quality_records), use_container_width=True, hide_index=True)
            except requests.RequestException as exc:
                st.warning(f"Operational diagnostics unavailable: {exc}")

    station_interval_minutes = int(stations[station].get("interval_minutes", 15))
    station_horizon_steps = int(stations[station].get("horizon_steps", 96))
    station_forecast_horizon_hours = float(stations[station].get("forecast_horizon_hours", 24))
    unit_text = interval_unit_label(display_unit, station_interval_minutes)
    demand_factor = demand_to_mw_factor(stations[station], station_interval_minutes) * unit_factor(
        "MW", display_unit, station_interval_minutes,
    )
    energy_factor = unit_factor("MWh", display_unit, station_interval_minutes)

    def format_value(value) -> str:
        return f"{value:,.2f} {display_unit}" if pd.notna(value) else "—"

    # ---------------------------------------------------------------------
    # Load saved actuals and forecasts through FastAPI
    # ---------------------------------------------------------------------
    try:
        # Only recent forecast history is needed for the current run and the
        # forecast-history selector. The API reads this tail without loading the
        # entire append-only forecasts.csv file.
        forecast_payload = api_get(
            api, "/results/forecasts", {"station": station, "limit": 2000, "include_live_progress": True}
        )
    except requests.RequestException as exc:
        render_initializing("Demand forecasting", "Waiting for a response from the forecast service.")
        if show_diagnostics:
            st.caption(str(exc))
        return

    forecasts = pd.DataFrame(forecast_payload.get("records", []))
    actuals = pd.DataFrame()

    if not forecasts.empty:
        forecasts = as_utc(forecasts, "forecast_timestamp_utc")
        forecasts = number_column(forecasts, "predicted_demand", demand_factor)

        for col in ["forecast_issued_at_utc", "forecast_origin_utc"]:
            if col in forecasts.columns:
                forecasts[col] = pd.to_datetime(
                    forecasts[col], utc=True, errors="coerce"
                )

    # ---------------------------------------------------------------------
    # Identify the latest usable forecast run
    # ---------------------------------------------------------------------
    # During replay, a predictor restart can append an initial forecast with a
    # newer issued-at timestamp but an older forecast origin. Select by the
    # furthest valid origin, preferring complete 96-step runs, not by issue time.
    latest_run = select_latest_complete_forecast_run(
        forecasts,
        horizon_steps=station_horizon_steps,
        interval_minutes=station_interval_minutes,
    )
    forecast_origin = None

    if not latest_run.empty:
        if (
            "forecast_origin_utc" in latest_run.columns
            and latest_run["forecast_origin_utc"].notna().any()
        ):
            forecast_origin = latest_run["forecast_origin_utc"].dropna().iloc[0]
        else:
            forecast_origin = (
                latest_run["forecast_timestamp_utc"].iloc[0]
                - pd.Timedelta(minutes=station_interval_minutes)
            )

    # Use the replay producer's acknowledged source timestamp as the graph clock
    # whenever no demand forecast exists. Without this boundary, the API's static
    # test CSV can expose October observations before the replay has reached them.
    # When a forecast exists, preserve the existing forecast-origin boundary.
    project_root = Path(__file__).resolve().parents[2]
    replay_timestamp = read_last_acked_timestamp(project_root, station)
    actuals_display_cutoff, actuals_cutoff_source = resolve_actuals_display_cutoff(
        stations[station].get("expected_stream_start_utc"),
        station_interval_minutes,
        replay_timestamp=replay_timestamp,
        forecast_origin=forecast_origin,
    )

    progress = st.session_state.setdefault("monitoring_timeline_progress", {})
    replay_start = stations[station].get("expected_stream_start_utc")
    live_through = forecast_payload.get("live_through_utc")
    predictor_run = forecast_payload.get("predictor_run")
    if forecast_origin is not None and (
        "forecast_origin_utc" not in latest_run or latest_run["forecast_origin_utc"].isna().all()
    ):
        latest_run = latest_run.assign(forecast_origin_utc=forecast_origin)
    cursor = forecast_cursor(
        latest_run.to_dict(orient="records"), live_through, replay_start, station_horizon_steps, station_interval_minutes,
    )
    if not timeline_is_live(progress, ("station", station, "demand"), cursor, predictor_run):
        render_initializing(
            "Demand forecasting", "Waiting for a complete forecast and a newer live forecast origin.",
            f"Replay progress · {fmt_utc(replay_timestamp or live_through)}" if replay_timestamp is not None or live_through else "",
        )
        return

    # Allow room for the displayed history and the forecast context.
    actual_params = {"station": station, "limit": 600, "recent_only": True}
    if actuals_display_cutoff is not None:
        actual_params["end_time"] = actuals_display_cutoff.isoformat()
    try:
        actual_payload = api_get(api, "/results/actuals", actual_params)
    except requests.RequestException as exc:
        st.warning(f"Demand history is temporarily unavailable: {exc}")
        actual_payload = {}
    actuals = pd.DataFrame(actual_payload.get("records", []))
    if not actuals.empty:
        actuals = as_utc(actuals, "utc_timestamp")
        actuals = number_column(actuals, "actual_demand", demand_factor)
        actuals = (
            actuals.drop_duplicates("utc_timestamp", keep="last")
            .sort_values("utc_timestamp")
            .reset_index(drop=True)
        )

    # Residual-load risk is computed from same-origin Demand, Solar and Wind
    # forecasts. The backend stores persistent origin assessments and
    # immutable per-origin snapshots; dashboard refreshes do not duplicate scoring.
    residual_payload: dict = {}
    residual_available = bool(stations[station].get('residual_model_checkpoint'))
    if residual_available:
        try:
            residual_payload = api_get(api, '/results/residual/forecast', {'station':station,'history_runs':24,'cached_only':True,'origin_utc':forecast_origin.isoformat() if forecast_origin is not None else None,'lookback_minutes':60})
        except requests.RequestException as exc:
            residual_payload = {'available':False,'message':f'Residual risk service unavailable: {exc}'}
    residual_origin = residual_payload.get("origin_utc")
    residual_cursor = forecast_cursor(
        [dict(row, forecast_origin_utc=residual_origin) for row in residual_payload.get("records", [])],
        live_through, replay_start, station_horizon_steps, station_interval_minutes,
    ) if residual_payload.get("available") else None
    residual_ready = timeline_is_live(
        progress, ("station", station, "residual"), residual_cursor, predictor_run,
    )
    # ---------------------------------------------------------------------
    # KPI cards
    # ---------------------------------------------------------------------
    latest_actual = (
        float(actuals.iloc[-1]["actual_demand"]) if not actuals.empty else None
    )
    first_forecast = (
        float(latest_run.iloc[0]["predicted_demand"])
        if not latest_run.empty else None
    )
    peak_forecast = (
        float(latest_run["predicted_demand"].max())
        if not latest_run.empty else None
    )
    average_forecast = (
        float(latest_run["predicted_demand"].mean())
        if not latest_run.empty else None
    )

    st.subheader("Demand · current forecast")
    k1, k2, k3, k4 = st.columns(4, gap="small")
    k1.metric("Latest actual", format_value(latest_actual))
    k2.metric("Next forecast", format_value(first_forecast))
    k3.metric("24h peak", format_value(peak_forecast))
    k4.metric("24h average", format_value(average_forecast))
    if forecast_origin is not None:
        st.caption(f"Forecast origin · {fmt_utc(forecast_origin)}")
    if show_diagnostics and not latest_run.empty and "forecast_issued_at_utc" in latest_run.columns:
        st.caption("Forecast processing time: " + fmt_utc(latest_run["forecast_issued_at_utc"].iloc[0]))

    # ---------------------------------------------------------------------
    # Main graph: solid actual line -> dashed forecast line
    # ---------------------------------------------------------------------
    st.subheader("Demand · actual vs forecast")

    fig = go.Figure()
    chart_start = None
    chart_end = None

    if not latest_run.empty:
        # Always show the available forecast path. Actual history is an optional
        # companion trace; an empty/misaligned actual response must not blank the
        # entire demand chart.
        forecast_targets = latest_run["forecast_timestamp_utc"].tolist()
        forecast_values = latest_run["predicted_demand"].astype(float).tolist()

        if forecast_origin is not None:
            history_start = forecast_origin - pd.Timedelta(hours=history_hours)
            actual_before_origin = actuals[
                actuals["utc_timestamp"] <= forecast_origin
            ].sort_values("utc_timestamp") if not actuals.empty else pd.DataFrame()
            actual_history = (
                actual_before_origin[
                    actual_before_origin["utc_timestamp"] >= history_start
                ].copy()
                if not actual_before_origin.empty
                else pd.DataFrame()
            )

            if not actual_history.empty:
                fig.add_trace(
                    go.Scatter(
                        x=actual_history["utc_timestamp"],
                        y=actual_history["actual_demand"],
                        mode="lines",
                        name="Actual demand",
                        line=dict(color=ACTUAL_COLOR, width=3.5),
                        connectgaps=False,
                        hovertemplate=(
                            f"Actual: %{{y:,.2f}} {unit_text}<br>"
                            "Time: %{x|%d %b %H:%M UTC}<extra></extra>"
                        ),
                    )
                )

            if not actual_before_origin.empty:
                anchor = actual_before_origin.iloc[-1]
                forecast_x = [anchor["utc_timestamp"]] + forecast_targets
                forecast_y = [float(anchor["actual_demand"])] + forecast_values
            else:
                forecast_x = forecast_targets
                forecast_y = forecast_values
                st.warning(
                    "Actual history is unavailable at the selected forecast origin. "
                    "Showing the forecast path without an actual-demand anchor."
                )

            fig.add_trace(
                go.Scatter(
                    x=forecast_x,
                    y=forecast_y,
                    mode="lines",
                    name="Forecast",
                    line=dict(color=FORECAST_COLOR, width=3.5, dash="dash"),
                    connectgaps=False,
                    hovertemplate=(
                        f"Forecast: %{{y:,.2f}} {unit_text}<br>"
                        "Time: %{x|%d %b %H:%M UTC}<extra></extra>"
                    ),
                )
            )
            mark_now(fig, forecast_origin)
            chart_start = history_start
        else:
            chart_start = forecast_targets[0]
        chart_end = latest_run["forecast_timestamp_utc"].max()

    if show_diagnostics:
        # Surface retrieval sizes and the selected forecast boundary so a blank/stale
        # chart can be diagnosed from the dashboard without opening server logs.
        st.caption(
            f"Demand API rows: {len(actuals):,} actuals · {len(forecasts):,} forecast points · "
            f"selected run: {len(latest_run):,} points · "
            f"origin: {fmt_utc(forecast_origin) if forecast_origin is not None else '—'}"
        )
        st.caption(
            f"Actual-data display cutoff: {fmt_utc(actuals_display_cutoff) if actuals_display_cutoff is not None else 'not configured'} "
            f"({actuals_cutoff_source.replace('_', ' ')}). "
            "The graph opens once a newer complete live forecast is available. "
            "Observations are shown through the forecast origin."
        )

    # Residual-load anomaly intervals are FORECAST risks, not labels on observed demand.
    # Overlay only when both chart and residual assessment share the same origin.
    if forecast_origin is not None and residual_ready and not latest_run.empty:
        from dashboard.residual_plot import add_spike_intervals
        assessment_origin = pd.to_datetime(residual_payload.get('origin_utc'), utc=True, errors='coerce')
        if pd.notna(assessment_origin) and assessment_origin == pd.Timestamp(forecast_origin):
            add_spike_intervals(fig, residual_payload, latest_run,
                                timestamp_col='forecast_timestamp_utc', value_col='predicted_demand',
                                interval_minutes=station_interval_minutes)
        elif show_diagnostics:
            st.caption('Residual spike overlay is waiting for a forecast origin matching the Demand chart.')

    if fig.data:
        apply_monitoring_axes(fig, chart_start, chart_end, history_hours, f"Electricity demand ({unit_text})",
                              display_unit, f"live-monitor-{station}-{history_hours}")
        st.plotly_chart(fig, use_container_width=True)

    # Residual-load spike outlook below Demand; score diagnostics live in Model Analysis.
    st.subheader('Residual Load Spike Detection')
    if residual_available and not residual_ready:
        render_initializing("Residual Load", "Waiting for a complete assessment at a newer live forecast origin.")
    elif residual_ready:
        lag = residual_payload.get('snapshot_lag_minutes')
        if lag is not None and lag > 0:
            st.caption(
                f'Latest completed residual assessment: {fmt_utc(pd.to_datetime(residual_payload.get("origin_utc"), utc=True))} '
                f'({lag} minutes behind current Demand forecast). '
                'Spike markers are not overlaid on a newer Demand origin, to avoid mislabeling predictions.'
            )
        summary = residual_summary(residual_payload)
        r1,r2,r3,r4 = st.columns(4,gap='small')
        for col, label, key in ((r1, 'Next residual', 'next'), (r2, '24h peak', 'peak'),
                                (r3, 'Peak 30m ramp', 'ramp')):
            raw = summary[key]
            col.metric(label, format_value(raw * energy_factor if raw is not None else None))
        r4.metric('Spike intervals',str(summary['flagged']))
        residual_fig, _ = residual_figures(
            residual_payload, display_unit=display_unit, interval_minutes=station_interval_minutes,
        )
        st.plotly_chart(residual_fig,use_container_width=True,config={'displayModeBar':False})
        with st.expander('Earlier residual-load forecast assessments',expanded=False):
            history = pd.DataFrame([{'origin_utc':item.get('origin_utc'),
                'flagged_points':item.get('flagged_points',sum(bool(r.get('upward_highlight')) for r in item.get('records',[])))}
                for item in residual_payload.get('history',[])])
            if not history.empty:st.dataframe(history,use_container_width=True,hide_index=True)
    else:
        st.caption(residual_payload.get("message", "Waiting for a Residual Load assessment from synchronized forecasts."))

    # ---------------------------------------------------------------------
    # Renewable generation monitoring (solar + onshore wind)
    # ---------------------------------------------------------------------
    st.subheader("Solar & Wind Energy · generation")
    renewable_params = {"station": station, "limit": 5000}
    renewable_params["defer_heavy"] = True
    if forecast_origin is not None:
        renewable_params["end_time"] = forecast_origin.isoformat()
    try:
        renewable_payload = api_get(api, "/results/renewables", renewable_params)
        if renewable_payload.get("history_pending"):
            st.caption("Loading Solar and Wind Energy observations in the background.")
        renewable_actuals = pd.DataFrame(renewable_payload.get("actuals", []))
        if not renewable_actuals.empty:
            renewable_actuals = as_utc(renewable_actuals, "utc_timestamp")
            for col in ["solar_actual_mwh", "wind_actual_mwh"]:
                if col in renewable_actuals.columns:
                    renewable_actuals[col] = pd.to_numeric(renewable_actuals[col], errors="coerce")
        renewable_forecasts = renewable_payload.get("forecasts", {}) or {}
        renewable_cfg = stations[station].get("renewables") or {}
        enabled_kinds = [
            kind for kind in ("solar", "wind")
            if (renewable_cfg.get("components_enabled") or {}).get(
                kind, (renewable_cfg.get(kind) or {}).get("enabled", bool(renewable_cfg.get(kind)))
            )
        ] if renewable_cfg.get("enabled") else list(renewable_forecasts)
        ready_kinds = {
            kind for kind in enabled_kinds if timeline_is_live(
                progress, ("station", station, kind), forecast_cursor(
                    renewable_forecasts.get(kind, []), live_through, replay_start,
                    station_horizon_steps, station_interval_minutes,
                ), predictor_run,
            )
        }
        pending = {"solar": "Solar", "wind": "Wind Energy"}
        for kind in enabled_kinds:
            if kind not in ready_kinds:
                render_initializing(pending[kind], "Waiting for a complete forecast at a newer live origin.")
        renewable_diagnostics = renewable_payload.get("diagnostics") or {}
        partial = [kind for kind in ("solar", "wind")
                   if renewable_forecasts.get(kind) and not renewable_diagnostics.get(f"{kind}_forecast_complete", True)]
        if partial:
            st.warning("Only an incomplete recent forecast run is available for: " + ", ".join(partial) + ". Some horizon steps may be missing until the next full run.")

        # In the no-demand-forecast state, /results/renewables intentionally remains
        # unbounded so its saved forecast series are not altered. Clamp only the
        # actual traces here to the same replay clock used by the demand graph.
        renewable_actuals = filter_actuals_to_cutoff(
            renewable_actuals, actuals_display_cutoff, "utc_timestamp"
        )

        renewable_summary = build_renewable_summaries(renewable_actuals, renewable_forecasts)
        for kind, label in (("solar", "Solar"), ("wind", "Wind Energy")):
            if kind not in ready_kinds:
                continue
            stats = renewable_summary[kind]
            st.markdown(f"**{label}** · {unit_text}")
            columns = st.columns(4, gap="small")
            for col, caption, key in zip(columns, ("Latest actual", "Next forecast", "Forecast peak", "Forecast average"),
                                         ("latest_actual", "first_forecast", "peak_forecast", "average_forecast")):
                raw = stats[key]
                col.metric(caption, format_value(raw * energy_factor if raw is not None else None))

        ready_forecasts = {kind: renewable_forecasts[kind] for kind in ready_kinds}
        renewable_fig, renewable_counts = build_renewable_figure(
            renewable_actuals, ready_forecasts,
            chart_start=chart_start, chart_end=chart_end,
            display_unit=display_unit, interval_minutes=station_interval_minutes, enabled_kinds=ready_kinds,
        )
        renewable_series_added = any(renewable_counts.values())
        if renewable_series_added:
            st.plotly_chart(renewable_fig, use_container_width=True)
        elif not enabled_kinds and not renewable_payload.get("history_pending"):
            st.caption("No Solar or Wind Energy models are configured for this station.")
        coverage = renewable_payload.get("coverage", {}) or {}
        coverage = {kind: dict(details or {}) for kind, details in coverage.items()}
        # Coverage labels must describe the rows actually visible in this graph,
        # not the end of the static test files that have not yet been replayed.
        for kind, value_column in (("solar", "solar_actual_mwh"), ("wind", "wind_actual_mwh")):
            details = coverage.setdefault(kind, {})
            if not renewable_actuals.empty and value_column in renewable_actuals.columns:
                visible = renewable_actuals.dropna(subset=[value_column])
                details["rows"] = int(len(visible))
                details["latest_actual_utc"] = (
                    visible["utc_timestamp"].max().isoformat() if not visible.empty else None
                )
            else:
                details["rows"] = 0
                details["latest_actual_utc"] = None

        if show_diagnostics:
            coverage_bits = []
            for kind in ("solar", "wind"):
                details = coverage.get(kind, {}) or {}
                latest = details.get("latest_actual_utc")
                coverage_bits.append(
                    f"{kind.title()}: {details.get('rows', 0):,} actual rows, latest {fmt_utc(latest) if latest else '—'}, "
                    f"test file {'available' if details.get('test_file_available') else 'not supplied'}"
                )
            if coverage_bits:
                st.caption(" · ".join(coverage_bits))
            actual_count = len(renewable_actuals) if not renewable_actuals.empty else 0
            st.caption(
                f"Renewable API rows: {actual_count:,} historical timestamps · "
                f"displayed in chart: Solar actual {renewable_counts['solar_actual']:,}, "
                f"Solar forecast {renewable_counts['solar_forecast']:,}, "
                f"Wind actual {renewable_counts['wind_actual']:,}, "
                f"Wind forecast {renewable_counts['wind_forecast']:,}."
            )
        if forecast_origin is not None:
            stale_kinds = []
            for kind in ("solar", "wind"):
                latest_text = (coverage.get(kind, {}) or {}).get("latest_actual_utc")
                if latest_text:
                    latest_ts = pd.to_datetime(latest_text, utc=True, errors="coerce")
                    if pd.notna(latest_ts) and (forecast_origin - latest_ts) > pd.Timedelta(minutes=station_interval_minutes):
                        stale_kinds.append(f"{kind} actuals end at {fmt_utc(latest_ts)}")
            if stale_kinds:
                render_status_strip([
                    ("Renewable observations delayed", "; ".join(stale_kinds), "warn")
                ])
    except requests.RequestException as exc:
        render_initializing("Solar and Wind Energy", "Waiting for a response from the forecast service.")
        if show_diagnostics:
            st.caption(str(exc))

    if show_diagnostics:
        # ---------------------------------------------------------------------
        # Latest forecast run details
        # ---------------------------------------------------------------------
        st.divider()
        st.subheader("Latest Forecast Run")

        if not latest_run.empty:
            display_columns = [
                col for col in [
                    "forecast_issued_at_utc",
                    "forecast_origin_utc",
                    "forecast_timestamp_utc",
                    "horizon_step",
                    "predicted_demand",
                    "weather_mode",
                    "model_file",
                ]
                if col in latest_run.columns
            ]

            selected_columns = latest_run[display_columns]
            st.dataframe(
                selected_columns,
                use_container_width=True,
                hide_index=True,
                column_config={"predicted_demand": f"Predicted demand ({unit_text})"},
            )

            download = selected_columns.assign(demand_unit=display_unit, interval_minutes=station_interval_minutes)
            st.download_button(
                "Download Latest Forecast Run",
                data=download.to_csv(index=False).encode("utf-8"),
                file_name=f"{station}_latest_forecast_{display_unit.lower()}.csv",
                mime="text/csv",
            )
        else:
            st.info("No saved forecast run is available yet.")

        # ---------------------------------------------------------------------
        # Forecast history
        # ---------------------------------------------------------------------
        st.subheader("Forecast History")

        if (
            not forecasts.empty
            and "forecast_issued_at_utc" in forecasts.columns
        ):
            run_times = sorted(
                forecasts["forecast_issued_at_utc"].dropna().unique(),
                reverse=True,
            )

            if run_times:
                run_labels = [
                    pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S UTC")
                    for value in run_times
                ]
                selected_label = st.selectbox(
                    "Select a saved forecast run", run_labels
                )
                selected_run_time = pd.Timestamp(
                    run_times[run_labels.index(selected_label)]
                )
                selected_run = forecasts[
                    forecasts["forecast_issued_at_utc"] == selected_run_time
                ].sort_values("forecast_timestamp_utc")

                history_fig = go.Figure()
                history_fig.add_trace(
                    go.Scatter(
                        x=selected_run["forecast_timestamp_utc"],
                        y=selected_run["predicted_demand"],
                        mode="lines+markers",
                        name="Saved Demand Forecast",
                        line=dict(color="#F57C00", width=3, dash="dot"),
                        marker=dict(size=4),
                    )
                )
                history_fig.update_layout(
                    title=f"Forecast run generated {selected_label}",
                    xaxis_title="Target timestamp (UTC)",
                    yaxis_title=f"Predicted demand ({unit_text})",
                    hovermode="x unified",
                    height=380,
                    template="plotly_dark",
                )
                st.plotly_chart(history_fig, use_container_width=True)
                st.caption(f"Saved forecast runs available: {len(run_times):,}")
        else:
            st.info("Forecast history will appear after the predictor saves its first run.")

    if show_forecast_table:
        st.subheader(
            f"Current Forecast — {station_horizon_steps:,} Steps "
            f"({station_forecast_horizon_hours:g} Hours · {station_interval_minutes}-Minute Interval)"
        )
        if not latest_run.empty:
            st.dataframe(
                latest_run,
                use_container_width=True,
                hide_index=True,
                column_config={"predicted_demand": f"Predicted demand ({unit_text})"},
            )
        else:
            st.info("No forecast steps are available.")

    # ---------------------------------------------------------------------
    # Storage status
    # ---------------------------------------------------------------------
    if show_diagnostics:
        with st.expander("Storage and data status", expanded=False):
            active_storage_backend = (readiness.get("storage") or {}).get("backend", "unknown")
            st.write(f"**Storage backend:** {active_storage_backend} (accessed through FastAPI).")
            st.write("**Forecast storage:** `data/processed/forecasts.csv`")
            st.write("**Actual demand storage:** `data/processed/actuals.csv`")
            st.write(f"**Forecast rows returned by API:** {len(forecasts):,}")
            st.write(f"**Unique actual timestamps returned by API:** {len(actuals):,}")
            st.write(
                "**Actual source:** "
                + (
                    actual_payload.get(
                        "message", "Streaming observations stored by the predictor."
                    )
                    if actuals.empty
                    else "Kafka → Spark → predictor → actuals.csv"
                )
            )
            st.write(
                "**Forecast source:** Kafka → Spark → selected model predictor → station forecasts file"
            )
            st.caption(
                "The main chart uses the latest saved forecast run and its forecast "
                "origin. It does not mix predictions from older forecast runs."
            )

    if show_diagnostics:
        st.divider()
        st.caption(
            f"Auto-refresh: "
            f"{'On (live panel only)' if auto_refresh else 'Off'}"
            f" · Refresh interval: {refresh_seconds}s"
            f" · Last render: {pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M:%S UTC}"
        )


# Streamlit >=1.38 (the project's declared minimum) supports fragments.
# Refresh only the live panel; avoid a full-page 2-second rerun that can lock
# the browser while re-rendering navigation, tables, and all charts.
if not hasattr(st, "fragment"):
    st.error(
        "This monitoring page needs Streamlit 1.38 or newer for responsive "
        "fragment refresh. Upgrade with: pip install -U streamlit"
    )
    st.stop()

if use_hierarchy_view:
    @st.fragment(run_every=float(refresh_seconds) if auto_refresh else None)
    def render_hierarchy_monitoring() -> None:
        descendant = selected_node
        while descendant["children"]:
            descendant = descendant["children"][0]
        interval = int(stations.get(descendant["id"], {}).get("interval_minutes", 15))
        window = min(2016, max(1, history_hours * 60 // interval + 1))
        view = (station, window, include_forecasts)
        try:
            payload = api_get(api, "/results/hierarchy", {
                "node": station, "window": window, "include_forecasts": include_forecasts, "include_history": True,
            })
            st.session_state["last_hierarchy_reading"] = (view, time.monotonic(), payload)
        except requests.RequestException as exc:
            previous = st.session_state.get("last_hierarchy_reading")
            if previous is None or previous[0] != view:
                render_initializing(selected_node["name"], "Waiting for the shared readings from FastAPI.")
                if show_diagnostics:
                    st.caption(str(exc))
                return
            _, fetched_at, payload = previous
            age = int(time.monotonic() - fetched_at)
            st.warning(f"Hierarchy refresh delayed. Latest shared data arrived {age}s ago.")
        progress = st.session_state.setdefault("monitoring_timeline_progress", {})
        render_group(payload, selected_node, group_metric, progress=progress, history_hours=history_hours,
                     include_forecasts=include_forecasts, show_forecast_table=show_forecast_table,
                     display_unit=display_unit)

    render_hierarchy_monitoring()
else:
    fragment_runner = st.fragment(
        run_every=float(refresh_seconds) if auto_refresh else None
    )(render_live_monitoring_panel)
    fragment_runner(api, station, stations, history_hours, show_forecast_table, auto_refresh,
                    refresh_seconds, show_diagnostics, display_unit)
