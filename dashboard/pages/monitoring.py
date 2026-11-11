"""Electricity demand monitoring: actual history connected to the selected model forecast."""

from datetime import date, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

try:
    from dashboard.ui import setup_page, render_status_strip
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
    from dashboard.renewable_plot import build_renewable_figure
    from dashboard.renewable_summary import build_renewable_summaries
    from dashboard.forecast_selection import select_latest_complete_forecast_run
    from dashboard.residual_plot import residual_figures, residual_summary
    from dashboard.replay_display import filter_actuals_to_cutoff, read_last_acked_timestamp, resolve_actuals_display_cutoff

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


def number_column(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    if frame.empty or column not in frame.columns:
        return pd.DataFrame()
    result = frame.copy()
    result[column] = pd.to_numeric(result[column], errors="coerce")
    return result.dropna(subset=[column])


def fmt_mw(value) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):,.0f} MW"


def fmt_utc(value) -> str:
    if value is None or pd.isna(value):
        return "—"
    return pd.to_datetime(value, utc=True).strftime("%Y-%m-%d %H:%M UTC")


# ---------------------------------------------------------------------
# Shared application shell
# ---------------------------------------------------------------------
api = setup_page("Live Monitoring", "monitoring")

st.title("⚡ Live Monitoring")

# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("Monitoring Controls")
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
preferred_station = requested_station if requested_station in station_ids else active_station
default_index = (
    station_ids.index(preferred_station) if preferred_station in station_ids else 0
)
station = st.selectbox("Station", station_ids, index=default_index)

def render_live_monitoring_panel(
    api: str,
    station: str,
    stations: dict,
    history_hours: int,
    show_forecast_table: bool,
    auto_refresh: bool,
    refresh_seconds: int,
    show_diagnostics: bool = False,
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

    # ---------------------------------------------------------------------
    # Load saved actuals and forecasts through FastAPI
    # ---------------------------------------------------------------------
    try:
        # Only recent forecast history is needed for the current run and the
        # forecast-history selector. The API reads this tail without loading the
        # entire append-only forecasts.csv file.
        forecast_payload = api_get(
            api, "/results/forecasts", {"station": station, "limit": 2000}
        )
    except requests.RequestException as exc:
        st.error(f"Could not retrieve forecasts from FastAPI: {exc}")
        st.stop()

    forecasts = pd.DataFrame(forecast_payload.get("records", []))
    actuals = pd.DataFrame()

    if not forecasts.empty:
        forecasts = as_utc(forecasts, "forecast_timestamp_utc")
        forecasts = number_column(forecasts, "predicted_demand")

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

    # Until the first complete forecast is saved, show an initialization state
    # instead of the actual-only fallback graph. This is display-only: the
    # existing demand/renewable chart windows and x-axis ranges remain unchanged.
    if latest_run.empty:
        replay_text = (
            fmt_utc(replay_timestamp) if replay_timestamp is not None
            else "Waiting for the first replay observation"
        )
        st.markdown(
            f"""
            <style>
              .forecast-init-card {{
                min-height: 320px; display:flex; flex-direction:column;
                align-items:center; justify-content:center; text-align:center;
                border:1px solid rgba(128,128,128,.28); border-radius:18px;
                padding:34px 24px; margin:12px 0 22px 0;
                background:linear-gradient(135deg, rgba(30,64,110,.10), rgba(14,165,160,.08));
              }}
              .forecast-init-spinner {{
                width:58px; height:58px; border-radius:50%;
                border:5px solid rgba(100,116,139,.22);
                border-top-color:#0ea5a4; border-right-color:#3b82f6;
                animation:forecast-spin 1.05s linear infinite;
                box-shadow:0 0 24px rgba(14,165,164,.15);
                margin-bottom:22px;
              }}
              @keyframes forecast-spin {{ to {{ transform:rotate(360deg); }} }}
              .forecast-init-title {{ font-size:23px; font-weight:700; margin-bottom:9px; }}
              .forecast-init-copy {{ max-width:650px; color:var(--text-color); opacity:.78;
                font-size:14px; line-height:1.6; }}
              .forecast-init-meta {{ margin-top:18px; font-size:12px; opacity:.65; }}
            </style>
            <div class="forecast-init-card">
              <div class="forecast-init-spinner"></div>
              <div class="forecast-init-title">Initializing demand forecasting</div>
              <div class="forecast-init-copy">
                Preparing the historical input window and waiting for the first
                complete forecast. Monitoring charts will appear when the forecast
                is ready.
              </div>
              <div class="forecast-init-meta">Replay progress: {replay_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # 5,000 rows comfortably covers the 72-hour view and forecast-context window.
    actual_params = {"station": station, "limit": 600}
    if actuals_display_cutoff is not None:
        actual_params["end_time"] = actuals_display_cutoff.isoformat()
    try:
        actual_payload = api_get(api, "/results/actuals", actual_params)
    except requests.RequestException as exc:
        st.error(f"Could not retrieve actuals from FastAPI: {exc}")
        st.stop()
    actuals = pd.DataFrame(actual_payload.get("records", []))
    if not actuals.empty:
        actuals = as_utc(actuals, "utc_timestamp")
        actuals = number_column(actuals, "actual_demand")
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
    k1.metric("Latest actual", fmt_mw(latest_actual))
    k2.metric("Next forecast", fmt_mw(first_forecast))
    k3.metric("24h peak", fmt_mw(peak_forecast))
    k4.metric("24h average", fmt_mw(average_forecast))
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
                        line=dict(color="#1976D2", width=3.5),
                        connectgaps=False,
                        hovertemplate=(
                            "Actual: %{y:,.0f} MW<br>"
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
                    line=dict(color="#F57C00", width=3.5, dash="dash"),
                    connectgaps=False,
                    hovertemplate=(
                        "Forecast: %{y:,.0f} MW<br>"
                        "Time: %{x|%d %b %H:%M UTC}<extra></extra>"
                    ),
                )
            )
            fig.add_vline(
                x=forecast_origin,
                line_width=2,
                line_dash="dash",
                line_color="#888888",
                annotation_text="Now",
                annotation_position="top",
                annotation_font_color="#dceafa",
            )
            chart_start = history_start
        else:
            # Legacy forecast rows without an explicit origin can still be plotted.
            chart_start = forecast_targets[0]
            st.warning("Forecast origin is missing; showing the forecast without a Now marker.")

        chart_end = latest_run["forecast_timestamp_utc"].max()
        tick_interval_hours = 6 if history_hours >= 48 else 3
        fig.update_xaxes(
            range=[chart_start, chart_end],
            dtick=tick_interval_hours * 60 * 60 * 1000,
            tickformat="%d %b<br>%H:%M",
            tickangle=0,
            tickfont=dict(size=11),
            automargin=True,
            ticklabeloverflow="allow",
        )

    elif not actuals.empty:
        # Actual-only fallback while waiting for a forecast run.
        latest_time = actuals["utc_timestamp"].max()
        chart_start = latest_time - pd.Timedelta(hours=history_hours)
        chart_end = latest_time
        actual_history = actuals[actuals["utc_timestamp"] >= chart_start]
        fig.add_trace(
            go.Scatter(
                x=actual_history["utc_timestamp"],
                y=actual_history["actual_demand"],
                mode="lines",
                name="Actual demand",
                line=dict(color="#1976D2", width=3.5),
                connectgaps=False,
            )
        )
        st.info("Waiting for a saved forecast run; showing actual demand only.")
    else:
        st.info("Neither actual-demand observations nor forecast records were returned by the results API.")

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
            "Before the replay reaches 1 October, the graph shows only pre-stream history; "
            "replay actuals are revealed up to the producer's acknowledged timestamp."
        )

    # Residual-load anomaly intervals are FORECAST risks, not labels on observed demand.
    # Overlay only when both chart and residual assessment share the same origin.
    if forecast_origin is not None and residual_payload.get('available') and not latest_run.empty:
        from dashboard.residual_plot import add_spike_intervals
        assessment_origin = pd.to_datetime(residual_payload.get('origin_utc'), utc=True, errors='coerce')
        if pd.notna(assessment_origin) and assessment_origin == pd.Timestamp(forecast_origin):
            add_spike_intervals(fig, residual_payload, latest_run,
                                timestamp_col='forecast_timestamp_utc', value_col='predicted_demand',
                                interval_minutes=station_interval_minutes)
        elif show_diagnostics:
            st.caption('Residual spike overlay is waiting for a forecast origin matching the Demand chart.')

    if fig.data:
        fig.update_layout(
            height=480,
            template="plotly_dark",
            hovermode="x unified",
            font=dict(color="#e8f1ff",size=12),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(8,27,48,.50)",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                x=0,
            ),
            margin=dict(l=25, r=25, t=65, b=35),
            uirevision=f"live-monitor-{station}-{history_hours}",
        )
        fig.update_xaxes(
            title_text="Time (UTC)",
            showgrid=True,
            gridcolor="#284365",
            zeroline=False,
            rangeslider_visible=False,
            tickangle=0,
            tickfont=dict(size=11,color="#d5e4fa"),
            title_font=dict(color="#d5e4fa"),
            automargin=True,
        )
        fig.update_yaxes(
            title_text="Electricity demand (MW)",
            showgrid=True,
            gridcolor="#284365",
            zeroline=False,
            tickfont=dict(color="#d5e4fa"), title_font=dict(color="#d5e4fa"),
            automargin=True,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Residual-load spike outlook below Demand; score diagnostics live in Model Analysis.
    st.subheader('Residual Load Spike Detection')
    if residual_payload.get('available'):
        lag = residual_payload.get('snapshot_lag_minutes')
        if lag is not None and lag > 0:
            st.caption(
                f'Latest completed residual assessment: {fmt_utc(pd.to_datetime(residual_payload.get("origin_utc"), utc=True))} '
                f'({lag} minutes behind current Demand forecast). '
                'Spike markers are not overlaid on a newer Demand origin, to avoid mislabeling predictions.'
            )
        summary = residual_summary(residual_payload)
        r1,r2,r3,r4 = st.columns(4,gap='small')
        fmt=lambda x: '—' if x is None else f'{x:,.0f}'
        r1.metric('Next residual',fmt(summary['next']))
        r2.metric('24h peak',fmt(summary['peak']))
        r3.metric('Peak 30m ramp',fmt(summary['ramp']))
        r4.metric('Spike intervals',str(summary['flagged']))
        residual_fig, _ = residual_figures(residual_payload)
        st.plotly_chart(residual_fig,use_container_width=True,config={'displayModeBar':False})
        with st.expander('Earlier residual-load forecast assessments',expanded=False):
            history = pd.DataFrame([{'origin_utc':item.get('origin_utc'),
                'flagged_points':item.get('flagged_points',sum(bool(r.get('upward_highlight')) for r in item.get('records',[])))}
                for item in residual_payload.get('history',[])])
            if not history.empty:st.dataframe(history,use_container_width=True,hide_index=True)
    else:
        st.caption(residual_payload.get('message','Awaiting synchronized Demand, Solar and Wind forecasts.'))

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
        if renewable_payload.get('pending'):
            st.info(renewable_payload.get('message', 'Solar/Wind data is loading.'))
            return  # Avoid incorrectly reporting missing generation while cache warms.
        renewable_actuals = pd.DataFrame(renewable_payload.get("actuals", []))
        if not renewable_actuals.empty:
            renewable_actuals = as_utc(renewable_actuals, "utc_timestamp")
            for col in ["solar_actual_mwh", "wind_actual_mwh"]:
                if col in renewable_actuals.columns:
                    renewable_actuals[col] = pd.to_numeric(renewable_actuals[col], errors="coerce")
        renewable_forecasts = renewable_payload.get("forecasts", {}) or {}
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
        for kind, label in (("solar", "Solar"), ("wind", "Wind")):
            stats = renewable_summary[kind]
            st.markdown(f"**{label}** · MWh per {station_interval_minutes}-minute interval")
            r1, r2, r3, r4 = st.columns(4, gap="small")
            def fmt_mwh(value):
                return f"{value:,.0f} MWh" if value is not None else "—"
            r1.metric("Latest actual", fmt_mwh(stats["latest_actual"]))
            r2.metric("Next forecast", fmt_mwh(stats["first_forecast"]))
            r3.metric("Forecast peak", fmt_mwh(stats["peak_forecast"]))
            r4.metric("Forecast average", fmt_mwh(stats["average_forecast"]))

        renewable_fig, renewable_counts = build_renewable_figure(
            renewable_actuals, renewable_forecasts,
            chart_start=chart_start, chart_end=chart_end,
        )
        renewable_series_added = any(renewable_counts.values())
        if renewable_series_added:
            st.plotly_chart(renewable_fig, use_container_width=True)
        else:
            st.info("Solar/wind actual and forecast records are not available for the selected Amprion time window.")
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
        st.warning(f"Renewable monitoring data is temporarily unavailable from FastAPI: {exc}")

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

            st.dataframe(
                latest_run[display_columns],
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "Download Latest Forecast Run",
                data=latest_run[display_columns].to_csv(index=False).encode("utf-8"),
                file_name=f"{station}_latest_forecast.csv",
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
                    yaxis_title="Predicted demand (MW)",
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

fragment_runner = st.fragment(
    run_every=float(refresh_seconds) if auto_refresh else None
)(render_live_monitoring_panel)
fragment_runner(api, station, stations, history_hours, show_forecast_table, auto_refresh,
                refresh_seconds, 
                show_diagnostics)
