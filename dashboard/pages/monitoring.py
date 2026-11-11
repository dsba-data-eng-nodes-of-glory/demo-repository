"""Electricity demand monitoring: actual history connected to the selected model forecast."""

from datetime import date, timedelta
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

try:
    from dashboard.ui import setup_page
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from dashboard.ui import setup_page

try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None


@st.cache_data(ttl=5, show_spinner=False)
def api_get(api_url: str, endpoint: str, params: dict | None = None) -> dict:
    response = requests.get(
        f"{api_url.rstrip('/')}{endpoint}", params=params, timeout=15
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

st.title("⚡ Electricity Demand — Live Monitoring")
st.caption(
    "Actual demand transitions into the latest configured forecast · "
    "Data loaded through FastAPI · All timestamps in UTC"
)

# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("Monitoring Controls")
    auto_refresh = st.checkbox("Enable auto-refresh", value=True)
    refresh_seconds = st.selectbox(
        "Refresh interval",
        [2, 5, 10, 15],
        index=1,
        format_func=lambda seconds: f"Every {seconds} seconds",
    )

    if auto_refresh:
        if st_autorefresh is not None:
            st_autorefresh(
                interval=refresh_seconds * 1000,
                key="electricity_poc_monitoring_refresh",
            )
        else:
            st.warning(
                "Auto-refresh package is missing. Install it with: "
                "pip install streamlit-autorefresh"
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

# ---------------------------------------------------------------------
# Station selection
# ---------------------------------------------------------------------
try:
    registry = api_get(api, "/stations")
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
        api, "/results/forecasts", {"station": station, "limit": 5000}
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
# Identify the latest complete forecast run
# ---------------------------------------------------------------------
latest_run = pd.DataFrame()
forecast_origin = None

if (
    not forecasts.empty
    and "forecast_issued_at_utc" in forecasts.columns
):
    valid_issue_times = forecasts["forecast_issued_at_utc"].dropna()

    if not valid_issue_times.empty:
        latest_issue_time = valid_issue_times.max()
        latest_run = forecasts[
            forecasts["forecast_issued_at_utc"] == latest_issue_time
        ].sort_values("forecast_timestamp_utc").reset_index(drop=True)

        # Prefer the explicit origin saved by the predictor.
        if (
            "forecast_origin_utc" in latest_run.columns
            and latest_run["forecast_origin_utc"].notna().any()
        ):
            forecast_origin = latest_run["forecast_origin_utc"].dropna().iloc[0]
        elif not latest_run.empty:
            # Fallback: first target timestamp minus the station's configured interval.
            forecast_origin = (
                latest_run["forecast_timestamp_utc"].iloc[0]
                - pd.Timedelta(minutes=station_interval_minutes)
            )

# Load actuals only after deriving the latest forecast origin. The API uses
# this cutoff so later rows in a historical source CSV cannot displace the
# currently replayed timeline. 5,000 rows comfortably covers the 72-hour view.
actual_params = {"station": station, "limit": 5000}
if forecast_origin is not None:
    actual_params["end_time"] = forecast_origin.isoformat()
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

st.subheader("Demand Overview")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Latest Stored Actual", fmt_mw(latest_actual))
k2.metric("First Forecast Step", fmt_mw(first_forecast))
k3.metric("24h Peak Forecast", fmt_mw(peak_forecast))
k4.metric("24h Average Forecast", fmt_mw(average_forecast))

forecast_annotations = []
if not latest_run.empty and "predicted_demand_class" in latest_run.columns:
    forecast_annotations.append(("First-step demand class", latest_run.iloc[0]["predicted_demand_class"]))
if not latest_run.empty and "predicted_cluster" in latest_run.columns:
    forecast_annotations.append(("First-step K-Means cluster", latest_run.iloc[0]["predicted_cluster"]))
if forecast_annotations:
    annotation_columns = st.columns(len(forecast_annotations))
    for column, (label, value) in zip(annotation_columns, forecast_annotations):
        column.metric(label, str(value) if pd.notna(value) else "—")

if forecast_origin is not None:
    st.caption(f"Forecast origin (Now): {fmt_utc(forecast_origin)}")
if not latest_run.empty and "forecast_issued_at_utc" in latest_run.columns:
    st.caption(
        "Latest forecast run issued: "
        + fmt_utc(latest_run["forecast_issued_at_utc"].iloc[0])
    )

# ---------------------------------------------------------------------
# Main graph: solid actual line -> dashed forecast line
# ---------------------------------------------------------------------
st.divider()
st.subheader("Live Demand Monitoring")
st.caption(
    "Solid line: actual demand · Dashed line: latest configured forecast · "
    "Vertical marker: forecast origin (Now)"
)

fig = go.Figure()

if forecast_origin is not None and not actuals.empty and not latest_run.empty:
    # The historical actual line must stop at the forecast origin.
    # This prevents later replayed actuals from appearing in the future area.
    actual_history = actuals[
        actuals["utc_timestamp"] <= forecast_origin
    ].copy()

    # Show the selected amount of actual history before the forecast origin.
    history_start = forecast_origin - pd.Timedelta(hours=history_hours)
    actual_history = actual_history[
        actual_history["utc_timestamp"] >= history_start
    ].copy()

    # Find the last actual observation at or before the origin.
    anchor_rows = actuals[
        actuals["utc_timestamp"] <= forecast_origin
    ].sort_values("utc_timestamp")

    if not anchor_rows.empty:
        anchor = anchor_rows.iloc[-1]
        anchor_time = anchor["utc_timestamp"]
        anchor_value = float(anchor["actual_demand"])

        # Actual history: blue solid line.
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

        # Forecast starts at the exact same anchor as the actual line.
        # This makes the two traces visually join at the Now boundary.
        forecast_x = [anchor_time] + latest_run[
            "forecast_timestamp_utc"
        ].tolist()
        forecast_y = [anchor_value] + latest_run[
            "predicted_demand"
        ].astype(float).tolist()

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

        # Vertical Now marker.
        fig.add_vline(
            x=forecast_origin,
            line_width=2,
            line_dash="dash",
            line_color="#888888",
            annotation_text="Now",
            annotation_position="top",
            annotation_font_color="#555555",
        )

        # Exact chart window: selected actual history + next 24 hours.
        chart_end = (
            latest_run["forecast_timestamp_utc"].max()
            if not latest_run.empty
            else forecast_origin + pd.Timedelta(hours=station_forecast_horizon_hours)
        )

        # Use fewer, readable labels across the 96-hour chart window.
        # Three-hour ticks become crowded when 72 hours of actuals plus
        # 24 hours of forecast are displayed.
        tick_interval_hours = 6 if history_hours >= 48 else 3
        fig.update_xaxes(
            range=[history_start, chart_end],
            dtick=tick_interval_hours * 60 * 60 * 1000,
            tickformat="%d %b<br>%H:%M",
            tickangle=0,
            tickfont=dict(size=11),
            automargin=True,
            ticklabeloverflow="allow",
        )

    else:
        st.warning(
            "No actual observation exists at or before the latest forecast "
            "origin. The forecast cannot be anchored to the actual line."
        )

elif not actuals.empty:
    # Actual-only fallback while waiting for the first forecast run.
    latest_time = actuals["utc_timestamp"].max()
    actual_history = actuals[
        actuals["utc_timestamp"] >= (
            latest_time - pd.Timedelta(hours=history_hours)
        )
    ]

    fig.add_trace(
        go.Scatter(
            x=actual_history["utc_timestamp"],
            y=actual_history["actual_demand"],
            mode="lines",
            name="Actual demand",
            line=dict(color="#1976D2", width=3.5),
        )
    )
    st.info("Waiting for the first saved forecast run.")

else:
    st.info("Waiting for actual-demand observations and forecast records.")

if fig.data:
    fig.update_layout(
        height=540,
        template="plotly_white",
        hovermode="x unified",
        xaxis_title="Time (UTC)",
        yaxis_title="Electricity demand (MW)",
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
        showgrid=True,
        gridcolor="#E5E7EB",
        zeroline=False,
        rangeslider_visible=False,
        tickangle=0,
        tickfont=dict(size=11),
        automargin=True,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#E5E7EB",
        zeroline=False,
    )
    st.plotly_chart(fig, use_container_width=True)

if not latest_run.empty and (
    "predicted_demand_class" in latest_run.columns
    or "predicted_cluster" in latest_run.columns
):
    annotation_views = st.columns(2)
    if "predicted_demand_class" in latest_run.columns:
        with annotation_views[0]:
            st.subheader("24-hour demand classes")
            class_counts = (
                latest_run["predicted_demand_class"].dropna().astype(str)
                .value_counts().rename_axis("Class").reset_index(name="Intervals")
            )
            if not class_counts.empty:
                st.bar_chart(class_counts, x="Class", y="Intervals", height=260)
    if "predicted_cluster" in latest_run.columns:
        with annotation_views[1]:
            st.subheader("24-hour K-Means clusters")
            cluster_counts = (
                pd.to_numeric(latest_run["predicted_cluster"], errors="coerce").dropna()
                .astype(int).value_counts().sort_index().rename_axis("Cluster")
                .reset_index(name="Intervals")
            )
            if not cluster_counts.empty:
                st.bar_chart(cluster_counts, x="Cluster", y="Intervals", height=260)

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
            "predicted_demand_class",
            "predicted_cluster",
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
                name="Saved LSTM Forecast",
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
            template="plotly_white",
        )
        st.plotly_chart(history_fig, use_container_width=True)
        st.caption(f"Saved forecast runs available: {len(run_times):,}")
else:
    st.info("Forecast history will appear after the predictor saves its first run.")

# ---------------------------------------------------------------------
# Optional forecast table (step count is station-specific)
# ---------------------------------------------------------------------
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
with st.expander("Storage and data status", expanded=False):
    st.write("**Storage type:** Local CSV files accessed through FastAPI.")
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

st.divider()
st.caption(
    f"Auto-refresh: "
    f"{'On' if auto_refresh and st_autorefresh is not None else 'Off / package missing'}"
    f" · Refresh interval: {refresh_seconds}s"
    f" · Last render: {pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M:%S UTC}"
)
