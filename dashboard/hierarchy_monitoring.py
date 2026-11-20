"""Live monitoring for derived nodes in the station hierarchy."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.monitoring_chart import ACTUAL_COLOR, FORECAST_COLOR, apply_monitoring_axes, mark_now, render_initializing
from dashboard.monitoring_progress import eligible_forecast, forecast_cursor, timeline_is_live, utc
from dashboard.monitoring_units import interval_unit_label, unit_factor

METRICS = {
    "demand_mw": ("Demand", "MW"),
    "solar_mwh": ("Solar", "MWh"),
    "wind_mwh": ("Wind Energy", "MWh"),
    "residual_mwh": ("Residual Load", "MWh"),
}


def _covered(row: dict, metric: str, leaf_count: int) -> bool:
    point = row.get(metric) or {}
    return point.get("value") is not None and point.get("reporting") == leaf_count


def group_cursor(payload: dict, metric: str, include_forecasts: bool) -> pd.Timestamp | None:
    """Use the moving shared observation or the matching complete forecast."""
    count = payload["leaf_count"]
    actuals = {utc(row["utc_timestamp"]) for row in payload.get("actuals", []) if _covered(row, metric, count)}
    actuals.discard(None)
    latest = max(actuals, default=None)
    start = payload.get("replay_start_utc")
    if not include_forecasts:
        return eligible_forecast(latest, latest, start)
    origin = utc((payload.get("forecast_origins_utc") or {}).get(metric))
    if origin is None or origin not in actuals:
        return None
    records = [
        {"forecast_timestamp_utc": row["utc_timestamp"], "forecast_origin_utc": origin}
        for row in payload.get("forecasts", []) if _covered(row, metric, count)
    ]
    return forecast_cursor(records, latest, start, payload["horizon_steps"], payload["interval_minutes"])


def build_group_chart(payload: dict, node_id: str, metric: str, cursor: pd.Timestamp, *,
                      history_hours: int, include_forecasts: bool, display_unit: str) -> go.Figure:
    label, source_unit = METRICS[metric]
    interval = payload["interval_minutes"]
    suffix = interval_unit_label(display_unit, interval)
    factor = unit_factor(source_unit, display_unit, interval)
    leaf_count = payload["leaf_count"]
    chart_start = cursor - pd.Timedelta(hours=history_hours)
    actuals = [(stamp, row[metric]) for row in payload.get("actuals", [])
               if (stamp := utc(row["utc_timestamp"])) is not None and chart_start <= stamp <= cursor]
    hover = "%{x|%d %b %H:%M UTC}<br>%{y:,.2f} " + suffix
    chart = go.Figure()
    chart.add_trace(go.Scatter(
        x=[stamp for stamp, _ in actuals],
        y=[point["value"] * factor if point["value"] is not None and point["reporting"] == leaf_count else None
           for _, point in actuals],
        mode="lines", name=f"Actual {label.lower()}", line=dict(color=ACTUAL_COLOR, width=3.5),
        connectgaps=False, hovertemplate=hover + "<extra></extra>",
    ))
    partial = [(stamp, point) for stamp, point in actuals
               if point["value"] is not None and 0 < point["reporting"] < leaf_count]
    if partial:
        chart.add_trace(go.Scatter(
            x=[stamp for stamp, _ in partial], y=[point["value"] * factor for _, point in partial],
            mode="markers", name="Partial actual coverage", visible="legendonly",
            marker=dict(color="#fbbf24", size=7),
            customdata=[(point["reporting"], leaf_count) for _, point in partial],
            hovertemplate=hover + "<br>%{customdata[0]}/%{customdata[1]} substations<extra></extra>",
        ))
    chart_end = cursor
    if include_forecasts:
        forecast = [(utc(row["utc_timestamp"]), row[metric]["value"] * factor)
                    for row in payload.get("forecasts", []) if _covered(row, metric, leaf_count)]
        anchor = next((point["value"] * factor for stamp, point in actuals
                       if stamp == cursor and point["reporting"] == leaf_count), None)
        if anchor is not None:
            forecast.insert(0, (cursor, anchor))
        if forecast:
            chart_end = forecast[-1][0]
            chart.add_trace(go.Scatter(
                x=[stamp for stamp, _ in forecast], y=[value for _, value in forecast],
                mode="lines", name="Forecast", line=dict(color=FORECAST_COLOR, width=3.5, dash="dash"),
                connectgaps=False, hovertemplate=hover + "<extra></extra>",
            ))
    mark_now(chart, cursor)
    ylabel = "Electricity demand" if metric == "demand_mw" else label
    apply_monitoring_axes(chart, chart_start, chart_end, history_hours, f"{ylabel} ({suffix})",
                          display_unit, f"hierarchy:{node_id}:{metric}:{history_hours}:{include_forecasts}")
    return chart


def render_group(
    payload: dict, node: dict, metric: str | None = None, *,
    progress: dict, history_hours: int, include_forecasts: bool = True,
    show_forecast_table: bool = False, display_unit: str = "MWh",
) -> None:
    st.subheader(node["name"])
    leaf_count = payload["leaf_count"]
    substations = payload.get("substations", [])
    edges = [stamp for row in substations if (stamp := utc(row.get("latest_demand_utc"))) is not None]
    if len(edges) == leaf_count and len(edges) > 1:
        lag = max(edges) - min(edges)
        if lag >= pd.Timedelta(minutes=payload["interval_minutes"]):
            st.caption(f"Substation replay times differ by {lag}. The charts align to the slowest station.")
    if include_forecasts and not payload.get("forecast_compatible", True):
        st.caption("A common forecast horizon is needed to sum forecasts for this group.")
        include_forecasts = False
    if payload.get("recovery"):
        recovery = payload["recovery"]
        st.caption(f"Demand recovery · {recovery['processed']}/{recovery['total']} Kafka records processed")

    run = tuple(sorted((payload.get("predictor_runs") or {}).items()))
    for name in (metric,) if metric else METRICS:
        label, source_unit = METRICS[name]
        cursor = group_cursor(payload, name, include_forecasts)
        key = ("hierarchy", node["id"], name, include_forecasts)
        if not timeline_is_live(progress, key, cursor, run):
            detail = ("Waiting for a complete shared forecast and a newer aligned forecast origin."
                      if include_forecasts else "Waiting for the next observation with complete substation coverage.")
            render_initializing(f"{node['name']} {label}", detail,
                                f"Shared interval · {cursor:%d %b %H:%M UTC}" if cursor is not None
                                else f"Demand readings received from {len(edges)}/{leaf_count} substations.")
            continue
        _render_metric(payload, node["id"], name, cursor, history_hours, include_forecasts,
                       show_forecast_table, display_unit, label, source_unit)

    with st.expander("Substation reporting details"):
        if st.checkbox("Show latest demand by substation", key=f"reporting_{node['id']}"):
            st.dataframe(substations, use_container_width=True, hide_index=True)


def _render_metric(payload, node_id, metric, cursor, history_hours, include_forecasts,
                   show_forecast_table, display_unit, label, source_unit):
    st.subheader(f"{label} · actual vs forecast" if include_forecasts else f"{label} · actual")
    interval = payload["interval_minutes"]
    factor = unit_factor(source_unit, display_unit, interval)
    count = payload["leaf_count"]
    latest = next((row[metric]["value"] for row in payload["actuals"]
                   if utc(row["utc_timestamp"]) == cursor and _covered(row, metric, count)), None)
    forecast_rows = [row for row in payload.get("forecasts", []) if _covered(row, metric, count)] if include_forecasts else []
    values = [row[metric]["value"] for row in forecast_rows]
    summary = (
        ("Latest actual", latest),
        ("Next forecast", values[0] if values else None),
        ("Forecast peak", max(values) if values else None),
        ("Forecast average", sum(values) / len(values) if values else None),
    )
    for (caption, value), column in zip(summary, st.columns(4, gap="small")):
        column.metric(caption, "—" if value is None else f"{value * factor:,.2f} {display_unit}")
    if include_forecasts:
        st.caption(f"Forecast origin · {cursor:%d %b %H:%M UTC}")
    chart = build_group_chart(payload, node_id, metric, cursor, history_hours=history_hours,
                              include_forecasts=include_forecasts, display_unit=display_unit)
    st.plotly_chart(chart, use_container_width=True, key=f"hierarchy:{node_id}:{metric}")
    if show_forecast_table and forecast_rows:
        suffix = interval_unit_label(display_unit, interval)
        st.dataframe([{"Forecast time (UTC)": row["utc_timestamp"], f"{label} ({suffix})": row[metric]["value"] * factor}
                      for row in forecast_rows], use_container_width=True, hide_index=True)
