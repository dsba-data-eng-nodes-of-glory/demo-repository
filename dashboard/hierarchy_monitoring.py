"""Live monitoring for derived nodes in the station hierarchy."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit.components.v1 import html as component_html

METRICS = {
    "demand_mw": ("Demand", "MW"),
    "solar_mwh": ("Solar", "MWh"),
    "wind_mwh": ("Wind Energy", "MWh"),
    "residual_mwh": ("Residual load", "MWh"),
}
CLOCK_HTML = """
<style>
body { margin:0; font:14px system-ui,sans-serif; text-align:right; color:#90a8c2; }
strong { color:#38bdf8; font-variant-numeric:tabular-nums; }
</style>
<span>Live clock · UTC </span><strong id="clock"></strong>
<script>
const clock = document.getElementById("clock");
function renderClock() { clock.textContent = new Date().toISOString().slice(11, 19); }
renderClock();
setInterval(renderClock, 1000);
</script>
"""


def show_clock() -> None:
    if hasattr(st, "iframe"):
        st.iframe(CLOCK_HTML, height=34)
    else:
        component_html(CLOCK_HTML, height=34)


def _trace(rows: list[dict], metric: str, forecast: bool) -> go.Scatter:
    points = [row[metric] for row in rows]
    completeness = ["Complete" if point["reporting"] == point["expected"] else "Partial" for point in points]
    return go.Scatter(
        x=[row["utc_timestamp"] for row in rows],
        y=[point["value"] for point in points], mode="lines+markers",
        name="Forecast sum" if forecast else "Actual sum", connectgaps=False,
        line={"color": "#fb923c" if forecast else "#38bdf8", "dash": "dash" if forecast else "solid"},
        marker={
            "color": ["#fbbf24" if status == "Partial" else "#fb923c" if forecast else "#38bdf8"
                      for status in completeness],
            "size": [8 if status == "Partial" else 4 for status in completeness],
        },
        customdata=[(status, point["reporting"], point["expected"]) for status, point in zip(completeness, points)],
        hovertemplate="%{x|%d %b %H:%M UTC}<br>%{y:,.2f}<br>%{customdata[0]} · "
                      "%{customdata[1]}/%{customdata[2]} substations<extra></extra>",
    )


def render_group(payload: dict, node: dict, metric: str) -> None:
    label, unit = METRICS[metric]
    actuals = payload.get("actuals", [])
    forecasts = payload.get("forecasts", [])
    leaf_count = payload["leaf_count"]
    origin = payload.get("forecast_origin_utc")
    latest = actuals[-1] if actuals else None
    point = latest[metric] if latest else {"value": None, "reporting": 0}
    st.subheader(f"{node['name']} · {label}")
    col1, col2, col3 = st.columns(3)
    suffix = f"{unit} / {payload['interval_minutes']} min" if unit == "MWh" else unit
    value = "—" if point["value"] is None else f"{point['value']:,.2f} {suffix}"
    col1.metric("Latest observed total", value)
    coverage = "Complete" if point["reporting"] == leaf_count else "Partial" if point["reporting"] else "Waiting"
    col2.metric("Registered substations reporting", f"{point['reporting']} / {leaf_count}", coverage, delta_color="off")
    col3.metric("Latest interval (UTC)", pd.Timestamp(latest["utc_timestamp"]).strftime("%d %b %H:%M") if latest else "—")
    if origin and any(row[metric]["value"] is not None for row in forecasts):
        st.caption(f"Forecast origin · {pd.Timestamp(origin):%d %b %H:%M UTC}")
    if not payload.get("forecast_compatible", True):
        st.caption("A common forecast horizon is needed to sum forecasts for this group.")
    traces = [_trace(rows, metric, is_forecast) for rows, is_forecast in ((actuals, False), (forecasts, True))
              if any(row[metric]["value"] is not None for row in rows)]
    if traces:
        chart = go.Figure(traces)
        chart.update_layout(
            template="plotly_dark", height=430, hovermode="x unified",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(8,27,48,.50)",
            xaxis_title="Time (UTC)", yaxis_title=f"{label} ({suffix})",
            legend={"orientation": "h"}, uirevision=f"{node['id']}:{metric}",
        )
        st.plotly_chart(chart, use_container_width=True)
    else:
        st.info(f"Waiting for {label.lower()} readings from the registered substations.")
    if metric == "residual_mwh":
        st.caption("Residual load uses demand, solar and Wind Energy from the same substation and interval.")
    if payload.get("recovery"):
        progress = payload["recovery"]
        st.caption(f"Demand recovery · {progress['processed']}/{progress['total']} Kafka records processed")
    with st.expander("Substation reporting details"):
        if st.checkbox("Show latest demand by substation", key=f"reporting_{node['id']}"):
            st.dataframe(payload["substations"], use_container_width=True, hide_index=True)
