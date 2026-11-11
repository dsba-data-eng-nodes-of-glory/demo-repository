"""Pure Plotly builder for the combined Amprion solar/wind monitoring panel."""
from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _utc(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    if frame.empty or column not in frame.columns:
        return pd.DataFrame()
    out = frame.copy()
    out[column] = pd.to_datetime(out[column], utc=True, errors="coerce")
    return out.dropna(subset=[column]).sort_values(column).reset_index(drop=True)


def build_renewable_figure(
    actual_records: pd.DataFrame | list[dict[str, Any]],
    forecast_records: dict[str, list[dict[str, Any]]] | None,
    *,
    chart_start: Any = None,
    chart_end: Any = None,
) -> tuple[go.Figure, dict[str, int]]:
    """Build one panel with separate Solar PV and Wind rows on a shared UTC axis.

    Separate rows retain a common time scale but give the lower-magnitude solar
    signal its own y-range, so it cannot be visually flattened by wind output.
    Returns exact displayed trace counts for dashboard diagnostics and tests.
    """
    actuals = pd.DataFrame(actual_records).copy()
    actuals = _utc(actuals, "utc_timestamp")
    forecasts = forecast_records or {}
    start = pd.to_datetime(chart_start, utc=True, errors="coerce") if chart_start is not None else None
    end = pd.to_datetime(chart_end, utc=True, errors="coerce") if chart_end is not None else None

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12,
        subplot_titles=("Solar PV generation", "Onshore wind generation"),
        row_heights=[0.5, 0.5],
    )
    counts = {"solar_actual": 0, "solar_forecast": 0, "wind_actual": 0, "wind_forecast": 0}
    specs = [
        ("solar", "solar_actual_mwh", "Solar", "#D69E00", 1),
        ("wind", "wind_actual_mwh", "Wind", "#159D82", 2),
    ]
    for kind, actual_col, label, line_color, row in specs:
        if not actuals.empty and actual_col in actuals.columns:
            actual_part = actuals.copy()
            actual_part[actual_col] = pd.to_numeric(actual_part[actual_col], errors="coerce")
            actual_part = actual_part.dropna(subset=[actual_col])
            if start is not None and pd.notna(start):
                actual_part = actual_part[actual_part["utc_timestamp"] >= start]
            if end is not None and pd.notna(end):
                actual_part = actual_part[actual_part["utc_timestamp"] <= end]
            if not actual_part.empty:
                fig.add_trace(go.Scatter(
                    x=actual_part["utc_timestamp"], y=actual_part[actual_col],
                    mode="lines", name=f"{label} actual", legendgroup=kind,
                    line=dict(color=line_color, width=2.5), connectgaps=False,
                    hovertemplate=(
                        f"{label} actual: %{{y:,.2f}} MWh / 15 min<br>"
                        "%{x|%d %b %H:%M UTC}<extra></extra>"
                    ),
                ), row=row, col=1)
                counts[f"{kind}_actual"] = len(actual_part)

        forecast_part = pd.DataFrame(forecasts.get(kind, []))
        forecast_part = _utc(forecast_part, "forecast_timestamp_utc")
        if not forecast_part.empty:
            if "predicted_generation_mwh" not in forecast_part.columns:
                forecast_part = pd.DataFrame()
            else:
                forecast_part["predicted_generation_mwh"] = pd.to_numeric(
                    forecast_part["predicted_generation_mwh"], errors="coerce"
                )
                forecast_part = forecast_part.dropna(subset=["predicted_generation_mwh"])
        if not forecast_part.empty and "forecast_origin_utc" in forecast_part.columns:
            forecast_part["forecast_origin_utc"] = pd.to_datetime(
                forecast_part["forecast_origin_utc"], utc=True, errors="coerce"
            )
            origins = forecast_part["forecast_origin_utc"].dropna()
            if not origins.empty:
                forecast_part = forecast_part[forecast_part["forecast_origin_utc"] == origins.max()]
        if not forecast_part.empty:
            if start is not None and pd.notna(start):
                forecast_part = forecast_part[forecast_part["forecast_timestamp_utc"] >= start]
            if end is not None and pd.notna(end):
                forecast_part = forecast_part[forecast_part["forecast_timestamp_utc"] <= end]
        if not forecast_part.empty:
            fig.add_trace(go.Scatter(
                x=forecast_part["forecast_timestamp_utc"],
                y=forecast_part["predicted_generation_mwh"],
                mode="lines", name=f"{label} forecast", legendgroup=kind,
                line=dict(color=line_color, width=2.5, dash="dash"), connectgaps=False,
                hovertemplate=(
                    f"{label} forecast: %{{y:,.2f}} MWh / 15 min<br>"
                    "%{x|%d %b %H:%M UTC}<extra></extra>"
                ),
            ), row=row, col=1)
            counts[f"{kind}_forecast"] = len(forecast_part)

    if any(counts.values()):
        if start is not None and pd.notna(start) and end is not None and pd.notna(end):
            fig.update_xaxes(range=[start, end])
        fig.update_layout(
            height=620, template="plotly_white", hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            margin=dict(l=45, r=25, t=80, b=45),
        )
        fig.update_xaxes(
            showgrid=True, gridcolor="#E5E7EB", tickformat="%d %b<br>%H:%M",
            automargin=True, row=2, col=1,
        )
        fig.update_xaxes(showticklabels=False, row=1, col=1)
        fig.update_yaxes(
            title_text="Solar (MWh / 15 min)", showgrid=True,
            gridcolor="#E5E7EB", rangemode="tozero", row=1, col=1,
        )
        fig.update_yaxes(
            title_text="Wind (MWh / 15 min)", showgrid=True,
            gridcolor="#E5E7EB", rangemode="tozero", row=2, col=1,
        )
    return fig, counts
