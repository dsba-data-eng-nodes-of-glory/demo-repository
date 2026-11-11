"""Pure, replay-safe KPI summaries for the existing Solar and Wind forecast feeds.

Use actuals *after* the monitoring replay watermark filter, and only the most
recent forecast origin for each component. Do not conflate MWh interval energy
with MW instantaneous demand; the former is MWh per source interval.
"""
from __future__ import annotations

from typing import Any
import math
import pandas as pd


def _valid_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (ValueError, TypeError):
        return None
    return number if math.isfinite(number) else None


def _series_summary(actual: pd.DataFrame, forecast_rows: Any, actual_column: str) -> dict[str, float | None]:
    latest_actual = None
    if not actual.empty and {"utc_timestamp", actual_column}.issubset(actual.columns):
        observed = actual[["utc_timestamp", actual_column]].copy()
        observed["utc_timestamp"] = pd.to_datetime(observed["utc_timestamp"], utc=True, errors="coerce")
        observed[actual_column] = pd.to_numeric(observed[actual_column], errors="coerce")
        observed = observed.dropna().sort_values("utc_timestamp")
        if not observed.empty:
            latest_actual = _valid_number(observed[actual_column].iloc[-1])

    result = {"latest_actual": latest_actual, "first_forecast": None, "peak_forecast": None,
              "average_forecast": None, "forecast_steps": 0}
    future = pd.DataFrame(forecast_rows or [])
    if future.empty or "predicted_generation_mwh" not in future.columns:
        return result
    if "forecast_origin_utc" in future.columns:
        future["forecast_origin_utc"] = pd.to_datetime(future["forecast_origin_utc"], utc=True, errors="coerce")
        if future["forecast_origin_utc"].notna().any():
            origin = future["forecast_origin_utc"].max()
            future = future[future["forecast_origin_utc"].eq(origin)]
    if "forecast_timestamp_utc" in future.columns:
        future["forecast_timestamp_utc"] = pd.to_datetime(future["forecast_timestamp_utc"], utc=True, errors="coerce")
        future = (future.dropna(subset=["forecast_timestamp_utc"])
                        .sort_values("forecast_timestamp_utc")
                        .drop_duplicates("forecast_timestamp_utc", keep="last"))
    elif "horizon_step" in future.columns:
        future["horizon_step"] = pd.to_numeric(future["horizon_step"], errors="coerce")
        future = future.dropna(subset=["horizon_step"]).sort_values("horizon_step")
    values = pd.to_numeric(future["predicted_generation_mwh"], errors="coerce").replace([float("inf"), -float("inf")], float("nan")).dropna()
    if values.empty:
        return result
    result.update(first_forecast=float(values.iloc[0]), peak_forecast=float(values.max()),
                  average_forecast=float(values.mean()), forecast_steps=int(len(values)))
    return result


def build_renewable_summaries(
    visible_actuals: pd.DataFrame,
    forecast_records: dict[str, list[dict[str, Any]]] | None,
) -> dict[str, dict[str, float | None]]:
    """Return Solar and Wind KPIs without generating more API requests."""
    forecast_records = forecast_records or {}
    return {
        "solar": _series_summary(visible_actuals, forecast_records.get("solar"), "solar_actual_mwh"),
        "wind": _series_summary(visible_actuals, forecast_records.get("wind"), "wind_actual_mwh"),
    }
