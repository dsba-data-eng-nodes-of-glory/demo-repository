"""Point-in-time matched renewable forecasting accuracy.

Actual solar/wind generation must be restricted to an actual observation
watermark BEFORE matching to the saved forecasts. A static test CSV may include
future observations, so evaluating it without that watermark is data leakage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGET_COL = {"solar": "solar_actual_mwh", "wind": "wind_actual_mwh"}


def match_renewable_forecasts(
    forecasts: pd.DataFrame,
    actuals: pd.DataFrame,
    kind: str,
    watermark_utc: str | pd.Timestamp | None,
    *,
    horizon_step: int = 1,
    limit: int = 300,
) -> pd.DataFrame:
    if kind not in TARGET_COL:
        raise ValueError("kind must be 'solar' or 'wind'")
    fields = ["forecast_timestamp_utc", "forecast_origin_utc", "actual_mwh", "predicted_mwh", "abs_error_mwh"]
    if watermark_utc is None or forecasts.empty or actuals.empty:
        return pd.DataFrame(columns=fields)
    cutoff = pd.to_datetime(watermark_utc, utc=True, errors="coerce")
    if pd.isna(cutoff):
        return pd.DataFrame(columns=fields)
    target_col = TARGET_COL[kind]
    if not {"forecast_timestamp_utc", "forecast_origin_utc", "predicted_generation_mwh"}.issubset(forecasts.columns) or not {"utc_timestamp", target_col}.issubset(actuals.columns):
        return pd.DataFrame(columns=fields)
    forecast = forecasts.copy()
    if "generation_type" in forecast:
        forecast = forecast[forecast["generation_type"].eq(kind)]
    forecast["forecast_timestamp_utc"] = pd.to_datetime(forecast["forecast_timestamp_utc"], utc=True, errors="coerce")
    forecast["forecast_origin_utc"] = pd.to_datetime(forecast["forecast_origin_utc"], utc=True, errors="coerce")
    if "horizon_step" in forecast:
        forecast["horizon_step"] = pd.to_numeric(forecast["horizon_step"], errors="coerce")
        forecast = forecast[forecast["horizon_step"].eq(horizon_step)]
    else:
        mins = (forecast["forecast_timestamp_utc"] - forecast["forecast_origin_utc"]).dt.total_seconds() / 60
        forecast = forecast[mins.eq(horizon_step * 15)]
    forecast["predicted_mwh"] = pd.to_numeric(forecast["predicted_generation_mwh"], errors="coerce")
    forecast = forecast.dropna(subset=["forecast_timestamp_utc", "forecast_origin_utc", "predicted_mwh"])
    # Every forecast must precede its own target timestamp (no ex-post 'forecasts').
    forecast = forecast[
        (forecast["forecast_origin_utc"] < forecast["forecast_timestamp_utc"])
        & (forecast["forecast_timestamp_utc"] <= cutoff)
    ]
    # Do not compare forecast_issued_at_utc (physical wall clock) with
    # forecast target times during replay. A forecast for October can be
    # calculated during a November replay. forecast_origin_utc is the
    # event-time origin and must strictly precede the target.
    if forecast.empty:
        return pd.DataFrame(columns=fields)
    forecast = forecast.sort_values("forecast_origin_utc").drop_duplicates("forecast_timestamp_utc", keep="last")
    actual = actuals[["utc_timestamp", target_col]].copy()
    actual["utc_timestamp"] = pd.to_datetime(actual["utc_timestamp"], utc=True, errors="coerce")
    actual["actual_mwh"] = pd.to_numeric(actual[target_col], errors="coerce")
    actual = actual[(actual["utc_timestamp"] <= cutoff)].dropna(subset=["utc_timestamp", "actual_mwh"])
    actual = actual.sort_values("utc_timestamp").drop_duplicates("utc_timestamp", keep="last")
    joined = forecast.merge(actual[["utc_timestamp", "actual_mwh"]], left_on="forecast_timestamp_utc", right_on="utc_timestamp", how="inner")
    if joined.empty:
        return pd.DataFrame(columns=fields)
    joined["abs_error_mwh"] = (joined["predicted_mwh"] - joined["actual_mwh"]).abs()
    return joined.sort_values("forecast_timestamp_utc").tail(limit).reset_index(drop=True)[fields]


def renewable_metrics(matched: pd.DataFrame) -> dict:
    if matched.empty:
        return {"matched_points": 0, "mae_mwh": None, "rmse_mwh": None, "wape_pct": None, "bias_mwh": None}
    error = matched["predicted_mwh"].to_numpy(dtype=float) - matched["actual_mwh"].to_numpy(dtype=float)
    actual = matched["actual_mwh"].to_numpy(dtype=float)
    denom = np.abs(actual).sum()
    return {
        "matched_points": len(matched),
        "mae_mwh": float(np.mean(np.abs(error))),
        "rmse_mwh": float(np.sqrt(np.mean(np.square(error)))),
        "wape_pct": float(np.abs(error).sum() / denom * 100) if denom > 0 else None,
        "bias_mwh": float(np.mean(error)),
    }
