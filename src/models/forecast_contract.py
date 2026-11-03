"""Time-based forecasting contract shared by registration, training and serving.

The user selects a source sampling interval in minutes. The forecasting product
keeps a seven-day historical lookback and a 24-hour forecast horizon; step
counts are derived from those durations for each station.
"""
from __future__ import annotations

from typing import Any


def _duration_minutes(config: dict[str, Any], duration_key: str, steps_key: str,
                      legacy_interval_key: str, legacy_interval_default: int,
                      legacy_steps_default: int, duration_default: int) -> int:
    """Resolve a duration in minutes, supporting older model.yaml files."""
    if duration_key in config and config[duration_key] is not None:
        value = float(config[duration_key])
        if value <= 0:
            raise ValueError(f"{duration_key} must be greater than zero.")
        return int(round(value * 60))
    legacy_interval = int(config.get(legacy_interval_key, legacy_interval_default))
    legacy_steps = int(config.get(steps_key, legacy_steps_default))
    return legacy_steps * legacy_interval if legacy_steps > 0 and legacy_interval > 0 else duration_default


def derive_forecast_contract(
    interval_minutes: int,
    model_config: dict[str, Any] | None = None,
    *,
    history_window_hours: float | None = None,
    forecast_horizon_hours: float | None = None,
) -> dict[str, int | float]:
    """Return history/horizon steps for a selected sampling interval.

    Intervals must be whole minutes and divide both configured duration windows
    exactly. This guarantees that the model forecast ends exactly 24 hours after
    its origin and uses exactly seven days of history.
    """
    cfg = model_config or {}
    try:
        interval = int(interval_minutes)
    except (TypeError, ValueError) as exc:
        raise ValueError("Source observation interval must be a whole number of minutes.") from exc
    if interval < 5 or interval > 1440:
        raise ValueError("Source observation interval must be between 5 and 1440 minutes.")

    if history_window_hours is None:
        history_minutes = _duration_minutes(
            cfg, "history_window_hours", "history_steps", "interval_minutes", 15, 672, 10080
        )
    else:
        history_minutes = int(round(float(history_window_hours) * 60))
    if forecast_horizon_hours is None:
        horizon_minutes = _duration_minutes(
            cfg, "forecast_horizon_hours", "horizon_steps", "interval_minutes", 15, 96, 1440
        )
    else:
        horizon_minutes = int(round(float(forecast_horizon_hours) * 60))

    if history_minutes <= 0 or horizon_minutes <= 0:
        raise ValueError("History window and forecast horizon must be positive durations.")
    if history_minutes % interval != 0 or horizon_minutes % interval != 0:
        raise ValueError(
            f"The selected {interval}-minute interval does not divide the configured "
            f"history ({history_minutes} minutes) and forecast horizon ({horizon_minutes} minutes) exactly. "
            "Choose an interval that produces whole steps in both windows."
        )

    history_steps = history_minutes // interval
    horizon_steps = horizon_minutes // interval
    if history_steps < 2 or horizon_steps < 1:
        raise ValueError("The selected interval produces too few history or forecast steps.")

    return {
        "interval_minutes": interval,
        "history_window_minutes": history_minutes,
        "forecast_horizon_minutes": horizon_minutes,
        "history_window_hours": history_minutes / 60.0,
        "forecast_horizon_hours": horizon_minutes / 60.0,
        "history_steps": history_steps,
        "horizon_steps": horizon_steps,
    }


def station_forecast_contract(
    station: dict[str, Any], model_config: dict[str, Any] | None = None
) -> dict[str, int | float]:
    """Resolve a station's interval against the shared duration contract."""
    cfg = model_config or {}
    interval = station.get("interval_minutes", cfg.get("interval_minutes", 15))
    return derive_forecast_contract(int(interval), cfg)
