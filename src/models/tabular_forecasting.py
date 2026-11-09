"""Shared tabular feature contract for tree-based electricity forecasts.

One training row represents one (forecast origin, lead step) pair. Features
use only information available at the forecast origin, plus calendar and
lead-time features for each future target. Lag and rolling windows are defined
in real time (minutes/hours), then converted to the station's sample interval.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

LEGACY_FEATURE_SCHEMA = "electricity_tabular_v1"
FEATURE_SCHEMA = "electricity_tabular_v2"  # Existing day + week models.
DAY_ONLY_FEATURE_SCHEMA = "electricity_tabular_v3"
TARGET_DAY_FEATURE = "demand_target_previous_day"
TARGET_WEEK_FEATURE = "demand_target_previous_week"


def target_demand_references(
    history: pd.DataFrame, future_timestamps: pd.DatetimeIndex,
    *, include_previous_week: bool = True,
) -> dict[str, np.ndarray]:
    """Match each target to demand 24h/168h earlier using observed history only.

    These are fixed durations in UTC, not civil-clock matches across daylight
    saving transitions. Missing references are rejected rather than filled
    with future actuals or silently shifted to another timestamp.
    """
    timestamps = pd.DatetimeIndex(pd.to_datetime(history["utc_timestamp"], utc=True))
    targets = pd.DatetimeIndex(pd.to_datetime(future_timestamps, utc=True))
    demand = pd.to_numeric(history["demand"], errors="coerce").to_numpy(dtype=float)
    if timestamps.empty or timestamps.hasnans or not timestamps.is_unique:
        raise ValueError("Seasonal references require nonempty, unique, valid history timestamps.")
    if targets.empty or targets.hasnans or (targets <= timestamps.max()).any():
        raise ValueError("Forecast targets must be valid timestamps after observed history.")
    if not np.isfinite(demand).all():
        raise ValueError("Seasonal references require finite historical demand.")
    references = {}
    reference_specs = [(TARGET_DAY_FEATURE, 24)]
    if include_previous_week:
        reference_specs.append((TARGET_WEEK_FEATURE, 168))
    for name, hours in reference_specs:
        reference_times = targets - pd.Timedelta(hours=hours)
        positions = timestamps.get_indexer(reference_times)
        if (positions < 0).any():
            missing = reference_times[positions < 0][0]
            raise ValueError(
                f"{name} needs observed history at {missing}; the reference is unavailable. "
                "Provide seven days of continuous history and a horizon of at most 24 hours."
            )
        references[name] = demand[positions]
    return references


def _steps_for_duration(duration_minutes: int, interval_minutes: int) -> int:
    """Map a real-time duration to at least one observation step."""
    return max(1, int(round(float(duration_minutes) / float(interval_minutes))))


def _unique_steps(durations: list[int], interval_minutes: int) -> list[int]:
    values: list[int] = []
    for duration in durations:
        step = _steps_for_duration(duration, interval_minutes)
        if step not in values:
            values.append(step)
    return values


def _lag_name(step: int) -> str:
    return f"demand_lag_{step}"


def _window_name(metric: str, step: int) -> str:
    return f"demand_{metric}_{step}"


def _safe_std(values: np.ndarray) -> float:
    return float(np.std(values, ddof=0)) if len(values) else 0.0


def build_origin_features(
    history: pd.DataFrame,
    future_timestamps: pd.DatetimeIndex,
    interval_minutes: int = 15,
    *,
    feature_schema: str = LEGACY_FEATURE_SCHEMA,
) -> pd.DataFrame:
    """Return one feature row for each future timestamp at this forecast origin."""
    if feature_schema not in {FEATURE_SCHEMA, DAY_ONLY_FEATURE_SCHEMA, LEGACY_FEATURE_SCHEMA}:
        raise ValueError(f"Unsupported tabular feature schema: {feature_schema}")
    required = {"utc_timestamp", "demand", "temperature"}
    missing = required - set(history.columns)
    if missing:
        raise ValueError(f"History is missing columns: {sorted(missing)}")
    if history.empty:
        raise ValueError("History window is empty.")
    if interval_minutes <= 0:
        raise ValueError("interval_minutes must be positive.")

    # Lag features: current interval, then approximately 1h, 3h, 6h, 24h, 7d.
    lag_durations = [interval_minutes, 60, 180, 360, 1440, 10080]
    lag_steps = _unique_steps(lag_durations, interval_minutes)
    # Rolling features use 1h, 3h, 24h and 7d windows.
    demand_windows = _unique_steps([60, 180, 1440, 10080], interval_minutes)
    temperature_windows = _unique_steps([60, 1440], interval_minutes)
    daily_window = _steps_for_duration(1440, interval_minutes)

    history = history.tail(max(max(lag_steps), max(demand_windows), max(temperature_windows))).copy()
    demand = pd.to_numeric(history["demand"], errors="coerce").to_numpy(dtype=float)
    temperature = pd.to_numeric(history["temperature"], errors="coerce").to_numpy(dtype=float)
    timestamps = pd.DatetimeIndex(pd.to_datetime(history["utc_timestamp"], utc=True))
    if not np.isfinite(demand).all() or not np.isfinite(temperature).all():
        raise ValueError("History has missing or non-finite values.")
    if len(timestamps) > 1:
        expected = pd.Timedelta(minutes=interval_minutes).value
        if not np.all(np.diff(timestamps.asi8) == expected):
            raise ValueError("History window is not continuous at the configured interval.")

    origin_temp = float(temperature[-1])
    base: dict[str, float] = {}
    for lag in lag_steps:
        if len(demand) < lag:
            raise ValueError(f"At least {lag} history observations are required for lag features.")
        base[_lag_name(lag)] = float(demand[-lag])

    for window in demand_windows:
        if len(demand) < window:
            raise ValueError(f"At least {window} history observations are required for rolling features.")
        values = demand[-window:]
        base[_window_name("mean", window)] = float(np.mean(values))
        base[_window_name("std", window)] = _safe_std(values)
        # Preserve the prior 24h and 7d extrema features at 15-minute interval.
        if window in {_steps_for_duration(1440, interval_minutes), _steps_for_duration(10080, interval_minutes)}:
            base[_window_name("min", window)] = float(np.min(values))
            base[_window_name("max", window)] = float(np.max(values))

    one_hour_steps = _steps_for_duration(60, interval_minutes)
    daily_steps = _steps_for_duration(1440, interval_minutes)
    if one_hour_steps > 1:
        base[f"demand_change_1_to_{one_hour_steps}"] = float(demand[-1] - demand[-one_hour_steps])
    if daily_steps > 1:
        base[f"demand_change_1_to_{daily_steps}"] = float(demand[-1] - demand[-daily_steps])

    base["temperature_last"] = origin_temp
    for window in temperature_windows:
        if len(temperature) < window:
            raise ValueError(f"At least {window} history observations are required for temperature features.")
        base[f"temperature_mean_{window}"] = float(np.mean(temperature[-window:]))
    if len(temperature) < daily_window:
        raise ValueError(f"At least {daily_window} history observations are required for temperature volatility.")
    base[f"temperature_std_{daily_window}"] = _safe_std(temperature[-daily_window:])

    future_timestamps = pd.DatetimeIndex(pd.to_datetime(future_timestamps, utc=True))
    if len(future_timestamps) == 0:
        raise ValueError("No future timestamps supplied.")
    references = (
        target_demand_references(
            history, future_timestamps,
            include_previous_week=feature_schema == FEATURE_SCHEMA,
        )
        if feature_schema in {FEATURE_SCHEMA, DAY_ONLY_FEATURE_SCHEMA} else {}
    )

    # No future weather feed exists in this POC; use last-observed persistence
    # consistently in both training and serving.
    horizon = len(future_timestamps)
    future_temp = np.full(horizon, origin_temp, dtype=float)
    rolling_1h_window = _steps_for_duration(60, interval_minutes)
    rolling_24h_window = _steps_for_duration(1440, interval_minutes)
    context_needed = max(rolling_1h_window, rolling_24h_window) - 1
    temperature_context = np.concatenate(
        [temperature[-context_needed:] if context_needed else temperature[:0], future_temp]
    )
    roll_1h = pd.Series(temperature_context).rolling(
        rolling_1h_window, min_periods=1
    ).mean().to_numpy()[-horizon:]
    roll_24h = pd.Series(temperature_context).rolling(
        rolling_24h_window, min_periods=1
    ).mean().to_numpy()[-horizon:]

    rows: list[dict[str, float]] = []
    for index, target_ts in enumerate(future_timestamps, start=1):
        minute_of_day = target_ts.hour * 60 + target_ts.minute
        weekday = target_ts.dayofweek
        day_of_year = target_ts.dayofyear - 1
        hour_angle = 2.0 * np.pi * minute_of_day / 1440.0
        weekday_angle = 2.0 * np.pi * weekday / 7.0
        year_angle = 2.0 * np.pi * day_of_year / 365.2425
        row = dict(base)
        row.update({
            "horizon_step": float(index),
            "lead_hours": float(index * interval_minutes / 60.0),
            "future_temperature_persistence": origin_temp,
            "future_temperature_rolling_1h": float(roll_1h[index - 1]),
            "future_temperature_rolling_24h": float(roll_24h[index - 1]),
            "target_hour_sin": float(np.sin(hour_angle)),
            "target_hour_cos": float(np.cos(hour_angle)),
            "target_weekday_sin": float(np.sin(weekday_angle)),
            "target_weekday_cos": float(np.cos(weekday_angle)),
            "target_year_sin": float(np.sin(year_angle)),
            "target_year_cos": float(np.cos(year_angle)),
            "target_is_weekend": float(weekday >= 5),
        })
        for name, values in references.items():
            row[name] = float(values[index - 1])
        rows.append(row)
    return pd.DataFrame(rows)


def build_training_examples(
    frame: pd.DataFrame,
    origins: np.ndarray,
    history_steps: int,
    horizon_steps: int,
    interval_minutes: int,
    *,
    feature_schema: str = LEGACY_FEATURE_SCHEMA,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Build sampled long-format supervised examples for selected forecast origins."""
    feature_frames: list[pd.DataFrame] = []
    targets: list[np.ndarray] = []
    origin_ids: list[np.ndarray] = []
    timestamps = pd.DatetimeIndex(pd.to_datetime(frame["utc_timestamp"], utc=True))
    demand = frame["demand"].to_numpy(dtype=float)
    for origin in np.asarray(origins, dtype=int):
        history = frame.iloc[origin - history_steps:origin]
        future_ts = timestamps[origin:origin + horizon_steps]
        if len(future_ts) != horizon_steps:
            continue
        feature_frames.append(build_origin_features(
            history, future_ts, interval_minutes, feature_schema=feature_schema
        ))
        targets.append(demand[origin:origin + horizon_steps])
        origin_ids.append(np.full(horizon_steps, origin, dtype=np.int64))
    if not feature_frames:
        raise ValueError("No training windows could be generated.")
    return (
        pd.concat(feature_frames, ignore_index=True),
        np.concatenate(targets).astype(float),
        np.concatenate(origin_ids),
    )


def json_safe_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    for key, value in parameters.items():
        if isinstance(value, (np.integer,)):
            value = int(value)
        elif isinstance(value, (np.floating,)):
            value = float(value)
        clean[str(key)] = value
    return clean
