"""Utilities for keeping forecasting windows inside continuous time segments."""
from __future__ import annotations

import numpy as np
import pandas as pd


def continuous_segment_bounds(timestamps, interval_minutes: int) -> list[tuple[int, int]]:
    """Return half-open row ranges split wherever the expected cadence breaks."""
    interval_minutes = int(interval_minutes)
    if interval_minutes <= 0:
        raise ValueError("interval_minutes must be positive.")
    ts = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True))
    if ts.hasnans:
        raise ValueError("Segment timestamps must be valid.")
    if not ts.is_monotonic_increasing or not ts.is_unique:
        raise ValueError("Segment timestamps must be sorted and unique.")
    if ts.empty:
        return []
    expected = pd.Timedelta(minutes=interval_minutes)
    breaks = np.flatnonzero(ts[1:] - ts[:-1] != expected) + 1
    edges = np.concatenate(([0], breaks, [len(ts)])).astype(int)
    return [(int(start), int(end)) for start, end in zip(edges[:-1], edges[1:])]


def valid_forecast_origins(
    timestamps,
    history_steps: int,
    horizon_steps: int,
    interval_minutes: int,
    *,
    target_start: int = 0,
    target_end: int | None = None,
) -> np.ndarray:
    """Find forecast origins whose full history and target stay inside one segment.

    ``target_end`` is exclusive. It lets chronological splits keep all training
    targets before the split and all validation targets at or after it.
    """
    history_steps = int(history_steps)
    horizon_steps = int(horizon_steps)
    if history_steps <= 0 or horizon_steps <= 0:
        raise ValueError("history_steps and horizon_steps must be positive.")
    ts = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True))
    end_limit = len(ts) if target_end is None else int(target_end)
    start_limit = max(0, int(target_start))
    if end_limit < start_limit or end_limit > len(ts):
        raise ValueError("Forecast target bounds are outside the timestamp range.")

    origins: list[np.ndarray] = []
    for segment_start, segment_end in continuous_segment_bounds(ts, interval_minutes):
        first = max(segment_start + history_steps, start_limit)
        last = min(segment_end - horizon_steps, end_limit - horizon_steps)
        if first <= last:
            origins.append(np.arange(first, last + 1, dtype=np.int64))
    return np.concatenate(origins) if origins else np.asarray([], dtype=np.int64)


def split_forecast_origins(
    timestamps,
    history_steps: int,
    horizon_steps: int,
    interval_minutes: int,
    validation_fraction: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    """Chronologically split each segment into training and validation targets."""
    if not 0.05 <= float(validation_fraction) <= 0.40:
        raise ValueError("validation_fraction must be between 0.05 and 0.40.")
    ts = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True))
    training: list[np.ndarray] = []
    validation: list[np.ndarray] = []
    for start, end in continuous_segment_bounds(ts, interval_minutes):
        split = start + int((end - start) * (1.0 - float(validation_fraction)))
        first_origin = start + int(history_steps)
        last_origin = end - int(horizon_steps)
        train_last = min(last_origin, split - int(horizon_steps))
        valid_first = max(first_origin, split)
        if first_origin <= train_last:
            training.append(np.arange(first_origin, train_last + 1, dtype=np.int64))
        if valid_first <= last_origin:
            validation.append(np.arange(valid_first, last_origin + 1, dtype=np.int64))
    empty = np.asarray([], dtype=np.int64)
    return (
        np.concatenate(training) if training else empty.copy(),
        np.concatenate(validation) if validation else empty.copy(),
    )


def sample_origins_by_segment(
    origins,
    timestamps,
    interval_minutes: int,
    maximum: int,
) -> np.ndarray:
    """Sample forecast origins while representing every eligible segment."""
    maximum = int(maximum)
    if maximum <= 0:
        raise ValueError("maximum must be positive.")
    origins = np.unique(np.asarray(origins, dtype=np.int64))
    if len(origins) <= maximum:
        return origins
    ts = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True))
    groups = []
    for start, end in continuous_segment_bounds(ts, interval_minutes):
        group = origins[(origins >= start) & (origins < end)]
        if len(group):
            groups.append(group)
    if maximum < len(groups):
        selected_groups = np.linspace(0, len(groups) - 1, maximum, dtype=int)
        return np.asarray([groups[index][len(groups[index]) // 2] for index in selected_groups], dtype=np.int64)

    sizes = np.asarray([len(group) for group in groups], dtype=int)
    allocation = np.ones(len(groups), dtype=int)
    remaining = maximum - len(groups)
    while remaining > 0:
        capacity = sizes - allocation
        active = np.flatnonzero(capacity > 0)
        if not len(active):
            break
        weights = capacity[active].astype(float)
        raw = remaining * weights / weights.sum()
        additions = np.minimum(np.floor(raw).astype(int), capacity[active])
        if additions.sum():
            allocation[active] += additions
            remaining -= int(additions.sum())
            continue
        order = active[np.argsort(raw - np.floor(raw))[::-1]]
        for group_index in order:
            if remaining == 0:
                break
            if allocation[group_index] < sizes[group_index]:
                allocation[group_index] += 1
                remaining -= 1

    sampled = [
        group[np.linspace(0, len(group) - 1, count, dtype=int)]
        for group, count in zip(groups, allocation)
    ]
    return np.sort(np.concatenate(sampled))
