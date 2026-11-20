"""Chronological target boundaries for model evaluation."""

import numpy as np
import pandas as pd


TEST_FRACTION = 0.15


def chronological_split_bounds(rows: int, validation_fraction: float, with_test: bool) -> tuple[int, int]:
    """Return exclusive train and validation ends as offsets into historical rows."""
    test_fraction = TEST_FRACTION if with_test else 0.0
    train_end = int(rows * (1 - validation_fraction - test_fraction))
    validation_end = int(rows * (1 - test_fraction)) if with_test else rows
    return train_end, validation_end


def archived_weather_history(frame: pd.DataFrame, columns: list[str], history: int) -> pd.DataFrame:
    """Keep every archive period and enough prior actuals to seed its first origin."""
    if not set(columns).issubset(frame.columns):
        raise ValueError("This dataset is missing archived day-ahead weather columns.")
    valid = np.flatnonzero(np.isfinite(frame[columns].to_numpy(dtype=float)).all(axis=1))
    if not len(valid):
        raise ValueError("No archived day-ahead weather overlaps this dataset.")
    return frame.iloc[max(0, valid[0] - history):valid[-1] + 1].reset_index(drop=True)


def forecast_split_origins(
    frame: pd.DataFrame, history: int, horizon: int, interval: int,
    validation_fraction: float, with_test: bool, weather_columns: list[str],
) -> tuple[list[np.ndarray], int, int]:
    """Return chronological target windows with continuous history and real forecast weather."""
    size = len(frame)
    train_end, val_end = chronological_split_bounds(size, validation_fraction, with_test)
    bounds = [(history, train_end), (max(history, train_end), val_end)]
    if with_test:
        bounds.append((max(history, val_end), size))
    timestamps = pd.DatetimeIndex(pd.to_datetime(frame["utc_timestamp"], utc=True)).as_unit("ns")
    breaks = np.r_[0, np.cumsum(np.diff(timestamps.asi8) != pd.Timedelta(minutes=interval).value)]
    missing_weather = np.zeros(size + 1, dtype=np.int64)
    if weather_columns:
        valid = np.isfinite(frame[weather_columns].to_numpy(dtype=float)).all(axis=1)
        missing_weather[1:] = np.cumsum(~valid)
    groups = []
    for first, end in bounds:
        starts = np.arange(first, end - horizon + 1, dtype=np.int64)
        continuous = breaks[starts + horizon - 1] == breaks[starts - history]
        covered = missing_weather[starts + horizon] == missing_weather[starts]
        groups.append(starts[continuous & covered])
    for label, starts, minimum in zip(("training", "validation", "test"), groups, (2, 1, 1)):
        if len(starts) < minimum:
            weather = " with archived day-ahead weather for every forecast step" if weather_columns else ""
            raise ValueError(f"The {label} split has {len(starts)} usable forecast windows{weather}. "
                             f"Provide more continuous data for the {history}-step history and {horizon}-step horizon.")
    return groups, train_end, val_end
