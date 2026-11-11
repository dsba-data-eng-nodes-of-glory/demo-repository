from __future__ import annotations

import argparse
import copy
import gc
import shutil
import json
import math
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


# -----------------------------------------------------------------------------
# Amprion wind-generation Dense NN
# Feature construction follows the tested solar/demand reference-style DNN:
# one engineered feature vector per forecast origin -> 96-step output vector.
# -----------------------------------------------------------------------------

REGION_NAME = "amprion"
PACKAGE_DIR = Path(__file__).resolve().parents[4]
DATA_DIR = PACKAGE_DIR / "data" / "training" / "amprion"
DEFAULT_OUTPUT_DIR = PACKAGE_DIR / "models" / "amprion" / "wind_dense_nn"

TIMESTAMP_COLUMN = "utc_timestamp"
TARGET_COLUMN = "Wind onshore [MWh]"
WIND_SPEED_COLUMN = "wind_speed_100m_mean_ms"
WIND_DIRECTION_COLUMN = "wind_direction_100m_circular_mean_deg"
CAPACITY_COLUMN = "Wind_onshore_installed_capacity [MW]"

# Requested names are primary. Common long names are accepted as aliases.
FUTURE_SPEED_CANDIDATES = [
    "future_wind_speed",
    "future_wind_speed_100m_mean_ms",
    "forecast_wind_speed_100m_mean_ms",
]
FUTURE_DIRECTION_CANDIDATES = [
    "future_wind_direction",
    "future_wind_direction_100m_circular_mean_deg",
    "forecast_wind_direction_100m_circular_mean_deg",
]

HISTORY_STEPS = 672       # 7 days at 15-minute resolution
HORIZON_STEPS = 96        # next 24 hours at 15-minute resolution
INTERVAL_MINUTES = 15
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
WIND_LAGS = [1, 4, 8, 16, 24, 48, 96, 192, 288, 384, 480, 576, 672]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_timestamps(values: pd.Series) -> pd.Series:
    raw = values.astype("string").str.strip()
    parsed = pd.to_datetime(raw, format="%d-%m-%Y %H:%M", errors="coerce", utc=True)
    mask = parsed.isna() & raw.notna() & raw.ne("")
    if mask.any():
        # format="mixed" avoids pandas inferring one date format for the whole
        # column and then rejecting later ISO timestamps (especially across days).
        parsed.loc[mask] = pd.to_datetime(
            raw.loc[mask], format="mixed", dayfirst=True, errors="coerce", utc=True
        )
    return parsed


def numeric(series: pd.Series) -> pd.Series:
    raw = series.astype("string").str.strip().str.replace(",", "", regex=False)
    raw = raw.replace({"": pd.NA, "-": pd.NA, "nan": pd.NA, "None": pd.NA})
    return pd.to_numeric(raw, errors="coerce")


def read_csv_robust(path: str | Path) -> tuple[pd.DataFrame, str]:
    """Read UTF-8 and common Excel/Windows encodings (including CP1252)."""
    path = Path(path)
    last_error = None
    for encoding in ("utf-8-sig", "cp1252", "latin1"):
        try:
            frame = pd.read_csv(path, low_memory=False, encoding=encoding)
            frame.columns = [str(c).replace("\xa0", " ").strip() for c in frame.columns]
            for column in frame.select_dtypes(include=["object"]).columns:
                frame[column] = frame[column].map(
                    lambda value: value.replace("\xa0", " ") if isinstance(value, str) else value
                )
            print(f"CSV encoding: {encoding}", flush=True)
            return frame, encoding
        except UnicodeDecodeError as exc:
            last_error = exc
    raise UnicodeError(
        f"Unable to decode {path}. Tried utf-8-sig, cp1252 and latin1. Last error: {last_error}"
    )


def find_column(columns: Iterable[str], candidates: list[str], label: str) -> str:
    available = set(columns)
    for candidate in candidates:
        if candidate in available:
            return candidate
    raise ValueError(
        f"Could not find {label}. Expected one of {candidates}. Available columns: {sorted(available)}"
    )


def circular_components(direction_degrees: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    direction = np.asarray(direction_degrees, dtype=np.float64)
    radians = np.deg2rad(np.mod(direction, 360.0))
    sine = np.sin(radians).astype(np.float32)
    cosine = np.cos(radians).astype(np.float32)
    invalid = ~np.isfinite(direction)
    sine[invalid] = np.nan
    cosine[invalid] = np.nan
    return sine, cosine


def circular_rolling_features(direction: np.ndarray, windows: list[int]) -> dict[tuple[int, str], np.ndarray]:
    """Trailing circular means and resultant lengths; no future observations used."""
    sine, cosine = circular_components(direction)
    sine_series = pd.Series(sine.astype(np.float64))
    cosine_series = pd.Series(cosine.astype(np.float64))
    output: dict[tuple[int, str], np.ndarray] = {}
    for window in windows:
        roll_sin = sine_series.rolling(window=window, min_periods=1).mean().to_numpy()
        roll_cos = cosine_series.rolling(window=window, min_periods=1).mean().to_numpy()
        resultant = np.hypot(roll_sin, roll_cos)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_sin = roll_sin / resultant
            mean_cos = roll_cos / resultant
        cancellation = np.isfinite(resultant) & (resultant < 1e-8)
        mean_sin[cancellation] = 0.0
        mean_cos[cancellation] = 0.0
        output[(window, "sin")] = mean_sin.astype(np.float32)
        output[(window, "cos")] = mean_cos.astype(np.float32)
        output[(window, "resultant")] = resultant.astype(np.float32)
    return output


def rolling_stats(values: np.ndarray, windows: list[int], stats: list[str]) -> dict[tuple[int, str], np.ndarray]:
    series = pd.Series(np.asarray(values, dtype=np.float64))
    output: dict[tuple[int, str], np.ndarray] = {}
    for window in windows:
        roll = series.rolling(window=window, min_periods=1)
        for stat in stats:
            if stat == "mean":
                values_out = roll.mean()
            elif stat == "std":
                values_out = roll.std(ddof=0).fillna(0.0)
            elif stat == "max":
                values_out = roll.max()
            elif stat == "min":
                values_out = roll.min()
            else:
                raise ValueError(f"Unsupported rolling statistic: {stat}")
            output[(window, stat)] = values_out.to_numpy(dtype=np.float32)
    return output


def split_points(n_rows: int) -> tuple[int, int]:
    return int(n_rows * TRAIN_FRACTION), int(n_rows * (TRAIN_FRACTION + VALIDATION_FRACTION))


def load_wind_data(path: str | Path, interval_minutes: int) -> tuple[pd.DataFrame, dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Training CSV not found: {path.resolve()}")
    frame, _ = read_csv_robust(path)
    frame.columns = [str(c).replace("\xa0", " ").strip() for c in frame.columns]

    future_speed_col = find_column(frame.columns, FUTURE_SPEED_CANDIDATES, "future wind speed column")
    future_direction_col = find_column(frame.columns, FUTURE_DIRECTION_CANDIDATES, "future wind direction column")
    required = [
        TIMESTAMP_COLUMN, TARGET_COLUMN, WIND_SPEED_COLUMN, WIND_DIRECTION_COLUMN,
        CAPACITY_COLUMN, future_speed_col, future_direction_col,
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available: {list(frame.columns)}")

    source_rows = len(frame)
    frame = frame[required].copy()
    frame[TIMESTAMP_COLUMN] = parse_timestamps(frame[TIMESTAMP_COLUMN])
    invalid_timestamps = int(frame[TIMESTAMP_COLUMN].isna().sum())
    for col in required[1:]:
        frame[col] = numeric(frame[col])
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=[TIMESTAMP_COLUMN])
    if frame.empty:
        raise ValueError("No valid timestamps remain in the training CSV.")

    # Aggregate duplicate timestamps. Wind direction must be averaged circularly.
    valid_timestamp_rows = len(frame)
    obs_sin, obs_cos = circular_components(frame[WIND_DIRECTION_COLUMN].to_numpy())
    fut_sin, fut_cos = circular_components(frame[future_direction_col].to_numpy())
    frame["_obs_dir_sin"], frame["_obs_dir_cos"] = obs_sin, obs_cos
    frame["_future_dir_sin"], frame["_future_dir_cos"] = fut_sin, fut_cos
    grouped = frame.groupby(TIMESTAMP_COLUMN, as_index=False).agg(
        **{
            TARGET_COLUMN: (TARGET_COLUMN, "mean"),
            WIND_SPEED_COLUMN: (WIND_SPEED_COLUMN, "mean"),
            CAPACITY_COLUMN: (CAPACITY_COLUMN, "mean"),
            future_speed_col: (future_speed_col, "mean"),
            "_obs_dir_sin": ("_obs_dir_sin", "mean"),
            "_obs_dir_cos": ("_obs_dir_cos", "mean"),
            "_future_dir_sin": ("_future_dir_sin", "mean"),
            "_future_dir_cos": ("_future_dir_cos", "mean"),
        }
    )
    obs_norm = np.hypot(grouped["_obs_dir_sin"], grouped["_obs_dir_cos"])
    fut_norm = np.hypot(grouped["_future_dir_sin"], grouped["_future_dir_cos"])
    grouped[WIND_DIRECTION_COLUMN] = (
        np.rad2deg(np.arctan2(grouped["_obs_dir_sin"], grouped["_obs_dir_cos"])) % 360.0
    )
    grouped[future_direction_col] = (
        np.rad2deg(np.arctan2(grouped["_future_dir_sin"], grouped["_future_dir_cos"])) % 360.0
    )
    grouped.loc[obs_norm < 1e-8, WIND_DIRECTION_COLUMN] = np.nan
    grouped.loc[fut_norm < 1e-8, future_direction_col] = np.nan
    grouped = grouped.drop(columns=["_obs_dir_sin", "_obs_dir_cos", "_future_dir_sin", "_future_dir_cos"])

    if (grouped[WIND_SPEED_COLUMN].dropna() < 0).any():
        raise ValueError(f"Negative observed values in {WIND_SPEED_COLUMN}.")
    if (grouped[future_speed_col].dropna() < 0).any():
        raise ValueError(f"Negative forecast values in {future_speed_col}.")
    if (grouped[CAPACITY_COLUMN].dropna() <= 0).any():
        raise ValueError(f"Zero or negative values in {CAPACITY_COLUMN}.")

    timestamps = pd.DatetimeIndex(grouped[TIMESTAMP_COLUMN]).sort_values()
    interval = pd.Timedelta(minutes=interval_minutes)
    offsets = (timestamps.asi8 - timestamps.asi8[0]) % interval.value
    if np.count_nonzero(offsets):
        examples = timestamps[np.flatnonzero(offsets)[:5]].astype(str).tolist()
        raise ValueError(f"Timestamps are not aligned to {interval_minutes}-minute grid. Examples: {examples}")

    grouped = grouped.set_index(TIMESTAMP_COLUMN).sort_index()
    full_index = pd.date_range(grouped.index.min(), grouped.index.max(), freq=interval, tz="UTC")
    unique_timestamp_count = len(grouped.index)
    duplicate_rows_aggregated = valid_timestamp_rows - unique_timestamp_count
    grouped = grouped.reindex(full_index)
    grouped.index.name = TIMESTAMP_COLUMN
    aligned = grouped.reset_index()
    # Capacity is slowly changing; carry forward only, never backfill from future.
    capacity_before = int(aligned[CAPACITY_COLUMN].isna().sum())
    aligned[CAPACITY_COLUMN] = aligned[CAPACITY_COLUMN].ffill()
    capacity_after = int(aligned[CAPACITY_COLUMN].isna().sum())
    if capacity_after:
        raise ValueError(
            f"{CAPACITY_COLUMN} has leading missing values. Provide capacity at or before the first timestamp."
        )

    diagnostics = {
        "source_rows": int(source_rows),
        "invalid_timestamps_removed": int(invalid_timestamps),
        "duplicate_rows_aggregated": int(duplicate_rows_aggregated),
        "regular_grid_rows": int(len(aligned)),
        "missing_target": int(aligned[TARGET_COLUMN].isna().sum()),
        "missing_observed_wind_speed": int(aligned[WIND_SPEED_COLUMN].isna().sum()),
        "missing_observed_wind_direction": int(aligned[WIND_DIRECTION_COLUMN].isna().sum()),
        "missing_future_wind_speed": int(aligned[future_speed_col].isna().sum()),
        "missing_future_wind_direction": int(aligned[future_direction_col].isna().sum()),
        "capacity_forward_filled": int(capacity_before - capacity_after),
        "future_speed_column": future_speed_col,
        "future_direction_column": future_direction_col,
    }
    print(
        f"Rows={source_rows:,}; invalid timestamps={invalid_timestamps:,}; "
        f"duplicates aggregated={diagnostics['duplicate_rows_aggregated']:,}; "
        f"regular-grid rows={len(aligned):,}", flush=True
    )
    print(
        "Missing values: " + ", ".join(
            f"{k}={v:,}" for k, v in diagnostics.items() if k.startswith("missing_")
        ), flush=True
    )
    print(f"UTC range: {aligned[TIMESTAMP_COLUMN].min()} to {aligned[TIMESTAMP_COLUMN].max()}", flush=True)

    # Standardize the internal names so feature construction is independent of aliases.
    aligned = aligned.rename(columns={
        future_speed_col: "_future_wind_speed",
        future_direction_col: "_future_wind_direction",
    })
    return aligned, diagnostics


def calendar_matrix(timestamps: pd.DatetimeIndex) -> np.ndarray:
    ts = pd.DatetimeIndex(timestamps)
    minute = ts.hour.to_numpy() * 60 + ts.minute.to_numpy()
    weekday = ts.dayofweek.to_numpy()
    day = ts.dayofyear.to_numpy() - 1
    minute_angle = 2.0 * np.pi * minute / 1440.0
    weekday_angle = 2.0 * np.pi * weekday / 7.0
    year_angle = 2.0 * np.pi * day / 365.2425
    return np.column_stack([
        np.sin(minute_angle), np.cos(minute_angle),
        np.sin(weekday_angle), np.cos(weekday_angle),
        np.sin(year_angle), np.cos(year_angle),
    ]).astype(np.float32)


def build_feature_matrix(
    frame: pd.DataFrame,
    starts: np.ndarray,
    interval_minutes: int = INTERVAL_MINUTES,
    chunk_size: int = 4000,
    future_capacity_mode: str = "known",
    cache_prefix: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, dict]:
    """Build solar-style features plus missing LSTM-aligned 15-minute sequences.

    Historical sequence ends at the issue timestamp. Future wind sequence uses
    only the forecast weather columns supplied for the next 96 valid timestamps.
    """
    if interval_minutes != 15:
        raise ValueError("This feature alignment is currently designed for 15-minute data.")
    if future_capacity_mode not in {"known", "persist"}:
        raise ValueError("future_capacity_mode must be 'known' or 'persist'.")
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1.")

    timestamps = pd.DatetimeIndex(frame[TIMESTAMP_COLUMN])
    target = frame[TARGET_COLUMN].to_numpy(dtype=np.float32)
    obs_speed = frame[WIND_SPEED_COLUMN].to_numpy(dtype=np.float32)
    obs_direction = frame[WIND_DIRECTION_COLUMN].to_numpy(dtype=np.float32)
    future_speed = frame["_future_wind_speed"].to_numpy(dtype=np.float32)
    future_direction = frame["_future_wind_direction"].to_numpy(dtype=np.float32)
    capacity = frame[CAPACITY_COLUMN].to_numpy(dtype=np.float32)

    n = len(frame)
    starts = np.asarray(starts, dtype=np.int64)
    horizon_offsets = np.arange(HORIZON_STEPS, dtype=np.int64)
    hourly_offsets = np.arange(0, HORIZON_STEPS, 4, dtype=np.int64)

    target_roll4 = pd.Series(target).rolling(4, min_periods=1).mean().to_numpy(dtype=np.float32)
    target_stats = rolling_stats(target, [4, 24, 96, 672], ["mean", "std", "max"])
    speed_stats = rolling_stats(obs_speed, [4, 96, 672], ["mean", "std"])
    direction_stats = circular_rolling_features(obs_direction, [4, 96, 672])
    capacity_stats = rolling_stats(capacity, [96, 672], ["mean"])
    calendar = calendar_matrix(timestamps)
    obs_dir_sin, obs_dir_cos = circular_components(obs_direction)
    fut_dir_sin, fut_dir_cos = circular_components(future_direction)

    names: list[str] = []
    # Target history: previous day/week profiles, exact lags, rolling statistics.
    names += [f"generation_previous_24h_hour_{i + 1:02d}" for i in range(24)]
    names += [f"generation_previous_week_same_24h_hour_{i + 1:02d}" for i in range(24)]
    names += [f"generation_lag_{lag}_steps" for lag in WIND_LAGS]
    for label in ["1h", "6h", "24h", "7d"]:
        names += [f"generation_{stat}_{label}" for stat in ["mean", "std", "max"]]

    # Historical observed wind profile and trailing summaries.
    names += [f"hist_wind_speed_hour_{i + 1:02d}" for i in range(24)]
    names += [f"hist_wind_direction_sin_hour_{i + 1:02d}" for i in range(24)]
    names += [f"hist_wind_direction_cos_hour_{i + 1:02d}" for i in range(24)]
    for label in ["1h", "24h", "7d"]:
        names += [f"hist_wind_speed_{stat}_{label}" for stat in ["mean", "std"]]
    for label in ["1h", "24h", "7d"]:
        names += [
            f"hist_wind_direction_rolling_{component}_{label}"
            for component in ["sin", "cos", "resultant"]
        ]

    # Forecast wind trajectory: hourly profiles + full 24-hour summaries.
    names += [f"future_wind_speed_hour_{i + 1:02d}" for i in range(24)]
    names += [f"future_wind_direction_sin_hour_{i + 1:02d}" for i in range(24)]
    names += [f"future_wind_direction_cos_hour_{i + 1:02d}" for i in range(24)]
    names += [f"future_wind_speed_{stat}_24h" for stat in ["mean", "std", "min", "max"]]
    names += ["future_wind_direction_mean_sin_24h", "future_wind_direction_mean_cos_24h", "future_wind_direction_resultant_24h"]

    # Installed capacity: current, historical, and future scheduled/persisted values.
    names += [
        "capacity_at_origin_mw", "capacity_mean_24h_mw", "capacity_mean_7d_mw",
        "capacity_change_24h_mw", "capacity_change_7d_mw",
    ]
    names += [f"future_capacity_hour_{i + 1:02d}_mw" for i in range(24)]
    names += [f"future_capacity_{stat}_24h_mw" for stat in ["mean", "std", "min", "max"]]

    # Origin calendar and future hourly time-of-day profile.
    names += [
        "origin_hour_sin", "origin_hour_cos", "origin_weekday_sin", "origin_weekday_cos",
        "origin_year_sin", "origin_year_cos",
    ]
    names += [f"future_hour_sin_{i + 1:02d}" for i in range(24)]
    names += [f"future_hour_cos_{i + 1:02d}" for i in range(24)]

    # Missing LSTM-aligned features added at native 15-minute resolution.
    # Historical weather sequence: last 24 hours available at the origin.
    names += [f"hist_wind_speed_15m_step_{i + 1:03d}" for i in range(96)]
    names += [f"hist_wind_direction_sin_15m_step_{i + 1:03d}" for i in range(96)]
    names += [f"hist_wind_direction_cos_15m_step_{i + 1:03d}" for i in range(96)]
    for label in ["1h", "24h"]:
        names += [f"hist_wind_speed_rolling_mean_{label}_15m_step_{i + 1:03d}" for i in range(96)]
    for label in ["1h", "24h"]:
        for component in ["sin", "cos"]:
            names += [f"hist_wind_direction_rolling_{label}_{component}_15m_step_{i + 1:03d}" for i in range(96)]
    names += [f"hist_capacity_15m_step_{i + 1:03d}_mw" for i in range(96)]
    for feature in ["hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "year_sin", "year_cos"]:
        names += [f"hist_{feature}_15m_step_{i + 1:03d}" for i in range(96)]

    # Future decoder-equivalent sequence: all 96 forecast timestamps.
    names += [f"future_wind_speed_15m_step_{i + 1:03d}" for i in range(96)]
    names += [f"future_wind_direction_sin_15m_step_{i + 1:03d}" for i in range(96)]
    names += [f"future_wind_direction_cos_15m_step_{i + 1:03d}" for i in range(96)]
    for label in ["1h", "24h"]:
        names += [f"future_wind_speed_rolling_mean_{label}_15m_step_{i + 1:03d}" for i in range(96)]
    for label in ["1h", "24h"]:
        for component in ["sin", "cos"]:
            names += [f"future_wind_direction_rolling_{label}_{component}_15m_step_{i + 1:03d}" for i in range(96)]
    names += [f"future_capacity_15m_step_{i + 1:03d}_mw" for i in range(96)]
    for feature in ["hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "year_sin", "year_cos"]:
        names += [f"future_{feature}_15m_step_{i + 1:03d}" for i in range(96)]

    # Large feature matrices are written incrementally to disk when cache_prefix
    # is supplied. This avoids retaining several GB of overlapping windows in RAM.
    if cache_prefix is not None:
        cache_prefix = Path(cache_prefix)
        cache_prefix.parent.mkdir(parents=True, exist_ok=True)
        X_store = np.lib.format.open_memmap(
            str(cache_prefix) + "_X.npy", mode="w+", dtype=np.float32,
            shape=(len(starts), len(names)),
        )
        Y_store = np.lib.format.open_memmap(
            str(cache_prefix) + "_Y.npy", mode="w+", dtype=np.float32,
            shape=(len(starts), HORIZON_STEPS),
        )
        S_store = np.lib.format.open_memmap(
            str(cache_prefix) + "_starts.npy", mode="w+", dtype=np.int64,
            shape=(len(starts),),
        )
        write_pos = 0
        X_parts: list[np.ndarray] | None = None
        Y_parts: list[np.ndarray] | None = None
        S_parts: list[np.ndarray] | None = None
    else:
        X_store = Y_store = S_store = None
        write_pos = 0
        X_parts = []
        Y_parts = []
        S_parts = []
    skipped_features = 0
    skipped_targets = 0

    for pos in range(0, len(starts), chunk_size):
        s = starts[pos:pos + chunk_size]
        cols: list[np.ndarray] = []

        # Previous 24h and previous week's corresponding 24h, hourly samples
        # from trailing 1-hour generation means (solar reference indexing).
        day_idx = s[:, None] - 93 + hourly_offsets[None, :]
        week_idx = s[:, None] - 669 + hourly_offsets[None, :]
        cols.append(target_roll4[day_idx])
        cols.append(target_roll4[week_idx])

        cols.append(np.column_stack([target[s - lag] for lag in WIND_LAGS]).astype(np.float32))
        for window in [4, 24, 96, 672]:
            cols.append(np.column_stack([
                target_stats[(window, stat)][s - 1] for stat in ["mean", "std", "max"]
            ]))

        # Historical observed wind: previous 24h hourly profile.
        hist_idx = s[:, None] - 96 + hourly_offsets[None, :]
        cols.append(obs_speed[hist_idx])
        cols.append(obs_dir_sin[hist_idx])
        cols.append(obs_dir_cos[hist_idx])
        for window in [4, 96, 672]:
            cols.append(np.column_stack([
                speed_stats[(window, stat)][s - 1] for stat in ["mean", "std"]
            ]))
        for window in [4, 96, 672]:
            cols.append(np.column_stack([
                direction_stats[(window, component)][s - 1]
                for component in ["sin", "cos", "resultant"]
            ]))

        # LSTM-aligned historical weather context at native 15-minute resolution.
        # All values end at s-1; no future observations enter the encoder context.
        hist_15m_idx = s[:, None] - 96 + np.arange(96, dtype=np.int64)[None, :]
        cols.append(obs_speed[hist_15m_idx])
        cols.append(obs_dir_sin[hist_15m_idx])
        cols.append(obs_dir_cos[hist_15m_idx])
        for window in [4, 96]:
            cols.append(speed_stats[(window, "mean")][hist_15m_idx])
        for window in [4, 96]:
            cols.append(direction_stats[(window, "sin")][hist_15m_idx])
            cols.append(direction_stats[(window, "cos")][hist_15m_idx])
        cols.append(capacity[hist_15m_idx])
        cols.append(calendar[hist_15m_idx].transpose(0, 2, 1).reshape(len(s), -1))

        # Future forecast wind: never substitute observed future wind here.
        future_idx = s[:, None] + horizon_offsets[None, :]
        future_hour_idx = s[:, None] + hourly_offsets[None, :]
        future_speed_all = future_speed[future_idx]
        future_direction_all = future_direction[future_idx]
        future_speed_profile = future_speed[future_hour_idx]
        future_sin_profile = fut_dir_sin[future_hour_idx]
        future_cos_profile = fut_dir_cos[future_hour_idx]
        cols.append(future_speed_profile)
        cols.append(future_sin_profile)
        cols.append(future_cos_profile)
        cols.append(np.column_stack([
            np.mean(future_speed_all, axis=1),
            np.std(future_speed_all, axis=1),
            np.min(future_speed_all, axis=1),
            np.max(future_speed_all, axis=1),
        ]).astype(np.float32))
        future_sin_mean = np.mean(fut_dir_sin[future_idx], axis=1)
        future_cos_mean = np.mean(fut_dir_cos[future_idx], axis=1)
        future_resultant = np.hypot(future_sin_mean, future_cos_mean)
        cols.append(np.column_stack([future_sin_mean, future_cos_mean, future_resultant]).astype(np.float32))

        # Preserve the complete 96-step forecast-weather trajectory, matching the
        # LSTM decoder inputs while continuing to use forecast (not observed) weather.
        cols.append(future_speed_all)
        cols.append(fut_dir_sin[future_idx])
        cols.append(fut_dir_cos[future_idx])

        # Build causal future rolling weather features from the last 95 observed
        # weather rows + this origin's own 96-step forecast trajectory.
        # This mirrors the LSTM forecast-mode construction without borrowing
        # forecasts from another issue time.
        hist_context_idx = s[:, None] - 95 + np.arange(95, dtype=np.int64)[None, :]
        combined_speed = np.concatenate([obs_speed[hist_context_idx], future_speed_all], axis=1)
        combined_dir_sin = np.concatenate([obs_dir_sin[hist_context_idx], fut_dir_sin[future_idx]], axis=1)
        combined_dir_cos = np.concatenate([obs_dir_cos[hist_context_idx], fut_dir_cos[future_idx]], axis=1)

        def trailing_mean_at_future(values: np.ndarray, window: int) -> np.ndarray:
            prefix = np.concatenate(
                [np.zeros((values.shape[0], 1), dtype=np.float64),
                 np.cumsum(values.astype(np.float64), axis=1)],
                axis=1,
            )
            end_positions = 95 + np.arange(HORIZON_STEPS, dtype=np.int64)
            start_positions = end_positions - window + 1
            sums = prefix[:, end_positions + 1] - prefix[:, start_positions]
            return (sums / float(window)).astype(np.float32)

        for window in [4, 96]:
            cols.append(trailing_mean_at_future(combined_speed, window))
        for window in [4, 96]:
            roll_sin = trailing_mean_at_future(combined_dir_sin, window)
            roll_cos = trailing_mean_at_future(combined_dir_cos, window)
            norm = np.hypot(roll_sin, roll_cos)
            with np.errstate(invalid="ignore", divide="ignore"):
                mean_sin = roll_sin / norm
                mean_cos = roll_cos / norm
            cancelled = np.isfinite(norm) & (norm < 1e-8)
            mean_sin[cancelled] = 0.0
            mean_cos[cancelled] = 0.0
            cols.append(mean_sin.astype(np.float32))
            cols.append(mean_cos.astype(np.float32))

        # Capacity is assumed to be a known schedule by default. Use --future-capacity-mode persist
        # if no future capacity plan is available at forecast issue time.
        origin_capacity = capacity[s - 1]
        if future_capacity_mode == "known":
            future_capacity_all = capacity[future_idx]
        else:
            future_capacity_all = np.repeat(origin_capacity[:, None], HORIZON_STEPS, axis=1)
        cols.append(np.column_stack([
            origin_capacity,
            capacity_stats[(96, "mean")][s - 1],
            capacity_stats[(672, "mean")][s - 1],
            origin_capacity - capacity[np.maximum(s - 1 - 96, 0)],
            origin_capacity - capacity[np.maximum(s - 1 - 672, 0)],
        ]).astype(np.float32))
        cols.append(future_capacity_all[:, hourly_offsets])
        cols.append(np.column_stack([
            np.mean(future_capacity_all, axis=1),
            np.std(future_capacity_all, axis=1),
            np.min(future_capacity_all, axis=1),
            np.max(future_capacity_all, axis=1),
        ]).astype(np.float32))

        # Calendar features.
        cols.append(calendar[s - 1])
        future_cal = calendar[future_hour_idx]
        cols.append(future_cal[:, :, 0])
        cols.append(future_cal[:, :, 1])

        # Full 96-step capacity and calendar trajectories, aligned to the
        # decoder timestamps just like the LSTM future feature sequence.
        cols.append(future_capacity_all)
        cols.append(calendar[future_idx].transpose(0, 2, 1).reshape(len(s), -1))

        X_chunk = np.column_stack(cols).astype(np.float32, copy=False)
        Y_chunk = target[future_idx].astype(np.float32, copy=False)
        finite_x = np.isfinite(X_chunk).all(axis=1)
        finite_y = np.isfinite(Y_chunk).all(axis=1)
        skipped_features += int((~finite_x).sum())
        skipped_targets += int((~finite_y).sum())
        valid = finite_x & finite_y
        if np.any(valid):
            if X_store is not None:
                n_valid = int(valid.sum())
                X_store[write_pos:write_pos + n_valid] = X_chunk[valid]
                Y_store[write_pos:write_pos + n_valid] = Y_chunk[valid]
                S_store[write_pos:write_pos + n_valid] = s[valid]
                write_pos += n_valid
            else:
                X_parts.append(X_chunk[valid])
                Y_parts.append(Y_chunk[valid])
                S_parts.append(s[valid])

    if X_store is not None:
        if write_pos == 0:
            raise ValueError(
                "No valid forecast-origin windows were built. Check missing target, weather and forecast values."
            )
        X_store.flush(); Y_store.flush(); S_store.flush()
        X = X_store[:write_pos]
        Y = Y_store[:write_pos]
        valid_starts = S_store[:write_pos]
    else:
        if not X_parts:
            raise ValueError(
                "No valid forecast-origin windows were built. Check missing target, weather and forecast values."
            )
        X = np.concatenate(X_parts, axis=0)
        Y = np.concatenate(Y_parts, axis=0)
        valid_starts = np.concatenate(S_parts, axis=0)
    if X.shape[1] != len(names):
        raise RuntimeError(f"Feature name mismatch: matrix has {X.shape[1]}, names have {len(names)}")
    diagnostics = {
        "candidate_origins": int(len(starts)),
        "valid_origins": int(len(X)),
        "skipped_invalid_features": int(skipped_features),
        "skipped_invalid_targets": int(skipped_targets),
    }
    return X, Y, names, valid_starts, diagnostics


def make_origin_splits(n_rows: int, history: int, horizon: int):
    train_end, validation_end = split_points(n_rows)
    train_starts = np.arange(history, train_end - horizon + 1, dtype=np.int64)
    validation_starts = np.arange(max(history, train_end), validation_end - horizon + 1, dtype=np.int64)
    test_starts = np.arange(max(history, validation_end), n_rows - horizon + 1, dtype=np.int64)
    return train_starts, validation_starts, test_starts


class OriginDataset(Dataset):
    def __init__(self, x: np.ndarray, y_scaled: np.ndarray):
        self.x = torch.from_numpy(np.asarray(x, dtype=np.float32))
        self.y = torch.from_numpy(np.asarray(y_scaled, dtype=np.float32))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


def fit_x_scaler(x_train: np.ndarray, block_rows: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """Fit a float32 scaler using bounded-memory, block-wise accumulation."""
    n_rows, n_features = x_train.shape
    sums = np.zeros(n_features, dtype=np.float64)
    sums_sq = np.zeros(n_features, dtype=np.float64)
    for offset in range(0, n_rows, block_rows):
        block = np.asarray(x_train[offset:offset + block_rows], dtype=np.float64)
        sums += block.sum(axis=0)
        sums_sq += np.square(block).sum(axis=0)
    mean64 = sums / max(n_rows, 1)
    variance = np.maximum(sums_sq / max(n_rows, 1) - np.square(mean64), 0.0)
    std64 = np.sqrt(variance)
    mean = mean64.astype(np.float32)
    std = std64.astype(np.float32)
    std[~np.isfinite(std) | (std < 1e-6)] = 1.0
    return mean, std


def scale_memmap_inplace(values: np.ndarray, mean, std=None, block_rows: int = 2048) -> None:
    """Scale a memmap/array in place without allocating a full-size copy."""
    for offset in range(0, len(values), block_rows):
        block = values[offset:offset + block_rows]
        if std is None:
            block -= mean
        else:
            block -= mean
            block /= std
    if hasattr(values, "flush"):
        values.flush()


def fit_target_scaler(y_train: np.ndarray, block_rows: int = 2048) -> tuple[float, float]:
    total = 0.0
    total_sq = 0.0
    count = 0
    for offset in range(0, len(y_train), block_rows):
        block = np.asarray(y_train[offset:offset + block_rows], dtype=np.float64)
        total += float(block.sum())
        total_sq += float(np.square(block).sum())
        count += block.size
    mean = total / max(count, 1)
    variance = max(total_sq / max(count, 1) - mean * mean, 0.0)
    std = math.sqrt(variance)
    if not np.isfinite(std) or std < 1e-8:
        std = 1.0
    return float(mean), float(std)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    squared = 0.0
    absolute = 0.0
    count = 0
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            prediction = model(x)
            error = prediction - y
            squared += float(torch.sum(error ** 2))
            absolute += float(torch.sum(torch.abs(error)))
            count += y.numel()
    if count == 0:
        return float("nan"), float("nan")
    return math.sqrt(squared / count), absolute / count


def evaluate_original_units(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_mean: float,
    target_std: float,
) -> tuple[dict, list[dict]]:
    model.eval()
    errors: list[np.ndarray] = []
    actuals: list[np.ndarray] = []
    with torch.inference_mode():
        for x, y_scaled in loader:
            prediction_scaled = model(x.to(device)).cpu().numpy()
            actual_scaled = y_scaled.numpy()
            prediction = prediction_scaled * target_std + target_mean
            actual = actual_scaled * target_std + target_mean
            errors.append((prediction - actual).astype(np.float64))
            actuals.append(actual.astype(np.float64))
    if not errors:
        raise ValueError("Test split contains no valid origins.")
    error = np.concatenate(errors, axis=0)
    actual = np.concatenate(actuals, axis=0)
    absolute = np.abs(error)
    denominator = np.sum(np.abs(actual))
    metrics = {
        "MAE_MWh": float(np.mean(absolute)),
        "RMSE_MWh": float(np.sqrt(np.mean(error ** 2))),
        "MSE_MWh2": float(np.mean(error ** 2)),
        "WAPE_pct": float(np.sum(absolute) / denominator * 100.0) if denominator > 0 else float("nan"),
        "Bias_MWh": float(np.mean(error)),
        "Max_absolute_error_MWh": float(np.max(absolute)),
        "P95_absolute_error_MWh": float(np.percentile(absolute, 95)),
    }
    horizon_rows = []
    for step in range(error.shape[1]):
        step_error = error[:, step]
        step_actual = actual[:, step]
        step_denominator = np.sum(np.abs(step_actual))
        horizon_rows.append({
            "horizon_step": step + 1,
            "lead_minutes": (step + 1) * INTERVAL_MINUTES,
            "lead_hours": (step + 1) * INTERVAL_MINUTES / 60.0,
            "MAE_MWh": float(np.mean(np.abs(step_error))),
            "RMSE_MWh": float(np.sqrt(np.mean(step_error ** 2))),
            "WAPE_pct": float(np.sum(np.abs(step_error)) / step_denominator * 100.0) if step_denominator > 0 else float("nan"),
            "Bias_MWh": float(np.mean(step_error)),
        })
    return metrics, horizon_rows


class DenseMultiOutput(nn.Module):
    """Same 256 -> 128 -> 64 -> 96 Dense NN architecture as demand DNN."""
    def __init__(self, input_size: int, horizon: int, hidden_1: int, hidden_2: int, hidden_3: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_2, hidden_3),
            nn.ReLU(),
            nn.Linear(hidden_3, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}", flush=True)
    print(f"Region: {REGION_NAME}", flush=True)
    print("Model: Dense NN | solar/demand features + missing LSTM-aligned 15-minute wind sequences", flush=True)
    print("Target: next 96 x 15-minute Wind onshore generation values", flush=True)

    frame, data_diagnostics = load_wind_data(args.input_file, args.interval_minutes)
    n_rows = len(frame)
    train_starts, validation_starts, test_starts = make_origin_splits(n_rows, args.history, args.horizon)
    if len(train_starts) == 0 or len(validation_starts) == 0 or len(test_starts) == 0:
        raise ValueError("Not enough rows for 70/15/15 chronological splits with the chosen history/horizon.")

    feature_args = {
        "interval_minutes": args.interval_minutes,
        "chunk_size": args.chunk_size,
        "future_capacity_mode": args.future_capacity_mode,
    }
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_cache_dir = output_dir / "feature_cache"
    feature_cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Disk-backed feature cache: {feature_cache_dir}", flush=True)

    X_train, Y_train, feature_names, starts_train_valid, diag_train = build_feature_matrix(
        frame, train_starts, **feature_args, cache_prefix=feature_cache_dir / "train"
    )
    print(f"Train features ready: {X_train.shape}", flush=True)
    X_val, Y_val, names_val, starts_val_valid, diag_val = build_feature_matrix(
        frame, validation_starts, **feature_args, cache_prefix=feature_cache_dir / "validation"
    )
    print(f"Validation features ready: {X_val.shape}", flush=True)
    X_test, Y_test, names_test, starts_test_valid, diag_test = build_feature_matrix(
        frame, test_starts, **feature_args, cache_prefix=feature_cache_dir / "test"
    )
    print(f"Test features ready: {X_test.shape}", flush=True)
    if names_val != feature_names or names_test != feature_names:
        raise RuntimeError("Feature order mismatch between train, validation and test.")

    # Fit feature and target normalization on training rows only. All arrays are
    # disk-backed and transformed in-place in small blocks to avoid RAM spikes.
    x_mean, x_std = fit_x_scaler(X_train, block_rows=args.scale_block_rows)
    target_mean, target_std = fit_target_scaler(Y_train, block_rows=args.scale_block_rows)
    for split_name, X_split, Y_split in [
        ("train", X_train, Y_train), ("validation", X_val, Y_val), ("test", X_test, Y_test)
    ]:
        print(f"Standardising {split_name} split in blocks...", flush=True)
        scale_memmap_inplace(X_split, x_mean, x_std, block_rows=args.scale_block_rows)
        scale_memmap_inplace(Y_split, target_mean, target_std, block_rows=args.scale_block_rows)
    Y_train_scaled = Y_train
    Y_val_scaled = Y_val
    Y_test_scaled = Y_test

    print(
        f"Origins: train={len(X_train):,}, validation={len(X_val):,}, test={len(X_test):,}",
        flush=True,
    )
    print(f"Feature count: {X_train.shape[1]}", flush=True)
    print("Added LSTM-aligned features: 96-step historical weather context; 96-step forecast weather; "
          "future rolling wind speed/direction (1h, 24h); 96-step capacity and calendar.", flush=True)
    print(
        "Feature groups: generation profiles/lags/rolling stats; historical observed wind; "
        "future forecast wind; installed capacity; calendar",
        flush=True,
    )
    print(f"Future capacity mode: {args.future_capacity_mode}", flush=True)
    print(
        f"Skipped origins (train/val/test): "
        f"{diag_train['skipped_invalid_features'] + diag_train['skipped_invalid_targets']:,}/"
        f"{diag_val['skipped_invalid_features'] + diag_val['skipped_invalid_targets']:,}/"
        f"{diag_test['skipped_invalid_features'] + diag_test['skipped_invalid_targets']:,}",
        flush=True,
    )

    train_ds = OriginDataset(X_train, Y_train_scaled)
    val_ds = OriginDataset(X_val, Y_val_scaled)
    test_ds = OriginDataset(X_test, Y_test_scaled)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    train_eval_loader = DataLoader(train_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=0)

    model = DenseMultiOutput(
        input_size=X_train.shape[1], horizon=args.horizon,
        hidden_1=args.hidden_1, hidden_2=args.hidden_2,
        hidden_3=args.hidden_3, dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-5
    )

    best_val = float("inf")
    best_state = None
    best_train_eval = None
    best_epoch = 0
    bad_epochs = 0
    history_log: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            loss = nn.functional.mse_loss(prediction, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        train_rmse, train_mae = evaluate(model, train_eval_loader, device)
        val_rmse, val_mae = evaluate(model, val_loader, device)
        scheduler.step(val_rmse)
        current_lr = optimizer.param_groups[0]["lr"]
        history_log.append({
            "epoch": epoch,
            "train_rmse_scaled": train_rmse,
            "train_mae_scaled": train_mae,
            "validation_rmse_scaled": val_rmse,
            "validation_mae_scaled": val_mae,
            "learning_rate": current_lr,
        })
        print(
            f"Epoch {epoch:02d}/{args.epochs} | train-eval RMSE scaled={train_rmse:.5f} | "
            f"validation RMSE scaled={val_rmse:.5f} | validation MAE scaled={val_mae:.5f} | "
            f"lr={current_lr:.2e}", flush=True,
        )

        if np.isfinite(val_rmse) and val_rmse < best_val:
            best_val = val_rmse
            best_train_eval = train_rmse
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print("Early stopping: validation RMSE did not improve.", flush=True)
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")
    model.load_state_dict(best_state)

    test_rmse_scaled, test_mae_scaled = evaluate(model, test_loader, device)
    test_metrics, horizon_metrics = evaluate_original_units(
        model, test_loader, device, target_mean, target_std
    )
    print(f"Best epoch: {best_epoch} | best validation RMSE scaled={best_val:.5f}", flush=True)
    print(f"Test RMSE scaled={test_rmse_scaled:.5f}", flush=True)
    print(f"Test MAE scaled={test_mae_scaled:.5f}", flush=True)
    print(f"Test RMSE in generation units (MWh)={test_metrics['RMSE_MWh']:.3f}", flush=True)
    print(f"Test MAE in generation units (MWh)={test_metrics['MAE_MWh']:.3f}", flush=True)
    print(f"Test WAPE={test_metrics['WAPE_pct']:.3f}% | Bias={test_metrics['Bias_MWh']:.3f} MWh", flush=True)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "amprion_wind_dense_solar_aligned.pt"
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_kind": "dense_multioutput_wind_solar_reference_aligned",
        "region_name": REGION_NAME,
        "target_column": TARGET_COLUMN,
        "target_unit": "MWh per 15-minute interval",
        "history": args.history,
        "horizon": args.horizon,
        "interval_minutes": args.interval_minutes,
        "feature_names": feature_names,
        "feature_mean": x_mean,
        "feature_std": x_std,
        "target_mean": target_mean,
        "target_std": target_std,
        "architecture": {
            "input_size": int(X_train.shape[1]),
            "hidden_1": args.hidden_1,
            "hidden_2": args.hidden_2,
            "hidden_3": args.hidden_3,
            "dropout": args.dropout,
        },
        "optimizer": "AdamW",
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "future_speed_column": data_diagnostics["future_speed_column"],
        "future_direction_column": data_diagnostics["future_direction_column"],
        "future_capacity_mode": args.future_capacity_mode,
        "wind_direction_encoding": "circular_sin_cos",
        "best_epoch": best_epoch,
        "best_validation_rmse_scaled": best_val,
        "test_metrics": test_metrics,
    }
    torch.save(checkpoint, model_path)

    metadata = {
        "input_file": str(Path(args.input_file).resolve()),
        "model_path": str(model_path),
        "region": REGION_NAME,
        "history_steps": args.history,
        "horizon_steps": args.horizon,
        "interval_minutes": args.interval_minutes,
        "split": {"train": TRAIN_FRACTION, "validation": VALIDATION_FRACTION, "test": 0.15},
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "feature_groups": {
            "generation_profiles": 48,
            "generation_lags": WIND_LAGS,
            "generation_rolling_windows_steps": [4, 24, 96, 672],
            "historical_observed_wind_speed_profile_steps": 24,
            "historical_observed_wind_direction_profiles_sin_cos_steps": 48,
            "historical_wind_speed_rolling_windows_steps": [4, 96, 672],
            "historical_wind_direction_rolling_windows_steps": [4, 96, 672],
            "future_forecast_wind_speed_profile_steps": 24,
            "future_forecast_wind_direction_profiles_sin_cos_steps": 48,
            "future_forecast_wind_speed_statistics": ["mean", "std", "min", "max"],
            "future_forecast_wind_direction_statistics": ["mean_sin", "mean_cos", "resultant"],
            "added_lstm_aligned_history_15m_steps": 96,
            "added_lstm_aligned_future_15m_steps": 96,
            "added_future_causal_rolling_weather_windows_steps": [4, 96],
            "added_full_resolution_capacity_and_calendar_trajectories": True,
            "capacity_future_mode": args.future_capacity_mode,
            "calendar_origin_and_future_hourly_profiles": True,
        },
        "target_scaling": {"mean": target_mean, "std": target_std},
        "architecture": checkpoint["architecture"],
        "optimizer": checkpoint["optimizer"],
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "epochs_requested": args.epochs,
        "best_epoch": best_epoch,
        "best_validation_rmse_scaled": best_val,
        "train_rmse_at_best_validation_scaled": best_train_eval,
        "test_rmse_scaled": test_rmse_scaled,
        "test_mae_scaled": test_mae_scaled,
        "test_metrics_original_units": test_metrics,
        "training_history": history_log,
        "origin_counts": {"train": len(train_ds), "validation": len(val_ds), "test": len(test_ds)},
        "split_diagnostics": {"train": diag_train, "validation": diag_val, "test": diag_test},
        "data_diagnostics": data_diagnostics,
        "note": (
            "Memory handling: feature matrices are created as disk-backed arrays and standardised in-place in blocks; temporary cache is removed after successful completion. "
            "Existing solar/demand-aligned engineered features are retained. Added missing LSTM-aligned 15-minute historical weather context (last 24h), complete 96-step future forecast weather, causal future rolling weather, and full-resolution capacity/calendar trajectories. "
            "Future wind speed/direction fields are treated as forecast inputs aligned to each future valid timestamp. "
            "They must be forecasts available at the forecast origin, not observed future weather. "
            "A single row per valid timestamp does not encode separate forecast vintages for every issue time; "
            "use an issue_timestamp/target_timestamp forecast archive for strict vintage-matched operational backtesting. "
            "Future capacity mode 'known' assumes the future capacity schedule is known at issue time; use 'persist' otherwise."
        ),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    pd.DataFrame(horizon_metrics).to_csv(output_dir / "test_horizon_metrics.csv", index=False)
    pd.DataFrame([test_metrics]).to_csv(output_dir / "test_overall_metrics.csv", index=False)
    pd.DataFrame(history_log).to_csv(output_dir / "training_history.csv", index=False)
    print(f"Model saved: {model_path}", flush=True)
    print(f"Metadata saved: {output_dir / 'metadata.json'}", flush=True)
    print(f"Horizon metrics saved: {output_dir / 'test_horizon_metrics.csv'}", flush=True)

    # Release mappings and remove temporary feature cache after successful save.
    del train_ds, val_ds, test_ds, X_train, X_val, X_test
    del Y_train, Y_val, Y_test, Y_train_scaled, Y_val_scaled, Y_test_scaled
    gc.collect()
    try:
        shutil.rmtree(feature_cache_dir)
        print("Temporary feature cache removed.", flush=True)
    except OSError as exc:
        print(f"Warning: could not remove temporary feature cache: {exc}", flush=True)
    print("Training completed.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Amprion wind generation Dense NN with solar-reference-aligned engineered features."
    )
    parser.add_argument("--input-file", type=Path, default=DATA_DIR / "wind_train.csv")
    parser.add_argument("--interval-minutes", type=int, default=INTERVAL_MINUTES)
    parser.add_argument("--history", type=int, default=HISTORY_STEPS)
    parser.add_argument("--horizon", type=int, default=HORIZON_STEPS)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--hidden-1", type=int, default=256)
    parser.add_argument("--hidden-2", type=int, default=128)
    parser.add_argument("--hidden-3", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--scale-block-rows", type=int, default=2048, help="Rows processed per block during scaling.")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--future-capacity-mode", choices=["known", "persist"], default="known",
        help="known uses the capacity series over the future horizon (assumed announced); persist repeats origin capacity.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    if args.interval_minutes != 15:
        raise ValueError("This feature design currently expects 15-minute input data.")
    if args.history < max(WIND_LAGS):
        raise ValueError(f"history must be >= {max(WIND_LAGS)} steps.")
    if args.horizon != HORIZON_STEPS:
        raise ValueError(f"This feature builder is configured for a {HORIZON_STEPS}-step (24-hour) output horizon.")
    if min(args.hidden_1, args.hidden_2, args.hidden_3, args.batch_size, args.eval_batch_size, args.epochs, args.patience, args.threads, args.scale_block_rows) <= 0:
        raise ValueError("Architecture widths, batch sizes, epochs, patience and threads must be positive.")
    if args.dropout < 0 or args.dropout >= 1:
        raise ValueError("dropout must be in [0, 1).")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("learning-rate must be positive and weight-decay cannot be negative.")
    train(args)


if __name__ == "__main__":
    main()
