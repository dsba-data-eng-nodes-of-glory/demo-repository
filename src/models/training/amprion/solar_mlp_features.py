from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TIMESTAMP_COLUMN = "utc_timestamp"
TARGET_COLUMN = "Photovoltaics [MWh]"
CLOUD_COLUMN = "cloud_cover_mean_pct"
RADIATION_COLUMN = "shortwave_radiation_mean_prev_hour_w_m2"
FORECAST_CLOUD_COLUMN = "forecast_cloud_cover_mean_pct"
FORECAST_RADIATION_COLUMN = "forecast_shortwave_radiation_mean_prev_hour_w_m2"
CAPACITY_COLUMN = "Photovoltaics_installed_capacity [MW]"

HISTORY_STEPS = 672
HORIZON_STEPS = 96
INTERVAL_MINUTES = 15


def parse_timestamps(values: pd.Series) -> pd.Series:
    """Parse canonical ISO UTC and legacy European timestamps without day/month inversion."""
    raw = values.astype("string").str.strip()
    # First handle the normalized ISO8601 values written by station onboarding.
    # A dayfirst=True fallback alone corrupts YYYY-MM-DD ISO dates after day 12.
    parsed = pd.to_datetime(raw, format="ISO8601", errors="coerce", utc=True)
    mask = parsed.isna() & raw.notna() & raw.ne("")
    if mask.any():
        parsed.loc[mask] = pd.to_datetime(raw.loc[mask], format="%d-%m-%Y %H:%M", errors="coerce", utc=True)
    mask = parsed.isna() & raw.notna() & raw.ne("")
    if mask.any():
        parsed.loc[mask] = pd.to_datetime(raw.loc[mask], format="mixed", dayfirst=True, errors="coerce", utc=True)
    return parsed


def numeric(series: pd.Series) -> pd.Series:
    raw = series.astype("string").str.strip().str.replace(",", "", regex=False)
    raw = raw.replace({"": pd.NA, "-": pd.NA, "nan": pd.NA, "None": pd.NA})
    return pd.to_numeric(raw, errors="coerce")


def load_solar_csv(
    path: str | Path,
    interval_minutes: int = INTERVAL_MINUTES,
    capacity_column: str = CAPACITY_COLUMN,
) -> tuple[pd.DataFrame, dict[str, int]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    frame = pd.read_csv(path, low_memory=False)
    frame.columns = [str(c).strip() for c in frame.columns]
    required = [
        TIMESTAMP_COLUMN, TARGET_COLUMN, CLOUD_COLUMN, RADIATION_COLUMN,
        FORECAST_CLOUD_COLUMN, FORECAST_RADIATION_COLUMN, capacity_column,
    ]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. Found columns: {list(frame.columns)}"
        )

    source_rows = len(frame)
    frame = frame[required].copy()
    frame[TIMESTAMP_COLUMN] = parse_timestamps(frame[TIMESTAMP_COLUMN])
    invalid_ts = int(frame[TIMESTAMP_COLUMN].isna().sum())
    for col in [
        TARGET_COLUMN, CLOUD_COLUMN, RADIATION_COLUMN,
        FORECAST_CLOUD_COLUMN, FORECAST_RADIATION_COLUMN, capacity_column,
    ]:
        frame[col] = numeric(frame[col])
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=[TIMESTAMP_COLUMN])
    if frame.empty:
        raise ValueError("No valid timestamps in input CSV")
    if interval_minutes <= 0:
        raise ValueError("interval_minutes must be positive")

    value_cols = [
        TARGET_COLUMN, CLOUD_COLUMN, RADIATION_COLUMN,
        FORECAST_CLOUD_COLUMN, FORECAST_RADIATION_COLUMN, capacity_column,
    ]
    frame = frame.groupby(TIMESTAMP_COLUMN, as_index=False)[value_cols].mean()
    frame = frame.sort_values(TIMESTAMP_COLUMN)
    ts = pd.DatetimeIndex(frame[TIMESTAMP_COLUMN])
    interval = pd.Timedelta(minutes=interval_minutes)
    offsets = (ts.asi8 - ts.asi8[0]) % interval.value
    if np.any(offsets):
        examples = ts[np.flatnonzero(offsets)[:5]].astype(str).tolist()
        raise ValueError(f"Timestamps are not aligned to {interval_minutes}-minute grid: {examples}")

    frame = frame.set_index(TIMESTAMP_COLUMN)
    full_idx = pd.date_range(frame.index.min(), frame.index.max(), freq=interval, tz="UTC")
    unique_count = len(frame)
    frame = frame.reindex(full_idx)
    inserted = len(frame) - unique_count
    frame.index.name = TIMESTAMP_COLUMN
    frame = frame.reset_index()

    if (frame[TARGET_COLUMN].dropna() < 0).any():
        raise ValueError(f"{TARGET_COLUMN} contains negative values")
    if ((frame[CLOUD_COLUMN].dropna() < 0) | (frame[CLOUD_COLUMN].dropna() > 100)).any():
        raise ValueError(f"{CLOUD_COLUMN} must be in the range 0..100")
    if (frame[RADIATION_COLUMN].dropna() < 0).any():
        raise ValueError(f"{RADIATION_COLUMN} contains negative values")
    if ((frame[FORECAST_CLOUD_COLUMN].dropna() < 0) |
            (frame[FORECAST_CLOUD_COLUMN].dropna() > 100)).any():
        raise ValueError(f"{FORECAST_CLOUD_COLUMN} must be in the range 0..100")
    if (frame[FORECAST_RADIATION_COLUMN].dropna() < 0).any():
        raise ValueError(f"{FORECAST_RADIATION_COLUMN} contains negative values")
    if (frame[capacity_column].dropna() < 0).any():
        raise ValueError(f"{capacity_column} contains negative values")

    diagnostics = {
        "source_rows": int(source_rows),
        "invalid_timestamps_removed": int(invalid_ts),
        "duplicate_timestamps_aggregated": int(source_rows - invalid_ts - unique_count),
        "inserted_missing_grid_rows": int(inserted),
        "missing_generation": int(frame[TARGET_COLUMN].isna().sum()),
        "missing_cloud_cover": int(frame[CLOUD_COLUMN].isna().sum()),
        "missing_shortwave_radiation": int(frame[RADIATION_COLUMN].isna().sum()),
        "missing_forecast_cloud_cover": int(frame[FORECAST_CLOUD_COLUMN].isna().sum()),
        "missing_forecast_shortwave_radiation": int(frame[FORECAST_RADIATION_COLUMN].isna().sum()),
        "missing_installed_capacity": int(frame[capacity_column].isna().sum()),
    }
    return frame, diagnostics


def split_points(n_rows: int) -> tuple[int, int]:
    return int(n_rows * 0.70), int(n_rows * 0.85)


def rolling_stats(values: np.ndarray, windows: list[int]) -> dict[tuple[int, str], np.ndarray]:
    s = pd.Series(np.asarray(values, dtype=np.float64))
    out: dict[tuple[int, str], np.ndarray] = {}
    for w in windows:
        roll = s.rolling(window=w, min_periods=1)
        out[(w, "mean")] = roll.mean().to_numpy(dtype=np.float32)
        out[(w, "std")] = roll.std(ddof=0).fillna(0.0).to_numpy(dtype=np.float32)
        out[(w, "max")] = roll.max().to_numpy(dtype=np.float32)
    return out


def calendar_matrix(timestamps: pd.DatetimeIndex) -> tuple[np.ndarray, list[str]]:
    ts = pd.DatetimeIndex(timestamps)
    minute = ts.hour.to_numpy() * 60 + ts.minute.to_numpy()
    weekday = ts.dayofweek.to_numpy()
    day = ts.dayofyear.to_numpy() - 1
    minute_angle = 2 * np.pi * minute / 1440.0
    weekday_angle = 2 * np.pi * weekday / 7.0
    year_angle = 2 * np.pi * day / 365.2425
    arr = np.column_stack([
        np.sin(minute_angle), np.cos(minute_angle),
        np.sin(weekday_angle), np.cos(weekday_angle),
        np.sin(year_angle), np.cos(year_angle),
    ]).astype(np.float32)
    return arr, ["origin_hour_sin", "origin_hour_cos", "origin_weekday_sin",
                 "origin_weekday_cos", "origin_year_sin", "origin_year_cos"]


def solar_elevation_noaa(timestamps: pd.DatetimeIndex, latitude: float, longitude: float) -> np.ndarray:
    """Vectorised NOAA approximation; timestamps must be timezone-aware UTC."""
    ts = pd.DatetimeIndex(timestamps)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    jd = ts.asi8.astype(np.float64) / 1e9 / 86400.0 + 2440587.5
    t = (jd - 2451545.0) / 36525.0

    l0 = np.mod(280.46646 + t * (36000.76983 + t * 0.0003032), 360.0)
    m = 357.52911 + t * (35999.05029 - 0.0001537 * t)
    ecc = 0.016708634 - t * (0.000042037 + 0.0000001267 * t)
    mr = np.deg2rad(m)
    center = (
        np.sin(mr) * (1.914602 - t * (0.004817 + 0.000014 * t))
        + np.sin(2 * mr) * (0.019993 - 0.000101 * t)
        + np.sin(3 * mr) * 0.000289
    )
    omega = 125.04 - 1934.136 * t
    lam = np.deg2rad(l0 + center - 0.00569 - 0.00478 * np.sin(np.deg2rad(omega)))
    seconds = 21.448 - t * (46.815 + t * (0.00059 - t * 0.001813))
    mean_obliq = 23.0 + (26.0 + seconds / 60.0) / 60.0
    obliq = np.deg2rad(mean_obliq + 0.00256 * np.cos(np.deg2rad(omega)))
    decl = np.arcsin(np.sin(obliq) * np.sin(lam))

    y = np.tan(obliq / 2.0) ** 2
    l0r = np.deg2rad(l0)
    eq_time = 4.0 * np.rad2deg(
        y * np.sin(2 * l0r)
        - 2 * ecc * np.sin(mr)
        + 4 * ecc * y * np.sin(mr) * np.cos(2 * l0r)
        - 0.5 * y * y * np.sin(4 * l0r)
        - 1.25 * ecc * ecc * np.sin(2 * mr)
    )

    minute_of_day = (ts.hour.to_numpy() * 60.0 + ts.minute.to_numpy()
                     + ts.second.to_numpy() / 60.0 + ts.microsecond.to_numpy() / 60e6)
    true_solar_minutes = np.mod(minute_of_day + eq_time + 4.0 * longitude, 1440.0)
    hour_angle = np.deg2rad(true_solar_minutes / 4.0 - 180.0)
    lat = math.radians(latitude)
    cos_zenith = (math.sin(lat) * np.sin(decl)
                  + math.cos(lat) * np.cos(decl) * np.cos(hour_angle))
    cos_zenith = np.clip(cos_zenith, -1.0, 1.0)
    elevation = 90.0 - np.rad2deg(np.arccos(cos_zenith))
    return elevation.astype(np.float32)


def _append_profile(
    cols: list[np.ndarray], names: list[str], prefix: str, values: np.ndarray,
    starts: np.ndarray, offsets: np.ndarray,
) -> None:
    mat = values[starts[:, None] + offsets[None, :]]
    cols.append(mat.astype(np.float32, copy=False))
    names.extend([f"{prefix}_{i + 1:02d}" for i in range(mat.shape[1])])


def build_feature_matrix(
    frame: pd.DataFrame,
    starts: np.ndarray,
    *,
    capacity_column: str = CAPACITY_COLUMN,
    include_solar_position: bool = False,
    latitude: float = 50.966111,
    longitude: float = 6.805556,
    chunk_size: int = 5000,
    target_mean: float | None = None,
    target_std: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], dict[str, int], np.ndarray]:
    """Build tabular origin features and 96-step target vectors.

    Historical weather profiles/statistics use observed cloud and radiation.
    All future weather profiles/statistics use forecast_cloud_cover_mean_pct and
    forecast_shortwave_radiation_mean_prev_hour_w_m2.

    Returns X, Y_scaled, weekly_baseline_raw, feature_names, diagnostics.
    All origins supplied in `starts` are considered; invalid windows are skipped.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")

    ts = pd.DatetimeIndex(frame[TIMESTAMP_COLUMN])
    y = frame[TARGET_COLUMN].to_numpy(dtype=np.float32)
    # Historical encoder features use observed weather. Future decoder-style
    # features use forecast weather from the two forecast columns only.
    cloud = frame[CLOUD_COLUMN].to_numpy(dtype=np.float32)
    rad = frame[RADIATION_COLUMN].to_numpy(dtype=np.float32)
    forecast_cloud = frame[FORECAST_CLOUD_COLUMN].to_numpy(dtype=np.float32)
    forecast_rad = frame[FORECAST_RADIATION_COLUMN].to_numpy(dtype=np.float32)
    capacity = frame[capacity_column].to_numpy(dtype=np.float32)
    n = len(frame)
    starts = np.asarray(starts, dtype=np.int64)
    h_offsets = np.arange(HORIZON_STEPS, dtype=np.int64)
    hour_offsets = np.arange(0, HORIZON_STEPS, 4, dtype=np.int64)
    pv_roll4 = pd.Series(y).rolling(4, min_periods=1).mean().to_numpy(dtype=np.float32)

    pv_stats = rolling_stats(y, [4, 24, 96, 672])
    cloud_stats = rolling_stats(cloud, [4, 96, 672])
    rad_stats = rolling_stats(rad, [4, 96, 672])
    cap_stats = rolling_stats(capacity, [96, 672])
    cal, cal_names = calendar_matrix(ts)
    elev = solar_elevation_noaa(ts, latitude, longitude) if include_solar_position else None
    if target_mean is None:
        target_mean = float(np.nanmean(y))
    if target_std is None:
        target_std = float(np.nanstd(y)) or 1.0
    if not np.isfinite(target_std) or target_std < 1e-8:
        target_std = 1.0

    # Establish the exact feature order once using names.
    names: list[str] = []
    # PV profiles (hourly aggregates, but each training origin remains 15-minute).
    names += [f"pv_previous_24h_hour_{i + 1:02d}" for i in range(24)]
    names += [f"pv_previous_week_same_24h_hour_{i + 1:02d}" for i in range(24)]
    lags = [1, 4, 8, 16, 24, 48, 96, 192, 288, 384, 480, 576, 672]
    names += [f"pv_lag_{lag}_steps" for lag in lags]
    for w, label in [(4, "1h"), (24, "6h"), (96, "24h"), (672, "7d")]:
        names += [f"pv_{stat}_{label}" for stat in ["mean", "std", "max"]]

    names += [f"hist_cloud_hour_{i + 1:02d}" for i in range(24)]
    names += [f"hist_radiation_hour_{i + 1:02d}" for i in range(24)]
    for variable in ["cloud", "radiation"]:
        for w, label in [(4, "1h"), (96, "24h"), (672, "7d")]:
            names += [f"hist_{variable}_{stat}_{label}" for stat in ["mean", "std"]]

    names += [f"future_cloud_hour_{i + 1:02d}" for i in range(24)]
    names += [f"future_radiation_hour_{i + 1:02d}" for i in range(24)]
    for variable in ["future_cloud", "future_radiation"]:
        names += [f"{variable}_{stat}_24h" for stat in ["mean", "std", "min", "max"]]

    names += cal_names
    names += [f"future_hour_sin_{i + 1:02d}" for i in range(24)]
    names += [f"future_hour_cos_{i + 1:02d}" for i in range(24)]
    names += [
        "capacity_at_origin", "capacity_mean_24h", "capacity_mean_7d",
        "capacity_change_vs_24h", "capacity_change_vs_7d",
        "future_capacity_first", "future_capacity_last", "future_capacity_mean",
        "future_capacity_std", "future_capacity_min", "future_capacity_max",
    ]
    names += [f"future_capacity_hour_{i + 1:02d}" for i in range(24)]

    if include_solar_position:
        names += ["solar_elevation_at_origin", "solar_elevation_previous_24h_mean",
                  "solar_elevation_previous_24h_max", "solar_elevation_future_mean",
                  "solar_elevation_future_min", "solar_elevation_future_max",
                  "solar_daylight_fraction_next_24h"]
        names += [f"future_solar_elevation_hour_{i + 1:02d}" for i in range(24)]

    X_parts: list[np.ndarray] = []
    Y_parts: list[np.ndarray] = []
    B_parts: list[np.ndarray] = []
    S_parts: list[np.ndarray] = []
    skipped_features = 0
    skipped_targets = 0

    for p in range(0, len(starts), chunk_size):
        s = starts[p:p + chunk_size]
        cols: list[np.ndarray] = []

        # Previous 24-hour PV profile and previous week's same 24-hour profile.
        pv_day_idx = s[:, None] - 93 + hour_offsets[None, :]
        pv_week_idx = s[:, None] - 669 + hour_offsets[None, :]
        cols.append(pv_roll4[pv_day_idx])
        cols.append(pv_roll4[pv_week_idx])

        lag_matrix = np.column_stack([y[s - lag] for lag in lags]).astype(np.float32)
        cols.append(lag_matrix)
        for w in [4, 24, 96, 672]:
            cols.append(np.column_stack([pv_stats[(w, stat)][s - 1] for stat in ["mean", "std", "max"]]))

        hist_weather_idx = s[:, None] - 96 + hour_offsets[None, :]
        cols.append(cloud[hist_weather_idx])
        cols.append(rad[hist_weather_idx])
        for arr_stats in [cloud_stats, rad_stats]:
            for w in [4, 96, 672]:
                cols.append(np.column_stack([arr_stats[(w, stat)][s - 1] for stat in ["mean", "std"]]))

        future_idx_hour = s[:, None] + hour_offsets[None, :]
        fut_cloud_hour = forecast_cloud[future_idx_hour]
        fut_rad_hour = forecast_rad[future_idx_hour]
        cols.append(fut_cloud_hour)
        cols.append(fut_rad_hour)
        fut_idx_all = s[:, None] + h_offsets[None, :]
        fut_cloud_all = forecast_cloud[fut_idx_all]
        fut_rad_all = forecast_rad[fut_idx_all]
        for mat in [fut_cloud_all, fut_rad_all]:
            cols.append(np.column_stack([
                np.mean(mat, axis=1), np.std(mat, axis=1),
                np.min(mat, axis=1), np.max(mat, axis=1),
            ]).astype(np.float32))

        # Origin time is the final historical timestamp, as in the V2 forecast issue convention.
        cols.append(cal[s - 1])
        fut_cal = cal[future_idx_hour]
        cols.append(fut_cal[:, :, 0])
        cols.append(fut_cal[:, :, 1])

        cap_origin = capacity[s - 1]
        cap_mean24 = cap_stats[(96, "mean")][s - 1]
        cap_mean7d = cap_stats[(672, "mean")][s - 1]
        cap_prev24 = capacity[np.maximum(0, s - 97)]
        cap_prev7d = capacity[s - 672]
        future_cap_profile = capacity[future_idx_hour]
        future_cap_all = capacity[fut_idx_all]
        cols.append(np.column_stack([
            cap_origin, cap_mean24, cap_mean7d,
            cap_origin - cap_prev24, cap_origin - cap_prev7d,
            future_cap_all[:, 0], future_cap_all[:, -1],
            np.mean(future_cap_all, axis=1), np.std(future_cap_all, axis=1),
            np.min(future_cap_all, axis=1), np.max(future_cap_all, axis=1),
        ]).astype(np.float32))
        cols.append(future_cap_profile)

        if include_solar_position and elev is not None:
            prev_elev_idx = s[:, None] - 96 + hour_offsets[None, :]
            fut_elev = elev[future_idx_hour]
            fut_elev_all = elev[fut_idx_all]
            cols.append(np.column_stack([
                elev[s - 1], np.mean(elev[prev_elev_idx], axis=1),
                np.max(elev[prev_elev_idx], axis=1), np.mean(fut_elev_all, axis=1),
                np.min(fut_elev_all, axis=1), np.max(fut_elev_all, axis=1),
                np.mean(fut_elev_all > 0, axis=1),
            ]).astype(np.float32))
            cols.append(fut_elev)

        X_chunk = np.column_stack(cols).astype(np.float32, copy=False)
        Y_raw = y[fut_idx_all]
        Y_scaled = ((Y_raw - target_mean) / target_std).astype(np.float32)
        baseline_idx = s[:, None] - 672 + h_offsets[None, :]
        baseline = y[baseline_idx].astype(np.float32, copy=False)

        finite_x = np.isfinite(X_chunk).all(axis=1)
        finite_y = np.isfinite(Y_scaled).all(axis=1)
        skipped_features += int((~finite_x).sum())
        skipped_targets += int((~finite_y).sum())
        valid = finite_x & finite_y & np.isfinite(baseline).all(axis=1)
        if np.any(valid):
            X_parts.append(X_chunk[valid])
            Y_parts.append(Y_scaled[valid])
            B_parts.append(baseline[valid])
            S_parts.append(s[valid])

    if not X_parts:
        raise ValueError(
            "No valid solar MLP windows were built. Check observed weather, "
            "forecast weather columns, target values and installed capacity."
        )
    X = np.concatenate(X_parts, axis=0)
    Y = np.concatenate(Y_parts, axis=0)
    baseline = np.concatenate(B_parts, axis=0)
    valid_starts = np.concatenate(S_parts, axis=0)
    if X.shape[1] != len(names):
        raise RuntimeError(f"Feature-name mismatch: matrix has {X.shape[1]} columns, names has {len(names)}")
    diagnostics = {
        "candidate_origins": int(len(starts)),
        "valid_origins": int(len(X)),
        "skipped_invalid_features": int(skipped_features),
        "skipped_invalid_targets": int(skipped_targets),
    }
    return X, Y, baseline, names, diagnostics, valid_starts


def metric_dict(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    a = np.asarray(actual, dtype=np.float64).ravel()
    p = np.asarray(predicted, dtype=np.float64).ravel()
    mask = np.isfinite(a) & np.isfinite(p)
    a, p = a[mask], p[mask]
    if not len(a):
        return {"MAE_MWh": float("nan"), "RMSE_MWh": float("nan"), "WAPE_pct": float("nan"),
                "R2": float("nan"), "Bias_MWh": float("nan")}
    err = p - a
    denom = float(np.sum(np.abs(a)))
    variance = float(np.sum((a - np.mean(a)) ** 2))
    return {
        "MAE_MWh": float(np.mean(np.abs(err))),
        "RMSE_MWh": float(np.sqrt(np.mean(err ** 2))),
        "WAPE_pct": float(np.sum(np.abs(err)) / denom * 100) if denom > 0 else float("nan"),
        "R2": float(1 - np.sum(err ** 2) / variance) if variance > 0 else float("nan"),
        "Bias_MWh": float(np.mean(err)),
    }
