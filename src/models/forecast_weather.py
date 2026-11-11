"""Strict lookup for point-in-time future weather forecast covariates."""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from threading import RLock
from typing import Iterable

import numpy as np
import pandas as pd


class ForecastWeatherUnavailable(RuntimeError):
    """Raised when the required future weather forecast horizon is incomplete."""


_WEATHER_CACHE_LOCK = RLock()
_WEATHER_CACHE: OrderedDict[tuple, pd.DataFrame] = OrderedDict()
_WEATHER_CACHE_MAX_ITEMS = 8


def _normalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [
        str(c).replace("\ufeff", "").replace("\ufffd", "").replace("\xa0", " ").strip()
        for c in frame.columns
    ]
    return frame


def load_weather_lookup(path: str | Path, timestamp_column: str, value_columns: Iterable[str]) -> pd.DataFrame:
    """Load timestamp-keyed covariates and cache by source-file signature.

    The streaming predictor calls this for every dense demand forecast. Cache
    the normalised lookup so each 15-minute inference does not reread and sort
    the entire multi-year weather CSV. The returned frame is a copy so callers
    cannot mutate the shared cache.
    """
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ForecastWeatherUnavailable(f"Forecast-weather file does not exist: {source}")
    try:
        stat = source.stat()
    except OSError as exc:
        raise ForecastWeatherUnavailable(f"Could not stat forecast-weather file {source}: {exc}") from exc
    columns = tuple(value_columns)
    cache_key = (
        str(source), stat.st_mtime_ns, stat.st_size, timestamp_column, columns,
    )
    with _WEATHER_CACHE_LOCK:
        cached = _WEATHER_CACHE.get(cache_key)
        if cached is not None:
            _WEATHER_CACHE.move_to_end(cache_key)
            return cached.copy()

    try:
        frame = _normalise_columns(pd.read_csv(source, encoding="utf-8-sig", encoding_errors="replace"))
    except Exception as exc:
        raise ForecastWeatherUnavailable(f"Could not read forecast-weather file {source}: {exc}") from exc
    missing = [c for c in [timestamp_column, *columns] if c not in frame.columns]
    if missing:
        raise ForecastWeatherUnavailable(f"{source} is missing forecast columns: {missing}")
    frame[timestamp_column] = pd.to_datetime(
        frame[timestamp_column].astype("string").str.strip(),
        format="mixed", dayfirst=True, utc=True, errors="coerce",
    )
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=[timestamp_column]).sort_values(timestamp_column)
    frame = frame.drop_duplicates(timestamp_column, keep="last").set_index(timestamp_column).sort_index()
    frame = frame[list(columns)]
    with _WEATHER_CACHE_LOCK:
        _WEATHER_CACHE[cache_key] = frame
        _WEATHER_CACHE.move_to_end(cache_key)
        while len(_WEATHER_CACHE) > _WEATHER_CACHE_MAX_ITEMS:
            _WEATHER_CACHE.popitem(last=False)
    return frame.copy()


def select_forecast_horizon(
    lookup: pd.DataFrame,
    future_timestamps: pd.DatetimeIndex,
    value_columns: Iterable[str],
    *,
    context: str,
) -> pd.DataFrame:
    """Return exactly the required future valid times, never observed-weather fallbacks."""
    future_timestamps = pd.DatetimeIndex(pd.to_datetime(future_timestamps, utc=True))
    if lookup.empty:
        raise ForecastWeatherUnavailable(f"{context}: forecast-weather lookup is empty")
    selected = lookup.reindex(future_timestamps)
    missing_mask = selected[list(value_columns)].isna().any(axis=1)
    if missing_mask.any():
        missing_times = [ts.isoformat() for ts in selected.index[missing_mask][:5]]
        raise ForecastWeatherUnavailable(
            f"{context}: {int(missing_mask.sum())}/{len(future_timestamps)} future forecast-weather rows are missing; "
            f"examples={missing_times}. No observed-weather substitution was made."
        )
    values = selected[list(value_columns)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ForecastWeatherUnavailable(f"{context}: forecast covariates contain non-finite values")
    return selected.reset_index(names="utc_timestamp")
