"""Station weather acquisition, hourly UTC alignment and persistent SQLite caching."""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from collections import OrderedDict
from contextlib import closing
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
import requests

from src.models.forecast_weather import ForecastWeatherUnavailable

LOG = logging.getLogger(__name__)
FIELDS = ("temperature_2m", "relative_humidity_2m", "cloud_cover", "shortwave_radiation",
          "wind_speed_100m", "wind_direction_100m")
HISTORY_COLUMNS = {
    "demand": {"temperature_2m": "temperature"},
    "solar": {"cloud_cover": "cloud_cover_mean_pct",
              "shortwave_radiation": "shortwave_radiation_mean_prev_hour_w_m2"},
    "wind": {"wind_speed_100m": "wind_speed_100m_mean_ms",
             "wind_direction_100m": "wind_direction_100m_circular_mean_deg"},
}
FORECAST_COLUMNS = {
    "demand": {"temperature_2m": "forecast_temperature"},
    "solar": {"cloud_cover": "forecast_cloud_cover_mean_pct",
              "shortwave_radiation": "forecast_shortwave_radiation_mean_prev_hour_w_m2"},
    "wind": {"wind_speed_100m": "future_wind_speed", "wind_direction_100m": "future_wind_direction"},
}
FORECAST_WEATHER_MODE = "open_meteo_forecast_temperature"
ARCHIVE_MODEL = "era5"
DEFAULT_FORECAST_MODEL = "ecmwf_ifs"
LIVE_REFRESH_SECONDS = 30 * 60
LIVE_GRACE = pd.Timedelta(hours=1)
LIVE_TTL_SECONDS = 6 * 3600
ARCHIVE_FINAL_DAYS = 8
ARCHIVE_START = pd.Timestamp("2024-01-01", tz="UTC")
ENDPOINTS = {
    "historical": ("archive-api", "archive", ""),
    "day_ahead": ("previous-runs-api", "forecast", "_previous_day1"),
    "recent_history": ("historical-forecast-api", "forecast", ""),
    "live": ("api", "forecast", ""),
    "live_day0": ("previous-runs-api", "forecast", ""),
}


def workbook_weather_config(frame: pd.DataFrame) -> dict[str, Any]:
    """One or more representative coordinates, with equal weight per point."""
    frame = frame.dropna(how="all").copy()
    frame.columns = [str(name).strip().lower() for name in frame.columns]
    required = {"location", "latitude", "longitude"}
    if not 1 <= len(frame) <= 12 or not required.issubset(frame.columns):
        raise ValueError("Station Location needs 1–12 rows with location, latitude and longitude.")
    sites = []
    for row in frame.to_dict("records"):
        name = str(row["location"]).strip() if pd.notna(row["location"]) else ""
        try:
            latitude, longitude = round(float(row["latitude"]), 6), round(float(row["longitude"]), 6)
        except (TypeError, ValueError) as exc:
            raise ValueError("Station Location needs numeric latitude and longitude for every point.") from exc
        if not name or not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
            raise ValueError("Station Location needs a name and valid latitude (±90) and longitude (±180).")
        sites.append({"location": name, "latitude": latitude, "longitude": longitude})
    if len({(row["latitude"], row["longitude"]) for row in sites}) != len(sites):
        raise ValueError("Station Location contains duplicate coordinates.")
    return {"provider": "open_meteo", "forecast_model": DEFAULT_FORECAST_MODEL, "sites": sites}


def station_weather(root: Path, station: str, entry: dict) -> StationWeather | None:
    config = entry.get("weather")
    if not config:
        return None
    validated = workbook_weather_config(pd.DataFrame(config["sites"]))
    model = config.get("forecast_model") or validated["forecast_model"]
    return StationWeather(root / "data" / "state" / station / "weather.sqlite3", validated["sites"], model)


def utc_index(timestamps) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True)).as_unit("ns")


def archive_training_span(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Choose the longest uninterrupted interval where archived forecasts exist."""
    valid = np.isfinite(frame[columns].to_numpy(dtype=float)).all(axis=1)
    edges = np.flatnonzero(np.diff(np.r_[False, valid, False]))
    if not len(edges):
        raise ValueError("No archived day-ahead weather overlaps this dataset. Use a more recent date range.")
    starts, ends = edges[::2], edges[1::2]
    span = int(np.argmax(ends - starts))
    return frame.iloc[starts[span]:ends[span]].reset_index(drop=True)


class StationWeather:
    def __init__(self, path: Path, sites: list[dict], forecast_model: str = DEFAULT_FORECAST_MODEL):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.sites = sites
        self.model = forecast_model
        self.location_key = json.dumps([(s["latitude"], s["longitude"]) for s in sites], separators=(",", ":"))
        self.session = requests.Session()
        self._frames: OrderedDict[tuple, tuple[float, pd.DataFrame]] = OrderedDict()
        self._retry_after: dict[str, float] = {}
        with closing(sqlite3.connect(self.path, timeout=15)) as conn, conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS weather_hourly (
                    series TEXT, utc_hour TEXT, temperature_2m REAL, relative_humidity_2m REAL,
                    cloud_cover REAL, shortwave_radiation REAL, wind_speed_100m REAL, wind_direction_100m REAL,
                    PRIMARY KEY (series, utc_hour)
                ) WITHOUT ROWID
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS weather_downloads (
                    series TEXT PRIMARY KEY, fetched_at_utc TEXT, kind TEXT, model TEXT, locations TEXT
                )
            """)

    def _key(self, kind: str, start: pd.Timestamp | None, end: pd.Timestamp | None) -> str:
        model = ARCHIVE_MODEL if kind == "historical" else self.model
        return json.dumps([kind, model, self.location_key, str(start), str(end)], separators=(",", ":"))

    def _get(self, kind: str, start: pd.Timestamp | None, end: pd.Timestamp | None) -> pd.DataFrame:
        host, route, suffix = ENDPOINTS[kind]
        api_key = os.getenv("OPEN_METEO_API_KEY")
        # The key is passed exclusively to Open-Meteo's authenticated hosts.
        if api_key:
            host = "customer-" + host
        params = {
            "latitude": ",".join(str(s["latitude"]) for s in self.sites),
            "longitude": ",".join(str(s["longitude"]) for s in self.sites),
            "hourly": ",".join(name + suffix for name in FIELDS),
            "timezone": "GMT",
            "wind_speed_unit": "ms",
            "models": ARCHIVE_MODEL if kind == "historical" else self.model,
        }
        if api_key:
            params["apikey"] = api_key
        if start is not None:
            params.update(start_date=start.strftime("%Y-%m-%d"), end_date=end.strftime("%Y-%m-%d"))
        else:
            params.update(past_days=1, forecast_days=3)
        url = f"https://{host}.open-meteo.com/v1/{route}"
        for attempt in range(3 if start is not None else 1):
            try:
                response = self.session.get(url, params=params, timeout=(12, 25 if start is not None else 15), allow_redirects=False)
                if response.status_code == 200:
                    result = response.json()
                    break
                if response.status_code not in {429, 500, 502, 503, 504}:
                    raise ForecastWeatherUnavailable(f"Open-Meteo {kind} returned HTTP {response.status_code}.")
                failure = f"HTTP {response.status_code}"
            except (requests.RequestException, ValueError) as exc:
                failure = type(exc).__name__
            if start is not None and attempt < 2:
                time.sleep(1 + attempt)
        else:
            raise ForecastWeatherUnavailable(f"Open-Meteo {kind} could not be downloaded ({failure}).")

        points = result if isinstance(result, list) else [result]
        if len(points) != len(self.sites):
            raise ForecastWeatherUnavailable(f"Open-Meteo {kind} returned incomplete site coverage.")
        frames = []
        try:
            for point in points:
                data = point["hourly"]
                frame = pd.DataFrame({name: data[name + suffix] for name in FIELDS},
                                     index=utc_index(data["time"]), dtype=float)
                frame = frame[~frame.index.duplicated(keep="last")].sort_index()
                frame = frame.replace([np.inf, -np.inf], np.nan)
                # Temperature is unbounded here. Other valid ranges follow the field units.
                for field, bound in (("relative_humidity_2m", 100), ("cloud_cover", 100),
                                     ("shortwave_radiation", np.inf), ("wind_speed_100m", np.inf),
                                     ("wind_direction_100m", 360)):
                    frame[field] = frame[field].where(frame[field].between(0, bound))
                frames.append(frame)
        except (KeyError, TypeError, ValueError) as exc:
            raise ForecastWeatherUnavailable(f"Open-Meteo {kind} returned an invalid hourly response.") from exc
        if len(frames) == 1:
            return frames[0]
        # Require all configured points and average direction on the unit circle.
        direction = "wind_direction_100m"
        matrix = pd.concat(frames, axis=1, keys=range(len(frames)))
        result = pd.DataFrame(index=matrix.index)
        for name in FIELDS:
            values = matrix.xs(name, axis=1, level=1)
            if name == direction:
                radians = np.deg2rad(values)
                sine, cosine = np.sin(radians).mean(axis=1, skipna=False), np.cos(radians).mean(axis=1, skipna=False)
                result[name] = (np.rad2deg(np.arctan2(sine, cosine)) % 360).where(np.hypot(sine, cosine) > 1e-8)
            else:
                result[name] = values.mean(axis=1, skipna=False)
        return result

    def _series(self, kind: str, start=None, end=None, ttl=None) -> pd.DataFrame:
        key = self._key(kind, start, end)
        clock = time.time()
        cached = self._frames.get((kind, start, end))
        if cached and (ttl is None or clock - cached[0] < ttl):
            self._frames.move_to_end((kind, start, end))
            return cached[1]
        with closing(sqlite3.connect(self.path, timeout=15)) as conn, conn:
            prefix = key + "::"
            row = conn.execute(
                "SELECT series, fetched_at_utc FROM weather_downloads WHERE substr(series, 1, ?) = ? "
                "ORDER BY fetched_at_utc DESC LIMIT 1", (len(prefix), prefix),
            ).fetchone() if kind == "live" else conn.execute(
                "SELECT series, fetched_at_utc FROM weather_downloads WHERE series = ?", (key,)
            ).fetchone()
            fetched = pd.Timestamp(row[1]).timestamp() if row else 0
            if row and (ttl is None or clock - fetched < ttl):
                frame = pd.read_sql_query(
                    "SELECT utc_hour, " + ", ".join(FIELDS) + " FROM weather_hourly WHERE series = ? ORDER BY utc_hour",
                    conn, params=(row[0],), index_col="utc_hour",
                ).astype(float)
                frame.index = utc_index(frame.index)
                self._remember((kind, start, end), fetched, frame)
                return frame

        if clock < self._retry_after.get(kind, 0):
            if kind == "live" and cached and clock - cached[0] < LIVE_TTL_SECONDS:
                return cached[1]
            raise ForecastWeatherUnavailable(f"Open-Meteo {kind} refresh is waiting to retry.")
        served_kind = kind
        try:
            if kind == "live":
                try:
                    fresh = self._get("live", None, None)
                except ForecastWeatherUnavailable:
                    fresh = self._get("live_day0", None, None)
                    served_kind = "live_day0"
            else:
                fresh = self._get(kind, start, end)
            if fresh.empty or (kind == "live" and not fresh.notna().any().any()):
                raise ForecastWeatherUnavailable(f"Open-Meteo {kind} returned no usable weather.")
        except ForecastWeatherUnavailable:
            self._retry_after[kind] = time.time() + 60
            if kind == "live":
                if cached is None and row:
                    # Recover a recently saved forecast after restarting the predictor.
                    with closing(sqlite3.connect(self.path, timeout=15)) as conn:
                        stale = pd.read_sql_query(
                            "SELECT utc_hour, " + ", ".join(FIELDS) + " FROM weather_hourly WHERE series = ? ORDER BY utc_hour",
                            conn, params=(row[0],), index_col="utc_hour",
                        ).astype(float)
                    stale.index = utc_index(stale.index)
                    cached = (fetched, stale)
                    self._remember((kind, start, end), *cached)
                if cached and time.time() - cached[0] < LIVE_TTL_SECONDS:
                    LOG.warning("Using cached station forecast after an Open-Meteo refresh failure.")
                    return cached[1]
            raise

        now = pd.Timestamp.now(tz="UTC")
        # Live snapshots retain retrieval time separately from meteorological valid time.
        series = key + "::" + uuid4().hex if kind == "live" else key
        with closing(sqlite3.connect(self.path, timeout=15)) as conn, conn:
            conn.execute("DELETE FROM weather_hourly WHERE series = ?", (series,))
            conn.executemany(
                "INSERT INTO weather_hourly VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                ((series, ts.isoformat(), *(None if pd.isna(x) else float(x) for x in values))
                 for ts, *values in fresh[list(FIELDS)].itertuples(name=None)),
            )
            conn.execute("INSERT OR REPLACE INTO weather_downloads VALUES (?, ?, ?, ?, ?)",
                         (series, now.isoformat(), served_kind,
                          ARCHIVE_MODEL if kind == "historical" else self.model, self.location_key))
        self._remember((kind, start, end), now.timestamp(), fresh)
        return fresh

    def _remember(self, key: tuple, fetched: float, frame: pd.DataFrame) -> None:
        self._frames[key] = (fetched, frame)
        self._frames.move_to_end(key)
        while len(self._frames) > 16:
            self._frames.popitem(last=False)

    def _dated(self, kind: str, hours: pd.DatetimeIndex) -> pd.DataFrame:
        """Cache bounded calendar-month requests, shared by uploads and replay."""
        today = pd.Timestamp.now(tz="UTC").normalize()
        archive_cutoff = today - pd.Timedelta(days=ARCHIVE_FINAL_DAYS)
        last = archive_cutoff if kind == "historical" else today
        if kind == "day_ahead":
            last += pd.Timedelta(days=1)
        first = hours.min().normalize()
        if kind == "day_ahead":
            first = max(first, ARCHIVE_START)
        if kind == "recent_history":
            first = max(first, archive_cutoff + pd.Timedelta(days=1))
        final = min(hours.max().normalize(), last)
        frames = []
        if first <= final:
            starts = pd.date_range(first.replace(day=1), final.replace(day=1), freq="MS", tz="UTC")
            for start in starts:
                end = min(start + pd.offsets.MonthEnd(0), last)
                if kind == "recent_history":
                    start = max(start, archive_cutoff + pd.Timedelta(days=1))
                ttl = 6 * 3600 if end >= archive_cutoff else None
                frames.append(self._series(kind, start, end, ttl))
        return pd.concat(frames).reindex(hours) if frames else pd.DataFrame(index=hours, columns=FIELDS, dtype=float)

    def historical(self, timestamps, *, live=False) -> pd.DataFrame:
        times = utc_index(timestamps)
        hours = times.floor("h").unique()
        if live:
            live_hours = self._series("live", ttl=LIVE_REFRESH_SECONDS)
            values = live_hours.reindex(hours)
            older = hours[hours < live_hours.index.min()]
            if len(older):
                values.update(self.historical(older))
            return values.reindex(times.floor("h")).set_axis(times)
        values = self._dated("historical", hours)
        cutoff = pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=ARCHIVE_FINAL_DAYS)
        recent_hours = hours[hours.normalize() > cutoff]
        if len(recent_hours):
            values.update(self._dated("recent_history", recent_hours))
        return values.reindex(times.floor("h")).set_axis(times)

    def forecast(self, timestamps, *, live: bool) -> pd.DataFrame:
        times = utc_index(timestamps)
        hours = times.floor("h")
        source = (self._series("live", ttl=LIVE_REFRESH_SECONDS) if live
                  else self._dated("day_ahead", hours.unique()))
        return source.reindex(hours).set_axis(times)

    def enrich(self, datasets: dict[str, pd.DataFrame]) -> None:
        all_times = utc_index(pd.concat([data["utc_timestamp"] for data in datasets.values()])).floor("h").unique()
        historical = self.historical(all_times)
        forecast = self.forecast(all_times, live=False)
        for kind, data in datasets.items():
            hours = utc_index(data["utc_timestamp"]).floor("h")
            observed = historical.reindex(hours)[list(HISTORY_COLUMNS[kind])]
            missing = observed.isna().any(axis=1)
            if missing.any():
                raise ForecastWeatherUnavailable(
                    f"Historical weather for {kind} is missing at {missing.index[missing][0].isoformat()} "
                    f"({int(missing.sum()):,} electricity rows). Check location and date coverage."
                )
            for source, target in HISTORY_COLUMNS[kind].items():
                data[target] = observed[source].to_numpy()
            for source, target in FORECAST_COLUMNS[kind].items():
                data[target] = forecast[source].reindex(hours).to_numpy()


def live_origin(origin: pd.Timestamp, *, replay: bool = False) -> bool:
    now = pd.Timestamp.now(tz="UTC")
    return not replay and now - LIVE_GRACE <= origin <= now + pd.Timedelta(minutes=5)
