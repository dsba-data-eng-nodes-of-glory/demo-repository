"""FastAPI routes for electricity demand results.

Optimised for append-only CSV storage: large result files are read from their
last rows rather than loading the entire file on every dashboard refresh.
"""
from __future__ import annotations

from collections import OrderedDict
from io import BytesIO
from pathlib import Path
from threading import RLock
import time

import pandas as pd
import yaml
from fastapi import APIRouter, Query

from backend.pipeline_manager import ROOT

router = APIRouter(prefix="/results", tags=["results"])

FORECAST_FILE = ROOT / "data/processed/forecasts.csv"
ACTUAL_FILE = ROOT / "data/processed/actuals.csv"
STATION_CONFIG_FILE = ROOT / "config/stations.yaml"

# Small in-process cache. File mtime/size invalidates entries as files change.
_CACHE: OrderedDict[tuple, tuple[float, tuple[int, int], pd.DataFrame]] = OrderedDict()
_CACHE_LOCK = RLock()
_CACHE_MAX_ITEMS = 12


def _station_output_path(station: str, kind: str) -> Path:
    """Return station-isolated output when configured, else legacy shared CSV."""
    try:
        with STATION_CONFIG_FILE.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}
        value = config.get("stations", {}).get(station, {}).get(f"{kind}_file")
        if value:
            path = Path(value)
            return path if path.is_absolute() else ROOT / path
    except (OSError, ValueError, yaml.YAMLError):
        pass
    return ACTUAL_FILE if kind == "actuals" else FORECAST_FILE


def _file_signature(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


def _cache_get(path: Path, key_suffix: tuple, ttl_seconds: float):
    key = (str(path.resolve()),) + key_suffix
    try:
        signature = _file_signature(path)
    except OSError:
        return key, None
    now = time.monotonic()
    with _CACHE_LOCK:
        item = _CACHE.get(key)
        if item is not None:
            cached_at, cached_signature, frame = item
            if cached_signature == signature and now - cached_at <= ttl_seconds:
                _CACHE.move_to_end(key)
                return key, frame.copy()
    return key, None


def _cache_put(key: tuple, path: Path, frame: pd.DataFrame) -> None:
    try:
        signature = _file_signature(path)
    except OSError:
        return
    with _CACHE_LOCK:
        _CACHE[key] = (time.monotonic(), signature, frame.copy())
        _CACHE.move_to_end(key)
        while len(_CACHE) > _CACHE_MAX_ITEMS:
            _CACHE.popitem(last=False)


def _read_csv_cached(path: Path, ttl_seconds: float = 300.0) -> pd.DataFrame:
    """Read a mostly static CSV once; re-read only when its file signature changes."""
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    key, cached = _cache_get(path, ("all",), ttl_seconds)
    if cached is not None:
        return cached
    try:
        frame = pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError, UnicodeDecodeError):
        return pd.DataFrame()
    _cache_put(key, path, frame)
    return frame


def _read_csv_tail(path: Path, max_rows: int, ttl_seconds: float = 1.0) -> pd.DataFrame:
    """Read only the final max_rows CSV records using reverse block reads.

    This is intended for the flat, one-record-per-line CSV files written by the
    POC. Quoted multi-line CSV fields are not supported by the storage format.
    """
    if max_rows <= 0 or not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()

    max_rows = int(max_rows)
    key, cached = _cache_get(path, ("tail", max_rows), ttl_seconds)
    if cached is not None:
        return cached

    try:
        with path.open("rb") as file:
            header = file.readline().rstrip(b"\r\n")
            header_end = file.tell()
            file.seek(0, 2)
            position = file.tell()
            chunks: list[bytes] = []
            newline_count = 0
            block_size = 64 * 1024

            while position > header_end and newline_count <= max_rows:
                start = max(header_end, position - block_size)
                file.seek(start)
                block = file.read(position - start)
                chunks.insert(0, block)
                newline_count += block.count(b"\n")
                position = start

        body = b"".join(chunks)
        lines = body.splitlines()
        # If reading began in the middle of a record, discard that partial line.
        if position > header_end and lines:
            lines = lines[1:]
        lines = lines[-max_rows:]
        if not lines:
            return pd.DataFrame()

        payload = header + b"\n" + b"\n".join(lines)
        frame = pd.read_csv(BytesIO(payload))
    except (OSError, ValueError, pd.errors.ParserError, UnicodeDecodeError):
        return pd.DataFrame()

    _cache_put(key, path, frame)
    return frame


def _json_safe_records(frame: pd.DataFrame) -> list:
    """Convert a DataFrame into JSON-safe records."""
    if frame.empty:
        return []
    clean = frame.replace([float("inf"), float("-inf")], float("nan"))
    clean = clean.astype(object).where(pd.notna(clean), None)
    return clean.to_dict(orient="records")


def _read_station_rows(
    path: Path,
    station: str,
    limit: int,
    scan_rows: int | None = None,
) -> pd.DataFrame:
    """Read recent rows for one station without loading a large CSV in full."""
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()

    limit = max(1, min(int(limit), 20000))
    # Scan extra rows in case several stations share a single append-only file.
    scan_rows = scan_rows or min(200000, max(50000, limit * 5))
    frame = _read_csv_tail(path, scan_rows)
    if frame.empty or "station_id" not in frame.columns:
        return pd.DataFrame()
    frame = frame[frame["station_id"].astype(str) == str(station)]
    return frame.tail(limit).reset_index(drop=True)


def _read_historical_actuals(station: str) -> pd.DataFrame:
    """Read static source actuals and the optional initial-history CSV."""
    if not STATION_CONFIG_FILE.exists():
        return pd.DataFrame()
    try:
        with STATION_CONFIG_FILE.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}
        station_config = config.get("stations", {}).get(station, {})
        input_file = station_config.get("input_file")
        if not input_file:
            return pd.DataFrame()
        input_path = Path(input_file)
        if not input_path.is_absolute():
            input_path = ROOT / input_path

        sources = [input_path]
        # The SCV/Amprion replay setup keeps the first 672 rows separately.
        history_path = input_path.with_name(f"{input_path.stem}_history{input_path.suffix}")
        if history_path.exists() and history_path.resolve() != input_path.resolve():
            sources.append(history_path)

        frames = []
        for source_path in sources:
            frame = _read_csv_cached(source_path)
            if frame.empty or "utc_timestamp" not in frame.columns:
                continue
            if "actual_demand" not in frame.columns and "demand" in frame.columns:
                frame = frame.rename(columns={"demand": "actual_demand"})
            if "actual_demand" not in frame.columns:
                continue
            frame = frame.copy()
            frame["station_id"] = station
            frames.append(frame)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True, sort=False)
    except (OSError, ValueError, yaml.YAMLError, pd.errors.ParserError):
        return pd.DataFrame()


def _combine_actuals(
    historical: pd.DataFrame,
    live: pd.DataFrame,
    station: str,
    limit: int,
    end_time: str | None = None,
) -> pd.DataFrame:
    """Merge source and live actuals, preferring live values at duplicate times.

    When end_time is omitted, the latest timestamp received by the live
    predictor is the replay watermark. This prevents future rows that happen
    to exist in the source CSV from pushing the current replay window out.
    """
    frames = []
    for priority, source_frame in ((0, historical), (1, live)):
        if source_frame.empty:
            continue
        frame = source_frame.copy()
        if "utc_timestamp" not in frame.columns:
            continue
        if "actual_demand" not in frame.columns and "demand" in frame.columns:
            frame = frame.rename(columns={"demand": "actual_demand"})
        if "actual_demand" not in frame.columns:
            continue
        frame["station_id"] = station
        frame["_timestamp_key"] = pd.to_datetime(
            frame["utc_timestamp"], utc=True, errors="coerce"
        )
        frame["actual_demand"] = pd.to_numeric(frame["actual_demand"], errors="coerce")
        frame = frame.dropna(subset=["_timestamp_key", "actual_demand"])
        frame["_priority"] = priority
        frames.append(frame)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True, sort=False)
    if end_time:
        cutoff = pd.to_datetime(end_time, utc=True, errors="coerce")
        if pd.notna(cutoff):
            combined = combined[combined["_timestamp_key"] <= cutoff]
    elif not live.empty and "utc_timestamp" in live.columns:
        # On replay, the newest live timestamp is the point-in-time cutoff.
        live_times = pd.to_datetime(live["utc_timestamp"], utc=True, errors="coerce")
        live_times = live_times.dropna()
        if not live_times.empty:
            combined = combined[combined["_timestamp_key"] <= live_times.max()]

    if combined.empty:
        return pd.DataFrame()

    combined = combined.sort_values(["_timestamp_key", "_priority"], kind="mergesort")
    combined = combined.drop_duplicates(["station_id", "_timestamp_key"], keep="last")
    combined["utc_timestamp"] = combined["_timestamp_key"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    combined = combined.sort_values("_timestamp_key").tail(max(1, min(int(limit), 20000)))
    combined = combined.drop(columns=["_timestamp_key", "_priority"], errors="ignore")
    return combined.reset_index(drop=True)


@router.get("/summary")
def summary(station: str | None = None):
    if station:
        frame = _read_station_rows(_station_output_path(station, "forecasts"), station, 1)
    else:
        frame = _read_csv_tail(FORECAST_FILE, 1)
    if frame.empty:
        return {"available": False, "rows": 0, "latest_forecast": None}
    latest = _json_safe_records(frame.tail(1))[0]
    return {"available": True, "rows": None, "latest_forecast": latest}


@router.get("/forecasts")
def forecasts(
    station: str,
    limit: int = Query(default=5000, ge=1, le=20000),
):
    frame = _read_station_rows(_station_output_path(station, "forecasts"), station, limit)
    if frame.empty:
        return {"records": [], "message": "No forecast records found for this station"}
    return {"records": _json_safe_records(frame)}


@router.get("/actuals")
def actuals(
    station: str,
    limit: int = Query(default=5000, ge=1, le=20000),
    end_time: str | None = Query(default=None, description="Optional UTC cutoff, normally the forecast origin"),
):
    historical = _read_historical_actuals(station)
    live = _read_station_rows(_station_output_path(station, "actuals"), station, limit)
    frame = _combine_actuals(
        historical=historical,
        live=live,
        station=station,
        limit=limit,
        end_time=end_time,
    )
    if frame.empty:
        return {"records": [], "message": "No actual observations found for this station"}
    return {"records": _json_safe_records(frame)}
