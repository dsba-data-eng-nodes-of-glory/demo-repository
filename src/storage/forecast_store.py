
"""Append forecast and actual observation rows to CSV storage on disk.

Actual observations:
    One record per (station_id, utc_timestamp).

Forecasts:
    Preserve every forecast run, including different forecast origins.
"""

import csv
import os
from pathlib import Path
from threading import RLock
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]

def storage_backend() -> str:
    override = os.getenv("STORAGE_BACKEND")
    if override:
        value = override.strip().lower()
        if value not in {"csv", "cassandra"}:
            raise ValueError(f"Unsupported STORAGE_BACKEND={value!r}; expected csv or cassandra")
        return value
    config_path = ROOT / "config" / "storage.yaml"
    if not config_path.exists():
        return "csv"
    with config_path.open("r", encoding="utf-8") as handle:
        value = str((yaml.safe_load(handle) or {}).get("backend", "csv")).strip().lower()
        if value not in {"csv", "cassandra"}:
            raise ValueError(f"Unsupported storage backend {value!r}; expected csv or cassandra")
        return value

FORECAST_FIELDS = [
    "station_id",
    "forecast_issued_at_utc",
    "forecast_origin_utc",
    "forecast_run_id",
    "forecast_timestamp_utc",
    "horizon_step",
    "predicted_demand",
    "model_file",
    "model_version",
    "algorithm",
    "experiment_id",
    "weather_mode",
]

ACTUAL_FIELDS = [
    "station_id",
    "utc_timestamp",
    "actual_demand",
    "temperature",
]

# Per-process indexes avoid rereading ever-growing CSV files for each small
# streaming append. The file signature invalidates an index if a file is reset
# or changed outside this process; the index is then rebuilt once.
_KEY_CACHE_LOCK = RLock()
_ACTUAL_KEYS_CACHE: dict[str, tuple[tuple[int, int], set[tuple[str, str]]]] = {}
_FORECAST_KEYS_CACHE: dict[str, tuple[tuple[int, int], set[tuple[str, str]]]] = {}


def _signature(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


def _load_keys(
    path: Path,
    cache: dict[str, tuple[tuple[int, int], set[tuple[str, str]]]],
    key_fields: tuple[str, str],
) -> set[tuple[str, str]]:
    cache_key = str(path.resolve())
    if not path.exists() or path.stat().st_size == 0:
        cache[cache_key] = ((0, 0), set())
        return cache[cache_key][1]

    signature = _signature(path)
    cached = cache.get(cache_key)
    if cached is not None and cached[0] == signature:
        return cached[1]

    keys: set[tuple[str, str]] = set()
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            first, second = row.get(key_fields[0]), row.get(key_fields[1])
            if first not in (None, "") and second not in (None, ""):
                keys.add((str(first), str(second)))
    cache[cache_key] = (signature, keys)
    return keys


def _refresh_key_cache(
    path: Path,
    cache: dict[str, tuple[tuple[int, int], set[tuple[str, str]]]],
    keys: set[tuple[str, str]],
) -> None:
    cache[str(path.resolve())] = (_signature(path), keys)


def _append_rows(
    path: str | Path,
    records: list[dict],
    fields: list[str],
) -> int:
    """Append records and create the CSV header when needed."""

    if not records:
        return 0

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    needs_header = not path.exists() or path.stat().st_size == 0
    if not needs_header and fields == FORECAST_FIELDS:
        # Inspect only the header during normal appends. A one-time schema
        # migration still streams the file row by row if new columns are needed.
        temp_path = path.with_suffix(path.suffix + ".schema.tmp")
        migrated = False
        with path.open("r", newline="", encoding="utf-8") as source:
            reader = csv.reader(source)
            old_fields = next(reader, [])
            if not set(fields).issubset(old_fields):
                migrated = True
                source.seek(0)
                dict_reader = csv.DictReader(source)
                with temp_path.open("w", newline="", encoding="utf-8") as target:
                    writer = csv.DictWriter(target, fieldnames=fields, extrasaction="ignore")
                    writer.writeheader()
                    for old_row in dict_reader:
                        writer.writerow(old_row)
        if migrated:
            temp_path.replace(path)
        else:
            temp_path.unlink(missing_ok=True)

    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fields,
            extrasaction="ignore",
        )

        if needs_header:
            writer.writeheader()

        writer.writerows(records)

    return len(records)


def append_forecasts(
    path: str | Path,
    records: list[dict],
) -> int:
    """Append forecast runs without removing historical forecasts."""
    if storage_backend() == "cassandra":
        from src.storage.cassandra_store import write_forecasts
        return write_forecasts(records)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _KEY_CACHE_LOCK:
        existing = _load_keys(
            path, _FORECAST_KEYS_CACHE, ("forecast_run_id", "horizon_step")
        )
        fresh: list[dict] = []
        batch_keys: set[tuple[str, str]] = set()
        for record in records:
            run_id = record.get("forecast_run_id")
            horizon_step = record.get("horizon_step")
            if run_id in (None, "") or horizon_step in (None, ""):
                fresh.append(record)
                continue
            key = (str(run_id), str(horizon_step))
            if key in existing or key in batch_keys:
                continue
            batch_keys.add(key)
            fresh.append(record)
        saved = _append_rows(path, fresh, FORECAST_FIELDS)
        existing.update(batch_keys)
        if path.exists():
            _refresh_key_cache(path, _FORECAST_KEYS_CACHE, existing)
        return saved


def append_actuals(
    path: str | Path,
    records: list[dict],
) -> int:
    """Append only new actual observations.

    Duplicate key:
        (station_id, utc_timestamp)

    Existing actual observations are preserved.
    """

    if not records:
        return 0
    if storage_backend() == "cassandra":
        from src.storage.cassandra_store import write_actuals
        return write_actuals(records)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with _KEY_CACHE_LOCK:
        existing_keys = _load_keys(
            path, _ACTUAL_KEYS_CACHE, ("station_id", "utc_timestamp")
        )

        new_records = []
        batch_keys: set[tuple[str, str]] = set()
        for record in records:
            station_id = record.get("station_id")
            timestamp = record.get("utc_timestamp")
            if station_id in (None, "") or timestamp in (None, ""):
                continue
            key = (str(station_id), str(timestamp))
            if key not in existing_keys and key not in batch_keys:
                new_records.append(record)
                batch_keys.add(key)

        saved = _append_rows(path, new_records, ACTUAL_FIELDS)
        existing_keys.update(batch_keys)
        if path.exists():
            _refresh_key_cache(path, _ACTUAL_KEYS_CACHE, existing_keys)
        return saved


def read_actuals(station: str, limit: int = 5000) -> list[dict[str, Any]]:
    if storage_backend() == "cassandra":
        from src.storage.cassandra_store import read_actuals as read
        return read(station, limit)
    return []

def read_forecasts(station: str, limit: int = 5000) -> list[dict[str, Any]]:
    if storage_backend() == "cassandra":
        from src.storage.cassandra_store import read_forecasts as read
        return read(station, limit)
    return []

def read_latest_forecast(station: str) -> list[dict[str, Any]]:
    if storage_backend() == "cassandra":
        from src.storage.cassandra_store import read_latest_forecast as read
        return read(station)
    return []

def read_quality_events(station: str, limit: int = 100) -> list[dict[str, Any]]:
    if storage_backend() == "cassandra":
        from src.storage.cassandra_store import read_quality_events as read
        return read(station, limit)
    path = ROOT / "data" / "quality" / "quality_events.jsonl"
    if not path.exists():
        return []
    import json
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines()[-max(1, int(limit) * 5):]:
        try:
            item = json.loads(line)
            if item.get("station_id") == station:
                rows.append(item)
        except json.JSONDecodeError:
            continue
    return rows[-max(1, int(limit)):][::-1]

def storage_health() -> dict[str, Any]:
    if storage_backend() == "cassandra":
        from src.storage.cassandra_store import health_check
        return health_check()
    return {"available": True, "backend": "csv", "note": "CSV mode has no remote database health check."}
