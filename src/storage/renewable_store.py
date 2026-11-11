"""Idempotent CSV persistence for solar and wind forecast runs."""
from __future__ import annotations

import csv
from pathlib import Path
from threading import RLock

FIELDS = [
    "station_id", "generation_type", "forecast_issued_at_utc", "forecast_origin_utc",
    "forecast_run_id", "forecast_timestamp_utc", "horizon_step",
    "predicted_generation_mwh", "model_version", "weather_mode",
]

_INDEX_LOCK = RLock()
_EXISTING_KEYS: dict[str, tuple[tuple[int, int], set[tuple[str, str, str]]]] = {}


def _file_signature(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


def _load_existing_keys(path: Path) -> set[tuple[str, str, str]]:
    key = str(path.resolve())
    if not path.exists() or path.stat().st_size == 0:
        _EXISTING_KEYS[key] = ((0, 0), set())
        return _EXISTING_KEYS[key][1]

    signature = _file_signature(path)
    cached = _EXISTING_KEYS.get(key)
    if cached is not None and cached[0] == signature:
        return cached[1]

    existing: set[tuple[str, str, str]] = set()
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            run_id = row.get("forecast_run_id")
            kind = row.get("generation_type")
            step = row.get("horizon_step")
            if run_id and kind and step not in (None, ""):
                existing.add((str(run_id), str(kind), str(step)))
    _EXISTING_KEYS[key] = (signature, existing)
    return existing


def append_renewable_forecasts(path: str | Path, records: list[dict]) -> int:
    if not records:
        return 0
    from src.storage.forecast_store import storage_backend
    if storage_backend() == "cassandra":
        from src.storage.cassandra_store import write_renewable_forecasts
        return write_renewable_forecasts(records)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _INDEX_LOCK:
        existing = _load_existing_keys(path)
        fresh = []
        batch: set[tuple[str, str, str]] = set()
        for record in records:
            run_id = record.get("forecast_run_id")
            kind = record.get("generation_type")
            step = record.get("horizon_step")
            if not run_id or not kind or step in (None, ""):
                continue
            key = (str(run_id), str(kind), str(step))
            if key not in existing and key not in batch:
                batch.add(key)
                fresh.append(record)
        if not fresh:
            return 0
        needs_header = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
            if needs_header:
                writer.writeheader()
            writer.writerows(fresh)
            handle.flush()
        existing.update(batch)
        _EXISTING_KEYS[str(path.resolve())] = (_file_signature(path), existing)
        return len(fresh)


ACTUAL_FIELDS = ["station_id", "generation_type", "utc_timestamp", "actual_generation_mwh", "source"]
_ACTUAL_KEYS_CACHE: dict[str, tuple[tuple[int, int], set[tuple[str, str, str]]]] = {}


def append_renewable_actuals(path: str | Path, records: list[dict]) -> int:
    """Save only observed generation; never convert forecast values into actuals."""
    if not records:
        return 0
    from src.storage.forecast_store import storage_backend
    if storage_backend() == "cassandra":
        from src.storage.cassandra_store import write_renewable_actuals
        return write_renewable_actuals(records)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _INDEX_LOCK:
        key = str(path.resolve())
        if not path.exists() or not path.stat().st_size:
            existing = set()
        else:
            signature = _file_signature(path)
            cached = _ACTUAL_KEYS_CACHE.get(key)
            if cached is not None and cached[0] == signature:
                existing = cached[1]
            else:
                with path.open(newline="", encoding="utf-8-sig") as handle:
                    existing = {
                        (r["station_id"], r["generation_type"], r["utc_timestamp"])
                        for r in csv.DictReader(handle)
                    }
        fresh, seen = [], set()
        for item in records:
            compound = (str(item["station_id"]), str(item["generation_type"]), str(item["utc_timestamp"]))
            if compound not in existing and compound not in seen:
                seen.add(compound)
                fresh.append(item)
        if not fresh:
            return 0
        header = not path.exists() or not path.stat().st_size
        with path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=ACTUAL_FIELDS, extrasaction="ignore")
            if header:
                writer.writeheader()
            writer.writerows(fresh)
        existing.update(seen)
        _ACTUAL_KEYS_CACHE[key] = (_file_signature(path), existing)
        return len(fresh)


def read_renewable_forecasts(station: str, limit: int = 20000, end_time: str | None = None) -> list[dict]:
    from src.storage.forecast_store import storage_backend
    if storage_backend() != "cassandra":
        return []
    from src.storage.cassandra_store import read_renewable_forecasts as read
    return read(station, limit=limit, end_time=end_time)


def read_renewable_actuals(station: str, limit: int = 20000, end_time: str | None = None) -> list[dict]:
    from src.storage.forecast_store import storage_backend, ROOT
    if storage_backend() == "cassandra":
        from src.storage.cassandra_store import read_renewable_actuals as read
        return read(station, limit=limit, end_time=end_time)
    # CSV mode supports the same operational observations without loading
    # station training files into memory for each request.
    if not __import__("re").fullmatch(r"[A-Za-z0-9_-]{1,100}", station):
        raise ValueError("Invalid station name")
    path = ROOT / "data" / "processed" / "stations" / station / "renewable_actuals.csv"
    if not path.is_file():
        return []
    from datetime import datetime, timezone
    cutoff = (datetime.fromisoformat(end_time.replace("Z", "+00:00")).astimezone(timezone.utc)
              if end_time else None)
    # Bound disk I/O: read the last ~2*limit CSV rows, not the whole file.
    # The operational CSV schema has no multi-line quoted values.
    from io import StringIO
    with path.open("rb") as handle:
        header = handle.readline().decode("utf-8-sig").strip()
        start = handle.tell()
        handle.seek(0, 2)
        position = handle.tell()
        fragments = []
        newlines = 0
        wanted = max(1, min(40000, limit * 2))
        while position > start and newlines <= wanted:
            chunk_start = max(start, position - 65536)
            handle.seek(chunk_start)
            fragment = handle.read(position - chunk_start)
            fragments.insert(0, fragment)
            newlines += fragment.count(b"\n")
            position = chunk_start
    lines = b"".join(fragments).splitlines()
    if position > start and lines:
        lines = lines[1:]
    payload = header + "\n" + b"\n".join(lines[-wanted:]).decode("utf-8")
    rows = list(csv.DictReader(StringIO(payload)))
    if cutoff:
        rows = [r for r in rows
                if datetime.fromisoformat(r["utc_timestamp"].replace("Z", "+00:00")).astimezone(timezone.utc) <= cutoff]
    return list(rows)[-max(1, int(limit)):]
