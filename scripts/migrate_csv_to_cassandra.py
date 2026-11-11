"""Explicit, replay-safe migration of existing operational CSV/SQLite to Cassandra.

Does NOT switch the active backend, remove files, or migrate training data.
Default is a non-mutating preflight. Run from the project root, e.g.:
    python -m scripts.migrate_csv_to_cassandra --station amprion
    python -m scripts.migrate_csv_to_cassandra --station amprion --apply
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import re
import sqlite3
from pathlib import Path
from typing import Callable, Iterator

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _resolve(value: str | None, fallback: str) -> Path:
    path = Path(value or fallback)
    return path if path.is_absolute() else ROOT / path


def _rows_for_station(path: Path, station: str) -> Iterator[dict]:
    if not path.is_file():
        return
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("station_id") == station:
                yield row


def _chunks(rows: Iterator[dict], size: int = 64):
    while part := list(itertools.islice(rows, size)):
        yield part


def migrate_csv(station: str, path: Path, write: Callable[[list[dict]], int]) -> int:
    total = 0
    for items in _chunks(_rows_for_station(path, station)):
        total += write(items)
    return total


def migrate_forecasts(station: str, path: Path, write: Callable[[list[dict]], int],
                      grouped: bool) -> int:
    """Group demand forecasts by run/origin; renewable writes accept mixed kinds."""
    if not grouped:
        return migrate_csv(station, path, write)
    total, buffer, previous = 0, [], None
    for record in _rows_for_station(path, station):
        key = (record.get("forecast_origin_utc"), record.get("forecast_run_id"))
        if buffer and key != previous:
            total += write(buffer)
            buffer = []
        buffer.append(record)
        previous = key
    if buffer:
        total += write(buffer)
    return total


def migrate_anomalies(station: str, path: Path, write: Callable) -> int:
    if not path.is_file():
        return 0
    total = 0
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
        for origin, fingerprint, payload in db.execute(
            "SELECT origin_utc, fingerprint, records_json FROM forecast_assessments ORDER BY origin_utc"
        ):
            write(station, origin, fingerprint, json.loads(payload))
            total += 1
    return total


def migrate_quality(station: str, path: Path, write: Callable) -> int:
    if not path.is_file():
        return 0
    total = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                if item.get("station_id") == station:
                    write(item)
                    total += 1
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station", required=True)
    parser.add_argument("--apply", action="store_true", help="Actually insert into Cassandra")
    args = parser.parse_args()
    station = args.station
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{1,31}", station):
        parser.error("Invalid station ID")
    config = yaml.safe_load((ROOT / "config" / "stations.yaml").read_text(encoding="utf-8")) or {}
    cfg = config.get("stations", {}).get(station)
    if cfg is None:
        parser.error("Station is not registered")
    paths = {
        "demand_actuals": _resolve(cfg.get("actuals_file"), f"data/processed/stations/{station}/actuals.csv"),
        "demand_forecasts": _resolve(cfg.get("forecasts_file"), f"data/processed/stations/{station}/forecasts.csv"),
        "renewable_forecasts": _resolve(cfg.get("renewable_forecasts_file"), f"data/processed/stations/{station}/renewable_forecasts.csv"),
        "renewable_actuals": ROOT / "data" / "processed" / "stations" / station / "renewable_actuals.csv",
        "anomaly_snapshots": ROOT / "data" / "processed" / "stations" / station / "anomaly_forecast_history.sqlite3",
        "quality_events": ROOT / "data" / "quality" / "quality_events.jsonl",
    }
    print(f"Source preflight for {station}; apply={args.apply}")
    for name, path in paths.items():
        print(f"  {name:22} {str(path) if path.is_file() else '(not found)'}")
    if not args.apply:
        print("No data written. Supply --apply to import existing operational history.")
        return
    # Require explicit Cassandra connection even when runtime backend remains csv.
    from src.storage.cassandra_store import (
        get_session, write_actuals, write_forecasts, write_renewable_forecasts,
        write_renewable_actuals, write_anomaly_assessment, write_quality_event,
    )
    get_session()
    counts = {
        "demand_actuals": migrate_csv(station, paths["demand_actuals"], write_actuals),
        "demand_forecasts": migrate_forecasts(station, paths["demand_forecasts"], write_forecasts, grouped=True),
        "renewable_forecasts": migrate_forecasts(station, paths["renewable_forecasts"], write_renewable_forecasts, grouped=False),
        "renewable_actuals": migrate_csv(station, paths["renewable_actuals"], write_renewable_actuals),
        "anomaly_snapshots": migrate_anomalies(station, paths["anomaly_snapshots"], write_anomaly_assessment),
        "quality_events": migrate_quality(station, paths["quality_events"], write_quality_event),
    }
    for name, count in counts.items():
        print(f"Imported {name}: {count}")
    print("Import complete. Source files unchanged. Switch storage.yaml only after verifying Cassandra queries.")


if __name__ == "__main__":
    main()
