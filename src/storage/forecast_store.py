
"""Append forecast and actual observation rows to CSV storage on disk.

Actual observations:
    One record per (station_id, utc_timestamp).

Forecasts:
    Preserve every forecast run, including different forecast origins.
"""

import csv
from pathlib import Path

FORECAST_FIELDS = [
    "station_id",
    "forecast_issued_at_utc",
    "forecast_origin_utc",
    "forecast_timestamp_utc",
    "horizon_step",
    "predicted_demand",
    "model_file",
    "weather_mode",
]

ACTUAL_FIELDS = [
    "station_id",
    "utc_timestamp",
    "actual_demand",
    "temperature",
]


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

    return _append_rows(path, records, FORECAST_FIELDS)


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

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    existing_keys = set()

    # Read existing observations to identify duplicates.
    if path.exists() and path.stat().st_size > 0:
        with path.open("r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)

            for row in reader:
                station_id = row.get("station_id")
                timestamp = row.get("utc_timestamp")

                if station_id and timestamp:
                    existing_keys.add((station_id, timestamp))

    new_records = []

    # Deduplicate both against existing CSV data
    # and within the incoming batch.
    for record in records:
        station_id = record.get("station_id")
        timestamp = record.get("utc_timestamp")

        if not station_id or not timestamp:
            continue

        key = (station_id, timestamp)

        if key not in existing_keys:
            new_records.append(record)
            existing_keys.add(key)

    saved = _append_rows(path, new_records, ACTUAL_FIELDS)

    return saved
