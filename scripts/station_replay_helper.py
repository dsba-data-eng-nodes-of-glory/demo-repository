"""Configuration and safe per-station output preparation for the replay BAT."""
from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_FILES = ("actuals.csv", "forecasts.csv")


def read_yaml(relative: str) -> dict[str, Any]:
    path = ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(f"Required configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def resolve_path(value: str, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def bat_value(value: Any, label: str) -> str:
    """Reject characters that could break the generated CALL-ed BAT file."""
    text = str(value)
    if any(char in text for char in ('"', "%", "&", "|", "<", ">", "^", "!", "\r", "\n")):
        raise ValueError(
            f"{label} contains a character unsupported by this BAT launcher: {text!r}"
        )
    return text.replace("/", "\\")


def station_settings(station_id: str) -> dict[str, str]:
    if not re.fullmatch(r"[a-zA-Z0-9_-]{2,32}", station_id):
        raise ValueError("Station ID must be 2-32 characters: letters, numbers, _ or -.")

    registry = read_yaml("config/stations.yaml")
    stations = registry.get("stations", {})
    if station_id not in stations:
        available = ", ".join(stations) or "(none)"
        raise ValueError(f"Station {station_id!r} is not registered. Available: {available}")

    station = stations[station_id]
    kafka = read_yaml("config/kafka.yaml")
    spark = read_yaml("config/spark.yaml")

    input_value = station.get("input_file")
    raw_topic = station.get("raw_topic")
    featured_topic = station.get("featured_topic")
    model_value = station.get("model_checkpoint")
    if not all((input_value, raw_topic, featured_topic, model_value)):
        raise ValueError(
            f"Station {station_id!r} must define input_file, raw_topic, "
            "featured_topic and model_checkpoint in config/stations.yaml."
        )

    input_path = resolve_path(str(input_value), "Station streaming CSV")
    history_value = station.get("history_file")
    if history_value:
        history_path = resolve_path(str(history_value), "Station history CSV")
    else:
        # Match csv_producer.py's default history-file convention.
        history_path = input_path.with_name(f"{station_id}_history.csv")
        if not history_path.is_file():
            raise FileNotFoundError(f"Station history CSV not found: {history_path}")

    model_path = resolve_path(str(model_value), "Trained model checkpoint")

    # The Spark app computes ROOT / f"{checkpoint_location}_{station_id}".
    checkpoint_base = str(spark.get("checkpoint_location", "checkpoints/spark_features"))
    spark_checkpoint = Path(f"{checkpoint_base}_{station_id}")
    if spark_checkpoint.is_absolute():
        checkpoint_path = spark_checkpoint
    else:
        checkpoint_path = ROOT / spark_checkpoint

    bootstrap = str(kafka.get("bootstrap_servers", "127.0.0.1:9092"))
    delay = station.get("stream_delay_seconds", station.get("delay", 0.3))
    try:
        delay_float = float(delay)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid stream delay for {station_id}: {delay!r}") from exc
    if delay_float < 0:
        raise ValueError("Stream delay must be zero or greater.")

    # Replaying a station must never truncate another station's Kafka topic.
    for other_id, other in stations.items():
        if other_id == station_id:
            continue
        if raw_topic == other.get("raw_topic") or featured_topic == other.get("featured_topic"):
            raise ValueError(
                f"Kafka topic is shared by {station_id!r} and {other_id!r}. "
                "Full replay is blocked to protect the other station. Give each station unique topics."
            )

    return {
        "STATION": bat_value(station_id, "station ID"),
        "INPUT_CSV": bat_value(input_path, "input_file"),
        "HISTORY_CSV": bat_value(history_path, "history_file"),
        "MODEL_CHECKPOINT": bat_value(model_path, "model_checkpoint"),
        "RAW_TOPIC": bat_value(raw_topic, "raw_topic"),
        "FEATURED_TOPIC": bat_value(featured_topic, "featured_topic"),
        "SPARK_CHECKPOINT": bat_value(checkpoint_path, "Spark checkpoint"),
        "BOOTSTRAP": bat_value(bootstrap, "Kafka bootstrap_servers"),
        "STREAM_DELAY": str(delay_float),
    }


def write_env(path: Path, settings: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["@echo off"]
    lines.extend(f'set "{key}={value}"' for key, value in settings.items())
    path.write_text("\r\n".join(lines) + "\r\n", encoding="ascii")



def validate_station_output_files(station_id: str) -> None:
    """Preflight shared output files before Kafka records are truncated."""
    for filename in OUTPUT_FILES:
        source = PROCESSED_DIR / filename
        if not source.exists() or source.stat().st_size == 0:
            continue
        with source.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "station_id" not in reader.fieldnames:
                raise ValueError(
                    f"Cannot safely reset station {station_id!r}: {source} "
                    "has no station_id column. No Kafka records or output files were changed."
                )


def prepare_station_outputs(station_id: str) -> Path | None:
    """Back up shared output CSVs, then remove only rows for the replayed station."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    backup_dir = PROCESSED_DIR / "reset_backups" / f"{station_id}_{stamp}"

    plans: list[tuple[Path, list[str], list[dict[str, str]], Path]] = []
    for filename in OUTPUT_FILES:
        source = PROCESSED_DIR / filename
        if not source.exists() or source.stat().st_size == 0:
            continue

        with source.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames
            if not fieldnames or "station_id" not in fieldnames:
                raise ValueError(
                    f"Cannot safely clean {source}: it has no station_id column. "
                    "No output files were changed."
                )
            kept_rows = [row for row in reader if row.get("station_id") != station_id]

        plans.append((source, fieldnames, kept_rows, source.with_name(source.name + ".replay_tmp")))

    if not plans:
        print("No existing processed output CSVs to clean.")
        return None

    backup_dir.mkdir(parents=True, exist_ok=True)
    staged: list[Path] = []
    try:
        # Back up all originals before changing either shared file.
        for source, _, _, _ in plans:
            shutil.copy2(source, backup_dir / source.name)

        # Stage every replacement first; do not partially clean on a parse/write error.
        for source, fieldnames, rows, temp_path in plans:
            with temp_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)
            staged.append(temp_path)

        for (source, _, _, temp_path) in plans:
            os.replace(temp_path, source)

        print(f"Backups saved: {backup_dir}")
        for source, _, kept_rows, _ in plans:
            print(
                f"Prepared {source.name}: retained {len(kept_rows):,} rows for other stations; "
                f"removed rows for {station_id}."
            )
        return backup_dir
    except Exception:
        for temp_path in staged:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--station", required=True)
    parser.add_argument("--write-env", type=Path)
    parser.add_argument("--prepare-output-files", action="store_true")
    parser.add_argument("--validate-output-files", action="store_true")
    args = parser.parse_args()

    try:
        settings = station_settings(args.station)
        if args.write_env:
            write_env(args.write_env, settings)
            print(f"Loaded configuration for station: {args.station}")
            print(f"Input CSV: {settings['INPUT_CSV']}")
            print(f"History CSV: {settings['HISTORY_CSV']}")
            print(f"Raw topic: {settings['RAW_TOPIC']}")
            print(f"Featured topic: {settings['FEATURED_TOPIC']}")
            print(f"Spark checkpoint: {settings['SPARK_CHECKPOINT']}")
        if args.validate_output_files:
            validate_station_output_files(args.station)
        if args.prepare_output_files:
            prepare_station_outputs(args.station)
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

def delete_station_output_rows(station_id: str) -> dict[str, int]:
    """Remove one station's rows from shared output CSVs, preserving all other stations.

    This permanent-delete path intentionally does not retain a copy of the removed
    station's rows. Both CSVs are parsed and staged before either is replaced.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    plans: list[tuple[Path, list[str], list[dict[str, str]], Path]] = []
    removed: dict[str, int] = {}

    for filename in OUTPUT_FILES:
        source = PROCESSED_DIR / filename
        if not source.exists() or source.stat().st_size == 0:
            removed[filename] = 0
            continue

        with source.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames
            if not fieldnames or "station_id" not in fieldnames:
                raise ValueError(
                    f"Cannot delete station rows from {source}: missing station_id column. "
                    "No station rows were removed from the output files."
                )
            kept_rows: list[dict[str, str]] = []
            removed_count = 0
            for row in reader:
                if row.get("station_id") == station_id:
                    removed_count += 1
                else:
                    kept_rows.append(row)

        temp_path = source.with_name(f".{source.name}.{station_id}.delete_tmp")
        plans.append((source, fieldnames, kept_rows, temp_path))
        removed[filename] = removed_count

    staged: list[Path] = []
    try:
        for source, fieldnames, kept_rows, temp_path in plans:
            with temp_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(kept_rows)
            staged.append(temp_path)

        for source, _, _, temp_path in plans:
            os.replace(temp_path, source)
    except Exception:
        for temp_path in staged:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise

    return removed


if __name__ == "__main__":
    raise SystemExit(main())
