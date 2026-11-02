"""Chunked replay of a station CSV into its configured raw Kafka topic.

Expected Amprion files:
  data/raw/amprion_history.csv : initial 672 observations
  data/raw/amprion.csv         : observations immediately after history

The stream file is read in chunks so startup does not require loading and
sorting the entire file into memory before the first Kafka message is sent.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import yaml
from kafka import KafkaProducer

ROOT = Path(__file__).resolve().parents[2]
REQUIRED_COLUMNS = ["utc_timestamp", "demand", "temperature"]


def load_yaml(relative_path: str) -> dict[str, Any]:
    path = ROOT / relative_path
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def normalize_chunk(raw: pd.DataFrame, source: Path) -> pd.DataFrame:
    """Apply the same timestamp/numeric cleaning as load_station_csv."""
    missing = set(REQUIRED_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"{source} is missing required columns: {sorted(missing)}")

    frame = raw[REQUIRED_COLUMNS].copy()
    frame["utc_timestamp"] = parse_utc_timestamps(frame["utc_timestamp"])

    frame["demand"] = pd.to_numeric(frame["demand"], errors="coerce")
    frame["temperature"] = pd.to_numeric(frame["temperature"], errors="coerce")
    frame = frame.dropna(subset=REQUIRED_COLUMNS)
    frame = frame.sort_values("utc_timestamp").drop_duplicates(
        "utc_timestamp", keep="last"
    )
    return frame.reset_index(drop=True)


def iter_clean_chunks(
    csv_path: Path, chunksize: int
) -> Iterator[pd.DataFrame]:
    """Read and normalise the stream incrementally."""
    for raw_chunk in pd.read_csv(csv_path, chunksize=chunksize):
        clean = normalize_chunk(raw_chunk, csv_path)
        if not clean.empty:
            yield clean


def parse_utc_timestamps(raw_timestamps: pd.Series) -> pd.Series:
    """Parse timestamps consistently: DD-MM-YYYY first, then ISO fallback.

    The explicit format matters for dates such as 07-01-2026, which pandas
    could otherwise interpret as July 1 instead of 7 January.
    """
    raw = raw_timestamps.astype("string").str.strip()
    parsed = pd.to_datetime(
        raw,
        format="%d-%m-%Y %H:%M",
        errors="coerce",
        utc=True,
    )
    fallback = parsed.isna() & raw.notna() & raw.ne("")
    if fallback.any():
        parsed.loc[fallback] = pd.to_datetime(
            raw.loc[fallback], errors="coerce", utc=True
        )
    return parsed


def read_history_end(history_path: Path) -> pd.Timestamp:
    if not history_path.exists():
        raise FileNotFoundError(f"History CSV not found: {history_path}")

    history = pd.read_csv(history_path, usecols=["utc_timestamp"])
    timestamps = parse_utc_timestamps(history["utc_timestamp"]).dropna()
    if timestamps.empty:
        raise ValueError(f"No valid timestamps found in {history_path}")
    return pd.Timestamp(timestamps.max())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publish a station's post-history CSV rows to its raw Kafka topic."
    )
    parser.add_argument("--station", required=True, help="Station key in config/stations.yaml")
    parser.add_argument("--csv", default=None, help="Streaming CSV path override")
    parser.add_argument("--history-file", default=None, help="History CSV path override")
    parser.add_argument("--topic", default=None, help="Optional Kafka topic override")
    parser.add_argument("--bootstrap", default=None, help="Optional Kafka bootstrap override")
    parser.add_argument(
        "--delay", type=float, default=2,
        help="Seconds between records; use 0 for fastest replay.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Maximum records; 0 sends all")
    parser.add_argument("--chunksize", type=int, default=2000, help="CSV rows read per chunk")
    parser.add_argument(
        "--skip-boundary-check", action="store_true",
        help="Skip validation that first stream timestamp follows history by one interval.",
    )
    args = parser.parse_args()

    if args.delay < 0 or args.limit < 0 or args.chunksize <= 0:
        raise ValueError("delay/limit must be non-negative and chunksize must be positive.")

    started = time.perf_counter()
    print("[1/5] Loading station and Kafka configuration...", flush=True)
    station_config = load_yaml("config/stations.yaml")
    kafka_config = load_yaml("config/kafka.yaml")
    model_config = load_yaml("config/model.yaml")
    stations = station_config.get("stations", {})
    if args.station not in stations:
        raise ValueError(f"Unknown station '{args.station}'. Configured stations: {list(stations)}")

    station = stations[args.station]
    csv_path = Path(args.csv or station["input_file"])
    if not csv_path.is_absolute():
        csv_path = ROOT / csv_path
    csv_path = csv_path.resolve()

    history_path = Path(args.history_file) if args.history_file else csv_path.with_name(
        f"{args.station}_history.csv"
    )
    if not history_path.is_absolute():
        history_path = ROOT / history_path
    history_path = history_path.resolve()

    topic = args.topic or station["raw_topic"]
    bootstrap = args.bootstrap or kafka_config["bootstrap_servers"]
    interval_minutes = int(model_config.get("interval_minutes", 15))
    if interval_minutes <= 0:
        raise ValueError("interval_minutes in config/model.yaml must be positive.")

    if not csv_path.exists():
        raise FileNotFoundError(f"Streaming CSV not found: {csv_path}")

    if not args.skip_boundary_check:
        print(f"[2/5] Checking history boundary: {history_path.name}...", flush=True)
        last_history_ts = read_history_end(history_path)
    else:
        last_history_ts = None

    print(f"[3/5] Opening streaming CSV in chunks of {args.chunksize:,} rows...", flush=True)
    chunk_iterator = iter_clean_chunks(csv_path, args.chunksize)
    first_chunk = next(chunk_iterator, None)
    if first_chunk is None:
        raise ValueError(f"No valid streaming rows found in {csv_path}.")

    first_stream_ts = pd.Timestamp(first_chunk["utc_timestamp"].iloc[0])
    if last_history_ts is not None:
        expected = last_history_ts + pd.Timedelta(minutes=interval_minutes)
        if first_stream_ts != expected:
            raise ValueError(
                "History/stream boundary is not continuous. "
                f"History ends at {last_history_ts}; expected stream to start at {expected}, "
                f"but it starts at {first_stream_ts}."
            )
        print(f"Boundary verified: {last_history_ts} -> {first_stream_ts}", flush=True)

    print(
        f"[4/5] Connecting to Kafka at {bootstrap}; topic '{topic}'...",
        flush=True,
    )
    producer = KafkaProducer(
        bootstrap_servers=bootstrap,
        value_serializer=lambda value: json.dumps(value).encode("utf-8"),
        key_serializer=lambda value: value.encode("utf-8"),
        acks="all",
        retries=5,
        max_block_ms=15000,
        request_timeout_ms=30000,
    )

    sent = 0
    previous_timestamp: pd.Timestamp | None = None
    pending_chunks = iter([first_chunk])

    def chunks_with_first() -> Iterator[pd.DataFrame]:
        yield from pending_chunks
        yield from chunk_iterator

    try:
        print(
            f"[5/5] Producer ready | delay={args.delay}s | limit={args.limit or 'all'} | "
            f"startup={time.perf_counter() - started:.1f}s",
            flush=True,
        )

        for chunk in chunks_with_first():
            for row in chunk.itertuples(index=False):
                timestamp = pd.Timestamp(row.utc_timestamp)
                if previous_timestamp is not None and timestamp <= previous_timestamp:
                    raise ValueError(
                        f"Non-increasing timestamp in {csv_path}: {timestamp} follows "
                        f"{previous_timestamp}. Check ordering/duplicates across CSV chunks."
                    )

                payload = {
                    "station_id": args.station,
                    "utc_timestamp": timestamp.isoformat(),
                    "demand": float(row.demand),
                    "temperature": float(row.temperature),
                }
                future = producer.send(topic, key=args.station, value=payload)
                sent += 1
                previous_timestamp = timestamp

                # Wait only for the first record so connection/topic errors are visible
                # immediately rather than remaining hidden in the producer background thread.
                if sent == 1:
                    metadata = future.get(timeout=30)
                    print(
                        f"First record acknowledged: {timestamp} | "
                        f"partition={metadata.partition} offset={metadata.offset}",
                        flush=True,
                    )

                if sent % 500 == 0:
                    producer.flush()
                    print(f"{args.station}: sent {sent:,} records to {topic}", flush=True)

                if args.limit and sent >= args.limit:
                    break
                if args.delay > 0:
                    time.sleep(args.delay)
            if args.limit and sent >= args.limit:
                break

        producer.flush()
        print(f"Completed: {sent:,} records sent to '{topic}'.", flush=True)
    finally:
        producer.close()


if __name__ == "__main__":
    main()
