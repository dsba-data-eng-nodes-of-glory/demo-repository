"""Chunked replay of a station CSV into its configured raw Kafka topic.

Expected station files:
  configured history_file : initial history window (step count derived from interval)
  configured input_file   : observations immediately after the history window

The stream file is read in chunks so startup does not require loading and
sorting the entire file into memory before the first Kafka message is sent.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
import time
from uuid import uuid4
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import yaml
try:
    from kafka import KafkaProducer
except ImportError:  # Keep file/progress helpers testable without the Kafka extra.
    KafkaProducer = None

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


def acquire_station_lock(lock_path: Path):
    """Acquire an OS-released exclusive lock for this station's producer.

    The lock is advisory and process-scoped: Windows releases it when the
    process exits or crashes, so a stale lock file does not block recovery.
    The lock file itself is intentionally retained between runs.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(" ")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        raise RuntimeError(
            f"A producer for this station is already running (lock: {lock_path}). "
            "Stop the existing producer before starting another one."
        ) from exc

    handle.seek(0)
    handle.truncate()
    handle.write(json.dumps({"pid": os.getpid(), "station_lock": str(lock_path)}))
    handle.flush()
    os.fsync(handle.fileno())
    return handle


def release_station_lock(handle) -> None:
    if handle is None:
        return
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def load_producer_progress(
    progress_path: Path,
    station: str,
    csv_path: Path,
    topic: str,
    replay: bool = False,
) -> pd.Timestamp | None:
    """Load the last broker-acknowledged source timestamp.

    A changed source path or topic is rejected rather than silently replaying
    the whole CSV or skipping records from a different stream. ``--replay`` is
    the explicit opt-in to discard the cursor and start at the first source row.
    """
    if replay:
        progress_path.unlink(missing_ok=True)
        return None
    if not progress_path.exists():
        return None

    try:
        state = json.loads(progress_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Could not read producer progress file {progress_path}: {exc}. "
            "Repair or remove this file only after checking the Kafka/output state."
        ) from exc

    if not isinstance(state, dict):
        raise RuntimeError(f"Producer progress file {progress_path} must contain a JSON object.")

    expected_source = str(csv_path.resolve())
    if state.get("station_id") != station or state.get("source_file") != expected_source:
        raise RuntimeError(
            f"Producer progress file {progress_path} belongs to a different station/source. "
            "Use --replay only after resetting the corresponding Kafka topics and outputs."
        )
    if state.get("topic") != topic:
        raise RuntimeError(
            f"Producer progress topic is {state.get('topic')!r}, but the configured topic is {topic!r}. "
            "Use --replay only after resetting the corresponding Kafka topics and outputs."
        )

    timestamp_value = state.get("last_acked_timestamp")
    timestamp = pd.to_datetime(timestamp_value, utc=True, errors="coerce")
    if pd.isna(timestamp):
        raise RuntimeError(
            f"Producer progress file {progress_path} has no valid last_acked_timestamp."
        )
    return pd.Timestamp(timestamp)


def _atomic_replace_progress_with_retry(temp_path: Path, progress_path: Path) -> None:
    """Retry transient Windows locks without ever writing directly to the cursor.

    Antivirus/indexers/other readers may briefly deny the atomic rename on
    Windows. A persistent error must stop the producer: the prior cursor stays
    valid, and the acknowledged batch may be replayed on the next startup.
    """
    max_attempts = 15
    for attempt in range(1, max_attempts + 1):
        try:
            os.replace(temp_path, progress_path)
            return
        except OSError as exc:
            # Do not retry disk-full, invalid-path, or permission errors that
            # cannot plausibly be caused by a transient file/share lock.
            transient = isinstance(exc, PermissionError) or getattr(exc, "winerror", None) in (5, 32, 33)
            if not transient:
                raise

            # Replacing a Windows read-only destination cannot succeed until
            # its attribute is corrected. Never silently change permissions.
            try:
                is_readonly = (
                    os.name == "nt"
                    and bool(progress_path.stat().st_file_attributes & stat.FILE_ATTRIBUTE_READONLY)
                )
            except (FileNotFoundError, AttributeError, OSError):
                is_readonly = False
            if is_readonly:
                raise RuntimeError(
                    f"Producer checkpoint is read-only: {progress_path}. "
                    'Stop the producer and run: attrib -R "' + str(progress_path) + '". '
                    "Do not delete the cursor or restart with --replay."
                ) from exc

            if attempt == max_attempts:
                raise RuntimeError(
                    f"Could not replace producer checkpoint {progress_path} "
                    f"after {max_attempts} attempts (last error: {exc}). "
                    "The old checkpoint remains intact; an already acknowledged "
                    "Kafka batch may be resent after restart. Do not delete the "
                    "cursor or use --replay. Check file handles, antivirus, and permissions."
                ) from exc

            if attempt in (1, 5, 10):
                print(
                    f"WARNING: Windows temporarily blocked checkpoint update "
                    f"({exc}); retrying ({attempt}/{max_attempts})...",
                    flush=True,
                )
            time.sleep(min(0.10 * (1.6 ** (attempt - 1)), 0.8))


def save_producer_progress(
    progress_path: Path,
    station: str,
    csv_path: Path,
    topic: str,
    timestamp: pd.Timestamp,
) -> None:
    """Atomically persist progress only after a Kafka acknowledgement."""
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "station_id": station,
        "source_file": str(csv_path.resolve()),
        "topic": topic,
        "last_acked_timestamp": pd.Timestamp(timestamp).isoformat(),
        "updated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    # A unique temporary name avoids collisions with an orphaned .tmp file
    # from a previous crash or with an external process scanning old files.
    temp_path = progress_path.with_name(
        f"{progress_path.name}.{os.getpid()}.{uuid4().hex}.tmp"
    )
    with temp_path.open("x", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    _atomic_replace_progress_with_retry(temp_path, progress_path)


def iter_rows_after_progress(
    chunks: Iterator[pd.DataFrame],
    resume_timestamp: pd.Timestamp | None,
) -> Iterator[tuple[Any, pd.Timestamp]]:
    """Yield only source rows after the saved cursor; require an exact match.

    The cursor row must still exist in the CSV. This prevents a replaced or
    truncated source file from silently causing data loss on restart.
    """
    cursor_found = resume_timestamp is None
    previous_source_timestamp: pd.Timestamp | None = None
    for chunk in chunks:
        for row in chunk.itertuples(index=False):
            timestamp = pd.Timestamp(row.utc_timestamp)
            if previous_source_timestamp is not None and timestamp <= previous_source_timestamp:
                raise ValueError(
                    "Source timestamps are not strictly increasing across CSV chunks: "
                    f"{timestamp} follows {previous_source_timestamp}."
                )
            previous_source_timestamp = timestamp

            if resume_timestamp is not None and not cursor_found:
                if timestamp < resume_timestamp:
                    continue
                if timestamp == resume_timestamp:
                    cursor_found = True
                    continue
                raise RuntimeError(
                    f"Saved producer cursor {resume_timestamp.isoformat()} is not present in the "
                    f"source CSV; the next row is {timestamp.isoformat()}. The source may have "
                    "been replaced or truncated. Restore it or perform an intentional full replay."
                )
            yield row, timestamp

    if resume_timestamp is not None and not cursor_found:
        raise RuntimeError(
            f"Saved producer cursor {resume_timestamp.isoformat()} was not found in the source CSV. "
            "The source may have been replaced or truncated. Restore it or perform an intentional full replay."
        )


def run_producer(args: argparse.Namespace) -> None:
    if args.delay < 0 or args.limit < 0 or args.chunksize <= 0 or args.checkpoint_every <= 0:
        raise ValueError(
            "delay/limit must be non-negative; chunksize/checkpoint_every must be positive."
        )

    started = time.perf_counter()
    print("[1/5] Loading station and Kafka configuration...", flush=True)
    station_config = load_yaml("config/stations.yaml")
    kafka_config = load_yaml("config/kafka.yaml")
    model_config = load_yaml("config/model.yaml")
    stations = station_config.get("stations", {})
    if args.station not in stations:
        raise ValueError(
            f"Unknown station '{args.station}'. Configured stations: {list(stations)}"
        )

    station = stations[args.station]
    csv_path = Path(args.csv or station["input_file"])
    if not csv_path.is_absolute():
        csv_path = ROOT / csv_path
    csv_path = csv_path.resolve()

    history_value = args.history_file or station.get("history_file")
    if history_value:
        history_path = Path(history_value)
    else:
        history_path = csv_path.with_name(f"{args.station}_history.csv")
    if not history_path.is_absolute():
        history_path = ROOT / history_path
    history_path = history_path.resolve()

    topic = args.topic or station["raw_topic"]
    bootstrap = (
        args.bootstrap
        or os.getenv("KAFKA_BOOTSTRAP_SERVERS")
        or kafka_config["bootstrap_servers"]
    )
    interval_minutes = int(
        station.get("interval_minutes", model_config.get("interval_minutes", 15))
    )
    if interval_minutes <= 0:
        raise ValueError(f"interval_minutes for station '{args.station}' must be positive.")
    if not csv_path.is_file():
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
    configured_start = station.get("expected_stream_start_utc")
    if configured_start:
        expected_configured_start = pd.to_datetime(configured_start, utc=True, errors="raise")
        if first_stream_ts != expected_configured_start:
            raise ValueError(
                "Configured stream start does not match the source file. "
                f"Expected {expected_configured_start.isoformat()} from "
                f"expected_stream_start_utc, but {csv_path.name} starts at "
                f"{first_stream_ts.isoformat()}. Check config/stations.yaml and input_file."
            )
    if last_history_ts is not None:
        expected = last_history_ts + pd.Timedelta(minutes=interval_minutes)
        if first_stream_ts != expected:
            raise ValueError(
                "History/stream boundary is not continuous. "
                f"History ends at {last_history_ts}; expected stream to start at {expected}, "
                f"but it starts at {first_stream_ts}."
            )
        print(f"Boundary verified: {last_history_ts} -> {first_stream_ts}", flush=True)

    progress_value = args.progress_file or str(
        ROOT / "data" / "state" / args.station / "producer_progress.json"
    )
    progress_path = Path(progress_value)
    if not progress_path.is_absolute():
        progress_path = ROOT / progress_path
    progress_path = progress_path.resolve()
    lock_path = progress_path.with_suffix(progress_path.suffix + ".lock")

    # The operating-system lock is automatically released if the producer is
    # terminated, preventing multiple local processes from publishing the same
    # station at the same time.
    lock_handle = acquire_station_lock(lock_path)
    producer = None
    try:
        resume_timestamp = load_producer_progress(
            progress_path,
            args.station,
            csv_path,
            topic,
            replay=args.replay,
        )
        if resume_timestamp is None:
            print("Producer cursor: no saved progress; publishing from the first source row.", flush=True)
        else:
            print(
                f"Producer cursor: resuming after last acknowledged source row "
                f"{resume_timestamp.isoformat()}.",
                flush=True,
            )

        def chunks_with_first() -> Iterator[pd.DataFrame]:
            yield first_chunk
            yield from chunk_iterator

        rows_to_send = iter_rows_after_progress(chunks_with_first(), resume_timestamp)
        first_new_row = next(rows_to_send, None)
        if first_new_row is None:
            print(
                f"No unpublished source rows remain for station '{args.station}'. "
                f"Last acknowledged timestamp: {resume_timestamp}.",
                flush=True,
            )
            return

        print(
            f"[4/5] Connecting to Kafka at {bootstrap}; topic '{topic}'...",
            flush=True,
        )
        if KafkaProducer is None:
            raise RuntimeError(
                "kafka-python is required to run the CSV producer. "
                "Install the project's Kafka dependencies in this environment."
            )
        producer = KafkaProducer(
            bootstrap_servers=bootstrap,
            value_serializer=lambda value: json.dumps(value).encode("utf-8"),
            key_serializer=lambda value: value.encode("utf-8"),
            acks="all",
            retries=int(kafka_config.get("producer_retries", 10)),
            max_in_flight_requests_per_connection=1,
            max_block_ms=15000,
            request_timeout_ms=30000,
        )

        sent = 0
        pending_futures = []
        pending_last_timestamp: pd.Timestamp | None = None
        previous_timestamp = resume_timestamp

        def flush_acknowledged_batch() -> None:
            nonlocal pending_last_timestamp
            if not pending_futures:
                return
            # Do not move the cursor until every send in this batch has been
            # acknowledged by Kafka. If a send fails, a restart may replay this
            # small batch, which the predictor treats as duplicate timestamps.
            producer.flush(timeout=60)
            for future in pending_futures:
                future.get(timeout=60)
            if pending_last_timestamp is not None:
                save_producer_progress(
                    progress_path,
                    args.station,
                    csv_path,
                    topic,
                    pending_last_timestamp,
                )
            pending_futures.clear()
            log_message = (
                f"{args.station}: acknowledged {sent:,} records; "
                f"cursor={pending_last_timestamp.isoformat()}"
            )
            print(log_message, flush=True)
            pending_last_timestamp = None

        def rows_with_first():
            yield first_new_row
            yield from rows_to_send

        print(
            f"[5/5] Producer ready | delay={args.delay}s | limit={args.limit or 'all'} | "
            f"checkpoint_every={args.checkpoint_every} | "
            f"startup={time.perf_counter() - started:.1f}s",
            flush=True,
        )

        for row, timestamp in rows_with_first():
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
            pending_futures.append(future)
            pending_last_timestamp = timestamp
            sent += 1
            previous_timestamp = timestamp

            if sent == 1:
                metadata = future.get(timeout=30)
                print(
                    f"First record acknowledged: {timestamp} | "
                    f"partition={metadata.partition} offset={metadata.offset}",
                    flush=True,
                )

            if len(pending_futures) >= args.checkpoint_every:
                flush_acknowledged_batch()

            if args.limit and sent >= args.limit:
                break
            if args.delay > 0:
                time.sleep(args.delay)

        flush_acknowledged_batch()
        print(
            f"Completed: {sent:,} new records sent to '{topic}'. "
            f"Progress file: {progress_path}",
            flush=True,
        )
    finally:
        try:
            if producer is not None:
                producer.close(timeout=60)
        finally:
            release_station_lock(lock_handle)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publish a station's source CSV to its configured raw Kafka topic."
    )
    parser.add_argument("--station", required=True, help="Station key in config/stations.yaml")
    parser.add_argument("--csv", default=None, help="Streaming CSV path override")
    parser.add_argument("--history-file", default=None, help="History CSV path override")
    parser.add_argument("--topic", default=None, help="Optional Kafka topic override")
    parser.add_argument("--bootstrap", default=None, help="Optional Kafka bootstrap override")
    parser.add_argument(
        "--delay", type=float, default=2,
        help="Seconds between newly published records; use 0 for fastest replay.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Maximum new records; 0 sends all")
    parser.add_argument("--chunksize", type=int, default=2000, help="CSV rows read per chunk")
    parser.add_argument(
        "--checkpoint-every", type=int, default=50,
        help="Flush and persist the source cursor after this many acknowledged sends.",
    )
    parser.add_argument(
        "--progress-file", default=None,
        help="Producer progress JSON path; defaults to data/state/<station>/producer_progress.json.",
    )
    parser.add_argument(
        "--replay", action="store_true",
        help="Discard this producer cursor and publish from row 1. Use only after resetting Kafka topics and outputs.",
    )
    parser.add_argument(
        "--skip-boundary-check", action="store_true",
        help="Skip validation that first stream timestamp follows history by one interval.",
    )
    run_producer(parser.parse_args())


if __name__ == "__main__":
    main()
