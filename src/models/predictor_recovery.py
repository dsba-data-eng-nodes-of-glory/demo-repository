"""Durable observation window and Kafka positions for a station predictor."""
from __future__ import annotations

import json
import logging
import math
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from typing import Any

import pandas as pd
from kafka.structs import TopicPartition

from src.quality.validator import validate_event

LOG = logging.getLogger(__name__)
GENERATION_FIELDS = ("solar_mwh", "solar_capacity_mw", "wind_mwh", "wind_capacity_mw")


def predictor_state_path(root: Path, station: str) -> Path:
    return root / "data" / "state" / station / "predictor_recovery.sqlite3"


def recovery_event(event: dict, payload: dict, message: Any, *, replay_weather: bool) -> dict:
    row = dict(event, partition=message.partition, offset=message.offset, is_simulated=replay_weather)
    for field in GENERATION_FIELDS:
        try:
            value = float(payload.get(field))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            row[field] = value
    return row


def contiguous_history(seed: list[dict], recent: list[dict], interval: pd.Timedelta, count: int) -> list[dict]:
    """Merge the seed with known readings and keep the latest uninterrupted window."""
    by_time = {row["utc_timestamp"]: row for row in seed + recent}
    result = []
    for ts in sorted(by_time, reverse=True):
        if result and result[-1]["utc_timestamp"] - ts != interval:
            break
        result.append(by_time[ts])
        if len(result) == count:
            break
    return result[::-1]


class PredictorRecovery:
    def __init__(self, path: Path, *, station: str, topic: str, group: str, interval_minutes: int, capacity: int):
        self.path = path
        self.capacity = capacity
        path.parent.mkdir(parents=True, exist_ok=True)
        identity = json.dumps([station, topic, group, interval_minutes])
        with closing(self.connect()) as db, db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("CREATE TABLE IF NOT EXISTS recovery_stream (identity TEXT PRIMARY KEY)")
            db.execute("CREATE TABLE IF NOT EXISTS recovery_positions (partition_id INTEGER PRIMARY KEY, next_offset INTEGER NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS recovery_events (event_time INTEGER PRIMARY KEY, event_json TEXT NOT NULL)")
            registered = db.execute("SELECT identity FROM recovery_stream").fetchone()
            if registered is not None and registered[0] != identity:
                raise RuntimeError(
                    f"Predictor recovery at {path} belongs to a different Kafka topic, group or interval. "
                    "Use the matching station settings or a fresh state for the changed stream."
                )
            db.execute("INSERT OR IGNORE INTO recovery_stream VALUES (?)", (identity,))

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5)
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    def load(self) -> tuple[list[dict], dict[int, int]]:
        with closing(self.connect()) as db:
            rows = db.execute(
                "SELECT event_json FROM recovery_events ORDER BY event_time DESC LIMIT ?", (self.capacity,)
            ).fetchall()
            offsets = dict(db.execute("SELECT partition_id, next_offset FROM recovery_positions"))
        events = [json.loads(raw) for (raw,) in reversed(rows)]
        for row in events:
            row["utc_timestamp"] = pd.Timestamp(row["utc_timestamp"])
        return events, offsets

    def save(self, records: list[dict], offsets: dict[int, int]) -> None:
        """Add accepted readings and next offsets in the same SQLite transaction."""
        with closing(self.connect()) as db, db:
            db.executemany(
                "INSERT OR IGNORE INTO recovery_events VALUES (?, ?)",
                ((row["utc_timestamp"].value, json.dumps(row, default=str, allow_nan=False))
                 for row in records),
            )
            db.executemany(
                "INSERT INTO recovery_positions VALUES (?, ?) "
                "ON CONFLICT(partition_id) DO UPDATE SET next_offset=max(next_offset, excluded.next_offset)",
                offsets.items(),
            )
            if records:
                db.execute(
                    "DELETE FROM recovery_events WHERE event_time < ("
                    "SELECT event_time FROM recovery_events ORDER BY event_time DESC LIMIT 1 OFFSET ?)",
                    (self.capacity - 1,),
                )


def restore_history(consumer, recovery: PredictorRecovery, topic: str, *, station: str,
                    interval_minutes: int, require_temperature: bool, default_replay: bool
                    ) -> tuple[list[dict], dict[int, int]]:
    events, saved = recovery.load()
    partitions = [TopicPartition(topic, number) for number in sorted(consumer.partitions_for_topic(topic))]
    if not partitions:
        raise RuntimeError(f"No Kafka partitions are available for {topic}.")
    earliest = consumer.beginning_offsets(partitions)
    latest = consumer.end_offsets(partitions)
    if not saved.keys() <= {partition.partition for partition in partitions} or any(
        saved[partition.partition] > latest[partition]
        for partition in partitions if partition.partition in saved
    ):
        raise RuntimeError(f"Predictor recovery offsets exceed the current Kafka log for {topic}.")
    ranges, committed = {}, {}
    for partition in partitions:
        offset = consumer.committed(partition)
        if offset is None or not earliest[partition] <= offset <= latest[partition]:
            continue
        if offset > saved.get(partition.partition, -1):
            first = max(earliest[partition], offset - recovery.capacity * 4)
            if first < offset:
                ranges[partition] = (first, offset)
            committed[partition.partition] = offset
    if not committed:
        return events, saved
    backfill = read_committed_history(
        consumer, ranges, station=station, interval_minutes=interval_minutes,
        require_temperature=require_temperature, default_replay=default_replay,
    )
    if ranges:
        consumer.unsubscribe()
    recovery.save(backfill, committed)
    return recovery.load()


def read_committed_history(consumer, partitions: dict[TopicPartition, tuple[int, int]], *,
                           station: str, interval_minutes: int, require_temperature: bool,
                           default_replay: bool) -> list[dict]:
    """Read a bounded slice of already committed Kafka records after first install."""
    if not partitions:
        return []
    consumer.assign(list(partitions))
    for partition, (first, _) in partitions.items():
        consumer.seek(partition, first)
    remaining = dict(partitions)
    records = []
    deadline = time.monotonic() + 30
    while remaining and time.monotonic() < deadline:
        batches = consumer.poll(timeout_ms=1000, max_records=500)
        for partition, messages in batches.items():
            if partition not in remaining:
                continue
            _, stop = remaining[partition]
            for message in messages:
                if message.offset >= stop:
                    break
                try:
                    payload = json.loads(message.value or b"")
                except (ValueError, UnicodeError):
                    continue
                if not isinstance(payload, dict):
                    continue
                event, _ = validate_event(payload, station, interval_minutes, require_temperature=require_temperature)
                if event is not None:
                    replay = payload.get("is_simulated")
                    records.append(recovery_event(
                        event, payload, message, replay_weather=default_replay if replay is None else replay is True,
                    ))
            if messages and messages[-1].offset >= stop - 1:
                remaining.pop(partition)
                consumer.pause(partition)
        if not batches:
            for partition, (_, stop) in list(remaining.items()):
                if consumer.position(partition) >= stop:
                    remaining.pop(partition)
                    consumer.pause(partition)
    if remaining:
        LOG.warning("Historical Kafka recovery reached its time limit. Recovered %s readings.", len(records))
    return records
