"""Show the newest Kafka reading while replaying the unconsumed offset range."""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import time

from kafka import KafkaConsumer
from kafka.structs import TopicPartition
import yaml

from src.ingestion.station_lock import acquire_station_lock, release_station_lock
from src.quality.validator import validate_event
from src.storage.live_actuals import begin_recovery, initialize, live_actuals_path, next_offset, save_readings

ROOT = Path(__file__).resolve().parents[2]
LOG = logging.getLogger(__name__)


def observations(messages, station: str, interval: int) -> list[tuple[str, float]]:
    readings = []
    for message in messages:
        try:
            payload = json.loads(message.value or b"")
            if not isinstance(payload, dict) or not isinstance(payload.get("utc_timestamp"), str):
                continue
            event, _ = validate_event(payload, station, interval, require_temperature=False)
            if event is not None:
                timestamp = event["utc_timestamp"].isoformat(timespec="microseconds")
                readings.append((timestamp, event["demand"]))
        except (ValueError, TypeError):
            LOG.warning("Could not read live observation at Kafka offset %s", message.offset)
    return readings


def consume(consumer, partition: TopicPartition, path: Path, station: str, interval: int,
            *, demo: bool = False) -> None:
    consumer.assign([partition])
    earliest = consumer.beginning_offsets([partition])[partition]
    saved = next_offset(path)
    end = consumer.end_offsets([partition])[partition]
    if saved is not None and saved > end:
        raise RuntimeError("Saved live actuals cursor is ahead of Kafka. Check the selected topic.")
    if saved is not None and saved < earliest:
        LOG.warning("Kafka no longer retains offsets %s to %s. Their readings may require source backfill.", saved, earliest - 1)
    position = max(earliest, saved if saved is not None else earliest)
    recovering_until = end if end - position > 1 else position
    if saved is None or saved < position:
        save_readings(path, [], position)
    if recovering_until > position:
        begin_recovery(path, position, recovering_until)
    published_edge = position
    if recovering_until > position:
        LOG.info("Catching up Kafka offsets %s to %s", position, recovering_until - 1)

    consumer.seek(partition, position)
    while True:
        catching_up = position < recovering_until
        if catching_up:
            tail = consumer.end_offsets([partition])[partition]
            if tail > published_edge and tail > position:
                consumer.seek(partition, tail - 1)
                latest = consumer.poll(timeout_ms=800, max_records=1).get(partition, [])
                if latest:
                    save_readings(path, observations(latest, station, interval))
                    published_edge = latest[-1].offset + 1
                consumer.seek(partition, position)
        batch_size = 1 if catching_up and demo else 100
        messages = consumer.poll(timeout_ms=700, max_records=batch_size).get(partition, [])
        if not messages:
            continue
        position = messages[-1].offset + 1
        save_readings(path, observations(messages, station, interval), position)
        if catching_up and position >= recovering_until:
            LOG.info("Live actuals catch-up complete through offset %s", recovering_until - 1)
        if catching_up and demo:
            time.sleep(1.2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--station", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    stations = yaml.safe_load((ROOT / "config/stations.yaml").read_text(encoding="utf-8"))["stations"]
    station = stations[args.station]
    kafka = yaml.safe_load((ROOT / "config/kafka.yaml").read_text(encoding="utf-8"))
    topic = station["featured_topic"]
    demo = bool(station.get("live_recovery_demo", False))
    path = live_actuals_path(ROOT, args.station)
    lock = acquire_station_lock(path.with_name("live_actuals.lock"), "live actuals reader")
    try:
        initialize(path, topic)
        bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS") or kafka["bootstrap_servers"]
        with KafkaConsumer(bootstrap_servers=bootstrap, group_id=None, enable_auto_commit=False,
                           auto_offset_reset="none") as consumer:
            partitions = consumer.partitions_for_topic(topic)
            if partitions != {0}:
                LOG.error("Independent live recovery requires a single ordered Kafka partition for %s.", topic)
                return
            consume(consumer, TopicPartition(topic, 0), path, args.station,
                    int(station.get("interval_minutes", 15)), demo=demo)
    finally:
        release_station_lock(lock)


if __name__ == "__main__":
    main()
