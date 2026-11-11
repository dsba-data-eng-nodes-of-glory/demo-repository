"""Persist Spark-routed dead-letter events into the quality-event store.

Run one shared instance for the shared DLQ topic:
    python -m src.quality.dlq_consumer
"""
from __future__ import annotations

import json
import logging
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from kafka import KafkaConsumer
from kafka.structs import OffsetAndMetadata, TopicPartition

from src.quality.event_store import record_quality_event

ROOT = Path(__file__).resolve().parents[2]
LOG = logging.getLogger("electricity-dlq-consumer")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def _event_from_message(message: Any, payload: dict[str, Any]) -> dict[str, Any]:
    station = str(payload.get("station_id") or "unknown")
    source_topic = str(payload.get("source_topic") or message.topic)
    source_partition = int(payload.get("source_partition", message.partition))
    source_offset = int(payload.get("source_offset", message.offset))
    rule = str(payload.get("rule_name") or payload.get("quality_reason") or "invalid_record")
    event_ts = (
        datetime.fromtimestamp(message.timestamp / 1000, tz=timezone.utc)
        if message.timestamp is not None and message.timestamp >= 0
        else datetime.now(timezone.utc)
    )
    return {
        "station_id": station,
        "event_timestamp": event_ts,
        "event_id": f"{source_topic}:{source_partition}:{source_offset}:{rule}",
        "severity": str(payload.get("severity", "error")),
        "rule_name": rule,
        "message": str(payload.get("error") or payload.get("quality_reason") or "Rejected input")[:2000],
        "source_topic": source_topic,
        "source_partition": source_partition,
        "source_offset": source_offset,
    }


def _load_config() -> tuple[dict[str, Any], str]:
    kafka_cfg = yaml.safe_load((ROOT / "config/kafka.yaml").read_text(encoding="utf-8")) or {}
    storage_cfg = yaml.safe_load((ROOT / "config/storage.yaml").read_text(encoding="utf-8")) or {}
    topic = str((storage_cfg.get("quality") or {}).get("dead_letter_topic")
                or kafka_cfg.get("dead_letter_topic", "electricity-dlq"))
    return kafka_cfg, topic


def main() -> None:
    kafka_cfg, topic = _load_config()
    bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS", kafka_cfg["bootstrap_servers"])
    backoff = 1.0

    while True:
        consumer = None
        try:
            consumer = KafkaConsumer(
                topic,
                bootstrap_servers=bootstrap,
                group_id="electricity-quality-events-v1",
                auto_offset_reset="earliest",
                enable_auto_commit=False,
                # Keep the consumer responsive while synchronous storage writes
                # happen; commits remain explicit and occur after durable writes.
                max_poll_records=50,
                max_poll_interval_ms=600_000,
                session_timeout_ms=30_000,
                heartbeat_interval_ms=10_000,
                request_timeout_ms=40_000,
            )
            LOG.info("Consuming DLQ topic '%s' from %s", topic, bootstrap)
            backoff = 1.0

            while True:
                batches = consumer.poll(timeout_ms=1000, max_records=50)
                if not batches:
                    continue

                committed: dict[TopicPartition, int] = {}
                for tp, messages in batches.items():
                    for message in sorted(messages, key=lambda item: item.offset):
                        try:
                            try:
                                payload = json.loads((message.value or b"").decode("utf-8"))
                                if not isinstance(payload, dict):
                                    raise ValueError("DLQ payload must be a JSON object")
                                event = _event_from_message(message, payload)
                            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                                event = {
                                    "station_id": "unknown",
                                    "event_timestamp": (
                                        datetime.fromtimestamp(message.timestamp / 1000, tz=timezone.utc)
                                        if message.timestamp is not None and message.timestamp >= 0
                                        else datetime.now(timezone.utc)
                                    ),
                                    "event_id": f"{message.topic}:{message.partition}:{message.offset}:invalid_dlq_json",
                                    "severity": "error",
                                    "rule_name": "invalid_dlq_json",
                                    "message": f"Malformed DLQ payload: {exc}"[:2000],
                                    "source_topic": message.topic,
                                    "source_partition": message.partition,
                                    "source_offset": message.offset,
                                }
                            # Only mark an offset complete after durable storage.
                            record_quality_event(event)
                            committed[tp] = message.offset + 1
                        except Exception:
                            LOG.exception(
                                "Could not persist DLQ record topic=%s partition=%s offset=%s; "
                                "leaving this offset uncommitted for retry.",
                                message.topic, message.partition, message.offset,
                            )
                            raise

                if committed:
                    consumer.commit({
                        tp: OffsetAndMetadata(offset, None)
                        for tp, offset in committed.items()
                    })
        except KeyboardInterrupt:
            LOG.info("Stopping quality event consumer")
            return
        except Exception:
            LOG.exception("DLQ consumer connection/processing failed; reconnecting in %.1fs", backoff)
            try:
                from src.storage.cassandra_store import close as close_cassandra
                close_cassandra()
            except Exception:
                LOG.debug("Cassandra close during reconnect failed", exc_info=True)
            time.sleep(backoff + random.uniform(0.0, min(0.5, backoff / 4)))
            backoff = min(backoff * 2, 30.0)
        finally:
            if consumer is not None:
                try:
                    consumer.close(autocommit=False)
                except Exception:
                    LOG.debug("Kafka consumer close failed", exc_info=True)


if __name__ == "__main__":
    main()
