from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest
from kafka.structs import TopicPartition

from src.models.predictor_recovery import PredictorRecovery, contiguous_history, recovery_event, restore_history


def event(number: int) -> dict:
    timestamp = pd.Timestamp("2026-10-01", tz="UTC") + pd.Timedelta(minutes=15 * number)
    return {
        "station_id": "amprion", "utc_timestamp": timestamp, "demand": 4000 + number,
        "temperature": None, "solar_mwh": 12.0, "solar_capacity_mw": 100.0,
        "wind_mwh": 8.0, "wind_capacity_mw": 80.0, "partition": 0, "offset": number,
    }


def checkpoint(tmp_path, capacity=672, group="predictor-amprion"):
    return PredictorRecovery(
        tmp_path / "predictor_recovery.sqlite3", station="amprion", topic="amprion-featured",
        group=group, interval_minutes=15, capacity=capacity,
    )


def test_reopens_only_latest_rows_and_retains_committed_position(tmp_path):
    state = checkpoint(tmp_path)
    state.save([event(i) for i in range(700)], {0: 700})
    reopened = checkpoint(tmp_path)
    reopened.save([dict(event(699), demand=99999)], {0: 699})
    rows, offsets = reopened.load()
    assert offsets == {0: 700}
    assert len(rows) == 672
    assert rows == [event(i) for i in range(28, 700)]
    with pytest.raises(RuntimeError, match="different Kafka topic, group or interval"):
        checkpoint(tmp_path, group="different-group")


def test_saved_rows_and_offsets_roll_back_together(tmp_path):
    state = checkpoint(tmp_path)
    state.save([event(0)], {0: 1})
    with pytest.raises(ValueError):
        state.save([event(1), dict(event(2), demand=float("nan"))], {0: 3})
    assert state.load() == ([event(0)], {0: 1})


def test_late_reading_repairs_gap_without_creating_actuals():
    minutes = pd.Timedelta(minutes=15)
    seed = [event(i) for i in range(4)]
    assert contiguous_history(seed, [event(5)], minutes, 4) == [event(5)]
    assert contiguous_history(seed, [event(5), event(4)], minutes, 4) == [event(i) for i in range(2, 6)]
    row = recovery_event(event(7), {"solar_mwh": "12", "wind_mwh": float("inf")},
                         SimpleNamespace(partition=1, offset=19), replay_weather=False)
    assert row["solar_mwh"] == 12 and row["partition"] == 1 and row["offset"] == 19


class CommittedKafka:
    def __init__(self, committed):
        self.offset = committed
        self.partition = TopicPartition("amprion-featured", 0)
        self.cursor = 0
        self.unsubscribed = False
        self.polled = 0

    def partitions_for_topic(self, _topic):
        return {0}

    def beginning_offsets(self, _partitions):
        return {self.partition: 0}

    def end_offsets(self, _partitions):
        return {self.partition: 750}

    def committed(self, _partition):
        return self.offset

    def assign(self, partitions):
        assert partitions == [self.partition]

    def seek(self, _partition, offset):
        self.cursor = offset

    def position(self, _partition):
        return self.cursor

    def poll(self, **options):
        self.polled += 1
        limit = min(750, self.cursor + options["max_records"])
        messages = []
        for number in range(self.cursor, limit):
            value = json.dumps(event(number), default=str).encode()
            messages.append(SimpleNamespace(value=value, offset=number, partition=0))
        self.cursor = limit
        return {self.partition: messages}

    def pause(self, _partition):
        pass

    def unsubscribe(self):
        self.unsubscribed = True


def restore(consumer, state):
    return restore_history(consumer, state, "amprion-featured", station="amprion",
                           interval_minutes=15, require_temperature=False, default_replay=False)


def test_migrates_existing_kafka_group_from_committed_history_only(tmp_path):
    broker = CommittedKafka(committed=720)
    rows, positions = restore(broker, checkpoint(tmp_path))
    assert broker.unsubscribed
    assert positions == {0: 720}
    assert len(rows) == 672 and rows[0]["offset"] == 48 and rows[-1]["offset"] == 719
    assert rows[-1]["wind_capacity_mw"] == 80.0


def test_local_checkpoint_ahead_of_kafka_is_reused_without_backfill(tmp_path):
    state = checkpoint(tmp_path)
    state.save([event(i) for i in range(700)], {0: 700})
    broker = CommittedKafka(committed=699)
    rows, positions = restore(broker, state)
    assert not broker.polled
    assert positions == {0: 700} and rows[-1]["offset"] == 699
