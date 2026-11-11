from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.producers.csv_producer import (
    acquire_station_lock,
    iter_clean_chunks,
    iter_rows_after_progress,
    load_producer_progress,
    release_station_lock,
    save_producer_progress,
)


def _write_source(path: Path) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "utc_timestamp": pd.date_range(
                "2026-10-01T00:00:00Z", periods=6, freq="15min"
            ).astype(str),
            "demand": [100, 101, 102, 103, 104, 105],
            "temperature": [10, 10, 11, 11, 12, 12],
        }
    )
    frame.to_csv(path, index=False)
    return frame


def test_producer_persists_acknowledged_cursor_and_resumes_after_it(tmp_path):
    source = tmp_path / "demand_test.csv"
    frame = _write_source(source)
    progress = tmp_path / "state" / "producer_progress.json"
    cursor = pd.Timestamp(frame.loc[2, "utc_timestamp"])

    save_producer_progress(progress, "amprion", source, "amprion-raw", cursor)
    loaded = load_producer_progress(progress, "amprion", source, "amprion-raw")

    assert loaded == cursor
    chunks = iter_clean_chunks(source, chunksize=2)
    output = list(iter_rows_after_progress(chunks, loaded))
    assert [timestamp for _, timestamp in output] == [
        pd.Timestamp(value) for value in frame.loc[3:, "utc_timestamp"]
    ]


def test_producer_refuses_a_missing_saved_cursor(tmp_path):
    source = tmp_path / "demand_test.csv"
    _write_source(source)
    missing_cursor = pd.Timestamp("2026-09-30T23:45:00Z")

    with pytest.raises(RuntimeError, match="not present in the source CSV"):
        list(iter_rows_after_progress(iter_clean_chunks(source, 2), missing_cursor))


def test_producer_progress_rejects_a_changed_topic_unless_explicit_replay(tmp_path):
    source = tmp_path / "demand_test.csv"
    _write_source(source)
    progress = tmp_path / "producer_progress.json"
    save_producer_progress(
        progress, "amprion", source, "amprion-raw", pd.Timestamp("2026-10-01T00:15:00Z")
    )

    with pytest.raises(RuntimeError, match="configured topic"):
        load_producer_progress(progress, "amprion", source, "new-raw-topic")

    assert load_producer_progress(
        progress, "amprion", source, "new-raw-topic", replay=True
    ) is None
    assert not progress.exists()


def test_station_producer_lock_prevents_a_second_local_instance(tmp_path):
    lock_path = tmp_path / "amprion.lock"
    first = acquire_station_lock(lock_path)
    try:
        with pytest.raises(RuntimeError, match="already running"):
            acquire_station_lock(lock_path)
    finally:
        release_station_lock(first)

    # The file remains, but the OS lock is released and a new process may start.
    second = acquire_station_lock(lock_path)
    release_station_lock(second)
