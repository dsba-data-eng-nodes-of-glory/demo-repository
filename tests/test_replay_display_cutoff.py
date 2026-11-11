import json

import pandas as pd

from dashboard.replay_display import (
    filter_actuals_to_cutoff,
    read_last_acked_timestamp,
    resolve_actuals_display_cutoff,
)


def test_initial_state_stops_actuals_immediately_before_configured_stream_start():
    cutoff, source = resolve_actuals_display_cutoff(
        "2026-10-01T00:00:00Z", 15, replay_timestamp=None, forecast_origin=None
    )
    assert cutoff == pd.Timestamp("2026-09-30T23:45:00Z")
    assert source == "before_stream_start"


def test_replay_progress_advances_the_display_cutoff():
    cutoff, source = resolve_actuals_display_cutoff(
        "2026-10-01T00:00:00Z", 15,
        replay_timestamp="2026-10-08T19:15:00+00:00",
    )
    assert cutoff == pd.Timestamp("2026-10-08T19:15:00Z")
    assert source == "producer_progress"


def test_stale_progress_before_stream_start_does_not_reveal_test_rows():
    cutoff, source = resolve_actuals_display_cutoff(
        "2026-10-01T00:00:00Z", 15,
        replay_timestamp="2026-09-30T12:00:00Z",
    )
    assert cutoff == pd.Timestamp("2026-09-30T23:45:00Z")
    assert source == "before_stream_start"


def test_forecast_origin_remains_the_cutoff_when_forecast_exists():
    cutoff, source = resolve_actuals_display_cutoff(
        "2026-10-01T00:00:00Z", 15,
        replay_timestamp="2026-10-08T19:15:00Z",
        forecast_origin="2026-10-08T18:45:00Z",
    )
    assert cutoff == pd.Timestamp("2026-10-08T18:45:00Z")
    assert source == "forecast_origin"


def test_bad_progress_file_or_wrong_station_is_ignored(tmp_path):
    progress = tmp_path / "data" / "state" / "amprion" / "producer_progress.json"
    progress.parent.mkdir(parents=True)
    progress.write_text(json.dumps({
        "station_id": "different_station",
        "last_acked_timestamp": "2026-10-08T19:15:00Z",
    }), encoding="utf-8")
    assert read_last_acked_timestamp(tmp_path, "amprion") is None


def test_progress_file_timestamp_is_read(tmp_path):
    progress = tmp_path / "data" / "state" / "amprion" / "producer_progress.json"
    progress.parent.mkdir(parents=True)
    progress.write_text(json.dumps({
        "station_id": "amprion",
        "last_acked_timestamp": "2026-10-08T19:15:00+00:00",
    }), encoding="utf-8")
    assert read_last_acked_timestamp(tmp_path, "amprion") == pd.Timestamp(
        "2026-10-08T19:15:00Z"
    )


def test_actual_frame_is_filtered_without_changing_forecast_related_data():
    frame = pd.DataFrame({
        "utc_timestamp": pd.to_datetime([
            "2026-09-30T23:45:00Z", "2026-10-01T00:00:00Z", "2026-10-02T00:00:00Z"
        ], utc=True),
        "solar_actual_mwh": [1.0, 2.0, 3.0],
    })
    filtered = filter_actuals_to_cutoff(frame, pd.Timestamp("2026-09-30T23:45:00Z"))
    assert len(filtered) == 1
    assert filtered.iloc[0]["solar_actual_mwh"] == 1.0
    assert len(frame) == 3  # The source frame is not mutated.
