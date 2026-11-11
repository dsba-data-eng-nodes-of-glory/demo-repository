import pandas as pd

from src.quality.validator import validate_event


def _recent_utc_timestamp(minutes_ago: int = 15) -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC").floor("15min") - pd.Timedelta(minutes=minutes_ago)


def test_valid_event_is_normalized_to_utc():
    expected = _recent_utc_timestamp(15)
    local_timestamp = expected.tz_convert("Etc/GMT-1").isoformat()
    event, issues = validate_event(
        {
            "station_id": "tennet",
            "utc_timestamp": local_timestamp,
            "demand": 42000,
            "temperature": 8.5,
        },
        expected_station="tennet",
        interval_minutes=15,
    )
    assert issues == []
    assert event["utc_timestamp"] == expected
    assert event["demand"] == 42000.0


def test_negative_demand_is_rejected():
    event, issues = validate_event(
        {
            "station_id": "tennet",
            "utc_timestamp": _recent_utc_timestamp(15).isoformat(),
            "demand": -1,
            "temperature": 8,
        },
        expected_station="tennet",
        interval_minutes=15,
    )
    assert event is None
    assert issues[0]["rule_name"] == "negative_demand"
    assert issues[0]["severity"] == "error"


def test_timestamp_gap_is_warning_not_invalid_record():
    target = _recent_utc_timestamp(0)
    previous = target - pd.Timedelta(minutes=30)
    event, issues = validate_event(
        {
            "station_id": "tennet",
            "utc_timestamp": target.isoformat(),
            "demand": 42000,
            "temperature": 8,
        },
        expected_station="tennet",
        interval_minutes=15,
        previous_timestamp=previous,
    )
    assert event is not None
    assert issues == []  # Predictor records the gap and resets its rolling buffer.


def test_wrong_station_is_rejected():
    event, issues = validate_event(
        {
            "station_id": "amprion",
            "utc_timestamp": _recent_utc_timestamp(15).isoformat(),
            "demand": 42000,
            "temperature": 8,
        },
        expected_station="tennet",
        interval_minutes=15,
    )
    assert event is None
    assert issues[0]["rule_name"] == "station_mismatch"


def test_far_future_timestamp_is_rejected():
    event, issues = validate_event(
        {
            "station_id": "tennet",
            "utc_timestamp": "2099-01-01T00:00:00Z",
            "demand": 42000,
            "temperature": 8,
        },
        expected_station="tennet",
        interval_minutes=15,
        max_future_skew_minutes=5,
    )
    assert event is None
    assert issues[0]["rule_name"] == "future_timestamp"
