from pathlib import Path
import tempfile

from src.ingestion.file_loader import load_station_csv


def test_custom_timestamp_column_format_and_timezone_to_utc():
    path = Path(tempfile.gettempdir()) / "electricity_station_timestamp_test.csv"
    path.write_text(
        "local_time,demand,temperature\n"
        "01-01-2026 00:00,100,10.0\n"
        "01-01-2026 00:15,110,10.2\n"
        "01-01-2026 00:30,105,10.1\n",
        encoding="utf-8",
    )
    try:
        frame = load_station_csv(
            path,
            interval_minutes=15,
            timestamp_column="local_time",
            timestamp_format="%d-%m-%Y %H:%M",
            source_timezone="Europe/Berlin",
        )
        assert len(frame) == 3
        assert str(frame.loc[0, "utc_timestamp"]) == "2025-12-31 23:00:00+00:00"
        assert list(frame.columns) == ["utc_timestamp", "demand", "temperature"]
    finally:
        path.unlink(missing_ok=True)


def test_auto_detect_columns_and_iso_utc_timestamps():
    path = Path(tempfile.gettempdir()) / "electricity_station_auto_timestamp_test.csv"
    path.write_text(
        "timestamp,load_mw,temp_c\n"
        "2026-01-01T00:00:00Z,100,10.0\n"
        "2026-01-01T00:15:00Z,110,10.2\n"
        "2026-01-01T00:30:00Z,105,10.1\n",
        encoding="utf-8",
    )
    try:
        frame = load_station_csv(path, interval_minutes=15, timestamp_format="auto")
        assert len(frame) == 3
        assert list(frame.columns) == ["utc_timestamp", "demand", "temperature"]
        assert frame.loc[0, "utc_timestamp"].isoformat() == "2026-01-01T00:00:00+00:00"
        assert frame.attrs["source_columns"] == {
            "timestamp": "timestamp", "demand": "load_mw", "temperature": "temp_c"
        }
    finally:
        path.unlink(missing_ok=True)


def test_selected_format_falls_back_to_auto_for_iso_values():
    path = Path(tempfile.gettempdir()) / "electricity_station_timestamp_fallback_test.csv"
    path.write_text(
        "utc_timestamp,demand,temperature\n"
        "2026-01-01T00:00:00Z,100,10.0\n"
        "2026-01-01T00:15:00Z,110,10.2\n"
        "2026-01-01T00:30:00Z,105,10.1\n",
        encoding="utf-8",
    )
    try:
        frame = load_station_csv(
            path, interval_minutes=15, timestamp_format="%d-%m-%Y %H:%M"
        )
        assert len(frame) == 3
        assert "auto fallback" in frame.attrs["timestamp_format_used"]
    finally:
        path.unlink(missing_ok=True)


def test_auto_detects_european_day_first_format_without_column_mapping():
    path = Path(tempfile.gettempdir()) / "electricity_station_european_test.csv"
    path.write_text(
        "utc_timestamp,demand,temperature\n"
        "01-01-2026 00:00,100,10.0\n"
        "01-01-2026 00:15,110,10.2\n"
        "01-01-2026 00:30,105,10.1\n",
        encoding="utf-8",
    )
    try:
        frame = load_station_csv(path, interval_minutes=15, timestamp_format="auto")
        assert len(frame) == 3
        assert str(frame.loc[1, "utc_timestamp"]) == "2026-01-01 00:15:00+00:00"
    finally:
        path.unlink(missing_ok=True)
