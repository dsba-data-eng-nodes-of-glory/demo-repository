"""Replay-aware actual-data display boundaries for the monitoring dashboard."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def read_last_acked_timestamp(project_root: str | Path, station: str) -> pd.Timestamp | None:
    """Read the producer's last Kafka-acknowledged source timestamp.

    The producer progress file is the replay clock. Static test CSV rows after
    this timestamp must not be presented as already-realised observations.
    """
    progress_path = Path(project_root) / "data" / "state" / str(station) / "producer_progress.json"
    try:
        with progress_path.open("r", encoding="utf-8") as handle:
            state = json.load(handle)
    except (OSError, ValueError, TypeError):
        return None

    recorded_station = state.get("station_id")
    if recorded_station and str(recorded_station) != str(station):
        return None

    value = state.get("last_acked_timestamp")
    if not value:
        return None
    timestamp = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(timestamp):
        return None
    return pd.Timestamp(timestamp)


def resolve_actuals_display_cutoff(
    expected_stream_start_utc,
    interval_minutes: int,
    replay_timestamp=None,
    forecast_origin=None,
) -> tuple[pd.Timestamp | None, str]:
    """Return the latest actual timestamp allowed on the monitoring graphs.

    Priority:
      1. A saved forecast origin, when a demand forecast exists.
      2. The producer's last acknowledged timestamp during replay.
      3. One interval before the configured stream start before replay begins.

    The last case deliberately shows the historical seed but none of the
    future/test observations from the replay stream.
    """
    def parse(value) -> pd.Timestamp | None:
        if value is None or value == "":
            return None
        parsed = pd.to_datetime(value, utc=True, errors="coerce")
        return None if pd.isna(parsed) else pd.Timestamp(parsed)

    origin = parse(forecast_origin)
    if origin is not None:
        return origin, "forecast_origin"

    stream_start = parse(expected_stream_start_utc)
    replay = parse(replay_timestamp)
    if replay is not None and (stream_start is None or replay >= stream_start):
        return replay, "producer_progress"

    if stream_start is not None:
        interval = max(1, int(interval_minutes))
        return stream_start - pd.Timedelta(minutes=interval), "before_stream_start"

    return None, "unbounded_no_replay_boundary"


def filter_actuals_to_cutoff(
    frame: pd.DataFrame,
    cutoff: pd.Timestamp | None,
    timestamp_column: str = "utc_timestamp",
) -> pd.DataFrame:
    """Return only actual rows at or before the replay cutoff."""
    if frame.empty or cutoff is None or timestamp_column not in frame.columns:
        return frame.copy()
    timestamps = pd.to_datetime(frame[timestamp_column], utc=True, errors="coerce")
    return frame.loc[timestamps <= pd.Timestamp(cutoff)].copy()
