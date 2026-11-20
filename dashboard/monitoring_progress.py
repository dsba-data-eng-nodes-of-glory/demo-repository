"""Determine when a monitoring view has observed a live advance."""
from __future__ import annotations

import pandas as pd


def utc(value) -> pd.Timestamp | None:
    try:
        stamp = pd.Timestamp(value)
    except (ValueError, TypeError, OverflowError):
        return None
    if pd.isna(stamp):
        return None
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def timeline_is_live(state: dict, key: tuple, cursor, run=None) -> bool:
    """Require a newer eligible timestamp on a later poll after initialization."""
    previous, ready, previous_run = state.get(key, (None, False, run))
    if previous_run != run:
        previous, ready = None, False
        state[key] = (None, False, run)
    stamp = utc(cursor)
    if stamp is None:
        # A missing result cannot be plotted. Keep the last known baseline so
        # a temporary partial read does not reset a live chart.
        return False
    if previous is None or stamp < previous:
        ready = False
    elif stamp > previous:
        ready = True
    state[key] = (stamp, ready, run)
    return ready


def eligible_forecast(origin, live_through, replay_start=None) -> pd.Timestamp | None:
    """Accept only forecast origins reached by the operational actual series."""
    origin, observed, start = utc(origin), utc(live_through), utc(replay_start)
    if origin is None or observed is None or origin > observed:
        return None
    return origin if start is None or origin >= start else None


def forecast_cursor(records, live_through, replay_start, steps: int, interval: int) -> pd.Timestamp | None:
    """Confirm the latest supplied run spans all configured forecast steps."""
    if len(records) != steps:
        return None
    stamps = {utc(row.get("forecast_timestamp_utc") or row.get("timestamp_utc")) for row in records}
    origins = {utc(row.get("forecast_origin_utc")) for row in records}
    if len(origins) != 1 or None in stamps or len(stamps) != steps:
        return None
    origin = eligible_forecast(origins.pop(), live_through, replay_start)
    if origin is None:
        return None
    expected = pd.date_range(start=origin + pd.Timedelta(minutes=interval), periods=steps, freq=f"{interval}min")
    return origin if stamps == set(expected) else None
