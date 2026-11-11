"""Reusable validation for normalized Kafka electricity events."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import math
import pandas as pd


def validate_event(
    payload: dict[str, Any],
    expected_station: str,
    interval_minutes: int,
    previous_timestamp: pd.Timestamp | None = None,
    max_future_skew_minutes: int = 5,
) -> tuple[dict[str, Any] | None, list[dict[str, str]]]:
    """Return a normalized event and quality issues; never silently discard data."""
    issues: list[dict[str, str]] = []
    station = payload.get("station_id")
    if station != expected_station:
        issues.append({"rule_name": "station_mismatch", "severity": "error", "message": f"Expected station {expected_station}, received {station}."})

    raw_ts = payload.get("utc_timestamp") or payload.get("timestamp")
    ts = pd.to_datetime(raw_ts, utc=True, errors="coerce")
    if pd.isna(ts):
        issues.append({"rule_name": "invalid_timestamp", "severity": "error", "message": "Timestamp is missing or cannot be parsed."})
    elif ts > pd.Timestamp.now(tz="UTC") + pd.Timedelta(minutes=max(0, int(max_future_skew_minutes))):
        issues.append({"rule_name": "future_timestamp", "severity": "error", "message": f"Timestamp is more than {max_future_skew_minutes} minutes ahead of current UTC time."})

    try:
        demand = float(payload.get("demand"))
    except (TypeError, ValueError):
        demand = float("nan")
    try:
        temperature = float(payload.get("temperature"))
    except (TypeError, ValueError):
        temperature = float("nan")

    if not math.isfinite(demand):
        issues.append({"rule_name": "invalid_demand", "severity": "error", "message": "Demand is missing or non-finite."})
    elif demand < 0:
        issues.append({"rule_name": "negative_demand", "severity": "error", "message": "Demand cannot be negative."})
    if not math.isfinite(temperature):
        issues.append({"rule_name": "invalid_temperature", "severity": "error", "message": "Temperature is missing or non-finite."})

    if not pd.isna(ts) and previous_timestamp is not None:
        delta = ts - previous_timestamp
        expected = pd.Timedelta(minutes=interval_minutes)
        if delta <= pd.Timedelta(0):
            issues.append({"rule_name": "non_increasing_timestamp", "severity": "warning", "message": "Timestamp is not later than the previous accepted observation."})
        elif delta != expected:
            issues.append({"rule_name": "interval_gap", "severity": "warning", "message": f"Expected interval {expected}; observed {delta}."})

    fatal_issues = [issue for issue in issues if issue["severity"] == "error"]
    if fatal_issues:
        return None, fatal_issues
    return {
        "station_id": expected_station,
        "utc_timestamp": pd.Timestamp(ts),
        "demand": demand,
        "temperature": temperature,
    }, []


def quality_event(
    station: str,
    issue: dict[str, str],
    topic: str,
    partition: int,
    offset: int,
    event_timestamp: datetime | None = None,
) -> dict[str, Any]:
    stable_timestamp = event_timestamp or datetime.now(timezone.utc)
    if stable_timestamp.tzinfo is None:
        stable_timestamp = stable_timestamp.replace(tzinfo=timezone.utc)
    else:
        stable_timestamp = stable_timestamp.astimezone(timezone.utc)
    return {
        "station_id": station,
        "event_timestamp": stable_timestamp,
        "event_id": f"{topic}:{partition}:{offset}:{issue['rule_name']}",
        "severity": issue["severity"],
        "rule_name": issue["rule_name"],
        "message": issue["message"],
        "source_topic": topic,
        "source_partition": partition,
        "source_offset": offset,
    }
