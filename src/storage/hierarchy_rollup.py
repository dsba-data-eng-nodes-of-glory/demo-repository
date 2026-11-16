"""Timestamp-aligned sums with explicit coverage for operational group monitoring."""
from __future__ import annotations

from collections import defaultdict
from math import isfinite
from typing import NamedTuple

import pandas as pd

METRICS = ("demand_mw", "solar_mwh", "wind_mwh", "residual_mwh")
FORECAST_VALUES = {"demand_mw": "predicted_demand", "solar_mwh": "predicted_generation_mwh",
                   "wind_mwh": "predicted_generation_mwh"}


class LeafResults(NamedTuple):
    demand: list[dict]
    generation: list[dict]
    forecasts: dict[str, list[dict]]
    recovery: dict | None = None


def _timestamp(value) -> pd.Timestamp | None:
    try:
        timestamp = pd.Timestamp(value)
        if pd.isna(timestamp):
            return None
        return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")
    except (ValueError, TypeError, OverflowError):
        return None


def _number(value) -> float | None:
    try:
        value = float(value)
        return value if isfinite(value) and value >= 0 else None
    except (ValueError, TypeError):
        return None


def _multiplier(entry: dict, interval: int) -> float:
    return 60 / interval if entry.get("residual_demand_unit") == "mwh_per_interval" else 1.0


def _add_residual(values: dict, interval: int) -> None:
    if all(metric in values for metric in METRICS[:3]):
        values["residual_mwh"] = values["demand_mw"] * interval / 60 - values["solar_mwh"] - values["wind_mwh"]


def _point(timestamp: pd.Timestamp, totals: dict, count: int) -> dict:
    return {
        "utc_timestamp": timestamp.isoformat(),
        **{metric: {"value": totals[metric][0], "reporting": totals[metric][1], "expected": count}
           if metric in totals else {"value": None, "reporting": 0, "expected": count}
           for metric in METRICS},
    }


def _accumulate(total: dict, values: dict) -> None:
    for metric, value in values.items():
        old_value, reporting = total.get(metric, (0.0, 0))
        total[metric] = (old_value + value, reporting + 1)


def _latest_runs(rows: list[dict], metric: str, horizon: int,
                 step: pd.Timedelta, origin_limit: pd.Timestamp | None) -> dict:
    """Choose a single complete model run at each origin. Never join truncated runs."""
    runs = defaultdict(dict)
    for row in rows:
        origin = _timestamp(row.get("forecast_origin_utc"))
        target = _timestamp(row.get("forecast_timestamp_utc"))
        value = _number(row.get(FORECAST_VALUES[metric]))
        if origin is None or target is None or value is None or origin_limit is None or origin > origin_limit:
            continue
        lead, remainder = divmod(target - origin, step)
        if remainder == pd.Timedelta(0) and 1 <= lead <= horizon:
            issue = _timestamp(row.get("forecast_issued_at_utc")) or origin
            tag = str(row.get("forecast_run_id") or issue)
            runs[(origin, issue, tag)][target] = value
    selected = {}
    for (origin, issue, tag), values in sorted(runs.items()):
        if len(values) == horizon:
            selected[origin] = values
    return selected


def roll_up(members: dict, inputs: dict[str, LeafResults], window: int, interval: int, horizon: int) -> dict:
    """Use only saved observations and add each leaf once per timestamp and metric."""
    step = pd.Timedelta(minutes=interval)
    offset_ns = next((int(entry["hierarchy_grid_offset_seconds"]) * 10**9
                      for entry in members.values() if entry.get("hierarchy_grid_offset_seconds") is not None), None)
    step_ns = step.value
    leaf_actuals = {}
    demand_edges = {}
    component_origins = {}
    for station, entry in members.items():
        data = inputs.get(station, LeafResults([], [], {}))
        points: dict = defaultdict(dict)
        factor = _multiplier(entry, interval)
        for row in data.demand:
            timestamp = _timestamp(row.get("utc_timestamp"))
            value = _number(row.get("actual_demand"))
            if timestamp is not None and value is not None and (
                offset_ns is None or timestamp.value % step_ns == offset_ns
            ):
                points[timestamp]["demand_mw"] = value * factor
        demand_edges[station] = max(points, default=None)
        for row in data.generation:
            timestamp = _timestamp(row.get("utc_timestamp"))
            kind = row.get("generation_type")
            value = _number(row.get("actual_generation_mwh"))
            if timestamp is not None and kind in {"solar", "wind"} and value is not None and (
                offset_ns is None or timestamp.value % step_ns == offset_ns
            ):
                points[timestamp][f"{kind}_mwh"] = value
        leaf_actuals[station] = points
        component_origins[station] = {
            metric: _latest_runs(data.forecasts.get(metric, []), metric, horizon, step, demand_edges[station])
            for metric in METRICS[:3]
        }

    counts: dict = defaultdict(int)
    for runs in component_origins.values():
        for origin in runs["demand_mw"]:
            counts[origin] += 1
    # Prefer full demand coverage, and still provide a flagged partial run
    # when the bounded window has no common origin.
    shared = [origin for origin, count in counts.items() if count == len(members)]
    origin = max(shared or counts, default=None)
    forecast_totals: dict = defaultdict(dict)
    if origin is not None:
        for station, runs in component_origins.items():
            factor = _multiplier(members[station], interval)
            leaf_points: dict = defaultdict(dict)
            for metric, forecasts in runs.items():
                for target, value in forecasts.get(origin, {}).items():
                    leaf_points[target][metric] = value * factor if metric == "demand_mw" else value
            for target, values in leaf_points.items():
                _add_residual(values, interval)
                _accumulate(forecast_totals[target], values)

    observed = max((timestamp for points in leaf_actuals.values() for timestamp in points), default=None)
    actual_records = []
    if observed is not None:
        start = observed - step * (window - 1)
        actual_totals: dict = defaultdict(dict)
        for points in leaf_actuals.values():
            for timestamp, values in points.items():
                if timestamp >= start and (observed - timestamp) % step == pd.Timedelta(0):
                    _add_residual(values, interval)
                    _accumulate(actual_totals[timestamp], values)
        earliest = min(actual_totals, default=observed)
        actual_records = [_point(timestamp, actual_totals.get(timestamp, {}), len(members))
                          for timestamp in pd.date_range(earliest, observed, freq=step)]
    return {
        "actuals": actual_records,
        "latest_demand_utc": {station: edge.isoformat() if edge is not None else None
                              for station, edge in demand_edges.items()},
        "forecasts": [_point(timestamp, totals, len(members)) for timestamp, totals in sorted(forecast_totals.items())],
        "forecast_origin_utc": origin.isoformat() if origin is not None else None,
    }
