"""Timestamp-aligned sums with explicit coverage for operational group monitoring."""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
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


def _parse_timestamp(value) -> pd.Timestamp | None:
    try:
        timestamp = pd.Timestamp(value)
        if pd.isna(timestamp):
            return None
        return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")
    except (ValueError, TypeError, OverflowError):
        return None


_cached_timestamp = lru_cache(maxsize=16384)(_parse_timestamp)


def _timestamp(value) -> pd.Timestamp | None:
    return _cached_timestamp(value) if isinstance(value, str) else _parse_timestamp(value)


def _number(value) -> float | None:
    try:
        value = float(value)
        return value if isfinite(value) and value >= 0 else None
    except (ValueError, TypeError):
        return None


def latest_demand_timestamp(rows: list[dict]) -> pd.Timestamp | None:
    return max((timestamp for row in rows
                if _number(row.get("actual_demand")) is not None
                and (timestamp := _timestamp(row.get("utc_timestamp"))) is not None), default=None)


def demand_to_mw_factor(entry: dict, interval: int) -> float:
    """Convert per-interval energy to average power for monitoring."""
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
    if origin_limit is None:
        return {}
    runs = defaultdict(dict)
    step_ns = step.value
    for row in rows:
        origin = _timestamp(row.get("forecast_origin_utc"))
        if origin is None or origin > origin_limit:
            continue
        target = _timestamp(row.get("forecast_timestamp_utc"))
        value = _number(row.get(FORECAST_VALUES[metric]))
        if target is None or value is None:
            continue
        lead, remainder = divmod(target.value - origin.value, step_ns)
        if remainder == 0 and 1 <= lead <= horizon:
            issue = _timestamp(row.get("forecast_issued_at_utc")) or origin
            tag = str(row.get("forecast_run_id") or issue)
            runs[(origin, issue, tag)][target] = value
    selected = {}
    for (origin, issue, tag), values in sorted(runs.items()):
        if len(values) == horizon:
            selected[origin] = values
    return selected


def _actual_points(data: LeafResults, factor: float, step_ns: int, offset_ns: int | None) -> dict:
    points: dict = defaultdict(dict)
    for row in data.demand:
        timestamp = _timestamp(row.get("utc_timestamp"))
        value = _number(row.get("actual_demand"))
        if timestamp is not None and value is not None and (
            offset_ns is None or timestamp.value % step_ns == offset_ns
        ):
            points[timestamp]["demand_mw"] = value * factor
    for row in data.generation:
        timestamp = _timestamp(row.get("utc_timestamp"))
        kind = row.get("generation_type")
        value = _number(row.get("actual_generation_mwh"))
        if timestamp is not None and kind in {"solar", "wind"} and value is not None and (
            offset_ns is None or timestamp.value % step_ns == offset_ns
        ):
            points[timestamp][f"{kind}_mwh"] = value
    return points


def roll_up(members: dict, inputs: dict[str, LeafResults], window: int, interval: int, horizon: int,
            *, as_of: pd.Timestamp | None = None, history: dict[str, LeafResults] | None = None) -> dict:
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
        factor = demand_to_mw_factor(entry, interval)
        points = _actual_points(data, factor, step_ns, offset_ns)
        demand_edges[station] = max((timestamp for timestamp, values in points.items() if "demand_mw" in values),
                                    default=None)
        if history and station in history:
            seed = _actual_points(history[station], factor, step_ns, offset_ns)
            for timestamp, values in points.items():
                seed[timestamp].update(values)
            points = seed
        leaf_actuals[station] = points
        component_origins[station] = {
            metric: _latest_runs(data.forecasts.get(metric, []), metric, horizon, step,
                                 min(as_of, demand_edges[station])
                                 if as_of is not None and demand_edges[station] is not None
                                 else demand_edges[station])
            for metric in METRICS[:3]
        }

    # Each metric gets its latest origin shared by every contributing station.
    # Residual load needs a single origin shared across all three inputs.
    shared_origins = {
        metric: set.intersection(*(set(runs[metric]) for runs in component_origins.values()))
        if component_origins else set()
        for metric in METRICS[:3]
    }
    shared_origins["residual_mwh"] = set.intersection(*shared_origins.values())
    common_origin = max(shared_origins["residual_mwh"], default=None)
    origins = {metric: common_origin or max(options, default=None) for metric, options in shared_origins.items()}
    forecast_totals: dict = defaultdict(dict)
    for station, runs in component_origins.items():
        factor = demand_to_mw_factor(members[station], interval)
        for metric in METRICS[:3]:
            for target, value in runs[metric].get(origins[metric], {}).items():
                _accumulate(forecast_totals[target], {metric: value * factor if metric == "demand_mw" else value})
        residual_origin = origins["residual_mwh"]
        if residual_origin is not None:
            for target, value in runs["demand_mw"][residual_origin].items():
                values = {"demand_mw": value * factor, "solar_mwh": runs["solar_mwh"][residual_origin][target],
                          "wind_mwh": runs["wind_mwh"][residual_origin][target]}
                _add_residual(values, interval)
                _accumulate(forecast_totals[target], {"residual_mwh": values["residual_mwh"]})

    observed = as_of if as_of is not None else max(
        (timestamp for points in leaf_actuals.values() for timestamp in points), default=None
    )
    actual_records = []
    if observed is not None:
        start = observed - step * (window - 1)
        actual_totals: dict = defaultdict(dict)
        for points in leaf_actuals.values():
            for timestamp, values in points.items():
                if start <= timestamp <= observed and (observed.value - timestamp.value) % step_ns == 0:
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
        "forecast_origin_utc": origins["demand_mw"].isoformat() if origins["demand_mw"] is not None else None,
        "forecast_origins_utc": {metric: origin.isoformat() if origin is not None else None
                                 for metric, origin in origins.items()},
    }
