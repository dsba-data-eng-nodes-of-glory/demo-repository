"""Selection helpers for complete forecast runs shown in monitoring."""
from __future__ import annotations

import pandas as pd


def select_latest_complete_forecast_run(
    forecasts: pd.DataFrame,
    *,
    horizon_steps: int = 96,
    interval_minutes: int = 15,
) -> pd.DataFrame:
    """Select the newest complete run; use newest partial run only as fallback.

    A run is complete only when it contains each horizon step exactly once and
    its target timestamps form a continuous interval. This prevents an
    interrupted/replayed partial append from replacing the last usable run.
    """
    if forecasts is None or forecasts.empty:
        return pd.DataFrame()
    frame = forecasts.copy()
    required = {"forecast_timestamp_utc", "predicted_demand"}
    if not required.issubset(frame.columns):
        return pd.DataFrame()

    frame["forecast_timestamp_utc"] = pd.to_datetime(
        frame["forecast_timestamp_utc"], utc=True, errors="coerce"
    )
    if "forecast_issued_at_utc" in frame.columns:
        frame["forecast_issued_at_utc"] = pd.to_datetime(
            frame["forecast_issued_at_utc"], utc=True, errors="coerce"
        )
    if "forecast_origin_utc" in frame.columns:
        frame["forecast_origin_utc"] = pd.to_datetime(
            frame["forecast_origin_utc"], utc=True, errors="coerce"
        )
    frame["predicted_demand"] = pd.to_numeric(frame["predicted_demand"], errors="coerce")
    frame = frame.dropna(subset=["forecast_timestamp_utc", "predicted_demand"])
    if frame.empty:
        return pd.DataFrame()

    group_columns = ["forecast_run_id"] if "forecast_run_id" in frame.columns else []
    if not group_columns:
        group_columns = [
            col for col in ("forecast_issued_at_utc", "forecast_origin_utc")
            if col in frame.columns
        ]
    if not group_columns:
        return frame.sort_values("forecast_timestamp_utc").tail(horizon_steps).reset_index(drop=True)

    candidates: list[tuple[pd.Timestamp, pd.Timestamp, pd.DataFrame, bool]] = []
    expected_steps = set(range(1, int(horizon_steps) + 1))
    step = pd.Timedelta(minutes=int(interval_minutes))
    for key, group in frame.groupby(group_columns, dropna=False, sort=False):
        group = group.sort_values("forecast_timestamp_utc").drop_duplicates(
            "forecast_timestamp_utc", keep="last"
        )
        complete = False
        if "horizon_step" in group.columns:
            steps = pd.to_numeric(group["horizon_step"], errors="coerce").dropna().astype(int)
            times = pd.DatetimeIndex(group["forecast_timestamp_utc"])
            complete = (
                set(steps.tolist()) == expected_steps
                and len(steps) == horizon_steps
                and len(times) == horizon_steps
                and (len(times) < 2 or (times[1:] - times[:-1] == step).all())
            )
        issue = (
            group["forecast_issued_at_utc"].max()
            if "forecast_issued_at_utc" in group
            else pd.Timestamp.min.tz_localize("UTC")
        )
        origin = (
            group["forecast_origin_utc"].max()
            if "forecast_origin_utc" in group
            else pd.Timestamp.min.tz_localize("UTC")
        )
        candidates.append((origin, issue, group, bool(complete)))

    if not candidates:
        return pd.DataFrame()

    # A predictor restart can append a newly issued startup forecast whose
    # origin is older than an already-produced replay forecast.  Issue-time
    # ordering alone therefore selects stale data.  Prefer the newest
    # *complete* origin; use issue time only to break ties at the same origin.
    complete_candidates = [item for item in candidates if item[3]]
    pool = complete_candidates or candidates
    _, _, selected, _ = max(pool, key=lambda item: (item[0], item[1]))
    return selected.sort_values("forecast_timestamp_utc").reset_index(drop=True)
