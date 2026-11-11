from __future__ import annotations

import pandas as pd

from dashboard.forecast_selection import select_latest_complete_forecast_run


def _run(run_id: str, origin: str, issue: str, steps: int) -> list[dict]:
    origin_ts = pd.Timestamp(origin)
    issue_ts = pd.Timestamp(issue)
    records = []
    for step in range(1, steps + 1):
        target = origin_ts + pd.Timedelta(minutes=15 * step)
        records.append({
            "forecast_run_id": run_id,
            "forecast_issued_at_utc": issue_ts,
            "forecast_origin_utc": origin_ts,
            "forecast_timestamp_utc": target,
            "horizon_step": step,
            "predicted_demand": 4000 + step,
        })
    return records


def test_latest_complete_run_wins_over_newer_partial_append():
    older_complete = _run(
        "complete", "2026-10-01T00:00:00Z", "2026-10-01T00:01:00Z", 96
    )
    newer_partial = _run(
        "partial", "2026-10-01T00:15:00Z", "2026-10-01T00:16:00Z", 31
    )
    selected = select_latest_complete_forecast_run(
        pd.DataFrame(older_complete + newer_partial),
        horizon_steps=96, interval_minutes=15,
    )
    assert selected["forecast_run_id"].eq("complete").all()
    assert len(selected) == 96
    assert selected["horizon_step"].min() == 1
    assert selected["horizon_step"].max() == 96


def test_latest_origin_wins_when_restart_appends_newer_issue_for_old_origin():
    progressed_run = _run(
        "progressed", "2026-10-02T12:00:00Z", "2026-11-09T18:00:00Z", 96
    )
    restarted_initial_run = _run(
        "restart-initial", "2026-10-01T00:00:00Z", "2026-11-09T23:00:00Z", 96
    )
    selected = select_latest_complete_forecast_run(
        pd.DataFrame(progressed_run + restarted_initial_run),
        horizon_steps=96, interval_minutes=15,
    )
    assert selected["forecast_run_id"].eq("progressed").all()
    assert len(selected) == 96


def test_newer_partial_origin_does_not_replace_older_complete_run():
    older_complete = _run(
        "complete", "2026-10-01T00:00:00Z", "2026-11-09T18:00:00Z", 96
    )
    newer_partial = _run(
        "partial", "2026-10-02T00:00:00Z", "2026-11-09T23:00:00Z", 24
    )
    selected = select_latest_complete_forecast_run(
        pd.DataFrame(older_complete + newer_partial),
        horizon_steps=96, interval_minutes=15,
    )
    assert selected["forecast_run_id"].eq("complete").all()
    assert len(selected) == 96
