import pandas as pd
from dashboard.model_metrics import (
    build_evaluation_frame,
    calculate_metrics,
    metrics_by_horizon,
    metrics_by_run,
)


def test_forecast_actual_matching_and_metrics():
    forecasts = pd.DataFrame([
        {"forecast_origin_utc": "2026-01-01T00:00:00Z", "forecast_timestamp_utc": ("2026-01-01T01:00:00Z" if m == 60 else f"2026-01-01T00:{m:02d}:00Z"), "forecast_issued_at_utc": "2026-01-01T00:00:01Z", "horizon_step": i, "predicted_demand": p, "model_file": "model.pt"}
        for i, (m, p) in enumerate([(15, 110), (30, 190), (45, 305), (60, 390)], start=1)
    ])
    actuals = pd.DataFrame([
        {"utc_timestamp": ("2026-01-01T01:00:00Z" if m == 60 else f"2026-01-01T00:{m:02d}:00Z"), "actual_demand": a}
        for m, a in [(15, 100), (30, 200), (45, 300), (60, 400)]
    ])
    evaluated = build_evaluation_frame(forecasts, actuals)
    assert len(evaluated) == 4
    score = calculate_metrics(evaluated)
    assert score["matched_observations"] == 4
    assert round(score["mae_mw"], 3) == 8.75
    assert round(score["bias_mw"], 3) == -1.25
    runs = metrics_by_run(evaluated, horizon_steps=4)
    assert len(runs) == 1 and bool(runs.iloc[0]["matured"])
    by_horizon = metrics_by_horizon(evaluated, runs)
    assert len(by_horizon) == 4


def test_unrealized_forecast_targets_are_excluded():
    forecasts = pd.DataFrame([
        {"forecast_origin_utc": "2026-01-01T00:00:00Z", "forecast_timestamp_utc": "2026-01-01T00:15:00Z", "horizon_step": 1, "predicted_demand": 110},
        {"forecast_origin_utc": "2026-01-01T00:00:00Z", "forecast_timestamp_utc": "2026-01-01T00:30:00Z", "horizon_step": 2, "predicted_demand": 120},
    ])
    actuals = pd.DataFrame([{"utc_timestamp": "2026-01-01T00:15:00Z", "actual_demand": 100}])
    evaluated = build_evaluation_frame(forecasts, actuals)
    assert len(evaluated) == 1
    assert evaluated.iloc[0]["horizon_step"] == 1
