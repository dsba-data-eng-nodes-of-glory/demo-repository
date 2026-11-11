"""No UI runtime required: verify deterministic snapshots and point-in-time matching."""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import pytest

from src.analytics.renewable_performance import match_renewable_forecasts, renewable_metrics


def test_renewable_matching_excludes_future_observations_and_wrong_lead():
    targets = pd.date_range("2026-10-01T01:00:00Z", periods=4, freq="15min")
    forecast = pd.DataFrame([
        {"generation_type": "solar", "forecast_timestamp_utc": t, "forecast_origin_utc": t - pd.Timedelta(minutes=15),
         "horizon_step": 1, "predicted_generation_mwh": v}
        for t, v in zip(targets, [10, 12, 14, 16])
    ] + [
        {"generation_type": "solar", "forecast_timestamp_utc": targets[0], "forecast_origin_utc": targets[0] - pd.Timedelta(hours=1),
         "horizon_step": 4, "predicted_generation_mwh": 5000},
        {"generation_type": "wind", "forecast_timestamp_utc": targets[0], "forecast_origin_utc": targets[0] - pd.Timedelta(minutes=15),
         "horizon_step": 1, "predicted_generation_mwh": 300},
    ])
    observed = pd.DataFrame({"utc_timestamp": targets,
                             "solar_actual_mwh": [9, 13, 15, 10000],
                             "wind_actual_mwh": [299, 300, 301, 302]})
    matched = match_renewable_forecasts(forecast, observed, "solar", targets[2], horizon_step=1)
    assert len(matched) == 3
    assert matched["predicted_mwh"].tolist() == [10, 12, 14]
    assert matched["actual_mwh"].tolist() == [9, 13, 15]
    assert renewable_metrics(matched)["mae_mwh"] == pytest.approx(1)
    assert match_renewable_forecasts(forecast, observed, "solar", None).empty
    assert len(match_renewable_forecasts(forecast, observed, "solar", targets[2], horizon_step=4)) == 1


def test_renewable_replay_wall_clock_issued_time_does_not_hide_valid_predictions():
    target = pd.Timestamp("2026-10-01T01:15:00Z")
    frame = pd.DataFrame([
        {"generation_type": "wind", "forecast_timestamp_utc": target,
         "forecast_origin_utc": target - pd.Timedelta(minutes=15),
         "forecast_issued_at_utc": target + pd.Timedelta(minutes=1),
         "horizon_step": 1, "predicted_generation_mwh": 100},
    ])
    actual = pd.DataFrame({"utc_timestamp": [target], "wind_actual_mwh": [99]})
    assert len(match_renewable_forecasts(frame, actual, "wind", target)) == 1


def test_results_endpoint_uses_actual_stream_watermark(monkeypatch, tmp_path):
    from backend import results_api as api
    forecast = pd.DataFrame({
        "generation_type": ["solar", "solar"],
        "forecast_origin_utc": ["2026-10-01T01:00:00Z", "2026-10-01T01:15:00Z"],
        "forecast_timestamp_utc": ["2026-10-01T01:15:00Z", "2026-10-01T01:30:00Z"],
        "predicted_generation_mwh": [10., 20.],
        "horizon_step": [1, 1],
    })
    actual = pd.DataFrame({"utc_timestamp": ["2026-10-01T01:15:00Z", "2026-10-01T01:30:00Z"],
                           "solar_actual_mwh": [9., 500.], "wind_actual_mwh": [100, 200]})
    monkeypatch.setattr(api, "_station_configuration", lambda station: {"renewables": {"enabled": True}})
    monkeypatch.setattr(api, "_actual_watermark", lambda station: pd.Timestamp("2026-10-01T01:15:00Z"))
    monkeypatch.setattr(api, "_renewable_file", lambda station, cfg: tmp_path / "forecast.csv")
    monkeypatch.setattr(api, "_read_csv_tail", lambda path, n, ttl_seconds=1.0: forecast)
    monkeypatch.setattr(api, "_load_configured_renewable_actuals", lambda station, end_time=None: (actual, {}))
    result = api.renewables_performance("amprion", horizon_step=1, limit=50)
    assert result["models"]["solar"]["metrics"]["matched_points"] == 1
    assert result["models"]["solar"]["metrics"]["mae_mwh"] == pytest.approx(1)
    assert result["watermark_utc"].startswith("2026-10-01T01:15")
    monkeypatch.setattr(api, "_actual_watermark", lambda station: None)
    assert api.renewables_performance("amprion", horizon_step=1, limit=50)["models"]["solar"]["metrics"]["matched_points"] == 0


