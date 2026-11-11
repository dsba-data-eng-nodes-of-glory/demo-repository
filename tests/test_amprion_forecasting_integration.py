from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np
import pytest
import torch
import yaml

from src.models.forecast_weather import (
    ForecastWeatherUnavailable,
    load_weather_lookup,
    select_forecast_horizon,
)

ROOT = Path(__file__).resolve().parents[1]



def test_amprion_replay_uses_october_holdout_and_training_tail_for_history():
    station_cfg = yaml.safe_load((ROOT / "config/stations.yaml").read_text())["stations"]["amprion"]
    assert station_cfg["input_file"] == "data/training/amprion/demand_test.csv"
    assert station_cfg["history_file"] == "data/training/amprion/demand_train.csv"
    assert station_cfg["expected_stream_start_utc"] == "2026-10-01T00:00:00Z"

    history_path = ROOT / station_cfg["history_file"]
    test_path = ROOT / station_cfg["input_file"]
    history = pd.read_csv(history_path, usecols=["utc_timestamp"])
    history_times = pd.to_datetime(
        history["utc_timestamp"], format="mixed", dayfirst=True, utc=True, errors="coerce"
    ).dropna()
    history_end = history_times.max()
    test = pd.read_csv(test_path, usecols=["utc_timestamp"])
    test_times = pd.to_datetime(
        test["utc_timestamp"], format="mixed", dayfirst=True, utc=True, errors="coerce"
    ).dropna()
    assert history_end == pd.Timestamp("2026-09-30T23:45:00Z")
    assert test_times.min() == pd.Timestamp("2026-10-01T00:00:00Z")
    assert test_times.diff().dropna().eq(pd.Timedelta(minutes=15)).all()


def test_windows_replay_launcher_uses_october_test_and_training_files():
    script = (ROOT / "full_replay_amprion.bat").read_text(encoding="utf-8")
    assert "data\\training\\amprion\\demand_test.csv" in script
    assert "data\\training\\amprion\\demand_train.csv" in script
    assert "renewable_forecasts.csv" in script
    assert "2026-10-01 00:00 UTC" in script
    assert "data\\raw\\amprion.csv" not in script


def test_historical_actual_api_uses_configured_history_file_for_first_forecast_origin():
    from backend.results_api import _read_historical_actuals, _combine_actuals

    origin = pd.Timestamp("2026-09-30T23:45:00Z")
    historical = _read_historical_actuals("amprion")
    assert not historical.empty
    timestamps = pd.to_datetime(
        historical["utc_timestamp"].astype("string").str.strip(),
        format="mixed", dayfirst=True, utc=True, errors="coerce",
    )
    assert timestamps.min() <= pd.Timestamp("2022-01-01T00:00:00Z")
    assert timestamps.max() >= pd.Timestamp("2026-10-19T23:45:00Z")

    context = _combine_actuals(
        historical, pd.DataFrame(), "amprion", 5000, end_time=origin.isoformat()
    )
    context_times = pd.to_datetime(context["utc_timestamp"], utc=True, errors="coerce")
    assert context_times.max() == origin
    assert context_times.min() <= origin - pd.Timedelta(hours=72)



def test_forecast_weather_lookup_parses_iso_and_requires_full_future_horizon(tmp_path):
    path = tmp_path / "forecast_weather.csv"
    pd.DataFrame({
        "utc_timestamp": [
            "2026-10-01T00:00:00Z",
            "2026-10-01T00:15:00Z",
            "2026-10-01T00:30:00Z",
        ],
        "forecast_temperature": [12.0, 12.5, 13.0],
    }).to_csv(path, index=False)
    lookup = load_weather_lookup(path, "utc_timestamp", ["forecast_temperature"])
    horizon = pd.date_range("2026-10-01T00:15:00Z", periods=2, freq="15min")
    selected = select_forecast_horizon(
        lookup, horizon, ["forecast_temperature"], context="test"
    )
    assert selected["forecast_temperature"].tolist() == [12.5, 13.0]
    with pytest.raises(ForecastWeatherUnavailable, match="No observed-weather substitution"):
        select_forecast_horizon(
            lookup, pd.date_range("2026-10-01T00:30:00Z", periods=2, freq="15min"),
            ["forecast_temperature"], context="test",
        )


def _demand_history_at(origin: pd.Timestamp) -> list[dict]:
    path = ROOT / "data/training/amprion/demand_train.csv"
    frame = pd.read_csv(path)
    frame["utc_timestamp"] = pd.to_datetime(
        frame["utc_timestamp"], format="mixed", dayfirst=True, utc=True, errors="coerce"
    )
    frame = frame.dropna(subset=["utc_timestamp"]).sort_values("utc_timestamp")
    history = frame[frame["utc_timestamp"] <= origin].tail(672)
    assert len(history) == 672
    return history[["utc_timestamp", "demand", "temperature"]].to_dict("records")


def test_amprion_demand_dense_serving_feature_and_forecast_contract():
    from src.models.demand_dense_forecaster import load_demand_dense_model, predict_demand

    station_cfg = yaml.safe_load((ROOT / "config/stations.yaml").read_text())[
        "stations"
    ]["amprion"]
    origin = pd.Timestamp("2026-09-30T23:45:00Z")
    rows = _demand_history_at(origin)
    future_ts = pd.date_range(origin + pd.Timedelta(minutes=15), periods=96, freq="15min")
    assert future_ts[0] == pd.Timestamp("2026-10-01T00:00:00Z")
    assert future_ts[-1] == pd.Timestamp("2026-10-01T23:45:00Z")
    model, checkpoint = load_demand_dense_model(
        ROOT / station_cfg["model_checkpoint"], "amprion", torch.device("cpu")
    )
    prediction = predict_demand(
        rows, model, checkpoint, torch.device("cpu"),
        ROOT / station_cfg["forecast_weather_file"], future_ts,
    )
    assert prediction.shape == (96,)
    assert (prediction >= 0).all()
    assert pd.Series(prediction).notna().all()


def test_amprion_solar_and_wind_models_produce_96_step_forecasts_with_holdout_weather():
    from src.models.renewable_forecaster import AmprionRenewableForecaster

    station_cfg = yaml.safe_load((ROOT / "config/stations.yaml").read_text())[
        "stations"]["amprion"]
    renewable_cfg = station_cfg["renewables"]
    assert (ROOT / renewable_cfg["solar"]["test_file"]).is_file()
    assert (ROOT / renewable_cfg["wind"]["test_file"]).is_file()
    runner = AmprionRenewableForecaster(
        ROOT, "amprion", renewable_cfg, device=torch.device("cpu")
    )
    # The first 24-hour holdout period begins immediately after the train cutoff.
    origin = pd.Timestamp("2026-09-30T23:45:00Z")
    records = runner.forecast_at(origin)
    counts = pd.DataFrame(records).groupby("generation_type").size().to_dict()
    assert counts == {"solar": 96, "wind": 96}

    # Changing realised generation after the forecast origin must not change
    # predictions; future target columns are masked before feature building.
    future_start = origin + pd.Timedelta(minutes=15)
    future_end = origin + pd.Timedelta(days=1)
    solar_future = runner.solar_frame["utc_timestamp"].between(future_start, future_end)
    wind_future = runner.wind_frame["utc_timestamp"].between(future_start, future_end)
    # Future realised targets and future observed weather are deliberately changed.
    # Serving forecasts must remain driven only by the forecast_* weather columns.
    runner.solar_frame.loc[solar_future, "Photovoltaics [MWh]"] = 999999.0
    for column in ("cloud_cover_mean_pct", "shortwave_radiation_mean_prev_hour_w_m2"):
        if column in runner.solar_frame.columns:
            runner.solar_frame.loc[solar_future, column] = 999999.0
    runner.wind_frame.loc[wind_future, "Wind onshore [MWh]"] = 999999.0
    for column in ("wind_speed_100m_mean_ms", "wind_direction_100m_circular_mean_deg"):
        if column in runner.wind_frame.columns:
            runner.wind_frame.loc[wind_future, column] = 999999.0
    altered_target_records = runner.forecast_at(origin)
    for kind in ("solar", "wind"):
        original = [r["predicted_generation_mwh"] for r in records if r["generation_type"] == kind]
        altered = [r["predicted_generation_mwh"] for r in altered_target_records if r["generation_type"] == kind]
        assert np.allclose(original, altered, rtol=0.0, atol=1e-5)

    modes = {kind: {r["weather_mode"] for r in records if r["generation_type"] == kind}
             for kind in ("solar", "wind")}
    assert modes == {
        "solar": {"forecast_cloud_and_radiation"},
        "wind": {"forecast_wind_speed_and_direction"},
    }
    assert all(record["predicted_generation_mwh"] >= 0 for record in records)


def test_renewables_results_api_returns_station_scoped_coverage():
    from fastapi.testclient import TestClient
    from backend.main import app

    response = TestClient(app).get(
        "/results/renewables", params={"station": "amprion", "limit": 10}
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["station"] == "amprion"
    assert {"solar", "wind"}.issubset(payload["coverage"])
    assert len(payload["actuals"]) <= 10


def test_renewable_monitoring_figure_keeps_solar_visible_and_wind_separate():
    from backend.results_api import _load_configured_renewable_actuals
    from dashboard.renewable_plot import build_renewable_figure

    # First renewable forecast origin is the final training timestamp; the
    # first displayed renewable forecast target is 1 October 2026 00:00 UTC.
    origin = pd.Timestamp("2026-09-30T23:45:00Z")
    actuals, coverage = _load_configured_renewable_actuals(
        "amprion", end_time=origin.isoformat()
    )
    start = origin - pd.Timedelta(hours=72)
    actuals = actuals[actuals["utc_timestamp"].between(start, origin)]
    future_times = pd.date_range(origin + pd.Timedelta(minutes=15), periods=96, freq="15min")
    forecast_records = {
        kind: [
            {
                "forecast_timestamp_utc": ts.isoformat(),
                "forecast_origin_utc": origin.isoformat(),
                "predicted_generation_mwh": 100.0 + step,
            }
            for step, ts in enumerate(future_times, start=1)
        ]
        for kind in ("solar", "wind")
    }

    figure, counts = build_renewable_figure(
        actuals, forecast_records, chart_start=start, chart_end=future_times[-1]
    )

    assert counts["solar_actual"] > 0
    assert counts["wind_actual"] > 0
    assert counts["solar_forecast"] == 96
    assert counts["wind_forecast"] == 96
    trace_names = {trace.name for trace in figure.data}
    assert {"Solar actual", "Solar forecast", "Wind actual", "Wind forecast"}.issubset(trace_names)
    # Solar and wind use separate y axes/rows, so wind magnitude cannot flatten solar.
    assert figure.data[0].yaxis == "y"
    assert figure.data[2].yaxis == "y2"
    assert coverage["solar"]["rows"] > 0 and coverage["wind"]["rows"] > 0


def test_renewable_api_run_selector_uses_latest_complete_96_step_run():
    from backend.results_api import _latest_renewable_run

    origin_old = pd.Timestamp("2026-09-30T23:45:00Z")
    origin_new = pd.Timestamp("2026-10-01T00:45:00Z")
    records = []
    for run_id, origin, count in [
        ("old-complete", origin_old, 96),
        ("new-partial", origin_new, 42),
    ]:
        for step in range(1, count + 1):
            records.append({
                "forecast_run_id": run_id,
                "forecast_origin_utc": origin,
                "forecast_issued_at_utc": origin + pd.Timedelta(seconds=5),
                "forecast_timestamp_utc": origin + pd.Timedelta(minutes=15 * step),
                "horizon_step": step,
                "predicted_generation_mwh": float(step),
            })
    selected, complete = _latest_renewable_run(pd.DataFrame(records))
    assert complete is True
    assert len(selected) == 96
    assert selected["forecast_run_id"].eq("old-complete").all()
