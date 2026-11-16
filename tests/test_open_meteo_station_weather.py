from __future__ import annotations

import asyncio
from copy import deepcopy
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from starlette.datastructures import UploadFile

from backend import station_api
from backend.training_data import store_source
from src.ingestion import station_weather as sw
from src.models.forecast_weather import ForecastWeatherUnavailable
from src.models.renewable_forecaster import AmprionRenewableForecaster
from src.models.streaming_predictor import generate_forecast
from src.models.train_renewable import sequence_arrays
from src.quality.validator import validate_event


def _point():
    return {"location": "Test region", "latitude": 51.5, "longitude": 7.5}


def _dates(start, end):
    return pd.date_range(start, end + pd.Timedelta(hours=23), freq="h", tz="UTC")


def _weather_response(times, value):
    return pd.DataFrame({name: value for name in sw.FIELDS}, index=times, dtype=float)


def test_sqlite_reuses_dated_weather_and_floors_valid_time(tmp_path, monkeypatch):
    def fake_download(self, kind, start, end):
        times = _dates(start, end)
        data = _weather_response(times, 2 if kind == "historical" else 6)
        data["temperature_2m"] += times.hour
        return data

    monkeypatch.setattr(sw.StationWeather, "_get", fake_download)
    cache = tmp_path / "weather.sqlite3"
    service = sw.StationWeather(cache, [_point()])
    times = pd.date_range("2025-10-02T10:45Z", periods=2, freq="15min")
    data = {"demand": pd.DataFrame({"utc_timestamp": times, "demand": [8, 9]})}
    service.enrich(data)
    assert data["demand"].temperature.tolist() == [12, 13]
    assert data["demand"].forecast_temperature.tolist() == [16, 17]

    def offline(*args):
        raise ForecastWeatherUnavailable("network unavailable")

    monkeypatch.setattr(sw.StationWeather, "_get", offline)
    reopened = sw.StationWeather(cache, [_point()])
    assert reopened.forecast(times, live=False).temperature_2m.tolist() == [16, 17]
    assert reopened.historical(times).temperature_2m.tolist() == [12, 13]


def test_multiple_points_use_circular_wind_direction(tmp_path, monkeypatch):
    locations = [_point(), {"location": "Other", "latitude": 51, "longitude": 8}]
    service = sw.StationWeather(tmp_path / "weather.sqlite3", locations)

    class Response:
        status_code = 200

        def json(self):
            out = []
            for direction in (350, 10):
                hourly = {name: [20] for name in sw.FIELDS}
                hourly["wind_direction_100m"] = [direction]
                hourly["time"] = ["2026-10-02T00:00"]
                out.append({"hourly": hourly})
            return out

    monkeypatch.setattr(service.session, "get", lambda *args, **kwargs: Response())
    frame = service._get("historical", pd.Timestamp("2026-10-02"), pd.Timestamp("2026-10-02"))
    assert min(frame.wind_direction_100m.iloc[0], 360 - frame.wind_direction_100m.iloc[0]) < 1e-6


def test_weatherless_onboard_and_retraining_use_same_location(tmp_path, monkeypatch):
    state = {"stations": {}, "active_station": None}
    monkeypatch.setattr(station_api, "ROOT", tmp_path)
    monkeypatch.setattr(station_api, "read_station_config", lambda: deepcopy(state))
    monkeypatch.setattr(station_api, "write_station_config", state.update)
    monkeypatch.setattr(station_api, "_read_model_settings", lambda: {"interval_minutes": 15})
    monkeypatch.setattr(station_api.manager, "ensure_station_topics", lambda key: {"name": key})

    def fake_download(self, kind, start, end):
        assert self.sites == [_point()]
        return _weather_response(_dates(start, end), 12 if kind == "historical" else 20)

    monkeypatch.setattr(sw.StationWeather, "_get", fake_download)
    grid = pd.date_range("2025-10-01", periods=2000, freq="15min", tz="UTC")
    energy = pd.DataFrame({"utc_timestamp": grid.astype(str), "demand": np.arange(len(grid)) + 1000})
    workbook = BytesIO()
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        pd.DataFrame([_point()]).to_excel(writer, sheet_name="Station Location", index=False)
        energy.to_excel(writer, sheet_name="Demand", index=False)
    workbook.seek(0)
    result = asyncio.run(station_api.onboard_station(
        station_id="test-station", file=UploadFile(filename="site.xlsx", file=workbook),
        display_name="New station", stream_fraction=0.2, timestamp_format="auto",
        source_timezone="UTC", interval_minutes=15, selected_models="demand_forecast",
        raw_topic=None, featured_topic=None, stream_input="kafka",
    ))
    assert result["registered"]
    entry = state["stations"]["test-station"]
    assert entry["stream_input"] == "kafka" and entry["weather"]["sites"] == [_point()]
    assert (tmp_path / "data/state/test-station/weather.sqlite3").exists()
    training = pd.read_csv(tmp_path / entry["training_file"])
    assert set(training.temperature) == {12} and set(training.forecast_temperature) == {20}
    seed = pd.read_csv(tmp_path / entry["history_file"])
    assert pd.Timestamp(seed.utc_timestamp.iloc[-1]) == grid[-1]

    upload = UploadFile(filename="training.csv", file=BytesIO(energy.to_csv(index=False).encode()))
    source = asyncio.run(store_source(root=tmp_path, station="test-station", entry=entry,
                                      component="demand_forecast", file=upload))
    assert source["rows"] == len(grid)
    assert pd.read_csv(tmp_path / source["path"]).forecast_temperature.eq(20).all()


def test_live_demand_uses_weather_and_keeps_actuals_when_missing():
    origin = pd.Timestamp.now(tz="UTC").floor("15min")
    raw = {"station_id": "test", "utc_timestamp": origin.isoformat(), "demand": 100.0}
    event, errors = validate_event(raw, "test", 15, require_temperature=False)
    assert errors == [] and event["temperature"] is None
    assert validate_event(raw, "test", 15)[0] is None

    class Weather:
        def historical(self, times, *, live):
            assert live
            return _weather_response(sw.utc_index(times), 11)

        def forecast(self, times, *, live):
            assert live
            result = _weather_response(times, 30)
            result["temperature_2m"] += np.arange(len(times))
            return result

    class Model:
        def predict(self, features):
            return features["future_temperature_forecast"].to_numpy()

    rows = [{"utc_timestamp": ts, "demand": 100, "temperature": 10}
            for ts in pd.date_range(end=origin, periods=672, freq="15min")]
    rows[-1] = event
    checkpoint = {"history": 672, "horizon": 96, "interval_minutes": 15, "station": "test",
                  "algorithm": "random_forest", "weather_mode": sw.FORECAST_WEATHER_MODE}
    result = generate_forecast(rows, Model(), checkpoint, torch.device("cpu"), Path("model.joblib"), weather=Weather())
    assert len(result) == 96
    assert result[0]["predicted_demand"] == 30 and result[-1]["predicted_demand"] == 125
    assert result[0]["weather_mode"] == "open_meteo_live"
    assert event["temperature"] == 11


@pytest.mark.parametrize("kind", ["solar", "wind"])
def test_renewable_live_future_needs_no_generation_rows(kind, tmp_path):
    from src.models.train_renewable import solar, wind

    origin = pd.Timestamp.now(tz="UTC").floor("15min")
    times = pd.date_range(end=origin, periods=672, freq="15min")
    builder = solar if kind == "solar" else wind
    target, capacity = builder.TARGET_COLUMN, builder.CAPACITY_COLUMN

    class Weather:
        def historical(self, times, *, live):
            assert live
            return _weather_response(times, 4)

        def forecast(self, times, *, live):
            assert live
            return _weather_response(times, 12)

    runner = AmprionRenewableForecaster(tmp_path, "test", {}, weather=Weather())
    data = pd.DataFrame({"utc_timestamp": times[:-1], target: 8.0, capacity: 42.0})
    for column in sw.HISTORY_COLUMNS[kind].values():
        data[column] = 4
    setattr(runner, f"{kind}_frame", data)
    setattr(runner, f"{kind}_enabled", True)
    runner._renewable_time_index[kind] = pd.DatetimeIndex(data.utc_timestamp)
    runner.accept_observation(origin, {f"{kind}_mwh": 9.5, f"{kind}_capacity_mw": 44})
    frame, future, mode = runner._managed_context(kind, origin, replay_weather=False)
    assert len(future) == 96 and mode == "open_meteo_live"
    assert runner.observations_at(origin, live=True)[0]["actual_generation_mwh"] == 9.5
    assert frame.iloc[672:][capacity].eq(44).all()
    _, weather_features, placeholders = sequence_arrays(kind, frame, np.array([672]))
    assert weather_features[0, :, 0].tolist() == [12] * 96
    assert np.count_nonzero(placeholders) == 0


def test_live_cache_uses_previous_runs_day0_and_reopens_after_failure(tmp_path, monkeypatch):
    now = pd.Timestamp.now(tz="UTC").floor("h")
    times = pd.date_range(now, periods=30, freq="h")
    cache = tmp_path / "weather.sqlite3"
    called = []

    def fallback(self, kind, start, end):
        called.append(kind)
        if kind == "live":
            raise ForecastWeatherUnavailable("primary unavailable")
        return _weather_response(times, 18)

    monkeypatch.setattr(sw.StationWeather, "_get", fallback)
    service = sw.StationWeather(cache, [_point()])
    assert service.forecast(times[:2], live=True).temperature_2m.tolist() == [18, 18]
    assert called == ["live", "live_day0"]
    fresh = sw.StationWeather(cache, [_point()])
    assert fresh.forecast(times[:2], live=True).temperature_2m.tolist() == [18, 18]
    assert called == ["live", "live_day0"]


def test_archive_training_span_does_not_bridge_unavailable_forecast():
    frame = pd.DataFrame({"forecast_temperature": [None, 2, 3, 4, None, 5, 6]})
    assert sw.archive_training_span(frame, ["forecast_temperature"]).forecast_temperature.tolist() == [2, 3, 4]
    with pytest.raises(ValueError, match="No archived day-ahead"):
        sw.archive_training_span(frame.iloc[:1], ["forecast_temperature"])
