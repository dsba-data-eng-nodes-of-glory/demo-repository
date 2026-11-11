from __future__ import annotations

import pandas as pd
import yaml

from backend import results_api


def _write_csv(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def test_renewables_endpoint_returns_latest_solar_and_wind_run(tmp_path, monkeypatch):
    root = tmp_path
    station_dir = root / "data" / "training" / "amprion"
    processed = root / "data" / "processed" / "stations" / "amprion"
    config_path = root / "config" / "stations.yaml"
    config_path.parent.mkdir(parents=True)

    station_config = {
        "stations": {
            "amprion": {
                "renewable_forecasts_file": "data/processed/stations/amprion/renewable_forecasts.csv",
                "renewables": {
                    "enabled": True,
                    "solar": {
                        "train_file": "data/training/amprion/solar_train.csv",
                        "test_file": "data/training/amprion/solar_test.csv",
                    },
                    "wind": {
                        "train_file": "data/training/amprion/wind_train.csv",
                        "test_file": "data/training/amprion/wind_test.csv",
                    },
                },
            }
        }
    }
    config_path.write_text(yaml.safe_dump(station_config), encoding="utf-8")
    monkeypatch.setattr(results_api, "ROOT", root)
    monkeypatch.setattr(results_api, "STATION_CONFIG_FILE", config_path)
    results_api._RENEWABLE_BASE_CACHE.clear()
    results_api._CACHE.clear()

    times = pd.date_range("2026-09-30 23:30", periods=4, freq="15min", tz="UTC")
    _write_csv(station_dir / "solar_train.csv", pd.DataFrame({
        "utc_timestamp": times[:2], "Photovoltaics [MWh]": [0.1, 0.2]
    }))
    _write_csv(station_dir / "solar_test.csv", pd.DataFrame({
        "utc_timestamp": times[2:], "Photovoltaics [MWh]": [0.3, 0.4]
    }))
    _write_csv(station_dir / "wind_train.csv", pd.DataFrame({
        "utc_timestamp": times[:2], "Wind onshore [MWh]": [1.1, 1.2]
    }))
    _write_csv(station_dir / "wind_test.csv", pd.DataFrame({
        "utc_timestamp": times[2:], "Wind onshore [MWh]": [1.3, 1.4]
    }))

    origin = pd.Timestamp("2026-10-01T00:15:00Z")
    forecast_rows = []
    for kind in ("solar", "wind"):
        for step in range(1, 97):
            forecast_rows.append({
                "station_id": "amprion",
                "generation_type": kind,
                "forecast_issued_at_utc": "2026-10-01T00:16:00Z",
                "forecast_origin_utc": origin.isoformat(),
                "forecast_run_id": f"{kind}-latest-run",
                "forecast_timestamp_utc": (origin + pd.Timedelta(minutes=15 * step)).isoformat(),
                "horizon_step": step,
                "predicted_generation_mwh": float(step),
                "model_version": "test-v1",
                "weather_mode": "forecast_weather",
            })
    _write_csv(processed / "renewable_forecasts.csv", pd.DataFrame(forecast_rows))

    payload = results_api.renewables(
        "amprion", limit=5000, end_time="2026-10-01T00:15:00Z"
    )
    assert len(payload["forecasts"]["solar"]) == 96
    assert len(payload["forecasts"]["wind"]) == 96
    assert payload["diagnostics"]["solar_forecast_points"] == 96
    assert payload["diagnostics"]["wind_forecast_points"] == 96
    assert len(payload["actuals"]) == 4


def test_renewable_actual_sources_are_cached_by_file_signature(tmp_path, monkeypatch):
    root = tmp_path
    station_dir = root / "data" / "training" / "amprion"
    config_path = root / "config" / "stations.yaml"
    config_path.parent.mkdir(parents=True)
    config = {
        "stations": {"amprion": {"renewables": {
            "enabled": True,
            "solar": {"train_file": "data/training/amprion/solar_train.csv"},
            "wind": {"train_file": "data/training/amprion/wind_train.csv"},
        }}}
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    times = pd.date_range("2026-10-01", periods=4, freq="15min", tz="UTC")
    _write_csv(station_dir / "solar_train.csv", pd.DataFrame({
        "utc_timestamp": times, "Photovoltaics [MWh]": [0.0, 0.1, 0.2, 0.3]
    }))
    _write_csv(station_dir / "wind_train.csv", pd.DataFrame({
        "utc_timestamp": times, "Wind onshore [MWh]": [1.0, 1.1, 1.2, 1.3]
    }))
    monkeypatch.setattr(results_api, "ROOT", root)
    monkeypatch.setattr(results_api, "STATION_CONFIG_FILE", config_path)
    results_api._RENEWABLE_BASE_CACHE.clear()
    results_api._CACHE.clear()

    first, _ = results_api._load_configured_renewable_actuals("amprion")
    second, _ = results_api._load_configured_renewable_actuals("amprion")
    assert len(first) == 4
    assert first.equals(second)
    assert len(results_api._RENEWABLE_BASE_CACHE) == 1
