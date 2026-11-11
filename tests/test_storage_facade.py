import csv

from src.storage.forecast_store import append_actuals, append_forecasts


def test_csv_actual_storage_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv("STORAGE_BACKEND", "csv")
    path = tmp_path / "actuals.csv"
    rows = [
        {"station_id": "tennet", "utc_timestamp": "2026-11-04T09:00:00Z", "actual_demand": 100, "temperature": 8},
        {"station_id": "tennet", "utc_timestamp": "2026-11-04T09:00:00Z", "actual_demand": 100, "temperature": 8},
    ]
    assert append_actuals(path, rows) == 1
    assert append_actuals(path, rows) == 0
    with path.open(newline="", encoding="utf-8") as handle:
        saved = list(csv.DictReader(handle))
    assert len(saved) == 1


def test_csv_forecast_schema_contains_stable_run_id(tmp_path, monkeypatch):
    monkeypatch.setenv("STORAGE_BACKEND", "csv")
    path = tmp_path / "forecasts.csv"
    rows = [{
        "station_id": "tennet",
        "forecast_issued_at_utc": "2026-11-04T09:00:00Z",
        "forecast_origin_utc": "2026-11-04T09:00:00Z",
        "forecast_run_id": "tennet|origin|gbt|v1|exp1",
        "forecast_timestamp_utc": "2026-11-04T09:15:00Z",
        "horizon_step": 1,
        "predicted_demand": 123.0,
        "model_file": "model.joblib",
        "model_version": "v1",
        "algorithm": "gbt",
        "experiment_id": "exp1",
        "weather_mode": "persistence",
    }]
    assert append_forecasts(path, rows) == 1
    # A crash/restart can replay the same origin before its Kafka offset is
    # committed. The stable run_id + horizon step makes that retry idempotent.
    assert append_forecasts(path, rows) == 0
    with path.open(newline="", encoding="utf-8") as handle:
        saved = list(csv.DictReader(handle))
    assert saved[0]["forecast_run_id"] == "tennet|origin|gbt|v1|exp1"
    assert len(saved) == 1


def test_csv_forecast_cache_rebuilds_after_external_file_change(tmp_path, monkeypatch):
    monkeypatch.setenv("STORAGE_BACKEND", "csv")
    path = tmp_path / "forecasts.csv"
    first = {
        "station_id": "tennet",
        "forecast_run_id": "run-1",
        "horizon_step": 1,
        "forecast_timestamp_utc": "2026-11-04T09:15:00Z",
        "predicted_demand": 123.0,
    }
    assert append_forecasts(path, [first]) == 1

    # Simulate a second process appending a distinct run to the same CSV.
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            "tennet,2026-11-04T09:00:00Z,2026-11-04T09:00:00Z,run-2,"
            "2026-11-04T09:15:00Z,1,124.0,model.joblib,v1,gbt,exp1,persistence\n"
        )
    second = {**first, "forecast_run_id": "run-2", "predicted_demand": 124.0}
    assert append_forecasts(path, [second]) == 0
