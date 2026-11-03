from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.tabular_forecasting import build_origin_features
from src.models import model_registry
from src.storage.forecast_store import FORECAST_FIELDS, append_forecasts


def _history(rows: int = 672) -> pd.DataFrame:
    timestamps = pd.date_range("2025-01-01", periods=rows, freq="15min", tz="UTC")
    index = np.arange(rows, dtype=float)
    return pd.DataFrame({
        "utc_timestamp": timestamps,
        "demand": 7000 + 300 * np.sin(index / 96 * 2 * np.pi) + index * 0.01,
        "temperature": 8 + 4 * np.sin(index / (96 * 7) * 2 * np.pi),
    })


def test_tabular_feature_builder_returns_one_row_per_future_step() -> None:
    history = _history()
    future = pd.date_range(
        history["utc_timestamp"].iloc[-1] + pd.Timedelta(minutes=15),
        periods=96,
        freq="15min",
        tz="UTC",
    )
    features = build_origin_features(history, future, interval_minutes=15)
    assert len(features) == 96
    assert "horizon_step" in features
    assert features["horizon_step"].iloc[0] == 1
    assert features["horizon_step"].iloc[-1] == 96
    assert np.isfinite(features.to_numpy(dtype=float)).all()


def test_tabular_feature_builder_rejects_non_contiguous_history() -> None:
    history = _history()
    history.loc[100, "utc_timestamp"] += pd.Timedelta(minutes=15)
    future = pd.date_range(
        pd.Timestamp("2025-01-08T00:00:00Z"), periods=2, freq="15min"
    )
    try:
        build_origin_features(history, future, interval_minutes=15)
    except ValueError as exc:
        assert "not continuous" in str(exc)
    else:
        raise AssertionError("Non-contiguous history should be rejected.")


def test_forecast_csv_migration_preserves_old_rows_and_adds_lineage(tmp_path: Path) -> None:
    path = tmp_path / "forecasts.csv"
    old_fields = [
        "station_id", "forecast_issued_at_utc", "forecast_origin_utc",
        "forecast_timestamp_utc", "horizon_step", "predicted_demand", "model_file",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=old_fields)
        writer.writeheader()
        writer.writerow({
            "station_id": "amprion",
            "forecast_issued_at_utc": "2025-01-01T00:00:00Z",
            "forecast_origin_utc": "2025-01-01T00:00:00Z",
            "forecast_timestamp_utc": "2025-01-01T00:15:00Z",
            "horizon_step": 1,
            "predicted_demand": 7000,
            "model_file": "old.pt",
        })

    append_forecasts(path, [{
        "station_id": "tennet",
        "forecast_issued_at_utc": "2025-01-01T00:15:00Z",
        "forecast_origin_utc": "2025-01-01T00:15:00Z",
        "forecast_timestamp_utc": "2025-01-01T00:30:00Z",
        "horizon_step": 1,
        "predicted_demand": 3000,
        "model_file": "model.joblib",
        "model_version": "v1",
        "algorithm": "xgboost",
        "experiment_id": "test-exp",
        "weather_mode": "last_observed_temperature_persistence",
    }])

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        assert reader.fieldnames == FORECAST_FIELDS
    assert len(rows) == 2
    assert rows[0]["station_id"] == "amprion"
    assert rows[0]["algorithm"] == ""
    assert rows[1]["algorithm"] == "xgboost"
    assert rows[1]["experiment_id"] == "test-exp"


def test_file_backed_registry_upsert_and_versioning(tmp_path: Path, monkeypatch) -> None:
    registry_path = tmp_path / "models" / "registry" / "experiments.json"
    monkeypatch.setattr(model_registry, "REGISTRY_PATH", registry_path)
    first = {
        "experiment_id": "exp-1", "station": "amprion", "algorithm": "xgboost",
        "version": "v1", "status": "candidate", "created_at": "2025-01-01T00:00:00Z",
    }
    model_registry.upsert_experiment(first)
    assert model_registry.get_experiment("exp-1")["status"] == "candidate"
    assert model_registry.next_version("amprion", "xgboost") == "v2"
    model_registry.upsert_experiment({**first, "status": "deployed"})
    items = model_registry.list_experiments("amprion")
    assert len(items) == 1
    assert items[0]["status"] == "deployed"
