"""The extra learning inputs are opt-in for Random Forest only."""
import json

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.dummy import DummyRegressor

from src.models import model_registry, train_tabular
from src.models.tabular_forecasting import (
    FEATURE_SCHEMA, LEGACY_FEATURE_SCHEMA, TARGET_DAY_FEATURE, TARGET_WEEK_FEATURE,
    build_training_examples,
)


@pytest.mark.parametrize("algorithm,expected_schema,feature_count", [
    ("random_forest", FEATURE_SCHEMA, 38),
    ("gbt", LEGACY_FEATURE_SCHEMA, 36),
    ("xgboost", LEGACY_FEATURE_SCHEMA, 36),
])
def test_training_selects_new_features_only_for_random_forest(
    tmp_path, monkeypatch, algorithm, expected_schema, feature_count,
):
    timestamps = pd.date_range("2025-01-01", periods=2000, freq="15min", tz="UTC")
    frame = pd.DataFrame({
        "utc_timestamp": timestamps.astype("datetime64[ns, UTC]"),
        "demand": 1000 + np.arange(2000, dtype=float), "temperature": 10.0,
    })
    config = tmp_path / "config"
    config.mkdir()
    (config / "stations.yaml").write_text(yaml.safe_dump({
        "stations": {"demo": {"training_file": "source.csv", "interval_minutes": 15}}
    }))
    (config / "model.yaml").write_text("history_window_hours: 168\nforecast_horizon_hours: 24\n")
    frame.to_csv(tmp_path / "source.csv", index=False)
    monkeypatch.setattr(train_tabular, "ROOT", tmp_path)
    monkeypatch.setattr(train_tabular, "load_station_csv", lambda *args, **kwargs: frame.copy())
    monkeypatch.setattr(model_registry, "REGISTRY_PATH", tmp_path / "registry.json")
    # This tests input routing, not any estimator's implementation.
    inputs_seen = []

    class InputRecorder(DummyRegressor):
        def fit(self, x, y):
            inputs_seen.append(x.copy())
            return super().fit(x, y)

    monkeypatch.setattr(train_tabular, "build_estimator", lambda *args: InputRecorder())
    monkeypatch.setattr(train_tabular.joblib, "dump", lambda model, path, **kwargs: path.write_bytes(b"test"))
    experiment_id = f"scope-{algorithm}"
    metadata = train_tabular.run_training({
        "station": "demo", "algorithm": algorithm, "experiment_id": experiment_id,
        "parameters": {"max_train_origins": 3, "max_validation_origins": 2},
        "training_options": {"validation_fraction": 0.15},
    })
    assert metadata["feature_schema"] == expected_schema
    assert len(metadata["feature_names"]) == feature_count
    assert inputs_seen[0].shape == (288, feature_count)
    references, _, _ = build_training_examples(frame, np.array([672, 1138, 1604]), 672, 96, 15)
    # Shared/default callers still receive the exact original feature matrix.
    if algorithm != "random_forest":
        np.testing.assert_array_equal(inputs_seen[0], references.to_numpy(dtype=np.float32))
        assert TARGET_DAY_FEATURE not in metadata["feature_names"]
        assert TARGET_WEEK_FEATURE not in metadata["feature_names"]
    else:
        assert TARGET_DAY_FEATURE in metadata["feature_names"]
        assert TARGET_WEEK_FEATURE in metadata["feature_names"]
    saved = json.loads((tmp_path / metadata["metadata_path"]).read_text())
    assert saved["feature_schema"] == expected_schema
    assert model_registry.get_experiment(experiment_id)["feature_schema"] == expected_schema
