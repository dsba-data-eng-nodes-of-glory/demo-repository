from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from src.models.tabular_forecasting import (
    FEATURE_SCHEMA, LEGACY_FEATURE_SCHEMA, TARGET_DAY_FEATURE, TARGET_WEEK_FEATURE,
    build_origin_features, build_training_examples, target_demand_references,
)


def history_and_targets(interval=15):
    steps = 10080 // interval
    history = pd.DataFrame({
        "utc_timestamp": pd.date_range("2025-03-23T00:00:00Z", periods=steps,
                                       freq=f"{interval}min"),
        "demand": np.arange(steps, dtype=float) + 1000,
        "temperature": np.full(steps, 10.0),
    })
    history["utc_timestamp"] = history["utc_timestamp"].astype("datetime64[ns, UTC]")
    targets = pd.date_range(history["utc_timestamp"].iloc[-1] + pd.Timedelta(minutes=interval),
                            periods=1440 // interval, freq=f"{interval}min")
    return history, targets


@pytest.mark.parametrize("interval", [15, 30, 60, 90, 120])
def test_target_features_follow_each_target_timestamp(interval):
    history, targets = history_and_targets(interval)
    features = build_origin_features(history, targets, interval, feature_schema=FEATURE_SCHEMA)
    steps = len(targets)
    np.testing.assert_array_equal(features[TARGET_DAY_FEATURE], history["demand"].iloc[-steps:])
    np.testing.assert_array_equal(features[TARGET_WEEK_FEATURE], history["demand"].iloc[:steps])
    assert features[TARGET_DAY_FEATURE].iloc[-1] == history["demand"].iloc[-1]


def test_references_reject_horizon_requiring_future_observations():
    history, targets = history_and_targets()
    extended = targets.append(pd.DatetimeIndex([targets[-1] + pd.Timedelta(minutes=15)]))
    with pytest.raises(ValueError, match="reference is unavailable"):
        build_origin_features(history, extended, feature_schema=FEATURE_SCHEMA)


def test_references_do_not_substitute_nearby_timestamps():
    history, targets = history_and_targets()
    with pytest.raises(ValueError, match="reference is unavailable"):
        target_demand_references(history, targets + pd.Timedelta(minutes=1))


def test_legacy_feature_schema_preserves_existing_model_inputs():
    history, targets = history_and_targets()
    legacy = build_origin_features(history, targets, feature_schema=LEGACY_FEATURE_SCHEMA)
    updated = build_origin_features(history, targets, feature_schema=FEATURE_SCHEMA)
    pd.testing.assert_frame_equal(legacy, build_origin_features(history, targets))
    assert TARGET_DAY_FEATURE not in legacy and TARGET_WEEK_FEATURE not in legacy
    pd.testing.assert_frame_equal(legacy, updated[legacy.columns])
    with pytest.raises(ValueError, match="Unsupported"):
        build_origin_features(history, targets, feature_schema="unknown")


def test_changing_future_labels_cannot_change_features():
    history, targets = history_and_targets()
    future = pd.DataFrame({"utc_timestamp": targets, "demand": 2000.0, "temperature": 99.0})
    frame = pd.concat([history, future], ignore_index=True)
    changed = frame.copy()
    changed.loc[len(history):, ["demand", "temperature"]] = -123456.0
    x, y, _ = build_training_examples(frame, np.array([len(history)]), len(history), len(targets), 15, feature_schema=FEATURE_SCHEMA)
    other_x, other_y, _ = build_training_examples(changed, np.array([len(history)]), len(history), len(targets), 15, feature_schema=FEATURE_SCHEMA)
    pd.testing.assert_frame_equal(x, other_x)
    assert not np.array_equal(y, other_y)


@pytest.mark.parametrize("schema", [LEGACY_FEATURE_SCHEMA, FEATURE_SCHEMA])
def test_serving_uses_saved_schema_and_matches_training_predictions(schema):
    import torch
    from src.models.streaming_predictor import generate_forecast

    history, targets = history_and_targets()
    features = build_origin_features(history, targets, feature_schema=schema)
    model = RandomForestRegressor(n_estimators=2, max_depth=2, random_state=42)
    model.fit(features.to_numpy(dtype=np.float32), np.arange(len(targets)))
    checkpoint = {
        "algorithm": "random_forest", "region_name": "demo", "feature_schema": schema,
        "feature_names": list(features), "history": len(history), "horizon": len(targets),
        "interval_minutes": 15,
    }
    rows = generate_forecast(history.to_dict("records"), model, checkpoint, torch.device("cpu"), Path("model.joblib"))
    np.testing.assert_array_equal(
        [row["predicted_demand"] for row in rows],
        model.predict(features.to_numpy(dtype=np.float32)),
    )
