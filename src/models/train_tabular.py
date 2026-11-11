"""Train RF, gradient-boosted-tree or XGBoost forecasting regressors.

The estimator is a shared one-step regressor with lead time as a feature. A
forecast is produced by creating one row per configured future observation.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, silhouette_score
from sklearn.multiclass import OneVsRestClassifier

from src.ingestion.file_loader import load_station_csv
from src.models.model_registry import next_version, upsert_experiment, utc_now
from src.models.forecast_contract import station_forecast_contract
from src.models.kmeans_forecasting import KMeansDemandForecaster
from src.models.time_segments import (
    continuous_segment_bounds, sample_origins_by_segment, split_forecast_origins,
)
from src.models.tabular_forecasting import (
    DAY_ONLY_FEATURE_SCHEMA,
    LEGACY_FEATURE_SCHEMA,
    build_training_examples,
    json_safe_parameters,
)

ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def build_estimator(algorithm: str, params: dict[str, Any]):
    if algorithm == "random_forest":
        allowed = {key: params[key] for key in (
            "n_estimators", "max_depth", "min_samples_leaf", "random_state", "n_jobs"
        ) if key in params}
        return RandomForestRegressor(**allowed)
    if algorithm == "gbt":
        allowed = {key: params[key] for key in (
            "max_iter", "max_leaf_nodes", "learning_rate", "l2_regularization", "random_state"
        ) if key in params}
        return HistGradientBoostingRegressor(**allowed)
    if algorithm == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise RuntimeError(
                "XGBoost is not installed. Install it with `pip install xgboost` and restart the backend."
            ) from exc
        allowed = {key: params[key] for key in (
            "n_estimators", "max_depth", "learning_rate", "subsample",
            "colsample_bytree", "reg_lambda", "random_state", "n_jobs"
        ) if key in params}
        allowed.update({"objective": "reg:squarederror", "tree_method": "hist"})
        return XGBRegressor(**allowed)
    if algorithm == "kmeans":
        allowed = {key: params[key] for key in (
            "n_clusters", "random_state", "n_init", "max_iter"
        ) if key in params}
        return KMeansDemandForecaster(**allowed)
    raise ValueError(f"Unsupported tabular algorithm: {algorithm}")


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    actual = np.asarray(actual, dtype=float).reshape(-1)
    predicted = np.asarray(predicted, dtype=float).reshape(-1)
    error = predicted - actual
    denominator = float(np.sum(np.abs(actual)))
    return {
        "mae_mw": float(mean_absolute_error(actual, predicted)),
        "rmse_mw": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)) if len(actual) > 1 else float("nan"),
        "wape_pct": float(np.sum(np.abs(error)) / denominator * 100.0) if denominator else float("nan"),
        "bias_mw": float(np.mean(error)),
        "max_absolute_error_mw": float(np.max(np.abs(error))) if len(error) else float("nan"),
        "matched_observations": int(len(actual)),
    }


def demand_classes(target: np.ndarray, spike_threshold: float,
                   low_threshold: float, medium_threshold: float) -> np.ndarray:
    """Apply fixed training-only thresholds to demand targets."""
    values = np.asarray(target, dtype=float).reshape(-1)
    return np.select(
        [values >= spike_threshold, values <= low_threshold, values <= medium_threshold],
        ["Spike", "Low", "Medium"],
        default="Normal",
    )


def train_demand_classifier(x_train: np.ndarray, y_train: np.ndarray,
                            x_valid: np.ndarray, y_valid: np.ndarray,
                            parameters: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Fit a GBT One-vs-Rest demand classifier with thresholds from train rows only."""
    y_train = np.asarray(y_train, dtype=float).reshape(-1)
    non_spike = y_train[y_train < np.quantile(y_train, 0.95)]
    if len(non_spike) < 3:
        raise ValueError("Not enough non-spike training samples to form demand classes.")
    low_threshold, medium_threshold = np.quantile(non_spike, [1 / 3, 2 / 3]).tolist()
    spike_threshold = float(np.quantile(y_train, 0.95))
    y_train_class = demand_classes(y_train, spike_threshold, low_threshold, medium_threshold)
    y_valid_class = demand_classes(y_valid, spike_threshold, low_threshold, medium_threshold)
    base = HistGradientBoostingClassifier(
        max_iter=min(30, int(parameters.get("max_iter", 30))),
        max_leaf_nodes=min(15, int(parameters.get("max_leaf_nodes", 15))),
        learning_rate=float(parameters.get("learning_rate", 0.05)),
        l2_regularization=float(parameters.get("l2_regularization", 0.01)),
        random_state=int(parameters.get("random_state", 42)),
    )
    classifier = OneVsRestClassifier(base, n_jobs=1)
    classifier.fit(x_train, y_train_class)
    predicted_class = classifier.predict(x_valid)
    classification_metrics = {
        "accuracy": float(accuracy_score(y_valid_class, predicted_class)),
        "macro_f1": float(f1_score(y_valid_class, predicted_class, labels=["Spike", "Low", "Medium", "Normal"], average="macro", zero_division=0)),
        "classes": [str(value) for value in classifier.classes_],
        "label_method": "training-only 95th percentile for Spike; lower and upper tertiles of remaining training demand",
        "thresholds_mw": {
            "spike_min": spike_threshold,
            "low_max": float(low_threshold),
            "medium_max": float(medium_threshold),
        },
    }
    return classifier, classification_metrics


def run_training(config: dict[str, Any]) -> dict[str, Any]:
    overall_started = time.perf_counter()
    station = str(config["station"])
    algorithm = str(config["algorithm"])
    # Only the target-matched previous-day input is enabled for new RF models.
    # Other algorithms retain their existing training feature contract.
    feature_schema = DAY_ONLY_FEATURE_SCHEMA if algorithm == "random_forest" else LEGACY_FEATURE_SCHEMA
    experiment_id = str(config["experiment_id"])
    parameters = json_safe_parameters(dict(config.get("parameters") or {}))
    training_options = dict(config.get("training_options") or {})

    station_registry = load_yaml(ROOT / "config" / "stations.yaml").get("stations", {})
    if station not in station_registry:
        raise ValueError(f"Unknown station '{station}'.")
    station_entry = station_registry[station]
    csv_value = station_entry.get("training_file") or station_entry.get("input_file")
    if not csv_value:
        raise ValueError(f"Station '{station}' has no training_file or input_file configured.")
    csv_path = Path(csv_value)
    if not csv_path.is_absolute():
        csv_path = ROOT / csv_path
    csv_path = csv_path.resolve()

    model_cfg = load_yaml(ROOT / "config" / "model.yaml")
    contract = station_forecast_contract(station_entry, model_cfg)
    history_steps = int(contract["history_steps"])
    horizon_steps = int(contract["horizon_steps"])
    interval_minutes = int(contract["interval_minutes"])

    validation_fraction = float(training_options.get("validation_fraction", 0.15))
    if not 0.05 <= validation_fraction <= 0.40:
        raise ValueError("validation_fraction must be between 0.05 and 0.40.")

    frame = load_station_csv(csv_path, interval_minutes=interval_minutes, allow_gaps=True)
    n_rows = len(frame)
    timestamps = pd.DatetimeIndex(pd.to_datetime(frame["utc_timestamp"], utc=True))
    all_segments = continuous_segment_bounds(timestamps, interval_minutes)
    train_origins, validation_origins = split_forecast_origins(
        timestamps, history_steps, horizon_steps, interval_minutes,
        validation_fraction=validation_fraction,
    )
    if len(train_origins) < 2 or len(validation_origins) < 1:
        raise ValueError(
            f"Not enough continuous data for training and validation: found {len(train_origins)} "
            f"training and {len(validation_origins)} validation forecast windows across "
            f"{len(all_segments)} time segments. Each window needs {history_steps} history rows "
            f"followed by {horizon_steps} target rows inside one segment."
        )
    train_origins = sample_origins_by_segment(
        train_origins, timestamps, interval_minutes,
        int(parameters.get("max_train_origins", 300)),
    )
    validation_origins = sample_origins_by_segment(
        validation_origins, timestamps, interval_minutes,
        int(parameters.get("max_validation_origins", 80)),
    )
    if len(train_origins) < 2 or len(validation_origins) < 1:
        raise ValueError("Not enough train/validation forecast origins after sampling.")

    x_train, y_train, train_origin_ids = build_training_examples(
        frame, train_origins, history_steps, horizon_steps, interval_minutes,
        feature_schema=feature_schema,
    )
    x_valid, y_valid, valid_origin_ids = build_training_examples(
        frame, validation_origins, history_steps, horizon_steps, interval_minutes,
        feature_schema=feature_schema,
    )
    feature_names = list(x_train.columns)
    x_valid = x_valid.reindex(columns=feature_names)
    if x_train.isna().any().any() or x_valid.isna().any().any():
        raise ValueError("Training feature matrix contains missing values.")

    estimator = build_estimator(algorithm, parameters)
    started = time.perf_counter()
    estimator.fit(x_train.to_numpy(dtype=np.float32), y_train)
    estimator_fit_seconds = time.perf_counter() - started

    latency_started = time.perf_counter()
    predicted = estimator.predict(x_valid.to_numpy(dtype=np.float32))
    inference_latency_ms = (time.perf_counter() - latency_started) * 1000.0 / max(1, len(validation_origins))
    predicted = np.asarray(predicted, dtype=float).reshape(-1)
    if len(predicted) != len(y_valid):
        raise RuntimeError(f"Estimator returned {len(predicted)} predictions for {len(y_valid)} validation rows.")
    metrics = calculate_metrics(y_valid, predicted)

    classification_model = None
    classification_metrics = None
    if algorithm == "gbt":
        classification_model, classification_metrics = train_demand_classifier(
            x_train.to_numpy(dtype=np.float32), y_train,
            x_valid.to_numpy(dtype=np.float32), y_valid, parameters,
        )

    clustering_metrics = None
    if algorithm == "kmeans":
        valid_clusters = estimator.predict_cluster(x_valid.to_numpy(dtype=np.float32))
        scaled_valid = estimator.scaler.transform(x_valid.to_numpy(dtype=np.float32))
        silhouette = None
        if len(np.unique(valid_clusters)) > 1 and len(np.unique(valid_clusters)) < len(valid_clusters):
            silhouette = float(silhouette_score(scaled_valid, valid_clusters))
        clustering_metrics = {
            "clusters": int(estimator.n_clusters),
            "silhouette": silhouette,
            "cluster_mean_demand_mw": [float(value) for value in estimator.cluster_means_],
            "cluster_profiles": [
                {
                    "cluster": int(cluster),
                    "training_samples": int(estimator.cluster_sizes_[cluster]),
                    "avg_demand_mw": float(estimator.cluster_means_[cluster]),
                }
                for cluster in range(estimator.n_clusters)
            ],
            "prediction_method": "nearest standardized feature centroid; forecast MW is training mean demand in the assigned cluster",
        }

    # Error profile by lead step is part of the experiment and comparison output.
    horizon_metrics = []
    predicted_matrix = predicted.reshape(len(validation_origins), horizon_steps)
    actual_matrix = y_valid.reshape(len(validation_origins), horizon_steps)
    for lead in range(horizon_steps):
        lead_metrics = calculate_metrics(actual_matrix[:, lead], predicted_matrix[:, lead])
        horizon_metrics.append({"horizon_step": lead + 1, **lead_metrics})

    training_seconds = time.perf_counter() - overall_started
    created = utc_now()
    version = next_version(station, algorithm)
    artifact_dir = ROOT / "models" / station / "experiments" / experiment_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / "model.joblib"
    metadata_path = artifact_dir / "metadata.json"
    joblib.dump(estimator, artifact_path, compress=3)
    classification_artifact_path = None
    if classification_model is not None:
        classification_artifact_path = artifact_dir / "demand_classifier.joblib"
        joblib.dump(classification_model, classification_artifact_path, compress=3)
    classification_model_size_bytes = (
        int(classification_artifact_path.stat().st_size)
        if classification_artifact_path else 0
    )
    total_model_size_bytes = int(artifact_path.stat().st_size) + classification_model_size_bytes

    validation_start = str(timestamps[int(validation_origins.min())].isoformat())
    validation_end_index = min(n_rows - 1, int(validation_origins.max()) + horizon_steps - 1)
    validation_end = str(timestamps[validation_end_index].isoformat())
    training_start_index = max(0, int(train_origins.min()) - history_steps)
    training_end_index = min(n_rows - 1, int(train_origins.max()) + horizon_steps - 1)
    metadata = {
        "experiment_id": experiment_id,
        "station": station,
        "region_name": station,
        "algorithm": algorithm,
        "version": version,
        "artifact_path": artifact_path.relative_to(ROOT).as_posix(),
        "metadata_path": metadata_path.relative_to(ROOT).as_posix(),
        "feature_schema": feature_schema,
        "feature_names": feature_names,
        "history_steps": history_steps,
        "horizon_steps": horizon_steps,
        "interval_minutes": interval_minutes,
        "history_window_hours": contract["history_window_hours"],
        "forecast_horizon_hours": contract["forecast_horizon_hours"],
        "training_rows": int(n_rows),
        "continuous_segment_count": int(len(all_segments)),
        "training_origins": int(len(train_origins)),
        "training_samples": int(len(y_train)),
        "validation_origins": int(len(validation_origins)),
        "validation_origin_count": int(len(validation_origins)),
        "validation_samples": int(len(y_valid)),
        "training_start_utc": str(timestamps[training_start_index].isoformat()),
        "training_end_utc": str(timestamps[training_end_index].isoformat()),
        "validation_start_utc": validation_start,
        "validation_end_utc": validation_end,
        "validation_fraction": validation_fraction,
        "evaluation_protocol": "chronological_holdout_sampled_origins_per_segment_v1",
        "parameters": parameters,
        "metrics": metrics,
        "demand_classification": classification_metrics,
        "clustering": clustering_metrics,
        "classification_artifact_path": (
            classification_artifact_path.relative_to(ROOT).as_posix()
            if classification_artifact_path else None
        ),
        "horizon_metrics": horizon_metrics,
        "training_duration_seconds": float(training_seconds),
        "estimator_fit_seconds": float(estimator_fit_seconds),
        "inference_latency_ms_per_origin": float(inference_latency_ms),
        "model_size_bytes": total_model_size_bytes,
        "regressor_size_bytes": int(artifact_path.stat().st_size),
        "classifier_model_size_bytes": classification_model_size_bytes,
        "training_csv": csv_path.relative_to(ROOT).as_posix() if csv_path.is_relative_to(ROOT) else str(csv_path),
        "created_at": created,
        "weather_mode": "last_observed_temperature_persistence",
        "note": "Future temperature uses last-observed persistence in both training and inference; no future actual weather is used.",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    registry_record = {
        "experiment_id": experiment_id,
        "station": station,
        "algorithm": algorithm,
        "version": version,
        "status": "candidate",
        "artifact_path": metadata["artifact_path"],
        "metadata_path": metadata["metadata_path"],
        "feature_schema": feature_schema,
        "history_steps": history_steps,
        "horizon_steps": horizon_steps,
        "interval_minutes": interval_minutes,
        "validation_start_utc": validation_start,
        "validation_end_utc": validation_end,
        "validation_origin_count": int(len(validation_origins)),
        "evaluation_protocol": metadata["evaluation_protocol"],
        "parameters": parameters,
        "metrics": metrics,
        "demand_classification": classification_metrics,
        "clustering": clustering_metrics,
        "training_duration_seconds": float(training_seconds),
        "estimator_fit_seconds": float(estimator_fit_seconds),
        "inference_latency_ms_per_origin": float(inference_latency_ms),
        "model_size_bytes": total_model_size_bytes,
        "created_at": created,
    }
    upsert_experiment(registry_record)
    return metadata
