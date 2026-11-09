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
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.ingestion.file_loader import load_station_csv
from src.models.model_registry import next_version, upsert_experiment, utc_now
from src.models.forecast_contract import station_forecast_contract
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


def sample_origins(origins: np.ndarray, maximum: int) -> np.ndarray:
    if maximum <= 0:
        raise ValueError("max_train_origins and max_validation_origins must be positive.")
    if len(origins) <= maximum:
        return origins
    return origins[np.unique(np.linspace(0, len(origins) - 1, maximum, dtype=int))]


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
    raise ValueError(f"Unsupported tabular algorithm: {algorithm}")


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    actual = np.asarray(actual, dtype=float).reshape(-1)
    predicted = np.asarray(predicted, dtype=float).reshape(-1)
    error = predicted - actual
    denominator = float(np.sum(np.abs(actual)))
    return {
        "mae_mw": float(mean_absolute_error(actual, predicted)),
        "rmse_mw": float(np.sqrt(mean_squared_error(actual, predicted))),
        "wape_pct": float(np.sum(np.abs(error)) / denominator * 100.0) if denominator else float("nan"),
        "bias_mw": float(np.mean(error)),
        "max_absolute_error_mw": float(np.max(np.abs(error))) if len(error) else float("nan"),
        "matched_observations": int(len(actual)),
    }


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

    frame = load_station_csv(csv_path, interval_minutes=interval_minutes)
    n_rows = len(frame)
    split_row = int(n_rows * (1.0 - validation_fraction))
    if split_row <= history_steps + horizon_steps or n_rows - split_row < horizon_steps:
        raise ValueError(
            f"Not enough rows ({n_rows}) for history={history_steps}, horizon={horizon_steps} "
            "and chronological validation."
        )

    train_origins = np.arange(history_steps, split_row - horizon_steps + 1, dtype=np.int64)
    validation_origins = np.arange(split_row, n_rows - horizon_steps + 1, dtype=np.int64)
    train_origins = sample_origins(train_origins, int(parameters.get("max_train_origins", 300)))
    validation_origins = sample_origins(
        validation_origins, int(parameters.get("max_validation_origins", 80))
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

    validation_start = str(pd.Timestamp(frame.iloc[split_row]["utc_timestamp"]).isoformat())
    validation_end = str(pd.Timestamp(frame.iloc[-1]["utc_timestamp"]).isoformat())
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
        "training_origins": int(len(train_origins)),
        "training_samples": int(len(y_train)),
        "validation_origins": int(len(validation_origins)),
        "validation_origin_count": int(len(validation_origins)),
        "validation_samples": int(len(y_valid)),
        "training_start_utc": str(pd.Timestamp(frame.iloc[0]["utc_timestamp"]).isoformat()),
        "training_end_utc": str(pd.Timestamp(frame.iloc[split_row - 1]["utc_timestamp"]).isoformat()),
        "validation_start_utc": validation_start,
        "validation_end_utc": validation_end,
        "validation_fraction": validation_fraction,
        "evaluation_protocol": "chronological_holdout_sampled_origins_v1",
        "parameters": parameters,
        "metrics": metrics,
        "horizon_metrics": horizon_metrics,
        "training_duration_seconds": float(training_seconds),
        "estimator_fit_seconds": float(estimator_fit_seconds),
        "inference_latency_ms_per_origin": float(inference_latency_ms),
        "model_size_bytes": int(artifact_path.stat().st_size),
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
        "training_duration_seconds": float(training_seconds),
        "estimator_fit_seconds": float(estimator_fit_seconds),
        "inference_latency_ms_per_origin": float(inference_latency_ms),
        "model_size_bytes": int(artifact_path.stat().st_size),
        "created_at": created,
    }
    upsert_experiment(registry_record)
    return metadata
