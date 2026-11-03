"""Model experiments, editable algorithm settings, registry and deployment APIs."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Literal

import yaml
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.pipeline_manager import (
    ROOT,
    manager,
    read_station_config,
    resolve_root_path,
    write_station_config,
)
from src.models.model_registry import get_experiment, list_experiments, upsert_experiment

router = APIRouter(prefix="/models", tags=["models"])
ALGORITHM_CONFIG_PATH = ROOT / "config" / "algorithm_configs.yaml"
ALGORITHMS = {
    "lstm": {"label": "LSTM", "family": "deep_learning", "description": "Sequence-to-sequence encoder-decoder LSTM."},
    "random_forest": {"label": "Random Forest", "family": "tree_ensemble", "description": "Bagged decision trees using lag, rolling and calendar features."},
    "gbt": {"label": "Gradient Boosted Trees", "family": "tree_boosting", "description": "Histogram gradient-boosted trees using tabular forecasting features."},
    "xgboost": {"label": "XGBoost", "family": "tree_boosting", "description": "Extreme gradient boosting; requires the xgboost package."},
}


class ExperimentRequest(BaseModel):
    station: str
    algorithm: Literal["lstm", "random_forest", "gbt", "xgboost"]
    parameters: dict[str, Any] = Field(default_factory=dict)
    training_options: dict[str, Any] = Field(default_factory=dict)


class CompareRequest(BaseModel):
    experiment_ids: list[str]


class DeployRequest(BaseModel):
    station: str
    experiment_id: str


class ConfigUpdateRequest(BaseModel):
    parameters: dict[str, Any]
    training_options: dict[str, Any] = Field(default_factory=dict)


def read_algorithm_config() -> dict[str, Any]:
    if not ALGORITHM_CONFIG_PATH.exists():
        return {"algorithms": {}}
    with ALGORITHM_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {"algorithms": {}}


def write_algorithm_config(config: dict[str, Any]) -> None:
    ALGORITHM_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp = ALGORITHM_CONFIG_PATH.with_suffix(".yaml.tmp")
    temp.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    temp.replace(ALGORITHM_CONFIG_PATH)


def validate_parameters(algorithm: str, supplied: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    unknown = set(supplied) - set(defaults)
    if unknown:
        raise HTTPException(status_code=422, detail=f"Unsupported {algorithm} parameter(s): {sorted(unknown)}")
    merged = {**defaults, **supplied}
    for name, value in merged.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            if not math.isfinite(float(value)):
                raise HTTPException(status_code=422, detail=f"Parameter '{name}' must be finite.")
            if value < 0:
                raise HTTPException(status_code=422, detail=f"Parameter '{name}' cannot be negative.")
    # Conservative guards for this CPU-oriented POC.
    caps = {
        "epochs": 200, "batch_size": 256, "hidden_size": 256,
        "n_estimators": 1500, "max_iter": 500, "max_depth": 20,
        "max_train_windows": 2000, "max_validation_windows": 500,
        "max_train_origins": 1200, "max_validation_origins": 300,
    }
    for name, cap in caps.items():
        if name in merged and float(merged[name]) > cap:
            raise HTTPException(status_code=422, detail=f"Parameter '{name}' cannot exceed {cap} in this POC.")
    strictly_positive = {
        "epochs", "batch_size", "hidden_size", "n_estimators", "max_iter",
        "max_depth", "max_leaf_nodes", "min_samples_leaf", "max_train_windows",
        "max_validation_windows", "max_train_origins", "max_validation_origins", "n_jobs",
        "learning_rate",
    }
    for name in strictly_positive:
        if name in merged and float(merged[name]) <= 0:
            raise HTTPException(status_code=422, detail=f"Parameter '{name}' must be greater than zero.")
    for name in ("subsample", "colsample_bytree"):
        if name in merged and not 0.1 <= float(merged[name]) <= 1.0:
            raise HTTPException(status_code=422, detail=f"Parameter '{name}' must be between 0.1 and 1.0.")
    return merged


@router.get("/status")
def model_status(station: str | None = None):
    config = read_station_config()
    station = station or config.get("active_station")
    stations = config.get("stations", {})
    if station not in stations:
        raise HTTPException(status_code=404, detail=f"Unknown station: {station}")

    entry = stations[station]
    active_model = entry.get("active_model") or {}
    checkpoint_value = active_model.get("artifact_path") or entry.get("model_checkpoint")
    checkpoint_path = resolve_root_path(checkpoint_value or "__missing_model__")
    metadata_value = active_model.get("metadata_path") or entry.get("model_metadata")
    metadata_path = resolve_root_path(metadata_value) if metadata_value else checkpoint_path.parent / "metadata.json"
    metadata = None
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            metadata = {"warning": "Model metadata exists but could not be parsed."}

    training = manager.station_status(station)["services"]["training"]
    return {
        "station": station,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_exists": checkpoint_path.is_file(),
        "metadata_path": str(metadata_path),
        "metadata_exists": metadata_path.is_file(),
        "metadata": metadata,
        "training_process": training,
        "model_version": active_model.get("version") or entry.get("model_version"),
        "model_algorithm": active_model.get("algorithm", "lstm" if checkpoint_path.suffix == ".pt" else None),
        "active_model": active_model or None,
        "model_status": entry.get("status", "unknown"),
    }


@router.get("/algorithms")
def list_algorithms():
    config = read_algorithm_config().get("algorithms", {})
    return {
        "algorithms": [
            {"key": key, **metadata, "defaults": config.get(key, {})}
            for key, metadata in ALGORITHMS.items()
        ]
    }


@router.get("/config")
def get_algorithm_configs():
    return read_algorithm_config()


@router.put("/config/{algorithm}")
def update_algorithm_config(algorithm: str, request: ConfigUpdateRequest):
    if algorithm not in ALGORITHMS:
        raise HTTPException(status_code=404, detail=f"Unknown algorithm: {algorithm}")
    config = read_algorithm_config()
    algorithm_configs = config.setdefault("algorithms", {})
    defaults = algorithm_configs.get(algorithm, {})
    merged = validate_parameters(algorithm, request.parameters, defaults)
    algorithm_configs[algorithm] = merged
    options = request.training_options or {}
    if options:
        fraction = float(options.get("validation_fraction", 0.15))
        if not 0.05 <= fraction <= 0.40:
            raise HTTPException(status_code=422, detail="validation_fraction must be between 0.05 and 0.40.")
        config.setdefault("training_options", {})["validation_fraction"] = fraction
    write_algorithm_config(config)
    return {
        "algorithm": algorithm,
        "parameters": merged,
        "training_options": config.get("training_options", {"validation_fraction": 0.15}),
        "saved": True,
    }


@router.post("/experiments/train")
def start_experiment(request: ExperimentRequest):
    station_config = read_station_config().get("stations", {})
    if request.station not in station_config:
        raise HTTPException(status_code=404, detail=f"Unknown station: {request.station}")
    algorithm_configs = read_algorithm_config().get("algorithms", {})
    defaults = algorithm_configs.get(request.algorithm)
    if defaults is None:
        raise HTTPException(status_code=500, detail=f"No default configuration found for {request.algorithm}.")
    parameters = validate_parameters(request.algorithm, request.parameters, defaults)
    training_options = request.training_options or read_algorithm_config().get("training_options", {"validation_fraction": 0.15})
    fraction = float(training_options.get("validation_fraction", 0.15))
    if not 0.05 <= fraction <= 0.40:
        raise HTTPException(status_code=422, detail="validation_fraction must be between 0.05 and 0.40.")
    training_options["validation_fraction"] = fraction
    try:
        return manager.start_model_experiment(
            request.station, request.algorithm, parameters, training_options
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/experiments")
def experiments(station: str | None = None):
    return {"experiments": list_experiments(station)}


@router.get("/experiments/{experiment_id}")
def experiment_detail(experiment_id: str):
    record = get_experiment(experiment_id)
    if not record:
        raise HTTPException(status_code=404, detail="Experiment not found.")
    metadata_path = record.get("metadata_path")
    if metadata_path:
        path = resolve_root_path(metadata_path)
        if path.is_file():
            try:
                record["metadata"] = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                record["metadata_warning"] = "Metadata file could not be read."
    return record


@router.post("/compare")
def compare_experiments(request: CompareRequest):
    if not 2 <= len(request.experiment_ids) <= 4:
        raise HTTPException(status_code=422, detail="Select between two and four experiments.")
    if len(set(request.experiment_ids)) != len(request.experiment_ids):
        raise HTTPException(status_code=422, detail="Select each experiment only once.")
    records = []
    for experiment_id in request.experiment_ids:
        record = get_experiment(experiment_id)
        if not record:
            raise HTTPException(status_code=404, detail=f"Experiment not found: {experiment_id}")
        if record.get("status") not in ("candidate", "deployed", "previous"):
            raise HTTPException(status_code=409, detail=f"Experiment {experiment_id} is not complete (status={record.get('status')}).")
        records.append(record)

    signature_fields = (
        "station", "history_steps", "horizon_steps", "interval_minutes",
        "validation_start_utc", "validation_end_utc", "validation_origin_count", "evaluation_protocol",
    )
    signatures = [tuple(record.get(field) for field in signature_fields) for record in records]
    comparable = all(signature == signatures[0] for signature in signatures[1:])
    warnings = []
    if not comparable:
        warnings.append("Experiments differ in station, forecast dimensions, validation period or evaluation protocol. Treat metric comparisons as indicative, not a controlled benchmark.")

    rows = []
    for record in records:
        metrics = record.get("metrics") or {}
        metadata = {}
        metadata_value = record.get("metadata_path")
        if metadata_value:
            metadata_path = resolve_root_path(metadata_value)
            if metadata_path.is_file():
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                except Exception:
                    metadata = {}
        rows.append({
            "experiment_id": record.get("experiment_id"),
            "model": f"{ALGORITHMS.get(record.get('algorithm'), {}).get('label', record.get('algorithm'))} {record.get('version', '')}".strip(),
            "algorithm": record.get("algorithm"),
            "station": record.get("station"),
            "status": record.get("status"),
            "MAE (MW)": metrics.get("mae_mw"),
            "RMSE (MW)": metrics.get("rmse_mw"),
            "WAPE (%)": metrics.get("wape_pct"),
            "Bias (MW)": metrics.get("bias_mw"),
            "Training time (s)": record.get("training_duration_seconds"),
            "Inference latency (ms/origin)": record.get("inference_latency_ms_per_origin"),
            "Model size (bytes)": record.get("model_size_bytes"),
            "Parameters": record.get("parameters", {}),
            "horizon_metrics": metadata.get("horizon_metrics", []),
            "validation_origin_count": metadata.get("validation_origins", metadata.get("validation_windows")),
            "feature_schema": record.get("feature_schema"),
        })
    return {"comparable": comparable, "warnings": warnings, "signature_fields": list(signature_fields), "results": rows}


@router.get("/registry")
def model_registry(station: str | None = None):
    records = list_experiments(station)
    config = read_station_config().get("stations", {})
    deployed = {
        station_id: (entry.get("active_model") or {}).get("experiment_id")
        for station_id, entry in config.items()
    }
    return {
        "experiments": records,
        "deployments": deployed,
    }


@router.post("/deploy")
def deploy_model(request: DeployRequest):
    config = read_station_config()
    stations = config.get("stations", {})
    if request.station not in stations:
        raise HTTPException(status_code=404, detail=f"Unknown station: {request.station}")
    record = get_experiment(request.experiment_id)
    if not record:
        raise HTTPException(status_code=404, detail="Experiment not found.")
    if record.get("station") != request.station:
        raise HTTPException(status_code=422, detail="The selected experiment belongs to a different station.")
    if record.get("status") not in ("candidate", "deployed", "previous"):
        raise HTTPException(status_code=409, detail="Only completed experiments can be deployed.")
    artifact_path = resolve_root_path(record.get("artifact_path", "__missing_artifact__"))
    metadata_path = resolve_root_path(record.get("metadata_path", "__missing_metadata__"))
    if not artifact_path.is_file() or not metadata_path.is_file():
        raise HTTPException(status_code=409, detail="Model artifact or metadata is missing; deployment was not changed.")

    process_status = manager.station_status(request.station).get("services", {})
    if process_status.get("predictor", {}).get("running"):
        raise HTTPException(
            status_code=409,
            detail="Stop this station's predictor before changing its model assignment. Restart the pipeline after deployment so the predictor loads the approved version.",
        )

    previous = stations[request.station].get("active_model") or {}
    stations[request.station]["active_model"] = {
        "experiment_id": record["experiment_id"],
        "algorithm": record["algorithm"],
        "version": record.get("version"),
        "artifact_path": record["artifact_path"],
        "metadata_path": record["metadata_path"],
        "feature_schema": record.get("feature_schema"),
        "deployed_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
    }
    stations[request.station]["model_version"] = record.get("version")
    stations[request.station]["model_algorithm"] = record["algorithm"]
    stations[request.station]["status"] = "model_deployed"
    write_station_config(config)

    for item in list_experiments(request.station):
        if item.get("status") == "deployed" and item.get("experiment_id") != record["experiment_id"]:
            item["status"] = "previous"
            upsert_experiment(item)
    record["status"] = "deployed"
    upsert_experiment(record)
    return {
        "station": request.station,
        "deployed_experiment_id": record["experiment_id"],
        "algorithm": record["algorithm"],
        "version": record.get("version"),
        "previous_experiment_id": previous.get("experiment_id"),
        "restart_required": True,
        "message": "Model assignment saved. Start the station pipeline to load this model version.",
    }
