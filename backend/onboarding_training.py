"""Advance registered stations through their initial model training queue."""
from __future__ import annotations

from fastapi import HTTPException

from backend.pipeline_manager import STATION_REGISTRY_LOCK, read_station_config, write_station_config
from src.models.model_registry import get_experiment

DEFAULT_MODELS = {
    "demand_forecast": "dense_nn",
    "solar_forecast": "lstm",
    "wind_forecast": "lstm",
    "residual_anomaly": "dense_autoencoder",
}
PENDING = {"queued", "running"}


def initial_training(components: list[str]) -> dict:
    return {component: {"algorithm": algorithm, "status": "queued"}
            for component, algorithm in DEFAULT_MODELS.items() if component in components}


def training_pending(entry: dict) -> bool:
    return any(task["status"] in PENDING for task in (entry.get("initial_training") or {}).values())


def advance_initial_training(manager) -> None:
    from backend.model_api import (
        DeployRequest, _deploy_model, component_algorithm_defaults, training_options_for, validate_parameters,
    )

    # Skip this tick if someone is registering, stopping or activating a station.
    if not STATION_REGISTRY_LOCK.acquire(blocking=False):
        return
    try:
        config = read_station_config()
        if manager.training_running():
            return
        stations = config.get("stations", {})
        task = next(((station, component, task) for station, entry in stations.items()
                     for component, task in (entry.get("initial_training") or {}).items()
                     if task["status"] == "running"), None)
        if task:
            station, component, item = task
            try:
                record = get_experiment(item["experiment_id"])
                if not record or record["status"] not in {"candidate", "deployed"}:
                    raise RuntimeError((record or {}).get("error") or "Training ended before a model was saved. Retry initial training.")
                current = stations[station].get("model_components", {}).get(component, {})
                if not current.get("active_experiment_id"):
                    _deploy_model(DeployRequest(station=station, experiment_id=item["experiment_id"]), manager)
                item["status"] = "ready"
            except Exception as exc:
                item.update(status="failed", error=_error_text(exc))
            # Deployment writes the registry as well. Preserve its new assignments.
            config = read_station_config()
            config["stations"][station]["initial_training"][component] = item
            write_station_config(config)
            return

        task = next(((station, component, task) for station, entry in stations.items()
                     for component, task in (entry.get("initial_training") or {}).items()
                     if task["status"] == "queued"), None)
        if not task:
            return
        station, component, item = task
        algorithm = item["algorithm"]
        try:
            options = training_options_for()
            defaults = component_algorithm_defaults(component, algorithm)
            if defaults is None:
                raise ValueError(f"No parameters are configured for {algorithm}/{component}.")
            parameters = validate_parameters(algorithm, {}, defaults)
            result = manager.start_model_experiment(station, algorithm, parameters, options, component=component)
            item.update(status="running", experiment_id=result["experiment_id"])
        except Exception as exc:
            item.update(status="failed", error=_error_text(exc))
        write_station_config(config)
    finally:
        STATION_REGISTRY_LOCK.release()


def _error_text(exc: Exception) -> str:
    return str(exc.detail) if isinstance(exc, HTTPException) else str(exc)
