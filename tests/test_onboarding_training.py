
from backend import onboarding_training, pipeline_manager, station_api
from src.models import model_registry


def test_initial_queue_deploys_and_retries_components(monkeypatch, tmp_path):
    study = "demo-substation"
    station_file = tmp_path / "stations.yaml"
    registry_file = tmp_path / "experiments.json"
    monkeypatch.setattr(pipeline_manager, "STATION_CONFIG", station_file)
    monkeypatch.setattr(model_registry, "REGISTRY_PATH", registry_file)
    plan = onboarding_training.initial_training(["demand_forecast", "solar_forecast", "wind_forecast", "residual_anomaly"])
    assert [(key, task["algorithm"]) for key, task in plan.items()] == [
        ("demand_forecast", "dense_nn"), ("solar_forecast", "lstm"), ("wind_forecast", "lstm"),
        ("residual_anomaly", "dense_autoencoder"),
    ]
    pipeline_manager.write_station_config({"stations": {study: {
        "model_components": {key: {"enabled": True} for key in plan},
        "initial_training": plan,
    }}})

    class TrainingManager:
        active = False
        started = []

        def training_running(self):
            return self.active

        def station_status(self, station):
            return {"services": {}}

        def start_model_experiment(self, station, algorithm, params, options, *, component):
            self.active = True
            experiment_id = f"{component}-{len(self.started)}"
            self.started.append(experiment_id)
            model_registry.upsert_experiment({
                "experiment_id": experiment_id, "station": station, "algorithm": algorithm,
                "component": component, "status": "running",
            })
            return {"experiment_id": experiment_id}

    manager = TrainingManager()
    advance = lambda: onboarding_training.advance_initial_training(manager)
    station = lambda: pipeline_manager.read_station_config()["stations"][study]
    advance()
    advance()
    assert manager.started == ["demand_forecast-0"]
    assert onboarding_training.training_pending(station())

    artifact = tmp_path / "model.joblib"
    metadata = tmp_path / "metadata.json"
    artifact.touch()
    metadata.write_text("{}")
    model_registry.upsert_experiment({
        "experiment_id": manager.started[-1], "status": "candidate",
        "artifact_path": str(artifact), "metadata_path": str(metadata), "version": "v1",
    })
    manager.active = False
    advance()
    assert station()["active_model"]["algorithm"] == "dense_nn"
    assert station()["initial_training"]["demand_forecast"]["status"] == "ready"
    advance()
    assert manager.started[-1] == "solar_forecast-1"

    manager.active = False
    model_registry.upsert_experiment({"experiment_id": manager.started[-1], "status": "failed", "error": "Training failed"})
    advance()
    assert station()["initial_training"]["solar_forecast"]["error"] == "Training failed"
    assert station_api.retry_initial_training(study)["queued"] == 1
    assert station()["initial_training"]["solar_forecast"] == {"algorithm": "lstm", "status": "queued"}
    assert station()["initial_training"]["demand_forecast"]["status"] == "ready"
