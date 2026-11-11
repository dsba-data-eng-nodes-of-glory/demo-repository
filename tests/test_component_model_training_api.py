"""Station/component training and deployment regression checks."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from fastapi import HTTPException

from backend import model_api


def _station_cfg() -> dict:
    return {
        "active_station": "new-station",
        "stations": {
            "new-station": {
                "model_checkpoint": "models/legacy.pt",
                "active_model": {"experiment_id": "demand-original", "artifact_path": "models/legacy.pt"},
                "model_components": {
                    "demand_forecast": {"enabled": True, "training_file": "data/demand.csv"},
                    "solar_forecast": {"enabled": True, "training_file": "data/solar.csv"},
                    "wind_forecast": {"enabled": True, "training_file": "data/wind.csv"},
                    "residual_anomaly": {"enabled": True, "training_file": "data/residual.csv"},
                },
                "renewables": {"solar": {"enabled": True}, "wind": {"enabled": True}},
            }
        },
    }


def test_components_have_five_algorithms_for_forecasting_and_only_ae_for_anomaly(monkeypatch):
    monkeypatch.setattr(model_api, "read_station_config", _station_cfg)
    response = model_api.model_components("new-station")
    for component in ("demand_forecast", "solar_forecast", "wind_forecast"):
        assert set(response["component_algorithms"][component]) == {
            "dense_nn", "lstm", "random_forest", "xgboost", "gbt"
        }
        assert response["enabled"][component] is True
    assert response["component_algorithms"]["residual_anomaly"] == ["dense_autoencoder"]
    assert response["enabled"]["residual_anomaly"] is True


def test_shared_renewable_profile_is_saved_once(monkeypatch, tmp_path):
    config_file = tmp_path / "algorithm_configs.yaml"
    config_file.write_text(model_api.ALGORITHM_CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(model_api, "ALGORITHM_CONFIG_PATH", config_file)
    old = model_api.component_algorithm_defaults("solar_forecast", "random_forest")
    assert old == model_api.component_algorithm_defaults("wind_forecast", "random_forest")
    response = model_api.update_component_config(
        "wind_forecast", "random_forest",
        model_api.ConfigUpdateRequest(parameters={"n_estimators": 17}),
    )
    assert response["shared_across_solar_and_wind"] is True
    assert model_api.component_algorithm_defaults("wind_forecast", "random_forest")["n_estimators"] == 17
    assert model_api.component_algorithm_defaults("solar_forecast", "random_forest")["n_estimators"] == 17
    assert model_api.component_algorithm_defaults("demand_forecast", "random_forest")["n_estimators"] != 17


def test_start_training_component_restrictions(monkeypatch):
    cfg = _station_cfg()
    cfg["stations"]["new-station"]["model_components"]["solar_forecast"]["enabled"] = False
    monkeypatch.setattr(model_api, "read_station_config", lambda: deepcopy(cfg))

    class StubManager:
        def start_model_experiment(self, *args, **kwargs):
            return {"args": args, "kwargs": kwargs}

    monkeypatch.setattr(model_api, "manager", StubManager())
    with pytest.raises(HTTPException) as exc:
        model_api.start_experiment(model_api.ExperimentRequest(
            station="new-station", component="solar_forecast", algorithm="dense_nn"
        ))
    assert exc.value.status_code == 409
    with pytest.raises(HTTPException) as exc:
        model_api.start_experiment(model_api.ExperimentRequest(
            station="new-station", component="residual_anomaly", algorithm="lstm"
        ))
    assert exc.value.status_code == 422
    success = model_api.start_experiment(model_api.ExperimentRequest(
        station="new-station", component="wind_forecast", algorithm="lstm"
    ))
    assert success["kwargs"]["component"] == "wind_forecast"
    assert success["args"][0:2] == ("new-station", "lstm")


def test_deploy_wind_preserves_demand_and_solar_assignments(monkeypatch, tmp_path):
    cfg = _station_cfg()
    original = deepcopy(cfg["stations"]["new-station"]["active_model"])
    artifact = tmp_path / "wind.joblib"
    meta = tmp_path / "wind.json"
    artifact.write_bytes(b"test-model")
    meta.write_text("{}", encoding="utf-8")
    record = {
        "experiment_id": "wind-exp-1", "station": "new-station", "component": "wind_forecast",
        "status": "candidate", "algorithm": "gbt", "version": "v1",
        "artifact_path": str(artifact), "metadata_path": str(meta),
    }
    updates = []
    monkeypatch.setattr(model_api, "read_station_config", lambda: deepcopy(cfg))
    monkeypatch.setattr(model_api, "write_station_config", lambda new: (cfg.clear(), cfg.update(deepcopy(new))))
    monkeypatch.setattr(model_api, "get_experiment", lambda _: deepcopy(record))
    monkeypatch.setattr(model_api, "list_experiments", lambda station: [deepcopy(record)])
    monkeypatch.setattr(model_api, "upsert_experiment", lambda r: updates.append(deepcopy(r)))
    monkeypatch.setattr(model_api, "resolve_root_path", Path)

    class NoPredictor:
        def station_status(self, _):
            return {"services": {"predictor": {"running": False}}}

    monkeypatch.setattr(model_api, "manager", NoPredictor())
    response = model_api.deploy_model(model_api.DeployRequest(station="new-station", experiment_id="wind-exp-1"))
    entry = cfg["stations"]["new-station"]
    assert response["component"] == "wind_forecast"
    assert entry["active_model"] == original
    assert entry["renewables"]["wind"]["checkpoint"] == str(artifact)
    assert entry["renewables"]["solar"] == {"enabled": True}
    assert entry["model_components"]["wind_forecast"]["active_experiment_id"] == "wind-exp-1"
    assert updates[-1]["status"] == "deployed"


def test_component_versioning_isolated(monkeypatch, tmp_path):
    from src.models import model_registry
    monkeypatch.setattr(model_registry, "REGISTRY_PATH", tmp_path / "experiments.json")
    model_registry.upsert_experiment({
        "experiment_id": "solar-1", "station": "station-x", "component": "solar_forecast",
        "algorithm": "dense_nn", "version": "v1", "status": "candidate",
    })
    model_registry.upsert_experiment({
        "experiment_id": "wind-1", "station": "station-x", "component": "wind_forecast",
        "algorithm": "dense_nn", "version": "v3", "status": "candidate",
    })
    assert model_registry.next_version("station-x", "dense_nn", "solar_forecast") == "v2"
    assert model_registry.next_version("station-x", "dense_nn", "wind_forecast") == "v4"
    assert model_registry.next_version("station-x", "dense_nn", "demand_forecast") == "v1"


def test_comparison_uses_mwh_for_renewables(monkeypatch, tmp_path):
    records = {
        f"exp-{i}": {
            "experiment_id": f"exp-{i}", "station": "station-x", "component": "solar_forecast",
            "algorithm": "gbt", "version": f"v{i}", "status": "candidate",
            "artifact_path": "not-needed.joblib", "metrics": {"mae_mwh": i * 10., "rmse_mwh": i * 12., "bias_mwh": -1., "wape_pct": 4.},
        } for i in (1, 2)
    }
    monkeypatch.setattr(model_api, "get_experiment", records.get)
    response = model_api.compare_experiments(model_api.CompareRequest(experiment_ids=["exp-1", "exp-2"]))
    assert response["comparable"] is True
    assert response["results"][0]["MAE (MWh)"] == 10.
    assert response["results"][0]["RMSE (MWh)"] == 12.
    assert response["results"][0]["Bias (MWh)"] == -1.
    assert "MAE (MW)" not in response["results"][0]
