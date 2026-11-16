from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from backend import model_api
from src.models import model_registry


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setattr(model_api, "ROOT", tmp_path)
    monkeypatch.setattr(model_registry, "REGISTRY_PATH", tmp_path / "experiments.json")
    stations = {"alpha": {}, "beta": {}}
    monkeypatch.setattr(model_api, "read_station_config", lambda: {"stations": stations})
    app = FastAPI()
    app.include_router(model_api.router)
    yield TestClient(app), tmp_path, stations


def make_version(root, key, status="candidate"):
    folder = root / "models" / "alpha" / "experiments" / key
    folder.mkdir(parents=True)
    artifact = folder / "model.pt"
    artifact.write_bytes(b"trained-model")
    model_registry.upsert_experiment({
        "experiment_id": key, "station": "alpha", "component": "demand_forecast",
        "algorithm": "dense_nn", "status": status, "artifact_path": str(artifact), "version": "v2",
    })
    return artifact


def test_removes_only_selected_version_and_keeps_version_sequence(project):
    client, root, _ = project
    artifact = make_version(root, "obsolete", "previous")
    remaining = make_version(root, "live", "deployed")
    dataset = root / "data.csv"
    dataset.write_bytes(b"source")
    model_registry.upsert_experiment({"experiment_id": "obsolete", "training_data_path": str(dataset)})
    assert client.get("/models/registry?station=alpha").json()["deletable_experiment_ids"] == ["obsolete"]
    assert client.delete("/models/experiments/obsolete?station=beta").status_code == 404
    assert client.delete("/models/experiments/obsolete?station=alpha").json() == {
        "deleted": True, "experiment_id": "obsolete", "files_removed": True,
    }
    assert not artifact.parent.exists()
    assert remaining.is_file() and dataset.is_file()
    assert client.get("/models/experiments/obsolete").status_code == 404
    assert client.delete("/models/experiments/obsolete?station=alpha").status_code == 404
    assert model_registry.next_version("alpha", "dense_nn") == "v3"


def test_protects_assignments_shared_files_and_training(project):
    client, root, stations = project
    active = make_version(root, "stale-status")
    ongoing = make_version(root, "training", "running")
    shared = make_version(root, "shared")
    pending = make_version(root, "auto-pending")
    stations["alpha"]["active_model"] = {"experiment_id": "stale-status"}
    stations["alpha"]["initial_training"] = {
        "demand_forecast": {"status": "running", "experiment_id": "auto-pending"},
    }
    stations["beta"]["renewables"] = {"wind": {"checkpoint": str(shared)}}
    for key in ("stale-status", "training", "shared", "auto-pending"):
        assert client.delete(f"/models/experiments/{key}?station=alpha").status_code == 409
    assert all(path.is_file() for path in (active, ongoing, shared, pending))


def test_does_not_delete_external_artifact(project):
    client, root, _ = project
    artifact = make_version(root, "old-version")
    external = root / "external-model.pt"
    external.touch()
    model_registry.upsert_experiment({"experiment_id": "old-version", "artifact_path": str(external)})
    assert client.delete("/models/experiments/old-version?station=alpha").status_code == 409
    assert artifact.is_file() and external.is_file()
