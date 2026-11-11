"""Regression tests for legacy Amprion and immutable component training datasets."""
from __future__ import annotations

import asyncio
from copy import deepcopy
from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest
import yaml
from fastapi import HTTPException
from starlette.datastructures import UploadFile

from backend import model_api, training_data


def station_entry(tmp_path: Path):
    return {
        "interval_minutes": 15, "history_steps": 672, "horizon_steps": 96,
        "history_file": "data/training/amprion/demand_train.csv",
        "input_file": "data/training/amprion/demand_test.csv",
        "model_checkpoint": "models/amprion/demand.pt",
        "residual_model_checkpoint": "models/amprion/residual_anomaly/model.joblib",
        "renewables": {
            "enabled": True,
            "solar": {"train_file": "data/training/amprion/solar_train.csv", "checkpoint": "models/amprion/solar.pt"},
            "wind": {"train_file": "data/training/amprion/wind_train.csv", "checkpoint": "models/amprion/wind.pt"},
        },
    }


def _demand_csv(rows: int = 1300, missing_temp: bool = False) -> bytes:
    ts = pd.date_range("2020-01-01", periods=rows, freq="15min", tz="UTC")
    payload = {"utc_timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"), "demand": [1000.0] * rows}
    if not missing_temp:
        payload["temperature"] = [5.0] * rows
    return pd.DataFrame(payload).to_csv(index=False).encode("utf-8")


def _wind_csv(rows: int = 1400, bad_speed: bool = False) -> bytes:
    ts = pd.date_range("2020-01-01", periods=rows, freq="15min", tz="UTC")
    payload = pd.DataFrame({
        "utc_timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "Wind onshore [MWh]": [750.] * rows,
        "wind_speed_100m_mean_ms": [8.] * rows,
        "wind_direction_100m_circular_mean_deg": [200.] * rows,
        "Wind_onshore_installed_capacity [MW]": [19000.] * rows,
        # Legacy Amprion forecast names, accepted as aliases.
        "forecast_wind_speed_100m_mean_ms": [-10. if bad_speed else 8.] * rows,
        "forecast_wind_direction_100m_circular_mean_deg": [220.] * rows,
    })
    return payload.to_csv(index=False).encode()


def test_legacy_station_infers_all_four_without_reonboarding(monkeypatch, tmp_path):
    entry = station_entry(tmp_path)
    registry = {"active_station": "amprion", "stations": {"amprion": entry}}
    monkeypatch.setattr(model_api, "read_station_config", lambda: deepcopy(registry))
    monkeypatch.setattr(model_api, "resolve_root_path", lambda value: tmp_path / value)
    result = model_api.model_components("amprion")
    assert result["legacy_station"] is True
    assert all(result["enabled"].values())
    assert result["sources"]["demand_forecast"]["registered_file"] == entry["history_file"]
    assert result["sources"]["solar_forecast"]["registered_file"].endswith("solar_train.csv")
    assert result["sources"]["wind_forecast"]["registered_file"].endswith("wind_train.csv")
    assert result["sources"]["residual_anomaly"]["registered_file"].endswith("demand_train.csv")
    assert result["component_algorithms"]["residual_anomaly"] == ["dense_autoencoder"]


def test_explicit_disabled_component_stays_disabled():
    entry = station_entry(Path("."))
    entry["model_components"] = {
        "demand_forecast": {"enabled": True},
        "solar_forecast": {"enabled": False},
        "wind_forecast": {"enabled": True},
        "residual_anomaly": {"enabled": False},
    }
    enabled = training_data.station_component_state(entry)
    assert enabled == {"demand_forecast": True, "solar_forecast": False,
                       "wind_forecast": True, "residual_anomaly": False}


def test_new_upload_isolated_by_station_and_component(tmp_path):
    entry = station_entry(tmp_path)
    original_registry_file = tmp_path / entry["history_file"]
    original_registry_file.parent.mkdir(parents=True)
    original_registry_file.write_text("old-data", encoding="utf-8")
    content = _demand_csv()
    upload = UploadFile(filename="fresh-dataset.csv", file=BytesIO(content))
    meta = asyncio.run(training_data.store_source(
        root=tmp_path, station="amprion", component="demand_forecast", entry=entry,
        file=upload, source_timezone="UTC",
    ))
    assert meta["rows"] == 1300
    assert meta["sha256"]
    assert meta["source_id"] in meta["path"]
    saved = training_data.load_source(tmp_path, "amprion", "demand_forecast", meta["source_id"])
    assert saved["rows"] == 1300
    assert saved["path"] == meta["path"]
    assert original_registry_file.read_text(encoding="utf-8") == "old-data"
    assert len(training_data.list_sources(tmp_path, "amprion", "demand_forecast")) == 1
    with pytest.raises(HTTPException) as e:
        training_data.load_source(tmp_path, "tennet", "demand_forecast", meta["source_id"])
    assert e.value.status_code == 404
    with pytest.raises(HTTPException) as e:
        training_data.load_source(tmp_path, "amprion", "solar_forecast", meta["source_id"])
    assert e.value.status_code == 404
    with pytest.raises(HTTPException) as e:
        training_data.load_source(tmp_path, "amprion", "demand_forecast", "../../bad")
    assert e.value.status_code == 422


def test_uploaded_file_change_blocks_retraining(tmp_path):
    entry = station_entry(tmp_path)
    meta = asyncio.run(training_data.store_source(
        root=tmp_path, station="amprion", component="demand_forecast", entry=entry,
        file=UploadFile(filename="data.csv", file=BytesIO(_demand_csv())),
    ))
    path = tmp_path / meta["path"]
    with path.open("a", encoding="utf-8") as out:
        out.write("changed\n")
    with pytest.raises(HTTPException) as exc:
        training_data.load_source(tmp_path, "amprion", "demand_forecast", meta["source_id"])
    assert exc.value.status_code == 409


def test_wind_legacy_column_alias_normalized(tmp_path):
    entry = station_entry(tmp_path)
    info = asyncio.run(training_data.store_source(
        root=tmp_path, station="amprion", component="wind_forecast", entry=entry,
        file=UploadFile(filename="wind.csv", file=BytesIO(_wind_csv())),
    ))
    frame = pd.read_csv(tmp_path / info["path"])
    assert "future_wind_speed" in frame
    assert "future_wind_direction" in frame
    assert "forecast_wind_speed_100m_mean_ms" not in frame


def test_wind_out_of_range_rejected_and_no_snapshot_left(tmp_path):
    entry = station_entry(tmp_path)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(training_data.store_source(
            root=tmp_path, station="amprion", component="wind_forecast", entry=entry,
            file=UploadFile(filename="bad_wind.csv", file=BytesIO(_wind_csv(bad_speed=True))),
        ))
    assert exc.value.status_code == 422
    assert training_data.list_sources(tmp_path, "amprion", "wind_forecast") == []


def test_api_retraining_passes_validated_dataset(monkeypatch, tmp_path):
    entry = station_entry(tmp_path)
    cfg = {"stations": {"amprion": entry}}
    monkeypatch.setattr(model_api, "ROOT", tmp_path)
    monkeypatch.setattr(model_api, "read_station_config", lambda: cfg)
    meta = asyncio.run(training_data.store_source(
        root=tmp_path, station="amprion", component="demand_forecast", entry=entry,
        file=UploadFile(filename="new.csv", file=BytesIO(_demand_csv())),
    ))
    class StubManager:
        def start_model_experiment(self, *args, **kwargs):
            return {"args": args, "kwargs": kwargs}
    monkeypatch.setattr(model_api, "manager", StubManager())
    result = model_api.start_experiment(model_api.ExperimentRequest(
        station="amprion", component="demand_forecast", algorithm="random_forest",
        training_data_id=meta["source_id"],
    ))
    assert result["kwargs"]["training_data_id"] == meta["source_id"]
    assert result["kwargs"]["training_data_path"] == meta["path"]
    assert result["kwargs"]["component"] == "demand_forecast"
    with pytest.raises(HTTPException) as exc:
        model_api.start_experiment(model_api.ExperimentRequest(
            station="amprion", component="solar_forecast", algorithm="dense_nn",
            training_data_id=meta["source_id"],
        ))
    assert exc.value.status_code == 404


def test_legacy_manager_prefers_train_history_over_short_stream(tmp_path, monkeypatch):
    # Only resolve choice, since starting subprocess would execute a full model run.
    entry = station_entry(tmp_path)
    assert training_data.registered_source(entry, "demand_forecast") == "data/training/amprion/demand_train.csv"


def test_contrast_css_and_shared_selector_present():
    root = Path(__file__).resolve().parents[1]
    style = (root / "dashboard" / "ui.py").read_text(encoding="utf-8")
    model_ui = (root / "dashboard" / "pages" / "model_management.py").read_text(encoding="utf-8")
    pipeline_ui = (root / "dashboard" / "pages" / "pipeline_control.py").read_text(encoding="utf-8")
    assert '[data-testid="stWidgetLabel"] p' in style
    assert '[data-testid="stExpander"] details > summary' in style
    assert "Model component to manage / retrain" in model_ui
    assert '/models/training-data/upload' in model_ui
    assert "render_comparison(station, selected_component)" in model_ui
    assert "Station model components" in pipeline_ui


def test_upload_solar_workbook_for_selected_sheet(tmp_path):
    import numpy as np
    entry = station_entry(tmp_path)
    ts = pd.date_range("2020-01-01", periods=1400, freq="15min", tz="UTC")
    template = pd.DataFrame({
        "utc_timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "Photovoltaics [MWh]": np.abs(np.sin(np.arange(len(ts))/60.)) * 2000,
        "cloud_cover_mean_pct": [50.] * len(ts),
        "shortwave_radiation_mean_prev_hour_w_m2": [300.] * len(ts),
        "forecast_cloud_cover_mean_pct": [45.] * len(ts),
        "forecast_shortwave_radiation_mean_prev_hour_w_m2": [310.] * len(ts),
        "Photovoltaics_installed_capacity [MW]": [12500.] * len(ts),
    })
    workbook = tmp_path / "solar.xlsx"
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        template.to_excel(writer, sheet_name="Solar", index=False)
    result = asyncio.run(training_data.store_source(
        root=tmp_path, station="amprion", component="solar_forecast", entry=entry,
        file=UploadFile(filename="solar.xlsx", file=BytesIO(workbook.read_bytes())),
    ))
    assert result["rows"] == 1400
    normalized = pd.read_csv(tmp_path / result["path"])
    assert list(normalized.columns) == list(template.columns)
    assert training_data.load_source(tmp_path, "amprion", "solar_forecast", result["source_id"])["path"] == result["path"]


def test_pipeline_manager_passes_new_dataset_path_to_experiment(monkeypatch, tmp_path):
    from backend import pipeline_manager
    from src.models import model_registry
    entry = station_entry(tmp_path)
    cfg = {"stations": {"amprion": entry}}
    csv_file = tmp_path / "data" / "training_uploads" / "amprion" / "solar_forecast" / "snapshot" / "data.csv"
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    csv_file.write_text("utc_timestamp,Photovoltaics [MWh]\n2020-01-01T00:00:00Z,0\n", encoding="utf-8")
    relative = csv_file.relative_to(tmp_path).as_posix()
    monkeypatch.setattr(pipeline_manager, "ROOT", tmp_path)
    monkeypatch.setattr(pipeline_manager, "read_station_config", lambda: cfg)
    monkeypatch.setattr(pipeline_manager, "resolve_root_path", lambda item: tmp_path / item)
    monkeypatch.setattr(model_registry, "REGISTRY_PATH", tmp_path / "registry.json")
    pipeline = pipeline_manager.PipelineManager()
    monkeypatch.setattr(pipeline, "start", lambda service, command, station: {"started": True, "pid": 123, "log": "mock.log"})
    result = pipeline.start_model_experiment(
        "amprion", "dense_nn", {"epochs": 1}, {"validation_fraction": 0.15},
        component="solar_forecast", training_data_path=relative, training_data_id="custom1",
    )
    exp = tmp_path / "models" / "amprion" / "experiments" / result["experiment_id"] / "experiment.json"
    saved = __import__("json").loads(exp.read_text(encoding="utf-8"))
    assert saved["training_data_path"] == relative
    assert saved["training_data_id"] == "custom1"
    assert saved["training_data_sha256"] == training_data.digest_file(csv_file)
    assert result["component"] == "solar_forecast"


def test_residual_trainer_consumes_aligned_three_model_data(monkeypatch, tmp_path):
    import json
    from src.models import run_experiment
    csv_path=tmp_path / 'data' / 'residual_training.csv'
    csv_path.parent.mkdir(parents=True,exist_ok=True)
    dates=pd.date_range('2020-01-01',periods=96*31,freq='15min',tz='UTC')
    pd.DataFrame({'utc_timestamp':dates.strftime('%Y-%m-%dT%H:%M:%SZ'),
       'demand':2000.,'Photovoltaics [MWh]':100.,'Wind onshore [MWh]':300.,
       'Photovoltaics_installed_capacity [MW]':5000.,
       'Wind_onshore_installed_capacity [MW]':5000.,
       'Residual Load':1600.}).to_csv(csv_path,index=False)
    config_dir=tmp_path / 'config';config_dir.mkdir()
    (config_dir/'stations.yaml').write_text(yaml.safe_dump({'stations':{'amprion':{
        'interval_minutes':15,'residual_model_checkpoint':'models/amprion/residual_anomaly/model.pt',
        'residual_demand_unit':'mwh_per_interval',
    }}}),encoding='utf-8')
    monkeypatch.setattr(run_experiment,'ROOT',tmp_path)
    def fake_train(command,cwd,check):
        assert 'src.models.residual.train_residual_ae' in command
        data=pd.read_csv(command[command.index('--input')+1]);assert 'Residual Load' in data
        out=Path(command[command.index('--output-dir')+1])
        (out/'residual_ae_portable.pt').write_bytes(b'weights-only-dummy')
        (out/'training_summary.json').write_text(json.dumps({'best_validation_loss':.002}),encoding='utf-8')
    monkeypatch.setattr(run_experiment.subprocess,'run',fake_train)
    record=run_experiment.run_residual_autoencoder({'station':'amprion',
        'component':'residual_anomaly','algorithm':'dense_autoencoder',
        'experiment_id':'amprion-test-residual', 'parameters':{'epochs':1},
        'training_data_path':csv_path.relative_to(tmp_path).as_posix()})
    assert record['status']=='candidate'
    assert record['component']=='residual_anomaly'
    assert (tmp_path/record['metadata_path']).is_file()


def test_http_upload_and_component_discovery_endpoints(monkeypatch, tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    cfg = {"stations": {"amprion": station_entry(tmp_path)}}
    monkeypatch.setattr(model_api, "ROOT", tmp_path)
    monkeypatch.setattr(model_api, "read_station_config", lambda: cfg)
    monkeypatch.setattr(model_api, "resolve_root_path", lambda item: tmp_path / item)

    class StatusStub:
        def station_status(self, station):
            return {"services": {"training": {"running": False}}}
    monkeypatch.setattr(model_api, "manager", StatusStub())
    app = FastAPI()
    app.include_router(model_api.router)
    client = TestClient(app)
    components = client.get("/models/components", params={"station": "amprion"})
    assert components.status_code == 200
    assert set(name for name, enabled in components.json()["enabled"].items() if enabled) == {
        "demand_forecast", "solar_forecast", "wind_forecast", "residual_anomaly"
    }
    response = client.post("/models/training-data/upload", data={"station": "amprion", "component": "demand_forecast"},
                           files={"file": ("new-data.csv", _demand_csv(), "text/csv")})
    assert response.status_code == 200, response.text
    source_id = response.json()["source_id"]
    listing = client.get("/models/training-data", params={"station": "amprion", "component": "demand_forecast"})
    assert listing.status_code == 200
    assert listing.json()["datasets"][0]["source_id"] == source_id
    cross = client.get("/models/training-data", params={"station": "amprion", "component": "wind_forecast"})
    assert cross.json()["datasets"] == []
