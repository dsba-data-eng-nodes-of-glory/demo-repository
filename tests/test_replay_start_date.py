from copy import deepcopy
from io import BytesIO
from argparse import Namespace
from unittest.mock import Mock

from fastapi import FastAPI
from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
import pytest
import yaml

from backend import station_api
from src.models import model_registry
from src.models import train_tabular
from src.producers import csv_producer


CUTOFF = pd.Timestamp("2026-10-01", tz="UTC")


def workbook(start, *, ren=False):
    times = pd.date_range(pd.Timestamp(start, tz="UTC"), CUTOFF + pd.Timedelta(days=2), freq="15min", tz="UTC")
    source = BytesIO()
    with pd.ExcelWriter(source, engine="openpyxl") as writer:
        pd.DataFrame({"utc_timestamp": times.astype(str), "demand": 4200 + 300 * np.sin(np.arange(len(times)) / 15),
                      "temperature": 8}).to_excel(writer, sheet_name="Demand", index=False)
        if ren:
            pd.DataFrame({"utc_timestamp": times.astype(str), "Photovoltaics [MWh]": 1.0,
                          "cloud_cover_mean_pct": 80.0, "shortwave_radiation_mean_prev_hour_w_m2": 50.0,
                          "forecast_cloud_cover_mean_pct": 82.0, "forecast_shortwave_radiation_mean_prev_hour_w_m2": 52.0,
                          "Photovoltaics_installed_capacity [MW]": 10.0}).to_excel(writer, sheet_name="Solar", index=False)
            pd.DataFrame({"utc_timestamp": times.astype(str), "Wind onshore [MWh]": 2.0,
                          "wind_speed_100m_mean_ms": 8.0, "wind_direction_100m_circular_mean_deg": 225.0,
                          "future_wind_speed": 8.2, "future_wind_direction": 226.0,
                          "Wind_onshore_installed_capacity [MW]": 20.0}).to_excel(writer, sheet_name="Wind", index=False)
    return source.getvalue()


@pytest.fixture
def project(monkeypatch, tmp_path):
    config = {"stations": {}}
    monkeypatch.setattr(station_api, "ROOT", tmp_path)
    monkeypatch.setattr(station_api, "_read_model_settings", lambda: {
        "interval_minutes": 15, "history_window_hours": 168, "forecast_horizon_hours": 24,
    })
    monkeypatch.setattr(station_api, "read_station_config", lambda: deepcopy(config))
    monkeypatch.setattr(station_api, "write_station_config", lambda doc: config.update(deepcopy(doc)))

    class Topics:
        def ensure_station_topics(self, station):
            return {"station": station}

    monkeypatch.setattr(station_api, "manager", Topics())
    app = FastAPI()
    app.include_router(station_api.router)
    with TestClient(app) as client:
        yield client, config, tmp_path


def register(client, name, source, *, cutoff="2026-10-01", models="demand_forecast"):
    return client.post("/stations/onboard", data={
        "station_id": name, "replay_start_date": cutoff, "selected_models": models,
        "hierarchy_path": "Country: Germany\nState: Hesse\nStation: Primary",
        "residual_demand_unit": "mw",
    }, files={"file": ("observations.xlsx", source)})


def times(root, file):
    return pd.to_datetime(pd.read_csv(root / file)["utc_timestamp"], utc=True)


def test_two_stations_share_the_same_first_replay_observation(project, monkeypatch):
    client, config, root = project
    producer = Mock()
    monkeypatch.setattr(csv_producer, "KafkaProducer", Mock(return_value=producer))
    monkeypatch.setattr(csv_producer, "ROOT", root)
    monkeypatch.setattr(csv_producer, "load_yaml", lambda path: {
        "config/stations.yaml": config, "config/kafka.yaml": {"bootstrap_servers": "localhost:9092"},
        "config/model.yaml": {},
    }[path])

    for name, start, renewable in [("station-a", "2026-09-12", True), ("station-b", "2026-09-15", False)]:
        result = register(client, name, workbook(start, ren=renewable),
                          models="demand_forecast,solar_forecast,wind_forecast" if renewable else "demand_forecast")
        assert result.status_code == 200, result.text
        entry = config["stations"][name]
        assert entry["expected_stream_start_utc"] == CUTOFF.isoformat()
        assert entry["initial_training"]["demand_forecast"]["status"] == "queued"
        assert times(root, entry["input_file"]).iloc[0] == CUTOFF
        assert times(root, entry["training_file"]).max() == CUTOFF - pd.Timedelta(minutes=15)
        history = times(root, entry["history_file"])
        assert len(history) == 672 and history.max() == CUTOFF - pd.Timedelta(minutes=15)
        assert entry["model_row_counts"]["demand_forecast"]["test_rows"] >= 96
        csv_producer.run_producer(Namespace(
            station=name, delay=0, limit=1, chunksize=300, checkpoint_every=1,
            csv=None, history_file=None, topic=None, bootstrap=None, progress_file=None,
            skip_boundary_check=False, replay=False,
        ))
        assert pd.Timestamp(producer.send.call_args.kwargs["value"]["utc_timestamp"]) == CUTOFF
        if renewable:
            for kind in ("solar", "wind"):
                renewable_cfg = entry["renewables"][kind]
                assert times(root, renewable_cfg["train_file"]).max() < CUTOFF
                assert times(root, renewable_cfg["replay_file"]).min() == CUTOFF


@pytest.mark.parametrize("date, message", [
    ("2026-09-01", "needs an observation"),
    ("2026-09-27", "usable historical rows"),
    ("2026-10-04", "needs an observation"),
])
def test_rejects_unsupported_replay_dates(project, date, message):
    client, config, _ = project
    response = register(client, "station-a", workbook("2026-09-24"), cutoff=date)
    assert response.status_code == 422
    assert message in response.json()["detail"]
    assert "station-a" not in config["stations"]


def test_dense_nn_evaluates_every_split_before_replay(project, monkeypatch):
    client, config, root = project
    response = register(client, "station-a", workbook("2026-09-15"))
    assert response.status_code == 200, response.text
    folder = root / "config"
    folder.mkdir()
    (folder / "stations.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    (folder / "model.yaml").write_text("device: cpu\n", encoding="utf-8")
    monkeypatch.setattr(train_tabular, "ROOT", root)
    monkeypatch.setattr(model_registry, "REGISTRY_PATH", root / "models" / "registry" / "experiments.json")
    meta = train_tabular.run_training({
        "station": "station-a", "algorithm": "dense_nn", "experiment_id": "dated-dense",
        "parameters": {"hidden_layer_sizes": "8", "max_iter": 4, "max_train_origins": 8, "max_validation_origins": 4},
    })
    assert meta["test_metrics"]["matched_observations"] == 4 * 96
    assert (pd.Timestamp(meta["training_end_utc"]) < pd.Timestamp(meta["validation_start_utc"])
            < pd.Timestamp(meta["validation_end_utc"]) < pd.Timestamp(meta["test_start_utc"])
            < pd.Timestamp(meta["test_end_utc"]) < CUTOFF)
