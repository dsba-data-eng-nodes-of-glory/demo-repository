import os
from pathlib import Path
import sys
import time

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend import pipeline_api, pipeline_manager as pm


def test_start_all_releases_csv_producers_together_and_keeps_other_stations_isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(pm, "ROOT", Path(__file__).resolve().parents[1])
    monkeypatch.setattr(pm, "STATION_CONFIG", tmp_path / "stations.yaml")
    monkeypatch.setattr(pm, "LOG_DIR", tmp_path)
    monkeypatch.setenv("QUALITY_CONSUMER_EXTERNAL", "true")
    stations = {"one": {}, "two": {}, "broken": {}, "training": {}, "external": {}}
    pm.write_station_config({"stations": stations})
    manager = pm.PipelineManager()
    monkeypatch.setattr(pipeline_api, "manager", manager)
    monkeypatch.setattr(manager, "get_runtime_environment", lambda: {"env": os.environ.copy()})
    monkeypatch.setattr(manager, "start_supervisor", lambda: None)

    sleeper = [sys.executable, "-c", "import time; time.sleep(30)"]
    worker = """
from pathlib import Path
import sys
import time
from src.producers.csv_producer import wait_for_group_replay

# Simulate noticeably different import, CSV and broker startup times.
time.sleep(float(sys.argv[2]))
start = wait_for_group_replay(Path(sys.argv[-1]))
time.sleep(max(0.0, start - time.monotonic()))
Path(sys.argv[1]).write_text(str(time.monotonic()))
time.sleep(30)
"""

    def prepare(station):
        if station == "training":
            raise RuntimeError("Initial model training is still queued.")
        commands = [("spark", sleeper), ("predictor", sleeper)]
        if station == "broken":
            commands.append(("producer", [sys.executable, "-c", "raise SystemExit(2)"]))
        elif station != "external":
            delay = "0.0" if station == "one" else "0.6"
            commands.append(("producer", [sys.executable, "-c", worker, str(tmp_path / station), delay]))
        return commands, {}

    monkeypatch.setattr(manager, "_prepare_pipeline", prepare)
    app = FastAPI()
    app.include_router(pipeline_api.router)
    try:
        with TestClient(app) as client:
            response = client.post("/pipeline/start-all")
            assert response.status_code == 200, response.text
            result = response.json()["stations"]
            assert result["one"]["status"] == result["two"]["status"] == "started", result
            assert result["external"]["status"] == "started"
            assert "producer" not in result["external"]["services"]
            assert "exited with code 2" in result["broken"]["message"]
            assert result["training"] == {"status": "failed", "message": "Initial model training is still queued."}
            assert pm.read_station_config()["active_stations"] == ["external", "one", "two"]
            first_pids = {name: manager.processes[f"{name}:producer"].pid for name in ("one", "two")}

            deadline = time.monotonic() + 4
            while not all((tmp_path / name).is_file() for name in first_pids):
                assert time.monotonic() < deadline
                time.sleep(0.03)
            times = [float((tmp_path / name).read_text()) for name in first_pids]
            assert abs(times[0] - times[1]) < 0.2
            assert "--group-ready-file" not in manager._supervised["one:producer"].command
            assert manager.processes["broken:spark"].poll() is not None

            repeated = client.post("/pipeline/start-all").json()["stations"]
            assert repeated["one"]["status"] == repeated["two"]["status"] == "already_running"
            assert first_pids == {name: manager.processes[f"{name}:producer"].pid for name in first_pids}
    finally:
        manager.stop_pipeline()
