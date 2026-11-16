from __future__ import annotations

from copy import deepcopy
import sys
import time

from backend import pipeline_manager as pm
from backend import station_api


def _wait_until(predicate, timeout=12):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    raise AssertionError("Timed out waiting for process supervisor")


def test_restarts_real_worker_and_cancels_retry_on_stop(tmp_path, monkeypatch):
    monkeypatch.setattr(pm, "LOG_DIR", tmp_path)
    manager = pm.PipelineManager()
    monkeypatch.setattr(manager, "get_runtime_environment", lambda: {"env": {}})
    monkeypatch.setattr(manager, "kafka_available", lambda: True)
    monkeypatch.setattr(manager, "RESTART_DELAYS", (0.05, 0.1))
    command = [sys.executable, "-c", "import time; time.sleep(60)"]
    clean_exit = [sys.executable, "-c", "pass"]
    try:
        manager.start("predictor", command, "demo")
        manager.start("producer", clean_exit, "demo")
        manager._supervise("predictor", command, "demo")
        manager._supervise("producer", clean_exit, "demo")
        old = manager.processes["demo:predictor"]
        completed = manager.processes["demo:producer"]
        completed.wait(timeout=5)
        old.kill()
        old.wait(timeout=5)
        _wait_until(lambda: manager.processes["demo:predictor"] is not old)
        assert manager.processes["demo:producer"] is completed
        replacement = manager.processes["demo:predictor"]
        replacement.kill()
        replacement.wait(timeout=5)
        manager.stop("predictor", "demo")
        manager._restart_failed_services()
        assert manager.processes["demo:predictor"] is replacement
        assert "demo:predictor" not in manager._supervised
    finally:
        manager.shutdown_supervisor()
        manager.stop_pipeline()


def test_restart_budget_is_bounded_and_exhaustion_is_visible(tmp_path, monkeypatch):
    monkeypatch.setattr(pm, "LOG_DIR", tmp_path)
    manager = pm.PipelineManager()
    monkeypatch.setattr(manager, "get_runtime_environment", lambda: {"env": {}})
    monkeypatch.setattr(manager, "kafka_available", lambda: True)
    monkeypatch.setattr(manager, "RESTART_DELAYS", (0.01,))
    command = [sys.executable, "-c", "raise SystemExit(1)"]
    try:
        manager.start("spark", command, "demo")
        manager._supervise("spark", command, "demo")
        _wait_until(lambda: manager.status()["processes"]["demo:spark"]["state"] == "failed")
        service = manager.status()["processes"]["demo:spark"]
        assert service["restart_attempts"] == 1
        assert service["return_code"] == 1
    finally:
        manager.shutdown_supervisor()
        manager.stop_pipeline()


def test_station_stop_preserves_recovery_and_does_not_reset(tmp_path, monkeypatch):
    original = {"stations": {"demo": {"enabled": True, "status": "active"}},
                "active_station": "demo", "active_stations": ["demo"]}
    state = deepcopy(original)
    manager = pm.PipelineManager()
    monkeypatch.setattr(pm, "read_station_config", lambda: deepcopy(state))
    monkeypatch.setattr(pm, "write_station_config", state.update)
    monkeypatch.setattr(station_api, "read_station_config", lambda: deepcopy(state))
    monkeypatch.setattr(station_api, "manager", manager)
    monkeypatch.setattr(manager, "reset_station_pipeline", lambda *_: (_ for _ in ()).throw(AssertionError("Unexpected reset")))
    file = tmp_path / "predictor_recovery.sqlite3"
    file.write_bytes(b"state")
    result = station_api.stop_station("demo")
    assert result["station"] == "demo" and "reset" not in result
    assert state["stations"]["demo"]["status"] == "stopped" and file.read_bytes() == b"state"
