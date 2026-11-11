"""Operational status API cache and station overview regression."""
from pathlib import Path

def test_pipeline_status_java_check_is_cached_for_dashboard(monkeypatch):
    from backend.pipeline_manager import PipelineManager

    manager = PipelineManager()
    calls = []
    def fake_java():
        calls.append(True)
        return {
            "python_executable": "python", "java_home": "java-home",
            "java_executable": "java", "java_version": "JDK test",
        }
    monkeypatch.setattr(manager, "get_runtime_environment", fake_java)
    monkeypatch.setattr(manager, "kafka_available", lambda: True)
    first = manager.status()
    second = manager.status()
    assert first["runtime"] == second["runtime"]
    assert len(calls) == 1
    assert first["kafka_tcp_available"] is True


def test_station_overview_uses_one_snapshot_and_lists_four_amprion_components(monkeypatch):
    from backend import station_api

    import yaml
    project_root = Path(__file__).resolve().parents[1]
    entry = yaml.safe_load((project_root / "config/stations.yaml").read_text(encoding="utf-8"))["stations"]["amprion"]
    monkeypatch.setattr(station_api, "resolve_root_path", lambda value: (project_root / value).resolve())
    snapshots = []
    fake_snapshot = {"kafka_tcp_available": False, "processes": {}}
    monkeypatch.setattr(station_api, "read_station_config", lambda: {
        "active_station": None, "stations": {"amprion": entry, "second": {"display_name": "Second"}}
    })
    monkeypatch.setattr(station_api.manager, "status", lambda: snapshots.append(True) or fake_snapshot)
    result = station_api.stations_overview()
    assert len(snapshots) == 1
    assert result["kafka_tcp_available"] is False
    assert set(result["statuses"]) == {"amprion", "second"}
    components = result["components"]["amprion"]
    assert set(components) == {"demand_forecast", "solar_forecast", "wind_forecast", "residual_anomaly"}
    assert all(item["enabled"] for item in components.values())
    assert all(item["training_data_exists"] for item in components.values())
