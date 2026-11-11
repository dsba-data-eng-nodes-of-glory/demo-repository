from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_station_replay_forecast_and_checkpoint_cadence():
    cfg = yaml.safe_load((ROOT / "config" / "stations.yaml").read_text(encoding="utf-8"))
    station = cfg["stations"]["amprion"]
    assert float(station["stream_delay_seconds"]) == 2.0
    assert int(station["interval_minutes"]) == 15
    assert int(station["forecast_every_steps"]) == 1
    assert int(station["producer_checkpoint_every"]) == 1
    assert int(station["renewables"]["forecast_every_steps"]) == 1


def test_launcher_passes_checkpoint_every_to_producer():
    source = (ROOT / "backend" / "pipeline_manager.py").read_text(encoding="utf-8")
    assert '"--checkpoint-every", str(producer_checkpoint_every)' in source
    assert 'entry.get("producer_checkpoint_every", 1)' in source


def test_monitoring_refresh_and_api_cache_are_live_friendly():
    source = (ROOT / "dashboard" / "pages" / "monitoring.py").read_text(encoding="utf-8")
    assert "@st.cache_data(ttl=1, show_spinner=False)" in source
    assert 'key="electricity_poc_monitoring_refresh_v4"' in source
    assert 'index=0,' in source
