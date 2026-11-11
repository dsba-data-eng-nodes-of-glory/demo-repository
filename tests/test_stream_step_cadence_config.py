from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]

def test_amprion_replay_advances_one_15_minute_row_every_two_seconds():
    config = yaml.safe_load((ROOT / "config" / "stations.yaml").read_text(encoding="utf-8"))
    station = config["stations"]["amprion"]
    assert station["interval_minutes"] == 15
    assert station["stream_delay_seconds"] == 2.0
    assert station["forecast_every_steps"] == 1
    assert station["renewables"]["forecast_every_steps"] == 1

def test_monitoring_defaults_to_two_second_refresh():
    source = (ROOT / "dashboard" / "pages" / "monitoring.py").read_text(encoding="utf-8")
    assert "[2, 5, 10, 15],\n        index=0" in source
