from pathlib import Path


def test_monitoring_waits_for_forecast_before_rendering_any_graph():
    source = Path(__file__).parents[1] / "dashboard" / "pages" / "monitoring.py"
    text = source.read_text(encoding="utf-8")
    assert "if latest_run.empty:" in text
    branch = text.split("if latest_run.empty:", 1)[1].split("# Fetch the full replay window", 1)[0]
    assert "forecast-init-card" in branch
    assert "return" in branch
    assert "Waiting for the first" in branch


def test_demand_graph_uses_configured_stream_start_not_rolling_history():
    source = Path(__file__).parents[1] / "dashboard" / "pages" / "monitoring.py"
    text = source.read_text(encoding="utf-8")
    assert "history_start = stream_start if stream_start is not None" in text
    assert '"limit": 20000' in text


def test_renewable_actuals_share_the_same_stream_start():
    source = Path(__file__).parents[1] / "dashboard" / "pages" / "monitoring.py"
    text = source.read_text(encoding="utf-8")
    assert 'renewable_actuals["utc_timestamp"] >= stream_start' in text
