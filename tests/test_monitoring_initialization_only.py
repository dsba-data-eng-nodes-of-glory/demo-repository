from pathlib import Path


def _source():
    return (Path(__file__).parents[1] / "dashboard" / "pages" / "monitoring.py").read_text(encoding="utf-8")


def test_initialization_placeholder_replaces_actual_only_wait_state():
    text = _source()
    assert "if latest_run.empty:" in text
    branch = text.split("if latest_run.empty:", 1)[1].split("# 5,000 rows comfortably", 1)[0]
    assert "forecast-init-card" in branch
    assert "forecast-init-spinner" in branch
    assert "return" in branch


def test_demand_chart_rolling_x_axis_window_is_unchanged():
    text = _source()
    assert "history_start = forecast_origin - pd.Timedelta(hours=history_hours)" in text
    assert '"range"' not in text.split("if not latest_run.empty:", 1)[1].split("# Renewable generation monitoring", 1)[0]
    assert 'actual_params = {"station": station, "limit": 600}' in text


def test_actual_history_filter_is_unchanged():
    text = _source()
    assert 'actuals["utc_timestamp"] <= forecast_origin' in text
    assert 'actual_before_origin["utc_timestamp"] >= history_start' in text


def test_renewable_chart_window_is_unchanged():
    text = _source()
    assert 'renewable_params = {"station": station, "limit": 5000}' in text
    assert 'renewable_actuals = filter_actuals_to_cutoff(' in text
    assert 'chart_start=chart_start, chart_end=chart_end' in text
    assert 'renewable_actuals["utc_timestamp"] >= stream_start' not in text
