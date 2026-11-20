import pandas as pd

from dashboard import monitoring_chart
from dashboard.replay_display import filter_actuals_to_cutoff, resolve_actuals_display_cutoff


START = pd.Timestamp("2026-10-01T00:00:00Z")
STEP = pd.Timedelta(minutes=15)


def test_initialization_card_reports_status_and_escapes_station_names(monkeypatch):
    calls = []
    monkeypatch.setattr(monitoring_chart.st, "markdown", lambda markup, **kwargs: calls.append((markup, kwargs)))
    monitoring_chart.render_initializing("Wind <station>", "Waiting for a live origin.")
    markup, options = calls[0]
    assert 'role="status"' in markup
    assert 'class="forecast-init-spinner"' in markup
    assert "Initializing Wind &lt;station&gt;" in markup
    assert "Waiting for a live origin." in markup
    assert options["unsafe_allow_html"]


def test_initial_replay_cutoff_excludes_every_unobserved_interval():
    limit, reason = resolve_actuals_display_cutoff(START, 15)
    assert (limit, reason) == (START - STEP, "before_stream_start")


def test_live_forecast_origin_is_the_cutoff_for_observations():
    cutoff, reason = resolve_actuals_display_cutoff(START, 15, replay_timestamp=START + STEP, forecast_origin=START)
    assert (cutoff, reason) == (START, "forecast_origin")
    frame = pd.DataFrame({"utc_timestamp": [START - STEP, START, START + STEP], "solar_actual_mwh": [1, 2, 3]})
    assert filter_actuals_to_cutoff(frame, cutoff)["solar_actual_mwh"].tolist() == [1, 2]
