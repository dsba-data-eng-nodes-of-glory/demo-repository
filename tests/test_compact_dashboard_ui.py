"""Compact dashboard contracts: meaningful renewables KPIs with replay-safe actuals."""
from __future__ import annotations

import pandas as pd

from dashboard.renewable_summary import build_renewable_summaries


def test_solar_wind_summary_uses_latest_origin_and_separate_units():
    actual = pd.DataFrame({
        "utc_timestamp": ["2026-10-01T00:15:00Z", "2026-10-01T00:30:00Z"],
        "solar_actual_mwh": [1, 2], "wind_actual_mwh": [70, 80],
    })
    forecasts = {
        "solar": [
            {"forecast_origin_utc": "2026-10-01T00:00:00Z", "forecast_timestamp_utc": "2026-10-01T00:15:00Z", "predicted_generation_mwh": 1000},
            {"forecast_origin_utc": "2026-10-01T00:30:00Z", "forecast_timestamp_utc": "2026-10-01T00:45:00Z", "predicted_generation_mwh": 3},
            {"forecast_origin_utc": "2026-10-01T00:30:00Z", "forecast_timestamp_utc": "2026-10-01T01:00:00Z", "predicted_generation_mwh": 5},
        ],
        "wind": [
            {"forecast_origin_utc": "2026-10-01T00:30:00Z", "forecast_timestamp_utc": "2026-10-01T00:45:00Z", "predicted_generation_mwh": 110},
            {"forecast_origin_utc": "2026-10-01T00:30:00Z", "forecast_timestamp_utc": "2026-10-01T01:00:00Z", "predicted_generation_mwh": 130},
        ],
    }
    result = build_renewable_summaries(actual, forecasts)
    assert result["solar"] == {"latest_actual": 2.0, "first_forecast": 3.0, "peak_forecast": 5.0,
                               "average_forecast": 4.0, "forecast_steps": 2}
    assert result["wind"]["latest_actual"] == 80
    assert result["wind"]["average_forecast"] == 120


def test_renewable_summary_missing_or_malformed_values_are_blank_not_fake_zero():
    partial = pd.DataFrame({"utc_timestamp": ["2026-10-01T00:00:00Z"],
                            "solar_actual_mwh": [float("nan")]})
    result = build_renewable_summaries(partial, {"solar": [{"predicted_generation_mwh": "bad"}]})
    assert result["solar"]["latest_actual"] is None
    assert result["solar"]["first_forecast"] is None
    assert result["wind"]["latest_actual"] is None
    assert result["wind"]["forecast_steps"] == 0


def test_replay_watermark_is_respected_by_caller_before_summarizing():
    from dashboard.replay_display import filter_actuals_to_cutoff
    records = pd.DataFrame({"utc_timestamp": pd.date_range("2026-10-01T00:00:00Z", periods=3, freq="15min"),
                            "solar_actual_mwh": [1, 2, 999], "wind_actual_mwh": [2, 3, 999]})
    visible = filter_actuals_to_cutoff(records, pd.Timestamp("2026-10-01T00:15:00Z"), "utc_timestamp")
    result = build_renewable_summaries(visible, {})
    assert result["solar"]["latest_actual"] == 2
    assert result["wind"]["latest_actual"] == 3


def test_compact_status_markup_escapes_user_supplied_station_name(monkeypatch):
    import types, sys
    # The theme imports Streamlit; don't require Streamlit in this pure unit test.
    import importlib.util
    source = importlib.util.spec_from_file_location("compact_ui_theme_test", "dashboard/ui.py")
    fake = types.SimpleNamespace(markdown=lambda body, unsafe_allow_html=None: captured.append(body))
    captured = []
    monkeypatch.setitem(sys.modules, "streamlit", fake)
    module = importlib.util.module_from_spec(source)
    source.loader.exec_module(module)
    module.render_status_strip([('<script>alert(1)</script>', 'Running', 'good')])
    assert '<script>' not in captured[0]
    assert '&lt;script&gt;' in captured[0]
    assert 'Running' in captured[0]
