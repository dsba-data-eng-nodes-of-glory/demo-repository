import pandas as pd

from backend.results_api import _combine_actuals


def test_combine_actuals_prefers_live_rows_and_applies_cutoff():
    historical = pd.DataFrame([
        {"utc_timestamp": "2026-11-04T09:00:00Z", "actual_demand": 100.0},
        {"utc_timestamp": "2026-11-04T09:15:00Z", "actual_demand": 110.0},
    ])
    live = pd.DataFrame([
        {"utc_timestamp": "2026-11-04T09:15:00Z", "actual_demand": 111.0},
        {"utc_timestamp": "2026-11-04T09:30:00Z", "actual_demand": 120.0},
    ])
    result = _combine_actuals(
        historical, live, station="tennet", limit=100,
        end_time="2026-11-04T09:15:00Z",
    )
    assert len(result) == 2
    assert result.iloc[-1]["actual_demand"] == 111.0
    assert result.iloc[-1]["station_id"] == "tennet"
