import pandas as pd
import pytest

from dashboard.hierarchy_monitoring import METRICS, build_group_chart, group_cursor
from dashboard.monitoring_chart import ACTUAL_COLOR, FORECAST_COLOR
from dashboard.monitoring_progress import forecast_cursor, timeline_is_live


START = pd.Timestamp("2026-10-01T00:00:00Z")
STEP = pd.Timedelta(minutes=15)


def test_initialization_requires_movement_and_resets_when_restarted():
    state = {}
    key = ("station", "amprion", "demand")
    assert not timeline_is_live(state, key, None, 101)
    assert not timeline_is_live(state, key, START, 101)
    assert not timeline_is_live(state, key, START, 101)
    assert timeline_is_live(state, key, START + STEP, 101)
    assert not timeline_is_live(state, key, None, 101)
    assert timeline_is_live(state, key, START + STEP, 101)
    assert not timeline_is_live(state, key, START + STEP, 102)
    assert timeline_is_live(state, key, START + 2 * STEP, 102)
    assert not timeline_is_live(state, key, START, 102)


def test_complete_forecast_requires_replay_to_reach_its_origin():
    origin = START
    records = [
        {"forecast_origin_utc": origin, "forecast_timestamp_utc": origin + step * STEP}
        for step in (1, 2)
    ]
    assert forecast_cursor(records, START, START, 2, 15) == START
    assert forecast_cursor(records, START, START + STEP, 2, 15) is None
    assert forecast_cursor(records, START - STEP, START, 2, 15) is None
    assert forecast_cursor(records[:1], START, START, 2, 15) is None
    assert forecast_cursor(records[:1] * 2, START, START, 2, 15) is None


def reading(stamp, reporting=4):
    values = [15_200, 100, 200, 3500]
    return {
        "utc_timestamp": stamp.isoformat(),
        **{key: {"value": value, "reporting": reporting, "expected": 4}
           for key, value in zip(METRICS, values)},
    }


@pytest.mark.parametrize("metric", METRICS)
def test_parent_chart_shares_amprion_axes_and_requires_full_coverage(metric):
    payload = {
        "leaf_count": 4, "interval_minutes": 15, "horizon_steps": 2,
        "replay_start_utc": START.isoformat(),
        "forecast_origins_utc": dict.fromkeys(METRICS, START.isoformat()),
        "actuals": [reading(START - 2 * STEP), reading(START - STEP, 2)],
        "forecasts": [reading(START + STEP), reading(START + 2 * STEP)],
    }
    assert group_cursor(payload, metric, True) is None
    assert group_cursor(payload, metric, False) is None
    payload["actuals"].append(reading(START))
    cursor = group_cursor(payload, metric, True)
    assert cursor == group_cursor(payload, metric, False) == START
    fig = build_group_chart(payload, "germany", metric, cursor,
                            history_hours=72, include_forecasts=True, display_unit="MWh")
    assert fig.layout.xaxis.range == (START - pd.Timedelta(hours=72), START + 2 * STEP)
    assert fig.layout.xaxis.dtick == 6 * 60 * 60 * 1000
    assert fig.layout.xaxis.tickformat == "%d %b<br>%H:%M"
    assert fig.layout.yaxis.title.text.endswith("(MWh / 15 min)")
    assert fig.layout.shapes[0].x0 == START
    assert fig.layout.annotations[0].text == "Now"
    actual, partial, forecast = fig.data
    assert actual.line.color == ACTUAL_COLOR and actual.y[1] is None
    assert partial.mode == "markers" and partial.visible == "legendonly"
    assert forecast.line.color == FORECAST_COLOR and forecast.line.dash == "dash"
    assert forecast.x[0] == START and forecast.y[0] == actual.y[-1]
    payload["forecasts"].pop()
    assert group_cursor(payload, metric, True) is None
