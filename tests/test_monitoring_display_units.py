import csv
import json
from io import StringIO
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd
import requests
import streamlit as st
from streamlit.testing.v1 import AppTest

from dashboard.residual_plot import residual_figures
from src.storage.station_hierarchy import build_tree


ROOT = Path(__file__).resolve().parents[1]
ORIGIN = "2026-10-03T13:45:00Z"
TARGET = "2026-10-03T14:00:00Z"


def test_monitoring_defaults_to_energy_and_switches_individual_and_group_views(monkeypatch):
    stations = {
        "demo_energy": {"hierarchy_path": [{"name": "Demo Grid", "level": "Country"}],
                        "interval_minutes": 15, "horizon_steps": 1,
                        "residual_demand_unit": "mwh_per_interval", "residual_model_checkpoint": "demo.pt"},
        "demo_power": {"interval_minutes": 30, "horizon_steps": 1, "residual_demand_unit": "mw"},
    }
    registry = {"active_station": "demo_energy", "stations": stations}
    tree = build_tree(stations)
    group = next(node for node in tree if node["children"])["id"]
    downloads = []
    phase = 0
    original_download = st.download_button

    def capture_download(*args, **kwargs):
        downloads.append(kwargs["data"].decode())
        return original_download(*args, **kwargs)

    def get(url, params=None, **kwargs):
        path = urlsplit(url).path
        station = (params or {}).get("station")
        step = pd.Timedelta(minutes=stations.get(station, {}).get("interval_minutes", 15))
        origin = (pd.Timestamp(ORIGIN) + step * phase).isoformat()
        target = (pd.Timestamp(origin) + step).isoformat()
        if path == "/stations":
            body = registry
        elif path == "/stations/hierarchy":
            body = {"nodes": tree}
        elif path == "/results/actuals":
            body = {"records": [{"utc_timestamp": origin,
                                 "actual_demand": 6000 if station == "demo_power" else 3033}]}
        elif path == "/results/forecasts":
            body = {"live_through_utc": origin, "records": [{
                "forecast_origin_utc": origin, "forecast_issued_at_utc": origin, "forecast_timestamp_utc": target,
                "forecast_run_id": "run", "horizon_step": 1,
                "predicted_demand": 6500 if station == "demo_power" else 3120,
            }]}
        elif path == "/results/renewables" and station == "demo_energy":
            body = {"actuals": [{"utc_timestamp": origin, "solar_actual_mwh": 10, "wind_actual_mwh": 20}],
                    "forecasts": {kind: [{"forecast_timestamp_utc": target, "forecast_origin_utc": origin,
                                         "predicted_generation_mwh": value}]
                                  for kind, value in (("solar", 12), ("wind", 24))}}
        elif path == "/results/residual/forecast":
            body = residual_payload()
            body["origin_utc"] = origin
            body["records"][0]["timestamp_utc"] = target
        elif path == "/results/hierarchy":
            levels = {"demand_mw": 12132, "solar_mwh": 10, "wind_mwh": 20, "residual_mwh": 3003}
            reading = {key: {"value": value, "reporting": 1, "expected": 1} for key, value in levels.items()}
            body = {"leaf_count": 1, "interval_minutes": 15, "horizon_steps": 1, "substations": [],
                    "forecast_origins_utc": dict.fromkeys(levels, origin),
                    "actuals": [{"utc_timestamp": origin, **reading}],
                    "forecasts": [{"utc_timestamp": target, **reading}]}
        else:
            body = {}
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps(body).encode()
        return response

    def run(page):
        page.run()
        assert not page.exception, page.exception
        return page

    def advance(page):
        nonlocal phase
        phase += 1
        st.cache_data.clear()
        run(page)

    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setattr(st, "download_button", capture_download)
    script = str(ROOT / "dashboard" / "pages" / "monitoring.py")
    page = run(AppTest.from_file(script))
    assert page.radio(key="monitoring_display_unit").value == "MWh"
    assert not page.get("plotly_chart")
    run(page)
    assert not page.get("plotly_chart")
    advance(page)
    assert len(page.get("plotly_chart")) == 1
    advance(page)
    assert [value.value for value in page.metric if value.label == "Latest actual"] == [
        "3,033.00 MWh", "10.00 MWh", "20.00 MWh",
    ]
    next(item for item in page.checkbox if item.label == "Show technical details").check()
    run(page)
    exported = next(csv.DictReader(StringIO(downloads[-1])))
    assert (float(exported["predicted_demand"]), exported["demand_unit"], exported["interval_minutes"]) == (3120, "MWh", "15")
    page.radio(key="monitoring_display_unit").set_value("MW")
    run(page)
    assert [value.value for value in page.metric if value.label == "Latest actual"] == [
        "12,132.00 MW", "40.00 MW", "80.00 MW",
    ]
    assert next(value.value for value in page.metric if value.label == "Next residual") == "12,336.00 MW"
    assert next(value.value for value in page.metric if value.label == "Peak 30m ramp") == "32.00 MW"
    exported = next(csv.DictReader(StringIO(downloads[-1])))
    assert (float(exported["predicted_demand"]), exported["demand_unit"]) == (12480, "MW")

    page = AppTest.from_file(script)
    page.query_params["station"] = group
    run(page)
    assert not page.get("plotly_chart")
    run(page)
    assert not page.get("plotly_chart")
    advance(page)
    assert len(page.get("plotly_chart")) == 4
    assert [value.value for value in page.metric if value.label == "Latest actual"] == [
        "3,033.00 MWh", "10.00 MWh", "20.00 MWh", "3,003.00 MWh",
    ]
    page.radio(key="monitoring_display_unit").set_value("MW")
    run(page)
    assert [value.value for value in page.metric if value.label == "Latest actual"] == [
        "12,132.00 MW", "40.00 MW", "80.00 MW", "12,012.00 MW",
    ]
    assert all("(MW)" in json.loads(chart.proto.spec)["layout"]["yaxis"]["title"]["text"]
               for chart in page.get("plotly_chart"))

    page = AppTest.from_file(script)
    page.query_params["station"] = "demo_power"
    run(page)
    assert not page.get("plotly_chart")
    advance(page)
    assert page.metric[0].value == "3,000.00 MWh"
    page.radio(key="monitoring_display_unit").set_value("MW")
    run(page)
    assert page.metric[0].value == "6,000.00 MW"


def residual_payload():
    return {"available": True, "origin_utc": ORIGIN, "records": [{
        "timestamp_utc": TARGET, "residual_load": 3084, "expected_residual_load": 3000,
        "residual_ramp_30m": 8, "statistical_score": 2.3, "autoencoder_score": 1.2,
        "statistical_threshold": 4, "autoencoder_threshold": 5,
        "upward_highlight": False, "context_valid": True,
    }]}


def test_residual_display_converts_physical_values_and_preserves_scores():
    payload = residual_payload()
    level, scores = residual_figures(payload, display_unit="MW")
    assert list(level.data[0].y) == [12336]
    assert list(level.data[1].y) == [12000]
    assert list(scores.data[0].y) == [2.3]
    assert scores.layout.shapes[0].y0 == 4
    assert payload["records"][0]["residual_load"] == 3084
