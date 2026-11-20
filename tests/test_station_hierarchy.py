from copy import deepcopy
from io import BytesIO

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend import results_api, station_api
from src.storage.hierarchy_rollup import LeafResults, roll_up
from src.storage.live_actuals import initialize, live_actuals_path, save_readings
from src.storage.station_hierarchy import build_tree, group_key, group_stations, parse_path, validate_path


def leaf(path, unit="mw"):
    return {"hierarchy_path": parse_path(path), "interval_minutes": 15, "horizon_steps": 2,
            "hierarchy_grid_offset_seconds": 0, "residual_demand_unit": unit}


def test_shared_roots_and_arbitrary_depth():
    prefix = "Country: Germany\nState: Hesse\nUtility: Utility A\nStation: Main"
    registry = {"a": leaf(prefix), "b": leaf(prefix.lower()), "c": leaf("Country: Germany\nState: Berlin\nStation: Main"),
                "legacy": {"display_name": "Amprion"}}
    roots = build_tree(registry)
    assert [root["name"] for root in roots] == ["Amprion", "Germany"]
    assert roots[1]["leaf_count"] == 3
    assert set(group_stations(registry, group_key(["Germany", "Hesse", "Utility A", "Main"]))) == {"a", "b"}
    assert set(group_stations(registry, group_key(["Germany", "Berlin", "Main"]))) == {"c"}
    assert validate_path({"a": registry["a"]}, parse_path(prefix.lower()), 15, 0) == parse_path(prefix)
    with pytest.raises(ValueError, match="same observation interval"):
        validate_path(registry, parse_path(prefix), 30, 0)
    with pytest.raises(ValueError, match="same UTC interval grid"):
        validate_path(registry, parse_path(prefix), 15, 60)
    with pytest.raises(ValueError, match="already registered"):
        validate_path(registry, parse_path("Region: Germany"), 15, 0)


def observation(minute, value):
    return {"utc_timestamp": f"2026-10-01T00:{minute:02d}:00Z", "actual_demand": value}


def run(origin, value, kind="demand_mw", steps=2):
    field = "predicted_demand" if kind == "demand_mw" else "predicted_generation_mwh"
    origin = pd.Timestamp(f"2026-10-01T00:{origin:02d}:00Z")
    return [{"forecast_origin_utc": origin.isoformat(), "forecast_timestamp_utc": (origin + pd.Timedelta(minutes=15 * i)).isoformat(),
             "forecast_run_id": origin.isoformat(), field: value} for i in range(1, steps + 1)]


def test_rollup_uses_common_origin_and_recovers_original_intervals():
    members = {"a": leaf("Germany", "mw"), "b": leaf("Germany", "mwh_per_interval")}
    left = LeafResults([observation(0, 20), observation(15, 21), observation(30, 22)], [
        {"utc_timestamp": observation(0, 0)["utc_timestamp"], "generation_type": kind,
         "actual_generation_mwh": 1} for kind in ("solar", "wind")
    ], {"demand_mw": run(0, 20) + run(15, 21) + run(30, 22, steps=1)})
    right = LeafResults([observation(0, 8), observation(30, 9)], [], {"demand_mw": run(0, 8) + run(30, 9, steps=1)})
    initial = roll_up(members, {"a": left, "b": right}, 3, 15, 2)
    assert initial["actuals"][1]["demand_mw"] == {"value": 21, "reporting": 1, "expected": 2}
    assert initial["actuals"][0]["residual_mwh"] == {"value": 3, "reporting": 1, "expected": 2}
    assert initial["forecast_origin_utc"] == "2026-10-01T00:00:00+00:00"
    assert initial["forecasts"][0]["demand_mw"]["value"] == 52

    right.demand.extend([observation(15, 9), observation(15, 9)])
    recovered = roll_up(members, {"a": left, "b": right}, 3, 15, 2)
    assert recovered["actuals"][1]["demand_mw"] == {"value": 57, "reporting": 2, "expected": 2}
    assert recovered["actuals"][-1] == initial["actuals"][-1]


def test_group_api_reads_operational_storage_without_opening_uploads(tmp_path, monkeypatch):
    parent = "Country: Germany\nState: Hesse\nStation: Main"
    state = {"stations": {f"demo-{i}": leaf(parent) for i in range(1, 5)}}
    for index, (name, entry) in enumerate(state["stations"].items(), 1):
        entry["input_file"] = f"uploads/{name}.csv"
        entry["actuals_file"] = f"out/{name}.csv"
        files = tmp_path / "uploads"
        files.mkdir(exist_ok=True)
        future = {**observation(45, 999), "station_id": name}
        pd.DataFrame([future]).to_csv(tmp_path / entry["input_file"], index=False)
        path = tmp_path / entry["actuals_file"]
        path.parent.mkdir(exist_ok=True)
        row = {**observation(0, index * 10), "station_id": name}
        pd.DataFrame([row]).to_csv(path, index=False)

    # Three substations are ahead of the predictor at :15. The other remains
    # absent from this interval until its own reader recovers the Kafka record.
    for index in range(1, 4):
        path = live_actuals_path(tmp_path, f"demo-{index}")
        initialize(path, f"demo-{index}-featured")
        row = observation(15, index * 10)
        save_readings(path, [(row["utc_timestamp"], row["actual_demand"])], 2)
    monkeypatch.setattr(results_api, "ROOT", tmp_path)
    monkeypatch.setattr(results_api, "storage_backend", lambda: "csv")
    monkeypatch.setattr(results_api, "read_station_config", lambda: deepcopy(state))
    monkeypatch.setattr(station_api, "read_station_config", lambda: deepcopy(state))
    app = FastAPI()
    app.include_router(station_api.router)
    app.include_router(results_api.router)
    with TestClient(app) as client:
        nodes = client.get("/stations/hierarchy").json()["nodes"]
        country = nodes[0]["id"]
        params = {"node": country, "window": 4, "include_forecasts": False}
        snapshot = client.get("/results/hierarchy", params=params).json()
        assert snapshot["leaf_count"] == 4
        assert snapshot["actuals"][0]["demand_mw"] == {"value": 100, "reporting": 4, "expected": 4}
        assert snapshot["aligned_through_utc"] == "2026-10-01T00:00:00+00:00"
        assert snapshot["actuals"][-1]["demand_mw"] == {"value": 100, "reporting": 4, "expected": 4}
        assert snapshot["substations"][0]["latest_demand_utc"] == "2026-10-01T00:15:00+00:00"

        path = live_actuals_path(tmp_path, "demo-4")
        initialize(path, "demo-4-featured")
        row = observation(15, 40)
        save_readings(path, [(row["utc_timestamp"], row["actual_demand"])], 2)
        recovered = client.get("/results/hierarchy", params=params).json()
        assert recovered["actuals"][-1]["demand_mw"] == {"value": 100, "reporting": 4, "expected": 4}
        assert client.get("/results/hierarchy", params={"node": "unknown"}).status_code == 404


def test_onboarding_contract_creates_only_leaves_and_reuses_the_parent(monkeypatch, tmp_path):
    class Topics:
        def ensure_station_topics(self, station):
            return {"station": station}

    state = {"active_station": None, "stations": {}}
    monkeypatch.setattr(station_api, "ROOT", tmp_path)
    monkeypatch.setattr(station_api, "_read_model_settings", lambda: {
        "interval_minutes": 15, "history_window_hours": 168, "forecast_horizon_hours": 24,
    })
    monkeypatch.setattr(station_api, "manager", Topics())
    monkeypatch.setattr(station_api, "read_station_config", lambda: deepcopy(state))
    monkeypatch.setattr(station_api, "write_station_config", lambda value: state.update(deepcopy(value)))
    source = BytesIO()
    timestamps = pd.date_range("2025-01-01", periods=1600, freq="15min", tz="UTC")
    with pd.ExcelWriter(source, engine="openpyxl") as writer:
        pd.DataFrame({"utc_timestamp": timestamps.astype(str), "demand": 1000,
                      "temperature": 5.0}).to_excel(writer, sheet_name="Demand", index=False)
    app = FastAPI()
    app.include_router(station_api.router)
    with TestClient(app) as client:
        for station, path in (("sub-1", "Germany\nHesse\nStation A"), ("sub-2", "germany\nhesse\nStation B")):
            response = client.post("/stations/onboard", files={"file": ("source.xlsx", source.getvalue())},
                                   data={"station_id": station, "hierarchy_path": path,
                                         "selected_models": "demand_forecast", "interval_minutes": 15})
            assert response.status_code == 200, response.text
            assert response.json()["hierarchy_path"][0] == {"level": "Country", "name": "Germany"}
        tree = client.get("/stations/hierarchy").json()["nodes"]
        assert tree[0]["name"] == "Germany"
        assert tree[0]["leaf_count"] == 2
        assert set(state["stations"]) == {"sub-1", "sub-2"}


def test_group_finds_shared_runs_after_fast_station_tail_has_moved_past_them(tmp_path, monkeypatch):
    from src.storage.station_hierarchy import group_key

    start = pd.Timestamp("2026-10-01T00:00:00Z")
    step = pd.Timedelta(minutes=15)
    stations = {}
    for station, last, demand, unit in [("fast", 300, 40, "mw"), ("slow", 4, 8, "mwh_per_interval")]:
        folder = tmp_path / "data" / "processed" / "stations" / station
        folder.mkdir(parents=True)
        entry = leaf("Country: Germany", unit)
        entry.update(actuals_file=str(folder / "actuals.csv"), forecasts_file=str(folder / "forecasts.csv"),
                     renewable_forecasts_file=str(folder / "renewable_forecasts.csv"), renewables={"enabled": True})
        stations[station] = entry
        # Each demand offset is an interval when a real observation was saved.
        actuals = [{"station_id": station, "utc_timestamp": (start + step * i).isoformat(), "actual_demand": demand}
                   for i in range(last + 1)]
        pd.DataFrame(actuals).to_csv(entry["actuals_file"], index=False)
        generation = [
            {"station_id": station, "utc_timestamp": row["utc_timestamp"], "generation_type": metric,
             "actual_generation_mwh": value}
            for row in actuals for metric, value in [("solar", 2), ("wind", 3)]
        ]
        pd.DataFrame(generation).to_csv(folder / "renewable_actuals.csv", index=False)
        forecasts = []
        renewable = []
        for i in range(last + 1):
            origin = start + step * i
            for lead in range(1, 3):
                base = {"station_id": station, "forecast_origin_utc": origin.isoformat(),
                        "forecast_timestamp_utc": (origin + step * lead).isoformat(),
                        "forecast_run_id": str(i)}
                forecasts.append({**base, "predicted_demand": demand})
                if station == "fast" or i < last:
                    renewable.extend({**base, "generation_type": k, "predicted_generation_mwh": v}
                                     for k, v in [("solar", 2), ("wind", 3)])
        pd.DataFrame(forecasts).to_csv(entry["forecasts_file"], index=False)
        pd.DataFrame(renewable).to_csv(entry["renewable_forecasts_file"], index=False)

    monkeypatch.setattr(results_api, "ROOT", tmp_path)
    monkeypatch.setattr(results_api, "storage_backend", lambda: "csv")
    monkeypatch.setattr(results_api, "read_station_config", lambda: {"stations": stations})
    group = results_api.hierarchy_results(group_key(["Germany"]), window=2, include_forecasts=True)
    assert group["actuals"][-1]["demand_mw"] == {"value": 72, "reporting": 2, "expected": 2}
    assert group["substations"][0]["latest_demand_utc"] == (start + step * 300).isoformat()
    assert set(group["forecast_origins_utc"].values()) == {(start + step * 3).isoformat()}
    assert len(group["forecasts"]) == 2
    for point in group["forecasts"]:
        assert [point[metric]["value"] for metric in ("demand_mw", "solar_mwh", "wind_mwh", "residual_mwh")] == [72, 4, 6, 8]


def test_optional_group_seed_adds_pre_replay_history_without_advancing_live_clock(tmp_path, monkeypatch):
    stations = {}
    earlier = "2026-09-30T23:45:00Z"
    start = "2026-10-01T00:00:00Z"
    for index, station in enumerate(("a", "b"), 1):
        entry = leaf("Germany", "mwh_per_interval")
        entry.update(
            expected_stream_start_utc=start, actuals_file=f"{station}_actuals.csv",
            history_file=f"{station}_history.csv", renewables={
                "enabled": True, "solar": {"train_file": f"{station}_solar.csv"},
                "wind": {"train_file": f"{station}_wind.csv"},
            },
        )
        stations[station] = entry
        row = {**observation(0, 10 * index), "station_id": station}
        pd.DataFrame([row]).to_csv(tmp_path / entry["actuals_file"], index=False)
        for filename, column, value in (
            (entry["history_file"], "demand", 20 * index),
            (entry["renewables"]["solar"]["train_file"], "Photovoltaics [MWh]", index),
            (entry["renewables"]["wind"]["train_file"], "Wind onshore [MWh]", 2 * index),
        ):
            pd.DataFrame({"utc_timestamp": [earlier, start], column: [value, 999]}).to_csv(
                tmp_path / filename, index=False,
            )

    monkeypatch.setattr(results_api, "ROOT", tmp_path)
    monkeypatch.setattr(results_api, "storage_backend", lambda: "csv")
    monkeypatch.setattr(results_api, "read_station_config", lambda: {"stations": stations})
    group = group_key(["Germany"])
    without_seed = results_api.hierarchy_results(group, window=4, include_forecasts=False)
    with_seed = results_api.hierarchy_results(group, window=4, include_forecasts=False, include_history=True)
    assert len(without_seed["actuals"]) == 1
    assert with_seed["actuals"][-1] == without_seed["actuals"][-1]
    assert with_seed["aligned_through_utc"] == pd.Timestamp(start).isoformat()
    assert with_seed["replay_start_utc"] == pd.Timestamp(start).isoformat()
    point = with_seed["actuals"][0]
    assert point["utc_timestamp"] == pd.Timestamp(earlier).isoformat()
    assert [point[key]["value"] for key in ("demand_mw", "solar_mwh", "wind_mwh", "residual_mwh")] == [
        240, 3, 6, 51,
    ]
    assert all(point[key]["reporting"] == 2 for key in ("demand_mw", "solar_mwh", "wind_mwh", "residual_mwh"))
