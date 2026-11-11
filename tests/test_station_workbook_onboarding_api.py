from __future__ import annotations

import asyncio
from copy import deepcopy
from io import BytesIO
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from starlette.datastructures import UploadFile

from backend import station_api


class _TopicManager:
    def ensure_station_topics(self, station: str):
        return {"station": station, "created": True}


def _demand_workbook(path: Path, rows: int = 1600) -> Path:
    timestamps = pd.date_range("2025-01-01", periods=rows, freq="15min", tz="UTC")
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame({
            "utc_timestamp": timestamps.astype(str),
            "demand": 10000 + pd.Series(range(rows)) * 0.2,
            "temperature": 5.0,
        }).to_excel(writer, sheet_name="Demand", index=False)
    return path


def test_onboard_registers_demand_forecast_and_stores_selection(monkeypatch, tmp_path):
    state = {"active_station": None, "stations": {}}

    monkeypatch.setattr(station_api, "ROOT", tmp_path)
    monkeypatch.setattr(station_api, "_read_model_settings", lambda: {
        "interval_minutes": 15,
        "history_window_hours": 168,
        "forecast_horizon_hours": 24,
    })
    monkeypatch.setattr(station_api, "read_station_config", lambda: deepcopy(state))

    def write_config(updated):
        state.clear()
        state.update(deepcopy(updated))

    monkeypatch.setattr(station_api, "write_station_config", write_config)
    monkeypatch.setattr(station_api, "manager", _TopicManager())

    workbook_path = _demand_workbook(tmp_path / "request.xlsx")
    buffer = BytesIO(workbook_path.read_bytes())
    upload = UploadFile(filename="station.xlsx", file=buffer)

    response = asyncio.run(station_api.onboard_station(
        station_id="Demo Station",
        file=upload,
        display_name="Demo Station",
        stream_fraction=0.20,
        timestamp_format="auto",
        source_timezone="UTC",
        interval_minutes=15,
        selected_models="demand_forecast",
        raw_topic=None,
        featured_topic=None,
    ))

    assert response["registered"] is True
    assert response["enabled_models"] == ["demand_forecast"]
    assert response["topics_ready"] is True
    entry = state["stations"]["demo-station"]
    assert entry["model_components"]["demand_forecast"]["enabled"] is True
    assert entry["model_components"]["solar_forecast"]["enabled"] is False
    assert (tmp_path / entry["training_file"]).is_file()
    assert (tmp_path / entry["history_file"]).is_file()
    assert (tmp_path / entry["input_file"]).is_file()
    assert (tmp_path / entry["archived_source_file"]).is_file()
    assert entry["history_rows"] == entry["history_steps"]


def test_onboard_defaults_to_all_four_model_flags_at_selection_parser_level():
    # The endpoint's empty selected_models form value resolves to all four;
    # detailed dataset gates are tested separately in test_station_workbook_loader.py.
    from src.ingestion.workbook_loader import parse_selected_models
    assert parse_selected_models("") == [
        "demand_forecast", "solar_forecast", "wind_forecast", "residual_anomaly"
    ]


def test_default_onboarding_registers_all_four_components_and_final_paths(monkeypatch, tmp_path):
    state = {"active_station": None, "stations": {}}

    monkeypatch.setattr(station_api, "ROOT", tmp_path)
    monkeypatch.setattr(station_api, "_read_model_settings", lambda: {
        "interval_minutes": 15,
        "history_window_hours": 168,
        "forecast_horizon_hours": 24,
    })
    monkeypatch.setattr(station_api, "read_station_config", lambda: deepcopy(state))

    def write_config(updated):
        state.clear()
        state.update(deepcopy(updated))

    monkeypatch.setattr(station_api, "write_station_config", write_config)
    monkeypatch.setattr(station_api, "manager", _TopicManager())
    # Row-count gates are separately validated in the loader/contract tests;
    # this test focuses on default selection, config wiring and final paths.
    monkeypatch.setattr(station_api, "_validate_model_row_counts", lambda *args: {"test": "validated"})

    timestamps = pd.date_range("2024-01-01", periods=4000, freq="15min", tz="UTC")
    workbook_path = tmp_path / "all_components.xlsx"
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        pd.DataFrame({
            "utc_timestamp": timestamps.astype(str),
            "demand": 10000.0,
            "temperature": 5.0,
        }).to_excel(writer, sheet_name="Demand", index=False)
        pd.DataFrame({
            "utc_timestamp": timestamps.astype(str),
            "Photovoltaics [MWh]": 0.0,
            "cloud_cover_mean_pct": 80.0,
            "shortwave_radiation_mean_prev_hour_w_m2": 0.0,
            "forecast_cloud_cover_mean_pct": 82.0,
            "forecast_shortwave_radiation_mean_prev_hour_w_m2": 0.0,
            "Photovoltaics_installed_capacity [MW]": 12000.0,
        }).to_excel(writer, sheet_name="Solar", index=False)
        pd.DataFrame({
            "utc_timestamp": timestamps.astype(str),
            "Wind onshore [MWh]": 15.0,
            "wind_speed_100m_mean_ms": 8.0,
            "wind_direction_100m_circular_mean_deg": 225.0,
            "Wind_onshore_installed_capacity [MW]": 18000.0,
            "future_wind_speed": 8.2,
            "future_wind_direction": 226.0,
        }).to_excel(writer, sheet_name="Wind", index=False)

    upload = UploadFile(filename="all_components.xlsx", file=BytesIO(workbook_path.read_bytes()))
    response = asyncio.run(station_api.onboard_station(
        station_id="all-four",
        file=upload,
        display_name="All Four",
        stream_fraction=0.20,
        timestamp_format="auto",
        source_timezone="UTC",
        interval_minutes=15,
        selected_models="",  # blank means the four default components
        raw_topic=None,
        featured_topic=None,
    ))

    expected = ["demand_forecast", "solar_forecast", "wind_forecast", "residual_anomaly"]
    assert response["enabled_models"] == expected
    entry = state["stations"]["all-four"]
    assert entry["enabled_models"] == expected
    assert all(entry["model_components"][name]["enabled"] for name in expected)
    assert entry["renewables"]["enabled"] is True
    assert entry["renewables"]["serving_adapter_supported"] is True
    for relative_path in response["files"].values():
        assert (tmp_path / relative_path).is_file()
    assert (tmp_path / entry["archived_source_file"]).is_file()
    assert ".all-four_onboard_" not in str(response["files"])
