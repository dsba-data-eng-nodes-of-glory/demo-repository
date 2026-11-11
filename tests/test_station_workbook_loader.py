from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from openpyxl import Workbook

from src.ingestion.workbook_loader import (
    DEFAULT_SELECTED_MODELS,
    load_station_workbook,
    parse_selected_models,
)


def _make_workbook(path: Path, rows: int = 8) -> Path:
    timestamps = pd.date_range("2025-01-01", periods=rows, freq="15min", tz="UTC")
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame({
            "utc_timestamp": timestamps.astype(str),
            "demand": range(100, 100 + rows),
            "temperature": [5.0] * rows,
        }).to_excel(writer, sheet_name="Demand", index=False)
        pd.DataFrame({
            "utc_timestamp": timestamps.astype(str),
            "Photovoltaics [MWh]": [0.0] * rows,
            "cloud_cover_mean_pct": [80.0] * rows,
            "shortwave_radiation_mean_prev_hour_w_m2": [0.0] * rows,
            "forecast_cloud_cover_mean_pct": [80.0] * rows,
            "forecast_shortwave_radiation_mean_prev_hour_w_m2": [0.0] * rows,
            "Photovoltaics_installed_capacity [MW]": [1000.0] * rows,
        }).to_excel(writer, sheet_name="Solar", index=False)
        pd.DataFrame({
            "utc_timestamp": timestamps.astype(str),
            "Wind onshore [MWh]": [10.0] * rows,
            "wind_speed_100m_mean_ms": [7.0] * rows,
            "wind_direction_100m_circular_mean_deg": [220.0] * rows,
            "Wind_onshore_installed_capacity [MW]": [2000.0] * rows,
            "future_wind_speed": [7.2] * rows,
            "future_wind_direction": [221.0] * rows,
        }).to_excel(writer, sheet_name="Wind", index=False)
    return path


def test_default_model_selection_enables_all_four():
    assert parse_selected_models(None) == DEFAULT_SELECTED_MODELS
    assert parse_selected_models("") == DEFAULT_SELECTED_MODELS


def test_individual_model_deselection_is_parsed():
    assert parse_selected_models('["demand_forecast", "wind_forecast"]') == [
        "demand_forecast", "wind_forecast"
    ]
    assert parse_selected_models("solar_forecast,residual_anomaly") == [
        "solar_forecast", "residual_anomaly"
    ]


def test_invalid_or_empty_model_selection_is_rejected():
    with pytest.raises(ValueError, match="Unknown model"):
        parse_selected_models("demand_forecast,unknown")
    with pytest.raises(ValueError, match="at least one model"):
        parse_selected_models(" , ")


def test_loader_reads_all_three_worksheets_and_canonical_columns(tmp_path):
    path = _make_workbook(tmp_path / "station.xlsx")
    datasets, diagnostics = load_station_workbook(path)
    assert set(datasets) == {"demand", "solar", "wind"}
    assert list(datasets["solar"].columns)[1:] == [
        "Photovoltaics [MWh]",
        "cloud_cover_mean_pct",
        "shortwave_radiation_mean_prev_hour_w_m2",
        "forecast_cloud_cover_mean_pct",
        "forecast_shortwave_radiation_mean_prev_hour_w_m2",
        "Photovoltaics_installed_capacity [MW]",
    ]
    assert diagnostics["selected_models"] == DEFAULT_SELECTED_MODELS
    assert diagnostics["worksheets"]["Demand"]["rows"] == 8


def test_deselected_sheets_are_not_required_or_loaded(tmp_path):
    path = tmp_path / "demand_only.xlsx"
    timestamps = pd.date_range("2025-01-01", periods=8, freq="15min", tz="UTC")
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame({
            "timestamp": timestamps.astype(str),
            "demand_mw": range(100, 108),
            "temperature": [5.0] * 8,
        }).to_excel(writer, sheet_name="Demand", index=False)
    datasets, _ = load_station_workbook(path, selected_models=["demand_forecast"])
    assert list(datasets) == ["demand"]
    assert list(datasets["demand"].columns) == ["utc_timestamp", "demand", "temperature"]


def test_loader_rejects_irregular_timestamps_with_sheet_name(tmp_path):
    path = _make_workbook(tmp_path / "irregular.xlsx")
    # Replace one timestamp with a one-hour jump.
    demand = pd.read_excel(path, sheet_name="Demand")
    demand.loc[3, "utc_timestamp"] = "2025-01-01T00:50:00Z"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        demand.to_excel(writer, sheet_name="Demand", index=False)
    with pytest.raises(ValueError, match="Worksheet 'Demand'.*irregular"):
        load_station_workbook(path, selected_models=["demand_forecast"])


def test_loader_rejects_invalid_solar_physical_values(tmp_path):
    path = _make_workbook(tmp_path / "bad_solar.xlsx")
    solar = pd.read_excel(path, sheet_name="Solar")
    solar.loc[2, "cloud_cover_mean_pct"] = 120
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame({
            "utc_timestamp": pd.date_range("2025-01-01", periods=8, freq="15min", tz="UTC").astype(str),
            "demand": range(100, 108),
            "temperature": [5.0] * 8,
        }).to_excel(writer, sheet_name="Demand", index=False)
        solar.to_excel(writer, sheet_name="Solar", index=False)
        pd.DataFrame({
            "utc_timestamp": pd.date_range("2025-01-01", periods=8, freq="15min", tz="UTC").astype(str),
            "Wind onshore [MWh]": [10.0] * 8,
            "wind_speed_100m_mean_ms": [7.0] * 8,
            "wind_direction_100m_circular_mean_deg": [220.0] * 8,
            "Wind_onshore_installed_capacity [MW]": [2000.0] * 8,
            "future_wind_speed": [7.2] * 8,
            "future_wind_direction": [221.0] * 8,
        }).to_excel(writer, sheet_name="Wind", index=False)
    with pytest.raises(ValueError, match="cloud-cover values must be within 0–100"):
        load_station_workbook(path, selected_models=["solar_forecast"])
