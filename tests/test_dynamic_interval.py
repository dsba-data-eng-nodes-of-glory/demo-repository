from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.forecast_contract import derive_forecast_contract
from src.models.tabular_forecasting import build_origin_features


@pytest.mark.parametrize(
    "interval,history_steps,horizon_steps",
    [
        (5, 2016, 288),
        (10, 1008, 144),
        (15, 672, 96),
        (20, 504, 72),
        (30, 336, 48),
        (60, 168, 24),
        (90, 112, 16),
        (120, 84, 12),
    ],
)
def test_forecast_contract_derives_steps_for_selected_interval(
    interval: int, history_steps: int, horizon_steps: int
) -> None:
    contract = derive_forecast_contract(interval, {
        "history_window_hours": 168,
        "forecast_horizon_hours": 24,
    })
    assert contract["history_steps"] == history_steps
    assert contract["horizon_steps"] == horizon_steps
    assert contract["interval_minutes"] == interval


def test_forecast_contract_rejects_interval_that_does_not_align_with_windows() -> None:
    with pytest.raises(ValueError, match="does not divide"):
        derive_forecast_contract(7, {
            "history_window_hours": 168,
            "forecast_horizon_hours": 24,
        })


@pytest.mark.parametrize("interval", [15, 30, 60])
def test_tree_feature_builder_uses_interval_specific_history_and_horizon(interval: int) -> None:
    history_steps = 10080 // interval
    horizon_steps = 1440 // interval
    timestamps = pd.date_range(
        "2026-01-01T00:00:00Z", periods=history_steps, freq=f"{interval}min", tz="UTC"
    )
    idx = np.arange(history_steps, dtype=float)
    history = pd.DataFrame({
        "utc_timestamp": timestamps,
        "demand": 5000 + np.sin(idx / 10) * 100,
        "temperature": 10 + np.cos(idx / 20),
    })
    future = pd.date_range(
        timestamps[-1] + pd.Timedelta(minutes=interval),
        periods=horizon_steps, freq=f"{interval}min", tz="UTC",
    )
    features = build_origin_features(history, future, interval_minutes=interval)
    assert len(features) == horizon_steps
    assert features["horizon_step"].iloc[0] == 1
    assert features["horizon_step"].iloc[-1] == horizon_steps
    assert np.isfinite(features.to_numpy(dtype=float)).all()


def test_csv_loader_accepts_station_selected_30_minute_interval(tmp_path) -> None:
    from src.ingestion.file_loader import load_station_csv

    path = tmp_path / "station_30min.csv"
    path.write_text(
        "timestamp,load_mw,temp_c\n"
        "2026-01-01T00:00:00Z,100,10.0\n"
        "2026-01-01T00:30:00Z,110,10.2\n"
        "2026-01-01T01:00:00Z,105,10.1\n",
        encoding="utf-8",
    )
    frame = load_station_csv(path, interval_minutes=30, timestamp_format="auto")
    assert len(frame) == 3
    assert frame["utc_timestamp"].diff().dropna().eq(pd.Timedelta(minutes=30)).all()


def test_csv_loader_rejects_data_that_does_not_match_selected_interval(tmp_path) -> None:
    from src.ingestion.file_loader import load_station_csv

    path = tmp_path / "station_interval_mismatch.csv"
    path.write_text(
        "timestamp,load_mw,temp_c\n"
        "2026-01-01T00:00:00Z,100,10.0\n"
        "2026-01-01T00:30:00Z,110,10.2\n"
        "2026-01-01T01:00:00Z,105,10.1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="selected source interval is 15 minutes"):
        load_station_csv(path, interval_minutes=15, timestamp_format="auto")
