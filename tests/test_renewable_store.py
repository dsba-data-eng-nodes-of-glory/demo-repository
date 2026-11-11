from __future__ import annotations

import csv

from src.storage.renewable_store import append_renewable_forecasts


def test_renewable_forecast_storage_is_idempotent_and_cached(tmp_path):
    path = tmp_path / "renewable_forecasts.csv"
    records = [
        {
            "station_id": "amprion",
            "generation_type": kind,
            "forecast_issued_at_utc": "2026-11-09T10:00:00Z",
            "forecast_origin_utc": "2026-11-09T10:00:00Z",
            "forecast_run_id": "amprion|origin|" + kind + "|v1",
            "forecast_timestamp_utc": "2026-11-09T10:15:00Z",
            "horizon_step": 1,
            "predicted_generation_mwh": 20.0,
            "model_version": "v1",
            "weather_mode": "forecast",
        }
        for kind in ("solar", "wind")
    ]

    assert append_renewable_forecasts(path, records) == 2
    assert append_renewable_forecasts(path, records) == 0
    with path.open(newline="", encoding="utf-8") as handle:
        saved = list(csv.DictReader(handle))
    assert len(saved) == 2
