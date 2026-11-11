from __future__ import annotations

import pandas as pd

from src.models import forecast_weather


def test_forecast_weather_lookup_is_cached_and_returns_independent_frames(tmp_path, monkeypatch):
    path = tmp_path / "forecast_weather.csv"
    pd.DataFrame(
        {
            "utc_timestamp": ["2026-10-01T00:00:00Z", "2026-10-01T00:15:00Z"],
            "forecast_temperature": [10.0, 11.0],
        }
    ).to_csv(path, index=False)

    original_read_csv = forecast_weather.pd.read_csv
    calls = {"count": 0}

    def counted_read_csv(*args, **kwargs):
        calls["count"] += 1
        return original_read_csv(*args, **kwargs)

    monkeypatch.setattr(forecast_weather.pd, "read_csv", counted_read_csv)
    first = forecast_weather.load_weather_lookup(
        path, "utc_timestamp", ["forecast_temperature"]
    )
    first.iloc[0, 0] = -999.0
    second = forecast_weather.load_weather_lookup(
        path, "utc_timestamp", ["forecast_temperature"]
    )

    assert calls["count"] == 1
    assert second.iloc[0, 0] == 10.0
