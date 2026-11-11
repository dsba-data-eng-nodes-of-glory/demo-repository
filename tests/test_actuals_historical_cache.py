from __future__ import annotations

import pandas as pd
import yaml

from backend import results_api


def _clear_actuals_caches() -> None:
    with results_api._HISTORICAL_ACTUALS_CACHE_LOCK:
        results_api._HISTORICAL_ACTUALS_CACHE.clear()
    with results_api._CACHE_LOCK:
        results_api._CACHE.clear()


def test_historical_actuals_are_prepared_cached_and_invalidated(tmp_path, monkeypatch):
    history_path = tmp_path / "demand_train.csv"
    input_path = tmp_path / "demand_test.csv"
    config_path = tmp_path / "stations.yaml"

    pd.DataFrame({
        "utc_timestamp": ["30-09-2026 23:45", "01-10-2026 00:00"],
        "demand": [100.0, 110.0],
    }).to_csv(history_path, index=False)
    pd.DataFrame({
        "utc_timestamp": ["01-10-2026 00:00", "01-10-2026 00:15"],
        "demand": [111.0, 120.0],
    }).to_csv(input_path, index=False)
    config_path.write_text(yaml.safe_dump({
        "stations": {
            "amprion": {
                "history_file": str(history_path),
                "input_file": str(input_path),
            }
        }
    }), encoding="utf-8")

    monkeypatch.setattr(results_api, "STATION_CONFIG_FILE", config_path)
    monkeypatch.setattr(results_api, "ROOT", tmp_path)
    _clear_actuals_caches()

    first = results_api._read_historical_actuals("amprion")
    second = results_api._read_historical_actuals("amprion")

    assert first is second  # Cache hit reuses prepared data; no full DataFrame copy.
    assert "_timestamp_key" in first.columns
    assert str(first["_timestamp_key"].dtype).startswith("datetime64[ns, UTC]")
    assert first["utc_timestamp"].tolist() == [
        "2026-09-30T23:45:00Z",
        "2026-10-01T00:00:00Z",
        "2026-10-01T00:15:00Z",
    ]
    # The later configured source (demand_test.csv) still wins on duplicates.
    assert first.loc[first["utc_timestamp"] == "2026-10-01T00:00:00Z", "actual_demand"].iloc[0] == 111.0

    live = pd.DataFrame([
        {"utc_timestamp": "2026-10-01T00:15:00Z", "actual_demand": 121.0},
        {"utc_timestamp": "2026-10-01T00:30:00Z", "actual_demand": 130.0},
    ])
    combined = results_api._combine_actuals(
        first, live, station="amprion", limit=10,
        end_time="2026-10-01T00:30:00Z",
    )
    assert combined["utc_timestamp"].tolist() == [
        "2026-09-30T23:45:00Z",
        "2026-10-01T00:00:00Z",
        "2026-10-01T00:15:00Z",
        "2026-10-01T00:30:00Z",
    ]
    assert combined["actual_demand"].tolist() == [100.0, 111.0, 121.0, 130.0]
    assert "_timestamp_key" not in combined.columns
    assert "_priority" not in combined.columns

    # Changing a source file signature must create a newly prepared cache entry.
    with input_path.open("a", encoding="utf-8") as handle:
        handle.write("01-10-2026 00:30,140.0\n")
    refreshed = results_api._read_historical_actuals("amprion")
    assert refreshed is not first
    assert len(refreshed) == 4
    assert refreshed.iloc[-1]["actual_demand"] == 140.0

    _clear_actuals_caches()
