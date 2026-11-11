"""Backend-independent regression tests for the optional Cassandra storage path.

These never connect to a live database; live-cluster tests are a deployment gate.
"""
import csv
import sys
import types
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src.storage import cassandra_store, renewable_store


def _time(hour=0):
    return f"2026-10-01T{hour:02d}:00:00+00:00"


def _renewable_rows(kind="solar", n=96):
    origin = pd.Timestamp("2026-10-01T00:00:00Z")
    return [{
        "station_id": "amprion", "generation_type": kind,
        "forecast_issued_at_utc": origin.isoformat(),
        "forecast_origin_utc": origin.isoformat(),
        "forecast_run_id": f"amprion|{kind}|{origin.isoformat()}|v1",
        "forecast_timestamp_utc": (origin + pd.Timedelta(minutes=15*i)).isoformat(),
        "horizon_step": i, "predicted_generation_mwh": 100.0+i,
        "model_version": "v1", "weather_mode": "weather_forecast",
    } for i in range(1,n+1)]


def test_renewable_backend_switch_keeps_csv_and_cassandra_distinct(monkeypatch, tmp_path):
    path = tmp_path / "renewables.csv"
    rows = _renewable_rows(n=2)
    monkeypatch.setenv("STORAGE_BACKEND", "csv")
    assert renewable_store.append_renewable_forecasts(path, rows) == 2
    assert renewable_store.append_renewable_forecasts(path, rows) == 0
    assert len(list(csv.DictReader(path.open()))) == 2

    monkeypatch.setenv("STORAGE_BACKEND", "cassandra")
    called = []
    monkeypatch.setattr(cassandra_store, "write_renewable_forecasts", lambda records: called.append(records) or len(records))
    assert renewable_store.append_renewable_forecasts(path, rows) == 2
    assert called == [rows]
    assert len(list(csv.DictReader(path.open()))) == 2


def test_generation_actuals_never_use_forecast_as_observed(monkeypatch, tmp_path):
    monkeypatch.setenv("STORAGE_BACKEND", "csv")
    rows = [{"station_id":"amprion", "generation_type":"solar",
             "utc_timestamp":_time(), "actual_generation_mwh":8.0, "source":"registered_historical_replay"}]
    path = tmp_path / "generation_actuals.csv"
    assert renewable_store.append_renewable_actuals(path,rows) == 1
    assert renewable_store.append_renewable_actuals(path,rows) == 0
    assert list(csv.DictReader(path.open()))[0]["actual_generation_mwh"] == "8.0"
    monkeypatch.setenv("STORAGE_BACKEND","cassandra")
    monkeypatch.setattr(cassandra_store, "write_renewable_actuals", lambda records: len(records))
    assert renewable_store.append_renewable_actuals(path, rows) == 1
    assert len(list(csv.DictReader(path.open()))) == 1


def _mock_driver(monkeypatch):
    cassandra = types.ModuleType("cassandra")
    cassandra.ConsistencyLevel = SimpleNamespace(ONE="ONE")
    query = types.ModuleType("cassandra.query")
    class BatchStatement:
        def __init__(self, **kwargs): self.rows=[]
        def add(self, stmt, params): self.rows.append((stmt,params))
    query.BatchStatement=BatchStatement
    query.BatchType=SimpleNamespace(UNLOGGED="UNLOGGED")
    monkeypatch.setitem(sys.modules,"cassandra",cassandra)
    monkeypatch.setitem(sys.modules,"cassandra.query",query)
    return BatchStatement


def test_cassandra_writes_are_partitioned_by_station_kind_day_and_monotonic(monkeypatch):
    Batch = _mock_driver(monkeypatch)
    statements = {name:name for name in (
        "renewable_forecast", "bucket_write", "renewable_latest_read",
        "renewable_latest_insert", "renewable_latest_cas")}
    monkeypatch.setattr(cassandra_store,"_PREPARED",statements)
    seen=[]
    class Session:
        def execute(self, stmt, args=None):
            seen.append((stmt,args))
            if stmt == "renewable_latest_read": return SimpleNamespace(one=lambda:None)
            if stmt == "renewable_latest_insert": return SimpleNamespace(one=lambda:SimpleNamespace(**{"[applied]":True}))
            return SimpleNamespace(one=lambda:None)
    session=Session()
    monkeypatch.setattr(cassandra_store,"get_session",lambda:session)
    rows = _renewable_rows("solar",n=2)+_renewable_rows("wind",n=2)
    assert cassandra_store.write_renewable_forecasts(rows)==4
    batches=[stmt for stmt, _ in seen if isinstance(stmt,Batch)]
    assert len(batches) == 2
    assert {params[1] for batch in batches for _,params in batch.rows} == {"solar","wind"}
    assert sorted(args[1] for stmt,args in seen if stmt=="bucket_write") == ["renewable_forecast_day_solar","renewable_forecast_day_wind"]
    assert not any("truncate" in str(stmt).lower() for stmt,args in seen)


def test_cassandra_anomaly_is_saved_at_specific_origin_and_checkpoint(monkeypatch):
    statements = {x:x for x in ("anomaly_write","bucket_write","anomaly_latest_read","anomaly_latest_insert","anomaly_latest_cas")}
    monkeypatch.setattr(cassandra_store,"_PREPARED",statements)
    seen=[]
    class Session:
        def execute(self, stmt, args=None):
            seen.append((stmt,args))
            if stmt == "anomaly_latest_read": return SimpleNamespace(one=lambda:None)
            if stmt == "anomaly_latest_insert":return SimpleNamespace(one=lambda:SimpleNamespace(**{"[applied]":True}))
            return SimpleNamespace(one=lambda:None)
    monkeypatch.setattr(cassandra_store,"get_session", lambda:Session())
    scores=[{"timestamp_utc":_time(1),"combined_anomaly_score":8.1,"classification":"YES","spike_cause":"LEVEL_SPIKE"}]
    cassandra_store.write_anomaly_assessment("amprion",_time(),"checkpoint-hash",scores)
    stored=next(args for stmt,args in seen if stmt=="anomaly_write")
    assert stored[0] == "amprion"
    assert stored[3] == "checkpoint-hash"
    assert '"combined_anomaly_score":8.1' in stored[7]
    assert stored[8] >= 60
    assert any(args[1] == "anomaly_day" for stmt,args in seen if stmt=="bucket_write")


def test_station_reset_preflight_rejects_unmanifested_legacy_cassandra_data(monkeypatch):
    statements={x:x for x in ("bucket_read","latest_actual_read","latest_pointer_read")}
    monkeypatch.setattr(cassandra_store,"_PREPARED",statements)
    class Session:
        def execute(self, stmt, args=None):
            if stmt == "bucket_read":return []
            if stmt == "latest_actual_read":return SimpleNamespace(one=lambda:SimpleNamespace(latest_timestamp=datetime(2026,10,1)))
            return SimpleNamespace(one=lambda:None)
    monkeypatch.setattr(cassandra_store,"get_session",lambda:Session())
    with pytest.raises(RuntimeError,match="Legacy Cassandra"):
        cassandra_store.clear_station_operational_data("amprion",dry_run=True)


def test_storage_backend_rejects_typo_instead_of_silent_csv_fallback(monkeypatch):
    from src.storage.forecast_store import storage_backend
    monkeypatch.setenv("STORAGE_BACKEND","cassanra")
    with pytest.raises(ValueError,match="Unsupported"):
        storage_backend()


def test_read_renewables_endpoint_uses_db_and_does_not_fallback_csv(monkeypatch):
    from backend import results_api
    monkeypatch.setattr(results_api,"storage_backend",lambda:"cassandra")
    monkeypatch.setattr(results_api,"_load_configured_renewable_actuals",lambda station,end_time=None:(pd.DataFrame(),{}))
    monkeypatch.setattr(results_api,"read_renewable_actuals",lambda *a,**kw:[])
    monkeypatch.setattr(results_api,"read_renewable_forecasts",lambda *a,**kw:_renewable_rows("solar")+_renewable_rows("wind"))
    result=results_api.renewables("amprion",limit=500)
    assert len(result["forecasts"]["solar"]) == 96
    assert len(result["forecasts"]["wind"]) == 96
    assert result["diagnostics"]["solar_forecast_complete"]
    assert result["diagnostics"]["wind_forecast_complete"]
    assert result["forecast_storage"] == "cassandra"


def test_csv_renewable_actuals_read_honours_cutoff_without_scanning_full_history(monkeypatch, tmp_path):
    from src.storage import forecast_store
    monkeypatch.setenv("STORAGE_BACKEND", "csv")
    monkeypatch.setattr(forecast_store, "ROOT", tmp_path)
    station_dir = tmp_path / "data" / "processed" / "stations" / "amprion"
    station_dir.mkdir(parents=True)
    path = station_dir / "renewable_actuals.csv"
    path.write_text(
        "station_id,generation_type,utc_timestamp,actual_generation_mwh,source\n"
        "amprion,solar,2026-10-01T00:00:00Z,100,replay\n"
        "amprion,wind,2026-10-01T00:15:00Z,200,replay\n"
        "amprion,solar,2026-10-01T00:30:00Z,300,replay\n",
        encoding="utf-8",
    )
    records = renewable_store.read_renewable_actuals(
        "amprion", limit=50, end_time="2026-10-01T00:15:00Z"
    )
    assert len(records) == 2
    assert {r["generation_type"] for r in records} == {"solar", "wind"}


def test_station_partition_clear_uses_only_station_scoped_queries(monkeypatch):
    statements={x:x for x in ("bucket_read","latest_actual_read","latest_pointer_read")}
    monkeypatch.setattr(cassandra_store, "_PREPARED", statements)
    seen=[]
    class Session:
        def execute(self, stmt, args=None):
            seen.append((stmt,args))
            if stmt == "bucket_read":
                if args[1] == "renewable_forecast_day_solar":
                    return [SimpleNamespace(bucket_value="2026-10-01")]
                return []
            return SimpleNamespace(one=lambda:None)
    monkeypatch.setattr(cassandra_store,"get_session",lambda:Session())
    result=cassandra_store.clear_station_operational_data("amprion", dry_run=False)
    assert result["cleared"]
    deletes=[(stmt,args) for stmt,args in seen if isinstance(stmt,str) and stmt.startswith("DELETE")]
    expected=[(stmt,args) for stmt,args in deletes if "renewable_forecasts_by_station_day" in stmt]
    assert len(expected)==1
    assert expected[0][1]==("amprion","solar",pd.Timestamp("2026-10-01").date())
    assert all(args[0] == "amprion" for stmt,args in deletes)
    assert not any("TRUNCATE" in stmt for stmt,args in deletes)
