"""Regression tests for the replacement residual-load operational component."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException

from src.models.residual.operational import aligned_residual_forecast, available_origins, build_residual_history
from src.models.residual.training_data import build_training_frame
from src.models.residual.snapshots import save_snapshot, read_snapshot, recent_snapshots, snapshot_path
from backend import station_api, model_api

ORIGIN = pd.Timestamp('2026-10-01T00:00:00Z')

def forecasts(origin=ORIGIN, count=96):
    targets=pd.date_range(origin + pd.Timedelta(minutes=15),periods=count,freq='15min',tz='UTC')
    demand=[{'forecast_origin_utc':origin.isoformat(),'forecast_timestamp_utc':t.isoformat(), 'predicted_demand':2000.0} for t in targets]
    renewable=[]
    for kind,value in [('solar',100.0),('wind',300.0)]:
        renewable += [{'forecast_origin_utc':origin.isoformat(),'forecast_timestamp_utc':t.isoformat(), 'generation_type':kind,'predicted_generation_mwh':value} for t in targets]
    return demand,renewable

def test_residual_calculated_by_same_origin_and_15min_mwh():
    d,r=forecasts()
    result=aligned_residual_forecast(d,r,ORIGIN,solar_capacity_mw=5000,wind_capacity_mw=5000)
    assert len(result)==96
    assert result['residual'].eq(1600).all()
    assert result['demand'].eq(2000).all()
    assert result.index[0] == ORIGIN + pd.Timedelta(minutes=15)
    mw=aligned_residual_forecast(d,r,ORIGIN,5000,5000,demand_unit='mw')
    assert mw['residual'].eq(100).all()  # 2000 MW = 500 MWh/15 min

def test_missing_target_or_origin_blocks_residual_alert():
    demand, renewable=forecasts()
    renewable.pop(0)
    with pytest.raises(ValueError,match='96 aligned'):
        aligned_residual_forecast(demand,renewable,ORIGIN,5000,5000)
    assert available_origins(demand,renewable)==[ORIGIN] # origin considered, completeness checked separately
    demand,renewable=forecasts()
    renewable[0]['forecast_origin_utc']=(ORIGIN + pd.Timedelta(minutes=15)).isoformat()
    with pytest.raises(ValueError,match='96 aligned'):
        aligned_residual_forecast(demand,renewable,ORIGIN,5000,5000)

def test_generation_exceeding_capacity_is_rejected():
    d,r=forecasts()
    with pytest.raises(ValueError,match='exceeds installed'):
        aligned_residual_forecast(d,r,ORIGIN,solar_capacity_mw=100,wind_capacity_mw=100)

def test_historical_training_joins_three_sources_and_normalizes_mw():
    timestamps=pd.date_range('2025-01-01',periods=96*31,freq='15min',tz='UTC')
    demand=pd.DataFrame({'utc_timestamp':timestamps,'demand':2000.})
    solar=pd.DataFrame({'utc_timestamp':timestamps,'Photovoltaics [MWh]':100.,'Photovoltaics_installed_capacity [MW]':5000.})
    wind=pd.DataFrame({'utc_timestamp':timestamps,'Wind onshore [MWh]':300.,'Wind_onshore_installed_capacity [MW]':5000.})
    result=build_training_frame(demand,solar,wind,demand_unit='mw')
    assert len(result)==96*31
    assert result['Residual Load'].eq(100.).all()
    solar=solar.drop(index=[100])
    with pytest.raises(ValueError,match='gaps'):
        build_training_frame(demand,solar,wind)

def test_residual_snapshots_are_stable_and_station_scoped(tmp_path):
    path = snapshot_path(tmp_path, 'amprion')
    origin = ORIGIN.isoformat()
    score_rows = [{'timestamp_utc':origin, 'residual_load':1700.0, 'upward_highlight':True}]
    save_snapshot(path, origin, 'fingerprint', score_rows)
    assert read_snapshot(path, origin, 'fingerprint') == score_rows
    assert read_snapshot(path, origin, 'another') is None
    save_snapshot(path, origin, 'fingerprint', [{'residual_load':9999.0}])
    assert read_snapshot(path, origin, 'fingerprint') == score_rows  # replay-safe
    save_snapshot(path, origin, 'new-calibration', [{'residual_load':1800.0}])
    assert read_snapshot(path, origin, 'new-calibration') == [{'residual_load':1800.0}]
    assert recent_snapshots(path, 10)[0]['model_fingerprint'] == 'new-calibrat'
    assert recent_snapshots(path, 10)[1]['flagged_points'] == 1
    assert snapshot_path(tmp_path, 'tennet') != path


def test_old_sqlite_scores_remain_accessible_after_state_machine_removed(tmp_path):
    import sqlite3
    legacy = tmp_path/'data'/'processed'/'stations'/'amprion'/'residual_alerts.sqlite'
    legacy.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(legacy) as db:
        db.execute('CREATE TABLE alert_events (event_id TEXT PRIMARY KEY,payload TEXT)')
        db.execute('INSERT INTO alert_events VALUES (?,?)', ('RL-0001', '{"state":"WARNING"}'))
    assert snapshot_path(tmp_path,'amprion') == legacy
    save_snapshot(legacy,ORIGIN.isoformat(),'a',[{'upward_highlight':True}])
    assert read_snapshot(legacy,ORIGIN.isoformat(),'a') == [{'upward_highlight':True}]
    with sqlite3.connect(legacy) as db:
        assert db.execute('SELECT count(*) FROM alert_events').fetchone()[0] == 1

def test_per_station_threshold_configuration_roundtrip(monkeypatch):
    state={'stations':{'amprion':{'residual_model_checkpoint':'models/a.pt'},'tennet':{}}}
    monkeypatch.setattr(station_api,'read_station_config',lambda:deepcopy(state))
    monkeypatch.setattr(station_api,'write_station_config',lambda x:(state.clear(),state.update(deepcopy(x))))
    result=station_api.put_residual_settings('amprion',station_api.ResidualDetectionSettingsRequest(
        decision_rule='both',stat_threshold=5.5,ae_threshold=20.0,upward_only=True))
    assert result['settings']['stat_threshold']==5.5
    assert station_api.get_residual_settings('amprion')['settings']['ae_threshold']==20.
    assert 'residual_detection_settings' not in state['stations']['tennet']
    with pytest.raises(HTTPException):station_api.put_residual_settings('none',station_api.ResidualDetectionSettingsRequest())

def test_old_demand_only_component_is_not_active(monkeypatch):
    monkeypatch.setattr(model_api,'read_station_config',lambda: {'stations':{'s':{'model_components':{'residual_anomaly':{'enabled':True}}}}})
    response=model_api.model_components('s')
    assert 'demand_anomaly' not in response['component_algorithms']
    assert response['component_algorithms']['residual_anomaly']==['dense_autoencoder']


def test_legacy_thresholds_supported_but_alert_rules_discarded():
    from src.models.residual.service import settings_for
    legacy={'residual_alert_settings': {'decision_rule':'either','stat_threshold':4.2,
              'ae_threshold':21.0,'min_sustain_intervals':9,'confirm_hits':2}}
    assert settings_for(legacy)=={'decision_rule':'either','stat_threshold':4.2,
                                 'ae_threshold':21.0,'upward_only':True}


def test_residual_spike_intervals_highlight_only_forecast_targets():
    import plotly.graph_objects as go
    from dashboard.residual_plot import flagged_intervals, add_spike_intervals
    timestamps=pd.date_range(ORIGIN+pd.Timedelta(minutes=15),periods=6,freq='15min',tz='UTC')
    flags=[False,True,True,False,True,False]
    payload={'records':[{'timestamp_utc':ts.isoformat(),'upward_highlight':v,'residual_load':1000+i}
                        for i,(ts,v) in enumerate(zip(timestamps,flags))]}
    windows=flagged_intervals(payload)
    assert len(windows)==2
    assert windows[0][1]-windows[0][0] == pd.Timedelta(minutes=30)
    demand=pd.DataFrame({'forecast_timestamp_utc':timestamps,'predicted_demand':[4000]*6})
    fig=go.Figure()
    assert add_spike_intervals(fig,payload,demand,'forecast_timestamp_utc','predicted_demand')==2
    assert len(fig.layout.shapes)==2
    assert len(fig.data)==1 and len(fig.data[0].x)==3
    assert all(float(y)==4000 for y in fig.data[0].y)
    assert add_spike_intervals(go.Figure(),{'records':[]},demand,'forecast_timestamp_utc','predicted_demand')==0


def test_monitoring_source_no_longer_renders_score_chart_or_sustained_ledger():
    from pathlib import Path
    monitoring=(Path(__file__).parents[1]/'dashboard'/'pages'/'monitoring.py').read_text()
    assert "st.subheader('Residual Load Spike Detection')" in monitoring
    assert 'residual_score_fig' not in monitoring
    assert 'sustained risk alerts' not in monitoring.lower()
    assert 'add_spike_intervals' in monitoring


def test_sustained_event_api_removed_and_threshold_save_discards_old_policy(monkeypatch):
    from backend import results_api
    assert not any(getattr(route,'path','').endswith('/residual/alerts') for route in results_api.router.routes)
    state={'stations':{'s':{'residual_alert_settings':{'stat_threshold':3.0,'ae_threshold':18.0,
                                         'confirm_hits':2,'resolve_misses':5}}}}
    monkeypatch.setattr(station_api,'read_station_config',lambda:deepcopy(state))
    monkeypatch.setattr(station_api,'write_station_config',lambda x:(state.clear(),state.update(deepcopy(x))))
    result=station_api.put_residual_settings('s',station_api.ResidualDetectionSettingsRequest(
        stat_threshold=4.0,ae_threshold=19.0))
    assert result['settings']['stat_threshold']==4.0
    assert 'residual_alert_settings' not in state['stations']['s']
    assert 'confirm_hits' not in result['settings']
