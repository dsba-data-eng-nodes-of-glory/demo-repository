"""HTTP dashboard requests should not synchronously compute residual AE or read cold renewable files."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pandas as pd

from backend import results_api
from src.models.residual.snapshots import save_snapshot


def _run_cached(tmp_path, monkeypatch, settings=None, origin=None, lookback=0):
    monkeypatch.setattr(results_api, 'ROOT', tmp_path)
    monkeypatch.setattr(results_api, '_station_configuration',
                        lambda station: {'residual_detection_settings': settings or {}})
    # The synchronous scoring path must never run when cached_only=True.
    monkeypatch.setattr(results_api, 'forecasts',
                        lambda *a, **kw: (_ for _ in ()).throw(AssertionError('forecast CSV scanned')))
    return results_api.residual_forecast_analysis('amprion', history_runs=2,
                                                   cached_only=True, origin_utc=origin, lookback_minutes=lookback)


def test_cached_only_is_fast_and_returns_pending_without_scoring(tmp_path, monkeypatch):
    out = _run_cached(tmp_path, monkeypatch, origin='2026-10-01T00:00:00Z')
    assert out['pending'] is True
    assert out['records'] == []
    assert not out['available']


def test_cached_only_returns_existing_origin_without_loading_models(tmp_path, monkeypatch):
    origin = '2026-10-01T00:00:00+00:00'
    file = tmp_path / 'data/processed/stations/amprion/residual_assessments.sqlite'
    rows = [{'timestamp_utc':'2026-10-01T00:15:00Z',
             'statistical_score':1.0,'statistical_threshold':10.0,
             'autoencoder_score':2.0,'autoencoder_threshold':12.0,
             'confirmation':'NORMAL','classification':'NO',
             'upward_highlight':False, 'residual_ramp_30m':-2.0,
             'threshold_source':'checkpoint_calibration'}]
    save_snapshot(file, origin, 'model-forecast-v1', rows)
    out = _run_cached(tmp_path, monkeypatch, origin='2026-10-01T00:00:00Z')
    assert out['available'] is True
    assert out['records'] == rows
    assert out['origin_utc'] == origin
    assert out['history'][0]['record_count'] == 1


def test_changed_override_does_not_reuse_old_snapshot(tmp_path, monkeypatch):
    origin = '2026-10-01T00:00:00+00:00'
    file = tmp_path / 'data/processed/stations/amprion/residual_assessments.sqlite'
    save_snapshot(file, origin, 'hash', [{
        'statistical_score':1.0, 'statistical_threshold':10.0,
        'autoencoder_score':2.0, 'autoencoder_threshold':12.0,
        'confirmation':'NORMAL','classification':'NO',
        'upward_highlight':False,'threshold_source':'checkpoint_calibration'}])
    out = _run_cached(tmp_path, monkeypatch, {'stat_threshold':8.0}, origin)
    assert out['pending'] is True
    assert not out['available']


def test_renewable_cache_warmup_returns_immediately(monkeypatch):
    release = Event()
    calls = []

    def slow_loader(station, end_time=None):
        calls.append(station)
        assert release.wait(timeout=2.0)
        return pd.DataFrame(), {'solar': {}, 'wind': {}}

    monkeypatch.setattr(results_api, '_load_configured_renewable_actuals', slow_loader)
    monkeypatch.setattr(results_api, '_RENEWABLE_WARM_JOBS', {})
    with ThreadPoolExecutor(max_workers=1) as executor:
        monkeypatch.setattr(results_api, '_RENEWABLE_WARM_POOL', executor)
        assert results_api._renewable_warmed_or_schedule('amprion') is False
        assert results_api._renewable_warmed_or_schedule('amprion') is False
        release.set()
        results_api._RENEWABLE_WARM_JOBS['amprion'].result(timeout=2.0)
        assert results_api._renewable_warmed_or_schedule('amprion') is True
        assert calls == ['amprion']


def _score_record(stat=10.0, ae=12.0, source='checkpoint_calibration'):
    return {'timestamp_utc': '2026-10-01T00:15:00Z',
            'statistical_score':1.0, 'statistical_threshold':stat,
            'autoencoder_score':2.0, 'autoencoder_threshold':ae,
            'confirmation':'NORMAL','classification':'NO',
            'upward_highlight':False,'residual_ramp_30m':-2.0,
            'threshold_source':source}

def test_recent_compatible_residual_origin_used_between_hourly_forecasts(tmp_path, monkeypatch):
    file = tmp_path / 'data/processed/stations/amprion/residual_assessments.sqlite'
    save_snapshot(file, '2026-10-01T10:00:00+00:00', 'hourly', [_score_record()])
    out = _run_cached(tmp_path, monkeypatch, origin='2026-10-01T10:45:00Z', lookback=60)
    assert out['available'] is True
    assert out['snapshot_lag_minutes'] == 45
    assert out['is_exact_origin'] is False
    assert out['origin_utc'] == '2026-10-01T10:00:00+00:00'
    # Never confuse a past-origin classification with the newest Demand curve.
    assert out['requested_origin_utc'] == '2026-10-01T10:45:00+00:00'

def test_scored_future_and_stale_origins_are_excluded(tmp_path, monkeypatch):
    file = tmp_path / 'data/processed/stations/amprion/residual_assessments.sqlite'
    save_snapshot(file, '2026-10-01T11:00:00+00:00', 'future', [_score_record()])
    save_snapshot(file, '2026-10-01T09:00:00+00:00', 'old', [_score_record()])
    out = _run_cached(tmp_path, monkeypatch, origin='2026-10-01T10:45:00Z', lookback=60)
    assert out['available'] is False
    assert 'No compatible' in out['message']

def test_same_origin_returns_matching_calibration_not_just_latest_snapshot(tmp_path, monkeypatch):
    file = tmp_path / 'data/processed/stations/amprion/residual_assessments.sqlite'
    origin = '2026-10-01T10:00:00+00:00'
    save_snapshot(file, origin, 'compatible', [_score_record()])
    save_snapshot(file, origin, 'newer-but-incompatible',
                  [_score_record(8.0, 11.0, 'station_override')])
    out = _run_cached(tmp_path, monkeypatch, origin='2026-10-01T10:00:00Z')
    assert out['available'] is True
    assert out['model_fingerprint'] == 'compatible'
    assert out['is_exact_origin'] is True

def test_current_active_threshold_used_for_compatible_selection(tmp_path, monkeypatch):
    file = tmp_path / 'data/processed/stations/amprion/residual_assessments.sqlite'
    origin = '2026-10-01T10:00:00+00:00'
    save_snapshot(file, origin, 'old', [_score_record()])
    save_snapshot(file, origin, 'new', [_score_record(3.0, 10.0, 'station_override')])
    out = _run_cached(tmp_path, monkeypatch,
        settings={'decision_rule':'either','stat_threshold':3.0,'ae_threshold':10.0},
        origin=origin)
    assert out['available'] is True
    assert out['model_fingerprint'] == 'new'

def test_no_origin_returns_most_recent_saved_matching_calibration(tmp_path, monkeypatch):
    file = tmp_path / 'data/processed/stations/amprion/residual_assessments.sqlite'
    save_snapshot(file, '2026-10-11T10:00:00+00:00','old-future',
        [_score_record(3.0, 10.0, 'station_override')])
    save_snapshot(file, '2026-10-08T10:00:00+00:00','new-replay',[_score_record()])
    out = _run_cached(tmp_path, monkeypatch)
    assert out['available'] is True
    assert out['origin_utc'] == '2026-10-08T10:00:00+00:00'
