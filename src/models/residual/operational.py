"""Align three *existing* forecast models and score forecast residual-load risk.

No forward target actuals are used. All three forecast origins must match exactly.
Input demand, solar and wind are expected in the same MWh/15-minute energy
units for the supplied Amprion checkpoint. New station data must use the same
contract or explicitly be normalized upstream.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.residual.portable import load_bundle
from src.models.residual.residual_ae_inference import score_residual_timestamps

STEP = pd.Timedelta(minutes=15)
HORIZON = 96
SOLAR = 'Photovoltaics [MWh]'
WIND = 'Wind onshore [MWh]'  # Source CSV column name; user-facing label is "Wind Energy".
SOLAR_CAP = 'Photovoltaics_installed_capacity [MW]'
WIND_CAP = 'Wind_onshore_installed_capacity [MW]'


def utc(values: Any) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(values, utc=True, errors='coerce', format='mixed'))


def _run(records: list[dict] | pd.DataFrame, column: str, origin: pd.Timestamp, kind: str | None = None) -> pd.Series:
    frame = pd.DataFrame(records).copy()
    if kind is not None:
        if 'generation_type' not in frame:
            raise ValueError('Renewable forecast records do not specify generation_type')
        frame = frame.loc[frame['generation_type'].eq(kind)]
    if frame.empty:
        raise ValueError(f'{kind or "demand"}: No forecast data')
    if not {'forecast_origin_utc', 'forecast_timestamp_utc', column}.issubset(frame):
        raise ValueError(f'{kind or "demand"}: Forecast schema is missing required fields')
    frame['origin'] = utc(frame['forecast_origin_utc'])
    frame = frame.loc[frame['origin'] == origin].copy()
    frame['target'] = utc(frame['forecast_timestamp_utc'])
    if frame.empty:
        raise ValueError(f'{kind or "demand"}: Forecast origin not found {origin}')
    frame['value'] = pd.to_numeric(frame[column], errors='coerce')
    frame = frame.drop_duplicates('target', keep='last').sort_values('target')
    targets = pd.date_range(start=origin + STEP, periods=HORIZON, freq=STEP, tz='UTC')
    if not frame['target'].reset_index(drop=True).equals(pd.Series(targets, name='target')):
        raise ValueError(f'{kind or "demand"}: Expected 96 aligned future targets at {origin}')
    values = frame['value'].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f'{kind or "demand"}: Missing/negative forecast values')
    return pd.Series(values, index=targets)


def aligned_residual_forecast(demand: list[dict], renewable: list[dict],
                              origin_utc: str | pd.Timestamp, solar_capacity_mw: float,
                              wind_capacity_mw: float, demand_unit: str='mwh_per_interval') -> pd.DataFrame:
    """Combine exactly matching 96-step forecasts and point-in-time known capacities.

    Does not read future observed generation or future measured capacities.
    """
    origin = pd.Timestamp(origin_utc)
    origin = origin.tz_localize('UTC') if origin.tzinfo is None else origin.tz_convert('UTC')
    if not np.isfinite([solar_capacity_mw, wind_capacity_mw]).all() or min(solar_capacity_mw, wind_capacity_mw) <= 0:
        raise ValueError('Missing positive installed capacity as of forecast origin')
    d = _run(demand, 'predicted_demand', origin)
    if demand_unit == 'mw':
        d = d / 4.0
    elif demand_unit != 'mwh_per_interval':
        raise ValueError('Unsupported residual demand unit (require mw or mwh_per_interval)')
    s = _run(renewable, 'predicted_generation_mwh', origin, 'solar')
    w = _run(renewable, 'predicted_generation_mwh', origin, 'wind')
    # Do not clip forecasts silently; invalid capacity utilization should be corrected at source.
    if (4 * s > solar_capacity_mw * 1.05).any() or (4 * w > wind_capacity_mw * 1.05).any():
        raise ValueError('Renewable forecast exceeds installed generation capacity; verify model units and inputs')
    return pd.DataFrame({
        'demand': d.to_numpy(), 'solar': s.to_numpy(), 'wind': w.to_numpy(),
        'residual': d.to_numpy() - s.to_numpy() - w.to_numpy(),
        'solar_capacity_mw': float(solar_capacity_mw), 'wind_capacity_mw': float(wind_capacity_mw),
    }, index=d.index.rename('timestamp_utc'))


def available_origins(demand: list[dict], renewable: list[dict]) -> list[pd.Timestamp]:
    """Newest origins present in all three models; complete run checked separately."""
    df = pd.DataFrame(demand)
    rf = pd.DataFrame(renewable)
    if df.empty or rf.empty or 'forecast_origin_utc' not in df or not {'forecast_origin_utc','generation_type'}.issubset(rf):
        return []
    dset = set(utc(df['forecast_origin_utc']).dropna())
    rset = {kind: set(utc(rf.loc[rf['generation_type'].eq(kind), 'forecast_origin_utc']).dropna()) for kind in ('solar','wind')}
    return sorted(dset & rset['solar'] & rset['wind'], reverse=True)


@lru_cache(maxsize=12)
def _source_frame(path: str, mtime: int, size: int, kind: str) -> pd.DataFrame:
    value_col = SOLAR if kind == 'solar' else WIND
    cap_col = SOLAR_CAP if kind == 'solar' else WIND_CAP
    frame = pd.read_csv(path, usecols=['utc_timestamp',value_col,cap_col], low_memory=False)
    frame['utc_timestamp'] = pd.to_datetime(frame['utc_timestamp'], format='mixed', dayfirst=True, utc=True, errors='coerce')
    for name in (value_col,cap_col):
        frame[name] = pd.to_numeric(frame[name].astype("string").str.replace(",", "", regex=False), errors='coerce')
    frame = frame.dropna(subset=['utc_timestamp']).sort_values('utc_timestamp').drop_duplicates('utc_timestamp', keep='last')
    return frame.rename(columns={value_col: kind, cap_col: f'{kind}_capacity_mw'}).set_index('utc_timestamp')


def _station_renewable_history(root: Path, cfg: dict, kind: str, origin: pd.Timestamp, rows: int = 450) -> pd.DataFrame:
    spec = (cfg.get('renewables') or {}).get(kind) or {}
    files=[]
    for field in ('train_file','test_file','data_file'):
        filename = spec.get(field)
        if not filename: continue
        file = Path(filename)
        file = file if file.is_absolute() else root / file
        if file.is_file() and file not in files:
            stat=file.stat()
            files.append((str(file),stat.st_mtime_ns,stat.st_size,kind))
    if not files:
        raise FileNotFoundError(f'Missing registered {kind} training/history CSV for residual-load context')
    frames=[_source_frame(*details) for details in files]
    combined=pd.concat(frames).sort_index()
    combined=combined.loc[~combined.index.duplicated(keep='last')]
    # IMPORTANT: only observed data at or before origin; never the future from historical test files.
    return combined.loc[combined.index <= origin].tail(rows)


def build_residual_history(root: Path, cfg: dict, origin_utc: str | pd.Timestamp,
                           demand_actuals: list[dict], rows: int = 450) -> tuple[pd.DataFrame,float,float]:
    origin = pd.Timestamp(origin_utc)
    origin = origin.tz_localize('UTC') if origin.tzinfo is None else origin.tz_convert('UTC')
    df = pd.DataFrame(demand_actuals)
    if df.empty or 'utc_timestamp' not in df:
        raise ValueError('Demand actual history is missing')
    name = 'actual_demand' if 'actual_demand' in df else 'demand'
    if name not in df:
        raise ValueError('Demand actual history must contain actual_demand or demand')
    df['timestamp'] = utc(df['utc_timestamp'])
    df['demand'] = pd.to_numeric(df[name], errors='coerce')
    df = df.dropna(subset=['timestamp']).loc[lambda data: data['timestamp'] <= origin]
    df = df.drop_duplicates('timestamp', keep='last').set_index('timestamp')[['demand']].sort_index().tail(rows)
    if df.empty:
        raise ValueError('No actual demand data before forecast origin')
    s = _station_renewable_history(root,cfg,'solar',origin,rows)
    w = _station_renewable_history(root,cfg,'wind',origin,rows)
    sc = s['solar_capacity_mw'].dropna()
    wc = w['wind_capacity_mw'].dropna()
    if sc.empty or wc.empty or sc.iloc[-1] <= 0 or wc.iloc[-1] <= 0:
        raise ValueError('Station solar/wind capacity as of forecast origin is unavailable')
    hist = df.join(s, how='inner').join(w, how='inner').sort_index().tail(rows)
    demand_unit=cfg.get('residual_demand_unit','mwh_per_interval')
    if demand_unit == 'mw':
        hist['demand'] = hist['demand'] / 4.0
    elif demand_unit != 'mwh_per_interval':
        raise ValueError('Unsupported station residual_demand_unit')
    # Strictly no time interpolation or future observed renewable use.
    hist['residual'] = hist['demand'] - hist['solar'] - hist['wind']
    hist.index.name = 'timestamp_utc'
    if len(hist) < 194:
        raise ValueError(f'At least 194 aligned previous observed quarter hours needed; got {len(hist)}')
    if len(hist.index) > 1 and not np.all(np.diff(hist.index.asi8) == STEP.value):
        raise ValueError('Recent residual-load observations contain gaps; scoring suspended')
    return hist,float(sc.iloc[-1]),float(wc.iloc[-1])


@lru_cache(maxsize=8)
def _bundle_cached(path: str, mtime: int, size: int) -> dict:
    return load_bundle(path)


def score_operational(demand_forecast: list[dict], renewables: list[dict],
                      historical: pd.DataFrame, checkpoint: str | Path,
                      origin: str | pd.Timestamp, solar_capacity_mw: float, wind_capacity_mw: float,
                      stat_threshold: float | None = None, ae_threshold: float | None = None,
                      decision_rule: str = 'both', demand_unit: str='mwh_per_interval') -> pd.DataFrame:
    """Run trained statistical/AE detector using model forecasts, NOT future actuals."""
    checkpoint = Path(checkpoint)
    sig=checkpoint.stat()
    bundle = _bundle_cached(str(checkpoint.resolve()), sig.st_mtime_ns, sig.st_size)
    target = aligned_residual_forecast(demand_forecast, renewables, origin, solar_capacity_mw, wind_capacity_mw, demand_unit)
    if not historical.empty and historical.index.max() >= target.index.min():
        raise ValueError('Forecast history contains target-time actuals; leakage prevented')
    scores = score_residual_timestamps(target,historical,bundle,batch_size=128,decision_rule=decision_rule)
    stat=float(stat_threshold) if stat_threshold is not None else float(bundle['stat_threshold'])
    ae=float(ae_threshold) if ae_threshold is not None else float(bundle['ae_threshold'])
    if not np.isfinite([stat,ae]).all() or min(stat,ae) <= 0:
        raise ValueError('Anomaly threshold overrides must be positive and finite')
    scores['statistical_threshold']=stat
    scores['autoencoder_threshold']=ae
    valid=scores['context_valid'].astype(bool)
    sf=valid & scores['statistical_score'].gt(stat)
    af=valid & scores['autoencoder_score'].gt(ae)
    flagged=(sf & af) if decision_rule=='both' else (sf | af)
    scores['statistical_flag']=sf
    scores['autoencoder_flag']=af
    scores['classification'] = np.where(~valid,'INSUFFICIENT_DATA',np.where(flagged,'YES','NO'))
    scores['confirmation']=np.where(~valid,'INSUFFICIENT_DATA',np.where(sf & af,'CONFIRMED',np.where(sf,'STAT_ONLY',np.where(af,'AE_ONLY','NORMAL'))))
    scores['upward_highlight']=flagged & scores['residual_ramp_30m'].gt(0)
    scores['threshold_source']='station_override' if stat_threshold is not None or ae_threshold is not None else 'checkpoint_calibration'
    return scores


def compact_score_records(scored: pd.DataFrame) -> list[dict]:
    columns=['timestamp_utc','demand','solar_energy','wind_energy','residual_load','expected_residual_load',
             'residual_deviation','residual_ramp_30m','residual_ramp_1h','statistical_score','statistical_threshold',
             'autoencoder_score','autoencoder_threshold','confirmation','classification','context_valid',
             'upward_highlight','anomaly_direction','threshold_source']
    output=scored.reindex(columns=columns).copy()
    output['timestamp_utc']=pd.to_datetime(output['timestamp_utc'],utc=True).dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    output=output.astype(object).where(pd.notna(output),None)
    return output.to_dict('records')
