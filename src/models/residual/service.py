"""Per-origin residual-load scoring with immutable forecast assessment history."""
from __future__ import annotations
from pathlib import Path
import hashlib
import json
import pandas as pd

from src.models.residual.operational import (
    available_origins, build_residual_history, score_operational,
    compact_score_records, _run,
)
from src.models.residual.snapshots import (
    read_snapshot, save_snapshot, recent_snapshots, snapshot_path,
)

DEFAULT_SETTINGS = {
    "decision_rule": "both",
    "stat_threshold": None,
    "ae_threshold": None,
    "upward_only": True,
}

def settings_for(cfg: dict) -> dict:
    # Preserve existing thresholds for older stations, but never retain a
    # obsolete state-machine fields from the legacy configuration.
    legacy = cfg.get("residual_alert_settings") or {}
    current = cfg.get("residual_detection_settings") or {}
    return {key: current.get(key, legacy.get(key, default))
            for key, default in DEFAULT_SETTINGS.items()}


def score_and_store(*,root:Path,station:str,cfg:dict,demand_forecasts:list[dict],renewable_forecasts:list[dict],
                    demand_history:list[dict],origin_utc:str|None=None,history_runs:int=24) -> dict:
    checkpoint_value=cfg.get('residual_model_checkpoint') or (cfg.get('model_components') or {}).get('residual_anomaly',{}).get('checkpoint')
    if not checkpoint_value:
        return {'available':False,'message':'Residual-load model is not trained/deployed','records':[]}
    checkpoint=Path(checkpoint_value)
    checkpoint=checkpoint if checkpoint.is_absolute() else root/checkpoint
    if not checkpoint.is_file():
        return {'available':False,'message':f'Residual-load model checkpoint is missing: {checkpoint}','records':[]}
    settings=settings_for(cfg)
    demand_unit=cfg.get('residual_demand_unit','mwh_per_interval')
    origins=[pd.Timestamp(origin_utc)] if origin_utc else available_origins(demand_forecasts,renewable_forecasts)
    if not origins:
        return {'available':False,'message':'Waiting for complete, same-origin Demand, Solar and Wind forecasts','records':[],
                'history':recent_snapshots(snapshot_path(root,station),history_runs)}
    store=snapshot_path(root,station)
    failures=[]
    for origin in origins[:25]:
        origin=origin.tz_localize('UTC') if origin.tzinfo is None else origin.tz_convert('UTC')
        try:
            historical,solar_cap,wind_cap=build_residual_history(root,cfg,origin,demand_history)
            # Fingerprint includes all target forecast values so retried model versions at
            # the same origin cannot silently reuse a stale score.
            demand=_run(demand_forecasts,'predicted_demand',origin).to_numpy(float)
            solar=_run(renewable_forecasts,'predicted_generation_mwh',origin,'solar').to_numpy(float)
            wind=_run(renewable_forecasts,'predicted_generation_mwh',origin,'wind').to_numpy(float)
            signature=checkpoint.stat()
            ident={'checkpoint':str(checkpoint.resolve()),'mtime':signature.st_mtime_ns,'size':signature.st_size,
                   'settings':settings,'demand':demand.tolist(),'solar':solar.tolist(),'wind':wind.tolist(),
                   'capacity':[solar_cap,wind_cap],'demand_unit':demand_unit}
            fingerprint=hashlib.sha256(json.dumps(ident,sort_keys=True,default=str).encode('utf8')).hexdigest()
            origin_string=origin.isoformat()
            records=read_snapshot(store,origin_string,fingerprint)
            if records is None:
                scored=score_operational(demand_forecasts,renewable_forecasts,historical,checkpoint,origin,
                        solar_cap,wind_cap,stat_threshold=settings.get('stat_threshold'),
                        ae_threshold=settings.get('ae_threshold'),decision_rule=settings['decision_rule'],
                        demand_unit=demand_unit)
                if scored['context_valid'].sum() < 90:
                    raise ValueError('Incomplete historic context for 24h residual scoring')
                if not settings.get('upward_only', True):
                    scored['upward_highlight'] = scored['classification'].eq('YES')
                records=compact_score_records(scored)
                # idempotency: keep origin assessment across API/dashboard refreshes.
                save_snapshot(store,origin_string,fingerprint,records)
            else:
                scored=pd.DataFrame(records)
                scored['timestamp_utc']=pd.to_datetime(scored['timestamp_utc'],utc=True)
            snapshots=recent_snapshots(store,history_runs)
            return {'available':True,'station':station,'origin_utc':origin_string,
                    'model_fingerprint':fingerprint[:12], 'records':records,'history':snapshots,
                    'settings':settings,'message':'Forecast-only residual-load assessment; events are predictions, not confirmed grid incidents.'}
        except (ValueError,KeyError,FileNotFoundError,IndexError,TypeError) as exc:
            failures.append(f'{origin}: {exc}')
            continue
    return {'available':False,'station':station,'records':[],
            'history':recent_snapshots(store,history_runs),
            'message':f'Residual scoring not ready: {failures[:2]}'}
