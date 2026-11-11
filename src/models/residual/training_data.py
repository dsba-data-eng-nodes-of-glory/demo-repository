"""Build a reproducible residual-load training dataset from aligned historical sources."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

COLUMNS = ['utc_timestamp','demand','Photovoltaics [MWh]','Wind onshore [MWh]',
           'Photovoltaics_installed_capacity [MW]','Wind_onshore_installed_capacity [MW]']

def build_training_frame(demand: pd.DataFrame, solar: pd.DataFrame, wind: pd.DataFrame, demand_unit: str='mwh_per_interval') -> pd.DataFrame:
    inputs = []
    for frame, fields in ((demand,['demand']),(solar,['Photovoltaics [MWh]','Photovoltaics_installed_capacity [MW]']),
                          (wind,['Wind onshore [MWh]','Wind_onshore_installed_capacity [MW]'])):
        missing={'utc_timestamp',*fields}-set(frame)
        if missing:raise ValueError(f'Residual-load source missing {sorted(missing)}')
        f=frame[['utc_timestamp',*fields]].copy()
        f['utc_timestamp']=pd.to_datetime(f['utc_timestamp'],utc=True,errors='coerce',format='mixed')
        if f['utc_timestamp'].isna().any() or f['utc_timestamp'].duplicated().any():
            raise ValueError('Residual inputs must have valid unique timestamps')
        for col in fields:f[col]=pd.to_numeric(f[col].astype("string").str.replace(",", "", regex=False),errors='coerce')
        if f[fields].isna().any().any() or not np.isfinite(f[fields].to_numpy(float)).all():
            raise ValueError('Residual-load inputs contain non-numeric or missing values')
        inputs.append(f)
    frame=inputs[0].merge(inputs[1],on='utc_timestamp',how='inner').merge(inputs[2],on='utc_timestamp',how='inner')
    frame=frame.sort_values('utc_timestamp').reset_index(drop=True)
    if len(frame)<96*30:raise ValueError('Residual training needs at least 30 days of aligned observations')
    if not np.all(np.diff(frame['utc_timestamp'].astype('int64').to_numpy())==15*60*10**9):
        raise ValueError('Residual historical data has gaps. Demand/Solar/Wind must share a continuous 15-minute timeline.')
    if (frame[COLUMNS[1:]]<0).any().any():raise ValueError('Residual inputs contain negative generation/demand/capacity')
    if (frame[['Photovoltaics_installed_capacity [MW]','Wind_onshore_installed_capacity [MW]']]<=0).any().any():
        raise ValueError('Installed capacity must be positive')
    for value,capacity in (('Photovoltaics [MWh]','Photovoltaics_installed_capacity [MW]'),('Wind onshore [MWh]','Wind_onshore_installed_capacity [MW]')):
        if (4*frame[value]>1.05*frame[capacity]).any():
            raise ValueError('Renewable generation/capacity inconsistent with MWh/15-minute units')
    if demand_unit not in ('mwh_per_interval','mw'):
        raise ValueError('demand_unit must be mwh_per_interval or mw')
    if demand_unit == 'mw':
        frame['demand'] = frame['demand'] / 4.0  # 15-minute MW mean -> MWh delivered
    frame['Residual Load']=frame['demand']-frame['Photovoltaics [MWh]']-frame['Wind onshore [MWh]']
    return frame

def read_registered_sources(root: Path, entry: dict) -> pd.DataFrame:
    def source(path):
        if not path: raise FileNotFoundError('Required Demand/Solar/Wind training source not registered')
        p=Path(path);p=p if p.is_absolute() else root/p
        if not p.is_file():raise FileNotFoundError(f'Required source not found: {p}')
        return pd.read_csv(p,low_memory=False)
    renewables=entry.get('renewables') or {}
    d=source(entry.get('training_file') or entry.get('history_file'))
    s=source((renewables.get('solar') or {}).get('train_file') or (renewables.get('solar') or {}).get('data_file'))
    w=source((renewables.get('wind') or {}).get('train_file') or (renewables.get('wind') or {}).get('data_file'))
    return build_training_frame(d,s,w,entry.get('residual_demand_unit','mwh_per_interval'))
