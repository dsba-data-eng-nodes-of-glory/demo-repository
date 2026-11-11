"""Shared Residual Load overview and two-chart risk visualization."""
from __future__ import annotations
import pandas as pd
import numpy as np
import plotly.graph_objects as go


def _frame(payload: dict) -> pd.DataFrame:
    data = pd.DataFrame(payload.get('records') or [])
    if data.empty: return data
    data['timestamp_utc']=pd.to_datetime(data['timestamp_utc'],utc=True,errors='coerce')
    for name in ('demand','solar_energy','wind_energy','residual_load','expected_residual_load',
                 'residual_ramp_30m','statistical_score','autoencoder_score',
                 'statistical_threshold','autoencoder_threshold'):
        if name in data:data[name]=pd.to_numeric(data[name],errors='coerce')
    return data.dropna(subset=['timestamp_utc']).sort_values('timestamp_utc')


def residual_figures(payload: dict) -> tuple[go.Figure,go.Figure]:
    """Forecast residual vs normal baseline and separate statistical/AE scores."""
    f = _frame(payload)
    level=go.Figure()
    score=go.Figure()
    if f.empty:return level,score
    level.add_trace(go.Scatter(x=f['timestamp_utc'],y=f['residual_load'],name='Forecast residual load',
                               mode='lines',line=dict(color='#38bdf8',width=2.7)))
    level.add_trace(go.Scatter(x=f['timestamp_utc'],y=f['expected_residual_load'],name='Historical expected',
                               mode='lines',line=dict(color='#a78bfa',dash='dash',width=2)))
    add_spike_intervals(level,payload,f,'timestamp_utc','residual_load')
    level.update_layout(title='Residual Load Spike Detection · next 24 hours',yaxis_title='Residual load (source MWh/15 min)',
                        xaxis_title='Forecast target time (UTC)')
    for value,name,color in (('statistical_score','Statistical deviation','#38bdf8'),
                             ('autoencoder_score','Autoencoder score','#fbbf24')):
        score.add_trace(go.Scatter(x=f['timestamp_utc'],y=f[value],mode='lines',name=name,
                                   line=dict(color=color,width=2)))
    if f['statistical_threshold'].notna().any():
        score.add_hline(y=float(f['statistical_threshold'].dropna().iloc[-1]),
                        line=dict(color='#38bdf8',dash='dot'),annotation_text='Stat threshold')
    if f['autoencoder_threshold'].notna().any():
        score.add_hline(y=float(f['autoencoder_threshold'].dropna().iloc[-1]),
                        line=dict(color='#fbbf24',dash='dot'),annotation_text='AE threshold')
    score.update_layout(title='Residual-load anomaly scores vs thresholds',
                        yaxis_title='Calibrated anomaly scores (not probabilities)',
                        xaxis_title='Forecast target time (UTC)')
    for fig in (level,score):
        fig.update_layout(template='plotly_dark',height=350,hovermode='x unified',
                          margin=dict(l=10,r=10,t=52,b=15),
                          paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(8,27,48,.5)',
                          legend=dict(orientation='h',y=1.12,x=0,font=dict(color='#e8f2ff')),
                          font=dict(color='#e8f2ff',size=12))
        fig.update_xaxes(tickfont=dict(color='#d5e4fa'),title_font=dict(color='#d5e4fa'),
                         gridcolor='#284365',linecolor='#56708f')
        fig.update_yaxes(tickfont=dict(color='#d5e4fa'),title_font=dict(color='#d5e4fa'),
                         gridcolor='#284365',linecolor='#56708f')
    return level,score


def residual_summary(payload:dict)->dict:
    f=_frame(payload)
    if f.empty:return {'next':None,'peak':None,'ramp':None,'flagged':0,'scored':0}
    return {'next':float(f['residual_load'].iloc[0]),
            'peak':float(f['residual_load'].max()),
            'ramp':float(f['residual_ramp_30m'].max()),
            'flagged':int(f['upward_highlight'].eq(True).sum()),
            'scored':int(f['context_valid'].eq(True).sum())}


def flagged_intervals(payload: dict, interval_minutes: int = 15) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Merge adjacent flagged forecast timestamps into visible contiguous bands.

    Only rows explicitly marked ``upward_highlight`` qualify. A rolling
    assessment of a new origin is not treated as a verified historical event.
    """
    f=_frame(payload)
    if f.empty or 'upward_highlight' not in f.columns:return []
    flagged=f.loc[f['upward_highlight'].eq(True)].sort_values('timestamp_utc')
    if flagged.empty:return []
    step=pd.Timedelta(minutes=interval_minutes)
    windows=[]
    start=last=flagged['timestamp_utc'].iloc[0]
    for timestamp in flagged['timestamp_utc'].iloc[1:]:
        if timestamp-last>step:
            windows.append((start-step/2,last+step/2))
            start=timestamp
        last=timestamp
    windows.append((start-step/2,last+step/2))
    return windows


def add_spike_intervals(fig:go.Figure,payload:dict,forecast_frame:pd.DataFrame,
                        timestamp_col:str,value_col:str,interval_minutes:int=15)->int:
    """Shade residual spike intervals on forecast graphs and mark their targets.

    Chart inputs are forecast-only; never assigns anomaly labels to actuals.
    Returns the number of highlighted bands. Safe with empty/partial data.
    """
    bands=flagged_intervals(payload,interval_minutes)
    if not bands:return 0
    frame=forecast_frame.copy()
    if timestamp_col not in frame or value_col not in frame:return 0
    frame[timestamp_col]=pd.to_datetime(frame[timestamp_col],utc=True,errors='coerce')
    frame[value_col]=pd.to_numeric(frame[value_col],errors='coerce')
    frame=frame.dropna(subset=[timestamp_col,value_col])
    if frame.empty:return 0
    # The warning is temporal: shade the actual matched forecast target time.
    times=frame[timestamp_col]
    for left,right in bands:
        if not ((times>=left)&(times<=right)).any(): continue
        fig.add_vrect(x0=left,x1=right,fillcolor='#f97316',opacity=0.20,
                      line_width=0,layer='below')
    flagged=_frame(payload)
    flagged=flagged.loc[flagged['upward_highlight'].eq(True),['timestamp_utc']]
    matched=frame.merge(flagged,left_on=timestamp_col,right_on='timestamp_utc',how='inner')
    if not matched.empty:
        fig.add_trace(go.Scatter(x=matched[timestamp_col],y=matched[value_col],
                                 mode='markers',name='Residual-load spike interval',
                                 marker=dict(color='#fb923c',size=9,symbol='diamond',
                                             line=dict(color='#fff3df',width=1)),
                                 hovertemplate='Residual-load spike risk<br>%{x|%d %b %H:%M UTC}<extra></extra>'))
    return len(bands)
