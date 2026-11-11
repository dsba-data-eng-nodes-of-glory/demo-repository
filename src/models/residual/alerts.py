"""Per-station stable 24-hour residual-load alert ledger.

Based on the supplied benchmark's 2-of-3 confirmation and bounded-miss
state machine. Event identities are durable across dashboard refreshes.
Only process a new forecast origin once (including after process restart).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import json
import sqlite3
from threading import RLock
import numpy as np
import pandas as pd

_DB_LOCK = RLock()

@dataclass
class Alert:
    event_id: str
    initial_start: str
    initial_end: str
    current_start: str
    current_end: str
    detected_at: str
    last_seen: str
    state: str = 'WATCH'
    history: list[bool] = field(default_factory=list)
    misses: int = 0
    seen_count: int = 1
    severity_index: float = 0.0
    peak_residual: float = 0.0
    matched_confirmation: str = 'STAT_ONLY'
    update_log: list[dict] = field(default_factory=list)

class PersistentAlerts:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn=sqlite3.connect(path, timeout=15)
        self.conn.execute('PRAGMA journal_mode=WAL')
        self.conn.execute('CREATE TABLE IF NOT EXISTS alert_events(event_id TEXT PRIMARY KEY, payload TEXT NOT NULL)')
        self.conn.execute('CREATE TABLE IF NOT EXISTS replay_progress(name TEXT PRIMARY KEY, value TEXT NOT NULL)')
        self.conn.execute('CREATE TABLE IF NOT EXISTS residual_assessments(origin_utc TEXT NOT NULL, fingerprint TEXT NOT NULL, records_json TEXT NOT NULL, PRIMARY KEY (origin_utc, fingerprint))')
        # Upgrade the initial single-origin schema without deleting prior snapshots.
        columns=self.conn.execute('PRAGMA table_info(residual_assessments)').fetchall()
        pk_cols=[row[1] for row in sorted(columns,key=lambda x:x[5]) if row[5]]
        if pk_cols != ['origin_utc','fingerprint']:
            self.conn.execute('ALTER TABLE residual_assessments RENAME TO residual_assessments_v1')
            self.conn.execute('CREATE TABLE residual_assessments(origin_utc TEXT NOT NULL, fingerprint TEXT NOT NULL, records_json TEXT NOT NULL, PRIMARY KEY (origin_utc, fingerprint))')
            self.conn.execute('INSERT OR IGNORE INTO residual_assessments SELECT origin_utc, fingerprint, records_json FROM residual_assessments_v1')
            self.conn.execute('DROP TABLE residual_assessments_v1')
        self.conn.commit()
        self.events: dict[str,Alert]={}
        for event_id,payload in self.conn.execute('SELECT event_id,payload FROM alert_events'):
            saved=json.loads(payload)
            self.events[event_id]=Alert(**{k:v for k,v in saved.items() if k in Alert.__dataclass_fields__})
        nums=[int(s.rsplit('-',1)[-1]) for s in self.events if s.startswith('RL-') and s.rsplit('-',1)[-1].isdigit()]
        self.next_id=max(nums,default=0)+1

    def close(self):
        self.conn.close()

    def saved_assessment(self, origin: str, fingerprint: str):
        row=self.conn.execute('SELECT fingerprint,records_json FROM residual_assessments WHERE origin_utc=? AND fingerprint=?',(origin,fingerprint)).fetchone()
        if row and row[0]==fingerprint:
            return json.loads(row[1])
        return None

    def save_assessment(self, origin: str, fingerprint: str, records: list[dict]):
        with self.conn:
            self.conn.execute('INSERT OR IGNORE INTO residual_assessments VALUES (?,?,?)',
                              (origin,fingerprint,json.dumps(records,allow_nan=False,default=str)))

    def recent_assessments(self, limit: int=36):
        rows=self.conn.execute('SELECT origin_utc, fingerprint,records_json FROM residual_assessments ORDER BY origin_utc DESC, rowid DESC LIMIT ?', (int(limit),)).fetchall()
        result=[]
        for a,b,c in rows:
            # Dashboard needs historic risk COUNTS and the origin, not 96 fully
            # scored rows for every preceding forecast; keep API payload bounded.
            records=json.loads(c)
            result.append({'origin_utc':a,'model_fingerprint':b[:12],
                           'flagged_points':sum(bool(p.get('upward_highlight')) for p in records),
                           'record_count':len(records)})
        return result

    def last_origin(self):
        row=self.conn.execute("SELECT value FROM replay_progress WHERE name='last_forecast_origin'").fetchone()
        return pd.Timestamp(row[0]) if row else None

    def update(self, candidates: list[dict], origin: pd.Timestamp, *, confirm_hits: int=2,
               resolving_misses: int=3, resolve_misses: int=5, match_tolerance_minutes: int=75,
               critical_ratio: float=1.6):
        # Coordinate FastAPI and predictor processes through the database write lock.
        # Do not trust events cached before another process acquired the lock.
        self.conn.execute('BEGIN IMMEDIATE')
        self.events={}
        for event_id,payload in self.conn.execute('SELECT event_id,payload FROM alert_events'):
            row=json.loads(payload)
            self.events[event_id]=Alert(**{k:v for k,v in row.items() if k in Alert.__dataclass_fields__})
        nums=[int(s.rsplit('-',1)[-1]) for s in self.events if s.startswith('RL-') and s.rsplit('-',1)[-1].isdigit()]
        self.next_id=max(nums,default=0)+1
        last=self.last_origin()
        if last is not None and origin <= last:
            self.conn.rollback()
            return False
        now=origin.isoformat()
        active=[a for a in self.events.values() if a.state!='RESOLVED']
        used=set()
        for c in candidates:
            c0,c1=pd.Timestamp(c['start']),pd.Timestamp(c['end'])
            best=None;best_dist=float('inf')
            for a in active:
                if a.event_id in used:continue
                a0,a1=pd.Timestamp(a.current_start),pd.Timestamp(a.current_end)
                gap=max(pd.Timedelta(0),a0-c1,c0-a1)
                if gap <= pd.Timedelta(minutes=match_tolerance_minutes):
                    distance=abs((c0+(c1-c0)/2)-(a0+(a1-a0)/2)).total_seconds()
                    if distance<best_dist:best,best_dist=a,distance
            if best is None:
                identifier=f'RL-{self.next_id:05d}';self.next_id+=1
                best=Alert(identifier,c0.isoformat(),c1.isoformat(),c0.isoformat(),c1.isoformat(),
                           now,now,history=[True],severity_index=float(c['risk']),
                           peak_residual=float(c['peak']),matched_confirmation=c['confirmation'])
                self.events[identifier]=best
                best.update_log.append({'issued_at':now,'status':'CREATED','risk':c['risk']})
            else:
                best.current_start=c0.isoformat();best.current_end=c1.isoformat()
                best.last_seen=now;best.misses=0;best.seen_count+=1
                best.severity_index=float(c['risk']);best.peak_residual=float(c['peak'])
                best.matched_confirmation=c['confirmation']
                best.history=(best.history+[True])[-3:]
                if best.state in ('RESOLVING','WATCH') and sum(best.history)>=confirm_hits:
                    best.state='WARNING'
            if (float(c['ratio'])>=critical_ratio and c0-origin<=pd.Timedelta(hours=4)
                    and sum(best.history)>=confirm_hits):
                best.state='CRITICAL'
            best.update_log.append({'issued_at':now,'status':best.state,'risk':c['risk']})
            used.add(best.event_id)
        for a in active:
            if a.event_id in used:continue
            a.misses+=1
            a.history=(a.history+[False])[-3:]
            if a.misses>=resolve_misses:
                a.state='RESOLVED'
                a.update_log.append({'issued_at':now,'status':'RESOLVED'})
            elif a.misses>=resolving_misses:
                a.state='RESOLVING'
        with self.conn:
            for identifier,alert in self.events.items():
                alert.update_log=alert.update_log[-100:]
                self.conn.execute('INSERT OR REPLACE INTO alert_events VALUES(?,?)',
                                  (identifier,json.dumps(alert.__dict__,default=str)))
            self.conn.execute("INSERT OR REPLACE INTO replay_progress VALUES('last_forecast_origin',?)",(now,))
        return True

    def records(self, limit: int=50) -> list[dict]:
        return sorted((alert.__dict__ for alert in self.events.values()),key=lambda r:r['detected_at'],reverse=True)[:limit]


def candidates_from_scores(scores: pd.DataFrame, *, min_sustain_intervals: int=2,
                           gap_tolerance: int=1, upward_only: bool=True) -> list[dict]:
    """Group contiguous qualified 15-minute intervals into persistent alert windows."""
    if scores.empty:return []
    qualified=scores['upward_highlight'].astype(bool).to_numpy() if upward_only else scores['classification'].eq('YES').to_numpy()
    ids=np.flatnonzero(qualified)
    if not len(ids):return []
    blocks=[]; current=[int(ids[0])]
    for index in ids[1:]:
        if int(index)-current[-1]<=1+gap_tolerance:current.append(int(index))
        else:blocks.append(current);current=[int(index)]
    blocks.append(current)
    events=[]
    for block in blocks:
        if len(block)<min_sustain_intervals:continue
        subset=scores.iloc[block]
        stat=pd.to_numeric(subset['statistical_score'],errors='coerce').to_numpy(float)
        ae=pd.to_numeric(subset['autoencoder_score'],errors='coerce').to_numpy(float)
        sthreshold=float(subset['statistical_threshold'].iloc[0]);athreshold=float(subset['autoencoder_threshold'].iloc[0])
        ratio=float(max(np.nanmax(stat)/max(sthreshold,1e-9),np.nanmax(ae)/max(athreshold,1e-9)))
        # Benchmark-style severity INDEX. Do not present it as probability.
        risk=float(np.clip(50+25*(ratio-1),0,100))
        events.append({'start':pd.Timestamp(subset['timestamp_utc'].iloc[0]),
                       'end':pd.Timestamp(subset['timestamp_utc'].iloc[-1])+pd.Timedelta(minutes=15),
                       'peak':float(subset['residual_load'].max()),'risk':risk,'ratio':ratio,
                       'confirmation':'CONFIRMED' if subset['confirmation'].eq('CONFIRMED').any() else 'SINGLE_STAGE',
                       'count':len(block)})
    return events


def update_alerts(path: Path, origin_utc: str, scored: pd.DataFrame, settings: dict) -> list[dict]:
    with _DB_LOCK:
        db=PersistentAlerts(path)
        try:
            candidates=candidates_from_scores(scored,min_sustain_intervals=int(settings.get('min_sustain_intervals',2)),
                                               upward_only=bool(settings.get('upward_only',True)))
            db.update(candidates,pd.Timestamp(origin_utc),confirm_hits=int(settings.get('confirm_hits',2)),
                      resolving_misses=int(settings.get('resolving_misses',3)),
                      resolve_misses=int(settings.get('resolve_misses',5)),
                      critical_ratio=float(settings.get('critical_ratio',1.6)))
            return db.records()
        finally:
            db.close()


def get_alerts(path: Path, limit=50):
    with _DB_LOCK:
        db=PersistentAlerts(path)
        try:return db.records(limit)
        finally:db.close()


def save_snapshot(path: Path, origin: str, fingerprint: str, records: list[dict]):
    with _DB_LOCK:
        db=PersistentAlerts(path)
        try: db.save_assessment(origin,fingerprint,records)
        finally: db.close()

def read_snapshot(path: Path, origin: str, fingerprint: str):
    with _DB_LOCK:
        db=PersistentAlerts(path)
        try:return db.saved_assessment(origin,fingerprint)
        finally:db.close()

def recent_snapshots(path: Path, limit=36):
    with _DB_LOCK:
        db=PersistentAlerts(path)
        try:return db.recent_assessments(limit)
        finally:db.close()
