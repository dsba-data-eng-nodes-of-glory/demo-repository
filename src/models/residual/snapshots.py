"""Durable, replay-safe residual-load score snapshots (no alert state machine).

For existing installations, keep using the legacy SQLite file if present so
previous assessments survive this UI migration. Legacy alert tables, if any,
are intentionally untouched and are never read or updated by this module.
"""
from __future__ import annotations
import json
import sqlite3
from pathlib import Path
from threading import RLock
from contextlib import contextmanager

_LOCK = RLock()

def snapshot_path(root: Path, station: str) -> Path:
    folder = root / "data" / "processed" / "stations" / station
    legacy = folder / "residual_alerts.sqlite"
    return legacy if legacy.is_file() else folder / "residual_assessments.sqlite"

@contextmanager
def _connect(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=15)
    conn.execute("PRAGMA busy_timeout=15000")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""CREATE TABLE IF NOT EXISTS residual_assessments (
        origin_utc TEXT NOT NULL, fingerprint TEXT NOT NULL, records_json TEXT NOT NULL,
        PRIMARY KEY (origin_utc, fingerprint))""")
    columns = conn.execute("PRAGMA table_info(residual_assessments)").fetchall()
    pk = [row[1] for row in sorted(columns, key=lambda r: r[5]) if row[5]]
    if pk != ["origin_utc", "fingerprint"]:
        # Earlier schema was keyed by origin only; preserve every saved assessment.
        conn.execute("ALTER TABLE residual_assessments RENAME TO residual_assessments_old")
        conn.execute("""CREATE TABLE residual_assessments (
            origin_utc TEXT NOT NULL, fingerprint TEXT NOT NULL, records_json TEXT NOT NULL,
            PRIMARY KEY (origin_utc, fingerprint))""")
        conn.execute("INSERT OR IGNORE INTO residual_assessments SELECT origin_utc,fingerprint,records_json FROM residual_assessments_old")
        conn.execute("DROP TABLE residual_assessments_old")
        conn.commit()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

@contextmanager
def _read_only(path: Path):
    """Non-blocking, read-only SQLite connection for dashboard polling.

    Avoid WAL-mode transitions and CREATE TABLE on every GET request: those
    can contend with the predictor's active snapshot writer on Windows.
    """
    conn = sqlite3.connect(path, timeout=0.25)
    conn.execute("PRAGMA query_only=ON")
    conn.execute("PRAGMA busy_timeout=250")
    try:
        yield conn
    finally:
        conn.close()


def read_snapshot(path: Path, origin: str, fingerprint: str):
    with _LOCK:
        with _connect(path) as conn:
            row=conn.execute("SELECT records_json FROM residual_assessments WHERE origin_utc=? AND fingerprint=?",
                             (origin, fingerprint)).fetchone()
            return json.loads(row[0]) if row else None

def save_snapshot(path: Path, origin: str, fingerprint: str, records: list[dict]):
    with _LOCK:
        with _connect(path) as conn:
            conn.execute("INSERT OR IGNORE INTO residual_assessments VALUES (?,?,?)",
                         (origin, fingerprint, json.dumps(records, allow_nan=False, default=str)))

def recent_snapshots(path: Path, limit: int = 36) -> list[dict]:
    if not path.is_file():
        return []
    try:
        with _read_only(path) as conn:
            rows=conn.execute("""SELECT origin_utc,fingerprint,records_json
                FROM residual_assessments ORDER BY origin_utc DESC, rowid DESC LIMIT ?""",(int(limit),)).fetchall()
    except sqlite3.Error:
        return []
    result=[]
    for origin,fingerprint,raw in rows:
        records=json.loads(raw)
        result.append({"origin_utc": origin, "model_fingerprint": fingerprint[:12],
                       "flagged_points":sum(bool(r.get("upward_highlight")) for r in records),
                       "record_count":len(records)})
    return result


def latest_snapshot(path: Path, origin: str | None = None) -> dict | None:
    """Read an existing forecast assessment quickly; never run scoring in GET."""
    if not path.is_file():
        return None
    try:
        with _read_only(path) as conn:
            if origin is None:
                row = conn.execute(
                    "SELECT origin_utc, fingerprint, records_json FROM residual_assessments "
                    "ORDER BY origin_utc DESC, rowid DESC LIMIT 1"
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT origin_utc, fingerprint, records_json FROM residual_assessments "
                    "WHERE origin_utc=? ORDER BY rowid DESC LIMIT 1", (origin,)
                ).fetchone()
    except sqlite3.Error:
        # During a Windows file lock, monitoring stays responsive and retries.
        return None
    if row is None:
        return None
    return {"origin_utc": row[0], "model_fingerprint": row[1][:12],
            "records": json.loads(row[2])}


def candidate_snapshots(path: Path, origin: str | None = None,
                        lookback_minutes: int = 0, limit: int = 64) -> list[dict]:
    """Return bounded, pre-scored candidates for compatible calibration selection.

    For Monitoring, never return a forecast origin AFTER the requested Demand
    origin or older than the stated lookback. For Model Analysis (no origin),
    order by insert time, not replay timestamp: replay can move backwards.
    No model inference or CSV reads occur here.
    """
    if not path.is_file():
        return []
    if origin is not None:
        from datetime import datetime, timedelta, timezone
        t = datetime.fromisoformat(origin.replace('Z', '+00:00'))
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        t = t.astimezone(timezone.utc)
        lo = (t - timedelta(minutes=max(0, int(lookback_minutes)))).isoformat()
        hi = t.isoformat()
        query = ("SELECT origin_utc,fingerprint,records_json FROM residual_assessments "
                 "WHERE origin_utc>=? AND origin_utc<=? "
                 "ORDER BY origin_utc DESC,rowid DESC LIMIT ?")
        params = (lo, hi, max(1, min(int(limit), 128)))
    else:
        query = ("SELECT origin_utc,fingerprint,records_json FROM residual_assessments "
                 "ORDER BY rowid DESC LIMIT ?")
        params = (max(1, min(int(limit), 128)),)
    try:
        with _read_only(path) as conn:
            rows = conn.execute(query, params).fetchall()
    except sqlite3.Error:
        return []
    return [{'origin_utc': origin, 'model_fingerprint': fingerprint[:12],
             'records': json.loads(raw)}
            for origin, fingerprint, raw in rows]
