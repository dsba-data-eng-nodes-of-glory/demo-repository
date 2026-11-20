"""Independent dashboard readings and the durable Kafka catch-up cursor."""
from __future__ import annotations

from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3

DB_NAME = "live_actuals.sqlite3"


def live_actuals_path(root: Path, station: str) -> Path:
    return root / "data" / "state" / station / DB_NAME


def initialize(path: Path, topic: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path, timeout=5)) as db, db:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("""CREATE TABLE IF NOT EXISTS readings (
            timestamp TEXT PRIMARY KEY, demand REAL NOT NULL, received_at TEXT NOT NULL)""")
        db.execute("""CREATE TABLE IF NOT EXISTS cursor (
            singleton INTEGER PRIMARY KEY CHECK(singleton = 1), topic TEXT NOT NULL,
            next_offset INTEGER, recovery_start INTEGER, recovery_end INTEGER)""")
        db.execute("INSERT OR IGNORE INTO cursor (singleton, topic) VALUES (1, ?)", (topic,))
        if db.execute("SELECT topic FROM cursor").fetchone()[0] != topic:
            raise ValueError("Live reading database belongs to another Kafka topic.")


def next_offset(path: Path) -> int | None:
    with closing(sqlite3.connect(path, timeout=5)) as db:
        return db.execute("SELECT next_offset FROM cursor").fetchone()[0]


def begin_recovery(path: Path, start: int, end: int) -> None:
    with closing(sqlite3.connect(path, timeout=5)) as db, db:
        db.execute("""UPDATE cursor
            SET recovery_start = CASE WHEN recovery_end > next_offset THEN recovery_start ELSE ? END,
                recovery_end = ?
            WHERE singleton = 1""", (start, end))


def save_readings(path: Path, records: list[tuple[str, float]], position: int | None = None) -> None:
    received_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with closing(sqlite3.connect(path, timeout=5)) as db, db:
        db.executemany("INSERT OR IGNORE INTO readings VALUES (?, ?, ?)",
                       ((timestamp, demand, received_at) for timestamp, demand in records))
        if position is not None:
            db.execute("UPDATE cursor SET next_offset = ? WHERE singleton = 1", (position,))


def read_live_actuals(path: Path, limit: int, end_time: str | None = None) -> dict:
    if not path.is_file():
        return {"records": [], "cursor": None, "recovery": None, "first_timestamp": None}
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=0.3)) as db:
        db.row_factory = sqlite3.Row
        db.execute("BEGIN")
        cursor = db.execute("SELECT next_offset, recovery_start, recovery_end FROM cursor").fetchone()
        if cursor is None:
            return {"records": [], "cursor": None, "recovery": None, "first_timestamp": None}
        first_timestamp = db.execute("SELECT MIN(timestamp) FROM readings").fetchone()[0]
        where = ""
        params = (limit,)
        if end_time:
            cutoff = datetime.fromisoformat(end_time.replace("Z", "+00:00")).astimezone(timezone.utc)
            # Both seconds-only and fractional UTC timestamps are stored.
            after = cutoff + timedelta(microseconds=1)
            where = "WHERE timestamp < ? "
            params = (after.isoformat(timespec="microseconds").replace(".000000+00:00", "+00:00"), limit)
        rows = db.execute(
            "SELECT timestamp AS utc_timestamp, demand AS actual_demand, received_at "
            f"FROM readings {where}ORDER BY timestamp DESC LIMIT ?", params,
        ).fetchall()
    current, start, end = cursor
    recovery = None
    if start is not None and end is not None:
        recovery = {"total": end - start, "processed": min(end, max(start, current or start)) - start}
    return {
        "records": [dict(row) for row in reversed(rows)], "cursor": current,
        "recovery": recovery, "first_timestamp": first_timestamp,
    }
