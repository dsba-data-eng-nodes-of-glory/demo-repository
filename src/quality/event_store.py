"""Persist data-quality events to Cassandra or a local JSONL fallback."""
from __future__ import annotations

import json
import os
from collections import deque
from pathlib import Path
from threading import Lock
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
_RECENT_LIMIT = 20_000
_RECENT_IDS: set[str] = set()
_RECENT_ORDER: deque[str] = deque()
_CACHE_READY = False
_LOCK = Lock()


def _load_recent_ids(path: Path) -> None:
    """Load the dedupe window once, not once per appended event."""
    global _CACHE_READY
    if _CACHE_READY:
        return
    if path.exists():
        # The cache is intentionally bounded. Read only a tail window so a
        # long-running JSONL history does not need to be loaded into memory.
        try:
            with path.open("rb") as handle:
                handle.seek(0, os.SEEK_END)
                size = handle.tell()
                handle.seek(max(0, size - 8 * 1024 * 1024), os.SEEK_SET)
                lines = handle.read().decode("utf-8", errors="replace").splitlines()
            for line in lines[-_RECENT_LIMIT:]:
                try:
                    event_id = str(json.loads(line).get("event_id") or "")
                except (json.JSONDecodeError, AttributeError):
                    continue
                if event_id and event_id not in _RECENT_IDS:
                    _RECENT_IDS.add(event_id)
                    _RECENT_ORDER.append(event_id)
        except OSError:
            # The subsequent append will surface a real storage failure.
            pass
    while len(_RECENT_ORDER) > _RECENT_LIMIT:
        _RECENT_IDS.discard(_RECENT_ORDER.popleft())
    _CACHE_READY = True


def record_quality_event(event: dict[str, Any]) -> None:
    override = os.getenv("STORAGE_BACKEND")
    config_path = ROOT / "config" / "storage.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    backend = (override or (config or {}).get("backend", "csv")).lower()
    if backend == "cassandra":
        from src.storage.cassandra_store import write_quality_event
        write_quality_event(event)
        return

    path = ROOT / "data" / "quality" / "quality_events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    event_id = str(event.get("event_id", ""))
    with _LOCK:
        _load_recent_ids(path)
        if event_id and event_id in _RECENT_IDS:
            return
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, default=str) + "\n")
            handle.flush()
        if event_id:
            _RECENT_IDS.add(event_id)
            _RECENT_ORDER.append(event_id)
            while len(_RECENT_ORDER) > _RECENT_LIMIT:
                _RECENT_IDS.discard(_RECENT_ORDER.popleft())
