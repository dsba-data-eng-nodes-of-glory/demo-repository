"""Small file-backed experiment registry for the electricity forecasting POC.

The registry is deliberately local and dependency-light. MLflow can replace
this storage layer later without changing the portal's experiment contract.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.ingestion.station_lock import acquire_station_lock, release_station_lock

ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = ROOT / "models" / "registry" / "experiments.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_document() -> dict[str, Any]:
    if not REGISTRY_PATH.exists():
        return {"schema_version": 1, "experiments": []}
    try:
        payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Model registry is not valid JSON: {REGISTRY_PATH}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("experiments"), list):
        raise RuntimeError(f"Model registry has an unsupported structure: {REGISTRY_PATH}")
    payload.setdefault("schema_version", 1)
    return payload


def _write_document(payload: dict[str, Any]) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix="experiments_", suffix=".json.tmp", dir=REGISTRY_PATH.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        Path(temp_name).replace(REGISTRY_PATH)
    finally:
        Path(temp_name).unlink(missing_ok=True)


@contextmanager
def _registry_write_lock():
    lock_path = REGISTRY_PATH.with_suffix(".lock")
    deadline = time.monotonic() + 5
    while True:
        try:
            handle = acquire_station_lock(lock_path, "model registry writer")
            break
        except RuntimeError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.05)
    try:
        yield
    finally:
        release_station_lock(handle)


def list_experiments(station: str | None = None) -> list[dict[str, Any]]:
    items = [item for item in _read_document()["experiments"] if not item.get("deleted_at")]
    if station:
        items = [item for item in items if item.get("station") == station]
    return sorted(items, key=lambda item: item.get("created_at", ""), reverse=True)


def get_experiment(experiment_id: str) -> dict[str, Any] | None:
    for item in _read_document()["experiments"]:
        if item.get("experiment_id") == experiment_id and not item.get("deleted_at"):
            return item
    return None


def upsert_experiment(record: dict[str, Any]) -> dict[str, Any]:
    with _registry_write_lock():
        payload = _read_document()
        items = payload["experiments"]
        for index, existing in enumerate(items):
            if existing.get("experiment_id") == record.get("experiment_id"):
                if existing.get("deleted_at"):
                    raise ValueError("A deleted experiment cannot be updated.")
                items[index] = {**existing, **record, "updated_at": utc_now()}
                saved = items[index]
                break
        else:
            saved = {**record, "created_at": record.get("created_at") or utc_now(), "updated_at": utc_now()}
            items.append(saved)
        _write_document(payload)
        return saved


def mark_experiment_deleted(experiment_id: str) -> None:
    """Retain only the identity and version needed to avoid reusing version numbers."""
    with _registry_write_lock():
        payload = _read_document()
        for index, item in enumerate(payload["experiments"]):
            if item.get("experiment_id") != experiment_id:
                continue
            if item.get("status") not in {"candidate", "previous", "failed"} or item.get("deleted_at"):
                raise ValueError("Only inactive, finished experiments can be deleted.")
            keys = ("experiment_id", "station", "algorithm", "component", "version")
            payload["experiments"][index] = {
                **{key: item[key] for key in keys if key in item}, "deleted_at": utc_now(),
            }
            _write_document(payload)
            return
        raise ValueError("Experiment not found.")


def next_version(station: str, algorithm: str, component: str = "demand_forecast") -> str:
    versions = []
    for item in _read_document()["experiments"]:
        if (item.get("station") == station and item.get("algorithm") == algorithm
                and item.get("component", "demand_forecast") == component):
            value = str(item.get("version", "v0")).lstrip("v")
            try:
                versions.append(int(value))
            except ValueError:
                continue
    return f"v{max(versions, default=0) + 1}"
