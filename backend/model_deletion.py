"""Remove an inactive, station-owned experiment and its managed artifacts."""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
import re
import shutil

from fastapi import HTTPException

from src.models.model_registry import list_experiments, mark_experiment_deleted

LOG = logging.getLogger(__name__)
DELETABLE_STATUSES = {"candidate", "previous", "failed"}
EXPERIMENT_PATH_KEYS = ("artifact_path", "metadata_path", "config_path", "training_summary_path")
STATION_PATH_KEYS = {"model_checkpoint", "model_metadata", "checkpoint", "metadata",
                     "artifact_path", "metadata_path", "residual_model_checkpoint", "residual_model_summary",
                     "training_file", "train_file", "test_file", "data_file", "input_file", "history_file"}


def station_references(stations: dict, root: Path) -> tuple[set, set[Path]]:
    ids, paths = set(), set()

    def collect(config: dict) -> None:
        for key, value in config.items():
            if isinstance(value, dict):
                collect(value)
            elif key in {"experiment_id", "active_experiment_id"} and isinstance(value, str):
                ids.add(value)
            elif key in STATION_PATH_KEYS and isinstance(value, str) and value:
                paths.add((root / value).resolve())

    for entry in stations.values():
        collect({key: value for key, value in entry.items() if key != "initial_training"})
        for task in (entry.get("initial_training") or {}).values():
            if task.get("status") in {"queued", "running"} and task.get("experiment_id"):
                ids.add(task["experiment_id"])
    return ids, paths


def can_delete(record: dict, ids: set, paths: set[Path], root: Path) -> bool:
    return (
        record.get("status") in DELETABLE_STATUSES
        and record.get("experiment_id") not in ids
        and not any((root / record[key]).resolve() in paths for key in EXPERIMENT_PATH_KEYS if record.get(key))
    )


def delete_inactive_experiment(record: dict, stations: dict, root: Path) -> bool:
    """Return whether disk cleanup completed after committing the deletion."""
    station, experiment_id = record["station"], record["experiment_id"]
    for name in (station, experiment_id):
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise HTTPException(status_code=409, detail="The experiment uses an unsupported storage path.")
    root = root.resolve()
    ids, paths = station_references(stations, root)
    if not can_delete(record, ids, paths, root):
        raise HTTPException(status_code=409, detail="This version is deployed, referenced by a station, or still training.")
    directory = root / "models" / station / "experiments" / experiment_id
    for other in list_experiments():
        keys = ("training_data_path",) if other.get("experiment_id") == experiment_id else (*EXPERIMENT_PATH_KEYS, "training_data_path")
        paths.update((root / other[key]).resolve() for key in keys if other.get(key))
    if any(path.is_relative_to(directory) for path in paths):
        raise HTTPException(status_code=409, detail="Another model or dataset references this experiment folder.")
    if any(not (root / record[key]).resolve().is_relative_to(directory)
           for key in EXPERIMENT_PATH_KEYS if record.get(key)):
        raise HTTPException(status_code=409, detail="This version has files outside its managed experiment folder.")

    digest = hashlib.sha256(experiment_id.encode()).hexdigest()[:20]
    staged = directory.with_name(f".deleted-{digest}")
    if any(path.resolve() != path or (path.exists() and not path.is_dir()) for path in (directory, staged)):
        raise HTTPException(status_code=409, detail="The model folder is not a safe managed directory.")
    if directory.exists() and staged.exists():
        raise HTTPException(status_code=409, detail="The experiment folder has a pending cleanup conflict.")

    try:
        if directory.exists():
            directory.rename(staged)
    except OSError as exc:
        raise HTTPException(status_code=409, detail=f"Could not move the model folder: {exc}") from exc

    try:
        mark_experiment_deleted(experiment_id)
    except (ValueError, RuntimeError, OSError) as exc:
        if staged.exists():
            staged.rename(directory)
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    if staged.exists():
        try:
            shutil.rmtree(staged)
        except OSError:
            LOG.exception("Model %s was removed from the registry; disk cleanup is pending at %s.", experiment_id, staged)
            return False
    return True
