from pathlib import Path
import json

from fastapi import APIRouter, HTTPException

from backend.pipeline_manager import read_station_config, ROOT, manager, resolve_root_path

router = APIRouter(prefix="/models", tags=["models"])


@router.get("/status")
def model_status(station: str | None = None):
    config = read_station_config()
    station = station or config.get("active_station")
    stations = config.get("stations", {})
    if station not in stations:
        raise HTTPException(status_code=404, detail=f"Unknown station: {station}")

    entry = stations[station]
    checkpoint_path = resolve_root_path(entry["model_checkpoint"])
    metadata_value = entry.get("model_metadata")
    metadata_path = resolve_root_path(metadata_value) if metadata_value else checkpoint_path.parent / "metadata.json"
    metadata = None
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            metadata = {"warning": "Model metadata exists but could not be parsed."}

    training = manager.station_status(station)["services"]["training"]
    return {
        "station": station,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_exists": checkpoint_path.is_file(),
        "metadata_path": str(metadata_path),
        "metadata_exists": metadata_path.is_file(),
        "metadata": metadata,
        "training_process": training,
        "model_version": entry.get("model_version"),
        "model_status": entry.get("status", "unknown"),
    }
