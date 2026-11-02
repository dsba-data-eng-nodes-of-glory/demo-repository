"""Station registry and onboarding endpoints."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from backend.pipeline_manager import (
    ROOT,
    manager,
    read_station_config,
    write_station_config,
    resolve_root_path,
)
from src.ingestion.file_loader import load_station_csv

router = APIRouter(prefix="/stations", tags=["stations"])


class DeleteStationRequest(BaseModel):
    confirm_station_id: str


def _station_key(value: str) -> str:
    key = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip()).lower().strip("-_")
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{1,31}", key):
        raise HTTPException(
            status_code=422,
            detail="Station ID must contain 2–32 letters, numbers, hyphens or underscores.",
        )
    return key


def _read_model_settings() -> dict[str, Any]:
    path = ROOT / "config" / "model.yaml"
    import yaml
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@router.get("")
def list_stations():
    return read_station_config()


@router.get("/active")
def active_station():
    config = read_station_config()
    station = config.get("active_station")
    return {"active_station": station, "available_stations": list(config.get("stations", {}).keys())}


@router.post("/onboard")
async def onboard_station(
    station_id: str = Form(...),
    file: UploadFile = File(...),
    display_name: str | None = Form(default=None),
    stream_fraction: float = Form(default=0.20),
):
    """Register a station from one continuous historical CSV.

    The earliest 80% is reserved for training/validation. The final 20% is
    saved as a separate simulated stream. The final lookback rows before that
    stream become the predictor's initial history seed.
    """
    key = _station_key(station_id)
    if not 0.10 <= stream_fraction <= 0.50:
        raise HTTPException(status_code=422, detail="stream_fraction must be between 0.10 and 0.50.")
    if not file.filename or Path(file.filename).suffix.lower() != ".csv":
        raise HTTPException(status_code=422, detail="Please upload a .csv file.")

    config = read_station_config()
    stations = config.setdefault("stations", {})
    if key in stations:
        raise HTTPException(status_code=409, detail=f"Station '{key}' is already registered.")

    model_settings = _read_model_settings()
    history_steps = int(model_settings.get("history_steps", 672))
    horizon_steps = int(model_settings.get("horizon_steps", 96))
    interval_minutes = int(model_settings.get("interval_minutes", 15))

    station_dir = ROOT / "data" / "raw" / "stations" / key
    station_dir.mkdir(parents=True, exist_ok=True)
    upload_path = station_dir / "source_upload.csv"

    try:
        with upload_path.open("wb") as destination:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                destination.write(chunk)

        frame = load_station_csv(upload_path, interval_minutes=interval_minutes)
        minimum_training_rows = int((history_steps + horizon_steps + 30) / (1.0 - 0.15)) + 1
        split_at = int(len(frame) * (1.0 - stream_fraction))
        if split_at < minimum_training_rows:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Dataset has {len(frame):,} valid rows. At least {minimum_training_rows:,} are required "
                    f"to build {history_steps} history steps, {horizon_steps} forecast steps, "
                    "and a chronological training/validation split."
                ),
            )
        if len(frame) - split_at < horizon_steps:
            raise HTTPException(
                status_code=422,
                detail=f"The reserved stream must contain at least {horizon_steps} rows.",
            )

        train_frame = frame.iloc[:split_at].copy()
        stream_frame = frame.iloc[split_at:].copy()
        history_frame = train_frame.tail(history_steps).copy()
        if len(history_frame) != history_steps:
            raise HTTPException(status_code=422, detail="Not enough rows for the initial history seed.")

        def save_csv(data: pd.DataFrame, path: Path) -> str:
            out = data[["utc_timestamp", "demand", "temperature"]].copy()
            out["utc_timestamp"] = pd.to_datetime(out["utc_timestamp"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            out.to_csv(path, index=False)
            return path.relative_to(ROOT).as_posix()

        training_path = station_dir / f"{key}_training.csv"
        history_path = station_dir / f"{key}_history.csv"
        stream_path = station_dir / f"{key}_stream.csv"
        training_rel = save_csv(train_frame, training_path)
        history_rel = save_csv(history_frame, history_path)
        stream_rel = save_csv(stream_frame, stream_path)

        checkpoint_rel = f"models/{key}/v1/{key}_encoder_decoder_lstm.pt"
        metadata_rel = f"models/{key}/v1/metadata.json"
        stations[key] = {
            "display_name": display_name.strip() if display_name and display_name.strip() else key.upper(),
            "enabled": False,
            "status": "registered",
            "training_file": training_rel,
            "history_file": history_rel,
            "input_file": stream_rel,
            "raw_topic": f"{key}-raw",
            "featured_topic": f"{key}-featured",
            "model_checkpoint": checkpoint_rel,
            "model_metadata": metadata_rel,
            "model_version": "v1",
            "stream_delay_seconds": 0.3,
            "interval_minutes": interval_minutes,
            "history_steps": history_steps,
            "horizon_steps": horizon_steps,
            "registered_source_file": file.filename,
            "registered_rows": int(len(frame)),
            "training_rows": int(len(train_frame)),
            "stream_rows": int(len(stream_frame)),
            "stream_start_utc": pd.Timestamp(stream_frame["utc_timestamp"].iloc[0]).isoformat(),
            "stream_end_utc": pd.Timestamp(stream_frame["utc_timestamp"].iloc[-1]).isoformat(),
        }
        write_station_config(config)
        return {
            "station": key,
            "display_name": stations[key]["display_name"],
            "registered": True,
            "training_file": training_rel,
            "history_file": history_rel,
            "stream_file": stream_rel,
            "training_rows": len(train_frame),
            "history_rows": len(history_frame),
            "stream_rows": len(stream_frame),
            "stream_start_utc": stations[key]["stream_start_utc"],
            "stream_end_utc": stations[key]["stream_end_utc"],
            "model_checkpoint": checkpoint_rel,
            "next_step": f"POST /stations/{key}/train",
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Station data validation failed: {exc}") from exc
    finally:
        await file.close()
        if upload_path.exists():
            upload_path.unlink(missing_ok=True)


@router.get("/{station}/status")
def station_status(station: str):
    config = read_station_config().get("stations", {})
    if station not in config:
        raise HTTPException(status_code=404, detail=f"Unknown station: {station}")
    status = manager.station_status(station)
    entry = config[station]
    services = status.get("services", {})
    if any(services.get(name, {}).get("running") for name in ("spark", "predictor", "producer")):
        lifecycle_status = "active"
    elif services.get("training", {}).get("running"):
        lifecycle_status = "training"
    elif entry.get("status") == "stopped":
        lifecycle_status = "stopped"
    elif services.get("training", {}).get("return_code") is not None:
        lifecycle_status = "model_ready" if status.get("model_ready") and services.get("training", {}).get("return_code") == 0 else "training_failed"
    elif entry.get("status") == "training":
        lifecycle_status = "training_failed"
    elif status.get("model_ready"):
        lifecycle_status = "model_ready"
    else:
        lifecycle_status = entry.get("status", "registered")
    status["lifecycle_status"] = lifecycle_status
    status.update({
        "display_name": entry.get("display_name", station),
        "registered": True,
        "training_file_exists": resolve_root_path(entry.get("training_file", "__missing__")).is_file(),
        "history_file_exists": resolve_root_path(entry.get("history_file", "__missing__")).is_file(),
        "stream_file_exists": resolve_root_path(entry.get("input_file", "__missing__")).is_file(),
        "model_metadata": entry.get("model_metadata"),
    })
    return status


@router.post("/{station}/train")
def train_station(station: str):
    try:
        result = manager.start_training(station)
        config = read_station_config()
        config["stations"][station]["status"] = "training"
        write_station_config(config)
        return result
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{station}/topics")
def create_station_topics(station: str):
    try:
        return manager.ensure_station_topics(station)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{station}/activate")
def activate_station(station: str):
    try:
        result = manager.start_pipeline(station)
        config = read_station_config()
        config["stations"][station]["enabled"] = True
        config["stations"][station]["status"] = "active"
        write_station_config(config)
        return result
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{station}/stop")
def stop_station(station: str):
    config = read_station_config().get("stations", {})
    if station not in config:
        raise HTTPException(status_code=404, detail=f"Unknown station: {station}")
    result = manager.stop_pipeline(station)
    updated = read_station_config()
    updated["stations"][station]["enabled"] = False
    updated["stations"][station]["status"] = "stopped"
    write_station_config(updated)
    return result


@router.delete("/{station}")
def delete_station(station: str, request: DeleteStationRequest):
    """Permanently delete one station and its owned records after typed confirmation."""
    if request.confirm_station_id.strip().lower() != station.strip().lower():
        raise HTTPException(
            status_code=422,
            detail="Type the exact station ID to confirm permanent deletion.",
        )
    try:
        return manager.delete_station(station, request.confirm_station_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
