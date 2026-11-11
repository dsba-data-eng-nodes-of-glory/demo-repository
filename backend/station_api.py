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
from src.models.forecast_contract import derive_forecast_contract
from src.models.time_segments import continuous_segment_bounds, split_forecast_origins

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
    timestamp_column: str | None = Form(default=None),
    timestamp_format: str = Form(default="auto"),
    source_timezone: str = Form(default="UTC"),
    interval_minutes: int = Form(default=15),
    raw_topic: str | None = Form(default=None),
    featured_topic: str | None = Form(default=None),
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
    interval_minutes = int(interval_minutes or model_settings.get("interval_minutes", 15))
    try:
        forecast_contract = derive_forecast_contract(interval_minutes, model_settings)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    history_steps = int(forecast_contract["history_steps"])
    horizon_steps = int(forecast_contract["horizon_steps"])

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

        frame = load_station_csv(
            upload_path,
            interval_minutes=interval_minutes,
            resample_to_interval=True,
            timestamp_column=timestamp_column,
            timestamp_format=timestamp_format,
            # The current onboarding UI intentionally assumes UTC for naive
            # timestamps. Offset-aware values are normalized by the loader.
            source_timezone=source_timezone or "UTC",
        )
        detected_source_columns = dict(frame.attrs.get("source_columns", {}))
        temperature_available = bool(frame.attrs.get("temperature_available", True))
        timestamp_format_used = frame.attrs.get("timestamp_format_used", timestamp_format)
        source_row_count = int(frame.attrs.get("source_row_count", len(frame)))
        segments = continuous_segment_bounds(frame["utc_timestamp"], interval_minutes)
        if not segments:
            raise HTTPException(
                status_code=422,
                detail="No observations remain after resampling to the selected interval.",
            )
        # The most recent continuous segment backs the live demo stream. Earlier
        # segments remain in the training CSV and are sampled independently by
        # the training window builder.
        latest_start, latest_end = segments[-1]
        latest_segment_rows = latest_end - latest_start
        if latest_segment_rows < history_steps + horizon_steps:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"The latest continuous segment has {latest_segment_rows:,} resampled rows; "
                    f"it needs at least {history_steps + horizon_steps:,} rows for a "
                    f"{history_steps}-step history and {horizon_steps}-step stream. "
                    "Earlier segments can train the model, but cannot be joined across a gap "
                    "to seed the live stream."
                ),
            )

        requested_stream_rows = max(horizon_steps, int(latest_segment_rows * stream_fraction))
        stream_rows = min(requested_stream_rows, latest_segment_rows - history_steps)
        stream_start = latest_end - stream_rows
        history_start = stream_start - history_steps
        # Keep every resampled observation in the training source. The model
        # trainer makes a chronological holdout inside each independent segment.
        train_frame = frame.copy()
        history_frame = frame.iloc[history_start:stream_start].copy()
        stream_frame = frame.iloc[stream_start:latest_end].copy()
        if len(history_frame) != history_steps:
            raise HTTPException(status_code=422, detail="Not enough rows for the initial history seed.")

        train_origins, validation_origins = split_forecast_origins(
            train_frame["utc_timestamp"], history_steps, horizon_steps, interval_minutes,
            validation_fraction=0.15,
        )
        if len(train_origins) < 2 or len(validation_origins) < 1:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"After resampling, there are {len(train_origins)} continuous training windows "
                    f"and {len(validation_origins)} validation windows across {len(segments)} segments. "
                    "The model needs at least two training windows and one validation window; "
                    "gaps are kept separate and never crossed."
                ),
            )

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
            "raw_topic": (raw_topic or "").strip() or f"{key}-raw",
            "featured_topic": (featured_topic or "").strip() or f"{key}-featured",
            "actuals_file": f"data/processed/stations/{key}/actuals.csv",
            "forecasts_file": f"data/processed/stations/{key}/forecasts.csv",
            # Store detected source mappings for reproducibility; users are not
            # asked to select these columns in the onboarding form.
            "timestamp_column": detected_source_columns.get("timestamp"),
            "demand_column": detected_source_columns.get("demand"),
            "temperature_column": detected_source_columns.get("temperature"),
            "temperature_available": temperature_available,
            "temperature_handling": (
                "measured" if temperature_available else "constant_zero_compatibility_placeholder"
            ),
            "timestamp_format": timestamp_format,
            "timestamp_format_used": timestamp_format_used,
            "source_timezone_assumption": "UTC for naive timestamps; timezone-aware values converted to UTC",
            "model_checkpoint": checkpoint_rel,
            "model_metadata": metadata_rel,
            "model_version": "v1",
            "stream_delay_seconds": 0.3,
            "interval_minutes": interval_minutes,
            "history_window_hours": forecast_contract["history_window_hours"],
            "forecast_horizon_hours": forecast_contract["forecast_horizon_hours"],
            "history_steps": history_steps,
            "horizon_steps": horizon_steps,
            "registered_source_file": file.filename,
            "registered_rows": int(len(frame)),
            "registered_source_rows": source_row_count,
            "resampling_method": frame.attrs.get("resampling_method", "mean"),
            "continuous_segment_count": int(len(segments)),
            "training_forecast_windows": int(len(train_origins)),
            "validation_forecast_windows": int(len(validation_origins)),
            "training_rows": int(len(train_frame)),
            "stream_rows": int(len(stream_frame)),
            "stream_start_utc": pd.Timestamp(stream_frame["utc_timestamp"].iloc[0]).isoformat(),
            "stream_end_utc": pd.Timestamp(stream_frame["utc_timestamp"].iloc[-1]).isoformat(),
        }
        write_station_config(config)
        # Topic names default to <station>-raw and <station>-featured when
        # left blank in the registration form. Try to create them immediately;
        # registration remains valid if Kafka is temporarily unavailable.
        topic_result = None
        topic_warning = None
        try:
            topic_result = manager.ensure_station_topics(key)
            config = read_station_config()
            config["stations"][key]["topics_ready"] = True
            write_station_config(config)
        except Exception as exc:
            topic_warning = str(exc)
            config = read_station_config()
            config["stations"][key]["topics_ready"] = False
            config["stations"][key]["topic_setup_message"] = topic_warning
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
            "detected_source_columns": detected_source_columns,
            "source_rows": source_row_count,
            "resampled_rows": len(frame),
            "continuous_segment_count": len(segments),
            "training_forecast_windows": len(train_origins),
            "validation_forecast_windows": len(validation_origins),
            "aggregation_method": "mean",
            "resampling_note": (
                f"Aggregated {source_row_count:,} source rows into {len(frame):,} "
                f"{interval_minutes}-minute mean values across {len(segments)} separate continuous segments. "
                "Each segment has its own chronological validation split; no training window crosses a gap. "
                "The latest segment supplies the demo stream."
            ),
            "temperature_available": temperature_available,
            "temperature_note": (
                None if temperature_available else
                "No temperature column was supplied. A constant 0 placeholder keeps the existing model input schema; it is not measured weather, so this station's models cannot learn temperature effects."
            ),
            "timestamp_format_used": timestamp_format_used,
            "model_checkpoint": checkpoint_rel,
            "topics": topic_result,
            "topics_ready": bool(topic_result),
            "topic_warning": topic_warning,
            "next_step": "Train a model from Model Management, then activate this station from Home.",
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
        "interval_minutes": int(entry.get("interval_minutes", 15)),
        "history_steps": int(entry.get("history_steps", 672)),
        "horizon_steps": int(entry.get("horizon_steps", 96)),
        "history_window_hours": float(entry.get("history_window_hours", 168)),
        "forecast_horizon_hours": float(entry.get("forecast_horizon_hours", 24)),
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
        result = manager.ensure_station_topics(station)
        config = read_station_config()
        config["stations"][station]["topics_ready"] = True
        config["stations"][station].pop("topic_setup_message", None)
        write_station_config(config)
        return result
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
    # Stop only this station, then automatically clear its Kafka records,
    # Spark checkpoint and station-scoped result files. Other stations remain active.
    stop_result = manager.stop_pipeline(station)
    # Migrate this station to isolated output files before reset. This permits
    # stopping it while other station predictors continue running. Legacy
    # shared CSVs remain untouched as historical archives.
    updated = read_station_config()
    station_entry = updated["stations"][station]
    station_entry.setdefault("actuals_file", f"data/processed/stations/{station}/actuals.csv")
    station_entry.setdefault("forecasts_file", f"data/processed/stations/{station}/forecasts.csv")
    write_station_config(updated)
    reset_result = manager.reset_station_pipeline(station)
    updated = read_station_config()
    updated["stations"][station]["enabled"] = False
    updated["stations"][station]["status"] = "stopped"
    write_station_config(updated)
    return {"station": station, "stopped": stop_result, "reset": reset_result,
            "message": f"{station} stopped and reset. Its source files and trained model were preserved."}


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
