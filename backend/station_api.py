"""Station registry and onboarding endpoints."""
from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.pipeline_manager import (
    ROOT,
    manager,
    read_station_config,
    write_station_config,
    resolve_root_path,
)
from src.ingestion.workbook_loader import (
    load_station_workbook,
    parse_selected_models,
)
from src.models.forecast_contract import derive_forecast_contract

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


def _write_csv(frame: pd.DataFrame, path: Path, relative_dir: Path | None = None) -> str:
    output = frame.copy()
    if "utc_timestamp" in output.columns:
        output["utc_timestamp"] = pd.to_datetime(output["utc_timestamp"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False)
    return ((relative_dir / path.name) if relative_dir is not None else path).relative_to(ROOT).as_posix()


def _validate_model_row_counts(
    datasets: dict[str, pd.DataFrame],
    selected_models: list[str],
    interval_minutes: int,
    stream_fraction: float,
    source_timezone: str,
    history_steps: int,
    horizon_steps: int,
) -> dict[str, Any]:
    """Fail onboarding early where the shipped trainers cannot form valid splits."""
    counts: dict[str, Any] = {}
    if any(model in selected_models for model in ("solar_forecast", "wind_forecast", "residual_anomaly")) and interval_minutes != 15:
        raise ValueError(
            "The selected solar, wind and residual-load anomaly trainers currently require 15-minute data. "
            "Set interval_minutes to 15 or deselect those models."
        )

    if "demand_forecast" in selected_models:
        frame = datasets["demand"]
        split_at = int(len(frame) * (1.0 - stream_fraction))
        minimum_training_rows = int((history_steps + horizon_steps + 30) / (1.0 - 0.15)) + 1
        if split_at < minimum_training_rows:
            raise ValueError(
                f"Worksheet 'Demand' has {len(frame):,} rows. At least {minimum_training_rows:,} are required "
                f"for the demand forecaster with {history_steps} history steps, {horizon_steps} forecast steps "
                "and a chronological validation split. Reduce stream_fraction only if appropriate, or add history."
            )
        stream_rows = len(frame) - split_at
        if stream_rows < horizon_steps:
            raise ValueError(
                f"Worksheet 'Demand' reserves {stream_rows:,} stream rows; at least {horizon_steps:,} are required."
            )
        counts["demand_forecast"] = {
            "total_rows": int(len(frame)),
            "training_rows": int(split_at),
            "stream_rows": int(stream_rows),
        }

    for model_name, dataset_name in (("solar_forecast", "solar"), ("wind_forecast", "wind")):
        if model_name not in selected_models:
            continue
        n_rows = len(datasets[dataset_name])
        train_end = int(n_rows * 0.70)
        val_end = int(n_rows * 0.85)
        if (
            train_end <= 672 + 96
            or val_end - train_end < 96
            or n_rows - val_end < 96
        ):
            raise ValueError(
                f"Worksheet '{dataset_name.title()}' has {n_rows:,} rows. The current Dense NN trainer needs "
                "a chronological 70/15/15 split, more than 672 history + 96 forecast rows in training, "
                "and at least 96 rows each in validation and test. Add more continuous history."
            )
        counts[model_name] = {
            "total_rows": int(n_rows),
            "train_rows": int(train_end),
            "validation_rows": int(val_end - train_end),
            "test_rows": int(n_rows - val_end),
        }

    if "residual_anomaly" in selected_models:
        from src.models.residual.training_data import build_training_frame
        demand_train = datasets['demand'].iloc[:int(len(datasets['demand'])*(1-stream_fraction))]
        combined = build_training_frame(demand_train,datasets['solar'],datasets['wind'])
        n=len(combined)
        fit_end=int(n*0.80); cal_start=int(n*0.90)
        if fit_end < 2*365*96 or cal_start-fit_end < 20*96 or n-cal_start < 20*96:
            raise ValueError("Residual Autoencoder needs at least two full years for fit plus 20 days for each validation and calibration split.")
        counts['residual_anomaly']={"total_rows":n,"fit_rows":fit_end,"validation_rows":cal_start-fit_end,"calibration_rows":n-cal_start}
    return counts


@router.get("")
def list_stations():
    return read_station_config()


@router.get("/template")
def download_station_template():
    """Download the exact workbook used by workbook onboarding."""
    path = ROOT / "templates" / "Station_Training_Template.xlsx"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Station workbook template is not installed.")
    return FileResponse(
        path=path,
        filename="Station_Training_Template.xlsx",
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


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
    timestamp_format: str = Form(default="auto"),
    source_timezone: str = Form(default="UTC"),
    interval_minutes: int = Form(default=15),
    selected_models: str = Form(default=""),
    residual_demand_unit: str = Form(default="mwh_per_interval"),
    raw_topic: str | None = Form(default=None),
    featured_topic: str | None = Form(default=None),
):
    """Register a station from the multi-sheet Excel training workbook.

    All four model components are enabled by default. ``selected_models`` may
    contain a comma-separated or JSON-array list of component IDs, allowing
    the onboarding UI to deselect each component independently.
    """
    key = _station_key(station_id)
    if not 0.10 <= stream_fraction <= 0.50:
        raise HTTPException(status_code=422, detail="stream_fraction must be between 0.10 and 0.50.")
    if not file.filename or Path(file.filename).suffix.lower() not in {".xlsx", ".xlsm"}:
        raise HTTPException(status_code=422, detail="Please upload the Station Training Template as an .xlsx workbook.")

    config = read_station_config()
    stations = config.setdefault("stations", {})
    if key in stations:
        raise HTTPException(status_code=409, detail=f"Station '{key}' is already registered.")

    try:
        selected = parse_selected_models(selected_models)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if not isinstance(residual_demand_unit, str):
        residual_demand_unit = 'mwh_per_interval'
    if residual_demand_unit not in ('mw','mwh_per_interval'):
        raise HTTPException(status_code=422, detail='Demand units must be MW or MWh per 15-minute interval')
    model_settings = _read_model_settings()
    interval_minutes = int(interval_minutes or model_settings.get("interval_minutes", 15))
    try:
        forecast_contract = derive_forecast_contract(interval_minutes, model_settings)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    history_steps = int(forecast_contract["history_steps"])
    horizon_steps = int(forecast_contract["horizon_steps"])

    station_parent = ROOT / "data" / "raw" / "stations"
    station_parent.mkdir(parents=True, exist_ok=True)
    station_dir = station_parent / key
    if station_dir.exists():
        raise HTTPException(status_code=409, detail=f"Station data folder already exists for '{key}'. Clean it up or choose another ID.")
    staging_dir = Path(tempfile.mkdtemp(prefix=f".{key}_onboard_", dir=station_parent))

    try:
        upload_path = staging_dir / "source_upload.xlsx"
        size_bytes = 0
        with upload_path.open("wb") as destination:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size_bytes += len(chunk)
                if size_bytes > 100 * 1024 * 1024:
                    raise HTTPException(status_code=413, detail="Workbook exceeds the 100 MB onboarding upload limit.")
                destination.write(chunk)
        if size_bytes == 0:
            raise HTTPException(status_code=422, detail="The uploaded workbook is empty.")

        try:
            datasets, workbook_diagnostics = load_station_workbook(
                upload_path,
                selected_models=selected,
                interval_minutes=interval_minutes,
                timestamp_format=timestamp_format,
                source_timezone=source_timezone or "UTC",
            )
            model_row_counts = _validate_model_row_counts(
                datasets, selected, interval_minutes, stream_fraction,
                source_timezone or "UTC", history_steps, horizon_steps,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        # Preserve the original workbook for traceability, and write normalized
        # model-facing CSVs alongside it. The source remains separate from the
        # generated training/stream partitions.
        source_filename = f"{key}_source.xlsx"
        upload_path.rename(staging_dir / source_filename)

        model_components: dict[str, dict[str, Any]] = {}
        relative_files: dict[str, str] = {}
        demand_frame = datasets.get("demand")
        demand_forecast_paths: dict[str, str] = {}

        if demand_frame is not None:
            if "demand_forecast" in selected:
                split_at = int(len(demand_frame) * (1.0 - stream_fraction))
                train_frame = demand_frame.iloc[:split_at].copy()
                stream_frame = demand_frame.iloc[split_at:].copy()
                history_frame = train_frame.tail(history_steps).copy()

                demand_training = staging_dir / f"{key}_demand_training.csv"
                demand_history = staging_dir / f"{key}_demand_history.csv"
                demand_stream = staging_dir / f"{key}_demand_stream.csv"
                demand_forecast_paths = {
                    "training_file": _write_csv(train_frame, demand_training, station_dir),
                    "history_file": _write_csv(history_frame, demand_history, station_dir),
                    "input_file": _write_csv(stream_frame, demand_stream, station_dir),
                }
                relative_files.update({f"demand_{name}": value for name, value in demand_forecast_paths.items()})
                model_components["demand_forecast"] = {
                    "enabled": True,
                    "algorithm": "lstm",
                    "status": "pending_training",
                    **demand_forecast_paths,
                    "checkpoint": f"models/{key}/v1/{key}_encoder_decoder_lstm.pt",
                    "metadata": f"models/{key}/v1/metadata.json",
                }
                if "temperature" not in demand_frame.columns:
                    raise HTTPException(status_code=422, detail="Worksheet 'Demand' must include temperature when demand_forecast is selected.")
            else:
                model_components["demand_forecast"] = {
                    "enabled": False, "algorithm": "lstm", "status": "deselected"
                }

        else:
            model_components["demand_forecast"] = {"enabled": False, "algorithm": "lstm", "status": "deselected"}

        # Residual-load data is derived from all three historical worksheets,
        # not a separate demand-only input. Preserve a validated combined training snapshot.
        if "residual_anomaly" in selected:
            from src.models.residual.training_data import build_training_frame
            # Train only before replay starts; no future test-stream measurements.
            demand_train=datasets['demand'].iloc[:int(len(datasets['demand'])*(1-stream_fraction))]
            residual_frame=build_training_frame(demand_train,datasets['solar'],datasets['wind'],residual_demand_unit)
            residual_file=staging_dir / f"{key}_residual_training.csv"
            residual_rel=_write_csv(residual_frame,residual_file,station_dir)
            relative_files['residual_training_file']=residual_rel
            model_components['residual_anomaly']={
                'enabled':True,'algorithm':'dense_autoencoder','status':'pending_training',
                'training_file':residual_rel,
                'checkpoint':f'models/{key}/residual_anomaly/residual_ae_portable.pt',
                'metadata':f'models/{key}/residual_anomaly/training_summary.json',
            }
        else:
            model_components['residual_anomaly']={'enabled':False,'algorithm':'dense_autoencoder','status':'deselected'}

        renewable_components: dict[str, Any] = {}
        for kind in ("solar", "wind"):
            model_id = f"{kind}_forecast"
            if model_id in selected:
                frame = datasets[kind]
                data_path = staging_dir / f"{key}_{kind}_data.csv"
                data_rel = _write_csv(frame, data_path, station_dir)
                relative_files[f"{kind}_data_file"] = data_rel
                if kind == "solar":
                    checkpoint = f"models/{key}/solar_dense_nn/solar_pv_mlp_v4.pt"
                    metadata = f"models/{key}/solar_dense_nn/solar_mlp_v4_metadata.json"
                    version = f"{key}-solar-mlp-v4"
                else:
                    checkpoint = f"models/{key}/wind_dense_nn/amprion_wind_dense_solar_aligned.pt"
                    metadata = f"models/{key}/wind_dense_nn/metadata.json"
                    version = f"{key}-wind-dense-v1"
                renewable_components[kind] = {
                    "enabled": True,
                    # One complete canonical dataset is used for training and
                    # later serving-time history/future-weather lookup.
                    "train_file": data_rel,
                    "test_file": None,
                    "data_file": data_rel,
                    "checkpoint": checkpoint,
                    "metadata": metadata,
                    "model_version": version,
                    "status": "pending_training",
                }
                model_components[model_id] = {
                    "enabled": True,
                    "algorithm": "dense_nn",
                    "status": "pending_training",
                    "training_file": data_rel,
                    "checkpoint": checkpoint,
                    "metadata": metadata,
                }
            else:
                renewable_components[kind] = {"enabled": False, "status": "deselected"}
                model_components[model_id] = {
                    "enabled": False, "algorithm": "dense_nn", "status": "deselected"
                }

        # Solar and wind may be trained and deployed independently. Newly
        # registered components remain pending until a model is deployed;
        # the serving adapter skips missing checkpoints safely.
        any_renewable_selected = any(x in selected for x in ("solar_forecast", "wind_forecast"))
        final_station_dir = station_dir
        staging_dir.rename(final_station_dir)

        display = display_name.strip() if display_name and display_name.strip() else key.upper()
        station_entry: dict[str, Any] = {
            "display_name": display,
            "enabled": False,
            "status": "registered",
            "enabled_models": selected,
            "model_components": model_components,
            "raw_topic": (raw_topic or "").strip() or f"{key}-raw",
            "featured_topic": (featured_topic or "").strip() or f"{key}-featured",
            "actuals_file": f"data/processed/stations/{key}/actuals.csv",
            "forecasts_file": f"data/processed/stations/{key}/forecasts.csv",
            "renewable_forecasts_file": f"data/processed/stations/{key}/renewable_forecasts.csv",
            "interval_minutes": interval_minutes,
            "history_window_hours": forecast_contract["history_window_hours"],
            "forecast_horizon_hours": forecast_contract["forecast_horizon_hours"],
            "history_steps": history_steps,
            "horizon_steps": horizon_steps,
            "source_timezone": source_timezone or "UTC",
            "timestamp_format": timestamp_format,
            "registered_source_file": file.filename,
            "archived_source_file": f"data/raw/stations/{key}/{source_filename}",
            "registered_file_size_bytes": int(size_bytes),
            "model_data_diagnostics": workbook_diagnostics,
            "model_row_counts": model_row_counts,
            "stream_fraction": float(stream_fraction),
            "residual_demand_unit": residual_demand_unit,
            "stream_delay_seconds": 0.3,
            "renewables": {
                "enabled": any_renewable_selected,
                "components_enabled": {
                    "solar": "solar_forecast" in selected,
                    "wind": "wind_forecast" in selected,
                },
                "serving_adapter_supported": any_renewable_selected,
                "interval_minutes": interval_minutes,
                "forecast_every_steps": 1,
                **renewable_components,
            },
        }
        if demand_frame is not None:
            station_entry.update({
                "registered_rows": int(len(demand_frame)),
                "demand_data_file": None,
                "demand_start_utc": pd.Timestamp(demand_frame["utc_timestamp"].iloc[0]).isoformat(),
                "demand_end_utc": pd.Timestamp(demand_frame["utc_timestamp"].iloc[-1]).isoformat(),
            })
        if "demand_forecast" in selected:
            train_frame = demand_frame.iloc[:int(len(demand_frame) * (1.0 - stream_fraction))]
            stream_frame = demand_frame.iloc[int(len(demand_frame) * (1.0 - stream_fraction)):]
            checkpoint_rel = f"models/{key}/v1/{key}_encoder_decoder_lstm.pt"
            station_entry.update({
                "training_file": demand_forecast_paths["training_file"],
                "history_file": demand_forecast_paths["history_file"],
                "input_file": demand_forecast_paths["input_file"],
                "model_checkpoint": checkpoint_rel,
                "model_metadata": f"models/{key}/v1/metadata.json",
                "model_version": "v1",
                "model_algorithm": "lstm",
                "training_rows": int(len(train_frame)),
                "history_rows": int(min(history_steps, len(train_frame))),
                "stream_rows": int(len(stream_frame)),
                "stream_start_utc": pd.Timestamp(stream_frame["utc_timestamp"].iloc[0]).isoformat(),
                "stream_end_utc": pd.Timestamp(stream_frame["utc_timestamp"].iloc[-1]).isoformat(),
            })
        if "residual_anomaly" in selected:
            station_entry["residual_model_checkpoint"] = model_components["residual_anomaly"]["checkpoint"]
            station_entry["residual_model_summary"] = model_components["residual_anomaly"]["metadata"]
            station_entry["residual_detection_settings"] = {
                "decision_rule": "both", "upward_only": True,
                "stat_threshold": None, "ae_threshold": None,
            }

        stations[key] = station_entry
        try:
            write_station_config(config)
        except Exception:
            shutil.rmtree(final_station_dir, ignore_errors=True)
            raise

        topic_result = None
        topic_warning = None
        if "demand_forecast" in selected:
            try:
                topic_result = manager.ensure_station_topics(key)
                updated = read_station_config()
                updated["stations"][key]["topics_ready"] = True
                write_station_config(updated)
            except Exception as exc:
                topic_warning = str(exc)
                updated = read_station_config()
                updated.setdefault("stations", {}).setdefault(key, {})["topics_ready"] = False
                updated["stations"][key]["topic_setup_message"] = topic_warning
                write_station_config(updated)
        else:
            updated = read_station_config()
            updated["stations"][key]["topics_ready"] = False
            updated["stations"][key]["topic_setup_message"] = (
                "Demand forecasting was deselected; demand streaming topics were not initialized."
            )
            write_station_config(updated)

        return {
            "station": key,
            "display_name": display,
            "registered": True,
            "enabled_models": selected,
            "model_components": model_components,
            "files": relative_files,
            "model_row_counts": model_row_counts,
            "workbook_diagnostics": workbook_diagnostics,
            "topics": topic_result,
            "topics_ready": bool(topic_result),
            "topic_warning": topic_warning,
            "renewable_serving_adapter_ready": any_renewable_selected,
            "next_step": "Configure and train the selected models from Model Management. The streaming pipeline remains a separate activation step.",
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Station workbook onboarding failed: {exc}") from exc
    finally:
        await file.close()
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)


def _status_from_snapshot(station: str, entry: dict, snapshot: dict) -> dict:
    """Compose a station status from one shared process/runtime snapshot."""
    processes = snapshot.get("processes", {})
    services = {
        name: processes.get(manager.process_key(name, station), {
            "running": False, "pid": None, "return_code": None,
        })
        for name in manager.SERVICE_NAMES
    }
    active_model = entry.get("active_model") or {}
    checkpoint_value = active_model.get("artifact_path") or entry.get("model_checkpoint")
    checkpoint = resolve_root_path(checkpoint_value) if checkpoint_value else ROOT / "__missing_model__"
    status = {
        "station": station,
        "kafka_tcp_available": bool(snapshot.get("kafka_tcp_available")),
        "services": services,
        "quality_consumer": processes.get("quality", {"running": False, "pid": None, "return_code": None}),
        "model_checkpoint": str(checkpoint),
        "model_ready": checkpoint.is_file(),
        "active_model": active_model or None,
        "topics": {"raw": entry.get("raw_topic"), "featured": entry.get("featured_topic")},
    }
    if any(services.get(name, {}).get("running") for name in ("spark", "predictor", "producer")):
        lifecycle = "active"
    elif services.get("training", {}).get("running"):
        lifecycle = "training"
    elif entry.get("status") == "stopped":
        lifecycle = "stopped"
    elif services.get("training", {}).get("return_code") is not None:
        lifecycle = "model_ready" if status["model_ready"] and services["training"]["return_code"] == 0 else "training_failed"
    elif entry.get("status") == "training":
        lifecycle = "training_failed"
    elif status["model_ready"]:
        lifecycle = "model_ready"
    else:
        lifecycle = entry.get("status", "registered")
    status.update({
        "lifecycle_status": lifecycle,
        "display_name": entry.get("display_name", station),
        "registered": True,
        "training_file_exists": bool(entry.get("training_file") and resolve_root_path(entry["training_file"]).is_file()),
        "history_file_exists": bool(entry.get("history_file") and resolve_root_path(entry["history_file"]).is_file()),
        "stream_file_exists": bool(entry.get("input_file") and resolve_root_path(entry["input_file"]).is_file()),
        "model_metadata": entry.get("model_metadata"),
        "enabled_models": entry.get("enabled_models", []),
        "model_components": entry.get("model_components", {}),
        "interval_minutes": int(entry.get("interval_minutes", 15)),
        "history_steps": int(entry.get("history_steps", 672)),
        "horizon_steps": int(entry.get("horizon_steps", 96)),
        "history_window_hours": float(entry.get("history_window_hours", 168)),
        "forecast_horizon_hours": float(entry.get("forecast_horizon_hours", 24)),
    })
    return status


@router.get("/overview")
def stations_overview():
    """Dashboard summary; a single process/Java/Kafka check for all stations.

    This endpoint is informational and never changes running processes.
    """
    from backend.training_data import station_component_state, registered_source
    config = read_station_config()
    entries = config.get("stations", {})
    snapshot = manager.status()
    statuses = {}
    component_statuses = {}
    for station, entry in entries.items():
        statuses[station] = _status_from_snapshot(station, entry, snapshot)
        enabled = station_component_state(entry)
        details = {}
        for component, selected in enabled.items():
            component_entry = (entry.get("model_components") or {}).get(component) or {}
            if component == "demand_forecast":
                checkpoint_value = (entry.get("active_model") or {}).get("artifact_path") or entry.get("model_checkpoint")
            elif component == "residual_anomaly":
                checkpoint_value = component_entry.get("checkpoint") or entry.get("residual_model_checkpoint")
            else:
                kind = component.split("_")[0]
                checkpoint_value = (entry.get("renewables") or {}).get(kind, {}).get("checkpoint") or component_entry.get("checkpoint")
            source = registered_source(entry, component)
            details[component] = {
                "enabled": bool(selected),
                "training_data_exists": bool(source and resolve_root_path(source).is_file()),
                "checkpoint_exists": bool(checkpoint_value and resolve_root_path(checkpoint_value).is_file()),
            }
        component_statuses[station] = details
    return {
        "active_station": config.get("active_station"),
        "stations": entries,
        "statuses": statuses,
        "components": component_statuses,
        "kafka_tcp_available": bool(snapshot.get("kafka_tcp_available")),
    }


@router.get("/{station}/status")
def station_status(station: str):
    entries = read_station_config().get("stations", {})
    if station not in entries:
        raise HTTPException(status_code=404, detail=f"Unknown station: {station}")
    return _status_from_snapshot(station, entries[station], manager.status())


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
    stop_result = manager.stop_pipeline(station)
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

# ---------------------------------------------------------------------------
# Per-station residual spike detection thresholds.
# Updates are explicit and stored in stations.yaml; they never retrain an AE.
# ---------------------------------------------------------------------------
from pydantic import Field
from typing import Literal
from src.models.residual.service import DEFAULT_SETTINGS, settings_for

class ResidualDetectionSettingsRequest(BaseModel):
    decision_rule: Literal['both', 'either'] = 'both'
    stat_threshold: float | None = Field(default=None, gt=0, le=10000)
    ae_threshold: float | None = Field(default=None, gt=0, le=10000)
    upward_only: bool = True

@router.get('/{station}/residual-settings')
def get_residual_settings(station: str):
    config=read_station_config()
    entry=config.get('stations',{}).get(station)
    if entry is None:raise HTTPException(status_code=404,detail='Unknown station')
    return {'station':station,'settings':settings_for(entry),
            'checkpoint':entry.get('residual_model_checkpoint')}

@router.put('/{station}/residual-settings')
def put_residual_settings(station: str, request: ResidualDetectionSettingsRequest):
    config=read_station_config()
    entry=config.get('stations',{}).get(station)
    if entry is None:raise HTTPException(status_code=404,detail='Unknown station')
    # No edits to the physical pretrained checkpoint; thresholds apply at inference.
    entry['residual_detection_settings']=request.model_dump() if hasattr(request,'model_dump') else request.dict()
    entry.pop('residual_alert_settings', None)
    write_station_config(config)
    return {'station':station,'settings':entry['residual_detection_settings'],
            'message':'Spike-detection thresholds saved for the next forecast origin.'}
