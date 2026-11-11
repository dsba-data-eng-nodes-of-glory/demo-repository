"""Immutable, station-scoped training-data uploads and legacy component discovery.

Uploads are *experiment inputs*, never replacements for a station's stream file,
active model, or original onboarding dataset. The payload is validated using
exactly the same canonical column names as workbook onboarding.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import HTTPException, UploadFile

from src.ingestion.workbook_loader import (
    MODEL_IDS, SOLAR_COLUMNS, WIND_COLUMNS, TIMESTAMP_ALIASES,
    DEMAND_ALIASES, TEMPERATURE_ALIASES,
    _read_sheet, load_station_workbook,
)

MAX_UPLOAD_BYTES = 100 * 1024 * 1024
SOURCE_ID_RE = re.compile(r"^[0-9a-f]{32}$")
COMPONENT_TO_SHEET = {
    "demand_forecast": "Demand", "residual_anomaly": "Demand+Solar+Wind",
    "solar_forecast": "Solar", "wind_forecast": "Wind",
}


def station_component_state(entry: dict[str, Any]) -> dict[str, bool]:
    """Honor explicit deselections; infer only for pre-workbook legacy entries."""
    current = entry.get("model_components")
    if isinstance(current, dict):
        return {name: bool(current.get(name, {}).get("enabled", False)) for name in MODEL_IDS}
    renewables = entry.get("renewables") or {}
    return {
        "demand_forecast": bool(entry.get("training_file") or entry.get("history_file") or entry.get("input_file") or entry.get("model_checkpoint")),
        "solar_forecast": bool((renewables.get("solar") or {}).get("train_file") or (renewables.get("solar") or {}).get("data_file") or (renewables.get("solar") or {}).get("checkpoint")),
        "wind_forecast": bool((renewables.get("wind") or {}).get("train_file") or (renewables.get("wind") or {}).get("data_file") or (renewables.get("wind") or {}).get("checkpoint")),
        "residual_anomaly": bool(entry.get("residual_model_checkpoint") or (entry.get("training_file") and (renewables.get('solar') or {}).get('train_file') and (renewables.get('wind') or {}).get('train_file'))),
    }


def registered_source(entry: dict[str, Any], component: str) -> str | None:
    rec = (entry.get("model_components") or {}).get(component) or {}
    if rec.get("training_file"):
        return str(rec["training_file"])
    if component == "demand_forecast":
        # history_file is the long historical training CSV in legacy Amprion;
        # input_file is its tiny replay/test partition and must not be preferred.
        return entry.get("training_file") or entry.get("history_file") or entry.get("input_file")
    if component == "residual_anomaly":
        return (rec.get('training_file') or entry.get('residual_training_file') or entry.get('training_file') or entry.get('history_file'))
    kind = component.split("_")[0]
    spec = (entry.get("renewables") or {}).get(kind) or {}
    return spec.get("train_file") or spec.get("data_file")


def _valid_source_path(root: Path, station: str, component: str, source_id: str) -> Path:
    if not SOURCE_ID_RE.fullmatch(source_id):
        raise HTTPException(status_code=422, detail="Invalid training dataset ID.")
    return root / "data" / "training_uploads" / station / component / source_id


def load_source(root: Path, station: str, component: str, source_id: str) -> dict[str, Any]:
    location = _valid_source_path(root, station, component, source_id)
    manifest_path = location / "manifest.json"
    if not manifest_path.is_file():
        raise HTTPException(status_code=404, detail="Training dataset was not found for this station and model component.")
    try:
        metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=409, detail="Training dataset manifest is unreadable.") from exc
    if metadata.get("station") != station or metadata.get("component") != component or metadata.get("source_id") != source_id:
        raise HTTPException(status_code=409, detail="Training dataset does not match the selected station/component.")
    path = location / "data.csv"
    if not path.is_file():
        raise HTTPException(status_code=409, detail="Training dataset file is missing.")
    # Detect edits/corruption before launching a lengthy training job.
    if digest_file(path) != metadata.get("sha256"):
        raise HTTPException(status_code=409, detail="Uploaded training data has changed since validation.")
    return {**metadata, "path": path.relative_to(root).as_posix()}


def list_sources(root: Path, station: str, component: str) -> list[dict[str, Any]]:
    folder = root / "data" / "training_uploads" / station / component
    if not folder.is_dir():
        return []
    output = []
    for candidate in sorted(folder.iterdir(), reverse=True):
        if not candidate.is_dir() or not SOURCE_ID_RE.fullmatch(candidate.name):
            continue
        try:
            metadata = json.loads((candidate / "manifest.json").read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        if (metadata.get("station"), metadata.get("component"), metadata.get("source_id")) != (station, component, candidate.name):
            continue
        if not (candidate / "data.csv").is_file():
            continue
        # The list view must remain fast with 100-MB files and many experiments.
        # Full SHA-256 verification still runs when the dataset is selected to train.
        output.append({**metadata, "path": (candidate / "data.csv").relative_to(root).as_posix()})
    return sorted(output, key=lambda item: item.get("uploaded_at", ""), reverse=True)


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv_component(path: Path, component: str, interval: int, timestamp_format: str, source_timezone: str) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path, low_memory=False)
    except Exception as exc:
        raise ValueError(f"Could not read the uploaded CSV: {exc}") from exc
    if component == 'residual_anomaly':
        from src.models.residual.residual_ae_common import load_residual_csv
        f=load_residual_csv(path, solar_col='Photovoltaics [MWh]', wind_col='Wind onshore [MWh]').reset_index()
        return f.rename(columns={'timestamp_utc':'utc_timestamp','solar':'Photovoltaics [MWh]',
                        'wind':'Wind onshore [MWh]','solar_capacity_mw':'Photovoltaics_installed_capacity [MW]',
                        'wind_capacity_mw':'Wind_onshore_installed_capacity [MW]','residual':'Residual Load'})
    sheet = COMPONENT_TO_SHEET[component]
    required = (
        ["utc_timestamp", "demand"] + (["temperature"] if component == "demand_forecast" else [])
        if sheet == "Demand" else SOLAR_COLUMNS if sheet == "Solar" else WIND_COLUMNS
    )
    aliases = {"utc_timestamp": TIMESTAMP_ALIASES}
    if sheet == "Demand":
        aliases.update({"demand": DEMAND_ALIASES, "temperature": TEMPERATURE_ALIASES})
    if sheet == "Wind":
        # Support both the new customer template and existing Amprion CSV names.
        aliases.update({
            "future_wind_speed": ("future_wind_speed", "forecast_wind_speed_100m_mean_ms"),
            "future_wind_direction": ("future_wind_direction", "forecast_wind_direction_100m_circular_mean_deg"),
        })
    return _read_sheet({sheet: frame}, sheet, required, timestamp_format, source_timezone, interval, aliases)


def _validate_values(component: str, frame: pd.DataFrame, *, interval: int, entry: dict[str, Any]) -> None:
    """Validate admissible training inputs and train/holdout horizon coverage."""
    if component != "demand_forecast" and interval != 15:
        raise ValueError("Solar, wind and residual-load anomaly training require 15-minute intervals.")
    if component == 'residual_anomaly':
        # benchmark training needs two years in fit and separate validation/calibration.
        n=len(frame)
        if int(n*.80)<2*365*96 or int(n*.90)-int(n*.80)<20*96 or n-int(n*.90)<20*96:
            raise ValueError('Residual-load anomaly requires >=2 fit years and >=20 validation/calibration days')
        return
    if component in {"demand_forecast", "residual_anomaly"}:
        if (frame["demand"] < 0).any():
            raise ValueError("Demand must not contain negative values.")
    elif component == "solar_forecast":
        negative = ["Photovoltaics [MWh]", "shortwave_radiation_mean_prev_hour_w_m2", "forecast_shortwave_radiation_mean_prev_hour_w_m2"]
        bounded = ["cloud_cover_mean_pct", "forecast_cloud_cover_mean_pct"]
        if (frame[negative] < 0).any().any() or ((frame[bounded] < 0) | (frame[bounded] > 100)).any().any():
            raise ValueError("Solar generation/radiation must be nonnegative and cloud cover must be 0–100%.")
        if (frame["Photovoltaics_installed_capacity [MW]"] <= 0).any():
            raise ValueError("Solar installed capacity must be positive.")
    elif component == "wind_forecast":
        negative = ["Wind onshore [MWh]", "wind_speed_100m_mean_ms", "future_wind_speed"]
        directions = ["wind_direction_100m_circular_mean_deg", "future_wind_direction"]
        if (frame[negative] < 0).any().any() or ((frame[directions] < 0) | (frame[directions] > 360)).any().any():
            raise ValueError("Wind generation/speed must be nonnegative and wind directions must be 0–360 degrees.")
        if (frame["Wind_onshore_installed_capacity [MW]"] <= 0).any():
            raise ValueError("Wind installed capacity must be positive.")
    n = len(frame)
    if component == "demand_forecast":
        history = int(entry.get("history_steps") or 672)
        horizon = int(entry.get("horizon_steps") or 96)
        # Allow separate chronological training/validation partitions for 15% holdout.
        if int(n * .85) <= history + horizon + 1 or n - int(n * .85) < horizon:
            raise ValueError(f"Demand needs more continuous history for {history}-step input, {horizon}-step forecast and validation.")
    elif component in {"solar_forecast", "wind_forecast"}:
        # Same 70/15/15 train/validation/test coverage used in train_renewable.
        if int(n * .70) < 672 + 96 + 2 or int(n * .85) - int(n * .70) < 96 or n - int(n * .85) < 96:
            raise ValueError("Renewable training needs enough history for 672 context steps and 96 forecast steps in each chronological split.")


def _canonical_data(component: str, frame: pd.DataFrame) -> pd.DataFrame:
    if component == "residual_anomaly":
        return frame
    return frame


async def store_source(*, root: Path, station: str, component: str, entry: dict[str, Any], file: UploadFile,
                       timestamp_format: str = "auto", source_timezone: str = "UTC") -> dict[str, Any]:
    if component not in MODEL_IDS:
        raise HTTPException(status_code=422, detail="Unknown model component.")
    ext = Path(file.filename or "").suffix.lower()
    if ext not in {".csv", ".xlsx", ".xlsm"}:
        raise HTTPException(status_code=422, detail="Upload a .csv, .xlsx or .xlsm training dataset.")
    interval = int(entry.get("interval_minutes") or 15)
    source_id = uuid.uuid4().hex
    folder = root / "data" / "training_uploads" / station / component
    folder.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".validate_", dir=folder))
    target = folder / source_id
    try:
        upload = staging / f"source{ext}"
        size = 0
        with upload.open("wb") as out:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="Training upload exceeds the 100 MB limit.")
                out.write(chunk)
        if size == 0:
            raise HTTPException(status_code=422, detail="Training upload is empty.")
        try:
            if ext == ".csv":
                frame = _read_csv_component(upload, component, interval, timestamp_format, source_timezone)
            else:
                selection = ['demand_forecast','solar_forecast','wind_forecast'] if component == 'residual_anomaly' else [component]
                data, _ = load_station_workbook(upload, selected_models=selection, interval_minutes=interval,
                                                 timestamp_format=timestamp_format, source_timezone=source_timezone)
                if component == 'residual_anomaly':
                    from src.models.residual.training_data import build_training_frame
                    frame=build_training_frame(data['demand'],data['solar'],data['wind'],entry.get('residual_demand_unit','mwh_per_interval'))
                else:
                    frame=data['demand' if component == 'demand_forecast' else component.split('_')[0]]
            _validate_values(component, frame, interval=interval, entry=entry)
        except (ValueError, KeyError, OSError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        output = _canonical_data(component, frame)
        if "utc_timestamp" in output:
            output = output.copy()
            output["utc_timestamp"] = pd.to_datetime(output["utc_timestamp"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        if "timestamp" in output:
            output = output.copy()
            output["timestamp"] = pd.to_datetime(output["timestamp"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        dataset_path = staging / "data.csv"
        output.to_csv(dataset_path, index=False)
        metadata = {
            "station": station, "component": component, "source_id": source_id,
            "original_filename": Path(file.filename or "upload").name[:255], "format": ext.lstrip("."),
            "rows": int(len(frame)), "interval_minutes": interval,
            "start_utc": frame["utc_timestamp"].iloc[0].isoformat(),
            "end_utc": frame["utc_timestamp"].iloc[-1].isoformat(),
            "source_timezone": source_timezone, "timestamp_format": timestamp_format,
            "original_size_bytes": size, "sha256": digest_file(dataset_path),
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
        }
        (staging / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        staging.rename(target)
        return {**metadata, "path": (target / "data.csv").relative_to(root).as_posix()}
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
