"""Experiment launcher used by the Model Management API."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from src.models.model_registry import next_version, upsert_experiment, utc_now
from src.models.train_tabular import run_training

ROOT = Path(__file__).resolve().parents[2]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp.replace(path)


def run_lstm(config: dict[str, Any]) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("The LSTM algorithm requires PyTorch. Install the project requirements and retry.") from exc
    station = str(config["station"])
    experiment_id = str(config["experiment_id"])
    params = dict(config.get("parameters") or {})
    training_options = dict(config.get("training_options") or {})
    artifact_dir = ROOT / "models" / station / "experiments" / experiment_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = artifact_dir / "model.pt"

    # Keep YAML parsing here so the runner can be used independently of FastAPI.
    import yaml
    station_doc = yaml.safe_load((ROOT / "config" / "stations.yaml").read_text(encoding="utf-8")) or {}
    station_entry = station_doc.get("stations", {}).get(station)
    if not station_entry:
        raise ValueError(f"Unknown station '{station}'.")
    csv_value = station_entry.get("training_file") or station_entry.get("input_file")
    if not csv_value:
        raise ValueError(f"Station '{station}' has no training CSV configured.")
    csv_path = Path(csv_value)
    if not csv_path.is_absolute():
        csv_path = ROOT / csv_path

    command = [
        sys.executable, "-u", "-m", "src.models.train",
        "--station", station,
        "--csv", str(csv_path.resolve()),
        "--output", str(checkpoint_path.resolve()),
        "--epochs", str(int(params.get("epochs", 6))),
        "--batch-size", str(int(params.get("batch_size", 2))),
        "--hidden-size", str(int(params.get("hidden_size", 32))),
        "--learning-rate", str(float(params.get("learning_rate", 0.001))),
        "--validation-fraction", str(float(training_options.get("validation_fraction", 0.15))),
        "--max-train-windows", str(int(params.get("max_train_windows", 400))),
        "--max-validation-windows", str(int(params.get("max_validation_windows", 100))),
        "--patience", str(int(params.get("patience", 2))),
    ]
    subprocess.run(command, cwd=ROOT, check=True)

    metadata_path = artifact_dir / "metadata.json"
    metadata = _read_json(metadata_path)
    version = next_version(station, "lstm")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint.update({
        "algorithm": "lstm",
        "experiment_id": experiment_id,
        "model_version": version,
    })
    temp_checkpoint = checkpoint_path.with_suffix(".pt.tmp")
    torch.save(checkpoint, temp_checkpoint)
    temp_checkpoint.replace(checkpoint_path)

    metrics = metadata.get("validation_metrics") or {}
    validation_start = metadata["validation_start_utc"]
    validation_end = metadata["validation_end_utc"]
    training_start = metadata["training_start_utc"]
    training_end = metadata["training_end_utc"]

    metadata.update({
        "experiment_id": experiment_id,
        "algorithm": "lstm",
        "version": version,
        "artifact_path": checkpoint_path.relative_to(ROOT).as_posix(),
        "metadata_path": metadata_path.relative_to(ROOT).as_posix(),
        "feature_schema": "lstm_seq2seq_v1",
        "parameters": params,
        "metrics": metrics,
        "training_start_utc": training_start,
        "training_end_utc": training_end,
        "validation_start_utc": validation_start,
        "validation_end_utc": validation_end,
        "continuous_segment_count": metadata.get("continuous_segment_count"),
        "evaluation_protocol": metadata.get(
            "evaluation_protocol", "chronological_holdout_sampled_origins_v1"
        ),
        "validation_fraction": float(training_options.get("validation_fraction", 0.15)),
        "model_size_bytes": int(checkpoint_path.stat().st_size),
        "created_at": utc_now(),
        "weather_mode": "last_observed_temperature_persistence",
    })
    _write_json(metadata_path, metadata)

    return {
        "experiment_id": experiment_id,
        "station": station,
        "algorithm": "lstm",
        "version": version,
        "status": "candidate",
        "artifact_path": metadata["artifact_path"],
        "metadata_path": metadata["metadata_path"],
        "feature_schema": metadata["feature_schema"],
        "history_steps": metadata["history"],
        "horizon_steps": metadata["horizon"],
        "interval_minutes": metadata["interval_minutes"],
        "validation_start_utc": validation_start,
        "validation_end_utc": validation_end,
        "validation_origin_count": int(metadata.get("validation_windows", 0)),
        "evaluation_protocol": metadata["evaluation_protocol"],
        "parameters": params,
        "metrics": metrics,
        "training_duration_seconds": metadata.get("training_duration_seconds"),
        "inference_latency_ms_per_origin": metadata.get("inference_latency_ms_per_origin"),
        "model_size_bytes": int(checkpoint_path.stat().st_size),
        "created_at": metadata["created_at"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Experiment JSON configuration path")
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config_path = config_path.resolve()
    config = _read_json(config_path)
    started = time.perf_counter()
    experiment_id = str(config["experiment_id"])
    station = str(config["station"])
    algorithm = str(config["algorithm"])
    upsert_experiment({
        "experiment_id": experiment_id,
        "station": station,
        "algorithm": algorithm,
        "status": "running",
        "parameters": config.get("parameters", {}),
        "config_path": config_path.relative_to(ROOT).as_posix() if config_path.is_relative_to(ROOT) else str(config_path),
        "created_at": config.get("created_at", utc_now()),
    })
    try:
        if algorithm == "lstm":
            record = run_lstm(config)
        else:
            record = run_training(config)
        # Use wall-clock experiment execution time for cross-algorithm comparison.
        # Algorithm-specific fit time remains available in model metadata where measured.
        record["training_duration_seconds"] = time.perf_counter() - started
        upsert_experiment(record)
        print(f"EXPERIMENT COMPLETED: {experiment_id}", flush=True)
    except Exception as exc:
        existing = {
            "experiment_id": experiment_id,
            "station": station,
            "algorithm": algorithm,
            "status": "failed",
            "error": str(exc),
            "traceback": traceback.format_exc()[-8000:],
            "training_duration_seconds": time.perf_counter() - started,
        }
        upsert_experiment(existing)
        raise


if __name__ == "__main__":
    main()
