"""Inference adapter for the supplied Amprion 672-to-96 Dense demand model."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.models.forecast_weather import load_weather_lookup, select_forecast_horizon
from src.models.training.amprion import Dense_NN_Forecast_Training as features


class DenseMultiOutput(nn.Module):
    def __init__(self, input_size: int, horizon: int, hidden_1: int, hidden_2: int, hidden_3: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_1, hidden_2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_2, hidden_3), nn.ReLU(), nn.Linear(hidden_3, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def load_demand_dense_model(checkpoint_path: Path, station_id: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if str(checkpoint.get("region_name", "")).lower() != station_id.lower():
        raise ValueError(f"Demand checkpoint region {checkpoint.get('region_name')} does not match {station_id}")
    history = int(checkpoint["history"]); horizon = int(checkpoint["horizon"])
    interval = int(checkpoint["interval_minutes"])
    if (history, horizon, interval) != (672, 96, 15):
        raise ValueError(f"Unsupported Dense demand contract {(history, horizon, interval)}; expected (672, 96, 15)")
    architecture = checkpoint["architecture"]
    model = DenseMultiOutput(
        int(architecture["input_size"]), horizon,
        int(architecture["hidden_1"]), int(architecture["hidden_2"]),
        int(architecture["hidden_3"]), float(architecture["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    if len(checkpoint.get("feature_names", [])) != int(architecture["input_size"]):
        raise ValueError("Demand model feature-name count does not match checkpoint input size")
    return model, checkpoint


def build_demand_feature_row(
    history_rows: list[dict[str, Any]],
    future_timestamps: pd.DatetimeIndex,
    forecast_weather_file: str | Path,
    checkpoint: dict,
) -> np.ndarray:
    history_steps = int(checkpoint["history"])
    horizon = int(checkpoint["horizon"])
    interval_minutes = int(checkpoint["interval_minutes"])
    if len(history_rows) != history_steps:
        raise ValueError(f"Dense demand model needs {history_steps} historical rows")
    timestamps = pd.DatetimeIndex(pd.to_datetime([r["utc_timestamp"] for r in history_rows], utc=True))
    interval = pd.Timedelta(minutes=interval_minutes)
    if len(timestamps) > 1 and not np.all(np.diff(timestamps.asi8) == interval.value):
        raise ValueError("Demand history is not a continuous 15-minute sequence")
    future_timestamps = pd.DatetimeIndex(pd.to_datetime(future_timestamps, utc=True))
    expected = pd.date_range(timestamps[-1] + interval, periods=horizon, freq=interval, tz="UTC")
    if not future_timestamps.equals(expected):
        raise ValueError("Requested demand forecast timestamps do not follow the history origin")

    weather = load_weather_lookup(forecast_weather_file, "utc_timestamp", ["forecast_temperature"])
    future_weather = select_forecast_horizon(weather, future_timestamps, ["forecast_temperature"], context="Amprion demand")
    history = pd.DataFrame({
        "utc_timestamp": timestamps,
        "demand": [float(r["demand"]) for r in history_rows],
        "temperature": [float(r["temperature"]) for r in history_rows],
        "forecast_temperature": np.nan,
    })
    future = pd.DataFrame({
        "utc_timestamp": future_timestamps,
        "demand": np.zeros(horizon, dtype=np.float32),  # target placeholder is never used as an input feature
        "temperature": np.zeros(horizon, dtype=np.float32),  # future observed temperature is deliberately not used
        "forecast_temperature": pd.to_numeric(future_weather["forecast_temperature"], errors="raise").to_numpy(dtype=np.float32),
    })
    model_frame = pd.concat([history, future], ignore_index=True)
    X, _Y, names, _valid_starts, diagnostics = features.build_feature_matrix(
        model_frame, np.asarray([history_steps], dtype=np.int64), interval_minutes=interval_minutes,
    )
    expected_names = list(checkpoint["feature_names"])
    if names != expected_names:
        raise ValueError("Demand serving feature order differs from the supplied training checkpoint")
    if X.shape != (1, len(expected_names)):
        raise ValueError(f"Demand feature builder returned unexpected shape {X.shape}")
    if diagnostics.get("valid_origins") != 1 or not np.isfinite(X).all():
        raise ValueError(f"Demand feature row is invalid: {diagnostics}")
    return X.astype(np.float32)


def predict_demand(
    rows: list[dict[str, Any]],
    model: nn.Module,
    checkpoint: dict,
    device: torch.device,
    forecast_weather_file: str | Path,
    future_timestamps: pd.DatetimeIndex,
) -> np.ndarray:
    X = build_demand_feature_row(rows, future_timestamps, forecast_weather_file, checkpoint)
    mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    scale = np.asarray(checkpoint["feature_std"], dtype=np.float32)
    if mean.shape != (X.shape[1],) or scale.shape != mean.shape:
        raise ValueError("Demand checkpoint feature scaler shape does not match feature row")
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)
    X_scaled = (X - mean) / scale
    with torch.inference_mode():
        prediction_scaled = model(torch.from_numpy(X_scaled).to(device)).cpu().numpy().reshape(-1)
    if len(prediction_scaled) != len(future_timestamps) or not np.isfinite(prediction_scaled).all():
        raise RuntimeError("Demand Dense model returned invalid forecast values")
    prediction = prediction_scaled * float(checkpoint["target_std"]) + float(checkpoint["target_mean"])
    return np.maximum(prediction.astype(float), 0.0)
