"""Train a station-specific encoder-decoder LSTM compatible with streaming_predictor.

The trainer intentionally uses the same feature names/order and normalization
contract consumed by src.models.streaming_predictor. Future temperature is
held at the last observed value during both training and inference, avoiding
train/serve leakage from unavailable future weather observations.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import yaml

from src.ingestion.file_loader import load_station_csv
from src.features.feature_schema import HISTORY_FEATURES, FUTURE_FEATURES

ROOT = Path(__file__).resolve().parents[2]


class EncoderDecoderLSTM(nn.Module):
    """Architecture expected by the existing streaming predictor."""

    def __init__(self, n_history_features: int, n_future_features: int, hidden_size: int = 32):
        super().__init__()
        self.encoder = nn.LSTM(n_history_features, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(n_future_features, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x_history: torch.Tensor, x_future: torch.Tensor) -> torch.Tensor:
        _, (hidden, cell) = self.encoder(x_history)
        decoded, _ = self.decoder(x_future, (hidden, cell))
        return self.output(decoded).squeeze(-1)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def calendar_features(ts: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    ts = pd.DatetimeIndex(ts)
    minute = ts.hour.to_numpy() * 60 + ts.minute.to_numpy()
    weekday = ts.dayofweek.to_numpy()
    year_day = ts.dayofyear.to_numpy() - 1
    a = 2.0 * np.pi * minute / 1440.0
    b = 2.0 * np.pi * weekday / 7.0
    c = 2.0 * np.pi * year_day / 365.2425
    return {
        "hour_sin": np.sin(a).astype(np.float32),
        "hour_cos": np.cos(a).astype(np.float32),
        "weekday_sin": np.sin(b).astype(np.float32),
        "weekday_cos": np.cos(b).astype(np.float32),
        "year_sin": np.sin(c).astype(np.float32),
        "year_cos": np.cos(c).astype(np.float32),
    }


def rolling_mean_np(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return np.asarray([], dtype=np.float32)
    cumulative = np.concatenate(([0.0], np.cumsum(values)))
    ends = np.arange(1, values.size + 1)
    starts = np.maximum(0, ends - max(1, int(window)))
    counts = ends - starts
    return ((cumulative[ends] - cumulative[starts]) / counts).astype(np.float32)


class WindowDataset(Dataset):
    """Build windows lazily to avoid allocating a huge 3-D training tensor."""

    def __init__(self, indices: np.ndarray, demand_scaled: np.ndarray,
                 history_matrix: np.ndarray, temperatures: np.ndarray,
                 timestamps: pd.DatetimeIndex, temp_mean: float, temp_std: float,
                 history_steps: int, horizon_steps: int, interval_minutes: int):
        self.indices = np.asarray(indices, dtype=np.int64)
        self.demand_scaled = demand_scaled
        self.history_matrix = history_matrix
        self.temperatures = temperatures
        self.timestamps = timestamps
        self.temp_mean = float(temp_mean)
        self.temp_std = float(temp_std)
        self.history_steps = int(history_steps)
        self.horizon_steps = int(horizon_steps)
        self.rolling_1h = max(1, int(round(60 / interval_minutes)))
        self.rolling_24h = max(1, int(round(1440 / interval_minutes)))
        self.context_needed = max(self.rolling_1h, self.rolling_24h) - 1
        self.future_calendar = calendar_features(timestamps)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        start = int(self.indices[item])
        h0 = start - self.history_steps
        h1 = start
        f1 = start + self.horizon_steps

        hist = self.history_matrix[h0:h1]
        last_temp = float(self.temperatures[start - 1])
        future_temp = np.full(self.horizon_steps, last_temp, dtype=np.float32)

        context = self.temperatures[start - self.context_needed:start]
        combined = np.concatenate((context, future_temp))
        roll_1h = rolling_mean_np(combined, self.rolling_1h)[-self.horizon_steps:]
        roll_24h = rolling_mean_np(combined, self.rolling_24h)[-self.horizon_steps:]

        future_map = {
            "future_temperature_scaled": ((future_temp - self.temp_mean) / self.temp_std).astype(np.float32),
            "temperature_rolling_mean_1h_scaled": ((roll_1h - self.temp_mean) / self.temp_std).astype(np.float32),
            "temperature_rolling_mean_24h_scaled": ((roll_24h - self.temp_mean) / self.temp_std).astype(np.float32),
        }
        for name in ("hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "year_sin", "year_cos"):
            future_map[name] = self.future_calendar[name][start:f1]
        future = np.column_stack([future_map[name] for name in FUTURE_FEATURES]).astype(np.float32)
        target = self.demand_scaled[start:f1].astype(np.float32)

        return (
            torch.from_numpy(hist.copy()),
            torch.from_numpy(future),
            torch.from_numpy(target.copy()),
        )


def sample_indices(indices: np.ndarray, maximum: int) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if maximum <= 0 or len(indices) <= maximum:
        return indices
    chosen = np.linspace(0, len(indices) - 1, num=maximum, dtype=np.int64)
    return indices[np.unique(chosen)]


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             demand_mean: float, demand_std: float, criterion: nn.Module) -> tuple[float, float]:
    model.eval()
    losses: list[float] = []
    absolute_errors: list[float] = []
    squared_errors: list[float] = []
    with torch.inference_mode():
        for x_hist, x_future, y in loader:
            x_hist, x_future, y = x_hist.to(device), x_future.to(device), y.to(device)
            pred = model(x_hist, x_future)
            losses.append(float(criterion(pred, y).item()))
            error = (pred - y).detach().cpu().numpy().reshape(-1)
            original_error = error * demand_std
            absolute_errors.extend(np.abs(original_error).tolist())
            squared_errors.extend(np.square(original_error).tolist())
    if not losses:
        return float("nan"), float("nan")
    return float(math.sqrt(float(np.mean(losses)))), float(np.mean(absolute_errors))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a station-specific 24-hour LSTM model.")
    parser.add_argument("--station", required=True)
    parser.add_argument("--csv", default=None, help="Training CSV; defaults to station training_file/input_file")
    parser.add_argument("--output", default=None, help="Checkpoint output override")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--max-train-windows", type=int, default=400)
    parser.add_argument("--max-validation-windows", type=int, default=100)
    parser.add_argument("--patience", type=int, default=2)
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.max_train_windows < 1:
        raise ValueError("epochs, batch-size and max-train-windows must be positive.")
    if not 0.05 <= args.validation_fraction <= 0.4:
        raise ValueError("validation-fraction must be between 0.05 and 0.40.")

    station_cfg = load_yaml(ROOT / "config" / "stations.yaml")
    model_cfg = load_yaml(ROOT / "config" / "model.yaml")
    stations = station_cfg.get("stations", {})
    if args.station not in stations:
        raise ValueError(f"Unknown station '{args.station}'. Configured: {list(stations)}")
    station = stations[args.station]

    csv_path = Path(args.csv or station.get("training_file") or station["input_file"])
    if not csv_path.is_absolute():
        csv_path = ROOT / csv_path
    csv_path = csv_path.resolve()

    checkpoint_path = Path(args.output or station["model_checkpoint"])
    if not checkpoint_path.is_absolute():
        checkpoint_path = ROOT / checkpoint_path
    checkpoint_path = checkpoint_path.resolve()

    history_steps = int(model_cfg.get("history_steps", 672))
    horizon_steps = int(model_cfg.get("horizon_steps", 96))
    interval_minutes = int(model_cfg.get("interval_minutes", 15))
    hidden_size = int(args.hidden_size or model_cfg.get("hidden_size", 32))
    torch_threads = int(model_cfg.get("torch_num_threads", 2))
    torch.set_num_threads(max(1, torch_threads))
    device = torch.device("cpu")  # Local POC is intentionally CPU-safe.

    frame = load_station_csv(csv_path, interval_minutes=interval_minutes)
    timestamps = pd.DatetimeIndex(pd.to_datetime(frame["utc_timestamp"], utc=True))
    demand = frame["demand"].to_numpy(dtype=np.float32)
    temperature = frame["temperature"].to_numpy(dtype=np.float32)
    n_rows = len(frame)

    split_row = int(n_rows * (1.0 - args.validation_fraction))
    if split_row <= history_steps + horizon_steps or n_rows - split_row < horizon_steps:
        raise ValueError(
            f"Not enough rows ({n_rows}) for history={history_steps}, horizon={horizon_steps} "
            "and the selected chronological validation split."
        )

    demand_mean = float(np.mean(demand[:split_row]))
    demand_std = float(np.std(demand[:split_row]))
    temp_mean = float(np.mean(temperature[:split_row]))
    temp_std = float(np.std(temperature[:split_row]))
    demand_std = demand_std if demand_std > 1e-8 else 1.0
    temp_std = temp_std if temp_std > 1e-8 else 1.0

    demand_scaled = ((demand - demand_mean) / demand_std).astype(np.float32)
    temperature_scaled = ((temperature - temp_mean) / temp_std).astype(np.float32)
    hist_calendar = calendar_features(timestamps)
    hist_map = {
        "demand_scaled": demand_scaled,
        "temperature_scaled": temperature_scaled,
        **hist_calendar,
    }
    history_matrix = np.column_stack([hist_map[name] for name in HISTORY_FEATURES]).astype(np.float32)

    # Each index is the first target row; its history ends immediately before it.
    train_indices = np.arange(history_steps, split_row - horizon_steps + 1, dtype=np.int64)
    val_indices = np.arange(split_row, n_rows - horizon_steps + 1, dtype=np.int64)
    train_indices = sample_indices(train_indices, args.max_train_windows)
    val_indices = sample_indices(val_indices, args.max_validation_windows)
    if len(train_indices) < 1 or len(val_indices) < 1:
        raise ValueError("The selected data split produced no training or validation windows.")

    train_ds = WindowDataset(train_indices, demand_scaled, history_matrix, temperature,
                             timestamps, temp_mean, temp_std, history_steps,
                             horizon_steps, interval_minutes)
    val_ds = WindowDataset(val_indices, demand_scaled, history_matrix, temperature,
                           timestamps, temp_mean, temp_std, history_steps,
                           horizon_steps, interval_minutes)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = EncoderDecoderLSTM(len(HISTORY_FEATURES), len(FUTURE_FEATURES), hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()

    best_val_rmse = float("inf")
    best_val_mae_original = float("nan")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    epochs_without_improvement = 0
    history_log: list[dict[str, float]] = []

    print(f"Training station={args.station} | rows={n_rows:,} | train windows={len(train_ds):,} | "
          f"validation windows={len(val_ds):,} | device={device}", flush=True)
    print(f"History={history_steps} | horizon={horizon_steps} | interval={interval_minutes} min | "
          f"features={len(HISTORY_FEATURES)}/{len(FUTURE_FEATURES)} | hidden={hidden_size}", flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        batch_losses: list[float] = []
        for x_hist, x_future, y in train_loader:
            x_hist, x_future, y = x_hist.to(device), x_future.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x_hist, x_future)
            loss = criterion(prediction, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_losses.append(float(loss.item()))

        val_rmse_scaled, val_mae_original = evaluate(
            model, val_loader, device, demand_mean, demand_std, criterion
        )
        train_rmse_scaled = float(math.sqrt(float(np.mean(batch_losses)))) if batch_losses else float("nan")
        history_log.append({
            "epoch": float(epoch),
            "train_rmse_scaled": train_rmse_scaled,
            "validation_rmse_scaled": val_rmse_scaled,
            "validation_mae_original": val_mae_original,
        })
        print(f"Epoch {epoch}/{args.epochs} | train RMSE scaled={train_rmse_scaled:.5f} | "
              f"validation RMSE scaled={val_rmse_scaled:.5f} | "
              f"validation MAE={val_mae_original:.3f}", flush=True)

        if np.isfinite(val_rmse_scaled) and val_rmse_scaled < best_val_rmse:
            best_val_rmse = val_rmse_scaled
            best_val_mae_original = val_mae_original
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print("Early stopping: validation metric did not improve.", flush=True)
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a finite validation metric; checkpoint not saved.")

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": best_state,
        "region_name": args.station,
        "history": history_steps,
        "horizon": horizon_steps,
        "interval_minutes": interval_minutes,
        "hidden_size": hidden_size,
        "history_features": list(HISTORY_FEATURES),
        "future_features": list(FUTURE_FEATURES),
        "scales": {
            args.station: {
                "demand_mean": demand_mean,
                "demand_std": demand_std,
                "temperature_mean": temp_mean,
                "temperature_std": temp_std,
            }
        },
        "weather_mode": "last_observed_temperature_persistence",
        "best_validation_rmse_scaled": best_val_rmse,
        "best_validation_mae_original": best_val_mae_original,
        "best_epoch": best_epoch,
        "training_rows": n_rows,
        "training_windows": len(train_ds),
        "validation_windows": len(val_ds),
    }

    temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(checkpoint_path)

    metadata = {
        "region_name": args.station,
        "model_path": str(checkpoint_path),
        "history": history_steps,
        "horizon": horizon_steps,
        "interval_minutes": interval_minutes,
        "hidden_size": hidden_size,
        "history_features": list(HISTORY_FEATURES),
        "future_features": list(FUTURE_FEATURES),
        "scales": checkpoint["scales"],
        "weather_mode": checkpoint["weather_mode"],
        "best_validation_rmse_scaled": best_val_rmse,
        "best_validation_mae_original": best_val_mae_original,
        "best_epoch": best_epoch,
        "training_csv": str(csv_path),
        "rows": n_rows,
        "training_windows": len(train_ds),
        "validation_windows": len(val_ds),
        "training_history": history_log,
        "note": "Future temperature uses last-observed persistence in both training and inference.",
    }
    metadata_path = checkpoint_path.parent / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"BEST MODEL SAVED: {checkpoint_path}", flush=True)
    print(f"METADATA SAVED: {metadata_path}", flush=True)
    print(f"Best validation RMSE (scaled): {best_val_rmse:.6f} at epoch {best_epoch}", flush=True)


if __name__ == "__main__":
    main()
