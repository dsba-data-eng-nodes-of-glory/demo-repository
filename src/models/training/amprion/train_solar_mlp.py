from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from .solar_mlp_features import (
    TIMESTAMP_COLUMN,
    TARGET_COLUMN,
    CAPACITY_COLUMN,
    FORECAST_CLOUD_COLUMN,
    FORECAST_RADIATION_COLUMN,
    HISTORY_STEPS,
    HORIZON_STEPS,
    INTERVAL_MINUTES,
    load_solar_csv,
    split_points,
    build_feature_matrix,
    metric_dict,
    )
    from .solar_mlp_model import SolarMultiOutputMLP
except ImportError:
    from solar_mlp_features import (
        TIMESTAMP_COLUMN, TARGET_COLUMN, CAPACITY_COLUMN, FORECAST_CLOUD_COLUMN,
        FORECAST_RADIATION_COLUMN, HISTORY_STEPS, HORIZON_STEPS, INTERVAL_MINUTES,
        load_solar_csv, split_points, build_feature_matrix, metric_dict,
    )
    from solar_mlp_model import SolarMultiOutputMLP

PACKAGE_DIR = Path(__file__).resolve().parents[4]
DATA_DIR = PACKAGE_DIR / "data" / "training" / "amprion"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=float), encoding="utf-8")


def fit_feature_scaler(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit per-column standardization using training origins only."""
    mean = np.mean(x_train, axis=0, dtype=np.float64).astype(np.float32)
    scale = np.std(x_train, axis=0, dtype=np.float64).astype(np.float32)
    scale[~np.isfinite(scale) | (scale < 1e-6)] = 1.0
    mean[~np.isfinite(mean)] = 0.0
    return mean, scale


def apply_feature_scaler_inplace(x: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    if x.ndim != 2 or x.shape[1] != len(mean):
        raise ValueError(f"Feature matrix shape {x.shape} does not match scaler width {len(mean)}")
    x -= mean[None, :]
    x /= scale[None, :]
    np.nan_to_num(x, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return np.ascontiguousarray(x, dtype=np.float32)


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    xt = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
    yt = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))
    return DataLoader(
        TensorDataset(xt, yt),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )


def predict_scaled(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    parts: list[np.ndarray] = []
    with torch.inference_mode():
        for xb, _yb in loader:
            xb = xb.to(device)
            pred = model(xb)
            parts.append(pred.cpu().numpy().astype(np.float32, copy=False))
    if not parts:
        return np.empty((0, HORIZON_STEPS), dtype=np.float32)
    return np.concatenate(parts, axis=0)


def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_mean: float,
    target_std: float,
) -> tuple[float, float]:
    model.eval()
    squared_sum = 0.0
    value_count = 0
    abs_sum = 0.0
    with torch.inference_mode():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            err = pred - yb
            squared_sum += float(torch.sum(err * err).item())
            abs_sum += float(torch.sum(torch.abs(err)).item())
            value_count += yb.numel()
    if value_count == 0:
        raise ValueError("Validation loader contains no values")
    rmse_scaled = math.sqrt(squared_sum / value_count)
    mae_mwh = (abs_sum / value_count) * target_std
    return rmse_scaled, mae_mwh


def atomic_torch_save(payload: dict, path: Path) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp_path)
    temp_path.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train direct multi-output MLP for Amprion-region solar PV forecasting."
    )
    parser.add_argument("--csv", type=Path, default=DATA_DIR / "solar_train.csv")
    parser.add_argument("--capacity-column", type=str, default=CAPACITY_COLUMN)
    parser.add_argument("--output-dir", type=Path, default=PACKAGE_DIR / "models" / "amprion" / "solar_dense_nn")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--hidden-1", type=int, default=256)
    parser.add_argument("--hidden-2", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-solar-position", action="store_true")
    parser.add_argument("--latitude", type=float, default=50.966111)
    parser.add_argument("--longitude", type=float, default=6.805556)
    parser.add_argument("--clip-negative", action="store_true")
    args = parser.parse_args()

    if min(args.epochs, args.batch_size, args.hidden_1, args.hidden_2,
           args.patience, args.torch_threads, args.chunk_size, args.eval_batch_size) <= 0:
        parser.error("epochs, batch sizes, hidden sizes, patience, threads and chunk size must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0 or args.min_delta < 0:
        parser.error("learning rate must be positive; weight decay and min delta must be nonnegative")
    if not 0 <= args.dropout < 1:
        parser.error("dropout must be in [0, 1)")
    if not -90 <= args.latitude <= 90 or not -180 <= args.longitude <= 180:
        parser.error("Latitude/longitude are outside valid ranges")

    set_seed(args.seed)
    torch.set_num_threads(args.torch_threads)
    try:
        torch.set_num_interop_threads(max(1, min(args.torch_threads, 2)))
    except RuntimeError:
        pass
    device = torch.device("cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78, flush=True)
    print("SOLAR PV V4 — MULTI-OUTPUT MLP", flush=True)
    print("All forecast origins retained: every 15 minutes (stride=1).", flush=True)
    print("One row per origin predicts all 96 future 15-minute values.", flush=True)
    print("Weather mode: forecasted future cloud cover and shortwave radiation.", flush=True)
    print("Historical weather features remain based on observed weather.", flush=True)
    print(f"CPU threads={args.torch_threads} | batch={args.batch_size} | hidden={args.hidden_1},{args.hidden_2}", flush=True)
    if args.include_solar_position:
        print(f"Solar position enabled at {args.latitude:.6f}, {args.longitude:.6f}", flush=True)
    print("=" * 78, flush=True)

    total_started = time.perf_counter()
    frame, data_diagnostics = load_solar_csv(args.csv, INTERVAL_MINUTES, args.capacity_column)
    n_rows = len(frame)
    train_end, val_end = split_points(n_rows)
    if train_end <= HISTORY_STEPS + HORIZON_STEPS or val_end - train_end < HORIZON_STEPS or n_rows - val_end < HORIZON_STEPS:
        raise ValueError("Not enough rows for chronological 70/15/15 split")

    target = frame[TARGET_COLUMN].to_numpy(dtype=np.float64)
    train_target_raw = target[:train_end]
    train_target_raw = train_target_raw[np.isfinite(train_target_raw)]
    if not len(train_target_raw):
        raise ValueError("No finite training target observations")
    target_mean = float(train_target_raw.mean())
    target_std = float(train_target_raw.std())
    if not np.isfinite(target_std) or target_std < 1e-8:
        target_std = 1.0

    # Exact V2 chronological boundaries and every possible 15-minute origin.
    train_starts = np.arange(HISTORY_STEPS, train_end - HORIZON_STEPS + 1, dtype=np.int64)
    val_starts = np.arange(max(HISTORY_STEPS, train_end), val_end - HORIZON_STEPS + 1, dtype=np.int64)
    test_starts = np.arange(max(HISTORY_STEPS, val_end), n_rows - HORIZON_STEPS + 1, dtype=np.int64)
    print(f"Rows={n_rows:,} | train={train_end:,} | validation={val_end-train_end:,} | test={n_rows-val_end:,}", flush=True)
    print(f"Candidate origins: train={len(train_starts):,} | validation={len(val_starts):,} | test={len(test_starts):,}", flush=True)
    print(f"History={HISTORY_STEPS} | horizon={HORIZON_STEPS} | interval={INTERVAL_MINUTES} min | stride=1", flush=True)

    prep_started = time.perf_counter()
    print("Building training feature matrix...", flush=True)
    X_train, y_train, _baseline_train, feature_names, d_train, starts_train = build_feature_matrix(
        frame, train_starts, capacity_column=args.capacity_column,
        include_solar_position=args.include_solar_position,
        latitude=args.latitude, longitude=args.longitude,
        chunk_size=args.chunk_size, target_mean=target_mean, target_std=target_std,
    )
    print(f"Train matrix={X_train.shape}; diagnostics={d_train}", flush=True)

    print("Building validation feature matrix...", flush=True)
    X_val, y_val, _baseline_val, val_feature_names, d_val, starts_val = build_feature_matrix(
        frame, val_starts, capacity_column=args.capacity_column,
        include_solar_position=args.include_solar_position,
        latitude=args.latitude, longitude=args.longitude,
        chunk_size=args.chunk_size, target_mean=target_mean, target_std=target_std,
    )
    if feature_names != val_feature_names:
        raise RuntimeError("Train and validation feature orders differ")
    print(f"Validation matrix={X_val.shape}; diagnostics={d_val}", flush=True)

    x_mean, x_scale = fit_feature_scaler(X_train)
    apply_feature_scaler_inplace(X_train, x_mean, x_scale)
    apply_feature_scaler_inplace(X_val, x_mean, x_scale)
    prep_seconds = time.perf_counter() - prep_started
    print(f"Feature preparation + scaling: {prep_seconds/60:.1f} min", flush=True)

    train_loader = make_loader(X_train, y_train, args.batch_size, shuffle=True)
    val_loader = make_loader(X_val, y_val, args.eval_batch_size, shuffle=False)
    del X_train, y_train, X_val, y_val

    model = SolarMultiOutputMLP(
        input_size=len(feature_names), output_size=HORIZON_STEPS,
        hidden_1=args.hidden_1, hidden_2=args.hidden_2, dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-5
    )
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_epoch = 0
    bad_epochs = 0
    history_rows: list[dict] = []
    checkpoint_path = args.output_dir / "solar_pv_mlp_v4.pt"
    working_checkpoint_path = args.output_dir / "solar_pv_mlp_v4_best_working.pt"
    # Do not accidentally reuse a checkpoint from an earlier run. Keep the prior
    # final checkpoint untouched until this run has produced its own best epoch.
    working_checkpoint_path.unlink(missing_ok=True)
    history_path = args.output_dir / "solar_mlp_v4_training_history.csv"
    epoch_started_total = time.perf_counter()

    try:
        for epoch in range(1, args.epochs + 1):
            epoch_started = time.perf_counter()
            model.train()
            loss_sum = 0.0
            values_seen = 0
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                optimizer.zero_grad(set_to_none=True)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                loss_sum += float(loss.detach().item()) * yb.numel()
                values_seen += yb.numel()

            train_rmse = math.sqrt(loss_sum / max(values_seen, 1))
            val_rmse, val_mae = evaluate_epoch(model, val_loader, device, target_mean, target_std)
            scheduler.step(val_rmse)
            current_lr = float(optimizer.param_groups[0]["lr"])
            elapsed = time.perf_counter() - epoch_started
            history_rows.append({
                "epoch": epoch, "train_rmse_scaled": train_rmse,
                "validation_rmse_scaled": val_rmse, "validation_mae_mwh": val_mae,
                "learning_rate": current_lr, "duration_seconds": elapsed,
            })
            pd.DataFrame(history_rows).to_csv(history_path, index=False)
            print(
                f"Epoch {epoch:02d}/{args.epochs} | train RMSE scaled={train_rmse:.5f} "
                f"| validation RMSE scaled={val_rmse:.5f} | validation MAE={val_mae:.3f} MWh "
                f"| lr={current_lr:.2e} | duration={elapsed/60:.1f} min",
                flush=True,
            )

            improved = val_rmse < best_val - args.min_delta
            if improved:
                best_val = val_rmse
                best_epoch = epoch
                bad_epochs = 0
                checkpoint = {
                    "algorithm": "Multi-output feed-forward MLP",
                    "model_state_dict": copy.deepcopy(model.state_dict()),
                    "input_size": len(feature_names),
                    "output_size": HORIZON_STEPS,
                    "hidden_1": args.hidden_1,
                    "hidden_2": args.hidden_2,
                    "dropout": args.dropout,
                    "x_mean": x_mean,
                    "x_scale": x_scale,
                    "target_mean": target_mean,
                    "target_std": target_std,
                    "feature_names": feature_names,
                    "history_steps": HISTORY_STEPS,
                    "horizon_steps": HORIZON_STEPS,
                    "interval_minutes": INTERVAL_MINUTES,
                    "best_epoch": best_epoch,
                    "best_validation_rmse_scaled": best_val,
                    "include_solar_position": bool(args.include_solar_position),
                    "latitude": args.latitude if args.include_solar_position else None,
                    "longitude": args.longitude if args.include_solar_position else None,
                }
                atomic_torch_save(checkpoint, working_checkpoint_path)
                print(f"  Best checkpoint saved at epoch {epoch}.", flush=True)
            else:
                bad_epochs += 1
                if bad_epochs >= args.patience:
                    print(f"Early stopping after {epoch} epochs; best epoch={best_epoch}.", flush=True)
                    break
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. The best checkpoint from completed epochs is preserved.", flush=True)

    if not working_checkpoint_path.exists():
        raise RuntimeError("No best checkpoint was saved in this run. Training may have been interrupted before validation completed.")

    print("Loading best checkpoint for test inference...", flush=True)
    # Checkpoint is created by this script; weights_only=False is needed for NumPy scaler arrays on newer PyTorch.
    try:
        best_checkpoint = torch.load(working_checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        best_checkpoint = torch.load(working_checkpoint_path, map_location="cpu")
    # Promote this run's best checkpoint only after at least one completed epoch.
    working_checkpoint_path.replace(checkpoint_path)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    model.eval()

    print("Building test feature matrix (all 15-minute origins)...", flush=True)
    X_test, y_test_scaled, baseline_test, test_feature_names, d_test, starts_test = build_feature_matrix(
        frame, test_starts, capacity_column=args.capacity_column,
        include_solar_position=args.include_solar_position,
        latitude=args.latitude, longitude=args.longitude,
        chunk_size=args.chunk_size, target_mean=target_mean, target_std=target_std,
    )
    if feature_names != test_feature_names:
        raise RuntimeError("Train and test feature orders differ")
    apply_feature_scaler_inplace(X_test, x_mean, x_scale)
    test_loader = make_loader(X_test, y_test_scaled, args.eval_batch_size, shuffle=False)
    del X_test
    pred_scaled = predict_scaled(model, test_loader, device)
    predicted = pred_scaled * target_std + target_mean
    actual = y_test_scaled * target_std + target_mean
    if args.clip_negative:
        predicted = np.maximum(predicted, 0.0)
    model_metrics = metric_dict(actual, predicted)
    baseline_metrics = metric_dict(actual, baseline_test)
    print("TEST MLP | " + " | ".join(f"{k}={v:.4f}" for k, v in model_metrics.items()), flush=True)
    print("TEST WEEKLY BASELINE | " + " | ".join(f"{k}={v:.4f}" for k, v in baseline_metrics.items()), flush=True)

    timestamps = pd.DatetimeIndex(frame[TIMESTAMP_COLUMN])
    origin_ns = timestamps.asi8[starts_test - 1].astype(np.int64)
    prediction_path = args.output_dir / "solar_mlp_v4_test_predictions.npz"
    np.savez_compressed(
        prediction_path,
        forecast_origin_ns=origin_ns,
        actual_mwh=actual.astype(np.float32),
        predicted_mwh=predicted.astype(np.float32),
        weekly_baseline_mwh=baseline_test.astype(np.float32),
        lead_minutes=(np.arange(1, HORIZON_STEPS + 1, dtype=np.int16) * INTERVAL_MINUTES),
    )

    metrics = {
        "model": model_metrics,
        "weekly_baseline": baseline_metrics,
        "model_mae_improvement_vs_weekly_baseline_pct": (
            (baseline_metrics["MAE_MWh"] - model_metrics["MAE_MWh"]) / baseline_metrics["MAE_MWh"] * 100
            if baseline_metrics["MAE_MWh"] else None
        ),
        "best_epoch": best_epoch,
        "best_validation_rmse_scaled": best_val,
        "training_history": history_rows,
        "feature_preparation_seconds": prep_seconds,
        "training_loop_seconds": time.perf_counter() - epoch_started_total,
        "feature_count": len(feature_names),
        "output_dimensions": HORIZON_STEPS,
        "train_origins": int(len(starts_train)),
        "validation_origins": int(len(starts_val)),
        "test_origins": int(len(starts_test)),
        "train_valid_origins": int(len(starts_train)),
        "validation_valid_origins": int(len(starts_val)),
        "test_valid_origins": int(len(starts_test)),
        "training_diagnostics": d_train,
        "validation_diagnostics": d_val,
        "test_diagnostics": d_test,
    }
    metrics_path = args.output_dir / "solar_mlp_v4_metrics.json"
    save_json(metrics_path, metrics)

    train_q90 = float(np.quantile(train_target_raw, 0.90))
    train_ramp = np.diff(train_target_raw)
    ramp_q90 = float(np.quantile(np.abs(train_ramp[np.isfinite(train_ramp)]), 0.90)) if len(train_ramp) else 0.0
    metadata = {
        "algorithm": "Multi-output feed-forward MLP",
        "target_column": TARGET_COLUMN,
        "capacity_column": args.capacity_column,
        "interval_minutes": INTERVAL_MINUTES,
        "history_steps": HISTORY_STEPS,
        "horizon_steps": HORIZON_STEPS,
        "split_ratios": [0.70, 0.15, 0.15],
        "train_split_rows": int(train_end),
        "validation_split_end_row": int(val_end),
        "train_origins": int(len(starts_train)),
        "validation_origins": int(len(starts_val)),
        "test_origins": int(len(starts_test)),
        "all_origins_stride": 1,
        "target_scaling": {"mean": target_mean, "std": target_std},
        "feature_names": feature_names,
        "feature_count": len(feature_names),
        "weather_mode": "forecast",
        "historical_weather_columns": ["cloud_cover_mean_pct", "shortwave_radiation_mean_prev_hour_w_m2"],
        "future_weather_columns": [FORECAST_CLOUD_COLUMN, FORECAST_RADIATION_COLUMN],
        "future_weather_note": (
            "Future cloud-cover and shortwave-radiation features are built only from the "
            "forecast columns. Historical weather features continue to use observed weather. "
            "For strict rolling-origin backtesting, forecast values must correspond to the "
            "forecast vintage available at each origin; a single value per valid timestamp "
            "does not preserve multiple forecast vintages."
        ),
        "include_solar_position": bool(args.include_solar_position),
        "latitude": args.latitude if args.include_solar_position else None,
        "longitude": args.longitude if args.include_solar_position else None,
        "training_parameters": {
            "epochs_requested": args.epochs, "batch_size": args.batch_size,
            "hidden_1": args.hidden_1, "hidden_2": args.hidden_2,
            "dropout": args.dropout, "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay, "patience": args.patience,
            "torch_threads": args.torch_threads, "chunk_size": args.chunk_size,
            "eval_batch_size": args.eval_batch_size,
        },
        "data_diagnostics": data_diagnostics,
        "training_history": history_rows,
        "best_epoch": best_epoch,
        "best_validation_rmse_scaled": best_val,
        "peak_threshold_mwh": train_q90,
        "ramp_abs_q90_mwh": ramp_q90,
        "model_metrics": model_metrics,
        "weekly_baseline_metrics": baseline_metrics,
        "feature_preparation_seconds": prep_seconds,
        "total_runtime_seconds": time.perf_counter() - total_started,
    }
    metadata_path = args.output_dir / "solar_mlp_v4_metadata.json"
    save_json(metadata_path, metadata)

    print("=" * 78, flush=True)
    print(f"BEST CHECKPOINT: {checkpoint_path}", flush=True)
    print(f"TEST PREDICTIONS: {prediction_path}", flush=True)
    print(f"METRICS: {metrics_path}", flush=True)
    print(f"METADATA: {metadata_path}", flush=True)
    print(f"Best epoch={best_epoch}; validation RMSE scaled={best_val:.6f}", flush=True)
    print(f"Origins train/validation/test={len(starts_train):,}/{len(starts_val):,}/{len(starts_test):,}", flush=True)
    print("Finished.", flush=True)


if __name__ == "__main__":
    main()
