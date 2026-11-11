"""Component-scoped renewable experiments using the original solar/wind feature builders.

The three data splits are chronological. Fit scalers only on the training
period. Future weather inputs come exclusively from uploaded forecast columns.
One forecast origin has exactly 672 past observations and 96 predicted steps.
"""
from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.model_registry import next_version, utc_now
from src.models.training.amprion import solar_mlp_features as solar
from src.models.training.amprion import solar_mlp_model
from src.models.training.amprion import Amprion_Wind_Dense as wind

ROOT = Path(__file__).resolve().parents[2]
HISTORY, HORIZON, INTERVAL = 672, 96, 15


def sample_origins(origins: np.ndarray, max_count: int) -> np.ndarray:
    """Deterministic chronological sampling to protect 8-GB laptop RAM."""
    if max_count < 1:
        raise ValueError("max_count must be positive")
    if len(origins) <= max_count:
        return origins
    return origins[np.unique(np.linspace(0, len(origins) - 1, max_count).astype(np.int64))]


def load_frame(kind: str, path: Path) -> pd.DataFrame:
    if kind == "solar":
        return solar.load_solar_csv(path, INTERVAL, solar.CAPACITY_COLUMN)[0]
    if kind == "wind":
        return wind.load_wind_data(path, INTERVAL)[0]
    raise ValueError(f"Unsupported renewable component: {kind}")


def split_origins(n_rows: int, validation_fraction: float, *, max_train: int, max_validation: int):
    test_fraction = 0.15
    train_end = int(n_rows * (1 - validation_fraction - test_fraction))
    val_end = int(n_rows * (1 - test_fraction))
    train = np.arange(HISTORY, train_end - HORIZON + 1, dtype=np.int64)
    valid = np.arange(max(train_end, HISTORY), val_end - HORIZON + 1, dtype=np.int64)
    test = np.arange(max(val_end, HISTORY), n_rows - HORIZON + 1, dtype=np.int64)
    if len(train) < 2 or len(valid) < 1 or len(test) < 1:
        raise ValueError("Dataset is too short for non-overlapping 70/15/15-style chronological train/validation/test targets.")
    return (sample_origins(train, max_train), sample_origins(valid, max_validation),
            sample_origins(test, max_validation), train_end, val_end)


def feature_matrix(kind: str, frame: pd.DataFrame, starts: np.ndarray, target_mean: float, target_std: float):
    if kind == "solar":
        X, y_scaled, _base, names, diag, valid_starts = solar.build_feature_matrix(
            frame, starts, chunk_size=256, target_mean=target_mean, target_std=target_std,
        )
        Y = y_scaled * target_std + target_mean
    else:
        X, Y, names, valid_starts, diag = wind.build_feature_matrix(
            frame, starts, interval_minutes=INTERVAL, chunk_size=256,
        )
    return np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.float32), names, valid_starts, diag


class RenewableLSTM(nn.Module):
    """True temporal encoder followed by a known-future-weather decoder."""
    def __init__(self, history_features: int, future_features: int, hidden_size: int):
        super().__init__()
        self.encoder = nn.LSTM(history_features, hidden_size, batch_first=True)
        self.decoder = nn.Sequential(nn.Linear(hidden_size + future_features, hidden_size),
                                     nn.ReLU(), nn.Linear(hidden_size, 1))

    def forward(self, history: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.encoder(history)
        context = hidden[-1].unsqueeze(1).expand(-1, future.shape[1], -1)
        return self.decoder(torch.cat((context, future), dim=-1)).squeeze(-1)


def sequence_arrays(kind: str, frame: pd.DataFrame, starts: np.ndarray):
    """No future observed target or observed future weather enters these arrays."""
    hidx = starts[:, None] - HISTORY + np.arange(HISTORY)[None, :]
    fidx = starts[:, None] + np.arange(HORIZON)[None, :]
    if kind == "solar":
        hist_cols = [solar.TARGET_COLUMN, solar.CLOUD_COLUMN, solar.RADIATION_COLUMN, solar.CAPACITY_COLUMN]
        future_cols = [solar.FORECAST_CLOUD_COLUMN, solar.FORECAST_RADIATION_COLUMN, solar.CAPACITY_COLUMN]
        hist = frame[hist_cols].to_numpy(dtype=np.float32)[hidx]
        future = frame[future_cols].to_numpy(dtype=np.float32)[fidx]
        target = frame[solar.TARGET_COLUMN].to_numpy(dtype=np.float32)[fidx]
    else:
        target_col = wind.TARGET_COLUMN
        obs = frame[[target_col, wind.WIND_SPEED_COLUMN, wind.CAPACITY_COLUMN]].to_numpy(dtype=np.float32)
        direction = np.deg2rad(frame[wind.WIND_DIRECTION_COLUMN].to_numpy(dtype=np.float32))
        hist = np.concatenate([obs[hidx], np.sin(direction[hidx])[..., None],
                               np.cos(direction[hidx])[..., None]], axis=-1)
        future_weather = frame[["_future_wind_speed", wind.CAPACITY_COLUMN]].to_numpy(dtype=np.float32)[fidx]
        future_direction = np.deg2rad(frame["_future_wind_direction"].to_numpy(dtype=np.float32))[fidx]
        future = np.concatenate([future_weather, np.sin(future_direction)[..., None],
                                 np.cos(future_direction)[..., None]], axis=-1)
        target = frame[target_col].to_numpy(dtype=np.float32)[fidx]
    # Known calendar features at target times are causal.
    ts = pd.DatetimeIndex(frame["utc_timestamp"])
    minutes = (ts.hour * 60 + ts.minute).to_numpy(dtype=np.float32)[fidx]
    cyc = 2 * np.pi * minutes / (24 * 60)
    future = np.concatenate([future, np.sin(cyc)[..., None], np.cos(cyc)[..., None]], axis=-1)
    if not (np.isfinite(hist).all() and np.isfinite(future).all() and np.isfinite(target).all()):
        raise ValueError("Missing values in renewable temporal features or labels")
    return hist.astype(np.float32), future.astype(np.float32), target.astype(np.float32)


def normalize_sequences(hist: np.ndarray, future: np.ndarray, stats: dict[str, Any] | None = None):
    if stats is None:
        h_mean = hist.mean(axis=(0, 1))
        h_scale = hist.std(axis=(0, 1))
        f_mean = future.mean(axis=(0, 1))
        f_scale = future.std(axis=(0, 1))
        h_scale = np.where(h_scale > 1e-6, h_scale, 1.)
        f_scale = np.where(f_scale > 1e-6, f_scale, 1.)
        stats = {
            "h_mean": h_mean.tolist(), "h_scale": h_scale.tolist(),
            "f_mean": f_mean.tolist(), "f_scale": f_scale.tolist(),
        }
    h_norm = (hist - np.asarray(stats["h_mean"], dtype=np.float32)) / np.asarray(stats["h_scale"], dtype=np.float32)
    f_norm = (future - np.asarray(stats["f_mean"], dtype=np.float32)) / np.asarray(stats["f_scale"], dtype=np.float32)
    return h_norm.astype(np.float32), f_norm.astype(np.float32), stats


def _fit_torch(model, x_train, y_train, x_val, y_val, params: dict, *, sequence: bool):
    threads = max(1, min(4, int(params.get("num_threads", 2))))
    torch.set_num_threads(threads)
    torch.manual_seed(int(params.get("random_state", 42)))
    bs = int(params.get("batch_size", 8 if sequence else 128))
    if sequence:
        train_loader = DataLoader(TensorDataset(*(torch.from_numpy(a) for a in (*x_train, y_train))),
                                  batch_size=bs, shuffle=True)
        val_inputs = [torch.from_numpy(a) for a in x_val]
    else:
        train_loader = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
                                  batch_size=bs, shuffle=True)
        val_inputs = [torch.from_numpy(x_val)]
    val_labels = torch.from_numpy(y_val)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(params.get("learning_rate", 0.001)),
                                   weight_decay=float(params.get("weight_decay", 1e-5)))
    best = math.inf
    best_weights = None
    best_epoch = 0
    wait = 0
    for epoch in range(1, int(params.get("epochs", 20)) + 1):
        model.train()
        for batch in train_loader:
            *features, target = batch
            optimizer.zero_grad(set_to_none=True)
            forecast = model(*features) if sequence else model(features[0])
            loss = nn.functional.mse_loss(forecast, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            val_pred = model(*val_inputs) if sequence else model(val_inputs[0])
            error = torch.mean((val_pred - val_labels).square()).item()
        print(f"Epoch={epoch} val_mse_scaled={error:.6f}", flush=True)
        if error < best - 1e-7:
            best, best_epoch = error, epoch
            best_weights = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= int(params.get("patience", 4)):
                break
    if best_weights is None:
        raise RuntimeError("No finite validation loss was produced")
    model.load_state_dict(best_weights)
    model.eval()
    return model, best_epoch


def _make_tree_model(algorithm: str, params: dict):
    if algorithm == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=int(params["n_estimators"]), max_depth=int(params["max_depth"]),
                                     min_samples_leaf=int(params["min_samples_leaf"]), n_jobs=int(params["n_jobs"]),
                                     random_state=int(params["random_state"]))
    if algorithm == "gbt":
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(max_iter=int(params["max_iter"]),
                                              max_leaf_nodes=int(params["max_leaf_nodes"]),
                                              learning_rate=float(params["learning_rate"]),
                                              l2_regularization=float(params["l2_regularization"]),
                                              random_state=int(params["random_state"]))
    if algorithm == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=int(params["n_estimators"]), max_depth=int(params["max_depth"]),
                            learning_rate=float(params["learning_rate"]), subsample=float(params["subsample"]),
                            colsample_bytree=float(params["colsample_bytree"]),
                            n_jobs=int(params["n_jobs"]), random_state=int(params["random_state"]),
                            tree_method="hist", objective="reg:squarederror")
    raise ValueError(f"Unsupported renewable tree algorithm: {algorithm}")


def expand_origin_features(X: np.ndarray, *, horizon: int = HORIZON) -> np.ndarray:
    """One learner across all lead steps, rather than 96 separate estimators."""
    lead = np.tile(np.arange(1, horizon + 1, dtype=np.float32), len(X))
    angle = (lead - 1) / horizon * 2 * np.pi
    return np.column_stack((np.repeat(X, horizon, axis=0), lead / horizon,
                            np.sin(angle), np.cos(angle))).astype(np.float32)


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    actual = np.asarray(actual, dtype=np.float64).reshape(-1)
    predicted = np.asarray(predicted, dtype=np.float64).reshape(-1)
    err = predicted - actual
    return {
        "mae_mwh": float(np.mean(np.abs(err))),
        "rmse_mwh": float(np.sqrt(np.mean(err ** 2))),
        "wape_pct": float(100 * np.sum(np.abs(err)) / max(1e-8, np.sum(np.abs(actual)))),
        "bias_mwh": float(np.mean(err)),
        "matched_observations": int(len(actual)),
    }


def run_training(config: dict[str, Any]) -> dict[str, Any]:
    import yaml
    started_at = time.perf_counter()
    station = str(config["station"])
    component = str(config["component"])
    kind = component.split("_")[0]
    algorithm = str(config["algorithm"])
    params = dict(config["parameters"])
    options = dict(config.get("training_options") or {})
    if algorithm not in {"dense_nn", "lstm", "random_forest", "gbt", "xgboost"}:
        raise ValueError(f"Unsupported renewable algorithm: {algorithm}")
    registry = yaml.safe_load((ROOT / "config" / "stations.yaml").read_text(encoding="utf-8"))
    entry = registry["stations"][station]
    if int(entry.get("interval_minutes", INTERVAL)) != INTERVAL:
        raise ValueError("Current renewable models require 15-minute observations")
    from backend.training_data import registered_source
    file_value = config.get("training_data_path") or registered_source(entry, component)
    if not file_value:
        raise FileNotFoundError(f"No {component} historical dataset is configured for {station}")
    data_path = Path(file_value)
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    frame = load_frame(kind, data_path)
    target_col = solar.TARGET_COLUMN if kind == "solar" else wind.TARGET_COLUMN
    all_targets = frame[target_col].to_numpy(dtype=np.float64)
    count = len(frame)
    train_starts, valid_starts, test_starts, train_end, val_end = split_origins(
        count, float(options.get("validation_fraction", 0.15)),
        max_train=int(params.get("max_train_origins", 256)),
        max_validation=int(params.get("max_validation_origins", 64)),
    )
    target_mean = float(np.mean(all_targets[:train_end]))
    target_std = float(np.std(all_targets[:train_end]))
    if target_std < 1e-6:
        target_std = 1.0
    exp = str(config["experiment_id"])
    artifact_dir = ROOT / "models" / station / "experiments" / exp
    artifact_dir.mkdir(parents=True, exist_ok=True)
    extra: dict[str, Any] = {}

    if algorithm == "lstm":
        htrain, ftrain, y_train = sequence_arrays(kind, frame, train_starts)
        hvalid, fvalid, y_valid = sequence_arrays(kind, frame, valid_starts)
        htest, ftest, y_test = sequence_arrays(kind, frame, test_starts)
        htrain, ftrain, seq_stats = normalize_sequences(htrain, ftrain)
        hvalid, fvalid, _ = normalize_sequences(hvalid, fvalid, seq_stats)
        htest, ftest, _ = normalize_sequences(htest, ftest, seq_stats)
        y_train_s = (y_train - target_mean) / target_std
        y_valid_s = (y_valid - target_mean) / target_std
        model = RenewableLSTM(htrain.shape[-1], ftrain.shape[-1], int(params["hidden_size"]))
        model, best_epoch = _fit_torch(model, (htrain, ftrain), y_train_s.astype(np.float32),
                                       (hvalid, fvalid), y_valid_s.astype(np.float32), params, sequence=True)
        with torch.inference_mode():
            pred = model(torch.from_numpy(htest), torch.from_numpy(ftest)).numpy() * target_std + target_mean
        artifact_path = artifact_dir / "model.pt"
        torch.save({
            "model_kind": "renewable_lstm_v1", "kind": kind, "model_state_dict": model.state_dict(),
            "history_features": htrain.shape[-1], "future_features": ftrain.shape[-1],
            "hidden_size": int(params["hidden_size"]), "seq_stats": seq_stats,
            "target_mean": target_mean, "target_std": target_std,
            "history": HISTORY, "horizon": HORIZON, "interval_minutes": INTERVAL,
        }, artifact_path)
        extra = {"best_epoch": best_epoch, "feature_schema": "renewable_sequence_lstm_v1"}
    else:
        Xtrain, y_train, names, train_origins, dtrain = feature_matrix(kind, frame, train_starts, target_mean, target_std)
        Xvalid, y_valid, valid_names, val_origins, dvalid = feature_matrix(kind, frame, valid_starts, target_mean, target_std)
        Xtest, y_test, test_names, test_origins, dtest = feature_matrix(kind, frame, test_starts, target_mean, target_std)
        if names != valid_names or names != test_names:
            raise RuntimeError("Renewable feature orders differ across chronological splits")
        if algorithm == "dense_nn":
            x_mean = Xtrain.mean(axis=0)
            x_scale = Xtrain.std(axis=0)
            x_scale = np.where(x_scale > 1e-6, x_scale, 1.0)
            Xt = ((Xtrain - x_mean) / x_scale).astype(np.float32)
            Xv = ((Xvalid - x_mean) / x_scale).astype(np.float32)
            Xe = ((Xtest - x_mean) / x_scale).astype(np.float32)
            if kind == "solar":
                model = solar_mlp_model.SolarMultiOutputMLP(
                    input_size=len(names), output_size=HORIZON,
                    hidden_1=int(params["hidden_1"]), hidden_2=int(params["hidden_2"]), dropout=float(params["dropout"]),
                )
            else:
                model = wind.DenseMultiOutput(
                    input_size=len(names), horizon=HORIZON, hidden_1=int(params["hidden_1"]),
                    hidden_2=int(params["hidden_2"]), hidden_3=int(params["hidden_3"]), dropout=float(params["dropout"]),
                )
            model, best_epoch = _fit_torch(model, Xt, ((y_train - target_mean) / target_std).astype(np.float32),
                                           Xv, ((y_valid - target_mean) / target_std).astype(np.float32),
                                           params, sequence=False)
            with torch.inference_mode():
                pred = model(torch.from_numpy(Xe)).numpy() * target_std + target_mean
            artifact_path = artifact_dir / "model.pt"
            checkpoint = {
                "model_state_dict": model.state_dict(), "target_mean": target_mean,
                "target_std": target_std, "feature_names": names, "history": HISTORY,
                "horizon": HORIZON, "interval_minutes": INTERVAL,
                "model_kind": f"renewable_{kind}_dense_v1", "best_epoch": best_epoch,
            }
            if kind == "solar":
                checkpoint.update({"input_size": len(names), "output_size": HORIZON,
                                   "hidden_1": int(params["hidden_1"]), "hidden_2": int(params["hidden_2"]),
                                   "dropout": float(params["dropout"]), "x_mean": x_mean, "x_scale": x_scale,
                                   "include_solar_position": False, "capacity_column": solar.CAPACITY_COLUMN})
            else:
                checkpoint.update({"feature_mean": x_mean, "feature_std": x_scale,
                                   "future_capacity_mode": "known",
                                   "architecture": {"input_size": len(names), "hidden_1": int(params["hidden_1"]),
                                                    "hidden_2": int(params["hidden_2"]), "hidden_3": int(params["hidden_3"]),
                                                    "dropout": float(params["dropout"])}})
            torch.save(checkpoint, artifact_path)
            extra = {"best_epoch": best_epoch, "feature_schema": f"renewable_{kind}_engineered_dense_v1"}
        else:
            model = _make_tree_model(algorithm, params)
            model.fit(expand_origin_features(Xtrain), y_train.reshape(-1))
            pred = model.predict(expand_origin_features(Xtest)).reshape(-1, HORIZON)
            artifact_path = artifact_dir / "model.joblib"
            joblib.dump({"estimator": model, "kind": kind, "feature_names": names,
                         "target_mean": target_mean, "target_std": target_std,
                         "history": HISTORY, "horizon": HORIZON, "interval_minutes": INTERVAL,
                         "algorithm": algorithm}, artifact_path, compress=3)
            extra = {"feature_schema": f"renewable_{kind}_engineered_stepwise_v1"}
        extra["feature_count"] = len(names)
        extra["feature_names"] = names
        extra["feature_diagnostics"] = {"train": dtrain, "validation": dvalid, "test": dtest}
    pred = np.maximum(np.asarray(pred, dtype=np.float64), 0.0)
    if pred.shape != np.asarray(y_test).shape:
        raise ValueError(f"Predicted matrix shape {pred.shape} differs from test {np.asarray(y_test).shape}")
    version = next_version(station, algorithm, component)
    metadata_path = artifact_dir / "metadata.json"
    unit_metrics = metrics(y_test, pred)
    metadata = {
        "station": station, "component": component, "algorithm": algorithm, "experiment_id": exp,
        "status": "candidate", "version": version,
        "artifact_path": artifact_path.relative_to(ROOT).as_posix(),
        "metadata_path": metadata_path.relative_to(ROOT).as_posix(),
        "history_steps": HISTORY, "horizon_steps": HORIZON, "interval_minutes": INTERVAL,
        "training_rows": int(train_end), "validation_rows": int(val_end-train_end),
        "test_rows": int(count-val_end), "training_origins": int(len(train_starts)),
        "validation_origin_count": int(len(valid_starts)), "test_origin_count": int(len(test_starts)),
        "training_start_utc": pd.Timestamp(frame.utc_timestamp.iloc[0]).isoformat(),
        "training_end_utc": pd.Timestamp(frame.utc_timestamp.iloc[train_end-1]).isoformat(),
        "validation_start_utc": pd.Timestamp(frame.utc_timestamp.iloc[train_end]).isoformat(),
        "validation_end_utc": pd.Timestamp(frame.utc_timestamp.iloc[val_end-1]).isoformat(),
        "test_start_utc": pd.Timestamp(frame.utc_timestamp.iloc[val_end]).isoformat(),
        "test_end_utc": pd.Timestamp(frame.utc_timestamp.iloc[-1]).isoformat(),
        "evaluation_protocol": "chronological_train_validation_test_sampled_origins_v1",
        "metrics": unit_metrics,
        "parameters": params, "forecast_weather_mode": "forecast_weather_columns_by_valid_timestamp",
        "weather_vintage_warning": (
            "For operational backtests forecast weather must be the vintage available at each forecast origin. "
            "A single forecast value per valid timestamp cannot reconstruct all historical forecast vintages."
        ),
        "model_size_bytes": int(artifact_path.stat().st_size),
        "training_duration_seconds": time.perf_counter() - started_at,
        "created_at": utc_now(), **extra,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    return metadata
