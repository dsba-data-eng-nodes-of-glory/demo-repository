from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


REGION_NAME = "amprion"
PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATA_ROOT = PROJECT_ROOT / "data" / "training" / "amprion"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "models" / "amprion"

TIMESTAMP_COLUMN = "utc_timestamp"
TARGET_COLUMN = "demand"
OBSERVED_TEMP_COLUMN = "temperature"
FORECAST_TEMP_COLUMN = "forecast_temperature"

HISTORY_STEPS = 672
HORIZON_STEPS = 96
INTERVAL_MINUTES = 15

# Direct adaptation of the tested solar reference lag list.
DEMAND_LAGS = [1, 4, 8, 16, 24, 48, 96, 192, 288, 384, 480, 576, 672]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_timestamps(values: pd.Series) -> pd.Series:
    raw = values.astype("string").str.strip()
    parsed = pd.to_datetime(
        raw, format="%d-%m-%Y %H:%M", errors="coerce", utc=True
    )
    mask = parsed.isna() & raw.notna() & raw.ne("")
    if mask.any():
        parsed.loc[mask] = pd.to_datetime(
            raw.loc[mask], dayfirst=True, errors="coerce", utc=True
        )
    return parsed


def numeric(series: pd.Series) -> pd.Series:
    raw = series.astype("string").str.strip().str.replace(",", "", regex=False)
    raw = raw.replace({"": pd.NA, "-": pd.NA, "nan": pd.NA, "None": pd.NA})
    return pd.to_numeric(raw, errors="coerce")


def split_points(n_rows: int) -> tuple[int, int]:
    return int(n_rows * 0.70), int(n_rows * 0.85)


def load_data(path: str | Path, interval_minutes: int) -> tuple[pd.DataFrame, dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path.resolve()}")

    frame = pd.read_csv(path, low_memory=False)
    frame.columns = [str(c).strip() for c in frame.columns]

    # Support the short column name used in some prepared files.
    if FORECAST_TEMP_COLUMN not in frame.columns and "forecast_temp" in frame.columns:
        frame = frame.rename(columns={"forecast_temp": FORECAST_TEMP_COLUMN})

    required = [
        TIMESTAMP_COLUMN, TARGET_COLUMN,
        OBSERVED_TEMP_COLUMN, FORECAST_TEMP_COLUMN
    ]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. Found: {list(frame.columns)}"
        )

    source_rows = len(frame)
    frame = frame[required].copy()
    frame[TIMESTAMP_COLUMN] = parse_timestamps(frame[TIMESTAMP_COLUMN])
    invalid_ts = int(frame[TIMESTAMP_COLUMN].isna().sum())

    for col in [TARGET_COLUMN, OBSERVED_TEMP_COLUMN, FORECAST_TEMP_COLUMN]:
        frame[col] = numeric(frame[col])

    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna(subset=[TIMESTAMP_COLUMN])
    if frame.empty:
        raise ValueError("No valid timestamps in input CSV.")

    frame = frame.groupby(TIMESTAMP_COLUMN, as_index=False)[
        [TARGET_COLUMN, OBSERVED_TEMP_COLUMN, FORECAST_TEMP_COLUMN]
    ].mean()
    frame = frame.sort_values(TIMESTAMP_COLUMN)

    ts = pd.DatetimeIndex(frame[TIMESTAMP_COLUMN])
    interval = pd.Timedelta(minutes=interval_minutes)
    offsets = (ts.asi8 - ts.asi8[0]) % interval.value
    if np.any(offsets):
        raise ValueError(
            f"Timestamps are not aligned to {interval_minutes}-minute grid."
        )

    frame = frame.set_index(TIMESTAMP_COLUMN)
    full_index = pd.date_range(
        frame.index.min(), frame.index.max(), freq=interval, tz="UTC"
    )
    frame = frame.reindex(full_index)
    frame.index.name = TIMESTAMP_COLUMN
    frame = frame.reset_index()

    diagnostics = {
        "source_rows": int(source_rows),
        "invalid_timestamps": int(invalid_ts),
        "regular_grid_rows": int(len(frame)),
        "missing_demand": int(frame[TARGET_COLUMN].isna().sum()),
        "missing_observed_temperature": int(frame[OBSERVED_TEMP_COLUMN].isna().sum()),
        "missing_forecast_temperature": int(frame[FORECAST_TEMP_COLUMN].isna().sum()),
    }
    print(
        f"Source rows={source_rows:,}; invalid timestamps={invalid_ts:,}; "
        f"regular-grid rows={len(frame):,}; "
        f"missing demand={diagnostics['missing_demand']:,}; "
        f"missing observed temp={diagnostics['missing_observed_temperature']:,}; "
        f"missing forecast temp={diagnostics['missing_forecast_temperature']:,}"
    )
    print(f"UTC range: {frame[TIMESTAMP_COLUMN].min()} to {frame[TIMESTAMP_COLUMN].max()}")
    return frame, diagnostics


def rolling_stats(values: np.ndarray, windows: list[int]) -> dict[tuple[int, str], np.ndarray]:
    series = pd.Series(np.asarray(values, dtype=np.float64))
    output = {}
    for window in windows:
        roll = series.rolling(window=window, min_periods=1)
        output[(window, "mean")] = roll.mean().to_numpy(dtype=np.float32)
        output[(window, "std")] = roll.std(ddof=0).fillna(0.0).to_numpy(dtype=np.float32)
        output[(window, "max")] = roll.max().to_numpy(dtype=np.float32)
    return output


def calendar_matrix(timestamps: pd.DatetimeIndex) -> tuple[np.ndarray, list[str]]:
    ts = pd.DatetimeIndex(timestamps)
    minute = ts.hour.to_numpy() * 60 + ts.minute.to_numpy()
    weekday = ts.dayofweek.to_numpy()
    day = ts.dayofyear.to_numpy() - 1
    minute_angle = 2 * np.pi * minute / 1440.0
    weekday_angle = 2 * np.pi * weekday / 7.0
    year_angle = 2 * np.pi * day / 365.2425
    values = np.column_stack([
        np.sin(minute_angle), np.cos(minute_angle),
        np.sin(weekday_angle), np.cos(weekday_angle),
        np.sin(year_angle), np.cos(year_angle),
    ]).astype(np.float32)
    names = [
        "origin_hour_sin", "origin_hour_cos",
        "origin_weekday_sin", "origin_weekday_cos",
        "origin_year_sin", "origin_year_cos",
    ]
    return values, names


def build_feature_matrix(
    frame: pd.DataFrame,
    starts: np.ndarray,
    interval_minutes: int = INTERVAL_MINUTES,
    chunk_size: int = 5000,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, dict]:
    """
    Solar-reference-style tabular feature matrix:
    one feature row per forecast origin; one 96-step target vector per origin.

    Origin convention: start s means the forecast is issued after row s-1.
    Target rows are s ... s+95.
    """
    if interval_minutes != 15:
        raise ValueError("This feature alignment currently expects 15-minute data.")
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1.")

    timestamps = pd.DatetimeIndex(frame[TIMESTAMP_COLUMN])
    demand = frame[TARGET_COLUMN].to_numpy(dtype=np.float32)
    observed_temp = frame[OBSERVED_TEMP_COLUMN].to_numpy(dtype=np.float32)
    forecast_temp = frame[FORECAST_TEMP_COLUMN].to_numpy(dtype=np.float32)

    n = len(frame)
    starts = np.asarray(starts, dtype=np.int64)
    horizon_offsets = np.arange(HORIZON_STEPS, dtype=np.int64)
    hourly_offsets = np.arange(0, HORIZON_STEPS, 4, dtype=np.int64)

    # Solar-reference-style rolling features.
    demand_roll4 = pd.Series(demand).rolling(4, min_periods=1).mean().to_numpy(dtype=np.float32)
    demand_stats = rolling_stats(demand, [4, 24, 96, 672])
    observed_temp_stats = rolling_stats(observed_temp, [4, 96, 672])
    forecast_temp_stats = rolling_stats(forecast_temp, [4, 96, 672])

    cal, cal_names = calendar_matrix(timestamps)

    names: list[str] = []

    # 1) Previous 24-hour demand profile and previous week's same 24-hour profile.
    names += [f"demand_previous_24h_hour_{i + 1:02d}" for i in range(24)]
    names += [f"demand_previous_week_same_24h_hour_{i + 1:02d}" for i in range(24)]

    # 2) Explicit demand lags and rolling demand statistics.
    names += [f"demand_lag_{lag}_steps" for lag in DEMAND_LAGS]
    for window, label in [(4, "1h"), (24, "6h"), (96, "24h"), (672, "7d")]:
        names += [f"demand_{stat}_{label}" for stat in ["mean", "std", "max"]]

    # 3) Historical observed-temperature profile and statistics.
    names += [f"hist_observed_temperature_hour_{i + 1:02d}" for i in range(24)]
    for window, label in [(4, "1h"), (96, "24h"), (672, "7d")]:
        names += [
            f"hist_observed_temperature_{stat}_{label}"
            for stat in ["mean", "std"]
        ]

    # 4) Future forecast-temperature profile and horizon statistics.
    names += [f"future_forecast_temperature_hour_{i + 1:02d}" for i in range(24)]
    names += [
        f"future_forecast_temperature_{stat}_24h"
        for stat in ["mean", "std", "min", "max"]
    ]

    # 5) Origin calendar and future hourly calendar profile.
    names += cal_names
    names += [f"future_hour_sin_{i + 1:02d}" for i in range(24)]
    names += [f"future_hour_cos_{i + 1:02d}" for i in range(24)]

    X_parts, Y_parts, S_parts = [], [], []
    skipped_features = 0
    skipped_targets = 0

    for pos in range(0, len(starts), chunk_size):
        s = starts[pos:pos + chunk_size]
        cols: list[np.ndarray] = []

        # Previous-day and previous-week profiles use 1-hour averages
        # represented by the last four 15-minute observations.
        day_profile_idx = s[:, None] - 93 + hourly_offsets[None, :]
        week_profile_idx = s[:, None] - 669 + hourly_offsets[None, :]
        cols.append(demand_roll4[day_profile_idx])
        cols.append(demand_roll4[week_profile_idx])

        # Exact lag alignment from the solar reference.
        lag_matrix = np.column_stack([demand[s - lag] for lag in DEMAND_LAGS])
        cols.append(lag_matrix.astype(np.float32))

        for window in [4, 24, 96, 672]:
            cols.append(np.column_stack([
                demand_stats[(window, stat)][s - 1]
                for stat in ["mean", "std", "max"]
            ]))

        # Historical observed temperature: prior 24h hourly profile
        # and trailing historical rolling statistics.
        hist_temp_idx = s[:, None] - 96 + hourly_offsets[None, :]
        cols.append(observed_temp[hist_temp_idx])
        for window in [4, 96, 672]:
            cols.append(np.column_stack([
                observed_temp_stats[(window, stat)][s - 1]
                for stat in ["mean", "std"]
            ]))

        # Future forecast temperature only: no observed future temperature.
        future_hour_idx = s[:, None] + hourly_offsets[None, :]
        future_temp_profile = forecast_temp[future_hour_idx]
        future_temp_all = forecast_temp[s[:, None] + horizon_offsets[None, :]]
        cols.append(future_temp_profile)
        cols.append(np.column_stack([
            np.mean(future_temp_all, axis=1),
            np.std(future_temp_all, axis=1),
            np.min(future_temp_all, axis=1),
            np.max(future_temp_all, axis=1),
        ]).astype(np.float32))

        # Origin timestamp is the final historical timestamp (s-1).
        cols.append(cal[s - 1])

        future_cal = cal[future_hour_idx]
        cols.append(future_cal[:, :, 0])
        cols.append(future_cal[:, :, 1])

        X_chunk = np.column_stack(cols).astype(np.float32, copy=False)
        Y_chunk = demand[s[:, None] + horizon_offsets[None, :]].astype(np.float32)

        finite_x = np.isfinite(X_chunk).all(axis=1)
        finite_y = np.isfinite(Y_chunk).all(axis=1)
        skipped_features += int((~finite_x).sum())
        skipped_targets += int((~finite_y).sum())
        valid = finite_x & finite_y
        if np.any(valid):
            X_parts.append(X_chunk[valid])
            Y_parts.append(Y_chunk[valid])
            S_parts.append(s[valid])

    if not X_parts:
        raise ValueError("No valid forecast-origin windows were built.")

    X = np.concatenate(X_parts, axis=0)
    Y = np.concatenate(Y_parts, axis=0)
    valid_starts = np.concatenate(S_parts, axis=0)

    if X.shape[1] != len(names):
        raise RuntimeError(
            f"Feature-name mismatch: matrix={X.shape[1]}, names={len(names)}"
        )

    diagnostics = {
        "candidate_origins": int(len(starts)),
        "valid_origins": int(len(X)),
        "skipped_invalid_features": int(skipped_features),
        "skipped_invalid_targets": int(skipped_targets),
    }
    return X, Y, names, valid_starts, diagnostics


def make_origin_splits(n_rows: int, history: int, horizon: int):
    train_end, val_end = split_points(n_rows)
    # Match the solar reference's strict within-partition target windows.
    train_starts = np.arange(history, train_end - horizon + 1, dtype=np.int64)
    val_starts = np.arange(max(history, train_end), val_end - horizon + 1, dtype=np.int64)
    test_starts = np.arange(max(history, val_end), n_rows - horizon + 1, dtype=np.int64)
    return train_starts, val_starts, test_starts


class OriginDataset(Dataset):
    def __init__(self, x: np.ndarray, y_scaled: np.ndarray):
        self.x = torch.from_numpy(np.asarray(x, dtype=np.float32))
        self.y = torch.from_numpy(np.asarray(y_scaled, dtype=np.float32))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


def fit_x_scaler(x_train: np.ndarray):
    mean = np.mean(x_train, axis=0, dtype=np.float64).astype(np.float32)
    std = np.std(x_train, axis=0, dtype=np.float64).astype(np.float32)
    std[~np.isfinite(std) | (std < 1e-6)] = 1.0
    return mean, std


def metrics_scaled(actual: np.ndarray, predicted: np.ndarray):
    error = predicted - actual
    return {
        "RMSE_scaled": float(np.sqrt(np.mean(error ** 2))),
        "MAE_scaled": float(np.mean(np.abs(error))),
    }


def evaluate(model, loader, device):
    model.eval()
    squared, absolute, count = 0.0, 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            diff = pred - y
            squared += float(torch.sum(diff ** 2))
            absolute += float(torch.sum(torch.abs(diff)))
            count += y.numel()
    if count == 0:
        return float("nan"), float("nan")
    return math.sqrt(squared / count), absolute / count


def train(args):
    set_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    print("Training device:", device)
    print("Region:", REGION_NAME)
    print("Feature alignment: SOLAR REFERENCE STYLE")
    print("Weather mode: forecast temperature for future horizon")

    frame, data_diagnostics = load_data(args.input_file, args.interval_minutes)
    n_rows = len(frame)
    train_starts, val_starts, test_starts = make_origin_splits(
        n_rows, args.history, args.horizon
    )

    X_train, Y_train_raw, feature_names, starts_train, diag_train = build_feature_matrix(
        frame, train_starts, args.interval_minutes, args.chunk_size
    )
    X_val, Y_val_raw, names_val, starts_val, diag_val = build_feature_matrix(
        frame, val_starts, args.interval_minutes, args.chunk_size
    )
    X_test, Y_test_raw, names_test, starts_test, diag_test = build_feature_matrix(
        frame, test_starts, args.interval_minutes, args.chunk_size
    )

    if names_val != feature_names or names_test != feature_names:
        raise RuntimeError("Feature order mismatch between data splits.")

    # Fit all scalers on training partition only.
    x_mean, x_std = fit_x_scaler(X_train)
    target_mean = float(np.mean(Y_train_raw, dtype=np.float64))
    target_std = float(np.std(Y_train_raw, dtype=np.float64))
    if not np.isfinite(target_std) or target_std < 1e-8:
        target_std = 1.0

    X_train = ((X_train - x_mean) / x_std).astype(np.float32)
    X_val = ((X_val - x_mean) / x_std).astype(np.float32)
    X_test = ((X_test - x_mean) / x_std).astype(np.float32)
    Y_train = ((Y_train_raw - target_mean) / target_std).astype(np.float32)
    Y_val = ((Y_val_raw - target_mean) / target_std).astype(np.float32)
    Y_test = ((Y_test_raw - target_mean) / target_std).astype(np.float32)

    print(
        f"Origins: train={len(X_train):,}, validation={len(X_val):,}, "
        f"test={len(X_test):,}"
    )
    print(f"Feature count: {X_train.shape[1]}")
    print("Feature groups: demand profiles + 13 lags + demand rolling stats + "
          "historical observed temperature + future forecast temperature + calendar")
    print(
        f"Skipped origins: train={diag_train['skipped_invalid_features'] + diag_train['skipped_invalid_targets']:,}, "
        f"validation={diag_val['skipped_invalid_features'] + diag_val['skipped_invalid_targets']:,}, "
        f"test={diag_test['skipped_invalid_features'] + diag_test['skipped_invalid_targets']:,}"
    )

    train_ds = OriginDataset(X_train, Y_train)
    val_ds = OriginDataset(X_val, Y_val)
    test_ds = OriginDataset(X_test, Y_test)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    train_eval_loader = DataLoader(
        train_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=0
    )

    model = DenseMultiOutput(
        input_size=X_train.shape[1],
        horizon=args.horizon,
        hidden_1=args.hidden_1,
        hidden_2=args.hidden_2,
        hidden_3=args.hidden_3,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-5
    )

    best_val = float("inf")
    best_state = None
    best_train_eval = None
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = nn.functional.mse_loss(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        train_rmse, _ = evaluate(model, train_eval_loader, device)
        val_rmse, _ = evaluate(model, val_loader, device)
        scheduler.step(val_rmse)
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train-eval RMSE scaled={train_rmse:.5f} | "
            f"validation RMSE scaled={val_rmse:.5f} | lr={lr:.2e}"
        )

        if val_rmse < best_val:
            best_val = val_rmse
            best_train_eval = train_rmse
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_rmse, test_mae = evaluate(model, test_loader, device)
    print(f"Test RMSE scaled={test_rmse:.5f}")
    print(f"Test MAE scaled={test_mae:.5f}")
    print(f"Test RMSE in demand units={test_rmse * target_std:.3f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / f"{REGION_NAME}_dense_solar_aligned_forecast.pt"
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_kind": "dense_multioutput_solar_reference_aligned",
        "region_name": REGION_NAME,
        "history": args.history,
        "horizon": args.horizon,
        "interval_minutes": args.interval_minutes,
        "feature_names": feature_names,
        "feature_mean": x_mean,
        "feature_std": x_std,
        "target_mean": target_mean,
        "target_std": target_std,
        "architecture": {
            "input_size": int(X_train.shape[1]),
            "hidden_1": args.hidden_1,
            "hidden_2": args.hidden_2,
            "hidden_3": args.hidden_3,
            "dropout": args.dropout,
        },
        "weather_mode": "forecast_temperature_for_future_horizon",
        "demand_lags": DEMAND_LAGS,
    }
    torch.save(checkpoint, model_path)

    metadata = {
        "input_file": str(Path(args.input_file).resolve()),
        "model_path": str(model_path),
        "region": REGION_NAME,
        "history_steps": args.history,
        "horizon_steps": args.horizon,
        "interval_minutes": args.interval_minutes,
        "feature_count": int(X_train.shape[1]),
        "feature_names": feature_names,
        "feature_groups": {
            "demand_profiles": 48,
            "demand_lags": DEMAND_LAGS,
            "demand_rolling_windows_steps": [4, 24, 96, 672],
            "historical_observed_temperature_profile_steps": 24,
            "historical_observed_temperature_rolling_windows_steps": [4, 96, 672],
            "future_forecast_temperature_profile_steps": 24,
            "future_forecast_temperature_statistics": ["mean", "std", "min", "max"],
            "calendar": cal_names_for_metadata(),
        },
        "target_scaling": {"mean": target_mean, "std": target_std},
        "best_validation_rmse_scaled": best_val,
        "train_rmse_at_best_validation_scaled": best_train_eval,
        "test_rmse_scaled": test_rmse,
        "test_mae_scaled": test_mae,
        "train_origins": len(train_ds),
        "validation_origins": len(val_ds),
        "test_origins": len(test_ds),
        "data_diagnostics": data_diagnostics,
        "split_diagnostics": {
            "train": diag_train, "validation": diag_val, "test": diag_test
        },
        "note": (
            "Feature alignment follows the attached tested solar feature-builder "
            "pattern. Historical demand and observed temperature are used only "
            "at or before forecast origin. Future temperature inputs use "
            "forecast_temperature only. Target is next 96 demand values."
        ),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print("Saved model:", model_path)
    print("Saved metadata:", output_dir / "metadata.json")


def cal_names_for_metadata():
    return [
        "origin_hour_sin", "origin_hour_cos",
        "origin_weekday_sin", "origin_weekday_cos",
        "origin_year_sin", "origin_year_cos",
        "future_hour_sin_01..24", "future_hour_cos_01..24",
    ]


class DenseMultiOutput(nn.Module):
    def __init__(self, input_size, horizon, hidden_1, hidden_2, hidden_3, dropout):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_2, hidden_3),
            nn.ReLU(),
            nn.Linear(hidden_3, horizon),
        )

    def forward(self, x):
        return self.network(x)


def main():
    parser = argparse.ArgumentParser(
        description="Amprion demand DNN with solar-reference-aligned features."
    )
    parser.add_argument(
        "--input-file", type=Path,
        default=DATA_ROOT / "demand_train.csv"
    )
    parser.add_argument("--interval-minutes", type=int, default=15)
    parser.add_argument("--history", type=int, default=HISTORY_STEPS)
    parser.add_argument("--horizon", type=int, default=HORIZON_STEPS)
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--hidden-1", type=int, default=256)
    parser.add_argument("--hidden-2", type=int, default=128)
    parser.add_argument("--hidden-3", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--output-dir", type=Path,
        default=DEFAULT_OUTPUT_ROOT / "demand_dense_nn"
    )
    args = parser.parse_args()
    if args.batch_size <= 0 or args.eval_batch_size <= 0:
        raise ValueError("Batch sizes must be positive.")
    if args.epochs <= 0 or args.patience <= 0:
        raise ValueError("Epochs and patience must be positive.")
    train(args)


if __name__ == "__main__":
    main()
