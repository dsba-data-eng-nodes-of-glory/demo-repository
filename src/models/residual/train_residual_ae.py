"""Fit 15-feature residual-load Dense Autoencoder with two-stage thresholds.

Splits are strictly chronological: fit (80%), early-stop validation (10%),
calibration (10%). The latter is not used to fit model weights or scaler.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, TensorDataset

from src.models.residual.residual_ae_common import (
    AE_GROUPS, FEATURE_NAMES, FEATURE_WEIGHTS, WINDOW_STEPS,
    extract_windows, fit_baseline, fit_statistical_reference,
    load_residual_csv, make_features, robust_location_scale,
    statistical_scores, valid_window_starts,
)
from src.models.residual.residual_ae_inference import ae_group_errors, ae_score_from_errors
from src.models.residual.residual_ae_model import DenseAutoencoder


def weighted_loss(reconstructed: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    weights = torch.tensor(FEATURE_WEIGHTS, dtype=target.dtype, device=target.device)
    squared = (reconstructed - target).square().reshape(-1, WINDOW_STEPS, len(FEATURE_NAMES))
    return (squared * weights.view(1, 1, -1)).sum(dim=2).mean() / weights.sum()


def main():
    p = argparse.ArgumentParser(description="Residual load: statistical + Dense AE")
    p.add_argument("--input", default="residual_load_train.csv")
    p.add_argument("--output-dir", default="residual_ae_model")
    p.add_argument("--timezone", default="Europe/Berlin")
    p.add_argument("--train-fraction", type=float, default=.80)
    p.add_argument("--validation-fraction", type=float, default=.10)
    p.add_argument("--train-stride", type=int, default=24)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--bottleneck-dim", type=int, default=32)
    p.add_argument("--stat-quantile", type=float, default=.99)
    p.add_argument("--ae-quantile", type=float, default=.99)
    p.add_argument("--screen-training-z", type=float, default=0.0,
                   help="Optional very conservative train-window contamination screen; 0 disables")
    p.add_argument("--num-threads", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if not (.6 <= args.train_fraction <= .9 and .05 <= args.validation_fraction <= .2 and
            args.train_fraction + args.validation_fraction <= .95):
        p.error("Invalid chronological train/validation split")
    if min(args.train_stride, args.batch_size, args.epochs, args.patience, args.num_threads) < 1:
        p.error("Stride/batch/epochs/patience/threads must all be >= 1")
    if not (.90 <= args.stat_quantile < 1 and .90 <= args.ae_quantile < 1):
        p.error("Threshold quantiles must be between 0.90 and 1 (exclusive)")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.num_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Customer workbook uses canonical [MWh] headers; benchmark exports may
    # use their shorter legacy column names. Accept either without changing
    # the meaning or unit of any feature.
    headers = set(pd.read_csv(args.input, nrows=0).columns)
    solar_header = 'Photovoltaics [MWh]' if 'Photovoltaics [MWh]' in headers else 'Photovoltaics'
    wind_header = 'Wind onshore [MWh]' if 'Wind onshore [MWh]' in headers else 'Wind onshore'
    df = load_residual_csv(args.input, solar_col=solar_header, wind_col=wind_header)
    n = len(df)
    fit_end = int(n * args.train_fraction)
    cal_start = int(n * (args.train_fraction + args.validation_fraction))
    if fit_end < 2 * 365 * 96 or cal_start - fit_end < 96 * 20 or n - cal_start < 96 * 20:
        raise ValueError("Dataset too short for robust trend + chronological validation/calibration")
    print(f"Rows={n:,}, fit=[0:{fit_end:,}], early-stop=[{fit_end:,}:{cal_start:,}], calibration=[{cal_start:,}:{n:,}]")
    baseline = fit_baseline(df["residual"].iloc[:fit_end], timezone=args.timezone)
    features, expected = make_features(df, baseline)
    raw = features.to_numpy(dtype=float)
    fit_valid = np.isfinite(raw[:fit_end]).all(axis=1)
    scaler = RobustScaler(quantile_range=(25, 75)).fit(raw[:fit_end][fit_valid])
    scaled = np.full(raw.shape, np.nan, dtype=np.float32)
    all_valid = np.isfinite(raw).all(axis=1)
    scaled[all_valid] = scaler.transform(raw[all_valid]).astype(np.float32)

    stat_reference = fit_statistical_reference(features.iloc[:fit_end])
    stat_scores, _ = statistical_scores(features, stat_reference)
    fit_starts = valid_window_starts(
        scaled, np.arange(0, fit_end - WINDOW_STEPS + 1, args.train_stride))
    early_starts = valid_window_starts(
        scaled, np.arange(fit_end, cal_start - WINDOW_STEPS + 1, args.train_stride))
    screened = 0
    if args.screen_training_z > 0:
        bad = (~np.isfinite(stat_scores) | (stat_scores > args.screen_training_z)).astype(np.int8)
        cs = np.concatenate(([0], np.cumsum(bad)))
        clean = fit_starts[(cs[fit_starts + WINDOW_STEPS] - cs[fit_starts]) == 0]
        if len(clean) >= .8 * len(fit_starts):
            screened = len(fit_starts) - len(clean)
            fit_starts = clean
        else:
            print("WARNING: training screen would exclude >20% windows, disabling")
    if len(fit_starts) < 100 or len(early_starts) < 50:
        raise ValueError(f"Insufficient valid fit/early-stop windows: {len(fit_starts)}/{len(early_starts)}")
    print(f"Training windows={len(fit_starts):,}, early-stop windows={len(early_starts):,}, screened={screened:,}")
    x_train = extract_windows(scaled, fit_starts)
    x_early = extract_windows(scaled, early_starts)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(x_train)),
                              batch_size=args.batch_size, shuffle=True)
    early_loader = DataLoader(TensorDataset(torch.from_numpy(x_early)),
                              batch_size=args.batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenseAutoencoder(WINDOW_STEPS * len(FEATURE_NAMES),
                             args.hidden_dim, args.bottleneck_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    best_loss, best_epoch, wait, best_state = float("inf"), 0, 0, None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = []
        for (xb,) in train_loader:
            xb = xb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = weighted_loss(model(xb), xb)
            loss.backward()
            optimizer.step()
            totals.append(float(loss.detach().cpu()))
        model.eval()
        vals = []
        with torch.inference_mode():
            for (xb,) in early_loader:
                xb = xb.to(device)
                vals.append(float(weighted_loss(model(xb), xb).cpu()))
        train_loss, val_loss = float(np.mean(totals)), float(np.mean(vals))
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss})
        print(f"Epoch {epoch:03d}: train={train_loss:.6f} val={val_loss:.6f}")
        if val_loss < best_loss - 1e-6:
            best_loss, best_epoch, wait = val_loss, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= args.patience:
                print(f"Early stopping; best epoch={best_epoch}")
                break
    if best_state is None:
        raise RuntimeError("Training did not complete")
    model.load_state_dict(best_state)
    model.eval()

    # Calibrate at each 15-minute timestamp in the last chronological segment.
    cal_positions = np.arange(cal_start, n)
    cal_starts = valid_window_starts(scaled, cal_positions - (WINDOW_STEPS - 1))
    if len(cal_starts) < 96 * 10:
        raise ValueError("Not enough continuous calibration windows")
    cal_end = cal_starts + WINDOW_STEPS - 1
    cal_stat = stat_scores[cal_end]
    stat_threshold = float(np.quantile(cal_stat[np.isfinite(cal_stat)], args.stat_quantile))
    cal_errors = ae_group_errors(model, scaled, cal_starts, device, args.batch_size)
    ae_reference = {k: robust_location_scale(cal_errors[k]) for k in AE_GROUPS}
    ae_scores, ae_z = ae_score_from_errors(cal_errors, ae_reference)
    ae_threshold = float(np.quantile(ae_scores[np.isfinite(ae_scores)], args.ae_quantile))
    s_flag, a_flag = cal_stat > stat_threshold, ae_scores > ae_threshold
    confirmed = s_flag & a_flag
    bundle = {
        "schema": "residual_ae_v2_capacity", "feature_names": FEATURE_NAMES,
        "window_steps": WINDOW_STEPS, "timezone": args.timezone,
        "source_units": "generation MWh/15 min; installed capacity MW; residual original energy units",
        "model_state_dict": best_state, "hidden_dim": args.hidden_dim,
        "bottleneck_dim": args.bottleneck_dim, "scaler": scaler,
        "baseline": baseline, "stat_reference": stat_reference,
        "stat_threshold": stat_threshold, "ae_reference": ae_reference,
        "ae_threshold": ae_threshold, "decision_rule": "both",
        "capacity_utilization_formula": "4 * renewable_generation_mwh_per_15min / installed_capacity_mw",
        "capacity_input_columns": ["Photovoltaics_installed_capacity [MW]", "Wind_onshore_installed_capacity [MW]"],
        "fit_end_utc": str(df.index[fit_end - 1]),
        "calibration_start_utc": str(df.index[cal_start]),
        "calibration_end_utc": str(df.index[-1]),
        "stat_quantile": args.stat_quantile, "ae_quantile": args.ae_quantile,
    }
    joblib.dump(bundle, out_dir / "residual_ae_bundle.joblib", compress=3)
    from src.models.residual.portable import save_portable
    save_portable(bundle, out_dir / "residual_ae_portable.pt")
    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)
    cal_table = pd.DataFrame({
        "timestamp_utc": df.index[cal_end],
        "residual_load": df["residual"].to_numpy()[cal_end],
        "expected_residual_load": np.asarray(expected)[cal_end],
        "statistical_score": cal_stat, "stat_threshold": stat_threshold,
        "autoencoder_score": ae_scores, "ae_threshold": ae_threshold,
        "statistical_flag": s_flag, "autoencoder_flag": a_flag,
        "classification": np.where(confirmed, "YES", "NO"),
    })
    cal_table.to_csv(out_dir / "calibration_scores.csv", index=False)
    summary = {
        "schema": "residual_ae_v2_capacity",
        "capacity_features": ["solar_capacity_utilization", "wind_capacity_utilization"],
        "input": str(args.input), "rows": n, "feature_names": FEATURE_NAMES,
        "fit_rows": fit_end, "validation_rows": cal_start - fit_end,
        "calibration_rows": n - cal_start, "training_windows": len(fit_starts),
        "earlystop_windows": len(early_starts), "calibration_windows": len(cal_starts),
        "screened_windows": screened, "best_epoch": best_epoch,
        "best_validation_loss": best_loss, "statistical_threshold": stat_threshold,
        "autoencoder_threshold": ae_threshold,
        "calibration_stage1_flags": int(s_flag.sum()),
        "calibration_stage2_flags": int(a_flag.sum()),
        "calibration_confirmed_flags": int(confirmed.sum()),
        "note": "Unlabelled calibration quantiles, not verified precision/recall or grid-incident accuracy",
    }
    (out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    print("Model saved:", out_dir / "residual_ae_bundle.joblib")
    print(f"Stat threshold={stat_threshold:.3f}, AE threshold={ae_threshold:.3f}")
    print(f"Calibration: stat={int(s_flag.sum())}, AE={int(a_flag.sum())}, both={int(confirmed.sum())} of {len(cal_starts)}")


if __name__ == "__main__":
    main()
