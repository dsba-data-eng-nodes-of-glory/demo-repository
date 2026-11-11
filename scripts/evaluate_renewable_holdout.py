"""Leakage-safe daily-origin evaluation for Amprion solar and wind holdout CSVs.

For each UTC day in the holdout, this script issues a 96-step forecast from
23:45 on the preceding day. The serving adapter receives the test file's
forecast_* weather covariates, and explicitly masks future realised generation.
Metrics are calculated only where the corresponding realised target exists.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.models.renewable_forecaster import AmprionRenewableForecaster
from src.models.training.amprion import solar_mlp_features
from src.models.training.amprion import Amprion_Wind_Dense as wind_features

ROOT = Path(__file__).resolve().parents[1]


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _metrics(group: pd.DataFrame) -> dict:
    actual = pd.to_numeric(group["actual_generation_mwh"], errors="coerce").to_numpy(float)
    predicted = pd.to_numeric(group["predicted_generation_mwh"], errors="coerce").to_numpy(float)
    valid = np.isfinite(actual) & np.isfinite(predicted)
    actual, predicted = actual[valid], predicted[valid]
    if not len(actual):
        return {
            "forecast_points": int(len(group)), "valid_actual_points": 0,
            "missing_actual_points": int(len(group)), "mae_mwh": np.nan,
            "rmse_mwh": np.nan, "wape_pct": np.nan, "r2": np.nan,
            "bias_mwh": np.nan,
        }
    error = predicted - actual
    denominator = float(np.abs(actual).sum())
    ss_total = float(np.square(actual - actual.mean()).sum())
    r2 = 1.0 - float(np.square(error).sum()) / ss_total if ss_total > 0 else np.nan
    return {
        "forecast_points": int(len(group)),
        "valid_actual_points": int(len(actual)),
        "missing_actual_points": int(len(group) - len(actual)),
        "mae_mwh": float(np.abs(error).mean()),
        "rmse_mwh": float(np.sqrt(np.square(error).mean())),
        "wape_pct": float(np.abs(error).sum() / denominator * 100.0) if denominator > 0 else np.nan,
        "r2": float(r2),
        "bias_mwh": float(error.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station", default="amprion", choices=["amprion"])
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--output-dir", default="data/processed/stations/amprion")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is not available")

    station_doc = yaml.safe_load((ROOT / "config/stations.yaml").read_text(encoding="utf-8"))
    station_cfg = station_doc["stations"][args.station]
    renewable_cfg = station_cfg["renewables"]
    if not renewable_cfg.get("enabled"):
        raise RuntimeError("Amprion renewable forecasts are disabled in config/stations.yaml")

    solar_test_path = _resolve(renewable_cfg["solar"]["test_file"])
    wind_test_path = _resolve(renewable_cfg["wind"]["test_file"])
    for path in (solar_test_path, wind_test_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required holdout CSV not found: {path}")

    solar_test, solar_diag = solar_mlp_features.load_solar_csv(solar_test_path, 15)
    wind_test, wind_diag = wind_features.load_wind_data(wind_test_path, 15)
    solar_actual = solar_test.set_index("utc_timestamp")[solar_mlp_features.TARGET_COLUMN]
    wind_actual = wind_test.set_index("utc_timestamp")[wind_features.TARGET_COLUMN]

    first_test_ts = min(solar_actual.index.min(), wind_actual.index.min())
    last_test_ts = max(solar_actual.index.max(), wind_actual.index.max())
    first_origin = first_test_ts.floor("D") - pd.Timedelta(minutes=15)
    last_origin = last_test_ts.floor("D") - pd.Timedelta(minutes=15)
    origins = pd.date_range(first_origin, last_origin, freq="1D", tz="UTC")
    if not len(origins):
        raise ValueError("No complete daily-origin holdout forecast periods were found")

    runner = AmprionRenewableForecaster(
        ROOT, args.station, renewable_cfg, device=torch.device(args.device)
    )
    rows: list[dict] = []
    failures: list[str] = []
    for origin in origins:
        records = runner.forecast_at(origin)
        by_kind = pd.DataFrame(records)
        for kind, actual_series in (("solar", solar_actual), ("wind", wind_actual)):
            part = by_kind[by_kind["generation_type"] == kind].copy() if not by_kind.empty else pd.DataFrame()
            if len(part) != 96:
                failures.append(f"{kind} at {origin.isoformat()}: expected 96 points, got {len(part)}")
                continue
            part["forecast_timestamp_utc"] = pd.to_datetime(
                part["forecast_timestamp_utc"], utc=True, errors="coerce"
            )
            part["actual_generation_mwh"] = part["forecast_timestamp_utc"].map(actual_series)
            part["horizon_step"] = pd.to_numeric(part["horizon_step"], errors="coerce").astype(int)
            part["forecast_origin_utc"] = origin
            rows.extend(part.to_dict("records"))

    predictions = pd.DataFrame(rows)
    if predictions.empty:
        raise RuntimeError("No renewable holdout predictions were generated")

    metric_rows = []
    horizon_rows = []
    for kind, group in predictions.groupby("generation_type"):
        metric_rows.append({"generation_type": kind, **_metrics(group)})
        for horizon, horizon_group in group.groupby("horizon_step"):
            horizon_rows.append({
                "generation_type": kind, "horizon_step": int(horizon),
                "lead_minutes": int(horizon) * 15,
                **_metrics(horizon_group),
            })

    metrics = pd.DataFrame(metric_rows)
    horizon_metrics = pd.DataFrame(horizon_rows)
    out_dir = _resolve(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out_dir / "renewable_holdout_predictions.csv", index=False)
    metrics.to_csv(out_dir / "renewable_holdout_metrics.csv", index=False)
    horizon_metrics.to_csv(out_dir / "renewable_holdout_horizon_metrics.csv", index=False)

    print("Amprion renewable holdout evaluation complete")
    print(f"Origins: {len(origins)} daily 23:45 UTC forecast origins")
    print(f"Forecast points: {len(predictions):,}")
    print(f"Solar loader diagnostics: {solar_diag}")
    print(f"Wind loader diagnostics: {wind_diag}")
    print(metrics.to_string(index=False))
    if failures:
        print("Forecast failures:")
        for failure in failures:
            print(f" - {failure}")
        raise RuntimeError(f"{len(failures)} renewable daily horizon(s) were incomplete")
    print(f"Outputs: {out_dir}")


if __name__ == "__main__":
    main()
