"""Pure pandas helpers for evaluating saved rolling electricity forecasts."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _to_utc(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    if frame.empty or column not in frame.columns:
        return pd.DataFrame()
    result = frame.copy()
    result[column] = pd.to_datetime(result[column], utc=True, errors="coerce")
    return result.dropna(subset=[column])


def build_evaluation_frame(
    forecasts: pd.DataFrame,
    actuals: pd.DataFrame,
) -> pd.DataFrame:
    """Join forecast targets to realized actual demand by exact UTC timestamp.

    A forecast row is evaluated only after an actual value exists at the exact
    forecast target timestamp. This deliberately excludes not-yet-realized
    forecast targets rather than treating missing actuals as forecast errors.
    """
    required_forecast = {"forecast_timestamp_utc", "predicted_demand"}
    required_actual = {"utc_timestamp", "actual_demand"}
    if forecasts.empty or actuals.empty:
        return pd.DataFrame()
    if not required_forecast.issubset(forecasts.columns):
        return pd.DataFrame()
    if not required_actual.issubset(actuals.columns):
        return pd.DataFrame()

    f = forecasts.copy()
    a = actuals.copy()

    if "station_id" in f.columns and "station_id" in a.columns:
        # APIs should already filter by station. Keep this guard in case an
        # upstream API response accidentally contains mixed station records.
        common = set(f["station_id"].dropna().astype(str)) & set(a["station_id"].dropna().astype(str))
        if len(common) == 1:
            station = next(iter(common))
            f = f[f["station_id"].astype(str) == station]
            a = a[a["station_id"].astype(str) == station]

    f["forecast_timestamp_utc"] = pd.to_datetime(
        f["forecast_timestamp_utc"], utc=True, errors="coerce"
    )
    a["utc_timestamp"] = pd.to_datetime(a["utc_timestamp"], utc=True, errors="coerce")
    f["predicted_demand"] = pd.to_numeric(f["predicted_demand"], errors="coerce")
    a["actual_demand"] = pd.to_numeric(a["actual_demand"], errors="coerce")
    f = f.dropna(subset=["forecast_timestamp_utc", "predicted_demand"])
    a = a.dropna(subset=["utc_timestamp", "actual_demand"])
    if f.empty or a.empty:
        return pd.DataFrame()

    if "forecast_origin_utc" in f.columns:
        f["forecast_origin_utc"] = pd.to_datetime(
            f["forecast_origin_utc"], utc=True, errors="coerce"
        )
    else:
        f["forecast_origin_utc"] = pd.NaT
    if "forecast_issued_at_utc" in f.columns:
        f["forecast_issued_at_utc"] = pd.to_datetime(
            f["forecast_issued_at_utc"], utc=True, errors="coerce"
        )
    else:
        f["forecast_issued_at_utc"] = pd.NaT

    # For old rows without an explicit forecast origin, use issue time as the
    # run identifier. Do not compare issue time to the simulation timestamps:
    # issue time is wall-clock time in this POC, while origin/target are data time.
    f["run_origin_utc"] = f["forecast_origin_utc"].fillna(f["forecast_issued_at_utc"])
    f = f.dropna(subset=["run_origin_utc"])
    if "horizon_step" in f.columns:
        f["horizon_step"] = pd.to_numeric(f["horizon_step"], errors="coerce")
    else:
        f["horizon_step"] = np.nan

    a = (
        a.sort_values("utc_timestamp")
        .drop_duplicates("utc_timestamp", keep="last")
        [["utc_timestamp", "actual_demand"]]
    )
    columns = [
        "run_origin_utc", "forecast_timestamp_utc", "forecast_issued_at_utc",
        "horizon_step", "predicted_demand",
    ]
    for lineage_column in ("model_file", "model_version", "algorithm", "experiment_id"):
        if lineage_column in f.columns:
            columns.append(lineage_column)
    f = f[columns].copy()

    evaluated = f.merge(
        a,
        left_on="forecast_timestamp_utc",
        right_on="utc_timestamp",
        how="inner",
        validate="many_to_one",
    )
    if evaluated.empty:
        return evaluated

    evaluated["error_mw"] = evaluated["predicted_demand"] - evaluated["actual_demand"]
    evaluated["absolute_error_mw"] = evaluated["error_mw"].abs()
    evaluated["squared_error_mw"] = evaluated["error_mw"] ** 2
    evaluated["actual_timestamp_utc"] = evaluated["utc_timestamp"]
    # Avoid duplicate forecast records within a run if an output file was
    # appended twice. Keep the last saved copy for each origin/target/model.
    dedupe_keys = ["run_origin_utc", "forecast_timestamp_utc", "horizon_step"]
    for lineage_column in ("model_file", "model_version", "algorithm", "experiment_id"):
        if lineage_column in evaluated.columns:
            dedupe_keys.append(lineage_column)
    evaluated = evaluated.drop_duplicates(dedupe_keys, keep="last")
    return evaluated.sort_values(
        ["actual_timestamp_utc", "run_origin_utc", "horizon_step"]
    ).reset_index(drop=True)


def calculate_metrics(frame: pd.DataFrame) -> dict[str, float | int | None]:
    """Return scale-aware operational metrics in MW and WAPE percent."""
    if frame.empty:
        return {
            "matched_observations": 0,
            "mae_mw": None,
            "rmse_mw": None,
            "wape_pct": None,
            "bias_mw": None,
        }
    error = pd.to_numeric(frame["error_mw"], errors="coerce").dropna()
    actual = pd.to_numeric(frame.loc[error.index, "actual_demand"], errors="coerce")
    valid = actual.notna()
    error = error.loc[valid]
    actual = actual.loc[valid]
    if error.empty:
        return {
            "matched_observations": 0,
            "mae_mw": None,
            "rmse_mw": None,
            "wape_pct": None,
            "bias_mw": None,
        }
    denominator = float(actual.abs().sum())
    wape = float(error.abs().sum() / denominator * 100.0) if denominator > 0 else None
    return {
        "matched_observations": int(len(error)),
        "mae_mw": float(error.abs().mean()),
        "rmse_mw": float(np.sqrt(np.square(error.to_numpy(dtype=float)).mean())),
        "wape_pct": wape,
        "bias_mw": float(error.mean()),
    }


def metrics_by_run(evaluated: pd.DataFrame, horizon_steps: int) -> pd.DataFrame:
    """Summarize each forecast origin; mark only sufficiently matured runs."""
    if evaluated.empty:
        return pd.DataFrame()
    keys = ["run_origin_utc"]
    for lineage_column in ("model_file", "model_version", "algorithm", "experiment_id"):
        if lineage_column in evaluated.columns:
            keys.append(lineage_column)
    rows: list[dict] = []
    for key, group in evaluated.groupby(keys, dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        metrics = calculate_metrics(group)
        row = dict(zip(keys, key))
        row.update(metrics)
        row["coverage_pct"] = (
            metrics["matched_observations"] / max(1, int(horizon_steps)) * 100.0
        )
        rows.append(row)
    result = pd.DataFrame(rows).sort_values("run_origin_utc").reset_index(drop=True)
    result["matured"] = result["matched_observations"] >= max(1, int(np.ceil(horizon_steps * 0.80)))
    return result


def metrics_by_horizon(evaluated: pd.DataFrame, matured_runs: pd.DataFrame) -> pd.DataFrame:
    """Return lead-time MAE/RMSE using only sufficiently matured forecast runs."""
    if evaluated.empty or matured_runs.empty or "horizon_step" not in evaluated.columns:
        return pd.DataFrame()
    matured_origins = set(matured_runs.loc[matured_runs["matured"], "run_origin_utc"])
    sample = evaluated[evaluated["run_origin_utc"].isin(matured_origins)].copy()
    sample = sample.dropna(subset=["horizon_step"])
    if sample.empty:
        return pd.DataFrame()
    rows = []
    for step, group in sample.groupby("horizon_step", sort=True):
        metrics = calculate_metrics(group)
        rows.append({"horizon_step": int(step), **metrics})
    return pd.DataFrame(rows).sort_values("horizon_step").reset_index(drop=True)
