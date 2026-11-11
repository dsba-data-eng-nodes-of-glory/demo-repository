"""Compare source demand with observations persisted by the streaming predictor.

Example:
    python -m scripts.compare_actuals --source data/raw/amprion.csv \
        --actuals data/processed/stations/amprion/actuals.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def parse_time(values: pd.Series) -> pd.Series:
    text = values.astype("string").str.strip()
    parsed = pd.to_datetime(text, format="%d-%m-%Y %H:%M", errors="coerce", utc=True)
    missing = parsed.isna() & text.notna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(text.loc[missing], errors="coerce", utc=True)
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare source and persisted actual demand")
    parser.add_argument("--source", required=True)
    parser.add_argument("--actuals", required=True)
    args = parser.parse_args()
    source_path = Path(args.source)
    actual_path = Path(args.actuals)
    if not source_path.is_absolute():
        source_path = ROOT / source_path
    if not actual_path.is_absolute():
        actual_path = ROOT / actual_path
    source = pd.read_csv(source_path)
    actuals = pd.read_csv(actual_path)
    source["utc_timestamp"] = parse_time(source["utc_timestamp"])
    actuals["utc_timestamp"] = parse_time(actuals["utc_timestamp"])
    source["demand"] = pd.to_numeric(source["demand"], errors="coerce")
    actuals["actual_demand"] = pd.to_numeric(actuals["actual_demand"], errors="coerce")
    source = source.dropna(subset=["utc_timestamp", "demand"])
    actuals = actuals.dropna(subset=["utc_timestamp", "actual_demand"])
    comparison = source.merge(actuals[["utc_timestamp", "actual_demand"]], on="utc_timestamp", how="inner")
    comparison["difference"] = (comparison["demand"] - comparison["actual_demand"]).abs()
    print("Matched observations:", len(comparison))
    print(comparison["difference"].describe())
    print("Exact matches:", int((comparison["difference"] < 0.01).sum()))
    print(comparison.nlargest(10, "difference")[
        ["utc_timestamp", "demand", "actual_demand", "difference"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
