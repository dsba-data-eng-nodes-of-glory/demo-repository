
"""Validate and normalize a station CSV before streaming."""

from pathlib import Path

import pandas as pd


REQUIRED = ["utc_timestamp", "demand", "temperature"]


def load_station_csv(
    path: str | Path,
    interval_minutes: int = 15,
) -> pd.DataFrame:
    """Load, clean, sort and validate station observations."""

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Input CSV not found: {path}"
        )

    if interval_minutes <= 0:
        raise ValueError(
            "interval_minutes must be greater than zero."
        )

    # Read source CSV
    df = pd.read_csv(path)

    # Check required columns
    missing = set(REQUIRED) - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    # Normalize timestamps
    raw_ts = df["utc_timestamp"].astype("string").str.strip()

    ts = pd.to_datetime(
        raw_ts,
        format="%d-%m-%Y %H:%M",
        errors="coerce",
        utc=True,
    )

    # Fallback for other supported timestamp formats
    fallback = ts.isna() & raw_ts.notna()

    if fallback.any():
        ts.loc[fallback] = pd.to_datetime(
            raw_ts.loc[fallback],
            errors="coerce",
            utc=True,
        )

    df["utc_timestamp"] = ts

    # Convert numerical columns
    for col in ["demand", "temperature"]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    # Remove invalid records
    df = df.dropna(subset=REQUIRED)

    # Sort and remove duplicate timestamps
    df = (
        df.sort_values("utc_timestamp")
        .drop_duplicates(
            subset=["utc_timestamp"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    if df.empty:
        raise ValueError(
            "CSV contains no valid rows after validation."
        )

    if len(df) < 2:
        raise ValueError(
            "At least 2 valid observations are required."
        )

    # Validate demand
    if (df["demand"] < 0).any():
        raise ValueError(
            "Negative demand values found in CSV."
        )

    # Validate continuous timestamp intervals
    expected = pd.Timedelta(
        minutes=interval_minutes
    )

    differences = df["utc_timestamp"].diff().dropna()

    irregular = differences[differences != expected]

    if not irregular.empty:
        raise ValueError(
            f"Found {len(irregular)} irregular timestamp intervals. "
            f"Expected continuous {interval_minutes}-minute observations. "
            f"Examples: {irregular.head(5).tolist()}"
        )

    # Return only model input columns
    return df[REQUIRED].reset_index(drop=True)
