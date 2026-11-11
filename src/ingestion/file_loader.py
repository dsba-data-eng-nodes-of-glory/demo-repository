"""Validate station CSVs and normalize observations to the POC UTC schema.

The onboarding UI does not ask users to map timestamp/demand/temperature
columns. Common column names are detected automatically. Naive timestamps are
interpreted as UTC; timezone-aware timestamps are converted to UTC.
"""
from __future__ import annotations

from pathlib import Path
import re
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd

REQUIRED = ["utc_timestamp", "demand", "temperature"]

TIMESTAMP_ALIASES = (
    "utc_timestamp", "timestamp_utc", "timestamp", "datetime", "date_time",
    "date time", "utc_time", "time_utc", "date", "time",
)
DEMAND_ALIASES = (
    "demand", "actual_demand", "demand_mw", "load", "load_mw",
    "demand_met", "demand_met_mw", "actual_load", "actual_load_mw",
    "consumption", "power_mw", "power",
)
TEMPERATURE_ALIASES = (
    "temperature", "temp", "temperature_c", "temp_c", "temperature_2m",
    "air_temperature", "t2m", "temperature_celsius",
)


def _normalise_column_name(value: Any) -> str:
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def _alias_match(columns: list[str], aliases: tuple[str, ...]) -> str | None:
    by_normalised = {_normalise_column_name(column): column for column in columns}
    for alias in aliases:
        found = by_normalised.get(_normalise_column_name(alias))
        if found is not None:
            return found
    return None


def _parse_auto(raw: pd.Series) -> pd.Series:
    """Parse mixed ISO, European and common datetime strings robustly."""
    text = raw.astype("string").str.strip().replace("", pd.NA)
    # Parse year-first ISO values separately. In recent pandas versions,
    # ``format='mixed', dayfirst=True`` can reinterpret ISO month/day fields.
    iso_mask = text.str.match(r"^\d{4}-\d{2}-\d{2}(?:[T ]|$)", na=False)
    parsed = pd.Series(pd.NaT, index=text.index, dtype="datetime64[ns, UTC]")
    if iso_mask.any():
        try:
            parsed.loc[iso_mask] = pd.to_datetime(
                text.loc[iso_mask], format="ISO8601", errors="coerce", utc=True
            )
        except (TypeError, ValueError):
            parsed.loc[iso_mask] = pd.to_datetime(
                text.loc[iso_mask], errors="coerce", utc=True
            )

    other_mask = ~iso_mask & text.notna()
    if other_mask.any():
        try:
            parsed.loc[other_mask] = pd.to_datetime(
                text.loc[other_mask], format="mixed", dayfirst=True,
                errors="coerce", utc=True,
            )
        except (TypeError, ValueError):  # Compatibility with older pandas releases.
            parsed.loc[other_mask] = pd.to_datetime(
                text.loc[other_mask], dayfirst=True, errors="coerce", utc=True
            )

    # Try month-first values (for example MM/DD/YYYY) that cannot be parsed
    # with dayfirst=True, then ISO separately for mixed-offset strings.
    missing = parsed.isna() & text.notna()
    if missing.any():
        try:
            month_first = pd.to_datetime(
                text.loc[missing], format="mixed", dayfirst=False, errors="coerce", utc=True
            )
        except (TypeError, ValueError):
            month_first = pd.to_datetime(
                text.loc[missing], dayfirst=False, errors="coerce", utc=True
            )
        parsed.loc[missing] = month_first

    missing = parsed.isna() & text.notna()
    if missing.any():
        try:
            iso = pd.to_datetime(text.loc[missing], format="ISO8601", errors="coerce", utc=True)
        except (TypeError, ValueError):
            iso = pd.to_datetime(text.loc[missing], errors="coerce", utc=True)
        parsed.loc[missing] = iso
    return parsed


def _parse_timestamps(raw: pd.Series, timestamp_format: str | None) -> tuple[pd.Series, str]:
    requested = (timestamp_format or "auto").strip()
    text = raw.astype("string").str.strip().replace("", pd.NA)
    if requested.lower() in {"", "auto", "auto detect", "detect automatically"}:
        parsed = _parse_auto(text)
        used = "auto"
    else:
        try:
            parsed = pd.to_datetime(text, format=requested, errors="coerce", utc=True)
        except (TypeError, ValueError):
            parsed = pd.Series(pd.NaT, index=text.index, dtype="datetime64[ns, UTC]")
        used = requested

        # A user-selected pattern is a hint, not a reason to reject a valid CSV.
        # Fill values rejected by it using robust per-value detection.
        missing = parsed.isna() & text.notna()
        if missing.any():
            fallback = _parse_auto(text.loc[missing])
            if fallback.notna().any():
                parsed.loc[missing] = fallback
                used = f"{requested} (auto fallback for non-matching values)"
    return parsed, used


def detect_station_columns(
    columns: list[str],
    timestamp_column: str | None = None,
    demand_column: str | None = None,
    temperature_column: str | None = None,
) -> dict[str, str]:
    """Resolve source columns from common headers, with legacy overrides."""
    if timestamp_column and timestamp_column in columns:
        ts_col = timestamp_column
    else:
        ts_col = _alias_match(columns, TIMESTAMP_ALIASES)
        if ts_col is None:
            # Fuzzy fallback for headers such as "Measurement Timestamp (UTC)".
            ts_col = next(
                (column for column in columns if "timestamp" in _normalise_column_name(column)
                 or "datetime" in _normalise_column_name(column)),
                None,
            )
    if not ts_col:
        raise ValueError(
            "Could not identify the timestamp column automatically. Expected a header such as "
            "utc_timestamp, timestamp, datetime or date_time. Available columns: "
            + ", ".join(map(str, columns))
        )

    dem_col = demand_column if demand_column in columns else _alias_match(columns, DEMAND_ALIASES)
    temp_col = temperature_column if temperature_column in columns else _alias_match(columns, TEMPERATURE_ALIASES)
    missing = []
    if not dem_col:
        missing.append("demand/load")
    if missing:
        raise ValueError(
            "Could not identify required source column(s): " + ", ".join(missing)
            + ". The uploaded CSV must include a timestamp and demand/load column. "
            "Temperature is optional; when absent, the compatibility placeholder is 0 and "
            "the model cannot learn temperature effects. "
            "Recognized headers include demand/load_mw, demand_met_mw and temperature/temp_c. "
            "Available columns: " + ", ".join(map(str, columns))
        )
    return {"timestamp": ts_col, "demand": dem_col, "temperature": temp_col}


def load_station_csv(
    path: str | Path,
    interval_minutes: int = 15,
    timestamp_column: str | None = None,
    timestamp_format: str = "auto",
    source_timezone: str = "UTC",
    demand_column: str | None = None,
    temperature_column: str | None = None,
    resample_to_interval: bool = False,
    allow_gaps: bool = False,
) -> pd.DataFrame:
    """Load and normalize a station series, optionally resampling or retaining gaps.

    ``timestamp_column``, ``demand_column`` and ``temperature_column`` remain
    optional for backwards compatibility. New onboarding calls automatically
    infer all three from their CSV headers.

    Timezone-aware values are converted to UTC. Naive values are assumed to
    already be UTC because the current registration UI intentionally has no
    separate timezone selector.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    if interval_minutes <= 0:
        raise ValueError("interval_minutes must be greater than zero.")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("The uploaded CSV contains no data rows.")
    resolved = detect_station_columns(
        list(df.columns), timestamp_column, demand_column, temperature_column
    )

    raw_ts = df[resolved["timestamp"]]
    ts, actual_format = _parse_timestamps(raw_ts, timestamp_format)
    bad = ts.isna()
    if bad.any():
        bad_values = raw_ts.loc[bad].astype("string").head(5).tolist()
        raise ValueError(
            f"Could not parse {int(bad.sum()):,} of {len(raw_ts):,} timestamps in detected column "
            f"'{resolved['timestamp']}' using '{timestamp_format}'. Sample values: {bad_values}. "
            "Choose the matching timestamp format or use Auto detect. Timezone-aware values are "
            "converted to UTC; timestamps without an offset are interpreted as UTC."
        )

    # New onboarding assumes naive source timestamps are UTC. Keep the legacy
    # source_timezone argument functional for existing callers that explicitly
    # pass a non-UTC zone. Offset-aware values are always converted to UTC.
    naive_mask = ~raw_ts.astype("string").str.strip().str.contains(
        r"(?:Z|[+-]\d{2}:?\d{2})$", case=False, regex=True, na=False
    )
    if source_timezone and source_timezone.upper() != "UTC" and naive_mask.any():
        try:
            naive_text = raw_ts.loc[naive_mask].astype("string").str.strip()
            fmt = None if timestamp_format in (None, "", "auto", "Auto detect") else timestamp_format
            try:
                local_naive = pd.to_datetime(naive_text, format=fmt, errors="coerce")
            except (TypeError, ValueError):
                local_naive = pd.Series(pd.NaT, index=naive_text.index)
            missing_local = local_naive.isna()
            if missing_local.any():
                try:
                    fallback_naive = pd.to_datetime(
                        naive_text.loc[missing_local], format="mixed", dayfirst=True, errors="coerce"
                    )
                except (TypeError, ValueError):
                    fallback_naive = pd.to_datetime(
                        naive_text.loc[missing_local], dayfirst=True, errors="coerce"
                    )
                local_naive.loc[missing_local] = fallback_naive
            if local_naive.isna().any():
                raise ValueError("Could not parse naive timestamps with the supplied source timezone.")
            local_utc = local_naive.dt.tz_localize(
                ZoneInfo(source_timezone), ambiguous="raise", nonexistent="raise"
            ).dt.tz_convert("UTC")
            ts.loc[naive_mask] = local_utc
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"Unknown timezone '{source_timezone}'.") from exc

    df = df.copy()
    df["utc_timestamp"] = ts
    df["demand"] = pd.to_numeric(df[resolved["demand"]], errors="coerce")
    temperature_available = resolved["temperature"] is not None
    if temperature_available:
        df["temperature"] = pd.to_numeric(df[resolved["temperature"]], errors="coerce")
    else:
        # Keep the canonical station schema usable by Spark and existing model
        # code. This is a constant compatibility value, not a measurement.
        df["temperature"] = 0.0

    numeric_columns = ["demand", "temperature"] if temperature_available else ["demand"]
    invalid_numeric = df[numeric_columns].isna().any(axis=1)
    if invalid_numeric.any():
        raise ValueError(
            f"Found {int(invalid_numeric.sum()):,} rows with missing or non-numeric "
            f"{'demand/temperature' if temperature_available else 'demand'} values. "
            "Check the uploaded CSV values; these columns are detected automatically."
        )

    df = df.dropna(subset=REQUIRED)
    df = df.sort_values("utc_timestamp").drop_duplicates(subset=["utc_timestamp"], keep="last").reset_index(drop=True)
    if df.empty:
        raise ValueError("CSV contains no valid rows after validation.")
    if len(df) < 2:
        raise ValueError("At least 2 valid observations are required.")
    if (df["demand"] < 0).any():
        raise ValueError("Negative demand values found. Check the demand column mapping.")

    source_row_count = int(len(df))
    expected = pd.Timedelta(minutes=interval_minutes)
    if resample_to_interval:
        # Aggregate observations into UTC-aligned interval buckets. Empty
        # buckets remain absent so long gaps are preserved as segment breaks.
        df = (
            df.set_index("utc_timestamp")[REQUIRED[1:]]
            .resample(expected, origin="start_day", label="left", closed="left")
            .mean()
            .dropna(subset=["demand"])
            .reset_index()
        )
    else:
        differences = df["utc_timestamp"].diff().dropna()
        if not allow_gaps:
            irregular = differences[differences != expected]
            if not irregular.empty:
                examples = [str(value) for value in irregular.head(5).tolist()]
                counts = differences.value_counts().head(5).to_dict()
                raise ValueError(
                    f"Found {len(irregular):,} irregular timestamp intervals. The selected source interval is "
                    f"{interval_minutes} minutes. Most common observed intervals: {counts}. "
                    f"Examples of mismatches: {examples}. Select the actual source interval or check for missing/duplicate rows."
                )
        elif ((differences % expected) != pd.Timedelta(0)).any():
            raise ValueError(
                f"Timestamps must align to the selected {interval_minutes}-minute grid even when gaps are allowed."
            )

    if df.empty:
        raise ValueError("CSV contains no observations after interval aggregation.")

    result = df[REQUIRED].reset_index(drop=True)
    result.attrs["source_columns"] = resolved
    result.attrs["temperature_available"] = temperature_available
    result.attrs["timestamp_format_used"] = actual_format
    result.attrs["naive_timestamp_timezone"] = "UTC"
    result.attrs["source_row_count"] = source_row_count
    result.attrs["resampling_method"] = "mean" if resample_to_interval else "none"
    result.attrs["interval_minutes"] = int(interval_minutes)
    return result
