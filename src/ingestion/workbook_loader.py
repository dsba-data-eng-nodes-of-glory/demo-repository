"""Read and validate a multi-model station onboarding workbook.

The workbook contract has three worksheets: Demand, Solar and Wind. Demand is
shared by the demand forecaster and the demand anomaly autoencoder. This module
keeps the model-facing CSV schemas explicit so the existing training scripts
can consume each model's own canonical column names.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
import pandas as pd

MODEL_IDS = (
    "demand_forecast",
    "solar_forecast",
    "wind_forecast",
    "residual_anomaly",
)
DEFAULT_SELECTED_MODELS = list(MODEL_IDS)

SHEET_FOR_MODEL = {
    "demand_forecast": "Demand",
    "residual_anomaly": "Demand",
    "solar_forecast": "Solar",
    "wind_forecast": "Wind",
}

SOLAR_COLUMNS = [
    "utc_timestamp",
    "Photovoltaics [MWh]",
    "cloud_cover_mean_pct",
    "shortwave_radiation_mean_prev_hour_w_m2",
    "forecast_cloud_cover_mean_pct",
    "forecast_shortwave_radiation_mean_prev_hour_w_m2",
    "Photovoltaics_installed_capacity [MW]",
]
WIND_COLUMNS = [
    "utc_timestamp",
    "Wind onshore [MWh]",
    "wind_speed_100m_mean_ms",
    "wind_direction_100m_circular_mean_deg",
    "Wind_onshore_installed_capacity [MW]",
    "future_wind_speed",
    "future_wind_direction",
]

TIMESTAMP_ALIASES = ("utc_timestamp", "timestamp", "datetime", "date_time", "date time")
DEMAND_ALIASES = ("demand", "demand_mw", "load", "load_mw", "actual_demand", "actual_load_mw")
TEMPERATURE_ALIASES = ("temperature", "temperature_c", "temp", "temp_c", "temperature_2m")


def parse_selected_models(value: str | Iterable[str] | None) -> list[str]:
    """Parse JSON-array or comma-separated form input; empty means all four."""
    if value is None:
        return list(DEFAULT_SELECTED_MODELS)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return list(DEFAULT_SELECTED_MODELS)
        if text.startswith("["):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError("selected_models must be a JSON array or comma-separated model IDs.") from exc
            if not isinstance(parsed, list):
                raise ValueError("selected_models must be a list of model IDs.")
            raw = parsed
        else:
            raw = [item.strip() for item in text.split(",") if item.strip()]
    else:
        raw = list(value)

    selected = []
    unknown = []
    for item in raw:
        model_id = str(item).strip().lower()
        if not model_id:
            continue
        if model_id not in MODEL_IDS:
            unknown.append(model_id)
        elif model_id not in selected:
            selected.append(model_id)
    if unknown:
        raise ValueError(f"Unknown model selection(s): {', '.join(unknown)}. Allowed: {', '.join(MODEL_IDS)}.")
    if not selected:
        raise ValueError("Select at least one model for this station.")
    return selected


def _clean_column_names(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.columns = [str(c).replace("\xa0", " ").strip() for c in result.columns]
    if result.columns.duplicated().any():
        duplicates = result.columns[result.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate column headers: {duplicates}.")
    return result


def _resolve_alias(columns: list[str], aliases: tuple[str, ...]) -> str | None:
    lookup = {re.sub(r"[^a-z0-9]+", "_", str(c).lower()).strip("_"): c for c in columns}
    for alias in aliases:
        key = re.sub(r"[^a-z0-9]+", "_", alias.lower()).strip("_")
        if key in lookup:
            return lookup[key]
    return None


def _timestamp_series(
    values: pd.Series,
    timestamp_format: str,
    source_timezone: str,
    sheet_name: str,
) -> pd.Series:
    raw = values.astype("string").str.strip()
    empty = raw.isna() | raw.eq("")
    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns, UTC]")

    # Parse offset-aware timestamps as absolute instants and localize naive
    # timestamps using the timezone supplied during station onboarding.
    aware_mask = raw.str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$", case=False, regex=True, na=False) & ~empty
    naive_mask = ~aware_mask & ~empty
    fmt = None if (timestamp_format or "auto").strip().lower() in {"", "auto", "auto detect"} else timestamp_format

    def parse_values(series: pd.Series) -> pd.Series:
        try:
            return pd.to_datetime(series, format=fmt or "mixed", dayfirst=True, errors="coerce")
        except (TypeError, ValueError):
            return pd.to_datetime(series, dayfirst=True, errors="coerce")

    if aware_mask.any():
        parsed.loc[aware_mask] = pd.to_datetime(raw.loc[aware_mask], format="mixed", errors="coerce", utc=True)
    if naive_mask.any():
        naive = parse_values(raw.loc[naive_mask])
        if getattr(naive.dt, "tz", None) is not None:
            naive = naive.dt.tz_localize(None)
        try:
            localized = naive.dt.tz_localize(
                ZoneInfo(source_timezone), ambiguous="raise", nonexistent="raise"
            ).dt.tz_convert("UTC")
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"Unknown source timezone '{source_timezone}'.") from exc
        except Exception as exc:
            raise ValueError(
                f"Could not localize naive timestamps in worksheet '{sheet_name}' using "
                f"timezone '{source_timezone}': {exc}"
            ) from exc
        parsed.loc[naive_mask] = localized

    if parsed.isna().any():
        examples = raw.loc[parsed.isna()].head(5).tolist()
        raise ValueError(
            f"Worksheet '{sheet_name}': {int(parsed.isna().sum()):,} timestamp(s) are blank or invalid. "
            f"Examples: {examples}. Check timestamp_format and source_timezone."
        )
    return parsed


def _validate_time_grid(
    frame: pd.DataFrame,
    sheet_name: str,
    interval_minutes: int,
) -> pd.DataFrame:
    frame = frame.sort_values("utc_timestamp").reset_index(drop=True)
    duplicated = frame["utc_timestamp"].duplicated(keep=False)
    if duplicated.any():
        examples = frame.loc[duplicated, "utc_timestamp"].head(5).astype(str).tolist()
        raise ValueError(f"Worksheet '{sheet_name}': duplicate timestamps found. Examples: {examples}.")
    if len(frame) < 2:
        raise ValueError(f"Worksheet '{sheet_name}' must contain at least two valid observations.")
    expected = pd.Timedelta(minutes=interval_minutes)
    diffs = frame["utc_timestamp"].diff().dropna()
    irregular = diffs.ne(expected)
    if irregular.any():
        examples = diffs.loc[irregular].head(5).astype(str).tolist()
        raise ValueError(
            f"Worksheet '{sheet_name}': found {int(irregular.sum()):,} irregular or missing intervals. "
            f"Expected every {interval_minutes} minutes. Example gaps: {examples}."
        )
    return frame


def _numeric_columns(frame: pd.DataFrame, columns: list[str], sheet_name: str) -> pd.DataFrame:
    frame = frame.copy()
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    values = frame[columns].to_numpy(dtype=np.float64)
    invalid = ~np.isfinite(values)
    if invalid.any():
        row_indexes, col_indexes = np.where(invalid)
        sample = [
            {"row": int(row + 2), "column": columns[int(col)], "value": str(frame.iloc[int(row)][columns[int(col)]])}
            for row, col in list(zip(row_indexes, col_indexes))[:5]
        ]
        raise ValueError(
            f"Worksheet '{sheet_name}': {int(invalid.sum()):,} blank, non-numeric or non-finite numeric "
            f"value(s) found. Examples (Excel row numbers): {sample}."
        )
    return frame


def _read_sheet(
    workbook: dict[str, pd.DataFrame],
    sheet_name: str,
    required_columns: list[str],
    timestamp_format: str,
    source_timezone: str,
    interval_minutes: int,
    aliases: dict[str, tuple[str, ...]] | None = None,
) -> pd.DataFrame:
    if sheet_name not in workbook:
        raise ValueError(f"Workbook is missing required worksheet '{sheet_name}'.")
    frame = _clean_column_names(workbook[sheet_name])
    aliases = aliases or {}
    renames: dict[str, str] = {}
    for canonical, candidates in aliases.items():
        if canonical not in frame.columns:
            detected = _resolve_alias(list(frame.columns), candidates)
            if detected:
                renames[detected] = canonical
    frame = frame.rename(columns=renames)
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(
            f"Worksheet '{sheet_name}' is missing required column(s): {missing}. "
            f"Available columns: {list(frame.columns)}."
        )
    frame = frame[required_columns].copy()
    frame["utc_timestamp"] = _timestamp_series(
        frame["utc_timestamp"], timestamp_format, source_timezone, sheet_name
    )
    numeric_cols = [column for column in required_columns if column != "utc_timestamp"]
    frame = _numeric_columns(frame, numeric_cols, sheet_name)
    frame = _validate_time_grid(frame, sheet_name, interval_minutes)
    return frame


def load_station_workbook(
    path: str | Path,
    selected_models: str | Iterable[str] | None = None,
    interval_minutes: int = 15,
    timestamp_format: str = "auto",
    source_timezone: str = "UTC",
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Load selected worksheet data and enforce model-specific source contracts."""
    workbook_path = Path(path)
    if not workbook_path.is_file():
        raise FileNotFoundError(f"Station workbook not found: {workbook_path}")
    if interval_minutes <= 0:
        raise ValueError("interval_minutes must be greater than zero.")

    selected = parse_selected_models(selected_models)
    needed_sheets = sorted({SHEET_FOR_MODEL[item] for item in selected})
    try:
        source = pd.read_excel(workbook_path, sheet_name=needed_sheets, engine="openpyxl")
    except ValueError as exc:
        # pandas may wrap missing sheet names in a ValueError; retain sheet-aware detail.
        raise ValueError(f"Could not read required workbook worksheets {needed_sheets}: {exc}") from exc
    except Exception as exc:
        raise ValueError(f"Could not read the Excel workbook: {exc}") from exc

    result: dict[str, pd.DataFrame] = {}
    diagnostics: dict[str, Any] = {
        "selected_models": selected,
        "worksheets": {},
        "interval_minutes": int(interval_minutes),
        "source_timezone": source_timezone,
    }

    if 'residual_anomaly' in selected and not {'demand_forecast','solar_forecast','wind_forecast'}.issubset(selected):
        raise ValueError('Residual-load anomaly needs Demand, Solar and Wind forecasting components all enabled.')
    demand_models = {"demand_forecast", "residual_anomaly"}.intersection(selected)
    if demand_models:
        demand_required = ["utc_timestamp", "demand"]
        if "demand_forecast" in selected:
            demand_required.append("temperature")
        demand_aliases = {
            "utc_timestamp": TIMESTAMP_ALIASES,
            "demand": DEMAND_ALIASES,
            "temperature": TEMPERATURE_ALIASES,
        }
        demand = _read_sheet(
            source, "Demand", demand_required, timestamp_format, source_timezone,
            interval_minutes, demand_aliases,
        )
        if (demand["demand"] < 0).any():
            raise ValueError("Worksheet 'Demand': negative demand values are not allowed.")
        result["demand"] = demand
        diagnostics["worksheets"]["Demand"] = {
            "rows": int(len(demand)),
            "start_utc": demand.utc_timestamp.iloc[0].isoformat(),
            "end_utc": demand.utc_timestamp.iloc[-1].isoformat(),
            "columns": list(demand.columns),
        }

    if "solar_forecast" in selected:
        solar = _read_sheet(
            source, "Solar", SOLAR_COLUMNS, timestamp_format, source_timezone,
            interval_minutes, {"utc_timestamp": TIMESTAMP_ALIASES},
        )
        nonnegative = [
            "Photovoltaics [MWh]", "shortwave_radiation_mean_prev_hour_w_m2",
            "forecast_shortwave_radiation_mean_prev_hour_w_m2",
        ]
        bounded = ["cloud_cover_mean_pct", "forecast_cloud_cover_mean_pct"]
        if (solar[nonnegative] < 0).any().any():
            raise ValueError("Worksheet 'Solar': generation and radiation values must be nonnegative.")
        if ((solar[bounded] < 0) | (solar[bounded] > 100)).any().any():
            raise ValueError("Worksheet 'Solar': cloud-cover values must be within 0–100 percent.")
        if (solar["Photovoltaics_installed_capacity [MW]"] <= 0).any():
            raise ValueError("Worksheet 'Solar': Photovoltaics_installed_capacity [MW] must be positive.")
        result["solar"] = solar
        diagnostics["worksheets"]["Solar"] = {
            "rows": int(len(solar)),
            "start_utc": solar.utc_timestamp.iloc[0].isoformat(),
            "end_utc": solar.utc_timestamp.iloc[-1].isoformat(),
            "columns": list(solar.columns),
        }

    if "wind_forecast" in selected:
        wind = _read_sheet(
            source, "Wind", WIND_COLUMNS, timestamp_format, source_timezone,
            interval_minutes, {"utc_timestamp": TIMESTAMP_ALIASES},
        )
        nonnegative = ["Wind onshore [MWh]", "wind_speed_100m_mean_ms", "future_wind_speed"]
        directional = ["wind_direction_100m_circular_mean_deg", "future_wind_direction"]
        if (wind[nonnegative] < 0).any().any():
            raise ValueError("Worksheet 'Wind': generation and wind-speed values must be nonnegative.")
        if ((wind[directional] < 0) | (wind[directional] > 360)).any().any():
            raise ValueError("Worksheet 'Wind': wind direction must be within 0–360 degrees.")
        if (wind["Wind_onshore_installed_capacity [MW]"] <= 0).any():
            raise ValueError("Worksheet 'Wind': Wind_onshore_installed_capacity [MW] must be positive.")
        result["wind"] = wind
        diagnostics["worksheets"]["Wind"] = {
            "rows": int(len(wind)),
            "start_utc": wind.utc_timestamp.iloc[0].isoformat(),
            "end_utc": wind.utc_timestamp.iloc[-1].isoformat(),
            "columns": list(wind.columns),
        }

    return result, diagnostics
