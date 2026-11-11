"""Residual-load anomaly features and robust baseline (15-minute station data).

Generation is interpreted as MWh per 15-minute interval for the capacity
utilisation ratios, while residual/ramp values retain their source units.
Verify this upstream assumption before operational dispatch decisions.
"""
from __future__ import annotations

import warnings
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

WINDOW_STEPS = 96
PHYSICAL_FEATURES = [
    "residual_deviation", "residual_ramp_30m", "residual_ramp_1h",
    "residual_change_24h", "renewable_share", "wind_change_1h",
    "solar_change_1h", "solar_capacity_utilization", "wind_capacity_utilization",
]
CALENDAR_FEATURES = [
    "tod_sin", "tod_cos", "dow_sin", "dow_cos", "year_sin", "year_cos",
]
STAT_SCREEN_FEATURES = PHYSICAL_FEATURES[:4]  # Screen residual requirement, not solar/wind alone
FEATURE_NAMES = PHYSICAL_FEATURES + CALENDAR_FEATURES
FEATURE_WEIGHTS = (1.0,) * 7 + (0.75, 0.75) + (0.20,) * 6
AE_GROUPS = {
    "RESIDUAL_LEVEL": (0,),
    "RESIDUAL_RAMP": (1, 2, 3),
    "RENEWABLE_CHANGE": (4, 5, 6),
    "CAPACITY_UTILIZATION": (7, 8),
}


def robust_location_scale(x: np.ndarray) -> tuple[float, float]:
    v = np.asarray(x, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 20:
        raise ValueError("At least 20 valid values needed for robust calibration")
    med = float(np.median(v))
    scale = float(1.4826 * np.median(np.abs(v - med)))
    if not np.isfinite(scale) or scale < 1e-10:
        scale = float(np.std(v))
    if not np.isfinite(scale) or scale < 1e-10:
        scale = 1e-8
    return med, scale


def load_residual_csv(
    path: str | Path,
    timestamp_col: str = "utc_timestamp",
    demand_col: str = "demand",
    solar_col: str = "Photovoltaics",
    wind_col: str = "Wind onshore",
    residual_col: str = "Residual Load",
    solar_capacity_col: str = "Photovoltaics_installed_capacity [MW]",
    wind_capacity_col: str = "Wind_onshore_installed_capacity [MW]",
) -> pd.DataFrame:
    """Load seven input columns, including installed capacities in MW.

    Columns are stripped; numbers may contain thousands separators. Recompute
    residual. Never interpolate missing intervals, generation or capacity.
    """
    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(x).strip() for x in df.columns]
    tcol, dcol, scol, wcol, rcol, sccol, wccol = [x.strip() for x in (
        timestamp_col, demand_col, solar_col, wind_col, residual_col,
        solar_capacity_col, wind_capacity_col,
    )]
    must_have = [tcol, dcol, scol, wcol, sccol, wccol]
    missing = [x for x in must_have if x not in df]
    if missing:
        raise ValueError(f"Missing columns {missing}. Available: {list(df.columns)}")
    ts = pd.to_datetime(df[tcol].astype("string").str.strip(),
                        format="mixed", dayfirst=True, utc=True, errors="coerce")
    if ts.isna().any():
        raise ValueError(f"Unparseable timestamps: {df.loc[ts.isna(), tcol].head(5).tolist()}")
    if ts.duplicated().any():
        raise ValueError("Duplicate timestamps found. Select a single forecast vintage per timestamp before scoring.")
    if (ts.astype("int64").to_numpy() % pd.Timedelta(minutes=15).value) .any():
        raise ValueError("Timestamps must align with 15-minute UTC intervals")

    def num(col: str) -> pd.Series:
        s = df[col].astype("string").str.replace(",", "", regex=False).str.strip()
        v = pd.to_numeric(s, errors="coerce")
        bad = v.isna() & s.notna() & s.ne("")
        if bad.any():
            raise ValueError(f"Non-numeric values in {col}: {s[bad].head(5).tolist()}")
        return v.astype(float)

    demand, solar, wind = num(dcol), num(scol), num(wcol)
    solar_capacity, wind_capacity = num(sccol), num(wccol)
    for col, cap in ((sccol, solar_capacity), (wccol, wind_capacity)):
        if (cap.dropna() <= 0).any():
            raise ValueError(f"{col} must be positive MW")
    for col, vals in ((dcol, demand), (scol, solar), (wcol, wind)):
        if (vals.dropna() < 0).any():
            raise ValueError(f"{col} contains negative values")
    if (demand.dropna() == 0).any():
        raise ValueError("Demand contains zero values; renewable share is undefined")
    computed = demand - solar - wind
    if rcol in df:
        given = num(rcol)
        check = (given - computed).abs()
        mismatches = (check > (0.05 + 1e-6 * computed.abs())) & check.notna()
        if mismatches.any():
            raise ValueError(f"Provided {rcol} disagrees with demand - solar - wind in {int(mismatches.sum()):,} rows. Check units and alignment.")
    out = pd.DataFrame({
        "demand": demand.to_numpy(), "solar": solar.to_numpy(),
        "wind": wind.to_numpy(), "residual": computed.to_numpy(),
        "solar_capacity_mw": solar_capacity.to_numpy(),
        "wind_capacity_mw": wind_capacity.to_numpy(),
    }, index=pd.DatetimeIndex(ts))
    out = out.sort_index()
    # Interpret generation as energy delivered in a 15-minute interval (MWh).
    # This assumption is required for capacity utilisation = (4 * MWh) / MW.
    for source, cap_col in (("solar", "solar_capacity_mw"), ("wind", "wind_capacity_mw")):
        util = 4.0 * out[source] / out[cap_col]
        if (util > 1.05).any():
            raise ValueError(
                f"{source} quarter-hour capacity utilisation exceeds 105%; "
                "check MW/MWh unit conventions and regional capacity coverage."
            )
    full = pd.date_range(out.index.min(), out.index.max(), freq="15min", tz="UTC")
    if len(full) != len(out):
        warnings.warn(f"Missing {len(full)-len(out)} 15-minute timestamp(s); preserving as NaN", RuntimeWarning)
    out = out.reindex(full)
    out.index.name = "timestamp_utc"
    return out


def combine_history_test(test: pd.DataFrame, history: pd.DataFrame | None) -> pd.DataFrame:
    if history is None:
        return test.copy()
    # Duplicate test timestamps intentionally override history, not vice versa.
    out = pd.concat([history, test])
    out = out.loc[~out.index.duplicated(keep="last")].sort_index()
    idx = pd.date_range(out.index.min(), out.index.max(), freq="15min", tz="UTC")
    return out.reindex(idx).rename_axis("timestamp_utc")


def fractional_year(idx: pd.DatetimeIndex) -> np.ndarray:
    local = idx
    sec = local.hour.to_numpy() * 3600 + local.minute.to_numpy() * 60
    days = np.where(local.is_leap_year, 366.0, 365.0)
    return local.year.to_numpy(dtype=float) + (local.dayofyear.to_numpy() - 1 + sec / 86400.0) / days


def _profile_keys(idx: pd.DatetimeIndex, timezone: str):
    local = idx.tz_convert(timezone)
    return local, local.month.to_numpy(), local.dayofweek.to_numpy(), (local.hour * 4 + local.minute // 15).to_numpy()


def fit_baseline(training_residual: pd.Series, timezone: str = "Europe/Berlin") -> dict:
    """Historical-only robust trend + month/weekday/quarter-hour expected residual."""
    idx = training_residual.index
    local, month, dow, slot = _profile_keys(idx, timezone)
    x = training_residual.to_numpy(dtype=float)
    daily = pd.Series(x, index=local).resample("D").median().dropna()
    annual = daily.groupby(daily.index.year).agg(["median", "count"])
    annual = annual[annual["count"] >= 120]
    if len(annual) < 2:
        raise ValueError("Baseline needs at least two reasonably complete historical years")
    years = annual.index.to_numpy(dtype=float) + 0.5
    y = annual["median"].to_numpy(dtype=float)
    slopes = [(y[j] - y[i]) / (years[j] - years[i]) for i in range(len(y)) for j in range(i + 1, len(y))]
    slope = float(np.median(slopes))
    center = float(np.median(years))
    intercept = float(np.median(y - slope * (years - center)))
    residual_detrended = x - (intercept + slope * (fractional_year(local) - center))
    frame = pd.DataFrame({"month": month, "dow": dow, "slot": slot, "v": residual_detrended}).dropna()
    if len(frame) < 96 * 60:
        raise ValueError("Insufficient valid training data for residual calendar profile")
    maps = {}
    for name, keys in (
        ("month_weekday_slot", ["month", "dow", "slot"]),
        ("weekday_slot", ["dow", "slot"]),
        ("month_slot", ["month", "slot"]), ("slot", ["slot"]),
    ):
        summary = frame.groupby(keys)["v"].agg(["median", "count"])
        min_count = {"month_weekday_slot": 8, "weekday_slot": 20,
                     "month_slot": 12, "slot": 10}[name]
        maps[name] = {((k,) if not isinstance(k, tuple) else k): float(row["median"])
                      for k, row in summary.iterrows() if row["count"] >= min_count}
    return {"timezone": timezone, "slope": slope, "center": center,
            "intercept": intercept, "maps": maps, "fallback": float(frame["v"].median())}


def expected_residual(index: pd.DatetimeIndex, baseline: dict) -> np.ndarray:
    local, months, dows, slots = _profile_keys(index, baseline["timezone"])
    maps = baseline["maps"]
    seasonal = np.empty(len(index), dtype=float)
    for i, (m, d, q) in enumerate(zip(months, dows, slots)):
        seasonal[i] = maps["month_weekday_slot"].get((int(m), int(d), int(q)),
                        maps["weekday_slot"].get((int(d), int(q)),
                        maps["month_slot"].get((int(m), int(q)),
                        maps["slot"].get((int(q),), baseline["fallback"]))))
    return baseline["intercept"] + baseline["slope"] * (
        fractional_year(local) - baseline["center"]) + seasonal


def previous_local_day(x: pd.Series, timezone: str) -> np.ndarray:
    """Prior day same Berlin wall-clock slot; UTC t-96 if DST slot did not exist."""
    local = x.index.tz_convert(timezone)
    dates = np.asarray(local.date, dtype=object)
    slots = (local.hour * 4 + local.minute // 15).to_numpy(dtype=int)
    lookup = pd.DataFrame({"day": dates, "slot": slots, "x": x.to_numpy(dtype=float)}).groupby(["day", "slot"])["x"].median().to_dict()
    prevkeys = [(d - timedelta(days=1), int(q)) for d, q in zip(dates, slots)]
    result = np.array([lookup.get(k, np.nan) for k in prevkeys], dtype=float)
    missing_slots = np.array([k not in lookup for k in prevkeys], dtype=bool)
    fallback = x.shift(WINDOW_STEPS).to_numpy(dtype=float)
    result[missing_slots] = fallback[missing_slots]
    return result


def make_features(df: pd.DataFrame, baseline: dict) -> tuple[pd.DataFrame, np.ndarray]:
    """Nine physical signals (including capacity utilisation), six calendar cycles."""
    local = df.index.tz_convert(baseline["timezone"])
    r = df["residual"]
    expected = expected_residual(df.index, baseline)
    # Absolute changes (source units), not percent changes: residual can reach zero/negative.
    feats = pd.DataFrame({
        "residual_deviation": r.to_numpy(dtype=float) - expected,
        "residual_ramp_30m": r - r.shift(2),
        "residual_ramp_1h": r - r.shift(4),
        "residual_change_24h": r.to_numpy(dtype=float) - previous_local_day(r, baseline["timezone"]),
        "renewable_share": (df["solar"] + df["wind"]) / df["demand"],
        "wind_change_1h": df["wind"] - df["wind"].shift(4),
        "solar_change_1h": df["solar"] - df["solar"].shift(4),
        "solar_capacity_utilization": 4.0 * df["solar"] / df["solar_capacity_mw"],
        "wind_capacity_utilization": 4.0 * df["wind"] / df["wind_capacity_mw"],
    }, index=df.index)
    minute = local.hour.to_numpy() * 60 + local.minute.to_numpy()
    tod = 2 * np.pi * minute / 1440.0
    dow = 2 * np.pi * local.dayofweek.to_numpy() / 7.0
    year = 2 * np.pi * (local.dayofyear.to_numpy() - 1) / np.where(local.is_leap_year, 366.0, 365.0)
    feats["tod_sin"], feats["tod_cos"] = np.sin(tod), np.cos(tod)
    feats["dow_sin"], feats["dow_cos"] = np.sin(dow), np.cos(dow)
    feats["year_sin"], feats["year_cos"] = np.sin(year), np.cos(year)
    return feats[FEATURE_NAMES].replace([np.inf, -np.inf], np.nan), expected


def valid_window_starts(scaled: np.ndarray, starts: np.ndarray, window: int = WINDOW_STEPS) -> np.ndarray:
    starts = np.asarray(starts, dtype=np.int64)
    starts = starts[(starts >= 0) & (starts + window <= len(scaled))]
    if len(starts) == 0:
        return starts
    bad = (~np.isfinite(scaled).all(axis=1)).astype(np.int8)
    cs = np.concatenate(([0], np.cumsum(bad)))
    return starts[(cs[starts + window] - cs[starts]) == 0]


def extract_windows(scaled: np.ndarray, starts: np.ndarray, window: int = WINDOW_STEPS) -> np.ndarray:
    if len(starts) == 0:
        return np.empty((0, window * scaled.shape[1]), dtype=np.float32)
    view = sliding_window_view(scaled, window_shape=window, axis=0)
    return np.asarray(view[starts].transpose(0, 2, 1), dtype=np.float32).reshape(len(starts), -1)


def fit_statistical_reference(train_features: pd.DataFrame) -> dict:
    medians, scales = [], []
    for f in PHYSICAL_FEATURES:
        m, s = robust_location_scale(train_features[f].to_numpy(dtype=float))
        medians.append(m)
        scales.append(s)
    return {"medians": np.asarray(medians), "scales": np.asarray(scales)}


def statistical_scores(features: pd.DataFrame, reference: dict) -> tuple[np.ndarray, np.ndarray]:
    raw = features[PHYSICAL_FEATURES].to_numpy(dtype=float)
    z = np.abs((raw - reference["medians"]) / reference["scales"])
    score = np.full(len(z), np.nan, dtype=float)
    valid = np.isfinite(z).all(axis=1)
    score[valid] = z[valid, :len(STAT_SCREEN_FEATURES)].max(axis=1)
    return score, z
