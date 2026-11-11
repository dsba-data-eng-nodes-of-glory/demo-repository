"""FastAPI routes for electricity demand results.

Optimised for append-only CSV storage: large result files are read from their
last rows rather than loading the entire file on every dashboard refresh.
"""
from __future__ import annotations

from collections import OrderedDict
from io import BytesIO
from pathlib import Path
from threading import RLock
import time
import logging

import pandas as pd
import yaml
from fastapi import APIRouter, Query

from backend.pipeline_manager import ROOT
from src.storage.forecast_store import storage_backend, read_actuals as cassandra_read_actuals, read_forecasts as cassandra_read_forecasts, read_latest_forecast as cassandra_read_latest_forecast, read_quality_events
from src.storage.renewable_store import read_renewable_forecasts, read_renewable_actuals

router = APIRouter(prefix="/results", tags=["results"])

FORECAST_FILE = ROOT / "data/processed/forecasts.csv"
ACTUAL_FILE = ROOT / "data/processed/actuals.csv"
STATION_CONFIG_FILE = ROOT / "config/stations.yaml"

# Small in-process cache. File mtime/size invalidates entries as files change.
_CACHE: OrderedDict[tuple, tuple[float, tuple[int, int], pd.DataFrame]] = OrderedDict()
_CACHE_LOCK = RLock()
_CACHE_MAX_ITEMS = 12
_RENEWABLE_BASE_CACHE: OrderedDict[tuple, tuple[pd.DataFrame, dict]] = OrderedDict()
_RENEWABLE_CACHE_MAX_ITEMS = 8
_RENEWABLE_CACHE_LOCK = RLock()

# Normalised historical demand is static during replay. Cache the prepared
# frame (UTC timestamp + numeric demand + sorted/deduplicated rows), not only
# the raw CSV, so dashboard refreshes do not re-parse ~168k source rows.
_HISTORICAL_ACTUALS_CACHE: OrderedDict[tuple, pd.DataFrame] = OrderedDict()
_HISTORICAL_ACTUALS_CACHE_LOCK = RLock()
_HISTORICAL_ACTUALS_CACHE_MAX_ITEMS = 8

_LOG = logging.getLogger(__name__)


def _station_output_path(station: str, kind: str) -> Path:
    """Return station-isolated output when configured, else legacy shared CSV."""
    try:
        with STATION_CONFIG_FILE.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}
        value = config.get("stations", {}).get(station, {}).get(f"{kind}_file")
        if value:
            path = Path(value)
            return path if path.is_absolute() else ROOT / path
    except (OSError, ValueError, yaml.YAMLError):
        pass
    return ACTUAL_FILE if kind == "actuals" else FORECAST_FILE


def _file_signature(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


def _cache_get(path: Path, key_suffix: tuple, ttl_seconds: float):
    key = (str(path.resolve()),) + key_suffix
    try:
        signature = _file_signature(path)
    except OSError:
        return key, None
    now = time.monotonic()
    with _CACHE_LOCK:
        item = _CACHE.get(key)
        if item is not None:
            cached_at, cached_signature, frame = item
            if cached_signature == signature and now - cached_at <= ttl_seconds:
                _CACHE.move_to_end(key)
                return key, frame.copy()
    return key, None


def _cache_put(key: tuple, path: Path, frame: pd.DataFrame) -> None:
    try:
        signature = _file_signature(path)
    except OSError:
        return
    with _CACHE_LOCK:
        _CACHE[key] = (time.monotonic(), signature, frame.copy())
        _CACHE.move_to_end(key)
        while len(_CACHE) > _CACHE_MAX_ITEMS:
            _CACHE.popitem(last=False)


def _read_csv_cached(path: Path, ttl_seconds: float = 300.0) -> pd.DataFrame:
    """Read a mostly static CSV once; re-read only when its file signature changes."""
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    key, cached = _cache_get(path, ("all",), ttl_seconds)
    if cached is not None:
        return cached
    try:
        frame = pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError, UnicodeDecodeError):
        return pd.DataFrame()
    _cache_put(key, path, frame)
    return frame


def _read_csv_tail(path: Path, max_rows: int, ttl_seconds: float = 1.0) -> pd.DataFrame:
    """Read only the final max_rows CSV records using reverse block reads.

    This is intended for the flat, one-record-per-line CSV files written by the
    POC. Quoted multi-line CSV fields are not supported by the storage format.
    """
    if max_rows <= 0 or not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()

    max_rows = int(max_rows)
    key, cached = _cache_get(path, ("tail", max_rows), ttl_seconds)
    if cached is not None:
        return cached

    try:
        with path.open("rb") as file:
            header = file.readline().rstrip(b"\r\n")
            header_end = file.tell()
            file.seek(0, 2)
            position = file.tell()
            chunks: list[bytes] = []
            newline_count = 0
            block_size = 64 * 1024

            while position > header_end and newline_count <= max_rows:
                start = max(header_end, position - block_size)
                file.seek(start)
                block = file.read(position - start)
                chunks.insert(0, block)
                newline_count += block.count(b"\n")
                position = start

        body = b"".join(chunks)
        lines = body.splitlines()
        # If reading began in the middle of a record, discard that partial line.
        if position > header_end and lines:
            lines = lines[1:]
        lines = lines[-max_rows:]
        if not lines:
            return pd.DataFrame()

        payload = header + b"\n" + b"\n".join(lines)
        frame = pd.read_csv(BytesIO(payload))
    except (OSError, ValueError, pd.errors.ParserError, UnicodeDecodeError):
        return pd.DataFrame()

    _cache_put(key, path, frame)
    return frame


def _json_safe_records(frame: pd.DataFrame) -> list:
    """Convert a DataFrame into JSON-safe records."""
    if frame.empty:
        return []
    clean = frame.replace([float("inf"), float("-inf")], float("nan"))
    clean = clean.astype(object).where(pd.notna(clean), None)
    return clean.to_dict(orient="records")


def _read_station_rows(
    path: Path,
    station: str,
    limit: int,
    scan_rows: int | None = None,
) -> pd.DataFrame:
    """Read recent rows for one station without loading a large CSV in full."""
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()

    limit = max(1, min(int(limit), 20000))
    # Scan extra rows in case several stations share a single append-only file.
    scan_rows = scan_rows or min(200000, max(50000, limit * 5))
    frame = _read_csv_tail(path, scan_rows)
    if frame.empty or "station_id" not in frame.columns:
        return pd.DataFrame()
    frame = frame[frame["station_id"].astype(str) == str(station)]
    return frame.tail(limit).reset_index(drop=True)


def _read_historical_actuals(station: str) -> pd.DataFrame:
    """Return cached, normalised static actuals for one station.

    The training and test CSVs are immutable during a replay. Their file
    signatures key this cache, so a changed/replaced source is automatically
    reloaded. The returned frame is treated as read-only by callers; merging
    creates a copy only of the small, requested tail.
    """
    if not STATION_CONFIG_FILE.exists():
        return pd.DataFrame()
    try:
        with STATION_CONFIG_FILE.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}
        station_config = config.get("stations", {}).get(station, {})
        input_file = station_config.get("input_file")
        history_file = station_config.get("history_file")
        if not input_file and not history_file:
            return pd.DataFrame()

        def configured_path(value: str | Path) -> Path:
            path = Path(value).expanduser()
            return path if path.is_absolute() else ROOT / path

        sources: list[Path] = []
        # Keep the configured history seed authoritative. Older station
        # entries can still use the legacy *_history.csv sibling convention.
        if history_file:
            sources.append(configured_path(history_file))
        elif input_file:
            input_path = configured_path(input_file)
            derived_history = input_path.with_name(
                f"{input_path.stem}_history{input_path.suffix}"
            )
            if derived_history.exists():
                sources.append(derived_history)
        if input_file:
            sources.append(configured_path(input_file))

        unique_sources: list[Path] = []
        seen_paths: set[Path] = set()
        for source_path in sources:
            resolved = source_path.resolve()
            if resolved not in seen_paths:
                unique_sources.append(resolved)
                seen_paths.add(resolved)

        # Include mtime_ns and size so replacing or appending a source creates
        # a new cache key. Missing paths also participate in invalidation.
        signatures = []
        for source_path in unique_sources:
            try:
                stat = source_path.stat()
                signatures.append((str(source_path), stat.st_mtime_ns, stat.st_size))
            except OSError:
                signatures.append((str(source_path), None, None))
        cache_key = (str(station), tuple(signatures))

        # Serialise cache misses: concurrent first page loads should not parse
        # the same large CSVs multiple times.
        with _HISTORICAL_ACTUALS_CACHE_LOCK:
            cached = _HISTORICAL_ACTUALS_CACHE.get(cache_key)
            if cached is not None:
                _HISTORICAL_ACTUALS_CACHE.move_to_end(cache_key)
                return cached

            frames = []
            for source_path in unique_sources:
                if not source_path.exists() or source_path.stat().st_size == 0:
                    continue
                # Do not also retain a second raw copy in the generic CSV
                # cache: this station-level cache owns the prepared frame.
                frame = pd.read_csv(source_path)
                if frame.empty or "utc_timestamp" not in frame.columns:
                    continue
                if "actual_demand" not in frame.columns and "demand" in frame.columns:
                    frame = frame.rename(columns={"demand": "actual_demand"})
                if "actual_demand" not in frame.columns:
                    continue
                frame = frame.copy()
                frame["station_id"] = station
                # Source files use day-first timestamps (for example,
                # 30-09-2026 23:45); keep the mixed parser used by this project.
                frame["_timestamp_key"] = pd.to_datetime(
                    frame["utc_timestamp"].astype("string").str.strip(),
                    format="mixed",
                    dayfirst=True,
                    utc=True,
                    errors="coerce",
                )
                frame["actual_demand"] = pd.to_numeric(
                    frame["actual_demand"], errors="coerce"
                )
                frame = frame.dropna(subset=["_timestamp_key", "actual_demand"])
                if not frame.empty:
                    frames.append(frame)

            if not frames:
                prepared = pd.DataFrame()
            else:
                prepared = pd.concat(frames, ignore_index=True, sort=False)
                # Stable sorting preserves source order for duplicate times;
                # the later configured source continues to win, as before.
                prepared = prepared.sort_values("_timestamp_key", kind="mergesort")
                prepared = prepared.drop_duplicates(
                    ["station_id", "_timestamp_key"], keep="last"
                )
                prepared["utc_timestamp"] = prepared["_timestamp_key"].dt.strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                )
                prepared = prepared.reset_index(drop=True)

            _HISTORICAL_ACTUALS_CACHE[cache_key] = prepared
            _HISTORICAL_ACTUALS_CACHE.move_to_end(cache_key)
            while len(_HISTORICAL_ACTUALS_CACHE) > _HISTORICAL_ACTUALS_CACHE_MAX_ITEMS:
                _HISTORICAL_ACTUALS_CACHE.popitem(last=False)
            return prepared
    except (OSError, ValueError, yaml.YAMLError, pd.errors.ParserError):
        return pd.DataFrame()


def _combine_actuals(
    historical: pd.DataFrame,
    live: pd.DataFrame,
    station: str,
    limit: int,
    end_time: str | None = None,
) -> pd.DataFrame:
    """Merge static and live actuals, preferring live values at duplicate times.

    Cached historical frames already carry a parsed, sorted ``_timestamp_key``.
    Only their final ``limit`` rows are copied for each request; this is enough
    to calculate the final top-N rows after adding live observations. Raw
    DataFrames passed by tests/other callers remain supported.
    """
    limit = max(1, min(int(limit), 20000))

    cutoff = None
    if end_time:
        parsed_cutoff = pd.to_datetime(end_time, utc=True, errors="coerce")
        if pd.notna(parsed_cutoff):
            cutoff = parsed_cutoff
    elif not live.empty and "utc_timestamp" in live.columns:
        live_times = pd.to_datetime(
            live["utc_timestamp"].astype("string").str.strip(),
            format="mixed",
            dayfirst=True,
            utc=True,
            errors="coerce",
        ).dropna()
        if not live_times.empty:
            # On replay, the newest live timestamp is the point-in-time cutoff.
            cutoff = live_times.max()

    frames = []

    if not historical.empty:
        if "_timestamp_key" in historical.columns:
            # This is the prepared immutable cache. Filter before copying and
            # do not parse the historical timestamp strings again.
            hist = historical
            if cutoff is not None:
                hist = hist.loc[hist["_timestamp_key"] <= cutoff]
            hist = hist.tail(limit).copy()
        else:
            # Backwards-compatible path for raw DataFrames supplied directly.
            hist = historical.copy()
            if "utc_timestamp" not in hist.columns:
                hist = pd.DataFrame()
            else:
                if "actual_demand" not in hist.columns and "demand" in hist.columns:
                    hist = hist.rename(columns={"demand": "actual_demand"})
                if "actual_demand" in hist.columns:
                    hist["_timestamp_key"] = pd.to_datetime(
                        hist["utc_timestamp"].astype("string").str.strip(),
                        format="mixed",
                        dayfirst=True,
                        utc=True,
                        errors="coerce",
                    )
                    hist["actual_demand"] = pd.to_numeric(
                        hist["actual_demand"], errors="coerce"
                    )
                    hist = hist.dropna(subset=["_timestamp_key", "actual_demand"])
                    if cutoff is not None:
                        hist = hist.loc[hist["_timestamp_key"] <= cutoff]
                    hist = hist.sort_values("_timestamp_key", kind="mergesort").tail(limit)
                else:
                    hist = pd.DataFrame()
        if not hist.empty:
            hist["station_id"] = station
            hist["_priority"] = 0
            frames.append(hist)

    if not live.empty:
        live_frame = live.copy()
        if "utc_timestamp" in live_frame.columns:
            if "actual_demand" not in live_frame.columns and "demand" in live_frame.columns:
                live_frame = live_frame.rename(columns={"demand": "actual_demand"})
            if "actual_demand" in live_frame.columns:
                live_frame["station_id"] = station
                live_frame["_timestamp_key"] = pd.to_datetime(
                    live_frame["utc_timestamp"].astype("string").str.strip(),
                    format="mixed",
                    dayfirst=True,
                    utc=True,
                    errors="coerce",
                )
                live_frame["actual_demand"] = pd.to_numeric(
                    live_frame["actual_demand"], errors="coerce"
                )
                live_frame = live_frame.dropna(
                    subset=["_timestamp_key", "actual_demand"]
                )
                if cutoff is not None:
                    live_frame = live_frame.loc[live_frame["_timestamp_key"] <= cutoff]
                live_frame = live_frame.sort_values(
                    "_timestamp_key", kind="mergesort"
                ).tail(limit)
                if not live_frame.empty:
                    live_frame["_priority"] = 1
                    frames.append(live_frame)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True, sort=False)
    if combined.empty:
        return pd.DataFrame()

    combined = combined.sort_values(
        ["_timestamp_key", "_priority"], kind="mergesort"
    )
    combined = combined.drop_duplicates(
        ["station_id", "_timestamp_key"], keep="last"
    )
    combined["utc_timestamp"] = combined["_timestamp_key"].dt.strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    combined = combined.sort_values("_timestamp_key").tail(limit)
    combined = combined.drop(
        columns=["_timestamp_key", "_priority"], errors="ignore"
    )
    return combined.reset_index(drop=True)


@router.get("/summary")
def summary(station: str | None = None):
    if storage_backend() == "cassandra":
        if station:
            records = cassandra_read_latest_forecast(station)
        else:
            try:
                with STATION_CONFIG_FILE.open("r", encoding="utf-8") as handle:
                    station_cfg = yaml.safe_load(handle) or {}
                candidates = station_cfg.get("active_stations") or [station_cfg.get("active_station")]
                all_records = [row for name in candidates if name for row in cassandra_read_latest_forecast(name)]
                records = sorted(all_records, key=lambda row: row.get("forecast_issued_at_utc", ""))[-96:]
            except (OSError, ValueError, yaml.YAMLError):
                records = []
        if not records:
            return {"available": False, "rows": 0, "latest_forecast": None}
        return {"available": True, "rows": len(records), "latest_forecast": records[0]}
    if station:
        frame = _read_station_rows(_station_output_path(station, "forecasts"), station, 1)
    else:
        frame = _read_csv_tail(FORECAST_FILE, 1)
    if frame.empty:
        return {"available": False, "rows": 0, "latest_forecast": None}
    latest = _json_safe_records(frame.tail(1))[0]
    return {"available": True, "rows": None, "latest_forecast": latest}


@router.get("/forecasts")
def forecasts(
    station: str,
    limit: int = Query(default=5000, ge=1, le=20000),
):
    if storage_backend() == "cassandra":
        records = cassandra_read_forecasts(station, limit)
        return {"records": records, "message": None if records else "No forecast records found for this station"}
    frame = _read_station_rows(_station_output_path(station, "forecasts"), station, limit)
    if frame.empty:
        return {"records": [], "message": "No forecast records found for this station"}
    return {"records": _json_safe_records(frame)}


@router.get("/actuals")
def actuals(
    station: str,
    limit: int = Query(default=5000, ge=1, le=20000),
    end_time: str | None = Query(default=None, description="Optional UTC cutoff, normally the forecast origin"),
):
    historical = _read_historical_actuals(station)
    if storage_backend() == "cassandra":
        live = pd.DataFrame(cassandra_read_actuals(station, limit))
    else:
        live = _read_station_rows(_station_output_path(station, "actuals"), station, limit)
    frame = _combine_actuals(
        historical=historical,
        live=live,
        station=station,
        limit=limit,
        end_time=end_time,
    )
    if frame.empty:
        return {"records": [], "message": "No actual observations found for this station"}
    return {"records": _json_safe_records(frame)}


@router.get("/quality/events")
def quality_events(
    station: str,
    limit: int = Query(default=100, ge=1, le=1000),
):
    """Return recent rejected records, timestamp gaps and quality warnings."""
    return {"records": read_quality_events(station, limit)}


def _load_configured_renewable_actuals(station: str, end_time: str | None = None) -> tuple[pd.DataFrame, dict]:
    """Return renewable actuals with an mtime-keyed cache of static CSV sources.

    The train/test source files are static during replay. Read and normalise them
    once per file signature; per-refresh work is then only a cheap time filter.
    """
    if not STATION_CONFIG_FILE.exists():
        return pd.DataFrame(), {"message": "Station configuration is unavailable."}
    try:
        with STATION_CONFIG_FILE.open("r", encoding="utf-8") as handle:
            station_doc = yaml.safe_load(handle) or {}
        cfg = station_doc.get("stations", {}).get(station, {}).get("renewables", {}) or {}
    except (OSError, ValueError, yaml.YAMLError):
        return pd.DataFrame(), {"message": "Could not read renewable station configuration."}
    if not cfg.get("enabled"):
        return pd.DataFrame(), {"message": "Renewable monitoring is not enabled for this station."}

    source_defs = {
        "solar": {"target": "Photovoltaics [MWh]", "api": "solar_actual_mwh"},
        "wind": {"target": "Wind onshore [MWh]", "api": "wind_actual_mwh"},
    }
    paths_by_kind: dict[str, list[Path]] = {}
    signature_parts: list[tuple] = []
    for kind in ("solar", "wind"):
        kind_cfg = cfg.get(kind, {}) or {}
        paths: list[Path] = []
        for key in ("train_file", "test_file"):
            value = kind_cfg.get(key)
            if not value:
                continue
            path = Path(value)
            if not path.is_absolute():
                path = ROOT / path
            paths.append(path)
        paths_by_kind[kind] = paths
        for path in paths:
            try:
                stat = path.stat()
                signature_parts.append((str(path.resolve()), stat.st_mtime_ns, stat.st_size))
            except OSError:
                signature_parts.append((str(path.resolve()), None, None))

    cache_key = (str(station), tuple(signature_parts))
    with _RENEWABLE_CACHE_LOCK:
        cached = _RENEWABLE_BASE_CACHE.get(cache_key)
        if cached is not None:
            base_frame, base_coverage = cached
            _RENEWABLE_BASE_CACHE.move_to_end(cache_key)
            base_frame = base_frame.copy()
            base_coverage = {key: dict(value) for key, value in base_coverage.items()}
        else:
            series_frames: list[pd.DataFrame] = []
            base_coverage: dict = {}
            for kind, definition in source_defs.items():
                frames: list[pd.DataFrame] = []
                paths = paths_by_kind[kind]
                for source_path in paths:
                    if not source_path.is_file():
                        continue
                    try:
                        # The dashboard only plots recent actual history (72h by default).
                        # Read a bounded tail from these static chronological CSVs, not
                        # every row in the multi-year training files.
                        frame = _read_csv_tail(source_path, 25000, ttl_seconds=300.0)
                        if frame.empty:
                            continue
                        frame.columns = [
                            str(column).replace("\ufeff", "").replace("\ufffd", "")
                            .replace("\xa0", " ").strip()
                            for column in frame.columns
                        ]
                        if "utc_timestamp" not in frame.columns or definition["target"] not in frame.columns:
                            _LOG.warning("Renewable source %s lacks expected columns for %s", source_path, kind)
                            continue
                        frame = frame[["utc_timestamp", definition["target"]]].copy()
                        frame["utc_timestamp"] = pd.to_datetime(
                            frame["utc_timestamp"].astype("string").str.strip(),
                            format="mixed", dayfirst=True, utc=True, errors="coerce",
                        )
                        # The supplied solar CSV quotes thousands-formatted values
                        # such as "2,145.52". Strip grouping commas before numeric cast.
                        target_values = frame[definition["target"]].astype("string").str.replace(",", "", regex=False)
                        frame[definition["target"]] = pd.to_numeric(target_values, errors="coerce")
                        frame = frame.dropna(subset=["utc_timestamp"])
                        frames.append(frame)
                    except (OSError, ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
                        _LOG.exception("Could not load renewable source %s: %s", source_path, exc)

                if frames:
                    current = pd.concat(frames, ignore_index=True).sort_values("utc_timestamp")
                    current = current.drop_duplicates("utc_timestamp", keep="last")
                    current = current.rename(columns={definition["target"]: definition["api"]})
                    series_frames.append(current)
                    base_coverage[kind] = {
                        "rows": int(current[definition["api"]].notna().sum()),
                        "latest_actual_utc": current["utc_timestamp"].max().isoformat() if not current.empty else None,
                        "configured_test_file": str(paths[1]) if len(paths) > 1 else None,
                        "test_file_available": bool(len(paths) > 1 and paths[1].is_file()),
                    }
                else:
                    base_coverage[kind] = {
                        "rows": 0, "latest_actual_utc": None,
                        "configured_test_file": str(paths[1]) if len(paths) > 1 else None,
                        "test_file_available": bool(len(paths) > 1 and paths[1].is_file()),
                    }

            if series_frames:
                base_frame = series_frames[0]
                for frame in series_frames[1:]:
                    base_frame = base_frame.merge(frame, on="utc_timestamp", how="outer")
                base_frame = base_frame.sort_values("utc_timestamp").tail(20000).reset_index(drop=True)
            else:
                base_frame = pd.DataFrame()

            # Retain a small number of station/source versions to bound memory.
            _RENEWABLE_BASE_CACHE[cache_key] = (base_frame.copy(), {k: dict(v) for k, v in base_coverage.items()})
            _RENEWABLE_BASE_CACHE.move_to_end(cache_key)
            while len(_RENEWABLE_BASE_CACHE) > _RENEWABLE_CACHE_MAX_ITEMS:
                _RENEWABLE_BASE_CACHE.popitem(last=False)

    cutoff = pd.to_datetime(end_time, utc=True, errors="coerce") if end_time else None
    current = base_frame
    if cutoff is not None and pd.notna(cutoff) and not current.empty:
        current = current[current["utc_timestamp"] <= cutoff].copy()
    else:
        current = current.copy()
    coverage = {key: dict(value) for key, value in base_coverage.items()}
    for kind, definition in source_defs.items():
        details = coverage.setdefault(kind, {"rows": 0, "latest_actual_utc": None})
        if not current.empty and definition["api"] in current.columns:
            values = current.dropna(subset=[definition["api"]])
            details["rows"] = int(len(values))
            details["latest_actual_utc"] = values["utc_timestamp"].max().isoformat() if not values.empty else None
        else:
            details["rows"] = 0
            details["latest_actual_utc"] = None
    return current, coverage


def _latest_renewable_run(
    records: pd.DataFrame,
    horizon_steps: int = 96,
    interval_minutes: int = 15,
) -> tuple[pd.DataFrame, bool]:
    """Prefer the latest *complete* run, falling back to the newest partial run.

    Append-only forecast files may contain interrupted restarts: selecting only
    the maximum origin can replace a usable 96-step forecast with 3 steps.
    Group by origin/run ID; never mix rows from different forecast origins.
    """
    if records is None or records.empty:
        return pd.DataFrame(), False
    if horizon_steps <= 0 or interval_minutes <= 0:
        raise ValueError("horizon_steps and interval_minutes must be positive")
    frame = records.copy()
    required = {"forecast_origin_utc", "forecast_timestamp_utc"}
    if not required.issubset(frame):
        return pd.DataFrame(), False
    for name in ("forecast_origin_utc", "forecast_timestamp_utc", "forecast_issued_at_utc"):
        if name in frame:
            frame[name] = pd.to_datetime(frame[name], utc=True, errors="coerce")
    frame = frame.dropna(subset=["forecast_origin_utc", "forecast_timestamp_utc"])
    if frame.empty:
        return frame, False
    if "horizon_step" in frame:
        frame["horizon_step"] = pd.to_numeric(frame["horizon_step"], errors="coerce")
    else:
        step = (frame["forecast_timestamp_utc"] - frame["forecast_origin_utc"]).dt.total_seconds() / (interval_minutes * 60)
        frame["horizon_step"] = step
    frame = frame[frame["horizon_step"].notna() & frame["horizon_step"].between(1, horizon_steps)]
    frame = frame[frame["horizon_step"].mod(1).eq(0)]
    if frame.empty:
        return frame, False
    # Use a run ID when one is available; otherwise origin uniquely identifies a run.
    run_cols = ["forecast_origin_utc"]
    if "forecast_run_id" in frame and frame["forecast_run_id"].notna().any():
        frame["forecast_run_id"] = frame["forecast_run_id"].fillna("").astype(str)
        run_cols.append("forecast_run_id")
    groups = []
    for _key, group in frame.groupby(run_cols, dropna=False, sort=False):
        group = group.sort_values("forecast_timestamp_utc").drop_duplicates("horizon_step", keep="last")
        origin = group["forecast_origin_utc"].max()
        actual = group.sort_values("horizon_step")["forecast_timestamp_utc"]
        expected = origin + pd.to_timedelta(pd.RangeIndex(1, horizon_steps + 1) * interval_minutes, unit="min")
        steps_complete = (len(group) == horizon_steps and
                          set(group["horizon_step"].astype(int)) == set(range(1, horizon_steps + 1)))
        complete = bool(steps_complete and actual.reset_index(drop=True).equals(pd.Series(expected)))
        issued = group["forecast_issued_at_utc"].max() if "forecast_issued_at_utc" in group else origin
        groups.append((origin, issued, complete, group.sort_values("forecast_timestamp_utc")))
    groups.sort(key=lambda r: (r[0], r[1]), reverse=True)
    for _origin, _issued, complete, group in groups:
        if complete:
            return group.reset_index(drop=True), True
    return groups[0][3].reset_index(drop=True), False


# Cold loading multi-year solar/wind reference CSVs may take several
# seconds on an HDD. Opt-in dashboard requests warm that cache off-thread,
# while ordinary API and test calls retain the original synchronous contract.
from concurrent.futures import ThreadPoolExecutor
_RENEWABLE_WARM_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix='renewable-cache')
_RENEWABLE_WARM_LOCK = RLock()
_RENEWABLE_WARM_JOBS: dict[str, object] = {}

def _renewable_warmed_or_schedule(station: str) -> bool:
    with _RENEWABLE_WARM_LOCK:
        job = _RENEWABLE_WARM_JOBS.get(station)
        if job is None:
            _RENEWABLE_WARM_JOBS[station] = _RENEWABLE_WARM_POOL.submit(
                _load_configured_renewable_actuals, station, None
            )
            return False
        if not job.done():
            return False
        try:
            job.result()
            return True
        except Exception as exc:
            _LOG.warning('Renewable history warm-up failed for %s: %s', station, exc)
            del _RENEWABLE_WARM_JOBS[station]
            return False


@router.get("/renewables")
def renewables(
    station: str,
    limit: int = Query(default=5000, ge=1, le=20000),
    end_time: str | None = Query(default=None, description="Optional UTC cutoff for actuals and forecast issue time"),
    defer_heavy: bool = False,
):
    """Return solar/wind actual series and the latest saved per-resource forecasts."""
    if defer_heavy and not _renewable_warmed_or_schedule(station):
        return {'station': station, 'pending': True, 'actuals': [],
                'forecasts': {'solar': [], 'wind': []},
                'message': 'Loading Solar and Wind history in the background.'}
    actual_frame, coverage = _load_configured_renewable_actuals(station, end_time=end_time)
    actual_records = _json_safe_records(actual_frame.tail(limit)) if not actual_frame.empty else []
    forecast_value = None
    try:
        with STATION_CONFIG_FILE.open("r", encoding="utf-8") as handle:
            cfg = (yaml.safe_load(handle) or {}).get("stations", {}).get(station, {})
        forecast_value = cfg.get("renewable_forecasts_file")
    except (OSError, ValueError, yaml.YAMLError):
        forecast_value = None
    forecast_path = Path(forecast_value) if forecast_value else ROOT / "data" / "processed" / "stations" / station / "renewable_forecasts.csv"
    if not forecast_path.is_absolute():
        forecast_path = ROOT / forecast_path
    forecast_groups = {"solar": [], "wind": []}
    forecast_run_complete = {"solar": False, "wind": False}
    forecast_rows_scanned = 0
    forecast_file_bytes = forecast_path.stat().st_size if forecast_path.is_file() and storage_backend() == "csv" else 0
    if storage_backend() == "cassandra":
        # The same selection/completeness logic is applied below to DB rows.
        forecast_rows = read_renewable_forecasts(station, limit=min(20000, max(2000, int(limit) * 2)), end_time=end_time)
        forecasts_frame = pd.DataFrame(forecast_rows)
        forecast_rows_scanned = len(forecasts_frame)
        if not forecasts_frame.empty:
            for col in ("forecast_issued_at_utc", "forecast_origin_utc", "forecast_timestamp_utc"):
                if col in forecasts_frame:
                    forecasts_frame[col] = pd.to_datetime(forecasts_frame[col], utc=True, errors="coerce")
            for kind in ("solar", "wind"):
                selected = forecasts_frame[forecasts_frame["generation_type"] == kind]
                if not selected.empty:
                    selected, complete = _latest_renewable_run(selected)
                    forecast_run_complete[kind] = complete
                    forecast_groups[kind] = _json_safe_records(selected)
    elif forecast_path.is_file() and forecast_file_bytes:
        try:
            # renewable_forecasts.csv is append-only. Read its tail, not the full
            # history, because the dashboard only displays the most recent run.
            scan_limit = min(20000, max(2000, int(limit) * 2))
            forecasts_frame = _read_csv_tail(forecast_path, scan_limit, ttl_seconds=1.0)
            forecast_rows_scanned = int(len(forecasts_frame))
            for col in ("forecast_issued_at_utc", "forecast_origin_utc", "forecast_timestamp_utc"):
                if col in forecasts_frame.columns:
                    forecasts_frame[col] = pd.to_datetime(forecasts_frame[col], utc=True, errors="coerce")
            forecasts_frame = forecasts_frame[forecasts_frame["station_id"].astype(str) == str(station)]
            if end_time and "forecast_origin_utc" in forecasts_frame.columns:
                cutoff = pd.to_datetime(end_time, utc=True, errors="coerce")
                if pd.notna(cutoff):
                    forecasts_frame = forecasts_frame[forecasts_frame["forecast_origin_utc"] <= cutoff]
            for kind in ("solar", "wind"):
                selected = forecasts_frame[forecasts_frame["generation_type"] == kind]
                if selected.empty:
                    continue
                # If the producer stopped mid-horizon, prefer the latest earlier
                # completed run instead of graphing a misleading short forecast.
                selected, complete = _latest_renewable_run(selected)
                forecast_run_complete[kind] = complete
                forecast_groups[kind] = _json_safe_records(selected)
        except (OSError, ValueError, KeyError, pd.errors.ParserError):
            pass
    # Overlay actual generation acknowledged by the replay; training/test
    # data remain static reference sources and may be missing after their end.
    generation_rows = pd.DataFrame(read_renewable_actuals(station, limit=limit, end_time=end_time))
    if not generation_rows.empty:
        generation_rows["utc_timestamp"] = pd.to_datetime(generation_rows["utc_timestamp"], utc=True, errors="coerce")
        generation_rows["actual_generation_mwh"] = pd.to_numeric(generation_rows["actual_generation_mwh"], errors="coerce")
        pivot = generation_rows.pivot_table(index="utc_timestamp", columns="generation_type",
                                            values="actual_generation_mwh", aggfunc="last").reset_index()
        pivot.columns.name = None
        pivot = pivot.rename(columns={"solar": "solar_actual_mwh", "wind": "wind_actual_mwh"})
        if not actual_frame.empty:
            actual_frame = pd.concat([actual_frame, pivot], ignore_index=True).sort_values("utc_timestamp")
            actual_frame = actual_frame.groupby("utc_timestamp", as_index=False).last()
        else:
            actual_frame = pivot
        actual_records = _json_safe_records(actual_frame.tail(limit))
    diagnostics = {
        "forecast_file_exists": bool(forecast_path.is_file()) if storage_backend() == "csv" else None,
        "forecast_file_bytes": int(forecast_file_bytes),
        "forecast_rows_scanned_from_tail": int(forecast_rows_scanned),
        "solar_forecast_points": len(forecast_groups["solar"]),
        "wind_forecast_points": len(forecast_groups["wind"]),
        "solar_forecast_complete": forecast_run_complete["solar"],
        "wind_forecast_complete": forecast_run_complete["wind"],
        "solar_latest_origin_utc": forecast_groups["solar"][0].get("forecast_origin_utc") if forecast_groups["solar"] else None,
        "wind_latest_origin_utc": forecast_groups["wind"][0].get("forecast_origin_utc") if forecast_groups["wind"] else None,
    }
    _LOG.info(
        "Renewables API station=%s actual_rows=%s forecast_tail_rows=%s solar_forecasts=%s wind_forecasts=%s file_bytes=%s",
        station, len(actual_records), forecast_rows_scanned,
        len(forecast_groups["solar"]), len(forecast_groups["wind"]), forecast_file_bytes,
    )
    return {
        "station": station, "actuals": actual_records, "forecasts": forecast_groups,
        "coverage": coverage, "forecast_storage": "cassandra" if storage_backend() == "cassandra" else str(forecast_path),
        "diagnostics": diagnostics,
    }


# ---------------------------------------------------------------------------
# Operational model analysis: point-in-time matched generation and durable AE
# forecast risk. These new endpoints leave the existing results API unchanged.
# ---------------------------------------------------------------------------

def _station_configuration(station: str) -> dict:
    from fastapi import HTTPException
    with STATION_CONFIG_FILE.open("r", encoding="utf-8") as handle:
        stations = (yaml.safe_load(handle) or {}).get("stations", {})
    if station not in stations:
        raise HTTPException(status_code=404, detail=f"Unknown station: {station}")
    return stations[station]


def _renewable_file(station: str, cfg: dict) -> Path:
    value = cfg.get("renewable_forecasts_file") or f"data/processed/stations/{station}/renewable_forecasts.csv"
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _actual_watermark(station: str) -> pd.Timestamp | None:
    """Use actual data *observed by the pipeline*, not the test CSV's last row."""
    if storage_backend() == "cassandra":
        recorded = pd.DataFrame(cassandra_read_actuals(station, 5))
    else:
        recorded = _read_station_rows(_station_output_path(station, "actuals"), station, 5)
    if recorded.empty or "utc_timestamp" not in recorded:
        return None
    latest = pd.to_datetime(recorded["utc_timestamp"], utc=True, errors="coerce").dropna()
    return latest.max() if len(latest) else None


@router.get("/renewables/performance")
def renewables_performance(
    station: str,
    horizon_step: int = Query(default=1, ge=1, le=96),
    limit: int = Query(default=300, ge=1, le=1000),
):
    """Match matured forecast targets to observed solar/wind, replay-safe.

    In this dataset renewable observations come from registered historical
    train/test CSVs. Demand actuals' streaming watermark gates the matches, so
    future static test rows are NEVER reported as live measurements.
    """
    from src.analytics.renewable_performance import match_renewable_forecasts, renewable_metrics

    cfg = _station_configuration(station)
    watermark = _actual_watermark(station)
    if watermark is None:
        return {"station": station, "horizon_step": horizon_step,
                "watermark_utc": None, "source": "historical_generation_replay_aligned",
                "message": "No acknowledged demand actual timestamp yet. Live forecast accuracy is unavailable.",
                "models": {kind: {"metrics": renewable_metrics(pd.DataFrame()), "records": []} for kind in ("solar", "wind")}}
    forecast_file = _renewable_file(station, cfg)
    forecast = (pd.DataFrame(read_renewable_forecasts(station, limit=20000))
                if storage_backend() == "cassandra" else
                _read_csv_tail(forecast_file, 20000, ttl_seconds=2.0))
    actuals, _ = _load_configured_renewable_actuals(station, end_time=watermark.isoformat())
    generation_rows = pd.DataFrame(read_renewable_actuals(station, limit=20000, end_time=watermark.isoformat()))
    if not generation_rows.empty:
        generation_rows["utc_timestamp"] = pd.to_datetime(generation_rows["utc_timestamp"], utc=True, errors="coerce")
        generation_rows["actual_generation_mwh"] = pd.to_numeric(generation_rows["actual_generation_mwh"], errors="coerce")
        pivot = generation_rows.pivot_table(index="utc_timestamp", columns="generation_type",
                                            values="actual_generation_mwh", aggfunc="last").reset_index()
        pivot.columns.name = None
        pivot = pivot.rename(columns={"solar": "solar_actual_mwh", "wind": "wind_actual_mwh"})
        if not actuals.empty:
            actuals = pd.concat([actuals, pivot], ignore_index=True).sort_values("utc_timestamp").groupby("utc_timestamp", as_index=False).last()
        else:
            actuals = pivot
    models = {}
    for kind in ("solar", "wind"):
        matched = match_renewable_forecasts(forecast, actuals, kind, watermark,
                                           horizon_step=horizon_step, limit=limit)
        models[kind] = {"metrics": renewable_metrics(matched),
                        "records": _json_safe_records(matched)}
    return {"station": station, "horizon_step": horizon_step,
            "watermark_utc": watermark.isoformat(),
            "source": "historical_generation_replay_aligned",
            "message": "Generation observations come from registered historical data, restricted to the live demand replay watermark. Connect observed generation ingestion for real production accuracy.",
            "models": models}


@router.get("/residual/forecast")
def residual_forecast_analysis(
    station: str,
    history_runs: int = Query(default=24, ge=1, le=100),
    cached_only: bool = False,
    origin_utc: str | None = None,
    lookback_minutes: int = 0,
):
    """Forecast-only residual-load anomaly analysis for all three models.

    No future actual demand or renewable-generation value is used. Results
    are snapshot-persisted by model input/threshold fingerprint.
    """
    # Monitoring must be fast even on first load or while the predictor is busy.
    # This opt-in path reads ONLY an already scored assessment from SQLite;
    # it never re-runs the Dense Autoencoder in an HTTP worker.
    if cached_only:
        from src.models.residual.snapshots import candidate_snapshots, recent_snapshots, snapshot_path
        from src.models.residual.service import settings_for
        cfg = _station_configuration(station)
        store = snapshot_path(ROOT, station)
        expected_origin = None
        if origin_utc:
            parsed = pd.to_datetime(origin_utc, utc=True, errors='coerce')
            if pd.isna(parsed):
                from fastapi import HTTPException
                raise HTTPException(status_code=422, detail='origin_utc must be an ISO UTC timestamp')
            expected_origin = parsed.isoformat()
        if lookback_minutes < 0 or lookback_minutes > 240:
            from fastapi import HTTPException
            raise HTTPException(status_code=422, detail='lookback_minutes must be between 0 and 240')
        # Demand arrives every 15 minutes while renewables/residuals may run
        # hourly. Inspect pre-scored candidates, never run inference in FastAPI.
        # Also handle multiple calibrations for the same forecast origin.
        candidates = candidate_snapshots(
            store, origin=expected_origin, lookback_minutes=lookback_minutes, limit=64
        )
        # Check explicit user overrides before showing a saved assessment as
        # current. Changing a threshold requires fresh predictor-side scoring.
        overrides = settings_for(cfg)
        def matches_settings(records):
            if not records:
                return False
            first = records[0]
            for option, field in (('stat_threshold','statistical_threshold'), ('ae_threshold','autoencoder_threshold')):
                expected = overrides.get(option)
                if expected is not None and abs(float(first.get(field, -1)) - float(expected)) > 1e-8:
                    return False
            any_override = overrides.get('stat_threshold') is not None or overrides.get('ae_threshold') is not None
            if (first.get('threshold_source') == 'station_override') != any_override:
                return False
            for row in records:
                if row.get('classification') not in ('YES', 'NO'):
                    continue
                sf, af = bool(row.get('statistical_flag')), bool(row.get('autoencoder_flag'))
                # Older saved rows omit component flags; the confirmation field
                # carries equivalent information.
                if 'statistical_flag' not in row:
                    sf = row.get('confirmation') in ('STAT_ONLY','CONFIRMED')
                if 'autoencoder_flag' not in row:
                    af = row.get('confirmation') in ('AE_ONLY','CONFIRMED')
                detected = (sf and af) if overrides.get('decision_rule') == 'both' else (sf or af)
                if (row['classification'] == 'YES') != detected:
                    return False
                if row['classification'] == 'YES':
                    upward = detected and (not overrides.get('upward_only', True)
                        or float(row.get('residual_ramp_30m') or 0) > 0)
                    if bool(row.get('upward_highlight')) != upward:
                        return False
            return True
        for saved in candidates:
            if not matches_settings(saved['records']):
                continue
            lag_minutes = None
            if expected_origin is not None:
                lag_minutes = int(round((pd.Timestamp(expected_origin) -
                    pd.Timestamp(saved['origin_utc'])).total_seconds() / 60))
            return {'available': True, 'station': station, **saved,
                    'settings': overrides, 'history': recent_snapshots(store, history_runs),
                    'requested_origin_utc': expected_origin, 'snapshot_lag_minutes': lag_minutes,
                    'is_exact_origin': lag_minutes in (None, 0),
                    'message': ('Saved assessment from last aligned renewable forecast; '
                                'no recalculation during dashboard refresh.')}
        if expected_origin is not None:
            explanation = ('No compatible residual assessment at or within '
                           f'{lookback_minutes} minutes before Demand origin {expected_origin}. '
                           'Demand updates more frequently than renewable/residual scoring; '
                           'also verify the active station thresholds.')
        else:
            explanation = ('No compatible saved residual assessment. Check active station '
                           'thresholds and predictor scoring logs.')
        return {'available': False, 'station': station, 'records': [],
                'history': recent_snapshots(store, history_runs), 'pending': True,
                'message': explanation}
    from src.models.residual.service import score_and_store
    cfg = _station_configuration(station)
    demand = forecasts(station, limit=4096).get('records', [])
    renewable_file = _renewable_file(station,cfg)
    if storage_backend() == 'cassandra':
        renewable = read_renewable_forecasts(station,limit=4096)
    else:
        renewable = _json_safe_records(_read_csv_tail(renewable_file,4096,ttl_seconds=1.0))
    if not demand or not renewable:
        return {'available':False,'station':station,'records':[],
                'message':'Waiting for synchronized Demand, Solar and Wind forecasts.'}
    origin_sets = __import__('src.models.residual.operational',fromlist=['available_origins']).available_origins(demand,renewable)
    if not origin_sets:
        return {'available':False,'station':station,'records':[],
                'message':'Demand, Solar and Wind do not yet have a shared forecast origin.'}
    # Up to the latest valid origin; historical actuals are always capped at that
    # origin. Static future test rows are never allowed to feed the scorer.
    history = actuals(station,limit=1200,end_time=origin_sets[0].isoformat()).get('records',[])
    return score_and_store(root=ROOT,station=station,cfg=cfg,
                           demand_forecasts=demand,renewable_forecasts=renewable,
                           demand_history=history,history_runs=history_runs)


