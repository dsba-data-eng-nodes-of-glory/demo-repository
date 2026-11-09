"""Streaming encoder-decoder LSTM predictor for a configured electricity station.

The trained model expects:
  * history_steps x 8 historical features (derived from the station interval)
  * horizon_steps x 9 future covariates (derived from the station interval)

The predictor loads its initial history window from a separate
`data/raw/{station}_history.csv` file. It writes one startup forecast from
that history, then updates the rolling window and forecasts after each
configured number of incoming Kafka observations.

Weather is deliberately kept simple for now: future temperature is held at the
latest observed temperature. This is a placeholder, not a weather forecast.
The model's feature dimensions and feature order remain compatible with the
provided encoder-decoder training script.

Run from the POC directory:
  python -m src.models.streaming_predictor --station amprion

For a fresh Kafka consumer group starting at the earliest available offset:
  python -m src.models.streaming_predictor --station amprion --replay

By default, the seed file is data/raw/amprion_history.csv when the station's
streaming input file is data/raw/amprion.csv. The first Kafka row must be the
next observation after the final seed row, using the station's configured interval.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import joblib
import pandas as pd
import torch
import yaml
from kafka import KafkaConsumer
from torch import nn

from src.ingestion.file_loader import load_station_csv
from src.storage.forecast_store import append_actuals, append_forecasts
from src.models.tabular_forecasting import LEGACY_FEATURE_SCHEMA, build_origin_features

ROOT = Path(__file__).resolve().parents[2]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("streaming-lstm-predictor")


class EncoderDecoderLSTM(nn.Module):
    """Must match the architecture used by the training script."""

    def __init__(
        self,
        n_history_features: int,
        n_future_features: int,
        hidden_size: int = 32,
    ) -> None:
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=n_history_features,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.decoder = nn.LSTM(
            input_size=n_future_features,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_size, 1)

    def forward(
        self,
        x_history: torch.Tensor,
        x_future: torch.Tensor,
    ) -> torch.Tensor:
        _, (hidden, cell) = self.encoder(x_history)
        decoded, _ = self.decoder(x_future, (hidden, cell))
        return self.output(decoded).squeeze(-1)


def load_yaml(relative_path: str) -> dict[str, Any]:
    path = ROOT / relative_path
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def calendar_feature_map(timestamps: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    """Create the exact cyclical calendar features used during training."""
    ts = pd.DatetimeIndex(timestamps)
    minute_of_day = ts.hour.to_numpy() * 60 + ts.minute.to_numpy()
    weekday = ts.dayofweek.to_numpy()
    day_of_year = ts.dayofyear.to_numpy() - 1

    hour_angle = 2.0 * np.pi * minute_of_day / 1440.0
    weekday_angle = 2.0 * np.pi * weekday / 7.0
    year_angle = 2.0 * np.pi * day_of_year / 365.2425

    return {
        "hour_sin": np.sin(hour_angle).astype(np.float32),
        "hour_cos": np.cos(hour_angle).astype(np.float32),
        "weekday_sin": np.sin(weekday_angle).astype(np.float32),
        "weekday_cos": np.cos(weekday_angle).astype(np.float32),
        "year_sin": np.sin(year_angle).astype(np.float32),
        "year_cos": np.cos(year_angle).astype(np.float32),
    }


def load_history_source(csv_path: Path, interval_minutes: int = 15) -> pd.DataFrame:
    """Load and clean the station's configured source CSV for historical seeding."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Historical seed CSV not found: {csv_path}")

    frame = load_station_csv(csv_path, interval_minutes=interval_minutes).copy()
    required = {"utc_timestamp", "demand", "temperature"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"Historical CSV {csv_path} is missing required columns: {sorted(missing)}"
        )

    frame["utc_timestamp"] = pd.to_datetime(
        frame["utc_timestamp"], utc=True, errors="coerce"
    )
    frame["demand"] = pd.to_numeric(frame["demand"], errors="coerce")
    frame["temperature"] = pd.to_numeric(frame["temperature"], errors="coerce")
    frame = frame.dropna(subset=["utc_timestamp", "demand", "temperature"])
    frame = frame[np.isfinite(frame["demand"]) & np.isfinite(frame["temperature"])]
    frame = frame[frame["demand"] >= 0]
    frame = (
        frame.sort_values("utc_timestamp")
        .drop_duplicates("utc_timestamp", keep="last")
        .reset_index(drop=True)
    )

    if frame.empty:
        raise ValueError(f"No valid historical rows found in {csv_path}")
    return frame[["utc_timestamp", "demand", "temperature"]]


def load_model(
    station_id: str,
    checkpoint_path: Path,
    model_cfg: dict[str, Any],
) -> tuple[Any, dict[str, Any], torch.device]:
    """Load the station's active registry model, falling back to its legacy LSTM."""
    station_doc = load_yaml("config/stations.yaml")
    station_entry = station_doc.get("stations", {}).get(station_id, {})
    active = station_entry.get("active_model") or {}
    if active.get("artifact_path"):
        checkpoint_path = Path(active["artifact_path"])
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_absolute():
        checkpoint_path = ROOT / checkpoint_path
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model artifact not found: {checkpoint_path}")

    algorithm = str(active.get("algorithm") or station_entry.get("model_algorithm") or "lstm").lower()
    device = torch.device(model_cfg.get("device", "cpu"))
    torch.set_num_threads(max(1, int(model_cfg.get("torch_num_threads", 2))))

    if algorithm in {"random_forest", "gbt", "xgboost"} or checkpoint_path.suffix.lower() in {".joblib", ".pkl"}:
        metadata_value = active.get("metadata_path")
        metadata_path = Path(metadata_value) if metadata_value else checkpoint_path.parent / "metadata.json"
        if not metadata_path.is_absolute():
            metadata_path = ROOT / metadata_path
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Model metadata not found: {metadata_path}")
        checkpoint = json.loads(metadata_path.read_text(encoding="utf-8"))
        if checkpoint.get("station") not in (None, station_id):
            raise ValueError(f"Model metadata station '{checkpoint.get('station')}' does not match '{station_id}'.")
        if checkpoint.get("algorithm") != algorithm:
            raise ValueError("Deployed algorithm and model metadata do not match.")
        model = joblib.load(checkpoint_path)
        checkpoint["region_name"] = station_id
        checkpoint["history"] = int(checkpoint.get("history_steps", 672))
        checkpoint["horizon"] = int(checkpoint.get("horizon_steps", 96))
        checkpoint["interval_minutes"] = int(checkpoint.get("interval_minutes", 15))
        checkpoint["algorithm"] = algorithm
        checkpoint["model_version"] = active.get("version") or checkpoint.get("version")
        log.info(
            "Loaded %s model %s for station=%s | history=%d | horizon=%d",
            algorithm, checkpoint_path, station_id, checkpoint["history"], checkpoint["horizon"],
        )
        return model, checkpoint, torch.device("cpu")

    # Legacy and newly trained LSTM checkpoints retain their original feature contract.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_station = checkpoint.get("region_name")
    if checkpoint_station != station_id:
        raise ValueError(
            f"Checkpoint region '{checkpoint_station}' does not match station '{station_id}'."
        )
    history_features = checkpoint["history_features"]
    future_features = checkpoint["future_features"]
    if len(history_features) != 8 or len(future_features) != 9:
        raise ValueError(
            "This predictor is written for the supplied LSTM schema: "
            f"8 history and 9 future features. Checkpoint has {len(history_features)} and {len(future_features)}."
        )
    history_steps = int(checkpoint["history"])
    horizon_steps = int(checkpoint["horizon"])
    interval_minutes = int(checkpoint["interval_minutes"])
    if history_steps <= 0 or horizon_steps <= 0 or interval_minutes <= 0:
        raise ValueError("Checkpoint history, horizon and interval must be positive.")
    model = EncoderDecoderLSTM(
        n_history_features=len(history_features),
        n_future_features=len(future_features),
        hidden_size=int(checkpoint["hidden_size"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    checkpoint["algorithm"] = "lstm"
    checkpoint.setdefault("model_version", active.get("version") or station_entry.get("model_version", "legacy"))
    log.info("Loaded LSTM model: %s", checkpoint_path)
    log.info(
        "Station=%s | history=%d | horizon=%d | interval=%d minutes | device=%s",
        station_id, history_steps, horizon_steps, interval_minutes, device,
    )
    log.info("Training weather mode in checkpoint: %s", checkpoint.get("weather_mode", "unknown"))
    log.warning(
        "Weather shortcut enabled: future temperature is held at the latest observed value. "
        "This is a temporary persistence assumption, not a weather forecast."
    )
    checkpoint["history_features"] = history_features
    checkpoint["future_features"] = future_features
    return model, checkpoint, device


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    return (
        pd.Series(values, dtype="float64")
        .rolling(window=max(1, window), min_periods=1)
        .mean()
        .to_numpy(dtype=np.float32)
    )


def build_model_inputs(
    rows: list[dict[str, Any]],
    checkpoint: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Build station-configured history and future inputs matching training features."""
    history_steps = int(checkpoint["history"])
    horizon_steps = int(checkpoint["horizon"])
    interval_minutes = int(checkpoint["interval_minutes"])
    interval = pd.Timedelta(minutes=interval_minutes)

    if len(rows) < history_steps:
        raise ValueError(f"Need {history_steps} history rows; received {len(rows)}.")

    rows = rows[-history_steps:]
    history_ts = pd.DatetimeIndex(
        pd.to_datetime([row["utc_timestamp"] for row in rows], utc=True)
    )
    if len(history_ts) > 1 and not np.all(np.diff(history_ts.asi8) == interval.value):
        raise ValueError(f"Historical input window is not continuous at the configured {interval_minutes}-minute interval.")

    demand = np.asarray([row["demand"] for row in rows], dtype=np.float32)
    historical_temp = np.asarray([row["temperature"] for row in rows], dtype=np.float32)
    if not np.isfinite(demand).all() or not np.isfinite(historical_temp).all():
        raise ValueError("History contains missing or non-finite demand/temperature values.")

    region = checkpoint["region_name"]
    scales = checkpoint["scales"][region]
    demand_mean = float(scales["demand_mean"])
    demand_std = float(scales["demand_std"])
    temp_mean = float(scales["temperature_mean"])
    temp_std = float(scales["temperature_std"])
    if demand_std == 0 or temp_std == 0:
        raise ValueError("Checkpoint contains a zero standard deviation in its scales.")

    history_calendar = calendar_feature_map(history_ts)
    history_map: dict[str, np.ndarray] = {
        "demand_scaled": ((demand - demand_mean) / demand_std).astype(np.float32),
        "temperature_scaled": ((historical_temp - temp_mean) / temp_std).astype(np.float32),
        **history_calendar,
    }
    history_feature_names = checkpoint["history_features"]
    missing_history = [name for name in history_feature_names if name not in history_map]
    if missing_history:
        raise ValueError(f"Cannot construct checkpoint history features: {missing_history}")
    x_history = np.column_stack(
        [history_map[name] for name in history_feature_names]
    ).astype(np.float32)

    origin = history_ts[-1]
    future_ts = pd.date_range(
        start=origin + interval,
        periods=horizon_steps,
        freq=interval,
        tz="UTC",
    )

    # Temporary no-weather-feed behaviour: persistence of latest observed temperature.
    future_temp = np.full(horizon_steps, historical_temp[-1], dtype=np.float32)

    # The training rolling means are trailing (current + prior timestamps).
    # Include the preceding 95 observed temperatures so the first future 24h
    # rolling mean has the same temporal context as the training calculation.
    rolling_window_1h = max(1, int(round(60 / interval_minutes)))
    rolling_window_24h = max(1, int(round(1440 / interval_minutes)))
    context_needed = max(rolling_window_1h, rolling_window_24h) - 1
    temperature_context = np.concatenate(
        [historical_temp[-context_needed:] if context_needed else historical_temp[:0], future_temp]
    )
    rolling_1h_all = rolling_mean(temperature_context, rolling_window_1h)
    rolling_24h_all = rolling_mean(temperature_context, rolling_window_24h)
    rolling_1h = rolling_1h_all[-horizon_steps:]
    rolling_24h = rolling_24h_all[-horizon_steps:]

    future_calendar = calendar_feature_map(future_ts)
    future_map: dict[str, np.ndarray] = {
        "future_temperature_scaled": ((future_temp - temp_mean) / temp_std).astype(np.float32),
        "temperature_rolling_mean_1h_scaled": ((rolling_1h - temp_mean) / temp_std).astype(np.float32),
        "temperature_rolling_mean_24h_scaled": ((rolling_24h - temp_mean) / temp_std).astype(np.float32),
        **future_calendar,
    }
    future_feature_names = checkpoint["future_features"]
    missing_future = [name for name in future_feature_names if name not in future_map]
    if missing_future:
        raise ValueError(f"Cannot construct checkpoint future features: {missing_future}")
    x_future = np.column_stack(
        [future_map[name] for name in future_feature_names]
    ).astype(np.float32)

    return x_history, x_future, future_ts


def generate_forecast(
    rows: list[dict[str, Any]],
    model: Any,
    checkpoint: dict[str, Any],
    device: torch.device,
    checkpoint_path: Path,
) -> list[dict[str, Any]]:
    algorithm = str(checkpoint.get("algorithm", "lstm")).lower()
    region = str(checkpoint.get("region_name", checkpoint.get("station", "unknown")))
    history_steps = int(checkpoint["history"])
    horizon_steps = int(checkpoint["horizon"])
    interval_minutes = int(checkpoint["interval_minutes"])
    history_timestamps = pd.DatetimeIndex(pd.to_datetime([row["utc_timestamp"] for row in rows], utc=True))
    if len(rows) < history_steps:
        raise ValueError(f"Need {history_steps} history rows; received {len(rows)}.")
    future_ts = pd.date_range(
        start=history_timestamps[-1] + pd.Timedelta(minutes=interval_minutes),
        periods=horizon_steps,
        freq=pd.Timedelta(minutes=interval_minutes),
        tz="UTC",
    )

    if algorithm in {"random_forest", "gbt", "xgboost"}:
        history_frame = pd.DataFrame(rows[-history_steps:])
        features = build_origin_features(
            history_frame, future_ts, interval_minutes,
            feature_schema=checkpoint.get("feature_schema", LEGACY_FEATURE_SCHEMA),
        )
        feature_names = checkpoint.get("feature_names", list(features.columns))
        missing = set(feature_names) - set(features.columns)
        if missing:
            raise ValueError(f"Serving feature builder is missing model features: {sorted(missing)}")
        predictions = np.asarray(model.predict(features.reindex(columns=feature_names)), dtype=float).reshape(-1)
    else:
        x_history, x_future, future_ts = build_model_inputs(rows, checkpoint)
        region = checkpoint["region_name"]
        scales = checkpoint["scales"][region]
        with torch.inference_mode():
            prediction_scaled = model(
                torch.from_numpy(x_history).unsqueeze(0).to(device),
                torch.from_numpy(x_future).unsqueeze(0).to(device),
            ).cpu().numpy().reshape(-1)
        predictions = prediction_scaled * float(scales["demand_std"]) + float(scales["demand_mean"])

    if len(predictions) != len(future_ts):
        raise RuntimeError(
            f"Model returned {len(predictions)} predictions for {len(future_ts)} timestamps."
        )
    issued_at = datetime.now(timezone.utc).isoformat()
    origin_at = pd.Timestamp(rows[-1]["utc_timestamp"]).isoformat()
    return [
        {
            "station_id": region,
            "forecast_issued_at_utc": issued_at,
            "forecast_origin_utc": origin_at,
            "forecast_timestamp_utc": timestamp.isoformat(),
            "horizon_step": step,
            "predicted_demand": float(value),
            "model_file": checkpoint_path.name,
            "model_version": checkpoint.get("model_version", "legacy"),
            "algorithm": algorithm,
            "experiment_id": checkpoint.get("experiment_id", ""),
            "weather_mode": checkpoint.get("weather_mode", "last_observed_temperature_persistence"),
        }
        for step, (timestamp, value) in enumerate(zip(future_ts, predictions), start=1)
    ]


def parse_event(payload: dict[str, Any]) -> dict[str, Any]:
    timestamp_value = payload.get("utc_timestamp") or payload.get("timestamp")
    if timestamp_value is None:
        raise ValueError("Kafka event is missing 'utc_timestamp'.")
    if "demand" not in payload:
        raise ValueError("Kafka event is missing 'demand'.")

    timestamp = pd.to_datetime(timestamp_value, utc=True, errors="coerce")
    demand = pd.to_numeric(payload.get("demand"), errors="coerce")
    temperature = pd.to_numeric(payload.get("temperature"), errors="coerce")
    if pd.isna(timestamp) or pd.isna(demand) or pd.isna(temperature):
        raise ValueError("Kafka event has an invalid timestamp, demand or temperature.")
    demand_value = float(demand)
    temperature_value = float(temperature)
    if demand_value < 0 or not np.isfinite(demand_value) or not np.isfinite(temperature_value):
        raise ValueError("Kafka event has negative demand or non-finite numeric values.")

    return {
        "utc_timestamp": pd.Timestamp(timestamp),
        "demand": demand_value,
        "temperature": temperature_value,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Streaming encoder-decoder LSTM predictor for a configured station."
    )
    parser.add_argument("--station", required=True, help="Station key in config/stations.yaml")
    parser.add_argument("--model", default=None, help="Optional checkpoint path override")
    parser.add_argument(
        "--history-file",
        default=None,
        help="Initial history CSV; defaults to <station>_history.csv beside input_file.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Forecast CSV output path; defaults to the station's configured forecasts_file.",
    )
    parser.add_argument(
        "--actual-output",
        default=None,
        help="Actual-observation CSV output path; defaults to the station's configured actuals_file.",
    )
    parser.add_argument(
        "--replay",
        action="store_true",
        help="Use a fresh consumer group and read the featured topic from its earliest offset.",
    )
    args = parser.parse_args()

    station_config = load_yaml("config/stations.yaml")
    kafka_config = load_yaml("config/kafka.yaml")
    model_config = load_yaml("config/model.yaml")
    stations = station_config.get("stations", {})
    if args.station not in stations:
        raise ValueError(
            f"Unknown station '{args.station}'. Configured stations: {list(stations)}"
        )
    station = stations[args.station]

    # ------------------------------------------------------------------
    # 1. LOAD MODEL AND INITIAL HISTORY
    # ------------------------------------------------------------------
    active_model = station.get("active_model") or {}
    checkpoint_value = args.model or active_model.get("artifact_path") or station["model_checkpoint"]
    checkpoint_path = Path(checkpoint_value)
    if not checkpoint_path.is_absolute():
        checkpoint_path = ROOT / checkpoint_path
    checkpoint_path = checkpoint_path.resolve()
    model, checkpoint, device = load_model(args.station, checkpoint_path, model_config)

    stream_csv_path = Path(station["input_file"])
    if not stream_csv_path.is_absolute():
        stream_csv_path = ROOT / stream_csv_path
    history_path = Path(args.history_file) if args.history_file else stream_csv_path.with_name(
        f"{args.station}_history.csv"
    )
    if not history_path.is_absolute():
        history_path = ROOT / history_path
    history_path = history_path.resolve()

    history_steps = int(checkpoint["history"])
    horizon_steps = int(checkpoint["horizon"])
    interval_minutes = int(checkpoint["interval_minutes"])
    history_source = load_history_source(history_path, interval_minutes=interval_minutes)
    interval = pd.Timedelta(minutes=interval_minutes)

    if len(history_source) < history_steps:
        raise ValueError(
            f"History file {history_path} contains {len(history_source)} valid rows; "
            f"the model requires {history_steps}."
        )

    seed_frame = history_source.tail(history_steps).copy()
    seed_timestamps = pd.DatetimeIndex(seed_frame["utc_timestamp"])
    if len(seed_timestamps) != history_steps or (
        len(seed_timestamps) > 1
        and not np.all(np.diff(seed_timestamps.asi8) == interval.value)
    ):
        raise ValueError(
            f"The last {history_steps} rows in {history_path} are not a continuous "
            f"{interval_minutes}-minute sequence. Check the history CSV before starting."
        )

    buffer: deque[dict[str, Any]] = deque(maxlen=history_steps)
    buffer.extend(
        {
            "utc_timestamp": pd.Timestamp(row.utc_timestamp),
            "demand": float(row.demand),
            "temperature": float(row.temperature),
        }
        for row in seed_frame.itertuples(index=False)
    )
    last_history_timestamp = buffer[-1]["utc_timestamp"]

    log.info(
        "Initial history loaded: %s | rows=%d | range=%s to %s",
        history_path,
        len(buffer),
        buffer[0]["utc_timestamp"],
        last_history_timestamp,
    )
    log.info(
        "Expected first streaming timestamp: %s",
        (last_history_timestamp + interval).isoformat(),
    )

    # ------------------------------------------------------------------
    # 2. OUTPUT PATHS AND FORECAST CADENCE
    # ------------------------------------------------------------------
    forecast_value = args.output or station.get("forecasts_file") or "data/processed/forecasts.csv"
    actual_value = args.actual_output or station.get("actuals_file") or "data/processed/actuals.csv"
    actual_output_path = Path(actual_value)
    output_path = Path(forecast_value)
    if not actual_output_path.is_absolute():
        actual_output_path = ROOT / actual_output_path
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    actual_output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Forecast cadence is counted in station observations, not wall-clock minutes.
    forecast_every = int(model_config.get("forecast_every_steps", 1))
    if forecast_every <= 0:
        raise ValueError("forecast_every_steps in config/model.yaml must be positive.")

    # ------------------------------------------------------------------
    # 3. INITIAL FORECAST FROM THE CONFIGURED HISTORY WINDOW
    # ------------------------------------------------------------------
    initial_records = generate_forecast(
        list(buffer), model, checkpoint, device, checkpoint_path
    )
    initial_saved = append_forecasts(output_path, initial_records)
    log.info(
        "Initial forecast saved: rows=%d (storage reported %d) | origin=%s | "
        "first target=%s | final target=%s",
        len(initial_records),
        initial_saved,
        initial_records[0]["forecast_origin_utc"],
        initial_records[0]["forecast_timestamp_utc"],
        initial_records[-1]["forecast_timestamp_utc"],
    )

    # valid_count tracks accepted Kafka observations only. The startup forecast
    # is the baseline, so the first subsequent Kafka event can trigger a new run.
    received_count = 0
    valid_count = 0
    last_forecast_valid_count: int | None = 0
    actual_batch: list[dict[str, Any]] = []
    last_actual_flush = time.monotonic()
    last_progress_log = 0

    def flush_actuals() -> None:
        nonlocal last_actual_flush
        if actual_batch:
            saved = append_actuals(actual_output_path, actual_batch)
            log.info("Saved %d actual observations to %s", saved, actual_output_path)
            actual_batch.clear()
        last_actual_flush = time.monotonic()

    # ------------------------------------------------------------------
    # 4. KAFKA CONSUMER
    # ------------------------------------------------------------------
    configured_group = f"{kafka_config['consumer_group']}-{args.station}"
    if args.replay:
        consumer_group = f"{configured_group}-replay-{uuid.uuid4().hex[:10]}"
        offset_policy = "earliest"
        log.info(
            "Replay mode: fresh group '%s' starts at the earliest available featured offset.",
            consumer_group,
        )
    else:
        consumer_group = configured_group
        offset_policy = "earliest"
        log.info(
            "Normal mode: using group '%s' (resumes its committed offset when present).",
            consumer_group,
        )

    consumer = KafkaConsumer(
        station["featured_topic"],
        bootstrap_servers=kafka_config["bootstrap_servers"],
        group_id=consumer_group,
        auto_offset_reset=offset_policy,
        enable_auto_commit=True,
        value_deserializer=lambda value: json.loads(value.decode("utf-8")),
    )

    log.info(
        "Listening to featured topic '%s' for station '%s'.",
        station["featured_topic"],
        args.station,
    )
    log.info(
        "History=%d | horizon=%d | interval=%d minutes | forecast cadence=%d observation(s)",
        history_steps,
        horizon_steps,
        interval_minutes,
        forecast_every,
    )
    log.info("Actual output: %s", actual_output_path)
    log.info("Forecast output: %s", output_path)

    try:
        for message in consumer:
            received_count += 1
            try:
                payload = message.value
                if payload.get("station_id") not in (None, args.station):
                    continue

                event = parse_event(payload)
                event_ts = event["utc_timestamp"]
                last_buffer_ts = buffer[-1]["utc_timestamp"] if buffer else None

                # Ignore old/duplicate messages that may already exist in Kafka
                # from an earlier run. The stream must begin after the seed file.
                if last_buffer_ts is not None and event_ts <= last_buffer_ts:
                    log.debug(
                        "Skipping duplicate/old Kafka timestamp %s (history ends at %s).",
                        event_ts,
                        last_buffer_ts,
                    )
                    continue

                # The first incoming message should be exactly one interval after
                # the seed. Any later gap resets the buffer to avoid forecasting
                # from a non-contiguous sequence.
                if last_buffer_ts is not None and event_ts - last_buffer_ts != interval:
                    log.error(
                        "Timestamp gap detected: %s -> %s. Resetting history buffer; "
                        "forecasting resumes after %d contiguous observations.",
                        last_buffer_ts,
                        event_ts,
                        history_steps,
                    )
                    buffer.clear()
                    last_forecast_valid_count = None

                buffer.append(event)
                valid_count += 1

                actual_batch.append(
                    {
                        "station_id": args.station,
                        "utc_timestamp": event_ts.isoformat(),
                        "actual_demand": event["demand"],
                        "temperature": event["temperature"],
                    }
                )
                if len(actual_batch) >= 100 or time.monotonic() - last_actual_flush >= 5:
                    flush_actuals()

                should_forecast = (
                    len(buffer) >= history_steps
                    and (
                        last_forecast_valid_count is None
                        or valid_count - last_forecast_valid_count >= forecast_every
                    )
                )
                if should_forecast:
                    records = generate_forecast(
                        list(buffer), model, checkpoint, device, checkpoint_path
                    )
                    saved = append_forecasts(output_path, records)
                    log.info(
                        "Saved %d forecast rows (storage reported %d); origin=%s; "
                        "first target=%s; final target=%s",
                        len(records),
                        saved,
                        records[0]["forecast_origin_utc"],
                        records[0]["forecast_timestamp_utc"],
                        records[-1]["forecast_timestamp_utc"],
                    )
                    last_forecast_valid_count = valid_count

                if valid_count - last_progress_log >= 100:
                    log.info(
                        "Kafka received=%s | valid=%s | history buffer=%s/%s",
                        f"{received_count:,}",
                        f"{valid_count:,}",
                        len(buffer),
                        history_steps,
                    )
                    last_progress_log = valid_count

            except Exception:
                log.exception("Failed to process Kafka offset %s", message.offset)

    except KeyboardInterrupt:
        log.info("Stopping predictor.")
    finally:
        flush_actuals()
        consumer.close()
        log.info("Kafka consumer closed.")


if __name__ == "__main__":
    main()
