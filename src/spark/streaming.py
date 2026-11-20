
"""Generic Spark Structured Streaming feature extraction for a configured station."""

import argparse
import json
import logging
import os
import signal
import time
from pathlib import Path
from threading import Event

import yaml

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    from_json,
    try_to_timestamp,
    hour,
    minute,
    dayofweek,
    dayofyear,
    sin,
    cos,
    lit,
    struct,
    to_json,
    when,
    coalesce,
    isnan,
    current_timestamp,
    expr,
    pmod,
    unix_micros,
)
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    DoubleType,
    BooleanType,
    IntegerType,
)


ROOT = Path(__file__).resolve().parents[2]


def load_yaml(relative_path):
    """Load YAML configuration."""

    with (ROOT / relative_path).open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_spark(spark_cfg, name):
    master_url = os.getenv("SPARK_MASTER_URL", spark_cfg.get("master", "local[2]"))
    shuffle_partitions = int(os.getenv("SPARK_SHUFFLE_PARTITIONS", spark_cfg.get("shuffle_partitions", 2)))
    spark_major = int(str(pyspark.__version__).split(".", 1)[0])
    scala_binary = str(spark_cfg.get("scala_binary") or ("2.13" if spark_major >= 4 else "2.12"))
    kafka_connector = spark_cfg.get("kafka_connector") or (
        f"org.apache.spark:spark-sql-kafka-0-10_{scala_binary}:{pyspark.__version__}"
    )
    builder = (SparkSession.builder.appName(name).master(master_url)
               .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
               .config("spark.sql.session.timeZone", "UTC")
               .config("spark.sql.adaptive.enabled", "false")
               .config("spark.ui.enabled", str(spark_cfg.get("ui_enabled", False)).lower())
               .config("spark.scheduler.mode", "FAIR"))
    if master_url.startswith("local"):
        builder = builder.config("spark.driver.memory", str(spark_cfg.get("driver_memory", "640m")))
    connector_jars = os.getenv("ELECTRICITY_SPARK_KAFKA_JARS")
    if connector_jars:
        builder = builder.config("spark.jars", connector_jars)
    else:
        builder = builder.config("spark.jars.packages", kafka_connector)
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


def start_station_query(spark, station_id, station, kafka_cfg, spark_cfg):
    managed_weather = bool(station.get("weather"))
    kafka_bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS", kafka_cfg.get("bootstrap_servers"))
    max_offsets = int(os.getenv("SPARK_MAX_OFFSETS_PER_TRIGGER", spark_cfg.get("max_offsets_per_trigger", 500)))
    # ---------------------------------------------------------
    # 2. Define incoming Kafka message schema
    # ---------------------------------------------------------

    schema = StructType([
        StructField("station_id", StringType(), True),
        StructField("utc_timestamp", StringType(), True),
        StructField("demand", DoubleType(), True),
        StructField("temperature", DoubleType(), True),
        StructField("forecast_temperature", DoubleType(), True),
        StructField("solar_mwh", DoubleType(), True),
        StructField("solar_capacity_mw", DoubleType(), True),
        StructField("wind_mwh", DoubleType(), True),
        StructField("wind_capacity_mw", DoubleType(), True),

        # Replay metadata
        StructField("is_simulated", BooleanType(), True),
        StructField("data_mode", StringType(), True),
        StructField("replay_cycle", IntegerType(), True),
    ])

    # ---------------------------------------------------------
    # 4. Read raw Kafka topic
    # ---------------------------------------------------------

    raw_reader = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", kafka_bootstrap)
        .option("subscribe", station["raw_topic"])
        .option("startingOffsets", spark_cfg.get("starting_offsets", "earliest"))
    )
    if max_offsets > 0:
        raw_reader = raw_reader.option("maxOffsetsPerTrigger", max_offsets)
    raw = raw_reader.load()

    # ---------------------------------------------------------
    # 5. Parse JSON messages
    # ---------------------------------------------------------

    parsed = (
        raw.select(
            col("value").cast("string").alias("raw_value"),
            col("topic").alias("source_topic"),
            col("partition").alias("source_partition"),
            col("offset").alias("source_offset"),
            from_json(col("value").cast("string"), schema).alias("record"),
        )
        .select("record.*", "raw_value", "source_topic", "source_partition", "source_offset")
    )

    # ---------------------------------------------------------
    # 6. Build both outputs from one parsed micro-batch
    # ---------------------------------------------------------

    future_skew_minutes = int(
        (load_yaml("config/storage.yaml").get("quality") or {}).get(
            "max_future_skew_minutes", 5
        )
    )
    dlq_topic = str(kafka_cfg.get("dead_letter_topic", "electricity-dlq"))

    def build_batch_outputs(batch_df):
        """Apply validation/features once to the supplied micro-batch."""
        ts = try_to_timestamp(col("utc_timestamp"))
        temperature_is_valid = (
            col("temperature").isNotNull()
            & (~isnan(col("temperature")))
            & (col("temperature") != lit(float("inf")))
            & (col("temperature") != lit(float("-inf")))
        )
        if managed_weather:
            temperature_is_valid = col("temperature").isNull() | temperature_is_valid
        offset = station.get("hierarchy_grid_offset_seconds")
        on_station_grid = lit(True)
        if offset is not None:
            interval_us = int(station["interval_minutes"]) * 60 * 10**6
            on_station_grid = pmod(unix_micros(ts), lit(interval_us)) == lit(int(offset) * 10**6)
        valid_condition = coalesce(
            (col("station_id") == lit(station_id))
            & col("utc_timestamp").isNotNull()
            & col("demand").isNotNull()
            & temperature_is_valid
            & (col("demand") >= 0)
            & (~isnan(col("demand")))
            & (col("demand") != lit(float("inf")))
            & (col("demand") != lit(float("-inf")))
            & ts.isNotNull()
            & on_station_grid
            & (
                ts
                <= current_timestamp()
                + expr(f"INTERVAL {max(0, future_skew_minutes)} MINUTES")
            ),
            lit(False),
        )

        quality_reason = (
            when(col("station_id").isNull(), lit("invalid_json_or_missing_station"))
            .when(col("station_id") != lit(station_id), lit("station_mismatch"))
            .when(col("utc_timestamp").isNull() | ts.isNull(), lit("invalid_timestamp"))
            .when(
                ts > current_timestamp() + expr(f"INTERVAL {max(0, future_skew_minutes)} MINUTES"),
                lit("future_timestamp"),
            )
            .when(~on_station_grid, lit("off_station_grid"))
            .when(col("demand").isNull(), lit("missing_demand"))
            .when(lit(not managed_weather) & col("temperature").isNull(), lit("missing_temperature"))
            .when(col("demand") < 0, lit("negative_demand"))
            .when(
                isnan(col("demand"))
                | (col("demand") == lit(float("inf")))
                | (col("demand") == lit(float("-inf"))),
                lit("non_finite_demand"),
            )
            .when(
                isnan(col("temperature"))
                | (col("temperature") == lit(float("inf")))
                | (col("temperature") == lit(float("-inf"))),
                lit("non_finite_temperature"),
            )
            .otherwise(lit("invalid_record"))
        )

        minute_of_day = hour(ts) * lit(60) + minute(ts)
        # Spark dayofweek: Sunday=1. Convert to Monday=0 ... Sunday=6.
        weekday_zero = (dayofweek(ts) + lit(5)) % lit(7)
        year_day_zero = dayofyear(ts) - lit(1)
        two_pi = lit(2.0 * 3.141592653589793)

        features = struct(
            "station_id", "utc_timestamp", ts.alias("timestamp_utc"),
            "demand", "temperature", "forecast_temperature",
            "solar_mwh", "solar_capacity_mw", "wind_mwh", "wind_capacity_mw",
            sin(two_pi * minute_of_day / lit(1440.0)).alias("hour_sin"),
            cos(two_pi * minute_of_day / lit(1440.0)).alias("hour_cos"),
            sin(two_pi * weekday_zero / lit(7.0)).alias("weekday_sin"),
            cos(two_pi * weekday_zero / lit(7.0)).alias("weekday_cos"),
            sin(two_pi * year_day_zero / lit(365.2425)).alias("year_sin"),
            cos(two_pi * year_day_zero / lit(365.2425)).alias("year_cos"),
            "is_simulated", "data_mode", "replay_cycle",
        )
        dead_letter = struct(
            lit(station_id).alias("station_id"), col("station_id").alias("source_station_id"),
            quality_reason.alias("quality_reason"), "source_topic",
            "source_partition", "source_offset", "raw_value",
        )
        return batch_df.select(
            lit(station_id).alias("key"),
            when(valid_condition, lit(station["featured_topic"])).otherwise(lit(dlq_topic)).alias("topic"),
            when(valid_condition, to_json(features)).otherwise(to_json(dead_letter)).alias("value"),
        )

    # ---------------------------------------------------------
    # 7. Single Structured Streaming source query, two Kafka sinks
    # ---------------------------------------------------------

    checkpoint_dir = spark_cfg.get(
        "checkpoint_location", "checkpoints/spark_features"
    )
    checkpoint_path = ROOT / f"{checkpoint_dir}_{station_id}"
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    def publish_microbatch(batch_df, batch_id: int) -> None:
        # The Kafka sink routes each row via its topic column. A single write
        # completes before Spark commits the source checkpoint for this batch.
        (build_batch_outputs(batch_df).write.format("kafka")
         .option("kafka.bootstrap.servers", kafka_bootstrap).save())
        print(f"{station_id}: Spark batch {batch_id} acknowledged (featured/DLQ)", flush=True)

    query = (
        parsed.writeStream
        .foreachBatch(publish_microbatch)
        .option("checkpointLocation", str(checkpoint_path))
        .outputMode("append")
        .trigger(processingTime=str(spark_cfg.get("trigger_interval", "1 second")))
        .start()
    )

    print(
        f"Spark streaming started for '{station_id}': "
        f"{station['raw_topic']} -> {station['featured_topic']} | DLQ={dlq_topic} "
        "(one query, one Kafka write per batch)",
        flush=True,
    )
    return query


def run_shared(spark, control_dir: Path, kafka_cfg, spark_cfg, stop_requested: Event) -> None:
    from src.spark.control import read_control, write_control

    logger = logging.getLogger(__name__)
    queries = {}
    status = {}
    retries = {}
    report_path = control_dir / "status.json"
    desired_path = control_dir / "desired.json"

    def publish():
        write_control(report_path, {"pid": os.getpid(), "stations": status})

    def stop_query(station):
        query = queries.pop(station, None)
        if query is not None and query.isActive:
            query.stop()

    publish()
    try:
        while control_dir.is_dir() and not stop_requested.is_set():
            desired = read_control(desired_path)
            changed = False
            for station, item in list(status.items()):
                if desired.get(station) != item["token"]:
                    stop_query(station)
                    retries.pop(station, None)
                    status.pop(station)
                    changed = True
            for station, token in desired.items():
                query = queries.get(station)
                if query is not None and query.isActive:
                    continue
                if time.monotonic() < retries.get(station, 0):
                    continue
                try:
                    if query is not None:
                        failure = query.exception()
                        logger.warning("%s: Spark query terminated: %s", station, failure)
                        queries.pop(station)
                    entry = load_yaml("config/stations.yaml")["stations"][station]
                    queries[station] = start_station_query(spark, station, entry, kafka_cfg, spark_cfg)
                    status[station] = {"token": token, "state": "running"}
                    retries.pop(station, None)
                except Exception as exc:
                    status[station] = {"token": token, "state": "restarting", "error": str(exc)[:500]}
                    retries[station] = time.monotonic() + 5
                    logger.exception("%s: Could not start Spark query; retrying in 5s.", station)
                changed = True
            if changed:
                publish()
            stop_requested.wait(0.35)
    finally:
        for station in list(queries):
            stop_query(station)


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--station", help="Station key in config/stations.yaml")
    mode.add_argument("--control-dir", type=Path, help="FastAPI shared Spark control directory")
    args = parser.parse_args()
    kafka_cfg = load_yaml("config/kafka.yaml")
    spark_cfg = load_yaml("config/spark.yaml")
    name = spark_cfg.get("app_name", "electricity-feature-extraction")
    stop_requested = Event()
    signal.signal(signal.SIGTERM, lambda *_: stop_requested.set())
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, lambda *_: stop_requested.set())
    spark = create_spark(spark_cfg, f"{name}-{args.station or 'shared'}")
    try:
        if args.control_dir is not None:
            run_shared(spark, args.control_dir, kafka_cfg, spark_cfg, stop_requested)
        else:
            station = load_yaml("config/stations.yaml")["stations"][args.station]
            query = start_station_query(spark, args.station, station, kafka_cfg, spark_cfg)
            try:
                while not stop_requested.is_set():
                    if query.awaitTermination(0.5):
                        raise RuntimeError("Spark source/output query stopped unexpectedly.")
            finally:
                if query.isActive:
                    query.stop()
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
