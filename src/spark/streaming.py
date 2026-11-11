
"""Generic Spark Structured Streaming feature extraction for a configured station."""

import argparse
import json
import time
from pathlib import Path

import yaml

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    from_json,
    to_timestamp,
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


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--station",
        required=True,
        help="Station key in config/stations.yaml",
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # 1. Load configuration
    # ---------------------------------------------------------

    kafka_cfg = load_yaml("config/kafka.yaml")
    station_cfg = load_yaml("config/stations.yaml")
    spark_cfg = load_yaml("config/spark.yaml")

    stations = station_cfg.get("stations", {})

    if args.station not in stations:
        raise ValueError(
            f"Unknown station '{args.station}'. "
            f"Configured stations: {list(stations)}"
        )

    station = stations[args.station]

    # ---------------------------------------------------------
    # 2. Define incoming Kafka message schema
    # ---------------------------------------------------------

    schema = StructType([
        StructField("station_id", StringType(), True),
        StructField("utc_timestamp", StringType(), True),
        StructField("demand", DoubleType(), True),
        StructField("temperature", DoubleType(), True),
        StructField("forecast_temperature", DoubleType(), True),

        # Replay metadata
        StructField("is_simulated", BooleanType(), True),
        StructField("data_mode", StringType(), True),
        StructField("replay_cycle", IntegerType(), True),
    ])

    # ---------------------------------------------------------
    # 3. Create Spark session
    # ---------------------------------------------------------

    import os
    master_url = os.getenv("SPARK_MASTER_URL", spark_cfg.get("master", "local[2]"))
    kafka_bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS", kafka_cfg.get("bootstrap_servers"))
    shuffle_partitions = int(os.getenv("SPARK_SHUFFLE_PARTITIONS", spark_cfg.get("shuffle_partitions", 2)))
    max_offsets = int(os.getenv("SPARK_MAX_OFFSETS_PER_TRIGGER", spark_cfg.get("max_offsets_per_trigger", 500)))
    spark_major = int(str(pyspark.__version__).split(".", 1)[0])
    scala_binary = str(spark_cfg.get("scala_binary") or ("2.13" if spark_major >= 4 else "2.12"))
    kafka_connector = spark_cfg.get("kafka_connector") or (
        f"org.apache.spark:spark-sql-kafka-0-10_{scala_binary}:{pyspark.__version__}"
    )

    spark = (
        SparkSession.builder
        .appName(
            f"{spark_cfg.get('app_name', 'electricity-feature-extraction')}"
            f"-{args.station}"
        )
        .master(master_url)
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.sql.session.timeZone", "UTC")
        .config(
            "spark.jars.packages",
            kafka_connector,
        )
        .getOrCreate()
    )

    spark.conf.set("spark.sql.session.timeZone", "UTC")
    spark.sparkContext.setLogLevel("WARN")

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
        ts = to_timestamp(col("utc_timestamp"))
        valid_condition = coalesce(
            (col("station_id") == lit(args.station))
            & col("utc_timestamp").isNotNull()
            & col("demand").isNotNull()
            & col("temperature").isNotNull()
            & (col("demand") >= 0)
            & (~isnan(col("demand")))
            & (~isnan(col("temperature")))
            & (col("demand") != lit(float("inf")))
            & (col("demand") != lit(float("-inf")))
            & (col("temperature") != lit(float("inf")))
            & (col("temperature") != lit(float("-inf")))
            & ts.isNotNull()
            & (
                ts
                <= current_timestamp()
                + expr(f"INTERVAL {max(0, future_skew_minutes)} MINUTES")
            ),
            lit(False),
        )

        valid = batch_df.filter(valid_condition)
        invalid = batch_df.filter(~valid_condition).withColumn(
            "quality_reason",
            when(col("station_id").isNull(), lit("invalid_json_or_missing_station"))
            .when(col("station_id") != lit(args.station), lit("station_mismatch"))
            .when(col("utc_timestamp").isNull() | ts.isNull(), lit("invalid_timestamp"))
            .when(
                ts > current_timestamp() + expr(f"INTERVAL {max(0, future_skew_minutes)} MINUTES"),
                lit("future_timestamp"),
            )
            .when(col("demand").isNull(), lit("missing_demand"))
            .when(col("temperature").isNull(), lit("missing_temperature"))
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
            .otherwise(lit("invalid_record")),
        )

        minute_of_day = hour(ts) * lit(60) + minute(ts)
        # Spark dayofweek: Sunday=1. Convert to Monday=0 ... Sunday=6.
        weekday_zero = (dayofweek(ts) + lit(5)) % lit(7)
        year_day_zero = dayofyear(ts) - lit(1)
        two_pi = lit(2.0 * 3.141592653589793)

        featured = (
            valid
            .withColumn("timestamp_utc", ts)
            .withColumn("hour_sin", sin(two_pi * minute_of_day / lit(1440.0)))
            .withColumn("hour_cos", cos(two_pi * minute_of_day / lit(1440.0)))
            .withColumn("weekday_sin", sin(two_pi * weekday_zero / lit(7.0)))
            .withColumn("weekday_cos", cos(two_pi * weekday_zero / lit(7.0)))
            .withColumn("year_sin", sin(two_pi * year_day_zero / lit(365.2425)))
            .withColumn("year_cos", cos(two_pi * year_day_zero / lit(365.2425)))
            .select(
                "station_id",
                "utc_timestamp",
                "timestamp_utc",
                "demand",
                "temperature",
                "forecast_temperature",
                "hour_sin",
                "hour_cos",
                "weekday_sin",
                "weekday_cos",
                "year_sin",
                "year_cos",
                "is_simulated",
                "data_mode",
                "replay_cycle",
            )
        )
        featured_output = featured.selectExpr(
            "CAST(station_id AS STRING) AS key",
            "to_json(struct(*)) AS value",
        )
        dlq_output = invalid.selectExpr(
            f"CAST('{args.station}' AS STRING) AS key",
            "to_json(named_struct("
            f"'station_id', '{args.station}', 'source_station_id', station_id, "
            "'quality_reason', quality_reason, "
            "'source_topic', source_topic, 'source_partition', source_partition, "
            "'source_offset', source_offset, 'raw_value', raw_value)) AS value",
        )
        return featured_output, dlq_output

    # ---------------------------------------------------------
    # 7. Single Structured Streaming source query, two Kafka sinks
    # ---------------------------------------------------------

    checkpoint_dir = spark_cfg.get(
        "checkpoint_location", "checkpoints/spark_features"
    )
    checkpoint_path = ROOT / f"{checkpoint_dir}_{args.station}"
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    def publish_microbatch(batch_df, batch_id: int) -> None:
        """Publish featured and DLQ records from the same cached source batch.

        Keeping both output writes inside one foreachBatch callback eliminates
        the second Kafka source query and duplicate parse/validation plan. The
        source checkpoint advances only after both sink writes complete.
        If either write fails, Spark retries this batch on restart (at-least-once
        delivery); downstream event/actual stores are idempotent by source key
        or timestamp.
        """
        batch_df.persist()
        try:
            featured_output, dlq_output = build_batch_outputs(batch_df)
            (
                featured_output.write
                .format("kafka")
                .option("kafka.bootstrap.servers", kafka_bootstrap)
                .option("topic", station["featured_topic"])
                .save()
            )
            (
                dlq_output.write
                .format("kafka")
                .option("kafka.bootstrap.servers", kafka_bootstrap)
                .option("topic", dlq_topic)
                .save()
            )
            print(
                f"Spark batch completed: batchId={batch_id} | "
                "featured+DLQ sinks acknowledged",
                flush=True,
            )
        finally:
            batch_df.unpersist()

    query = (
        parsed.writeStream
        .foreachBatch(publish_microbatch)
        .option("checkpointLocation", str(checkpoint_path))
        .outputMode("append")
        .start()
    )

    print(
        f"Spark streaming started for '{args.station}': "
        f"{station['raw_topic']} -> {station['featured_topic']} | DLQ={dlq_topic} "
        "(one source query, shared micro-batch)",
        flush=True,
    )
    try:
        while True:
            if not query.isActive:
                raise RuntimeError("Spark source/output query stopped unexpectedly.")
            query.awaitTermination(10000)
            progress = query.lastProgress
            if progress:
                print(
                    "Spark progress: " + json.dumps(
                        {
                            "batchId": progress.get("batchId"),
                            "inputRowsPerSecond": progress.get("inputRowsPerSecond"),
                            "processedRowsPerSecond": progress.get("processedRowsPerSecond"),
                            "numInputRows": progress.get("numInputRows"),
                        }
                    ),
                    flush=True,
                )
    finally:
        query.stop()


if __name__ == "__main__":
    main()
