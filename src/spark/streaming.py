
"""Generic Spark Structured Streaming feature extraction for a configured station."""

import argparse
from pathlib import Path

import yaml

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

    spark = (
        SparkSession.builder
        .appName(
            f"{spark_cfg.get('app_name', 'electricity-feature-extraction')}"
            f"-{args.station}"
        )
        .master(spark_cfg.get("master", "local[2]"))
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.sql.session.timeZone", "UTC")
        .config(
            "spark.jars.packages",
            "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.6",
        )
        .getOrCreate()
    )

    spark.conf.set("spark.sql.session.timeZone", "UTC")
    spark.sparkContext.setLogLevel("WARN")

    # ---------------------------------------------------------
    # 4. Read raw Kafka topic
    # ---------------------------------------------------------

    raw = (
        spark.readStream
        .format("kafka")
        .option(
            "kafka.bootstrap.servers",
            kafka_cfg["bootstrap_servers"],
        )
        .option(
            "subscribe",
            station["raw_topic"],
        )
        .option(
            "startingOffsets",
            spark_cfg.get("starting_offsets", "earliest"),
        )
        .load()
    )

    # ---------------------------------------------------------
    # 5. Parse JSON messages
    # ---------------------------------------------------------

    parsed = (
        raw.select(
            from_json(
                col("value").cast("string"),
                schema,
            ).alias("record")
        )
        .select("record.*")
    )

    # ---------------------------------------------------------
    # 6. Validate incoming records
    # ---------------------------------------------------------

    ts = to_timestamp(col("utc_timestamp"))

    valid = (
        parsed
        .filter(
            (col("station_id") == lit(args.station))
            & col("utc_timestamp").isNotNull()
            & col("demand").isNotNull()
            & col("temperature").isNotNull()
            & (col("demand") >= 0)
            & ts.isNotNull()
        )
    )

    # ---------------------------------------------------------
    # 7. Calendar feature engineering
    # ---------------------------------------------------------

    minute_of_day = (
        hour(ts) * lit(60)
        + minute(ts)
    )

    # Spark dayofweek: Sunday=1.
    # Convert to Monday=0 ... Sunday=6.
    weekday_zero = (
        dayofweek(ts) + lit(5)
    ) % lit(7)

    year_day_zero = dayofyear(ts) - lit(1)

    two_pi = lit(2.0 * 3.141592653589793)

    featured = (
        valid

        .withColumn(
            "timestamp_utc",
            ts,
        )

        .withColumn(
            "hour_sin",
            sin(two_pi * minute_of_day / lit(1440.0)),
        )

        .withColumn(
            "hour_cos",
            cos(two_pi * minute_of_day / lit(1440.0)),
        )

        .withColumn(
            "weekday_sin",
            sin(two_pi * weekday_zero / lit(7.0)),
        )

        .withColumn(
            "weekday_cos",
            cos(two_pi * weekday_zero / lit(7.0)),
        )

        .withColumn(
            "year_sin",
            sin(two_pi * year_day_zero / lit(365.2425)),
        )

        .withColumn(
            "year_cos",
            cos(two_pi * year_day_zero / lit(365.2425)),
        )

        .select(
            "station_id",
            "utc_timestamp",
            "timestamp_utc",
            "demand",
            "temperature",
            "forecast_temperature",

            # Calendar features
            "hour_sin",
            "hour_cos",
            "weekday_sin",
            "weekday_cos",
            "year_sin",
            "year_cos",

            # Replay metadata
            "is_simulated",
            "data_mode",
            "replay_cycle",
        )
    )

    # ---------------------------------------------------------
    # 8. Prepare Kafka output
    # ---------------------------------------------------------

    kafka_output = featured.selectExpr(
        "CAST(station_id AS STRING) AS key",
        "to_json(struct(*)) AS value",
    )

    # ---------------------------------------------------------
    # 9. Station-specific checkpoint
    # ---------------------------------------------------------

    checkpoint_dir = spark_cfg.get(
        "checkpoint_location",
        "checkpoints/spark_features",
    )

    checkpoint_path = (
        ROOT / f"{checkpoint_dir}_{args.station}"
    )

    checkpoint_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # 10. Publish featured observations
    # ---------------------------------------------------------

    query = (
        kafka_output.writeStream
        .format("kafka")
        .option(
            "kafka.bootstrap.servers",
            kafka_cfg["bootstrap_servers"],
        )
        .option(
            "topic",
            station["featured_topic"],
        )
        .option(
            "checkpointLocation",
            str(checkpoint_path),
        )
        .outputMode("append")
        .start()
    )

    print(
        f"Spark streaming started for '{args.station}': "
        f"{station['raw_topic']} -> {station['featured_topic']}"
    )

    query.awaitTermination()


if __name__ == "__main__":
    main()
