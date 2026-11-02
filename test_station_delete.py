"""Small isolated integration test for station-scoped deletion (no Kafka required)."""
from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import backend.pipeline_manager as pm
import scripts.station_replay_helper as helper


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_station_delete_preserves_other_station_data() -> None:
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        (root / "config").mkdir()
        (root / "logs").mkdir()
        (root / "data" / "processed").mkdir(parents=True)
        (root / "data" / "raw" / "stations" / "amprion").mkdir(parents=True)
        (root / "data" / "raw" / "stations" / "tennet").mkdir(parents=True)
        (root / "models" / "amprion_seq2seq_10Y").mkdir(parents=True)
        (root / "models" / "tennet").mkdir(parents=True)
        (root / "checkpoints").mkdir()
        (root / "checkpoints" / "spark_features_amprion").mkdir()

        config = {
            "active_station": "amprion",
            "stations": {
                "amprion": {
                    "input_file": "data/raw/stations/amprion/amprion_stream.csv",
                    "history_file": "data/raw/stations/amprion/amprion_history.csv",
                    "training_file": "data/raw/stations/amprion/amprion_training.csv",
                    "model_checkpoint": "models/amprion_seq2seq_10Y/model.pt",
                    "model_metadata": "models/amprion_seq2seq_10Y/metadata.json",
                    "raw_topic": "amprion-raw",
                    "featured_topic": "amprion-featured",
                },
                "tennet": {
                    "input_file": "data/raw/stations/tennet/tennet_stream.csv",
                    "history_file": "data/raw/stations/tennet/tennet_history.csv",
                    "training_file": "data/raw/stations/tennet/tennet_training.csv",
                    "model_checkpoint": "models/tennet/model.pt",
                    "model_metadata": "models/tennet/metadata.json",
                    "raw_topic": "tennet-raw",
                    "featured_topic": "tennet-featured",
                },
            },
        }
        import yaml
        (root / "config" / "stations.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        (root / "config" / "spark.yaml").write_text("checkpoint_location: checkpoints/spark_features\n", encoding="utf-8")
        (root / "config" / "kafka.yaml").write_text("bootstrap_servers: 127.0.0.1:9092\n", encoding="utf-8")

        for station in ("amprion", "tennet"):
            for suffix in ("stream", "history", "training"):
                write_csv(root / "data" / "raw" / "stations" / station / f"{station}_{suffix}.csv", [
                    {"utc_timestamp": "2026-01-01T00:00:00Z", "demand": "1", "temperature": "2"}
                ])
            model_folder = "amprion_seq2seq_10Y" if station == "amprion" else station
            (root / "models" / model_folder / "model.pt").write_bytes(b"model")
            (root / "models" / model_folder / "metadata.json").write_text("{}", encoding="utf-8")

        write_csv(root / "data" / "processed" / "actuals.csv", [
            {"station_id": "amprion", "utc_timestamp": "a", "actual_demand": "1"},
            {"station_id": "tennet", "utc_timestamp": "b", "actual_demand": "2"},
        ])
        write_csv(root / "data" / "processed" / "forecasts.csv", [
            {"station_id": "amprion", "forecast_timestamp_utc": "a", "predicted_demand": "1"},
            {"station_id": "tennet", "forecast_timestamp_utc": "b", "predicted_demand": "2"},
        ])

        # Redirect module paths to the isolated fixture.
        pm.ROOT = root
        pm.LOG_DIR = root / "logs"
        pm.STATION_CONFIG = root / "config" / "stations.yaml"
        pm.SPARK_CONFIG = root / "config" / "spark.yaml"
        pm.KAFKA_CONFIG = root / "config" / "kafka.yaml"
        helper.ROOT = root
        helper.PROCESSED_DIR = root / "data" / "processed"

        manager = pm.PipelineManager()
        manager.kafka_available = lambda: True
        manager._assert_station_stopped = lambda station: None
        manager._assert_station_training_stopped = lambda station: None
        manager._assert_other_predictors_stopped = lambda station: None
        manager._ensure_reset_topics = lambda station, raw, featured: {"created": [], "topics": [raw, featured]}
        manager._clear_topic_records = lambda raw, featured: "Validated Kafka partition low_watermarks"

        result = manager.delete_station("amprion", "amprion")
        assert result["deleted"] is True
        assert not (root / "data" / "raw" / "stations" / "amprion").exists()
        assert not (root / "models" / "amprion_seq2seq_10Y").exists()
        assert (root / "data" / "raw" / "stations" / "tennet" / "tennet_stream.csv").exists()
        assert (root / "models" / "tennet" / "model.pt").exists()

        for filename in ("actuals.csv", "forecasts.csv"):
            with (root / "data" / "processed" / filename).open(newline="", encoding="utf-8-sig") as stream:
                rows = list(csv.DictReader(stream))
            assert [row["station_id"] for row in rows] == ["tennet"], (filename, rows)

        updated = pm.read_station_config()
        assert list(updated["stations"]) == ["tennet"]
        assert updated["active_station"] == "tennet"
        print("PASS: selected station removed; other station data and output rows preserved.")


