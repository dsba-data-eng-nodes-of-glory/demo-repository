from pathlib import Path


def test_spark_routes_featured_and_rejected_records_in_one_kafka_write():
    source = (Path(__file__).resolve().parents[1] / "src/spark/streaming.py").read_text(encoding="utf-8")
    assert source.count(".writeStream") == 1
    assert source.count(".save()") == 1
    assert ".foreachBatch(publish_microbatch)" in source
    assert 'lit(station["featured_topic"])).otherwise(lit(dlq_topic)).alias("topic")' in source
