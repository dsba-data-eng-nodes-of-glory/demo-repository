from pathlib import Path


def test_spark_pipeline_uses_one_source_query_for_featured_and_dlq_sinks():
    source = (Path(__file__).resolve().parents[1] / "src/spark/streaming.py").read_text(
        encoding="utf-8"
    )
    assert source.count(".writeStream") == 1
    assert ".foreachBatch(publish_microbatch)" in source
    assert 'option("topic", station["featured_topic"])' in source
    assert 'option("topic", dlq_topic)' in source
    assert "batch_df.persist()" in source
    assert "batch_df.unpersist()" in source
