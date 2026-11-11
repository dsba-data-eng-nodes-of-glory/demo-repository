import json
from pathlib import Path

from src.quality import event_store


def test_jsonl_dedupe_cache_loads_once_and_suppresses_duplicate(tmp_path, monkeypatch):
    data_dir = tmp_path / "data" / "quality"
    data_dir.mkdir(parents=True)
    target = data_dir / "quality_events.jsonl"
    target.write_text(
        json.dumps({"event_id": "old-1"}) + "\n" +
        json.dumps({"event_id": "old-2"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(event_store, "ROOT", tmp_path)
    monkeypatch.setattr(event_store, "_CACHE_READY", False)
    monkeypatch.setattr(event_store, "_RECENT_IDS", set())
    monkeypatch.setattr(event_store, "_RECENT_ORDER", event_store.deque())
    event_store.record_quality_event({"event_id": "old-2", "message": "duplicate"})
    assert len(target.read_text(encoding="utf-8").splitlines()) == 2

    event_store.record_quality_event({"event_id": "new-1", "message": "new"})
    event_store.record_quality_event({"event_id": "new-1", "message": "duplicate"})
    rows = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
    assert [row["event_id"] for row in rows] == ["old-1", "old-2", "new-1"]


def test_jsonl_dedupe_cache_does_not_rescan_after_initialization(tmp_path, monkeypatch):
    data_dir = tmp_path / "data" / "quality"
    data_dir.mkdir(parents=True)
    target = data_dir / "quality_events.jsonl"
    target.write_text(json.dumps({"event_id": "seed"}) + "\n", encoding="utf-8")
    monkeypatch.setattr(event_store, "ROOT", tmp_path)
    monkeypatch.setattr(event_store, "_CACHE_READY", False)
    monkeypatch.setattr(event_store, "_RECENT_IDS", set())
    monkeypatch.setattr(event_store, "_RECENT_ORDER", event_store.deque())

    event_store.record_quality_event({"event_id": "one"})
    target.write_text("corrupt-but-not-reloaded\n", encoding="utf-8")
    event_store.record_quality_event({"event_id": "two"})
    rows = target.read_text(encoding="utf-8").splitlines()
    assert rows == [
        "corrupt-but-not-reloaded",
        json.dumps({"event_id": "two"}, default=str),
    ]
