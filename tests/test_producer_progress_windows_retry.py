"""Regression checks for transient Windows locks in the producer cursor."""
from pathlib import Path
import errno
import json
import pandas as pd
import pytest
import src.producers.csv_producer as cp


def _save(path: Path, source: Path, when: str):
    cp.save_producer_progress(path, "amprion", source, "amprion-raw", pd.Timestamp(when))


def test_transient_checkpoint_rename_failure_recovers(monkeypatch, tmp_path):
    state = tmp_path / "producer_progress.json"
    source = tmp_path / "input.csv"
    _save(state, source, "2026-10-01T12:00:00Z")
    real_replace = cp.os.replace
    tries = []
    monkeypatch.setattr(cp.time, "sleep", lambda *_: None)

    def flaky(src, dst):
        tries.append((src, dst))
        if len(tries) <= 3:
            raise PermissionError(13, "Access is denied", str(dst))
        return real_replace(src, dst)

    monkeypatch.setattr(cp.os, "replace", flaky)
    _save(state, source, "2026-10-01T12:15:00Z")
    assert len(tries) == 4
    assert cp.load_producer_progress(state, "amprion", source, "amprion-raw") == pd.Timestamp("2026-10-01T12:15:00Z")
    assert not list(tmp_path.glob("*.tmp"))


def test_permanent_checkpoint_lock_preserves_last_good_cursor(monkeypatch, tmp_path):
    state = tmp_path / "producer_progress.json"
    source = tmp_path / "input.csv"
    _save(state, source, "2026-10-01T12:00:00Z")
    monkeypatch.setattr(cp.time, "sleep", lambda *_: None)
    monkeypatch.setattr(cp.os, "replace", lambda *_: (_ for _ in ()).throw(PermissionError(13, "Access denied")))
    with pytest.raises(RuntimeError, match="old checkpoint remains intact"):
        _save(state, source, "2026-10-01T12:15:00Z")
    assert cp.load_producer_progress(state, "amprion", source, "amprion-raw") == pd.Timestamp("2026-10-01T12:00:00Z")
    assert len(list(tmp_path.glob("*.tmp"))) == 1


def test_non_permission_error_not_retried(monkeypatch, tmp_path):
    source = tmp_path / "input.csv"
    state = tmp_path / "producer_progress.json"
    attempts = []
    def bad(*_):
        attempts.append(1)
        raise OSError(errno.ENOSPC, "No space left")
    monkeypatch.setattr(cp.os, "replace", bad)
    with pytest.raises(OSError, match="No space left"):
        _save(state, source, "2026-10-01T12:00:00Z")
    assert len(attempts) == 1
