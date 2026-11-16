"""Cross-platform process lock for a station service."""
from __future__ import annotations

import json
import os
from pathlib import Path

def acquire_station_lock(lock_path: Path, service: str = "producer"):
    """Acquire an OS-released exclusive lock for one service at a station.

    The lock is advisory and process-scoped: Windows releases it when the
    process exits or crashes, so a stale lock file does not block recovery.
    The lock file itself is intentionally retained between runs.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(" ")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        raise RuntimeError(
            f"A {service} for this station is already running (lock: {lock_path}). "
            f"Stop the existing {service} before starting another one."
        ) from exc

    handle.seek(0)
    handle.truncate()
    handle.write(json.dumps({"pid": os.getpid(), "station_lock": str(lock_path)}))
    handle.flush()
    os.fsync(handle.fileno())
    return handle


def release_station_lock(handle) -> None:
    if handle is None:
        return
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()
