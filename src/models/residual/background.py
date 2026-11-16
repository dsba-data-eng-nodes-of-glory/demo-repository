"""Run residual inference outside the station's Kafka processing loop."""
from __future__ import annotations

from threading import Condition, Thread
from typing import Callable
import logging

LOG = logging.getLogger(__name__)


class ResidualWorker:
    """Keep the current assessment and at most the newest waiting origin."""

    def __init__(self, assess: Callable):
        self._assess = assess
        self._condition = Condition()
        self._waiting: tuple | None = None
        self._closing = False
        self._thread: Thread | None = None

    def submit(self, *assessment) -> None:
        with self._condition:
            if self._closing:
                return
            if self._thread is None:
                self._thread = Thread(target=self._run, name="residual-assessment", daemon=True)
                self._thread.start()
            self._waiting = assessment
            self._condition.notify()

    def close(self) -> None:
        with self._condition:
            self._closing = True
            self._condition.notify()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._closing or self._waiting is not None)
                if self._waiting is None:
                    return
                assessment, self._waiting = self._waiting, None
            try:
                self._assess(*assessment)
            except Exception:
                LOG.exception("Background residual assessment failed.")
