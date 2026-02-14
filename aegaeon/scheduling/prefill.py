from __future__ import annotations

import queue
from typing import Any

from .base import ScheduledRequest
from .run_batch import BatchRunner, BatchResult


class PrefillScheduler:
    """Simple FIFO prefill scheduler."""

    def __init__(self, runner: BatchRunner, max_batch_size: int = 16) -> None:
        self._runner = runner
        self._max_batch_size = max_batch_size
        self._q: queue.Queue[ScheduledRequest] = queue.Queue()

    def submit(self, req: ScheduledRequest) -> None:
        self._q.put(req)

    def step(self) -> BatchResult | None:
        if self._q.empty():
            return None

        batch: list[ScheduledRequest] = []
        while len(batch) < self._max_batch_size and not self._q.empty():
            batch.append(self._q.get_nowait())
        return self._runner.run(batch=batch, phase="prefill")

    def stats(self) -> dict[str, Any]:
        return {"queue_depth": self._q.qsize(), "max_batch_size": self._max_batch_size}
