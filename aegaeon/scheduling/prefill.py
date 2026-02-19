from __future__ import annotations

import queue
from typing import Any

from aegaeon.types import Batch, BatchResult, Request


class PrefillScheduler:
    """Simple FIFO prefill scheduler."""

    def __init__(self, runner: Any, max_batch_size: int = 16) -> None:
        self._runner = runner
        self._max_batch_size = max_batch_size
        self._q: queue.Queue[Request] = queue.Queue()

    def submit(self, req: Request) -> None:
        self._q.put(req)

    def step(self) -> BatchResult | None:
        if self._q.empty():
            return None

        batch: list[Request] = []
        while len(batch) < self._max_batch_size and not self._q.empty():
            batch.append(self._q.get_nowait())
        return self._runner.run(Batch(requests=batch, phase="prefill"))

    def stats(self) -> dict[str, Any]:
        return {"queue_depth": self._q.qsize(), "max_batch_size": self._max_batch_size}
