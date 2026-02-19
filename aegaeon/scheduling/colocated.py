from __future__ import annotations

from collections import deque
from typing import Any

from aegaeon.types import Batch, BatchResult, Request


class ColocatedScheduler:
    """Scheduler for colocated prefill/decode workers.

    It greedily batches requests that share the same model as the head request,
    up to `max_batch_size`.
    """

    def __init__(self, runner: Any, max_batch_size: int = 16) -> None:
        self._runner = runner
        self._max_batch_size = max_batch_size
        self._queue: deque[Request] = deque()

    def submit(self, req: Request) -> None:
        self._queue.append(req)

    def step(self) -> BatchResult | None:
        if not self._queue:
            return None

        head = self._queue.popleft()
        target_model = head.model
        batch: list[Request] = [head]
        deferred: deque[Request] = deque()

        while self._queue and len(batch) < self._max_batch_size:
            req = self._queue.popleft()
            if req.model == target_model:
                batch.append(req)
            else:
                deferred.append(req)

        while deferred:
            self._queue.appendleft(deferred.pop())

        return self._runner.run(Batch(requests=batch, phase="colocate"))

    def stats(self) -> dict[str, Any]:
        return {"queue_depth": len(self._queue), "max_batch_size": self._max_batch_size}
