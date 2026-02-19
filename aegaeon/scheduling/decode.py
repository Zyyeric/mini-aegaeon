from __future__ import annotations

from collections import deque
from typing import Any

from aegaeon.types import Batch, BatchResult, Request


class DecodeScheduler:
    """Round-robin decode scheduler over active sequences."""

    def __init__(self, runner: Any, max_batch_size: int = 16) -> None:
        self._runner = runner
        self._max_batch_size = max_batch_size
        self._active: deque[Request] = deque()

    def submit(self, req: Request) -> None:
        self._active.append(req)

    def step(self) -> BatchResult | None:
        if not self._active:
            return None

        batch: list[Request] = []
        while self._active and len(batch) < self._max_batch_size:
            req = self._active.popleft()
            batch.append(req)
            # In a real engine integration this would depend on EOS/finish state.
            self._active.append(req)

        return self._runner.run(Batch(requests=batch, phase="decode"))

    def stats(self) -> dict[str, Any]:
        return {"active_sequences": len(self._active), "max_batch_size": self._max_batch_size}
