from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import ScheduledRequest


@dataclass(slots=True)
class BatchResult:
    outputs: dict[str, Any]


class BatchRunner:
    """Adapter boundary for running batched inference with vLLM.

    The skeleton keeps the call shape stable while deferring real engine wiring.
    """

    def __init__(self, engine: Any | None = None) -> None:
        self._engine = engine

    def run(self, batch: list[ScheduledRequest], phase: str) -> BatchResult:
        if self._engine is None:
            # Placeholder path for local testing and policy development.
            fake = {req.request_id: {"phase": phase, "tokens": [1]} for req in batch}
            return BatchResult(outputs=fake)

        # Real shape when binding to vLLM:
        # return self._engine.run_batch(batch=batch, phase=phase)
        raise NotImplementedError("Bind a concrete vLLM engine implementation")
