from __future__ import annotations

from typing import Any

from aegaeon.types import Batch, BatchResult, Request


class BatchRunner:
    """Adapter boundary for running batched inference."""

    def __init__(self, engine: Any | None = None) -> None:
        self._engine = engine

    def run(self, batch: list[Request] | Batch, phase: str | None = None) -> BatchResult:
        batch_obj = batch if isinstance(batch, Batch) else Batch(requests=batch, phase=str(phase or ""))
        requests = batch_obj.requests
        phase_name = batch_obj.phase

        if not requests:
            return BatchResult(outputs={})

        if self._engine is None:
            return BatchResult(
                outputs={
                    req.request_id: {"phase": phase_name, "tokens": [int(t) for t in req.input_ids]}
                    for req in requests
                }
            )

        prompts = [req.input_ids for req in requests]
        sampling_params = (
            requests[0].sampling_params
            if len(requests) == 1
            else [req.sampling_params for req in requests]
        )
        model = requests[0].model
        weight_manager = getattr(self._engine, "weight_manager", None)
        if weight_manager is not None and hasattr(weight_manager, "select_model"):
            weight_manager.select_model(model)
        try:
            raw = self._engine.generate(prompts, sampling_params=sampling_params)
        except TypeError:
            raw = self._engine.generate(prompts, sampling_params)
        return BatchResult(outputs=self._normalize_outputs(requests, phase_name, raw))

    @staticmethod
    def _normalize_outputs(batch: list[Request], phase: str, raw: Any) -> dict[str, Any]:
        if isinstance(raw, list) and len(raw) == len(batch):
            out: dict[str, Any] = {}
            for req, result in zip(batch, raw, strict=False):
                if isinstance(result, str):
                    out[req.request_id] = {"phase": phase, "text": result}
                elif hasattr(result, "text"):
                    out[req.request_id] = {"phase": phase, "text": str(result.text)}
                else:
                    out[req.request_id] = {"phase": phase, "result": result}
            return out

        return {req.request_id: {"phase": phase, "result": raw} for req in batch}


__all__ = ["BatchRunner", "BatchResult"]
