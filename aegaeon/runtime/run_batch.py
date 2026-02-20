from __future__ import annotations

from typing import Any

from aegaeon.types import Batch, BatchResult, Request


class BatchRunner:
    """Adapter boundary for running batched inference."""

    def __init__(self, engine: Any | None = None) -> None:
        self._engine = engine

    def run(self, batch: Batch) -> BatchResult:
        requests = batch.requests
        phase_name = batch.phase

        if not requests:
            return BatchResult(outputs={})

        if self._engine is None:
            if phase_name in {"decode", "colocate"}:
                outputs: dict[str, Any] = {}
                for req in requests:
                    tok = self._offline_next_token(req)
                    outputs[req.request_id] = {"phase": phase_name, "tokens": [tok]}
                return BatchResult(outputs=outputs)
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

    @staticmethod
    def _offline_next_token(req: Request) -> int:
        if not req.input_ids:
            return 0
        params = req.sampling_params if isinstance(req.sampling_params, dict) else {}
        total = int(params.get("_offline_total_steps", 0))
        remaining = int(params.get("_offline_remaining_steps", 1))
        step_idx = 0 if total <= 0 else max(total - remaining - 1, 0)
        return int(req.input_ids[step_idx % len(req.input_ids)])


__all__ = ["BatchRunner", "BatchResult"]
