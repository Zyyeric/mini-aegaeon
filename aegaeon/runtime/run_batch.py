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
        sampling_params = self._normalize_sampling_params(sampling_params)
        model = requests[0].model
        if hasattr(self._engine, "select_model"):
            self._engine.select_model(model)
        weight_manager = getattr(self._engine, "weight_manager", None)
        if weight_manager is not None and hasattr(weight_manager, "select_model"):
            weight_manager.select_model(model)
        try:
            raw = self._engine.generate(prompts, sampling_params=sampling_params)
        except TypeError:
            raw = self._engine.generate(prompts, sampling_params)
        return BatchResult(outputs=self._normalize_outputs(requests, phase_name, raw))

    @staticmethod
    def _normalize_sampling_params(sampling_params: Any) -> Any:
        # mini-sgl backend expects minisgl.core.SamplingParams, while offline flow uses plain dicts.
        try:
            from aegaeon.backend import SamplingParams as MiniSGLSamplingParams
        except Exception:
            return sampling_params

        def convert_one(p: Any) -> Any:
            if isinstance(p, MiniSGLSamplingParams):
                return p
            if not isinstance(p, dict):
                return p
            max_tokens = p.get("max_tokens")
            if not isinstance(max_tokens, int) or max_tokens <= 0:
                max_tokens = p.get("max_new_tokens")
            if not isinstance(max_tokens, int) or max_tokens <= 0:
                max_tokens = 1
            temperature = p.get("temperature", 0.0)
            if not isinstance(temperature, (int, float)):
                temperature = 0.0
            top_k = p.get("top_k", -1)
            if not isinstance(top_k, int):
                top_k = -1
            top_p = p.get("top_p", 1.0)
            if not isinstance(top_p, (int, float)):
                top_p = 1.0
            ignore_eos = p.get("ignore_eos", True)
            if not isinstance(ignore_eos, bool):
                ignore_eos = True
            return MiniSGLSamplingParams(
                temperature=float(temperature),
                top_k=top_k,
                top_p=float(top_p),
                ignore_eos=ignore_eos,
                max_tokens=max_tokens,
            )

        if isinstance(sampling_params, list):
            return [convert_one(x) for x in sampling_params]
        return convert_one(sampling_params)

    @staticmethod
    def _normalize_outputs(batch: list[Request], phase: str, raw: Any) -> dict[str, Any]:
        if isinstance(raw, list) and len(raw) == len(batch):
            out: dict[str, Any] = {}
            for req, result in zip(batch, raw, strict=False):
                if isinstance(result, str):
                    out[req.request_id] = {"phase": phase, "text": result}
                elif isinstance(result, dict):
                    # Preserve backend fields (e.g., token_ids) for accurate offline TTFT/TBT accounting.
                    out[req.request_id] = {"phase": phase, **result}
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
