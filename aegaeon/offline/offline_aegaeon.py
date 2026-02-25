from __future__ import annotations

import time
from dataclasses import dataclass, field
from uuid import uuid4

from aegaeon.config import ProxyConfig
from aegaeon.proxy.metadata_store import InstanceStatus, RequestPhase
from aegaeon.proxy.proxy import Proxy
from aegaeon.proxy.router import RequestEnvelope
from aegaeon.runtime import InstanceRuntime
from aegaeon.server.launch import build_local_instances
from aegaeon.types import Request


@dataclass(slots=True)
class _OfflineRequestState:
    request_id: str
    model: str
    instance_id: str
    instance_mode: str
    queued: bool
    enqueued_ns: int
    remaining_steps: int
    first_token_ns: int | None = None
    last_emit_ns: int | None = None
    emitted_tokens: int = 0
    tbt_samples_ns: list[int] = field(default_factory=list)
    output: dict | None = None


class OfflineAegaeon:

    def __init__(
        self,
        deployment_mode: str = "colocation",
        prefill_count: int = 0,
        decode_count: int = 0,
        colocated_count: int = 1,
        metadata_backend: str = "shared_memory",
        proxy_cfg: ProxyConfig | None = None,
        model_cache_budget_bytes: int = 8 * 1024 * 1024 * 1024,
        backend: str = "none",
        backend_model: str | None = None,
        backend_memory_ratio: float = 0.5,
        backend_max_live_workers: int = 1,
        backend_model_switching: bool = False,
    ) -> None:
        self.proxy = Proxy._build(
            proxy_cfg
            or ProxyConfig(
                metadata_backend=metadata_backend,
                deployment_mode=deployment_mode,
            ),
            create=True,
        )
        self.instances: dict[str, InstanceRuntime] = build_local_instances(
            self.proxy,
            deployment_mode=deployment_mode,
            prefill_count=prefill_count,
            decode_count=decode_count,
            colocated_count=colocated_count,
            model_cache_budget_bytes=model_cache_budget_bytes,
            backend=backend,
            backend_model=backend_model,
            backend_memory_ratio=backend_memory_ratio,
            backend_max_live_workers=backend_max_live_workers,
            backend_model_switching=backend_model_switching,
        )
        self.counter = 0
        self._pending: dict[str, _OfflineRequestState] = {}
        self._completed: dict[str, dict] = {}

    def request(self, payload: dict) -> dict:
        request_id = self.enqueue(payload)
        completed = self.run_until_complete({request_id})
        return completed[request_id]

    def enqueue(self, payload: dict) -> str:
        model = payload.get("model")
        if not isinstance(model, str) or not model:
            raise ValueError("`model` is required")

        request_id = str(payload.get("request_id") or f"req-{uuid4().hex}")
        req = RequestEnvelope(
            request_id=request_id,
            model=model,
            payload=payload,
            phase=RequestPhase.PREFILL,
        )

        decision = self.proxy.route(req)
        runtime = self.instances.get(decision.instance_id)
        if runtime is None:
            raise KeyError(f"instance not found: {decision.instance_id}")

        runtime.ensure_model_ready(model)

        prompt = payload.get("prompt")
        prompt_token_ids = payload.get("prompt_token_ids")
        if prompt is None and isinstance(payload.get("messages"), list):
            prompt = " ".join(str(m.get("content", "")) for m in payload["messages"] if isinstance(m, dict))
        if isinstance(prompt_token_ids, list) and prompt_token_ids:
            input_ids = [int(x) for x in prompt_token_ids if isinstance(x, int)]
            if not input_ids:
                input_ids = [0]
        else:
            prompt_text = str(prompt or "")
            input_ids = [ord(c) % 256 for c in prompt_text] or [0]
        uid = self.counter
        self.counter += 1
        sampling_params = payload.get("sampling_params", {})
        if not isinstance(sampling_params, dict):
            sampling_params = {}
        sampling_params = dict(sampling_params)
        sampling_params["_offline_remaining_steps"] = self._remaining_steps(payload, sampling_params)
        sampling_params["_offline_total_steps"] = int(sampling_params["_offline_remaining_steps"])

        runtime.submit(
            Request(
                request_id=request_id,
                model=model,
                input_ids=input_ids,
                uid=uid,
                sampling_params=sampling_params,
            )
        )
        self._pending[request_id] = _OfflineRequestState(
            request_id=request_id,
            model=model,
            instance_id=decision.instance_id,
            instance_mode=runtime.cfg.mode,
            queued=decision.queued,
            enqueued_ns=time.perf_counter_ns(),
            remaining_steps=(
                int(sampling_params["_offline_remaining_steps"])
                if runtime.cfg.mode in {"decode", "colocated"}
                else 1
            ),
        )
        return request_id

    def run_until_complete(self, request_ids: set[str] | None = None, max_idle_loops: int = 1000) -> dict[str, dict]:
        target = set(request_ids) if request_ids is not None else set(self._pending.keys())
        if not target:
            return {}

        idle_loops = 0
        while not target.issubset(self._completed.keys()):
            progressed = self._step_once()
            if progressed:
                idle_loops = 0
                continue
            idle_loops += 1
            if idle_loops > max_idle_loops:
                waiting = sorted(target - set(self._completed.keys()))
                raise RuntimeError(f"offline loop stalled; waiting for {waiting}")

        return {rid: self._completed[rid] for rid in target}

    def benchmark_by_model(self) -> dict[str, dict]:
        grouped: dict[str, dict[str, list[float]]] = {}
        for result in self._completed.values():
            model = str(result["model"])
            bucket = grouped.setdefault(model, {"ttft_ms": [], "tbt_ms": []})
            ttft_ms = result.get("ttft_ms")
            if isinstance(ttft_ms, (int, float)):
                bucket["ttft_ms"].append(float(ttft_ms))
            for sample in result.get("tbt_ms_samples", []):
                bucket["tbt_ms"].append(float(sample))

        summary: dict[str, dict] = {}
        for model, samples in grouped.items():
            ttft = samples["ttft_ms"]
            tbt = samples["tbt_ms"]
            summary[model] = {
                "requests": len(ttft),
                "ttft_ms_avg": (sum(ttft) / len(ttft)) if ttft else None,
                "tbt_ms_avg": (sum(tbt) / len(tbt)) if tbt else None,
                "ttft_ms_samples": ttft,
                "tbt_ms_samples": tbt,
            }
        return summary

    def _step_once(self) -> bool:
        progressed = False
        for instance_id, runtime in self.instances.items():
            result = runtime.step()
            if result is None:
                self._publish_status(instance_id, runtime)
                continue
            progressed = True
            now_ns = time.perf_counter_ns()
            for request_id, output in result.outputs.items():
                state = self._pending.get(request_id)
                if state is None:
                    continue
                self._record_emission(state, now_ns, output)
                if state.instance_mode in {"decode", "colocated"}:
                    state.remaining_steps = max(state.remaining_steps - 1, 0)
                else:
                    state.remaining_steps = 0
                if state.remaining_steps == 0:
                    self._finalize_request(state)
            self._publish_status(instance_id, runtime)
        return progressed

    def _record_emission(self, state: _OfflineRequestState, ts_ns: int, output: dict | object) -> None:
        token_count = self._count_tokens(output)
        state.output = output if isinstance(output, dict) else {"result": output}
        if state.first_token_ns is None:
            state.first_token_ns = ts_ns
        elif state.last_emit_ns is not None:
            delta = ts_ns - state.last_emit_ns
            if token_count > 0:
                per_token = max(delta // token_count, 1)
                state.tbt_samples_ns.extend([per_token] * token_count)
        state.last_emit_ns = ts_ns
        state.emitted_tokens += max(token_count, 0)

    def _finalize_request(self, state: _OfflineRequestState) -> None:
        completed_ns = state.last_emit_ns or time.perf_counter_ns()
        first_ns = state.first_token_ns or completed_ns
        ttft_ms = (first_ns - state.enqueued_ns) / 1_000_000.0
        self._completed[state.request_id] = {
            "request_id": state.request_id,
            "model": state.model,
            "instance_id": state.instance_id,
            "queued": state.queued,
            "output": state.output,
            "ttft_ms": ttft_ms,
            "tbt_ms_samples": [x / 1_000_000.0 for x in state.tbt_samples_ns],
            "tbt_ms_avg": (
                (sum(state.tbt_samples_ns) / len(state.tbt_samples_ns) / 1_000_000.0)
                if state.tbt_samples_ns
                else None
            ),
            "tokens_emitted": state.emitted_tokens,
            "completed_ns": completed_ns,
        }
        self._pending.pop(state.request_id, None)

    def _publish_status(self, instance_id: str, runtime: InstanceRuntime) -> None:
        scheduler_stats = runtime.stats().get("scheduler", {})
        queue_depth = scheduler_stats.get("queue_depth", scheduler_stats.get("active_sequences", 0))
        self.proxy.update_instance_status(
            instance_id,
            InstanceStatus(
                current_models=runtime.requested_models(),
                queue_depth=int(queue_depth),
            ),
        )

    @staticmethod
    def _remaining_steps(payload: dict, sampling_params: dict) -> int:
        for key in ("_offline_remaining_steps", "max_new_tokens", "max_tokens"):
            value = sampling_params.get(key)
            if value is None:
                value = payload.get(key)
            if isinstance(value, int) and value > 0:
                return value
        return 1

    @staticmethod
    def _count_tokens(output: dict | object) -> int:
        if isinstance(output, dict):
            token_ids = output.get("token_ids")
            if isinstance(token_ids, list):
                return len(token_ids)
            tokens = output.get("tokens")
            if isinstance(tokens, list):
                return len(tokens)
        return 0

    def clear_completed(self) -> None:
        self._completed.clear()

    def pending_request_ids(self) -> set[str]:
        return set(self._pending.keys())

    def completed_request_ids(self) -> set[str]:
        return set(self._completed.keys())

    def preload_models(self, models: list[str], per_instance: bool = True) -> None:
        """Preload model weights into DRAM before benchmarking.

        If `per_instance` is True, each local instance marks each model as ready,
        which also updates routing metadata current_models.
        """
        targets = [m for m in models if isinstance(m, str) and m]
        if not targets:
            return

        if per_instance:
            for instance_id, runtime in self.instances.items():
                for model in targets:
                    runtime.ensure_model_ready(model)
                self._publish_status(instance_id, runtime)
            return

        # Load once through a single runtime (shared model cache), then publish status.
        first_runtime = next(iter(self.instances.values()), None)
        if first_runtime is not None:
            for model in targets:
                first_runtime.ensure_model_ready(model)
        for instance_id, runtime in self.instances.items():
            self._publish_status(instance_id, runtime)

    def close(self, unlink: bool = True) -> None:
        for rt in self.instances.values():
            if hasattr(rt, "close"):
                rt.close()
        self.proxy.shutdown(unlink=unlink)


def offline_aegaeon(**kwargs: object) -> OfflineAegaeon:
    return OfflineAegaeon(**kwargs)
