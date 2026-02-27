from __future__ import annotations

import multiprocessing as mp
import os
import queue as pyqueue
import time
import traceback
from dataclasses import asdict, is_dataclass
from typing import Any

from aegaeon.memory.weight_manager import TensorMap, WeightManager
from aegaeon.utils import nvtx_range


def _sampling_to_payload(sampling_params: Any) -> Any:
    def _one(sp: Any) -> dict[str, Any]:
        if isinstance(sp, dict):
            max_tokens = sp.get("max_tokens", sp.get("max_new_tokens", 1))
            return {
                "temperature": float(sp.get("temperature", 0.0)),
                "top_k": int(sp.get("top_k", -1)),
                "top_p": float(sp.get("top_p", 1.0)),
                "ignore_eos": bool(sp.get("ignore_eos", True)),
                "max_tokens": int(max_tokens if isinstance(max_tokens, int) else 1),
            }
        if is_dataclass(sp):
            d = asdict(sp)
            return {
                "temperature": float(d.get("temperature", 0.0)),
                "top_k": int(d.get("top_k", -1)),
                "top_p": float(d.get("top_p", 1.0)),
                "ignore_eos": bool(d.get("ignore_eos", True)),
                "max_tokens": int(d.get("max_tokens", 1)),
            }
        return {
            "temperature": float(getattr(sp, "temperature", 0.0)),
            "top_k": int(getattr(sp, "top_k", -1)),
            "top_p": float(getattr(sp, "top_p", 1.0)),
            "ignore_eos": bool(getattr(sp, "ignore_eos", True)),
            "max_tokens": int(getattr(sp, "max_tokens", 1)),
        }

    if isinstance(sampling_params, list):
        return [_one(x) for x in sampling_params]
    return _one(sampling_params)


def _attach_token_timestamps(out: Any, token_timestamps: list[float]) -> Any:
    """Attach per-token timestamps to worker outputs when possible."""
    if not token_timestamps:
        return out
    if not isinstance(out, list) or not out:
        return out

    if len(out) == 1 and isinstance(out[0], dict):
        out[0]["_token_timestamps_s"] = list(token_timestamps)
        return out

    token_counts: list[int] = []
    for item in out:
        if not isinstance(item, dict):
            token_counts = []
            break
        ids = item.get("token_ids")
        if isinstance(ids, list):
            token_counts.append(len(ids))
            continue
        toks = item.get("tokens")
        if isinstance(toks, list):
            token_counts.append(len(toks))
            continue
        token_counts = []
        break

    if token_counts and sum(token_counts) == len(token_timestamps):
        offset = 0
        for item, cnt in zip(out, token_counts, strict=False):
            if isinstance(item, dict):
                item["_token_timestamps_s"] = list(token_timestamps[offset : offset + cnt])
            offset += cnt
        return out

    # Fallback: attach to first output entry.
    if isinstance(out[0], dict):
        out[0]["_token_timestamps_s"] = list(token_timestamps)
    return out


def _set_tensor_by_state_key(root: Any, key: str, tensor: Any) -> None:
    parts = key.split(".")
    cur = root
    for p in parts[:-1]:
        if p.isdigit():
            idx = int(p)
            if hasattr(cur, "op_list"):
                cur = cur.op_list[idx]
            elif isinstance(cur, list):
                cur = cur[idx]
            else:
                raise RuntimeError(f"cannot resolve numeric path '{p}' in key '{key}'")
        else:
            cur = getattr(cur, p)
    last = parts[-1]
    if last.isdigit():
        idx = int(last)
        if hasattr(cur, "op_list"):
            cur.op_list[idx] = tensor
        elif isinstance(cur, list):
            cur[idx] = tensor
        else:
            raise RuntimeError(f"cannot assign numeric leaf '{last}' in key '{key}'")
    else:
        setattr(cur, last, tensor)


def _evict_model_weights_to_meta(model: Any, torch_mod: Any) -> int:
    """Move currently loaded model tensors off GPU by replacing with meta tensors."""
    evicted = 0
    state = model.state_dict()
    for name, param in state.items():
        if not isinstance(param, torch_mod.Tensor):
            continue
        if not param.is_cuda:
            continue
        meta_t = torch_mod.empty(param.shape, dtype=param.dtype, device="meta")
        _set_tensor_by_state_key(model, name, meta_t)
        evicted += 1
    return evicted


def _worker_main(
    model: str,
    memory_ratio: float,
    model_switching: bool,
    model_switching_budget_model_path: str | None,
    num_page_override: int | None,
    use_dummy_weight: bool,
    req_q: mp.Queue,
    resp_q: mp.Queue,
    worker_idx: int,
) -> None:
    import os

    os.environ["MASTER_PORT"] = str(32000 + worker_idx)
    import torch
    from minisgl.core import SamplingParams
    from minisgl.llm import LLM

    llm = LLM(
        model,
        dtype=torch.bfloat16,
        offload_linear_weight_to_cpu=False,
        memory_ratio=memory_ratio,
        model_switching=model_switching,
        model_switching_budget_model_path=model_switching_budget_model_path,
        num_page_override=num_page_override,
        use_dummy_weight=use_dummy_weight,
        cuda_graph_max_bs=0,
    )
    resident_model: str | None = None

    try:
        while True:
            item = req_q.get()
            if item is None:
                break
            cmd = item.get("cmd", "generate")
            if cmd == "load_weights_for_batch":
                with nvtx_range("offline_colocate/worker/load_weights_cmd"):
                    try:
                        state_dict = item["state_dict"]
                        target_model = item.get("model")
                        if (
                            isinstance(target_model, str)
                            and resident_model is not None
                            and resident_model == target_model
                        ):
                            # Active model already resident on this worker.
                            resp_q.put({"ok": True, "skipped": True})
                            continue
                        # Prevent transient 2x peak memory on repeated loads:
                        # drop current model tensors before staging the next state dict on GPU.
                        with nvtx_range("offline_colocate/worker/evict_before_load"):
                            _ = _evict_model_weights_to_meta(llm.engine.model, torch)
                            torch.cuda.empty_cache()
                        gpu_state_dict: dict[str, Any] = {}
                        with nvtx_range("offline_colocate/worker/h2d_state_dict"):
                            for name, tensor in state_dict.items():
                                if not isinstance(tensor, torch.Tensor):
                                    raise TypeError(f"state_dict[{name}] must be torch.Tensor")
                                gpu_state_dict[name] = tensor.to(device=llm.engine.device, non_blocking=True)
                        with nvtx_range("offline_colocate/worker/load_state_dict"):
                            llm.engine.model.load_state_dict(gpu_state_dict)
                        resident_model = str(target_model) if isinstance(target_model, str) else model
                        resp_q.put({"ok": True})
                    except Exception:
                        resp_q.put({"ok": False, "err": traceback.format_exc()})
                continue
            if cmd == "evict_weights":
                with nvtx_range("offline_colocate/worker/evict_weights_cmd"):
                    try:
                        evicted = _evict_model_weights_to_meta(llm.engine.model, torch)
                        if evicted > 0:
                            torch.cuda.empty_cache()
                        resident_model = None
                        resp_q.put({"ok": True, "evicted": int(evicted)})
                    except Exception:
                        resp_q.put({"ok": False, "err": traceback.format_exc()})
                continue

            prompts = item["prompts"]
            payload = item["sampling_params"]

            if isinstance(payload, list):
                sampling_params = [SamplingParams(**x) for x in payload]
            else:
                sampling_params = SamplingParams(**payload)

            with nvtx_range("offline_colocate/worker/generate_cmd"):
                try:
                    original_send = llm.send_result
                    token_timestamps: list[float] = []

                    def instrumented_send(reply: Any):
                        now = time.perf_counter()
                        try:
                            for _ in reply:
                                token_timestamps.append(now)
                        except TypeError:
                            pass
                        return original_send(reply)

                    llm.send_result = instrumented_send
                    try:
                        with nvtx_range("offline_colocate/worker/llm_generate"):
                            out = llm.generate(prompts, sampling_params)
                    finally:
                        llm.send_result = original_send
                    out = _attach_token_timestamps(out, token_timestamps)
                    resp_q.put({"ok": True, "out": out})
                except Exception:
                    resp_q.put({"ok": False, "err": traceback.format_exc()})
    finally:
        try:
            llm.shutdown()
        except Exception:
            pass


class MiniSGLMultiBackend:
    """
    One mini-sgl backend process per model.
    Aegaeon scheduler selects model via `select_model(model)` before `generate(...)`.
    """

    def __init__(
        self,
        *,
        weight_manager: WeightManager,
        memory_ratio: float = 0.5,
        max_live_workers: int | None = None,
        model_switching: bool = False,
        model_switching_budget_model_path: str | None = None,
        global_kv_budget_models: list[str] | None = None,
        global_kv_worker_count: int | None = None,
        use_dummy_weight: bool = False,
    ) -> None:
        self._memory_ratio = memory_ratio
        self._max_live_workers = max_live_workers if (max_live_workers is None or max_live_workers > 0) else 1
        self._model_switching = bool(model_switching)
        self._model_switching_budget_model_path = model_switching_budget_model_path
        self._global_kv_budget_models = list(global_kv_budget_models or [])
        self._global_kv_worker_count = global_kv_worker_count
        self._use_dummy_weight = bool(use_dummy_weight)
        self._active_model: str | None = None
        self._ctx = mp.get_context("spawn")
        self._workers: dict[str, tuple[mp.Process, mp.Queue, mp.Queue]] = {}
        self._worker_counter = 0
        self._last_used_step: dict[str, int] = {}
        self._step_counter = 0
        self._num_pages_override_by_model: dict[str, int] = {}
        self._global_per_worker_kv_budget_bytes: int | None = None
        self.weight_manager = weight_manager

    @staticmethod
    def _cache_per_page_bytes(model: str) -> int:
        from minisgl.utils import cached_load_hf_config

        cfg = cached_load_hf_config(model)
        hidden_size = int(getattr(cfg, "hidden_size"))
        num_hidden_layers = int(getattr(cfg, "num_hidden_layers"))
        num_attention_heads = int(getattr(cfg, "num_attention_heads"))
        num_kv_heads = int(getattr(cfg, "num_key_value_heads", num_attention_heads))
        head_dim = hidden_size // max(num_attention_heads, 1)
        dtype_itemsize = 2  # bfloat16
        page_size = 1
        return (
            2  # key + value
            * head_dim
            * num_kv_heads
            * page_size
            * dtype_itemsize
            * num_hidden_layers
        )

    def _compute_global_num_pages_override(self, model: str) -> int | None:
        if not self._model_switching:
            return None
        if model in self._num_pages_override_by_model:
            return self._num_pages_override_by_model[model]

        models = self._global_kv_budget_models or [model]
        worker_count = self._global_kv_worker_count or len(models)
        worker_count = max(int(worker_count), 1)

        try:
            import torch
            from minisgl.models import estimate_hf_weight_nbytes_from_safetensors
        except Exception:
            return None

        if self._global_per_worker_kv_budget_bytes is None:
            free_bytes = int(torch.cuda.mem_get_info()[0])
            reserve_bytes = 0
            for m in models:
                try:
                    reserve_bytes = max(reserve_bytes, int(estimate_hf_weight_nbytes_from_safetensors(m)))
                except Exception:
                    continue
            usable_bytes = max(0, free_bytes - reserve_bytes)
            global_kv_budget = int(self._memory_ratio * usable_bytes)
            self._global_per_worker_kv_budget_bytes = max(global_kv_budget // worker_count, 0)
        per_worker_kv_budget = self._global_per_worker_kv_budget_bytes

        cache_per_page = self._cache_per_page_bytes(model)
        if cache_per_page <= 0:
            return None
        num_pages = max(2, per_worker_kv_budget // cache_per_page)
        self._num_pages_override_by_model[model] = int(num_pages)
        return int(num_pages)

    def _stop_worker(self, model: str) -> None:
        item = self._workers.pop(model, None)
        self._last_used_step.pop(model, None)
        if item is None:
            return
        p, req_q, _resp_q = item
        try:
            req_q.put(None)
        except Exception:
            pass
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()
            p.join(timeout=2)

    @staticmethod
    def _wait_worker_msg(
        p: mp.Process,
        resp_q: mp.Queue,
        *,
        cmd_name: str,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        with nvtx_range(f"offline_colocate/backend/wait_worker_msg:{cmd_name}"):
            if timeout_s is None:
                timeout_s = float(os.getenv("AEGAEON_WORKER_TIMEOUT_S", "300"))
            deadline = time.time() + timeout_s
            while True:
                if not p.is_alive():
                    raise RuntimeError(
                        f"mini-sgl worker died while waiting for '{cmd_name}' response "
                        f"(exitcode={p.exitcode})"
                    )
                remaining = deadline - time.time()
                if remaining <= 0:
                    raise TimeoutError(f"timed out waiting for worker response to '{cmd_name}'")
                try:
                    msg = resp_q.get(timeout=min(1.0, remaining))
                    if not isinstance(msg, dict):
                        raise RuntimeError(f"invalid worker response for '{cmd_name}': {type(msg)}")
                    return msg
                except pyqueue.Empty:
                    continue

    @staticmethod
    def _send_load_weights(
        p: mp.Process,
        req_q: mp.Queue,
        resp_q: mp.Queue,
        model: str,
        state_dict: TensorMap,
    ) -> None:
        req_q.put({"cmd": "load_weights_for_batch", "model": model, "state_dict": state_dict})
        msg = MiniSGLMultiBackend._wait_worker_msg(p, resp_q, cmd_name="load_weights_for_batch")
        if not msg.get("ok", False):
            raise RuntimeError(msg.get("err", "worker command failed: load_weights_for_batch"))

    @staticmethod
    def _send_evict_weights(
        p: mp.Process,
        req_q: mp.Queue,
        resp_q: mp.Queue,
    ) -> None:
        req_q.put({"cmd": "evict_weights"})
        msg = MiniSGLMultiBackend._wait_worker_msg(p, resp_q, cmd_name="evict_weights")
        if not msg.get("ok", False):
            raise RuntimeError(msg.get("err", "worker command failed: evict_weights"))

    def prepare_for_batch(self) -> TensorMap:
        return self.weight_manager.prepare_for_batch()

    def after_batch(self) -> None:
        with nvtx_range("offline_colocate/backend/after_batch"):
            self.weight_manager.after_batch()
            # Keep only the active model weights resident on GPU.
            for m, worker in list(self._workers.items()):
                if m == self._active_model:
                    continue
                p, req_q, resp_q = worker
                if not p.is_alive():
                    continue
                with nvtx_range("offline_colocate/backend/after_batch_evict_other_model"):
                    self._send_evict_weights(p, req_q, resp_q)

    def load_weights_for_batch(self, state_dict: TensorMap) -> None:
        with nvtx_range("offline_colocate/backend/load_weights_for_batch"):
            if self._active_model is None:
                raise RuntimeError("Active model is not selected. Call select_model(model) first.")
            if self._active_model not in self._workers:
                self.select_model(self._active_model)
            p, req_q, resp_q = self._workers[self._active_model]
            if not p.is_alive():
                raise RuntimeError(f"mini-sgl worker died for model: {self._active_model}")
            self._send_load_weights(p, req_q, resp_q, self._active_model, state_dict)

    def select_model(self, model: str) -> None:
        with nvtx_range(f"offline_colocate/backend/select_model:{model}"):
            self._step_counter += 1
            prev_model = self._active_model
            self._active_model = model
            if model not in self._workers:
                if self._max_live_workers is not None:
                    while len(self._workers) >= self._max_live_workers:
                        # Evict least-recently-used non-target worker before spawning a new one.
                        victims = [m for m in self._workers.keys() if m != model]
                        if not victims:
                            break
                        victim = min(victims, key=lambda m: self._last_used_step.get(m, -1))
                        self._stop_worker(victim)
                req_q: mp.Queue = self._ctx.Queue()
                resp_q: mp.Queue = self._ctx.Queue()
                idx = self._worker_counter
                self._worker_counter += 1
                p = self._ctx.Process(
                    target=_worker_main,
                    args=(
                        model,
                        self._memory_ratio,
                        self._model_switching,
                        self._model_switching_budget_model_path,
                        self._compute_global_num_pages_override(model),
                        self._use_dummy_weight,
                        req_q,
                        resp_q,
                        idx,
                    ),
                    daemon=True,
                )
                with nvtx_range("offline_colocate/backend/spawn_worker"):
                    p.start()
                self._workers[model] = (p, req_q, resp_q)
            p, req_q, resp_q = self._workers[model]
            if not p.is_alive():
                raise RuntimeError(f"mini-sgl worker died for model: {model}")
            self._last_used_step[model] = self._step_counter
            # On model switch, proactively evict previous model weights from GPU.
            if prev_model and prev_model != model and prev_model in self._workers:
                pp, preq, presp = self._workers[prev_model]
                if pp.is_alive():
                    with nvtx_range("offline_colocate/backend/evict_prev_model_on_switch"):
                        self._send_evict_weights(pp, preq, presp)

    def generate(self, prompts: list[list[int]], sampling_params: Any) -> Any:
        with nvtx_range("offline_colocate/backend/generate"):
            if self._active_model is None:
                raise RuntimeError("Active model is not selected. Call select_model(model) first.")
            if self._active_model not in self._workers:
                self.select_model(self._active_model)
            p, req_q, resp_q = self._workers[self._active_model]
            if not p.is_alive():
                raise RuntimeError(f"mini-sgl worker died for model: {self._active_model}")
            with nvtx_range("offline_colocate/backend/prepare_for_batch"):
                state_dict = self.prepare_for_batch()
            self.load_weights_for_batch(state_dict)
            try:
                with nvtx_range("offline_colocate/backend/send_generate_request"):
                    req_q.put(
                        {
                            "prompts": prompts,
                            "sampling_params": _sampling_to_payload(sampling_params),
                        }
                    )
                msg = self._wait_worker_msg(p, resp_q, cmd_name="generate")
                if not msg.get("ok", False):
                    raise RuntimeError(msg.get("err", "Unknown worker error"))
                return msg["out"]
            finally:
                self.after_batch()

    def shutdown(self) -> None:
        for model in list(self._workers.keys()):
            self._stop_worker(model)
