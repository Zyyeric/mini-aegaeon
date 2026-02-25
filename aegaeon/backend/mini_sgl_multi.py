from __future__ import annotations

import multiprocessing as mp
import traceback
from dataclasses import asdict, is_dataclass
from typing import Any

from aegaeon.memory.weight_manager import TensorMap, WeightManager


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


def _worker_main(
    model: str,
    memory_ratio: float,
    model_switching: bool,
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
        dtype=torch.float16,
        offload_linear_weight_to_cpu=False,
        memory_ratio=memory_ratio,
        model_switching=model_switching,
        use_dummy_weight=use_dummy_weight,
        cuda_graph_max_bs=0,
    )

    try:
        while True:
            item = req_q.get()
            if item is None:
                break
            cmd = item.get("cmd", "generate")
            if cmd == "load_weights_for_batch":
                try:
                    state_dict = item["state_dict"]
                    llm.engine.model.load_state_dict(state_dict)
                    resp_q.put({"ok": True})
                except Exception:
                    resp_q.put({"ok": False, "err": traceback.format_exc()})
                continue

            prompts = item["prompts"]
            payload = item["sampling_params"]

            if isinstance(payload, list):
                sampling_params = [SamplingParams(**x) for x in payload]
            else:
                sampling_params = SamplingParams(**payload)

            try:
                out = llm.generate(prompts, sampling_params)
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
        max_live_workers: int = 1,
        model_switching: bool = False,
        use_dummy_weight: bool = False,
    ) -> None:
        self._memory_ratio = memory_ratio
        self._max_live_workers = max(1, int(max_live_workers))
        self._model_switching = bool(model_switching)
        self._use_dummy_weight = bool(use_dummy_weight)
        self._active_model: str | None = None
        self._ctx = mp.get_context("spawn")
        self._workers: dict[str, tuple[mp.Process, mp.Queue, mp.Queue]] = {}
        self._worker_counter = 0
        self._last_used_step: dict[str, int] = {}
        self._step_counter = 0
        # Keep Aegaeon's weight-manager contract with ModelCache-backed manager.
        self.weight_manager = weight_manager

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
    def _send_load_weights(req_q: mp.Queue, resp_q: mp.Queue, state_dict: TensorMap) -> None:
        req_q.put({"cmd": "load_weights_for_batch", "state_dict": state_dict})
        msg = resp_q.get()
        if not msg.get("ok", False):
            raise RuntimeError(msg.get("err", "worker command failed: load_weights_for_batch"))

    def prepare_for_batch(self) -> TensorMap:
        return self.weight_manager.prepare_for_batch()

    def after_batch(self) -> None:
        self.weight_manager.after_batch()

    def load_weights_for_batch(self, state_dict: TensorMap) -> None:
        if self._active_model is None:
            raise RuntimeError("Active model is not selected. Call select_model(model) first.")
        if self._active_model not in self._workers:
            self.select_model(self._active_model)
        p, req_q, resp_q = self._workers[self._active_model]
        if not p.is_alive():
            raise RuntimeError(f"mini-sgl worker died for model: {self._active_model}")
        self._send_load_weights(req_q, resp_q, state_dict)

    def select_model(self, model: str) -> None:
        self._step_counter += 1
        self._active_model = model
        if model not in self._workers:
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
                    self._use_dummy_weight,
                    req_q,
                    resp_q,
                    idx,
                ),
                daemon=True,
            )
            p.start()
            self._workers[model] = (p, req_q, resp_q)
        p, req_q, resp_q = self._workers[model]
        if not p.is_alive():
            raise RuntimeError(f"mini-sgl worker died for model: {model}")
        self._last_used_step[model] = self._step_counter

    def generate(self, prompts: list[list[int]], sampling_params: Any) -> Any:
        if self._active_model is None:
            raise RuntimeError("Active model is not selected. Call select_model(model) first.")
        if self._active_model not in self._workers:
            self.select_model(self._active_model)
        p, req_q, resp_q = self._workers[self._active_model]
        if not p.is_alive():
            raise RuntimeError(f"mini-sgl worker died for model: {self._active_model}")
        state_dict = self.prepare_for_batch()
        self.load_weights_for_batch(state_dict)
        try:
            req_q.put(
                {
                    "prompts": prompts,
                    "sampling_params": _sampling_to_payload(sampling_params),
                }
            )
            msg = resp_q.get()
            if not msg.get("ok", False):
                raise RuntimeError(msg.get("err", "Unknown worker error"))
            return msg["out"]
        finally:
            self.after_batch()

    def shutdown(self) -> None:
        for model in list(self._workers.keys()):
            self._stop_worker(model)
