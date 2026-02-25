from __future__ import annotations

import multiprocessing as mp
import os
import traceback
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from aegaeon.memory.weight_manager import WeightManager


def _ensure_minisgl_importable() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    minisgl_python_dir = repo_root / "third-party" / "mini-sglang" / "python"
    path_str = str(minisgl_python_dir)
    if path_str not in os.sys.path:
        os.sys.path.insert(0, path_str)


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
    req_q: mp.Queue,
    resp_q: mp.Queue,
    worker_idx: int,
) -> None:
    os.environ["MASTER_PORT"] = str(32000 + worker_idx)
    _ensure_minisgl_importable()
    import torch
    from minisgl.core import SamplingParams
    from minisgl.llm import LLM

    llm = LLM(
        model,
        dtype=torch.float16,
        offload_linear_weight_to_cpu=False,
        memory_ratio=memory_ratio,
        model_switching=model_switching,
        cuda_graph_max_bs=0,
    )
    gpu_loaded = True

    def _park_to_cpu() -> None:
        nonlocal gpu_loaded
        if not gpu_loaded:
            return
        state = llm.engine.model.state_dict()
        cpu_state = {
            k: v.detach().to("cpu", non_blocking=False).pin_memory()
            for k, v in state.items()
        }
        llm.engine.model.load_state_dict(cpu_state)
        torch.cuda.synchronize(llm.engine.device)
        torch.cuda.empty_cache()
        gpu_loaded = False

    def _unpark_to_gpu() -> None:
        nonlocal gpu_loaded
        if gpu_loaded:
            return
        state = llm.engine.model.state_dict()
        dev = llm.engine.device
        gpu_state = {
            k: v.detach().to(device=dev, non_blocking=True)
            for k, v in state.items()
        }
        llm.engine.model.load_state_dict(gpu_state)
        torch.cuda.synchronize(dev)
        gpu_loaded = True

    try:
        while True:
            item = req_q.get()
            if item is None:
                break
            cmd = item.get("cmd", "generate")
            if cmd == "park":
                try:
                    _park_to_cpu()
                    resp_q.put({"ok": True})
                except Exception:
                    resp_q.put({"ok": False, "err": traceback.format_exc()})
                continue
            if cmd == "unpark":
                try:
                    _unpark_to_gpu()
                    resp_q.put({"ok": True})
                except Exception:
                    resp_q.put({"ok": False, "err": traceback.format_exc()})
                continue

            prompts = item["prompts"]
            payload = item["sampling_params"]
            if not gpu_loaded:
                _unpark_to_gpu()

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
    ) -> None:
        self._memory_ratio = memory_ratio
        self._max_live_workers = max(1, int(max_live_workers))
        self._model_switching = bool(model_switching)
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
    def _send_control(req_q: mp.Queue, resp_q: mp.Queue, cmd: str) -> None:
        req_q.put({"cmd": cmd})
        msg = resp_q.get()
        if not msg.get("ok", False):
            raise RuntimeError(msg.get("err", f"worker command failed: {cmd}"))

    def select_model(self, model: str) -> None:
        self._step_counter += 1
        prev_model = self._active_model
        self._active_model = model
        if prev_model is not None and prev_model != model and prev_model in self._workers:
            p_prev, req_prev, resp_prev = self._workers[prev_model]
            if not p_prev.is_alive():
                raise RuntimeError(f"mini-sgl worker died for model: {prev_model}")
            self._send_control(req_prev, resp_prev, "park")
        if model not in self._workers:
            req_q: mp.Queue = self._ctx.Queue()
            resp_q: mp.Queue = self._ctx.Queue()
            idx = self._worker_counter
            self._worker_counter += 1
            p = self._ctx.Process(
                target=_worker_main,
                args=(model, self._memory_ratio, self._model_switching, req_q, resp_q, idx),
                daemon=True,
            )
            p.start()
            self._workers[model] = (p, req_q, resp_q)
        p, req_q, resp_q = self._workers[model]
        if not p.is_alive():
            raise RuntimeError(f"mini-sgl worker died for model: {model}")
        self._send_control(req_q, resp_q, "unpark")
        self._last_used_step[model] = self._step_counter

    def generate(self, prompts: list[list[int]], sampling_params: Any) -> Any:
        if self._active_model is None:
            raise RuntimeError("Active model is not selected. Call select_model(model) first.")
        if self._active_model not in self._workers:
            self.select_model(self._active_model)
        p, req_q, resp_q = self._workers[self._active_model]
        if not p.is_alive():
            raise RuntimeError(f"mini-sgl worker died for model: {self._active_model}")
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

    def shutdown(self) -> None:
        for model in list(self._workers.keys()):
            self._stop_worker(model)
