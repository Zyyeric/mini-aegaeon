from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Lock, RLock
from typing import Any, Mapping


@dataclass(slots=True)
class ModelCacheEntry:
    model: str
    nbytes: int
    state_dict: dict[str, Any]
    ref_count: int = 0
    raw_chunk_sizes: list[int] = field(default_factory=list)


class ModelCache:
    """Pinned-CPU model state cache.

    Each model is stored as a flat state_dict mapping name -> CPU pinned tensor.
    """

    def __init__(self, budget_bytes: int) -> None:
        self._budget = budget_bytes
        self._used = 0
        self._entries: dict[str, ModelCacheEntry] = {}
        self._lock = RLock()
        self._model_locks: dict[str, Lock] = defaultdict(Lock)

    def put_state_dict(
        self,
        model: str,
        state_dict: Mapping[str, Any],
        raw_chunk_sizes: list[int] | None = None,
    ) -> ModelCacheEntry:
        with self._lock:
            pinned_state = self._pin_state_dict(state_dict)
            nbytes = self._state_dict_nbytes(pinned_state)

            old = self._entries.get(model)
            old_bytes = old.nbytes if old is not None else 0
            next_used = self._used - old_bytes + nbytes
            if next_used > self._budget:
                raise MemoryError("model cache budget exceeded")

            entry = ModelCacheEntry(
                model=model,
                nbytes=nbytes,
                state_dict=pinned_state,
                ref_count=0 if old is None else old.ref_count,
                raw_chunk_sizes=list(raw_chunk_sizes or []),
            )
            self._entries[model] = entry
            self._used = next_used
            return entry

    def ensure_model(
        self,
        model: str,
        loader_fn: Any,
    ) -> tuple[ModelCacheEntry, bool]:
        with self._model_lock(model):
            with self._lock:
                existing = self._entries.get(model)
                if existing is not None:
                    return existing, False
            state_dict, raw_chunk_sizes = loader_fn()
            entry = self.put_state_dict(
                model=model,
                state_dict=state_dict,
                raw_chunk_sizes=raw_chunk_sizes,
            )
            return entry, True

    def get_state_dict(self, model: str) -> dict[str, Any] | None:
        with self._lock:
            entry = self._entries.get(model)
            if entry is None:
                return None
            return entry.state_dict

    def get_nbytes(self, model: str) -> int | None:
        with self._lock:
            entry = self._entries.get(model)
            return None if entry is None else entry.nbytes

    def has_model(self, model: str) -> bool:
        with self._lock:
            return model in self._entries

    def acquire(self, model: str) -> int:
        with self._lock:
            entry = self._entries.get(model)
            if entry is None:
                raise KeyError(f"model not found in cache: {model}")
            entry.ref_count += 1
            return entry.ref_count

    def release(self, model: str) -> int:
        with self._lock:
            entry = self._entries.get(model)
            if entry is None:
                raise KeyError(f"model not found in cache: {model}")
            if entry.ref_count <= 0:
                raise ValueError(f"model ref_count already 0: {model}")
            entry.ref_count -= 1
            return entry.ref_count

    def evict(self, model: str) -> None:
        with self._model_lock(model):
            with self._lock:
                old = self._entries.get(model)
                if old is None:
                    return
                if old.ref_count > 0:
                    raise ValueError(f"cannot evict model with active references: {model}")
                self._entries.pop(model, None)
                self._used -= old.nbytes

    def models(self) -> set[str]:
        with self._lock:
            return set(self._entries.keys())

    def stats(self) -> dict[str, int]:
        with self._lock:
            tensors = sum(len(entry.state_dict) for entry in self._entries.values())
            refs = sum(entry.ref_count for entry in self._entries.values())
            return {
                "budget_bytes": self._budget,
                "used_bytes": self._used,
                "entries": len(self._entries),
                "tensors": tensors,
                "ref_count": refs,
            }

    @contextmanager
    def _model_lock(self, model: str):
        lock = self._model_locks[model]
        lock.acquire()
        try:
            yield
        finally:
            lock.release()

    @staticmethod
    def _pin_state_dict(state_dict: Mapping[str, Any]) -> dict[str, Any]:
        try:
            import torch
        except Exception as exc:  # pragma: no cover - runtime env dependent
            raise RuntimeError("torch is required to store pinned model state_dict") from exc

        out: dict[str, Any] = {}
        for name, tensor in state_dict.items():
            if not isinstance(name, str) or not name:
                raise ValueError("state_dict keys must be non-empty strings")
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"state_dict[{name}] must be torch.Tensor")

            cpu_tensor = tensor.detach().to("cpu")
            if not cpu_tensor.is_pinned():
                cpu_tensor = cpu_tensor.pin_memory()
            out[name] = cpu_tensor
        return out

    @staticmethod
    def _state_dict_nbytes(state_dict: Mapping[str, Any]) -> int:
        total = 0
        for tensor in state_dict.values():
            if hasattr(tensor, "numel") and hasattr(tensor, "element_size"):
                total += int(tensor.numel()) * int(tensor.element_size())
        return total
