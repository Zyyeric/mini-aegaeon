from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ModelCacheEntry:
    model: str
    nbytes: int
    buffer: object


class ModelCache:
    """
    Pinned CPU model cache placeholder.

    When `torch` is available, buffers are allocated with `pin_memory=True`.
    Fallback is a Python bytearray to keep the API runnable without extra deps.
    """

    def __init__(self, budget_bytes: int) -> None:
        self._budget = budget_bytes
        self._used = 0
        self._entries: dict[str, ModelCacheEntry] = {}

    def put(self, model: str, nbytes: int) -> ModelCacheEntry:
        if model in self._entries:
            return self._entries[model]
        if self._used + nbytes > self._budget:
            raise MemoryError("model cache budget exceeded")

        buf = self._alloc_pinned(nbytes)
        entry = ModelCacheEntry(model=model, nbytes=nbytes, buffer=buf)
        self._entries[model] = entry
        self._used += nbytes
        return entry

    def get(self, model: str) -> ModelCacheEntry | None:
        return self._entries.get(model)

    def evict(self, model: str) -> None:
        old = self._entries.pop(model, None)
        if old:
            self._used -= old.nbytes

    def stats(self) -> dict[str, int]:
        return {
            "budget_bytes": self._budget,
            "used_bytes": self._used,
            "entries": len(self._entries),
        }

    @staticmethod
    def _alloc_pinned(nbytes: int) -> object:
        try:
            import torch

            return torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
        except Exception:
            return bytearray(nbytes)
