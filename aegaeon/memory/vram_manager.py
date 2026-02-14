from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ModelResidency:
    model: str
    bytes_reserved: int


class VRAMManager:
    """Tracks GPU memory reservations owned by this instance."""

    def __init__(self, budget_bytes: int) -> None:
        self._budget = budget_bytes
        self._used = 0
        self._models: dict[str, ModelResidency] = {}

    def reserve(self, model: str, bytes_needed: int) -> bool:
        if model in self._models:
            return True
        if self._used + bytes_needed > self._budget:
            return False

        self._models[model] = ModelResidency(model=model, bytes_reserved=bytes_needed)
        self._used += bytes_needed
        return True

    def release(self, model: str) -> None:
        resident = self._models.pop(model, None)
        if resident:
            self._used -= resident.bytes_reserved

    def stats(self) -> dict[str, int]:
        return {
            "budget_bytes": self._budget,
            "used_bytes": self._used,
            "free_bytes": self._budget - self._used,
            "loaded_models": len(self._models),
        }
