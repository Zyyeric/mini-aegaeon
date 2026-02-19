from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

TensorMap = Mapping[str, Any]


class WeightManager(ABC):
    """Pre/post batch hook provider for model-state management."""

    @abstractmethod
    def prepare_for_batch(self) -> TensorMap:
        """Return a flat state_dict mapping parameter name -> tensor-like object."""
        pass

    @abstractmethod
    def after_batch(self) -> None:
        pass
