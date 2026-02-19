from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

from aegaeon.memory.weight_manager import TensorMap, WeightManager


class Inference_Backend(ABC):
    """Base backend wrapper with weight-manager lifecycle hooks."""

    def __init__(self, *args: Any, weight_manager: WeightManager, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.weight_manager = weight_manager

    def prepare_for_batch(self) -> TensorMap:
        return self.weight_manager.prepare_for_batch()

    def after_batch(self) -> None:
        self.weight_manager.after_batch()

    @abstractmethod
    def load_weights_for_batch(self, state_dict: TensorMap) -> None:
        pass

    def generate(self, prompts: Sequence[str] | Sequence[list[int]], sampling_params: Any) -> Any:
        state_dict = self.prepare_for_batch()
        self.load_weights_for_batch(state_dict)
        try:
            return super().generate(prompts, sampling_params)
        finally:
            self.after_batch()
