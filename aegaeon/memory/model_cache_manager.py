from __future__ import annotations

from typing import Any

from aegaeon.memory.weight_manager import TensorMap, WeightManager

from .model_cache import ModelCache


class ModelCacheManager(WeightManager):
    """Weight manager backed by ModelCache pinned-CPU entries."""

    def __init__(
        self,
        cache: ModelCache,
        device: str = "cuda",
        dtype: Any | None = None,
        non_blocking: bool = True,
    ) -> None:
        self._cache = cache
        self._device = device
        self._dtype = dtype
        self._non_blocking = non_blocking
        self._active_model: str | None = None
        self._gpu_state_dict: dict[str, Any] | None = None

    def select_model(self, model: str) -> None:
        self._active_model = model

    def prepare_for_batch(self) -> TensorMap:
        try:
            import torch
        except Exception as exc:  # pragma: no cover - runtime env dependent
            raise RuntimeError("torch is required for ModelCacheManager") from exc

        if self._active_model is None:
            raise ValueError("active model is not set; call select_model(model) first")

        cpu_state_dict = self._cache.get_state_dict(self._active_model)
        if cpu_state_dict is None:
            raise KeyError(f"model not found in cache: {self._active_model}")

        device = torch.device(self._device)
        gpu_state: dict[str, Any] = {}
        for name, tensor in cpu_state_dict.items():
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"state_dict[{name}] must be torch.Tensor")
            moved = tensor.to(device=device, non_blocking=self._non_blocking)
            if self._dtype is not None:
                moved = moved.to(dtype=self._dtype)
            gpu_state[name] = moved

        self._gpu_state_dict = gpu_state
        return gpu_state

    def after_batch(self) -> None:
        if self._gpu_state_dict is None:
            return

        # CPU pinned cache is the source of truth. After batch, release GPU copies only.
        self._gpu_state_dict = None
