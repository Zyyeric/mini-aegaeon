from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from aegaeon.config import MemoryConfig, RuntimeConfig
from aegaeon.memory import HFWeightLoader, ModelCache
from aegaeon.proxy.metadata_store import (
    InstanceInfo,
    InstanceRole,
    InstanceStatus,
    MetadataStore,
)
from aegaeon.runtime.run_batch import BatchRunner
from aegaeon.scheduling import (
    ColocatedScheduler,
    DecodeScheduler,
    PrefillScheduler,
)
from aegaeon.types import Request

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class RuntimeState:
    device: str | None = None
    running: bool = True
    requested_models: set[str] = field(default_factory=set)


class InstanceRuntime:
    
    def __init__(
        self,
        cfg: RuntimeConfig,
        mem_cfg: MemoryConfig,
        backend_engine: Any | None = None,
        model_cache: ModelCache | None = None,
        weight_loader: HFWeightLoader | None = None,
    ) -> None:
        self.cfg = cfg
        self.state = RuntimeState()

        self.model_cache = model_cache or ModelCache(mem_cfg.model_cache_budget_bytes)
        self.weight_loader = weight_loader or HFWeightLoader()
        self.backend = backend_engine

        runner = BatchRunner(engine=self.backend)
        if cfg.mode == "prefill":
            self.scheduler = PrefillScheduler(runner, max_batch_size=cfg.max_batch_size)
        elif cfg.mode == "decode":
            self.scheduler = DecodeScheduler(runner, max_batch_size=cfg.max_batch_size)
        elif cfg.mode == "colocated":
            self.scheduler = ColocatedScheduler(runner, max_batch_size=cfg.max_batch_size)
        else:
            raise ValueError(f"unsupported mode: {cfg.mode}")

    def submit(self, req: Request) -> None:
        self.scheduler.submit(req)

    def step(self) -> Any | None:
        return self.scheduler.step()

    def ensure_model_ready(self, model: str) -> None:
        if model in self.state.requested_models:
            return

        def _loader() -> tuple[dict[str, Any], list[int]]:
            state_dict = self.weight_loader.load_state_dict(model)
            chunk_sizes = [int(t.numel()) * int(t.element_size()) for t in state_dict.values()]
            return state_dict, chunk_sizes

        _, loaded = self.model_cache.ensure_model(model, _loader)
        if loaded:
            LOGGER.info("Model %s loaded into shared cache", model)
        else:
            LOGGER.info("Model %s reused from shared cache", model)
        self.model_cache.acquire(model)
        self.state.requested_models.add(model)

    def cold_start_model(self, model: str) -> None:
        LOGGER.info("Fetching %s from huggingface or local storage", model)
        self.weight_loader.fetch_to_model_cache(model, self.model_cache)

    def requested_models(self) -> set[str]:
        return set(self.state.requested_models)

    def stats(self) -> dict[str, dict[str, int]]:
        return {
            "model_cache": self.model_cache.stats(),
            "scheduler": self.scheduler.stats(),
        }

    def publish_metadata(
        self,
        store: MetadataStore,
        endpoint: str,
        models: list[str],
    ) -> None:
        scheduler_stats = self.scheduler.stats()
        queue_depth = scheduler_stats.get("queue_depth", scheduler_stats.get("active_sequences", 0))
        if self.cfg.mode == "prefill":
            role = InstanceRole.PREFILL
        elif self.cfg.mode == "decode":
            role = InstanceRole.DECODE
        else:
            role = InstanceRole.COLOCATED

        store.register_instance(InstanceInfo(instance_id=self.cfg.instance_id, role=role, endpoint=endpoint))
        store.update_instance_status(
            self.cfg.instance_id,
            InstanceStatus(current_models=set(models), queue_depth=queue_depth),
        )

    def close(self) -> None:
        if self.backend is not None and hasattr(self.backend, "shutdown"):
            try:
                self.backend.shutdown()
            except Exception:
                LOGGER.exception("Failed to shutdown backend for instance %s", self.cfg.instance_id)
