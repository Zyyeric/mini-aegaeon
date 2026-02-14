from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aegaeon.config import MemoryConfig, RuntimeConfig
from aegaeon.memory import ModelCache, SlabKVBackend, VRAMManager
from aegaeon.proxy.metadata_store import (
    InstanceInfo,
    InstancePhase,
    InstanceRole,
    InstanceStatus,
    MetadataStore,
)
from aegaeon.scheduling import BatchRunner, DecodeScheduler, PrefillScheduler, ScheduledRequest


@dataclass(slots=True)
class RuntimeState:
    running: bool = True


class InstanceRuntime:
    """Per-instance process that owns scheduling and memory managers."""

    def __init__(self, cfg: RuntimeConfig, mem_cfg: MemoryConfig) -> None:
        self.cfg = cfg
        self.state = RuntimeState()

        self.vram = VRAMManager(mem_cfg.vram_budget_bytes)
        self.model_cache = ModelCache(mem_cfg.model_cache_budget_bytes)
        self.kv_backend = SlabKVBackend(mem_cfg.kv_slab_bytes, mem_cfg.kv_chunk_bytes)

        runner = BatchRunner(engine=None)
        if cfg.mode == "prefill":
            self.scheduler = PrefillScheduler(runner, max_batch_size=cfg.max_batch_size)
        elif cfg.mode == "decode":
            self.scheduler = DecodeScheduler(runner, max_batch_size=cfg.max_batch_size)
        else:
            raise ValueError(f"unsupported mode: {cfg.mode}")

    def submit(self, req: ScheduledRequest) -> None:
        self.scheduler.submit(req)

    def step(self):
        return self.scheduler.step()

    def load_model(self, model: str, vram_bytes: int, cpu_cache_bytes: int) -> bool:
        if not self.vram.reserve(model, vram_bytes):
            return False
        self.model_cache.put(model, cpu_cache_bytes)
        return True

    def stats(self) -> dict[str, dict[str, int]]:
        return {
            "vram": self.vram.stats(),
            "model_cache": self.model_cache.stats(),
            "kv": self.kv_backend.stats(),
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
        role = InstanceRole.PREFILL if self.cfg.mode == "prefill" else InstanceRole.DECODE
        phase = InstancePhase.PREFILLING if self.cfg.mode == "prefill" else InstancePhase.DECODING
        store.register_instance(InstanceInfo(instance_id=self.cfg.instance_id, role=role, endpoint=endpoint))
        store.update_instance_status(
            self.cfg.instance_id,
            InstanceStatus(current_models=set(models), phase=phase, queue_depth=queue_depth),
        )
