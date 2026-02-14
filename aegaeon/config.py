from dataclasses import dataclass


@dataclass(slots=True)
class ProxyConfig:
    shared_memory_name: str = "aegaeon-metadata"
    shared_memory_size: int = 1024 * 1024
    metadata_backend: str = "shared_memory"  # shared_memory | redis
    metadata_lock_path: str | None = None


@dataclass(slots=True)
class RuntimeConfig:
    instance_id: str
    mode: str  # prefill | decode
    max_batch_size: int = 16


@dataclass(slots=True)
class MemoryConfig:
    vram_budget_bytes: int = 16 * 1024 * 1024 * 1024
    model_cache_budget_bytes: int = 64 * 1024 * 1024 * 1024
    kv_slab_bytes: int = 4 * 1024 * 1024 * 1024
    kv_chunk_bytes: int = 1 * 1024 * 1024
