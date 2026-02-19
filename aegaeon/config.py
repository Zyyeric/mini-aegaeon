from dataclasses import dataclass


@dataclass(slots=True)
class ProxyConfig:
    shared_memory_name: str = "aegaeon-metadata"
    shared_memory_size: int = 1024 * 1024
    metadata_backend: str = "shared_memory"  # shared_memory | redis
    metadata_lock_path: str | None = None
    deployment_mode: str = "disaggregated"  # disaggregated | colocation


@dataclass(slots=True)
class RuntimeConfig:
    instance_id: str
    mode: str  # prefill | decode | colocated
    max_batch_size: int = 16


@dataclass(slots=True)
class MemoryConfig:
    model_cache_budget_bytes: int = 64 * 1024 * 1024 * 1024
