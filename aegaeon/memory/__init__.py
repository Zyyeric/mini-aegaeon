"""Memory management primitives for Aegaeon."""

from .kv_slab import SlabAllocationError, SlabKVBackend
from .model_cache import ModelCache, ModelCacheEntry
from .vram_manager import VRAMManager

__all__ = [
    "SlabAllocationError",
    "SlabKVBackend",
    "ModelCache",
    "ModelCacheEntry",
    "VRAMManager",
]
