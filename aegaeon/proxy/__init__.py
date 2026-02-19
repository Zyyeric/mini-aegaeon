"""Proxy layer for routing requests via shared metadata."""

from .metadata_store import (
    InstanceInfo,
    InstanceRole,
    InstanceStatus,
    MetadataStore,
    RequestAssignment,
    RequestPhase,
    RoutingSnapshot,
    SharedMetadataStore,
)
from .posix_shm_store import PosixShmMetadataStore
from .redis_store import RedisMetadataStore
from .router import RequestEnvelope, RouteDecision
from .proxy import Proxy

__all__ = [
    "InstanceInfo",
    "InstanceRole",
    "InstanceStatus",
    "MetadataStore",
    "RequestAssignment",
    "RequestPhase",
    "RoutingSnapshot",
    "SharedMetadataStore",
    "PosixShmMetadataStore",
    "RedisMetadataStore",
    "RequestEnvelope",
    "RouteDecision",
    "Proxy",
]
