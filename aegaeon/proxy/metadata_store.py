from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any


class InstanceRole(str, Enum):
    PREFILL = "PREFILL"
    DECODE = "DECODE"


class InstancePhase(str, Enum):
    IDLE = "IDLE"
    PREFILLING = "PREFILLING"
    DECODING = "DECODING"


class RequestPhase(str, Enum):
    PREFILL = "PREFILL"
    DECODE = "DECODE"


@dataclass(slots=True)
class InstanceInfo:
    instance_id: str
    role: InstanceRole
    endpoint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "role": self.role.value,
            "endpoint": self.endpoint,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "InstanceInfo":
        return cls(
            instance_id=str(payload["instance_id"]),
            role=InstanceRole(payload["role"]),
            endpoint=str(payload["endpoint"]),
        )


@dataclass(slots=True)
class InstanceStatus:
    current_models: set[str]
    phase: InstancePhase
    queue_depth: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_models": sorted(self.current_models),
            "phase": self.phase.value,
            "queue_depth": int(self.queue_depth),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "InstanceStatus":
        models = payload.get("current_models", [])
        return cls(
            current_models=set(str(m) for m in models),
            phase=InstancePhase(payload["phase"]),
            queue_depth=int(payload["queue_depth"]),
        )


@dataclass(slots=True)
class RequestAssignment:
    request_id: str
    model_id: str
    phase: RequestPhase
    instance_id: str
    endpoint: str
    queued: bool
    ts_ns: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "model_id": self.model_id,
            "phase": self.phase.value,
            "instance_id": self.instance_id,
            "endpoint": self.endpoint,
            "queued": self.queued,
            "ts_ns": int(self.ts_ns),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RequestAssignment":
        return cls(
            request_id=str(payload["request_id"]),
            model_id=str(payload["model_id"]),
            phase=RequestPhase(payload["phase"]),
            instance_id=str(payload["instance_id"]),
            endpoint=str(payload.get("endpoint", "")),
            queued=bool(payload.get("queued", False)),
            ts_ns=int(payload["ts_ns"]),
        )


@dataclass(slots=True)
class RoutingSnapshot:
    instances: dict[str, InstanceStatus]
    infos: dict[str, InstanceInfo]
    ts_ns: int


class MetadataStore(ABC):
    @abstractmethod
    def register_instance(self, info: InstanceInfo) -> None:
        pass

    @abstractmethod
    def update_instance_status(self, instance_id: str, status: InstanceStatus) -> None:
        pass

    @abstractmethod
    def sync_for_routing(self) -> RoutingSnapshot:
        pass

    @abstractmethod
    def record_assignment(
        self,
        request_id: str,
        model_id: str,
        phase: RequestPhase,
        instance_id: str,
        endpoint: str,
        queued: bool,
    ) -> None:
        pass

    @abstractmethod
    def get_assignment(self, request_id: str) -> RequestAssignment | None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass

    @abstractmethod
    def unlink(self) -> None:
        pass


class SharedMetadataStore(MetadataStore):

    def __init__(
        self,
        name: str,
        size: int,
        create: bool = True,
        backend: str = "shared_memory",
        lock_path: str | None = None,
        redis_client: Any | None = None,
        redis_url: str = "redis://localhost:6379/0",
    ) -> None:
        if backend == "shared_memory":
            from .posix_shm_store import PosixShmMetadataStore

            self._impl: MetadataStore = PosixShmMetadataStore(
                name=name,
                size=size,
                create=create,
                lock_path=lock_path,
            )
        elif backend == "redis":
            from .redis_store import RedisMetadataStore

            self._impl = RedisMetadataStore(redis_client=redis_client, redis_url=redis_url)
        else:
            raise ValueError("backend must be 'shared_memory' or 'redis'")

    def register_instance(self, info: InstanceInfo) -> None:
        self._impl.register_instance(info)

    def update_instance_status(self, instance_id: str, status: InstanceStatus) -> None:
        self._impl.update_instance_status(instance_id, status)

    def sync_for_routing(self) -> RoutingSnapshot:
        return self._impl.sync_for_routing()

    def record_assignment(
        self,
        request_id: str,
        model_id: str,
        phase: RequestPhase,
        instance_id: str,
        endpoint: str,
        queued: bool,
    ) -> None:
        self._impl.record_assignment(
            request_id=request_id,
            model_id=model_id,
            phase=phase,
            instance_id=instance_id,
            endpoint=endpoint,
            queued=queued,
        )

    def get_assignment(self, request_id: str) -> RequestAssignment | None:
        return self._impl.get_assignment(request_id)

    def close(self) -> None:
        self._impl.close()

    def unlink(self) -> None:
        self._impl.unlink()
