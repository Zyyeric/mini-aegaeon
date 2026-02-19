from __future__ import annotations

import json
from time import time_ns
from typing import Any

from .metadata_store import (
    InstanceInfo,
    InstanceStatus,
    MetadataStore,
    RequestAssignment,
    RequestPhase,
    RoutingSnapshot,
)


class RedisMetadataStore(MetadataStore):
    """Redis backend for metadata synchronization."""

    def __init__(self, redis_client: Any | None = None, redis_url: str = "redis://localhost:6379/0") -> None:
        if redis_client is None:
            try:
                import redis  # type: ignore
            except Exception as exc:  # pragma: no cover
                raise RuntimeError("redis package is required for RedisMetadataStore") from exc
            redis_client = redis.Redis.from_url(redis_url, decode_responses=True)
        self.client = redis_client

    @staticmethod
    def _inst_key(instance_id: str) -> str:
        return f"inst:{instance_id}"

    @staticmethod
    def _info_key(instance_id: str) -> str:
        return f"inst_info:{instance_id}"

    @staticmethod
    def _req_key(request_id: str) -> str:
        return f"req:{request_id}"

    def register_instance(self, info: InstanceInfo) -> None:
        default_status = InstanceStatus(current_models=set(), queue_depth=0)
        pipe = self.client.pipeline()
        pipe.sadd("inst_index", info.instance_id)
        pipe.set(self._info_key(info.instance_id), json.dumps(info.to_dict(), separators=(",", ":")))
        pipe.set(self._inst_key(info.instance_id), json.dumps(default_status.to_dict(), separators=(",", ":")))
        pipe.execute()

    def update_instance_status(self, instance_id: str, status: InstanceStatus) -> None:
        payload = json.dumps(status.to_dict(), separators=(",", ":"))
        pipe = self.client.pipeline()
        pipe.sadd("inst_index", instance_id)
        pipe.set(self._inst_key(instance_id), payload)
        pipe.execute()

    def sync_for_routing(self) -> RoutingSnapshot:
        instance_ids = sorted(self.client.smembers("inst_index"))
        if not instance_ids:
            return RoutingSnapshot(instances={}, infos={}, ts_ns=time_ns())

        statuses = self.client.mget([self._inst_key(iid) for iid in instance_ids])
        infos_raw = self.client.mget([self._info_key(iid) for iid in instance_ids])

        instances: dict[str, InstanceStatus] = {}
        infos: dict[str, InstanceInfo] = {}
        for instance_id, status_raw, info_raw in zip(instance_ids, statuses, infos_raw, strict=False):
            if status_raw is not None:
                instances[instance_id] = InstanceStatus.from_dict(json.loads(status_raw))
            if info_raw is not None:
                infos[instance_id] = InstanceInfo.from_dict(json.loads(info_raw))

        return RoutingSnapshot(instances=instances, infos=infos, ts_ns=time_ns())

    def record_assignment(
        self,
        request_id: str,
        model_id: str,
        phase: RequestPhase,
        instance_id: str,
        endpoint: str,
        queued: bool,
    ) -> None:
        assignment = RequestAssignment(
            request_id=request_id,
            model_id=model_id,
            phase=phase,
            instance_id=instance_id,
            endpoint=endpoint,
            queued=queued,
            ts_ns=time_ns(),
        )
        self.client.set(self._req_key(request_id), json.dumps(assignment.to_dict(), separators=(",", ":")))

    def get_assignment(self, request_id: str) -> RequestAssignment | None:
        raw = self.client.get(self._req_key(request_id))
        if raw is None:
            return None
        return RequestAssignment.from_dict(json.loads(raw))

    def close(self) -> None:
        return

    def unlink(self) -> None:
        instance_ids = list(self.client.smembers("inst_index"))
        keys: list[str] = ["inst_index"]
        keys.extend(self._inst_key(iid) for iid in instance_ids)
        keys.extend(self._info_key(iid) for iid in instance_ids)
        for req_key in self.client.scan_iter("req:*"):
            keys.append(req_key)
        if keys:
            self.client.delete(*keys)
