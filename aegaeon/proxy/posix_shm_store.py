from __future__ import annotations

import contextlib
import fcntl
import json
import os
import threading
from multiprocessing import shared_memory
from pathlib import Path
from time import time_ns
from typing import Any

from .metadata_store import (
    InstanceInfo,
    InstancePhase,
    InstanceStatus,
    MetadataStore,
    RequestAssignment,
    RequestPhase,
    RoutingSnapshot,
)


class PosixShmMetadataStore(MetadataStore):
    """POSIX shared memory backend storing a single synchronized JSON blob."""

    def __init__(
        self,
        name: str,
        size: int,
        create: bool = True,
        lock_path: str | None = None,
    ) -> None:
        self._size = size
        self._thread_lock = threading.Lock()
        self._lock_path = lock_path or os.path.join("/tmp", f"{name}.lock")
        Path(self._lock_path).touch(exist_ok=True)
        self._lock_fd = open(self._lock_path, "r+b")
        self._shm = shared_memory.SharedMemory(name=name, create=create, size=size)

        if create:
            self._write_blob({"instances": {}, "assignments": {}})

    def register_instance(self, info: InstanceInfo) -> None:
        with self._sync_lock():
            blob = self._read_blob()
            instances = blob.setdefault("instances", {})
            current = instances.get(info.instance_id, {})
            current["info"] = info.to_dict()
            current.setdefault(
                "status",
                InstanceStatus(current_models=set(), phase=InstancePhase.IDLE, queue_depth=0).to_dict(),
            )
            instances[info.instance_id] = current
            self._write_blob(blob)

    def update_instance_status(self, instance_id: str, status: InstanceStatus) -> None:
        with self._sync_lock():
            blob = self._read_blob()
            instances = blob.setdefault("instances", {})
            current = instances.get(instance_id, {})
            current["status"] = status.to_dict()
            instances[instance_id] = current
            self._write_blob(blob)

    def sync_for_routing(self) -> RoutingSnapshot:
        with self._sync_lock():
            blob = self._read_blob()
            instances_blob = blob.get("instances", {})
            instances: dict[str, InstanceStatus] = {}
            infos: dict[str, InstanceInfo] = {}
            for instance_id, node in instances_blob.items():
                status_raw = node.get("status")
                info_raw = node.get("info")
                if not isinstance(status_raw, dict):
                    continue
                instances[str(instance_id)] = InstanceStatus.from_dict(status_raw)
                if isinstance(info_raw, dict):
                    infos[str(instance_id)] = InstanceInfo.from_dict(info_raw)
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
        with self._sync_lock():
            blob = self._read_blob()
            assignments = blob.setdefault("assignments", {})
            assignment = RequestAssignment(
                request_id=request_id,
                model_id=model_id,
                phase=phase,
                instance_id=instance_id,
                endpoint=endpoint,
                queued=queued,
                ts_ns=time_ns(),
            )
            assignments[request_id] = assignment.to_dict()
            self._write_blob(blob)

    def get_assignment(self, request_id: str) -> RequestAssignment | None:
        with self._sync_lock():
            blob = self._read_blob()
            raw = blob.get("assignments", {}).get(request_id)
            if raw is None:
                return None
            return RequestAssignment.from_dict(raw)

    def close(self) -> None:
        self._shm.close()
        self._lock_fd.close()

    def unlink(self) -> None:
        self._shm.unlink()

    @contextlib.contextmanager
    def _sync_lock(self):
        with self._thread_lock:
            fcntl.flock(self._lock_fd.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(self._lock_fd.fileno(), fcntl.LOCK_UN)

    def _read_blob(self) -> dict[str, Any]:
        raw = bytes(self._shm.buf)
        end = raw.find(b"\x00")
        if end < 0:
            end = len(raw)
        payload = raw[:end].decode("utf-8") if end > 0 else "{}"
        return json.loads(payload or "{}")

    def _write_blob(self, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        if len(raw) >= self._size:
            raise ValueError(f"metadata size {len(raw)} exceeds shared memory capacity {self._size}")
        self._shm.buf[: len(raw)] = raw
        self._shm.buf[len(raw) :] = b"\x00" * (self._size - len(raw))
