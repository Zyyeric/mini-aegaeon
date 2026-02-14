from __future__ import annotations

from dataclasses import dataclass
from threading import Lock

from aegaeon.config import ProxyConfig

from .metadata_store import (
    InstanceInfo,
    InstancePhase,
    InstanceRole,
    InstanceStatus,
    MetadataStore,
    RequestPhase,
    RoutingSnapshot,
    SharedMetadataStore,
)
from .router import RequestEnvelope, RouteDecision


@dataclass(slots=True)
class ProxyLayer:
    cfg: ProxyConfig
    store: MetadataStore
    _pending_by_instance: dict[str, list[str]]
    _pending_lock: Lock

    @classmethod
    def create(cls, cfg: ProxyConfig) -> "ProxyLayer":
        return cls._build(cfg, create=True)

    @classmethod
    def attach(cls, cfg: ProxyConfig) -> "ProxyLayer":
        return cls._build(cfg, create=False)

    @classmethod
    def _build(cls, cfg: ProxyConfig, create: bool) -> "ProxyLayer":
        store = SharedMetadataStore(
            name=cfg.shared_memory_name,
            size=cfg.shared_memory_size,
            create=create,
            backend=cfg.metadata_backend,
            lock_path=cfg.metadata_lock_path,
        )
        return cls(cfg=cfg, store=store, _pending_by_instance={}, _pending_lock=Lock())

    def register_instance(self, info: InstanceInfo) -> None:
        self.store.register_instance(info)

    def update_instance_status(self, instance_id: str, status: InstanceStatus) -> None:
        self.store.update_instance_status(instance_id, status)

    def sync_instance_metadata(self) -> RoutingSnapshot:
        return self.store.sync_for_routing()

    def route(self, req: RequestEnvelope) -> RouteDecision:
        snapshot = self.sync_instance_metadata()
        decision = self._choose_instance(
            request_id=req.request_id,
            snapshot=snapshot,
            model_id=req.model,
            phase=req.phase,
        )
        self.store.record_assignment(
            request_id=req.request_id,
            model_id=req.model,
            phase=req.phase,
            instance_id=decision.instance_id,
            endpoint=decision.endpoint,
            queued=decision.queued,
        )
        return decision

    def _choose_instance(
        self,
        request_id: str,
        snapshot: RoutingSnapshot,
        model_id: str,
        phase: RequestPhase,
    ) -> RouteDecision:
        prefill_candidates: list[tuple[str, InstanceStatus, InstanceInfo]] = []
        for instance_id, status in snapshot.instances.items():
            info = snapshot.infos.get(instance_id)
            if info is None or info.role != InstanceRole.PREFILL:
                continue
            if model_id not in status.current_models:
                continue
            if phase == RequestPhase.PREFILL and status.phase not in {
                InstancePhase.IDLE,
                InstancePhase.PREFILLING,
            }:
                continue
            prefill_candidates.append((instance_id, status, info))

        if prefill_candidates:
            instance_id, _, info = min(prefill_candidates, key=lambda x: (x[1].queue_depth, x[0]))
            return RouteDecision(
                request_id=request_id,
                instance_id=instance_id,
                endpoint=info.endpoint,
                phase=phase,
                queued=False,
            )

        fallback_candidates: list[tuple[str, InstanceStatus, InstanceInfo]] = []
        for instance_id, status in snapshot.instances.items():
            info = snapshot.infos.get(instance_id)
            if info is None or info.role != InstanceRole.PREFILL:
                continue
            fallback_candidates.append((instance_id, status, info))

        if not fallback_candidates:
            raise KeyError("no prefill instance available")

        instance_id, status, info = min(fallback_candidates, key=lambda x: (x[1].queue_depth, x[0]))
        with self._pending_lock:
            q = self._pending_by_instance.setdefault(instance_id, [])
            q.append(request_id)
        self.update_instance_status(
            instance_id,
            InstanceStatus(
                current_models=status.current_models,
                phase=status.phase,
                queue_depth=status.queue_depth + 1,
            ),
        )
        return RouteDecision(
            request_id=request_id,
            instance_id=instance_id,
            endpoint=info.endpoint,
            phase=phase,
            queued=True,
        )

    def publish_instance_metadata(
        self,
        instance_id: str,
        role: InstanceRole,
        endpoint: str,
        models: set[str],
        phase: InstancePhase,
        queue_depth: int,
    ) -> None:
        self.store.register_instance(InstanceInfo(instance_id=instance_id, role=role, endpoint=endpoint))
        self.store.update_instance_status(
            instance_id,
            InstanceStatus(current_models=models, phase=phase, queue_depth=queue_depth),
        )

    def shutdown(self, unlink: bool = False) -> None:
        self.store.close()
        if unlink:
            self.store.unlink()
