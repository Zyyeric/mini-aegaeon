from __future__ import annotations

from dataclasses import dataclass
from threading import Lock

from aegaeon.config import ProxyConfig

from .metadata_store import (
    InstanceInfo,
    InstanceRole,
    InstanceStatus,
    MetadataStore,
    RequestPhase,
    RoutingSnapshot,
    SharedMetadataStore,
)
from .router import RequestEnvelope, RouteDecision


@dataclass(slots=True)
class Proxy:
    cfg: ProxyConfig
    store: MetadataStore
    _pending_by_instance: dict[str, list[str]]
    _pending_lock: Lock

    @classmethod
    def _build(cls, cfg: ProxyConfig, create: bool) -> "Proxy":
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
        best_model: tuple[str, InstanceStatus, InstanceInfo] | None = None
        best_fallback: tuple[str, InstanceStatus, InstanceInfo] | None = None

        for instance_id, status in snapshot.instances.items():
            info = snapshot.infos.get(instance_id)
            if info is None:
                continue

            if self.cfg.deployment_mode == "disaggregated" and info.role != InstanceRole.PREFILL:
                continue
            if self.cfg.deployment_mode not in {"disaggregated", "colocation"}:
                raise ValueError(f"unsupported deployment mode: {self.cfg.deployment_mode}")

            current = (instance_id, status, info)
            if best_fallback is None or (status.queue_depth, instance_id) < (
                best_fallback[1].queue_depth,
                best_fallback[0],
            ):
                best_fallback = current

            if model_id in status.current_models and (
                best_model is None
                or (status.queue_depth, instance_id) < (best_model[1].queue_depth, best_model[0])
            ):
                best_model = current

        selected = best_model if best_model is not None else best_fallback
        if selected is None:
            raise KeyError("no routable instance available")

        queued = best_model is None
        instance_id, status, info = selected
        with self._pending_lock:
            q = self._pending_by_instance.setdefault(instance_id, [])
            q.append(request_id)
        self.update_instance_status(
            instance_id,
            InstanceStatus(
                current_models=status.current_models,
                queue_depth=status.queue_depth + 1,
            ),
        )
        return RouteDecision(
            request_id=request_id,
            instance_id=instance_id,
            endpoint=info.endpoint,
            phase=phase,
            queued=queued,
        )

    def publish_instance_metadata(
        self,
        instance_id: str,
        role: InstanceRole,
        endpoint: str,
        models: set[str],
        queue_depth: int,
    ) -> None:
        self.store.register_instance(InstanceInfo(instance_id=instance_id, role=role, endpoint=endpoint))
        self.store.update_instance_status(
            instance_id,
            InstanceStatus(current_models=models, queue_depth=queue_depth),
        )

    def shutdown(self, unlink: bool = False) -> None:
        self.store.close()
        if unlink:
            self.store.unlink()
