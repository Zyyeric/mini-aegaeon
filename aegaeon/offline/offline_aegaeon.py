from __future__ import annotations

from uuid import uuid4

from aegaeon.config import ProxyConfig
from aegaeon.server.launch import build_local_instances
from aegaeon.proxy.metadata_store import InstanceStatus, RequestPhase
from aegaeon.proxy.proxy import Proxy
from aegaeon.proxy.router import RequestEnvelope
from aegaeon.runtime import InstanceRuntime
from aegaeon.types import Request


class OfflineAegaeon:

    def __init__(
        self,
        deployment_mode: str = "colocation",
        prefill_count: int = 0,
        decode_count: int = 0,
        colocated_count: int = 1,
        metadata_backend: str = "shared_memory",
        proxy_cfg: ProxyConfig | None = None,
    ) -> None:
        self.proxy = Proxy._build(
            proxy_cfg
            or ProxyConfig(
                metadata_backend=metadata_backend,
                deployment_mode=deployment_mode,
            ),
            create=True,
        )
        self.instances: dict[str, InstanceRuntime] = build_local_instances(
            self.proxy,
            deployment_mode=deployment_mode,
            prefill_count=prefill_count,
            decode_count=decode_count,
            colocated_count=colocated_count,
        )
        self.counter = 0

    def request(self, payload: dict) -> dict:
        model = payload.get("model")
        if not isinstance(model, str) or not model:
            raise ValueError("`model` is required")

        request_id = str(payload.get("request_id") or f"req-{uuid4().hex}")
        req = RequestEnvelope(
            request_id=request_id,
            model=model,
            payload=payload,
            phase=RequestPhase.PREFILL,
        )

        decision = self.proxy.route(req)
        runtime = self.instances.get(decision.instance_id)
        if runtime is None:
            raise KeyError(f"instance not found: {decision.instance_id}")

        current_status = self.proxy.sync_instance_metadata().instances.get(decision.instance_id)
        runtime.ensure_model_ready(model)

        prompt = payload.get("prompt")
        if prompt is None and isinstance(payload.get("messages"), list):
            prompt = " ".join(str(m.get("content", "")) for m in payload["messages"] if isinstance(m, dict))
        prompt_text = str(prompt or "")
        input_ids = [ord(c) % 256 for c in prompt_text][:128] or [0]
        uid = self.counter
        self.counter += 1

        runtime.submit(
            Request(
                request_id=request_id,
                model=model,
                input_ids=input_ids,
                uid=uid,
                sampling_params=payload.get("sampling_params", {}),
            )
        )
        result = runtime.step()

        queue_after = max((current_status.queue_depth if current_status is not None else 1) - 1, 0)
        self.proxy.update_instance_status(
            decision.instance_id,
            InstanceStatus(
                current_models=runtime.requested_models(),
                queue_depth=queue_after,
            ),
        )

        output = None if result is None else result.outputs.get(request_id)
        return {
            "request_id": request_id,
            "model": model,
            "instance_id": decision.instance_id,
            "queued": decision.queued,
            "output": output,
        }

    def close(self, unlink: bool = True) -> None:
        self.proxy.shutdown(unlink=unlink)
