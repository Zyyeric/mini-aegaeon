from __future__ import annotations

import argparse

from aegaeon.config import MemoryConfig, ProxyConfig, RuntimeConfig
from aegaeon.memory import HFWeightLoader, ModelCache, ModelCacheManager
from aegaeon.proxy import InstanceRole
from aegaeon.proxy.proxy import Proxy
from aegaeon.runtime import InstanceRuntime
from aegaeon.runtime.topology import detect_accelerator_slots
from aegaeon.server.http_server import run_proxy_server


def build_local_instances(
    proxy: Proxy,
    deployment_mode: str,
    prefill_count: int,
    decode_count: int,
    colocated_count: int,
    model_cache_budget_bytes: int = 8 * 1024 * 1024 * 1024,
    backend: str = "none",
    backend_model: str | None = None,
    backend_memory_ratio: float = 0.5,
    backend_max_live_workers: int = 1,
    backend_model_switching: bool = False,
) -> dict[str, InstanceRuntime]:
    slots = detect_accelerator_slots()
    if deployment_mode == "disaggregated":
        if colocated_count != 0:
            raise ValueError("disaggregated mode requires colocated_count=0")
        if prefill_count <= 0:
            raise ValueError("disaggregated mode requires prefill_count > 0")
        total = prefill_count + decode_count
    elif deployment_mode == "colocation":
        if prefill_count != 0 or decode_count != 0:
            raise ValueError("colocation mode requires prefill_count=0 and decode_count=0")
        if colocated_count <= 0:
            raise ValueError("colocation mode requires colocated_count > 0")
        total = colocated_count
    else:
        raise ValueError(f"unsupported deployment mode: {deployment_mode}")

    if total <= 0:
        raise ValueError("at least one instance must be configured")
    if all(slot.kind == "cpu" for slot in slots):
        raise RuntimeError("no GPU/MIG resources detected")

    runtimes: dict[str, InstanceRuntime] = {}
    shared_model_cache = ModelCache(model_cache_budget_bytes)
    shared_weight_loader = HFWeightLoader()
    backend_engine = None
    if backend == "mini_sgl":
        from aegaeon.backend import MiniSGLMultiBackend
        import torch

        # Keep cached weights in the same dtype used by mini-sgl workers.
        shared_weight_loader = HFWeightLoader(dtype=torch.float16)
        shared_weight_manager = ModelCacheManager(cache=shared_model_cache, device="cpu")
        backend_engine = MiniSGLMultiBackend(
            weight_manager=shared_weight_manager,
            memory_ratio=backend_memory_ratio,
            max_live_workers=backend_max_live_workers,
            model_switching=backend_model_switching,
        )

    if deployment_mode == "disaggregated":
        launch_plan: list[tuple[str, int]] = [
            *(("prefill", i) for i in range(prefill_count)),
            *(("decode", i) for i in range(decode_count)),
        ]
    else:
        launch_plan = [*(("colocated", i) for i in range(colocated_count))]

    for mode, idx in launch_plan:
        instance_id = f"{mode}-{idx}"
        rt = InstanceRuntime(
            RuntimeConfig(instance_id=instance_id, mode=mode),
            MemoryConfig(model_cache_budget_bytes=model_cache_budget_bytes),
            backend_engine=backend_engine,
            model_cache=shared_model_cache,
            weight_loader=shared_weight_loader,
        )

        if mode == "prefill":
            role = InstanceRole.PREFILL
        elif mode == "decode":
            role = InstanceRole.DECODE
        else:
            role = InstanceRole.COLOCATED
        proxy.publish_instance_metadata(
            instance_id=instance_id,
            role=role,
            endpoint=f"local://{instance_id}",
            models=set(),
            queue_depth=0,
        )
        runtimes[instance_id] = rt

    return runtimes


def run_proxy_endpoint(
    host: str,
    port: int,
    deployment_mode: str,
    prefill_count: int,
    decode_count: int,
    colocated_count: int,
    metadata_backend: str,
    model_cache_budget_bytes: int = 8 * 1024 * 1024 * 1024,
) -> None:
    proxy = Proxy._build(
        ProxyConfig(metadata_backend=metadata_backend, deployment_mode=deployment_mode),
        create=True,
    )
    runtimes = build_local_instances(
        proxy,
        deployment_mode,
        prefill_count,
        decode_count,
        colocated_count,
        model_cache_budget_bytes=model_cache_budget_bytes,
    )
    print(f"proxy listening on http://{host}:{port} with local_instances={len(runtimes)}")
    run_proxy_server(proxy=proxy, host=host, port=port, local_instances=runtimes)


def launch_server() -> None:
    parser = argparse.ArgumentParser(description="Launch Aegaeon online server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--metadata-mode",
        choices=["shared_memory", "redis"],
        default="shared_memory",
        help="Metadata backend mode",
    )
    parser.add_argument("--prefill-count", type=int, default=1)
    parser.add_argument("--decode-count", type=int, default=0)
    parser.add_argument("--colocated-count", type=int, default=0)
    parser.add_argument("--model-cache-budget-gb", type=float, default=8.0)
    parser.add_argument(
        "--deployment-mode",
        choices=["disaggregated", "colocation"],
        default="disaggregated",
        help="Serving topology mode: disaggregated and colocation are mutually exclusive.",
    )

    args = parser.parse_args()

    run_proxy_endpoint(
        host=args.host,
        port=args.port,
        deployment_mode=args.deployment_mode,
        prefill_count=args.prefill_count,
        decode_count=args.decode_count,
        colocated_count=args.colocated_count,
        metadata_backend=args.metadata_mode,
        model_cache_budget_bytes=int(args.model_cache_budget_gb * 1024 * 1024 * 1024),
    )


if __name__ == "__main__":
    launch_server()
