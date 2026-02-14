from __future__ import annotations

import argparse

from aegaeon.config import MemoryConfig, ProxyConfig, RuntimeConfig
from aegaeon.proxy import InstancePhase, InstanceRole, RequestEnvelope, RequestPhase
from aegaeon.proxy.http_server import run_proxy_server
from aegaeon.proxy.service import ProxyLayer
from aegaeon.runtime import InstanceRuntime
from aegaeon.scheduling import ScheduledRequest


def run_proxy() -> None:
    proxy = ProxyLayer.create(ProxyConfig())
    proxy.publish_instance_metadata(
        instance_id="prefill-1",
        role=InstanceRole.PREFILL,
        endpoint="http://127.0.0.1:9000",
        models={"llama3-8b"},
        phase=InstancePhase.PREFILLING,
        queue_depth=2,
    )
    proxy.publish_instance_metadata(
        instance_id="decode-1",
        role=InstanceRole.DECODE,
        endpoint="http://127.0.0.1:9001",
        models={"llama3-8b"},
        phase=InstancePhase.DECODING,
        queue_depth=4,
    )

    decision = proxy.route(
        RequestEnvelope(
            request_id="req-1",
            model="llama3-8b",
            payload={"prompt": "hello"},
            phase=RequestPhase.PREFILL,
        )
    )
    print(f"route decision: {decision}")
    print(f"assignment: {proxy.store.get_assignment('req-1')}")
    proxy.shutdown(unlink=True)


def run_proxy_endpoint(host: str, port: int) -> None:
    proxy = ProxyLayer.create(ProxyConfig())
    print(f"proxy listening on http://{host}:{port}")
    run_proxy_server(proxy=proxy, host=host, port=port)


def run_instance(mode: str) -> None:
    rt = InstanceRuntime(
        RuntimeConfig(instance_id=f"{mode}-1", mode=mode),
        MemoryConfig(
            vram_budget_bytes=4 * 1024 * 1024 * 1024,
            model_cache_budget_bytes=8 * 1024 * 1024 * 1024,
            kv_slab_bytes=256 * 1024 * 1024,
            kv_chunk_bytes=1 * 1024 * 1024,
        ),
    )

    rt.load_model("llama3-8b", vram_bytes=2 * 1024 * 1024 * 1024, cpu_cache_bytes=512 * 1024 * 1024)
    rt.submit(
        ScheduledRequest(
            request_id="req-1",
            model="llama3-8b",
            input_ids=[1, 2, 3],
            sampling_params={"temperature": 0.7},
        )
    )
    print(f"batch result: {rt.step()}")
    print(f"stats: {rt.stats()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aegaeon rough skeleton")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("proxy")
    p_proxy_server = sub.add_parser("proxy-server")
    p_proxy_server.add_argument("--host", default="0.0.0.0")
    p_proxy_server.add_argument("--port", type=int, default=8080)

    p_instance = sub.add_parser("instance")
    p_instance.add_argument("--mode", choices=["prefill", "decode"], required=True)

    args = parser.parse_args()

    if args.command == "proxy":
        run_proxy()
    elif args.command == "proxy-server":
        run_proxy_endpoint(args.host, args.port)
    elif args.command == "instance":
        run_instance(args.mode)


if __name__ == "__main__":
    main()
