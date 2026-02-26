from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _wrap_method(obj: Any, method_name: str, counter: dict[str, int]) -> None:
    original = getattr(obj, method_name)

    def _wrapped(*args: Any, **kwargs: Any) -> Any:
        counter[method_name] = counter.get(method_name, 0) + 1
        return original(*args, **kwargs)

    setattr(obj, method_name, _wrapped)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline mini-sgl contract smoke test with dummy weights."
    )
    parser.add_argument(
        "--models",
        default="Qwen/Qwen3-0.6B,Qwen/Qwen3-0.6B",
        help="Comma-separated model names used in requests.",
    )
    parser.add_argument("--num-requests", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--colocated-count", type=int, default=1)
    parser.add_argument("--model-cache-budget-gb", type=float, default=8.0)
    parser.add_argument("--backend-memory-ratio", type=float, default=0.5)
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        raise SystemExit("Provide at least one model in --models.")
    unique_models = list(dict.fromkeys(models))
    if len(unique_models) < 2:
        raise SystemExit("Provide at least two different models in --models for switching test.")
    switch_models = unique_models[:2]
    budget_model = switch_models[0]
    try:
        from minisgl.models import estimate_hf_weight_nbytes_from_safetensors

        budget_model = max(
            switch_models,
            key=lambda m: int(estimate_hf_weight_nbytes_from_safetensors(m)),
        )
    except Exception:
        pass

    from aegaeon.offline import OfflineAegaeon

    offline = OfflineAegaeon(
        deployment_mode="colocation",
        prefill_count=0,
        decode_count=0,
        colocated_count=args.colocated_count,
        metadata_backend="shared_memory",
        model_cache_budget_bytes=int(args.model_cache_budget_gb * 1024 * 1024 * 1024),
        backend="mini_sgl",
        backend_model=budget_model,
        backend_memory_ratio=args.backend_memory_ratio,
        backend_model_switching=True,
        backend_use_dummy_weight=True,
    )
    try:
        lifecycle_counts: dict[str, dict[str, int]] = {}
        for instance_id, runtime in offline.instances.items():
            backend = runtime.backend
            if backend is None:
                continue
            counters = {"prepare_for_batch": 0, "load_weights_for_batch": 0, "after_batch": 0}
            lifecycle_counts[instance_id] = counters
            _wrap_method(backend, "prepare_for_batch", counters)
            _wrap_method(backend, "load_weights_for_batch", counters)
            _wrap_method(backend, "after_batch", counters)

        request_ids: list[str] = []
        request_model_seq: list[str] = []
        for i in range(args.num_requests):
            model = switch_models[i % 2]
            request_model_seq.append(model)
            request_ids.append(
                offline.enqueue(
                    {
                        "request_id": f"contract-{i}",
                        "model": model,
                        "prompt": f"Contract test prompt {i}",
                        "max_new_tokens": args.max_new_tokens,
                    }
                )
            )

        offline.run_until_complete(set(request_ids))
        switch_count = sum(
            1 for prev, cur in zip(request_model_seq, request_model_seq[1:], strict=False) if prev != cur
        )

        failed = False
        for instance_id, counters in lifecycle_counts.items():
            prep = counters["prepare_for_batch"]
            load = counters["load_weights_for_batch"]
            done = counters["after_batch"]
            if prep <= 0 or load <= 0 or done <= 0 or prep != load or load != done:
                failed = True
                print(
                    f"[FAIL] {instance_id}: prepare={prep} load={load} after={done}",
                    file=sys.stderr,
                )

        print(
            json.dumps(
                {
                    "instances": lifecycle_counts,
                    "model_sequence": request_model_seq,
                    "switches": switch_count,
                    "ok": not failed,
                },
                indent=2,
            )
        )
        if failed:
            raise SystemExit(1)
    finally:
        offline.close(unlink=True)


if __name__ == "__main__":
    main()
