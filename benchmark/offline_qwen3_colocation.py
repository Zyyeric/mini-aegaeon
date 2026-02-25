from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
import random
import sys
import time
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    rank = (len(s) - 1) * p
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return s[lo]
    w = rank - lo
    return s[lo] * (1.0 - w) + s[hi] * w


def _make_prompt(i: int, prompt_chars: int) -> str:
    base = (
        f"Request {i}: summarize how colocation scheduling impacts TTFT/TBT on L4 GPUs. "
        "Return concise bullet points."
    )
    if len(base) >= prompt_chars:
        return base[:prompt_chars]
    return (base + " ") * (prompt_chars // (len(base) + 1)) + base[: prompt_chars % (len(base) + 1)]


def _summary_stats(samples: list[float]) -> dict[str, float | None]:
    return {
        "avg": (sum(samples) / len(samples)) if samples else None,
        "p50": _percentile(samples, 0.50),
        "p90": _percentile(samples, 0.90),
        "p99": _percentile(samples, 0.99),
    }


def _pick_model(i: int, models: list[str], policy: str, group_size: int) -> str:
    if policy == "round_robin":
        return models[i % len(models)]
    if policy == "random":
        return random.choice(models)
    if policy == "grouped":
        return models[(i // max(group_size, 1)) % len(models)]
    raise ValueError(f"unknown model mix policy: {policy}")


def _instance_switching_stats(request_order: list[str], completed: dict[str, dict]) -> dict[str, dict]:
    seq_by_instance: dict[str, list[str]] = defaultdict(list)
    for rid in request_order:
        item = completed[rid]
        seq_by_instance[str(item["instance_id"])].append(str(item["model"]))

    out: dict[str, dict] = {}
    for instance_id, seq in seq_by_instance.items():
        switches = 0
        for prev, cur in zip(seq, seq[1:], strict=False):
            if prev != cur:
                switches += 1
        out[instance_id] = {
            "requests": len(seq),
            "models_seen": dict(Counter(seq)),
            "switches": switches,
            "switch_rate": (switches / (len(seq) - 1)) if len(seq) > 1 else 0.0,
        }
    return out


def _load_trace(trace_json: str, default_max_tokens: int) -> list[dict]:
    data = json.loads(Path(trace_json).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("trace json must be a list of request entries")
    out: list[dict] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        model = item.get("model")
        prompt_ids = item.get("prompt_token_ids")
        if not isinstance(model, str) or not model:
            continue
        if not isinstance(prompt_ids, list) or not prompt_ids:
            continue
        cleaned_ids = [int(x) for x in prompt_ids if isinstance(x, int)]
        if not cleaned_ids:
            continue
        max_new_tokens = item.get("max_new_tokens", default_max_tokens)
        if not isinstance(max_new_tokens, int) or max_new_tokens <= 0:
            max_new_tokens = default_max_tokens
        try:
            arrival_offset_ms = float(item.get("arrival_offset_ms", 0.0))
        except (TypeError, ValueError):
            arrival_offset_ms = 0.0
        out.append(
            {
                "request_id": str(item.get("request_id", f"trace-{i}")),
                "model": model,
                "prompt_token_ids": cleaned_ids,
                "max_new_tokens": max_new_tokens,
                "arrival_offset_ms": max(arrival_offset_ms, 0.0),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline Aegaeon benchmark (colocation)")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--models", default="", help="Comma-separated model list. Overrides --model.")
    parser.add_argument("--model-mix-policy", choices=["round_robin", "grouped", "random"], default="round_robin")
    parser.add_argument("--group-size", type=int, default=8, help="Used only when --model-mix-policy=grouped")
    parser.add_argument("--num-requests", type=int, default=64)
    parser.add_argument("--prompt-chars", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--colocated-count", type=int, default=2)
    parser.add_argument("--model-cache-budget-gb", type=float, default=64.0)
    parser.add_argument("--metadata-backend", choices=["shared_memory", "redis"], default="shared_memory")
    parser.add_argument("--preload-models", action="store_true")
    parser.add_argument("--preload-per-instance", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trace-json", default="", help="Replay deterministic request trace JSON")
    parser.add_argument("--backend", choices=["none", "mini_sgl"], default="mini_sgl")
    parser.add_argument("--backend-model", default="", help="Model path used to initialize backend engine.")
    parser.add_argument("--backend-memory-ratio", type=float, default=0.5)
    parser.add_argument(
        "--backend-max-live-workers",
        type=int,
        default=1,
        help="Maximum number of simultaneously live backend model workers on GPU.",
    )
    parser.add_argument(
        "--backend-model-switching",
        action="store_true",
        help="Enable model-switching memory policy in mini-sgl backend.",
    )
    parser.add_argument("--out-json", default="", help="Optional output JSON path")
    args = parser.parse_args()

    random.seed(args.seed)
    models = [m.strip() for m in args.models.split(",") if m.strip()] if args.models else [args.model]
    if not models:
        raise SystemExit("Provide at least one model via --model or --models.")

    try:
        import torch  # noqa: F401
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency: torch") from exc

    from aegaeon.runtime.topology import detect_accelerator_slots

    slots = detect_accelerator_slots()
    non_cpu_slots = [s for s in slots if s.kind != "cpu"]
    if not non_cpu_slots:
        raise SystemExit("No NVIDIA GPU/MIG detected by `nvidia-smi -L`.")
    if len(non_cpu_slots) < args.colocated_count:
        raise SystemExit(
            f"Requested colocated_count={args.colocated_count}, but only {len(non_cpu_slots)} accelerator slot(s) detected."
        )

    from aegaeon.offline import OfflineAegaeon

    backend_model = args.backend_model or models[0]
    offline = OfflineAegaeon(
        deployment_mode="colocation",
        prefill_count=0,
        decode_count=0,
        colocated_count=args.colocated_count,
        metadata_backend=args.metadata_backend,
        model_cache_budget_bytes=int(args.model_cache_budget_gb * 1024 * 1024 * 1024),
        backend=args.backend,
        backend_model=backend_model,
        backend_memory_ratio=args.backend_memory_ratio,
        backend_max_live_workers=args.backend_max_live_workers,
        backend_model_switching=args.backend_model_switching,
    )
    try:
        warmup_s = 0.0
        if args.preload_models:
            warmup_t0 = time.perf_counter()
            offline.preload_models(models=models, per_instance=args.preload_per_instance)
            warmup_s = time.perf_counter() - warmup_t0

        if args.trace_json:
            trace_reqs = _load_trace(args.trace_json, default_max_tokens=args.max_new_tokens)
            if not trace_reqs:
                raise SystemExit(f"No valid requests loaded from trace: {args.trace_json}")
            models = sorted({str(x["model"]) for x in trace_reqs})
        else:
            trace_reqs = []
            for i in range(args.num_requests):
                model = _pick_model(i=i, models=models, policy=args.model_mix_policy, group_size=args.group_size)
                trace_reqs.append(
                    {
                        "request_id": f"bench-{i}",
                        "model": model,
                        "prompt": _make_prompt(i, args.prompt_chars),
                        "max_new_tokens": args.max_new_tokens,
                        "arrival_offset_ms": float(i),
                    }
                )

        request_ids: list[str] = []
        request_order: list[str] = []
        t0 = time.perf_counter()
        for req in trace_reqs:
            target_t = t0 + (float(req["arrival_offset_ms"]) / 1000.0)
            now_t = time.perf_counter()
            if target_t > now_t:
                time.sleep(target_t - now_t)

            payload = {
                "request_id": req["request_id"],
                "model": req["model"],
                "sampling_params": {"max_new_tokens": int(req["max_new_tokens"])},
            }
            if "prompt_token_ids" in req:
                payload["prompt_token_ids"] = req["prompt_token_ids"]
            else:
                payload["prompt"] = req.get("prompt", "")

            rid = offline.enqueue(payload)
            request_ids.append(rid)
            request_order.append(rid)

        completed = offline.run_until_complete(set(request_ids))
        t1 = time.perf_counter()

        by_model = offline.benchmark_by_model()
        ttft_samples_all: list[float] = []
        tbt_samples_all: list[float] = []
        per_model_report: dict[str, dict] = {}
        for model in models:
            model_summary = by_model.get(model, {})
            ttft_samples = [float(v) for v in model_summary.get("ttft_ms_samples", [])]
            tbt_samples = [float(v) for v in model_summary.get("tbt_ms_samples", [])]
            ttft_samples_all.extend(ttft_samples)
            tbt_samples_all.extend(tbt_samples)
            per_model_report[model] = {
                "requests": int(model_summary.get("requests", 0)),
                "ttft_ms": _summary_stats(ttft_samples),
                "tbt_ms": _summary_stats(tbt_samples),
                "ttft_ms_samples": ttft_samples,
                "tbt_ms_samples": tbt_samples,
            }

        total_tokens = sum(int(item.get("tokens_emitted", 0)) for item in completed.values())
        wall_s = t1 - t0

        report = {
            "models": models,
            "model_mix_policy": args.model_mix_policy,
            "num_requests": len(request_ids),
            "prompt_chars": args.prompt_chars,
            "max_new_tokens": args.max_new_tokens,
            "colocated_count": args.colocated_count,
            "model_cache_budget_gb": args.model_cache_budget_gb,
            "trace_json": args.trace_json if args.trace_json else None,
            "preload_models": args.preload_models,
            "preload_per_instance": args.preload_per_instance,
            "warmup_time_s": warmup_s,
            "wall_time_s": wall_s,
            "req_per_s": (len(request_ids) / wall_s) if wall_s > 0 else None,
            "token_per_s": (total_tokens / wall_s) if wall_s > 0 else None,
            "ttft_ms": _summary_stats(ttft_samples_all),
            "tbt_ms": _summary_stats(tbt_samples_all),
            "per_model": per_model_report,
            "instance_switching": _instance_switching_stats(request_order, completed),
        }

        print(json.dumps(report, indent=2))
        if args.out_json:
            out_path = Path(args.out_json)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    finally:
        offline.close(unlink=True)


if __name__ == "__main__":
    main()
