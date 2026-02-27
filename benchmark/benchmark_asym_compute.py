from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from benchmark.benchmark_core import RawResult, process_benchmark_results
from benchmark.trace_generator import TraceGenerator
from minisgl.core import SamplingParams
from minisgl.llm import LLM
from transformers import AutoTokenizer


@dataclass
class RunMetrics:
    model: str
    run_idx: int
    ttft_ms: float
    tbt_mean_ms: float
    tbt_p99_ms: float
    tokens: int
    e2e_s: float


def _load_trace(trace_json: str, default_model: str, default_max_tokens: int) -> list[dict]:
    data = json.loads(Path(trace_json).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("trace json must be a list of request entries")
    out: list[dict] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        prompt_ids = item.get("prompt_token_ids")
        if not isinstance(prompt_ids, list) or not prompt_ids:
            continue
        cleaned_ids: list[int] = []
        for x in prompt_ids:
            if isinstance(x, int):
                cleaned_ids.append(x)
        if not cleaned_ids:
            continue
        offset_ms = item.get("arrival_offset_ms", 0.0)
        try:
            offset_ms_f = float(offset_ms)
        except (TypeError, ValueError):
            offset_ms_f = 0.0
        max_new_tokens = item.get("max_new_tokens", default_max_tokens)
        if not isinstance(max_new_tokens, int) or max_new_tokens <= 0:
            max_new_tokens = default_max_tokens
        out.append(
            {
                "request_id": str(item.get("request_id", f"trace-{i}")),
                "model": str(item.get("model", default_model)),
                "prompt_token_ids": cleaned_ids,
                "max_new_tokens": max_new_tokens,
                "arrival_offset_ms": max(offset_ms_f, 0.0),
            }
        )
    return out


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


def _summary(samples: list[float]) -> dict[str, float | None]:
    return {
        "avg": (sum(samples) / len(samples)) if samples else None,
        "p50": _percentile(samples, 0.50),
        "p90": _percentile(samples, 0.90),
        "p99": _percentile(samples, 0.99),
    }


def _make_prompt_ids(tokenizer: AutoTokenizer, prompt_len: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    vocab = max(int(tokenizer.vocab_size), 1024)
    return [rng.randint(1, vocab - 1) for _ in range(prompt_len)]


def _run_one_model(
    model: str,
    prompt_len: int,
    output_tokens: int,
    runs: int,
    memory_ratio: float,
    seed: int,
    trace_reqs: list[dict] | None = None,
) -> tuple[list[RunMetrics], float]:
    tokenizer = AutoTokenizer.from_pretrained(model)
    prompt_ids = _make_prompt_ids(tokenizer, prompt_len=prompt_len, seed=seed)

    t_init_start = time.perf_counter()
    llm = LLM(
        model,
        dtype=torch.bfloat16,
        offload_linear_weight_to_cpu=True,
        memory_ratio=memory_ratio,
        cuda_graph_max_bs=0,
    )
    init_s = time.perf_counter() - t_init_start

    original_send = llm.send_result
    token_timestamps: list[float] = []

    def instrumented_send(reply):
        now = time.perf_counter()
        for _ in reply:
            token_timestamps.append(now)
        return original_send(reply)

    llm.send_result = instrumented_send

    all_runs: list[RunMetrics] = []
    raw_results: list[RawResult] = []
    try:
        if trace_reqs:
            requests = trace_reqs
        else:
            base_trace = TraceGenerator(
                output_length=output_tokens,
                start_timestamp=0.0,
                interval_s=0.001,
                jitter_s=0.0,
                seed=seed,
                tokenizer=tokenizer,
                input_length=prompt_len,
            ).generate(runs)
            requests = [
                {
                    "prompt_token_ids": prompt_ids,
                    "max_new_tokens": int(x.output_length),
                    "arrival_offset_ms": float(x.timestamp * 1000.0),
                }
                for x in base_trace
            ]

        replay_t0 = time.perf_counter()
        for run_idx, req in enumerate(requests):
            token_timestamps.clear()
            target_t = replay_t0 + (float(req.get("arrival_offset_ms", 0.0)) / 1000.0)
            now_t = time.perf_counter()
            if target_t > now_t:
                time.sleep(target_t - now_t)
            sp = SamplingParams(
                temperature=0.0,
                ignore_eos=True,
                max_tokens=int(req.get("max_new_tokens", output_tokens)),
            )
            prompt_for_run = req.get("prompt_token_ids", prompt_ids)
            t0 = time.perf_counter()
            out = llm.generate([prompt_for_run], sp)
            t1 = time.perf_counter()

            token_ids = out[0].get("token_ids", []) if out else []
            if not token_timestamps:
                continue
            raw_results.append(
                RawResult(
                    tics=list(token_timestamps),
                    output_len=len(token_ids),
                    message="",
                    input_len=len(prompt_for_run),
                )
            )
            ttft_ms = (token_timestamps[0] - t0) * 1000.0
            tbts_ms = [
                (token_timestamps[i + 1] - token_timestamps[i]) * 1000.0
                for i in range(len(token_timestamps) - 1)
            ]
            all_runs.append(
                RunMetrics(
                    model=model,
                    run_idx=run_idx,
                    ttft_ms=ttft_ms,
                    tbt_mean_ms=float(np.mean(tbts_ms)) if tbts_ms else 0.0,
                    tbt_p99_ms=float(np.percentile(tbts_ms, 99)) if tbts_ms else 0.0,
                    tokens=len(token_ids),
                    e2e_s=t1 - t0,
                )
            )
    finally:
        del llm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if raw_results:
        process_benchmark_results(raw_results)

    return all_runs, init_s


def main() -> None:
    parser = argparse.ArgumentParser(
        description="asymCompute benchmark (TTFT/TBT per model, comparable to mini-aegaeon report)"
    )
    parser.add_argument(
        "--models",
        required=True,
        help="Comma-separated model ids or local paths (same set used for mini-aegaeon benchmark).",
    )
    parser.add_argument("--prompt-length", type=int, default=512)
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--memory-ratio", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trace-json", default="")
    parser.add_argument("--out-json", default="benchmark/results/minisglang_offload.json")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        raise SystemExit("No models provided.")

    per_model: dict[str, dict] = {}
    per_run: list[dict] = []
    total_init_s = 0.0
    total_tokens = 0
    total_e2e_s = 0.0

    for i, model in enumerate(models):
        trace_reqs: list[dict] | None = None
        if args.trace_json:
            loaded = _load_trace(args.trace_json, default_model=model, default_max_tokens=args.output_tokens)
            # This benchmark process serves one model; keep only matching trace entries.
            trace_reqs = [x for x in loaded if x["model"] == model]
            if not trace_reqs:
                raise SystemExit(f"No trace entries for model={model} in {args.trace_json}")
        runs, init_s = _run_one_model(
            model=model,
            prompt_len=args.prompt_length,
            output_tokens=args.output_tokens,
            runs=args.runs,
            memory_ratio=args.memory_ratio,
            seed=args.seed + i,
            trace_reqs=trace_reqs,
        )
        total_init_s += init_s

        ttft_samples = [r.ttft_ms for r in runs]
        tbt_samples = [r.tbt_mean_ms for r in runs]
        tbt_p99_samples = [r.tbt_p99_ms for r in runs]
        tokens = [r.tokens for r in runs]
        e2e = [r.e2e_s for r in runs]
        total_tokens += int(sum(tokens))
        total_e2e_s += float(sum(e2e))

        per_model[model] = {
            "runs": len(runs),
            "init_time_s": init_s,
            "ttft_ms": _summary(ttft_samples),
            "tbt_ms": _summary(tbt_samples),
            "tbt_p99_ms": _summary(tbt_p99_samples),
            "tokens_avg": (sum(tokens) / len(tokens)) if tokens else None,
            "e2e_s_avg": (sum(e2e) / len(e2e)) if e2e else None,
            "ttft_ms_samples": ttft_samples,
            "tbt_ms_samples": tbt_samples,
        }
        per_run.extend([r.__dict__ for r in runs])

    all_ttft = [x for m in per_model.values() for x in m["ttft_ms_samples"]]
    all_tbt = [x for m in per_model.values() for x in m["tbt_ms_samples"]]

    report = {
        "service": "asymCompute",
        "offload_linear_weight_to_cpu": True,
        "models": models,
        "prompt_length": args.prompt_length,
        "output_tokens": args.output_tokens,
        "runs": args.runs,
        "memory_ratio": args.memory_ratio,
        "total_init_time_s": total_init_s,
        "total_tokens": total_tokens,
        "total_e2e_s": total_e2e_s,
        "ttft_ms": _summary(all_ttft),
        "tbt_ms": _summary(all_tbt),
        "per_model": per_model,
        "per_run": per_run,
    }

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
