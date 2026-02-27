from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from typing import Any
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from benchmark.benchmark_core import BenchmarkTrace, generate_prompt


@dataclass(frozen=True)
class TraceGenerator:
    output_length: int = 64
    start_timestamp: float = 0.0
    interval_s: float = 0.5
    jitter_s: float = 0.0
    seed: int = 0
    tokenizer: Any | None = None
    input_length: int | None = None

    def generate(self, num_requests: int) -> list[BenchmarkTrace]:
        if num_requests <= 0:
            raise ValueError("num_requests must be > 0")
        if self.interval_s <= 0:
            raise ValueError("interval_s must be > 0")
        if self.jitter_s < 0:
            raise ValueError("jitter_s must be >= 0")

        rng = random.Random(self.seed)
        traces: list[BenchmarkTrace] = []
        for i in range(num_requests):
            base_ts = self.start_timestamp + i * self.interval_s
            ts = max(0.0, base_ts + rng.uniform(-self.jitter_s, self.jitter_s))
            if self.input_length is not None and self.tokenizer is not None:
                message = generate_prompt(self.tokenizer, self.input_length, rng=rng)
            elif self.input_length is not None:
                message = " ".join(["token"] * self.input_length)
            else:
                message = f"trace-{i}"
            traces.append(
                BenchmarkTrace(
                    timestamp=ts,
                    message=message,
                    output_length=self.output_length,
                    input_length=self.input_length,
                )
            )
        return sorted(traces, key=lambda x: x.timestamp)


BenchmarkTraceGenerator = TraceGenerator


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic request trace for benchmarking")
    parser.add_argument("--model", default="", help="Model id/path for all trace requests")
    parser.add_argument("--models", default="", help="Comma-separated model ids/paths")
    parser.add_argument(
        "--model-mix-policy",
        choices=["round_robin", "random"],
        default="round_robin",
        help="How to choose model per request when --models is set.",
    )
    parser.add_argument("--num-requests", type=int, default=5)
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--arrival-rate-rps", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    if args.num_requests <= 0:
        raise SystemExit("--num-requests must be > 0")
    if args.prompt_tokens <= 0:
        raise SystemExit("--prompt-tokens must be > 0")
    if args.max_new_tokens <= 0:
        raise SystemExit("--max-new-tokens must be > 0")
    if args.arrival_rate_rps <= 0:
        raise SystemExit("--arrival-rate-rps must be > 0")

    models = [m.strip() for m in args.models.split(",") if m.strip()] if args.models else []
    if not models:
        if not args.model:
            raise SystemExit("Provide --model or --models")
        models = [args.model]

    rng = random.Random(args.seed)
    interval_s = 1.0 / args.arrival_rate_rps
    base_trace = TraceGenerator(
        output_length=args.max_new_tokens,
        start_timestamp=0.0,
        interval_s=interval_s,
        jitter_s=0.0,
        seed=args.seed,
        input_length=args.prompt_tokens,
    ).generate(args.num_requests)

    trace: list[dict] = []
    for i, bt in enumerate(base_trace):
        if len(models) == 1:
            model = models[0]
        elif args.model_mix_policy == "random":
            model = rng.choice(models)
        else:
            model = models[i % len(models)]
        prompt_token_ids = [rng.randint(1, 10_000) for _ in range(args.prompt_tokens)]
        trace.append(
            {
                "request_id": f"trace-{i}",
                "model": model,
                "prompt_token_ids": prompt_token_ids,
                "max_new_tokens": bt.output_length,
                "arrival_offset_ms": bt.timestamp * 1000.0,
            }
        )

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(trace, indent=2), encoding="utf-8")
    print(json.dumps({"out_json": str(out), "num_requests": len(trace)}, indent=2))


if __name__ == "__main__":
    main()
