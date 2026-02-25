from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


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
    interval_ms = 1000.0 / args.arrival_rate_rps

    trace: list[dict] = []
    for i in range(args.num_requests):
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
                "max_new_tokens": args.max_new_tokens,
                "arrival_offset_ms": i * interval_ms,
            }
        )

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(trace, indent=2), encoding="utf-8")
    print(json.dumps({"out_json": str(out), "num_requests": len(trace)}, indent=2))


if __name__ == "__main__":
    main()
