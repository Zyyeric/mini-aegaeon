from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path


def _parse_models(models_csv: str) -> list[str]:
    return [m.strip() for m in models_csv.split(",") if m.strip()]


def _measure_one_child(
    out_q: mp.Queue,
    model: str,
    *,
    master_port: int,
    memory_ratio: float,
    model_switching: bool,
    budget_model: str | None,
    use_dummy_weight: bool,
) -> None:
    llm = None
    try:
        import torch
        from minisgl.llm import LLM

        # Do not touch any torch.cuda API before LLM() construction.
        # minisgl Engine asserts CUDA is not initialized at startup.
        free_before = None

        os.environ["MASTER_PORT"] = str(master_port)
        t0 = time.perf_counter()
        llm = LLM(
            model,
            dtype=torch.bfloat16,
            offload_linear_weight_to_cpu=False,
            memory_ratio=memory_ratio,
            model_switching=model_switching,
            model_switching_budget_model_path=budget_model,
            use_dummy_weight=use_dummy_weight,
            cuda_graph_max_bs=0,
        )
        init_s = time.perf_counter() - t0

        free_after = None
        try:
            if torch.cuda.is_available():
                free_after = float(torch.cuda.mem_get_info()[0]) / (1024.0**3)
        except Exception:
            free_after = None

        rec = {
            "model": model,
            "master_port": master_port,
            "init_time_s": init_s,
            "free_before_gib": free_before,
            "free_after_gib": free_after,
            "free_drop_gib": (
                (free_before - free_after)
                if isinstance(free_before, float) and isinstance(free_after, float)
                else None
            ),
        }
        out_q.put({"ok": True, "record": rec})
    except Exception as exc:
        out_q.put({"ok": False, "err": f"{type(exc).__name__}: {exc}"})
    finally:
        if llm is not None:
            try:
                llm.shutdown()
            except Exception:
                pass


def _measure_one(
    model: str,
    *,
    master_port: int,
    memory_ratio: float,
    model_switching: bool,
    budget_model: str | None,
    use_dummy_weight: bool,
) -> dict:
    ctx = mp.get_context("spawn")
    out_q: mp.Queue = ctx.Queue()
    p = ctx.Process(
        target=_measure_one_child,
        args=(out_q, model),
        kwargs={
            "master_port": master_port,
            "memory_ratio": memory_ratio,
            "model_switching": model_switching,
            "budget_model": budget_model,
            "use_dummy_weight": use_dummy_weight,
        },
        daemon=False,
    )
    p.start()
    msg = out_q.get(timeout=900)
    p.join(timeout=10)
    if p.is_alive():
        p.terminate()
        p.join(timeout=2)

    if not isinstance(msg, dict) or not msg.get("ok", False):
        err = msg.get("err", "worker measurement failed") if isinstance(msg, dict) else "invalid response"
        raise RuntimeError(str(err))

    rec = msg.get("record")
    if not isinstance(rec, dict):
        raise RuntimeError("worker returned invalid record")
    return rec


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-time mini-sgl worker initialization time measurement for TTFT adjustment"
    )
    parser.add_argument(
        "--models",
        required=True,
        help="Comma-separated model ids/paths in the same initialization order as your run.",
    )
    parser.add_argument("--memory-ratio", type=float, default=0.3)
    parser.add_argument("--model-switching", action="store_true")
    parser.add_argument("--budget-model", default="", help="Optional budget model path for model switching.")
    parser.add_argument("--use-dummy-weight", action="store_true")
    parser.add_argument("--master-port-base", type=int, default=33000)
    parser.add_argument("--out-json", default="")
    args = parser.parse_args()

    models = _parse_models(args.models)
    if not models:
        raise SystemExit("No models provided.")

    budget_model = args.budget_model.strip() or None

    records: list[dict] = []
    for i, model in enumerate(models):
        rec = _measure_one(
            model,
            master_port=args.master_port_base + i,
            memory_ratio=args.memory_ratio,
            model_switching=args.model_switching,
            budget_model=budget_model,
            use_dummy_weight=args.use_dummy_weight,
        )
        records.append(rec)

    subtract_s = sum(float(x["init_time_s"]) for x in records[1:])
    result = {
        "models_in_order": models,
        "memory_ratio": args.memory_ratio,
        "model_switching": bool(args.model_switching),
        "budget_model": budget_model,
        "use_dummy_weight": bool(args.use_dummy_weight),
        "worker_init": records,
        "ttft_subtract_seconds_except_first": subtract_s,
        "ttft_subtract_ms_except_first": subtract_s * 1000.0,
    }

    print(json.dumps(result, indent=2))
    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
