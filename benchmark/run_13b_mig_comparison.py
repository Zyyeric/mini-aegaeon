from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_MODELS = (
    "meta-llama/Llama-2-13b-hf,"
    "meta-llama/Llama-2-13b-chat-hf,"
    "Qwen/Qwen1.5-14B,"
    "Qwen/Qwen1.5-14B-Chat,"
    "Qwen/Qwen3-14B"
)


def _run(cmd: list[str], cwd: Path, env: dict[str, str]) -> None:
    print("$", " ".join(cmd))
    p = subprocess.run(cmd, cwd=str(cwd), env=env, check=False)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed with code {p.returncode}: {' '.join(cmd)}")


def _summary(samples: list[float]) -> dict[str, float | None]:
    if not samples:
        return {"avg": None, "p50": None, "p90": None, "p99": None}
    s = sorted(samples)
    n = len(s)
    p50 = s[int(0.50 * (n - 1))]
    p90 = s[int(0.90 * (n - 1))]
    p99 = s[int(0.99 * (n - 1))]
    return {"avg": sum(s) / n, "p50": p50, "p90": p90, "p99": p99}


def _to_local_model_ids(models: list[str], local_root: Path) -> list[str]:
    out: list[str] = []
    for m in models:
        p = local_root / m.replace("/", "__")
        out.append(str(p))
    return out


def _merge_minisglang_reports(
    report_paths: list[Path],
    out_json: Path,
    prompt_length: int,
    output_tokens: int,
    runs: int,
) -> dict:
    per_model: dict[str, dict] = {}
    all_ttft: list[float] = []
    all_tbt: list[float] = []
    per_run: list[dict] = []
    total_init = 0.0
    total_tokens = 0
    total_e2e = 0.0

    for rp in report_paths:
        data = json.loads(rp.read_text(encoding="utf-8"))
        model = data["models"][0]
        pm = data["per_model"][model]
        per_model[model] = pm
        all_ttft.extend(float(x) for x in pm.get("ttft_ms_samples", []))
        all_tbt.extend(float(x) for x in pm.get("tbt_ms_samples", []))
        per_run.extend(data.get("per_run", []))
        total_init += float(data.get("total_init_time_s", 0.0))
        total_tokens += int(data.get("total_tokens", 0))
        total_e2e += float(data.get("total_e2e_s", 0.0))

    merged = {
        "service": "asymCompute-mig-partitioned",
        "offload_linear_weight_to_cpu": True,
        "models": sorted(per_model.keys()),
        "prompt_length": prompt_length,
        "output_tokens": output_tokens,
        "runs": runs,
        "total_init_time_s": total_init,
        "total_tokens": total_tokens,
        "total_e2e_s": total_e2e,
        "ttft_ms": _summary(all_ttft),
        "tbt_ms": _summary(all_tbt),
        "per_model": per_model,
        "per_run": per_run,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MIG-partitioned asymCompute vs single-instance mini-aegaeon comparison"
    )
    parser.add_argument("--models", default=DEFAULT_MODELS)
    parser.add_argument(
        "--mig-uuids",
        required=True,
        help="Comma-separated MIG UUIDs for mini-sglang runs, e.g. MIG-xxx,MIG-yyy,MIG-zzz",
    )
    parser.add_argument(
        "--model-source",
        choices=["hf", "local"],
        default="local",
        help="Use HF ids directly or local snapshot directories.",
    )
    parser.add_argument("--local-root", default="benchmark/local_models")
    parser.add_argument("--predownload", action="store_true")
    parser.add_argument("--aegaeon-cuda-devices", default="0", help="CUDA_VISIBLE_DEVICES for mini-aegaeon run")
    parser.add_argument("--colocated-count", type=int, default=1)
    parser.add_argument("--num-requests", type=int, default=64)
    parser.add_argument("--prompt-length", type=int, default=512)
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--runs", type=int, default=5, help="Per-model runs for mini-sglang")
    parser.add_argument("--memory-ratio", type=float, default=0.85)
    parser.add_argument("--model-cache-budget-gb", type=float, default=84.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--master-port-base", type=int, default=29600)
    parser.add_argument("--results-dir", default="benchmark/results/mig_vs_aegaeon_13b")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    migs = [x.strip() for x in args.mig_uuids.split(",") if x.strip()]
    if not models:
        raise SystemExit("No models provided.")
    if not migs:
        raise SystemExit("No MIG UUIDs provided.")

    local_root = (repo_root / args.local_root).resolve()
    if args.predownload:
        cmd = [
            sys.executable,
            "benchmark/predownload_models.py",
            "--models",
            ",".join(models),
            "--local-root",
            str(local_root),
        ]
        _run(cmd=cmd, cwd=repo_root, env=os.environ.copy())

    model_refs = models if args.model_source == "hf" else _to_local_model_ids(models, local_root)
    if args.model_source == "local":
        missing = [m for m in model_refs if not Path(m).exists()]
        if missing:
            raise SystemExit(
                "Missing local model snapshots:\n"
                + "\n".join(missing)
                + "\nRun with --predownload or switch --model-source hf."
            )

    results_dir = (repo_root / args.results_dir).resolve()
    minisgl_dir = results_dir / "minisglang_per_model"
    minisgl_dir.mkdir(parents=True, exist_ok=True)

    # 1) asymCompute on MIG partitions (concurrent in waves)
    minisgl_reports: list[Path] = []
    wave_count = int(math.ceil(len(model_refs) / len(migs)))
    for wave in range(wave_count):
        procs: list[tuple[subprocess.Popen, str, Path]] = []
        for i, mig in enumerate(migs):
            model_idx = wave * len(migs) + i
            if model_idx >= len(model_refs):
                continue
            model = model_refs[model_idx]
            safe = model.replace("/", "__")
            out_json = minisgl_dir / f"{safe}.json"
            minisgl_reports.append(out_json)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = mig
            env["MASTER_PORT"] = str(args.master_port_base + i)
            cmd = [
                sys.executable,
                "benchmark/benchmark_minisglang_offload.py",
                "--models",
                model,
                "--prompt-length",
                str(args.prompt_length),
                "--output-tokens",
                str(args.output_tokens),
                "--runs",
                str(args.runs),
                "--memory-ratio",
                str(args.memory_ratio),
                "--seed",
                str(args.seed + model_idx),
                "--out-json",
                str(out_json),
            ]
            print("$", " ".join(cmd), f"[CUDA_VISIBLE_DEVICES={mig}]")
            p = subprocess.Popen(cmd, cwd=str(repo_root), env=env)
            procs.append((p, model, out_json))

        for p, model, out_json in procs:
            rc = p.wait()
            if rc != 0:
                raise RuntimeError(f"mini-sglang run failed for model={model}, out={out_json}, rc={rc}")

    minisgl_merged_json = results_dir / "minisglang_offload_merged.json"
    _merge_minisglang_reports(
        report_paths=minisgl_reports,
        out_json=minisgl_merged_json,
        prompt_length=args.prompt_length,
        output_tokens=args.output_tokens,
        runs=args.runs,
    )

    # 2) mini-aegaeon single-instance shared-GPU switching benchmark
    aegaeon_json = results_dir / "mini_aegaeon.json"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.aegaeon_cuda_devices
    cmd = [
        sys.executable,
        "benchmark/offline_qwen3_colocation.py",
        "--models",
        ",".join(model_refs),
        "--model-mix-policy",
        "round_robin",
        "--num-requests",
        str(args.num_requests),
        "--prompt-chars",
        str(args.prompt_length),
        "--max-new-tokens",
        str(args.output_tokens),
        "--colocated-count",
        str(args.colocated_count),
        "--model-cache-budget-gb",
        str(args.model_cache_budget_gb),
        "--out-json",
        str(aegaeon_json),
    ]
    _run(cmd=cmd, cwd=repo_root, env=env)

    # 3) plot comparison
    plots_dir = results_dir / "plots"
    cmd = [
        sys.executable,
        "benchmark/plot_ttft_tbt_compare.py",
        "--aegaeon-json",
        str(aegaeon_json),
        "--minisglang-json",
        str(minisgl_merged_json),
        "--out-dir",
        str(plots_dir),
    ]
    _run(cmd=cmd, cwd=repo_root, env=os.environ.copy())

    print("\nExperiment completed.")
    print(f"- mini-sglang merged: {minisgl_merged_json}")
    print(f"- mini-aegaeon:       {aegaeon_json}")
    print(f"- plots:              {plots_dir}")


if __name__ == "__main__":
    main()
