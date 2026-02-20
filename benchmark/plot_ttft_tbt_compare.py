from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _pick_metric(report: dict, model: str, metric: str) -> float | None:
    per_model = report.get("per_model", {}).get(model, {})
    block = per_model.get(metric, {})
    value = block.get("avg")
    return float(value) if isinstance(value, (int, float)) else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot TTFT/TBT per model comparing mini-aegaeon vs mini-sglang-offload"
    )
    parser.add_argument("--aegaeon-json", required=True)
    parser.add_argument("--minisglang-json", required=True)
    parser.add_argument("--out-dir", default="benchmark/results/plots")
    args = parser.parse_args()

    aeg = _load(args.aegaeon_json)
    msgl = _load(args.minisglang_json)

    models_aeg = set(aeg.get("per_model", {}).keys())
    models_msgl = set(msgl.get("per_model", {}).keys())
    models = sorted(models_aeg & models_msgl)
    if not models:
        raise SystemExit("No overlapping models found between the two reports.")

    x = np.arange(len(models))
    w = 0.36

    aeg_ttft = [(_pick_metric(aeg, m, "ttft_ms") or np.nan) for m in models]
    msgl_ttft = [(_pick_metric(msgl, m, "ttft_ms") or np.nan) for m in models]
    aeg_tbt = [(_pick_metric(aeg, m, "tbt_ms") or np.nan) for m in models]
    msgl_tbt = [(_pick_metric(msgl, m, "tbt_ms") or np.nan) for m in models]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig1, ax1 = plt.subplots(figsize=(max(10, len(models) * 1.4), 5))
    ax1.bar(x - w / 2, aeg_ttft, width=w, label="mini-aegaeon")
    ax1.bar(x + w / 2, msgl_ttft, width=w, label="mini-sglang offload")
    ax1.set_title("TTFT Comparison (ms)")
    ax1.set_ylabel("TTFT avg (ms)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=30, ha="right")
    ax1.grid(axis="y", alpha=0.3)
    ax1.legend()
    fig1.tight_layout()
    ttft_png = out_dir / "ttft_compare.png"
    fig1.savefig(ttft_png, dpi=180)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(max(10, len(models) * 1.4), 5))
    ax2.bar(x - w / 2, aeg_tbt, width=w, label="mini-aegaeon")
    ax2.bar(x + w / 2, msgl_tbt, width=w, label="mini-sglang offload")
    ax2.set_title("TBT Comparison (ms)")
    ax2.set_ylabel("TBT avg (ms)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=30, ha="right")
    ax2.grid(axis="y", alpha=0.3)
    ax2.legend()
    fig2.tight_layout()
    tbt_png = out_dir / "tbt_compare.png"
    fig2.savefig(tbt_png, dpi=180)
    plt.close(fig2)

    merged = {
        "models": models,
        "mini_aegaeon": {
            "ttft_ms_avg": dict(zip(models, aeg_ttft, strict=False)),
            "tbt_ms_avg": dict(zip(models, aeg_tbt, strict=False)),
        },
        "mini_sglang_offload": {
            "ttft_ms_avg": dict(zip(models, msgl_ttft, strict=False)),
            "tbt_ms_avg": dict(zip(models, msgl_tbt, strict=False)),
        },
        "artifacts": {
            "ttft_plot": str(ttft_png),
            "tbt_plot": str(tbt_png),
        },
    }
    merged_json = out_dir / "comparison_merged.json"
    merged_json.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(json.dumps(merged, indent=2))


if __name__ == "__main__":
    main()
