from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _normalize_model(model: str) -> str:
    # Local snapshots are saved like ".../Qwen__Qwen3-14B"; normalize to "Qwen/Qwen3-14B".
    if "__" in model:
        return model.split("/")[-1].replace("__", "/")
    return model


def _short_label(model: str) -> str:
    return model.split("/")[-1]


def _pick_metric(report: dict, model_key: str, metric: str) -> float | None:
    per_model = report.get("per_model", {}).get(model_key, {})
    block = per_model.get(metric, {})
    value = block.get("avg")
    if isinstance(value, (int, float)):
        return float(value)
    # Some reports may only expose scalar run mean for TBT.
    if metric == "tbt_ms":
        scalar = per_model.get("tbt_mean_ms")
        if isinstance(scalar, (int, float)):
            return float(scalar)
    return None


def _pick_samples(report: dict, model_key: str, metric: str) -> list[float]:
    per_model = report.get("per_model", {}).get(model_key, {})
    if metric == "tbt_ms":
        # Prefer per-request TBT samples when present (mini-aegaeon),
        # to match asymCompute granularity more closely.
        preferred = per_model.get("tbt_ms_per_request_samples")
        if isinstance(preferred, list):
            out_pref: list[float] = []
            for x in preferred:
                if isinstance(x, (int, float)):
                    out_pref.append(float(x))
            if out_pref:
                return out_pref
    key = f"{metric}_samples"
    raw = per_model.get(key, [])
    if not isinstance(raw, list):
        return []
    out: list[float] = []
    for x in raw:
        if isinstance(x, (int, float)):
            out.append(float(x))
    return out


def _ecdf(values: list[float]) -> tuple[np.ndarray, np.ndarray]:
    if not values:
        return np.array([]), np.array([])
    x = np.sort(np.array(values, dtype=float))
    y = np.arange(1, len(x) + 1, dtype=float) / float(len(x))
    return x, y


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot TTFT/TBT per model comparing mini-aegaeon vs AsymCompute"
    )
    parser.add_argument("--aegaeon-json", required=True)
    parser.add_argument("--asymcompute-json")
    parser.add_argument("--minisglang-json")
    parser.add_argument("--out-dir", default="benchmark/results/plots")
    args = parser.parse_args()
    asym_json = args.asymcompute_json or args.minisglang_json
    if not asym_json:
        raise SystemExit("Provide --asymcompute-json (preferred) or --minisglang-json.")

    aeg = _load(args.aegaeon_json)
    asym = _load(asym_json)

    aeg_map = {_normalize_model(m): m for m in aeg.get("per_model", {}).keys()}
    asym_map = {_normalize_model(m): m for m in asym.get("per_model", {}).keys()}
    models = sorted(set(aeg_map.keys()) & set(asym_map.keys()))
    if not models:
        raise SystemExit("No overlapping models found between the two reports.")

    x = np.arange(len(models))
    w = 0.36

    def _val_or_nan(v: float | None) -> float:
        return float(v) if v is not None else np.nan

    aeg_ttft = [_val_or_nan(_pick_metric(aeg, aeg_map[m], "ttft_ms")) for m in models]
    asym_ttft = [_val_or_nan(_pick_metric(asym, asym_map[m], "ttft_ms")) for m in models]
    aeg_tbt = [_val_or_nan(_pick_metric(aeg, aeg_map[m], "tbt_ms")) for m in models]
    asym_tbt = [_val_or_nan(_pick_metric(asym, asym_map[m], "tbt_ms")) for m in models]
    labels = [_short_label(m) for m in models]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig1, ax1 = plt.subplots(figsize=(max(10, len(models) * 1.4), 5))
    ax1.bar(x - w / 2, aeg_ttft, width=w, label="mini-aegaeon")
    ax1.bar(x + w / 2, asym_ttft, width=w, label="AsymCompute")
    ax1.set_title("TTFT Comparison (ms)")
    ax1.set_ylabel("TTFT avg (ms)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=0, ha="center")
    ax1.grid(axis="y", alpha=0.3)
    ax1.legend()
    fig1.tight_layout()
    ttft_png = out_dir / "ttft_compare.png"
    fig1.savefig(ttft_png, dpi=180)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(max(10, len(models) * 1.4), 5))
    ax2.bar(x - w / 2, aeg_tbt, width=w, label="mini-aegaeon")
    ax2.bar(x + w / 2, asym_tbt, width=w, label="AsymCompute")
    finite_tbt = [v for v in (aeg_tbt + asym_tbt) if np.isfinite(v) and v > 0]
    use_log_tbt = False
    if finite_tbt:
        tbt_min = min(finite_tbt)
        tbt_max = max(finite_tbt)
        use_log_tbt = (tbt_max / tbt_min) >= 20.0
    if use_log_tbt:
        ax2.set_yscale("log")
        ax2.set_title("TBT Comparison (ms, log scale)")
    else:
        ax2.set_title("TBT Comparison (ms)")
    ax2.set_ylabel("TBT avg (ms)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=0, ha="center")
    ax2.grid(axis="y", alpha=0.3)
    ax2.legend()
    fig2.tight_layout()
    tbt_png = out_dir / "tbt_compare.png"
    fig2.savefig(tbt_png, dpi=180)
    plt.close(fig2)

    aeg_ttft_samples_all: list[float] = []
    asym_ttft_samples_all: list[float] = []
    aeg_tbt_samples_all: list[float] = []
    asym_tbt_samples_all: list[float] = []
    for m in models:
        aeg_ttft_samples_all.extend(_pick_samples(aeg, aeg_map[m], "ttft_ms"))
        asym_ttft_samples_all.extend(_pick_samples(asym, asym_map[m], "ttft_ms"))
        aeg_tbt_samples_all.extend(_pick_samples(aeg, aeg_map[m], "tbt_ms"))
        asym_tbt_samples_all.extend(_pick_samples(asym, asym_map[m], "tbt_ms"))

    fig3, ax3 = plt.subplots(figsize=(9, 5))
    x_aeg_ttft, y_aeg_ttft = _ecdf(aeg_ttft_samples_all)
    x_asym_ttft, y_asym_ttft = _ecdf(asym_ttft_samples_all)
    if len(x_aeg_ttft) > 0:
        ax3.plot(x_aeg_ttft, y_aeg_ttft, label="mini-aegaeon")
    if len(x_asym_ttft) > 0:
        ax3.plot(x_asym_ttft, y_asym_ttft, label="AsymCompute")
    ax3.set_title("TTFT CDF")
    ax3.set_xlabel("TTFT (ms)")
    ax3.set_ylabel("CDF")
    ax3.grid(alpha=0.3)
    ax3.legend()
    fig3.tight_layout()
    ttft_cdf_png = out_dir / "ttft_cdf.png"
    fig3.savefig(ttft_cdf_png, dpi=180)
    plt.close(fig3)

    fig4, ax4 = plt.subplots(figsize=(9, 5))
    x_aeg_tbt, y_aeg_tbt = _ecdf(aeg_tbt_samples_all)
    x_asym_tbt, y_asym_tbt = _ecdf(asym_tbt_samples_all)
    if len(x_aeg_tbt) > 0:
        ax4.plot(x_aeg_tbt, y_aeg_tbt, label="mini-aegaeon")
    if len(x_asym_tbt) > 0:
        ax4.plot(x_asym_tbt, y_asym_tbt, label="AsymCompute")
    finite_tbt_samples = [v for v in (aeg_tbt_samples_all + asym_tbt_samples_all) if v > 0]
    if finite_tbt_samples:
        tbt_min = min(finite_tbt_samples)
        tbt_max = max(finite_tbt_samples)
        if tbt_min > 0 and (tbt_max / tbt_min) >= 20.0:
            ax4.set_xscale("log")
    ax4.set_title("TBT CDF")
    ax4.set_xlabel("TBT (ms)")
    ax4.set_ylabel("CDF")
    ax4.grid(alpha=0.3)
    ax4.legend()
    fig4.tight_layout()
    tbt_cdf_png = out_dir / "tbt_cdf.png"
    fig4.savefig(tbt_cdf_png, dpi=180)
    plt.close(fig4)

    merged = {
        "models": models,
        "mini_aegaeon": {
            "ttft_ms_avg": dict(zip(models, aeg_ttft, strict=False)),
            "tbt_ms_avg": dict(zip(models, aeg_tbt, strict=False)),
        },
        "asymcompute": {
            "ttft_ms_avg": dict(zip(models, asym_ttft, strict=False)),
            "tbt_ms_avg": dict(zip(models, asym_tbt, strict=False)),
        },
        "artifacts": {
            "ttft_plot": str(ttft_png),
            "tbt_plot": str(tbt_png),
            "ttft_cdf_plot": str(ttft_cdf_png),
            "tbt_cdf_plot": str(tbt_cdf_png),
        },
    }
    merged_json = out_dir / "comparison_merged.json"
    merged_json.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(json.dumps(merged, indent=2))


if __name__ == "__main__":
    main()
