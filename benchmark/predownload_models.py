from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-download Hugging Face models to local paths")
    parser.add_argument(
        "--models",
        required=True,
        help="Comma-separated HF repo ids, e.g. Qwen/Qwen3-0.6B,Qwen/Qwen2.5-0.5B-Instruct",
    )
    parser.add_argument("--local-root", default="benchmark/local_models")
    parser.add_argument("--revision", default="")
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency: huggingface_hub") from exc

    root = Path(args.local_root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    model_ids = [m.strip() for m in args.models.split(",") if m.strip()]
    if not model_ids:
        raise SystemExit("No model ids provided.")

    print("Downloading models to:", root)
    for model_id in model_ids:
        local_dir = root / model_id.replace("/", "__")
        local_dir.mkdir(parents=True, exist_ok=True)
        path = snapshot_download(
            repo_id=model_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            revision=args.revision or None,
            allow_patterns=["*.json", "*.safetensors", "tokenizer*", "special_tokens_map.json"],
        )
        print(f"{model_id} -> {path}")

    print("\nUse these local directories with --models in offline_qwen3_colocation.py")


if __name__ == "__main__":
    main()
