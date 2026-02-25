# mini-aegaeon

`mini-aegaeon` is a lightweight implementation of Alibaba's Paper Aegaeon: Effective GPU Pooling for Concurrent LLM Serving on the Market

## Current Status

Implemented and usable now:

- Proxy load-balancer
- Share Memory Metadata Storage 
- Instance Runtime 
- Offline Version 

Still minimal / prototype:

- Inference Backend integration is simplified and minmal for now 
- Online version is not runnable for now 

## Environment

Project requires Python 3.10+ 

```bash
uv sync
```

## Metadata Backends

- `shared_memory` (default): `PosixShmMetadataStore`
- `redis`: `RedisMetadataStore` (Not tested)

Both implement the same `MetadataStore` interface.

## Run Online Proxy Endpoint (In Progress)

From `mini-aegaeon/`:

```bash
python -m aegaeon --host 0.0.0.0 --port 8080
```

Main routes:

- `POST /v1/chat/completions`
- `POST /v1/completions`

## Offline Benchmark Workflow

### 1) Predownload Models Locally (recommended)

This avoids including Hugging Face download time in benchmark TTFT.

```bash
python benchmark/predownload_models.py \
  --models "Qwen/Qwen3-0.6B,Qwen/Qwen2.5-0.5B-Instruct,Qwen/Qwen2.5-1.5B-Instruct,Qwen/Qwen2.5-3B-Instruct" \
  --local-root benchmark/local_models
```

Downloaded snapshots are ignored by git via:

- `benchmark/local_models/`

### 2) Run Benchmark

Single model:

```bash
CUDA_VISIBLE_DEVICES=0,1 python benchmark/offline_qwen3_colocation.py \
  --model benchmark/local_models/Qwen__Qwen3-0.6B \
  --colocated-count 2
```

Multi-model switching:

```bash
CUDA_VISIBLE_DEVICES=0,1 python benchmark/offline_qwen3_colocation.py \
  --models "benchmark/local_models/Qwen__Qwen3-0.6B,benchmark/local_models/Qwen__Qwen2.5-0.5B-Instruct,benchmark/local_models/Qwen__Qwen2.5-1.5B-Instruct,benchmark/local_models/Qwen__Qwen2.5-3B-Instruct" \
  --model-mix-policy round_robin \
  --colocated-count 2 \
  --num-requests 64 \
  --max-new-tokens 64 \
  --model-cache-budget-gb 64
```

Optional warmup preload before timed window:

```bash
--preload-models --preload-per-instance
```

## Benchmark Outputs

`benchmark/offline_qwen3_colocation.py` prints JSON including:

- aggregate throughput:
  - `req_per_s`
  - `token_per_s`
- latency:
  - `ttft_ms` (`avg`, `p50`, `p90`, `p99`)
  - `tbt_ms` (`avg`, `p50`, `p90`, `p99`)
- per-model breakdown:
  - `per_model[model].ttft_ms`
  - `per_model[model].tbt_ms`
- model-switch behavior:
  - `instance_switching[instance].switches`
  - `instance_switching[instance].switch_rate`

## Model Mix Policy

`--model-mix-policy` controls request-to-model assignment:

- `round_robin`: maximum switching pressure
- `grouped`: bursts by model (`--group-size` controls burst length)
- `random`: stochastic mix
