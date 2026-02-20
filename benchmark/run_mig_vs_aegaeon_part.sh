#!/usr/bin/env bash
set -euo pipefail

# Run one part at a time:
#   mig_on   : enable MIG and create partitions
#   mig_off  : destroy MIG partitions and disable MIG
#   sglang   : run mini-sglang offload benchmark on MIG UUIDs
#   aegaeon  : run mini-aegaeon benchmark on full GPU
#   plot     : plot TTFT/TBT comparison from existing JSON outputs

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PART=""
GPU_INDEX=0
MIG_PROFILE_ID=19
MIG_COUNT=3
MIG_UUIDS=""
MASTER_PORT_BASE=29600

MODELS="meta-llama/Llama-2-13b-hf,meta-llama/Llama-2-13b-chat-hf,Qwen/Qwen1.5-14B,Qwen/Qwen1.5-14B-Chat,Qwen/Qwen3-14B"
MODEL_SOURCE="local" # local | hf
LOCAL_ROOT="benchmark/local_models"
PREDOWNLOAD=0

AEGAEON_CUDA_DEVICES="0"
COLOCATED_COUNT=1
NUM_REQUESTS=64
PROMPT_LENGTH=512
OUTPUT_TOKENS=64
RUNS=5
MEMORY_RATIO=0.85
MODEL_CACHE_BUDGET_GB=84
SEED=0
RESULTS_DIR="benchmark/results/mig_vs_aegaeon_13b"

usage() {
  cat <<EOF
Usage:
  bash benchmark/run_mig_vs_aegaeon_part.sh --part <mig_on|mig_off|sglang|aegaeon|plot> [options]

Core options:
  --part <name>                     Which part to run
  --models "<csv>"                  Model ids (default: 13B/14B set)
  --model-source <local|hf>         Use local snapshots or HF ids (default: local)
  --local-root <path>               Local snapshot root (default: benchmark/local_models)
  --predownload                     Pre-download models before sglang/aegaeon run (local mode)
  --results-dir <path>              Output directory (default: benchmark/results/mig_vs_aegaeon_13b)

MIG control:
  --gpu-index <int>                 GPU index for MIG commands (default: 0)
  --mig-profile-id <int>            MIG profile id (default: 19 -> 1g.12gb on many GPUs)
  --mig-count <int>                 Number of MIG partitions to create (default: 3)
  --mig-uuids "<csv>"               MIG UUIDs for sglang part
  --master-port-base <int>          Base MASTER_PORT for concurrent sglang jobs (default: 29600)

Benchmark knobs:
  --aegaeon-cuda-devices "<ids>"    CUDA_VISIBLE_DEVICES for aegaeon run (default: 0)
  --colocated-count <int>           Aegaeon colocated count (default: 1)
  --num-requests <int>              Aegaeon requests (default: 64)
  --prompt-length <int>             Prompt length/chars (default: 512)
  --output-tokens <int>             Generated tokens (default: 64)
  --runs <int>                      mini-sglang runs per model (default: 5)
  --memory-ratio <float>            mini-sglang memory ratio (default: 0.85)
  --model-cache-budget-gb <float>   Aegaeon model cache budget (default: 84)
  --seed <int>                      Random seed (default: 0)

Examples:
  bash benchmark/run_mig_vs_aegaeon_part.sh --part mig_on
  bash benchmark/run_mig_vs_aegaeon_part.sh --part sglang --mig-uuids "MIG-aaa,MIG-bbb,MIG-ccc" --predownload
  bash benchmark/run_mig_vs_aegaeon_part.sh --part mig_off
  bash benchmark/run_mig_vs_aegaeon_part.sh --part aegaeon
  bash benchmark/run_mig_vs_aegaeon_part.sh --part plot
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --part) PART="$2"; shift 2 ;;
    --gpu-index) GPU_INDEX="$2"; shift 2 ;;
    --mig-profile-id) MIG_PROFILE_ID="$2"; shift 2 ;;
    --mig-count) MIG_COUNT="$2"; shift 2 ;;
    --mig-uuids) MIG_UUIDS="$2"; shift 2 ;;
    --master-port-base) MASTER_PORT_BASE="$2"; shift 2 ;;
    --models) MODELS="$2"; shift 2 ;;
    --model-source) MODEL_SOURCE="$2"; shift 2 ;;
    --local-root) LOCAL_ROOT="$2"; shift 2 ;;
    --predownload) PREDOWNLOAD=1; shift 1 ;;
    --aegaeon-cuda-devices) AEGAEON_CUDA_DEVICES="$2"; shift 2 ;;
    --colocated-count) COLOCATED_COUNT="$2"; shift 2 ;;
    --num-requests) NUM_REQUESTS="$2"; shift 2 ;;
    --prompt-length) PROMPT_LENGTH="$2"; shift 2 ;;
    --output-tokens) OUTPUT_TOKENS="$2"; shift 2 ;;
    --runs) RUNS="$2"; shift 2 ;;
    --memory-ratio) MEMORY_RATIO="$2"; shift 2 ;;
    --model-cache-budget-gb) MODEL_CACHE_BUDGET_GB="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --results-dir) RESULTS_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "${PART}" ]]; then
  echo "--part is required"
  usage
  exit 1
fi

mkdir -p "${RESULTS_DIR}"

split_csv_to_array() {
  local csv="$1"
  local -n out_arr="$2"
  IFS=',' read -r -a out_arr <<< "${csv}"
}

build_model_refs() {
  local -n in_models="$1"
  local -n out_refs="$2"
  out_refs=()
  if [[ "${MODEL_SOURCE}" == "hf" ]]; then
    out_refs=("${in_models[@]}")
    return
  fi
  for m in "${in_models[@]}"; do
    local safe="${m//\//__}"
    out_refs+=("${LOCAL_ROOT}/${safe}")
  done
}

predownload_if_requested() {
  local models_csv="$1"
  if [[ "${PREDOWNLOAD}" -eq 1 && "${MODEL_SOURCE}" == "local" ]]; then
    python benchmark/predownload_models.py --models "${models_csv}" --local-root "${LOCAL_ROOT}"
  fi
}

run_mig_on() {
  echo "[MIG ON] Enabling MIG on GPU ${GPU_INDEX} and creating ${MIG_COUNT} partition(s) profile ${MIG_PROFILE_ID}"
  sudo nvidia-smi -i "${GPU_INDEX}" -mig 1
  sleep 2
  sudo nvidia-smi mig -i "${GPU_INDEX}" -dci || true
  sudo nvidia-smi mig -i "${GPU_INDEX}" -dgi || true
  sleep 2
  local profiles
  profiles="$(yes "${MIG_PROFILE_ID}" | head -n "${MIG_COUNT}" | paste -sd, -)"
  sudo nvidia-smi mig -i "${GPU_INDEX}" -cgi "${profiles}" -C
  sleep 2
  nvidia-smi -L
  echo "Collect MIG UUIDs with:"
  echo "  nvidia-smi -L | grep MIG | grep -o 'MIG-[^)]*'"
}

run_mig_off() {
  echo "[MIG OFF] Destroying MIG instances and disabling MIG on GPU ${GPU_INDEX}"
  sudo nvidia-smi mig -i "${GPU_INDEX}" -dci || true
  sudo nvidia-smi mig -i "${GPU_INDEX}" -dgi || true
  sleep 2
  sudo nvidia-smi -i "${GPU_INDEX}" -mig 0
  sleep 2
  nvidia-smi -L
}

run_sglang() {
  local models_arr model_refs migs
  split_csv_to_array "${MODELS}" models_arr
  predownload_if_requested "${MODELS}"
  build_model_refs models_arr model_refs

  if [[ -z "${MIG_UUIDS}" ]]; then
    echo "--mig-uuids is required for --part sglang"
    exit 1
  fi
  split_csv_to_array "${MIG_UUIDS}" migs
  if [[ "${#migs[@]}" -eq 0 ]]; then
    echo "No MIG UUIDs parsed from --mig-uuids"
    exit 1
  fi

  local per_model_dir="${RESULTS_DIR}/minisglang_per_model"
  mkdir -p "${per_model_dir}"

  if [[ "${MODEL_SOURCE}" == "local" ]]; then
    for m in "${model_refs[@]}"; do
      if [[ ! -d "${m}" ]]; then
        echo "Missing local model dir: ${m}"
        echo "Run with --predownload or pass --model-source hf"
        exit 1
      fi
    done
  fi

  local total="${#model_refs[@]}"
  local n_mig="${#migs[@]}"
  local wave_count=$(( (total + n_mig - 1) / n_mig ))

  echo "[SGLANG] Running ${total} model(s) across ${n_mig} MIG partition(s) in ${wave_count} wave(s)"

  for ((wave=0; wave<wave_count; wave++)); do
    pids=()
    for ((i=0; i<n_mig; i++)); do
      idx=$(( wave * n_mig + i ))
      if (( idx >= total )); then
        continue
      fi
      model="${model_refs[$idx]}"
      safe="${model//\//__}"
      out_json="${per_model_dir}/${safe}.json"
      mig="${migs[$i]}"
      port=$(( MASTER_PORT_BASE + i ))

      echo "  wave=${wave} slot=${i} model=${model} mig=${mig} port=${port}"
      CUDA_VISIBLE_DEVICES="${mig}" MASTER_PORT="${port}" \
      python benchmark/benchmark_minisglang_offload.py \
        --models "${model}" \
        --prompt-length "${PROMPT_LENGTH}" \
        --output-tokens "${OUTPUT_TOKENS}" \
        --runs "${RUNS}" \
        --memory-ratio "${MEMORY_RATIO}" \
        --seed "$(( SEED + idx ))" \
        --out-json "${out_json}" &
      pids+=("$!")
    done
    for pid in "${pids[@]}"; do
      wait "${pid}"
    done
  done

  local merged_json="${RESULTS_DIR}/minisglang_offload_merged.json"
  python - "${per_model_dir}" "${merged_json}" "${PROMPT_LENGTH}" "${OUTPUT_TOKENS}" "${RUNS}" <<'PY'
import json
import math
import sys
from pathlib import Path

per_model_dir = Path(sys.argv[1])
merged_json = Path(sys.argv[2])
prompt_length = int(sys.argv[3])
output_tokens = int(sys.argv[4])
runs = int(sys.argv[5])

def summary(xs):
    if not xs:
        return {"avg": None, "p50": None, "p90": None, "p99": None}
    s = sorted(xs)
    n = len(s)
    return {
        "avg": sum(s) / n,
        "p50": s[int(0.50 * (n - 1))],
        "p90": s[int(0.90 * (n - 1))],
        "p99": s[int(0.99 * (n - 1))],
    }

per_model = {}
per_run = []
all_ttft = []
all_tbt = []
total_init = 0.0
total_tokens = 0
total_e2e = 0.0

for p in sorted(per_model_dir.glob("*.json")):
    d = json.loads(p.read_text())
    model = d["models"][0]
    pm = d["per_model"][model]
    per_model[model] = pm
    all_ttft.extend(pm.get("ttft_ms_samples", []))
    all_tbt.extend(pm.get("tbt_ms_samples", []))
    per_run.extend(d.get("per_run", []))
    total_init += float(d.get("total_init_time_s", 0.0))
    total_tokens += int(d.get("total_tokens", 0))
    total_e2e += float(d.get("total_e2e_s", 0.0))

merged = {
    "service": "mini-sglang-offload-mig-partitioned",
    "offload_linear_weight_to_cpu": True,
    "models": sorted(per_model.keys()),
    "prompt_length": prompt_length,
    "output_tokens": output_tokens,
    "runs": runs,
    "total_init_time_s": total_init,
    "total_tokens": total_tokens,
    "total_e2e_s": total_e2e,
    "ttft_ms": summary([float(x) for x in all_ttft]),
    "tbt_ms": summary([float(x) for x in all_tbt]),
    "per_model": per_model,
    "per_run": per_run,
}
merged_json.write_text(json.dumps(merged, indent=2), encoding="utf-8")
print(json.dumps({"merged_json": str(merged_json), "models": merged["models"]}, indent=2))
PY
}

run_aegaeon() {
  local models_arr model_refs
  split_csv_to_array "${MODELS}" models_arr
  predownload_if_requested "${MODELS}"
  build_model_refs models_arr model_refs

  if [[ "${MODEL_SOURCE}" == "local" ]]; then
    for m in "${model_refs[@]}"; do
      if [[ ! -d "${m}" ]]; then
        echo "Missing local model dir: ${m}"
        echo "Run with --predownload or pass --model-source hf"
        exit 1
      fi
    done
  fi

  local joined_models
  joined_models="$(IFS=,; echo "${model_refs[*]}")"
  local out_json="${RESULTS_DIR}/mini_aegaeon.json"

  echo "[AEGAEON] Running single-instance switching workload on CUDA_VISIBLE_DEVICES=${AEGAEON_CUDA_DEVICES}"
  CUDA_VISIBLE_DEVICES="${AEGAEON_CUDA_DEVICES}" \
  python benchmark/offline_qwen3_colocation.py \
    --models "${joined_models}" \
    --model-mix-policy round_robin \
    --num-requests "${NUM_REQUESTS}" \
    --prompt-chars "${PROMPT_LENGTH}" \
    --max-new-tokens "${OUTPUT_TOKENS}" \
    --colocated-count "${COLOCATED_COUNT}" \
    --model-cache-budget-gb "${MODEL_CACHE_BUDGET_GB}" \
    --out-json "${out_json}"

  echo "Saved: ${out_json}"
}

run_plot() {
  local aeg_json="${RESULTS_DIR}/mini_aegaeon.json"
  local sgl_json="${RESULTS_DIR}/minisglang_offload_merged.json"
  local out_dir="${RESULTS_DIR}/plots"
  if [[ ! -f "${aeg_json}" ]]; then
    echo "Missing ${aeg_json}"
    exit 1
  fi
  if [[ ! -f "${sgl_json}" ]]; then
    echo "Missing ${sgl_json}"
    exit 1
  fi
  python benchmark/plot_ttft_tbt_compare.py \
    --aegaeon-json "${aeg_json}" \
    --minisglang-json "${sgl_json}" \
    --out-dir "${out_dir}"
}

case "${PART}" in
  mig_on) run_mig_on ;;
  mig_off) run_mig_off ;;
  sglang) run_sglang ;;
  aegaeon) run_aegaeon ;;
  plot) run_plot ;;
  *) echo "Unsupported --part: ${PART}"; usage; exit 1 ;;
esac
