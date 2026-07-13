#!/usr/bin/env bash
# Agentic-RL-CA shared helpers + safety guards. Sourced by run_*.sh launchers.
# Pattern inherited from reward_models/scripts/_common.sh.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO_DIR

# --- Cluster paths (env-var indirection; no cluster-internal paths in configs) ---
export MODEL_PATH="${MODEL_PATH:-/mnt/hdfs/mlsys/models/Qwen3-1.7B}"
export MODEL_PATH_FALLBACK="${MODEL_PATH_FALLBACK:-/mnt/hdfs/mlsys/models/Qwen3-4B}"   # pre-registered Wave-0 fallback
export HDFS_PROJECT="${HDFS_PROJECT:-/mnt/hdfs/mlsys/users/xiaoxuan/agentic_rl_ca}"
export SEARCHR1_DATA="${SEARCHR1_DATA:-/mnt/hdfs/mlsys/users/xiaoxuan/searchr1/searchr1_data}"
export DATA_DIR="${DATA_DIR:-$REPO_DIR/data/searchR1_processed_direct}"
export SEARCH_URL="${SEARCH_URL:-http://127.0.0.1:8000/retrieve}"
export WANDB_PROJECT="${WANDB_PROJECT:-agentic-rl-ca}"
export VENV="${VENV:-$HOME/xiaoxuan/envs/verl-agent}"

require_model_path() {
  if [[ ! -d "$MODEL_PATH" ]]; then
    echo "ERROR: MODEL_PATH does not exist: $MODEL_PATH" >&2
    echo "Do NOT download a model. Ask the user to confirm the correct path." >&2
    exit 2
  fi
}

require_data() {
  for f in "$DATA_DIR/train.parquet" "$DATA_DIR/val_2048.parquet"; do
    if [[ ! -f "$f" ]]; then
      echo "ERROR: missing $f — run scripts/prep_data.sh first." >&2
      exit 2
    fi
  done
}

require_retriever() {
  local url="${1:-$SEARCH_URL}"
  if ! curl -s --max-time 10 -X POST "$url" -H 'Content-Type: application/json' \
      -d '{"queries":["test"],"topk":1,"return_scores":true}' | grep -q result; then
    echo "ERROR: retriever not answering at $url — run scripts/retriever_serve.sh first." >&2
    exit 2
  fi
}

# Refuse GPU worker launches without the user's recorded confirmation (CONFIRM_WORKER=1).
confirm_worker() {
  local gpus="$1" gpu_type="$2" runtime="$3" job="$4" outdir="$5" cmd="$6"
  cat <<EOF
I am about to initiate a Merlin worker under the Arnold group with the following resources:

- number of GPUs: ${gpus}
- GPU type: ${gpu_type}
- expected runtime: ${runtime}
- job name: ${job}
- output directory: ${outdir}
- command to be run: ${cmd}

Please confirm before I start.
EOF
  if [[ "${CONFIRM_WORKER:-0}" != "1" ]]; then
    echo
    echo "Refusing to launch: set CONFIRM_WORKER=1 only after the user confirms." >&2
    exit 3
  fi
}

activate_env() {
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  cd "$REPO_DIR"
  export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
}
