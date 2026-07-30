#!/bin/bash
# B200 ONE-CYCLE PILOT (user request 2026-07-31): GRPO fast config, TOTAL_STEPS=1,
# identical settings to the H100 fast run (KV 0.5, DYNBSZ 24576, partial cycle 8) for a
# direct cycle-1 comparison (H100 fast cycle 1 = 672s total / 587s gen, released 325).
# B200 specifics: LD_PRELOAD system libcuda (image ships stale 575 compat lib -> error
# 803), model staged to /tmp (cross-region HDFS-FUSE is ~25MB/s), faiss GPU->CPU fallback
# (conda faiss may lack sm_100).
set -uo pipefail
export PYTHONUNBUFFERED=1

# --- B200 fix 1: bypass the stale compat libcuda (validated by b200_probe2.sh) ---
export LD_PRELOAD=/lib/x86_64-linux-gnu/libcuda.so.1
# --- B200 fix 2: vLLM's vendored flash-attn ships sm_80/90 ONLY (cuobjdump-verified) and
# segfaults in CUDA-graph capture on sm_100; Triton backend JIT-compiles per-arch.
# Training-side flash-attn (standalone wheel) HAS sm_100 and is unaffected.
export VLLM_ATTENTION_BACKEND=TRITON_ATTN

REPO=${REPO:-/home/tiger/xiaoxuan/arlca-8b}
export PYTHONPATH="$REPO"
export VENV=/home/tiger/xiaoxuan/envs/agentic-rl-ca
HDFS_MODEL=/mnt/hdfs/mlsys/models/Qwen3-8B-Base
export DATA_DIR=/mnt/hdfs/mlsys/users/xiaoxuan/agentic_rl_ca/data_asearcher_base
export SEARCHR1_DATA=/mnt/hdfs/mlsys/users/xiaoxuan/searchr1/searchr1_data
export HF_DATASETS_CACHE=/tmp/hf_datasets_cache
PROTOCOL=asearcher_32turn_8b
STAGE=/tmp/searchR1
RETR_ENV=/home/tiger/xiaoxuan/envs/searchr1-retr-conda
MM=/home/tiger/xiaoxuan/tools/bin/micromamba
export MAMBA_ROOT_PREFIX=/home/tiger/xiaoxuan/tools/mamba
HLOG=/mnt/hdfs/mlsys/users/xiaoxuan/agentic_rl_ca/logs/p8b_b200_pilot
mkdir -p "$HLOG" "$REPO/outputs" "$STAGE"
rm -f "$HLOG/DONE" "$HLOG/FAILED"

status=FAILED
finish() { echo "$status $(date -u +%FT%TZ)" > "$HLOG/$status"
  kill "${WATCHDOG_PID:-}" 2>/dev/null || true
  cp "$REPO/outputs/retriever/retriever_b200_pilot.log" "$HLOG/" 2>/dev/null || true
  pkill -9 -f retrieval_server.py 2>/dev/null || true; echo "[worker] exit: $status"; }
trap finish EXIT

eval "$(grep -E '^export (HF_TOKEN|WANDB_API_KEY)=' /home/tiger/.bashrc)" || true
[ -z "${WANDB_API_KEY:-}" ] && export WANDB_MODE=offline
echo "==== B200-PILOT start $(date -u) on $(hostname) ===="
uname -m; nvidia-smi -L | head -2; nvidia-smi | sed -n 3p

# --- preflight ---
[ -f "$DATA_DIR/train.parquet" ] || { echo "PREFLIGHT FAIL: HDFS FUSE"; exit 1; }
[ -d "$HDFS_MODEL" ] || { echo "PREFLIGHT FAIL: model path"; exit 1; }
uv venv -p 3.11 "$VENV" 2>/dev/null; source "$VENV/bin/activate"; export VIRTUAL_ENV="$VENV"
python -c 'import torch; torch.cuda.init(); assert torch.cuda.device_count() >= 8, torch.cuda.device_count(); print("CUDA OK:", torch.cuda.get_device_name(0), "x", torch.cuda.device_count())' \
  || { echo "PREFLIGHT FAIL: CUDA init (LD_PRELOAD fix insufficient?)"; exit 1; }

# --- stage model to local /tmp (one slow cross-region read instead of many) ---
echo "[stage] model -> /tmp ($(date -u))"
mkdir -p /tmp/qwen3-8b-base
time cp "$HDFS_MODEL"/*.safetensors "$HDFS_MODEL"/*.json "$HDFS_MODEL"/*.txt /tmp/qwen3-8b-base/ 2>/dev/null
export MODEL_PATH=/tmp/qwen3-8b-base
[ -f "$MODEL_PATH/config.json" ] || { echo "model staging failed"; exit 1; }

# --- stage retriever index ---
echo "[stage] index -> /tmp ($(date -u))"
[ -f "$STAGE/e5_Flat.index" ] || time cp "$SEARCHR1_DATA/e5_Flat.index" "$STAGE/" || exit 1
[ -f "$STAGE/wiki-18.jsonl" ] || time cp "$SEARCHR1_DATA/wiki-18.jsonl" "$STAGE/" || exit 1

# --- retriever: try GPU faiss, fall back to CPU if unhealthy in 25 min ---
mkdir -p "$REPO/outputs/retriever"
RETR_LOG="$REPO/outputs/retriever/retriever_b200_pilot.log"
CLEAN=(env -u VIRTUAL_ENV -u PYTHONPATH -u PYTHONHOME)
serve_retriever() {  # $1 = "--faiss_gpu" or ""
  ( cd "$REPO" && unset VIRTUAL_ENV PYTHONPATH PYTHONHOME \
    && export PATH="$RETR_ENV/bin:$PATH" LD_LIBRARY_PATH="$RETR_ENV/lib" \
       OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false \
    && exec "$RETR_ENV/bin/python" examples/search/retriever/retrieval_server.py \
      --index_path "$STAGE/e5_Flat.index" --corpus_path "$STAGE/wiki-18.jsonl" \
      --topk 5 --retriever_name e5 --retriever_model intfloat/e5-base-v2 $1 --port 8000 \
  ) >> "$RETR_LOG" 2>&1 &
}
export SEARCH_URL="http://127.0.0.1:8000/retrieve"
health_wait() {  # $1 = tries (10s each); returns 0 when healthy
  for i in $(seq 1 "$1"); do
    pgrep -f retrieval_server.py >/dev/null || return 1
    sleep 10
    CODE=$(curl -s -m 10 -o /dev/null -w '%{http_code}' -X POST "$SEARCH_URL" -H 'Content-Type: application/json' \
        -d '{"queries":["health probe"],"topk":5,"return_scores":true}') || CODE=000
    [ "$CODE" = "200" ] && return 0
  done
  return 1
}
: > "$RETR_LOG"
RETR_MODE=gpu
serve_retriever "--faiss_gpu"
if health_wait 150; then
  echo "[retriever] healthy (GPU faiss)"
else
  echo "[retriever] GPU faiss unhealthy on B200 -> falling back to CPU faiss"
  RETR_MODE=cpu
  pkill -9 -f retrieval_server.py 2>/dev/null; sleep 10
  serve_retriever ""
  health_wait 240 || { echo "[retriever] CPU fallback also unhealthy"; tail -30 "$RETR_LOG"; exit 1; }
  echo "[retriever] healthy (CPU faiss) — NOTE: retrieval latency not H100-comparable"
fi
echo "RETR_MODE=$RETR_MODE" > "$HLOG/RETR_MODE"
( fails=0
  while true; do
    sleep 120
    CODE=$(curl -s -m 30 -o /dev/null -w '%{http_code}' -X POST "$SEARCH_URL" -H 'Content-Type: application/json' \
        -d '{"queries":["health probe"],"topk":1,"return_scores":true}') || CODE=000
    if [ "$CODE" = "200" ]; then fails=0; continue; fi
    fails=$((fails+1))
    if [ "$fails" -ge 3 ]; then
      echo "[watchdog] RESTARTING retriever ($RETR_MODE) $(date -u +%FT%TZ)"
      pkill -9 -f retrieval_server.py 2>/dev/null
      for w in $(seq 1 12); do pgrep -f retrieval_server.py >/dev/null || break; sleep 5; done
      [ "$RETR_MODE" = gpu ] && serve_retriever "--faiss_gpu" || serve_retriever ""
      fails=0
    fi
  done ) &
WATCHDOG_PID=$!

# --- one cycle, fast config, same knobs as H100 arms ---
export EXP_NAME="p8bpilot_b200_grpo_fast_s0"
export DYNBSZ=1 DYNBSZ_TOK=24576
echo "==== B200-PILOT train: 1 cycle $(date -u) ===="
TOTAL_STEPS=1 VAL_FREQ=1000000 SAVE_FREQ=1000000 VAL_BEFORE_TRAIN=False RESUME=disable \
  timeout 4h bash "$REPO/scripts/run_condition.sh" token_grpo 0 "$PROTOCOL" \
  +env.partial_rollout_enable=true +env.partial_rollout_cycle_turns=8 +env.partial_rollout_max_age=4
RC=$?
echo "==== B200-PILOT train exit=$RC $(date -u) ===="
[ "$RC" = "0" ] && status=DONE
echo "==== B200-PILOT complete ($status) $(date -u) ===="
