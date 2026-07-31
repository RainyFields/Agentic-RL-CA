#!/bin/bash
# P8B-FAST: distribution-safe fast config profiling (user spec 2026-07-30).
# GRPO only, comparable to p8bprof_grpo_partial (same data/seed/batch/4 cycles/partial
# args). On top: DYNBSZ=1 (24,576 tok/GPU dynamic micro-batching, actor/logprob/ref) and
# chunked prefill + max_num_batched_tokens=16384 (run_condition.sh). NOT changed:
# template, GPU_MEMORY_UTIL(0.5), remove_padding, balance_batch, grad ckpt, sampling,
# any algorithm setting. OOM ladder: 24576 -> 16384 -> revert DYNBSZ (reported, run
# named p8bprof_grpo_fast_nodynbsz). Own log/marker paths (a second p8b worker may run
# concurrently). Deviation to note in the report: p8bprof_grpo_partial ran at KV 0.6
# (pre-memory-fix); this runs at the protocol's 0.5.
set -uo pipefail
export PYTHONUNBUFFERED=1

REPO=${REPO:-/home/tiger/xiaoxuan/arlca-8b}
export PYTHONPATH="$REPO"
export VENV=/home/tiger/xiaoxuan/envs/agentic-rl-ca
export MODEL_PATH=/mnt/hdfs/mlsys/models/Qwen3-8B-Base
export DATA_DIR=/mnt/hdfs/mlsys/users/xiaoxuan/agentic_rl_ca/data_asearcher_base
export SEARCHR1_DATA=/mnt/hdfs/mlsys/users/xiaoxuan/searchr1/searchr1_data
export HF_DATASETS_CACHE=/tmp/hf_datasets_cache
PROTOCOL=asearcher_32turn_8b
STAGE=/tmp/searchR1
RETR_ENV=/home/tiger/xiaoxuan/envs/searchr1-retr-conda
MM=/home/tiger/xiaoxuan/tools/bin/micromamba
export MAMBA_ROOT_PREFIX=/home/tiger/xiaoxuan/tools/mamba
HLOG=/mnt/hdfs/mlsys/users/xiaoxuan/agentic_rl_ca/logs/p8b_fast
mkdir -p "$HLOG" "$REPO/outputs" "$STAGE"
rm -f "$HLOG/DONE" "$HLOG/FAILED"

status=FAILED
finish() { echo "$status $(date -u +%FT%TZ)" > "$HLOG/$status"
  kill "${WATCHDOG_PID:-}" 2>/dev/null || true
  cp "$REPO/outputs/retriever/retriever_p8b_fast.log" "$HLOG/" 2>/dev/null || true
  pkill -9 -f retrieval_server.py 2>/dev/null || true; echo "[worker] exit: $status"; }
trap finish EXIT

eval "$(grep -E '^export (HF_TOKEN|WANDB_API_KEY)=' /home/tiger/.bashrc)" || true
[ -z "${WANDB_API_KEY:-}" ] && export WANDB_MODE=offline
echo "==== P8B-FAST start $(date -u) on $(hostname) ===="; nvidia-smi -L | head -2 || true

# ---- preflight ----
[ -f "$DATA_DIR/train.parquet" ] || { echo "PREFLIGHT FAIL: HDFS FUSE"; exit 1; }
[ -d "$MODEL_PATH" ] || { echo "PREFLIGHT FAIL: model path"; exit 1; }
[ -d "$REPO/.git" ] || [ -f "$REPO/.git" ] || { echo "PREFLIGHT FAIL: repo worktree"; exit 1; }
uv venv -p 3.11 "$VENV" 2>/dev/null; source "$VENV/bin/activate"; export VIRTUAL_ENV="$VENV"
python -c 'import vllm, flash_attn, verl' 2>/dev/null || { echo "venv incomplete"; exit 1; }
python - <<'EOF' || { echo "PREFLIGHT FAIL: NVML"; exit 1; }
import ctypes
lib = ctypes.CDLL("libnvidia-ml.so.1")
assert lib.nvmlInit_v2() == 0, "nvmlInit failed"
n = ctypes.c_uint()
assert lib.nvmlDeviceGetCount_v2(ctypes.byref(n)) == 0 and n.value >= 8, f"NVML sees {n.value}"
EOF

# ---- faiss env (idempotent, prebuilt) ----
CLEAN=(env -u VIRTUAL_ENV -u PYTHONPATH -u PYTHONHOME)
"${CLEAN[@]}" "$MM" run -p "$RETR_ENV" python -c 'import faiss; assert faiss.get_num_gpus()>0' 2>/dev/null \
  || { echo "faiss env missing/broken (expected prebuilt)"; exit 1; }

# ---- retriever (thread-capped, top-5), own log ----
[ -f "$STAGE/e5_Flat.index" ] || cp "$SEARCHR1_DATA/e5_Flat.index" "$STAGE/" || exit 1
[ -f "$STAGE/wiki-18.jsonl" ] || cp "$SEARCHR1_DATA/wiki-18.jsonl" "$STAGE/" || exit 1
mkdir -p "$REPO/outputs/retriever"
RETR_LOG="$REPO/outputs/retriever/retriever_p8b_fast.log"
serve_retriever() {
  ( cd "$REPO" && unset VIRTUAL_ENV PYTHONPATH PYTHONHOME \
    && export PATH="$RETR_ENV/bin:$PATH" LD_LIBRARY_PATH="$RETR_ENV/lib" \
       OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false \
    && exec "$RETR_ENV/bin/python" examples/search/retriever/retrieval_server.py \
      --index_path "$STAGE/e5_Flat.index" --corpus_path "$STAGE/wiki-18.jsonl" \
      --topk 5 --retriever_name e5 --retriever_model intfloat/e5-base-v2 --faiss_gpu --port 8000 \
  ) >> "$RETR_LOG" 2>&1 &
}
: > "$RETR_LOG"; serve_retriever
export SEARCH_URL="http://127.0.0.1:8000/retrieve"
up=0
for i in $(seq 1 240); do
  pgrep -f retrieval_server.py >/dev/null || { echo "[retriever] DIED"; tail -30 "$RETR_LOG"; exit 1; }
  sleep 10
  CODE=$(curl -s -m 10 -o /tmp/h.json -w '%{http_code}' -X POST "$SEARCH_URL" -H 'Content-Type: application/json' \
      -d '{"queries":["health probe"],"topk":5,"return_scores":true}') || CODE=000
  [ "$CODE" = "200" ] && { echo "[retriever] healthy t=$((i*10))s"; up=1; break; }
done
[ "$up" = 1 ] || { echo "[retriever] health timeout"; exit 1; }
( fails=0
  while true; do
    sleep 120
    CODE=$(curl -s -m 30 -o /dev/null -w '%{http_code}' -X POST "$SEARCH_URL" -H 'Content-Type: application/json' \
        -d '{"queries":["health probe"],"topk":1,"return_scores":true}') || CODE=000
    if [ "$CODE" = "200" ]; then fails=0; continue; fi
    fails=$((fails+1)); echo "[watchdog] retriever health fail $fails/3 (code $CODE)"
    if [ "$fails" -ge 3 ]; then
      echo "[watchdog] RESTARTING retriever $(date -u +%FT%TZ)"
      pkill -9 -f retrieval_server.py 2>/dev/null
      for w in $(seq 1 12); do pgrep -f retrieval_server.py >/dev/null || break; sleep 5; done
      serve_retriever; fails=0
    fi
  done ) &
WATCHDOG_PID=$!

# ---- attempt runner (ladder) ----
PR_ARGS=(+env.partial_rollout_enable=true +env.partial_rollout_cycle_turns=8 +env.partial_rollout_max_age=4)
ALOG=/tmp/p8b_fast_attempt.log
attempt() {  # $1 = EXP_NAME
  export EXP_NAME="$1"
  export DUMP_TRAIN_SAMPLE=1 DUMP_TRAIN_PATH="/tmp/p8b_dump/grpo_fast"
  mkdir -p "$DUMP_TRAIN_PATH"; rm -f "$DUMP_TRAIN_PATH/rollout_log.jsonl"
  echo "==== P8B-FAST attempt EXP=$1 DYNBSZ=${DYNBSZ:-0} DYNBSZ_TOK=${DYNBSZ_TOK:-} $(date -u) ===="
  TOTAL_STEPS=4 VAL_FREQ=1000000 SAVE_FREQ=1000000 VAL_BEFORE_TRAIN=False RESUME=disable \
    timeout 6h bash "$REPO/scripts/run_condition.sh" token_grpo 0 "$PROTOCOL" "${PR_ARGS[@]}" 2>&1 | tee "$ALOG"
  local RC=${PIPESTATUS[0]}
  echo "==== P8B-FAST attempt EXP=$1 exit=$RC $(date -u) ===="
  pkill -9 -f verl.trainer.main_ppo 2>/dev/null || true
  "$VENV/bin/ray" stop --force >/dev/null 2>&1 || true
  sleep 30
  return $RC
}
oomed() { grep -aqE "CUDA out of memory|OutOfMemoryError|torch\.OutOfMemory|cumem.*create_and_map" "$ALOG"; }

export DYNBSZ=1 DYNBSZ_TOK=24576
if attempt p8bprof_grpo_fast; then
  status=DONE
elif oomed; then
  echo "==== P8B-FAST FALLBACK: OOM at 24576 -> DYNBSZ_TOK=16384 ===="
  export DYNBSZ_TOK=20480  # >= max_seq_len 17408 required
  if attempt p8bprof_grpo_fast; then
    status=DONE; echo "FALLBACK_16384_USED $(date -u +%FT%TZ)" > "$HLOG/FALLBACK_16384"
  elif oomed; then
    echo "==== P8B-FAST REVERT: OOM at 16384 -> DYNBSZ off (chunked prefill only) ===="
    export DYNBSZ=0; unset DYNBSZ_TOK
    attempt p8bprof_grpo_fast_nodynbsz && status=DONE
    echo "DYNBSZ_REVERTED $(date -u +%FT%TZ)" > "$HLOG/DYNBSZ_REVERTED"
  fi
fi

# persist the per-turn dump
if [ -f "/tmp/p8b_dump/grpo_fast/rollout_log.jsonl" ]; then
  zstd -q -f "/tmp/p8b_dump/grpo_fast/rollout_log.jsonl" -o "$HLOG/rollout_grpo_fast.jsonl.zst" \
    && rm -rf /tmp/p8b_dump/grpo_fast
fi
echo "==== P8B-FAST complete ($status) $(date -u) ===="
