#!/bin/bash
# P8B ARM: full 8B ASearcher run, FAST config (user decision 2026-07-30):
# partial rollout (cycle 8 / max_age 4) + DYNBSZ 24576 + chunked prefill, 75 steps,
# val@25 (+ step-0 val), ckpt@25 (keep 1), wandb online ca-rung3-8b-asearcher.
# COND=token_grpo | turn_ppo_b0 (critic offload + critic dyn-bsz auto via protocol/run_condition).
# 3 attempts with RESUME=auto (rung-4 arm pattern).
set -uo pipefail
export PYTHONUNBUFFERED=1

COND="${COND:?set COND=token_grpo|turn_ppo_b0}"
REPO=${REPO:-/home/tiger/xiaoxuan/Agentic-RL-CA/.claude/worktrees/sync-partial-rollout}
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
HLOG=/mnt/hdfs/mlsys/users/xiaoxuan/agentic_rl_ca/logs/p8b_async_arm_${COND}
mkdir -p "$HLOG" "$REPO/outputs" "$STAGE"
rm -f "$HLOG/DONE" "$HLOG/FAILED"

status=FAILED
finish() { echo "$status $(date -u +%FT%TZ)" > "$HLOG/$status"
  kill "${WATCHDOG_PID:-}" 2>/dev/null || true
  cp "$REPO/outputs/retriever/retriever_p8b_async_arm_${COND}.log" "$HLOG/" 2>/dev/null || true
  pkill -9 -f retrieval_server.py 2>/dev/null || true; echo "[worker] exit: $status"; }
trap finish EXIT

eval "$(grep -E '^export (HF_TOKEN|WANDB_API_KEY)=' /home/tiger/.bashrc)" || true
[ -z "${WANDB_API_KEY:-}" ] && export WANDB_MODE=offline
echo "==== P8B-ASYNC-ARM $COND start $(date -u) on $(hostname) ===="; nvidia-smi -L | head -2 || true

# ---- preflight ----
[ -f "$DATA_DIR/train.parquet" ] || { echo "PREFLIGHT FAIL: HDFS FUSE"; exit 1; }
[ -d "$MODEL_PATH" ] || { echo "PREFLIGHT FAIL: model path"; exit 1; }
[ -d "$REPO/.git" ] || [ -f "$REPO/.git" ] || { echo "PREFLIGHT FAIL: repo worktree"; exit 1; }
uv venv -p 3.11 "$VENV" 2>/dev/null; source "$VENV/bin/activate"; export VIRTUAL_ENV="$VENV"
python -c 'import vllm, flash_attn, verl' 2>/dev/null || { echo "venv incomplete"; exit 1; }
python - <<'EOF' || { echo "PREFLIGHT FAIL: NVML"; exit 1; }
import ctypes
lib = ctypes.CDLL("libnvidia-ml.so.1")
assert lib.nvmlInit_v2() == 0
n = ctypes.c_uint()
assert lib.nvmlDeviceGetCount_v2(ctypes.byref(n)) == 0 and n.value >= 8, f"NVML sees {n.value}"
EOF
CLEAN=(env -u VIRTUAL_ENV -u PYTHONPATH -u PYTHONHOME)
"${CLEAN[@]}" "$MM" run -p "$RETR_ENV" python -c 'import faiss; assert faiss.get_num_gpus()>0' 2>/dev/null \
  || { echo "faiss env missing/broken"; exit 1; }

# ---- retriever ----
[ -f "$STAGE/e5_Flat.index" ] || cp "$SEARCHR1_DATA/e5_Flat.index" "$STAGE/" || exit 1
[ -f "$STAGE/wiki-18.jsonl" ] || cp "$SEARCHR1_DATA/wiki-18.jsonl" "$STAGE/" || exit 1
mkdir -p "$REPO/outputs/retriever"
RETR_LOG="$REPO/outputs/retriever/retriever_p8b_async_arm_${COND}.log"
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

# ---- the arm: 75 steps, fast config, 3 attempts with resume ----
export EXP_NAME="${COND}_qwen3-8b-base_32turn_async75_s0"
export DYNBSZ=1 DYNBSZ_TOK="${DYNBSZ_TOK:-24576}"
PR_ARGS=(+env.partial_rollout_enable=true +env.partial_rollout_cycle_turns=8 +env.partial_rollout_max_age=4 actor_rollout_ref.rollout.mode=async +env.async_rollout_enable=true)
for attempt in 1 2 3; do
  echo "==== P8B-ASYNC-ARM $COND attempt $attempt $(date -u) ===="
  TOTAL_STEPS=75 VAL_FREQ=25 SAVE_FREQ=25 VAL_BEFORE_TRAIN=True RESUME=auto \
    bash "$REPO/scripts/run_condition.sh" "$COND" 0 "$PROTOCOL" "${PR_ARGS[@]}"
  RC=$?
  echo "==== P8B-ASYNC-ARM $COND attempt $attempt exit=$RC $(date -u) ===="
  if [ "$RC" = "0" ]; then status=DONE; break; fi
  pkill -9 -f verl.trainer.main_ppo 2>/dev/null || true
  pkill -9 -f "ray::" 2>/dev/null || true
  "$VENV/bin/ray" stop --force >/dev/null 2>&1 || true
  # drain until GPUs actually free (OOM-wedged workers can hold memory for minutes;
  # 2026-07-31: a 60s fixed sleep burned two retries on 'Total available GPUs 0')
  for d in $(seq 1 60); do
    MAXUSED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
    [ "${MAXUSED:-99999}" -lt 2000 ] && { echo "[retry] GPUs drained after $((d*20))s"; break; }
    sleep 20
  done
done
echo "==== P8B-ASYNC-ARM $COND complete ($status) $(date -u) ===="
