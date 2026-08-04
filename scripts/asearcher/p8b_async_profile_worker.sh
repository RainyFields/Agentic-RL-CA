#!/bin/bash
# LEAN ASYNC ROLLOUT — Phase 2 matched profile + smoke (2026-08-04 plan).
# On ONE 8xH100 worker, back to back, IDENTICAL config except the collector:
#   phase A: engine smoke      — bare AsyncLLM capability test (async_llm_smoke.py)
#   phase B: sync baseline     — token_grpo, N steps, rollout_profiling (zombie tokens)
#   phase C: async collector   — token_grpo, N steps, mode=async + async_rollout_enable
#   phase D: (optional) turn_ppo_b0 async smoke, TURNPPO_STEPS>0 to enable
# Code runs from the DEV WORKTREE (not the frozen arm checkout).
set -uo pipefail
export PYTHONUNBUFFERED=1

REPO=${REPO:-/home/tiger/xiaoxuan/Agentic-RL-CA/.claude/worktrees/sync-partial-rollout}
export PYTHONPATH="$REPO"
export VENV=/home/tiger/xiaoxuan/envs/agentic-rl-ca
export MODEL_PATH=/tmp/qwen3-8b-base
HDFS_MODEL=/mnt/hdfs/mlsys/models/Qwen3-8B-Base
export DATA_DIR=/mnt/hdfs/mlsys/users/xiaoxuan/agentic_rl_ca/data_asearcher_base
export SEARCHR1_DATA=/mnt/hdfs/mlsys/users/xiaoxuan/searchr1/searchr1_data
export HF_DATASETS_CACHE=/tmp/hf_datasets_cache
PROTOCOL=asearcher_32turn_8b
STAGE=/tmp/searchR1
RETR_ENV=/home/tiger/xiaoxuan/envs/searchr1-retr-conda
MM=/home/tiger/xiaoxuan/tools/bin/micromamba
export MAMBA_ROOT_PREFIX=/home/tiger/xiaoxuan/tools/mamba
HLOG=/mnt/hdfs/mlsys/users/xiaoxuan/agentic_rl_ca/logs/p8b_async_profile
N_STEPS="${N_STEPS:-3}"
TURNPPO_STEPS="${TURNPPO_STEPS:-0}"
mkdir -p "$HLOG" "$REPO/outputs" "$STAGE"
rm -f "$HLOG/DONE" "$HLOG/FAILED"

status=FAILED
finish() { echo "$status $(date -u +%FT%TZ)" > "$HLOG/$status"
  kill "${WATCHDOG_PID:-}" 2>/dev/null || true
  cp "$REPO/outputs/retriever/retriever_async_profile.log" "$HLOG/" 2>/dev/null || true
  pkill -9 -f retrieval_server.py 2>/dev/null || true; echo "[worker] exit: $status"; }
trap finish EXIT

eval "$(grep -E '^export (HF_TOKEN|WANDB_API_KEY)=' /home/tiger/.bashrc)" || true
export WANDB_MODE=offline   # profile runs: keep wandb out of the picture
echo "==== ASYNC-PROFILE start $(date -u) on $(hostname) ===="; nvidia-smi -L | head -2 || true

# ---- preflight ----
[ -f "$DATA_DIR/train.parquet" ] || { echo "PREFLIGHT FAIL: HDFS FUSE"; exit 1; }
uv venv -p 3.11 "$VENV" 2>/dev/null; source "$VENV/bin/activate"; export VIRTUAL_ENV="$VENV"
python -c 'import vllm, flash_attn, verl' 2>/dev/null || { echo "venv incomplete"; exit 1; }
CLEAN=(env -u VIRTUAL_ENV -u PYTHONPATH -u PYTHONHOME)
"${CLEAN[@]}" "$MM" run -p "$RETR_ENV" python -c 'import faiss; assert faiss.get_num_gpus()>0' 2>/dev/null \
  || { echo "faiss env missing/broken"; exit 1; }

# ---- model to /tmp (idempotent; keepalive worker may have staged it already) ----
if [ ! -f "$MODEL_PATH/config.json" ]; then
  mkdir -p "$MODEL_PATH"
  ls "$HDFS_MODEL"/*.safetensors "$HDFS_MODEL"/*.json "$HDFS_MODEL"/*.txt 2>/dev/null \
    | xargs -P 8 -I{} cp {} "$MODEL_PATH"/
fi
[ -f "$MODEL_PATH/config.json" ] || { echo "model staging failed"; exit 1; }

# ---- phase A: engine smoke (START_PHASE=B/C skips) ----
START_PHASE="${START_PHASE:-A}"
if [ "$START_PHASE" = "A" ]; then
  echo "==== PHASE A: AsyncLLM engine smoke $(date -u) ===="
  TP=1 GPU_UTIL=0.5 python "$REPO/scripts/asearcher/async_llm_smoke.py" 2>&1 | tail -20
  SMOKE_RC=${PIPESTATUS[0]}
  echo "==== PHASE A exit=$SMOKE_RC ===="
  [ "$SMOKE_RC" = "0" ] || { echo "ENGINE SMOKE FAILED — aborting (sync-P1 fallback path)"; exit 1; }
fi

# ---- retriever (verbatim from p8b_arm_worker.sh — battle-tested block) ----
# Reuse a healthy retriever from a previous attempt (phase-C-only reruns).
if [ "$(curl -s -m 10 -o /dev/null -w '%{http_code}' -X POST http://127.0.0.1:8000/retrieve \
      -H 'Content-Type: application/json' \
      -d '{"queries":["health probe"],"topk":1,"return_scores":true}' 2>/dev/null)" = "200" ]; then
  echo "[retriever] already healthy — reusing"
  export SEARCH_URL="http://127.0.0.1:8000/retrieve"
  RETRIEVER_REUSED=1
fi
[ -f "$STAGE/e5_Flat.index" ] || cp "$SEARCHR1_DATA/e5_Flat.index" "$STAGE/" || exit 1
[ -f "$STAGE/wiki-18.jsonl" ] || cp "$SEARCHR1_DATA/wiki-18.jsonl" "$STAGE/" || exit 1
mkdir -p "$REPO/outputs/retriever"
RETR_LOG="$REPO/outputs/retriever/retriever_async_profile.log"
serve_retriever() {
  ( cd "$REPO" && unset VIRTUAL_ENV PYTHONPATH PYTHONHOME \
    && export PATH="$RETR_ENV/bin:$PATH" LD_LIBRARY_PATH="$RETR_ENV/lib" \
       OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false \
    && exec "$RETR_ENV/bin/python" examples/search/retriever/retrieval_server.py \
      --index_path "$STAGE/e5_Flat.index" --corpus_path "$STAGE/wiki-18.jsonl" \
      --topk 5 --retriever_name e5 --retriever_model intfloat/e5-base-v2 --faiss_gpu --port 8000 \
  ) >> "$RETR_LOG" 2>&1 &
}
if [ "${RETRIEVER_REUSED:-0}" = "1" ]; then
  up=1
else
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
fi
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

run_phase() { # $1=tag $2=cond $3=steps, rest = extra hydra args
  local TAG="$1" COND="$2" STEPS="$3"; shift 3
  echo "==== PHASE $TAG: $COND $STEPS steps $(date -u) ===="
  export EXP_NAME="asyncprof_${TAG}_${COND}_s0"
  TOTAL_STEPS="$STEPS" VAL_FREQ=1000000 SAVE_FREQ=1000000 VAL_BEFORE_TRAIN=False \
    DYNBSZ=1 DYNBSZ_TOK=20480 \
    timeout 4h bash "$REPO/scripts/run_condition.sh" "$COND" 0 "$PROTOCOL" \
      +env.partial_rollout_enable=true +env.partial_rollout_cycle_turns=8 \
      +env.partial_rollout_max_age=4 +env.rollout_profiling=true "$@"
  local RC=$?
  echo "==== PHASE $TAG exit=$RC $(date -u) ===="
  pkill -9 -f verl.trainer.main_ppo 2>/dev/null || true
  pkill -9 -f "ray::" 2>/dev/null || true
  "$VENV/bin/ray" stop --force >/dev/null 2>&1 || true
  sleep 30
  return $RC
}

ASYNC_ARGS=(actor_rollout_ref.rollout.mode=async +env.async_rollout_enable=true)

# ---- phase B: sync baseline (START_PHASE=C skips) ----
if [ "$START_PHASE" != "C" ]; then
  run_phase B_sync token_grpo "$N_STEPS" || { echo "SYNC BASELINE FAILED"; exit 1; }
fi

# ---- phase C: async collector (START_PHASE=D skips) ----
if [ "$START_PHASE" != "D" ]; then
  run_phase C_async token_grpo "$N_STEPS" "${ASYNC_ARGS[@]}" || { echo "ASYNC RUN FAILED"; exit 1; }
fi

# ---- phase D: optional turn-PPO async smoke ----
if [ "$TURNPPO_STEPS" -gt 0 ]; then
  run_phase D_tppo turn_ppo_b0 "$TURNPPO_STEPS" "${ASYNC_ARGS[@]}" || echo "TURNPPO ASYNC FAILED (non-fatal)"
fi

status=DONE
echo "==== ASYNC-PROFILE complete $(date -u) ===="
