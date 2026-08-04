#!/bin/bash
# LEAN ASYNC ROLLOUT — B200 matched profile (user request 2026-08-05).
# One 8xB200 worker, sequential phases, identical training config to the H100 profile
# (except the B200's own fastest-stable knobs: DYNBSZ 24576, critic on-GPU):
#   phase A: AsyncLLM engine smoke under the B200 env (TRITON_ATTN etc.)
#   phase B: sync GRPO baseline, 3 steps (+rollout_profiling)  [also fills the missing
#            old-engine B200 GRPO leg from the killed 2026-08-04 profiling run]
#   phase C: async GRPO, 3 steps
#   phase D: async turn-PPO, 2 steps
# B200 pod fixes inherited from p8b_b200_prof_worker.sh (libcuda resolve, TRITON_ATTN,
# no egress, torch flat retriever — no faiss supports sm_100).
set -uo pipefail
export PYTHONUNBUFFERED=1

# pick the first real (>1MB) non-compat libcuda
LIBCUDA=""
for c in $(ldconfig -p 2>/dev/null | awk '/libcuda\.so\.1/{print $NF}' | grep -v compat) \
         /lib/x86_64-linux-gnu/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so.1; do
  if [ -f "$c" ] && [ "$(stat -Lc %s "$c" 2>/dev/null || echo 0)" -gt 1000000 ]; then LIBCUDA="$c"; break; fi
done
if [ -n "$LIBCUDA" ]; then export LD_PRELOAD="$LIBCUDA"; echo "[libcuda] preloading $LIBCUDA"; else unset LD_PRELOAD; echo "[libcuda] no real system libcuda found"; fi
export VLLM_ATTENTION_BACKEND=TRITON_ATTN
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline

REPO=${REPO:-/home/tiger/xiaoxuan/Agentic-RL-CA/.claude/worktrees/sync-partial-rollout}
export PYTHONPATH="$REPO"
export VENV=/home/tiger/xiaoxuan/envs/agentic-rl-ca
HDFS_MODEL=/mnt/hdfs/mlsys/models/Qwen3-8B-Base
export DATA_DIR=/mnt/hdfs/mlsys/users/xiaoxuan/agentic_rl_ca/data_asearcher_base
export SEARCHR1_DATA=/mnt/hdfs/mlsys/users/xiaoxuan/searchr1/searchr1_data
export HF_DATASETS_CACHE=/tmp/hf_datasets_cache
PROTOCOL=asearcher_32turn_8b
STAGE=/tmp/searchR1
RETR_ENV=/home/tiger/xiaoxuan/envs/searchr1-retr-conda
HLOG=/mnt/hdfs/mlsys/users/xiaoxuan/agentic_rl_ca/logs/p8b_b200_async_profile
N_STEPS="${N_STEPS:-3}"
TURNPPO_STEPS="${TURNPPO_STEPS:-2}"
mkdir -p "$HLOG" "$REPO/outputs" "$STAGE"
rm -f "$HLOG/DONE" "$HLOG/FAILED"

status=FAILED
finish() { echo "$status $(date -u +%FT%TZ)" > "$HLOG/$status"
  kill "${WATCHDOG_PID:-}" 2>/dev/null || true
  cp "$REPO/outputs/retriever/retriever_b200_async_prof.log" "$HLOG/" 2>/dev/null || true
  pkill -9 -f torch_retrieval_server.py 2>/dev/null || true; echo "[worker] exit: $status"; }
trap finish EXIT

eval "$(grep -E '^export HF_TOKEN=' /home/tiger/.bashrc)" || true
echo "==== B200-ASYNC-PROF start $(date -u) on $(hostname) ===="
uname -m; nvidia-smi -L | head -2; nvidia-smi | sed -n 3p

[ -f "$DATA_DIR/train.parquet" ] || { echo "PREFLIGHT FAIL: HDFS FUSE"; exit 1; }
uv venv -p 3.11 "$VENV" 2>/dev/null; source "$VENV/bin/activate"; export VIRTUAL_ENV="$VENV"
python -c 'import torch; torch.cuda.init(); assert torch.cuda.device_count() >= 8' \
  || { echo "PREFLIGHT FAIL: CUDA init"; exit 1; }

echo "[stage] model -> /tmp ($(date -u))"
mkdir -p /tmp/qwen3-8b-base
ls "$HDFS_MODEL"/*.safetensors "$HDFS_MODEL"/*.json "$HDFS_MODEL"/*.txt 2>/dev/null \
  | xargs -P 8 -I{} cp {} /tmp/qwen3-8b-base/
export MODEL_PATH=/tmp/qwen3-8b-base
[ -f "$MODEL_PATH/config.json" ] || { echo "model staging failed"; exit 1; }

# ---- phase A: engine smoke under the B200 env ----
echo "==== PHASE A: AsyncLLM engine smoke (B200) $(date -u) ===="
TP=1 GPU_UTIL=0.5 python "$REPO/scripts/asearcher/async_llm_smoke.py" 2>&1 | tail -12
SMOKE_RC=${PIPESTATUS[0]}
echo "==== PHASE A exit=$SMOKE_RC ===="
[ "$SMOKE_RC" = "0" ] || { echo "B200 ENGINE SMOKE FAILED"; exit 1; }

echo "[stage] index -> /tmp ($(date -u))"
[ -f "$STAGE/e5_Flat.index" ] || time cp "$SEARCHR1_DATA/e5_Flat.index" "$STAGE/" || exit 1
[ -f "$STAGE/wiki-18.jsonl" ] || time cp "$SEARCHR1_DATA/wiki-18.jsonl" "$STAGE/" || exit 1
if [ ! -f "$STAGE/e5_flat_fp16.npy" ]; then
  echo "[extract] faiss index -> fp16 npy ($(date -u))"
  CLEAN=(env -u VIRTUAL_ENV -u PYTHONPATH -u PYTHONHOME -u LD_PRELOAD)
  time "${CLEAN[@]}" "$RETR_ENV/bin/python" "$REPO/scripts/asearcher/extract_flat_index.py" \
    "$STAGE/e5_Flat.index" "$STAGE/e5_flat_fp16.npy" "$STAGE/e5_flat_meta.json" || exit 1
fi

# ---- torch retrieval server (main venv, embedding shards over 8 GPUs) ----
mkdir -p "$REPO/outputs/retriever"
RETR_LOG="$REPO/outputs/retriever/retriever_b200_async_prof.log"
serve_retriever() {
  ( export TOKENIZERS_PARALLELISM=false \
    && exec python "$REPO/scripts/asearcher/torch_retrieval_server.py" \
      --emb_npy "$STAGE/e5_flat_fp16.npy" --emb_meta "$STAGE/e5_flat_meta.json" \
      --corpus_path "$STAGE/wiki-18.jsonl" --topk 5 --port 8000 \
  ) >> "$RETR_LOG" 2>&1 &
}
: > "$RETR_LOG"; serve_retriever
export SEARCH_URL="http://127.0.0.1:8000/retrieve"
up=0
for i in $(seq 1 120); do
  pgrep -f torch_retrieval_server.py >/dev/null || { echo "[retriever] DIED"; tail -30 "$RETR_LOG"; exit 1; }
  sleep 10
  CODE=$(curl -s -m 10 -o /tmp/h.json -w '%{http_code}' -X POST "$SEARCH_URL" -H 'Content-Type: application/json' \
      -d '{"queries":["health probe"],"topk":5,"return_scores":true}') || CODE=000
  [ "$CODE" = "200" ] && { echo "[retriever] healthy (torch flat) t=$((i*10))s"; up=1; break; }
done
[ "$up" = 1 ] || { echo "[retriever] health timeout"; tail -30 "$RETR_LOG"; exit 1; }
( fails=0
  while true; do
    sleep 120
    CODE=$(curl -s -m 30 -o /dev/null -w '%{http_code}' -X POST "$SEARCH_URL" -H 'Content-Type: application/json' \
        -d '{"queries":["health probe"],"topk":1,"return_scores":true}') || CODE=000
    if [ "$CODE" = "200" ]; then fails=0; continue; fi
    fails=$((fails+1)); echo "[watchdog] torch retriever fail $fails/3 (code $CODE)"
    if [ "$fails" -ge 3 ]; then
      echo "[watchdog] RESTARTING torch retriever $(date -u +%FT%TZ)"
      pkill -9 -f torch_retrieval_server.py 2>/dev/null
      for w in $(seq 1 12); do pgrep -f torch_retrieval_server.py >/dev/null || break; sleep 5; done
      serve_retriever; fails=0
    fi
  done ) &
WATCHDOG_PID=$!

run_phase() { # $1=tag $2=cond $3=steps, rest = extra hydra args
  local TAG="$1" COND="$2" STEPS="$3"; shift 3
  echo "==== PHASE $TAG: $COND $STEPS steps $(date -u) ===="
  export EXP_NAME="b200asyncprof_${TAG}_${COND}_s0"
  TOTAL_STEPS="$STEPS" VAL_FREQ=1000000 SAVE_FREQ=1000000 VAL_BEFORE_TRAIN=False RESUME=disable \
    DYNBSZ=1 DYNBSZ_TOK=24576 CRITIC_PARAM_OFFLOAD=False CRITIC_OPTIM_OFFLOAD=False \
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

run_phase B_sync token_grpo "$N_STEPS" || { echo "B200 SYNC BASELINE FAILED"; exit 1; }
run_phase C_async token_grpo "$N_STEPS" "${ASYNC_ARGS[@]}" || { echo "B200 ASYNC RUN FAILED"; exit 1; }
if [ "$TURNPPO_STEPS" -gt 0 ]; then
  run_phase D_tppo turn_ppo_b0 "$TURNPPO_STEPS" "${ASYNC_ARGS[@]}" || echo "B200 TURNPPO ASYNC FAILED (non-fatal)"
fi

status=DONE
echo "==== B200-ASYNC-PROF complete $(date -u) ===="
