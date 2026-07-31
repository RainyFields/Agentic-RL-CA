#!/bin/bash
# P8B-PROF: profiling for the 8B ASearcher-consistent runs (Qwen3-8B-Base, unfiltered
# Base-35k, 32-turn/16k/1024, top-5 + 5k block cap, full-response history, overflow
# terminate). 3 train steps per arm (token_grpo with step-0 val, then turn_ppo_b0) at the
# TARGET batch config — measures step-time breakdown, memory headroom, turns histogram,
# length-termination + block-cap hit rates, base-model grammar compliance. No checkpoints.
# Runs from the asearcher-8b-32turn WORKTREE so the gated env changes apply without
# touching the main checkout (p3v2 + rung-4 workers auto-resume from there).
set -uo pipefail
export PYTHONUNBUFFERED=1

REPO=/home/tiger/xiaoxuan/Agentic-RL-CA/.claude/worktrees/asearcher-8b-32turn
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
HLOG=/mnt/hdfs/mlsys/users/xiaoxuan/agentic_rl_ca/logs/p8b_profile
mkdir -p "$HLOG" "$REPO/outputs" "$STAGE" "$(dirname "$MM")"
rm -f "$HLOG/DONE" "$HLOG/FAILED"

status=FAILED
finish() { echo "$status $(date -u +%FT%TZ)" > "$HLOG/$status"
  kill "${WATCHDOG_PID:-}" 2>/dev/null || true
  cp "$REPO/outputs/retriever/retriever_p8b_profile.log" "$HLOG/" 2>/dev/null || true
  for c in token_grpo turn_ppo_b0; do
    d=/tmp/p8b_prof_dump/$c
    [ -f "$d/rollout_log.jsonl" ] && zstd -q -f "$d/rollout_log.jsonl" -o "$HLOG/rollout_log_${c}.jsonl.zst"
  done
  pkill -9 -f retrieval_server.py 2>/dev/null || true; echo "[worker] exit: $status"; }
trap finish EXIT

# ---- 0. pod preflight (fail fast -> relaunch draws a fresh pod) ----
nvidia-smi -L | grep -q GPU || { echo "PREFLIGHT: no GPUs"; exit 3; }
"$VENV/bin/python" -c 'import pynvml; pynvml.nvmlInit(); assert pynvml.nvmlDeviceGetCount()==8' \
  || { echo "PREFLIGHT: NVML broken (Ray would see 0 GPUs)"; exit 3; }
[ -f /mnt/hdfs/mlsys/models/Qwen3-8B-Base/config.json ] || { echo "PREFLIGHT: HDFS mount dead"; exit 3; }
[ -f "$DATA_DIR/train.parquet" ] || { echo "PREFLIGHT: data missing"; exit 3; }

eval "$(grep -E '^export (HF_TOKEN|WANDB_API_KEY)=' /home/tiger/.bashrc)" || true
export WANDB_PROJECT=ca-rung3-8b-asearcher
[ -z "${WANDB_API_KEY:-}" ] && export WANDB_MODE=offline
echo "==== P8B-PROF start $(date -u) on $(hostname) ===="; nvidia-smi -L | head -2 || true

# ---- 1. training venv (cached on /home) ----
uv venv -p 3.11 "$VENV" 2>/dev/null; source "$VENV/bin/activate"; export VIRTUAL_ENV="$VENV"
python -c 'import vllm, flash_attn, verl, agent_system' 2>/dev/null || { echo "venv incomplete"; exit 1; }

# ---- 2. GPU-faiss conda env (idempotent; persists on /home) ----
CLEAN=(env -u VIRTUAL_ENV -u PYTHONPATH -u PYTHONHOME)
if [ ! -x "$MM" ]; then
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -C "$(dirname "$(dirname "$MM")")" -xj bin/micromamba || exit 1
fi
if ! "${CLEAN[@]}" "$MM" run -p "$RETR_ENV" python -c 'import faiss,torch,transformers; assert faiss.get_num_gpus()>0' 2>/dev/null; then
  echo "[conda] building faiss-gpu env..."
  rm -rf "$RETR_ENV"
  "${CLEAN[@]}" "$MM" create -y -p "$RETR_ENV" -c conda-forge python=3.11 \
      "mkl=2024.*" "faiss-gpu=1.10.0" "pytorch=*=cpu*" transformers fastapi uvicorn datasets numpy 2>&1 | tail -4 || exit 1
fi

# ---- 3. stage index + serve GPU-faiss retriever (thread-capped; direct env python) ----
[ -f "$STAGE/e5_Flat.index" ] || cp "$SEARCHR1_DATA/e5_Flat.index" "$STAGE/" || exit 1
[ -f "$STAGE/wiki-18.jsonl" ] || cp "$SEARCHR1_DATA/wiki-18.jsonl" "$STAGE/" || exit 1
mkdir -p "$REPO/outputs/retriever"
RETR_LOG="$REPO/outputs/retriever/retriever_p8b_profile.log"
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
echo "[retriever] serving (thread-capped); health gate up to 40min..."
up=0
for i in $(seq 1 240); do
  pgrep -f retrieval_server.py >/dev/null || { echo "[retriever] DIED"; tail -30 "$RETR_LOG"; exit 1; }
  sleep 10
  CODE=$(curl -s -m 10 -o /tmp/h.json -w '%{http_code}' -X POST "$SEARCH_URL" -H 'Content-Type: application/json' \
      -d '{"queries":["who won the first nobel prize in physics"],"topk":5,"return_scores":true}') || CODE=000
  [ "$CODE" = "200" ] && { echo "[retriever] healthy t=$((i*10))s"; up=1; break; }
done
[ "$up" = 1 ] || { echo "[retriever] health timeout"; exit 1; }

# ---- 3b. watchdog: SIGKILL + wait-dead + relaunch on 3 consecutive probe failures ----
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

# ---- 4. profiling: 3 steps per arm at TARGET config (256x5, 16k/1024, 32 turns) ----
overall=0
for COND in token_grpo turn_ppo_b0; do
  export EXP_NAME="prof_${COND}_qwen3-8b-base_${PROTOCOL}_s0"
  export DUMP_TRAIN_SAMPLE=1
  export DUMP_TRAIN_PATH=/tmp/p8b_prof_dump/${COND}
  mkdir -p "$DUMP_TRAIN_PATH"; rm -f "$DUMP_TRAIN_PATH/rollout_log.jsonl"
  VBT=False; [ "$COND" = token_grpo ] && VBT=True   # one step-0 greedy val (both arms share the init policy)
  echo "==== P8B-PROF $COND: 3 steps $(date -u) ===="
  VAL_BEFORE_TRAIN=$VBT RESUME=disable bash "$REPO/scripts/run_condition.sh" "$COND" 0 "$PROTOCOL" \
      trainer.total_training_steps=3 trainer.test_freq=1000000 trainer.save_freq=1000000
  RC=$?
  echo "==== P8B-PROF $COND exit=$RC ===="
  [ "$RC" = "0" ] || overall=1
  pkill -9 -f 'ray::' 2>/dev/null; sleep 20   # clean GPU state between arms
  ray stop --force 2>/dev/null; sleep 10
done
[ "$overall" = "0" ] && status=DONE
echo "==== P8B-PROF complete ($status) $(date -u) ===="
