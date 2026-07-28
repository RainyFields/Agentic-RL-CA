#!/bin/bash
# P8B-PROFILE: profiling pass for the 8B ASearcher-consistent protocol (grilled 2026-07-29).
# ONE worker, four short sequential configs:
#   1. token_grpo   vanilla         (2 steps + val-before-train -> val wall-clock + step-0 compliance)
#   2. turn_ppo_b0  vanilla         (2 steps; actor+critic+ref memory picture)
#   3. token_grpo   partial rollout (4 update-cycles at cycle_turns=8)
#   4. turn_ppo_b0  partial rollout (4 update-cycles)
# plus PoC tests T1/T3-lite (scripted-action resume identity vs the live retriever) BEFORE training.
# Measures: step-time breakdown, OOM headroom at MICRO_BSZ=1, turns histogram + ctx_overflow
# rate + info-cap hit rate (rollout dumps), grammar compliance, retriever health at top-5.
# Retriever hardening inherited from p3 (OMP caps, SIGKILL watchdog, no-mamba restart).
set -uo pipefail
export PYTHONUNBUFFERED=1

REPO=${REPO:-/home/tiger/xiaoxuan/arlca-8b}          # durable worktree on branch sync-partial-rollout-poc
# The venv's editable install resolves verl/agent_system to the MAIN checkout (which runs
# live waves on another branch); PYTHONPATH wins over .pth entries -> branch code loads.
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
HLOG=/mnt/hdfs/mlsys/users/xiaoxuan/agentic_rl_ca/logs/p8b_profile
mkdir -p "$HLOG" "$REPO/outputs" "$STAGE"
rm -f "$HLOG/DONE" "$HLOG/FAILED"

status=FAILED
finish() { echo "$status $(date -u +%FT%TZ)" > "$HLOG/$status"
  kill "${WATCHDOG_PID:-}" 2>/dev/null || true
  cp "$REPO/outputs/retriever/retriever_p8b_profile.log" "$HLOG/" 2>/dev/null || true
  pkill -9 -f retrieval_server.py 2>/dev/null || true; echo "[worker] exit: $status"; }
trap finish EXIT

eval "$(grep -E '^export (HF_TOKEN|WANDB_API_KEY)=' /home/tiger/.bashrc)" || true
[ -z "${WANDB_API_KEY:-}" ] && export WANDB_MODE=offline
echo "==== P8B-PROFILE start $(date -u) on $(hostname) ===="; nvidia-smi -L | head -2 || true

# ---- 0. pod preflight, part 1: filesystem (fail fast -> relaunch draws a fresh pod) ----
[ -f "$DATA_DIR/train.parquet" ] || { echo "PREFLIGHT FAIL: HDFS FUSE"; exit 1; }
[ -d "$MODEL_PATH" ] || { echo "PREFLIGHT FAIL: model path"; exit 1; }
[ -d "$REPO/.git" ] || [ -f "$REPO/.git" ] || { echo "PREFLIGHT FAIL: repo worktree missing"; exit 1; }

# ---- 1. training venv (cached) ----
uv venv -p 3.11 "$VENV" 2>/dev/null; source "$VENV/bin/activate"; export VIRTUAL_ENV="$VENV"
python -c 'import vllm, flash_attn, verl' 2>/dev/null || { echo "venv incomplete"; exit 1; }

# ---- 1b. pod preflight, part 2: NVML via ctypes (no python package needed; tests the
# exact broken-pod mode — CUDA fine but libnvidia-ml.so missing, so Ray sees 0 GPUs) ----
python - <<'EOF' || { echo "PREFLIGHT FAIL: NVML"; exit 1; }
import ctypes
lib = ctypes.CDLL("libnvidia-ml.so.1")
assert lib.nvmlInit_v2() == 0, "nvmlInit failed"
n = ctypes.c_uint()
assert lib.nvmlDeviceGetCount_v2(ctypes.byref(n)) == 0 and n.value >= 8, f"NVML sees {n.value} GPUs"
EOF

# ---- 2. GPU-faiss conda env (idempotent) ----
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

# ---- 3. stage index + serve retriever (thread-capped; direct env python) ----
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
echo "[retriever] serving (top-5); health gate up to 40min..."
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

# ---- 4. PoC tests T1 + T3-lite (scripted actions, live retriever, no GPU model) ----
echo "==== T1/T3-lite: partial-rollout resume identity ===="
( cd "$REPO" && "$VENV/bin/python" tests/test_partial_rollout_resume.py ) 2>&1 | tee "$HLOG/t1_resume_identity.log"
T1_RC=${PIPESTATUS[0]}
echo "T1 exit=$T1_RC"
[ "$T1_RC" = "0" ] || { echo "T1 FAILED — aborting profiling (partial-rollout state carry is broken)"; exit 1; }

# ---- 5. four short profiling configs ----
run_cfg() {  # run_cfg <tag> <cond> <total_steps> <val_before_train> <extra hydra args...>
  local TAG="$1" C="$2" STEPS="$3" VBT="$4"; shift 4
  echo "==== P8B-PROFILE cfg=$TAG start $(date -u) ===="
  export EXP_NAME="p8bprof_${TAG}"
  export DUMP_TRAIN_SAMPLE=1 DUMP_TRAIN_PATH="/tmp/p8b_dump/${TAG}"
  mkdir -p "$DUMP_TRAIN_PATH"; rm -f "$DUMP_TRAIN_PATH/rollout_log.jsonl"
  TOTAL_STEPS="$STEPS" VAL_FREQ=1000000 SAVE_FREQ=1000000 VAL_BEFORE_TRAIN="$VBT" \
    RESUME=disable timeout 6h bash "$REPO/scripts/run_condition.sh" "$C" 0 "$PROTOCOL" "$@"
  local RC=$?
  echo "==== P8B-PROFILE cfg=$TAG exit=$RC $(date -u) ===="
  # clean up ray/training processes so the next config starts on free GPUs (a 6h
  # timeout would otherwise orphan the whole ray cluster)
  pkill -9 -f verl.trainer.main_ppo 2>/dev/null || true
  "$VENV/bin/ray" stop --force >/dev/null 2>&1 || true
  sleep 30
  # persist the per-turn dump (compressed) + free /tmp
  if [ -f "$DUMP_TRAIN_PATH/rollout_log.jsonl" ]; then
    zstd -q -f "$DUMP_TRAIN_PATH/rollout_log.jsonl" -o "$HLOG/rollout_${TAG}.jsonl.zst" && rm -rf "$DUMP_TRAIN_PATH"
  fi
  return $RC
}

overall=0
run_cfg grpo_vanilla token_grpo   2 True                                        || overall=1
run_cfg ppo_vanilla  turn_ppo_b0  2 False                                       || overall=1
run_cfg grpo_partial token_grpo   4 False +env.partial_rollout_enable=true \
        +env.partial_rollout_cycle_turns=8 +env.partial_rollout_max_age=4       || overall=1
run_cfg ppo_partial  turn_ppo_b0  4 False +env.partial_rollout_enable=true \
        +env.partial_rollout_cycle_turns=8 +env.partial_rollout_max_age=4       || overall=1

[ "$overall" = "0" ] && status=DONE
echo "==== P8B-PROFILE complete ($status) $(date -u) ===="
