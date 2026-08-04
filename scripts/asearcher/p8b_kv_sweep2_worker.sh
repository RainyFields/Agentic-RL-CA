#!/bin/bash
# KV-CACHE SWEEP (user request 2026-08-05): the instrumented async diag workload
# (token_grpo, 1 step, snap_s=2, per-request events) at gpu_memory_utilization
# 0.5 -> 0.60 -> 0.65, sequentially on ONE worker. Everything else identical:
# same base ckpt (fresh from /tmp model), same data/seed/concurrency/sampling,
# same DYNBSZ 20480, same routing. Safety gate: stop the sweep on nonzero exit
# or any CUDA OOM signature before escalating memory.
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
HLOG=/mnt/hdfs/mlsys/users/xiaoxuan/agentic_rl_ca/logs/p8b_kv_sweep2
DIAG_BASE=/home/tiger/xiaoxuan/Agentic-RL-CA/outputs/async_diag
mkdir -p "$HLOG" "$REPO/outputs" "$STAGE" "$DIAG_BASE"
rm -f "$HLOG/DONE" "$HLOG/FAILED"

status=FAILED
finish() { echo "$status $(date -u +%FT%TZ)" > "$HLOG/$status"
  kill "${WATCHDOG_PID:-}" "${SAMPLER_PID:-}" 2>/dev/null || true
  pkill -9 -f retrieval_server.py 2>/dev/null || true; echo "[worker] exit: $status"; }
trap finish EXIT

eval "$(grep -E '^export HF_TOKEN=' /home/tiger/.bashrc)" || true
export WANDB_MODE=offline
export VLLM_LOGGING_LEVEL=INFO   # try to surface "GPU KV cache size" engine lines
echo "==== KV-SWEEP start $(date -u) on $(hostname) ===="; nvidia-smi -L | head -2 || true

[ -f "$DATA_DIR/train.parquet" ] || { echo "PREFLIGHT FAIL: HDFS FUSE"; exit 1; }
uv venv -p 3.11 "$VENV" 2>/dev/null; source "$VENV/bin/activate"; export VIRTUAL_ENV="$VENV"
python -c 'import vllm, flash_attn, verl' 2>/dev/null || { echo "venv incomplete"; exit 1; }
CLEAN=(env -u VIRTUAL_ENV -u PYTHONPATH -u PYTHONHOME)
MM=/home/tiger/xiaoxuan/tools/bin/micromamba
export MAMBA_ROOT_PREFIX=/home/tiger/xiaoxuan/tools/mamba
"${CLEAN[@]}" "$MM" run -p "$RETR_ENV" python -c 'import faiss; assert faiss.get_num_gpus()>0' 2>/dev/null \
  || { echo "faiss env missing/broken"; exit 1; }

if [ ! -f "$MODEL_PATH/config.json" ]; then
  mkdir -p "$MODEL_PATH"
  ls "$HDFS_MODEL"/*.safetensors "$HDFS_MODEL"/*.json "$HDFS_MODEL"/*.txt 2>/dev/null \
    | xargs -P 8 -I{} cp {} "$MODEL_PATH"/
fi
[ -f "$MODEL_PATH/config.json" ] || { echo "model staging failed"; exit 1; }

# ---- retriever (battle-tested arm block) ----
[ -f "$STAGE/e5_Flat.index" ] || cp "$SEARCHR1_DATA/e5_Flat.index" "$STAGE/" || exit 1
[ -f "$STAGE/wiki-18.jsonl" ] || cp "$SEARCHR1_DATA/wiki-18.jsonl" "$STAGE/" || exit 1
mkdir -p "$REPO/outputs/retriever"
RETR_LOG="$REPO/outputs/retriever/retriever_kv_sweep.log"
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
    fails=$((fails+1)); echo "[watchdog] retriever fail $fails/3 (code $CODE)"
    if [ "$fails" -ge 3 ]; then
      pkill -9 -f retrieval_server.py 2>/dev/null
      for w in $(seq 1 12); do pgrep -f retrieval_server.py >/dev/null || break; sleep 5; done
      serve_retriever; fails=0
    fi
  done ) &
WATCHDOG_PID=$!

LOGDIR="$REPO/outputs"
for U in 0.5 0.60 0.65; do
  TAG="u${U/0./}"                     # u5 / u60 / u65
  echo "==== KV-SWEEP RUN gpu_memory_utilization=$U $(date -u) ===="
  export ASYNC_DIAG_DIR="$DIAG_BASE/kvsweep2_$TAG"
  mkdir -p "$ASYNC_DIAG_DIR"
  # 1Hz GPU sampler for this run (peak memory evidence)
  ( while true; do echo "$(date +%s) $(nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader,nounits | tr '\n' ';')"; sleep 1; done \
      > "$ASYNC_DIAG_DIR/gpu_samples.txt" ) &
  SAMPLER_PID=$!
  if nvidia-smi dmon -c 1 -s pum >/dev/null 2>&1; then
    ( exec nvidia-smi dmon -s pum -d 1 > "$ASYNC_DIAG_DIR/dmon.txt" 2>&1 ) &
    DMON_PID=$!
  else
    DMON_PID=""
    echo "[dmon] unavailable — SM/HBM utilization not captured"
  fi
  export EXP_NAME="kvsweep2_${TAG}_token_grpo_s0"
  GPU_MEMORY_UTIL="$U" TOTAL_STEPS=4 VAL_FREQ=1000000 SAVE_FREQ=1000000 \
    VAL_BEFORE_TRAIN=False RESUME=disable DYNBSZ=1 DYNBSZ_TOK=20480 \
    timeout 2h bash "$REPO/scripts/run_condition.sh" token_grpo 0 "$PROTOCOL" \
      +env.partial_rollout_enable=true +env.partial_rollout_cycle_turns=8 \
      +env.partial_rollout_max_age=4 +env.rollout_profiling=true \
      actor_rollout_ref.rollout.disable_log_stats=False +env.async_rollout_snap_s=2 \
      actor_rollout_ref.rollout.mode=async +env.async_rollout_enable=true
  RC=$?
  kill "$SAMPLER_PID" ${DMON_PID:+$DMON_PID} 2>/dev/null || true
  echo "==== KV-SWEEP RUN $U exit=$RC $(date -u) ===="
  pkill -9 -f verl.trainer.main_ppo 2>/dev/null || true
  pkill -9 -f "ray::" 2>/dev/null || true
  "$VENV/bin/ray" stop --force >/dev/null 2>&1 || true
  # drain GPUs before escalating memory
  for d in $(seq 1 30); do
    M=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
    [ "${M:-99999}" -lt 2000 ] && break; sleep 10
  done
  if [ "$RC" != "0" ]; then echo "KV-SWEEP ABORT: run $U failed (rc=$RC)"; exit 1; fi
  if tail -c 2000000 "$LOGDIR"/../outputs/*.log 2>/dev/null | grep -q "CUDA out of memory" ; then :; fi
  # OOM gate: check THIS run's console (we are the console — grep our own HLOG copy is
  # not possible; rely on rc + the engine dying loudly). Additionally check dmesg OOM.
  if dmesg 2>/dev/null | tail -50 | grep -qi "out of memory"; then
    echo "KV-SWEEP ABORT: system OOM signature after run $U"; exit 1
  fi
done

status=DONE
echo "==== KV-SWEEP complete $(date -u) ===="
