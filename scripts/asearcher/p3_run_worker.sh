#!/bin/bash
# P3: the P2 comparison RELAUNCHED on stage-1+2 FILTERED ASearcher-Base (band 0<pass<8 of the
# closed-book keep-set; 7438 train / 512 val). Identical protocol to P2 (150 steps, full batch,
# val 256 @ 25) for comparability. Retriever hardening from the diag campaign baked in:
# BLAS/OMP thread caps (silent-livelock fix), watchdog with SIGKILL restart, no-mamba restart path.
set -uo pipefail
export PYTHONUNBUFFERED=1

REPO=/home/tiger/xiaoxuan/Agentic-RL-CA
export VENV=/home/tiger/xiaoxuan/envs/agentic-rl-ca
export MODEL_PATH=/mnt/hdfs/mlsys/models/Qwen3-4B-Instruct-2507
export DATA_DIR=/mnt/hdfs/mlsys/users/xiaoxuan/agentic_rl_ca/data_asearcher_base_filtered_x6
export SEARCHR1_DATA=/mnt/hdfs/mlsys/users/xiaoxuan/searchr1/searchr1_data
export HF_DATASETS_CACHE=/tmp/hf_datasets_cache
PROTOCOL=asearcher_8turn_4b
COND="${COND:?set COND=token_grpo|turn_ppo_b0}"
STAGE=/tmp/searchR1
RETR_ENV=/home/tiger/xiaoxuan/envs/searchr1-retr-conda
MM=/home/tiger/xiaoxuan/tools/bin/micromamba
export MAMBA_ROOT_PREFIX=/home/tiger/xiaoxuan/tools/mamba
HLOG=/mnt/hdfs/mlsys/users/xiaoxuan/agentic_rl_ca/logs/p3_asearcher_${COND}
mkdir -p "$HLOG" "$REPO/outputs" "$STAGE" "$(dirname "$MM")"
rm -f "$HLOG/DONE" "$HLOG/FAILED"

status=FAILED
finish() { echo "$status $(date -u +%FT%TZ)" > "$HLOG/$status"
  kill "${WATCHDOG_PID:-}" 2>/dev/null || true
  cp "$REPO/outputs/retriever/retriever_p3_${COND}.log" "$HLOG/" 2>/dev/null || true
  pkill -9 -f retrieval_server.py 2>/dev/null || true; echo "[worker] exit: $status"; }
trap finish EXIT

eval "$(grep -E '^export (HF_TOKEN|WANDB_API_KEY)=' /home/tiger/.bashrc)" || true
export WANDB_PROJECT=ca-rung3-4b-feasibility
[ -z "${WANDB_API_KEY:-}" ] && export WANDB_MODE=offline
echo "==== P3 $COND (filtered base) start $(date -u) on $(hostname) ===="; nvidia-smi -L | head -2 || true

# ---- 1. training venv (cached) ----
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
RETR_LOG="$REPO/outputs/retriever/retriever_p3_${COND}.log"
serve_retriever() {
  ( cd "$REPO" && unset VIRTUAL_ENV PYTHONPATH PYTHONHOME \
    && export PATH="$RETR_ENV/bin:$PATH" LD_LIBRARY_PATH="$RETR_ENV/lib" \
       OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false \
    && exec "$RETR_ENV/bin/python" examples/search/retriever/retrieval_server.py \
      --index_path "$STAGE/e5_Flat.index" --corpus_path "$STAGE/wiki-18.jsonl" \
      --topk 3 --retriever_name e5 --retriever_model intfloat/e5-base-v2 --faiss_gpu --port 8000 \
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
      -d '{"queries":["who won the first nobel prize in physics"],"topk":3,"return_scores":true}') || CODE=000
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

# ---- 4. full RL run: identical protocol to P2, filtered data, new run names ----
export EXP_NAME="${COND}_qwen3-4b-2507_${PROTOCOL}_basefiltered150_s0"
export MICRO_BSZ_OVERRIDE=1
echo "==== P3 train: $COND 150 steps, filtered base (7438q band 0<p<8) ===="
RESUME=disable bash "$REPO/scripts/run_condition.sh" "$COND" 0 "$PROTOCOL"
RC=$?
echo "P3 run_condition exit=$RC"
[ "$RC" = "0" ] && status=DONE
echo "==== P3 $COND complete ($status) $(date -u) ===="
