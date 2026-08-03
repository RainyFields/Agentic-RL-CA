#!/bin/bash
# P8B EVAL: run one 8B checkpoint over the ASearcher-paper evaluation suite
# (10 benchmarks / 7,152 questions, data_source-tagged so verl emits per-benchmark metrics).
#   CKPT=<hdfs ckpt dir or HF dir> LABEL=<tag> [EVAL_REPEAT=1] [EVAL_TEMP=0] bash p8b_eval_worker.sh
# Default is GREEDY single-sample scored by the env's own reward, which is STRICT exact
# match (search/.../utils.py:compute_score) — the same metric as val/asearcher_base/em, so
# these numbers are directly comparable to the training vals. NOT sub-EM. For the paper's
# Avg@4 protocol set EVAL_TEMP=0.6 and point EVAL_FILE at a --repeat 4 parquet.
# Rollouts are dumped so judge_rollouts.py can score them later (LLM judge + EM + sub-EM)
# without re-running generation; see p8b_judge_worker.sh.
set -uo pipefail
export PYTHONUNBUFFERED=1

CKPT="${CKPT:?set CKPT=<checkpoint dir>}"
LABEL="${LABEL:?set LABEL=<run tag>}"
COND="${COND:-token_grpo}"          # only sets adv_estimator for config validity
REPO=${REPO:-/home/tiger/xiaoxuan/arlca-8b}
export PYTHONPATH="$REPO"
export VENV=/home/tiger/xiaoxuan/envs/agentic-rl-ca
export DATA_DIR=/mnt/hdfs/mlsys/users/xiaoxuan/agentic_rl_ca/data_asearcher_base
EVAL_FILE="${EVAL_FILE:-/mnt/hdfs/mlsys/users/xiaoxuan/agentic_rl_ca/data_asearcher_eval/asearcher_eval.parquet}"
export SEARCHR1_DATA=/mnt/hdfs/mlsys/users/xiaoxuan/searchr1/searchr1_data
export HF_DATASETS_CACHE=/tmp/hf_datasets_cache
PROTOCOL=asearcher_32turn_8b
STAGE=/tmp/searchR1
RETR_ENV=/home/tiger/xiaoxuan/envs/searchr1-retr-conda
MM=/home/tiger/xiaoxuan/tools/bin/micromamba
export MAMBA_ROOT_PREFIX=/home/tiger/xiaoxuan/tools/mamba
HLOG=/mnt/hdfs/mlsys/users/xiaoxuan/agentic_rl_ca/logs/p8b_eval_${LABEL}
mkdir -p "$HLOG" "$REPO/outputs" "$STAGE"
rm -f "$HLOG/DONE" "$HLOG/FAILED"

status=FAILED
finish() { echo "$status $(date -u +%FT%TZ)" > "$HLOG/$status"
  kill "${WATCHDOG_PID:-}" 2>/dev/null || true
  pkill -9 -f retrieval_server.py 2>/dev/null || true; echo "[worker] exit: $status"; }
trap finish EXIT

eval "$(grep -E '^export (HF_TOKEN|WANDB_API_KEY)=' /home/tiger/.bashrc)" || true
[ -z "${WANDB_API_KEY:-}" ] && export WANDB_MODE=offline
echo "==== P8B-EVAL $LABEL start $(date -u) on $(hostname) ===="

[ -f "$EVAL_FILE" ] || { echo "PREFLIGHT FAIL: eval parquet $EVAL_FILE"; exit 1; }
uv venv -p 3.11 "$VENV" 2>/dev/null; source "$VENV/bin/activate"; export VIRTUAL_ENV="$VENV"
python -c 'import vllm, verl' 2>/dev/null || { echo "venv incomplete"; exit 1; }
python - <<'EOF' || { echo "PREFLIGHT FAIL: NVML"; exit 1; }
import ctypes
lib = ctypes.CDLL("libnvidia-ml.so.1"); assert lib.nvmlInit_v2() == 0
n = ctypes.c_uint(); assert lib.nvmlDeviceGetCount_v2(ctypes.byref(n)) == 0 and n.value >= 8
EOF

# ---- merge verl checkpoint -> HF if needed ----
MODEL_DIR="$CKPT"
if [ -d "$CKPT/actor" ]; then
  MERGED="/tmp/eval_merged_${LABEL}"
  if [ ! -f "$MERGED/config.json" ]; then
    echo "[eval] merging $CKPT/actor -> $MERGED"
    python "$REPO/scripts/model_merger.py" merge --backend fsdp \
      --local_dir "$CKPT/actor" --target_dir "$MERGED" || exit 1
  fi
  MODEL_DIR="$MERGED"
fi
export MODEL_PATH="$MODEL_DIR"
[ -f "$MODEL_PATH/config.json" ] || { echo "no model at $MODEL_PATH"; exit 1; }

# ---- retriever (top-5, thread-capped) ----
[ -f "$STAGE/e5_Flat.index" ] || cp "$SEARCHR1_DATA/e5_Flat.index" "$STAGE/" || exit 1
[ -f "$STAGE/wiki-18.jsonl" ] || cp "$SEARCHR1_DATA/wiki-18.jsonl" "$STAGE/" || exit 1
mkdir -p "$REPO/outputs/retriever"
RETR_LOG="$REPO/outputs/retriever/retriever_eval_${LABEL}.log"
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
  pgrep -f retrieval_server.py >/dev/null || { echo "[retriever] DIED"; tail -20 "$RETR_LOG"; exit 1; }
  sleep 10
  CODE=$(curl -s -m 10 -o /dev/null -w '%{http_code}' -X POST "$SEARCH_URL" -H 'Content-Type: application/json' \
      -d '{"queries":["health probe"],"topk":5,"return_scores":true}') || CODE=000
  [ "$CODE" = "200" ] && { echo "[retriever] healthy t=$((i*10))s"; up=1; break; }
done
[ "$up" = 1 ] || { echo "[retriever] health timeout"; exit 1; }
( fails=0
  while true; do
    sleep 120
    CODE=$(curl -s -m 30 -o /dev/null -w '%{http_code}' -X POST "$SEARCH_URL" -H 'Content-Type: application/json' \
        -d '{"queries":["probe"],"topk":1,"return_scores":true}') || CODE=000
    if [ "$CODE" = "200" ]; then fails=0; continue; fi
    fails=$((fails+1))
    if [ "$fails" -ge 3 ]; then
      pkill -9 -f retrieval_server.py 2>/dev/null
      for w in $(seq 1 12); do pgrep -f retrieval_server.py >/dev/null || break; sleep 5; done
      serve_retriever; fails=0
    fi
  done ) &
WATCHDOG_PID=$!

# ---- val_only pass over the eval suite; dump rollouts for later judge scoring ----
export EXP_NAME="p8beval_${LABEL}"
export DUMP_TRAIN_SAMPLE=1 DUMP_TRAIN_PATH="/tmp/p8b_eval_dump/${LABEL}"
mkdir -p "$DUMP_TRAIN_PATH"; rm -f "$DUMP_TRAIN_PATH/rollout_log.jsonl"
EVAL_TEMP="${EVAL_TEMP:-0}"
VAL_BATCH_OVERRIDE="${VAL_BATCH_OVERRIDE:-512}"
export VAL_BATCH_OVERRIDE
echo "==== P8B-EVAL $LABEL: val_only over $(basename "$EVAL_FILE") temp=$EVAL_TEMP $(date -u) ===="
VAL_FILE="$EVAL_FILE" TOTAL_STEPS=1 VAL_FREQ=1 SAVE_FREQ=1000000 VAL_BEFORE_TRAIN=True \
  RESUME=disable timeout 20h bash "$REPO/scripts/run_condition.sh" "$COND" 0 "$PROTOCOL" \
  trainer.val_only=True \
  actor_rollout_ref.rollout.val_kwargs.temperature="$EVAL_TEMP" \
  actor_rollout_ref.rollout.val_kwargs.do_sample=$([ "$EVAL_TEMP" = "0" ] && echo False || echo True)
RC=$?
echo "==== P8B-EVAL $LABEL exit=$RC $(date -u) ===="
[ "$RC" = "0" ] && status=DONE
if [ -f "$DUMP_TRAIN_PATH/rollout_log.jsonl" ]; then
  zstd -q -f "$DUMP_TRAIN_PATH/rollout_log.jsonl" -o "$HLOG/rollout_eval_${LABEL}.jsonl.zst" \
    && rm -rf "$DUMP_TRAIN_PATH"
fi
echo "==== P8B-EVAL $LABEL complete ($status) $(date -u) ===="
