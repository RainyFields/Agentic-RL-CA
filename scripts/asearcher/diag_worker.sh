#!/bin/bash
# Stage-2 pass-rate diagnostic worker (ATTACHED). One split per worker (DIAG_SPLIT via wrapper).
# Waits for the closed-book keep-set, then runs base Qwen3-4B-Instruct-2507 IN-SETUP
# (search tool, wiki-18 GPU-faiss, 8-turn, 16k) with 8 independent rollouts per kept
# question via trainer val_only (question replicated 8x in data; val_kwargs n stays 1
# because the val env pool cannot exceed data.val_batch_size). Post-processes the rollout
# dump into per-question pass rates + the 0<pass<8 band keep-set.
set -uo pipefail
export PYTHONUNBUFFERED=1

REPO=/home/tiger/xiaoxuan/Agentic-RL-CA
export VENV=/home/tiger/xiaoxuan/envs/agentic-rl-ca
export MODEL_PATH=/mnt/hdfs/mlsys/models/Qwen3-4B-Instruct-2507
export SEARCHR1_DATA=/mnt/hdfs/mlsys/users/xiaoxuan/searchr1/searchr1_data
export HF_DATASETS_CACHE=/tmp/hf_datasets_cache
PROTOCOL=asearcher_8turn_4b
SPLIT="${DIAG_SPLIT:?set DIAG_SPLIT=base|lrm}"
KEEP=$REPO/outputs/closed_book_filter/$SPLIT/keep_ge2.jsonl
CB_HLOG=/mnt/hdfs/mlsys/users/xiaoxuan/agentic_rl_ca/logs/closed_book_filter
STAGE=/tmp/searchR1
DIAG=/tmp/diag_$SPLIT
RETR_ENV=/home/tiger/xiaoxuan/envs/searchr1-retr-conda
MM=/home/tiger/xiaoxuan/tools/bin/micromamba
export MAMBA_ROOT_PREFIX=/home/tiger/xiaoxuan/tools/mamba
HLOG=/mnt/hdfs/mlsys/users/xiaoxuan/agentic_rl_ca/logs/passrate_diag_${SPLIT}
RESULT=$REPO/outputs/passrate_diag/$SPLIT
mkdir -p "$HLOG" "$REPO/outputs" "$STAGE" "$DIAG" "$RESULT"
rm -f "$HLOG/DONE" "$HLOG/FAILED"

status=FAILED
finish() { echo "$status $(date -u +%FT%TZ)" > "$HLOG/$status"
  cp -r "$RESULT"/* "$HLOG/" 2>/dev/null || true
  cp "$REPO/outputs/retriever/retriever_diag_$SPLIT.log" "$HLOG/" 2>/dev/null || true
  pkill -f retrieval_server.py 2>/dev/null || true; echo "[worker] exit: $status"; }
trap finish EXIT

eval "$(grep -E '^export (HF_TOKEN|WANDB_API_KEY)=' /home/tiger/.bashrc)" || true
export WANDB_PROJECT=ca-rung3-4b-feasibility
[ -z "${WANDB_API_KEY:-}" ] && export WANDB_MODE=offline
echo "==== passrate diag ($SPLIT) start $(date -u) on $(hostname) ===="; nvidia-smi -L || true

# ---- 1. training venv (cached) ----
uv venv -p 3.11 "$VENV" 2>/dev/null; source "$VENV/bin/activate"; export VIRTUAL_ENV="$VENV"
python -c 'import vllm, flash_attn, verl, agent_system' 2>/dev/null || { echo "venv incomplete"; exit 1; }

# ---- 2. GPU-faiss conda env (idempotent; persists on /home) ----
CLEAN=(env -u VIRTUAL_ENV -u PYTHONPATH -u PYTHONHOME)
if [ ! -x "$MM" ]; then
  echo "[conda] installing micromamba..."
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -C "$(dirname "$(dirname "$MM")")" -xj bin/micromamba || exit 1
fi
if ! "${CLEAN[@]}" "$MM" run -p "$RETR_ENV" python -c 'import faiss,torch,transformers; assert faiss.get_num_gpus()>0' 2>/dev/null; then
  echo "[conda] building faiss-gpu env (conda-forge: mkl + faiss-gpu + cpu-torch)..."
  rm -rf "$RETR_ENV"
  "${CLEAN[@]}" "$MM" create -y -p "$RETR_ENV" -c conda-forge python=3.11 \
      "mkl=2024.*" "faiss-gpu=1.10.0" "pytorch=*=cpu*" transformers fastapi uvicorn datasets numpy 2>&1 | tail -6 || exit 1
fi
"${CLEAN[@]}" "$MM" run -p "$RETR_ENV" python -c 'import faiss,torch; print("faiss",faiss.__version__,"gpus",faiss.get_num_gpus(),"torch",torch.__version__)' || exit 1

# ---- 3. stage index while the closed-book filter (other worker) finishes ----
[ -f "$STAGE/e5_Flat.index" ] || cp "$SEARCHR1_DATA/e5_Flat.index" "$STAGE/" || exit 1
[ -f "$STAGE/wiki-18.jsonl" ] || cp "$SEARCHR1_DATA/wiki-18.jsonl" "$STAGE/" || exit 1

# ---- 4. wait for the closed-book keep-set (shared /home; up to 6h) ----
echo "[gate] waiting for $KEEP ..."
ok=0
for i in $(seq 1 360); do
  [ -f "$CB_HLOG/FAILED" ] && { echo "[gate] closed-book filter FAILED — aborting"; exit 1; }
  # the file is written atomically-enough (single writer, then worker moves on to lrm);
  # require DONE marker OR stats.json for THIS split so we never read a partial file
  if [ -s "$KEEP" ] && [ -s "$REPO/outputs/closed_book_filter/$SPLIT/stats.json" ]; then ok=1; break; fi
  sleep 60
done
[ "$ok" = 1 ] || { echo "[gate] timeout waiting for keep-set"; exit 1; }
echo "[gate] keep-set ready: $(wc -l < "$KEEP") rows"

# ---- 5. serve GPU-faiss retriever ----
mkdir -p "$REPO/outputs/retriever"
( cd "$REPO" && unset VIRTUAL_ENV PYTHONPATH PYTHONHOME \
  && exec "$MM" run -p "$RETR_ENV" python examples/search/retriever/retrieval_server.py \
    --index_path "$STAGE/e5_Flat.index" --corpus_path "$STAGE/wiki-18.jsonl" \
    --topk 3 --retriever_name e5 --retriever_model intfloat/e5-base-v2 --faiss_gpu --port 8000 \
) > "$REPO/outputs/retriever/retriever_diag_$SPLIT.log" 2>&1 &
RETR_PID=$!
export SEARCH_URL="http://127.0.0.1:8000/retrieve"
echo "[retriever] GPU-faiss serving (pid $RETR_PID); health gate up to 40min..."
up=0
for i in $(seq 1 240); do
  kill -0 "$RETR_PID" 2>/dev/null || { echo "[retriever] DIED"; tail -30 "$REPO/outputs/retriever/retriever_diag_$SPLIT.log"; exit 1; }
  sleep 10
  CODE=$(curl -s -m 10 -o /tmp/h.json -w '%{http_code}' -X POST "$SEARCH_URL" -H 'Content-Type: application/json' \
      -d '{"queries":["who won the first nobel prize in physics"],"topk":3,"return_scores":true}') || CODE=000
  [ "$CODE" = "200" ] && { echo "[retriever] healthy t=$((i*10))s"; up=1; break; }
done
[ "$up" = 1 ] || { echo "[retriever] health timeout"; exit 1; }

# ---- 6. diag parquet (8x replication) on worker-local /tmp ----
python "$REPO/scripts/asearcher/make_diag_parquet.py" \
  --keep "$KEEP" --split "$SPLIT" --out-dir "$DIAG" --rep 8 || exit 1

# ---- 7. val_only in-setup rollouts: 8 per question, temp 1.0, dump to /tmp ----
export DATA_DIR="$DIAG"
export DUMP_TRAIN_SAMPLE=1
export DUMP_TRAIN_PATH="$DIAG/dump"          # tens of GB possible -> worker-local /tmp only
mkdir -p "$DUMP_TRAIN_PATH"; rm -f "$DUMP_TRAIN_PATH/rollout_log.jsonl"
export EXP_NAME="diag_passrate_${SPLIT}"
export VAL_BATCH_OVERRIDE=2048               # env pool size == concurrent rollouts
export MICRO_BSZ_OVERRIDE=1
echo "==== diag rollouts: $SPLIT, 8x per question, val_only ===="
RESUME=disable bash "$REPO/scripts/run_condition.sh" token_grpo 0 "$PROTOCOL" \
  trainer.val_only=True \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
  actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
  trainer.validation_data_dir="$DIAG/valdump"
RC=$?
echo "diag run_condition exit=$RC"
[ "$RC" = "0" ] || exit 1

# ---- 8. per-question pass rates + band keep-set (small; shared /home + HDFS) ----
python "$REPO/scripts/asearcher/passrate_from_dump.py" \
  --dump "$DUMP_TRAIN_PATH/rollout_log.jsonl" --keep "$KEEP" --out-dir "$RESULT" || exit 1
# keep the raw dump retrievable for CA analysis (compressed, HDFS only)
zstd -q -T8 "$DUMP_TRAIN_PATH/rollout_log.jsonl" -o "$HLOG/rollout_log.jsonl.zst" 2>/dev/null \
  || gzip -c "$DUMP_TRAIN_PATH/rollout_log.jsonl" > "$HLOG/rollout_log.jsonl.gz" || true
status=DONE
echo "==== passrate diag ($SPLIT) complete $(date -u) ===="
cat "$RESULT/stats.json" || true
