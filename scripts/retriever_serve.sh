#!/usr/bin/env bash
# Phase 1.2 — retriever standup (tr1-validated recipe, co-located on the training worker).
# Serves Search-R1 retrieval_server.py (e5 + 61G flat faiss index, fp16-sharded across the
# GPUs visible to it) from the existing conda-forge faiss-gpu env (pip wheels lack sm_90).
# Usage (on the GPU worker):
#   scripts/retriever_serve.sh              # foreground child + health gate; leaves server up
#   RETR_GPUS=0,1 scripts/retriever_serve.sh
# Health-gated: exits 0 only after a live /retrieve responds (index load from HDFS FUSE can
# take tens of minutes). API contract: POST {"queries":[...],"topk":k,"return_scores":true}.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

SEARCHR1_REPO="${SEARCHR1_REPO:-$HOME/xiaoxuan/searchr1/Search-R1}"
RETR_ENV="${RETR_ENV:-$HOME/xiaoxuan/envs/searchr1-retr-conda}"
MM="${MM:-$HOME/xiaoxuan/tools/bin/micromamba}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/xiaoxuan/tools/mamba}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/mnt/hdfs/mlsys/users/xiaoxuan/hf_cache/hub}"
RETR_GPUS="${RETR_GPUS:-}"            # empty = all visible GPUs (faiss shards the index)
RETR_PORT="${RETR_PORT:-8000}"
LOG_DIR="$REPO_DIR/outputs/retriever"
mkdir -p "$LOG_DIR"

[[ -f "$SEARCHR1_DATA/e5_Flat.index" ]] || { echo "missing index at $SEARCHR1_DATA" >&2; exit 2; }

echo "[retriever] starting (gpus='${RETR_GPUS:-all}', port=$RETR_PORT)..."
(
  cd "$SEARCHR1_REPO"
  [[ -n "$RETR_GPUS" ]] && export CUDA_VISIBLE_DEVICES="$RETR_GPUS"
  exec "$MM" run -p "$RETR_ENV" python search_r1/search/retrieval_server.py \
      --index_path  "$SEARCHR1_DATA/e5_Flat.index" \
      --corpus_path "$SEARCHR1_DATA/wiki-18.jsonl" \
      --topk 3 --retriever_name e5 --retriever_model intfloat/e5-base-v2 \
      --faiss_gpu
) > "$LOG_DIR/retriever.log" 2>&1 &
RETR_PID=$!
echo "$RETR_PID" > "$LOG_DIR/retriever.pid"

echo "[retriever] health gate (up to 40 min for HDFS-FUSE index load)..."
for i in $(seq 1 240); do
  kill -0 "$RETR_PID" 2>/dev/null || { echo "[retriever] DIED — tail of log:"; tail -20 "$LOG_DIR/retriever.log"; exit 1; }
  sleep 10
  CODE=$(curl -s -m 10 -o "$LOG_DIR/health.json" -w '%{http_code}' -X POST "http://127.0.0.1:$RETR_PORT/retrieve" \
      -H 'Content-Type: application/json' \
      -d '{"queries":["who won the first nobel prize in physics"],"topk":3,"return_scores":true}') || CODE=000
  if [[ "$CODE" == "200" ]]; then
    echo "[retriever] healthy at t=$((i*10))s; sample:"; head -c 400 "$LOG_DIR/health.json"; echo
    exit 0
  fi
done
echo "[retriever] health timeout (40 min)"; exit 1
