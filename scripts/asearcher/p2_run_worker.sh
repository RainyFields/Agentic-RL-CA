#!/bin/bash
# P2 feasibility RL run (ATTACHED). Rung-3 / ASearcher-Base, Qwen3-4B-Instruct-2507, single tool,
# 8-turn, 16k prompt, NO cold start. Full batch, 150 steps, GPU-faiss retriever (co-located).
# Parameterized by CONDITION (token_grpo | turn_ppo_b0) via a wrapper (mlx --envs is broken).
# Writes DONE/FAILED to $HLOG. wandb online (project ca-rung3-4b-feasibility).
set -uo pipefail
export PYTHONUNBUFFERED=1

REPO=/home/tiger/xiaoxuan/Agentic-RL-CA
export VENV=/home/tiger/xiaoxuan/envs/agentic-rl-ca
export MODEL_PATH=/mnt/hdfs/mlsys/models/Qwen3-4B-Instruct-2507
export DATA_DIR=/mnt/hdfs/mlsys/users/xiaoxuan/agentic_rl_ca/data_asearcher_base
export SEARCHR1_DATA=/mnt/hdfs/mlsys/users/xiaoxuan/searchr1/searchr1_data
# corpus Arrow cache -> worker-local /tmp, NOT shared /home (the July fleet-hang root cause)
export HF_DATASETS_CACHE=/tmp/hf_datasets_cache
PROTOCOL=asearcher_8turn_4b
COND="${COND:?set COND=token_grpo|turn_ppo_b0}"
STAGE=/tmp/searchR1
# conda faiss-gpu env (sm_90; pip faiss-gpu-cu12 lacks H100 kernels). Persist on /home.
RETR_ENV=/home/tiger/xiaoxuan/envs/searchr1-retr-conda
MM=/home/tiger/xiaoxuan/tools/bin/micromamba
export MAMBA_ROOT_PREFIX=/home/tiger/xiaoxuan/tools/mamba
HLOG=/mnt/hdfs/mlsys/users/xiaoxuan/agentic_rl_ca/logs/p2_asearcher_${COND}
mkdir -p "$HLOG" "$REPO/outputs" "$STAGE" "$(dirname "$MM")"
rm -f "$HLOG/DONE" "$HLOG/FAILED"

status=FAILED
finish() { echo "$status $(date -u +%FT%TZ)" > "$HLOG/$status"
  cp "$REPO/outputs/retriever/retriever.log" "$HLOG/" 2>/dev/null || true
  pkill -f retrieval_server.py 2>/dev/null || true; echo "[worker] exit: $status"; }
trap finish EXIT

eval "$(grep -E '^export (HF_TOKEN|WANDB_API_KEY)=' /home/tiger/.bashrc)" || true
export WANDB_PROJECT=ca-rung3-4b-feasibility
[ -z "${WANDB_API_KEY:-}" ] && export WANDB_MODE=offline
echo "==== P2 $COND start $(date -u) on $(hostname) ===="; nvidia-smi -L || true

# ---- 1. training venv (cached from P1) ----
uv venv -p 3.11 "$VENV" 2>/dev/null; source "$VENV/bin/activate"; export VIRTUAL_ENV="$VENV"
python -c 'import vllm, flash_attn, verl, agent_system' 2>/dev/null || { echo "venv incomplete — run P1 first"; exit 1; }

# ---- 2. GPU-faiss conda env (idempotent; persists on /home) ----
if [ ! -x "$MM" ]; then
  echo "[conda] installing micromamba..."
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -C "$(dirname "$(dirname "$MM")")" -xj bin/micromamba || exit 1
fi
if ! "$MM" run -p "$RETR_ENV" python -c 'import faiss; assert faiss.get_num_gpus()>0' 2>/dev/null; then
  echo "[conda] building faiss-gpu env (conda-forge, sm_90)..."
  "$MM" create -y -p "$RETR_ENV" -c conda-forge -c pytorch python=3.11 "faiss-gpu=1.10.0" 2>&1 | tail -5 || exit 1
  "$MM" run -p "$RETR_ENV" pip install -q transformers torch fastapi uvicorn datasets numpy 2>&1 | tail -3 || exit 1
fi
"$MM" run -p "$RETR_ENV" python -c 'import faiss,torch; print("faiss",faiss.__version__,"gpus",faiss.get_num_gpus())' || exit 1

# ---- 3. stage index + serve GPU-faiss retriever (co-located; shards flat index across GPUs) ----
[ -f "$STAGE/e5_Flat.index" ] || cp "$SEARCHR1_DATA/e5_Flat.index" "$STAGE/" || exit 1
[ -f "$STAGE/wiki-18.jsonl" ] || cp "$SEARCHR1_DATA/wiki-18.jsonl" "$STAGE/" || exit 1
mkdir -p "$REPO/outputs/retriever"
( cd "$REPO" && exec "$MM" run -p "$RETR_ENV" python examples/search/retriever/retrieval_server.py \
    --index_path "$STAGE/e5_Flat.index" --corpus_path "$STAGE/wiki-18.jsonl" \
    --topk 3 --retriever_name e5 --retriever_model intfloat/e5-base-v2 --faiss_gpu --port 8000 \
) > "$REPO/outputs/retriever/retriever.log" 2>&1 &
RETR_PID=$!
export SEARCH_URL="http://127.0.0.1:8000/retrieve"
echo "[retriever] GPU-faiss serving (pid $RETR_PID); health gate up to 40min..."
up=0
for i in $(seq 1 240); do
  kill -0 "$RETR_PID" 2>/dev/null || { echo "[retriever] DIED"; tail -30 "$REPO/outputs/retriever/retriever.log"; exit 1; }
  sleep 10
  CODE=$(curl -s -m 10 -o /tmp/h.json -w '%{http_code}' -X POST "$SEARCH_URL" -H 'Content-Type: application/json' \
      -d '{"queries":["who won the first nobel prize in physics"],"topk":3,"return_scores":true}') || CODE=000
  [ "$CODE" = "200" ] && { echo "[retriever] healthy t=$((i*10))s"; up=1; break; }
done
[ "$up" = 1 ] || { echo "[retriever] health timeout"; exit 1; }

# ---- 4. full RL run: 150 steps, real batch, single seed, val 256, wandb online ----
echo "==== P2 train: $COND 150 steps, 4B/8turn/16k, ASearcher-Base ===="
RESUME=disable bash "$REPO/scripts/run_condition.sh" "$COND" 0 "$PROTOCOL"
RC=$?
echo "P2 run_condition exit=$RC"
[ "$RC" = "0" ] && status=DONE
echo "==== P2 $COND complete ($status) $(date -u) ===="
