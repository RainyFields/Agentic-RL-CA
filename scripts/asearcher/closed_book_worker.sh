#!/bin/bash
# Closed-book (stage-1) filter worker (ATTACHED). vLLM-only, NO retriever.
# Runs base Qwen3-4B-Instruct-2507 tool-free on ASearcher Base + LRM, drops parametric-answerable.
set -uo pipefail
export PYTHONUNBUFFERED=1
REPO=/home/tiger/xiaoxuan/Agentic-RL-CA
export VENV=/home/tiger/xiaoxuan/envs/agentic-rl-ca
MODEL=/mnt/hdfs/mlsys/models/Qwen3-4B-Instruct-2507
RAW=/mnt/hdfs/mlsys/users/xiaoxuan/agentic_rl_ca/data_asearcher_raw
HLOG=/mnt/hdfs/mlsys/users/xiaoxuan/agentic_rl_ca/logs/closed_book_filter
OUT=$REPO/outputs/closed_book_filter
mkdir -p "$HLOG" "$OUT"
rm -f "$HLOG/DONE" "$HLOG/FAILED"
status=FAILED
finish(){ echo "$status $(date -u +%FT%TZ)" > "$HLOG/$status"; cp -r "$OUT"/* "$HLOG/" 2>/dev/null || true; echo "[worker] exit: $status"; }
trap finish EXIT
export HF_DATASETS_CACHE=/tmp/hf_datasets_cache
eval "$(grep -E '^export (HF_TOKEN|WANDB_API_KEY)=' /home/tiger/.bashrc)" || true

echo "==== closed-book filter start $(date -u) on $(hostname) ===="; nvidia-smi -L | head -1
uv venv -p 3.11 "$VENV" 2>/dev/null; source "$VENV/bin/activate"; export VIRTUAL_ENV="$VENV"
python -c 'import vllm, transformers' || { echo "venv incomplete"; exit 1; }

for split in base lrm; do
  echo "==== filtering $split ===="
  TP=8 python "$REPO/scripts/asearcher/closed_book_filter.py" \
     --input "$RAW/$split.jsonl" --split "$split" --out-dir "$OUT" --model "$MODEL" --n 4 --temp 1.0 \
     || { echo "$split filter failed"; exit 1; }
done
status=DONE
echo "==== closed-book filter complete $(date -u) ===="
for s in base lrm; do echo "--- $s ---"; cat "$OUT/$s/stats.json" 2>/dev/null; done
