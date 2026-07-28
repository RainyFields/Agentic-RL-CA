#!/bin/bash
# Phase 3.3 horizon eval worker (ATTACHED): venv -> retriever -> [smoke val_2048 self-check] ->
# full-set test.parquet eval, for one or more checkpoints run sequentially.
# Invoked by a tiny wrapper baking env vars (mlx --envs is broken):
#   exec env EVAL_SPECS="CKPT|LABEL|COND[;CKPT|LABEL|COND...]" PROTOCOL=4turn_think2k bash worker_eval.sh
# The enriched per-episode JSONL (§4 schema) + paper_table are copied back to the shared repo;
# the ~8GB HF-merged models stay on worker-local /tmp (300G) so the 26G-free shared volume is safe.
set -uo pipefail
export PYTHONUNBUFFERED=1

REPO=/home/tiger/xiaoxuan/Agentic-RL-CA
export VENV=/home/tiger/xiaoxuan/envs/agentic-rl-ca
export DATA_DIR="${DATA_DIR:-$REPO/data/searchR1_processed_direct}"
PROTOCOL="${PROTOCOL:-4turn_think2k}"
SMOKE="${SMOKE:-1}"
SPECS="${EVAL_SPECS:?set EVAL_SPECS as CKPT|LABEL|COND[;...]}"
JOB_TAG="${JOB_TAG:-$(echo "$SPECS" | md5sum | cut -c1-8)}"
TMPOUT=/tmp/arlca_eval
SHARED_OUT="$REPO/outputs/eval_full"
HLOG="/mnt/hdfs/mlsys/users/xiaoxuan/agentic_rl_ca/logs/eval_${JOB_TAG}"
mkdir -p "$HLOG" "$SHARED_OUT" "$TMPOUT"
rm -f "$HLOG/DONE" "$HLOG/FAILED"

status=FAILED
finish() {
  echo "$status $(date -u +%FT%TZ)" > "$HLOG/$status"
  pkill -f retrieval_server.py 2>/dev/null || true
  echo "[worker-eval] exit: $status (tag=$JOB_TAG)"
}
trap finish EXIT

eval "$(grep -E '^export (HF_TOKEN|WANDB_API_KEY)=' /home/tiger/.bashrc)" || true
[ -z "${WANDB_API_KEY:-}" ] && export WANDB_MODE=offline

echo "==== worker_eval start $(date -u) $(hostname) proto=$PROTOCOL tag=$JOB_TAG ===="
echo "specs: $SPECS"
nvidia-smi -L || true

source "$VENV/bin/activate" 2>/dev/null || { echo "venv missing — run phase1 job first"; exit 1; }
python -c "import torch, vllm, verl, credit_assignment" || { echo "venv incomplete"; exit 1; }

# ---- retriever (blocks until healthy) ----
bash "$REPO/scripts/retriever_serve.sh" || { echo "retriever failed"; exit 1; }

# pre-merge a verl FSDP ckpt -> HF once (shared by smoke + full); echoes the model dir to use
pre_merge() {  # ckpt label  -> prints model dir
  local ckpt="$1" label="$2"
  if [[ -d "$ckpt/actor" ]]; then
    local merged="$TMPOUT/merged/$label"
    if [[ ! -f "$merged/config.json" ]]; then
      echo "[merge] $ckpt/actor -> $merged" >&2
      python "$REPO/scripts/model_merger.py" merge --backend fsdp \
          --local_dir "$ckpt/actor" --target_dir "$merged" >&2 || { echo "[merge] FAILED" >&2; return 1; }
    fi
    echo "$merged"
  else
    echo "$ckpt"   # already an HF dir (e.g. the pre-RL floor)
  fi
}

run_one() {  # model_dir label cond valfiles outsuffix
  local model="$1" label="$2" cond="$3" valfiles="$4" suffix="$5"
  local out="$TMPOUT/${label}${suffix}"
  rm -rf "$out"; mkdir -p "$out"
  echo "==== eval ${label}${suffix} ($(date -u)) model=$model val=$(basename "$valfiles") ===="
  OUT_DIR="$out" VAL_FILES="$valfiles" EVAL_VAL_BATCH="${EVAL_VAL_BATCH:-512}" \
    bash "$REPO/scripts/eval_search_full.sh" "$model" "${label}${suffix}" "$cond" "$PROTOCOL"
}

copyback() {  # srcdir label
  local src="$1" label="$2"
  mkdir -p "$SHARED_OUT/$label"
  cp "$src"/val_trajectories_step*.jsonl "$SHARED_OUT/$label/" 2>/dev/null || true
  cp "$src"/paper_table.* "$SHARED_OUT/$label/" 2>/dev/null || true
  echo "[worker-eval] $label artifacts -> $SHARED_OUT/$label"
}

validate_dump() {  # jsonl-glob
  python - "$1" <<'PY'
import sys, json, glob
files = sorted(glob.glob(sys.argv[1]))
assert files, f"no jsonl matched {sys.argv[1]}"
f = files[-1]
req = {"data_source","em","turns","index","n_search_calls","tokens_generated","truncated_at_cap","answered"}
n = ni = ns = nt = 0
with open(f) as fh:
    for line in fh:
        r = json.loads(line); n += 1
        miss = req - set(r); assert not miss, f"missing §4 fields {miss}"
        if r.get("index") is not None and int(r.get("index", -1)) >= 0: ni += 1
        if r.get("n_search_calls") is not None: ns += 1
        if r.get("tokens_generated"): nt += 1
assert n > 0, "empty dump"
print(f"[validate] {f}: {n} recs; index_valid={ni/n:.3f} nsearch_set={ns/n:.3f} tokens_pos={nt/n:.3f}")
assert ni/n > 0.95, f"index join-key missing for {(1-ni/n)*100:.1f}% — index passthrough broke; aborting before full run"
print("[validate] §4 schema OK")
PY
}

IFS=';' read -ra ARR <<< "$SPECS"
first=1
for spec in "${ARR[@]}"; do
  [[ -z "${spec// }" ]] && continue
  IFS='|' read -r ckpt label cond <<< "$spec"
  cond="${cond:-token_grpo}"
  model="$(pre_merge "$ckpt" "$label")" || { echo "merge failed for $label"; exit 1; }

  # smoke (val_2048) once, on the first checkpoint, to validate the §4 dump end-to-end before any full run
  if [[ "$SMOKE" == "1" && "$first" == "1" ]]; then
    echo "---- SMOKE val_2048 on $label ----"
    run_one "$model" "$label" "$cond" "$DATA_DIR/val_2048.parquet" "_smoke" || { echo "smoke run failed"; exit 1; }
    validate_dump "$TMPOUT/${label}_smoke/val_trajectories_step*.jsonl" || { echo "smoke validate failed"; exit 1; }
    copyback "$TMPOUT/${label}_smoke" "${label}_smoke"
  fi
  first=0

  # full test.parquet (2 attempts; idempotent — overwrites OUT_DIR, merge is cached)
  ok=0
  for att in 1 2; do
    echo "---- FULL test.parquet on $label (attempt $att) ----"
    if run_one "$model" "$label" "$cond" "$DATA_DIR/test.parquet" ""; then ok=1; break; fi
    echo "[worker-eval] full attempt $att failed for $label; retriever health-check"
    curl -s -m 10 -X POST "${SEARCH_URL:-http://127.0.0.1:8000/retrieve}" -H 'Content-Type: application/json' \
        -d '{"queries":["health"],"topk":1,"return_scores":true}' >/dev/null 2>&1 || \
        bash "$REPO/scripts/retriever_serve.sh" || true
    sleep 20
  done
  [[ "$ok" == "1" ]] || { echo "full eval failed for $label after retries"; exit 1; }
  copyback "$TMPOUT/$label" "$label"
done

status=DONE
echo "==== worker_eval end $(date -u): $status ===="
