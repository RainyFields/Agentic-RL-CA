#!/bin/bash
# Wave training worker job (ATTACHED): env-check -> retriever -> crash-resume training loop.
# Invoked via a tiny wrapper baking env vars (mlx --envs is broken; see merlin skill §9):
#   exec env COND=b1 SEED=0 PROTOCOL=4turn bash worker_train.sh
# verl resume_mode=auto continues from the latest checkpoint in default_local_dir (HDFS),
# so a crash costs at most save_freq steps.
set -uo pipefail
export PYTHONUNBUFFERED=1

REPO=/home/tiger/xiaoxuan/Agentic-RL-CA
export VENV=/home/tiger/xiaoxuan/envs/agentic-rl-ca
COND="${COND:?set COND}"
SEED="${SEED:-0}"
PROTOCOL="${PROTOCOL:-4turn}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-6}"
HLOG=/mnt/hdfs/mlsys/users/xiaoxuan/agentic_rl_ca/logs/train_${COND}_s${SEED}_${PROTOCOL}
mkdir -p "$HLOG"
rm -f "$HLOG/DONE" "$HLOG/FAILED"

status=FAILED
finish() {
  echo "$status $(date -u +%FT%TZ)" > "$HLOG/$status"
  pkill -f retrieval_server.py 2>/dev/null || true
  echo "[worker-train] exit: $status ($COND s$SEED $PROTOCOL)"
}
trap finish EXIT

eval "$(grep -E '^export (HF_TOKEN|WANDB_API_KEY)=' /home/tiger/.bashrc)" || true
[ -z "${WANDB_API_KEY:-}" ] && export WANDB_MODE=offline

echo "==== worker_train start $(date -u) $(hostname) cond=$COND seed=$SEED proto=$PROTOCOL ===="
nvidia-smi -L || true

# ---- env: venv is persistent on /home/tiger (built by phase1 job); verify, repair if needed ----
source "$VENV/bin/activate" 2>/dev/null || { echo "venv missing — run phase1 job first"; exit 1; }
python -c "import torch, vllm, verl, gym, agent_system, credit_assignment" || {
  echo "venv incomplete — repairing"; uv pip install -e "$REPO" gym || exit 1
  python -c "import verl, credit_assignment" || exit 1
}

# ---- retriever ----
bash "$REPO/scripts/retriever_serve.sh" || { echo "retriever failed"; exit 1; }

# ---- crash-resume training loop ----
for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  echo "==== training attempt $attempt/$MAX_ATTEMPTS $(date -u) ===="
  if bash "$REPO/scripts/run_condition.sh" "$COND" "$SEED" "$PROTOCOL"; then
    status=DONE
    break
  fi
  echo "[worker-train] attempt $attempt failed; retriever health-check then resume"
  curl -s -m 10 -X POST "${SEARCH_URL:-http://127.0.0.1:8000/retrieve}" \
      -H 'Content-Type: application/json' \
      -d '{"queries":["health"],"topk":1,"return_scores":true}' >/dev/null 2>&1 || {
    echo "[worker-train] retriever dead — restarting"
    pkill -f retrieval_server.py 2>/dev/null || true
    sleep 10
    bash "$REPO/scripts/retriever_serve.sh" || { echo "retriever restart failed"; exit 1; }
  }
  sleep 30
done

echo "==== worker_train end $(date -u): $status ===="
