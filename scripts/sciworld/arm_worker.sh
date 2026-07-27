#!/bin/bash
# Rung-4 training arm worker (ATTACHED). COND=sw_orm_grpo | sw_prm_rtg (set by wrapper).
# 150 steps, val every 25, wandb ONLINE (ca-rung4-4b). Crash-resume loop: up to 3
# attempts, verl resume_mode=auto picks up the latest checkpoint on HDFS.
set -uo pipefail
export PYTHONUNBUFFERED=1

REPO=/home/tiger/xiaoxuan/Agentic-RL-CA
export VENV=/home/tiger/xiaoxuan/envs/agentic-rl-ca
export MODEL_PATH=/mnt/hdfs/mlsys/models/Qwen3-4B-Instruct-2507
export HF_DATASETS_CACHE=/tmp/hf_datasets_cache
COND="${COND:?set COND=sw_orm_grpo|sw_prm_rtg}"
SEED="${SEED:-0}"
HLOG=/mnt/hdfs/mlsys/users/xiaoxuan/agentic_rl_ca/logs/rung4_sciworld_${COND}
mkdir -p "$HLOG" "$REPO/outputs"
rm -f "$HLOG/DONE" "$HLOG/FAILED"

status=FAILED
finish() { echo "$status $(date -u +%FT%TZ)" > "$HLOG/$status"; echo "[arm-worker] exit: $status"; }
trap finish EXIT

echo "==== rung4 $COND s$SEED start $(date -u) on $(hostname) ===="; nvidia-smi -L || true

if ! command -v java >/dev/null 2>&1; then
  if [ -x /home/tiger/xiaoxuan/tools/jre17/bin/java ]; then
    export PATH=/home/tiger/xiaoxuan/tools/jre17/bin:$PATH
    export JAVA_HOME=/home/tiger/xiaoxuan/tools/jre17
  else
    (sudo apt-get update -y && sudo apt-get install -y openjdk-17-jre-headless) || \
    (apt-get update -y && apt-get install -y openjdk-17-jre-headless) || true
  fi
fi
java -version 2>&1 | head -1 || { echo "FATAL: no java"; exit 1; }

source "$VENV/bin/activate"
python -c 'import vllm, verl, agent_system, scienceworld, gymnasium' || { echo "venv incomplete"; exit 1; }
eval "$(grep -E '^export (HF_TOKEN|WANDB_API_KEY)=' /home/tiger/.bashrc)" || true
[ -z "${WANDB_API_KEY:-}" ] && export WANDB_MODE=offline

cd "$REPO"
for attempt in 1 2 3; do
  echo "==== attempt $attempt $(date -u) ===="
  bash scripts/sciworld/run_sciworld.sh "$COND" "$SEED" sciworld_4b \
    trainer.resume_mode=auto \
    2>&1 | tee -a "$REPO/outputs/rung4_${COND}.log"
  rc=${PIPESTATUS[0]}
  if [ "$rc" -eq 0 ]; then status=DONE; break; fi
  echo "==== attempt $attempt failed rc=$rc; retrying after 60s ===="; sleep 60
done

echo "==== rung4 $COND exit $status $(date -u) ===="
