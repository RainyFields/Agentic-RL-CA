#!/bin/bash
# Rung-4 mini-smoke (ATTACHED): 2 full-geometry training steps (64 groups x G=8, PRM arm)
# to verify the UPDATE phase completes after A14 (12288 padding + dynamic bsz + strata)
# and to measure true step wall-clock + /tmp watermark. wandb offline.
set -uo pipefail
export PYTHONUNBUFFERED=1

REPO=/home/tiger/xiaoxuan/Agentic-RL-CA
export VENV=/home/tiger/xiaoxuan/envs/agentic-rl-ca
export MODEL_PATH=/mnt/hdfs/mlsys/models/Qwen3-4B-Instruct-2507
export HF_DATASETS_CACHE=/tmp/hf_datasets_cache
HLOG=/mnt/hdfs/mlsys/users/xiaoxuan/agentic_rl_ca/logs/rung4_mini_smoke
mkdir -p "$HLOG" "$REPO/outputs"
rm -f "$HLOG/DONE" "$HLOG/FAILED"

status=FAILED
finish() { echo "$status $(date -u +%FT%TZ)" > "$HLOG/$status"; echo "[mini-smoke] exit: $status"; }
trap finish EXIT

echo "==== rung4 mini-smoke start $(date -u) on $(hostname) ===="; nvidia-smi -L || true

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
export WANDB_MODE=offline

# /tmp watermark sampler (60s cadence) so we can see the spill profile post-hoc
( while true; do echo "[tmpwatch] $(date -u +%T) $(df -h /tmp | tail -1 | awk '{print $3" used, "$4" free"}')"; sleep 60; done ) &
TMPWATCH=$!

cd "$REPO"
t0=$(date +%s)
TOTAL_STEPS_OVERRIDE=2 VAL_BEFORE_TRAIN=False VAL_FREQ_OVERRIDE=1000000 \
  RESUME=disable EXP_NAME=mini_smoke_prm \
  bash scripts/sciworld/run_sciworld.sh sw_prm_rtg 0 sciworld_4b \
  trainer.save_freq=1000000 \
  2>&1 | tee "$REPO/outputs/rung4_mini_smoke.log"
rc=${PIPESTATUS[0]}
kill $TMPWATCH 2>/dev/null || true
echo "[mini-smoke] rc=$rc total_wall=$(( $(date +%s) - t0 ))s"

grep -E "step:1|step:2" "$REPO/outputs/rung4_mini_smoke.log" | tail -4
cp "$REPO/outputs/rung4_mini_smoke.log" "$HLOG/" 2>/dev/null || true

[ "$rc" -eq 0 ] && status=DONE
echo "==== rung4 mini-smoke end $status $(date -u) ===="
