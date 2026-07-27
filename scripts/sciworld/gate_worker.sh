#!/bin/bash
# Rung-4 zero-shot gate worker (ATTACHED). Base Qwen3-4B-Instruct-2507, production
# sciworld wrapper/prompt/truncation, GATE mode: 75 groups x G=5 = 375 episodes,
# temp-1, one rollout, per-turn dump -> gate_analysis.py. Writes DONE/FAILED +
# report to $HLOG. No wandb (single step; keep the ca-rung4-4b project clean).
set -uo pipefail
export PYTHONUNBUFFERED=1

REPO=/home/tiger/xiaoxuan/Agentic-RL-CA
export VENV=/home/tiger/xiaoxuan/envs/agentic-rl-ca
export MODEL_PATH=/mnt/hdfs/mlsys/models/Qwen3-4B-Instruct-2507
export HF_DATASETS_CACHE=/tmp/hf_datasets_cache
HLOG=/mnt/hdfs/mlsys/users/xiaoxuan/agentic_rl_ca/logs/rung4_gate
mkdir -p "$HLOG" "$REPO/outputs"
rm -f "$HLOG/DONE" "$HLOG/FAILED"

status=FAILED
finish() { echo "$status $(date -u +%FT%TZ)" > "$HLOG/$status"; echo "[gate-worker] exit: $status"; }
trap finish EXIT

echo "==== rung4 gate start $(date -u) on $(hostname) ===="; nvidia-smi -L || true

# ---- java (ScienceWorld JVM): PATH -> shared-/home JRE -> apt ----
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

# ---- training venv (shared /home; scienceworld+gymnasium installed from dev node) ----
source "$VENV/bin/activate"
python -c 'import vllm, verl, agent_system, scienceworld, gymnasium' || { echo "venv incomplete"; exit 1; }

eval "$(grep -E '^export (HF_TOKEN|WANDB_API_KEY)=' /home/tiger/.bashrc)" || true
export WANDB_MODE=offline

# ---- single-JVM sanity before spawning 375 actors ----
python - <<'EOF' || exit 1
from scienceworld import ScienceWorldEnv
e = ScienceWorldEnv("", envStepLimit=100)
e.load("melt", 0, "easy"); e.reset()
obs, r, d, i = e.step("look around")
assert len(obs) > 0
e.close(); print("[gate-worker] single-JVM sanity OK")
EOF

# ---- gate run (PRM dump is strictly more informative; reward mode does not affect rollout) ----
cd "$REPO"
t0=$(date +%s)
GATE=1 RESUME=disable bash scripts/sciworld/run_sciworld.sh sw_prm_rtg 0 sciworld_4b \
  2>&1 | tee "$REPO/outputs/sciworld_gate/train.log"
rc=$?
echo "[gate-worker] gate run rc=$rc wall=$(( $(date +%s) - t0 ))s"

DUMP="$REPO/outputs/sciworld_gate/prm_rtg/rollout_log.jsonl"
[ -s "$DUMP" ] || { echo "FATAL: no rollout dump at $DUMP"; exit 1; }

python3 scripts/sciworld/gate_analysis.py "$DUMP" --out "$REPO/outputs/sciworld_gate/report" \
  2>&1 | tee "$REPO/outputs/sciworld_gate/report_stdout.txt"

cp -r "$REPO/outputs/sciworld_gate/report" "$HLOG/" || true
zstd -q -f "$DUMP" -o "$HLOG/rollout_log.jsonl.zst" 2>/dev/null || cp "$DUMP" "$HLOG/" || true
cp "$REPO/outputs/sciworld_gate/train.log" "$HLOG/" 2>/dev/null || true

status=DONE
echo "==== rung4 gate done $(date -u) ===="
