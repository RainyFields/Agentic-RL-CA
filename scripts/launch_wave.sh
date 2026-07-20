#!/usr/bin/env bash
# Launch a wave of training workers (one 8xH100 worker per condition/seed).
#   scripts/launch_wave.sh wave1            # pre-registered Wave-1 slate (8 runs)
#   scripts/launch_wave.sh custom b1:0 b1:1 gigpo:0
# Requires the user's per-wave OK (CONFIRM_WORKER=1). Counts existing workers first —
# quota is ~6 concurrent 8-GPU workers on mlsys_inference (a 7th queues silently).
# Each launch client is setsid-protected (client death == worker teardown).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

WAVE="${1:?usage: launch_wave.sh <wave1|custom> [cond:seed ...]}"
PROTOCOL="${PROTOCOL:-4turn_think2k}"
case "$WAVE" in
  wave1)
    # Plan §Phase 4 Wave 1: RQ3 triad at equal seeds + RQ1/RQ2 start (8 workers).
    RUNS=(turn_ppo_b0:0 turn_ppo_b0:1 b1:0 b1:1 b1_shuffle:0 b1_shuffle:1 token_ppo:0 gigpo:0)
    ;;
  custom)
    RUNS=("${@:2}")
    ;;
  *) echo "unknown wave: $WAVE" >&2; exit 2 ;;
esac

N_EXISTING=$(timeout 120 mlx worker list 2>/dev/null | tail -n +2 | grep -c . || echo 0)
echo "[wave] existing workers: $N_EXISTING; requested: ${#RUNS[@]} (quota ~6 concurrent)"

confirm_worker "8 per run (${#RUNS[@]} runs)" "H100-SXM-80GB" "days (500 steps + resume loop)" \
  "arlca-$WAVE" "$HDFS_PROJECT/checkpoints" "worker_train.sh per condition"

LAUNCH_DIR="$HOME/xiaoxuan/worker_logs/launches"
mkdir -p "$LAUNCH_DIR" "$REPO_DIR/scripts/wave_wrappers"

for run in "${RUNS[@]}"; do
  cond="${run%%:*}"; seed="${run##*:}"
  # ALIAS_TAG (e.g. "250") namespaces alias + wrapper filename so a new wave can reuse a
  # cond:seed without overwriting the previous wave's wrapper (a live worker crash-resume
  # re-execs its wrapper — overwriting one would silently switch its protocol).
  alias="arlca-${ALIAS_TAG:+${ALIAS_TAG}-}${cond//_/-}-s${seed}"
  wrapper="$REPO_DIR/scripts/wave_wrappers/.${alias}.sh"
  cat > "$wrapper" <<EOF
#!/bin/bash
exec env COND=$cond SEED=$seed PROTOCOL=${PROTOCOL:-4turn_think2k} bash $REPO_DIR/scripts/worker_train.sh
EOF
  chmod +x "$wrapper"
  LOG="$LAUNCH_DIR/$(date +%Y%m%d_%H%M%S)_${alias}.log"
  setsid nohup mlx worker launch \
    --resourcetype arnold --usergroup mlsys_inference \
    --cluster cloudnative-maliva \
    --queuename compute-598-aliyun.va-cloudnative-aigcp-mlsys.inference-guarantee \
    --gpu 8 --type H100-SXM-80GB --alias "$alias" --no-input \
    -- bash "$wrapper" > "$LOG" 2>&1 &
  echo "[wave] launched $alias (client pid $!, log $LOG)"
  printf '%s\t%s\t%s\t8xH100 mlsys_inference\t-\t-\tRUNNING\t-\t%s\n' \
    "$(date '+%Y-%m-%d %H:%M %Z')" "$alias" "Agentic-RL-CA $WAVE: $cond seed $seed ($PROTOCOL)" "$LOG" \
    >> "$HOME/xiaoxuan/worker_logs/workers.tsv"
  sleep 5
done
echo "[wave] all launched — monitor via worker_logs/launches/*.log and W&B project $WANDB_PROJECT"
