#!/bin/bash
# Queue #4.5 v3 (recovery after 2026-07-17 disk-full wedged the v2 critic merge):
# same as v2 but (a) SKIPS run_diag for labels whose prefix_values.json already has
# "states" (v2 collected them), (b) always re-merges the critic fresh (v2 left a
# corrupt sparse critic_merged for s1_step100). See decision_log 2026-07-17.
set -uo pipefail
REPO=/home/tiger/xiaoxuan/Agentic-RL-CA
CK_S1=/mnt/hdfs/mlsys/users/xiaoxuan/agentic_rl_ca/checkpoints/turn_ppo_b0_qwen3-1.7b_4turn_think2k_s1
CK_S0=/mnt/hdfs/mlsys/users/xiaoxuan/agentic_rl_ca/checkpoints/turn_ppo_b0_qwen3-1.7b_4turn_think2k_s0
bash "$REPO/scripts/retriever_serve.sh" || { echo "[diag-worker] exit: FAILED (retriever)"; exit 1; }
source "$REPO/scripts/_common.sh"
activate_env
cd "$REPO"
rc=0
run_one() {  # run_one <ckpt_dir> <label>
  local CKPT="$1" LABEL="$2" OUT="$REPO/outputs/diag/$2"
  if [ -f "$OUT/prefix_values.json" ] && grep -q '"states"' "$OUT/prefix_values.json"; then
    echo "[diag-worker] $LABEL diag SKIP (states present)"
  else
    [ -f "$OUT/prefix_values.json" ] && mv "$OUT/prefix_values.json" "$OUT/prefix_values_v1.json"
    bash "$REPO/scripts/run_diag.sh" "$CKPT" "$LABEL" 4turn_think2k || { echo "[diag-worker] $LABEL diag FAILED"; return 1; }
  fi
  rm -rf "$OUT/critic_merged"
  python3 "$REPO/scripts/model_merger.py" merge --backend fsdp \
    --local_dir "$CKPT/critic" --target_dir "$OUT/critic_merged" \
    || { echo "[diag-worker] $LABEL critic-merge FAILED"; return 1; }
  python3 -m credit_assignment.critic_score --critic "$OUT/critic_merged" \
    --prefix-values "$OUT/prefix_values.json" --out "$OUT/critic_values.json" \
    || { echo "[diag-worker] $LABEL critic-score FAILED"; return 1; }
  echo "[diag-worker] $LABEL DONE"
}
run_one "$CK_S1/global_step_100" b0_s1_step100 || rc=1
run_one "$CK_S1/global_step_200" b0_s1_step200 || rc=1
run_one "$CK_S1/global_step_500" b0_s1_step500 || rc=1
run_one "$CK_S0/global_step_500" b0_s0_step500 || rc=1
if [ $rc -eq 0 ]; then
  python3 -m credit_assignment.lambda_gate --diag-root "$REPO/outputs/diag" \
    --labels b0_s1_step100 b0_s1_step200 b0_s1_step500 b0_s0_step500 \
    --gate-labels b0_s1_step100 b0_s1_step200 b0_s1_step500 \
    --out "$REPO/outputs/diag/lambda_gate.json" || rc=1
fi
if [ $rc -eq 0 ]; then
  echo "[diag-worker] exit: DONE (f8a-v3 full: diag+critic+gate)"
else
  echo "[diag-worker] exit: FAILED (f8a-v3 — see per-step lines)"
  exit 1
fi
