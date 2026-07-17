#!/bin/bash
# Queue #4.5 v2 (state-dump rerun, 2026-07-16): F8a with critic scoring end to end.
# v1 runs collected vhat only — diag_runner didn't persist prefix states, so critic
# V_phi couldn't be scored (decision_log 2026-07-16). This rerun regenerates the diag
# WITH states on B0-s1 {100,200,500} + B0-s0 {500}, merges each critic, scores states,
# and evaluates the lambda-sweep gate (docs/plan_lambda_sweep.md §2a).
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
  [ -f "$OUT/prefix_values.json" ] && mv "$OUT/prefix_values.json" "$OUT/prefix_values_v1.json"
  bash "$REPO/scripts/run_diag.sh" "$CKPT" "$LABEL" 4turn_think2k || { echo "[diag-worker] $LABEL diag FAILED"; return 1; }
  if [ ! -f "$OUT/critic_merged/config.json" ]; then
    python3 "$REPO/scripts/model_merger.py" merge --backend fsdp \
      --local_dir "$CKPT/critic" --target_dir "$OUT/critic_merged" \
      || { echo "[diag-worker] $LABEL critic-merge FAILED"; return 1; }
  fi
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
  echo "[diag-worker] exit: DONE (f8a-v2 full: diag+critic+gate)"
else
  echo "[diag-worker] exit: FAILED (f8a-v2 — see per-step lines)"
  exit 1
fi
