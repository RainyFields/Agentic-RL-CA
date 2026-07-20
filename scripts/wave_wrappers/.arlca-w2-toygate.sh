#!/bin/bash
# Wave-2 toy gate: validate the two NEW code paths (turn_grpo estimator, turn_ppo_redist
# Dirichlet redistribution) on the 4turn non-thinking/512 protocol before fleet launch.
exec env PROTOCOL=4turn TOY_OUT=search_toy_wave2 SKIP_BASE_EVAL=1 \
  TOYGATE_CONDS="turn_grpo turn_ppo_redist" \
  bash /home/tiger/xiaoxuan/Agentic-RL-CA/scripts/worker_phase1_toygate.sh
