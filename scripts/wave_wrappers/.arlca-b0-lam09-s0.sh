#!/bin/bash
# Lambda-sweep stage 1 (docs/plan_lambda_sweep.md; grid {0.5,0.8} from grill Q3 +
# {0.9,0.95} added by user 2026-07-17 for dose-response/comparability): turn_ppo_b0
# seed 0, LAM_OVERRIDE=0.9, 4turn_think2k. LAUNCH ONLY per gate/queue decision.
exec env COND=turn_ppo_b0 SEED=0 PROTOCOL=4turn_think2k LAM_OVERRIDE=0.9 bash /home/tiger/xiaoxuan/Agentic-RL-CA/scripts/worker_train.sh
