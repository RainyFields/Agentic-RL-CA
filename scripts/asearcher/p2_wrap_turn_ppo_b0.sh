#!/bin/bash
exec env COND=turn_ppo_b0 MICRO_BSZ_OVERRIDE=1 LOGPROB_MICRO_OVERRIDE=1 \
  bash /home/tiger/xiaoxuan/Agentic-RL-CA/scripts/asearcher/p2_run_worker.sh
