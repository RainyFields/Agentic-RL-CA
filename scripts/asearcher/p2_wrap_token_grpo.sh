#!/bin/bash
exec env COND=token_grpo MICRO_BSZ_OVERRIDE=1 LOGPROB_MICRO_OVERRIDE=1 \
  bash /home/tiger/xiaoxuan/Agentic-RL-CA/scripts/asearcher/p2_run_worker.sh
