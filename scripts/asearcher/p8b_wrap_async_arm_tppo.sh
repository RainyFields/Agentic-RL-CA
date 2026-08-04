#!/bin/bash
export COND=turn_ppo_b0
export DYNBSZ_TOK=20480  # both arms on the profile-validated budget (24576 OOM'd GRPO before)
exec bash /home/tiger/xiaoxuan/Agentic-RL-CA/.claude/worktrees/sync-partial-rollout/scripts/asearcher/p8b_async_arm_worker.sh
