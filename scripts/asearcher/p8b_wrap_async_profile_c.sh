#!/bin/bash
# Attached-launch wrapper: phase C (+D) of the async profile on a fresh pod.
# stdout streams to the mlx client log on the devbox (the pattern that never loses logs).
export START_PHASE=C N_STEPS=3 TURNPPO_STEPS=2
exec bash /home/tiger/xiaoxuan/Agentic-RL-CA/.claude/worktrees/sync-partial-rollout/scripts/asearcher/p8b_async_profile_worker.sh
