#!/bin/bash
# Attached-launch wrapper: phase D ONLY (async turn-PPO 2 steps) on its own H100 worker,
# parallel with the phase-C worker.
export START_PHASE=D N_STEPS=3 TURNPPO_STEPS=2
exec bash /home/tiger/xiaoxuan/Agentic-RL-CA/.claude/worktrees/sync-partial-rollout/scripts/asearcher/p8b_async_profile_worker.sh
