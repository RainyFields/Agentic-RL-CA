#!/bin/bash
# Bottleneck-diagnosis run (user request 2026-08-05): ONE async GRPO step with
# engine stat logging (Running/Waiting/KV%/tok/s every ~10s per engine) + driver
# scheduler snapshots every 2s. Fresh EXP_NAME so nothing resumes.
export START_PHASE=C N_STEPS=1 TURNPPO_STEPS=0
export EXP_NAME_OVERRIDE=asyncdiag
export PROFILE_EXTRA="actor_rollout_ref.rollout.disable_log_stats=False +env.async_rollout_snap_s=2"
export ASYNC_DIAG_DIR=/home/tiger/xiaoxuan/Agentic-RL-CA/outputs/async_diag
exec bash /home/tiger/xiaoxuan/Agentic-RL-CA/.claude/worktrees/sync-partial-rollout/scripts/asearcher/p8b_async_profile_worker.sh
