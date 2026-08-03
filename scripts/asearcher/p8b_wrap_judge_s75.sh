#!/bin/bash
# Judge the finished-arm eval dumps (GRPO + turn-PPO @ step 75) in one worker pass.
# Judging is generation-free, so every checkpoint's dump is scored back to back on one slot.
# Add `label=path` pairs here as the remaining arms (hcapo_ans, token_ppo, gigpo) finish.
export DUMPS="grpo_s75=/mnt/hdfs/mlsys/users/xiaoxuan/agentic_rl_ca/logs/p8b_eval_grpo_s75/rollout_eval_grpo_s75.jsonl.zst turnppo_s75=/mnt/hdfs/mlsys/users/xiaoxuan/agentic_rl_ca/logs/p8b_eval_turnppo_s75/rollout_eval_turnppo_s75.jsonl.zst"
exec bash /home/tiger/xiaoxuan/arlca-8b/scripts/asearcher/p8b_judge_worker.sh
