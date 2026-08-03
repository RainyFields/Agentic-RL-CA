#!/bin/bash
# Eval the finished GRPO arm (val 0.520 @ step 75) over the ASearcher eval suite.
# Wrapper because mlx `--envs` is broken on this queue — the vars have to be baked in.
export CKPT=/mnt/hdfs/mlsys/users/xiaoxuan/agentic_rl_ca/checkpoints/token_grpo_qwen3-8b-base_32turn_fast75_s0/global_step_75
export LABEL=grpo_s75
export COND=token_grpo
exec bash /home/tiger/xiaoxuan/arlca-8b/scripts/asearcher/p8b_eval_worker.sh
