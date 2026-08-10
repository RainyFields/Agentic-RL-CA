#!/bin/bash
# Rubric-judge the step-75 GRPO eval dump (Phase-1 launch gates: flag precision + p_t
# calibration). Capped at 2000 trajectories — enough for both gates; the full-dump pass
# happens after the prompt is validated.
export DUMPS="grpo_s75=/mnt/hdfs/mlsys/users/xiaoxuan/agentic_rl_ca/logs/p8b_eval_grpo_s75/rollout_eval_grpo_s75.jsonl.zst"
export GOLDSET=/mnt/hdfs/mlsys/users/xiaoxuan/agentic_rl_ca/data_asearcher_eval/asearcher_eval.parquet
export MAX_TRAJS=2000
exec bash /home/tiger/xiaoxuan/arlca-8b-judge/scripts/asearcher/p8b_rubric_judge_worker.sh
