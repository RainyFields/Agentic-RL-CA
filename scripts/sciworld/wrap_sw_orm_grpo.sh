#!/bin/bash
# mlx wrapper: bake COND (mlx --envs is broken; memory merlin-parallel-worker-quota)
export COND=sw_orm_grpo
exec bash /home/tiger/xiaoxuan/Agentic-RL-CA/scripts/sciworld/arm_worker.sh
