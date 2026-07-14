#!/bin/bash
exec env COND=turn_ppo_b0 SEED=0 PROTOCOL=4turn bash /home/tiger/xiaoxuan/Agentic-RL-CA/scripts/worker_train.sh
