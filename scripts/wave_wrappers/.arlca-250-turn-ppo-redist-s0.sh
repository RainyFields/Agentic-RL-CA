#!/bin/bash
exec env COND=turn_ppo_redist SEED=0 PROTOCOL=4turn_250 bash /home/tiger/xiaoxuan/Agentic-RL-CA/scripts/worker_train.sh
