#!/bin/bash
exec env COND=token_ppo SEED=0 PROTOCOL=4turn_250 bash /home/tiger/xiaoxuan/Agentic-RL-CA/scripts/worker_train.sh
