#!/bin/bash
# Wave-2 wrapper (user OK 2026-07-15): b1_shuffle seed 2 on 4turn_think2k (micro8 via protocol).
exec env COND=b1_shuffle SEED=2 PROTOCOL=4turn_think2k bash /home/tiger/xiaoxuan/Agentic-RL-CA/scripts/worker_train.sh
