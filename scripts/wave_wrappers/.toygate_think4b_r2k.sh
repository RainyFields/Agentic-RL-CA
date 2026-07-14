#!/bin/bash
exec env MODEL_PATH=/mnt/hdfs/mlsys/models/Qwen3-4B PROTOCOL=4turn_think2k TOY_OUT=search_toy_think4b_r2k BASE_LABEL=wave0_base_think4b_r2k bash /home/tiger/xiaoxuan/Agentic-RL-CA/scripts/worker_phase1_toygate.sh
