#!/bin/bash
exec env MODEL_PATH=/mnt/hdfs/mlsys/models/Qwen3-1.7B PROTOCOL=4turn_think TOY_OUT=search_toy_think17 BASE_LABEL=wave0_base_think17 bash /home/tiger/xiaoxuan/Agentic-RL-CA/scripts/worker_phase1_toygate.sh
