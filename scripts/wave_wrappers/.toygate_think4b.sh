#!/bin/bash
exec env MODEL_PATH=/mnt/hdfs/mlsys/models/Qwen3-4B PROTOCOL=4turn_think TOY_OUT=search_toy_think4b BASE_LABEL=wave0_base_think4b bash /home/tiger/xiaoxuan/Agentic-RL-CA/scripts/worker_phase1_toygate.sh
