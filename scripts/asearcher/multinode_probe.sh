#!/bin/bash
# 2-node probe: how does mlx expose node roles, and does a Ray cluster already exist?
echo "==== MULTINODE PROBE $(date -u) host=$(hostname) ===="
echo "--- arnold/metis/ray env ---"
env | grep -iE "^(ARNOLD|METIS|RAY|MLX_)" | sort | head -30
echo "--- gpus ---"; nvidia-smi -L | wc -l
echo "--- is ray already running? ---"
source /home/tiger/xiaoxuan/envs/agentic-rl-ca/bin/activate 2>/dev/null
ray status 2>&1 | head -15
echo "--- ray address env ---"; echo "RAY_ADDRESS=${RAY_ADDRESS:-unset}"
echo "==== PROBE done $(date -u) on $(hostname) ===="
