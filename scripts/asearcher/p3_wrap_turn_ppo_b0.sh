#!/bin/bash
# Wrapper: P3 turn-PPO gae_turn on filtered ASearcher-Base (mlx --envs broken; bake env + preflight).
for i in $(seq 1 18); do
  nvidia-smi -L 2>/dev/null | grep -q "GPU 0" && break
  echo "[preflight] NVML not up yet ($i/18)"; sleep 10
done
nvidia-smi -L 2>/dev/null | grep -q "GPU 0" \
  || { echo "[preflight] BROKEN POD (no NVML after 3min) — exiting for relaunch"; exit 1; }
for i in $(seq 1 18); do
  [ -d /mnt/hdfs/mlsys/models/Qwen3-4B-Instruct-2507 ] && break
  echo "[preflight] HDFS mount not up yet ($i/18)"; sleep 10
done
[ -d /mnt/hdfs/mlsys/models/Qwen3-4B-Instruct-2507 ] \
  || { echo "[preflight] BROKEN POD (no HDFS after 3min) — exiting for relaunch"; exit 1; }
echo "[preflight] pod OK (NVML + HDFS)"
export COND=turn_ppo_b0
exec bash /home/tiger/xiaoxuan/Agentic-RL-CA/scripts/asearcher/p3_run_worker.sh
