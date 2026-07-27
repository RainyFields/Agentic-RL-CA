#!/bin/bash
# Wrapper: pass-rate diagnostic on ASearcher-Base keep-set (mlx --envs is broken; bake env here).
# Pod preflight (fail FAST so a relaunch draws a fresh pod; retry a few min first in case
# mounts/driver are still initializing):
# - NVML guard (worker 1020295 / n124-105-045): libnvidia-ml.so missing -> CUDA fine
#   (faiss/vLLM work) but Ray's pynvml autodetect sees 0 GPUs and dies 10min in.
# - HDFS guard (worker 1020296 / n124-104-159): /mnt/hdfs FUSE dead -> MODEL_PATH missing
#   at trainer start and even the FAILED marker write fails.
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
export DIAG_SPLIT=base
exec bash /home/tiger/xiaoxuan/Agentic-RL-CA/scripts/asearcher/diag_worker.sh
