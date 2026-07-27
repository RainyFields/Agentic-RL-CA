#!/bin/bash
# Wrapper: pass-rate diagnostic on ASearcher-LRM keep-set (mlx --envs is broken; bake env here).
# Pod preflight — see diag_wrap_base.sh for the two observed broken-pod modes (NVML, HDFS).
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
export DIAG_SPLIT=lrm
export DIAG_VAL_BATCH=1280
exec bash /home/tiger/xiaoxuan/Agentic-RL-CA/scripts/asearcher/diag_worker2.sh
