#!/bin/bash
# Wrapper: P8B profiling worker (mlx --envs broken; bake preflight + exec from the
# asearcher-8b-32turn worktree so the gated env changes apply).
for i in $(seq 1 18); do
  nvidia-smi -L 2>/dev/null | grep -q "GPU 0" && break
  echo "[preflight] NVML not up yet ($i/18)"; sleep 10
done
nvidia-smi -L 2>/dev/null | grep -q "GPU 0" \
  || { echo "[preflight] BROKEN POD (no NVML after 3min) — exiting for relaunch"; exit 1; }
for i in $(seq 1 18); do
  [ -d /mnt/hdfs/mlsys/models/Qwen3-8B-Base ] && break
  echo "[preflight] HDFS mount not up yet ($i/18)"; sleep 10
done
[ -d /mnt/hdfs/mlsys/models/Qwen3-8B-Base ] \
  || { echo "[preflight] BROKEN POD (no HDFS after 3min) — exiting for relaunch"; exit 1; }
echo "[preflight] pod OK (NVML + HDFS)"
exec bash /home/tiger/xiaoxuan/Agentic-RL-CA/.claude/worktrees/asearcher-8b-32turn/scripts/asearcher/p8b_profile_worker.sh
