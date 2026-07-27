#!/bin/bash
# Wrapper: pass-rate diagnostic on ASearcher-Base keep-set (mlx --envs is broken; bake env here).
# NVML guard (worker 1020295, pod n124-105-045, 2026-07-28): some pods come up without
# libnvidia-ml.so — CUDA runtime works (faiss/vLLM fine) but Ray's pynvml GPU autodetect
# sees 0 GPUs and init_workers dies 10min in. Fail FAST so a relaunch gets a fresh pod;
# retry a few min first in case the driver mount is still initializing.
for i in $(seq 1 18); do
  nvidia-smi -L 2>/dev/null | grep -q "GPU 0" && break
  echo "[nvml-guard] NVML not up yet ($i/18)"; sleep 10
done
nvidia-smi -L 2>/dev/null | grep -q "GPU 0" \
  || { echo "[nvml-guard] BROKEN POD (no NVML after 3min) — exiting for relaunch"; exit 1; }
export DIAG_SPLIT=base
exec bash /home/tiger/xiaoxuan/Agentic-RL-CA/scripts/asearcher/diag_worker.sh
