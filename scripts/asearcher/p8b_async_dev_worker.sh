#!/bin/bash
# Async-rollout dev worker (Phase 0-2, lean async plan 2026-08-04): stage the model,
# mark READY, then hold the pod (worker lifetime == this script's lifetime). All actual
# work (smoke test, collector dev, matched profile) is driven over SSH from the devbox.
set -uo pipefail
export PYTHONUNBUFFERED=1

HDFS_MODEL=/mnt/hdfs/mlsys/models/Qwen3-8B-Base
MARK=/home/tiger/xiaoxuan/Agentic-RL-CA/outputs/p8b_async_dev
mkdir -p "$(dirname "$MARK")"
rm -f "$MARK.READY" "$MARK.FAILED"

echo "==== ASYNC-DEV worker start $(date -u) on $(hostname) ===="
nvidia-smi -L | head -2

[ -f "$HDFS_MODEL/config.json" ] || { echo "PREFLIGHT FAIL: HDFS FUSE"; touch "$MARK.FAILED"; exit 1; }

echo "[stage] model -> /tmp/qwen3-8b-base ($(date -u))"
mkdir -p /tmp/qwen3-8b-base
ls "$HDFS_MODEL"/*.safetensors "$HDFS_MODEL"/*.json "$HDFS_MODEL"/*.txt 2>/dev/null \
  | xargs -P 8 -I{} cp {} /tmp/qwen3-8b-base/
[ -f /tmp/qwen3-8b-base/config.json ] || { echo "model staging failed"; touch "$MARK.FAILED"; exit 1; }
echo "[stage] done ($(date -u))"

echo "$(hostname) $(date -u +%FT%TZ)" > "$MARK.READY"
echo "==== ASYNC-DEV READY — holding pod ===="
while true; do sleep 300; done
