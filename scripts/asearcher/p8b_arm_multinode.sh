#!/bin/bash
# Multi-node (N x 8xH100) arm launcher. mlx runs THIS SAME SCRIPT on every node and does
# NOT start Ray for us (verified by multinode_probe 2026-08-03), so we form the cluster:
#   role 0  -> ray head, wait for all nodes to join, then run the trainer (NNODES=N)
#   role >0 -> ray worker joined to the head, idle until the head's run ends
# Env: COND (required), NNODES (default from ARNOLD_WORKER_NUM), plus the usual arm knobs.
set -uo pipefail
export PYTHONUNBUFFERED=1

COND="${COND:?set COND=...}"
REPO=${REPO:-/home/tiger/xiaoxuan/arlca-8b}
export PYTHONPATH="$REPO"
export VENV=/home/tiger/xiaoxuan/envs/agentic-rl-ca
RAY_PORT="${RAY_PORT:-6379}"

echo "==== P8B-MN [$COND] $(date -u) host=$(hostname) ===="
# --- topology discovery: print everything relevant, then resolve role/head/size ---
echo "--- topology vars ---"
env | grep -aE "^(ARNOLD_(ID|WORKER_NUM|WORKER_[0-9]+_(HOST|PORT)|NUM_)|METIS_|MLX_ROLE)" | sort
ROLE="${ARNOLD_ID:-${METIS_TASK_INDEX:-0}}"
NNODES="${NNODES:-${ARNOLD_WORKER_NUM:-${METIS_WORKER_NUM:-1}}}"
HEAD="${ARNOLD_WORKER_0_HOST:-${METIS_WORKER_0_HOST:-}}"
[ -z "$HEAD" ] && [ "$ROLE" = "0" ] && HEAD="$(hostname -i | awk '{print $1}')"
echo "[topology] role=$ROLE nnodes=$NNODES head=$HEAD"
if [ -z "$HEAD" ] || [ "$NNODES" = "1" ]; then
  echo "MULTINODE FAIL: could not resolve topology (role=$ROLE nnodes=$NNODES head=$HEAD)"
  echo "  -> falling back is NOT automatic; relaunch single-node if this persists"
  exit 1
fi

source "$VENV/bin/activate" 2>/dev/null; export VIRTUAL_ENV="$VENV"
export NNODES N_GPUS_PER_NODE=8

if [ "$ROLE" = "0" ]; then
  echo "[ray] starting head on $HEAD:$RAY_PORT"
  ray start --head --port="$RAY_PORT" --num-gpus=8 --dashboard-host=0.0.0.0 || exit 1
  export RAY_ADDRESS="$HEAD:$RAY_PORT"
  # wait for every node to register (8 GPUs each)
  want=$((NNODES * 8)); ok=0
  for i in $(seq 1 90); do
    have=$(ray status 2>/dev/null | grep -oE "[0-9]+\.0/[0-9]+\.0 GPU" | head -1 | sed -E 's#.*/([0-9]+)\.0 GPU#\1#')
    echo "[ray] cluster GPUs: ${have:-0}/$want"
    [ "${have:-0}" -ge "$want" ] && { ok=1; break; }
    sleep 10
  done
  [ "$ok" = 1 ] || { echo "MULTINODE FAIL: only ${have:-0}/$want GPUs joined"; ray stop --force; exit 1; }
  echo "[ray] cluster formed ($want GPUs) — running the arm"
  COND="$COND" bash "$REPO/scripts/asearcher/p8b_arm_worker.sh"
  RC=$?
  echo "==== P8B-MN [$COND] head done rc=$RC $(date -u) ===="
  ray stop --force >/dev/null 2>&1
  exit $RC
else
  echo "[ray] joining head $HEAD:$RAY_PORT as role $ROLE"
  for i in $(seq 1 60); do
    ray start --address="$HEAD:$RAY_PORT" --num-gpus=8 && break
    echo "[ray] head not up yet (try $i)"; sleep 10
  done
  # idle while the head drives the run; exiting here would tear this node out of the cluster
  while ray status --address="$HEAD:$RAY_PORT" >/dev/null 2>&1; do sleep 60; done
  echo "==== P8B-MN [$COND] worker $ROLE: head cluster gone, exiting $(date -u) ===="
fi
