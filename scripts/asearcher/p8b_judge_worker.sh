#!/bin/bash
# P8B JUDGE: serve gpt-oss-120b (decision A3) on one worker and MBE-score eval rollout dumps.
# Generation already happened in p8b_eval_worker.sh; this is a pure post-hoc scoring pass, so
# every checkpoint's dump can be judged on a single worker back to back.
#
#   DUMPS="grpo=/mnt/hdfs/.../rollout_eval_grpo.jsonl.zst turn_ppo=/mnt/hdfs/.../rollout_eval_turn_ppo.jsonl.zst" \
#     bash p8b_judge_worker.sh
#
# Optional: JUDGE_TP (default 4), JUDGE_MODEL_PATH, JUDGE_CONCURRENCY (default 64),
#           REASONING_EFFORT (default low), EVALSET.
set -uo pipefail
export PYTHONUNBUFFERED=1

DUMPS="${DUMPS:?set DUMPS=\"label=path [label=path ...]\"}"
REPO=${REPO:-/home/tiger/xiaoxuan/arlca-8b}
export PYTHONPATH="$REPO"
export VENV=/home/tiger/xiaoxuan/envs/agentic-rl-ca
JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-/mnt/hdfs/mlsys/models/gpt-oss-120b}"
JUDGE_TP="${JUDGE_TP:-4}"
JUDGE_PORT="${JUDGE_PORT:-8100}"
JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-64}"
REASONING_EFFORT="${REASONING_EFFORT:-low}"
EVALSET="${EVALSET:-/mnt/hdfs/mlsys/users/xiaoxuan/agentic_rl_ca/data_asearcher_eval/asearcher_eval.parquet}"
HLOG=/mnt/hdfs/mlsys/users/xiaoxuan/agentic_rl_ca/logs/p8b_judge
OUT_DIR="$REPO/outputs/judge"
mkdir -p "$HLOG" "$OUT_DIR"
rm -f "$HLOG/DONE" "$HLOG/FAILED"

status=FAILED
finish() { echo "$status $(date -u +%FT%TZ)" > "$HLOG/$status"
  pkill -9 -f "vllm serve" 2>/dev/null || true; echo "[judge-worker] exit: $status"; }
trap finish EXIT

echo "==== P8B-JUDGE start $(date -u) on $(hostname) ===="
[ -f "$EVALSET" ] || { echo "PREFLIGHT FAIL: evalset $EVALSET"; exit 1; }
[ -d "$JUDGE_MODEL_PATH" ] || { echo "PREFLIGHT FAIL: judge model $JUDGE_MODEL_PATH"; exit 1; }
source "$VENV/bin/activate"; export VIRTUAL_ENV="$VENV"
python -c 'import vllm, httpx' || { echo "venv incomplete"; exit 1; }

# ---- stage weights to local disk (HDFS FUSE is far too slow to serve shards from) ----
STAGE_MODEL="${STAGE_MODEL:-/tmp/$(basename "$JUDGE_MODEL_PATH")}"
if [ ! -f "$STAGE_MODEL/config.json" ]; then
  echo "[judge] staging $JUDGE_MODEL_PATH -> $STAGE_MODEL $(date -u)"
  mkdir -p "$STAGE_MODEL"
  cp -r "$JUDGE_MODEL_PATH"/. "$STAGE_MODEL"/ || { echo "PREFLIGHT FAIL: stage copy"; exit 1; }
fi
df -h /tmp | tail -1

# ---- judge server ----
SRV_LOG="$REPO/outputs/judge_server.log"
: > "$SRV_LOG"
vllm serve "$STAGE_MODEL" \
  --served-model-name gpt-oss-120b \
  --port "$JUDGE_PORT" \
  --tensor-parallel-size "$JUDGE_TP" \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --disable-log-requests >> "$SRV_LOG" 2>&1 &
SRV_PID=$!

up=0
for i in $(seq 1 240); do
  kill -0 "$SRV_PID" 2>/dev/null || { echo "[judge] server died during startup"; tail -40 "$SRV_LOG"; exit 1; }
  sleep 10
  CODE=$(curl -s -m 10 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$JUDGE_PORT/v1/models") || CODE=000
  [ "$CODE" = "200" ] && { echo "[judge] server healthy t=$((i*10))s"; up=1; break; }
done
[ "$up" = 1 ] || { echo "[judge] server health timeout"; tail -40 "$SRV_LOG"; exit 1; }

# ---- one scoring pass per dump ----
rc_all=0
for spec in $DUMPS; do
  LABEL="${spec%%=*}"; DUMP="${spec#*=}"
  if [ ! -f "$DUMP" ]; then echo "[judge] SKIP $LABEL: no dump at $DUMP"; rc_all=1; continue; fi
  echo "==== P8B-JUDGE $LABEL $(date -u) ===="
  python "$REPO/scripts/asearcher/judge_rollouts.py" \
    --dump "$DUMP" --label "$LABEL" --out-dir "$OUT_DIR" --evalset "$EVALSET" \
    --judge-url "http://127.0.0.1:$JUDGE_PORT/v1" --model gpt-oss-120b \
    --concurrency "$JUDGE_CONCURRENCY" --reasoning-effort "$REASONING_EFFORT"
  rc=$?; [ "$rc" = 0 ] || rc_all=1
  echo "==== P8B-JUDGE $LABEL exit=$rc $(date -u) ===="
  cp "$OUT_DIR/$LABEL".summary.{json,md} "$HLOG/" 2>/dev/null || true
  gzip -c "$OUT_DIR/$LABEL.items.jsonl" > "$HLOG/$LABEL.items.jsonl.gz" 2>/dev/null || true
done

[ "$rc_all" = 0 ] && status=DONE
echo "==== P8B-JUDGE complete ($status) $(date -u) ===="
