#!/bin/bash
# RUBRIC JUDGE: serve gpt-oss-120b and run per-turn rubric judging (judge_turns.py) over
# rollout dumps. Same serve pattern as p8b_judge_worker.sh; differences: rubric client,
# 32k model len (a 32-turn trajectory renders to ~25k tok), goldset per dump.
#
#   DUMPS="grpo_s75=/mnt/hdfs/.../rollout_eval_grpo_s75.jsonl.zst" \
#   GOLDSET=/mnt/hdfs/.../asearcher_eval.parquet bash p8b_rubric_judge_worker.sh
#
# Optional: JUDGE_TP (default 4), JUDGE_CONCURRENCY (default 32), MAX_TRAJS (default 0=all),
#           DOC_CHARS (default 600), REASONING_EFFORT (default low).
set -uo pipefail
export PYTHONUNBUFFERED=1

DUMPS="${DUMPS:?set DUMPS=\"label=path [label=path ...]\"}"
GOLDSET="${GOLDSET:?set GOLDSET=parquet path}"
REPO=${REPO:-/home/tiger/xiaoxuan/arlca-8b-judge}
export PYTHONPATH="$REPO"
export VENV=/home/tiger/xiaoxuan/envs/agentic-rl-ca
JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-/mnt/hdfs/mlsys/models/gpt-oss-120b}"
JUDGE_TP="${JUDGE_TP:-4}"
JUDGE_PORT="${JUDGE_PORT:-8100}"
JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-32}"
MAX_TRAJS="${MAX_TRAJS:-0}"
DOC_CHARS="${DOC_CHARS:-600}"
REASONING_EFFORT="${REASONING_EFFORT:-low}"
HLOG=/mnt/hdfs/mlsys/users/xiaoxuan/agentic_rl_ca/logs/p8b_rubric_judge
OUT_DIR="$REPO/outputs/judge_turns"
mkdir -p "$HLOG" "$OUT_DIR"
rm -f "$HLOG/DONE" "$HLOG/FAILED"

status=FAILED
finish() { echo "$status $(date -u +%FT%TZ)" > "$HLOG/$status"
  pkill -9 -f "vllm serve" 2>/dev/null || true; echo "[rubric-worker] exit: $status"; }
trap finish EXIT

echo "==== RUBRIC-JUDGE start $(date -u) on $(hostname) ===="
[ -f "$GOLDSET" ] || { echo "PREFLIGHT FAIL: goldset $GOLDSET"; exit 1; }
[ -d "$JUDGE_MODEL_PATH" ] || { echo "PREFLIGHT FAIL: judge model $JUDGE_MODEL_PATH"; exit 1; }
source "$VENV/bin/activate"; export VIRTUAL_ENV="$VENV"
python -c 'import vllm, httpx' || { echo "venv incomplete"; exit 1; }

# ---- stage weights to local disk (skip the 61G metal/ dir; marker written post-copy) ----
STAGE_MODEL="${STAGE_MODEL:-/tmp/$(basename "$JUDGE_MODEL_PATH")}"
if [ ! -f "$STAGE_MODEL/.staged" ]; then
  echo "[rubric] staging $JUDGE_MODEL_PATH -> $STAGE_MODEL $(date -u)"
  mkdir -p "$STAGE_MODEL"
  ( cd "$JUDGE_MODEL_PATH" && ls | grep -v "^metal$" ) | \
    xargs -P 8 -I{} cp -r "$JUDGE_MODEL_PATH/{}" "$STAGE_MODEL/" \
    || { echo "PREFLIGHT FAIL: stage copy"; exit 1; }
  touch "$STAGE_MODEL/.staged"
  echo "[rubric] staged $(du -sh --apparent-size "$STAGE_MODEL" | cut -f1) $(date -u)"
fi
df -h /tmp | tail -1

# ---- judge server ----
SRV_LOG="$REPO/outputs/judge_server.log"
: > "$SRV_LOG"
vllm serve "$STAGE_MODEL" \
  --served-model-name gpt-oss-120b \
  --port "$JUDGE_PORT" \
  --tensor-parallel-size "$JUDGE_TP" \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --disable-log-requests >> "$SRV_LOG" 2>&1 &
SRV_PID=$!

up=0
for i in $(seq 1 240); do
  kill -0 "$SRV_PID" 2>/dev/null || { echo "[rubric] server died during startup"; tail -40 "$SRV_LOG"; exit 1; }
  sleep 10
  CODE=$(curl -s -m 10 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$JUDGE_PORT/v1/models") || CODE=000
  [ "$CODE" = "200" ] && { echo "[rubric] server healthy t=$((i*10))s"; up=1; break; }
done
[ "$up" = 1 ] || { echo "[rubric] server health timeout"; tail -40 "$SRV_LOG"; exit 1; }

# ---- one rubric pass per dump ----
rc_all=0
for spec in $DUMPS; do
  LABEL="${spec%%=*}"; DUMP="${spec#*=}"
  if [ ! -f "$DUMP" ]; then echo "[rubric] SKIP $LABEL: no dump at $DUMP"; rc_all=1; continue; fi
  echo "==== RUBRIC-JUDGE $LABEL $(date -u) ===="
  python "$REPO/scripts/asearcher/judge_turns.py" \
    --dump "$DUMP" --label "$LABEL" --out-dir "$OUT_DIR" --goldset "$GOLDSET" \
    --judge-url "http://127.0.0.1:$JUDGE_PORT/v1" --model gpt-oss-120b \
    --concurrency "$JUDGE_CONCURRENCY" --doc-chars "$DOC_CHARS" \
    --max-trajs "$MAX_TRAJS" --reasoning-effort "$REASONING_EFFORT"
  rc=$?; [ "$rc" = 0 ] || rc_all=1
  echo "==== RUBRIC-JUDGE $LABEL exit=$rc $(date -u) ===="
  gzip -c "$OUT_DIR/$LABEL.rubrics.jsonl" > "$HLOG/$LABEL.rubrics.jsonl.gz" 2>/dev/null || true
done

[ "$rc_all" = 0 ] && status=DONE
echo "==== RUBRIC-JUDGE complete ($status) $(date -u) ===="
