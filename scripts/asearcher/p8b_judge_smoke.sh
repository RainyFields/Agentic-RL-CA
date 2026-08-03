#!/bin/bash
# P8B JUDGE SMOKE: prove gpt-oss-120b actually serves under our vLLM before the real eval
# dumps land. Short and self-terminating — it must not camp on a worker slot.
#
# Checks, in order: weights stage to /tmp, vLLM starts and reports healthy, the chat endpoint
# answers, `reasoning_effort` is accepted or cleanly rejected, and judge_rollouts.py scores a
# synthetic dump end to end. Exits either way; read outputs/p8b_judge_smoke.log for the verdict.
set -uo pipefail
export PYTHONUNBUFFERED=1

REPO=${REPO:-/home/tiger/xiaoxuan/arlca-8b}
export PYTHONPATH="$REPO"
export VENV=/home/tiger/xiaoxuan/envs/agentic-rl-ca
SRC_MODEL="${SRC_MODEL:-/mnt/hdfs/mlsys/models/gpt-oss-120b}"
STAGE_MODEL="${STAGE_MODEL:-/tmp/gpt-oss-120b}"
JUDGE_TP="${JUDGE_TP:-4}"
JUDGE_PORT="${JUDGE_PORT:-8100}"
EVALSET="${EVALSET:-/mnt/hdfs/mlsys/users/xiaoxuan/agentic_rl_ca/data_asearcher_eval/asearcher_eval.parquet}"
HLOG=/mnt/hdfs/mlsys/users/xiaoxuan/agentic_rl_ca/logs/p8b_judge_smoke
OUT_DIR=/tmp/judge_smoke
mkdir -p "$HLOG" "$OUT_DIR"
rm -f "$HLOG/DONE" "$HLOG/FAILED"

status=FAILED
finish() { echo "$status $(date -u +%FT%TZ)" > "$HLOG/$status"
  pkill -9 -f "vllm serve" 2>/dev/null || true
  cp "$OUT_DIR"/smoke.summary.md "$HLOG/" 2>/dev/null || true
  echo "[judge-smoke] exit: $status"; }
trap finish EXIT

echo "==== P8B-JUDGE-SMOKE start $(date -u) on $(hostname) ===="
nvidia-smi -L | head -2
[ -f "$EVALSET" ] || { echo "PREFLIGHT FAIL: evalset"; exit 1; }
source "$VENV/bin/activate"; export VIRTUAL_ENV="$VENV"
python -c 'import vllm, httpx; print("vllm", vllm.__version__)' || { echo "venv incomplete"; exit 1; }

# ---- stage weights (HDFS FUSE is too slow to serve 60GB of shards from directly) ----
if [ ! -f "$STAGE_MODEL/config.json" ]; then
  echo "[stage] copying $SRC_MODEL -> $STAGE_MODEL $(date -u)"
  mkdir -p "$STAGE_MODEL"
  cp -r "$SRC_MODEL"/. "$STAGE_MODEL"/ || { echo "PREFLIGHT FAIL: stage copy"; exit 1; }
fi
du -sh "$STAGE_MODEL"; df -h /tmp | tail -1

# ---- server ----
SRV_LOG="$REPO/outputs/judge_smoke_server.log"; : > "$SRV_LOG"
vllm serve "$STAGE_MODEL" \
  --served-model-name gpt-oss-120b \
  --port "$JUDGE_PORT" \
  --tensor-parallel-size "$JUDGE_TP" \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 >> "$SRV_LOG" 2>&1 &
SRV_PID=$!

up=0
for i in $(seq 1 180); do   # up to 30 min for load
  kill -0 "$SRV_PID" 2>/dev/null || { echo "[judge] SERVER DIED during startup"; tail -60 "$SRV_LOG"; exit 1; }
  sleep 10
  CODE=$(curl -s -m 10 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$JUDGE_PORT/v1/models") || CODE=000
  [ "$CODE" = "200" ] && { echo "[judge] SERVER HEALTHY t=$((i*10))s"; up=1; break; }
done
[ "$up" = 1 ] || { echo "[judge] SERVER HEALTH TIMEOUT"; tail -60 "$SRV_LOG"; exit 1; }

# ---- does this build accept the gpt-oss reasoning_effort extension? ----
for eff in low ""; do
  body='{"model":"gpt-oss-120b","messages":[{"role":"user","content":"Reply with exactly: CORRECT"}],"max_tokens":64,"temperature":0'
  [ -n "$eff" ] && body="$body,\"reasoning_effort\":\"$eff\""
  body="$body}"
  CODE=$(curl -s -m 120 -o /tmp/jr.json -w '%{http_code}' -X POST \
    "http://127.0.0.1:$JUDGE_PORT/v1/chat/completions" -H 'Content-Type: application/json' -d "$body")
  echo "[judge] reasoning_effort='${eff:-<omitted>}' -> HTTP $CODE"
  head -c 400 /tmp/jr.json; echo
done

# ---- synthetic dump -> full judge_rollouts.py pass ----
python - <<'EOF'
import sys; sys.path.insert(0, "/home/tiger/xiaoxuan/arlca-8b")
from agent_system.environments.prompts.search import SEARCH_TEMPLATE_NO_HIS
import pandas as pd, json, uuid, random
random.seed(7)
df = pd.read_parquet("/mnt/hdfs/mlsys/users/xiaoxuan/agentic_rl_ca/data_asearcher_eval/asearcher_eval.parquet")
df = df.groupby("data_source", group_keys=False).head(6)
with open("/tmp/judge_smoke/dump.jsonl", "w") as out:
    for row in df.itertuples(index=False):
        q = row.extra_info["question"]; tgt = [str(t) for t in row.reward_model["ground_truth"]["target"]]
        obs = SEARCH_TEMPLATE_NO_HIS.format(task_description=q); tid = str(uuid.uuid4())
        # half get the gold answer, half a wrong one -> the judge must separate them
        resp = f"<think>ok</think><answer>{tgt[0]}</answer>" if random.random() < 0.5 \
               else "<think>ok</think><answer>Paris</answer>"
        out.write(json.dumps({"traj_uid": tid, "turn_index": 0, "uid": str(uuid.uuid4()),
                              "data_source": row.data_source, "observation": obs,
                              "raw_model_response": resp, "env_won": False,
                              "early_stop_reason": ""}) + "\n")
print("[smoke] synthetic dump written")
EOF

python "$REPO/scripts/asearcher/judge_rollouts.py" \
  --dump /tmp/judge_smoke/dump.jsonl --label smoke --out-dir "$OUT_DIR" --evalset "$EVALSET" \
  --judge-url "http://127.0.0.1:$JUDGE_PORT/v1" --model gpt-oss-120b --concurrency 16
RC=$?
echo "==== P8B-JUDGE-SMOKE judge exit=$RC $(date -u) ===="
[ "$RC" = 0 ] && status=DONE
echo "==== P8B-JUDGE-SMOKE complete ($status) $(date -u) ===="
