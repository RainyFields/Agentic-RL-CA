#!/bin/bash
# P1 feasibility smoke (ATTACHED — worker lives exactly as long as this script).
# Rung-3 / ASearcher-Base, Qwen3-4B-Instruct-2507, single tool (search), 8-turn, 16k prompt.
# Does BOTH: (a) infra smoke — runs the real 4B/16k rollout+update, catches OOM/crash;
#            (b) answerability — rollout EM (pass-rate) on ASearcher-Base under wiki-18.
# CPU faiss (no conda build; reliable for a smoke). GPU-conda faiss deferred to P2 throughput.
# Writes DONE/FAILED + a summary JSON to $HLOG. No secrets in this file.
set -uo pipefail
export PYTHONUNBUFFERED=1

REPO=/home/tiger/xiaoxuan/Agentic-RL-CA
export VENV=/home/tiger/xiaoxuan/envs/agentic-rl-ca
export MODEL_PATH=/mnt/hdfs/mlsys/models/Qwen3-4B-Instruct-2507
export DATA_DIR=/mnt/hdfs/mlsys/users/xiaoxuan/agentic_rl_ca/data_asearcher_base
export SEARCHR1_DATA=/mnt/hdfs/mlsys/users/xiaoxuan/searchr1/searchr1_data
PROTOCOL=asearcher_8turn_4b
STAGE=/tmp/searchR1
HLOG=/mnt/hdfs/mlsys/users/xiaoxuan/agentic_rl_ca/logs/p1_asearcher_smoke
DUMP="$REPO/outputs/p1_asearcher_smoke/token_grpo"
mkdir -p "$HLOG" "$REPO/outputs" "$STAGE"
rm -f "$HLOG/DONE" "$HLOG/FAILED"

status=FAILED
finish() {
  echo "$status $(date -u +%FT%TZ)" > "$HLOG/$status"
  cp "$REPO/outputs/retriever/retriever.log" "$HLOG/" 2>/dev/null || true
  cp -r "$DUMP" "$HLOG/rollout_dump" 2>/dev/null || true
  cp "$HLOG/../p1_summary.json" "$HLOG/" 2>/dev/null || true
  pkill -f retrieval_server.py 2>/dev/null || true
  echo "[worker] exit: $status"
}
trap finish EXIT

eval "$(grep -E '^export (HF_TOKEN|WANDB_API_KEY)=' /home/tiger/.bashrc)" || true
[ -z "${WANDB_API_KEY:-}" ] && export WANDB_MODE=offline

echo "==== P1 smoke start $(date -u) on $(hostname) ===="; nvidia-smi -L || true

# ---- 1. training venv (idempotent; verified vllm0.11/torch2.8/flash-attn combo) ----
if [ -x "$VENV/bin/python" ]; then
  PYV=$("$VENV/bin/python" -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null)
  [ "$PYV" != "3.11" ] && { echo "venv py$PYV -> recreate"; rm -rf "$VENV"; }
fi
uv venv -p 3.11 "$VENV" || exit 1
source "$VENV/bin/activate"; export VIRTUAL_ENV="$VENV"
python -c 'import vllm' 2>/dev/null || uv pip install "vllm==0.11.0" || exit 1
if ! python -c 'import flash_attn' 2>/dev/null; then
  ABI=$(python -c 'import torch;print(torch._C._GLIBCXX_USE_CXX11_ABI)')
  CPTAG=$(python -c 'import sys;print(f"cp{sys.version_info.major}{sys.version_info.minor}")')
  [ "$ABI" = "True" ] && ABITAG="TRUE" || ABITAG="FALSE"
  uv pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3.post1+cu12torch2.8cxx11abi${ABITAG}-${CPTAG}-${CPTAG}-linux_x86_64.whl" || exit 1
fi
python -c 'import flash_attn' || exit 1
uv pip install -e "$REPO" >/dev/null 2>&1 || uv pip install -e "$REPO" || exit 1
uv pip install gym faiss-cpu >/dev/null 2>&1 || uv pip install gym faiss-cpu || exit 1

# ---- 2. gates: imports + 4B forward ----
python - <<EOF || exit 1
import torch, vllm, transformers, verl, faiss, agent_system
print("imports OK | torch", torch.__version__, "| vllm", vllm.__version__, "| faiss", faiss.__version__, "| cuda", torch.cuda.is_available())
from transformers import AutoModelForCausalLM, AutoTokenizer
m = AutoModelForCausalLM.from_pretrained("$MODEL_PATH", dtype=torch.bfloat16, device_map="cuda")
tok = AutoTokenizer.from_pretrained("$MODEL_PATH")
print("4B forward OK, logits", m(tok("hi", return_tensors="pt").input_ids.cuda()).logits.shape)
del m; torch.cuda.empty_cache()
EOF

# ---- 3. stage index+corpus HDFS -> local /tmp (faster load), serve CPU-faiss retriever ----
[ -f "$STAGE/e5_Flat.index" ] || cp "$SEARCHR1_DATA/e5_Flat.index" "$STAGE/" || exit 1
[ -f "$STAGE/wiki-18.jsonl" ] || cp "$SEARCHR1_DATA/wiki-18.jsonl" "$STAGE/" || exit 1
mkdir -p "$REPO/outputs/retriever"
# Cap BLAS/OMP threads: CPU-faiss flat search segfaults OpenBLAS on many-core nodes
# ("tried to allocate too many memory regions" — buffers exceed OpenBLAS's compiled limit).
( cd "$REPO" \
  && export OMP_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 MKL_NUM_THREADS=32 \
            NUMEXPR_NUM_THREADS=32 VECLIB_MAXIMUM_THREADS=32 \
  && exec python examples/search/retriever/retrieval_server.py \
    --index_path "$STAGE/e5_Flat.index" --corpus_path "$STAGE/wiki-18.jsonl" \
    --topk 3 --retriever_name e5 --retriever_model intfloat/e5-base-v2 --port 8000 \
) > "$REPO/outputs/retriever/retriever.log" 2>&1 &
RETR_PID=$!
echo "[retriever] CPU-faiss serving (pid $RETR_PID); health gate up to 40min..."
export SEARCH_URL="http://127.0.0.1:8000/retrieve"
up=0
for i in $(seq 1 240); do
  kill -0 "$RETR_PID" 2>/dev/null || { echo "[retriever] DIED"; tail -30 "$REPO/outputs/retriever/retriever.log"; exit 1; }
  sleep 10
  CODE=$(curl -s -m 10 -o /tmp/health.json -w '%{http_code}' -X POST "$SEARCH_URL" -H 'Content-Type: application/json' \
      -d '{"queries":["who won the first nobel prize in physics"],"topk":3,"return_scores":true}') || CODE=000
  [ "$CODE" = "200" ] && { echo "[retriever] healthy at t=$((i*10))s"; head -c 300 /tmp/health.json; echo; up=1; break; }
done
[ "$up" = 1 ] || { echo "[retriever] health timeout"; exit 1; }

# ---- 4. SMOKE = short token_grpo run on ASearcher-Base (infra OOM check + rollout pass-rate) ----
# TOY=1 -> tiny batch + rollout dump; override to ~10 steps for a better answerability sample.
# MICRO_BSZ_OVERRIDE=1 : safest for 4B @ 16k (raise later once memory headroom is known).
echo "==== smoke: token_grpo x10 steps, 4B/8turn/16k, ASearcher-Base ===="
TOY=1 TOTAL_STEPS=10 MICRO_BSZ_OVERRIDE=1 LOGPROB_MICRO_OVERRIDE=1 TOY_OUT=p1_asearcher_smoke \
  RESUME=disable \
  bash "$REPO/scripts/run_condition.sh" token_grpo 0 "$PROTOCOL"
RC=$?
echo "smoke run_condition exit=$RC"

# ---- 5. answerability readout from rollout dump (EM/won per terminal turn) ----
python - <<EOF || true
import json, os, glob
paths = glob.glob("$DUMP/rollout_log.jsonl")
rows=[]
for p in paths:
    for l in open(p):
        try: rows.append(json.loads(l))
        except: pass
# terminal turns carry env_reward/won; treat reward>0 (EM hit) as success
term=[r for r in rows if r.get("done") or r.get("won") is not None or r.get("env_reward") is not None]
succ=[r for r in term if (r.get("won") or (r.get("env_reward") or 0)>0)]
n=len(term); s=len(succ)
summ={"n_terminal_rollouts":n,"successes":s,"pass_rate":(s/n if n else None),
      "n_all_turn_rows":len(rows),"smoke_exit":$RC}
json.dump(summ, open("$HLOG/../p1_summary.json","w"), indent=2)
print("ANSWERABILITY:", json.dumps(summ))
EOF

[ "$RC" = "0" ] && status=DONE
echo "==== P1 smoke complete ($status) $(date -u) ===="
