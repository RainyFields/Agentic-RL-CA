#!/bin/bash
# B200 PROFILING (hardware/algorithm comparison, user request 2026-08-04): 8 fast-config
# steps of turn_ppo_b0 then 8 of token_grpo, back to back on ONE 8xB200 worker (the OCI
# pool has exactly 8 B200 free). Timings land in the step logs (timing_s/*) — mined
# afterwards against the H100 arms' steps 1-8 for a position-matched comparison.
#
# Config = each hardware's fastest stable: partial rollout 8/4 + chunked prefill on both;
# DYNBSZ 24576 + critic ON-GPU here (192GB) vs 20480 + critic offload on H100. The DYNBSZ
# delta is deliberate and must be stated in the report (normalize update times per token).
# Everything model/data/rollout-side is identical to the H100 arms.
set -uo pipefail
export PYTHONUNBUFFERED=1

# Pod images in this pool are heterogeneous: pick the first real (>1MB) non-compat libcuda.
LIBCUDA=""
for c in $(ldconfig -p 2>/dev/null | awk '/libcuda\.so\.1/{print $NF}' | grep -v compat) \
         /lib/x86_64-linux-gnu/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so.1; do
  if [ -f "$c" ] && [ "$(stat -Lc %s "$c" 2>/dev/null || echo 0)" -gt 1000000 ]; then LIBCUDA="$c"; break; fi
done
if [ -n "$LIBCUDA" ]; then export LD_PRELOAD="$LIBCUDA"; echo "[libcuda] preloading $LIBCUDA"; else unset LD_PRELOAD; echo "[libcuda] no real system libcuda found — no preload"; fi
export VLLM_ATTENTION_BACKEND=TRITON_ATTN
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1  # useast1b has no egress
export WANDB_MODE=offline

REPO=${REPO:-/home/tiger/xiaoxuan/arlca-8b}
export PYTHONPATH="$REPO"
export VENV=/home/tiger/xiaoxuan/envs/agentic-rl-ca
HDFS_MODEL=/mnt/hdfs/mlsys/models/Qwen3-8B-Base
export DATA_DIR=/mnt/hdfs/mlsys/users/xiaoxuan/agentic_rl_ca/data_asearcher_base
export SEARCHR1_DATA=/mnt/hdfs/mlsys/users/xiaoxuan/searchr1/searchr1_data
export HF_DATASETS_CACHE=/tmp/hf_datasets_cache
PROTOCOL=asearcher_32turn_8b
STAGE=/tmp/searchR1
RETR_ENV=/home/tiger/xiaoxuan/envs/searchr1-retr-conda
HLOG=/mnt/hdfs/mlsys/users/xiaoxuan/agentic_rl_ca/logs/p8b_prof_b200
mkdir -p "$HLOG" "$REPO/outputs" "$STAGE"
rm -f "$HLOG/DONE" "$HLOG/FAILED"

status=FAILED
finish() { echo "$status $(date -u +%FT%TZ)" > "$HLOG/$status"
  kill "${WATCHDOG_PID:-}" 2>/dev/null || true
  cp "$REPO/outputs/retriever/retriever_b200_prof.log" "$HLOG/" 2>/dev/null || true
  pkill -9 -f torch_retrieval_server.py 2>/dev/null || true; echo "[worker] exit: $status"; }
trap finish EXIT

eval "$(grep -E '^export HF_TOKEN=' /home/tiger/.bashrc)" || true
echo "==== B200-PROF start $(date -u) on $(hostname) ===="
uname -m; nvidia-smi -L | head -2; nvidia-smi | sed -n 3p

[ -f "$DATA_DIR/train.parquet" ] || { echo "PREFLIGHT FAIL: HDFS FUSE"; exit 1; }
uv venv -p 3.11 "$VENV" 2>/dev/null; source "$VENV/bin/activate"; export VIRTUAL_ENV="$VENV"
python -c 'import torch; torch.cuda.init(); assert torch.cuda.device_count() >= 8' \
  || { echo "PREFLIGHT FAIL: CUDA init"; exit 1; }

echo "[stage] model -> /tmp ($(date -u))"
mkdir -p /tmp/qwen3-8b-base
time cp "$HDFS_MODEL"/*.safetensors "$HDFS_MODEL"/*.json "$HDFS_MODEL"/*.txt /tmp/qwen3-8b-base/ 2>/dev/null
export MODEL_PATH=/tmp/qwen3-8b-base
[ -f "$MODEL_PATH/config.json" ] || { echo "model staging failed"; exit 1; }
echo "[stage] index -> /tmp ($(date -u))"
[ -f "$STAGE/e5_Flat.index" ] || time cp "$SEARCHR1_DATA/e5_Flat.index" "$STAGE/" || exit 1
[ -f "$STAGE/wiki-18.jsonl" ] || time cp "$SEARCHR1_DATA/wiki-18.jsonl" "$STAGE/" || exit 1

# ---- extract flat index -> fp16 npy (CPU faiss from the conda env; one-time) ----
if [ ! -f "$STAGE/e5_flat_fp16.npy" ]; then
  echo "[extract] faiss index -> fp16 npy ($(date -u))"
  CLEAN=(env -u VIRTUAL_ENV -u PYTHONPATH -u PYTHONHOME -u LD_PRELOAD)
  time "${CLEAN[@]}" "$RETR_ENV/bin/python" "$REPO/scripts/asearcher/extract_flat_index.py" \
    "$STAGE/e5_Flat.index" "$STAGE/e5_flat_fp16.npy" "$STAGE/e5_flat_meta.json" || exit 1
fi

# ---- torch retrieval server (main venv, GPUs) ----
mkdir -p "$REPO/outputs/retriever"
RETR_LOG="$REPO/outputs/retriever/retriever_b200_prof.log"
serve_retriever() {
  ( export TOKENIZERS_PARALLELISM=false \
    && exec python "$REPO/scripts/asearcher/torch_retrieval_server.py" \
      --emb_npy "$STAGE/e5_flat_fp16.npy" --emb_meta "$STAGE/e5_flat_meta.json" \
      --corpus_path "$STAGE/wiki-18.jsonl" --topk 5 --port 8000 \
  ) >> "$RETR_LOG" 2>&1 &
}
: > "$RETR_LOG"; serve_retriever
export SEARCH_URL="http://127.0.0.1:8000/retrieve"
up=0
for i in $(seq 1 120); do
  pgrep -f torch_retrieval_server.py >/dev/null || { echo "[retriever] DIED"; tail -30 "$RETR_LOG"; exit 1; }
  sleep 10
  CODE=$(curl -s -m 10 -o /tmp/h.json -w '%{http_code}' -X POST "$SEARCH_URL" -H 'Content-Type: application/json' \
      -d '{"queries":["health probe"],"topk":5,"return_scores":true}') || CODE=000
  [ "$CODE" = "200" ] && { echo "[retriever] healthy (torch flat) t=$((i*10))s"; up=1; break; }
done
[ "$up" = 1 ] || { echo "[retriever] health timeout"; tail -30 "$RETR_LOG"; exit 1; }
( fails=0
  while true; do
    sleep 120
    CODE=$(curl -s -m 30 -o /dev/null -w '%{http_code}' -X POST "$SEARCH_URL" -H 'Content-Type: application/json' \
        -d '{"queries":["health probe"],"topk":1,"return_scores":true}') || CODE=000
    if [ "$CODE" = "200" ]; then fails=0; continue; fi
    fails=$((fails+1)); echo "[watchdog] torch retriever fail $fails/3 (code $CODE)"
    if [ "$fails" -ge 3 ]; then
      echo "[watchdog] RESTARTING torch retriever $(date -u +%FT%TZ)"
      pkill -9 -f torch_retrieval_server.py 2>/dev/null
      for w in $(seq 1 12); do pgrep -f torch_retrieval_server.py >/dev/null || break; sleep 5; done
      serve_retriever; fails=0
    fi
  done ) &
WATCHDOG_PID=$!

export DYNBSZ=1 DYNBSZ_TOK=24576
export CRITIC_PARAM_OFFLOAD=False CRITIC_OPTIM_OFFLOAD=False
PARTIAL=(+env.partial_rollout_enable=true +env.partial_rollout_cycle_turns=8 +env.partial_rollout_max_age=4)

rc_all=0
for COND in turn_ppo_b0 token_grpo; do
  export EXP_NAME="p8bprofb200_${COND}_s0"
  echo "==== B200-PROF $COND: 8 steps $(date -u) ===="
  TOTAL_STEPS=8 VAL_FREQ=1000000 SAVE_FREQ=1000000 VAL_BEFORE_TRAIN=False RESUME=disable \
    timeout 6h bash "$REPO/scripts/run_condition.sh" "$COND" 0 "$PROTOCOL" "${PARTIAL[@]}"
  RC=$?
  echo "==== B200-PROF $COND exit=$RC $(date -u) ===="
  [ "$RC" = 0 ] || rc_all=1
  # settle between conditions: make sure the previous Ray cluster is fully gone
  ray stop --force >/dev/null 2>&1 || true; sleep 30
done

[ "$rc_all" = 0 ] && status=DONE
echo "==== B200-PROF complete ($status) $(date -u) ===="
