#!/bin/bash
# Phase 1 worker job (ATTACHED — worker lives exactly as long as this script).
# 1. build/reuse project venv (uv; vllm 0.11.0 -> torch 2.8.0; flash-attn prebuilt wheel)
# 2. verification gates (imports + tiny model forward)
# 3. retriever standup (co-located, health-gated)
# 4. base-model val_2048 eval (Wave-0 gate: macro-EM floor) — also end-to-end test of the
#    Phase-3 eval port
# 5. toy gate: 7 conditions x tiny run + trajectory dumps + gate metrics
# Writes DONE/FAILED marker to $HLOG. No secrets in this file.
set -uo pipefail
export PYTHONUNBUFFERED=1

REPO=/home/tiger/xiaoxuan/Agentic-RL-CA
export VENV=/home/tiger/xiaoxuan/envs/agentic-rl-ca
HLOG=/mnt/hdfs/mlsys/users/xiaoxuan/agentic_rl_ca/logs/phase1
mkdir -p "$HLOG" "$REPO/outputs"
rm -f "$HLOG/DONE" "$HLOG/FAILED"

status=FAILED
finish() {
  echo "$status $(date -u +%FT%TZ)" > "$HLOG/$status"
  cp -r "$REPO/outputs/search_toy" "$HLOG/" 2>/dev/null || true
  cp -r "$REPO/outputs/eval_full/wave0_base" "$HLOG/" 2>/dev/null || true
  pkill -f retrieval_server.py 2>/dev/null || true
  echo "[worker] exit: $status"
}
trap finish EXIT

eval "$(grep -E '^export (HF_TOKEN|WANDB_API_KEY)=' /home/tiger/.bashrc)" || true
[ -z "${WANDB_API_KEY:-}" ] && export WANDB_MODE=offline   # never hang on a wandb login prompt

echo "==== Phase1 worker start $(date -u) on $(hostname) ===="
nvidia-smi -L || true

# ---- 1. env (idempotent; project venv, NOT SP6's) ----
FA_TRUE="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3.post1+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
FA_FALSE="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3.post1+cu12torch2.8cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
uv venv "$VENV" || { echo "uv venv failed"; exit 1; }
source "$VENV/bin/activate"
export VIRTUAL_ENV="$VENV"
uv pip install "vllm==0.11.0" || { echo "vllm install failed"; exit 1; }
ABI=$(python -c 'import torch;print(torch._C._GLIBCXX_USE_CXX11_ABI)')
[ "$ABI" = "True" ] && FA="$FA_TRUE" || FA="$FA_FALSE"
uv pip install "$FA" || echo "WARN flash-attn failed (training needs it — will surface at toy gate)"
uv pip install -e "$REPO" || { echo "verl-agent -e failed"; exit 1; }
uv pip install gym || { echo "gym failed"; exit 1; }

# ---- 2. gates ----
python - <<'EOF' || exit 1
import torch, vllm, transformers, verl, gym
import agent_system, credit_assignment
print("imports OK | torch", torch.__version__, "| vllm", vllm.__version__,
      "| transformers", transformers.__version__, "| cuda", torch.cuda.is_available())
from transformers import AutoModelForCausalLM, AutoTokenizer
m = AutoModelForCausalLM.from_pretrained("/mnt/hdfs/mlsys/models/Qwen3-1.7B",
                                         dtype=torch.bfloat16, device_map="cuda")
tok = AutoTokenizer.from_pretrained("/mnt/hdfs/mlsys/models/Qwen3-1.7B")
ids = tok("hello", return_tensors="pt").input_ids.cuda()
print("forward OK, logits", m(ids).logits.shape)
del m; torch.cuda.empty_cache()
EOF

# ---- 3. retriever (backgrounds itself inside the script; health-gated) ----
bash "$REPO/scripts/retriever_serve.sh" || { echo "retriever failed"; exit 1; }

# ---- 4. base-model Wave-0 gate eval on val_2048 (once, greedy) ----
VAL_FILES="$REPO/data/searchR1_processed_direct/val_2048.parquet" EVAL_VAL_BATCH=1024 \
  bash "$REPO/scripts/eval_search_full.sh" /mnt/hdfs/mlsys/models/Qwen3-1.7B wave0_base token_grpo 4turn \
  || { echo "base-model val failed"; exit 1; }

# ---- 5. toy gate (7 conditions) ----
bash "$REPO/scripts/run_toy_gate.sh" || { echo "toy gate failed"; exit 1; }

status=DONE
echo "==== Phase1 worker complete $(date -u) ===="
