#!/bin/bash
# B200 feasibility probe: shared-storage mounts + Hopper-wheel behavior on Blackwell.
echo "==== B200 PROBE $(date -u) host=$(hostname) ===="
uname -m; nvidia-smi -L | head -10
echo "--- shared /home ---"; ls -d /home/tiger/xiaoxuan && echo HOME_OK || echo HOME_MISSING
echo "--- HDFS mounts ---"
ls /mnt/hdfs/mlsys/models/Qwen3-8B-Base/config.json 2>&1
ls /mnt/hdfs/mlsys/users/xiaoxuan/agentic_rl_ca/data_asearcher_base/train.parquet 2>&1
ls -la /mnt/hdfs/mlsys/users/xiaoxuan/searchr1/searchr1_data/e5_Flat.index 2>&1
echo "--- HDFS read speed (100MB) ---"
timeout 180 dd if=/mnt/hdfs/mlsys/users/xiaoxuan/searchr1/searchr1_data/e5_Flat.index of=/dev/null bs=4M count=25 2>&1 | tail -1
echo "--- venv (Hopper wheels) on Blackwell ---"
source /home/tiger/xiaoxuan/envs/agentic-rl-ca/bin/activate 2>/dev/null || echo "VENV ACTIVATE FAIL"
python - <<'PYEOF'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("device:", torch.cuda.get_device_name(0), "cap:", torch.cuda.get_device_capability(0))
x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
y = (x @ x).sum().item()
print("bf16 matmul OK:", y == y)
for mod in ("flash_attn", "vllm"):
    try:
        m = __import__(mod); print(mod, getattr(m, "__version__", "?"), "IMPORT OK")
    except Exception as e:
        print(mod, "FAIL:", repr(e)[:200])
PYEOF
echo "==== B200 PROBE done $(date -u) ===="
