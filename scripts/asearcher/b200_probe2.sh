#!/bin/bash
# B200 probe v2: diagnose CUDA error 803 (driver/userspace mismatch) and test fixes.
# Finding from probe v1 + dev-node cuobjdump: our env's torch/flash-attn wheels ALREADY
# carry sm_100 cubins — the failure is pod-image-level, not wheel-arch.
echo "==== B200 PROBE2 $(date -u) host=$(hostname) ===="
uname -m
echo "--- driver ---"
nvidia-smi | head -4
nvidia-smi -q 2>/dev/null | grep -E "Driver Version|CUDA Version" | head -2
echo "--- libcuda resolution ---"
ldconfig -p | grep -E "libcuda\.so" | head -5
ls -la /usr/local/cuda*/compat/ 2>/dev/null | head -8
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "--- attempt A: torch as-is ---"
source /home/tiger/xiaoxuan/envs/agentic-rl-ca/bin/activate
python -c "import torch; torch.cuda.init(); print('A OK:', torch.cuda.get_device_name(0))" 2>&1 | tail -2
echo "--- attempt B: sanitized LD_LIBRARY_PATH (drop cuda compat dirs) ---"
CLEAN_LD=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v "compat" | paste -sd:)
env LD_LIBRARY_PATH="$CLEAN_LD" python -c "import torch; torch.cuda.init(); print('B OK:', torch.cuda.get_device_name(0))" 2>&1 | tail -2
echo "--- attempt C: force system libcuda via LD_PRELOAD ---"
SYSCUDA=$(ldconfig -p | grep "libcuda.so.1" | grep -v compat | head -1 | awk '{print $NF}')
echo "system libcuda: $SYSCUDA"
[ -n "$SYSCUDA" ] && env LD_PRELOAD="$SYSCUDA" python -c "import torch; torch.cuda.init(); print('C OK:', torch.cuda.get_device_name(0))" 2>&1 | tail -2
echo "--- attempt D (full stack UNDER LD_PRELOAD fix): matmul + GEMM bench + flash_attn + vllm ---"
export LD_PRELOAD="$SYSCUDA"
python - <<'PYEOF' 2>&1 | tail -8
import os, ctypes
try:
    import torch
    torch.cuda.init()
    x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
    print("matmul OK:", float((x @ x).sum()) == float((x @ x).sum()))
    import flash_attn
    from flash_attn import flash_attn_varlen_func
    q = torch.randn(128, 8, 64, device="cuda", dtype=torch.bfloat16)
    cu = torch.tensor([0, 128], device="cuda", dtype=torch.int32)
    o = flash_attn_varlen_func(q, q, q, cu, cu, 128, 128)
    print("flash_attn varlen OK:", tuple(o.shape))
    # quick bf16 GEMM throughput datapoint (8192^3, 20 iters)
    import time
    a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    for _ in range(3): (a @ a)
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(20): (a @ a)
    torch.cuda.synchronize()
    tflops = 20 * 2 * 8192**3 / (time.time() - t0) / 1e12
    print(f"bf16 GEMM: {tflops:.0f} TFLOPS")
    import vllm
    print("vllm import OK:", vllm.__version__)
except Exception as e:
    print("D FAIL:", repr(e)[:300])
PYEOF
echo "==== B200 PROBE2 done $(date -u) ===="
