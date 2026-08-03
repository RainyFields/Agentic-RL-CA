#!/bin/bash
# 2-node NCCL transport + bandwidth probe. Answers: does cross-node FSDP use RDMA/IB
# (fast) or TCP sockets (slow enough to make multi-node pointless for us)?
set -uo pipefail
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET
ROLE="${ARNOLD_ID:-0}"; WORLD="${ARNOLD_WORKER_NUM:-1}"
HEAD="${ARNOLD_WORKER_0_HOST:-}"; PORT="${ARNOLD_WORKER_0_PORT:-10100}"
echo "==== NCCL PROBE role=$ROLE world=$WORLD head=$HEAD port=$PORT host=$(hostname) $(date -u) ===="
echo "--- fabric inventory ---"
ls /dev/infiniband 2>/dev/null && echo "(infiniband devices present)" || echo "(no /dev/infiniband)"
ibv_devinfo 2>/dev/null | grep -E "hca_id|state|rate" | head -8 || echo "(no ibv_devinfo)"
ip -o link show | awk '{print $2, $NF}' | head -8
source /home/tiger/xiaoxuan/envs/agentic-rl-ca/bin/activate
MASTER_ADDR="$HEAD" MASTER_PORT="$PORT" RANK="$ROLE" WORLD_SIZE="$WORLD" python - <<'PY'
import os, time, torch, torch.distributed as dist
torch.cuda.set_device(0)
dist.init_process_group("nccl", init_method="env://",
                        rank=int(os.environ["RANK"]), world_size=int(os.environ["WORLD_SIZE"]))
r = dist.get_rank()
for mb in (64, 256, 1024):
    x = torch.ones(mb * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
    for _ in range(3):
        dist.all_reduce(x)
    torch.cuda.synchronize(); dist.barrier()
    t0 = time.time()
    n = 10
    for _ in range(n):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / n
    # ring allreduce moves 2*(N-1)/N * size per rank
    ws = dist.get_world_size()
    busbw = (x.numel() * 4) * 2 * (ws - 1) / ws / dt / 1e9
    if r == 0:
        print(f"[nccl] allreduce {mb:5d} MB: {dt*1000:8.1f} ms  busbw {busbw:6.1f} GB/s", flush=True)
dist.destroy_process_group()
PY
echo "==== NCCL PROBE done $(date -u) ===="
