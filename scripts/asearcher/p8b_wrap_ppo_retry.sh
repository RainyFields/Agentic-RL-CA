#!/bin/bash
# Retry wrap: re-run ONLY the vanilla turnPPO profiling config under the fixed memory
# posture (KV 0.5 + critic offload now default in the 8B protocol). mlx cannot pass
# env or quoted `bash -c` args reliably — hence this file (p3 wrap pattern).
export ONLY_CFG=ppo_vanilla
exec bash /home/tiger/xiaoxuan/arlca-8b/scripts/asearcher/p8b_profile_worker.sh
