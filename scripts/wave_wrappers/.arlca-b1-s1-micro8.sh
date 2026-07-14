#!/bin/bash
# b1 seed 1 RELAUNCH wrapper (2026-07-14, decision_log): OOM livelock — 5 attempts never
# passed step ~75 on MICRO_BSZ=16. MICRO_BSZ/LOGPROB_MICRO halved to 8 for THIS RUN ONLY
# (throughput/memory only; PPO_MINI_BATCH=512 locked, gradient math identical).
# resume_mode=auto continues from the HDFS step-50 checkpoint.
exec env COND=b1 SEED=1 PROTOCOL=4turn_think2k MICRO_BSZ_OVERRIDE=8 LOGPROB_MICRO_OVERRIDE=8 bash /home/tiger/xiaoxuan/Agentic-RL-CA/scripts/worker_train.sh
