#!/bin/bash
# token_ppo seed 0 RELAUNCH wrapper (2026-07-14): worker exhausted 6 attempts (3 organic
# fragmentation OOMs at steps ~100/102/125 + 3 allocator-incident startup deaths).
# MICRO_BSZ/LOGPROB_MICRO=8 for THIS RUN ONLY (throughput/memory; PPO_MINI_BATCH=512
# locked, math identical). resume_mode=auto continues from HDFS ckpt step 100.
# Launch when a quota slot frees — AFTER gigpo-s0 (RQ2, zero steps banked) per plan
# front-loading. decision_log 2026-07-14.
exec env COND=token_ppo SEED=0 PROTOCOL=4turn_think2k MICRO_BSZ_OVERRIDE=8 LOGPROB_MICRO_OVERRIDE=8 bash /home/tiger/xiaoxuan/Agentic-RL-CA/scripts/worker_train.sh
