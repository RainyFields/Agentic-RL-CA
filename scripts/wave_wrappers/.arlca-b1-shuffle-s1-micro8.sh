#!/bin/bash
# b1_shuffle seed 1 RELAUNCH wrapper (2026-07-14): OOM cycle 100<->119 burning attempts
# (has ckpt 100 banked). MICRO_BSZ/LOGPROB_MICRO=8 for THIS RUN ONLY (memory/throughput;
# PPO_MINI_BATCH=512 locked). Use only if the current worker exhausts 6 attempts.
exec env COND=b1_shuffle SEED=1 PROTOCOL=4turn_think2k MICRO_BSZ_OVERRIDE=8 LOGPROB_MICRO_OVERRIDE=8 bash /home/tiger/xiaoxuan/Agentic-RL-CA/scripts/worker_train.sh
