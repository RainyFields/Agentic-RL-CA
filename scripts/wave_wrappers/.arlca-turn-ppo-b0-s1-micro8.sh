#!/bin/bash
# turn_ppo_b0 seed 1 RELAUNCH wrapper (2026-07-14): organic OOMs at steps ~93/~135
# (ckpt 100 banked). MICRO_BSZ/LOGPROB_MICRO=8 for THIS RUN ONLY (memory/throughput;
# PPO_MINI_BATCH=512 locked). Use only if the current worker exhausts 6 attempts.
exec env COND=turn_ppo_b0 SEED=1 PROTOCOL=4turn_think2k MICRO_BSZ_OVERRIDE=8 LOGPROB_MICRO_OVERRIDE=8 bash /home/tiger/xiaoxuan/Agentic-RL-CA/scripts/worker_train.sh
