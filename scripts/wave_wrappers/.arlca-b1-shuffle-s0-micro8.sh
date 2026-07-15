#!/bin/bash
# b1_shuffle seed 0 RELAUNCH wrapper (2026-07-14): OOM livelock pattern — crashes at
# steps ~61/~69/~97, never saved the step-100 checkpoint (each resume replays from 50).
# MICRO_BSZ/LOGPROB_MICRO=8 for THIS RUN ONLY (memory/throughput; PPO_MINI_BATCH=512
# locked, gradient math identical). Use only if the current worker exhausts 6 attempts.
exec env COND=b1_shuffle SEED=0 PROTOCOL=4turn_think2k MICRO_BSZ_OVERRIDE=8 LOGPROB_MICRO_OVERRIDE=8 bash /home/tiger/xiaoxuan/Agentic-RL-CA/scripts/worker_train.sh
