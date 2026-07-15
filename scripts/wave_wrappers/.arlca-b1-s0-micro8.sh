#!/bin/bash
# b1 seed 0 RELAUNCH wrapper (2026-07-14): organic OOMs at ~74/~75 on attempts 2-3 of
# the 16:31 relaunch (arlca-b1-s0b) — same pre-ckpt-100 pattern as b1-s1.
# MICRO_BSZ/LOGPROB_MICRO=8 for THIS RUN ONLY. Use only if the worker exhausts 6 attempts.
exec env COND=b1 SEED=0 PROTOCOL=4turn_think2k MICRO_BSZ_OVERRIDE=8 LOGPROB_MICRO_OVERRIDE=8 bash /home/tiger/xiaoxuan/Agentic-RL-CA/scripts/worker_train.sh
