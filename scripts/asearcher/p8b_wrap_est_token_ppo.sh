#!/bin/bash
# Resume of the 08-03 estimator arm (died step ~42-47, ckpt only @25) — runs the
# SAME code the arm trained on (e5a8956) from a pinned checkout; SAVE_FREQ=5
# bounds pod-reclaim replay (c606ec8 lesson).
export COND=token_ppo
export DYNBSZ_TOK=20480  # >= max_seq_len 17408; the budget the GRPO arm finished on (24576 OOM'd)
export REPO=/home/tiger/xiaoxuan/arlca-8b-est
export SAVE_FREQ=5
exec bash $REPO/scripts/asearcher/p8b_arm_worker.sh
