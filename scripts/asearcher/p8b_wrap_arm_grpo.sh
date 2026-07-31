#!/bin/bash
export COND=token_grpo
export DYNBSZ_TOK=20480  # post-OOM fallback; MUST be >= MAX_PROMPT+MAX_RESPONSE=17408 (16384 asserted out)
exec bash /home/tiger/xiaoxuan/arlca-8b/scripts/asearcher/p8b_arm_worker.sh
