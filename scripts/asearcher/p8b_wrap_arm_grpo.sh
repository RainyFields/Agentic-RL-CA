#!/bin/bash
export COND=token_grpo
export DYNBSZ_TOK=16384  # post-OOM fallback (cycle ~35 at 24576 with 17-turn inflation)
exec bash /home/tiger/xiaoxuan/arlca-8b/scripts/asearcher/p8b_arm_worker.sh
