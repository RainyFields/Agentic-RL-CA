#!/bin/bash
export COND=hcapo_ans
export DYNBSZ_TOK=20480  # >= max_seq_len 17408; the budget the GRPO arm finished on (24576 OOM'd)
exec bash /home/tiger/xiaoxuan/arlca-8b/scripts/asearcher/p8b_arm_worker.sh
