#!/usr/bin/env bash
# Wave-2 protocol: identical to protocol_4turn.sh (512-resp non-thinking, Search-R1
# lengths, matched budget) with a 250-step total budget (user-locked 2026-07-20).
# Protocol token "4turn_250" stamps EXP_NAME/ckpt dirs so runs never collide with
# 500-step 4turn runs.
source "$(dirname "${BASH_SOURCE[0]}")/protocol_4turn.sh"
export TOTAL_STEPS=250
