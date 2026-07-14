#!/usr/bin/env bash
# 8-turn horizon stress test on the LOCKED main protocol (1.7B think @2048).
# PROMOTED to required if shuffle_active_frac < ~30% at Wave 1.
source "$(dirname "${BASH_SOURCE[0]}")/protocol_4turn_think2k.sh"
export MAX_STEPS=8
export HISTORY_LENGTH=8
export MAX_PROMPT_LENGTH=8192
