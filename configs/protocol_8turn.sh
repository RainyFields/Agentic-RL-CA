#!/usr/bin/env bash
# 8-turn horizon stress test (pre-registered; PROMOTED to required if shuffle_active_frac < ~30%
# at Wave 1 — plan §Pre-registered decision rules). Same variables as protocol_4turn.sh.

source "$(dirname "${BASH_SOURCE[0]}")/protocol_4turn.sh"

export MAX_STEPS=8
export HISTORY_LENGTH=8
export MAX_PROMPT_LENGTH=8192
