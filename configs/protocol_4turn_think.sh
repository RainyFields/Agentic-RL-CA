#!/usr/bin/env bash
# 4-turn protocol, THINKING mode (pre-registered fallback for degenerate reasoning:
# toy gate 2026-07-13 showed the non-thinking template kills the <think> block —
# 16/6555 turns with non-empty reasoning). Native Qwen3 thinking + doubled response
# budget. History memory stores only projected actions, so prompt lengths are unchanged.

source "$(dirname "${BASH_SOURCE[0]}")/protocol_4turn.sh"

export ENABLE_THINKING=True
export MAX_RESPONSE_LENGTH=1024
