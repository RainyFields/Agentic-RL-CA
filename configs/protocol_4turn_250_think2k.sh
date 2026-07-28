#!/usr/bin/env bash
# Thinking-mode ablation, arm B: THINKING @2048, 250-step budget.
# Identical to Wave-2 (protocol_4turn_250) in every respect except enable_thinking + response
# length, so B-vs-A isolates the thinking flag at a matched 2048-token budget, and
# A-vs-Wave2(512) isolates the response budget within non-thinking. Token "4turn_250_think2k"
# keeps EXP_NAME/ckpt dirs distinct from every existing run.
source "$(dirname "${BASH_SOURCE[0]}")/protocol_4turn_250.sh"
export ENABLE_THINKING=True
export MAX_RESPONSE_LENGTH=2048
