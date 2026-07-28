#!/usr/bin/env bash
# Thinking-mode ablation, arm A: NON-thinking @2048, 250-step budget.
# The control that removes the response-length confound: same generous 2048 budget as the
# thinking arm, thinking flag OFF. Pairs with protocol_4turn_250_think2k.sh (arm B) for a
# single-variable A/B, and with Wave-2's non-thinking@512 for the budget axis.
source "$(dirname "${BASH_SOURCE[0]}")/protocol_4turn_250.sh"
export ENABLE_THINKING=False
export MAX_RESPONSE_LENGTH=2048
