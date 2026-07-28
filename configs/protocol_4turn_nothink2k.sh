#!/usr/bin/env bash
# Thinking-mode ablation CONTROL: NON-thinking with the SAME 2048-token response cap and the same
# 500-step budget as protocol_4turn_think2k.sh. Pairs 1:1 with the existing 4B thinking runs so the
# ONLY variable is enable_thinking (protocol_4turn.sh's 512 cap would confound mode with budget).
# NB for non-thinking the 2048 cap is non-binding (measured clip 0.000 at 512), so this changes the
# control's behaviour ~none while making the comparison exact.
source "$(dirname "${BASH_SOURCE[0]}")/protocol_4turn.sh"
export ENABLE_THINKING=False
export MAX_RESPONSE_LENGTH=2048
