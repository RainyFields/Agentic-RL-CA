#!/usr/bin/env bash
# 4-turn thinking protocol, response 2048 (truncation fix: 1024 capped 15-21% of turns with
# no action emitted — pre-registered rule: raise lengths, never accept truncation).
source "$(dirname "${BASH_SOURCE[0]}")/protocol_4turn_think.sh"
export MAX_RESPONSE_LENGTH=2048
