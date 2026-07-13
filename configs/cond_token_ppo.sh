#!/usr/bin/env bash
# RQ1 arm: token-level PPO (stock verl `gae`) — GAE over every token, token-level value head.
export CONDITION=token_ppo
export ADV_ESTIMATOR=gae
export EXTRA_OVERRIDES=()          # critic auto-enabled for gae
