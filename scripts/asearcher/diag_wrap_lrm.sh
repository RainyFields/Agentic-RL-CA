#!/bin/bash
# Wrapper: pass-rate diagnostic on ASearcher-LRM keep-set (mlx --envs is broken; bake env here).
export DIAG_SPLIT=lrm
exec bash /home/tiger/xiaoxuan/Agentic-RL-CA/scripts/asearcher/diag_worker.sh
