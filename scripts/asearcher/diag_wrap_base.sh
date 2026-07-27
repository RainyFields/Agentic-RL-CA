#!/bin/bash
# Wrapper: pass-rate diagnostic on ASearcher-Base keep-set (mlx --envs is broken; bake env here).
export DIAG_SPLIT=base
exec bash /home/tiger/xiaoxuan/Agentic-RL-CA/scripts/asearcher/diag_worker.sh
