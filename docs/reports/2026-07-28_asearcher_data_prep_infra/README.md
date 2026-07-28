# ASearcher (rung 3): data preparation, context-length setup, and training infra

`asearcher_data_prep_infra.pdf` — summary of the 4B feasibility campaign's data pipeline
(raw ASearcher Base/LRM → parquet → closed-book stage-1 filter → in-setup pass-rate stage-2
filter), the 16k/8-turn context budget and its P1/P2 evidence, and the worker/retriever
infrastructure including the 2026-07-28 reliability hardening (pod preflight, SIGKILL
watchdog, mamba-free restarts, BLAS thread caps).

Rebuild: `tectonic asearcher_data_prep_infra.tex`

Source scripts: `scripts/asearcher/` (converter, closed-book filter, diag parquet builder,
pass-rate scorer, worker scripts + wrappers), `configs/protocol_asearcher_8turn_4b.sh`,
`scripts/run_condition.sh`.

Data/results on HDFS: `.../users/xiaoxuan/agentic_rl_ca/{data_asearcher_raw, data_asearcher_base,
logs/closed_book_filter, logs/passrate_diag_*}`.
