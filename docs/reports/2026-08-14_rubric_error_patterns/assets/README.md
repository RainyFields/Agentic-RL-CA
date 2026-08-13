# Assets — rubric error-pattern paper (2026-08-14)

Each figure ships as vector PDF (used by the paper), PNG (preview), and the exact data
behind it as CSV. Regenerate everything with `uv run ../make_paper_figs.py`.

| Figure | Data | Content |
|---|---|---|
| `fig_searches_vs_sufficiency.{pdf,png}` | `fig_searches_vs_sufficiency.csv` | Distribution of searches issued per trajectory vs searches to judge sufficiency (first p_t ≥ 0.9), clipped at 17 |
| `fig_criterion_fail.{pdf,png}` | `fig_criterion_fail.csv` | Failure rate of the 5 rubric criteria by search-turn position (0–9) |
| `fig_readiness.{pdf,png}` | `fig_readiness.csv` | Mean judge readiness p_t by turn position, split by final EM outcome |
| `fig_auc.{pdf,png}` | `fig_auc.csv` | Rank AUC of each judge signal (trajectory means, p_final, fewer-searches) for predicting EM |

Source data: `outputs/judge_turns/grpo_s75.rubrics.jsonl` (2,000 trajectories; also archived
gzipped on HDFS at `logs/p8b_rubric_judge/grpo_s75.rubrics.jsonl.gz`).
