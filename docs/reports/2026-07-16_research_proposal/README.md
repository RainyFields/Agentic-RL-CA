# Research proposal — When does intermediate reward actually help? (2026-07-16)

`proposal.pdf` — A4 LaTeX research proposal covering the project's hypotheses (H1–H5 / RQ1–RQ5),
the chosen experiment (7-arm credit-assignment comparison with the B1-shuffle density control),
the training dataset (Search-R1: NQ+HotpotQA train, wiki-18 retrieval, 51,713-question 7-dataset
eval), and preliminary Wave 0–1 results, with a reward-model framing (implicit critics /
group-relative baselines vs explicit step rewards).

## Rebuild

```bash
cd assets && python3 extract_metrics.py && python3 build_figures.py
cd .. && tectonic proposal.tex
```

- `assets/extract_metrics.py` — parses per-step metrics out of the launch logs in
  `~/xiaoxuan/worker_logs/launches/` (last-occurrence-per-step; pre/post-relaunch logs merged)
  into `assets/csv/<run>.csv`.
- `assets/build_figures.py` — builds `assets/figs/fig_{learning_curves,reward_turns,tripwire,final_bars}.{png,pdf}`
  in the scientific-figure-making house style.
- Example trajectory in the appendix: extracted from the wave-0 full-set eval log
  (`20260715_213615_arlca-wave0-eval.log`, triviaqa dump, EM=1).

Data snapshot date: 2026-07-16 ~10:45 PDT (gigpo/token_ppo and third seeds still training;
bars marked `*` in the figure).
