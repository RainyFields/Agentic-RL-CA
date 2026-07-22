# ScienceWorld inspection (Wave-0 M5) — report package

**Deliverable:** `scienceworld_inspection.pdf` — environment inspection: sandbox, tasks/evaluation,
trial counts, rollout speed, a full gold trajectory, expert-data availability, published baselines,
and challenges. Source `scienceworld_inspection.tex` (compile with `tectonic scienceworld_inspection.tex`).

Companion structure audit: `audit_report.md` (task-family map, oracle-reward semantics, per-family
turn caps, isolation probes).

## Rerun
```
source ~/xiaoxuan/envs/sw_inspect/bin/activate   # scienceworld 1.2.3, OpenJDK 17
python scripts/scienceworld/audit_structure.py --out <dir>/assets --vars-per-task 3 --step-limit 300
python scripts/scienceworld/analyze_audit.py   --audit <dir>/assets
python scripts/scienceworld/bench_and_probe.py --out <dir>/assets
tectonic scienceworld_inspection.tex
```

## assets/
- `task_map.csv`, `families.csv`, `turn_caps.csv` — 30-task enumeration, family grouping, proposed caps
- `gold_traces.jsonl` — 90 gold replays (per-step action/reward/score/obs-head)
- `subgoal_semantics.md` — oracle reward analysis
- `speed_bench.json` — rollout-speed measurements (this node)
- `sandbox_probe.json` — action templates, valid-combo counts, initial observation
- `full_traj_*.json` — two complete gold trajectories with untruncated observations
- `horizon_hist.png` — gold-path length by family
- `html/` — 12 annotated trajectory dumps (green = subgoal firing)

## Key numbers
- Engine: ~40k-line Scala JAR on Java (py4j), one JVM process per env instance; parallelism is process-level.
- 30 tasks / 10 families / 7,200 total variations (paper); train/dev/test ≈ 50/25/25 per task.
- Dense cumulative 0–100 subgoal score; per-step reward = score delta; magnitudes heterogeneous (1–72);
  wrong `focus on` → −100 + instant termination. Score never leaks into observation text.
- Rollout speed on this node: ~14 steps/s (~65–73 ms/step), 0.22 s per episode load+reset.
- Expert data: env-generated gold action sequences (free) — the rung-4 SFT source.
- Baselines: paper best DRRN 0.17/1.0 (larger LMs worse); modern LLM agents ETO 65–74, SwiftSage 84.7,
  SFT/BC 53–74 (strong) — GiGPO/verl-agent & RAGEN lineages do NOT report ScienceWorld.
