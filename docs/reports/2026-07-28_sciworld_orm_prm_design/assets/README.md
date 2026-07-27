# Probe scripts — rerunnable evidence for the design doc

Env: `~/xiaoxuan/envs/sw_inspect/bin/python` (scienceworld 1.2.3 + OpenJDK 17). CPU-only,
each script runs in minutes on the dev node.

- `probe_optional.py` — replays gold paths on 8 tasks (one per family flavor), then parses
  `getGoalProgressStr()`. Shows: at score==100, ALL Sequential subgoals are complete while
  Unordered are partial (e.g. boil 3/3 seq, 5/15 unordered) → sequence completion forces 100
  regardless of unordered credit; no optional-marked entries inside the Sequential section.
- `seq_counts.py` — counts Sequential vs Unordered subgoals for all 30 tasks (variation 0,
  `easy`). The N_seq ∈ {1..4} finding (9 tasks have N_seq=1) that killed ordered-only PRM.
- `show_seq.py` — prints the Sequential subgoal descriptions for 5 example tasks, showing
  that for N_seq=1 tasks the single required state is the terminal answer action
  ("focus on the correct answer box", "focus on the grown fruit").

Aggregate-vs-sequential firing comparison in the design doc additionally uses
`../2026-07-23_scienceworld_audit/assets/gold_traces.jsonl` (n_subgoal_firings field).
