# Copyright 2026 Agentic-RL-CA project. Apache-2.0.
"""Phase 3 — paper-protocol validation metrics (per-TRAJECTORY EM).

The stock `val/{ds}/test_score` averages the reward over turn-ROWS, so trajectories with
more turns weigh more (length bias) and non-terminal rows dilute with zeros. The Search-R1
paper metric is per-question EM. Here: dedupe by traj_uid, EM from the TERMINAL turn's env
reward (step-shaping never lands on the terminal turn, so this stays pure EM under B1 arms).

Emits: val/{ds}/em, val-core/macro_em (mean over datasets), val-core/micro_em (mean over
trajectories), val/{ds}/avg_turns, plus per-trajectory records for the JSONL dump.
CPU-testable (pure numpy).
"""
from collections import defaultdict

import numpy as np


def compute_trajectory_val_metrics(
    traj_uids,
    data_sources,
    turn_indices,
    env_rewards,
    env_dones,
    responses=None,
):
    """All inputs are 1-D per-ROW arrays (one row per active turn). Returns
    (metric_dict, per_traj_records)."""
    rows_of = defaultdict(list)
    n = len(traj_uids)
    for i in range(n):
        rows_of[traj_uids[i]].append(i)

    records = []
    for tuid, idxs in rows_of.items():
        idxs = sorted(idxs, key=lambda i: int(turn_indices[i]))
        last = idxs[-1]
        terminal_done = bool(env_dones[last])
        em = float(env_rewards[last]) if terminal_done else 0.0
        # guard: EM must be binary at the terminal turn (no shaping leakage)
        assert em in (0.0, 1.0), f"non-binary terminal reward {em} for traj {tuid}"
        records.append(
            {
                "traj_uid": str(tuid),
                "data_source": str(data_sources[last]),
                "em": em,
                "turns": len(idxs),
                "terminated": terminal_done,
                "response_last_turn": (responses[last] if responses is not None else None),
            }
        )

    by_ds = defaultdict(list)
    for r in records:
        by_ds[r["data_source"]].append(r)

    metric_dict = {}
    per_ds_em = {}
    for ds, rs in sorted(by_ds.items()):
        ems = [r["em"] for r in rs]
        turns = [r["turns"] for r in rs]
        per_ds_em[ds] = float(np.mean(ems))
        metric_dict[f"val/{ds}/em"] = per_ds_em[ds]
        metric_dict[f"val/{ds}/avg_turns"] = float(np.mean(turns))
        metric_dict[f"val/{ds}/n_traj"] = len(rs)
    if per_ds_em:
        metric_dict["val-core/macro_em"] = float(np.mean(list(per_ds_em.values())))
        metric_dict["val-core/micro_em"] = float(np.mean([r["em"] for r in records]))
        metric_dict["val-core/n_traj"] = len(records)
        metric_dict["val-core/avg_turns"] = float(np.mean([r["turns"] for r in records]))
        metric_dict["val-core/unterminated_frac"] = float(
            np.mean([not r["terminated"] for r in records])
        )
    return metric_dict, records
