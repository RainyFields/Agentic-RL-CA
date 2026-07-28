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
import re
from collections import defaultdict

import numpy as np


_ANSWER_RE = re.compile(r"<answer>.*?</answer>", re.DOTALL)


def compute_trajectory_val_metrics(
    traj_uids,
    data_sources,
    turn_indices,
    env_rewards,
    env_dones,
    responses=None,
    tool_callings=None,
    resp_tok_counts=None,
    item_indices=None,
    questions=None,
    parse_statuses=None,
    max_turns=None,
):
    """All inputs are 1-D per-ROW arrays (one row per active turn). Returns
    (metric_dict, per_traj_records).

    The optional per-row arrays below enrich the per-trajectory JSONL for the Phase-3.3
    horizon-stratified eval (§4 schema). They are only wired up when the caller sets
    trainer.validation_data_dir (the full-set eval), so training-time validation — which
    passes none of them — produces byte-identical metric_dict and the original record keys:
      tool_callings   -> n_search_calls  (executed <search> queries; per-traj constant)
      resp_tok_counts -> tokens_generated (summed over the trajectory's turns)
      item_indices    -> index            (stable dataset join key -> hop/type annotations)
      questions       -> question         (raw text; secondary join key)
      parse_statuses  -> terminal parse status ('answer' == emitted a well-formed answer)
      max_turns       -> turn cap, for the truncated_at_cap flag
    """
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
        resp_last = responses[last] if responses is not None else None
        turns = len(idxs)
        # did the policy emit a well-formed <answer>…</answer> on the terminal turn?
        if parse_statuses is not None:
            answered = str(parse_statuses[last]) == "answer"
        elif resp_last is not None:
            answered = bool(_ANSWER_RE.search(resp_last))
        else:
            answered = terminal_done and em == 1.0
        rec = {
            "traj_uid": str(tuid),
            "data_source": str(data_sources[last]),
            "em": em,
            "turns": turns,
            "terminated": terminal_done,
            "response_last_turn": resp_last,
        }
        if item_indices is not None:
            _ix = item_indices[last]
            rec["index"] = int(_ix) if _ix is not None else None
        if questions is not None:
            _q = questions[last]
            rec["question"] = str(_q) if _q is not None else None
        if tool_callings is not None:
            rec["n_search_calls"] = int(tool_callings[last])
        if resp_tok_counts is not None:
            rec["tokens_generated"] = int(sum(int(resp_tok_counts[i]) for i in idxs))
        if any(x is not None for x in (tool_callings, resp_tok_counts, parse_statuses, max_turns)):
            rec["answered"] = bool(answered)
            # truncated_at_cap: ran to the turn cap without ever emitting a valid answer
            cap = int(max_turns) if max_turns is not None else None
            rec["truncated_at_cap"] = bool((cap is None or turns >= cap) and not answered)
        records.append(rec)

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
