# Copyright 2026 Agentic-RL-CA project. Apache-2.0.
"""Phase 3b.2 — pure planning/aggregation for the credit-alignment diagnostic runner.

The GPU runner (diag_runner.py) samples, for each turn prefix s_t of a base trajectory,
K outcome-only continuations from the SAME checkpoint that generated the trajectory
(plan §3b: continuations under any other policy make dV-hat a value under the wrong
policy). These helpers are CPU-testable: job planning (which (traj, depth) prefixes get
K vs the K=16 noise-check budget) and V-hat aggregation into diagnostic.build_pairs
input format.
"""
import random


def plan_continuation_jobs(traj_depths, k, k_check=None, k_check_frac=0.0, seed=0):
    """traj_depths: {traj_id: [depth, ...]} — snapshot depths available per trajectory
    (depth 0 = root, so (0,1) turn-pairs exist). Returns a list of job dicts
    {"traj": id, "depth": d, "k": int}, deterministic given seed.

    A k_check_frac fraction of TRAJECTORIES (not prefixes — the noise check needs the
    whole trajectory at K=16 so its pairs are comparable) is upgraded to k_check."""
    rng = random.Random(seed)
    traj_ids = sorted(traj_depths)
    n_check = int(round(len(traj_ids) * k_check_frac)) if k_check else 0
    check_set = set(rng.sample(traj_ids, n_check)) if n_check else set()
    jobs = []
    for t in traj_ids:
        kk = k_check if t in check_set else k
        jobs.extend({"traj": t, "depth": int(d), "k": kk} for d in sorted(traj_depths[t]))
    return jobs


def aggregate_vhat(job_results):
    """job_results: iterable of {"traj", "depth", "rewards": [float, ...]} (one entry per
    job; rewards are the K terminal outcomes). Returns
    {traj: {depth: {"vhat": mean, "k": n, "rewards": [...]}}}."""
    out = {}
    for r in job_results:
        rewards = [float(x) for x in r["rewards"]]
        assert rewards, f"job {(r['traj'], r['depth'])} has no continuation rewards"
        out.setdefault(r["traj"], {})[int(r["depth"])] = {
            "vhat": sum(rewards) / len(rewards),
            "k": len(rewards),
            "rewards": rewards,
        }
    return out


def prefix_values_for_pairs(vhat_by_traj, terminal_rewards=None):
    """Convert aggregate_vhat output into diagnostic.build_pairs' prefix_values format:
    flat {(traj, depth): value}. If terminal_rewards ({traj: float}) is given, the
    terminal state s_T is appended at depth = max(depth)+1 with its OBSERVED outcome —
    the last turn's dV-hat is then R(tau) - V-hat(s_{T-1})."""
    out = {}
    for t, per_depth in vhat_by_traj.items():
        depths = sorted(int(d) for d in per_depth)
        for d in depths:
            out[(t, d)] = per_depth[d]["vhat"]
        if terminal_rewards is not None and t in terminal_rewards and depths:
            out[(t, depths[-1] + 1)] = float(terminal_rewards[t])
    return out
