# Copyright 2026 Agentic-RL-CA project. Apache-2.0.
"""CARL estimator core (arXiv 2512.04949; constants + deltas locked in docs/methods_note.md).

Pure tree math, independent of the rollout loop (which lands later, per plan sequencing):
  - node values by unweighted child-mean Bellman recursion (Eq. 10):
        V(leaf) = mean of terminal rewards observed at that leaf
        V(u)    = mean over DISTINCT children c of V(c)
  - a source state u is CRITICAL iff it has >1 distinct child (Eq. 12 context)
  - edge advantage A(e=(u,v)) = V(v) − V(u) for edges with critical source (Eq. 11)
  - non-critical edges are DROPPED from the update set (Eq. 13) — paper behavior;
    `drop_noncritical=False` keeps them with zero advantage (ablation; Seq-MIS-style
    loss exclusion equivalence).

Node identity is the caller's contract: carl node ids must be sha1 hashes of the FULL next
policy-visible tokenized prompt (hardened identity — plan Phase 2b.3); this module treats
them as opaque strings. Rows are deduped by (traj_uid, turn_index) to guard adjust_batch
copy-duplicates.
"""
from collections import defaultdict


def carl_resume_schedule(group_snapshots, n_resume, include_root=False):
    """Round-robin assignment of phase-2 resumes over one group's snapshot entries.

    group_snapshots: list of dicts with at least {"depth": int} (loop adds snap/node).
    Returns the list of n_resume chosen entries (repeats allowed), ordered by the
    round-robin cycle over depth-sorted candidates. Depth-0 (root) states are excluded
    unless include_root — resuming the root is just a fresh resample. Falls back to the
    full list when filtering leaves nothing (a 1-turn trajectory only has its root)."""
    cands = [s for s in group_snapshots if include_root or s["depth"] >= 1]
    if not cands:
        cands = list(group_snapshots)
    if not cands:
        return []
    cands = sorted(cands, key=lambda s: s["depth"])
    return [cands[k % len(cands)] for k in range(n_resume)]


def build_carl_rows(traj_uid, turn_index, src, dst, env_done, env_reward):
    """Assemble compute_carl_edge_advantages input rows from per-row batch columns.

    Terminal rows carry the trajectory's terminal reward: env_done rows, plus the last
    row of any trajectory that never env-finished (early-stop truncation) — its env
    reward (typically 0) is the observed outcome at that leaf."""
    n = len(traj_uid)
    last_row = {}
    for i in range(n):
        k = traj_uid[i]
        if k not in last_row or turn_index[i] > turn_index[last_row[k]]:
            last_row[k] = i
    rows = []
    for i in range(n):
        terminal = bool(env_done[i]) or (last_row[traj_uid[i]] == i)
        rows.append({
            "key": (str(traj_uid[i]), int(turn_index[i])),
            "src": str(src[i]),
            "dst": str(dst[i]),
            "terminal_reward": float(env_reward[i]) if terminal else None,
        })
    return rows


def compute_node_values(edges, leaf_rewards):
    """edges: iterable of (src, dst) pairs (may repeat). leaf_rewards: {node: [r, ...]}
    for nodes where trajectories terminated. Returns {node: V}.

    V(node) = mean over distinct children of V(child); for childless nodes (leaves),
    the mean of terminal rewards recorded there. Nodes that both have children and
    recorded terminal rewards (a truncation/answer collision) average children values
    and terminal rewards as one pooled set of child-outcomes."""
    children = defaultdict(set)
    nodes = set()
    for s, d in edges:
        children[s].add(d)
        nodes.update((s, d))
    nodes.update(leaf_rewards.keys())

    values = {}

    # iterative post-order (the tree can be deep; avoid recursion limits)
    def value_of(root):
        stack = [(root, False)]
        while stack:
            n, expanded = stack.pop()
            if n in values:
                continue
            kids = children.get(n, set())
            if not expanded:
                stack.append((n, True))
                stack.extend((k, False) for k in kids if k not in values)
            else:
                outcomes = [values[k] for k in kids]
                outcomes += [float(r) for r in leaf_rewards.get(n, [])]
                assert outcomes, f"node {n!r} has no children and no terminal rewards"
                values[n] = sum(outcomes) / len(outcomes)
        return values[root]

    for n in nodes:
        value_of(n)
    return values


def compute_carl_edge_advantages(rows, drop_noncritical=True):
    """rows: list of dicts with keys
         key         — (traj_uid, turn_index) identity for dedupe
         src, dst    — node ids (opaque strings)
         terminal_reward — float or None (set on the trajectory's final row)
       Returns (advantages: {key: float}, kept: {key: bool}, metrics: dict).
    """
    # dedupe by (traj_uid, turn_index) — guards DataProto adjust_batch copy-duplicates
    uniq = {}
    for r in rows:
        uniq.setdefault(r["key"], r)
    rows_u = list(uniq.values())

    edges = [(r["src"], r["dst"]) for r in rows_u]
    leaf_rewards = defaultdict(list)
    for r in rows_u:
        if r.get("terminal_reward") is not None:
            leaf_rewards[r["dst"]].append(float(r["terminal_reward"]))

    values = compute_node_values(edges, dict(leaf_rewards))

    children = defaultdict(set)
    for s, d in edges:
        children[s].add(d)
    critical = {n for n, kids in children.items() if len(kids) > 1}

    advantages, kept = {}, {}
    for r in rows_u:
        is_crit = r["src"] in critical
        if is_crit:
            advantages[r["key"]] = values[r["dst"]] - values[r["src"]]
            kept[r["key"]] = True
        else:
            advantages[r["key"]] = 0.0
            kept[r["key"]] = not drop_noncritical

    src_nodes = {r["src"] for r in rows_u}
    metrics = {
        "carl/tree_nodes": len(set(values)),
        "carl/critical_state_frac": (len(critical & src_nodes) / len(src_nodes)) if src_nodes else 0.0,
        "carl/kept_row_frac": (sum(kept.values()) / len(rows_u)) if rows_u else 0.0,
        "carl/n_rows_unique": len(rows_u),
        "carl/n_rows_input": len(rows),
        "carl/n_leaves": len(leaf_rewards),
    }
    return advantages, kept, metrics
