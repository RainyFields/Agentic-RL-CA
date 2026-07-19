# Copyright 2026 Agentic-RL-CA project. Apache-2.0.
"""Phase 3b.3 — per-method credit-alignment GPU runner (offline; NEVER a training signal).

Extends diag_runner.py from single-trajectory critic scoring to a GROUPED rollout so a
critic-FREE, group-relative method's ASSIGNED per-turn advantage A_t can be reconstructed the
way training computes it (needs the G-trajectory group + anchor states), then correlated against
the ideal per-turn credit dV-hat_t (MC continuation-value delta) in method_align.

Flow, per checkpoint:
  (A) roll N prompts x G trajectories from the checkpoint (uid = prompt, traj_uid = trajectory),
      snapshotting driver-local state at every depth and logging per-turn (anchor_obs, reward);
  (A') compute A_t = method's assigned per-turn advantage on the grouped batch (method_advantage,
       which calls the exact training estimator);
  (B) for every prefix s_t, K outcome-only continuations FROM THE SAME CHECKPOINT -> V-hat(s_t).
Dumps method_values.json ({traj: {turn: A_t}}) + prefix_values.json (base/vhat/states), consumed
by method_align.

  python -m credit_assignment.diag_runner_methods --method gigpo \
      --model <merged_hf_dir> --data data/searchR1_processed_direct/val_2048.parquet \
      --out outputs/diag_methods/<label> --n-prompts 64 --group-n 5 --k 8 \
      --search-url http://127.0.0.1:8000/retrieve
Protocol constants default to the LOCKED 4turn_think2k values (MUST match the checkpoint).
"""
import argparse
import json
import os

import numpy as np

from credit_assignment.diag_plan import aggregate_vhat, plan_continuation_jobs
from credit_assignment.diag_runner import Generator, build_env_manager
from credit_assignment.method_advantage import gigpo_per_turn_advantage

METHODS = {"gigpo": gigpo_per_turn_advantage}


def base_rollout(mgr, gen, obs, size, max_iters, tids, snap_sink, row_sink):
    """Drive `size` grouped env slots to termination. Per active slot at each depth: snapshot
    (snap_sink: tid, depth, snap, obs_text) pre-action, then log the turn (row_sink: tid, depth,
    anchor_obs, reward) post-action. tids[i] = traj_uid of slot i. Returns terminal rewards."""
    is_done = np.zeros(size, dtype=bool)
    terminal = np.zeros(size, dtype=np.float32)
    turns_done = np.zeros(size, dtype=np.int64)
    for it in range(max_iters):
        active = ~is_done
        act_idx = [i for i in range(size) if active[i]]
        snaps = mgr.snapshot(act_idx)
        pre_text = {i: obs["text"][i] for i in act_idx}
        for i, sn in zip(act_idx, snaps):
            snap_sink(tids[i], it, sn, obs["text"][i])
        texts = gen.generate([obs["text"][i] for i in range(size)])
        obs, rewards, dones, _ = mgr.step(texts)
        rewards = np.asarray(rewards, dtype=np.float32)
        dones = np.asarray(dones, dtype=bool).reshape(-1)
        for i in act_idx:
            row_sink(tids[i], it, pre_text[i], float(rewards[i]))
            turns_done[i] += 1
            if dones[i]:
                terminal[i] = rewards[i]
        is_done |= dones
        if is_done.all():
            break
    return terminal, turns_done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=list(METHODS))
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-prompts", type=int, default=64)
    ap.add_argument("--group-n", type=int, default=5)   # G, must match training GROUP_SIZE
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--k-check", type=int, default=16)
    ap.add_argument("--k-check-frac", type=float, default=0.125)
    ap.add_argument("--pool", type=int, default=320, help="env worker slots")
    ap.add_argument("--seed", type=int, default=0)
    # LOCKED protocol defaults (configs/protocol_4turn_think2k.sh) — must match ckpt
    ap.add_argument("--max-turns", type=int, default=4)
    ap.add_argument("--history-length", type=int, default=4)
    ap.add_argument("--max-prompt-length", type=int, default=4096)
    ap.add_argument("--max-response-length", type=int, default=2048)
    ap.add_argument("--truncation", default="left")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--enable-thinking", action="store_true", default=True)
    ap.add_argument("--no-thinking", dest="enable_thinking", action="store_false")
    ap.add_argument("--search-url", default="http://127.0.0.1:8000/retrieve")
    ap.add_argument("--topk", type=int, default=3)
    # GiGPO config (configs/cond_gigpo.sh)
    ap.add_argument("--step-advantage-w", type=float, default=1.0)
    ap.add_argument("--gigpo-mode", default="mean_std_norm")
    ap.add_argument("--gigpo-similarity-thresh", type=float, default=0.9)
    ap.add_argument("--gigpo-enable-similarity", action="store_true", default=True)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = ap.parse_args()

    import pandas as pd
    os.makedirs(args.out, exist_ok=True)
    df = pd.read_parquet(args.data)
    rng = np.random.RandomState(args.seed)
    sel = rng.choice(len(df), size=min(args.n_prompts, len(df)), replace=False)
    # grouped: each prompt replicated G times -> uid = prompt, traj_uid = f"p{p}_{g}"
    kwargs_all, tid_all, uid_all = [], [], []
    for p in sel:
        row = dict(df.iloc[p]["env_kwargs"])
        for g in range(args.group_n):
            kwargs_all.append(row)
            tid_all.append(f"p{p}_{g}")
            uid_all.append(f"p{p}")
    uid_by_tid = dict(zip(tid_all, uid_all))

    mgr, _ = build_env_manager(args, pool=args.pool)
    gen = Generator(args)

    # ---------------- phase A: grouped base trajectories + per-turn rows -------------
    snapshots, states, base = {}, {}, {}
    rows = []   # {uid, traj_uid, turn_index, anchor_obs, reward}
    for start in range(0, len(kwargs_all), args.pool):
        chunk = kwargs_all[start:start + args.pool]
        tids = tid_all[start:start + args.pool]
        size = len(chunk)
        obs, _ = mgr.reset(kwargs=chunk)

        def snap_sink(tid, depth, snap, obs_text):
            snapshots.setdefault(tid, {})[depth] = snap
            states.setdefault(tid, {})[depth] = obs_text

        def row_sink(tid, depth, anchor_obs, reward):
            rows.append({"uid": uid_by_tid[tid], "traj_uid": tid,
                         "turn_index": depth, "anchor_obs": anchor_obs, "reward": reward})

        terminal, turns_done = base_rollout(mgr, gen, obs, size, args.max_turns,
                                            tids, snap_sink, row_sink)
        for i in range(size):
            base[tids[i]] = {"question": chunk[i].get("question", ""),
                             "data_source": str(chunk[i].get("data_source", "unknown")),
                             "final_reward": float(terminal[i]), "turns": int(turns_done[i])}
        print(f"[diagM] base {start + size}/{len(kwargs_all)} "
              f"mean_EM={np.mean([b['final_reward'] for b in base.values()]):.3f}", flush=True)

    # ---------------- phase A': method-assigned per-turn advantage -------------------
    A = METHODS[args.method](
        rows, gamma=args.gamma, step_advantage_w=args.step_advantage_w,
        mode=args.gigpo_mode, enable_similarity=args.gigpo_enable_similarity,
        similarity_thresh=args.gigpo_similarity_thresh)
    method_values = {}
    for (tid, turn), a in A.items():
        method_values.setdefault(tid, {})[str(turn)] = a
    with open(os.path.join(args.out, "method_values.json"), "w") as fh:
        json.dump({"method": args.method, "values": method_values,
                   "config": {"step_advantage_w": args.step_advantage_w, "mode": args.gigpo_mode,
                              "enable_similarity": args.gigpo_enable_similarity,
                              "similarity_thresh": args.gigpo_similarity_thresh}}, fh)
    print(f"[diagM] method={args.method} advantages for {len(A)} turn-rows", flush=True)

    # ---------------- phase B: K continuations per prefix ---------------------------
    traj_depths = {t: sorted(snaps) for t, snaps in snapshots.items()}
    jobs = plan_continuation_jobs(traj_depths, k=args.k, k_check=args.k_check,
                                  k_check_frac=args.k_check_frac, seed=args.seed)
    slots = [(j, rep) for j in jobs for rep in range(j["k"])]
    print(f"[diagM] {len(jobs)} prefix jobs -> {len(slots)} continuation rollouts", flush=True)
    results = {(j["traj"], j["depth"]): [] for j in jobs}
    for start in range(0, len(slots), args.pool):
        wave = slots[start:start + args.pool]
        k = len(wave)
        mgr.memory.reset(batch_size=k)
        mgr.tasks = [""] * k
        snaps = [snapshots[j["traj"]][j["depth"]] for j, _ in wave]
        obs = mgr.restore_batch(snaps, indices=list(range(k)))
        # continuations are group-free single rollouts (no snapshots/rows needed): drive each
        # restored prefix to termination and record the terminal outcome (mirrors run_turns).
        is_done = np.zeros(k, dtype=bool); term = np.zeros(k, dtype=np.float32)
        for it in range(args.max_turns):
            if is_done.all():
                break
            texts = gen.generate([obs["text"][i] for i in range(k)])
            obs, rewards, dones, _ = mgr.step(texts)
            rewards = np.asarray(rewards, dtype=np.float32); dones = np.asarray(dones, dtype=bool).reshape(-1)
            for i in range(k):
                if not is_done[i] and dones[i]:
                    term[i] = rewards[i]
            is_done |= dones
        for (j, _), r in zip(wave, term):
            results[(j["traj"], j["depth"])].append(float(r))
        print(f"[diagM] continuations {min(start + k, len(slots))}/{len(slots)}", flush=True)

    vhat = aggregate_vhat([{"traj": t, "depth": d, "rewards": rs}
                           for (t, d), rs in results.items()])
    payload = {
        "config": {k: v for k, v in vars(args).items()},
        "base": base,
        "vhat": {t: {str(d): v for d, v in per.items()} for t, per in vhat.items()},
        "states": {t: {str(d): s for d, s in per.items()} for t, per in states.items()},
    }
    with open(os.path.join(args.out, "prefix_values.json"), "w") as fh:
        json.dump(payload, fh)
    print(f"[diagM] wrote prefix_values.json + method_values.json "
          f"({len(base)} trajectories, {len(jobs)} prefixes)", flush=True)


if __name__ == "__main__":
    main()
