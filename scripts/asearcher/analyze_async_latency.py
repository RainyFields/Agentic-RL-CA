"""Latency-distribution analysis for the async rollout diag run (2026-08-05 spec).

Inputs:
  --events   events_<runid>.jsonl from the async collector (one row per LLM request)
  --log      the diag worker console log (per-engine vLLM stat lines, KV capacity)
  --out      output dir (CSV, percentile tables, PNG histograms/CDFs, diagnosis)

Components (wall-clock, all processes on one node):
  transport      = t_rpc_recv    - w_enq          (driver->Ray-actor delivery)
  ttft_total     = t_first_token - w_enq          (user-visible TTFT)
  ttft_service   = t_first_token - t_rpc_recv     (in-engine queue + prefill; the
                                                   engine scheduler does not expose
                                                   the queue/prefill split)
  decode         = t_srv_done    - t_first_token
  tpot           = decode / max(n_tokens - 1, 1)
  e2e            = w_recv        - w_enq
  env_wait       = w_env_done    - w_recv
  turn           = w_env_done    - w_enq
Trajectory latency = last w_env_done - first w_enq per traj (completed turns only).
"""
import argparse
import json
import os
import re
from collections import defaultdict

import numpy as np


def pctl_row(name, arr):
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return f"| {name} | 0 | - | - | - | - | - | - |"
    q = np.percentile(a, [50, 90, 95, 99])
    return (f"| {name} | {len(a)} | {a.mean():.3f} | {q[0]:.3f} | {q[1]:.3f} "
            f"| {q[2]:.3f} | {q[3]:.3f} | {a.max():.3f} |")


HDR = ("| component | n | mean | p50 | p90 | p95 | p99 | max |\n"
       "|---|---|---|---|---|---|---|---|")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--warmup_s", type=float, default=30.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = [json.loads(l) for l in open(args.events)]
    rows = [r for r in rows if r.get("w_enq") and r.get("t_srv_done")]
    t_start = min(r["w_enq"] for r in rows)
    t_end = max(r["w_env_done"] for r in rows)
    warm = [r for r in rows if r["w_enq"] - t_start < args.warmup_s]
    body = [r for r in rows if r["w_enq"] - t_start >= args.warmup_s]

    def comp(r):
        n = max(r["n_tokens"] - 1, 1)
        return dict(
            transport=r["t_rpc_recv"] - r["w_enq"],
            ttft_total=(r["t_first_token"] - r["w_enq"]) if r.get("t_first_token") else float("nan"),
            ttft_service=(r["t_first_token"] - r["t_rpc_recv"]) if r.get("t_first_token") else float("nan"),
            decode=(r["t_srv_done"] - r["t_first_token"]) if r.get("t_first_token") else float("nan"),
            tpot=((r["t_srv_done"] - r["t_first_token"]) / n) if r.get("t_first_token") else float("nan"),
            e2e=r["w_recv"] - r["w_enq"],
            env_wait=r["w_env_done"] - r["w_recv"],
            turn=r["w_env_done"] - r["w_enq"],
        )

    C = [dict(r, **comp(r)) for r in body]
    L = ["# Async rollout latency analysis", "",
         f"events: {len(rows)} requests total; {len(warm)} excluded as warm-up "
         f"(first {args.warmup_s:.0f}s); collection span {t_end - t_start:.1f}s", ""]

    # ---- headline percentile table ----
    L += ["## Latency components (s)", "", HDR]
    for k in ("transport", "ttft_total", "ttft_service", "decode", "tpot", "e2e",
              "env_wait", "turn"):
        L.append(pctl_row(k, [c[k] for c in C]))
    tok = sum(c["n_tokens"] for c in C)
    span = max(c["w_recv"] for c in C) - min(c["w_enq"] for c in C)
    L += ["", f"generated tokens: {tok} over {span:.1f}s -> "
              f"**{tok / span:.0f} tok/s aggregate** ({tok / span / 8:.0f} tok/s/engine)"]

    # ---- trajectory latency + phase fractions ----
    by_traj = defaultdict(list)
    for c in C:
        by_traj[c["traj"]].append(c)
    tlat = [max(x["w_env_done"] for x in v) - min(x["w_enq"] for x in v)
            for v in by_traj.values()]
    L += ["", "## Trajectory completion latency (s)", "", HDR, pctl_row("trajectory", tlat)]
    tot_turn = sum(c["turn"] for c in C)
    fr = {
        "transport+queue+prefill (ttft_total)": sum(c["ttft_total"] for c in C if np.isfinite(c["ttft_total"])),
        "decode": sum(c["decode"] for c in C if np.isfinite(c["decode"])),
        "tool/env wait": sum(c["env_wait"] for c in C),
    }
    L += ["", "## Where trajectory time goes (sum over all turns)", ""]
    for k, v in fr.items():
        L.append(f"- {k}: {v:.0f}s ({100 * v / tot_turn:.1f}%)")

    # ---- strata ----
    def strat(title, keyf, bins):
        L.extend(["", f"## {title}", "", HDR])
        for name, lo, hi in bins:
            sel = [c for c in C if lo <= keyf(c) < hi]
            L.append(pctl_row(f"{name} (e2e)", [c["e2e"] for c in sel]))
            L.append(pctl_row(f"{name} (ttft_svc)", [c["ttft_service"] for c in sel]))

    strat("By context length", lambda c: c["prompt_len"],
          [("0-2k", 0, 2048), ("2-4k", 2048, 4096), ("4-8k", 4096, 8192),
           ("8-12k", 8192, 12288), ("12k+", 12288, 1 << 30)])
    strat("By generated length", lambda c: c["n_tokens"],
          [("0-128", 0, 128), ("128-256", 128, 256), ("256-512", 256, 512),
           ("512+", 512, 1 << 30)])
    strat("By in-flight requests on engine at enqueue", lambda c: c.get("inflight_engine_at_enq", -1),
          [("0-40", 0, 40), ("40-80", 40, 80), ("80-120", 80, 120), ("120+", 120, 10000)])
    third = (t_end - t_start - args.warmup_s) / 3
    strat("By collection phase", lambda c: c["w_enq"] - t_start - args.warmup_s,
          [("early", 0, third), ("middle", third, 2 * third), ("late", 2 * third, 1e18)])

    L.extend(["", "## Per-engine (e2e s / tok/s share)", "", HDR])
    for e in range(8):
        sel = [c for c in C if c.get("engine") == e]
        L.append(pctl_row(f"engine {e}", [c["e2e"] for c in sel]))

    # ---- engine stat lines from the log (running/waiting/KV%) ----
    stat_re = re.compile(
        r"\(AsyncvLLMServer pid=(\d+)\).*Running: (\d+) reqs?, Waiting: (\d+) reqs?, "
        r"GPU KV cache usage: ([0-9.]+)%")
    stats = defaultdict(lambda: {"run": [], "wait": [], "kv": []})
    for line in open(args.log, errors="replace"):
        m = stat_re.search(line)
        if m:
            pid = m.group(1)
            stats[pid]["run"].append(int(m.group(2)))
            stats[pid]["wait"].append(int(m.group(3)))
            stats[pid]["kv"].append(float(m.group(4)))
    if stats:
        L += ["", "## vLLM engine stats (per engine over run; 10s cadence)", "",
              "| engine(pid) | run mean/p50/p90 | wait mean/p50/p90 | KV% mean/max |",
              "|---|---|---|---|"]
        for pid, d in sorted(stats.items()):
            r, w, k = map(np.array, (d["run"], d["wait"], d["kv"]))
            L.append(f"| {pid} | {r.mean():.0f}/{np.median(r):.0f}/{np.percentile(r, 90):.0f} "
                     f"| {w.mean():.0f}/{np.median(w):.0f}/{np.percentile(w, 90):.0f} "
                     f"| {k.mean():.0f}/{k.max():.0f} |")
    cap = set(re.findall(r"GPU KV cache size: ([0-9,]+) tokens", open(args.log, errors="replace").read()))
    L.append("")
    L.append(f"runtime KV capacity lines: {sorted(cap) if cap else 'NOT FOUND (analytical ~160k/engine)'}")

    # ---- plots ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, key in zip(axes.flat, ("ttft_total", "ttft_service", "tpot", "e2e")):
        v = np.array([c[key] for c in C])
        v = v[np.isfinite(v)]
        ax.hist(v, bins=60)
        ax2 = ax.twinx()
        ax2.plot(np.sort(v), np.linspace(0, 1, len(v)), color="crimson")
        ax.set_title(key)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "latency_hists.png"), dpi=140)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].scatter([c["prompt_len"] for c in C], [c["e2e"] for c in C], s=4, alpha=0.3)
    axes[0].set_xlabel("context length (tokens)"); axes[0].set_ylabel("e2e latency (s)")
    axes[1].scatter([c.get("inflight_engine_at_enq", -1) for c in C],
                    [c["e2e"] for c in C], s=4, alpha=0.3)
    axes[1].set_xlabel("in-flight on engine at enqueue"); axes[1].set_ylabel("e2e latency (s)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "latency_scatter.png"), dpi=140)

    # ---- CSV ----
    import csv
    keys = sorted(C[0].keys())
    with open(os.path.join(args.out, "requests.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(C)

    open(os.path.join(args.out, "latency_report.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nwrote {args.out}/latency_report.md, requests.csv, *.png")


if __name__ == "__main__":
    main()
