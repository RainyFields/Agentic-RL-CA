"""KV-sweep comparison: 0.5 / 0.60 / 0.65 instrumented runs (2026-08-05 spec).

Reads outputs/async_diag/kvsweep_u{5,60,65}/events_*.jsonl (+ gpu_samples.txt) and
the sweep worker log (resolved config dumps + KV capacity lines). Warm-up: first
30s of each run's FIRST collection excluded (same rule every run).
"""
import glob
import json
import os
import re
import sys

import numpy as np

BASE = "/home/tiger/xiaoxuan/Agentic-RL-CA/outputs/async_diag"
LOG = "/home/tiger/xiaoxuan/Agentic-RL-CA/outputs/p8b_kv_sweep.log"
RUNS = [("0.5", "kvsweep_u5"), ("0.60", "kvsweep_u60"), ("0.65", "kvsweep_u65")]
WARMUP_S = 30.0


def pct(a, q):
    return float(np.percentile(np.asarray(a, dtype=float), q)) if len(a) else float("nan")


def load_run(d):
    files = sorted(glob.glob(os.path.join(BASE, d, "events_*.jsonl")),
                   key=os.path.getmtime)
    rows = []
    span = 0.0
    for i, f in enumerate(files):
        rr = [json.loads(l) for l in open(f)]
        rr = [r for r in rr if r.get("w_enq") and r.get("t_srv_done")]
        if not rr:
            continue
        t0 = min(r["w_enq"] for r in rr)
        t1 = max(r["w_env_done"] for r in rr)
        span += t1 - t0
        cut = t0 + (WARMUP_S if i == 0 else 0.0)
        rows += [r for r in rr if r["w_enq"] >= cut]
    return rows, span


def stats(name, rows, span, gpu_file):
    from collections import defaultdict
    C = []
    for r in rows:
        n = max(r["n_tokens"] - 1, 1)
        ft = r.get("t_first_token")
        C.append(dict(r,
                      ttft=(ft - r["w_enq"]) if ft else float("nan"),
                      decode=(r["t_srv_done"] - ft) if ft else float("nan"),
                      tpot=((r["t_srv_done"] - ft) / n) if ft else float("nan"),
                      e2e=r["w_recv"] - r["w_enq"]))
    by_traj = defaultdict(list)
    for c in C:
        by_traj[c["traj"]].append(c)
    tlat = [max(x["w_env_done"] for x in v) - min(x["w_enq"] for x in v)
            for v in by_traj.values()]
    toks = [sum(x["n_tokens"] for x in v) for v in by_traj.values()]
    turns = [len(v) for v in by_traj.values()]
    tot_tok = sum(c["n_tokens"] for c in C)
    tpots = [c["tpot"] for c in C if np.isfinite(c["tpot"])]
    peak_mem = 0
    try:
        for line in open(gpu_file):
            parts = line.strip().split(" ", 1)
            if len(parts) == 2:
                for g in parts[1].split(";"):
                    g = g.strip()
                    if g:
                        try:
                            peak_mem = max(peak_mem, int(g.split(",")[1]))
                        except (IndexError, ValueError):
                            pass
    except FileNotFoundError:
        pass
    tok_s = tot_tok / span if span else float("nan")
    return {
        "run": name, "n_req": len(C), "n_traj": len(by_traj), "span_s": span,
        "tok_s": tok_s, "traj_min": len(by_traj) / span * 60 if span else float("nan"),
        "eff_conc": tok_s * float(np.mean(tpots)) / 8 if tpots else float("nan"),
        "tlat": tlat,
        "ttft": [c["ttft"] for c in C], "decode": [c["decode"] for c in C],
        "tpot": tpots, "e2e": [c["e2e"] for c in C],
        "tok_per_traj": float(np.mean(toks)), "turns_per_traj": float(np.mean(turns)),
        "peak_mem_gb": peak_mem / 1024,
    }


def main():
    log = open(LOG, errors="replace").read()
    caps = re.findall(r"gpu_memory_utilization=([0-9.]+) .*?|GPU KV cache size: ([0-9,]+) tokens", log)
    kv_caps = sorted(set(c[1] for c in caps if c[1]))
    resolved = re.findall(r"actor_rollout_ref\.rollout\.gpu_memory_utilization=([0-9.]+)", log)

    out = []
    S = []
    for name, d in RUNS:
        rows, span = load_run(d)
        if not rows:
            print(f"run {name}: NO DATA — skipped")
            continue
        S.append(stats(name, rows, span, os.path.join(BASE, d, "gpu_samples.txt")))

    out.append("# KV-cache sweep — matched instrumented workload (token_grpo, 1 step)")
    out.append("")
    out.append(f"resolved CLI overrides seen in log: gpu_memory_utilization = {sorted(set(resolved))}")
    out.append(f"runtime engine KV capacities: {kv_caps} tokens")
    out.append("")
    hdr = ("| metric | " + " | ".join(s["run"] for s in S) + " |")
    out.append(hdr)
    out.append("|---" * (len(S) + 1) + "|")

    def row(label, fn, fmt="{:.2f}"):
        out.append(f"| {label} | " + " | ".join(fmt.format(fn(s)) for s in S) + " |")

    row("requests (post-warm-up)", lambda s: s["n_req"], "{:.0f}")
    row("completed trajectories", lambda s: s["n_traj"], "{:.0f}")
    row("collection span (s)", lambda s: s["span_s"], "{:.0f}")
    row("agg gen tok/s", lambda s: s["tok_s"], "{:.0f}")
    row("trajectories/min", lambda s: s["traj_min"], "{:.1f}")
    row("eff. decode concurrency /engine", lambda s: s["eff_conc"], "{:.1f}")
    row("gen tokens/traj", lambda s: s["tok_per_traj"], "{:.0f}")
    row("turns/traj", lambda s: s["turns_per_traj"], "{:.2f}")
    for key, label in (("tlat", "traj latency"), ("ttft", "TTFT"),
                       ("decode", "decode"), ("e2e", "req e2e")):
        for q, qn in ((50, "p50"), (90, "p90"), (99, "p99")):
            row(f"{label} {qn} (s)", lambda s, k=key, q=q: pct(s[k], q))
        row(f"{label} mean (s)", lambda s, k=key: float(np.nanmean(s[k])))
    row("TPOT p50 (ms)", lambda s: pct(s["tpot"], 50) * 1000, "{:.1f}")
    row("TPOT p90 (ms)", lambda s: pct(s["tpot"], 90) * 1000, "{:.1f}")
    row("peak GPU mem (GB)", lambda s: s["peak_mem_gb"], "{:.1f}")

    if len(S) > 1:
        base = S[0]
        out.append("")
        out.append("## Speedups vs 0.5")
        out.append("| ratio | " + " | ".join(s["run"] for s in S[1:]) + " |")
        out.append("|---" * len(S) + "|")
        out.append("| agg tok/s | " + " | ".join(f"{s['tok_s'] / base['tok_s']:.3f}x" for s in S[1:]) + " |")
        out.append("| traj/min | " + " | ".join(f"{s['traj_min'] / base['traj_min']:.3f}x" for s in S[1:]) + " |")
        out.append("| traj-latency p50 (lower=better) | " + " | ".join(
            f"{pct(s['tlat'], 50) / pct(base['tlat'], 50):.3f}x" for s in S[1:]) + " |")
        out.append("| traj-latency p90 | " + " | ".join(
            f"{pct(s['tlat'], 90) / pct(base['tlat'], 90):.3f}x" for s in S[1:]) + " |")
        out.append("| eff. decode concurrency | " + " | ".join(
            f"{s['eff_conc'] / base['eff_conc']:.3f}x" for s in S[1:]) + " |")

    txt = "\n".join(out)
    print(txt)
    open(os.path.join(BASE, "kv_sweep_comparison.md"), "w").write(txt + "\n")
    print(f"\nwrote {BASE}/kv_sweep_comparison.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
