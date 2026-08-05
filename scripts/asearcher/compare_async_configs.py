"""Combined async-rollout configuration comparison (2026-08-05).

Covers the KV sweep v2 (A@0.5 / A@0.60 / A@0.65 — sticky, 17,408) and the
prefill/routing sweep (B: 32k+sticky, C: 17,408+least_loaded, D: 32k+least_loaded,
all @0.65), 4 steps each. Warm-up rule (uniform): the FIRST events file of each run
(the cold-engine 1,280-fresh burst) is excluded; all later collections are measured.

Engine-truth stats come from [async_engine_stats] lines bucketed per run by the
"RUN <tag>" markers in each worker log. Cumulative counters (preempt/ptok/gtok/
budget/prefix) are diffed last-minus-first within the measured window.
"""
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

BASE = "/home/tiger/xiaoxuan/Agentic-RL-CA/outputs/async_diag"
OUT = "/home/tiger/xiaoxuan/Agentic-RL-CA/outputs/async_diag/combined"
RUNS = [
    ("A@0.50", "kvsweep2_u5", "/home/tiger/xiaoxuan/Agentic-RL-CA/outputs/p8b_kv_sweep2.log", "0.5"),
    ("A@0.60", "kvsweep2_u60", "/home/tiger/xiaoxuan/Agentic-RL-CA/outputs/p8b_kv_sweep2.log", "0.60"),
    ("A@0.65", "kvsweep2_u65", "/home/tiger/xiaoxuan/Agentic-RL-CA/outputs/p8b_kv_sweep2.log", "0.65"),
    ("B 32k", "prefill_B", "/home/tiger/xiaoxuan/Agentic-RL-CA/outputs/p8b_prefill_sweep.log", "B"),
    ("C LL", "prefill_C", "/home/tiger/xiaoxuan/Agentic-RL-CA/outputs/p8b_prefill_sweep.log", "C"),
    ("D 32k+LL", "prefill_D", "/home/tiger/xiaoxuan/Agentic-RL-CA/outputs/p8b_prefill_sweep.log", "D"),
]


def pct(a, q):
    a = np.asarray([x for x in a if np.isfinite(x)], dtype=float)
    return float(np.percentile(a, q)) if len(a) else float("nan")


def load_events(d):
    files = sorted(glob.glob(os.path.join(BASE, d, "events_*.jsonl")), key=os.path.getmtime)
    if not files:
        return [], 0.0
    rows, span = [], 0.0
    for f in files[1:]:  # skip file 1 = warm-up burst
        rr = [json.loads(l) for l in open(f)]
        rr = [r for r in rr if r.get("w_enq") and r.get("t_srv_done") and r.get("t_first_token")]
        if not rr:
            continue
        span += max(r["w_env_done"] for r in rr) - min(r["w_enq"] for r in rr)
        rows += rr
    return rows, span


def parse_engine_stats(log_path, run_key):
    """Bucket [async_engine_stats] lines by run using the RUN markers."""
    marker = re.compile(r"==== (?:KV-SWEEP RUN gpu_memory_utilization=|PREFILL-SWEEP RUN )([\w.]+)")
    es_re = re.compile(
        r"\[async_snap\] wall=([0-9.]+) assigned=(\[[^\]]*\]) in_gen=(\[[^\]]*\]) in_env=(\[[^\]]*\])")
    cur = None
    out = []
    for line in open(log_path, errors="replace"):
        m = marker.search(line)
        if m:
            cur = m.group(1)
            continue
        if cur != run_key:
            continue
        m = es_re.search(line)
        if m:
            out.append({
                "wall": float(m.group(1)),
                "in_gen": json.loads(m.group(3)), "in_env": json.loads(m.group(4)),
            })
    return out


def analyze(label, d, log_path, run_key):
    rows, span = load_events(d)
    if not rows:
        return None
    comp = []
    for r in rows:
        n = max(r["n_tokens"] - 1, 1)
        comp.append(dict(r,
                         ttft=r["t_first_token"] - r["w_enq"],
                         tpot=(r["t_srv_done"] - r["t_first_token"]) / n,
                         e2e=r["w_recv"] - r["w_enq"]))
    by_traj = defaultdict(list)
    for c in comp:
        by_traj[c["traj"]].append(c)
    tot_gen = sum(c["n_tokens"] for c in comp)
    tot_prompt = sum(c["prompt_len"] for c in comp)
    eng_req = defaultdict(int)
    eng_gtok = defaultdict(int)
    for c in comp:
        eng_req[c["engine"]] += 1
        eng_gtok[c["engine"]] += c["n_tokens"]
    req_counts = [eng_req.get(e, 0) for e in range(8)]
    cv = float(np.std(req_counts) / np.mean(req_counts)) if np.mean(req_counts) else float("nan")
    mm = max(req_counts) / max(min(req_counts), 1)

    es = parse_engine_stats(log_path, run_key)
    stats = {}
    if es:
        # restrict to the measured window (post first-file span isn't directly known;
        # use the events' measured wall range)
        w0 = min(r["w_enq"] for r in rows)
        w1 = max(r["w_env_done"] for r in rows)
        win = [e for e in es if w0 <= e["wall"] <= w1] or es
        gv = [v for e in win for v in e["in_gen"]]
        ev = [v for e in win for v in e["in_env"]]
        stats = {
            "infl_mean": float(np.mean(gv)), "infl_p50": pct(gv, 50), "infl_p90": pct(gv, 90),
            "inenv_mean": float(np.mean(ev)),
        }
    gpu_file = os.path.join(BASE, d, "gpu_samples.txt")
    peak_mem, powers, utils = 0, [], []
    try:
        for line in open(gpu_file):
            parts = line.strip().split(" ", 1)
            if len(parts) != 2:
                continue
            for g in parts[1].split(";"):
                g = g.strip()
                if not g:
                    continue
                try:
                    u, mem, pw = (g.split(",") + [None, None, None])[:3]
                    utils.append(int(u))
                    peak_mem = max(peak_mem, int(mem))
                    if pw is not None:
                        powers.append(float(pw))
                except (ValueError, TypeError):
                    pass
    except FileNotFoundError:
        pass
    return {
        "label": label, "n_req": len(comp), "n_traj": len(by_traj), "span": span,
        "gen_tok_s": tot_gen / span if span else float("nan"),
        "prompt_tok_s": tot_prompt / span if span else float("nan"),
        "traj_min": len(by_traj) / span * 60 if span else float("nan"),
        "req_s": len(comp) / span if span else float("nan"),
        "tot_gen": tot_gen, "tot_prompt": tot_prompt,
        "ttft": [c["ttft"] for c in comp], "tpot": [c["tpot"] for c in comp],
        "e2e": [c["e2e"] for c in comp],
        "tlat": [max(x["w_env_done"] for x in v) - min(x["w_enq"] for x in v)
                 for v in by_traj.values()],
        "cv": cv, "maxmin": mm, "req_counts": req_counts,
        "peak_mem_gb": peak_mem / 1024,
        "power_mean": float(np.mean(powers)) if powers else float("nan"),
        "util_mean": float(np.mean(utils)) if utils else float("nan"),
        "util_p50": pct(utils, 50), "util_p90": pct(utils, 90),
        **stats,
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    S = [r for r in (analyze(*run) for run in RUNS) if r]
    L = ["# Combined async-rollout configuration comparison",
         "", "Warm-up = first collection file per run excluded; 4-step runs.", ""]
    L.append("| metric | " + " | ".join(s["label"] for s in S) + " |")
    L.append("|---" * (len(S) + 1) + "|")

    def row(name, fn, fmt="{:.2f}"):
        vals = []
        for s in S:
            try:
                vals.append(fmt.format(fn(s)))
            except (KeyError, TypeError):
                vals.append("—")
        L.append(f"| {name} | " + " | ".join(vals) + " |")

    row("measured span (s)", lambda s: s["span"], "{:.0f}")
    row("requests / trajectories", lambda s: s["n_req"], "{:.0f}")
    row("gen tok/s (agg)", lambda s: s["gen_tok_s"], "{:.0f}")
    row("prompt tok/s (agg)", lambda s: s["prompt_tok_s"], "{:.0f}")
    row("trajectories/min", lambda s: s["traj_min"], "{:.1f}")
    row("requests/s", lambda s: s["req_s"], "{:.1f}")
    row("TTFT p50 (s)", lambda s: pct(s["ttft"], 50))
    row("TTFT p90 (s)", lambda s: pct(s["ttft"], 90))
    row("TTFT p99 (s)", lambda s: pct(s["ttft"], 99))
    row("TPOT p50 (ms)", lambda s: pct(s["tpot"], 50) * 1000, "{:.1f}")
    row("TPOT p90 (ms)", lambda s: pct(s["tpot"], 90) * 1000, "{:.1f}")
    row("req e2e p50 (s)", lambda s: pct(s["e2e"], 50))
    row("req e2e p90 (s)", lambda s: pct(s["e2e"], 90))
    row("traj latency p50 (s)", lambda s: pct(s["tlat"], 50))
    row("traj latency p90 (s)", lambda s: pct(s["tlat"], 90))
    row("in-flight/engine mean (driver)", lambda s: s["infl_mean"], "{:.1f}")
    row("in-flight/engine p50", lambda s: s["infl_p50"], "{:.0f}")
    row("in-flight/engine p90", lambda s: s["infl_p90"], "{:.0f}")
    row("in-env/engine mean", lambda s: s["inenv_mean"], "{:.2f}")
    row("engine req CoV", lambda s: s["cv"], "{:.3f}")
    row("engine req max/min", lambda s: s["maxmin"], "{:.2f}")
    row("GPU util mean %", lambda s: s["util_mean"], "{:.0f}")
    row("GPU util p90 %", lambda s: s["util_p90"], "{:.0f}")
    row("power mean (W)", lambda s: s["power_mean"], "{:.0f}")
    row("peak GPU mem (GB)", lambda s: s["peak_mem_gb"], "{:.1f}")

    # plots
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = [s["label"] for s in S]

    def bar_plot(vals, ylabel, fname):
        fig, ax = plt.subplots(figsize=(7.5, 4))
        ax.bar(labels, vals, color="#4477AA", edgecolor="black")
        ax.set_ylabel(ylabel)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.0f}" if v > 10 else f"{v:.2f}", ha="center", va="bottom")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, fname), dpi=140)

    bar_plot([s["span"] for s in S], "measured collection span (s)", "collection_time.png")
    bar_plot([s["gen_tok_s"] for s in S], "generated tokens/s", "gen_tok_s.png")
    bar_plot([pct(s["tpot"], 50) * 1000 for s in S], "TPOT p50 (ms)", "tpot.png")
    bar_plot([pct(s["ttft"], 90) for s in S], "TTFT p90 (s)", "ttft.png")
    bar_plot([s["cv"] for s in S], "engine request-count CoV", "engine_skew.png")

    txt = "\n".join(L)
    open(os.path.join(OUT, "combined_comparison.md"), "w").write(txt + "\n")
    print(txt)
    print(f"\nwrote {OUT}/combined_comparison.md + 5 plots")


if __name__ == "__main__":
    main()
