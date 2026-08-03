#!/usr/bin/env python3
"""Combine per-checkpoint judge summaries into one cross-arm comparison table.

`judge_rollouts.py` writes `<label>.summary.json` per checkpoint. This collates them into
the table the estimator-sweep write-up needs: one row per arm, wiki-answerable and live-web
kept apart, judge / EM / sub-EM side by side.

  python compare_eval_summaries.py outputs/judge/*.summary.json --out outputs/judge/comparison.md

Order the arms explicitly with --order to keep the table stable across reruns:
  --order grpo_s75,turnppo_s75,hcapo_ans_s75,token_ppo_s75,gigpo_s75
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LIVEWEB = ["frames", "GAIA", "xbench-deepsearch"]


def fmt(x) -> str:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return "—"
    return "—" if f != f else f"{f:.3f}"  # NaN-safe


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("summaries", nargs="+")
    ap.add_argument("--out", default="")
    ap.add_argument("--order", default="", help="comma-separated labels, to pin row order")
    args = ap.parse_args()

    loaded = {}
    for p in args.summaries:
        s = json.loads(Path(p).read_text())
        loaded[s.get("label", Path(p).stem)] = s
    if not loaded:
        print("no summaries given")
        return 1

    if args.order:
        wanted = [x.strip() for x in args.order.split(",") if x.strip()]
        labels = [x for x in wanted if x in loaded] + [x for x in loaded if x not in wanted]
    else:
        labels = sorted(loaded)

    L = ["# ASearcher eval suite — cross-arm comparison", ""]
    L += [
        "Judge = the frozen MBE prompt (decision A3). EM is strict exact match, the metric the",
        "search env actually rewards and the training vals report; sub-EM is the permissive",
        "substring variant. Live-web benchmarks (frames / GAIA / xbench-deepsearch) are kept",
        "separate because our offline wiki-18 retriever cannot serve them — those numbers",
        "measure the corpus gap, not the credit-assignment method.",
        "",
        "## Wiki-answerable (7 benchmarks)",
        "",
        "| arm | n_q | judge | EM | sub-EM |",
        "|---|---|---|---|---|",
    ]
    for lb in labels:
        g = loaded[lb].get("wiki_answerable", {})
        L.append(
            f"| {lb} | {g.get('n_questions','—')} | {fmt(g.get('macro_judge'))} | "
            f"{fmt(g.get('macro_em'))} | {fmt(g.get('macro_subem'))} |"
        )

    L += ["", "## Live-web (corpus gap, not a method comparison)", "",
          "| arm | n_q | judge | EM | sub-EM |", "|---|---|---|---|---|"]
    for lb in labels:
        g = loaded[lb].get("live_web", {})
        L.append(
            f"| {lb} | {g.get('n_questions','—')} | {fmt(g.get('macro_judge'))} | "
            f"{fmt(g.get('macro_em'))} | {fmt(g.get('macro_subem'))} |"
        )

    benches = sorted({b for s in loaded.values() for b in s.get("by_benchmark", {})})
    L += ["", "## Per-benchmark judge score", "",
          "| benchmark | " + " | ".join(labels) + " |",
          "|---" * (len(labels) + 1) + "|"]
    for b in benches:
        tag = " *(live-web)*" if b in LIVEWEB else ""
        row = [fmt(loaded[lb].get("by_benchmark", {}).get(b, {}).get("judge")) for lb in labels]
        L.append(f"| {b}{tag} | " + " | ".join(row) + " |")

    L += ["", "## Scorer agreement and answer coverage", "",
          "| arm | judge-vs-EM | judge✓EM✗ | judge✗EM✓ | EM-vs-env reward | no `<answer>` | judge failures |",
          "|---|---|---|---|---|---|---|"]
    for lb in labels:
        s = loaded[lb]
        t = s.get("judge_vs_em", {})
        e = s.get("em_vs_env_reward", {})
        L.append(
            f"| {lb} | {fmt(t.get('agreement'))} | {t.get('judge1_em0','—')} | "
            f"{t.get('judge0_em1','—')} | {fmt(e.get('agreement'))} | "
            f"{fmt(s.get('no_answer_frac'))} | {s.get('judge_failures','—')} |"
        )
    L.append("")

    md = "\n".join(L)
    if args.out:
        Path(args.out).write_text(md)
        print(f"wrote {args.out}")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
