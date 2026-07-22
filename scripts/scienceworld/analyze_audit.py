#!/usr/bin/env python3
"""Phase-2 analysis of the ScienceWorld structure audit (Wave-0 M5).

Reads gold_traces.jsonl + task_map.csv from audit_structure.py and produces:
  families.csv          task -> family mapping with horizon stats
  subgoal_semantics.md  oracle reward semantics tables (firing density,
                        magnitudes, negative deltas, terminal composition)
  turn_caps.csv         proposed per-family turn caps (~2x max sampled gold len)
  horizon_hist.png      gold-path-length distribution by family
  html/<task>_v<var>.html   annotated trajectory dumps (subgoal firings inline)
Run: python analyze_audit.py --audit <dir> [--dump-tasks short,median,long picks]
"""

import argparse
import html as html_mod
import json
from collections import defaultdict
from pathlib import Path

FAMILIES = {
    "matter-state": ["boil", "freeze", "melt", "change-the-state-of-matter-of"],
    "chemistry-mix": ["chemistry-mix", "chemistry-mix-paint-secondary-color",
                      "chemistry-mix-paint-tertiary-color"],
    "find-thing": ["find-animal", "find-living-thing", "find-non-living-thing", "find-plant"],
    "life-stages": ["identify-life-stages-1", "identify-life-stages-2"],
    "lifespan": ["lifespan-longest-lived", "lifespan-longest-lived-then-shortest-lived",
                 "lifespan-shortest-lived"],
    "plant-growth": ["grow-fruit", "grow-plant"],
    "genetics": ["mendelian-genetics-known-plant", "mendelian-genetics-unknown-plant"],
    "inclined-plane": ["inclined-plane-determine-angle",
                       "inclined-plane-friction-named-surfaces",
                       "inclined-plane-friction-unnamed-surfaces"],
    "electricity": ["power-component", "power-component-renewable-vs-nonrenewable-energy",
                    "test-conductivity", "test-conductivity-of-unknown-substances"],
    "thermal-measure": ["measure-melting-point-known-substance",
                        "measure-melting-point-unknown-substance", "use-thermometer"],
}
TASK2FAM = {t: f for f, ts in FAMILIES.items() for t in ts}


def load_traces(path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return [r for r in rows if "error" not in r]


def html_dump(rec, out_path):
    rows = []
    for s in rec["trace"]:
        cls = "fire" if s["score_delta"] > 0 else ("neg" if s["score_delta"] < 0 else "")
        badge = (f'<span class="badge">+{s["score_delta"]} → {s["score"]}</span>'
                 if s["score_delta"] > 0 else
                 (f'<span class="badge neg">{s["score_delta"]} → {s["score"]}</span>'
                  if s["score_delta"] < 0 else ""))
        rows.append(
            f'<tr class="{cls}"><td>{s["t"]}</td>'
            f'<td><code>{html_mod.escape(s["action"])}</code> {badge}</td>'
            f'<td class="obs">{html_mod.escape(s["obs_head"])}</td></tr>')
    doc = f"""<meta charset="utf-8"><title>{rec['task']} v{rec['variation']}</title>
<style>
 body{{font:14px/1.4 system-ui;margin:2em;max-width:1100px}}
 table{{border-collapse:collapse;width:100%}} td{{border:1px solid #ddd;padding:4px 8px;vertical-align:top}}
 tr.fire{{background:#e6ffe6}} tr.neg{{background:#ffe6e6}}
 .badge{{background:#2a2;color:#fff;border-radius:4px;padding:1px 6px;font-size:12px}}
 .badge.neg{{background:#c33}} .obs{{color:#555;font-size:12px}} code{{white-space:pre-wrap}}
</style>
<h2>{rec['task']} — variation {rec['variation']} (gold replay)</h2>
<p><b>Task:</b> {html_mod.escape(rec['task_description'])}</p>
<p>gold_len={rec['gold_len']} · final_score={rec['final_score']} · done={rec['done']}
 · subgoal firings={rec['n_subgoal_firings']} at steps {rec['firing_steps']}
 (magnitudes {rec['firing_magnitudes']}) · negative deltas: {rec['negative_deltas'] or 'none'}</p>
<table><tr><th>t</th><th>action (green = subgoal fired)</th><th>observation (first 200 chars)</th></tr>
{''.join(rows)}</table>"""
    out_path.write_text(doc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", required=True)
    args = ap.parse_args()
    audit = Path(args.audit)
    recs = load_traces(audit / "gold_traces.jsonl")

    by_task = defaultdict(list)
    for r in recs:
        by_task[r["task"]].append(r)

    # families.csv + turn_caps.csv
    import csv
    fam_rows, cap_rows = [], []
    by_fam = defaultdict(list)
    for task, rs in by_task.items():
        fam = TASK2FAM.get(task, "UNMAPPED")
        lens = [r["gold_len"] for r in rs]
        by_fam[fam].extend(lens)
        fam_rows.append({"task": task, "family": fam, "gold_len_min": min(lens),
                         "gold_len_max": max(lens),
                         "mean_firings": round(sum(r["n_subgoal_firings"] for r in rs) / len(rs), 1),
                         "all_reach_100": all(r["final_score"] == 100 for r in rs)})
    for fam, lens in sorted(by_fam.items()):
        cap_rows.append({"family": fam, "n_sampled": len(lens), "gold_len_min": min(lens),
                         "gold_len_max": max(lens), "proposed_turn_cap": 2 * max(lens)})
    with open(audit / "families.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fam_rows[0].keys())); w.writeheader(); w.writerows(sorted(fam_rows, key=lambda r: r["family"]))
    with open(audit / "turn_caps.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cap_rows[0].keys())); w.writeheader(); w.writerows(cap_rows)

    # subgoal semantics
    all_firings = [m for r in recs for m in r["firing_magnitudes"]]
    negs = [(r["task"], r["variation"], d) for r in recs for d in r["negative_deltas"]]
    fire_density = [r["n_subgoal_firings"] / r["gold_len"] for r in recs if r["gold_len"]]
    with open(audit / "subgoal_semantics.md", "w") as f:
        f.write("# Oracle subgoal-reward semantics (gold replays)\n\n")
        f.write(f"- traces analyzed: {len(recs)} ({len(by_task)} tasks)\n")
        f.write(f"- score scale: cumulative 0-100 per episode; per-step `reward` = score delta\n")
        f.write(f"- all gold replays reach 100: {all(r['final_score'] == 100 for r in recs)}\n")
        f.write(f"- subgoal firing density on gold paths: min {min(fire_density):.2f} / "
                f"median {sorted(fire_density)[len(fire_density)//2]:.2f} / max {max(fire_density):.2f} "
                f"firings per step\n")
        f.write(f"- firing magnitudes: min {min(all_firings)} / max {max(all_firings)} "
                f"(distinct values: {sorted(set(all_firings))})\n")
        f.write(f"- negative deltas observed on gold paths: {len(negs)}"
                + (f" — {negs[:20]}\n" if negs else " (none)\n"))
        f.write("\n| family | tasks | sampled gold-len range | proposed cap (2x max) |\n|---|---|---|---|\n")
        for c in cap_rows:
            f.write(f"| {c['family']} | {len(FAMILIES.get(c['family'], []))} | "
                    f"{c['gold_len_min']}-{c['gold_len_max']} | {c['proposed_turn_cap']} |\n")

    # horizon histogram
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fams = sorted(by_fam)
        fig, ax = plt.subplots(figsize=(9, 4.5))
        data = [by_fam[f] for f in fams]
        ax.boxplot(data, tick_labels=fams, vert=True)
        ax.set_ylabel("gold path length (steps)")
        ax.set_title("ScienceWorld horizon by task family (sampled gold paths)")
        plt.xticks(rotation=40, ha="right"); plt.tight_layout()
        fig.savefig(audit / "horizon_hist.png", dpi=150)
    except Exception as e:  # noqa: BLE001
        print("figure skipped:", e)

    # HTML dumps: shortest / median / longest trace overall + per-family extremes capped at 12 files
    htmldir = audit / "html"; htmldir.mkdir(exist_ok=True)
    ordered = sorted(recs, key=lambda r: r["gold_len"])
    picks = {id(r): r for r in (ordered[0], ordered[len(ordered)//2], ordered[-1])}
    for fam, _ in list(by_fam.items()):
        fam_recs = sorted([r for r in recs if TASK2FAM.get(r["task"]) == fam], key=lambda r: r["gold_len"])
        if fam_recs:
            picks[id(fam_recs[-1])] = fam_recs[-1]
    for r in list(picks.values())[:14]:
        html_dump(r, htmldir / f"{r['task']}_v{r['variation']}.html")
    print(f"wrote families.csv, turn_caps.csv, subgoal_semantics.md, horizon_hist.png, "
          f"{len(picks)} HTML dumps -> {htmldir}")


if __name__ == "__main__":
    main()
