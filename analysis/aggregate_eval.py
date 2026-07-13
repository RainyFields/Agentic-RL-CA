#!/usr/bin/env python3
"""Phase 3.2 — aggregate a full-set eval JSONL dump into the paper-table CSV/JSON.

Input: val_trajectories_step*.jsonl written by the patched _validate
       (one record per trajectory: traj_uid, data_source, em, turns, terminated).
Output: <out_prefix>.csv (per-dataset row + macro/micro) and <out_prefix>.json,
        with turns min/median/max per dataset (Table-11 trajectory-info commitment).
"""
import argparse
import glob
import json

import numpy as np

# Search-R1 paper table order (single-hop, then multi-hop)
DS_ORDER = ["nq", "triviaqa", "popqa", "hotpotqa", "2wikimultihopqa", "musique", "bamboogle"]
MULTI_HOP = {"hotpotqa", "2wikimultihopqa", "musique", "bamboogle"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="file or glob")
    ap.add_argument("--out_prefix", required=True)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    files = sorted(glob.glob(args.jsonl))
    assert files, f"no files match {args.jsonl}"
    recs = []
    for f in files:
        with open(f) as fh:
            recs += [json.loads(l) for l in fh if l.strip()]

    by_ds = {}
    for r in recs:
        by_ds.setdefault(r["data_source"], []).append(r)

    rows = []
    for ds in DS_ORDER + sorted(set(by_ds) - set(DS_ORDER)):
        if ds not in by_ds:
            continue
        rs = by_ds[ds]
        turns = np.array([r["turns"] for r in rs])
        rows.append(
            {
                "dataset": ds,
                "subgroup": "multi-hop" if ds in MULTI_HOP else "single-hop",
                "n": len(rs),
                "em": float(np.mean([r["em"] for r in rs])),
                "turns_min": int(turns.min()),
                "turns_median": float(np.median(turns)),
                "turns_max": int(turns.max()),
                "avg_turns": float(turns.mean()),
                "unterminated_frac": float(np.mean([not r["terminated"] for r in rs])),
            }
        )

    ems = [row["em"] for row in rows]
    summary = {
        "label": args.label,
        "n_total": len(recs),
        "macro_em": float(np.mean(ems)),
        "micro_em": float(np.mean([r["em"] for r in recs])),
        "macro_em_single_hop": float(
            np.mean([row["em"] for row in rows if row["subgroup"] == "single-hop"])
        ),
        "macro_em_multi_hop": float(
            np.mean([row["em"] for row in rows if row["subgroup"] == "multi-hop"])
        ),
        "source_files": files,
    }

    with open(args.out_prefix + ".json", "w") as f:
        json.dump({"summary": summary, "per_dataset": rows}, f, indent=2)
    cols = list(rows[0].keys())
    with open(args.out_prefix + ".csv", "w") as f:
        f.write(",".join(cols) + "\n")
        for row in rows:
            f.write(",".join(str(row[c]) for c in cols) + "\n")
        f.write(f"macro,,{summary['n_total']},{summary['macro_em']},,,,,\n")
        f.write(f"micro,,{summary['n_total']},{summary['micro_em']},,,,,\n")

    print(json.dumps(summary, indent=2))
    for row in rows:
        print(f"  {row['dataset']:>18s}  n={row['n']:>6d}  EM={row['em']:.4f}  "
              f"turns med={row['turns_median']:.0f} [{row['turns_min']},{row['turns_max']}]")


if __name__ == "__main__":
    main()
