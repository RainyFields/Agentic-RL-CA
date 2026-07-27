#!/usr/bin/env python3
"""Stage-2 pass-rate filter: per-question in-setup pass rates from a rollout dump.

Input: rollout_log.jsonl written by the val_only diagnostic run (DUMP_TRAIN_SAMPLE=1) —
per-turn records with traj_uid / turn_index / env_won / observation. Trajectories are
grouped by traj_uid; the question is extracted from the turn-0 observation (search
template "Your question: ..."), a trajectory is a pass if any turn has env_won.

Outputs under --out-dir:
  scored.jsonl       per-question: n_traj, pass_count, pass_rate
  stats.json         pass-count histogram, band fractions, match diagnostics
  band_keep.jsonl    ORIGINAL rows (from --keep) with 0 < pass_count < n_traj
                     (the informative-gradient band: not all-fail, not all-pass)
Run: python passrate_from_dump.py --dump <rollout_log.jsonl> --keep <keep_ge2.jsonl> --out-dir <dir>
"""
import argparse, json, re
from collections import defaultdict
from pathlib import Path

Q_RE = re.compile(r"Your question: (.*?)\n\n(?:Prior to this step|Now it's)", re.S)


def norm_ws(s):
    return " ".join(str(s).split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--keep", required=True, help="keep_ge2.jsonl with the original rows")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--expected-rep", type=int, default=8)
    args = ap.parse_args()

    trajs = defaultdict(lambda: {"turn0_obs": None, "min_turn": 10**9, "won": False, "rows": 0})
    n_lines = bad_lines = 0
    with open(args.dump, encoding="utf-8") as f:
        for line in f:
            n_lines += 1
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                bad_lines += 1
                continue
            t = trajs[r["traj_uid"]]
            t["rows"] += 1
            ti = int(r.get("turn_index", 0))
            if ti < t["min_turn"]:
                t["min_turn"] = ti
                t["turn0_obs"] = r.get("observation", "")
            if r.get("env_won"):
                t["won"] = True

    per_q = defaultdict(lambda: [0, 0])  # norm_q -> [n_traj, pass_count]
    q_text = {}
    no_q = 0
    for t in trajs.values():
        m = Q_RE.search(t["turn0_obs"] or "")
        if not m:
            no_q += 1
            continue
        q = m.group(1).strip()
        k = norm_ws(q)
        q_text.setdefault(k, q)
        per_q[k][0] += 1
        per_q[k][1] += 1 if t["won"] else 0

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    hist = defaultdict(int)
    with open(out / "scored.jsonl", "w") as f:
        for k, (n, p) in per_q.items():
            hist[p] += 1
            f.write(json.dumps({"question": q_text[k], "n_traj": n, "pass_count": p,
                                "pass_rate": round(p / n, 4)}, ensure_ascii=False) + "\n")

    keep_rows = [json.loads(l) for l in open(args.keep, encoding="utf-8")]
    # mirror make_diag_parquet's drop (invalid slice never entered the diag parquet)
    keep_rows = [r for r in keep_rows if r.get("source") != "compose_chain_qa_invalid"]
    matched = unmatched = 0
    band = []
    for r in keep_rows:
        k = norm_ws(r["question"])
        if k not in per_q:
            unmatched += 1
            continue
        matched += 1
        n, p = per_q[k]
        if 0 < p < n:
            band.append(r)
    with open(out / "band_keep.jsonl", "w") as f:
        for r in band:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total_q = len(per_q)
    odd_rep = sum(1 for n, _ in per_q.values() if n != args.expected_rep)
    stats = {
        "dump_lines": n_lines, "bad_lines": bad_lines, "n_traj": len(trajs),
        "questions_scored": total_q, "traj_without_question": no_q,
        "questions_not_expected_rep": odd_rep,
        "pass_hist": {str(k): hist[k] for k in sorted(hist)},
        "all_fail_frac": round(hist[0] / total_q, 4) if total_q else None,
        "all_pass_frac": round(
            sum(c for p, c in hist.items() if p >= args.expected_rep) / total_q, 4)
            if total_q else None,
        "keep_rows": len(keep_rows), "matched": matched, "unmatched": unmatched,
        "band_keep_count": len(band),
    }
    json.dump(stats, open(out / "stats.json", "w"), indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
