"""Mine the matched sync-vs-async profile from a p8b_async_profile_worker.sh log.

Usage: python mine_async_profile.py <worker_log> [--out table.md]

Pulls, per phase (B_sync / C_async / D_tppo):
  - per-step: timing_s/gen, timing_s/step, perf/total_num_tokens, response tokens/s,
    partial/released_traj + pending
  - sync only:  [sync_rollout_profile] zombie token lines
  - async only: [async_rollout] cycle lines (wall, max_concurrent_gen, overlaps)
"""
import argparse
import json
import re
import sys
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    phase = None
    data = defaultdict(lambda: {"steps": [], "zombie": [], "async_cycles": []})
    step_re = re.compile(r"step:(\d+) .*")

    def grab(line, key):
        m = re.search(re.escape(key) + r":([-\d.]+)", line)
        return float(m.group(1)) if m else None

    for line in open(args.log, errors="replace"):
        m = re.search(r"==== PHASE (\w+): ", line)
        if m:
            phase = m.group(1)
            continue
        if phase is None:
            continue
        if "[sync_rollout_profile]" in line:
            m = re.search(r"active=(\d+) zombie=(\d+) zombie_ratio=([\d.]+)", line)
            if m:
                data[phase]["zombie"].append(
                    (int(m.group(1)), int(m.group(2)), float(m.group(3))))
        if "[async_rollout] cycle done" in line:
            m = re.search(r"(\d+) trajectories, (\d+) turns, wall ([\d.]+)s, "
                          r"max_concurrent_gen (\d+), env-inside-gen overlaps (\d+)", line)
            if m:
                data[phase]["async_cycles"].append({
                    "traj": int(m.group(1)), "turns": int(m.group(2)),
                    "wall_s": float(m.group(3)), "max_conc": int(m.group(4)),
                    "overlaps": int(m.group(5))})
        if re.search(r"step:\d+ - ", line):
            rec = {"step": int(re.search(r"step:(\d+)", line).group(1))}
            for k in ("timing_s/gen", "timing_s/step", "perf/total_num_tokens",
                      "perf/throughput", "partial/released_traj",
                      "partial/pending_traj", "response_length/mean",
                      "episode/reward/mean"):
                v = grab(line, k)
                if v is not None:
                    rec[k] = v
            data[phase]["steps"].append(rec)

    lines = ["# Matched sync vs async collector profile", ""]
    for ph in sorted(data):
        d = data[ph]
        if not d["steps"]:
            continue
        lines.append(f"## Phase {ph}")
        lines.append("")
        lines.append("| step | gen s | step s | tokens | released | pending | reward |")
        lines.append("|---|---|---|---|---|---|---|")
        for r in d["steps"]:
            lines.append(
                f"| {r['step']:.0f} | {r.get('timing_s/gen', 0):.0f} "
                f"| {r.get('timing_s/step', 0):.0f} "
                f"| {r.get('perf/total_num_tokens', 0):.0f} "
                f"| {r.get('partial/released_traj', 0):.0f} "
                f"| {r.get('partial/pending_traj', 0):.0f} "
                f"| {r.get('episode/reward/mean', 0):.3f} |")
        gens = [r.get("timing_s/gen", 0) for r in d["steps"]]
        steps_s = [r.get("timing_s/step", 0) for r in d["steps"]]
        lines.append("")
        lines.append(f"totals: gen {sum(gens):.0f}s, step {sum(steps_s):.0f}s "
                     f"over {len(gens)} steps")
        if d["zombie"]:
            ta = sum(z[0] for z in d["zombie"])
            tz = sum(z[1] for z in d["zombie"])
            lines.append(f"zombie tokens: active={ta} zombie={tz} "
                         f"ratio={tz / (ta + tz):.4f}")
        if d["async_cycles"]:
            mc = max(c["max_conc"] for c in d["async_cycles"])
            ov = sum(c["overlaps"] for c in d["async_cycles"])
            lines.append(f"async: max_concurrent_gen={mc}, "
                         f"env-inside-gen overlaps={ov} (zombie tokens = 0 by construction)")
        lines.append("")

    b = data.get("B_sync", {}).get("steps", [])
    c = data.get("C_async", {}).get("steps", [])
    if b and c:
        n = min(len(b), len(c))
        gb = sum(r.get("timing_s/gen", 0) for r in b[:n])
        gc = sum(r.get("timing_s/gen", 0) for r in c[:n])
        sb = sum(r.get("timing_s/step", 0) for r in b[:n])
        sc = sum(r.get("timing_s/step", 0) for r in c[:n])
        lines.append("## Matched comparison (first %d steps)" % n)
        lines.append("")
        lines.append(f"- rollout gen:  sync {gb:.0f}s vs async {gc:.0f}s -> "
                     f"**{gb / gc:.2f}x**" if gc else "- gen: n/a")
        lines.append(f"- full step:    sync {sb:.0f}s vs async {sc:.0f}s -> "
                     f"**{sb / sc:.2f}x**" if sc else "- step: n/a")
    out = "\n".join(lines)
    if args.out:
        open(args.out, "w").write(out + "\n")
        print(f"wrote {args.out}")
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
