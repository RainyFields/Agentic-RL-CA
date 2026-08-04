#!/usr/bin/env python3
"""Mine verl per-step phase timings from training logs into one JSON.

Rerunnable data source for the hardware/algorithm profiling report. Each input log is a
console log containing the per-step metrics line ("timing_s/step: ... training/global_step:N").
When a step number appears more than once (crash-resume replays), the LAST occurrence wins —
i.e. the values from the attempt that actually produced the run's checkpoints.

  python mine_timings.py --out assets/timings.json \
      grpo_h100=../../.../outputs/p8b_arm_grpo.log \
      turn_ppo_h100=../../.../outputs/p8b_arm_ppo.log \
      b200=../../.../outputs/p8b_prof_b200.log

The b200 log holds two runs (turn_ppo_b0 then token_grpo); they are split on the
"B200-PROF <cond>: " banner into keys turn_ppo_b200 / grpo_b200.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

PAT_T = re.compile(r"timing_s/([a-z_]+):([0-9.]+)")
PAT_P = re.compile(r"perf/(total_num_tokens|throughput|time_per_step|mfu/actor|mfu/critic):([0-9.]+)")
PAT_TPT = re.compile(r"timing_per_token_ms/(gen|update_actor|update_critic):([0-9.]+)")
PAT_G = re.compile(r"training/global_step:([0-9]+)")
B200_BANNER = re.compile(r"B200-PROF (token_grpo|turn_ppo_b0): ")


def parse_metrics_line(line: str) -> dict | None:
    if "timing_s/step" not in line:
        return None
    g = PAT_G.search(line)
    if not g:
        return None
    d = {k: float(v) for k, v in PAT_T.findall(line)}
    d.update({k.replace("/", "_"): float(v) for k, v in PAT_P.findall(line)})
    d.update({f"tpt_{k}": float(v) for k, v in PAT_TPT.findall(line)})
    d["global_step"] = int(g.group(1))
    return d


def mine_plain(path: str) -> dict[int, dict]:
    steps: dict[int, dict] = {}
    with open(path, errors="replace") as fh:
        for line in fh:
            d = parse_metrics_line(line)
            if d:
                steps[d["global_step"]] = d
    return steps


def mine_b200(path: str) -> dict[str, dict[int, dict]]:
    cur = None
    out: dict[str, dict[int, dict]] = {"turn_ppo_b200": {}, "grpo_b200": {}}
    key = {"turn_ppo_b0": "turn_ppo_b200", "token_grpo": "grpo_b200"}
    with open(path, errors="replace") as fh:
        for line in fh:
            m = B200_BANNER.search(line)
            if m:
                cur = key[m.group(1)]
                continue
            d = parse_metrics_line(line)
            if d and cur:
                out[cur][d["global_step"]] = d
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="name=path pairs; name 'b200' splits on run banners")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    result: dict[str, dict] = {}
    for spec in args.inputs:
        name, path = spec.split("=", 1)
        if name == "b200":
            for k, v in mine_b200(path).items():
                result[k] = {str(s): d for s, d in sorted(v.items())}
        else:
            result[name] = {str(s): d for s, d in sorted(mine_plain(path).items())}

    for name, steps in result.items():
        ks = sorted(int(s) for s in steps)
        rng = f"{ks[0]}..{ks[-1]}" if ks else "EMPTY"
        print(f"{name}: {len(ks)} steps ({rng})")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
