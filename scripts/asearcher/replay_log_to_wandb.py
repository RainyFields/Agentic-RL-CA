#!/usr/bin/env python3
"""Recover killed runs' metrics into online W&B by replaying the console log.

The .wandb transaction logs of killed workers are truncated (unexpected EOF) and
`wandb sync` refuses them; the trainer's per-step metric lines in the launch log are
complete, so re-log them. Ranger format: "step:N - key:val - key:val - ...".

Run: python replay_log_to_wandb.py --log <launch.log> --name <run_name> [--project ...]
"""
import argparse, re

STEP_RE = re.compile(r"step:(\d+) - (.+)$")
KV_RE = re.compile(r"([A-Za-z0-9_\-./@]+):(-?[0-9]+\.?[0-9]*(?:[eE][+-]?\d+)?)(?: - |$)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--project", default="ca-rung3-4b-feasibility")
    ap.add_argument("--entity", default="rainyfields")
    args = ap.parse_args()

    steps = {}
    with open(args.log, errors="replace") as f:
        for line in f:
            m = STEP_RE.search(line)
            if not m:
                continue
            step = int(m.group(1))
            kv = {k: float(v) for k, v in KV_RE.findall(m.group(2))}
            if kv:
                steps.setdefault(step, {}).update(kv)
    if not steps:
        raise SystemExit(f"no step lines found in {args.log}")

    import wandb
    run = wandb.init(project=args.project, entity=args.entity, name=args.name,
                     mode="online", notes=f"replayed from {args.log} (original .wandb truncated by worker kill)")
    for step in sorted(steps):
        run.log(steps[step], step=step)
    run.finish()
    print(f"replayed {len(steps)} steps -> {args.entity}/{args.project}/{args.name}")


if __name__ == "__main__":
    main()
