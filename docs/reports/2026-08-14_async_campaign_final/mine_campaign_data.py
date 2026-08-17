"""Mine per-step training series for the async-campaign final report.

Sources are the arms' console logs. Two families:
- engine comparison (async vs old sync engine): token GRPO + turn-PPO on each
- estimator comparison (old engine, fast75): GRPO, turn-PPO, token-PPO, GiGPO, HCAPO

For the three estimator arms that died at steps 42-47 and were resumed from the
step-25 checkpoint (2026-08-14, *_r2.log), the resume log wins for steps >= 26 —
that lineage is the one that reaches step 75. Replayed steps within a single log
(pod-reclaim replays) resolve to the LAST occurrence.

Writes campaign_data.json next to itself. Rerun any time; it picks up whatever
steps the resumed arms have produced so far.
"""
import json
import re
from pathlib import Path

OUT_DIR = Path(__file__).parent
LOGS = Path("/home/tiger/xiaoxuan/Agentic-RL-CA/outputs")

# arm -> ordered list of logs (later logs override overlapping steps)
SOURCES = {
    # engine comparison
    "async_grpo": ["p8b_async_arm_grpo.log"],
    "async_tppo": ["p8b_async_arm_tppo.log"],
    "old_grpo":   ["p8b_arm_grpo.log"],
    "old_tppo":   ["p8b_arm_ppo.log"],
    # estimator sweep extras (old engine); _r2 = resumed-from-25 relaunch
    "token_ppo":  ["p8b_arm_token_ppo.log", "p8b_arm_token_ppo_r2.log"],
    "gigpo":      ["p8b_arm_gigpo.log", "p8b_arm_gigpo_r2.log"],
    "hcapo_ans":  ["p8b_arm_hcapo_ans.log", "p8b_arm_hcapo_ans_r2.log"],
}

PATTERNS = {
    "released": r"partial/released_traj:([\d.]+)",
    "resumed": r"partial/resumed_traj:([\d.]+)",
    "dropped_stale": r"partial/dropped_stale_groups:([\d.]+)",
    "train_sr": r"episode/success_rate:([\d.]+)",
    "val_em": r"val-core/macro_em[\"']?[:=]\s*([\d.]+)",
    "val_turns": r"val-core/avg_turns[\"']?[:=]\s*([\d.]+)",
    "num_actions": r"episode/num_actions/mean:([\d.]+)",
    "step_wall_s": r"timing_s/step:([\d.]+)",
    "gen_s": r"timing_s/gen:([\d.]+)",
    "resp_len_p10": r"lengthdiag/resp_len/p10:([\d.]+)",
    "resp_len_p50": r"lengthdiag/resp_len/p50:([\d.]+)",
    "resp_len_p90": r"lengthdiag/resp_len/p90:([\d.]+)",
    "valid_action": r"episode/valid_action_ratio:([\d.]+)",
}


def mine_log(path, steps, standalone_vals, from_step=0):
    for line in open(path, errors="replace"):
        m = re.search(r"step:(\d+) - ", line)
        if m:
            s = int(m.group(1))
            if s < from_step and s != 0:
                continue
            d = {}
            for k, pat in PATTERNS.items():
                r = re.search(pat, line)
                if r:
                    d[k] = float(r.group(1))
            if d:
                steps[s] = {**steps.get(s, {}), **d}
        elif "val-core/macro_em" in line:
            v = re.search(PATTERNS["val_em"], line)
            if v:
                standalone_vals.append(float(v.group(1)))


data = {}
for arm, logs in SOURCES.items():
    steps, standalone = {}, []
    for i, name in enumerate(logs):
        p = LOGS / name
        if not p.exists():
            continue
        # resume logs replay from the step-25 ckpt; they own steps >= 26
        mine_log(p, steps, standalone, from_step=26 if i > 0 else 0)
    xs = sorted(steps)
    cum, cum_released = 0.0, {}
    for s in xs:
        cum += steps[s].get("released", 0.0)
        cum_released[s] = cum
    data[arm] = {
        "steps": {str(s): steps[s] for s in xs},
        "cum_released": {str(s): cum_released[s] for s in xs},
        "standalone_vals": standalone,
        "max_step": xs[-1] if xs else None,
        "vals": {str(s): steps[s]["val_em"] for s in xs if "val_em" in steps[s]},
    }
    print(f"{arm:11s} steps={len(xs):3d} max={xs[-1] if xs else '-':>3} "
          f"cum_released={cum:9.0f} vals={data[arm]['vals']} "
          f"standalone={[round(v, 3) for v in standalone]}")

json.dump(data, open(OUT_DIR / "campaign_data.json", "w"))
print("wrote campaign_data.json")
