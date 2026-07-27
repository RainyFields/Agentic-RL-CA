import re, json
from scienceworld import ScienceWorldEnv

TASKS = ["boil", "chemistry-mix", "grow-plant", "power-component",
         "find-animal", "measure-melting-point-known-substance",
         "mendelian-genetics-known-plant", "identify-life-stages-1"]

def parse_gp(gp):
    seq, unord, section = [], [], None
    for line in gp.splitlines():
        if line.startswith("Sequential Subgoals"): section = "seq"; continue
        if line.startswith("Unordered and Optional"): section = "un"; continue
        m = re.match(r"^(\d+)\t(true|false)\t\s*(\S+)\t(.*)$", line)
        if m and section:
            (seq if section == "seq" else unord).append((m.group(2) == "true", m.group(3), m.group(4)))
    return seq, unord

e = ScienceWorldEnv("", envStepLimit=400)
for t in TASKS:
    try:
        e.load(t, 0, "easy", generateGoldPath=True)
        gold = e.get_gold_action_sequence()
        e.reset()
        score = 0
        for a in gold:
            obs, r, done, info = e.step(a)
            score = info.get("score", 0)
            if done: break
        seq, unord = parse_gp(e.get_goal_progress())
        n_seq = len(seq); n_done = sum(1 for s in seq if s[0])
        un_done = sum(1 for u in unord if u[0])
        print(f"{t}: final_score={score} seq {n_done}/{n_seq} "
              f"unordered {un_done}/{len(unord)} "
              f"incomplete_seq={[ (c,d[:40]) for ok,c,d in seq if not ok ]}")
    except Exception as ex:
        print(f"{t}: ERROR {ex}")
