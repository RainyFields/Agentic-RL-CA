import re
from scienceworld import ScienceWorldEnv
e = ScienceWorldEnv("", envStepLimit=10)
for t in ["inclined-plane-determine-angle", "mendelian-genetics-known-plant", "grow-fruit", "boil", "grow-plant"]:
    e.load(t, 0, "easy"); e.reset()
    section = None
    print(f"=== {t}")
    for line in e.get_goal_progress().splitlines():
        if line.startswith("Sequential Subgoals"): section="s"; continue
        if line.startswith("Unordered"): section=None; continue
        if section=="s" and re.match(r"^\d+\t", line):
            print("   REQUIRED:", line.split("\t")[-1].strip())
