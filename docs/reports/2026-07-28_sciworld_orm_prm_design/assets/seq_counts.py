import re
from scienceworld import ScienceWorldEnv
e = ScienceWorldEnv("", envStepLimit=10)
for t in sorted(e.get_task_names()):
    e.load(t, 0, "easy"); e.reset()
    gp = e.get_goal_progress()
    section, nseq, nun = None, 0, 0
    for line in gp.splitlines():
        if line.startswith("Sequential Subgoals"): section="s"; continue
        if line.startswith("Unordered and Optional"): section="u"; continue
        if re.match(r"^\d+\t", line):
            if section=="s": nseq+=1
            elif section=="u": nun+=1
    print(f"{t}\tseq={nseq}\tunordered={nun}")
