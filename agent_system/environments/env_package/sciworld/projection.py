# Rung-4 ScienceWorld action projection: parse "Thought: ...\nAction: ..." ReAct replies.
# Returns (actions, valids, parse_statuses) — 3-tuple like alfworld_projection, so
# parse_status feeds the SP3.1 diagnostics. valids here = FORMAT validity only; the
# env-manager refines is_action_valid with the env's own no-match response after step().

import re
from typing import List

_ACTION_RE = re.compile(r"[Aa]ction\s*:\s*(.+?)(?:\n|$)")


def _clean(action: str) -> str:
    action = action.strip().strip("`'\"").rstrip(".!").strip()
    # strip markdown bold/italics the model sometimes emits
    action = re.sub(r"[*_]{1,3}", "", action).strip()
    return action


def sciworld_projection(actions: List[str]):
    n = len(actions)
    out, valids, statuses = [None] * n, [0] * n, ["invalid_format"] * n

    for i, raw in enumerate(actions):
        raw = raw if isinstance(raw, str) else ""
        # CJK output -> invalid (mirror alfworld guard)
        if re.search(r"[一-鿿]", raw):
            out[i] = raw[-40:]
            continue
        matches = _ACTION_RE.findall(raw)
        if not matches:
            out[i] = _clean(raw.splitlines()[-1] if raw.strip() else "")[-60:] or "look around"
            statuses[i] = "no_action_line"
            continue
        cand = _clean(matches[-1])  # last Action: line wins (models sometimes revise)
        if not cand:
            out[i] = "look around"
            statuses[i] = "empty_action"
            continue
        out[i] = cand[:200]
        valids[i] = 1
        statuses[i] = "react_ok" if len(matches) == 1 else "react_multi_action"

    return out, valids, statuses
