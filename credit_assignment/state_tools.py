# Copyright 2026 Agentic-RL-CA project. Apache-2.0.
"""Phase 2b/3b — build exact env/manager snapshots from a logged trajectory prefix.

Search state is fully driver-local and deterministic given the query, so a prefix state is
completely determined by (question, ground_truth, data_source, max_turns, the sequence of
projected actions + retrieved <information> blocks). This lets the credit-alignment
diagnostic resume from prefixes of trajectories logged at training time WITHOUT having
stored live snapshots — it only needs the JSONL dumps.

Output snapshot dicts are exactly the structure SearchEnvironmentManager.restore_batch
expects: {"env": <SearchEnv state>, "memory": <SearchMemory._data[i]>, "task": question}.
Pure python — CPU-testable.
"""
from copy import deepcopy

from credit_assignment.step_rewards import b1_hit


def build_snapshot_from_prefix(
    question: str,
    ground_truth,
    data_source: str,
    max_turns: int,
    steps,
    track_b1: bool = False,
):
    """steps: list of dicts with keys
        "search"      — the PROJECTED action string stored by the manager
                        (e.g. "<search>query</search>"; projection output, not raw response)
        "information" — the raw observation string returned by the env for that action
                        (e.g. "\\n<information>...</information>\\n"; may be "")
    Returns a snapshot dict for one env slot at the state AFTER executing `steps`.
    """
    chat_history = []
    b1_given = False
    for s in steps:
        chat_history.append({"role": "assistant", "content": s["search"]})
        info = s.get("information", "")
        if info:
            chat_history.append({"role": "user", "content": info})
            if track_b1 and not b1_given and b1_hit(info, ground_truth):
                b1_given = True

    env_state = {
        "ground_truth": deepcopy(ground_truth),
        "max_turns": int(max_turns),
        "data_source": data_source,
        "chat_history": chat_history,
        "done": False,
        "turns": len(steps),
        "_b1_given": b1_given,
    }
    memory = [
        {"search": s["search"], "information": s.get("information", "")} for s in steps
    ]
    return {"env": env_state, "memory": memory, "task": question}
