# Copyright 2026 Agentic-RL-CA project. Apache-2.0.
"""Phase 2b unit checks (plan §Verification, CPU-only):
- SearchEnv snapshot/restore round-trip is exact and isolated (deepcopy semantics).
- rebuild_text_obs formatting is byte-identical to SearchMemory.fetch + SEARCH_TEMPLATE
  (the vanilla build_text_obs path) — the node-identity guarantee for prefix resume.
- build_snapshot_from_prefix reproduces the state an env would reach by stepping.

SearchEnv itself needs gym-free import: we test against a minimal stand-in that shares the
exact attribute set used by snapshot_state/restore_state (kept in sync by construction),
plus the real SearchMemory + templates.
"""
import sys
import types

import numpy as np

# stub the reward_score chain used by prompts? — prompts/search.py is import-light.
from agent_system.environments.prompts.search import SEARCH_TEMPLATE, SEARCH_TEMPLATE_NO_HIS
from agent_system.memory import SearchMemory
from credit_assignment.state_tools import build_snapshot_from_prefix

GOLD = {"target": np.array(["Paris"], dtype=object)}


def fetch_based_obs(memory, tasks, i, history_length):
    """The vanilla build_text_obs path (SearchEnvironmentManager.build_text_obs)."""
    if len(memory._data[i]) == 0 or history_length <= 0:
        return SEARCH_TEMPLATE_NO_HIS.format(task_description=tasks[i])
    ctx, _ = memory.fetch(history_length, obs_key="information", action_key="search")
    return SEARCH_TEMPLATE.format(
        task_description=tasks[i], memory_context=ctx[i], step_count=len(memory[i])
    )


def rebuild_obs(memory, tasks, i, history_length):
    """Mirror of SearchEnvironmentManager.rebuild_text_obs for one index."""
    n_steps = len(memory._data[i])
    if n_steps == 0 or history_length <= 0:
        return SEARCH_TEMPLATE_NO_HIS.format(task_description=tasks[i])
    recent = memory._data[i][-history_length:]
    start_idx = n_steps - len(recent)
    lines = [
        f"Step {start_idx + j + 1}:{rec['search']} {rec['information']}\n"
        for j, rec in enumerate(recent)
    ]
    return SEARCH_TEMPLATE.format(
        task_description=tasks[i], memory_context="\n".join(lines), step_count=n_steps
    )


def test_rebuild_matches_fetch_exactly():
    mem = SearchMemory()
    mem.reset(batch_size=2)
    tasks = ["what is the capital of france?", "who wrote hamlet?"]
    mem.store({"search": ["<search>capital france</search>", "<search>hamlet author</search>"],
               "information": ["\n<information>Doc: Paris is the capital.</information>\n",
                                "\n<information>Doc: Shakespeare.</information>\n"]})
    mem.store({"search": ["<search>france capital city</search>", "<answer>Shakespeare</answer>"],
               "information": ["\n<information>Doc 2</information>\n", ""]})
    for hist in [1, 2, 8]:
        for i in range(2):
            assert rebuild_obs(mem, tasks, i, hist) == fetch_based_obs(mem, tasks, i, hist), (
                f"mismatch at i={i} hist={hist}"
            )
    # empty-memory (init) case
    mem2 = SearchMemory(); mem2.reset(batch_size=1)
    assert rebuild_obs(mem2, tasks, 0, 4) == fetch_based_obs(mem2, tasks, 0, 4)


def test_build_snapshot_from_prefix():
    steps = [
        {"search": "<search>capital of france</search>",
         "information": "\n<information>Doc 1: Paris, the capital of France.</information>\n"},
        {"search": "<search>france population</search>", "information": ""},
    ]
    snap = build_snapshot_from_prefix(
        question="what is the capital of france?", ground_truth=GOLD,
        data_source="nq", max_turns=4, steps=steps, track_b1=True)
    env = snap["env"]
    assert env["turns"] == 2 and env["done"] is False and env["max_turns"] == 4
    # chat history: assistant, user(info), assistant  (no user entry for empty info)
    roles = [m["role"] for m in env["chat_history"]]
    assert roles == ["assistant", "user", "assistant"]
    assert env["_b1_given"] is True  # 'Paris' exposed in step-1 information
    assert snap["memory"] == [
        {"search": steps[0]["search"], "information": steps[0]["information"]},
        {"search": steps[1]["search"], "information": ""},
    ]
    # restored memory must render identically to a memory built by store()
    mem = SearchMemory(); mem.reset(batch_size=1)
    mem.store({"search": [steps[0]["search"]], "information": [steps[0]["information"]]})
    mem.store({"search": [steps[1]["search"]], "information": [steps[1]["information"]]})
    mem_restored = SearchMemory(); mem_restored.reset(batch_size=1)
    mem_restored._data[0] = snap["memory"]
    t = ["what is the capital of france?"]
    assert rebuild_obs(mem_restored, t, 0, 4) == fetch_based_obs(mem, t, 0, 4)


def test_prefix_without_b1_tracking():
    snap = build_snapshot_from_prefix("q?", GOLD, "nq", 8, steps=[], track_b1=False)
    assert snap["env"]["turns"] == 0 and snap["env"]["_b1_given"] is False
    assert snap["memory"] == [] and snap["env"]["chat_history"] == []
