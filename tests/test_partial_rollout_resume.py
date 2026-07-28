"""T1 (+T3-lite) from docs/design/2026-07-29_sync_partial_rollout.md.

Resume-identity test for synchronous partial rollout, using SCRIPTED actions (no GPU
model needed) against a live retriever (SEARCH_URL env, default localhost:8000).

  T1: run a trajectory k turns in manager A; snapshot; restore into manager B's TAIL
      slots via reset_partial (fresh episodes occupy the head); assert the rebuilt
      policy-visible prompt is byte-identical to A's next prompt, then step both
      managers with the same action and assert the following prompts also match.
  T3-lite: reset_partial with zero snapshots must equal a vanilla reset (byte-identical
      first prompts).

Run on a worker with the retriever up:
  $VENV/bin/python tests/test_partial_rollout_resume.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from omegaconf import OmegaConf

from agent_system.environments.env_manager import SearchEnvironmentManager
from agent_system.environments.env_package.search import build_search_envs, search_projection

SEARCH_URL = os.environ.get("SEARCH_URL", "http://127.0.0.1:8000/retrieve")

QUESTIONS = [
    {"question": "who won the first nobel prize in physics", "ground_truth": "Wilhelm Conrad Rontgen", "data_source": "t1"},
    {"question": "what is the capital city of australia", "ground_truth": "Canberra", "data_source": "t1"},
]
ACTIONS_T1 = "<think>I should look this up.</think><search>first nobel prize physics winner</search>"
ACTIONS_T2 = "<think>Let me refine.</think><search>Wilhelm Rontgen nobel prize 1901</search>"
ACTIONS_T3 = "<think>Done.</think><search>nobel laureate physics 1901</search>"


def make_manager(pool: int, asearcher_history: bool = True):
    cfg = OmegaConf.create({
        "env": {
            "env_name": "search",
            "seed": 0,
            "max_steps": 8,
            "history_length": 8,
            "rollout": {"n": 1},
            "asearcher_history": asearcher_history,
            "ctx_terminate": False,
            "search": {
                "search_url": SEARCH_URL,
                "topk": 5,
                "timeout": 90,
                "log_requests": False,
                "info_char_cap": 5000,
            },
        },
        "data": {"max_prompt_length": 16384, "truncation": "left"},
        "actor_rollout_ref": {"model": {"path": ""}},
    })
    envs = build_search_envs(seed=0, env_num=pool, group_n=1, is_train=True, env_config=cfg.env)
    return SearchEnvironmentManager(envs, search_projection, cfg)


def main():
    B = len(QUESTIONS)
    # ---- reference trajectory in manager A ----
    mgr_a = make_manager(pool=B + 2)
    obs_a, _ = mgr_a.reset(kwargs=QUESTIONS)
    obs_a1, _, _, _ = mgr_a.step([ACTIONS_T1] * B)
    obs_a2, _, _, _ = mgr_a.step([ACTIONS_T2] * B)   # A's prompt after 2 turns
    snaps = mgr_a.snapshot(list(range(B)))            # snapshot at exactly this point

    # ---- T3-lite: reset_partial with no snapshots == vanilla reset ----
    mgr_c = make_manager(pool=B + 2)
    obs_c, _ = mgr_c.reset_partial(kwargs=QUESTIONS, snapshots=[])
    mgr_c2 = make_manager(pool=B + 2)
    obs_c2, _ = mgr_c2.reset(kwargs=QUESTIONS)
    assert obs_c["text"] == obs_c2["text"], "T3-lite FAIL: reset_partial([]) != reset()"
    print("T3-lite PASS: reset_partial with zero snapshots == vanilla reset")

    # ---- T1: restore into TAIL slots behind one fresh episode ----
    mgr_b = make_manager(pool=B + 2)
    fresh = [{"question": "what is the tallest mountain on earth", "ground_truth": "Mount Everest", "data_source": "t1"}]
    obs_b, _ = mgr_b.reset_partial(kwargs=fresh, snapshots=snaps)
    assert len(obs_b["text"]) == 1 + B
    restored_prompts = obs_b["text"][1:]
    assert restored_prompts == obs_a2["text"], (
        "T1 FAIL: restored prompt != reference prompt\n--- restored ---\n%s\n--- reference ---\n%s"
        % (restored_prompts[0][:2000], obs_a2["text"][0][:2000]))
    print("T1 PASS (part 1): restored prompts byte-identical to reference")

    # step BOTH with the same action; prompts must stay identical (env state carried:
    # turns counter, chat_history, memory). Fresh slot gets a no-op think-only action.
    obs_a3, _, dones_a, _ = mgr_a.step([ACTIONS_T3] * B)
    obs_b3, _, dones_b, _ = mgr_b.step(["<think>fresh slot</think><search>everest height</search>"] + [ACTIONS_T3] * B)
    assert obs_b3["text"][1:] == obs_a3["text"], "T1 FAIL: prompts diverged after post-restore step"
    assert list(dones_b[1:]) == list(dones_a), "T1 FAIL: done flags diverged after restore"
    print("T1 PASS (part 2): post-restore step keeps prompts + dones byte-identical")

    # history-format sanity: with asearcher_history the prompt must retain the think text
    assert "<think>I should look this up.</think>" in obs_a2["text"][0], \
        "asearcher_history FAIL: think digest missing from history"
    assert "<information>" in obs_a2["text"][0], "history missing <information> block"
    print("format PASS: think digest + information blocks retained in history")
    print("ALL PASS")


if __name__ == "__main__":
    main()
