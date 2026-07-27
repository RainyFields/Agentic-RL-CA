# Local validation for SciWorldEnvironmentManager: prompt assembly + 3-tier truncation.
# Run: /home/tiger/xiaoxuan/envs/agentic-rl-ca/bin/python tests/test_sciworld_manager.py
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from functools import partial
from omegaconf import OmegaConf
from agent_system.environments.env_package.sciworld import build_sciworld_envs, sciworld_projection
from agent_system.environments.env_manager import SciWorldEnvironmentManager

MODEL = "/mnt/hdfs/mlsys/models/Qwen3-4B-Instruct-2507"


def make_config(max_prompt_length=16384, recent_k=20, margin=768):
    return OmegaConf.create({
        "data": {"max_prompt_length": max_prompt_length},
        "actor_rollout_ref": {"model": {"path": MODEL}},
        "env": {
            "history_length": 999,
            "sciworld": {"reward_mode": "prm", "recent_k": recent_k,
                         "prompt_token_margin": margin, "env_step_limit": 2500,
                         "tokenizer_path": None},
        },
    })


def main():
    envs = build_sciworld_envs(seed=42, env_num=1, group_n=2,
                               resources_per_worker={"num_cpus": 0.5}, is_train=True,
                               env_kwargs={"reward_mode": "prm", "env_step_limit": 2500})
    mgr = SciWorldEnvironmentManager(envs, partial(sciworld_projection), make_config())

    obs, infos = mgr.reset({})
    p0 = obs["text"][0]
    assert "Task:" in p0 and "Observation (turn 1):" in p0 and "focus on" in p0
    assert obs["anchor"][0] in p0
    print("ok: initial prompt structure")
    print("---- initial prompt tokens:", mgr._ntok(p0))

    fake = "Thought: I will look around to see the room.\nAction: look around"
    for t in range(4):
        obs, rewards, dones, infos = mgr.step([fake, fake])
    p4 = obs["text"][0]
    assert "Observation (turn 5):" in p4
    assert p4.count("Thought: I will look around") == 4, "full recent turns must render"
    assert infos[0]["is_action_valid"]
    print("ok: history accumulates as full turns inside K")

    # tiny budget forces tier 2 (action-only) and tier 3 (drops) + emergency demotion
    mgr2 = SciWorldEnvironmentManager(envs, partial(sciworld_projection),
                                      make_config(max_prompt_length=1400, recent_k=3, margin=100))
    obs, infos = mgr2.reset({})
    for t in range(8):
        obs, rewards, dones, infos = mgr2.step([fake, fake])
    p = obs["text"][0]
    ntok = mgr2._ntok(p)
    assert ntok <= 1400 - 100, f"prompt {ntok} exceeds budget"
    c_before = dict(mgr2.trunc_counts)
    assert c_before["tier2_turns"] > 0, c_before
    print(f"ok: three-tier truncation under tiny budget (prompt={ntok} tok, counts={c_before})")

    # _process_batch surface
    total_batch_list = [[{"active_masks": True}]]
    total_infos = [[infos[0]]]
    from collections import defaultdict
    success = defaultdict(list)
    mgr2._process_batch(0, total_batch_list, total_infos, success)
    for k in ["success_rate", "sw_score", "sw_seq_progress", "sw_focus_death", "sw_cap_hit"]:
        assert k in success, k
    print("ok: _process_batch keys:", sorted(success.keys()))

    mgr.close()
    print("ALL PASS")


if __name__ == "__main__":
    main()
