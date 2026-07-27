# Local (CPU-only) validation for the rung-4 sciworld env package.
# Run: /home/tiger/xiaoxuan/envs/agentic-rl-ca/bin/python tests/test_sciworld_env.py
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent_system.environments.env_package.sciworld import (
    ROSTER, TURN_CAP, sciworld_projection, build_sciworld_envs)
from agent_system.environments.env_package.sciworld.envs import _seq_progress


def test_projection():
    acts, valids, statuses = sciworld_projection([
        "Thought: I should look.\nAction: look around",
        "Action: go to kitchen.",
        "no action line here",
        "Thought: hmm\nAction: open door\nAction: **go to outside**",
        "",
    ])
    assert acts[0] == "look around" and valids[0] == 1 and statuses[0] == "react_ok"
    assert acts[1] == "go to kitchen" and valids[1] == 1
    assert valids[2] == 0 and statuses[2] == "no_action_line"
    assert acts[3] == "go to outside" and statuses[3] == "react_multi_action"
    assert valids[4] == 0
    print("ok: projection")


def test_seq_parse():
    gp = ("Completed keys: \n----\nSequential Subgoals:\n----\n"
          "0\ttrue\t X\tfocus on substance\n1\tfalse\t Y\tliquid\n----\n"
          "Unordered and Optional Subgoals:\n----\n0\ttrue\t Z\tnear water\n")
    assert _seq_progress(gp) == (1, 2)
    print("ok: seq parse")


def test_envs():
    envs = build_sciworld_envs(seed=7, env_num=2, group_n=2,
                               resources_per_worker={"num_cpus": 0.5},
                               is_train=True,
                               env_kwargs={"reward_mode": "prm", "env_step_limit": 2500})
    obs, img, infos = envs.reset()
    assert len(obs) == 4 and img is None
    # group members share (task, variation)
    assert (infos[0]["task"], infos[0]["variation"]) == (infos[1]["task"], infos[1]["variation"])
    assert (infos[2]["task"], infos[2]["variation"]) == (infos[3]["task"], infos[3]["variation"])
    assert infos[0]["task"] in ROSTER
    assert infos[0]["task_description"]

    o, _, r, d, inf = envs.step(["look around"] * 4)
    assert len(o) == 4 and all(isinstance(x, float) for x in r)
    assert all(inf[i]["env_action_valid"] for i in range(4)), "look around must parse"
    o, _, r, d, inf = envs.step(["blargh nonsense"] * 4)
    assert not any(inf[i]["env_action_valid"] for i in range(4))
    envs.close()
    print("ok: envs reset/step/group-sharing")


def test_return_equivalence_and_death():
    # same seed => same (task, variation) draws in both reward modes
    def run(mode, actions_fn, seed=11):
        envs = build_sciworld_envs(seed=seed, env_num=1, group_n=1,
                                   resources_per_worker={"num_cpus": 0.5},
                                   is_train=True,
                                   env_kwargs={"reward_mode": mode, "env_step_limit": 2500})
        obs, _, infos = envs.reset()
        total, P, done, steps = 0.0, 0.0, False, 0
        last_info = infos[0]
        while not done and steps < 40:
            a = actions_fn(steps, last_info)
            obs, _, r, d, inf = envs.step([a])
            total += r[0]
            P = inf[0]["progress"]
            done = bool(d[0])
            last_info = inf[0]
            steps += 1
        envs.close()
        return total, P, last_info

    # trajectory A: idle waits (no progress)
    tot_prm, P_prm, _ = run("prm", lambda s, i: "wait")
    tot_orm, P_orm, info_orm = run("orm", lambda s, i: "wait")
    assert abs(tot_prm - P_prm) < 1e-9 and abs(tot_orm - P_orm) < 1e-9, \
        f"return equivalence broken: prm {tot_prm}/{P_prm} orm {tot_orm}/{P_orm}"
    assert abs(tot_prm - tot_orm) < 1e-9

    # trajectory B: immediate wrong focus -> death; clip means reward stays >= 0
    tot_d, P_d, info_d = run("prm", lambda s, i: "focus on agent")
    assert info_d["focus_death"] or info_d["cap_hit"] or P_d >= 0
    assert tot_d >= 0.0 and abs(tot_d - P_d) < 1e-9
    print(f"ok: return equivalence (idle P={P_prm:.3f}; death P={P_d:.3f}, "
          f"focus_death={info_d['focus_death']})")


def test_cap_hit():
    envs = build_sciworld_envs(seed=3, env_num=1, group_n=1,
                               resources_per_worker={"num_cpus": 0.5},
                               is_train=True,
                               env_kwargs={"reward_mode": "orm", "env_step_limit": 2500})
    obs, _, infos = envs.reset()
    cap = infos[0]["turn_cap"]
    done, steps = False, 0
    while not done and steps < cap + 5:
        _, _, _, d, inf = envs.step(["wait"])
        done, steps = bool(d[0]), steps + 1
    assert steps == cap and inf[0]["cap_hit"], f"cap {cap} vs steps {steps}, {inf[0]}"
    envs.close()
    print(f"ok: cap enforcement (task={infos[0]['task']}, cap={cap})")


def test_val_assignment_fixed():
    kw = {"reward_mode": "orm", "env_step_limit": 2500}
    a = build_sciworld_envs(seed=1007, env_num=6, group_n=1,
                            resources_per_worker={"num_cpus": 0.5}, is_train=False, env_kwargs=kw)
    _, _, i1 = a.reset()
    _, _, i2 = a.reset()
    assert [(x["task"], x["variation"]) for x in i1] == [(x["task"], x["variation"]) for x in i2]
    a.close()
    print("ok: val assignment fixed across resets:",
          [(x["task"], x["variation"]) for x in i1[:3]])


if __name__ == "__main__":
    test_projection()
    test_seq_parse()
    test_envs()
    test_return_equivalence_and_death()
    test_cap_hit()
    test_val_assignment_fixed()
    print("ALL PASS")
