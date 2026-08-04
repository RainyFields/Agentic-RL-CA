"""Essential correctness tests for the lean async rollout collector (2026-08-04 plan).

CPU-only: a fake AsyncLLM server (records exact request/response ids + timing) and a
fake search-env manager (controllable episode lengths + search latency) drive the REAL
driver — real TrajectoryCollector.preprocess_single_sample, real Qwen tokenizer, real
row assembly, real _pr_groups bookkeeping.

Covers the 5 required cases:
  1. a trajectory that finishes on its first turn is never generated again
  2. trajectories with different lengths progress independently (no lockstep)
  3. search for one trajectory overlaps generation for another (timestamp proof)
  4. an unfinished trajectory snapshots at the cycle boundary and resumes correctly
  5. uid/group membership, rewards, tokens, policy versions remain correct

Run:  PYTHONPATH=. python tests/test_async_rollout.py
"""
import asyncio
import sys
import zlib
import time
from collections import defaultdict

import numpy as np
import torch
from omegaconf import OmegaConf

from verl import DataProto
from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector

MODEL = "/mnt/hdfs/mlsys/models/Qwen3-8B-Base"
RESP_CAP = 16


def make_config(n=2, cycle_turns=3, max_steps=6, routing="sticky"):
    return OmegaConf.create({
        "data": {"max_prompt_length": 512, "truncation": "left",
                 "return_raw_chat": False, "apply_chat_template_kwargs": {}},
        "env": {"rollout": {"n": n}, "max_steps": max_steps,
                "async_rollout_enable": True,
                "async_rollout_routing": routing,
                "partial_rollout_cycle_turns": cycle_turns,
                "partial_rollout_max_age": 4},
        "actor_rollout_ref": {"rollout": {
            "temperature": 1.0, "top_p": 1.0, "top_k": -1,
            "response_length": RESP_CAP,
            "val_kwargs": {"temperature": 0.6, "top_p": 1.0, "top_k": -1,
                           "do_sample": False}}},
        "algorithm": {"adv_estimator": "grpo", "filter_groups": {"enable": False}},
    })


class FakeServer:
    """Mimics AsyncvLLMServer.generate_token_ids as a Ray-free awaitable .remote()."""

    def __init__(self, eos_id, gen_delay=0.05, delay_fn=None):
        self.eos_id = eos_id
        self.gen_delay = gen_delay
        self.delay_fn = delay_fn  # optional (prompt_ids) -> seconds, for engineered tests
        self.calls = []          # (request_id, prompt_ids, t_start, t_end)
        self.responses = {}      # request_id -> token_ids returned
        outer = self

        class _M:
            def remote(self, prompt_ids, sp, rid):
                return outer._gen(prompt_ids, sp, rid)
        self.generate_token_ids = _M()

        class _A:
            def remote(self, rid):
                async def _noop():
                    return None
                return _noop()
        self.abort_request = _A()

    async def _gen(self, prompt_ids, sp, rid):
        t0 = time.monotonic()
        if self.delay_fn is not None:
            await asyncio.sleep(self.delay_fn(prompt_ids))
        else:
            # heterogeneous latency per request: without it all trajectories stay
            # accidentally phase-aligned and timing tests can't discriminate
            await asyncio.sleep(self.gen_delay * (0.3 + (zlib.crc32(rid.encode()) % 100) / 50.0))
        # deterministic ids derived from the request; end with EOS
        ids = [100 + (len(self.calls) % 50), 200, self.eos_id]
        lps = [-0.1, -0.2, -0.3]
        t1 = time.monotonic()
        self.calls.append((rid, list(prompt_ids), t0, t1))
        self.responses[rid] = list(ids)
        return {"token_ids": ids, "logprobs": lps, "finish_reason": "stop"}


class FakeEnvs:
    """Mimics SearchEnvironmentManager per-slot API + snapshot/success_evaluator."""

    class _Pool:
        def __init__(self, bs):
            self.batch_size = bs

    def __init__(self, pool, search_delay=0.05):
        self.envs = self._Pool(pool)
        self.search_delay = search_delay
        self.state = [None] * pool            # per-slot {steps, ep_len, task}
        self.step_log = []                    # (slot, task, t_start, t_end)
        self.reward_for = {}                  # task -> final reward

    def size_pool(self, pool):
        self.state = [None] * pool

    async def areset_one(self, idx, kwargs):
        self.state[idx] = {"steps": 0, "ep_len": int(kwargs["ep_len"]),
                           "task": kwargs["question"],
                           "search_delay": kwargs.get("search_delay")}
        self.reward_for.setdefault(kwargs["question"], float(kwargs.get("reward", 1.0)))
        return {"text": f"TASK {kwargs['question']} step 0",
                "anchor": kwargs["question"]}, {}

    async def arestore_one(self, idx, snap):
        self.state[idx] = dict(snap["env"])
        s = self.state[idx]
        return {"text": f"TASK {s['task']} step {s['steps']}", "anchor": s["task"]}

    async def astep_one(self, idx, text_action):
        t0 = time.monotonic()
        s0 = self.state[idx]
        d = s0.get("search_delay")
        if d is None:
            d = self.search_delay * (0.3 + (zlib.crc32((s0["task"] + str(s0["steps"])).encode()) % 100) / 50.0)
        await asyncio.sleep(d)
        s = self.state[idx]
        s["steps"] += 1
        done = s["steps"] >= s["ep_len"]
        reward = self.reward_for[s["task"]] if done else 0.0
        info = {"is_action_valid": True, "won": done and reward > 0,
                "tool_calling": 1.0, "data_source": "fake"}
        t1 = time.monotonic()
        self.step_log.append((idx, s["task"], t0, t1))
        return ({"text": f"TASK {s['task']} step {s['steps']}", "anchor": s["task"]},
                reward, done, info)

    def snapshot(self, idxs):
        return [{"env": dict(self.state[i]), "memory": [], "task": self.state[i]["task"]}
                for i in idxs]

    def success_evaluator(self, total_infos, total_batch_list, episode_rewards,
                          episode_lengths):
        return {"success_rate": np.array([float(r > 0) for r in episode_rewards])}


def make_gen_batch(questions, ep_lens, tokenizer, policy_version=7, rewards=None,
                   search_delays=None):
    B = len(questions)
    raw_prompt = np.empty(B, dtype=object)
    data_source = np.empty(B, dtype=object)
    env_kwargs = np.empty(B, dtype=object)
    for i, q in enumerate(questions):
        raw_prompt[i] = [{"role": "user", "content": q}]
        data_source[i] = "fake"
        env_kwargs[i] = {"question": q, "ep_len": ep_lens[i],
                         "reward": (rewards or {}).get(q, 1.0), "ground_truth": "x",
                         "search_delay": (search_delays or {}).get(q)}
    gb = DataProto.from_single_dict({
        "input_ids": torch.zeros(B, 4, dtype=torch.long),
        "attention_mask": torch.ones(B, 4, dtype=torch.long),
        "raw_prompt": raw_prompt, "data_source": data_source,
        "env_kwargs": env_kwargs,
    })
    gb.meta_info = {"eos_token_id": tokenizer.eos_token_id, "do_sample": True,
                    "policy_version": policy_version}
    return gb


class FakeManager:
    def __init__(self, servers):
        self.async_llm_servers = servers


def run_collection(collector, gb, servers, envs):
    return collector.multi_turn_loop(
        gen_batch=gb, actor_rollout_wg=None, envs=envs, is_train=True,
        async_rollout_manager=FakeManager(servers))


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    eos = tok.eos_token_id
    failures = []

    def check(name, cond, detail=""):
        status = "PASS" if cond else "FAIL"
        print(f"[{status}] {name}" + (f" — {detail}" if detail else ""), flush=True)
        if not cond:
            failures.append(name)

    # ---------------- tests 1-3 + 5: one collection, mixed lengths ----------------
    cfg = make_config(n=2, cycle_turns=8, max_steps=8)
    collector = TrajectoryCollector(config=cfg, tokenizer=tok, processor=None)
    servers = [FakeServer(eos, gen_delay=0.08), FakeServer(eos, gen_delay=0.08)]
    envs = FakeEnvs(pool=4, search_delay=0.08)
    # 2 groups x n=2 members; group A episodes finish in 1 turn, group B in 4 turns
    gb = make_gen_batch(["qA", "qB"], [1, 4], tok, policy_version=7,
                        rewards={"qA": 1.0, "qB": 0.0})
    t0 = time.monotonic()
    out = run_collection(collector, gb, servers, envs)
    wall = time.monotonic() - t0

    all_calls = servers[0].calls + servers[1].calls
    by_traj = defaultdict(list)
    for rid, pids, a, b in all_calls:
        by_traj[rid.split(":")[1]].append((rid, a, b))

    # test 1: first-turn finishers generated exactly once
    rows = out.non_tensor_batch
    tuid_by_uid = defaultdict(set)
    for i in range(len(out.batch["responses"])):
        tuid_by_uid[rows["uid"][i]].add(rows["traj_uid"][i])
    one_turn_trajs = [t for t, calls in by_traj.items() if len(calls) == 1]
    four_turn_trajs = [t for t, calls in by_traj.items() if len(calls) == 4]
    check("1. first-turn finish never regenerated",
          len(one_turn_trajs) == 2 and len(four_turn_trajs) == 2 and len(by_traj) == 4,
          f"calls per traj: {sorted(len(c) for c in by_traj.values())}")

    # test 2: independent progress — serial time would be (2*1 + 2*4) turns * 0.16s
    serial = (2 * 1 + 2 * 4) * 0.16
    check("2. no lockstep (wall << serial)", wall < serial * 0.7,
          f"wall {wall:.2f}s vs serial {serial:.2f}s")

    # test 3: ENGINEERED overlap scenario (deterministic): trajectory qL has slow
    # generation (0.5s), trajectory qS has fast generation but slow search (0.3s).
    # qS's search must run strictly inside qL's generation window, and qS must submit
    # its NEXT generation before qL's current generation finishes.
    cfg3 = make_config(n=1, cycle_turns=8, max_steps=8)
    coll3 = TrajectoryCollector(config=cfg3, tokenizer=tok, processor=None)

    def delay_fn(pids):
        return 0.5 if "qL" in tok.decode(pids) else 0.01
    srv3 = FakeServer(eos, delay_fn=delay_fn)
    envs3 = FakeEnvs(pool=2, search_delay=0.01)
    gb3 = make_gen_batch(["qL", "qS"], [3, 3], tok,
                         search_delays={"qL": 0.01, "qS": 0.3})
    run_collection(coll3, gb3, [srv3], envs3)

    def call_task3(pids):
        return "qL" if "qL" in tok.decode(pids) else "qS"
    overlap = False
    for slot, task, e0, e1 in envs3.step_log:
        if task != "qS":
            continue
        for rid, pids, g0, g1 in srv3.calls:
            if call_task3(pids) == "qL" and g0 < e0 and e1 < g1:
                overlap = True
                break
        if overlap:
            break
    submit_during_other = False
    for r1, p1, a1, b1 in srv3.calls:
        if call_task3(p1) != "qS" or int(r1.split(":")[2]) == 0:
            continue
        for r2, p2, a2, b2 in srv3.calls:
            if call_task3(p2) == "qL" and a2 < a1 < b2:
                submit_during_other = True
                break
        if submit_during_other:
            break
    check("3. search overlaps generation across trajectories",
          overlap and submit_during_other,
          f"qS-search-inside-qL-gen={overlap} qS-next-gen-during-qL-gen={submit_during_other}")

    # test 5a: released batch has whole uid groups, exact tokens, policy version
    n_rows = len(out.batch["responses"])
    group_ok = all(len(m) == 2 for m in tuid_by_uid.values())
    pv_ok = all(int(v) == 7 for v in rows["policy_version"])
    tok_ok = True
    for i in range(n_rows):
        rid = None  # match by traj+turn
        t, turn = rows["traj_uid"][i], int(rows["turn_index"][i])
        want = None
        for r, pids, _, _ in all_calls:
            if r.split(":")[1] == t and int(r.split(":")[2]) == turn:
                want = servers[0].responses.get(r) or servers[1].responses.get(r)
        got = out.batch["responses"][i][:3].tolist()
        if want is None or got != want:
            tok_ok = False
    rew_ok = True
    for i in range(n_rows):
        # qA groups won (reward 1 on final turn), qB groups didn't
        r = float(rows["rewards"][i])
        er = float(rows["episode_rewards"][i])
        if rows["env_done"][i]:
            expect = 1.0 if by_traj[rows["traj_uid"][i]] and len(by_traj[rows["traj_uid"][i]]) == 1 else 0.0
            if abs(r - expect) > 1e-6 or abs(er - expect) > 1e-6:
                rew_ok = False
    check("5. uid groups whole / tokens exact / rewards / policy_version",
          group_ok and pv_ok and tok_ok and rew_ok and n_rows == 2 * 1 + 2 * 4,
          f"rows={n_rows} groups={dict((k[:6], len(v)) for k, v in tuid_by_uid.items())} "
          f"pv_ok={pv_ok} tok_ok={tok_ok} rew_ok={rew_ok}")

    # attention mask sanity: 3 real tokens -> mask sums to 3 on response side
    resp_attn = out.batch["attention_mask"][:, -RESP_CAP:]
    check("5b. response attention mask counts real tokens",
          bool((resp_attn.sum(-1) == 3).all()), f"sums={resp_attn.sum(-1).tolist()[:6]}")

    # ---------------- test 4: snapshot at cycle boundary + resume ----------------
    cfg2 = make_config(n=2, cycle_turns=3, max_steps=8)
    collector2 = TrajectoryCollector(config=cfg2, tokenizer=tok, processor=None)
    servers2 = [FakeServer(eos, gen_delay=0.02)]
    envs2 = FakeEnvs(pool=4, search_delay=0.02)
    # episodes need 5 turns; cycle budget is 3 -> must pause + resume
    gb1 = make_gen_batch(["qC"], [5], tok, policy_version=1)
    out1 = run_collection(collector2, gb1, servers2, envs2)
    pend1 = sum(len(g["pending"]) for g in collector2._pr_groups.values())
    check("4a. cycle 1: nothing released, 2 pending with snapshots",
          out1 is None and pend1 == 2,
          f"released={'none' if out1 is None else len(out1.batch['responses'])} pending={pend1}")
    snaps_ok = all(rec["snap"] is not None and rec["snap"]["env"]["steps"] == 3
                   and rec["turn_off"] == 3
                   for g in collector2._pr_groups.values()
                   for rec in g["pending"].values())
    check("4b. snapshots taken at exact end-of-turn state (steps=3, turn_off=3)", snaps_ok)

    gb2 = make_gen_batch(["qD"], [1], tok, policy_version=2)
    out2 = run_collection(collector2, gb2, servers2, envs2)
    assert out2 is not None
    r2 = out2.non_tensor_batch
    qc_rows = [i for i in range(len(out2.batch["responses"]))
               if int(r2["episode_lengths"][i]) == 5]
    turn_seq = sorted(int(r2["turn_index"][i]) for i in qc_rows)
    pv_mix = sorted(set(int(r2["policy_version"][i]) for i in qc_rows))
    check("4c. resumed trajectories released whole: turns 0..4 per member, "
          "policy versions {1,2} across the seam",
          len(qc_rows) == 10 and turn_seq == sorted([0, 1, 2, 3, 4] * 2) and pv_mix == [1, 2],
          f"n={len(qc_rows)} turns={turn_seq} pv={pv_mix}")

    # ---------------- test 6: least-loaded routing with prefix-affinity tiebreak ----
    cfg6 = make_config(n=2, cycle_turns=8, max_steps=8, routing="least_loaded")
    coll6 = TrajectoryCollector(config=cfg6, tokenizer=tok, processor=None)
    servers6 = [FakeServer(eos, gen_delay=0.20), FakeServer(eos, gen_delay=0.02)]
    envs6 = FakeEnvs(pool=4, search_delay=0.02)
    gb6 = make_gen_batch(["qE", "qF"], [4, 4], tok, policy_version=9)
    out6 = run_collection(coll6, gb6, servers6, envs6)
    c0, c1 = len(servers6[0].calls), len(servers6[1].calls)
    n_rows6 = len(out6.batch["responses"]) if out6 is not None else 0
    # the fast engine must absorb more requests than the 5x-slower one, both must
    # serve traffic, and the released batch stays complete (4 trajs x 4 turns)
    check("6. least-loaded routing balances load, output intact",
          c0 > 0 and c1 > c0 and n_rows6 == 16,
          f"slow-engine calls={c0} fast-engine calls={c1} rows={n_rows6}")

    print()
    if failures:
        print(f"FAILED: {failures}")
        return 1
    print("ALL ASYNC COLLECTOR TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
