"""Lean async rollout collector — Agentic-RL-CA, 2026-08-04 plan.

Replaces the turn-level lockstep of `TrajectoryCollector._turn_loop`
    [full-batch generate -> barrier -> full-batch env step -> barrier] x turns
with trajectory-level asynchronous execution: each trajectory is one coroutine
    generate -> env step (search) -> next generate -> ... -> done/pause
that never waits on unrelated trajectories. The OUTER trainer stays synchronous:
one policy per collection, engine sleeps during the FSDP update.

Semantics preserved from the sync engine (see rollout_loop.py):
  - prompt construction/truncation: reuses preprocess_single_sample verbatim
  - response tensor assembly: mirrors vllm_rollout_spmd.generate_sequences
    (pad to response_length, get_response_mask, position-id extension)
  - per-turn metadata: uid / traj_uid / policy_version / turn_index / rewards /
    is_action_valid / early-stop counters / rollout_records — same keys, same types
  - partial rollout: SAME `_pr_groups` state format (uid-atomic release, max_age
    staleness drop, snapshot/resume) — sync and async collections interoperate
  - GRPO group integrity is an OUTPUT-ASSEMBLY property here (rows buffered per
    uid; only complete groups released), never a scheduling barrier.

Scheduling:
  - bounded slot pool = the env pool; resumed trajectories are packed first
    (staleness priority), fresh prompts queue and BACKFILL freed slots immediately
  - a finished trajectory releases its slot at once and is never generated again
  - per-trajectory turn budget `env.partial_rollout_cycle_turns` bounds staleness
    per collection (global horizon still enforced by the env's own max_turns)
  - drain (collection boundary): stop submitting new turns; each in-flight turn
    runs to its end-of-turn checkpoint under a bounded timeout (generation is
    capped by response_length; a hard grace timeout cancels + aborts the request
    and discards the partial turn — the env has NOT stepped, so snapshotting at
    the previous turn boundary is exact).
"""
import asyncio
import time
import uuid
from collections import deque
from typing import Dict, List, Optional

import numpy as np
import torch

from verl import DataProto
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _sampling_params(cfg_rollout, meta_info) -> Dict:
    """Per-request SamplingParams kwargs, mirroring vllm_rollout_spmd's
    do_sample / validate / train branches."""
    do_sample = bool(meta_info.get("do_sample", True))
    validate = bool(meta_info.get("validate", False))
    if not do_sample:
        sp = dict(temperature=0.0, top_p=1.0, top_k=-1)
    elif validate:
        vk = cfg_rollout.get("val_kwargs", None)
        sp = dict(
            temperature=float(vk.temperature) if vk is not None else 1.0,
            top_p=float(vk.top_p) if vk is not None else 1.0,
            top_k=int(vk.top_k) if vk is not None else -1,
        )
    else:
        sp = dict(
            temperature=float(cfg_rollout.temperature),
            top_p=float(cfg_rollout.top_p),
            top_k=int(cfg_rollout.get("top_k", -1)),
        )
    sp["n"] = 1
    sp["max_tokens"] = int(cfg_rollout.response_length)
    sp["logprobs"] = 0  # sampled-token logprob -> rollout_log_probs (diff metric)
    return sp


def _make_turn_row(collector, prow: Dict, resp_ids: List[int],
                   resp_logprobs: Optional[List[float]], eos_token_id) -> Dict:
    """Assemble one training row (tensor part) from the prompt-side dict produced by
    preprocess_single_sample and the generated response ids. Mirrors the sync path:
    vllm_rollout_spmd.generate_sequences tensor block, specialized to one row."""
    cfg_rollout = collector.config.actor_rollout_ref.rollout
    resp_cap = int(cfg_rollout.response_length)
    pad_id = collector.tokenizer.pad_token_id

    prompt_ids = prow["input_ids"]              # (P,) left-padded
    prompt_attn = prow["attention_mask"]        # (P,)
    prompt_pos = prow["position_ids"]           # (P,)

    resp = pad_2d_list_to_length([list(resp_ids)], pad_id, max_length=resp_cap)[0]
    if resp_logprobs is not None:
        rlp = pad_2d_list_to_length([list(resp_logprobs)], -1, max_length=resp_cap)[0].to(torch.float32)
    else:
        rlp = torch.full((resp_cap,), -1.0, dtype=torch.float32)

    seq = torch.cat([prompt_ids, resp], dim=-1)
    delta_pos = torch.arange(1, resp_cap + 1, device=prompt_pos.device)
    resp_pos = prompt_pos[-1:] + delta_pos
    position_ids = torch.cat([prompt_pos, resp_pos], dim=-1)
    resp_attn = get_response_mask(
        response_id=resp.unsqueeze(0), eos_token=eos_token_id, dtype=prompt_attn.dtype
    )[0]
    attention_mask = torch.cat([prompt_attn, resp_attn], dim=-1)

    row = {
        "prompts": prompt_ids,
        "responses": resp,
        "input_ids": seq,
        "rollout_log_probs": rlp,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        # prompt-side non-tensors carried through (same as sync union)
        "prompt_pretrunc_len": prow["prompt_pretrunc_len"],
        "anchor_obs": prow["anchor_obs"],
        "index": prow["index"],
        "data_source": prow["data_source"],
        "response_token_count": np.int64(resp_attn.sum().item()),
    }
    if "raw_prompt" in prow:
        row["raw_prompt"] = prow["raw_prompt"]
    return row


class _Ctx:
    """Shared driver state for one collection."""

    def __init__(self, collector, envs, servers, sp, eos_token_id, cycle_turns,
                 policy_version, profile: bool):
        self.collector = collector
        self.envs = envs
        self.servers = servers
        self.sp = sp
        self.eos_token_id = eos_token_id
        self.cycle_turns = cycle_turns
        self.policy_version = policy_version
        self.profile = profile
        self.stop = asyncio.Event()          # drain: no NEW turns after this
        self.run_id = uuid.uuid4().hex[:8]
        self.events: List[Dict] = []         # minimal per-turn timing events
        self.rollout_records: List[Dict] = []
        self.hard_grace_s = float(
            collector.config.env.get("async_rollout_hard_grace_s", 300.0))

    def event(self, **kw):
        if self.profile:
            self.events.append(kw)

    # ---- pass-2 diagnostics (env.async_rollout_snap_s > 0): per-slot phase tracking
    # for the periodic scheduler snapshot. Phases: gen (submitted to vLLM), env
    # (waiting on tool/search), idle/done (slot free). ----
    def init_diag(self, n_servers: int, snap_s: float):
        self.snap_s = snap_s
        self.slot_phase: Dict[int, str] = {}
        self.prompt_lens: List[int] = []
        self.finished_traj = 0

    def set_phase(self, slot: int, phase: str):
        if getattr(self, "snap_s", 0):
            self.slot_phase[slot] = phase


# --------------------------------------------------------------------------- #
# one trajectory = one coroutine
# --------------------------------------------------------------------------- #

async def _run_trajectory(ctx: _Ctx, slot: int, unit: Dict) -> Dict:
    """Run one trajectory in env slot `slot` for at most ctx.cycle_turns turns.

    unit: {uid, traj_uid, gen_row(DataProto 1 row), env_kwargs|None, snap|None,
           turn_off}
    Returns {unit, rows, infos, ep_rew, ep_len, tool, done, snap(new)|None}.
    """
    envs, collector = ctx.envs, ctx.collector
    tokenizer = collector.tokenizer

    if unit.get("snap") is not None:
        obs = await envs.arestore_one(slot, unit["snap"])
    else:
        obs, _ = await envs.areset_one(slot, unit["env_kwargs"])

    rows, infos = [], []
    ep_rew = 0.0
    tool_c = 0.0
    t_off = int(unit["turn_off"])
    invalid = repeated = nochange = 0
    prev_action = prev_obs_text = None
    early_stopped, early_reason = False, ""
    done = False
    steps = 0

    es_invalid = collector.config.env.get("early_stop_invalid_count", None)
    es_repeated = collector.config.env.get("early_stop_repeated_action_count", None)
    es_nochange = collector.config.env.get("early_stop_no_state_change_count", None)

    while steps < ctx.cycle_turns and not done and not ctx.stop.is_set():
        prow = collector.preprocess_single_sample(
            item=0,
            gen_batch=unit["gen_row"],
            obs={"text": [obs["text"]], "image": None, "anchor": [obs["anchor"]]},
        )

        rid = f"{ctx.run_id}:{unit['traj_uid']}:{t_off + steps}"
        server = ctx.servers[slot % len(ctx.servers)]  # sticky per slot: prefix cache
        if getattr(ctx, "snap_s", 0):
            ctx.prompt_lens.append(len(prow["raw_prompt_ids"]))
            ctx.set_phase(slot, "gen")
        t0 = time.monotonic()
        ref = server.generate_token_ids.remote(list(prow["raw_prompt_ids"]), ctx.sp, rid)
        try:
            gen = await asyncio.wait_for(asyncio.ensure_future(ref), ctx.hard_grace_s)
        except asyncio.TimeoutError:
            # drain hard-cancel: discard the partial turn — the env has NOT stepped,
            # so the trajectory state is exactly the previous end-of-turn checkpoint.
            import ray
            ray.cancel(ref, force=False)
            try:
                await server.abort_request.remote(rid)
            except Exception:
                pass
            print(f"[async_rollout] generation timeout ({ctx.hard_grace_s}s) "
                  f"traj={unit['traj_uid']} turn={t_off + steps} — turn discarded", flush=True)
            break
        t1 = time.monotonic()

        resp_ids = gen["token_ids"]
        text_action = tokenizer.decode(resp_ids, skip_special_tokens=True)
        row = _make_turn_row(collector, prow, resp_ids, gen["logprobs"], ctx.eos_token_id)

        ctx.set_phase(slot, "env")
        next_obs, reward, env_done, info = await envs.astep_one(slot, text_action)
        t2 = time.monotonic()
        ctx.event(slot=slot, traj=unit["traj_uid"], turn=t_off + steps,
                  t_gen0=t0, t_gen1=t1, t_env1=t2)

        # ---- per-turn bookkeeping (mirrors _turn_loop field-for-field) ----
        is_valid = bool(np.asarray(info.get("is_action_valid", True)).item()) \
            if not isinstance(info.get("is_action_valid", True), bool) else bool(info["is_action_valid"])
        if not is_valid:
            invalid += 1
        if prev_action is not None and text_action == prev_action:
            repeated += 1
        else:
            repeated = 0
        prev_action = text_action
        cur_obs_text = next_obs["text"]
        if prev_obs_text is not None and cur_obs_text == prev_obs_text:
            nochange += 1
        else:
            nochange = 0
        prev_obs_text = cur_obs_text

        if not env_done:
            reason = ""
            if es_invalid is not None and invalid >= es_invalid:
                reason = f"invalid_count>={es_invalid}"
            elif es_repeated is not None and repeated >= es_repeated:
                reason = f"repeated_action>={es_repeated}"
            elif es_nochange is not None and nochange >= es_nochange:
                reason = f"no_state_change>={es_nochange}"
            if reason:
                early_stopped, early_reason = True, reason

        if "tool_calling" in info:
            tool_c += float(info["tool_calling"])

        # numpy scalars, matching the sync path's element types — downstream code
        # calls numpy methods on per-item values (e.g. is_action_valid.astype at
        # apply_invalid_action_penalty; Python bool/float there crashes).
        row.update({
            "uid": unit["uid"],
            "traj_uid": unit["traj_uid"],
            "policy_version": np.int64(ctx.policy_version),
            "is_action_valid": np.bool_(is_valid),
            "tool_calling": bool(info.get("tool_calling", False)),
            "b1_hit": bool(info.get("b1_hit", False)),
            "rewards": np.float32(reward),
            "active_masks": np.bool_(True),
            "turn_index": np.int64(t_off + steps),
            "parse_status": str(info.get("parse_status", "unknown")),
            "env_reward": np.float32(reward),
            "env_done": np.bool_(env_done),
            "env_won": np.bool_(env_done and reward > 0),
            "invalid_count_so_far": np.int64(invalid),
            "repeated_count_so_far": np.int64(repeated),
            "early_stopped": np.bool_(early_stopped),
            "early_stop_reason": str(early_reason),
        })
        ctx.rollout_records.append({
            "traj_uid": str(unit["traj_uid"]), "turn_index": int(t_off + steps),
            "uid": str(unit["uid"]),
            "data_source": str(row.get("data_source", "")),
            "prompt_pretrunc_len": int(row.get("prompt_pretrunc_len", -1)),
            "parse_status": str(info.get("parse_status", "unknown")),
            "is_action_valid": is_valid,
            "env_reward": float(reward), "env_done": bool(env_done),
            "env_won": bool(env_done and reward > 0),
            "invalid_count_so_far": int(invalid),
            "repeated_count_so_far": int(repeated),
            "response_token_count": int(row["response_token_count"]),
            "early_stopped": bool(early_stopped), "early_stop_reason": str(early_reason),
            "observation": obs["text"],
            "raw_model_response": text_action,
            "sw_task": str(info.get("task", "")),
            "sw_variation": int(info.get("variation", -1)),
            "sw_progress": float(info.get("progress", 0.0)),
            "sw_won": bool(info.get("won", False)),
            "sw_focus_death": bool(info.get("focus_death", False)),
            "sw_cap_hit": bool(info.get("cap_hit", False)),
            "sw_env_crash": bool(info.get("env_crash", False)),
            "information": str(next_obs.get("anchor", "")),
        })

        rows.append(row)
        infos.append(info)
        ep_rew += float(reward)
        steps += 1
        done = bool(env_done) or early_stopped
        obs = next_obs

    ctx.set_phase(slot, "done")
    if getattr(ctx, "snap_s", 0) and done:
        ctx.finished_traj += 1
    snap = None
    if not done:
        snap = envs.snapshot([slot])[0]

    return {
        "unit": unit, "rows": rows, "infos": infos,
        "ep_rew": ep_rew, "ep_len": float(t_off + steps), "tool": tool_c,
        "done": done, "snap": snap,
    }


# --------------------------------------------------------------------------- #
# the driver: bounded slot pool, immediate backfill, no barriers
# --------------------------------------------------------------------------- #

async def _collect(ctx: _Ctx, units: List[Dict], pool: int,
                   drain_timeout_s: Optional[float]) -> List[Dict]:
    queue = deque(units)
    free_slots = list(range(pool))
    running: Dict[asyncio.Task, Dict] = {}
    results: List[Dict] = []

    snap_task = None
    if getattr(ctx, "snap_s", 0):
        async def _snapshots():
            n_srv = len(ctx.servers)
            while True:
                await asyncio.sleep(ctx.snap_s)
                phases = list(ctx.slot_phase.items())
                by = {"gen": 0, "env": 0, "done": 0}
                per_engine = [0] * n_srv
                for s, p in phases:
                    by[p] = by.get(p, 0) + 1
                    if p == "gen":
                        per_engine[s % n_srv] += 1
                pl = sorted(ctx.prompt_lens[-2048:])
                pct = (lambda q: pl[min(len(pl) - 1, int(q * len(pl)))]) if pl else (lambda q: 0)
                print(f"[async_snap] t={time.monotonic():.0f} in_gen={by['gen']} "
                      f"in_env={by['env']} slots_done={by['done']} "
                      f"finished_traj={ctx.finished_traj} "
                      f"gen_per_engine={per_engine} "
                      f"ctx_p50={pct(0.5)} ctx_p90={pct(0.9)} ctx_p99={pct(0.99)}",
                      flush=True)
        snap_task = asyncio.create_task(_snapshots())

    watchdog = None
    if drain_timeout_s:
        async def _watch():
            await asyncio.sleep(drain_timeout_s)
            ctx.stop.set()
            print(f"[async_rollout] drain: stop submitting new turns after "
                  f"{drain_timeout_s}s", flush=True)
        watchdog = asyncio.create_task(_watch())

    while queue or running:
        while queue and free_slots and not ctx.stop.is_set():
            unit = queue.popleft()
            slot = free_slots.pop()
            task = asyncio.create_task(_run_trajectory(ctx, slot, unit))
            running[task] = (slot, unit)
        if not running:
            break  # stop set; unstarted units are handled by the caller
        done_set, _ = await asyncio.wait(running.keys(), return_when=asyncio.FIRST_COMPLETED)
        for task in done_set:
            slot, unit = running.pop(task)
            free_slots.append(slot)
            exc = task.exception()
            if exc is not None:
                # lean failure handling: this trajectory restarts from scratch next
                # collection (nothing trained on it this cycle); log loudly.
                print(f"[async_rollout] trajectory {unit['traj_uid']} FAILED: "
                      f"{exc!r} — will restart next collection", flush=True)
                results.append({"unit": unit, "rows": [], "infos": [],
                                "ep_rew": 0.0, "ep_len": float(unit["turn_off"]),
                                "tool": 0.0, "done": False, "snap": None,
                                "failed": True})
            else:
                results.append(task.result())

    if snap_task is not None:
        snap_task.cancel()
    if watchdog is not None:
        watchdog.cancel()
    # unstarted units (drain fired while queued): nothing generated, nothing to
    # snapshot — report them back so group bookkeeping can requeue them fresh.
    for unit in queue:
        results.append({"unit": unit, "rows": [], "infos": [], "ep_rew": 0.0,
                        "ep_len": float(unit["turn_off"]), "tool": 0.0,
                        "done": False, "snap": None, "unstarted": True})
    return results


# --------------------------------------------------------------------------- #
# entry points (called from TrajectoryCollector.multi_turn_loop)
# --------------------------------------------------------------------------- #

def _overlap_metrics(ctx: _Ctx, wall_s: float) -> Dict:
    """Cheap aggregate proof-of-overlap numbers from the event list."""
    m = {"async/wall_s": wall_s, "async/turns": len(ctx.rollout_records)}
    if not ctx.events:
        return m
    gen_iv = [(e["t_gen0"], e["t_gen1"]) for e in ctx.events]
    env_iv = [(e["t_gen1"], e["t_env1"]) for e in ctx.events]
    m["async/gen_busy_s"] = sum(b - a for a, b in gen_iv)
    m["async/env_busy_s"] = sum(b - a for a, b in env_iv)
    # max concurrent in-flight generations (sweep line)
    pts = sorted([(a, 1) for a, _ in gen_iv] + [(b, -1) for _, b in gen_iv])
    cur = peak = 0
    for _, d in pts:
        cur += d
        peak = max(peak, cur)
    m["async/max_concurrent_gen"] = peak
    # timestamp overlap proof: at least one env step strictly inside another
    # trajectory's generation interval
    overlaps = 0
    gen_sorted = sorted(gen_iv)
    import bisect
    starts = [a for a, _ in gen_sorted]
    for i, e in enumerate(ctx.events):
        a, b = env_iv[i]
        j = bisect.bisect_right(starts, a)
        for g0, g1 in gen_sorted[max(0, j - 64):j + 1]:
            if g0 < a and b < g1 and ctx.events[i]["slot"] != -1:
                overlaps += 1
                break
    m["async/env_inside_other_gen"] = overlaps
    return m


def async_multi_turn_loop(collector, gen_batch: DataProto, async_rollout_manager,
                          envs, is_train: bool = True):
    """Async replacement for the lockstep collection. Routed from
    TrajectoryCollector.multi_turn_loop when +env.async_rollout_enable=true.

    Train: partial-rollout semantics over the SAME collector._pr_groups state
    (uid-atomic release, max_age, snapshot/resume). Val: every prompt runs to
    completion, vanilla-equivalent output.
    """
    cfg = collector.config
    cfg_env = cfg.env
    servers = async_rollout_manager.async_llm_servers
    sp = _sampling_params(cfg.actor_rollout_ref.rollout, gen_batch.meta_info)
    eos_token_id = gen_batch.meta_info.get("eos_token_id", collector.tokenizer.eos_token_id)
    profile = bool(cfg_env.get("async_rollout_profile", True))
    pool = envs.envs.batch_size
    n = cfg_env.rollout.n if cfg_env.rollout.n > 0 else 1

    if is_train:
        cycle_turns = int(cfg_env.get("partial_rollout_cycle_turns", cfg_env.max_steps))
        policy_version = int(gen_batch.meta_info.get("policy_version", -1))
    else:
        cycle_turns = int(cfg_env.max_steps)
        policy_version = -1

    ctx = _Ctx(collector, envs, servers, sp, eos_token_id, cycle_turns,
               policy_version, profile)
    ctx.init_diag(len(servers), float(cfg_env.get("async_rollout_snap_s", 0) or 0))
    drain_timeout = cfg_env.get("async_rollout_drain_timeout_s", None)
    envs.size_pool(pool)

    # ---------------- build the work queue ----------------
    units: List[Dict] = []
    if is_train:
        if not hasattr(collector, "_pr_groups"):
            collector._pr_groups = {}
            collector._pr_cycle = 0
        collector._pr_cycle += 1
        max_age = int(cfg_env.get("partial_rollout_max_age", 4))

        # staleness bound: drop whole groups with over-age pending members
        dropped = [u for u, g in collector._pr_groups.items()
                   if g["pending"] and (collector._pr_cycle - g["born"]) > max_age]
        for u in dropped:
            del collector._pr_groups[u]

        # resumed units FIRST (oldest work drains first)
        for u, g in collector._pr_groups.items():
            for t, rec in list(g["pending"].items()):
                units.append({"uid": u, "traj_uid": t, "gen_row": rec["gen_row"],
                              "env_kwargs": rec.get("env_kwargs"),
                              "snap": rec["snap"], "turn_off": rec["turn_off"]})
        # fresh groups: every gen_batch prompt enters the queue (no drop-surplus;
        # freed slots backfill mid-collection)
        for gi in range(len(gen_batch)):
            gen_row = gen_batch.select_idxs([gi])
            ek = gen_row.non_tensor_batch.pop("env_kwargs", None)
            ek = ek[0] if ek is not None else None
            assert ek is not None, "async rollout: search env requires env_kwargs per prompt"
            u = str(uuid.uuid4())
            members = [str(uuid.uuid4()) for _ in range(n)]
            collector._pr_groups[u] = {"born": collector._pr_cycle,
                                       "members": set(members), "done": {}, "pending": {}}
            for t in members:
                units.append({"uid": u, "traj_uid": t, "gen_row": gen_row,
                              "env_kwargs": ek, "snap": None, "turn_off": 0})
    else:
        ek_arr = gen_batch.non_tensor_batch.pop("env_kwargs", None)
        uid = None
        for gi in range(len(gen_batch)):
            if gi % n == 0:
                uid = str(uuid.uuid4())
            units.append({"uid": uid, "traj_uid": str(uuid.uuid4()),
                          "gen_row": gen_batch.select_idxs([gi]),
                          "env_kwargs": ek_arr[gi] if ek_arr is not None else None,
                          "snap": None, "turn_off": 0})

    # ---------------- run ----------------
    t_wall0 = time.monotonic()
    results = asyncio.run(_collect(ctx, units, pool, drain_timeout))
    wall_s = time.monotonic() - t_wall0
    collector._dump_rollout_records(ctx.rollout_records)
    am = _overlap_metrics(ctx, wall_s)
    print(f"[async_rollout] cycle done: {len(results)} trajectories, "
          f"{am.get('async/turns', 0)} turns, wall {wall_s:.1f}s, "
          f"max_concurrent_gen {am.get('async/max_concurrent_gen', 0)}, "
          f"env-inside-gen overlaps {am.get('async/env_inside_other_gen', 0)}", flush=True)

    # ---------------- assemble output ----------------
    if not is_train:
        total_batch_list = [r["rows"] for r in results if r["rows"]]
        total_infos = [r["infos"] for r in results if r["rows"]]
        ep_rew = np.array([r["ep_rew"] for r in results if r["rows"]], dtype=np.float32)
        ep_len = np.array([r["ep_len"] for r in results if r["rows"]], dtype=np.float32)
        tools = np.array([r["tool"] for r in results if r["rows"]], dtype=np.float32)
        tuids = np.array([r["unit"]["traj_uid"] for r in results if r["rows"]], dtype=object)
        success = envs.success_evaluator(
            total_infos=total_infos, total_batch_list=total_batch_list,
            episode_rewards=ep_rew, episode_lengths=ep_len)
        return collector._finalize_rollout(total_batch_list, ep_rew, ep_len,
                                           success, tuids, tools)

    # train: merge results into _pr_groups (same bookkeeping as partial_multi_turn_loop)
    for r in results:
        unit = r["unit"]
        u, t = unit["uid"], unit["traj_uid"]
        g = collector._pr_groups.get(u)
        if g is None:
            continue  # group was dropped stale while running (should not happen)
        prev = g["pending"].pop(t, None)
        rows = (prev["rows"] + r["rows"]) if prev else r["rows"]
        infos = (prev["infos"] + r["infos"]) if prev else r["infos"]
        rew = (prev["ep_rew"] if prev else 0.0) + r["ep_rew"]
        tools = (prev["tool"] if prev else 0.0) + r["tool"]
        if r["done"]:
            g["done"][t] = {"rows": rows, "infos": infos, "ep_rew": rew,
                            "ep_len": r["ep_len"], "tool": tools}
        else:
            # unfinished (budget/drain/unstarted/failed): snapshot if we have one,
            # else the trajectory restarts from scratch next collection
            restart = r["snap"] is None
            g["pending"][t] = {
                "rows": [] if restart else rows,
                "infos": [] if restart else infos,
                "snap": r["snap"],
                "gen_row": unit["gen_row"],
                "env_kwargs": unit.get("env_kwargs"),
                "turn_off": 0 if restart else int(r["ep_len"]),
                "ep_rew": 0.0 if restart else rew,
                "ep_len": r["ep_len"], "tool": 0.0 if restart else tools,
            }

    # atomic uid-group release (identical to the sync partial loop)
    released = []
    for u in list(collector._pr_groups):
        g = collector._pr_groups[u]
        if not g["pending"] and len(g["done"]) == len(g["members"]):
            for t, rec in g["done"].items():
                released.append((u, t, rec))
            del collector._pr_groups[u]

    pend_traj = sum(len(g["pending"]) for g in collector._pr_groups.values())
    held_done = sum(len(g["done"]) for g in collector._pr_groups.values())
    ages = [collector._pr_cycle - g["born"]
            for g in collector._pr_groups.values() if g["pending"]]
    collector._partial_metrics = {
        "partial/released_traj": len(released),
        "partial/pending_traj": pend_traj,
        "partial/held_complete_traj": held_done,
        "partial/open_groups": len(collector._pr_groups),
        "partial/dropped_stale_groups": len(dropped),
        "partial/fresh_groups": len(gen_batch),
        "partial/resumed_traj": sum(1 for r in results if r["unit"]["snap"] is not None),
        "partial/mean_open_group_age": float(np.mean(ages)) if ages else 0.0,
        **am,
    }
    if not released:
        return None

    total_batch_list = [rec["rows"] for _, _, rec in released]
    total_infos = [rec["infos"] for _, _, rec in released]
    ep_rew = np.array([rec["ep_rew"] for _, _, rec in released], dtype=np.float32)
    ep_len = np.array([rec["ep_len"] for _, _, rec in released], dtype=np.float32)
    tools = np.array([rec["tool"] for _, _, rec in released], dtype=np.float32)
    tuids = np.array([t for _, t, _ in released], dtype=object)
    success = envs.success_evaluator(
        total_infos=total_infos, total_batch_list=total_batch_list,
        episode_rewards=ep_rew, episode_lengths=ep_len)
    out = collector._finalize_rollout(total_batch_list, ep_rew, ep_len,
                                      success, tuids, tools)
    out.meta_info["partial_metrics"] = collector._partial_metrics
    return out
