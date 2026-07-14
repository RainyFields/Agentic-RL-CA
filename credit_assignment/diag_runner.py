# Copyright 2026 Agentic-RL-CA project. Apache-2.0.
"""Phase 3b.2 — credit-alignment diagnostic GPU runner (offline; NEVER a training signal).

For one checkpoint: (A) sample base trajectories from the checkpoint policy, snapshotting
the exact driver-local state at every depth (Phase-2b infra); (B) for every prefix s_t,
sample K outcome-only continuations FROM THE SAME CHECKPOINT (plan §3b review point 4 —
continuations under any other policy make dV-hat a value under the wrong policy) and
estimate V-hat(s_t) = mean terminal EM. Output JSON feeds diagnostic.build_pairs +
credit_alignment_stats against each method's assigned turn advantages (dumped at
training time), pooled across trajectories with bootstrap CIs.

Standalone single-GPU vLLM — no ray/FSDP; run on a worker with the retriever healthy:
  python -m credit_assignment.diag_runner \
      --model <merged_hf_dir> --data data/searchR1_processed_direct/val_2048.parquet \
      --out outputs/diag/<label> --n-traj 256 --k 8 --k-check 16 --k-check-frac 0.125 \
      --search-url http://127.0.0.1:8000/retrieve
Protocol constants default to the LOCKED 4turn_think2k values; override to match the
checkpoint's training protocol (they MUST match or prefix states are off-policy).
"""
import argparse
import json
import os

import numpy as np
from omegaconf import OmegaConf

from credit_assignment.diag_plan import aggregate_vhat, plan_continuation_jobs


def build_env_manager(args, pool):
    from agent_system.environments.env_manager import SearchEnvironmentManager
    from agent_system.environments.env_package.search import build_search_envs, search_projection

    cfg = OmegaConf.create({
        "env": {
            "env_name": "search",
            "seed": args.seed,
            "max_steps": args.max_turns,
            "history_length": args.history_length,
            "rollout": {"n": 1},
            "search": {"search_url": args.search_url, "topk": args.topk,
                       "timeout": 30, "log_requests": False},
        },
        "data": {"max_prompt_length": args.max_prompt_length,
                 "truncation": args.truncation},
    })
    envs = build_search_envs(seed=args.seed, env_num=pool, group_n=1,
                             is_train=False, env_config=cfg.env)
    return SearchEnvironmentManager(envs, search_projection, cfg), cfg


class Generator:
    """Chat-templated, protocol-truncated prompt -> sampled response, via vLLM."""

    def __init__(self, args):
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        self.tokenizer = AutoTokenizer.from_pretrained(args.model)
        self.llm = LLM(model=args.model, dtype="bfloat16",
                       gpu_memory_utilization=args.gpu_memory_utilization,
                       max_model_len=args.max_prompt_length + args.max_response_length,
                       enforce_eager=False, seed=args.seed)
        self.sampling = SamplingParams(temperature=args.temperature, top_p=args.top_p,
                                       max_tokens=args.max_response_length)
        self.enable_thinking = args.enable_thinking
        self.max_prompt_length = args.max_prompt_length
        self.truncation = args.truncation

    def prompt_ids(self, obs_text):
        """Mirror of TrajectoryCollector's raw_prompt_ids pipeline (node identity)."""
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": obs_text}],
            add_generation_prompt=True, tokenize=False,
            enable_thinking=self.enable_thinking)
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(ids) > self.max_prompt_length:
            if self.truncation == "left":
                ids = ids[-self.max_prompt_length:]
            elif self.truncation == "right":
                ids = ids[:self.max_prompt_length]
        return ids

    def generate(self, obs_texts):
        try:
            from vllm import TokensPrompt
        except ImportError:  # older vLLM layouts
            from vllm.inputs import TokensPrompt
        prompts = [TokensPrompt(prompt_token_ids=self.prompt_ids(t)) for t in obs_texts]
        outs = self.llm.generate(prompts, self.sampling, use_tqdm=False)
        return [o.outputs[0].text for o in outs]


def run_turns(mgr, gen, obs, size, max_iters, snapshot_sink=None, depths0=None):
    """Drive `size` env slots from `obs` to termination. snapshot_sink(slot, depth, snap)
    is called pre-action at every visited depth (base-trajectory phase). Returns
    (terminal_rewards, n_turns_done) per slot."""
    depths0 = depths0 if depths0 is not None else np.zeros(size, dtype=np.int64)
    is_done = np.zeros(size, dtype=bool)
    terminal = np.zeros(size, dtype=np.float32)
    turns_done = depths0.astype(np.int64).copy()
    for it in range(max_iters):
        active = ~is_done
        if snapshot_sink is not None:
            act_idx = [i for i in range(size) if active[i]]
            snaps = mgr.snapshot(act_idx)
            for i, sn in zip(act_idx, snaps):
                snapshot_sink(i, int(depths0[i] + it), sn)
        texts = gen.generate([obs["text"][i] for i in range(size)])
        obs, rewards, dones, infos = mgr.step(texts)
        rewards = np.asarray(rewards, dtype=np.float32)
        dones = np.asarray(dones, dtype=bool).reshape(-1)
        for i in range(size):
            if active[i]:
                turns_done[i] += 1
                if dones[i]:
                    terminal[i] = rewards[i]
        is_done |= dones
        if is_done.all():
            break
    return terminal, turns_done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="merged HF checkpoint dir")
    ap.add_argument("--data", required=True, help="parquet with env_kwargs rows")
    ap.add_argument("--out", required=True, help="output dir")
    ap.add_argument("--n-traj", type=int, default=256)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--k-check", type=int, default=16)
    ap.add_argument("--k-check-frac", type=float, default=0.125)
    ap.add_argument("--pool", type=int, default=256, help="env worker slots")
    ap.add_argument("--seed", type=int, default=0)
    # LOCKED protocol defaults (configs/protocol_4turn_think2k.sh) — must match ckpt
    ap.add_argument("--max-turns", type=int, default=4)
    ap.add_argument("--history-length", type=int, default=4)
    ap.add_argument("--max-prompt-length", type=int, default=4096)
    ap.add_argument("--max-response-length", type=int, default=2048)
    ap.add_argument("--truncation", default="left")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--enable-thinking", action="store_true", default=True)
    ap.add_argument("--no-thinking", dest="enable_thinking", action="store_false")
    ap.add_argument("--search-url", default="http://127.0.0.1:8000/retrieve")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = ap.parse_args()

    import pandas as pd
    os.makedirs(args.out, exist_ok=True)
    df = pd.read_parquet(args.data)
    rng = np.random.RandomState(args.seed)
    rows = df.iloc[rng.choice(len(df), size=min(args.n_traj, len(df)), replace=False)]
    kwargs_all = [dict(r) for r in rows["env_kwargs"]]

    mgr, _ = build_env_manager(args, pool=args.pool)
    gen = Generator(args)

    # ---------------- phase A: base trajectories + per-depth snapshots -------------
    snapshots = {}      # traj_id -> {depth: snapshot}
    base = {}           # traj_id -> {"question", "data_source", "final_reward", "turns"}
    for start in range(0, len(kwargs_all), args.pool):
        chunk = kwargs_all[start:start + args.pool]
        size = len(chunk)
        obs, _ = mgr.reset(kwargs=chunk)

        def sink(i, depth, snap, _start=start):
            snapshots.setdefault(f"t{_start + i}", {})[depth] = snap

        terminal, turns_done = run_turns(mgr, gen, obs, size, args.max_turns,
                                         snapshot_sink=sink)
        for i in range(size):
            tid = f"t{start + i}"
            base[tid] = {
                "question": chunk[i].get("question", ""),
                "data_source": str(chunk[i].get("data_source", "unknown")),
                "final_reward": float(terminal[i]),
                "turns": int(turns_done[i]),
            }
        print(f"[diag] base {start + size}/{len(kwargs_all)} "
              f"mean_EM={np.mean([b['final_reward'] for b in base.values()]):.3f}", flush=True)

    # ---------------- phase B: K continuations per prefix ---------------------------
    traj_depths = {t: sorted(d for d in snaps) for t, snaps in snapshots.items()}
    jobs = plan_continuation_jobs(traj_depths, k=args.k, k_check=args.k_check,
                                  k_check_frac=args.k_check_frac, seed=args.seed)
    slots = [(j, rep) for j in jobs for rep in range(j["k"])]
    print(f"[diag] {len(jobs)} prefix jobs -> {len(slots)} continuation rollouts", flush=True)

    results = {(j["traj"], j["depth"]): [] for j in jobs}
    for start in range(0, len(slots), args.pool):
        wave = slots[start:start + args.pool]
        k = len(wave)
        mgr.memory.reset(batch_size=k)
        mgr.tasks = [""] * k
        snaps = [snapshots[j["traj"]][j["depth"]] for j, _ in wave]
        obs = mgr.restore_batch(snaps, indices=list(range(k)))
        depths0 = np.array([j["depth"] for j, _ in wave], dtype=np.int64)
        terminal, _ = run_turns(mgr, gen, obs, k, args.max_turns, depths0=depths0)
        for (j, _), r in zip(wave, terminal):
            results[(j["traj"], j["depth"])].append(float(r))
        print(f"[diag] continuations {min(start + k, len(slots))}/{len(slots)}", flush=True)

    vhat = aggregate_vhat([{"traj": t, "depth": d, "rewards": rs}
                           for (t, d), rs in results.items()])

    payload = {
        "config": {k: v for k, v in vars(args).items()},
        "base": base,
        "vhat": {t: {str(d): v for d, v in per.items()} for t, per in vhat.items()},
    }
    out_path = os.path.join(args.out, "prefix_values.json")
    with open(out_path, "w") as fh:
        json.dump(payload, fh)
    print(f"[diag] wrote {out_path} ({len(base)} trajectories, {len(jobs)} prefixes)", flush=True)


if __name__ == "__main__":
    main()
