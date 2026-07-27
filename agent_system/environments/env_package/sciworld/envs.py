# Rung-4 ScienceWorld env package. Ray-actor idiom (one JVM per rollout slot, like
# alfworld), driver-side (task, variation) sampling so groups share an identical
# instance and every draw is logged. Reward per design doc
# docs/reports/2026-07-28_sciworld_orm_prm_design/design.md:
#   P_t = max_{s<=t} max(0, score_s) / 100          (native aggregate score, clipped)
#   reward_mode=prm: r_t = P_t - P_{t-1}
#   reward_mode=orm: r_t = P_T at termination (any cause), else 0
# Return-equivalent by telescoping; focus-death penalized only via foregone reward.

import random
import re
import numpy as np
import gymnasium as gym
import ray

from .task_roster import ROSTER, TURN_CAP, FAMILY, SIMPLIFICATION

_NO_MATCH = "No known action matches that input"
_SEQ_LINE = re.compile(r"^\d+\t(true|false)\t", re.M)


def _seq_progress(goal_progress_str: str):
    """(n_done, n_total) over the Sequential Subgoals section of getGoalProgressStr."""
    seq_section = goal_progress_str.split("Unordered and Optional")[0]
    seq_part = seq_section.split("Sequential Subgoals")[-1]
    flags = _SEQ_LINE.findall(seq_part)
    return sum(1 for f in flags if f == "true"), len(flags)


class SciworldWorker:
    """One ScienceWorldEnv (one JVM) per actor; reused across episodes via load()."""

    def __init__(self, env_step_limit):
        from scienceworld import ScienceWorldEnv
        self.env = ScienceWorldEnv("", envStepLimit=env_step_limit)
        self.task = None
        self.cap = None
        self.steps = 0
        self.P = 0.0
        self.done = False

    def reset(self, task, variation):
        self.env.load(task, variation, SIMPLIFICATION, generateGoldPath=False)
        obs, info = self.env.reset()
        self.task, self.variation = task, variation
        self.cap = TURN_CAP[task]
        self.steps, self.P, self.done = 0, 0.0, False
        task_desc = self.env.get_task_description()
        return obs, {
            "task": task, "variation": variation, "family": FAMILY[task],
            "task_description": task_desc, "turn_cap": self.cap,
            "won": False, "score_raw": 0.0, "progress": 0.0,
            "seq_done": 0, "seq_total": 0,
            "focus_death": False, "cap_hit": False, "env_action_valid": True,
        }

    def step(self, action, reward_mode):
        if self.done:  # defensive: trainer masks finished slots, but keep it safe
            return "", 0.0, True, {
                "task": self.task, "variation": self.variation, "family": FAMILY[self.task],
                "won": self.P >= 1.0, "score_raw": self.P * 100, "progress": self.P,
                "seq_done": 0, "seq_total": 0,
                "focus_death": False, "cap_hit": False, "env_action_valid": False,
            }
        obs, _, env_done, info = self.env.step(action)
        self.steps += 1
        score = float(info.get("score", 0.0))
        focus_death = score < 0
        P_new = max(self.P, max(0.0, score) / 100.0)
        dP = P_new - self.P
        self.P = P_new

        cap_hit = (not env_done) and (not focus_death) and self.steps >= self.cap
        done = bool(env_done) or focus_death or cap_hit
        self.done = done

        if reward_mode == "prm":
            reward = dP
        else:  # orm
            reward = self.P if done else 0.0

        won = self.P >= 1.0
        seq_done, seq_total = _seq_progress(self.env.get_goal_progress())
        if won and seq_total and seq_done != seq_total:
            # design-doc runtime assertion: score==100 must imply full sequential completion
            print(f"[sciworld] ASSERT-VIOLATION task={self.task} var={self.variation}: "
                  f"P=1.0 but seq {seq_done}/{seq_total}")

        return obs, reward, done, {
            "task": self.task, "variation": self.variation, "family": FAMILY[self.task],
            "won": won, "score_raw": score, "progress": self.P,
            "seq_done": seq_done, "seq_total": seq_total,
            "focus_death": focus_death, "cap_hit": cap_hit,
            "env_action_valid": _NO_MATCH not in obs,
        }


def _variation_splits(tasks):
    """Query env-provided train/dev variation splits once, on the driver (one JVM)."""
    from scienceworld import ScienceWorldEnv
    env = ScienceWorldEnv("", envStepLimit=10)
    splits = {}
    for t in tasks:
        env.load(t, 0, SIMPLIFICATION)
        # scienceworld 1.2.3 exposes these only in (deprecated-warning) camelCase
        splits[t] = {
            "train": list(env.getVariationsTrain()),
            "dev": list(env.getVariationsDev()),
        }
    env.close()
    return splits


class SciworldEnvs(gym.Env):
    def __init__(self, seed, env_num, group_n, resources_per_worker,
                 is_train=True, env_kwargs={}):
        super().__init__()
        if not ray.is_initialized():
            ray.init()

        self.env_num, self.group_n = env_num, group_n
        self.num_processes = env_num * group_n
        self.is_train = is_train
        self.reward_mode = env_kwargs.get("reward_mode", "prm")
        assert self.reward_mode in ("prm", "orm"), self.reward_mode
        env_step_limit = env_kwargs.get("env_step_limit", 2500)
        self.rng = random.Random(seed)
        self.episode_idx = 0

        self.splits = _variation_splits(ROSTER)
        # fixed pre-drawn val assignment: 8 dev variations/task (dedicated RNG, seed
        # fixed by design doc; tasks with <8 dev variations contribute repeats)
        vrng = random.Random(20260728)
        self.val_assignment = []
        for t in ROSTER:
            devs = sorted(self.splits[t]["dev"])
            picks = vrng.sample(devs, 8) if len(devs) >= 8 else [devs[i % len(devs)] for i in range(8)]
            self.val_assignment.extend((t, v) for v in picks)

        worker_cls = ray.remote(**resources_per_worker)(SciworldWorker)
        self.workers = [worker_cls.remote(env_step_limit) for _ in range(self.num_processes)]
        self.last_assignment = [None] * self.num_processes

    def _draw_assignments(self):
        """One (task, variation) per group; uniform-per-task then uniform-variation."""
        assignments = []
        for _ in range(self.env_num):
            if self.is_train:
                t = self.rng.choice(ROSTER)
                v = self.rng.choice(self.splits[t]["train"])
            else:
                t, v = self.val_assignment[len(assignments) % len(self.val_assignment)]
            assignments.append((t, v))
        return assignments

    def reset(self):
        if self.is_train:
            groups = self._draw_assignments()
        else:
            groups = [self.val_assignment[g % len(self.val_assignment)] for g in range(self.env_num)]
        self.episode_idx += 1

        futures = []
        for i, worker in enumerate(self.workers):
            t, v = groups[i // self.group_n]
            self.last_assignment[i] = (t, v)
            futures.append(worker.reset.remote(t, v))
        results = ray.get(futures)

        text_obs = [obs for obs, _ in results]
        infos = [info for _, info in results]
        return text_obs, None, infos

    def step(self, actions):
        assert len(actions) == self.num_processes
        futures = [w.step.remote(actions[i], self.reward_mode) for i, w in enumerate(self.workers)]
        results = ray.get(futures)
        text_obs = [r[0] for r in results]
        rewards = [r[1] for r in results]
        dones = [r[2] for r in results]
        infos = [r[3] for r in results]
        return text_obs, None, rewards, dones, infos

    def close(self):
        for w in self.workers:
            ray.kill(w)


def build_sciworld_envs(seed, env_num, group_n, resources_per_worker,
                        is_train=True, env_kwargs={}):
    return SciworldEnvs(seed, env_num, group_n, resources_per_worker, is_train, env_kwargs)
