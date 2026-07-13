# Copyright 2026 Agentic-RL-CA project. Apache-2.0.
"""B1 privileged answer-exposure step reward (plan Phase 2.1).

B1 is NOT a ground-truth progress measure — useful retrievals may lack the answer string and
answer strings may appear in useless retrievals. It is a *privileged, verifiable* shaping
signal: normalize_answer(gold) substring-matches this turn's retrieved <information> block.
First-hit-only (anti-farming): at most one positive step reward per trajectory.

Pure functions, no env/config dependencies — unit-testable on CPU.
"""
import importlib.util
import os

# Load the Search-R1 EM utils by file path: the package route
# (agent_system.environments.env_package.search...) executes env __init__s that import
# gym/vllm — unavailable on CPU login nodes where the unit tests run. utils.py itself only
# needs re/string.
_UTILS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "agent_system", "environments", "env_package", "search", "third_party",
    "skyrl_gym", "envs", "search", "utils.py",
)
_spec = importlib.util.spec_from_file_location("_searchr1_em_utils", _UTILS_PATH)
_em_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_em_utils)
subem_check = _em_utils.subem_check


def extract_targets(ground_truth):
    """ground_truth arrives as {'target': array([...])} (verl-agent env_kwargs) or a bare
    str/list; return a list of gold answer strings."""
    if isinstance(ground_truth, dict):
        targets = ground_truth.get("target", [])
    else:
        targets = ground_truth
    if isinstance(targets, str):
        return [targets]
    return list(targets)


def b1_hit(observation_text, ground_truth) -> bool:
    """True iff any normalized gold answer is a substring of the normalized observation
    (the retrieved <information> block for this turn)."""
    if not observation_text:
        return False
    targets = extract_targets(ground_truth)
    if not targets:
        return False
    return bool(subem_check(observation_text, targets))


def compute_b1_step_reward(observation_text, ground_truth, already_given: bool, w_step: float):
    """Returns (bonus, hit, given). `hit` is the raw exposure signal (logged as retrieval-hit
    rate regardless of gating); `bonus` is w_step on the FIRST hit only; `given` is the updated
    first-hit latch."""
    hit = b1_hit(observation_text, ground_truth)
    if hit and not already_given:
        return float(w_step), True, True
    return 0.0, hit, already_given
