# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# SP3.1 (2026-06-30): weak admissible-command matching + parse_status classification.
# Backward compatible: default behavior still requires <think>...</think><action>...</action>.
# When weak_prompt=True, accepts plain action text (first non-empty line). Either way, the
# extracted text is matched against the CURRENT admissible commands and classified as one of:
#   exact_admissible_match | normalized_admissible_match | ambiguous_match
#   no_admissible_match     | invalid_format
from typing import List
import re
import random

PARSE_STATUSES = (
    "exact_admissible_match",
    "normalized_admissible_match",
    "ambiguous_match",
    "no_admissible_match",
    "invalid_format",
)


def _normalize(s: str) -> str:
    """lower + strip + drop surrounding quotes/punctuation + collapse whitespace."""
    s = s.strip().lower()
    s = s.strip("\"'`")
    s = re.sub(r"[.!?;:,]+$", "", s)          # trailing punctuation
    s = re.sub(r"\s+", " ", s).strip()
    # ALFWorld: the expert/ReAct phrasing "put X in/on Y" is the admissible command "move X to Y"
    s = re.sub(r"^put (.+?) in/on (.+)$", r"move \1 to \2", s)
    return s


def _extract_tagged(text: str):
    """Strict path: require <think>...</think> and return the <action>...</action> content."""
    low = text.lower()
    if "<think>" not in low or "</think>" not in low:
        return None
    s, e = low.find("<action>"), low.find("</action>")
    if s == -1 or e == -1 or e < s:
        return None
    return text[s + len("<action>"):e].strip()


def _weak_candidates(text: str):
    """Ordered candidate action strings from a weak (action-only) response: <action> content first,
    then each non-empty line (leading 'action:'/'answer:'/'>' stripped), then the whole text."""
    cands = []
    low = text.lower()
    s, e = low.find("<action>"), low.find("</action>")
    if s != -1 and e != -1 and e > s:
        cands.append(text[s + len("<action>"):e].strip())
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^(action\s*:|answer\s*:|>\s*)", "", line, flags=re.IGNORECASE).strip()
        if line:
            cands.append(line)
    whole = text.strip()
    if whole:
        cands.append(whole)
    return list(dict.fromkeys(cands))  # dedup, preserve order


def _best_weak_match(text: str, admissible: List[str]):
    """Try each candidate line in order; return the first exact/normalized admissible match, else the
    best-effort (ambiguous > no_admissible_match) over candidates."""
    cands = _weak_candidates(text)
    if not cands:
        return None, "invalid_format"
    fallback = (cands[0], "no_admissible_match")
    for c in cands:
        matched, status = _match_admissible(c, admissible)
        if status in ("exact_admissible_match", "normalized_admissible_match"):
            return matched, status
        if status == "ambiguous_match" and fallback[1] == "no_admissible_match":
            fallback = (c, "ambiguous_match")
    return fallback


def _match_admissible(cand: str, admissible: List[str]):
    """Return (matched_action, status). Tries exact -> normalized -> unambiguous substring."""
    if not admissible:
        return cand, "no_admissible_match"
    # exact
    for a in admissible:
        if cand == a:
            return a, "exact_admissible_match"
    ncand = _normalize(cand)
    norm = [(_normalize(a), a) for a in admissible]
    # normalized exact
    eqs = [a for na, a in norm if na == ncand]
    if len(eqs) == 1:
        return eqs[0], "normalized_admissible_match"
    if len(eqs) > 1:
        return cand, "ambiguous_match"
    # unambiguous substring (candidate contains an admissible, or vice versa)
    subs = [a for na, a in norm if na and (na in ncand or ncand in na)]
    subs = list(dict.fromkeys(subs))
    if len(subs) == 1:
        return subs[0], "normalized_admissible_match"
    if len(subs) > 1:
        return cand, "ambiguous_match"
    return cand, "no_admissible_match"


def alfworld_projection(actions: List[str], action_pools: List[List[str]],
                        weak_prompt: bool = False,
                        match_mode: str = "weak_text_match",
                        replace_invalid_with_random: bool = False):
    """Process raw model outputs into env actions.

    Returns (actions, valids, parse_statuses). valids[i]=1 iff the action was matched to an
    admissible command (exact or normalized). ambiguous/no-match/invalid -> valids[i]=0 (kept on the
    existing invalid-action path; never silently replaced unless replace_invalid_with_random=True).
    """
    n = len(actions)
    valids = [0] * n
    statuses = ["invalid_format"] * n

    for i in range(n):
        original_str = actions[i]
        admissible = [a for a in (action_pools[i] or []) if a != "help"]

        # Chinese characters -> invalid format
        if re.search(r"[一-鿿]", original_str):
            actions[i] = original_str[-30:]
            valids[i], statuses[i] = 0, "invalid_format"
            continue

        if weak_prompt:
            matched, status = _best_weak_match(original_str, admissible)
        else:
            cand = _extract_tagged(original_str)
            matched, status = (None, "invalid_format") if cand is None else _match_admissible(cand, admissible)
        if status == "invalid_format" or matched is None:
            actions[i] = original_str[-30:]
            valids[i], statuses[i] = 0, "invalid_format"
            continue

        statuses[i] = status
        if status in ("exact_admissible_match", "normalized_admissible_match"):
            actions[i] = matched
            valids[i] = 1
        else:
            valids[i] = 0
            if replace_invalid_with_random and admissible:
                actions[i] = random.choice(admissible)  # opt-in only; status stays as classified
            else:
                actions[i] = matched if matched is not None else original_str[-30:]  # pass through; env rejects

    return actions, valids, statuses
