#!/usr/bin/env python3
"""Per-turn rubric judging of ASearcher rollout dumps (judge-reward-arms campaign).

Unlike judge_rollouts.py (trajectory-level CORRECT/INCORRECT), this script grades every
search turn with binary rubric criteria plus a readiness score p_t, one judge call per
trajectory:

  1. fold the per-turn dump rows into full trajectories (`traj_uid`, all turns);
  2. render each search turn as [Turn t] QUERY + RETRIEVED (docs truncated per --doc-chars);
  3. join gold answers from the parquet on question text;
  4. ask gpt-oss-120b for JSON: per-turn {well_formed, relevant, novel, progression,
     info_gain, p} + p0 + sufficiency_turn + notes;
  5. write one record per trajectory to <label>.rubrics.jsonl (resumable).

Structured output: tries OpenAI `response_format: json_schema` first; if the server
rejects it (400), falls back to plain prompting for the whole run and parses the last
JSON object out of the reply. Parse failure retries once; a second failure emits the
NEUTRAL fallback (flags=1, p = linear ramp 0 -> 0.5, fallback=true) so downstream
training never crashes on judge failure — monitor the fallback rate.

Usage (judge server up — see p8b_judge_worker.sh):
  python judge_turns.py --dump <rollout.jsonl[.zst]> --label grpo_s75 \
      --goldset /mnt/hdfs/.../data_asearcher_base/val_2048.parquet \
      --out-dir outputs/judge_turns --judge-url http://127.0.0.1:8100/v1
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from agent_system.environments.env_package.search.third_party.skyrl_gym.envs.search.utils import (  # noqa: E402
    em_check,
    extract_solution,
    normalize_answer,
)

DEFAULT_PROMPT = REPO / "configs" / "judge_prompts" / "asearcher_rubric_v1.txt"

_QUESTION_RE = re.compile(
    r"Your question:\s*(.*?)\s*\n\s*\n"
    r"(?:Now it's your turn to respond|Prior to this step, you have already taken)",
    re.DOTALL,
)
_SEARCH_RE = re.compile(r"<search>(.*?)</search>", re.DOTALL)
_DOC_SPLIT_RE = re.compile(r"(?=Doc \d+:)")

RUBRIC_KEYS = ("well_formed", "relevant", "novel", "progression", "info_gain")

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "p0": {"type": "number"},
        "turns": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "t": {"type": "integer"},
                    **{k: {"type": "integer", "enum": [0, 1]} for k in RUBRIC_KEYS},
                    "p": {"type": "number"},
                },
                "required": ["t", *RUBRIC_KEYS, "p"],
            },
        },
        "sufficiency_turn": {"type": ["integer", "null"]},
        "notes": {"type": "string"},
    },
    "required": ["p0", "turns", "sufficiency_turn", "notes"],
}


def open_maybe_zst(path: str) -> io.TextIOBase:
    if path.endswith(".zst"):
        import zstandard

        fh = open(path, "rb")
        return io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh), encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def load_goldset(parquet_path: str) -> dict[str, dict]:
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    gold: dict[str, dict] = {}
    for row in df.itertuples(index=False):
        question = row.extra_info["question"]
        targets = [str(t) for t in row.reward_model["ground_truth"]["target"]]
        key = normalize_answer(question)
        if key in gold:
            gold[key]["targets"] = list(dict.fromkeys(gold[key]["targets"] + targets))
            continue
        gold[key] = {"question": question, "targets": targets}
    print(f"[rubric] goldset: {len(gold)} unique questions from {len(df)} rows")
    return gold


def collect_trajectories(dump_path: str, max_trajs: int | None) -> list[dict]:
    """Fold per-turn rows into full trajectories, all turns kept and sorted."""
    turns: dict[str, dict[int, dict]] = defaultdict(dict)
    with open_maybe_zst(dump_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            tid = rec.get("traj_uid")
            if tid is None:
                continue
            turns[tid][rec.get("turn_index", 0)] = rec

    trajs = []
    for tid, by_idx in turns.items():
        rows = [by_idx[i] for i in sorted(by_idx)]
        obs0 = rows[0].get("observation") or ""
        m = _QUESTION_RE.search(obs0)
        last = rows[-1]
        trajs.append(
            {
                "traj_uid": tid,
                "uid": last.get("uid"),
                "data_source": last.get("data_source"),
                "question": m.group(1).strip() if m else "",
                "rows": rows,
                "prediction": extract_solution(last.get("raw_model_response") or ""),
                "env_won": bool(last.get("env_won")),
                "env_done": bool(last.get("env_done")),
            }
        )
        if max_trajs and len(trajs) >= max_trajs:
            break
    print(f"[rubric] {len(trajs)} trajectories folded from {dump_path}")
    return trajs


def render_turns(t: dict, doc_chars: int) -> tuple[str, list[int]]:
    """Render search turns for the judge. Returns (text, judged turn_index list)."""
    parts, judged = [], []
    for row in t["rows"]:
        resp = row.get("raw_model_response") or ""
        m = _SEARCH_RE.search(resp)
        if not m:
            continue  # answer turn / invalid action: not a search turn
        query = " ".join(m.group(1).split())
        info = row.get("information") or ""
        payload = info
        jm = re.search(r"\{.*\}", info, re.DOTALL)
        if jm:
            try:
                payload = json.loads(jm.group(0)).get("result", info)
            except (json.JSONDecodeError, AttributeError):
                pass
        docs = [d.strip() for d in _DOC_SPLIT_RE.split(payload) if d.strip()]
        rendered = " ... ".join(f"[{d[:doc_chars]}]" for d in docs) if docs else "(no documents returned)"
        ti = row.get("turn_index", 0)
        judged.append(ti)
        parts.append(f"[Turn {ti}] QUERY: {query}\nRETRIEVED: {rendered}")
    return "\n".join(parts), judged


def neutral_fallback(judged: list[int]) -> dict:
    n = max(1, len(judged))
    return {
        "p0": 0.0,
        "turns": [
            {"t": ti, **{k: 1 for k in RUBRIC_KEYS}, "p": round(0.5 * (i + 1) / n, 3)}
            for i, ti in enumerate(judged)
        ],
        "sufficiency_turn": None,
        "notes": "judge fallback",
        "fallback": True,
    }


def parse_rubric(content: str, reasoning: str, judged: list[int]) -> dict | None:
    """Extract and validate the rubric JSON; realign turn indices when recoverable."""
    for text in (content, reasoning):
        if not text:
            continue
        # last complete JSON object wins (models sometimes narrate before the JSON)
        candidates = re.findall(r"\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}", text, re.DOTALL)
        for blob in reversed(candidates):
            try:
                obj = json.loads(blob)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict) or "turns" not in obj:
                continue
            rows = obj.get("turns") or []
            if len(rows) != len(judged):
                continue
            ok = True
            for i, (r, ti) in enumerate(zip(rows, judged)):
                if not all(r.get(k) in (0, 1) for k in RUBRIC_KEYS):
                    ok = False
                    break
                p = r.get("p")
                if not isinstance(p, (int, float)) or not (0.0 <= p <= 1.0):
                    ok = False
                    break
                r["t"] = ti  # trust position over the echoed index
            if not ok:
                continue
            p0 = obj.get("p0")
            obj["p0"] = float(p0) if isinstance(p0, (int, float)) and 0 <= p0 <= 1 else 0.0
            st = obj.get("sufficiency_turn")
            obj["sufficiency_turn"] = st if isinstance(st, int) and st in judged else (
                None if st is None else next((r["t"] for r in rows if r["p"] >= 0.9), None)
            )
            obj["fallback"] = False
            return obj
    return None


async def judge_all(
    trajs: list[dict],
    template: str,
    url: str,
    model: str,
    concurrency: int,
    doc_chars: int,
    reasoning_effort: str,
    out_fh,
) -> None:
    import httpx

    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    done = 0
    total = len(trajs)
    stats = {"fallback": 0, "schema_mode": True, "effort_on": bool(reasoning_effort)}
    limits = httpx.Limits(max_connections=concurrency + 8, max_keepalive_connections=concurrency)

    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0), limits=limits) as client:
        async def one(t: dict) -> None:
            nonlocal done
            turns_text, judged = render_turns(t, doc_chars)
            rubric = None
            if judged:
                prompt = template.format(
                    question=t["question"],
                    gold_answers="; ".join(t["targets"]),
                    final_answer=t["prediction"] if t["prediction"] is not None else "NONE",
                    turns=turns_text,
                )
                base = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                    "max_tokens": 4096,
                }
                async with sem:
                    for attempt in range(3):
                        payload = dict(base)
                        if stats["effort_on"]:
                            payload["reasoning_effort"] = reasoning_effort
                        if stats["schema_mode"]:
                            payload["response_format"] = {
                                "type": "json_schema",
                                "json_schema": {"name": "rubric", "schema": JSON_SCHEMA},
                            }
                        try:
                            r = await client.post(f"{url}/chat/completions", json=payload)
                            if r.status_code == 400:
                                # drop optional extensions once, for the whole run
                                if stats["schema_mode"]:
                                    stats["schema_mode"] = False
                                    print("[rubric] server rejected json_schema; plain prompting", flush=True)
                                    continue
                                if stats["effort_on"]:
                                    stats["effort_on"] = False
                                    print("[rubric] server rejected reasoning_effort; dropping", flush=True)
                                    continue
                            r.raise_for_status()
                            msg = r.json()["choices"][0]["message"]
                            rubric = parse_rubric(
                                msg.get("content") or "", msg.get("reasoning_content") or "", judged
                            )
                            if rubric is not None:
                                break
                        except Exception:  # transient server / socket errors
                            pass
                        await asyncio.sleep(min(2**attempt, 15))
            if rubric is None:
                rubric = neutral_fallback(judged)
                if judged:
                    stats["fallback"] += 1
            rec = {
                "traj_uid": t["traj_uid"],
                "uid": t["uid"],
                "data_source": t["data_source"],
                "question": t["question"],
                "targets": t["targets"],
                "prediction": t["prediction"],
                "n_turns": len(t["rows"]),
                "judged_turns": judged,
                "env_won": t["env_won"],
                "em": t["em"],
                "rubric": rubric,
            }
            async with lock:
                out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out_fh.flush()
                done += 1
                if done % 100 == 0 or done == total:
                    print(f"[rubric] {done}/{total} (fallbacks {stats['fallback']})", flush=True)

        await asyncio.gather(*(one(t) for t in trajs))
    print(f"[rubric] done: {total} trajs, {stats['fallback']} fallbacks "
          f"({100.0 * stats['fallback'] / max(1, total):.2f}%)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--goldset", required=True, help="parquet with extra_info.question + reward_model.ground_truth.target")
    ap.add_argument("--out-dir", default="outputs/judge_turns")
    ap.add_argument("--judge-url", default="http://127.0.0.1:8100/v1")
    ap.add_argument("--model", default="gpt-oss-120b")
    ap.add_argument("--prompt", default=str(DEFAULT_PROMPT))
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--doc-chars", type=int, default=600)
    ap.add_argument("--max-trajs", type=int, default=0)
    ap.add_argument("--reasoning-effort", default="low")
    args = ap.parse_args()

    template = Path(args.prompt).read_text()
    gold = load_goldset(args.goldset)
    trajs = collect_trajectories(args.dump, args.max_trajs or None)

    matched, unmatched = [], 0
    for t in trajs:
        g = gold.get(normalize_answer(t["question"])) if t["question"] else None
        if g is None:
            unmatched += 1
            continue
        t["targets"] = g["targets"]
        t["em"] = em_check(t["prediction"], t["targets"]) if t["prediction"] is not None else 0
        matched.append(t)
    if unmatched:
        print(f"[rubric] WARNING: {unmatched}/{len(trajs)} trajectories had no gold match")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.label}.rubrics.jsonl"
    seen: set[str] = set()
    if out_path.exists():
        with open(out_path) as fh:
            for line in fh:
                try:
                    seen.add(json.loads(line)["traj_uid"])
                except (json.JSONDecodeError, KeyError):
                    pass
        print(f"[rubric] resume: {len(seen)} trajectories already judged")
    todo = [t for t in matched if t["traj_uid"] not in seen]
    print(f"[rubric] judging {len(todo)} trajectories -> {out_path}")
    if not todo:
        return
    with open(out_path, "a") as out_fh:
        asyncio.run(
            judge_all(
                todo, template, args.judge_url, args.model,
                args.concurrency, args.doc_chars, args.reasoning_effort, out_fh,
            )
        )


if __name__ == "__main__":
    main()
