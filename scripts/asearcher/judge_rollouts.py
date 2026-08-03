#!/usr/bin/env python3
"""Post-hoc LLM-judge (MBE) scoring of an ASearcher eval rollout dump.

The eval worker (`p8b_eval_worker.sh`) runs generation once and dumps every turn to
`rollout_log.jsonl(.zst)`. This script re-scores those dumps without re-running the agent:

  1. group the per-turn dump rows into trajectories (`traj_uid`), keep the last turn;
  2. extract the committed `<answer>` payload exactly the way the training reward does
     (`extract_solution` from the Search-R1 sub-EM scorer);
  3. join the gold answers in from the eval parquet on question text;
  4. ask the frozen judge (gpt-oss-120b per decision A3) CORRECT / INCORRECT;
  5. aggregate per benchmark, and separately for the wiki-answerable benchmarks vs the
     live-web ones our offline wiki-18 retriever cannot serve.

Rule-based sub-EM is recomputed alongside the judge verdict, so every run also yields the
judge-vs-EM calibration table decision A3 asks for.

Usage (judge server must already be up — see p8b_judge_worker.sh):
  python judge_rollouts.py --dump <rollout.jsonl[.zst]> --label grpo_step75 \
      --out-dir outputs/judge --judge-url http://127.0.0.1:8100/v1

Re-running with the same --out-dir resumes: trajectories already present in
`<label>.items.jsonl` are not re-judged.
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from agent_system.environments.env_package.search.third_party.skyrl_gym.envs.search.utils import (  # noqa: E402
    extract_solution,
    normalize_answer,
    subem_check,
)

DEFAULT_EVALSET = "/mnt/hdfs/mlsys/users/xiaoxuan/agentic_rl_ca/data_asearcher_eval/asearcher_eval.parquet"
DEFAULT_PROMPT = REPO / "configs" / "judge_prompts" / "asearcher_mbe_v1.txt"

# Answerable from the offline wiki-18 corpus the agent actually searches.
WIKI_BENCHMARKS = [
    "NQ_rand1000",
    "TriviaQA_rand1000",
    "PopQA_rand1000",
    "HotpotQA_rand1000",
    "2WikiMultihopQA_rand1000",
    "Musique_rand1000",
    "Bamboogle",
]
# Written for a live-web agent; scores here measure the corpus gap, not credit assignment.
LIVEWEB_BENCHMARKS = ["frames", "GAIA", "xbench-deepsearch"]

# agent_system/environments/prompts/search.py wraps the question between "Your question:"
# and one of two fixed continuations. Anchor on those, not on the first blank line —
# multi-paragraph questions (GAIA, frames) contain blank lines of their own.
_QUESTION_RE = re.compile(
    r"Your question:\s*(.*?)\s*\n\s*\n"
    r"(?:Now it's your turn to respond|Prior to this step, you have already taken)",
    re.DOTALL,
)


def open_maybe_zst(path: str) -> io.TextIOBase:
    if path.endswith(".zst"):
        import zstandard

        fh = open(path, "rb")
        reader = zstandard.ZstdDecompressor().stream_reader(fh)
        return io.TextIOWrapper(reader, encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def load_goldset(parquet_path: str) -> dict[str, dict]:
    """normalized question -> {question, data_source, targets}."""
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    gold: dict[str, dict] = {}
    collisions = 0
    for row in df.itertuples(index=False):
        question = row.extra_info["question"]
        targets = [str(t) for t in row.reward_model["ground_truth"]["target"]]
        key = normalize_answer(question)
        if key in gold:
            # Same question in two benchmarks: keep the union of accepted answers so a
            # correct prediction is never marked wrong by an arbitrary tie-break.
            collisions += 1
            merged = list(dict.fromkeys(gold[key]["targets"] + targets))
            gold[key]["targets"] = merged
            continue
        gold[key] = {"question": question, "data_source": row.data_source, "targets": targets}
    if collisions:
        print(f"[judge] {collisions} duplicate questions across benchmarks (answers merged)")
    print(f"[judge] goldset: {len(gold)} unique questions from {len(df)} rows")
    return gold


def collect_trajectories(dump_path: str) -> list[dict]:
    """Fold the per-turn dump into one record per trajectory."""
    last_turn: dict[str, dict] = {}
    first_obs: dict[str, str] = {}
    n_rows = 0
    with open_maybe_zst(dump_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_rows += 1
            tid = rec.get("traj_uid")
            if tid is None:
                continue
            turn = rec.get("turn_index", 0)
            # Turn 0 carries the question-bearing observation; later turns are a fallback
            # for a trajectory whose turn 0 never made it into the dump.
            if turn == 0 or tid not in first_obs:
                first_obs[tid] = rec.get("observation") or ""
            prev = last_turn.get(tid)
            if prev is None or turn >= prev.get("turn_index", -1):
                last_turn[tid] = rec

    trajs = []
    for tid, rec in last_turn.items():
        obs = first_obs.get(tid, "")
        m = _QUESTION_RE.search(obs)
        question = m.group(1).strip() if m else ""
        response = rec.get("raw_model_response") or ""
        answer = extract_solution(response)
        trajs.append(
            {
                "traj_uid": tid,
                "uid": rec.get("uid"),
                "data_source": rec.get("data_source"),
                "n_turns": rec.get("turn_index", 0) + 1,
                "question": question,
                "prediction": answer,  # None when the episode never emitted <answer>
                "env_won": bool(rec.get("env_won")),
                "early_stop_reason": rec.get("early_stop_reason") or "",
            }
        )
    print(f"[judge] {n_rows} turn rows -> {len(trajs)} trajectories")
    return trajs


def attach_gold(trajs: list[dict], gold: dict[str, dict]) -> list[dict]:
    matched, unmatched = [], 0
    for t in trajs:
        g = gold.get(normalize_answer(t["question"])) if t["question"] else None
        if g is None:
            unmatched += 1
            continue
        t["targets"] = g["targets"]
        # The dump's own data_source is authoritative; fall back to the goldset's.
        t["data_source"] = t.get("data_source") or g["data_source"]
        t["subem"] = (
            subem_check(t["prediction"], t["targets"]) if t["prediction"] is not None else 0
        )
        matched.append(t)
    if unmatched:
        pct = 100.0 * unmatched / max(1, len(trajs))
        print(f"[judge] WARNING: {unmatched} trajectories ({pct:.2f}%) had no gold match")
    return matched


# ----------------------------------------------------------------------------- judging
def build_prompt(template: str, t: dict) -> str:
    gold = "; ".join(t["targets"])
    pred = t["prediction"] if t["prediction"] is not None else "(no answer given)"
    return template.format(question=t["question"], gold=gold, prediction=pred)


_VERDICT_RE = re.compile(r"\b(CORRECT|INCORRECT)\b", re.IGNORECASE)


def parse_verdict(content: str, reasoning: str) -> tuple[int | None, str]:
    """Last explicit verdict token wins; INCORRECT must not be read as CORRECT."""
    for text in (content, reasoning):
        if not text:
            continue
        hits = _VERDICT_RE.findall(text)
        if hits:
            return (0 if hits[-1].upper() == "INCORRECT" else 1), hits[-1].upper()
    return None, ""


async def judge_all(
    trajs: list[dict],
    template: str,
    url: str,
    model: str,
    concurrency: int,
    max_retries: int,
    reasoning_effort: str,
    out_fh,
) -> None:
    import httpx

    sem = asyncio.Semaphore(concurrency)
    done = 0
    total = len(trajs)
    lock = asyncio.Lock()
    # `reasoning_effort` is a gpt-oss extension; drop it for the whole run the first time a
    # server rejects it rather than burning every retry on the same 400.
    effort_ok = {"on": bool(reasoning_effort)}
    limits = httpx.Limits(max_connections=concurrency + 8, max_keepalive_connections=concurrency)

    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0), limits=limits) as client:
        async def one(t: dict) -> None:
            nonlocal done
            base = {
                "model": model,
                "messages": [{"role": "user", "content": build_prompt(template, t)}],
                "temperature": 0.0,
                "max_tokens": 2048,
            }
            verdict, raw = None, ""
            async with sem:
                for attempt in range(max_retries):
                    payload = dict(base)
                    if effort_ok["on"]:
                        payload["reasoning_effort"] = reasoning_effort
                    try:
                        r = await client.post(f"{url}/chat/completions", json=payload)
                        if r.status_code == 400 and effort_ok["on"]:
                            effort_ok["on"] = False
                            print("[judge] server rejected reasoning_effort; dropping it", flush=True)
                            continue
                        r.raise_for_status()
                        msg = r.json()["choices"][0]["message"]
                        verdict, raw = parse_verdict(
                            msg.get("content") or "", msg.get("reasoning_content") or ""
                        )
                        if verdict is not None:
                            break
                        raw = f"unparseable: {(msg.get('content') or '')[:160]}"
                    except Exception as exc:  # transient server / socket errors
                        raw = f"error: {type(exc).__name__}: {exc}"[:200]
                    await asyncio.sleep(min(2 ** attempt, 30))
            rec = {
                "traj_uid": t["traj_uid"],
                "data_source": t["data_source"],
                "question": t["question"],
                "targets": t["targets"],
                "prediction": t["prediction"],
                "n_turns": t["n_turns"],
                "env_won": t["env_won"],
                "subem": t["subem"],
                "judge": verdict,  # None = judge failed; excluded from the judge mean
                "judge_raw": raw,
            }
            async with lock:
                out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out_fh.flush()
                done += 1
                if done % 200 == 0 or done == total:
                    print(f"[judge] {done}/{total}", flush=True)

        await asyncio.gather(*(one(t) for t in trajs))


# -------------------------------------------------------------------------- aggregation
def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def aggregate(items: list[dict]) -> dict:
    """Avg@k semantics: average the samples of a question first, then average questions."""
    per_q: dict[tuple[str, str], dict[str, list]] = defaultdict(
        lambda: {"subem": [], "judge": [], "turns": []}
    )
    judge_failed = 0
    judge_attempted = any(it.get("judge_raw") for it in items)
    for it in items:
        key = (it["data_source"], normalize_answer(it["question"]))
        per_q[key]["subem"].append(float(it["subem"]))
        per_q[key]["turns"].append(float(it["n_turns"]))
        if it["judge"] is None:
            # In --no-judge mode every item is unjudged by design, not a failure.
            judge_failed += 1 if judge_attempted else 0
        else:
            per_q[key]["judge"].append(float(it["judge"]))

    by_bench: dict[str, dict] = defaultdict(
        lambda: {"subem": [], "judge": [], "turns": [], "n_q": 0, "n_traj": 0}
    )
    for (ds, _q), v in per_q.items():
        b = by_bench[ds]
        b["n_q"] += 1
        b["n_traj"] += len(v["subem"])
        b["subem"].append(_mean(v["subem"]))
        b["turns"].append(_mean(v["turns"]))
        if v["judge"]:
            b["judge"].append(_mean(v["judge"]))

    benches = {
        ds: {
            "n_questions": b["n_q"],
            "n_trajectories": b["n_traj"],
            "subem": _mean(b["subem"]),
            "judge": _mean(b["judge"]),
            "avg_turns": _mean(b["turns"]),
        }
        for ds, b in sorted(by_bench.items())
    }

    def group(names: list[str]) -> dict:
        present = [n for n in names if n in benches]
        q_sub, q_judge = [], []
        for (ds, _q), v in per_q.items():
            if ds not in present:
                continue
            q_sub.append(_mean(v["subem"]))
            if v["judge"]:
                q_judge.append(_mean(v["judge"]))
        return {
            "benchmarks": present,
            "n_questions": len(q_sub),
            "macro_subem": _mean([benches[n]["subem"] for n in present]),
            "macro_judge": _mean([benches[n]["judge"] for n in present]),
            "micro_subem": _mean(q_sub),
            "micro_judge": _mean(q_judge),
        }

    # judge-vs-EM calibration (decision A3): where do the two scorers disagree?
    both = [it for it in items if it["judge"] is not None]
    tbl = {
        "judge1_em1": sum(1 for it in both if it["judge"] == 1 and it["subem"] == 1),
        "judge1_em0": sum(1 for it in both if it["judge"] == 1 and it["subem"] == 0),
        "judge0_em1": sum(1 for it in both if it["judge"] == 0 and it["subem"] == 1),
        "judge0_em0": sum(1 for it in both if it["judge"] == 0 and it["subem"] == 0),
        "n_scored": len(both),
    }
    tbl["agreement"] = (
        (tbl["judge1_em1"] + tbl["judge0_em0"]) / tbl["n_scored"] if tbl["n_scored"] else float("nan")
    )

    return {
        "n_trajectories": len(items),
        "judge_failures": judge_failed,
        "no_answer_frac": _mean([1.0 if it["prediction"] is None else 0.0 for it in items]),
        "wiki_answerable": group(WIKI_BENCHMARKS),
        "live_web": group(LIVEWEB_BENCHMARKS),
        "by_benchmark": benches,
        "judge_vs_em": tbl,
    }


def render_markdown(label: str, summary: dict) -> str:
    lines = [f"# Judge summary — {label}", ""]
    for gname, gkey in (("Wiki-answerable", "wiki_answerable"), ("Live-web (corpus gap)", "live_web")):
        g = summary[gkey]
        lines += [
            f"**{gname}** ({g['n_questions']} questions, {len(g['benchmarks'])} benchmarks): "
            f"judge macro {g['macro_judge']:.3f} / micro {g['micro_judge']:.3f} · "
            f"sub-EM macro {g['macro_subem']:.3f} / micro {g['micro_subem']:.3f}",
            "",
        ]
    lines += ["| benchmark | n_q | judge | sub-EM | avg turns |", "|---|---|---|---|---|"]
    for ds, b in summary["by_benchmark"].items():
        tag = " *(live-web)*" if ds in LIVEWEB_BENCHMARKS else ""
        lines.append(
            f"| {ds}{tag} | {b['n_questions']} | {b['judge']:.3f} | {b['subem']:.3f} | {b['avg_turns']:.2f} |"
        )
    t = summary["judge_vs_em"]
    lines += [
        "",
        f"Judge-vs-EM agreement: {t['agreement']:.3f} over {t['n_scored']} trajectories "
        f"(judge✓EM✗ {t['judge1_em0']}, judge✗EM✓ {t['judge0_em1']}).",
        f"Trajectories with no `<answer>`: {summary['no_answer_frac']:.3f}. "
        f"Judge failures: {summary['judge_failures']}.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, help="rollout_log.jsonl or .jsonl.zst from the eval worker")
    ap.add_argument("--label", required=True, help="checkpoint tag, used for output filenames")
    ap.add_argument("--out-dir", default="outputs/judge")
    ap.add_argument("--evalset", default=DEFAULT_EVALSET)
    ap.add_argument("--prompt", default=str(DEFAULT_PROMPT))
    ap.add_argument("--judge-url", default=os.environ.get("JUDGE_URL", "http://127.0.0.1:8100/v1"))
    ap.add_argument("--model", default=os.environ.get("JUDGE_MODEL", "gpt-oss-120b"))
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--max-retries", type=int, default=4)
    ap.add_argument("--reasoning-effort", default="low", help="gpt-oss knob; empty string to omit")
    ap.add_argument("--limit", type=int, default=0, help="judge only the first N trajectories (smoke test)")
    ap.add_argument("--no-judge", action="store_true", help="rule-based sub-EM only, no server needed")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    items_path = out_dir / f"{args.label}.items.jsonl"

    template = Path(args.prompt).read_text()
    gold = load_goldset(args.evalset)
    trajs = attach_gold(collect_trajectories(args.dump), gold)
    if args.limit:
        trajs = trajs[: args.limit]

    already: dict[str, dict] = {}
    if items_path.exists():
        with open(items_path) as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("judge") is not None or args.no_judge:
                    already[rec["traj_uid"]] = rec
        print(f"[judge] resuming: {len(already)} trajectories already scored")

    todo = [t for t in trajs if t["traj_uid"] not in already]
    if args.no_judge:
        with open(items_path, "a") as fh:
            for t in todo:
                rec = {
                    "traj_uid": t["traj_uid"],
                    "data_source": t["data_source"],
                    "question": t["question"],
                    "targets": t["targets"],
                    "prediction": t["prediction"],
                    "n_turns": t["n_turns"],
                    "env_won": t["env_won"],
                    "subem": t["subem"],
                    "judge": None,
                    "judge_raw": "",
                }
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                already[t["traj_uid"]] = rec
    elif todo:
        print(f"[judge] judging {len(todo)} trajectories via {args.judge_url} ({args.model})")
        with open(items_path, "a") as fh:
            asyncio.run(
                judge_all(
                    todo,
                    template,
                    args.judge_url.rstrip("/"),
                    args.model,
                    args.concurrency,
                    args.max_retries,
                    args.reasoning_effort,
                    fh,
                )
            )
        with open(items_path) as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                already[rec["traj_uid"]] = rec

    items = list(already.values())
    summary = aggregate(items)
    summary["label"] = args.label
    summary["dump"] = args.dump
    summary["judge_model"] = None if args.no_judge else args.model
    summary["judge_prompt"] = os.path.basename(args.prompt)

    (out_dir / f"{args.label}.summary.json").write_text(json.dumps(summary, indent=2))
    md = render_markdown(args.label, summary)
    (out_dir / f"{args.label}.summary.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
