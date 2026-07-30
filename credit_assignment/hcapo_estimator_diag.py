#!/usr/bin/env python3
"""HCAPO estimator diagnostic — offline, on logged rollouts; no training.

Given a per-turn rollout_log.jsonl (HISR_COLLECT=1 dump: observation / raw_model_response /
parse_status / information / turn_index / traj_uid) and a HF model dir, teacher-force THREE
matched scoring passes over the same response tokens (same 1/|a_t| normalisation, same T_temp;
only the injected suffix differs):

    m_no    : plain prompt                       (on-policy conditioning)
    m_hind  : prompt + "final observation was: " + s_final   (training's hindsight pass)
    m_gold  : prompt + "final observation was: " + gold answer (leakage probe; optional)

and report, per the pre-registered analysis plan (2026-07-30):
  (1) within-trajectory corr(m, |a_t|) + partial corr controlling turn_index
      -> separates the length channel from the position channel; computed for m_hind, m_no
         and the two-pass lift dm = m_hind - m_no (nuisance cancels to first order in dm).
  (2) corr(rho_t, turn_index) and rho_t by turn type (search vs answer).
  (3) clip saturation rate by turn type (raw rho beyond [0.8, 1.2]).
  (4) rho built from dm vs Eq.7's rho built from m_hind: within-trajectory Spearman,
      top-credit-turn agreement, and where each puts credit (answer vs query turns).
      + an excluding-final-turn variant of Eq.7 (leakage mitigation the lift cannot give).

All correlations are WITHIN-trajectory: variables are demeaned per trajectory, then pooled.
"""
import argparse
import json
import re
from collections import defaultdict

import numpy as np
import torch

T_TEMP = 5.0
CLIP_LO, CLIP_HI = 0.8, 1.2
HIND_PREFIX = "\nThe episode's final observation was: "
MAX_PROMPT = 4096
Q_RE = re.compile(r"Your question:\s*(.+)")


# ---------------------------------------------------------------- data
def load_trajectories(path):
    by = defaultdict(list)
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            by[r["traj_uid"]].append(r)
    trajs = []
    for uid, rows in by.items():
        rows = sorted(rows, key=lambda x: x["turn_index"])
        trajs.append(rows)
    return trajs


ANS_RE = re.compile(r"<answer>.*?</answer>", re.DOTALL)
SRCH_RE = re.compile(r"<search>.*?</search>", re.DOTALL)


def turn_type(row):
    # the search env's infos carry no parse_status (dump shows 'unknown'), so classify from
    # the response text itself
    resp = str(row.get("raw_model_response", "") or "")
    if ANS_RE.search(resp):
        return "answer"
    if SRCH_RE.search(resp):
        return "search"
    p = str(row.get("parse_status", ""))
    if p in ("search", "answer"):
        return p
    return "invalid"


def s_final_of(rows):
    """Training's s_final = anchor obs at the LAST turn = the information delivered by the
    action of the second-to-last row. T=1 trajectories have rho == 1 identically -> skipped."""
    if len(rows) < 2:
        return None
    info = rows[-2].get("information", "")
    return str(info) if info else None


def question_of(rows):
    m = Q_RE.search(rows[0].get("observation", "") or "")
    return m.group(1).strip() if m else None


def load_gold(parquet):
    import pyarrow.parquet as pq
    t = pq.read_table(parquet, columns=["extra_info", "reward_model"])
    gold = {}
    for ei, rm in zip(t["extra_info"].to_pylist(), t["reward_model"].to_pylist()):
        q = " ".join(str(ei.get("question", "")).lower().strip().rstrip("?. ").split())
        tgt = rm.get("ground_truth", {}).get("target")
        if q and tgt is not None and len(tgt):
            gold[q] = str(tgt[0])
    return gold


# ---------------------------------------------------------------- scoring
class Scorer:
    def __init__(self, model_dir):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=torch.bfloat16, device_map="cuda")
        self.model.eval()

    def prompt_ids(self, obs, suffix=None):
        ids = self.tok.apply_chat_template(
            [{"role": "user", "content": obs}], add_generation_prompt=True,
            tokenize=True, enable_thinking=False)
        if suffix:
            ids = ids + self.tok.encode(suffix, add_special_tokens=False)
        return ids[-MAX_PROMPT:]                      # left-truncate, as training does

    @torch.no_grad()
    def mean_logprob(self, p_ids, r_ids):
        ids = torch.tensor([p_ids + r_ids], device="cuda")
        logits = self.model(ids).logits[0]
        lp = torch.log_softmax(logits[len(p_ids) - 1:-1].float(), dim=-1)
        tok_lp = lp[torch.arange(len(r_ids)), torch.tensor(r_ids, device="cuda")]
        return float(tok_lp.mean()), len(r_ids)


# ---------------------------------------------------------------- rho + stats helpers
def rho_from(m_vec, clip=True):
    x = np.asarray(m_vec, dtype=np.float64) / T_TEMP
    x = x - x.max()
    e = np.exp(x)
    r = e / e.mean()
    return np.clip(r, CLIP_LO, CLIP_HI) if clip else r


def pooled_within_corr(pairs):
    """pairs: list of (x_vec, y_vec) per trajectory. Demean within trajectory, pool, Pearson."""
    xs, ys = [], []
    for x, y in pairs:
        x = np.asarray(x, float); y = np.asarray(y, float)
        if len(x) < 2:
            continue
        xs.append(x - x.mean()); ys.append(y - y.mean())
    if not xs:
        return float("nan"), 0
    X = np.concatenate(xs); Y = np.concatenate(ys)
    if X.std() < 1e-12 or Y.std() < 1e-12:
        return float("nan"), len(X)
    return float(np.corrcoef(X, Y)[0, 1]), len(X)


def pooled_partial_corr(triples):
    """corr(x, y | z) on within-trajectory-demeaned pooled data (regress z out of both)."""
    xs, ys, zs = [], [], []
    for x, y, z in triples:
        if len(x) < 3:
            continue
        xs.append(np.asarray(x, float) - np.mean(x))
        ys.append(np.asarray(y, float) - np.mean(y))
        zs.append(np.asarray(z, float) - np.mean(z))
    if not xs:
        return float("nan"), 0
    X, Y, Z = map(np.concatenate, (xs, ys, zs))
    if Z.std() > 1e-12:
        X = X - Z * (X @ Z) / (Z @ Z)
        Y = Y - Z * (Y @ Z) / (Z @ Z)
    if X.std() < 1e-12 or Y.std() < 1e-12:
        return float("nan"), len(X)
    return float(np.corrcoef(X, Y)[0, 1]), len(X)


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    if np.std(ra) < 1e-12 or np.std(rb) < 1e-12:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout-log", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gold-parquet", default=None)
    ap.add_argument("--max-trajs", type=int, default=100000)
    args = ap.parse_args()

    trajs = load_trajectories(args.rollout_log)[: args.max_trajs]
    gold = load_gold(args.gold_parquet) if args.gold_parquet else {}
    sc = Scorer(args.model)

    per_traj = []          # each: dict of per-turn arrays
    n_t1 = n_skipped = 0
    gold_hits = gold_misses = 0
    for rows in trajs:
        if len(rows) < 2:
            n_t1 += 1
            continue
        sfin = s_final_of(rows)
        if not sfin:
            n_skipped += 1
            continue
        q = question_of(rows)
        qn = " ".join((q or "").lower().strip().rstrip("?. ").split())
        g = gold.get(qn)
        if gold:
            gold_hits += int(g is not None); gold_misses += int(g is None)
        rec = dict(m_no=[], m_hind=[], m_gold=[], length=[], ti=[], ttype=[])
        ok = True
        for r in rows:
            resp = r.get("raw_model_response") or ""
            r_ids = sc.tok.encode(resp, add_special_tokens=False)[:2048]
            if len(r_ids) == 0:
                ok = False
                break
            obs = r.get("observation") or ""
            mno, _ = sc.mean_logprob(sc.prompt_ids(obs), r_ids)
            mhd, _ = sc.mean_logprob(sc.prompt_ids(obs, HIND_PREFIX + sfin), r_ids)
            rec["m_no"].append(mno)
            rec["m_hind"].append(mhd)
            if g is not None:
                mgd, _ = sc.mean_logprob(sc.prompt_ids(obs, HIND_PREFIX + g), r_ids)
                rec["m_gold"].append(mgd)
            rec["length"].append(len(r_ids))
            rec["ti"].append(int(r["turn_index"]))
            rec["ttype"].append(turn_type(r))
        if not ok:
            n_skipped += 1
            continue
        if g is None:
            rec["m_gold"] = None
        per_traj.append(rec)

    # ---- derived per-trajectory quantities ----
    for rec in per_traj:
        mh = np.array(rec["m_hind"]); mn = np.array(rec["m_no"])
        rec["dm"] = mh - mn
        rec["rho_eq7"] = rho_from(mh)
        rec["rho_eq7_raw"] = rho_from(mh, clip=False)
        rec["rho_dm"] = rho_from(rec["dm"])
        rec["rho_dm_raw"] = rho_from(rec["dm"], clip=False)
        rec["rho_gold"] = rho_from(np.array(rec["m_gold"])) if rec["m_gold"] else None

    out = {"config": dict(rollout_log=args.rollout_log, model=args.model, T_temp=T_TEMP,
                          clip=[CLIP_LO, CLIP_HI], n_traj_scored=len(per_traj),
                          n_traj_T1_skipped=n_t1, n_traj_other_skipped=n_skipped,
                          gold_join=dict(hits=gold_hits, misses=gold_misses)),
           "preregistered_note": "Predictions logged 2026-07-30 before running: (P1) corr(m,len)>0 "
           "within-traj for BOTH m_hind and m_no (shared nuisance), surviving turn-index control; "
           "(P2) dm shrinks the length corr but legacy front-loading predicts corr(dm,len)>0 "
           "residual; (P4) rho(dm) ranks turns differently from Eq.7's rho(m_hind); direction of "
           "credit movement (answer vs query turns) reported, not assumed."}

    # (1) length vs position channels
    for name in ("m_hind", "m_no", "dm"):
        c, n = pooled_within_corr([(rec[name] if name == "dm" else rec[name], rec["length"])
                                   for rec in per_traj])
        pc, pn = pooled_partial_corr([(rec[name] if name == "dm" else rec[name],
                                       rec["length"], rec["ti"]) for rec in per_traj])
        out[f"corr_{name}_len"] = dict(r=c, n=n, partial_r_ctrl_turnindex=pc, n_partial=pn)

    # (2) rho vs turn index / turn type
    for rname in ("rho_eq7", "rho_dm"):
        c, n = pooled_within_corr([(rec[rname], rec["ti"]) for rec in per_traj])
        vals = {"search": [], "answer": [], "invalid": []}
        for rec in per_traj:
            for v, t in zip(rec[rname], rec["ttype"]):
                vals[t].append(float(v))
        out[f"{rname}_stats"] = dict(
            corr_turnindex=dict(r=c, n=n),
            by_type={t: dict(mean=float(np.mean(v)), std=float(np.std(v)), n=len(v))
                     for t, v in vals.items() if v})

    # (3) clip saturation by turn type
    for rname in ("rho_eq7_raw", "rho_dm_raw"):
        sat = {t: [0, 0, 0] for t in ("search", "answer", "invalid")}   # [n, at_lo, at_hi]
        for rec in per_traj:
            for v, t in zip(rec[rname], rec["ttype"]):
                sat[t][0] += 1
                sat[t][1] += int(v <= CLIP_LO)
                sat[t][2] += int(v >= CLIP_HI)
        out[f"saturation_{rname}"] = {
            t: dict(n=n, frac_at_lo=lo / n, frac_at_hi=hi / n)
            for t, (n, lo, hi) in sat.items() if n}

    # (4) eq7 vs lift: rank agreement + credit location (+ gold, + excluding-final-turn)
    sp, agree, agree_n = [], 0, 0
    sp_gold = []
    sp_exfinal = []
    for rec in per_traj:
        if len(rec["rho_eq7"]) >= 3:
            s = spearman(rec["rho_eq7"], rec["rho_dm"])
            if np.isfinite(s):
                sp.append(s)
            if rec["rho_gold"] is not None:
                sg = spearman(rec["rho_dm"], rec["rho_gold"])
                if np.isfinite(sg):
                    sp_gold.append(sg)
            # leakage-relevant argmax stat: how often each estimator crowns the FINAL turn
            # (the answer turn s_final trivially "explains") as the top-credit turn.
            sp_exfinal.append(int(int(np.argmax(rec["rho_eq7"])) == len(rec["rho_eq7"]) - 1))
        if len(rec["rho_eq7"]) >= 2:
            agree_n += 1
            agree += int(int(np.argmax(rec["rho_eq7"])) == int(np.argmax(rec["rho_dm"])))
    out["rank_agreement"] = dict(
        spearman_eq7_vs_dm=dict(mean=float(np.mean(sp)) if sp else None, n_traj=len(sp)),
        top_turn_agreement=dict(frac=agree / agree_n if agree_n else None, n=agree_n),
        spearman_dm_vs_gold=dict(mean=float(np.mean(sp_gold)) if sp_gold else None,
                                 n_traj=len(sp_gold)),
        frac_eq7_top_turn_is_final=dict(mean=float(np.mean(sp_exfinal)) if sp_exfinal else None,
                                        n_traj=len(sp_exfinal)))

    def credit_loc(rname):
        a, s = [], []
        for rec in per_traj:
            r = rec[rname]
            if r is None:
                continue
            for v, t in zip(r, rec["ttype"]):
                (a if t == "answer" else s if t == "search" else []).append(float(v))
        return dict(mean_answer=float(np.mean(a)) if a else None,
                    mean_search=float(np.mean(s)) if s else None,
                    answer_minus_search=(float(np.mean(a) - np.mean(s)) if a and s else None))
    out["credit_location"] = {r: credit_loc(r) for r in ("rho_eq7", "rho_dm", "rho_gold")}

    # per-turn arrays: makes every future re-analysis GPU-free
    out["per_traj"] = [
        dict(m_no=list(map(float, r["m_no"])), m_hind=list(map(float, r["m_hind"])),
             m_gold=(list(map(float, r["m_gold"])) if r["m_gold"] else None),
             length=r["length"], ti=r["ti"], ttype=r["ttype"])
        for r in per_traj]
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
