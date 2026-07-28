#!/usr/bin/env python3
"""Phase 3.3 horizon-stratified analysis — A1–A5 + headline table + per-method verdicts.

Pure post-processing over the per-episode eval JSONL ⋈ H_gold table (see horizon_common.py).
Every figure/table is regenerated from outputs/eval_full/<label>/*.jsonl; no rollouts re-run.

  A1  Score vs required hops         -> fig_a1_score_vs_hops        (pooled + within MuSiQue/2Wiki)
  A2  Horizon-gain vs GRPO           -> fig_a2_horizon_gain + a2_horizon_gain.csv (contrast + bootstrap CI)
  A3  Spend vs need                  -> fig_a3_spend_vs_need        (median turns/searches vs H_gold)
  A4  Turn-distribution violins      -> fig_a4_turn_violins         (success/fail split)
  A5  Secondary splits               -> fig_a5_secondary            (hotpot bridge/comp; popqa head/tail)
  headline EM table                  -> headline_em.csv
  verdicts                           -> verdicts.txt

Run:  python horizon_figures.py            # uses whatever methods are present under outputs/eval_full/
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "analysis", "figures"))
sys.path.insert(0, os.path.dirname(__file__))
from figstyle import PALETTE, apply_publication_style, finalize_figure  # noqa: E402
import horizon_common as H  # noqa: E402

apply_publication_style(font_size=13, axes_linewidth=1.8)
HERE = os.path.dirname(__file__)
FIGS = os.path.join(HERE, "..", "figs")
RES = os.path.join(HERE, "..", "results")
os.makedirs(FIGS, exist_ok=True)
os.makedirs(RES, exist_ok=True)

MCOLOR = {"floor": PALETTE.get("neutral", "#9aa0a6"), "token_ppo": PALETTE.get("violet", "#7b5cff"),
          "turn_ppo": PALETTE.get("red_strong", "#d1495b"), "token_grpo": PALETTE["blue_main"],
          "gigpo": PALETTE.get("teal", "#2a9d8f")}
STRATA = [1, 2, 3, 4]


# ----------------------------- helpers -----------------------------
def present_methods(data):
    return [m for m in H.METHOD_ORDER if m in data]


def em_by_stratum(df, scope=None):
    """Mean EM per H_gold stratum (optionally restricted to a data_source scope)."""
    d = df if scope is None else df[df.data_source == scope]
    g = d.groupby("H_gold")["em"].agg(["mean", "size"])
    return {int(h): (float(g.loc[h, "mean"]), int(g.loc[h, "size"])) for h in g.index}


def _contrast_value(em_m, em_ref, hi, lo):
    """Δ(hi) − Δ(lo) where Δ(h) = EM_m(h) − EM_ref(h); hi/lo are lists of strata pooled by size."""
    def pooled(emd, hs, counts):
        num = sum(emd[h][0] * counts[h] for h in hs if h in emd)
        den = sum(counts[h] for h in hs if h in emd)
        return num / den if den else np.nan
    return None  # replaced by bootstrap-aware version below


def bootstrap_contrast(df_m, df_ref, hi, lo, scope=None, nboot=1000, seed_stride=7):
    """Cluster (item) bootstrap of the top-minus-bottom horizon-gain contrast
    Δ_m(hi) − Δ_m(lo), Δ(h)=EM_m(h)−EM_ref(h). Items are shared → paired resample by index.
    Returns (point, lo_ci, hi_ci). No Math.random: deterministic RNG seeded per resample index."""
    dm = df_m if scope is None else df_m[df_m.data_source == scope]
    dr = df_ref if scope is None else df_ref[df_ref.data_source == scope]
    # align on index (shared items)
    m = dm.set_index("index")[["em", "H_gold"]].rename(columns={"em": "em_m"})
    r = dr.set_index("index")[["em"]].rename(columns={"em": "em_r"})
    j = m.join(r, how="inner")
    if j.empty:
        return (np.nan, np.nan, np.nan)
    hg = j["H_gold"].values
    em_m = j["em_m"].values
    em_r = j["em_r"].values
    idx_by = {h: np.where(hg == h)[0] for h in set(hg)}

    def contrast(sel_em_m, sel_em_r, sel_hg):
        def pooled(hs):
            mask = np.isin(sel_hg, hs)
            if mask.sum() == 0:
                return np.nan
            return sel_em_m[mask].mean() - sel_em_r[mask].mean()
        return pooled(hi) - pooled(lo)

    point = contrast(em_m, em_r, hg)
    boots = []
    n = len(j)
    for b in range(nboot):
        rng = np.random.default_rng(1234 + b * seed_stride)
        samp = rng.integers(0, n, n)          # resample items with replacement
        boots.append(contrast(em_m[samp], em_r[samp], hg[samp]))
    boots = np.array([x for x in boots if np.isfinite(x)])
    lo_ci, hi_ci = np.percentile(boots, [2.5, 97.5])
    return (float(point), float(lo_ci), float(hi_ci))


# ----------------------------- A1 -----------------------------
def a1_score_vs_hops(data):
    methods = present_methods(data)
    panels = [("Pooled (all 7 tasks)", None, STRATA),
              ("Within MuSiQue", "musique", [2, 3, 4]),
              ("Within 2Wiki", "2wikimultihopqa", [2, 4])]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    rows = []
    for ax, (title, scope, xs) in zip(axes, panels):
        for m in methods:
            emd = em_by_stratum(data[m], scope)
            y = [emd.get(h, (np.nan, 0))[0] for h in xs]
            ax.plot(xs, y, marker="o", lw=2.2, ms=7, color=MCOLOR[m], label=H.METHOD_PRETTY[m])
            for h in xs:
                if h in emd:
                    rows.append({"panel": title, "method": m, "H_gold": h,
                                 "EM": emd[h][0], "n": emd[h][1]})
        ax.set_title(title)
        ax.set_xlabel("required hops  $H_{gold}$")
        ax.set_xticks(xs)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("EM")
    axes[0].legend(frameon=False, fontsize=10, loc="upper right")
    fig.suptitle("A1 — EM vs required hops (does credit fidelity buy high-hop robustness?)", y=1.02)
    finalize_figure(fig, os.path.join(FIGS, "fig_a1_score_vs_hops.png"))
    pd.DataFrame(rows).to_csv(os.path.join(RES, "a1_score_vs_hops.csv"), index=False)
    return pd.DataFrame(rows)


# ----------------------------- A2 -----------------------------
def a2_horizon_gain(data):
    methods = [m for m in present_methods(data) if m not in (H.REF_METHOD, "floor")]
    if H.REF_METHOD not in data:
        print("[A2] no GRPO reference present — skipping")
        return None
    ref = data[H.REF_METHOD]
    scopes = [("pooled", None, [3, 4], [1]),
              ("musique", "musique", [3, 4], [2]),
              ("2wiki", "2wikimultihopqa", [4], [2])]
    rows = []
    for m in methods:
        for sname, scope, hi, lo in scopes:
            pt, lo_ci, hi_ci = bootstrap_contrast(data[m], ref, hi, lo, scope=scope)
            sig = np.isfinite(lo_ci) and (lo_ci > 0 or hi_ci < 0)
            rows.append({"method": m, "scope": sname, "contrast_hi": "+".join(map(str, hi)),
                         "contrast_lo": "+".join(map(str, lo)), "delta_contrast": pt,
                         "ci_lo": lo_ci, "ci_hi": hi_ci, "excludes_0": bool(sig)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "a2_horizon_gain.csv"), index=False)
    # figure: contrast ± CI per (method, scope)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    labels, ys, i = [], [], 0
    for m in methods:
        for sname, *_ in scopes:
            r = df[(df.method == m) & (df.scope == sname)].iloc[0]
            ax.errorbar(r["delta_contrast"], i,
                        xerr=[[r["delta_contrast"] - r["ci_lo"]], [r["ci_hi"] - r["delta_contrast"]]],
                        fmt="o", color=MCOLOR[m], ms=7, capsize=4,
                        lw=2 if r["excludes_0"] else 1, alpha=1 if r["excludes_0"] else 0.55)
            labels.append(f"{H.METHOD_PRETTY[m]} · {sname}")
            i += 1
    ax.axvline(0, color="#888", ls="--", lw=1.2)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel(r"horizon-gain contrast  $\Delta_m(\mathrm{high}) - \Delta_m(\mathrm{low})$  vs GRPO")
    ax.set_title("A2 — horizon gain over GRPO (CI excludes 0 ⇒ solid)")
    ax.grid(True, axis="x", alpha=0.3)
    finalize_figure(fig, os.path.join(FIGS, "fig_a2_horizon_gain.png"))
    return df


# ----------------------------- A3 -----------------------------
def a3_spend_vs_need(data):
    methods = present_methods(data)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    rows = []
    for ax, (col, lab) in zip(axes, [("turns", "median turns to <answer>"),
                                     ("n_search_calls", "median <search> calls")]):
        for m in methods:
            d = data[m]
            if d[col].isna().all():
                continue
            g = d.groupby("H_gold")[col].median()
            xs = [h for h in STRATA if h in g.index]
            ax.plot(xs, [g.loc[h] for h in xs], marker="s", lw=2.2, ms=7,
                    color=MCOLOR[m], label=H.METHOD_PRETTY[m])
            for h in xs:
                rows.append({"method": m, "metric": col, "H_gold": h, "median": float(g.loc[h])})
        lims = [1, 4]
        ax.plot(lims, lims, ls=":", color="#aaa", lw=1.5, label="y=x (ref only)")
        ax.set_xlabel("required hops  $H_{gold}$")
        ax.set_ylabel(lab)
        ax.set_xticks(STRATA)
        ax.grid(True, alpha=0.3)
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle("A3 — spend vs need (does the policy escalate search with required hops?)", y=1.02)
    finalize_figure(fig, os.path.join(FIGS, "fig_a3_spend_vs_need.png"))
    pd.DataFrame(rows).to_csv(os.path.join(RES, "a3_spend_vs_need.csv"), index=False)
    return pd.DataFrame(rows)


# ----------------------------- A4 -----------------------------
def a4_turn_violins(data):
    methods = present_methods(data)
    tasks = ["nq", "triviaqa", "popqa", "hotpotqa", "2wikimultihopqa", "musique"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    for ax, ds in zip(axes.flat, tasks):
        parts, poss, cols = [], [], []
        for j, m in enumerate(methods):
            d = data[m]
            v = d[d.data_source == ds]["turns"].values.astype(float)
            if len(v) < 5:
                continue
            parts.append(v)
            poss.append(j)
            cols.append(MCOLOR[m])
        if parts:
            vp = ax.violinplot(parts, positions=poss, showmedians=True, widths=0.8)
            for b, c in zip(vp["bodies"], cols):
                b.set_facecolor(c)
                b.set_alpha(0.55)
        ax.set_title(H.DS_PRETTY[ds])
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([H.METHOD_PRETTY[m] for m in methods], rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("turns")
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("A4 — turn-count distribution per task (single-hop = 'knows when to stop' probe)", y=1.01)
    finalize_figure(fig, os.path.join(FIGS, "fig_a4_turn_violins.png"))


# ----------------------------- A5 -----------------------------
def a5_secondary(data):
    methods = present_methods(data)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    # HotpotQA bridge vs comparison
    rows = []
    ax = axes[0]
    groups = ["bridge", "comparison"]
    x = np.arange(len(groups))
    w = 0.8 / max(1, len(methods))
    for j, m in enumerate(methods):
        d = data[m]
        d = d[d.data_source == "hotpotqa"]
        ys = [d[d.hotpot_type == g]["em"].mean() for g in groups]
        ax.bar(x + j * w, ys, w, color=MCOLOR[m], label=H.METHOD_PRETTY[m])
        for g, y in zip(groups, ys):
            rows.append({"split": "hotpot", "group": g, "method": m, "EM": float(y)})
    ax.set_xticks(x + w * (len(methods) - 1) / 2)
    ax.set_xticklabels(["bridge (2-hop compose)", "comparison"])
    ax.set_ylabel("EM")
    ax.set_title("A5a — HotpotQA composition type")
    ax.grid(True, axis="y", alpha=0.3)
    # PopQA head vs tail popularity
    ax = axes[1]
    groups = ["head", "tail"]
    x = np.arange(len(groups))
    for j, m in enumerate(methods):
        d = data[m]
        d = d[d.data_source == "popqa"]
        ys = [d[d.popqa_pop_bucket == g]["em"].mean() for g in groups]
        ax.bar(x + j * w, ys, w, color=MCOLOR[m], label=H.METHOD_PRETTY[m])
        for g, y in zip(groups, ys):
            rows.append({"split": "popqa", "group": g, "method": m, "EM": float(y)})
    ax.set_xticks(x + w * (len(methods) - 1) / 2)
    ax.set_xticklabels(["head (popular)", "tail (rare → must search)"])
    ax.set_title("A5b — PopQA entity popularity")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
    finalize_figure(fig, os.path.join(FIGS, "fig_a5_secondary.png"))
    pd.DataFrame(rows).to_csv(os.path.join(RES, "a5_secondary.csv"), index=False)


# ----------------------------- headline table -----------------------------
def headline_table(data):
    methods = present_methods(data)
    rows = []
    for m in methods:
        d = data[m]
        rec = {"method": m}
        per_ds = []
        for ds in H.DS_ORDER:
            v = d[d.data_source == ds]["em"]
            e = float(v.mean()) if len(v) else np.nan
            rec[ds] = e
            per_ds.append(e)
        rec["macro_em"] = float(np.nanmean(per_ds))
        rec["micro_em"] = float(d["em"].mean())
        rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "headline_em.csv"), index=False)
    return df


# ----------------------------- verdicts -----------------------------
def verdicts(data, a1, a2, headline):
    lines = ["Phase 3.3 — per-method horizon verdicts (auto-generated from the analysis).", ""]
    methods = [m for m in present_methods(data) if m not in ("floor", H.REF_METHOD)]
    ref_h = headline.set_index("method").loc[H.REF_METHOD, "macro_em"] if H.REF_METHOD in headline["method"].values else np.nan
    for m in methods:
        hm = headline.set_index("method").loc[m, "macro_em"]
        line = f"[{H.METHOD_PRETTY[m]}] macro-EM {hm:.3f} vs GRPO {ref_h:.3f} (Δ {hm-ref_h:+.3f}). "
        if a2 is not None:
            for sc in ["musique", "2wiki", "pooled"]:
                r = a2[(a2.method == m) & (a2.scope == sc)]
                if len(r):
                    r = r.iloc[0]
                    tag = "significant" if r["excludes_0"] else "n.s."
                    line += f"{sc} horizon-gain {r['delta_contrast']:+.3f} [{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}] ({tag}). "
        lines.append(line)
    open(os.path.join(RES, "verdicts.txt"), "w").write("\n".join(lines))
    return "\n".join(lines)


def main():
    labels = dict(H.DEFAULT_METHOD_LABELS)
    # auto-discover any extra eval labels present (e.g. ppo arms added later)
    for m, lab in [("token_ppo", "eval4b_token_ppo_s0"), ("turn_ppo", "eval4b_turn_ppo_s0")]:
        if os.path.isdir(os.path.join(H.EVAL_ROOT, lab)):
            labels[m] = lab
    data = H.load_all(labels)
    if not data:
        print("no eval outputs found yet under", H.EVAL_ROOT)
        return 1
    print("methods loaded:", list(data.keys()))
    hl = headline_table(data)
    print("\n=== headline EM (method × task) ===")
    print(hl.round(4).to_string(index=False))
    a1 = a1_score_vs_hops(data)
    a2 = a2_horizon_gain(data)
    a3_spend_vs_need(data)
    a4_turn_violins(data)
    a5_secondary(data)
    v = verdicts(data, a1, a2, hl)
    print("\n=== verdicts ===\n" + v)
    print("\nfigs ->", FIGS, "\nresults ->", RES)
    return 0


if __name__ == "__main__":
    sys.exit(main())
