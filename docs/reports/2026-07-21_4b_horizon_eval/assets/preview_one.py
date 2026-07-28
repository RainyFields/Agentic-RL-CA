#!/usr/bin/env python3
"""Single-method horizon preview (e.g. GRPO alone, before the other arms land).
Prints: join-guard check, headline EM (per task + macro/micro), EM vs H_gold (pooled +
within-MuSiQue + within-2Wiki), spend-vs-need (median turns/searches per stratum), per-stratum
cap-fail rate, and the A5 secondary splits. Pure post-processing.
Usage: preview_one.py <label>   e.g.  preview_one.py eval4b_token_grpo_s0
"""
import sys
import numpy as np
import pandas as pd
import horizon_common as H

pd.set_option("display.width", 140)


def main(label):
    d = H.load_eval(label)                 # prints coverage + index→task mismatch guard
    mm = d.attrs.get("ds_mismatch", 0.0)
    print(f"\n[guard] index→task mismatch = {mm:.4f} "
          + ("(OK — global index, H_gold reliable)" if mm <= 0.01
             else "(!! H_gold UNRELIABLE — local index)"))

    print("\n=== headline EM (per task) ===")
    per = d.groupby("data_source")["em"].agg(n="size", EM="mean")
    per = per.reindex(H.DS_ORDER)
    print(per.round(4).to_string())
    macro = float(per["EM"].mean())
    print(f"  macro-EM (7-task mean) = {macro:.4f} | micro-EM = {d['em'].mean():.4f} "
          f"| n = {len(d)}")

    print("\n=== A1  EM vs required hops (H_gold) ===")
    def emhop(scope, xs):
        dd = d if scope is None else d[d.data_source == scope]
        g = dd.groupby("H_gold")["em"].agg(n="size", EM="mean")
        return " | ".join(f"H{h}: {g.loc[h,'EM']:.3f} (n={int(g.loc[h,'n'])})" for h in xs if h in g.index)
    print("  pooled        :", emhop(None, [1, 2, 3, 4]))
    print("  within-MuSiQue:", emhop("musique", [2, 3, 4]))
    print("  within-2Wiki  :", emhop("2wikimultihopqa", [2, 4]))

    print("\n=== A3  spend vs need (median per stratum) + cap-fail rate ===")
    g = d.groupby("H_gold").agg(n=("em", "size"), EM=("em", "mean"),
                                med_turns=("turns", "median"),
                                med_search=("n_search_calls", "median"),
                                mean_turns=("turns", "mean"),
                                mean_search=("n_search_calls", "mean"),
                                cap_fail=("cap_fail", "mean"))
    print(g.round(3).to_string())

    print("\n=== A5  secondary splits ===")
    ht = d[d.data_source == "hotpotqa"].groupby("hotpot_type")["em"].agg(n="size", EM="mean")
    print("  HotpotQA bridge vs comparison:")
    print(ht.round(4).to_string().replace("\n", "\n    "))
    pq = d[d.data_source == "popqa"].groupby("popqa_pop_bucket")["em"].agg(n="size", EM="mean")
    print("  PopQA head vs tail (s_pop median cut):")
    print(pq.round(4).to_string().replace("\n", "\n    "))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "eval4b_token_grpo_s0")
