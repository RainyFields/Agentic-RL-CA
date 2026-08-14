"""Cross-arm analysis of judge items: turn inflation, no-answer failure modes,
answered-conditional accuracy. Writes items_analysis.json next to itself."""
import json
from collections import defaultdict
from pathlib import Path

SRC = Path("/home/tiger/xiaoxuan/arlca-8b/outputs/judge")
ARMS = {"old GRPO": "grpo_s75", "old tPPO": "turnppo_s75",
        "async GRPO": "agrpo_s75", "async tPPO": "atppo_s75"}
WIKI = {"NQ_rand1000", "TriviaQA_rand1000", "PopQA_rand1000", "HotpotQA_rand1000",
        "2WikiMultihopQA_rand1000", "Musique_rand1000", "Bamboogle"}

def pct(xs, q):
    xs = sorted(xs); return xs[min(len(xs)-1, int(q*len(xs)))]

out = {}
items_by_arm = {}
for arm, lab in ARMS.items():
    items = [json.loads(l) for l in open(SRC / f"{lab}.items.jsonl")]
    items_by_arm[arm] = items
    wiki = [it for it in items if it["data_source"] in WIKI]
    turns = [it["n_turns"] for it in wiki]
    noans = [it for it in wiki if it["prediction"] is None]
    ans = [it for it in wiki if it["prediction"] is not None]
    d = {
        "n_wiki": len(wiki),
        "turns_mean": sum(turns)/len(turns), "turns_med": pct(turns, .5),
        "turns_p90": pct(turns, .9), "turns_at_cap_frac": sum(t >= 32 for t in turns)/len(turns),
        "noans_frac": len(noans)/len(wiki),
        "noans_turns_mean": sum(it["n_turns"] for it in noans)/max(1, len(noans)),
        "noans_at_cap_frac": sum(it["n_turns"] >= 32 for it in noans)/max(1, len(noans)),
        "judge_all": sum(it["judge"] for it in wiki)/len(wiki),
        "judge_answered": sum(it["judge"] for it in ans)/max(1, len(ans)),
        "em_answered": sum(it["em"] for it in ans)/max(1, len(ans)),
        "noans_by_bench": {b: sum(1 for it in wiki if it["data_source"] == b and it["prediction"] is None) /
                              max(1, sum(1 for it in wiki if it["data_source"] == b)) for b in sorted(WIKI)},
        "turns_by_bench": {b: sum(it["n_turns"] for it in wiki if it["data_source"] == b) /
                              max(1, sum(1 for it in wiki if it["data_source"] == b)) for b in sorted(WIKI)},
        "turns_hist": {str(lo): sum(1 for t in turns if lo <= t < lo+4) for lo in range(0, 36, 4)},
    }
    out[arm] = d
    print(f"{arm:11s} turns mean {d['turns_mean']:5.2f} med {d['turns_med']:2d} p90 {d['turns_p90']:2d} "
          f"cap {d['turns_at_cap_frac']:.3f} | noans {d['noans_frac']:.3f} "
          f"(mean turns {d['noans_turns_mean']:5.2f}, at-cap {d['noans_at_cap_frac']:.2f}) | "
          f"judge all {d['judge_all']:.3f} answered {d['judge_answered']:.3f}")

# paired: per-question judge deltas between async tPPO and old tPPO on wiki
def key(it): return (it["data_source"], it["question"])
old = {key(it): it for it in items_by_arm["old tPPO"] if it["data_source"] in WIKI}
new = {key(it): it for it in items_by_arm["async tPPO"] if it["data_source"] in WIKI}
common = set(old) & set(new)
flips = {"both_right": 0, "old_only": 0, "new_only": 0, "both_wrong": 0}
old_right_new_noans = 0
for k in common:
    o, n = old[k]["judge"], new[k]["judge"]
    flips[("both_right" if n else "old_only") if o else ("new_only" if n else "both_wrong")] += 1
    if o and new[k]["prediction"] is None: old_right_new_noans += 1
out["paired_tppo"] = {**flips, "n_common": len(common), "old_right_new_noanswer": old_right_new_noans}
print("paired tPPO (old vs async):", out["paired_tppo"])

json.dump(out, open(Path(__file__).parent / "items_analysis.json", "w"), indent=1)
