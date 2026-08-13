# /// script
# requires-python = ">=3.10"
# dependencies = ['matplotlib', 'seaborn', 'pandas', 'numpy']
# ///
"""Error-pattern analysis of per-turn rubric judgments (grpo_s75, 2000 trajectories).

Computes all report statistics (printed as one JSON blob to stdout) and renders the
report's SVG figures. Re-runnable:
  uv run make_plots.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, '/home/tiger/.claude/skills/generate-report/scripts')
from plots import bar, line, save_svg, PALETTE, CATEGORICAL  # noqa: E402

RESULTS = '/home/tiger/xiaoxuan/arlca-8b-judge/outputs/judge_turns/grpo_s75.rubrics.jsonl'
OUT = Path(__file__).resolve().parent
CRITERIA = ['well_formed', 'relevant', 'novel', 'progression', 'info_gain']
WIKI = {'NQ_rand1000', 'TriviaQA_rand1000', 'PopQA_rand1000', 'HotpotQA_rand1000',
        '2WikiMultihopQA_rand1000', 'Musique_rand1000', 'Bamboogle'}

recs = [json.loads(l) for l in open(RESULTS)]
rows, trajs = [], []
for r in recs:
    rub = r['rubric']
    turns = rub.get('turns') or []
    if not turns:
        continue
    n_s = len(turns)
    # sufficiency position (ordinal among search turns), from judge's sufficiency_turn
    st = rub.get('sufficiency_turn')
    t_index = [t['t'] for t in turns]
    st_pos = t_index.index(st) if st in t_index else None
    for pos, t in enumerate(turns):
        rows.append({'traj': r['traj_uid'], 'pos': pos, 'em': r['em'],
                     'p': t['p'], **{c: t[c] for c in CRITERIA},
                     'post_suff': st_pos is not None and pos > st_pos})
    # longest consecutive novel=0 run
    streak = best = 0
    for t in turns:
        streak = streak + 1 if t['novel'] == 0 else 0
        best = max(best, streak)
    trajs.append({
        'em': r['em'], 'n_search': n_s, 'wiki': r['data_source'] in WIKI,
        'data_source': r['data_source'], 'sufficient': st_pos is not None,
        'suff_pos': st_pos, 'over_search': (n_s - 1 - st_pos) if st_pos is not None else 0,
        'p_final': turns[-1]['p'], 'p0': rub.get('p0', 0.0),
        'max_dup_streak': best,
        **{f'mean_{c}': np.mean([t[c] for t in turns]) for c in CRITERIA},
    })

td = pd.DataFrame(trajs)
tu = pd.DataFrame(rows)


def rank_auc(score, label):
    s, y = np.asarray(score, float), np.asarray(label, int)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    # Mann-Whitney U via average ranks (tie-safe)
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind='mergesort')
    ranks = np.empty_like(allv)
    ranks[order] = np.arange(1, len(allv) + 1)
    i = 0
    sv = allv[order]
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j + 2) / 2
        i = j + 1
    u = ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2
    return float(u / (len(pos) * len(neg)))


stats = {
    'n_trajs': len(td), 'em_mean': td.em.mean(), 'n_turn_rows': len(tu),
    'searches_mean': td.n_search.mean(), 'searches_median': float(td.n_search.median()),
    'searches_mean_wiki': td[td.wiki].n_search.mean(),
    'em_wiki': td[td.wiki].em.mean(), 'n_wiki': int(td.wiki.sum()),
    'sufficient_rate': td.sufficient.mean(),
    'sufficient_rate_wiki': td[td.wiki].sufficient.mean(),
    'suff_pos_mean': td[td.sufficient].suff_pos.mean() + 1,  # 1-based "searches to sufficiency"
    'over_search_mean_suff': td[td.sufficient].over_search.mean(),
    'over_search_rate': (td[td.sufficient].over_search > 0).mean(),
    'post_suff_search_frac': tu.post_suff.mean(),
    'fail_rates': {c: 1 - tu[c].mean() for c in CRITERIA},
    'fail_rates_pos0': {c: 1 - tu[tu.pos == 0][c].mean() for c in CRITERIA},
    'fail_rates_pos3plus': {c: 1 - tu[tu.pos >= 3][c].mean() for c in CRITERIA},
    'dup_streak3plus_rate': (td.max_dup_streak >= 3).mean(),
    'dup_streak_mean': td.max_dup_streak.mean(),
    'p_final_em1': td[td.em == 1].p_final.mean(), 'p_final_em0': td[td.em == 0].p_final.mean(),
    'p0_nonzero_rate': (td.p0 > 0).mean(),
    'auc': {f'mean_{c}': rank_auc(td[f'mean_{c}'], td.em) for c in CRITERIA},
    'auc_p_final': rank_auc(td.p_final, td.em),
    'auc_n_search': rank_auc(-td.n_search, td.em),  # fewer searches -> success?
    'em_by_source': td.groupby('data_source').agg(em=('em', 'mean'), n=('em', 'size'),
                                                  searches=('n_search', 'mean'),
                                                  over=('over_search', 'mean')).to_dict('index'),
}
print(json.dumps(stats, indent=1, default=float))

# ---- fig 1: searches per rollout vs searches-to-sufficiency ----
h1 = td.n_search.clip(upper=16).value_counts(normalize=True).sort_index()
suff = (td[td.sufficient].suff_pos + 1).clip(upper=16).value_counts(normalize=True).sort_index()
d1 = pd.concat([
    pd.DataFrame({'searches': h1.index, 'frac': h1.values, 'what': 'searches issued'}),
    pd.DataFrame({'searches': suff.index, 'frac': suff.values, 'what': 'searches to sufficiency (p≥0.9)'}),
])
f = bar(d1, x='searches', y='frac', hue='what',
        title='Searches issued vs searches actually needed (judge p≥0.9)',
        xlabel='searches (clipped at 16)', ylabel='fraction of trajectories')
save_svg(f, OUT / 'searches_vs_sufficiency.svg')

# ---- fig 2: criterion failure rate by turn position ----
mx = 10
d2 = (tu[tu.pos < mx].melt(id_vars=['pos'], value_vars=CRITERIA)
      .groupby(['pos', 'variable'])['value'].mean().rsub(1).reset_index(name='fail'))
f = line(d2, x='pos', y='fail', hue='variable', markers=True,
         title='Criterion failure rate by search-turn position',
         xlabel='search turn position (0-based)', ylabel='failure rate')
save_svg(f, OUT / 'criterion_fail_by_turn.svg')

# ---- fig 3: readiness p by turn position, split by outcome ----
d3 = (tu[tu.pos < mx].assign(outcome=tu.em.map({1: 'EM = 1', 0: 'EM = 0'}))
      .groupby(['pos', 'outcome'])['p'].mean().reset_index())
f = line(d3, x='pos', y='p', hue='outcome', markers=True,
         title='Judge readiness p_t by turn position and final outcome',
         xlabel='search turn position (0-based)', ylabel='mean p_t')
save_svg(f, OUT / 'readiness_by_turn.svg')

# ---- fig 4: over-search distribution (sufficient trajs) ----
ov = td[td.sufficient].over_search.clip(upper=12).value_counts(normalize=True).sort_index()
f = bar(pd.DataFrame({'extra': ov.index, 'frac': ov.values}), x='extra', y='frac',
        title='Searches issued after sufficiency was reached',
        xlabel='post-sufficiency searches (clipped at 12)', ylabel='fraction of sufficient trajectories')
save_svg(f, OUT / 'over_search_dist.svg')

# ---- fig 5: discrimination AUC per signal ----
d5 = pd.DataFrame({
    'signal': [*CRITERIA, 'p_final', 'fewer searches'],
    'auc': [stats['auc'][f'mean_{c}'] for c in CRITERIA] + [stats['auc_p_final'], stats['auc_n_search']],
}).sort_values('auc')
f = bar(d5, x='auc', y='signal', orientation='h',
        title='Predicting trajectory success (EM) — rank AUC per signal',
        xlabel='AUC (0.5 = chance)')
f.axes[0].axvline(0.5, color=PALETTE.get('gray', '#7d99b1'), lw=1, ls='--')
save_svg(f, OUT / 'discrimination_auc.svg')

print('figures written', file=sys.stderr)
