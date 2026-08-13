# /// script
# requires-python = ">=3.10"
# dependencies = ['matplotlib', 'pandas', 'numpy']
# ///
"""Publication figures (figures4papers house style) for the rubric error-pattern paper.

Writes assets/fig_*.{pdf,png} plus the exact data behind each figure as assets/fig_*.csv.
Re-runnable:  uv run make_paper_figs.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, '/home/tiger/.claude/skills/scientific-figure-making/assets')
from figstyle import (apply_publication_style, PALETTE, make_lines, make_grouped_bar,
                      make_single_bar, finalize_figure)
from matplotlib import pyplot as plt

RESULTS = '/home/tiger/xiaoxuan/arlca-8b-judge/outputs/judge_turns/grpo_s75.rubrics.jsonl'
ASSETS = Path(__file__).resolve().parent / 'assets'
ASSETS.mkdir(exist_ok=True)
CRITERIA = ['well_formed', 'relevant', 'novel', 'progression', 'info_gain']

apply_publication_style(font_size=15)

recs = [json.loads(l) for l in open(RESULTS)]
rows, trajs = [], []
for r in recs:
    turns = r['rubric'].get('turns') or []
    if not turns:
        continue
    st = r['rubric'].get('sufficiency_turn')
    t_index = [t['t'] for t in turns]
    st_pos = t_index.index(st) if st in t_index else None
    for pos, t in enumerate(turns):
        rows.append({'pos': pos, 'em': r['em'], 'p': t['p'], **{c: t[c] for c in CRITERIA}})
    trajs.append({'em': r['em'], 'n_search': len(turns), 'sufficient': st_pos is not None,
                  'suff_pos': st_pos, 'p_final': turns[-1]['p'],
                  **{f'mean_{c}': np.mean([t[c] for t in turns]) for c in CRITERIA}})
tu, td = pd.DataFrame(rows), pd.DataFrame(trajs)


def rank_auc(score, label):
    s, y = np.asarray(score, float), np.asarray(label, int)
    pos, neg = s[y == 1], s[y == 0]
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind='mergesort')
    ranks = np.empty_like(allv)
    ranks[order] = np.arange(1, len(allv) + 1)
    i, sv = 0, allv[order]
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j + 2) / 2
        i = j + 1
    u = ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2
    return float(u / (len(pos) * len(neg)))


# ---- Fig 1: searches issued vs searches to sufficiency ----
cap = 17
issued = td.n_search.clip(upper=cap).value_counts(normalize=True).sort_index()
needed = (td[td.sufficient].suff_pos + 1).clip(upper=cap).value_counts(normalize=True).sort_index()
cats = list(range(1, cap + 1))
s_iss = [float(issued.get(c, 0)) for c in cats]
s_need = [float(needed.get(c, 0)) for c in cats]
pd.DataFrame({'searches': cats, 'issued': s_iss, 'to_sufficiency': s_need}).to_csv(
    ASSETS / 'fig_searches_vs_sufficiency.csv', index=False)
fig, ax = plt.subplots(figsize=(8.2, 4.2))
make_grouped_bar(ax, [str(c) if c < cap else f'{cap}+' for c in cats], [s_need, s_iss],
                 ['to sufficiency ($p_t \\geq 0.9$)', 'issued by policy'],
                 ylabel='Fraction of trajectories',
                 colors=[PALETTE['blue_main'], PALETTE['red_strong']])
ax.set_xlabel('Search turns')
ax.legend()
finalize_figure(fig, str(ASSETS / 'fig_searches_vs_sufficiency.png'))

# ---- Fig 2: criterion failure rate by turn position ----
mx = 10
d2 = (tu[tu.pos < mx].melt(id_vars=['pos'], value_vars=CRITERIA)
      .groupby(['pos', 'variable'])['value'].mean().rsub(1).reset_index(name='fail'))
d2.pivot(index='pos', columns='variable', values='fail').to_csv(ASSETS / 'fig_criterion_fail.csv')
fig, ax = plt.subplots(figsize=(8.2, 4.4))
series = [{'x': g.pos.values, 'y': g.fail.values, 'label': c}
          for c, g in ((c, d2[d2.variable == c]) for c in CRITERIA)]
make_lines(ax, series, ylabel='Failure rate', xlabel='Search turn position (0-based)',
           smooth=False)
ax.set_ylim(-0.03, 1.05)
ax.legend(ncol=2, fontsize=12)
finalize_figure(fig, str(ASSETS / 'fig_criterion_fail.png'))

# ---- Fig 3: readiness p_t by position and outcome ----
d3 = (tu[tu.pos < mx].assign(outcome=tu.em.map({1: 'EM=1', 0: 'EM=0'}))
      .groupby(['pos', 'outcome'])['p'].mean().reset_index())
d3.pivot(index='pos', columns='outcome', values='p').to_csv(ASSETS / 'fig_readiness.csv')
fig, ax = plt.subplots(figsize=(7.2, 4.2))
make_lines(ax, [
    {'x': d3[d3.outcome == 'EM=1'].pos.values, 'y': d3[d3.outcome == 'EM=1'].p.values,
     'label': 'EM $=1$', 'color': PALETTE['blue_main']},
    {'x': d3[d3.outcome == 'EM=0'].pos.values, 'y': d3[d3.outcome == 'EM=0'].p.values,
     'label': 'EM $=0$', 'color': PALETTE['red_strong']},
], ylabel='Mean readiness $p_t$', xlabel='Search turn position (0-based)', smooth=False)
ax.set_ylim(0, 1.02)
ax.legend()
finalize_figure(fig, str(ASSETS / 'fig_readiness.png'))

# ---- Fig 4: discrimination AUC ----
sig = [*(f'mean_{c}' for c in CRITERIA), 'p_final']
auc = {s: rank_auc(td[s], td.em) for s in sig}
auc['fewer_searches'] = rank_auc(-td.n_search, td.em)
order = sorted(auc, key=auc.get)
labels = [s.replace('mean_', '') for s in order]
vals = [auc[s] for s in order]
pd.DataFrame({'signal': order, 'auc': vals}).to_csv(ASSETS / 'fig_auc.csv', index=False)
fig, ax = plt.subplots(figsize=(8.2, 4.2))
make_single_bar(ax, labels, vals, ylabel='Rank AUC vs. EM', annotate=True, fmt='{:.2f}',
                colors=[PALETTE['neutral']] * (len(vals) - 2) + [PALETTE['green_3'], PALETTE['blue_main']])
ax.axhline(0.5, color='black', lw=1.2, ls='--')
ax.set_ylim(0.4, 0.8)
ax.tick_params(axis='x', labelsize=12)
plt.setp(ax.get_xticklabels(), rotation=18, ha='right')
finalize_figure(fig, str(ASSETS / 'fig_auc.png'))

print(json.dumps({'auc': auc, 'n_trajs': len(td)}, indent=1))
