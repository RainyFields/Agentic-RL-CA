# /// script
# requires-python = ">=3.10"
# dependencies = ['matplotlib', 'seaborn', 'pandas']
# ///
"""Plots for the ASearcher s75 evaluation report (GRPO vs turn-PPO)."""
import json
import sys

import pandas as pd

sys.path.insert(0, '/home/tiger/.claude/skills/generate-report/scripts')
from plots import bar, save_svg  # noqa: E402

OUT = '/home/tiger/xiaoxuan/Agentic-RL-CA/.claude/worktrees/sync-partial-rollout/docs/reports/2026-08-05_asearcher_eval_s75'
SRC = '/home/tiger/xiaoxuan/arlca-8b/outputs/judge'
ARMS = {'GRPO': 'grpo_s75', 'turn-PPO': 'turnppo_s75'}
ORDER = ['NQ_rand1000', 'TriviaQA_rand1000', 'PopQA_rand1000',
         'HotpotQA_rand1000', '2WikiMultihopQA_rand1000', 'Musique_rand1000',
         'Bamboogle', 'GAIA', 'frames', 'xbench-deepsearch']
SHORT = {'NQ_rand1000': 'NQ', 'TriviaQA_rand1000': 'TriviaQA',
         'PopQA_rand1000': 'PopQA', 'HotpotQA_rand1000': 'HotpotQA',
         '2WikiMultihopQA_rand1000': '2Wiki', 'Musique_rand1000': 'Musique',
         'Bamboogle': 'Bamboogle', 'GAIA': 'GAIA*', 'frames': 'frames*',
         'xbench-deepsearch': 'xbench*'}

summaries = {arm: json.load(open(f'{SRC}/{lab}.summary.json'))
             for arm, lab in ARMS.items()}

rows = []
for arm, s in summaries.items():
    for b in ORDER:
        d = s['by_benchmark'][b]
        rows.append({'benchmark': SHORT[b], 'arm': arm, 'judge': d['judge'],
                     'em': d['em'], 'subem': d['subem'],
                     'avg_turns': d['avg_turns']})
df = pd.DataFrame(rows)

fig = bar(df, x='benchmark', y='judge', hue='arm',
          title='Judge score by benchmark (step-75 checkpoints; * = live-web, corpus gap)',
          ylabel='judge score')
fig.set_size_inches(10.5, 4.2)
save_svg(fig, f'{OUT}/judge_by_benchmark.svg')

ladder = []
for arm, s in summaries.items():
    w = s['wiki_answerable']
    for metric, val in (('strict EM', w['macro_em']),
                        ('sub-EM', w['macro_subem']),
                        ('judge', w['macro_judge'])):
        ladder.append({'metric': metric, 'arm': arm, 'score': val})
fig = bar(pd.DataFrame(ladder), x='metric', y='score', hue='arm',
          title='Wiki-answerable macro score under the three scorers (7 benchmarks, 6,115 q)',
          ylabel='macro score')
fig.set_size_inches(7.2, 4.0)
save_svg(fig, f'{OUT}/metric_ladder.svg')

fig = bar(df, x='benchmark', y='avg_turns', hue='arm',
          title='Average turns per question (search-efficiency)',
          ylabel='avg turns')
fig.set_size_inches(10.5, 4.0)
save_svg(fig, f'{OUT}/avg_turns.svg')
print('plots written')
