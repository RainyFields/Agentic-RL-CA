# ASearcher dataset inspection — machine summary

## Train (ASearcher-train-data)

### ASearcher-Base-35k.jsonl — 35583 rows
- fields: ['question', 'answer', 'source', 'aug_answer', 'qid']
- trajectory fields: NONE
- question words: {'min': 4, 'median': 23, 'mean': 28.3, 'p95': 60, 'max': 485}
- sources: {'compose_chain_qa': 8537, 'opensource-fm8k': 7991, 'opensource': 7927, 'compose_group_qa': 6375, 'compose_chain_qa_invalid': 4753}

### ASearcher-LRM-35k.jsonl — 35054 rows
- fields: ['question', 'answer', 'source', 'id', 'idx']
- trajectory fields: NONE
- question words: {'min': 1, 'median': 43, 'mean': 45.8, 'p95': 88, 'max': 639}
- sources: {'(no source field)': 24026, 'chain_qa_fuzzy': 3527, 'math': 2930, 'train_v1:webwalker_hard': 2572, 'train_v1:compose_group_qa': 1999}

## Test (ASearcher-test-data)

| subset | rows | ref-steps | traj |
|---|---|---|---|
| GAIA | 103 | True | NONE |
| frames | 824 | False | NONE |
| xbench-deepsearch | 100 | True | ['reference_steps'] |
| Bamboogle | 125 | False | NONE |
| 2WikiMultihopQA_rand1000 | 1000 | False | NONE |
| HotpotQA_rand1000 | 1000 | False | NONE |
| Musique_rand1000 | 1000 | False | NONE |
| NQ_rand1000 | 1000 | False | NONE |
| PopQA_rand1000 | 1000 | False | NONE |
| TriviaQA_rand1000 | 1000 | False | NONE |
