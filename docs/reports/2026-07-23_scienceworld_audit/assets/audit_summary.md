# ScienceWorld structure audit — raw pass

simplification=easy step_limit=300 vars/task=3 seed=0

| task | max_var | train/dev/test | gold_len (sampled) | final_scores | subgoal firings | errors |
|---|---|---|---|---|---|---|
| boil | 30 | 14/7/9 | [81, 41, 143] | [100, 100, 100] | [8, 6, 10] | 0 |
| change-the-state-of-matter-of | 30 | 14/7/9 | [26, 20, 27] | [100, 100, 100] | [4, 6, 4] | 0 |
| chemistry-mix | 32 | 16/8/8 | [24, 18, 22] | [100, 100, 100] | [5, 6, 7] | 0 |
| chemistry-mix-paint-secondary-color | 36 | 18/9/9 | [11, 15, 15] | [100, 100, 100] | [5, 5, 5] | 0 |
| chemistry-mix-paint-tertiary-color | 36 | 18/9/9 | [30, 24, 20] | [100, 100, 100] | [8, 8, 8] | 0 |
| find-animal | 300 | 150/75/75 | [16, 12, 14] | [100, 100, 100] | [5, 5, 5] | 0 |
| find-living-thing | 300 | 150/75/75 | [10, 12, 10] | [100, 100, 100] | [4, 5, 5] | 0 |
| find-non-living-thing | 300 | 150/75/75 | [7, 7, 5] | [100, 100, 100] | [4, 4, 3] | 0 |
| find-plant | 300 | 150/75/75 | [12, 8, 12] | [100, 100, 100] | [4, 5, 5] | 0 |
| freeze | 30 | 14/7/9 | [28, 64, 27] | [100, 100, 100] | [5, 4, 5] | 0 |
| grow-fruit | 126 | 62/31/33 | [85, 67, 81] | [100, 100, 100] | [20, 14, 18] | 0 |
| grow-plant | 126 | 62/31/33 | [68, 68, 66] | [100, 100, 100] | [10, 10, 10] | 0 |
| identify-life-stages-1 | 14 | 6/3/5 | [29, 27, 30] | [100, 100, 100] | [8, 6, 8] | 0 |
| identify-life-stages-2 | 10 | 4/2/4 | [12, 16, 10] | [100, 100, 100] | [5, 7, 4] | 0 |
| inclined-plane-determine-angle | 168 | 84/42/42 | [60, 99, 172] | [100, 100, 100] | [11, 11, 11] | 0 |
| inclined-plane-friction-named-surfaces | 1386 | 692/346/348 | [53, 71, 64] | [100, 100, 100] | [11, 11, 11] | 0 |
| inclined-plane-friction-unnamed-surfaces | 162 | 80/40/42 | [86, 54, 202] | [100, 100, 100] | [11, 10, 10] | 0 |
| lifespan-longest-lived | 125 | 62/31/32 | [8, 6, 6] | [100, 100, 100] | [3, 3, 3] | 0 |
| lifespan-longest-lived-then-shortest-lived | 125 | 62/31/32 | [9, 7, 7] | [100, 100, 100] | [4, 4, 4] | 0 |
| lifespan-shortest-lived | 125 | 62/31/32 | [8, 8, 6] | [100, 100, 100] | [3, 3, 3] | 0 |
| measure-melting-point-known-substance | 436 | 218/109/109 | [29, 21, 27] | [100, 100, 100] | [10, 11, 11] | 0 |
| measure-melting-point-unknown-substance | 300 | 150/75/75 | [94, 92, 94] | [100, 100, 100] | [7, 6, 11] | 0 |
| melt | 30 | 14/7/9 | [27, 35, 41] | [100, 100, 100] | [5, 6, 6] | 0 |
| mendelian-genetics-known-plant | 120 | 60/30/30 | [125, 123, 130] | [100, 100, 100] | [27, 19, 29] | 0 |
| mendelian-genetics-unknown-plant | 480 | 240/120/120 | [126, 130, 130] | [100, 100, 100] | [25, 29, 28] | 0 |
| power-component | 20 | 10/5/5 | [15, 9, 15] | [100, 100, 100] | [5, 4, 5] | 0 |
| power-component-renewable-vs-nonrenewable-energy | 20 | 10/5/5 | [15, 30, 26] | [100, 100, 100] | [6, 5, 5] | 0 |
| test-conductivity | 900 | 450/225/225 | [22, 45, 27] | [100, 100, 100] | [4, 6, 7] | 0 |
| test-conductivity-of-unknown-substances | 600 | 300/150/150 | [25, 39, 35] | [100, 100, 100] | [6, 7, 6] | 0 |
| use-thermometer | 540 | 270/135/135 | [22, 18, 24] | [100, 100, 100] | [10, 9, 10] | 0 |
