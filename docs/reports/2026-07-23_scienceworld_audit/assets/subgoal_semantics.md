# Oracle subgoal-reward semantics (gold replays)

- traces analyzed: 90 (30 tasks)
- score scale: cumulative 0-100 per episode; per-step `reward` = score delta
- all gold replays reach 100: True
- subgoal firing density on gold paths: min 0.05 / median 0.27 / max 0.62 firings per step
- firing magnitudes: min 1 / max 72 (distinct values: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 25, 26, 27, 28, 29, 33, 34, 35, 36, 38, 42, 47, 50, 54, 55, 58, 63, 66, 67, 68, 72])
- negative deltas observed on gold paths: 0 (none)

| family | tasks | sampled gold-len range | proposed cap (2x max) |
|---|---|---|---|
| chemistry-mix | 3 | 11-30 | 60 |
| electricity | 4 | 9-45 | 90 |
| find-thing | 4 | 5-16 | 32 |
| genetics | 2 | 123-130 | 260 |
| inclined-plane | 3 | 53-202 | 404 |
| life-stages | 2 | 10-30 | 60 |
| lifespan | 3 | 6-9 | 18 |
| matter-state | 4 | 20-143 | 286 |
| plant-growth | 2 | 66-85 | 170 |
| thermal-measure | 3 | 18-94 | 188 |
