# Rung-4 ScienceWorld task roster (frozen by design doc
# docs/reports/2026-07-28_sciworld_orm_prm_design/design.md, decision A13).
#
# Caps are per-task: 2x the max sampled gold-path length from the M5 audit
# (docs/reports/2026-07-23_scienceworld_audit/assets/gold_traces.jsonl, 3 train
# variations/task, simplification "easy"). Roster rule: included iff cap <= ~190.
# Do not edit caps without re-measuring gold lengths under the same simplification.

SIMPLIFICATION = "easy"  # resolves to: noElectricalAction, openDoors, selfWateringFlowerPots, teleportAction

# task -> (family, per-task turn cap, included)
TASK_TABLE = {
    "boil":                                             ("matter-state",    286, False),
    "change-the-state-of-matter-of":                    ("matter-state",     54, True),
    "chemistry-mix":                                    ("chemistry-mix",    48, True),
    "chemistry-mix-paint-secondary-color":              ("chemistry-mix",    30, True),
    "chemistry-mix-paint-tertiary-color":               ("chemistry-mix",    60, True),
    "find-animal":                                      ("find-thing",       32, True),
    "find-living-thing":                                ("find-thing",       24, True),
    "find-non-living-thing":                            ("find-thing",       14, True),
    "find-plant":                                       ("find-thing",       24, True),
    "freeze":                                           ("matter-state",    128, True),
    "grow-fruit":                                       ("plant-growth",    170, True),
    "grow-plant":                                       ("plant-growth",    136, True),
    "identify-life-stages-1":                           ("life-stages",      60, True),
    "identify-life-stages-2":                           ("life-stages",      32, True),
    "inclined-plane-determine-angle":                   ("inclined-plane",  344, False),
    "inclined-plane-friction-named-surfaces":           ("inclined-plane",  142, True),
    "inclined-plane-friction-unnamed-surfaces":         ("inclined-plane",  404, False),
    "lifespan-longest-lived":                           ("lifespan",         16, True),
    "lifespan-longest-lived-then-shortest-lived":       ("lifespan",         18, True),
    "lifespan-shortest-lived":                          ("lifespan",         16, True),
    "measure-melting-point-known-substance":            ("thermal-measure",  58, True),
    "measure-melting-point-unknown-substance":          ("thermal-measure", 188, True),
    "melt":                                             ("matter-state",     82, True),
    "mendelian-genetics-known-plant":                   ("genetics",        260, False),
    "mendelian-genetics-unknown-plant":                 ("genetics",        260, False),
    "power-component":                                  ("electricity",      30, True),
    "power-component-renewable-vs-nonrenewable-energy": ("electricity",      60, True),
    "test-conductivity":                                ("electricity",      90, True),
    "test-conductivity-of-unknown-substances":          ("electricity",      78, True),
    "use-thermometer":                                  ("thermal-measure",  48, True),
}

ROSTER = sorted(t for t, (_, _, inc) in TASK_TABLE.items() if inc)  # 25 tasks
FAMILY = {t: fam for t, (fam, _, _) in TASK_TABLE.items()}
TURN_CAP = {t: cap for t, (_, cap, _) in TASK_TABLE.items()}
MAX_ROSTER_CAP = max(TURN_CAP[t] for t in ROSTER)  # 188

assert len(ROSTER) == 25, f"roster drifted: {len(ROSTER)}"

# A14 horizon strata (straggler mitigation): each training step draws its whole batch
# from ONE stratum, so the step's turn-round count is bounded by that stratum's max cap
# instead of the global 188. The rotation sequence is proportional to stratum sizes
# (10/10/5), which preserves the uniform-per-task marginal across training.
STRATA = {
    "short": [t for t in ROSTER if TURN_CAP[t] <= 32],   # 10 tasks, caps 14-32
    "mid":   [t for t in ROSTER if 32 < TURN_CAP[t] <= 90],   # 10 tasks, caps 48-90
    "long":  [t for t in ROSTER if TURN_CAP[t] > 90],    # 5 tasks, caps 128-188
}
STRATA_SEQUENCE = ["short", "short", "mid", "mid", "long"]
assert [len(STRATA[s]) for s in ("short", "mid", "long")] == [10, 10, 5]
