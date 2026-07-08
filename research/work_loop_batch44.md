# W-Loop Batch 44: Hostile E3 Toy - Exact-Tool And Supplied-Geometry Absorbers

**Date:** 2026-07-08
**Mode:** CPU-only. No GPU used.
**Role:** Work loop. Make the E3 toy hostile and test whether teacher tomography survives equal-geometry baselines.

## Executive Verdict

The hostile absorber test kills the friendly E3 toy at the supplied-geometry claim ceiling.

Output artifact:

```text
experiments/e3_teacher_tomography_hostile_result.json
```

Overall terminal token:

```text
E3_TOY_ABSORBED_BY_EXACT_DOMAIN_TOOL
```

50-seed mean hidden candidate-ranking accuracy:

| Method | Mean hidden accuracy | E3 margin |
|---|---:|---:|
| B13 exact domain tool hidden constructor | 1.0000 | -14.12 pp |
| E3 source-specific lesson packets | 0.8588 | - |
| B15 nuisance oracle supplied geometry | 0.8588 | 0.00 pp |
| B10+ enhanced counterfactual augmentation | 0.7512 | +10.75 pp |
| Best ordinary pre-hostile absorber | 0.5034 | +35.53 pp |
| B10 old augmentation no tomography | 0.3594 | +49.94 pp |
| Active hard examples average label | 0.4916 | +36.72 pp |
| Shuffled teacher identity | 0.5034 | +35.53 pp |
| Shuffled teacher measurements | 0.4909 | +36.79 pp |

Absorber verdicts from the result JSON:

```text
B13: B44_B13_KILL_E3_IF_EXACT_TOOL_HIDDEN_ACC_GE_E3
B15: B44_B15_KILL_TEACHER_IDENTITY_IF_NUISANCE_ORACLE_WITHIN_5PP_OR_BEATS_E3
B10+: B44_B10P_CONFIRM_TEACHER_SIGNAL_RESIDUAL_IF_E3_BEATS_TRANSFORM_AUG_BY_5PP
```

Interpretation:

- The exact tool fully absorbs the toy: 1.0000 hidden accuracy, 50/50 seeds above E3.
- The nuisance oracle fully absorbs the teacher-identity claim: it matches E3 exactly on 50/50 seeds once given the nuisance geometry and transformation rule.
- Enhanced transform-only augmentation does not fully absorb E3 in this implementation: E3 beats it by 10.75 pp mean and in 46/50 seeds.
- That residual is not enough to preserve the E3 claim, because the stronger equal-geometry absorber B15 matches E3 and B13 beats it.

Claim ceiling: the friendly E3 result was a supplied-geometry result. It remains true that source-specific teacher measurements beat ordinary baselines when the geometry is not supplied, but the hostile toy does not prove teacher tomography adds value beyond the supplied nuisance geometry itself.

## Files Read First

Mandated files read before experiment execution:

- `research/VISION.md`
- `research/STATUS.md`
- `research/work_loop_batch43.md`
- `research/question_loop_batch51.md`
- `research/dual_loop_supervisor_checkin_41.md`
- `code/e3_teacher_tomography.py`
- `experiments/e3_teacher_tomography_result_50seed.json`

Important grounding from those files:

- `VISION.md`: mechanisms are replaceable; the five sacred outcomes are fixed.
- `STATUS.md`: supplied geometry has repeatedly been absorbed in FrameSeed and WGD; equal-information baselines are mandatory.
- `work_loop_batch43.md`: the friendly toy reached E3 0.8588 versus best ordinary 0.5034 but explicitly deferred exact-tool and equal-geometry absorption.
- `question_loop_batch51.md`: B13 exact tool, B15 nuisance oracle, and B10 counterfactual augmentation are required absorbers.
- `dual_loop_supervisor_checkin_41.md`: B44 must make the toy hostile and kill E3 if equal geometry absorbs the signal.
- `e3_teacher_tomography_result_50seed.json`: friendly baseline values reproduced the B43 summary.

## Gate Chain

```text
register -> design-gate -> implement -> dry-run -> smoke -> repair -> full run -> evidence-gate -> commit
```

Gate execution:

```text
python -m py_compile code/e3_teacher_tomography.py
python code/e3_teacher_tomography.py --smoke --output experiments/e3_teacher_tomography_hostile_smoke.json
python code/e3_teacher_tomography.py --seeds 50 --epochs 450 --packet-limit 128 --output experiments/e3_teacher_tomography_hostile_result.json
git add code\e3_teacher_tomography.py
git add code\e3_teacher_tomography.py experiments\e3_teacher_tomography_hostile_result.json research\work_loop_batch44.md
```

Dry-run result: passed.

Smoke result: passed and emitted `E3_TOY_ABSORBED_BY_EXACT_DOMAIN_TOOL`.

Full-run result: passed and emitted `E3_TOY_ABSORBED_BY_EXACT_DOMAIN_TOOL`.

Commit result: blocked by sandbox `.git` write permission.

```text
fatal: Unable to create 'C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-sutra/.git/index.lock': Permission denied
```

## Code Changes

Modified:

```text
code/e3_teacher_tomography.py
```

Added hostile absorber support:

- B13 `B13_exact_domain_tool_hidden_constructor`: admitted exact domain tool with hidden constructor and direct true ranking reconstruction.
- B15 `B15_nuisance_oracle_supplied_geometry`: supplied nuisance bits and transformation geometry, no teacher identity, no teacher margins.
- B10+ `B10_plus_enhanced_counterfactual_augmentation`: same transformation generator, labels propagated from calibration labels by preserve/flip semantics, no teacher margins.

Added per-absorber precommit token table in `PRECOMMIT["absorber_tests"]`.

Added evidence fields:

- `absorber_verdicts`
- `e3_minus_b10_plus_enhanced_augmentation_pp`
- `e3_minus_b13_exact_domain_tool_pp`
- `e3_minus_b15_nuisance_oracle_pp`
- `mean_realized_vs_best_non_exact_absorber`
- `mean_realized_vs_best_all_absorber`

## Precommitted Absorber Tokens

### B13 Exact Domain Tool

```text
confirm: B44_B13_CONFIRM_EXACT_TOOL_GRANTED_AND_SCORED
kill:    B44_B13_KILL_E3_IF_EXACT_TOOL_HIDDEN_ACC_GE_E3
void:    B44_B13_VOID_IF_TOOL_USES_TEACHER_SIGNALS
```

Result: kill.

Evidence:

```text
B13 mean hidden acc: 1.0000
E3 mean hidden acc: 0.8588
E3 - B13: -14.12 pp
E3 below B13 seeds: 50/50
```

### B15 Nuisance Oracle

```text
confirm: B44_B15_CONFIRM_TEACHER_IDENTITY_RESIDUAL_IF_E3_BEATS_NUISANCE_ORACLE_BY_5PP
kill:    B44_B15_KILL_TEACHER_IDENTITY_IF_NUISANCE_ORACLE_WITHIN_5PP_OR_BEATS_E3
void:    B44_B15_VOID_IF_ORACLE_USES_TEACHER_ROLE_MAP_OR_HIDDEN_SET_LOOKUP
```

Result: kill.

Evidence:

```text
B15 mean hidden acc: 0.8588
E3 mean hidden acc: 0.8588
E3 - B15: 0.00 pp
B15 matched E3 packet labels: min 1.0 across seeds
E3 equal B15 seeds: 50/50
E3 above B15 seeds: 0/50
```

### B10+ Enhanced Augmentation

```text
confirm: B44_B10P_CONFIRM_TEACHER_SIGNAL_RESIDUAL_IF_E3_BEATS_TRANSFORM_AUG_BY_5PP
kill:    B44_B10P_KILL_E3_IF_TRANSFORM_AUGMENTATION_WITHOUT_TEACHERS_WITHIN_5PP_OR_BEATS_E3
void:    B44_B10P_VOID_IF_AUGMENTATION_USES_TEACHER_MARGINS_OR_HIDDEN_TEST_LABELS
```

Result: confirm residual against B10+ only.

Evidence:

```text
B10+ mean hidden acc: 0.7512
E3 mean hidden acc: 0.8588
E3 - B10+: +10.75 pp
E3 above B10+ seeds: 46/50
B10+ transformation conflict max: 0
```

This is not a final E3 win because B15 and B13 are stronger equal-geometry absorbers.

## Iteration 1 - Register And Grounding

**Attack target:** The work loop could over-trust the friendly B43 result.

**Precommitted tokens:**

```text
B44_I1_CONFIRM_HOSTILE_BATCH_IF_LIVE_DOCS_REQUIRE_EQUAL_GEOMETRY_ABSORBERS
B44_I1_KILL_FRIENDLY_CLAIM_IF_B43_ALREADY_NAMES_EXACT_TOOL_AS_NEXT_KILL
B44_I1_VOID_IF_MANDATED_FILES_CONFLICT_ON_TASK
```

**Result:** `B44_I1_CONFIRM_HOSTILE_BATCH_IF_LIVE_DOCS_REQUIRE_EQUAL_GEOMETRY_ABSORBERS`.

The mandated files agree: B43 was friendly, B51 and supervisor #41 require exact-tool and supplied-geometry absorbers. The live task is not to extend the headline; it is to try to kill it.

**Attack on conclusion:** Reading the right files is not evidence. Define the absorbers precisely before code.

## Iteration 2 - Design Gate For Absorber Semantics

**Attack target:** The absorber definitions could either be too weak to matter or so strong they become an uninformative strawman.

**Precommitted tokens:**

```text
B44_I2_CONFIRM_DESIGN_IF_B13_B15_B10P_HAVE_DISTINCT_INFORMATION_GRANTS
B44_I2_KILL_DESIGN_IF_B15_IS_ONLY_RENAMED_E3
B44_I2_VOID_IF_BASELINES_USE_TEACHER_MARGINS_WHEN_DECLARED_NO_TEACHER
```

**Design:**

- B13 is the admitted exact tool: direct hidden-constructor reconstruction.
- B15 is supplied nuisance geometry: knows the nuisance bits and transformation rule, but not teacher identity or teacher margins.
- B10+ is transform-only augmentation: starts from calibration labels and propagates labels through preserve/flip transform semantics.

**Result:** `B44_I2_CONFIRM_DESIGN_IF_B13_B15_B10P_HAVE_DISTINCT_INFORMATION_GRANTS`.

The absorbers are distinct. B13 is an explicit oracle, B15 tests whether geometry alone replaces teacher identity, and B10+ tests whether the transformation generator alone plus calibration labels is enough.

**Attack on conclusion:** Design clarity still does not mean implementation honesty. Put the precommit into code before running.

## Iteration 3 - Implement Precommit And Hostile Baselines

**Attack target:** The implementation could leave the exact tool diagnostic outside the terminal gate, repeating the B43 weakness.

**Precommitted tokens in code:**

```text
E3_TOY_ABSORBED_BY_EXACT_DOMAIN_TOOL
E3_TOY_ABSORBED_BY_NUISANCE_ORACLE
E3_TOY_ABSORBED_BY_ENHANCED_AUGMENTATION
E3_TOY_HOSTILE_SIGNAL_SURVIVES_SUPPLIED_GEOMETRY
```

**Work:** Added transformation metadata, labeled transform augmentation, nuisance-oracle labels, exact-tool run entries, hostile margins, and per-absorber verdicts.

**Result:** implemented in `code/e3_teacher_tomography.py` before experiment execution.

**Attack on conclusion:** Code that looks plausible can still fail syntactically or report the wrong schema. Dry-run next.

## Iteration 4 - Dry Run

**Attack target:** The new code might not even compile.

**Precommitted tokens:**

```text
B44_I4_CONFIRM_DRY_RUN_IF_PY_COMPILE_PASSES
B44_I4_KILL_IMPLEMENTATION_IF_CODE_DOES_NOT_COMPILE
B44_I4_VOID_IF_COMPILE_TEST_DOES_NOT_TOUCH_MODIFIED_FILE
```

**Command:**

```text
python -m py_compile code/e3_teacher_tomography.py
```

**Result:** `B44_I4_CONFIRM_DRY_RUN_IF_PY_COMPILE_PASSES`.

`py_compile` passed.

**Attack on conclusion:** Compilation says nothing about experiment behavior. Smoke before full run.

## Iteration 5 - Smoke Test

**Attack target:** The hostile baselines might break runtime behavior or JSON writing.

**Precommitted tokens:**

```text
B44_I5_CONFIRM_SMOKE_IF_HOSTILE_JSON_WRITES_AND_B13_IS_ADMITTED
B44_I5_KILL_RUNNER_IF_SMOKE_ERRORS_OR_MISSING_HOSTILE_BASELINES
B44_I5_VOID_IF_SMOKE_OVERWRITES_B43_ARTIFACTS
```

**Command:**

```text
python code/e3_teacher_tomography.py --smoke --output experiments/e3_teacher_tomography_hostile_smoke.json
```

**Result:** `B44_I5_CONFIRM_SMOKE_IF_HOSTILE_JSON_WRITES_AND_B13_IS_ADMITTED`.

Smoke emitted:

```text
terminal_token: E3_TOY_ABSORBED_BY_EXACT_DOMAIN_TOOL
E3: 0.5938
B13 exact tool: 1.0000
B15 nuisance oracle: 0.5938
B10+: 0.5000
```

**Attack on conclusion:** One seed is not evidence. Run the full 50-seed sweep.

## Iteration 6 - Full 50-Seed Hostile Sweep

**Attack target:** The smoke result may be seed noise, and the friendly 50-seed E3 margin may survive hostile baselines.

**Precommitted tokens:**

```text
B44_I6_CONFIRM_FULL_SWEEP_IF_50_SEEDS_COMPLETE_WITH_HOSTILE_METRICS
B44_I6_KILL_E3_IF_TERMINAL_TOKEN_IS_EXACT_OR_SUPPLIED_GEOMETRY_ABSORPTION
B44_I6_VOID_IF_ANY_HOSTILE_BASELINE_IS_MISSING_FROM_SUMMARY
```

**Command:**

```text
python code/e3_teacher_tomography.py --seeds 50 --epochs 450 --packet-limit 128 --output experiments/e3_teacher_tomography_hostile_result.json
```

**Result:** `B44_I6_KILL_E3_IF_TERMINAL_TOKEN_IS_EXACT_OR_SUPPLIED_GEOMETRY_ABSORPTION`.

Full run emitted:

```text
terminal_token: E3_TOY_ABSORBED_BY_EXACT_DOMAIN_TOOL
```

**Attack on conclusion:** Overall terminal token could hide which absorber mattered. Audit B13, B15, and B10+ separately.

## Iteration 7 - Evidence Gate For B13 Exact Tool

**Attack target:** The exact tool might be reported but not actually scored against E3.

**Precommitted token:**

```text
B44_B13_KILL_E3_IF_EXACT_TOOL_HIDDEN_ACC_GE_E3
```

**Result:** kill.

B13 reached 1.0000 mean hidden accuracy. E3 reached 0.8588. B13 beat E3 in 50/50 seeds. This is the strongest absorber and sets the overall terminal token.

**Attack on conclusion:** Exact tools are intentionally brutal. Maybe E3 still adds value beyond non-exact supplied geometry. Audit B15.

## Iteration 8 - Evidence Gate For B15 Nuisance Oracle

**Attack target:** The exact tool is too obvious; the real question is whether teacher identity matters once the nuisance geometry is supplied.

**Precommitted token:**

```text
B44_B15_KILL_TEACHER_IDENTITY_IF_NUISANCE_ORACLE_WITHIN_5PP_OR_BEATS_E3
```

**Result:** kill.

B15 matched E3 exactly:

```text
E3 mean hidden acc: 0.8588
B15 mean hidden acc: 0.8588
E3 - B15: 0.00 pp
E3 equal B15 seeds: 50/50
```

The B15 labels matched E3 packet labels for every seed. Therefore the teacher-identity signal in the friendly toy is replaceable by the supplied nuisance geometry. This is the FrameSeed/WGD failure pattern.

**Attack on conclusion:** Maybe transformation generation alone, without nuisance oracle labels, also absorbs E3. Audit B10+.

## Iteration 9 - Evidence Gate For B10+ Enhanced Augmentation

**Attack target:** The weaker augmentation baseline might already explain everything, making B15 unnecessary.

**Precommitted token:**

```text
B44_B10P_CONFIRM_TEACHER_SIGNAL_RESIDUAL_IF_E3_BEATS_TRANSFORM_AUG_BY_5PP
```

**Result:** residual against B10+ only.

B10+ reached 0.7512. E3 reached 0.8588. E3 beat B10+ by +10.75 pp mean and in 46/50 seeds.

This means transformations plus calibration-label propagation did not fully teach the hidden structure under the current budget. However, this does not rescue E3 because B15 shows that once the nuisance geometry itself is supplied, teacher identity buys no additional measured value.

**Attack on conclusion:** A residual against one weaker absorber is tempting to spin. Set the claim ceiling and write the narrative honestly.

## Iteration 10 - Claim Ceiling And Terminal Verdict

**Attack target:** The first positive signal from B43 could still leak into the narrative.

**Final token:**

```text
B44_E3_TOY_KILLED_BY_EXACT_TOOL_AND_SUPPLIED_GEOMETRY
```

What survived:

- E3 still beats the old ordinary baselines by +35.53 pp.
- E3 still beats old B10 average-teacher augmentation by +49.94 pp.
- E3 beats B10+ transform-only augmentation by +10.75 pp mean.

What did not survive:

- E3 does not beat an admitted exact domain tool.
- E3 does not beat a nuisance oracle supplied the transformation geometry.
- Teacher identity has no measured residual once the nuisance geometry and transformation rule are supplied.
- Packet-value forecast fails against hostile absorbers: `forecast_ok_non_exact=false`, `forecast_ok_all_absorbers=false`.

Operationally, this kills the current E3 toy as a paradigm-shift claim. A narrower research question remains possible: can E3 discover the geometry in a domain where it is not hand-supplied and where no exact tool is cheaper? That is not proven here.

## Evidence Table

| Quantity | Value |
|---|---:|
| Seeds | 50 |
| Epochs | 450 |
| Packet limit | 128 |
| E3 mean hidden acc | 0.8588 |
| B13 exact tool mean hidden acc | 1.0000 |
| B15 nuisance oracle mean hidden acc | 0.8588 |
| B10+ enhanced augmentation mean hidden acc | 0.7512 |
| E3 - B13 | -14.12 pp |
| E3 - B15 | 0.00 pp |
| E3 - B10+ | +10.75 pp |
| E3 equal B15 seeds | 50/50 |
| E3 below B13 seeds | 50/50 |
| E3 above B10+ seeds | 46/50 |

## NARRATIVE SECTION

### 1. Given only what was measured, what is the gossip-magazine one-sentence story?

When the hidden rule is not handed out, the tiny AI can learn it from expert disagreement, but once the rule or nuisance geometry is handed to a boring baseline, the magic disappears.

### 2. Does it survive "isn't that obvious?" and "so what?"

Only weakly.

Against ordinary baselines, the result is not obvious: E3 still beats active learning, teacher averaging, shuffled sensors, and transform-only augmentation. But against the hostile question, the answer is obvious in the bad way: if a baseline gets the real geometry, it can match or beat E3.

The "so what?" is mostly negative but useful: B44 tells us not to celebrate the friendly toy. E3 must be judged only in domains where geometry is discovered, not supplied.

### 3. Mission test: would a random person say "that changes everything" or just "that's sensible"?

They would say "that's sensible." The measured hostile result does not change everything. It says the previous toy win was not enough because the useful rule was effectively supplied by the experiment structure.

### 4. If the honest narrative is boring or obvious, say so.

The honest narrative is boring: an exact tool solves the toy, and a nuisance-geometry oracle matches E3. The only interesting residue is that source-specific teacher measurements beat transform-only augmentation, but that residue is not enough to keep the current E3 toy alive as a moonshot claim.

## Next Constraint

Do not move to a larger E3 claim unless the next test removes hand-supplied geometry from E3 or gives the same geometry to every serious baseline up front. The next version must measure geometry discovery, not geometry use.
