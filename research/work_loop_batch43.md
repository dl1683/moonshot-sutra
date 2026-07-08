# W-Loop Batch 43: E3 Functional Teacher Tomography Toy

**Date:** 2026-07-08
**Mode:** CPU-only. No GPU used.
**Role:** Work loop. Fix the collection gate, implement the first E3 toy, run it, and attack the result.

## Executive Verdict

Priority 1 is closed at the collection level: `code/compare_ablations.py` has
been restored as a compact comparator module, and `code/test_eklavya_e2.py`
collects again.

Priority 2 produced runnable E3 code, not a protocol document:

```text
code/e3_teacher_tomography.py
experiments/e3_teacher_tomography_result.json
experiments/e3_teacher_tomography_result_50seed.json
```

The 50-seed stress result emits:

```text
E3_TOY_SIGNAL_SOURCE_SPECIFIC_COUNTERFACTUAL_LESSON
```

Mean hidden candidate-ranking accuracy:

| Method | Mean hidden accuracy |
|---|---:|
| E3 source-specific lesson packets | 0.8588 |
| Best ordinary absorber | 0.5034 |
| CE-only | 0.4938 |
| Best single teacher | 0.5022 |
| Naive teacher average | 0.5019 |
| Weighted vote | 0.4922 |
| Active hard-example average-label baseline | 0.4916 |
| Shuffled teacher measurements | 0.4909 |
| Shuffled teacher identity | 0.5034 |
| Counterfactual augmentation without tomography | 0.3594 |

Main margins from `experiments/e3_teacher_tomography_result_50seed.json`:

```text
E3 - best ordinary absorber: +35.53 pp
E3 - CE-only: +36.50 pp
E3 - best single teacher: +35.66 pp
E3 - average/weighted vote: +35.69 pp
E3 - active hard-example mining: +36.72 pp
E3 - shuffled sensors: +35.53 pp
E3 - augmentation-only: +49.94 pp
```

Packet-value forecast also passed:

```text
mean_prior: 0.44125
mean_realized_vs_best_baseline: 0.2815625
forecast_ok: true
```

Claim ceiling: this proves only that the central E3 object exists in a friendly
controlled toy. It does not prove useful AI, natural-language transfer, or a
paradigm shift. It earns the next kill test.

## Operational Note: Commits Blocked

The prompt required commits after logical units. I attempted the first commit
after restoring the comparator:

```text
git add code/compare_ablations.py; git commit -m "Restore E2 ablation comparator"
```

The sandbox refused `.git` writes:

```text
fatal: Unable to create '.git/index.lock': Permission denied
```

So no commits could be made in this environment. I kept staging explicit and did
not rely on broad git operations.

## Files Read

Required files:

- `research/VISION.md`
- `research/STATUS.md`
- `research/work_loop_batch42.md`
- `research/question_loop_batch50.md`
- `research/dual_loop_supervisor_checkin_40.md`
- `research/EKLAVYA_DOCTRINE.md`
- `code/eklavya_e2_training.py`
- `code/eklavya_e2_router.py`
- `code/eklavya_e2_losses.py`
- `code/s0_architecture.py`
- `code/s0_configs.py`
- `code/test_eklavya_e2.py`

Additional live repo constraint read:

- `research/question_loop_batch51.md`

Reason: it already existed in the checkout and narrowed E3 to counterfactual
ranking geometry, source identity, shuffled-sensor controls, and active-learning
absorption.

## CPU Gates Run

```text
python -m py_compile code/compare_ablations.py
python -m py_compile code/e3_teacher_tomography.py
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'; python -m pytest code/test_eklavya_e2.py --collect-only -q
python code/e3_teacher_tomography.py --smoke --output experiments/e3_teacher_tomography_smoke.json
python code/e3_teacher_tomography.py --seeds 20 --epochs 450 --packet-limit 128 --output experiments/e3_teacher_tomography_result.json
python code/e3_teacher_tomography.py --seeds 50 --epochs 450 --packet-limit 128 --output experiments/e3_teacher_tomography_result_50seed.json
```

Collection result:

```text
473 tests collected
```

Focused comparator tests were partially blocked by the known sandbox temp issue:
non-`tmp_path` comparator tests passed, while `tmp_path` setup failed because
Python could not access pytest temp roots. This is an environment verdict, not a
comparator logic failure.

## Experiment Design

The toy domain is `controlled_candidate_ranking_world_v0`.

Hidden state:

```text
z0, z1: latent ranking factors
n0, n1: nuisance bits that transform the observed surface
irrelevant slots: distractors
true ranking: z0 xor z1
observed features: x0 = z0 xor n0, x1 = z1 xor n1, n0, n1, distractors
hidden transform: n0=1, n1=0
```

Teachers:

| Teacher | Sensor role | What it measures | What it refuses |
|---|---|---|---|
| `surface_lexical` | surface-biased teacher | `x0 xor x1` shortcut | nuisance correction |
| `semantic_z0` | semantic sensor | latent factor `z0` | final ranking alone |
| `verifier_z1` | verifier sensor | latent factor `z1` and flip localization | surface style prior |

E3 lesson packet:

```text
semantic_z0 and verifier_z1 are complementary sensors.
Ranking flips when exactly one sensor flips.
Irrelevant-slot and nuisance-preserving transforms should preserve ranking.
Single-latent counterfactual flips should flip ranking.
```

The student is teacher-free at inference: a tiny linear readout over generic
monomial features up to degree 4. The feature map is not target-specific; it is
a small generic interaction basis so the experiment tests lesson content rather
than raw MLP optimization luck.

Baselines implemented:

- B0 CE-only same student
- B4 best single teacher
- B5 naive teacher average
- B6 calibrated weighted vote
- B7 shuffled teacher measurements
- B8 shuffled teacher identity
- B9 active/hard-example mining with average teacher labels
- B10 counterfactual augmentation without tomography
- exact domain tool diagnostic, labeled `formal_oracle_not_admitted`

## Iteration 1 - Grounding And Hard Redirect

**Attack target:** The prompt could be read as reviving E2.

**Precommitted tokens:**

```text
B43_I1_CONFIRM_E3_IF_LIVE_SUPERVISOR_REDIRECTS_E2
B43_I1_KILL_E2_MAINLINE_IF_Q_LOOP_SAYS_PROXY_TRAP
B43_I1_VOID_IF_LIVE_DOCS_CONFLICT
```

**Result:** `B43_I1_CONFIRM_E3_IF_LIVE_SUPERVISOR_REDIRECTS_E2`.

`work_loop_batch42.md`, `question_loop_batch50.md`, and supervisor check-in #40
converge: E2 is instrumentation and absorber machinery; E3 is the live test.
`question_loop_batch51.md` further narrows the claim to counterfactual ranking
geometry and source-specific teacher measurements.

**Attack on conclusion:** Reading doctrine is not work. First fix the hard CPU
gate.

## Iteration 2 - Fix The Comparator Gate

**Attack target:** The E2/E3 infrastructure cannot even collect tests.

**Precommitted tokens:**

```text
B43_I2_CONFIRM_CPU_GATE_RESTORED_IF_TEST_EKLAVYA_E2_COLLECTS
B43_I2_KILL_GATE_IF_COMPARE_ABLATIONS_API_STILL_MISSING
B43_I2_VOID_IF_PYTEST_ENVIRONMENT_BLOCKS_COLLECTION
```

**Work:** Added `code/compare_ablations.py` with the API required by
`code/test_eklavya_e2.py`: `RunSummary`, `analyze_run`, `load_eval_results`,
`export_csv`, `evaluate_decision_rules`, `DECISION_RULES`, and
`GOLDFREE_RULES`.

**Result:** `B43_I2_CONFIRM_CPU_GATE_RESTORED_IF_TEST_EKLAVYA_E2_COLLECTS`.

`code/test_eklavya_e2.py` collected 473 tests. Focused tests that use no pytest
temp fixtures passed; temp-fixture tests remain blocked by sandbox temp
permissions.

**Attack on conclusion:** Restoring a comparator is infrastructure, not the E3
experiment. Move to runnable E3 code.

## Iteration 3 - First E3 Toy Precommit

**Attack target:** A toy can easily smuggle the answer.

**Precommitted token set in code:**

```text
signal: E3_TOY_SIGNAL_SOURCE_SPECIFIC_COUNTERFACTUAL_LESSON
absorbers: CE-only, single teacher, average/weighted vote, active hard mining,
           augmentation-only, shuffled measurements, shuffled identity
voids: nonfinite, empty split, exact tool granted
```

**Design decision:** binary candidate ranking, not generic classification prose.
Teachers emit signed margins as sensor readings. E3 compiles source-specific
counterfactual lessons; baselines get the same student class and comparable
teacher measurement budget.

**Attack on conclusion:** A precommit is cheap. Smoke before full run.

## Iteration 4 - Smoke V0 With Raw MLP

**Attack target:** The first implementation may only look plausible.

**Result:** smoke failed hard.

```text
terminal_token: E3_TOY_ABSORBED_BY_CE_ONLY
E3 hidden acc: 0.2812
CE-only hidden acc: 0.5000
```

This killed the first raw-MLP packet path. Either packet labels were wrong, or
the student could not absorb the structure.

**Attack on conclusion:** A smoke loss does not distinguish bad packets from bad
optimization. Run the full v0 sweep.

## Iteration 5 - Full V0 Sweep

**Attack target:** The smoke failure might be a tiny-budget artifact.

**Result:** full v0 still failed.

```text
terminal_token: E3_TOY_ABSORBED_BY_SINGLE_TEACHER
E3 hidden acc: 0.4477
best single teacher hidden acc: 0.5703
```

The first full result said the toy allowed one latent sensor plus calibration
labels to absorb the E3 advantage.

**Attack on conclusion:** Maybe the packet rule is good but the student cannot
learn the required interaction. Diagnose packet rule and student separately.

## Iteration 6 - Diagnose Packet Rule Versus Student

**Attack target:** E3 is dead in this toy.

**Diagnostic result on seed 0 at 1200 epochs:**

```text
calibration_packet_acc: 1.0
packet_value_prior: 0.40625
E3 hidden acc: 0.484375
B4_single_semantic_z0 hidden acc: 0.765625
```

The packet orientation was right. The problem was not teacher-rule inference; it
was the student/feature setup and a toy leak where a single sensor could combine
with calibration examples.

**Attack on conclusion:** Do not tune labels post hoc. Change only the student
landing zone so it can represent the counterfactual interaction generically.

## Iteration 7 - Add Generic Interaction Landing Zone

**Attack target:** The old MLP made optimization noise the result.

**Change:** replaced the raw-bit MLP with a tiny linear readout over generic
monomial features up to degree 4. This is not a target-specific feature; it is a
small interaction basis that lets XOR-style counterfactual ranking be learnable.

**Smoke result:** improved but still failed source-specificity.

```text
terminal_token: E3_TOY_ABSORBED_BY_TEACHER_AVERAGE_OR_WEIGHTED_VOTE
E3 hidden acc: 0.5938
shuffled measurements hidden acc: 0.6250
```

**Attack on conclusion:** One seed can overstate shuffled controls. Run the full
sweep before claiming or killing.

## Iteration 8 - Full 20-Seed Signal

**Attack target:** The smoke source-specificity failure may generalize.

**Result:** 20-seed full sweep passed.

```text
terminal_token: E3_TOY_SIGNAL_SOURCE_SPECIFIC_COUNTERFACTUAL_LESSON
E3 hidden acc: 0.8656
best ordinary absorber: 0.5203
E3 - best ordinary: +34.53 pp
E3 - active hard mining: +37.03 pp
E3 - shuffled sensors: +35.31 pp
```

The source-specific packet now beat CE-only, single teachers, average/weighted
vote, active hard-example mining, shuffled sensors, and augmentation-only.

**Attack on conclusion:** Twenty seeds are not enough for a hostile reviewer.
Stress with more seeds.

## Iteration 9 - 50-Seed Stress And Absorber Review

**Attack target:** The 20-seed signal is a seed artifact.

**Result:** 50-seed stress reproduced the signal.

```text
terminal_token: E3_TOY_SIGNAL_SOURCE_SPECIFIC_COUNTERFACTUAL_LESSON
E3 hidden acc: 0.8588
best ordinary absorber: 0.5034
E3 - best ordinary: +35.53 pp
forecast_ok: true
```

The exact domain tool diagnostic is 1.0 but marked `formal_oracle_not_admitted`.
If the exact hidden constructor is granted to baselines, this toy is absorbed by
that oracle. The current claim is only that teacher-specific sensor measurements
can teach the hidden transform without true hidden labels or teacher calls at
inference.

**Attack on conclusion:** A friendly synthetic world is not a moonshot. Set the
claim ceiling and next kill.

## Iteration 10 - Final Claim Ceiling

**Attack target:** A toy signal can become overclaiming.

**Final token:**

```text
B43_E3_TOY_SIGNAL_AT_FRIENDLY_CLAIM_CEILING_NEXT_EXACT_TOOL_KILL_REQUIRED
```

What is won:

- The first E3 toy is runnable.
- Teacher disagreement is not used as the target; source-specific sensor roles
  are inverted into a counterfactual ranking lesson.
- The student is teacher-free at inference.
- Shuffling measurements or teacher identity destroys the advantage.
- Active hard-example mining and raw teacher-output baselines do not match the
  hidden-transform transfer.

What is not won:

- Natural-language E3.
- Public usefulness.
- Exact-tool robustness.
- Proof that the probe generator is not doing too much work.
- Any claim that E3 is a paradigm shift outside this controlled toy.

Next hostile test:

```text
Give baselines the same transformation generator, add a nuisance-oracle and
exact-tool absorber, and require E3 to win only if source-specific teacher
measurements still add value beyond supplied augmentation geometry.
```

## NARRATIVE SECTION

**Gossip-magazine story:** Instead of copying big AI answers, a tiny AI learned
why the expert AIs disagreed in a toy world, and that reason transferred when
the surface form changed.

**Obvious? test:** Not obvious in the measured toy, because raw teacher copying,
best single teacher, average voting, weighted voting, and hard-example mining
all stayed near chance on the hidden transform while E3 reached 0.8588.

**Trivial? test:** Still vulnerable. The toy gives E3 a hand-authored
transformation family and explicit teacher roles. Shuffled identity failing is a
real anti-triviality check, but the next batch must test whether augmentation or
an exact tool absorbs the result when granted the same geometry.

**Mission test:** This is mission-aligned only as a cheap falsifier. A positive
toy says inspectable, shareable lesson packets might matter; it does not yet
make AI cheap, ubiquitous, or useful to everyone. The next result must move from
friendly toy signal to a harder domain where exact tools and augmentation get
first refusal.
