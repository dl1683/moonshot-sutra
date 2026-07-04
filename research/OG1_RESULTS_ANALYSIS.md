# OG-1 Results Analysis

**Date:** 2026-07-03
**Role:** Correctness Engineer + Architecture Theorist
**Subject:** OG-1 Operational Geometry toy experiment, seed 0 complete and seed 1 partial

## Executive Verdict

OG-1 is not a validation of Operational Geometry. It is an informative weak-positive / negative diagnostic.

The teacher and task are valid enough for a toy probe: teacher MCQ is 99.4%, so the task is learnable. But Variant D's seed 0 gain over CE-only is small: +2.6pp MCQ. That is below the original +8pp gate and below what I would trust from one seed with 500 eval examples. The only credible active gain is the counterfactual channel: D beats B by +2.4pp MCQ and +3.7pp transformed accuracy on seed 0. That gain is not yet separable from ordinary counterfactual data augmentation.

The invariance loss is operationally dead in C/D. The F ~= D anomaly is therefore expected, not mysterious. E's collapse is real in the sense that the code adversarially mislabels ranking and counterfactual targets while keeping CE correct. It is not evidence that OG theory works; it is evidence that the auxiliary losses have real gradient authority when labels are wrong.

The hard conclusion: OG-1 does not prove architecture-independent lesson geometry. It shows that a small byte-patch student can be nudged by structured counterfactual examples, while ranking and invariance as implemented add little or nothing. OG-1b is worth running, but only after fixing the measurement and control design.

## Design Context

The Operational Geometry frame in `research/DEEP_RETHINK.md` makes a strong distinction:

```text
Coordinate losses transfer coordinates.
Behavioral losses transfer behavior.
Invariant losses transfer knowledge.
```

The formal lesson object is `(C, Y, R, T_p, T_c, P, V)`: context family, candidate space, ranking relation, preserving transforms, counterfactual transforms, program/rule, and verifier.

OG-1 was designed to test whether ranking, preserving-invariance, and counterfactual losses could beat CE-only on a synthetic binding task using public behavioral variables. The planned variants were A: CE, B: CE + ranking, C: CE + ranking + invariance, D: CE + ranking + invariance + counterfactual, E: shuffled labels, and F: random preserving pairs.

The literature notes support the components individually: ranking / preference distillation, energy scoring, counterfactual augmentation, invariant-risk-style consistency, and black-box behavioral training. But literature support for components is not evidence that this implementation extracted operational geometry.

## Correctness Audit

### E's Catastrophe Is Genuine, But Narrower Than Claimed

E does not corrupt CE. The main CE target remains `tokens + [correct]`. The shuffled behavior occurs only in the auxiliary ranking and counterfactual losses.

For ranking, the code copies the distractors, shuffles them, chooses one distractor as `fake_correct`, and moves the true `correct` answer into the fake distractor list. For counterfactuals, it does the same thing with `cf_distractors` and `cf_correct`.

I ran a generator-level check over 10,000 examples. `fake_correct == correct` occurred 0 times. That is expected: E always picks a distractor as the fake gold label. So E is an adversarial-label control, not a random permutation control where the true answer sometimes remains gold.

Therefore E's 27.2% MCQ is a genuine implementation result: wrong auxiliary labels pull the model almost to chance despite correct CE. But the correct interpretation is limited:

- It proves the auxiliary ranking/CF losses are not inert.
- It proves wrong labels are destructive.
- It does not prove the correct labels encode uniquely Operational Geometry rather than ordinary supervised augmentation.
- It does not prove the losses are not regularizers in general; it proves adversarially mislabeled auxiliary losses are not harmless regularization.

### Teacher Is Not Used During Student Training

`train_student(student, teacher, ...)` accepts `teacher`, but never uses it. The ranking and counterfactual labels are generated from synthetic metadata, not from teacher behavior.

This is acceptable for a controlled toy rule-learning experiment, but it weakens the teacher-tomography claim. OG-1 is not actually querying a teacher to infer operational invariants. It is training from a known symbolic generator with handcrafted transforms. Real Sutra will not have metadata fields telling it the correct causal binding.

### Invariance Loss Is Dead In The Correct Variants

Claude's diagnosis is mostly right, but incomplete. The invariance loss computes symmetric KL between student score distributions over only four candidates, with `tau = 1.0` and `reduction="batchmean"` on a 1D vector.

Problems:

- Four candidates are too few.
- `tau = 1.0` is soft for average byte log-prob scores, so early distributions are close to uniform.
- `batchmean` on a 1D vector divides by the candidate dimension, shrinking the KL by about 4x relative to the intended one-example KL.
- Preserving transforms are often too easy: swap, irrelevant edit, and other-entity rename may already yield almost identical candidate distributions.
- The loss matches the student's own score distribution across transforms; it does not inject new correctness information.

F's `L_inv` grows because unrelated contexts eventually produce different score distributions. That proves the code path can produce nonzero invariance loss. C/D's near-zero `L_inv` means the preserving-invariance term contributes effectively no useful gradient under the correct transform setup.

So D's active recipe is approximately:

```text
CE + 0.35 * ranking + 0.25 * counterfactual
```

not full OG-1.
### Counterfactual Coverage Is Incomplete

The design specified two counterfactual transforms:

1. Query the other entity.
2. Change the queried slot.

The code implements only query-other-entity. I also checked the query-other transform over 10,000 generated examples. The answer changed 91.23% of the time. All non-changing cases came from the `actn` attribute because both entities can share the same action. So even the implemented counterfactual is not always answer-changing. That is not fatal, but it means the counterfactual loss is slightly diluted and narrower than the design.

### LR Schedule And Checkpointing Are A Real Confound

Student training uses Adam at constant `1e-3` for 8K steps with no decay. Seed 0 D appears to converge around 6K and regress by 8K:

```text
CE:   1.12 -> 0.94 -> 0.71 -> 0.95
rank: 0.82 -> 0.11 -> 0.054 -> 1.34
cf:   0.82 -> 2.40 -> 0.054 -> 1.55
```

The 8K checkpoint is probably not the best D checkpoint. But this does not rescue the result. It means OG-1's checkpointing protocol is undercontrolled. OG-1b must log validation at multiple checkpoints or use a fixed validation-selection rule.

### Evaluation Has A Minor Tie Bias

`evaluate_mcq()` shuffles choices. `evaluate_transformed_mcq()` does not; it uses `[correct] + distractors`. If scores tie exactly, `np.argmax` picks index 0, which is the correct answer. The observed transformed accuracies are far from 100%, so this is not the main explanation, but OG-1b should shuffle choices or explicitly randomize tie-breaking in transformed eval too.

### Reproducibility Minor: Python `hash(v)`

The code calls `torch.manual_seed(seed + hash(v) % 10000)`. Python hash randomization can differ across processes. In this code it likely has little effect because the base state is loaded before that call and there is no dropout, but it is still a reproducibility footgun for future variants.

## Assessment Of Claude's 8 Findings

| # | Finding | Verdict |
|---|---------|---------|
| 1 | Invariance loss is dead | Correct operationally for C/D. Diagnosis should include 4-candidate softness, `batchmean` scale shrinkage, and trivial preserving transforms. |
| 2 | D's effective recipe is CE + ranking + CF | Correct for current run. |
| 3 | No LR decay makes 8K suboptimal | Plausible and supported by dynamics. Not proven without 6K eval, but enough to require fixed checkpoint selection in OG-1b. |
| 4 | Ranking alone does not help | Correct for current evidence. B ~= A seed 0, B < A seed 1. Not a universal verdict on ranking. |
| 5 | Counterfactual is the key | Correct as the only visible active positive channel, but the magnitude is small and not yet separable from data augmentation. |
| 6 | E catastrophe proves losses shape learning | Correct narrowly. E is adversarial-label auxiliary training, not proof of OG. |
| 7 | Missing second counterfactual transform | Correct. The implementation lacks change-queried-slot. |
| 8 | Current CF is data augmentation with relabeling | Correct and central. This is the critical unresolved issue. |

## Gate Assessment

Original gates on seed 0:

| Gate | Required | Seed 0 | Assessment |
|------|----------|--------|------------|
| D vs A MCQ | +8pp | +2.6pp | Fail. Not close. |
| D vs B transforms | +3pp | +3.7pp | Pass, but one seed only. |
| D vs E MCQ | +6pp | +29.0pp | Pass, but E is adversarial-label sanity control. |
| D vs F MCQ | +6pp | +0.6pp | Fail; expected because invariance is dead. |
| BPB degradation | <=5% | +0.8% | Pass. |

The original +8pp D-vs-A gate was aggressive for width 64, 8K online steps, no LR decay, and a byte-patch student. But it was not crazy as a moonshot-validation gate. If OG had extracted a strong transferable invariant, a simple binding task should have shown a large gain.

The right interpretation:

- The gate was too high for "continue exploring."
- The gate was appropriate for "validated."
- OG-1 failed validation.
- OG-1 produced enough signal to justify OG-1b.

## Statistical Signal Assessment

### D vs A MCQ Is Not Trustworthy Yet

Seed 0:

```text
A = 53.6%
D = 56.2%
gap = +2.6pp
```

But A moved from 53.6% on seed 0 to 50.6% on seed 1, a 3.0pp swing. With 500 eval examples, a single accuracy estimate around 50-56% has binomial standard error around 2.2pp. A two-model unpaired difference has rough standard error around 3pp before adding training-seed variance.

So +2.6pp D-A on one complete seed is not a reliable effect. It is suggestive at best.

### D vs B Transform Gain Is More Interesting But Preliminary

Seed 0:

```text
B avg_trans = 48.5%
D avg_trans = 52.2%
gap = +3.7pp
```

This is the strongest positive result because it is on the transform metric the CF/invariance machinery was supposed to help. But it is still only one seed, and the transformed eval has deterministic candidate ordering. Treat this as a hypothesis for OG-1b, not as a pass.

### C Is A Warning

Seed 1 C has the best final CE trajectory among A/B/C but the worst transform score:

```text
C CE final = 0.70
C avg_trans = 41.5%
A avg_trans = 45.6%
B avg_trans = 45.0%
```

That says lower training CE is not reliable evidence of learning the binding geometry. It also says the current invariance/ranking combination can interfere with robust behavior.

### F ~= D Does Not Rescue Invariance

Seed 0:

```text
D MCQ = 56.2%
F MCQ = 55.6%
D avg_trans = 52.2%
F avg_trans = 54.3%
```

Given dead invariance in D, F is effectively another CE + rank + CF variant with a mostly separate, low-weight random-invariance nuisance. F matching D is therefore expected. If anything, F's better transform score reinforces the conclusion that the current invariance term is not the mechanism.
## Gate Recalibration For OG-1b

Separate debug gates from validation gates.

### Debug Gates

These decide whether the run is healthy:

1. Teacher MCQ >= 95%.
2. A reaches at least 50% MCQ by 8K, or the student/training setup is too weak to diagnose OG.
3. No checkpoint regression larger than 2pp from selected validation checkpoint to final checkpoint without being reported.
4. `L_inv` nonzero and gradient-bearing under deliberately mismatched preserving/random pairs.
5. Per-example eval records are saved for paired bootstrap or McNemar-style tests.
6. BPB degradation for OG variants <= 5% vs matched CE baseline.

### Realistic Toy Gates

For a width-64, 8K-12K toy:

| Comparison | OG-1b gate |
|------------|------------|
| D_rel vs A MCQ | >= +3pp mean over 3 seeds and paired CI excludes 0, or >= +5pp raw mean if CI is not yet implemented |
| D_rel vs B transforms | >= +2pp mean over 3 seeds and CI excludes 0 |
| D_rel vs best data-augmentation baseline | >= +2pp on held-out transform families |
| D_rel vs E adversarial | >= +15pp sanity gap |
| Fixed invariance vs random-invariance F | >= +2pp on preserving-consistency stress tests, not necessarily MCQ |
| BPB degradation | <= +5% |

For a larger width-128 / longer run:

| Comparison | Stronger gate |
|------------|---------------|
| D_rel vs A MCQ | >= +5pp mean over 3 seeds |
| D_rel vs best augmentation baseline | >= +3pp on held-out transforms |
| Transform transfer | improvement on at least one transform family not trained directly |

Do not use D vs E as a primary success criterion. E is a health check for label authority.

## How To Fix Invariance

The current KL over 4 candidate softmaxes is too weak. Options, ranked:

1. Replace or supplement KL with centered score-vector matching:

```text
s0 = scores_orig - mean(scores_orig)
s1 = scores_trans - mean(scores_trans)
L_inv = huber(zscore(s0), zscore(s1))
```

This keeps gradients when softmax distributions are near-uniform and removes global score-offset dependence.

2. Match pairwise margins, not probabilities:

```text
margin_i = score(correct) - score(distractor_i)
L_inv = mean((margin_orig_i - margin_trans_i)^2)
```

This directly preserves the decision geometry.

3. Increase candidates from 4 to 8-16, with hard negatives. Four choices make the simplex too small and the entropy too high.

4. Sweep tau: `1.0, 0.5, 0.25, 0.1`. Log entropy, max probability, and gradient norm. Do not choose tau by vibes.

5. Fix the `batchmean` scaling. For a single vector, use `reduction="sum"` or shape it as `[1, K]` so `batchmean` divides by batch size, not candidate count.

6. Train on all preserving transforms per example at least during diagnostics. Randomly sampling one transform hides which transform is dead or harmful.

7. Add a contrastive transform objective: preserve pairs should have similar score geometry, counterfactual pairs should differ in the predicted structured direction, and unrelated pairs should not be forced together.

Do not accept "invariance adds nothing" yet. The implementation has not tested a live invariance loss.

## What OG-1b Should Be

Finish the pending seed only for archival completeness. Do not use it to make a go/no-go decision until the controls are fixed.

### Training Fixes

1. Use LR warmup plus cosine decay, or at minimum decay after 6K.
2. Save and evaluate checkpoints at 4K, 6K, 8K, and 12K.
3. Select checkpoint by a fixed validation split, not by post hoc test score.
4. Log loss components, gradient norms per component, candidate entropy, correct-vs-best-wrong margin, and transform consistency.
5. Increase eval to at least 2,000 examples or save per-example outcomes for paired tests.

### Variant Fixes

Use these variants:

| Variant | Purpose |
|---------|---------|
| A | CE only |
| A_more | CE only with matched extra examples / compute |
| A_aug | CE on ordinary additional generated examples |
| A_cf_ce | CE on original + counterfactual token sequences, no relational loss |
| B | CE + ranking |
| C_fixed | CE + ranking + fixed live invariance |
| D_aug | CE + ranking + counterfactual relabeling, current style |
| D_rel | CE + ranking + fixed invariance + relational counterfactual objective |
| E_adv | Adversarial-label auxiliary sanity control |
| F_rand_inv | Fixed invariance objective paired with unrelated contexts |

The key comparison is not D vs A. It is:

```text
D_rel vs A_cf_ce
```

If D_rel cannot beat a matched counterfactual data-augmentation baseline on held-out transform families, Operational Geometry has not shown up.

### Counterfactual Fixes

1. Add change-queried-slot.
2. Reject no-op counterfactuals where the answer does not change.
3. Train on one counterfactual family and test on the other.
4. Add two-hop counterfactuals: change slot and entity together.
5. Measure whether the model learns the causal map, not just the edited examples.

### Invariance Evaluation

Add metrics beyond accuracy:

```text
preserve_agreement = P(argmax original == argmax transformed)
preserve_margin_delta = abs(margin original - margin transformed)
cf_direction_accuracy = P(predicted answer changes according to phi)
unrelated_separation = distance(original, unrelated) > distance(original, preserve)
```

This will distinguish real geometry from lucky MCQ movement.
## Is The CF Gain Operational Geometry Or Data Augmentation?

Current answer: treat it as data augmentation until proven otherwise.

The current CF loss presents a modified context and trains the correct answer for that modified context:

```text
query Alice color -> red
query Bob color   -> blue
```

That is ordinary supervised counterfactual augmentation with relabeling. It is useful, and it is literature-supported, but it is not by itself Operational Geometry.

Operational Geometry would train or test the relationship:

```text
When the queried entity changes from Alice to Bob,
the answer should change from binding[Alice][slot] to binding[Bob][slot].
```

The distinction matters. Data augmentation says:

```text
Here is another labeled example.
```

Operational Geometry says:

```text
Here is the rule governing how labels move under interventions.
```

How to distinguish:

1. Matched augmentation baseline: train `A_cf_ce` on the same number of original and counterfactual examples. If D only matches `A_cf_ce`, the gain is augmentation.
2. Held-out transform transfer: train on query-other and test on change-slot, or train on single transforms and test on composed transforms. Data augmentation should mostly help seen edits; relational geometry should transfer.
3. Causal direction objective: given original scores and edit type, predict which candidate should rise and which should fall. Reward correct score deltas, not just final answer CE.
4. Minimal-data scaling: if D_rel needs far fewer CF examples than A_cf_ce for the same held-out transfer, that supports geometry.
5. OOD binding size: train on 2 entities and test on 3 entities; train on 3 attributes and test on held-out attribute families where possible. Relational rules should extrapolate better than memorized augmented pairs.
6. Teacher-query version: remove metadata labels, query a teacher/verifier over neighborhoods, infer transforms from behavior, then train. If the system still works, it is closer to Eklavya/OG.

Until these controls pass, CF gain is not an OG result. It is a useful augmentation result.

## Translation To Real Sutra

The toy binding task is clean because the world state is explicit:

```text
name -> action/color/room
query -> exact slot lookup
```

Real byte-level language modeling is not like that. At scale:

1. The candidate space is huge. MCQ-style scoring over 4 candidates becomes ranking over many continuations.
2. Transform labels are noisy. Paraphrases, irrelevant edits, entity swaps, temporal changes, and causal edits need verifier support.
3. There is no metadata oracle. Teacher behavior must be probed, and teacher errors/disagreements must be modeled.
4. BPB remains a weak proxy. A model can improve byte prediction without improving benchmark behavior.
5. Byte-patch architecture adds mechanical burden. Patch lag and byte spelling may consume capacity that token models spend on semantics.
6. Counterfactual generation becomes the core bottleneck. Bad counterfactuals create E-like destructive gradients.
7. The energy/ranking head may need to be explicit. Using the LM head as an energy scorer is convenient, but it may not be the best scoring interface for long continuations.

The correct Sutra mapping is not "add CF examples to LM training." It is:

```text
teacher behavior probes
-> infer lesson graph with preserving and counterfactual transforms
-> compile curricula and relational objectives
-> train byte model plus energy/ranking interface
-> validate on held-out transformation families and benchmark tasks
```

The strongest real-world version of OG is not a loss term. It is a data engine and verifier loop that turns teacher behavior into reusable intervention structure.

## Final Judgment

OG-1 partially falsifies the naive version of the design:

- Ranking alone is not enough.
- Current invariance is nonfunctional.
- Current D is not full Operational Geometry.
- Current CF gain is indistinguishable from counterfactual data augmentation.
- The original validation gates fail.

OG-1 does not falsify the deeper Operational Geometry thesis:

- E shows auxiliary behavioral labels have real causal force.
- D vs B transform improvement suggests counterfactual structure may help.
- The F anomaly is explained by dead invariance, not by failure of invariance as a concept.
- The experiment found exactly the next design obligation: distinguish relational geometry from augmented supervision.

Proceed to OG-1b, but with a stricter claim boundary:

```text
OG-1 result: counterfactual supervised structure is promising but unproven.
OG-1b target: prove relational transform learning beats matched augmentation.
No Sutra-scale translation should be claimed until that passes.
```
