# OG-1 Seed 0 Correctness and Architecture Analysis

Date: 2026-07-03
Role: Correctness Engineer + Architecture Theorist
Inputs: `research/DEEP_RETHINK.md`, `code/toy_opgeom_og1.py`, `research/RESEARCH_NOTES.md`, seed 0 results plus seed 1 partials.

## Executive Verdict

OG-1 seed 0 is informative, but it does not validate Operational Geometry as originally gated.

The strongest real result is not "D wins"; it is that wrong behavioral structure is actively destructive. Variant E collapses to chance because the implementation trains ranking and counterfactual losses on intentionally wrong labels. That is a genuine control result, not a simple implementation bug.

The positive D signal is weaker. D beats A by +2.6pp MCQ and B by +3.7pp transformed accuracy on seed 0, but:

1. Invariance is dead, so D is effectively `CE + ranking + counterfactual`.
2. F is nearly D, because F keeps the active correct ranking and counterfactual losses; only its dead invariance channel is corrupted.
3. The active counterfactual loss is equivalent to supervised counterfactual data augmentation, not yet a distinct OG relationship loss.
4. One seed is far too small for a +2.6pp MCQ claim, especially with seed 1 A/B already moving in the opposite direction.
5. Training was not optimized: constant `1e-3` Adam appears to overshoot after about 6K steps.

The right conclusion:

> OG-1 falsified naive ranking/invariance expectations, confirmed that correct vs incorrect behavioral labels matter, and found a plausible counterfactual-supervision signal. It did not yet prove architecture-independent operational geometry.

## Correctness Review

### E Catastrophe: Genuine, but More Adversarial Than the Label Implies

Variant E is implemented as wrong-label training, not merely shuffled/noisy ranking.

In `train_student()`, for E ranking:

```python
shuffled_distractors = list(distractors)
rng.shuffle(shuffled_distractors)
fake_correct = shuffled_distractors[0]
fake_distractors = [correct] + shuffled_distractors[1:]
```

The fake correct answer is always selected from the distractors, so it is never the true label. The true correct answer is explicitly demoted into the distractor set.

For E counterfactual:

```python
cf_distractors_shuffled = list(cf_distractors)
rng.shuffle(cf_distractors_shuffled)
fake_cf_correct = cf_distractors_shuffled[0]
fake_cf_dis = [cf_correct] + cf_distractors_shuffled[1:]
```

Again, the fake counterfactual correct answer is always wrong, and the true counterfactual correct answer is treated as a distractor.

So E's 27.2% MCQ is credible. It means the auxiliary losses are not harmless regularizers. They can override CE and train anti-knowledge. This is a real result.

But the wording should be corrected:

- Current label: "shuffled labels"
- More accurate label: adversarial wrong-label OG control

A true random-label control should sample uniformly among all four candidates. The current E samples uniformly among the three wrong candidates, which is harsher and will push toward chance or below.

### Invariance Loss Is Functionally Dead

The reported dynamics match the code. `L_inv` is a symmetric KL between candidate score distributions:

```python
p = F.log_softmax(scores_orig / tau, dim=-1)
q = F.softmax(scores_trans.detach() / tau, dim=-1)
kl1 = F.kl_div(p, q, reduction="batchmean")
```

The intended object is sensible, but in this setup it has almost no useful gradient:

1. There are only 4 candidates.
2. Scores are average byte log-probabilities and are close together early.
3. With `tau=1.0`, softmaxes are near-uniform.
4. KL between uniform distributions is zero.
5. `reduction="batchmean"` on a 1D tensor divides by candidate count, shrinking an already tiny signal by about 4x.

Therefore C and D are not testing the designed preserving-invariance component. C is effectively B. D is effectively B plus counterfactual.

This also explains why F is not a useful negative control for D: F corrupts the invariance pair, but the invariance pair carries nearly no gradient. F keeps correct ranking and correct counterfactual losses, so F should be close to D. It is.

### Ranking Alone Does Not Help

Seed 0: B is +0.2pp over A.
Seed 1 partial: B is -3.2pp below A.

This is consistent with the code and task. CE already trains the answer token with 5x weight at the answer position. The ranking loss asks the same model head to prefer the same answer among same-attribute candidates. That is not much new information.

Ranking may still matter on real multiple-choice continuation tasks where candidates are long and CE is diffuse, but OG-1 does not show ranking as independently valuable.

### Counterfactual Is the Only Plausibly Active Positive Signal

Seed 0:

- D - B MCQ: +2.4pp
- D - B avg transformed: +3.7pp
- D - C MCQ: +3.6pp
- D - C avg transformed: +2.5pp

Given dead invariance and weak ranking, the active new ingredient is counterfactual training: "query the other entity."

That is real signal. It teaches a second lookup query for the same factual scene and forces entity binding rather than shallow answer-position memorization.

But this does not yet distinguish OG from ordinary counterfactual data augmentation.

### Scoring Implementation Is Mostly Correct for This Toy

`score_candidate()` and `score_candidates_batch()` score candidate words from the `answ` patch:

```python
n_ctx_patches = len(context_bytes) // self.patch_size
pred_patch = n_ctx_patches - 1
```

Because each word is exactly 4 bytes and each word is one patch, this is the right patch: the hidden state after `answ` predicts the answer word patch.

Two caveats:

1. Candidate scores are averaged per byte. Since all words are length 4 here, this is equivalent to sum up to a constant. It would bias variable-length candidates in real use.
2. The byte decoder predicts all 4 bytes of the next patch in parallel from the previous patch state. It cannot condition on byte 1 when predicting byte 2 of the same candidate. That is acceptable for this toy but becomes a real limitation for byte-level continuation scoring.

### Evaluation Is Too Thin for 2-3pp Claims

Each variant is evaluated on 500 MCQ examples. Around 50-56% accuracy, the independent binomial SE for a single model is about 2.2pp. For a model difference, independent SE is about 3.1pp, lower if paired but still nontrivial.

D - A = +2.6pp is only 13 additional correct examples out of 500. That is not a reliable effect from one seed.

The transformed metric averages 3 transforms x 500 examples, so D - B = +3.7pp is more plausible, but it is still weakened by the fact that F beats D on avg_trans in seed 0.

## Gate Assessment

### Original Gates

| Gate | Seed 0 Result | Verdict |
|---|---:|---|
| D vs A MCQ >= +8pp | +2.6pp | Fail |
| D vs B transformed >= +3pp | +3.7pp | Pass, but fragile |
| D vs E MCQ >= +6pp | +29.0pp | Pass, strong sanity result |
| D vs F MCQ >= +6pp | +0.6pp | Fail, expected because F keeps active losses |
| BPB degradation <= 5% | +0.8% | Pass |

### Were the Gates Too Aggressive?

The +8pp D-vs-A gate was aggressive but not conceptually wrong. For a first proof of a new paradigm, a large effect is the right demand.

The problem is that the implementation did not actually activate all designed channels. The gate is not too aggressive for full OG-1; it is too aggressive for the effective recipe that actually ran.

The D-vs-F gate is miscalibrated under this implementation. F corrupts only the dead invariance channel. It is not a control for counterfactual signal, ranking signal, or full OG structure.

### Recalibrated OG-1b Gates

For OG-1b, separate engineering sanity gates from scientific validation gates.

Engineering gates:

1. Teacher MCQ >= 98% preferred; >=95% minimum.
2. `L_inv` active: nonzero candidate entropy movement and invariance gradient norm >= 5% of CE gradient after warmup.
3. Save/evaluate checkpoints at 4K, 6K, 8K, and best validation checkpoint.
4. Use LR decay or plateau selection; do not report only the final overshot checkpoint.
5. Report mean +/- std over at least 5 seeds, not 1-3.

Scientific gates:

1. D beats A by >= 4pp mean MCQ, with paired bootstrap CI excluding 0.
2. D beats B by >= 3pp on held-out preserving and counterfactual transforms, CI excluding 0.
3. Correct-label D beats adversarial-label E by >= 10pp. This remains a sanity control, not proof of OG.
4. Correct invariance beats random-pair invariance by >= 2pp on preserving consistency once invariance is active.
5. D beats a matched counterfactual data-augmentation baseline by >= 2pp, or OG novelty is not established.
6. BPB degradation remains <= 5% relative or <= 0.05 absolute BPB, whichever is stricter for the scale.

## Signal Assessment

The 2.6pp D-A gap is not meaningful yet.

Reasons:

1. It is one seed.
2. It is only 13 examples on a 500-example evaluation.
3. Seed 1 partial already shows A and B moving by several points.
4. The final checkpoint appears suboptimal due to no LR decay.
5. D and F are essentially tied, which means the measured D advantage is not specific to the intended full OG structure.

The D-B transformed gap is more interesting, but should be treated as a lead, not a conclusion. It is likely the counterfactual term improving general binding behavior.

## Fixing Dead Invariance

Do not only tune `tau`. The loss object needs to be made supervised, margin-aware, and instrumented.

Recommended fixes:

1. Change KL reduction: use shape `[1, C]` or `reduction="sum"` instead of 1D `batchmean`.
2. Lower temperature: sweep `tau in {0.05, 0.1, 0.2, 0.5}` and track entropy.
3. Add transformed-context supervised ranking:

```python
L_preserve_rank = CE(score(T_p(x), Y) / tau, gold)
```

This cannot be zero just because original and transformed distributions are both uniform.

4. Add margin consistency on score gaps:

```python
g_orig = scores_orig - scores_orig[gold]
g_trans = scores_trans - scores_trans[gold]
L_gap_inv = smooth_l1(g_orig, g_trans)
```

5. Use harder/larger candidate sets. More candidates alone does not fix uniform-uniform KL, but it makes the score geometry richer once ranking starts to form.
6. Track per-loss gradient norms. A loss whose scalar value is nonzero can still be irrelevant if its gradient is clipped away or dominated.
7. Consider an EMA-teacher consistency target from the same student. Self-consistency between two simultaneously weak branches is fragile.

The most practical OG-1b invariance recipe:

```python
L_inv_total =
    0.50 * CE(scores_trans / tau_rank, gold)
  + 0.25 * KL(stopgrad(softmax(scores_orig / tau_cons)), softmax(scores_trans / tau_cons))
  + 0.25 * SmoothL1(center(scores_orig), center(scores_trans))
```

This tests both "same answer remains correct" and "relative candidate geometry is preserved."

## Next Experiment

Do not scale directly to harder tasks yet. Run OG-1b first.

OG-1b should be a cleanup experiment, not a bigger experiment:

1. Finish seeds 1 and 2 for historical continuity, but do not let them decide the strategy alone.
2. Add checkpoint selection at 6K and LR decay.
3. Add the missing counterfactual transform: "change queried slot."
4. Add a real counterfactual evaluation split, not only preserving transforms.
5. Add active invariance fixes above.
6. Add data-augmentation baselines:
   - A0: CE only, original examples.
   - A1: CE only, matched extra original examples.
   - A2: CE on original + preserving transformed examples.
   - A3: CE/ranking on original + counterfactual examples.
   - D: OG losses.
7. Add controls that corrupt each active channel independently:
   - wrong ranking only
   - wrong counterfactual only
   - random invariance only
   - random all active channels
8. Run at least 5 seeds.

Only after OG-1b beats the matched augmentation controls should the ladder move to a harder synthetic compositional task.

## OG vs Data Augmentation

Claude's concern is correct: current `L_cf` is essentially counterfactual data augmentation with relabeling.

It trains:

```python
CE(score(T_c(x), Y_c), cf_gold)
```

That is supervised learning on an edited example. There is no explicit loss on the relationship between original and counterfactual behavior.

To test true OG, add a relationship/equivariance loss. Examples:

### Pairwise Rank Transport

For preserving transforms:

```python
rank_order(scores(x, Y)) == rank_order(scores(T_p(x), Y))
```

For counterfactual transforms with a candidate map `phi`:

```python
rank_order(scores(x, Y)) == rank_order(scores(T_c(x), phi(Y)))
```

This requires transforms whose candidate mapping is defined. "Change queried slot" can define a map between attribute-specific candidate spaces only through metadata; "query other entity" is a lookup change, not a simple candidate relabeling.

### Program-Consistency Loss

Train the model to satisfy the same latent program across original and counterfactual examples:

```text
answer = binding[query_entity][query_attr]
```

The neural loss can approximate this by enforcing all generated neighborhood examples from one latent scene to be solved consistently. This is closer to OG: the lesson is the lookup program, not any single augmented row.

### Required Baseline

If D does not beat `CE + counterfactual augmentation` at matched compute and matched example count, the honest conclusion is:

> OG-1 discovered useful counterfactual augmentation, not a new training principle.

That is still useful, but it is not enough for the Sutra thesis.

## Translation to Real Sutra Byte-Level LM

The real Sutra implication is not "add this exact loss to byte LM pretraining."

The implication is:

1. Keep CE as the base language-modeling spine.
2. Use teachers as behavioral data engines, not hidden/logit targets.
3. Compile lesson neighborhoods: original prompts, paraphrases, preserving transforms, counterfactual edits, hard negatives, and verifier labels.
4. Train byte-level Sutra to score full continuations and candidate completions, not just next-byte marginals.
5. Use OG losses as auxiliary sequence-level energy/ranking losses over continuation sets.
6. Always compare against matched data augmentation and matched CE compute.

For HellaSwag-like tasks:

- `C`: context prefix.
- `Y`: candidate endings.
- `T_p`: paraphrases, irrelevant context edits, style changes that preserve the best ending.
- `T_c`: minimal premise/action edits that change the best ending.
- `R`: teacher/verifier ranking over endings.
- `V`: held-out transformed candidate ranking.

For byte-level modeling, candidate scoring should be sequence log-likelihood over full endings with length normalization and calibration. A separate energy/reranker head may be cleaner than forcing the byte decoder alone to carry all ranking geometry.

## Brutal Bottom Line

OG-1 seed 0 is a useful negative/partial-positive experiment:

- Validated: Wrong behavioral labels are destructive; teacher/lesson labels are active causal signals.
- Suggested: Counterfactual supervision may help binding and transformed robustness.
- Not validated: Full Operational Geometry.
- Not validated: Invariance learning.
- Not validated: Ranking-only improvement.
- Not validated: Novelty beyond counterfactual data augmentation.
- Not statistically established: D's +2.6pp over A.

The next correct move is OG-1b with active invariance, missing counterfactuals, LR decay/checkpointing, data-augmentation controls, and 5+ seeds. Scaling now would hide the mechanism instead of clarifying it.
