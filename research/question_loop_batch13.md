# QUESTION LOOP - Batch 13: Attack Prior Floor Interpretation + Margin Shadow Design

Date: 2026-07-07

Iterations: 85-91

## Grounding

I read the requested context in order:

1. `research/VISION.md`
2. `research/dual_loop_supervisor_checkin_8.md`
3. `research/work_loop_batch9.md`
4. `research/question_loop_batch12.md`
5. `tmp_coordinate_inheritance_v1/smoke128_repair/preflight_metrics.json`
6. `research/work_loop_batch8.md`
7. `code/coordinate_inheritance.py`

No GPU runs, training runs, benchmark runs, or experiments were performed for this batch. This is analysis only.

## Binding Interpretation Entering Batch 13

The v1 smoke is killed under the original precommitted gate:

```text
FAIL_STAGE1_V1_PREFLIGHT
DO_NOT_RUN_STAGE2
```

The supervisor's new interpretation may be useful, but it is not allowed to erase that verdict. The honest state is:

```text
v1 failed the old disruption gate.
The prior-floor decomposition is a post-failure causal hypothesis.
The revised coordinate-specific gate can only be a v2 gate unless rerun under a new predeclared protocol.
```

The strongest positive evidence is still nontrivial:

| Readout | Copied vs random lift | Gaussian destroyed-input NLL | Coordinate lift above Gaussian | Gaussian retained fraction |
|---|---:|---:|---:|---:|
| token-end | 5.75 nats | 16.072 | 3.82 nats | 33.5% |
| patch-boundary | 5.36 nats | 16.440 | 2.83 nats | 47.3% |

But the strongest hostile fact is also nontrivial:

```text
When the adapted input is replaced by same-norm Gaussian noise, copied Qwen layers still retain 33-47% of the copied-vs-random lift.
```

The code makes the interpretation narrow:

- Stage 1 is token-space next-token NLL through a Qwen head, not byte BPB.
- `generic_pretrained` is Qwen layers 14-17, not a non-Qwen generic pretrained control.
- The 128-sequence smoke trains on 80% of collected samples and evaluates on 26 sequences per readout.
- Benchmark mode currently reports accuracy deltas, not gold-vs-best-wrong margins, and uses validation by default.
- Benchmark mode includes random, shuffled, Qwen-middle, rotated, and dim-permuted variants, but not same-norm Gaussian destroyed-input, inverse recovery, true-embedding upper bound, Wide7, or native non-Qwen controls.

That means the next margin shadow is not just a command to run. It needs a stricter measurement contract.

---

## Iteration 85: Is The Prior Floor Interpretation Correct?

### Steelman

The supervisor's decomposition is algebraically clean if the Gaussian replacement is accepted as a destroyed-input copied-core baseline:

```text
total lift = random_calibrated_nll - copied_calibrated_nll
prior floor = random_calibrated_nll - copied_gaussian_nll
coordinate-specific lift = copied_gaussian_nll - copied_calibrated_nll
```

Under that accounting:

- token-end has 3.82 nats of input-dependent lift above the destroyed-input floor;
- patch-boundary has 2.83 nats of input-dependent lift above the destroyed-input floor;
- both exceed the old 2.0-nat large-signal scale;
- Qwen middle layers 14-17 lose by 3.9-4.1 nats, so the signal is not simply any copied Qwen depth;
- exact inverse recovery is 100%, so the machinery can preserve the calibrated path when the inverse is applied.

The generous interpretation is:

```text
The old copied-vs-random number mixed two real things: a pretrained Qwen-core prior and a coordinate-sensitive byte-to-Qwen signal. The disruption test did not show that the coordinate signal is fake. It showed that copied pretrained layers have an unconditional floor that random layers do not have.
```

That is a plausible discovery. Pretrained transformer blocks are not blank functions. Their LayerNorm/RMSNorm statistics, attention priors, MLP feature maps, residual scales, and Qwen head coupling can produce better-than-random token distributions even when the input is damaged.

### Attack

The decomposition is not yet causally valid. It is an accounting identity wearing a causal label.

The term prior floor assumes the Gaussian baseline is coordinate-free. It is not. Same-norm Gaussian replacement preserves several pieces of structure:

- per-position norm scale from the adapted hidden stream;
- sequence length and attention mask;
- readout-specific adapter output scale;
- the Qwen residual dimensionality;
- the Qwen head, Qwen final norm, and copied early-layer normalization statistics;
- possible norm-position correlations created by the byte codec and adapter;
- the fact that every example is still scored against real Qwen token labels.

So the destroyed-input floor may still be partly coordinate-dependent. A hostile reviewer can say:

```text
Your prior floor is not pure prior. It is a Qwen-shaped, norm-conditioned, readout-conditioned input distribution routed through Qwen's own head.
```

The opposite attack is also possible: the floor may not be a language prior at all. It may be a broken-control artifact:

- Random layers plus copied Qwen head are an extremely weak baseline.
- Random layers have no calibrated relationship to the final norm/head.
- Copied Qwen layers may emit logits near a stable unigram-like attractor under many embedding-shaped inputs.
- RMSNorm can erase scale information and turn many random directions into tolerable head-facing activations.
- The Qwen LM head may carry high-frequency token bias independent of useful sequence processing.

In that reading, prior floor is too flattering. The safer term is:

```text
destroyed-input copied-core advantage over a weak random-core baseline
```

The coordinate-specific term is also overnamed. It is not yet coordinate-specific reasoning signal. It is:

```text
the NLL difference between adapted codec states and same-norm Gaussian noise under copied early Qwen layers and a Qwen head.
```

That difference can be produced by token identity reconstruction, local lexical smoothing, nearest-embedding manifold matching, or Qwen head compatibility. Those are coordinate-dependent, but they are not necessarily benchmark-relevant.

### Alternative Decompositions

A more hostile decomposition would split the total lift into at least six components:

| Component | Possible source | Current isolation status |
|---|---|---|
| Head/unigram prior | copied norm/head emits better token-frequency distribution than random core | not isolated |
| Norm/statistics prior | RMSNorm and residual scales stabilize logits for any same-norm input | not isolated |
| Qwen-family depth prior | early Qwen blocks are better coupled to Qwen head than random blocks | partly isolated by layers 14-17 only |
| Adapter lexical reconstruction | codec+adapter recovers token identity or nearest Qwen embedding | partly supported, not separated |
| Sequence/context computation | attention uses previous adapted positions | not isolated |
| Functional task discrimination | gold choices improve relative to wrong choices | untested |

The current floor decomposition collapses the first three into prior floor and the next three into coordinate-specific. That is too coarse for the causal story.

### Required Tests To Make The Decomposition Defensible

The prior-floor story becomes defensible only if the destroyed-input baseline is triangulated:

| Control | What it tests |
|---|---|
| zero input / constant input | head and norm unconditional floor |
| per-position norm-only random directions | norm-statistics floor |
| covariance-matched Gaussian from adapted states | marginal distribution floor |
| sequence-shuffled adapted states | token identity vs contextual order |
| label-shuffled adapted states | input manifold without label alignment |
| nearest Qwen embedding replacement | tokenizer-emulator story |
| head-only zero-layer path | final norm/head contribution |
| copied layers with reset norms | normalization-statistics floor |

The required reporting should be:

```text
coordinate lift above the maximum destroyed-input floor across predeclared destroyed baselines
```

not just above same-norm Gaussian.

### What Survived

The algebra survived. There is a real difference between copied calibrated inputs and copied Gaussian inputs. The 2.8-3.8 nat above-Gaussian gap is too large to dismiss as numerical noise.

### What Died

This sentence died:

```text
The prior floor is architecture + pretraining, not coordinate geometry.
```

Maybe. But the current evidence does not prove that separation. The floor may be partly coordinate-dependent, and the above-floor component may be lexical coordinate dependence rather than reasoning geometry.

### NARRATIVE ATTACK

**Strongest that's obvious dismissal:** You discovered that Qwen blocks plus a Qwen head behave less terribly than random blocks plus a Qwen head, even on random embedding-shaped inputs.

**Strongest that's trivial dismissal:** Subtracting one broken-control score from another broken-control score is not a theory of coordinate inheritance.

**What the result would need to be for the narrative to be unkillable:** Multiple destroyed-input baselines agree on a stable floor, the above-floor lift survives content/rare/candidate-margin slices, and internal disruptions show which Qwen subcircuits carry the signal.

### Attack On The Next Defense

The next defense will say the old disruption threshold was simply wrong because the floor is large. That may be true, but using a post-failure floor estimate to pass a new gate is exactly where motivated reasoning enters.

---
## Iteration 86: Is Redefining The Disruption Gate Goalpost Moving?

### Steelman

The old gate was:

```text
same-norm Gaussian disruption must retain <=20% of total copied-vs-random lift.
```

That gate assumes the copied core has little value when its input coordinates are destroyed. v1 falsified that assumption. If pretrained copied layers have a large unconditional floor, then a collapse-to-20% requirement punishes a real property of pretrained layers rather than testing coordinate dependence.

The new gate:

```text
coordinate-specific lift above destroyed-input floor >= 2.0 nats
```

is closer to the causal question:

```text
Does the calibrated byte-derived input add substantial value beyond copied-core prior behavior?
```

This is a better question than:

```text
Does every copied-core advantage vanish when the input is destroyed?
```

The old gate was a first-pass guardrail. A guardrail can be revised after it reveals an unanticipated confound. In that sense, the redefinition is not automatically dishonest.

### Attack

It is goalpost moving if it is used to rescue v1.

The sequence of events matters:

1. A precommitted gate required same-norm Gaussian retention <=20%.
2. v1 retained 33.5% on token-end and 47.3% on patch-boundary.
3. The run was correctly marked `FAIL_STAGE1_V1_PREFLIGHT`.
4. The supervisor then introduced a new metric under which both readouts pass.

A hostile reviewer does not need to claim bad faith. They only need to say:

```text
You failed the falsification test, then reinterpreted the failed control as a discovery and changed the pass criterion to one your failed run already satisfies.
```

That is a textbook optics problem.

The 2.0-nat threshold is also not calibrated. It was meaningful when comparing copied layers to random layers because it asked whether copied pretrained layers have a large effect over a broken baseline. Reusing 2.0 nats for copied calibrated vs copied Gaussian is plausible, but it is not prevalidated. The right threshold depends on:

- destroyed-input variance across seeds;
- sample size and confidence interval;
- whether the above-floor lift appears in candidate margins;
- whether it survives multiple destroyed baselines;
- whether generic pretrained controls show the same above-floor lift;
- whether the coordinate lift is concentrated in content positions.

The patch-boundary case is especially dangerous. The old gate failed at 47.3% retention, meaning nearly half of the copied-vs-random lift survived input destruction. The new metric says patch-boundary passes because 2.83 nats remain above Gaussian. Both facts are true. But for the moonshot, a patch-boundary path whose copied core is almost half prior floor is not clean byte-native evidence.

### Honest Governance

The only defensible handling is:

```text
v1 remains killed under the old gate.
The prior-floor interpretation becomes a v2 hypothesis.
The v2 gate must be predeclared before a fresh run or fresh held-out evaluation.
Old and new metrics must both be reported.
```

The new gate should be stricter than the supervisor's current form:

```text
coordinate lift above the strongest destroyed-input floor >= 2.0 nats
and functional-margin lift above destroyed/random controls >= predeclared threshold
and old disruption retention is reported as residual prior-floor size, not hidden.
```

If the new gate is used, it should produce a causal label, not a binary rescue:

| Pattern | Honest label |
|---|---|
| high total lift, high floor, low margins | surface compatibility |
| high above-floor NLL, positive margins, generic close | generic pretrained language geometry |
| high above-floor NLL, positive margins, generic loses | candidate coordinate-inheritance evidence |
| high floor, low above-floor lift | copied-core prior only |
| low floor, high above-floor lift | clean coordinate dependence |

### What Survived

The old 20% gate may be wrong in kind. It can falsely penalize legitimate pretrained priors. A revised metric is scientifically reasonable.

### What Died

This defense died:

```text
This is NOT goalpost moving.
```

Too strong. It is not inherently goalpost moving as a future v2 metric. It is goalpost moving if used to relabel the v1 smoke from killed to passed.

### NARRATIVE ATTACK

**Strongest that's obvious dismissal:** Your no-input control failed, so you decided no-input performance was a feature.

**Strongest that's trivial dismissal:** Post hoc threshold changes are common in prototype work. They are also why reviewers demand preregistration and fresh validation.

**What the result would need to be for the narrative to be unkillable:** The revised gate is frozen before a new evaluation, passes on fresh data with confidence intervals, reports old retention honestly, and is accompanied by positive functional margins.

### Attack On The Next Defense

The next defense will say the functional-margin shadow makes the revised gate honest. It can, but only if the margin test itself is not a weak, noisy, post hoc accuracy check.

---

## Iteration 87: What Makes A Functional-Margin Shadow Test Decisive?

### Steelman

The functional-margin shadow is the right next experiment because it tests the exact unknown Batch 12 identified:

```text
Does the coordinate-inheritance NLL signal contain task-discriminative function, or only lexical/token-manifold compatibility?
```

It is cheap because it can use the existing v1 adapter. It does not require training. It can compare the inherited path against random, shuffled, destroyed-input, rotated, and Qwen-middle controls on the same examples.

The decisive metric is not aggregate NLL. It is:

```text
margin = score(best_wrong_completion) - score(gold_completion)
```

where larger positive margin means the model prefers the gold completion over the strongest distractor. This directly tests whether the NLL advantage changes multiple-choice decisions.

### Attack

A weak margin shadow can mislead in both directions.

The current benchmark implementation is not yet the required test. It:

- defaults to validation, not explicitly train-safe shadow subsets;
- reports accuracy and bootstrap accuracy deltas, not gold-vs-best-wrong margins;
- uses token-space candidate scoring through Qwen token labels;
- includes Qwen-middle `generic_pretrained_layers`, but no true non-Qwen generic control;
- includes rotated and dim-permuted controls, but not same-norm Gaussian destroyed-input;
- does not include inverse recovery;
- does not include true-embedding truncated upper bound;
- does not include Wide7 as an evaluated baseline;
- does not report content-token or candidate-distinguishing-position slices.

So the decisive test requires implementation changes before execution.

### Minimum Decisive Design

The margin shadow should be predeclared like this:

| Requirement | Reason |
|---|---|
| train-safe HellaSwag, PIQA, ARC-Easy, ARC-Challenge subsets | avoids public benchmark claim while testing task shape |
| same examples for all variants | enables paired margins and paired accuracy |
| main, random, shuffled, Qwen-middle, Gaussian destroyed, dim-permuted, rotated, inverse-recovered, true-embedding | separates adapter-only, ordering, prior floor, coordinate disruption, and upper-bound stories |
| both token-end and patch-boundary readouts where feasible | tests byte-native readout, not only token endpoints |
| per-example gold-vs-best-wrong margin | avoids hiding cancellation under aggregate NLL |
| accuracy plus mean/median margin plus margin win rate | accuracy alone is too noisy |
| paired bootstrap CIs and McNemar-style paired flips | +1pp can be sampling noise |
| length-normalized and total-logprob variants | catches length normalization artifacts |
| content-token/candidate-distinguishing slices | tests whether gains hit the decision-bearing words |
| NLL-lift to margin-lift correlation | tests whether Stage 1 signal points at task decisions |

The key reporting table should look like:

| Metric | Main must beat |
|---|---|
| MCQ accuracy | random and strongest destroyed-input control |
| mean gold-vs-best-wrong margin | random and strongest destroyed-input control |
| paired margin win rate | >50% against controls with CI lower bound above 50% |
| Qwen preference agreement | random/destroyed/generic controls |
| content-token margin lift | function-token-only lift |
| patch-boundary margin lift | not collapse relative to token-end |

### How Much Is Enough?

The supervisor's kill condition says:

```text
kill if inherited path shows <+1pp MCQ accuracy over destroyed-input and random controls
```

That is a reasonable kill floor. It is not a promotion threshold.

My adversarial interpretation:

| Result | Interpretation |
|---|---|
| <+1pp over random and destroyed on all datasets | kill as surface compatibility |
| +1 to +2pp with wide CI or no margin lift | weak hint, not enough to proceed confidently |
| +2pp on one dataset only | likely dataset/scoring artifact |
| +2pp on all three families with positive paired margins | alive, but still fragile |
| +3 to +5pp with CI lower bound >0 and margin lift on content tokens | meaningful Stage 2 justification |
| >=+5pp across HellaSwag/PIQA/ARC and generic/destroyed controls | serious update |

For small subsets, +1pp is almost unmeasurable. On 256 examples, one point is roughly 2.5 examples. That is not evidence. If the test is small, the margin distribution matters more than accuracy.

### False Positives

The margin test can produce a false positive if:

- destroyed/random controls are too broken and main only beats bad baselines;
- Qwen head/tokenizer priors favor benchmark answer style;
- length normalization gives main a different bias than controls;
- context truncation removes information unevenly;
- choices share boilerplate and main wins on formatting tokens;
- Qwen pretraining contamination makes the Qwen head prefer gold-like endings;
- PIQA/ARC prompt format creates answer-position artifacts;
- HellaSwag activity labels leak obvious continuation priors;
- the same adapter was trained on data too close to benchmark text;
- only token-end works while patch-boundary fails;
- mean margins improve because a few easy examples dominate;
- accuracy improves but gold-vs-best-wrong margins remain tiny.

### False Negatives

The margin test can also produce a false negative if:

- the 4-layer early core is too shallow for benchmarks even though coordinate bootstrapping is useful downstream;
- the adapter was trained for embedding MSE, not ranking;
- token-end scoring underuses patch-boundary byte information;
- the subset is too small or too hard;
- continuation scoring is brittle for PIQA/ARC formatting;
- deeper layers or small finetuning are required to expose task function;
- current benchmark mode's token-space scorer is too far from the eventual byte-native model.

But these are not full rescues. If the current inherited path has huge NLL lift and zero margin shadow, the default interpretation should be:

```text
The measured Stage 1 signal is the wrong kind of signal for the moonshot.
```

### What Survived

Functional margins are the right cheap test. They are more decisive than another NLL gate because they measure candidate discrimination directly.

### What Died

This weak version died:

```text
If accuracy is barely above random/destroyed by +1pp, proceed.
```

No. +1pp is a kill threshold, not a success threshold. A barely positive result should be classified as weak evidence unless paired margins, confidence intervals, and slices agree.

### NARRATIVE ATTACK

**Strongest that's obvious dismissal:** You ran a tiny multiple-choice proxy and found a couple more lucky guesses than a deliberately broken control.

**Strongest that's trivial dismissal:** Token-space Qwen-head scoring can rank benchmark endings a little better than random layers. That is not a byte-native reasoning model.

**What the result would need to be for the narrative to be unkillable:** Main inherited beats random, destroyed, rotated, shuffled, Qwen-middle, and fair generic controls on paired margins across HellaSwag/PIQA/ARC; the gain is content-token concentrated; patch-boundary does not collapse; and NLL lift correlates with margin lift.

### Attack On The Next Defense

The next defense will say even a small +2pp is meaningful because the test is cheap and early. It is meaningful as a signal. It is not enough to relax the hostile gate chain.

---
## Iteration 88: If Margins Are Positive, What's The Strongest Remaining Attack?

Assume the shadow shows:

```text
main inherited path = +2pp MCQ accuracy over random and destroyed-input controls
```

### Steelman

That would be the first direct evidence that the coordinate signal has a functional shadow. It would damage the most dangerous Batch 12 attack:

```text
The inherited signal may be only lexical/token-manifold compatibility.
```

If the +2pp comes with positive gold-vs-best-wrong margin shifts, and if destroyed-input controls lose those margins, the result would show that the calibrated adapted states are not just lowering NLL uniformly across all choices. They are moving decisions.

That would justify a fresh revised Stage 1 or a carefully scoped Stage 2 prototype, provided the old v1 kill verdict remains recorded.

### Attack

The strongest remaining attack is:

```text
You found a weak Qwen-head continuation-ranking graft, not a democratized 121M byte model.
```

+2pp over random and destroyed controls is not close to the Vision target. The Vision bar is not beats broken controls. It is:

```text
beat SmolLM2-135M class baselines decisively enough to make people question scale assumptions.
```

A +2pp shadow still leaves major confounds alive:

| Remaining attack | Why it still matters |
|---|---|
| Bad controls | random/destroyed controls may be too weak; main must beat strong generic and tokenized controls |
| Qwen head prior | token-space scoring may exploit pretrained head biases without byte-native intelligence |
| Train-safe only | shadow subsets are not public benchmark evidence |
| Small effect | +2pp may be noise or prompt/scoring artifact |
| No compression | uncompressed copied Qwen blocks are not a <=121M active student |
| No byte-native proof | byte codec may be a tokenizer emulator |
| Shallow-depth story | 2-4 layer success still points at lexical/embedding processing |
| No non-Qwen generic control | may be generic pretrained language geometry |
| No Wide7 comparison | beating destroyed/random does not beat the existing byte baseline |
| No robustness | no typo/OOV/cross-tokenizer evidence |

The adversary will also ask whether the margin gain is broad or concentrated:

- Does HellaSwag move but PIQA/ARC stay flat?
- Do only short completions improve?
- Do margins improve only on examples where all choices are easy?
- Do wrong choices improve almost as much as gold choices?
- Do content-token margins move, or only punctuation/format tokens?
- Does patch-boundary reproduce the effect?
- Does inverse recovery restore the margin after disruption?

If the answer is weak, the +2pp result is a hint, not a proof.

### Required Interpretation If Positive

The correct label would be:

```text
FUNCTIONAL_SHADOW_PRESENT__NOT_YET_MOONSHOT_EVIDENCE
```

not:

```text
REASONING_GEOMETRY_TRANSPLANTED
```

The next gate should require:

1. replicate on a larger shadow subset;
2. report paired margin CIs;
3. include same-norm Gaussian destroyed-input, inverse recovery, and true-embedding upper bound;
4. compare against Wide7 or the current byte baseline where possible;
5. add a fair non-Qwen pretrained control before strong claims;
6. run internal disruptions to locate the signal;
7. proceed to Stage 2 only as a prototype benchmark, not public evidence.

### What Survived

Positive margins would keep coordinate inheritance alive. They would prove the NLL signal is not completely canceled across multiple-choice candidates.

### What Died

This stronger conclusion would still die:

```text
+2pp over destroyed/random controls proves the moonshot path.
```

No. It proves only that the current graft has some task-facing signal.

### NARRATIVE ATTACK

**Strongest that's obvious dismissal:** A copied Qwen fragment with a Qwen head can rank some benchmark endings slightly better than broken copies of itself.

**Strongest that's trivial dismissal:** Two points over bad controls is not beating Arjun. It is a smoke signal.

**What the result would need to be for the narrative to be unkillable:** The positive margin is large, paired, replicated, content-bearing, patch-boundary visible, destroyed by coordinate disruption, restored by inverse recovery, superior to generic pretrained controls, and eventually retained after <=121M compression.

### Attack On The Next Defense

The next defense will say small early margins justify continued work. They do, but only continued hostile testing. They do not justify softer language.

---

## Iteration 89: If Margins Are Flat, Is There Any Rescue Path?

Assume the shadow shows:

```text
main inherited path <+1pp over destroyed-input and random controls
```

### Steelman

There are narrow rescue interpretations.

The current v1 path uses only the existing adapter and a truncated 4-layer Qwen-shaped core. It was trained for embedding reconstruction and token NLL, not for candidate ranking. Early layers may provide lexical coordinate bootstrapping that only becomes useful after deeper reasoning layers, margin training, or compression-aware integration.

So flat margins do not prove that no coordinate idea can ever work. The Stage 1 NLL signal may still have value as:

- a codec diagnostic;
- a byte-to-token embedding bridge;
- a lexical front-end initializer;
- a way to warm-start a later byte model;
- evidence that copied early layers can process codec-derived states better than random layers.

One could propose a new hypothesis:

```text
Coordinate inheritance is a lexical coordinate bootstrap, not a standalone reasoning transplant.
```

That hypothesis might still matter if it helps a later training pipeline converge faster.

### Attack

For the moonshot, flat margins are close to fatal.

The whole reason for the functional-margin shadow is that this project has repeatedly seen surface metrics improve while reasoning benchmarks stay flat. If the current system has:

- 5.75 nats token-end copied-vs-random lift;
- 5.36 nats patch-boundary copied-vs-random lift;
- 2.83-3.82 nats above-Gaussian lift;
- Qwen middle-layer controls losing badly;
- exact inverse recovery;

and still cannot move gold-vs-best-wrong margins by even +1pp over destroyed/random controls, then the default conclusion is:

```text
The coordinate-inheritance signal is not task-discriminative in the form currently measured.
```

At that point, more Stage 1 NLL work becomes suspect. Tuning the adapter, changing depth, or choosing different disruptions risks becoming dashboard repair again.

### Kill Handling

If margins are flat, the verdict should be:

```text
PASS_SURFACE_COMPATIBILITY
FAIL_FUNCTIONAL_GEOMETRY
BLOCK_STAGE2_BENCHMARK_ESCALATION
DEMOTE_COORDINATE_INHERITANCE_TO_CODEC_DIAGNOSTIC_UNTIL_NEW_HYPOTHESIS
```

The direction should not proceed to public-style Stage 2 benchmarks on the theory that a larger run might rescue it. The cheap test was designed to answer exactly that.

### Remaining Rescue Paths

Only three rescue paths remain, and each must be framed as a new thesis:

| Path | What changes | Why it is not a simple continuation |
|---|---|---|
| Deeper functional inheritance | inherit deeper teacher components and evaluate margins first | current early-layer NLL story failed the function test |
| Margin-trained coordinate bridge | train adapter or small student directly on teacher preference margins | becomes supervised/ranking distillation, not pure coordinate inheritance |
| Lexical front-end utility | use inherited coordinates only as a byte lexical initializer | demotes the claim from reasoning geometry to front-end optimization |

The most moonshot-aligned pivot is not keep chasing NLL. It is:

```text
Move Eklavya toward task-discriminative, gap-driven teacher preference transfer where the primary training signal is functional margin, not token NLL.
```

That means the new core question becomes:

```text
Can a small byte model learn where teachers disagree and improve exactly those decision margins with minimal compute?
```

That is closer to the Vision than a beautiful NLL preflight with flat benchmarks.

### What Survived

Coordinate inheritance may survive as infrastructure or diagnostic. The byte codec and adapter can still be useful tools.

### What Died

This direction as the main moonshot path dies if margins are flat:

```text
early copied Qwen coordinate compatibility is the missing reasoning bridge
```

Flat margins would say it is not, at least not in the current form.

### NARRATIVE ATTACK

**Strongest that's obvious dismissal:** You built an excellent token-manifold adapter and then discovered it cannot choose better answers.

**Strongest that's trivial dismissal:** Lower NLL without decision-margin lift is exactly the failure pattern this project already knows.

**What the result would need to be for any rescue to be credible:** A new predeclared mechanism changes the functional-margin result, not just the NLL table. Until then, coordinate inheritance is a codec diagnostic, not a moonshot engine.

### Attack On The Next Defense

The next defense will say deeper layers or finetuning may reveal the effect. Maybe. But that is a new experiment class, not a license to ignore a flat margin shadow.

---
## Iteration 90: The Depth Curve And The Lexical Story

### Steelman

The depth curve is not purely bad for coordinate inheritance.

The results show that copied early Qwen layers are special:

| Depth | Token-end frozen gain | Patch-boundary frozen gain | Token-end copied advantage | Patch-boundary copied advantage |
|---:|---:|---:|---:|---:|
| 2 | 85.8% | 74.1% | 5.399 | 4.304 |
| 4 | 83.2% | 73.0% | 5.989 | 4.518 |
| 6 | 83.7% | 61.5% | 7.016 | 3.398 |
| 8 | 81.0% | 52.8% | 7.499 | 3.297 |

The shallow layers are exactly where one would expect the transition from embedding coordinates to early lexical features. A byte model that cannot enter a pretrained lexical coordinate system may never reach the higher-level circuits. In that view, the inherited shallow layers are not supposed to be complete reasoning. They are a coordinate bootstrap.

The fact that Qwen layers 14-17 lose badly also supports layer-order specificity. The adapter is not simply feeding any Qwen block successfully.

### Attack

The depth curve damages the reasoning geometry transplant narrative.

If the strongest frozen-core behavior is at 2 layers, and 4 layers is the best balance, the result points toward:

```text
lexical/embedding coordinate compatibility
```

not:

```text
deep reasoning geometry transfer
```

Patch-boundary is the byte-native bottleneck, and it gets worse with depth. At 6 and 8 layers, patch-boundary frozen-core gain fails badly. That matters more than token-end NLL improving with depth because the Vision's byte story cannot depend only on token-end anchors.

The hostile reading:

```text
The adapter reconstructs early Qwen embedding-like states. Shallow Qwen layers plus the Qwen head exploit that for token prediction. Deeper Qwen processing is not stable on byte-derived patch-boundary states, so the claimed reasoning geometry does not cross the bridge.
```

That is a much narrower result.

### Why Shallow Is Not A Minor Detail

Reasoning benchmark performance usually depends on multi-layer composition:

- retrieving relevant context;
- integrating event structure;
- distinguishing plausible distractors;
- tracking negation and affordances;
- resolving commonsense relations;
- suppressing high-frequency but wrong continuations.

The v1 depth curve does not show those functions. It shows early-layer token NLL compatibility. That can be valuable, but it is not yet the missing small-model intelligence mechanism.

The code also cautions against overreading the curve:

- depth-curve metrics are from the smoke eval split, 26 sequences per readout;
- frozen-core gain is a ratio against a 5-step finetune path, not a stable training curve;
- random baselines change with depth, so copied advantage is partly baseline-sensitive;
- patch-boundary labels can repeat within a token, which can distort what deeper means for the byte surface.

Those limitations do not save the reasoning narrative. They only say the depth curve is a warning, not a final theorem.

### Required Depth Tests

The next depth question should not be:

```text
Which depth gives best NLL?
```

It should be:

```text
Which depth gives best functional margin above destroyed/random/generic controls?
```

Predictions:

| Pattern | Interpretation |
|---|---|
| 2 layers best NLL and best margins | lexical front-end is doing almost everything |
| 2 layers best NLL, deeper best margins | NLL and reasoning signal separate |
| deeper layers improve token-end but hurt patch-boundary margins | token-end artifact, byte readout failure |
| all depths flat on margins | surface compatibility |
| 4-6 layers improve margins and internal disruptions localize attention/MLP function | possible functional coordinate inheritance |

### What Survived

The depth curve supports a real early-layer compatibility signal. It also supports keeping 4 layers as an engineering compromise for the current prototype.

### What Died

This phrase should stay dead:

```text
reasoning geometry transplant
```

The current depth evidence supports:

```text
early Qwen coordinate compatibility
```

or, more generously:

```text
lexical coordinate bootstrapping
```

### NARRATIVE ATTACK

**Strongest that's obvious dismissal:** Early transformer layers process embeddings and lexical features. You copied early layers and got an early-layer effect.

**Strongest that's trivial dismissal:** A byte-to-token embedding bridge is useful engineering, not proof that small models can inherit reasoning.

**What the result would need to be for the narrative to be unkillable:** Functional margins improve at depths where contextual circuits matter, patch-boundary remains strong, and internal disruptions show that attention/MLP computations beyond lexical manifold matching are causally responsible.

### Attack On The Next Defense

The next defense will say lexical coordinate bootstrapping may be the necessary first step. True, but then the story must stop claiming it has found the reasoning bridge until deeper functional evidence appears.

---

## Iteration 91: Internal Disruptions - Where Does The Coordinate Signal Live?

### Steelman

Internal disruptions are the right next diagnostic after the margin shadow because input destruction is too blunt. The current Gaussian disruption says:

```text
copied Qwen blocks have a floor even when adapted inputs are destroyed
```

It does not say which internal components create that floor or the above-floor coordinate lift.

Resetting attention, resetting MLPs, and disrupting residual projections can separate causal stories:

- attention carries contextual composition;
- MLPs carry lexical/semantic feature maps and token-frequency transformations;
- normalization carries residual scale and distribution priors;
- residual projections carry coordinate-basis continuity across layers;
- the LM head carries token-frequency and embedding-unembedding coupling.

### Attack

Internal disruptions can become another dashboard if not designed as causal classification. Resetting a module and watching NLL worsen is not enough. Many resets make a transformer OOD. The question is not:

```text
Can we break the model?
```

The question is:

```text
Which component explains coordinate-specific functional margin above the destroyed-input floor?
```

The experiments must therefore report both:

```text
NLL retained fraction
functional-margin retained fraction
```

relative to:

```text
baseline copied calibrated
strongest destroyed-input floor
random/shuffled controls
```

### Discriminating Experiment Set

Use the same examples and same saved adapter. For each variant, evaluate token-end and patch-boundary NLL plus margin shadow.

| Variant | Design | What it reveals |
|---|---|---|
| copied baseline | copied layers 0-3 | reference signal |
| same-norm Gaussian input | current destroyed-input floor | input-independent copied-core floor |
| head-only / zero-layer | adapter into copied norm/head without transformer blocks if implementable | final head/unigram floor |
| reset attention only | reinitialize Q/K/V/O projections, keep MLPs/norms | whether contextual attention is needed |
| reset QK only | break attention routing, keep value/output maps | whether attention pattern selection matters |
| reset V/O only | keep attention weights, break transported values | whether attention content transport matters |
| reset MLP only | reinitialize gate/up/down projections, keep attention/norms | whether MLP lexical feature maps carry the lift |
| reset gate/up only | break feature detection | whether nonlinear feature basis matters |
| reset down only | break feature write-back into residual stream | whether residual write coordinates matter |
| reset norms only | randomize RMSNorm weights or replace with neutral weights | whether norm statistics create the floor |
| inter-layer residual permutation | permute hidden dimensions between layers without conjugating weights | whether layer-to-layer coordinate continuity matters |
| conjugated residual permutation | apply mathematically matched permutations to adjacent weights where possible | sanity check that disruption is basis, not numerical damage |
| per-position sequence shuffle | keep adapted state distribution, break order/context | lexical token identity vs contextual computation |
| nearest-embedding replacement | replace adapter outputs with nearest Qwen token embeddings | tokenizer-emulator vs continuous coordinate signal |
| covariance-matched Gaussian | match adapted output covariance, not just norm | marginal distribution floor |

### Pattern Interpretation

| Observed pattern | Causal story |
|---|---|
| attention reset barely hurts NLL or margins | signal is mostly MLP/head lexical prior, not contextual reasoning |
| attention reset preserves NLL but kills margins | NLL is lexical, margins need contextual attention |
| MLP reset kills NLL and margins | MLP feature maps carry most coordinate signal |
| MLP reset preserves margins but hurts NLL | NLL was lexical smoothing, margins use attention/context |
| norms reset removes Gaussian floor | prior floor is largely normalization/statistics |
| head-only retains most floor | floor is mostly final norm/head unigram or embedding bias |
| residual permutation kills above-floor lift | coordinate basis continuity across layers is causal |
| conjugated permutation recovers | disruption targeted basis rather than simply damaging weights |
| nearest-embedding replacement matches main | codec+adapter is mostly a tokenizer emulator |
| main beats nearest-embedding on margins | continuous byte-derived coordinates add information beyond token identity |
| sequence shuffle keeps NLL but kills margins | local lexical identity drives NLL; order/context drives task function |
| sequence shuffle keeps margins | benchmark signal may be answer-style prior or leakage |

### Most Discriminating Causal Classifier

The cleanest classification table would use retained fractions:

```text
component_retention =
  (metric_variant - metric_floor)
  /
  (metric_copied - metric_floor)
```

where `metric` is both NLL improvement and gold-vs-best-wrong margin improvement. This avoids mixing prior floor with coordinate-specific lift.

Then classify:

| Class | Necessary pattern |
|---|---|
| head/norm prior | head-only or norm-preserving destroyed inputs retain most floor, but margins flat |
| lexical MLP compatibility | MLP reset kills NLL; attention reset does not; margins weak |
| contextual attention function | attention reset kills margins more than NLL |
| residual coordinate inheritance | inter-layer basis disruption kills both NLL and margins; conjugated inverse recovers |
| tokenizer emulator | nearest-embedding replacement matches main |
| functional coordinate signal | baseline margins exceed destroyed/random/generic; internal resets produce predicted selective collapses |

### Precommit Rules

Internal disruptions should be predeclared with:

1. at least 3 random reset seeds per reset type;
2. paired examples across all variants;
3. both readouts;
4. NLL and margin retained fractions;
5. confidence intervals;
6. a could-have-won sanity control, such as inverse/conjugated recovery where possible;
7. no narrative promotion from NLL-only internal disruption.

### What Survived

Internal disruptions are valuable and should be done if the margin shadow is positive or ambiguous. They can turn there is a floor into a causal map of where the signal lives.

### What Died

This shortcut died:

```text
If attention reset or MLP reset hurts NLL, we have localized reasoning geometry.
```

No. Hurting NLL localizes fragility. It localizes reasoning only if functional margins move in the predicted way.

### NARRATIVE ATTACK

**Strongest that's obvious dismissal:** Of course resetting transformer submodules breaks a transformer. That does not tell you what made the benchmark decision.

**Strongest that's trivial dismissal:** Attention/MLP ablations are standard diagnostics. They become meaningful only when tied to a causal story and a functional metric.

**What the result would need to be for the narrative to be unkillable:** Internal disruptions selectively destroy the coordinate-specific margin signal, inverse/basis sanity controls recover it, and the surviving pattern rules out head prior, norm prior, tokenizer emulation, and generic pretrained compatibility.

---

## Batch 13 Final Verdict

The supervisor's floor decomposition is useful, but not yet causally clean. It should be treated as:

```text
destroyed-input copied-core floor + above-floor input-sensitive NLL lift
```

not as proven separation between prior and coordinate geometry.

The gate redefinition is scientifically plausible but procedurally dangerous. The old v1 result remains killed. The new coordinate-specific gate can only be a v2 predeclared metric, and it must be paired with functional margins.

The functional-margin shadow is the right next experiment, but the pass standard must be stricter than a barely positive accuracy delta. A +1pp threshold is a kill floor, not evidence of a moonshot. The decisive question is whether inherited coordinates move gold-vs-best-wrong margins across HellaSwag/PIQA/ARC in a way destroyed, random, shuffled, rotated, Qwen-middle, and eventually fair non-Qwen controls cannot.

Hostile final statement:

```text
If the margin shadow is flat, coordinate inheritance becomes a codec diagnostic.
If the margin shadow is barely positive, coordinate inheritance remains alive but weak.
If the margin shadow is large, paired, content-bearing, patch-boundary-visible, and disruption-sensitive, then the direction earns the next hostile stage.
Nothing in v1 yet earns the phrase reasoning geometry transplant.
```
