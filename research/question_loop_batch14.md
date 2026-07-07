# QUESTION LOOP - Batch 14: Attack Margin Shadow Results + Direction Endgame

Date: 2026-07-07

Iterations: 92-98

## Grounding

I read the requested context in order:

1. `research/VISION.md`
2. `research/dual_loop_supervisor_checkin_8.md`
3. `research/question_loop_batch13.md`
4. `research/work_loop_batch9.md`
5. `tmp_coordinate_inheritance_v2/dry_margin/functional_margin_shadow.json`
6. `code/coordinate_inheritance.py`

No GPU runs, training runs, benchmark runs, or experiments were performed for this batch. This is analysis only.

## Binding State Entering Batch 14

W-Loop B10 is running the 100-example functional-margin shadow on HellaSwag, PIQA, and ARC-Easy. Those results are not available here. The n=2 HellaSwag dry run is not evidence of effect size, but it is evidence of which failure modes are live.

The dry run showed:

| Variant | Accuracy | Mean margin: best wrong NLL - gold NLL |
|---|---:|---:|
| main inherited | 50% | +0.394 |
| gaussian destroyed input | 50% | +0.999 |
| random core | 0% | -0.495 |
| shuffled core | 0% | -0.770 |
| generic pretrained core | 50% | +0.103 |
| true embedding truncated Qwen | 50% | +0.084 |
| inverse recovered rotation | 50% | +0.395 |
| full Qwen teacher | 0% | -0.940 |

The dry-run verdict was correctly:

```text
FAIL_FUNCTIONAL_MARGIN_SHADOW
```

because n=2 cannot measure a +1pp threshold and because main inherited did not beat gaussian destroyed input.

The precommitted B10 verdict tokens are:

```text
PASS_FUNCTIONAL_MARGIN_SHADOW - inherited >= +1pp MCQ accuracy over destroyed-input AND random on >=2 of 3 benchmarks
FAIL_FUNCTIONAL_MARGIN_SHADOW - inherited < +1pp MCQ accuracy advantage
MARGINAL_FUNCTIONAL_MARGIN - inherited +1-2pp, ambiguous
```

The hostile interpretation must preserve Batch 13's stricter standard:

```text
+1pp is a kill-floor threshold, not moonshot evidence.
Positive accuracy without paired margin lift is weak.
Positive margins against broken controls do not prove byte-native reasoning.
Nothing yet earns "reasoning geometry transplant."
```

The implementation makes the result narrow:

- Functional-margin shadow is token-space candidate scoring through adapted codec states, copied Qwen layers, and the Qwen LM head.
- The benchmark readout defaults to `token_end`, not `patch_boundary`.
- Functional-margin mode forces `benchmark_split=train`, so it is a train-safe shadow, not public benchmark evidence.
- The margin is `best_wrong_nll_per_token - gold_nll_per_token`; positive means the model prefers the gold completion.
- `gaussian_destroyed_input` preserves per-position adapter-output norms while replacing directions with random same-norm noise.
- `inverse_recovered_rotation` is rotate-then-inverse, which is numerically an identity sanity control, not evidence that arbitrary rotations work.
- `generic_pretrained_core` is Qwen layers 14-17, not a non-Qwen generic pretrained model.

## Iteration 92: The n=2 Gaussian Anomaly

### Steelman

The benign first answer is simple: n=2 is noise. One HellaSwag item had a large positive margin and one had a negative margin. The gaussian destroyed-input variant could look stronger than main inherited by accident, especially under per-token length-normalized scoring where a single candidate-length pattern can dominate.

There is also a nonfatal technical interpretation. The gaussian destroyed-input control is not "nothing." It preserves:

- sequence length;
- attention mask;
- candidate tokenization;
- per-position adapter-output norm;
- Qwen hidden dimensionality;
- copied Qwen layers;
- copied final norm and Qwen LM head;
- the scoring format and answer-choice priors.

So a high gaussian margin does not mean random noise has intelligence. It means copied Qwen layers plus a Qwen head may have a strong norm-conditioned language prior. That was already the v1 prior-floor discovery.

A narrow benign story would be:

```text
The inherited coordinates lower next-token NLL, but same-norm Gaussian sometimes regularizes away an overconfident wrong candidate under a shallow 4-layer Qwen-head scorer. The dry-run anomaly is a scoring artifact until it survives paired confidence intervals.
```

That story remains possible if the 100-example run shows main inherited beating gaussian on paired accuracy and margins even though n=2 did not.

### Attack

If gaussian destroyed input stays higher than main inherited at n=100, the current thesis is in serious trouble.

The core claim is not merely:

```text
copied Qwen layers have a useful prior
```

The core claim is:

```text
byte-derived inherited coordinates add task-discriminative information beyond copied-core priors.
```

If same-norm random directions outperform the actual adapted directions, then the adapted coordinate directions are not carrying the decisive benchmark signal. Worse, they may be carrying anti-signal: structured but wrong lexical commitments that make distractors look more plausible.

The hostile conclusion would be:

```text
The adapter learned to produce Qwen-shaped vectors for NLL, but the directions it produces do not help candidate discrimination. The useful part is norm/statistics plus copied Qwen-head prior, not inherited geometry.
```

That is not a small caveat. It flips the causal story.

Under the supervisor's prior-floor decomposition, gaussian destroyed input was supposed to be the floor below main inherited. If gaussian becomes the ceiling or ties the ceiling, then the decomposition collapses as promotional evidence. There is no above-floor functional lift. There is only:

```text
pretrained Qwen scorer floor >= adapted inherited coordinate scorer
```

The adapter would still be useful as a norm generator, but that is far from "Intelligence = Geometry, not Scale." It is closer to:

```text
Qwen hidden-state norm statistics plus a copied Qwen head are enough to rank some benchmark endings.
```

### What This Would Mean For The Entire Thesis

If the pattern holds across HellaSwag, PIQA, and ARC-Easy with paired uncertainty:

| Pattern | Meaning |
|---|---|
| gaussian accuracy >= main accuracy on >=2/3 benchmarks | no functional coordinate advantage over destroyed directions |
| gaussian mean margin > main mean margin | actual adapted directions are worse than norm-only destroyed inputs |
| gaussian paired margin win rate >50% vs main | gaussian dominates per-example, not just by outliers |
| random/shuffled still lose | copied Qwen prior matters, but coordinate order/direction still unproven |
| generic also clusters with main | Qwen-family pretrained scorer, not early-layer inherited geometry |

The thesis should then be relabeled:

```text
SURFACE_COMPATIBILITY_WITH_PRETRAINED_QWEN_PRIOR
```

not:

```text
FUNCTIONAL_COORDINATE_INHERITANCE
```

The main direction should be blocked from Stage 2. The only honest continuation would be a new hypothesis, such as norm/statistics transfer or margin-trained adapter transfer, not the current coordinate-inheritance claim.

### Is There A Benign Interpretation If It Holds?

Only weak ones.

Benign interpretation 1:

```text
The adapter was trained for embedding reconstruction and token NLL, not functional margins.
```

That is true, but it demotes the current path. It says the NLL-trained adapter is the wrong object for the moonshot until a margin-trained bridge proves otherwise.

Benign interpretation 2:

```text
The shallow 4-layer inherited core cannot expose the useful coordinate signal.
```

Maybe. But then the current shallow-core preflight is not a decisive positive stage. Deeper inheritance would need to start with functional margins, not another NLL gate.

Benign interpretation 3:

```text
Gaussian preserves useful norm information, and norms are part of geometry.
```

This is mathematically defensible and narratively fatal. If the project retreats from directional coordinate inheritance to "norm geometry," it needs a new mechanism and new controls. Norms alone are unlikely to carry enough structured reasoning to beat SmolLM2.

### What Survived

If gaussian remains strong, a copied Qwen prior floor survives. The scoring system may still reveal useful facts about Qwen-head priors and byte-to-Qwen norm calibration.

### What Died

This dies if gaussian beats or ties main on the 100-example shadow:

```text
The inherited coordinate directions add functional benchmark discrimination.
```

### NARRATIVE ATTACK

**Strongest that's obvious dismissal:** Your destroyed-input control beat the thing you claimed was carrying geometry.

**Strongest that's trivial dismissal:** You discovered that Qwen blocks and a Qwen head can rank answers from norm-shaped noise.

**What the result would need to be for the narrative to be unkillable:** Main inherited must beat gaussian on paired margins and accuracy across multiple benchmarks, with lower confidence bounds above zero and no collapse on content-bearing slices.

### Attack On The Next Defense

The next defense will say "n=2 is meaningless." Correct for effect size. Wrong for threat modeling. The anomaly names the exact control that can kill the direction.

---

## Iteration 93: Inverse Recovery = Main And The Rotation Invariance Problem

### Steelman

At face value, inverse recovered rotation matching main inherited could be read positively:

```text
The system can tolerate a coordinate transform if the inverse transform restores the original gauge before Qwen layers consume it.
```

That is a useful sanity check. It says the implementation can apply a transform and recover the original path without large numerical drift. In the dry run, inverse recovered rotation had mean margin +0.395 versus main inherited +0.394, a delta of about -0.0015. That is exactly what an identity sanity control should do.

The generous interpretation is:

```text
The benchmark scoring path is deterministic enough that a could-have-won transform/recovery control returns the expected answer.
```

### Attack

The stronger prompt interpretation is wrong. `inverse_recovered_rotation` does not show that any rotation of inherited coordinates gives the same benchmark discrimination.

The code defines it as:

```python
def rotate_with_inverse(x):
    return (x @ rot) @ rot.t()
```

For an orthogonal matrix, this is approximately:

```text
x @ I = x
```

So inverse recovered rotation matching main is not evidence of rotation invariance. It is evidence that the transform was undone before the model saw the final inputs. A hostile reviewer will not let this be called a rotation-invariance result.

The actual rotation-invariance test would be one of:

```text
main inherited vs rotated no inverse
main inherited vs random orthogonal direction replacement
main inherited vs hidden-dimension permutation
main inherited vs a fully conjugated model-weight rotation
```

The functional-margin code currently includes inverse recovery but not rotated-no-inverse in benchmark mode. That means the dry-run margin artifact cannot answer whether directions matter. It can only answer whether the inverse sanity path behaves like main.

### What Would Prove Norms, Not Directions?

The norms-not-directions attack requires a different observed pattern:

| Test | Norm-only implication |
|---|---|
| same-norm Gaussian ~= main | directions are unnecessary once norms are preserved |
| rotated no-inverse ~= main | Qwen layers are insensitive to basis directions, unlikely under real coordinate use |
| dim permutation ~= main | feature identities are not causally important |
| constant-norm random directions ~= main | even per-position norm variation is unnecessary |
| per-position norm-only baseline predicts margins | margins are a scalar confidence artifact |
| nearest-token embedding ~= main | adapter is mostly a tokenizer emulator |

The dry run gives one alarming piece of that pattern: same-norm Gaussian has higher mean margin than main. It does not give a real no-inverse rotation margin.

### Direction Signal vs Norm Signal

If B10 returns:

```text
main ~= inverse recovered
main <= gaussian destroyed
main ~= generic pretrained
random/shuffled << main
```

then the best causal label is:

```text
QWEN_PRETRAINED_PRIOR_WITH_NORM_CONDITIONING
```

not:

```text
ROTATION-INVARIANT COORDINATE SIGNAL
```

True coordinate inheritance should be basis-sensitive in the model's native gauge. Qwen weights are not arbitrary rotation-invariant functions. Attention projections, MLP gates, RMSNorm weights, and the LM head all expect specific hidden directions. If arbitrary rotations did not hurt, that would be devastating, not liberating.

The only way rotation invariance becomes benign is under a fully conjugated basis transform of the model weights, where every affected weight is transformed consistently. That would prove mathematical gauge equivalence, not that directions are irrelevant.

### Required Interpretation Rule

Going forward:

```text
inverse recovered rotation is a sanity control, not a positive evidence control.
```

It can catch implementation bugs. It cannot support the thesis. If it fails, the pipeline is broken. If it passes, nothing important is proven.

### What Survived

The code's inverse recovery path appears numerically sane in the dry run. That is useful for debugging.

### What Died

This inference dies:

```text
inverse recovered rotation equals main, therefore the coordinate signal is rotation-invariant.
```

No. It is an identity map after recovery.

### NARRATIVE ATTACK

**Strongest that's obvious dismissal:** You rotated the vector and then unrotated it, then called the unchanged result a discovery.

**Strongest that's trivial dismissal:** Identity controls are basic plumbing checks. They do not prove geometry transfer.

**What the result would need to be for the narrative to be unkillable:** No-inverse coordinate disruptions must damage main, inverse or conjugated recovery must restore it, and same-norm direction destruction must not match or beat main on margins.

### Attack On The Next Defense

The next defense will say rotation recovery shows the machinery is well-behaved. Fine. But a well-behaved identity check does not answer whether the signal lives in norms or directions.

---

## Iteration 94: What If Margins Are Flat Across ALL Variants?

Assume the 100-example run shows:

```text
main, generic, gaussian, inverse, true-embedding ~= 50% accuracy
random, shuffled ~= 25% accuracy
```

### Steelman

There is still a positive result in that world. Random and shuffled cores losing while copied Qwen-family variants cluster above them says pretrained Qwen structure matters. The adapter plus copied Qwen layers are not equivalent to random layers. That is consistent with the v1 finding that copied early layers carry a large prior floor and that shuffled layer order can damage the path.

The most generous story is:

```text
The benchmark shadow detects a real pretrained-core effect, but the current sample and accuracy metric are too coarse to separate main inherited coordinates from other Qwen-derived controls.
```

If paired margins show main has a small but consistent edge even with equal aggregate accuracy, the direction remains alive as weak evidence.

### Attack

Flat Qwen-based margins would be a direct strike against coordinate inheritance.

The story would become:

```text
Any sufficiently Qwen-shaped pretrained scorer plus the adapter can recover roughly the same benchmark signal. The specific inherited early-layer coordinates do not matter.
```

That is not a moonshot. It is a family-prior story.

The key distinction is:

```text
Qwen prior beats random
```

versus:

```text
inherited coordinates beat the strongest Qwen prior control
```

Only the second supports coordinate geometry. If main merely joins the Qwen cluster, it is not the causal driver.

### Tightest Analysis To Distinguish The Stories

The analysis should be paired, per-example, and strongest-control based. Aggregate accuracy is too blunt.

#### 1. Define The Strongest Alternative Floor

For each benchmark and each example, compute:

```text
strongest_control_margin =
  max(
    gaussian_destroyed_input_margin,
    generic_pretrained_core_margin,
    shuffled_core_margin,
    random_core_margin
  )

coordinate_margin_residual =
  main_inherited_margin - strongest_control_margin
```

Do not let main claim victory over random if gaussian or generic already does the same thing.

Promotion requires:

```text
mean coordinate_margin_residual > 0
paired CI lower bound > 0
paired win rate > 50%
```

on at least 2 of 3 benchmarks. If this fails, label the result Qwen-prior compatibility.

#### 2. Build A Paired Flip Table

For every control:

| Flip class | Meaning |
|---|---|
| main correct, control wrong | possible coordinate contribution |
| main wrong, control correct | coordinate anti-signal |
| both correct same pred | shared Qwen prior or easy item |
| both wrong same pred | shared distractor bias |
| both wrong different pred | unstable scoring noise |

Main needs a positive paired flip balance against gaussian and generic, not just against random.

#### 3. Compare Margin Distributions, Not Just Means

Report:

- mean paired margin delta;
- median paired margin delta;
- paired sign-test win rate;
- 25th/75th percentile deltas;
- fraction of examples where main beats every control;
- fraction where gaussian beats main;
- fraction where generic beats main;
- outlier contribution to the mean.

If main's average advantage comes from a handful of huge easy wins while gaussian wins most examples, the coordinate story fails.

#### 4. Check Prediction Identity

If main, gaussian, and generic make the same top-1 predictions on most examples, then main is not adding independent decision structure. It is riding the same Qwen-head preference surface.

Critical table:

| Pair | Top-1 agreement | Full-ranking agreement | Interpretation |
|---|---:|---:|---|
| main vs gaussian | high | high | direction signal absent or weak |
| main vs generic | high | high | early-layer specificity absent |
| main vs true embedding | high | high | adapter emulates token embedding path |
| main vs Qwen teacher | high | high | Qwen-head prior/contamination risk |

High agreement with gaussian is more damaging than high agreement with full Qwen, because gaussian has no meaningful adapted directions.

#### 5. Slice By Causal Relevance

Main must beat controls on slices where reasoning should matter:

- hard examples where Qwen teacher is not confidently correct;
- examples with close gold-vs-best-wrong margins;
- long contexts;
- long answer choices;
- distractors with similar lexical overlap;
- PIQA affordance conflicts;
- ARC science relation questions;
- HellaSwag temporal/event-continuation items.

If main only improves easy or answer-style items, the signal is not reasoning geometry.

#### 6. Decompose Gold And Wrong NLL Separately

A positive margin can arise in two ways:

```text
gold NLL decreases
best-wrong NLL increases
```

or:

```text
both gold and wrong improve, but gold improves slightly more
```

Coordinate inheritance should preferentially help gold over plausible distractors. If all candidates improve together, the adapter is improving fluency, not discrimination.

#### 7. Fit A Variant Effect Model

Use a simple item-paired model:

```text
margin ~ variant + benchmark + choice_length + item_random_effect
```

The important coefficient is:

```text
main_inherited - strongest_Qwen_control
```

If that coefficient is not positive with uncertainty away from zero, the coordinate claim fails.

#### 8. Apply A Control-Dominance Rule

Predeclare:

```text
If gaussian_destroyed_input or generic_pretrained_core matches or beats main on paired margin in >=2/3 benchmarks, the result is not coordinate-inheritance evidence, regardless of random/shuffled failure.
```

This prevents the loop from celebrating a weak baseline win while ignoring a stronger hostile control.

### Possible Outcomes

| Pattern | Verdict |
|---|---|
| main beats random/shuffled only | pretrained Qwen prior, not coordinate evidence |
| main beats gaussian but not generic | early-layer specificity unproven |
| main beats generic but not gaussian | directions unproven, norm prior too strong |
| main beats both gaussian and generic on paired margins | coordinate story alive |
| true embedding does not beat main | scorer path is suspect or sample too noisy |
| full Qwen teacher is weak on the same subset | shadow subset/scoring format may be pathological |

### What Survived

Flat Qwen-variant clustering would preserve the finding that pretrained Qwen-shaped computation is better than random or shuffled computation.

### What Died

This would die:

```text
the inherited coordinates are the cause of benchmark discrimination.
```

### NARRATIVE ATTACK

**Strongest that's obvious dismissal:** Every Qwen-flavored variant got the same answer quality, so the specific inherited coordinates did not matter.

**Strongest that's trivial dismissal:** Beating random layers is not a theory of intelligence. Random layers are a broken control.

**What the result would need to be for the narrative to be unkillable:** Main must beat the strongest Qwen-family and destroyed-input controls on paired margins, not just exceed random and shuffled baselines.

### Attack On The Next Defense

The next defense will point to random/shuffled collapse. That proves some pretrained structure matters. It does not prove this pretrained structure is inherited coordinate geometry.

---

## Iteration 95: The Endgame Probability After Margin Shadow

### Steelman

A clean margin-shadow pass would be a real update. It would be the first evidence that the Stage 1 NLL signal has a task-facing shadow. If main inherited beats random and gaussian on at least 2 of 3 benchmarks, the direction deserves one more hostile stage.

The strongest pass pattern would be:

```text
main beats random, shuffled, gaussian, and generic on paired margins;
the advantage appears on HellaSwag, PIQA, and ARC-Easy;
the gain is not driven by length or answer-position artifacts;
true embedding remains a meaningful upper bound;
random/shuffled stay weak;
gaussian no longer matches main.
```

That would justify a Stage 2 prototype, still with restricted language:

```text
FUNCTIONAL_SHADOW_PRESENT__NOT_YET_MOONSHOT_EVIDENCE
```

### Attack

Even a precommitted PASS does not make the endgame likely.

The Vision target is not:

```text
beat destroyed-input and random controls by +1pp
```

The Vision target is:

```text
beat SmolLM2-135M class baselines with a byte-native 121M model and far less compute.
```

Between a margin shadow and that target stand several failure walls:

- token-space Qwen-head scorer must become byte-native model quality;
- uncompressed copied-Qwen fragments must become a 121M active student;
- train-safe shadow gains must survive public validation/test distributions;
- early-layer lexical effects must become reasoning benchmark gains;
- Qwen-family controls must be beaten by fair non-Qwen controls;
- patch-boundary behavior must not collapse;
- the approach must beat strong small-model baselines, not only Wide7 or broken controls.

### Probability If Margins PASS

The probability depends on pass quality.

| Margin-shadow outcome | Probability of eventually beating SmolLM2-level benchmarks | Interpretation |
|---|---:|---|
| Bare PASS: +1pp on 2/3, wide CIs, weak margins | 5-10% | alive but fragile |
| Solid PASS: +2-4pp on 2/3, positive paired margins, gaussian/generic lose | 10-18% | worth Stage 2 |
| Strong PASS: +4pp or more on all 3, lower CIs >0, content slices positive | 18-30% | serious update |
| Spectacular PASS: large gains, patch-boundary visible, internal disruptions causal | 30-40% | still not guaranteed due to compression/byte-native wall |

Those numbers are not probabilities of the current B10 pass. They are conditional probabilities after observing those pass qualities.

If the only pass is the literal precommitted +1pp threshold, the honest number is closer to 5-10% than to 25%. +1pp is a minimal survival signal.

### Probability If Margins FAIL

If B10 fails by showing <+1pp main advantage over gaussian and random on all 3 benchmarks:

```text
Probability that current early-layer coordinate inheritance beats SmolLM2-level benchmarks: 0-3%.
```

Not exactly zero, because a radically different future method could reuse pieces of the infrastructure. But for the current direction as the primary moonshot, it is effectively dead.

The viable remainder would be:

```text
coordinate inheritance as codec diagnostic or norm/statistics transfer substrate
```

not:

```text
coordinate inheritance as the main reasoning bridge.
```

### Cost-Benefit Of Continuing vs Pivoting Right Now

Since W-Loop B10 is already running and cheap relative to training, the correct immediate action is not to pivot before reading its results. Let the decisive test finish.

But the continuation budget should be precommitted now:

| B10 result | Action |
|---|---|
| FAIL on all 3 | stop coordinate inheritance as primary direction; pivot |
| gaussian/generic >= main on >=2/3 | stop as coordinate evidence, even if random loses |
| MARGINAL +1-2pp | allow at most one bounded replication/control batch, no Stage 2 promotion |
| solid PASS | proceed to Stage 2 prototype with hostile labels |
| strong PASS | proceed to Stage 2 plus internal causal disruptions |

The cost of continuing after a fail is not just GPU time. It is attention capture. The project has already spent v0, v1, v2 design energy on a signal that may be a Qwen-head prior. Continuing after the functional gate fails would risk optimizing dashboards instead of searching for the breakthrough.

### Honest Endgame Verdict Before Results Arrive

Before B10 results, the rational stance is:

```text
wait for the running margin shadow;
prepare the pivot;
do not invest in new coordinate-inheritance repairs unless main beats gaussian/generic on paired functional margins.
```

### What Survived

If margins pass cleanly, the direction earns one more hostile stage. It does not earn public claims.

### What Died

If margins fail flatly, this dies:

```text
more Stage 1 NLL repair can rescue the moonshot.
```

### NARRATIVE ATTACK

**Strongest that's obvious dismissal:** Your endgame is beating SmolLM2, but your current pass token only asks to beat broken controls by one point.

**Strongest that's trivial dismissal:** A proxy pass is not a benchmark result, and a benchmark result is not a compressed byte-native model.

**What the result would need to be for the narrative to be unkillable:** Large, paired, replicated functional margins must survive strong controls and then convert into public benchmark gains in a <=121M byte-native student.

### Attack On The Next Defense

The next defense will say a weak pass keeps hope alive. Hope is not the invariant. Paradigm shift is the invariant.

---

## Iteration 96: Pivot Targets If Coordinate Inheritance Dies

If B10 kills coordinate inheritance, the pivot should preserve the manifesto:

```text
Intelligence = Geometry, not Scale.
```

But it should stop copying hidden-coordinate gauges and stop using NLL as the primary proof. The next directions should start from functional margins, disagreement, and causal error geometry.

### Pivot 1: Functional Margin Distillation

| Field | Proposal |
|---|---|
| Experiment | Train the byte student on teacher pairwise margins: gold vs strongest wrong, teacher-preferred vs student-preferred, and near-miss distractor pairs. |
| Hypothesis | The transferable geometry is not hidden-state coordinate basis; it is decision-boundary geometry between plausible alternatives. |
| Success criterion | With the same compute budget, margin distillation beats uniform token KD by >=+5pp on train-safe HellaSwag/PIQA/ARC shadow and >=+3pp on held-out validation, with paired CIs above zero. |
| One-sentence gossip story | "Instead of copying a teacher's brain coordinates, Sutra learned the shape of the teacher's hardest decisions." |

Why it avoids the current trap:

```text
The primary metric is functional candidate discrimination from day one, not token NLL.
```

### Pivot 2: Disagreement Geometry Router

| Field | Proposal |
|---|---|
| Experiment | Build a multi-teacher router that trains only on examples where teachers disagree and the student is wrong or uncertain. Route lessons by disagreement type: commonsense, physical affordance, science fact, temporal continuation, lexical trap. |
| Hypothesis | Intelligence lives in the geometry of disagreement and correction, not in teacher consensus or hidden-state imitation. |
| Success criterion | At fixed token/update budget, routed disagreement training beats uniform KD and single-teacher KD by >=+5pp on held-out MCQ margins and improves calibration on teacher-disagreement slices. |
| One-sentence gossip story | "Sutra got smarter by studying only the fights between its teachers." |

Why it avoids the current trap:

```text
The control is not random layers. The control is uniform distillation under the same compute.
```

### Pivot 3: Low-Rank Decision Subspace Transfer

| Field | Proposal |
|---|---|
| Experiment | For each benchmark-style item, collect teacher logit differences and gradients for gold-vs-distractor pairs; learn a low-rank subspace of decision directions that the student must align to. |
| Hypothesis | Cross-teacher intelligence is a low-dimensional functional subspace, not a full hidden-coordinate basis. |
| Success criterion | A rank-constrained student adapter improves paired margins over logit KD by >=+3pp and transfers to held-out tasks not used to fit the subspace. |
| One-sentence gossip story | "The teachers disagreed in millions of dimensions, but the useful reasoning directions fit in a tiny compass." |

Why it avoids the current trap:

```text
It transfers directions of decisions, not raw Qwen hidden directions.
```

### Pivot 4: Counterfactual Minimal-Pair Curriculum

| Field | Proposal |
|---|---|
| Experiment | Build or mine minimal pairs where a small semantic change flips the answer: tool affordance, negation, temporal order, entity role, physical constraint. Train the byte model to preserve the invariant and flip only the causal feature. |
| Hypothesis | Reasoning geometry is contrastive: the model becomes intelligent by learning which small changes should and should not move the answer. |
| Success criterion | Student improves >=+10pp on held-out minimal-pair tests and >=+3pp on standard MCQ benchmarks over a token-KD baseline. |
| One-sentence gossip story | "Sutra learned common sense by watching one word change the world." |

Why it avoids the current trap:

```text
It tests causal sensitivity directly instead of inferring it from NLL.
```

### Pivot 5: Error Atlas And Surgical Skill Patches

| Field | Proposal |
|---|---|
| Experiment | Cluster student failures by margin geometry, teacher disagreement, and counterfactual sensitivity; train small targeted patches or curricula for each cluster. |
| Hypothesis | Improvability comes from mapping the geometry of failures, then repairing local regions without retraining everything. |
| Success criterion | Each patch improves its target failure cluster by >=+15pp while regressing untargeted clusters by <=1pp, and cumulative patches beat uniform extra training per compute. |
| One-sentence gossip story | "Instead of making the model bigger, Sutra learned to operate on its own mistakes." |

Why it avoids the current trap:

```text
It serves the Vision's improvability outcome and makes failures surgically addressable.
```

### Pivot 6: Byte-Native Teacher Debate Compression

| Field | Proposal |
|---|---|
| Experiment | For each hard item, ask multiple teachers for compressed rationales or decision features, then train the byte model on the minimal feature set needed to reproduce the correct margin. |
| Hypothesis | Teacher reasoning can be compressed as sparse decision evidence rather than dense logits or hidden states. |
| Success criterion | Sparse rationale-feature training beats token KD at the same budget on hard-item margins, with ablations showing the feature set is necessary. |
| One-sentence gossip story | "The teachers did not give Sutra answers; they gave it the few facts that made the answer inevitable." |

Why it avoids the current trap:

```text
The story is transparent and causal enough for hostile reviewers to inspect.
```

### Pivot Priority

If coordinate inheritance dies, the strongest immediate pivot is:

```text
Functional Margin Distillation + Disagreement Geometry Router
```

Those are closest to Eklavya's original thesis:

```text
teachers are instruments, not masters;
the student learns from their disagreements, not their consensus.
```

They also align with the functional-margin lesson from v2:

```text
make the decision margin the training target, not an after-the-fact shadow.
```

### NARRATIVE ATTACK

**Strongest that's obvious dismissal:** If coordinate inheritance dies, do not repackage it as "geometry" and keep drilling the same hole.

**Strongest that's trivial dismissal:** Hidden-state copying is not the only geometric theory. It may be the least robust one.

**What the pivot would need to be for the narrative to be unkillable:** The next experiment must start with functional margins, fair baselines, held-out validation, and a story a reviewer can understand without trusting a hidden-coordinate dashboard.

### Attack On The Next Defense

The next defense will say coordinate inheritance infrastructure can be reused. Fine. Reuse infrastructure. Do not reuse the claim.

---

## Iteration 97: The Sunk Cost Audit

### Steelman

The project has not been irrational so far.

v0 exposed failure modes. v1 repaired patch-boundary frozen-core gain and discovered the prior floor. B13 tightened the interpretation. B10's functional-margin shadow is exactly the right cheap test before expensive escalation.

That is legitimate scientific iteration:

```text
new failure -> sharper causal question -> cheaper decisive test
```

The sunk-cost fallacy begins only when failure no longer sharpens the hypothesis and instead generates excuses to keep the same claim alive.

### Attack

The danger is already visible.

The sequence is:

1. v0 killed.
2. v1 killed under the precommitted disruption gate.
3. v1 failure was reinterpreted as prior-floor discovery.
4. v2 margin shadow was introduced to decide whether the signal is functional.
5. The n=2 dry run showed the most hostile control, gaussian destroyed input, with higher margin than main.

That does not prove the direction is dead. But it means the next failure must be allowed to kill it. If every failed gate becomes "new insight, one more repair," the loop becomes a sunk-cost machine.

### Bright-Line Abandonment Rule

Permanently abandon coordinate inheritance as the primary moonshot direction if any of the following occurs in the 100-example B10 result:

```text
1. FAIL_FUNCTIONAL_MARGIN_SHADOW on all 3 benchmarks:
   main inherited has <+1pp accuracy advantage over gaussian destroyed input and random controls.

2. Gaussian dominance:
   gaussian destroyed input matches or beats main inherited on paired margin in >=2/3 benchmarks.

3. Generic dominance:
   generic pretrained Qwen middle-layer core matches or beats main inherited on paired margin in >=2/3 benchmarks.

4. No margin residual:
   main's margin over the strongest control has CI lower bound <=0 on all benchmarks.

5. Accuracy-only pass:
   main gets a nominal +1pp accuracy pass but mean/median paired margins do not improve against gaussian and generic.
```

Abandon here means:

```text
No Stage 2 benchmark escalation.
No more NLL-only repairs.
No public narrative around coordinate inheritance.
Demote the code to diagnostics or reusable infrastructure.
Pivot the main research line.
```

### Rule For A Marginal Result

If B10 returns `MARGINAL_FUNCTIONAL_MARGIN`, allow exactly one bounded follow-up:

```text
one replication/control batch;
same predeclared metrics;
must include paired margins and strongest-control residual;
no architecture expansion;
no deeper inheritance unless the marginal result becomes solid.
```

If the follow-up does not produce:

```text
>=+3pp over gaussian and generic on >=2/3 benchmarks
and paired margin CI lower bound >0
```

then abandon.

### Rule For A Pass

If B10 returns a solid or strong pass, coordinate inheritance is not abandoned, but it remains on probation.

Next-stage kill rules:

| Stage | Kill condition |
|---|---|
| Stage 2 prototype | does not beat Wide7 by a meaningful margin on the same benchmark family |
| Strong-control replication | non-Qwen generic or tokenized sibling control matches main |
| Patch-boundary check | token-end margin exists but patch-boundary collapses |
| Internal disruptions | resets do not localize functional margin signal beyond head/norm priors |
| Compression/student step | uncompressed Qwen-head gain does not survive into <=121M byte-native student |

### Why This Is Not Too Harsh

The Vision says:

```text
paradigm shift or failure
```

The project does not need to prove coordinate inheritance is impossible. It only needs to decide whether coordinate inheritance deserves to be the main path. If it cannot beat destroyed-input and generic controls on a cheap functional shadow, it does not.

### What Survived

The past loops were legitimate because each failure exposed a sharper test. The process has not yet become sunk cost.

### What Died

This dies now:

```text
Coordinate inheritance gets unlimited repairs because it once produced large NLL gains.
```

Large NLL gains already failed to settle the question.

### NARRATIVE ATTACK

**Strongest that's obvious dismissal:** You spent hundreds of iterations proving the dashboard can be repaired, not that the model is smarter.

**Strongest that's trivial dismissal:** A failed preflight that keeps spawning new preflights is not moonshot discipline. It is attachment.

**What the result would need to be for persistence to be rational:** Each new loop must increase functional evidence against the strongest controls, not merely rename the failure mode.

### Attack On The Next Defense

The next defense will say "but we learned something." Learning something is not the standard. Learning the wrong direction is dead is a successful outcome only if you stop.

---

## Iteration 98: Meta-Question - Is The Dual-Loop Working?

### Steelman

The dual-loop is working in one important sense: it prevented premature promotion.

Evidence:

- v0 was killed instead of rationalized.
- v1 repaired one failure and then was killed by the disruption gate.
- The prior-floor discovery was separated from the moonshot claim.
- Batch 13 refused to let +1pp become success evidence.
- W-Loop B10 is running the right cheap functional test before Stage 2.
- The code now reports stronger controls, gaussian destroyed input, inverse sanity, true-embedding upper bound, and margin deltas.

That is real progress. A less adversarial process would likely have claimed:

```text
5 nats copied-vs-random advantage proves inherited reasoning geometry.
```

The dual-loop stopped that overclaim.

### Attack

The dual-loop may still be optimizing for not being fooled more than for finding the breakthrough.

Hostile process critique:

```text
You have built an impressive falsification bureaucracy around one idea, but the idea itself may be too narrow. The loops keep making better gates for coordinate inheritance instead of generating enough alternative moonshot candidates.
```

The process risks four pathologies.

#### Pathology 1: Gate Proliferation

Every failure creates a better gate. That is good until it becomes a substitute for invention. The question is not only:

```text
Can coordinate inheritance survive one more stricter test?
```

It is:

```text
Is coordinate inheritance still the best use of scarce research attention?
```

#### Pathology 2: False Precision

Verdict tokens are useful, but they can create a false sense of rigor:

```text
PASS_FUNCTIONAL_MARGIN_SHADOW
```

sounds decisive even when the pass threshold is +1pp on 100 examples. The token is only as strong as the metric behind it.

#### Pathology 3: Post-Failure Metric Migration

The prior-floor reinterpretation may be scientifically valid, but it happened after v1 failed. The dual-loop must treat every post-failure reinterpretation as a new hypothesis requiring fresh validation, not as a rescue of the failed run.

#### Pathology 4: Local Search Trap

The loops have spent enormous energy around Qwen hidden-state inheritance:

- adapter calibration;
- readout conditioning;
- layer copying;
- gaussian disruption;
- rotation sanity;
- benchmark shadow through Qwen head.

That may be too local. The manifesto says geometry, not scale. It does not say hidden-coordinate copying from Qwen.

### What A Hostile External Reviewer Would Say

A hostile but competent reviewer would likely say:

```text
The process is unusually honest for an independent lab, but the evidence is still internal, control-sensitive, and too Qwen-dependent. The loop is good at preventing obvious self-deception. It has not yet shown that it can convert falsification into a broader invention portfolio.
```

They would also say:

```text
Your strongest result so far is not a benchmark gain. It is that copied Qwen layers have a large prior floor and your controls were too weak. That is useful, but it is not the moonshot.
```

And:

```text
If the margin shadow fails and you keep going, the process has failed its own invariants.
```

### Is The Adversarial Stance Helping Or Hurting?

It is helping with truth. It may be hurting with search breadth.

The fix is not to become less adversarial. The fix is to make the adversarial loop attack a portfolio, not one cherished mechanism.

Proposed process rule:

```text
For every two falsification batches spent on the current mechanism, one batch must generate or sharpen a competing mechanism with a predeclared functional test.
```

Another rule:

```text
No mechanism gets a third repair cycle unless it has already produced functional evidence against its strongest controls.
```

Coordinate inheritance is now at that line.

### Process Changes

1. Maintain a live graveyard of killed hypotheses with the exact kill reason.
2. Separate "diagnostic insight" from "moonshot evidence" in every report.
3. Require every post-hoc reinterpretation to become a fresh predeclared test.
4. Use strongest-control residuals, not random-control wins, as the default metric.
5. Reserve a fixed fraction of loop effort for new mechanisms, especially functional-margin-first ideas.
6. Make the endgame metric visible in every batch: SmolLM2-level benchmarks in a <=121M byte-native model.
7. When a decisive gate fails, pivot without another repair batch unless the user explicitly reopens the direction as historical diagnostics.

### What Survived

The dual-loop is valuable. It has protected the project from premature claims and forced the current decisive test into existence.

### What Died

This dies if B10 fails and the loop keeps repairing:

```text
The dual-loop is hostile enough to kill its favorite idea.
```

The process only works if it can actually stop.

### NARRATIVE ATTACK

**Strongest that's obvious dismissal:** You built a machine for arguing with yourself, but it still circles the same pet mechanism.

**Strongest that's trivial dismissal:** Internal rigor is table stakes. A paradigm shift needs external functional wins.

**What the process would need to be for the narrative to be unkillable:** The loop must kill failed mechanisms quickly, generate competing hypotheses, and convert surviving ideas into held-out functional benchmark gains.

### Attack On The Next Defense

The next defense will say the dual-loop is working because it found the margin-shadow test. Yes. The final proof of the process is whether it obeys the margin-shadow result.

---

## Batch 14 Final Verdict

The dry run must not be overinterpreted, but its failure pattern is exactly the dangerous one:

```text
gaussian destroyed input had higher mean margin than main inherited;
inverse recovered rotation was identical to main because it is an identity recovery check;
random and shuffled were bad, but those are weak controls;
generic and true-embedding clustered near main on n=2 accuracy.
```

The decisive B10 interpretation should be:

| B10 pattern | Verdict |
|---|---|
| main fails to beat random and gaussian by +1pp on all 3 | `FAIL_FUNCTIONAL_MARGIN_SHADOW`; pivot |
| main beats random but not gaussian/generic | copied Qwen prior or norm floor; pivot as primary direction |
| main gets +1-2pp with weak paired margins | `MARGINAL_FUNCTIONAL_MARGIN`; one bounded replication only |
| main beats gaussian and generic on paired margins in >=2/3 | alive for Stage 2 prototype, not public evidence |
| main shows large, paired, content-bearing gains across all 3 | serious update, still must survive byte-native compression |

Hostile final statement:

```text
If gaussian or generic can do what main does, coordinate inheritance is not the moonshot.
If main only beats random and shuffled, the project has beaten bad controls.
If the margin shadow fails, stop repairing this direction and pivot to functional-margin-first geometry.
If the margin shadow passes weakly, continue only under probation.
Only a strong paired margin residual over the strongest controls earns the next hostile stage.
```

