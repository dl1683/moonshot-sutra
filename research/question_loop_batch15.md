# QUESTION LOOP - Batch 15: Attack the Pivot Before Implementation

Date: 2026-07-07

Iterations: 99-105

## Grounding

I read the requested local context in order:

1. `research/VISION.md`
2. `research/dual_loop_supervisor_checkin_9.md`
3. `research/question_loop_batch14.md`
4. `research/work_loop_batch10.md`
5. `tmp_coordinate_inheritance_v2/margin_shadow_smoke50/functional_margin_shadow.json`
6. `code/coordinate_inheritance.py`

No GPU runs, training runs, benchmark runs, or experiments were performed for this batch. This is analysis only.

I also checked primary prior-art sources for the novelty attack in Iteration 103:

- [Ranking Distillation: Learning Compact Ranking Models With High Performance for Recommender System](https://arxiv.org/abs/1809.07428)
- [Improving Neural Ranking via Lossless Knowledge Distillation](https://arxiv.org/abs/2109.15285)
- [SLiC-HF: Sequence Likelihood Calibration with Human Feedback](https://arxiv.org/abs/2305.10425)
- [RRHF: Rank Responses to Align Language Models with Human Feedback without tears](https://arxiv.org/abs/2304.05302)
- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)
- [Adaptive Multi-Teacher Multi-level Knowledge Distillation](https://arxiv.org/abs/2103.04062)
- [Multi-Level Optimal Transport for Universal Cross-Tokenizer Knowledge Distillation on Language Models](https://arxiv.org/abs/2412.14528)

## Binding State Entering Batch 15

Coordinate inheritance is permanently dead as the primary moonshot direction.

The decisive W-Loop B10 artifact says:

| Benchmark | Main inherited | Random core | Gaussian destroyed | Main - random | Main - Gaussian | Gate |
|---|---:|---:|---:|---:|---:|---|
| HellaSwag | 20.0% | 26.0% | 24.0% | -6.0pp | -4.0pp | FAIL |
| PIQA | 42.0% | 58.0% | 42.0% | -16.0pp | 0.0pp | FAIL |
| ARC-Easy | 22.0% | 26.0% | 16.0% | -4.0pp | +6.0pp | FAIL |

Precommitted verdict:

```text
FAIL_FUNCTIONAL_MARGIN_SHADOW
SURFACE_COMPATIBILITY_ONLY
```

The old adapter training objective in `code/coordinate_inheritance.py` was embedding reconstruction:

```text
loss = MSE(adapter(codec_hidden), teacher_embedding)
     + 0.10 * cosine_loss
     + 0.01 * norm_loss
```

The benchmark infrastructure already computes candidate continuation NLLs, top-1 predictions, gold-vs-best-wrong margins, teacher rankings, paired deltas, and the functional-margin verdict. But it does not implement byte-decoder training or byte BPB. The artifact explicitly limits the current benchmark path:

```text
Benchmark mode is token-space candidate scoring through byte-derived codec states.
It does not implement a byte decoder or byte BPB.
```

That matters. Functional Margin Distillation can avoid the exact NLL-to-function disconnect that killed coordinate inheritance, but it can still die at the student-compression boundary.

## Hostile Batch Verdict Up Front

Functional Margin Distillation is alive only as a cheap admission test. It is not yet a moonshot.

The pivot is materially different from coordinate inheritance only if the training target is the student's own functional decision margin in a byte-native or byte-facing path. If the project merely changes the adapter loss while still relying on Qwen heads, Qwen labels, and benchmark-format choices, a hostile reviewer will call it ordinary ranking distillation through a fragile scaffold.

Disagreement Geometry Router is not ready to be the second experiment until teacher-disagreement density is measured. If useful disagreement is sparse, correlated, or mostly prompt/tokenizer noise, the router is a story without a training set.

The next W-Loop should therefore not implement a large new system. It should run an admission battery:

```text
1. Functional Margin Distillation shadow smoke.
2. Same-budget token-KD / CE / label-only controls.
3. Teacher disagreement density audit.
4. Artifact controls for length, answer position, prompt template, and train/validation split.
```

If the pivot cannot beat ordinary baselines under these gates, it should be killed quickly or demoted to infrastructure.

---

## Iteration 99: Is Functional Margin Distillation Actually Different?

### Steelman

Functional Margin Distillation is different in the one way that matters most after B10: the primary training signal is candidate discrimination, not hidden-coordinate reconstruction.

Coordinate inheritance tried to make byte-derived states look like Qwen hidden coordinates, then asked later whether those coordinates helped answer multiple-choice questions. B10 showed that this sequencing was fatal. The adapter learned Qwen-shaped lexical/manifold compatibility, not task-discriminative function.

Functional Margin Distillation reverses the priority:

```text
Train the student so gold completions outrank plausible wrong completions.
Measure the same margin during training and evaluation.
Do not promote NLL lift unless it improves functional choice margins.
```

The margin object is also tokenizer-friendlier than hidden states. Every teacher can score the same textual context and candidate completions under its own tokenizer. The transferable object is not a token embedding vector. It is a relation:

```text
candidate A should be preferred to candidate B by this much under this context.
```

That is closer to the Vision's "geometry, not scale" thesis than coordinate copying was. The geometry is no longer a hidden gauge. It is a decision boundary over alternatives.

The existing infrastructure can support a cheap first version:

- `load_limited_benchmark` already produces benchmark-style context/choice records.
- `score_teacher_completion` already obtains teacher continuation scores.
- `score_completion_token_space` already obtains student/scaffold continuation scores.
- `build_choice_prediction_record` already computes gold-vs-best-wrong margin.
- `bootstrap_scalar_delta` already compares paired margins.

The strongest benign story is:

```text
Coordinate inheritance failed because it trained a representational proxy.
Functional Margin Distillation trains the behavior directly.
```

### Attack

The phrase "decision-boundary geometry" can hide a much more ordinary thing:

```text
supervised pairwise ranking loss on multiple-choice examples.
```

A hostile reviewer will say the pivot is not a new learning mechanism. It is the standard move after soft-target KD fails: train on preferences, margins, or rankings.

Even worse, Functional Margin Distillation may inherit the same deepest failure:

```text
The training target can look functional while still being surface-compatible only.
```

Failure modes:

| Failure mode | Hostile interpretation |
|---|---|
| MCQ-format overfit | The student learns benchmark answer style, option length, and lexical overlap, not reasoning. |
| Teacher-margin mimicry | The student copies Qwen's confidence quirks without gaining robust task skill. |
| Label dependency | If the gold answer comes from HellaSwag/PIQA/ARC labels, the method is ordinary supervised fine-tuning with a ranking loss. |
| Teacher error poison | Qwen teacher accuracy in the B10 smoke was only 48% on HellaSwag and 54% on ARC-Easy. A teacher-margin target can train confident wrong preferences. |
| Compression wall | A small byte-native student may not have enough capacity to realize the teacher boundary, even if a Qwen-headed scaffold can. |
| Non-generative collapse | A model can rank four choices better while becoming worse at open-ended language modeling. |
| Length/calibration artifact | Per-token or per-byte normalization can make shorter or tokenization-favored choices look better. |
| Hard-negative pathology | "Strongest wrong" may be strongest because of annotation artifacts, not because it reveals a causal reasoning boundary. |
| Benchmark contamination risk | Training on benchmark-style train examples can improve held-out slices without proving a general teacher-learning protocol. |
| Single-teacher lock-in | A Qwen-only margin objective contradicts the manifesto's multi-teacher claim unless it is explicitly labeled as the first baseline. |

The deepest attack is this:

```text
Functional margins are not automatically more causal than NLL.
They are only more task-facing.
```

If the pair is:

```text
Question: What is the best continuation?
Gold choice vs strongest wrong choice.
```

then the target may encode dataset annotation style rather than the latent structure that makes the answer correct. The model can improve a margin by increasing the wrong candidate NLL for superficial reasons. It does not have to learn the causal distinction.

The pivot is truly different only if the margin target is used to train the actual student path and if the gain survives controls that destroy shallow answer artifacts:

- train/validation separation;
- length-matched distractors;
- answer-position shuffles;
- prompt-template variations;
- same-budget token KD;
- same-budget label-only CE;
- teacher-wrong filtering;
- counterfactual minimal-pair probes.

Otherwise, it is the same pattern as before:

```text
large internal training signal -> exciting story -> no proof of robust downstream intelligence.
```

### Deepest Failure Case

The worst plausible outcome is not immediate failure. It is a misleading weak success:

```text
FMD improves train-safe HellaSwag/PIQA/ARC margins by +3pp,
but only against weak controls,
only in the Qwen-headed scoring scaffold,
only on examples where Qwen already has answer-style priors,
and not in a byte-native student.
```

That would recreate the old trap with a stronger-looking metric.

B10 killed "NLL implies function." B15 must not replace it with:

```text
margin improvement implies intelligence.
```

Margin improvement is evidence only after strongest controls fail.

### What Survives

Functional Margin Distillation survives as the right next cheap test because it directly targets the failure metric that killed coordinate inheritance.

### What Dies

This claim dies before implementation:

```text
Pairwise-margin training is automatically a new moonshot mechanism.
```

No. It is a conventional family of objectives until the byte-native, multi-teacher, data-efficient evidence says otherwise.

### NARRATIVE ATTACK

**Strongest that's obvious dismissal:** You stopped copying hidden states and started training on answer rankings. That is the obvious baseline, not a breakthrough.

**Strongest that's trivial dismissal:** Pairwise ranking losses are old. Calling them "decision-boundary geometry" does not make them new.

**What the result would need to be for the narrative to be unkillable:** A <=121M byte-native student must beat same-budget token KD and label-only controls on held-out functional margins, with artifact controls failing to explain the gain.

### Attack On The Next Defense

The next defense will say the old objective was embedding MSE and the new objective is margins. Correct. But "different objective" is not enough. The question is whether the new objective creates transferable intelligence rather than a better benchmark-specific scorer.

---

## Iteration 100: How Does Margin Distillation Actually Work?

### Steelman

Functional Margin Distillation can be made concrete without mystery.

For a context `x` and choices `c_1 ... c_n`, define teacher and student continuation losses:

```text
teacher_loss_i = NLL_T(c_i | x) normalized by bytes or tokens
student_loss_i = NLL_S(c_i | x) normalized by bytes or tokens
```

For labeled MCQ training, let `g` be the gold choice and let `w` be a hard wrong choice:

```text
w = argmin_{i != g} teacher_loss_i
```

or, for a student-correction curriculum:

```text
w = argmin_{i != g} student_loss_i
```

The student margin is:

```text
m_S(x, g, w) = student_loss_w - student_loss_g
```

Positive means the student prefers the gold completion.

The teacher margin is:

```text
m_T(x, g, w) = teacher_loss_w - teacher_loss_g
```

A simple first loss can combine ranking pressure, teacher-margin regression, and language-model anchoring:

```text
L_rank = softplus(gamma * (target_margin - m_S))
L_reg  = Huber(m_S - clip(alpha * m_T, -M, M))
L_lm   = ordinary next-token or byte-level CE on the same text

L_total = L_rank + lambda_reg * L_reg + lambda_lm * L_lm
```

A DPO-like variant is also possible:

```text
delta_S   = logp_S(g | x) - logp_S(w | x)
delta_ref = logp_ref(g | x) - logp_ref(w | x)

L_dpo_style = -log sigmoid(beta * (delta_S - delta_ref))
```

The student does not need a classification head. It can see each candidate as a continuation and learn by adjusting sequence probability. That keeps the task aligned with generative modeling:

```text
Input: bytes for context + candidate completion.
Output: sequence log probability / NLL over the candidate region.
Training target: gold completion should have lower NLL than the hard wrong completion.
```

There are two data modes.

Mode A: labeled benchmark-style MCQ.

```text
Use HellaSwag/PIQA/ARC train-safe examples.
Gold is the dataset label.
Teacher supplies margin strength and hard-negative selection.
```

Mode B: unlabeled text converted into contrastive choices.

```text
Positive: true next sentence/span from corpus.
Negatives: retrieved near-miss continuation, teacher-generated corruption,
           shuffled entity/causal/temporal variant, or student-preferred wrong continuation.
Teacher supplies preferences over these alternatives.
```

Mode A is easiest and most diagnostic. Mode B is closer to the democratization moonshot because it does not depend on benchmark labels.

### Attack

The concrete version exposes the pivot's first hard problem:

```text
If the gold label comes from benchmark data, the method is no longer pure teacher distillation.
```

It becomes:

```text
supervised MCQ fine-tuning + teacher-shaped hard negatives.
```

That may be useful, but it is not the manifesto's "multi-teacher cross-architecture KD" by itself.

If the method uses unlabeled text, another problem appears:

```text
What is the gold choice?
```

The true next span is not always the only plausible continuation. A retrieved or generated distractor may be equally coherent. Teacher preference may reflect style, memorization, or likelihood bias rather than correctness. The student could learn "which continuation Qwen likes" instead of "which continuation is causally right."

The teacher signal is also not neutral. In B10, full Qwen teacher continuation scoring on the smoke subset was:

| Benchmark | Qwen teacher accuracy | Mean margin |
|---|---:|---:|
| HellaSwag | 48.0% | -0.062 |
| PIQA | 74.0% | +0.138 |
| ARC-Easy | 54.0% | -0.179 |

That means a naive teacher-margin target would be dirty on HellaSwag and ARC-Easy. The teacher is useful, but not an oracle. If the loss forces the student to match teacher margins on teacher-wrong examples, it trains the student toward incorrect decisions.

The loss must therefore distinguish three cases:

| Case | Correct handling |
|---|---|
| Teacher ranks gold first with positive margin | Use as clean margin supervision. |
| Teacher ranks gold second but close | Use weakly or as uncertainty signal. |
| Teacher confidently ranks wrong first | Do not distill the teacher margin as truth; record as teacher failure/disagreement. |

The student must also see benchmark-style choices during training in a way that cannot be gamed by answer formatting. The training path must randomize:

- choice order;
- prompt templates;
- answer prefixes;
- choice length distribution;
- whitespace/capitalization where safe;
- gold position;
- candidate normalization.

Otherwise, a margin gain can be an artifact.

### Required First Implementation Contract

Functional Margin Distillation should be defined with this minimum contract:

```text
Training example:
  context: UTF-8 text bytes
  choices: textual candidate continuations
  gold: optional gold index
  teacher_scores: per-teacher normalized continuation losses
  hard_wrong: strongest non-gold choice by teacher or student

Student computation:
  score each candidate by continuation NLL in the student path
  normalize by bytes for cross-tokenizer comparison
  compute m_S = loss_wrong - loss_gold

Loss:
  rank gold above hard wrong
  regress only trusted teacher margins
  retain general LM ability with a CE/BPB anchor

Evaluation:
  same gold-vs-best-wrong margin definition
  held-out examples
  paired deltas against same-budget baselines
```

### What Survives

The mechanism is implementable with the current benchmark scorer plus a new training loop.

### What Dies

This dies:

```text
Functional Margin Distillation can be vaguely described and still be evaluated fairly.
```

No. The loss, labels, hard-negative rule, teacher filtering, normalization, and baseline controls must be predeclared.

### NARRATIVE ATTACK

**Strongest that's obvious dismissal:** You are training on multiple-choice answers with a ranking loss. That is not a new learning theory unless it beats ordinary supervised and KD baselines.

**Strongest that's trivial dismissal:** If the benchmark label supplies the gold answer, the teacher is not teaching correctness. The dataset is.

**What the result would need to be for the narrative to be unkillable:** The teacher must add measurable value beyond label-only training, especially on hard negatives and held-out counterfactual slices.

### Attack On The Next Defense

The next defense will say unlabeled text can be converted into contrastive pairs. Maybe. But that moves the hardest problem into negative construction and teacher trust. A bad negative generator will make a fake curriculum.

---

## Iteration 101: The Multi-Teacher Problem

### Steelman

Functional margins are one of the cleanest ways to approach multi-teacher cross-tokenizer KD.

The key move is to stop aligning token logits and hidden states. Each teacher receives the same textual context and candidate completions. Each teacher scores each candidate under its own tokenizer. The protocol then compares choices at the sequence level:

```text
score_{teacher k}(x, c_i) = -NLL_k(c_i | x) / byte_count(c_i)
```

For each teacher:

```text
margin_k(g, j) = score_k(g) - score_k(j)
```

Now Qwen, Mamba, hybrid models, and other architectures can contribute without sharing a tokenizer or hidden dimension. The object being transferred is:

```text
preference over textual alternatives
```

not:

```text
token-by-token distribution over a shared vocabulary
```

That is compatible with the Eklavya thesis:

```text
teachers are instruments, not masters.
```

A reasonable multi-teacher target can use robust aggregation:

```text
normalized_margin_k = zscore_or_temperature_calibrate(margin_k)
target_margin = weighted_median_k(normalized_margin_k)
disagreement = variance_k(normalized_margin_k) + top1_entropy + ranking_distance
```

The router can then learn from the disagreement metadata:

- teacher top-1 disagreement;
- margin variance;
- pairwise ranking distance;
- which teacher family is correct on which slice;
- whether the student is wrong or uncertain.

This avoids the worst tokenization trap because no teacher's token IDs become the student's target.

### Attack

This does not solve the tokenizer problem. It sidesteps one version of it.

Sequence-level NLLs are not automatically comparable across tokenizers. Even if normalized by bytes, teachers differ in:

- tokenizer granularity;
- token boundary placement;
- prompt template sensitivity;
- context formatting;
- calibration;
- pretraining distribution;
- length bias;
- answer-prefix bias;
- model family likelihood scale.

A Qwen margin of `+0.20 nats/byte` and a Mamba margin of `+0.20 nats/byte` may not mean the same thing. Without calibration, the highest-variance teacher can dominate the aggregate.

The prior-art attack is also sharp. Cross-tokenizer KD is an active research area, including methods such as Multi-Level Optimal Transport that explicitly target token- and sequence-level distribution mismatch. Multi-teacher KD also exists, including adaptive instance-level weighting. Therefore Sutra cannot claim novelty for:

```text
using multiple teachers
```

or:

```text
distilling across tokenizers
```

The possible novelty must be narrower:

```text
byte-native student + functional choice margins + disagreement-gated lessons
under a single-GPU democratization constraint.
```

That is a harder but more honest claim.

The teacher-disagreement case is where the pivot can break.

If teachers disagree on which completion is gold and the dataset has a gold label:

```text
Use the dataset gold as the correctness anchor.
Use teacher disagreement as confidence/diagnostic signal.
Do not train the student to copy a teacher that confidently prefers a wrong answer.
```

If teachers disagree and no gold label exists:

```text
Do not pretend the majority vote is truth.
```

The system needs one of:

- a verifier;
- a counterfactual construction where the positive is known by design;
- a high-confidence adjudication rule;
- a held-out human or benchmark label;
- or a "do not train correctness, only log disagreement" path.

Otherwise, multi-teacher training becomes:

```text
average the biases of several small models and hope the average is intelligence.
```

The hardest version of the tokenizer gap remains unsolved:

```text
How does the student learn the dark knowledge inside teacher distributions
when teachers have incompatible vocabularies?
```

Choice-level margins discard most of the teacher distribution. That may be the right tradeoff for a first experiment, but it is not full cross-tokenizer KD. It is compressed preference distillation.

### Required Multi-Teacher Admission Rules

Before a multi-teacher training run, measure:

| Gate | Required evidence |
|---|---|
| Teacher quality | Each teacher's accuracy and margin on the chosen slice. |
| Teacher diversity | Top-1 disagreement, ranking disagreement, and margin variance. |
| Useful disagreement | Fraction where at least one teacher is correct and at least one is wrong. |
| Calibration | Per-teacher margin scale after byte normalization and temperature calibration. |
| Oracle routing ceiling | Accuracy if an oracle selected the correct teacher per item. |
| Best-single-teacher baseline | Router must beat the best teacher, not the average teacher. |
| Prompt stability | Disagreement must survive prompt-template perturbation. |

If the oracle routing ceiling is not meaningfully above the best single teacher, the router is dead before training.

### What Survives

Sequence-level functional margins are a pragmatic way to start cross-tokenizer learning without token alignment.

### What Dies

This dies:

```text
Multi-teacher margin aggregation solves tokenizer mismatch.
```

It does not. It creates a tokenizer-agnostic evaluation surface and throws away token-level distribution detail.

### NARRATIVE ATTACK

**Strongest that's obvious dismissal:** You avoided token alignment by reducing every teacher to a scalar choice score. That is useful, but it is not a solution to cross-tokenizer KD.

**Strongest that's trivial dismissal:** Multi-teacher KD and cross-tokenizer KD already exist. The word "multi-teacher" is not the novelty.

**What the result would need to be for the narrative to be unkillable:** Multiple architecturally different teachers must produce complementary, calibrated, held-out gains in a byte-native student that no single teacher or averaged teacher can match.

### Attack On The Next Defense

The next defense will say disagreement is the point. Correct. Then the first required experiment is not training. It is measuring whether useful disagreement exists.

---

## Iteration 102: What Does The First Cheap Experiment Look Like?

### Steelman

The first experiment should not try to prove the moonshot. It should test one narrow question:

```text
Can a margin-trained byte-facing student path improve held-out gold-vs-wrong
choice margins over same-budget ordinary baselines?
```

The cheapest useful experiment is an FMD shadow smoke, not a full student run.

#### Proposed Experiment: `FMD_SHADOW_288`

Architecture:

```text
Frozen codec_phase1.5 encoder
Readout-conditioned adapter from codec hidden states to Qwen hidden size
Copied 4-layer Qwen core + Qwen LM head as the temporary scorer
Trainable component: adapter only, initialized fresh
```

This is not the final moonshot architecture. It is a cheap objective admission test. The label should stay hostile:

```text
FMD_ADAPTER_SHADOW_NOT_BYTE_NATIVE_PROOF
```

Training data:

```text
96 train-safe examples each from HellaSwag, PIQA, ARC-Easy = 288 examples.
For each example, score all choices with full Qwen teacher.
Use gold-vs-hardest-wrong pairs.
Filter or downweight examples where the teacher confidently ranks a wrong answer first.
Randomize choice order and answer formatting where the dataset permits it.
```

Held-out evaluation:

```text
48 disjoint train-safe examples each from HellaSwag, PIQA, ARC-Easy = 144 examples.
Optional second check on validation split if cache/offline access permits it.
```

Loss:

```text
m_S = student_loss_wrong - student_loss_gold
m_T = teacher_loss_wrong - teacher_loss_gold

L_rank = softplus(gamma * (target_margin - m_S))
L_reg  = Huber(m_S - clip(alpha * m_T, -M, M)) only on teacher-trusted examples
L_lm   = ordinary continuation CE/BPB anchor

L = L_rank + 0.25 * L_reg + lambda_lm * L_lm
```

Baselines under the same data/update budget:

| Baseline | Why it is required |
|---|---|
| MSE adapter from coordinate inheritance | Tests whether FMD beats the dead objective. |
| Label-only CE/ranking | Tests whether the teacher adds anything beyond labels. |
| Uniform token KD / CE | Tests whether pairwise margins beat ordinary distillation. |
| Random-label or random-teacher margins | Tests whether the pipeline is just regularizing. |
| Length/position artifact control | Tests whether gains come from option format. |

Primary metrics:

```text
MCQ accuracy
mean and median gold-vs-best-wrong margin
paired margin delta vs each baseline
paired sign-test win rate
train-heldout generalization gap
length/position-controlled residual
```

Precommitted verdict tokens:

```text
PASS_FMD_SHADOW
  Held-out accuracy improves >=+5pp over same-budget token-KD or label-only baseline
  on >=2 of 3 benchmarks, AND paired mean margin CI lower bound >0 on >=2 of 3,
  AND no benchmark regresses worse than -2pp.

MARGINAL_FMD_SHADOW
  Held-out accuracy improves +2pp to +5pp on >=2 of 3,
  but margin CI or artifact controls are ambiguous.
  Allows one bounded replication only.

FAIL_FMD_SHADOW
  Held-out accuracy improves <+2pp over same-budget baselines on >=2 of 3,
  OR paired margin CI lower bound <=0 on all benchmarks,
  OR gains are train-only.

FORMAT_ARTIFACT_FMD
  Gains disappear under choice-order, length-matched, or prompt-template controls.

TEACHER_POISON_FMD
  Gains occur mostly by copying teacher-wrong preferences or regress on teacher-correct slices.
```

Kill condition:

```text
If FMD cannot beat same-budget token-KD / label-only ranking by >=+2pp
on held-out examples in >=2 of 3 benchmarks, do not scale it.
```

Earning a second experiment requires more:

```text
PASS_FMD_SHADOW plus clean artifact controls.
```

Then the second experiment can train the actual byte-native S0/Sutra path or a small LoRA/adaptor on the real student decoder.

### Attack

This experiment is still dangerously scaffolded.

The copied 4-layer Qwen core and Qwen head are precisely the region where coordinate inheritance produced misleading NLL lift. Even if the adapter learns margins here, the result may not transfer to a byte-native decoder.

A hostile reviewer will say:

```text
You trained a bridge into a Qwen scorer to imitate Qwen's choice preferences.
Of course it can move Qwen-facing margins.
Where is the small byte model?
```

That attack is fair. Therefore the experiment must be labeled as an admission test only. It can answer:

```text
Does the margin objective have trainable signal in the existing scaffold?
```

It cannot answer:

```text
Does Sutra learn intelligence?
```

The experiment is also at risk of being too small. With 48 held-out examples per benchmark, +5pp can mean only a few flips. Bootstrap CIs will be wide. Verdict tokens must not pretend this is decisive moonshot evidence.

The strongest alternative first experiment may be a no-training data audit:

```text
Before training anything, score 500 train-safe examples with each teacher,
measure teacher accuracy, margin quality, hard-negative quality, and
teacher/student disagreement density.
```

If teacher margins are dirty or disagreement is sparse, training is premature.

### Minimum Honest First Step

The first W-Loop should do both:

```text
1. Teacher-margin data audit.
2. Tiny FMD shadow train only if the audit passes.
```

Audit pass thresholds:

| Gate | Pass condition |
|---|---|
| Teacher usable slices | Teacher ranks gold first on >=60% HellaSwag, >=70% PIQA, >=60% ARC-Easy in the sampled train-safe slice, or teacher-wrong examples are filtered. |
| Hard-negative quality | Hard wrong choices are not mostly shortest/longest option artifacts. |
| Margin separability | Teacher gold-vs-best-wrong margin is positive on a useful fraction of examples. |
| Baseline headroom | Baseline student/scaffold has enough wrong examples to improve without ceiling effects. |

If this audit fails, Functional Margin Distillation should not proceed until the data construction is fixed.

### What Survives

A tiny FMD shadow is the right cheap experiment if it is explicitly labeled as an objective admission test.

### What Dies

This dies:

```text
A 50-100 example margin smoke can prove the pivot.
```

No. It can only earn a second experiment.

### NARRATIVE ATTACK

**Strongest that's obvious dismissal:** Your first "student" is still a Qwen-headed scaffold. That is not Sutra.

**Strongest that's trivial dismissal:** A few held-out flips on tiny MCQ slices are not a paradigm shift.

**What the result would need to be for the narrative to be unkillable:** The first smoke must beat strong same-budget baselines cleanly, then a real byte-native student must retain the gain at larger held-out scale.

### Attack On The Next Defense

The next defense will say cheap tests are allowed to be narrow. Yes. But narrow tests need narrow labels. Do not let `PASS_FMD_SHADOW` become "Sutra learned teacher decision geometry."

---

## Iteration 103: The Narrative Test

### Steelman

The proposed story has a clean contrast:

```text
Instead of copying a teacher's brain coordinates,
Sutra learned the shape of the teacher's hardest decisions.
```

It is understandable, it directly incorporates the B10 falsification, and it avoids hidden-state mysticism. A reader can grasp the mechanism:

```text
The model sees hard alternatives.
The teacher tells it which side of the boundary is right.
The student learns to reproduce the boundary.
```

This is much stronger narratively than:

```text
We projected byte embeddings into Qwen hidden space and preserved NLL.
```

The story also aligns with Eklavya:

```text
teachers are instruments;
hard choices reveal transferable lessons;
consensus is less valuable than boundary cases.
```

If the final system becomes:

```text
multi-teacher, byte-native, disagreement-routed, hard-negative curriculum
```

then the story can survive as a real research narrative.

### Attack

As stated, the story does not survive "that's trivial."

A hostile literature-aware reviewer will say:

```text
You reinvented ranking distillation / preference optimization / contrastive KD.
```

That attack has teeth.

Ranking distillation existed in recommender and learning-to-rank settings years before this pivot. Neural ranking work includes listwise distillation. LLM alignment methods such as SLiC-HF, RRHF, and DPO use sequence likelihood, preference pairs, and ranking-style losses to align models without PPO-style RL. Multi-teacher KD and cross-tokenizer KD also have direct prior art.

So the following claims are not novel:

| Claim | Status |
|---|---|
| Train a student from teacher rankings | Prior art. |
| Use pairwise preference/ranking losses | Prior art. |
| Use teacher scores on candidate outputs | Prior art. |
| Avoid hidden-state matching | Prior art. |
| Multi-teacher KD | Prior art. |
| Cross-tokenizer KD | Prior art. |
| Direct preference optimization style loss | Prior art. |

The possible novelty is a conjunction, not a component:

```text
byte-native small model
+ multiple architecturally diverse teachers
+ tokenizer-agnostic sequence-level functional margins
+ disagreement/error routed lesson selection
+ retained held-out gains over same-budget KD
+ single-consumer-GPU constraint
```

If any of those terms is missing, the novelty weakens sharply.

The narrative should therefore not claim:

```text
We invented functional margin distillation.
```

It should claim, if proven:

```text
We showed that a byte-native small model can retain cross-tokenizer,
multi-teacher decision-boundary gains more efficiently than ordinary KD.
```

That is a much harder sentence, but it is the moonshot sentence.

### The Triviality Trap

The phrase "teacher's hardest decisions" is vulnerable because most ranking methods already focus on hard negatives or preference pairs.

A hostile reviewer can reduce the story to:

```text
You did hard-negative distillation on multiple-choice benchmarks.
```

The only way out is empirical:

- same-budget token KD loses;
- label-only ranking loses;
- single-teacher loses to multi-teacher;
- averaged teachers lose to disagreement routing;
- byte-native student retains the gain;
- gains transfer to held-out tasks or counterfactual slices;
- artifact controls fail to explain the gain.

Without those results, the story is marketing.

### What Would Count As Actual Novelty

The pivot can become nontrivial if it establishes one of these:

| Novelty candidate | Required proof |
|---|---|
| Byte-native margin retention | Byte model keeps teacher decision-boundary gains without shared tokenizer. |
| Disagreement as curriculum | Disagreement-routed examples beat uniform KD at fixed update budget. |
| Cross-architecture complementarity | Diverse teachers produce gains no single teacher can reproduce. |
| Surgical improvability | Error clusters can be repaired with local lesson packets and little regression. |
| Data efficiency | A tiny number of teacher-scored hard pairs produces benchmark gains that ordinary KD misses. |

The story should be held back until one of these is real.

### What Survives

The phrase is useful as an internal hypothesis label.

### What Dies

This dies:

```text
Functional Margin Distillation is narratively sufficient as the pivot.
```

No. It is a baseline loss family until the broader Eklavya mechanism proves retained gains.

### NARRATIVE ATTACK

**Strongest that's obvious dismissal:** After hidden-state copying failed, you moved to ranking loss. That is the standard fallback.

**Strongest that's trivial dismissal:** "Teacher's hardest decisions" means hard-negative KD. The field already has that.

**What the result would need to be for the narrative to be unkillable:** The result must show that the byte-native, multi-teacher, disagreement-routed protocol beats ordinary ranking KD and token KD under the same compute.

### Attack On The Next Defense

The next defense will say the combination is novel. Maybe. But combinations are cheap to name and hard to prove. The burden is to show that the combination produces retained functional gains that the components do not.

---

## Iteration 104: Disagreement Router - Does Enough Disagreement Exist?

### Steelman

At 0.6B-1.7B scale, there should be nontrivial disagreement.

Small models are far from saturated on HellaSwag, PIQA, ARC-Easy, ARC-Challenge, MMLU-style cloze, and Winogrande-style tasks. Different architectures and tokenizers may have different failure modes:

- transformer vs SSM/hybrid sequence biases;
- BPE vs byte/other tokenization artifacts;
- commonsense vs science fact weaknesses;
- short vs long continuation behavior;
- lexical overlap traps;
- physical affordance traps.

If teachers are imperfect and differently imperfect, disagreement can be a rich training signal. The router does not need all examples. It needs examples where:

```text
student is wrong or uncertain
AND teacher opinions diverge
AND at least one teacher has useful signal.
```

Even if teacher top-1 agreement is high on easy examples, disagreement may concentrate exactly where the student needs help. That is the steelman:

```text
Consensus teaches little. Disagreement marks the boundary.
```

The router can also expand the disagreement set by constructing near-miss candidates:

- student-preferred wrong vs gold;
- teacher-preferred wrong vs gold;
- counterfactual edits;
- retrieved hard negatives;
- teacher-generated plausible distractors.

So low raw disagreement on benchmark choices would not automatically kill the idea.

### Attack

The disagreement router may have no useful fuel.

The relevant quantity is not raw teacher disagreement. It is useful teacher disagreement:

```text
useful_disagreement =
  teacher top-1 differs
  AND student is wrong or low-margin
  AND at least one teacher is correct
  AND the disagreement is stable under prompt/template perturbation
  AND the correct side can be identified without leaking the benchmark label.
```

This set may be tiny.

If teachers agree on 90% of examples and the remaining 10% is mostly random calibration noise, the router is dead. If teachers disagree because one tokenizer likes a shorter option, the router is worse than dead: it trains on artifacts.

The worst case is correlated wrongness:

```text
Teachers disagree in confidence but not in correctness.
They share the same web priors, shallow heuristics, and benchmark biases.
```

Then the router learns to sort teacher personalities, not solve tasks.

Another hostile possibility:

```text
The best single teacher already dominates.
```

If Qwen is better on most slices and Mamba adds little complementary correctness, a router cannot beat a best-teacher baseline except by overfitting. Multi-teacher language becomes ornamental.

### Required Disagreement Density Audit

Before training a router, run a no-training audit on a train-safe sample:

```text
Benchmarks: HellaSwag, PIQA, ARC-Easy, ARC-Challenge if available.
Examples: at least 500 per benchmark if cache permits; otherwise minimum 200.
Teachers: Qwen3-0.6B plus at least one architecturally different teacher
          in the 0.6B-1.7B range.
Student: current S0 or strongest available byte-facing baseline.
```

Metrics:

| Metric | Definition | Router implication |
|---|---|---|
| Top-1 disagreement rate | Fraction where teachers choose different answers. | Raw fuel. |
| Ranking disagreement | Pairwise Kendall or full-ranking mismatch. | Boundary richness. |
| Useful disagreement rate | Teachers disagree and at least one is correct while student is wrong/uncertain. | Trainable fuel. |
| Oracle routing ceiling | Accuracy if an oracle chooses a correct teacher when one exists. | Maximum possible router gain. |
| Best-teacher gap | Oracle ceiling minus best single teacher. | Room for routing. |
| Noise sensitivity | Disagreement stability under prompt template/choice order changes. | Artifact risk. |
| Slice complementarity | Which teacher wins by task type. | Router feature validity. |

Precommitted verdict tokens:

```text
PASS_DISAGREEMENT_DENSITY
  Useful disagreement >=15% on >=2 of 3 benchmark families
  AND oracle routing ceiling beats best single teacher by >=5pp
  AND prompt perturbation preserves >=70% of disagreement labels.

MARGINAL_DISAGREEMENT_DENSITY
  Useful disagreement 8-15% or oracle ceiling +2pp to +5pp.
  Allows router only as secondary diagnostic.

FAIL_DISAGREEMENT_DENSITY
  Useful disagreement <8% on >=2 of 3,
  OR oracle routing ceiling beats best single teacher by <2pp,
  OR disagreement is prompt/length unstable.
```

If `FAIL_DISAGREEMENT_DENSITY` fires, do not build the router.

### What If Disagreement Is Mostly Noise?

Then the router should be demoted to an Error Atlas diagnostic:

```text
Use disagreement to label where teachers are unreliable.
Do not use it as training supervision.
```

Better pivot targets in that world:

1. Counterfactual Minimal-Pair Curriculum.
2. Error Atlas and Surgical Skill Patches.
3. Byte-Native Teacher Debate Compression with verifier/adjudication.

These require correctness anchors. They are less dependent on the raw existence of useful teacher fights.

### What Survives

The disagreement thesis is still aligned with Eklavya, but only after a density audit.

### What Dies

This dies:

```text
Teacher disagreement is automatically a large, useful training set.
```

No. It may be sparse, correlated, noisy, or artifact-driven.

### NARRATIVE ATTACK

**Strongest that's obvious dismissal:** Your router studies fights between teachers, but the teachers rarely fight where it matters.

**Strongest that's trivial dismissal:** If disagreement just marks noisy examples, you built a noise router.

**What the result would need to be for the narrative to be unkillable:** Useful disagreement must be abundant enough, stable enough, and complementary enough that a router can beat the best single teacher and uniform KD.

### Attack On The Next Defense

The next defense will say disagreement can be generated with synthetic hard negatives. Yes, but then the claim changes. The core resource is no longer natural teacher disagreement. It is counterfactual data construction.

---

## Iteration 105: What Should The Dual-Loop Change?

### Steelman

The dual-loop worked in its most important recent test.

It killed coordinate inheritance after the precommitted functional-margin shadow failed. It did not allow the 3-4 nats NLL lift to become a public claim. It preserved the distinction between:

```text
diagnostic insight
```

and:

```text
moonshot evidence
```

That is rare and valuable. The process should not become less adversarial.

For the pivot, the dual-loop should preserve:

- precommitted verdict tokens;
- strongest-control residuals;
- graveyard discipline;
- narrative attacks;
- analysis before implementation;
- held-out functional gates.

### Attack

The process can still fail by becoming an excellent falsification machine around a mediocre search strategy.

B14 named the pathologies:

- gate proliferation;
- false precision;
- post-failure metric migration;
- local search trap.

B15 adds a new one:

```text
baseline laundering.
```

After a favorite mechanism dies, the project can pivot to an ordinary baseline and give it moonshot language because it is cleaner than the failed mechanism. Functional Margin Distillation is at risk of that. It is a good baseline. It is not yet the breakthrough.

The dual-loop should change in six ways.

### Change 1: Split Admission Gates From Moonshot Gates

Every new direction gets two labels:

```text
ADMISSION_EVIDENCE
MOONSHOT_EVIDENCE
```

Functional-margin shadow gains in a Qwen-headed scaffold can be admission evidence. They cannot be moonshot evidence.

Moonshot evidence requires:

```text
byte-native student
held-out benchmark gain
same-budget strong baselines
artifact controls
retained gain after compression
```

### Change 2: Run A Pivot Portfolio, Not A Serial Obsession

The next W-Loop should not spend the whole batch deepening only Functional Margin Distillation.

Minimum portfolio:

| Probe | Purpose |
|---|---|
| FMD shadow smoke | Test whether margin objective trains. |
| Teacher-margin data audit | Test whether teacher signals are clean enough. |
| Disagreement density audit | Test whether router fuel exists. |
| Counterfactual/minimal-pair sketch | Keep a correctness-anchored alternative alive. |

This is not "do everything." It is a small evidence board. If FMD fails, the loop should not need another full reset to know where to turn.

### Change 3: Require Same-Budget Strong Baselines From Day One

The old coordinate inheritance controls were good for a hidden-state claim. The new pivot needs different controls:

- same-budget token KD;
- same-budget label-only CE/ranking;
- same-budget DPO/RRHF-style pairwise preference baseline;
- random teacher margins;
- teacher-confidence-only curriculum;
- length/position artifact controls;
- best-single-teacher baseline for multi-teacher claims;
- averaged-teacher baseline for router claims.

No more promotion over weak controls.

### Change 4: Add A Novelty Gate

Every pivot report must include:

```text
What part is prior art?
What part is the new claim?
What evidence would separate the new claim from the prior-art baseline?
```

For FMD, the novelty gate says:

```text
Ranking loss is prior art.
Byte-native cross-tokenizer multi-teacher retained gain is the possible new claim.
```

If a batch cannot name the separable novelty, it should not use moonshot language.

### Change 5: Reduce False Precision

Verdict tokens should be paired with actual counts and uncertainty:

```text
flips won/lost
mean margin delta
median margin delta
sign-test win rate
bootstrap CI
artifact-control residual
train/held-out gap
```

Do not let a token like `PASS_FMD_SHADOW` hide that the pass was three examples on a tiny sample.

### Change 6: One Repair Cycle Unless Functional Evidence Is Strong

No new pivot should get a long repair chain before it has clean held-out functional evidence.

Rule:

```text
One failed admission smoke allows one bounded redesign only if the failure
identifies a specific fixable artifact.

A second failure kills or demotes the direction.
```

The loop exists to find the breakthrough, not to become loyal to the newest vocabulary.

### Revised Cadence

Recommended cadence:

```text
Q-Loop: hostile design and novelty audit.
W-Loop: small portfolio probe with precommitted gates.
Q-Loop: compare survivors against prior art and moonshot bar.
W-Loop: scale only the survivor that beat strongest same-budget baselines.
```

For B11 specifically:

```text
Do not start with a full implementation of the final FMD system.
Start with FMD_SHADOW_288 + teacher-margin audit + disagreement density audit.
```

### What Survives

The dual-loop remains valuable. It killed coordinate inheritance cleanly.

### What Dies

This dies:

```text
The next named pivot inherits the emotional energy of the dead direction.
```

No. The pivot starts at zero evidence.

### NARRATIVE ATTACK

**Strongest that's obvious dismissal:** You are good at killing the previous idea, but now you are in love with the next obvious baseline.

**Strongest that's trivial dismissal:** A rigorous process around ordinary KD is still ordinary KD.

**What the process would need to be for the narrative to be unkillable:** The loop must compare multiple live mechanisms, kill weak ones fast, and reserve moonshot language for byte-native retained gains over serious baselines.

### Attack On The Next Defense

The next defense will say focus is necessary. True. But focus before admission evidence is how local search traps form. The first pivot batch needs breadth; the second can focus.

---

## Batch 15 Final Verdict

Functional Margin Distillation should be the first pivot probe, but only under hostile labels:

```text
FMD_ADMISSION_TEST
not
SUTRA_DECISION_GEOMETRY_PROVEN
```

The honest state is:

| Direction | Status | Reason |
|---|---|---|
| Coordinate Inheritance | Permanently dead | Failed functional-margin shadow on all 3 benchmarks. |
| Functional Margin Distillation | Alive as first cheap probe | Directly targets candidate margins, but ranking KD is prior art and compression may fail. |
| Disagreement Geometry Router | Alive only after audit | Needs useful teacher disagreement density and oracle-routing headroom. |
| Counterfactual Minimal-Pair Curriculum | Should remain warm | More correctness-anchored if teacher disagreement is sparse/noisy. |
| Error Atlas / Surgical Patches | Should remain warm | Serves improvability and can use failures as structured data. |

Precommit the next W-Loop around these verdict tokens:

```text
PASS_FMD_SHADOW
MARGINAL_FMD_SHADOW
FAIL_FMD_SHADOW
FORMAT_ARTIFACT_FMD
TEACHER_POISON_FMD

PASS_DISAGREEMENT_DENSITY
MARGINAL_DISAGREEMENT_DENSITY
FAIL_DISAGREEMENT_DENSITY
```

Hostile final statement:

```text
The pivot is not dead, but the story is not yet earned.
Functional margins are the correct next measurement target.
They are not automatically a new learning mechanism.
Teacher disagreement is the right manifesto direction.
It is not automatically abundant or useful.
The dual-loop must now prove it can search a portfolio,
not merely rename the failure mode after killing coordinate inheritance.
```

If FMD fails against same-budget token KD and label-only ranking, do not repair it for multiple cycles. Move to counterfactual minimal pairs, Error Atlas, or teacher-debate compression with correctness anchors.

