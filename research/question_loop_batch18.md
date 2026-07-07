# QUESTION LOOP - Batch 18: Attack The Token-Level Pivot

Date: 2026-07-07

Iterations: 120-126

## Grounding

I read the requested local context in order:

1. `research/VISION.md`
2. `research/dual_loop_supervisor_checkin_12.md`
3. `research/dual_loop_supervisor_checkin_11.md`
4. `research/question_loop_batch17.md`
5. `research/work_loop_batch13.md`
6. `research/work_loop_batch12.md`
7. `code/margin_distillation.py`

No GPU runs, training runs, benchmark runs, or experiments were performed. This is analysis only.

External sources checked for the literature-dependent parts:

- SmolLM2-135M model card: https://huggingface.co/HuggingFaceTB/SmolLM2-135M
- SmolLM2 paper: https://arxiv.org/abs/2502.02737
- Foundational knowledge distillation: https://arxiv.org/abs/1503.02531
- Adaptive Multi-Teacher Multi-level KD: https://arxiv.org/abs/2103.04062
- Confidence-Aware Multi-Teacher KD: https://arxiv.org/abs/2201.00007
- Adaptive Multi-Teacher KD with Meta-Learning: https://arxiv.org/abs/2306.06634
- Multi-Teacher KD with Reinforcement Learning: https://arxiv.org/abs/2502.18510
- LoRA: https://arxiv.org/abs/2106.09685
- QLoRA: https://arxiv.org/abs/2305.14314
- ByT5: https://arxiv.org/abs/2105.13626
- MEGABYTE: https://arxiv.org/abs/2305.07185
- Byte Latent Transformer: https://arxiv.org/abs/2412.09871
- Universal Logit Distillation: https://arxiv.org/abs/2402.12030
- Multi-Level Optimal Transport for cross-tokenizer KD: https://arxiv.org/abs/2412.14528

## Binding Facts Entering Batch 18

The byte-native mainline is demoted, not dead.

`FAIL_S0_CAPACITY` is binding:

| Test | Train movement | Held-out result |
|---|---:|---|
| Frozen residual head | 38.5% -> 60.1% | 0/3 pass |
| 50-step native full fine-tune | 38.5% -> 58.0% | 0/3 pass |
| 100-step native full fine-tune | 38.5% -> 73.6% | 1/3 pass, ARC-Easy only |

The important lesion is not optimizer death. Wide7 absorbed train labels, but did not generalize broadly.

`UPGRADE_TEACHER` is binding:

| Benchmark | Qwen3-0.6B | SmolLM2-360M | Action |
|---|---:|---:|---|
| HellaSwag | 49.5% | 56.0% | SmolLM2 primary |
| PIQA | 67.5% | 65.0% | Qwen remains useful |
| ARC-Easy | 34.0% | 56.5% | SmolLM2 primary |

`PASS_DISAGREEMENT` is real but label-anchored:

| Benchmark | Useful disagreement | Oracle gap over best teacher |
|---|---:|---:|
| HellaSwag | 17.0% | +4.0pp |
| PIQA | 20.0% | +8.0pp |
| ARC-Easy | 34.5% | +7.0pp |
| Aggregate | 23.8% | +6.3pp |

This is fuel, not a router. The audit used labels to identify usefulness. A deployed or held-out router cannot use held-out labels.

The official SmolLM2-135M model card confirms the prompt's target numbers:

| Metric | SmolLM2-135M base |
|---|---:|
| HellaSwag | 42.1 |
| ARC average | 43.9 |
| PIQA | 68.4 |
| MMLU cloze | 31.5 |
| WinoGrande | 51.3 |
| OpenBookQA | 34.6 |

The same card states that SmolLM2-135M was pretrained on 2T tokens, uses a Transformer decoder, and was trained on 64 H100 GPUs. That makes it a strong mechanism control and a terrible compute-fair comparison to the local byte student.

## Hostile Verdict Up Front

The token-level pivot is valid only under one definition:

```text
SmolLM2-135M is a terminal control for the Eklavya protocol, not evidence for the byte-native Sutra architecture and not a compute-fair comparison to S0.
```

The hostile reviewer can knock down any softer claim.

If SmolLM2-135M improves under label-only CE, that is ordinary fine-tuning. If it improves under single-teacher KD, that is ordinary KD. If it improves under a static best-teacher or uniform teacher ensemble, that is ordinary ensemble distillation. Eklavya exists only in the residual:

```text
same student + same data + same budget + no held-out label leakage, where disagreement-aware multi-teacher routing beats label-only, single-teacher KD, static best-teacher imitation, uniform/entropy teacher mixing, and random/shuffled routing.
```

The precommitted >=3pp residual on >=2/3 benchmarks is a minimum continuation bar, not a moonshot bar. With 48 examples per benchmark, one example is 2.08pp, so a per-benchmark ">=3pp" pass means at least 2 extra correct examples (+4.17pp) on that benchmark. Anything smaller is noise-level engineering.

Final batch token:

```text
TOKEN_CONTROL_VALID_BUT_NOT_MOONSHOT_YET
```

---

## Iteration 120: Is SmolLM2-135M The Right Control Student?

### Steelman

SmolLM2-135M is the right control if the question is narrowed to the mechanism:

```text
Can the Eklavya protocol produce residual held-out gains in a same-scale student that already has benchmark-facing language function?
```

It is same order of magnitude as Sutra/Wide7. It is open. It is small enough to fit local adaptation budgets. It has published benchmark function. The model card reports HellaSwag 42.1, ARC average 43.9, PIQA 68.4, WinoGrande 51.3, and MMLU cloze 31.5, with 2T-token pretraining on 64 H100s. That is exactly why it is useful: it removes the "student cannot learn at all" confound that killed MarginStudent and Wide7.

Mechanism control means the byte substrate is deliberately held out of the claim. The controlled variable is not tokenizer, compute history, or architecture identity. The controlled variable is the training protocol:

| Condition | Meaning |
|---|---|
| SmolLM2 zero-shot | base pretrained function |
| label-only CE | ordinary supervised adaptation |
| single-teacher KD | ordinary soft-label distillation |
| static best-teacher imitation | copying the strongest available teacher |
| uniform/entropy teacher mix | generic ensemble distillation |
| random/shuffled routing | route complexity without useful routing |
| disagreement routing | the Eklavya-specific mechanism |

If the final row beats every row above it under identical data and update budgets, Eklavya survives as a protocol. If not, the protocol has not earned another byte-native repair cycle.

There is also a fair reason to choose the 135M model rather than SmolLM2-360M as student. SmolLM2-360M is already a strong teacher in the local audit. If the student is 360M, the experiment becomes less relevant to the original small-model democratization target. A 135M pretrained student is the cleanest available same-class functional engine.

### Attack

SmolLM2-135M is not a fair comparison target for the local byte model.

The byte model trained locally on far less compute and failed a small MCQ capacity test. SmolLM2-135M was trained by Hugging Face on 2T tokens using 64 H100 GPUs. A hostile reviewer will say:

```text
You could not make your 121M byte model work, so you imported a 2T-token BPE model and moved the goalposts from "Sutra learns differently" to "fine-tuning a competent public checkpoint works."
```

That attack is correct unless the paper trail uses strict language:

```text
This is not Sutra proof.
This is not byte proof.
This is not compute-efficiency proof.
This is a protocol falsification control.
```

There are two opposite failure modes.

First, SmolLM2-135M may be too capable. It may already encode the relevant benchmark function so well that 288 examples add little. Then a null Eklavya result might reflect saturation or evaluation granularity, not protocol death. But this objection cuts both ways: if the protocol cannot add value on examples where teachers visibly disagree and an oracle gap exists, then the practical claim is weak.

Second, SmolLM2-135M may be capable enough that any improvement is trivial. That is the more dangerous narrative failure. A label-only bump or single-KD bump is not Eklavya. It is expected behavior for a pretrained model. The burden is not "does SmolLM2 improve?" The burden is:

```text
Does disagreement-aware multi-teacher routing improve beyond ordinary fine-tuning and ordinary KD?
```

The teacher portfolio also weakens the mechanism-control claim. SmolLM2-360M and SmolLM2-135M share model family and likely data lineage. That is useful for clean transfer, but it undercuts the diverse-teacher story. Qwen supplies some diversity, yet B12/B13 show Qwen is weak or poisoned on ARC-Easy. A "multi-teacher" gain from mostly trusting SmolLM2-360M is not a rich Eklavya story.

### Decision

Use SmolLM2-135M, but name it precisely:

```text
terminal token-level Eklavya protocol control
```

Do not call it:

```text
Sutra evidence
byte-native evidence
compute-fair evidence
democratization proof
```

The only valid positive claim after B14 is:

```text
The Eklavya routing/training protocol produced residual gains in a competent same-scale open student beyond ordinary label-only and KD baselines.
```

That would justify a byte-return design. It would not itself complete the moonshot.

### NARRATIVE ATTACK

**Strongest "that's obvious" dismissal:** A pretrained 135M BPE model trained on 2T tokens learns from task supervision better than a locally trained byte scout.

**Strongest "that's trivial" dismissal:** You tested whether fine-tuning SmolLM2 works. The field already knows fine-tuning works.

**What would make the narrative hard to kill:** The Eklavya row must beat label-only, single-teacher KD, static best-teacher imitation, uniform/entropy teacher mixing, and random/shuffled routing under matched budget, with gains concentrated on label-free teacher-disagreement cases.

### Attack On The Next Defense

The next defense will say SmolLM2 is only a control. Correct. Then the result must be written as a control result, even if the numbers look exciting.

---
## Iteration 121: What Baselines Kill The Eklavya Claim?

### Steelman

The minimal baseline set in check-in #12 is directionally right:

1. SmolLM2-135M zero-shot.
2. Label-only fine-tune.
3. Single-teacher KD from SmolLM2-360M.
4. Best-teacher imitation.

Those separate the obvious explanations:

| Explanation | Baseline that tests it |
|---|---|
| The pretrained student already had the skill | zero-shot |
| Labels alone explain the gain | label-only CE |
| Any strong teacher explains the gain | single-teacher KD |
| The best teacher alone explains the gain | best-teacher imitation |

The foundational KD paper frames distillation as compressing teacher or ensemble behavior into a student. The multi-teacher KD literature then shows that teacher weighting is itself a known central problem: instance-level weights, confidence-aware weights, meta-weight networks, and RL teacher-weight agents all exist in prior work. Therefore Eklavya cannot claim novelty or value merely because multiple teachers were used. It must beat simpler multi-teacher aggregators.

LoRA is also a legitimate terminal adaptation mode. The LoRA paper reports that freezing base weights and training low-rank matrices can match or exceed full fine-tuning in several settings while greatly reducing trainable parameters and memory. QLoRA strengthens the practical point: parameter-efficient adaptation is ordinary engineering now. If B14 uses LoRA, that choice does not make the result special. It just makes the experiment cheaper.

### Attack

The baseline list in check-in #12 is not sufficient.

A hostile reviewer needs the following strict baselines:

| Baseline | Strict? | Why |
|---|---|---|
| Zero-shot SmolLM2-135M | Yes | Establishes base function. |
| Label-only CE | Yes | Kills "teacher value" if matched or beaten. |
| Single-teacher KD: SmolLM2-360M | Yes | Kills "multi-teacher value" if matched or beaten. |
| Single-teacher KD: Qwen3-0.6B | Yes, if Qwen contributes to router | Shows whether Qwen adds value or poison. |
| Static best-teacher by benchmark | Yes | Kills routing if per-benchmark fixed teacher suffices. |
| Uniform teacher average | Yes | Kills "disagreement routing" if simple ensemble averaging suffices. |
| Entropy/confidence-weighted teacher mix | Yes | Kills routing if generic confidence weighting suffices. |
| Random routing over teachers | Yes | Detects route-complexity and stochastic regularization effects. |
| Shuffled router labels/weights | Yes | Detects leakage and overfit in learned routing. |
| Same-data no-teacher control | Conditional | Required if the Eklavya arm uses extra unlabeled or generated data. |
| LoRA label-only vs LoRA Eklavya | Yes if LoRA is used | Keeps trainable-parameter budget matched. |
| Full fine-tune label-only vs full Eklavya | Strongly recommended | Prevents "LoRA bottleneck" objections on a 135M model. |
| Oracle label router | Ceiling only, not a baseline | Shows headroom but cannot be claimed. |

The strictest objection concerns LoRA versus full fine-tune. SmolLM2-135M is small enough that full fine-tuning is not absurd. If B14 uses only LoRA and fails, the reviewer can say the adapter bottleneck killed the result. If B14 uses only LoRA and passes, the reviewer can ask whether full label-only fine-tuning would erase the residual. Therefore the cleanest terminal design is one of:

```text
Option A: Full fine-tune is the binding experiment.
Option B: LoRA is binding only if label-only, KD, and Eklavya all use identical LoRA rank/targets, and at least the label-only winner is spot-checked with full fine-tuning.
```

The "best-teacher imitation" baseline must also be defined carefully:

| Variant | Meaning | Role |
|---|---|---|
| Static global best teacher | Always use teacher with best aggregate train calibration. | Weak baseline. |
| Static per-benchmark best teacher | Use SmolLM2 for HellaSwag/ARC and Qwen for PIQA if train calibration says so. | Required. |
| Oracle per-example best teacher | Uses labels per example. | Ceiling only, illegal as deployed baseline. |

If Eklavya only beats the global static teacher but loses to per-benchmark static best, the router did not earn its complexity. It just rediscovered benchmark identity.

### Required Baseline Board For W-Loop B14

Binding B14 should report this table:

| Arm | Train signal | Teacher policy | Trainable params |
|---|---|---|---|
| A0 | none | none | none |
| A1 | labels | none | matched |
| A2 | labels + KD | SmolLM2-360M only | matched |
| A3 | labels + KD | Qwen3-0.6B only | matched |
| A4 | labels + KD | static per-benchmark best teacher | matched |
| A5 | labels + KD | uniform teacher mix | matched |
| A6 | labels + KD | entropy/confidence teacher mix | matched |
| A7 | labels + KD | random teacher route on disagreement cases | matched |
| A8 | labels + KD | shuffled learned-router weights | matched |
| A9 | labels + KD | Eklavya learned/calibrated disagreement router | matched |

Optional but valuable:

| Arm | Why |
|---|---|
| no-label KD only | Separates hard-label reliance from teacher transfer. |
| larger train set sensitivity | Tests sample efficiency slope. |
| held-out unseen benchmark | Tests transfer beyond the tuned triad. |
| repeated seeds | Needed if small sample variance is high. |

### What Kills The Claim

The claim is killed if any of these happen:

1. Label-only CE matches Eklavya within 1pp aggregate.
2. Single-teacher SmolLM2 KD matches Eklavya within 1pp aggregate.
3. Static per-benchmark best teacher matches Eklavya within 1pp aggregate.
4. Uniform or entropy teacher mixing matches Eklavya within 1pp aggregate.
5. Random or shuffled routing matches Eklavya within 1pp aggregate.
6. Eklavya improves only on the training split or only by train-label leakage.
7. Eklavya gains disappear under a matched full fine-tune label-only check.

### NARRATIVE ATTACK

**Strongest "that's obvious" dismissal:** You used more teacher signal than the baseline and got more performance.

**Strongest "that's trivial" dismissal:** A static ensemble or confidence weighted teacher average would have done the same thing.

**What would make the narrative hard to kill:** A matched-budget Eklavya arm beats label-only, both single-teacher KDs, static per-benchmark teacher, uniform/entropy ensemble, random routing, and shuffled routing, with the largest lift on teacher-disagreement examples.

### Attack On The Next Defense

The next defense will say too many baselines slow the terminal test. That is not acceptable. Without these baselines, a positive B14 result is not terminal. It is just another ambiguous bump.

---

## Iteration 122: How Should Disagreement Routing Work?

### Steelman

The label-anchored B12 disagreement audit is useful because it proves headroom. It does not define a deployable router. A real router must use only features available before seeing the held-out label:

| Feature family | Examples |
|---|---|
| Teacher distribution features | top probability, margin, entropy, logit/NLL spread |
| Teacher disagreement features | top-1 disagreement, Jensen-Shannon divergence, pairwise KL, rank correlation |
| Calibration features | teacher-specific reliability from train calibration bins |
| Student gap features | student entropy, student-teacher KL, student margin, current loss proxy |
| Instance features | benchmark/task id if allowed, number of choices, context length, choice length skew |

The multi-teacher KD literature supports this direction. AMTML-KD learns instance-level teacher importance weights. CA-MKD attacks the problem that label-free averaging or heuristic mixing can mislead the student when teachers are low quality, and it uses labels to assign sample-wise reliability. MMKD uses a meta-weight network to tailor ensemble teacher knowledge to the student. MTKD-RL explicitly treats teacher weighting as a decision problem using teacher performance and teacher-student gaps as state.

The practical B14 router should be simpler than the literature's heavier versions because the train set is tiny. A deep meta-router over 288 examples is an overfit machine. Use a calibrated low-capacity router:

```text
Input: teacher/student choice-distribution features for one MCQ item.
Output: soft teacher weights w_t and optional KD strength alpha.
Target: maximize train-split reliability under cross-validation, then freeze or jointly train with strong regularization.
```

### Attack

The phrase "disagreement routing" hides the central unresolved problem:

```text
Which teacher should be trusted when they disagree, without using the answer?
```

Teacher confidence alone is not enough. B12 showed Qwen was confidently wrong on ARC-Easy 65.5% of the time. If Qwen's confidence is miscalibrated, a confidence router amplifies poison. Teacher agreement entropy alone is not enough either: high disagreement tells us a case is interesting, not which teacher is correct.

A learned meta-classifier is tempting, but with 288 training examples it can learn benchmark and length artifacts. ARC-Easy in B13 already had a length-skew warning for hard negatives. A router that keys on benchmark identity or choice length may pass the local split and fail outside it.

Therefore the router must be precommitted and low capacity.

### Concrete Router Design For B14

Use a two-stage calibrated soft router, not a black-box hard router.

#### Stage 1: Precompute distributions

For every train and held-out MCQ item, compute:

```text
q_smol = softmax(-choice_nll_smol360 / T_teacher)
q_qwen = softmax(-choice_nll_qwen / T_teacher)
p_student = softmax(-choice_nll_student / T_student)
```

Use choice-level distributions, not raw token logits, so the method is tokenizer-independent at the MCQ interface.

#### Stage 2: Calibrate teacher reliability on training labels only

Fit a tiny reliability model per teacher:

```text
r_t(x) = P(teacher t is correct | features_t(x), features_pair(x))
```

Recommended model:

```text
logistic regression or monotone calibrated bins
```

Allowed features:

- teacher top-1 margin;
- teacher entropy;
- teacher positive margin proxy;
- top-1 disagreement flag;
- JS divergence between teachers;
- student-teacher KL;
- number of choices;
- context/choice length diagnostics.

Dangerous features:

- raw example id;
- held-out labels;
- post-hoc benchmark-specific hand rules derived from held-out performance.

Use cross-fitting inside the 288 training examples:

```text
split train into K folds
fit reliability model on K-1 folds
predict reliability on the held fold
use out-of-fold reliability for student training targets
refit final reliability model on all train examples only for held-out inference
```

This prevents the router from seeing its own labels for target construction.

#### Stage 3: Build the teacher target

Compute soft weights:

```text
w_t(x) = softmax(beta * logit(r_t(x)) - gamma * entropy(q_t(x)))
q_route(x) = sum_t w_t(x) * q_t(x)
```

Use a floor to avoid collapse:

```text
w_t <- (1 - epsilon) * w_t + epsilon / n_teachers
```

For two teachers, this is enough. Do not train a deep router unless this simple router passes first.

#### Stage 4: Train the student

Use the matched objective:

```text
L = CE(y, p_student)
    + lambda_kd * T^2 * KL(q_route || p_student_T)
    + lambda_margin * pairwise_margin_loss(q_route, p_student)
    + lambda_entropy * route_entropy_regularizer
```

Where:

- `CE(y, p_student)` is included in every supervised arm that uses labels;
- `KL(q_route || p_student_T)` is the soft-label teacher transfer;
- `pairwise_margin_loss` preserves choice ranking margins;
- the entropy regularizer prevents the router from becoming an unreported static single-teacher selector.

Report a no-hard-label variant only as an optional diagnostic. The binding comparison should keep labels present because the label-only baseline is the primary adversary.

#### Stage 5: Evaluate slices

Report:

| Slice | Why |
|---|---|
| all held-out examples | primary score |
| teacher-agreement examples | checks consensus damage |
| top-1 disagreement examples | actual Eklavya target zone |
| high-confidence conflict examples | tests calibration |
| Qwen-only-correct predicted cases | tests whether Qwen adds value |
| Smol-only-correct predicted cases | tests whether router mostly copies Smol |

The router earns the name only if it improves the disagreement slice without damaging consensus cases.

### What The Literature Actually Says

The literature does not hand this project a solved Eklavya router. It says:

1. Equal averaging is often too naive.
2. Teacher weighting can be instance-specific.
3. Ground-truth labels can improve teacher reliability estimation.
4. Meta-weight networks and RL policies are possible but add complexity.
5. Bad teacher predictions can mislead the student.

That supports a calibrated router, not a post-hoc oracle router.

### NARRATIVE ATTACK

**Strongest "that's obvious" dismissal:** Your router used labels to learn which teacher was right, so it is supervised model selection, not discovery.

**Strongest "that's trivial" dismissal:** Confidence weighting or a static per-benchmark teacher rule explains the result.

**What would make the narrative hard to kill:** A precommitted, low-capacity, cross-fitted, label-free-at-inference router beats static, uniform, confidence, random, and shuffled routers, with positive disagreement-slice lift and no held-out label leakage.

### Attack On The Next Defense

The next defense will say a learned meta-router is more powerful. It is also easier to overfit. The terminal test needs a router a hostile reviewer can audit, not a small neural net that learns local artifacts.

---
## Iteration 123: Can Token-Level Results Transfer Back To Byte-Native?

### Steelman

If SmolLM2-135M shows a real Eklavya residual, it helps the byte-native dream in one narrow but important way:

```text
It proves the protocol has value when the student is capable.
```

That would change the byte-native problem from:

```text
Maybe Eklavya itself is empty.
```

to:

```text
Eklavya works somewhere; build or initialize a byte student that can receive it.
```

The protocol can be partially ported because the B14 interface can be choice-level:

```text
context + choices -> teacher choice distributions -> student choice distribution loss
```

That is tokenizer-independent. The teacher can score choices with BPE NLL, the byte student can score choices with byte NLL, and the distillation target can live over MCQ choices rather than teacher vocabulary tokens. This is exactly the part of the current `margin_distillation.py` infrastructure that remains reusable: forced-choice scoring, pairwise margins, and benchmark-facing accuracy.

There is also external support for byte models in principle. ByT5 shows token-free byte-to-byte models can be competitive and robust. MEGABYTE shows a patch/global byte architecture can make long byte sequences more tractable. BLT shows byte-level LLMs can match tokenization-based LLM performance at scale with dynamic patching and improved robustness. Cross-tokenizer KD work such as ULD and MultiLevelOT shows that tokenizer mismatch is a known, attackable distillation problem rather than a permanent wall.

### Attack

Token-level success does not transfer the learned router to byte-native.

The SmolLM2 student has:

- BPE tokenizer;
- a pretrained token embedding space;
- a Transformer decoder coordinate system;
- 2T tokens of pretraining;
- likely family overlap with SmolLM2-360M teacher.

The byte student has:

- raw byte interface;
- patch-local/global architecture;
- weaker benchmark function;
- no BPE logit space;
- already failed short-budget label generalization.

A router trained on SmolLM2-135M student uncertainty may not mean anything for Wide7. Student gap features are student-specific. Teacher-student KL is tokenizer/scorer-specific. Even choice-level distributions can shift because a byte LM's NLL is sensitive to byte length and formatting in different ways.

So the byte return cannot be:

```text
Copy the learned SmolLM2 router into S0.
```

It must be:

```text
Reuse the protocol design and teacher reliability features, then recalibrate on a byte student that has already passed label-only capacity.
```

### Concrete Byte-Return Paths

#### Path A: Functional choice-level return

This is the cheapest legitimate return.

1. Build or select a byte student that passes label-only MCQ capacity.
2. Use B14's teacher distributions over choices as tokenizer-independent targets.
3. Train byte student with label CE + routed KD over choices.
4. Compare against byte label-only, byte single-teacher KD, byte static teacher, byte uniform ensemble, and byte random route.

This does not solve token-level pretraining transfer. It only ports the functional Eklavya objective.

#### Path B: Cross-tokenizer bridge

Use ULD or MultiLevelOT-style alignment to distill token-level teacher logits into a byte/token-mismatched student:

```text
BPE teacher distributions -> optimal-transport aligned token/byte-span targets -> byte student
```

This is a real research project. It should be promoted to its own architecture line with baselines, not treated as a small repair.

#### Path C: Byte model rebirth

Train a new byte model from scratch or continued pretraining with the Eklavya protocol embedded:

```text
teacher-generated curriculum + choice/ranking distillation + byte CE
```

This is expensive and only justified if B14 produces strong protocol evidence. Otherwise it is another speculative architecture rebuild.

#### Path D: Hybrid byte interface over pretrained token core

Use a pretrained token core and learn a byte front-end/adapter or byte-to-token latent bridge. This may preserve some tokenizer-universal interface benefits, but it is no longer pure byte-native Sutra. It is a pragmatic democratization engineering path.

### What Token-Level Success Would And Would Not Prove

| Result | Proves | Does not prove |
|---|---|---|
| SmolLM2 Eklavya residual | Protocol can add value on a capable small student. | Byte-native can absorb it. |
| Learned SmolLM2 router works | Teacher reliability features can be useful. | Router transfers to byte student unchanged. |
| Single-teacher KD fails but routing passes | Disagreement matters in token setting. | Cross-tokenizer alignment is solved. |
| Token student beats best teacher | Strong small-model distillation event. | Sutra architecture is vindicated. |

### NARRATIVE ATTACK

**Strongest "that's obvious" dismissal:** Success on SmolLM2 tells us pretrained BPE students can exploit teacher labels. It says nothing about raw byte students.

**Strongest "that's trivial" dismissal:** The byte dream was abandoned, and the project is now a small-model PEFT/KD project.

**What would make the narrative hard to kill:** Token-level Eklavya residual must be followed by a byte-return plan that starts with a new label-capable byte student, uses tokenizer-independent choice-level targets first, and only then attempts cross-tokenizer logit/representation alignment.

### Attack On The Next Defense

The next defense will say the Vision allows substrate pivots. True. But if the byte substrate is dropped permanently, the public story must change. Do not pretend a token-level result vindicates a byte-native architecture.

---

## Iteration 124: What Would Make The Narrative Survive "That's Just Fine-Tuning"?

### Steelman

The dismissal is severe but not unbeatable:

```text
You fine-tuned a small model with teachers and it got slightly better. That's engineering.
```

The way to beat it is not by insisting the method has a special name. The way to beat it is by showing a property ordinary fine-tuning does not explain.

Candidate properties:

| Property | Why it matters |
|---|---|
| Residual over label-only | Shows teachers add value beyond labels. |
| Residual over single-teacher KD | Shows multi-teacher structure matters. |
| Residual over static/ensemble teachers | Shows routing matters. |
| Disagreement-slice lift | Shows gains come from teacher conflict, not consensus. |
| Sample efficiency | Shows protocol learns more from fewer labels. |
| Transfer | Shows not just local benchmark tuning. |
| Robustness | Shows routing does not overfit length/format artifacts. |
| Teacher-removal retention | Shows student internalized the signal. |
| Inference efficiency | Shows one small student replaces multiple teachers. |

The strongest narrative would be:

```text
A 135M open student, trained on the same tiny supervision budget, captures a large fraction of a multi-teacher oracle gap and beats all ordinary baselines, especially where teachers disagree, without needing the teachers at inference.
```

That is not "just fine-tuning." That is a measured learning mechanism.

### Attack

The current >=3pp bar is not enough for a moonshot.

On 48 held-out examples per benchmark, two examples are +4.17pp. A two-example movement on two small slices can be real enough for a continuation decision but not enough for a public paradigm claim. The repo's invariant says "paradigm-shifting or nothing." B14's >=3pp criterion is an admission gate, not a moonshot gate.

The moonshot narrative also cannot rest on aggregate accuracy alone. A router could gain by exploiting benchmark identity:

```text
HellaSwag -> trust SmolLM2
PIQA -> trust Qwen
ARC-Easy -> trust SmolLM2
```

That might be useful, but it is not the deep Eklavya thesis. The thesis says teacher disagreements contain transferable invariants and student gaps can be addressed selectively. Therefore the bar must include disagreement-local evidence.

### Concrete Bars

#### Minimum continuation bar

Continue Eklavya only if all are true:

1. Eklavya beats the best non-Eklavya trained baseline by >=3pp aggregate.
2. Eklavya beats that baseline on >=2/3 benchmarks.
3. On 48-example benchmark slices, each per-benchmark pass is at least +2 examples (+4.17pp).
4. Eklavya beats label-only, SmolLM2-only KD, Qwen-only KD, static per-benchmark teacher, uniform/entropy teacher mix, random routing, and shuffled routing.
5. Eklavya improves the top-1 teacher-disagreement slice.
6. Consensus-slice accuracy does not regress by more than 1 example per benchmark.
7. Held-out labels are untouched by router training and hyperparameter choice.

This justifies continuing the protocol. It is not yet public moonshot evidence.

#### Strong Eklavya bar

Call it strong only if:

1. Eklavya beats the strongest non-Eklavya baseline by >=5pp aggregate.
2. It passes all 3 benchmarks by at least +2 examples each, or 2/3 benchmarks by at least +3 examples each.
3. It captures >=50% of the local oracle gap over the best single teacher. The B12 oracle gap was +6.3pp aggregate, so this means about +3.15pp or more over the best single-teacher achievable baseline.
4. The disagreement-slice gain is larger than the consensus-slice gain.
5. A second seed or bootstrap/paired resampling does not flip the conclusion.
6. The gain survives a matched full fine-tune check if LoRA is the binding arm.

This is meaningful protocol evidence.

#### Moonshot candidate bar

Call it a moonshot candidate only if:

1. The 135M student after Eklavya matches or beats the best available 360M/600M teacher policy on aggregate held-out accuracy, or beats it by >=3pp on >=2/3 benchmarks.
2. It does so with one small student at inference, not an ensemble.
3. It beats all ordinary fine-tuning/KD/routing controls.
4. The result transfers to at least one not-tuned benchmark or task family.
5. The protocol demonstrates sample efficiency: same score with <=50% of the labels of label-only CE, or >=5pp over label-only at the same label budget.
6. A byte-return or tokenizer-robust follow-up has a concrete positive gate.

This would make people question efficient learning assumptions. Anything below that is engineering, possibly useful engineering.

### NARRATIVE ATTACK

**Strongest "that's obvious" dismissal:** A small public model moved by a few examples on tiny held-out slices after supervised training.

**Strongest "that's trivial" dismissal:** Your gain is within the space of PEFT hyperparameters, seed variance, and teacher ensemble heuristics.

**What would make the narrative hard to kill:** Eklavya must deliver a robust residual over the strongest non-Eklavya baseline, capture a large fraction of the oracle disagreement gap, improve disagreement cases specifically, transfer outside the tuned triad, and leave one cheap student at inference.

### Attack On The Next Defense

The next defense will say >=3pp was precommitted. Yes. It was precommitted as a terminal survival gate. It was never enough to call the result breathtaking.

---
## Iteration 125: Hostile Reviewer Reads The Full Repo Including Token-Level Pivot

### Steelman

A fair hostile reviewer would respect the process.

They would see:

- the Vision names clear external targets;
- multiple failed directions were documented instead of hidden;
- `FAIL_SCAFFOLD` tested the simplest possible label-only capacity condition;
- `FAIL_S0_CAPACITY` removed the "wrong student" excuse by testing Wide7;
- `UPGRADE_TEACHER` corrected a weak teacher choice;
- `PASS_DISAGREEMENT` found real oracle headroom;
- FMD was not consumed on an unqualified student;
- check-in #12 makes the token pivot terminal rather than open-ended.

They would also respect the phrase:

```text
mechanism control, not manifesto proof
```

That is the honest way to use SmolLM2-135M.

### Attack

Their honest summary would still be brutal:

```text
This is a rigorous negative-results repo that repeatedly failed to turn proxy signals into benchmark function. The byte-native architecture failed label-only capacity in both a tiny scaffold and the strongest available trained byte checkpoint. The one positive result is a label-anchored oracle disagreement ceiling over existing teachers. The project is now importing a heavily pretrained BPE student to test whether the KD/routing protocol has any value when the student is already competent.
```

They would attack:

| Attack | Evidence |
|---|---|
| Goalpost movement | Byte-native Sutra target demoted after failure; SmolLM2 imported. |
| External compute laundering | SmolLM2-135M used 2T-token/64-H100 pretraining. |
| No working Sutra student | MarginStudent and Wide7 failed capacity gates. |
| No proven router | PASS_DISAGREEMENT used labels to define usefulness. |
| Teacher diversity weak | SmolLM2 teacher/student share family; Qwen is weak on ARC. |
| Tiny eval slices | 48 held-out examples per benchmark makes pp claims granular. |
| Benchmark engineering risk | Train-safe MCQ slices can reward format and length artifacts. |
| Baseline burden high | PEFT, full fine-tune, KD, static ensembles, and confidence routing all exist. |
| Novelty crowded | Multi-teacher weighting and meta-routing have prior art. |
| Public claim absent | No public validation benchmark result yet beats small-model champions. |

They would respect the negative results, not the model results.

The one thing they would say the project should have done differently from the start:

```text
Run label-only capacity gates and token-level mechanism controls before inventing multiple byte-native objectives.
```

The project spent six objective-kill cycles discovering a student-capacity problem. A more ruthless start would have run:

```text
1. Can the intended student learn labels?
2. Can any competent same-size student exploit Eklavya beyond label-only?
3. Only then build the byte-native substrate.
```

That is the lesson to preserve.

### What They Would Demand From B14

They would demand:

1. A locked held-out split.
2. No held-out labels in router training, hyperparameter selection, or threshold tuning.
3. Matched data, steps, trainable parameters, and evaluation code across arms.
4. Full baseline board from Iteration 121.
5. Disagreement-slice and consensus-slice reporting.
6. Exact count deltas, not only percentage points.
7. A written rule that label-only improvement without Eklavya residual is ordinary fine-tuning and triggers pivot.

### NARRATIVE ATTACK

**Strongest "that's obvious" dismissal:** The project discovered that weak students fail and competent pretrained students are easier to tune.

**Strongest "that's trivial" dismissal:** The surviving idea is multi-teacher KD with a router, which is already a known family of methods.

**What would make the narrative hard to kill:** B14 must produce a clean residual that ordinary fine-tuning, single-teacher KD, static teacher selection, generic ensemble weighting, and random routing cannot explain.

### Attack On The Next Defense

The next defense will say the repo's kill discipline is itself impressive. It is. But process is not the moonshot. The terminal experiment must deliver a working residual or kill Eklavya as the mainline.

---

## Iteration 126: Final Decision Framework - What Constitutes Eklavya Evidence?

### Steelman

The terminal decision board from check-in #12 is directionally correct:

| Result | Action |
|---|---|
| SmolLM2 mechanism shows Eklavya residual >=3pp | Eklavya survives. |
| Label-only improves but no Eklavya residual | Eklavya is ordinary fine-tuning. |
| Nothing improves | Both dead. Full moonshot pivot. |

This is the right shape. It just needs stricter definitions so B14 cannot be interpreted after the fact.

### Attack

"Eklavya residual >=3pp" is underspecified.

Residual over what?

- zero-shot?
- label-only?
- single-teacher KD?
- best teacher?
- uniform ensemble?
- confidence router?
- random router?

If the answer is not "all of them," the hostile reviewer has an opening.

Which split?

- train-safe local held-out?
- public validation?
- cross-benchmark transfer?

If held-out labels touch router design, the result is invalid.

Which metric?

- accuracy?
- margins?
- disagreement-slice accuracy?
- aggregate?

Accuracy alone can hide route damage. Margins alone can hide top-1 failure.

Therefore B14 needs an evidence hierarchy.

### Evidence Hierarchy

#### Level 0: Validity prerequisites

No decision token is legal unless all are true:

1. Same train and held-out examples across all arms.
2. Same scoring semantics across all arms.
3. Same adaptation budget across trained arms.
4. Same trainable-parameter regime across trained arms, or explicit LoRA/full comparison.
5. Held-out labels never used for router training, threshold selection, early stopping, teacher choice, or hyperparameter tuning.
6. Results report exact correct counts and percentage points.
7. Results report aggregate, per-benchmark, disagreement-slice, and consensus-slice scores.

If Level 0 fails:

```text
INVALID_EKLAVYA_TEST
```

Rerun or do not claim anything.

#### Level 1: Student capacity

Label-only must improve over zero-shot:

```text
PASS_TOKEN_STUDENT_CAPACITY
  label-only improves aggregate by >=2pp and improves >=2/3 benchmarks by at least +1 held-out example each.
```

If label-only does not improve:

```text
FAIL_TOKEN_STUDENT_CAPACITY
```

Interpretation:

```text
SmolLM2-135M under this tiny adaptation protocol did not function as a useful student. Because this is the terminal Eklavya test, do not open another byte-native FMD repair on this evidence.
```

The supervisor may decide separately whether to try SmolLM2-360M as a different student, but that is no longer the promised B14 terminal test.

#### Level 2: Ordinary fine-tuning

If label-only improves and Eklavya does not beat the strongest non-Eklavya trained baseline by more than 1pp aggregate:

```text
ORDINARY_FINE_TUNING
```

Action:

```text
Pivot Eklavya out of mainline. The protocol added no meaningful value beyond ordinary adaptation.
```

If single-teacher KD improves but Eklavya does not beat it:

```text
ORDINARY_KD
```

Action:

```text
Do not call it Eklavya. At most, keep single-teacher KD as engineering.
```

#### Level 3: Marginal Eklavya

If Eklavya beats some baselines but fails the strict residual bar:

```text
MARGINAL_EKLAVYA
```

Concrete cases:

- beats single-teacher KD but not label-only;
- beats label-only but not static per-benchmark best teacher;
- beats static teacher but not uniform/entropy ensemble;
- beats all baselines by <3pp aggregate;
- passes only 1/3 benchmarks;
- gains appear only outside the teacher-disagreement slice;
- random or shuffled router is within 1pp aggregate.

Action:

```text
Do not continue as a moonshot mainline. Write it as engineering evidence or pivot.
```

Because this is terminal, "marginal" does not earn another open-ended repair cycle.

#### Level 4: Minimum continuation evidence

Eklavya survives only if:

```text
PASS_EKLAVYA_MECHANISM
```

All required:

1. Eklavya beats the strongest non-Eklavya trained baseline by >=3pp aggregate.
2. Eklavya beats that baseline on >=2/3 benchmarks.
3. Each per-benchmark pass is at least +2 held-out examples if using n=48 per benchmark.
4. Eklavya beats label-only CE.
5. Eklavya beats SmolLM2-360M single-teacher KD.
6. Eklavya beats Qwen single-teacher KD if Qwen contributes to the router.
7. Eklavya beats static per-benchmark best-teacher imitation.
8. Eklavya beats uniform/entropy teacher mixing.
9. Eklavya beats random and shuffled routing by >=3pp aggregate.
10. Eklavya improves the teacher-disagreement slice.
11. Consensus-slice regression is <=1 held-out example per benchmark.

Action:

```text
Eklavya protocol survives. Design byte-return path or openly accept a token-level Eklavya identity. No public moonshot claim yet.
```

#### Level 5: Strong Eklavya evidence

Strong evidence requires:

```text
STRONG_EKLAVYA
```

All required:

1. >=5pp aggregate over the strongest non-Eklavya trained baseline.
2. Passes all 3 benchmarks by at least +2 held-out examples each, or passes 2/3 benchmarks by at least +3 examples each.
3. Captures >=50% of the local oracle gap over the best single-teacher policy.
4. Disagreement-slice gain exceeds consensus-slice gain.
5. Result survives a second seed or paired/bootstrap resampling.
6. Result survives a matched full fine-tune check if LoRA is binding.
7. No extra data advantage over baselines.

Action:

```text
Continue Eklavya as a serious protocol line. Next required artifact is a byte-return or tokenizer-robust transfer design with precommitted gates.
```

#### Level 6: Moonshot candidate

Moonshot candidate requires:

```text
MOONSHOT_CANDIDATE
```

All required:

1. A 135M-class student after Eklavya matches or beats the best available 360M/600M teacher policy on aggregate held-out accuracy, or beats it by >=3pp on >=2/3 benchmarks.
2. It beats all ordinary adaptation, KD, ensemble, static-routing, and random-routing controls.
3. It transfers to at least one not-tuned benchmark or task family.
4. It shows sample efficiency: same score with <=50% labels versus label-only, or >=5pp over label-only at the same label budget.
5. It keeps inference cheap: one student, no teacher ensemble at inference.
6. It either returns to byte-native with a positive label-capacity + Eklavya residual result, or the project explicitly updates identity from Sutra byte-native to token-level Eklavya.

Action:

```text
Promote to public moonshot candidate. Still require public-scale validation before claiming paradigm shift.
```

### Final B14 Decision Table

| B14 outcome | Token | Action |
|---|---|---|
| Validity prerequisites fail | `INVALID_EKLAVYA_TEST` | Rerun or make no claim. |
| Label-only fails to improve | `FAIL_TOKEN_STUDENT_CAPACITY` | Terminal test failed; pivot main moonshot. |
| Label-only improves; Eklavya <= best baseline +1pp | `ORDINARY_FINE_TUNING` or `ORDINARY_KD` | Eklavya mainline dead. |
| Eklavya beats one weak baseline only | `MARGINAL_EKLAVYA` | No mainline continuation. |
| Eklavya >=3pp aggregate over all strict baselines and >=2/3 benchmarks | `PASS_EKLAVYA_MECHANISM` | Protocol survives; byte-return or token identity decision. |
| Eklavya >=5pp aggregate, robust, disagreement-local, captures >=50% oracle gap | `STRONG_EKLAVYA` | Continue protocol line seriously. |
| 135M student beats strongest teacher policy and transfers | `MOONSHOT_CANDIDATE` | Public validation path opens. |

### What Constitutes Eklavya Evidence?

Minimum:

```text
Not "student improved."
Not "teacher KD improved."
Not "multi-teacher average improved."
Only: disagreement-aware routing produced residual held-out gains over all ordinary baselines under matched budget and no held-out label leakage.
```

Moonshot:

```text
A small open student internalizes teacher disagreement so effectively that it beats stronger teacher policies or reaches their level at far lower inference cost, with transfer and sample-efficiency evidence.
```

Just engineering:

```text
Any result explained by label-only fine-tuning, single-teacher KD, static teacher choice, uniform/confidence ensemble, PEFT hyperparameters, or small held-out variance.
```

### NARRATIVE ATTACK

**Strongest "that's obvious" dismissal:** You adapted a pretrained model on MCQ labels and teacher outputs. That is ordinary downstream training.

**Strongest "that's trivial" dismissal:** Your "router" is a teacher ensemble weighting heuristic with insufficient baselines.

**What would make the narrative hard to kill:** The final experiment must show that disagreement-aware routing creates a residual no ordinary baseline can explain, on the exact cases where disagreement matters, with enough magnitude to survive count-level and seed-level scrutiny.

### Attack On The Next Defense

The next defense will say a marginal positive result deserves one more repair. No. This is the terminal Eklavya test. Marginal means the protocol becomes an engineering note or dies as the moonshot mainline.

---

## Batch 18 Final Verdict

The SmolLM2-135M pivot is justified only as a terminal mechanism control.

It is not fair compute comparison to S0/Wide7. It is not byte-native evidence. It is not a Sutra architecture result. It is a way to answer the one question still alive after `FAIL_SCAFFOLD` and `FAIL_S0_CAPACITY`:

```text
Does disagreement-driven multi-teacher learning add residual value in a competent small student beyond ordinary fine-tuning and ordinary KD?
```

If B14 says yes under the strict Level 4 bar, Eklavya survives as a protocol and the next decision is byte-return versus token-level identity.

If B14 says no, the KD moonshot is dead. Do not spend the preserved FMD repair shot on another weak byte student. Pivot the moonshot.

Final hostile statement:

```text
The project has earned one final Eklavya test, not one final excuse. SmolLM2 is allowed to save the protocol only by beating the strongest ordinary baselines. It is not allowed to save the narrative by being easier to fine-tune.
```
