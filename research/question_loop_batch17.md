# QUESTION LOOP - Batch 17: The Architecture Question

Date: 2026-07-07

Iterations: 113-119

## Grounding

I read the requested local context in order:

1. `research/VISION.md`
2. `research/dual_loop_supervisor_checkin_11.md`
3. `research/dual_loop_supervisor_checkin_10.md`
4. `research/question_loop_batch16.md`
5. `research/work_loop_batch12.md`
6. `tmp_margin_distillation_b12/scaffold_capacity_50/scaffold_capacity.json`
7. `tmp_margin_distillation_b12/teacher_audit/teacher_margin_audit.json`
8. `tmp_margin_distillation_b12/disagreement_audit_smol/disagreement_density_audit.json`
9. `code/margin_distillation.py`
10. `code/s0_training.py`, `code/s0_option_c_training.py`, `code/s0_architecture.py`, and `code/s0_configs.py`

No GPU runs, benchmark runs, training runs, or new experiments were performed.
I did run a CPU-only parameter-count print for the existing S0 configs.

External primary sources checked for the literature-dependent parts:

- MEGABYTE: https://arxiv.org/abs/2305.07185
- ByT5: https://arxiv.org/abs/2105.13626
- Byte Latent Transformer: https://arxiv.org/abs/2412.09871
- Multi-Level Optimal Transport for cross-tokenizer KD: https://arxiv.org/abs/2412.14528
- LoRA: https://arxiv.org/abs/2106.09685
- T-Few / parameter-efficient few-shot fine-tuning: https://arxiv.org/abs/2205.05638
- ADAPET / few-shot pattern exploiting training: https://arxiv.org/abs/2103.11955
- FLAN / instruction tuning: https://arxiv.org/abs/2109.01652

## Binding Facts Entering Batch 17

The B11 `MarginStudent` is dead as an objective-test substrate.

It failed the simplest capacity probe:

| Benchmark | Untrained | Label-CE trained | Delta | Mean margin delta |
|---|---:|---:|---:|---:|
| HellaSwag | 20.83% | 20.83% | +0.00pp | -0.0932 |
| PIQA | 56.25% | 54.17% | -2.08pp | +0.0059 |
| ARC-Easy | 22.92% | 14.58% | -8.33pp | -0.1210 |

The student config in the binding raw artifact:

```text
frozen codec: 8.36M params, codec_dim=256, patch_size=4
trainable student: d_model=256, n_layers=2, n_heads=4, n_kv_heads=4
decoder: decoder_dim=256, decoder_layers=1, decoder_heads=4
training: label_only_choice_ce, 50 steps, batch_examples=6, lr=2e-4
```

The exact code path confirms this is not S0. It is a frozen codec feeding a
randomly initialized shallow global reasoner and byte decoder.

The real S0/Wide7 configs are materially different:

| Config | Params | Patch | Width | Layers | Heads/KV |
|---|---:|---:|---:|---:|---:|
| S0 P4 | 121.7M | 4 | 576 | 30 | 9/3 |
| S0 P8 | 124.1M | 8 | 576 | 30 | 9/3 |
| D640 | 145.3M | 4 | 640 | 30 | 10/2 |
| D768 | 156.2M | 4 | 768 | 22 | 12/4 |
| Wide7 | 121.7M | 4 | 1152 | 7 | 18/6 |

The teacher situation is mixed:

| Benchmark | Qwen acc | Positive margin | Confident wrong |
|---|---:|---:|---:|
| HellaSwag | 49.5% | 49.5% | 49.5% |
| PIQA | 67.5% | 67.5% | 28.5% |
| ARC-Easy | 34.0% | 34.0% | 65.5% |

The one real positive result is disagreement fuel:

| Benchmark | Qwen | SmolLM2-360M | Useful disagreement | Oracle ceiling | Oracle gap |
|---|---:|---:|---:|---:|---:|
| HellaSwag | 49.5% | 58.5% | 17.0% | 62.5% | +4.0pp |
| PIQA | 67.5% | 71.5% | 20.0% | 79.5% | +8.0pp |
| ARC-Easy | 34.0% | 54.5% | 34.5% | 61.5% | +7.0pp |
| Aggregate | 50.3% | 61.5% | 23.8% | 67.8% | +6.3pp |

## Hostile Batch Verdict Up Front

The current `MarginStudent` failed because the project tried to perform
functional benchmark learning through the weakest possible birth story:

```text
frozen small codec states -> random shallow reasoner -> random tiny decoder
```

That is not a trainable student in the sense needed for Eklavya. It is a
low-bandwidth adapter scaffold with no inherited semantic basis.

But the failure does not yet kill byte-native Sutra. It kills this shortcut:

```text
freeze a small codec, attach a tiny random global module, then expect 50-step
teacher or label supervision to create MCQ competence
```

The next question is not "which loss should be used?"

It is:

```text
Can a pretrained byte-native LM with real language-modeling capacity, S0 or
Wide7, learn benchmark-facing MCQ discrimination under direct labels?
```

If yes, the current failure is an adapter-scaffold failure. FMD can be tested on
S0 with the preserved repair shot.

If no, then byte-native at this size and training state is probably not the
right near-term vehicle. At that point, use SmolLM2-360M as a token-level
Eklavya mechanism control, not as Sutra proof. If even token-level Eklavya gives
only ordinary supervised fine-tuning, pivot the moonshot away from KD.

Decision token for this batch:

```text
PIVOT_STUDENT_NOT_MOONSHOT_YET
```

---

## Iteration 113: Why Did The MarginStudent Fail?

### Steelman

The charitable interpretation is that the B12 result is narrower than the
supervisor language makes it sound.

It proves:

```text
The exact B11 MarginStudent did not improve held-out MCQ accuracy after
50 supervised label-CE steps on 288 train-safe examples.
```

It does not prove:

```text
All byte-level models cannot learn MCQ.
All Sutra-style architectures are doomed.
All longer supervised fine-tuning would fail.
All teacher-margin methods are worthless.
```

The failed student is intentionally tiny. `code/margin_distillation.py` names it
a prototype and says it is not final S0. Architecturally it has:

- a frozen `CausalByteTransformer` codec;
- a trainable RMSNorm and projection;
- a 2-layer 256-dim `GlobalReasoner`;
- a 1-layer 256-dim `ByteDecoder`;
- random initialization for the trainable path;
- no native S0 byte encoder;
- no pretrained reasoner;
- no pretrained decoder.

That design is a shadow scaffold, not an LM. It asks the frozen codec to carry
enough semantic structure that a shallow random readout can quickly learn MCQ
choice ranking. The failure could therefore be caused by several local factors:

| Candidate factor | Charitable reading |
|---|---|
| Capacity | 2x256 global layers and a 1-layer decoder are too small for the benchmark function. |
| Initialization | The trainable path starts too far from a useful LM coordinate frame. |
| Interface | Frozen codec patch states are not a sufficient semantic interface for MCQ. |
| Budget | 50 steps over 288 examples is a short diagnostic, not full fine-tuning. |
| Scoring path | Byte-NLL continuation scores may be a poor readout for MCQ under this scaffold. |

The raw JSON shows gradients existed and the loss moved batch-by-batch. That
rules out a completely dead optimizer. But it does not isolate the failure.
The model may have learned something about the sampled training batches while
moving held-out margins in the wrong direction.

So the steelman is:

```text
FAIL_SCAFFOLD kills the B11 MarginStudent as an objective-test scaffold, but it
does not identify which architectural lesion is primary.
```

### Attack

The charitable reading is too forgiving if it becomes permission to repair the
same scaffold.

The B12 result is exactly the humiliating baseline the project needed. Labels
are the easiest possible teacher. If label CE over gold choices cannot move the
held-out benchmark function, then teacher margins, hard negatives, disagreement
routers, and FMD are downstream complexity.

The strongest diagnosis is not "50 steps were too few." It is:

```text
The student had no semantic birth.
```

The frozen codec may be useful for compression or byte alignment, but the
trainable path is random and shallow. A 2-layer module over frozen byte-patch
states is closer to a nonlinear probe than to a pretrained student. The
benchmark task requires using broad language priors. The B11 scaffold tries to
infer those priors from 288 MCQ examples or 50 shard-continuation examples.
That is not a serious learning geometry.

The failure also attacked margins, not only top-1 accuracy:

- HellaSwag held accuracy flat while margin delta moved negative.
- ARC-Easy regressed -8.33pp and margin delta moved negative.
- PIQA showed a tiny positive margin delta but negative accuracy.

This looks like the training signal perturbed a brittle prior rather than
installing competence.

### Which Factor Most Likely Killed It?

Ranked likelihood:

| Rank | Factor | Why |
|---:|---|---|
| 1 | Initialization / missing semantic basis | The trainable path is random and shallow; CBD's lesson is that small students need inherited coordinates. |
| 2 | Interface loss | A frozen 4-layer 256-dim codec may not expose MCQ-relevant semantic variables in a linearly/readily trainable form. |
| 3 | Trainable capacity | 2x256 global layers plus 1 decoder layer may be too small to reconstruct task function from codec states. |
| 4 | Scoring mismatch | Choice byte-NLL can be dominated by continuation/style priors unless the LM is already competent. |
| 5 | Budget alone | 50 steps is short, but a usable scaffold should have shown at least one clean positive held-out movement. |

The key phrase is "budget alone." More steps might improve this scaffold, but
if the project must spend real training budget to make this tiny adapter
capacity viable, it is no longer a cheap objective-testing scaffold. It becomes
a new architecture project, and S0/Wide7 is the better architecture project.

### Can We Diagnose Which?

Yes, but not by another teacher objective. The diagnostic tree should be:

| Test | What it separates |
|---|---|
| Train-set overfit report for the same B12 run | If train accuracy did not move, capacity/optimizer/readout is broken. If train moves but held-out fails, generalization/format is broken. |
| More-step label CE on MarginStudent | Separates "50 steps too few" from structural unlearnability, but should not reopen FMD. |
| Linear probe or MLP probe on frozen codec states | Tests whether codec states carry recoverable MCQ signal. |
| Unfreeze codec vs frozen codec | Tests interface bottleneck vs trainable-head bottleneck. |
| S0/Wide7 same scoring path | Tests whether pretrained byte LM birth fixes the issue. |
| Token-level SmolLM2 same data split | Tests whether the task/data budget itself is adequate for a real pretrained LM. |

Only two of these matter strategically now:

```text
S0/Wide7 same scoring path
SmolLM2 token-level control
```

The rest are postmortem diagnostics.

### What Survives

The broader Eklavya thesis survives only because `PASS_DISAGREEMENT` is real.
There is teacher complementarity to exploit.

### What Dies

This dies permanently:

```text
Use the B11 MarginStudent for future objective experiments.
```

### NARRATIVE ATTACK

**Strongest "that's obvious" dismissal:** You attached a tiny random module to
a frozen codec and discovered it is not a language model.

**Strongest "that's trivial" dismissal:** Labels did not help. No teacher loss
should be discussed on this scaffold again.

**What would make the narrative hard to kill:** Show that the real pretrained
S0 or Wide7 learns the same MCQ capacity check. Then the failure is clearly a
shadow-scaffold failure, not byte-native death.

### Attack On The Next Defense

The next defense will say the failed scaffold was only a prototype. Correct.
Then stop letting prototype failures consume objective-loop attention. Use the
real student or stop claiming student evidence.

---

## Iteration 114: Can S0/Wide7 Learn MCQ?

### Steelman

S0/Wide7 has a plausible path that MarginStudent did not.

S0 is a real byte-level LM:

- native byte encoder;
- patch-local mixer;
- nonlinear patch aggregator;
- causal global reasoner;
- byte decoder with cross-attention;
- 121.7M parameters for P4;
- trained to eval_bpb around the supervisor-stated 1.900;
- reported HellaSwag zero-shot around 26.3%.

Wide7 is the same parameter class with a different geometry:

```text
121.7M params, D=1152, 7 layers
```

This matters because the B11 failure may have been a "birth" failure. S0 and
Wide7 have at least learned a byte-level language-modeling basis. Fine-tuning a
pretrained LM on downstream tasks is normal. The literature broadly supports
that pretrained models can be adapted by full fine-tuning, PEFT, prompt/pattern
methods, and instruction tuning. LoRA shows that downstream adaptation can be
effective with far fewer trainable parameters than full fine-tuning. T-Few and
ADAPET show that few-shot or low-resource task adaptation can work when the
base model and task interface are favorable. FLAN shows that supervised
instruction mixtures can move zero-shot generalization.

Byte-level modeling is also not a dead literature branch. ByT5, MEGABYTE, and
BLT all support the feasibility of token-free or byte-patch models, with the
important caveat that successful byte models usually pay with more training,
careful architecture, or larger scale.

So the steelman:

```text
Supervised MCQ fine-tuning on S0/Wide7 is expected to work better than on the
MarginStudent because S0/Wide7 begins with a learned LM coordinate frame.
```

### Attack

"Expected to work better" is not the same as "expected to reach the moonshot
bar."

The S0 zero-shot number is weak:

```text
HellaSwag 26.3% at 121M
```

That is only slightly above four-way chance and far below the Vision's target
zone. A hostile reviewer will say:

```text
You are not fine-tuning a competent small model. You are trying to rescue a
weak byte LM with benchmark supervision.
```

The literature does not say that any small pretrained model can learn hard MCQ
benchmarks from 288 examples and 50 updates. The stronger few-shot results
usually rely on larger models, instruction-tuned bases, curated task templates,
or task mixtures. S0 is not instruction-tuned. It is a byte LM scout.

The fine-tuning objective also matters. There are at least three different
"MCQ training" claims:

| Method | What it proves | Risk |
|---|---|---|
| Label CE over `-choice_nlls` | The generative scorer can adapt to choose answers. | May overfit format or harm BPB. |
| Add a classification/ranking head | The representation contains task signal. | No longer pure LM scoring; weaker Sutra proof. |
| Full instruction-style SFT | The model can follow task formats. | May become ordinary benchmark tutoring. |

The B13 capacity check should start with label CE over the existing forced
choice scoring path because that matches the FMD decision path. But if it fails
at 50 steps, the conclusion should be bounded:

```text
S0 failed short-budget MCQ adaptation under this scoring path.
```

Not automatically:

```text
S0 can never fine-tune.
```

### How Many Examples And Steps?

For the kill gate, keep the short diagnostic:

```text
288 examples, 144 held-out, 50+ steps
```

That is the continuity check against B12. But a fair S0 MCQ fine-tuning attempt
would need a larger budget.

Pragmatic expectation:

| Purpose | Examples | Steps | Interpretation |
|---|---:|---:|---|
| Smoke capacity | 288 train-safe | 50-200 | Should show directionality if the scoring path is usable. |
| Overfit diagnostic | 288 train-safe | 500-1000 | Should overfit train if the model can absorb the objective. |
| Real task adaptation | 2k-20k across MCQ families | 1k-10k | Tests whether MCQ skill generalizes beyond tiny slices. |
| Moonshot evidence | Public/large held-out, strong baselines | repeated seeds or deterministic paired eval | Required before claiming anything public. |

Fifty steps should not be treated as a fair final fine-tune. It is a cheap
capacity gate. If S0 cannot even overfit 288 examples after a longer diagnostic,
the byte architecture or scoring path is in serious trouble.

### Required S0/Wide7 Capacity Protocol

Minimum:

1. Use the same train/eval split as B12.
2. Evaluate untrained checkpoint with the same forced-choice byte-NLL scorer.
3. Train label-only choice CE.
4. Report train accuracy and held-out accuracy.
5. Report gold-vs-best-wrong margin deltas.
6. Include random-label or shuffled-label control if budget allows.
7. Track BPB before/after to detect benchmark overfit damage.

Decision tokens:

```text
PASS_S0_CAPACITY
  Held-out +5pp on >=2/3 benchmarks, positive margin deltas, train movement clean.

MEMORIZATION_ONLY_S0
  Train moves strongly, held-out <+2pp.

FAIL_S0_SHORT_CAPACITY
  50-200 step diagnostic shows no held-out movement.

FAIL_S0_OVERFIT_CAPACITY
  500-1000 step diagnostic cannot even overfit train.
```

The last token is the dangerous one. It would indict the current byte scoring
interface much more strongly than the B12 MarginStudent failure.

### What Survives

S0/Wide7 is still the cheapest honest way to test whether byte-native can be
the student.

### What Dies

This dies:

```text
If S0 learns MCQ labels, that alone is Eklavya evidence.
```

No. That is ordinary supervised fine-tuning. Eklavya begins only when teacher
signals beat label-only under identical budget.

### NARRATIVE ATTACK

**Strongest "that's obvious" dismissal:** Pretrained models can be fine-tuned.
If S0 improves, you have rediscovered supervised task adaptation.

**Strongest "that's trivial" dismissal:** A 26.3% HellaSwag base is too weak to
carry a paradigm claim without a massive gain over baselines.

**What would make the narrative hard to kill:** S0 or Wide7 must first pass
label-only capacity, then teacher margins or disagreement must add residual
held-out gain beyond label-only and token-KD.

### Attack On The Next Defense

The next defense will say that direct MCQ training is not the manifesto. That
is true, and irrelevant. It is the admission test for whether the student can
receive any manifesto-shaped lesson.

---

## Iteration 115: The CBD Structural Lesson Revisited

### Steelman

CBD's local lesson remains severe:

```text
small-model benchmark function comes from inherited representation, not from
short clever post-hoc losses into a randomly born student.
```

The prior batch framed this correctly. CBD-style chain-init preserves a
coordinate system:

```text
large model -> intermediate model -> small model
```

The student is not asked to invent the whole feature basis from sparse
supervision. It is born inside an already functional representational family.

A byte-native chain-init would try to give Sutra the same advantage while
preserving the byte substrate:

```text
strong token teacher -> intermediate bridge -> byte-native S0/Wide7/Sutra
```

There are several conceivable bridge designs:

| Bridge | Mechanism |
|---|---|
| Sequence-level KD | Match teacher preference/NLL over full continuations or MCQ choices. |
| Span alignment | Map teacher BPE token spans to byte spans and supervise pooled states. |
| Cross-tokenizer optimal transport | Align teacher/student token distributions without one-to-one token matching. |
| Soft-DTW style alignment | Align sequences with different segmentation lengths. |
| Teacher-generated corpus | Use teacher outputs/rationales as byte-native text data, then train Sutra normally. |
| Intermediate byte model | Distill into a larger byte model first, then compress byte-to-byte. |

The literature confirms that cross-tokenizer KD is not impossible. Multi-Level
Optimal Transport exists specifically because same-tokenizer assumptions limit
ordinary KD; it aligns token and sequence distributions without requiring
token-by-token identity. More recent cross-tokenizer methods also treat
alignment as a core problem, not as a footnote.

So the steelman:

```text
Byte-native chain-init is theoretically possible, but it is an alignment
research program, not a weight-copy trick.
```

### Attack

The phrase "chain-init" can launder an unsolved problem.

Same-tokenizer chain-init has three advantages that byte-native Eklavya lacks:

1. Embedding rows and LM heads live in the same vocabulary space.
2. Intermediate hidden states correspond to similar token boundaries.
3. The student architecture often resembles the teacher architecture.

Those properties make distillation dense. Cross-tokenizer byte distillation
does not have them. A BPE token may correspond to one byte, many bytes, or a
semantic-ish subword merge. The teacher's next-token distribution is not a
next-byte distribution. A BPE logit vector cannot be copied into a byte logit
head.

The hostile reduction:

```text
Cross-tokenizer chain-init is not initialization. It is another distillation
objective with an alignment module.
```

That does not mean it is bad. It means it should not be expected to have CBD's
same advantage unless the alignment itself is proven.

The current repo's failures are exactly what happens when alignment is too
weak:

- coordinate inheritance produced compatibility proxies but not MCQ gains;
- FMD used sequence-level margins but did not transfer to benchmark function;
- the MarginStudent could not learn labels.

So "do CBD but byte-native" is not a small patch. It is a new core research
direction.

### Can You Chain From BPE To Byte?

At the function level: yes.

At the weight-coordinate level: generally no.

At the token-logit level: only through an alignment approximation.

The cleanest byte-native chain is probably not direct BPE-to-byte. It is:

```text
BPE teacher -> large/intermediate byte model -> smaller byte model
```

The first arrow is expensive because it solves cross-tokenizer transfer. The
second arrow is easier because it is byte-to-byte.

A cheaper near-term chain is:

```text
BPE teacher -> MCQ/continuation functional labels -> S0/Wide7
```

But that is FMD or supervised distillation, not CBD-style coordinate
inheritance.

### What Would Count As Byte-Native Chain Evidence?

Minimum admission evidence:

1. A byte student initialized or trained from a bridge beats byte CE-only
   pretraining at equal compute.
2. It beats label-only MCQ fine-tuning when evaluated on held-out MCQ.
3. It does not collapse byte BPB.
4. It approaches a token-chain baseline or beats it on openness/robustness
   constraints.
5. It shows teacher-retained gains after teacher removal.

Without those, "byte CBD" is just a name for a hard engineering project.

### What Survives

The CBD structural lesson survives:

```text
birth/initialization likely matters more than another loss.
```

### What Dies

This dies:

```text
Cross-tokenizer chain-init can be treated as CBD with byte inputs.
```

No. Same-tokenizer chain-init transfers coordinates. BPE-to-byte transfer must
learn a translation of coordinates before it can preserve them.

### NARRATIVE ATTACK

**Strongest "that's obvious" dismissal:** CBD works because it keeps the
teacher's coordinate system. Your byte student deliberately leaves that system.

**Strongest "that's trivial" dismissal:** Cross-tokenizer KD is already a known
alignment problem. Calling it chain-init does not solve it.

**What would make the narrative hard to kill:** A byte-native or byte-bridge
chain must beat both byte CE-only training and ordinary token-chain baselines
under clear compute and data accounting.

### Attack On The Next Defense

The next defense will say byte-native chain-init is the moonshot. Maybe. But
then it must be promoted from "repair tactic" to "main architecture research,"
with its own baselines and kill gates.

---

## Iteration 116: SmolLM2-360M As Student Baseline

### Steelman

SmolLM2-360M is the shortest path to a functional Eklavya mechanism test.

The B12 audit already showed:

```text
SmolLM2-360M aggregate: 61.5%
Qwen3-0.6B aggregate: 50.3%
oracle over both: 67.8%
oracle gap over best teacher: +6.3pp
```

That means SmolLM2 is not only a stronger teacher; it is also a plausible
student or base model for a fast supervised/KD experiment.

The steelman for using it:

1. It answers whether disagreement fuel can be turned into actual model gain
   when the student is trainable.
2. It supplies a real pretrained semantic basis.
3. It avoids the byte-scaffold bottleneck.
4. It produces a calibration result quickly.
5. It can expose whether FMD/disagreement has residual value beyond label-only.

If a token-level SmolLM2 student cannot use teacher margins or disagreement,
the Eklavya KD thesis becomes much weaker. If it can, then the project learns:

```text
The protocol has life; the byte student is the bottleneck.
```

That is valuable information.

### Attack

SmolLM2-360M is also the easiest path to baseline laundering.

A hostile reviewer will say:

```text
You failed to train your own byte-native 121M student, then switched to a
stronger pretrained BPE model and renamed fine-tuning as Eklavya.
```

That attack is mostly fair unless the claim is narrowed.

SmolLM2 does not prove Sutra:

- it is not byte-native;
- it is not 121M;
- it is not trained under this repo's compute constraints;
- it inherits external tokenizer and pretraining choices;
- it already starts above all local students.

Fine-tuning SmolLM2 with labels or margins could produce numbers, but the first
interpretation would be ordinary task adaptation. Even a router gain could be
engineering unless it beats strong baselines:

| Baseline | Why it matters |
|---|---|
| SmolLM2 zero-shot | Shows raw base ability. |
| SmolLM2 label-only fine-tune | Separates labels from teacher value. |
| SmolLM2 single-teacher KD | Separates multi-teacher value from any KD. |
| Best-teacher imitation | Separates routing from copying the strongest model. |
| Random/shuffled margins | Detects regularization/data exposure effects. |

The correct role for SmolLM2 is:

```text
mechanism control, not manifesto proof
```

### Does It Serve The Manifesto?

Conditionally.

It serves the manifesto if it answers a structural question:

```text
Can disagreement-driven multi-teacher learning produce retained gains in a
small open student beyond ordinary supervised fine-tuning?
```

It does not serve the manifesto if it becomes:

```text
Use an existing small BPE model and tune it until benchmark numbers look good.
```

The democratization moonshot is not byte purity. The Vision says byte-level is
a means. But the manifesto still requires a result that changes assumptions
about efficient learning. A SmolLM2 experiment changes assumptions only if the
Eklavya mechanism adds something nontrivial.

### What Would Make SmolLM2 Eklavya Evidence?

Admission evidence:

```text
SmolLM2 + disagreement/FMD beats SmolLM2 label-only and single-teacher KD
on held-out MCQ slices under identical budget.
```

Stronger evidence:

```text
The learned router or margin policy transfers across benchmarks or teachers,
and gains survive when labels are removed from routing decisions.
```

Moonshot evidence:

```text
A smaller or cheaper open student reaches or beats stronger models through
multi-teacher disagreement with much less compute, robust controls, and public
validation.
```

SmolLM2 can start the evidence chain. It cannot finish it alone.

### What Survives

SmolLM2 is the right token-level control lane.

### What Dies

This dies:

```text
SmolLM2 fine-tuning equals Sutra progress.
```

No. It is Eklavya protocol evidence only if teacher/disagreement residuals beat
ordinary fine-tuning.

### NARRATIVE ATTACK

**Strongest "that's obvious" dismissal:** A stronger pretrained model gets
better when fine-tuned.

**Strongest "that's trivial" dismissal:** You moved from architecture research
to benchmark engineering.

**What would make the narrative hard to kill:** SmolLM2 must demonstrate a
multi-teacher/disagreement residual over label-only and single-teacher KD, then
that residual must guide the next byte-native student design.

### Attack On The Next Defense

The next defense will say the Vision permits mechanism pivots. Correct. But it
does not permit relabeling an easier baseline as the moonshot.

---

## Iteration 117: Is Byte-Native A Load-Bearing Constraint?

### Steelman

Byte-native has real strategic value.

The Vision's byte argument is not aesthetic. It targets:

1. tokenizer lock-in;
2. cross-architecture transfer;
3. multilingual/noisy robustness;
4. simpler open substrate;
5. democratized reproducibility;
6. no proprietary vocabulary moat.

The literature supports parts of this. ByT5 argues token-free models reduce
preprocessing debt and improve robustness to noise. MEGABYTE and BLT show that
byte-patch models can be viable when architecture and scale are right.

Dropping byte-native risks losing the repo's most distinctive architectural
claim. A token-level student makes the project compete with the entire PEFT/KD
ecosystem on its home field. That is a hard place to be novel.

So the steelman:

```text
Byte-native is not sacred, but it is one of the few properties that could make
Sutra more than another fine-tuned small model.
```

### Attack

Byte-native is only load-bearing if it carries load.

The Vision explicitly says:

```text
the architecture is a means to an end
```

The sacred outcomes are intelligence, improvability, democratized development,
data efficiency, and inference efficiency. If byte-native blocks all functional
evidence, loyalty to it violates the invariant.

A working token-level multi-teacher student is better than a non-working
byte-level student in one crucial sense:

```text
it can falsify or validate Eklavya as a learning protocol
```

The byte substrate can return later if the protocol is worth carrying back.
But if the protocol cannot produce residual gains even in an easy token-level
student, then byte-native is not the first bottleneck. Eklavya itself is.

The hostile reviewer will frame it this way:

```text
You are using byte-native as a moral shield for low performance.
```

That attack must be prevented by decision discipline:

- byte-native gets S0/Wide7 capacity check;
- byte-native gets FMD only after capacity pass;
- if byte-native fails capacity, token-level controls run;
- if token-level controls prove the mechanism, design a byte return;
- if token-level controls are boring, pivot away from KD.

### What We Lose If We Drop Byte-Native

| Loss | Severity |
|---|---|
| Tokenizer-universal student interface | High |
| Strongest architectural novelty | High |
| Direct Sutra identity | High |
| Robustness/no-tokenizer story | Medium-high |
| Single shared byte substrate for diverse teachers | Medium |
| Differentiation from ordinary PEFT/KD | High |

### What We Gain

| Gain | Severity |
|---|---|
| Immediate trainable semantic basis | High |
| Strong baselines and ecosystem tooling | High |
| Faster mechanism iteration | High |
| Cleaner distinction between protocol failure and byte failure | High |
| Better odds of positive functional evidence | High |

### Is The Manifesto Better Served By Token-Level?

For the next diagnostic: yes, if byte S0/Wide7 fails capacity.

For the final moonshot claim: not unless the token-level result proves a
mechanism that is more important than tokenization.

The correct statement:

```text
Byte-native is a preferred mechanism, not an invariant. It earns continued
mainline status only if S0/Wide7 can learn function.
```

### What Survives

Byte-native survives as a high-value constraint, not as a veto over evidence.

### What Dies

This dies:

```text
The project must stay byte-native even if byte-native remains nonfunctional.
```

No. That would optimize identity over democratization.

### NARRATIVE ATTACK

**Strongest "that's obvious" dismissal:** Tokenization-free modeling is useful,
but your byte model is not useful yet.

**Strongest "that's trivial" dismissal:** Dropping byte-native makes the work
look like standard small-model fine-tuning unless the teacher protocol itself
is novel and measured.

**What would make the narrative hard to kill:** Use token-level students as
controls to prove Eklavya residuals, then either return to byte-native with a
better initialization path or openly change the manifesto mechanism.

### Attack On The Next Defense

The next defense will say "byte-native is democratization." No. Democratization
is the outcome. Byte-native is one candidate mechanism for it.

---

## Iteration 118: What Would A Hostile Reviewer Say Now?

### Steelman

A hostile reviewer would not dismiss everything.

They would find one thing genuinely impressive:

```text
The project killed its own scaffold with the simplest possible supervised
capacity test and preserved the FMD repair shot instead of laundering it.
```

They would also respect `PASS_DISAGREEMENT`:

```text
23.8% useful disagreement and +6.3pp oracle gap over the best teacher is a real
signal that teacher complementarity exists.
```

The process is better than the result. The repo has shown:

- adversarial self-review;
- precommitted kill gates;
- raw artifacts;
- separation of admission evidence from moonshot evidence;
- willingness to kill favored mechanisms;
- an honest move from objective blame to architecture blame.

That is not enough for a moonshot claim, but it is enough to justify one more
high-information architecture decision.

### Attack

The same reviewer would summarize the current state brutally:

```text
This is a rigorous negative-results repo with no positive functional student
result. It has six killed directions, a proven-dead adapter scaffold, a weak
primary teacher, and one interesting no-training disagreement ceiling.
```

They would list the indictment:

| Attack | Evidence |
|---|---|
| No working student | `FAIL_SCAFFOLD`: 0/3 benchmarks improved after label CE. |
| No KD residual | FMD repair shot skipped; prior FMD prototype failed. |
| Weak primary teacher | Qwen is 34% on ARC-Easy with 65.5% confident-wrong. |
| Objective churn | KD, Brainseed, evidence-native, coordinate inheritance, FMD. |
| Byte model underperforms targets | S0 HellaSwag stated at 26.3%, target is SmolLM2-class. |
| Disagreement is only a ceiling | Useful disagreement uses labels to define usefulness. |
| Novelty not yet isolated | Pairwise margins, PEFT, KD, routing, and byte models all have prior art. |
| Benchmark-overfit risk | Small train-safe MCQ slices can reward format learning. |

Their honest one-paragraph review:

```text
The repo has found an interesting teacher-disagreement resource but has not
shown that any Sutra student can absorb it. The current scaffold is dead. The
next evidence must use a real pretrained student or stop. If S0/Wide7 cannot
learn labels, byte-native should be demoted from mainline. If a token-level
student can exploit disagreement, the protocol may survive but the Sutra
architecture claim does not.
```

### What Would They Recommend?

They would recommend a two-lane decision board, not another theory loop:

#### Lane 1: Byte-native capacity

```text
S0/Wide7 label-only MCQ capacity check under the same scoring path.
```

If this fails even at overfit-diagnostic budget, stop byte-native as the
near-term mainline.

#### Lane 2: Token-level mechanism control

```text
SmolLM2-360M label-only vs teacher-margin vs disagreement control.
```

This should answer whether Eklavya has mechanism value when the student is
known to be competent.

The reviewer's likely kill rules:

| Result | Recommendation |
|---|---|
| S0 passes, FMD residual passes | Continue byte-native Eklavya with stronger teachers. |
| S0 passes, FMD residual fails | Use S0 for supervised/counterfactual curriculum; demote FMD. |
| S0 fails, SmolLM2 residual passes | Pivot student/protocol; byte-native becomes future architecture research. |
| S0 fails, SmolLM2 residual fails | Abandon Eklavya KD as main moonshot. |
| SmolLM2 only label-only improves | Do not call it Eklavya; it is ordinary fine-tuning. |

### What Survives

The one impressive object is not a model. It is the combination of:

```text
PASS_DISAGREEMENT + kill discipline
```

That is enough for a final architecture decision, not enough for a public
claim.

### What Dies

This dies:

```text
The repo can persuade a hostile reviewer through process alone.
```

No. The next persuasive artifact must be a working student or a clean pivot.

### NARRATIVE ATTACK

**Strongest "that's obvious" dismissal:** You have an oracle disagreement
table, not a learning system.

**Strongest "that's trivial" dismissal:** The only thing that worked was a
post-hoc label-anchored analysis over two existing models.

**What would make the narrative hard to kill:** A trainable student must turn
disagreement into held-out gains over label-only, single-teacher, and
best-teacher baselines.

### Attack On The Next Defense

The next defense will say the disagreement result validates Eklavya. It
validates the fuel. It does not validate the engine.

---

## Iteration 119: Decision - Pivot KD Approach, Pivot Student, Or Pivot Moonshot?

### Steelman For Option A: Fix The Student And Continue Eklavya KD

Option A:

```text
Fix the student: S0 capacity check, Wide7 check, chain-init, architecture
change, then continue Eklavya KD only on a student that passes capacity.
```

This option best preserves the original moonshot:

- byte-native or byte-interface remains live;
- Eklavya disagreement remains connected to Sutra;
- FMD repair shot is preserved for a real student;
- `PASS_DISAGREEMENT` can become useful;
- the project does not abandon the architecture before testing S0.

The strongest reason for A is that B12 killed the shadow scaffold, not S0.
S0/Wide7 has not failed the direct label capacity check. Running that check is
high information and cheap relative to a project pivot.

### Attack On Option A

Option A is also the easiest way to keep moving the goalposts.

If S0 fails, the project may say:

```text
try Wide7
try D640
try D768
try chain-init
try byte bridge
try another decoder
```

That could become architecture churn. The invariant does not allow indefinite
loyalty to byte-native.

Option A must therefore be bounded:

```text
S0/Wide7 get a capacity board. If they cannot learn labels, no FMD, no router,
no new teacher objective on byte-native until there is a new birth mechanism
with its own precommitted evidence case.
```

### Steelman For Option B: Drop Byte-Native And Use Token-Level Students

Option B:

```text
Use token-level students such as SmolLM2-360M and make Eklavya work on existing
models.
```

This has the highest probability of producing functional results quickly.

It answers the core protocol question:

```text
Does disagreement-driven multi-teacher learning add value beyond ordinary
fine-tuning when the student is actually trainable?
```

It also respects the Vision's statement that mechanisms are negotiable. A
working token-level Eklavya is better than a non-working byte-level identity
project.

### Attack On Option B

Option B has lower paradigm-shift density.

The field already knows how to fine-tune pretrained token LMs. It already knows
PEFT. It already knows KD. It already knows ensemble/routing ideas. SmolLM2 is
larger and externally pretrained. If the project switches to SmolLM2 and gets
modest gains, a hostile reviewer will call it an engineering project.

Option B becomes moonshot-relevant only if:

```text
multi-teacher disagreement produces a robust residual that ordinary label-only,
single-teacher KD, and best-teacher imitation do not.
```

Otherwise it is not Eklavya. It is benchmark tuning.

### Steelman For Option C: Abandon Eklavya KD And Pivot Moonshot

Option C:

```text
Abandon Eklavya KD as the mainline and pivot to CTI, renormalization, CWC, or
another moonshot.
```

The steelman is brutal:

- six direction kills;
- no positive functional student result;
- proven-dead scaffold;
- teacher weakness;
- byte S0 far below target;
- novelty crowded by prior art;
- process increasingly sophisticated while function remains absent.

A moonshot lab should not become attached to a beautiful failure. If the
highest-EV opportunity is elsewhere, pivot.

### Attack On Option C

Option C is premature today because the project has one clean positive signal:

```text
PASS_DISAGREEMENT
```

And it has one untested decisive student:

```text
S0/Wide7
```

Abandoning Eklavya before testing a real pretrained student would discard the
only validated fuel source in the repo. It would replace an evidenced problem
with unevaluated alternatives.

The honest move is not full abandonment. It is a bounded student pivot.

### Ranking By Expected Value x Probability Of Paradigm Shift

This ranking is not a statistical estimate. It is a decision ranking after the
current evidence.

| Rank | Option | Functional probability | Paradigm-shift upside | EV x P verdict |
|---:|---|---:|---:|---|
| 1 | A. Pivot student/architecture, continue Eklavya only after capacity | Medium-low | High | Best current bet |
| 2 | B. Token-level Eklavya control on SmolLM2-style students | High for numbers, low-medium for paradigm | Medium | Best calibration lane |
| 3 | C. Abandon Eklavya KD entirely | Unknown | Unknown-high | Hold as trigger, not current move |

Concrete probability language:

| Claim | Current honest probability |
|---|---:|
| B11 MarginStudent ever becomes moonshot scaffold | <1% |
| S0/Wide7 passes a short MCQ capacity check | 35-45% |
| S0/Wide7 plus FMD/disagreement beats label-only cleanly | 10-20% |
| Token-level SmolLM2 shows some fine-tuning gain | 60-80% |
| Token-level SmolLM2 shows a true Eklavya residual over label-only/single-teacher | 20-35% |
| Byte-native Eklavya reaches paradigm-shifting evidence without chain-init | 3-7% |
| Byte-native Eklavya after student/initialization pivot reaches admission evidence | 15-25% |
| Current line reaches public moonshot result without student pivot | ~0% |

### Concrete Recommendation

Recommendation:

```text
Choose A now: pivot the student, not the moonshot.
Run B as the calibration lane if A fails or in parallel if budget allows.
Reserve C as the mandatory trigger if both byte-native capacity and token-level
Eklavya residual fail.
```

Operational decision tree:

1. **Never use B11 MarginStudent again for objective experiments.**
2. **Run S0/Wide7 MCQ capacity check.**
   - If pass: use SmolLM2 as stronger teacher; run preserved
     `FMD_SHADOW_288` on S0/Wide7 with label-only, token-KD, random-margin, and
     artifact controls.
   - If fail short but can overfit train: diagnose generalization/scoring path;
     do not run FMD yet.
   - If cannot overfit: demote byte-native as near-term student substrate.
3. **Run SmolLM2 token-level mechanism control.**
   - If teacher margins/disagreement beat label-only and single-teacher KD,
     Eklavya survives as a protocol and byte-native becomes an architecture
     return problem.
   - If only label-only improves, do not call it Eklavya.
4. **Pivot moonshot if both fail.**
   - If S0/Wide7 cannot learn and SmolLM2 shows no Eklavya residual, abandon
     KD as mainline and move to a different moonshot.

### What Survives

The Eklavya question survives in narrowed form:

```text
Can teacher disagreement produce residual retained gain in a trainable small
student beyond ordinary supervised or single-teacher adaptation?
```

### What Dies

This dies:

```text
Continue changing KD losses on the current byte-codec scaffold.
```

This also dies:

```text
Byte-native remains the mainline even if S0/Wide7 cannot learn labels.
```

### NARRATIVE ATTACK

**Strongest "that's obvious" dismissal:** After six kills, your recommendation
is still "one more architecture test."

**Strongest "that's trivial" dismissal:** The likely successful path is just
fine-tuning SmolLM2.

**What would make the narrative hard to kill:** The next work loop must produce
one of two clean outcomes: a byte-native student that learns labels and then
gets teacher residual, or a token-level Eklavya residual that proves the
protocol before a byte return. Anything else triggers pivot.

### Attack On The Next Defense

The next defense will say this preserves too many options. The answer is to
make the options conditional and terminal. A is first. B is calibration. C is
the trigger if both fail.

---

## Batch 17 Final Verdict

The architecture answer is:

```text
The B11 MarginStudent failed because it was not a real student. It was a frozen
codec plus a shallow random adapter asked to acquire benchmark function from
tiny supervision. The exact primary cause cannot be isolated from B12 alone,
but the dominant lesion is missing semantic initialization, likely compounded
by codec-interface loss and insufficient trainable capacity.
```

The project-level answer is:

```text
Do not pivot the moonshot yet. Pivot the student immediately.
```

Next admissible evidence:

1. S0/Wide7 label-only MCQ capacity.
2. If capacity passes, preserved FMD_SHADOW_288 on S0/Wide7 with stronger
   teacher portfolio and hard baselines.
3. If capacity fails, token-level SmolLM2 mechanism control.
4. If token-level control shows no Eklavya residual, pivot away from KD.

Final hostile statement:

```text
The current repo has fuel without an engine. The next engine test is S0/Wide7,
not another loss. If the real byte engine fails, prove the Eklavya protocol in
a token-level engine or stop calling KD the moonshot.
```

