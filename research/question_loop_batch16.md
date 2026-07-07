# QUESTION LOOP - Batch 16: Attack After A Second Kill

Date: 2026-07-07

Iterations: 106-112

## Grounding

I read the required local context in order:

1. `research/VISION.md`
2. `research/dual_loop_supervisor_checkin_10.md`
3. `research/dual_loop_supervisor_checkin_9.md`
4. `research/question_loop_batch15.md`
5. `research/work_loop_batch11.md`
6. `research/work_loop_batch10.md`
7. `code/margin_distillation.py`
8. `code/coordinate_inheritance.py`

Additional targeted local context used for Iteration 109 and 111:

- `README.md` was checked and found stale relative to Check-ins 9/10.
- No `CLAUDE.md` exists in this checkout, so I do not infer the missing moonshot list beyond the names in the prompt.
- `research/codex_r49_cbd_strategy.md`, `research/codex_r60_strategic_reassessment.md`, and targeted CBD passages in `research/DEEP_RETHINK.md` were checked for the local CBD comparison.

No GPU runs, training runs, benchmark runs, or experiments were performed. This batch is analysis only.

## Binding State Entering Batch 16

The project now has six direction deaths:

| # | Direction | Kill evidence |
|---|---|---|
| 1 | Gradient KD | <=0.7pp on HellaSwag |
| 2 | Brainseed v0 | All scorers worse than codec-only |
| 3 | Evidence-Native v0 | Evidence training hurt evidence use |
| 4 | Evidence-Native v1 | Internalization gate +0.47pp |
| 5 | Coordinate-Inheritance v0/v1/v2 | Main inherited worse than random on all 3 benchmark slices |
| 6 | FMD prototype, unlabeled | -12pp HellaSwag, 0pp PIQA, 0pp ARC-Easy |

The important commonality is not just "KD failed." The commonality is:

```text
proxy or auxiliary training signal moved
but benchmark-facing forced-choice function did not improve
```

In B10, coordinate inheritance produced surface compatibility but lost to random cores on HellaSwag, PIQA, and ARC-Easy. In B11, margin training loss fell from 1.1920 to 0.5628, but the benchmark result was flat or worse. The optimizer was not dead. The functional transfer was dead.

## Hostile Batch Verdict Up Front

The next decisive question is not "which objective should we try next?"

It is:

```text
Can the current codec -> small student/scaffold -> benchmark scorer learn any
held-out benchmark discrimination under the easiest honest supervision?
```

If the answer is no, then changing objectives is theater. The current scaffold is the bottleneck or the scale is too small for the scaffold, and FMD_SHADOW_288 should not be treated as the main event.

If the answer is yes, then the six-kill pattern becomes more specific:

```text
the scaffold can learn function, but the proxy/KD signals have not contained
the right functional information.
```

That distinction is cheap to test with a supervised label-only scaffold capacity check. It should be run before or in lockstep with FMD_SHADOW_288.

The most hostile honest verdict:

```text
Current Eklavya KD, as practiced in this repo, is probably a dead end at this
scale unless the next portfolio probe shows direct supervised capacity first.
```

The broader Eklavya philosophy is not dead. The current implementation pattern is under indictment.

---

## Iteration 106: Is The Scaffold The Bottleneck?

### Steelman

The scaffold should not be convicted too quickly.

The six deaths did not all test the same mechanism:

- Gradient KD tested soft/proxy imitation.
- Brainseed tested extracted scoring/readout variants.
- Evidence-native tested evidence-conditioned judgment and internalization.
- Coordinate inheritance tested Qwen-coordinate compatibility.
- FMD prototype tested teacher-ranked natural continuation pairs.

Each death has an obvious local explanation. Coordinate inheritance trained embedding MSE and token-space NLL, then evaluated MCQ function. FMD B11 trained on shard-derived natural continuations, then evaluated benchmark choices. Evidence-native could have failed because retrieved evidence and serialization created artifacts. Brainseed could have failed because frozen codec signals were not semantically addressable.

So the benign interpretation is:

```text
The scaffold has not failed a fair functional-supervision test.
The objectives have failed.
```

B11 is especially weak as a scaffold indictment. `code/margin_distillation.py` uses a tiny randomly initialized byte-autoregressive student:

- frozen codec;
- trainable projection;
- two-layer global reasoner;
- one-layer byte decoder;
- 50 unlabeled training examples;
- 10 gradient steps.

That is not the trained 121M S0/Wide7. Ten steps on arbitrary continuation snippets is too little to prove the byte scaffold cannot learn benchmark discrimination.

The correct charitable position is:

```text
Do not assume architecture death from objective failure.
Run the direct capacity check.
```

### Attack

The scaffold has been protected for too long by local explanations.

Every time the project changes the story, the same broad shape reappears:

```text
codec-derived states or byte-facing student
small train budget
teacher/scaffold signal appears to move
benchmark function does not move
```

The constant is not the loss. The constant is the substrate and evaluation path. A hostile reviewer will not care that each failure has a different local excuse. Six excuses can be one syndrome.

The syndrome may be:

1. The codec preserves lexical/byte regularities but not the task-semantic state needed for MCQ discrimination.
2. The tiny student starts too far from any useful representation for short KD to matter.
3. The Qwen-headed scaffold creates misleading compatibility signals but not student-native function.
4. The benchmark scorer is dominated by length/style priors rather than learned reasoning.
5. Ten to hundreds of updates are incapable of moving a random byte student into semantic space.

The hard criticism:

```text
If simple supervised MCQ training cannot move this scaffold, then FMD,
disagreement routing, teacher debate, and margin losses are all downstream
ornaments on an untrainable path.
```

The project should stop asking whether the next objective is clever until it has asked whether the student can learn the task under the least clever objective.

### Required Scaffold Capacity Check

The check should be treated as a kill gate, not a side audit.

Use the same candidate scoring path intended for FMD_SHADOW_288. Train on benchmark-style train-safe examples with simple labels.

Minimum variants:

| Variant | Purpose |
|---|---|
| Untrained scaffold | Establish current prior and length/style bias. |
| Label-only CE/ranking | Easiest honest functional supervision. |
| Random labels | Detect regularization or format-only effects. |
| Train-label shuffle held-out | Detect memorization and leakage. |
| Same examples, more steps if cheap | Distinguish no-capacity from under-training. |

Required measurements:

```text
train accuracy
held-out accuracy
gold-vs-best-wrong margin
paired sign-test win rate
train-heldout gap
artifact-control residual
```

Verdict tokens:

```text
PASS_SCAFFOLD_CAPACITY
  Label-only training improves held-out accuracy >=+5pp over untrained
  on >=2 of 3 benchmarks, with positive paired margin deltas.

MEMORIZATION_ONLY_SCAFFOLD
  Train improves >=+15pp but held-out improves <+2pp.

FAIL_SCAFFOLD_CAPACITY
  Train does not improve meaningfully, OR held-out improves <+2pp
  on >=2 of 3 benchmarks under simple label supervision.
```

If `FAIL_SCAFFOLD_CAPACITY` fires, do not repair FMD. Change the scaffold or abandon this KD line.

### What Survives

The scaffold survives only as an untested hypothesis. It is not exonerated.

### What Dies

This dies:

```text
Assume the architecture is fine and keep changing objectives.
```

No. After six kills, that assumption is no longer defensible.

### NARRATIVE ATTACK

**Strongest "that's obvious" dismissal:** You kept using the same tiny byte scaffold and are surprised that new losses do not create reasoning.

**Strongest "that's trivial" dismissal:** Before inventing a teacher-learning theory, prove the student can learn labels.

**What would make the narrative hard to kill:** A simple label-only scaffold capacity check must show held-out functional improvement. Then FMD can claim to test teacher value rather than scaffold viability.

### Attack On The Next Defense

The next defense will say label-only training is not the moonshot. Correct. That is exactly why it is the right diagnostic. If the non-moonshot baseline cannot work, the moonshot mechanism has no substrate.

---

## Iteration 107: FMD_SHADOW_288 Design Review

### Steelman

FMD_SHADOW_288 is the best remaining one-shot repair of Functional Margin Distillation because it fixes the specific B11 failure artifact.

B11 trained on:

```text
natural shard continuation ranking -> benchmark MCQ evaluation
```

That mismatch is real. A model can learn that one arbitrary text span is more corpus-natural than another without learning to distinguish a correct HellaSwag ending from a plausible distractor.

B15's FMD_SHADOW_288 moves the target distribution closer to the evaluation distribution:

```text
96 train-safe examples each from HellaSwag, PIQA, ARC-Easy = 288 train
48 disjoint train-safe examples each = 144 held-out
gold-vs-hardest-wrong pairs
teacher-wrong filtering or downweighting
same-budget baselines
artifact controls
precommitted PASS/MARGINAL/FAIL tokens
```

The design also has the right hostile label:

```text
FMD_ADAPTER_SHADOW_NOT_BYTE_NATIVE_PROOF
```

That label matters. The test can answer:

```text
Does teacher-margin supervision add benchmark-facing signal beyond ordinary
same-budget baselines in the current scaffold?
```

It cannot answer:

```text
Has Sutra learned intelligence?
```

The steelman is that FMD_SHADOW_288 is not trying to be the moonshot. It is an admission test after a data-mismatch failure.

### Attack

FMD_SHADOW_288 is dangerously close to supervised MCQ fine-tuning with extra steps.

If the gold label comes from HellaSwag/PIQA/ARC-Easy, then the teacher is not teaching correctness. The dataset is. The teacher can supply:

- hard-negative choice;
- margin strength;
- confidence filtering;
- maybe curriculum order.

But the correctness anchor is ordinary supervised benchmark data.

A hostile reviewer will reduce the method to:

```text
You trained on benchmark-format MCQ examples and improved on benchmark-format
MCQ examples.
```

If it works, the first explanation will be:

```text
MCQ fine-tuning helps on MCQ benchmarks.
```

The burden is to prove a residual:

```text
teacher margins add value beyond label-only MCQ training
```

not merely:

```text
benchmark-style training beats unlabeled shard training
```

### Design Failure Modes

| Failure mode | Why it matters |
|---|---|
| Label-only baseline wins | FMD is unnecessary supervised ranking. |
| Token-KD baseline wins | Pairwise margins are not the right teacher object. |
| Random margins help similarly | The pipeline is regularization or data exposure, not teacher geometry. |
| Length/position controls erase gains | The model learned answer formatting. |
| Teacher-wrong examples drive gains | The student copied Qwen quirks or benchmark priors, not correctness. |
| Train-only gain | The sample is too small or memorized. |
| PIQA-only gain | The method may exploit binary/short-choice format. |
| HellaSwag regression persists | Continuation ranking still conflicts with commonsense ending selection. |
| Qwen-headed adapter path works but byte-native path fails | The result is scaffold-specific, not Sutra evidence. |
| FMD ties label-only but gets renamed "geometry" | Baseline laundering. |

The strongest technical attack is teacher weakness. B10/B11 samples show Qwen was weak or unstable on HellaSwag and ARC-Easy. If the teacher is at 38-54% on the same task families, then margin regression can poison exactly the examples where labels are most valuable.

Teacher filtering is mandatory, but filtering creates another attack:

```text
If you keep only teacher-correct examples, then the teacher becomes a confidence
filter over labels, not a source of new knowledge.
```

That may be useful. It is not yet Eklavya.

### If It Works, What Does It Prove?

There are four possible success interpretations:

| Observed result | Honest interpretation |
|---|---|
| FMD beats untrained only | MCQ supervision or training exposure helped. Weak evidence. |
| FMD beats random margins but ties label-only | Labels matter; teacher margins do not. Kill FMD as KD. |
| FMD beats label-only by >=+2pp but not token-KD | Teacher helps, but margin objective is not special. |
| FMD beats label-only and token-KD by >=+5pp on >=2 benchmarks with artifact controls clean | FMD earns a second byte-native experiment. |

Only the last result keeps the FMD story alive.

### Required Amendment

FMD_SHADOW_288 should not be run without the scaffold capacity check. The two should share the same data split and scoring path.

The decisive comparison is:

```text
FMD residual = FMD_SHADOW_288 - label-only capacity check
```

If that residual is small, FMD is not the mechanism. If label-only fails, FMD failure is uninterpretable because the scaffold may be dead.

### What Survives

FMD_SHADOW_288 survives as a bounded admission test.

### What Dies

This dies:

```text
If FMD_SHADOW_288 works, it proves teacher-margin geometry.
```

No. It proves that only if it beats same-budget label-only, token-KD, random-margin, and artifact controls.

### NARRATIVE ATTACK

**Strongest "that's obvious" dismissal:** You trained on multiple-choice data and got better at multiple choice.

**Strongest "that's trivial" dismissal:** The teacher is just choosing hard negatives for supervised fine-tuning.

**What would make the narrative hard to kill:** FMD must beat label-only MCQ training and token-KD under identical data/update budgets, with clean held-out gains and artifact controls.

### Attack On The Next Defense

The next defense will say "benchmark-style data was the missing piece." That may be true, but if benchmark-style label-only training also works, then FMD was not the missing piece. The missing piece was direct supervision.

---

## Iteration 108: The 6-Kill Root Cause

### Steelman

The cleanest single root cause is the proxy-to-function gap.

Every dead direction improved or optimized a target that was not the final functional benchmark:

| Direction | Training or selection signal | Functional failure |
|---|---|---|
| Gradient KD | teacher probability/gradient proxy | <=0.7pp HellaSwag |
| Brainseed | codec-derived scorer/readout | worse than codec-only |
| Evidence-Native v0/v1 | evidence/rationale serialization and internalization proxy | no functional evidence-use gain |
| Coordinate inheritance | embedding MSE and token-space NLL | worse than random on MCQ margins |
| FMD prototype | shard continuation teacher margins | flat/worse benchmark MCQ margins |

The steelman says:

```text
Stop training proxies. Train the exact functional margin.
```

Under this interpretation, FMD_SHADOW_288 is justified because it is the first test to train directly on benchmark-style decision margins rather than a remote proxy.

### Attack

Proxy-to-function gap is true but may be too shallow.

It does not explain why the project repeatedly chooses proxies. The deeper cause may be:

```text
The student does not have enough representation capacity or initialization to
make direct functional supervision data-efficient, so the project keeps looking
for clever proxy signals that can bootstrap it.
```

The root cause candidates are not mutually exclusive:

| Candidate root cause | Explains all six? | Cheap discriminant |
|---|---|---|
| Proxy-to-function gap | Yes, at surface level | Train direct label-only functional margins. |
| Student capacity | Plausible | Can label-only training overfit and generalize at all? |
| Scaffold design | Plausible | Compare same data on current scaffold vs direct S0/Wide7 or simpler classifier head. |
| Scale | Plausible | Same method at 10, 50, 200, 1000 steps on label-only data. |
| Teacher quality | Explains KD/FMD, not all non-teacher failures | Teacher-margin audit by benchmark and slice. |
| Data/domain mismatch | Explains B11 strongly | Benchmark-style vs shard-style split. |

The hostile claim:

```text
There is no single proven root cause yet because the project has not run the
one diagnostic that separates objective failure from scaffold failure.
```

### One Cheap Experiment That Tests The Most

The scaffold capacity check is the highest-information cheap experiment.

It partitions the world:

| Result | Interpretation |
|---|---|
| Label-only cannot improve train or held-out | Scaffold/optimizer/scoring path is broken or too weak. Stop objective search. |
| Label-only overfits train but no held-out | Scale/data split/generalization problem. FMD success would likely be benchmark-specific. |
| Label-only improves held-out, FMD does not | Teacher margins/objective are the problem. Use supervised/counterfactual curriculum instead. |
| Label-only improves, FMD improves more | Teacher margins contain useful residual signal. FMD earns second experiment. |
| Label-only improves only PIQA | The scaffold can learn shallow binary-choice artifacts, not general reasoning. |

Add a no-training teacher audit in parallel:

```text
teacher accuracy
teacher-wrong rate
margin separability
hard-negative length/position artifacts
disagreement density if second teacher exists
```

But if only one experiment is allowed, run label-only capacity first.

### What Survives

Proxy-to-function gap remains the best descriptive pattern.

### What Dies

This dies:

```text
The 6-kill root cause is solved: just train directly on benchmark margins.
```

No. Direct-margin training could still fail if the student/scaffold cannot learn or if the scale is too small.

### NARRATIVE ATTACK

**Strongest "that's obvious" dismissal:** Your losses optimized things other than the benchmark, so of course benchmarks did not improve.

**Strongest "that's trivial" dismissal:** The deeper issue is not the proxy. It is that a tiny scratch byte model has no semantic basis to update.

**What would make the narrative hard to kill:** A direct functional-supervision check must show the scaffold can learn, then FMD must show teacher residual beyond that.

### Attack On The Next Defense

The next defense will say "FMD_SHADOW_288 is the direct functional test." Not unless label-only and token-KD controls are equally strong. Otherwise it confounds direct supervision with teacher-margin learning.

---

## Iteration 109: What Did CBD Do Differently?

### Steelman

CBD is the uncomfortable external proof that small models can carry much more benchmark function than this repo's current 121M byte models show.

The local notes state CBD reaches 42.65% HellaSwag at 138M through chain distillation. The project is around the 26-27% HellaSwag zone for S0/Wide7-style byte models. That gap is too large to dismiss as noise.

The benign lesson is:

```text
The benchmark target is not physically impossible at this parameter count.
The missing ingredient is transfer.
```

CBD's structural idea is simple:

```text
large model -> intermediate anchor -> smaller anchor -> final small model
```

The student does not have to invent a world model from scratch. It inherits one through a chain.

That fits the Vision better than endless scratch training. The Vision does not sanctify any one architecture. It says architecture is a means to democratized intelligence. If chain-style inheritance is the only path to stop-scrolling small-model performance, the project should learn from it.

### Attack

The comparison is structurally unfair.

CBD is not merely "better KD." Local strategy notes identify the crucial differences:

| CBD advantage | Current Sutra/Eklavya disadvantage |
|---|---|
| Same tokenizer through the chain | Cross-tokenizer BPE-to-byte transfer |
| Similar architecture family | Transformer teacher to byte-patch scaffold |
| Pretrained anchors | Scratch or tiny random student paths |
| Weight/init continuity | Loss-only or adapter-only transfer |
| Smaller compression gaps | Large teacher to tiny byte student jumps |
| Teacher shapes representation during formation | Post-hoc low-budget correction or short smoke training |

CBD preserves a coordinate system. Sutra has been trying to translate across coordinate systems.

That distinction is not cosmetic. It explains why CBD can compress knowledge while current Eklavya prototypes mainly create low-bandwidth supervisory hints.

The hostile reviewer will say:

```text
CBD transfers an already-formed feature basis.
You are asking a scratch byte scaffold to infer that basis from a few proxy losses.
These are not comparable experiments.
```

### Should We Adopt CBD?

Adopt the structural lesson, not the claim.

The structural lesson:

```text
initialization and coordinate continuity beat clever post-hoc losses when the
student is tiny.
```

The claim to avoid:

```text
If we do byteified CBD, Eklavya is proven.
```

A pure CBD adoption would be a strong engineering baseline, but it weakens novelty. It becomes:

```text
CBD with a byte interface
```

That may still be valuable if it beats token-chain CBD per compute, openness, or tokenizer universality. But it is not the same as proving disagreement-driven multi-teacher learning.

The more honest path is:

1. Treat CBD-style chain-init as the strongest baseline/fallback.
2. Use a byteified chain or Sutra-family anchor only if the current scaffold capacity check fails or FMD residual is weak.
3. Require Eklavya to add retained gain on top of the CBD-style baseline, not instead of it.

### The Architectural Lesson

CBD suggests this repo's repeated failures may be about birth, not tutoring.

The current project keeps trying to teach a newborn small model with sparse lessons. CBD starts the child with inherited structure.

If the model's representation basis is absent, then:

```text
FMD, disagreement routing, debate compression, and evidence internalization are
not lessons. They are whispers into a system without the right coordinate frame.
```

### What Survives

CBD does not invalidate Eklavya as a later-stage enrichment or routing protocol.

### What Dies

This dies:

```text
Post-training clever KD alone is likely to catch CBD at 121M-138M.
```

The local evidence does not support that. Without a birth/initialization mechanism, the probability is low.

### NARRATIVE ATTACK

**Strongest "that's obvious" dismissal:** CBD works because it transfers weights and coordinate continuity. Your method transfers weak gradients or margins into a scratch byte model.

**Strongest "that's trivial" dismissal:** The field already knows same-family distillation works. Byteifying it is engineering unless the byte interface adds measurable value.

**What would make the narrative hard to kill:** A byte-native or byte-interface chain-init must approach or beat CBD-style performance, and Eklavya must add retained gain over that chain baseline.

### Attack On The Next Defense

The next defense will say "CBD comparison is invalid, so ignore it." Invalid as a direct fairness comparison, yes. Invalid as a strategic warning, no. It tells us that small-model benchmark function likely comes from inherited representation, not tiny proxy updates.

---

## Iteration 110: Alternative Portfolios If FMD_SHADOW_288 Fails

### Steelman

If FMD_SHADOW_288 fails, the project is not literally out of ideas. Several alternatives remain live because they attack different parts of the failure pattern:

1. Counterfactual minimal-pair curriculum can train causal distinctions directly.
2. Error Atlas / Surgical patches can turn failures into targeted repair packets.
3. Teacher debate compression can create richer student-native training text.
4. Direct S0/Wide7 fine-tuning can establish the baseline capacity floor.
5. Abandoning KD can free the project from local search.
6. Changing the student architecture can address the constant scaffold risk.

The steelman is that FMD failure should not automatically kill all Eklavya variants. It should kill teacher-margin ranking as the current formulation.

### Attack

Most of the list can become renamed versions of prior failure.

| Candidate | Distinct mechanism? | Hostile reduction |
|---|---|---|
| Counterfactual minimal-pair curriculum | Yes, if counterfactuals isolate causal features and labels are verified. | FMD with generated distractors. |
| Error Atlas / Surgical patches | Yes, if patches are local, regression-tested, and reusable. | Post-hoc benchmark tutoring. |
| Teacher debate compression | Weak unless independently adjudicated. | Teacher-as-data/rationale distillation with more text. |
| Direct S0/Wide7 fine-tuning | Baseline, not moonshot. | Ordinary supervised fine-tuning. |
| Abandon KD entirely | Strategically distinct, not a mechanism. | Giving up after negative results. |
| Change student architecture | Mechanistically distinct if scaffold capacity fails. | Architecture churn unless capacity diagnosis is clear. |

### Honest Ranking

If FMD_SHADOW_288 fails, the ranking should depend on the scaffold capacity result.

#### If Label-Only Capacity Passes

The scaffold can learn benchmark function. Then the problem is teacher-margin KD, not the student.

Rank:

1. **Counterfactual minimal-pair curriculum.** Best distinct mechanism. It directly attacks causal distinctions rather than teacher likelihood quirks.
2. **Error Atlas / Surgical patches.** Strong if it produces reusable local repairs and regression controls.
3. **Direct S0/Wide7 fine-tuning.** Essential baseline and maybe useful product path, but not a moonshot claim.
4. **Teacher debate compression.** Keep only if debate outputs are verified and beat ordinary teacher-as-data.
5. **Change student architecture.** Defer if scaffold capacity passed.
6. **Abandon KD entirely.** Prepare, but do not trigger solely from FMD failure if direct capacity exists.

#### If Label-Only Capacity Fails

The current scaffold cannot learn even the easy thing. Then objective portfolios are mostly wasted.

Rank:

1. **Change student architecture entirely / drop the current codec scaffold.** Highest leverage because the constant substrate is indicted.
2. **Direct S0/Wide7 fine-tuning on a stronger or pretrained base.** Use it as a new capacity floor, not as FMD support.
3. **Abandon KD and pivot to another moonshot.** Serious option if architecture change is too costly.
4. **Counterfactual minimal-pair curriculum.** Only after a learnable scaffold exists.
5. **Error Atlas / Surgical patches.** Cannot patch a student that cannot absorb patches.
6. **Teacher debate compression.** Lowest priority; it adds complexity before capacity.

### Mechanism Classification

The actually different mechanisms are:

```text
counterfactual causal data
student architecture / initialization change
chain-init or inherited representation birth
project-level pivot away from KD
```

The likely renamed mechanisms are:

```text
teacher debate without verification
FMD with synthetic distractors but no causal controls
Error Atlas that only selects more MCQ fine-tuning examples
direct fine-tuning presented as Eklavya
```

### What Survives

The portfolio approach survives, but only if it stops serially renaming the same supervision pattern.

### What Dies

This dies:

```text
After FMD fails, move to the next teacher-shaped loss.
```

No. If FMD fails, the next move is capacity/architecture/counterfactual causality, not another margin synonym.

### NARRATIVE ATTACK

**Strongest "that's obvious" dismissal:** Your alternatives are all ways to fine-tune on better examples.

**Strongest "that's trivial" dismissal:** Direct fine-tuning is the baseline. Debate and patches are curriculum engineering unless they beat that baseline.

**What would make the narrative hard to kill:** The next survivor must show a mechanism-specific residual over direct supervised fine-tuning and random/shuffled curriculum controls.

### Attack On The Next Defense

The next defense will say these alternatives preserve the Eklavya spirit. Spirit is not evidence. Each alternative must name what it transfers that label-only training does not.

---

## Iteration 111: Honest Moonshot Probability

### Steelman

The project is not irrational to continue one more bounded portfolio probe.

Reasons:

1. The Vision bar is intentionally extreme; most serious mechanisms should die.
2. The dual-loop has prevented false promotion.
3. FMD_SHADOW_288 has not been tested in its benchmark-style form.
4. Scaffold capacity has not been directly tested.
5. Teacher-margin and disagreement audits have not been run.
6. CBD suggests small-model function is possible if transfer is solved.

So the steelman is:

```text
One more cheap, decisive evidence board is justified.
```

Not one more open-ended repair cycle. One board.

### Attack

Six consecutive kills should radically lower confidence.

The honest probability should separate levels:

| Claim | Probability after six kills |
|---|---:|
| Current codec -> tiny/scaffold -> short KD path produces paradigm shift | 1-3% |
| FMD_SHADOW_288 produces clean admission evidence over label-only/token-KD | 10-20% |
| Broader Eklavya KD at this scale reaches CBD/SmolLM2-level without chain-init or architecture change | 3-7% |
| Broader Sutra project succeeds after major architecture/initialization pivot | 8-15% |
| Current line produces a public paradigm-shifting result without major pivot | <5% |

These are not scientific estimates. They are decision probabilities after the graveyard evidence. The important move is not the exact number. It is that the number is now low enough to force a project-level option value comparison.

### Is The Dual-Loop Becoming Sunk Cost?

Partly, yes.

The dual-loop is excellent at killing claims. But a process can be rigorous and still locally trapped. The signs:

- repeated local repairs around the codec/scaffold;
- new names for teacher-shaped supervision;
- promotion hopes attached to admission tests;
- no positive functional result after many loops;
- stale docs still describing older live directions;
- increasing process sophistication without increasing benchmark function.

The sunk-cost failure mode is:

```text
because the loop is adversarial, every new attempt feels disciplined enough to
deserve another attempt.
```

That cannot continue.

### Should The Project Pivot To CTI, Renormalization, Or CWC?

This checkout has no `CLAUDE.md`, so I cannot evaluate the full alternative list from local source. Based only on the prompt's names, the answer is conditional:

```text
Do not pivot before the scaffold capacity/FMD/audit board.
Do pivot if that board produces no clean positive functional movement.
```

The reason to wait one board is not hope. It is information value. The capacity check is cheap and settles whether the current failure is objective-level or substrate-level.

But the board must be final for this line:

```text
If label-only fails, FMD fails, and teacher/disagreement audits are poor,
Eklavya KD should be abandoned as the main moonshot at this scale.
```

At that point, continuing would be sunk cost.

### Honest Recommendation

Run W-Loop B12 as the terminal admission board for current Eklavya KD:

1. Scaffold capacity check.
2. FMD_SHADOW_288.
3. Teacher-margin audit.
4. Disagreement density audit if feasible.

Then:

| Board result | Decision |
|---|---|
| Capacity fails | Drop current scaffold. No more objective repairs. |
| Capacity passes, FMD fails | Kill FMD. Move to counterfactual/Error Atlas or direct FT baseline. |
| Capacity passes, FMD beats label-only/token-KD | One byte-native replication allowed. |
| Teacher audit fails | Stop Qwen-margin KD or upgrade teacher. |
| Disagreement audit fails | Do not build router. |
| All fail or marginal | Pivot project-level mainline away from Eklavya KD. |

### What Survives

One more evidence board survives.

### What Dies

This dies:

```text
Eklavya KD remains the presumed mainline because it is philosophically aligned.
```

No. Philosophy does not outrank six functional kills.

### NARRATIVE ATTACK

**Strongest "that's obvious" dismissal:** You have no positive benchmark result and keep asking for one more formulation.

**Strongest "that's trivial" dismissal:** A project that only produces careful negative results is not a democratization moonshot.

**What would make the narrative hard to kill:** The next board must produce a clean, held-out, same-budget-controlled functional gain, or the project must pivot without euphemism.

### Attack On The Next Defense

The next defense will say moonshots require persistence. True. But persistence means staying loyal to the invariant, not to the implementation. The invariant is democratized intelligence, not this scaffold.

---

## Iteration 112: What Would Convince A Hostile Reviewer?

### Steelman

A hostile reviewer can be convinced, but not by rhetoric.

The repo has some credibility:

- It precommitted gates.
- It killed its favorite direction.
- It separated diagnostic insight from moonshot evidence.
- It preserved negative results.
- It uses controls and bootstrap deltas.
- It is now asking the right scaffold-capacity question.

That process credibility matters. A reviewer may trust the falsification discipline more than the current mechanism.

### Attack

A fresh adversarial reviewer would say:

```text
You have six dead directions, no positive functional result, a stale README,
weak teachers, tiny samples, moving mechanisms, and a repeated pattern where
training losses improve but benchmark accuracy does not.
```

They would attack the repo on these fronts:

| Reviewer attack | Evidence they cite |
|---|---|
| No positive functional result | Graveyard has six kills and no benchmark lift. |
| Objective migration | KD -> Brainseed -> evidence -> coordinates -> margins. |
| Baseline laundering | Ranking loss and MCQ fine-tuning dressed as geometry. |
| Scaffold unproven | No direct label-only capacity proof. |
| Teacher weak | Qwen weak on HellaSwag/ARC samples. |
| Samples too small | 50-example and 144 held-out gates have wide uncertainty. |
| Benchmark overfit risk | Train-safe MCQ data can teach dataset format. |
| Novelty weak | Pairwise ranking, KD, multi-teacher, and routing all have prior art. |
| External competitor stronger | CBD reaches the target zone through chain-init. |
| Process over product | The loop is good at killing, not yet good at producing. |

The harsh reviewer summary:

```text
This is a rigorous negative-results repo around a low-probability small-model
KD thesis. It has not yet shown that the student can learn the benchmark, that
the teacher adds value, or that the byte scaffold is the right substrate.
```

That is fair.

### Minimum Evidence Needed To Change Their Mind

The minimum evidence is not one number. It is a chain:

#### 1. Capacity Evidence

```text
The exact student/scaffold learns label-only MCQ supervision.
```

Required:

- held-out +5pp on >=2 of 3 benchmark families;
- positive paired margin deltas;
- random-label control flat;
- train/held-out gap reported;
- artifact controls clean.

Without this, nothing else matters.

#### 2. Teacher Residual Evidence

```text
Teacher margins beat label-only and token-KD under the same budget.
```

Required:

- FMD >=+2pp over label-only and token-KD on >=2 of 3 benchmarks for admission;
- >=+5pp for strong admission;
- margin CI lower bound >0;
- teacher-wrong filtering reported;
- no PIQA-only artifact pass.

#### 3. Byte-Native Replication

```text
The gain survives outside a Qwen-headed adapter shadow.
```

Required:

- actual S0/Wide7/Sutra path, not only Qwen scorer;
- same-budget supervised and KD baselines;
- no BPB collapse;
- validation split or held-out train-safe split with leakage checks.

#### 4. Mechanism-Specific Novelty

One of these must be proven:

| Claim | Required evidence |
|---|---|
| FMD novelty | Beats label-only, token-KD, random margins, and artifact controls. |
| Disagreement routing | Useful disagreement abundant and router beats best single teacher. |
| Counterfactual curriculum | Counterfactual structure beats correct-continuation-only and shuffled explanations. |
| Error Atlas | Local patches fix error clusters with bounded regressions and reuse across splits. |
| Byte chain-init | Byte interface adds value over ordinary CBD-style chain or matches it under harder constraints. |

#### 5. Scale And Robustness

Admission evidence is not moonshot evidence. Moonshot evidence needs:

- >=500 examples per benchmark family or public validation where allowed;
- repeated seeds or deterministic paired evaluation;
- strong baselines;
- confidence intervals;
- exact artifacts;
- held-out benchmarks beyond the training format;
- comparison to Wide7, SmolLM2/Pythia/CBD targets where appropriate.

### What Would Not Convince Them

These should be explicitly rejected:

- training loss decreases;
- NLL improves;
- BPB improves;
- adapter shadow improves without label-only residual;
- PIQA-only gain;
- train-only gain;
- teacher confidence curves;
- qualitative examples;
- "the mechanism is aligned with the Vision";
- one small held-out slice without controls.

### Hostile Reviewer Verdict Today

Today they would not fund or believe the moonshot claim.

They might fund one terminal diagnostic board because the process is disciplined and the question is now sharp:

```text
Can the scaffold learn direct functional supervision, and does the teacher add
anything beyond that?
```

If that board fails, they would recommend project-level pivot.

### What Survives

The falsification process survives as a project asset.

### What Dies

This dies:

```text
A seventh named KD variant can persuade a hostile reviewer without first
settling scaffold capacity and baseline residuals.
```

No. The next evidence must be brutally simple.

### NARRATIVE ATTACK

**Strongest "that's obvious" dismissal:** You have built a careful way to discover that tiny models trained on proxies do not get smarter.

**Strongest "that's trivial" dismissal:** The minimum next proof is ordinary: show supervised learning works, then show the teacher adds value.

**What would make the narrative hard to kill:** A byte-native student must show held-out benchmark gains over label-only, token-KD, random/shuffled, and artifact controls, then retain those gains at a larger scale.

### Attack On The Next Defense

The next defense will say the moonshot requires unusual evidence, not ordinary baselines. Backwards. The more unusual the claim, the more humiliatingly ordinary the baselines must be.

---

## Batch 16 Final Verdict

FMD_SHADOW_288 is allowed exactly one shot, but it is no longer the first question. The first question is scaffold capacity.

The next W-Loop should run a terminal admission board:

```text
1. Scaffold capacity check: label-only supervised MCQ on the exact scaffold.
2. FMD_SHADOW_288: teacher margins on benchmark-style data.
3. Teacher-margin data audit: accuracy, margin quality, hard-negative quality.
4. Disagreement density audit: useful disagreement and oracle routing ceiling.
```

Decision rules:

| Result | Required action |
|---|---|
| `FAIL_SCAFFOLD_CAPACITY` | Stop changing objectives. Drop or redesign scaffold. |
| `PASS_SCAFFOLD_CAPACITY` + `FAIL_FMD_SHADOW` | Kill FMD. Teacher margins do not add value. |
| `PASS_FMD_SHADOW` without label-only residual | Do not promote. It is MCQ fine-tuning. |
| Teacher audit fails | Stop Qwen-margin KD or upgrade teacher before training. |
| Disagreement audit fails | Do not build router. |
| All probes fail or marginal | Abandon Eklavya KD as mainline at this scale and pivot project-level. |

Hostile final statement:

```text
After six kills, the burden is no longer to invent a better loss.
The burden is to prove the student can learn function at all.
If it can, prove the teacher adds residual value over labels.
If it cannot, the current Eklavya KD line is dead at this scale.
```

The invariant remains: paradigm-shifting democratized intelligence or nothing. The current scaffold does not get loyalty. Only evidence does.
