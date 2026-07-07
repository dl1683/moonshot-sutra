# QUESTION LOOP - Batch 12: Attack v1 Repairs + Stage 1 Signal Interpretation

Date: 2026-07-07

Iterations: 78-84

## Grounding

I read the requested context in order:

1. `research/VISION.md`
2. `research/dual_loop_supervisor_checkin_7.md`
3. `research/work_loop_batch8.md`
4. `research/question_loop_batch11.md`
5. `research/question_loop_batch10.md`
6. `tmp_coordinate_inheritance_full/preflight_metrics.json`
7. `research/DEEP_RETHINK.md`
8. `code/coordinate_inheritance.py`

No GPU runs, training runs, benchmark runs, or experiments were performed for this batch. This is analysis only.

I also checked a small set of external historical references for the NLL-to-task translation discussion in Iteration 80:

- Gadre et al., 2024, ["Language models scale reliably with over-training and on downstream tasks"](https://arxiv.org/abs/2403.08540)
- Isik et al., 2024, ["Scaling Laws for Downstream Task Performance of Large Language Models"](https://arxiv.org/abs/2402.04177)
- Magnusson et al., 2023, ["Paloma: A Benchmark for Evaluating Language Model Fit"](https://arxiv.org/abs/2312.10523)
- Schaeffer et al., 2023, ["Are Emergent Abilities of Large Language Models a Mirage?"](https://arxiv.org/abs/2304.15004)

## Binding Interpretation Entering Batch 12

Coordinate-inheritance v0 is killed. The direction is alive, but only as a hostile, kill-gated hypothesis.

The massive Stage 1 NLL signal is real in the narrow sense:

```text
calibrated codec states + copied first 4 Qwen layers + Qwen head
beat
calibrated codec states + random/shuffled Qwen-shaped layers + Qwen head
by 4.70 to 6.13 nats/token.
```

But the signal is not yet a reasoning result, not yet a byte-native result, not yet a compression result, and not yet a moonshot result.

The code makes the narrowness explicit:

- Stage 1 evaluates token-space next-token NLL through a Qwen LM head.
- The "true embedding" upper bound is true Qwen embeddings into the same truncated copied-core/head setup, not full native Qwen inference.
- Token-end and patch-boundary samples are converted into sequences of Qwen token IDs and scored by shifted token CE.
- Patch-boundary sequences can repeat token labels within one token.
- Rotation is input-gauge disruption/recovery, not a full residual-basis transform through all model weights.
- Frozen-core gain compares frozen copied core to a very short finetune path that also finetunes a copied adapter.
- Benchmark mode, if later used, chooses continuations by token NLL per scored token through byte-derived codec states. It does not produce byte BPB or native byte decoding evidence.

Batch 12's thesis:

```text
v1 can probably pass the two borderline v0 gates. That is not the same as making the geometry claim stronger. The decisive question is whether v1 adds independent evidence, or only tunes the experiment until the precommitted numbers turn green.
```

---

## Iteration 78: Will v1 Repairs Just Game The Thresholds?

### Steelman

The v1 repair plan is not arbitrary. It targets the two exact ways v0 failed.

Patch-boundary frozen-core gain was 66.3% against a 70% gate. The patch-boundary stream is the byte-native bottleneck, and the v0 adapter was shared across token-end and patch-boundary states. A readout-conditioned adapter is a plausible repair because token-end and patch-boundary codec states are not the same distribution.

Token-end no-inverse rotation retained 33.0% of inherited lift against a 30% collapse gate. The v0 rotation was only an input-gauge disruption. A stronger rotation or full residual-basis transform is a plausible repair because the old test may have been too weak to destroy the relevant coordinate organization.

Layer-depth curves are also reasonable. Four copied layers may be a bad depth: too shallow to expose reasoning, too shallow to test compression, and possibly too fragile around residual scale. Testing 2/4/6/8 layers can reveal whether the signal grows with teacher-ordered depth or whether v0 hit a lucky local optimum.

The generous reading:

```text
v1 is not moving the goalposts. It is fixing known measurement weaknesses before re-running the same hostile Stage 1 gate.
```

### Attack

The repair plan is dangerously optimized to the failed rows rather than to the claim.

Failure 1 was 66.3% vs 70%. A readout-conditioned adapter could push patch-boundary frozen-core gain above 70% by reducing frozen copied NLL, by reducing post-finetune incremental gain, or by overfitting patch-boundary statistics. Only the first of those is evidence for better inherited geometry. The other two are gate engineering.

Failure 2 was 33.0% vs 30%. A stronger disruption can trivially reduce retained lift if it makes inputs less natural. But "make the broken control worse" is not the same as "prove the unbroken path is geometry." A control can be made too easy to defeat. If no-inverse rotation collapses to 5% while inverse recovers, that checks algebra. It still may not say anything about HellaSwag margins.

Layer-depth curves also invite cherry-picking. If 2 layers fail, 4 borderline fails, 6 passes, and 8 destabilizes, v1 can choose 6 and call it geometry. But a true depth story should be predicted before the run:

```text
teacher-ordered depth should improve functional margins and layer trajectory more than random/shuffled/generic controls, not merely minimize one NLL gate.
```

Generic pretrained controls could become another threshold game if they are implemented as unfair broken controls. Feeding non-Qwen layers into a Qwen head, or using a badly matched adapter, would make "generic pretrained" lose by construction.

The risk is that v1 repairs produce this:

| Metric | v0 | v1 | Interpretation |
|---|---:|---:|---|
| patch frozen-core | 66.3% | 71% | threshold pass |
| token rotation retention | 33.0% | 19% | stronger broken control |
| copied advantage | huge | huge | same surface signal |
| benchmark margin shadow | untested | untested | no new geometry proof |

That would be a cleaner Stage 1 report but not a deeper result.

### Distinguishing Genuine Improvement From Threshold Gaming

v1 is genuine improvement only if the repairs improve independent observables that were not directly optimized.

| Diagnostic | Genuine improvement | Threshold gaming |
|---|---|---|
| Patch-conditioned adapter | improves held-out patch-boundary margins, rare-token slices, and first-byte/inside-token behavior | only moves frozen-core ratio from 66.3% to just above 70% |
| Stronger rotation | destroys NLL and functional-margin lift under a predeclared full-basis transform; inverse recovers both | chooses a destructive transform that randomizes inputs until control loses |
| Depth curve | teacher-ordered depth improves predictably on NLL, margins, and trajectory diagnostics | best depth is selected post hoc from noisy NLL gates |
| Generic pretrained control | fair core-native benchmark comparison, generic control could plausibly win | incompatible head/adapter makes generic control fail by design |
| Adapter budget | still tiny, still shared or predeclared, no core-specific leakage | per-readout/core-specific adapters become hidden distillation budget |
| Seeds/data | pass has margin across seeds, domains, token-frequency slices | one seed and one data slice turn green |

The minimal anti-gaming rule:

```text
No v1 repair is allowed to count as stronger geometry evidence unless it also improves a benchmark-facing functional-margin shadow test.
```

Stage 1 NLL can certify codec/gauge compatibility. It cannot certify reasoning geometry without a margin shadow.

### What Survived

The v1 repairs are worth doing. A shared adapter was likely underfitting two readout distributions, and input-only rotation was an incomplete gauge test.

The v0 failures were close enough that v1 may reasonably pass.

### What Died

This defense died:

```text
If v1 flips the two red Stage 1 cells to green, the geometry proof becomes clean.
```

No. A threshold pass proves only that the v1 artifact satisfies the preflight contract. The interpretation remains conditional on hard controls and functional margins.

### Narrative Attack

**Strongest "that's obvious" dismissal:** You tuned a byte-to-Qwen adapter until Qwen layers liked their own embedding-like inputs, then chose controls that were easy to break.

**Strongest "that's trivial" dismissal:** A separate adapter for a separate input distribution and a stronger perturbation for a failed control are routine engineering. They do not show reasoning transfer.

**What the result would need to be for the narrative to be unkillable:** v1 passes Stage 1 with large margins across seeds and domains, improves a train-safe MCQ margin preflight, beats a fair generic pretrained control, and shows that disruptions lose functional-margin lift while correct inverse recovery restores it.

### Attack On The Next Defense

The next defense will say the generic pretrained control answers whether this is Qwen-specific geometry. That is only true if the control is correctly defined. "Different Qwen layers" is not generic pretraining.

---
## Iteration 79: The Generic Pretrained Control Minefield

### Steelman

Batch 11 demanded a hard control:

```text
Does the byte adapter need this teacher's coordinate system, or does any pretrained language transformer become useful once a small adapter feeds it?
```

This is a necessary question. Random, shuffled, and rotated controls are broken by design. A real pretrained control could plausibly work. If it works, the claim changes from source-specific coordinate inheritance to generic pretrained language geometry. That is weaker, but still potentially useful.

Testing Qwen layers from a different range, such as layers 14-17 instead of 0-3, is an easy first step. It uses the same local checkpoint and hidden size. It can test whether the v0 signal is specific to early layers or whether many Qwen layer blocks can process the adapter output.

### Attack

Different Qwen depth is not a generic pretrained control.

It is:

```text
same checkpoint
same tokenizer
same embedding table
same LM head
same residual dimensionality
same pretraining distribution
same family quirks
different specialization depth
```

That tests depth specificity, not genericity. It cannot distinguish "Qwen coordinate system" from "any pretrained coordinate system" because it is still Qwen's coordinate system.

If middle Qwen layers also work well, there are several interpretations:

1. The adapter output is Qwen-manifold-like enough for many Qwen layers.
2. Qwen residual space has broad family-level compatibility across depth.
3. The Qwen LM head and normalization dominate the NLL measurement.
4. The metric is detecting token-distribution compatibility, not reasoning.
5. Early-layer inheritance is not special.

That does not kill coordinate inheritance, but it narrows it:

```text
The result is Qwen-family residual compatibility, not necessarily inherited early reasoning geometry.
```

If only early Qwen layers work, that also cuts both ways. It supports a clean layer-order story, but it may mean the signal is lexical/embedding-level rather than reasoning-level:

```text
The codec+adapter reconstructs the teacher embedding surface well enough for early token-statistical processing, but middle/deep reasoning transforms do not survive the byte bridge.
```

This would be a warning, not a triumph.

### The Strongest Test Of Qwen-Specific Geometry Vs Any Pretraining

The strongest test must be core-native. A non-Qwen pretrained model should not be forced through a Qwen head or a Qwen tokenizer if the claim is about a plausible alternative coordinate system.

A fair generic-pretrained control needs:

| Component | Main Qwen path | Generic control path |
|---|---|---|
| Tokenization for labels | Qwen tokenizer | control model tokenizer |
| Adapter target | Qwen embeddings | control model embeddings |
| Core | copied Qwen layers | copied control-model layers |
| Head/norm | Qwen norm/head | control model norm/head |
| Adapter params | <= same budget | <= same budget |
| Training data | same bytes | same bytes |
| Metrics | normalized NLL lift, true-embedding closure, MCQ margins | normalized NLL lift, true-embedding closure, MCQ margins |

The control should be allowed to win. If the control cannot win by construction, it is not a control.

Useful tiers:

| Tier | Control | What it tests |
|---|---|---|
| 1 | Qwen same checkpoint, different depth | depth specialization |
| 2 | Qwen-family different checkpoint or adjacent model | family-level coordinates |
| 3 | same hidden-size non-Qwen LM, native head | generic pretrained language coordinates |
| 4 | different architecture/family with projection adapter | architecture robustness |
| 5 | pretrained on narrow/different-domain text | language/domain prior |
| 6 | random/permuted-text pretrained | architecture/training dynamics without semantics |
| 7 | tokenized compressed sibling | byte wrapping vs token compression |

For Stage 1 NLL, raw NLLs across different heads/vocabs are not directly comparable. The comparison should be normalized inside each model:

```text
normalized lift =
  (random_same_arch_nll - copied_pretrained_nll)
  /
  (random_same_arch_nll - true_embedding_same_core_nll)
```

Then compare:

```text
Qwen normalized lift vs generic normalized lift
Qwen functional margins vs generic functional margins
Qwen teacher-agreement margins vs generic teacher-agreement margins
```

For the moonshot story, benchmark margins matter more than NLL. If a non-Qwen pretrained control gets similar HellaSwag/PIQA/ARC lift, the honest claim is:

```text
pretrained language geometry helps byte models
```

not:

```text
Qwen reasoning geometry was transplanted
```

### What Survived

Testing Qwen middle layers is useful, but only as a depth curve. It can answer whether early-layer inheritance is special or whether the adapter creates a general Qwen residual stream.

The generic-pretrained demand survives and becomes stricter: the control must be native to its own tokenizer/head and evaluated by normalized lift and functional margins.

### What Died

This shortcut died:

```text
If layers 14-17 fail, then generic pretrained controls are handled.
```

No. Layers 14-17 are same-model different-depth. They are not generic pretraining.

### Narrative Attack

**Strongest "that's obvious" dismissal:** You tested Qwen against broken Qwen and then called other Qwen layers "generic." The experiment never left the teacher's coordinate family.

**Strongest "that's trivial" dismissal:** Pretrained layers are useful. If Llama, Pythia, SmolLM, or Qwen all work once given a small adapter, the result is ordinary pretrained initialization for byte inputs.

**What the result would need to be for the narrative to be unkillable:** A fair native-head non-Qwen pretrained control, with equal adapter budget and equal data, gets far less NLL-normalized lift and far less MCQ margin lift than the inherited Qwen path, while a Qwen-family control lands in an interpretable middle zone.

### Attack On The Next Defense

The next defense will point to the 6.13-nat NLL advantage. But NLL advantage is not benchmark advantage. The project has already seen BPB/NLL improvements fail to move HellaSwag.

---

## Iteration 80: NLL -> Benchmark Translation Probability

### Steelman

The token-end copied advantage is enormous: 6.13 nats/token over random, with a 95% bootstrap CI around [5.96, 6.32]. Patch-boundary advantage is also enormous: 4.70 nats/token.

This is not a tiny loss improvement. In ordinary LM scaling, lower validation loss tends to correlate with better downstream performance across model families and scales. Gadre et al. model a relationship between perplexity and downstream top-1 error across many models. More generally, next-token loss is not a useless metric; it is the training objective that produces most modern language-model capability.

The generous inference is:

```text
If copied inherited layers are 4-6 nats/token better than random on byte-derived Qwen-token prediction, then some continuation-ranking advantage is likely.
```

### Attack

The historical evidence supports a weak-to-moderate prior, not a strong one.

Across normal pretrained language models, validation loss and benchmarks often co-move because the models differ by broad quality: more data, more parameters, better optimization, and better representations. That does not automatically transfer to this local intervention:

```text
same frozen byte codec
same small adapter type
same truncated 4-layer Qwen-shaped core
same Qwen head
NLL measured on Qwen token labels
benchmark target is candidate continuation ranking
```

This is not a normal family scaling curve. It is a highly structured graft.

The local project evidence is hostile:

| Prior result | Surface metric | Benchmark/judgment |
|---|---|---|
| E1 post-training KD | BPB improved ~33% | HellaSwag +0.56pp |
| Option C from scratch KD | BPB improved ~42% | HellaSwag roughly +0.7pp |
| Wide7 | much better BPB and byte accuracy | reasoning benchmarks flat |
| toy readout work | hidden information present | real S0 HellaSwag energy probe flat |
| codec Phase 1 | token retrieval signal real | semantic addressability unproven |

External evidence is also mixed. Gadre et al. show loss can be linked to downstream error in controlled scaling settings, but Isik et al. explicitly show cases where cross-entropy improves while downstream task scores fluctuate or worsen under distribution/task misalignment. Paloma argues against treating one aggregate perplexity as universal fit, and notes that frequent strings can dominate loss. Schaeffer et al. warn that metric choices can make capability curves look qualitatively different from smoother underlying changes.

The v0 NLL number has additional local caveats:

- Token-end true-embedding NLL through the same truncated setup is 12.009 with only 0.844% next-token accuracy. Being within 0.042 nats of that is not being within 0.042 nats of full Qwen.
- Copied calibrated token-end next-token accuracy is 0.704%. Patch-boundary is 0.449%. The model can be much less bad than random and still not rank hard continuations well.
- NLL can be dominated by common tokens, punctuation, articles, spaces, and continuation boilerplate.
- HellaSwag rewards choosing the least absurd completion among plausible continuations. Many choice-critical differences sit on content words, verbs, physical affordances, and event order, not on high-frequency local tokens.
- Candidate ranking is a difference of sequence log-likelihoods. A large shared improvement on all choices can cancel out in the gold-vs-distractor margin.

### Probability Estimate

Question:

```text
Given the v0 6.13-nat token-end copied advantage, what is the realistic probability this translates to >= +8pp HellaSwag over Wide7?
```

My adversarial price:

```text
Uncompressed/prototype token-space Stage 2: 15-25%
Compressed <=121M Stage 3/5 path: 5-10%
```

The signal is too large to price at 1-2%. But the project's history and the metric mismatch make 50%+ unjustified.

I would update sharply upward only after a functional-margin preflight:

| Evidence | Update |
|---|---|
| inherited frozen path beats Wide7 by >= +3pp on train-safe HellaSwag proxy | meaningful positive update |
| gold-vs-best-wrong margin improves by >= 0.05 nats/token equivalent | stronger than accuracy alone |
| inherited beats adapter+random and generic pretrained controls on margins | causally useful |
| NLL lift correlates with margin lift across examples | NLL signal is aligned |
| rare/content-token slices retain large lift | less likely to be function-word surface gain |
| token-end and patch-boundary margin lifts agree | less likely to be token-boundary artifact |

### What The NLL Distribution Must Look Like

For 6 nats NLL advantage to plausibly imply >= +8pp HellaSwag, the gain must be distributed like this:

| Slice | Required pattern |
|---|---|
| Common function tokens | not the dominant source of lift |
| Content nouns/verbs/adjectives | large retained lift |
| Rare tokens | >= 50% of frequent-token lift |
| High-entropy positions | lift concentrated where teacher has multiple plausible continuations |
| Candidate-distinguishing positions | gold completion improves more than best distractor |
| Long-range event tokens | lift persists after the first few choice tokens |
| HellaSwag activity/event words | lift appears on words tied to physical/social plausibility |
| Patch-boundary positions | >= 70% of token-end functional-margin lift |

The most important measurement is not:

```text
NLL(main) < NLL(random)
```

It is:

```text
[NLL(gold) - NLL(best_wrong)]_main
is better than
[NLL(gold) - NLL(best_wrong)]_controls.
```

### What Survived

NLL is still a useful preflight. A 6-nat copied-vs-random gap is not noise. It says the byte-derived stream is much more compatible with copied Qwen layers than with random or shuffled layers.

### What Died

This inference died:

```text
6 nats/token copied advantage makes >= +8pp HellaSwag likely.
```

No. It makes benchmark lift plausible enough to test, not likely enough to believe.

### Narrative Attack

**Strongest "that's obvious" dismissal:** Language models get lower NLL when given better token-like embeddings. That says nothing about whether they can choose the correct HellaSwag ending.

**Strongest "that's trivial" dismissal:** The copied path predicts frequent Qwen tokens less terribly than random layers. This is token-surface repair, not reasoning.

**What the result would need to be for the narrative to be unkillable:** The NLL lift must have a benchmark-margin shadow: content-token lift, rare-token lift, gold-vs-distractor margin lift, Qwen preference agreement, and control collapse all point in the same direction.

### Attack On The Next Defense

The next defense will lean on the narrative: "within 0.04 nats of true embeddings." That phrase is rhetorically powerful and technically narrow.

---
## Iteration 81: The Narrative Under Pressure

### Steelman

The current narrative is strong because it ties together three concrete facts:

```text
A 263K-parameter adapter maps frozen byte-codec states into Qwen embedding gauge.
Copied Qwen layers, not the adapter alone, explain the NLL lift.
At token-end, calibrated copied layers are within 0.042 nats of true Qwen embeddings under the same truncated copied-core/head evaluation.
```

The adapter-does-the-work story is materially weakened by v0:

```text
random_calibrated ~= random baseline
copied_calibrated >> random_calibrated
adapter params = 263K
```

So the best positive story is:

```text
The byte codec plus tiny adapter enters a teacher-compatible coordinate gauge; once there, copied pretrained layers supply real computation that broken controls cannot reproduce.
```

That is a serious component signal.

### Attack

The narrative overclaims at three points.

#### "Transplants reasoning geometry"

No reasoning has been shown.

The current evidence is token-space NLL through a Qwen head. There is no HellaSwag/PIQA/ARC result, no candidate-margin result, no teacher preference agreement result, no semantic counterfactual result, and no byte-native robustness result.

The code scores sequences of Qwen token labels. That is compatible with geometry transfer, but it is also compatible with:

```text
lexical embedding manifold repair
frequency-prior reconstruction
Qwen head compatibility
local next-token smoothing
teacher-token identity reconstruction
```

Those are not reasoning.

#### "Through a byte codec"

The byte codec is frozen, but it was trained to retrieve Qwen token embeddings. That makes it a byte-to-Qwen-token channel, not evidence by itself of byte-native reasoning.

The current path is:

```text
bytes -> codec trained against Qwen embeddings -> adapter -> Qwen layers -> Qwen head
```

An adversarial reviewer will call this:

```text
a frozen byte tokenizer emulator feeding a truncated token model
```

To make "through a byte codec" meaningful, the result must show an advantage that a tokenized sibling does not get, or robustness/cross-tokenizer behavior that makes bytes load-bearing.

#### "Within 0.04 nats"

Within 0.04 nats of what?

Not native Qwen. Not full Qwen inference. Not full teacher HellaSwag behavior. It is within 0.04 nats of:

```text
true Qwen embeddings -> copied first 4 Qwen layers -> copied Qwen norm/head
on the sampled token-end anchor stream
```

That upper bound itself has:

```text
token-end NLL = 12.009
next-token accuracy = 0.844%
```

So the precise claim is:

```text
The adapter nearly matches the true-embedding input for this truncated 4-layer Qwen-head token-NLL task.
```

That is impressive. It is not the same as:

```text
The byte model nearly matches Qwen reasoning.
```

### The Unkillable Narrative

An unkillable narrative would avoid prestige words until the controls earn them.

Current overclaim:

```text
A 263K-parameter adapter transplants pretrained reasoning geometry through a byte codec so faithfully that the result is within 0.04 nats of the teacher's native embeddings.
```

Harder, accurate narrative:

```text
A frozen byte codec plus a 263K adapter can put byte-derived states into a Qwen-compatible gauge where copied Qwen layers, not adapter-only or broken controls, recover teacher-like token prediction. The open question is whether that coordinate compatibility carries functional benchmark margins and survives compression.
```

Unkillable narrative requires this result:

```text
A <=121M active byte-input model, using inherited coordinates with <=2M adapter budget and no broad core retraining, reaches >=42.7% HellaSwag, beats SmolLM2-135M on at least 3 of the 5 Vision benchmarks, beats Wide7 by >=8pp, beats the best fair generic-pretrained byte-wrapped control by >=3pp HellaSwag, survives tokenized-sibling and byte-robustness tests, and shows that coordinate disruptions destroy both NLL and candidate-margin lift while exact inverse recovery restores them.
```

Even that should be worded carefully:

```text
evidence for inherited coordinate geometry
```

not:

```text
proof that reasoning is geometry
```

### What Survived

The adapter attribution result survived. The adapter-only story is badly damaged by v0 because the random calibrated core does not recover meaningful lift.

The component claim survives:

```text
copied Qwen layers are doing something load-bearing for token-space NLL.
```

### What Died

This public-facing sentence died:

```text
The result is within 0.04 nats of the teacher's native embeddings.
```

It needs qualification every time. Without qualification, it is misleading.

### Narrative Attack

**Strongest "that's obvious" dismissal:** You trained a small bridge into Qwen's own embedding space and showed that Qwen's own layers liked it.

**Strongest "that's trivial" dismissal:** This is adapter-mediated transfer learning through a token head. It has not beaten a benchmark, a generic pretrained control, or a tokenized sibling.

**What the result would need to be for the narrative to be unkillable:** A benchmark-winning, compressed, byte-input model where adapter-only, random, shuffled, rotated, generic-pretrained, and tokenized-sibling controls all fail in the predicted ways, while functional margins track the teacher.

### Attack On The Next Defense

The next defense will say Stage 1 is only the first stage; the later gates handle benchmarks and compression. That is true. So price the whole endgame, not just the next repair.

---

## Iteration 82: The Endgame Probability

### Steelman

The five-stage gate system is disciplined:

1. Stage 1: codec/gauge preflight.
2. Stage 2: uncompressed byteified inheritance benchmark gate.
3. Stage 3: 121M compression gate.
4. Stage 4: byte-native story gate.
5. Stage 5: moonshot promotion against SmolLM2/CBD-class targets.

This prevents one clean component result from becoming public victory language. It also makes the direction falsifiable at multiple points.

Given v0's huge NLL gap, v1 has earned a serious chance to try Stage 2 if it passes Stage 1 cleanly.

### Attack

The path has severe attrition.

My current probability estimates from where we are:

| Milestone | Conditional probability | Main reason |
|---|---:|---|
| v1 passes Stage 1 cleanly | 60-70% | failures were borderline and repairs are targeted |
| Stage 2 reaches >=35% HellaSwag and >=+8pp over Wide7 | 25-40% conditional on Stage 1 | NLL may not become candidate-margin lift |
| Stage 3 reaches >=35% at <=121M active | 20-35% conditional on Stage 2 | compression may delete exactly the useful circuits |
| Stage 4 proves byte-native advantage | 20-35% conditional on Stage 3 | tokenized sibling likely strong on clean text |
| Stage 5 beats SmolLM2/CBD-class target | 10-20% conditional on Stage 4 | 42%+ at <=121M active is a high bar |

Multiplying naively gives a very low number. Dependencies are not independent: if Stage 2 is very strong, Stage 3/5 probabilities improve. But from the current evidence, the honest all-in probability of reaching Stage 5 is roughly:

```text
3-8%
```

If v1 passes Stage 1 cleanly and also shows functional-margin shadow evidence, I would update to:

```text
8-15%
```

Without margin evidence, a green Stage 1 alone should not update much above the low single digits for Stage 5.

### Failure Modes By Stage

#### Stage 1: Codec/Gauge Preflight

Likely failure modes:

- readout-conditioned adapter passes NLL but fails rare/content/margin slices;
- stronger rotation destroys inputs in a way that proves little;
- generic pretrained control recovers too much lift;
- layer-depth result is non-monotonic and post hoc;
- pass is one-seed/one-data-slice fragile;
- functional-margin preflight is flat.

Stage 1 can die even while copied-vs-random NLL remains huge.

#### Stage 2: Uncompressed Byteified Inheritance

Likely failure modes:

- NLL lift cancels across all HellaSwag choices;
- HellaSwag score lands 28-32%, not >=35%;
- main beats random/shuffled but not Wide7 by +8pp;
- token-end readout works, patch-boundary readout fails;
- Qwen preference agreement is weak;
- generic pretrained control is close;
- benchmark mode's token-space continuation scoring is too far from real byte decoding.

This is the most likely death point because it directly tests the historical BPB/NLL-to-benchmark disconnect.

#### Stage 3: 121M Compression

Likely failure modes:

- full/uncompressed inherited model works but 121M loses lift;
- important capability lives in depth/MLP memory/head structure pruned away;
- adapter/core drift becomes ordinary retraining;
- 121M dense cannot store/retrieve enough teacher function;
- active MoE meets active params but not simplicity/inference goals;
- random/generic controls catch up under equal compression budget.

If Stage 2 passes, Stage 3 becomes the next most likely death point.

#### Stage 4: Byte-Native Story

Likely failure modes:

- tokenized compressed sibling wins clean benchmarks;
- byte model shows no typo/OCR/Unicode/OOV advantage;
- byte decoder/BPB is poor;
- byte I/O is only a tokenizer emulator around token-shaped reasoning;
- cross-tokenizer teacher transfer is not demonstrated;
- streaming/partial-token benefits do not appear.

This stage can demote the byte story even if coordinate inheritance improves benchmarks.

#### Stage 5: Moonshot Promotion

Likely failure modes:

- HellaSwag stalls in 35-40%;
- PIQA/ARC/WinoGrande/MMLU do not move together;
- SmolLM2 remains stronger on 3/5 Vision benchmarks;
- CBD-like 42.65% remains out of reach;
- result is real but not stop-scrolling;
- public narrative must be narrowed to "controlled inherited-coordinate signal."

### Most Likely Death Point

The most likely death point is Stage 2:

```text
NLL compatibility does not produce enough HellaSwag/PIQA/ARC candidate-margin lift.
```

If Stage 2 survives cleanly, then Stage 3 compression becomes the next most likely death point.

### What Survived

Coordinate inheritance remains the strongest mainline because it attacks the right failure mode: 121M byte models do not discover benchmark-grade semantic coordinates from scratch under this compute budget.

The endgame is not impossible. A 3-8% moonshot probability is enough to justify one disciplined v1, not enough to relax gates.

### What Died

This optimism died:

```text
The massive Stage 1 signal makes Stage 5 plausibly likely.
```

No. It makes Stage 2 worth testing after a clean Stage 1. Stage 5 remains a long-shot until benchmark margins appear.

### Narrative Attack

**Strongest "that's obvious" dismissal:** Of course copied pretrained layers can improve a token-head preflight. The hard part is beating small pretrained models after compression.

**Strongest "that's trivial" dismissal:** A 30-35% HellaSwag result would be a useful engineering signal, not a paradigm shift. CBD and SmolLM2 are the public bar.

**What the result would need to be for the narrative to be unkillable:** The whole ladder works: Stage 1 clean, Stage 2 >=35% and +8pp, Stage 3 <=121M retention, Stage 4 byte-native advantage, Stage 5 SmolLM2/CBD-class comparison won under controls.

### Attack On The Next Defense

The next defense will say the gate system already protects against overclaim. But the gates themselves may be arbitrary or wrong in kind.

---
## Iteration 83: Should The Gate Thresholds Be Re-examined?

### Steelman

The Stage 1 thresholds were useful because they forced v0 to self-kill before benchmark escalation.

They were also concrete:

| Gate | Purpose |
|---|---|
| copied advantage >=2 nats | inherited core must beat random |
| gap closure >=60% or absolute gap small | adapter must repair codec-to-Qwen gauge |
| frozen-core gain >=70% | copied core must be load-bearing before adaptation |
| no-inverse rotation <=30% retained | advantage must depend on coordinate organization |
| inverse recovery >=80% | rotation machinery must be algebraically sane |
| adapter <=2M | adapter cannot be the hidden model |

The fact that v0 failed two close gates and was killed means the gate culture is working.

### Attack: Thresholds Too Lenient

Several gates are too easy to pass for the wrong reason.

#### Copied advantage >=2 nats

Random Qwen-shaped layers with a Qwen head are an extremely weak baseline. A trained transformer beating random layers does not imply source-specific geometry. v0 got 4.70-6.13 nats, but the baseline may be so broken that the effect size overstates interpretive strength.

Required repair:

```text
main must beat the best fair generic pretrained control, not only random and shuffled controls.
```

#### Gap closure >=60%

Closure is inflated when the random baseline is terrible. If random NLL is very bad, a model can close a large percentage of the random-to-true gap without being near useful behavior.

Required repair:

```text
absolute gap, normalized lift, and functional margin shadow must all be reported.
```

#### True-embedding gap

The true-embedding upper bound is not full Qwen. It is true embeddings into the same truncated copied-core/head. At token-end, true NLL is still 12.009 and next-token accuracy is only 0.844%.

Required repair:

```text
rename it "truncated true-embedding upper bound" and stop using it rhetorically as native Qwen proximity.
```

#### Frozen-core gain >=70%

The ratio depends on a 5-step finetune path that also trains the adapter. It is not a stable estimate of how much the copied core contributes under meaningful adaptation.

Required repair:

```text
report ratio across finetune budgets, adapter-frozen variants, LoRA/full-core variants, and held-out domains.
```

#### Rotation <=30% retained

The 30% threshold is a round number. Input-only rotation is also not a full model-basis transform. Passing it can mean the broken input is unnatural, not that reasoning geometry was disrupted.

Required repair:

```text
use predeclared full-basis transforms and measure functional-margin retention, not only NLL retention.
```

#### Missing gates

Stage 1 lacks:

- generic pretrained controls;
- adapter+LM-head-only control;
- core-specific adapter controls;
- functional-margin preflight;
- token-frequency/content/rare slices;
- first-byte/inside-token slices;
- tokenized sibling;
- repeated seeds;
- data-domain slices.

The current Stage 1 can pass while still being:

```text
surface-compatible, generic-pretrained, non-byte-native, and benchmark-flat.
```

### Attack: Thresholds Too Strict Or Wrong In Kind

Some gates may also punish legitimate geometry.

#### Rotation collapse may be too strict

A transformer may have partial gauge-invariant structure: norm statistics, frequency directions, layernorm behavior, attention pattern priors, or lexical clusters that survive a rotation. Some partial retention may be expected and good. Requiring near-total collapse can reward over-destructive controls.

Better question:

```text
Does no-inverse disruption destroy the teacher-specific functional margins that matter, and does inverse recovery restore them?
```

Not:

```text
Does every NLL lift component vanish?
```

#### Frozen-core >=70% may be too rigid

Byte inputs are not native Qwen tokens. Some low-rank adaptation may be the honest mechanism by which inherited geometry becomes usable. If 60% frozen plus small, low-drift adaptation yields strong benchmark margins and preserves teacher function, it might be better evidence than 72% frozen NLL with no margins.

Better classification:

| Pattern | Claim |
|---|---|
| frozen core carries >=70% benchmark lift | coordinate transfer |
| low-rank/small-drift core carries >=70% benchmark lift | coordinate adaptation |
| full finetune creates most lift | good initialization/retraining |

#### 60% gap closure is not obviously meaningful

The right required closure depends on the upper bound, task, readout, and control distribution. A fixed 60% could be too low for token-end and too high for patch-boundary if patch-boundary is inherently noisier.

Better approach:

```text
calibrate thresholds against null and plausible-control distributions, not round numbers.
```

### Were The Thresholds Set With Good Priors?

They were useful but not well-calibrated.

They look like first-pass adversarial round numbers chosen to prevent obvious self-deception. That was appropriate for v0. But after v0 produced a huge NLL signal with two borderline failures, v1 needs a more statistical framework:

1. Define story classes before the run: geometry transfer, geometry adaptation, good initialization, adapter does the work, generic pretraining, surface compatibility.
2. For each metric, identify which story it separates.
3. Use continuous effect sizes and confidence intervals.
4. Calibrate thresholds from nulls and plausible controls.
5. Require functional-margin evidence before benchmark escalation.
6. Treat a near-threshold pass as weak evidence unless it generalizes across seeds/slices.

### Revised Gate Philosophy

Do not ask:

```text
Did every precommitted number turn green?
```

Ask:

```text
Which causal story is still alive after the result?
```

Minimum v1 interpretation table:

| Result pattern | Honest label |
|---|---|
| NLL pass, margins flat | surface compatibility |
| NLL + margins pass, generic close | generic pretrained language geometry |
| NLL + margins pass, generic loses, tokenized sibling wins | Qwen-specific but byte wrapper not load-bearing |
| frozen/low-drift inherited core carries benchmark lift, controls fail | coordinate inheritance evidence |
| full finetune creates lift | good initialization/retraining |

### What Survived

The original Stage 1 gates were valuable as a first kill switch. They prevented premature benchmark runs from a failed preflight.

### What Died

This assumption died:

```text
The Stage 1 threshold values are themselves epistemically authoritative.
```

No. They are guardrails. The interpretation has to move from pass/fail thresholds to causal story classification.

### Narrative Attack

**Strongest "that's obvious" dismissal:** You set arbitrary round-number gates, barely missed two, then designed v1 to barely pass them.

**Strongest "that's trivial" dismissal:** Random/shuffled/rotated controls and NLL thresholds are standard sanity checks. They do not constitute a theory of reasoning transfer.

**What the result would need to be for the narrative to be unkillable:** The gates are calibrated against fair controls and tied to functional outcomes; the model passes with margin, not by one or two percentage points; and the surviving story is coordinate inheritance rather than surface compatibility.

### Attack On The Next Defense

The next defense will say the exact thresholds can be refined later. But there is one unknown that cannot be repaired by threshold tuning if it goes the wrong way.

---

## Iteration 84: The Single Most Dangerous Unknown

### Steelman

After 11 Q-loop batches and v0 Stage 1, the direction has learned a lot:

- ordinary byte KD improved surface loss but not HellaSwag;
- hidden-state alignment was non-identifying across architectures;
- toy readout gaps did not transfer to real HellaSwag;
- width/depth changes improved BPB but not reasoning;
- evidence-native v0 failed against controls;
- the semantic codec learned real byte-to-token embedding retrieval;
- coordinate inheritance produced a huge copied-vs-random NLL signal;
- adapter-only is unlikely to explain v0's NLL lift.

So the strongest current hypothesis is:

```text
Small byte models need inherited pretrained coordinates because they cannot discover benchmark-grade semantic address space from scratch under available compute.
```

### Attack: The Unknown

The single most dangerous unknown is:

```text
Does the inherited-coordinate signal contain task-discriminative teacher function, or only lexical/token-manifold compatibility?
```

This is more dangerous than a technical gate failure.

If patch-boundary frozen-core gain stays at 66%, that can be repaired. If rotation retention is 33%, the control can be improved. If 4 layers are suboptimal, depth can be tuned. If the adapter is too simple, a still-small adapter can be tried.

But if the 4-6 nat NLL lift is mostly:

```text
common token prediction
Qwen embedding manifold repair
Qwen head compatibility
frequency/hub structure
local lexical smoothing
```

and not:

```text
gold-vs-distractor candidate margin
teacher preference agreement
event/commonsense distinction
content-token discrimination
robust semantic addressability
```

then the whole coordinate-inheritance direction is conceptually wrong as the moonshot path.

It would mean:

```text
The byte codec can feed Qwen-like token surfaces into copied Qwen layers, but the transferred object is not the function that makes the teacher useful on reasoning benchmarks.
```

That is fatal because the Vision does not need a better token-surface component. It needs a 121M-ish model that beats strong small baselines on benchmarks.

### Why This Is The Most Dangerous

It subsumes the other risks:

- Generic pretrained controls matter because lexical/token-manifold compatibility may be generic.
- NLL-to-benchmark uncertainty matters because lexical lift may not become candidate margins.
- Narrative risk matters because "reasoning geometry" may be just "embedding compatibility."
- Compression risk matters because benchmark-relevant function may not be in the part of the inherited signal that compresses.
- Byte-native risk matters because a tokenized sibling may capture lexical compatibility more cheaply.

If this unknown resolves negatively, coordinate inheritance may still be an interesting codec diagnostic, but not the moonshot.

### Cheapest Experiment To Test It

The cheapest decisive experiment is not more training. It is a functional-margin shadow test using the existing Stage 1 machinery and saved adapter.

Run on train-safe HellaSwag/PIQA/ARC subsets only, no public benchmark claim:

| Variant | Purpose |
|---|---|
| inherited copied core | main |
| adapter + random core | adapter-only |
| shuffled copied core | broken depth |
| no-inverse rotation | coordinate disruption |
| inverse rotation | sanity recovery |
| best fair generic pretrained control | any-pretraining |
| true-embedding truncated Qwen path | local upper bound |
| Wide7 | byte baseline |

Metrics:

| Metric | Kill signal |
|---|---|
| train-safe MCQ accuracy | main <= +1pp over Wide7 |
| gold-vs-best-wrong margin | no meaningful lift over controls |
| Qwen pairwise preference agreement | main not above random/generic by >=5pp |
| NLL lift -> margin lift Spearman | <0.20 |
| content-token lift | much smaller than function-token lift |
| rare-token lift | <50% of frequent-token lift |
| patch-boundary margin lift | <50-70% of token-end margin lift |
| shared-choice NLL cancellation | all choices improve similarly, margin flat |

Precommit the kill condition:

```text
If v1 passes Stage 1 NLL but the functional-margin shadow is flat, label the result PASS_SURFACE_COMPATIBILITY / FAIL_FUNCTIONAL_GEOMETRY and block Stage 2 benchmark escalation.
```

This experiment is cheap because it uses the existing adapter and scoring code. It does not require GPU training. It only requires careful candidate scoring and slicing.

### What Survived

The direction is still worth testing because v0 killed the adapter-only story and produced a large copied-core signal. That is rare enough to pursue.

### What Died

This comfort died:

```text
Because the Stage 1 NLL signal is massive, the remaining risks are mostly engineering.
```

No. The biggest remaining risk is conceptual: the measured signal may be the wrong kind of signal.

### Narrative Attack

**Strongest "that's obvious" dismissal:** You rediscovered that pretrained token models like token-like inputs. The byte codec is a tokenizer emulator, not a reasoning bridge.

**Strongest "that's trivial" dismissal:** The whole result is lexical manifold matching. Useful, but far from a paradigm shift.

**What the result would need to be for the narrative to be unkillable:** The same signal that improves NLL also improves candidate margins on hard commonsense examples, survives patch-boundary byte readout, loses under coordinate disruption, recovers under inverse repair, beats generic pretrained controls, and remains after compression.

### Attack On The Next Defense

The next defense will say "run Stage 2 and find out." The answer is yes, but only after v1 Stage 1 includes a functional-margin shadow. Running public-style benchmarks from an NLL-only pass risks repeating the exact project failure mode: mistaking a real surface signal for transferable intelligence.

---

## Batch 12 Final Verdict

v1 repairs should proceed, but the interpretation must be tightened before any promotion.

The current state is:

```text
Coordinate-inheritance v0: killed.
Coordinate-inheritance direction: alive.
Stage 1 NLL signal: real and large.
Adapter-does-the-work story: strongly damaged.
Reasoning-geometry story: not yet established.
NLL-to-benchmark translation: unpriced until functional-margin shadow.
Moonshot probability from current evidence: low single digits to high single digits.
```

The next W-loop should not merely make the two failed gates pass. It should make the result harder to misinterpret.

Required v1 additions before Stage 2:

1. Treat Qwen middle-layer tests as a depth curve, not generic pretrained control.
2. Add fair native-head generic pretrained controls or explicitly mark them missing.
3. Add functional-margin shadow tests on train-safe benchmark subsets.
4. Slice NLL/margins by token frequency, content/function class, readout, and candidate-distinguishing positions.
5. Use stronger/full-basis disruption only if predeclared and measured on margins as well as NLL.
6. Report gate results as causal story classification, not just pass/fail.

The hostile final statement:

```text
If v1 only changes 66.3% to 70%+ and 33% to <=20%, it has repaired a dashboard.
If v1 also shows that inherited coordinates move gold-vs-distractor margins in a way generic/adapted/broken controls cannot, then it has repaired the proof.
```

