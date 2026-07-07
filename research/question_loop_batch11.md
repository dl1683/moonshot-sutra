# QUESTION LOOP - Batch 11: Attack Coordinate-Inheritance Deeper

Date: 2026-07-07

Iterations: 71-77

## Grounding

I read the requested context in order:

1. `research/VISION.md`
2. `research/DEEP_RETHINK.md`
3. `research/dual_loop_supervisor_checkin_6.md`
4. `research/question_loop_batch10.md`
5. `research/question_loop_batch9.md`
6. `research/work_loop_batch5.md`
7. `code/chain_init_codec_probe.py`

Binding state entering Batch 11:

- Evidence-native is permanently dead as a birth mechanism.
- Brainseed is dead as a birth artifact.
- Coordinate-inheritance is the sole mainline, but still `MAINLINE_BUT_NOT_BELIEVED`.
- The only empirical coordinate-inheritance signal is Batch 5's copied-vs-random 4-layer Qwen probe:

| Readout | Random 4-layer NLL | Copied Qwen 4-layer NLL | Copied Advantage |
|---|---:|---:|---:|
| token-end codec input | 17.17 | 15.52 | 1.65 nats/token |
| patch-boundary codec input | 18.34 | 16.62 | 1.72 nats/token |

The same probe also showed the codec penalty is larger than the inheritance advantage:

| Readout | Copied + codec input | Copied + true teacher embeddings | Codec Penalty |
|---|---:|---:|---:|
| token-end | 15.52 | 11.94 | 3.58 nats/token |
| patch-boundary | 16.62 | 12.52 | 4.10 nats/token |

Batch 10 set a five-stage gate system. Batch 11 attacks those gates themselves.

## Batch 11 Thesis

Batch 10 asked:

```text
Can inherited coordinates survive the byte codec, disruption controls, and
compression?
```

Batch 11 asks a harsher question:

```text
If Stage 1 passes, what exactly passed?
```

A nominal Stage 1 pass could mean any of these:

1. `geometry_transfer`: the codec lands in Qwen's coordinate gauge, and copied Qwen layers supply load-bearing reasoning structure.
2. `good_initialization`: pretrained layers are a better starting point than random layers, but the result is ordinary fine-tuning.
3. `adapter_does_the_work`: the calibration adapter learns a token-space compiler, and the inherited core is expensive decoration.
4. `generic_pretraining`: any pretrained language transformer works about as well as Qwen.
5. `nll_surface_match`: token NLL improves, but benchmark judgment remains flat.

Only the first story supports the strong coordinate-inheritance thesis.

## Iteration 71: Calibration-Adapter Attack - The Adapter May Be The Model

### Current Strongest Position

After Batch 10, the defense is:

```text
The codec is imperfect, but a small calibration adapter can map codec states into
the Qwen embedding gauge. Then a frozen inherited Qwen core should outperform
random or disrupted cores.
```

### Attack

This puts the calibration adapter in the center of the proof. That is dangerous.

If the adapter maps:

```text
codec_state -> Qwen-compatible activation
```

then the adapter may be doing most of the work. It may learn token-boundary repair, token-frequency priors, Qwen embedding anisotropy, norm/covariance correction, local next-token smoothing, a low-rank tokenizer emulator, or generic language-like activations that can drive any pretrained core.

If so, copied Qwen layers are not proof of inherited reasoning geometry. They are a large frozen nonlinear readout downstream of an adapter trained to feed them.

The Batch 5 probe already contains the warning sign: the identity codec-to-Qwen-LM-head path is terrible, random layers are terrible, and copied layers are only less terrible. That means the adapter has room to create an apparently large NLL gain by fixing input statistics, without transferring reasoning.

### The Hard Counterfactual

The real comparison is not:

```text
adapter + copied Qwen core > raw codec + random core
```

The real comparison is:

```text
same adapter budget + random/pretrained/generic core
vs
same adapter budget + inherited Qwen core
```

Even more important:

```text
one shared adapter evaluated across cores
vs
core-specific adapters trained separately for each core
```

If core-specific adapters let random or generic-pretrained cores approach copied Qwen, the story is adapter training, not inherited geometry.

### Required New Gate: Adapter Attribution

Define:

```text
Lift_NLL(M) = NLL(random_core_raw_codec) - NLL(M)
Lift_Bench(M) = score(M) - score(Wide7)
```

For Stage 1, the following variants are mandatory:

| Variant | Purpose |
|---|---|
| raw codec + copied core | current inheritance signal |
| calibrated codec + copied core | main Stage 1 candidate |
| calibrated codec + random Qwen-shaped core | adapter-only control |
| calibrated codec + shuffled copied core | broken-geometry control |
| calibrated codec + generic pretrained core | any-pretraining control |
| true Qwen embeddings + copied core | upper bound |
| calibration adapter + Qwen LM head only | tests whether adapter learns a shallow token predictor |

The adapter is classified as load-bearing if:

| Condition | Classification |
|---|---|
| adapter + random core gets >= 50% of main NLL lift | adapter is a major confound |
| adapter + random core gets >= 70% of main NLL lift | adapter is the model for Stage 1 |
| adapter + generic pretrained core is within 1.0 nat/token of main | not Qwen-specific geometry |
| adapter + random/generic benchmark score is within 2pp HellaSwag of main | not coordinate-inheritance evidence |
| core-specific adapters erase copied-vs-random gap to < 0.75 nats/token | adapter training dominates |

Minimum geometry-transfer survival:

| Gate | Required |
|---|---:|
| copied core over adapter + random core | >= 1.5 nats/token on token-end and patch-boundary |
| copied core over adapter + generic pretrained core | >= 0.75 nats/token and later >= 3pp HellaSwag |
| adapter + random core share of main NLL lift | <= 30% |
| adapter + random core share of main benchmark lift | <= 25% |
| adapter params for Stage 1 | <= 2M |
| adapter training budget | separately reported; no hidden large-scale KD |

### What Survived

The adapter is not disallowed. Gauge repair is necessary. But it must be treated as a suspect component, not a neutral pipe.

### What Died

This inference died:

```text
Small adapter + copied core beating random core proves inherited geometry.
```

No. It may prove the adapter learned how to drive a pretrained nonlinear system.

### Attack On The Next Defense

The next defense will say the inherited core should not be frozen forever. Maybe Qwen geometry needs to adapt to byte-space inputs. That may be true, but it opens an even worse confound: when does adaptation become retraining?

## Iteration 72: Fine-Tuning Attack - A Frozen Core May Be Too Strict, But An Unfrozen Core May Prove Nothing

### Current Strongest Position

After Iteration 71, the defense becomes:

```text
The adapter-only controls are fair for Stage 1, but inherited Qwen layers were
trained for token embeddings. They may need some fine-tuning to adapt to codec
inputs. Frozen-core tests are too harsh.
```

### Attack

This is plausible, but it threatens the whole claim.

If the core remains frozen and works, coordinate inheritance has a clean story:

```text
The byte adapter found the teacher gauge; the inherited core already contains the
useful computation.
```

If the core must be fine-tuned heavily, the story becomes:

```text
A pretrained model is a good initialization for another training run.
```

That is useful, but it is not the strong geometry claim. It is the ordinary transfer-learning claim.

The project has already learned this lesson repeatedly: a training signal can improve an easy observable while not transferring judgment. If full-core fine-tuning is allowed to rescue every failure, coordinate inheritance becomes unfalsifiable.

### The Adaptation Budget Must Be Part Of The Claim

There are three different claims:

| Result Pattern | Honest Interpretation |
|---|---|
| frozen core + small adapter gets most of the lift | coordinate geometry transfers |
| low-rank/limited core adaptation gets most of the lift | inherited geometry adapts |
| full-core fine-tuning creates most of the lift | good initialization / retraining |

The difference is measurable.

### Required New Gate: Coordinate Drift Accounting

For every adapted inherited core, report:

| Diagnostic | Required |
|---|---|
| parameter update norm per layer | `||delta W|| / ||W||` |
| residual-stream CKA/RSA vs original Qwen on tokenized text | before and after adaptation |
| Qwen token-input NLL degradation | original token embeddings through adapted core |
| layerwise logit-lens or top-k agreement | original Qwen vs adapted core |
| candidate-margin correlation with original Qwen | held-out HellaSwag/PIQA/ARC train-only contexts |
| adaptation examples/tokens/bytes | exact budget |
| comparison to random and generic pretrained cores at same budget | mandatory |

Classification thresholds:

| Classification | Required Pattern |
|---|---|
| `geometry_transfer` | frozen core + small adapter retains >= 70% of final NLL lift and >= 70% of final benchmark lift |
| `geometry_adaptation` | low-rank or <= 5% trainable core params retain >= 70% of final lift; original-Qwen function degradation <= 10% relative NLL or <= 0.3 nats/token |
| `good_initialization` | frozen core gets < 50% of final lift, but inherited full fine-tune beats random/generic early |
| `retraining` | random/generic catches up within 2x adaptation budget, or inherited advantage appears only after full-core updates |
| `coordinate_destruction` | adapted core loses candidate-margin correlation with original Qwen while benchmarks improve |

Full-core fine-tuning can be an engineering path. It cannot be counted as coordinate-inheritance evidence unless it preserves the original coordinate function by the diagnostics above.

### What Survived

Fine-tuning is allowed as polish. It is not allowed to create the result and then borrow the language of inheritance.

### What Died

This escape hatch died:

```text
If frozen inherited layers fail, just unfreeze them and still call it geometry.
```

No. Once the result depends on broad updates, the burden shifts to proving that the coordinates survived the updates.

### Attack On The Next Defense

The next defense will say Stage 1 is only a cheap NLL preflight. Benchmarks come later. But the project has already seen BPB/NLL improve while reasoning stayed flat. NLL cannot be allowed to certify geometry.

## Iteration 73: NLL-Preflight Attack - Better Token Loss May Not Predict Reasoning

### Current Strongest Position

After Iteration 72, the defense becomes:

```text
We can separate the claims cleanly. Stage 1 only checks codec/gauge compatibility
using token NLL. If it passes, Stage 2 handles benchmarks.
```

### Attack

This repeats the central failure pattern of the entire project.

The project has repeatedly improved surface likelihood while downstream judgment did not move:

| Direction | Likelihood/Surface Result | Benchmark/Judgment Result |
|---|---|---|
| E1 / Option C | BPB improved dramatically | HellaSwag nearly flat |
| Wide7 | BPB much better than S0 | reasoning benchmarks flat |
| byte-marginal KD | better byte prediction | no task transfer |
| Brainseed codec charts | real token-identity signal | downstream scorers lost to codec-only |

So a Stage 1 NLL pass says only:

```text
The byte-derived activation stream is more compatible with Qwen's next-token
surface distribution than a broken control.
```

It does not say:

```text
The inherited coordinates produce better candidate discrimination.
```

HellaSwag/PIQA/ARC are not pure language-modeling metrics. They are candidate-ranking tests. NLL and candidate judgment are correlated in large enough models, but this project exists because that correlation keeps breaking at 121M byte scale.

### Required New Gate: Functional Margin Preflight

Stage 1 must include a tiny benchmark-facing preflight before expensive training.

Use train-only or validation-safe subsets. Do not claim final benchmark numbers from this. Use it only to test whether NLL lift points in the right direction.

Variants:

| Variant | Required |
|---|---|
| calibrated inherited frozen core | main |
| calibrated random core | adapter-only |
| calibrated generic pretrained core | generic pretraining |
| shuffled/rotated copied core | disruption |
| true-Qwen token model | teacher upper bound |
| Wide7 | byte baseline |

Metrics:

| Metric | Why |
|---|---|
| MCQ candidate accuracy on 500-1000 train-safe examples | direct task proxy |
| gold-vs-best-wrong margin | less noisy than accuracy |
| pairwise candidate preference agreement with Qwen | functional geometry |
| per-example correlation between NLL gain and margin gain | checks predictiveness |
| error-overlap with Qwen and Wide7 | identifies whether inherited model fails like teacher or like byte baseline |

Stage 1 NLL is allowed to promote to Stage 2 only if:

| Gate | Required |
|---|---:|
| inherited frozen/calibrated vs Wide7 MCQ proxy | >= +3pp or >= +0.05 mean gold-margin lift |
| inherited vs adapter + random MCQ proxy | >= +2pp |
| inherited vs generic pretrained MCQ proxy | >= +1pp at Stage 1, later >= +3pp at Stage 2 |
| pairwise preference agreement with Qwen | >= 60% and >= 5pp over random/generic controls |
| NLL improvement -> margin improvement consistency | >= 60% of examples with NLL improvement also improve gold margin, or Spearman >= 0.25 |
| shuffled/rotated controls | lose >= 70% of margin lift |

If NLL passes but the functional margin preflight is flat, then Stage 1 should be labeled:

```text
PASS_SURFACE_COMPATIBILITY
FAIL_REASONING_PREFLIGHT
```

That is not permission for a major benchmark run.

### What Survived

NLL remains a useful cheap diagnostic. It can find catastrophic gauge mismatch early.

### What Died

This gate died:

```text
Copied-vs-random NLL >= 2 nats/token means coordinate inheritance is ready for
benchmark training.
```

No. It is ready only if NLL lift also has a candidate-margin shadow.

### Attack On The Next Defense

The next defense will say the benchmark-facing preflight and disruptions define geometry enough. But the word "geometry" is still doing too much work. It needs an operational definition or the claim can absorb any result.

## Iteration 74: Geometry-Definition Attack - Define The Transferable Object Or Stop Using The Word

### Current Strongest Position

After Iteration 73, the defense becomes:

```text
Geometry is not just NLL. It is the teacher's internal coordinate system, tested
by adapter attribution, disruption controls, and functional benchmark margins.
```

### Attack

That is closer, but still underspecified.

What is "reasoning geometry" operationally?

| Candidate Meaning | Problem |
|---|---|
| copied weight matrices | then it is weight transfer / compression |
| residual-stream representation manifold | gauge-dependent and codec-sensitive |
| attention patterns | input statistics change under byte wrapping |
| MLP key-value memories | may be memorized facts, not reasoning geometry |
| loss landscape shape | changes under input distribution shift and fine-tuning |
| benchmark candidate margins | behavior, not internal geometry |
| layerwise teacher trajectory | closer, but still needs controls |

The project cannot use "geometry" as a prestige word for "pretrained model internals." The claim must identify a measurable object that can be preserved, disrupted, repaired, and compressed.

### Operational Definition

For coordinate-inheritance, define:

```text
Reasoning geometry = the teacher-specific layerwise transformation of context
distinctions into candidate/action preference margins, represented in the
teacher residual-stream gauge and preserved under a compact byte-to-gauge adapter.
```

That definition has four observable components:

1. `input_gauge`: codec+adapter states match Qwen embedding/residual statistics.
2. `layer_trajectory`: byteified activations follow the teacher's layerwise representational path better than controls.
3. `functional_margins`: candidate preference margins agree with the teacher on held-out task items.
4. `disruption_repair`: breaking the coordinate organization destroys the advantage; applying the correct inverse restores it.

No single component is enough.

### Required Geometry Diagnostics

| Diagnostic | Required For Geometry Claim |
|---|---:|
| embedding mean/norm/covariance match | report before and after adapter |
| effective rank of adapter output covariance | >= 50% of true Qwen embedding effective rank |
| nearest-neighbor token identity top-1/top-10 by slice | report token-end, patch-boundary, first-byte, rare-token |
| layerwise CKA/RSA to true-Qwen token trajectory | main >= generic/random by >= 0.10 absolute average |
| candidate-margin Spearman with Qwen | >= 0.45 uncompressed, >= 0.35 after 121M compression |
| Qwen pairwise preference agreement | >= 65% uncompressed, >= 60% after compression |
| disruption controls | lose >= 70% of functional margin lift |
| correct inverse rotation | recover >= 80% of functional margin lift |
| adapter-only/random controls | <= 25-30% of functional margin lift |

The thresholds are intentionally not enough to claim benchmark success. They only license the phrase:

```text
inherited coordinate geometry appears load-bearing
```

### What Survived

The geometry claim can be made falsifiable, but only as a conjunction of gauge, trajectory, function, and disruption evidence.

### What Died

This claim died:

```text
If copied Qwen layers work better, then reasoning geometry transferred.
```

Copied layers are evidence only if the teacher-specific function they implement survives in a way that broken, generic, and adapter-only variants cannot reproduce.

### Attack On The Next Defense

The next defense will say the disruption controls make this precise. But controls can be optimized to fail. The hard control is not broken Qwen. The hard control is a plausible pretrained model with different coordinates.

## Iteration 75: Control-Optimization Attack - Easy Controls Let You Win The Wrong Trial

### Current Strongest Position

After Iteration 74, the defense becomes:

```text
We can prove geometry by showing that shuffled layers, random rotations, and
random cores lose the functional-margin lift while copied Qwen retains it.
```

### Attack

Those controls are necessary, but they are still too easy.

Random cores are supposed to lose. Shuffled layers are supposed to lose. Random rotations without inverse are supposed to lose. Those controls prove only:

```text
Broken transformers are worse than non-broken transformers.
```

The hard alternative is:

```text
Any pretrained language transformer gives useful coordinates once a calibration
adapter learns how to feed it.
```

If that is true, the result may still be valuable, but the thesis changes:

```text
from: Qwen reasoning geometry was inherited
to: pretrained language geometry is a useful byte-model initialization prior
```

That is a weaker, more obvious claim.

### Required Hard Controls

Stage 1 and Stage 2 must include controls that could plausibly work:

| Control | Interpretation If It Matches Main |
|---|---|
| same Qwen architecture, different Qwen checkpoint not used for codec targets | family-level geometry, not exact checkpoint |
| same-size non-Qwen token LM with its own byte adapter | generic language geometry |
| same architecture pretrained on different-domain text | pretraining/domain prior |
| same architecture pretrained on permuted/randomized text | architecture/training dynamics without real semantics |
| tokenized compressed Qwen sibling | byte wrapper cost/benefit |
| Qwen layers with only embeddings/LM head changed | tests whether vocabulary head dominates |
| Qwen layers with MLPs reset but attention kept | tests memory vs routing |
| Qwen layers with attention reset but MLPs kept | tests routing vs memory |

The most important comparisons are not the broken ones. They are:

```text
main inherited Qwen vs best plausible pretrained control
main byteified Qwen vs tokenized compressed sibling
main inherited Qwen vs same-family different checkpoint
```

### Control Outcome Taxonomy

| Outcome | Honest Claim |
|---|---|
| main beats all plausible pretrained controls by >= 3pp HellaSwag and >= 2pp aggregate | Qwen-specific coordinate inheritance |
| same-family Qwen matches within 1pp | Qwen-family coordinate inheritance |
| non-Qwen pretrained matches within 2pp | generic pretrained language coordinates |
| different-domain pretrained gets >= 70% of lift | pretraining prior, not teacher reasoning |
| random/permuted-text pretrained gets >= 50% of lift | architecture/statistical prior, not language geometry |
| tokenized sibling dominates clean benchmarks and byte model has no robustness win | byte interface is branding |

Batch 10 required "best generic-pretrained control" to lose by >= 3pp. Batch 11 strengthens that rule:

```text
If the hard controls are not run, the result cannot be described as
teacher-specific geometry. It can only be described as copied-vs-broken lift.
```

### Prediction

The random and shuffled controls will fail badly. That will look satisfying and prove little.

At least one plausible pretrained control will recover a meaningful fraction of the inherited model's Stage 1 NLL lift, likely 40-70%. If the calibration adapter is core-specific, the generic-pretrained control may get even closer. This will force a narrower claim unless benchmark margins separate the main model.

### What Survived

Disruption controls remain mandatory. They just cannot be the only controls.

### What Died

This proof strategy died:

```text
Random/shuffled/rotated controls lose, therefore the inherited Qwen geometry is
special.
```

No. The hard question is whether any real pretrained coordinate system works.

### Attack On The Next Defense

The next defense will say the codec is specifically trained into Qwen embedding space, so generic pretrained controls are not expected to match. That defense points to the next risk: the codec may be a degenerate channel that learns generic language-like activations, not preserved Qwen reasoning coordinates.

## Iteration 76: Degenerate-Codec Attack - The Channel May Emit Language-Like Activations, Not Coordinates

### Current Strongest Position

After Iteration 75, the defense becomes:

```text
The codec is Qwen-targeted. If a generic pretrained control performs worse, that
shows the byte-to-Qwen coordinate channel is specific and meaningful.
```

### Attack

The codec may still be degenerate.

It maps bytes to 256-dim states, then an alignment head maps those states to 1024-dim Qwen embedding-like vectors. Patch-boundary top-1 was only 37.89% in Batch 5 diagnostics, and first-byte top-1 was only 20.08%. That means many patch positions do not recover the intended token identity.

But token identity is not even the full requirement. Qwen layers expect correct token identity, correct position/boundary timing, correct embedding norm, correct anisotropy, correct covariance, correct rare-token relations, correct multi-token phrase geometry, and correct residual scale after repeated layers.

A degenerate codec can learn a shortcut:

```text
emit plausible high-frequency Qwen-like vectors that make the LM head less
surprised on average
```

That can improve NLL without preserving the conditional distinctions needed for reasoning.

### Degenerate Channel Failure Modes

| Failure Mode | Symptom |
|---|---|
| frequency prior channel | common-token slices pass; rare-token slices fail |
| boundary oracle dependence | token-end passes; patch-boundary and first-byte fail |
| covariance mimicry | NLL improves; token identity and margins stay poor |
| generic activation compiler | works with many pretrained cores |
| local smoothing | improves next-token NLL; no candidate-margin lift |
| adapter memorization | per-occurrence random or held-out-domain controls collapse |
| output-head exploitation | Qwen LM head gets easier logits without core reasoning |

### Required Channel Diagnostics

| Diagnostic | Required |
|---|---:|
| per-occurrence random target control | near chance, not fixed-permutation 51x-above-chance behavior |
| token-frequency slices | rare-token lift >= 50% of common-token lift |
| first-byte slice | not allowed to be the silent failure; report and gate |
| patch-boundary vs token-end | patch-boundary must carry >= 70% of token-end functional-margin lift |
| covariance/effective-rank match | adapter output rank >= 50% of true Qwen embedding rank |
| nearest-neighbor entropy | no collapse to frequent-token hubs |
| byte-edit sensitivity | minimal semantic-preserving edits preserve margins; semantic-changing edits move margins |
| same bytes, shuffled context | margins change appropriately; codec cannot ignore context |
| cross-domain slice | no pass based only on training-shard style |

Minimum survival thresholds:

| Gate | Required |
|---|---:|
| patch-boundary inherited functional-margin lift / token-end lift | >= 70% |
| first-byte inherited NLL lift / last-byte inherited NLL lift | >= 50% or design bypasses first-byte scoring explicitly |
| rare-token inherited NLL lift / frequent-token lift | >= 50% |
| nearest-neighbor hub share | top 1% token hubs account for <= 25% of nearest neighbors |
| semantic-preserving paraphrase margin stability | >= 70% agreement |
| semantic-changing counterfactual margin movement | >= 60% expected direction |

If the codec fails these while copied-vs-random NLL passes, the result is:

```text
PASS_GENERIC_ACTIVATION_COMPATIBILITY
FAIL_COORDINATE_CHANNEL
```

### What Survived

The codec may still be the crucial bridge. But "retrieves token identity above chance" is not enough.

### What Died

This assumption died:

```text
If the adapter output works with Qwen layers, then the coordinate structure
survived the byte channel.
```

No. It may be a lossy channel that emits language-like averages.

### Attack On The Final Defense

The final defense will say that if all these gates pass, coordinate inheritance is real enough. But even a real signal can land at 30% HellaSwag. That would be interesting and still fail the Vision.

## Iteration 77: Endgame Attack - A Real 30% Result Is Still Not A Moonshot

### Current Strongest Position

After Iteration 76, the defense becomes:

```text
If coordinate inheritance passes adapter attribution, fine-tuning drift checks,
functional-margin preflight, geometry diagnostics, hard pretrained controls, and
codec channel diagnostics, then the result is real.
```

### Attack

It may be real and still not matter enough.

The Vision standard is not "real signal." It is:

```text
paradigm shift or failure
```

CBD already reports 42.65% HellaSwag at 138M. SmolLM2-135M is the Vision target at 42.1% HellaSwag plus strong PIQA/ARC/WinoGrande/MMLU numbers. A coordinate-inheritance result that lands at 30-35% HellaSwag is not enough to make people question assumptions about small models.

Possible outcomes:

| Result | Scientific Meaning | Moonshot Meaning |
|---|---|---|
| 28-30% HellaSwag at 121M | weak lift over Wide7 | failure as mainline |
| 30-35% HellaSwag with clean controls | serious research signal | not a stop-scrolling result |
| 35-40% HellaSwag with lift retention and hard controls | strong technical result | still below champion target |
| >= 42.7% HellaSwag at <= 121M active with controls | home-run candidate | survives public comparison |
| beats SmolLM2 on 3/5 Vision benchmarks plus HellaSwag | Vision-level result | promotion candidate |

The uncomfortable point:

```text
Stage 3's >=35% HellaSwag gate is enough to keep researching. It is not enough
to declare the moonshot alive in the public sense.
```

### The Specific Result That Survives "Obvious" And "Trivial"

Coordinate-inheritance can survive both attacks, but only with a very specific result.

It survives "that's obvious" only if:

| Requirement | Threshold |
|---|---:|
| adapter + random core share of benchmark lift | <= 25% |
| best plausible pretrained control share of benchmark lift | <= 50% |
| main vs best generic pretrained control | >= +3pp HellaSwag and >= +2pp aggregate |
| shuffled/rotated/no-inverse controls | lose >= 70% of inherited lift |
| correct inverse rotation | recovers >= 80% of inherited lift |
| frozen or low-rank adapted inherited core | retains >= 70% of final lift |
| layerwise/function geometry diagnostics | pass the Iteration 74 thresholds |

It survives "that's trivial" only if:

| Requirement | Threshold |
|---|---:|
| active params | <= 121M active, total params disclosed |
| HellaSwag | >= 42.7% or beats the current CBD 138M number if that number is treated as the target |
| Vision benchmark table | beats SmolLM2-135M on at least 3 of 5 listed benchmarks |
| clean gap to tokenized compressed sibling | within 2pp aggregate or offset by byte robustness wins |
| typo/OCR/Unicode/OOV robustness | byteified model beats tokenized sibling by >= 5pp aggregate or drops <= 50% as much |
| inference | runs on the RTX 5090 laptop with documented VRAM and throughput |

Anything weaker must be named accurately.

| Weaker Result | Honest Label |
|---|---|
| 35% uncompressed but < 35% at 121M | byteified inheritance works, compression failed |
| 35% at 121M but generic pretrained controls are close | pretrained initialization helps byte models |
| 35% at 121M with Qwen-specific controls clean | real coordinate-inheritance signal, not moonshot yet |
| 30% at 121M | interesting diagnostic, not mainline victory |
| tokenized sibling dominates and byte robustness is absent | byte wrapper is not load-bearing |

### What Survived

Coordinate inheritance remains the only mainline worth running because it attacks the true failure mode: small byte models do not cheaply discover benchmark-grade semantic coordinates from scratch.

### What Died

This lowering of standards died:

```text
If coordinate inheritance reaches 30-35% HellaSwag, the moonshot is back.
```

No. That may justify another research iteration. It does not satisfy the Vision.

### Final Answer To The Batch 11 Synthesis Question

Can coordinate-inheritance produce a result that survives both "that's obvious" and "that's trivial" at the same time?

Yes, but only this:

```text
A <= 121M active byteified inherited-coordinate model reaches at least 42.7%
HellaSwag, beats SmolLM2-135M on at least 3 of the 5 Vision benchmarks, fits and
runs on the RTX 5090 laptop, and shows through adapter-only, generic-pretrained,
same-family, shuffled, rotated, inverse-rotation, tokenized-sibling, and
byte-robustness controls that the lift comes from preserved teacher-specific
coordinate geometry rather than adapter training, generic pretraining, or token
compression.
```

That result is not obvious because the controls prove the coordinate system is load-bearing.

That result is not trivial because it beats the known small-model target rather than landing in the 30-35% research-signal zone.

Everything below that is either:

```text
real but not a moonshot
```

or:

```text
useful engineering but not the advertised geometry thesis
```

## Predictions For W-Loop B8 Stage 1

These predictions are deliberately concrete. They are not hopes.

### Batch 10 Stage 1 Gates

| Stage 1 Gate | Prediction | Expected Outcome |
|---|---|---|
| copied vs random NLL, token-end >= 2.0 nats/token | nominal pass | adapter calibration likely pushes copied advantage to about 2.2-2.8 nats/token |
| copied vs random NLL, patch-boundary >= 2.0 nats/token | borderline nominal pass | expected about 1.8-2.4 nats/token; first-byte slice remains weak |
| gap to true-embedding upper bound, token-end <= 1.5 nats or >= 60% closure | borderline fail | adapter may close 40-55% of the 3.58-nat gap, leaving about 1.6-2.1 nats |
| gap to true-embedding upper bound, patch-boundary <= 2.0 nats or >= 60% closure | fail | adapter may close 30-45% of the 4.10-nat gap, leaving about 2.3-2.9 nats |
| adapter size <= 2M params | pass | RMSNorm/affine/low-rank adapter can fit this |
| frozen-core gain >= 70% of post-finetune NLL gain | fail or borderline | full/LoRA adaptation likely adds too much; frozen core may retain only 45-65% |
| rotation sanity: no-inverse collapses, correct inverse recovers >= 80% | partial pass | hand-coded inverse should recover; learned adapters may recover too much from broken rotations, weakening the control |

### Additional Batch 11 Predictions

| New Question | Prediction |
|---|---|
| adapter + random core share of NLL lift | uncomfortably high, likely 40-60% after core-specific adapter training |
| adapter + generic pretrained core | recovers nontrivial lift, likely 40-70% of main Stage 1 NLL lift |
| NLL lift vs functional-margin lift | weak correlation; token-end NLL will look better than candidate margins |
| token-end vs patch-boundary | token-end will overstate readiness; patch-boundary and first-byte slices remain bottlenecks |
| true-embedding upper bound | still much better than codec+adapter, proving the channel remains lossy |
| full-depth frozen inherited core | sensitive to residual statistics; raw deep copied core unstable without careful gauge repair |
| first benchmark proxy | likely low-30s at best uncompressed if token-space scoring is used; not enough to infer 121M success |

The likely Stage 1 headline:

```text
Calibration makes copied Qwen layers look much more compatible with codec inputs,
but the adapter and generic-pretrained controls recover enough of the gain that
the result does not yet prove teacher-specific reasoning geometry.
```

## Revised Falsification Thresholds

### Distinguishing The Three Main Stories

Let:

```text
Main = calibrated codec + inherited Qwen core
AdapterRandom = calibrated codec + random Qwen-shaped core
GenericPretrained = calibrated codec + plausible non-Qwen pretrained core
FineTunedMain = Main after allowed core adaptation
Wide7 = byte-from-scratch baseline
```

Use both NLL and benchmark margins. NLL alone cannot classify the story.

| Story | Necessary Pattern |
|---|---|
| `geometry_transfer` | Main frozen or low-rank adapted gets >= 70% of final benchmark lift; AdapterRandom <= 25% of lift; GenericPretrained <= 50% of lift; disruptions lose >= 70%; inverse recovers >= 80%; functional-margin and layer-trajectory diagnostics pass |
| `good_initialization` | frozen Main gets < 50% of final lift; inherited full fine-tune beats random early; random/generic narrow gap by 2x budget; coordinate drift is large |
| `adapter_does_the_work` | AdapterRandom gets >= 50% of NLL lift or >= 40% benchmark lift; core-specific adapters erase copied-vs-random gap; adapter+LM-head has strong NLL without benchmark margins |
| `generic_pretraining` | GenericPretrained is within 2pp HellaSwag or 1.0 nat/token of Main; non-Qwen controls recover >= 70% of lift |
| `ordinary_distillation` | high-capacity adapter or broad core fine-tuning creates the result; copied geometry diagnostics fail; random/generic catch up with enough training |

### Hard Kill Conditions

Any of the following should block coordinate-inheritance promotion:

| Kill Condition | Consequence |
|---|---|
| adapter + random/generic core gets >= 70% of main benchmark lift | kill teacher-specific geometry claim |
| NLL Stage 1 passes but functional-margin preflight is <= +1pp over Wide7 | block Stage 2 promotion |
| patch-boundary margin lift < 50% of token-end margin lift | byte deployment not ready |
| frozen/low-rank core gets < 40% of final lift | classify as retraining, not inheritance |
| 121M model < 35% HellaSwag | kill as Sutra mainline result |
| 121M model 35-40% but tokenized sibling dominates and no byte robustness | demote byte-native story |
| 121M model < 42% and no clear path to champion-level benchmark wins | do not use moonshot language |

## Batch 11 Final Verdict

Coordinate-inheritance remains alive, but Batch 10's Stage 1 gate is too easy to pass for the wrong reasons.

The corrected stance:

```text
Stage 1 NLL is only a surface-compatibility preflight.
Stage 1 must add adapter attribution, functional-margin preflight, coordinate
drift accounting, hard pretrained controls, and codec channel diagnostics before
it can be interpreted as geometry evidence.
```

The project should proceed with W-Loop B8, but the adversarial expectation is:

```text
W-Loop B8 will likely find a real compatibility improvement and an ambiguous
causal story.
```

That ambiguity is not a failure of the loop. It is exactly what Batch 11 is meant to catch before the project mistakes "the adapter learned a useful token-space bridge" for "reasoning geometry transferred."

Coordinate inheritance is still the mainline. It is still not believed.
