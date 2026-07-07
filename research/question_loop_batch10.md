# QUESTION LOOP - Batch 10: Attack Coordinate-Inheritance Sutra

Date: 2026-07-07

Iterations: 64-70

## Grounding

I read the requested context in order:

1. `research/VISION.md`
2. `research/DEEP_RETHINK.md`
3. `research/question_loop_batch9.md`
4. `research/dual_loop_supervisor_checkin_5.md`
5. `research/work_loop_batch6.md`
6. `research/work_loop_batch5.md`
7. `code/chain_init_codec_probe.py`
8. `code/evidence_native_sutra.py`

Additional local checks:

- `research/STATUS.md` is partially stale relative to Batch 9: it still lists Evidence-Native as the live moonshot candidate, while Batch 9 demoted it after v0 failed gates.
- `research/question_loop_batch10.md` did not exist before this batch.

## Binding State Entering Batch 10

Batch 9 promoted coordinate-inheritance chain-init as the mainline and demoted evidence-native v1 to a kill-gated support branch.

The new mainline claim is:

```text
Reasoning capability is concentrated in transferable coordinate geometry.
If that geometry is preserved, a small byte-native runtime learns and judges far
more efficiently than a same-size random model. If the geometry is disrupted, the
advantage collapses.
```

The only direct empirical support is the Batch 5 compatibility probe:

| Readout | Random 4-layer NLL | Copied Qwen 4-layer NLL | Delta |
|---|---:|---:|---:|
| token-end codec input | 17.17 | 15.52 | -1.65 nats/token |
| patch-boundary codec input | 18.34 | 16.62 | -1.72 nats/token |

This is a positive sign, but the absolute losses are enormous. It is not byte perplexity, not HellaSwag, not a trained byte decoder, not a compressed 121M model, and not proof of reasoning. The probe uses oracle Qwen token boundaries for `token_end`, a Qwen-shaped truncated core, and the copied Qwen LM head.

The most honest starting point:

```text
Coordinate inheritance has earned one serious kill-gated prototype.
It has not earned belief.
```

## Iteration 64: CBD Attack - This May Be Compressed Distillation With A Harder Interface

### Current Strongest Position

After Batch 9, the defense is:

```text
Chain-init is not merely distillation. It preserves the teacher's reasoning
coordinate geometry, then wraps that geometry in a byte-native runtime. The
positive 1.7 nats/token signal shows copied coordinates carry usable structure
through the codec.
```

### Attack

An outside reviewer will call this CBD with extra pain.

CBD already showed the commercially relevant fact: a large teacher can be compressed through anchors into a 138M student with 42.65% HellaSwag. It did so with the advantages `DEEP_RETHINK.md` repeatedly identifies: same tokenizer, same family, smaller capacity gaps, inherited weights, and token-level distributions.

Coordinate inheritance currently says:

```text
Use a Qwen teacher.
Copy Qwen layers.
Train adapters around Qwen layers.
Compress Qwen layers.
Keep byte I/O.
```

That is not automatically a paradigm shift. It may be:

```text
CBD, but with a lossy byte adapter before the student and a harder decoder after it.
```

If a tokenized CBD student is simpler, better, and already published, nobody should care that the worse version happens to consume bytes. The byte interface adds novelty only if it buys something measurable: tokenizer independence, robustness to typos/OCR/Unicode, cross-teacher reuse across tokenizer families, lower active inference cost, or better updateability/evidence use because facts can be externalized while geometry remains compact.

### What Survived

The CBD attack does not kill coordinate inheritance. It clarifies the only defensible novelty:

```text
CBD transfers behavior inside one tokenized family.
Coordinate inheritance must prove that the transferable object is a coordinate
system that survives byte wrapping, compression, and disruption tests.
```

### What Died

This slogan died:

```text
Copied Qwen layers beating random layers means we found the moonshot.
```

No. It means copied layers are less terrible than random layers when fed codec states.

### Required Repair

Coordinate-inheritance v0 must include a CBD-equivalent token baseline:

| Baseline | Purpose |
|---|---|
| Tokenized compressed Qwen-family student, same active params | Tests whether bytes add anything over direct CBD/token compression |
| Byteified inherited-coordinate model | Main candidate |
| Same inherited core with token I/O retained | Separates byte adapter cost from core inheritance |
| Wide7 byte-from-scratch | Existing byte-native baseline |
| CBD literature number | External target, not a controlled local baseline |

Minimum survival condition:

```text
If byteified coordinate inheritance is worse than the tokenized compressed baseline
on clean benchmarks, it must beat that baseline on byte-native robustness and
cross-tokenizer tests by enough to justify the interface.
```

### Attack On The Next Defense

The next defense will say the byte interface is worth it because the codec maps bytes into Qwen-like embeddings. But the codec is lossy. The +1.7 nats/token signal may be the ceiling, not the floor.

## Iteration 65: Codec Bottleneck Attack - The Bridge May Destroy The Geometry

### Current Strongest Position

After Iteration 64, the defense becomes:

```text
This is not just CBD because the semantic codec lets raw bytes enter the inherited
coordinate system. The codec is the bridge from byte substrate to teacher geometry.
```

### Attack

The bridge is the weakest load-bearing part.

The codec facts are mixed:

- Phase 1 retrieval was strong above chance at token identity.
- Patch-boundary top-1 was only about 37.89%.
- Token-end behavior was much stronger than patch-boundary behavior.
- Offset slices showed first-byte positions are weak: 20.08% top-1 in Batch 5.
- Phase 1/Phase 2 supervision originally mismatched token ends and 4-byte patch boundaries.
- Batch 5 true teacher embeddings were far better than codec embeddings through the same copied layers.

The probe itself shows the bottleneck:

| Condition | Token-end NLL | Patch-boundary NLL |
|---|---:|---:|
| Copied 4 layers + codec input | 15.52 | 16.62 |
| Copied 4 layers + true teacher embeddings | 11.94 | 12.52 |
| Codec penalty | 3.58 | 4.10 |

That gap is bigger than the copied-vs-random win:

```text
token_end inheritance win:       1.65 nats
token_end codec penalty:         3.58 nats
patch_boundary inheritance win:  1.72 nats
patch_boundary codec penalty:    4.10 nats
```

The positive signal may therefore mean only:

```text
Copied Qwen layers still recognize a damaged version of their own embedding
space slightly better than random layers do.
```

That is not enough. The moonshot needs the byte adapter to preserve enough of the coordinate geometry for downstream judgment after compression.

### What Survived

The codec remains useful infrastructure. It gives a measurable byte-to-token bridge and a concrete bottleneck to optimize.

### What Died

This assumption died:

```text
Because the codec retrieves token identities, it preserves reasoning coordinates.
```

Token identity is not reasoning geometry. A lossy tokenizer mimic can recover token labels while destroying the continuous relations Qwen layers use.

### Required Repair

Before any benchmark run is interpreted as coordinate inheritance, Work Loop must run a codec-bridge calibration ladder:

| Gate | Required for v0 preflight |
|---|---:|
| Copied vs random NLL, token-end | copied better by >= 2.0 nats/token after calibration |
| Copied vs random NLL, patch-boundary | copied better by >= 2.0 nats/token after calibration |
| Codec-input gap to true-embedding upper bound, token-end | <= 1.5 nats/token or >= 60% of gap closed |
| Codec-input gap to true-embedding upper bound, patch-boundary | <= 2.0 nats/token or >= 60% of gap closed |
| First-byte slice | >= 35% top-1 or demonstrably bypassed by a non-first-byte readout design |
| Token-boundary oracle dependence | token_end and patch_boundary both reported; no promotion from token_end alone |
| Calibration adapter size | <= 2M trainable params for the first preflight |
| Calibration data | reported in bytes/tokens; no hidden large-scale KD budget |

If the adapter needs a high-capacity network or long KD training to make codec states usable, then coordinate inheritance is no longer a cheap coordinate transfer. It becomes ordinary distillation through a byte adapter.

### Attack On The Next Defense

The next defense will say copied layers beating random layers under calibrated codec input proves inherited coordinates matter. But random layers are an easy control. The hard question is whether the specific teacher coordinate system matters more than any decent pretrained initialization.

## Iteration 66: Control Attack - Random/Shuffled Controls Are Too Easy To Beat

### Current Strongest Position

After Iteration 65, the defense becomes:

```text
We can prove geometry by showing inherited Qwen layers beat random layers and
collapse under shuffled layers or random rotations.
```

### Attack

Those controls are necessary but too weak.

Random layers are not a serious alternative. Shuffled layers are not a serious alternative. Random residual rotations without matching inverse transforms are not a serious alternative. These controls answer only:

```text
Does a trained transformer work better than a broken transformer?
```

The real question is:

```text
Does this model benefit from the specific inherited teacher coordinate system,
or does it benefit from any pretrained transformer prior?
```

That demands nuanced controls.

### Required Controls

Coordinate-inheritance v0 must include these controls, or explicitly label itself incomplete:

| Control | What It Tests |
|---|---|
| Same Qwen architecture, random layers | Basic non-inheritance baseline |
| Same Qwen layers, shuffled layer order | Depth organization |
| Same Qwen layers, blocks independently permuted internally | Fine coordinate organization |
| Same Qwen layers, random residual rotation without inverse | Raw basis dependence |
| Same Qwen layers, random residual rotation with matching inverse | Gauge-invariant sanity check |
| Same architecture pretrained on random/permuted text | Generic pretraining dynamics without real language |
| Same architecture pretrained on narrow/different-domain text | Domain pretraining vs reasoning geometry |
| Same-family Qwen checkpoint not used for codec targets | Family-level geometry vs exact checkpoint match |
| Same-size non-Qwen token model with its own byte adapter | Any pretrained language geometry vs Qwen-specific geometry |
| Tokenized compressed Qwen baseline | Byte interface cost/benefit |
| Wide7 from scratch | Existing byte-native baseline |

The important controls are not the obviously broken ones. They are the plausible alternatives:

```text
Any pretrained language model may provide a useful initialization.
Any same-tokenizer family checkpoint may work.
A tokenized compressed baseline may dominate the byteified one.
```

### Exact Control Gates

Let:

```text
Lift(model) = model_score - Wide7_from_scratch_score
```

For HellaSwag and aggregate reasoning benchmarks:

| Gate | Required |
|---|---:|
| Main inherited model vs Wide7 | HellaSwag >= +8pp or HellaSwag >= 35%, whichever is stricter |
| Main inherited model vs random-layer control | retains >= 80% of total lift over Wide7 |
| Shuffled/depth-disrupted controls | lose >= 70% of inherited lift |
| Random-rotation without inverse | lose >= 70% of inherited lift |
| Random-rotation with correct inverse | recover >= 80% of inherited lift |
| Best generic-pretrained control | main inherited model beats it by >= 3pp HellaSwag and >= 2pp aggregate |
| Same-family different checkpoint | reported separately; if it matches main, claim becomes family coordinate inheritance, not specific teacher geometry |
| Non-Qwen pretrained control | main inherited model beats it, or the thesis generalizes to pretrained coordinate inheritance and must say so |

If the best generic-pretrained control matches the inherited model, the geometry claim is weaker:

```text
Pretraining helps small byte models.
```

That is useful, but it is not the advertised proof that Qwen reasoning coordinates were inherited.

### What Survived

Coordinate inheritance can survive this attack only by becoming more precise: exact checkpoint inheritance, family-level coordinate inheritance, generic pretrained language geometry, or just initialization quality. Each is a different claim. v0 must not blur them.

### What Died

This control story died:

```text
Random layers and shuffled layers are enough to prove geometry.
```

They are enough to prove the model is not random. They are not enough to prove the load-bearing object is teacher-specific coordinate geometry.

### Attack On The Next Defense

The next defense will say the nuanced controls can be passed, and then coordinate inheritance becomes real. But even if uncompressed inheritance works, the target is 121M. Compression may destroy the geometry first.

## Iteration 67: Compression Attack - The Geometry May Live In The Parameters You Delete

### Current Strongest Position

After Iteration 66, the defense becomes:

```text
With fair controls, inherited coordinates can show a real advantage. Then we
compress/prune/distill the inherited core to the 121M target.
```

### Attack

This is the hardest part and the most likely failure.

Qwen3-0.6B is roughly 600M parameters. The target is 121M. That is about 5x compression. CBD crosses the capacity gap with anchors. It does not take a 600M teacher, wrap it in a lossy byte interface, and crush it directly to 121M while preserving benchmark competence.

The coordinate-inheritance story says:

```text
Reasoning geometry is compressible.
```

But the current evidence says only:

```text
The first four copied Qwen layers are less bad than random first four layers
under codec inputs.
```

The valuable geometry may live in depth interactions across all 28 layers, attention head specialization, MLP key-value memories, layer norm statistics, residual scale, high-rank subspaces, or rare-feature circuits that benchmarks need but average NLL barely notices.

If pruning deletes exactly the rare circuits that produce HellaSwag/ARC gains, NLL can look acceptable while reasoning collapses. The project has already seen this pattern: BPB improved while HellaSwag stayed flat.

### Required Compression Ladder

No single 121M result is interpretable without a ladder:

| Stage | Purpose |
|---|---|
| Full byteified inherited core | Shows the byte adapter can preserve capability before compression |
| 300M dense or active | Tests moderate compression |
| 180M dense or active | Tests near-small compression |
| <= 121M dense | Target dense model |
| <= 121M active MoE, if used | Target active-compute model |
| Tokenized compressed sibling at each size | Tests byte cost |
| Coordinate-disrupted compressed controls | Tests whether compression preserved coordinates |

### Exact Compression Gates

Let:

```text
Lift_full = full_byteified_inherited - Wide7
Lift_121  = inherited_121M_or_121M_active - Wide7
```

Minimum survival:

| Gate | Required |
|---|---:|
| Full byteified inherited core | HellaSwag >= 35% and >= +8pp over Wide7 |
| 300M stage | retains >= 75% of `Lift_full` |
| 180M stage | retains >= 60% of `Lift_full` |
| <= 121M dense or active stage | retains >= 50% of `Lift_full` and HellaSwag >= 35% |
| <= 121M stage vs Wide7 | HellaSwag >= +8pp, PIQA/ARC aggregate >= +3pp |
| <= 121M stage vs best generic-pretrained control | HellaSwag >= +3pp |
| Coordinate-disrupted 121M controls | lose >= 70% of `Lift_121` |
| 8-bit quantization | loses <= 1pp HellaSwag |
| 4-bit quantization, if claimed | loses <= 3pp HellaSwag |
| Inference fit | runs on the RTX 5090 laptop with documented VRAM and tokens/sec |

Moonshot-public gate:

```text
At <= 121M active params, beat SmolLM2-135M on HellaSwag and at least 3 of the
5 Vision target benchmarks, with matched or clearly lower active inference cost.
```

If the full inherited model works but the 121M model does not, the result is not dead as engineering. But the 121M Sutra moonshot is not won.

### What Survived

Compression is allowed as the central experiment. CBD proves small models can store compressed inherited capability.

### What Died

This assumption died:

```text
If coordinates can be inherited, they can probably be compressed 5x.
```

That is the main theorem, not a detail.

### Attack On The Next Defense

The next defense will say that even if compression is hard, a byteified inherited model still changes assumptions about byte-native models. But if the coordinate system comes from a tokenized model and the clean tokenized baseline is better, byte-native may be branding.

## Iteration 68: Byte-Native Attack - The Model May Be Token-Shaped Internally

### Current Strongest Position

After Iteration 67, the defense becomes:

```text
The inherited model can be compressed, and the byte interface gives Sutra a
universal substrate while keeping the teacher's reasoning coordinates.
```

### Attack

The coordinate system comes from a tokenized model.

The codec maps bytes into Qwen embedding space. The inherited layers are Qwen layers. The LM head is Qwen-shaped. The probe evaluates Qwen token next-token loss.

So the obvious adversarial summary is:

```text
This is a token model with a byte tokenizer emulator in front of it.
```

That may be the right engineering path, but it weakens the byte-native claim.

The byte adapter learns to infer token boundaries and token identities. The inherited core reasons over token-shaped semantic units. The output head is token-space unless a byte decoder is added later. If the byte decoder merely converts token-shaped hidden states back into bytes, the system is byte I/O, not byte-native reasoning.

The Vision allows mechanisms to pivot, but it still claims byte-level substrate removes tokenizer lock-in and provides a universal interface. Coordinate inheritance must prove that byte I/O is not just a worse tokenizer.

### Required Byte-Native Advantage Tests

The byteified model must be compared against a tokenized compressed sibling, not only random byte baselines.

| Test | Required for byte-native claim |
|---|---:|
| Clean HellaSwag/PIQA/ARC gap to tokenized sibling | byte model within 2pp aggregate, or stronger elsewhere |
| 5% character typo/OCR noise | byte model beats tokenized sibling by >= 5pp aggregate or drops <= 50% as much |
| Unicode/mixed-script/code-mixed text | byte model beats tokenized sibling by >= 5pp aggregate |
| New-symbol/OOV adaptation | byte model adapts without vocabulary rebuild and beats tokenized sibling after equal update budget |
| Cross-tokenizer teacher adapter | at least two teacher families can feed the same byte substrate with lower adapter cost than training separate token students |
| Byte decoder quality | byte BPB no worse than Wide7 by >10% after adaptation |
| Streaming/partial-token scoring | byte model shows a real advantage before token boundary is complete |

If byteified inheritance wins only on clean token-style benchmarks, and tokenized compression wins there too, the byte-native story is weak.

### What Survived

The byte substrate remains strategically valuable if it supports robustness, cross-tokenizer teacher integration, raw-text deployment without tokenizer constraints, evidence/retrieval over arbitrary text, or future multi-teacher coordinate adapters.

### What Died

This claim died:

```text
Byte I/O automatically makes inherited Qwen coordinates byte-native.
```

No. It makes them byte-wrapped. Byte-native has to be earned by tests where byte access matters.

### Attack On The Next Defense

The next defense will say the byte wrapper is only the I/O layer; the real claim is coordinate geometry. But that has a familiar failure mode: Brainseed already found that teacher geometry is gauge-dependent. The codec may be imposing the wrong basis.

## Iteration 69: Gauge Attack - This May Reopen The Brainseed Failure

### Current Strongest Position

After Iteration 68, the defense becomes:

```text
The byte interface does not need to be the reasoning substrate. It only needs to
feed the inherited coordinate system. The reasoning geometry lives in the copied
core, not in the raw byte encoder.
```

### Attack

This is exactly where Brainseed warns us.

`DEEP_RETHINK.md` made the key distinction:

```text
Coordinate losses transfer coordinates.
Behavioral losses transfer behavior.
Invariant losses transfer knowledge.
```

Brainseed failed because hidden states are gauge-dependent. A representation can contain knowledge without exposing that knowledge in an architecture-independent coordinate system.

Coordinate inheritance tries to dodge this by copying the teacher layers directly. That helps only if the input to those layers is in the same gauge the layers expect.

But codec states are not Qwen token embeddings. They are learned byte-derived approximations passed through an alignment head. Even if the nearest token identity is often correct, the continuous basis can be wrong: norm distribution, anisotropy, covariance, token-frequency geometry, residual scale, boundary timing, rare-token relations, and multi-token phrase geometry can all be off.

The inherited layers expect Qwen's embedding manifold. The codec emits an approximation to that manifold. The 3.58-4.10 nats/token gap to true embeddings says the approximation is not currently good enough.

### Required Gauge Tests

Coordinate-inheritance v0 must distinguish four stories:

1. Exact gauge inheritance: codec emits Qwen-basis embeddings, copied Qwen layers work.
2. Learned gauge repair: a small adapter maps codec states into Qwen basis.
3. Generic semantic adapter: a larger adapter learns enough to drive any pretrained core.
4. Ordinary distillation: end-to-end training teaches the whole system behavior, not inherited coordinates.

Only 1 and 2 support the strong coordinate story. 3 is weaker. 4 is not the thesis.

Exact tests:

| Gate | Required |
|---|---:|
| Frozen inherited core + small adapter | clears NLL and benchmark preflight before full-core finetuning |
| Adapter size | <= 2M params for preflight; <= 5% of final params for v0 |
| Adapter training budget | separately reported; no hidden massive KD |
| Raw codec -> Qwen embedding covariance match | mean/norm/covariance diagnostics reported before and after adapter |
| Random orthogonal target codec | fails without inverse, recovers with inverse |
| Learned inverse from random rotation | must be small and data-efficient if claimed as gauge repair |
| High-capacity adapter ablation | if only high-capacity adapter works, classify as distillation, not coordinate inheritance |
| Frozen-core vs finetuned-core | frozen-core must show most of the gain; finetuning may polish but not create the result |
| Layerwise depth curve | copied layers should improve with depth in teacher order; shuffled depth should collapse |

### What Survived

Coordinate inheritance can survive gauge dependence if it treats gauge alignment as the main engineering target, not a solved detail.

The core claim becomes:

```text
The byte adapter learns a compact map into the teacher's coordinate gauge, and
once there, the inherited core supplies reasoning geometry cheaply.
```

That is still interesting.

### What Died

This version died:

```text
Any semantic byte codec can feed copied teacher layers.
```

No. The codec has to land in the right gauge, at the right positions, with the right residual statistics.

### Attack On The Final Defense

The final defense will say that all these gates are hard, but coordinate inheritance is still the strongest remaining path because every other path died. That may be true, but "last thing standing" is not the same as "strongest path."

## Iteration 70: Last-Thing-Standing Attack - Is This Actually Strongest?

### Current Strongest Position

After Iteration 69, the defense becomes:

```text
Coordinate inheritance has many risks, but it is the only path that directly
addresses the repeated failure: 121M byte models do not discover benchmark-grade
reasoning geometry from scratch. Therefore it is the strongest remaining path.
```

### Attack

This is partly correct and partly dangerous.

The dangerous version is:

```text
Everything else failed, so coordinate inheritance must be right.
```

That is not evidence. That is survivor bias.

Coordinate inheritance could fail for all seven reasons in this batch:

1. It may be inferior CBD.
2. The codec may destroy the coordinates.
3. The controls may reveal only generic pretraining, not teacher geometry.
4. Compression may delete the useful circuits.
5. The byte interface may add no advantage over tokenized compression.
6. Gauge mismatch may force ordinary distillation.
7. The final model may be useful engineering but not a paradigm shift.

The stronger version is:

```text
Coordinate inheritance is the strongest remaining path because it targets the
failure mode that all prior experiments converged on.
```

That is defensible.

The repeated failure history says:

| Direction | What Failed |
|---|---|
| E1/Option C byte KD | BPB improved, task judgment flat |
| Brainseed | codec charts/scorers did not become downstream judgment |
| Operational geometry toys | hidden-state geometry did not transfer cleanly |
| Energy probe on S0 | no hidden HellaSwag competence surfaced |
| Width/depth swap | byte modeling improved, reasoning did not |
| Evidence-native v0 | evidence training made evidence use worse |
| Chain-init probe | weak positive copied-vs-random NLL signal |

So coordinate inheritance is not strongest because it is beautiful. It is strongest because it is the only remaining hypothesis that attacks the central bottleneck:

```text
Small byte-native models can learn surface form cheaply, but not benchmark-grade
semantic/reasoning coordinates from scratch under the available compute.
```

If coordinates cannot be discovered cheaply, maybe they must be inherited.

### Final Verdict

Coordinate inheritance survives Batch 10, but only as a hostile, kill-gated mainline.

```text
PROVISIONALLY_KEEP_COORDINATE_INHERITANCE_AS_MAINLINE
DO_NOT_TREAT_CHAIN_INIT_AS_PROVEN_GEOMETRY
MAKE_CBD_AND_TOKENIZED_BASELINES_FIRST_CLASS_CONTROLS
MAKE_CODEC_GAUGE_ALIGNMENT_THE_PRIMARY_PREFLIGHT
MAKE_121M_COMPRESSION_THE_CENTRAL_THEOREM
MAKE_BYTE_NATIVE_ADVANTAGE_A_REQUIRED_STORY_GATE
```

The answer to the synthesis question:

```text
Coordinate inheritance is genuinely the strongest remaining path, but only
because it is the best-aligned response to the failure evidence. It is also the
last thing standing. The project must not confuse those facts.
```

It is strongest by negative evidence plus one weak positive signal. That is enough to run v0. It is not enough to believe v0 will work.

## Coordinate-Inheritance v0: Exact Gates

These gates define the first real prototype. They are intentionally harder than the Batch 5 probe and stricter than "copied beats random."

### Stage 0: Artifact Requirements

| Requirement | Gate |
|---|---|
| Reproducible artifacts | JSON metrics, model manifests, parameter counts, data manifests, seed list |
| Reported variants | main inherited, random, shuffled, rotated, generic pretrained, tokenized sibling, Wide7 |
| Seeds | >= 3 seeds for benchmark claims; 1 seed acceptable only for preflight NLL |
| Evaluation | HellaSwag, PIQA, ARC-Easy/Challenge, LAMBADA, WinoGrande if feasible |
| Statistics | paired bootstrap or McNemar-style paired tests for MCQ deltas |
| Compute accounting | adapter bytes/tokens, train steps, VRAM, tokens/sec |

### Stage 1: Codec/Gauge Preflight

| Gate | Required |
|---|---:|
| Copied vs random NLL, token-end | >= 2.0 nats/token copied advantage |
| Copied vs random NLL, patch-boundary | >= 2.0 nats/token copied advantage |
| Gap to true-embedding upper bound, token-end | <= 1.5 nats/token or >= 60% gap closure |
| Gap to true-embedding upper bound, patch-boundary | <= 2.0 nats/token or >= 60% gap closure |
| Adapter size | <= 2M params for preflight |
| Frozen-core gain | frozen inherited core achieves >= 70% of post-finetune NLL gain |
| Rotation sanity | no-inverse rotation collapses; correct inverse recovers >= 80% |

Failure here means:

```text
KILL_COORDINATE_INHERITANCE_V0_BEFORE_BENCHMARK_TRAINING
```

### Stage 2: Uncompressed Byteified Inheritance Gate

This stage can exceed 121M. It asks whether byte wrapping preserves inherited capability at all.

| Gate | Required |
|---|---:|
| HellaSwag | >= 35% and >= +8pp over Wide7 |
| PIQA/ARC aggregate | >= +3pp over Wide7 |
| Best generic-pretrained control | main model >= +3pp HellaSwag |
| Shuffled/rotated controls | lose >= 70% of inherited lift |
| Tokenized sibling | byteified model within 4pp aggregate or has clear byte robustness advantage |
| Byte BPB | no worse than Wide7 by >10% after adaptation |

Failure here means:

```text
DEMOTE_COORDINATE_INHERITANCE_TO_CODEC_DIAGNOSTIC
```

### Stage 3: 121M Compression Gate

This is the actual Sutra v0 gate.

| Gate | Required |
|---|---:|
| Params | <= 121M dense params or <= 121M active params with total params disclosed |
| HellaSwag | >= 35% and >= +8pp over Wide7 |
| PIQA/ARC aggregate | >= +3pp over Wide7 |
| Lift retention | >= 50% of full uncompressed inherited lift |
| Best generic-pretrained control | main model >= +3pp HellaSwag |
| Coordinate-disrupted 121M controls | lose >= 70% of 121M lift |
| 8-bit quantization | <= 1pp HellaSwag loss |
| RTX 5090 laptop inference | fits with documented VRAM and throughput |

Failure here means:

```text
COORDINATE_INHERITANCE_WORKS_ONLY_AS_LARGER_MODEL_OR_ENGINEERING_BASELINE
```

It may still be useful. It is not the 121M moonshot.

### Stage 4: Byte-Native Story Gate

| Gate | Required |
|---|---:|
| Clean benchmark gap to tokenized compressed sibling | <= 2pp aggregate or offset by robustness wins |
| Typo/OCR noise | byteified model >= +5pp over tokenized sibling or drops <= 50% as much |
| Unicode/code-mixed/OOV | byteified model >= +5pp over tokenized sibling |
| Cross-tokenizer teacher adapter | at least two teacher families tested, or claim remains Qwen-only |
| Streaming partial-token slice | byte model shows measurable advantage before token boundary completion |

Failure here means:

```text
BYTE_NATIVE_IS_INTERFACE_BRANDING_NOT_A_PARADIGM_ADVANTAGE
```

### Stage 5: Moonshot-Promotion Gate

This is not required for v0 survival, but it is required before public victory language.

| Gate | Required |
|---|---:|
| HellaSwag | beat SmolLM2-135M target from Vision |
| Vision benchmark table | beat SmolLM2 on at least 3 of 5 listed benchmarks |
| Active inference cost | <= SmolLM2-class active params and clearly laptop-runnable |
| Controls | random/generic/disrupted/tokenized controls reported |
| Claim discipline | public claim says "inherited coordinate geometry" only if controls support that wording |

## Predicted Work Loop Findings

My predictions for the Work Loop are deliberately concrete:

1. **Calibration will help a lot.** A small RMSNorm/affine/low-rank adapter will close a meaningful part of the codec-to-true-embedding gap. The copied-vs-random delta will likely clear 2 nats/token after calibration.

2. **Patch-boundary will remain worse than token-end.** Token-end results will look encouraging first. Patch-boundary and first-byte slices will be the bottleneck for real byte-native deployment.

3. **Full-depth inherited core will be unstable without gauge repair.** Feeding raw codec states into many copied Qwen layers will not automatically scale the 4-layer signal. Layer norms and residual statistics will amplify mismatch.

4. **Generic pretrained controls will narrow the story.** Random and shuffled controls will fail badly, but at least one pretrained control will recover a nontrivial fraction of the inherited model's gain. This will force the claim from "specific Qwen checkpoint geometry" toward either "Qwen-family coordinate geometry" or "pretrained language coordinates."

5. **Uncompressed byteified inheritance will probably beat Wide7.** If the adapter is decent, token-space or multiple-choice scoring should exceed the 26-27% Wide7/S0 region. A first uncompressed HellaSwag result in the low-to-mid 30s is plausible.

6. **35% HellaSwag is achievable before 121M compression; 42% is not likely in v0.** A 42% result at 121M active would be the home run. I do not expect it on the first prototype.

7. **Compression will be the likely failure point.** The full inherited model may work; the 121M dense model may lose too much. A 121M active MoE or 180-250M dense compromise is more likely to retain capability.

8. **The tokenized compressed sibling will beat the byteified model on clean text.** The byte model can still survive if it shows robustness or cross-tokenizer advantages. If those tests are skipped, the byte-native claim remains unproven.

9. **Evidence-native will look better on top of inherited coordinates than from scratch.** A small evidence/readout head may finally show positive evidence sensitivity once the core has usable semantic coordinates. That would support evidence-native as an application layer, not as the birth mechanism.

10. **The first public-safe claim will be narrower than the Vision.** The likely honest claim after v0 is not "121M beats SmolLM2." It is more likely:

```text
Inherited coordinates give a byte-wrapped small model a controlled, non-random
reasoning lift that survives some disruption controls and fails others.
```

That is a serious research signal, but not yet the stop-scrolling result.

## Final Synthesis

Coordinate inheritance is not allowed to inherit the romance of the original Sutra thesis for free.

The original romantic claim was:

```text
A small byte-native model can discover intelligence through better learning geometry.
```

The surviving claim is colder:

```text
A small byte-native runtime may need to inherit reasoning geometry discovered by
large-scale token models, then compress and expose it through bytes.
```

This is less pure, but it is more consistent with the evidence.

The project should proceed, but with this adversarial posture:

```text
Coordinate inheritance is guilty of being CBD-with-bytes until proven otherwise.
The codec is guilty of destroying geometry until calibrated otherwise.
The controls are guilty of being too easy until nuanced otherwise.
Compression is guilty of killing the signal until the 121M ladder proves otherwise.
Byte-native is guilty of branding until robustness/cross-tokenizer tests prove otherwise.
Gauge alignment is guilty of reopening Brainseed until frozen-core tests prove otherwise.
```

Only if it survives those charges does the home-run story become credible:

```text
Reasoning is a transferable coordinate system. It can be inherited, byte-wrapped,
compressed, disrupted, and measured. A laptop can run the compressed geometry
that once required a much larger training process.
```

Until then, the correct status is:

```text
MAINLINE_BUT_NOT_BELIEVED
```

