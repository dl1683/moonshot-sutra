# WORK LOOP - Batch 2: Build Gauntlet + First Transplant

Date: 2026-07-07

Artifacts:
- `code/toy_weight_transplant_gauntlet.py`
- `research/work_loop_batch2.md`

Commands run:

```powershell
python code/toy_weight_transplant_gauntlet.py --tier tier1
python code/toy_weight_transplant_gauntlet.py --tier all
python code/toy_weight_transplant_gauntlet.py --tier all --json
python -m py_compile code/toy_weight_transplant_gauntlet.py
```

Compute policy honored: CPU-only, no GPU training, no large checkpoint loading.

## Batch 2 Executive Verdict

Tier 1 passed the precommitted linear known-gauge gauntlet. Raw per-layer SVD is coordinate-dependent: an orthogonal gauge leaves it unchanged, but a non-orthogonal hidden gauge changes the transplant by 0.7687 relative Frobenius drift and raises held-out MSE from 0.0409 to 4.1939. Exact function transplant and chart/Procrustes transplant remain at numerical zero error under the same gauge.

Tier 2 and Tier 2.5 were implemented as lightweight analytic binding smoke tests, not as trained transformer claims. They validate the structure of the next gauntlet layer: Procrustes/operator, Jacobian sketch, and MLP-slot methods solve the controlled binding task; raw SVD, shuffled pairs, wrong circuit, frequency-matched, random chart, random byte codec, and shuffled byte codec controls fail.

Main headline: raw weight copying dies as a transplant story; coordinate charts are the minimum ticket to play.

## Iteration 11: Gauntlet Specification

### Register
- Object: `code/toy_weight_transplant_gauntlet.py` as one clean CPU-only gauntlet file.
- Tier 1 target: two-layer linear teacher `F = W2 @ W1` with hidden gauge freedom.
- Verdict token: `CONFIRM_GAUNTLET_SPEC` if the test has exact-positive and fake-negative cases; `KILL_GAUNTLET_SPEC` if controls can pass by accident.

### Design-Gate
Passed. Tier 1 uses input dim 24, hidden dim 48, output dim 16, transplant rank 8. The canonical teacher includes active hidden directions plus decoy directions killed by `W2`. A non-orthogonal gauge scales active directions down and decoys up while preserving `W2 @ W1` exactly.

Controls:
- Orthogonal gauge: raw SVD should remain stable.
- Non-orthogonal gauge: raw SVD should change.
- Exact composite-function transplant: must work in all gauges.
- Chart/Procrustes transplant: paired activations define the hidden coordinate chart.
- Matched-spectrum random function: must fail.

### Implement
Implemented in `Tier1Config`, `LinearTeacher`, `make_linear_teacher`, `gauge_teacher`, `raw_w1_svd_transplant`, `exact_function_transplant`, and `chart_procrustes_transplant`.

### Dry-Run
The dimensions are tiny; all operations are NumPy SVD/pseudoinverse. No training.

### Smoke
`python code/toy_weight_transplant_gauntlet.py --tier tier1` ran successfully.

### Repair
No Tier 1 repair needed.

### Evidence-Gate
Positive claims deferred until Iteration 13 run numbers.

### Commit
`CONFIRM_GAUNTLET_SPEC`.

### Narrative Gate
Gossip headline: The brain-copy machine now has a lie detector for fake coordinate copying.

## Iteration 12: Tier 1 Implementation

### Register
- Build the exact linear known-gauge case.
- Verdict token: `CONFIRM_TIER1_IMPLEMENTED` if code exposes raw-gauge failure and chart/exact success; `KILL_TIER1_IMPLEMENTED` if raw SVD cannot be made to fail without cheating.

### Design-Gate
Passed. The teacher function rank is exactly 8. Raw SVD keeps top `W1` hidden singular coordinates. The non-orthogonal gauge makes decoy directions dominate `W1` singular structure without changing the function.

### Implement
Implemented:
- `raw_w1_svd_transplant`: coordinate-dependent per-layer SVD.
- `exact_function_transplant`: SVD of the gauge-invariant composite operator.
- `chart_procrustes_transplant`: paired hidden activations estimate a chart from rotated hidden coordinates back to canonical coordinates, then solve the output operator in that chart.

### Dry-Run
Expected behavior:
- Raw base: not exact but usable.
- Raw orthogonal gauge: same as raw base.
- Raw non-orthogonal gauge: large drift and larger MSE.
- Exact/chart: near floating-point zero.

### Smoke
Tier 1 script completed with all gates passing.

### Repair
None.

### Evidence-Gate
Held for Iteration 13 exact numbers.

### Commit
`CONFIRM_TIER1_IMPLEMENTED`.

### Narrative Gate
Gossip headline: Singular vectors looked smart until someone changed the room lighting.

## Iteration 13: Tier 1 Run and Analysis

### Register
- Run exact Tier 1 gauntlet and report numbers.
- Verdict token: `CONFIRM_TIER1_GAUGE` if exact/chart pass and raw non-orthogonal fails; `KILL_TIER1_GAUGE` otherwise.

### Design-Gate
Precommitted gates:
- Raw non-orthogonal drift >= 0.50.
- Raw orthogonal drift < 1e-10.
- Exact transplant MSE <= 1e-20.
- Exact drift <= 1e-10.
- Chart transplant MSE <= 1e-20.
- Matched-spectrum random cosine < 0.50.

### Implement
No code changes during analysis.

### Dry-Run
Command:

```powershell
python code/toy_weight_transplant_gauntlet.py --tier tier1
```

### Smoke
Pass.

### Repair
No repair.

### Evidence-Gate
Exact numbers:

| Metric | Value |
|---|---:|
| true rank | 8 |
| raw base MSE | 0.0408808130 |
| raw orthogonal MSE | 0.0408808130 |
| raw non-orthogonal MSE | 4.1939318412 |
| raw orthogonal drift | 1.7196371097e-15 |
| raw non-orthogonal drift | 0.7687197232 |
| raw non-orthogonal cosine to true | 0.6007979429 |
| exact base MSE | 4.0549433535e-29 |
| exact non-orthogonal MSE | 4.1799116041e-29 |
| exact non-orthogonal drift | 1.6729763799e-15 |
| chart non-orthogonal MSE | 1.0803931357e-27 |
| chart non-orthogonal drift | 1.2922608109e-14 |
| random spectrum MSE | 12.6629096704 |
| random spectrum cosine | 0.0233104050 |

All Tier 1 gates passed.

### Commit
`CONFIRM_TIER1_GAUGE` and `KILL_RAW_SVD_AS_GEOMETRY`.

### Narrative Gate
Gossip headline: We caught SVD copying the coordinate system instead of the computation.

## Iteration 14: Tier 2 Nonlinear Binding Design

### Register
- Design a CPU-only nonlinear binding gauntlet.
- Verdict token: `CONFIRM_TIER2_DESIGN` if it tests actual transplant mechanisms and controls; `VOID_TIER2_DESIGN` if it becomes another training-loss toy.

### Design-Gate
Passed with a constrained scope. Tier 2 is analytic, not trained. It uses a 4-stage binding operator corresponding to encode facts, bind key/value slots, query, and score. Config records teacher width 128/layers 4 and student width 64/layers 4, while the executable geometry uses teacher dim 64 and student dim 32 for fast CPU linear algebra.

Teacher task:
- Context has two entities.
- Each entity has color, room, and action values.
- Query asks for one entity/attribute.
- Candidate set has 4 same-attribute choices.

Methods:
- Procrustes operator transplant.
- Jacobian sketch of fact-local slot operators.
- MLP-slot transplant.
- Raw SVD without chart.

Controls:
- Shuffled key/value pairs.
- Wrong-circuit query attribute.
- Frequency-matched random values.
- Random chart.

### Implement
Designed in `BindingGeometry`, `BindingExample`, and `Tier2Config`.

### Dry-Run
All methods score MCQ choices by `q^T M v`, where `M` is the transplanted context memory.

### Smoke
Deferred to Iteration 16.

### Repair
None.

### Evidence-Gate
No positive claim yet.

### Commit
`CONFIRM_TIER2_DESIGN`.

### Narrative Gate
Gossip headline: The next test asks whether we can steal the lookup trick, not just the weights.

## Iteration 15: Tier 2 Method Implementation

### Register
- Implement Tier 2 transplant methods and controls.
- Verdict token: `CONFIRM_TIER2_IMPLEMENTED` if every method and control is executable from the same harness.

### Design-Gate
Passed. The binding memory is nonlinear in context because facts construct a context-specific matrix. The transplanted object is not a dataset; it is an operator/slot system.

### Implement
Implemented:
- `procrustes_operator_memory`: maps teacher memory through estimated teacher-to-student charts.
- `jacobian_slot_memory`: sums fact-local derivative operators, mapped into student geometry.
- `student_slot_memory`: maps MLP-like key/value slots into student space.
- `raw_svd_memory`: truncates teacher memory and naively resizes without a chart.
- Controls for shuffled values, wrong circuit, frequency-matched random values, and random chart.

### Dry-Run
Expected pass/fail:
- Procrustes/Jacobian/slots near 100%.
- Raw/chartless and fake controls near 25% chance, with shuffled allowed up to 45% because examples can accidentally preserve some bindings.

### Smoke
`python code/toy_weight_transplant_gauntlet.py --tier all` reached Tier 2 successfully.

### Repair
No Tier 2 repair needed.

### Evidence-Gate
Held for Iteration 16 numbers.

### Commit
`CONFIRM_TIER2_IMPLEMENTED`.

### Narrative Gate
Gossip headline: The fake transplants get the same vocabulary, but not the trick.

## Iteration 16: Tier 2 Run and Analysis

### Register
- Run Tier 2 analytic binding gauntlet.
- Verdict token: `CONFIRM_TIER2_SMOKE` if real methods pass and controls fail; `VOID_TIER2_REAL_CLAIM` if this is mistaken for trained-transformer evidence.

### Design-Gate
Precommitted gates:
- Teacher >= 99%.
- Procrustes/Jacobian/MLP slots >= 95%.
- Raw SVD and controls <= 45%.

### Implement
No new implementation during run.

### Dry-Run
Command included in full run:

```powershell
python code/toy_weight_transplant_gauntlet.py --tier all
```

### Smoke
Pass.

### Repair
None.

### Evidence-Gate
Exact Tier 2 MCQ accuracy over 800 examples:

| Method | Accuracy |
|---|---:|
| teacher | 1.00000 |
| procrustes_operator | 1.00000 |
| jacobian_sketch | 1.00000 |
| mlp_slots | 1.00000 |
| raw_svd_no_chart | 0.27875 |
| shuffled_pairs_control | 0.41500 |
| wrong_circuit_control | 0.28500 |
| frequency_matched_control | 0.24750 |
| random_chart_control | 0.23750 |

All Tier 2 smoke gates passed.

### Commit
`CONFIRM_TIER2_SMOKE` and `VOID_TIER2_REAL_MODEL_CLAIM`.

### Narrative Gate
Gossip headline: The slot map works; the slot-shaped decoys do not.

## Iteration 17: Tier 2.5 Byte-Patch Student Design

### Register
- Bridge from token-style binding geometry to byte-patch inputs.
- Verdict token: `CONFIRM_TIER25_DESIGN` if the codec acts as a chart; `KILL_CODEC_AS_CHART` if byte mapping cannot separate correct from shuffled/random controls.

### Design-Gate
Passed with a key constraint: the byte codec must not collapse query keys. Values remain 4-byte words. Query keys are `entity:attr` phrases, representing composition over multiple byte patches. This mirrors the real Phase 1/Phase 2 mismatch: patch states are only useful if the codec supervises the positions where the reasoner consumes them.

### Implement
Designed `BytePatchCodec` with three modes:
- real codec chart: byte phrases map to student chart vectors with small noise.
- random codec: byte phrases map to random vectors.
- shuffled codec: correct vectors are permuted across byte phrases.

### Dry-Run
Expected behavior:
- Real byte codec chart should preserve Tier 2 slot performance.
- Random/shuffled/wrong-circuit controls should fail.

### Smoke
Initial run failed, as described in Iteration 18 repair.

### Repair
Deferred.

### Evidence-Gate
No positive claim yet.

### Commit
`CONFIRM_TIER25_DESIGN_WITH_REPAIR_REQUIRED`.

### Narrative Gate
Gossip headline: The byte bridge only works if it points to the right word, at the right patch.

## Iteration 18: Implement and Run Tier 2.5

### Register
- Implement and run byte-patch codec variant.
- Verdict token: `CONFIRM_TIER25_CODEC_CHART` if real codec passes and random/shuffled/wrong controls fail.

### Design-Gate
Passed after repair. The first implementation truncated `entity:attr` to 4 bytes, causing all attributes for one entity to collide. That produced only 48.0% for the real codec and exposed the same supervision-location risk described in `DEEP_RETHINK.md`.

### Implement
Repair:
- Added `key_to_bytes(word)` for full query-key byte phrases.
- Kept `word_to_bytes(value)` for 4-byte value patches.
- Updated `BytePatchCodec` key lookup to use full key bytes.

### Dry-Run
Reran Tier 2.5 alone:

```powershell
python code/toy_weight_transplant_gauntlet.py --tier tier25
```

### Smoke
Pass.

### Repair
Completed.

### Evidence-Gate
Exact Tier 2.5 MCQ accuracy over 800 examples:

| Method | Accuracy |
|---|---:|
| byte_codec_chart | 1.00000 |
| random_byte_codec_control | 0.24875 |
| shuffled_byte_codec_control | 0.21000 |
| wrong_circuit_with_codec_control | 0.16875 |

All Tier 2.5 gates passed after the collision repair.

### Commit
`CONFIRM_TIER25_CODEC_CHART`.

### Narrative Gate
Gossip headline: The codec is not decoration; it is the passport between architectures.

## Iteration 19: All-Results Analysis and Adversarial Review

### Register
- Decide what actually passed and what is still unproven.
- Verdict token: `CONFIRM_BATCH2_GAUNTLET` if the results survive hostile review; `VOID_BATCH2_MOONSHOT_CLAIM` for any overclaim.

### Design-Gate
Pass condition: exact Tier 1 plus controlled Tier 2/2.5 separation. No real-checkpoint claim allowed.

### Implement
No new implementation.

### Dry-Run
Full run repeated successfully, and `python -m py_compile code/toy_weight_transplant_gauntlet.py` passed.

### Smoke
Pass.

### Repair
Only Tier 2.5 key-collision repair was needed.

### Evidence-Gate
What passed:
- Tier 1 proves raw per-layer SVD is not gauge-invariant geometry.
- Tier 1 proves composite-function transplant is invariant in the exact linear case.
- Tier 1 proves paired activation charts can undo a known non-orthogonal gauge in the linear case.
- Tier 2 proves the harness can distinguish real slot/operator transplant from fake controls on an analytic nonlinear binding operator.
- Tier 2.5 proves a byte-patch codec chart can bridge the analytic token/byte gap when the chart supervises the consumed positions.

What did not pass because it was not tested:
- No real Sutra checkpoint.
- No trained 4-layer transformer teacher/student transplant.
- No HellaSwag/PIQA/ARC lift.
- No claim that Phase 1 codec semantics are sufficient; current repo evidence still says token-identity retrieval, not semantic addressability.
- No claim of zero-gradient intelligence transfer. The better target is speed-of-learning from a geometry-initialized model.

Hostile reviewer verdict:
- Tier 1 is hard evidence.
- Tier 2/2.5 are useful gauntlet scaffolds, but too analytic to count as real nonlinear transplant proof.
- The most important new operational lesson is the codec-as-chart framing: the existing Phase 1 codec is not just an encoder; it is the only available cross-architecture coordinate map.

### Commit
`CONFIRM_BATCH2_GAUNTLET` plus `VOID_REAL_SUTRA_CLAIM`.

### Narrative Gate
Gossip headline: We found the lockpick for fake intelligence transfer, not yet the vault.

## Iteration 20: Tier 3 Roadmap for Real Sutra Checkpoint

### Register
- Specify how to proceed from gauntlet to real models without wasting GPU.
- Verdict token: `CONFIRM_TIER3_ROADMAP` if compute, gates, and controls are concrete.

### Design-Gate
Tier 3 should not start with a full training run. It should start with frozen caches and small CPU/GPU-light linear algebra.

### Implement
Roadmap only; no real checkpoint loaded in this batch.

### Dry-Run
Proposed sequence:

1. Freeze the codec-as-chart surface.
   - Use existing Phase 1 codec outputs at token ends and at 4-byte patch boundaries.
   - Measure rare-token and token-frequency slices.
   - Add the per-occurrence random target control if not already available.

2. Build a real activation chart cache.
   - Teacher: Qwen3-0.6B or the current local teacher used by codec Phase 1.
   - Student: Sutra Wide7 or S0 frozen checkpoint.
   - Anchors: clean token-end/patch-end alignments only, plus dense 4-byte boundary diagnostic.
   - Store teacher embeddings/selected hiddens, codec states, Sutra patch states, and candidate margins.

3. Track A: chart quality before transplant.
   - Procrustes/CCA from codec/Sutra states to teacher token embedding space.
   - Controls: shuffled pairs, random chart, fixed permutation, per-occurrence random target, token-frequency baseline.
   - Gate: real chart retrieval/margin prediction beats all controls by >=10pp on held-out anchors and holds on rare-token slices.

4. Track B: frozen operator transplant.
   - Start with small modules: tied-embedding KD head, low-rank candidate energy, or layer-local Jacobian operator.
   - No end-to-end student training as evidence.
   - Evaluate immediate candidate-margin movement on held-out HellaSwag/PIQA slices.
   - Gate: >=2pp over frozen baseline and >=2pp over shuffled/random transplant controls.

5. Track C: born-knowing speed-of-learning test.
   - Initialize from the best chart/operator transplant.
   - Train only a small continuation run, CPU/GPU budget predeclared.
   - Main metric: steps to threshold, not gradient purity.
   - Gate: transplanted init reaches 35% HellaSwag in <=5K steps where matched random/chart-shuffled init is still <=30%, or reaches the same accuracy with >=10x fewer steps.

### Smoke
Not run in Batch 2 by design.

### Repair
The first repair likely needed is dense codec supervision at the patch positions the reasoner consumes, because Tier 2.5 reproduced the collision/supervision-location failure mode.

### Evidence-Gate
Precommitted Tier 3 gates:
- Chart gate: true codec/Sutra chart beats shuffled/random/frequency controls by >=10pp on held-out anchors.
- Frozen transplant gate: >=2pp held-out MCQ lift over baseline and >=2pp over fake transplants.
- Speed gate: 10x fewer steps to a meaningful threshold, or no claim.
- Kill: any gain matched by shuffled chart, random chart, frequency baseline, or extra-data control.

Compute estimate:
- Cache extraction: minutes to low hours depending on checkpoint availability; no training.
- Linear algebra: CPU feasible for 10k-50k anchors if batched.
- Frozen probes: CPU or small GPU, but no full model training required.
- Speed test: only after chart/frozen gates pass; predeclare <=5K steps.

### Commit
`CONFIRM_TIER3_ROADMAP`.

### Narrative Gate
Gossip headline: The moonshot is no longer zero training; it is starting life with a map that random models spend 100K steps trying to invent.

## Final Batch 2 Synthesis

The gauntlet now exists and runs. Tier 1 is the important result: raw SVD is killed as geometry, while exact composite-function transplant and chart-based transplant survive the gauge attack. Tier 2 and 2.5 are useful scaffolds: they show the harness can separate true operator/slot transfer from fake controls, and they sharpen the semantic codec into a concrete coordinate-chart role.

The next honest breakthrough target is not `5pp HellaSwag from transplant alone`. It is a speed-of-learning result: a Sutra initialization or frozen module built from teacher geometry reaches a meaningful accuracy threshold with an order-of-magnitude fewer steps than a matched random or shuffled-chart initialization.

