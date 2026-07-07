# WORK LOOP - Batch 3: Tier 3.0 Brainseed Chart Probe

Date: 2026-07-07

Artifacts:
- `code/tier3_brainseed_chart_probe.py`
- `research/work_loop_batch3.md`

Commands run:

```powershell
python -m py_compile code/tier3_brainseed_chart_probe.py
python code/tier3_brainseed_chart_probe.py --synthetic-smoke --synthetic-n 256
python code/tier3_brainseed_chart_probe.py --chart-only --no-artifacts --num-sequences 16 --batch-size 4 --seq-len 256 --json
python code/tier3_brainseed_chart_probe.py --chart-only --no-artifacts --num-sequences 64 --batch-size 4 --seq-len 256 --json
python -m py_compile code/tier3_brainseed_chart_probe.py
```

Compute policy honored:
- Chart-only mode only.
- CPU only for real codec forwards.
- No teacher candidate forward passes.
- No Sutra checkpoint loading.
- No training.
- No `C:/sutra_fast/brainseed_v0/` artifact was written because Gate A did not pass.

## Batch 3 Executive Verdict

Gate A does not pass. The trained Phase 1 codec is excellent at token-end anchors,
but it is not good enough at the 4-byte patch boundaries consumed by the Sutra
Phase 2 path.

The 64-sequence confirmation run gives:

| Anchor type | N | Real top-1 | Real top-5 | Real top-10 | Best control top-1 | Gap |
|---|---:|---:|---:|---:|---:|---:|
| Token end | 3992 | 86.57% | 95.57% | 97.82% | 4.31% | +82.26pp |
| Patch boundary | 4096 | 23.71% | 29.88% | 35.30% | 1.25% | +22.46pp |
| Rare patch boundary | 1893 | 18.28% | 30.96% | 40.52% | 0.11% | +18.17pp |

Precommitted Gate A says patch-boundary must reach top-1 >=30% or top-10 >=65%.
It reaches neither. The correct verdict is:

```text
ROUTE_PHASE_1_5_DENSE_PATCH_SUPERVISION
VOID_BRAINSEED_SCORER_RUN
VOID_GATE_B_CLAIM
```

Narrative gate: the bridge exists at the positions where it was trained, but the
bridge is broken at the positions Sutra actually consumes.

## Iteration 21: Design Chart Probe Spec

### Register
- Build `code/tier3_brainseed_chart_probe.py`.
- Required first mode: `--chart-only`, no teacher candidate forwards.
- Verdict token: `CONFIRM_TIER3_0_CHART_FALSIFIER` if the script can test token-end
  and patch-boundary chart quality with controls.

### Design-Gate
Passed. The script implements:
- CLI matching the R65 decision surface, plus local-shard chart audit controls.
- Codec checkpoint loading from `C:/sutra_fast/codec_phase1/codec_final.pt`.
- Teacher embedding loading from `C:/sutra_fast/teacher_embeddings.pt`.
- Qwen tokenizer loading with offline/local default.
- Local byte-shard sampling from `C:/sutra_fast/data/shards_diverse`.
- Token-end anchor extraction using the Phase 1 token decode boundary method.
- 4-byte patch-boundary anchor extraction by assigning each patch end to the
  teacher token span containing that byte position.
- In-batch token-id retrieval metrics: top-1, top-5, top-10.
- Frequency slices: rare and frequent.
- Controls: per-occurrence random target, fixed shuffled target, random codec,
  rotated chart, and frequency lookup.

### Implement
Created `code/tier3_brainseed_chart_probe.py`.

### Dry-Run
Synthetic fixture planned before real checkpoint loading.

### Smoke
Deferred to Iteration 22/23.

### Evidence-Gate
No positive real-model claim yet.

### Commit
`CONFIRM_TIER3_0_CHART_FALSIFIER_DESIGN`.

### Narrative Gate
Gossip headline: We are not asking if the baby knows facts; we are checking if
its eyes point at the right world.

## Iteration 22: Implement Chart-Only Mode

### Register
- Implement chart-only real codec path.
- Verdict token: `CONFIRM_CHART_ONLY_IMPLEMENTED` if it can run without teacher
  candidate forwards.

### Design-Gate
Passed. The chart-only path loads only:
- codec checkpoint,
- teacher embedding table,
- tokenizer,
- byte shards.

It does not load Qwen model weights.

### Implement
Implemented:
- `ByteShardSampler`
- `find_anchor_sets`
- `collect_chart_features`
- `topk_retrieval`
- `evaluate_anchor_set`
- `gate_a`

### Dry-Run
`python -m py_compile code/tier3_brainseed_chart_probe.py` passed.

### Smoke
Synthetic smoke run:

| Anchor | Real top-1 | Best control top-1 | Gate |
|---|---:|---:|---|
| Token end | 100.00% | 1.17% | PASS |
| Patch boundary | 100.00% | 1.17% | PASS |

### Repair
No repair needed.

### Evidence-Gate
Only validates the metric/control machinery on a synthetic fixture.

### Commit
`CONFIRM_CHART_ONLY_IMPLEMENTED`.

### Narrative Gate
Gossip headline: The lie detector can tell a real map from a fake map before we
try it on the real brain.

## Iteration 23: Implement Controls

### Register
- Add all five R65 controls.
- Verdict token: `CONFIRM_CONTROLS_IMPLEMENTED` if controls are present and fail
  in synthetic smoke.

### Design-Gate
Passed. Controls are:
- Per-occurrence random target: each anchor receives an independent random token
  embedding target.
- Fixed shuffled target: a fixed random permutation maps true token IDs to wrong
  teacher embeddings.
- Random codec: same codec architecture with random weights.
- Rotated chart: signed-permutation orthogonal rotation of the real projected
  teacher-space chart.
- Frequency lookup: query-independent ranking by sampled token frequency.

### Implement
Controls are integrated into `evaluate_anchor_set` and reported for total,
rare, and frequent slices.

### Dry-Run
Synthetic control behavior was near chance:
- Per-occurrence random target top-1: 0.00% token end, 0.39% patch.
- Fixed shuffled target top-1: 0.78%.
- Random codec top-1: 0.78%.
- Rotated chart top-1: 0.00-0.39%.
- Frequency lookup top-1: 1.17%.

### Smoke
Passed.

### Repair
No repair needed.

### Evidence-Gate
Controls are strong enough to catch fake retrieval in the synthetic fixture.

### Commit
`CONFIRM_CONTROLS_IMPLEMENTED`.

### Narrative Gate
Gossip headline: The decoy maps all look like paper maps in the dark; the test
turns the lights on.

## Iteration 24: Run Gate A Chart-Only

### Register
- Run real chart-only audit on local shard data.
- Verdict token: `PASS_GATE_A`, `ROUTE_PHASE_1_5_DENSE_PATCH_SUPERVISION`, or
  `KILL_PROCEED_TO_TRANSPLANT`.

### Design-Gate
Use CPU and no artifacts first:

```powershell
python code/tier3_brainseed_chart_probe.py --chart-only --no-artifacts --num-sequences 16 --batch-size 4 --seq-len 256 --json
```

Then confirm with larger sample:

```powershell
python code/tier3_brainseed_chart_probe.py --chart-only --no-artifacts --num-sequences 64 --batch-size 4 --seq-len 256 --json
```

### Implement
No new code during run.

### Dry-Run
`py_compile` had already passed.

### Smoke
Both real runs completed on CPU.

### Repair
No code repair. A separate tiny HellaSwag cache probe hung after attempting a
network fallback; it was terminated and not used for evidence.

### Evidence-Gate

16-sequence run:

| Anchor type | N | Real top-1 | Real top-5 | Real top-10 | Best control top-1 | Gap |
|---|---:|---:|---:|---:|---:|---:|
| Token end | 956 | 92.57% | 98.54% | 99.48% | 4.29% | +88.28pp |
| Patch boundary | 1024 | 29.39% | 41.02% | 49.80% | 1.17% | +28.22pp |
| Rare patch boundary | 623 | 28.41% | 46.55% | 56.66% | 0.64% | +27.77pp |

64-sequence confirmation:

| Anchor type | N | Real top-1 | Real top-5 | Real top-10 | Best control top-1 | Gap |
|---|---:|---:|---:|---:|---:|---:|
| Token end | 3992 | 86.57% | 95.57% | 97.82% | 4.31% | +82.26pp |
| Patch boundary | 4096 | 23.71% | 29.88% | 35.30% | 1.25% | +22.46pp |
| Rare patch boundary | 1893 | 18.28% | 30.96% | 40.52% | 0.11% | +18.17pp |

Controls in the 64-sequence patch-boundary run:

| Control | Top-1 | Top-5 | Top-10 |
|---|---:|---:|---:|
| Per-occurrence random target | 0.02% | 0.24% | 0.34% |
| Fixed shuffled target | 0.00% | 0.17% | 0.24% |
| Random codec | 0.15% | 0.29% | 0.46% |
| Rotated chart | 0.10% | 0.24% | 0.37% |
| Frequency lookup | 1.25% | 9.52% | 13.33% |

Precommitted patch-boundary gate:
- top-1 >=30% OR top-10 >=65%: FAIL.
- real-vs-best-control gap >=15pp: PASS.
- rare top-1 >=15% and gap >=8pp: PASS.

### Commit
`ROUTE_PHASE_1_5_DENSE_PATCH_SUPERVISION`.

### Narrative Gate
Gossip headline: The codec can name tokens at the token boundary, but Sutra eats
at patch boundaries and the signal thins out before dinner.

## Iteration 25: Analyze Gate A Results

### Register
- Decide whether to proceed to Gate B scorer.
- Verdict token: `VOID_BRAINSEED_SCORER_RUN` if Gate A fails.

### Design-Gate
R65 is binding: if Gate A fails at patch-boundary but passes token-end, document
the mismatch and propose Phase 1.5 dense patch-boundary supervision. Do not run
teacher-margin extraction or frozen scorer.

### Implement
No scorer run.

### Dry-Run
Not applicable.

### Smoke
Not applicable.

### Repair
Required repair is Phase 1.5:
- Fine-tune the codec with supervision at every 4-byte patch boundary.
- For each patch end position, supervise against the teacher token containing
  that byte position.
- Keep the same InfoNCE retrieval objective and the same controls.
- Rerun Gate A before any Brainseed scorer.

### Evidence-Gate
The token-end chart is real:
- 86.57% top-1 over 3992 token-end anchors.
- +82.26pp over best control.
- Rare token-end top-1: 75.02%.

The patch-boundary chart is not sufficient:
- 23.71% top-1, below 30%.
- 35.30% top-10, far below 65%.
- This is above controls, but not above the precommitted operational threshold.

### Commit
`VOID_GATE_B_CLAIM`.
`VOID_FULL_TIER3_TRAINING_RUN`.
`CONFIRM_PHASE_1_5_IS_NEXT`.

### Narrative Gate
Gossip headline: We found the bridge, but it does not reach the door the model
actually uses.

## Iterations 26-30: Frozen Brainseed Scorer

Not run.

Reason: Gate A did not pass. Running Qwen teacher margins, fitting a Brainseed
scorer, or writing `C:/sutra_fast/brainseed_v0/` artifacts would violate the R65
gate chain.

Verdict token:

```text
VOID_BRAINSEED_V0_SCORER_UNTIL_PATCH_GATE_PASSES
```

## Final Batch 3 Synthesis

The Tier 3.0 probe is built and smoke-tested. The first real chart audit is
informative and negative in the exact way R65 anticipated.

The trained Phase 1 codec is not fake: token-end retrieval is far beyond chance
and all controls fail. But the Phase 1/Phase 2 boundary mismatch is real. Patch
boundaries carry some token signal, but not enough to satisfy the precommitted
chart gate. This blocks Brainseed v0.

Honest conclusion:

```text
The real codec is a strong token-end chart, not yet a sufficient Sutra patch-boundary chart.
```

Next action:

```text
Phase 1.5 dense patch-boundary supervision, then rerun Gate A.
```

No positive Brainseed or real Sutra claim is supported yet.
