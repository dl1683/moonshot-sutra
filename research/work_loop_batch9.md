# WORK LOOP - Batch 9: Coordinate-Inheritance v1 Repair Smoke

Date: 2026-07-07

Iterations: 81-90 requested; executed through v1 repair and 128-sequence smoke. Full 1000-sequence Stage 1 was not run because the smoke failed the precommitted disruption gate.

## Artifacts

- `code/coordinate_inheritance.py`
- `tmp_coordinate_inheritance_v1/dry/preflight_metrics.json`
- `tmp_coordinate_inheritance_v1/smoke128/preflight_metrics.json`
- `tmp_coordinate_inheritance_v1/smoke128_repair/preflight_metrics.json`
- `tmp_coordinate_inheritance_v1/smoke128_repair/calibration_adapter.pt`

## Commands Run

```powershell
python -m py_compile code/coordinate_inheritance.py
python code/coordinate_inheritance.py --mode preflight --codec-checkpoint C:/sutra_fast/codec_phase1.5/codec_final.pt --output-dir tmp_coordinate_inheritance_v1/dry --device cuda --layers 2 --num-sequences 4 --seq-len 128 --batch-size 1 --max-positions-per-sequence 32 --adapter-steps 2 --adapter-batch-anchors 64 --finetune-core-steps 1 --finetune-batch-sequences 1 --depth-curve-layers 2 --bootstrap-samples 5 --progress --json
python code/coordinate_inheritance.py --mode preflight --codec-checkpoint C:/sutra_fast/codec_phase1.5/codec_final.pt --output-dir tmp_coordinate_inheritance_v1/smoke128 --device cuda --layers 4 --num-sequences 128 --seq-len 512 --batch-size 4 --max-positions-per-sequence 128 --adapter-steps 300 --adapter-batch-anchors 2048 --finetune-core-steps 5 --finetune-batch-sequences 1 --finetune-lr 1e-5 --depth-curve-layers 2 4 6 8 --bootstrap-samples 200 --progress
python code/coordinate_inheritance.py --mode preflight --codec-checkpoint C:/sutra_fast/codec_phase1.5/codec_final.pt --output-dir tmp_coordinate_inheritance_v1/smoke128_repair --device cuda --layers 4 --num-sequences 128 --seq-len 512 --batch-size 4 --max-positions-per-sequence 128 --adapter-steps 300 --adapter-batch-anchors 2048 --finetune-core-steps 5 --finetune-batch-sequences 1 --finetune-lr 1e-5 --depth-curve-layers 2 4 6 8 --bootstrap-samples 200 --progress
```

## Precommitted Verdict Tokens

```text
PASS_STAGE1_V1_PREFLIGHT__PROCEED_TO_STAGE2
FAIL_STAGE1_V1_PREFLIGHT__DO_NOT_RUN_STAGE2
PASS_STAGE2_UNCOMPRESSED_BYTEIFIED_INHERITANCE
FAIL_STAGE2__DEMOTE_COORDINATE_INHERITANCE_TO_CODEC_DIAGNOSTIC
```

Smoke verdict used here:

```text
FAIL_STAGE1_V1_PREFLIGHT__DO_NOT_RUN_STAGE2
DO_NOT_RUN_1000_SEQUENCE_FULL_PREFLIGHT_FROM_FAILED_SMOKE
DO_NOT_RUN_STAGE2_BENCHMARKS
```

## Implementation Changes

`code/coordinate_inheritance.py` was modified in place only. No new Python files were created.

1. Readout-conditioned adapter: `--adapter-conditioning readout` trains separate RMSNorm+Linear maps for token-end and patch-boundary readouts. Parameter count is 526,848, under the 2M cap.
2. Generic pretrained control: `generic_pretrained_calibrated` copies Qwen layers 14-17 into the 4-layer student shell. This tests a different pretrained layer range rather than the teacher's early-layer geometry.
3. Stronger disruptions: hidden-dimension permutation, zeroed 50% dimensions, and same-norm Gaussian replacement are all reported. After the first smoke showed permutation was too weak, same-norm Gaussian replacement became the primary stronger-disruption gate before any full run.
4. Layer depth curve: 2/4/6/8 inherited layers are evaluated on the same trained adapter and data split, with frozen-core retention reported per readout.
5. Benchmark mode now supports the readout-conditioned adapter, generic pretrained layer control, and dimension-permuted control. It was not run because Stage 1 smoke failed.

## Iteration 81 Diagnosis

### Failure 1: Patch-boundary frozen-core gain

The v0 failure was consistent with a shared adapter being slightly suboptimal at patch positions. v1's readout-conditioned adapter repaired this on the 128-sequence smoke:

| Readout | v1 frozen-core gain | Gate |
|---|---:|---|
| token-end | 82.6% | PASS |
| patch-boundary | 76.3% | PASS |

Alternative interpretation: this may be a smoke-set effect rather than a guaranteed full-run repair. But the direction is exactly what the repair predicted: patch-boundary improves after separating readout parameters.

### Failure 2: Rotation/disruption retention

The rotation failure was not just that orthogonal rotation was weak. Stronger input destruction also retained large lift:

| Readout | legacy no-inverse rotation retained | dim permutation retained | zeroed 50% retained | same-norm Gaussian retained | Required primary |
|---|---:|---:|---:|---:|---:|
| token-end | 36.3% | 40.4% | 90.4% | 33.5% | <=20% |
| patch-boundary | 47.6% | 50.7% | 79.0% | 47.3% | <=20% |

Alternative interpretation: copied Qwen layers plus the Qwen head carry a pretrained language-prior floor that beats random Qwen-shaped layers even when the byte-derived input signal is destroyed. That means copied-vs-random NLL overstates coordinate-specific signal unless a destroyed-input copied-core prior is subtracted or controlled.

## Is This The Right Experiment?

Yes for v1 smoke. It directly tested the two B8 failure modes under the requested repairs: readout-conditioned calibration and stronger disruption controls, while adding the generic layer-range control and depth curve.

No for promotion. The smoke failed the stronger-disruption criterion, so the full 1000-sequence preflight would be a more expensive confirmation of a known failed gate rather than a promotion-quality experiment.

## 128-Sequence Smoke Gate Table

Run: `tmp_coordinate_inheritance_v1/smoke128_repair/preflight_metrics.json`

| Gate | Required | Token-end | Patch-boundary | Overall |
|---|---:|---:|---:|---|
| Adapter params | <=2M | 526,848 | 526,848 | PASS |
| Copied vs random advantage | >=2.0 nats | 5.750 | 5.360 | PASS |
| Gap to true / closure | <=1.5 or >=60%; <=2.0 or >=60% | 0.279 / 95.4% | 1.323 / 80.2% | PASS |
| Frozen-core gain | >=70% | 82.6% | 76.3% | PASS |
| Generic pretrained gap | copied better by >=0.75 nats | 4.081 | 3.859 | PASS |
| Rotation inverse recovery | >=80% | 100.0% | 100.0% | PASS |
| Strong disruption retained lift | <=20% | 33.5% | 47.3% | FAIL |

Final smoke result:

```text
FAIL_STAGE1_V1_PREFLIGHT
```

## Control NLL Table

| Readout | Copied | Random | Generic range 14-17 | Shuffled | Legacy rotated | Dim-permuted | Zeroed 50% | Gaussian norm noise | True embedding |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| token-end | 12.250 | 18.000 | 16.331 | 17.886 | 15.913 | 15.678 | 12.802 | 16.072 | 11.971 |
| patch-boundary | 13.615 | 18.974 | 17.474 | 18.390 | 16.425 | 16.259 | 14.739 | 16.440 | 12.292 |

The good news: generic middle layers lose badly, so the main copied early layers are not just "any Qwen layer range" on this smoke.

The bad news: destroyed-input copied layers still beat random by too much. The copied core has a strong prior floor independent of usable byte-to-Qwen coordinates.

## Layer Depth Curve

| Depth | Token-end frozen gain | Patch-boundary frozen gain | Token-end copied advantage | Patch-boundary copied advantage | Verdict |
|---:|---:|---:|---:|---:|---|
| 2 | 85.8% | 74.1% | 5.399 | 4.304 | PASS frozen-core gate |
| 4 | 83.2% | 73.0% | 5.989 | 4.518 | PASS frozen-core gate; best balance |
| 6 | 83.7% | 61.5% | 7.016 | 3.398 | FAIL patch frozen-core |
| 8 | 81.0% | 52.8% | 7.499 | 3.297 | FAIL patch frozen-core |

Depth answer: 4 layers was not obviously wrong. Deeper layers improve token-end NLL advantage but make patch-boundary frozen retention worse. For this byte readout, 4 layers remains the least-bad Stage 1 depth.

## Adversarial Falsification

The strongest hostile reading is now:

```text
The adapter repairs byte-to-Qwen gauge enough to expose a real early-layer advantage,
but a large fraction of copied-vs-random NLL lift is a Qwen-core/Qwen-head language
prior that survives when the input coordinates are destroyed.
```

This does not kill every coordinate-inheritance signal. It kills v1 as Stage 1 evidence because the disruption control cannot isolate coordinate dependence cleanly.

## Why Full Stage 1 Was Not Run

The user's gate chain required smoke before full run. The smoke failed the primary stronger-disruption gate on both readouts:

- token-end retained 33.5% of inherited lift under same-norm Gaussian replacement, target <=20%.
- patch-boundary retained 47.3%, target <=20%.

Running the 1000-sequence preflight after that would violate the loop contract. Stage 2 benchmarks were also blocked.

## v2 Repair Proposal

1. Add a copied-core destroyed-input prior baseline and report coordinate lift above that prior, not only above random layers.
2. Add LM-head-only and copied-core constant-input controls to quantify the unconditional Qwen prior floor.
3. Add internal disruption controls: reset attention only, reset MLP only, randomize per-layer residual projections, or shuffle blocks internally. Input-only destruction is not isolating the causal object.
4. Add functional margin smoke on HellaSwag/PIQA train-safe subsets before any full benchmark run. NLL remains too easy to misread.
5. Keep 4 layers as the main Stage 1 depth unless the patch-boundary readout is redesigned.

## Final Verdict

```text
COORDINATE_INHERITANCE_V1_REPAIRED_PATCH_BOUNDARY_FROZEN_GAIN
GENERIC_LAYER_RANGE_CONTROL_LOSES
BUT_DESTROYED_INPUTS_STILL_RETAIN_TOO_MUCH_COPIED_CORE_LIFT
FAIL_STAGE1_V1_PREFLIGHT_AT_128_SEQUENCE_SMOKE
DO_NOT_RUN_FULL_STAGE1
DO_NOT_RUN_STAGE2
```

## Narrative Section

1. Gossip-magazine one-sentence story given only what was measured: A tiny readout-conditioned adapter made byte states look convincingly Qwen-shaped, but the copied Qwen core kept doing suspiciously well even when its input was replaced by random same-norm noise.
2. Does it survive "isn't that obvious?" and "so what?": The adapter repair is not obvious and the generic-layer failure is useful, but the disruption failure means the result does not yet prove coordinate inheritance rather than a pretrained Qwen prior floor.
3. If boring, say so: Not boring as a diagnostic, but boring as a moonshot claim. It is a killed preflight, not a paradigm-shift result.