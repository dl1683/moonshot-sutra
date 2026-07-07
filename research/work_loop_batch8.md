# WORK LOOP - Batch 8: Coordinate-Inheritance Prototype

Date: 2026-07-07

Iterations: 71-80

## Artifacts

Requested production artifact directory `C:/sutra_fast/coordinate_inheritance/` could not be created from this sandbox:

```text
New-Item : Access to the path 'coordinate_inheritance' is denied.
```

Repo-local artifacts were written instead:

- `code/coordinate_inheritance.py`
- `research/work_loop_batch8.md`
- `tmp_coordinate_inheritance_smoke/preflight_metrics.json`
- `tmp_coordinate_inheritance_smoke128/preflight_metrics.json`
- `tmp_coordinate_inheritance_smoke128_ft/preflight_metrics.json`
- `tmp_coordinate_inheritance_full/preflight_metrics.json`
- `tmp_coordinate_inheritance_full/calibration_adapter.pt`

## Commands Run

```powershell
python -m py_compile code/coordinate_inheritance.py
python code/coordinate_inheritance.py --mode preflight --codec-checkpoint C:/sutra_fast/codec_phase1.5/codec_final.pt --output-dir tmp_coordinate_inheritance_smoke --device cuda --layers 2 --num-sequences 4 --seq-len 128 --batch-size 1 --max-positions-per-sequence 32 --adapter-steps 2 --adapter-batch-anchors 64 --bootstrap-samples 20 --progress --json
python code/coordinate_inheritance.py --mode preflight --codec-checkpoint C:/sutra_fast/codec_phase1.5/codec_final.pt --output-dir tmp_coordinate_inheritance_smoke128 --device cuda --layers 4 --num-sequences 128 --seq-len 512 --batch-size 4 --max-positions-per-sequence 128 --adapter-steps 300 --adapter-batch-anchors 2048 --bootstrap-samples 200 --progress --json
python code/coordinate_inheritance.py --mode preflight --codec-checkpoint C:/sutra_fast/codec_phase1.5/codec_final.pt --output-dir tmp_coordinate_inheritance_finetune_smoke2 --device cuda --layers 2 --num-sequences 16 --seq-len 256 --batch-size 2 --max-positions-per-sequence 64 --adapter-steps 50 --adapter-batch-anchors 512 --finetune-core-steps 2 --finetune-batch-sequences 1 --bootstrap-samples 50 --progress --json
python code/coordinate_inheritance.py --mode preflight --codec-checkpoint C:/sutra_fast/codec_phase1.5/codec_final.pt --output-dir tmp_coordinate_inheritance_smoke128_ft --device cuda --layers 4 --num-sequences 128 --seq-len 512 --batch-size 4 --max-positions-per-sequence 128 --adapter-steps 300 --adapter-batch-anchors 2048 --finetune-core-steps 5 --finetune-batch-sequences 1 --finetune-lr 1e-5 --bootstrap-samples 200 --progress --json
python code/coordinate_inheritance.py --mode preflight --codec-checkpoint C:/sutra_fast/codec_phase1.5/codec_final.pt --output-dir tmp_coordinate_inheritance_full --device cuda --layers 4 --num-sequences 1000 --seq-len 512 --batch-size 4 --max-positions-per-sequence 128 --adapter-steps 1000 --adapter-batch-anchors 4096 --finetune-core-steps 5 --finetune-batch-sequences 1 --finetune-lr 1e-5 --bootstrap-samples 1000 --progress --json
```

## Executive Verdict

```text
FAIL_STAGE1_CODEC_GAUGE_PREFLIGHT
KILL_COORDINATE_INHERITANCE_V0_BEFORE_BENCHMARK_TRAINING
DO_NOT_RUN_STAGE2_BENCHMARKS_FROM_THIS_PREFLIGHT
```

The important result is mixed but decisive under the precommitted gate. The calibration adapter works: it is tiny, it closes most of the codec-to-true-embedding gap, and copied Qwen layers beat random layers by far more than the required 2 nats/token on both token-end and patch-boundary streams.

But Stage 1 still fails two controls in the 1000-sequence full preflight:

1. Patch-boundary frozen-core retention is `66.3%`, below the required `70%`.
2. Token-end no-inverse rotation retains `33.0%` of inherited lift, above the allowed `30%` collapse threshold.

Per the Q-Loop B10 contract, this kills coordinate-inheritance v0 before benchmark training. Stage 2 code exists as a prototype mode, but HellaSwag/PIQA/ARC runs were not launched because the gate chain says not to proceed.

## Iteration 71: Architecture Design

Implemented architecture:

```text
bytes -> frozen Phase 1.5 codec encoder -> calibration adapter -> copied Qwen layers -> Qwen norm/LM head
```

Concrete choices:

- Codec: `C:/sutra_fast/codec_phase1.5/codec_final.pt`, frozen.
- Adapter: `RMSNorm(256) -> Linear(256, 1024)`, 263,424 params.
- Inherited core: first `N` Qwen3-0.6B layers, tested at 2 and 4 layers; full preflight used 4.
- Output: copied Qwen final norm + LM head; metric is token-space next-token NLL.

Controls implemented:

- Random Qwen-shaped layers with copied Qwen head.
- Shuffled copied layer order.
- Random orthogonal input-gauge rotation without inverse.
- Random orthogonal rotation with inverse recovery sanity.
- True Qwen embedding input upper bound.
- Short fp32 copied-core finetune for frozen-core retention.

Not implemented in this batch:

- Generic pretrained control with a different model family.
- Tokenized compressed sibling.
- Byte decoder and byte BPB.
- Full model-weight basis rotation; current rotation is input-gauge disruption/recovery.

## Iterations 72-73: Preflight Pipeline + Adapter

`code/coordinate_inheritance.py` now supports:

- codec/tokenizer/Qwen loading in offline mode;
- token-end and patch-boundary anchor collection from shard bytes;
- small adapter training against raw Qwen embeddings;
- copied/random/shuffled/rotation/true-embedding NLL evaluation;
- bootstrap CIs over per-sequence NLL deltas;
- optional benchmark prototype mode gated by saved adapter.

Full preflight adapter training:

| Metric | Value |
|---|---:|
| Adapter params | 263,424 |
| Adapter gate <=2M | PASS |
| Training anchors | 199,817 |
| Steps | 1000 |
| Batch anchors | 4096 |
| Loss first | 0.60636 |
| Loss last | 0.04841 |
| Loss min | 0.04674 |

The adapter is not the failure point. It fit the codec-to-Qwen embedding gauge cheaply.

## Iteration 74: 128-Sequence Smoke

The 128-sequence smoke was run before the full pass. It showed the right qualitative pattern and exposed the borderline patch-boundary frozen-core problem.

| Readout | Random NLL | Copied NLL | True NLL | Random-Copied | Gap closure | Frozen/core finetune ratio | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| token-end | 17.333 | 12.175 | 11.971 | 5.159 | 96.2% | 74.5% | PASS |
| patch-boundary | 17.392 | 13.499 | 12.292 | 3.892 | 76.3% | 68.9% | FAIL |

The failure was close, but close is still fail. That justified the full 1000-sequence preflight rather than benchmark escalation.

## Iteration 75: Full Stage 1 Preflight

Full run settings:

| Field | Value |
|---|---:|
| Sequences per readout | 1000 |
| Eval sequences per readout | 200 |
| Seq len | 512 bytes |
| Max anchors per sequence | 128 |
| Copied layers | 4 |
| Adapter steps | 1000 |
| Core finetune steps | 5 |

### Token-End Results

| Variant | NLL | Next-token acc |
|---|---:|---:|
| copied + calibrated | 12.051 | 0.704% |
| random + calibrated | 18.182 | 0.000% |
| shuffled + calibrated | 18.111 | 0.004% |
| copied + raw codec | 15.497 | 0.054% |
| copied + true embeddings | 12.009 | 0.844% |
| copied + no-inverse rotation | 16.157 | 0.041% |
| copied + inverse recovery | 12.051 | 0.704% |
| copied + calibrated + finetuned core | 9.699 | 3.917% |

Token-end gate metrics:

| Gate | Required | Observed | Status |
|---|---:|---:|---|
| Copied vs random advantage | >= 2.0 nats | 6.131 nats, CI [5.957, 6.322] | PASS |
| Gap to true / closure | <=1.5 nats or >=60% | 0.042 nats, 99.3% closure | PASS |
| Frozen-core gain fraction | >=70% | 72.3% | PASS |
| No-inverse rotation collapse | <=30% lift retained | 33.0% retained | FAIL |
| Correct inverse recovery | >=80% | 100.0% | PASS |

### Patch-Boundary Results

| Variant | NLL | Next-token acc |
|---|---:|---:|
| copied + calibrated | 13.200 | 0.449% |
| random + calibrated | 17.898 | 0.000% |
| shuffled + calibrated | 18.460 | 0.000% |
| copied + raw codec | 16.527 | 0.012% |
| copied + true embeddings | 12.541 | 0.701% |
| copied + no-inverse rotation | 16.493 | 0.012% |
| copied + inverse recovery | 13.200 | 0.457% |
| copied + calibrated + finetuned core | 10.811 | 1.827% |

Patch-boundary gate metrics:

| Gate | Required | Observed | Status |
|---|---:|---:|---|
| Copied vs random advantage | >= 2.0 nats | 4.698 nats, CI [4.542, 4.838] | PASS |
| Gap to true / closure | <=2.0 nats or >=60% | 0.659 nats, 87.7% closure | PASS |
| Frozen-core gain fraction | >=70% | 66.3% | FAIL |
| No-inverse rotation collapse | <=30% lift retained | 29.9% retained | PASS |
| Correct inverse recovery | >=80% | 100.0% | PASS |

### Stage 1 Gate Table

| Gate | Token-end | Patch-boundary | Overall |
|---|---:|---:|---:|
| Adapter <=2M | PASS | PASS | PASS |
| Copied advantage >=2 nats | PASS | PASS | PASS |
| Gap closure / true gap | PASS | PASS | PASS |
| Frozen-core >=70% of finetune gain | PASS | FAIL | FAIL |
| No-inverse rotation collapses | FAIL | PASS | FAIL |
| Correct inverse recovers >=80% | PASS | PASS | PASS |

Final Stage 1 result:

```text
FAIL_STAGE1_CODEC_GAUGE_PREFLIGHT
```

## Iterations 76-78: Benchmark Setup And Halt

Benchmark mode was implemented in `code/coordinate_inheritance.py`:

```powershell
python code/coordinate_inheritance.py --mode benchmark --adapter-checkpoint <adapter.pt> --benchmarks hellaswag piqa arc_easy arc_challenge
```

It supports main inherited, random, shuffled, and no-inverse rotated variants, and uses token-space continuation scoring through byte-derived codec states.

It was not run for Stage 2 because Stage 1 failed. This is the correct behavior under the gate chain. Running HellaSwag after a failed gauge preflight would only produce an uninterpretable number.

## Iteration 79: Control Analysis

The calibration result is real:

- Token-end raw codec copied NLL: 15.497 -> calibrated copied NLL: 12.051.
- Patch-boundary raw codec copied NLL: 16.527 -> calibrated copied NLL: 13.200.
- Calibrated patch-boundary closes 87.7% of the random-to-true gap.
- Calibrated token-end nearly reaches the true-embedding truncated upper bound.

The inherited-core signal is also real:

- Token-end copied advantage over random: 6.13 nats/token.
- Patch-boundary copied advantage over random: 4.70 nats/token.
- Shuffled layer order collapses or reverses the benefit.

But the two failed controls matter:

1. Patch-boundary finetuning creates too much extra held-out gain. Frozen copied coordinates explain 66.3% of post-finetune gain, not 70%. This says the inherited core is load-bearing, but not yet load-bearing enough at the exact byte-native readout surface.
2. Token-end no-inverse rotation keeps 33.0% of inherited lift. That is close to the collapse threshold but fails the precommitted bound. The token-end stream may still expose enough lexical/token-identity structure for a damaged gauge to limp above the collapse criterion.

## Iteration 80: Batch 8 Synthesis

Coordinate-inheritance v0 is killed before benchmark training. That does not mean the direction is permanently dead; it means this v0 cannot be promoted to Stage 2 under the adversarial contract.

The evidence says:

```text
A tiny adapter can repair codec-to-Qwen gauge surprisingly well.
Copied Qwen layers are strongly better than random and shuffled layers.
The specific inherited geometry is partly load-bearing.
But the Stage 1 disruption/frozen-core proof is not clean enough.
```

Recommended next repairs before any benchmark run:

1. Run a layer-depth curve at 2/4/6 layers with the same full 1000-sequence gate. The current 4-layer truncated head may overstate finetune gain or rotation robustness.
2. Add a patch-boundary-specific adapter or offset-conditioned adapter while staying <=2M params. The patch surface is the byte-native bottleneck.
3. Replace the input-only rotation sanity with a true residual-basis transform or a stronger destructive rotation control. The current input-gauge test is useful but not the final geometry proof.
4. Increase frozen-core preflight rigor before finetune: evaluate whether longer adapter training or held-out adapter early stopping improves patch-boundary frozen retention above 70% without core updates.
5. Do not run HellaSwag/PIQA/ARC until Stage 1 passes. The Stage 2 benchmark code is ready, but the gate says stop.

Final verdict:

```text
COORDINATE_INHERITANCE_V0_SHOWS_STRONG_PREFLIGHT_SIGNAL
BUT_FAILS_PRECOMMITTED_STAGE1_CONTROLS
KILL_V0_BEFORE_STAGE2
LOOP_CONTINUES_WITH_GAUGE_AND_PATCH_BOUNDARY_REPAIR
```
