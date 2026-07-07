# WORK LOOP - Batch 4: Phase 1.5 Dense Patch-Boundary Supervision

Date: 2026-07-07

Artifacts:
- `code/toy_weight_transplant_gauntlet.py`
- `code/codec_phase1_train.py`
- `research/work_loop_batch4.md`
- `C:/sutra_fast/codec_phase1.5/toy_degradation_curve.json`
- `C:/sutra_fast/codec_phase1.5/toy_degradation_curve.svg`
- `C:/sutra_fast/codec_phase1.5/codec_final.pt`
- `C:/sutra_fast/brainseed_v0/chart_metrics.json`

Commands run:

```powershell
python -m py_compile code/toy_weight_transplant_gauntlet.py
python code/toy_weight_transplant_gauntlet.py --tier tier25 --n-eval 100 --chart-noise 0.0 --json
python code/toy_weight_transplant_gauntlet.py --degradation-curve --n-eval 800 --curve-probe-trials 2000 --curve-output-dir C:/sutra_fast/codec_phase1.5
python -m py_compile code/codec_phase1_train.py
python code/codec_phase1_train.py --phase 1.5 --resume-checkpoint C:/sutra_fast/codec_phase1/codec_final.pt --data-dir C:/sutra_fast/data/shards_diverse --output-dir tmp_codec_phase15_dryrun --steps 1 --batch-size 2 --seq-len 256 --max-anchors 32 --warmup-steps 1 --log-every 1 --save-every 1000
python code/codec_phase1_train.py --phase 1.5 --resume-checkpoint C:/sutra_fast/codec_phase1/codec_final.pt --data-dir C:/sutra_fast/data/shards_diverse --output-dir tmp_codec_phase15_smoke --steps 100 --batch-size 4 --seq-len 512 --max-anchors 64 --warmup-steps 20 --log-every 10 --save-every 100
python code/codec_phase1_train.py --phase 1.5 --resume-checkpoint C:/sutra_fast/codec_phase1/codec_final.pt --data-dir C:/sutra_fast/data/shards_diverse --output-dir C:/sutra_fast/codec_phase1.5 --steps 5000 --batch-size 8 --seq-len 4096 --max-anchors 128 --warmup-steps 500 --log-every 50 --save-every 1000
python code/tier3_brainseed_chart_probe.py --codec-checkpoint C:/sutra_fast/codec_phase1.5/codec_final.pt --chart-only --no-artifacts --num-sequences 64 --batch-size 4 --seq-len 256 --json
python code/tier3_brainseed_chart_probe.py --codec-checkpoint C:/sutra_fast/codec_phase1.5/codec_final.pt --output-dir C:/sutra_fast/brainseed_v0 --num-sequences 64 --batch-size 4 --seq-len 256 --extract-hellaswag 512 --extract-piqa 512 --eval-hellaswag 1024 --eval-piqa 1024 --ranks 32 64 --json
```

Compute policy honored:
- Toy degradation curve was CPU-only.
- Phase 1.5 training used the GPU, no Sutra training, no teacher model.
- Gate A chart audit used CPU and no teacher candidate forwards.
- Gate B loaded Qwen3-0.6B only after Gate A passed.
- No Gate C / Sutra learning-acceleration run was launched.

## Batch 4 Executive Verdict

Phase 1.5 repaired the formal R65 chart gate but did not validate Brainseed v0.

```text
CONFIRM_TOY_DEGRADATION_CLIFF
CONFIRM_PHASE_1_5_DENSE_PATCH_SUPERVISION_IMPLEMENTED
PASS_GATE_A_WITH_PHASE_1_5
FAIL_GATE_B_FROZEN_SCORER
VOID_BRAINSEED_V0_CLAIM
VOID_GATE_C_TRAINING_RUN
```

Phase 1.5 improved patch-boundary retrieval substantially:

| Metric | Batch 3 Phase 1 | Batch 4 Phase 1.5 | Delta |
|---|---:|---:|---:|
| Token-end top-1 | 86.57% | 78.38% | -8.19pp |
| Token-end top-10 | 97.82% | 93.41% | -4.41pp |
| Patch-boundary top-1 | 23.71% | 37.89% | +14.18pp |
| Patch-boundary top-10 | 35.30% | 57.28% | +21.98pp |
| Rare patch top-1 | 18.28% | 32.44% | +14.16pp |
| Rare patch top-10 | 40.52% | 62.76% | +22.24pp |

Formal R65 Gate A passes because patch top-1 is above 30%, patch gap is +36.65pp,
rare patch top-1 is above 15%, and controls remain near floor.

But the mandatory toy degradation curve found a cliff: transplant accuracy falls
below 50% when measured chart top-1 is about 49.35%, and the first point above
80% transplant accuracy is about 90% chart top-1. Phase 1.5's held-out patch
chart at 37.89% is a formal Gate A pass, not evidence that transplant should work.
The scorer test confirmed that caution: frozen Brainseed ridge did not beat
codec-only on HellaSwag or PIQA.

Narrative gate: the codec repair is useful infrastructure, but the moonshot claim
is blocked. The headline is not "born knowing"; it is "we repaired the map enough
to test the seed, and the seed did not yet contain useful judgment."

## Iteration 31: Toy Degradation Curve

### Register
- Add `chart_noise` to Tier 2.5.
- Calibrate noise levels for chart top-1 near 90/80/70/60/50/40/25%.
- Measure transplant accuracy at each calibrated level before Phase 1.5.

### Design-Gate
Passed. `chart_noise` is per-occurrence lookup corruption:
- `0.0`: return the correct key/value chart vector.
- `1.0`: return a random wrong chart vector of the same kind.
- Calibration measures actual key/value chart top-1, not just the configured probability.

This is intentionally harsh because the adversarial question is whether transplant
requires multiple critical anchors to be correct.

### Implement
Modified `code/toy_weight_transplant_gauntlet.py`:
- `Tier25Config.chart_noise`
- `BytePatchCodec._maybe_corrupt_lookup`
- `chart_top1_accuracy`
- `calibrate_chart_noise`
- `run_tier25_degradation_curve`
- `write_degradation_svg`
- CLI: `--degradation-curve`, `--curve-targets`, `--curve-output-dir`

### Dry-Run
`python -m py_compile code/toy_weight_transplant_gauntlet.py` passed.

Zero-noise smoke:

| Metric | Value |
|---|---:|
| byte codec chart | 100.00% |
| random control | 20.00% |
| shuffled control | 20.00% |
| wrong-circuit control | 16.00% |

### Evidence-Gate

| Target chart | Calibrated noise | Measured chart | Transplant acc | Random | Shuffled | Wrong circuit |
|---:|---:|---:|---:|---:|---:|---:|
| 90% | 0.10 | 90.02% | 86.00% | 24.87% | 21.00% | 19.75% |
| 80% | 0.20 | 79.50% | 71.75% | 22.38% | 34.75% | 26.62% |
| 70% | 0.30 | 71.27% | 62.50% | 22.38% | 29.50% | 33.00% |
| 60% | 0.41 | 60.13% | 51.25% | 23.13% | 24.37% | 29.38% |
| 50% | 0.50 | 49.35% | 43.50% | 27.87% | 23.87% | 29.25% |
| 40% | 0.60 | 40.22% | 36.63% | 21.50% | 24.75% | 30.00% |
| 25% | 0.75 | 24.77% | 30.50% | 26.50% | 23.87% | 26.50% |

Precommitted checks:
- `transplant_accuracy > 80% at chart_top1 = 25%`: FAIL.
- `transplant_accuracy < 50% at chart_top1 = 50%`: PASS.

Verdict:

```text
CLIFF_PHASE_1_5_NEEDS_ABOVE_50_CHART_TOP1
```

### Narrative Gate
The glasses-cliff is real in the toy. A 25-40% chart is not a graceful degradation
zone for this transplant harness; it is mostly noise. Phase 1.5 therefore needs
to target more than the R65 formal 30% patch top-1 if the goal is actual transplant.

## Iteration 32: Design Phase 1.5 Training

### Register
- Add token-end plus patch-boundary anchors to `codec_phase1_train.py`.
- For each patch boundary, target the teacher token span containing that byte.
- Keep InfoNCE objective and architecture fixed.

### Design-Gate
Passed with one important constraint: supervising all token ends plus all patch
boundaries per sequence would make the in-batch InfoNCE matrix too large.
At `seq_len=4096`, each sequence has about 1009 token-end anchors and 1024 patch
anchors. With batch 8 this would be roughly 16K anchors and a 16K x 16K similarity
matrix.

Anchor mixing strategy:
- Keep `max_anchors=128` per sequence for the same compute envelope as Phase 1.
- In Phase 1.5, sample 75% patch-boundary anchors and 25% token-end anchors.
- Deduplicate exact `(position, token_id)` overlap.
- Log found vs used anchors every training interval.

Capacity assessment:
- The 4-layer / 256-dim codec has enough capacity to learn token identity at token ends.
- Phase 1.5 asks the same encoder to make intermediate states useful, not to store 4x more independent facts.
- Given the toy cliff, capacity is questionable if held-out patch chart stays below 50% after 5000 steps.
- No architecture growth was made before this first dense-supervision test because the repair is supposed to be cheap and automatic.

### Narrative Gate
The fix is allowed only as cheap repair. If it needs repeated architecture/hyperparameter
search, it becomes a tokenizer pretraining project rather than a moonshot seed.

## Iteration 33: Implement Phase 1.5 Training

### Register
- Add `--phase 1|1.5`.
- Add warm-start from Phase 1 checkpoint.
- Fix padded anchor masking.

### Implement
Modified `code/codec_phase1_train.py`:
- `find_codec_anchors` finds token spans once and emits token-end and patch-boundary anchors.
- Patch-boundary anchor rule matches the Gate A probe: byte position `P-1, 2P-1, ...` maps to the teacher token whose span contains that byte.
- `infonce_loss_flat` applies InfoNCE only over real anchors, not padded token-id-0 anchors.
- `--resume-checkpoint` warm-starts Phase 1.5 from `C:/sutra_fast/codec_phase1/codec_final.pt`.
- Training logs anchor density and saves Phase 1.5 checkpoint config.

### Dry-Run
`python -m py_compile code/codec_phase1_train.py` passed.

One-step dry run:

| Field | Value |
|---|---:|
| seq_len | 256 |
| batch | 2 |
| max anchors | 32 |
| anchors in batch | 61 |
| avg token used / seq | 6.5 |
| avg patch used / seq | 24.0 |
| loss | 2.2806 |
| top-1 | 42.62% |

### Narrative Gate
The implementation changes the chart positions, not the ontology. This is a legitimate
repair, not a new claim.

## Iteration 34: Smoke Phase 1.5 Training

### Register
- Run 100 steps on a small sample.
- Verify loss decreases, no NaN/Inf, checkpoint saves, anchors are correct.

### Smoke
Command used `seq_len=512`, `batch=4`, `max_anchors=64`, `steps=100`.

| Step | Loss | Top-1 | Anchors | Token used / seq | Patch used / seq |
|---:|---:|---:|---:|---:|---:|
| 1 | 3.5718 | 22.71% | 251 | 14.75 | 48.0 |
| 50 | 3.0999 | 30.68% | 251 | 14.61 | 48.0 |
| 100 | 2.6162 | 33.87% | 248 | 14.45 | 48.0 |

Checkpoint saved:

```text
tmp_codec_phase15_smoke/codec_step100.pt
tmp_codec_phase15_smoke/codec_final.pt
```

### Evidence-Gate
Smoke passed:
- loss decreased;
- no NaN/Inf;
- checkpoint saved;
- anchor counts matched the intended 75/25 patch/token mix.

### Narrative Gate
The repair is mechanically sound and cheap enough to try once. It does not yet
prove the chart is good enough.

## Iteration 35: Full Phase 1.5 Training Run

### Register
- Train Phase 1.5 for 5000 steps from the Phase 1 checkpoint.
- Save final checkpoint to `C:/sutra_fast/codec_phase1.5/codec_final.pt`.

### Training Configuration

| Field | Value |
|---|---:|
| steps | 5000 |
| batch size | 8 |
| seq len | 4096 |
| max anchors / seq | 128 |
| patch fraction | 0.75 |
| patch anchors used / seq | 96.0 |
| token anchors used / seq | 31.25 |
| anchors used / seq | 127.25 |
| avg patch anchors found / seq | 1023.97 |
| avg token anchors found / seq | 1009.58 |
| avg overlap found / seq | 253.06 |
| elapsed | 1532.8s |

Training curve highlights:

| Step | Loss | Top-1 |
|---:|---:|---:|
| 1 | 6.2514 | 4.42% |
| 1000 | 3.9702 | 21.95% |
| 2000 | 4.0637 | 23.85% |
| 3000 | 3.7869 | 25.34% |
| 4000 | 3.4575 | 29.57% |
| 4350 | 3.3307 | 32.16% |
| 4950 | 3.2878 | 30.34% |
| 5000 | 3.6411 | 26.15% |

The printed `best_acc=0.703125` is inherited from the warm-start checkpoint and
should not be interpreted as Phase 1.5 dense-anchor accuracy. The useful Phase
1.5 evidence is the logged dense-anchor curve above and the held-out Gate A probe.

Checkpoints saved:

```text
C:/sutra_fast/codec_phase1.5/codec_step1000.pt
C:/sutra_fast/codec_phase1.5/codec_step2000.pt
C:/sutra_fast/codec_phase1.5/codec_step3000.pt
C:/sutra_fast/codec_phase1.5/codec_step4000.pt
C:/sutra_fast/codec_phase1.5/codec_step5000.pt
C:/sutra_fast/codec_phase1.5/codec_final.pt
```

### Evidence-Gate
Training was numerically stable and produced the requested final checkpoint.

### Narrative Gate
The repair remained a single warm-start run, not an open-ended search. That keeps
the narrative alive enough to test, but the toy cliff means the final chart must
be judged harshly.

## Iteration 36: Re-run Gate A With Phase 1.5 Codec

### Register
- Run the same 64-sequence chart-only audit as Batch 3.
- Apply precommitted R65 Gate A thresholds.

### Evidence-Gate

| Anchor type | N | Real top-1 | Real top-5 | Real top-10 | Best control top-1 | Gap |
|---|---:|---:|---:|---:|---:|---:|
| Token end | 3992 | 78.38% | 89.08% | 93.41% | 4.31% | +74.07pp |
| Patch boundary | 4096 | 37.89% | 49.34% | 57.28% | 1.25% | +36.65pp |
| Rare patch boundary | 1893 | 32.44% | 51.51% | 62.76% | 0.11% | +32.33pp |

Patch-boundary controls:

| Control | Top-1 | Top-5 | Top-10 |
|---|---:|---:|---:|
| Per-occurrence random target | 0.02% | 0.17% | 0.29% |
| Fixed shuffled target | 0.07% | 0.20% | 0.24% |
| Random codec | 0.15% | 0.29% | 0.46% |
| Rotated chart | 0.05% | 0.22% | 0.34% |
| Frequency lookup | 1.25% | 9.52% | 13.33% |

R65 Gate A checks:
- token-end top-1 >= 50%: PASS.
- token-end real-vs-fixed-shuffled gap >= 25pp: PASS.
- per-occurrence random target within 2x chance: PASS.
- patch-boundary top-1 >= 30% or top-10 >= 65%: PASS by top-1.
- patch-boundary real-vs-best-control gap >= 15pp: PASS.
- rare patch top-1 >= 15%: PASS.
- rare patch beats best control by >= 8pp: PASS.

Verdict:

```text
PASS_GATE_A_WITH_PHASE_1_5
```

### Narrative Gate
The bridge now reaches the door according to the formal chart gate. But it is
still a narrow bridge: patch top-1 is 37.89%, below the toy-degradation >50%
usefulness warning.

## Iteration 37: Analyze Phase 1.5 Gate A Results

### Register
- Decide whether to proceed to Gate B.

### Analysis
Phase 1.5 did exactly what it was supposed to do under R65:
- patch top-1 improved +14.18pp;
- patch top-10 improved +21.98pp;
- rare patch top-1 improved +14.16pp;
- controls remained near floor.

The cost was token-end degradation:
- token-end top-1 fell from 86.57% to 78.38%;
- token-end top-10 fell from 97.82% to 93.41%.

This trade is acceptable for Gate A because Phase 2 consumes patch boundaries,
not token ends. It is not enough to overclaim transplant viability because the
toy cliff says the current 37.89% patch chart is below the likely robust-transfer
zone.

Decision:

```text
PROCEED_TO_GATE_B_AS_FORMAL_FALSIFIER
LOWER_PRIOR_FROM_TOY_CLIFF
```

### Narrative Gate
The fix is real and automatic, but not headline-worthy by itself. The scorer must
show visible semantic judgment or Brainseed remains only a chart repair.

## Iteration 38: Design Frozen Brainseed v0 Scorer

### Register
- Use the already implemented minimal frozen scorer path in `tier3_brainseed_chart_probe.py`.
- Keep it teacher-free at evaluation.

### Design-Gate
The scorer is a ridge model over frozen codec pair features:

```text
context_feature = pooled codec feature(context)
candidate_feature = pooled codec feature(candidate)
pair_feature = [context, candidate, context*candidate, abs(context-candidate)]
target = Qwen candidate mean log-likelihood
fit = PCA rank 32/64 + ridge regression
```

This is a minimal Gate B falsifier, not the full just-distill baseline suite. It
is still enough to fail Gate B if it cannot beat codec-only cosine scoring.

### Narrative Gate
If this wins, it is still teacher-margin regression and needs boring distillation
baselines before public claims. If it loses to codec-only, Brainseed v0 is dead
without needing those baselines.

## Iteration 39: Extract Teacher Margins and Build Scorer

### Register
- Extract 512 HellaSwag-train and 512 PIQA-train examples.
- Score candidates with Qwen3-0.6B.
- Fit rank 32 and rank 64 ridge scorers.
- Save artifacts to `C:/sutra_fast/brainseed_v0/`.

### Evidence-Gate
Teacher and datasets loaded from local cache/offline mode. Artifacts saved:

```text
C:/sutra_fast/brainseed_v0/basis_B_rank32.fp16.pt
C:/sutra_fast/brainseed_v0/basis_B_rank64.fp16.pt
C:/sutra_fast/brainseed_v0/energy_E_rank32.fp16.pt
C:/sutra_fast/brainseed_v0/energy_E_rank64.fp16.pt
C:/sutra_fast/brainseed_v0/chart_metrics.json
C:/sutra_fast/brainseed_v0/codec_manifest.json
```

### Narrative Gate
The artifact exists, but existence is not signal. Gate B decides.

## Iteration 40: Evaluate Scorer and Gate B Verdict

### Register
- Evaluate on 1024 HellaSwag validation and 1024 PIQA validation examples.
- Compare real Brainseed ridge to codec-only.

### Evidence-Gate

| Rank | HellaSwag Brainseed | HellaSwag codec-only | Delta | PIQA Brainseed | PIQA codec-only | Delta |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 26.27% | 27.25% | -0.98pp | 47.36% | 48.83% | -1.46pp |
| 64 | 27.15% | 27.25% | -0.10pp | 47.27% | 48.83% | -1.56pp |

Gate B requires:
- HellaSwag real Brainseed beats codec-only by >=5pp: FAIL.
- PIQA real Brainseed beats codec-only by >=3pp: FAIL.
- Aggregate lift over controls positive: FAIL.

Verdict:

```text
FAIL_GATE_B_FROZEN_SCORER
VOID_BRAINSEED_V0_CLAIM
VOID_GATE_C_LEARNING_ACCELERATION
```

### Narrative Gate
This is the adversary won over: the chart repair worked, but the seed did not
add judgment. A small ridge over codec features and teacher margins is not enough
to produce born-knowing behavior.

## Final Batch 4 Synthesis

What survived:
- Dense patch-boundary supervision is the correct repair for the Phase 1 / Phase 2
  position mismatch.
- The 256-dim / 4-layer codec can improve patch-boundary chart quality without
  architecture growth.
- Formal Gate A now passes with strong control gaps.

What died:
- Treating formal Gate A pass as enough for transplant viability. The toy curve
  says the useful-transplant target is likely above 50% chart top-1, and Phase 1.5
  reached 37.89% patch top-1.
- Frozen Brainseed v0 ridge scorer as currently implemented. It loses to codec-only
  on both HellaSwag and PIQA.
- Any Gate C / learning-acceleration run in this batch.

Next honest options:
1. Improve the chart toward the toy-derived >50% patch target before another scorer.
   Likely knobs: more Phase 1.5 steps, higher patch fraction, larger max anchors,
   or a 6-layer / 384-dim codec if cheap runs plateau.
2. Replace the minimal ridge scorer with the just-distill baseline suite before
   reviving Brainseed: MLP-on-codec, bilinear-on-codec, retrieval-lite, length/frequency,
   and matched teacher-margin distillation.
3. If the goal is the moonshot result rather than preserving Brainseed, route back
   to chain-init / byteified pretrained-core options because Gate B did not show
   a visible birth jump.

Final verdict:

```text
PHASE_1_5_REPAIR_SUCCESSFUL_FOR_GATE_A
BRAINSEED_V0_SCORER_FAILED_GATE_B
NO_REAL_SUTRA_OR_BORN_KNOWING_CLAIM_SUPPORTED
```
