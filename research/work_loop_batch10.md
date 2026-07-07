# WORK LOOP - Batch 10: Functional-Margin Shadow + v2 Metrics Gate

Date: 2026-07-07

Iterations: 91-100 requested. Executed Iterations 91-93 through the functional-margin shadow smoke. Iterations 94-100 were blocked by the precommitted kill condition and were not run.

## Artifacts

- `code/coordinate_inheritance.py`
- `tmp_coordinate_inheritance_v2/dry_margin/functional_margin_shadow.json`
- `tmp_coordinate_inheritance_v2/margin_shadow_smoke50/functional_margin_shadow.json`
- `research/work_loop_batch10.md`

The interrupted `tmp_coordinate_inheritance_v2/margin_shadow_smoke100*` attempts were not used for verdict. The final verdict is based on the completed 50-example train-safe smoke.

## Commands Run

```powershell
python -m py_compile code/coordinate_inheritance.py
python code/coordinate_inheritance.py --mode benchmark --functional-margin-shadow --codec-checkpoint C:/sutra_fast/codec_phase1.5/codec_final.pt --output-dir tmp_coordinate_inheritance_v2/dry_margin --device cuda --layers 4 --adapter-checkpoint tmp_coordinate_inheritance_v1/smoke128_repair/calibration_adapter.pt --benchmarks hellaswag --benchmark-examples 2 --benchmark-split train --benchmark-readout token_end --benchmark-max-bytes 1536 --bootstrap-samples 5 --progress
python code/coordinate_inheritance.py --mode benchmark --functional-margin-shadow --codec-checkpoint C:/sutra_fast/codec_phase1.5/codec_final.pt --output-dir tmp_coordinate_inheritance_v2/margin_shadow_smoke50 --device cuda --layers 4 --adapter-checkpoint tmp_coordinate_inheritance_v1/smoke128_repair/calibration_adapter.pt --benchmarks hellaswag piqa arc_easy --benchmark-examples 50 --benchmark-split train --benchmark-readout token_end --benchmark-max-bytes 1536 --bootstrap-samples 200 --progress
```

A 100-example attempt was stopped because ARC-Easy cache resolution stalled after HellaSwag and PIQA had already reproduced fail rows. `code/coordinate_inheritance.py` was then patched to load cached ARC Arrow files directly, and the completed 50-example smoke was run from scratch.

## Precommitted Verdict Tokens

```text
PASS_FUNCTIONAL_MARGIN_SHADOW - inherited path >= +1pp MCQ accuracy over destroyed-input and random on >=2 of 3 benchmarks
FAIL_FUNCTIONAL_MARGIN_SHADOW - inherited path < +1pp MCQ accuracy advantage
MARGINAL_FUNCTIONAL_MARGIN - inherited +1-2pp, ambiguous
```

Final verdict:

```text
FAIL_FUNCTIONAL_MARGIN_SHADOW
FAIL_FUNCTIONAL_MARGIN
SURFACE_COMPATIBILITY_ONLY
DO_NOT_RUN_V2_STAGE1
DO_NOT_RUN_STAGE2
```

## Implementation Changes

Only `code/coordinate_inheritance.py` was modified.

1. Added functional-margin shadow support to benchmark mode via `--functional-margin-shadow`.
2. Added `--benchmark-split` so train-safe subsets can be selected explicitly.
3. Added per-example candidate score recording: all continuation NLLs, predicted ranking, gold-vs-best-wrong margin, and correctness.
4. Added full Qwen teacher continuation scoring for top-1 and full-ranking agreement.
5. Added required variants:
   - `main_inherited`
   - `random_core`
   - `shuffled_core`
   - `generic_pretrained_core`
   - `gaussian_destroyed_input`
   - `inverse_recovered_rotation`
   - `true_embedding_truncated_qwen`
6. Added functional-shadow verdict computation against random core and Gaussian destroyed-input controls.
7. Added direct cached Arrow loading for ARC-Easy and ARC-Challenge to avoid slow `load_dataset` cache resolution in offline mode.

No v2 Stage 1 metric implementation was added because the functional-margin shadow failed first.

## Iteration 91: Functional-Margin Shadow Design

The shadow test uses the existing byte-to-Qwen benchmark scoring path, with the saved v1 readout-conditioned adapter:

```text
tmp_coordinate_inheritance_v1/smoke128_repair/calibration_adapter.pt
```

For each benchmark example and each variant, the scorer records:

- MCQ accuracy: whether the gold continuation is ranked first by lowest NLL/token.
- Functional margin: `best_wrong_nll_per_token - gold_nll_per_token`, so positive is good.
- Qwen preference agreement: top-1 and exact full-ranking agreement with full `Qwen/Qwen3-0.6B` continuation scoring.

The final smoke uses train splits only:

| Benchmark | Split | Examples | Readout |
|---|---|---:|---|
| HellaSwag | train | 50 | token_end |
| PIQA | train | 50 | token_end |
| ARC-Easy | train | 50 | token_end |

## Iteration 92: Margin Shadow Smoke Results

Completed artifact:

```text
tmp_coordinate_inheritance_v2/margin_shadow_smoke50/functional_margin_shadow.json
```

Gate summary:

| Benchmark | Main inherited | Random core | Gaussian destroyed | Main - random | Main - Gaussian | Gate |
|---|---:|---:|---:|---:|---:|---|
| HellaSwag | 20.0% | 26.0% | 24.0% | -6.0pp | -4.0pp | FAIL |
| PIQA | 42.0% | 58.0% | 42.0% | -16.0pp | 0.0pp | FAIL |
| ARC-Easy | 22.0% | 26.0% | 16.0% | -4.0pp | +6.0pp | FAIL |

Precommitted pass condition required inherited to beat both random and Gaussian by at least +1pp on at least 2 of 3 benchmarks. It passed 0 of 3.

## Margin and Teacher-Agreement Results

| Benchmark | Variant | Accuracy | Mean margin | Qwen top-1 agreement |
|---|---|---:|---:|---:|
| HellaSwag | full Qwen teacher | 48.0% | -0.062 | n/a |
| HellaSwag | main inherited | 20.0% | -1.135 | 24.0% |
| HellaSwag | random core | 26.0% | -0.681 | 22.0% |
| HellaSwag | Gaussian destroyed | 24.0% | -0.774 | 32.0% |
| HellaSwag | true-embedding truncated | 28.0% | -0.952 | 32.0% |
| PIQA | full Qwen teacher | 74.0% | +0.138 | n/a |
| PIQA | main inherited | 42.0% | -0.029 | 40.0% |
| PIQA | random core | 58.0% | +0.209 | 52.0% |
| PIQA | Gaussian destroyed | 42.0% | +0.029 | 40.0% |
| PIQA | true-embedding truncated | 46.0% | -0.034 | 52.0% |
| ARC-Easy | full Qwen teacher | 54.0% | -0.179 | n/a |
| ARC-Easy | main inherited | 22.0% | -1.724 | 36.0% |
| ARC-Easy | random core | 26.0% | -1.227 | 36.0% |
| ARC-Easy | Gaussian destroyed | 16.0% | -2.031 | 30.0% |
| ARC-Easy | true-embedding truncated | 34.0% | -1.295 | 44.0% |

The inherited path did not show benchmark-facing margin lift. On HellaSwag and ARC-Easy, random core had higher accuracy and better mean margin than main inherited. On PIQA, random core strongly beat main inherited in both accuracy and margin.

## Iteration 93: Analysis and Verdict

The functional-margin shadow is flat to negative.

The strongest possible favorable interpretation is narrow: ARC-Easy main inherited beat Gaussian destroyed by +6pp. But the precommitted gate required beating both Gaussian and random, and ARC main lost to random by -4pp. That is not evidence of coordinate-specific candidate discrimination.

The hostile interpretation is now the best supported one:

```text
The v1 adapter plus copied Qwen core improves token-space NLL in Stage 1, but the signal does not translate into gold-vs-distractor multiple-choice ranking on train-safe HellaSwag, PIQA, or ARC-Easy.
```

This exactly matches the B12 concern: copied Qwen coordinates can produce lexical/manifold compatibility without task-discriminative function.

## Iterations 94-100: Blocked by Kill Condition

The hard kill condition fired before v2 metrics:

```text
If functional-margin shadow is flat (<+1pp inherited over destroyed-input AND random on all 3 benchmarks): STOP IMMEDIATELY. Write batch10.md with the verdict FAIL_FUNCTIONAL_MARGIN / SURFACE_COMPATIBILITY_ONLY. Do NOT proceed to v2 metrics or Stage 2.
```

Therefore these were not run:

- v2 prior-floor decomposition implementation in the Stage 1 gate table.
- Revised 1000-sequence Stage 1.
- Stage 2 HellaSwag/PIQA/ARC benchmark escalation.
- ARC-Challenge benchmark escalation.

This halt is intentional, not an infrastructure failure.

## Adversarial Falsification

The result does not merely miss a threshold. It reverses the expected sign on two of the three benchmarks against random core:

- HellaSwag: main 20%, random 26%.
- PIQA: main 42%, random 58%.
- ARC-Easy: main 22%, random 26%.

If a hostile reviewer asks whether the 2.8-3.8 nat coordinate-specific Stage 1 NLL lift contains task-discriminative function, the honest answer from this smoke is no.

The result also blocks a rescue based on teacher preference agreement. Main inherited did not track full Qwen strongly:

- HellaSwag top-1 agreement: 24%.
- PIQA top-1 agreement: 40%.
- ARC-Easy top-1 agreement: 36%.

Those are not enough to support a claim of inherited benchmark-facing teacher function.

## What Survived

The Stage 1 NLL finding survives as a codec/Qwen-surface compatibility diagnostic. It remains real that copied early Qwen layers plus the saved adapter beat random layers on token-space NLL and that the v1 adapter fixed patch-boundary frozen-core gain.

But that surviving fact is not the moonshot. It is not evidence that inherited coordinates move hard candidate margins.

## What Died

This claim died:

```text
The coordinate-specific NLL lift is likely to become benchmark discrimination.
```

The work should not proceed to v2 Stage 1 or Stage 2 from this artifact. The correct label is:

```text
SURFACE_COMPATIBILITY_ONLY
```

## Narrative Section

1. Gossip-magazine one-sentence story given only what was measured: The copied Qwen core looked useful on token NLL, but when asked to pick the right answer, it was no better than random layers and often worse.
2. Does it survive "isn't that obvious?" and "so what?": It survives as a falsification, not as a promotion. The result directly answers the dangerous unknown from Q-Loop B12 and shows the Stage 1 signal was not task-discriminative enough to justify escalation.
3. If boring, say so: This is boring as a moonshot and valuable as a kill. It prevents another expensive loop from mistaking surface compatibility for intelligence transfer.
