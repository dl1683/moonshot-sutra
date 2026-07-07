# W-Loop B16: CTI-1 Board 1 - Random Transformer Modular Arithmetic

**Date:** 2026-07-07
**Verdict token:** `PROXY_ONLY_LAW`
**Task:** addition mod 97
**Model:** random-init 4-layer transformer, 1,015,905 parameters
**Device:** NVIDIA GeForce RTX 5090 Laptop GPU

---

## Grounding

Read first, in order:

1. `research/VISION.md`
2. `research/CTI_PRECOMMIT_SPEC.md`
3. `research/work_loop_batch15.md`
4. `research/dual_loop_supervisor_checkin_13.md`

Binding interpretation: CTI-1 Board 1 is a clean compute-distortion lab. No pretrained weights were loaded. The primary measurement is `D_func = 1 - held_out_accuracy` over the full held-out modular-arithmetic split at every checkpoint.

## Smoke Run

Completed before the full run; smoke summary artifact is listed below.

The smoke path trained `label_only` for 10 steps, recorded checkpoint step 10, and verified `cumulative_flops = 6 * total_params * batch_size * checkpoint_step`.

## Configuration

| Parameter | Value |
|---|---:|
| p | 97 |
| Train examples | 4704 |
| Held-out examples | 4705 |
| Batch size | 512 |
| Learning rate | 0.001 |
| Weight decay | 1.0 |
| Max steps | 3000 |
| Checkpoints | 10, 30, 100, 300, 1000, 3000 |
| Fit-only checkpoints | 10, 30, 100 |
| Held-out forecast checkpoints | 300, 1000, 3000 |

## Prediction Lock

Predictions were written after all three interventions reached step 100 and before any training or evaluation at steps 300, 1000, or 3000. The lock record is in `tmp_work_loop_b16/cti1_board1_predictions.json`.

CTI predicted step-3000 ranking:

```text
quarter_data < label_only < shuffled_labels
```

Actual step-3000 ranking:

```text
label_only < quarter_data < shuffled_labels
```

## Checkpoint Matrix

| Intervention | Step | GFLOPs | D_func | D_proxy | D_gap | Train Acc | Held-out Acc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| label_only | 10 | 31.209 | 0.991711 | 4.597750 | 0.018284 | 2.66% | 0.83% |
| label_only | 30 | 93.626 | 0.991711 | 4.547988 | 0.030401 | 3.87% | 0.83% |
| label_only | 100 | 312.086 | 0.982147 | 4.175396 | 0.083975 | 10.18% | 1.79% |
| label_only | 300 | 936.258 | 0.442721 | 0.695862 | 0.353010 | 91.03% | 55.73% |
| label_only | 1000 | 3120.860 | 0.636557 | 2.893342 | 0.017297 | 38.07% | 36.34% |
| label_only | 3000 | 9362.580 | 0.007014 | 0.050794 | 0.005738 | 99.87% | 99.30% |
| quarter_data | 10 | 31.209 | 0.985547 | 4.178445 | 0.130955 | 14.54% | 1.45% |
| quarter_data | 30 | 93.626 | 0.976621 | 3.654136 | 0.241076 | 26.45% | 2.34% |
| quarter_data | 100 | 312.086 | 0.915197 | 1.063144 | 0.834414 | 91.92% | 8.48% |
| quarter_data | 300 | 936.258 | 0.883103 | 0.040224 | 0.883103 | 100.00% | 11.69% |
| quarter_data | 1000 | 3120.860 | 0.920723 | 0.041519 | 0.920723 | 100.00% | 7.93% |
| quarter_data | 3000 | 9362.580 | 0.952391 | 0.045270 | 0.952391 | 100.00% | 4.76% |
| shuffled_labels | 10 | 31.209 | 0.991498 | 4.479432 | 0.027000 | 3.55% | 0.85% |
| shuffled_labels | 30 | 93.626 | 0.988735 | 4.309101 | 0.041669 | 5.29% | 1.13% |
| shuffled_labels | 100 | 312.086 | 0.990011 | 3.926052 | 0.091839 | 10.18% | 1.00% |
| shuffled_labels | 300 | 936.258 | 0.990011 | 2.369382 | 0.377553 | 38.75% | 1.00% |
| shuffled_labels | 1000 | 3120.860 | 0.991711 | 0.334593 | 0.942816 | 95.11% | 0.83% |
| shuffled_labels | 3000 | 9362.580 | 0.989586 | 0.053069 | 0.978106 | 98.85% | 1.04% |

## Forecast Scores

| Forecaster | MAE held-out | Predicted best | Actual best | Top-1 correct |
| --- | --- | --- | --- | --- |
| cti_power_law | 0.226068 | quarter_data | label_only | False |
| b0_last_point | 0.215232 | quarter_data | label_only | False |
| b1_linear_log_compute | 0.228051 | quarter_data | label_only | False |
| b2_independent_power_law | 0.226068 | quarter_data | label_only | False |
| b3_proxy_only | 0.194245 | quarter_data | label_only | False |
| b4_random_intervention_ranking | 0.255520 | shuffled_labels | label_only | False |

B2 is identical to the CTI per-intervention power-law forecast on this single task board, so a strict beat-all-baselines verdict is impossible in this board unless the law is later made cross-task or shared-parameter.

## Intervention Shift Classification

| Intervention | alpha | D_inf | delta alpha vs label_only | Classification |
| --- | --- | --- | --- | --- |
| label_only | 0.004257 | 0.000000 | 0.000000 | reference |
| shuffled_labels | 5.000000 | 0.989370 | 4.995743 | exponent_shift |
| quarter_data | 0.031949 | 0.000000 | 0.027692 | indeterminate_small_shift |

## Grokking Check

Detected at step 300.

The label-only `D_func` checkpoints were:

```json
{
  "10": 0.991710945802338,
  "30": 0.991710945802338,
  "100": 0.9821466524973432,
  "300": 0.44272051009564295,
  "1000": 0.6365568544102019,
  "3000": 0.0070138150903293894
}
```

## Artifacts

- `tmp_work_loop_b16\cti1_board1_config.json`
- `tmp_work_loop_b16\cti1_board1_train_log.jsonl`
- `tmp_work_loop_b16\cti1_board1_checkpoints.csv`
- `tmp_work_loop_b16\cti1_board1_predictions.json`
- `tmp_work_loop_b16\cti1_board1_scores.csv`
- `tmp_work_loop_b16\cti1_board1_summary.json`
- `research/work_loop_batch16.md`
- `tmp_work_loop_b16\smoke\cti1_board1_smoke_summary.json`

## NARRATIVE SECTION

What happened: the board ran cleanly and produced the missing object from B15: functional distortion at every log-spaced compute point for all three interventions. The shuffled-label negative control measured memorization-only compute, while quarter-data tested whether fewer labels changed the curve.

Did the power law predict: by the strict precommit token, `PROXY_ONLY_LAW`. The CTI power-law MAE was 0.226068; the best forecaster was `b3_proxy_only` at 0.194245.

Did grokking break the form: Detected at step 300. If a sudden post-plateau jump appears in later boards, a broken power law or phase-transition form is the honest model.

Gossip-magazine story: the laptop tried to predict which tiny training idea was worth the electricity before it finished. This board is only a first lab test, not a manifesto result.

Does it survive "that's obvious?": the setup survives as a measurement because the forecast was locked before the held-out checkpoints existed. The claim does not get to outrun the score table.
