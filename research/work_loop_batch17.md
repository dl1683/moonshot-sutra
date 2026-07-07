# W-Loop B17: CTI-1 Board 2 - SmolLM2-135M LoRA MCQ

**Date:** 2026-07-07
**Verdict token:** `PROXY_ONLY_LAW`
**Task:** HellaSwag, PIQA, ARC-Easy forced-choice MCQ
**Model:** SmolLM2-135M with rank-16 LoRA, 1,843,200 trainable / 136,358,208 total parameters
**Device:** NVIDIA GeForce RTX 5090 Laptop GPU

---

## Grounding

Read first, in order:

1. `research/VISION.md`
2. `research/CTI_PRECOMMIT_SPEC.md`
3. `research/work_loop_batch16.md`
4. `research/work_loop_batch15.md`
5. `research/question_loop_batch21.md`

Binding interpretation: Board 2 is the monotone adaptation check after Board 1's grokking confound. The primary measurement is `D_func = 1 - mean_accuracy` over HellaSwag, PIQA, and ARC-Easy at every log-spaced checkpoint. The train/eval split is reconstructed from B14 with split seed `20260707`; the training seed for this board is `42`.

## Smoke Run

Completed before the full run. The smoke path loaded `HuggingFaceTB/SmolLM2-135M` with `local_files_only=True`, trained `label_only` for 10 steps on cuda, recorded checkpoint step 10, and ran MCQ evaluation at step 10.

Smoke artifact: `tmp_work_loop_b17\smoke\cti1_board2_smoke_summary.json`.

## Configuration

| Parameter | Value |
|---|---:|
| Train examples | 288 |
| Held-out examples | 144 |
| Batch size | 12 |
| Learning rate | 0.0002 |
| Weight decay | 0.01 |
| Max steps | 3000 |
| Checkpoints | 10, 30, 100, 300, 1000, 3000 |
| Fit-only checkpoints | 10, 30, 100 |
| Held-out forecast checkpoints | 300, 1000, 3000 |
| Primary compute formula | `cumulative_flops = 6 * total_parameters_with_lora * batch_examples * checkpoint_step` |

## Teacher Cache

Teacher cache source: `reused_b14_teacher_choice_cache`. Cache/split validation: `True`.

## Prediction Lock

Predictions were written after all three interventions reached step 100 and before any training or evaluation at steps 300, 1000, or 3000. The lock record is in `tmp_work_loop_b17/cti1_board2_predictions.json`.

CTI predicted step-3000 ranking:

```text
label_only < single_teacher < shuffled_labels
```

Actual step-3000 ranking:

```text
single_teacher < label_only < shuffled_labels
```

## Checkpoint Matrix

| Intervention | Step | GFLOPs | D_func | D_proxy | D_gap | Train Obj Acc | Train True Acc | Held-out Acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| label_only | 10 | 98.178 | 0.444444 | 1.045384 | 0.000000 | 55.56% | 55.56% | 55.56% |
| label_only | 30 | 294.534 | 0.430556 | 0.933187 | 0.065972 | 63.54% | 63.54% | 56.94% |
| label_only | 100 | 981.779 | 0.437500 | 0.571052 | 0.274306 | 83.68% | 83.68% | 56.25% |
| label_only | 300 | 2945.337 | 0.520833 | 0.033622 | 0.520833 | 100.00% | 100.00% | 47.92% |
| label_only | 1000 | 9817.791 | 0.541667 | 0.000798 | 0.541667 | 100.00% | 100.00% | 45.83% |
| label_only | 3000 | 29453.373 | 0.569444 | 0.000100 | 0.569444 | 100.00% | 100.00% | 43.06% |
| shuffled_labels | 10 | 98.178 | 0.444444 | 1.356179 | 0.173611 | 38.19% | 55.56% | 55.56% |
| shuffled_labels | 30 | 294.534 | 0.444444 | 1.264760 | 0.149306 | 40.62% | 54.17% | 55.56% |
| shuffled_labels | 100 | 981.779 | 0.479167 | 0.944435 | 0.048611 | 56.94% | 47.22% | 52.08% |
| shuffled_labels | 300 | 2945.337 | 0.569444 | 0.106023 | 0.559028 | 98.96% | 31.25% | 43.06% |
| shuffled_labels | 1000 | 9817.791 | 0.611111 | 0.000767 | 0.611111 | 100.00% | 31.94% | 38.89% |
| shuffled_labels | 3000 | 29453.373 | 0.611111 | 0.000124 | 0.611111 | 100.00% | 31.94% | 38.89% |
| single_teacher | 10 | 98.178 | 0.451389 | 0.544982 | 0.006944 | 55.56% | 55.56% | 54.86% |
| single_teacher | 30 | 294.534 | 0.437500 | 0.500516 | 0.055556 | 61.81% | 61.81% | 56.25% |
| single_teacher | 100 | 981.779 | 0.437500 | 0.391773 | 0.232639 | 79.51% | 79.51% | 56.25% |
| single_teacher | 300 | 2945.337 | 0.465278 | 0.282521 | 0.454861 | 98.96% | 98.96% | 53.47% |
| single_teacher | 1000 | 9817.791 | 0.437500 | 0.270035 | 0.437500 | 100.00% | 100.00% | 56.25% |
| single_teacher | 3000 | 29453.373 | 0.437500 | 0.268458 | 0.434028 | 99.65% | 99.65% | 56.25% |

## Forecast Scores

| Forecaster | MAE held-out | Predicted best | Actual best | Top-1 correct |
| --- | --- | --- | --- | --- |
| cti_power_law | 0.086820 | label_only | single_teacher | False |
| b0_last_point | 0.077932 | label_only | single_teacher | False |
| b1_linear_log_compute | 0.076674 | single_teacher | single_teacher | True |
| b2_per_benchmark_power_law | 0.088291 | label_only | single_teacher | False |
| b3_proxy_only | 0.075236 | single_teacher | single_teacher | True |
| b4_random_intervention_ranking | 0.077932 | label_only | single_teacher | False |

## Intervention Shift Classification

| Intervention | alpha | D_inf | delta alpha vs label_only | Classification |
| --- | --- | --- | --- | --- |
| label_only | 5.000000 | 0.434013 | 0.000000 | reference |
| single_teacher | 4.999970 | 0.437471 | -0.000030 | indeterminate_small_shift |
| shuffled_labels | 0.000001 | 0.456019 | -4.999999 | exponent_shift |

## Monotone Check

| Intervention | Monotone nonincreasing D_func | D_func drop 10->3000 | Regression steps |
| --- | --- | --- | --- |
| label_only | False | -0.125000 | 100, 300, 1000, 3000 |
| single_teacher | False | 0.013889 | 300 |
| shuffled_labels | False | -0.166667 | 100, 300, 1000 |

## Artifacts

- `tmp_work_loop_b17\cti1_board2_config.json`
- `tmp_work_loop_b17\cti1_board2_train_log.jsonl`
- `tmp_work_loop_b17\cti1_board2_checkpoints.csv`
- `tmp_work_loop_b17\cti1_board2_predictions.json`
- `tmp_work_loop_b17\cti1_board2_scores.csv`
- `tmp_work_loop_b17\cti1_board2_summary.json`
- `research/work_loop_batch17.md`
- `tmp_work_loop_b17\smoke\cti1_board2_smoke_summary.json`

## NARRATIVE SECTION

Does CTI predict on a monotone task: by the strict precommit token, `PROXY_ONLY_LAW`. The CTI aggregate power-law MAE was 0.086820; the best forecaster was `b3_proxy_only` at 0.075236.

If yes, grokking was the confound. If no, CTI is dead or at least not alive on the natural monotone adaptation domain this board was designed to rescue.

Honest gossip-magazine story: the laptop got the clean monotone rematch after grokking spoiled the first board. It saw only the first three checkpoints, locked its prediction, then had to call the final intervention ranking before the late values were opened. The score table is the story; no thermodynamics language gets to outrun it.
