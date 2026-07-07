# W-Loop B15: CTI-0 Eklavya Salvage Measurement

**Date:** 2026-07-07
**Verdict token:** `CTI_SALVAGE_INFORMATIVE`
**Scope:** artifact-only CPU analysis of B14 results; no model loading, no dataset loading, no GPU experiment.

---

## Grounding Files Read First

Read in the requested order:

1. `research/VISION.md`
2. `research/dual_loop_supervisor_checkin_13.md`
3. `research/CTI_PRECOMMIT_SPEC.md`
4. `research/work_loop_batch14.md`
5. `research/question_loop_batch19.md`, with iterations 131-132 pulled explicitly after the first full read clipped in terminal output.

The binding interpretation is simple: Eklavya is dead as a mechanism, but B14 is useful salvage data because it contains multiple interventions with proxy trajectories and final held-out function measurements. CTI only earns the name if it predicts held-out functional distortion, not if it decorates training-loss curves after the fact.

---

## Artifacts Produced

| Artifact | Purpose |
|---|---|
| `code/cti_salvage.py` | Reusable CTI-0 extraction/fitting CLI for B14-style artifacts. |
| `tmp_work_loop_b15/cti0_b14_compute_proxy_curves.csv` | 80-row compute/proxy table: 5 conditions x 16 logged steps. |
| `tmp_work_loop_b15/cti0_b14_final_functional_distortion.csv` | Final held-out `D_func` table by condition. |
| `tmp_work_loop_b15/cti0_b14_proxy_powerlaw_fits.csv` | Power-law fit parameters for `D_proxy(C)`. |
| `tmp_work_loop_b15/cti0_salvage_summary.json` | Machine-readable summary, rankings, diagnostics, limitations, artifact paths. |
| `tmp_work_loop_b15/cti0_proxy_loss_powerlaw_fits.png` | Training-loss proxy curves with fitted power laws. |
| `tmp_work_loop_b15/cti0_functional_distortion_by_condition.png` | Final `D_func` points by condition. |
| `tmp_work_loop_b15/cti0_step30_proxy_vs_final_dfunc.png` | Early proxy vs final functional distortion diagnostic. |

PNG files were programmatically loaded after generation and had nonblank pixel ranges. The in-app image viewer was blocked by the Windows sandbox wrapper, so visual QA was limited to file existence and pixel-array validation.

---

## Method

Input artifact:

```text
tmp_work_loop_b14/smollm2_mechanism_control.json
```

No model was loaded. Therefore the `local_files_only=True` constraint was trivially satisfied: this batch did not call `from_pretrained` or touch model weights.

Compute accounting followed the user-specified salvage estimate:

```text
C = 6 * trainable_params * batch_size * step
```

For all five B14 conditions:

| Quantity | Value |
|---|---:|
| LoRA trainable params | 1,843,200 |
| Batch examples | 12 |
| FLOPs per step | 132,710,400 |
| Step 30 compute | 3,981,312,000 FLOPs |
| Step 150 compute | 19,906,560,000 FLOPs |

Important caveat: this is the precommitted salvage estimate over trainable LoRA parameters. It is consistent across conditions and useful for relative CTI-0 diagnostics, but it under-specifies the full frozen-backbone forward/backward activation cost of LoRA fine-tuning.

Distortion definitions used here:

| Symbol | Measurement |
|---|---|
| `D_proxy` | Logged training loss at steps 1, 10, 20, ..., 150. |
| `D_func` | `1 - mean_accuracy` over final held-out HellaSwag, PIQA, ARC-Easy. |

B14 does not contain intermediate held-out evaluations. Therefore `D_func(C)` cannot be fit as a curve in CTI-0.

---

## Phase 1: Extracted Compute-Distortion Data

Final functional distortion:

| Condition | Mean acc | `D_func` | Final proxy loss | Final batch acc | HellaSwag | PIQA | ARC-Easy |
|---|---:|---:|---:|---:|---:|---:|---:|
| label_only | 0.548611 | 0.451389 | 0.319695 | 1.000 | 0.4167 | 0.6875 | 0.5417 |
| single_teacher | 0.590278 | 0.409722 | 0.372057 | 0.833 | 0.3958 | 0.7500 | 0.6250 |
| oracle | 0.583333 | 0.416667 | 0.371479 | 0.917 | 0.4167 | 0.7500 | 0.5833 |
| non_oracle | 0.569444 | 0.430556 | 0.361885 | 1.000 | 0.4375 | 0.7083 | 0.5625 |
| random | 0.583333 | 0.416667 | 0.382852 | 0.833 | 0.3958 | 0.7500 | 0.6042 |

The B14 result already shows the core CTI warning sign: label-only has the best final proxy loss and perfect final batch accuracy, but the worst held-out functional distortion.

---

## Phase 2: Proxy Power-Law Fits

Fitted form:

```text
D_proxy(C) = D_inf + k * C^(-alpha)
```

For numerical stability the script fits on normalized compute `C / C_final` and writes both normalized `k` and literal `k_flops` to the CSV. The table below reports normalized `k` because it is easier to read across conditions.

| Condition | `D_inf` | `k_norm` | `alpha` | RMSE | R2 |
|---|---:|---:|---:|---:|---:|
| label_only | 4.602e-23 | 0.623261 | 0.149941 | 0.286799 | 0.299723 |
| single_teacher | 1.072e-17 | 0.404232 | 0.029683 | 0.097387 | 0.030579 |
| oracle | 6.778e-23 | 0.379783 | 0.129445 | 0.078824 | 0.554760 |
| non_oracle | 1.772e-15 | 0.421201 | 0.026092 | 0.058432 | 0.063232 |
| random | 8.941e-15 | 0.413326 | 0.095491 | 0.082074 | 0.383205 |

Alpha range: `0.0260919` to `0.1499413`, range `0.1238494`.

Interpretation:

- The proxy trajectories do not look identical. Label-only and oracle produce steeper proxy-loss exponents than single-teacher and non-oracle.
- The fit quality is weak for several conditions because B14 logged sparse, noisy batch losses rather than dense held-out losses. The near-zero `D_inf` values are extrapolation artifacts of short-window monotone power-law fitting, not credible asymptotic claims.
- The proxy fit is still useful as a diagnostic: interventions change the apparent proxy-loss geometry, but that geometry does not cleanly map to held-out function.

---

## Phase 3: Functional Distortion Diagnostic

B14 has only one final `D_func` point per condition, so this is not a CTI curve. It is a ranking diagnostic.

Early proxy ranking at step 30, lower is better:

```text
non_oracle < random < oracle < single_teacher < label_only
```

Final functional distortion ranking, lower is better:

```text
single_teacher < oracle = random < non_oracle < label_only
```

Step-30 proxy ranking does **not** predict the final functional winner. It picks `non_oracle`; final held-out function picks `single_teacher`.

Rank statistics:

| Diagnostic | Value |
|---|---:|
| Step-30 proxy vs final `D_func` Spearman rho | 0.0513 |
| Step-30 proxy vs final `D_func` Kendall tau | -0.1054 |
| Step-30 pairwise ranking accuracy, ties excluded | 4/9 = 0.4444 |
| Final proxy vs final `D_func` Spearman rho | -0.8208 |
| Final proxy pairwise ranking accuracy, ties excluded | 1/9 = 0.1111 |

The final-proxy result is the sharpest salvage signal: by the end of training, lower training loss is anti-aligned with held-out function in this tiny five-condition board. That is exactly the proxy-function divergence CTI is supposed to handle.

---

## CTI-0 Verdict

`CTI_SALVAGE_INFORMATIVE`

Reason:

1. B14 provides extractable compute/proxy trajectories for all five interventions.
2. Apparent proxy-law parameters differ across interventions, especially in `alpha`.
3. Proxy-function divergence is visible: label-only optimizes proxy best and function worst; single-teacher is function best without being proxy best.

What this verdict does **not** mean:

- It is not `PASS_CTI_LAW_0`.
- It does not prove a predictive law.
- It does not fit `D_func(C)` because intermediate held-out functional evaluations do not exist.
- It does not rescue Eklavya as a mechanism.

---

## Limitations Of The Salvage Data

1. **No intermediate `D_func`.** B14 only evaluated held-out MCQ accuracy at the final checkpoint. CTI requires log-spaced held-out functional evaluations.
2. **Sparse proxy logs.** Each condition has only 16 logged batch-loss points, and those points are noisy minibatch measurements.
3. **Single seed.** There is no variance estimate over seeds for proxy-law parameters or final rankings.
4. **Small held-out slices.** Final function uses n=48 per benchmark on train-safe held-out splits, not public validation benchmarks.
5. **Equal final compute only.** All interventions end at 150 steps, so final `D_func` comparisons are five same-budget points rather than compute curves.
6. **LoRA compute estimate is rough.** The salvage formula uses trainable params as requested; full hardware FLOPs would require deeper instrumentation.
7. **Post-hoc fitting.** The proxy fits were performed after B14 completed. They can inform CTI-1, but they cannot count as a precommitted forecast.

---

## Phase 4: CTI-1 Real Experiment Design

CTI-1 must collect the data B14 lacks: `D_func` at every log-spaced compute checkpoint.

### Core Question

Can early compute-distortion measurements predict later held-out functional distortion and intervention ranking better than naive baselines?

### Models

| Model birth | Concrete model | Purpose |
|---|---|---|
| Random tiny transformer | Approximately 1M parameters, trained from scratch | Clean local compute-distortion lab without pretrained history. |
| SmolLM2-135M LoRA | Cached `HuggingFaceTB/SmolLM2-135M`, loaded with `local_files_only=True` | Pretrained small-LM adaptation board continuous with B14. |

Optional later extension after CTI-1 gate: SmolLM2-360M LoRA or a second random transformer width for cross-size transfer. Do not add this until the first board produces clean curves.

### Tasks

| Task family | Concrete task | `D_func` |
|---|---|---|
| Modular arithmetic | Addition mod p, e.g. p=97 or p=113, fixed train/held split over pairs `(a,b)` | `1 - held_out_accuracy` on exact answer. |
| MCQ forced choice | HellaSwag, PIQA, ARC-Easy using existing B14 scoring path | `1 - mean held_out_accuracy` over benchmarks. |

The modular task is the low-noise curve laboratory. The MCQ task is the Eklavya-continuity check where proxy-function divergence already appeared.

### Checkpoints

Evaluate every intervention at exactly these update steps:

```text
10, 30, 100, 300, 1000, 3000
```

At each checkpoint record:

| Metric | Required? | Notes |
|---|---|---|
| `C` cumulative FLOPs | Primary x-axis | Use consistent formula per model family; also log wall-clock and peak VRAM. |
| `D_func` | Primary y-axis | Held-out exact accuracy for modular arithmetic; held-out MCQ mean accuracy for MCQ. |
| `D_proxy` | Diagnostic | Training loss on current minibatches plus a fixed train-proxy slice. |
| `D_gap` | Diagnostic | `abs(train_accuracy - held_out_accuracy)`. |
| `D_margin` | Diagnostic | MCQ gold-vs-best-wrong margin; modular logit margin if available. |
| `D_cal` | Diagnostic | ECE where probability outputs are meaningful. |

### Interventions

Keep CTI-1 small enough to finish and adversarially interpretable.

| Board | Interventions |
|---|---|
| Random transformer / modular arithmetic | label-only CE, shuffled-label control, 25% vs 100% data. |
| SmolLM2-135M LoRA / MCQ | label-only CE, single-teacher KD, non-oracle routed KD if teacher cache is reused. |
| SmolLM2-135M LoRA / modular arithmetic | label-only CE and shuffled-label control, to separate pretrained-language birth from algorithmic task geometry. |

If budget is tight, prioritize:

1. random 1M transformer on modular arithmetic: label-only vs shuffled-label;
2. SmolLM2-135M LoRA on MCQ: label-only vs single-teacher;
3. add non-oracle routed KD only after the first two produce valid checkpoint curves.

### Prediction Protocol

Fit only on early points:

```text
seen checkpoints: 10, 30, 100
held-out checkpoints: 300, 1000, 3000
```

For every model/task/intervention:

1. Fit `D_func(C) = D_inf + k*C^(-alpha)` on checkpoints 10, 30, 100.
2. Predict `D_func` at 300, 1000, 3000 before reading those values.
3. Predict intervention ranking at step 3000 from data through step 100.
4. Compare against baseline forecasters from `CTI_PRECOMMIT_SPEC.md`: last-point, linear log-C extrapolation, per-task independent power law, proxy-only forecast, random intervention ranking.

### CTI-1 Output Contract

Write all generated experiment artifacts into `tmp_work_loop_b16/` or the next loop's declared temp directory:

```text
cti1_config.json
cti1_train_log.jsonl
cti1_eval_checkpoints.csv
cti1_fit_phase_predictions.json
cti1_forecaster_scores.csv
cti1_summary.md
```

Every checkpoint row must include enough metadata to audit leakage:

```text
model_birth, model_name, task_family, task_id, intervention, seed,
checkpoint_step, cumulative_flops, trainable_params, batch_size,
train_examples_seen, eval_split_id, d_proxy, d_func, d_gap, d_margin, d_cal
```

### CTI-1 Kill Criteria

Use the precommit spec tokens, not new language:

| Token | CTI-1 trigger |
|---|---|
| `INVALID_CTI` | Any leak, missing checkpoint evals, inconsistent compute accounting, or accidental nonlocal model loading. |
| `NO_PREDICTIVE_LAW` | CTI forecaster fails to beat all baselines on held-out checkpoints in >=2/3 task boards. |
| `PROXY_ONLY_LAW` | Proxy curves fit but `D_func` forecasts/rankings fail. |
| `PASS_CTI_LAW_0` | `D_func` predictions beat all baselines on >=2/3 task boards and classify at least one intervention as exponent-shift vs constant-shift before full budget. |

---

## Implementation Notes For Next Worker

Run the salvage analysis with:

```bash
python code/cti_salvage.py --input tmp_work_loop_b14/smollm2_mechanism_control.json --output-dir tmp_work_loop_b15
```

The B14 MCQ harness already has the relevant scoring path in `code/smollm2_mechanism_control.py`, and the lower-level choice scoring utilities remain in `code/coordinate_inheritance.py`. CTI-1 should not reuse B14's final-only eval structure; it should move evaluation into the training loop at the checkpoint schedule.

For SmolLM2 model loading in CTI-1, every call must use cached-only loading, for example:

```python
AutoTokenizer.from_pretrained(name, local_files_only=True)
AutoModelForCausalLM.from_pretrained(name, local_files_only=True)
```

Do not run GPU-heavy CTI-1 work until a separate batch explicitly authorizes it.

---

## NARRATIVE SECTION

**Gossip-magazine headline:** The failed Eklavya experiment left behind a useful clue: the run that learned the training batch best generalized worst.

**Survives "isn't that obvious?":** Yes, but only as a diagnostic. Everyone expects overfitting to happen; the useful part is that the same B14 board now gives CTI an exact measurement target: predict when compute improves function instead of merely improving proxy loss.

**Survives "that's trivial?":** Not yet as a law. CTI-0 is not a breakthrough. It is a clean salvage measurement showing why CTI-1 must evaluate held-out function at every compute checkpoint.

**The honest narrative:** B14 does not contain enough data to prove CTI. It does contain enough data to justify the pivot's first real experiment. Proxy-loss geometry differs by intervention, and proxy/function divergence is visible. The next hostile-reviewer-proof move is not more theory language; it is a log-spaced checkpoint experiment that forecasts held-out functional distortion before the full run is visible.
