# WORK LOOP - Batch 12: Portfolio Probe

Date: 2026-07-07

## Grounding

Read first, in order:

1. `research/VISION.md`
2. `research/dual_loop_supervisor_checkin_10.md`
3. `research/dual_loop_supervisor_checkin_9.md`
4. `research/question_loop_batch15.md`
5. `research/work_loop_batch11.md`
6. `code/margin_distillation.py`
7. `code/coordinate_inheritance.py`
8. `tmp_margin_distillation_b11/smoke50/functional_margin_distillation_smoke.json`

Primary metric throughout: benchmark-facing forced-choice functional margins and MCQ accuracy, not training loss.

## Artifacts

- `tmp_margin_distillation_b12/scaffold_capacity_50/scaffold_capacity.json`
- `tmp_margin_distillation_b12/scaffold_capacity_50/scaffold_capacity_student.pt`
- `tmp_margin_distillation_b12/teacher_audit/teacher_margin_audit.json`
- `tmp_margin_distillation_b12/disagreement_audit_smol/disagreement_density_audit.json`
- `research/work_loop_batch12.md`

Implementation caveat: the requested persistent patch to `code/margin_distillation.py` was blocked by the Windows sandbox wrapper when using the patch tool. The probes were run with temporary `C:/tmp` harnesses that import the exact existing `MarginStudent`, `prepare_scoring_batch`, `evaluate_student_rankings`, `evaluate_teacher_rankings`, and bootstrap utilities from `code/margin_distillation.py` / `code/coordinate_inheritance.py`. This preserves the evaluation semantics but should be promoted into `code/margin_distillation.py` before any future reuse.

## Probe A: Scaffold Capacity Check

Question: Can the exact B11 byte-codec `MarginStudent` learn benchmark discrimination at all from supervised labels?

Setup:

- Architecture: exact `MarginStudent` from `code/margin_distillation.py`.
- Codec: `C:/sutra_fast/codec_phase1.5/codec_final.pt`.
- Training data: 288 train-safe examples, 96 each from HellaSwag, PIQA, ARC-Easy.
- Held-out eval: 144 disjoint train-safe examples, 48 each.
- Objective: supervised choice CE over student byte-NLL scores, `cross_entropy(-choice_nlls, gold_label)`.
- Definitive run: 50 gradient steps, batch 6 examples, CUDA, `max_bytes=768`.

A 30-step run was also executed inside the allowed 20-50 step range and produced only a marginal signal. Because this is the critical kill gate, the 50-step fair-chance run is the binding verdict.

| Benchmark | Untrained acc | Label-CE acc | Delta | Mean margin delta | Verdict contribution |
|---|---:|---:|---:|---:|---|
| HellaSwag | 20.83% | 20.83% | +0.00pp | -0.0932 | FAIL |
| PIQA | 56.25% | 54.17% | -2.08pp | +0.0059 | FAIL |
| ARC-Easy | 22.92% | 14.58% | -8.33pp | -0.1210 | FAIL |

Precommitted token:

```text
FAIL_SCAFFOLD
```

Interpretation: supervised labels did not improve accuracy by >=+5pp on >=2/3 benchmarks. It improved zero benchmarks at 50 steps and regressed PIQA/ARC-Easy. The bottleneck is the student/scaffold or this short-training interface, not merely the KD objective.

Immediate kill condition fired: Probe C is skipped.

## Probe B: Teacher-Margin Data Audit

Question: Is Qwen/Qwen3-0.6B a useful teacher for these benchmarks?

Setup:

- Teacher: `Qwen/Qwen3-0.6B`.
- Examples: 200 train-safe examples per benchmark.
- No training.
- Metrics: teacher accuracy, gold-vs-best-wrong margin, positive margin fraction, confident-wrong fraction, hard-negative length/position diagnostics.

| Benchmark | Qwen acc | Positive margin | Mean margin | Median margin | Confident wrong | Teacher verdict contribution |
|---|---:|---:|---:|---:|---:|---|
| HellaSwag | 49.5% | 49.5% | -0.0263 | -0.0193 | 49.5% | MARGINAL |
| PIQA | 67.5% | 67.5% | +0.1920 | +0.1255 | 28.5% | PASS |
| ARC-Easy | 34.0% | 34.0% | -1.2170 | -0.6826 | 65.5% | FAIL |

Hard-negative quality notes:

- HellaSwag hard wrong choices were not dominated by position or length: best-wrong shortest 24.5%, longest 23.5%.
- PIQA is binary, so position is balanced by construction; shortest/longest overlap is expected and not diagnostic.
- ARC-Easy is deeply teacher-poisoned: 65.5% confident-wrong by the 0.05 margin threshold.

Precommitted token:

```text
MARGINAL_TEACHER
```

Interpretation: Qwen is usable on PIQA, borderline on HellaSwag, and actively bad on ARC-Easy. This is not a full teacher kill because accuracy <45% happened on 1/3 benchmarks, not >=2/3. It is also not a pass because only PIQA clears the accuracy/positive-margin gate.

## Probe C: FMD_SHADOW_288

Status:

```text
SKIPPED_BY_FAIL_SCAFFOLD
```

Reason: Probe C was explicitly conditional on Probes A and B passing, and the hard kill says to skip Probe C if Probe A shows `FAIL_SCAFFOLD`. Probe A did fail. Running margin training after the exact student failed supervised label training would launder an objective experiment into an architecture failure.

Required baselines therefore remain unrun:

- Untrained baseline: not rerun under FMD_SHADOW because Probe C was killed.
- Label-only CE: run in Probe A and failed as scaffold capacity evidence.
- Random-margin target: not run because FMD_SHADOW was not legally opened.

Precommitted FMD verdict token is not assigned because the experiment was conditionally skipped, not executed.

## Probe D: Disagreement Density Audit

Question: Do teachers disagree enough to fuel a router?

Setup:

- Qwen predictions reused from Probe B on the same 200 examples per benchmark.
- Preferred second teacher attempt: `state-spaces/mamba-790m-hf`.
- Mamba loaded, but Transformers fell back to the slow sequential Mamba path because required kernels were unavailable. After 25/200 HellaSwag examples took too long, the Mamba run was stopped and marked infeasible for this batch.
- Feasible second teacher: `HuggingFaceTB/SmolLM2-360M`, scored sequentially to avoid simultaneous VRAM pressure.
- This is a no-training, label-anchored ceiling diagnostic, not a deployable gold-free router.

| Benchmark | Qwen acc | SmolLM2 acc | Top-1 disagreement | Useful disagreement | Oracle ceiling | Oracle gap over best |
|---|---:|---:|---:|---:|---:|---:|
| HellaSwag | 49.5% | 58.5% | 25.0% | 17.0% | 62.5% | +4.0pp |
| PIQA | 67.5% | 71.5% | 20.0% | 20.0% | 79.5% | +8.0pp |
| ARC-Easy | 34.0% | 54.5% | 49.0% | 34.5% | 61.5% | +7.0pp |
| Aggregate | 50.33% | 61.50% | 31.33% | 23.83% | 67.83% | +6.33pp |

Precommitted token:

```text
PASS_DISAGREEMENT
```

Interpretation: there is enough label-anchored disagreement fuel in the Qwen/SmolLM2 pair: useful disagreement is 23.83% aggregate and oracle ceiling beats the best single teacher by +6.33pp. This does not rescue FMD_SHADOW because the student scaffold failed, and it does not yet prove a router because the audit uses labels to define useful disagreement.

## Batch Verdict

| Probe | Token | Action |
|---|---|---|
| A. Scaffold capacity | `FAIL_SCAFFOLD` | Student/scaffold bottleneck. Hard-kill Probe C. |
| B. Teacher audit | `MARGINAL_TEACHER` | Qwen is mixed: PIQA usable, HellaSwag borderline, ARC poisoned. |
| C. FMD_SHADOW_288 | `SKIPPED_BY_FAIL_SCAFFOLD` | Not legally opened. No FMD repair shot consumed. |
| D. Disagreement density | `PASS_DISAGREEMENT` | Router fuel exists with feasible SmolLM2 fallback; Mamba deferred due missing kernels. |

The actionable conclusion is not "try a cleverer FMD loss." The exact byte-codec `MarginStudent` failed even direct supervised benchmark labels at the fair 50-step budget. That means more teacher-margin shaping on this scaffold is not justified. The only positive signal in the batch is teacher disagreement density, but a router needs a student that can actually absorb corrections.

## Next Obligations

1. Treat the B11 `MarginStudent` architecture as the immediate bottleneck for short-budget benchmark discrimination.
2. Do not run FMD_SHADOW_288 on this scaffold unless the supervisor explicitly overrides the hard kill.
3. If the disagreement-router direction continues, first pair it with a stronger or pretrained student path, then rerun scaffold capacity before router training.
4. Promote the temporary B12 probe harness into `code/margin_distillation.py` before reuse; the executed semantics are valid, but the persistent-code requirement remains a cleanup obligation.

## NARRATIVE SECTION

One-sentence story: We ran the portfolio probe after the FMD prototype kill, and the decisive result is that the byte-codec `MarginStudent` could not reliably learn held-out MCQ discrimination even from supervised labels.

Does it survive "isn't that obvious?": Yes. The supervised check is stronger than another KD loss ablation: it shows the failure is below the teacher-objective layer. If labels cannot move the scaffold, teacher margins are not the first bottleneck.

Does it survive "so what?": Yes as a kill gate. It prevents the project from spending the one FMD repair shot on a student architecture that failed the simpler task. The useful-disagreement pass is valuable, but only as a future routing resource after the student bottleneck is fixed.

If boring, say so: The negative scaffold result is boring as moonshot evidence and important as triage. The only live excitement is the disagreement density, but without a trainable student path it is fuel without an engine.
