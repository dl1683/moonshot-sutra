# WORK LOOP - Batch 11: Functional-Margin Distillation Prototype

Date: 2026-07-07

Iterations: 101-110 requested. Executed 101-105 through the precommitted smoke. Iterations 106-110 were blocked by the hard kill condition.

## Artifacts

- `code/margin_distillation.py`
- `tmp_margin_distillation_b11/plumbing/functional_margin_distillation_smoke.json`
- `tmp_margin_distillation_b11/smoke50/functional_margin_distillation_smoke.json`
- `tmp_margin_distillation_b11/smoke50/margin_student.pt`
- `research/work_loop_batch11.md`

The `tmp_margin_distillation_b11/` outputs are local ignored scratch artifacts. The canonical committed implementation is `code/margin_distillation.py`.

## Iteration 101: Survey

Standard KD starts with Hinton, Vinyals, and Dean (2015): train a smaller model to match teacher soft targets, usually via temperature-softened probability distributions. That transfers class-relative dark knowledge better than hard labels, but its native object is still a probability vector, not an explicit decision boundary. Source: https://arxiv.org/abs/1503.02531

Relational KD (Park et al., 2019) moves away from per-example mimicry and transfers distances/angles among examples. That matters for this pivot because it says the object being distilled can be a relationship geometry, not hidden coordinates. Source: https://arxiv.org/abs/1904.05068

Contrastive representation distillation (Tian, Krishnan, Isola, 2019) argues that KL over outputs misses structural information and uses contrastive objectives to preserve teacher representation relations. It supports the general idea that relative comparisons can carry more transferable signal than pointwise logits. Source: https://arxiv.org/abs/1910.10699

The closest decision-boundary precedent is ranking distillation. In neural ranking, Margin-MSE distillation trains students to match teacher score differences for query/document pairs rather than raw scores, explicitly adapting across architectures whose score magnitudes differ. Source: https://arxiv.org/abs/2010.02666

For LLMs, PLaD uses pseudo-preference pairs and a ranking loss so the student learns relative output quality rather than only imitating teacher likelihoods. That is the direct black-box behavioral-distillation analogue for this batch. Source: https://arxiv.org/abs/2406.02886

Project-local notes add the same warning: `research/RESEARCH_NOTES.md` says "behavioral losses transfer behavior" and logs ranking/preference KD as the right lineage; `research/FIELD_SURVEY_JUNE2026.md` says the live competitive bar is functional benchmark transfer, especially against Wide7 and CBD, not internal NLL lift.

Working conclusion:

```text
Soft-label KD transfers softened class preferences.
RKD/contrastive KD transfer relational structure.
Margin/ranking KD transfers score differences and decision-boundary shape.
This pivot should train on teacher continuation NLL differences and judge only by forced-choice margins.
```

## Iteration 102: Loss Design

The implemented loss uses unlabeled text, not labeled MCQ data.

For each sampled shard context, the script builds four candidate continuations: one actual following span and three natural spans from other shard locations. Qwen scores each candidate by continuation NLL/token. The student scores the same candidate by byte NLL/byte through a frozen codec plus trainable tiny byte-autoregressive core.

For every candidate pair `(i, j)`:

```text
teacher_margin_ij = teacher_nll_j - teacher_nll_i
student_margin_ij = student_byte_nll_j - student_byte_nll_i
```

Positive margin means candidate `i` is preferred over candidate `j`. The loss is:

```text
L = weighted_ranknet(student_margin, sign(teacher_margin))
    + lambda * smooth_l1(student_margin, clipped_teacher_margin)
```

This is deliberately margin-facing:

- lower student score means preferred continuation;
- pairwise rank loss enforces ordering;
- SmoothL1 tries to preserve teacher margin size;
- tiny teacher gaps are filtered or down-weighted;
- no benchmark labels enter training.

## Iteration 103: Implementation

Created `code/margin_distillation.py`.

Key implementation pieces:

1. `MarginStudent`: frozen `codec_phase1.5` encoder, trainable input projection, tiny causal global reasoner, and byte decoder.
2. `make_unlabeled_candidate_sets`: builds unlabeled candidate groups from `C:/sutra_fast/data/shards_diverse/`.
3. `attach_teacher_targets`: uses `Qwen/Qwen3-0.6B` continuation NLLs as black-box teacher margin targets.
4. `pairwise_margin_loss`: RankNet-style pairwise ranking plus margin regression.
5. `evaluate_student_rankings`: forced-choice HellaSwag/PIQA/ARC-Easy scoring with the same prediction record semantics as `coordinate_inheritance.py`.
6. `margin_smoke_verdict`: precommitted PASS/FAIL/MARGINAL token logic.

Design limitation: this is a tiny margin prototype, not a full 121M S0 fine-tune. I chose it because the batch requested a 5-10 step smoke, and a full S0 run would mostly test optimizer overhead rather than whether a margin objective has any immediate functional signal.

## Commands Run

```powershell
python -m py_compile code\margin_distillation.py
```

Plumbing smoke:

```powershell
python code\margin_distillation.py --mode smoke --codec-checkpoint C:/sutra_fast/codec_phase1.5/codec_final.pt --data-dir C:/sutra_fast/data/shards_diverse --output-dir tmp_margin_distillation_b11/plumbing --device cuda --train-examples 2 --train-steps 1 --train-batch-examples 1 --benchmark-examples 1 --bootstrap-samples 5 --max-bytes 512 --context-chars 160 --continuation-chars 48 --train-seq-len 1024 --progress
```

Precommitted smoke:

```powershell
python code\margin_distillation.py --mode smoke --codec-checkpoint C:/sutra_fast/codec_phase1.5/codec_final.pt --data-dir C:/sutra_fast/data/shards_diverse --output-dir tmp_margin_distillation_b11/smoke50 --device cuda --train-examples 50 --train-steps 10 --train-batch-examples 4 --benchmark-examples 50 --benchmark-split train --bootstrap-samples 200 --max-bytes 768 --context-chars 320 --continuation-chars 96 --train-seq-len 2048 --progress --save-predictions
```

## Precommitted Verdict Tokens

```text
PASS_MARGIN_SMOKE - margin-trained student shows >=+3pp functional margin improvement over untrained baseline on >=2 of 3 benchmarks
FAIL_MARGIN_SMOKE - <+3pp improvement or regression
MARGINAL_MARGIN - +1-3pp, ambiguous, needs more data
```

Hard kill condition:

```text
If the margin distillation prototype shows no improvement (<=0pp) over untrained baseline on functional margins across all 3 benchmarks after reasonable training (>=50 examples, >=10 steps): write batch11.md with verdict FAIL_MARGIN_PROTOTYPE and analysis of why. Do NOT invest in scaling a flat signal.
```

## Iteration 104: Smoke Results

Smoke setup:

| Field | Value |
|---|---:|
| Teacher-margin training examples | 50 |
| Candidate continuations per example | 4 |
| Pairwise comparisons per step | 24 |
| Gradient steps | 10 |
| Eval examples per benchmark | 50 |
| Eval split | train-safe |
| Device | CUDA |
| Runtime | 51.0s |

Training loss did move:

| Step | Loss | Rank loss mean | Margin loss mean | Grad norm |
|---:|---:|---:|---:|---:|
| 1 | 1.1920 | 0.9606 | 0.9257 | 3.865 |
| 5 | 0.6572 | 0.5186 | 0.5545 | 2.581 |
| 10 | 0.5628 | 0.4564 | 0.4257 | 2.043 |

So the optimizer was not dead. The model learned something about the shard-derived teacher ranking task.

Functional margin result:

| Benchmark | Qwen teacher | Baseline untrained | Margin-trained | Delta | Margin delta |
|---|---:|---:|---:|---:|---:|
| HellaSwag | 38.0% | 34.0% | 22.0% | **-12.0pp** | -0.0307 |
| PIQA | 76.0% | 52.0% | 52.0% | **0.0pp** | -0.0052 |
| ARC-Easy | 36.0% | 30.0% | 30.0% | **0.0pp** | -0.0061 |

Bootstrap deltas:

| Benchmark | Accuracy delta CI95 | Margin delta CI95 |
|---|---:|---:|
| HellaSwag | [-24.0pp, -1.95pp] | [-0.0461, -0.0117] |
| PIQA | [-14.0pp, +16.0pp] | [-0.0206, +0.0071] |
| ARC-Easy | [-10.0pp, +10.0pp] | [-0.0522, +0.0250] |

Precommitted smoke verdict:

```text
FAIL_MARGIN_SMOKE
```

Hard-kill verdict:

```text
FAIL_MARGIN_PROTOTYPE
```

The hard kill fired because the margin-trained student improved by <=0pp on all three benchmarks after 50 examples and 10 steps.

## Iteration 105: Analysis

The failure is not an infrastructure failure:

- Qwen and codec loaded offline.
- Teacher target generation produced 150/150 candidate sets above the minimum spread before truncation to 50 training examples.
- The pairwise loss had 24 comparisons per step and decreased substantially.
- Student benchmark scoring produced paired baseline/trained records and bootstrap deltas.

The failure is functional:

```text
The training objective moved on unlabeled continuation rankings, but the learned signal did not improve gold-vs-wrong benchmark discrimination.
```

Most likely causes:

1. Data mismatch: shard-derived natural-continuation ranking teaches local continuation fit. HellaSwag/PIQA/ARC require answer-choice discrimination under benchmark prompts. Teacher NLL over arbitrary text continuations may be too weakly coupled to those benchmark margins.
2. Architecture mismatch: the tiny randomly initialized student has no meaningful prior. Ten steps can fit superficial continuation preferences but not induce benchmark-facing decision structure. This is a prototype limitation, but the hard gate was designed exactly to avoid scaling a flat signal.
3. Objective scale mismatch: Qwen token NLL/token margins and student byte NLL/byte margins are not naturally calibrated. The SmoothL1 term may shape score magnitude without preserving the functional boundary that matters.
4. Teacher weakness on two eval slices: Qwen3-0.6B scored only 38% on this HellaSwag sample and 36% on this ARC-Easy sample. A single weak teacher's margins are not reliable enough to define the target geometry there.
5. Baseline prior contamination: the untrained byte scorer already has length/style biases. Margin training may have strengthened corpus-continuation priors that actively hurt HellaSwag answer endings.

The result blocks the easy positive story:

```text
Training on teacher continuation margins is not automatically enough to produce benchmark-facing margins.
```

## Iterations 106-110: Blocked By Hard Kill

The conditional instruction said to try listwise, contrastive, or disagreement-weighted repairs if the smoke failed. The hard kill condition is stricter and fired exactly:

```text
<=0pp improvement across all 3 benchmarks after >=50 examples and >=10 steps.
```

Therefore I did not run scaling and did not spend another training run on a repair variant. That is the correct kill discipline for this batch.

Repair candidates if the user explicitly reopens the direction:

1. Listwise Plackett-Luce over benchmark-style unlabeled candidate sets, not arbitrary continuation snippets.
2. Teacher-disagreement weighted margins: only train examples where two or more teachers disagree and the target teacher is confident.
3. Counterfactual minimal-pair curriculum: generate semantically close candidates where the teacher margin flips for a causal reason.
4. Use a pretrained S0/Wide7 checkpoint as the student base instead of a tiny random margin student.
5. Remove the margin-regression term and test pure rank loss to avoid cross-tokenizer scale mismatch.

Those are not run here because the flat signal failed the predeclared gate.

## What Survived

- The functional-margin evaluation route is reusable.
- `code/margin_distillation.py` is a canonical prototype harness for future explicitly approved variants.
- The project now has a concrete negative result separating "loss decreases on teacher preferences" from "benchmark margins improve."
- The codec remains useful as input infrastructure, but not proven as a decision-boundary transfer substrate in this smoke.

## What Died

This claim died:

```text
A first-order teacher-margin loss on unlabeled natural continuations will quickly create benchmark-facing functional margins in a byte student.
```

This broader claim is not dead, but remains unproven and should not be scaled from this evidence:

```text
Functional Margin Distillation is the new primary moonshot.
```

The next serious attempt must change the target distribution or teacher setup before any scale-up. More examples of the same natural-continuation margin target are not justified.

## NARRATIVE SECTION

One-sentence story: We stopped copying Qwen's hidden coordinates, trained a byte student on Qwen's decision margins instead, and the first functional smoke still did not make it better at choosing answers.

Does it survive "isn't that obvious?": It survives as a falsification. The loss went down, so the negative result is not "nothing trained"; it is "the trained margin signal did not transfer to the benchmark-facing boundary."

Does it survive "so what?": Yes, as a kill gate. It prevents the project from spending a larger run on a target that was flat or harmful on all three functional checks.

If boring, say so: This is boring as a moonshot result and useful as process discipline. Functional margins remain the right kind of metric, but this first margin-distillation formulation does not earn scale.
