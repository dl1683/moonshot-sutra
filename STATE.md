# State

**Last updated:** 2026-09-04 01:15 ET
**Current state:** E1 numeric pass + causal FAIL. Loss is algebraically just avg-teacher KD. E1.5 absorber gate needed before any claim.
**Blackboard:** `1d65d9fb`

## Direction

Eklavya method applied to small modality-specific models (embeddings first,
then vision, then audio). Simplified per Devansh directive: core philosophy
is stealing/learning from existing models, not over-engineered probe tomography.

Key Codex design gate corrections adopted:
- ModernBERT-base (149M) as student — untrained for embeddings, real room to learn
- Calibrated pairwise margins, not just ranks
- Avoid Kill #9 (routing) and Kill #14 (supplied geometry)
- B4 stack-and-distill as decisive hostile absorber (deferred to E2)

## E1 Results (COMPLETE — tomography passes kill criterion)

Student: ModernBERT-base (149M, untrained for embeddings)
Teachers: all-MiniLM-L12-v2 (33M), bge-large-en-v1.5 (335M)
Data: 500 MS MARCO pairs (400 train, 100 eval), 600 steps per arm

| Arm | Type | Baseline | Final MRR | **Gain** |
|-----|------|----------|-----------|----------|
| B0  | contrastive | 0.343 | 0.514 | +0.171 |
| B2  | single-teacher KD | 0.366 | 0.540 | +0.173 |
| B3  | multi-teacher avg KD | 0.312 | 0.548 | +0.236 |
| **E1** | **tomography** | 0.301 | **0.561** | **+0.260** |

**Kill criterion:** E1 MRR > B3 MRR + 0.01 = 0.558. **PASSES** (0.561 > 0.558)
**Gain comparison** (controls for random projection confound): E1 +0.260 vs B3 +0.236 = **+0.024 margin**
**Hit@1:** E1 best (0.36), **Hit@5:** E1 best (0.87)

**Caveats:** 100 eval pairs → 95% CI ~±0.03 on MRR. Margin is thin.
Random projection confound (different baselines per arm). Needs E2 confirmation.

### Codex Evidence Gate Verdict (2026-09-04)

**E1 passed the numeric gate but FAILED the causal evidence gate.**

Claim ceiling (Codex): "Multi-view probability-pooled teacher supervision produced
a small encouraging result in one noisy run and warrants controlled replication."
The claims "tomography helps", "teacher invariants were extracted", and "Eklavya
beat ordinary KD" are NOT justified.

**Critical algebraic finding:** avg(KL(P_t||Q)) = KL(avg(P_t)||Q) + C where C
has no student gradient. The tomography loss is gradient-equivalent to distilling
the arithmetic mean of teacher distributions. It does NOT preserve teacher identity.

**B3 was catching up:** E1 led B3 by 0.053 at step 400 but only 0.013 at step 600.
Higher asymptote not demonstrated.

### 200-pair Replication (n=50 eval) — DOES NOT REPLICATE

| Arm | Type | Baseline | Final MRR | **Gain** |
|-----|------|----------|-----------|----------|
| B0  | contrastive | 0.383 | 0.467 | +0.084 |
| **B2** | **single-teacher KD** | 0.346 | **0.587** | **+0.241** |
| B3  | multi-teacher avg KD | 0.357 | 0.553 | +0.196 |
| E1  | tomography | 0.388 | 0.564 | +0.176 |

In the 200-pair run, single-teacher KD dominates. Tomography is 3rd on gain
(+0.176 vs B3 +0.196 vs B2 +0.241). Rank order of arms changed between runs.
Combined with the algebraic identity, the 500-pair "signal" is likely noise or
a confounded compute/augmentation artifact.

## V2 Results (COMPLETED — already-trained student)

MiniLM-L6-v2 (22M) student, BGE-large (335M) + MiniLM-L12 (33M) teachers,
500 MSMARCO pairs, 10 docs/query, LR 5e-6, tau 0.02, 500 steps.

| Arm | MRR | Gain |
|-----|-----|------|
| Baseline | 0.8450 | — |
| A: Ranking KD | 0.8250 | -0.0200 |
| B: Tomography | 0.8150 | -0.0300 |
| C: Contrastive | 0.8350 | -0.0100 |
| D: Aug Contrastive | 0.8450 | +0.0000 |

**Verdict:** All teacher KD destructive for already-trained student.
Confirms ModernBERT-base blank-slate is the correct approach.

## V2-R2 Results (COMPLETED — Codex R2 prescribed experiment, KILL #15)

Codex R2 specified: frozen MiniLM-L6-v2 encoder + trainable 384→384 residual
projection head. Single teacher (MiniLM-L12-v2). 300 MSMARCO pairs (10 docs),
210 train / 45 val / 45 test. B4c absorber = same support mask + weighting but
NO response-delta target.

| Arm | nDCG@10 | Gain |
|-----|---------|------|
| Baseline (frozen) | 0.9344 | — |
| aug_contrastive | 0.9262 | -0.0082 |
| kd | 0.9426 | +0.0082 |
| b4c (absorber) | 0.9262 | -0.0082 |
| **eklavya** | **0.9262** | **-0.0082** |

**Eklavya vs B4c delta: +0.0000** (threshold: +0.005)
**Verdict: DEAD.** Response-delta targets carry zero information beyond what
teacher example/weight selection already provides. Only ordinary KD improves
the student. Kill #15.

## Pipeline Code

- `code/experiment_e1.py` — 4-arm text embedding experiment (running)
- `code/experiment_e2.py` — scaled E2 with confound fixes (proj_seed, warmup, 3 teachers)
- `code/experiment_v1.py` — 4-arm vision embedding experiment (CIFAR-100)
- `code/experiment_a1.py` — 4-arm audio embedding experiment (ESC-50/synthetic)
- `code/eval_mteb.py` — MTEB evaluation pipeline for shipping
- `code/export_model.py` — sentence-transformers export for HuggingFace
- `code/embed_tomography.py` — signature extraction, probes, loss functions
- `code/train_student.py` — student training loop
- `code/data_loader.py` — hard toy data, MS MARCO loader
- `code/run.py` — canonical single-command runner

## Known confound

Each E1 arm creates a fresh ModernBERTEmbedder with a random nn.Linear(768, 384)
projection, giving different baselines. Gains are more comparable than absolute
MRR. Fixed in experiment_e2.py with `proj_seed` parameter ensuring identical
initialization across arms.

## What survived the 14 kills

1. The absorption-ladder methodology
2. Teacher tomography as a primitive
3. Retained gain after teacher removal = soul test
4. The five sacred outcomes as fixed points
5. Supplied geometry keeps getting absorbed — must infer, not receive

## What is dead (do not revisit)

See `research/STATUS.md` for the full 14-kill record.

## Kill #15 Analysis

Text embedding response-delta Eklavya is dead. The decisive test (Codex R2
B4c absorber) shows that response-delta matching provides zero information
beyond teacher example selection + support weighting. Only ordinary KD
(matching teacher similarity scores) improves the student.

**What this means for the program:**
- Response-delta as a training signal is killed for text embeddings
- Teacher tomography may still work via OTHER mechanisms (not response-delta)
- Vision embeddings offer architecturally diverse teachers (DINOv2 = self-supervised,
  CLIP = contrastive lang-img, SigLIP = sigmoid) — different training signals,
  not just different weights on the same architecture
- The question becomes: does teacher diversity at the architectural level
  provide information that KD cannot capture?

## Codex E2 Audit — FAILED

E2 code has material deficiencies (Codex found):
- No replicate seeds or hierarchical uncertainty analysis
- Equal steps not equal compute (student FLOPs)
- Missing B4 augmented-contrastive and B5 oracle-single absorber arms
- Nomic v1.5 encoded without required search_query:/search_document: prefixes
- Test-set leakage: winning arm selected on test set
- Three teachers introduced during what should be 2-teacher replication first

## Live threads

1. **E1.5 absorber gate** — must design before any further claims (Codex directive)
2. **Loss function redesign** — current loss algebraically doesn't preserve teacher identity
3. **Vision V1** — experiment code exists but shares the same broken loss
4. **200-pair E1** — COMPLETE, does not replicate 500-pair signal

## Next

1. Design E1.5: factor Views × Targets, 7+ arms, 3 seeds, equal compute, per-query ranks
2. Fix the loss function to actually preserve teacher identity (signal s23)
3. Only then: run E1.5 absorber gate on GPU
4. Vision V1 blocked until loss is fixed (would inherit same algebraic problem)
