# State

**Last updated:** 2026-09-03 20:29 ET
**Current state:** E1 COMPLETE — tomography passes kill criterion. V1 vision launching.
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

## Live threads

1. **V1 vision** — launching on GPU now (DINOv2-small student, DINOv2-base+CLIP teachers, CIFAR-100)
2. **E2 scaled** — experiment code ready, fixes projection confound, 5000 pairs, 3 teachers
3. **Audio A1** — experiment code ready with synthetic data fallback
4. **MTEB eval** — pipeline ready for model evaluation before shipping
5. **Codex evidence gate** — reviewing E1 results for overclaims

## Next

1. V1 running on GPU → analyze results → cross-modality evidence
2. E2 scaled text experiment to confirm E1 margin with fixed confound
3. First model that beats MiniLM-L6 on MTEB → ship to HuggingFace
4. If V1 shows tomography signal in vision too → strong cross-modality story
