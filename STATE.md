# State

**Last updated:** 2026-09-03 (session 3 — continued from context compaction)
**Current state:** E1 experiment finishing — B0/B2 done, B3 finishing, E1 (tomography) next.
**Blackboard:** `539efcd4` (Codex-created)

## Direction

Eklavya method applied to small modality-specific models (embeddings first,
then vision, then audio). Simplified per Devansh directive: core philosophy
is stealing/learning from existing models, not over-engineered probe tomography.

Key Codex design gate corrections adopted:
- ModernBERT-base (149M) as student — untrained for embeddings, real room to learn
- Calibrated pairwise margins, not just ranks
- Avoid Kill #9 (routing) and Kill #14 (supplied geometry)
- B4 stack-and-distill as decisive hostile absorber (deferred to E2)

## E1 Partial Results

| Arm | Type | Final MRR | Gain MRR | Status |
|-----|------|-----------|----------|--------|
| B0  | contrastive | 0.514 | +0.171 | DONE |
| B2  | single-teacher KD | 0.540 | +0.196 | DONE |
| B3  | multi-teacher avg KD | 0.504* | — | Running (step 400) |
| E1  | tomography | — | — | Pending |

*B3 intermediate result: averaging hurts vs single-teacher. Heterogeneous
teachers create blurred signal. This is potentially good for tomography
which preserves per-teacher/per-probe structure.

Baseline MRR (untrained ModernBERT-base): 0.343

## Active Experiment

**E1 (running on GPU):**
- Student: ModernBERT-base (149M, untrained for embeddings)
- Teachers: all-MiniLM-L12-v2 (33M), bge-large-en-v1.5 (335M)
- Data: 500 MS MARCO pairs with BM25 hard negatives (500 train, 100 eval)
- 600 steps per arm, lr=2e-5, tau=0.05
- Arms:
  - B0: Contrastive only (InfoNCE, no teacher) — DONE MRR 0.514
  - B2: Single-teacher KD (best teacher, identity probe only) — DONE MRR 0.540
  - B3: Multi-teacher average KD (averaged scores, identity only) — running
  - E1: Full tomography (multi-probe, multi-teacher KL) — pending
- Kill: if E1 MRR <= best baseline MRR + 0.01, tomography didn't help

## Pipeline Code

- `code/embed_tomography.py` — signature extraction, probes, loss functions
- `code/train_student.py` — student training loop
- `code/data_loader.py` — hard toy data, MS MARCO loader
- `code/run.py` — canonical single-command runner
- `code/experiment_e1.py` — 4-arm text embedding experiment
- `code/experiment_v1.py` — 4-arm vision embedding experiment (CIFAR-100)
- `code/experiment_a1.py` — 4-arm audio embedding experiment (ESC-50/synthetic)
- `code/eval_mteb.py` — MTEB evaluation pipeline for shipping
- `code/vision_tomography.py` — vision probe transforms and signature extraction

## What survived the 14 kills

1. The absorption-ladder methodology
2. Teacher tomography as a primitive
3. Retained gain after teacher removal = soul test
4. The five sacred outcomes as fixed points
5. Supplied geometry keeps getting absorbed — must infer, not receive

## What is dead (do not revisit)

See `research/STATUS.md` for the full 14-kill record.

## Live threads

1. **Embedding E1** — 4-arm experiment finishing on GPU (B0/B2 done, B3 running, E1 next)
2. **Vision V1** — experiment code ready, launches after E1 frees GPU
3. **Audio A1** — experiment code ready with synthetic data fallback
4. **MTEB eval** — pipeline ready for model evaluation before shipping
5. **AGI Thesis** — accumulation framework grounds persistence
6. **Sangam** — findings integrated (Pareto council, relational principles)

## Next

1. E1 finishes -> analyze results -> Codex evidence gate
2. If tomography signal: scale up in E2 (more data, bigger teachers)
3. If absorbed: ship best KD recipe, mechanism is secondary
4. Launch V1 on GPU after E1 completes
5. Prepare A1 with real audio data (ESC-50 download)
6. First model that beats MiniLM-L6 on MTEB -> ship to HuggingFace
