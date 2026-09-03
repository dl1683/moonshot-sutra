# State

**Last updated:** 2026-09-03
**Current state:** E1 experiment running — ModernBERT-base student, MS MARCO data, 4 arms on GPU.
**Blackboard:** `539efcd4` (Codex-created, replacing stale `1d65d9fb`)

## Direction

Eklavya method applied to small modality-specific models (embeddings first,
then vision, then audio). Simplified per Devansh directive: core philosophy
is stealing/learning from existing models, not over-engineered probe tomography.

Key Codex design gate corrections adopted:
- ModernBERT-base (149M) as student — untrained for embeddings, real room to learn
- Calibrated pairwise margins, not just ranks
- Avoid Kill #9 (routing) and Kill #14 (supplied geometry)
- B4 stack-and-distill as decisive hostile absorber (deferred to E2)

## Active Experiment

**E1 (running on GPU):**
- Student: ModernBERT-base (149M, untrained for embeddings)
- Teachers: all-MiniLM-L12-v2 (33M), bge-large-en-v1.5 (335M)
- Data: 500 MS MARCO pairs with BM25 hard negatives (500 train, 100 eval)
- 600 steps per arm, lr=2e-5, tau=0.05
- Arms:
  - B0: Contrastive only (InfoNCE, no teacher)
  - B2: Single-teacher KD (best teacher, identity probe only)
  - B3: Multi-teacher average KD (averaged scores, identity only)
  - E1: Full tomography (multi-probe, multi-teacher KL)
- Kill: if E1 MRR <= best baseline MRR + 0.01, tomography didn't help

## Pipeline Code

- `code/embed_tomography.py` — signature extraction, probes, loss functions
- `code/train_student.py` — student training loop
- `code/data_loader.py` — hard toy data, MS MARCO loader
- `code/run.py` — canonical single-command runner
- `code/experiment_e1.py` — 4-arm comparative experiment

## What survived the 14 kills

1. The absorption-ladder methodology
2. Teacher tomography as a primitive
3. Retained gain after teacher removal = soul test
4. The five sacred outcomes as fixed points
5. Supplied geometry keeps getting absorbed — must infer, not receive

## What is dead (do not revisit)

See `research/STATUS.md` for the full 14-kill record.

## Live threads

1. **Embedding model via Eklavya** — E1 experiment running on GPU
2. **Vision model landscape** — scout running in parallel
3. **AGI Thesis integration** — accumulation framework grounds persistence
4. **Sangam coordination** — findings integrated (Pareto council, relational principles)

## Next (after E1 results)

- If signal: E2 with bigger teachers (Qwen3-Embedding-4B), more data, B4 hostile absorber
- If absorbed: pivot to Sangam's relational principles or functional concept transfer
- Either way: run Codex evidence gate on results
- Vision: begin Eklavya for vision encoders (DINOv2 family)
