# State

## Eklavya Mission (anchor — read first)

**Eklavya is not a technique. Eklavya is the idea:** billions of dollars of
accumulated knowledge sit in pretrained models and get thrown away every time
someone builds a new system. Can we unlock that knowledge and let new, smaller
models inherit it — so AI development becomes "build on what exists" instead
of "start from scratch"? Every mechanism we test (tomography, KD, indexed
heads) is expendable. The mission is not.

---

**Last updated:** 2026-09-04
**Current state:** Ship v0 DONE (MRR 0.562, unfrozen KD). Matched B0 control running. MTEB eval pending (blocked on B0). E16 implementation ready.
**Blackboard:** `1d65d9fb`

## Direction

**E1.5 corrected adjudication: 3/3 seeds complete. Codex verdict: INCONCLUSIVE.**

Cross-seed E15 vs B4c deltas (valid seeds only — B4c catastrophic on seed_271):
seed_42 = -0.0006, seed_137 = -0.0095. Both negative. Mean = -0.005.

Seed 42: B0(0.779) >> B2=B4c(0.753) > E15(0.752) > E15_id(0.748) > B3(0.724)
Seed 137: B0(0.761) > E15_id(0.754) > B2(0.752) > B4c(0.747) > B3(0.739) > E15(0.737)
Seed 271: B0(0.767) > E15_id(0.748) > E15(0.746) > B3(0.742) > B2(0.740) > B4c(0.220) [BROKEN]

B4c seed_271 catastrophically broken (MRR=0.22 vs 0.75 on other seeds) — genuine
training instability, reproducible. This invalidates seed_271 for the paired test
but is itself informative: the B4c absorber is fragile.

**Key finding: B0 (contrastive only, no teachers) WINS all 3 seeds.** Mean 0.769.
All KD variants (B2, B3, E15) hurt vs pure contrastive on frozen encoder.

Codex evidence gate verdict (2026-09-04): INCONCLUSIVE. E15 fails its positive
threshold, but valid-seed CI (df=1) includes both 0 and +0.005, so the
precommitted kill criterion is also unresolved. All 3 seeds share the same 200
eval queries (training-randomness replicates, not data replications). B4c
instability (seed_271) is genuine evidence of algorithmic fragility.

Consistently unfavorable to tested mechanisms, but not a terminal kill. The
frozen-encoder restriction may be WHY KD doesn't help — unfreezing could change
the dynamics fundamentally. Ship mode (standard KD, unfrozen encoder) is the
next artifact.

Codex V1 evidence gate (2026-09-04): V1 is an exploratory debugging run, not
valid evidence. Both "tomography dead" and "standard KD wins" are overclaimed:
- Text V2-R2: ceiling-saturated, precommitted criteria not executed
- Text E1: noise-level, 200-pair contradicts, algebraic identity flaw
- Vision V1: catastrophic forgetting masks method differences, probe-target
  misalignment, no seed control, no CI

The Eklavya philosophy ("steal from existing models") survives. Output-mimicry
mechanisms (tomography, KD loss, teacher-indexed heads) are exhausted but not
formally killed. Strategic pivot: teacher value lies in DATA CURATION, not loss
modification. E16 Boundary Inheritance (teacher-guided negative mining) is the
highest-leverage untested mechanism (Codex strategic recommendation 2026-09-04).
Ship mode (unfrozen encoder, standard KD) is the engineering artifact baseline.

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

## V2-R2 Results (COMPLETED — NARROW NEGATIVE, not terminal kill)

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

**Codex evidence gate reclassification (2026-09-04):**
`V2-R2 NARROW NEGATIVE — ceiling-saturated, terminal criterion not executed`

The experiment validly shows this frozen-MiniLM/head config produced no Eklavya
advantage over B4c. It does NOT show response deltas contain zero useful info.
- Baseline nDCG 0.9344 = 37 of 45 queries already rank-1. Each nonzero gain is
  exactly one query moving between rank 1 and rank 2. Evaluation cannot resolve
  the precommitted +0.005 threshold.
- Run had 10 docs, 1 domain, 1 seed, no CI. Precommit required 32 docs, 2
  domains, 3 seeds, paired positive CI, unseen intervention template.
- Eklavya-specific loss activation rate and sign balance were not recorded.
- Not a terminal kill. Corrected adjudication (E1.5) required.

## Pipeline Code

- `code/experiment_e1.py` — multi-mode experiment runner:
  - `--mode e1` (default): 4-arm text embedding experiment
  - `--mode e1.5`: corrected 6-arm adjudication with teacher-indexed heads
  - `--mode ship`: standard KD at scale for artifact production
- `code/experiment_v1.py` — 4-arm vision embedding experiment (CIFAR-100)
- `code/experiment_a1.py` — 4-arm audio embedding experiment (ESC-50/synthetic)
- `code/eval_mteb.py` — MTEB evaluation (handles raw checkpoints + sentence-transformers)
- `code/export_model.py` — sentence-transformers export (loads trained encoder + projection)
- `code/embed_tomography.py` — signature extraction, probes, loss functions
- `code/data_loader.py` — hard toy data, MS MARCO loader, hard negative mining

## Known confound

Each E1 arm creates a fresh ModernBERTEmbedder with a random nn.Linear(768, 384)
projection, giving different baselines. Gains are more comparable than absolute
MRR. Fixed in E1.5 via `proj_seed` parameter ensuring identical initialization
across arms. (The standalone `experiment_e2.py` was deleted 2026-09-04; ship
mode is now `experiment_e1.py --mode ship`.)

## What survived the 14 kills

1. The absorption-ladder methodology
2. Teacher tomography as a primitive
3. Retained gain after teacher removal = soul test
4. The five sacred outcomes as fixed points
5. Supplied geometry keeps getting absorbed — must infer, not receive

## What is dead (do not revisit)

See `research/STATUS.md` for the full 14-kill record.

## V2-R2 / "Kill #15" Analysis (RECLASSIFIED)

V2-R2 is a narrow negative, not a terminal kill. Two independent Codex evidence
gates confirmed: the experiment was ceiling-saturated and did not execute its
own precommitted criteria.

**What remains open:**
- Response-delta as a training signal is NOT conclusively killed — the test
  lacked resolution to detect small effects
- The tomography loss has a separate fundamental flaw: algebraic identity
  avg(KL(P_t||Q)) = KL(avg(P_t)||Q) + C (C has no student gradient), meaning
  teacher identity is erased through averaging. This must be fixed regardless
  of evaluation quality.
- Corrected adjudication (E1.5) with identity-preserving loss, 32-doc hard
  negatives, 2 domains, 3 seeds will settle the question

**Two independent problems to solve before E1.5:**
1. Loss function: break the algebraic identity (teacher identity preservation)
2. Evaluation: 32-doc raw-student hard-negative pools, proper statistical power

## Codex Design Gate (2026-09-04): NO-GO → Redesign

Codex design gate reviewed the original E1.5 and issued NO-GO. Key findings:

1. **Pairwise margin loss has dead zone**: when teachers disagree, gradient = 0
   for |d| ≤ m. Verified by running dummy student.
2. **Reverse KL ≈ forward KL at low tau**: geometric vs arithmetic mean differs
   by ~0.007 at tau=0.5. Not a meaningful fix.
3. **B4c was NOT support-matched**: E1.5 had 5,952 teacher constraints vs B4c
   186 gold constraints. Not a fair comparison.
4. **Negation probes create false labels** for B4c (gold doc may be irrelevant
   to negated query).
5. **Blocking bugs**: evaluate() lacked eval mode, RNG not properly seeded,
   bootstrap absent (used 1.96 instead of t-distribution), potential grad crash.
6. **B3 was miscalibrated**: averaged raw cosine scores instead of calibrated
   softmax distributions.

**Fix implemented**: Teacher-indexed auxiliary heads. Each teacher gets its own
nn.Linear(dim, dim) head over the shared encoder. Student embedding goes
through shared encoder → proj → teacher-specific head → similarity. This
genuinely breaks the algebraic identity: Q_t ≠ Q_t' for different teachers.
B4c absorber is now support-matched: same heads, same probe × teacher support
structure, but gold targets instead of teacher distributions.

## V1 Vision Results (COMPLETE — standard KD wins)

Student: DINOv2-small (21M), 256-dim projection
Teachers: DINOv2-base (86M, 768-dim) + CLIP-ViT-B/32 (86M, 512-dim)
Data: CIFAR-100, 300 train / 100 eval pairs, 600 steps, lr=1e-5

**ALL arms catastrophically destroy pretrained features.** End-to-end fine-tuning
of DINOv2-small on 300 CIFAR-100 pairs wipes out pretrained knowledge.

| Arm | Baseline MRR | Final MRR | **Gain** | Rank |
|-----|-------------|-----------|----------|------|
| B2 KD single (DINOv2-base) | 0.812 | 0.478 | **-0.334** | **1st** |
| V1 tomography | 0.871 | 0.521 | -0.350 | 2nd |
| B0 contrastive | 0.828 | 0.425 | -0.403 | 3rd |
| B3 KD avg | 0.839 | 0.430 | -0.409 | 4th |
| B4c aug_contrastive | 0.836 | 0.403 | -0.434 | 5th |

**Findings (gain-based, not absolute MRR which is init-confounded):**
1. Teacher KD regularizes against forgetting: B2/V1 are least destructive
2. Per-teacher normalization beats averaging: V1 beats B3 by 0.059
3. Single compatible teacher beats multi-teacher tomography: B2 beats V1 by 0.016
4. Probe augmentations harmful: B4c is 0.031 worse than B0

Code's built-in "PASSES" verdict (V1 MRR > B4c + 0.01) is confounded by V1's
higher random baseline (+0.035). Gain comparison is fairer and shows V1 does
not beat B2. Codex evidence gate pending.

## Codex V1 Evidence Gate (2026-09-04): FAIL

Codex verdict: V1 is exploratory debugging, not valid evidence. Key findings:
- Metrics not fairly comparable (0.059 baseline spread from random init)
- Probe-target misalignment (4/7 KL targets see different images than student)
- Catastrophic forgetting masks method differences
- Neither "per-teacher norm beats avg" nor "standard KD wins" is established
- Method-level death claim is scientifically invalid
- Recommendation: run corrected E1.5; do not use V1 for shipping pivot

Full verdict: `outputs/V1_cifar100/codex_evidence_gate.txt`

## Live threads

1. **Ship mode** — ARTIFACT TRACK. Standard KD at scale with unfrozen encoder
   (ModernBERT-base + BGE-large, 50K pairs, 10K steps). The model IS the
   deliverable. This tests KD in the unfrozen regime where it may actually work.
2. **E16 Boundary Inheritance** — MECHANISM TRACK. Teacher-guided negative mining.
   Teacher picks training comparisons; student trains with pure B0 contrastive.
   Codex-recommended highest-leverage untested mechanism (2026-09-04).

## Next

1. ~~Fix E1.5 code defects~~ DONE
2. ~~Run corrected E1.5 text experiment~~ DONE (3/3 seeds)
3. ~~Codex evidence gate~~ DONE (INCONCLUSIVE — narrow negative, not terminal kill)
4. ~~Ship mode v0~~ DONE (5K pairs, 3K steps, kd_weight=0.7, unfrozen: MRR 0.335→0.562)
5. Matched B0 control (kd_weight=0, same setup) — RUNNING. Determines if KD helped.
6. MTEB quick eval — PENDING (blocked on B0). Competitive targets: floor ~56 (MiniLM-L6), realistic ~62
7. ~~E16 Boundary Inheritance~~ IMPLEMENTED. Pilot blocked on B0 + MTEB completion.
