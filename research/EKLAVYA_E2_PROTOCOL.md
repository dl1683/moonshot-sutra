# Eklavya E2 Protocol — Multi-Teacher Byte-Level Knowledge Distillation

Status: CANONICAL. Produced through 3-round adversarial deliberation (R1-R3, June 2026).

## 1. Purpose

E2 extends E1 single-teacher KD to multi-teacher KD while preserving teacher identity, purifying disagreement, and proving retained gain after teacher removal. E2 is a **multi-teacher lesson compiler**, not "E1 repeated five times."

The correct primitive:
- Same student gap, same byte position
- Multiple teacher measurements with preserved identity
- Purify or route the signal based on disagreement
- Train through constrained per-teacher projection ports
- Prove retained gain by teacher ablation

## 2. Relationship to E1

E2 inherits from E1:
- S0 student architecture (121.7M, byte-level, patch-global)
- Byte-level KL loss shape (`topk_tail_kl`)
- `AlignProjection` pattern (LayerNorm → Linear, no bias)
- `overlap_pool` for byte-to-patch mapping
- Gradient budget principle (teacher ≤ budget × CE)
- Cache-only training (no teacher models loaded during training)
- Gap-selected positions (not dense KD)

E2 changes from E1:
- Single teacher → 5 candidate teachers (3 causal + 1 semantic + 1 SSM)
- Single AlignProjection → per-teacher projection ports
- Fixed KL target → purified KL target (router + purifier)
- Single gradient budget → per-teacher + total teacher caps
- E1 files remain frozen — E2 is separate code files

## 3. Design Thesis

E2 is not repeated single-teacher KD. The emergent property is **source ecology**: heterogeneous teachers become instruments that expose which structures are invariant across measurement systems and which are teacher-local artifacts.

- Consensus teaches stable byte-level structure
- Disagreement teaches routing, uncertainty, and gap taxonomy
- Ablation proves which instrument contributed

This cannot be obtained by running E1 five separate times, because separate runs destroy the shared coordinate system needed to compare teacher behavior on the same student gaps.

**E2 thesis**: E2 converts heterogeneous teacher disagreement into byte-native purified lesson targets, then proves teacher-specific retained gain after teacher removal.

## 4. Preconditions and Admission Gates

Before E2 begins:
1. E1 checkpoint exists with stable single-teacher KD gains
2. S0 training has converged (burn-in passed, BPB < target)
3. Teacher feasibility profiling (E2.0) passes for each candidate — see [E2 Teacher Feasibility](EKLAVYA_E2_TEACHER_FEASIBILITY.md)
4. Pilot cache (25-50 shards) is built and validated
5. E1 test suite (28 tests) still passes
6. E2 test suite (226 tests) passes

## 5. Teacher Roster and Roles

| ID | Name | Family | Role | Dim | KL | Align | Semantic | Prior | VRAM | Grad Cap |
|----|------|--------|------|-----|----|----|------|-------|------|------|
| 0 | t0_anchor_decoder | Decoder | Anchor | 2048 | ✓ | ✓ | ✗ | 0.40 | 3.4 GB | 0.10 |
| 1 | t1_diversity_hybrid | Hybrid | Diversity | 2048 | ✓ | ✓ | ✗ | 0.20 | 2.4 GB | 0.10 |
| 2 | t2_control_decoder | Decoder | Control | 1024 | ✓ | ✓ | ✗ | 0.15 | 1.2 GB | 0.10 |
| 3 | t3_semantic_embedding | Embedding | Semantic | 768 | ✗ | ✗ | ✓ | 0.10 | 0.6 GB | 0.05 |
| 4 | t4_diversity_ssm | SSM | Diversity | 1536 | ✓ | ✗ | ✗ | 0.15 | 1.6 GB | 0.05 |

Roles:
- **Anchor**: Primary signal source, always active, highest prior
- **Control**: Cheaper decoder to detect when anchor signal is idiosyncratic
- **Diversity**: Different architecture family (hybrid/SSM) for cross-architecture invariants
- **Semantic**: Embedding-space geometry, not causal byte prediction

## 6. Teacher-Port Architecture

Per-teacher projection head:
```
P_t(x) = LayerNorm_t(W_t(LayerNorm_s(x)))
W_t: 576 → teacher_dim, bias=False
```

Keep ports deliberately weak — they must reveal whether S0 states can linearly support the teacher signal, not become hidden students.

Semantic teacher port adds L2 normalization:
```
SemanticTeacherPort: LayerNorm → Linear → L2-normalize
Loss: 1 - cosine(projected_student, teacher_embedding)
```

Warm-start: copy E1 AlignProjection into anchor port. Initialize other ports randomly but keep their loss weights at 0 during warmup.

Implementation: `code/eklavya_e2_losses.py` → `MultiTeacherProjectionPorts`

## 7. Cache Architecture

Binary cache, not Parquet. Sequential streaming into PyTorch.

```
eklavya_e2_cache/
  manifest.json
  teacher_registry.json
  positions.bin            # shared position manifest
  teachers/
    t0_anchor_decoder/
      kl_records.bin       # per-teacher KL records
      align_records.bin    # per-teacher alignment records
      teacher_embeddings.pt
    t1_diversity_hybrid/
    t2_control_decoder/
    t3_semantic_embedding/
    t4_diversity_ssm/
  aggregate/
    route_records.bin      # runtime telemetry (written during E2.4, not pre-built)
```

Record formats (all little-endian struct.pack):
- **PositionRecord** (28 bytes): position_id, shard_id, seq_offset, patch_idx, gold_byte, student_nll, student_entropy, reason_mask
- **E2KLRecord** (60 bytes @ K=16): position_id, patch_idx, tail_prob, entropy, logp_gold, top_bytes[K], top_probs[K]
- **E2AlignRecord** (14 bytes): position_id, byte_start, byte_len, token_id, align_quality
- **RouteRecord** (variable): position_id, n_teachers, jsd, route_entropy, teacher_ids[n], weights[n]

Implementation: `code/eklavya_e2_cache.py`

## 8. Position Selection

Gap-selected, not dense. Generated in one student pass (no teacher loaded):
- Student high NLL (> nll_floor=3.5)
- Student high entropy
- High teacher disagreement (from pilot)
- Random controls (5% of total)

Initial target: 10M-20M positions from 25-50 shards (5-15 GB cache).
Full 565-shard cache only after pilot proves retained gain.

Selection reason encoded as bitfield:
```
HIGH_NLL = 1, HIGH_ENTROPY = 2, DISAGREEMENT = 4, CONTROL = 8
```

## 9. Teacher Signal Types

| Signal | Teachers | Loss Type | Notes |
|--------|----------|-----------|-------|
| Byte KL | All causal (0,1,2,4) | topk_tail_kl | First-byte marginal distribution |
| Embedding alignment | Decoders with tokenizer (0,1,2) | MSE on projected states | overlap_pool patch → token mapping |
| Semantic geometry | Embedding teacher (3) | Cosine loss | Static token embedding alignment (table lookup, not contextual) |
| Confidence/entropy | All | Telemetry only | Not a training signal |

For cross-architecture teachers (SSM, hybrid):
- Byte KL: allowed immediately
- Hidden alignment: off by default, low-rank bridge only after pilot
- Layerwise hidden MSE: forbidden
- Attention-map imitation: forbidden

## 10. Disagreement Measurement

Jensen-Shannon divergence as sensor:
```
q̄_i = Σ_t a_t,i q_t,i
D_i = H(q̄_i) - Σ_t a_t,i H(q_t,i)
```

JSD is used for routing decisions and telemetry, not as a sufficient training target by itself.

Implementation: `code/eklavya_e2_router.py` → `disagreement_jsd()`

## 11. Router and Purifier

### TeacherRouterV0

Simplified Plackett-Luce-style score with 3 terms:
```
score_t,i = log(prior_t) + α·z(log q_t,i[y_i]) - β·z(H(q_t,i))
a_t,i = softmax(score_t,i / τ)
```

Initial constants: α=1.0, β=0.5, τ=1.0. No grid search.

Router versions for validation:
- V0: prior only
- V1: prior + gold likelihood
- V2: prior + gold likelihood - entropy (full)

### Purifier

Three modes:
- **arithmetic**: weighted average (safe default for low disagreement)
- **log_pool**: weighted geometric mean (sharper, rewards agreement)
- **route**: pick highest-weight teacher only (high disagreement)

Start with arithmetic mean. Add log-pool as ablation.

Implementation: `code/eklavya_e2_router.py` → `route_teachers()`, `purify_byte_target()`

## 12. Loss Functions

Full E2 objective:
```
L_E2 = L_CE
     + Σ_t λ_align_t · L_align_t
     + Σ_t λ_sem_t · L_sem_t
     + λ_kl · mean_i w_i · KL(q*_i ‖ p_i)
     + λ_cal · mean_i w_i · (H(p_i) - stopgrad(H(q*_i)))²
```

Where q* is the purified target from router + purifier.

Implementation: `code/eklavya_e2_losses.py`

## 13. Curriculum

From E1 checkpoint, not from raw S0:

| Phase | Steps | Description | What's Active |
|-------|-------|-------------|--------------|
| E2.0 | — | Teacher feasibility profiling | No student update, measure cache validity |
| E2.1 | 500-1000 | Projection-port warmup | Freeze S0, train anchor port only (align loss) |
| E2.2 | 1500-3000 | Low-conflict consensus KD | Anchor + control where JSD low |
| E2.3 | 2000-4000 | Semantic geometry landing | Add embedding teacher, cosine loss |
| E2.4 | 6000-12000 | Disagreement-routed KD | Full router, all admitted teachers |
| E2.5 | — | Ownership and ablation | No teachers, measure retained gain |

Loss ramp per teacher:
```
λ_align_t(step) = λ_max_t · sigmoid_ramp(step, warmup=1000)
λ_kl_t(step) = λ_max_t · sigmoid_ramp(step, warmup=1500)
```

Anchor preservation: keep t0_anchor_decoder align weight live during warmup, cap new teacher gradients separately.

## 14. Cross-Architecture Handling

For SSM and hybrid teachers:

Allowed immediately:
- First-byte KL
- Entropy/confidence telemetry
- Prefix sensitivity
- Long-context gap specialization

Allowed after pilot:
- Low-rank bridge from student hidden → teacher summary state

Forbidden initially:
- Layerwise hidden MSE
- Attention-map imitation
- Forcing S0 Transformer states into SSM recurrent geometry

## 15. Teacher Weighting and Utility Ledger

Static priors are initialization only:
```
Anchor (1.7B):      0.40 (anchor)
Hybrid (1.2B):      0.20 (hybrid diversity)
Control (0.6B):     0.15 (control decoder)
Embedding (300M):   0.10 (semantic only)
SSM (780M):         0.15 (SSM diversity, after admission)
```

Adapt by retained gain:
```
π_t,g ← EMA(π_t,g, retained_gain_t,g / total_teacher_cost_t,g)
```

A teacher is admitted permanently only if:
- contribution_t > 0 on hidden ownership slices
- Collateral damage ≤ predeclared floor
- Raw-average baseline loses to routed/purified E2
- Single-teacher explanation is insufficient

Log gradient cosine between teacher losses. If a teacher consistently opposes the anchor and ablation shows no retained gain, delete it.

## 16. Gradient Budgeting

Per-teacher and total caps:
```
total teacher gradient ≤ 0.30 × CE gradient
per causal teacher   ≤ 0.10 × CE gradient
semantic geometry    ≤ 0.05 × CE gradient
cross-arch bridge    ≤ 0.05 × CE gradient
```

Algorithm per microstep:
1. Save existing accumulated grads
2. Backward CE, capture CE grads
3. For each teacher loss: backward, capture, scale to per-teacher cap
4. Sum scaled teacher grads
5. Scale total to total cap
6. Restore: saved + CE + capped teacher grads

Compatible with AMP GradScaler — all backward calls use scaler.scale().

Implementation: `code/eklavya_e2_losses.py` → `apply_multi_teacher_gradient_budget()`

## 17. Minimum Viable Ablations

Full ablation plan with commands, metrics, and decision rules: [EKLAVYA_E2_ABLATION_PLAN.md](EKLAVYA_E2_ABLATION_PLAN.md)

| ID | Condition | Question |
|----|-----------|----------|
| A0 | E1 checkpoint + CE continuation | Does E2 beat doing nothing? |
| A1 | E1 + anchor decoder (1.7B) only | Does multi-teacher beat single? |
| A2 | E2 all admitted teachers | Full system performance |
| A3 | E2 minus strongest non-anchor | Does best diversity teacher contribute? |
| A4 | E2 minus semantic teacher(s) | Do embeddings help? |
| A5 | E2 raw arithmetic mean (no router) | Does routing matter? |
| A6 | E2 shuffled teacher targets | Are real signals necessary? |

Information value ranking: A2 vs A0 > A2 vs A1 > A5 vs A2 > A6 vs A2 > A3 vs A2 > A4 vs A2.

Only run full leave-one-out after A2 clearly beats A1.

## 18. Retained-Gain and Ownership Tests

Contribution score:
```
contribution_t = score(E2_all) - score(E2_minus_t)
utility_t = contribution_t / (cache_cost_t + training_cost_delta_t)
```

Evaluation protocol:
- Teacher-free retained gain: run E2 student without any teacher signals
- Dependence gap: how much does removing each teacher hurt?
- Collateral damage: does removing one teacher help other domains?
- Teacher-specific contribution: which gap classes does each teacher own?

## 19. Implementation Plan

All E2 code in separate files, E1 remains frozen:

| File | Purpose | Status |
|------|---------|--------|
| `code/eklavya_e2_cache.py` | Teacher registry, binary records, cache I/O | ✅ Built |
| `code/eklavya_e2_router.py` | MultiTeacherBatch, PL router, purifier | ✅ Built |
| `code/eklavya_e2_losses.py` | Projection ports, losses, gradient budget | ✅ Built |
| `code/test_eklavya_e2.py` | 226 tests passing | ✅ Built |
| `code/eklavya_e2_training.py` | E2 trainer with curriculum + real teacher losses | ✅ Built |
| `code/eklavya_e2_cache_builder.py` | Two-pass cache builder (student gaps → teacher records) | ✅ Built |
| `research/EKLAVYA_E2_PROTOCOL.md` | This document | ✅ Written |

GPU-free build steps (all complete):
1. Teacher registry + binary record structs ✅
2. Position manifest writer/reader with tests ✅
3. Per-teacher KL/align cache readers ✅
4. MultiTeacherBatch join by position_id ✅
5. TeacherRouterV0 (PL-style) ✅
6. Arithmetic purifier and routed purifier ✅
7. MultiTeacherProjectionPorts ✅
8. E2 KL loss on synthetic distributions ✅
9. Per-teacher gradient budget on toy model ✅
10. Full test suite ✅

GPU-required steps (pending):
- Position manifest generation (student forward pass)
- Per-teacher cache building (one teacher at a time)
- E2 training with curriculum
- All ablations

## 20. Failure Modes and Deletion Rules

Drop a teacher when:
- Retained gain is zero or negative on all gap classes
- Gradient cosine consistently opposes anchor
- Cache-to-training cost exceeds contribution

Drop router when:
- A5 (raw arithmetic) matches or beats A2 (full router)
- Router entropy is maximal (uniform routing = no signal)

Drop semantic loss when:
- A4 matches A2
- Cosine loss stagnates during E2.3

Drop log-pool when:
- Arithmetic mean outperforms in ablation
- Top-K approximation artifacts dominate

Scale up cache only when:
- Pilot (25-50 shards) shows clear retained gain
- A2 > A0 by decisive margin (not noise)

## 21. Decision Log

| Round | Date | Decision | Rationale |
|-------|------|----------|-----------|
| R1 | 2026-06-27 | E2 = multi-teacher lesson compiler | Not repeated E1. Preserve teacher identity. |
| R1 | 2026-06-27 | Cache-only training | No teacher models loaded during training |
| R1 | 2026-06-27 | Delete raw averaging as default | Only for low-disagreement consensus |
| R1 | 2026-06-27 | Embedding teachers are semantic, not causal | Different loss type, separate port |
| R2 | 2026-06-27 | Binary cache, not Parquet | Sequential streaming, no PyArrow dependency |
| R2 | 2026-06-27 | 3-term PL router (α=1, β=0.5, τ=1) | Simplified from R1's 6-term score |
| R2 | 2026-06-27 | 7 ablation configs (A0-A6) | Cut from R1's 12 controls |
| R2 | 2026-06-27 | Arithmetic mean first, log-pool as ablation | Geometric pool over-penalizes missing top-K |
| R2 | 2026-06-27 | Gap-selected 5-15 GB pilot cache | Not full 90 GB immediately |
| R2 | 2026-06-27 | Per-teacher grad cap 0.10, total 0.30 | Extends E1's 0.30 budget to multi-teacher |
| R3 | 2026-06-27 | Protocol doc outline: 21 sections | Scaffold from R1/R2 content |
| R3 | 2026-06-27 | Delete artifacts/ and tools/ | Historical scaffolding, not active |
| R3 | 2026-06-27 | Archive 7 non-canonical research docs | Preserve provenance, reduce surface area |
| R4 | 2026-06-27 | Fix frozen CE, uint8 indexing, semantic teacher, embedding table | 4 runtime-crash bugs in cache builder + training loop |
| R5 | 2026-06-27 | **GO verdict** — all code review complete | No runtime-crash findings remaining |
