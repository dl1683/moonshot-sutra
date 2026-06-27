# Sutra & Eklavya

**Sutra** is a from-scratch language model designed to be the world's most
efficient learning mechanism — absorbing knowledge from multiple pretrained
teachers of diverse architectures and using their reach as signal. It currently
uses a byte-level patch-global architecture, but the architecture serves the
goal; the goal is not to build a byte model.

**Eklavya** is the multi-teacher learning protocol that trains Sutra. It extracts
invariants from diverse teacher models (different architectures, different
tokenizers) and routes them based on measured disagreement. The key insight:
teachers are instruments, not masters. The student learns from their
disagreements, not their consensus.

## Why Bytes (For Now)?

Tokenizers create invisible walls. Bytes provide a simpler and more universal
interface for cross-architecture knowledge transfer than token-projection or
chunk-matching alternatives. Every teacher's output can be compared at the byte
level, regardless of internal tokenization.

This is how Eklavya becomes practical: a universal substrate where any teacher's
knowledge can be absorbed without vocabulary alignment hacks.

## Current State

**S0 (Scout Build)** — implementation complete, training imminent.

| Component | Status | Key File |
|-----------|--------|----------|
| Architecture (121.7M params) | Built & tested | `code/s0_architecture.py` |
| Training loop + burn-in | Ready | `code/s0_training.py` |
| Data pipeline (byte shards) | Complete (565 shards, 141 GiB) | `code/prepare_byte_shards.py` |
| Pre-training preflight | All checks pass | `code/preflight.py` |
| Causality regression tests | Passing | `code/test_overfit.py` |
| Burn-in verdict automation | Ready | `code/burnin_verdict.py` |
| Live training monitor | Ready | `code/monitor.py` |
| Evaluation + generation | Ready | `code/s0_eval.py` |

**E1 (Single-Teacher KD)** — implementation complete, tested, pending S0 checkpoint.

| Component | Status | Key File |
|-----------|--------|----------|
| Teacher signal cache builder | Built & tested | `code/eklavya_cache.py` |
| KD training loop (3-phase) | Built & tested | `code/eklavya_training.py` |
| Unit tests (28 tests) | All passing | `code/test_eklavya.py` |

**E2 (Multi-Teacher KD)** — fully wired with mmap-backed cache, integration tests, and GPU launch checklist. Ready for GPU.

| Component | Status | Key File |
|-----------|--------|----------|
| Teacher registry (5 teachers) | Built & tested | `code/eklavya_e2_cache.py` |
| Binary cache records & I/O | Built & tested | `code/eklavya_e2_cache.py` |
| PL-style router & purifier | Built & tested | `code/eklavya_e2_router.py` |
| Projection ports & losses | Built & tested | `code/eklavya_e2_losses.py` |
| Multi-teacher gradient budget | Built & tested | `code/eklavya_e2_losses.py` |
| Calibration loss | Built & tested | `code/eklavya_e2_training.py` |
| Unit tests (367 E2 tests) | All passing | `code/test_eklavya_e2.py` |
| Data loader tests (8 tests) | All passing | `code/test_overfit.py` |
| Cache builder (2-pass) | Built | `code/eklavya_e2_cache_builder.py` |
| E2 trainer with curriculum | Built & reviewed | `code/eklavya_e2_training.py` |
| Ablation evaluation harness | Built | `code/eval_e2.py` |
| Ablation modes (A0-A9c + BLD) | Implemented & tested | `code/eklavya_e2_training.py` |
| Ablation comparison & decisions | Built | `code/compare_ablations.py` |
| Protocol document | Written | [E2 Protocol](research/EKLAVYA_E2_PROTOCOL.md) |
| Monitoring protocol | Written | [E2 Monitoring](research/E2_MONITORING_PROTOCOL.md) |

### S0 Architecture

```
ByteEncoder (P=4 bytes → 1 patch state)
  └─ 2-layer local mixer (patch-isolated, no future leakage)
  └─ Gated MLP aggregator → D=576 patch state

GlobalReasoner (30-layer causal transformer)
  └─ GQA: 9 heads / 3 KV heads
  └─ SwiGLU FFN (1536 intermediate)
  └─ RoPE positional encoding
  └─ Activation checkpointing (5.5 GB peak VRAM)

ByteDecoder (4-layer causal decoder)
  └─ Cross-attention to nearby patch states
  └─ Autoregressively predicts 4 bytes per patch
  └─ Shift-by-one: hidden[i] predicts bytes of patch i+1
```

### Data

Three admitted Common Pile subsets (Public Domain / CC0 only):
- `arxiv_abstracts` — scientific text
- `caselaw_access_project` — legal text
- `biodiversity_heritage_library` — natural history

565 shards x 256 MiB = 141 GiB total. 50K training steps at batch 64 covers
~0.09 epochs.

## Build Order

```
D0: Data admission (source/license filtering)        ✅ Complete
S0: Byte/patch scout (121.7M, fixed compute)          ⏳ Training next
E1: Single-teacher byte-level KD (anchor teacher → S0)   📐 Designed
E2: Multi-teacher KD (5 teachers → S0)                📐 Infrastructure built
T0: Teacher feasibility profiling (parallel)           ◻ After S0
G0: Gap mapping on real student traces                 ◻ After S0 trained
P0: Packet compilation for observed gaps               ◻ After G0
G1: Integrated runtime (350-500M, all 7 interfaces)   ◻ After P0
O0: Ownership/credit/efficiency gates                  ◻ After G1
```

## Design Documents

- [Vision](research/VISION.md) — what Sutra and Eklavya are, claims and non-claims
- [SE1 Canonical Spec](research/SE1_CANONICAL_SPEC.md) — frozen build spec,
  produced through multi-round adversarial deliberation (R1-R12)
- [Eklavya E1 Protocol](research/EKLAVYA_E1_PROTOCOL.md) — byte-level KD design
  (single-teacher, 3-phase schedule, sparse caching)
- [Eklavya E2 Protocol](research/EKLAVYA_E2_PROTOCOL.md) — multi-teacher KD design
  (5-teacher roster, PL router, purifier, gradient budget, 16 ablations)
- [E2 Teacher Feasibility](research/EKLAVYA_E2_TEACHER_FEASIBILITY.md) — per-teacher
  admission checklist (6 checks, admit/drop rules)
- [E2 Ablation Plan](research/EKLAVYA_E2_ABLATION_PLAN.md) — E2.5 ownership tests
  (A0-A9c + A5a/b/c + BLD, two-phase strategy, decision rules)
- [Eklavya Doctrine](research/EKLAVYA_DOCTRINE.md) — learning protocol design
- [Ground-Up Future Design](research/GROUND_UP_FUTURE_DESIGN.md) — first-principles
  architecture (unconstrained by prior assumptions)
- [W0 Registries](research/W0_REGISTRIES.md) — materialized interface, lesson,
  measurement, and threshold registries for G1
- [Data Admission](research/DATA_ADMISSION.md) — admitted/held/rejected sources,
  license posture, shard preparation

## Quick Start (CPU-only validation)

```bash
pip install torch numpy transformers pytest
cd code
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest test_overfit.py test_eklavya.py test_eklavya_e2.py test_burnin_verdict.py test_export_log_csv.py test_utilities.py -v
```

All 492 tests run on CPU without any data, models, or GPU. They validate
the S0 architecture (config presets, loss shape, LR schedule, causality),
the full E1/E2 infrastructure (binary record I/O, router, purifier, losses,
gradient budget, cache builder, position manifest, teacher records, training
config), the E2 monitor anomaly detection (route entropy collapse,
gradient budget suppression, zero teacher signal, disagreement routing),
the burn-in verdict system (hard fail detection, soft concerns, trajectory
analysis), log export (train/eval CSV with teacher losses and route stats),
and operational security (opsec pattern scanning, OneDrive path guards).

## Hardware

Single NVIDIA RTX 5090 Laptop (24 GB VRAM). S0 fits comfortably at 5.5 GB peak
with activation checkpointing (batch=4, seq=4096).

## Philosophy

Intelligence = Geometry, not Scale. Mathematical structure beats brute-force
parameters. If the theory is right, you don't need a data center.
