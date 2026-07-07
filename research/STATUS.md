# Project Status

**Last updated:** 2026-07-07 (post-check-in #7)
**Current loop state:** W-Loop B9 running, Q-Loop B12 running
**Live moonshot candidate:** Coordinate-Inheritance Sutra v1 (v0 KILLED at Stage 1, repairs in progress)

This file is the current source of truth. Older research files remain as
provenance, but this page governs the live interpretation of the project.

## Executive Summary

Coordinate-inheritance v0 ran a full 1000-sequence Stage 1 preflight and produced
MASSIVE signal: +6.13 nats/token copied-vs-random advantage (3x threshold), 99.3%
gap closure at token-end, and 0% adapter share of NLL lift. The adapter-does-the-work
story is killed by the data.

However, v0 FAILED two precommitted Stage 1 controls:
1. Patch-boundary frozen-core gain = 66.3% (need >=70%)
2. Token-end rotation no-inverse retention = 33% (need <=30%)

Both are borderline. v0 is killed before benchmark training per the gate chain. v1
repairs are in progress (readout-conditioned adapter, stronger disruption controls,
layer depth curve, generic pretrained control).

The surviving thesis:

**A small byte-native runtime may need to inherit reasoning geometry discovered by
large-scale token models, then compress and expose it through bytes. The geometry
itself — not just pretrained initialization — is load-bearing. The v0 Stage 1 data
strongly supports this, pending two control repairs.**

## Alive, Dead, In Progress

| Track | Status | Current role | Notes |
|-------|--------|--------------|-------|
| Coordinate-Inheritance v1 | **MAINLINE** | Sole moonshot candidate | v0 Stage 1 massive signal but 2 borderline gate failures; v1 repairs in W-Loop B9 |
| Coordinate-Inheritance v0 | **KILLED** | Precursor | Stage 1 failed; signal preserved, controls insufficient |
| Chain-init | Alive as baseline | Engineering fallback | Subsumed by coordinate-inheritance |
| Codec | Alive as infrastructure | Byte-to-token bridge | Phase 1.5: 37.89% patch top-1 |
| Evidence-Native direction | **PERMANENTLY DEAD** | Falsified | 2 architectures, 2 corpora, 2 scales |
| Brainseed v0 | **DEAD** | Negative-result science | All scorers worse than codec-only |

## Current Dual-Loop State

| Loop | Batch | Status | Artifact |
|------|-------|--------|----------|
| Work Loop | B9 (iter 81-90) | **Running** | v1 repairs + Stage 1 re-run; Stage 2 if passes |
| Work Loop | B8 (iter 71-80) | Completed | v0 Stage 1 preflight — KILLED (check-in #7) |
| Question Loop | B12 (iter 78-84) | **Running** | Attack v1 repairs + Stage 1 interpretation |
| Question Loop | B11 (iter 71-77) | Completed | Adapter attribution framework, 5 story classifications |
| Supervisor | Check-in 7 | Completed | v0 Stage 1 assessment, v1 repair orders |

## Key Findings

1. **Coordinate-inheritance v0 Stage 1: massive signal, borderline failures.**
   - Copied advantage: 6.13 nats/token (3x threshold)
   - Gap closure: 99.3% at token-end
   - Adapter attribution: 0% adapter share (adapter+random = baseline)
   - Layer geometry matters: shuffled collapses, rotation with inverse recovers perfectly
   - Failed: patch frozen-core 66.3% (70%), token rotation 33% (30%)

2. **Adapter-does-the-work story: KILLED.**
   - Same adapter + random core = 18.18 NLL
   - Same adapter + copied core = 12.05 NLL
   - 6.13 nat gap is entirely core geometry, not adapter learning

3. **Missing controls (needed for v1):**
   - Generic pretrained control (non-Qwen layers)
   - Readout-conditioned adapter (separate for token-end vs patch-boundary)
   - Stronger disruption (beyond input-gauge rotation)

4. **CBD remains the competitor:** 42.65% HellaSwag at 138M via chain KD.

## Coordinate-Inheritance Gate System (from Q-Loop B10)

| Stage | Gate | Status |
|-------|------|--------|
| 0 | Artifact requirements (3 seeds, paired stats) | Pending |
| 1 | Codec/gauge preflight (>=2 nats, >=60% closure, frozen-core >=70%, rotation collapses) | **v0 FAILED (2 borderline); v1 in progress** |
| 2 | Uncompressed benchmark (>=35% HellaSwag, >=+8pp over Wide7) | Blocked by Stage 1 |
| 3 | 121M compression (>=35% HellaSwag at 121M active) | Pending |
| 4 | Byte-native story (robustness/cross-tokenizer advantages) | Pending |
| 5 | Moonshot promotion (beat SmolLM2-135M targets) | Pending |

## v0 Stage 1 Results (1000 sequences, for reference)

| Gate | Token-End | Patch-Boundary | Required | Overall |
|------|-----------|----------------|----------|---------|
| Copied advantage | 6.13 nats | 4.70 nats | >=2.0 | PASS |
| Gap closure | 99.3% | 87.7% | >=60% | PASS |
| Gap to true NLL | 0.042 nats | 0.66 nats | <=1.5/2.0 | PASS |
| Frozen core gain | 72.3% | 66.3% | >=70% | FAIL |
| Rotation no-inverse | 33% retained | 30% retained | <=30% | FAIL |
| Rotation inverse | 100% | 100% | >=80% | PASS |
| Adapter params | 263K | - | <=2M | PASS |

## Artifact Index

### Pivot and Supervisory Documents

- `research/dual_loop_supervisor_checkin_7.md` - v0 Stage 1 assessment, v1 repair orders
- `research/dual_loop_supervisor_checkin_6.md` - Evidence-native DEAD, coordinate-inheritance formal mainline
- `research/dual_loop_supervisor_checkin_5.md` - Evidence-native v0 post-mortem
- `research/dual_loop_supervisor_checkin_4.md` - Brainseed dead, evidence-native mainline candidate

### Loop Batches

- Work loop: B1-B7 (historical), B8 (v0 Stage 1, completed), B9 (v1 repairs, running)
- Question loop: B1-B10 (historical), B11 (adapter attribution, completed), B12 (v1 attacks, running)

### Coordinate-Inheritance Artifacts

- `code/coordinate_inheritance.py` - Full implementation (preflight + benchmark modes)
- `tmp_coordinate_inheritance_full/preflight_metrics.json` - v0 Stage 1 raw data
- `tmp_coordinate_inheritance_full/calibration_adapter.pt` - v0 calibration adapter (263K params)

### Evidence-Native v1 Artifacts (archival)

- `code/evidence_native_v1.py` - v1 implementation
- `tmp_evidence_native_v1_full2/` - v1 results (2 of 3 seeds completed)

## Wording Rules For Fresh Readers

Use:
- "v0 Stage 1 produced massive signal but failed two borderline controls — v1 repairs in progress."
- "The adapter-does-the-work story is killed: same adapter + random core = baseline."
- "CBD at 42.65% HellaSwag is the number to beat."
- "No benchmark-level evidence yet — Stage 2 blocked by Stage 1."

Avoid:
- "Coordinate-inheritance works" — v0 failed Stage 1; v1 untested.
- "The geometry proof is complete" — two controls failed.
- "We're close to benchmarks" — Stage 2 is gated on clean Stage 1 pass.
