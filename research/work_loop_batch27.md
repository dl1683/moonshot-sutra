# Work Loop Batch 27: FRAMESEED-0 Audit Harness Implementation

**Date:** 2026-07-07  
**Role:** Work-Loop worker  
**Batch:** W-Loop B27  
**Scope:** Harness first. No learner optimization. No hidden HFA. No packet-template tuning.  
**Primary artifact:** `code/frameseed0_harness.py`  
**Tests:** `code/test_frameseed0_harness.py`  
**Audit output:** `experiments/frameseed0_b27_audit.json`

## Batch Verdict

```text
HARNESS GATE IMPLEMENTED. NO PERFORMANCE RUNS. NO FRAMESEED SIGNAL CLAIMED.
```

The B27 artifact is an audit harness, not an experiment result. It implements the pre-implementation gate demanded by Supervisor Check-in #26 and Q-Loop B34: generator audit, constructor noninterference, provenance checking, canonical packet bit accounting, baseline adapter parity, sabotage controls, MI dry-runs, and terminal-token golden controls.

The 10,000 dry-run generator MI audit passed with worst normalized MI `0.004915049666568067` under threshold `0.05`. This is evidence that the generator audit is executable, not evidence of learner performance.

## Validation Commands

```powershell
python -m py_compile code/frameseed0_harness.py
python code/frameseed0_harness.py --dry-run-worlds 10000 --mi-threshold 0.05 --output experiments/frameseed0_b27_audit.json
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'; pytest -q code/test_frameseed0_harness.py
```

Pytest result: `10 passed`. The default pytest invocation hit an unrelated globally installed plugin permission error before test collection; disabling plugin autoload scoped pytest to this repo.

## 20 Iterations

### W27-I1: Directive Grounding

Read `research/dual_loop_supervisor_checkin_26.md`, `research/frameseed_0_precommit_spec.md`, `research/question_loop_batch34.md`, and `research/VISION.md`. The binding order is audit harness before performance: no L3 optimization, no hidden HFA, no packet-template tuning.

### W27-I2: Component Boundary

Split the implementation into explicit harness surfaces: generator, packet constructor, serializer, budget ledger, baseline views, smuggling audit, MI audit, token scorer, golden controls, and top-level audit runner.

### W27-I3: RNG Stream Design

Implemented purpose-split RNG streams for world structure, names, orientations, hidden queries, packet construction, learner tie-breaks, baseline tie-breaks, and ablations. Each stream is derived by SHA-256 from public seed, purpose, and namespace, and records draw counts.

### W27-I4: Boolean World Generator

Implemented FRAMESEED-0 dry-run worlds with two effect bits, two paired bits, `m` background bits, admitted two-input Boolean kernels, random role-to-slot permutation, random orientations, random 96-bit surface names, and bijective paired-bit map.

### W27-I5: Intervention Semantics

Implemented query labeling where only edits to effect-role surface slots update the target cause. Edits to paired or background slots change visible queries but not the label source. The world audit requires at least one decisive intervention.

### W27-I6: Public Transcript Boundary

Implemented `PublicTranscript` as the only input to the packet constructor. It contains public schema and oracle facts, not latent role maps, generator seed namespaces, hidden labels, kernel internals, or split metadata.

### W27-I7: Blind Constructor

Implemented `BlindPacketConstructor`, which infers support slots from public label-changing single-slot edit facts. It emits examples, intervention examples, counterexample metadata, invariant hints, verifier clauses, and a representation patch through formal packet entries.

### W27-I8: Constructor Provenance

Every packet entry carries provenance IDs. The audit verifies that entries cite known public transcript facts and that representation-patch support slots are justified by cited label-changing facts.

### W27-I9: Canonical Packet Serialization

Implemented canonical JSON byte serialization as the current harness serializer and computes packet bits as byte length times eight. This is a deterministic stand-in for the eventual canonical binary serializer and is used identically for all systems in this harness.

### W27-I10: Packet Smuggling Static Check

Implemented executable-field banned-term scanning for role and hidden metadata terms including causal, spurious, nuisance, alias, hidden family, rho, beta, and pi. Current constructed packets pass this static check.

### W27-I11: Budget Ledger

Implemented independent budget recomputation with required cost categories: packet bits, oracle query bits, oracle answer bits, final program bits, learned library bits, residual sibling teaching bits, failed query bits, and verifier expansion bits.

### W27-I12: Baseline Adapter Parity

Implemented baseline views for L3, TD-H0, L0 rote/NN, L1 active, L2 CEGIS, RAG, nuisance oracle, and library learning. The parity audit requires identical packet hash, packet bits, task bundle hash, and query budget.

### W27-I13: Field-Denial Failure Control

Added a parity-failure test where `l2_cegis` is denied the packet `entries` field. The baseline packet hash parity audit fails as intended. This makes Q34's baseline translation confound executable.

### W27-I14: Packet-Order Control

Implemented packet reverse and rotate controls. They check entry multiset stability and bit-length invariance under order changes. This does not claim learner invariance yet; it verifies the serializer and audit surface are order-aware.

### W27-I15: Sabotage Control

Implemented support-swap sabotage: rewrite representation-patch support slots to transcript-inert slots while leaving provenance unchanged. The provenance audit rejects the sabotaged packet because inert slots lack cited label-changing facts.

### W27-I16: Generator MI Audit

Implemented the 10,000-world dry-run MI audit over latent role category versus slot index, name prefix, orientation, kernel ID, and sibling ID versus target role-map bucket. The precommitted audit passed at threshold `0.05`.

### W27-I17: Token Scorer

Implemented terminal-token assignment from synthetic `TokenEvidence` with hardened precedence: smuggling/parity/leakage void first, Boolean trap, representation prior, L3-threshold negative, absorption precedence, signal, then negative.

### W27-I18: Golden Token Controls

Added golden controls for smuggling void, Boolean trap, representation prior, negative low-L3, teaching-dimension absorption, library-learning absorption, nuisance-oracle absorption, CEGIS precedence over active/RAG, and clean signal-shaped evidence.

### W27-I19: Top-Level Audit Runner

Implemented `run_preimplementation_audit`, which composes manifest freeze, world audit, packet serialization, constructor provenance, budget recomputation, baseline parity, packet-order controls, sabotage detection, generator MI, and golden token controls. It records `no_performance_runs = true` and `hidden_hfa_reported = false`.

### W27-I20: Verification And Remaining Boundary

Verified compilation, 10,000-world MI audit, and focused pytest coverage. Remaining work belongs to later batches: actual L0/L1/L2/RAG/library learners, hidden query generation, ablations with HFA, AFTD computation from raw logs, representation-noncontainment solver, and final performance-token assignment. B27 intentionally stops before those surfaces.

## Artifact Summary

- `code/frameseed0_harness.py`: B27 harness implementation.
- `code/test_frameseed0_harness.py`: focused audit-contract tests.
- `experiments/frameseed0_b27_audit.json`: 10,000-world dry-run audit output.

## Non-Claims

This batch does not claim `FRAMESEED_T3R_SIGNAL`, HFA, AFTD separation, baseline non-absorption, representation noncontainment, or packet effectiveness. It only claims that the audit harness now makes the first implementation confounds executable.