# Project Status

**Last updated:** 2026-07-07
**Current state:** PCCP-H absorption ladder complete, packaging for pivot
**Fixed points:** The five sacred outcomes in `research/VISION.md`

## Completed Absorption Ladder

| Level | Description | Status | Evidence |
|---|---|---|---|
| B0 | Prior art (CEGIS/ILP/DreamCoder overlap) | Absorbed | Theorem Part 3 |
| B1 | Single-field invariance (FDM-0) | **ABSORBED** | `code/pccp0_witness.py` FDM section |
| B2 | Metamorphic relations (Relation Miner v0) | **ABSORBED** | `code/pccp0_b2_relations.py` |
| B3 | Decomposition discovery | **ABSORBED** | `code/pccp0_b3_decomposition.py` |
| B3.5 | Decomposition synthesis value | **Real but novelty absorbed** | Same as B3 (107x-276x reduction) |
| B4 | Transformation grammar discovery | Open | Not tested |
| B5 | Universal frame formation | Impossible | Theoretical |

Every testable discovery level from B1 through B3 has been absorbed by exhaustive
baselines with equal information. The absorption was predicted by the Q-Loop before
confirmed by the W-Loop.

## What Was Produced

### Concrete Results

1. **After-frame separation theorem** (`research/PCCP_THEOREM_DRAFT.md`):
   - Part 1: Observational equivalence impossibility — proved
   - Part 2: Nuisance-entropy rate-distortion gap Omega(m) — proved
   - Part 3: Restricted verifier discovery for monotone conjunctions — proved

2. **Finite PCCP-A witness** (`code/pccp0_witness.py`):
   - Constant-length causal rule vs growing reconstruction proxy
   - Hidden-family transfer passes, proxy baseline fails
   - Verdict: `FINITE_PCCP_A_SEPARATION` (narrower than full STRONG_PCCP)

3. **B1-B3 absorption suites** (three self-contained Python scripts):
   - Each includes role permutation, smuggling audits, equal-information baselines
   - Each honestly reports absorption as the primary result

4. **B3 synthesis value**: decomposed search reduces synthesis cost 107x-276x
   in the toy Boolean world, but the decomposition boundary itself is absorbed

### Methodology Contributions

1. Absorption ladder: B0-B5 hierarchy with precommit verdict tokens
2. Smuggling audits: DSL, verifier, transformation, label leakage checks
3. Role permutation controls: discovery must work role-blind
4. Equal-information baselines: exhaustive gets same budget and data
5. Hidden-family transfer: clauses frozen before evaluation
6. Honest absorption reporting: negative results as primary evidence

### What Was NOT Produced

1. A non-absorbed discovery mechanism above B3
2. Evidence that PCCP-H beats neural-tool agents or generic synthesis
3. A practical tool beyond toy Boolean worlds
4. A moonshot that makes intelligence cheap

## Current Position

PCCP-H is an audit/verification methodology, not a discovery paradigm.
The absorption ladder is the project's strongest contribution — publishable
as methodology if positioned honestly. The project must pivot to a new
moonshot direction that directly attacks the manifesto.

## Adversarial Review Result

Fresh-eyes hostile review (`research/adversarial_review_final.md`):
- Verdict: OVERCLAIMED narrowly (STRONG_PCCP token too strong — now fixed)
- Honesty: 8/10, Rigor: 7/10, Novelty: 5/10, Moonshot: 3/10
- Publishability: 6/10, Code: 8/10, Methodology: 8/10

## Operating Rules

- Start from `research/VISION.md`
- Treat every mechanism as replaceable
- Think first, formalize first, test small
- CPU-first experiments unless user explicitly authorizes larger runs
- Negative results are valuable when they remove a real assumption

## Artifact Index

### Active Canon

- `research/VISION.md` - first-principles vision and five sacred outcomes
- `research/STATUS.md` - current state (this file)
- `research/PCCP_PRECOMMIT_SPEC.md` - PCCP-H specification
- `research/PCCP_THEOREM_DRAFT.md` - three-part theorem
- `research/adversarial_review_final.md` - hostile fresh-eyes review
- `research/DEEP_RETHINK.md` - historical kill log

### Executable Evidence

- `code/pccp0_witness.py` - after-frame witness + FDM-0 B1 absorption
- `code/pccp0_b2_relations.py` - B2 metamorphic relation absorption
- `code/pccp0_b3_decomposition.py` - B3 decomposition absorption + synthesis value

### Historical Record

- `research/work_loop_batch*.md` - W-Loop iterations
- `research/question_loop_batch*.md` - Q-Loop iterations
- `research/dual_loop_supervisor_checkin_*.md` - supervisor assessments
