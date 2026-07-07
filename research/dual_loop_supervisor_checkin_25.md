# Dual-Loop Supervisor Check-in #25

**Date: 2026-07-07**
**Reviewing: W-Loop B25 (FRAMESEED-0 precommit spec) + Q-Loop B33 (adversarial pre-test)**

---

## 1. What Was Produced

Both loops completed concurrently:

| Loop | Batch | Output | Lines | Tokens |
|---|---|---|---:|---:|
| W-Loop B25 | I225-I234 | `research/frameseed_0_precommit_spec.md` | 1588 | 203K |
| Q-Loop B33 | I225-I231 | `research/question_loop_batch33.md` | 792 | 140K |

W-Loop wrote a 10-iteration precommit spec for FRAMESEED-0 with world design,
learner architecture, packet format, baselines, measurement protocol, smuggling
audit, verdict tokens, teaching ladder, kill conditions, and end-to-end test.

Q-Loop attacked FrameSeed from 7 angles: prior art absorption, frame-vs-selection,
nuisance growth trick, Boolean toy trap, teaching dimension overlap, strongest
positive scenario, and final pre-test verdict.

## 2. The Gap: Q-Loop Caught What W-Loop Missed

Because both ran concurrently, the W-Loop didn't see Q-Loop B33's corrections.
The spec is structurally solid but has 8 missing upgrades that the Q-Loop correctly
identified as critical:

| # | Missing From Spec | Q-Loop Source | Risk If Not Fixed |
|---|---|---|---|
| 1 | T3-R formalization | I225, I226, I231 | "Frame-bearing" stays vague; any teaching set satisfies it |
| 2 | Teaching dimension baseline | I225, I229 | Optimal teaching sets absorb the vaccine trivially |
| 3 | Library-learning baseline | I225, I229 | DreamCoder-style MDL macro learning absorbs the frame |
| 4 | Nuisance-oracle fairness baselines | I227 | Compactness gap is rigged against reconstructive baselines |
| 5 | AFTD metric | I229 | No amortized measure; single-task teaching absorbs |
| 6 | Sibling-task transfer requirement | I226, I229 | Selection looks like transmission without multi-task test |
| 7 | Representation-noncontainment contract | I226 | Frame primitive already in learner DSL = absorption |
| 8 | Boolean escape clause | I228 | PCCP-H fate repeats if only Boolean results |

**This is exactly the productive tension the dual-loop is designed for.** The
Q-Loop is the immune system. The W-Loop drove forward; the Q-Loop caught the
exposures. The supervisor merges them.

## 3. Supervisor Assessment

### W-Loop B25 Quality

**STRONG structurally, INCOMPLETE strategically.** The spec is well-organized,
has precise world generation, proper learner definition, good absorption baselines
(active learning, CEGIS, RAG), clear verdict tokens, and honest narrative gates.
It would be a solid spec for original T3.

But original T3 was killed by Q-Loop B33 (I225). Without the T3-R upgrade, the
spec is vulnerable to the teaching dimension literature — the exact prior art
that would dismiss the result at a conference.

### Q-Loop B33 Quality

**EXCELLENT.** The 7-iteration adversarial attack is the sharpest question-loop
batch yet. Key innovations:

- T3-R definition with representation-noncontainment contract (I226)
- AFTD (Amortized Frame Teaching Dimension) as the formal object (I229)
- Nuisance-oracle fairness suite (I227)
- Sibling-task transfer as the frame-vs-selection discriminator (I226)
- Boolean escape timeline with FRAMESEED-SHEETS-0 (I228)
- Claim ceiling even on success (I230)

This is the Q-Loop doing exactly what it should: staying ahead and hardening the
spec before implementation wastes batches.

### Narrative Gate

Honest one-sentence narrative:

```
The precommit spec defines a fair test, but the Q-Loop found that the test
as written would be absorbed by teaching dimension and machine teaching
literature — it needs 8 specific upgrades before implementation.
```

- Survives "isn't that obvious?": The upgrades are not obvious; they came from
  deep adversarial analysis of prior art.
- Survives "so what?": If the upgrades make the spec unkillable, we have a
  genuinely novel test. If they reveal the spec can't be hardened, we save
  batches.
- Narrative verdict: ALIVE. The direction survives, the spec needs surgery.

## 4. Directives

### CRITICAL: W-Loop B26 Must Incorporate Q-Loop B33 Before Implementation

**Do NOT implement FRAMESEED-0 from the current spec.** The next W-Loop batch
must modify the spec to incorporate all 8 Q-Loop corrections:

```
W-Loop B26 Task: FRAMESEED-0 Spec Hardening

Required modifications:
1. Replace T3 with T3-R throughout. Define T3-R per Q-Loop I226/I231.
2. Add teaching dimension baseline (optimal teaching set over H0).
3. Add library-learning baseline (MDL macro learner with same tasks).
4. Add nuisance-oracle fairness baselines (oracle causal mask, function-only
   MDL, invariant active learner, CEGIS with no reconstruction penalty).
5. Define AFTD metric (amortized frame teaching dimension).
6. Add sibling-task transfer requirement (≥2 sibling tasks, same frame).
7. Add representation-noncontainment contract (bounded non-reachability,
   no low-cost named primitive, no equivalent teaching set).
8. Add Boolean escape clause (FRAMESEED-SHEETS-0 spec by W28, run by W29).
9. Add new verdict tokens: FRAMESEED_T3R_SIGNAL,
   FRAMESEED_T3_ABSORBED_BY_TEACHING_DIMENSION,
   FRAMESEED_T3_ABSORBED_BY_REPRESENTATION_PRIOR,
   FRAMESEED_T3_ABSORBED_BY_NUISANCE_ORACLE,
   FRAMESEED_T3_ABSORBED_BY_LIBRARY_LEARNING,
   FRAMESEED_T3_BOOLEAN_TRAP.
10. Set claim ceiling: even on T3R_SIGNAL, claim only "controlled evidence
    for amortized frame-teaching separation."
```

After hardening, the spec should be re-audited by Codex before implementation
begins.

### Q-Loop B34: Prepare for Implementation Review

Once W-Loop B26 hardens the spec, Q-Loop B34 should:
- Verify the hardened spec satisfies all I225-I231 requirements
- Pre-design the adversarial review criteria for FRAMESEED-0 results
- Identify the single most likely failure mode

### Hard Clock Update

Original: 5 W-Loop batches from B25.
Current: B26 is spec hardening (not counted against implementation clock).
Revised: 4 W-Loop batches from B27 for FRAMESEED_T3R_SIGNAL or absorption.

## 5. Timeline

| Phase | Batch | Target |
|---|---|---|
| Spec (done) | W25, Q33 | FRAMESEED-0 spec + adversarial pre-test |
| **Spec hardening** | **W26** | **Incorporate Q33 corrections** |
| Implementation review | Q34 | Verify hardened spec, pre-design adversarial criteria |
| First implementation | W27 | FRAMESEED-0 code + run |
| Assessment | S26 | T3R signal or absorption? |
| Second domain spec | W28 | FRAMESEED-SHEETS-0 |
| Second domain run | W29 | Typed domain or BOOLEAN_TRAP |
| Milestone gate | W30, S27 | Adversarial review if signal holds |

---

## 6. Supervisor Verdict

```
BOTH LOOPS STRONG. Q-LOOP CAUGHT 8 CRITICAL GAPS IN W-LOOP SPEC.

NEXT: W-Loop B26 hardens spec with Q33 corrections. No implementation until
hardened spec passes re-audit.

HARD CLOCK: 4 batches from B27 to signal or absorption.

THE DUAL-LOOP TENSION IS WORKING AS DESIGNED.
```
