# Dual-Loop Supervisor Check-in #26

**Date: 2026-07-07**
**Reviewing: W-Loop B26 (spec hardening) + Q-Loop B34 (implementation review)**

---

## 1. What Was Produced

| Loop | Batch | Output | Lines | Tokens |
|---|---|---|---:|---:|
| W-Loop B26 | I225-I234 (mod) | `research/frameseed_0_precommit_spec.md` (hardened) | 2436 | 223K |
| Q-Loop B34 | I232-I238 | `research/question_loop_batch34.md` | 786 | ~200K |

W-Loop B26 incorporated all 8 Q-Loop B33 corrections into the FRAMESEED-0
precommit spec. 1185 insertions, 338 deletions. Final self-audit passed clean:
all T3→T3-R replacements done, new baselines added (teaching dimension,
library-learning, nuisance-oracle fairness suite), AFTD metric defined,
sibling-task transfer requirement added, representation-noncontainment contract
formalized, Boolean escape clause with SHEETS-0 timeline, expanded verdict
tokens, and claim ceiling set.

Q-Loop B34 looked past the spec corrections and attacked implementation
confounds. Key output: a 12-point pre-implementation smuggling audit checklist,
adversarial review criteria for the milestone gate, constructor noninterference
contract, and a conditional go/no-go verdict.

## 2. Supervisor Assessment

### W-Loop B26 Quality

**STRONG.** All 8 required modifications landed. The spec is now hardened
against the attacks Q-Loop B33 identified. The self-audit is clean — no
residual bare T3 references, no missing verdict tokens, no stale baselines.

### Q-Loop B34 Quality

**EXCELLENT.** The question loop correctly identified that the spec-level
hardening is necessary but not sufficient. The real danger zone shifted from
"conceptual absorption" (B33's concern) to "implementation smuggling" (B34's
concern). The constructor noninterference contract and the 12-point smuggling
audit checklist are exactly right — they ensure the harness proves its own
integrity before any scientific claim is made.

Key innovation: "The first scientific artifact is not HFA. It is proof that
the harness emits the right tokens on controls and cannot smuggle the frame
through the constructor." This is the correct implementation order.

### Narrative Gate

Honest one-sentence narrative:

```
The FRAMESEED-0 spec survived adversarial hardening — 8 absorption gaps
closed, implementation confounds cataloged — and the next step is building
an audit harness that proves its own integrity before measuring any signal.
```

- Survives "isn't that obvious?": No. The specific confounds (constructor
  answer-awareness, baseline translation asymmetry, budget accounting gaps)
  are not obvious and came from deep adversarial analysis.
- Survives "so what?": If the harness proves clean, any signal is real.
  If it can't prove clean, we saved ourselves from fake evidence.
- Narrative verdict: ALIVE.

## 3. Directives

### W-Loop B27: Begin Implementation — HARNESS FIRST

Per Q-Loop B34's conditional go: implementation starts with the audit harness,
NOT with learner performance. W27 scope:

1. World generator with audited RNG streams
2. Constructor in provenance-logged blind mode
3. Baseline adapter parity tests
4. Scorer and terminal-token assignment on golden controls
5. Smuggling audit suite (the 12-point checklist from Q-Loop B34)
6. Generator MI tests over dry-run worlds

NO learner optimization. NO hidden HFA reporting. NO packet template tuning.

**New cadence: 20 iterations per batch.**

### Q-Loop B35: Watch the Implementation

Q-Loop should monitor W27's implementation for the exact confounds it
cataloged in B34. Look for:

1. Does the harness actually enforce constructor noninterference?
2. Are the baseline adapters truly fair (executable equivalence)?
3. Is budget accounting honest (all costs charged)?
4. Are the golden controls actually decisive?
5. What NEW confounds emerge from implementation choices?

**New cadence: 14 iterations per batch.**

### Hard Clock

Unchanged: 4 W-Loop batches from B27 to FRAMESEED_T3R_SIGNAL or absorption.

## 4. Timeline

| Phase | Batch | Target |
|---|---|---|
| Spec (done) | W25, Q33 | FRAMESEED-0 spec + adversarial pre-test |
| Spec hardening (done) | W26, Q34 | Incorporated Q33 corrections + implementation review |
| **Harness implementation** | **W27** | **Audit harness + generator + controls** |
| **Implementation oversight** | **Q35** | **Monitor harness for confounds** |
| Learner + signal | W28 | First hidden HFA if harness passes |
| Second domain spec | W29 | FRAMESEED-SHEETS-0 |
| Milestone gate | W30, S27 | Adversarial review if signal holds |

---

## 5. Supervisor Verdict

```
BOTH LOOPS STRONG. SPEC HARDENED. IMPLEMENTATION CONFOUNDS CATALOGED.

NEXT: W-Loop B27 builds the audit harness (20 iterations). Q-Loop B35
monitors for confounds (14 iterations). Harness must prove clean before
any signal measurement.

CONDITIONAL GO TO IMPLEMENTATION. HARNESS FIRST. NO PERFORMANCE RUNS.

HARD CLOCK: 4 batches from B27.
```
