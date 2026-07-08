# Dual-Loop Supervisor Check-in #40

**Date: 2026-07-08**
**Reviewing: W-Loop B42 (Eklavya CPU audit) + Q-Loop B50 (direction challenge)**

---

## 1. Loop Convergence

Both loops independently reached the same conclusion: E2-as-KD is stuck in the
proxy/function trap.

- W-Loop: E2 not GPU-ready (comparator missing, tests can't collect). Even if
  fixed, the first burden is oracle ceiling — and Kill #9 showed oracle routing
  couldn't beat single-teacher. Claim ceiling: "permission to run Phase 2."
- Q-Loop: 13-kill failure synthesis reveals a pattern — every killed arc confused
  a measurement surface with the functional geometry behind it. KD transfers
  output distributions (proxy), not functional invariants (function).

## 2. Decision

```
REDIRECT: Eklavya E2 → E3 Functional Teacher Tomography
E2 demoted to instrumentation + absorber baseline
```

E3 core claim: multiple heterogeneous teachers can be used as sensors to infer
compact, student-ownable functional lessons that transfer across hidden
transformations better and cheaper than raw KD and ordinary baselines.

## 3. Mission-Drift Check

- "Does this make intelligence cheaper?" → Not yet. Analysis only. But the
  direction is right: inspectable lesson packets that communities can share.
- "Positive capability or documenting failure?" → Compass work, not miles.
  Next batch must produce runnable code.
- "Killing without proposing?" → Q-Loop proposed 3 directions. No stall.

## 4. Narrative Gate

Headline: "Scientists discover how to teach a laptop AI by studying what expert
AIs disagree about — not by copying their answers."

- Survives "isn't that obvious?" → Yes. KD has been tried for a decade. 13 kills
  prove the obvious approaches fail.
- Survives "that's trivial?" → Only if it reduces to active learning. Burden on
  E3 to show teacher-ecology structure provides something active learning doesn't.
- Mission test → If lesson packets are inspectable, reusable, and shareable, it
  directly serves democratized development.

**Verdict: NARRATIVE ALIVE, UNEARNED.** Zero evidence yet.

## 5. Steering for Next Batch

### W-Loop B43 (10 iterations):
1. Fix CPU gate: restore compare_ablations.py or remove dependency from
   test_eklavya_e2.py so the test suite collects.
2. Design and IMPLEMENT the first E3 toy experiment — a concrete, CPU-runnable
   controlled domain with:
   - 2-3 simple "teachers" (can be rule-based or small synthetic models)
   - Lesson packet construction from teacher disagreement
   - Tiny student training on lesson packets vs raw teacher outputs
   - Hidden transformation test for retained gain
   - Terminal tokens and absorber baselines
3. This is not a protocol document. It is runnable code with a verdict.

### Q-Loop B51 (7 iterations):
1. Sharpen E3: what SPECIFIC functional geometry claim must E3 make?
2. Design the absorber roster — what ordinary baselines must E3 beat?
3. Find the cheapest falsification: what kills E3 fastest?
4. Cross-domain analogy search: where has "inferring structure from
   multi-instrument disagreement" worked outside ML?
5. Attack: is "teacher tomography" just active learning with extra steps?

## 6. Repo State

Post-cleanup: 28 code files, 6 Eklavya docs, core infrastructure, audit trail.
No dead weight. Hygiene pass not due until after B44.
