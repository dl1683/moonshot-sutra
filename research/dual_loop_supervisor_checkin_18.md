# Dual-Loop Supervisor Check-in #18

**Date: 2026-07-07**
**Reviewing: Q-Loop B27 (fresh-eyes consolidation), W-Loop B20 (theorem draft — pending)**
**Status: PARTIAL — B27 assessed, B20 still running**

---

## 1. Q-Loop B27 Assessment

B27 is the sharpest batch this loop has produced. It attacks the process, not
just the content, and reaches an honest verdict: PCCP-H is a valid mainline but
not yet a paradigm shift.

### What Codex Got Right

1. **Trajectory audit (I183)**: 15 concrete substantive changes catalogued from
   B24 to B26. Not rhetorical drift — real narrowing and hardening. The verdict
   ("big swing as doctrine, incomplete as paradigm") is exactly correct.

2. **Outsider simulation (I185)**: The reviewer-reaction table is the most
   useful artifact in B27. Every row is an honest prediction of what each field
   would actually say. The aggregate: "promising integration/evaluation agenda,
   incremental until verifier discovery and baseline-beating evidence exist."
   This is our real position.

3. **Blind-spot identification (I186)**: The before-frame / after-frame split is
   the most important insight in recent batches. PCCP-H deeply attacks "given a
   function and verifier, what should knowledge be?" It barely touches "how does
   a system discover the function, verifier, and intervention grammar?"
   Frame formation is the gap between useful tool and paradigm shift.

4. **Theorem provability (I187)**: Honest grading. Part 1 provable. Part 2
   fragile without a precise coding model. Part 3 restricted to exact learning.
   "The theory is provable in pieces; the paradigm claim is not proved by those
   pieces." Correct.

5. **Minimum viable moonshot (I188)**: Theorem + executable witness + baseline
   parity + one real discovery move. Not theorem alone, not toy alone. This is
   the right formulation of the deliverable.

6. **Narrative attack**: "Gates are good hygiene, but they are not a mechanism."
   This sentence is the most important thing B27 produced. It names the exact
   failure mode the loop has been drifting toward: methodological excellence
   without mechanism.

### What Codex Missed or Understated

1. **The discovery move IS the moonshot.** B27 correctly identifies frame
   formation as the gap, but treats it as one of several things to fix. It is
   the ONLY thing that matters. Everything else — theorem, spec, baselines,
   smuggling controls — is hygiene around the central question: can a cheap
   system discover what to verify? If the answer is no, PCCP-H is a good
   benchmark discipline. If yes, it is a paradigm shift. There is no middle.

2. **Cost of the theorem.** B27 recommends "freeze rhetoric until the theorem
   exists" but does not ask: is the theorem the right next investment? The
   theorem proves that functional compression beats surface compression when the
   function is known. Experts already know this (B27 says so). The theorem is
   necessary for credibility but does not move the paradigm needle. The question
   is whether B20's theorem draft lands cleanly enough to serve as foundation
   without consuming another 3-4 batches of refinement.

3. **Neural-tool baseline.** B27 mentions it but does not stress it enough. If
   GPT-5 + tools + counterexample feedback can solve the same finite worlds as
   PCCP-H under equal information, the paradigm claim is dead regardless of
   theorems. This baseline is existential, not just one of eleven.

### Narrative Gate

**Honest one-sentence narrative given only what survived B27:**

```text
We have a rigorous benchmark discipline for function-preserving intelligence
artifacts, but no mechanism that discovers the function cheaply.
```

Does it survive "isn't that obvious?" — Barely. The benchmark discipline is
unusually hostile and the prior-art integration is thorough. But "use explicit
functions and verifiers instead of proxy losses" is not a paradigm shift.

Does it survive "that's trivial?" — Yes. The spec, smuggling controls, and
baseline parity requirements are genuinely non-trivial. The problem is that
non-trivial methodology is not the same as non-trivial mechanism.

**Narrative verdict: ALIVE BUT FRAGILE.** The narrative becomes unkillable only
when a cheap system discovers a nontrivial verifier or decomposition that no
baseline found, on a problem where the discovery matters.

---

## 2. W-Loop B20 Assessment

**PENDING.** Codex is still writing the theorem draft. Will be assessed when
complete and folded into this document or a follow-up.

---

## 3. Cross-Loop Synthesis

### The State of Play

| Dimension | Status |
|---|---|
| Direction | PCCP-H as interventional semantic MDL — stable, honest |
| Theory | Three-part theorem identified; provability assessed; draft in progress |
| Spec | Hostile, complete, substrate-open |
| Evidence | Zero. No artifact, no baseline comparison, no discovery move |
| Narrative | Alive but fragile — discipline is not mechanism |
| Biggest gap | Frame formation / verifier discovery |

### The Fork

The project is at a fork:

**Path A: Prove the theorem, build the witness, compete baselines.**
This is the safe path. It produces a credible artifact. It may produce a
paradigm-level result if the baselines fail and discovery emerges naturally.
Risk: the theorem is unsurprising, the baselines match, and PCCP-H becomes a
good benchmark paper, not a moonshot.

**Path B: Attack frame formation directly.**
This is the hard path. Skip ahead to the question B27 identified as central:
can a cheap system discover what to verify? This is where the paradigm lives.
Risk: without the theorem and baseline infrastructure, any discovery claim has
no credibility foundation.

**Path C (recommended): Prove Part 1, build the finite witness, AND attack
discovery in parallel.**
The theorem (Part 1 only — observational equivalence) is quick and credible.
The finite witness demonstrates the separation concretely. In parallel, the
Q-Loop attacks frame formation: what is the cheapest mechanism that can propose
a nontrivial verifier clause or decomposition boundary that the system did not
start with?

This is the smallest bet that keeps both the credibility path and the moonshot
path alive.

---

## 4. Directives

### W-Loop B21: Build the finite witness

When B20 completes:
- Assess the theorem draft honestly (which parts proved, which conjectured)
- If Part 1 is clean: commit it and move to the executable witness
- Design the smallest finite world family that instantiates the theorem
- Build the PCCP-0 harness: world generator, DSL, verifier, synthesis engine
- Run CEGIS/ILP/neural-tool baselines under equal information
- This is the first real experiment. Pre-commit verdict tokens.
- CPU only. Finite worlds. No GPU.

### Q-Loop B28: Attack frame formation

The central question:
- What is the cheapest mechanism that can discover a nontrivial verifier clause,
  obligation, metamorphic relation, or decomposition boundary?
- Survey: how do existing systems discover specifications? (Spec mining, Daikon,
  invariant inference, contract inference, property-based testing generators,
  metamorphic relation discovery, causal discovery algorithms)
- Can any of these be composed into a PCCP-B front-end that proposes obligations
  the human did not write?
- What is the smallest demo where system-discovered obligations catch a failure
  that human-written obligations miss?
- Be honest: if this reduces to "run Daikon + property testing," say so.

### Constraints
- CPU only, small experiments only
- No rhetoric about paradigm shift until evidence exists
- Theorem is foundation, not finish line
- Discovery is the moonshot, not compression

---

## 5. Supervisor Verdict

```text
PCCP-H MAINLINE CONTINUES. DIRECTION STABLE. EVIDENCE PHASE BEGINS.
```

The theory loop has done its job: direction is grounded, spec is hostile,
blind spots are named. Now the project needs artifacts, not arguments.

The single most important question for the next 2 batches:

```text
Can a cheap system discover what to verify?
```

Everything else is hygiene around that question.

**Next check-in: after W-Loop B21 + Q-Loop B28.**
