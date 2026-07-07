# Supervisor Check-in #16: PCCP-H Convergence Assessment

**Date:** 2026-07-07
**Batches reviewed:** W-Loop B18 (PCCP precommit spec) + Q-Loop B25 (PCCP adversarial stress test)
**Context:** Cross-loop synthesis — spec written, then attacked

---

## What Was Produced

### W-Loop B18: PCCP Precommit Spec v0 (1129 lines)
- Formal artifact tuple (L, P, I, O, C, V, R, M)
- Exact verifier contract (finite worlds, no proxy metrics)
- 10 strong baselines (memorization, decision tree, CEGIS, ILP, symbolic regression, SAT/SMT, DreamCoder, reconstruction compressor, tiny neural, random program)
- Smuggling controls checklist (10 risks)
- Compression metric (MDL over AST length)
- 3-variable separation theorem sketch
- Precommitted verdict tokens (PCCP_SIGNAL, STRONG_PCCP, MOONSHOT_PCCP, KILL_PCCP, VOID)
- Balanced criterion (f) — substrate-neutral
- 4-phase roadmap (PCCP-0 → PCCP-1 → PCCP-2 → PCCP-B)

### Q-Loop B25: Adversarial Stress Test (1639 lines, I169-I175)
- Prior-art audit: CEGIS, ILP, DreamCoder, symbolic regression, causal discovery, proof-carrying code, spec mining, exact query learning, computational mechanics, AIXI, active inference, category theory, reservoir computing
- Competing directions scored on 6 criteria
- PCCP evolution: pure PCCP-A demoted → PCCP-H (hybrid verifier-centered) becomes mainline

---

## Cross-Loop Synthesis

The loops converged productively. B18 wrote a rigorous spec. B25 stress-tested the underlying direction. The tension:

| B18 assumed | B25 challenged |
|---|---|
| PCCP-A is the mainline | PCCP-A is just program synthesis under a given verifier |
| Novel direction | Every technical organ is prior art (CEGIS, ILP, DreamCoder, etc.) |
| Non-neural core | Neural perception/proposal may be necessary; anti-NN bias corrected |
| Verifier discovery deferred to PCCP-B | Must include a mini-gate NOW or direction is just formal tools |
| Toy benchmark sufficient | Must include scaling, neural-tool baselines, decomposition |

**Resolution: PCCP-H.** The spec is sound but needs upgrades per B25's 12-point list.

---

## What B25 Killed

1. **Pure PCCP-A as standalone paradigm** — too close to existing synthesis
2. **"PCCP is a new algorithm"** — every piece exists in prior art
3. **Anti-neural bias** — corrected per user directive
4. **"All intelligence decomposes into verifiers"** — likely false in strong form
5. **Toy formal puzzle as sufficient for moonshot claim**

## What Survived

1. **Function-first discipline** — the kill history's core lesson
2. **Executable, checkable, repairable artifacts** — strongest 5-outcome answer
3. **PCCP-H** — hybrid verifier-centered, prior art as baselines, neural where helpful
4. **Verifier discovery as moonshot extension** — must be grounded, not vague
5. **Precommit discipline** — spec-before-implementation is correct

## Narrative Verdict

**Surviving headline:** "A laptop AI learned the rulebook, checked its own work, and fixed the broken rule without retraining. The surprising part was not that it used programs. The surprising part was that it beat the usual AI AND the usual program synthesis tools under tests it had never seen."

- Survives "obvious?" — Only if it beats CEGIS/ILP/DreamCoder, not just neural
- Survives "trivial?" — Only with smuggling controls, scaling, and verifier discovery
- **NARRATIVE ALIVE but requires beating prior-art synthesis, not just neural**

---

## Directives

### W-Loop B19: Upgrade Precommit Spec

The spec is good but B25 identified gaps. W-Loop B19 should upgrade `research/PCCP_PRECOMMIT_SPEC.md` with:

1. **Neural-tool baseline** — tool-using neural agent with ordinary tests/verifiers
2. **Hybrid evaluation** — score pure PCCP, hybrid PCCP-H, and neural-tool by 5 outcomes
3. **Verifier discovery mini-gate** — restricted task where system induces or refines at least one verifier/property
4. **Decomposition gate** — messy task where system proposes partial verifiers and residual uncertainty
5. **Scaling gates** — explicit variation dimensions (graph size, DSL size, intervention families, rule interactions)
6. **Prior-art novelty declaration** — explicit acknowledgment of what is NOT new
7. **Rename strongest candidate to PCCP-H** in the spec

Items already in the spec that B25 confirmed are solid: smuggling controls, CEGIS/ILP baselines, theorem target, verdict tokens, criterion (f).

### Q-Loop B26: Theory Grounding

With PCCP-H stabilized, the question loop should shift from "kill the direction" to "what is the theory?" Specifically:

1. What is the formal relationship between PCCP-H and MDL/algorithmic information theory?
2. Can the 3-variable separation theorem be strengthened or generalized?
3. What class of functions admits verifier discovery? Is there a formal boundary?
4. What does computational mechanics (epsilon-machines, causal states) contribute to the causal compression part?
5. Is there a rate-distortion-style bound: "for target function complexity C, the shortest PCCP artifact has length >= f(C)"?
6. What is the formal relationship between PCCP repair locality and proof-theoretic cut elimination?
7. Does the PCCP framework connect to any known impossibility results (Rice's theorem, Gödel limits)?

### Constraints (unchanged)
- CPU only, theory and spec only
- No implementation, no training
- Codex does all work; Claude supervises
- Balanced substrate evaluation

---

## Summary

The dual loop is working. B18 built the foundation (rigorous spec). B25 stress-tested it and forced an evolution from pure PCCP to PCCP-H. The direction is STRONGER for having survived the attack — it's now more honest about prior art, more balanced on substrate, and more grounded about what's actually new.

The possible new contribution is not an algorithm but an artifact contract: compact executable causal structure + proof/test obligations + hidden-intervention survival + local repair + human-labor accounting. Whether this is a paradigm shift or "just good engineering discipline" depends on whether the verifier discovery extension (PCCP-B) can be made to work.

Next: W-Loop B19 upgrades the spec. Q-Loop B26 grounds the theory.
