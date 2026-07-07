# Supervisor Check-in #17: Theory Grounded — Ready for Theorem Work

**Date:** 2026-07-07
**Batches reviewed:** W-Loop B19 (spec upgrade) + Q-Loop B26 (theory grounding)
**Context:** Theory phase complete. Direction stabilized. First proof work next.

---

## What Was Produced

### W-Loop B19: PCCP-H Spec v1 (1346 lines)
- Renamed to PCCP-H throughout
- Added neural-tool agent baseline (Section 6.11)
- Added hybrid substrate shootout (Section 6A)
- Added decomposition gate (Section 3.9)
- Added scaling gates (Section 3.10)
- Updated verdict tokens, smuggling controls, prior-art declaration, narrative gate

### Q-Loop B26: Theory Grounding (1532 lines, I176-I182)
- **PCCP-H = interventional semantic MDL**: not a new algorithm, but a principled restriction of AIT to verified interventional function preservation
- **Three-part theorem package identified:**
  1. Observational-equivalence impossibility (obs can't distinguish interventionally different SCMs)
  2. Nuisance-entropy rate-distortion gap (reconstruction spends rate on noise, PCCP stays short)
  3. Restricted verifier-discovery theorem (exact-learnable verifier classes give PCCP-B tractable foothold)
- **Theory map**: MDL/AIT for compression, rate-distortion for bounds, causal inference for identifiability, computational mechanics for quotienting, exact learning for verifier discovery, proof theory for certificates, complexity theory for limits
- **Honest position**: "the theory grounds the artifact contract, not the acronym"

---

## Supervisor Assessment

### The Direction Is Stabilized

After 3 Q-Loop batches (B24→B25→B26, 21 iterations) and 2 W-Loop batches (B18→B19), the direction has converged through genuine adversarial refinement:

| Phase | What happened | What survived |
|---|---|---|
| B24: Proposal | PCCP proposed as pure non-neural | Function-first executable artifacts |
| B25: Attack | Pure PCCP-A killed, prior art exposed | PCCP-H (hybrid, honest about prior art) |
| B26: Theory | Theory grounded as interventional MDL | Three-part theorem package |

The direction is no longer changing each iteration. It's time to prove things.

### What's Right

1. **"Interventional semantic MDL" is a real theoretical identity.** It's not a collage of buzzwords. It has a specific formal meaning: compress by functional equivalence under intervention, not by surface reconstruction.

2. **The three-part theorem is provable.** Part 1 (observational equivalence) follows from causal identifiability. Part 2 (nuisance gap) follows from rate-distortion with explicit coding assumptions. Part 3 (verifier discovery) follows from exact learning theory for restricted classes. None require new mathematics — they require careful assembly.

3. **The honest framing is strong.** "PCCP-H is not a new algorithm, it's an artifact contract for interventional functional compression" — this is harder to dismiss than "we invented a new AI paradigm."

### What's Still At Risk

1. **The theorem hasn't been proved yet.** The 3-variable sketch from B18 is an example, not a proof. The coding model needs to be made precise. The nuisance lower bound needs a formal information-theoretic argument.

2. **Verifier discovery remains the moonshot gap.** The exact-learning foothold is clean for restricted classes, but open-world verifier discovery is unaddressed. This is where the paradigm shift lives or dies.

3. **Prior-art baseline comparison hasn't been done.** We haven't actually run CEGIS/ILP/DreamCoder against the PCCP benchmark. The theoretical claim needs empirical backing in a toy world.

### Narrative Verdict

**Current honest headline:** "We proved that surface-optimized AI provably discards what matters for the function, and that a tiny proof-carrying program preserves it. The theory explains 10 failures and predicts a new kind of artifact."

- Survives "obvious?" — Yes, IF the theorem is proved with precise coding assumptions
- Survives "trivial?" — Only with strong baselines and honest coding model
- **NARRATIVE ALIVE and strengthening.** The theory gives the story teeth.

---

## Directives

### W-Loop B20: Prove the Theorem

The spec and theory are ready. The next work-loop task is the FIRST MATHEMATICAL WORK:

1. **Formalize Part 1: Observational-Equivalence Impossibility.**
   - Define two SCMs that are observationally equivalent but interventionally different
   - Prove that no function of the observational distribution alone can distinguish them
   - This is the easiest part and the hardest to knock down

2. **Formalize Part 2: Nuisance-Entropy Rate-Distortion Gap.**
   - State the coding model precisely (not "m-bit budget" — exact encoder class)
   - Prove the lower bound: reconstruction-optimal encoding spends rate on nuisance
   - Prove the upper bound: PCCP artifact length is O(K(F) + decode overhead), independent of m
   - Tight gap: Omega(m) separation in artifact length

3. **Write as a self-contained theorem document** (`research/PCCP_THEOREM_DRAFT.md`)
   - Definitions, lemmas, theorem statements, proofs
   - Each claim must be either proved or marked as [CONJECTURE: requires X]
   - The 3-variable construction is an example, not the theorem

4. **CPU-only. Pure mathematics. No code yet.**

### Q-Loop B27: Fresh-Eyes Consolidation

The Q-Loop has been attacking PCCP for 3 batches. Time for a different kind of scrutiny:

1. **Read the ENTIRE trajectory** (B24→B25→B26, all supervisor check-ins, the spec)
2. **Ask: are we fooling ourselves?** Have we converged too fast? Did adversarial attacks actually change anything or just adjust rhetoric? Is the surviving claim genuinely different from "use formal methods + program synthesis"?
3. **Ask: is this still the biggest swing?** The manifesto says "paradigm-shifting or nothing." Is interventional semantic MDL a paradigm shift or a useful insight? What would make it paradigm-level?
4. **Ask: what are we NOT seeing?** Three batches of internal argumentation can create blind spots. What would an outsider with zero context notice that we've missed?
5. **Stress-test the three-part theorem from a mathematician's perspective.** Is it provable as stated? Are the assumptions hiding something? Is the nuisance lower bound tight?

### Constraints (unchanged)
- CPU only, theory and proof only
- No implementation, no training
- Codex does all work; Claude supervises

---

## Summary

The dual loop has produced a stabilized direction (PCCP-H = interventional semantic MDL) with a clear theoretical identity, an honest prior-art position, and a three-part theorem package ready to be proved. This is the strongest the project has been since the paradigm-open reset.

Next phase: prove the theorem (W-Loop B20) while a fresh-eyes batch checks for blind spots (Q-Loop B27).
