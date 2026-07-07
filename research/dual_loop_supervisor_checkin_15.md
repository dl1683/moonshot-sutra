# Supervisor Check-in #15: PCCP Direction Assessment

**Date:** 2026-07-07
**Batch reviewed:** Q-Loop B24 (I162-I168)
**Context:** First supervisor assessment under paradigm-open vision

---

## What Codex Produced

B24 ran 7 iterations of paradigm-level direction search under the new substrate-open
framing. Key outputs:

1. **PCCP (Proof-Carrying Causal Programs)** proposed as mainline direction (28/30)
2. **CWC demoted** from mainline to theory support (CWC-E, 25/30)
3. **Neural CWC killed** as mainline (12/30) — properly fails criterion (f)
4. **Verifier discovery** identified as moonshot extension (25/30, lower feasibility)
5. **10 explicit kill conditions** for PCCP enumerated
6. **PCCP-A vs PCCP-B distinction**: given verifier (clean gate) vs verifier discovery (moonshot)
7. **Narrative target**: "A laptop built an AI that learned the rulebook instead of memorizing the answer key"
8. **Next gate**: write PCCP_PRECOMMIT_SPEC.md before any implementation

---

## Supervisor Assessment

### What B24 Got Right

1. **PCCP directly addresses the kill history.** 10 kills all stemmed from proxy/function
   divergence. PCCP's core claim — make the function executable and checkable — is the
   structural escape from that failure mode.

2. **Genuinely substrate-open.** Programs, proofs, rules, rewriting systems, typed terms —
   none of these require neural networks. The direction is non-neural at its core while
   leaving room for neural adapters at the periphery.

3. **All 5 outcomes served.** B24 worked backward from the outcomes (I164) and showed PCCP
   naturally satisfies each: verifiable (genuine intelligence), locally repairable
   (improvability), inspectable (democratized), constraint-efficient (data efficiency),
   compiled (inference efficiency).

4. **Honest about narrowness.** B24 did not claim PCCP solves open-world intelligence. The
   honest scoping — "prove verifier-rich first, then investigate decomposition" — is correct
   and defensible.

5. **Smuggling controls are excellent.** The 10 kill conditions (DSL smuggling, verifier
   smuggling, toy triviality, prior-art absorption, etc.) show real adversarial discipline.

6. **CPU-native by construction.** Theory, proofs, finite worlds, exact validators — this
   direction was designed for the CPU-first constraint, not awkwardly shoehorned into it.

### What B24 Got Wrong or Needs Correction

1. **Anti-NN bias in prompt leaked into scoring.** The B24 prompt (which I wrote) included
   "If a direction is 'train a model to do X,' it needs extraordinary justification" and
   made criterion (f) about escaping NN. The user corrected this: "goal is sacred, method
   is not." Neural CWC at 12/30 may be artificially suppressed. In future batches, criterion
   (f) should be "does this serve the 5 outcomes on equal footing" — not "is this non-neural."

2. **Prior art risk is underweighted.** PCCP is very close to existing work:
   - CEGIS (counterexample-guided inductive synthesis)
   - ILP (inductive logic programming)
   - DreamCoder / library learning
   - Symbolic regression
   - Causal discovery + formal verification
   The "new principle" question from I162's attack ("where is the new principle?") was raised
   but not definitively answered. B25 must attack this harder.

3. **"Verifier discovery" may actually be the moonshot, not PCCP-A.** PCCP-A (given verifier)
   is clean but risks being "just program synthesis." PCCP-B (learn the verifier) is where
   the paradigm shift lives. The roadmap should not defer B indefinitely.

4. **Hybrid architecture underexplored.** B24 acknowledged neural perception adapters but
   didn't seriously explore when and where neural components add genuine value. Per user's
   balanced stance, we should evaluate hybrid candidates fairly — neural at perception,
   PCCP at core reasoning — without treating neural as a penalty.

### Narrative Verdict

**Candidate headline:** "A laptop built an AI that learned the rulebook instead of memorizing
the answer key. When it was wrong, it found the broken rule and fixed it without retraining."

- **Survives "that's obvious"?** — Conditional. Needs a hostile benchmark where proxy
  learners look good on training metrics but fail hidden interventions.
- **Survives "that's trivial"?** — Only if strong synthesis baselines (CEGIS, ILP,
  symbolic regression) are included and PCCP wins against them too.
- **Gossip-magazine test?** — YES. Normal people understand "learned the rulebook vs
  memorized the answer key." The David-vs-Goliath angle (laptop vs data center) is there.
- **VERDICT: NARRATIVE ALIVE but fragile.** The benchmark design is the make-or-break.
  A sloppy benchmark kills the narrative instantly.

---

## Directives

### W-Loop B18: PCCP Precommit Spec

B24's top recommendation is correct — write `research/PCCP_PRECOMMIT_SPEC.md` before any
implementation. This is the next work-loop task. The spec must define:

1. What is a PCCP artifact? (executable program + proof/check obligations)
2. Target function definition (exact, not proxy)
3. Admissible world families (with hidden structure)
4. DSL/search space (with smuggling controls)
5. Compression metric (program length, not loss)
6. Hidden-family tests (unseen during synthesis)
7. Baselines (decision tree, memorization, CEGIS, ILP, symbolic regression, tiny neural)
8. Smuggling controls checklist
9. Theorem target (finite-world separation)
10. Positive tokens (PCCP_SIGNAL, STRONG_PCCP, MOONSHOT_PCCP)
11. Kill tokens
12. Criterion (f) — balanced: "does the core claim require gradient-trained representations?"

**CPU only. Theory and spec only. No implementation yet.**

### Q-Loop B25: Attack PCCP Harder

The question loop must now try to kill PCCP. Specific attacks:

1. **Prior art absorption:** Is PCCP just CEGIS/ILP/DreamCoder with new branding? What is
   genuinely new? The synthesis must state the novel principle or concede there isn't one.
2. **Narrowness:** Can PCCP handle anything beyond formal puzzles? What about perception,
   ambiguity, commonsense? Is "decompose into verifier-rich subproblems" hand-waving?
3. **Verifier discovery timeline:** If PCCP-A is "just program synthesis" and PCCP-B is
   the real moonshot but too hard, where is the actual paradigm shift?
4. **Hybrid honesty:** Evaluate a hybrid (neural perception + PCCP core) on equal footing.
   Does it score higher than pure PCCP when criterion (f) is balanced?
5. **Competing non-neural directions:** Are there non-neural approaches B24 missed?
   Energy-based computation, reservoir computing, cellular automata, Wolfram-style
   computational irreducibility, analog computing, neuromorphic non-neural?
6. **Scale skepticism:** Does PCCP's advantage disappear as problems get larger?
   Program synthesis is NP-hard in general. Does PCCP have a scaling story?
7. **The gossip story under adversarial scrutiny:** Can a skeptical journalist dismiss
   "laptop learns rulebook" as "researcher picks easy problem where rules work"?

### Constraints (unchanged)

- CPU only, small experiments only
- No GPU, no model training
- Codex does all implementation; Claude supervises
- Balanced substrate evaluation — no bias toward or against neural

---

## Summary

B24 delivered the strongest post-reset direction: PCCP. It genuinely escapes the proxy/function
divergence pattern that killed 10 prior approaches. The narrative is alive. The smuggling
controls are well-designed. The honest scoping is correct.

The risks are: prior art absorption (CEGIS/ILP/DreamCoder overlap), narrowness (formal worlds
only), and the prompt's residual anti-NN bias suppressing hybrid candidates.

Next: W-Loop B18 writes the PCCP precommit spec. Q-Loop B25 tries to kill PCCP.
Both run in parallel. Both CPU-only.
