# Dual-Loop Supervisor Check-in #19

**Date: 2026-07-07**
**Reviewing: Q-Loop B28 (frame formation attack), W-Loop B21 (finite witness)**

---

## 1. W-Loop B21 Assessment: The First Evidence

B21 produced a working executable witness (`code/pccp0_witness.py`, 715 lines)
that demonstrates the theorem-predicted separation on CPU.

### Results

| m (nuisance bits) | PCCP length | Hidden acc | Recon proxy length | Recon proxy hidden | Recon+PCCP length |
|---|---|---|---|---|---|
| 0 | 9 | 1.000 | 4 | fails | grows |
| 4 | 9 | 1.000 | 8 | fails | grows |
| 8 | 9 | 1.000 | 12 | fails | grows |

- PCCP length **constant at 9** across all m (0-8)
- Reconstruction proxy length **grows linearly** with m
- Reconstruction proxy **fails** hidden interventions (uses spurious shortcut)
- Verifier-aware reconstruction control passes but is longer
- **Verdict token: STRONG_PCCP**

### What Codex Got Right

1. Clean DSL with no target-specific primitives (Var, Const, Not, Bin, If)
2. Genuine hidden/seen split with structurally different intervention families
3. Two reconstruction baselines: unfair proxy AND fair verifier-aware control
4. Honest smuggling audit embedded in the code
5. Correct scope limitations: "does not prove verifier discovery, open-world
   frame formation, novelty over CEGIS/SyGuS, or scaling beyond tiny worlds"

### What B21 Does Not Prove

1. That PCCP beats CEGIS/SyGuS/ILP under equal information (they would find
   the same 9-node program)
2. That verifier discovery works (the verifier was human-given)
3. That the separation holds for non-trivial tasks
4. Anything about PCCP-H or frame formation

### Supervisor Verdict on B21

```text
STRONG_PCCP CONFIRMED. The after-frame separation is real. Move on.
```

The theorem is validated by a concrete executable witness. This completes the
after-frame evidence. The project can now cite: "In finite worlds with nuisance
entropy m, functional compression stays O(K(F)) while surface reconstruction
grows Omega(m)." This is proved and demonstrated.

---

## 2. Q-Loop B28 Assessment: The Moonshot Pivots

B28 is the most important Q-Loop batch in the project's history. It converts
the abstract "frame formation" gap into a concrete mechanism (FDM-0), an
assumption ladder (B0-B5), and an honest prior-art/neural-tool threat model.

### Key Contributions

1. **Prior-art survey (I190)**: Daikon, QuickCheck, metamorphic testing, spec
   mining, causal discovery, contract inference, ICE/Horn learning — all mapped
   to PCCP-B. Verdict: they solve parameter discovery inside a frame, not frame
   formation itself.

2. **FDM-0 algorithm (I191)**: Active effect screening — generic perturbations,
   paired traces, effect signatures, clause grammar, MDL scoring. Concrete and
   implementable on CPU.

3. **Smuggling controls (I192)**: 10 controls. "A clause is not discovered
   because it appears in output." Role permutation, name randomization, negative
   controls, cross-encoding checks.

4. **Assumption ladder (I193)**: B0 (prior art) through B5 (impossible). The
   sweet spot is B1 (role discovery) and B2 (metamorphic relation discovery).
   This is the honest scope.

5. **Neural-tool baseline (I194)**: "GPT-5 + tools may already be the best
   frame-formation engine." If true, PCCP-H becomes compiler/audit layer. This
   is the existential baseline.

6. **Smallest demo (I195)**: Concrete design — partial verifier V0 missing
   spurious-invariance, P_bad exploiting S, FDM-0 discovering the missing
   obligation, V1 catching the failure.

7. **Verdict tokens (I196)**: FRAME_SIGNAL, DISCOVERY_SIGNAL, DISCOVERY_ABSORBED
   — clean precommit.

### What Codex Missed

1. **Cost model for FDM-0.** The algorithm is described but the query complexity
   is not bounded. For B1-level role discovery in finite worlds, the cost is
   O(fields * values * examples). This should be stated.

2. **The composition question.** B28 focuses on single-field invariance. Real
   discovery needs composed relations (e.g., "if C0 changes, the output XORs").
   The jump from B1 to B2 is where the real difficulty lives.

### Narrative Gate

**Honest one-sentence narrative given only what survived B21 + B28:**

```text
A tiny proof-carrying program stays short while the surface memorizer grows,
and the project now has a concrete algorithm to discover the missing rules.
```

Does it survive "isn't that obvious?" — The separation result does for experts.
The FDM-0 algorithm is not yet demonstrated. So: half alive.

Does it survive "that's trivial?" — The FDM-0 design could reduce to
perturbation testing, which would be trivial. Not yet tested.

**Narrative verdict: ALIVE — conditional on FDM-0 beating prior-art baselines.**

---

## 3. Cross-Loop Synthesis

### The State of Play

| Dimension | Status |
|---|---|
| Direction | PCCP-H as interventional semantic MDL — stable |
| Theory | Three-part theorem proved (Parts 1-2 clean, Part 3 restricted) |
| Spec | Hostile, complete, substrate-open |
| Evidence | STRONG_PCCP on finite witness (after-frame separation) |
| Frame formation | FDM-0 designed, not yet built |
| Narrative | Conditional on FDM-0 results |
| Biggest gap | Does FDM-0 beat prior-art and neural-tool baselines? |

### Phase Transition

The project has completed the THEORY phase:
- Theorem proved
- Spec written
- After-frame separation demonstrated

The project now enters the DISCOVERY phase:
- Build FDM-0 into the witness
- Test against Daikon/metamorphic/causal baselines
- Test against neural-tool baseline
- Measure absorption

---

## 4. Directives

### W-Loop B22: Add FDM-0 to the witness

Extend `code/pccp0_witness.py` with the Frame Discovery Module:

1. Start with partial verifier V0 that deliberately omits spurious-invariance
2. Generate P_bad that passes V0 by exploiting S
3. Implement FDM-0: generic single-field perturbations, paired-trace effect
   estimation, clause mining, MDL scoring
4. FDM-0 proposes O_new (spurious-invariance obligation)
5. V1 = V0 + O_new rejects P_bad
6. Add baselines: random clause search, exhaustive single-field invariance check
7. Add role permutation across worlds (fields shuffled, no role labels)
8. Report: does FDM-0 discover the obligation? Does it transfer across worlds?
9. Pre-commit: FRAME_SIGNAL, DISCOVERY_ABSORBED, VOID tokens

### Q-Loop B29: Test the existential threats

The two biggest threats from B28:
1. "Discovery is trivial" — perturbation testing already does this
2. "Neural-tool baseline absorbs PCCP-B"

This batch should:
- Design the strongest Daikon/metamorphic/perturbation-testing baseline
- Design the strongest neural-tool baseline protocol
- Ask: what would FDM-0 need to do that NEITHER of these can?
- Survey: are there real-world finite-domain tasks where frame formation
  matters and is not currently solved?
- Design the B2-level experiment: metamorphic RELATION discovery, not just
  single-field invariance

### Constraints
- CPU only, small experiments only
- No rhetoric beyond what evidence supports
- Discovery is the moonshot, not compression

---

## 5. Supervisor Verdict

```text
THEORY PHASE COMPLETE. DISCOVERY PHASE BEGINS.
```

The after-frame story is proved and witnessed. PCCP-H is credible as an
artifact contract. The project's future depends entirely on whether FDM-0
discovers useful obligations that prior art and neural tools cannot.

**Next check-in: after W-Loop B22 + Q-Loop B29.**
