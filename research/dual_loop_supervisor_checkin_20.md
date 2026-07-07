# Dual-Loop Supervisor Check-in #20

**Date: 2026-07-07**
**Reviewing: W-Loop B22 (FDM-0 witness), Q-Loop B29 (existential threats)**

---

## 1. W-Loop B22 Assessment: The Honest Result

B22 extended the finite witness with FDM-0 — the Frame Discovery Module designed
in B28. The experiment ran cleanly across 8 role-permuted worlds with m=4
nuisance bits.

### Results

| Metric | Value |
|---|---|
| FDM-0 finds spurious field | Yes, all 8 permutations |
| FDM-0 V1 rejects P_bad | Yes, all 8 permutations |
| FDM-0 V1 accepts true program | Yes, all 8 permutations |
| Exhaustive single-field matches | Yes, all 8 permutations |
| Random clause baseline success rate | 62-69% |
| Role permutation control | True (S at 5 distinct positions) |
| **FDM_VERDICT_TOKEN** | **DISCOVERY_ABSORBED** |

### What Codex Got Right

1. Clean separation: FDM-0 sees only x0..xN, no role labels. Role maps are
   post-hoc audit data only.
2. Honest comparison: exhaustive single-field checking runs on exactly the same
   data and budget.
3. Smuggling audit extended correctly: FDM-0 does not see role labels, the
   perturbation grammar is generic (single-field binary replacement), baselines
   get equal information.
4. The MDL scoring and shortcut_MI correctly identify S as the unique spurious
   candidate (stable_score=1.0, shortcut_MI=1.0), distinct from nuisance fields
   (stable=1.0, shortcut_MI=0.0) and causal fields (stable=0.4).

### What B22 Does Not Prove

1. That FDM-0 adds anything over exhaustive perturbation testing at B1.
2. That discovery works for relations, compositions, or decompositions.
3. That the result would differ with a richer transformation grammar.
4. Anything about the neural-tool baseline.

### Supervisor Verdict on B22

```text
DISCOVERY_ABSORBED CONFIRMED. B1 is testing hygiene, not the moonshot.
```

This is the expected result. B28 predicted it (I197) and B29 proved it
theoretically. The executable witness now demonstrates it empirically. FDM-0
at B1 is engineering, not discovery.

**The one positive note**: FDM-0's MDL scoring distinguishes spurious from
nuisance fields (spurious_candidate vs stable_invariant). Exhaustive checking
does not make this distinction without the shortcut_MI metric. This is not
enough for FRAME_SIGNAL — both methods reject P_bad equally — but it is a
real feature that may matter at B2 where false positives multiply.

---

## 2. Q-Loop B29 Assessment: The Sharpest Batch Yet

B29 ran 7 iterations (I197-I203) testing existential threats. This is one of
the most honest Q-Loop batches in the project.

### Key Findings

| Iteration | Finding | Impact |
|---|---|---|
| I197 | B1 discovery absorbed by O(n*\|E\|) exhaustive screen | Concede B1 |
| I198 | Live claim starts at B2 (metamorphic relations) | Refocuses project |
| I199 | Neural-tool baseline protocol (NTB-0) fully designed | Existential test ready |
| I200 | Real finite domains exist (APIs, contracts, IAM, pipelines) | Application targets identified |
| I201 | B2 experiment designed: relation miner with T and Phi | Next build target |
| I202 | B3 decomposition: first paradigm-level target | Longer-term goal |
| I203 | Final verdict: B1 absorbed, B2 live, B3 open | Clear roadmap |

### What Codex Got Right

1. **I197 is devastating and honest.** The exhaustive single-field algorithm is
   specified precisely with query complexity. No hand-waving. FDM-0 at B1 is
   absorbed, period.

2. **I198 correctly identifies the boundary.** The table of what's trivial vs.
   nontrivial (unary covariance vs. pairwise relations vs. conditional
   invariances vs. decomposition boundaries) is the clearest decomposition of
   the discovery hierarchy the project has produced.

3. **I199 designs NTB-0 precisely.** The sealed bundle, equal-information table,
   forbidden information list, and win/lose criteria are implementable. This is
   not "someday we should test against neural tools." This is a runnable
   protocol.

4. **I200 grounds the application.** Software protocols, smart contracts, IAM,
   data pipelines — these are bounded finite domains where intervention queries
   are cheap and discovered obligations compile into tests. Not medical judgment
   or legal reasoning.

5. **I201 designs the B2 experiment concretely.** Relation Miner v0: enumerate
   input transforms T × output transforms Phi, score by exact paired-label
   agreement, MDL, negative controls, hidden transfer. Query complexity
   O(|E| * n² * d² * |Phi|) for k=2 interactions.

6. **The narrative attack is self-aware.** "The hard part was that the human
   supplied a valid perturbation API and a target oracle. Once those exist,
   discovery is ordinary testing."

### What Codex Missed

1. **No explicit B2 world design.** I201 sketches the relation miner algorithm
   but doesn't fully specify the two-component or covariance-testing world.
   This is needed for W-Loop B23.

2. **The third threat (I203 final section) deserves more weight.** The
   human-written perturbation grammar may be the real intelligence. Even if
   FDM-0 beats NTB-0 at B2, the victory doesn't transfer if humans smuggled
   the transformation grammar. The information ledger separating field/relation/
   transformation/output/decomposition/goal discovery is crucial.

3. **No query-efficiency hypothesis.** B29 identifies that FDM-0 might have an
   edge through active query selection, but doesn't propose what that active
   policy would look like. For B2, the exhaustive baseline is O(|E|*n²*d²*|Phi|),
   which is still small for finite worlds. The active edge only matters if
   worlds are large enough for brute force to hurt.

### Narrative Gate

**Honest one-sentence narrative given only what survived B22 + B29:**

```text
The laptop found the missing rule, but so did a for-loop — the real question
is whether it can find rules that a for-loop and a tool-using genius both miss.
```

Does it survive "isn't that obvious?" — Yes, the absorption concession is not
obvious to most AI researchers who haven't thought about discovery baselines.

Does it survive "that's trivial?" — The current RESULT is trivial (B1 absorbed).
The QUESTION is not trivial (can B2/B3 discovery survive?).

**Narrative verdict: THE QUESTION IS ALIVE. THE RESULT IS NOT YET.**

---

## 3. Cross-Loop Synthesis

### The State of Play

| Dimension | Status |
|---|---|
| Direction | PCCP-H as interventional semantic MDL — stable |
| Theory | Three-part theorem proved (Parts 1-2 clean) |
| Spec | Hostile, complete, substrate-open |
| After-frame evidence | STRONG_PCCP on finite witness |
| B1 discovery | ABSORBED by exhaustive perturbation testing |
| B2 discovery | Designed, not yet built |
| B3 discovery | Open, potentially paradigm-level |
| Neural-tool baseline | Protocol designed (NTB-0), not yet run |
| Biggest gap | Does B2 relation discovery survive absorption? |

### Phase Status

The project completed THEORY phase (supervisor #19) and is now in DISCOVERY
phase. The first discovery result (B1) is absorbed. The live discovery claim
must now demonstrate B2 or B3.

### The Absorption Ladder

```text
B0: prior art (CEGIS/SyGuS/Daikon) — absorbed
B1: single-field invariance — ABSORBED (B22 confirmed)
B2: metamorphic relation discovery — LIVE, UNTESTED
B3: decomposition discovery — OPEN, POTENTIALLY PARADIGM-LEVEL
B4: open-world discovery — UNSOLVED
B5: universal frame formation — IMPOSSIBLE
```

The honest claim is narrow: PCCP-H needs to show that B2 relation discovery
is not absorbed by exhaustive metamorphic relation mining or NTB-0.

---

## 4. Directives

### W-Loop B23: Build the B2 Absorption Suite

Build a new experiment file `code/pccp0_b2_relations.py` (or extend the witness)
that tests metamorphic relation discovery:

1. **B2 World**: Same base structure (2 causal, m nuisance, 1 spurious) but the
   missing obligations are RELATIONS, not just invariances:
   - `flip(C0) -> NOT(y)` — covariance
   - `flip(C1) -> NOT(y)` — covariance
   - `flip(C0, C1) -> identity(y)` — pair composition
   - `change(S) -> identity(y)` — invariance (absorbed by B1)

2. **Partial Verifier V0**: Checks seen examples and some basic interventions.
   Deliberately omits covariance and composition obligations. V0 allows P_bad
   (a program that satisfies seen cases but has wrong covariance properties).

3. **Relation Miner v0**: Extend FDM-0 with:
   - Input transform grammar T: single-field flips, pair flips
   - Output relation grammar Phi: {identity, NOT} for binary output
   - Score: exact paired-label agreement, MDL, negative controls
   - Compile discovered relations into verifier clauses

4. **Baselines**:
   a. Exhaustive metamorphic relation mining over same T × Phi
   b. No discovery (just V0)
   c. Random relation search

5. **Pre-committed verdict tokens**:
   - `B2_DISCOVERY_SIGNAL`: Relation miner finds a relation that catches a hidden
     failure, AND exhaustive metamorphic mining does NOT find an equal/better
     clause at comparable cost, OR the miner has a measured cost/transfer advantage
   - `B2_DISCOVERY_ABSORBED`: Exhaustive metamorphic mining matches
   - `VOID`: Task too small, grammar answer-shaped, or baselines handicapped

6. **Role permutation, smuggling audit, narrative gate**: mandatory as before.

### Q-Loop B30: Design the B2 World and Attack the Relation Miner

Focus on ensuring the B2 experiment is not trivially absorbed:

1. What B2 world would make exhaustive relation mining expensive but targeted
   relation mining cheap? (Large T but sparse true relations)
2. What P_bad passes V0 but has wrong covariance? Design it explicitly.
3. Can active query selection reduce Q_B2 compared to exhaustive? When?
4. Is there a B2 world where MDL scoring matters (many spurious relations hold
   by finite coincidence)?
5. What is the smallest B2 world where FDM-0 has a cost advantage over
   exhaustive metamorphic mining?
6. How does the B2 result change if the output relation grammar Phi is unknown?
7. What is the gossip-magazine sentence for a positive B2 result?

### Constraints
- CPU only, small experiments only
- The B2 experiment must be honest about what the human supplied (T, Phi, oracle)
- No rhetoric beyond what evidence supports
- If B2 is also absorbed, say so and move to B3

---

## 5. Supervisor Verdict

```text
B1 ABSORBED. MOVE TO B2. THE MOONSHOT LIVES OR DIES ON RELATION DISCOVERY.
```

The after-frame story is proved. The B1 discovery story is honestly absorbed.
PCCP-H's value as an artifact contract (compile, verify, repair) remains intact
regardless. But the MOONSHOT claim — cheap formal discovery that beats brute
force and neural tools — needs B2 at minimum.

The project is in a healthy but precarious position: honest, well-tested, with
clear next steps, but the live claim is getting narrower with each absorption.
If B2 is also absorbed, the honest move is:
- PCCP-H becomes the compiler/audit layer
- Discovery substrate is whatever wins (exhaustive mining, neural tools, or novel)
- The paradigm-level claim requires B3 decomposition

**Next check-in: after W-Loop B23 + Q-Loop B30.**
