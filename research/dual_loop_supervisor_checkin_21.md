# Dual-Loop Supervisor Check-in #21

**Date: 2026-07-07**
**Reviewing: W-Loop B23 (B2 relation suite), Q-Loop B30 (B2 absorption attack)**

---

## 1. W-Loop B23 Assessment: B2 Absorbed

B23 built the full B2 metamorphic relation discovery suite. Clean execution
across 8 role-permuted worlds.

### Results

| Metric | Value |
|---|---|
| Relation miner finds flip(C0)->NOT | Yes, all 8 permutations |
| Relation miner finds flip(C1)->NOT | Yes, all 8 permutations |
| Relation miner finds flip(C0,C1)->id | Yes, all 8 permutations |
| Relation miner V1 rejects P_bad_B2 | Yes, all 8 permutations |
| Exhaustive MR mining matches | Yes, all 8 permutations |
| B1-only insufficient for covariance | True |
| Clause sets identical | True |
| **B2_VERDICT_TOKEN** | **B2_DISCOVERY_ABSORBED** |

### What B23 Validated

1. B2 is a real level jump — B1-only invariance cannot catch covariance violations
2. The relation miner works correctly: discovers covariance, compiles clauses, rejects shortcuts
3. Exhaustive metamorphic mining finds identical results at identical cost
4. Role permutation control passes — discovery is role-blind across all 8 worlds
5. P_bad_B2 design is honest: passes V0_B2 + B1 invariance, fails only covariance checks

### What B23 Does Not Prove

1. That FDM-0 / Relation Miner has any edge over exhaustive enumeration
2. That B2 discovery matters when transformation grammars are supplied
3. Anything about B3 decomposition or B4 transformation grammar discovery

---

## 2. Q-Loop B30 Assessment: The Sharpest Prediction

B30 predicted B2_DISCOVERY_ABSORBED before B23 finished building the test. This
is the Q-Loop working exactly as designed — direction ahead of implementation.

### Key Findings

| Iteration | Finding |
|---|---|
| I204 | B2 world specified: P_bad=C1, support shortcut, 7,680 queries for exhaustive |
| I205 | Exhaustive fails only for very large T, large d, high arity, or expensive oracle |
| I206 | Active query selection: O(|E|n) vs O(|E|n²) but baseline can do the same |
| I207 | MDL needed for false positives but absorbed — baseline uses same MDL |
| I208 | Composition closure algebraically sound but absorbed — baseline composes too |
| I209 | **Transformation grammar smuggling is the deepest threat** |
| I210 | B2 absorbed, B3 smallest non-absorbed level, B4 is the real problem |

### The Critical Insight (I209)

B30's deepest contribution is I209: the human-supplied transformation grammar IS
the intelligence. When T = {single flips, pair flips} and Phi = {id, NOT}, the
discovery problem reduces to measuring which supplied transforms hold. The real
frame formation problem is DISCOVERING T and Phi, not mining over them.

This reframes the entire absorption ladder:

```text
B1: mine invariances over supplied perturbation grammar — ABSORBED
B2: mine relations over supplied (T, Phi) — ABSORBED
B3: discover decomposition boundaries (T partially implicit) — OPEN
B4: discover transformation grammar T from typed observations — THE REAL PROBLEM
```

---

## 3. Cross-Loop Synthesis

### The Absorption Ladder (Updated)

| Level | Description | Status | Evidence |
|---|---|---|---|
| B0 | Prior art (CEGIS/SyGuS/Daikon) | Absorbed | Theorem Part 3, prior art survey |
| B1 | Single-field invariance | **ABSORBED** | B22: exhaustive single-field screening |
| B2 | Metamorphic relation discovery | **ABSORBED** | B23: exhaustive MR mining matches |
| B3 | Decomposition discovery | **OPEN** | Not yet tested |
| B4 | Transformation grammar discovery | **OPEN** | Not yet tested |
| B5 | Universal frame formation | Impossible | Theoretical |

### What PCCP-H Still Has

Despite B1 and B2 absorption, PCCP-H retains real value:

1. **After-frame separation**: STRONG_PCCP — proved and witnessed
2. **Artifact contract**: compile, verify, repair, audit — substrate-independent
3. **Honest absorption testing**: the methodology itself (precommit verdicts,
   smuggling audits, role permutation, baseline comparison) is original
4. **The theorem**: Parts 1-2 clean, information-theoretic separation

### What PCCP-H Does Not Have

1. A non-absorbed discovery mechanism
2. Evidence that any formal system beats exhaustive enumeration + neural tools
3. A paradigm-level claim beyond artifact discipline

### The Decision Point

The project is at a fork:

**Path A: Build B3 decomposition discovery.**
- Two-component finite world, dependency blocks, local repair
- Could demonstrate that decomposition reduces search/synthesis/repair cost
- Risk: may also be absorbed by sensitivity clustering + exhaustive interaction testing

**Path B: Jump to B4 transformation grammar discovery.**
- The real moonshot according to B30
- System discovers useful transformations from typed observations + weak edit priors
- Risk: may be too hard for the current phase, intractable without strong priors

**Path C: Accept absorption and reposition PCCP-H.**
- PCCP-H becomes the verifier/compiler/audit layer around whatever discovery wins
- Focus on the practical tool: API/contract/IAM/pipeline verification
- The moonshot becomes the methodology, not the discovery mechanism

### Supervisor Assessment

B30 is right: the transformation grammar is the real intelligence. But B30 also
says B4 is "tractable only with explicit priors." We cannot jump to B4 without
first understanding what B3 looks like.

**Recommendation: Path A first (B3), then reassess.**

B3 is the natural next step because:
1. Decomposition discovery is partially about discovering which fields interact,
   which is a structured version of T discovery
2. If B3 is absorbed, the honest conclusion is Path C (reposition)
3. If B3 shows a non-absorbed edge, it opens a path toward B4
4. B3 is implementable on CPU with the existing finite-world infrastructure

---

## 4. Directives

### W-Loop B24: Build the B3 Decomposition Discovery Suite

Build `code/pccp0_b3_decomposition.py`:

1. **Two-Component World**:
   - Component A: causal bits A0, A1; target_A = A0 XOR A1
   - Component B: causal bits B0, B1; target_B = B0 AND B1
   - Combined target: y = (target_A, target_B) or y = target_A XOR target_B
   - Nuisance bits per component, shared spurious bit
   - All fields role-permuted into flat observation vector

2. **Partial Verifier V0_B3**:
   - Checks seen examples, some single-field interventions
   - Does NOT know the component boundary
   - Does NOT check cross-component independence

3. **P_bad_B3**: A program that entangles components — uses A-side fields
   to compute B-side output or vice versa. Passes V0_B3 but fails when
   component independence is tested.

4. **Decomposition Miner**:
   - Estimate first-order sensitivity: which fields affect which outputs
   - Build dependency graph (field → output channel)
   - Cluster into components
   - Compile component-local independence clauses:
     "perturbing field outside component A does not change output_A"
   - V1 = V0 + component independence → reject P_bad_B3

5. **Baselines**:
   a. Exhaustive interaction testing (all field × output pairs)
   b. No discovery (V0_B3 only)
   c. B2-only relation mining (single/pair flips, identity/NOT)
   d. Random field clustering

6. **Pre-committed Verdict Tokens**:
   - `B3_DISCOVERY_SIGNAL`: Decomposition miner finds component boundary
     that catches a hidden failure AND beats exhaustive interaction testing
     on cost, synthesis reduction, or repair locality
   - `B3_DISCOVERY_ABSORBED`: Exhaustive interaction testing matches
   - `B3_SYNTHESIS_VALUE`: Decomposition reduces synthesis search space
     (even if absorbed on clause discovery, decomposed search is cheaper)
   - `VOID`: World too simple, boundary obvious, baselines handicapped

### Q-Loop B31: Attack B3 and Test the Reposition Option

1. Is B3 decomposition discovery absorbed by exhaustive interaction testing
   for small finite worlds? (probably yes — be honest)
2. Does decomposition provide SYNTHESIS VALUE even if clause discovery is absorbed?
   (this is the real question — decomposed search is exponentially cheaper)
3. If B3 is also absorbed, what does Path C (reposition) look like concretely?
4. What is the practical PCCP-H tool? API metamorphic testing + verifier compilation?
5. Is the absorption ladder itself a publishable contribution?
6. What is the honest narrative for the project if B3 is absorbed?
7. Is there a B3.5 between B3 and B4 that is non-absorbed and tractable?

### Constraints
- CPU only, small experiments only
- Honest about absorption at every level
- If B3 is absorbed, the recommendation should be Path C reposition
- The absorption ladder and methodology may be the real contribution

---

## 5. Supervisor Verdict

```text
B1 ABSORBED. B2 ABSORBED. B3 IS THE LAST STAND BEFORE REPOSITION.
```

The project has been extraordinarily honest. Two absorption results in one
session, both predicted by the Q-Loop before the W-Loop confirmed them. The
methodology is working perfectly: precommit, test, absorb, move up.

If B3 is also absorbed, the honest narrative becomes:

```text
We built a proof-carrying causal program framework with a clean theorem,
demonstrated the after-frame separation, and systematically proved that
discovery at levels B1-B3 is absorbed by existing techniques. The framework
survives as a verification/audit discipline. The moonshot discovery claim
requires transformation grammar discovery (B4), which remains open.
```

That is a respectable research contribution. It is not the home run. But it
is honest, and honest negatives published well are more valuable than
overclaimed positives.

**Next check-in: after W-Loop B24 + Q-Loop B31.**
