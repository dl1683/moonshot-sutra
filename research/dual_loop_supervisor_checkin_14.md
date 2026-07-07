# Supervisor Check-in #14: Cross-Board CTI Death Certificate

**Date:** 2026-07-07  
**Loops reviewed:** W-Loop B16 (Board 1), W-Loop B17 (Board 2), Q-Loop B21, Q-Loop B22  
**Verdict:** CTI smooth power law `D(C) = D_inf + k*C^(-alpha)` is **DEAD** across both test domains  

---

## 1. The Evidence — Both Boards Failed

### Board 1: Modular Arithmetic (mod-97, random 1M transformer)
- **Verdict:** `PROXY_ONLY_LAW`
- **CTI power-law MAE:** 0.226 — beaten by b3_proxy_only (0.194)
- **Winner prediction:** ALL forecasters predicted quarter_data; actual winner was label_only
- **Root cause:** Grokking — double phase transition (D_func 0.99→0.44→0.64→0.007) is fundamentally incompatible with smooth monotone power law

### Board 2: SmolLM2-135M LoRA MCQ (HellaSwag/PIQA/ARC-Easy)
- **Verdict:** `PROXY_ONLY_LAW`
- **CTI power-law MAE:** 0.087 — beaten by b1_linear (0.077), b3_proxy_only (0.075)
- **Winner prediction:** CTI predicted label_only (WRONG); b1 and b3 predicted single_teacher (CORRECT)
- **Root cause:** Memorization trap — label_only memorized training data (100% train, 43% held-out at step 3000), D_func WORSENED with more compute

### Cross-Board Pattern
| Property | Board 1 | Board 2 |
|---|---|---|
| Verdict | PROXY_ONLY_LAW | PROXY_ONLY_LAW |
| CTI beat baselines? | No | No |
| CTI predicted winner? | No | No |
| D_func monotone? | No (grokking) | No (memorization trap) |
| Proxy-function aligned? | Only early | Diverged after step 100 |

**The smooth power law was beaten on BOTH a grokking-dominated task AND a monotone-looking task.** The "monotone task" wasn't even monotone — label_only shows compute hurts function. The CTI law form has no domain where it works.

---

## 2. What Codex's Own Skepticism Caught vs Missed

### Caught (correctly):
- B22 (Q-Loop) prophetically identified grokking as a confound in I145 — before Board 1 results confirmed it
- B22 correctly proposed PIVOT-CONTINUE: kill smooth law, continue as regime-aware classification
- B17 correctly identified label_only as memorization trap
- All pre-committed verdict tokens applied honestly — no goalpost moving

### Missed:
- **Neither loop predicted that the "monotone" Board 2 would ALSO be non-monotone.** The entire Board 2 design assumed MCQ would be a clean monotone domain to "rescue" CTI from grokking. It wasn't. label_only WORSENED with compute. This is a deeper failure than grokking — it means the smooth monotone assumption is broken for standard supervised learning too.
- **D_gap early warning was identified (B22) but not pre-committed as a formal forecaster.** If it had been, it would have beaten CTI: D_gap at step 100 for label_only was 0.274 (growing), for single_teacher was 0.233 (similar), for shuffled_labels was 0.049 (spuriously low because train_obj uses shuffled labels). By step 300, label_only D_gap was 0.52 — screaming memorization.

---

## 3. Narrative Gate (MANDATORY)

**Honest one-sentence gossip-magazine narrative given ONLY what survived:**

*"We tried to find a universal law that predicts how AI performance improves with more compute — and proved it doesn't exist in the simple form everyone assumed, because more training can actually make models WORSE."*

**"Isn't that obvious?" test:** Partially — practitioners know overfitting exists. But the specific claim that a smooth power law governs D_func(compute) IS taken seriously in the scaling-laws literature (Chinchilla, etc.). Showing it fails even as a forecaster on two very different domains, beaten by trivially simple baselines, has some bite.

**"That's trivial" test:** FAILS. This is a negative result. We proved something doesn't work. That's a paper, not a moonshot. "Scaling laws don't hold at small scale" is not the kind of headline that goes viral. There's no David-beats-Goliath story here. There's no "you won't believe" moment. The strongest version is "we discovered that more training makes AI dumber" — but that's just overfitting, which everyone already knows.

**Narrative verdict: DEAD.** The science is real but the story is not a moonshot. CTI as a smooth universal law is falsified, but that falsification doesn't serve the manifesto's goal of making intelligence cheap and democratic. It's a contribution to the scaling-laws literature, not a paradigm shift.

---

## 4. What's Still Alive

Despite the smooth law dying, the DATA from both boards reveals genuinely interesting phenomena:

### 4a. Memorization Traps Are Detectable
- Board 2 label_only: D_gap grows from 0 (step 10) → 0.27 (step 100) → 0.52 (step 300) → 0.57 (step 3000)
- Board 1 quarter_data: memorized fully, never generalized
- D_gap at step 100 is a cheap early warning signal that a training run will waste compute

### 4b. KD Prevents Memorization Collapse
- Board 2 single_teacher: held-out accuracy stays at 56.25% from step 100 through step 3000, while label_only collapses from 56% to 43%
- The teacher signal regularizes against memorization. This is KNOWN but the specific "KD as memorization insurance" framing connects to the manifesto.

### 4c. Regime-Aware Classification (B22's Pivot)
- Classify training curves into regimes (monotone, grokking, memorization-trap, proxy-only) using early features (D_gap, proxy slope, etc.)
- Predict WHICH regime a training run is in, rather than extrapolating a scalar law
- This survives the Board 2 data: label_only would be classified "memorization-trap" by D_gap features

---

## 5. Supervisor Decision

### Kill: CTI as smooth universal power law
The law form `D(C) = D_inf + k*C^(-alpha)` is dead. 3 boards (B14 salvage, Board 1, Board 2), 3 failures. No domain where it works. No narrative that makes this a moonshot.

### Assess: Regime-aware classification pivot
The B22 pivot has a thread: cheap early features (D_gap, proxy slope at step 100) predict whether to continue, stop, or change strategy. But the narrative is "a better heuristic for when to stop training" — useful, not paradigm-shifting. Fails Invariant #1 (swing for home run).

### The real question for Q-Loop B23:
We've now killed Eklavya (kill #9) AND CTI smooth law. We're burning through directions. The Q-Loop must step back to the manifesto level:

1. **Is there a version of CTI that IS a moonshot?** Not "predict D_func from compute" (dead) but something deeper — a phase-transition theory, an information-theoretic bound, a thermodynamic identity?
2. **Or should we pivot entirely?** The CLAUDE.md lists 5 Nobel-track directions. CTI was #1. Renormalization was #2. CDMD was #4. Are those still alive?
3. **What did we LEARN from CTI's death?** The key lesson: proxy metrics diverge from function. Training loss goes down while performance goes down. This is the memorization-trap insight. Is THAT the moonshot seed — not predicting compute scaling but detecting and preventing compute waste?

The regime-aware classification can continue as a low-cost exploration (CPU-only, small experiments) but is NOT the moonshot direction. The Q-Loop needs to find what IS.

---

## 6. Directives for Next Batches

### Q-Loop B23 (7 iterations, direction-critical):
- MANDATORY: step back to manifesto/paradigm level
- Attack: is there ANY version of compute-thermodynamics that's a moonshot, or should we pivot to renormalization/CDMD?
- The memorization-trap detection finding is a component, not the moonshot — what moonshot does it feed?
- Score every proposed direction on narrative: can a gossip-magazine reader understand and feel the wow?
- Bring the kill history: Eklavya (kill #9), CTI smooth law (kill #10). What pattern are we missing?

### W-Loop B18 (10 iterations, depends on B23):
- HOLD until Q-Loop B23 gives direction
- If B23 says "try regime-aware": implement CPU-only classification experiment on existing Board 1+2 data (no new training)
- If B23 says "pivot": design the new direction's first cheap experiment
- ALL experiments CPU-only, small scale

### Constraint:
- **CPU ONLY, small experiments** (user directive 2026-07-07)
- No GPU chunks. Keep everything fast and cheap.

---

## 7. Commit Log

| Batch | Commit | Verdict |
|---|---|---|
| W-Loop B15 (CTI-0) | 7d58490 | CTI_SALVAGE_INFORMATIVE |
| Q-Loop B21 | ef5db0d | 7 attacks on CTI design |
| W-Loop B16 (Board 1) | 5311d78 | PROXY_ONLY_LAW |
| Q-Loop B22 | 00bb371 | PIVOT-CONTINUE |
| W-Loop B17 (Board 2) | 8f5c373 | PROXY_ONLY_LAW |
| Supervisor #14 | (this commit) | CTI smooth law DEAD |
