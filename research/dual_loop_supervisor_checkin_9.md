# Dual-Loop Supervisor Check-In #9

**Date:** 2026-07-07
**Loops completed:** W-Loop B10 (iterations 91-93, halted by kill), Q-Loop B14 (iterations 92-98)
**Status:** COORDINATE INHERITANCE KILLED. Functional-margin shadow FAILED on all 3 benchmarks. Direction permanently abandoned as primary moonshot.

---

## What Happened

### W-Loop B10: The Decisive Experiment

Codex implemented the functional-margin shadow test into `code/coordinate_inheritance.py`, ran a 2-example dry run (plumbing check), then ran the full 50-example smoke across HellaSwag, PIQA, and ARC-Easy train-safe subsets with all 7 control variants.

**The result was unambiguous:**

| Benchmark | Main inherited | Random core | Gaussian destroyed | Main - Random | Main - Gaussian | Gate |
|---|---:|---:|---:|---:|---:|---|
| HellaSwag | 20.0% | 26.0% | 24.0% | **-6.0pp** | **-4.0pp** | **FAIL** |
| PIQA | 42.0% | 58.0% | 42.0% | **-16.0pp** | 0.0pp | **FAIL** |
| ARC-Easy | 22.0% | 26.0% | 16.0% | **-4.0pp** | +6.0pp | **FAIL** |

**Precommitted verdict: `FAIL_FUNCTIONAL_MARGIN_SHADOW` / `SURFACE_COMPATIBILITY_ONLY`**

The kill condition required main inherited to beat BOTH random AND gaussian by >=+1pp on >=2 of 3 benchmarks. It passed **0 of 3**. Worse: main inherited LOST to random core on all three benchmarks.

This is not a marginal miss. On PIQA, random core beat main inherited by **16 percentage points**. The inherited coordinate directions are actively anti-signal for candidate discrimination.

**Iterations 94-100 were blocked by the kill condition.** No v2 Stage 1, no Stage 2.

### Q-Loop B14: Pre-Positioned for This Outcome

7 iterations attacking margin shadow methodology, endgame probability, pivot targets, sunk cost, and process meta-critique. Key outputs arrived before W-Loop B10 completed:

1. **Gaussian anomaly analysis**: If gaussian >= main, adapted directions carry anti-signal. CONFIRMED by B10 on HellaSwag.
2. **5 bright-line kill conditions**: B10 triggered condition #1 (FAIL on all 3 benchmarks) and condition #2 (gaussian dominance on HellaSwag).
3. **Endgame probability if FAIL**: 0-3% chance of reaching SmolLM2-level. Direction effectively dead.
4. **6 concrete pivot targets**: Functional Margin Distillation, Disagreement Geometry Router, Low-Rank Decision Subspace, Counterfactual Minimal-Pair Curriculum, Error Atlas, Byte-Native Teacher Debate.
5. **Sunk cost audit**: "If every failed gate becomes 'new insight, one more repair,' the loop becomes a sunk-cost machine."
6. **Process critique**: The dual-loop prevents overclaims but may be stuck in local search around one mechanism. Needs broader invention portfolio.

---

## Supervisor Assessment

### The Direction Is Dead. This Is Not Debatable.

The functional-margin shadow was the precommitted decisive test. The result is not merely below threshold — it's **reversed**. Main inherited produces worse candidate discrimination than random layers. The 3-4 nats of "coordinate-specific" NLL lift translates to **negative** benchmark function.

**What this means:**
- The NLL advantage was real but was lexical/manifold/surface compatibility, not task-discriminative function
- The adapter learned to produce Qwen-shaped vectors that minimize token NLL, but those vectors do not help distinguish gold from wrong completions — they actively hurt
- The "coordinate geometry" was a gauge artifact: it looked like structure but carried no downstream information
- Q-Loop B12's most dangerous question — "Does the signal contain task-discriminative function or only lexical/manifold compatibility?" — is now answered: **lexical/manifold compatibility only**

### What Survives From Coordinate Inheritance

1. **The codec** (`codec_phase1.5`) — byte-to-embedding encoder works and is reusable
2. **The benchmark scoring infrastructure** — functional-margin shadow mode in `coordinate_inheritance.py` is a good evaluation tool
3. **The prior-floor discovery** — pretrained layers have a large unconditional NLL floor (33-47%), which is useful knowledge
4. **The dual-loop methodology** — it correctly killed the direction before expensive Stage 2 investment

### What Dies

- Coordinate inheritance as the primary moonshot direction
- Any claim that NLL-based coordinate alignment produces reasoning transfer
- Hidden-state copying as a theory of intelligence transfer
- Further repair cycles on this direction

### Narrative Gate

**One-sentence story:** "We built a careful pipeline to inherit reasoning geometry from a teacher, ran the decisive test, and found that the inherited coordinates were actively worse than random noise at answering questions."

**"Isn't that obvious?"** — No. The NLL signal was large (3-4 nats above destroyed-input floor), consistent across readouts, early-layer-specific, and survived rotation sanity checks. Most researchers would have proceeded to Stage 2 training on that evidence. The functional-margin shadow caught what NLL gates could not.

**"That's trivial?"** — The falsification is not trivial. It proves that NLL-to-benchmark transfer is not automatic even with large, consistent NLL advantages. That's an empirical contribution.

**Narrative verdict:** The honest headline is a kill, not a discovery. "We proved coordinate inheritance doesn't work" is not a moonshot story. The moonshot story must come from the pivot.

---

## The Pivot

### What Q-Loop B14 Recommended

The top two pivots from B14 iteration 96, selected for manifesto alignment:

**Pivot 1: Functional Margin Distillation**
- Train the byte student on teacher pairwise margins (gold vs strongest wrong), not hidden states
- Make decision-boundary geometry the training target from day one
- Story: "Instead of copying a teacher's brain coordinates, Sutra learned the shape of the teacher's hardest decisions."

**Pivot 2: Disagreement Geometry Router**
- Multi-teacher router that trains only on examples where teachers disagree and student is wrong
- Route lessons by disagreement type
- Story: "Sutra got smarter by studying only the fights between its teachers."

### Supervisor's Pivot Assessment

Both pivots serve the manifesto ("Intelligence = Geometry, not Scale") and avoid the exact failure mode that killed coordinate inheritance:

| Property | Coordinate Inheritance (DEAD) | Functional Margin Distillation | Disagreement Router |
|---|---|---|---|
| Primary metric | Token NLL | MCQ margins from day 1 | MCQ margins on disagreement slices |
| What transfers | Hidden-state coordinates | Decision-boundary geometry | Correction geometry |
| Failure mode caught early? | No (NLL hid it) | Yes (margins are the target) | Yes (margins on hard slices) |
| Training required? | No (zero-shot copy) | Yes (but targeted) | Yes (but data-efficient) |
| Aligns with Eklavya? | Weakly (single-teacher copy) | Strongly (margin = function) | Strongly (disagreement = lesson) |

**Both pivots align better with Eklavya's thesis** — "teachers are instruments, not masters; the student learns from their disagreements, not their consensus."

### Pivot Priority

The supervisor recommends:

1. **Functional Margin Distillation** as the first pivot experiment — it directly addresses the failure mode (NLL ≠ function) and can reuse the codec + benchmark infrastructure
2. **Disagreement Geometry Router** as the second experiment — it is closer to the original Eklavya vision and serves the multi-teacher narrative
3. Both experiments MUST start with functional margins as the primary gate, not NLL

---

## Decisions

### D1: Permanently Kill Coordinate Inheritance as Primary Direction

Coordinate inheritance is demoted to **historical diagnostic infrastructure**. No more repair cycles, no Stage 2, no public claims. The code stays for infrastructure reuse; the direction is dead.

### D2: Pivot to Functional-Margin-First Geometry

The next W-Loop batch designs and implements a functional-margin distillation prototype. The training target is benchmark-facing margins from day one.

### D3: Maintain Kill Discipline

Q-Loop B14's process critique is correct: "No mechanism gets a third repair cycle unless it has already produced functional evidence against its strongest controls." The pivot directions get the same adversarial gate chain.

### D4: Preserve What We Learned

The graveyard entry for coordinate inheritance must be precise:
- **Killed by:** FAIL_FUNCTIONAL_MARGIN_SHADOW on all 3 benchmarks (W-Loop B10)
- **Kill evidence:** Main inherited -6pp/-16pp/-4pp vs random on HellaSwag/PIQA/ARC-Easy
- **Root cause:** NLL coordinate alignment produces surface compatibility, not task-discriminative function
- **Lesson:** NLL metrics can show large, consistent, specific signal that has zero or negative benchmark correlation. Always test functional margins before investing in architecture-level escalation.

---

## Confidence Table

| Claim | Confidence | Evidence |
|-------|-----------|---------|
| Coordinate inheritance produces task-discriminative function | **2%** | Main inherited LOST to random on all 3 benchmarks |
| NLL coordinate lift is surface compatibility only | **95%** | 3-4 nats NLL lift → negative benchmark margins |
| Functional margin distillation can work | 40% | No evidence yet, but avoids the NLL-function disconnect |
| Disagreement router can work | 35% | No evidence yet, but aligns with Eklavya thesis |
| Next direction will be the moonshot | 15% | Realistic probability given project history |

---

## Launch Orders

### Q-Loop B15 (Iterations 99-105): Attack the Pivot

**Goal:** Take the pivot targets from B14 iteration 96 and attack them adversarially BEFORE implementation. The new directions must survive hostile review.

**Angles:**
1. Functional Margin Distillation — what are the ways this fails? What if decision margins don't compress well into a byte-native model?
2. Disagreement Geometry Router — is there actually enough teacher disagreement at this scale? What if teachers agree on almost everything?
3. Are we pivoting TO something or just FROM something? Is the pivot principled or reactive?
4. What does the new direction need to show in its FIRST cheap experiment to earn a second?
5. What is the one-sentence story? Does it survive "that's trivial"?
6. The dual-loop process — should we change anything about how we run the loops for the pivot?
7. External competitive landscape — has anyone already done functional-margin distillation or disagreement routing?

### W-Loop B11 (Iterations 101-110): Functional-Margin Distillation Prototype

**Goal:** Design and implement the first pivot experiment. Start with functional margins as the primary target.

**Specific tasks:**
1. Design the margin distillation loss (gold-vs-wrong NLL pairwise ranking)
2. Implement in `code/coordinate_inheritance.py` or a new canonical file (Codex decides)
3. Use existing codec + Qwen teacher + benchmark infrastructure
4. Run a smoke with 50-100 training examples and functional-margin evaluation
5. Precommit verdict tokens before running

**Kill condition:** If margin distillation shows <+3pp functional margin improvement over baseline S0 on >=2 of 3 benchmarks, the direction needs rethinking.

---

## Dual-Loop Status

| Loop | Last Batch | Status | Next |
|------|-----------|--------|------|
| W-Loop | B10 (KILLED direction) | Coordinate inheritance dead | B11: Functional margin distillation pivot |
| Q-Loop | B14 (direction endgame) | Pivot targets prepared | B15: Attack the pivot |

**The dual-loop survived its hardest test: it killed its own favorite idea.** Now it must prove it can invent, not just falsify.
