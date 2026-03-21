# Sutra Scratchpad — Strategic Test Queue

Every experiment must target a **diagnosed bottleneck**. No random testing.
Format: Mechanism → What bottleneck it solves → How we'd know it worked → Priority.

---

## PRIORITY 1: Continuous-Spectrum Memory (v0.5.5)

**Bottleneck**: Current scratchpad is gist-only. Can't do exact recall. Router is O(n²).
**Mechanism**: Continuous-resolution memory — learnable zoom from gist to exact per-query.
**Success signal**: Exact-recall probe accuracy >80% at 200+ token distance, AND BPT improves.
**Failure signal**: Collapses to always-coarse or always-exact. Or BPT regresses.
**Status**: Codex designing continuous-spectrum variant (not 3 discrete levels).

---

## SCALE-UP IDEAS (dim=1024) — Bottleneck-Annotated

### Contractive Hyperspherical Core
**Bottleneck it solves**: Training instability at scale. Warm-start fragility (Peri-LN broke warm-start). Grokfast divergence (gradient magnitudes differ across dimensions).
**Why this specifically**: If all vectors are unit-norm, gradient magnitudes are bounded by construction. No more "works at dim=128, diverges at dim=768." Scale-invariant.
**Success signal**: Same hyperparameters work at dim=128, 384, 768, 1024 without tuning.
**Priority**: HIGH — solves our #1 scaling problem (things that work small break large).

### RG/MERA Stage Pyramid
**Bottleneck it solves**: Flat sequence processing. No hierarchical structure. Can't distinguish word-level vs paragraph-level vs document-level patterns.
**Why this specifically**: Language IS hierarchical. Current architecture treats every token at the same scale. Multi-scale processing is the most convergent finding across all 15 research domains.
**Success signal**: Better performance on long-context tasks. Chunk-level coherence improves.
**Priority**: MEDIUM — important for long context, but not our immediate bottleneck.

### Spatially Coupled Predictive-Coding Graph
**Bottleneck it solves**: Scratchpad broadcasts are blurry (averaged state). Router is dense O(n²). Communication carries raw state, not prediction error.
**Why this specifically**: Coding theory proves sparse local iterative = near-optimal. Predictive coding says only errors should propagate. Spatial coupling says suboptimal local becomes optimal global.
**Success signal**: Same BPT with 10x fewer communication bits. OR better BPT with sparse messages.
**Priority**: MEDIUM-HIGH — directly improves the scratchpad/router which are our core modules.

### Grokfast (from scratch only)
**Bottleneck it solves**: Training efficiency. Slow convergence = expensive experiments.
**Why this specifically**: +11% at dim=128 is real. Failed at dim=768 warm-start, but that might be warm-start-specific, not Grokfast-specific.
**Success signal**: Loss improves faster than baseline in first 5K steps from scratch.
**Failure signal**: Diverges again (even from scratch) → mechanism is fundamentally incompatible with Sutra's gradient landscape.
**Priority**: LOW — only test from scratch, not worth warm-start debugging.

---

## MECHANISMS TO RETEST AT SCALE — Strategic Prioritization

### Tier 1: Solves a diagnosed bottleneck
| Mechanism | Bottleneck it solves | Why it might work at 105M | Test order |
|-----------|---------------------|--------------------------|------------|
| **Error scratchpad (delayed)** | Scratchpad carries raw state not errors | Late steps were 2.2x better — needs capacity to exploit | 1st |
| **nGPT hypersphere** | Scale-dependent hyperparameters | Angles/norms are scale-invariant by construction | 2nd |

### Tier 2: Might help but bottleneck is unclear
| Mechanism | What it does | Why it might work | Test order |
|-----------|-------------|-------------------|------------|
| Dendritic neurons | More expressive per-neuron | Model may be capacity-limited at 105M | 3rd |
| Lambda halting | Adaptive compute per token | AUROC was 0.36 — model couldn't learn meta-control at 69M | 4th |

### Tier 3: Unlikely to help — test only if Tier 1-2 exhaust
| Mechanism | Why probably not | Test if |
|-----------|-----------------|---------|
| Complex embeddings | -36% is catastrophic, not scale-dependent | Everything else fails |
| CfC time constants | -14%, theoretical basis is weak for this arch | Domain-specific need |
| Surprise bank | Hurt EVERY arm it touched | Never, unless fundamentally redesigned |

---

## DATA EXPERIMENTS — Strategic

| Experiment | Bottleneck | Success signal | Priority |
|-----------|-----------|----------------|----------|
| **Diverse corpus training** | Academic paper mimicry | Generation uses varied styles/topics | IMMEDIATE (step 15K restart) |
| NCA pre-pre-training | Cold start inefficiency | Faster convergence in first 1K steps | MEDIUM |
| Data mixing optimization | Suboptimal source ratios | BPT improves by adjusting proportions | AFTER diverse baseline |
| Compression ratio as eval | No cheap quality metric | r>0.9 correlation with generation quality | LOW |

---

## CHROME METHODOLOGY — Strategic Improvements

**Problem**: dim=128 Chrome gave false positive for Grokfast (+11% → diverges at 768).

**Fix hierarchy**:
1. **dim=128**: Bug-finding + obvious loser killing only. Never ship from this.
2. **dim=768 canary (200-1000 steps)**: Decision gate for warm-start mechanisms.
3. **Generation eval (10 random questions)**: Decision gate for shipping. BPT is secondary.

**What transfers across scales**: mechanism class (simple shared state > complex control), stability properties, causality, warm-start continuity.
**What doesn't transfer**: hyperparameter optima, absolute BPT numbers, optimizer trick effectiveness.

---

## OPEN QUESTIONS

1. Why does Grokfast diverge at dim=768? Gradient magnitude scaling? Or warm-start specific?
2. Will Net2Net widening actually activate the extra dimensions, or stay dead?
3. The delayed-start principle: is step 3 universal or should the delay adapt per mechanism?
4. Continuous-spectrum memory: can we derive the optimal resolution from information theory?
