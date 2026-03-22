# Sutra Scratchpad - Strategic Test Queue

Every experiment must target a diagnosed bottleneck. No random testing.
Format: Mechanism -> What bottleneck it solves -> How we'd know it worked -> Priority.

---

## CURRENT STRATEGIC DIRECTION: Shared-Core Cognitive Spectrum (Codex R4 2026-03-22)

**CRITICAL FRAMING (User Insight 2026-03-22):** Don't converge too early. The v0.7.0 spec and v1.0 roadmap are HYPOTHESES, not commitments. Each warm-start step is a Chrome probe that earns the right to proceed. If the data contradicts the plan, we pivot. Keep the search space wide.

**CODEX R4 (Tesla+Leibniz) DESIGN: "Shared-Core Cognitive Spectrum Sutra"**
- R4 confidence: 7/8/7/7/6 (up from R3's 6/7/6/6/5)
- Full spec: results/tesla_leibniz_r4_output.md
- 8-step warm-start roadmap: v0.6.0b-rd12 -> v0.6.1-rd12 -> v0.6.2-modes -> v0.6.3-armt -> v0.6.4-corexfer -> v0.6.5-corelite -> Recurrence Gate -> v0.7.0-clean1024
- THE GATE still holds: D>1 must beat D=1 at matched compute on simplified core.

### v0.6.0b-rd12 Implementation Plan (NEXT GPU TASK after 20K)

**Purpose:** Random-depth training to fix early-pass collapse. P0 intervention.

**Model changes (launch_v060a.py):**
- Add `n_steps` parameter to `forward()`: `def forward(self, x, y=None, n_steps=None, ...)`
- Default `n_steps = self.max_steps` (backward compatible)
- Change loop: `for p in range(n_steps):`
- Allocate history for `n_steps` passes, not `self.max_steps`
- Logits computed from mu at the SAMPLED final pass (not always pass 11)

**Training changes (train_v060a.py -> train_v060b.py):**
- Sample D per batch: `P(D=d) = d^alpha / sum_i i^alpha` for d in {1..12}
- Alpha schedule: ramp from 1.0 to 2.0 over first 1K steps
- `L_total = L_final + 0.20 * L_probe` (L_step = 0)
- L_final = CE(logits at sampled depth D, y)
- L_probe still uses mu_hist but only for visited passes
- Warm-start: load v0.6.0a weights, RESET optimizer (WSD restart)
- LR: fresh WSD schedule, 500-step warmup, 3K total steps

**Success criteria:**
- late_pct drops from 91.5% to <70%
- Passes 1-6 contribute materially to BPT
- Final BPT matches or beats source checkpoint (7.14)
- Greedy trigram diversity improves over 0.265

**Pre-training gate:** Codex Correctness + Performance + Scaling review loop BEFORE launch.

**Exact code changes (pseudocode for Codex audit):**

```python
# launch_v060a.py — forward() signature change
def forward(self, x, y=None, n_steps=None, ...):
    if n_steps is None:
        n_steps = self.max_steps  # backward compatible

    # History allocation uses n_steps, not self.max_steps
    if collect_history:
        mu_hist = torch.zeros(B, T, n_steps, self.dim, ...)
        pi_hist = torch.zeros(B, T, n_steps, N_STAGES, ...)

    for p in range(n_steps):  # was: range(self.max_steps)
        ... # same body, unchanged

    # Logits from mu at sampled final pass
    final = self.ln(mu)
    logits = F.linear(final, self.emb.weight) / math.sqrt(self.dim)

    aux = {
        "mu_hist": mu_hist,
        "compute_cost": torch.tensor(float(n_steps), ...),
        "avg_steps": n_steps,
        ...
    }

    # L_step computation skipped when n_steps < self.max_steps
    # L_probe targets adjusted for visited passes only

# train_v060b.py — random depth sampling
def sample_depth(max_d, alpha):
    """P(D=d) = d^alpha / Z for d in {1..max_d}"""
    weights = torch.arange(1, max_d + 1, dtype=torch.float) ** alpha
    return torch.multinomial(weights, 1).item() + 1

# Training loop
alpha = 1.0 + min(step / 1000, 1.0)  # ramp 1.0 -> 2.0 over 1K steps
D = sample_depth(12, alpha)
logits, aux = model(x, y=y, n_steps=D)
L_final = CE(logits, y)
L_total = L_final + 0.20 * L_probe  # NO L_step
```

**CODEX R2 (Tesla+Leibniz) REVISION:** The 7-stage bank is dead. Replace with 1 shared block + lightweight control. Random-depth is P0 (not adaLN). The recurrence gate: D>1 must beat D=1 at matched compute on a simplified shared-core model, or recurrence dies at 68M.

**USER PRIORITY (2026-03-22): PRECISE-GENERAL MEMORY SPECTRUM.** The architecture needs a CONTINUOUS retrieval spectrum from general reasoning (distributed, multi-pass) to precise recall (exact fact lookup). Knowledge-task benchmark gap (SciQ 25.9% vs Pythia 74%, LAMBADA ~1% vs 32.6%) is the strongest competitive weakness. Probe D confirmed current scratchpad is imprecise workspace only. The model must learn WHEN to go precise vs general — this is content-dependent and connects to elastic compute (Outcome 5). This is a FIRST-CLASS DESIGN CONSTRAINT for Round 3 and beyond.

### STRATEGIC PRINCIPLE: Top-Down + Bottom-Up (User Insight 2026-03-22)

**Two complementary processes, neither alone is sufficient:**

1. **TOP-DOWN (primary):** Design the ideal Sutra v1.0 architecture from first principles. What properties must it have? What does it look like unconstrained by current code? This is the DESTINATION. Use Tesla-style mental modeling — fully map the ideal system before building.

2. **BOTTOM-UP (supporting):** The Leibniz research loop discovers mechanisms and validates them through Chrome probes. But each mechanism is evaluated against "does this move us toward the v1.0 vision?" — not just "does this improve current BPT?"

3. **THE BRIDGE:** Reverse-engineer the path from v0.6.0a → v1.0 as a sequence of warm-started incremental versions. Each step is validated, each builds on the last, and each moves toward the ideal architecture.

**The Leibniz loop is a TOOL, not the goal.** It informs the vision and validates specific mechanisms. But the architecture vision drives the priorities, not the other way around.

### v1.0 DESTINATION (Codex Tesla Session 2026-03-22) — HYPOTHESIS, NOT PLAN

**Multiscale Asynchronous Belief Graph at 0.8B-1.2B params.** Current 68M = mechanism incubator.

**Working roadmap (subject to revision at every probe):** v0.6.1→0.6.5 (anti-collapse) → v0.6.6-7 (successive refinement + SwiGLU) → v0.7.0 (stable core) → v0.7.1-3 (route/content split, 2-level scratchpad, calibrated uncertainty) → v0.8.0 (async halting) → v0.8.1-2 (multiscale + scale jump to 135M-350M) → v0.9.0 (latent comm + ABI freeze) → v0.9.1 (tokenizer) → v0.9.2 (active compression) → v1.0 (online active compression + competitive scale)

**8 Properties we're EXPLORING (non-negotiable as goals, negotiable as implementations):** explicit uncertainty, content/control separation, successive refinement, sparse exact recall, revision under contradiction, elastic async compute, stable module ABI, function-preserving growth.

**Full details in RESEARCH.md under "Codex Tesla Session."**

### STRATEGIC PRINCIPLE: Iterative Warm-Start Evolution (User Insight 2026-03-22)

**Do NOT jump from v0.6.0a to v0.7.0 in one step.** Decompose architectural changes into a sequence of small warm-started steps (v0.6.1 → v0.6.2 → ... → v0.7.0). Each step:
- Warm-starts from the previous version's best checkpoint
- Adds exactly ONE mechanism change
- Trains for fewer steps (maybe 3-5K, not 20K) — just enough to validate the mechanism helps
- If it helps: keep and move to next step. If it hurts: revert and try alternatives.

**Why this works:** The v0.1→v0.5.4 progression used exactly this strategy and reached 5.25 BPT. v0.6.0a trained from scratch and is still at 7.54 at 9K. Warm-start evolution is faster because:
1. Each step starts from compressed knowledge (not random init)
2. First few K steps after warm-start have HIGH information gain (model learning new mechanism's effect)
3. Effectively resets the power-law learning curve at each generation — always on the steep part
4. Lower risk — isolate which changes help vs hurt

**Constraint on v0.7.0 design:** PREFER additive changes (new params initialized randomly on top of existing weights) over destructive changes (reshape existing params, break weight compatibility). This means:
- adaLN pass conditioning: ADDITIVE (new LN params, existing weights unchanged) ✓
- Orthogonal BayesianWrite: ADDITIVE (modifies update rule, weights compatible) ✓
- z-loss + WSD: ADDITIVE (loss/schedule change, no weight change) ✓
- Vocab change: DESTRUCTIVE (reshapes all embeddings) ✗ — defer or do last
- SwiGLU: PARTIALLY DESTRUCTIVE (changes FFN shape) — needs careful warm-start strategy

Codex Leibniz Rounds 1+2 (2026-03-22) produced the v0.7.0 spec below.

**Cross-cutting principles:**
1. Preserve geometry under repeated composition
2. Separate content from control
3. Compress before communicating
4. Favor depth over width
5. Use weak auxiliaries, not dominant ones
6. Keep coarse semantics and fine specialization separate

---

## v0.7.0 SPEC (Codex Round 2 Approved)

**Goal:** Targeted simplification + anti-collapse repair. Recover trainability while preserving late-pass behavior.

### P1: Pass-Conditioned GatedRMSNorm/adaLN + Sequence RoPE + QK-norm
**Bottleneck**: Shared-parameter fixed-point collapse.
**Mechanism**: adaLN/GatedRMSNorm conditioned on pass index (proven by TMLT). Sequence RoPE replaces learned pos embeddings. QK-norm for stability.
**Why not dual-axis RoPE**: Pass index is computation-time control, not spatial geometry. Different objects shouldn't be encoded the same way. adaLN is proven; dual-axis RoPE is untested.
**Warm-start**: ADDITIVE — new LN params initialized, existing weights unchanged. ✓
**Chrome Probe A**: Baseline vs adaLN-only vs adaLN+RoPE+QK-norm. Gate on BPT, pass cosine collapse, trigram diversity.

### P2: Conflict-Aware BayesianWrite
**Bottleneck**: Monotone lambda = confidence only increases. Can't reduce confidence on conflicting evidence.
**Mechanism**: Agreement raises precision, conflict/novelty can decay it. Separate parallel refinement from orthogonal correction. Bounded forgetting (not unbounded).
**Novelty**: Genuine gap — no prior work on orthogonal STATE updates in iterative LMs (only weight matrices).
**Warm-start**: ADDITIVE — modifies update rule, weights compatible. ✓
**Chrome Probe B**: Monotone vs conflict-aware. Kill if lambda oscillates, NaNs, or late-pass contribution drops.

### P3: Param-Matched SwiGLU StageBank
**Bottleneck**: SiLU FFN is weaker than SwiGLU (universal evidence).
**Mechanism**: Replace 768→1536→768 SiLU with param-matched 768→1024 SwiGLU.
**Warm-start**: PARTIALLY DESTRUCTIVE — FFN shapes change. Needs careful init strategy.
**Chrome Probe C**: SiLU vs SwiGLU, gate on BPT + throughput.

### P4: Remove Pheromone
**Bottleneck**: Positive feedback is wrong sign for an attractor problem. Not load-bearing.
**Mechanism**: Delete pheromone injection/update from forward pass.
**Warm-start**: ADDITIVE (removal). ✓

### P5: WSD + z-loss + Pass-Collapse Metrics
**Bottleneck**: Cosine schedule may be suboptimal. z-loss prevents logit drift. No current collapse monitoring.
**Mechanism**: WSD schedule, z-loss term, log pass-to-pass cosine contraction + lambda decay stats + router logit norms.
**Warm-start**: ADDITIVE (training recipe change). ✓
**Chrome Probe D**: 2×2 canary for cosine/WSD × z-loss on/off.

### P6 (PROBE-ONLY): Passwise Successive Refinement
**Bottleneck**: Early passes contribute almost nothing (BPT 20-22 for passes 0-7, then dramatic drop in 8-11).
**Mechanism**: Force early passes to do coarse compression via progressive targets (from Fractal Embeddings insight).
**Chrome Probe E**: Short branch comparing current L_step vs coarse-target early passes.

### DEFERRED TO v0.8.0+
- Micro-experts (KILLED — zero evidence below 100M)
- Latent communication bottleneck
- Tokenizer redesign (recurrence changes the math)
- More than 12 passes
- CTI controller / active compression (instrument in v0.7.0, promote in v0.8.0 if calibration probes pass)

### IMMEDIATE INFERENCE PATCH: Anti-Repetition Decoding
**Status**: Repetition penalty + n-gram blocking implemented in gen_quality_test.py. LZ penalty recommended for evaluation.

---

## KILLED DIRECTIONS (do not revisit)

- Full-token MTP at 68M
- Massive flat MoE replacing 7-stage semantics
- Treating Muon optimizer as the main fix
- Directly importing MLA/NSA/Lightning wholesale
- z-loss as the repetition SOLUTION (it's stability, not cure)
- Label smoothing, dropout, stronger exact intermediate CE
- FP8/GQA before fixing writer/routing/state collapse
- Grokfast (overfits at dim=768)
- Syndrome scratchpad (rho=-0.002, no signal)
- Resonant write dither (0% effect)
- L_regret (7.8% reduction, below 30% threshold)
- Full-vocab CE intermediate loss (catastrophic — collapses late pass improvement to 3.5)
- Dense baseline comparison (irrelevant — compete against best-in-class, not vanilla)

---

## CHROME METHODOLOGY

**What transfers across scales**: mechanism class, stability properties, causality.
**What doesn't transfer**: hyperparameter optima, absolute BPT numbers.
**Key learning**: dim=128 Chrome gave false positive for Grokfast. Decision gate is dim=768 canary + generation quality eval.

---

## PRIORITY 7 (NEW): Active Compression — Breaking the Power Law

**Bottleneck**: Power-law learning curve (BPT = 99.5 * step^(-0.283)) means diminishing returns per training step. Later samples give less information because the model treats all data identically regardless of what it already knows.
**Thesis**: Power laws are a SYMPTOM of passive architectures, not a law of nature. An architecture that measures its own uncertainty and steers learning toward high-information-gain samples should learn at a constant or accelerating rate.
**Mechanism candidates**:
- Lambda/precision as a training-time signal: high lambda (confident) = downweight gradient, low lambda (uncertain) = upweight gradient
- Per-stage uncertainty: route training compute to stages with highest uncertainty
- Information gain estimation: measure BPT reduction per sample, prioritize samples that reduce BPT most
- Self-paced curriculum driven by the model's own probe loss, not external heuristics
**Success signal**: Learning curve exponent improves (from -0.283 toward -0.5 or better). Same BPT reached in fewer steps.
**Failure signal**: Model avoids hard examples (takes the easy way out). Uncertainty estimates are miscalibrated → steers learning wrong.
**Connection to Sacred 5**: Serves Outcomes 1 (intelligence), 2 (improvability — the model improves its own learning), 4 (data efficiency).
**Both-sides check**: Self-paced learning literature shows models CAN game their own difficulty metric. Requires well-calibrated uncertainty (Moonshot 3). Curriculum learning results are mixed — sometimes hurts. BUT: those results are for external curricula on passive models. An architecture that natively measures information gain per pass is different from bolting curriculum on top.
**Status**: User insight (2026-03-22). Needs Codex design review and literature survey on information-gain-driven training.

---

## PRIORITY 8 (NEW): Random-Depth Training — Fixing Gradient Structure

**ROOT CAUSE OF COLLAPSE (derived 2026-03-22):** L_final contributes 97-99% of gradient signal at ALL passes. L_step at coef=0.25 contributes only 1-3%. Since all passes share weights, they ALL optimize for final-pass output. Early passes converge to safe fixed-points because they can't improve final-pass output yet.

**Mechanism:** Sample training depth D from distribution each step. Run only D passes. Compute L_final on pass D output. When D=4, pass 3 IS the final pass and gets 100% of L_final gradient.

**Why this is better than adaLN alone:**
- adaLN gives per-pass behavior capacity but doesn't change gradient structure (still 97% from L_final)
- Random-depth changes gradient structure: early passes sometimes get 100% of L_final gradient
- 9.3x increase in early-pass gradient signal (from 1% to 8.3%+ under uniform depth)
- ALSO saves compute: average depth 6.5-9.2 passes (54-77% of current cost)
- Matches Relaxed Recursive Transformer training strategy (randomized loop counts, ICLR 2025)

**Distribution options:**
- Uniform: avg 6.5 passes, 54% compute, equal signal to all passes
- Geometric (alpha=0.85): avg 8.3 passes, 69% compute, biased toward deep
- Deep-biased: avg 9.2 passes, 77% compute, mostly 8-12 with occasional short runs

**Implementation:** ~20 lines of code change to train_v060a.py. Sample D per batch, loop only D times, compute L_final on pass D.

**Chrome Probe:** v0.6.0a + random-depth vs v0.6.0a fixed-12 vs v0.6.1 adaLN vs v0.6.1 adaLN + random-depth. Gate on collapse Gini coefficient, early-pass contribution, total BPT.

**Warm-start:** FULLY COMPATIBLE — no architecture change, only training recipe. Can be applied to any existing checkpoint.

---

## POST-CRITIQUE R2 BRANCH HYPOTHESES (2026-03-22)

These are raw strategic branches, not commitments.

### Track A: Minimal Relaxed-Recursive Decoder

- Strong shared causal block
- `4-8` passes, not `12+` by default
- per-pass `adaLN` or low-rank depth adapters
- scratchpad only if it proves value over the dense control
- no seven-stage ABI claims yet

### Track B: Deep-Thin Dense Control

- Mandatory falsification baseline
- deeper-thinner stack, shared embeddings, `GQA`, `SwiGLU`
- no recurrence ideology
- use it to answer the real question: does recurrence buy anything on this hardware budget?

### Track C: Hybrid Local-State + Exact Memory

- limited recurrence for iterative refinement
- sparse exact long-range recall as the real non-negotiable
- no async / multiscale graph until the simple version wins

### Hard Decision Gate

- A partially shared recurrent model in the `135M-350M` band must beat a deep-thin dense control at matched compute.
- If it cannot, the belief-graph `v1.0` roadmap dies.

### Raw Hypotheses Worth Testing

- Pass disagreement may be more valuable first as a training-time weighting signal than as an inference controller.
- If the current trainer already saturates `24GB` at `68M/12` passes, the first scale fix is probably fewer passes before more parameters.
- Warm-start should be treated as an accelerator only, not a design constraint.

## OPEN QUESTIONS

1. Does the CTI universal law (D(C) = D_inf + kC^(-alpha)) predict per-pass marginal improvement in Sutra?
2. Can progressive prefix supervision from Fractal Embeddings be applied to recurrent passes (force early passes = coarse compressors)?
3. What is the rate-distortion-optimal vocab size at 68M params when accounting for sequence-length inflation?
4. Is attractor collapse detectable via pass-to-pass cosine contraction or Jacobian spectral radius?
5. Does warm-starting v0.7.0 from v0.6.0a work with dual-axis RoPE replacing learned embeddings?
6. Can power-law scaling be broken by architectures that measure and maximize their own information gain per training step? What's the theoretical limit?
7. Is there a provable relationship between calibrated uncertainty and optimal learning rate scheduling?
8. Which v0.7.0 architectural changes are warm-start compatible with v0.6.0a checkpoints? Which require partial retraining? (Critical: v0.7.0 MUST warm-start from v0.6.0a, not train from scratch.)
9. v0.5.4's lower BPT (5.25 at 20K) came from warm-starting from prior versions. v0.6.0a trains from scratch. The compound warm-start strategy: v0.6.0a→v0.7.0→v0.8.0 each warm-starting means each generation builds on compressed knowledge. How do we ensure architectural changes don't break warm-start compatibility?
10. **Is 12 passes the right number?** Collapse metrics show passes 0-5 contribute only 3.1% of total BPT improvement. Truncation probe: 10 passes BPT=18.4, 12 passes BPT=7.6. The model effectively uses only 2-3 passes. Would a model TRAINED with 4-6 passes learn to use each pass better than 12 passes where 10 are wasted? This is a 3x inference speedup if it works. **Key test:** train v0.6.0a from scratch with max_steps=4. Compare BPT at matched compute.
11. **Is pass conditioning (adaLN) the right fix for collapse?** Alternatives to consider: (a) fewer passes, (b) LoRA adapters per pass (rank-4 = 0.44M, cheaper than adaLN), (c) Jacobian regularization forcing passes apart, (d) progressive training (start with 4 passes, add more), (e) partial weight sharing (first/last passes unique, middle shared). The Codex critique loop should evaluate ALL alternatives before we commit.
12. **The fundamental question:** Does recurrence at 68M params provide genuine value over a single-pass model of the same size? The collapse data suggests it might not — at least not with the current architecture. If a 68M single-pass model achieves similar BPT with 12x less inference cost, the entire premise needs revision.
