# T+L Round 2: Ekalavya Protocol Refinement

This is Round 2. Round 1 output is provided below in full. Claude has executed
the research and probe requests from Round 1. Your job: refine the design using
the NEW empirical data, resolve the open questions, and push all 5 outcomes
toward 9/10.

---

## ROUND 1 OUTPUT (COMPLETE — your previous design)

**Phase A — 8 Assumption Challenges:**
1. Warm-start from 60K: YES (confidence 9/10)
2. Qwen3-0.6B as anchor (not Q1.7B): YES, Q1.7B delayed specialist (9/10)
3. All teachers audit every span, but ONE supervises: YES, routed not averaged (9/10)
4. Per-span routing (not per-sample): YES on byte-span scaffold (8/10)
5. Temporal surface separation (state early, logit main, semantic tail): YES (9/10)
6. Multi-depth state primer at relative depths 1/3, 2/3, 1.0: YES, low-weight early-only (7/10)
7. Keep byte spans, add GW for compatibility scoring: YES (8/10)
8. Periodic conflict audit (not full PCGrad every step): YES (7/10)

**Phase C — Full Design:**
- Teachers: Q0.6B anchor, LFM structural, Q1.7B delayed hard-span, EmbeddingGemma semantic
- Routing: per-span, score = pi_t * compat * conf * gap^0.75 * (0.25 + novel)^0.5
- New modules: state hub d=384, semantic hub d=256, analytic router, conflict controller (~7.2M params)
- Schedule: 24K continuation from 60K checkpoint, AdamW, student LR 1e-4, ports 3e-4
- Alpha curriculum: state 0→0.04 (off by 4K), sem 0.01→0.03, q06 0→0.45, lfm delayed to 1K, q17 delayed to 4K
- Conflict: every 250 steps, cos(base,bucket) < -0.10 → multiply alpha by 0.8

**Confidence:** O1=6, O2=8, O3=7, O4=7, O5=7

---

## NEW DATA SINCE ROUND 1

### A. 60K Control Arm Benchmarks (lm-eval)

| Benchmark | Sutra 60K (197M) | Pythia-160M (300B tok) | SmolLM2-135M (2T tok) |
|-----------|------------------|------------------------|-----------------------|
| ARC-Easy | 45.4% | ~43% | ~43.9% |
| ARC-Challenge (norm) | 25.3% | ~25% | — |
| HellaSwag (norm) | 29.0% | ~30% | 42.1% |
| PIQA (norm) | 59.4% | ~62% | 68.4% |
| WinoGrande | 51.3% | ~51% | — |
| SciQ | 67.9% | — | — |
| LAMBADA (acc) | 28.0% | — | — |
| BPT | 3.5726 | — | — |

**Key insight:** We match/beat Pythia on ARC and WinoGrande with 375x less data. HellaSwag (-13pp) and PIQA (-9pp) gaps vs SmolLM2 are the primary targets for Ekalavya. These require commonsense knowledge that only comes from massive data — exactly what teacher models have.

### B. Committee Map Audit (Probe 1 — 50 windows, 16 spans each)

**Route shares with R1 formula:**
- Q0.6B: **87.5%** (700/800 spans) — DOMINATES, formula is biased by pi=0.5
- LFM: 5.1% (41 spans)
- Q1.7B: 7.4% (59 spans)
- EmbeddingGemma: 0.0% (encoder, no logit metrics possible)

**Disagreement:** 14.1% of spans had margin < 20% between top two teachers.

**Compatibility (truncated-dim cosine — UNRELIABLE):**
- Q0.6B: 0.006 (near random)
- LFM: -0.035 (near random, slightly negative)
- Q1.7B: ~0.000 (random)
- EmbeddingGemma: NaN (different forward pass)
- **CONCLUSION: Truncated-dim cosine is useless for cross-architecture compatibility. Need CKA or proper OT.**

**Teacher Confidence (mean max prob on hardest 25% student-entropy tokens):**
- Q0.6B: 0.332
- LFM: 0.321
- Q1.7B: 0.373 (highest — larger model is more confident)
- **CONCLUSION: All teachers have similar confidence. Q1.7B slightly more confident but all in 0.32-0.37 range.**

**KL Gap (mean KL on shared vocab, temp=2.0):**
- Q0.6B: 2.80
- LFM: 2.08 (lowest gap — closest to student)
- Q1.7B: 3.15 (highest gap — most to teach)
- **CONCLUSION: LFM is closest to student distributions. Q1.7B has most knowledge gap. Gap increases later in sequence (more context = more teacher advantage).**

**Student Difficulty by Span Position:**
- Span 0 (start): 3.44 entropy — HARDEST
- Span 8 (middle): 2.44 entropy
- Span 15 (end): 2.29 entropy — EASIEST
- **CONCLUSION: Student struggles most at sequence start (less context). Teachers should help most on early spans.**

**Novelty:** ALL NaN — cross-dimensional cosine between teacher hidden states of different dimensionality produced NaN. Need proper metric (CKA, GW, or matched projection).

**Shared Vocabulary:**
- Q0.6B: 14,822 / 16,000 student tokens (92.6% overlap with Qwen tokenizer)
- LFM: 15,036 / 16,000 (94.0% overlap)
- Q1.7B: 14,822 / 16,000 (92.6% — same as Q0.6B, same tokenizer family)
- **CONCLUSION: >92% vocab overlap with all generative teachers. Cross-tokenizer alignment is NOT the hard problem — cross-architecture representation matching IS.**

**What the committee map tells us about the R1 design:**
1. The routing formula gives Q0.6B 87.5% because pi=0.5 dominates. Need flatter priors or remove prior from formula.
2. EmbeddingGemma cannot participate in logit-level routing. It should ONLY operate on semantic surface (L_sem). Don't try to route it.
3. Compatibility metric is broken. CKA or learned projector similarity needed.
4. LFM has the LOWEST gap — it's actually the closest teacher to the student. This challenges R1's assumption that it's a "structural specialist." It may be a better anchor than Q0.6B for some content.
5. The 14.1% disagreement rate is LOW — teachers mostly agree. Routing may be less critical than expected; the REAL question is whether ANY teacher signal helps.

### C. Cross-Domain Research Synthesis (6 domains, 18 hypotheses)

**5 Converging Meta-Principles (from immunology, neuroscience, physics, economics, ecology, network science):**

1. **Signal Quality Gating (Hard gates, not soft weights)** — Every biological/physical/economic system that integrates multiple signals has a HARD exclusion mechanism. Teachers that hurt should get ZERO gradient, not small gradient. Dual criteria: compatible AND not destructive.

2. **Bottleneck Compression Forces Common Code** — ALL teacher signals should pass through a shared low-dim bottleneck (32-64 dims) before routing. This forces discovery of universal cross-architecture structure. Sources: JAK-STAT (immunology), RG coarse-graining (physics), federated learning consensual representation.

3. **Inverse Effectiveness** — The biggest multi-source gains come where individual signals are WEAKEST. Weight teacher contribution INVERSELY to student competence. Where student is strong → minimal teacher signal. Where student struggles → maximum signal. This is the OPPOSITE of confidence-based weighting.

4. **Partial Consensus is Natural** — Don't require all 4 teachers to agree. 2-3 forming a coherent core per token is the natural state. Router should find and use the coherent core, ignore outliers.

5. **Temporal Staging** — Phase 1 (high T): all teachers on, soft equal weights, student develops basic representations. Phase 2 (cooling): niches discovered from data, routing differentiates. Phase 3 (low T): hard gating, committed niches, strong KD.

**High-Priority Hypotheses:**
- **Inverse effectiveness weighting**: alpha_t(x) = softmax(-student_loss(x) / T). Biggest help where student is weakest.
- **Hard gating by content type**: Binary on/off per teacher per domain. Early = detection mode (all on), late = sampling mode (hard gates).
- **Free energy formulation**: F = sum_t alpha_t * KD_loss_t - T * H(routing). Anneal routing entropy over training.
- **Log pooling of teacher logits**: Average logits (not probs). Theoretically optimal under external Bayesianity.
- **Selection before weighting**: Binary include/exclude gate BEFORE soft weights.
- **Niche partitioning**: Discover teacher specialization from data, then enforce hard boundaries.
- **Quorum sensing**: Strong KD when ≥N teachers agree, weak below threshold. Threshold adapts over training.
- **Shared bottleneck router**: All teacher reps projected into 32-64 dim shared space before routing decisions.

### D. Research Gaps (agents still running — will be available for R3)

The following R1 research requests are still being executed and results will inform R3:
1. GW/Sinkhorn implementation cost for 16x16 spans
2. TinyLLM/TAID/TCS exact mechanics
3. Neuroscience routing analogies (detailed)
4. Biology/ecology analogies (detailed)
5. Physics analogies (detailed)

For R2, use the cross-domain synthesis above (which IS complete) plus the committee map data.

---

## YOUR TASK FOR ROUND 2

Given the new empirical data:

**Phase A**: Re-evaluate your R1 assumptions. Which ones does the committee map data CONFIRM vs CHALLENGE? Specifically:
- The routing formula gives Q0.6B 87.5%. Is this correct or should routing be flatter?
- LFM has the lowest KL gap. Should IT be the anchor instead of Q0.6B?
- Compatibility metric is broken. What replaces truncated cosine?
- EmbeddingGemma can't participate in routing. How should it be handled?
- The cross-domain research suggests inverse effectiveness and hard gating. Do these change your curriculum design?

**Phase B**: What additional probes or research would resolve remaining uncertainty?

**Phase C**: Produce the REFINED design with:
- Updated routing formula incorporating committee map findings
- Integration of cross-domain principles (especially inverse effectiveness, bottleneck compression, hard gating)
- Resolution of the compatibility metric problem
- Updated alpha curriculum based on actual teacher-student gap data
- EVERY implementation detail specified (exact dims, formulas, schedules)

**Confidence scores**: Must cite specific evidence from THIS round's new data. +1 from R1 requires NEW evidence, not design refinement.
