MANDATORY FIRST STEP: Read CLAUDE.md in this repository root. Every rule there is binding.
Then read research/VISION.md — especially the "Design Philosophy" section at the top.
Then read research/TESLA_LEIBNIZ_CODEX_PROMPT.md — it defines your role, your questioning principle, your output format, and how you must think. Follow it exactly.
Then read research/ARCHITECTURE.md — it contains the full architecture evolution through 7 rounds.
Then read research/RESEARCH.md — it contains all field research, probe results, and dead ends.

CONTEXT YOU MUST KNOW:
- DESIGN PHILOSOPHY: Nothing is sacred except outcomes. 5 sacred OUTCOMES, mechanisms negotiable.
- We are in Round 7 of the Tesla+Leibniz architecture design loop.
- Round 6 output and R6 stabilization results are the primary evidence.
- THIS IS A CRITICAL DECISION POINT: 3 consecutive hybrid variants have failed stability at 100M. We need to decide whether to continue iterating hybrid blocks or pivot.

---

## YOUR PREVIOUS ROUND'S OUTPUT (Round 6, Summary)

Round 6 prescribed:
1. Remove β entirely first (R6-F), softmax backup (R6-S)
2. Add per-layer telemetry
3. Start full corpus MiniPLM scoring
4. Stop gate iteration at O1=7/10

R6 hypothesis: "The per-channel β vectors are the main instability source."

---

## NEW FINDINGS SINCE ROUND 6

### Finding 1: R6 Stabilization Gate — BOTH VARIANTS FAIL

R6-F (no β, SS-RMSNorm branches) and R6-S (softmax mixing) both run at 24×512, 5K steps.

**Final metrics (step 5000):**

| Model | Params | BPT | kurtosis_max | max_act |
|-------|--------|-----|-------------|---------|
| Transformer 24×512 | 90.2M | **4.6669** | **1.7** | **41.6** |
| R6-F (no β) | 93.4M | 4.7768 | 3.5 | 58.5 |
| R6-S (softmax) | 93.4M | 4.7661 | 2.6 | 55.0 |

**ALL fail success criteria (>=+0.05 BPT, kurtosis<=2.1, max_act<=52).**

**Trajectory pattern:** Both hybrids learn MUCH faster early (R6-F peak +0.54 BPT at step 2000), but the advantage erodes monotonically from step 3000. By step 5000 the transformer wins on BPT by 0.10-0.11.

### Finding 2: ROOT CAUSE IDENTIFIED (Telemetry)

Per-layer telemetry at step 5000 reveals:

| Layer | a_rms (attn proj out) | c_rms (conv proj out) | fused_rms |
|-------|----------------------|----------------------|-----------|
| 0 | 0.236 | 0.085 | 0.364 |
| 1 | 1.095 | 0.148 | 0.224 |
| 22 | 2.174 | 0.465 | 0.063 |
| 23 | 1.413 | 0.398 | 0.145 |

**Diagnosis:**
1. Attention projection output grows 6-8x from layer 0 to 23 (pre-normalization)
2. Conv projection output grows ~5x
3. Fused RMS SHRINKS from 0.36 to 0.08 — branches grow in magnitude but diverge in direction
4. Mean fusion cancels the growing magnitudes, making the block contribution shrink
5. Outlier activations from individual branches before cancellation create kurtosis spikes
6. This compounds across 24 layers

**The projection (256→512) is the amplification channel.** Not β. Not softmax. Not SS-RMSNorm. The projections grow unbounded and the mean of two diverging high-magnitude vectors approaches zero while creating outliers.

### Finding 3: Historical Comparison Confirms Pattern

| Variant | Design | Step 5000 kurt | max_act | BPT vs T |
|---------|--------|---------------|---------|----------|
| P-block (42M) | Full-dim, no proj, fixed 0.5 | 0.82 | 18.15 | — |
| HEMD-R5-G (93M) | Proj + β + RMSNorm | 3.94 | 63.6 | +0.11 |
| R6-F (93M) | Proj + no β + SS-RMSNorm | 3.5 | 58.5 | -0.11 |
| R6-S (93M) | Proj + softmax + SS-RMSNorm | 2.6 | 55.0 | -0.10 |

Only the P-block (full-dim, no projection) was stable. All projected variants show the same kurtosis explosion pattern.

### Finding 4: MiniPLM 500-Window Results (O4)

Diff score: 0.268 ± 0.105. Top-50% threshold: 0.263.
Source ranking: gutenberg > wikipedia > fineweb > math > wildchat > eli5.
Consistent with pilot. Infrastructure validated at scale.

---

## SPECIFIC QUESTIONS FOR THIS ROUND

1. **Should we abandon projected branches?** The 42M P-block (full-dim, no projection) was stable. But full-dim branches cost more params. With GQA + full-dim branches, is the param budget viable at 100M?

2. **Post-fusion normalization as last resort?** Adding norm AFTER mean fusion but BEFORE W_out would control fused magnitude regardless of branch divergence. This was option (c) from R6 evidence doc but never tested.

3. **Weight normalization on projections?** Spectral norm or weight norm on attn_proj/conv_proj to cap projection magnitude growth.

4. **Mixed schedule (few hybrid + many attention)?** If hybrid layers only help early in the network, use them strategically rather than everywhere.

5. **Should we just ship the pure transformer?** We've run 3 hybrid variants at 100M. All fail stability. The transformer is consistently stable and achieves better final BPT. At what point is the hybrid direction empirically dead at this scale?

6. **Is 100M too small for projected branches?** The 42M P-block worked with full-dim branches. At 200M+ with more capacity, projections might have room to learn without destabilizing. Should we test at a larger scale?
