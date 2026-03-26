# T+L Round 7 Evidence Brief

## R6 Stabilization Gate — BOTH VARIANTS FAIL

### Final metrics (step 5000)

| Model | Params | BPT | kurtosis_max | max_act |
|-------|--------|-----|-------------|---------|
| Transformer 24x512 | 90.2M | **4.6669** | **1.7** | **41.6** |
| R6-F (no β, SS-RMSNorm) | 93.4M | 4.7768 | 3.5 | 58.5 |
| R6-S (softmax mix, SS-RMSNorm) | 93.4M | 4.7661 | 2.6 | 55.0 |

**Success criteria: >=+0.05 BPT, kurtosis<=2.1, max_act<=52. ALL FAIL.**

Transformer WINS on both BPT and stability. Both hybrids end BEHIND the transformer.

### Full R6-F trajectory

| Step | R6-F BPT | T BPT | Delta | R6-F kurt | T kurt | R6-F max_act | T max_act |
|------|----------|-------|-------|-----------|--------|-------------|-----------|
| 500 | 7.84 | 8.41 | +0.57 | 0.8 | 0.6 | 34.9 | 14.7 |
| 1000 | 7.02 | 6.86 | -0.16 | 0.4 | 0.6 | 23.4 | 15.7 |
| 1500 | 5.81 | 6.33 | +0.52 | 0.5 | 0.3 | 21.0 | 17.6 |
| 2000 | 5.49 | 6.03 | +0.54 | 0.8 | 0.3 | 27.0 | 18.4 |
| 2500 | 5.23 | 5.68 | +0.45 | 0.6 | 0.4 | 25.1 | 21.6 |
| 3000 | 5.13 | 5.30 | +0.17 | 1.3 | 0.8 | 38.2 | 29.8 |
| 3500 | 5.04 | 5.41 | +0.38 | 2.0 | 1.3 | 42.4 | 33.6 |
| 4000 | 5.03 | 5.05 | +0.01 | 2.2 | 1.7 | 47.2 | 42.4 |
| 4500 | 4.91 | 4.86 | -0.05 | 3.0 | 1.4 | 53.4 | 39.7 |
| 5000 | 4.78 | 4.67 | -0.11 | 3.5 | 1.7 | 58.5 | 41.6 |

### Full R6-S trajectory

| Step | R6-S BPT | T BPT | Delta | R6-S kurt | T kurt | R6-S max_act | T max_act |
|------|----------|-------|-------|-----------|--------|-------------|-----------|
| 500 | 8.78 | 8.41 | -0.37 | 1.1 | 0.6 | 37.7 | 14.7 |
| 1000 | 7.20 | 6.86 | -0.34 | 0.3 | 0.6 | 22.5 | 15.7 |
| 1500 | 6.00 | 6.33 | +0.33 | 0.4 | 0.3 | 23.3 | 17.6 |
| 2000 | 5.76 | 6.03 | +0.27 | 0.6 | 0.3 | 27.0 | 18.4 |
| 2500 | 5.53 | 5.68 | +0.15 | 0.8 | 0.4 | 32.2 | 21.6 |
| 3000 | 5.61 | 5.30 | -0.31 | 1.0 | 0.8 | 32.9 | 29.8 |
| 3500 | 5.22 | 5.41 | +0.19 | 1.2 | 1.3 | 33.9 | 33.6 |
| 4000 | 5.19 | 5.05 | -0.14 | 2.4 | 1.7 | 50.3 | 42.4 |
| 4500 | 5.02 | 4.86 | -0.16 | 2.1 | 1.4 | 48.2 | 39.7 |
| 5000 | 4.77 | 4.67 | -0.10 | 2.6 | 1.7 | 55.0 | 41.6 |

### Key observations

1. **Both hybrids learn MUCH faster early** — R6-F peaks at +0.54 BPT at step 2000
2. **Advantage erodes monotonically** starting around step 3000
3. **By step 5000, transformer wins** — hybrids end ~0.10-0.11 behind
4. **R6-S has better mid-training stability** (kurt 1.2 at step 3500 vs R6-F 2.0) but still fails
5. **Pattern identical across N, F, S blocks** — the instability is NOT from β, softmax, or branch norm choice
6. **R6-S softmax barely learned asymmetry** — alpha ranged 0.495-0.527, essentially 0.5 everywhere

## ROOT CAUSE: Branch Projection Magnitude Accumulation

### Telemetry diagnosis (R6-F at step 5000)

Per-layer branch RMS from telemetry:

| Layer | a_rms (attn proj) | c_rms (conv proj) | fused_rms |
|-------|-------------------|-------------------|-----------|
| 0 | 0.236 | 0.085 | 0.364 |
| 1 | 1.095 | 0.148 | 0.224 |
| 2 | 0.585 | 0.134 | 0.379 |
| ... | ... | ... | ... |
| 21 | 1.019 | 0.378 | 0.076 |
| 22 | 2.174 | 0.465 | 0.063 |
| 23 | 1.413 | 0.398 | 0.145 |

**The smoking gun:**
1. **Attention projection output (a_rms) grows 6-8x from layer 0 to layer 23.** This is BEFORE normalization.
2. **Conv projection output (c_rms) grows ~5x.**
3. **Fused output (fused_rms) SHRINKS from 0.36 to 0.08** — the branches are growing in magnitude but diverging in direction, so their mean cancels out.
4. **SS-RMSNorm scales are modest** (0.5-1.2 range) — not the source.
5. The shrinking fused RMS means the hybrid block contributes less and less to the residual stream in deeper layers — it's becoming a dying branch.

### Why this happens

The 256→512 projection creates an amplification channel. Each layer's projection weights can grow to increase their branch's contribution. But since the two branches point in increasingly different directions (high a_rms + high c_rms but low fused_rms), the mean fusion cancels the growing magnitudes. This means:

- The residual stream is dominated by the identity connection (x + m where m→0)
- The outlier activations from individual branches before cancellation create kurtosis spikes
- Deeper layers see increasingly distorted inputs, compounding the effect

### Previous variants showed IDENTICAL pattern

| Variant | Step 5000 kurt | Step 5000 max_act | Final BPT vs T |
|---------|---------------|-------------------|----------------|
| HEMD-R5-G (β + RMSNorm) | 3.94 | 63.6 | +0.11 |
| R6-F (no β, SS-RMSNorm) | 3.5 | 58.5 | -0.11 |
| R6-S (softmax, SS-RMSNorm) | 2.6 | 55.0 | -0.10 |

The instability is slightly reduced from N→F→S but the pattern is the same. The root cause is the projected branches, not the mixing mechanism.

## What we know for certain

1. **Hybrid local+global complementarity produces real learning efficiency** — up to +0.54 BPT advantage at step 2000
2. **This advantage is destroyed by structural instability** by step 5000
3. **The instability is in the projection → normalization → fusion pipeline**, not in any particular mixing weight
4. **The 42M P-block (full-dim, no projection) was stable** (kurtosis 0.82 at step 5000). The projection is the key difference.

## MiniPLM 500-Window Results (O4)

| Metric | Value |
|--------|-------|
| Diff score (mean ± std) | 0.268 ± 0.105 |
| Top-50% threshold | 0.263 |
| Reference NLL | 2.670 ± 0.672 |
| Teacher NLL | 2.402 ± 0.648 |

Source ranking: gutenberg (0.294) > wikipedia (0.286) > fineweb (0.251) > openwebmath (0.239) > wildchat (0.232) > eli5 (0.222)

Consistent with 200-window pilot. Pipeline validated at 500-window scale.

## Questions for Round 7

1. **Should we abandon projected branches entirely and go back to the full-dim P-block with GQA?** The 42M P-block was stable. The P-block uses full-dim branches (no projection) at the cost of more params per block. With GQA this would save some attention params.

2. **Post-fusion normalization:** Before giving up on projected branches, should we try adding a normalization after the mean fusion but before W_out? This would control the fused magnitude regardless of branch divergence. The Codex R6 evidence doc listed this as option (c).

3. **Gradient clipping on projections:** Could we apply weight norm or spectral norm to the branch projections (attn_proj, conv_proj) to prevent magnitude growth?

4. **Fewer hybrid layers:** If 24 hybrid layers accumulate too much projection error, what about a mixed schedule (e.g., 6 hybrid + 18 attention)? The early-learning advantage was real — maybe we only need hybrid layers in the early/middle of the network.

5. **Should we just go with the pure transformer at 100M and move to production?** We've spent 6 rounds trying to beat it. The hybrid learns faster early but can't sustain the advantage. Is the transformer simply the better architecture at this scale?

6. **At what point do we cut losses on the hybrid approach?** Codex R6 said O1=7/10 to promote. Current O1 evidence: hybrid family learns more efficiently but EVERY variant tested at 100M scale ends behind the transformer. Is this a "the hybrid doesn't work at 100M" finding or a "we haven't found the right hybrid yet" finding?
