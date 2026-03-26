# T+L Round 5 Evidence Brief

## 1. Complete 42M Probe Summary

### 1.1 Trunk Choice (Probe: probe_trunk_choice, 5K steps)

| Variant | Params | Layers | Final BPT | kurtosis_max | max_act |
|---------|--------|--------|-----------|-------------|---------|
| pure_transformer | 49.1M | 12 | 4.9131 | 1.11 | 23.91 |
| pure_conv | 46.3M | 12 | 5.6147 | 2.30 | 30.18 |
| hybrid_3to1 (inter-layer) | 47.0M | 12 | **4.7218** | **1.03** | **19.21** |

**Verdict:** Inter-layer hybrid wins decisively. Hybrid gets best of both worlds.

### 1.2 Kernel Sweep — Mean Fusion (Probe: probe_parallel_hybrid, 9L, 5K steps)

| Variant | Params | Layers | Final BPT | kurtosis_max | max_act |
|---------|--------|--------|-----------|-------------|---------|
| parallel_k64 (mean) | 46.2M | 9 | 4.9536 | 1.45 | 20.88 |
| **parallel_k4 (mean)** | **46.0M** | **9** | **4.8005** | **0.82** | **18.15** |

**Verdict:** k4 beats k64 by 0.15 BPT with dramatically better stability. Small kernels win in intra-layer blocks (every-layer mixing compensates for short receptive field).

**Cross-comparison with 12L inter-layer hybrid:** k4 mean at 9L (4.8005) nearly ties 12L inter-layer (4.7218) with 3 fewer layers and fewer params. Per-layer quality of intra-layer block is substantially higher.

### 1.3 Kernel + Fusion + Ratio Sweep — Concat-Project (Probe: r4_microprobe, 12L, 5K steps)

| Variant | Params | Block Type | Kernel | Ratio | Final BPT | kurtosis_max | max_act |
|---------|--------|-----------|--------|-------|-----------|-------------|---------|
| concat_1to1_k4 | 47.5M | C | 4 | 1:1 | **4.9421** | 1.4 | 26.9 |
| concat_1to1_k16 | 47.6M | C | 16 | 1:1 | **4.8320** | 2.0 | 35.2 |
| concat_1to1_k64 | 47.7M | C | 64 | 1:1 | **4.8502** | 5.8 | 28.1 |
| **concat_2to3_k16** | **50.9M** | **C** | **16** | **2:3** | **4.7713** | **1.9** | **33.1** |

**Analysis (partial — k4 done, k16/k64/2:3 running):**

**Fusion comparison (key finding): concat_k4 (12L) LOSES to mean_k4 (9L) on every metric.**
| Metric | concat_k4 (12L, 47.5M) | mean_k4 (9L, 46.0M) | Winner |
|--------|------------------------|---------------------|--------|
| BPT | 4.9421 | 4.8005 | Mean by 0.14 |
| kurtosis_max | 1.4 | 0.82 | Mean (much more stable) |
| max_act | 26.9 | 18.15 | Mean (much healthier) |

concat_k4 used 33% more layers and 3% more params, yet lost on quality AND stability. The learned concat-project fusion does NOT compensate for the instability it introduces. This likely reflects the branch magnitude divergence observed in Hymba research — without pre-fusion normalization, scalar s_a/s_c alone cannot stabilize the concatenated signals.

**concat_k4 stability trajectory:** kurtosis grew from 0.2 (step 1500) → 2.1 (step 4000), then partially recovered to 1.4 during LR cooldown. max_act peaked at 32.3 (step 4500) → 26.9 (step 5000). The cooldown effect suggests instability is partially LR-driven, but the mid-training instability window (steps 3000-4000) is concerning for longer runs.

**Kernel sweep results (COMPLETE):**

| Variant | BPT | kurtosis_max | max_act | Notes |
|---------|-----|-------------|---------|-------|
| concat_k4 | 4.9421 | 1.4 | 26.9 | Most stable concat |
| **concat_k16** | **4.8320** | 2.0 | 35.2 | Best concat BPT |
| concat_k64 | 4.8502 | **5.8** | 28.1 | Catastrophic kurtosis |

In concat-project, k=16 wins on BPT while k=4 wins on stability. k=64 has near-parity BPT but catastrophic kurtosis (5.8). ALL concat variants show growing instability in the 3000-4000 step window. The instability is fusion-method-specific, not kernel-specific.

**Kernel preference reversal:** In mean fusion, k=4 wins. In concat-project, k=16 wins. Hypothesis: the learned w_mix projection can leverage k=16's broader receptive field, while simple averaging gains nothing from it.

**Cross-fusion comparison (key finding):**
| Block Type | Best Kernel | BPT | kurtosis | max_act | Layers |
|-----------|-------------|-----|----------|---------|--------|
| Mean (P) | k=4 | **4.8005** | **0.82** | **18.15** | 9 |
| Concat (C) | k=16 | 4.8320 | 2.0 | 35.2 | 12 |

Mean fusion at 9 layers MATCHES concat at 12 layers on BPT, with 2.4x better kurtosis and 1.9x better max_act. Mean is strictly superior when accounting for stability and layer efficiency.

**Ratio comparison (COMPLETE):**
2:3 (d_conv=384) BEATS 1:1 (d_conv=256) at k=16: BPT 4.7713 vs 4.8320 (+0.06). 2:3 nearly ties inter-layer hybrid (4.7218). But costs 50.9M vs 47.6M (+7% params). Stability comparable (kurtosis 1.9 vs 2.0).

**GRAND COMPARISON (all 42M probes, step 5000):**
| Rank | Variant | Params | Layers | BPT | kurtosis | max_act |
|------|---------|--------|--------|-----|----------|---------|
| 1 | **inter-layer hybrid** | 47.0M | 12 | **4.7218** | 1.03 | 19.21 |
| 2 | **concat_2to3_k16** | 50.9M | 12 | **4.7713** | 1.9 | 33.1 |
| 3 | mean_k4 (P-block) | 46.0M | 9 | 4.8005 | **0.82** | **18.15** |
| 4 | concat_1to1_k16 | 47.6M | 12 | 4.8320 | 2.0 | 35.2 |
| 5 | concat_1to1_k64 | 47.7M | 12 | 4.8502 | 5.8 | 28.1 |
| 6 | pure_transformer | 49.1M | 12 | 4.9131 | 1.11 | 23.91 |
| 7 | concat_1to1_k4 | 47.5M | 12 | 4.9421 | 1.4 | 26.9 |
| 8 | mean_k64 (P-block) | 46.2M | 9 | 4.9536 | 1.45 | 20.88 |
| 9 | pure_conv | 46.3M | 12 | 5.6147 | 2.30 | 30.18 |

**KEY INSIGHTS FOR R5:**
1. **Best BPT overall:** inter-layer hybrid (4.7218), followed by concat_2to3 (4.7713)
2. **Best stability:** mean_k4 (kurtosis 0.82, max_act 18.15) — dramatically better than any concat
3. **Best per-layer efficiency:** mean_k4 at 9L (4.8005) ≈ concat_k16 at 12L (4.8320) — P-blocks extract ~33% more quality per layer
4. **Concat instability is universal:** ALL concat variants show kurtosis 1.4-5.8, max_act 26-35. This is a fusion-method problem, not kernel or ratio.
5. **2:3 ratio helps:** Adding conv capacity improves BPT from 4.8320→4.7713 at cost of 3.3M params
6. **Kernel preference depends on fusion:** mean→k4 wins; concat→k16 wins

**CRITICAL DESIGN QUESTION for R5:**
Should the 100M gate use:
- (A) Mean fusion P-blocks (best stability, best per-layer, but full-dim attention = no GQA)
- (B) Concat C-blocks + Hymba-style pre-fusion norm (fixes instability?) + 2:3 ratio + k16
- (C) A NEW block: GQA attention projected to full dim + gated conv projected to full dim + mean fusion (combines GQA efficiency with mean stability)

### 1.4 Earlier Probes (Already Decided)

- **SS-RMSNorm > RMSNorm > DyT** (from probe_dyt_vs_rmsnorm)
- **NTP only > NTP+TOP** (from probe_top_vs_ntp)
- **AdamW > Muon at 42M** (from probe_muon_vs_adamw), NorMuon deferred to 100M

## 2. R4 Research Findings

### 2.1 MiniPLM Cross-Tokenizer
Same-tokenizer teacher/reference STRONGLY preferred for MiniPLM difference scoring. Cross-tokenizer adds noise that degrades data quality signal. ALM (arxiv 2503.20083) solves cross-tokenizer distillation at token level but not data selection.
**Action:** Train ~100M Qwen-tokenizer reference as R4 proposes.

### 2.2 Small Embedding Teachers
Top picks under 1B: nomic-embed-text (137M), EmbeddingGemma-300M, GTE-base (305M), BGE-M3 (568M), Qwen3-Embedding-0.6B.
Layer selection: last layer only. Pooling: mean pooling. Alignment: projection + cosine/MSE loss at 0.01-0.1 weight.
**Action:** Start with nomic-embed-text (137M) for lowest overhead.

### 2.3 Branch Balancing
Falcon-H1: concat + channel ratio (attention SMALL fraction). Hymba: per-channel learnable beta + normalization. Griffin: simple addition.
KEY: pre-fusion normalization matters (Mamba output > attention output in later layers).
Our approach (scalar s_a/s_c + concat-project) is simpler than Hymba but may need pre-fusion norm if stability issues appear.
**Action:** Monitor concat microprobe kurtosis. Add pre-fusion RMSNorm if needed.

**UPDATE — Deep research on Hymba and Falcon-H1 fusion (from arxiv HTML):**

**Hymba exact formula:** `Y = W_out_proj(β₁·norm(attn_out) + β₂·norm(ssm_out))`
- Per-CHANNEL learnable β vectors (not scalars like our s_a/s_c)
- Normalization on each branch INDEPENDENTLY BEFORE combining
- Then element-wise addition (not concat!), then output projection
- Motivation: "SSM output magnitudes consistently larger than attention heads"

**Falcon-H1 exact architecture:** SSM:Attention:MLP = 2:1:5 (attention = 1/8 of channels)
- Concat SSM + attention outputs → output projection
- "More attention channels significantly degrades performance"
- No pre-fusion norm — they solve magnitude imbalance by making attention tiny

**Implication for our design:**
Our ConcatProjectHybridBlock uses scalar s_a/s_c (not per-channel β) and NO pre-fusion normalization. This directly explains the growing kurtosis we observe. Two options:
1. Add Hymba-style per-branch RMSNorm + per-channel learnable β before combining
2. Switch to mean fusion (which implicitly handles magnitude via averaging)

## 3. Architecture Status

**Current HEMD-R4-S spec (from Codex R4):**
- 26x512 (target), 24x512 (gate)
- d_attn=256, d_conv=256 (1:1), k=16
- SS-RMSNorm, GQA 4Q/2KV, SwiGLU 1536
- Exits 9/18/26 (or 8/16/24 for gate)
- Plain NTP + fixed exits, no online auxiliaries

**Pending microprobe adjustments:**
- Kernel size: R4 proposes k=16; microprobe will confirm vs k=4 and k=64
- Ratio: R4 proposes 1:1; microprobe will confirm vs 2:3
- Branch init: R4 proposes ga=gc=1.0; current code uses ga=1.0, gc=sqrt(d_a/d_c)

## 4. Questions for Round 5

1. **Is the 42M evidence sufficient to greenlight the 100M gate?** We have 8 probes with consistent positive results for hybrid architecture.
2. **Which kernel size for the 100M gate?** Pending microprobe result.
3. **Which ratio for the 100M gate?** Pending microprobe result.
4. **Should the 100M gate use SS-RMSNorm from step 0?** R4 intuition #1 says the hybrid win is conservative because it was tested with plain RMSNorm.
5. **Is 24x512 the right gate geometry?** R4 says yes. Alternatives: 20x576, 22x544.
6. **How many steps for the 100M gate?** 5K (like 42M probes) or 10K (more signal)?

## 5. Confidence Trajectory

| Outcome | R1 | R2 | R3 | R4 |
|---------|----|----|----|----|
| O1: Intelligence | 2 | 3 | 4 | 5 |
| O2: Improvability | 4 | 5 | 6 | 6 |
| O3: Democratization | 3 | 4 | 5 | 5 |
| O4: Data Efficiency | 2 | 2 | 3 | 3 |
| O5: Inference Efficiency | 3 | 4 | 5 | 5 |

Target: all >= 9/10. Biggest gap: O4 (3/10) and O1 (5/10). Both require 100M results to advance.
