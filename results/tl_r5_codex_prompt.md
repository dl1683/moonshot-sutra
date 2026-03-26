MANDATORY FIRST STEP: Read CLAUDE.md in this repository root. Every rule there is binding.
Then read research/VISION.md — especially the "Design Philosophy" section at the top.
Then read research/TESLA_LEIBNIZ_CODEX_PROMPT.md — it defines your role, your questioning principle, your output format, and how you must think. Follow it exactly.
Then read research/ARCHITECTURE.md — it contains the full architecture evolution through 4 rounds.
Then read research/RESEARCH.md — it contains all field research, probe results, and dead ends.

CONTEXT YOU MUST KNOW:
- DESIGN PHILOSOPHY: Nothing is sacred except outcomes. 5 sacred OUTCOMES, mechanisms negotiable.
- We are in Round 5 of the Tesla+Leibniz architecture design loop.
- Round 4 (your previous round's output) is pasted verbatim below.
- New findings since Round 4 are listed after the Round 4 output.

---

## YOUR PREVIOUS ROUND'S OUTPUT (Round 4, Verbatim)

(Read the file results/tl_round4_output.md for the full R4 output. It is 80 lines containing: 9 assumption challenges, 3 research requests, 7 probe requests, confidence scores [5/6/5/3/5], intuitions, and the HEMD-R4-S design proposal with block math.)

---

## NEW FINDINGS SINCE ROUND 4

### Finding 1: Parallel Hybrid Probe Results (mean fusion, P-blocks, 9L, 5K steps)

Two variants tested: k=64 and k=4, both with Hymba-style mean fusion.

| Variant | Params | Final BPT | kurtosis_max | max_act |
|---------|--------|-----------|-------------|---------|
| parallel_k64 (mean) | 46.2M | 4.9536 | 1.45 | 20.88 |
| **parallel_k4 (mean)** | **46.0M** | **4.8005** | **0.82** | **18.15** |

k4 wins by 0.15 BPT with dramatically better stability. Validates R4 intuition #2: "Small-k conv will beat k=64 in intra-layer blocks."

Key trajectory insight: k4 learned much faster early (1.07 BPT gap at step 1000), the gap narrowed by step 2500, then k4 pulled ahead again in cooldown. k4's final advantage is in both BPT quality and activation health.

Cross-comparison with inter-layer hybrid: k4 mean at 9L (4.8005) nearly ties 12L inter-layer hybrid (4.7218) with 3 fewer layers and fewer params. Per-layer quality of the intra-layer block is substantially higher than inter-layer mixing.

### Finding 2: R4 Microprobe Results (concat-project, C-blocks, 12L, 5K steps)

Four variants tested with the HEMD-R4-S concat-then-project fusion block:

| Variant | Params | Kernel | Ratio | Final BPT | kurtosis_max | max_act |
|---------|--------|--------|-------|-----------|-------------|---------|
| concat_1to1_k4 | 47.5M | 4 | 1:1 | **4.9421** | 1.4 | 26.9 |
| concat_1to1_k16 | 47.6M | 16 | 1:1 | **4.8320** | 2.0 | 35.2 |
| concat_1to1_k64 | 47.7M | 64 | 1:1 | **4.8502** | 5.8 | 28.1 |
| **concat_2to3_k16** | **50.9M** | **16** | **2:3** | **4.7713** | **1.9** | **33.1** |

**Kernel sweep analysis:** In concat-project, k=16 WINS (4.8320 BPT), k=64 ties (4.8502), k=4 lags (4.9421). This REVERSES the mean fusion result where k=4 wins. Hypothesis: learned w_mix projection leverages broader receptive field. BUT all concat variants show growing instability (kurtosis 1.4-5.8). k=64 has catastrophic kurtosis=5.8 at step 5000.

**Fusion comparison:** Mean k4 at 9L (4.8005) BEATS best concat (k16 at 12L, 4.8320) with 2.4x better kurtosis (0.82 vs 2.0) and 1.9x better max_act (18.15 vs 35.2) using 25% fewer layers and 3% fewer params. **Mean fusion is strictly superior** at this scale.

**Ratio comparison:** 2:3 (d_conv=384) BEATS 1:1 (d_conv=256) at k=16: BPT 4.7713 vs 4.8320 (+0.06). 2:3 nearly ties the inter-layer hybrid (4.7218) — only 0.05 gap. But 2:3 costs 50.9M vs 47.6M params (+7%). Stability similar (kurtosis 1.9 vs 2.0). The extra conv capacity DOES help — contrary to Falcon-H1's finding that more SSM HURTS. Key difference: Falcon-H1 tests SSM fraction at fixed total capacity. Our 2:3 ADDS capacity (3.3M more params), so the comparison is not like-for-like.

**concat_k4 full result:** BPT=4.9421 (final), kurtosis peaked at 2.1 (step 4000, recovered to 1.4 at step 5000 during cooldown), max_act peaked at 32.3 (step 4500, recovered to 26.9). CRITICAL: mean_k4 at 9L BEATS concat_k4 at 12L on EVERY metric (BPT 4.80 vs 4.94, kurtosis 0.82 vs 1.4, max_act 18.15 vs 26.9). Learned concat-project fusion does NOT beat simple averaging — it adds instability without BPT gain. This likely means pre-fusion normalization is needed OR mean fusion is simply better.

### Finding 3: Complete 42M Probe Summary (8 probes total)

| Probe | Key Finding |
|-------|-------------|
| DyT vs RMSNorm | SS-RMSNorm > RMSNorm > DyT |
| NTP vs TOP | Plain NTP > NTP+TOP |
| Muon vs AdamW | AdamW > Muon at 42M (Muon late instability) |
| Trunk choice | Inter-layer hybrid (4.72) > transformer (4.91) > conv (5.61) |
| Parallel hybrid | k4 mean (4.80) > k64 mean (4.95), both 9L |
| R4 microprobe | concat_2to3_k16 best BPT (4.77), mean_k4 better stability. ALL concat variants unstable (kurtosis 1.4-5.8) |

### Finding 4: R4 Research Findings

**MiniPLM cross-tokenizer:** Same-tokenizer teacher/reference STRONGLY preferred for difference scoring. Per-byte normalization reduces but doesn't eliminate noise. ALM (arxiv 2503.20083) solves cross-tokenizer distillation at token level but not data selection. Action: train ~100M Qwen-tokenizer reference.

**Small embedding teachers under 1B:** Top picks: nomic-embed-text (137M), EmbeddingGemma-300M, GTE-base (305M), BGE-M3 (568M), Qwen3-Embedding-0.6B. Layer: last only. Pooling: mean. Alignment: projection + cosine/MSE at 0.01-0.1 weight. Recommendation: start with nomic-embed-text (137M).

**Branch balancing for hybrid decoders (DEEP RESEARCH UPDATE):**

**Hymba exact formula:** `Y = W_out_proj(β₁·norm(attn_out) + β₂·norm(ssm_out))`
- Per-CHANNEL learnable β vectors, NOT per-branch scalars
- Each branch INDEPENDENTLY NORMALIZED before combining
- Element-wise addition (not concat), then output projection
- Motivation: "SSM output magnitudes consistently larger than attention"

**Falcon-H1:** SSM:Attention:MLP = 2:1:5 channel ratio. Attention = 1/8 of channels.
- Concat SSM + attention → output projection
- "More attention channels significantly DEGRADES performance"
- No pre-fusion norm — they solve magnitude imbalance by making attention tiny (1/8)

**Our empirical finding confirms this:** Our ConcatProjectHybridBlock uses scalar s_a/s_c with NO pre-fusion norm, and shows growing kurtosis (0.2→2.1 for k4, 0.2→2.6 for k16) and max_act spikes (10.9→26.9 for k4, 12.2→35.2 for k16). Meanwhile, mean fusion (P-blocks) with 0.5*(a+c) stays stable (kurtosis 0.82, max_act 18.15 at step 5000).

**KEY: Mean fusion (P-blocks) achieves BETTER BPT than concat (C-blocks) at fewer layers with better stability.** This may mean Hymba-style norm+β is the minimum fix for concat, or mean fusion is simply superior for our scale.

### Finding 5: Current HEMD-R4-S Spec (from Section 11 of ARCHITECTURE.md)

26x512 (target), 24x512 (promotion gate). d_attn=256, d_conv=256 (1:1), k=16 (pending microprobe). SS-RMSNorm, GQA 4Q/2KV, SwiGLU 1536. Exits 9/18/26. Plain NTP + fixed exits only. AdamW default, NorMuon as probe arm.

---

## SPECIFIC QUESTIONS FOR THIS ROUND

1. Based on the microprobe results, should the 100M gate use k=4, k=16, or k=64?
2. Should the gate use 1:1 or 2:3 branch ratio? Falcon-H1 research shows attention should be SMALL — does this change the recommendation?
3. Is the 42M evidence sufficient to greenlight the 100M gate, or do we need more probes?
4. The concat block learns slower than mean fusion but may converge better. Should the 100M gate use more steps (10K instead of 5K)?
5. Should we add pre-fusion normalization (Hymba-style) to the concat block before the 100M gate?
6. What's the simplest O4 action we can take IN PARALLEL with the 100M gate (not blocking it)?
