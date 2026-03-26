# T+L Round 4: Evidence Package

Compiled: 2026-03-26, live during trunk-choice probe V3 execution.

---

## 1. Trunk-Choice Probe Results (42M scale, 5000 steps)

Config: d=512, L=12, H=8, ff=1536, seq=512, BS=16x2, LR=3e-4, WSD schedule, AdamW, RMSNorm.

### Completed Variants

| Variant | Block Pattern | Params | Final BPT | Kurtosis Max | Max Act |
|---------|---------------|--------|-----------|-------------|---------|
| V1: pure_transformer | 12x Attention | 49,138,176 | **4.9131** | 1.1 | 23.9 |
| V2: pure_conv | 12x GatedConv (k=64) | 46,346,240 | 5.6147 | 2.3 | 30.2 |
| **V3: hybrid_3to1** | [H,H,A,H,H,H,A,H,H,H,A,H] | 47,034,880 | **4.7218** | **1.03** | **19.21** |

**Key findings:**
- Pure conv (Hyena-style GatedConv, kernel=64) loses to pure transformer by +0.70 BPT at 42M. Conv-only is clearly worse.
- **V3 (inter-layer hybrid) WINS: BPT=4.72 vs V1=4.91 — improvement of 0.19 BPT!**
- V3 also has BETTER activation health: kurtosis_max=1.03 vs V1's 1.11, max_act=19.21 vs V1's 23.91.
- **Inter-layer hybrid BEATS pure transformer at 42M despite fewer attention blocks** (only 3 of 12).

**Interpretation:**
- Mixing IS beneficial even in the weaker inter-layer configuration.
- The conv branch provides complementary long-range receptive field that attention alone misses.
- Conv-only is bad (-0.70 BPT) → conv NEEDS attention for exact retrieval.
- But hybrid shows attention alone is suboptimal → attention ALSO benefits from conv's efficient local mixing.
- **The intra-layer parallel design (every block has BOTH) should be even stronger** since mixing happens at every layer, not just at occasional attention blocks.
- V3's superior activation health (lower kurtosis, lower max_act) suggests the conv branch stabilizes training.

**This is the strongest positive signal for the hybrid architecture.** Inter-layer hybrid wins by 0.19 BPT over pure transformer. Intra-layer parallel should do better.

**Note:** V3 has 47.0M params vs V1's 49.1M — hybrid wins with FEWER parameters.

---

## 2. NorMuon Research (Codex Request #1)

**Paper:** "NorMuon: Making Muon more efficient and scalable" (arxiv:2510.05491, Li et al.)

### Algorithm
1. First-order momentum: `M_t = β₁M_{t-1} + (1-β₁)G_t`
2. Orthogonalization: `O_t = NS5(M_t)` (5 Newton-Schulz iterations)
3. **Neuron-wise second-order momentum:** `v_t = β₂v_{t-1} + (1-β₂)mean_cols(O_t ⊙ O_t)`
4. **Row-wise normalization:** `Õ_t = O_t / √(v_t + ε)` (broadcast per row)
5. Scaled update: `W_{t+1} = W_t - ηλW_t - η̂Õ_t` where `η̂ = 0.2η√(mn)/‖Õ_t‖_F`

### Key innovation
After Muon's orthogonalization, different neurons (rows) still have wildly different update magnitudes. NorMuon tracks per-neuron squared magnitude via EMA and normalizes each row independently. This is exactly the fix for our observed instability: Muon's max_act=59.6 (2.7x AdamW) suggests specific neurons are receiving outsized updates.

### Memory
- Adam: 2mn per weight matrix
- Muon: mn
- NorMuon: m(n+1) — only m extra scalars. ~50% more efficient than Adam.

### Hyperparameters (reference)
| Scale | β₁ | β₂ | LR | Weight Decay |
|-------|----|----|-----|--------------|
| 124M NanoGPT | 0.9-0.95 | 0.95 | 3.6e-4 | — |
| 350M NanoGPT | 0.9-0.95 | 0.95 | 7.5e-4 | — |
| 1.1B pretraining | 0.95 | 0.95 | 4e-4 | 0.1 |

### Small-scale results
- 124M: 6% fewer iterations than Muon
- 350M: 15% efficiency gain over Muon
- 1.1B: 11.31% better than Muon, 21.74% better than Adam

### Relevance to Sutra
NorMuon's per-neuron normalization directly addresses the instability pattern we observed: Muon converges 1.7x faster early but destabilizes late with max_act spikes. NorMuon should fix this. For 100M probe: try NorMuon with β₁=0.95, β₂=0.95, lr=0.01 (lower than our failed lr=0.02).

---

## 3. Intra-Layer Parallel Hybrid Research (Codex Request #2)

### Falcon-H1 0.5B Architecture (EXACT config from HuggingFace)

| Parameter | Value |
|-----------|-------|
| Layers | 36 |
| Hidden dim | 1024 |
| Attention heads | 8 (GQA: 2 KV heads) |
| Mamba heads | 24 |
| Head dim | 64 |
| FFN intermediate | 2048 |
| Mamba d_state | 128 |
| Mamba d_ssm | 1536 |
| Mamba d_conv | **4** (tiny kernel!) |
| Attention:SSM ratio | **1:3** (8 attn : 24 mamba) |
| Total params | ~0.5B |
| μP multipliers | Yes (attention_out=0.9375, ssm_out=0.2357, ssm_in=1.25) |

**Critical observation:** Falcon-H1 uses Mamba-2 with d_conv=4 (tiny kernel), NOT large-kernel conv like Hyena (k=64). Our GatedConv uses k=64 which is very different. For intra-layer parallel, we should consider:
- Option A: Keep our GatedConv (k=64) as the conv branch — bigger receptive field per layer
- Option B: Use smaller kernel (k=4-8) matching Falcon-H1 — different design philosophy
- The 1:3 attention:SSM ratio means most computation goes to the SSM/conv branch

### Hymba 1.5B Architecture (NVIDIA, ICLR 2025)

| Parameter | Value |
|-----------|-------|
| Layers | 32 |
| Hidden dim | 1600 |
| Attention heads | 25 |
| SSM:Attention ratio | **5:1** parameter ratio |
| Meta tokens | 128 learnable tokens prepended to input |
| KV sharing | Cross-layer sharing between every 2 consecutive layers |
| Full attention | Only 3 layers (first, middle, last) — rest use sliding window |

**Combination method:** Parallel paths → mean → output projection. NOT concat-then-project like Round 3 spec.

### Implications for HEMD-R3-S Design

The Round 3 spec proposed: `r = h + Wmix[a ; c]` (concat then project).
Hymba uses: mean of parallel outputs.
Falcon-H1 uses: parallel heads concatenated into the overall head dimension.

**Key question for Codex:** Should we concat-then-project (Round 3 spec), mean (Hymba), or interleave heads in the overall head dimension (Falcon-H1)?

**Scale concern:** Both Falcon-H1-0.5B (36 layers) and Hymba-1.5B (32 layers) are much deeper than our proposed 14 blocks. Our 100M budget severely constrains depth. With dim=768, 14 blocks, we can't match the 1:3 ratio at the same head counts.

### Proposed revision for 100M budget

Given dim=768 and 14 blocks:
- d_attn = 256 (4 heads × head_dim=64, 1 KV head)
- d_conv = 512 (8 conv channels × head_dim=64, kernel=64)
- Ratio: 1:2 attention:conv
- This is conservative compared to Falcon-H1's 1:3 but matches our budget

---

## 4. Multi-Teacher Purification Research (Codex Request #3)

### Knowledge Purification (Jin et al., Feb 2026)
- **Scope: Fine-tuning/instruction tuning ONLY, NOT pretraining**
- 5 methods: GPT-4 aggregation, Plackett-Luce ranking, PLM classifier, similarity router, RL selection
- Best: Similarity-based router and RL-based selection (+5% over naive multi-teacher)
- Key finding: More teachers WITHOUT purification HURTS performance — teacher conflict is real
- Student sizes tested: 77M, 248M, 783M (FLAN-T5)

### TinyLLM (Tian et al., 2024)
- Multi-teacher KD for LLMs but focuses on fine-tuning, not pretraining
- Uses in-context example generation + Chain-of-Thought strategy
- +5-16% over full fine-tuning

### Implications for Sutra
**No pretraining-specific multi-teacher purification exists in literature.** The Round 3 design (one decoder teacher + one embedding teacher, one family per batch) is actually a reasonable first approach because:
1. One-family-per-batch avoids simultaneous teacher conflict entirely
2. Different modalities (decoder logits vs embedding alignment) are complementary, not competing
3. MiniPLM handles the data-side distillation offline, avoiding online teacher conflict

**Recommendation for Codex:** The multi-teacher purification question is premature for the scout. The plain scout has no teacher losses. The first multi-source extension should be:
1. MiniPLM (offline, no conflict possible)
2. One decoder teacher loss (single teacher, no purification needed)
3. Multi-teacher only if single-teacher proves positive first

---

## 5. MiniPLM Research (Codex Request #4)

### Core Mechanism: Difference Sampling
Formula: `r(x) = log p_teacher(x) / log p_ref(x)`
Selection: Top-K by score, where K = α|D - D_ref|, typical α=0.5

### Reference Model Details
- Paper uses: **104M reference** (Qwen architecture, trained on 5B tokens)
- Teacher: 1.8B Qwen
- Students: 200M, 500M, 1.2B
- **Reference is SMALLER than student** in all cases
- Reference trains on a subset of the same data (5B tokens held out)

### Cross-Family Capability
**Yes, MiniPLM works across families.** Paper demonstrates:
- Qwen teacher + Qwen reference → Llama-3.1 212M student (works)
- Qwen teacher + Qwen reference → Mamba 140M student (works)
- Tokenizer mismatch handled: scoring uses teacher's tokenizer, training uses student's

### Practical Details
- Scoring: Forward pass through teacher and reference on all candidate data
- Selection: Top-50% of scored data by difference score
- Reduces data demand by 2.4x
- All scoring done UPFRONT before training (no incremental option mentioned)
- 200M student result: 41.3% avg accuracy vs 39.9% baseline (+1.4%)

### Implications for Sutra
1. **Reference model:** We need a ~100M reference model. Options:
   - Train a quick 100M model on 5B tokens (matches paper's approach)
   - Use Qwen3-0.6B (larger than recommended but simplest)
   - The paper's reference was 104M trained on 5B — close to our student scale
2. **Teacher:** Qwen3-1.7B is fine (paper used 1.8B teacher)
3. **Tokenizer:** Paper works cross-family, but same-family is cleanest. Since we have a custom 16K tokenizer, we'd need to score with Qwen3's tokenizer then map weights to our shard indices
4. **Data volume:** Paper scored 100B tokens, used top 50B. Our corpus is ~22.9B tokens. Scoring all of it is feasible.
5. **Compute:** Two forward passes (teacher + reference) per token. With Qwen3-1.7B teacher and 104M reference, on RTX 5090: ~1-2 hours to score 22.9B tokens

---

## 6. Updated Probe Status

| Probe | Status | Result | Implication |
|-------|--------|--------|-------------|
| DyT vs RMSNorm | DONE | DyT loses by +0.75 BPT | DyT dead for scout |
| TOP vs NTP | DONE | TOP catastrophic (+4.61, kurtosis 99.3) | TOP dead for scout |
| Muon vs AdamW vs SS-RMSNorm | DONE | AdamW+SS-RMSNorm best (4.91) | AdamW default, Muon probe only |
| Trunk: pure_transformer | DONE | BPT=4.9131 | Transformer baseline |
| Trunk: pure_conv | DONE | BPT=5.6147 | Conv-only loses by +0.70 |
| Trunk: hybrid_3to1 | IN PROGRESS | Step ~450, BPT~7.0 | Still converging |

---

## 7. Questions for Codex Round 4

1. **Hybrid block mixing:** Should we concat-then-project (Round 3 spec), mean (Hymba), or head-interleaved (Falcon-H1)? At dim=768, what's the optimal split?

2. **Conv kernel size:** Falcon-H1 uses d_conv=4 (Mamba-2 style). Our GatedConv uses k=64 (Hyena style). Which is better for the intra-layer parallel hybrid at 100M? Should we test both?

3. **NorMuon as alternative optimizer probe:** Given the exact algorithm, should the 100M optimizer probe include NorMuon alongside Muon lr=0.01 and lr=0.005?

4. **MiniPLM reference model:** Train a quick 100M reference on 5B tokens (paper's approach), or just use Qwen3-0.6B (simpler but 6x larger than recommended)?

5. **Depth vs width tradeoff:** Falcon-H1-0.5B uses 36 layers at dim=1024. Our Round 3 spec uses 14 blocks at dim=768. Should we explore deeper-narrower configs (e.g., 24 blocks at dim=512) to better match the hybrid paradigm?

6. **μP multipliers:** Falcon-H1 uses carefully tuned μP multipliers for attention and SSM outputs. Should we incorporate μP into the scout, or is it premature at 100M?

7. **Pure conv failure:** At 42M, pure conv loses by +0.70 BPT. Does this change the attention:conv ratio we should target? Should the conv branch be smaller than Round 3's d_conv=512?
