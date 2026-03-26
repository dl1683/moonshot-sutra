# Scratchpad

Working space for half-finished thoughts, emerging ideas, and in-progress reasoning. Things move OUT of here into RESEARCH.md (when validated) or get deleted (when falsified).

**Rule:** Nothing here is a conclusion. Everything here is a question or a sketch.

---

## Pre-Round-1 Design Space Analysis (2026-03-26)

### The Core Problem
SmolLM2-135M trains on 2T tokens. We have ~23B. That's an 87x data disadvantage.
Pythia-160M trains on 300B tokens. Still 13x more than us.
To compete, we need either: (a) dramatically better data efficiency, or (b) knowledge absorption from existing models, or (c) both.

### Key Insight: The Data Efficiency Stack
Multiple techniques compound. Each addresses a different angle:

1. **Offline KD from teacher models** — MiniPLM shows 2.2-2.4x data efficiency. This is the single biggest lever.
   - Question: which teachers? How many? What representations to steal?
   - The OFFLINE approach is key — generate soft targets once, store to disk, train student against them
   - This means we can use models too big to run concurrently (just process data in advance)

2. **Multi-Token Prediction (TOP variant)** — Proven at 340M scale. ~20-50% more learning per example.
   - TOP is a learning-to-rank loss, not exact token prediction → works at small scale
   - Only needs one extra unembedding layer (~0.6% overhead)
   - Can combine with curriculum scheduling for more benefit

3. **N-gram memory** — Offloads pattern memorization to CPU table, frees neural capacity for reasoning
   - At our scale, a huge fraction of capacity is wasted memorizing "the → cat", "in → the", etc.
   - A 1M-entry table in CPU RAM costs ~48MB but could free 10-20% of neural capacity
   - The GATING is the key mechanism — table provides candidates, model decides relevance

4. **Structural priors** — Hyperbolic geometry, sheaf structure, etc.
   - HELM shows 4% gains from hyperbolic geometry. That's free intelligence.
   - Mixed-curvature (product manifolds) could be even better
   - Question: is this additive with the other techniques? Probably yes (orthogonal mechanisms)

5. **Architecture efficiency** — Hyena Edge shows gated convolutions beat attention
   - O(N log N) vs O(N²) — but at seq_len=512, this barely matters
   - The real win: gated convolutions may learn more efficiently from fewer tokens
   - Hybrid: keep attention for long-range, use convolutions for local patterns

### The Compound Effect
If these stack multiplicatively:
- 2x from offline KD
- 1.3x from TOP
- 1.2x from n-gram memory
- 1.04x from hyperbolic geometry
- 1.1x from architecture efficiency
= 2x * 1.3 * 1.2 * 1.04 * 1.1 ≈ 3.56x effective data

23B tokens * 3.56x = ~82B effective tokens. That closes the gap with Pythia-160M (300B tokens) significantly, though still behind SmolLM2-135M (2T tokens). But SmolLM2 uses a standard transformer — if our architecture extracts more per token inherently, the gap narrows further.

### Open Questions for Codex
1. What's the optimal model size for 24GB VRAM training? (200M? 400M? Depends on batch size)
2. Should we use shared-weight looping (LoopFormer-style) or unshared layers?
3. How to implement offline KD practically? Pre-compute soft targets from multiple teachers?
4. Is hyperbolic attention practical at our scale? (exp/log maps add overhead)
5. What's the right hybrid mix? (X% attention + Y% convolution + Z% SSM?)

### Risky Ideas Worth Testing
- **LoopFormer + early exit**: Shared blocks with time-conditioning, tokens exit at different loop iterations
- **Hyperbolic Engram**: N-gram memory in hyperbolic space (hierarchical lookup)
- **Cross-architecture distillation**: Steal from Mamba (SSM) AND Pythia (transformer) simultaneously
- **DyT everywhere**: Replace ALL normalization with DyT, design for quantization from step 1

---

## Cached Teacher Models (already downloaded on this machine!)

### Small LMs (can coexist with student during training):
- Pythia: 70M, 160M, 410M, 1B, 1.4B, 2.8B (+ deduped variants)
- SmolLM2: 135M, 360M, 1.7B
- GPT-2: 124M, 355M, 774M, 1.5B
- GPT-Neo: 125M, 1.3B, 2.7B
- Mamba: 130M, 370M, 790M, 1.4B (+ Mamba2 variants: 130M-2.7B)
- RWKV: 169M, 430M, 1.5B (RWKV4), 1.6B-14B (RWKV6), 191M-2.9B (RWKV7)
- OPT: 125M, 350M, 1.3B, 2.7B
- Cerebras-GPT: 111M, 256M, 590M, 1.3B
- TinyLlama: 1.1B
- StableLM: 1.6B, 3B
- Granite-4.0: micro, tiny, 350M, 1B (+ hybrid variants!)
- Falcon-H1: 0.5B, 1.5B, 3B (SSM-attention hybrid!)
- Zamba2: 1.2B, 2.7B (Mamba-attention hybrid)
- LFM2/2.5: 1.2B, 2.6B (Liquid AI — gated convolution hybrid!)
- Hymba: 1.5B (NVIDIA hybrid)

### Encoder models (rich representations, very small):
- BERT: base (110M), large (340M), + many variants
- RoBERTa: base, large
- DeBERTa: base, v3-base, v3-small
- DistilBERT: 66M
- Sentence transformers: all-MiniLM-L6-v2, all-mpnet-base-v2

### Embedding models:
- BGE: small, base, large, m3
- E5: small, base, large
- GTE: Qwen2-1.5B
- EmbeddingGemma: 300M
- Nomic-embed
- Stella: 1.5B

### Architecture diversity we can steal from:
- **Transformers**: Pythia, GPT-2, Qwen, Gemma, Llama, Phi
- **SSMs/Mamba**: Mamba 1/2, Falcon-Mamba
- **RWKV** (linear attention): RWKV4-7
- **Hybrids**: Falcon-H1, Zamba2, Granite-4.0-h, Hymba, LFM2, Jamba
- **Gated convolutions**: StripedHyena, LFM2
- **Encoders**: BERT, RoBERTa, DeBERTa
- **Diffusion LM**: DiffuGPT

This is a GOLDMINE for multi-source learning. We can generate offline soft targets from dozens of models.

---

## VRAM Budget Analysis (2026-03-26)

### Model Size vs Training Feasibility on RTX 5090 (24GB)

| Config | Params | Model(BF16) | AdamW | Grads | Act (BS=32) | TOTAL | MaxBS |
|--------|--------|-------------|-------|-------|-------------|-------|-------|
| 100M | 40.6M | 0.08GB | 0.32GB | 0.08GB | 5.13GB | 5.62GB | 133 |
| 135M (SmolLM2-class) | 82.2M | 0.16GB | 0.66GB | 0.16GB | 7.70GB | 8.69GB | 86 |
| 160M (Pythia-class) | 137.9M | 0.28GB | 1.10GB | 0.28GB | 10.27GB | 11.92GB | 63 |
| 200M [ckpt] | 175.6M | 0.35GB | 1.40GB | 0.35GB | 1.61GB | 3.72GB | 393 |
| 350M [ckpt] | 368.4M | 0.74GB | 2.95GB | 0.74GB | 2.68GB | 7.11GB | 208 |
| 400M [ckpt] | 435.5M | 0.87GB | 3.48GB | 0.87GB | 3.22GB | 8.45GB | 165 |

[ckpt] = gradient checkpointing. SwiGLU FFN, RoPE, RMSNorm assumed. seq_len=512.

### Chinchilla Analysis (tokens = 20x params)

| Size | Chinchilla-optimal | Our 22.9B tokens | Ratio |
|------|-------------------|-----------------|-------|
| 100M | 2.0B | 22.9B | 11.4x OVER |
| 135M | 2.7B | 22.9B | 8.5x OVER |
| 200M | 4.0B | 22.9B | 5.7x OVER |
| 350M | 7.0B | 22.9B | 3.3x OVER |
| 1.1B | 22.0B | 22.9B | ~1.0x OPTIMAL |

**Insight**: Chinchilla-optimal for our data budget = ~1.1B params. But we're not optimizing for Chinchilla — we're optimizing for PERFORMANCE at inference. Smaller model + KD = better than larger model from scratch at equivalent data. Over-training makes models more robust.

### KD VRAM Overhead

- **Offline KD**: ZERO GPU overhead during training. Storage: top-128 logits = ~1KB/token = ~23TB for full corpus (too much). Solution: stream soft targets, or use top-32 (~256B/token = ~5.9TB), or use feature-level distillation.
- **Online KD**: Teacher in inference mode (no grads)
  - Pythia-70M: 0.14GB | Pythia-160M: 0.32GB | SmolLM2-135M: 0.27GB
  - Pythia-410M: 0.82GB | SmolLM2-360M: 0.72GB | Mamba-790M: 1.6GB
  - **Conclusion**: Online KD feasible for teachers up to ~1B alongside 200M student

### Training Speed Estimates

| Size | Tokens/sec | Tokens/day | Full epoch (22.9B) |
|------|-----------|-----------|-------------------|
| 100M | ~80K | 6.9B | 3.3 days |
| 160M | ~55K | 4.8B | 4.8 days |
| 200M | ~45K | 3.9B | 5.9 days |
| 350M | ~25K | 2.2B | 10.6 days |

### Sweet Spot Analysis

**150-250M params** appears optimal:
- Over-trained on our data (5-8x Chinchilla) = robust
- Fits easily on 24GB with room for online KD teachers
- BS=32+ feasible with gradient checkpointing
- Full epoch in ~5-6 days = can do 4+ epochs in a month
- Room for auxiliary losses (TOP, KD) without VRAM pressure
- Small enough for rapid iteration, large enough for meaningful benchmarks

**Question for Codex**: Should we go 200M (faster iteration) or 350M (more capacity, slower)?

---

## Offline KD Feasibility Analysis (2026-03-26)

### Disk Budget
- 3,216 GB free on C:
- Training data occupies ~634GB (246 shards)

### Storage per approach (full 22.9B token corpus)
| Method | Storage | Feasible? |
|--------|---------|-----------|
| Full logits (FP16) | 733 TB | NO |
| Top-16 logits | 1.47 TB | Barely (45% of free space) |
| Top-32 logits | 2.93 TB | NO |
| Hidden states (d=768) | 35 TB | NO |

### Practical Strategy: ONLINE KD (Hybrid)
Full-corpus offline KD is impractical. **Online KD with co-resident teachers is the way.**

- **3-4 small teachers loaded in FP16** (total ~2-4GB VRAM)
- Teacher forward pass per batch = ~30-50% compute overhead
- **Zero disk overhead**, flexible (swap teachers anytime)
- Complement with MiniPLM-style data reweighting (free)

**Recommended teacher ensemble:**
1. Qwen3-0.6B (~1.2GB) - best quality sub-1B, 36T tokens
2. Mamba-370M (~0.74GB) - SSM architecture diversity
3. SmolLM2-135M (~0.27GB) - cheap reference, 2T tokens
4. Pythia-160M (~0.32GB) - deduped Pile, interpretable
Total: ~2.5GB VRAM for 4 diverse teachers

**Alternative for large teachers** (1B+): Process top-16 logits for first 2-3B tokens (~130-200GB), use online for the rest.

---

