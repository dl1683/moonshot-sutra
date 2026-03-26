# Sutra Research Log (Fresh Start — 2026-03-25)

**This document starts clean. Prior research (12K+ lines) was products of path-dependent assumptions and implementation-specific conclusions. The data below is what we're building from.**

---

## Hardware Constraints

- **GPU:** Single NVIDIA RTX 5090 Laptop (24GB VRAM)
- **RAM:** 68GB system
- **No cloud, no clusters, no multi-GPU**
- **Data:** 22.9B tokens available, 246 shards, 18 sources, 16K custom BPE tokenizer
- **Target:** Trainable from scratch on this hardware, deployable on edge (phones, laptops, embedded)

---

## Competitive Landscape (to be expanded by T+L sessions)

| Model | Params | Training Data | Key Benchmarks |
|-------|--------|--------------|----------------|
| Pythia-160M | 160M | 300B tokens (The Pile) | HellaSwag ~30%, PIQA ~62%, ARC-E ~43% |
| SmolLM2-135M | 135M | 2T tokens | HellaSwag 42.1%, PIQA 68.4% |
| Phi-4 | 5.6B | ~10T tokens | frontier-class |
| Qwen3-4B | 4B | ~18T tokens | frontier-class |
| Gemma-3-1B | 1B | ~6T tokens | strong mid-range |

**Key insight:** Training tokens matter enormously at this scale. SmolLM2-135M trained on 2T tokens — that's a 100x+ data advantage over anything we can do in a single GPU budget. The architecture must be dramatically more data-efficient to compensate.

---

## Research Areas (2025-2026 Literature)

### 1. DeepSeek Innovations

#### 1.1 Engram — Conditional N-gram Memory (Jan 2026)

**Paper:** arXiv:2601.07372 — "Conditional Memory via Scalable Lookup"

**Core idea:** Separate knowledge retrieval from reasoning computation. Instead of forcing transformer layers to reconstruct common patterns (entities, phrases, syntax) through expensive matrix multiplications, store them in a hash-indexed embedding table with O(1) lookup.

**How it works:**
1. Vocabulary projection collapses token IDs into canonical form (~23% compression)
2. N-gram hashing extracts suffix 2-grams and 3-grams, hashes through K independent heads
3. Context-aware gating compares retrieved memory against current hidden state — if memory contradicts global context, gate approaches zero
4. Fusion via depthwise causal convolution + SiLU + residual

**Results (27B scale):** MMLU +3.0, BBH +5.0, ARC-Challenge +3.7, HumanEval +3.0, MATH +2.4. Long-context NIAH: 97.0 vs 84.2. Embedding table can be offloaded to CPU DRAM with <3% throughput penalty.

**Why this matters for small models:** At 98M params, a huge fraction of capacity is spent memorizing common patterns. Offloading even a fraction to a lookup table frees neural capacity for reasoning. The embedding table can be arbitrarily large because it lives in CPU RAM, not GPU VRAM. The gating mechanism is lightweight (a few projections + sigmoid + conv).

**Optimal placement:** Layers [2, 15] — early placement offloads static pattern reconstruction, later placement benefits from richer contextual gating.

**Sparsity allocation:** 20-25% of sparse params to Engram is optimal (U-shaped scaling law).

**Relevance:** O1 (frees capacity for reasoning), O2 (knowledge failures = check the table), O3 (community contributes domain-specific N-gram tables), O4 (store facts once instead of learning from millions of examples), O5 (O(1) lookup vs neural forward pass).

#### 1.2 Multi-Token Prediction (Dec 2024)

**Paper:** DeepSeek-V3 (arXiv:2412.19437), building on Meta's MTP (arXiv:2404.19737, ICML 2024)

**Core idea:** Instead of predicting only the next token, simultaneously predict next D+1 tokens. DeepSeek's key innovation: predictions are sequential (maintaining causal chain), not parallel.

**Architecture:** Each MTP module = shared embedding + shared output head + one transformer block + projection matrix. Hidden state at depth k concatenates previous depth's state with ground-truth future token embedding.

**Training loss:** `L = L_main + (lambda/D) * sum(L_k)` where lambda=0.3, D=1.

**Key property:** MTP modules are DISCARDED at inference. Zero overhead. OR retain for speculative decoding → 1.8x inference speedup (85-90% acceptance rate).

**Small-scale validation:** BabyLM challenge — even 130M-param models trained on just 10M tokens showed improvements, particularly on entity tracking and discourse modeling.

**Mathematical principle:** Denser supervision — same hidden state forced to encode information about future token distribution, not just immediate next token. Maximizes `I(h_t; t_{t+1}, ..., t_{t+D})` instead of just `I(h_t; t_{t+1})`.

**Relevance:** O1 (forward-looking representations), O4 (D+1 supervision signals per example = direct data efficiency multiplier), O5 (zero inference overhead or 1.8x speedup via speculative decoding).

#### 1.3 Native Sparse Attention — NSA (Feb 2025)

**Paper:** arXiv:2502.11089 (ACL 2025)

**Core idea:** Three parallel sparse attention branches trained natively (not post-hoc): compression branch (block-level coarse attention), selection branch (top-n fine-grained blocks), sliding window branch (local patterns). Combined via learned sigmoid gates.

**Results (27B, 270B tokens):** Outperforms full attention on 7/9 metrics. Long-context: +3.2%. Reasoning after fine-tuning: +5.4pp. Training speedup at 64K context: 9x forward, 6x backward.

**Mathematical principle:** Multi-resolution attention — rate-distortion decomposition where compression captures low-frequency global structure and selection recovers high-frequency details.

**Note for small models:** At 512-token contexts, full attention is already cheap (262K ops per head). NSA's value appears at 4K+ contexts. The multi-branch gating idea is more interesting than the sparsity itself at our scale.

### 2. Geometric Representations

#### 2.1 Hyperbolic Neural Networks

**Foundational:** Nickel & Kiela (NeurIPS 2017) — Poincare Embeddings. 2D hyperbolic embeddings achieve MAP 0.989 on WordNet; 200D Euclidean achieves only 0.87. **100x dimension compression with better quality.**

**Why:** Hyperbolic space has exponential volume growth (e^r vs r^d). Trees with branching factor b have ~b^d nodes at depth d. Euclidean embedding needs O(b^d) dims. Hyperbolic space: 2 dims suffice.

**HELM — Hyperbolic LLMs (NeurIPS 2025, arXiv:2505.24722):** First fully hyperbolic LLMs at billion scale. Mixture-of-Curvature Experts (HELM-MiCE) where each expert operates at different curvature. Hyperbolic Multi-Head Latent Attention (HMLA). **Consistent 4% gains over Euclidean architectures (LLaMA, DeepSeek) on MMLU, ARC.** Especially strong on harder reasoning benchmarks.

**HypLoRA (2024, arXiv:2410.04010):** Hyperbolic parameter-efficient fine-tuning. Up to **17.3% improvement over Euclidean LoRA**. Key finding: pre-trained embeddings already exhibit inherent hierarchical structure that hyperbolic adapters exploit.

**Mixed-Curvature / Product Manifolds (ICLR 2019):** Real data has mixed structure — some tree-like (hyperbolic), some cyclical (spherical), some flat (Euclidean). Product manifolds combine spaces with different curvatures. Dramatic reduction in embedding distortion.

**Practical:** Trains on single GPU. Main overhead is exp/log maps (element-wise). Lorentz model more stable than Poincare ball. Can be adopted incrementally (just embeddings, or attention, or fully hyperbolic).

**Relevance:** O1 (4% benchmark gains), O2 (curvature is interpretable/tunable), O4 (structural priors reduce data needs), O5 (fewer dimensions needed = parameter savings). **Directly aligned with "Intelligence = Geometry" thesis — literally using better geometry.**

#### 2.2 Sheaf Neural Networks

**Core idea:** A sheaf assigns a vector space to each node and edge of a graph, plus linear maps (restriction maps) encoding HOW information at one node relates to neighbors. The sheaf Laplacian generalizes the standard graph Laplacian — standard GNNs are the trivial-sheaf special case.

**Key papers:**
- Neural Sheaf Diffusion (Bodnar et al., NeurIPS 2022, arXiv:2202.04579): Standard GNNs implicitly assume trivial sheaf (all restriction maps = identity). Learning non-trivial sheaves controls oversmoothing and handles heterophily.
- Copresheaf Topological NNs (NeurIPS 2025, arXiv:2505.21251): Unification — copresheaves subsume GNNs, attention, sheaf NNs, and topological NNs within single formalism. Enables heterogeneous task-specific latent spaces.
- Sheaf-theoretic perspective on attention (Jan 2026, arXiv:2601.21207): Attention mechanisms ARE cellular sheaves. Algebraic-topological basis for studying local feature alignment and global consistency.
- Sheaf Cohomology of Predictive Coding (2024, arXiv:2511.11092): Linear predictive coding = diffusion under sheaf Laplacian. Cohomology characterizes irreducible error patterns that more inference cannot fix.

**Why for small models:** Standard attention uses one shared geometry for ALL token relationships. Sheaves let a small model express relationship-specific geometries without proportional parameter increase. Restriction maps parameterize a family of transformations, not one.

**Practical:** Overhead is modest — restriction maps are small d×d linear maps per edge. Most work is on graphs, not sequences; adapting to autoregressive LM is novel territory but the copresheaf framework shows transformers are already a special case.

**Relevance:** O1 (richer per-parameter expressiveness), O2 (modular restriction maps = localizable fixes), O3 (relationship-specific components can be independently improved), O4 (structural priors reduce search space).

### 3. Alternative Architectures

#### 3.1 Hyena Edge / Gated Convolutions (Liquid AI, ICLR 2025)

**Core idea:** Replace 2/3 of attention layers with gated long convolutions (Hyena-Y family). Discovered via STAR evolutionary architecture search optimized for target hardware.

**Hyena operator:** Interleaves implicitly parametrized long convolutions (filters from small FFN) with multiplicative element-wise gating. Hyena-Y variant (no convolutions in gates) won the search.

**Results (1.3B, 100B tokens):**
| Metric | GQA-Transformer++ | Hyena Edge |
|--------|-------------------|------------|
| HellaSwag | 49.3% | **52.8%** |
| PIQA | 71.1% | **72.3%** |
| WinoGrande | 51.4% | **54.8%** |

On-device: 30% faster prefill/decode, less RAM at all sequence lengths. Subquadratic complexity: O(N log N) vs O(N^2).

**Relevance:** O1 (matches or beats transformers), O5 (30% faster on edge hardware, subquadratic). The key finding: **you don't need attention for every layer** — 2/3 can use gated convolutions and the model gets BETTER.

#### 3.2 Elastic Depth / Adaptive Compute

**LoopFormer (ICLR 2026):** Budget-conditioned elastic depth with shortcut-consistency training.

**MoR (NeurIPS 2025):** Token-level adaptive recursion depth, 135M-1.7B scale. Warning: capacity bottleneck at 135M; routing overhead can exceed benefits at small scale.

**MiniPLM (ICLR 2025):** Offline KD for pre-training — 2.2x acceleration, 2.4x data efficiency.

**Mamba-3 (ICLR 2026):** Hybrid SSM+attention, complex-valued states, MIMO.

---

## What's Next

This research feeds into fresh T+L design sessions. No architecture is assumed. The T+L workflow will evaluate these options (and others it discovers through its own research) against the 5 outcomes and propose the best path forward.
