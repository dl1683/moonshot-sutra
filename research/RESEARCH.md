# Sutra Research Log

**Field research informing architecture design. This document contains knowledge about the field — not implementation-specific conclusions from our own experiments.**

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

#### 1.2 Multi-Token Prediction — Deep Dive (Dec 2024 – Mar 2026)

**Primary sources:** Meta (arXiv:2404.19737, ICML 2024), DeepSeek-V3 (arXiv:2412.19437), MuToR (arXiv:2505.10518, NeurIPS 2025), MTP Curriculum (arXiv:2505.22757, ACL 2025), Token Order Prediction (arXiv:2508.19228), MTP Self-Distillation (arXiv:2602.06019)

**Core idea:** Instead of predicting only the next token, predict next D+1 tokens simultaneously, extracting denser supervision from each training example. Maximizes `I(h_t; t_{t+1}, ..., t_{t+D})` instead of just `I(h_t; t_{t+1})`.

---

##### 1.2.1 Meta's Parallel MTP (arXiv:2404.19737, ICML 2024)

**Architecture:** n independent output heads on top of a shared transformer trunk. Each head is a single transformer layer responsible for predicting a specific future offset token. All heads share a single unembedding matrix (linear projection from hidden dim d to vocab size V). This shared unembedding is critical — replicating it n times would require (d, nV) matrices, prohibitively expensive at scale.

**Training loss:** Multi-token cross-entropy factorized across n independent heads:
```
L_MTP = (1/n) * sum_{k=1}^{n} CE(head_k(trunk(x)), t_{i+k})
```
Used as auxiliary loss alongside standard next-token prediction.

**Memory-efficient training (KEY TECHNIQUE):** The bottleneck is that vocab size V >> hidden dim d, so materializing logit gradients for all n heads simultaneously would be prohibitive. Solution: sequentially compute forward and backward passes for each head, accumulating gradients at the trunk. Iteration starts from the head furthest from the trunk. This reduces peak GPU memory to approximately that of a single-head model while training n heads. The only overhead is a slight loss of FSDP communication-computation overlap from sequential backward passes — fixable with better implementation.

**Results by model size (300M to 13B, code benchmarks):**
- **300M–1.3B: MTP performs WORSE than NTP baseline.** The architectural overhead of independent prediction heads is costly without sufficient model capacity to leverage lookahead information.
- **7B (4-token prediction, 200B code tokens):** +3.8% MBPP pass@1, +1.2% HumanEval pass@1
- **13B (4-token prediction):** +12% HumanEval, +17% MBPP vs comparable NTP models

**Optimal head count:** n=4 generally best for token-level transformers. n=8 best for byte-level transformers.

**Code vs natural language:**
- Code: significant gains (HumanEval, MBPP). Code has longer-range structural dependencies requiring understanding of future patterns.
- Natural language: mixed. Summarization improved (ROUGE-L F1 gains). Multiple-choice NLP benchmarks: no significant improvement, sometimes degradation with n=4.

**Self-speculative decoding (inference):**
- 4-token prediction: **3.0x speedup on code, 2.7x on text** (2.5 accepted tokens from 3 suggestions)
- 8-byte prediction: **6.4x speedup**
- Modules discarded if not used for speculation — zero inference overhead

**Byte-level variant (IMPRESSIVE):** 7B byte-level transformer on 314B bytes: +67% problems solved on MBPP, +20% on HumanEval vs next-byte baseline. Trained on 1.7x LESS data than token-level models. This is the strongest data efficiency evidence.

**CRITICAL FINDING FOR SUTRA: MTP does NOT help small models (< ~1.3B) in Meta's parallel formulation.** The paper's own authors attribute this to capacity limitations — the model cannot simultaneously learn the trunk representation AND the lookahead structure. This was "a likely reason why multi-token prediction has so far been largely overlooked."

---

##### 1.2.2 DeepSeek's Sequential MTP (arXiv:2412.19437)

**Key difference from Meta:** Sequential prediction chain instead of parallel heads. Each MTP module's prediction at depth k conditions on the output of depth k-1, maintaining a complete causal chain at every depth.

**Architecture per MTP module (4 components):**
1. **Shared embedding layer** Emb(·) — shared with main model
2. **Shared output head** OutHead(·) — shared with main model
3. **Unshared Transformer block** TRM_k(·) — one per depth k
4. **Unshared projection matrix** M_k ∈ R^{d × 2d} — one per depth k

**Exact computation at depth k:**
```
Step 1 — Projection (combine previous depth output with target token embedding):
  h'_i^k = M_k * [RMSNorm(h_i^{k-1}) ; RMSNorm(Emb(t_{i+k}))]

  where [;] is concatenation, h_i^{k-1} is the hidden state from depth k-1
  (or the main model output for k=1), and t_{i+k} is the ground-truth
  future token at offset k.

Step 2 — Transformer processing:
  h_{1:T-k}^k = TRM_k(h'_{1:T-k}^k)

  Note: sequence length reduces by 1 at each depth (T-k positions)

Step 3 — Prediction:
  P_{i+k+1}^k = OutHead(h_i^k)

  Shared output head maps to vocabulary logits
```

**Training loss:**
```
Per-depth loss:    L_MTP^k = -(1/T) * sum_{i=2+k}^{T+1} log P_i^k[t_i]
Aggregate MTP:     L_MTP = (lambda/D) * sum_{k=1}^{D} L_MTP^k
Total:             L = L_main + L_MTP
```

**Hyperparameters used by DeepSeek-V3:**
- D = 1 (predict ONE additional token beyond next-token). Despite the general framework supporting D > 1, they found D=1 sufficient.
- lambda = 0.3 for first 10T tokens, reduced to lambda = 0.1 for remaining 4.8T tokens

**Parameter overhead in DeepSeek-V3:** Total MTP module = 14B params (on top of 671B main model = ~2% overhead). Of those 14B, ~11.5B are unique (the 0.9B embedding + 0.9B output head are shared with the main model). Each MTP module is essentially one full transformer block + one d×2d projection matrix.

**Inference:** MTP modules DISCARDED entirely — zero inference overhead. OR retained for speculative decoding:
- **Acceptance rate for next token: 85-90%** (above 80% threshold for useful speculation)
- **1.8x throughput speedup** via speculative decoding
- Practical deployments (SGLang on AMD GPUs): 1.25-2.11x speedup (Random dataset), 1.36-1.80x (ShareGPT)

**Why sequential > parallel for DeepSeek:** The causal chain means each depth's prediction benefits from knowing what the model predicted at the previous depth. This is more principled than parallel heads that each independently guess different future tokens from the same trunk representation.

---

##### 1.2.3 Small-Model Challenges and Solutions

**THE PROBLEM:** Meta's results show MTP hurts at 300M-1.3B scale. The MTP Curriculum paper (ACL 2025) explains WHY: "smaller language models struggle with the MTP objective due to the fact that they struggle to deal with morphological and semantic dependencies between multiple tokens at once from the get-go."

**Solution 1 — MTP Curriculum (arXiv:2505.22757, ACL 2025):**
- Tested on 1.3B and 3B models, MiniPile dataset (~1.7B subword tokens)
- Head configurations tested: k ∈ {1, 2, 4, 8}, both Linear Layers (LL) and Transformer Layers (TL) heads
- **Forward curriculum:** Start at k=1 (NTP only), gradually increase active heads over training. Schedule: k_current(e) = min(k_max, floor(e/(E/k_max)) + 1). The model learns single-token prediction first, then progressively harder multi-token prediction.
- **Reverse curriculum:** Start at k=k_max, reduce to k=1. The opposite ramp.
- Results (1.3B subword, 4 LL heads):
  - NTP baseline: 1.19 bits-per-byte
  - Static MTP: 1.19 BPB (no improvement!)
  - Forward curriculum: 1.16 BPB (-2.46%)
  - Reverse curriculum: 1.12 BPB (-5.99%)
  - BLEU score: NTP 3.96, Forward +4.05%, Reverse +15.80%
- Trade-off: Reverse curriculum gives best NTP/quality but loses speculative decoding benefit (heads converge to NTP behavior). Forward curriculum retains speculative decoding while still improving.
- k=2-4 with Transformer Layer heads showed best trade-offs
- Byte-level models tolerate higher k values better than subword models
- **No extra compute overhead** — just changes which heads are active during training

**Solution 2 — MuToR: Register Tokens (arXiv:2505.10518, NeurIPS 2025):**
- Instead of adding prediction heads, interleave learnable register tokens into the input sequence
- Each register predicts a future token at randomly sampled offset d ∈ {1,...,d_max}
- **Architecture:** Single shared register embedding (~2K extra params vs 110M-550M for head-based approaches). Registers get position IDs matching their prediction targets.
- **Attention mask rules:** Regular tokens CANNOT attend to registers. Registers CANNOT attend to other registers. Registers attend ONLY to preceding regular tokens. This prevents interference with NTP objective.
- **Inference:** Registers discarded — zero overhead
- Tested on Gemma 2B, Llama 3 8B, LlamaGen-B (111M), GPT2-L
- Results (Gemma 2B): GSM8K 42.10% vs 38.87% baseline, MATH500 68.33% vs 66.09%
- Outperforms Meta's multi-head baseline across all experiments with negligible parameter overhead
- Compatible with LoRA: "LoRA-MuToR matches or exceeds full-finetuning Next-Token performance"
- In optimal setup, each register predicts up to 15 future tokens — would require 14 additional transformer heads in prior approaches
- **KEY INSIGHT:** The register approach decouples MTP from model capacity. A 111M model can still do useful multi-token prediction because the register doesn't compete with the trunk for capacity.
- **LIMITATION:** Sequence length increases ~100% (one register per token). Sparse variant (80 random registers) achieves similar results with only 30% sequence length increase.

**Solution 3 — Token Order Prediction / TOP (arXiv:2508.19228):**
- Argues MTP is fundamentally TOO DIFFICULT as an auxiliary objective for small models
- Alternative: instead of predicting exact future tokens, predict the ORDER of upcoming tokens by proximity using a learning-to-rank loss (ListNet)
- Only requires a single additional unembedding layer (vs multiple transformer layers for MTP)
- Tested on 340M, 1.8B, 7B parameters (FineWeb-Edu, 52-104B tokens)
- **TOP improved even 340M models** on several benchmarks (e.g., +2.50 on SciQ)
- While MTP's 7B variant underperformed NTP baselines on standard NLP benchmarks, TOP showed consistent improvements at all scales
- **IMPORTANT FOR SUTRA:** This is the most promising MTP variant for sub-1B models. It extracts the "planning" benefit without the capacity overhead.

**Solution 4 — Self-Distillation MTP (arXiv:2602.06019):**
- Converts a pretrained NTP model into a standalone MTP model using online self-distillation
- No auxiliary verifier or specialized inference code needed
- GSM8K: 3x+ faster decoding at <5% accuracy drop
- Retains exact same architecture as pretrained checkpoint
- Most relevant for post-training acceleration, less for from-scratch data efficiency

---

##### 1.2.4 Training Cost Analysis

**Meta's parallel approach (D extra heads):**
- Each head = one transformer layer + shared unembedding. For a model with hidden dim d and vocab V:
  - Per-head params: ~4d^2 (one transformer layer) = relatively small
  - Shared unembedding: d × V (not replicated per head — critical optimization)
  - For d=768, V=16K, D=2: ~2.4M per head × 2 = ~4.8M extra params (~5% of a 100M model)
- Memory-efficient training: sequential forward/backward per head means peak memory ≈ single-head model
- FLOPs overhead: approximately (D × one_transformer_layer_cost) per training step. For D=2 on a 100M model with ~24 layers, overhead ≈ 2/24 = ~8% more forward-pass FLOPs, but the unembedding (d→V projection) is the dominant cost per head
- **Practical estimate for 100M model, D=2:** ~5-15% more training FLOPs, ~5% more parameters, negligible extra peak memory (with sequential backward pass)

**DeepSeek's sequential approach (D modules):**
- Per module: one transformer block + one d×2d projection matrix
- DeepSeek-V3: 14B MTP on 671B main model = 2% parameter overhead
- At our scale (100M model, d=768): one module ≈ 4d^2 (transformer block) + 2d^2 (projection) = ~6 × 768^2 ≈ 3.5M params per module
- For D=1 (DeepSeek's choice): ~3.5M extra ≈ 3.5% overhead on 100M model
- For D=2: ~7M extra ≈ 7% overhead on 100M model
- Sequential forward means each module sees shorter sequence (T-k), slightly reducing compute per module

**MuToR register approach:**
- ~2K extra parameters (negligible)
- BUT: sequence length doubles (one register per token) or +30% (sparse variant)
- This means ~2x attention cost per training step (for full register placement)
- Sparse variant: ~1.3x training cost

**Curriculum approach:**
- Zero extra cost over standard MTP — just changes training schedule

**Summary for 100M model on 24GB VRAM:**
| Approach | Extra Params | Extra VRAM (training) | Extra FLOPs/step |
|----------|-------------|----------------------|-----------------|
| Meta parallel D=2 | ~5M (5%) | Negligible (sequential backward) | ~10-15% |
| DeepSeek sequential D=1 | ~3.5M (3.5%) | ~3.5% | ~5-8% |
| MuToR (full) | ~2K (0%) | ~2x (seq length doubles) | ~2x |
| MuToR (sparse 80) | ~2K (0%) | ~1.3x | ~1.3x |
| TOP (rank loss) | ~0.6M (0.6%) | Negligible | ~3-5% |
| Curriculum wrapper | 0 (over base MTP) | 0 | 0 |

---

##### 1.2.5 Speculative Decoding Details

**How it works with MTP:**
1. The MTP head(s) draft D candidate next tokens in a single forward pass
2. The main model verifies ALL D candidates simultaneously in one forward pass (this is the key — verification is parallel)
3. Accept the longest prefix of candidates that match the main model's distribution
4. If all D candidates accepted: D+1 tokens generated in 2 forward passes (vs D+1 passes for standard autoregressive)
5. If first candidate rejected: fall back to standard single-token generation for that step (no worse than baseline)

**DeepSeek-V3 numbers:** D=1 (predict one extra token), acceptance rate >80% (85-90%), 1.8x throughput speedup. Practical: 1.25-2.11x depending on dataset and hardware.

**Meta's numbers:** D=3 (4-token prediction), self-speculative decoding. Code: 3.0x speedup. Text: 2.7x speedup. Byte-level with D=7: 6.4x speedup.

**Compatibility with small models:** Speculative decoding is MOST beneficial for memory-bound inference (where the model fits in memory but generation is bottlenecked by sequential forward passes). For small models on edge devices, inference is often compute-bound, not memory-bound. The speedup may be lower than at DeepSeek-V3 scale. However, for latency-sensitive applications (chatbots, real-time translation), even modest speedups from speculation are valuable.

**Implementation:** The MTP heads can be retained at inference specifically for speculation, or discarded if speculation is not needed. This is a deployment-time choice, not an architecture choice.

---

##### 1.2.6 Data Efficiency — The Core Claim

**The hypothesis:** D+1 supervision signals per training example should multiply data efficiency. Training on N tokens with MTP D=2 should approximate training on 3N tokens with NTP.

**Evidence FOR:**
- Byte-level results are the strongest evidence: 7B byte-level model with 8-byte prediction trained on 314B bytes (116B token equivalent) with 1.7x LESS data beat the token-level NTP baseline. This directly demonstrates data efficiency.
- BabyLM challenge: even 130M-param models on just 10M tokens showed improvements on entity tracking and discourse modeling
- MiniPile curriculum results: MTP with curriculum on 1.7B tokens achieved perplexity improvements that typically require more training data
- The mathematical argument is sound: each position's hidden state must encode information about multiple future tokens, creating a richer gradient signal

**Evidence AGAINST a simple "D+1 multiplier":**
- Small models (< 1.3B) see NO data efficiency gain from standard MTP — the capacity cost eats the benefit
- NLP benchmarks (not code) show inconsistent improvements — the benefit is task-dependent
- No paper has shown a clean "train on N tokens with MTP = train on 2N tokens with NTP" comparison with matched compute
- The perplexity gains from MTP are modest (2-6% BPB improvement with curriculum) — not the 2-3x you'd expect from a true multiplier

**Honest assessment:** MTP improves data efficiency, but NOT as a simple linear multiplier. The benefit is:
1. **Representation quality improvement** (hidden states encode richer structure) — this is real and consistent at sufficient scale
2. **Implicit planning** (model learns to "think ahead") — strongest for code and structured tasks
3. **Regularization effect** (harder objective reduces overfitting) — valuable when training for multiple epochs on limited data
4. **The multiplier framing is misleading.** A more accurate statement: "MTP extracts 20-50% more learning from the same data at sufficient model scale, with the benefit concentrated on generative/planning tasks."

---

##### 1.2.7 Interaction with Other Techniques

**MTP + Knowledge Distillation:**
- No direct study of MTP + KD during pre-training found
- MiniPLM (ICLR 2025) achieves 2.2x pre-training acceleration and 2.4x data efficiency through OFFLINE KD — conceptually compatible with MTP (different supervision signals)
- Self-distillation MTP (arXiv:2602.06019) uses online distillation to convert NTP models to MTP — shows the two techniques compose naturally
- Cross-architecture distillation (MOHAWK framework, Transformer→SSM) operates at the mixing-matrix level and should be orthogonal to the prediction objective
- **Hypothesis for Sutra:** MTP + KD could be multiplicative — KD provides richer per-token supervision from the teacher, MTP provides richer per-position supervision from future tokens. Both attack data efficiency from different angles.

**MTP + Elastic Compute / Early Exit:**
- No direct study found combining MTP training with early-exit inference
- Conceptual compatibility: MTP trains the hidden states to encode more information (about future tokens), which should make early-exit classifiers more accurate at shallower depths
- Early-exit tokens see fewer layers of quantization error accumulation — natural synergy
- LoopFormer's shortcut-consistency training is conceptually similar to MTP curriculum (both train the model to produce good outputs at variable compute levels)
- **Potential conflict:** MTP adds training cost, early-exit reduces inference cost. The training-time investment of MTP should pay off in more capable early-exit decisions at inference time.

**MTP + Non-Standard Architectures (Hyena, SSMs, Mamba):**
- MTP is architecture-AGNOSTIC in principle — it's a training objective, not a structural constraint. Any model with a hidden representation and an output head can add MTP heads.
- For SSMs/Mamba: MTP heads would branch off the final hidden state (same as for transformers). No architectural incompatibility.
- For Hyena/gated convolutions: same principle — attach prediction heads to the shared trunk output
- The DeepSeek-V3 sequential formulation (TRM_k blocks) assumes transformer blocks in the MTP modules — for SSM-based models, these could be replaced with SSM blocks or even simple MLPs
- **UNTESTED TERRITORY:** No published results on MTP + pure SSM or MTP + Hyena. This is a potential novel contribution for Sutra.

---

##### 1.2.8 Relevance to Sutra (100M-500M, Single GPU, From Scratch)

**What works at our scale:**
1. **TOP (Token Order Prediction)** — The most promising for sub-1B models. Demonstrated improvements at 340M params. Single extra unembedding layer. Negligible overhead. Works on standard NLP benchmarks where regular MTP fails.
2. **MTP Curriculum (forward schedule)** — If using standard MTP heads, the curriculum is mandatory at small scale. Static MTP showed zero improvement at 1.3B on MiniPile. Curriculum achieved 2.5-6% BPB improvement.
3. **MuToR registers (sparse variant)** — Tested at 111M (LlamaGen-B). Negligible params. 30% sequence overhead with sparse placement. Decouples MTP from model capacity.
4. **DeepSeek sequential D=1** — The most parameter-efficient MTP variant. Only 3.5% overhead on 100M model. The sequential chain is more principled than parallel heads. Worth testing with curriculum scheduling.

**What probably does NOT work at our scale:**
- Meta's parallel MTP without curriculum — demonstrated to hurt below 1.3B
- High D values (D > 2) — capacity-limited models cannot leverage deep prediction chains
- Full MuToR register placement — 2x sequence length is too expensive for training

**Recommended configuration for Sutra prototype:**
- DeepSeek-style sequential MTP with D=1 + forward curriculum schedule
- OR: TOP as auxiliary loss (simpler, proven at 340M)
- Start MTP/TOP after initial convergence (delay to step 3K+, per our "delayed novel mechanisms" rule)
- lambda=0.3 initially, decay to 0.1 as training progresses (following DeepSeek schedule)
- Memory-efficient: sequential backward pass through heads

**The prize:** Even modest data efficiency gains (20-30%) are hugely valuable when training on limited data (our ~22.9B tokens vs competitors' 300B-2T tokens). MTP/TOP is one of the cheapest ways to squeeze more learning from each training example.

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

**LoopFormer (ICLR 2026, arXiv:2602.11451):** Budget-conditioned elastic depth with shortcut-consistency training.

**Architecture:** Shared transformer blocks looped L times. Each iteration conditioned on (t, delta_t) via AdaLN — sinusoidal Fourier features → 2-layer MLP → 4 modulation vectors (gate_msa, scale_msa, gate_mlp, scale_mlp). Zero-initialized gates = identity at init. Config tested: 3 blocks × 8 loops, d=2048, ~278M unique params, 24-layer equivalent FLOPs.

**Shortcut-consistency training:** L = L_full + 0.1*L_short + 0.1*L_consistency. Each batch processed TWICE: full trajectory (all L iterations) and random short trajectory (S < L steps with random step sizes summing to 1.0). Consistency = MSE(stopgrad(x_full), x_short). Self-distillation from full-compute to partial-compute. Training overhead: ~1.5x FLOPs.

**Results (278M, 25B tokens):** 3x8 LoopFormer (278M params) nearly matches 24x1 non-looped (1B params): 44.81% vs 45.27% avg across 10 tasks. 3.6x parameter reduction for 0.46% accuracy loss. At half budget (3x4): 43.73% vs 44.93% (non-looped 12x1). At quarter budget (3x2): 40.36% vs 42.73% — **non-looped baseline wins at each fixed budget.**

**Critical limitation for small models:** SEQUENCE-LEVEL budget only. All tokens get same compute. No per-token routing. Paper explicitly acknowledges this. LoopFormer solves "one model, many global budgets" — NOT "easy tokens exit early, hard tokens get full compute." Also not tested below 278M params.

**What to steal:** AdaLN time-conditioning (cleanest way to tell shared layers which iteration), shortcut-consistency training (self-distillation objective for elastic compute).

**AdaPonderLM (Feb 2026, arXiv:2603.01914):** Deterministic gate pruning for TOKEN-LEVEL adaptive depth. Works at Pythia-70M scale. Gate mechanism prunes layers per-token based on learned importance. ~6-8% FLOPs reduction. Modest savings but correct granularity.

**MoR — Mixture of Recursions (NeurIPS 2025, arXiv:2507.10524):** Token-level adaptive recursion depth via expert-choice routing, 135M-1.7B scale. Warning: **capacity bottleneck at 135M** — recursive capacity insufficient when shared blocks must serve all roles. Routing overhead can exceed benefits at small scale.

**MiniPLM (ICLR 2025):** Offline KD for pre-training — 2.2x acceleration, 2.4x data efficiency.

**Mamba-3 (ICLR 2026):** Hybrid SSM+attention, complex-valued states, MIMO.

### 4. Quantization-Native Architecture Design (Outcome 5: Inference Efficiency)

**Goal:** Design the architecture for low precision from day one. Not "train FP16, quantize later" — but "train in low precision, deploy in low precision, every design choice serves quantization."

#### 4.1 BitNet b1.58 — Ternary Quantization During Training

**Papers:** arXiv:2310.11453 (BitNet), arXiv:2402.17764 (BitNet b1.58), arXiv:2407.09527 (Reloaded), arXiv:2411.05882 ("When are 1.58 bits enough?"), arXiv:2504.12285 (BitNet b1.58 2B4T)

**How ternary QAT works:**
1. Maintain FP16/BF16 "shadow weights" throughout training (the real learned parameters)
2. On each forward pass, quantize weights to {-1, 0, +1} via absmean quantization: `W_q = round(W / mean(|W|))` clamped to {-1, 0, 1}
3. Activations quantized to INT8 via absmax per-token quantization
4. Backward pass uses Straight-Through Estimator (STE): gradients flow through the quantizer as if rounding never happened, updating the FP16 shadow weights
5. At inference, only ternary weights are kept — shadow weights discarded

**BitNet b1.58 Reloaded (small-scale validation, arXiv:2407.09527):**
- First study of 1.58-bit QAT on small networks (100K–48M params, Mistral-like)
- Introduces median-based scaling variant (vs. original mean-based): more robust for small models and extreme LR schedules
- **Scaling law discovered: a 1.58-bit model needs ~2x the hidden dimension to match FP16 quality at the same task.** This means a 1.58-bit 200M model ≈ a FP16 100M model in capacity, but uses ~1/10th the memory
- Tested internal dims 32, 64, 128, 256 — scaling law holds across all
- Acts as implicit regularizer — sometimes OUTPERFORMS full-precision counterparts

**"When are 1.58 bits enough?" (arXiv:2411.05882):**
- Bottom-up exploration from XOR to MLPs to GNNs to LMs
- Simple tasks (XOR): 8 hidden units needed in 1.58-bit vs. 2 in FP16
- Practical tasks (text/node classification): 1.58-bit achieves 98.5–98.8% of FP16 performance when sufficiently overparameterized
- Capacity compensation is SUB-proportional to bit-width decrease — you don't need 10x params for 10x compression, more like 2x

**BitNet b1.58 2B4T (arXiv:2504.12285):**
- First natively trained 1-bit LLM at 2B scale, 4T training tokens, MIT license
- Competitive with LLaMA 3.2 1B, Gemma-3 1B, Qwen2.5 1.5B, SmolLM2 1.7B
- Avg benchmark score 55.01 vs GPTQ-INT4 Qwen2.5 1.5B: 52.15, AWQ-INT4: 51.17
- Memory footprint: 0.4GB vs 0.7GB for INT4 alternatives
- bitnet.cpp inference: 2.37–6.17x speedup over llama.cpp, 82% energy reduction

**Independent validation status:** Primarily Microsoft's own evaluations. Community testing on HuggingFace and LMSYS Arena. No large-scale independent reproduction study found as of March 2026. The 2B4T model is publicly available under MIT license, so independent verification is possible. Microsoft's own disclaimer: "not recommended for commercial applications without further testing."

**Relevance to Sutra (100M–500M target):**
- At our scale (100M–500M), the 2x hidden-dim scaling law is KEY. A 200M ternary model has the capacity of ~100M FP16 but uses 1/10th the memory at inference
- Ternary matmuls = additions and subtractions only. No multiplications. Massive inference speedup on ANY hardware
- The "16-to-1.58" training strategy (start FP16, transition to ternary) further improves quality
- Median-based scaling is critical for small models — use this, not mean-based
- **CONCERN:** The 2x capacity tax is significant at our scale. A 100M ternary model may only have 50M-equivalent intelligence. Need to evaluate whether the inference gains justify the capacity cost vs. INT4/INT8 QAT

#### 4.2 Dynamic Tanh (DyT) — Normalization Replacement

**Paper:** arXiv:2503.10622, "Transformers without Normalization" (CVPR 2025, FAIR/Meta/NYU/MIT/Princeton)

**What it is:** Drop-in replacement for LayerNorm / RMSNorm.
```
DyT(x) = gamma * tanh(alpha * x) + beta
```
where alpha, gamma, beta are learnable per-channel parameters.

**Why it exists:** Layer normalization computes per-sample statistics (mean, variance) — expensive, sequential, hard to parallelize. DyT is element-wise — no reduction operations, trivially parallel.

**The quantization connection (CRITICAL for Sutra):**
- LayerNorm/RMSNorm are a PRIMARY CAUSE of activation outliers in LLMs. The channel-wise scaling factors in normalization layers amplify certain channels disproportionately, creating "massive activations" (values >1000x typical magnitude)
- These outliers are THE reason INT4 PTQ fails catastrophically — a single outlier channel destroys the quantization range for the entire tensor
- DyT's tanh naturally bounds activations. In experiments on LLaMA-1B: max activation magnitude drops from >1400 to ~47
- Near-zero kurtosis means activations are well-behaved for quantization

**Performance:** Matches or exceeds normalized counterparts on: ViT, ConvNeXt, MAE, DINO, DiT, LLaMA, wav2vec 2.0, DNA modeling. No extensive hyperparameter tuning needed.

**Mathematical derivation:** DyT can be formally derived from RMSNorm. The paper shows an exact element-wise counterpart (DyISRU = Dynamic Inverse Square Root Unit), proving that normalization's statistical reduction was UNNECESSARY — the per-element squashing was doing the actual work.

**Relevance to Sutra:**
- DIRECT enabler of aggressive quantization. If we design with DyT from day one, we eliminate the #1 source of quantization-hostile outliers
- Simpler compute graph (no reductions) → faster inference on edge hardware
- Fewer ops per layer → better throughput in quantized inference engines
- Compatible with ternary/INT4/INT8 — the bounded activation range plays well with ALL quantization schemes
- **STRONG CANDIDATE for inclusion regardless of other architecture choices**

#### 4.3 QAT vs PTQ for Shared-Weight / Recurrent / Early-Exit Models

**The compounding error problem (validated in our experiments):**
INT4 PTQ on shared-weight recurrent models is catastrophic. Error at each pass compounds: if quantization introduces error epsilon per pass, after N passes the accumulated error grows roughly as O(N * epsilon) or worse (correlated errors can compound super-linearly). With 12 recurrent passes, even small per-pass errors destroy output quality.

**Does QAT solve it? YES, substantially:**
- QAT includes quantization noise in the training loop, so the model LEARNS to be robust to its own quantization error
- For SSMs (S5 networks), QAT achieves <1% accuracy drop on MNIST and most LRA tasks
- QAT eliminates the gap between FP8 and INT8 formats — the network trains to reduce outlier sensitivity
- The "16-to-quantized" strategy (pre-train FP16, then QAT fine-tune) is most practical for our scale

**For early-exit / elastic depth models:**
- Early exits see FEWER quantization error accumulation steps — natural advantage
- Per-exit-head quantization is an active research area (NAS over exit placement + quantization level)
- Mixed precision per layer is viable: higher precision for early exits (they matter more for easy tokens), lower precision for later layers
- **Key finding:** jointly optimizing exit architecture + quantization is a combinatorial NAS problem — but at our scale (100M–500M), the search space is tractable

**Practical recommendation for Sutra:**
1. Train in FP16/BF16 (or FP8 if hardware supports it natively)
2. Use QAT fine-tuning to INT8 activations + INT4 or ternary weights
3. If using recurrent/shared-weight passes: QAT is MANDATORY, PTQ will fail
4. For elastic compute: higher precision on early-exit paths, aggressive quantization on full-depth paths
5. Consider BitNet-style ternary QAT from scratch if the 2x capacity tax is acceptable

**Key papers:**
- EfficientQAT (ACL 2025): efficient QAT for LLMs
- Teacher Intervention (arXiv:2302.11812): faster QAT convergence via teacher guidance
- Fairy2i (arXiv:2512.02901): complex-valued {±1, ±i} weights, pushing beyond ternary
- arXiv:2407.11722: quantized pre-training for transformer LMs (shows MSE grows across layers — confirms compounding problem)

#### 4.4 Block Floating Point / Microscaling (MX) Formats

**Standard:** OCP Microscaling Formats (MX) v1.0, September 2023. Backed by AMD, Arm, Intel, Meta, Microsoft, NVIDIA, Qualcomm.

**How it works:**
- A block of K elements (typically K=32) shares a single scaling factor (exponent)
- Each element stores a small mantissa (4–8 bits)
- The shared exponent captures the block's dynamic range; individual mantissas capture relative values
- Result: better quality-per-bit than fixed-point INT formats because the dynamic range adapts per block

**Defined formats:**
| Format | Element bits | Shared scale | Effective bpw | Use case |
|--------|-------------|-------------|---------------|----------|
| MXFP8 | 8 (E5M2 or E4M3) | 8-bit per 32 elements | ~8.25 | Training + inference |
| MXFP6 | 6 | 8-bit per 32 elements | ~6.25 | Inference, mixed training |
| MXFP4 | 4 | 8-bit per 32 elements | ~4.25 | Aggressive inference |
| MXINT8 | 8 (integer) | 8-bit per 32 elements | ~8.25 | Integer-friendly hardware |

**Quality results (2025 empirical studies):**
- MXFP4 training of Llama3 7B on 100M tokens: viable, loss curves track FP16
- Mixed MX4+MX6 forward+backward achieves best quality/efficiency tradeoff
- Both INT4 and MXFP4 recover 98–99% of FP16 accuracy for large models
- **Challenge:** full MX4 training (forward + backward) shows gradient bias from block-overflow in layer-norm affine blocks → requires hybrid precision and runtime monitoring
- MicroMix (2025): mixed-precision MX achieves ≥95% FP16 zero-shot accuracy with 20–46% kernel-level speedup

**Hardware support (current as of March 2026):**
| Vendor | Hardware | MX Support |
|--------|----------|-----------|
| NVIDIA | Blackwell (RTX 5090) | FP4, FP6, FP8 natively. NVFP4 = 4-bit element + FP8 scale per 16 values (~4.5 effective bpw). ~3.5x memory reduction vs FP16, 1% or less accuracy loss |
| NVIDIA | Hopper (H100/H200) | FP8 only |
| AMD | CDNA 4 (MI350+) | MXFP4, MXFP6, MXFP8 |
| AMD | CDNA 3 (MI300X) | FP8, packed MXFP8 |
| AMD | Versal AI Edge Gen 2 | MX6, MX9 |
| Intel | Upcoming | Expected MX support |

**RTX 5090 specifics (OUR HARDWARE):**
- Native FP4 Tensor Cores — doubles TFLOPS vs FP8 on same silicon
- NVFP4: 4-bit values + FP8 scale per 16 elements + FP32 tensor-level second scale
- Demonstrated: 3x throughput vs FP16, 5.2x VRAM reduction on diffusion models
- ~1% accuracy loss on DeepSeek-R1 language tasks with PTQ to NVFP4
- **This is our deployment target format.** Design the architecture so it quantizes cleanly to NVFP4/FP4

**Relevance to Sutra:**
- MXFP4/NVFP4 is likely our optimal inference format — directly supported by our RTX 5090
- Better than INT4 because the floating-point element format handles outliers more gracefully
- The block-sharing structure means we should design layers where activation distributions are SIMILAR within blocks of 16–32 elements (which DyT helps with)
- For training: MXFP8 is safe. MXFP4 training is bleeding-edge but viable with hybrid precision
- **Architecture implication:** avoid designs that create extreme variance WITHIN blocks of 32 consecutive elements. Smooth, well-distributed activations are key

#### 4.5 Quantization-Aware Architecture Design Principles

**The fundamental insight:** Outlier channels are NOT inherent to neural networks — they are CAUSED by specific architectural and training choices. Design them out.

**Outlier-Safe Pre-Training (OSP, ACL 2025, arXiv:2506.19697) — THE key paper:**

Three interventions that prevent outlier formation during training:
1. **Muon optimizer** (instead of Adam): eliminates "privileged bases" — Adam's per-parameter learning rates create channel-wise imbalances that become outliers. Muon maintains rotation invariance
2. **Single-Scale RMSNorm** (instead of standard RMSNorm): removes the per-channel gamma scaling that amplifies certain channels
3. **Learnable embedding projection**: redistributes activation magnitudes from embedding matrices, preventing the initial outlier seed

**Results (1.4B params, 1T tokens):**
- OSP model: average kurtosis 0.04 (near-Gaussian). Standard Adam model: kurtosis 1818.56 (extreme outliers!)
- 4-bit PTQ benchmark: OSP scores 35.7/50 avg across 10 tasks. Adam baseline: 26.5/50. That's a **35% relative improvement** in quantized quality from training changes alone
- Training overhead: only 2%

**Activation function choice:**
- **ReLU**: Most quantization-friendly. 50% sparsity (all negatives zeroed). Sparse matmuls can skip zero-weighted operations. 41–74% speedup on AI accelerators vs SiLU. But: lower model quality than smooth activations
- **SiLU/GeLU**: Better model quality, denser activations, harder to quantize. The non-zero negative tail creates more complex distributions
- **DyT (tanh-based)**: Bounds activations naturally. Best of both worlds — smooth activation with bounded range
- **Practical choice for Sutra:** SiLU for quality during training, with DyT replacing normalization to control outliers. Or: ReLU if we're BitNet-style (ternary weights don't need smooth gradients as much). Needs empirical validation

**Normalization placement:**
- **Pre-norm** (before attention/FFN): more stable training, but the norm layer itself can create outliers
- **Post-norm** (after attention/FFN): less stable but some evidence of better quantization behavior
- **DyT replacement**: eliminates the question entirely — no per-sample statistics, bounded activations
- **Single-Scale RMSNorm** (from OSP): if keeping normalization, remove per-channel gamma

**Rotation/Hadamard transforms (post-hoc fixes, but informative for design):**
- QuaRot, QuIP#, SmoothRot: apply Hadamard rotations to redistribute outlier energy across channels
- These WORK but are post-hoc patches. Better to design the architecture so outliers never form
- The Hadamard insight: if all channels have similar magnitude, quantization works perfectly. Design for this property

**Architecture-level principles for quantization-native design:**
1. **No per-channel scaling in normalization** — use DyT or Single-Scale RMSNorm
2. **Use an optimizer that maintains rotation invariance** — Muon, or SGD with proper scheduling
3. **Bound activations inherently** — tanh, sigmoid, or other bounded functions in critical paths
4. **Design for smooth activation distributions within blocks** — MX formats work best when 16–32 consecutive elements have similar magnitudes
5. **Avoid residual stream magnitude growth** — use gating (sigmoid * value) instead of additive residuals where possible, or normalize residual contributions
6. **If using shared weights/recurrence: QAT is mandatory** — design the training loop with quantization noise from the start
7. **Test quantization at EVERY checkpoint** — don't wait until the end to discover the model is quantization-hostile
8. **Consider ternary (BitNet) for maximum inference efficiency** — accept the 2x capacity tax if deployment constraints demand it

#### 4.6 Edge Deployment Formats and Inference Engines

**Format landscape (March 2026):**

| Format | Engine | Best for | Speed | Quality |
|--------|--------|----------|-------|---------|
| GGUF Q4_K_M | llama.cpp | CPU inference, broad HW | 2.6x vs FP16 | ~+0.05 ppl (7B) |
| GGUF Q5_K_M | llama.cpp | CPU, quality-sensitive | 2.0x vs FP16 | ~+0.04 ppl (7B) |
| GGUF Q8_0 | llama.cpp | CPU, near-lossless | 1.3x vs FP16 | ~lossless |
| NVFP4 | TensorRT | RTX 5090 / Blackwell | 3x vs FP16 | ~1% loss |
| AWQ INT4 | vLLM/TRT | GPU inference | 2–3x vs FP16 | ~2% loss |
| GPTQ INT4 | ExLlamaV2 | GPU inference | 2–3x vs FP16 | ~2% loss |
| BitNet 1.58b | bitnet.cpp | CPU, extreme efficiency | 2.4–6.2x vs llama.cpp | competitive with INT4 |
| ONNX INT4 | ONNX Runtime | Cross-platform | varies | varies |

**Mobile/phone inference (actual numbers):**
- Quantized 7B models: 100–500ms latency on high-end smartphones
- ARM-based (phones/tablets): 4.1x faster than FP16, 67% energy reduction, 45 tok/s on smartphone HW
- Raspberry Pi: viable with Q4_K_M GGUF for small models
- CPU-only GGUF INT4: ~12 tok/s on embedded/IoT hardware

**For Sutra (100M–500M target):**
- At our model size, even FP16 is fast on phones. The quantization benefit is primarily MEMORY, not speed
- A 200M FP16 model = ~400MB. INT4 = ~100MB. Ternary = ~50MB. FP4 = ~100MB
- At this scale, we can likely ship Q5_K_M or Q6_K without meaningful quality loss
- **NVFP4 on RTX 5090 is our primary deployment path** — native hardware support, 3x throughput
- For phones: GGUF Q4_K_M via llama.cpp, or ONNX INT4 for cross-platform
- For extreme edge (IoT/embedded): BitNet ternary if quality is sufficient

**Key GGUF quantization types explained:**
- Q4_K_M: ~4.5 effective bits/weight, mixed precision (Q6_K for attention+FFN tensors, Q4_K for others). Best balance of size/quality for deployment
- Q5_K_M: ~5.5 effective bits/weight, similar mixed-precision strategy. Recommended when quality matters more than size
- Q8_0: 8-bit uniform. Near-lossless but large. Use for validation, not deployment
- K-quant formats use hierarchical super-block structures with mixed precision — more sophisticated than simple rounding

#### 4.7 Synthesis: Sutra's Quantization Strategy

**Design-time decisions (before ANY training):**
1. Replace all LayerNorm/RMSNorm with DyT or Single-Scale RMSNorm — eliminate outlier sources
2. If using shared weights/recurrence: commit to QAT from step 1 (or at least from step 1000+)
3. Design activation distributions to be smooth within blocks of 16–32 elements (serves NVFP4/MXFP4)
4. Test quantization (PTQ to INT4) at every eval checkpoint — early warning system
5. Consider Muon optimizer or rotation-invariant training to prevent privileged-basis formation

**Training-time decisions:**
- Primary format: BF16 training with FP8 Tensor Cores where possible
- QAT fine-tuning: INT8 activations + INT4/ternary weights in final phase
- Or: BitNet-style ternary QAT from scratch (if the 2x capacity tax is acceptable — needs empirical probe)
- Outlier monitoring: track activation kurtosis per layer at each checkpoint

**Deployment-time decisions:**
- RTX 5090 / Blackwell: NVFP4 via TensorRT (3x throughput, 3.5x memory reduction)
- Laptop/desktop CPU: GGUF Q4_K_M via llama.cpp
- Phone/tablet: GGUF Q4_K_M or ONNX INT4
- Extreme edge: BitNet ternary via bitnet.cpp (if we train a ternary variant)

**Open questions for T+L sessions:**
1. At 200M params, is the 2x capacity tax of BitNet ternary worth the inference gains vs. INT4/FP4?
2. Can DyT + OSP-style training make standard INT4 PTQ reliable enough to skip QAT entirely?
3. For elastic compute with early exits: should early-exit heads use higher precision than full-depth heads?
4. How does NVFP4's block-of-16 structure interact with our architecture's internal dimensions? (Dims should be multiples of 16)
5. Is there a hybrid: ternary weights for FFN (largest layers) + INT8 for attention (more precision-sensitive)?

#### 4.8 Muon Optimizer — Compute Efficiency and Outlier Prevention

**Papers:** arXiv:2502.16982 ("Muon is Scalable for LLM Training"), Keller Jordan blog

**What it is:** Muon approximately orthogonalizes the moving average of gradients (momentum) using Newton-Schulz iteration, then uses the result to update parameters. Maintains rotation invariance — no "privileged bases" form.

**Key findings:**
- **2x compute efficiency vs AdamW** in scaling law experiments. Achieves comparable performance at ~52% of AdamW's training FLOPs
- Steps until convergence: ~2/3 of AdamW
- Optimizer memory: somewhat less than AdamW
- Scaling validated: 3B/16B MoE model "Moonlight" trained with Muon on 5.7T tokens
- Two critical techniques for scaling: (1) add weight decay, (2) adjust per-parameter update scale
- Turbo-Muon (2025): spectral preconditioning reduces NS steps from 5→4, ~20% computational overhead reduction

**Quantization connection (from OSP paper):**
- Eliminates "privileged bases" that cause activation outliers
- Adam's per-parameter LR creates channel-wise imbalances → outliers → quantization failure
- Muon + Single-Scale RMSNorm + learnable embedding projection = near-Gaussian activations (kurtosis 0.04 vs 1818.56 for Adam)
- 35% relative improvement in 4-bit PTQ quality from training changes alone

**Relevance to Sutra:**
- 2x training efficiency would double our effective compute budget
- Outlier-free activations serve O5 (quantization-native deployment)
- Weight decay + per-param scaling needed at our scale (159M)
- Compatible with DyT (both attack outliers from different angles)
- **STRONG CANDIDATE**: Muon + DyT could eliminate need for post-hoc quantization tricks entirely

---

## Prior Experiments (Summary — No Checkpoints Retained)

We ran several experiments at 98M scale with a 12-layer dense decoder (d=768, 16K tokenizer, fixed exits at layers 4/8/12). All checkpoints have been deleted. Key observations:

- **Plain AR control outperformed all auxiliary modules** we tested (MTP D=1, learned halting, n-gram memory fusion). However, we discovered multiple implementation bugs during this period, so these failures may reflect our bugs rather than fundamental limitations of the approaches.
- **Post-hoc exit calibration showed promise** for inference efficiency (entropy thresholds on fixed exits).
- **WSD learning rate schedule** worked well for training dynamics.
- **16K custom tokenizer** was consistently the biggest single win.
- **The model was severely undertrained** — at step 5000 (~82M tokens) it was at only 4% of Chinchilla-optimal (~2B tokens for 98M params).

**These observations inform but do not constrain future architecture decisions.** The next T+L session should treat the architecture as an open question.

---

## 5. Multi-Teacher Distillation Survey (Outcome 4: Data Efficiency)

**Purpose:** Comprehensive survey of pretrained models that could serve as teachers for a multi-source learning pipeline. Student will be 100M-500M params, trained from scratch on RTX 5090 (24GB VRAM). Goal: extract maximum knowledge from existing pretrained models to compensate for our limited training data (~22.9B tokens vs competitors' 300B-2T tokens).

**Key constraint:** Student (~200M, FP16) needs ~400MB-1GB VRAM for weights + optimizer states (~3-4GB total during training with Adam). With gradient activations and batch data, student training consumes ~8-12GB. That leaves **12-16GB for teacher model(s)** during online distillation, or zero GPU overhead for offline distillation.

---

### 5.1 Distillation Approaches Overview

#### Online vs. Offline Distillation

| Approach | How it works | VRAM cost | Disk cost | Flexibility |
|----------|-------------|-----------|-----------|-------------|
| **Online** | Teacher loaded on GPU, forward pass each batch | Teacher VRAM + student VRAM | None | Teacher must fit alongside student |
| **Offline** | Pre-compute teacher outputs, store to disk, load during training | Zero GPU overhead | Potentially massive | Any teacher, any number of teachers |
| **MiniPLM-style** | Teacher selects/reweights training data offline, no runtime cost | Zero | Minimal (reweighted corpus) | Cross-family, no architecture constraints |

**Recommended hybrid strategy:**
1. **Offline logit distillation** for large/expensive teachers — pre-compute once, reuse forever
2. **Online feature matching** for small teachers that fit on GPU alongside student
3. **MiniPLM-style data reweighting** as a free baseline that works with any teacher

#### What to Extract from Teachers

| Signal | Description | Storage per token | Quality |
|--------|-------------|-------------------|---------|
| **Full logits** | Complete probability distribution over vocab | vocab_size * 2 bytes (FP16) = 32KB for 16K vocab | Richest signal, captures full teacher uncertainty |
| **Top-K logits** | Only K highest-probability tokens + indices | K * 4 bytes (FP16 value + uint16 index) | ~95% of value for K=128, massive storage savings |
| **Hidden states** | Intermediate layer representations | hidden_dim * 2 bytes per layer | Rich for feature matching, architecture-dependent |
| **Attention maps** | Attention weight matrices | seq_len * seq_len * n_heads * 2 bytes | Only from transformer teachers, expensive |
| **CKA features** | Gram matrix of hidden states | batch_size^2 * 4 bytes per layer | Architecture-agnostic, allows cross-architecture distillation |

---

### 5.2 Category 1: Small Language Models (Sub-1B, Can Coexist on GPU)

These models can run in FP16 alongside the student with VRAM to spare. Best candidates for online distillation.

| Model | Params | Arch | Hidden Dim | Layers | Heads | Training Data | VRAM (FP16) | Strengths |
|-------|--------|------|------------|--------|-------|---------------|-------------|-----------|
| **Pythia-70M** | 70M | Transformer (GPT-NeoX) | 512 | 6 | 8 | 300B tokens (The Pile) | ~140MB | Interpretability baseline, 154 checkpoints available |
| **Pythia-160M** | 160M | Transformer (GPT-NeoX) | 768 | 12 | 12 | 300B tokens (The Pile) | ~320MB | Good quality/size ratio, same data order as all Pythia |
| **Pythia-410M** | 410M | Transformer (GPT-NeoX) | 1024 | 24 | 16 | 300B tokens (The Pile) | ~820MB | Strong for size, deep (24L), RoPE |
| **SmolLM2-135M** | 135M | Transformer (Llama-like) | ~576 | ~12 | GQA | 2T tokens (FineWeb-Edu + DCLM + Stack) | ~270MB | Trained on 15x more data than Pythia, GPT-2 tokenizer |
| **SmolLM2-360M** | 360M | Transformer (Llama-like) | ~960 | ~24 | GQA | 4T tokens | ~720MB | 4T token training, strong benchmarks for size |
| **GPT-2 Small** | 124M | Transformer (original) | 768 | 12 | 12 | ~40B tokens (WebText) | ~250MB | Oldest but well-understood, canonical architecture |
| **GPT-2 Medium** | 355M | Transformer (original) | 1024 | 24 | 16 | ~40B tokens (WebText) | ~710MB | Good hidden dim match for ~200M student |
| **Mamba-130M** | 130M | Pure SSM (Mamba-1) | ~768 | ~24 | N/A (SSM) | 300B tokens (The Pile) | ~260MB | Completely different architecture = diverse representations |
| **Mamba-370M** | 370M | Pure SSM (Mamba-1) | ~1024 | ~48 | N/A (SSM) | 300B tokens (The Pile) | ~740MB | Good SSM teacher, linear-time inference |
| **Mamba-790M** | 790M | Pure SSM (Mamba-1) | ~1536 | ~48 | N/A (SSM) | 300B tokens (The Pile) | ~1.6GB | Strongest pure SSM sub-1B |
| **Mamba2-130M** | 130M | Pure SSM (Mamba-2) | ~768 | ~24 | N/A (SSM) | 300B tokens | ~260MB | Mamba2 = better architecture than Mamba1 |
| **Mamba2-370M** | 370M | Pure SSM (Mamba-2) | ~1024 | ~48 | N/A (SSM) | 300B tokens | ~740MB | Improved SSM, multi-head structure, d_state=128 |
| **Falcon-H1-0.5B** | 500M | Hybrid (Transformer+Mamba) | ~1024 | ~24 | GQA+SSM | Large (undisclosed) | ~1GB | Hybrid = both attention AND SSM representations |
| **Granite-4.0-Micro** | ~350M | Dense Hybrid (GQA+Mamba2) | ~1024 | ~24 | GQA | 15T tokens | ~700MB | IBM, trained on 15T tokens, Apache 2.0 |
| **Granite-4.0-H-Micro** | ~350M | Dense Hybrid (Mamba2 variant) | ~1024 | ~24 | GQA+SSM | 15T tokens | ~700MB | Mamba2 hybrid variant of Micro |
| **Qwen3-0.6B** | 600M | Transformer (GQA) | ~1024 | 28 | 16Q/8KV | 36T+ tokens | ~1.2GB | Best sub-1B transformer, 100+ languages, tied embeddings |

**Analysis — Sub-1B concurrent fit on 24GB:**

With ~12-16GB free alongside student training, we could fit:
- **5-8 models from the 100-400M range simultaneously** (e.g., Pythia-160M + SmolLM2-135M + Mamba-130M + Mamba2-130M + GPT-2 Small = ~1.25GB total)
- **3-4 models from the 400-800M range** (e.g., Pythia-410M + SmolLM2-360M + Mamba-370M + Falcon-H1-0.5B = ~3.3GB)
- **Maximum diversity online combo:** Pythia-410M (transformer) + Mamba-790M (SSM) + Falcon-H1-0.5B (hybrid) + Granite-4.0-Micro (hybrid, 15T data) = ~4.1GB, leaving ~8-12GB for student

**Most complementary sub-1B set (recommended):**
1. **Qwen3-0.6B** — best quality, transformer attention patterns, 36T tokens of training
2. **Mamba-790M** — completely different architecture (SSM), different inductive biases
3. **Granite-4.0-Micro** — hybrid arch, 15T tokens, Apache license
4. **SmolLM2-135M** — tiny but trained on 2T tokens, good cheap reference

Total: ~3.8GB. Fits easily alongside student.

---

### 5.3 Category 2: Medium Language Models (1B-4B, Quantized for Concurrent Use)

These require quantization to coexist with student, or should be used offline.

| Model | Params | Arch | Hidden Dim | Layers | Training Data | VRAM (FP16) | VRAM (Q8) | VRAM (Q4) | Strengths |
|-------|--------|------|------------|--------|---------------|-------------|-----------|-----------|-----------|
| **Pythia-1B** | 1B | Transformer | 2048 | 16 | 300B (Pile) | ~2GB | ~1GB | ~0.6GB | Research-grade, deduped variant available |
| **Pythia-1.4B** | 1.4B | Transformer | 2048 | 24 | 300B (Pile) | ~2.8GB | ~1.5GB | ~0.8GB | Deep, good representations |
| **SmolLM2-1.7B** | 1.7B | Transformer (Llama) | ~2048 | ~24 | 11T tokens | ~3.4GB | ~1.7GB | ~1GB | 11T tokens = massive data advantage |
| **Qwen3-1.7B** | 1.7B | Transformer (GQA) | 2048 | 28 | 36T+ tokens | ~3.4GB | ~2GB | ~1GB | Strong reasoning, multilingual, GQA |
| **Gemma-3-1B** | 1B | Transformer (interleaved attn) | ~2048 | 26 | ~6T tokens | ~2GB | ~1GB | ~0.6GB | Interleaved global/local attention (1:5 ratio), 32K context |
| **Granite-4.0-Tiny** | ~1B | Hybrid (Mamba2+Transformer) | ~2048 | ~24 | 15T tokens | ~2GB | ~1GB | ~0.6GB | Hybrid, 15T tokens, Apache 2.0 |
| **Granite-4.0-H-1B** | ~1.5B | Hybrid (Mamba2 heavy) | ~2048 | ~24 | 15T tokens | ~3GB | ~1.5GB | ~0.9GB | 70-80% less memory than transformer equivalent |
| **Falcon-H1-1.5B** | 1.5B | Hybrid (Transformer+Mamba) | ~2048 | ~24 | Large | ~3GB | ~1.5GB | ~0.9GB | "Rivals many 7B-10B models" per TII |
| **LFM2.5-1.2B** | 1.2B | Novel Hybrid (Conv+GQA) | ~2048 | 16 (10 conv + 6 GQA) | 10T tokens | ~2.4GB | ~1.2GB | ~0.7GB | 2x faster CPU inference than Qwen3, 10T data |
| **Mamba-1.4B** | 1.4B | Pure SSM | ~2048 | ~48 | 300B (Pile) | ~2.8GB | ~1.5GB | ~0.8GB | Largest pure Mamba-1 at reasonable size |
| **Mamba2-1.3B** | 1.3B | Pure SSM (Mamba-2) | ~2048 | ~48 | 300B | ~2.6GB | ~1.3GB | ~0.8GB | Mamba-2 improved architecture |

**Quantization for concurrent use:**
- At Q4, most 1-1.7B models fit in ~0.6-1GB VRAM
- We could run **3-4 quantized medium teachers + student** simultaneously
- **But:** Q4 quantization degrades teacher signal quality. Better to run fewer teachers in higher precision or use offline approach.

**Best medium teacher set (Q8, concurrent):**
1. **Qwen3-1.7B** (Q8, ~2GB) — strongest reasoning, 36T tokens
2. **Falcon-H1-1.5B** (Q8, ~1.5GB) — hybrid architecture diversity
3. **LFM2.5-1.2B** (Q8, ~1.2GB) — novel conv+attention hybrid, 10T data

Total: ~4.7GB at Q8. Fits alongside student.

**Offline-only candidates (too large for concurrent but worth pre-computing):**

| Model | Params | VRAM (FP16) | Why offline | What to extract |
|-------|--------|-------------|------------|-----------------|
| **Phi-4-mini** | 3.8B | ~7.6GB | Too large for concurrent | Top-K logits, strong reasoning signal |
| **Qwen3-4B** | 4B | ~8GB | Too large for concurrent | Top-K logits, multilingual knowledge |
| **Gemma-3-4B** | 4B | ~8GB | 128K context model | Top-K logits, interleaved attention patterns |
| **Pythia-2.8B** | 2.8B | ~5.6GB | Marginal fit, better offline | Hidden states at all 32 layers for scaling analysis |
| **Mamba-2.8B** | 2.8B | ~5.6GB | Marginal fit, better offline | SSM state representations |

---

### 5.4 Category 3: Encoder Models (Bidirectional, Rich Representations)

Encoder models provide fundamentally different representations than decoder LMs — bidirectional context means they "see" the whole input, making their hidden states richer for semantic understanding. They are typically small and cheap.

| Model | Params | Arch | Hidden Dim | Layers | Embedding Dim | VRAM (FP16) | Strengths |
|-------|--------|------|------------|--------|---------------|-------------|-----------|
| **BERT-base** | 110M | Encoder (Transformer) | 768 | 12 | 768 | ~220MB | Canonical, bidirectional, well-studied |
| **BERT-large** | 340M | Encoder (Transformer) | 1024 | 24 | 1024 | ~680MB | Deeper representations |
| **RoBERTa-base** | 125M | Encoder (Transformer) | 768 | 12 | 768 | ~250MB | Better training recipe than BERT (160GB data, no NSP) |
| **RoBERTa-large** | 355M | Encoder (Transformer) | 1024 | 24 | 1024 | ~710MB | Strong baseline for NLU tasks |
| **DistilBERT** | 66M | Encoder (Transformer) | 768 | 6 | 768 | ~130MB | Distilled from BERT, 97% of performance at 60% size |
| **all-MiniLM-L6-v2** | 22M | Encoder (Transformer) | 384 | 6 | 384 | ~44MB | Tiny but effective sentence embeddings, distilled from MiniLM |
| **DeBERTa-v3-base** | 184M | Encoder (disentangled attn) | 768 | 12 | 768 | ~370MB | Disentangled attention, often beats BERT/RoBERTa |

**How to use encoder teachers for decoder student distillation:**
- **NOT for logit distillation** (different output structure — MLM vs. autoregressive)
- **FOR hidden state / representation matching:**
  - Extract encoder hidden states for each text chunk
  - Use CKA or projection-based alignment to match student's internal representations
  - Forces student to learn bidirectional semantic understanding despite being autoregressive
- **FOR data annotation:**
  - Use encoder to score/classify training data quality
  - Pre-compute NLI, sentiment, topic labels as auxiliary supervision signals

**Key insight:** Encoder representations are complementary to decoder LM logits. A decoder teacher tells the student "what comes next"; an encoder teacher tells the student "what this text means." Both are valuable.

---

### 5.5 Category 4: Embedding Models (Semantic Representations)

Embedding models produce dense vector representations optimized for semantic similarity. They compress text understanding into fixed-size vectors — useful as representation targets.

| Model | Params | Arch | Output Dim | VRAM (FP16) | MTEB Score | Strengths |
|-------|--------|------|------------|-------------|------------|-----------|
| **EmbeddingGemma-308M** | 308M | Encoder (bidirectional Gemma3) | 768 (MRL: 128-768) | ~400MB | SOTA <500M | Google, bi-directional from decoder, 100+ langs |
| **bge-small-en-v1.5** | 33M | Encoder (BERT) | 384 | ~70MB | ~62 | Tiny, fast, good baseline |
| **bge-base-en-v1.5** | 109M | Encoder (BERT) | 768 | ~220MB | ~64 | Balanced quality/speed |
| **bge-large-en-v1.5** | 335M | Encoder (BERT) | 1024 | ~700MB | ~65 | Best BGE English |
| **e5-small-v2** | 33M | Encoder (BERT) | 384 | ~70MB | ~62 | Fastest, 16ms latency |
| **e5-base-v2** | 109M | Encoder (BERT) | 768 | ~220MB | ~63 | Balanced |
| **e5-large-v2** | 335M | Encoder (BERT) | 1024 | ~700MB | ~64 | Best E5 English |
| **nomic-embed-text-v1.5** | 137M | Encoder (long-context BERT) | 768 (MRL: 64-768) | ~300MB | ~65 | 8192-token context, Matryoshka dims |
| **bge-m3** | 568M | Encoder (BERT) | 1024 | ~700MB | ~67 | Trilingual, dense+sparse hybrid retrieval |
| **Qwen3-Embedding-0.6B** | 600M | Decoder-as-encoder (Qwen3) | 32-1024 (MRL) | ~700MB | 65.5 | Best <1B on MTEB, 100+ langs, instruction-aware |
| **GTE-Qwen2-1.5B** | 1.5B | Decoder-as-encoder | 4096 | ~1.5GB (Q8) | 67.2 | Long context, high dim |
| **nomic-embed-code** | ~137M | Encoder | ~768 | ~300MB | N/A | SOTA code retrieval |

**How to use embedding model teachers:**
- **Contrastive distillation:** Train student's hidden states to produce embeddings that match the teacher's similarity structure (not the exact vectors, but which pairs are close/far)
- **Semantic anchoring:** Use embedding similarities as soft labels for training data clustering
- **Representation regularization:** Add a loss term that encourages student hidden states at certain layers to align with pre-computed embeddings

**Most useful embedding teachers:**
1. **EmbeddingGemma-308M** — SOTA at sub-500M, bidirectional Gemma3 backbone, MRL
2. **Qwen3-Embedding-0.6B** — decoder-based (same paradigm as student), instruction-aware
3. **nomic-embed-text-v1.5** — long context (8K), Matryoshka, lightweight

---

### 5.6 Category 5: Code Models

Code understanding requires different representations than natural language — syntax trees, control flow, variable scoping. Code teachers provide these.

| Model | Params | Arch | Hidden Dim | VRAM (FP16) | Training | Strengths |
|-------|--------|------|------------|-------------|----------|-----------|
| **CodeBERT** | 125M | Encoder (RoBERTa) | 768 | ~250MB | CodeSearchNet (6 PLs) | Bidirectional code+NL, code search, clone detection |
| **StarCoder2-3B** | 3B | Decoder (GQA) | ~2560 | ~6GB | 3.3T code tokens (Stack v2) | 80+ PLs, FIM objective, 16K context, GQA |
| **Codestral-Mamba-7B** | 7B | Pure SSM (Mamba) | ~4096 | ~5GB (Q4) | Code-focused (Mistral) | SSM architecture for code, fast inference |
| **nomic-embed-code** | ~137M | Encoder | ~768 | ~300MB | Code-focused | SOTA code retrieval embeddings |

**For code understanding in Sutra's multi-source pipeline:**
- **CodeBERT (online, 250MB)** — cheap, bidirectional, good for code semantics
- **StarCoder2-3B (offline only)** — too large for concurrent, but excellent logits for code tokens in training data
- **nomic-embed-code (online, 300MB)** — code embedding alignment

---

### 5.7 Offline Distillation: Storage Requirements Analysis

For 22.9B training tokens with a 16K vocabulary:

#### Full Logit Storage (NOT recommended)

```
22.9B tokens * 16,384 vocab * 2 bytes (FP16) = ~751 TB
```
Completely impractical.

#### Top-K Logit Storage (RECOMMENDED)

Per token, store K (value, index) pairs:
- K=32: 32 * (2 bytes value + 2 bytes index) = 128 bytes/token
- K=64: 64 * 4 = 256 bytes/token
- K=128: 128 * 4 = 512 bytes/token

| K | Bytes/token | Storage for 22.9B tokens | Quality retention |
|---|-------------|--------------------------|-------------------|
| 32 | 128 | **2.93 TB** | ~90% of full logit signal |
| 64 | 256 | **5.86 TB** | ~95% of full logit signal |
| 128 | 512 | **11.72 TB** | ~98% of full logit signal |

**Practical recommendation:** K=64, which requires ~5.9TB per teacher. For 2-3 offline teachers, that is 12-18TB. This is feasible with external storage but expensive.

#### MiniPLM-Style (CHEAPEST — Store Nothing)

MiniPLM does not store logits at all. Instead:
1. Run teacher once over entire corpus
2. Compute per-document "difficulty scores" (divergence between teacher and a small reference model)
3. Reweight/resample the training corpus to emphasize documents where teacher knowledge is most valuable
4. Student trains on the reweighted corpus with standard NTP loss

**Storage cost:** Only the reweighted sample indices — negligible (a few MB).
**Quality:** 2.2x pre-training acceleration, 2.4x data efficiency (MiniPLM paper results on Qwen 200M-1.2B students with Qwen-1.8B teacher).
**Limitation:** Only captures data selection signal, not the rich probability distributions.

#### Hidden State Storage

Per token per layer: hidden_dim * 2 bytes (FP16).
For a 768-dim model with 12 layers, storing ALL layer hidden states:
```
22.9B tokens * 12 layers * 768 dims * 2 bytes = ~421 TB
```
Impractical. But storing only the FINAL hidden state of a single teacher:
```
22.9B tokens * 768 dims * 2 bytes = ~35.1 TB
```
Still too large. **Hidden state distillation must be online, not offline.**

#### Recommended Offline Strategy

1. **MiniPLM data reweighting** from Qwen3-1.7B or Phi-4-mini — free, improves data efficiency
2. **Top-64 logits** from 1-2 high-quality teachers (Qwen3-1.7B + Mamba-2.8B) stored to disk — ~12TB total
3. **Online feature matching** with 2-3 small teachers loaded on GPU — zero disk cost

---

### 5.8 Distillation Frameworks and Methods

#### Existing Frameworks

| Framework | Paper | Method | Cross-Arch? | Scale Tested | Key Idea |
|-----------|-------|--------|-------------|--------------|----------|
| **MiniPLM** | ICLR 2025 | Offline data reweighting via difference sampling | Yes | 200M-1.2B student, 1.8B teacher (Qwen) | Adjust corpus difficulty using teacher-reference LM divergence |
| **MOHAWK** | NeurIPS 2024 | 3-phase progressive distillation | Yes (Transformer→SSM) | Phi-1.5→Mamba-2 | Phase 1: match mixing matrices. Phase 2: match hidden states per block. Phase 3: match end-to-end predictions |
| **Pre-training Distillation DSE** | ACL 2025 | Design space exploration | Same arch | 200M-1.8B | Top-p-k logit truncation, temperature tuning, offline vs online comparison |
| **DistilBERT** | 2019 | Triple loss (CE + distill + cosine) | No (BERT→smaller BERT) | 66M student, 110M teacher | Standard KD with hidden state cosine alignment |
| **TinyBERT** | 2020 | Multi-layer feature distillation | No (BERT→smaller BERT) | Various | Attention + hidden state + embedding + prediction matching |

#### Key Techniques for Multi-Teacher

**Logit-level aggregation (simplest):**
```
L_distill = sum_t [ w_t * KL(student_logits || teacher_t_logits) ]
```
Where w_t are per-teacher weights (uniform, learned, or based on teacher confidence).

**Feature-level matching via CKA (architecture-agnostic):**
- Compute Gram matrix G_s = H_s @ H_s^T for student hidden states H_s
- Compute Gram matrix G_t = H_t @ H_t^T for teacher hidden states H_t
- CKA(G_s, G_t) measures representational similarity without requiring same dimensionality
- Loss = 1 - CKA(G_s, G_t) summed over selected layer pairs
- **Works across architectures** (transformer student, SSM teacher, encoder teacher, etc.)

**Progressive Knowledge Distillation (PKD):**
- Architecture-agnostic: modular approach that doesn't require matching layer counts
- Distill in stages: first match early representations, then intermediate, then final
- Allows different teachers for different stages

**Knowledge purification (for >3 teachers):**
- Research shows performance DECLINES as teacher count increases beyond a threshold
- Solution: consolidate teacher knowledge into a single "purified" signal before distilling
- Practical: average top-K logits from multiple teachers, or use attention-weighted combination

#### Cross-Architecture Distillation (Critical for Sutra)

Since Sutra's architecture is not yet finalized, the distillation method must be **architecture-agnostic**:

1. **CKA-based feature matching** — works regardless of student architecture (transformer, SSM, hybrid, novel)
2. **Logit-level KD** — works as long as student has a language modeling head over the same vocabulary
3. **MOHAWK-style progressive** — if student has identifiable "blocks," can match block-level representations
4. **MiniPLM data reweighting** — completely architecture-independent

**Warning from literature:** Flex-KD (2025) found that matching only final-layer representations often outperforms multi-layer matching. Start simple.

---

### 5.9 Maximum Concurrent Teachers on 24GB GPU

**Scenario: ~200M student training in FP16+Adam**

| Component | VRAM |
|-----------|------|
| Student weights (200M, FP16) | ~400MB |
| Adam optimizer states (2x weights, FP32) | ~1.6GB |
| Gradients (FP16) | ~400MB |
| Activations (batch=32, seq=512) | ~2-4GB |
| **Total student training** | **~5-7GB** |
| **Remaining for teachers** | **~17-19GB** |

**Feasible concurrent teacher configurations:**

| Config | Teachers | Total Teacher VRAM | Remaining Buffer | Diversity Score |
|--------|----------|--------------------|------------------|-----------------|
| **A (Maximum diversity)** | Qwen3-0.6B + Mamba-790M + Granite-Micro + EmbeddingGemma-308M + CodeBERT | ~3.8GB | ~14GB | HIGH — 5 architectures |
| **B (Strongest signal)** | Qwen3-1.7B (Q8) + Falcon-H1-1.5B (Q8) + Mamba-1.4B (Q8) | ~5GB | ~13GB | MEDIUM — 3 architectures, larger teachers |
| **C (Quality + diversity)** | Qwen3-0.6B + Mamba-790M + Granite-Micro + Qwen3-Embedding-0.6B | ~3.6GB | ~15GB | HIGH — decoder + SSM + hybrid + embedding |
| **D (Aggressive, all medium)** | Qwen3-1.7B (Q8) + SmolLM2-1.7B (Q8) + LFM2.5-1.2B (Q8) + Mamba-1.4B (Q8) | ~6.4GB | ~12GB | HIGH — 4 architectures, all well-trained |

**Recommended: Config C or D**, depending on whether online hidden-state matching (C, needs forward pass through each teacher per batch) or offline logit pre-computation + online embedding alignment (D, heavier teachers but less per-batch cost) is preferred.

---

### 5.10 Recommended Multi-Teacher Pipeline

#### Phase 0: Data Reweighting (Free, Do This First)
- Run **Qwen3-1.7B** (or Phi-4-mini) over entire 22.9B-token corpus
- Compute per-document divergence scores (MiniPLM method)
- Reweight training data to emphasize high-information documents
- **Cost:** One-time GPU inference pass (~hours). **Benefit:** 2x+ data efficiency.

#### Phase 1: Offline Logit Pre-computation (Before Training)
- **Teacher 1: Qwen3-1.7B** — general knowledge, reasoning, multilingual
- **Teacher 2: Mamba-2.8B** — SSM perspective, different inductive bias
- Store top-64 logits per token per teacher
- Storage: ~12TB for both teachers across 22.9B tokens
- **Alternative if storage is limited:** Pre-compute only for the first epoch (subset), refresh for subsequent epochs

#### Phase 2: Online Multi-Teacher Training
Load on GPU alongside student:
- **Qwen3-0.6B** (1.2GB) — live logit + hidden state teacher
- **Mamba-790M** (1.6GB) — live SSM representation teacher
- **EmbeddingGemma-308M** (400MB) — semantic embedding anchor
- **CodeBERT** (250MB) — code token representation teacher

**Training loss:**
```
L = L_NTP + alpha * L_offline_KD + beta * L_online_KD + gamma * L_CKA + delta * L_embed
```
Where:
- L_NTP = standard next-token prediction on reweighted data
- L_offline_KD = KL divergence against stored Qwen3-1.7B/Mamba-2.8B logits
- L_online_KD = KL divergence against live Qwen3-0.6B/Mamba-790M logits
- L_CKA = CKA representation matching against teacher hidden states
- L_embed = contrastive alignment with EmbeddingGemma representations

#### Phase 3: Validation
Compare against:
- Student trained with standard NTP only (same data, same compute)
- Student trained with single-teacher KD
- Published baselines (SmolLM2, Pythia, etc.)

---

### 5.11 Key Findings and Open Questions

**Key findings:**
1. **MiniPLM data reweighting is free and proven** — 2.2x acceleration, 2.4x data efficiency. No reason not to use it.
2. **Offline top-K logit distillation is the standard approach** — K=64 balances quality vs. storage. ~6TB per teacher for our corpus.
3. **CKA enables cross-architecture distillation** — critical since Sutra's architecture is still TBD.
4. **More teachers is NOT always better** — performance can decline beyond 3-4 teachers due to conflicting gradients. Knowledge purification or dynamic weighting needed.
5. **Encoder teachers (BERT, RoBERTa) provide complementary signal** — bidirectional understanding that decoder-only teachers cannot provide.
6. **Embedding teachers provide semantic structure** — useful as regularization, not primary distillation signal.
7. **Architecture diversity > teacher size** — a transformer + SSM + hybrid ensemble teaches more than 3 transformers of the same family.
8. **Sub-1B teachers have been trained on 300B-36T tokens** — even "small" teachers have seen 13-1500x more data than our student will, making them valuable knowledge sources.
9. **Flex-KD finding (2025): final-layer-only matching often beats multi-layer** — start simple, add complexity only if justified.
10. **Our custom 16K tokenizer creates a vocabulary mismatch** — logit-level distillation requires either (a) teacher fine-tuned on our tokenizer, (b) token-level alignment mapping, or (c) representation-level distillation only. This is a critical implementation detail.

**Open questions for T+L:**
1. **Vocabulary mismatch problem:** Our 16K custom tokenizer vs. teachers' tokenizers (GPT-2 50K, Qwen 152K, etc.). How to align logits across different vocabularies? Options: (a) use a shared tokenizer, (b) learn a vocabulary projection, (c) skip logit KD and use only representation matching, (d) retokenize teacher outputs with our tokenizer.
2. **Optimal teacher weighting:** Static (hand-tuned alpha per teacher) vs. dynamic (learned per-batch or per-token teacher attention)?
3. **When to start KD during training:** From step 0, or delay until student has basic competence (step 1K+)?
4. **Interaction with MTP/TOP:** If using multi-token prediction as auxiliary loss, does KD provide redundant signal or complementary signal?
5. **Memory-optimal online strategy:** Forward pass through 4 teachers per batch is expensive — should we cycle teachers (different teacher each batch) instead of running all simultaneously?
6. **Is offline logit storage (12TB) worth it** vs. just using more/larger online teachers?

### 5.12 Meta-Learning KD: Disagreement-Driven Self-Improving Distillation (2026-03-26)

**Status: THEORETICAL FRAMEWORK — needs Codex evaluation and probe design**

**Core thesis:** Multi-teacher KD should have ACCELERATING returns (inverted power law), not diminishing returns. Each training round produces not just a better model, but a better understanding of WHAT to learn next and FROM WHOM.

**The Disagreement Taxonomy:**

When multiple teachers produce outputs on the same input, their agreement/disagreement patterns form a structured space:

1. **Consensus knowledge** — All teachers agree → high-confidence, easy to absorb, train on this first
2. **Architecture-dependent disagreement** — Transformers agree with each other, SSMs agree with each other, but the families disagree → structural bias, HIGHEST VALUE for cross-architecture student
3. **Capacity-dependent disagreement** — Large teachers agree, small teachers disagree → knowledge gated by capacity
4. **Family-dependent disagreement** — Same-company models agree, cross-company disagree → training data/recipe bias
5. **Universal uncertainty** — All teachers disagree → genuinely ambiguous, de-prioritize

**The Inverted Power Law Mechanism:**

Standard training: each additional token provides less marginal information (diminishing returns).
Meta-learning KD: each training round produces:
- Better model (standard)
- Better diagnostic of what the model can't learn (disagreement analysis)
- Better curriculum (which data to show next)
- Better teacher weighting (which teacher to trust for which concept)

If the diagnostic improves faster than the model, learning ACCELERATES.

**Implementation sketch:**
1. Every N steps, run disagreement analysis on a held-out sample (not full corpus)
2. Classify student failures into the 5 categories above
3. Update teacher attention weights per-category
4. Update data sampling weights (upweight high-disagreement regions)
5. Log the category distribution as an internal metric (predicts benchmark performance better than BPT)

**Connection to Outcome 2 (Improvability):** The disagreement taxonomy IS the diagnostic system. When the model fails a benchmark, check which disagreement category the failure falls into → surgical fix.

**Falsification criteria:**
- If disagreement patterns are random (no structure), the taxonomy is useless
- If weighted KD doesn't beat uniform KD, the routing is useless
- If learning efficiency doesn't accelerate over training, the meta-learning claim is false

**Open questions for Codex:** Full list in `research/SCRATCHPAD.md` under "Meta-Learning KD" section.

---

## 6. March 2026 Research Survey: Architecture Design Inputs

**Date:** 2026-03-26. Comprehensive web research across 5 areas to feed the next Tesla+Leibniz architecture design session.

---

### 6.1 Small Model Architectures (100M-4B) — State of the Art

#### 6.1.1 MobileLLM (Meta, ICML 2024) — arXiv:2402.14905

**The reference for sub-billion architecture design.**

Key findings for optimal sub-billion architecture:
- **Deep and thin beats wide and shallow.** Optimal depth ~30 layers for sub-billion scale.
- **Embedding sharing** reduces params ~10% with only 0.2-0.6% accuracy drop.
- **Grouped Query Attention** with head_dim=64, num_heads ~4x kv_heads.
- **Block-wise weight sharing** provides free depth increase with no param increase and marginal latency overhead.
- Results: +2.7%/+4.3% accuracy over prior SOTA at 125M/350M. Published ICML 2024.

**Sutra relevance:** Validates deep+thin + weight sharing as the right structural prior for small models. Our recurrent depth approach aligns with this.

#### 6.1.2 Optimal Architecture Study (HuggingFace, Dec 2025)

19 model configs across 12 architecture families trained on 1B tokens at ~70M params.

**Critical discovery — two-tier performance pattern:**
- Models cluster into HIGH tier (~38.5% avg) and LOW tier (~32% avg) with a 6pt gap and almost nothing between.
- **Hidden dimension threshold:** models need hidden_size >= 512 OR specific compensating depths (32 or 64+ layers).
- **32 layers is the "Goldilocks depth"** — best single config: 32L/384d/77M params = 38.50% avg.
- **Architecture family barely matters at 70M scale** — all 12 architectures within ~2% of each other.
- Best config: 32 layers, hidden=384, 8 attn heads, 4 KV heads, FF=1024, RoPE, RMSNorm.
- Diffusion models trade -1.33% accuracy for +3.8x throughput.
- WSD (Warmup-Stable-Decay) scheduler gives 10x training efficiency (100M tokens vs 1B).

**Sutra relevance:** At our target scale (70M-4B), architecture choice matters less than depth and hidden dim threshold. Weight sharing + recurrence is the way to get depth without proportional param cost.

#### 6.1.3 SmolLM2 (HuggingFace, Feb 2025) — arXiv:2502.02737

**The current SOTA small model family.**

- 135M/360M/1.7B parameter variants.
- Trained on 2T/4T/11T tokens respectively — massive data investment.
- Multi-stage training with manual refinement of dataset mixing rates between stages.
- New datasets: FineMath, Stack-Edu, SmolTalk for math/code/instruction.
- 1.7B outperforms Qwen2.5-1.5B and Llama3.2-1B.
- Benchmarks: HellaSwag 68.7%, ARC-Avg 60.5%, PIQA 77.6% (1.7B model).

**Sutra relevance:** Demonstrates data-centric approach. Our data budget (~23B tokens) is 100-500x smaller. Must compensate with architecture + multi-teacher KD.

#### 6.1.4 Qwen3 Family (Alibaba, May 2025) — arXiv:2505.09388

**Dense models: 0.6B, 1.7B, 4B, 8B, 14B, 32B.**

- Qwen3-1.7B/4B/8B/14B/32B-Base matches Qwen2.5-3B/7B/14B/32B/72B-Base respectively — each matches the next-size-up prior gen.
- 4-stage training: long CoT cold start -> reasoning RL -> thinking mode fusion -> general RL.
- Fine-tuned Qwen3-4B matches or exceeds GPT-OSS-120B (30x larger teacher) on 7/8 benchmarks.
- Architecture uses Gated DeltaNet in Qwen3-Next variant (linear attention replacement).

**Sutra relevance:** Primary competition at 4B. Their 4-stage training pipeline is sophisticated. Their Gated DeltaNet exploration signals the field is moving beyond pure attention.

#### 6.1.5 Phi-4-mini (Microsoft, Mar 2025) — arXiv:2503.01743

**Dense decoder-only, 3.8B params, 128K context.**

- 200K vocabulary, GQA, shared embedding.
- Trained on 5T tokens — heavy synthetic "textbook-quality" data.
- HellaSwag 83.5%, ARC-C 83.7%.
- Strategic synthetic data investment rather than raw web crawl.

**Sutra relevance:** Shows synthetic data from teachers can compensate for raw data volume. Aligns with our multi-teacher KD strategy.

#### 6.1.6 Gemma 3 (Google, Mar 2025) — arXiv:2503.19786

**1B/4B/12B/27B variants. Multimodal.**

- 5:1 ratio of local to global attention layers (local context capped at 1024 tokens).
- Trained WITH distillation from larger models.
- Gemma3-4B-IT competitive with Gemma2-27B-IT.
- 1B variant: 2,585 tokens/sec during prefill.
- QAT models released for consumer GPU deployment.

**Sutra relevance:** Their local/global attention ratio and distillation-from-larger-models approach are directly relevant.

#### 6.1.7 BitNet b1.58 2B4T (Microsoft, Apr 2025) — arXiv:2504.12285

**First natively-trained 1-bit LLM at 2B scale, 4T tokens.**

- Weights quantized to {-1, 0, +1} (1.58 bits), activations INT8.
- Quantization-Aware Training from scratch — not post-hoc quantization.
- MMLU 52.1%, ARC-C 68.5%, HellaSwag 84.3%, GSM8K 58.38%, WinoGrande 71.90%.
- Outperforms Qwen2.5-1.5B INT4 on most benchmarks.
- ARM CPU speedup 1.37-5.07x, energy reduction 55-70%.
- x86 CPU speedup 2.37-6.17x, energy reduction 72-82%.
- Can run 100B model on single CPU at human reading speed (5-7 tok/s).
- BitNet a4.8 variant: 4-bit activations, only 55% of params active, 3-bit KV cache.

**Sutra relevance:** CRITICAL for Outcome 5 (inference efficiency). Proves you can train natively at extreme quantization and maintain quality. Our architecture should be designed for 1.58-bit or INT4 from day one.

---

### 6.2 Hybrid Architectures (SSM + Attention)

#### 6.2.1 Hymba (NVIDIA, Nov 2024, ICLR 2025) — arXiv:2411.13676

**Hybrid-head architecture specifically designed for small language models.**

- Integrates attention heads for high-resolution recall + SSM heads for efficient context summarization WITHIN the same layer.
- Learnable "meta tokens" prepended to sequences — act as learned cache initialization, similar to metamemory.
- 1.5B model outperforms all open-source models of similar size.
- >50% of attention computation can be replaced by cheaper SSM computation without sacrificing performance.
- KV cache highly correlated across heads/layers — can be shared (GQA + cross-layer KV sharing).
- 10x less cache memory than pure transformers on A100.

**Sutra relevance:** HIGHLY relevant. Shows SSM+attention hybrid at the HEAD level (not just layer interleaving) is superior. Meta tokens concept aligns with scratchpad/memory ideas.

#### 6.2.2 Mamba-3 (ICLR 2026) — arXiv:2603.15569

**Latest SSM architecture. Three core improvements:**

1. **Trapezoidal discretization** replaces Euler's method — better continuous signal approximation.
2. **Complex-valued state updates** — mathematically equivalent to data-dependent RoPE, captures the expressive power of complex dynamics while retaining real-valued recurrence speed.
3. **MIMO formulation** — multi-input multi-output enables richer state tracking + better hardware parallelism during decoding.

Results at 1.5B:
- Mamba-3 (SISO): +0.6 pts over Gated DeltaNet (previously best non-transformer).
- Mamba-3 (MIMO): +1.2 pts additional = total +1.8 pts over GDN, +2.2 pts over Transformers.
- 57.6% average downstream accuracy.
- Achieves comparable perplexity to Mamba-2 with HALF the state size.

**Sutra relevance:** The trapezoidal discretization and complex dynamics are derivable from first principles (SSM theory). MIMO is a genuine capability advance. Should be evaluated for our state-update stage.

#### 6.2.3 RWKV-7 "Goose" (Mar 2025) — arXiv:2503.14456

**Linear-time recurrent model with expressive dynamic state evolution.**

- Generalized delta rule with vector-valued gating and in-context learning rates.
- Four model sizes: 0.19B, 0.4B, 1.5B, 2.9B.
- 2.9B achieves 3B SoTA on multilingual, matches English SoTA — trained on dramatically fewer tokens than competitors.
- 1.5B: MMLU 43.3% (up from RWKV-6's 25.1%).
- 2.9B: 71.5% avg English accuracy with 5.6T tokens — matches Qwen2.5-3B (71.4%) trained on 18T tokens (3.2x data efficiency).
- Can perform state tracking and recognize all regular languages (exceeds Transformers under standard complexity conjectures).
- Constant memory usage, constant inference time per token.

**Sutra relevance:** CRITICAL. Demonstrates massive data efficiency through better architecture. 3.2x data efficiency over Qwen at same scale. The dynamic state evolution with in-context learning rates is a mechanism worth deriving from first principles.

#### 6.2.4 Gated DeltaNet (ICLR 2025/2026)

**Linear attention variant combining Mamba2 gating + delta rule.**

- Alpha (decay gate) controls memory decay/reset.
- Beta (update gate) controls how strongly new inputs modify state.
- Adopted by Qwen3-Next as its linear attention layer.
- Consistently surpasses Mamba2 and DeltaNet on language modeling, common-sense reasoning, in-context retrieval, length extrapolation.
- Hybrid GDN + sliding window attention achieves improved training efficiency + superior task performance.

**Sutra relevance:** The field is converging on gated linear attention as the non-attention workhorse. GDN is the current best. Mamba-3 surpasses it.

#### 6.2.5 Hybrid Architecture Patterns (Field Consensus 2025-2026)

**Empirical finding across Jamba, Granite 4, Zamba, Bamba, Hymba:**
- Only 7-8% of layers need to be full attention (1 in 8 or 1 in 9 ratio).
- This small fraction closes the gap to pure transformers and often EXCEEDS pure transformer performance.
- IBM Granite 4: 9 Mamba blocks per 1 Transformer block.
- Zamba hypothesis: "one attention layer is all you need."
- Jamba: 1 transformer layer out of every 8 total.

**Sutra relevance:** If we use a hybrid, the attention fraction should be ~10%, not 50%. Most computation should be linear-time (SSM/convolution/recurrence).

---

### 6.3 Adaptive Compute / Elastic Depth

#### 6.3.1 Mixture of Recursions (MoR) — NeurIPS 2025, arXiv:2507.10524

**First framework unifying parameter sharing + token-level adaptive depth + memory-efficient KV caching.**

- Lightweight routers assign token-specific recursion depths end-to-end.
- Tested at 135M, 360M, 730M, 1.7B with 3 recursions (1/3 unique params).
- At equal training compute, MoR with 2 recursions outperforms vanilla transformers: perplexity 2.75 vs 2.78, accuracy 43.1% vs 42.3%, with 50% fewer params.
- Underperforms vanilla only at 135M (capacity bottleneck) — gap closes at scale.
- Up to 2.18x inference throughput via continuous depth-wise batching + early exit.
- KV cache reduction ~50%.
- Matched baseline accuracy with 25% fewer FLOPs, 19% faster training, 25% less peak memory.

**Sutra relevance:** DIRECTLY relevant to our elastic compute vision. MoR proves token-level adaptive depth works. The capacity bottleneck at 135M is concerning for our small-scale experiments — may need 360M+ to see benefits.

#### 6.3.2 LoopFormer (Feb 2026) — arXiv:2602.11451

**Elastic-depth looped transformer with shortcut-consistency training.**

- Trained on variable-length trajectories for budget-conditioned reasoning.
- Shortcut-consistency scheme aligns trajectories of different lengths: shorter loops remain informative, longer loops refine.
- Narrows perplexity gap to non-looped baseline at higher budgets.
- Outperforms other looped variants, especially at higher compute budgets.
- Representation dynamics show sustained evolution through mid-depths (vs. early-exit baselines which remain flat).
- Authors: University of Toronto / Vector Institute.

**Sutra relevance:** The shortcut-consistency training is a key technique — ensures the model produces useful output at ANY depth, not just maximum depth. This is exactly what Sutra needs for elastic compute.

#### 6.3.3 AdaPonderLM (Mar 2026) — arXiv:2603.01914

**Self-supervised recurrent LM with learned token-wise early exiting.**

- Iteration-specific MLP gates with monotonic halting mask.
- KV reuse mechanism for halted tokens — train-test consistency.
- Tested on Pythia 70M-410M (pretraining) and up to 2.8B (continued pretraining).
- Reduces inference compute ~10% while maintaining comparable perplexity and accuracy.
- Learned gates allocate more computation to high-NLL (hard) tokens.
- Under iso-FLOPs, learned halting consistently outperforms fixed pruning.

**Sutra relevance:** Validates self-supervised halting (no manual labels needed). The monotonic halting mask is a clean mechanism. 10% compute reduction is modest — MoR achieves more — but AdaPonderLM's gate design is simpler.

#### 6.3.4 TIDE (Mar 2026) — arXiv:2603.21365

**Post-training per-token early exit. Works on ANY pretrained model.**

- Tiny learned routers at periodic checkpoint layers.
- No model retraining required.
- Calibration on 2,000 WikiText samples takes <3 minutes, produces ~4MB router checkpoint.
- DeepSeek R1 8B: 98-99% of tokens exit early during autoregressive decoding, 7.2% prefill latency reduction, 6.6% throughput improvement.
- Qwen3 8B: 8.1% throughput improvement at batch 8.

**Sutra relevance:** Can be applied to any model we build as a free post-training optimization. But modest gains compared to training-time approaches.

#### 6.3.5 PonderLM / PonderLM-2 (May-Sep 2025) — arXiv:2505.20674, 2509.23184

**Pretraining models to "ponder" in continuous latent space.**

PonderLM: Instead of generating a real token, model yields a weighted sum of all token embeddings according to predicted distribution, fed back as input for another forward pass. Self-supervised, no human annotations. Demonstrated on GPT-2, Pythia, LLaMA.

PonderLM-2 (CRITICAL RESULT):
- Generates an intermediate latent thought (last hidden state) before predicting the next token.
- **PonderLM-2-Pythia-1.4B significantly surpasses vanilla Pythia-2.8B** on both LM and downstream tasks.
- PonderLM-2-Pythia-1.26B matches Pythia-2.8B with 55% fewer parameters.
- PonderLM-2-Pythia-1.4B reaches Pythia-2.8B's final performance with 62% less training data.
- At identical inference cost, a model with 1 extra latent thought per token outperforms a standard model with DOUBLE the parameters.

**Sutra relevance:** EXTREMELY relevant. Proves latent pondering provides massive parameter efficiency (2x effective capacity). This is exactly the kind of "Intelligence = Geometry" result we're looking for. A 1.4B model beating a 2.8B model through better computation structure, not more params.

#### 6.3.6 Huginn-3.5B (Feb 2025) — arXiv:2502.05171

**Depth-recurrent language model with scalable latent computation.**

- Physical architecture: 2 Prelude + 4 Recurrent + 2 Coda blocks (only 8 unique blocks).
- At 132 unrolls, 8-layer physical model behaves like 132-layer virtual model.
- Performance on math/logic improves consistently with more recurrent steps (4 to 32+).
- Can match computation equivalent to 50B parameters at sufficient unrolling.
- Latent reasoning: model "thinks" internally through recurrent refinement before emitting prediction.
- Caveat: improvements from recurrence are modest vs explicit Chain-of-Thought; lags behind best CoT-augmented models on GSM8K.

**Sutra relevance:** Validates recurrent depth with weight sharing at 3.5B scale. The Prelude/Recurrent/Coda pattern maps to our stage decomposition. Caveat about CoT gap is important — latent reasoning may have limits.

#### 6.3.7 Relaxed Recursive Transformers (Google DeepMind, ICLR 2025) — arXiv:2410.20672

**Converting existing LLMs into smaller recursive transformers with layer-wise LoRA.**

- Each looped layer gets multiple LoRA modules (one per iteration).
- Recursive Gemma 1B outperforms TinyLlama 1.1B and Pythia 1B — even recovers most performance of original Gemma 2B (no sharing).
- Achieves 25-55% parameter cost reduction for comparable performance.
- Continuous Depth-wise Batching: new inference paradigm for recursive transformers + early exit, potential 2-3x throughput gains.

**Sutra relevance:** LoRA per recursion iteration is an elegant way to add per-loop specialization without abandoning weight sharing. The cost is minimal (low-rank deltas). This could replace or complement our stage-specific parameters.

#### 6.3.8 Inner Thinking Transformer (ITT) — ACL 2025, arXiv:2502.13842

**Each transformer layer = one thinking step. Dynamic deepening at token level.**

- Adaptive Token Routing selects and weights important tokens for inner thinking.
- Thinking Step Encoding + Residual Thinking Connection.
- ITT layer iterates thinking multiple times, accumulating results.
- Tested at 162M (ITT-4) and 230M/460M configurations.

**Sutra relevance:** Similar concept to our recurrent passes but with explicit per-token routing. The accumulation-based approach (rather than replacement) is worth considering.

#### 6.3.9 Latent Thinking Tokens (Multi-paper trend, 2025-2026)

**Pause tokens (arXiv:2310.02226):** Training+inference with learnable delay tokens. 1B model: +18% SQuAD, +8% CommonSenseQA.

**Coconut / Continuous Thought (arXiv:2412.06769):** Feed last hidden state back as next input instead of decoded token. Enables breadth-first search in latent space. Outperforms CoT on logical reasoning requiring search.

**Token Assorted (arXiv:2502.03275):** Mix latent and text tokens.

**Adaptive Latent CoT (arXiv:2602.08220):** Variable-length latent CoT trajectories before each token. Longer for hard tokens, shorter for easy. Emerges from one-stage pretraining.

**Latent Lookahead (Mar 2026, arXiv:2603.20219):** Multi-step lookahead in latent space before committing to next token. Substantially outperforms AR and non-AR baselines on planning tasks (maze, Sudoku, ProsQA).

**Sutra relevance:** The field is rapidly converging on latent computation as a key efficiency mechanism. Multiple independent lines of evidence show that thinking in continuous space before committing to discrete tokens provides substantial gains. This validates our recurrent pass design.

---

### 6.4 Multi-Source / Multi-Teacher Learning (Updated)

#### 6.4.1 Knowledge Purification (Feb 2026) — arXiv:2602.01064

**Addresses the critical problem: conflicting rationales among multiple teachers.**

- Distillation performance DECLINES as number of teachers increases — knowledge conflict is real.
- Solution: consolidate multiple teacher rationales into a single purified rationale before distillation.
- Five methods: aggregation, routing, RL-based selection.
- LLM routing methods best on out-of-domain datasets.
- Significantly enhances KD performance across student models and datasets.

**Sutra relevance:** CRITICAL finding. Confirms our concern about multi-teacher conflicts. We need purification/routing, not naive averaging.

#### 6.4.2 SAMerging (Dec 2025) — arXiv:2512.21288

**Model merging as multi-teacher KD on scarce unlabeled data.**

- Sharpness-Aware Minimization (SAM) to find flat minima.
- +4.5% on TA-8, +11.7% on TALL-20 over AdaMerging.
- Works with as few as 16 examples per task.
- 10x fewer calibration data than prior methods.

**Sutra relevance:** SAM for flat minima in multi-teacher settings is a transferable technique.

#### 6.4.3 Pre-training Distillation Design Space (ACL 2025) — arXiv:2410.16215

**Systematic exploration: logits processing, loss selection, scaling law, offline vs online.**

- Teacher: GLM-4-9B, Student: 1.9B.
- Key finding: **larger students benefit MORE from pre-training distillation.**
- Counter-intuitive: **larger teacher does NOT guarantee better results.**
- Validates distillation during pretraining (not just fine-tuning).

**Sutra relevance:** Validates pre-training distillation. The "larger student benefits more" finding suggests we should push our model toward 1B+ to maximize KD gains. Teacher selection matters more than teacher size.

#### 6.4.4 Complementary Knowledge Transfer (arXiv:2310.17653)

**Arbitrary pretrained model pairs have complementary knowledge — even across families.**

- Confidence-based, hyperparameter-free data partitioning: models autonomously adopt teacher/student roles.
- Works even when model families or performances differ.
- Motivated by continual learning lens.

**Sutra relevance:** We can extract useful signal even from small, weak teachers. Architecture diversity matters more than teacher quality.

#### 6.4.5 Universal Sparse Autoencoders (ICML 2025) — arXiv:2502.03714

**Single SAE that ingests activations from ANY model and decodes them for ANY other model.**

- Jointly learns universal concept space across multiple models.
- Discovers semantically coherent universal concepts (colors, textures, parts, objects).
- Enables coordinated activation maximization across models.
- Cross-model, cross-task, cross-dataset.

**Sutra relevance:** A potential mechanism for cross-architecture distillation. Instead of aligning teacher-student representations directly, map both to a universal concept space.

#### 6.4.6 Model Stitching / Feature Transfer (2025-2026) — arXiv:2506.06609

**Affine mappings between residual streams effectively transfer features between models.**

- Small and large models learn highly similar representation spaces.
- Simple linear transformations sufficient for cross-model feature transfer.
- Validates that representation-level distillation (not just logit-level) is viable.

**Sutra relevance:** If representations are similar across models, our CKA-based distillation approach is well-motivated.

#### 6.4.7 MiniPLM (ICLR 2025) — arXiv:2410.17215

(Covered in Section 5 but updated numbers:)
- **2.2x pre-training acceleration.**
- Offline Difference Sampling: adjusts training data distribution based on teacher-reference discrepancy.
- Supports KD across model families (flexibility via corpus-level operation).
- Teacher: Qwen 1.8B -> Students: 200M, 500M, 1.2B.
- Benefit extends to larger training scales (scaling curve extrapolation).

#### 6.4.8 DSKD — Cross-Tokenizer KD (EMNLP 2024, updated Mar 2026) — arXiv:2603.22056

**Dual-Space Knowledge Distillation solves vocabulary mismatch between teacher and student LLMs.**

- Exact Token Alignment (ETA) algorithm aligns tokens across differently tokenized sequences.
- Dual-space: distills in both logit space and hidden state space simultaneously.
- CMA (Cross-Model Attention) projects student embeddings into teacher space as queries.
- CMA-GA variant uses adversarial learning for better cross-model alignment.
- **Reaches or surpasses same-tokenizer KD performance** — cross-tokenizer is no longer a barrier.
- Open source: https://github.com/songmzhang/DSKD

**Sutra relevance:** CRITICAL for O4. Our 16K custom tokenizer differs from all teachers (Qwen, Phi, etc.). DSKD proves we can distill from any teacher regardless of tokenizer. The dual-space approach aligns with our hidden-state distillation plans. Could replace our current MiniPLM-only pipeline with direct online KD.

#### 6.4.9 Knowledge Purification — Multi-Teacher Conflict Resolution (Feb 2026) — arXiv:2602.01064

**Five purification methods for resolving conflicting knowledge from multiple teacher LLMs.**

- Problem: multiple teachers produce conflicting rationales due to hallucinations, inconsistent reasoning paths, and expertise differences.
- **Best methods:** similarity-based router (+5% over naive multi-teacher) and RL-based teacher selection.
- Consolidates multiple rationales into a single purified rationale before distillation.
- All 5 methods improve distilled student performance vs. naive multi-teacher.

**Sutra relevance:** Directly validates R7 intuition that "teacher-budget routing beats raw KD weight." When we add multi-teacher, use similarity router per source/domain, NOT naive averaging. One family per batch, routed by similarity.

#### 6.4.10 ARMADA — Cross-Modal KD to Text-Only Students (Mar 2026) — arXiv:2603.10877

**First architecture-agnostic framework for distilling from black-box VLM teachers to text-only LMs.**

- TS Aligner module bridges teacher's multimodal abstraction space to student's text space.
- Works with black-box teachers (API-only, no internal access needed).
- +3.4% NLU improvement, +2.6% generative reasoning boost.
- Validated on DeBERTa-v2-1.4B, OPT-1.3B, LLaMA-{3B, 7B, 8B}.
- No multimodal pre-training or teacher fine-tuning required.

**Sutra relevance:** Proves that text-only students CAN absorb structural knowledge from vision-language models without becoming multimodal. The TS Aligner concept maps directly to our Universal Teacher Interface idea — a shared latent space that any teacher family projects into. Could extract CLIP's semantic structure into Sutra without adding vision capability.

#### 6.4.11 Falcon-H1 Budget Allocation (Jul 2025) — arXiv:2507.22448

**Falcon-H1 parallel hybrid allocates SSM:Attention:MLP = 2:1:5 (8 total chunks).**

- At 0.5B: 36L, dim=1024, 8 attn heads + 24 SSM heads (GQA 8Q/2KV). MLP gets 62.5%.
- At 1.5B: 24L, dim=2048, 8 attn + 48 SSM heads. Same budget ratio.
- Attention and SSM run IN PARALLEL, concatenated before output projection.
- "Relatively small fraction of attention is sufficient" — most capacity goes to SSM + MLP.
- MLP still dominates budget at 5/8 = 62.5%, similar to our 60.6%.

**Sutra relevance:** Falcon-H1's 62.5% MLP allocation is nearly identical to our current 60.6% FFN. But Falcon succeeds where we fail because: (1) their branches are SSM heads (stateful, not projected) and attention heads (standard), not projected dual-branch; (2) outputs are concatenated with a shared output projection, not averaged; (3) at 0.5B+ they have enough capacity. At our 100M scale, the question remains: is 60% too much for FFN when the novel mechanism gets only 17%?

---

### 6.5 Data Efficiency

#### 6.5.1 Synthetic Continued Pretraining (ICLR 2025 Oral)

**EntiGraph: entity-centric augmentation converting small corpus into large synthetic corpus.**

- Breaks text into entities, uses LM to describe inter-entity relations.
- Iteratively "fills in" the knowledge graph underlying the corpus.
- Accuracy scaling is log-linear in synthetic token count.
- Significantly outperforms continued pretraining on source documents or paraphrases.

**Sutra relevance:** Can amplify our 23B token corpus by generating synthetic entity-relation descriptions. Potentially 5-10x data amplification.

#### 6.5.2 DoReMi (NeurIPS 2023) + Online Data Mixing (2024-2025)

**Domain Reweighting with Minimax Optimization.**

- Small 280M proxy model sets domain weights for 8B model training.
- +6.5% average few-shot accuracy over default Pile domain weights.
- Reaches baseline accuracy with 2.6x fewer training steps.
- Online Data Mixing (ODM) further improves: 4.8% lower perplexity vs Pile Weights.

**Sutra relevance:** Domain reweighting is free and proven. Should be standard practice for our training. Even our small proxy model can set weights for larger runs.

#### 6.5.3 MTP Curriculum (ACL 2025) — arXiv:2505.22757

**Curriculum from NTP to MTP for small models.**

- Gradually increases prediction complexity during pretraining.
- Enables smaller LMs to better leverage MTP objective.
- Improves downstream NTP performance and generation quality.
- Addresses Meta's finding that MTP hurts small models — the curriculum may solve this.

**Sutra relevance:** If we use MTP, curriculum approach may overcome the small-model capacity bottleneck.

#### 6.5.4 Data Quality > Quantity (Field Consensus)

- Phi-3/4: GPT-4 judges content quality, keeps only score >= 7 from 10T tokens -> 1.5T filtered.
- SmolLM2: manual refinement of mixing rates between training stages.
- Pangu Pro MoE: 3-phase (general 9.6T -> reasoning 3T synthetic CoT -> annealing 0.4T curriculum).
- A 13-year-old human learns from <100M tokens; LLMs use 15T. The efficiency gap is enormous.

**Sutra relevance:** Our 23B tokens, if high-quality and well-mixed, could be sufficient with the right architecture + KD.

---

### 6.6 Recurrent Depth / Weight Sharing (Updated)

#### 6.6.1 TokenFormer (ICLR 2025 Spotlight) — arXiv:2410.23168

**Treats model parameters as tokens — fully attention-based scaling.**

- Replaces ALL linear projections with token-parameter attention (input tokens = queries, params = keys/values).
- Progressive scaling: 124M -> 354M -> 757M -> 1.4B using only 30B additional tokens (1/10th compute of from-scratch).
- 1.4B: perplexity 11.77 vs Transformer's 11.63, but at 1/3rd training cost.

**Sutra relevance:** Progressive scaling is an exciting idea — train small, scale up cheaply. But the mechanism (attention over params) may be too expensive at small scale.

#### 6.6.2 Field Consensus on Recursive Transformers (2025-2026)

- Weight sharing via recurrence reaches comparable/higher performance at 25-55% parameter cost.
- Universal Transformer pattern validated across multiple papers.
- Spiral-like refinement within loops, larger state changes between blocks.
- Huginn: 8 physical layers -> 132 virtual layers at 3.5B scale.
- MoR: router-based token-level depth selection.
- LoopFormer: budget-conditioned with shortcut consistency.
- Relaxed Recursive: LoRA per iteration for specialization.

**Sutra relevance:** Strong consensus that recurrent depth + weight sharing is the right approach for parameter-efficient models. Our design should use this as the backbone.

---

### 6.7 Edge Deployment / Quantization-Native

#### 6.7.1 Quartet: Native FP4 Training (NeurIPS 2025) — arXiv:2505.14669

**End-to-end FP4 training that is near-lossless in the large-data regime.**

- All major computations (linear layers) in FP4.
- New low-precision scaling law quantifying performance trade-offs across bit-widths.
- Almost 2x speedup over FP8 on NVIDIA Blackwell RTX 5090.
- Attains lowest loss across token-to-parameter ratios.
- Improves upon LUQ-INT4 by 10% relative loss.
- Requires ~15% fewer params and 5x less data to reach same loss.

**Sutra relevance:** DIRECTLY relevant — we HAVE an RTX 5090. FP4 native training could halve our training time. Architecture must be compatible with FP4 from the start.

#### 6.7.2 BitNet Design Principles (2024-2025)

Key takeaways for quantization-native architecture:
- Train natively at target precision — not post-hoc quantization.
- Ternary weights {-1, 0, +1} eliminate FP multiply entirely.
- Activations can be quantized more aggressively (INT4/FP4) with proper training.
- RMS normalization before quantization is critical.
- Embedding and output head can also be quantized with minimal loss.

---

### 6.8 Speculative Decoding (Inference Speed)

#### 6.8.1 EAGLE-3 (2025-2026)

- Fuses hidden states from multiple intermediate layers (not just final).
- 2-6x speedup depending on model size.
- No separate draft model needed — uses target model's own representations.

#### 6.8.2 LayerSkip (Meta, ACL 2024, integrated 2025)

- Self-speculative decoding via early exit.
- Up to 2.16x speedup on summarization.
- Training recipe: layer dropout (low for early, high for later layers) + early exit loss.
- Integrated into HF transformers (Nov 2024), PyTorch torchtune (Dec 2024), HF trl (Mar 2025).

#### 6.8.3 Field Consensus (2025-2026)

- Speculative decoding is now production standard (vLLM, SGLang, TensorRT-LLM).
- Self-speculative (early exit) is preferred for small models (no separate draft model).
- Mirror-SD (2026): up to 5.8x speedup via branch-complete rollouts from early-exit signals.

**Sutra relevance:** Our elastic compute design (variable recurrent depth) is a NATURAL fit for self-speculative decoding. Early exits at fewer passes = draft tokens, full passes = verification. This should be designed in from the start, not bolted on.

---

### 6.9 Updated Competitive Landscape (March 2026)

| Model | Params | Tokens | Key Results | Data Eff. |
|-------|--------|--------|-------------|-----------|
| SmolLM2-135M | 135M | 2T | HS 42.1%, PIQA 68.4% | Low |
| SmolLM2-1.7B | 1.7B | 11T | HS 68.7%, ARC 60.5% | Low |
| MobileLLM-350M | 350M | ~1T | +4.3% over prior SOTA | Medium |
| BitNet 2B4T | 2B | 4T | HS 84.3%, MMLU 52.1% | Medium |
| Qwen3-0.6B | 0.6B | ~8T | Matches Qwen2.5-1.5B | Medium |
| Qwen3-4B | 4B | ~18T | Matches Qwen2.5-7B | Medium |
| Phi-4-mini | 3.8B | 5T | HS 83.5%, ARC 83.7% | Med-High |
| Gemma-3-1B | 1B | 2T | Distilled from larger | Med-High |
| RWKV-7 2.9B | 2.9B | 5.6T | 71.5% avg (matches Qwen2.5-3B@18T) | **HIGH** |
| Huginn-3.5B | 3.5B | ? | Scales to 50B-equiv compute | High |
| PonderLM-2 1.4B | 1.4B | 300B | Beats Pythia-2.8B | **VERY HIGH** |

**Key insight: The data efficiency leaders are all architectural innovations (RWKV-7, PonderLM-2, Huginn) not just data curation approaches (SmolLM2, Phi-4).** This validates the Sutra thesis: better architecture = better data efficiency.

---

### 6.10 Synthesis: Implications for Sutra Architecture Design

**Architecture decisions supported by March 2026 evidence:**

1. **Recurrent depth with weight sharing is validated.** MoR, LoopFormer, Huginn, Relaxed Recursive, PonderLM-2 all prove this works. PonderLM-2's 2x effective capacity from latent pondering is the strongest result.

2. **Hybrid SSM+Attention with ~10% attention fraction.** Hymba, Jamba, Granite, Zamba, Bamba all converge on this. Mamba-3 is the latest SSM; Gated DeltaNet is the latest linear attention.

3. **Token-level adaptive depth is ready.** MoR provides the clearest framework. LoopFormer's shortcut-consistency training is key for quality at all depths.

4. **Multi-teacher KD needs purification.** Knowledge Purification paper confirms naive multi-teacher hurts. Need routing/selection mechanism.

5. **Quantization-native from day one.** BitNet and Quartet prove native low-bit training works. Design for FP4/INT4 from the start. Quartet gives 2x speedup on our RTX 5090.

6. **Data efficiency through latent computation.** PonderLM-2 (2x params effective), RWKV-7 (3.2x data efficiency), curriculum MTP all show architecture can substitute for data.

7. **Speculative decoding is free with elastic depth.** Self-speculative decoding from early exits is production-ready and integrates naturally with variable-depth recurrence.

8. **32+ layers optimal even at 70-100M scale.** Deep-and-thin with weight sharing is the right structural prior.

**Open questions for T+L session:**
1. Mamba-3 vs RWKV-7 vs Gated DeltaNet for the non-attention component?
2. PonderLM-2 style latent pondering vs MoR router-based adaptive depth?
3. Hymba-style hybrid heads (SSM+attention within layers) vs layer-interleaving?
4. How to integrate multi-teacher KD with curriculum MTP?
5. Quartet FP4 training on RTX 5090 — implementation readiness?

---

### 6.10.1 Round 2 Evidence Addendum (2026-03-26)

This addendum records the specific evidence that drove the Round 2 architecture update in `research/ARCHITECTURE.md`.

- **Local probe: DyT vs RMSNorm (`42M`, `5000` steps).**
  - RMSNorm finished at `5.08` BPT versus `5.83` for DyT.
  - DyT showed higher activation kurtosis and worse generation.
  - Main mechanistic lesson: DyT is not enough as a **drop-in** because it does not normalize the residual stream.

- **Local probe: TOP vs NTP COMPLETED — CATASTROPHIC FAILURE.**
  - NTP-only: BPT `5.07`, kurtosis `0.35` avg / `1.15` max, max activation `22.0`.
  - NTP+TOP (weight=0.05, K=4, activation at step 200): BPT `9.68` (+4.6 worse), kurtosis `18.75` avg / `99.3` max, max activation `133.8`.
  - TOP loss magnitude (~20) comparable to CE (~8), contributing ~12% gradient magnitude that conflicts with NTP gradients.
  - TOP is DEAD at 42M scale with this configuration. Would need weight ≤0.001 or activation at step 2000+ if retried.

- **Muon optimizer evidence is now load-bearing.**
  - Muon offers roughly `2x` compute efficiency versus AdamW and materially better outlier behavior.
  - The strongest external pairing for our purposes is `Muon + Single-Scale RMSNorm`.

- **MiniPLM is now a concrete implementation path, not just a literature note.**
  - The existing shard-weight path in `code/data_loader.py` is sufficient to support offline difference sampling once teacher-reference scoring is added.

- **SmolLM2 sharpened the O4 priority.**
  - Quality filtering should happen before fancy KD.
  - High-quality `23B` tokens are more valuable than an unfiltered `23B` token pool.

- **Multi-teacher evidence tightened the supervision design.**
  - Adaptive weighting matters.
  - Architecture diversity matters more than picking several similar "best" teachers.
  - This supports adaptive family rotation rather than naive static averaging.

- **TOP is now REJECTED as a mainline auxiliary at 42M scale.**
  - External evidence at `340M+` is strong but does not transfer down to `42M`.
  - Local probe showed catastrophic failure: +4.6 BPT worse, kurtosis explosion.
  - Combined with MTP also hurting at 42M (from prior session), there are NO validated auxiliary losses for the scout scale.
  - The scout should train on plain NTP only. Auxiliaries revisited at 200M+.

- **Hybrid architecture research (March 2026) strengthens the HEMD design.**
  - Systematic analysis (arxiv:2510.04800): inter-layer 1:5 attention-to-SSM optimal for efficiency.
  - Intra-layer hybrid (parallel attention+SSM heads) > inter-layer (sequential blocks).
  - "Never place Transformer blocks at the front" — middle layers, evenly distributed.
  - At 350M: hybrid NLL=2.860 vs pure transformer NLL=2.882 — small but real gain.
  - Hymba (NVIDIA): 1.5B intra-layer hybrid gives 11.67x cache reduction, 3.49x throughput vs Llama-3.2-3B.

- **Muon optimizer at small scale confirmed by Essential AI.**
  - Tested across 100M-4B params, works at all scales.
  - Default: lr=0.02, momentum=0.95, nesterov=True, weight_decay=0.1.
  - Muon LR ~67x higher than AdamW LR. Our probe config (0.02) is correct.
  - Only for 2D hidden layer params; embeddings/norms/biases use AdamW.

- **Competitive baselines sharpened.**
  - SmolLM2-135M: HellaSwag 42.1%, ARC 43.9%, PIQA 68.4% (trained on 2T tokens).
  - SmolLM2-360M: HellaSwag 54.5%, ARC 53.0% (trained on 4T tokens).
  - Data gap: SmolLM2-135M uses 87x our tokens. Pythia-160M uses 13x.

- **Muon probe COMPLETED: AdamW+SS-RMSNorm wins, Muon loses at 42M scale.**
  - V1 (AdamW+RMSNorm): final BPT `4.99`, kurtosis_max `1.2`, max_act `22.9`.
  - **V2 (AdamW+SS-RMSNorm): final BPT `4.91` — WINNER.** Kurtosis_max `1.1`, max_act `22.0`. Best BPT, best activation health, fewest params.
  - V3 (Muon+SS-RMSNorm): final BPT `5.01` — LOSES. Kurtosis_max `1.8`, max_act `59.6`.
  - SS-RMSNorm beats RMSNorm by 0.08 BPT with fewer params. Validated for quantization.
  - Muon at lr=0.02: 1.7x faster early convergence (step 1000-1500) but periodic instability (BPT regressions at steps 2000, 3500-4000). Max activations 2.7x higher than AdamW. Final BPT 0.10 worse than V2.
  - **Muon not dead as concept** — possible fixes: lower LR, longer training, NorMuon. But NOT validated at 42M/5000 steps.

- **Falcon-H1 (TII, May-July 2025): Production intra-layer hybrid architecture.**
  - Family of 0.5B to 34B models. Parallel attention + Mamba-2 SSM heads within EVERY block.
  - 0.5B specs: dim=1024, 36 layers, 8 attention heads + 24 SSM heads (1:3 ratio), head_dim=64, state_dim=128.
  - Outputs concatenated then projected. GQA with Q/KV ratio=2.
  - "A relatively small fraction of attention is sufficient for good performance."
  - "A well-designed 1.5B hybrid from 2025 does what a vanilla 7B from 2024 could do."
  - Confirms Hymba's finding: intra-layer parallel hybrid > inter-layer sequential.

- **NorMuon (arxiv:2510.05491): Strict improvement over Muon.**
  - 21.74% better training efficiency than Adam, 11.31% over Muon at 1.1B.
  - Adds neuron-wise normalization after orthogonalization.
  - Negligible additional memory overhead.
  - Not yet in PyTorch — would need custom implementation.

- **MiniPLM confirmed at ICLR 2025. Open source: https://github.com/thu-coai/MiniPLM**
  - Benefit MORE pronounced at 500M and 1.2B than at 200M.
  - "Down-samples common patterns, filters noisy signals, avoids wasting compute on easy knowledge."

### Round 4 Research (2026-03-26)

- **Trunk-choice probe COMPLETE: hybrid WINS over pure transformer!**
  - V1 pure_transformer (12A, 49.1M): BPT=4.9131.
  - V2 pure_conv (12H, 46.3M): BPT=5.6147 — loses by +0.70 BPT.
  - **V3 hybrid_3to1 (3A+9H, 47.0M): BPT=4.7218 — WINS by 0.19 BPT over pure transformer!**
  - V3 also has better activation health: kurtosis_max=1.03 (vs 1.11), max_act=19.2 (vs 23.9).
  - Hybrid wins with FEWER params (47M vs 49M).
  - Inter-layer mixing helps: conv provides long-range receptive field, attention provides exact retrieval.
  - Intra-layer parallel (every block has both) should be even stronger.

- **NorMuon algorithm fully characterized (arxiv:2510.05491).**
  - After Muon's Newton-Schulz orthogonalization, computes per-neuron (row-wise) second-order momentum: `v_t = β₂v_{t-1} + (1-β₂)mean_cols(O_t ⊙ O_t)`.
  - Normalizes each row: `Õ_t = O_t / √(v_t + ε)`.
  - Memory overhead: only m extra scalars per m×n weight matrix.
  - Hypers: β₁=0.95, β₂=0.95, lr=3.6e-4 (124M), lr=7.5e-4 (350M).
  - Small-scale results: 6% gain at 124M, 15% at 350M, 11.31% at 1.1B.
  - Directly addresses our Muon instability: per-neuron normalization would prevent the max_act=59.6 spike.

- **Falcon-H1 0.5B exact config obtained from HuggingFace.**
  - 36 layers, dim=1024, 8 attn heads, 24 mamba heads, head_dim=64, FFN=2048.
  - Mamba-2 with d_conv=4 (tiny!), d_state=128, d_ssm=1536.
  - 1:3 attention:SSM ratio.
  - μP-style multipliers: attention_out=0.9375, ssm_out=0.2357, ssm_in=1.25.
  - Does NOT tie embeddings. vocab_size=32784.

- **Hymba 1.5B architecture details.**
  - 32 layers, dim=1600, 25 attention heads.
  - 5:1 SSM:attention PARAMETER ratio.
  - Combination: parallel paths → mean → output projection (NOT concat).
  - 128 meta tokens (learnable cache initialization, acts as attention drain backstop).
  - Cross-layer KV sharing between every 2 consecutive layers.
  - Only 3 full-attention layers (first, middle, last) — rest use sliding window.

- **Knowledge Purification (Jin et al., Feb 2026): fine-tuning only, NOT pretraining.**
  - 5 methods: GPT-4 aggregation, Plackett-Luce ranking, PLM classifier, similarity router, RL selection.
  - Best: similarity router and RL selection (+5% over naive multi-teacher).
  - Key finding: more teachers WITHOUT purification HURTS performance.
  - Tested on 77M, 248M, 783M students. NOT applicable to pretraining stage.
  - No pretraining-specific multi-teacher purification exists in literature.

- **MiniPLM practical details fully characterized.**
  - Difference Sampling formula: `r(x) = log p_teacher(x) / log p_ref(x)`.
  - Reference model: 104M trained on 5B tokens (SMALLER than all students).
  - Works CROSS-FAMILY: Qwen teacher/ref → Llama and Mamba students.
  - All data scored UPFRONT (no incremental option). Top-50% selected.
  - 200M student: 41.3% vs 39.9% baseline (+1.4% average accuracy).
  - Our corpus (22.9B tokens) is feasible to score in ~1-2 hours on RTX 5090.

- **Design tension identified: depth vs width at 100M.**
  - Falcon-H1-0.5B: 36 layers × dim=1024 = 0.5B params.
  - Our Round 3 spec: 14 blocks × dim=768 = ~100M params.
  - Scaling down: 14 blocks at dim=768 may be too shallow for the hybrid paradigm.
  - Alternative: 24 blocks × dim=512 = ~100M params but much deeper.
  - Falcon-H1 and Hymba both prefer DEPTH over WIDTH for hybrids.
  - HuggingFace study at 70M: 32 layers w/ hidden=384 beats 12 layers w/ hidden=512 (+0.35% avg). But transformer-only, not validated for hybrids.

- **Mamba-3 (ICLR 2026, Together AI): Removes causal conv, adds complex states.**
  - Eliminates the short causal convolution that Mamba-1/2 relied on.
  - Exponential-trapezoidal discretization induces implicit conv-like behavior.
  - Complex-valued state updates for richer state tracking (solves parity that Mamba-2 cannot).
  - MIMO formulation: 4x decode FLOPs at fixed state size, better hardware utilization.
  - 180M results: Mamba-3 MIMO ppl=16.46 vs SISO ppl=16.59 on FineWeb-Edu.
  - Compatible with hybrid: 5:1 SSM:attention ratio tested.
  - Requires custom Triton kernels. Implementation complexity: HIGH.
  - Implication: conv branch could evolve toward Mamba-3 style recurrence later. Start with simpler GatedConv for scout.

### T+L Round 4 Architectural Decisions (2026-03-26)

**Source: `results/tl_round4_output.md` (Codex R4 — authoritative)**

**Key design pivots from Round 3 → Round 4:**

1. **Depth/width: 14×768 → 24-26×512.** "Depth should dominate width." Falcon-H1, Hymba, and local 12×512 probe all point to depth wins. 24×512 for promotion gate, 26×512 for full scout. Keeps proven width from 42M probes.

2. **Conv kernel: k=64 → k=16 (pending microprobe).** Not k=8 as pre-R4 proposed — Codex leans k=16 as default but requests k=4/k=16/k=64 sweep. Rationale: sparse inter-layer probe used k=64 to compensate for infrequent mixing; intra-layer blocks can use shorter kernels since local mixing happens every layer.

3. **Branch ratio: 2:3 → 1:1 (d_attn=256, d_conv=256).** Major pivot from pre-R4 proposal. Codex challenge #6: pure conv losing "argues for more attention while the branch is only GatedConv." 1:1 for GatedConv scout; revisit conv-heavy only after a stronger SSM branch exists. Pending microprobe confirmation.

4. **Fusion: concat-then-project confirmed.** Branch scales ga=gc=1.0 init (Codex proposes equal init, unlike pre-R4's sqrt(d_a/d_c)). Microprobe to compare: mean 1:1 vs scaled-concat 1:1 vs scaled-concat 2:3.

5. **NorMuon replaces Muon lr=0.005 in optimizer probe.** Same as pre-R4. Per-neuron normalization targets observed late-stage spike failure mode.

6. **GQA 4Q/2KV retained.** KV-cache efficient inference.

7. **First 100M scout: brutally simple online.** Plain NTP + fixed exits only. No memory, no online teachers, no TOP, no MTP, no learned halting. Confidence 9/10 for simplicity.

8. **MiniPLM: train custom ~100M Qwen-tokenizer reference.** Use Qwen3-0.6B only as quick pipeline pilot. Same-tokenizer teacher/reference validity to be researched.

**Confidence: 5/6/5/3/5 across 5 outcomes** (O1 up from 4 → 5 after trunk-choice positive result). Key blocker: still no 100M intra-layer win. Key unlock: 100M hybrid beating matched pure transformer with better generations.

### Round 4 Research Findings (2026-03-26)

#### R1: MiniPLM Cross-Tokenizer Validity

**Question:** Is MiniPLM-style difference scoring valid when teacher and reference use different tokenizers after per-byte normalization?

**Answer: Same-tokenizer teacher/reference is STRONGLY preferred.** MiniPLM's difference score (teacher_loss - reference_loss) requires comparable per-token losses. Cross-tokenizer pairs produce different sequence lengths and per-token distributions — per-byte normalization (loss/num_bytes) reduces but doesn't eliminate the noise.

Key finding: **arxiv 2503.20083 "Universal Cross-Tokenizer Distillation via Approximate Likelihood Matching" (ALM)** provides a principled cross-tokenizer distillation method by identifying comparable token chunks and minimizing likelihood differences. This is the state of the art for cross-tokenizer KD. However, it addresses token-level distillation, not MiniPLM-style data selection scoring.

For MiniPLM data reweighting specifically: the difference score's quality degrades with tokenizer mismatch because the reference model's loss distribution shifts in non-semantic ways. **Recommendation: train the ~100M Qwen-tokenizer reference as R4 proposes.** Use Qwen3-0.6B only as pipeline pilot to validate MiniPLM infrastructure before investing in the reference.

Also relevant: **tokenkit** (github.com/bminixhofer/tokenkit) provides tools for cross-tokenizer transfer including model conversion and vocabulary mapping.

#### R2: Best Small Embedding Teachers Under 1B

**Top candidates for representation alignment into decoder students:**

| Model | Params | Key Strength |
|-------|--------|-------------|
| nomic-embed-text | 137M | Smallest, good baseline, Matryoshka support |
| EmbeddingGemma-300M | 300M | Google DeepMind, derived from Gemma 3 |
| GTE-multilingual-base | 305M | Alibaba, strong multilingual |
| mxbai-embed-large | 335M | Strong English MRL |
| BGE-M3 | 568M | BAAI, 100+ languages, dense+sparse+multi-vec |
| Qwen3-Embedding-0.6B | 600M | Qwen family match if using Qwen tokenizer |

**Layer selection:** Last layer of the embedding model only. Research shows intermediate-layer matching doesn't improve alignment quality and over-constrains the student.

**Pooling strategy:** Mean pooling > last-token for decoder-only students. Removing causal attention mask during embedding extraction also works well for decoder-only models.

**Alignment recipe:** Projection layer (student_dim → teacher_dim) + cosine or MSE loss on pooled embeddings. Weight 0.01-0.1 relative to CE loss. Multi-teacher distillation should use staged training (one teacher at a time) or family-per-batch sampling.

**Recommendation for Sutra:** Start with nomic-embed-text (137M) — smallest, good quality, minimal VRAM overhead for parallel teacher inference. Upgrade to BGE-M3 or Qwen3-Embedding if results warrant.

#### R3: Lightest Branch-Balancing Recipes for Two-Path Hybrid Decoders

| Model | Fusion Method | Balance Mechanism | Key Detail |
|-------|-------------|------------------|------------|
| **Falcon-H1** | Concat before output proj | Channel allocation ratio | attn:SSM:MLP ≈ 2:1:5. **Critical: more attention channels HURTS.** Attention should be SMALL fraction. |
| **Hymba** | Weighted sum + norm | Per-channel learnable β₁,β₂ | Y = W_out(β₁·norm(attn) + β₂·norm(ssm)). Extra norm needed because SSM output magnitude > attention, especially in later layers. |
| **Griffin** | Addition | No learned gating | Recurrence + attention outputs simply added |
| **Jamba** | Interleaved layers | N/A (inter-layer) | Not intra-layer; alternating Mamba and attention layers |

**Key insights for Sutra:**

1. **Falcon-H1's most important finding: attention should be the SMALLER branch.** Having more attention channels significantly degrades performance, while SSM↔MLP switching has weaker effect. This *may* argue against R4's 1:1 ratio — but our conv surrogate is weaker than Mamba-2, so more attention may be needed to compensate. The microprobe (1:1 vs 2:3) will resolve this.

2. **Hymba's per-channel β is the lightest effective recipe.** Our current scalar s_a/s_c is even simpler — one scalar per branch rather than per-channel. The microprobe will test whether this suffices.

3. **Branch-output normalization before fusion is important.** Hymba explicitly normalizes both outputs before combining because SSM output magnitudes grow in later layers. Our current code does NOT normalize before concat. If the microprobe shows kurtosis/max_act problems in the concat variants, adding pre-fusion norm is the first fix.

4. **No model uses complex gating networks.** All successful hybrids use either simple scalars, per-channel vectors, or just addition/concat. This validates our approach of keeping branch balancing lightweight.

**Recommendation (UPDATED by R5 microprobe):** Scalars DID fail. All concat variants show kurtosis 1.4-5.8. R5 verdict: switch to **normalized additive fusion** — per-branch RMSNorm + per-channel β (Hymba-style) + mean combination. This is now the HEMD-R5-G block design.

---

### T+L Round 5 Findings (2026-03-26)

#### R4 Microprobe Results (Complete)

4-variant concat-project probe, all C-blocks, 12L, dim=512, SS-RMSNorm, 5K steps:

| Variant | Params | Kernel | Ratio | BPT | kurtosis_max | max_act |
|---------|--------|--------|-------|-----|-------------|---------|
| concat_1to1_k4 | 47.5M | 4 | 1:1 | 4.9421 | 1.4 | 26.9 |
| concat_1to1_k16 | 47.6M | 16 | 1:1 | 4.8320 | 2.0 | 35.2 |
| concat_1to1_k64 | 47.7M | 64 | 1:1 | 4.8502 | 5.8 | 28.1 |
| concat_2to3_k16 | 50.9M | 16 | 2:3 | **4.7713** | 1.9 | 33.1 |

Plus mean fusion results (P-blocks, 9L):
| mean_k4 | 46.0M | 4 | full-dim | **4.8005** | **0.82** | **18.15** |
| mean_k64 | 46.2M | 64 | full-dim | 4.9536 | 1.45 | 20.88 |

**Key findings:**
1. **Kernel preference is fusion-dependent:** Mean→k=4 wins. Concat→k=16 wins. Hypothesis: learned w_mix projection leverages broader receptive field in concat.
2. **Concat instability is universal:** ALL concat variants show growing kurtosis in 3000-4000 step window. This is a fusion-method problem, not kernel or ratio.
3. **Mean fusion strictly dominates:** mean_k4 at 9L (4.80) beats concat_k16 at 12L (4.83) with 2.4x better kurtosis, 1.9x better max_act, 25% fewer layers.
4. **2:3 ratio helps but is capacity-confounded:** +0.06 BPT but +7% params. Not a pure ratio effect.
5. **Hymba research confirmed:** Our scalar s_a/s_c with no pre-fusion norm is the exact failure mode Hymba identified and solved.

#### Hymba Fusion — Deep Technical Details (from arxiv)

Exact formula: `Y = W_out_proj(β₁·norm(attn_out) + β₂·norm(ssm_out))`
- Per-CHANNEL learnable β vectors (not scalars)
- Each branch INDEPENDENTLY normalized (RMSNorm) before combining
- Element-wise ADDITION (not concat), then output projection
- Motivation: "SSM output magnitudes consistently larger than attention heads"
- Architecture uses 5:1 SSM:attention parameter ratio

#### Falcon-H1 Fusion — Deep Technical Details (from arxiv 2507.22448)

Channel allocation: SSM:Attention:MLP = 2:1:5 (attention = 1/8 of channels)
- "More attention channels significantly degrades performance"
- Concat SSM + attention outputs → output projection (no pre-fusion norm)
- Solves magnitude imbalance by making attention tiny, not by normalizing
- SA_M (semi-parallel) configuration: SSM + attention in parallel, MLP sequential

#### R5 Architecture Decision: Normalized Additive Fusion

The HEMD-R5-G block synthesizes:
- **From Hymba:** per-branch normalization + per-channel β before combining
- **From mean fusion probes:** additive combination (0.5*(a+c)) for stability
- **From GQA:** 4Q/2KV grouped-query attention for inference efficiency
- **From kernel probes:** k=4 (the stable-family winner)

Each branch (d=256) → project to full dim (512) → independent RMSNorm → per-channel β → average → output projection.

This is the first block design informed by both empirical probes AND deep literature research on branch fusion stability.

#### R6 100M Gate Results & Stability Analysis

**100M gate (24×512, 5K steps): HEMD-R5-G vs pure transformer**

| Model | Params | Final BPT | kurtosis_max | max_act |
|-------|--------|-----------|-------------|---------|
| Transformer 24×512 | 90.2M | 4.8087 | 1.66 | 44.8 |
| HEMD-R5-G 24×512 | 93.5M | 4.7019 | 3.94 | 63.6 |

BPT: +0.11 (passes ≥0.10 criterion). Stability: FAILS (kurtosis 2.4x worse).

**Trajectory insights:**
- Hybrid learns faster early (step 1000-1500: up to +0.73 BPT advantage)
- Mid-training advantage erodes (step 2000-3000: 0-0.03)
- Late recovery during cooldown (step 4500-5000: +0.11-0.13)
- Instability grows monotonically — kurtosis peaks 5.09 at step 4500

**Root cause hypothesis:** Per-channel β vectors (dim=512 each) reintroduce unbounded scale after normalization. Compounds across 24 layers. Evidence: the 42M P-block (fixed 0.5 mean, no β) was stable (kurtosis 0.82).

**R6 fix strategy (Codex-designed):**
- **R6-F (primary):** Remove β entirely, use SS-RMSNorm for branches, keep projected branches + 0.5*(a+c) mean fusion
- **R6-S (backup):** Replace β with softmax mixing (α_a + α_c = 1 per channel)
- Both include per-layer telemetry for branch RMS, norm scales, fused RMS

**Key insight:** The 100M gate validated the hybrid FAMILY (local+global complementarity produces genuine BPT gains). The exact fusion mechanism (per-channel β) is what failed. The BPT gain appears before instability grows, suggesting the underlying complementarity is real.

#### MiniPLM Data Scoring (O4 Progress)

**Pilot (200 windows, 10% corpus):** Infrastructure validated end-to-end.
- Teacher: Qwen3-1.7B-Base, Reference: Qwen3-0.6B-Base
- Diff score: 0.264 ± 0.113
- Source ranking: gutenberg (0.37) > wikipedia (0.29) > fineweb (0.28) > math (0.23) > wildchat (0.21)
- Interpretation: literature/factual text has highest teacher-reference gap = highest training value for small models

**Full scoring (500 windows) completed.** Source-level shard weights derived and saved to `results/miniplm_shard_weights.json`. Range: 0.80x (TinyStories) to 1.17x (Gutenberg). O4 data-shaping probe config ready at `code/q7_o4_data.json`.

### 6.4.12 LFM2: Interleaved Pure-Block Hybrid (Liquid AI, Nov 2025)

**Source:** arxiv.org/abs/2511.23404

LFM2 (350M-8.3B params) uses a different hybrid strategy than intra-layer fusion:
- **Architecture:** 16 blocks = 10 gated short conv + 6 GQA attention, interleaved
- Each block is PURE (either conv-only or attention-only) — no intra-layer fusion
- Conv block: `out = Wout(DWConv1D(Win(x)) * sigmoid(Wgate(x)))` — simple gated conv
- Found via hardware-in-the-loop NAS over operator types per layer position
- Key finding: "most benefits from hybrid SSM/linear-attention can be captured by short convolutional submodules + small number of global attention layers"
- Scales from 350M to 8.3B with consistent stability

**Sutra relevance:** Our hybrid instability comes from FUSION (mixing attn + conv outputs in same block). LFM2 avoids this by keeping blocks pure. Their conv:attn ratio (62.5%:37.5%) is inverted from our 6G+18A (25%:75%) but both validate "conv for local, attention for global." Key question for architecture design: should we switch to inter-layer interleaving instead of fighting intra-layer fusion stability?

### 6.4.13 Gemma 2 Logit Soft-Capping (Google, Jun 2024)

**Source:** arxiv.org/abs/2408.00118

Technique to prevent activation/logit growth without hard truncation:
- Formula: `x ← cap * tanh(x / cap)`
- Applied to attention logits (cap=50) and final output logits (cap=30)
- Smooth, differentiable, bounded — activations cannot exceed ±cap
- Proven at scale with minor quality impact when removed during inference

**Sutra relevance:** Directly applicable to our max_act problem in hybrid conv branches. Our 6G+18A variant hit max_act=58.5 (threshold 52) while achieving BPT 4.686 (beating transformer). Applying soft-capping to conv output before fusion (e.g., cap=40) would solve the max_act violation while preserving the BPT advantage.

### 6.4.14 QK-Norm and Training Stability (2024-2025 Standard)

**Sources:** Multiple — now adopted by Gemini 2.0, DeepSeek-V3, Gemma 2, etc.

Applying RMSNorm/LayerNorm to Q and K vectors before attention score computation:
- Prevents attention logit growth over training
- Allows 1.5x higher learning rates
- Prevents attention entropy collapse (softmax → one-hot)
- Now considered standard practice for any transformer training

**Sutra relevance:** We already use SS-RMSNorm on block input. Should verify QK-norm is applied within the attention computation. Could improve stability of attention branches in hybrid blocks.

### 6.4.15 ZClip: Adaptive Gradient Spike Mitigation (Apr 2025)

**Source:** arxiv.org/abs/2504.02507

Z-score-based adaptive gradient clipping:
- Tracks running mean + std of gradient norms via EMA
- Clips only when current gradient norm is statistically anomalous (high z-score)
- Outperforms fixed-threshold and percentile-based clipping on 1B LLaMA
- Proactively adapts to training dynamics without assumptions about gradient scale evolution

**Sutra relevance:** Could help with activation/gradient spikes in conv branches. Unlike fixed clipping, adapts to the changing dynamics of training (early steps have different gradient distributions than late steps).

### 6.4.16 Cross-Tokenizer Distillation: The Problem is SOLVED (2024-2026)

**The biggest perceived blocker to multi-family KD — tokenizer mismatch — has been solved by at least 3 independent methods.**

#### DSKD: Dual-Space Knowledge Distillation (EMNLP 2024, updated 2025)
**Source:** arxiv.org/abs/2406.17328, github.com/songmzhang/DSKD

Projects teacher/student hidden states into each other's representation spaces, then computes KL divergence in unified output spaces. For different-vocabulary models, uses Exact Token Alignment (ETA) to align tokens across differently tokenized sequences.
- **Cross-tokenizer KD performs AS WELL AS same-tokenizer KD** — the mismatch is fully overcome
- DSKDv2 (2025): better projector initialization, supports on-policy distillation
- DSKD-CMA-GA (2025): generative adversarial alignment for distribution matching
- Tested on instruction-following, math reasoning, code generation
- **Sutra relevance:** Our 16K custom tokenizer is NO LONGER a blocker for logit-level KD from any teacher family

#### ULD: Universal Logit Distillation (ICML 2024)
**Source:** arxiv.org/abs/2402.12030

Uses optimal transport (OT) to align logit distributions across different tokenizers. The transport plan maps teacher tokens to student tokens based on distribution geometry, not string matching.
- Works across completely different tokenizer families
- No need for token alignment preprocessing
- **Sutra relevance:** Alternative to DSKD — simpler but potentially less precise

#### MultiLevelOT: Multi-Level Optimal Transport (AAAI 2025)
**Source:** ojs.aaai.org/index.php/AAAI/article/view/34543

Extends ULD with token-level AND sequence-level alignment using Sinkhorn distance approximation of Wasserstein distance. Captures complex distribution structures that single-level OT misses.
- Most sophisticated cross-tokenizer method
- **Sutra relevance:** Best option if we want maximum signal transfer quality

#### Approximate Likelihood Matching (2025)
**Source:** arxiv.org/abs/2503.20083

Principled method that directly matches the likelihood of text under teacher and student, bypassing token alignment entirely. First method to enable effective distillation across "fundamentally different" tokenizers.
- **Sutra relevance:** Potentially the cleanest approach — no token alignment, no optimal transport, just likelihood matching

#### Cross-Tokenizer Likelihood Scoring (Dec 2025)
**Source:** arxiv.org/abs/2512.14954

Exploits BPE's implicit recursive structure to derive a probabilistic framework for cross-tokenizer likelihood scoring. No token alignment — uses the tokenizer's own structure. Applied to Qwen2.5-1.5B with 12% memory reduction and up to 4% performance improvement on evaluated tasks, 2% improvement on GSM8K.
- **Sutra relevance:** This paper's BPE recursive approach is a logit-level analogue of our byte-span bridge. Could be combined with our representation-level KD for richer supervision.

#### CDM: Contextual Dynamic Mapping (ACL Findings 2025)
**Source:** aclanthology.org/2025.findings-acl.419

Uses contextual information to enhance sequence alignment precision and dynamically improve vocabulary mapping between mismatched tokenizers. Addresses both sequence misalignment and vocabulary composition mismatch.

#### DWA-KD: Dual-Space Weighting and Time-Warped Alignment (Feb 2026)
**Source:** arxiv.org/abs/2602.21669

Combines DSKD's dual-space projection with time-warped alignment to handle sequence-length mismatches. Adds learned per-token weighting.

#### DSKDv2 with Key-Query Matching (Mar 2026)
**Source:** arxiv.org/abs/2603.22056

Latest DSKD iteration. Uses key-query matching for vocabulary-mismatch alignment, improving over the ETA (Exact Token Alignment) used in DSKDv1. General framework for cross-architecture, cross-tokenizer KD.

**Key synthesis:** The tokenizer mismatch that ARCHITECTURE.md flagged as the reason to prefer representation-first KD (Section A7) is no longer a valid concern. We can now do FULL logit-level KD from any teacher family. This dramatically expands the design space for multi-source learning. **Our byte-span bridge for hidden-state alignment is a valid parallel approach. Future: add logit-level KD (ULD/MultiLevelOT) on top of our representation-level CKA for richer multi-surface supervision.**

### 6.4.17 Knowledge Purification: Scaling Multi-Teacher KD (Feb 2026)

**Source:** arxiv.org/abs/2602.01064

THE critical paper for multi-teacher KD scaling. Addresses the fundamental problem: **performance DECLINES as teacher count increases** due to conflicting rationales (hallucinations, inconsistent reasoning paths, different expertise domains).

Five purification methods tested:
1. **Similarity-based router** — routes each sample to the most aligned teacher
2. **RL-based teacher selection** — learned policy for which teacher to use when
3. **Rationale consolidation** — merge multiple teacher rationales into one
4. **Confidence-weighted averaging** — weight teachers by per-sample confidence
5. **Voting/ensemble** — majority vote among teacher outputs

Results: **Similarity-based router and RL-based selection consistently achieve highest accuracy across all student models.** Both purification approaches not only improve performance but also generalize to out-of-domain tasks.

**Sutra relevance:** DIRECTLY applicable. When distilling from 5+ teacher families, we MUST use knowledge purification to prevent degradation. The similarity-based router is the simplest effective method — dynamically routes each training sample to the teacher whose knowledge is most relevant. This is NOT the failed 2Q rotation system — it's dynamic per-sample routing, not static per-batch rotation.

### 6.4.18 Cross-Architecture Distillation: Transformer to SSM (2025)

**Source:** arxiv.org/abs/2510.19266 (CAB), goombalab.github.io/blog/2024/distillation-part1-mohawk/ (MOHAWK)

Two complementary approaches for distilling knowledge across fundamentally different architectures:

#### CAB (Cross-Architecture Bridge)
Lightweight attention bridge enables token-level supervision from transformer teacher to SSM student. Unlike output-only KD, CAB aligns intermediate representations through flexible layer-wise matching.
- More data-efficient than naive distillation
- Works across vision and language
- **Sutra relevance:** If our student is hybrid (conv+attention), we can align conv blocks with SSM teachers and attention blocks with transformer teachers

#### MOHAWK (NeurIPS 2024)
3-phase progressive distillation: (1) match mixing matrices, (2) match hidden states per block, (3) match end-to-end predictions. Specifically designed for Transformer→Mamba transfer.
- Progressive phases prevent catastrophic forgetting
- **Sutra relevance:** Already noted in RESEARCH.md. The progressive approach may work for multi-family KD too — align representations first, then logits.

### 6.4.19 Dynamic Teacher Weighting in Multi-Teacher KD (2025)

**Source:** Multiple papers, emergentmind.com/topics/adaptive-weighting-in-multi-teacher-knowledge-distillation

Key finding: **Adaptive, instance-aware weighting nearly always outperforms static schemes.**

Methods:
- **RL-based teacher selection** — trains a small policy network to choose which teacher(s) to attend to per sample
- **Meta-learning weighting** — learns per-sample teacher weights as a meta-objective
- **Confidence routing** — weight each teacher by its prediction confidence on the current sample
- **Student-guided selection** — weight teachers based on alignment with student's current predictions

The DiverseDistill framework handles heterogeneous teacher committees by dynamically weighting based on the student's understanding of each teacher's expertise.

**Sutra relevance:** Static rotation (the failed 2Q system) vs dynamic routing (this) — the difference is learning WHEN each teacher is most useful, not just cycling through them. Critical for our multi-family setup where different teachers excel at different content types.

### 6.4.19b Knowledge Purification in Multi-Teacher KD (2026)

**Source:** arxiv.org/abs/2602.01064, "Exploring Knowledge Purification in Multi-Teacher Knowledge Distillation for LLMs"

**Core finding: Routing to a single best teacher per sample beats combining all teachers.** Teacher conflicts (hallucinations, inconsistent reasoning, expertise mismatches) are the #1 failure mode of multi-teacher KD.

5 purification methods tested:
1. **Knowledge Aggregation** — GPT-4 synthesizes all rationales into one (expensive, offline)
2. **Plackett-Luce Ranking** — probabilistic ranking by historical teacher performance (no training)
3. **PLM Classifier** — MLP predicts routing probabilities per sample (lightweight)
4. **Similarity-based Router** — contrastive learning on embeddings, cosine routing (BEST overall)
5. **RL-based Teacher Selection** — policy network learns optimal teacher per question type

**Key result:** Similarity-based router achieves highest Conflict Mitigation Value (CMV) across all student sizes. Routing (select one teacher) consistently beats combining (average all teachers). The mechanism: routing eliminates exposure to conflicting signals rather than trying to harmonize them.

**Sutra relevance:** Directly validates our 4-bucket routing MVP. Our approach routes samples to Specialist-Q, Specialist-L, or Consensus based on per-sample disagreement metrics — this IS similarity-based routing. Key difference from this paper: their routing is question→teacher similarity, ours is disagreement + student-gap based. Ours is arguably more principled because it accounts for the student's current knowledge state.

### 6.4.20 Hybrid Scheduling Theoretical Foundations (2026)

**Source:** results/research_hybrid_scheduling.md (background research agent, March 2026)

Key theoretical results for block scheduling in hybrid architectures:
1. **SSM must come BEFORE attention** (arxiv.org/abs/2603.08859): Proven, not empirical. SSM compresses context → attention retrieves from compressed state. Attention-first = no better than pure model.
2. **Only 2% of attention heads handle ALL retrieval** (arxiv.org/abs/2602.11374): The rest can be replaced with recurrent layers for 5-6x memory savings.
3. **Early layers 10x more important than late** (arxiv.org/abs/2603.22473): Layer 0 does most aggressive representation transformation. Ablating early third of SSM = -55% MMLU; ablating late third = -5%.
4. **Parallel hybrids are Pareto-optimal** for quality-efficiency (arxiv.org/abs/2510.04800): But harder to stabilize (confirmed by our R5-R7 experience).
5. **"Never place transformers at the front"**: Three independent theoretical explanations converge.
6. **Mamba-3 BCNorm** (ICLR 2026): RMSNorm after B,C projections stabilizes SSM training, analogous to QK-norm in transformers.
7. **Falcon-H1 dampening multipliers**: 35 fine-grained muP groups with non-learnable dampening tensors for activation control.

### 6.4.21 Codex Architecture Theorist Review: Meta-Learning KD Framework (2026-03-26)

**Source:** Codex GPT-o4-mini-high, Architecture Theorist persona, reviewing SCRATCHPAD.md meta-learning framework.

**Key verdicts on the meta-learning KD framework from SCRATCHPAD.md:**

1. **Category-theoretic framing is overkill for implementation.** Acceptable as scratchpad language for "structure-preserving transfer," but the operational math should be disagreement geometry: pairwise divergence matrices, clustering, and conditional mutual information. Keep category language in research, not in code.

2. **"Inverted power law" is not a law yet.** Rename to "transient adaptive acceleration." Can only appear when teacher diversity is real, routing quality improves over time, AND student is far from capacity ceiling. Fails under teacher redundancy, objective mismatch, bridge noise, and student unlearnability.

3. **5-category decomposition is unidentifiable with 3 teachers.** Missing categories include objective/surface mismatch, bridge-induced disagreement, domain expertise, calibration-only disagreement. With one Qwen, one LFM, one EmbeddingGemma, capacity-dependent and family-dependent effects are confounded.

4. **Knowledge State Map is useful ONLY if it directly drives routing and curriculum.** If it becomes a hand-built ontology of "syntax/facts/reasoning" it becomes noise. Treat as a slice-eval mastery matrix, not a universal capability theory.

**Concrete MVP recommendations:**
- Expose multi-depth teacher states (teacher at relative depths 1/3, 2/3, 1.0 matched to student exits 7/15/23)
- Per-sample audit metrics (per-sample state loss and cosine semantic loss, not batch-aggregated)
- Split committee by surface (Qwen/LFM routed on state surface, EmbeddingGemma fixed semantic anchor)
- 4-bucket routing every 500 steps: Consensus, Specialist-Q, Specialist-L, Uncertain/Bridge-noisy
- PCGrad for multi-task gradient conflict (CE vs state KD vs semantic KD)
- Curriculum: consensus first, specialist second, uncertain last

**Decision:** Do NOT implement full category theory or 5-way ontology first. Start with routed KD + 4-bucket audit + exit-depth tracking. Kill the framework if it doesn't beat static multi_3family at equal teacher FLOPs.

### 6.4.22 Codex Correctness Engineer Audit: KD Training Code (2026-03-26)

**Source:** Codex GPT-o4-mini-high, Correctness Engineer persona, auditing dense_baseline.py KD infrastructure.

**Bugs found and status:**
- **HIGH: Multi-teacher unfair loss scaling** — alpha applied per-teacher, so 2-teacher arm gets 2x total KD weight. **FIXED: divide alpha by n_teachers per surface.**
- **HIGH: Student byte offsets hardcoded to None** — student falls back to position-based pooling, misaligned with teacher byte-span pooling. **FIXED: compute via re-encoding (prior session).**
- **HIGH: offset_mapping device mismatch** — HF tokenizer returns CPU tensor, attention_mask on GPU. **Working in practice** (probe runs successfully — HF handles gracefully for slow tokenizers). Left as-is since probe validates.
- **MEDIUM: KD surfaces silently disappear** — hidden=None skipped without warning. **FIXED: assert added.**
- **MEDIUM: Only final hidden state used for KD** — depth-8/16 planned but not implemented. **By design for first probe, deferred.**
- **MEDIUM: Projector LR lost after decay** — identified by LR comparison. **FIXED: is_projector flag (prior session).**
- **MEDIUM: Warm-start VRAM spike** — full checkpoint loaded to GPU. **FIXED: map_location="cpu".**
- **MEDIUM: compute_token_kd_loss unsafe for vocab mismatch** — dormant code. **Deferred** until token-level KD is activated.

**Assumption note:** HF tokenizer offset_mapping returns character offsets, not byte offsets. The "byte-span bridge" is technically a character-span bridge on Unicode text. Functionally equivalent for most content.

### 6.4.23 Codex Strategic Review: KD Approach and Next Steps (2026-03-26)

**Source:** Codex GPT-o4-mini-high, comprehensive strategic review of all KD hypotheses and decisions.

**Major decisions:**

1. **NO 120K production run** on current KD stack. Current +0.023 BPT at step 1500 justifies more KD work but NOT 6+ GPU days on current system.
2. **KD confirmed as mainline** approach.
3. **Abort arm 3** (multi_3family) — unfair loss-scaling bug makes win uninterpretable, loss unhelpful. Violates signal-first rule.
4. **Head-start hypothesis: partially approved, not concluded.** Gap narrowing (0.059→0.042→0.023) consistent with transient acceleration, especially since KD loss saturates early and this is a 5K CE warm-start. At ~106M tokens by step 1500, too early to call the limit. Extrapolating to zero at step 3000 NOT justified.
5. **Logit-level KD: APPROVED as next major test.** Direct supervision on teacher's predictive distribution is the real advantage. Cross-tokenizer blocker removed via DSKDv2. Prefer key-query path over ETA unless ETA is materially simpler.
6. **15K validation gate: approved** — validate best logit-enhanced variant against control, not current system alone.
7. **Alpha tuning: do NOT blanket-increase.** 22.9B is corpus size; actual 120K plan only sees ~1.97B tokens. Alpha should follow signal quality, not corpus size. If tuning: sweep or schedule. If forced to choose, front-load state alpha and decay rather than increasing constant semantic weight.
8. **No bigger teacher yet.** Change supervision surface first, keep teacher fixed. Otherwise experiment is confounded.
9. **Routing/scheduling: premature.** Build fair static baseline with stronger KD surfaces first.

**Meta-process corrections:**
- Architecture search too long: AGREED.
- RMFD before signal validation: AGREED.
- "Fundamentals produced understanding not novelty": Codex correction — novelty was wrong KPI. The real mistake was letting theory pull implementation order ahead of signal validation.

**Codex-mandated execution plan:**
1. Let arm 2 finish to 3000
2. Kill arm 3
3. Reconcile probe-state inconsistency in ledger
4. Implement minimal single-teacher cross-tokenizer logit KD
5. Run 3K ablation: control vs rep-KD vs logit-KD vs rep+logit-KD (same Qwen3-1.7B)
6. Take winner to 15K with lm-eval at 5K/10K/15K
7. Launch 120K ONLY if sustained loss advantage + benchmark lift

**Process rule:** One teacher, one new surface, one clean codepath → short ablation → 15K gate → only then: teacher-size ablation → fair static multi-teacher → routing.

### 6.4.24 Codex Correctness Engineer Review: Cross-Tokenizer Logit KD (2026-03-26)

**Source:** Codex GPT-o4-mini, Correctness Engineer persona, read-only review of `compute_cross_tok_logit_kd()` and integration in `train_kd()`.

**3 HIGH issues found (ALL FIXED):**
1. **Causal alignment bug:** Nearest byte-end alignment could pair student position to a LATER teacher position, leaking future text. Fix: Changed from `abs()` distance to causal `signed_diff >= 0` with argmin — only aligns to teacher positions whose char-end <= student char-end.
2. **Loss scaling mismatch:** `F.kl_div(reduction="batchmean")` divides by B only, not B*T. With T=512, alpha_logit=1.0 would be ~512x too large vs CE loss. Fix: Changed to `reduction="none"`, sum over vocab dim, then divide by count of valid positions.
3. **Teacher truncation masking:** If teacher tokenizer is finer-grained, teacher runs out of tokens before student. Late student positions get wrong supervision. Fix: Added `valid_mask` that detects positions where no valid teacher token exists, zeros out KL loss there.

**3 MEDIUM issues found (ALL FIXED):**
1. **Missing offset fallback:** Added edge-case returns for None offsets and N_shared=0.
2. **Unfair ablation weights:** Rep-only total=0.8, logit-only=1.0, rep+logit=1.8. Fix: Normalized all arms to total=1.0 (rep_only: state=0.625, semantic=0.375; logit_only: logit=1.0; rep+logit: state=0.3125, semantic=0.1875, logit=0.5).
3. **`train_kd_phased()` stale `use_hidden`:** Still used `use_hidden = has_kd` instead of surface-aware check. Fix: Changed to `needs_hidden = any("state" in t.surfaces or "semantic" in t.surfaces for t in current_teachers)`.

**Additional note:** Temperature validation added (`assert temperature > 0`). Memory analysis: real pressure from `(B,T,N_shared)` tensors (~116 MiB each at N_shared=14822), not `(B,T_s,T_t)` alignment tensor (~8 MiB). Total extra memory ~350 MiB for logit KD path.

**Gradient flow:** Confirmed correct — student path stays in-graph, teacher under `no_grad()`.

**Status:** All HIGH and MEDIUM fixes applied. **Re-review PASSED (2026-03-26):** Codex Correctness Engineer confirmed all 6 fixes correct, no new regressions. Tier 1 loop CLEAN.

### 6.4.25 KD First Probe Final Results: Rep KD = Head-Start Only (2026-03-26)

**Experiment:** 2-arm probe, 3000 steps each from step-5000 warm-start. Arm 1 = control (NTP only). Arm 2 = single Qwen3-1.7B teacher, rep KD (state CKA alpha=0.5 + semantic relational alpha=0.3).

**Final BPT comparison:**

| Step | Control BPT | KD BPT | Delta |
|------|-------------|--------|-------|
| 500 | 4.9536 | 4.8946 | -0.059 |
| 1000 | 4.9194 | 4.8772 | -0.042 |
| 1500 | 4.8725 | 4.8493 | -0.023 |
| 2000 | 4.8250 | 4.7888 | -0.036 |
| 2500 | 4.8583 | 4.7142 | -0.144 |
| 3000 | **4.5579** | **4.5500** | **-0.008** |

**Stability at step 3000:** KD arm kurtosis_max=17.6 (vs 4.8 control, 3.6x ratio). KD arm max_act=80.5 (vs 69.0 control).

**Key findings:**

1. **Rep KD provides transient acceleration only.** Gap peaked at -0.059 (step 500), narrowed to -0.008 (noise) by step 3000. The student learns the same features with or without rep KD — the teacher just helps find them faster.

2. **WSD × KD interaction creates a transient artifact.** Step 2500 showed -0.144 gap because control REGRESSED during early LR decay (4.8250→4.8583) while KD continued improving. Both converged during full decay. Control's total decay-phase drop was -0.300 BPT; KD's was -0.164.

3. **CKA + relational losses are GLOBAL STRUCTURE losses** (span-pooled, batch-level Gram matrices). These saturate when the student's representations become structurally aligned with the teacher's — which happens faster than functional convergence. CKA convergence ≠ functional convergence.

4. **Kurtosis divergence is a warning.** 3.6x kurtosis ratio at 3000 steps. If this grows linearly with training length, could reach 50+ by 15K steps. Needs monitoring.

**Implication for logit KD:** Logit KD provides TOKEN-LEVEL PREDICTION supervision (not global structure). Should be harder to saturate because the student must match the teacher's output distribution, not just its representation geometry. The 4-arm surface ablation (NOW RUNNING) tests whether logit KD provides persistent vs transient advantage.

### 6.4.26 Codex Scaling Expert + Architecture Theorist: KD Probe Interpretation (2026-03-26)

**Source:** Codex GPT-5.4, Scaling Expert + Architecture Theorist combined persona.

**Key findings:**

1. **Rep KD = fast geometric regularizer, not durable teacher.** CKA loss doesn't fully saturate (ends at 0.036) but the objective mismatch is the issue: 16-span CKA + semantic Gram teach global structure fast but don't sharpen token prediction after coarse geometry is learned.

2. **Kurtosis warning: RED FLAG for long runs.** 17.6 vs 4.8 at 3K steps. Critical: after WSD decay starts, control kurtosis RECOVERS (6.7→4.8) while KD gets WORSE (7.8→17.6). Thresholds for 15K gate: kurtosis >2x control OR max_act >1.15x control = yellow/red. Would NOT launch 120K with this kurtosis signature without fixing it first.

3. **Logit KD should be more persistent.** Supervises the actual prediction target (next-token distribution) at every position, not just global span geometry. 92.6% student vocab overlap is "unusually favorable." Could still degrade to head-start at 90M if student lacks capacity.

4. **T=2.0 / K=64 confirmed.** If sweeping: sweep temperature (1.5, 2.0, 2.5) not K. K=64 retains ~95% of full-logit signal.

5. **Persistence thresholds: 10K = evidence, 15K = proof.** Logit KD must be >0.02 BPT better than control at 10K AND >0.015-0.02 at 15K with at least one lm-eval lift. If <0.01 by 10K, it's another head-start.

6. **90M is below the common KD comfort zone.** Ratio 1:19 (90M/1.7B). Literature success at 1:1.5 to 1:9. Recommendation: if logit KD weak at 15K, scale student to 166-200M rather than scaling teacher.

**Codex verdict:** "Rep KD is acting like a fast geometric regularizer, not a durable teacher. Logit KD is the right next surface, T=2.0/K=64 is a good first shot, 15K is the real persistence gate, and 90M is usable for scouting but probably below the size where a 1.7B teacher's gains become durable."

### 6.4.27 Codex Architecture Theorist: KD Probe Deep Interpretation (2026-03-26)

**Source:** Codex GPT-5.4, Architecture Theorist persona (separate session from §6.4.26).

**Unique insights not captured in §6.4.26:**

1. **Informational saturation, not just numeric saturation.** CKA/Gram losses align global geometry fast, then stop telling the student *which token boundary to move*. KD loss keeps decreasing (0.032→0.018, -43.8%) but this is a measure of global alignment, not prediction-relevant information transfer. Treat rep KD as saturated once bin-avg KD loss is ~0.020 or lower and anneal it to near-zero before decay starts (fade around step 2000, ≤10% by step 2400).

2. **~~TEMPERATURE CONFLICT~~ RESOLVED (see §6.4.28).** This session recommended T=1.0 based on Minitron, but literature synthesis shows: (a) Minitron was same-tokenizer at ~1:2 ratio, doesn't apply; (b) cross-tokenizer consensus is T=2.0 (DSKDv2, DWA-KD, Design Space paper all use T=2.0); (c) T∈{1.0, 1.5, 2.0} all similar for pre-training KD. **T=2.0 is correct for our setup. No sweep needed unless logit KD shows zero benefit.**

3. **Staged transfer option.** For 1:19 ratio, literature suggests intermediate teacher: 1.7B→300-400M→90M. Alternative: keep direct 1.7B→90M but make logit KD dominant late signal while rep KD is early-only.

4. **6K minimum screen** for persistence (2x rep-KD horizon). 3K is inconclusive. Convincing evidence: live gap at both 10-12K and final 15K, with ΔBPT≤-0.03 at 6K and ≤-0.02 at 15K.

### 6.4.28 Cross-Tokenizer Logit KD: Literature Synthesis and Parameter Resolution (2026-03-26)

**Source:** Web research sweep covering cross-tokenizer KD temperature, top-K, and capacity ratio literature.

**Temperature resolution (§6.4.27 conflict RESOLVED):**

The §6.4.26/§6.4.27 conflict — T=2.0 vs T=1.0 — is now resolved by literature consensus:

- **DSKDv2 (EMNLP 2024, updated 2025):** Uses T=2.0 as default for cross-tokenizer distillation. Their ETA alignment is specifically designed for T=2.0 softening.
- **DWA-KD (Feb 2026):** Uses T=2.0 for dual-space cross-tokenizer alignment.
- **Pre-training Distillation Design Space (ACL 2025):** Systematic sweep finds T∈{1.0, 1.5, 2.0} all perform similarly for pre-training KD. No statistically significant difference. Recommends T=2.0 as a safe default but notes robustness across the range.
- **Minitron (NVIDIA, 2024):** T=1.0 finding was for SAME-TOKENIZER distillation at moderate capacity ratios (~1:2). Does NOT generalize to cross-tokenizer settings.

**Conclusion:** T=2.0 is correct for our setup (cross-tokenizer, ETA alignment). The §6.4.27 Minitron recommendation doesn't apply because: (a) different tokenizers require more softening to align across vocabulary mismatch, and (b) their tested ratio was ~1:2, not 1:19. **The SCRATCHPAD theory predicting T=1.0 outperforms T=2.0 at extreme ratios remains untested but is LOWER PRIORITY** — if the current T=2.0 ablation shows logit KD benefit, there's no need to sweep T.

**Top-K validation:**

- **Peng et al. (ACL 2025):** Most comprehensive sweep — K∈{1,3,5,10,20,50,100} with top-p pre-filtering. **K=50 was best: 39.6% avg. K=100: 38.3%.** Even K=1 (hard teacher label) helped. Two-stage top-p→top-k achieves 4,000x storage reduction (~15TB for 100B tokens).
- **BiLD (COLING 2025):** K=8-50 actively helps. LLM logit kurtosis is 2-3 orders of magnitude higher than vision models (Qwen-4B kurtosis: 46K-135K vs ResNet: 782-995). **Top-8 logits cover 99.5-99.998% of probability mass** in LLMs. K=8 optimal for task-tuning.
- **MultiLevelOT (AAAI 2025):** Uses K=50 for cross-tokenizer OT. "Smaller k insufficiently captures full sentence structure; larger k introduces noise."
- **⚠️ Sparse Logit Sampling (ACL 2025, Oral):** **CRITICAL WARNING — naive top-K truncation produces BIASED gradient estimates.** Mathematically proven that truncating to top-K and renormalizing gives biased teacher distribution estimates. Their importance-sampling alternative provides unbiased estimates with similar sparsity. Practical impact may be small (Peng et al. K=50 still works well) but worth noting for theoretical rigor.
- **Our K=64:** Slightly above the Peng et al. optimal K=50. Acceptable. The bias warning is noted but practical results across multiple papers suggest K∈[32,64] works fine for pre-training KD.

**Capacity ratio concern (the 1:19 problem):**

- **Distillation Scaling Laws (Busbridge et al., ICML 2025):** No fixed optimal ratio. Capacity gap follows a broken power law with U-shaped curve: student CE decreases to an optimum then increases with increasing teacher size. Students 143M-12.6B, teachers 198M-7.75B, up to 512B tokens. **Supervised learning always outperforms distillation given enough student compute/tokens** — the crossover scales with student size.
- **Peng et al. (ACL 2025):** Explicit threshold — **"Pre-training distillation is effective when student reaches ~10% of teacher size."** Our 90M/1.7B = 5.3%, below this threshold. Also: "A larger teacher does not necessarily guarantee better results" — their 32B teacher didn't always beat 9B for same student.
- **Our ratio:** 1:19 at 5.3%, below the 10% empirical threshold. This is a genuine concern, not just theoretical.
- **Mitigations available:** (a) Scale student to 166M (ratio 1:8.8, above 10% threshold); (b) Multi-teacher distillation may compensate by providing diverse signal; (c) Top-K filtering becomes MORE important at extreme ratios (teacher long-tail is pure noise for small student).
- **Cross-tokenizer alignment partially compensates:** Our 92.6% student vocab overlap means most of the student's representational capacity is covered.

**Cross-tokenizer alignment methods (comprehensive survey):**

- **DSKDv2/ETA (EMNLP 2024, updated 2025):** Exact Token Alignment — matches positions where previous tokens match, current tokens match, and teacher's predicted token exists in student vocab. **>90% of tokens exactly matched** across different tokenizers. Projector init: W_t→s = W_t · (W_s)+ (pseudo-inverse).
- **DWA-KD (Feb 2026):** Dual-level: token-level via dual-space KL with **entropy weighting** (upweights tokens where student uncertain + teacher confident) + sequence-level via Soft-DTW. Outperforms DSKD by 1.5-2.0 ROUGE-L. **The entropy weighting concept could improve our byte-span bridge.**
- **CDM (Feb 2025):** Coverage varies wildly by pair: Llama3→Gemma2: 85.5% seq/67.8% vocab; Llama3→OPT: 89.0% seq/34.5% vocab; Phi3→Qwen2: 31.5% seq/65.7% vocab. **Cross-tokenizer coverage is highly pair-dependent.**
- **ALM/tokenkit (Mar 2025):** Byte-position matching — **closest published method to our byte-span bridge.** First pure distillation (not auxiliary) across fundamentally different tokenizers. Llama 3.2 3B → Qwen2 tokenizer: ALM 77.1 avg vs DSKD 75.8.
- **Cross-Tokenizer Likelihood Scoring (Dec 2024):** Exploits recursive BPE structure for exact probability computation. Qwen2.5-Math-7B → Gemma2-2B: >2% GSM8K improvement. Also supports vocabulary trimming (151K → 16-64K tokens with up to 4% improvement + 12% memory reduction).
- **MultiLevelOT (AAAI 2025):** OT-based, no explicit token mapping needed. Token + sequence level Sinkhorn. Reduces teacher-student gap by >71% on QED.

**Synthesis for current ablation and 15K gate:**

| Parameter | Setting | Confidence | Fallback |
|-----------|---------|------------|----------|
| Temperature | T=2.0 | HIGH (cross-tokenizer consensus, 6 papers) | T=1.0 only if logit KD shows zero benefit |
| Top-K | K=64 | MEDIUM (K=50 optimal per Peng et al.; bias warning noted) | K=32 if marginal; importance sampling if bias matters |
| Capacity ratio | 1:19 (5.3%) | LOW (below 10% threshold per Peng et al.) | Scale student to 166M if KD fails |
| Alignment | ETA (14,822 shared tokens) | HIGH (92.6% coverage) | DWA-KD entropy weighting; ALM byte-position matching |

**Key papers cited:** Peng et al. ACL 2025 (design space), Busbridge et al. ICML 2025 (scaling laws), Sparse Logit Sampling ACL 2025 Oral (bias warning), DWA-KD arXiv 2602.21669 (entropy weighting), ALM/tokenkit arXiv 2503.20083 (byte matching), CDM arXiv 2502.11104 (coverage data), Cross-Tokenizer Likelihood arXiv 2512.14954 (BPE recursion).

### 6.4.29 Extreme-Ratio KD Mitigations: Research Survey (2026-03-26)

**Context:** Our 90M:1.7B ratio (1:19, 5.3%) is below Peng et al.'s 10% effectiveness threshold. This survey identifies proven mitigations for extreme capacity gaps.

**1. Divergence choice — AMiD alpha-mixture (arXiv 2510.15982, 2025):**
- Introduces assistant distribution interpolating between teacher and student: `r = (λ·p^((1-α)/2) + (1-λ)·q^((1-α)/2))^(2/(1-α))`
- **Tested at 15x ratio (GPT-2 XL 1.5B → GPT-2 0.1B) — closest published setup to ours**
- α=-3 to -5 empirically optimal (mode-seeking behavior); λ=0.1 (closer to student)
- +1.64 ROUGE-L over best prior method. Unifies forward KL, reverse KL, JSD into one framework
- **Key insight for us:** At extreme ratios, forward KL forces mode-covering (impossible for 90M student). Mode-seeking (reverse KL / AMiD α<<-1) lets student focus on representable modes

**2. Reverse KL — MiniLLM (arXiv 2306.08543):**
- Replaces forward KL with reverse KL for generative models. Better at avoiding mode-covering that overwhelms small students. Standard approach for extreme-ratio LLM KD.

**3. Curriculum — MiniPLM Difference Sampling (ICLR 2025, arXiv 2410.17215):**
- Offline curriculum for pre-training: `score(x) = log p_teacher(x) / p_ref(x)` where p_ref is small reference model
- Down-samples easy instances, up-samples hard, removes noisy
- **1.8B → 200M (9x): +1.4 pts average. Achieves same performance with 2.2x less compute**
- Directly applicable: use our step-5K checkpoint as reference, score all 246 shards offline
- We already ran a MiniPLM 10% pilot (task #237) — results available

**4. Curriculum — DA-KD Difficulty-Aware (ICML 2025):**
- Per-sample Distillation Difficulty Scores from CE loss, stratified into tiers
- **Outperforms SOTA by 2% with half training cost. Surpasses teacher at 4.7x compression**

**5. Curriculum — POCL Progressive Overload (arXiv 2506.05695, June 2025):**
- Rising temperature schedule tau=1.0→2.0 across 4 stages, difficulty-ranked data
- At **15x ratio (1.5B→0.1B)**: +2.33 ROUGE-L. Larger gains at larger capacity gaps
- **Rising temperature is "critical" for gains** — validates a dynamic tau schedule for our 15K gate

**6. Temperature scheduling — Unified Revisit (arXiv 2603.02430, March 2026):**
- Unexpectedly large τ (10-20) become optimal with longer training
- Fine-grained datasets prefer τ≥10; coarse-grained prefer τ~1
- Smaller τ works better early, larger τ better late — **crossover effect**
- **Implication:** Our static τ=2.0 may be suboptimal. A rising schedule (τ=1.5→4.0) could help

**7. Feature matching — Flex-KD Task-Tangent (arXiv 2507.10155, July 2025):**
- Identifies functionally relevant teacher dimensions via Jacobian of outputs w.r.t. hidden states
- Selects top-d_S coordinates by functional contribution — **no learned projector needed**
- Tested at 7.9x ratio (BERT 110M→14M). **Final layer only is sufficient — multi-layer adds noise**
- Outperforms CKA-based methods especially under severe dimension mismatch

**8. CKA limitations (IJCAI 2024, arXiv 2401.11824 + arXiv 2509.25253):**
- CKA has known bias in low-sample, high-dimensional regimes
- High CKA does NOT guarantee functionally similar features
- **Geometry-aware alternatives** (Procrustes distance, Feature Gram Matrix norms) show statistically significant improvements over CKA

**9. Speculative KD (ICLR 2025, arXiv 2410.11325):**
- Student proposes tokens, teacher replaces poorly-ranked ones. On-the-fly training aligned with student's distribution
- **Tested Qwen 7B→0.5B (14x ratio).** Outperforms both supervised and on-policy KD

**10. Compute-optimal teacher (Busbridge et al. ICML 2025):**
- A smaller, better-trained teacher may outperform a larger one. Capacity gap manifests as U-shaped curve
- **Qwen3-0.6B (1:7 ratio, within comfort zone) might be a better primary teacher than Qwen3-1.7B (1:19)**
- This is testable: add Qwen3-0.6B as a committee member and compare per-token signal quality

**11. Curriculum Temperature for KD — CTKD (AAAI 2023, arXiv 2211.16231):**
- Learnable temperature module with gradient reversal. Key insight: **constant T is suboptimal** for a growing student
- Temperature should INCREASE as student progresses (easy→hard curriculum)
- Adversarial training: temperature module tries to maximize distillation loss (increase difficulty)
- Negligible computational overhead, plug-and-play with existing KD frameworks
- **Directly validates our rising τ schedule and supports the inverted-U α concept**

**12. WSD-α (alpha warmup for KD):**
- Multiple sources reference starting α at 0 and linearly increasing during warmup phase
- InDistill (2024) introduces explicit warmup stage for KD with curriculum-based layer-wise distillation
- **Our inverted-U adds a TAPER phase during WSD decay — novel extension not seen in literature**

**Synthesis — ranked by impact × feasibility for our 15K gate:**

| Mitigation | Impact | Effort | When to Deploy |
|------------|--------|--------|----------------|
| Rising τ schedule (1.5→4.0) | HIGH | TRIVIAL | 15K gate (modify get_lr_wsd to also schedule τ) |
| Reverse KL / AMiD divergence | HIGH | LOW | 15K gate (replace KL computation) |
| MiniPLM data reweighting | MEDIUM | LOW (pilot done) | After 15K gate proves KD value |
| Qwen3-0.6B as primary teacher | MEDIUM | LOW | Phase 3 committee (test alongside 1.7B) |
| CKA → Feature Gram Matrix | MEDIUM | LOW | If rep KD arm shows promise |
| Difficulty-aware curriculum | MEDIUM | MEDIUM | Phase 4 specialist bursts |
| Speculative KD | HIGH | MEDIUM | Requires implementation work |
| Flex-KD task-tangent | LOW | HIGH (Jacobians) | Not feasible during pre-training |

### 6.4.30 Surface Ablation Results: 4-Arm KD at 90M (2026-03-26)

**Setup:** 4-arm ablation from 5K warm-start, 3000 steps each, WSD schedule (decay at step 2400). Teacher: Qwen3-1.7B-Base (1:19 ratio). All KD arms use flat α=1.0 total.

| Arm | Config | BPT@3K | Δ vs Control | Verdict |
|-----|--------|--------|-------------|---------|
| 1. Control | No KD | 4.498 | — | Baseline |
| 2. Rep-only | CKA+semantic, α=1.0 | 4.516 | +0.018 (worse) | HEAD-START ONLY (3rd confirmation) |
| 3. Logit-only | Cross-tok ETA, T=2.0, K=64, α=1.0 | 4.783 | +0.285 (MUCH worse) | ACTIVELY HARMFUL at flat α=1.0 |
| 4. Rep+logit | state=0.3125,sem=0.1875,logit=0.5 | RUNNING | — | Pending |

**Key findings:**

**1. Rep KD = confirmed head-start, 3rd time.** Trajectory: +0.038→-0.002→-0.094→-0.130→+0.004→+0.018. Peaked at -0.130 BPT advantage mid-training, collapsed during WSD decay. Control consolidated 15x better during decay (-0.329 vs -0.022 BPT improvement). Kurtosis spiked 3.7→5.6 during decay. **Basin compatibility theory:** Rep KD's basin is optimized for CKA loss geometry, not NTP loss — shallower on the NTP surface, unstable during consolidation.

**2. Logit KD = persistent penalty at flat α=1.0.** Constant ~+0.27 BPT penalty throughout stable-LR phase. KD loss oscillated 1.5-2.2 without convergence — student cannot represent 1.7B teacher distribution at 5.3% capacity. Gradient competition: KD loss magnitude (~2.0) comparable to CE (~3.7), creating destructive interference. **Not a surface failure — a mechanism failure.** α=1.0 from step 0 at 1:19 ratio overwhelms NTP learning.

**3. The ablation answers "does flat α=1.0 work at 1:19?" — NO.** Neither surface works with static maximum alpha. The MECHANISM (alpha scheduling, temperature scheduling) matters more than the surface choice.

**4. Inverted-U alpha schedule derived from evidence.** α should warm up (let NTP stabilize), peak mid-training (student ready to absorb), then taper during WSD decay (let NTP reconsolidate). Specific schedule: 0.2→0.7 over 2K steps, hold 0.7 for 8K, taper to 0.1 over 2K, hold 0 for 3K.

**5. Crude heuristic principle (from user).** Making implicit mechanisms explicit, even crudely, yields orders-of-magnitude improvement. Flat α is an implicit "all-or-nothing" choice. Rising α is a crude explicit mechanism that creates a tunable lever. Even a bad schedule should beat flat α=1.0.

**Still open:** Arm 4 tests surface orthogonality (do rep and logit interfere or complement?). Codex strategic review pending for 15K gate design.

### 6.4.31 Codex Strategic Review: 15K Gate Design (2026-03-26)

**Codex role:** Scaling Expert + Architecture Theorist. Reviewed full ablation evidence and proposed 15K gate design.

**Decision 1:** Test mechanism fixes (α schedule + τ schedule + confidence gating), NOT teacher swap. Keep Qwen3-1.7B. Answer one clean question: can repaired token-level KD persist at 90M?

**Decision 2:** Logit-only surface. Rep KD failed persistence 3 times. No rep surface in primary arm.

**Decision 3:** Lower peak α than proposed (0.60, not 0.70). Logit KD's penalty was much larger than rep KD's — logit arm should run cooler.
- α: 0.10→0.60 over 0-2K steps, hold 0.60 for 2K-10K, taper 0.60→0.10 over 10K-12K, then 0.0 for 12K-15K
- τ: 1.5→3.0 linearly by 10K, then hold
- K=64, ETA alignment, forward KL
- Confidence gating: ×1.5 if teacher p_max>0.5, ×0.3 if p_max<0.1, else ×1.0

**Decision 4:** No AMiD/reverse KL in first gate. Fallback if scheduled forward-KL is stable but still flat.

**Decision 5:** Stop rules — by 6K: non-positive vs control; by 10K: ΔBPT ≤ -0.02; by 15K: ΔBPT ≤ -0.015 + lm-eval lift. Abort if kurtosis > 2× control or max_act > 1.15× control. If fail → scale to 166M (d=768, 24L, 12H, 1:8.8 ratio).

### 6.4.32 Codex Correctness Engineer Audit: 15K Gate Mechanisms (2026-03-26)

**Codex role:** Correctness Engineer (Tier 1). Reviewed alpha schedule, tau schedule, confidence gating code. **Verdict: CLEAN (zero HIGH, two MEDIUM).**

**MEDIUM-1:** Off-by-one at step=1 — alpha_frac=0.1674 vs desired 0.167 (0.04%). Negligible; all other boundaries exact. Not fixing.

**MEDIUM-2:** Confidence gating computes p_max over top-K (K=64) shared-vocab subset, not full teacher distribution. Overestimates teacher confidence. Thresholds (0.5/0.1) are semantically different from true teacher confidence. Acceptable for first gate — recalibrate if gating proves important.

**Validated:** Backwards compatibility confirmed. step_tau correctly wired. alpha_frac=0 correctly zeros KD loss. Schedule boundaries all correct. Config internally consistent. Schedule is local-step (1..15K), not global-step — intentional and correct.

**Optimization noted:** Teacher forward pass still runs for steps 12001-15000 (alpha=0), wasting ~20% compute. Fix: add `if (sa_state + sa_semantic + sa_logit) > 0:` guard around teacher block.

### 7. Fundamentals-First: Mathematical Structures for Knowledge Routing (2026-03-26)

**Methodology note:** These are NOT "X applied to KD" papers. These are the FUNDAMENTAL mathematical structures themselves, studied on their own terms, with connections to our KD system derived from first principles. Per user directive: study the domain deeply, then derive applications — don't search for pre-existing intersections.

#### 7.1 Optimal Transport — The Geometry of Moving Distributions

**Core idea:** OT provides the mathematically optimal way to transform one probability distribution into another with minimum cost. The Wasserstein distance measures "how much work" it takes to rearrange one distribution into another.

**Key structures:**
1. **Wasserstein distance W_p**: The infimum of expected transport cost over all couplings (joint distributions) of two marginals. Metrizes weak convergence. Unlike KL divergence, it's a true metric and is well-defined even between distributions with non-overlapping support.

2. **Wasserstein barycenter**: The distribution minimizing weighted sum of squared Wasserstein distances to N source distributions. This is the canonical "average" of multiple distributions in Wasserstein space. Computing it is NP-hard in general dimension (Altschuler et al. 2022), but entropic regularization (Sinkhorn) provides efficient O(n²/ε²) approximation.

3. **Gromov-Wasserstein (GW) distance**: Compares metric measure spaces with DIFFERENT underlying metrics. Instead of matching point-to-point (standard OT), it matches structure-to-structure — finding a coupling that preserves pairwise distances within each space. Quadratic optimization problem. Key property: invariant to isometries of each space.

4. **Multi-marginal OT (MMOT)**: Generalizes OT to simultaneously transport N≥3 distributions. Unlike pairwise OT, captures joint relationships that pairwise comparisons miss. Cost function structure critically determines tractability — pairwise decomposable costs reduce to N(N-1)/2 standard OT problems.

5. **Entropic regularization (Sinkhorn)**: Adds entropy penalty to OT formulation. Trades optimality for computational tractability. The Sinkhorn distance interpolates between OT (ε→0) and maximum entropy coupling (ε→∞). Each Sinkhorn iteration is a matrix-vector product — GPU-friendly.

**Derived connections to KD (our analysis, not from papers):**
- **Multi-teacher KD IS multi-marginal OT.** N teacher distributions + 1 student = N+1-marginal transport problem. The optimal joint coupling specifies exactly how much knowledge to take from each teacher for each input.
- **Cross-architecture KD IS Gromov-Wasserstein.** Different teachers (transformer, SSM, encoder) have different internal metric structures. GW compares structures, not positions. Our byte-span CKA is a simplified GW metric — CKA compares Gram matrices (internal structure) across different representation spaces.
- **Wasserstein barycenter as consensus signal.** For consensus windows (all teachers agree), the WB is the theoretically optimal "average" teacher signal — more principled than simple averaging or uniform weighting.
- **Sinkhorn is implementable.** Entropy-regularized OT can run on GPU in the inner training loop with modest overhead if M (span count) is small (our M=16 → 16×16 transport matrices → negligible cost).

#### 7.2 Information Geometry — The Natural Geometry of Probability Distributions

**Core idea:** Statistical manifolds (spaces of probability distributions) have a unique natural Riemannian geometry induced by the Fisher information metric. This geometry determines the "right" way to measure distances, compute averages, and project between distributions.

**Key structures:**
1. **Fisher information metric**: The UNIQUE Riemannian metric on statistical manifolds that is invariant under sufficient statistics (Amari-Chentsov theorem). At each point p, it defines how "different" nearby distributions look: g_ij = E_p[∂_i log p · ∂_j log p]. It determines the natural notion of distance between probability distributions.

2. **Dual connections (α-connections)**: Statistical manifolds carry two natural affine connections: the e-connection (exponential, α=1) and the m-connection (mixture, α=-1), which are dual with respect to the Fisher metric. Exponential families are e-flat; mixture families are m-flat.

3. **Dually flat structure**: When a manifold is flat in both e and m connections, a generalized Pythagorean theorem holds: D_KL(p||r) = D_KL(p||q) + D_KL(q||r) when q is the projection of p onto a flat submanifold. This gives "orthogonal" decompositions of divergence into independent components.

4. **I-projection and M-projection**: I-projection of p onto submanifold S: min_{q∈S} KL(q||p) — finds the closest distribution in S that is "information-preserving." M-projection: min_{q∈S} KL(p||q) — finds the closest distribution in S that is "moment-matching." These are dual operations. I-projection is forward KD (student → teacher KL). M-projection is reverse KD (teacher → student KL).

5. **Natural gradient**: The gradient in Fisher-Rao metric: ∇̃f = F^{-1} ∇f where F is the Fisher information matrix. Follows the intrinsic geometry of the parameter space, not the flat Euclidean geometry. This is why natural gradient descent converges faster — it respects the curvature of the loss landscape.

**Derived connections to KD (our analysis):**
- **The student IS a submanifold.** The student model's limited capacity restricts it to a submanifold of all possible distributions. Training IS projecting the teacher distribution onto this submanifold. The dual structure (I vs M projection) explains why forward vs reverse KL give different results in distillation.
- **Routing = selecting the information-geometric nearest teacher.** For each input, the "right" teacher is the one whose projection onto the student submanifold loses the least Fisher-Rao distance. This is more principled than comparing raw losses — it accounts for the curvature of the student's representational capacity.
- **The Pythagorean theorem predicts when routing helps.** If teacher contributions are "orthogonal" in the information-geometric sense (their projections onto the student submanifold are independent), multi-teacher KD decomposes into independent teacher projections — diversity is perfectly additive. If they're "parallel" (redundant), the second teacher adds nothing.

#### 7.3 Ensemble Theory — When and Why Combining Predictions Works

**Core idea:** The fundamental theory of why committees outperform individuals, and critically, when they DON'T.

**Key structures:**
1. **Ambiguity decomposition (Krogh-Vedelsby 1995)**: For squared loss, ensemble error = average member error - diversity. Diversity ≡ average squared deviation of members from ensemble mean. This is an identity (not an approximation). Diversity is ALWAYS non-negative → ensemble always ≤ average member. But you can't optimize diversity independently of accuracy — they're coupled.

2. **Generalized ambiguity (Brown et al. 2023, JMLR)**: Extended to classification with any twice-differentiable loss (logistic, exponential, cross-entropy). The decomposition has the same structure but with generalized divergence measures replacing squared error. For cross-entropy: diversity = average KL from member predictions to ensemble prediction.

3. **Condorcet's jury theorem**: A committee of independent agents, each with accuracy > 50%, has committee accuracy → 100% as committee size → ∞. The KEY conditions: (a) each member is better than random, (b) members are conditionally independent given the ground truth. Violation of (b) is the primary failure mode of ensembles.

4. **Oracle inequality / PAC-Bayes bounds**: The optimal convex combination of N predictors has generalization error ≤ min_i error_i + O(log(N)/n), where n is sample size. Adding more ensemble members costs only logarithmically in generalization.

5. **When ensembles FAIL**: (a) Correlated errors — if members fail together, diversity is illusory. (b) Systematic bias — if all members are biased the same way, ensemble inherits the bias. (c) "Negative diversity" — if members disagree in a way that INCREASES error (e.g., one member's correct prediction is overwhelmed by others' incorrect predictions).

**Derived connections to KD (our analysis):**
- **Multi-teacher KD obeys the ambiguity decomposition.** If we treat the teachers as an ensemble providing supervision, the KD error = average teacher KD error - teacher diversity. This predicts: multi-teacher KD helps IFF teachers are diverse AND individually better-than-random on the KD task. If teachers are redundant (e.g., two fine-tuned variants of the same base model), the second teacher adds nothing.
- **Architecture diversity IS Condorcet independence.** Transformers, SSMs, and encoders have fundamentally different inductive biases → their errors are more likely to be conditionally independent → stronger "jury" effect. This is the theoretical justification for cross-architecture distillation over same-family multi-teacher.
- **The failure modes predict our routing buckets.** Consensus = all teachers agree (high accuracy, low diversity → easy knowledge). Specialist = one teacher disagrees (high diversity, maybe high accuracy → valuable knowledge). Uncertain = all teachers disagree with each other (may be high diversity but LOW accuracy → dangerous, use cautiously).
- **"Negative diversity" predicts when routing MUST default to single teacher.** If teachers disagree and averaging their signals increases student error, routing to the single most confident teacher is better than combining.

#### 7.4 Synthesis: Novel Mechanism Candidates for Codex Review

From combining the three fundamental domains, several novel mechanisms emerge that don't exist in any published KD paper:

1. **Gromov-Wasserstein routing**: Instead of comparing teacher-student losses directly, compute the GW distance between teacher and student span Gram matrices. Route to the teacher with lowest GW distance (most structurally compatible with student's current representation). This is what CKA approximates — but GW provides the theoretically optimal structural comparison.

2. **Information-geometric projection routing**: For each input, compute the Fisher-Rao distance from student distribution to each teacher's projection onto the student submanifold. Route to the teacher whose projection preserves the most information. This accounts for the curvature of the student's capacity limitations.

3. **Wasserstein barycenter consensus**: For consensus windows, compute the entropic Wasserstein barycenter of all teacher representations (via Sinkhorn on the M=16 span Gram matrices). This is more principled than averaging and naturally handles teachers with different representation geometries.

4. **Ambiguity-aware scheduling**: Track teacher diversity (average KL from member predictions to ensemble prediction) over time. When diversity is high and individual accuracy is high → route aggressively. When diversity is high but accuracy is low → default to consensus-only. When diversity is low → single best teacher suffices.

5. **Multi-marginal transport loss**: Replace the sum of pairwise KD losses with a multi-marginal OT loss that finds the joint-optimal transport from all teachers to student simultaneously. Captures cross-teacher dependencies that pairwise losses miss.

**Status:** These are DERIVED mechanisms from fundamentals. NOT yet validated. Requires Codex evaluation for feasibility at our scale (90M student, M=16 spans, 3 teachers) before any implementation.

#### 7.5 Rate-Distortion Theory & Information Bottleneck — The Fundamental Limits of Knowledge Compression

**Core idea:** Rate-distortion theory (Shannon 1959) defines the minimum bit-rate R(D) needed to encode a source with distortion ≤ D. The information bottleneck (Tishby 1999) applies this to representation learning: find compressed T minimizing I(T;X) while maximizing I(T;Y).

**Key structures:**

1. **Rate-distortion function R(D)**: For source X and distortion d, R(D) = min_{p(t|x): E[d(x,t)]≤D} I(X;T). Monotonically decreasing — more distortion tolerance = lower rate needed. Convex. Parametric solution via Lagrange multiplier β: p(t|x) ∝ p(t)·exp(-β·d(x,t)).

2. **Information bottleneck (IB)**: min I(T;X) - β·I(T;Y). When β→0, maximum compression (discard everything). When β→∞, preserve all relevant information. The IB curve traces optimal compression-relevance tradeoffs.

3. **Successive refinement (Equitz-Cover 1991)**: Two-stage lossy encoding — coarse description first, then refinement. A source is "successively refinable" if two-stage encoding achieves the same R(D) as single-stage at each rate point. Gaussian sources are successively refinable for MSE distortion. Log-loss (cross-entropy) is universally successively refinable (Kostina-Verdú 2020).

4. **Multiple description coding**: N parallel descriptions (encoders) of the same source. Decoder receives any subset and must reconstruct. The rate-distortion region characterizes achievable rate-tuples for given distortion constraints. Diversity between descriptions is the key resource.

5. **Wyner-Ziv coding (source coding with side information)**: Compress X knowing decoder has correlated side information Y. The rate needed drops: R_WZ(D) ≤ R(D). The reduction depends on I(X;Y) — more informative side information = more compression.

**Derived connections to KD (our analysis):**

- **The student IS a rate-distortion system.** Student capacity = rate R, prediction error = distortion D. There exists a fundamental R(D) below which the student CANNOT achieve distortion D regardless of training. KD changes the effective source — instead of compressing raw text (high entropy), the student compresses the teacher's cleaned-up representation (lower entropy, lower R(D)).

- **KD as Wyner-Ziv coding.** The teacher's output is "side information" available at the decoder (student). This reduces the rate needed: the student needs fewer parameters to achieve the same BPT because the teacher's signal reduces the effective source entropy. This is the information-theoretic justification for why KD works with smaller students.

- **Phased RMFD IS successive refinement.** Our 4-phase approach (CE-only → single teacher → committee → consolidation) is a successive refinement code. Phase 1 = coarse description (CE). Phase 2 = first refinement (single teacher). Phase 3 = second refinement (committee adds diversity). The fundamental theorem: this is optimal IF teacher knowledge is successively refinable — and cross-entropy loss IS universally successively refinable (Kostina-Verdú 2020), providing theoretical justification for our phased design.

- **Multi-teacher KD IS multiple description coding.** Each teacher is a "description" of the ground truth from a different perspective. The student (decoder) receives all descriptions. The rate-distortion region predicts: multiple descriptions outperform single descriptions IFF descriptions are diverse AND decoder has capacity. This predicts our routing outcome — cross-architecture teachers (transformer, SSM, encoder) provide diverse descriptions, but the student needs sufficient capacity to combine them.

- **Optimal teacher selection = maximize complementary MI.** For each sample, the "right" teacher maximizes I(T_teacher; Y | T_student) — the mutual information between teacher and target CONDITIONAL on what the student already knows. This is more principled than minimizing raw loss (which favors teachers that repeat what the student can already predict).

- **Capacity saturation predicts routing kill point.** When the student's rate R reaches the fundamental R(D) limit, no additional teacher information helps — the bottleneck is student capacity, not teacher signal. This predicts WHEN to stop adding teachers and start increasing student capacity instead.

#### 7.6 Codex Architecture Theorist Evaluation: Fundamentals-Derived Mechanisms (2026-03-26)

**Source:** Codex GPT-o4-mini-high, Architecture Theorist persona, evaluating §7.4 novel mechanisms.

**Compute context:** Student training ~13.8 TFLOPs/step, current KD ~3.6 GFLOPs/step, step time ~4-6s with 3 online teachers. Any mechanism on 16×16 span geometry is cheap; anything needing Fisher/Jacobian is not.

**Verdicts:**

1. **Ambiguity-aware scheduling** — PRIORITY 1 (implement now). ~Free overhead (<0.05 GFLOP, <5ms). Distinguishes useful disagreement from toxic disagreement. Upgrades static 4-bucket heuristic to adaptive policy: high-diversity + high-reliability → specialist routing, high-diversity + low-reliability → consensus-only or skip KD, low-diversity → single best teacher.

2. **Gromov-Wasserstein routing** — PRIORITY 2. +0.2-0.5 GFLOP, +5-80ms. Real structural coupling on 16×16 span-distance matrices vs CKA scalar. 10-20 entropic GW iterations, use lowest-GW teacher for specialist state-surface routing. Keep existing CKA/cosine losses unchanged.

3. **Wasserstein barycenter consensus** — PRIORITY 3. <0.1 GFLOP, +2-10ms. Only inside consensus bucket. 3-teacher Sinkhorn barycenter over 16 span masses. Geometry-respecting average vs plain averaging which blurs incompatible supports. Only if consensus bucket is large enough and simple averaging is measurably lossy.

4. **Multi-marginal OT loss** — PRIORITY 4 (ablation only). +1-3 GFLOP, +20-100ms. Span-level only (16^4 = 65K entries manageable). Do NOT replace main KD objective. Only revisit after 1-3 work.

5. **Info-geometric projection routing** — KILL. +15-40 TFLOPs/step (1-3 extra backward passes), +2-6GB. Not feasible at our scale. Only a degraded proxy is implementable, and that's just a heuristic weighting rule.

**Novelty assessment:** Individual mechanisms (OT, barycenters, adaptive weighting) are NOT novel in isolation. They exist in prior art. The novelty is in the **byte-span, cross-family, routed system design** — the integration of cross-tokenizer alignment via byte spans, multi-architecture teachers with surface-specific routing, and disagreement-driven adaptive scheduling. This combination doesn't exist in any published system.

**Recommended system architecture:**
```
4-Bucket Audit (existing)
  → Ambiguity-Aware Scheduling (controller, PRIORITY 1)
    → GW routing (specialist state-surface only, PRIORITY 2)
    → WB consensus (consensus bucket only, PRIORITY 3)
    → Semantic anchor (EmbeddingGemma, fixed weight, unchanged)
```

**Implementation order:** (1) Ambiguity-aware scheduling on top of 4 buckets → must beat static multi_3family at equal teacher FLOPs. (2) GW routing for specialist state-surface with stop-grad routing. (3) WB consensus only if bucket is large enough and simple averaging is measurably lossy. (4) Defer Fisher routing. (5) Kill full MMOT as mainline.
