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

---

## Prior Experiments (Summary — No Checkpoints Retained)

We ran several experiments at 98M scale with a 12-layer dense decoder (d=768, 16K tokenizer, fixed exits at layers 4/8/12). All checkpoints have been deleted. Key observations:

- **Plain AR control outperformed all auxiliary modules** we tested (MTP D=1, learned halting, n-gram memory fusion). However, we discovered multiple implementation bugs during this period, so these failures may reflect our bugs rather than fundamental limitations of the approaches.
- **Post-hoc exit calibration showed promise** for inference efficiency (entropy thresholds on fixed exits).
- **WSD learning rate schedule** worked well for training dynamics.
- **16K custom tokenizer** was consistently the biggest single win.
- **The model was severely undertrained** — at step 5000 (~82M tokens) it was at only 4% of Chinchilla-optimal (~2B tokens for 98M params).

**These observations inform but do not constrain future architecture decisions.** The next T+L session should treat the architecture as an open question.
5. **T+L Round 4** — after step 10000 benchmarks and A/B results
