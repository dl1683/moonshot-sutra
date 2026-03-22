# Leibniz Research Brief #1: Architectural Innovations for Sutra v0.7.0

**Compiled by Claude (librarian role). All facts, no analysis. Codex does the science.**

---

## SECTION A: SUTRA CURRENT STATE (for context)

- 68.3M params, 12 recurrent passes, shared parameters across passes
- StageBank: 7 expert FFNs (768→1536→768 each, GELU activation), routing via SwitchingKernel
- LocalRouter: Q/K/V projections with window=4 local attention, k=8 retrieval
- BayesianWrite: precision-weighted state update `mu_new = (lam*mu + alpha*m)/(lam+alpha)`
- Scratchpad: 8-slot external memory with read/write projections
- Pheromone: cross-position information accumulator (decays with rho=0.90)
- Loss: L_final (full-vocab CE) + 0.25*L_step (sampled CE, 32 candidates) + 0.20*L_probe (shadow gain prediction)
- Optimizer: AdamW (lr=3.5e-4, betas=0.9/0.95, wd=0.01, clip=0.5)
- Schedule: cosine with 1500-step warmup
- Normalization: GatedLayerNorm (pre- and post- on bank, route, write)
- Positional: learned embeddings (2048 × 768 = 1.57M params)
- Activation: GELU in stage bank FFNs
- Tokenizer: GPT-2 BPE (vocab 50,257)
- Training: bf16, seq_len=512, batch=4×16=64 effective

Step 9200, BPT 7.54 (improving). 20B tokens seen of 20.72B available.

### Critical findings from Chrome probes:
- L_step Goldilocks: sampled CE (32 candidates) is the right intermediate loss. Full-vocab CE = catastrophic. No loss = degrading.
- Late passes (7-11) contribute 63% of BPT improvement
- Generation quality at 9K: severe repetition (trigram diversity 0.265), no factual knowledge
- Pheromone: killed at dim=768 (0% effect)
- Grokfast: killed (overfits)
- Resonant Write Dither: killed (0% effect)
- Reversible Writer: inconclusive (gates collapsed)

---

## SECTION B: OPTIMIZER INNOVATIONS

### B1. Muon Optimizer (Keller Jordan, Oct 2024)
- Replaces AdamW for 2D weight matrices in hidden layers
- Mechanism: standard momentum update M_t, then replace with polar(M_t) = UV^T from SVD
- Uses Newton-Schulz iteration instead of SVD: X_{k+1} = X_k(aI + bX_k^T X_k + c(X_k^T X_k)^2), coefficients (3.4445, -4.7750, 2.0315), 5-10 iterations
- Memory: 4 bytes/param (momentum only) vs AdamW's 8 bytes/param (first+second moment)
- Results: 1.35x speedup over AdamW on NanoGPT/FineWeb. 10-15% fewer tokens to same loss at 100M scale
- Now in PyTorch core (torch.optim.Muon as of 2.10)
- LR for Muon is ~0.02 (much higher than AdamW's 3.5e-4)
- Only applies to 2D params. Embeddings, biases, LN must use AdamW
- Validated at 100M-4B scale. Originally developed on NanoGPT-scale (124M)
- Note: at 130M with 16x Chinchilla ratio, NAdamW and SOAP can match Muon

### B2. MuonClip (Moonshot AI / Kimi K2, July 2025)
- Muon + QK-Clip stabilization
- QK-Clip: after each Muon step, compute max QK attention score. If > tau (100): rescale W_q by eta^alpha, W_k by eta^(1-alpha) where eta = tau/max_score, alpha=0.5
- Clips at weight level, not output level — preserves gradient flow
- Used to train Kimi K2 (1T params) on 15.5T tokens with zero loss spikes
- Kimi recipe: LR 2e-4 constant for 10T tokens, cosine decay to 2e-5 for 5.5T

### B3. Other Optimizers
- SOAP (Shampoo+Adam hybrid): 1.4x speedup at 130M, advantage shrinks with scale
- Sophia: 2x initial reports, less community adoption, Hessian estimation overhead
- Lion: fastest initial convergence, sign-based minimal memory, falls behind later
- Schedule-Free AdamW: eliminates LR scheduling, less validated at small scale

---

## SECTION C: ARCHITECTURE MICRO-DECISIONS (2024-2026 CONSENSUS)

### C1. SwiGLU Activation
- FFN: out = (x @ W_gate) * SiLU(x @ W_up) then @ W_down
- 3 weight matrices instead of 2; reduce ff_dim by 2/3 to match FLOPs
- Used by: LLaMA, Mistral, PaLM, Qwen3, OLMo 2, Gemma 3, MobileLLM
- Consistent perplexity improvement over GELU/ReLU across all scales

### C2. RMSNorm + QK-Norm
- RMSNorm replaces LayerNorm (no mean subtraction, just variance normalization)
- QK-Norm: RMSNorm on Q and K before attention dot product
- OLMo 2: post-norm + QK-norm dramatically improves stability, enables higher LR
- Prevents attention logit explosion
- Used by: OLMo 2, Gemma 3, Qwen 3

### C3. RoPE (Rotary Position Embeddings)
- Apply rotation matrices to Q and K based on position
- No learnable parameters (saves 1.57M params in our case)
- Better length generalization than learned positional embeddings
- Base theta=10000 (standard), 500000 for long context
- Used by: LLaMA, Qwen3, OLMo 2, Mistral, Gemma 3 (everything modern)

### C4. Deep-and-Thin Architecture (MobileLLM, ICML 2024)
- At sub-billion: depth > width for same param count
- MobileLLM-125M: 30 layers achieved +2.7% on commonsense vs 12 layers same params
- Below 10 layers: poor reasoning. 20+ layers: significant improvement
- Specifically validated at 125M-350M scale

### C5. Grouped Query Attention (GQA)
- 4:1 to 8:1 query-to-KV head ratio
- Reduces KV cache size. Minimal quality loss vs MHA
- Used by: LLaMA 2/3, Gemma 3, Phi-4-mini, MobileLLM, Qwen3

### C6. Weight Tying
- Tie input/output embeddings. Saves 10-20% params at small scale
- MobileLLM: only 0.2% accuracy drop with 10% param reduction
- Already used by Sutra

---

## SECTION D: TRAINING RECIPE INNOVATIONS

### D1. WSD Schedule (Warmup-Stable-Decay)
- (1) Warmup: 2000 steps to peak LR. (2) Stable: constant peak LR for most of training. (3) Decay: linear decay to 0 over final 10-20%
- Often beats cosine final loss
- Key advantage: stable checkpoint can branch into multiple decay experiments
- SmolLM2-135M/360M specifically validated with WSD (20% decay phase)

### D2. Data Strategy (SmolLM2)
- Single-stage training with consistently high-quality data outperformed multi-stage curriculum for 135M/360M
- Filter DCLM with FineWeb-Edu classifier, remove score-0, downsample scores 1-2
- SmolLM2 biggest gains came from DATA, not architecture (dense 1.7B beats Qwen2.5-3B through 36T tokens of cleaner data)

### D3. Multi-Token Prediction / MTP (Meta, adopted by DeepSeek-V3)
- Predict next K tokens simultaneously using sequential prediction heads
- DeepSeek: MTP lambda=0.3 for first 10T tokens, lambda=0.1 for remaining 4.8T
- MTP heads discarded at inference — zero inference cost
- Meta found MTP only helps above ~1B params. Likely negative at 68M
- The SCHEDULING insight (decay auxiliary loss weight over training) is interesting

### D4. Sequence Packing
- Pack multiple short sequences into context with attention masking
- Eliminates up to 50% padding waste

### D5. muP (Maximal Update Parameterization)
- Find hyperparameters on small proxy, transfer to target scale
- Reduces HP tuning cost by ~93%
- Relevant if planning scale-up (tune at 10M, transfer to 68M, then 200M+)

---

## SECTION E: LOSS & REGULARIZATION

### E1. z-loss
- Penalty on logit magnitude: z_loss = 1e-4 * log(sum(exp(2*logits)))
- Prevents logit drift and numerical instability
- Used in PaLM, Gemma. Zero downside

### E2. Label Smoothing: DO NOT USE in pretraining
- ICLR 2025: label smoothing during pretraining harms transfer learning

### E3. Zero Dropout (confirmed)
- ACL 2025: specifically validated on Pythia 160M
- Single-epoch pretraining cannot overfit — dropout regularization is harmful
- All major 2024-2025 models train with zero dropout

### E4. Gradient Clipping: 1.0
- Universal default. Sutra uses 0.5 (more aggressive than standard)

---

## SECTION F: CHINESE LAB INNOVATIONS

### F1. Multi-Head Latent Attention / MLA (DeepSeek-V2, May 2024)
- Compresses KV into shared low-rank latent: c_KV = x @ W_DKV (d_model → d_c where d_c << n_heads * d_head)
- At attention time: K = c_KV @ W_UK, V = c_KV @ W_UV (up-project on the fly)
- Decoupled RoPE: positional encoding handled separately to avoid corrupting compressed latent
- 93.3% KV cache reduction. 5.76x generation throughput increase
- DeepSeek-V2: d_c = 512 vs original 14K dimensions

### F2. DeepSeekMoE Fine-Grained Experts (Jan 2024)
- Instead of N large experts, use mN smaller experts: mN experts of size d_ff/m, activate mK
- Same compute per token, dramatically more expert combinations
- 2 shared experts (always active) + 64 routed experts (each 0.25x standard FFN), 6 routed activated
- Combinatorial explosion in possible expert configurations

### F3. Auxiliary-Loss-Free Load Balancing (DeepSeek-V3, Dec 2024)
- Replace load balance aux loss with dynamic bias: b_i added to routing score
- After each step: overloaded → decrease b_i by gamma; underloaded → increase by gamma
- gamma=0.001 during training, 0.0 for final annealing
- Achieves balance without degrading routing signal (aux loss causes ~0.5-1% quality drop)

### F4. FP8 Mixed Precision Training (DeepSeek-V3)
- FP8 for all GEMMs, higher precision for everything else
- Block quantization: weights per 128×128 block, activations per 1×128 tile
- Requires dims divisible by 128 for clean block boundaries

### F5. Lightning Attention (MiniMax, 2025)
- Hybrid: intra-block quadratic attention + inter-block linear accumulation
- O_intra = [(Q K^T) * M] V (standard within block)
- O_inter = Q * KV_{prev} (linear rolling accumulator)
- Complexity: O(nd^2 + nBd) instead of O(n^2 d)
- MiniMax architecture: 7 lightning attention : 1 softmax attention ratio

### F6. Native Sparse Attention / NSA (DeepSeek, Feb 2025)
- Three parallel branches: compression (MLP compresses tokens to block reps), selection (select top blocks by attention score), sliding window (local)
- All three combined. Hardware-aligned (contiguous memory access)
- Natively trainable end-to-end

### F7. Qwen3 Architecture Notes
- Dense models: standard transformer, nothing exotic
- Qwen3-1.7B matches Qwen2.5-3B through DATA (36T tokens), not architecture
- MoE: no shared experts (simpler than DeepSeek), 128 experts / 8 active
- Global-batch load balancing loss (balance at global level, not micro-batch)

---

## SECTION G: OUR OWN MOONSHOT RESULTS

### G1. Fractal Embeddings (Moonshot, validated)
- Hierarchy-aligned progressive prefix supervision
- Shorter prefixes for coarse categories, full embeddings for fine
- Block dropout forces scale specialization
- +5.36% L0, +6.47% L1 on Qwen3-0.6B. Validated with 5 seeds
- Hierarchy randomization: TRUE hierarchy helps, RANDOM hierarchy hurts
- Discovery: larger models benefit MORE from fractal structure
- Head-only training outperforms backbone fine-tuning
- Core principle: geometric structure in embedding space creates efficiency gains

### G2. Latent Space Reasoning (Moonshot)
- Random tokens can unlock hidden capabilities in pretrained models
- Relates to scratchpad/memory design: external tokens provide "reasoning space"
- Implies: giving the model explicit scratch space (like Sutra's scratchpad) has theoretical grounding

### G3. Self-Constructing Intelligence (Moonshot J)
- Networks can emerge from random initialization through pure evolution
- XOR, AND, OR solved through evolution (no training)
- Evolutionary minimum (3 neurons) differs from theoretical minimum (2 neurons)
- Hybrid system with better operators finds 2-neuron solutions

### G4. CTI Universal Law (Moonshot)
- Universal compute-distortion law: D(C) = D_inf + k*C^(-alpha)
- THE manifesto in equation form: same curve regardless of architecture
- Relates to Sutra's recurrent passes: each pass adds compute, reducing distortion

---

## SECTION H: TOKENIZER CONSIDERATIONS

### H1. Vocab Size Scaling
- NeurIPS 2024: optimal vocab scales with model size — smaller models need smaller vocabs
- At 68M params with vocab=50,257: embedding = 38.6M params (56.5% of model!)
- At 68M params with vocab=32,000: embedding = 24.6M params (36%)
- Freed params (14M) could go to more layers or wider hidden dim

### H2. Current Landscape
| Model | Vocab Size |
|-------|-----------|
| SmolLM2 | 49,152 |
| MobileLLM | 32,000 |
| Gemma 3 | 262,144 |
| Phi-4-mini | 200,064 |
| LLaMA 3 | 128,256 |
| GPT-2 (Sutra) | 50,257 |

---

## END OF BRIEF

This brief is raw research. Codex: figure out what drives these innovations, what principles connect them, what novel mechanisms we could derive for Sutra's specific constraints (12-pass recurrent, shared params, 68M scale), and what follow-up questions need answers.
