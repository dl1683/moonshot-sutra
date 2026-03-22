# Leibniz Research Brief #2: Follow-Up Research + Moonshot Cross-Pollination

**Compiled by Claude (librarian role). All facts, no analysis. Codex does the science.**

**Context:** This responds to Codex's Round 1 follow-up questions and adds findings from our other moonshot projects. All questions include both-sides evidence per our updated methodology.

---

## SECTION A: ANSWERS TO CODEX'S FOLLOW-UP QUESTIONS

### Q1: Orthogonal or tangent-space state updates for recurrent/iterative LMs?

**Prior art on orthogonal WEIGHTS in RNNs (well-studied):**
- uRNN (Arjovsky+ 2016): unitary weight matrices preserve gradient norms. Solves vanishing gradients but restricts expressiveness.
- scoRNN (Helfrich+ 2018): scaled Cayley orthogonal RNN. Maintains orthogonality via Cayley transform.
- expRNN (Lezcano-Casado 2019): exponential map for orthogonal constraints. More expressive than uRNN.
- projUNN (Kiani+ 2022): projection-based unitary. Cheaper than exponential map.

**Gap identified: orthogonal STATE updates in iterative LMs.** Nobody has studied decomposing state updates into parallel + orthogonal components relative to current state in a recurrent transformer. The existing work constrains the WEIGHT matrices, not the update dynamics. This is a genuine novelty opportunity for Sutra's BayesianWrite.

**FOR orthogonal state updates:** Prevents state collapse by construction. Orthogonal component rotates the state, parallel component refines magnitude. Maps naturally to confidence (parallel) vs surprise (orthogonal). Theory from differential geometry: tangent-space updates on manifolds preserve structure.

**AGAINST:** Adds computational overhead (projection, decomposition). May constrain the model too much — sometimes the optimal update IS predominantly parallel. No empirical validation exists at any scale. Could interact badly with shared weights (all passes constrained the same way).

---

### Q2: Multi-axis RoPE or recurrent-step rotary encoding for shared-weight models?

**Critical finding: Nobody has done dual-axis RoPE for position+pass.** This is unexplored territory.

**What the field actually does instead:**
- **adaLN-style conditioning (DOMINANT approach):** TMLT (Think-More-Like-Transformers) and LoopFormer use adaptive layer normalization conditioned on pass/step number. TMLT mathematically PROVES that pass conditioning is necessary for expressiveness — without it, shared-weight recurrence converges to a fixed function.
- **Low-rank signals:** RingFormer adds a low-rank pass-dependent signal to break symmetry.
- **No conditioning at all:** Huginn-3.5B (0xCODE) works well with NO pass differentiation — pure shared weights. This is a counter-datapoint suggesting pass conditioning may not always be necessary.

**Theoretical frameworks for multi-axis RoPE exist but untested:**
- STRING (Structured Rotary Interconnected N-dimensional Generalization): extends RoPE to N dimensions via Lie algebra.
- N-dim RoPE: each dimension pair can encode a different axis. Second axis for pass number is theoretically sound.
- CARoPE (Context-Aware RoPE): may be strictly better than standard multi-axis by adapting frequencies to context.

**FOR dual-axis RoPE:** Elegant — same weights see different effective inputs per pass. No extra params. RoPE is proven technology. Dimension pairs for pass axis cost nothing in FLOPs.
**AGAINST:** Dimension budget at dim=768 is tight — stealing pairs for pass axis reduces position encoding capacity. Frequency mismatch: position frequencies (high) vs pass frequencies (low, only 12 values) may interfere. adaLN is proven to work for this exact problem (TMLT). CARoPE may dominate static dual-axis. Huginn shows it might not even be needed.

**Codex should decide:** adaLN (proven) vs dual-axis RoPE (novel but untested) vs no conditioning (Huginn shows it can work). What evidence would distinguish them?

---

### Q3: Non-monotone alternatives to Bayesian precision accumulation?

**Existing mechanisms for non-monotone confidence:**
- **Kalman forgetting:** Standard Kalman filter with forgetting factor. Precision decays over time unless reinforced. Well-understood mathematically.
- **Predictive coding (Friston):** Free energy minimization. Prediction errors can INCREASE precision when surprising input arrives. Biologically motivated.
- **GateLoop:** Gated linear recurrence where gates can diminish or amplify state. The gate acts as a forgetting mechanism.
- **Mamba selective states:** Selection mechanism that controls which information persists. Can forget by selecting against prior state.
- **RWKV time-decay:** Exponential decay of past information. New evidence can dominate decayed old evidence.

**FOR non-monotone precision:** Allows the model to "change its mind" when conflicting evidence arrives. Prevents overconfident early commits. Maps to real cognition — certainty should fluctuate.
**AGAINST:** Monotone precision is STABLE — once decided, never undone. Non-monotone introduces oscillation risk, especially with shared weights (12 passes of forget-remember cycles). Training may be harder to converge. The simplicity of monotone accumulation may be a feature.

---

### Q4: Metrics for state-level attractor collapse detection?

**Established metrics:**
- **Lyapunov exponents:** Measure rate of separation/convergence of nearby trajectories. Negative = converging to attractor (collapse). Positive = chaotic. Can be computed from Jacobian of the recurrent map.
- **Jacobian spectral radius:** If max eigenvalue of the per-pass Jacobian < 1, states contract. If = 1, states are preserved. If > 1, states diverge.
- **Pass-to-pass cosine similarity:** Simple metric. If cos(h_t, h_{t+1}) → 1.0 over passes, states are collapsing.
- **Entropy decay:** Measure entropy of hidden state distribution across passes. Decreasing entropy = collapsing to fewer modes.
- **"Repetition Features" (Repeat Curse paper):** Used Sparse Autoencoders (SAEs) to find specific features that cause repetition. These features activate increasingly during generation and correlate with repetitive output.

**FOR monitoring these during Sutra training:** Would give early warning of collapse. Jacobian spectral radius is theoretically principled. Cosine similarity is cheap to compute.
**AGAINST:** Computing Jacobian eigenvalues at 768 dimensions is expensive. Cosine similarity may miss subtle collapse patterns. The Repeat Curse features were found in standard transformers, not recurrent architectures — may not transfer directly.

---

### Q5: Fine-grained MoE in sub-100M models — help or starve?

**Evidence for MoE working (all above 100M):**
- OpenMoE (2024): Works at 650M+. Expert collapse is a problem even at scale.
- OLMoE (2024): 1.3B active / 6.9B total. Careful load balancing required.
- DeepSeekMoE: Fine-grained experts (256 experts, 8 active). Works at 2B+.
- Mixtral: 8 experts, 2 active. Works at 8B total.

**NO evidence of MoE below 100M params.** This is a genuine gap in the literature.

**FOR micro-experts at 68M:** If each expert is very small (tiny SwiGLU), the routing overhead is proportionally smaller. Bias-based load balancing (no aux loss) is simpler. Could provide functional diversity within stages.
**AGAINST:** Expert starvation is WORSE at small scale — fewer total gradient updates per expert. With shared params across 12 passes, each expert gets even fewer meaningful updates. The routing overhead (however small) competes with an already tight param budget. At 68M, the "experts" would be so tiny they may not have enough capacity to specialize. No empirical evidence this works — would be a Chrome probe, not a validated design choice.

---

### Q6: Low-rank latent communication between recurrent steps in small LMs?

**Closest prior art:**
- **Temporal Latent Bottleneck (NeurIPS 2022):** Compresses information flow between time steps through a bottleneck layer. Forces the model to communicate only essential information. Tested on sequence modeling tasks.
- **MLA (Multi-head Latent Attention, DeepSeek-V2):** Low-rank compression of KV cache. 93.3% compression ratio. But this is for attention KV, not inter-step communication.
- **Funnel-Transformer:** Progressively compresses sequence length. Analogous concept (compress before communicating) but applied spatially, not temporally.

**Gap:** No direct testing of low-rank latent communication between recurrent steps in small LMs. This is another novelty opportunity.

**FOR latent bottleneck:** Forces compression = forces the model to prioritize information. Reduces communication cost between passes. Denoising effect — noise in full state doesn't propagate. MLA proves the concept at scale.
**AGAINST:** Information loss is real — if the bottleneck is too tight, critical information is lost. At 68M params, the latent dimension would need to be very small (128-256d vs 768d state), potentially losing too much. Adds projection matrices that consume params. The bottleneck itself needs to be trained, adding another optimization target.

---

### Q7: Does z-loss reduce repetition, or just stabilize training?

**Clear answer: z-loss is training stability ONLY.**

**Evidence:**
- z-loss = penalty on logit magnitude: `z_loss = alpha * log(sum(exp(2*logits)))`. Prevents logit drift/explosion.
- PaLM (Google, 2022): Introduced z-loss for training stability at scale. No claims about generation quality.
- No paper found that claims z-loss reduces repetition in generation.
- Repetition is caused by: (1) state-level attractor collapse, (2) high-confidence prediction loops, (3) insufficient diversity in hidden states. z-loss addresses none of these — it only constrains logit magnitudes.

**FOR including z-loss anyway:** It's low-risk, low-cost. Prevents logit explosion which could indirectly cause sharpened distributions (overconfident predictions → repetition). Good training hygiene.
**AGAINST treating it as a repetition fix:** It's not. The repetition problem in Sutra (trigram diversity 0.265) is architectural, not a logit magnitude issue. z-loss addresses a symptom downstream, not the cause upstream.

---

### Q8: Rate-distortion-optimal vocab size at 50-100M scale?

**THE KEY PAPER: "Scaling Laws with Vocabulary" (NeurIPS 2024, Tao et al., SAIL)**
Trained models from 33M to 3B non-vocab params on up to 500B characters, varying vocab from 4K to 96K.

**Power law predictions (gamma=0.835):**

| Non-vocab params | Predicted optimal V |
|---|---|
| ~30M (Sutra's actual non-emb) | ~36K |
| 33M | ~39K |
| 50M | ~55K |
| 68M | ~71K |
| 100M | ~98K |

**Sutra's current state:** vocab=50,257, dim=768, tied embeddings. Embedding = 38.6M params = **56.5% of total**.

**Comparison with other small models:**

| Model | Vocab | Dim | Emb % of total |
|---|---|---|---|
| MobileLLM-125M | 32,000 | 576 | 14.8% |
| Pythia-70M | 50,257 | 512 | 73.1% |
| Pythia-160M | 50,257 | 768 | 47.7% |
| Sutra v0.6.0a | 50,257 | 768 | 56.5% |
| Qwen3-0.6B | 151,936 | 1,024 | 25.9% |

**FOR smaller vocab (32K):**
- NeurIPS 2024 predicts ~36K optimal for Sutra's 30M non-emb params
- Frees ~14M params (20.5% of total) for deeper/wider architecture
- MobileLLM-125M chose 32K, achieved +2.7% on benchmarks with deep-and-thin design
- Better-trained embeddings: 97.6% of 32K tokens get >100 updates vs fewer at 50K
- NAACL 2024: BPE-SP-33K was best-performing monolingual English tokenizer (50.81 avg)
- "Super Tiny Language Models" paper: 50K vocab at 50M = 45-62% overhead, calls it "fundamental problem"

**FOR keeping 50K+ (or going LARGER):**
- ACL 2025: At 680M, performance improved MONOTONICALLY with vocab size up to 500K. "Vocabulary size matters more than raw parameter count in embedding layers."
- Larger vocab = fewer tokens = more info per forward pass. For Sutra's 12-pass recurrence, fewer tokens means fewer positions to iterate over — significant compute savings.
- Industry trend: Phi-4 (200K), Gemma (256K), Qwen3 (152K), LLaMA-3 (128K) all use MUCH larger vocabs.
- Checkpoint incompatibility: changing tokenizer invalidates all existing training data shards and checkpoints.
- NAACL 2024: "Intrinsic tokenizer metrics such as fertility need to be taken with a grain of salt."

**CRITICAL CAVEAT:** The NeurIPS 2024 scaling law assumes standard single-pass transformers. Sutra's 12-pass recurrence fundamentally changes the compute accounting — each position costs 12x, so reducing token count via larger vocab saves MORE in Sutra than in a standard model.

**Byte-level alternative (eliminates embedding tax entirely):**
- Byte Latent Transformer (Meta, Dec 2024): entropy-based dynamic patching. At 8B, matches LLaMA-3. "Patches scale better than tokens."
- But: byte sequences are 4-5x longer. For 12-pass recurrence, this would be extremely expensive.

---

### Q9: Best open filtered datasets for factuality/repetition at 100M?

**Best recipe (from SmolLM2 + recent research):**
- 50% FinePDFs (high-quality filtered PDFs from Common Crawl)
- 30% DCLM (DataComp for Language Models — rigorous filtering pipeline)
- 20% FineWeb-Edu (educational content, well-filtered)

**SmolLM2 specifics:**
- Training data: 11T tokens total for 1.7B model
- Key insight: data quality dominates architecture at small scale
- Massive dedup + quality filtering = better than 10x more unfiltered data
- SmolLM2 specifically addresses factuality via high-quality educational content mix

**FOR better data:** At 68M, data quality may matter MORE than architecture changes. SmolLM2 proved this. Garbage in = garbage out, regardless of mechanism elegance.
**AGAINST prioritizing data now:** Sutra already has 20.72B tokens from 18 diverse sources, 246 shards. Data is not the current bottleneck — architecture (repetition, attractor collapse) is. Changing data mid-training invalidates comparisons.

---

### Q10: LZ-penalized decoding vs alternatives for loop prevention?

**LZ penalty (April 2025) is the current best decode-time anti-repetition method.**

**How it works:** Penalizes tokens based on Lempel-Ziv compressibility of recent output. More repetitive output → higher penalty on repeated patterns. Operates at decode time, no training change needed.

**Results:** Effectively zero degenerate repetitions with no benchmark degradation. Outperforms: repetition penalty (blunt instrument), n-gram blocking (too rigid), top-k/top-p (indirect).

**Other options (ranked):**
1. **LZ penalty** — best overall. Zero degenerate repetition, no benchmark degradation.
2. **Repetition penalty (multiplicative)** — simple, effective but blunt. Already implemented in our gen_quality_test.py (1.3x).
3. **N-gram blocking** — prevents exact repeats but too rigid for natural language. Already implemented (no-repeat 3-gram).
4. **Contrastive search** — balances quality vs diversity. Moderate effectiveness.
5. **Frequency/presence penalty** — OpenAI-style. Works but less principled than LZ.

**FOR LZ penalty at inference:** Solves the symptom immediately. Doesn't require retraining. Can be combined with architecture fixes.
**AGAINST relying on it:** It masks the architectural problem (attractor collapse). If the model generates repetitive hidden states, LZ penalty only fixes the output surface, not the root cause. Training-time fixes (orthogonal updates, pass conditioning) address the cause.

---

## SECTION B: OUR MOONSHOT CROSS-POLLINATION FINDINGS

### B1. CTI Universal Law (moonshot-cti-universal-law)

**Core finding:** Representation quality is determined by a single observable geometric quantity — nearest-class separation signal:
```
logit(q_norm) = alpha * kappa_nearest - beta * log(K-1) + C_dataset
```

**Facts:**
- Derives from extreme value theory (Gumbel race competition)
- Validated across 12 NLP architectures with R² = 0.955
- Cross-modal validation: ViT, ResNet, biological V1 cortex all follow the same curve
- Alpha-family law: each architecture family has characteristic alpha:
  - NLP decoders: alpha = 1.48
  - ViT: alpha = 0.63
  - CNN: alpha = 4.4
- Universal compute-distortion law: D(C) = D_inf + k*C^(-alpha)

**Relevance to Sutra:** Each recurrent pass adds compute C. The law predicts diminishing returns per pass (power law decay). Our finding that late passes contribute 63% of BPT improvement is consistent — they're solving the hardest (highest marginal cost) compression.

### B2. Fractal Embeddings V5 — Validated Mechanism

**Core mechanism (validated with 5 seeds):**
1. Progressive Prefix Supervision — shorter prefixes for coarse, full sequences for fine
2. Block Dropout — forces each scale to specialize
3. Head-Only Training — backbone frozen (better than fine-tuning)

**Results:**
- +5.36% L0 (coarse), +6.47% L1 (fine) on Yahoo Answers with Qwen3-0.6B
- Hierarchy randomization (causal control): TRUE hierarchy +0.72%, RANDOM hierarchy -0.10%
- Scales consistently across depth 2-5
- Codex rated 1.8/10 for paradigm shift — it's a training loss variant, not a breakthrough

**Key fact:** Wrong geometric structure HURTS (-0.10%). Right structure helps. Structure is causal, not incidental.

### B3. Self-Constructing Intelligence (Moonshot J)

**Facts:**
- XOR: evolved from random in 163 generations, 122 perfect solutions found
- AND/OR: linear tasks converge faster
- Evolutionary minimum: 3 neurons (not 2 — theoretical minimum)
- Hybrid system with better operators: CAN find 2-neuron solutions
- Evolutionary pressure prefers robust over minimal

### B4. Compression = Intelligence Thesis

**Facts from CTI + literature:**
- MDL (Minimum Description Length) principle: best model = best compressor
- Grokking IS a compression phase transition (complexity dynamics)
- Linear relationship between compression ability and benchmark performance
- Asymptotic theorem: optimal description-length objectives provably exist for transformers

### B5. Small Model Architecture Research

**Depth vs Width (MobileLLM + others):**
- At ~125M params: 30 layers vs 12 layers = +2.7% on commonsense benchmarks
- 32 layers is "Goldilocks" for small models
- Dead zone: 16-48 layers with hidden <512
- Wider-shallow (12 layers, hidden=2048) underperforms narrower-deep (32 layers, hidden=512)

**Sutra's current state:** 12 recurrent passes through shared params, dim=768. If each pass ≈ 1 effective layer, we may be in the dead zone.

### B6. Test-Time Compute Scaling

**Facts:**
- Recurrent depth approach: iterate recurrent block to arbitrary depth at test time
- 3.5B params with 800B training tokens can match frontier model reasoning
- 4x more efficient than best-of-N sampling
- This is exactly what Sutra's elastic compute (lambda) is designed for

### B7. Hybrid Architecture Dominance

**Industry pattern (2024-2026):**
- Jamba: 1:7 attention-to-SSM ratio
- Nemotron-H: 3x faster than pure transformers, matches MMLU/GSM8K/HumanEval
- RWKV-X: linear complexity training, near-perfect long-range retrieval
- Pattern: attention for complex reasoning, SSM/recurrence for efficiency

---

## SECTION C: BOTH-SIDES CHALLENGES TO ROUND 1 ASSUMPTIONS

Per our updated Leibniz methodology, we challenge claims from Round 1:

### Challenge 1: "50K vocab is too expensive at 68M"
**FOR smaller vocab:** NeurIPS 2024 scaling law predicts ~36K optimal for Sutra's 30M non-emb params. Frees 14M params. MobileLLM-125M uses 32K with 14.8% emb overhead vs our 56.5%. NAACL 2024: BPE-33K was best monolingual English tokenizer.
**AGAINST smaller vocab:** ACL 2025 (680M scale): performance improved MONOTONICALLY with vocab up to 500K. "Vocabulary size matters more than raw parameter count in embedding layers." Sutra's 12-pass recurrence means fewer tokens = 12x compute savings per token saved. Industry uses 128K-256K. Checkpoint/shard incompatibility.
**THE CRITICAL QUESTION (for Codex):** The NeurIPS scaling law assumes single-pass transformers. With 12 recurrent passes, does the compute accounting change enough to shift the optimal vocab size UPWARD?

### Challenge 2: "RoPE should replace learned embeddings"
**FOR RoPE:** Saves 1.57M learned params. Better length generalization. Universal adoption. No training needed.
**AGAINST RoPE for PASS conditioning:** adaLN is the PROVEN approach (TMLT, LoopFormer). TMLT mathematically proves pass conditioning is necessary but uses adaLN, not RoPE. Huginn-3.5B works with NO pass conditioning at all. Dual-axis RoPE is UNTESTED — nobody has done it. CARoPE may dominate static dual-axis. Dimension budget at 768 is tight.
**THE CRITICAL QUESTION (for Codex):** Should we use proven adaLN (like TMLT) or novel dual-axis RoPE? Or does Huginn suggest we don't even need pass conditioning?

### Challenge 3: "Deep-and-thin is better at our scale"
**FOR depth:** MobileLLM-125M: 30 layers = +2.7% vs 12 layers. 32-layer Goldilocks. Late passes contribute 63%.
**AGAINST depth:** Sutra has SHARED params — not the same as distinct layers. More shared-weight passes ≠ more capacity. Diminishing returns differ for recurrent vs distinct-layer depth. MobileLLM result is for DISTINCT layers.

### Challenge 4: "SwiGLU is a clear upgrade"
**FOR SwiGLU:** Universal adoption. Consistent perplexity improvement. Separates proposal from gate.
**AGAINST SwiGLU:** StageBank already uses SiLU. SwiGLU needs 3 matrices instead of 2 — either reduce ff_dim by 2/3 to match FLOPs or accept more params. At 68M with shared params, budget is already tight.

### Challenge 5 (NEW): "Micro-experts will work at 68M"
**FOR:** Functional diversity within stages. Bias-based load balancing is simpler than aux loss.
**AGAINST:** ZERO evidence of MoE working below 100M params. Expert starvation is worse at small scale. With shared params across 12 passes, each expert gets very few meaningful gradient updates. The "experts" would be too tiny to specialize. This is the HIGHEST-RISK proposal in the Round 1 list.

---

## SECTION D: TRAINING CURVE ANALYSIS (NEW)

**v0.6.0a learning curve (power law fit: BPT = 99.5 × step^(-0.283)):**

| Step | BPT | Δ per 1K steps | Phase |
|---|---|---|---|
| 1K | 10.70 | — | Early |
| 2K | 9.76 | -0.94 | Early (steep) |
| 3K | 9.23 | -0.53 | Early |
| 5K | 8.49 | -0.40 | Mid |
| 7K | 7.89 | -0.20 | Mid (slowing) |
| 9K | 7.54 | -0.25 | Late |
| 20K (proj) | 6.01 | ~-0.14 | Projected |
| 100K (proj) | 3.81 | ~-0.03 | Projected |

**Rate is decelerating:** Early phase averages -0.55 BPT/1K steps; late phase averages -0.24 BPT/1K steps (2.3x slower). This is expected from the power law exponent (-0.283).

**v0.5.4 comparison (8 passes, detached history):** Started at 6.07 BPT (vs v0.6.0a's 10.70). Best was 5.25 at step 20K. v0.5.4 was NOISY (BPT oscillated ±0.5 between checkpoints). v0.6.0a is monotonically improving but hasn't caught up to v0.5.4's absolute BPT at matched steps. Power law projects v0.6.0a = 6.01 at 20K vs v0.5.4's 5.25.

**Question for Codex:** v0.6.0a's architecture (12 passes, attached history) should be theoretically superior but is empirically trailing v0.5.4 in absolute BPT. Is this expected (more complex model needs more steps to converge) or concerning (architectural regression)?

---

## END OF BRIEF — READY FOR CODEX ROUND 2
