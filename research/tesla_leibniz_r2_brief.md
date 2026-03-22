# Tesla+Leibniz Round 2 Research Brief

This document compiles all research findings, probe results, and benchmark data
gathered between Round 1 and Round 2. Every finding here was generated in response
to Round 1's specific research and probe requests.

---

## Research Request 1: Recursive LMs 100M-350M — Concrete Recipes

Round 1 asked: "What successful recursive/shared-weight LMs exist in the 100M-350M range?
How much is shared, what pass identity, how is depth sampled, is exact memory present?"

### Findings: 14-Model Comparison Table

| Model | Params | Shared Fraction | Pass Identity | Depth Sampling | Exact Memory |
|-------|--------|----------------|---------------|----------------|--------------|
| **Universal Transformer** | Various | 100% shared | Timestep embed | Fixed depth | None |
| **PonderNet** | Various | 100% shared | Step embedding | Learned halting (geometric prior) | None |
| **AdaPonderLM** | ~125M | 100% shared | Step embedding | Halting network per token | None |
| **ITT-162M** | 162M | 100% shared core | Iteration embedding | Fixed depth (trained variable) | None |
| **Huginn-0125 3.5B** (reference) | 3.5B | 99.6% shared | None (pure shared) | Fixed 40 passes | None |
| **MoR (Mixture of Rounds)** | Various | 100% shared + per-round MoE | Round-specific expert selection | Fixed depth | None |
| **LoopFormer** | Various | 100% shared | Position in loop | Fixed depth | None |
| **SpiralFormer** | Various | 100% shared | Spiral position embed | Fixed depth | None |
| **ALBERT** | 12M-235M | 100% cross-layer sharing | Layer index embed | Fixed depth (N/A) | None |
| **Block Recurrent Transformer** | Various | Partial (recurrent block shared) | Block position | Fixed depth | Recurrent state acts as memory |
| **Feedback Transformer** | Various | Partial (feedback connections) | Layer index | Fixed depth | None |
| **Latent Reasoning** | 70M-350M | Core shared, router per-pass | Pass-conditioned norm | Random depth (deep-biased) | None |
| **Ouro** | Various | Mostly shared + lightweight adapters | AdaLN (pass-conditioned) | Fixed depth, eval-time variable | None |
| **ARMT** | ~160M | Shared + associative memory | Segment position | Fixed segments | Yes — associative read/write |

### Key Patterns Across Successful Models:
1. **Pass identity is nearly universal** — only Huginn omits it, and Huginn is 3.5B (20x our scale). At <200M, every successful model uses some form of pass/step/round embedding.
2. **Random-depth training is rare but powerful** — Latent Reasoning is the only model that explicitly samples depth during training. PonderNet/AdaPonderLM learn halting, which is different. Most use fixed depth.
3. **Exact memory is extremely rare** — only ARMT and Block Recurrent Transformer have anything resembling exact recall. This is a major gap in the field.
4. **Lightweight per-pass adapters beat pure sharing** — Ouro (AdaLN), MoR (per-round experts), Latent Reasoning (pass-conditioned norm) all outperform pure weight sharing at comparable scales.

---

## Research Request 2: Intermediate Objectives for Recurrence

Round 1 asked: "What keeps intermediate states useful for later passes without forcing
them to become premature output heads?"

### Findings: Top 5 Intermediate Objectives Ranked

1. **Seq-VCR (Sequential Visual Conditioning with Residuals)** — adapted for language: each pass predicts the RESIDUAL improvement, not the full output. Intermediate passes optimize for "how much did I improve the representation?" not "can I predict the next token?" This avoids the premature-output-head problem entirely. Evidence: ITT-style models show that residual prediction naturally distributes work across passes.

2. **Shortcut-Consistency (Self-Distillation)** — the final pass's logits become the soft target for intermediate passes. Each pass tries to be consistent with the eventual output, but the target is the model's own deeper computation, not ground truth. This creates a curriculum: early passes learn what the full model eventually settles on, without being forced to independently solve the full problem. Evidence: used in Universal Transformer variants, Consistent Training for depth-adaptive models.

3. **Self-Distillation from Detached Final Pass** — similar to shortcut-consistency but with stop-gradient on the teacher. The final pass's output is treated as a fixed target (detached from the graph). Intermediate passes minimize KL-divergence to this detached target. This prevents gradient collapse where all passes co-optimize and converge to the same function.

4. **Mutual Information Maximization** — maximize MI between intermediate representations and a useful signal (e.g., future tokens, semantic categories) without requiring token prediction. Each pass is rewarded for encoding information that WILL be useful, not for predicting outputs NOW. Evidence: information-theoretic objectives in contrastive learning show this can guide representation quality without task-specific heads.

5. **Progressive Difficulty Curriculum** — don't apply intermediate loss on ALL tokens. Apply it only on tokens that the model can reasonably handle at that depth. Easy tokens get supervised at early passes, hard tokens only at late passes. This respects the natural difficulty gradient of recurrence. Evidence: PonderNet's halting mechanism implicitly does this; explicit curricula have been used in depth-adaptive training.

### Critical Warning:
The repo already discovered (L_step_exact 3-arm probe) that full-vocab intermediate CE is **catastrophically bad** — it forces intermediate passes to become premature output heads and destroys late-pass contribution. Any intermediate objective MUST be softer than full token prediction. The top 3 above all satisfy this constraint.

---

## Research Request 3: Function-Preserving Growth Methods

Round 1 asked: "How do we grow/warm-start causal LMs through architectural edits
without losing already-learned knowledge?"

### Findings: Validated Methods

1. **WSD Schedule (MiniCPM)** — Warmup-Stable-Decay learning rate schedule. Key insight: the "stable" phase can be interrupted for architectural changes, then training resumes with a new warmup. MiniCPM showed this works for growing model width and depth. The decay phase should happen only at the very end of training.

2. **Zero-Init Birth (Net2DeeperNet)** — add new components initialized so they have ZERO influence on the existing computation. For a new layer: init as identity function. For a new adapter: init output projection to zeros. The model starts training with the new capacity available but not yet active, then gradually learns to use it. Evidence: widely used in progressive growing (ProGAN, GPT-style layer insertion). Critical for warm-start compatibility.

3. **Net2WiderNet** — widen layers by duplicating neurons and halving their output weights. Preserves the function exactly (same forward pass before and after). Can increase hidden dimension, add attention heads, or grow FFN width. Evidence: validated for transformers, ResNets. The key constraint: only works for width changes, not for fundamentally different architecture components.

4. **LoRA-Style Growth** — add low-rank adapters alongside frozen pretrained weights. The adapter starts at zero (via zero-init of one matrix). Training updates only the adapter. Later, the adapter can be merged into the base weights. Evidence: LoRA, QLoRA, DoRA all show this preserves base model capability while adding new capacity.

5. **Selective Weight Transfer** — for components that are architecturally compatible (e.g., same-dimension embeddings, same-shape projections), directly copy weights. For incompatible components, use zero-init birth. This allows partial warm-start even across architecture changes. Evidence: used in Phi model family, Llama training continuations.

### Key Insight for Sutra:
The safest warm-start path is: (1) keep architecturally compatible components (embeddings, shared content block if dimensions match), (2) zero-init any new components (adapters, new routing, new memory), (3) use WSD schedule with fresh warmup after the architectural change. This means the v0.6.0a → v0.7.0 transition CAN preserve learned knowledge if the shared core dimensions are preserved.

---

## Research Request 4: Exact-Recall Mechanisms for Recurrent Models

Round 1 asked: "What exact-recall mechanisms work with recurrent models on one GPU,
without collapsing into full dense attention?"

### Findings: Top Candidates

1. **ARMT (Associative Recurrent Memory Transformer)** — uses an associative memory matrix that supports exact key-value write and read operations. Keys are written as outer products; reads use inner products for exact matching. Memory scales O(d^2) not O(T). Validated on tasks requiring exact recall over 10K+ tokens. Compatible with recurrence because the memory is a fixed-size state, not growing with sequence length. **Top candidate for Sutra.**

2. **FwPKM (Fast Weight Programmers with Product-Key Memory)** — combines fast weight matrices (delta-rule updates) with product-key addressing for efficient lookup. The product-key mechanism enables O(sqrt(N)) lookup instead of O(N). Fast weights provide exact overwrite capability. Evidence: competitive with Transformer-XL on language modeling while being more memory-efficient.

3. **Zoology Finding: 82% Recall Gap** — a systematic study (Zoology, ICLR 2024) showed that SSMs and recurrent models have an **82% gap** vs. attention on exact-recall tasks (associative recall, phonebook lookup). This gap is NOT closed by scaling — larger recurrent models don't fix it. The gap IS closed by adding explicit memory mechanisms (like ARMT-style associative memory). This validates the Sutra scratchpad as architecturally necessary, not just a nice-to-have.

4. **Linear Attention with Decay** — RetNet, GLA, and HGRN2 use linear attention with exponential decay factors. These approximate exact recall for recent tokens but degrade for distant ones. Not truly "exact" but much cheaper than full attention. Could complement an exact-recall mechanism for different temporal scales.

5. **Hybrid Approaches** — several successful models (Griffin, Jamba, RecurrentGemma) combine a recurrent backbone with sparse attention windows or explicit memory for exact recall. The consensus: recurrence for general processing, explicit memory for exact recall. These are complementary, not competing.

### Key Insight for Sutra:
The scratchpad is architecturally validated by external evidence. The 82% recall gap means
recurrence CANNOT solve exact recall alone — an explicit memory mechanism is mandatory.
The current 8-slot scratchpad is a reasonable first draft. ARMT-style associative memory
or product-key memory could be a more principled replacement. The critical question is
whether the memory should be fixed-size (current scratchpad) or content-addressable (ARMT).

---

## Research Request 5: Tokenizer & Embedding Tax for Sub-200M Models

Round 1 asked: "Do smaller or hierarchical vocabularies improve total intelligence per
total parameter once recurrent depth is in the picture?"

### Findings: The Embedding Tax Is Severe

**Current Sutra situation:** 68.3M total params, GPT-2 tokenizer (50,257 vocab), dim 768.
Embeddings consume 40.2M params (58.8% of total). Only 28.1M params for recurrent compute.
With 12 passes, effective compute reuse is 28.1M x 12 = 337.2M, but embeddings are used only 1x.

**Comparison with successful sub-200M models:**

| Model | Total | Vocab | Dim | Embed Params | Embed % | Tied? |
|-------|-------|-------|-----|-------------|---------|-------|
| Pythia-70M | 70M | 50,304 | 512 | 51.5M (untied) | 73.6% | No |
| SmolLM2-135M | 135M | 49,152 | 576 | 28.3M | 21.0% | Yes |
| MobileLLM-125M | 125M | 32,000 | 512 | 16.4M | 13.1% | Yes |
| **Sutra v0.6.0a** | **68.3M** | **50,257** | **768** | **40.2M** | **58.8%** | **Yes** |

**Key finding:** Sutra has the WORST embedding-to-compute ratio of any model in this comparison.
MobileLLM achieves 13.1% by using vocab 32K with dim 512. SmolLM2 achieves 21% with dim 576.
Sutra's combination of large vocab (50K) and large dim (768) is extremely expensive.

**Scaling law research (Tao et al., NeurIPS 2024):** Optimal vocab size scales with model size,
but SLOWER than non-embedding params. A 68M model is dramatically over-vocabularied at 50K.
Rule of thumb: embedding params should be ~20% of total model parameters.

### Mitigation Options (ranked by impact):

| Option | Embed Params | Embed % | Freed Params | Seq Length Impact |
|--------|-------------|---------|-------------|-------------------|
| **ALBERT-style factored (E=128)** | 8.1M | 11.9% | **32.1M** | None (same tokenizer) |
| Vocab 16K, tied | 13.9M | 20.3% | 26.3M | ~1.3x longer |
| Vocab 32K (MobileLLM-style) | 26.2M | 38.3% | 14.0M | ~1.1x longer |
| Byte-level (vocab 256) | ~2M | 2.9% | ~38M | ~3-4x longer |

**Top recommendation: ALBERT-style factored embedding (V x E + E x H where E=128).**
- Preserves the existing GPT-2 tokenizer — no data pipeline changes, no checkpoint surgery
- Frees 32.1M params — more than DOUBLING the compute core budget
- Zero sequence length impact
- Mathematically justified: input embeddings are context-independent and genuinely low-rank
- Used successfully in ALBERT, mBERT variants, and several sub-200M models

**Over-Tokenized Transformer finding (ICML 2025):** Larger input vocab HELPS but larger output
vocab HURTS at small scale. Suggests asymmetric strategy: large factored input vocab (cheap via
n-gram composition), small output vocab. A 400M model with 128x input vocab matched a 1B baseline.

**Recurrence makes the tax WORSE:** In a standard 12-layer transformer, each layer has unique weights
(~28M compute params used once each = 28M effective). In Sutra, 28M compute params reused 12x = 337M
effective compute. The embedding "dead weight" is proportionally 12x worse than in a comparable
standard transformer.

---

## Probe E: Pass Disagreement as Intrinsic Uncertainty

Round 1 requested: "Compute per-token disagreement features across passes and correlate with
future-gain, hard-token classes, and repetition."

### Results (CPU probe, step 12K checkpoint, 20,480 tokens):

**Best feature: `ce_spread` (cross-entropy spread across passes)**
- Correlation with future_gain: **r = +0.959** (near-perfect positive)
- Correlation with final_ce: **r = -0.873** (strong negative — high disagreement = easy final prediction)
- High-disagreement tokens (Q4) get **4.8x the benefit** from additional passes vs low-disagreement (Q1)

**Phase transition at pass 10-11:**
- Cosine similarity between consecutive passes drops from 0.94 (pass 9→10) to **0.258** (pass 10→11)
- KL divergence spikes from 0.15 (pass 9→10) to **5.1** (pass 10→11)
- This confirms the "deferred computation" pattern: passes 0-9 are near-equilibrium, pass 11 performs a massive rewrite

**Per-pass CE trajectory:**
- Pass 0: 13.25, Pass 5: 12.81, Pass 9: 11.28, Pass 10: 9.94, **Pass 11: 4.73**
- 63% of total CE reduction happens in the last 2 passes

**Interpretation and alternatives:**
- Obvious: "ce_spread is a near-perfect elastic compute signal — high spread = more passes needed"
- Alternative 1: "ce_spread is just a proxy for token difficulty (hard tokens naturally vary more across passes) — it's not telling us anything the loss doesn't already know"
- Alternative 2: "the correlation is driven by the phase transition (pass 10-11 rewrite), not by genuine per-token variation — if you fix the collapse, the correlation might disappear"
- Distinguishing experiment: after fixing gradient structure (random-depth training), re-run this probe. If ce_spread still correlates with future_gain, it's a genuine signal. If not, it was an artifact of deferred computation.

**Verdict: STRONG positive.** ce_spread is a validated candidate for elastic compute control.
But the signal may partly be an artifact of the current training recipe's collapse pattern.

---

## lm-eval Benchmarks (Directional, step ~11.8K, CPU)

**Important context:** This is a 68.3M model at step ~12K, trained from scratch on <0.4B tokens
with no warm-start and no multi-teacher learning. These numbers are DIRECTIONAL ONLY — they
establish a baseline, not a claim. The model is 10-100x smaller than typical benchmarked models
and has seen a tiny fraction of their training data.

| Benchmark | Accuracy | Random Baseline | Delta |
|-----------|----------|----------------|-------|
| SciQ | 25.9% | ~25% | +0.9% |
| PIQA | 54.8% | ~50% | **+4.8%** |
| WinoGrande | 49.8% | ~50% | -0.2% |
| ARC-Easy | 27.9% | ~25% | +2.9% |
| ARC-Challenge | 20.1% | ~25% | -4.9% |
| HellaSwag | 25.7% | ~25% | +0.7% |
| LAMBADA | 0.76% | ~0% | +0.76% |

**Summary:** Near-random on most benchmarks. PIQA shows the strongest above-random signal (+4.8%),
suggesting some physical/commonsense reasoning is emerging. ARC-Challenge is below random, which may
indicate answer-format issues rather than knowledge absence. LAMBADA near-zero confirms the model
cannot do long-range exact recall (consistent with the 82% recall gap finding).

**Comparison with Pythia-70M at similar training stage:** Pythia-70M at 300M tokens was also
near-random on most benchmarks. Intelligence at this scale emerges slowly. The relevant comparison
will be at 20K steps (~0.65B tokens) and after warm-start/multi-teacher improvements.

---

## Summary of All Findings for Round 2

| R1 Request | Key Finding | Implication for Design |
|------------|------------|----------------------|
| Recursive LM recipes | Pass identity universal at <200M, random-depth rare but powerful, exact memory extremely rare | Add pass-conditioned norm, implement random-depth training, scratchpad is differentiated |
| Intermediate objectives | Residual prediction + shortcut-consistency + self-distillation top 3; full-vocab intermediate CE is catastrophic | Use soft intermediate objectives, never full token prediction at intermediate passes |
| Growth methods | WSD + zero-init birth + selective weight transfer validated | v0.6.0a→v0.7.0 warm-start IS feasible if dimensions preserved |
| Exact recall | ARMT + FwPKM top candidates; 82% recall gap validates scratchpad necessity | Scratchpad is architecturally mandatory; upgrade to content-addressable memory |
| Embedding tax | 58.8% params in embeddings (worst ratio); ALBERT factored frees 32.1M params | ALBERT-style factored embeddings should be high priority |
| Probe E (disagreement) | ce_spread r=0.96 with future gain; 4.8x benefit ratio; phase transition at pass 10-11 | Elastic compute signal is validated but may be partly a collapse artifact |
| lm-eval benchmarks | Near-random except PIQA +4.8%; LAMBADA 0.76% confirms recall gap | At 68M/0.4B tokens, intelligence hasn't emerged yet; this is expected |
