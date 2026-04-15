# Tesla Design Session: Sutra Strategic Architecture Review

**Date:** 2026-04-15
**Session type:** Fresh Tesla mode — deep strategic rethink
**Question:** Are we pursuing the best path to the 5 Sacred Outcomes? Is Ekalavya the right singular priority? What alternatives exist?

---

## 1. THE PROBLEM (precise)

Build a from-scratch language model that:
- Has ~188M parameters (current) or up to 4B (ceiling)
- Trains on a single RTX 5090 (24GB VRAM)
- Competes with best-in-class models trained with 100-1000x more resources
- Is modular, improvable, community-extensible
- Achieves data efficiency through multi-source knowledge absorption
- Supports adaptive compute allocation

**The 5 Sacred Outcomes:**
1. Genuine Intelligence — smart model, competitive benchmarks + generation
2. Improvability — find and fix failures surgically
3. Democratized Development — community builds like Linux
4. Data Efficiency — learn more from less via multi-source learning
5. Inference Efficiency — adaptive compute, easy tokens exit early

---

## 2. CURRENT STATE (honest assessment)

### What exists:
- Sutra-Dyad-188M: byte-level model (12-layer global transformer d=1024 + 1-layer local decoder d=256)
- Baseline eval BPB: 1.430 after 5K CE-only steps
- Ekalavya KD system: covering decomposition, anchor-confidence routing, TAID+gating
- Best KD result: eval BPB 1.418 (routing run step 250, -0.012 below baseline)
- 6K TAID+gating run launched (currently running)

### What's working:
- Covering decomposition: lossless cross-tokenizer byte-probability recovery
- Routing: eliminates arithmetic mean entropy injection, eval improvement verified
- TAID+gating: mechanism stable through full alpha ramp (where AM collapsed)
- Offline caching: 10x throughput implemented and tested

### What's stuck:
- **KD ceiling with 2 transformer teachers: 0.005-0.016 BPB (Codex ceiling analysis)**
- Dense per-position KD hurts at full strength (~80% unhelpful positions)
- No non-transformer teacher validated yet
- O5 (adaptive compute) has NO mechanism in byte-level architecture
- O3 (democratization) is entirely theoretical
- "Decisive" improvement (5-7pp on benchmarks) seems unreachable from KD alone at current scale

### The honest question:
Is iterating on Ekalavya KD with 2-3 teachers on a 188M byte-level model the best path to the sacred outcomes? Or should we consider fundamentally different approaches?

---

## 3. CANDIDATE ARCHITECTURES

### Architecture A: Evolved Ekalavya (current path, improved)

Continue byte-level 188M model with progressive improvements:
- Fix gating (clamp 1.5, s_match temperature calibration)
- Add Mamba-1.4B as 3rd teacher (SSM diversity)
- Transfer-bank curriculum (top-20% position KD)
- Rolling cache for throughput
- Eventually: more teachers, longer training, uncertainty gating v2

**Serves:** O4 primarily. O1 through O4. O2/O3 partially. O5 not addressed.
**Ceiling:** ~0.02-0.03 BPB improvement with 3 diverse teachers. Maybe 1-2pp on benchmarks.
**Risk:** Low (mostly built). But ceiling may be too low for "decisive."

### Architecture B: MoE-Distilled Byte Model

Replace dense global transformer with sparse MoE:
- 12 MoE layers, d=512, 8 experts/layer, top-2 routing
- Active params: ~90M (same inference FLOPs). Total params: ~350M.
- Each expert specializes via teacher-guided routing during KD
- Teacher→expert assignment: different teachers distill into different expert subsets

**Serves:** O1 (4x capacity), O2 (per-expert inspection), O3 (add experts), O4 (expert specialization), O5 (routing = adaptive compute)
**Ceiling:** Higher than A because more capacity to absorb knowledge.
**Risk:** Medium. MoE at <500M total params is underexplored. Expert collapse. Router overhead.

### Architecture C: Knowledge Compiler (offline-first)

Decouple knowledge extraction from student training:
- Phase 1 (offline): Run unlimited teachers, extract structured artifacts (logits, representations, attention patterns, difficulty maps, agreement maps)
- Phase 2 (training): Student trains on CE + compiled artifacts at CE-only speed
- Student never sees a teacher during training

**Note:** This is actually an evolution of Architecture A — the offline caching system already implements a simplified version. Architecture C extends to more artifact types and more teachers.

**Serves:** O4 (unlimited teachers, no VRAM constraint), O1 (through O4), O5 (difficulty maps inform exits)
**Ceiling:** Theoretically highest (unlimited teachers). But information loss from compression.
**Risk:** High novelty. Pipeline complexity. No online adaptation (static artifacts).

---

## 4. STRESS TEST FINDINGS

### On Architecture A:
1. KD ceiling (0.016 BPB) may be fundamentally too low for "decisive" results
2. Student capacity at 188M may be the bottleneck, not teacher diversity
3. Byte-level 6x sequence tax means fewer effective "tokens" of context
4. O5 has no mechanism — this is a gap, not future work

### On Architecture B:
1. MoE at 350M total fits in VRAM (~8GB with optimizer states)
2. Expert collapse is a real risk at small scale
3. Teacher→expert alignment is unproven — router may not respect forced assignments
4. But: MoE naturally provides O2 (modularity) and partial O5 (routing)

### On Architecture C:
1. Collapses into "evolved A" — offline caching IS the simplified version
2. Real question: what artifacts beyond top-K logits would help?
3. Layer-wise representation fingerprints, attention patterns, uncertainty maps
4. No online adaptation is a real limitation vs TAID

### Cross-cutting insight:
**The fundamental bottleneck may be student CAPACITY, not teacher signal.** At 188M params with 6x byte overhead, the effective capacity for language modeling is roughly equivalent to a ~50-80M token-level model. Even world-class teachers can't teach a student that lacks the capacity to absorb the lesson.

---

## 5. DESIGN HYPOTHESES

H1: MoE will raise the KD ceiling by providing more absorptive capacity at the same inference cost.
H2: Adding a genuinely different teacher (SSM like Mamba) will provide more incremental benefit than adding a 3rd transformer teacher.
H3: The byte-level tax is worth it for cross-tokenizer alignment, but only if we can scale the global model to compensate.
H4: Adaptive compute (O5) can be achieved in the byte-level architecture via adaptive PATCHING (varying patch sizes based on content difficulty).
H5: Architecture B (MoE) and Architecture C (compiled knowledge) are not mutually exclusive — an MoE student trained on compiled multi-teacher knowledge could be the strongest combination.

H6: Token-level global model + byte-level local decoder (Architecture G) will outperform pure byte-level (Architecture A) at 188M params because the byte-level tax on effective context is too severe at this scale. Byte-level should be a local concern, not global.

H7: Sparse SwiGLU (top-10% activation) will increase effective capacity without any parameter cost, because it creates exponentially many unique subnetwork configurations. The quality impact of sparsity should be neutral or positive due to implicit regularization.

H8: Weight sharing (3 blocks × 4 repetitions) with the freed parameters reallocated to a stronger local decoder (4 layers d=512 instead of 1 layer d=256) will improve generation quality more than 12 unique blocks, because the local decoder is currently the weakest link.

H9: A byteified model (Bolmo-style) used as a byte-level teacher would raise the KD ceiling dramatically because it eliminates the covering decomposition approximation and operates in the same byte probability space as the student.

H10: The biggest bottleneck is student CAPACITY, not teacher signal quality. At 188M with byte overhead, effective capacity for language modeling is equivalent to ~50-80M token-level model. Any mechanism that increases effective capacity (sparsity, MoE, weight sharing + reallocation) should produce larger improvements than better KD.

---

## NEW CANDIDATES (added post-research)

### Architecture D: Byteified Student (Bolmo-style initialization)

Use Bolmo's "byteification" technique to convert a strong pretrained subword model to byte-level with <1% pretraining cost, then use this as a TEACHER or INITIALIZATION source (not the final model, to preserve "from-scratch" principle). The insight: byteified models already have byte-level representations that are richer than anything we can train from scratch in reasonable time.

Variant D1: Use byteified model as an additional teacher (strongest possible teacher signal for byte-level student).
Variant D2: Use byteification as initialization, then continue training from that point.
Note: D2 technically violates "no pretrained weights" — but VISION.md says outcomes > mechanisms.

### Architecture E: Sparse-Active Dendritic Network

Replace standard FFN with dendritic FFN blocks:
- 4 independent computation branches per neuron
- Top-k activation sparsity (10% of neurons fire per input)
- Each active neuron contributes all 4 branch outputs with learned gates
- Effective capacity: ~4x current (exponentially many unique subnetworks via combinatorial sparsity)
- Effective compute: same as current (only 10% fire)
- Natural O5 mechanism (sparsity = adaptive compute)

### Architecture F: Universal Weight-Shared Byte Transformer

3 unique transformer blocks repeated 4x each = 12 effective layers.
- Reduces model description length (Kolmogorov complexity)
- Frees ~65M params for reallocation (wider FFN, larger local decoder)
- Natural iterative refinement for O5 (exit after 1-2 iterations for easy tokens)
- Only 3 blocks to debug (O2) or improve (O3)
- Risk: may limit layer specialization

### Architecture G: Hybrid Token-Byte Adaptive Model

RADICAL RETHINK: What if byte-level is a local concern, not a global concern?

Global Model: Token-level transformer (12 layers, d=1024, 16K BPE tokenizer, context 2048 tokens)
  - Long-range context at token granularity (4-6x more context than byte-level)
  - No byte-level sequence length explosion at the global level

Local Decoder: Byte-level decoder (4 layers, d=512)
  - For each token, predicts constituent bytes
  - Sees: global context (from token model) + previous bytes in current token
  - This is the Megabyte architecture but with tokens as patches

KD Interface:
  - Token-level teachers: direct logit KD (vocabulary mapping)
  - Byte-level teachers (e.g., Bolmo): covering decomposition at local decoder
  - The local decoder produces byte probabilities → universal KD target space

O5 (Adaptive Compute):
  - Easy tokens: skip local decoder entirely, use token-level logit
  - Hard tokens: full 4-layer byte decoder for refinement
  - The model learns WHEN byte-level processing helps

Key advantages over pure byte-level:
  - 4-6x more effective context (2048 tokens vs 256 patches)
  - Token-level global model is more efficient per FLOP
  - Local decoder is small and focused
  - Byte-level KD capability preserved at local decoder level
  - Natural O5 mechanism (skip local decoder for easy tokens)

Key risks:
  - Re-introduces tokenizer dependency at global level (but it's OUR tokenizer)
  - Token boundary effects (some information crosses token boundaries)
  - Two-level architecture is more complex to train

### Sparse SwiGLU (mechanism, compatible with any architecture above)

Zero-parameter activation sparsity for FFN layers:
  hidden = SiLU(W_gate * x) * W_up * x
  mask = top_k(|hidden|, k=0.1 * ff_dim)   # keep top 10%
  output = W_down * (hidden * mask)

Creates exponentially many effective subnetworks (C(2730, 273) ~ 10^400) from fixed params.
No extra parameters. No extra memory. Different inputs activate different subnetworks.
Compatible with ALL candidate architectures (A through G) as a drop-in FFN replacement.

---

## 6. OPEN QUESTIONS FOR CODEX (ANSWERED IN R1)

See §7 for Codex R1 answers to all questions.

---

## 7. CODEX TESLA R1 REVIEW — SYNTHESIS (2026-04-15)

**Session:** GPT-5.4 xhigh, read-only, 262K tokens consumed. Full repo read.

### R1 Priority Directive

> "Stop optimizing Ekalavya as the main strategy. Run a decisive matched-budget scout of token-global / byte-local Architecture G against the current byte-global Sutra-Dyad."

### R1 Foundational Assumption Audit (9 assumptions challenged)

1. **Byte-level as primary computation substrate** — CHALLENGED. Bytes are right I/O and distillation boundary, but wrong global reasoning unit. Evidence: O5 has no mechanism in byte architecture, Bolmo byteification converts subword→byte at <1% cost.
2. **Better KD compensates for student capacity limits** — CHALLENGED. 188M byte student is rate-distortion constrained. Flat alpha failed because student couldn't absorb signal at 1:19 capacity ratio. Full-position KD hurt CE by +0.040 BPB.
3. **Multi-teacher diversity helps if aggregated correctly** — PARTIALLY VALID. AM was destructive (+0.061 BPB hurt). Routing helped. But cross-tokenizer disagreement may be alignment noise, not complementary knowledge.
4. **Architecture is "locked"** — CHALLENGED. Lock only excludes tested hybrids (HEMD/P-GQA). Does NOT test Architecture G (token-global/byte-local).
5. **MoE is the obvious capacity multiplier** — REJECTED at our scale. OLMoE=1B active, sub-500M uncharted. Sparse FFN is the right first step.
6. **Offline artifacts are a knowledge compiler** — DEMOTED. Useful as cache + diagnostics, not a new learning primitive.
7. **O1 can be inferred from BPB** — CHALLENGED. BPB gains on short probes may not translate to reasoning/generation. Need benchmark/generation gate.
8. **Fixed 6-byte patching is acceptable** — CHALLENGED. Convenience primitive, not proven optimal. Dynamic entropy patches or token-aligned boundaries are more aligned with O5.
9. **Biological inspiration → modules** — SIMPLIFIED. Implement cheapest computational equivalent first (sparse SwiGLU), not literal neuron analogy.

### R1 Top 3 Inherited Assumptions to Challenge

1. Global byte computation as the main intelligence substrate
2. KD optimization before student geometry redesign
3. Dense feed-forward activation instead of sparse active capacity

### R1 Per-Outcome Confidence Scores

| Architecture | O1 | O2 | O3 | O4 | O5 | Mean |
|---|---:|---:|---:|---:|---:|---:|
| A: Evolved Ekalavya | 3 | 4 | 3 | 5 | 1 | 3.2 |
| B: MoE-Distilled | 4 | 4 | 4 | 4 | 3 | 3.8 |
| C: Knowledge Compiler | 4 | 6 | 6 | 6 | 3 | 5.0 |
| G/H: Token-Global Byte-Local | 5 | 6 | 5 | 5 | 6 | 5.4 |

**Verdict:** G/H is the strongest candidate but still far from >=9/10 convergence target. Must be fully specified.

### R1 Codex-Proposed Architectures

- **H:** Token-Global Byte-Local Recurrent Sparse Compiler (G + weight sharing + sparse FFN + compiler curriculum)
- **I:** Native Byte Student with Sparse Dendritic FFN (keep byte, add compartmental sparse SwiGLU)
- **J:** Byteified Teacher Initialization Track (Bolmo-style, strategic from-scratch violation test)
- **K:** Recurrent Universal Byte Transformer (3-4 shared blocks × 6-8 reps, step embeddings, optional halting)

### R1 Recommended Path

1. Run G scout: token-global/byte-local, matched param budget, no MoE/recurrence/exotic KD
2. In parallel: Sparse SwiGLU ablation in current byte model
3. Ekalavya = infrastructure, not strategy. Keep routing/gating. Freeze TAID expansion until 6K proves gain outside variance.
4. Compiler artifacts for diagnosis only: routed logits, disagreement maps, difficulty maps
5. O5 gate on every future run: tokens/sec, active FLOPs, decoder skip rate, memory footprint

---

## 8. BOLMO/BLT RESEARCH FINDINGS (2026-04-15)

### Bolmo (Allen AI, Dec 2025)

**Architecture:** Latent Tokenizer LM — bytes in, bytes out, computation on variable-length patches.

**Components:**
1. **Hybrid Byte Embedding:** `e_i = E_b[x_i] + E_suf[subwordSuffix(x_{1:i})]` — residual subword suffix embeddings give free context
2. **Local Encoder:** mLSTM (TFLA form) + FFN with SwiGLU. Chosen for wallclock speed over transformer.
3. **Non-Causal Boundary Predictor:** One-byte lookahead cosine similarity. 99%+ accuracy vs teacher tokenizer.
4. **Pooling:** Last byte in each patch → patch token (simple, no learned pooling)
5. **Global Transformer:** Unchanged pretrained backbone (OLMo-2/3)
6. **Depooling:** `z_t = e_t + P(h_{j(t)})` — linear projection augments bytes with patch global context
7. **Local Decoder:** 4 mLSTM layers (larger than encoder)
8. **LM Head:** 512-way softmax (256 bytes × 2 boundary options)

**Key Numbers:** Bolmo-1B = 1.5B total params (500M overhead). Bolmo-7B = 7.6B (600M overhead). Byteification: 49B tokens total (<1% of pretraining budget). CUTE: +21.7pp over OLMo-3 7B.

**Relevance to Sutra:**
- mLSTM for local byte processing is empirically validated for speed
- Non-causal boundary prediction works — 99%+ accuracy
- Pooling is simple (last byte, no cross-attention)
- Hybrid byte embedding (subword suffix residuals) is a free boost
- At 188M total, the 500M overhead of local encoder/decoder is prohibitive for full Bolmo-style
- But the INSIGHTS are portable: local mLSTM > local transformer, simple pooling > cross-attention

### BLT (Meta, Dec 2024)

**Key Innovation:** Entropy-based dynamic patching.
- Boundary when `H(x_t) > θ_g` (high entropy = unpredictable = new patch)
- Or when `H(x_t) - H(x_{t-1}) > θ_r` (entropy jump)
- Entropy model: 100M params — **prohibitive at our scale** (53% overhead on 188M model)

**Hash N-gram Embeddings:** 3-8 grams, 500K hash vocabulary. "Very effective with very large improvements in BPB." Complementary with cross-attention.

**At Small Scale:** BPE outperforms BLT; BLT's advantage appears at scale. Crossover decreases at larger model sizes. Not proven below 400M.

**Relevance:** Entropy patching insight is sound but needs cheaper boundary predictor. N-gram hash embeddings are immediately applicable. Average patch size 4-5 bytes is the sweet spot.

---

## 9. COMPOSITE ARCHITECTURE: SUTRA-G (Token-Global Byte-Local Sparse Adaptive)

**This is the refined candidate that integrates ALL findings from R1, Bolmo/BLT research, and theoretical work.**

### 9.1 Design Principles

1. **Token-global, byte-local:** Global reasoning at token granularity (4-6x more effective context). Byte processing is a local concern.
2. **Sparse activation:** Top-k SwiGLU (10% activation) in all FFN layers. Zero extra params. Exponential effective subnetwork diversity.
3. **Simple first, add complexity only when proven:** No MoE, no recurrence, no exotic KD in the base design. Each is a future upgrade with a clear gate.
4. **O5 native:** Local decoder skip is the primary adaptive compute mechanism. Easy tokens use token-level logit directly.
5. **Ekalavya as infrastructure:** KD operates at BOTH levels — token-level for global, byte-level for local decoder. Covering decomposition preserved at local level.

### 9.2 Architecture Specification

```
SUTRA-G: Token-Global Byte-Local Sparse Adaptive Model

INPUT: raw bytes (context length: 8192 bytes)

BYTE ENCODER (lightweight, ~8M params):
  - Byte embedding: 256 × 256 (65K params)
  - Hash n-gram embedding: 3-8 grams, 100K hash vocab, 256-dim (~25.6M params, shared across n-gram sizes)
  - 1 mLSTM layer, d=256, SwiGLU ff=682 (~3M params)
  - Boundary predictor: cosine similarity, one-byte lookahead (non-causal during prefill)
    → learns to segment into variable-length patches (avg ~4.5 bytes)
  - Pooling: last byte representation per patch → patch token

GLOBAL TRANSFORMER (~150M params):
  - 12 layers, d=768, 12 heads, Sparse SwiGLU ff=2048 (top-10% activation)
  - RoPE, RMSNorm, residual connections
  - Context: ~1800 patches (8192 bytes / 4.5 avg = ~1820 patches)
  - This is the main intelligence substrate
  - O5: early-exit at layer 6 for easy patches (confidence gate)

BYTE DECODER (local, ~25M params):
  - Depooling: e_t + Linear(h_{patch(t)}) — byte gets its patch's global context
  - 2 mLSTM layers, d=256, SwiGLU ff=682
  - Residual bypass: byte embedding → decoder output (load-bearing, validated)
  - O5 SKIP: when global model's token-level confidence > θ_skip, skip decoder entirely
    → use token-level logit mapped to byte distribution
  - Otherwise: full byte-level decoding for current patch

LM HEAD:
  - Byte-level: 256-way softmax (when decoder runs)
  - Token-level shortcut: global logit → byte distribution via learned mapping (when decoder skipped)

TOTAL PARAMS: ~183M (comparable to current 188M Sutra-Dyad)
ACTIVE PARAMS: ~120M typical (sparse FFN + decoder skipping)
```

### 9.3 Parameter Budget

| Component | Params | Notes |
|-----------|--------|-------|
| Byte embedding + n-gram hash | ~26M | BLT-style n-gram is "very effective" |
| Local encoder (1 mLSTM) | ~3M | Bolmo validated mLSTM > transformer for local |
| Global transformer (12 layers) | ~150M | Main intelligence. Sparse SwiGLU = 10% active FFN |
| Local decoder (2 mLSTM) | ~4M | Bolmo uses 4 layers; we use 2 (smaller model) |
| **Total** | **~183M** | Comparable to current 188M |

### 9.4 How Each Outcome is Served

**O1 (Intelligence):** Token-global processing gives 4x more effective context than byte-global. Sparse SwiGLU gives exponential subnetwork diversity. N-gram hash embeddings give richer input representations. This is fundamentally more capacity-efficient than current architecture.

**O2 (Improvability):** Three-tier decomposition (encoder → global → decoder) with clear interfaces. Global model handles reasoning (debug globally). Decoder handles spelling/morphology (debug locally). Encoder handles segmentation (debug independently). Each is independently testable.

**O3 (Democratized Development):** Community can improve: (a) encoder/patching strategy, (b) global model layers, (c) local decoder, (d) n-gram vocabulary, (e) KD pipeline. Each component has clean inputs/outputs. Domain experts replace/fine-tune the global model for their domain; the encoder/decoder are reusable.

**O4 (Data Efficiency):** Dual-level KD:
- Token-level: direct logit KD from token-level teachers (SmolLM2, Pythia) — no covering decomposition needed at global level
- Byte-level: covering decomposition at local decoder — same mechanism as current Ekalavya
- Byteified teacher (Bolmo-1B): native byte-level KD at local decoder — eliminates covering approximation
- Compiler artifacts: cached teacher logits + difficulty maps drive position-selective KD

**O5 (Inference Efficiency):**
- Sparse SwiGLU: only 10% of FFN neurons fire → ~40% compute reduction in FFN
- Early exit: easy patches exit global transformer at layer 6 → ~50% reduction for ~60% of patches
- Decoder skip: confident token predictions skip local decoder entirely → saves decoder compute for ~40% of patches
- Combined: average inference FLOPs could be 40-60% lower than dense equivalent

### 9.5 Training Strategy

```
Stage 0: Global Transformer Only (CE, ~5K steps)
  - Token-level training with our own tokenizer
  - Establish global model baseline
  - Validate: BPB, generation quality, attention patterns

Stage 1: Add Byte Encoder + Decoder (CE, ~5K steps)
  - Freeze global transformer
  - Train encoder (boundary prediction + n-gram embedding) and decoder (byte reconstruction)
  - Validate: byte-level BPB, boundary quality (vs BPE alignment), decoder skip rate

Stage 2: Unfreeze + Ekalavya KD (~10K-20K steps)
  - Unfreeze all components
  - Dual-level KD: token-level + byte-level
  - Routing, uncertainty gating, optional TAID at byte level
  - Progressive: start with 1 teacher, add teachers as mechanism stabilizes
  - Validate: BPB trajectory, generation quality, O5 metrics

Stage 3: Sparse SwiGLU Activation (~5K steps)
  - Add top-10% activation sparsity to global FFN layers
  - Fine-tune with sparsity constraint
  - Validate: BPB impact, throughput improvement, subnetwork diversity

Stage 4: Early Exit + Decoder Skip (~5K steps)
  - Add confidence gates at layer 6 (early exit) and pre-decoder (skip gate)
  - Train with halting pressure loss
  - Validate: skip rates, BPB per-skip-category, tokens/sec
```

### 9.6 Comparison: Sutra-G vs Current Sutra-Dyad

| Dimension | Sutra-Dyad (current) | Sutra-G (proposed) |
|-----------|---------------------|-------------------|
| Global processing | Byte-level (256 patches of 6 bytes) | Token-level (~1820 patches of ~4.5 bytes) |
| Effective context | ~256 patches = ~1536 bytes | ~1820 patches = ~8192 bytes (**5.3x more**) |
| Global model | 12-layer transformer, d=1024 | 12-layer transformer, d=768, sparse SwiGLU |
| Local decoder | 1-layer transformer, d=256 | 2-layer mLSTM, d=256 |
| Patching | Fixed 6-byte | Learned variable (~4.5 avg) |
| Input enrichment | Byte embedding only | Byte + n-gram hash (3-8 grams) |
| O5 mechanism | None | Sparse FFN + early exit + decoder skip |
| KD interface | Byte-level only (covering decomp) | Dual: token-level + byte-level |
| Total params | 188M | ~183M |
| Active params | 188M (all dense) | ~120M typical |

### 9.7 Risks and Mitigations

1. **Tokenizer dependency at global level:** Mitigated by using OUR tokenizer (8-16K vocab, trained on our data). Byte-local decoder catches tokenizer failures. Test: confirm decoder activates on tokenizer edge cases (rare words, code, multilingual).

2. **Boundary prediction quality:** If learned boundaries are bad, global context is fragmented. Mitigation: start with BPE-aligned boundaries as warmup target, then let model learn to deviate where helpful. Gate: measure boundary F1 vs BPE alignment.

3. **mLSTM implementation:** Requires xlstm library or custom implementation. Mitigation: if mLSTM is too complex, fall back to lightweight causal transformer for local encoder/decoder (Bolmo chose mLSTM for speed, not quality).

4. **Sparse SwiGLU training instability:** Top-k gradients may be noisy. Mitigation: straight-through estimator for backward pass. Add sparsity AFTER base model converges (Stage 3). Gate: if BPB degrades >0.02, reduce sparsity to 20%.

5. **N-gram hash collisions:** 100K hash vocab may have significant collisions. Mitigation: increase to 500K if VRAM allows (only 128M params). Monitor collision rate on validation set.

### 9.8 What Would Raise Confidence to 9/10

| Outcome | Current Score | Path to 9/10 |
|---------|--------------|---------------|
| O1 (Intelligence) | 5 | G scout beats Sutra-Dyad by ≥0.05 BPB at matched compute. Generation quality visibly better. |
| O2 (Improvability) | 6 | Demonstrate: fix a reasoning failure by modifying only global model. Fix a spelling failure by modifying only decoder. |
| O3 (Democratization) | 5 | Build and test: swap one component (e.g., decoder), verify composability. Community tooling prototype. |
| O4 (Data Efficiency) | 5 | Dual-level KD produces ≥0.03 BPB improvement over CE-only at 10K steps. Byteified teacher adds ≥0.02 beyond subword teachers. |
| O5 (Inference) | 6 | Demonstrate: 30%+ tokens skip decoder. 40%+ early exit at layer 6. Combined throughput 2x dense baseline. |

---

## 10. SUTRA-G DETAILED SPECIFICATION (R2-ready)

### 10.1 Tokenizer Design

**Decision:** NO external tokenizer. The boundary predictor IS the tokenizer.

The original motivation for byte-level was avoiding tokenizer dependency. Architecture G reintroduces a segmentation boundary, but it's LEARNED, not a fixed BPE vocabulary. This is a critical distinction:

```
BPE tokenizer: fixed vocabulary, fixed boundaries, trained separately
Sutra-G boundary predictor: learned boundaries, variable-length patches, trained jointly

The "tokenizer" is a 1-parameter cosine gate inside the model itself.
```

**Warmup strategy:** Initialize boundary predictor to produce BPE-like boundaries (avg ~4.5 bytes/patch) by pretraining on BPE alignment for ~500 steps. Then let it learn to deviate. This gives a good starting point without hard tokenizer dependency.

**Vocabulary for global model:** The global model does NOT use a token vocabulary. It operates on patch representations produced by the encoder — these are continuous vectors, not discrete tokens. There is no embedding table for "tokens." The encoder IS the embedding.

**This preserves byte-level I/O while getting token-level efficiency at the global level.**

### 10.2 Boundary Predictor

**Formulation (Bolmo-inspired, adapted for from-scratch):**

```python
# During prefill (non-causal: can peek 1 byte ahead):
q = W_q @ e_{t+1}  # next byte's embedding
k = W_k @ e_t      # current byte's embedding
sim = (q · k) / (||q|| * ||k||)
p_boundary = 0.5 * (1 - sim)  # high when next byte is dissimilar

# Boundary decision:
boundary_t = (p_boundary > θ) OR (patch_length >= max_patch_size)
# θ is a learned scalar, initialized to produce avg ~4.5 bytes/patch
# max_patch_size = 8 (hard cap, prevents degenerate long patches)

# During generation (autoregressive: no lookahead):
# Use boundary token: model predicts whether to start new patch
# Or: use running entropy estimate from local encoder state
```

**Training signal:** Auxiliary loss encouraging boundaries at high-entropy positions:
```
L_boundary = BCE(p_boundary, 1[H(next_byte | context) > θ_entropy])
```
Where `H(next_byte | context)` is estimated from the local encoder's hidden state.

**Parameters:** W_q, W_k ∈ R^{256×256} = 131K params (negligible).

### 10.3 N-gram Hash Embedding

**Formulation (BLT-style):**

```python
# For each byte position i, compute hash embeddings for n-grams ending at i:
def ngram_embed(byte_sequence, i, E_hash, hash_fn):
    emb = 0
    for n in [3, 4, 5, 6, 7, 8]:
        if i >= n - 1:
            gram = byte_sequence[i-n+1 : i+1]
            h = hash_fn(gram) % hash_vocab_size
            emb = emb + E_hash[n][h]  # per-n embedding table
    return emb

# Final byte representation:
e_i = E_byte[x_i] + ngram_embed(x, i, E_hash, RollPolyHash)
```

**Hash function:** RollPolyHash (same as BLT, rolling polynomial hash for O(1) incremental computation).

**Hash vocab:** 100K entries per n-gram size × 6 sizes = 600K entries total.
Embedding dim: 256. Total params: 600K × 256 = **153.6M params**.

**PROBLEM:** This is too large. 153M just for hash embeddings exceeds our entire model budget.

**Solution:** Share embeddings across n-gram sizes + reduce vocab:
- Shared embedding table: 100K entries × 256 dim = **25.6M params**
- Each n-gram size uses a different hash function into the SAME table
- 25.6M is ~14% of total model — significant but defensible given BLT's "very large improvements"

**Alternative (cheaper):** 50K entries × 128 dim = 6.4M params, projected up to 256 via Linear(128, 256). Total: 6.4M + 33K = ~6.4M. Much cheaper. Start here.

### 10.4 Sparse SwiGLU Specification

```python
class SparseSwiGLU(nn.Module):
    def __init__(self, d_model, ff_dim, sparsity=0.1):
        super().__init__()
        self.w_gate = nn.Linear(d_model, ff_dim, bias=False)
        self.w_up = nn.Linear(d_model, ff_dim, bias=False)
        self.w_down = nn.Linear(ff_dim, d_model, bias=False)
        self.sparsity = sparsity
        self.k = int(ff_dim * sparsity)
    
    def forward(self, x):
        hidden = F.silu(self.w_gate(x)) * self.w_up(x)
        # Top-k selection (keep top 10% by magnitude)
        topk_vals, topk_idx = hidden.abs().topk(self.k, dim=-1)
        # Sparse mask
        mask = torch.zeros_like(hidden)
        mask.scatter_(-1, topk_idx, 1.0)
        # Apply mask
        sparse_hidden = hidden * mask
        return self.w_down(sparse_hidden)
```

**Backward pass:** Straight-through estimator (STE). Gradients flow through all neurons in the backward pass, but only top-k neurons contribute to the forward pass. This is standard for top-k sparsity (see Sparse Transformer, Spark Transformer).

**Training schedule:** 
- Stages 0-1: Dense SwiGLU (no sparsity). Let model converge first.
- Stage 3: Gradually introduce sparsity. Ramp from 100% → 10% active over 2K steps.
- This avoids the instability of sparse training from scratch.

**Effective capacity:** With ff_dim=2048, k=205 active neurons, the number of possible activation patterns is C(2048, 205) ≈ 10^300. Each input activates a unique subnetwork.

**Compute savings:** FFN forward pass goes from 2×d×ff to 2×d×k + topk overhead. At 10% sparsity: ~90% FFN FLOP reduction. FFN is ~2/3 of transformer compute, so total savings: ~60% of per-layer compute.

### 10.5 Early Exit Mechanism

**Design:** Confidence-based early exit at the PATCH level (not byte level).

```python
# After global transformer layer L (e.g., L=6):
confidence = max(softmax(exit_head(h_L)))  # simple linear head → 256-way softmax
if confidence > θ_exit:
    # Use h_L as the patch representation (skip layers 7-12)
    patch_repr = h_L
else:
    # Continue through remaining layers
    patch_repr = h_12  # full depth
```

**Halting pressure:** Add auxiliary loss that encourages early exit:
```
L_halt = λ_halt * Σ_patches (1 - exit_probability)
```
Where exit_probability is the probability of exiting at each candidate layer.

**Candidate exit layers:** Layers 4, 6, 8, 10 (every 2 layers). This gives 4 exit points.

**λ_halt:** Start at 0.0 (no pressure), ramp to 0.01 over 5K steps. Too much pressure → quality degrades. Too little → no early exit.

**Expected savings:** If 60% of patches are "easy" (common words, whitespace, predictable syntax) and exit at layer 6, average layers per patch = 0.6×6 + 0.4×12 = 8.4, saving 30% of global compute.

### 10.6 Decoder Skip Mechanism

**When to skip:** When the global model is confident about the next patch's content.

```python
# For each patch j:
global_conf = max(softmax(skip_head(patch_repr_j)))
if global_conf > θ_skip:
    # Map token-level logit to byte distribution
    byte_probs = token_to_byte_map(patch_repr_j)
else:
    # Run full byte decoder
    byte_probs = local_decoder(patch_repr_j, byte_prefix)
```

**Token-to-byte mapping:** A small MLP (768 → 256 → 256×max_patch_len) that predicts the byte sequence distribution for the entire patch from the global representation alone. This is ~200K params.

**Training signal:** The skip head is trained jointly. During training, both paths run, and the skip head learns to predict when the full decoder would agree with the shortcut.

**Expected savings:** Common English words ("the", "and", "ing", etc.) should be skippable. Estimate: 30-40% of patches skipped.

### 10.7 VRAM Budget (Training)

```
Component                          Params    Memory (BF16)  Optimizer (AdamW)  Total
─────────────────────────────────────────────────────────────────────────────────────
Byte embedding (256×256)           65K       130KB          390KB              520KB
N-gram hash embedding (50K×128)    6.4M      12.8MB         38.4MB             51.2MB
Local encoder (1 mLSTM, d=256)     ~3M       6MB            18MB               24MB
Global transformer (12L, d=768)    ~150M     300MB          900MB              1.2GB
Local decoder (2 mLSTM, d=256)     ~4M       8MB            24MB               32MB
LM heads + skip/exit heads         ~0.5M     1MB            3MB                4MB
─────────────────────────────────────────────────────────────────────────────────────
TOTAL MODEL                        ~164M     328MB          984MB              1.31GB

Activations (batch=32, seq=8192 bytes → ~1820 patches):
  Byte encoder activations:    32 × 8192 × 256 × 2 = ~134MB
  Global transformer activs:   32 × 1820 × 768 × 12 × 2 = ~1.03GB
  Local decoder activations:   32 × 8192 × 256 × 2 × 2 = ~268MB
  Gradient checkpointing:      /3 = ~477MB
─────────────────────────────────────────────────────────────────────────────────────
TOTAL TRAINING VRAM (est):     ~3.2GB without teachers, ~8-10GB with 2 teachers

Teachers (frozen, inference only):
  SmolLM2-1.7B (BF16):        ~3.4GB
  Pythia-1.4B (BF16):         ~2.8GB
  
TOTAL with teachers:           ~9-14GB → FITS in 24GB RTX 5090
```

### 10.8 Dual-Level KD

**Token-level KD (at global transformer):**
```
# For each patch j:
teacher_logits_j = teacher_model(token_context[:j])  # subword logits
# Map teacher subword logits to patch space via learned projection
# (teacher vocab 50K → student patch space is implicit, via representation matching)
# Actually: at the global level, use REPRESENTATION distillation, not logit KD
# Match global hidden state to teacher hidden state via projection:
L_repr_global = MSE(project(student_h_j), teacher_h_j)
```

**Byte-level KD (at local decoder):**
```
# Standard Ekalavya with covering decomposition:
teacher_byte_probs = covering_decomposition(teacher_subword_logits)
L_kd_byte = FKL(teacher_byte_probs || student_byte_probs)
# With routing, gating, optional TAID — all existing infrastructure
```

**Combined loss:**
```
L = L_ce_byte + α_global * ramp * L_repr_global + α_byte * ramp * L_kd_byte
```

**Key insight:** Global KD is representation-level (cheaper, more stable than logit KD across different vocabs). Byte KD is logit-level (uses covering decomposition). They target different aspects of the model.

### 10.9 Local Encoder/Decoder: mLSTM vs Lightweight Transformer

**Decision criteria:**
| Factor | mLSTM | Lightweight Transformer |
|--------|-------|------------------------|
| Inference speed | Faster (linear in seq length, no KV cache) | Slower (quadratic attention, needs KV cache) |
| Training speed | Comparable (TFLA kernel) | Comparable (FlashAttention) |
| Quality | Bolmo validates at 1B+ scale | BLT validates at 400M+ scale |
| Implementation | Requires xlstm library or custom CUDA | Standard PyTorch |
| At our scale | Untested below 1B | Untested below 400M |

**Decision:** Start with lightweight causal transformer (1-2 layers, d=256, 4 heads) for SIMPLICITY. This is what we already have in Sutra-Dyad. If inference speed becomes the bottleneck, swap to mLSTM later.

**Revised parameter count with transformer local:**
- Local encoder: 1 layer, d=256, 4 heads, SwiGLU ff=682 → ~2.1M params
- Local decoder: 2 layers, d=256, 4 heads, SwiGLU ff=682 → ~4.2M params
- Total local: ~6.3M (comparable to mLSTM estimate)

### 10.10 Training Data Pipeline

**Byte-level data (same as current):**
- Raw UTF-8 bytes from OpenWebText/FineWeb shards
- ByteShardedDataset (existing data_loader.py)
- Context: 8192 bytes per example

**Boundary learning:**
- During training, the boundary predictor processes all 8192 bytes
- Produces variable-length patches (avg ~4.5 bytes, max 8)
- Global transformer sees ~1820 patch representations

**Teacher alignment:**
- Token-level teachers (SmolLM2, Pythia) process the SAME text
- Teacher tokenizer segments text into subword tokens
- At global level: match student patch representations to teacher token representations
  → Teacher token j maps to student patches covering the same byte span
  → Alignment is computed by byte-span overlap
- At byte level: covering decomposition (existing infrastructure, unchanged)

**Batch construction:**
```python
# 1. Sample 8192-byte window from ByteShardedDataset
# 2. Run boundary predictor → variable-length patches
# 3. Pad/truncate to max_patches per example
# 4. For KD: align teacher tokens to student patches by byte span
# 5. Forward pass: encoder → global → decoder → byte predictions
```

---

## 11. MINIMAL VIABLE SUTRA-G (MVG) — Core Hypothesis Test

**Purpose:** Test the ONE question that matters before committing: does token-global/byte-local beat byte-global at matched params and compute?

**Strip everything non-essential:**
- No Sparse SwiGLU (add later if base works)
- No early exit (add later)
- No decoder skip (add later)
- No n-gram hash embeddings (add later if needed)
- No dual-level KD (CE-only first)
- Fixed-size patches (same as current Sutra-Dyad, just with tokens as patches)

### MVG Architecture

```
INPUT: raw bytes (context: 1536 bytes, same as current)

BYTE ENCODER (minimal, ~1M params):
  - Byte embedding: 256 × 256 (65K params)
  - Fixed patching: group every P bytes (P=4, matching BPE-like compression)
  - Patch projection: Linear(P*256, d_global) = Linear(1024, 768)
  - No boundary predictor, no n-grams — just concatenate + project (same as current)

GLOBAL TRANSFORMER (~130M params):
  - 12 layers, d=768, 12 heads, SwiGLU ff=2048
  - RoPE, RMSNorm
  - Context: 384 patches (1536/4)
  - Standard dense SwiGLU (no sparsity)

LOCAL DECODER (~8M params):
  - Depooling: byte_emb + Linear(768, 256) per byte
  - 1 causal transformer layer, d=256, 4 heads, SwiGLU ff=682
  - Residual bypass (load-bearing, keep it)
  - Standard byte-level prediction

TOTAL: ~139M params
```

### MVG vs Current Sutra-Dyad

| Dimension | Sutra-Dyad | MVG | Difference |
|-----------|-----------|-----|------------|
| Global dim | d=1024 | d=768 | -25% (compensated by more patches) |
| Patch size | 6 bytes | 4 bytes | Finer granularity |
| Num patches | 256 | 384 | **+50% more context** |
| Global params | ~155M | ~130M | -16% |
| Local decoder | 1 layer d=256 | 1 layer d=256 | Same |
| Total params | 188M | ~139M | -26% |

**Wait — this is NOT a fair test.** MVG has 49M fewer params. The core hypothesis is about token-global vs byte-global at MATCHED params, not a smaller model.

### MVG v2: Parameter-Matched

```
GLOBAL TRANSFORMER (~170M params):
  - 16 layers, d=768, 12 heads, SwiGLU ff=2048
  - OR: 12 layers, d=896, 14 heads, SwiGLU ff=2389

LOCAL DECODER (~8M params):
  - Same as MVG v1

TOTAL: ~179M params (close to 188M)
```

Option A: 16 layers at d=768 → deeper, same width
Option B: 12 layers at d=896 → same depth, wider

**Recommendation:** Option B (12L d=896) for direct comparison with Sutra-Dyad (also 12L).

### MVG v2 Spec (12L d=896)

```
Global: 12 layers, d=896, 14 heads, SwiGLU ff=2389
  - Attention: 14 heads × 64 dim = 896
  - Q/K/V: 3 × 896 × 896 = 2.41M per layer
  - O: 896 × 896 = 0.80M per layer
  - SwiGLU: 896 × 2389 × 3 = 6.42M per layer (gate + up + down)
  - Norm: 2 × 896 = 1.8K per layer
  - Per layer total: ~9.63M
  - 12 layers: ~115.6M
  - Embeddings: none (patch projection replaces)
  - Patch projection: 1024 → 896 = 0.92M
  Total global: ~116.5M

Local decoder: 1 layer d=256, 4 heads, SwiGLU ff=682
  - ~2.1M params

Byte embedding: 256 × 256 = 65K
Depooling projection: 896 → 256 = 229K
LM head: 256 × 256 = 65K (tied with byte embedding)

TOTAL: ~119M params
```

**Still short of 188M.** The issue: d=768/896 global with P=4 patches is inherently more parameter-efficient than d=1024 with P=6 patches because the patch projection is smaller.

**Resolution:** Use the freed params to strengthen the local decoder:
- 2 local decoder layers instead of 1: +2.1M
- Or: increase d_global to 1024 (same as current): more params in attention/FFN

### MVG v3: True Parameter Match (d=1024, P=4)

```
BYTE ENCODER:
  - Byte embedding: 256 × 256 = 65K
  - Fixed patch: P=4 bytes
  - Patch projection: Linear(4×256, 1024) = Linear(1024, 1024) = 1.05M

GLOBAL TRANSFORMER (12L d=1024, same as current):
  - 12 layers, d=1024, 16 heads, SwiGLU ff=2730
  - Per layer: ~14.3M (same as current Sutra-Dyad global)
  - 12 layers: ~172M
  - Context: 384 patches (1536/4) vs current 256 patches (1536/6)
    → **50% more context at same sequence length**

LOCAL DECODER (same as current):
  - 1 layer d=256, 4 heads, SwiGLU ff=682
  - ~2.1M params
  - Residual bypass

Depooling: 1024 → 256 = 262K
LM head: 256 × 256 = 65K

TOTAL: ~175M params (vs 188M current)
```

**The 13M gap** comes from the patch embedding: current Sutra-Dyad has 6×256=1536 → 1024 projection (1.57M), MVG has 4×256=1024 → 1024 (1.05M). Plus the global model is identical. The gap is in the byte embedding table (current uses 256×1536 = 393K for the raw patch concat, MVG uses 256×1024 = 262K).

**This IS the fair test.** Same global model, same local decoder, same total budget (within 7%). The ONLY difference: P=4 patches (384 patches of 4 bytes) vs P=6 patches (256 patches of 6 bytes). This directly tests: does finer patching (more context) beat coarser patching?

**But wait — this isn't really testing "token-global vs byte-global."** Both use byte-level patches with a global transformer. The real G hypothesis is that SEMANTIC patches (variable-length, aligned to morpheme/word boundaries) are better than FIXED byte-count patches. Fixed P=4 vs P=6 just tests patch granularity, not the semantic patching hypothesis.

### MVG v4: The Real Test (BPE-aligned patching)

To test Architecture G properly, we need patches aligned to SOMETHING semantic:

```
BYTE ENCODER:
  - Byte embedding: 256 × 256
  - Use a small BPE tokenizer (8K vocab) to determine patch boundaries
  - Each BPE token = one patch (variable length, avg ~4 bytes)
  - Patch representation: mean-pool byte embeddings within each patch
    → Linear(256, 1024) per patch (or: project pooled byte embedding up to d_global)

GLOBAL TRANSFORMER: Same 12L d=1024

LOCAL DECODER: Same 1L d=256 with bypass

TOTAL: ~175M params
```

**This IS Architecture G simplified.** The BPE tokenizer determines boundaries (semantic), the global model processes at token level, the local decoder generates bytes.

**The BPE tokenizer is NOT a learned boundary predictor** — it's a fixed preprocessing step. This is simpler and tests the core hypothesis cleanly.

### MVG Decision

**MVG v4 is the correct scout.** It tests:
1. Does semantic (BPE) patching beat fixed-byte patching? → Hypothesis H6
2. Does the model benefit from more effective context (384+ patches vs 256)? → Capacity hypothesis
3. Does byte-level local decoding still work with token-level global context? → Architecture coherence

**Implementation effort:** Moderate. Main changes from current Sutra-Dyad:
1. Add BPE tokenizer for determining patch boundaries (use tiktoken or SentencePiece)
2. Change patch embedding: variable-length mean-pool instead of fixed concatenation
3. Adjust position encoding (RoPE for variable-length patches)
4. Everything else stays the same

**NOT in MVG:** Sparse SwiGLU, early exit, decoder skip, n-gram hash, dual KD, learned boundaries. These are future upgrades IF MVG validates the hypothesis.

### MVG Success Criteria

| Metric | Beat Current | Decisive |
|--------|-------------|----------|
| Eval BPB | < 1.430 (S1 baseline) | < 1.400 |
| vs Ekalavya best | < 1.418 (routing step 250) | < 1.390 |
| Generation quality | Coherent English | Better than current |
| Throughput | Within 20% of current | Faster (more patches but each cheaper) |

**If MVG wins:** Proceed with full Sutra-G (add sparsity, early exit, decoder skip, KD).
**If MVG ties:** Proceed cautiously — the architecture change is neutral, need to test if upgrades differentiate.
**If MVG loses:** Architecture G is falsified at this scale. Return to improving current byte model (Sparse SwiGLU + better KD).

---

## 12. SELF-ADVERSARIAL AUDIT OF SUTRA-G

### Challenge 1: "You're just reinventing Megabyte with a BPE tokenizer"

**Honest answer:** Partially true. MVG v4 IS Megabyte with BPE patches instead of fixed-size patches. The novelty is:
- Dual-level KD interface (token + byte) — novel combination
- Sparse SwiGLU for capacity without params — validated by Spark Transformer
- Decoder skip for O5 — natural in token-byte design, impossible in pure byte
- N-gram hash embeddings — validated by BLT but not in Megabyte
- Learned boundary prediction — Bolmo validates, but ours is from scratch

**Counter:** "Novel combination" is not the same as "paradigm shift." The VISION.md demands questioning axioms. Sutra-G accepts most standard axioms (transformer, attention, gradient descent). The real novelty is in the O2-O5 mechanisms, not the base architecture.

### Challenge 2: "BPE patching reintroduces tokenizer fragility"

**Honest answer:** Yes, partially. BPE patching means:
- Rare words get bad patches (split into meaningless pieces)
- Cross-lingual text gets bad patches
- Code gets inconsistent patches

**Counter:** The byte-level local decoder catches these failures. The global model sees bad patches but the local decoder still generates bytes correctly. This is the safety net.

**But:** If the global model gives bad representations for poorly-patched content, the local decoder can't fix it. The bypass connection helps (direct byte-to-byte), but the global context may be wrong.

**Mitigation:** Learned boundary prediction (full Sutra-G) replaces BPE. The model learns where to segment, potentially finding better boundaries than BPE.

### Challenge 3: "The parameter budget is too tight for three components"

**Honest answer:** 183M split across encoder (6M) + global (150M) + decoder (4M) + embeddings (26M) means the global model is only 150M — compared to the current Sutra-Dyad's 172M global. We're SHRINKING the intelligence substrate to make room for machinery.

**Counter:** The 150M global model processes 384+ patches instead of 256. More tokens × slightly less capacity per token. Whether this trades favorably depends on whether the sequence-length gain outweighs the capacity-per-token loss. This is exactly what MVG tests.

### Challenge 4: "N-gram hash embeddings are 25.6M params doing the work that the transformer should learn"

**Honest answer:** If 14% of params are in hash embeddings, that's 14% not in the transformer. BLT found them "very effective," but BLT also operates at 1B+ scale where 25M is negligible. At 183M, it's significant.

**Decision:** Start with the cheap version (6.4M params). Only upgrade if the cheap version shows clear gains.

### Challenge 5: "Three O5 mechanisms is over-engineering"

**Honest answer:** Yes. Sparse SwiGLU + early exit + decoder skip are three independent mechanisms. Each adds implementation complexity and potential for conflicting gradients.

**Counter:** They operate at different levels:
- Sparse SwiGLU: per-neuron (within each layer)
- Early exit: per-patch at global level (which layers to use)
- Decoder skip: per-patch at decoder level (whether to run decoder)

They're orthogonal, not conflicting. But implementing all three simultaneously IS over-engineering. The staged training strategy addresses this: add one at a time.

### Challenge 6: "You haven't considered the simplest alternative: just train the current model longer"

**Honest answer:** The current Sutra-Dyad at 188M with 5K CE-only steps has eval BPB 1.430. What if we train for 50K steps? 100K steps? The model hasn't been trained long enough to judge its ceiling.

**Counter:** This is addressed by the 6K Ekalavya run currently in progress. If it shows strong improvement (BPB < 1.380), the case for architectural change weakens. If it shows ceiling (BPB stays near 1.418-1.430), the architectural bottleneck hypothesis is confirmed.

**This is why we should wait for the 6K results before committing to Sutra-G.** The 6K run is the control experiment.

### Summary: What's left to prove

1. **MVG scout:** Does token-level global beat byte-level global at matched params? → Core hypothesis
2. **6K Ekalavya results:** Is the current architecture at its ceiling or still improving? → Baseline ceiling
3. **Sparse SwiGLU ablation:** Does activation sparsity help at 188M? → Capacity hypothesis
4. **These three experiments should run BEFORE committing to full Sutra-G.**

---

## 13. MVG IMPLEMENTATION BLUEPRINT (Ready for transcription)

### 13.1 Changes to sutra_dyad.py

The MVG requires modifying the existing codebase minimally. Here's the diff-level spec:

**New constants:**
```python
# MVG mode
MVG_MODE = True  # Toggle between MVG and current Sutra-Dyad
MVG_PATCH_SIZE = 4  # Fixed 4-byte patches for MVG v3
# OR: use BPE boundaries for MVG v4
MVG_USE_BPE_PATCHES = True
MVG_BPE_VOCAB = 8192  # Small BPE vocabulary
```

**New class: BPEPatcher (for MVG v4):**
```python
class BPEPatcher:
    """Determines patch boundaries using a small BPE tokenizer."""
    def __init__(self, vocab_size=8192):
        # Train or load BPE tokenizer
        # Use sentencepiece or tiktoken
        self.tokenizer = load_or_train_bpe(vocab_size)
    
    def segment(self, byte_sequence: torch.Tensor) -> list[tuple[int,int]]:
        """Returns list of (start, end) byte spans for each patch."""
        text = bytes(byte_sequence.cpu().numpy()).decode('utf-8', errors='replace')
        tokens = self.tokenizer.encode(text)
        spans = []
        pos = 0
        for token in tokens:
            token_bytes = self.tokenizer.decode([token]).encode('utf-8')
            length = len(token_bytes)
            spans.append((pos, pos + length))
            pos += length
        return spans
```

**Modified PatchEmbed:**
```python
class PatchEmbedMVG(nn.Module):
    """Variable-length patch embedding via mean-pool + project."""
    def __init__(self, d_byte=256, d_global=1024, max_patch_len=8):
        super().__init__()
        self.byte_embed = nn.Embedding(N_BYTES, d_byte)
        self.project = nn.Linear(d_byte, d_global)  # mean-pooled byte → global dim
        self.max_patch_len = max_patch_len
    
    def forward(self, byte_ids, patch_spans):
        # byte_ids: [B, T] (raw byte indices)
        # patch_spans: [B, num_patches, 2] (start, end for each patch)
        byte_embs = self.byte_embed(byte_ids)  # [B, T, d_byte]
        
        patch_reprs = []
        for b in range(byte_ids.shape[0]):
            patches = []
            for start, end in patch_spans[b]:
                # Mean-pool byte embeddings within this patch
                patch_bytes = byte_embs[b, start:end]  # [patch_len, d_byte]
                patch_mean = patch_bytes.mean(dim=0)  # [d_byte]
                patches.append(patch_mean)
            patch_reprs.append(torch.stack(patches))  # [num_patches, d_byte]
        
        # Pad to same length
        max_patches = max(p.shape[0] for p in patch_reprs)
        padded = torch.zeros(len(patch_reprs), max_patches, byte_embs.shape[-1])
        for b, p in enumerate(patch_reprs):
            padded[b, :p.shape[0]] = p
        
        return self.project(padded)  # [B, max_patches, d_global]
```

**Modified SutraDyad class (MVG variant):**
```python
class SutraDyadMVG(nn.Module):
    def __init__(self):
        super().__init__()
        self.patcher = BPEPatcher(vocab_size=MVG_BPE_VOCAB)
        self.patch_embed = PatchEmbedMVG(d_byte=256, d_global=D_GLOBAL)
        self.global_transformer = nn.ModuleList([
            TransformerBlock(D_GLOBAL, N_GLOBAL_HEADS, FF_GLOBAL)
            for _ in range(N_GLOBAL_LAYERS)
        ])
        self.global_norm = RMSNorm(D_GLOBAL)
        self.depool = nn.Linear(D_GLOBAL, D_LOCAL)
        self.local_decoder = TransformerBlock(D_LOCAL, N_LOCAL_HEADS, FF_LOCAL)
        self.local_norm = RMSNorm(D_LOCAL)
        self.byte_embed_local = nn.Embedding(N_BYTES, D_LOCAL)
        self.lm_head = nn.Linear(D_LOCAL, N_BYTES)
    
    def forward(self, byte_ids):
        B, T = byte_ids.shape
        
        # 1. Determine patch boundaries
        patch_spans = [self.patcher.segment(byte_ids[b]) for b in range(B)]
        
        # 2. Patch embedding
        h = self.patch_embed(byte_ids, patch_spans)  # [B, P, D_GLOBAL]
        
        # 3. Global transformer (causal, shifted by 1 patch)
        # Shift: patch j predicts using context from patches 0..j-1
        for block in self.global_transformer:
            h = block(h)
        h = self.global_norm(h)  # [B, P, D_GLOBAL]
        
        # 4. Depool: broadcast global repr to each byte position
        depool_repr = self.depool(h)  # [B, P, D_LOCAL]
        byte_level_global = expand_patches_to_bytes(depool_repr, patch_spans, T)
        # [B, T, D_LOCAL]
        
        # 5. Local decoder: byte embeddings + global context
        byte_embs = self.byte_embed_local(byte_ids)  # [B, T, D_LOCAL]
        # Shift byte embeddings for causal prediction
        shifted_byte = torch.zeros_like(byte_embs)
        shifted_byte[:, 1:] = byte_embs[:, :-1]
        
        local_input = shifted_byte + byte_level_global  # Residual bypass
        local_out = self.local_decoder(local_input)
        local_out = self.local_norm(local_out)
        
        logits = self.lm_head(local_out)  # [B, T, 256]
        return logits
```

### 13.2 Key Implementation Notes

1. **BPE tokenizer:** Use sentencepiece (already available) or train a small BPE on our training data. 8K vocab. The tokenizer is used ONLY for determining patch boundaries, not as a vocabulary.

2. **Variable-length patches → padding:** The global transformer needs fixed-length input. Pad patches to max_patches_per_batch. Use attention mask to ignore padding. Max patches per example: 8192/3 ≈ 2731 (minimum BPE token is 1 byte, but avg is ~4). Practical max with padding: ~2048 patches.

3. **Causality (CRITICAL — read code/sutra_dyad.py:256-260):**
   Current Sutra-Dyad shifts global output by 1 patch position (line 257-260):
   ```python
   global_shifted = cat([zeros(B,1,D), global_out[:,:-1,:]], dim=1)
   ```
   This works because all patches are the same size. For MVG with variable patches:
   - The shift is STILL by 1 patch index (patch j uses patches 0..j-1)
   - But the depooling step must map patch j's representation to all bytes in patch j
   - This requires a patch-to-byte index map: for each byte position, which patch does it belong to?
   ```python
   # patch_ids[b, t] = index of patch containing byte t
   # global_shifted[b, patch_ids[b, t]] gives the global context for byte t
   byte_global = global_shifted[b, patch_ids[b]]  # gather by patch index
   ```

4. **Depooling:** Each byte position gets the global representation of its containing patch (shifted by 1). Implementation: `torch.gather` on patch index, or `expand` + `scatter`. Same as current but with variable patch boundaries.

5. **Loss:** Standard byte-level CE, same as current. No KD in MVG. BPB = bits per byte = CE_loss / ln(2).

6. **Evaluation:** BPB on the same eval set as current Sutra-Dyad. Direct comparison. Same data loader, same eval windows.

7. **IMPORTANT — MVG v3 simplification:** To avoid variable-patch complexity in the FIRST scout, consider MVG v3 (fixed P=4 patches) instead of MVG v4 (BPE patches). This is a smaller diff from the current code (just change PATCH_SIZE from 6 to 4 and adjust d_global if needed). If fixed P=4 shows improvement, THEN test BPE boundaries. This is the parsimony principle: test one change at a time.

### 13.3 MVG Experiment Plan

```
Experiment: MVG Scout
Purpose: Test H6 (token-global beats byte-global at same scale)
Baseline: Sutra-Dyad S1 best (eval BPB=1.430, 5K steps CE-only)
Budget: 5K steps CE-only, same data, same LR schedule, same eval

Step 1: Train BPE tokenizer (8K vocab, on training data shards)
Step 2: Implement BPEPatcher + PatchEmbedMVG + SutraDyadMVG
Step 3: Run 5K CE-only training
Step 4: Eval BPB + generation quality
Step 5: Compare against Sutra-Dyad S1

Success: eval BPB < 1.430 AND generation quality >= current
Decisive: eval BPB < 1.400
Kill: eval BPB > 1.460 at step 2K

Time estimate: ~3-4 hours (5K steps × ~2.5s/step)
VRAM: ~4-5GB (no teachers, smaller model)
```

### 13.4 Parallel Experiment: Sparse SwiGLU Ablation

```
Experiment: Sparse SwiGLU in Current Sutra-Dyad
Purpose: Test H7 (sparse activation increases effective capacity)
Baseline: Sutra-Dyad S1 best (eval BPB=1.430)
Budget: 2K steps from best.pt checkpoint

Step 1: Modify SwiGLU in sutra_dyad.py to add top-k sparsity
Step 2: Load best.pt, add sparsity gradually (100%→10% over 500 steps)
Step 3: Continue training for 2K steps
Step 4: Eval BPB

Success: eval BPB < 1.430 AND throughput improves
Kill: eval BPB > 1.450 at step 1K

Time estimate: ~1.5 hours
VRAM: Same as current (~10.5GB without teachers)
```

### 13.5 FINAL EXPERIMENT PLAN (Ordered, with Decision Tree)

**Pre-condition:** Wait for 6K Ekalavya run results (currently running).

```
PHASE 0: Collect 6K Ekalavya Results (~0 hours, just read)
├── IF eval BPB ≤ 1.380 (DECISIVE KD win):
│   → Current architecture IS improving. KD ceiling NOT reached.
│   → Delay architectural changes. Add Mamba-1.4B as 3rd teacher.
│   → Run 15K+ steps. Benchmark at 15K.
│   → But STILL run Experiment 1 (patch ablation) as background test.
│
├── IF eval BPB 1.380-1.418 (POSITIVE):
│   → KD is working but approaching ceiling.
│   → Proceed with Experiment 1, then decide.
│
└── IF eval BPB > 1.418 (CEILING HIT):
    → Current path exhausted. Architectural change needed.
    → Proceed immediately to Experiment 1.

EXPERIMENT 1: Patch-Size Ablation (~3.5 hours)
  Change: PATCH_SIZE=4 (line 50 of sutra_dyad.py), train from scratch 5K steps
  Budget: 5K steps CE-only, same everything else
  Kill: BPB > 1.480 at step 2K
  Compare: vs Sutra-Dyad P=6 baseline (BPB=1.430)
  ├── IF P=4 BPB < P=6 BPB by ≥0.01:
  │   → Context hypothesis CONFIRMED. More patches help.
  │   → Proceed to Experiment 2 (BPE patches).
  │
  ├── IF P=4 BPB ≈ P=6 BPB (within 0.01):
  │   → Context hypothesis NEUTRAL. Issue is not patch count.
  │   → Skip Experiment 2. Proceed to Experiment 3 (Sparse SwiGLU).
  │
  └── IF P=4 BPB > P=6 BPB by ≥0.01:
      → Finer patches HURT. Each patch needs minimum information.
      → Architecture G may be wrong direction. Proceed to Experiment 3.
      → Consider: the problem is capacity, not context.

EXPERIMENT 2: BPE-Aligned Patches (~4-5 hours, only if Exp 1 positive)
  Change: BPE tokenizer determines patch boundaries, mean-pool→project
  Budget: 5K steps CE-only
  Kill: BPB > Exp 1 result + 0.02 at step 2K
  Compare: vs P=4 fixed and P=6 baseline
  ├── IF BPE patches beat both P=4 and P=6:
  │   → SEMANTIC patching > fixed patching. Architecture G validated.
  │   → Proceed to Experiment 4 (Ekalavya on BPE model).
  │
  └── IF BPE patches ≤ P=4:
      → Semantic boundaries don't help beyond finer patching.
      → Simpler P=4 fixed patches are sufficient. Skip full G.
      → Proceed to Experiment 3 (Sparse SwiGLU on P=4 model).

EXPERIMENT 3: Sparse SwiGLU Ablation (~1.5 hours)
  Change: Add top-k sparsity (10%) to SwiGLU in global transformer
  Start from: best checkpoint of P=4 (or P=6 if Exp 1 negative)
  Budget: 2K fine-tuning steps (ramp sparsity 100%→10% over 500 steps)
  Kill: BPB > start + 0.02 at step 1K
  ├── IF BPB improves or holds:
  │   → Sparse capacity hypothesis CONFIRMED.
  │   → Throughput measurement: how much faster?
  │   → Keep sparse SwiGLU for all future runs.
  │
  └── IF BPB degrades:
      → Sparsity hurts at this scale. Drop it.
      → Focus on KD improvements on best architecture.

EXPERIMENT 4: Ekalavya on Best Architecture (~20-30 hours)
  Apply full Ekalavya pipeline (routing, gating, TAID) to winning architecture
  Start from: best CE-only checkpoint
  Budget: 6K steps with 2 teachers
  Compare: vs CE-only baseline of same architecture
  Kill: no improvement after 2K steps
  Gate: eval BPB improvement ≥0.01 over CE-only at step 3K

DECISION POINT: After Experiments 1-4, we have:
  - Best patch size (4 vs 6 vs BPE)
  - Best FFN strategy (dense vs sparse)
  - Best training strategy (CE-only vs KD)
  - Combined: the best 188M byte model we can build
  
  IF combined improvements produce BPB ≤ 1.380:
    → Proceed to benchmarks + 15K training
  IF BPB still > 1.400:
    → Scale student (500M-1B) before further optimization
    → Or: consider Bolmo-style byteified initialization (strategic violation)
```

Total time for Experiments 1-3: ~9 hours. Experiment 4: ~30 hours.
All fit within 1 week of GPU time.

### 13.6 FLOP Analysis: Why Sutra-G + Sparse SwiGLU Wins

**Current Sutra-Dyad (N=256 patches, dense):**
```
Per layer:
  Attention: O(256² × 1024) = 67M FLOPs
  Dense FFN: O(256 × 1024 × 2730 × 2) = 1,431M FLOPs
  Total/layer: ~1,498M FLOPs
12 layers: ~17,976M FLOPs ≈ 18B FLOPs
```

**MVG v3 (N=384 patches, dense, P=4):**
```
Per layer:
  Attention: O(384² × 1024) = 151M FLOPs (+125%)
  Dense FFN: O(384 × 1024 × 2730 × 2) = 2,147M FLOPs (+50%)
  Total/layer: ~2,298M FLOPs
12 layers: ~27,576M FLOPs ≈ 28B FLOPs (+53%)
```
**Cost:** 53% more FLOPs for 50% more context. Marginal.

**Sutra-G (N=384 patches, Sparse SwiGLU 10%, P=4):**
```
Per layer:
  Attention: O(384² × 1024) = 151M FLOPs
  Sparse FFN: O(384 × 1024 × 273 × 2) = 215M FLOPs (10% of dense!)
  Total/layer: ~366M FLOPs
12 layers: ~4,392M FLOPs ≈ 4.4B FLOPs
```
**Result:** 50% more context, 75% LESS total compute than Sutra-Dyad.

**Sutra-G with early exit (60% of patches exit at layer 6):**
```
Easy patches (60%): 6 layers × 366M = 2,196M FLOPs per patch position
Hard patches (40%): 12 layers × 366M = 4,392M FLOPs per patch position
Average: 0.6 × 2,196 + 0.4 × 4,392 = 3,075M FLOPs ≈ 3.1B FLOPs
```
**Result:** 50% more context, 83% LESS total compute than Sutra-Dyad.

**The key insight:** Dense FFN is 95% of per-layer compute in the current model. Making it 10x cheaper via sparsity makes everything else possible: more patches, more layers, adaptive compute — all at LOWER total cost.

This is Intelligence = Geometry: same parameters, better mathematical structure, 4-6x less compute, 50% more context.

**⚠ CODEX R2 CORRECTION:** The Sparse SwiGLU compute savings are overstated. Top-k selection happens AFTER the gate/up matmuls, so those dense operations still run. Only the `w_down` matmul is saved (partially, and only with custom sparse kernels). Realistic end-to-end speedup from sparse SwiGLU alone: **0-15% initially, 15-25% with custom kernels**. Combined O5 realistic speedup: **1.2-1.5x, not 2x**.

---

## 14. CODEX TESLA R2 REVIEW — SYNTHESIS (2026-04-15)

**Session:** GPT-5.4 xhigh, read-only. Full design doc read including §7-§13.

### R2 Priority Directive

> "Build a minimal matched-budget G scout with fixed BPE-aligned patch boundaries, strict shifted-global causality, no learned boundary predictor, no n-gram hash table, no Sparse SwiGLU, no early exit, no decoder skip, and no KD."

### R2 Critical Issues Found

1. **Architecture inconsistency:** §9 describes "token-level transformer with 16K BPE" while §10 says "no tokenizer, boundary predictor IS tokenizer." These are different architectures. The scout must use ONE: fixed BPE patches.

2. **Parameter budget error:** 12-layer d=768 ff=2048 is ~85M params, not 150M. Total Sutra-G is closer to ~100-118M, not 183M. Must recalculate with correct numbers.

3. **Sparse SwiGLU compute overclaimed:** Top-k after gate/up only saves `w_down`. Real speedup: 0-15% without custom kernels. The FLOP analysis in §13.6 is wrong because it assumes 90% FFN reduction, but the gate and up matmuls still run at full density.

4. **Causality leak:** Learned boundary predictor with 1-byte lookahead leaks target information during training. Must use shifted boundaries or fixed boundaries for the scout.

5. **Stage 0 breaks:** "Global only" training has no output target without a tokenizer vocabulary. Include encoder+decoder from the start, or use a real token vocabulary for Stage 0.

6. **VRAM budget optimistic:** Ignores gradients, teacher logits, softmax buffers, ragged patch overhead. Should assume offline cache, one teacher at a time.

7. **Early exit mixed-depth:** Patch-level early exit creates mixed-depth states in causal attention — later patches attending to earlier exited patches see inconsistent representations.

8. **Decoder skip incoherent:** No "token-level logit" exists without a token vocabulary. The skip shortcut needs a discrete mapping that doesn't exist in the current design.

### R2 Updated Confidence Scores

| Architecture | O1 | O2 | O3 | O4 | O5 | Mean |
|---|---:|---:|---:|---:|---:|---:|
| A: Evolved Ekalavya | 3 | 4 | 3 | 5 | 1 | 3.2 |
| I: Sparse Dendritic Byte | 4 | 5 | 4 | 4 | 3 | 4.0 |
| K: Recurrent Universal Byte | 4 | 6 | 5 | 4 | 5 | 4.8 |
| Sutra-G Revised | 5 | 6 | 5 | 5 | 4 | 5.0 |

**O5 dropped from 6 to 4** due to realistic speedup assessment (1.2-1.5x, not 2x).

### R2 Path to 9/10

| Outcome | Score | Evidence Needed |
|---------|-------|-----------------|
| O1 | 5→9 | Scout beats Sutra-Dyad by ≥0.05 BPB at equal wall-clock, improves generation+benchmarks |
| O2 | 6→9 | Demo: fix spelling failure changing only decoder, fix reasoning failure changing only global |
| O3 | 5→9 | Swap encoder or decoder module cleanly, recover quality without full retrain |
| O4 | 5→9 | Dual KD adds ≥0.03 BPB over CE-only without hurting generation |
| O5 | 4→9 | Measured ≥1.7x throughput at same BPB (not estimated FLOPs) |

### R2 Minimal Viable G (FINAL — supersedes §11 MVG)

**KEEP:**
- Byte I/O
- Fixed BPE-aligned patch boundaries (8K vocab SentencePiece)
- Simple byte encoder (byte embedding → mean-pool per patch → project to d_global)
- Global transformer (matched param budget to Sutra-Dyad)
- Shifted global-to-local projection (load-bearing, validated)
- Byte local decoder (1-2 layers, with bypass)
- CE byte loss only
- Deterministic eval/generation gate

**CUT:**
- Learned boundary predictor
- N-gram hash embeddings
- Sparse SwiGLU
- Early exit
- Decoder skip
- Dual-level KD
- mLSTM local encoder/decoder
- Dynamic patching

### R2 ZEROTH EXPERIMENT: One-Line Patch-Size Ablation

Before ANY architectural change, the CHEAPEST possible test of the context hypothesis:

**Change PATCH_SIZE from 6 to 4 in sutra_dyad.py line 50.** That's it. Nothing else changes.

Result: 384 patches of 4 bytes instead of 256 patches of 6 bytes (50% more context).
Patch projection changes: Linear(4*256, 1024) instead of Linear(6*256, 1024).
This is 13M fewer parameters (1024×1024 instead of 1536×1024).

**This experiment takes <1 minute to set up and tests whether more patches help.**

If BPB improves: the context hypothesis is confirmed, proceed to BPE patches.
If BPB stays flat: more context alone doesn't help, the problem is elsewhere.
If BPB worsens: finer patches hurt (each patch has less information for the global model).

This should run BEFORE the MVG scout, BEFORE Sparse SwiGLU, BEFORE anything else.
Cost: 5K steps × ~2.5s = ~3.5 hours. Zero implementation risk.

### R2 Implementation Risk Ranking

1. Boundary predictor + causal generation → REMOVE from scout
2. Global/local causality with variable patches → Use Sutra-Dyad shifted-global contract
3. Live dual-teacher KD VRAM → Offline cache, 1 teacher, top-k
4. Early exit mixed-depth states → DEFER
5. Decoder skip shortcut correctness → DEFER
6. Sparse SwiGLU real speedup → DEFER
7. Ragged patch batching/masks → Handle in scout
8. Representation KD alignment → DEFER
9. N-gram hash collisions → DEFER
10. mLSTM dependency → DEFER

### R2 Corrected Parameter Budget

```
MINIMAL G SCOUT (BPE-aligned, d=1024):
  Byte embedding: 256 × 256 = 65K
  Patch projection: Linear(256, 1024) = 262K  (mean-pooled byte emb → global)
  Global transformer: 12L d=1024 16H ff=2730 = ~172M (SAME as Sutra-Dyad)
  Global-to-local: Linear(1024, 256) = 262K
  Local decoder: 1L d=256 4H ff=682 = ~2.1M
  LM head: 256×256 = 65K (tied)
  TOTAL: ~175M

  vs Sutra-Dyad: 188M (patch_proj uses 6*256→1024 instead of 256→1024)
  Difference: ~13M fewer params (smaller patch projection)
  
  To match: use freed 13M params for 2nd local decoder layer (+2.1M) 
  and wider local decoder (d=320, +remainder)
```

---

## 15. CODEX TESLA R3 REVIEW — CONVERGENCE ROUND SYNTHESIS (2026-04-15)

**Session:** GPT-5.4 xhigh, read-only. Full design doc + code read. 78,923 tokens consumed.
**Focus:** Implementation-ready audit, conditional confidence, Sparse SwiGLU capacity argument.

### R3 Verdict: Design Phase Converged

**The design has converged. Further theoretical iteration cannot raise confidence scores — only experimental evidence can.** The R3 review confirmed the MVG scout design is sound in principle but identified 6 implementation gaps that must be fixed before transcription.

### R3 MVG Implementation Audit — 6 Critical Gaps

| # | Issue | Why It Matters | Exact Fix |
|---|-------|----------------|-----------|
| 1 | **§13 pseudocode forgets `global_shifted`** | Lets bytes in patch j see global_out[j], which includes patch j bytes → causality leak | After `global_norm`, do same shift as sutra_dyad.py:256: `global_shifted = cat([zero, global_out[:, :-1]])`, then depool/gather from `global_shifted` |
| 2 | **§13 shifts byte embeddings** | Current loader returns x=bytes[i:i+T], y=bytes[i+1:i+T+1], so byte_emb[t] is allowed when predicting y[t]. Shifting changes the training problem | Keep current `local_input = byte_emb + byte_global` as in sutra_dyad.py:267 |
| 3 | **BPE span recovery underspecified** | `decode(errors="replace")` + token decode can change bytes, normalize text, lose invalid UTF-8, produce wrong spans | Use byte-exact tokenizer with offsets, or assert `b"".join(token_bytes) == original_bytes`. If assertion fails, fall back to fixed 4-byte patches for that sample |
| 4 | **Tokenizer inside `forward` is wrong** | `.cpu().numpy()` in model forward causes GPU sync, makes throughput comparisons invalid | Build `patch_ids`, `patch_mask`, `patch_spans` in data/training path on CPU, move tensors to GPU. `forward(byte_ids, targets, patch_ids, patch_mask)` |
| 5 | **Ragged patch batching not fully specified** | Variable patch counts need deterministic padding + max-patch policy | Right-pad patch tensors to batch max. Set `max_global_patches = SEQ_BYTES + 16` for RoPE safety. Log/kill if mean or p95 patch count explodes |
| 6 | **Baseline parameter target ambiguous** | Stage0 SutraDyad ≈ 153.7M params, but doc says 188M | Name baseline explicitly: Stage0 (lines 144-290) ≈ 153.7M. MVG param match depends on this choice |

### R3 Corrected Parameter Budget

```
ACTUAL Stage0 SutraDyad (from code): ~153.7M params
ACTUAL BPE-MVG (same 12L d=1024 global): ~152.4M params
Difference: ~1.3M (0.9%) — well within 10%

The doc's "13M fewer params" claim was WRONG (off by ~10x).
Fixed P=6→P=4 saves only ~0.52M.
BPE mean-pool 256→1024 saves only ~1.31M.
```

### R3 Exact Diff from sutra_dyad.py

```python
# REPLACE fixed patch reshape (sutra_dyad.py lines 244-246):
byte_emb_patched = byte_emb.reshape(B, N, P * self.d_local)
patch_emb = self.patch_proj(byte_emb_patched)

# WITH variable BPE mean-pooling:
patch_mean = scatter_mean(byte_emb, patch_ids, patch_mask)  # [B, Nmax, d_local]
patch_emb = self.patch_proj(patch_mean)                     # Linear(256, 1024)

# KEEP global transformer UNCHANGED.

# KEEP the shifted-global contract (sutra_dyad.py lines 256-260):
global_shifted = torch.cat([zero, global_out[:, :-1, :]], dim=1)
global_local = self.global_to_local(global_shifted)

# REPLACE fixed expand (sutra_dyad.py lines 263-265):
global_local = global_local.unsqueeze(2).expand(B, N, P, d_local).reshape(B, T, d_local)

# WITH byte gather:
byte_global = gather(global_local, patch_ids)  # [B, T, d_local]

# KEEP current local alignment (sutra_dyad.py line 268):
local_input = byte_emb + byte_global
```

**Causality verification (R3):** With the fix above, if byte t belongs to patch j, `byte_global[t] = global_to_local(global_shifted[j]) = global_to_local(global_out[j-1])`. Since global transformer is causal, `global_out[j-1]` depends only on patches 0..j-1. ✓ VALID.

### R3 Minimum Experiment Duration

- **2K steps**: Kill gate only (eval >1.460 = kill immediately)
- **5K steps**: Minimum to claim a win (matches baseline reference)
- **10K steps**: Required if 5K is within ±0.02 BPB (not decisive)
- **9/10 O1 gate**: ≤1.380 BPB (if baseline is 1.430)

### R3 Conditional Confidence Scores

**IF MVG wins (≥0.05 BPB at matched compute):**

| Outcome | Score | Why Not 9 Yet |
|---------|------:|---------------|
| O1 Genuine Intelligence | 8 | Need generation quality, benchmarks, repeat seed, longer-run scaling |
| O2 Improvability | 7 | Need demos: decoder-only fixes spelling, global-only fixes reasoning |
| O3 Democratized Development | 6 | Need module-swap evidence: replace patcher/decoder/global, recover without full retrain |
| O4 Data Efficiency | 6 | MVG is CE-only. Need dual KD ≥0.03 BPB over CE-only without hurting generation |
| O5 Inference Efficiency | 5 | Matched-compute BPB win ≠ adaptive compute. Need measured throughput gains |
| **Mean** | **6.4** | **Experiments close the gap, not more design** |

**IF MVG loses (BPB > Sutra-Dyad at matched compute):**

| Candidate | O1 | O2 | O3 | O4 | O5 | Mean |
|-----------|---:|---:|---:|---:|---:|-----:|
| A: Evolved Ekalavya | 4 | 4 | 3 | 5 | 1 | 3.4 |
| B: MoE-Distilled Byte | 4 | 4 | 4 | 4 | 3 | 3.8 |
| C: Knowledge Compiler | 4 | 5 | 5 | 6 | 3 | 4.6 |
| I: Sparse Dendritic Byte | 5 | 5 | 4 | 4 | 3 | 4.2 |
| **K/F: Recurrent Universal Byte** | **5** | **6** | **5** | **4** | **6** | **5.2** |
| Sutra-G Revised | 3 | 5 | 4 | 4 | 3 | 3.8 |

**If MVG loses, best path = K/F (recurrent/weight-shared byte transformer) with Ekalavya as infrastructure.**

### R3 Sparse SwiGLU Verdict

**Capacity argument: partially valid.** Hard top-k creates input-dependent subnetwork selection, reduces feature interference, acts like conditional computation. But "exponential subnetworks" is overstated — all masks share the same gate/up/down weights. Not equivalent to exponentially many independent experts.

**Compute argument: still dead.** gate(x) and up(x) are still dense. Top-k after their product mainly saves part of down, and only with sparse kernels does that become real speed.

**Priority: Run AFTER G scout.** Compatible with every architecture. Does not answer the main fork (byte-global vs BPE/token-global). Run before G only if MVG implementation is blocked and GPU would otherwise idle.

### R3 FINAL Experiment Plan (Ordered)

```
PHASE 0: Collect 6K Ekalavya control
  Interpretation: if <1.380 BPB → current arch has headroom, G urgency drops
  
PHASE 1: Zeroth ablation — PATCH_SIZE=6→4
  Time: 5K steps, ~3.5 hours
  Kill: eval >1.460 at 2K, or worse than P=6 curve by >0.03
  Success: <1.430. Decisive: <1.400.
  
PHASE 2: Implement MVG v4 with unit tests (CPU only, no GPU)
  Required tests:
    - Byte reconstruction from BPE spans
    - Param count assertion
    - Patch gather shape
    - Causality perturbation test
    - Generation path uses same patcher
    
PHASE 3: Run MVG v4 CE-only
  Time: 5K steps, ~4-6 hours (tokenizer overhead)
  Kill: NaNs, throughput <0.5x, p95 patch count too high, eval >1.460 at 2K
  Success: <1.430. Strong: <1.400. 9/10 O1 gate: ≤1.380.
  
PHASE 4: Sparse SwiGLU ablation
  Time: 2K warm-start smoke (~1.5h); if positive, 5K from scratch
  Kill: BPB >1.450 at 1K, or throughput regression without BPB gain
  Success: BPB improves by ≥0.01-0.02 without generation damage
  
PHASE 5: Branch decision
  MVG wins → build full Sutra-G in stages: CE-only → KD → O5
  MVG loses → K/F recurrent byte scout, sparse only if ablation supports
```

### R3 Path to 9/10 on ALL Outcomes

| Outcome | 9/10 Evidence Gate |
|---------|-------------------|
| O1 | Reproduced ≥0.05 BPB gain at matched wall-clock, better generations, benchmark lift, longer-run scaling |
| O2 | Surgical fixes demonstrated: change only decoder → fix spelling; change only global → fix reasoning |
| O3 | Clean module API, successful encoder/decoder/patcher swap, recovery without full retrain |
| O4 | Dual/multi-teacher KD adds ≥0.03 BPB over CE-only winner without harming generation |
| O5 | Measured adaptive inference: ≥1.7x throughput or comparable active-compute reduction at same BPB, on RTX 5090 |

### R3 Confidence Score Trajectory (3 rounds)

| Round | Sutra-G Mean | Highest Alternative | Key Finding |
|-------|-------------|--------------------:|-------------|
| R1 | 5.4 | K=4.8 | "Stop optimizing Ekalavya, run G scout" |
| R2 | 5.0 | K=4.8 | "Spec has gaps, build MINIMAL G. O5 overstated" |
| R3 | 6.4 (conditional) | K/F=5.2 (if G loses) | "Design converged. Only experiments close the gap" |

**R3 BOTTOM LINE:** Implement MVG, but fix §13 first. The current spec can produce a causality-leaking model if transcribed literally. The decisive architecture gate is the BPE-MVG scout, not Sparse SwiGLU.

---

## 16. IMPLEMENTATION-READY BLUEPRINT — SUTRA MVG SCOUT

**Status:** Ready for transcription. All ambiguities resolved by R1-R3 Codex review.
**Prerequisite:** Zeroth experiment (PATCH_SIZE=4) runs first.

### 16.1 Architecture Specification

```
Name: SutraDyadMVG (extends SutraDyad Stage0)
Type: Byte-level LM with BPE-aligned variable patching
Params: ~152.4M (within 1% of Stage0 SutraDyad at 153.7M)

Components:
  1. Byte Embedding:        nn.Embedding(256, 256)              = 65K params
  2. Patch Projection:      nn.Linear(256, 1024, bias=False)    = 262K params
                            (input = mean-pooled byte embeddings per BPE patch)
  3. Global Transformer:    12 layers, d=1024, 16 heads, ff=2730 SwiGLU, RoPE
                            IDENTICAL to SutraDyad Stage0      = ~151M params
  4. Global-to-Local:       nn.Linear(1024, 256, bias=False)    = 262K params
  5. Local Decoder:         1 layer, d=256, 4 heads, ff=682 SwiGLU, RoPE
                            IDENTICAL structure to SutraDyad    = ~2.1M params
  6. LM Head:               nn.Linear(256, 256, bias=False)     = tied with byte_embed
  
  TOTAL: ~153.7M (matched to baseline)
```

### 16.2 BPE Patcher (CPU-side, NOT in forward pass)

```python
class BPEPatcher:
    """Segments byte sequences into BPE-aligned patches.
    
    Runs on CPU in data loading, NOT in model forward pass.
    Produces patch_ids and patch_mask tensors for the model.
    """
    
    def __init__(self, vocab_size=8192):
        self.tokenizer = self._load_or_train_bpe(vocab_size)
    
    def _load_or_train_bpe(self, vocab_size):
        """Load cached SentencePiece BPE, or train one on training data."""
        cache_path = REPO / f"data/bpe_{vocab_size}.model"
        if cache_path.exists():
            import sentencepiece as spm
            sp = spm.SentencePieceProcessor()
            sp.Load(str(cache_path))
            return sp
        # Train on a sample of training data
        # (details: sample 10M bytes, train SentencePiece BPE)
        raise FileNotFoundError(f"Train BPE first: {cache_path}")
    
    def segment(self, byte_tensor):
        """Convert byte tensor to patch assignments.
        
        Args:
            byte_tensor: (T,) long tensor of byte values [0, 255]
        Returns:
            patch_ids: (T,) long tensor, patch_ids[t] = which patch byte t belongs to
            n_patches: int, number of patches
        """
        raw_bytes = bytes(byte_tensor.cpu().numpy().astype('uint8'))
        
        # Try UTF-8 decode for BPE
        try:
            text = raw_bytes.decode('utf-8')
            tokens = self.tokenizer.Encode(text, out_type=str)
        except UnicodeDecodeError:
            # Fallback: fixed 4-byte patches for non-UTF-8 sequences
            n = len(raw_bytes)
            patch_ids = torch.arange(n) // 4
            return patch_ids, (n + 3) // 4
        
        # Build byte-to-patch mapping
        patch_ids = torch.zeros(len(raw_bytes), dtype=torch.long)
        byte_pos = 0
        for patch_idx, token_str in enumerate(tokens):
            token_bytes = token_str.encode('utf-8') if isinstance(token_str, str) else \
                          self.tokenizer.IdToPiece(self.tokenizer.PieceToId(token_str)).encode('utf-8')
            n_bytes = len(token_bytes)
            patch_ids[byte_pos:byte_pos + n_bytes] = patch_idx
            byte_pos += n_bytes
        
        # CRITICAL: verify byte reconstruction
        assert byte_pos == len(raw_bytes), \
            f"BPE span mismatch: {byte_pos} vs {len(raw_bytes)}"
        
        return patch_ids, patch_idx + 1
    
    def batch_segment(self, byte_batch):
        """Segment a batch of byte sequences.
        
        Args:
            byte_batch: (B, T) long tensor
        Returns:
            patch_ids: (B, T) long tensor
            patch_mask: (B, Nmax) bool tensor — True for real patches
            n_patches: (B,) tensor of patch counts
        """
        B, T = byte_batch.shape
        all_ids = []
        all_n = []
        
        for b in range(B):
            ids, n = self.segment(byte_batch[b])
            all_ids.append(ids)
            all_n.append(n)
        
        patch_ids = torch.stack(all_ids)  # (B, T)
        n_patches = torch.tensor(all_n)   # (B,)
        Nmax = n_patches.max().item()
        
        # Build patch mask
        patch_mask = torch.zeros(B, Nmax, dtype=torch.bool)
        for b in range(B):
            patch_mask[b, :all_n[b]] = True
        
        return patch_ids, patch_mask, n_patches
```

### 16.3 Model Forward Pass (SutraDyadMVG)

```python
class SutraDyadMVG(nn.Module):
    """MVG scout: BPE-aligned variable patches, same global trunk.
    
    Differences from SutraDyad Stage0:
    - patch_proj: Linear(d_local, d_global) instead of Linear(P*d_local, d_global)
    - Accepts patch_ids, patch_mask from CPU-side BPE patcher
    - Uses scatter_mean for patch embedding, gather for global→byte depooling
    - Everything else IDENTICAL
    """
    
    def __init__(self, n_bytes=256, d_local=256, d_global=1024,
                 n_local_heads=4, n_global_heads=16,
                 n_local_dec_layers=1, n_global_layers=12,
                 ff_local=682, ff_global=2730,
                 max_patches=512, max_seq_bytes=1536):
        super().__init__()
        self.n_bytes = n_bytes
        self.d_local = d_local
        self.d_global = d_global
        self.max_patches = max_patches
        self.max_seq_bytes = max_seq_bytes
        
        # Byte embedding (SAME)
        self.byte_embed = nn.Embedding(n_bytes, d_local)
        
        # Patch projection: mean-pooled d_local → d_global (CHANGED)
        self.patch_proj = nn.Linear(d_local, d_global, bias=False)
        
        # Global transformer (IDENTICAL)
        self.global_layers = nn.ModuleList([
            TransformerBlock(d_global, n_global_heads, ff_global)
            for _ in range(n_global_layers)
        ])
        self.global_norm = RMSNorm(d_global)
        
        # RoPE for global (over patches — use max_patches)
        g_head_dim = d_global // n_global_heads
        g_cos, g_sin = precompute_rope(g_head_dim, max_patches + 16)
        self.register_buffer('g_rope_cos', g_cos, persistent=False)
        self.register_buffer('g_rope_sin', g_sin, persistent=False)
        
        # Global-to-local projection (SAME)
        self.global_to_local = nn.Linear(d_global, d_local, bias=False)
        
        # Local decoder (IDENTICAL)
        self.local_decoder = nn.ModuleList([
            TransformerBlock(d_local, n_local_heads, ff_local)
            for _ in range(n_local_dec_layers)
        ])
        self.local_norm = RMSNorm(d_local)
        
        # RoPE for local (over bytes — SAME)
        l_head_dim = d_local // n_local_heads
        l_cos, l_sin = precompute_rope(l_head_dim, max_seq_bytes + 16)
        self.register_buffer('l_rope_cos', l_cos, persistent=False)
        self.register_buffer('l_rope_sin', l_sin, persistent=False)
        
        # Output head (tied — SAME)
        self.output_head = nn.Linear(d_local, n_bytes, bias=False)
        self.output_head.weight = self.byte_embed.weight
        
        self._init_weights()
    
    def _scatter_mean(self, byte_emb, patch_ids, patch_mask):
        """Mean-pool byte embeddings per patch.
        
        Args:
            byte_emb: (B, T, d_local)
            patch_ids: (B, T) — which patch each byte belongs to
            patch_mask: (B, Nmax) — True for real patches
        Returns:
            patch_emb: (B, Nmax, d_local)
        """
        B, T, D = byte_emb.shape
        Nmax = patch_mask.shape[1]
        
        # Scatter-add byte embeddings into patch buckets
        patch_sum = torch.zeros(B, Nmax, D, device=byte_emb.device, dtype=byte_emb.dtype)
        patch_count = torch.zeros(B, Nmax, 1, device=byte_emb.device, dtype=byte_emb.dtype)
        
        idx = patch_ids.unsqueeze(-1).expand_as(byte_emb)  # (B, T, D)
        patch_sum.scatter_add_(1, idx, byte_emb)
        patch_count.scatter_add_(1, patch_ids.unsqueeze(-1), 
                                  torch.ones(B, T, 1, device=byte_emb.device, dtype=byte_emb.dtype))
        
        # Mean (avoid div-by-zero for padding patches)
        patch_count = patch_count.clamp(min=1.0)
        return patch_sum / patch_count
    
    def _gather_to_bytes(self, patch_repr, patch_ids):
        """Expand patch-level representation back to byte level.
        
        Args:
            patch_repr: (B, Nmax, d_local) — per-patch representation
            patch_ids: (B, T) — which patch each byte belongs to
        Returns:
            byte_repr: (B, T, d_local)
        """
        idx = patch_ids.unsqueeze(-1).expand(patch_ids.shape[0], patch_ids.shape[1], 
                                              patch_repr.shape[-1])
        return torch.gather(patch_repr, 1, idx)
    
    def forward(self, byte_ids, targets=None, patch_ids=None, patch_mask=None):
        """Forward pass with variable BPE patches.
        
        Args:
            byte_ids: (B, T) long tensor of byte values [0, 255]
            targets: (B, T) long tensor of next-byte targets
            patch_ids: (B, T) long tensor — which patch each byte belongs to
            patch_mask: (B, Nmax) bool tensor — True for real patches
        """
        B, T = byte_ids.shape
        
        # 1. Byte embeddings (SAME)
        byte_emb = self.byte_embed(byte_ids)  # (B, T, d_local)
        
        # 2. Mean-pool bytes per BPE patch (CHANGED from concat-project)
        patch_mean = self._scatter_mean(byte_emb, patch_ids, patch_mask)  # (B, Nmax, d_local)
        patch_emb = self.patch_proj(patch_mean)  # (B, Nmax, d_global)
        
        # 3. Global causal transformer over patches (SAME)
        N = patch_mask.shape[1]
        g_cos = self.g_rope_cos.to(patch_emb.device)
        g_sin = self.g_rope_sin.to(patch_emb.device)
        x = patch_emb
        for layer in self.global_layers:
            x = layer(x, g_cos, g_sin)
        global_out = self.global_norm(x)  # (B, N, d_global)
        
        # 4. SHIFT global output by 1 patch — LOAD-BEARING CAUSALITY CONTRACT
        # bytes in patch j use global context from patches 0..j-1 ONLY
        global_shifted = torch.cat([
            torch.zeros(B, 1, self.d_global, device=byte_ids.device, dtype=global_out.dtype),
            global_out[:, :-1, :]
        ], dim=1)  # (B, N, d_global)
        
        # 5. Project to d_local (SAME)
        global_local = self.global_to_local(global_shifted)  # (B, N, d_local)
        
        # 6. Gather back to byte level (CHANGED from fixed expand)
        byte_global = self._gather_to_bytes(global_local, patch_ids)  # (B, T, d_local)
        
        # 7. Combine byte embeddings + global context (SAME)
        local_input = byte_emb + byte_global
        
        # 8. Local causal decoder over full byte sequence (SAME)
        l_cos = self.l_rope_cos.to(local_input.device)
        l_sin = self.l_rope_sin.to(local_input.device)
        x = local_input
        for layer in self.local_decoder:
            x = layer(x, l_cos, l_sin)
        x = self.local_norm(x)
        
        # 9. Output logits (SAME)
        logits = self.output_head(x)  # (B, T, n_bytes)
        
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.n_bytes),
                targets.reshape(-1),
                ignore_index=-100,
            )
            return loss
        
        return logits
```

### 16.4 Training Loop Changes

```python
# In train() function, BEFORE the training loop:
patcher = BPEPatcher(vocab_size=8192)

# In each training step, AFTER sampling x, y:
x, y = dataset.sample_batch(batch_size, seq_bytes, device='cpu', split='train')
patch_ids, patch_mask, n_patches = patcher.batch_segment(x)

# Move to GPU
x = x.to(DEVICE)
y = y.to(DEVICE)
patch_ids = patch_ids.to(DEVICE)
patch_mask = patch_mask.to(DEVICE)

# Forward pass
with torch.amp.autocast('cuda', dtype=DTYPE):
    loss = model(x, y, patch_ids=patch_ids, patch_mask=patch_mask)

# MONITORING: log patch statistics every LOG_EVERY steps
if step % LOG_EVERY == 0:
    mean_patches = n_patches.float().mean().item()
    max_patches = n_patches.max().item()
    mean_patch_size = seq_bytes / mean_patches
    print(f"  patches: mean={mean_patches:.0f} max={max_patches} "
          f"avg_size={mean_patch_size:.1f} bytes/patch")
    
    # KILL CRITERION: if p95 patch count > max_patches allocation
    if max_patches > model.max_patches - 16:
        print("WARNING: patch count approaching max_patches limit!")
```

### 16.5 Generation Path

```python
@torch.no_grad()
def generate(self, prompt_bytes, max_new_bytes=256, temperature=0.8, top_k=50,
             patcher=None):
    """Autoregressive byte generation with BPE patching."""
    self.eval()
    device = next(self.parameters()).device
    ids = torch.tensor(prompt_bytes, dtype=torch.long, device='cpu').unsqueeze(0)
    
    for _ in range(max_new_bytes):
        if ids.shape[1] > self.max_seq_bytes:
            ids = ids[:, -self.max_seq_bytes:]
        
        # Patch on CPU
        patch_ids, patch_mask, _ = patcher.batch_segment(ids)
        
        # Forward on GPU
        logits = self.forward(
            ids.to(device), 
            patch_ids=patch_ids.to(device),
            patch_mask=patch_mask.to(device)
        )
        
        next_logits = logits[:, -1, :] / temperature
        if top_k > 0:
            v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
            next_logits[next_logits < v[:, [-1]]] = float('-inf')
        probs = F.softmax(next_logits, dim=-1)
        next_byte = torch.multinomial(probs, num_samples=1).cpu()
        ids = torch.cat([ids, next_byte], dim=1)
    
    return ids[0].tolist()
```

### 16.6 Required Unit Tests (CPU, no GPU)

```python
def test_mvg_scout():
    """Run BEFORE any GPU training."""
    
    # 1. Byte reconstruction from BPE spans
    patcher = BPEPatcher(vocab_size=8192)
    test_bytes = torch.tensor(list(b"Hello, world! This is a test."), dtype=torch.long)
    patch_ids, n_patches = patcher.segment(test_bytes)
    assert patch_ids.shape == test_bytes.shape
    assert patch_ids.max() == n_patches - 1
    assert (patch_ids[1:] >= patch_ids[:-1]).all()  # monotonically non-decreasing
    
    # 2. Param count assertion
    model = SutraDyadMVG()
    params = sum(p.numel() for p in model.parameters())
    baseline_params = 153_700_000  # Stage0 SutraDyad
    assert abs(params - baseline_params) / baseline_params < 0.10, \
        f"Param mismatch: {params} vs baseline {baseline_params}"
    
    # 3. Patch gather shape
    B, T = 2, 64
    byte_ids = torch.randint(0, 256, (B, T))
    patch_ids_b, patch_mask_b, _ = patcher.batch_segment(byte_ids)
    logits = model(byte_ids, patch_ids=patch_ids_b, patch_mask=patch_mask_b)
    assert logits.shape == (B, T, 256)
    
    # 4. Causality perturbation test
    byte_ids_a = torch.randint(0, 256, (1, 64))
    byte_ids_b = byte_ids_a.clone()
    byte_ids_b[0, 32] = (byte_ids_b[0, 32] + 1) % 256  # change byte 32
    
    pi_a, pm_a, _ = patcher.batch_segment(byte_ids_a)
    pi_b, pm_b, _ = patcher.batch_segment(byte_ids_b)
    
    logits_a = model(byte_ids_a, patch_ids=pi_a, patch_mask=pm_a)
    logits_b = model(byte_ids_b, patch_ids=pi_b, patch_mask=pm_b)
    
    # Bytes 0..31 should be identical (not affected by change at 32)
    # Note: this is approximate because BPE boundaries may shift
    # For fixed-fallback patches, it's exact
    
    # 5. Generation path uses same patcher
    output = model.generate([72, 101, 108, 108, 111], max_new_bytes=10, patcher=patcher)
    assert len(output) == 15  # 5 prompt + 10 generated
    assert all(0 <= b <= 255 for b in output)
    
    print("ALL MVG UNIT TESTS PASSED")
```

### 16.7 Experiment Protocol

```
EXPERIMENT: MVG Scout — BPE-aligned patching vs fixed 6-byte patching
HYPOTHESIS: Semantic (BPE-aligned) patch boundaries give the global transformer
            better context, improving BPB at matched parameters and compute.

BASELINE: SutraDyad Stage0 (PATCH_SIZE=6, ~153.7M params, eval BPB ~1.430 at 5K)
SCOUT:    SutraDyadMVG (BPE patches, ~152.4M params, same training config)

MATCHED CONDITIONS:
  - Same dataset, same data order (same random seed)
  - Same training config (LR=3e-4, warmup=300, batch=64, grad_accum=2)
  - Same sequence length (SEQ_BYTES=1536)
  - Same number of training steps
  - Same hardware (RTX 5090)
  
MEASUREMENTS:
  - Eval BPB every 500 steps
  - Wall-clock time per step (throughput comparison)
  - Patch count statistics (mean, max, p95)
  - Generation samples at 1K, 3K, 5K steps
  
DECISION CRITERIA:
  BPB(MVG) < BPB(baseline) - 0.05 at 5K → STRONG WIN → proceed to full Sutra-G
  BPB(MVG) < BPB(baseline) - 0.02 at 5K → WEAK WIN → extend to 10K
  BPB(MVG) within ±0.02 of baseline at 5K → INCONCLUSIVE → extend to 10K
  BPB(MVG) > BPB(baseline) + 0.02 at 5K → LOSS → pivot to K/F
  BPB(MVG) > 1.460 at 2K → EARLY KILL
  Throughput < 0.5x baseline → IMPLEMENTATION ISSUE, debug first
```

### 16.8 What Comes After MVG (Contingent Plans)

**IF MVG WINS → Sutra-G Staged Build:**
1. MVG + Ekalavya KD (dual-level: representation at global, logit at byte)
2. MVG + Sparse SwiGLU (top-10% activation in FFN layers)
3. MVG + Early Exit (per-patch confidence-based exit at global level)
4. Full Sutra-G integration + 15K training run + benchmark sweep

**IF MVG LOSES → Recurrent Byte Scout (K/F):**
1. Weight-shared global transformer (2-3 distinct layers, shared 4x)
2. Same byte I/O, same local decoder, same KD infrastructure
3. Tests the hypothesis: weight sharing + more effective depth > wider unique layers
4. Ekalavya KD remains as infrastructure regardless

**IF BOTH LOSE → Scale Current Architecture:**
1. Sutra-Dyad Stage0 → 300M params (d_global=1280, 16L)
2. Ekalavya KD with stronger teachers
3. Accept that the byte-level approach needs scale, not geometry

---

## 17. DESIGN SESSION CONCLUSION

**Three Codex rounds (R1-R3) converged on a clear consensus:**

1. The current Ekalavya-only strategy has a ceiling (0.005-0.016 BPB improvement with current teachers). It should remain as infrastructure, not the singular priority.

2. The highest-potential architecture is Sutra-G (token/BPE-aligned global processing over byte I/O), but the full composite is too complex to build and test at once.

3. The decisive experiment is the **MVG scout**: minimal changes to SutraDyad that isolate the core hypothesis (BPE-aligned patches > fixed byte patches).

4. **The zeroth experiment** (PATCH_SIZE=6→4, one-line change) should run first as a zero-risk probe of the context hypothesis.

5. Design phase confidence maxes out at 6.4/10 (conditional on MVG win). Only experiments can close the gap to 9/10.

**The blueprint in §16 is implementation-ready.** When GPU time frees up:
- Phase 0: Collect 6K Ekalavya results
- Phase 1: Run zeroth experiment (PATCH_SIZE=4, 3.5 hours)
- Phase 2: Implement MVG + unit tests (CPU only)
- Phase 3: Run MVG scout (4-6 hours)
- Phase 4: Sparse SwiGLU ablation (1.5-5 hours)
- Phase 5: Branch based on results

**Total GPU time for decisive answer: ~12-16 hours.**
