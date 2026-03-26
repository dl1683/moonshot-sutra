# Scratchpad

Working space for half-finished thoughts, emerging ideas, and in-progress reasoning. Things move OUT of here into RESEARCH.md (when validated) or get deleted (when falsified).

**Rule:** Nothing here is a conclusion. Everything here is a question or a sketch.

---

## META-PROCESS ANALYSIS: What Are We Learning About How We Learn? (2026-03-26)

**Status: ACTIVE — update at every checkpoint, every experiment**

### Process Failures (things we'd do differently)

1. **Architecture search went 5 rounds too long.** Final spread between ALL variants was 0.07-0.18 BPT. Should have set a "search dimension value" threshold: if total spread across all tested variants is < 0.2 BPT, the dimension is exhausted. Stop searching it. We wasted ~4 rounds of compute.

2. **Built full RMFD system before validating basic KD signal.** Should have run a 500-step "does KD provide ANY signal?" micro-probe before implementing byte-span bridges, CKA losses, multi-phase training. The probe running now should have been the FIRST thing after Round 3.

3. **Fundamentals research produced understanding but not novelty.** All 5 derived mechanisms were reinventions (Codex confirmed). The UNIQUE thing — byte-span cross-tokenizer alignment — emerged from solving a PRACTICAL constraint, not from surveying abstract math. **Codex correction (§6.4.23):** Novelty was the wrong KPI. The REAL mistake was letting theory pull implementation order ahead of signal validation. Theory should INFORM design, but signal validation gates implementation.

### Process Wins (things to keep doing)

1. **Codex correctness audits before training** — caught 3 real bugs, saved potentially wasted GPU hours.
2. **Kill rules declared before experiments** — prevented post-hoc rationalization of hybrid variants.
3. **Fundamentals-first approach** — even though it didn't produce novel mechanisms, it gave us deep understanding of WHY routing/scheduling works. This understanding will guide future decisions.
4. **Competitive baseline tracking** — knowing SmolLM-135M scores gives us a real target.

### Open Meta-Questions

1. **Is the KD advantage a head-start or a limit change?** Gap trajectory: -0.059 (500) → -0.042 (1000) → -0.023 (1500) → **-0.036 (2000)**. The narrowing REVERSED at step 2000 — gap widened from -0.023 to -0.036. This weakens the pure head-start hypothesis. Codex verdict (§6.4.23): "partially approved, not as conclusion — too early to call at ~106M tokens." Await steps 2500/3000 for full picture.

2. **What determines the theoretical MAXIMUM KD benefit?** Rate-distortion theory says the teacher reduces effective source entropy. But HOW MUCH depends on teacher-student mismatch, cross-tokenizer alignment quality, and alpha tuning. We haven't explored alpha at all.

3. **Are we measuring the right thing?** BPT doesn't predict benchmarks well at low step counts. Should we switch to a benchmark-focused eval earlier? Or is BPT still the right signal during early training?

4. **Should we invest in DATA instead of METHODS?** Our 22.9B tokens is 13x less than Pythia, 26x less than SmolLM. Maybe the biggest gain is data expansion, not better KD math.

5. **What's the opportunity cost of monitoring?** We spend significant time polling training logs. Should we build better automated eval infrastructure so results are automatically logged and analyzed?

### Process Rules (emerging from this analysis)

- **Dimension exhaustion rule:** If spread across all tested variants in a dimension is < 0.2 BPT, stop searching that dimension.
- **Signal-first rule:** Validate basic signal with a 500-step micro-probe before committing to full implementation.
- **Meta-checkpoint rule:** Every 5K-step eval asks not just "did metrics improve?" but "is this the right metric? Is this the right approach? What would 2x the improvement?"
- **Opportunity cost rule:** Before starting any work, ask "what ELSE could we do with this time/compute?" and pick the highest-expected-value option.

---

## Meta-Learning KD: Category-Theoretic Disagreement Analysis (2026-03-26)

**Status: EVALUATED BY CODEX — category theory overkill, MVP refined below**

**Codex Verdict (Architecture Theorist, 2026-03-26):** Category-theoretic framing acceptable as research language but NOT operational math. The 5-category decomposition is unidentifiable with 3 teachers. "Inverted power law" renamed to "transient adaptive acceleration" — not proven. Full review in RESEARCH.md §6.4.21.

### Core Thesis
Multi-teacher KD should be a SELF-IMPROVING process with accelerating returns, not diminishing returns. The manifesto says Intelligence = Geometry, not Scale — this applies to the LEARNING PROCESS itself, not just the architecture.

### The Three-Layer Framework

**Layer 1: Category-Theoretic Disagreement Grouping**

Each teacher T_i defines a functor F_i: Data → Predictions. The key objects are:
- **Agreement kernel** K(T_i, T_j) = {x : F_i(x) ≈ F_j(x)} — where two teachers agree
- **Disagreement manifold** D(T_i, T_j) = Data \ K — where they differ
- **Natural transformations** between functors capture HOW they disagree (probability mass shift, different top-1, different confidence level)

Practical decomposition of teacher outputs into categories:
1. **Consensus knowledge** — ALL teachers agree. This is "free" knowledge, easy to absorb.
2. **Architecture-dependent** — Transformers agree, SSMs disagree (or vice versa). Reveals structural bias.
3. **Capacity-dependent** — Large teachers agree, small teachers disagree. Knowledge gated by model capacity.
4. **Family-dependent** — Teachers from same family agree, cross-family disagree. Corporate training bias.
5. **Universally uncertain** — ALL teachers disagree with each other. Genuinely ambiguous data.

**Hypothesis**: Category 2 (architecture-dependent) is the HIGHEST VALUE for KD, because it reveals knowledge that can only be accessed via that architecture family. Distilling this into our student gives us cross-architecture capabilities no single model has.

**Layer 2: Inverted Power Law (Learning to Learn)**

Standard training: diminishing returns. More data → smaller marginal improvement.
Meta-learning KD: each training round produces:
- A better model (standard)
- A better understanding of WHAT the model can't learn (diagnostic)
- A better strategy for WHAT to learn next (curriculum)
- A better weighting for WHICH teacher to trust (routing)

This creates accelerating returns: the more you learn, the more precisely you can identify what to learn next → faster convergence.

**Mechanism**: After each eval checkpoint:
1. Run disagreement analysis across all teachers
2. Classify student failures by category (consensus vs arch-dependent vs capacity-gated)
3. Update teacher weights: upweight teachers that provide knowledge the student currently lacks
4. Update data curriculum: prioritize samples in high-disagreement regions
5. Optionally update architecture: if a knowledge type is consistently unlearnable, the architecture itself may need modification

**Layer 3: Knowledge State Map (Diagnostic-First Learning)**

Maintain a structured map of the student's knowledge state:
- Per-domain: syntax, factual recall, reasoning, world knowledge, etc.
- Per-category: which of the 5 disagreement categories is the student strongest/weakest in
- Per-teacher: which teacher provides the most value for the student's current state

This map is the BETTER INTERNAL METRIC the user asked for. Instead of BPT (which doesn't predict benchmarks), we track:
- "Student matches teacher consensus on X% of factual recall samples"
- "Student fails on Y% of capacity-gated reasoning samples"
- "Student can now absorb architecture-specific knowledge from SSM teachers"

These predict real benchmark performance because they measure actual capabilities, not just perplexity.

### Connection to Other Ideas
- **Ekalavya Protocol**: This IS the evolved Ekalavya — not just absorbing from teachers, but learning HOW to absorb
- **Rework internal metrics**: The knowledge state map IS the better internal metric
- **Decisive victory path**: If we can demonstrate accelerating returns from KD (inverted power law), that's PROVABLE data efficiency with scaling evidence

### Open Questions for Codex
1. How to implement disagreement analysis cheaply? (Can't afford full inference on all teachers every checkpoint)
2. What's the right granularity? Per-token? Per-sequence? Per-domain?
3. Is category theory overkill? Simpler clustering (KL divergence between teacher pairs) might suffice
4. How to validate the inverted power law claim? Need a metric that shows improving learning efficiency over time
5. Can we use this framework to PREDICT which new teachers would be most valuable before downloading them?

### Falsification Criteria
- If disagreement analysis shows no structure (random), the category theory angle is useless
- If teacher weighting by disagreement doesn't improve over uniform weighting, the routing is useless
- If learning efficiency doesn't accelerate (flat or diminishing curve), the meta-learning claim is false

---

## REFINED MVP: Routed KD with 4-Bucket Audit (2026-03-26)

**Status: DESIGN — pending probe results before implementation**

Based on Codex Architecture Theorist review. Stripped of category theory overhead, focused on operational disagreement geometry.

### Per-Sample Audit Metrics

**State loss per sample per teacher:**
```
l_state_i(x) = ||G(P_i @ S_student(x)) - G(S_teacher_i(x))||^2_F / M^2
```
where G(·) is row-normalized span Gram matrix over M=16 byte spans.

**Semantic loss per sample per teacher:**
```
l_sem_i(x) = 1 - cos(P_i @ z_student(x), z_teacher_i(x))
```
where z = mean-pooled hidden state.

Both return (B,) vectors, not scalars. This enables per-sample routing.

### 4-Bucket Classification (every 500 steps, on 256-window held-out audit)

Given teachers Q (Qwen) and L (LFM):
- **Consensus**: d_QL < q30 AND mean decoder entropy < q30 → Use equal weights
- **Specialist-Q**: d_QL > q70, student gap to Q > q70, Q has lower entropy → Route to Q only
- **Specialist-L**: d_QL > q70, student gap to L > q70, L has lower entropy → Route to L only
- **Uncertain**: Both decoders high entropy → Semantic-only (EmbeddingGemma)
- (Bridge-noisy: <12/16 spans occupied → Semantic-only — detect bad tokenization)

Where d_QL = pairwise divergence between Q and L span Gram matrices on each sample.

### Multi-Depth Teacher States

Teacher hidden states at relative depths 1/3, 2/3, 1.0 matched to student exits 7, 15, 23.
Currently TeacherAdapter returns only last hidden state — need multi-depth extraction.

### Surface Split

Route Qwen vs LFM on state surface only. Keep EmbeddingGemma as fixed low-weight semantic anchor:
```
L = L_CE + 0.4 * (w_Q * l_Q_state + w_L * l_L_state) + 0.1 * l_E_sem
```
where w_Q, w_L are bucket-derived weights (not learned — discrete routing).

### Kill Rule
- Routed KD must beat static multi_3family at equal teacher FLOPs
- Must lower specialist-bucket deficit
- Must not worsen stability (kurtosis/max_act) by more than ~10%
- If fails → kill routing, keep simple uniform KD + MiniPLM

### Implementation Order (AFTER probe shows signal)
1. Per-sample loss functions (modify compute_state_kd_loss, compute_semantic_kd_loss)
2. Held-out audit set creation (256 windows from validation)
3. 4-bucket classification function
4. Bucket-aware loss weighting in train_kd_phased()
5. Multi-depth teacher state extraction
6. Probe: routed vs uniform at equal FLOPs

### Questions for Fundamentals Research — ANSWERED (2026-03-26)
- Optimal Transport: Is the Wasserstein barycenter the right consensus average? → YES, Codex P3. Only if consensus bucket is large enough and simple averaging measurably lossy. +2-10ms via Sinkhorn on 16×16 span matrices.
- Information Geometry: Should bucket thresholds use Fisher-Rao? → NO, KILLED. Fisher routing needs +15-40 TFLOP/step. Not feasible at our scale. Stick with CKA/cosine.
- Ensemble Theory: Does ambiguity decomposition predict when routing helps? → YES, this is ambiguity-aware scheduling (Codex P1). Diversity × reliability gates routing decisions.

### Codex-Approved Mechanism Stack (2026-03-26)

**Source:** Codex Architecture Theorist evaluation of 5 fundamentals-derived mechanisms. See RESEARCH.md §7.6.

```
LAYER 0: 4-Bucket Audit (existing design, every 500 steps)
  │
LAYER 1: Ambiguity-Aware Scheduling [PRIORITY 1, ~free]
  │  Track: pairwise teacher disagreement + rolling bucket win-rate
  │  Gate:  high-diversity + high-reliability → specialist routing
  │         high-diversity + low-reliability → consensus-only or skip KD
  │         low-diversity → single best teacher
  │
LAYER 2: GW Routing [PRIORITY 2, +5-80ms/step]
  │  Only for specialist state-surface decisions
  │  16×16 span-distance matrices, 10-20 entropic GW iterations
  │  Stop-grad routing, existing CKA losses unchanged
  │
LAYER 3: WB Consensus [PRIORITY 3, +2-10ms/step]
  │  Only inside consensus bucket
  │  3-teacher Sinkhorn barycenter over 16 span masses
  │  Replace simple averaging with geometry-respecting average
  │
KILLED: Info-geometric projection routing (+15-40 TFLOP — not feasible)
DEFERRED: Multi-marginal OT loss (ablation only, after P1-P3 work)
```

**Implementation order:** After probe confirms signal, implement P1 → measure → P2 → measure → P3 if needed.

**The novelty is the SYSTEM, not the components:** byte-span cross-tokenizer alignment + multi-architecture teachers + surface-specific routing + disagreement-driven adaptive scheduling = a combination that doesn't exist in any published system.

---

## First KD Probe Design (DRAFT — pending Codex R8 approval)

**Goal:** Validate that KD provides measurable improvement over pure CE training at 5K steps.

### Setup
- **Student:** 24×512 transformer, ~93M params (same as probe architecture)
- **Teacher:** Qwen3-0.6B (~600M params, 1024 dim, 28 layers, GQA)
  - Loaded in FP16, inference only (no gradients): ~1.2GB VRAM
  - Student training: ~8GB VRAM → combined fit easily on 24GB
- **Baseline:** Same student, pure CE loss, same data
- **Duration:** 5K steps (matching existing probes for comparison)

### Cross-Tokenizer Challenge
- Student tokenizer: 16K custom BPE
- Teacher tokenizer: Qwen3 152K BPE
- Same input text → different tokenization → different sequence lengths

**Simplest approach (Level 1): Sequence-level representation matching**
- Input: raw text string
- Student tokenizes with 16K tokenizer → forward pass → get final hidden state, mean-pool over sequence → h_student (512-dim)
- Teacher tokenizes with Qwen3 tokenizer → forward pass → get final hidden state, mean-pool over sequence → h_teacher (1024-dim)
- Projection: Linear(1024, 512) learned projection maps teacher dim to student dim
- Loss: L_kd = CosineEmbeddingLoss(projected_teacher, h_student) or MSE
- Total: L = L_ce + alpha * L_kd (alpha=0.5 initially)
- **Pro:** No position alignment needed. Dead simple.
- **Con:** Loses per-token signal. Only captures sequence-level semantics.

**Better approach (Level 2): Token-level alignment via character offsets**
- For each student token, find which teacher tokens cover the same characters
- Average teacher hidden states for aligned positions
- Per-position loss: L_kd_pos = MSE(projected_teacher_aligned[t], h_student[t])
- **Pro:** Per-token signal. Richer supervision.
- **Con:** Character alignment code needed. Multiple teacher tokens per student token common.

**Best approach (Level 3): Logit-level via shared vocabulary projection**
- Build mapping: student_vocab → character sequences → teacher_vocab
- For each student token ID, compute weighted sum of teacher logits for corresponding teacher tokens
- KL divergence on aligned probability distributions
- **Pro:** Richest signal. Standard KD.
- **Con:** Alignment matrix complex. May need DSKD-style learned projection.

**Recommendation:** Start with Level 1, prove KD helps at all, then upgrade to Level 2/3.

### Training Loop Changes (in dense_baseline.py)
```
# In training loop, after computing CE loss:
# 1. Get student hidden states
student_out = model(inp, return_hidden=True)
logits = student_out['logits']
h_student = student_out['hidden'].mean(dim=1)  # (B, D_student)

# 2. Get teacher hidden states (no grad)
with torch.no_grad():
    teacher_tokens = teacher_tokenizer(raw_text, ...)
    teacher_out = teacher_model(teacher_tokens)
    h_teacher = teacher_out.last_hidden_state.mean(dim=1)  # (B, D_teacher)

# 3. Project and compute KD loss
h_teacher_proj = projection(h_teacher)  # (B, D_student)
kd_loss = F.mse_loss(h_student, h_teacher_proj.detach())

# 4. Combined loss
loss = ce_loss + alpha * kd_loss
```

### Data Pipeline Modification
- Current: ShardedDataset returns (input_ids, target_ids) — pre-tokenized
- Needed: Also return RAW TEXT for teacher tokenization
- Modification: data loader stores shard as tokens but also reconstructs text for teacher
- **OR:** Pre-tokenize shards for each teacher (stored alongside student shards)
- **Simplest:** Decode student tokens back to text → re-encode with teacher tokenizer (lossy but fast)

### Kill Rule for KD Probe
- KD probe passes if: BPT with KD >= BPT without KD + 0.1 at 5K steps
- Also check: generation quality (greedy decode sample), kurtosis/max_act stability
- If pass: proceed to multi-teacher, Level 2 alignment
- If fail: investigate why (wrong teacher? wrong loss? wrong alpha?)

### lm-eval Correlation Data (2026-03-26)
Both architectures at 5K steps produce identical near-random benchmarks (~33.5% ARC-Easy, ~17% ARC-Challenge). This confirms: training duration + KD >> architecture choice. The bottleneck is KNOWLEDGE, not architecture.

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

