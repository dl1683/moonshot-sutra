# Tesla+Leibniz Round 3 Research Brief

This document compiles all research findings, probe results, and benchmark data
gathered between Round 2 and Round 3. Every finding here was generated in response
to Round 2's specific research and probe requests.

---

## Probe Results (Completed)

### Probe D: Scratchpad Load-Bearing Audit (CPU, step ~12K)

R2 requested: "Run fixed-slice eval in four modes: full system, read-disabled, write-disabled, and scratchpad removed."

| Mode | BPT | Delta | Delta% | Recall top-100 |
|------|-----|-------|--------|----------------|
| full | 5.8353 | — | — | 0.74 |
| no_read | 6.0260 | +0.191 | +3.27% | 0.72 |
| no_write | 5.8353 | +0.000 | +0.00% | 0.72 |
| removed | 6.0260 | +0.191 | +3.27% | 0.72 |

**Verdict: LOAD-BEARING (+3.27% when removed).**

Key findings:
- `no_read == removed` — the ENTIRE scratchpad benefit flows through the read path
- Write path has ZERO independent effect on BPT
- Recall impact is modest (0.74→0.72), suggesting scratchpad acts as general shared workspace, not precise associative recall
- R2 predicted "exact recall should specifically collapse when memory is impaired" — this was PARTIALLY confirmed (recall drops) but the effect is small
- The scratchpad is load-bearing but NOT yet functioning as true exact memory — more like a general optimizer

**Implications for R2's design proposal:** The "keep scratchpad for now, move toward associative/product-key memory post-gate" recommendation is validated. Current scratchpad is a positive but crude first draft.

### Probe: Embedding Factorization Feasibility (CPU, step ~12K)

R2 requested: "Compute randomized SVD of the current tied embedding matrix and reconstruct ranks 64, 128, and 256."

| Rank | Variance Explained | Frob Error | NN Preservation | Param Savings | KL Drift |
|------|-------------------|------------|-----------------|---------------|----------|
| 64 | 17.6% | 0.908 | 1.0% | 91.5% | 2.54 |
| 128 | 26.5% | 0.857 | 2.4% | 83.1% | 2.58 |
| 256 | 43.4% | 0.752 | 8.6% | 66.2% | 2.58 |

**Verdict: RISKY — warm-start factorization NOT feasible.**

Key findings:
- Singular value spectrum is flat: SV1=1773, SVs 2-768 all ~238
- Even rank 512 explains only 74% of variance
- NN preservation at rank 128 is 2.4% — embedding geometry is destroyed
- R2 predicted "if rank 128 preserves most lexical geometry, factorization becomes realistic" — this is FALSIFIED
- The model has spread information across all 768 dimensions — the embedding is genuinely full-rank at step 12K

**Implications for R2's design proposal:** "Do not confound the next gate with embedding surgery" is confirmed. Factored embeddings MUST be designed into the next architecture from scratch (V×E + E×H with E=128-256), not retrofitted to the current model.

### Probe E: Pass Disagreement (from R1, still valid)

- ce_spread correlates r=+0.959 with future_gain
- Q4/Q1 future_gain ratio: 4.79x
- Phase transition at pass 10-11: cosine 0.258, KL 5.1

This signal is from the CURRENT collapsed model. R2 noted: "Re-run after anti-collapse repair to see if it survives."

---

## Competitive Analysis: v0.5.4 → v0.6.0a → Pythia (User-Requested for R3)

**This question was specifically raised by the project lead for Round 3 analysis.**

### The Data

**v0.5.4** (warm-started from v0.5.3→v0.5.2, 21K steps, BPT 5.25, 1.7B tokens on MiniPile, 8 passes):

| Benchmark | v0.5.4 | Pythia-70M | Random |
|-----------|--------|-----------|--------|
| PIQA | 54.8% | 60.5% | 50% |
| WinoGrande | 49.8% | 51.9% | 50% |
| HellaSwag | 25.7% | 27.2% | 25% |
| ARC-C | 20.1% | 21.4% | 25% |
| ARC-E | 27.9% | 38.5% | 25% |
| SciQ | 25.9% | 74.0% | 25% |
| LAMBADA | ~1% | 32.6% | 0% |

**v0.6.0a** (from scratch, 12.6K steps, BPT 7.14, ~413M tokens from 20.72B corpus, 12 passes):
- lm-eval at step ~12K showed similar near-random performance on most benchmarks
- PIQA strongest signal at ~55%, consistent with v0.5.4

### Context for Honest Comparison

| Factor | v0.5.4 | v0.6.0a | Pythia-70M |
|--------|--------|---------|------------|
| Params | 68.3M | 68.3M | 70M |
| Training tokens | 1.7B | ~0.4B (at step 12K) | 300B |
| Warm-start | Yes (v0.5.3→v0.5.2) | No (from scratch) | No |
| Architecture | 8-pass recurrent + Grokfast | 12-pass recurrent + attached history | Standard transformer |
| Training hardware | 1x RTX 5090 | 1x RTX 5090 | 64x A100 |
| Best BPT | 5.25 | 7.14 | ~3.56 (est) |

### Questions for Codex R3

1. v0.5.4 at BPT 5.25 and v0.6.0a at BPT 7.14 achieve similar benchmark scores. Does BPT improvement not translate to benchmark improvement at this near-random scale? Or is the benchmark plateau a measurement artifact (both near random baseline)?
2. v0.5.4 with warm-start + 4x more tokens still couldn't break past near-random on most benchmarks. What does that say about the architecture's ceiling?
3. Sutra achieves 90-96% of Pythia on reasoning tasks (PIQA, WinoGrande, HellaSwag) with 176x less data. Is that efficiency real, or are all 68M models near-random on these benchmarks regardless?
4. What milestone (tokens consumed, BPT, or architectural change) would we expect to see benchmark breakout — i.e., clearly above random on 3+ benchmarks?
5. Given v0.5.4's advantages (warm-start, more training, lower BPT) did NOT translate to better benchmarks, should benchmark improvement be the gating criterion for the recurrence gate, or is BPT + generation quality more diagnostic at this scale?

---

## Research Request 1: ARMT/FwPKM Code-Level Survey

R2 requested: "Code-level survey of ARMT, FwPKM, and smallest credible associative-memory variants for recurrent LMs. Update equations, asymptotic cost, actual memory sizes, warm-start compatibility."

### Findings: 10-System Comparison

| System | Write Rule | Read Rule | Param Overhead | Warm-Start |
|--------|-----------|-----------|----------------|------------|
| **ARMT** | Delta-rule + correction on A matrix | A·φ(q) / z^T·φ(q) | ~6% (8M/137M) | **HIGH** (add-on, proven GPT-2) |
| **PKM (ResM)** | None (slow weights) | Product-key top-k sum | ~360% (value tables) | **HIGH** (proven BERT warm-start) |
| **Titans LTM** | Gradient surprise + momentum on MLP | MLP forward pass | ~10-30% est | **HIGH** (TPTT proven Llama-1B) |
| **FwPKM** | Chunk-level gradient on V table | Product-key top-k sum | ~360% (3.6x!) | MODERATE |
| **Gated DeltaNet** | S = αS + β(v - αSk)k^T | Sq | 0% (IS the layer) | LOW (Liger: moderate) |
| **GLA/GSA** | S = G⊙S + k^T v | Sq / slot attention | 0% (IS the layer) | MODERATE (T2R) |
| **RWKV-7** | s = diag(w)s + k^T v | r * (s·q) | 0% (IS the model) | LOW |
| **DeltaNet** | S = S(I - βkk^T) + βvk^T | Sq | 0% | LOW |
| **Linear Attn** | S += φ(k)v^T | Sφ(q) / z^Tφ(q) | 0% | MODERATE |

**Key findings for Sutra:**

1. **ARMT is the best warm-start-compatible exact memory.** Only ~6% overhead, adds per-layer associative matrix alongside existing model. Proven on GPT-2 (137M). Uses delta-rule write (corrects existing entries, not just appends). Read via normalized query-key matching. This is the closest match to what we need for the "precise" end of the retrieval spectrum.

2. **PKM with ResM is the gold standard for warm-start memory augmentation.** Kim & Jung 2020 showed: init from pretrained BERT, add PKM as residual `x' = LN(x + FFN(x) + PKM(x))`, two-stage training. Matched BERT-Large accuracy. BUT: ~360% parameter overhead (value tables are huge). For a 68M model, this would add ~245M params — too much.

3. **Titans via TPTT is surprisingly warm-start friendly.** Converts pretrained transformers to Titans using linearized attention injected in parallel, sharing Q/K/V weights. Only LoRA fine-tuning (rank 8) on 500 samples achieved 20% EM improvement. The "gradient surprise" write mechanism is compelling — memory updates proportional to how surprising the input is.

4. **Gated DeltaNet is the strongest pure recurrent memory at 340M.** Wiki ppl 27.01 vs Transformer++ 31.52. But it replaces attention entirely — not a warm-start add-on.

**Recommendation for precise memory design:** ARMT-style delta-rule memory with ~6% overhead, added as a side-path alongside the shared core. Can be warm-started. For the "general" end, keep the shared recurrent core. The routing between precise (ARMT-like) and general (core) is the learned spectrum the user requested.

---

## Research Request 2: Soft Intermediate Objectives for Recursive LMs

R2 requested: "Detached final-pass KL, shortcut-consistency, residual prediction, and representation-level alternatives that improved LM without turning intermediate passes into premature output heads."

### Findings (from R1 survey + existing RESEARCH.md, web agents still running)

Key findings from existing research (R1 brief + RESEARCH.md):
- **Three-arm probe is definitive:** Sampled L_step = KEEP, Full-vocab exact L_step = KILL (catastrophic 3.5 BPT regression), None = DEGRADE (10.6 vs 7.59). Weak supervision helps, strong supervision destroys late passes.
- **PonderNet approach:** Geometric prior over halting, trained via KL against the prior. Only works with adaptive depth (not fixed).
- **CALM (Google):** Early exit based on calibrated confidence. Trains with full-depth teacher, distills to early-exit student. Requires separate training phases.
- **Detached final-pass KL:** Use stop-grad on final pass logits as soft target for intermediate passes. Cheaper than full-vocab CE. Not tested in our setup.
- **Residual prediction:** Predict (h_final - h_current) at each pass. Gives gradient flow without forcing output heads. Novel for recursive LMs.
- **Representation alignment (CKA, mutual info):** Ensure intermediate pass representations are "heading toward" the final representation. Never tested for LM training.

**Gap:** No published evidence of representation-level intermediate objectives improving LM quality in recursive models. This remains our most novel territory.

**Recommendation:** After random-depth is working, test detached-KL (cheapest) and residual prediction (most principled) head-to-head against current sampled L_step.

---

## Research Request 3: Factored Embeddings Warm-Start Path

R2 requested: "Whether there is a practical warm-start path for rank-limited input/output factorization."

### Findings (from R1 research + probe results, web agent still running)

**Probe result settles the warm-start question: NOT FEASIBLE.** SVD shows the current embedding is full-rank (rank 128 = 26.5% variance, NN preservation 2.4%). Retrofitting factorization would destroy lexical geometry.

For the NEXT architecture (from scratch):
- **ALBERT-style V×E + E×H:** With V=50257, H=768, E=128: saves 32.1M params (83% reduction). ALBERT showed this works from scratch with small quality loss.
- **Practical E values:** Literature suggests E=128-256 for V~50K vocabularies. Below 128 quality degrades significantly.
- **Current embedding tax:** 40.2M params (58.8% of model). Only 28.2M remains for actual compute. This is the single biggest capacity constraint.
- **Design implication:** Factored embeddings should be a P0 for any next architecture started from scratch. Frees ~32M params for compute core.

---

## Research Request 4: Lightweight Pass-Identity Mechanisms (<350M)

R2 requested: "adaLN, low-rank depth adapters, RingFormer-style depth signals, per-pass LoRA. Parameter overhead, training stability, strongest evidence below 350M."

### Findings (from R1 14-model survey + web agent still running)

Key findings from R1's 14-model comparison:

| Mechanism | Used By | Overhead (68M model) | Warm-Start | Evidence |
|-----------|---------|---------------------|------------|----------|
| **adaLN** | Ouro, DiT, PixArt | ~0.3% (pass-conditioned scale/shift) | ADDITIVE ✓ | Proven in Ouro for recursive LM |
| **Iteration embedding** | ITT-162M, PonderNet | ~0.01% (one vector per pass) | ADDITIVE ✓ | ITT matched Pythia-160M |
| **Step embedding** | Universal Transformer, AdaPonderLM | ~0.01% | ADDITIVE ✓ | Standard, well-tested |
| **Pass-conditioned norm** | Latent Reasoning | ~0.1% | ADDITIVE ✓ | Works with random-depth |
| **Per-pass LoRA** | (proposed, not standard) | ~1-3% (rank 8-16 per pass × 12 passes) | ADDITIVE ✓ | Untested for recursive LM |
| **None (pure sharing)** | Huginn-3.5B | 0% | N/A | Only works at 3.5B+ |

**Consensus from R1:** Sub-200M recursive LMs almost unanimously use pass identity. Only Huginn at 3.5B succeeds without it. Sutra's collapse profile (passes 0-9 near-identical) is the exact failure mode pass identity is meant to address.

**R2's recommendation:** adaLN or low-rank depth adapters on the same shared core, with random-depth held constant.

**Strongest evidence:** ITT-162M with iteration embeddings + random-depth training matched Pythia-160M on benchmarks with 100% shared weights. Ouro with adaLN showed eval-time variable depth working.

---

## Research Request 5: Multi-Source Learning Recipes (Single GPU)

R2 requested: "Stage-specific teacher absorption, best embedding models for memory/address alignment, best small LMs for hidden-state transfer, avoiding teacher noise."

### Findings (from R1 + existing research, web agent still running)

From R2 assumption analysis:
- One explicit dual-teacher probe at tiny scale HURT by 8.6% — warning that undifferentiated teacher signal acts as noise
- Module-local teacher signals (especially on hard late-gain tokens) may succeed where global distillation failed
- R2 recommends: delay heavy teacher training until recurrence gate passes, but design interfaces for it NOW

From R1 research:
- **Cached teacher outputs:** Pre-compute teacher representations offline, store to disk. For 1B teacher on 1M tokens: ~6GB storage. Avoids running teacher during student training.
- **Hard-token selective distillation:** Use ce_spread (our validated signal, r=0.959) to select tokens where student struggles. Only distill on these.
- **Module-specific approach:** Memory module aligned to embedding teacher (BGE-large-en-v1.5 or E5-large). Core aligned to small LM teacher (Pythia-70M or SmolLM-135M) on hard tokens only.

**Key constraint:** Must not overwhelm the student. Low loss coefficient (0.01-0.05), warm-start teacher integration AFTER the recurrence gate passes.

---

## Research Request 6: Function-Preserving Growth Recipes

R2 requested: "Zero-init birth, WSD restarts, selective transfer, real examples of adding modules without wiping base behavior."

### Findings (from R1 research + existing RESEARCH.md, web agent still running)

From R1 research:
- **Zero-init birth:** Zero the output projection of new modules so model starts with identity behavior. MiniCPM demonstrated this for width/depth expansion. Key: the new module's residual contribution starts at exactly zero.
- **WSD schedule restarts:** Reset LR to warmup phase when adding new components. MiniCPM used this to add layers during training. The "stable" phase of WSD provides a clean insertion point.
- **Net2WiderNet:** Duplicate neurons and scale. Function-preserving by construction. Works for FFN expansion. Not directly applicable for adding new module types.
- **Selective freeze → unfreeze:** (1) Freeze base model. (2) Train only new module for N steps. (3) Gradually unfreeze base with low LR. Standard in adapter training.
- **LoRA-then-merge:** Train new capability as LoRA adapter, then merge weights. Preserves base capabilities well for small adaptations.

**For Sutra specifically:** The most relevant path is: (1) add ARMT-style memory module with zero-init output gate, (2) train only memory params for 1-2K steps, (3) unfreeze shared core with low LR, (4) WSD restart for the combined system. This is the function-preserving path to adding precise memory.

---

## Training Status Update

- v0.6.0a at step 12,600 as of this brief
- Best BPT: 7.14 (step 12,000)
- GPU PID 2040 still active, training toward 20K
- Rolling checkpoint updated ~06:14 UTC
- Power law fit: BPT = 99.5 × step^(-0.283) — predicts ~5.5 BPT at 20K

## Priority Nudge: Precise Memory vs Imprecise Memory (User Direction)

**The knowledge-task gap is the biggest competitive weakness.** SciQ at 25.9% (vs Pythia's 74%) and LAMBADA at ~1% (vs Pythia's 32.6%) show that Sutra cannot recall exact facts. This is not a data problem alone — it's a memory architecture problem.

Probe D confirmed: the current scratchpad is an **imprecise shared workspace** (general optimizer), not a precise associative memory. It gives +3.27% BPT (good) but only +2% recall (insufficient for knowledge tasks).

**The distinction that matters:**
- **Imprecise memory** = "I vaguely remember something about X" → general context, helps with reasoning tasks (PIQA, WinoGrande). This is what the current scratchpad provides.
- **Precise memory** = "The capital of France is Paris" → exact fact retrieval, required for knowledge tasks (SciQ, LAMBADA, factual generation). This is what we lack.

**Design priority for Round 3:** The architecture needs a **continuous retrieval spectrum** — not a binary "precise OR general" but a learned mechanism that decides for each query/token where on the spectrum to operate:

- **Pure general reasoning** ← "Why does X lead to Y?" → relies on distributed representations, multi-pass refinement, pattern completion
- **Mixed** ← "Based on the data, X is likely because..." → needs both reasoning AND fact grounding
- **Pure precise retrieval** → "The capital of France is ___" → needs exact fact lookup with zero approximation error

The model must learn WHEN to go precise vs general. This is content-dependent and connects directly to:
1. **Outcome 1 (Intelligence):** Real-world tasks require BOTH. A model that can reason but can't recall facts, or can recall but can't reason, fails most useful tasks.
2. **Outcome 5 (Inference Efficiency):** The retrieval spectrum IS a form of elastic compute — precise retrieval may be cheap (one lookup), general reasoning may be expensive (many passes), and the model allocates accordingly.
3. **The stage-superposition thesis:** Different tokens ARE at different processing stages. Some need the "memory lookup" stage, others need the "reasoning refinement" stage. This is what stage routing was always supposed to enable.

The current scratchpad provides only the general end of this spectrum. The R3 design must propose how to achieve the precise end AND the learned routing between them.

**This connects to:** R2's recommendation to "move toward associative or product-key memory so recall is explicit rather than merely shared-workspace-like." The user is elevating this from "nice to have post-gate" to "load-bearing design requirement" and specifically requesting a CONTINUOUS SPECTRUM with learned routing — not just adding a lookup table.

### Companion Principle: Dynamic Multi-Mode Processing (User Direction)

**The retrieval spectrum is one dimension. The other dimension is COGNITIVE MODE.**

Real intelligence doesn't do one thing at a time. When processing a complex task, you simultaneously:
- **Look things up** (precise memory retrieval)
- **Verify** what you just retrieved against context
- **Generate** continuations while still reasoning about earlier content
- **Think about what's missing** (meta-cognitive gap detection)
- **Critique** your own output as it forms

The original superposition thesis was exactly this: different tokens/positions are at different processing stages simultaneously. The 7-stage bank was the MECHANISM for this (each stage = a different cognitive mode). R2 killed the 7 heavyweight FFN banks, but the CONCEPT must survive in the simplified architecture.

**Design constraint for R3:** The simplified shared-core model must still support multiple simultaneous cognitive modes. This could be:
- Stage probabilities (pi) modulating a single shared core differently for different positions
- Separate lightweight control paths (not full FFNs) for lookup vs reasoning vs verification
- The pass identity mechanism doubling as a "what cognitive mode am I in" signal
- Content-dependent routing through the retrieval spectrum at each pass

**Why this matters for the mission:**
- **Improvability (Outcome 2):** If cognitive modes are explicit and separable, someone can improve the "verification" mode without touching the "generation" mode
- **Democratization (Outcome 3):** Domain experts can specialize specific cognitive modes for their domain
- **Intelligence (Outcome 1):** Models that can only do one thing at a time are fundamentally limited. Multi-mode processing is what makes complex reasoning possible.

**This is what makes Sutra different from "just another recurrent LM."** Without this, we're building PonderNet with extra steps. With it, we're building something that processes information the way intelligence actually works.

---

## Summary: What R2 Proposed and What the Data Says

| R2 Proposal | Probe/Research Finding | Status |
|-------------|----------------------|--------|
| Keep scratchpad, move toward associative memory post-gate | Probe D: LOAD-BEARING but crude | ✅ Validated |
| Don't confound gate with embedding surgery | SVD: full-rank, factorization destroys geometry | ✅ Validated |
| ce_spread as future elastic compute signal | Probe E: r=0.959, but pre-collapse-fix | ⏳ Re-run needed post-fix |
| Build minimal shared-core canary (THE GATE) | Not yet executed (GPU busy) | ⏳ Blocked on GPU |
| Random-depth as primary anti-collapse mechanism | Not yet tested | ⏳ Blocked on GPU |
| adaLN or low-rank pass identity | Research pending | ⏳ Awaiting research |
| Soft intermediate supervision (not full-vocab CE) | Research pending | ⏳ Awaiting research |
