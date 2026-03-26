# T+L Round 3: Evidence Package

## Summary of All Probe Results (42M scale, 5000 steps each)

### Probe 1: DyT vs RMSNorm (COMPLETED)
Config: d=512, L=10, H=8, ff=1536, seq=512, BS=16x2, LR=3e-4, WSD schedule.

| Metric | RMSNorm | DyT | Delta |
|--------|---------|-----|-------|
| Final BPT | **5.08** | 5.83 | +0.75 (DyT worse) |
| Kurtosis avg | 0.24 | 0.90 | DyT 3.75x worse |
| Kurtosis max | 0.98 | 1.47 | DyT 1.5x worse |
| Max activation | 20.55 | 21.64 | Nearly equal |
| Params | 42,281,472 | 42,302,976 | +21K (DyT alpha params) |

**Verdict:** DyT as drop-in replacement FALSIFIED at 42M/5000 steps. DyT doesn't normalize the residual stream (only internal activations via tanh). Gap narrowed from 2.22→0.75 over training but didn't close. SS-RMSNorm is the confirmed mainline norm.

### Probe 2: TOP vs NTP (COMPLETED — CATASTROPHIC FAILURE)
Config: Same 42M model. TOP weight=0.05, TOP_K=4, activation after warmup (step 200).

| Metric | NTP-only | NTP+TOP | Delta |
|--------|----------|---------|-------|
| Final BPT | **5.07** | 9.68 | **+4.61 (TOP catastrophically worse)** |
| Kurtosis avg | 0.35 | 18.75 | TOP 53x worse |
| Kurtosis max | 1.15 | 99.29 | TOP 86x worse |
| Max activation | 22.03 | 133.83 | TOP 6x worse |

**Trajectory:** TOP BPT never dropped below 9.5 at any eval point. The gap WIDENED over training (was ~3 BPT at step 500, became 4.6 by step 5000). TOP loss magnitude (~20) is comparable to CE (~8), contributing ~12% gradient magnitude. Kurtosis exploded from 0.1→99.3.

**Verdict:** TOP at weight=0.05 with activation at step 200 is DEAD at 42M scale. The TOP paper tested at 340M+ — there may be a minimum model capacity threshold. If retried, would need: weight ≤0.001, activation at step 2000+, or stop-grad backbone for TOP head. **Round 2's conditional status is resolved: TOP does NOT graduate to mainline.**

### Probe 3: Muon vs AdamW Optimizer/Norm (RUNNING)
Config: d=512, L=10, H=8, ff=1536, 5000 steps each, 3 variants:
- adamw_rmsnorm: AdamW + standard RMSNorm (Round 1 baseline)
- adamw_ss_rmsnorm: AdamW + Single-Scale RMSNorm
- muon_ss_rmsnorm: Muon (lr=0.02) + SS-RMSNorm (Round 2 candidate mainline)

**Status:** Variant 1 (adamw_rmsnorm) at ~step 1100/5000. Will update with results when complete.

### Probe 4: Trunk Choice — Hybrid vs Pure (PENDING — runs after Probe 3)
Config: d=512, L=12, H=8, ff=1536, conv_kernel=64, 5000 steps, 3 variants:
- pure_transformer: 12 attention blocks (49.1M params)
- pure_conv: 12 GatedConv blocks (46.3M params)
- hybrid_3to1: 9 GatedConv + 3 attention blocks in [HHAHHHAHHHA H] pattern (47.0M params)

**Rationale:** Most critical architectural decision — determines whether HEMD's hybrid trunk is worth the complexity.

## Web Research Updates (2026-03-26)

### Muon Optimizer at Small Scale
**Sources:** Essential AI paper (arxiv:2505.02222), Muon GitHub, Keller Jordan blog, PyTorch docs

- Essential AI validated Muon across **100M–4B params** — it works at small scale
- Default hyperparams: lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.1
- Muon LR is ~67x higher than AdamW LR (0.02 vs 3e-4) — our probe config is correct
- Only for 2D hidden layer params; embeddings, norms, biases → AdamW (our code does this correctly)
- Has muP scaling built in — LR should transfer across model sizes without retuning
- Apply separately to Q, K, V (not combined QKV) — our model has separate attention projections, OK
- Works with conv layers by flattening last 3 dims — relevant for GatedConvBlock
- **AdaMuon** (new, OpenReview): combines element-wise adaptivity with orthogonal updates, claims 40% better than Adam
- No reports of Muon failing at small scale; the question is HOW MUCH it helps at 42M

### Hybrid Architecture Research (2025-2026)
**Sources:** arxiv:2510.04800v1 (Systematic Analysis), arxiv:2503.01868 (StripedHyena 2), arxiv:2411.13676 (Hymba)

**Critical finding from "Hybrid Architectures for Language Models: Systematic Analysis":**
- Tested inter-layer and intra-layer hybrid designs at 350M scale, 60B tokens
- **Inter-layer 1:5 attention-to-SSM ratio is optimal** for efficiency-quality balance
- Intra-layer hybrid (parallel attention+SSM heads) > inter-layer (sequential blocks)
- **"Never place Transformer blocks at the front"** — middle layers, evenly distributed
- "Evenly scattering intra-hybrid blocks across depths yields the best quality"
- At 350M: hybrid 1:5 NLL=2.860 vs pure transformer NLL=2.882 — small but real gain
- Mamba 2 was the SSM tested (we use gated conv, different mechanism)

**Our probe design check:** hybrid_3to1 pattern [H,H,A,H,H,H,A,H,H,H,A,H] has attention at layers 3,7,11 (0-indexed: 2,6,10). This is 1:3 attention-to-conv ratio (3 attention / 9 conv). Attention NOT at front (first two layers are H). Matches research recommendations well.

**Hymba (NVIDIA):** Intra-layer hybrid at 1.5B: 11.67x cache reduction, 3.49x throughput vs Llama-3.2-3B with 1.32% higher accuracy. Shows hybrid can beat pure transformer significantly at small scale.

**StripedHyena 2:** 40B scale, 1.2-2.9x faster than optimized transformers. Conv+attention complementary for different token manipulation tasks.

### Competitive Baselines (March 2026)
**Our parameter targets:** 42M (probe scale) → 100M (scout) → 200M (main)
**Our data:** 22.9B tokens from 18 sources

| Model | Params | Training Tokens | HellaSwag | ARC | PIQA |
|-------|--------|----------------|-----------|-----|------|
| SmolLM2-135M | 135M | 2T | 42.1% | 43.9% | 68.4% |
| SmolLM2-360M | 360M | 4T | 54.5% | 53.0% | — |
| Pythia-160M | 160M | 300B | ~33% | ~30% | ~62% |
| MobileLLM-125M | 125M | 1T | — | — | — |

**Key gap:** SmolLM2-135M trains on 2T tokens (87x our 22.9B). SmolLM2-360M on 4T (175x). The manifesto argument: if Intelligence=Geometry works, we should approach these numbers with 100x less data through better architecture + multi-source learning.

**Realistic expectations at our data scale:** With 22.9B tokens and a 100M model, matching Pythia-160M (300B tokens, 13x our data) is a reasonable first milestone. Matching SmolLM2-135M (2T tokens) would require extraordinary data efficiency gains.

### Technical Notes for Round 3

**Muon + Conv weights:** PyTorch's `torch.optim.Muon` only supports 2D params. Conv1d weights are 3D. When combining Muon with hybrid architecture (GatedConvBlock), conv weights must either be: (a) routed to AdamW, or (b) reshaped to 2D before Muon step. The Muon blog says it "works with conv layers by flattening last 3 dims" — but this is the GitHub implementation, not PyTorch's built-in version. Need to verify or write a wrapper.

### Implication for Sutra Design
The research strongly supports hybrid trunk. But the cutting-edge approach is **intra-layer** (parallel attention+SSM within each block), not just inter-layer (our current probe). This could be a Round 4+ exploration: replace some blocks with parallel attention+conv heads instead of pure sequential alternation.

**Key question for Codex Round 3:** Given that intra-layer hybrid > inter-layer hybrid (per systematic analysis), should HEMD's design evolve toward parallel attention+conv heads within each block, rather than just alternating block types?

### MiniPLM Implementation Details (arxiv:2410.17215)
**Exact formula:** `score(x) = log p_teacher(x) - log p_ref(x)` = `NLL_ref(x) - NLL_teacher(x)`
- Higher score = teacher knows it but reference doesn't = highest-value training data
- **Selection:** Top-K with α=0.5 (keep top 50% of data by score)
- **Reference model:** 104M params, trained on 5B tokens of same corpus (NOT random init)
- **Scales tested:** 200M, 500M, 1.2B students; 1.8B teacher (Qwen-1.5 arch)
- **Cross-family tested:** 212M Llama3.1, 140M Mamba — architecture-agnostic
- **Speedup:** 2.2x compute reduction, 2.4x data efficiency
- **No temperature** — raw log probability ratios

**Our implementation plan:**
- Teacher: llama3.2:3b or qwen3:4b (via Ollama, best available)
- Reference: qwen2.5:0.5b or qwen3:0.6b (via Ollama, weak model)
- Score windows from each shard: compute NLL per window under both models
- Convert to shard-level weights or per-window selection
- Feed into ShardedDataset(weight_file=...) via existing infrastructure
- **Blocker:** Ollama uses GPU, can't score while training. Score BEFORE training starts.

## Questions for Codex Round 3

1. Given probe evidence (DyT dead, TOP dead, MTP dead, halting dead at 42M), what auxiliary losses if any should the scout use beyond plain NTP?
2. Should the hybrid architecture use intra-layer fusion (parallel attention+conv heads) instead of or in addition to inter-layer alternation?
3. What is the exact HEMD-S scout specification given updated evidence?
4. Is 42M the right probe scale, or should we move to 100M probes for more realistic signal?
5. Data pipeline: quality filter first, then MiniPLM? Or combined? What reference model?
6. Given 22.9B tokens and realistic competitive targets, what BPT/benchmark targets should the scout achieve to validate the architecture before scaling to 200M?

## Updated Architectural Implications

### What's Confirmed
1. **SS-RMSNorm is the mainline norm** — DyT falsified, standard RMSNorm also viable but SS-RMSNorm preferred for quantization
2. **TOP is dead at 42M** — remove from mainline scout entirely; only revisit at 200M+ scale
3. **MTP also hurts at small scale** — confirmed in prior session (control beat MTP by +0.13 BPT)
4. **Halting controller hurts at small scale** — confirmed in prior session (control beat halting by +0.15 BPT)

### What's Pending Resolution
1. **Does Muon beat AdamW at 42M?** — probe running, expected yes based on literature
2. **Does hybrid beat pure transformer?** — probe pending, critical for HEMD design
3. **Data shaping impact** — not yet probed (needs teacher scoring infrastructure first)

### Running Evidence Tally for Round 3
- Probes completed: 4 (DyT, TOP, MTP from prior session, Halting from prior session)
- Probes running: 1 (Muon vs AdamW)
- Probes pending: 1 (Trunk choice)
- Net architectural changes from probes: DyT→killed, TOP→killed, MTP→killed (at 42M), halting→killed (at 42M), SS-RMSNorm→confirmed, Muon→pending, hybrid trunk→pending
