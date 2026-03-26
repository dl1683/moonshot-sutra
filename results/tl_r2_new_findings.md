# T+L Round 2: New Findings Since Round 1

## Probe Results

### DyT vs RMSNorm (42M, 5000 steps, matched config)
Config: d=512, L=10, H=8, ff=1536, seq=512, BS=16x2, LR=3e-4, WSD schedule.

| Metric | RMSNorm | DyT | Delta |
|--------|---------|-----|-------|
| Final BPT (step 5000) | **5.08** | 5.83 | +0.75 (DyT worse) |
| Kurtosis avg | 0.24 | 0.90 | +0.66 (DyT worse) |
| Kurtosis max | 0.98 | 1.47 | +0.49 (DyT worse) |
| Max activation | 20.55 | 21.64 | +1.09 (nearly equal) |
| Params | 42,281,472 | 42,302,976 | +21K |

**Convergence trajectory (BPT gap at each eval):**
- Step 500: gap=2.22 (DyT far behind)
- Step 1000: gap=2.14
- Step 1500: gap=1.69
- Step 2000: gap=1.52
- Step 2500: gap=1.20
- Step 3000: gap=1.37
- Step 3500: gap=1.03
- Step 4000: gap=0.97
- Step 4500: gap=0.74
- Step 5000: gap=0.75

**Key observations:**
1. DyT converges SLOWER but gap closes steadily from 2.22 to 0.75
2. DyT kurtosis starts extreme (29.3 at step 500) but drops to 1.5 by step 5000
3. DyT max activation starts at 176.4 (!!) but drops to 21.6 — converging with RMSNorm
4. At step 4500, DyT max_act (22.2) was actually LOWER than RMSNorm (26.3)
5. DyT does NOT bound the residual stream — only squashes internal activations via tanh. The residual connection x + block(DyT(x)) accumulates without scale control
6. Generation quality: RMSNorm more coherent; DyT shows repetition/degeneration
7. With more training (>5000 steps), the gap might close further

**Interpretation:** DyT as a drop-in replacement for RMSNorm hurts at 42M/5000 steps. The problem is NOT the tanh itself but the lack of residual stream normalization. RMSNorm actively controls scale; DyT doesn't. Two possible fixes:
- (a) DyT + residual scaling (e.g., divide residual by sqrt(L))
- (b) Longer training — DyT alpha needs time to learn proper scales
- (c) Single-Scale RMSNorm (from OSP paper) as a middle ground

### TOP vs NTP probe (RUNNING — results available for Round 3)
Currently training. NTP-only variant first, then NTP+TOP. Same 42M config, 5000 steps each.

## Web Research Findings

### TOP (Token Order Prediction) — arXiv:2508.19228
- Tested at 340M, 1.8B, 7B on FineWeb-Edu (52-104B tokens)
- TOP improved 340M models on standard NLP benchmarks (e.g., +2.50 SciQ)
- MTP HURTS small models (<1.3B); TOP addresses this via ranking instead of exact prediction
- Requires only one extra unembedding layer (~0.6% overhead)
- On synthetic star graph task: TOP enables pathfinding where NTP/MTP/DS-MTP all fail
- **Verdict: most promising small-model auxiliary loss. Our probe will validate at 42M.**

### DyT — arXiv:2503.10622 (CVPR 2025)
- Drop-in replacement for LayerNorm/RMSNorm across ViT, ConvNeXt, MAE, DINO, DiT, LLaMA, wav2vec 2.0, DNA
- Matches or exceeds normalized counterparts in all tested settings
- Authors: Jiachen Zhu, Xinlei Chen, Kaiming He, Yann LeCun, Zhuang Liu (FAIR/NYU/MIT/Princeton)
- **Our probe shows DyT doesn't match RMSNorm at 42M/5000 steps.** Possible explanations: (1) paper's LLaMA experiments used much larger models, (2) init_alpha=0.5 may be wrong for our scale, (3) 5000 steps insufficient

### Muon Optimizer — arXiv:2502.16982
- **2x compute efficiency vs AdamW** (achieves same perf at 52% FLOPs)
- Steps to convergence: ~2/3 of AdamW
- Orthogonalizes gradient momentum via Newton-Schulz iteration
- Eliminates "privileged bases" — prevents activation outliers from forming
- Moonlight: 3B/16B MoE trained with Muon on 5.7T tokens
- OSP paper: Muon + Single-Scale RMSNorm → kurtosis 0.04 (vs Adam's 1818.56!)
- **Strong candidate: 2x training efficiency + outlier prevention = major win for O1 and O5**

### SmolLM2 Training Techniques
- 135M: 2T tokens, 360M: 4T tokens on curated data mix
- Key: FineWeb-Edu classifier used to filter DCLM (quality filtering, not just data mixing)
- Single-stage training with consistently high-quality data (no multi-stage)
- Stack-Edu for code, InfiMath for math, Cosmopedia for structured knowledge
- **Implication: data quality filtering > data volume. We should quality-filter our 22.9B tokens.**

### MiniPLM — ICLR 2025
- "Difference Sampling": sample instances where teacher scores high but reference small model scores low
- Offline teacher inference → zero training cost
- Cross-family KD (can use any teacher regardless of architecture)
- 2.2x pre-training acceleration, 2.4x data efficiency
- **Directly implementable via our existing `score_shards_teacher()` in data_loader.py**

### Multi-Teacher KD (2025-2026 survey)
- Adaptive teacher weighting critical (not static averaging)
- GRPO-KD: augments GRPO with adaptively weighted multi-teacher distillation
- Teacher contributions should be weighted by student's understanding of each teacher's expertise
- Multi-round, parallel, and dual-level aggregation are core innovations
- **Key: teacher diversity matters more than teacher quality. Different architectures = complementary knowledge.**

## Implications for Round 2 Design

1. **DyT needs modification** — pure drop-in replacement underperforms. Consider: DyT for internal activations + residual stream scaling, or Single-Scale RMSNorm, or Muon optimizer to prevent outliers at the source
2. **Muon is the highest-leverage change** — 2x training efficiency AND outlier prevention. Should replace AdamW in the mainline design
3. **TOP probe is running** — if it shows gains at 42M, it validates the auxiliary planning loss for O1
4. **MiniPLM Difference Sampling** should be the O4 baseline — zero-cost, architecture-agnostic, already partially implemented
5. **Data quality filtering** (SmolLM2 approach) may matter more than teacher distillation — quality > quantity
