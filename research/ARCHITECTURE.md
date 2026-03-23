# Sutra Architecture Reference (v0.6.0b — current best)

**Last updated: 2026-03-23 | Update this file whenever the architecture changes.**

This is the single source of truth for ALL architectural design decisions. Every T+L round MUST read this file. Every component must justify its existence — "inherited from GPT-2" or "everyone does it" is NOT justification.

---

## Parameter Budget Summary

| Component | Params | % of Total | Category | Justified? |
|-----------|--------|-----------|----------|------------|
| **emb** (50257 × 768) | 38,597,376 | 56.5% | Infrastructure | **NO — see Embedding Tax** |
| **stage_bank** (7 × FFN 768→1536→768) | 16,536,583 | 24.2% | Intelligence | PARTIAL — 7 banks fragmenting |
| **router** (LocalRouter, window=4, k=8) | 4,723,200 | 6.9% | Intelligence | YES — cross-position communication |
| **scratchpad** (8-slot memory) | 2,373,896 | 3.5% | Intelligence | YES — +3.27% when removed |
| **writer** (BayesianWrite) | 2,360,832 | 3.5% | Intelligence | YES — precision-weighted updates |
| **pos_emb** (2048 × 768) | 1,572,864 | 2.3% | Infrastructure | QUESTIONABLE — fixed, learned |
| **frozen_cache** | 591,168 | 0.9% | Dead weight | **NO — ablation-only, never used in production** |
| **init_mu** (Linear 768→768) | 590,592 | 0.9% | Infrastructure | YES — maps embedding to state space |
| **init_lam** (Linear 768→768) | 590,592 | 0.9% | Infrastructure | YES — initializes precision |
| **transition** (SwitchingKernel2) | 258,901 | 0.4% | Intelligence | PARTIAL — mode_gate is sequence-global |
| **gain_probe** | 116,481 | 0.2% | Diagnostic | Shadow only — doesn't train core |
| **LayerNorms** (6 Gated + 1 standard) | ~10,759 | 0.0% | Infrastructure | YES — stabilization |
| **TOTAL** | **68,323,243** | **100%** | | |

**Intelligence fraction: 38.5%** (stage_bank + router + scratchpad + writer + transition)
**Infrastructure/overhead: 61.5%** (emb + pos_emb + init + frozen_cache + probe + LN)

---

## THE EMBEDDING TAX (Critical Unquestioned Decision)

**Problem:** GPT-2 tokenizer (50,257 vocab × 768 dim) consumes 56.5% of all parameters.
- 75.8% of vocab tokens never appear in our training data (100K sample)
- Top 8,192 tokens cover 96.1% of data; top 16,384 cover 100%
- ~29M params (42.6%) are dead weight for unused tokens
- Embedding is NOT low-rank: rank-256 captures only 45.9% variance → ALBERT won't work
- Weight-tied with output head (saves one copy)

**Why this was never questioned:** T+L rounds focused on new mechanisms (controller, modes), never audited inherited infrastructure. This is the biggest efficiency win available.

**Alternatives (for T+L to decide):**
1. **Custom 16K BPE** → save 26M params (1.86x core capacity at same total)
2. **Custom 8K BPE** → save 32M params (2.07x core capacity)
3. **Vocab pruning** → keep top 16K GPT-2 tokens, slice weights (warm-startable)
4. **Byte-level + CNN** → 256 vocab, sequences 3.5x longer, ~38M saved

**Migration path (head transplant):** Core operates in R^768, doesn't care about vocab. Train new tokenizer → retokenize data → freeze core → train embedding 2K steps → unfreeze.

---

## Architecture Diagram

```
Input tokens x ∈ {0..50256}^T
         │
         ▼
┌─────────────────────────────┐
│  emb(x) + pos_emb(0..T-1)  │  Embedding: 40.2M params (58.8%)
│  h ∈ R^{B×T×768}           │  INHERITED: GPT-2 tokenizer, unoptimized
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  mu = init_mu(h)            │  State initialization: 1.2M params
│  lam = softplus(init_lam(h))│  mu ∈ R^768, lam ∈ R^768_+
│  pi = [0,0,1,0,0,0,0]      │  Start at stage 3 (Local Construction)
│  mem = scratchpad.init()    │  8 slots × 768 dim
│  pheromone = 0              │  Cross-position signal
└─────────────┬───────────────┘
              │
              ▼
┌═══════════════════════════════════════════════════════════════┐
║  RECURRENT LOOP: repeat P times (P=12 train, P=8-12 infer)  ║
║                                                               ║
║  ┌──────────────────────────────────────┐                    ║
║  │  1. STAGE TRANSITION                  │  258K params      ║
║  │  K = SwitchingKernel2(mu)             │  7×7 Markov kernel║
║  │  pi_evolved = pi @ K                  │  + 2-mode gate    ║
║  │  (content-dependent, graph-masked)    │  KNOWN BUG:       ║
║  │                                        │  mode_gate is     ║
║  │  Mode gate: sequence-global average   │  sequence-global,  ║
║  │  mix = softmax(gate(mean(mu)))        │  NOT per-token    ║
║  │  mode = sum(mix_m * bias_m)           │                    ║
║  └──────────────┬───────────────────────┘                    ║
║                 │                                             ║
║  ┌──────────────▼───────────────────────┐                    ║
║  │  2. STAGE BANK (compute)              │  16.5M params     ║
║  │  7 independent FFNs (768→1536→768)    │  (24.2% of model) ║
║  │  Only top-2 active stages computed    │                    ║
║  │  stage_out = sum(pi_s * FFN_s(mu))    │  QUESTION: 7 banks ║
║  │  evidence = Linear(stage_out) → R^7   │  fragment scarce   ║
║  │                                        │  data. 1 shared   ║
║  │  pi_new = top2(pi_evolved * softmax(  │  SwiGLU may be    ║
║  │           evidence/0.5))              │  better (R10)     ║
║  └──────────────┬───────────────────────┘                    ║
║                 │                                             ║
║  ┌──────────────▼───────────────────────┐                    ║
║  │  3. LOCAL ROUTING (cross-position)    │  4.7M params      ║
║  │  messages = LocalRouter(mu)           │  (6.9%)           ║
║  │  Window=4, k=8 sparse retrieval       │                    ║
║  │  Gated by pi[:,:,3:4] (Route stage)   │  Uses learned     ║
║  │                                        │  queries+keys,    ║
║  │  + scratchpad.read(mu, mem) * 0.1     │  not attention     ║
║  │  + pheromone * 0.05 * mem_ctx         │                    ║
║  └──────────────┬───────────────────────┘                    ║
║                 │                                             ║
║  ┌──────────────▼───────────────────────┐                    ║
║  │  4. BAYESIAN WRITE (state update)     │  2.4M params      ║
║  │  mu_new, lam = writer(mu, lam, msg,   │  (3.5%)           ║
║  │                       pi[:,:,4:5])    │                    ║
║  │  Precision-weighted: high-lam updates │  DERIVED: info     ║
║  │  resist noisy messages                │  theory (Bayesian  ║
║  │  mu = mu_new + stage_out * 0.1        │  evidence accum.)  ║
║  └──────────────┬───────────────────────┘                    ║
║                 │                                             ║
║  ┌──────────────▼───────────────────────┐                    ║
║  │  5. SCRATCHPAD UPDATE                 │  2.4M params      ║
║  │  mem = scratchpad.write(mu, mem, pi5) │  (3.5%)           ║
║  │  Prefix-causal gated EMA             │  LOAD-BEARING:     ║
║  │  8 slots, read via attention          │  +3.27% when       ║
║  │  Write gated by stage 5 probability   │  removed           ║
║  └──────────────┬───────────────────────┘                    ║
║                 │                                             ║
║  ┌──────────────▼───────────────────────┐                    ║
║  │  6. PHEROMONE UPDATE (no params)      │                    ║
║  │  deposit = pi_4 * tanh(||dmu||)       │  Cross-position    ║
║  │  phero = 0.9 * phero + deposit        │  signal, no_grad   ║
║  │  Only active after pass 3             │  INCONCLUSIVE:     ║
║  │                                        │  never isolated    ║
║  └──────────────┬───────────────────────┘                    ║
║                 │                                             ║
║  Gated LayerNorms wrap steps 2, 3, 4     │  ~10K params      ║
║  (pre_bank_ln, post_bank_ln, etc.)       │                    ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
              │
              ▼
┌─────────────────────────────┐
│  OUTPUT                      │
│  final = LayerNorm(mu)       │
│  logits = final @ emb.T      │  Weight-tied with embedding
│          / sqrt(768)         │
│  logits ∈ R^{B×T×50257}     │
└─────────────────────────────┘
```

---

## State Representation

Each token position carries a state tuple `(mu, lam, pi)`:

| State | Shape | Meaning | Derived From |
|-------|-------|---------|-------------|
| **mu** | R^768 | Hidden state (content) | Bayesian posterior mean |
| **lam** | R^768_+ | Precision (confidence per dimension) | Bayesian evidence accumulation |
| **pi** | Delta^7 | Stage distribution (processing phase) | Content-dependent Markov chain |
| **mem** | R^{8×768} | Scratchpad memory (shared) | Prefix-causal gated EMA |
| **pheromone** | R^T | Cross-position activity signal | Exponential decay accumulator |

---

## Training Configuration (v0.6.0b-rd12)

| Parameter | Value | Justification |
|-----------|-------|---------------|
| **Depth** | D ~ P(D=d) ∝ d^α, d∈{1..12}, α ramps 0.5→1.0 over 1K steps | Random-depth: biased toward deep, but all passes see final loss |
| **Loss** | L_final + 0.10 * L_step (attached) | L_step trains inter-pass dynamics |
| **Optimizer** | AdamW (β1=0.9, β2=0.95) | Standard for small LMs |
| **LR** | 1.5e-4, cosine decay to 1e-5 | INHERITED from v0.6.0a |
| **Batch** | 4 × 512 tokens | VRAM-constrained (24GB) |
| **Weight decay** | 0.1 | Standard |
| **Grad clip** | 1.0 | Standard |
| **Warm-start** | From v0.6.0a step 20K | Fresh optimizer (WSD restart) — CAUSED knowledge loss: SciQ -12.3%, LAMBADA -9.7% |

---

## Design Decision Audit

Every design decision must be either DERIVED (from first principles), VALIDATED (by experiment), or flagged as INHERITED (carried forward without justification).

| Decision | Status | Justification | First Questioned? |
|----------|--------|---------------|-------------------|
| GPT-2 tokenizer (50K vocab) | **INHERITED** | None. 75.8% of vocab unused. 56.5% of params. | R10+ (2026-03-23) |
| dim=768 | **INHERITED** | Copied from GPT-2 small. Not derived for this model/scale. | Never |
| ff_dim=1536 (2x dim) | **INHERITED** | Standard 2x ratio. Not derived. | Never |
| 7 stage graph | DERIVED | Codex debate (4 rounds, R4). Compressed from 12 stages. | R1 (v0.5.0) |
| Top-2 projection | DERIVED | Bounded compute per position. Sparse execution. | R1 (v0.5.0) |
| Bayesian write | DERIVED | Info theory: precision-weighted > residual addition. | R1 (v0.5.0) |
| Scratchpad (8 slots) | VALIDATED | +3.27% BPT when removed. Load-bearing. | R3 (v0.5.4) |
| LocalRouter (window=4, k=8) | **INHERITED** | Window size and k chosen arbitrarily. Not tuned. | Never |
| Learned positional embedding | **INHERITED** | Standard choice. RoPE, ALiBi, NoPE not considered. | Never |
| Weight tying (emb = lm_head) | VALIDATED | Standard, saves 38.6M params. | Implicit |
| 12 recurrent passes | VALIDATED | D=8-9 optimal (probes), D=12 for training diversity. | R6 |
| Pheromone routing | **INCONCLUSIVE** | Never isolated. Effect unknown. Adds complexity. | Never formally |
| 6 Gated LayerNorms | VALIDATED | Peri-LN pattern from v0.5.4. Stabilizes training. | R2 |
| Frozen cache (591K params) | **DEAD WEIGHT** | Ablation-only. Never used in production. Delete. | Now |
| Gain probe (116K params) | DIAGNOSTIC | Shadow-only, doesn't affect training. Justified for halting research. | R8 |
| max_seq_len=2048 | **INHERITED** | GPT-2 context length. Not derived for this model. | Never |
| SiLU activation | **INHERITED** | Standard. GELU, ReLU^2, not compared. | Never |
| AdamW (β1=0.9, β2=0.95) | **INHERITED** | Standard hyperparams. Not tuned. | Never |
| Cosine LR decay | **INHERITED** | Standard schedule. WSD, linear, not compared. | Partially (WSD restart for v0.6.0b) |
| Init: pi starts at stage 3 | **ARBITRARY** | "Local Construction." Why not stage 1? | Never |
| Pheromone rho=0.90 | **ARBITRARY** | Decay rate never tuned. | Never |
| Scratchpad EMA decay=0.9 | **ARBITRARY** | Decay rate never tuned. | Never |
| Stage bank evidence temp=0.5 | **ARBITRARY** | Temperature never tuned. | Never |

**Count: 7 DERIVED/VALIDATED, 9 INHERITED, 4 ARBITRARY, 1 DEAD WEIGHT, 1 INCONCLUSIVE, 1 DIAGNOSTIC**

---

## Key Empirical Results (v0.6.0b step 9000, 18 checkpoints)

| Metric | Value | Context |
|--------|-------|---------|
| **test_bpt** | **6.9155** | Step 9000 (NEW BEST — broke 7.14 plateau, 0.228 improvement) |
| **D=8 BPT** | 7.3421 (step 9000), 6.9014 (trough best, step 3000) | D=8-10 optimal depth |
| **D10 beats D12** | 100% of 18 checkpoints | Elastic compute definitively validated |
| **Optimal depth** | Oscillates D=8/9/10, NEVER D=12 | Passes 9-12 net negative in 93% of checkpoints |
| **Entropy spread** | 1.22-1.25x (stable) | No pass collapse (vs v0.6.0a's 2x cliff) |
| **Entropy shift** | Min entropy at pass 12 (4.878) at step 9000 | Late passes compressing for first time (prev min at pass 4-5) |
| **Trough D8 trend** | -0.042 BPT/1K steps (R²=0.99) | Post-restart: 7.10→6.99→6.96 |
| **Projected D8 at 15K** | ~6.65 | Would beat v0.6.0a (6.79) by 0.14 BPT |
| **Train loss** | ~4.83 (block avg) | Still declining at -0.08/1K steps |
| **Train-test gap** | ~+0.32 at step 7500 | Overfitting — capacity-starved (26M intelligence params) |
| **Mode entropy** | 0.002 | Controller is pass-global, NOT content-dependent |
| **MI(mode, token)** | 0.019 | Near-zero content dependence |
| **Oracle halting** | D=8 saves 33% at 0.0004 BPT cost | Elastic compute validated |

## Benchmark Results (all evaluated versions)

| Benchmark | v0.5.4 20K | v0.6.0a 20K | v0.6.0b 2K | v0.6.1 1K | Pythia-160M | Gap (best) |
|-----------|-----------|------------|-----------|----------|------------|-----------|
| **BPT** | N/A | **6.79** | 7.37 | 7.21 | N/A | N/A |
| SciQ | 33.6% | **48.1%** | 35.8% | 36.1% | ~74% | -26% |
| PIQA | 54.1% | **54.5%** | 52.9% | 54.4% | ~62% | -7.5% |
| LAMBADA | 1.8% | **11.2%** | 1.5% | 2.6% | ~32.6% | -21.4% |
| ARC-Easy | 29.7% | **31.3%** | 31.0% | 31.1% | ~43% | -11.7% |
| ARC-Challenge | 20.2% | 17.5% | **18.7%** | 16.7% | ~28% | -9.3% |
| HellaSwag | 25.8% | 25.7% | **25.9%** | 25.8% | ~30% | -4.1% |
| WinoGrande | 49.1% | **51.5%** | 51.2% | 48.9% | ~53% | -1.5% |

**Key observations:**
- v0.5.4 (detached history, 8 passes) is worst overall — validates attached history as key improvement
- v0.5.4 surprisingly best on ARC-Challenge (20.2%) — possibly noise or different failure modes
- v0.6.0a (20K steps, BPT 6.79) is still our best benchmark performer
- v0.6.1 (1K steps, BPT 7.21) matches v0.6.0a on PIQA/ARC-Easy but worse on WinoGrande/ARC-Chall
- v0.6.0b/v0.6.1 much worse on knowledge/recall (SciQ, LAMBADA) — BPT regression from WSD restart
- BPT alone doesn't predict benchmark quality — v0.6.0b BPT 7.37 ≈ v0.6.1 BPT 7.21 but different scores
- All models far below Pythia-160M on knowledge tasks (SciQ, LAMBADA)
- Closest on WinoGrande (-1.5%) and HellaSwag (-4.1%)

---

## What This File Is For

1. **T+L rounds:** Codex reads this before every design session. Questions anything marked INHERITED or ARBITRARY.
2. **Inherited Paradigm Audit:** The Decision Audit table makes inherited assumptions visible and questionable.
3. **Architecture changes:** When ANY component changes, update this file FIRST. The code follows the doc.
4. **Parameter accountability:** Every param must earn its keep. If a component's justification is "inherited," it's a candidate for redesign.
