# T+L Round 15 Brief — The Decisive Round

## SITUATION

14 rounds of Ekalavya KD experiments. ONE positive result (R13). That result just proved fragile (R14). This is the decisive round: we either find a fundamentally stronger approach or we make a strategic decision about the entire KD research direction.

## COMPLETE EXPERIMENT HISTORY (R5-R14)

### Phase 1: Warm-Start KD from 60K CE Checkpoint

All warm-start experiments start from `control_197m_60k_step60000.pt` (BPT=3.573, 60K steps CE-only).

| Round | Surface | Teacher | Schedule | BPT | vs CE 3K (3.5766) |
|-------|---------|---------|----------|-----|-------------------|
| R5-R6 | Logit TAID | Qwen 0.6B (3x) | beta 0→1/2K | 3.5856 | +0.009 (NOISE) |
| R7 | Multi-2T (logit+semantic) | Qwen 0.6B + EmbGemma | TAID + flat | 3.5971 | +0.021 (NOISE) |
| R8 | WSD-alpha (best WS) | Qwen 0.6B + EmbGemma | alpha decay to 0 | 3.5816 | +0.005 (NOISE) |
| R8b | Qlogit-only WSD-alpha | Qwen 0.6B only | alpha decay to 0 | 3.5904 | +0.014 (NOISE) |
| R9a | MiniPLM curated CE | — (curated data) | CE on top-20% | CATASTROPHIC | forgetting |
| R9b | RKL hard 10% | Qwen 1.7B (9x) | WSD-alpha | 3.5934 | +0.017 (FAIL) |
| R10 | Offline sparse replay | Qwen 1.7B cached | WSD-alpha | 3.6474 | +0.071 (KILLED) |
| R11v1 | RKP + DSKD ETA | Qwen 1.7B | alpha=0.7 | KILLED at 500 | too aggressive |
| R11v2 | RKP + DSKD ETA | Qwen 1.7B | alpha=0.3, LR=1e-4 | 3.5789 | +0.006 (NOISE) |

**VERDICT: All warm-start KD is net-negative. CE-only continuation always wins or ties.**

### Phase 2: From-Scratch KD (197M, 3K steps)

CE-WSD baseline: BPT=4.9703 at 3K (the number to beat).

| Round | Surface | Teacher | Key Params | BPT@3K | vs CE-WSD |
|-------|---------|---------|-----------|--------|-----------|
| R12a | Cross-tok logit | Qwen 1.7B | alpha=0.9, tau=1.0 | CATASTROPHIC | +2.15@1K |
| R12b | Cross-tok logit | Qwen 1.7B | alpha=0.9, tau=0.5 | CATASTROPHIC | +2.29@1K |
| R12c | Cross-tok logit | Qwen 1.7B | alpha=0.3, tau=0.5 | 4.9507 | -0.020 (WSD artifact) |
| **R13** | **State InfoNCE** | **Qwen 1.7B** | **alpha ramp→0.30, spans=32** | **4.8576** | **-0.113 (POSITIVE)** |
| R14 | Phase-pulse state | Qwen 1.7B | alpha→0.55, spans=16 | 6.8263@1K | +0.040@1K (KILLED) |

### R13 — The Only Positive Result (Details)

Config: byte-span InfoNCE cosine loss, 32 spans per window, 3 student/teacher layer pairs (7↔8, 15↔16, 23↔24), contrastive temperature=0.07, depth weights [0.30, 0.50, 0.20], 3 linear projectors (768→2048, bias=False), projector LR=6e-4.

Alpha schedule: ramp 0.06→0.30 (steps 1-400), hold 0.30 (401-2200), decay 0.30→0.03 (2201-3000).

| Step | R13 BPT | CE-WSD | Delta |
|------|---------|--------|-------|
| 500 | 7.578 | 7.6915 | -0.114 |
| 1000 | 6.4738 | 6.786 | **-0.312** (peak) |
| 1500 | 6.0146 | 6.0502 | -0.036 (compressed) |
| 2000 | 5.4598 | 5.7261 | -0.266 (reopened) |
| 2500 | 5.1334 | 5.3518 | -0.218 |
| 3000 | 4.8576 | 4.9703 | **-0.113** |

Key observation: Gap oscillated. Peaked at -0.312 (step 1000), compressed to -0.036 (step 1500), reopened to -0.266 (step 2000). The signal is real but unstable.

### R14 — Attempted Optimization, Made It Worse

Changes from R13: alpha_max 0.30→0.55, 6-phase alpha schedule, n_spans 32→16.

R14 at step 500: BPT=7.4591 (-0.232 vs CE, double R13's gap — promising).
R14 at step 1000: BPT=6.8263 (+0.040 vs CE — BEHIND baseline). KILLED.

Diagnosis: alpha=0.55 sustained for 700 steps overwhelmed CE gradient. The contrastive loss was being optimized (StateKD dropped 4.16→1.84) but at the expense of language modeling. R13's alpha_max=0.30 appears near-optimal; R14 proved the gain is fragile.

## KEY QUESTIONS FOR CODEX R15

1. **Is R13's -0.113 BPT improvement worth pursuing?** It's ONE positive result out of 14 attempts. R14 proved it's fragile. Is this a genuine signal or a lucky hyperparameter sweet spot?

2. **Should we scale R13 to 6K/15K?** If the -0.113 gap at 3K holds or grows at 6K, it validates hidden-state KD as a real mechanism. If it compresses (like the step 1500 compression within R13), it's a transient artifact.

3. **Is cross-tokenizer KD fundamentally broken for our setting?** 14 rounds, 1 fragile positive. Same-tokenizer (R5-R8 with Qwen 0.6B) was also noise. The issue may not be cross-tokenizer but KD itself at this scale.

4. **What about different knowledge surfaces?** We tested:
   - Logit KD (forward KL, reverse KL, TAID, DSKD ETA) → dead for cross-tok, noise for same-tok
   - Hidden-state KD (byte-span InfoNCE) → one positive (R13)
   - Semantic KD (relational Gram, EmbeddingGemma) → noise
   - Multi-surface → no better than single
   What surfaces HAVEN'T we tried?

5. **Is the student architecture the problem?** Maybe 197M with SwiGLU/GQA/SS-RMSNorm has limited capacity for absorbing teacher knowledge at 3K steps. Would a simpler architecture or longer training reveal the KD signal?

6. **Should we pivot entirely?** R14's Codex directive: "stop spending cycles on cross-tokenizer hidden-state KD and pivot to same-tokenizer teacher training." But same-tokenizer was also noise. What's left?

## AVAILABLE RESOURCES

- GPU: RTX 5090 24GB (FREE — R14 just killed, nothing running)
- Teachers already loaded/tested: Qwen3-1.7B (cross-tok), Qwen3-0.6B (same-tok), EmbeddingGemma-300M (same-dim encoder)
- Infrastructure: byte_span_pool, span_infonce_loss, TAID, DSKD ETA, RKP KnowledgePorts, linear projectors, alpha scheduling — all implemented in dense_baseline.py
- Student: Sutra-24A-197M, 24 layers, 768d, 12 heads GQA, exits at 7/15/23

## WHAT HASN'T BEEN TRIED

1. **Same-tokenizer teacher at larger scale**: Qwen3-4B or Qwen3-8B (would need aggressive quantization)
2. **Multi-teacher from scratch**: All from-scratch experiments used single teacher
3. **Progressive KD**: Train with Qwen 0.6B first, then switch to 1.7B
4. **Feature distillation with learned projectors**: Current projectors are simple linear. Nonlinear projectors, or projection to a shared latent space
5. **Attention transfer**: Never tried matching attention patterns
6. **Optimal transport alignment**: Literature suggests OT-based cross-tokenizer matching
7. **From-scratch with same-tokenizer teacher**: Use Qwen3-0.6B from step 0 (we only tested it warm-start)
8. **Longer from-scratch runs**: R13 was only 3K steps. Maybe state KD needs more time to accumulate

## YOUR TASK

Design R15. Be decisive. Either:
A) Prescribe a specific experiment that has high probability of showing DECISIVE improvement (not 0.1 BPT noise, but 0.5+ BPT or benchmark wins), OR
B) Make the strategic call to pivot away from standard KD entirely and propose what we do instead for O4 (Data Efficiency).

Every previous round's proposals were incremental tweaks to a broken paradigm. If the paradigm is broken, say so and propose the replacement.
