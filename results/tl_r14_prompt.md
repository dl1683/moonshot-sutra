# T+L R14 — Hidden-State KD: First Positive Result, But Marginal

## Round: 14
## Previous round: R13 (Hidden-State Contrastive KD Design)

---

## CRITICAL EVIDENCE (READ ALL BEFORE DECIDING)

### 1. R13 HIDDEN-STATE KD: FIRST EVER POSITIVE KD RESULT

After 11 failed logit KD experiments (all 0.00 BPT improvement), hidden-state contrastive KD via byte-span InfoNCE produced the **first measurable improvement from scratch**:

| Step | CE-WSD Control | State KD (R13) | Delta |
|------|---------------|----------------|-------|
| 500  | 7.6915 | 7.5780 | -0.114 |
| 1000 | 6.7860 | 6.4738 | **-0.312** (peak) |
| 1500 | 6.0502 | 6.0146 | -0.036 (compressed) |
| 2000 | 5.7261 | 5.4598 | -0.266 (reopened) |
| 2500 | 5.3518 | 5.1334 | -0.218 |
| 3000 | 4.9703 | **4.8576** | **-0.113** |

**What this proves:**
1. Cross-tokenizer representation transfer IS possible — vocabulary mismatch was the bottleneck, not teacher-student capacity gap
2. The improvement is NOT a head-start artifact — gap reopened after compression at step 1500
3. Improvement is consistent across all exits (7: -0.080, 15: -0.089, final: -0.113)
4. The model has healthier activations (kurtosis 0.71 vs 0.85)

**BUT: 4.8576 > 4.82 R13 kill threshold.**

### 2. THE GAP PROFILE IS INTERESTING

The gap peaked at -0.312 (step 1000), compressed to -0.036 (step 1500), then reopened to -0.266 (step 2000). This oscillating pattern suggests:
- Hidden-state KD has strongest effect in the early-to-mid training regime
- The compression at step 1500 might be the transition where the student's own learning starts dominating
- The reopening at step 2000 suggests the KD signal provides genuine structural guidance that CE alone doesn't

### 3. WHAT WORKED VS WHAT DIDN'T IN PRIOR EXPERIMENTS

**ALL failed (11 experiments):**
- Cross-tokenizer logit KD: DSKD ETA alignment is too lossy (byte-offset + shared vocab)
- Warm-start KD: TAID, plain KD, reverse KL, AKL, sparse replay, surface matching, KnowledgePorts — all 0.00-0.01 BPT
- From-scratch logit KD: alpha=0.3/0.9 with tau=0.5/1.0 — catastrophic or noise

**The ONLY thing that worked:**
- Hidden-state contrastive KD via byte-span pooled InfoNCE (-0.113 BPT at 3K)

**Also validated:**
- WSD schedule: -0.42 BPT free improvement
- TAID over plain KD: +0.052 BPT (from earlier Qwen3-0.6B teacher experiments)

### 4. IMPLEMENTATION DETAILS OF R13

- 3 independent linear projectors (768→2048, bias=False)
- Student layers [7, 15, 23] → Teacher layers [8, 16, 24] (Qwen3-1.7B, 28 layers)
- InfoNCE cosine loss, temperature=0.07, n_spans=32
- Depth weights: [0.30, 0.50, 0.20] — middle layer gets most weight
- Alpha ramp: 0.06→0.30 (1-400), hold at 0.30 (401-2200), decay 0.30→0.03 (2201-3000)
- Additive loss: L = L_CE + alpha * L_state (NOT mixture)
- Projector LR=6e-4 (2x base), WD=0.01
- Throughput: ~10 steps/min (limited by Python-level byte_span_pool loops)

### 5. WHAT WE HAVE

- Architecture: Sutra-24A-197M (24L, 768d, 12h GQA, SwiGLU, SS-RMSNorm, exits at 7/15/23)
- Teacher: Qwen3-1.7B-Base (2048d, cached, ~3.4GB VRAM)
- Working train_state_kd_scratch function with resume support
- compute_span_infonce_loss (validated — produces genuine improvement)
- byte_span_pool (works but very slow — Python loops)
- 22.9B tokens in 246 shards
- Single RTX 5090 (24GB)
- CE-WSD 3K baseline: BPT=4.9703
- State KD 3K result: BPT=4.8576

### 6. USER DIRECTIVE (MANDATORY)
"No small deviations should be considered. If it's knowledge transfer from some of the best models in the world, we should be growing much quicker."

The bar is DECISIVE, VISIBLE growth. -0.113 BPT is measurable but not decisive.

---

## QUESTIONS FOR CODEX

1. **Continue or kill?** 4.8576 technically exceeds the 4.82 kill threshold, but this is the first positive KD result ever. Should we optimize the approach or pivot?

2. **If continue: what to change?** The most promising knobs:
   - Higher alpha_max (0.30 was conservative for an additive loss — the literature uses 0.5-1.0 for state KD)
   - Longer hold phase (gap compressed at 1500, maybe ramp was too fast)
   - Different projector (MLP vs linear?)
   - Fewer spans (32 might be too fine-grained, losing the global structure — try 8 or 16?)
   - MSE loss instead of InfoNCE (simpler, possibly more direct gradient signal)
   - Combine with CE auxiliary exits (the exit losses might interfere with KD)

3. **If pivot: where to?** Options:
   - Same-tokenizer teacher training (expensive but guarantees logit KD works)
   - Offline enriched-data KD (teacher processes data offline, student trains on enhanced data)
   - Hybrid: hidden-state KD + TAID logit KD (TAID works with Q0.6, maybe with Q1.7B too if combined with state KD)
   - Abandon single-teacher, go multi-teacher with committee routing

4. **The oscillating gap pattern** — what does the compression-reopening at steps 1000-2000 tell us about the learning dynamics? Should the alpha schedule be reshaped to match these phases?

5. **Scale-up feasibility** — if -0.113 at 3K, what can we expect at 6K, 15K, 60K? Does hidden-state KD benefit scale more (deeper training) or less (diminishing returns)?

Design a CONCRETE protocol with EXACT parameters. No ranges. The next experiment must either show decisive improvement or definitively kill hidden-state KD.
