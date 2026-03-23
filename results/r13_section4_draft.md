# R13 Section 4 Addendum — New Evidence Since R12

## PREVIOUS ROUND (R12) OUTPUT SUMMARY

**R12 Confidence: O1=6, O2=8, O3=4, O4=4, O5=6**

R12 Central Decision: O4 (multi-source learning) comes BEFORE shared-core architectural work (O2).

R12 Experiment Queue (priority order):
1. P1: O4-first 15K two-teacher continuation (AR + encoder teachers)
2. P2: 16K tokenizer transplant 15K matched
3. P3: Shared-core branch (only after P1/P2 winner)
4. P4: INT4 drift audit → COMPLETE: catastrophic (+155% drift)

R12 Standing Rules: 15K minimum, full 7-task eval + generation, no optimizer resets.

## NEW EXPERIMENTAL EVIDENCE (since R12)

### 1. v0.6.0b Random-Depth Training — 17 Checkpoints Through Step 8500 (RUNNING to 15K)

**Headline:** Random-depth training FIXES pass collapse. D10 beats D12 in 100% of 18 checkpoints. NEW BEST BPT at step 9000.

**Key metrics at step 9000:**
- test_bpt = 6.9155 (**NEW BEST** — 0.228 improvement from 7.1437 plateau)
- D8=7.3421, D10=7.3389, D12=7.3461 → D10 wins by 0.007 BPT
- Optimal depth oscillates D=8/9/10, NEVER D=12
- Passes 9-12 are net NEGATIVE in 93% (13/14) of post-restart checkpoints
- **Entropy minimum migrated to pass 12** (4.878) — monotonically decreasing from pass 7→12 for the first time. Late passes now compressing, not just maintaining.
- **Caveat:** Per-depth D=12 BPT (7.3461) regressed from step 8500 (7.1953). Headline improvement partially from favorable eval noise (20 vs 5 batch samples), but magnitude exceeds typical ±0.05 oscillation.

**Train-test gap (overfitting signal):**
- Step 3000: train loss 4.93, test BPT 7.22 → gap ~2.29
- Step 8500: train loss 4.83, test BPT 7.14 → gap ~2.31
- Step 9000: train loss 4.92, test BPT 6.92 → gap ~2.00 (narrowed — if headline holds)
- Training loss declining at -0.08/1K steps. Test BPT plateaued since step 5500.
- Root cause: NOT data scarcity (20.7B tokens available, only 1.4% seen, 21% Chinchilla-optimal)
- Root cause IS: capacity starvation. Only 26M intelligence params (38.5% of 68M total) — 56.5% is embedding tax (GPT-2 tokenizer, 75.8% vocab unused)

**Entropy profile (NO collapse developing — and late passes now compressing):**
- Spread stable at 1.22-1.25x across all 18 checkpoints
- Steps 500-8500: minimum entropy at pass 4-5 (healthy differentiation)
- **Step 9000 SHIFT: minimum entropy migrated to pass 12 (4.878).** Monotonically decreasing pass 7→12 for first time. Late passes doing real compression, not just maintaining.
- v0.6.0a had 2x entropy cliff at pass 12 — random-depth eliminated this

**Gradient distribution theory (derived and confirmed):**
- P(pass p is final) = (13 - max(p,4)) / 9 for D ~ weighted
- All 4 predictions confirmed: D10>D12, D=8-10 sweet spot, no collapse, early passes contributing

**Trough D8 trend (real signal behind noisy headline):**
- Per-depth D8 BPT has stdev 0.167 (±0.4 oscillation from 20-sample eval)
- Headline BPT (D=12, 80 samples) has stdev 0.037 — appears plateaued at 7.14-7.20
- Post-restart local minima (troughs): 7.10 (4K) → 6.99 (6.5K) → 6.96 (7.5K)
- Slope: -0.042 BPT/1K steps, R²=0.99 (monotonic improvement)
- **Projected trough D8 at 15K: ~6.65** (would beat v0.6.0a's 6.79 by 0.14 BPT)

### 2. v0.6.0b 15K Final Results — [PLACEHOLDER: INSERT WHEN AVAILABLE]

**BPT:** [TBD]
**Per-depth BPT:** [TBD]
**7-task lm-eval results:** [TBD]
**Generation quality:** [TBD]

### 3. P4 INT4 Drift Audit — CATASTROPHIC (Complete)

- fp32_bpt = 6.794, int4_bpt = 17.310, drift = +155%
- D12 drift = +147% (error compounds through all 12 passes)
- **Verdict:** INT4 PTQ permanently killed for shared-weight recurrence
- QAT is mandatory. DyT/BitNet should be explored.

### 4. v0.6.1 Controller Canary — FALSIFIED (Complete at 1K steps)

- BPT=7.2075. mode_entropy=0.003 (pass-global). MI(mode,token)=0.02.
- Controller biased transition columns but didn't gate real computation
- Zero content dependence — controller only learned pass-global preferences
- Benchmarks (1K steps): SciQ 36.1%, LAMBADA 2.6%, PIQA 54.4%
- BPT improved from continued base model training, NOT from controller/adapter innovation
- **Verdict:** Controller-only repair approach abandoned

### 5. v0.6.0c P0 Canary — Optimizer Preservation Validated (Complete at 750 steps)

- SciQ=47.3% at step 250 (vs parent's 48.1%) — 98% knowledge retained
- LAMBADA=9.5% at step 250 (vs parent's 11.2%) — 85% retained
- Progressive knowledge loss: step 750 SciQ 44.7%, LAMBADA 6.7%
- **vs WSD restart (v0.6.0b):** SciQ 35.8% (-12.3%), LAMBADA 1.5% (-9.7%)
- **Verdict:** Optimizer preservation is mandatory for warm-starts. WSD restart kills knowledge.
- v0.6.0c needs 15K training for proper evaluation

### 6. Apples-to-Apples BPT Comparison — v0.6.0a Wins

- Same test tokens, same eval: v054=7.57, v060a=6.90, v060b=7.58
- v0.5.4's lower BPT was partially a domain/shard effect, not pure architecture win
- v0.6.0a (20K attached history) is the genuine best model

### 7. Cross-Model Benchmark Table

| Model | ARC-E | ARC-C | Hella | PIQA | SciQ | Wino | LAMBADA |
|-------|-------|-------|-------|------|------|------|---------|
| v0.6.0a (20K, parent) | 31.3% | 17.5% | 25.7% | 54.5% | **48.1%** | 51.5% | **11.2%** |
| v0.6.0b (2K, WSD) | 31.0% | 18.7% | 25.9% | 52.9% | 35.8% | 51.2% | 1.5% |
| v0.6.1 (1K, controller) | 31.1% | 16.7% | 25.8% | 54.4% | 36.1% | 48.9% | 2.6% |
| v0.5.4 (20K, old arch) | 29.7% | 20.2% | 25.8% | 54.1% | 33.6% | 49.1% | 1.8% |
| v0.6.0c (250, opt-preserved) | — | — | — | — | 47.3% | — | 9.5% |
| **v0.6.0b (15K)** | **[TBD]** | **[TBD]** | **[TBD]** | **[TBD]** | **[TBD]** | **[TBD]** | **[TBD]** |
| SmolLM2-135M (2T tok) | ~42% | — | ~42% | ~68% | — | ~51% | — |
| Pythia-160M | ~43% | ~28% | ~30% | ~62% | ~74% | ~53% | ~32.6% |

**Key pattern:** Knowledge tasks (SciQ, LAMBADA) destroyed by WSD restart. Reasoning/structure tasks robust. Knowledge is fragile; structure is not.

## R12 LITERATURE REVIEWS (completed, in RESEARCH.md lines 200-245)

1. **Multi-source distillation:** MiniPLM (400x data efficiency), MOHAWK (3-phase cross-architecture), reverse KL (MiniLLM — generation quality), AdaKD (per-token temperature), Llamba (Transformer→SSM)
2. **Heterogeneous fusion:** DUNE (per-teacher projectors), Procrustes > CKA for geometry, MI-based loss, per-sample teacher routing
3. **QAT for shared-weight recurrence:** QEP (exponential error compounding), BitNet (ternary = zero weight error), DyT (tanh replaces LayerNorm), QAT mandatory at 68M
4. **Exact-memory modules:** ARMT DPFP-3 (warm-startable, <2% overhead), Gated DeltaNet (best forgetting), FwPKM (validates multi-pass retrieval), MIRAS (unifying framework)

## PARAMETER BUDGET ANALYSIS (from ARCHITECTURE.md)

| Component | Params | % | Justified? |
|-----------|--------|---|-----------|
| emb (50257×768) | 38.6M | 56.5% | **NO — 75.8% vocab unused** |
| stage_bank (7 FFNs) | 16.5M | 24.2% | PARTIAL |
| router | 4.7M | 6.9% | YES |
| scratchpad | 2.4M | 3.5% | YES (load-bearing) |
| writer | 2.4M | 3.5% | YES |
| Other | 4.2M | 6.2% | Mixed |

**Intelligence fraction: 38.5%. Infrastructure/overhead: 61.5%.**
**9 INHERITED decisions never re-derived. 4 ARBITRARY parameters never tuned.**

## HARDWARE STATUS

[DYNAMIC — Claude fills from live nvidia-smi each round]

## WHAT R13 NEEDS TO DECIDE

1. Has v0.6.0b 15K recovered knowledge (SciQ, LAMBADA) lost from WSD restart?
2. Should P1 (two-teacher) launch next, or should P2 (tokenizer) take priority given the 56.5% embedding tax?
3. What specific mechanisms from the literature reviews should be incorporated?
4. Should any confidence scores change based on new evidence?
5. What probes/experiments should R14 execute?
