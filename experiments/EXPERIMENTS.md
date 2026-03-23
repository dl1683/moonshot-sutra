# Experiments Summary

Reverse chronological. See ledger.jsonl for machine-readable details.

### [OK] p4-int4-drift-audit (2026-03-23)
**Purpose:** P4: INT4 post-training quantization drift audit on v0.6.0a step 20K
**Key metrics:** fp32_bpt=6.794, int4_bpt=17.310, drift=+155%, D12 drift=+147%
**Learned:** CATASTROPHIC: INT4 PTQ destroys shared-weight recurrent models. Error compounds over 12 passes. QAT mandatory. DyT/BitNet should be explored early. Validates NeurIPS 2025 QEP prediction.

### [RUNNING] v060b-rd12-15k (2026-03-22 → ongoing)
**Purpose:** v0.6.0b-rd12 extended to 15K steps (WSD restart, random-depth from v0.6.0a)
**Key metrics (step 6500):** best_bpt=7.1434 (step 5500), D8=6.985, D12=6.991, D8 beats D12
**Notes:** LR schedule discontinuity at step 3000→3500 (cosine recalculated 5K→15K, 2.17x LR jump). WSD optimizer reset at start destroyed knowledge (SciQ -12.3%, LAMBADA -9.7%). Pass collapse fixed. ETA 15K: ~14:30 2026-03-23.

### [OK] v060c-p0-canary (2026-03-23)
**Purpose:** P0: Knowledge-preserving random-depth canary (optimizer preserved from v0.6.0a)
**Key metrics:** SciQ=47.3% (step 250), LAMBADA=9.5% (step 250), D8-D12 gap=-0.002
**Learned:** Optimizer preservation recovered 70-90% of knowledge vs WSD reset. Step 250 is sweet spot (pass collapse fixed, knowledge barely degraded). Progressive knowledge loss with more random-depth steps (step 750: SciQ 44.7%, LAMBADA 6.7%). Rolling checkpoint lost — must restart from parent for 15K.

### [FAIL] v061-controller-canary (2026-03-23)
**Purpose:** v0.6.1 TokenwiseController + PassAdapters (falsified at 1K steps)
**Key metrics:** BPT=7.2075, mode_entropy=0.003 (pass-global), SciQ=36.1%, LAMBADA=2.6%
**Learned:** Controller-only repair FALSIFIED. Modes biased transition columns but didn't gate real computation. Zero content dependence (MI(mode,token)=0.02). BPT improved from continued base training, NOT controller/adapter innovation. Branch concept abandoned.

### [OK] v060a-20k-complete (2026-03-22)
**Purpose:** v0.6.0a training milestone: 20K steps complete
**Key metrics:** best_bpt=6.7946, sciq=48.1%, piqa=54.5%, lambada_acc=11.2%, pass_collapse_cos=0.293
**Learned:** New best BPT at 20K (beat 17K's 6.8284). Pass collapse confirmed structural and stable (entropy cliff at pass 12 unchanged from 14K). SciQ and PIQA show real learning. LAMBADA 11.2% is meaningful signal. Training stopped for v0.6.0b-rd12 warm-start.

### [OK] collapse-metrics-step10K (2026-03-22)
**Purpose:** Measure pass-to-pass collapse profile at step 10K
**Key metrics:** final_bpt=7.3525, total_improvement=12.654, late_improvement=11.5775
**Learned:** SEVERE COLLAPSE: passes 0-9 cosine 0.93-0.997 (near fixed-point). Pass 10→11 does 60% of ALL compression (15.02→7.35

### [OK] probe-token-type-dynamics-14k (2026-03-22)
**Purpose:** Per-token-type pass entropy dynamics at step 14K
**Key metrics:** ws_entropy_floor=3.71, fw_entropy_floor=4.49, hard_entropy_floor=5.33
**Learned:** Collapse is UNIVERSAL across token types but entropy floor differs. Whitespace compresses most (drop 6.47 bits), hard to

### [OK] probe-gen-quality-14k (2026-03-22)
**Purpose:** Generation quality test at step 14K best checkpoint, temp=0.8
**Key metrics:** trigram_diversity=0.973, top5_repeat_ratio=0.032
**Learned:** Major improvement from 9K (trigram_div 0.265 -> 0.973). Reasonable English syntax but no factual knowledge, number repet

### [OK] probe-pass-dynamics-14k (2026-03-22)
**Purpose:** Per-pass hidden state dynamics: cosine, delta norm, logit entropy at step 14K
**Key metrics:** cos_pass10_11=0.236, delta_norm_pass11=2.109, logit_entropy_pass11=5.09
**Learned:** EXTREME pass collapse confirmed at hidden state level. cos(p10,p11)=0.236 (near-orthogonal), logit entropy flat at 10.1 

### [OK] probe-knn-ceiling-14k (2026-03-22)
**Purpose:** kNN-LM ceiling test: post-hoc exact retrieval at 128K datastore, step 14K
**Key metrics:** whitespace_delta_pct=-2.8, function_word_delta_pct=0.0, hard_delta_pct=0.0
**Learned:** NEGATIVE result: kNN retrieval HURTS all categories except whitespace. 128K datastore too small, model representations t

### [OK] lstep-exact-3arm (2026-03-22)
**Purpose:** 3-arm L_step probe: baseline (sampled) vs exact (full-vocab CE on passes 7-10) v
**Key metrics:** baseline_bpt=9.066, exact_bpt=9.209, none_bpt=9.005
**Learned:** DEFINITIVE: Sampled L_step is Goldilocks zone. Exact CE causes catastrophic pass convergence (passes 8-11 all produce BP

### [OK] probe-pass-disagreement-rerun (2026-03-22)
**Purpose:** Rerun pass disagreement with fixed collect_history bug, step ~13900
**Key metrics:** ws_disagree=0.7432, fw_disagree=0.7821, number_disagree=0.6361
**Learned:** INVERTED disagreement: easy tokens HIGH (0.78), hard tokens LOW (0.61). Model lacks mechanism to vary strategy by token 

### [OK] probe-embedding-svd (2026-03-22)
**Purpose:** SVD factorization feasibility of tied embedding matrix at ranks 64/128/256
**Key metrics:** rank64_var=0.176, rank128_var=0.265, rank256_var=0.434
**Learned:** RISKY: Embedding matrix is nearly full-rank (flat singular spectrum after SV1=1773, SVs 2-768 all ~238). Rank-128 captur

### [OK] probe-d-scratchpad-audit (2026-03-22)
**Purpose:** 4-mode scratchpad ablation: full vs no_read vs no_write vs removed
**Key metrics:** full_bpt=5.8353, no_read_bpt=6.026, no_write_bpt=6.0201
**Learned:** LOAD-BEARING: +3.27% BPT when removed. Read path is what matters (no_read = removed). Write alone also hurts (+3.17%). R

### [OK] gen-quality-step9000 (2026-03-22)
**Purpose:** Generation quality test across 15 diverse prompts at step 9000
**Key metrics:** greedy_trigram_diversity=0.265, greedy_ascii_ratio=1.0, greedy_word_diversity=0.26
**Learned:** Severe repetition in greedy decoding (trigram_div=0.265). Model loops within 10-20 tokens. No factual knowledge. Expecte

### [OK] probe-e-pass-disagreement (2026-03-22)
**Purpose:** Measure per-token pass disagreement as intrinsic uncertainty signal for elastic 
**Key metrics:** best_feature=ce_spread, best_correlation=0.959, future_gain_q4_vs_q1_ratio=4.79
**Learned:** STRONG POSITIVE: ce_spread r=+0.96 with future gain. High-disagreement tokens get 4.8x benefit from additional passes. P

### [OK] lstep-ablation-2arm (2026-03-22)
**Purpose:** 2-arm L_step ablation: baseline (sampled CE) vs no L_step
**Key metrics:** baseline_bpt=9.054, no_lstep_bpt=9.162, baseline_late_improv=12.35
**Learned:** Sampled L_step is BENEFICIAL. Without it, late improvement drops 19% (13.5→11.0) and best pass shifts from 11→10. Go

### [OK] lm-eval-step12K (2026-03-22)
**Purpose:** Directional lm-eval benchmarks on step ~12K checkpoint
**Key metrics:** piqa=0.5479, arc_easy=0.279, hellaswag=0.2572
**Learned:** Near-random on all except PIQA (+4.8%). Expected at 68M/0.4B tokens from scratch. LAMBADA near-zero confirms exact-recal

### [OK] v060a-step10K (2026-03-22)
**Purpose:** v0.6.0a eval checkpoint
**Key metrics:** test_bpt=7.4888, best_bpt=7.4888, lr=0.000344
**Learned:** BPT still monotonically improving. Power law predicted 7.38, actual 7.49 (slightly worse). Improvement 9K→10K = 0.052 

### [OK] v060a-gradient-path-audit (2026-03-21)
**Purpose:** Trace which parameters each v0.6.0a loss term updates before re-launching dense-
**Key metrics:** l_final_grad_norm=15.45, l_step_grad_norm=0.28, l_probe_grad_norm=0.29
**Learned:** Confirmed detached-history split: L_final updates recurrent core + readout, L_step updates only final LN + tied embeddin

### [??] fineweb-tokenize-v2-sharded (2026-03-20)
**Purpose:** Tokenize FineWeb-Edu with shard-based disk writes
**Key metrics:** N/A
**Learned:** RUNNING: shard-based, peak RAM ~4GB, writes every 200 chunks. Restart from scratch.

### [FAIL] fineweb-tokenize-v1 (2026-03-20)
**Purpose:** Tokenize 40GB FineWeb-Edu with GPT-2 BPE
**Key metrics:** tokens_completed=4324885915, pct_complete=49.5
**Learned:** FAIL: OOM at 49.5% (4.32B tokens). Python list accumulation ~70GB RAM. Fixed with shard-based writes.

### [OK] chrome-true-optimal (2026-03-20)
**Purpose:** True optimal: Peri-LN + Pheromone + Grokfast (no Surprise Bank)
**Key metrics:** test_bpt=6.154, baseline_bpt=7.317, delta_pct=15.9
**Learned:** TRUE OPTIMAL: +15.9% BPT. Removing Surprise Bank drag gained extra 2.3pp over ablation best.

### [OK] chrome-v054-ablation-6arm (2026-03-20)
**Purpose:** 6-arm ablation: baseline, Peri-LN, +Surprise, +Pheromone, Full, +Grokfast
**Key metrics:** baseline_bpt=7.317, peri_ln_bpt=6.888, peri_surprise_bpt=7.004
**Learned:** Winner: v0.5.4+Grokfast (+13.6%). Surprise Bank KILLED (hurts all arms). Peri-LN+Pheromone=+5.9%.

### [OK] chrome-peri-ln (2026-03-20)
**Purpose:** Test Peri-LN (pre+post LayerNorm on each sublayer)
**Key metrics:** test_bpt=6.995, baseline_bpt=7.182, delta_pct=2.6
**Learned:** PASS: +2.6% BPT with fewer params. Zero-cost normalization. Already in OLMo2/Gemma2/Gemma3.

### [OK] chrome-grokfast-a095-l2 (2026-03-20)
**Purpose:** Test Grokfast gradient EMA filter (alpha=0.95, lambda=2.0)
**Key metrics:** test_bpt=6.468, baseline_bpt=7.265, delta_pct=11.0
**Learned:** STRONG PASS: +11.0% BPT. Benefit grows with training steps. 5 lines of code.

### [FAIL] chrome-grokfast-a098-l5 (2026-03-20)
**Purpose:** Test Grokfast (alpha=0.98, lambda=5.0) - too aggressive
**Key metrics:** test_bpt=NaN, baseline_bpt=7.265
**Learned:** KILL: lambda=5.0 too aggressive, caused NaN divergence.

### [FAIL] chrome-v054-error-scratchpad (2026-03-20)
**Purpose:** Test error scratchpad (write delta not state) on v0.5.3
**Key metrics:** test_bpt=7.343, baseline_bpt=7.317, delta_pct=-0.35
**Learned:** KILL: early noise from meaningless deltas. Late steps 2.2x better but overall BPT worse.

### [FAIL] chrome-v054-pheromone-router (2026-03-20)
**Purpose:** Test pheromone-style decaying scalar trace in router
**Key metrics:** test_bpt=7.327, baseline_bpt=7.317, delta_pct=-0.14
**Learned:** KILL: marginal BPT but best late-step value. Pheromone trace needs time to accumulate.

### [FAIL] chrome-v054-depth-drop (2026-03-20)
**Purpose:** Test depth-drop bootstrap (random truncation + KL to teacher)
**Key metrics:** test_bpt=NaN, baseline_bpt=7.317
**Learned:** KILL: KL to teacher diverged to NaN at step 169. Numerically unstable at dim=128.

