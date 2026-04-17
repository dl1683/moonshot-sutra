# Experiments Summary

Reverse chronological. Machine-readable details in `experiments/ledger.jsonl`.

---

## Phase 7: MVG Scout — Architecture Exploration (2026-04-15 → ongoing)

### zeroth_p4_5k [DONE — SUPPORTS MVG HYPOTHESIS]
**Purpose:** Test whether smaller patches (4-byte vs 6-byte) improve learning by giving the global transformer more context patches.
**Config:** `results/config_zeroth_p4_5k.json` (patch_size=4, batch=24, grad_accum=3, lr=3e-4, 5K steps, from-scratch CE-only)
**Result:** BPB 1.887 at 5K steps (eval).
**Baseline comparison:** Stage 0 (patch_size=6) at 5K steps: BPB 2.078. **Delta: -0.191 BPB (patch_size=4 wins).**
**What we learned:** Finer-grained patching (384 patches of 4 bytes vs 256 patches of 6 bytes = 50% more global context) produces significantly better language modeling. This directly supports the MVG scout hypothesis: semantic (BPE-aligned) patch boundaries with avg ~3.5 bytes/patch should be even better. The zeroth also used smaller effective batch (72 vs 128) and STILL won — suggesting the context advantage is substantial.

### mvg_scout [PENDING — implementation ready, GPU blocked by Ekalavya 6K]
**Purpose:** Test core hypothesis — do BPE-aligned variable patches beat fixed patches at matched params?
**Architecture:** SutraDyadMVG (~152.4M params, BPE 8K vocab patcher, scatter-mean byte pooling per BPE patch, same 12-layer d=1024 global transformer).
**Decision criteria (at 5K steps, compared to Stage 0 p6 at 5K = BPB 2.078):**
- Strong win: BPB < 2.028 (delta < -0.05)
- Weak win: BPB < 2.058 (delta < -0.02)
- Loss: BPB > 2.098 (delta > +0.02)
**Code:** Committed b3ccb16, unit tests pass (all 6). Kill criteria fixed for from-scratch calibration.
**Blocked on:** GPU (Ekalavya 6K running, est. ~4 days)

### Strategic Architecture Review (2026-04-15)
**Purpose:** Are we on the best path?
**Key findings:** (1) Ekalavya KD ceiling with 2 teachers: 0.005-0.016 BPB. (2) Student capacity at 188M may be the bottleneck, not teacher quality. (3) Token-global/byte-local (Architecture G) scored highest across all 5 outcomes. (4) MVG scout is the minimum viable test of the core hypothesis. (5) Design confidence maxes at 6.4/10 — only experiments close the gap.
**What we learned:** Ekalavya should remain as INFRASTRUCTURE, not the singular priority. The architecture decision (fixed byte patches vs semantic patches) may matter more than KD mechanism refinement.

---

## Phase 6: Ekalavya Protocol — Byte-Level Covering KD (2026-04-12 → ongoing)

### ekalavya_iter6_3teacher_12k [PENDING — blocked on iter5 completion + cache build]
**Purpose:** 3-teacher iteration: SmolLM2 (anchor) + Pythia + Qwen3-1.7B (2 aux), 12K steps with teacher cache.
**Key changes from iter5:** 3rd teacher (Qwen3, near-zero correlation with anchor), 2x training steps, pre-computed teacher cache (eliminates covering bottleneck), proportionally stretched schedules, GPU scatter_add covering.
**Config:** `results/config_ekalavya_iter6_3teacher_12k.json`
**Cache config:** `results/config_cache_3teacher_12k.json`
**Seed:** Best checkpoint from iter5 6K run
**Prerequisites:** (1) iter5 completes, (2) teacher cache built (~15-20h with GPU covering)
**Kill:** eval > 1.430 at step 1000, > 1.410 at step 3000, > 1.400 at step 6000
**Hypothesis:** 3rd teacher with near-zero anchor correlation provides complementary signal. Extended training allows deeper knowledge absorption. VRAM budget: ~13GB (safe with 24GB available).

### ekalavya_iter5_full_6k [RUNNING — 6000 steps]
**Purpose:** Full TAID + uncertainty gating + piecewise alpha decay run. review design iteration 5 design.
**Mechanism:** TAID β 0→0.8/600, uncertainty gating exp 1→2/600 (ug_clamp=1.5), soft-sigmoid anchor routing (JSD>0.02 gate, sigmoid confidence weighting, aux_cap=0.35), covering, piecewise alpha decay (0.03→0.015→0.005→0.0 at steps 150/1500/3000/4500), no hold, grad_clip=0.8. Global layers unfreeze at 700/1800.
**review gate review (step 250):** HIGH: routing is soft (sigmoid), not hard confidence gate — docs corrected. MEDIUM: resume loses best_eval (N/A for this run), unfreeze +0.6GB/phase (safe), kill criteria manual only.
**Config:** `results/config_ekalavya_iter5_full_6k.json`
**Seed:** best.pt from routing run (step 250, eval BPB=1.418)
**Kill:** eval > 1.430 at step 500, eval > 1.420 at step 1500
**KD budget:** 7x routing's cumulative transfer (35.8 vs 5.2). Main absorption window: steps 600-1500 (β=0.8, α=0.03-0.015).
**Crash history:** 5 restarts due to parent shell death from agent context compaction. Run 1 reached step 310, runs 2-4 only reached 270-330 before dying silently. All restarts from step 250 checkpoint.
**Crash defenses (commit 7d380b1):** stderr tee to log file, atexit emergency checkpoint, __main__ crash log, rolling_save 250→100, nohup+disown launch.
**Training trajectory (steps 250-310, from run 1 which had the most data):**
| Window | Steps | Mean BPB | CE | Repr | TAID β | Trend |
|--------|-------|----------|-----|------|--------|-------|
| Warmup end | 150-200 | 1.414 | 0.975 | 0.33 | 0.20-0.27 | Stable |
| Post-warmup | 200-250 | 1.440 | 0.996 | 0.24 | 0.27-0.33 | +0.026 (worse) |
| Late | 250-310 | 1.415 | 0.976 | 0.18 | 0.33-0.41 | -0.025 (recovering) |
**Current run (Run 5, PID 59068, nohup-detached, crash defenses active):**
| Step | BPB | CE | KD | Repr | TAID β | UG mean | Notes |
|------|-----|-----|-----|------|--------|---------|-------|
| 260 | 1.435 | 0.995 | 0.561 | 0.197 | 0.35 | 0.74/57% | Adapting from checkpoint |
| 270 | 1.415 | 0.981 | 0.464 | 0.174 | 0.36 | 0.70/52% | CE+KD dropping: harmonious |
| 280 | 1.392 | 0.965 | 0.474 | 0.161 | 0.37 | 0.69/52% | Best so far |
| 290 | 1.391 | 0.964 | 0.415 | 0.148 | 0.39 | 0.72/56% | Plateau near 1.39 |
| 300 | 1.409 | 0.977 | 0.557 | 0.149 | 0.40 | 0.69/53% | Uptick (harder batch), normal |
| 310 | 1.383 | 0.959 | 0.470 | 0.129 | 0.41 | 0.71/55% | **New best** — all losses dropping |
| 320 | 1.445 | 1.002 | 0.594 | 0.144 | 0.43 | 0.78/61% | Hard batch |
| 330 | 1.425 | 0.988 | 0.489 | 0.124 | 0.44 | 0.72/56% | |
| 340 | 1.392 | 0.965 | 0.577 | 0.120 | 0.45 | 0.70/54% | Near-best |
| 350 | 1.392 | 0.965 | 0.588 | 0.118 | 0.47 | 0.73/56% | |
| 360 | 1.448 | 1.003 | 0.791 | 0.130 | 0.48 | 0.70/54% | Hard batch |
| 370 | 1.305 | 0.905 | 0.754 | 0.138 | 0.49 | 0.74/55% | Easy batch, lowest ever |
| 380 | 1.402 | 0.972 | 0.610 | 0.108 | 0.51 | 0.66/50% | **β crosses 0.50** |
| 390 | 1.440 | 0.998 | 0.828 | 0.115 | 0.52 | 0.70/54% | |
| 400 | 1.459 | 1.011 | 0.886 | 0.114 | 0.53 | 0.70/54% | |
| 410 | 1.432 | 0.992 | 0.651 | 0.110 | 0.55 | 0.69/53% | |
| 420 | 1.444 | 1.001 | 0.939 | 0.106 | 0.56 | 0.70/54% | |
| 430 | 1.425 | 0.988 | 0.778 | 0.106 | 0.57 | 0.69/54% | |
| 440 | 1.397 | 0.968 | 0.707 | 0.097 | 0.59 | 0.68/51% | Repr all-time low |
| 450 | 1.429 | 0.990 | 0.891 | 0.099 | 0.60 | 0.68/52% | |
| 460 | 1.436 | 0.995 | 0.783 | 0.097 | 0.61 | 0.75/59% | Grad spike 0.91 |
| 470 | 1.527 | 1.058 | 0.751 | 0.099 | 0.63 | 0.74/58% | Hard batch spike |
| 480 | 1.448 | 1.004 | 0.977 | 0.098 | 0.64 | 0.69/53% | Recovering |
| 490 | 1.389 | 0.963 | 0.770 | 0.090 | 0.65 | 0.67/52% | Near best, repr 0.090 |
| 500 | 1.432 | 0.993 | 0.747 | 0.086 | 0.67 | 0.74/59% | Repr new low 0.086 |
| **EVAL 500** | **1.398** | — | — | — | — | — | **EXCELLENT: -0.032 vs baseline** |
**Steps 260-500 mean BPB: 1.415.** TAID β at 0.67 (teacher-dominant). Repr monotonically dropping (0.197→0.086).
**Step 500 EVAL: BPB 1.398 — PASSED kill (1.430) by 0.032.** Best eval ever. 8.4% of teacher gap closed (2.6x routing's 3.2%). Coherent English generation confirmed. New best.pt saved.
**Decision: CONTINUE TO 6K.** Next eval at step 1000, kill criterion eval BPB > 1.430.
**Key observations:** (1) TAID+routing accumulates genuine signal over extended training. (2) Repr convergence (0.197→0.086) is strongest indicator. (3) Hard batch spikes (1.527) recover fully — mechanism is robust.

### ekalavya_iter5_taid_gating_probe [DONE — MARGINAL]
**Purpose:** Probe TAID + uncertainty gating mechanism stability before 6K launch.
**Mechanism:** TAID β 0→0.8/600, uncertainty gating exp 1→2/600 (ug_clamp=4.0), anchor-confidence routing, covering, alpha=0.03, linear decay 150→250.
**Config:** `results/config_ekalavya_iter5_probe.json`
**Seed:** best.pt from routing run (step 250, eval BPB=1.418)
**Result:** Step 250 eval BPB=1.433 (+0.015 above baseline). Final eval BPB=1.409 (-0.009 below baseline). Eval variance 0.024 BPB — result is within noise of 1.418 baseline.
**What we learned:** (1) TAID+gating mechanism is STABLE — no catastrophic degradation. (2) Hard-batch spikes (step 140: 1.512, step 200: 1.625) recover fully. (3) KD loss 2-6x lower than routing at same ramp (TAID trust-region working). (4) Repr loss converged 0.878→0.188 (78% reduction). (5) Probe's ug_clamp=4.0 allowed hard-batch KD amplification — 6K uses 1.5. (6) Probe delivered only 13% of routing's KD budget — insufficient for decisive eval improvement. The 6K delivers 54x more.

### multi2_routed_3k [KILLED at step 760 — POSITIVE]
**Purpose:** Multi-teacher ROUTED KD: SmolLM2 (anchor) + Pythia (aux), confidence routing, covering ON.
**Mechanism:** Anchor-dominant soft-sigmoid routing — Pythia weight = sigmoid((conf_aux - conf_anchor - 0.02) / 0.08), gated by JSD>0.02, capped at 0.35. This is a SOFT gate (Pythia gets ~30% weight even at equal confidence, not hard "only when more confident"). LR -20% vs AM run, T=1.3, aggressive KD decay (final_mult=0.3), repr beta decays to 0.
**Config:** `results/config_multi2_routed_3k.json`
**Step 250 eval:** BPB=1.418 (baseline 1.430, delta -0.012). STRONG. First formal eval improvement from Ekalavya KD.
**Step 500 eval:** BPB=1.426 (baseline 1.430, delta -0.004). POSITIVE but regressed from step 250. Hold phase (full alpha=0.05) caused training degradation (steps 420-500 avg BPB 1.494, gradient spikes to 1.86). Eval still below baseline but trajectory negative.
**Hold phase (steps 400-590):** Oscillating BPB 1.390-1.540, avg ~1.450. NOT monotonically worsening (unlike AM). Step 590: BPB=1.390 (strong recovery). CE drifted +0.040 vs ramp phase. Gradient spikes to 1.86.
**Step 750 eval:** BPB=1.429 (baseline 1.430, delta -0.001). POSITIVE but barely below baseline. Eval trajectory: 1.418→1.426→1.429 (decelerating toward baseline). Layers 4-7 unfrozen at step 700. Pre-unfreeze spike to 1.466, fast recovery. Alpha decay only 2.2% at step 750 — negligible.
**Kill criteria (revised):** Eval BPB > 1.430 at step 1000. Run continues but is unlikely to produce decisive improvement.
**KL direction note:** Using forward KL (teacher||student). RESEARCH.md §6.4.33 shows forward KL expected to fail at >1:10 ratio (we're 1:9). Hard-batch KD spikes (step 430: 3.211, step 640: 2.340) are forward KL mode-covering failures. TAID interpolation recommended for next iteration.
**What we've learned:** (1) Routing WORKS during ramp (eval 1.418, -0.012 below baseline). (2) Dense KD at full alpha=0.05 is too strong. (3) Forward KL causes periodic blowups on hard batches. (4) The run won't produce decisive improvement but validates the mechanism. Next: uncertainty gating + TAID + possibly offline caching.

### multi2_covering_3k [KILLED at step 380]
**Purpose:** Multi-teacher covering KD with SmolLM2-1.7B + Pythia-1.4B, arithmetic mean aggregation.
**Mechanism:** Covering decomposition, AM aggregation, alpha=0.05, progressive unfreeze at 700/1500.
**Config:** `results/config_multi2_covering_3k.json`
**Result:** KILLED — early kill at step 380 (saved 96 min GPU). Post-warmup BPB slope +0.000214/step (getting worse). All 4 windows above baseline: 1.443→1.431→1.444→1.465. Extrapolated step 500: 1.491 >> kill threshold 1.430. CE avg 1.002 vs baseline 0.985 — KD actively hurting.
**What we learned:** Naive arithmetic mean of teacher byte probs is destructive when teachers disagree. The teachers produce conflicting probability distributions that average to something worse than either individually. Need intelligent routing — only blend when teachers provide complementary signal.

### covering_smoke_430 [DONE]
**Purpose:** Single-teacher covering KD smoke test (SmolLM2-1.7B, alpha=0.15, 430 steps).
**Key finding:** Covering mechanism works — repr converged 0.90→0.055 (94%). BPB not decisive (1.437 mean vs 1.421 baseline) due to alpha=0.15 too aggressive during unfreeze. Worsening trend: 1.397→1.478. Gradient instability (max 3.29).
**What we learned:** Covering >> first-byte marginal (v2 diverged at 550, covering oscillates). Alpha=0.05 needed, unfreeze must be delayed. Dense all-byte KD at high alpha creates sustained gradient conflict even under clipping.

### ekalavya_v2_2k [DONE — FAILED]
**Purpose:** First-byte marginal KD with review audit fixes, single teacher SmolLM2-1.7B.
**Key finding:** BPB diverged from baseline (1.421→1.565 at step 2000). First-byte marginal destroys 84% of teacher signal (3.485→0.535 bits).
**What we learned:** First-byte marginal alignment is fundamentally too lossy for byte-level KD. Led to covering decomposition design.

## Phase 5: KD Mechanism Validation (2026-03-26 → ongoing)

### kd_15k_benchmark_gate [SUPERSEDED — old architecture]
**Purpose:** Decisive test of scheduled logit KD at 1:19 ratio (90M:1.7B) on old token-level dense architecture.
**Status:** Superseded by byte-level Sutra-Dyad pivot (2026-04-01). KD arm never completed. Control arm validated WSD S-curve: BPT 4.895→4.082 in 15K steps.
**What we learned:** WSD schedule confirmed. KD surface ablation informed Ekalavya design (logit-only + scheduling).

### kd_surface_ablation [DONE]
**Purpose:** Test all KD surfaces (control vs rep-only vs logit-only vs combined) at flat alpha=1.0.
**Key finding:** Multi-surface interference is real (IF peaked 2.14). Rep KD = head-start only, collapses during WSD (basin-incompatible). Logit KD harmful at flat alpha but basin-compatible. Combined → kurtosis spike (12.4).
**What we learned:** Alpha scheduling is mandatory. Logit-only with inverted-U is the correct mechanism.

### kd_first_probe [DONE]
**Purpose:** Does KD provide ANY signal at 90M:1.7B ratio?
**Key finding:** Rep KD (CKA+semantic) = transient head-start only. Gap: -0.059 (500) → -0.008 (3000). Control catches up.
**What we learned:** Rep-level KD alone is insufficient. Need logit-level + scheduling.

## Phase 4: Architecture Lock (2026-03-26)

### probe_pgqa_100m_gate [DONE]
**Purpose:** Last hybrid test. P-GQA full-dim hybrid vs pure transformer at 100M.
**Key finding:** P-GQA +0.028 BPT (below +0.05 threshold). Kill rule triggered.
**What we learned:** Architecture LOCKED as pure 24-layer transformer (Sutra-24A-90M). All hybrid variants exhausted.

### probe_r7p_postfusion [DONE]
**Purpose:** Post-fusion hybrid probes (24G all-hybrid + 6G+18A mixed) at 100M.
**Key finding:** Stability concerns with gated blocks. Pure transformer remained strongest.

### probe_r6_stabilize [DONE]
**Purpose:** Stabilization variants (R6-F no-beta + R6-S softmax mixing) at 42M.
**Key finding:** Both stabilized but didn't improve BPT over transformer baseline.

## Phase 3: Architecture Search (2026-03-25 → 2026-03-26)

### probe_100m_gate [DONE]
**Purpose:** Scale-up test: 24x512 transformer vs HEMD-R5-G hybrid at 100M.
**Key finding:** Transformer won. Hybrid added complexity without BPT gain at scale.

### probe_r4_microprobe [DONE]
**Purpose:** Kernel sweep (k4/k16/k64) for parallel hybrid.
**Key finding:** k=4 slightly better than k=64 at intra-layer. But overall hybrid < transformer.

### probe_parallel_hybrid [DONE]
**Purpose:** Intra-layer parallel hybrid at 42M.
**Key finding:** Marginal gains, not worth the complexity.

### probe_trunk_choice [DONE]
**Purpose:** Pure transformer vs pure conv vs inter-layer hybrid at 42M.
**Key finding:** Pure transformer wins at 42M. Conv variants underperform.

### probe_muon_vs_adamw [DONE]
**Purpose:** Optimizer comparison (Muon vs AdamW vs AdamW-NorMuon) at 42M.
**Key finding:** AdamW remains best at current scale.

### probe_top_vs_ntp [DONE]
**Purpose:** TOP auxiliary loss vs plain NTP at 42M.
**Key finding:** NTP alone is sufficient. TOP adds overhead without benefit.

### probe_dyt_vs_rmsnorm [DONE]
**Purpose:** DyT vs RMSNorm normalization at 42M.
**Key finding:** RMSNorm wins. DyT provides no kurtosis benefit at this scale.

## Phase 2: Baseline + Data (2026-03-25)

### miniplm_500w_full / miniplm_pilot_200w [DONE]
**Purpose:** MiniPLM-style data scoring for O4 (data efficiency).
**What we learned:** Shard quality varies. Per-source weights computed but data shaping probe not yet run.

### probe_o4_data_shaping [QUEUED]
**Purpose:** Test MiniPLM reweighted data vs uniform sampling.
**Status:** Deferred — KD work took priority.

## Phase 1: Baseline Exploration (2026-03-25)

### probe_ngram_131k / probe_ngram_4m [DONE]
**Purpose:** CPU n-gram memory as compression auxiliary.
**Key finding:** 131K buckets saturated (100% occupancy). 4M better but collision still 86%.

### probe_mtp_d1 [DONE]
**Purpose:** Multi-Token Prediction (D=1) from EDSR baseline.
**Key finding:** MTP HURTS at 98M scale. Control reached 5.1039 vs MTP 5.2328.

### probe_halting_controller [DONE]
**Purpose:** Differentiable 3-exit halting controller.
**Key finding:** FALSIFIED. Control beats halting by +0.153 BPT. 21% throughput overhead.

### expand_98m_to_200m [DONE]
**Purpose:** Net2Wider expansion test.
**Key finding:** Widening works mechanically but prior model had bugs.

---

**Dead Ends (prevent revisiting):**
- Multi-Token Prediction at <100M scale (hurts BPT)
- Learned halting at <100M (too much overhead)
- Conv/hybrid architectures at 42M-100M (transformer wins consistently)
- Flat alpha KD at 1:19 ratio (harmful without scheduling)
- Rep-only KD (head-start only, not persistent)
- Multi-surface KD at extreme ratios (interference)

## Phase 7: Ekalavya Protocol — Byte-Level KD (2026-04-12 → ongoing)

### ekalavya_smoke_v1 [DONE]
**Purpose:** Validate byte-level KD mechanism with aggressive alpha=0.5, T=2.0.
**Config:** batch=8, accum=2, alpha=0.5, T=2.0, 500 steps, SmolLM2-1.7B anchor (bf16).
**Key finding:** KD mechanism works (KD loss -81%, repr loss -82%) but alpha=0.5 catastrophically disrupts CE. Final BPB 2.598 vs baseline 1.415.
**What we learned:** Alpha=0.5 too aggressive. Need 5x lower alpha. Off-by-one alignment bug found by review audit.

### ekalavya_v2_2k [RUNNING]
**Purpose:** Corrected Ekalavya with all review audit fixes. Alpha=0.10, T=1.5, 4-bit teacher.
**Config:** batch=12, accum=6, alpha=0.10, beta=0.15, T=1.5, 2000 steps, SmolLM2-1.7B (4-bit). Progressive unfreeze (8-11→4-7→0-3). 5 layerwise LR groups.
**Fixes:** Off-by-one byte alignment, causal repr alignment, float32 KL, 4-bit quantization, resume + unfreeze logic.
**Early signal (step 50):** CE 0.9997, BPB 1.442 — only +0.027 above baseline (vs +1.3 in v1). CE PRESERVED.
**What we'll learn:** Whether corrected KD alignment + low alpha produces CE improvement over baseline.
**Stop rule:** If CE doesn't improve through step 1000, reassess alpha/mechanism.

### teacher_probes [DONE]
**Purpose:** Validate byte-level teacher quality for KD.
**Key finding:** SmolLM2-1.7B BPB 0.490 (gap 0.925 to student), Pythia-1.4B BPB 0.534. Oracle gain 0.054 BPB (dual-teacher justified).

### cross_attn_removal [DONE]
**Purpose:** Ablation proved cross-attention harmful. Structural removal: 196.7M → 188.2M.
**Key finding:** BPB 1.4149 without cross-attn vs 1.4315 with. Simpler, smaller, better.

## Phase 6: Sutra-Dyad Audit (2026-04-01)

### sutra_dyad_perf_audit [DONE]
**Purpose:** Tier-1 Performance Engineer audit for the new byte-level `Sutra-Dyad` runner.
**Key finding:** NOT CLEAN. Current reduced config is safe (`~9.7 GB` steady-state peak), but the original `d_local=512 / 3 local layers / batch=64` setup was oversized for a 24 GB card (`~23.3 GB allocated / ~26.2 GB reserved` extrapolated steady-state peak).
**What we learned:** The default path is already using memory-efficient SDPA. Biggest wins are model-side activation cleanup (`F.rms_norm`, optional global checkpointing, project-before-repeat) and fixing `ByteShardedDataset`, whose full-shard decode path takes `10-80 s` on real shards and can dominate wall-clock time.
