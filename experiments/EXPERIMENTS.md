# Experiments Summary

Reverse chronological. Machine-readable details in `experiments/ledger.jsonl`.

---

## Phase 6: Ekalavya Protocol — Byte-Level Covering KD (2026-04-12 → ongoing)

### ekalavya_iter5_taid_gating_probe [RUNNING — 250 steps]
**Purpose:** Probe TAID + uncertainty gating from Codex T+L iteration 5. Validate that progressive intermediate targets (TAID) + per-position gating fix hold-phase degradation.
**Mechanism:** TAID β 0→0.8/600, uncertainty gating exp 1→2/600, anchor-confidence routing, covering, alpha=0.03 (lowered from 0.05), no hold, grad_clip=0.8.
**Config:** `results/config_ekalavya_iter5_probe.json`
**Seed:** best.pt from routing run (step 250, eval BPB=1.418)
**Kill:** step 250 eval > 1.418 = failed
**Early data:** Prior run crashed (computer shutdown). Relaunched 07:52 April 15. Awaiting step 10 data.
**Status:** Running (relaunched). Step 10+ pending.

### multi2_routed_3k [KILLED at step 760 — POSITIVE]
**Purpose:** Multi-teacher ROUTED KD: SmolLM2 (anchor) + Pythia (aux), confidence routing, covering ON.
**Mechanism:** Anchor-dominant confidence routing — Pythia contributes only where JSD>0.02 AND it's more confident than SmolLM2. LR -20% vs AM run, T=1.3, aggressive KD decay (final_mult=0.3), repr beta decays to 0.
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
**Purpose:** First-byte marginal KD with Codex audit fixes, single teacher SmolLM2-1.7B.
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
**What we learned:** Alpha=0.5 too aggressive. Need 5x lower alpha. Off-by-one alignment bug found by Codex audit.

### ekalavya_v2_2k [RUNNING]
**Purpose:** Corrected Ekalavya with all Codex audit fixes. Alpha=0.10, T=1.5, 4-bit teacher.
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
