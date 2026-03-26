# Scratchpad

Working space for half-finished thoughts, emerging ideas, and in-progress reasoning. Things move OUT of here into RESEARCH.md (when validated) or get deleted (when falsified).

**Rule:** Nothing here is a conclusion. Everything here is a question or a sketch.

---

## META-PROCESS ANALYSIS: What Are We Learning About How We Learn? (2026-03-26)

**Status: ACTIVE — update at every checkpoint, every experiment**

### Process Failures (things we'd do differently)

1. **Architecture search went 5 rounds too long.** Final spread between ALL variants was 0.07-0.18 BPT. Should have set a "search dimension value" threshold: if total spread across all tested variants is < 0.2 BPT, the dimension is exhausted. Stop searching it. We wasted ~4 rounds of compute.

2. **Built full RMFD system before validating basic KD signal.** Should have run a 500-step "does KD provide ANY signal?" micro-probe before implementing byte-span bridges, CKA losses, multi-phase training. The probe running now should have been the FIRST thing after Round 3.

3. **Fundamentals research produced understanding but not novelty.** All 5 derived mechanisms were reinventions (Codex confirmed). The UNIQUE thing — byte-span cross-tokenizer alignment — emerged from solving a PRACTICAL constraint, not from surveying abstract math. **Codex correction (§6.4.23):** Novelty was the wrong KPI. The REAL mistake was letting theory pull implementation order ahead of signal validation. Theory should INFORM design, but signal validation gates implementation.

### Process Wins (things to keep doing)

1. **Codex correctness audits before training** — caught 3 real bugs, saved potentially wasted GPU hours.
2. **Kill rules declared before experiments** — prevented post-hoc rationalization of hybrid variants.
3. **Fundamentals-first approach** — even though it didn't produce novel mechanisms, it gave us deep understanding of WHY routing/scheduling works. This understanding will guide future decisions.
4. **Competitive baseline tracking** — knowing SmolLM-135M scores gives us a real target.

### Open Meta-Questions

1. **Is the KD advantage a head-start or a limit change? ANSWERED: HEAD-START (for rep-KD).** Gap: -0.059 (500) → -0.042 → -0.023 → -0.036 → -0.144 (2500, WSD artifact) → **-0.008 (3000, noise)**. Rep-KD (CKA+semantic) provides transient acceleration only. Control catches up during WSD decay. Still open: does LOGIT KD change this? The surface ablation tests this directly.

2. **What determines the theoretical MAXIMUM KD benefit?** Rate-distortion theory says the teacher reduces effective source entropy. But HOW MUCH depends on teacher-student mismatch, cross-tokenizer alignment quality, and alpha tuning. We haven't explored alpha at all.

3. **Are we measuring the right thing?** BPT doesn't predict benchmarks well at low step counts. Should we switch to a benchmark-focused eval earlier? Or is BPT still the right signal during early training?

4. **Should we invest in DATA instead of METHODS? ANSWERED: NO (for current horizon).** Even at 120K steps we see only 8.6% of our 22.9B corpus (22 tokens/param ≈ Chinchilla optimal). The gap with competitors isn't data VARIETY — it's total training tokens seen (SmolLM-135M: 600B = 4444 tok/param). We'd need ~2.2M steps to match SmolLM's overtraining. KD is the lever to close this gap with fewer steps — exactly the manifesto thesis.

5. **What's the opportunity cost of monitoring?** We spend significant time polling training logs. Should we build better automated eval infrastructure so results are automatically logged and analyzed?

### Process Rules (emerging from this analysis)

- **Dimension exhaustion rule:** If spread across all tested variants in a dimension is < 0.2 BPT, stop searching that dimension.
- **Signal-first rule:** Validate basic signal with a 500-step micro-probe before committing to full implementation.
- **Meta-checkpoint rule:** Every 5K-step eval asks not just "did metrics improve?" but "is this the right metric? Is this the right approach? What would 2x the improvement?"
- **Opportunity cost rule:** Before starting any work, ask "what ELSE could we do with this time/compute?" and pick the highest-expected-value option.

---

## KD Training Dynamics Analysis: WSD × KD Interaction (2026-03-26)

**Status: COMPLETE — rep KD = head-start only, gap converged to -0.008 at step 3000**

### Observed Pattern: KD amplifies LR decay consolidation

| Step | Control BPT | KD BPT | Gap | LR Phase |
|------|-------------|--------|-----|----------|
| 500  | 4.9536 | 4.8946 | -0.059 | Stable (3e-4) |
| 1000 | 4.9194 | 4.8772 | -0.042 | Stable |
| 1500 | 4.8725 | 4.8493 | -0.023 | Stable |
| 2000 | 4.8250 | 4.7888 | -0.036 | Stable |
| 2500 | 4.8583 | 4.7142 | **-0.144** | Decay (2.52e-4) |
| 3000 | 4.5579 | 4.5500 | **-0.008** | Decay (1e-5) |

**Three-phase interpretation:**
1. **Steps 500-1500 (stable LR, gap narrowing):** Initial KD head-start erodes. Control catches up on NTP alone. Consistent with head-start hypothesis.
2. **Steps 1500-2400 (stable LR, gap stabilizing):** Gap reversal at step 2000 (-0.023 → -0.036). KD advantage stabilizes — no longer just head-start.
3. **Steps 2400-3000 (LR decay, gap explodes):** Control REGRESSES at step 2500 (4.8250 → 4.8583). KD continues improving (4.7888 → 4.7142). KD provides better consolidation signal during LR cooldown.

**Hypothesis: KD acts as an implicit regularizer during LR decay.** When LR drops, the model transitions from exploration to consolidation. Without KD, the model must consolidate from NTP signal alone — which is noisy (each batch is random). With KD, the teacher provides a smoother, more informative consolidation target. This is mathematically analogous to how KD with soft labels provides richer gradient information than hard labels (Hinton 2015).

**KD loss trend supports this:**
| Steps | Mean KD Loss | Interpretation |
|-------|-------------|----------------|
| 0-500 | 0.0317 | High — teacher signal novel |
| 500-1000 | 0.0235 | Drop — student absorbing |
| 1000-1500 | 0.0255 | Slight rise — new data regions |
| 1500-2000 | 0.0241 | Stable |
| 2000-2500 | 0.0197 | Still decreasing — not saturated! |
| 2500+ | 0.0149 (step 2650) | Lowest yet — active learning during decay |

**Key: KD loss is NOT saturating.** If KD were pure head-start, we'd expect KD loss to plateau (student fully caught up to teacher). Instead it's still decreasing at step 2650, suggesting the student has more to absorb.

**RESULT (step 3000):** KD BPT = **4.5500**, Control BPT = **4.5579**. Gap = **-0.0079** (essentially a tie).

**Prediction was WRONG.** The KD-amplified-decay hypothesis failed. Control recovered MORE during decay (4.8583→4.5579 = -0.300 drop) than KD (4.7142→4.5500 = -0.164 drop). The step 2500 "explosion" was a transient WSD artifact — the control simply started its decay-consolidation later.

**Revised trajectory interpretation:**
- Steps 500-1500: gap narrowing (-0.059 → -0.023) = head-start erosion
- Steps 1500-2500: gap unstable due to WSD schedule interaction
- Step 3000: gap closed to noise level (-0.008)

**Conclusion: Representation KD (CKA + semantic) provides a transient head-start that vanishes by 3000 steps with WSD schedule.** This is consistent with Codex's earlier assessment ("too early to call") being correct — it wasn't too early, it was the wrong signal.

**Critical implication for surface ablation:** Rep-only KD ≈ control at 3000 steps. If logit KD also converges to control, then KD needs more steps to show sustained benefit. If logit KD shows a PERSISTENT gap at 3000 steps, it provides qualitatively different value (distribution-level vs representation-level supervision). This is exactly what the 4-arm ablation is designed to test.

### Early Ablation Signal: Rep-Only α=1.0 HURTS (arm 2 step 500)

| Metric | Control (arm 1) | Rep-only α=1.0 (arm 2) | Delta |
|--------|-----------------|------------------------|-------|
| BPT | 5.0211 | 5.0591 | **+0.038 (WORSE)** |
| kurtosis_max | 3.1 | 3.2 | OK |
| max_act | 50.5 | 58.6 | 1.16x (borderline yellow) |

First probe with α_total=0.8 showed -0.059 (BETTER). α=1.0 is too aggressive for rep-only. The KD objectives (CKA + Gram) compete with NTP at this weight.

**Implication for logit KD:** Logit KD at α=1.0 should NOT show this penalty because it directly optimizes the prediction distribution (aligned with BPT metric). If arm 3 (logit-only, α=1.0) ALSO shows a penalty → KD weight itself is the issue. If arm 3 shows benefit → rep KD specifically is the problem at high weight.

**Implication for 15K:** Use lower rep KD weight. First probe's α=0.8 was better. For combined (rep+logit), keep total α≤0.8 with rep portion ≤0.3.

**Codex-mandated monitoring thresholds (§6.4.26):**
- **Kurtosis:** >2x control = yellow, >3x = red. Watch for kurtosis growing DURING WSD decay (bad sign).
- **Max_act:** >1.15x control = yellow.
- **3K persistence:** if logit KD gap >0.02 BPT at 3K, promising. If <0.01, another head-start.
- **15K proof gate:** >0.015 BPT at 15K + at least one lm-eval lift = persistent KD.
- **Contingency if all surfaces fail:** scale student to 166-200M (Codex says 90M 1:19 ratio is below KD comfort zone).

### ~~RESOLVED~~: Temperature/Top-K Conflict (see RESEARCH.md §6.4.28)

**RESOLVED by literature synthesis.** T=2.0 is correct for cross-tokenizer KD:
- DSKDv2, DWA-KD both use T=2.0 for cross-tokenizer distillation
- Design Space paper (ACL 2025): T∈{1.0, 1.5, 2.0} all similar for pre-training KD
- Minitron's T=1.0 finding was same-tokenizer at ~1:2 ratio — doesn't apply to our 1:19 cross-tokenizer setup
- K=64 is slightly above BiLD optimal (K=8-50) but acceptable

**No temperature sweep needed** unless logit KD shows zero benefit at T=2.0.

#### Theoretical Resolution: Capacity Ratio Modulates Optimal (T, K)

**Key insight: the optimal (temperature, top-K) settings depend on the student:teacher capacity ratio.**

**The argument:**
1. At **moderate ratios** (1:2 to 1:9, typical in literature): the student has enough capacity to absorb MOST of the teacher's output distribution. Higher T reveals more dark knowledge → helpful. Larger K (or no K) preserves more of the distribution → helpful. This is the regime Minitron measured: teacher ~8B, student ~4B, ratio ~1:2. Their conclusion (T=1.0, K→none) was for moderate ratios.

2. At **extreme ratios** (1:19, our case): the student CANNOT absorb the full teacher distribution — it lacks representational capacity. Two failure modes emerge:
   - **High T (2.0):** Flattens the teacher distribution, revealing dark knowledge on rank 30-64+ tokens. But the student doesn't have capacity to model these subtle distinctions → gradient noise, not signal. The KD loss pulls the student toward a flatter distribution than it can represent.
   - **Low T (1.0):** Keeps the teacher distribution peaked. The student sees fewer "dark knowledge" tokens but the signal is cleaner — focused on the top predictions the student CAN learn. Less noise, more transferable signal per gradient step.

3. Similarly for **top-K at extreme ratios:**
   - **K=64:** Focuses on the teacher's top 64 tokens per position. At 1:19 ratio, the student might only have capacity to model the top 10-20 plausible tokens anyway. K=64 is already generous.
   - **K=none (full vocab):** The teacher's opinion on rank 100+ tokens is pure noise for a 90M student. Including them dilutes the gradient with signal the student can't absorb.

**Prediction:** For 90M:1.7B (1:19):
- T=1.0 should outperform T=2.0 (cleaner signal for low-capacity student)
- K=64 should be fine or BETTER than K=none (focuses signal on learnable knowledge)
- **This REVERSES the Minitron recommendation** which was measured at ~1:2 ratio

**But Minitron's recommendation may still apply if we scale the student:**
- At 166M (1:8.8 ratio): in the "comfort zone" → T=1.0, K→128+ may be better
- At 200M (1:7.5 ratio): T and K closer to Minitron regime

**Testable prediction from current ablation:** If logit arms (T=2.0, K=64) show a PENALTY at 90M (like rep-only did), the most likely explanation is T=2.0 is too high for this ratio. Re-test with T=1.0 before concluding logit KD fails.

**Conversely:** If logit arms show benefit even at T=2.0/1:19 → the method is robust and T=1.0 would likely be even better (more signal for 15K).

---

## Theoretical Analysis: Why Logit KD Should Differ from Rep KD (2026-03-26)

**Status: HYPOTHESIS — awaiting ablation results for validation/falsification**

### The Saturation Argument

Rep KD (CKA + relational Gram) supervises STRUCTURAL similarity of internal representations. Two distinct failure modes:

1. **CKA saturates before functional convergence.** CKA measures kernel alignment between student and teacher hidden states. It is possible for CKA loss to be low (representations look structurally similar) while the student's output quality remains far from the teacher's. CKA convergence ≠ functional convergence. This explains the head-start pattern: CKA loss drops fast (0.032→0.015 in first 2500 steps), representations get structurally aligned, but the student still needs to learn HOW TO USE those representations through NTP.

2. **Relational Gram is a low-rank signal.** The Gram matrix captures pairwise similarities within a sequence. For 512 tokens, this is a 512×512 matrix — but the information content is bounded by the rank of the teacher's hidden states (typically much less than 512). The relational signal is relatively low-dimensional.

In contrast, logit KD supervises the student's OUTPUT DISTRIBUTION to match the teacher's predictions. This is a FUNCTIONAL supervision signal — it doesn't care about representation structure, only about prediction quality.

### Why Logit KD Should Be Harder to Saturate

1. **Output KL is a tighter bound.** KL(teacher || student) on output distributions directly measures the gap between student and teacher predictions. The student cannot "look aligned" while being functionally different — the loss explicitly measures functional distance.

2. **Dark knowledge signal is per-token and high-dimensional.** For each token, the teacher provides a full probability distribution over K=64 tokens (after top-k filtering). This encodes nuanced information: "token A is likely (p=0.3), token B is half as likely (p=0.15), tokens C-F are plausible alternatives (p=0.05 each)." The student gets 64 bits of gradient information per token position, per batch.

3. **The signal refreshes with new data.** Unlike CKA (which can saturate when representations are structurally aligned regardless of the specific batch), logit KD provides unique supervision for each new batch because the teacher's output distribution depends on the specific input sequence.

### Prediction: Logit KD Pattern

If logit KD is qualitatively different from rep KD, we should see:
- Initial gap similar to rep KD or possibly smaller (logit KD is noisier due to cross-tokenizer alignment)
- Gap WIDENS or STAYS CONSTANT during stable LR phase (unlike rep KD which narrowed)
- Gap persists through WSD decay (control doesn't catch up like it did with rep KD)

**Kill condition for logit KD:** If logit-only arm converges to within 0.02 BPT of control by step 3000, logit KD is ALSO head-start-only at this scale. Would require longer training to be useful.

---

## Rate-Distortion Theory of KD at Extreme Capacity Ratios (2026-03-26)

**Status: HYPOTHESIS — novel theoretical framework, awaiting ablation validation**

### The Core Insight: KD at 1:19 is Information SELECTION, Not Compression

Standard KD (ratios 1:2 to 1:9): student learns a compressed version of teacher. Has enough capacity to approximate most of the teacher's behavior. Temperature and top-K are tuning knobs for SIGNAL QUALITY.

Extreme KD (ratio 1:19, our case): student CANNOT compress the teacher — it must SIMPLIFY. The student's limited capacity forces it to SELECT which aspects of the teacher's knowledge to learn. This changes the optimization landscape fundamentally.

### Rate-Distortion Framing

The student at 90M params has a fixed "rate budget" R (in bits of representational capacity). The teacher provides knowledge across the input distribution. Different KD surfaces define different distortion metrics:

- **Rep KD distortion:** d_rep(x) = ||P·S_student(x) - S_teacher(x)||² — measures structural distance in hidden state space
- **Logit KD distortion:** d_logit(x) = KL(p_teacher(·|x) || p_student(·|x)) — measures prediction distribution distance

Rate-distortion theory says: given rate R, the optimal encoder minimizes E[d(x)] by allocating more bits to inputs where distortion is highest. At extreme ratios, R is so small that the student can only achieve low distortion on a SUBSET of the input distribution.

### What Each Surface Selects

**Rep KD at 1:19:** Student matches teacher on inputs where projected representations are easy to align. CKA and Gram losses are "structural" — they reward geometric similarity regardless of which specific tokens are being predicted. This biases toward **common patterns** (data manifold modes with low-rank structure).

**Logit KD at 1:19:** Student matches teacher on inputs where the teacher's top-K predictions are most concentrated. High-confidence teacher predictions (easy tokens) have more KL mass in top-K → stronger gradient signal. This biases toward **teacher-confident predictions** (tokens the teacher "knows well").

### Predictions

1. **Rep KD should plateau earlier** at extreme ratios because structural alignment saturates when the student exhausts its geometric capacity. CKA can be low while functional performance is still poor. This is exactly what we observed (head-start only, gap closed by step 3000).

2. **Logit KD should have a different learning curve shape:** Initial progress may be SLOWER (cross-tokenizer noise, high-dimensional KL), but the signal doesn't saturate because KL divergence on new batches always provides novel gradient information. The teacher's opinion on "what comes next" is input-dependent and refreshes with each batch.

3. **Top-K filtering should be BENEFICIAL at extreme ratios** (contrary to Minitron which tested moderate ratios). At 1:19, the student can only model the top ~10-20 plausible tokens per position anyway. Showing it the teacher's full distribution (K=none) adds noise from tokens the student can't represent. K=64 focuses the gradient on the learnable portion of the teacher's distribution. **UPDATE (§6.4.28 expanded):** Peng et al. ACL 2025 found K=50 optimal; our K=64 is acceptable. ⚠️ Sparse Logit Sampling (ACL 2025 Oral) proves naive top-K truncation gives biased gradient estimates — but Peng et al. K=50 still works well in practice, so bias is tolerable at these K values.

4. **Combined (rep+logit) at extreme ratios may INTERFERE.** Both surfaces compete for the same limited parameter budget. Rep KD pulls representations toward structural alignment; logit KD pulls outputs toward distribution matching. At moderate ratios these are complementary. At 1:19, they may be allocating the same bits to conflicting objectives → the orthogonality prediction from §info-geo becomes a genuine test.

5. **Temperature prediction:** ~~Low T (1.0) should outperform high T (2.0) at extreme ratios~~ **PARTIALLY CONTRADICTED by literature** (see §6.4.28). Cross-tokenizer consensus is T=2.0. The capacity-ratio × temperature interaction may still exist but is weaker than predicted — T∈{1.0, 1.5, 2.0} all similar per Design Space paper. The more important effect is probably cross-tokenizer alignment quality, not temperature.

### Connection to Manifesto

This framing directly serves "Intelligence = Geometry": the student doesn't need MORE parameters, it needs BETTER selection of what to learn from the teacher. The optimal KD surface at extreme ratios is the one that selects the highest-value knowledge for the student's capacity budget. This is a geometric problem (information geometry on the student's parameter manifold), not a scale problem.

### Falsification

- If arm 4 (rep+logit) is ADDITIVE rather than interfering → rate budget is not the binding constraint at 90M, and the extreme-ratio effects are weaker than predicted
- If logit KD also converges to control by 3K → the rate-distortion framing is correct but the "refreshing gradient" advantage isn't enough at this horizon
- If T=2.0 logit KD works well → capacity ratio doesn't modulate optimal temperature as strongly as predicted

### Cross-Tokenizer Signal Loss

Our logit KD uses only N_shared ≈ 14,822 tokens out of student's 16K vocabulary and teacher's 152K vocabulary. This is ~92.6% of student vocab but only ~9.7% of teacher vocab. We're only seeing the teacher's opinion on tokens the student knows — which is most of the student's world but a tiny fraction of the teacher's world.

**Implication:** The cross-tokenizer alignment may attenuate the logit KD signal. If the teacher's dark knowledge is concentrated on rare tokens (which are less likely to be in shared vocabulary), we lose high-value signal. This is testable: compare logit KD loss magnitude across positions — if some positions have very few valid shared tokens, the signal quality drops there.

### Information-Geometric Predictions for 4-Arm Ablation (TESTABLE)

Derived from §7.2 (Information Geometry) in RESEARCH.md:

**Prediction 1: Orthogonality Test.** Rep KD constrains manifold SHAPE; logit KD constrains OUTPUT DISTRIBUTION. If these are "orthogonal" in the information-geometric sense (Fisher-Rao metric), the combined arm should show additive benefits:
- Let Δ_rep = BPT_control - BPT_rep_only
- Let Δ_logit = BPT_control - BPT_logit_only
- Let Δ_both = BPT_control - BPT_rep_plus_logit
- **Orthogonal:** Δ_both ≈ Δ_rep + Δ_logit (additive)
- **Redundant:** Δ_both ≈ max(Δ_rep, Δ_logit) (dominated)
- **Interfering:** Δ_both < max(Δ_rep, Δ_logit) (destructive)

**Prediction 2: WSD Consolidation.** Logit KD should show MORE improvement during WSD decay (steps 2400-3000) than control, because it provides per-token consolidation signal richer than NTP alone. Rep KD showed LESS improvement during decay (control: -0.300, rep: -0.164). If logit KD shows >-0.300 during decay → it provides a consolidation advantage rep KD doesn't.

**Prediction 3: Kurtosis Divergence.** Rep KD caused 3.67x kurtosis at 3K. Logit KD operates on output distributions, not internal representations — should produce LESS kurtosis divergence than rep KD. If logit-only kurtosis < 2x control → logit KD is a safer signal.

---

## Student Scaling Contingency (if 90M fails KD) (2026-03-26)

**Status: CONTINGENCY PLAN — only activate if logit KD fails at 90M**

Codex §6.4.26/§6.4.27: 90M:1.7B = 1:19, below KD comfort zone (1:1.5 to 1:9). **Peng et al. ACL 2025 confirms: "effective when student reaches ~10% of teacher size" — our 5.3% is below this threshold.** If logit KD doesn't persist at 90M, scale student before giving up on KD.

| Config | Width | Depth | Heads | Params | Ratio | VRAM(est) |
|--------|-------|-------|-------|--------|-------|-----------|
| Current | 512 | 24 | 8 | 90M | 1:19 | 7.3GB |
| **Scale-166M** | **768** | **24** | **12** | **196M** | **1:8.8** | **8.4GB** |
| **Scale-200M** | **832** | **24** | **13** | **229M** | **1:7.5** | **8.7GB** |
| Scale-250M | 896 | 24 | 14 | 265M | 1:6.5 | 9.0GB |

**Recommendation:** 166M (d=768) is the sweet spot — brings ratio to 1:8.8 (within comfort zone), same depth so teacher layer mapping is unchanged, only ~1GB more VRAM. The 200M option (d=832) has 13 heads (unusual) but head_dim=64 works. All fit easily in 24GB.

**Decision rule:** If logit KD gap < 0.01 BPT at 15K → scale student to 166M and re-run 15K gate.

### Arm-by-Arm Numerical Predictions (Added 2026-03-26, pre-arms-3/4 data)

Based on control trajectory (this ablation): BPT 5.02→4.91→4.83→4.83→4.68→4.50

**Arm 2 (rep_only, α=1.0): ~~CONFIRMED WORSE~~ → SURPRISING CROSSOVER.**
- Step 500: predicted penalty from α too high. Actual: +0.038 BPT worse. ✓ CONFIRMED.
- Step 1000: Recovery to tied (-0.002). ✓ Predicted.
- **Step 1500: -0.094 BPT BETTER than control (4.736 vs 4.830). ✗ PREDICTION FAILED.**
  - We predicted convergence to ~control ±0.02. Actual gap is 4.7x our prediction range.
  - Kurtosis: 4.3 (rep) vs 4.9 (control) — healthier activation distribution.
  - Max activation: 62.9 (rep) vs 79.9 (control) — less extreme activations.
  - **3-phase pattern: penalty → recovery → advantage.** The α=1.0 initial disruption may have forced the student to reorganize its representation space, ultimately finding a BETTER configuration.
  - **Critical question: does advantage survive WSD decay?** Control improved -0.185 during decay. If rep KD's advantage persists through decay, this changes everything.
- ~~Final 3K: predict convergence to ~control ±0.02 BPT~~ **REVISED: Predict final gap depends on decay behavior. If rep consolidates better during decay, could see -0.05 to -0.10 advantage.**

**Arm 3 (logit_only, α=1.0): THE CRITICAL TEST.**
- Step 500: predict BPT ≤ control (5.02 or below). Logit KD is prediction-aligned, should NOT show penalty.
- Step 1000-2000: predict gap HOLDS or WIDENS (unlike rep KD which narrowed).
- Step 3000: predict BPT ≤ 4.47 (>0.02 better than control's 4.50) = "promising" per Codex threshold.
- Kurtosis: predict < 2x control (< 10.0) — logit KD operates on outputs, not activations.
- **If step 500 BPT > control → T=2.0 or K=64 settings are problematic. Do NOT conclude logit KD fails.**

**Arm 4 (rep+logit, total α=1.0: state=0.3125, sem=0.1875, logit=0.5):**
- Rate-distortion theory predicts INTERFERENCE at extreme ratios (rep and logit compete for bits).
- Predict: BPT between arm 2 and arm 3. If BPT < arm 3 → orthogonality (info-geo prediction wins).
- If BPT > arm 2 AND > arm 3 → destructive interference, confirms extreme-ratio prediction.

**Decision matrix (at step 3000):**
| Arm 3 vs Control | Arm 4 vs Arm 3 | Interpretation | 15K Action |
|-------------------|----------------|----------------|------------|
| ≤ -0.02 (better) | additive | Logit KD persistent, surfaces orthogonal | Run rep+logit at 15K |
| ≤ -0.02 (better) | worse than arm 3 | Logit KD persistent, surfaces interfere | Run logit-only at 15K |
| -0.01 to -0.02 | any | Weak signal, possibly head-start | Run logit-only at 15K, monitor carefully |
| > -0.01 | any | Logit KD also transient at 90M | Re-test T=1.0/K=256, or scale student |

**NEW: Extreme-ratio mitigations for Codex #274 review (from §6.4.29 research):**
Regardless of which arm wins, the 15K gate should consider:
1. **Rising τ schedule** (τ=1.5→4.0): POCL shows "critical" at 15x ratio. Trivial to implement
2. **Reverse KL / AMiD (α=-3 to -5)**: Mode-seeking instead of mode-covering. Tested at exact 15x ratio
3. **MiniPLM data reweighting**: 2.2x compute savings, pilot already done (task #237)
4. **Qwen3-0.6B as alternative primary teacher**: 1:7 ratio (within comfort zone) vs 1:19
These are QUESTIONS FOR CODEX, not decisions. Present as options in the evidence document.

---

## Meta-Learning KD: Category-Theoretic Disagreement Analysis (2026-03-26)

**Status: EVALUATED BY CODEX — category theory overkill, MVP refined below**

**Codex Verdict (Architecture Theorist, 2026-03-26):** Category-theoretic framing acceptable as research language but NOT operational math. The 5-category decomposition is unidentifiable with 3 teachers. "Inverted power law" renamed to "transient adaptive acceleration" — not proven. Full review in RESEARCH.md §6.4.21.

### Core Thesis
Multi-teacher KD should be a SELF-IMPROVING process with accelerating returns, not diminishing returns. The manifesto says Intelligence = Geometry, not Scale — this applies to the LEARNING PROCESS itself, not just the architecture.

### The Three-Layer Framework

**Layer 1: Category-Theoretic Disagreement Grouping**

Each teacher T_i defines a functor F_i: Data → Predictions. The key objects are:
- **Agreement kernel** K(T_i, T_j) = {x : F_i(x) ≈ F_j(x)} — where two teachers agree
- **Disagreement manifold** D(T_i, T_j) = Data \ K — where they differ
- **Natural transformations** between functors capture HOW they disagree (probability mass shift, different top-1, different confidence level)

Practical decomposition of teacher outputs into categories:
1. **Consensus knowledge** — ALL teachers agree. This is "free" knowledge, easy to absorb.
2. **Architecture-dependent** — Transformers agree, SSMs disagree (or vice versa). Reveals structural bias.
3. **Capacity-dependent** — Large teachers agree, small teachers disagree. Knowledge gated by model capacity.
4. **Family-dependent** — Teachers from same family agree, cross-family disagree. Corporate training bias.
5. **Universally uncertain** — ALL teachers disagree with each other. Genuinely ambiguous data.

**Hypothesis**: Category 2 (architecture-dependent) is the HIGHEST VALUE for KD, because it reveals knowledge that can only be accessed via that architecture family. Distilling this into our student gives us cross-architecture capabilities no single model has.

**Layer 2: Inverted Power Law (Learning to Learn)**

Standard training: diminishing returns. More data → smaller marginal improvement.
Meta-learning KD: each training round produces:
- A better model (standard)
- A better understanding of WHAT the model can't learn (diagnostic)
- A better strategy for WHAT to learn next (curriculum)
- A better weighting for WHICH teacher to trust (routing)

This creates accelerating returns: the more you learn, the more precisely you can identify what to learn next → faster convergence.

**Mechanism**: After each eval checkpoint:
1. Run disagreement analysis across all teachers
2. Classify student failures by category (consensus vs arch-dependent vs capacity-gated)
3. Update teacher weights: upweight teachers that provide knowledge the student currently lacks
4. Update data curriculum: prioritize samples in high-disagreement regions
5. Optionally update architecture: if a knowledge type is consistently unlearnable, the architecture itself may need modification

**Layer 3: Knowledge State Map (Diagnostic-First Learning)**

Maintain a structured map of the student's knowledge state:
- Per-domain: syntax, factual recall, reasoning, world knowledge, etc.
- Per-category: which of the 5 disagreement categories is the student strongest/weakest in
- Per-teacher: which teacher provides the most value for the student's current state

This map is the BETTER INTERNAL METRIC the user asked for. Instead of BPT (which doesn't predict benchmarks), we track:
- "Student matches teacher consensus on X% of factual recall samples"
- "Student fails on Y% of capacity-gated reasoning samples"
- "Student can now absorb architecture-specific knowledge from SSM teachers"

These predict real benchmark performance because they measure actual capabilities, not just perplexity.

### Connection to Other Ideas
- **Ekalavya Protocol**: This IS the evolved Ekalavya — not just absorbing from teachers, but learning HOW to absorb
- **Rework internal metrics**: The knowledge state map IS the better internal metric
- **Decisive victory path**: If we can demonstrate accelerating returns from KD (inverted power law), that's PROVABLE data efficiency with scaling evidence

### Open Questions for Codex
1. How to implement disagreement analysis cheaply? (Can't afford full inference on all teachers every checkpoint)
2. What's the right granularity? Per-token? Per-sequence? Per-domain?
3. Is category theory overkill? Simpler clustering (KL divergence between teacher pairs) might suffice
4. How to validate the inverted power law claim? Need a metric that shows improving learning efficiency over time
5. Can we use this framework to PREDICT which new teachers would be most valuable before downloading them?

### Falsification Criteria
- If disagreement analysis shows no structure (random), the category theory angle is useless
- If teacher weighting by disagreement doesn't improve over uniform weighting, the routing is useless
- If learning efficiency doesn't accelerate (flat or diminishing curve), the meta-learning claim is false

---

## REFINED MVP: Routed KD with 4-Bucket Audit (2026-03-26)

**Status: DESIGN — pending probe results before implementation**

Based on Codex Architecture Theorist review. Stripped of category theory overhead, focused on operational disagreement geometry.

### Per-Sample Audit Metrics

**State loss per sample per teacher:**
```
l_state_i(x) = ||G(P_i @ S_student(x)) - G(S_teacher_i(x))||^2_F / M^2
```
where G(·) is row-normalized span Gram matrix over M=16 byte spans.

**Semantic loss per sample per teacher:**
```
l_sem_i(x) = 1 - cos(P_i @ z_student(x), z_teacher_i(x))
```
where z = mean-pooled hidden state.

Both return (B,) vectors, not scalars. This enables per-sample routing.

### 4-Bucket Classification (every 500 steps, on 256-window held-out audit)

Given teachers Q (Qwen) and L (LFM):
- **Consensus**: d_QL < q30 AND mean decoder entropy < q30 → Use equal weights
- **Specialist-Q**: d_QL > q70, student gap to Q > q70, Q has lower entropy → Route to Q only
- **Specialist-L**: d_QL > q70, student gap to L > q70, L has lower entropy → Route to L only
- **Uncertain**: Both decoders high entropy → Semantic-only (EmbeddingGemma)
- (Bridge-noisy: <12/16 spans occupied → Semantic-only — detect bad tokenization)

Where d_QL = pairwise divergence between Q and L span Gram matrices on each sample.

### Multi-Depth Teacher States

Teacher hidden states at relative depths 1/3, 2/3, 1.0 matched to student exits 7, 15, 23.
Currently TeacherAdapter returns only last hidden state — need multi-depth extraction.

### Surface Split

Route Qwen vs LFM on state surface only. Keep EmbeddingGemma as fixed low-weight semantic anchor:
```
L = L_CE + 0.4 * (w_Q * l_Q_state + w_L * l_L_state) + 0.1 * l_E_sem
```
where w_Q, w_L are bucket-derived weights (not learned — discrete routing).

### Kill Rule
- Routed KD must beat static multi_3family at equal teacher FLOPs
- Must lower specialist-bucket deficit
- Must not worsen stability (kurtosis/max_act) by more than ~10%
- If fails → kill routing, keep simple uniform KD + MiniPLM

### Implementation Order (AFTER probe shows signal)
1. Per-sample loss functions (modify compute_state_kd_loss, compute_semantic_kd_loss)
2. Held-out audit set creation (256 windows from validation)
3. 4-bucket classification function
4. Bucket-aware loss weighting in train_kd_phased()
5. Multi-depth teacher state extraction
6. Probe: routed vs uniform at equal FLOPs

### Questions for Fundamentals Research — ANSWERED (2026-03-26)
- Optimal Transport: Is the Wasserstein barycenter the right consensus average? → YES, Codex P3. Only if consensus bucket is large enough and simple averaging measurably lossy. +2-10ms via Sinkhorn on 16×16 span matrices.
- Information Geometry: Should bucket thresholds use Fisher-Rao? → NO, KILLED. Fisher routing needs +15-40 TFLOP/step. Not feasible at our scale. Stick with CKA/cosine.
- Ensemble Theory: Does ambiguity decomposition predict when routing helps? → YES, this is ambiguity-aware scheduling (Codex P1). Diversity × reliability gates routing decisions.

### Codex-Approved Mechanism Stack (2026-03-26)

**Source:** Codex Architecture Theorist evaluation of 5 fundamentals-derived mechanisms. See RESEARCH.md §7.6.

```
LAYER 0: 4-Bucket Audit (existing design, every 500 steps)
  │
LAYER 1: Ambiguity-Aware Scheduling [PRIORITY 1, ~free]
  │  Track: pairwise teacher disagreement + rolling bucket win-rate
  │  Gate:  high-diversity + high-reliability → specialist routing
  │         high-diversity + low-reliability → consensus-only or skip KD
  │         low-diversity → single best teacher
  │
LAYER 2: GW Routing [PRIORITY 2, +5-80ms/step]
  │  Only for specialist state-surface decisions
  │  16×16 span-distance matrices, 10-20 entropic GW iterations
  │  Stop-grad routing, existing CKA losses unchanged
  │
LAYER 3: WB Consensus [PRIORITY 3, +2-10ms/step]
  │  Only inside consensus bucket
  │  3-teacher Sinkhorn barycenter over 16 span masses
  │  Replace simple averaging with geometry-respecting average
  │
KILLED: Info-geometric projection routing (+15-40 TFLOP — not feasible)
DEFERRED: Multi-marginal OT loss (ablation only, after P1-P3 work)
```

**Implementation order:** After probe confirms signal, implement P1 → measure → P2 → measure → P3 if needed.

**The novelty is the SYSTEM, not the components:** byte-span cross-tokenizer alignment + multi-architecture teachers + surface-specific routing + disagreement-driven adaptive scheduling = a combination that doesn't exist in any published system.

---

## First KD Probe Design (DRAFT — pending Codex R8 approval)

**Goal:** Validate that KD provides measurable improvement over pure CE training at 5K steps.

### Setup
- **Student:** 24×512 transformer, ~93M params (same as probe architecture)
- **Teacher:** Qwen3-0.6B (~600M params, 1024 dim, 28 layers, GQA)
  - Loaded in FP16, inference only (no gradients): ~1.2GB VRAM
  - Student training: ~8GB VRAM → combined fit easily on 24GB
- **Baseline:** Same student, pure CE loss, same data
- **Duration:** 5K steps (matching existing probes for comparison)

### Cross-Tokenizer Challenge
- Student tokenizer: 16K custom BPE
- Teacher tokenizer: Qwen3 152K BPE
- Same input text → different tokenization → different sequence lengths

**Simplest approach (Level 1): Sequence-level representation matching**
- Input: raw text string
- Student tokenizes with 16K tokenizer → forward pass → get final hidden state, mean-pool over sequence → h_student (512-dim)
- Teacher tokenizes with Qwen3 tokenizer → forward pass → get final hidden state, mean-pool over sequence → h_teacher (1024-dim)
- Projection: Linear(1024, 512) learned projection maps teacher dim to student dim
- Loss: L_kd = CosineEmbeddingLoss(projected_teacher, h_student) or MSE
- Total: L = L_ce + alpha * L_kd (alpha=0.5 initially)
- **Pro:** No position alignment needed. Dead simple.
- **Con:** Loses per-token signal. Only captures sequence-level semantics.

**Better approach (Level 2): Token-level alignment via character offsets**
- For each student token, find which teacher tokens cover the same characters
- Average teacher hidden states for aligned positions
- Per-position loss: L_kd_pos = MSE(projected_teacher_aligned[t], h_student[t])
- **Pro:** Per-token signal. Richer supervision.
- **Con:** Character alignment code needed. Multiple teacher tokens per student token common.

**Best approach (Level 3): Logit-level via shared vocabulary projection**
- Build mapping: student_vocab → character sequences → teacher_vocab
- For each student token ID, compute weighted sum of teacher logits for corresponding teacher tokens
- KL divergence on aligned probability distributions
- **Pro:** Richest signal. Standard KD.
- **Con:** Alignment matrix complex. May need DSKD-style learned projection.

**Recommendation:** Start with Level 1, prove KD helps at all, then upgrade to Level 2/3.

### Training Loop Changes (in dense_baseline.py)
```
# In training loop, after computing CE loss:
# 1. Get student hidden states
student_out = model(inp, return_hidden=True)
logits = student_out['logits']
h_student = student_out['hidden'].mean(dim=1)  # (B, D_student)

# 2. Get teacher hidden states (no grad)
with torch.no_grad():
    teacher_tokens = teacher_tokenizer(raw_text, ...)
    teacher_out = teacher_model(teacher_tokens)
    h_teacher = teacher_out.last_hidden_state.mean(dim=1)  # (B, D_teacher)

# 3. Project and compute KD loss
h_teacher_proj = projection(h_teacher)  # (B, D_student)
kd_loss = F.mse_loss(h_student, h_teacher_proj.detach())

# 4. Combined loss
loss = ce_loss + alpha * kd_loss
```

### Data Pipeline Modification
- Current: ShardedDataset returns (input_ids, target_ids) — pre-tokenized
- Needed: Also return RAW TEXT for teacher tokenization
- Modification: data loader stores shard as tokens but also reconstructs text for teacher
- **OR:** Pre-tokenize shards for each teacher (stored alongside student shards)
- **Simplest:** Decode student tokens back to text → re-encode with teacher tokenizer (lossy but fast)

### Kill Rule for KD Probe
- KD probe passes if: BPT with KD >= BPT without KD + 0.1 at 5K steps
- Also check: generation quality (greedy decode sample), kurtosis/max_act stability
- If pass: proceed to multi-teacher, Level 2 alignment
- If fail: investigate why (wrong teacher? wrong loss? wrong alpha?)

### lm-eval Correlation Data (2026-03-26)
Both architectures at 5K steps produce identical near-random benchmarks (~33.5% ARC-Easy, ~17% ARC-Challenge). This confirms: training duration + KD >> architecture choice. The bottleneck is KNOWLEDGE, not architecture.

---

## Pre-Round-1 Design Space Analysis (2026-03-26)

### The Core Problem
SmolLM2-135M trains on 2T tokens. We have ~23B. That's an 87x data disadvantage.
Pythia-160M trains on 300B tokens. Still 13x more than us.
To compete, we need either: (a) dramatically better data efficiency, or (b) knowledge absorption from existing models, or (c) both.

### Key Insight: The Data Efficiency Stack
Multiple techniques compound. Each addresses a different angle:

1. **Offline KD from teacher models** — MiniPLM shows 2.2-2.4x data efficiency. This is the single biggest lever.
   - Question: which teachers? How many? What representations to steal?
   - The OFFLINE approach is key — generate soft targets once, store to disk, train student against them
   - This means we can use models too big to run concurrently (just process data in advance)

2. **Multi-Token Prediction (TOP variant)** — Proven at 340M scale. ~20-50% more learning per example.
   - TOP is a learning-to-rank loss, not exact token prediction → works at small scale
   - Only needs one extra unembedding layer (~0.6% overhead)
   - Can combine with curriculum scheduling for more benefit

3. **N-gram memory** — Offloads pattern memorization to CPU table, frees neural capacity for reasoning
   - At our scale, a huge fraction of capacity is wasted memorizing "the → cat", "in → the", etc.
   - A 1M-entry table in CPU RAM costs ~48MB but could free 10-20% of neural capacity
   - The GATING is the key mechanism — table provides candidates, model decides relevance

4. **Structural priors** — Hyperbolic geometry, sheaf structure, etc.
   - HELM shows 4% gains from hyperbolic geometry. That's free intelligence.
   - Mixed-curvature (product manifolds) could be even better
   - Question: is this additive with the other techniques? Probably yes (orthogonal mechanisms)

5. **Architecture efficiency** — Hyena Edge shows gated convolutions beat attention
   - O(N log N) vs O(N²) — but at seq_len=512, this barely matters
   - The real win: gated convolutions may learn more efficiently from fewer tokens
   - Hybrid: keep attention for long-range, use convolutions for local patterns

### The Compound Effect
If these stack multiplicatively:
- 2x from offline KD
- 1.3x from TOP
- 1.2x from n-gram memory
- 1.04x from hyperbolic geometry
- 1.1x from architecture efficiency
= 2x * 1.3 * 1.2 * 1.04 * 1.1 ≈ 3.56x effective data

23B tokens * 3.56x = ~82B effective tokens. That closes the gap with Pythia-160M (300B tokens) significantly, though still behind SmolLM2-135M (2T tokens). But SmolLM2 uses a standard transformer — if our architecture extracts more per token inherently, the gap narrows further.

### Open Questions for Codex
1. What's the optimal model size for 24GB VRAM training? (200M? 400M? Depends on batch size)
2. Should we use shared-weight looping (LoopFormer-style) or unshared layers?
3. How to implement offline KD practically? Pre-compute soft targets from multiple teachers?
4. Is hyperbolic attention practical at our scale? (exp/log maps add overhead)
5. What's the right hybrid mix? (X% attention + Y% convolution + Z% SSM?)

### Risky Ideas Worth Testing
- **LoopFormer + early exit**: Shared blocks with time-conditioning, tokens exit at different loop iterations
- **Hyperbolic Engram**: N-gram memory in hyperbolic space (hierarchical lookup)
- **Cross-architecture distillation**: Steal from Mamba (SSM) AND Pythia (transformer) simultaneously
- **DyT everywhere**: Replace ALL normalization with DyT, design for quantization from step 1

---

## Cached Teacher Models (already downloaded on this machine!)

### Small LMs (can coexist with student during training):
- Pythia: 70M, 160M, 410M, 1B, 1.4B, 2.8B (+ deduped variants)
- SmolLM2: 135M, 360M, 1.7B
- GPT-2: 124M, 355M, 774M, 1.5B
- GPT-Neo: 125M, 1.3B, 2.7B
- Mamba: 130M, 370M, 790M, 1.4B (+ Mamba2 variants: 130M-2.7B)
- RWKV: 169M, 430M, 1.5B (RWKV4), 1.6B-14B (RWKV6), 191M-2.9B (RWKV7)
- OPT: 125M, 350M, 1.3B, 2.7B
- Cerebras-GPT: 111M, 256M, 590M, 1.3B
- TinyLlama: 1.1B
- StableLM: 1.6B, 3B
- Granite-4.0: micro, tiny, 350M, 1B (+ hybrid variants!)
- Falcon-H1: 0.5B, 1.5B, 3B (SSM-attention hybrid!)
- Zamba2: 1.2B, 2.7B (Mamba-attention hybrid)
- LFM2/2.5: 1.2B, 2.6B (Liquid AI — gated convolution hybrid!)
- Hymba: 1.5B (NVIDIA hybrid)

### Encoder models (rich representations, very small):
- BERT: base (110M), large (340M), + many variants
- RoBERTa: base, large
- DeBERTa: base, v3-base, v3-small
- DistilBERT: 66M
- Sentence transformers: all-MiniLM-L6-v2, all-mpnet-base-v2

### Embedding models:
- BGE: small, base, large, m3
- E5: small, base, large
- GTE: Qwen2-1.5B
- EmbeddingGemma: 300M
- Nomic-embed
- Stella: 1.5B

### Architecture diversity we can steal from:
- **Transformers**: Pythia, GPT-2, Qwen, Gemma, Llama, Phi
- **SSMs/Mamba**: Mamba 1/2, Falcon-Mamba
- **RWKV** (linear attention): RWKV4-7
- **Hybrids**: Falcon-H1, Zamba2, Granite-4.0-h, Hymba, LFM2, Jamba
- **Gated convolutions**: StripedHyena, LFM2
- **Encoders**: BERT, RoBERTa, DeBERTa
- **Diffusion LM**: DiffuGPT

This is a GOLDMINE for multi-source learning. We can generate offline soft targets from dozens of models.

---

## VRAM Budget Analysis (2026-03-26)

### Model Size vs Training Feasibility on RTX 5090 (24GB)

| Config | Params | Model(BF16) | AdamW | Grads | Act (BS=32) | TOTAL | MaxBS |
|--------|--------|-------------|-------|-------|-------------|-------|-------|
| 100M | 40.6M | 0.08GB | 0.32GB | 0.08GB | 5.13GB | 5.62GB | 133 |
| 135M (SmolLM2-class) | 82.2M | 0.16GB | 0.66GB | 0.16GB | 7.70GB | 8.69GB | 86 |
| 160M (Pythia-class) | 137.9M | 0.28GB | 1.10GB | 0.28GB | 10.27GB | 11.92GB | 63 |
| 200M [ckpt] | 175.6M | 0.35GB | 1.40GB | 0.35GB | 1.61GB | 3.72GB | 393 |
| 350M [ckpt] | 368.4M | 0.74GB | 2.95GB | 0.74GB | 2.68GB | 7.11GB | 208 |
| 400M [ckpt] | 435.5M | 0.87GB | 3.48GB | 0.87GB | 3.22GB | 8.45GB | 165 |

[ckpt] = gradient checkpointing. SwiGLU FFN, RoPE, RMSNorm assumed. seq_len=512.

### Chinchilla Analysis (tokens = 20x params)

| Size | Chinchilla-optimal | Our 22.9B tokens | Ratio |
|------|-------------------|-----------------|-------|
| 100M | 2.0B | 22.9B | 11.4x OVER |
| 135M | 2.7B | 22.9B | 8.5x OVER |
| 200M | 4.0B | 22.9B | 5.7x OVER |
| 350M | 7.0B | 22.9B | 3.3x OVER |
| 1.1B | 22.0B | 22.9B | ~1.0x OPTIMAL |

**Insight**: Chinchilla-optimal for our data budget = ~1.1B params. But we're not optimizing for Chinchilla — we're optimizing for PERFORMANCE at inference. Smaller model + KD = better than larger model from scratch at equivalent data. Over-training makes models more robust.

### KD VRAM Overhead

- **Offline KD**: ZERO GPU overhead during training. Storage: top-128 logits = ~1KB/token = ~23TB for full corpus (too much). Solution: stream soft targets, or use top-32 (~256B/token = ~5.9TB), or use feature-level distillation.
- **Online KD**: Teacher in inference mode (no grads)
  - Pythia-70M: 0.14GB | Pythia-160M: 0.32GB | SmolLM2-135M: 0.27GB
  - Pythia-410M: 0.82GB | SmolLM2-360M: 0.72GB | Mamba-790M: 1.6GB
  - **Conclusion**: Online KD feasible for teachers up to ~1B alongside 200M student

### Training Speed Estimates

| Size | Tokens/sec | Tokens/day | Full epoch (22.9B) |
|------|-----------|-----------|-------------------|
| 100M | ~80K | 6.9B | 3.3 days |
| 160M | ~55K | 4.8B | 4.8 days |
| 200M | ~45K | 3.9B | 5.9 days |
| 350M | ~25K | 2.2B | 10.6 days |

### Sweet Spot Analysis

**150-250M params** appears optimal:
- Over-trained on our data (5-8x Chinchilla) = robust
- Fits easily on 24GB with room for online KD teachers
- BS=32+ feasible with gradient checkpointing
- Full epoch in ~5-6 days = can do 4+ epochs in a month
- Room for auxiliary losses (TOP, KD) without VRAM pressure
- Small enough for rapid iteration, large enough for meaningful benchmarks

**Question for Codex**: Should we go 200M (faster iteration) or 350M (more capacity, slower)?

---

## Offline KD Feasibility Analysis (2026-03-26)

### Disk Budget
- 3,216 GB free on C:
- Training data occupies ~634GB (246 shards)

### Storage per approach (full 22.9B token corpus)
| Method | Storage | Feasible? |
|--------|---------|-----------|
| Full logits (FP16) | 733 TB | NO |
| Top-16 logits | 1.47 TB | Barely (45% of free space) |
| Top-32 logits | 2.93 TB | NO |
| Hidden states (d=768) | 35 TB | NO |

### Practical Strategy: ONLINE KD (Hybrid)
Full-corpus offline KD is impractical. **Online KD with co-resident teachers is the way.**

- **3-4 small teachers loaded in FP16** (total ~2-4GB VRAM)
- Teacher forward pass per batch = ~30-50% compute overhead
- **Zero disk overhead**, flexible (swap teachers anytime)
- Complement with MiniPLM-style data reweighting (free)

**Recommended teacher ensemble:**
1. Qwen3-0.6B (~1.2GB) - best quality sub-1B, 36T tokens
2. Mamba-370M (~0.74GB) - SSM architecture diversity
3. SmolLM2-135M (~0.27GB) - cheap reference, 2T tokens
4. Pythia-160M (~0.32GB) - deduped Pile, interpretable
Total: ~2.5GB VRAM for 4 diverse teachers

**Alternative for large teachers** (1B+): Process top-16 logits for first 2-3B tokens (~130-200GB), use online for the rest.

---

