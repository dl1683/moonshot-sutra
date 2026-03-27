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
- **Crude heuristic rule (from Devansh, 2026-03-26):** When a system is implicitly doing something, making it explicit — even crudely — yields orders-of-magnitude more improvement than leaving it implicit. Ask: "what is our system IMPLICITLY doing that we could make EXPLICIT?" The explicit mechanism doesn't need to be optimal; it needs to exist so it becomes a tunable lever. Examples: flat compute allocation → crude early exit; flat KD alpha → crude confidence weighting; unbounded search → crude bounds.

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

### New Literature (2025-2026, found during arm 4 monitoring)

**Cross-Tokenizer KD advances:**
- **Universal Cross-Tokenizer Distillation via Approximate Likelihood Matching** (NeurIPS 2025, arxiv 2503.20083): Principled cross-tokenizer distillation without shared-vocab mapping. Tested at Qwen 7B → Gemma 2B (~3.5:1 ratio). Not tested at extreme ratios.
- **Cross-Tokenizer Likelihood Scoring** (arxiv 2512.14954): Exploits BPE recursive structure for exact cross-tokenizer KL divergence. +4% on evaluated tasks, 12% memory reduction for Qwen2.5-1.5B. Only moderate ratios tested.
- **CTPD: Cross-Tokenizer Preference Distillation** (Jan 2026, arxiv 2601.11865): First cross-tokenizer preference distillation. Qwen 7B → Llama 1B (1:7 ratio).

**Alpha scheduling for KD pretraining:**
- **Pre-training Distillation for LLMs** (ACL 2025 Long, aclanthology 2025.acl-long.181): Uses warmup-stable-decay alpha schedule for KD loss weight. Confirms varying α over training outperforms constant α. No extreme ratios tested.

**Key gap confirmed:** No published work combines (1) inverted-U α, (2) rising τ, (3) confidence gating, at (4) extreme ratios (>1:10) with (5) cross-tokenizer alignment. Our 15K gate is genuinely novel in this combination. The ACL 2025 paper validates the general principle that α scheduling helps, but our specific inverted-U shape with WSD-aligned taper and confidence gating is new.

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

### NEW: Basin Compatibility Theory (from arm 2 step 2500 collapse, 2026-03-26)

**Why did rep KD collapse during WSD decay?** Control improved -0.144 BPT during early decay (steps 2000→2500); rep-only improved only -0.010. The 15x gap needs explaining.

**Basin compatibility hypothesis:** During stable LR (exploration), the optimizer moves through loss landscape seeking good regions. Rep KD shifts WHERE the student explores — toward the teacher's representational geometry. This finds a different basin than NTP-only. During WSD decay (consolidation), the optimizer must settle into a local minimum. The key: **rep KD's basin is optimized for CKA+Gram similarity, not for NTP loss.** When LR drops and the model consolidates, the NTP loss surface is what determines the quality of the final minimum. Rep KD's basin may be geometrically aligned with the teacher but SHALLOWER on the NTP surface.

**Evidence from step 2500:**
- Kurtosis spiked from 3.7→5.6 during decay (rep-only). Higher kurtosis = sharper activations = less stable under perturbation.
- Control's kurtosis: 5.4→5.0 during the same interval. Stabilizing, not destabilizing.
- The rep KD basin appears LESS stable under LR changes, not more.

**Implication for logit KD:** This is the key differentiator. Logit KD's loss is KL(teacher_dist || student_dist) — this is ALIGNED with the NTP objective (both optimize prediction distributions). The logit KD basin should be NTP-compatible: settling into this basin during decay should HELP consolidation, not hurt it.

**Testable prediction (arm 3):** If logit KD shows BETTER consolidation than control during WSD decay (unlike rep KD which showed worse), it confirms the basin compatibility theory and means logit KD provides PERSISTENT benefit, not just a head-start.

### Prediction: Logit KD Pattern

If logit KD is qualitatively different from rep KD, we should see:
- Initial gap similar to rep KD or possibly smaller (logit KD is noisier due to cross-tokenizer alignment)
- Gap WIDENS or STAYS CONSTANT during stable LR phase (unlike rep KD which narrowed)
- Gap persists through WSD decay (control doesn't catch up like it did with rep KD)
- **NEW: Logit KD should show BETTER consolidation during decay than control** (basin compatibility). This is the opposite of rep KD's pattern. If confirmed, it's the strongest evidence for logit KD's persistent value.

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

### Orthogonality Analysis: Step 1000 Data (INTERFERENCE CONFIRMED)

**Test at step 1000 (with alpha-corrected linear scaling):**
- Δ_rep = 4.911 - 4.909 = +0.002 (rep slightly better)
- Δ_logit = 4.911 - 5.095 = -0.184 (logit much worse)
- Δ_both_actual = 4.911 - 5.039 = -0.128

**Naive additive (full α):** Δ_both = +0.002 + (-0.184) = -0.182. Actual -0.128 < |-0.182| → LESS bad than additive.

**Alpha-corrected linear (α halved in arm 4):** Expected = 0.5×(+0.002) + 0.5×(-0.184) = -0.091. Actual -0.128 → **41% WORSE than linear prediction** (0.128/0.091 = 1.41).

**Verdict: INTERFERENCE.** The two surfaces compete for the same parameter budget at 1:19 ratio. Adding rep KD to logit KD doesn't help — it actively makes the logit penalty worse by ~41%. The student can barely absorb ONE KD surface at this ratio; asking it to absorb two is destructive.

**Rate-distortion interpretation:** At capacity R (90M params), the student can minimize distortion on ONE KD surface or the NTP loss, but not multiple simultaneously. Adding a second surface splits the gradient between competing objectives, each of which individually fails to converge (logit KD loss never drops; CKA converges but doesn't help BPT). The joint optimization is strictly worse than either alone.

**Implication for 15K gate:** CONFIRMS logit-only (no rep KD). The 15K config already uses logit-only. The question is whether MECHANISM improvements (α schedule, τ schedule, confidence gating) can rescue the logit signal — not whether to combine surfaces.

**Implication for RMFD (future):** Multi-surface KD at extreme ratios needs TEMPORAL separation (one surface at a time), not spatial combination (both surfaces simultaneously). A curriculum approach — rep KD early for stabilization, then switch to logit KD for knowledge transfer — might avoid the interference. This is testable as a future experiment.

**NEW: Interference is EMERGENT (step 500 vs 1000):**
- Step 500 orthogonality: α-corrected linear prediction = 0.5×(+0.038) + 0.5×(+0.257) = +0.148. Actual = +0.147. **Interference factor: 0.99 — PERFECT ADDITIVITY.**
- Step 1000 orthogonality: α-corrected linear prediction = 0.5×(-0.002) + 0.5×(+0.184) = +0.091. Actual = +0.128. **Interference factor: 1.41 — 41% WORSE.**
- Step 1500 orthogonality: α-corrected linear prediction = 0.5×(-0.094) + 0.5×(+0.271) = +0.089. Actual = +0.110. **Interference factor: 1.24 — 24% WORSE (declining from 1.41).**
- Step 2000 orthogonality: α-corrected linear prediction = 0.5×(-0.130) + 0.5×(+0.272) = +0.071. Actual = +0.152. **Interference factor: 2.14 — MASSIVE SPIKE.**
- Step 2500 orthogonality: α-corrected linear prediction = 0.5×(+0.004) + 0.5×(+0.327) = +0.166. Actual = +0.210. **Interference factor: 1.26 — WSD reduced IF from 2.14.**
- Step 3000 orthogonality: α-corrected linear prediction = 0.5×(+0.018) + 0.5×(+0.285) = +0.151. Actual = +0.153. **Interference factor: 1.01 — PERFECT ADDITIVITY restored at low LR.**
- **Full interference trajectory: 0.99 → 1.41 → 1.24 → 2.14 → 1.26 → 1.01.** Full cycle: additive → interference → plateau spike → WSD mitigation → additivity restored.
- **Absolute gap trajectory: +0.147 → +0.128 → +0.110 → +0.152 → +0.210 → +0.153.** Peaked during WSD onset (step 2500), then narrowed during deep WSD.
- **Step 2000 regression:** Arm 4 BPT went from 4.940 (step 1500) to 4.979 (step 2000) — WORSE despite 500 more steps. Control was flat (4.830→4.827). Rep-only improved (-0.039). Only arm 4 regressed.
- **Step 2500 kurtosis spike:** 4.5 → 5.8 (now ABOVE control's 5.0). Same WSD instability pattern as arm 2.
- **Step 3000 kurtosis ALARM:** 12.4 — nearly 2.5× control (5.0), 2× logit-only (6.6). Combined surfaces create severe activation concentration during deep WSD despite reasonable BPT. Would likely diverge in longer training.
- **Arm 4 deep WSD recovery (2500→3000): -0.242 (-4.9%) vs control -0.185 (-4.0%).** Combined arm recovers MORE than control during deep WSD — logit component drives this (matching arm 3 pattern). But the absolute gap of +0.153 means the recovery isn't enough to catch up.
- **Interpretation:** At step 500, the surfaces operate on nearly disjoint parameter subsets (rep KD pulls shape, logit KD pulls output distribution, minimal overlap). By step 1000, gradient updates have entangled the parameters — both surfaces now modify overlapping weights, creating destructive interference. This is a CAPACITY SATURATION effect: the 90M student has enough "free" capacity at step 500 to accommodate both surfaces, but as training progresses and the model's parameter budget gets committed to specific features, the surfaces start competing for the same weights.
- **Temporal separation corollary:** If interference grows over time, the optimal strategy is SHORT-DURATION multi-surface exposure: use both surfaces for ~500 steps (while additive), then commit to one surface for the remainder. This is the exact structure of the Ekalavya curriculum (Phase 1: stabilization with rep, Phase 2+: logit-only for knowledge).

**Interference trajectory model (REVISED with step 2000 — prior model FALSIFIED):**
- Full trajectory: 0.99 (step 500) → 1.41 (step 1000) → 1.24 (step 1500) → **2.14 (step 2000)**
- **Prior model (declining-interference) was WRONG.** Step 2000 shows interference doubles when training enters plateau.
- **Revised model: Interference is ANTI-CORRELATED with NTP learning rate.**
  - Phase 1 (steps 0-500): Fresh parameters, surfaces non-overlapping → additive (IF≈1.0)
  - Phase 2 (steps 500-1000): Surfaces entangle → initial interference peak (IF≈1.4)
  - Phase 3 (steps 1000-1500): Partial disentanglement → temporary decline (IF≈1.2)
  - **Phase 4 (steps 1500-2000): PLATEAU AMPLIFICATION.** Control barely improves (-0.003), NTP gradient signal weakest → KD surfaces dominate gradient, compete freely → interference spikes to 2.14.
  - Phase 5 prediction (steps 2400-3000, WSD decay): LR drops → ALL gradients weaken equally → interference may persist since it's a ratio, not an absolute magnitude
- **Key insight:** Interference factor measures RELATIVE gradient competition, not absolute. When NTP signal weakens at plateau, the KD surfaces fill the vacuum and compete with each other. The model can't learn from NTP (plateau) AND can't learn from KD (interference). It regresses.
- **Revised implication for RMFD:** Multi-surface KD is TOXIC during plateau regions. MUST be paired with alpha scheduling that reduces KD weight when NTP signal weakens. The inverted-U schedule (alpha peaks in high-LR region, zeros during WSD) is even more important than we thought.
- **Step 2500 prediction:** WSD decay begins at step 2400. Two competing effects: (a) LR drop reduces gradient magnitudes equally → interference ratio unchanged, (b) arm 3 (logit-only) historically shows 23% MORE WSD recovery than control → logit component may benefit. Predict IF drops to ~1.5 (partial WSD mitigation) with BPT 4.83-4.88. Arm 4 BPT still worse than control.
- **Step 3000 prediction:** WSD nearly complete. LR ~1e-5. All gradients tiny. Arm 3 typically shows dramatic WSD recovery. Predict IF drops to ~1.1-1.3, BPT 4.62-4.70. Arm 4 lands in 4.60-4.75 range (per decision matrix: expected interference row).

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
- **Step 2000: -0.130 BPT BETTER (4.698 vs 4.827). CROSSOVER WIDENING. ✓ THEORY CONFIRMED.**
  - Kurtosis: 3.7 (rep) vs 5.4 (control) — 31% lower. Max act: 56.2 vs 81.4 — 31% lower.
  - Trajectory: +0.038 → -0.002 → -0.094 → **-0.130**. Monotonically improving gap.
  - This is NOT noise — consistent widening across 3 consecutive evals.
- **Theoretical interpretation: Structured disruption → better local optima. CONFIRMED by step 2000.** α=1.0 rep KD at extreme ratios acts like a structured regularizer. It disrupts the student's initial representation space (step 500 penalty), forces relearning in the teacher's geometric basin (step 1000 recovery), and converges to a flatter minimum with better generalization (step 1500+ advantage). Lower kurtosis and max_act support the "flatter minimum" hypothesis. Analogous to sharpness-aware minimization but the perturbation direction is INFORMED by teacher geometry rather than random. First probe (α=0.8) was too gentle to force the basin escape — α=1.0 is the critical threshold.
- **Step 2500 (WSD decay): GAP COLLAPSED. +0.004 BPT (tied/slightly worse).** BPT=4.6880 vs control 4.6838.
  - Control improved -0.144 during decay (4.827→4.684). Rep-only improved only -0.010 (4.698→4.688).
  - **15x consolidation gap.** Decay massively favored control.
  - Kurtosis SPIKED: 3.7→5.6 (now above control's 5.0). Max_act: 56.2→70.2 (also above control's 68.4).
  - **The "structured disruption" advantage was a TRAINING DYNAMIC, not permanent knowledge transfer.**
  - This perfectly mirrors the first KD probe: gap peaked mid-training, collapsed during WSD decay.
  - **Rep KD at α=1.0 = confirmed head-start only. Twice.** Different configs (α=0.8 first time, α=1.0 second), same result.
  - 500 more steps of deep decay remain. Predict: gap stays near zero or goes slightly negative (control may overshoot slightly during aggressive decay). Final gap will be ≤0.01 either direction.
- **REVISED structured disruption theory:** α=1.0 rep KD does push the student into a different basin during stable-LR training. But this basin is NOT "flatter" — it's just DIFFERENT. During WSD decay consolidation, the control finds an equally good or better local minimum. The kurtosis spike (3.7→5.6) during decay suggests rep-only's basin may actually be LESS stable under LR changes. **The advantage was trajectory, not topology.**

**Arm 3 (logit_only, α=1.0): THE CRITICAL TEST.**
- ~~Step 500: predict BPT ≤ control. Logit KD is prediction-aligned, should NOT show penalty.~~
- **Step 500: BPT=5.2783. +0.257 WORSE than control (5.021). PREDICTION WRONG.**
  - Penalty is 6.8x worse than rep-only's +0.038. This is not a subtle effect.
  - KD loss at 1.79 — comparable magnitude to CE (3.72). α=1.0 causes massive gradient competition.
  - Kurtosis 3.4, max_act 57.6 — healthy. Problem is NOT activation collapse. Problem is gradient allocation.
  - **Root cause hypothesis: flat α=1.0 + T=2.0 + extreme ratio = too much KD gradient noise in early training.**
  - **Crude heuristic diagnosis:** The implicit behavior (flat α=1.0 from step 0) is maximally harmful at initialization. The student barely knows NTP; asking it to simultaneously match a 1.7B teacher's flattened distribution is asking it to learn two things at once when it can barely do one. A rising α schedule would be a crude-but-directional fix.
- **Step 1000: BPT=5.095. +0.184 (partial recovery from +0.257).** Gap narrowed 0.073 in 500 steps. KD loss flat at 1.84 — NOT declining. Student is not learning from teacher logits.
- **Step 1500: BPT=5.102. +0.272 (GAP WIDENED BACK). RECOVERY STALLED AND REVERSED.**
  - Went from +0.257→+0.184→+0.272. The apparent recovery at step 1000 was transient.
  - KD loss at 2.07 — still oscillating ~1.7-2.1 with zero convergence.
  - This is where rep-only was at -0.094 (BETTER than control). Logit-only is +0.272 (MUCH worse).
  - **Interpretation: cross-tokenizer logit KD at α=1.0/T=2.0/1:19 is ACTIVELY HARMFUL.** Not just noise — it's pulling the student in the wrong direction. The gradient from KL(teacher||student) on a 92.6% shared vocabulary with causal alignment noise is destructive at this ratio.
  - Kurtosis 4.1, max_act 59.1 — healthy. The damage is in learning, not in activations.
- Step 2000-3000: Predict continued underperformance. Gap may narrow slightly during WSD decay if KD gradient weakens with LR, but recovery to control is unlikely.
- **Kurtosis: 3.4 — CONFIRMED < 2x control (3.1×2=6.2). ✓ Logit KD is a safer signal for activations.**
- **15K action regardless of step 3000 result:** The crude heuristic analysis identified rising α as TRIVIAL and HIGH impact. Recommend for 15K gate even if arm 3 result is disappointing — the flat α=1.0 is clearly suboptimal.

**Arm 4 (rep+logit, total α=1.0: state=0.3125, sem=0.1875, logit=0.5):**
- Rate-distortion theory predicts INTERFERENCE at extreme ratios (rep and logit compete for bits).
- α_logit=0.5 (half of arm 3's 1.0) → logit penalty should be ~halved
- α_state=0.3125 + α_semantic=0.1875 = 0.5 (half of arm 2's 1.0) → rep effect also ~halved
- If penalty scales linearly: logit contribution ~+0.143, rep contribution ~+0.009 → total ~+0.152 → BPT≈4.650
- **Predict BPT@3K: 4.60-4.75 (between arm 2 and arm 3, much closer to arm 2)**

**Orthogonality test (now using ACTUAL arm 1-3 data):**
- Δ_rep = 4.498 - 4.516 = -0.018 (rep slightly worse)
- Δ_logit = 4.498 - 4.783 = -0.285 (logit much worse)
- Additive (both penalties stack): Δ_both = -0.303, BPT ≈ 4.801 (WORST — but unrealistic since α halved)
- Linear α scaling (most likely): Δ_both ≈ -0.018/2 + -0.285/2 = -0.152, BPT ≈ 4.650
- Synergistic (rep alignment helps logit): BPT < 4.650
- Interfering (surfaces compete): BPT > 4.750

**Step-by-step predictions for arm 4:**
| Step | Control | Predicted Arm 4 | Predicted Δ | **Actual** | **Actual Δ** | Match? |
|------|---------|-----------------|-------------|------------|--------------|--------|
| 500  | 5.021   | 5.10-5.15       | +0.08-0.13  | **5.169**  | **+0.147**   | SLIGHTLY ABOVE — logit penalty worse than halved |
| 1000 | 4.911   | 4.98-5.02       | +0.07-0.11  | **5.039**  | **+0.128**   | ABOVE — penalty recovery stalled, logit dominates |
| 1500 | 4.830   | 4.90-4.95       | +0.07-0.12  | **4.940**  | **+0.110**   | ✓ IN RANGE. Interference: 1.24 (declining from 1.41) |
| 2000 | 4.827   | 4.88-4.94       | +0.05-0.11  | **4.979**  | **+0.152**   | ✗ ABOVE RANGE. Regression! IF=2.14. Plateau amplifies interference. |
| 2500 | 4.684   | 4.73-4.78       | +0.05-0.10  | **4.893**  | **+0.210**   | ✗ ABOVE RANGE. IF=1.26 (WSD helped IF), but gap widened. Kurtosis 5.8 (above ctrl 5.0). |
| 3000 | 4.498   | 4.60-4.75       | +0.10-0.25  | **4.651**  | **+0.153**   | ✓ IN RANGE. IF=1.01 (additivity restored at low LR). Kurtosis 12.4 (ALARMING). Decision: logit-only inverted-U at 15K. |

**Arm 4 @500 Analysis:** Actual Δ=+0.147 vs predicted +0.08-0.13. Logit penalty is NOT halved — it's +0.147 vs arm 3's +0.257 at step 500, so ~57% of full logit penalty (predicted ~50%). The rep surface ISN'T fully compensating. However, kurtosis=2.5 is remarkably LOW (control=3.1, arm 2=3.2, arm 3=3.4 at step 500) — the combination is stabilizing activations even if not helping BPT. Max_act=50.4 also healthy. This suggests rep KD IS doing something (stabilization) even though it doesn't translate to BPT benefit. Key question: will stabilization translate to better WSD consolidation?

**Arm 4 @1500 Analysis:** Actual BPT=4.940, Δ=+0.110 vs control 4.830. Predicted range +0.07-0.12 → ✓ WITHIN. Penalty narrowing accelerating: -0.019/500 steps (500→1000) → -0.018/500 steps (1000→1500). Kurtosis=3.9 — LOWER than control (4.9). This is remarkable: the combined arm has HEALTHIER activations than control, despite worse BPT. Max_act=57.1 stable. Orthogonality test: α-corrected linear predicts +0.089, actual +0.110. Interference factor 1.24 (declining from 1.41 at step 1000). **Interpretation:** As training progresses, the surfaces are partially disentangling — the initial entanglement peak has passed. But interference persists above additive (24% excess penalty). The declining trend suggests that by step 2000-2500, interference may approach linear (especially as WSD decay reduces gradient magnitudes).

**Arm 4 @2000 Analysis:** Actual BPT=4.979, Δ=+0.152 vs control 4.827. **REGRESSION from step 1500 (4.940).** This is arm 4 getting WORSE despite 500 more steps. Control was flat (4.830→4.827). Rep-only improved (-0.039). Only arm 4 regressed. Kurtosis 4.5 (up from 3.9), max_act 60.1 (up from 57.1). Orthogonality: α-corrected linear predicts +0.071 (the lowest yet due to rep-only's improving delta), actual +0.152. **IF=2.14 — HIGHEST interference in the entire run.** Prior model (declining IF) was WRONG. Root cause: the training plateau (control barely moved -0.003 over 500 steps) creates a vacuum where NTP gradient signal is weakest, and the two KD surfaces fill this vacuum and compete with each other freely. The model can't learn NTP (plateau) AND can't learn from KD (interference). Net result: regression. **This strongly validates the inverted-U alpha schedule for 15K gate** — alpha must decrease when NTP signal weakens.

**Arm 4 @1000 Analysis:** Actual Δ=+0.128 vs predicted +0.07-0.11. Penalty narrowed only -0.019 in 500 steps. For comparison: arm 2 recovered -0.040 (from +0.038 to -0.002); arm 3 recovered -0.073 (from +0.257 to +0.184, then reversed). Arm 4 is recovering SLOWER than either individual arm. More critically, kurtosis rose from 2.5→3.4 and max_act from 50.4→57.9 — the step-500 stabilization advantage is GONE. This means the rep surface's stabilization effect was transient (first ~500 steps only). The combined arm now tracks as a ~50% amplitude version of arm 3's logit penalty, with the rep surface contributing almost nothing beyond initial stabilization. **Prediction revision for 1500:** penalty will plateau like arm 3 did, probably in range +0.10 to +0.15. The rep surface won't produce the crossover (arm 2's -0.094 at 1500) because the logit penalty dominates.

**Decision matrix (REVISED with actual arms 1-3 data):**
| Arm 4 BPT@3K | vs Control | vs Arm 2 (4.516) | vs Arm 3 (4.783) | Interpretation | 15K Action |
|---------------|-----------|-------------------|-------------------|----------------|------------|
| < 4.50 | BETTER | BETTER | BETTER | Synergy — surfaces complement. Remarkable. | Run rep+logit with inverted-U α at 15K |
| 4.50-4.60 | +0.00-0.10 | Better | Better | Linear scaling — halving α halves penalty | Run logit with inverted-U α at 15K (surface doesn't matter much, mechanism does) |
| 4.60-4.75 | +0.10-0.25 | Worse | Better | Expected — penalty from logit, mild rep help | Run logit with inverted-U α at 15K |
| > 4.75 | > +0.25 | Worse | ~Same/Worse | Interference — combined is as bad or worse | Abandon rep, try logit-only with inverted-U α |

**NEW: Extreme-ratio mitigations for Codex #274 review (from §6.4.29 research):**
Regardless of which arm wins, the 15K gate should consider:
1. **Rising τ schedule** (τ=1.5→4.0): POCL shows "critical" at 15x ratio. Trivial to implement
2. **Reverse KL / AMiD (α=-3 to -5)**: Mode-seeking instead of mode-covering. Tested at exact 15x ratio
3. **MiniPLM data reweighting**: 2.2x compute savings, pilot already done (task #237)
4. **Qwen3-0.6B as alternative primary teacher**: 1:7 ratio (within comfort zone) vs 1:19
These are QUESTIONS FOR CODEX, not decisions. Present as options in the evidence document.

### Crude Heuristic Analysis: Making Implicit Mechanisms Explicit (2026-03-26)

**Principle (from Devansh):** A crude explicit mechanism >> no mechanism. The value isn't precision but existence — explicit = tunable lever.

**Current implicit mechanisms in our KD system and proposed crude-but-directional fixes:**

| Implicit Behavior | What's Wrong | Crude Explicit Heuristic | Effort | Expected Impact |
|---|---|---|---|---|
| Every token gets equal KD weight | Teacher confident on some tokens, guessing on others. Flat weight wastes gradient on noise. | **Confidence gating:** if max(teacher_softmax) > 0.5, scale KD loss ×1.5; if < 0.1, scale ×0.3. | LOW (5 lines) | MEDIUM — focuses gradient on learnable knowledge |
| Static T=2.0 for all training | Early training needs focused signal; later training benefits from broader dark knowledge. | **Rising τ:** τ=1.5 for steps 0-5K, linear ramp to 3.0 by 10K, hold at 3.0. | TRIVIAL (3 lines) | HIGH per POCL paper |
| Static α=1.0 for all training | α=1.0 causes initial penalty (proven: arm 2 step 500). Student needs NTP stability first. | **Rising α:** α=0.3 for steps 0-1K, linear ramp to 1.0 by 3K. | TRIVIAL (3 lines) | MEDIUM — avoids early disruption |
| All shared tokens weighted equally | High-frequency tokens have reliable alignment stats; rare tokens are noisy. | **Frequency-weighted KD:** weight each shared token by log(freq+1) during loss. | LOW (10 lines) | LOW-MEDIUM |
| Uniform data sampling | MiniPLM scores show some shards are much more informative for KD. | **MiniPLM shard weighting:** sample shards proportional to teacher-student divergence. Pilot done (task #237). | LOW (already built) | MEDIUM per MiniPLM paper (2.2× savings) |
| No gradient quality monitoring | Bad KD gradients (high variance, conflicting with NTP) silently dilute training. | **Gradient cosine monitor:** if cos(∇NTP, ∇KD) < -0.1 for 3 steps, halve α temporarily. | MEDIUM (20 lines) | MEDIUM — prevents destructive interference |

**The meta-point for Codex #274:** Even if the ablation shows weak KD signal, there are 6 crude heuristics that could amplify whatever signal exists. The ablation tests the SURFACE (rep vs logit); these heuristics optimize the MECHANISM. Both matter. Recommend Codex pick the top 2-3 for the 15K gate.

**Connection to manifesto:** This IS "Intelligence = Geometry" applied to the training process itself. We're not adding more parameters or data — we're adding STRUCTURE to how existing signal flows. The crude heuristic principle says even rough structure >> no structure.

### Derived: Inverted-U Alpha Schedule for 15K Gate (2026-03-26)

**First-principles derivation from ablation evidence:**

The 3K ablation proved two things: (1) flat α=1.0 from step 0 causes persistent penalty, (2) WSD decay collapses whatever KD advantage existed. Both are symptoms of α being WRONG at different training phases:
- Early: student geometry far from teacher → large noisy KD gradients → high α = destructive
- Mid: student geometry established → KD gradients informative → high α = beneficial
- Late (WSD decay): NTP must reconsolidate → KD pulls toward KD-optimal basin → high α = destructive again

**Optimal shape = inverted U**, not monotonic ramp:
```
α(t):  0.2 ──ramp──> 0.7 ──hold──> 0.7 ──taper──> 0.1 ──hold──> 0.0
       |    2K steps   |   8K steps   |    2K steps  |   3K steps |
       0              2K             10K            12K          15K
       [NTP stabilize] [KD peak zone] [decay onset]  [pure NTP]
```

**Why NOT ramp to 1.0?** At 1:19 ratio, student can never represent teacher distribution. KD loss oscillates 1.5-2.2 even at α=1.0 — there's no convergence to wait for. Capping at 0.7 limits gradient competition while still providing substantial dark knowledge signal.

**Why taper during WSD decay?** Basin compatibility theory (confirmed in this ablation): the KD-optimal basin is SHALLOWER on the NTP surface. During consolidation (LR decay), NTP determines final quality. KD interference during consolidation = worse final BPT. Taper α alongside LR to let NTP reconsolidate.

**Falsifiable predictions:**
- Inverted-U at 15K should show BPT < control by ≥0.02 (persistence threshold)
- No kurtosis spike during WSD decay (unlike arm 2's 3.7→5.6)
- KD loss should DECREASE during peak zone (student absorbing knowledge)
- If inverted-U STILL shows penalty → the issue is ratio, not schedule. Next: try Qwen3-0.6B (1:7)

**WSD consolidation rates (deep WSD: steps 2500-3000, all arms):**
- Control: 4.684 → 4.498 = -0.185 (3.97%) — baseline consolidation
- Arm 2 (rep): 4.688 → 4.516 = -0.172 (3.67%) — LESS consolidation (rep features resist NTP reconsolidation)
- Arm 3 (logit): 5.011 → 4.783 = -0.228 (4.55%) — MORE consolidation, **23% better than control**
- **Key: logit KD recovers 23% MORE during deep WSD than control.** The flat-α=1.0 penalty is NOT permanent feature corruption — it's gradient competition during constant-LR that WSD partially undoes. This means the 15K gate's clean WSD window (α=0 during decay) should recover even more of the penalty.
- **Arm 2 pre-WSD stagnation (steps 2000→2500):** Control improved -0.144, rep improved only -0.010. **15× consolidation gap.** Rep KD's mid-training advantage evaporates immediately when training dynamics shift.

**Critical asymmetry in arm 2 vs arm 3 during WSD decay:**
- **Arm 2 (rep-only):** LOST advantage during WSD. BPT went from -0.130 (step 2000) to +0.018 (step 3000). Rep KD's benefit EVAPORATED during consolidation.
- **Arm 3 (logit-only):** RECOVERED during WSD. BPT went from +0.327 (step 2500) to +0.285 (step 3000). Logit KD's penalty SHRANK during consolidation.
- **Interpretation:** Rep KD creates basin structure that conflicts with NTP's optimal basin (α on different surface). Logit KD adds knowledge that's COMPATIBLE with NTP consolidation (same surface — output distribution). WSD decay exposes this: it drives the model toward the NTP-optimal basin, which destroys rep alignment but preserves logit knowledge.
- **The inverted-U schedule EXPLOITS this asymmetry:** High α during peak transfers maximum logit knowledge; zero α during decay allows clean NTP consolidation that PRESERVES the transferred knowledge. If we used rep KD instead, the WSD decay would destroy whatever was gained.

**Literature support:**
- CTKD (AAAI 2023): constant T is suboptimal. Temperature should INCREASE as student progresses. Validates rising α/τ.
- POCL (June 2025): rising τ at 15x ratio (our exact setup). Staged exposure improves convergence significantly.
- InDistill (2024): explicit warmup stage for KD. α starts at 0, linearly increases.
- **Our TAPER during WSD decay is NOVEL.** No paper we found tapers α alongside LR decay. This is our contribution from the basin compatibility finding.

**Implementation cost:** TRIVIAL using phased training mode. 4 phases, each with different alpha.

### Pre-Registered Predictions for 15K Gate (2026-03-26)

**Config:** Logit-only, α peak=0.60 inverted-U, τ=1.5→3.0, confidence gating ON. From 5K warm-start. Eval every 1K steps.

**15K Gate Checkpoint Predictions:**

| Step | Control BPT (est) | KD Arm Predicted | Predicted Δ | Rationale |
|------|-------------------|-----------------|-------------|-----------|
| 1000 | **4.895** | 4.88-4.91 | -0.01 to +0.02 | α ≈ 0.25 (ramping). Very little KD signal yet. Should track control. |
| 2000 | **4.824** | 4.79-4.84 | -0.03 to +0.02 | α approaching peak (0.60). First meaningful KD signal. |
| 3000 | **4.680** | 4.63-4.71 | -0.05 to +0.03 | α at peak. KD should be active. This is the FIRST real test. |
| 4000 | **4.632** | — | — | **ACTUAL.** Slower than predicted (expected ~4.59). Plateau onset. |
| 5000 | **4.612** | 4.55-4.63 | -0.06 to +0.02 | **ACTUAL control.** Was ~4.55 predicted, 0.062 worse. Deceleration confirmed. Kurtosis=10.1. |
| 6000 | **4.616** | 4.55-4.63 | ≤0 (STOP RULE) | **ACTUAL.** Deep plateau. -0.003 from 5K. Kurtosis SPIKED to 41.7 (see analysis below). |
| 7500 | ~4.50 | 4.43-4.52 | -0.07 to +0.02 | **Actual control ~4.52 (interpolated).** |
| 9000 | **4.442** | 4.38-4.46 | -0.06 to +0.02 | **ACTUAL.** Kurtosis 227.1 spike confirmed transient at 10K. |
| 10000 | **4.420** | 4.36-4.44 | ≤-0.02 (STOP RULE) | **ACTUAL.** Decelerating (-0.022/1K). Named checkpoint saved. |
| 11000 | **4.512** | — | — | **ACTUAL.** Regression (+0.092). Eval noise — confirmed at 12K (4.374). |
| 12000 | **4.374** | 4.31-4.40 | -0.06 to +0.03 | **ACTUAL.** WSD decay starts. Named checkpoint saved. α→0 for KD arm. |
| 13000 | **4.284** | 4.22-4.31 | -0.06 to +0.03 | **ACTUAL.** 33% WSD. -0.090/1K. Kurtosis spike 257.7 (TRANSIENT). |
| 14000 | **4.1314** | 4.07-4.15 | -0.06 to +0.02 | **ACTUAL.** 67% WSD. -0.152/1K (ACCELERATING! 1.69× the 12-13K rate). Kurtosis 442.8 (periodic spike). |
| 15000 | **4.0820** | ≤4.067 | ≤-0.015 (PROOF) | **ACTUAL. CONTROL COMPLETE.** KD arm STARTED. KD needs ≤4.067 at 15K for proof. |

**Scenarios and Decision Tree:**

| Outcome | BPT Gap@15K | lm-eval lift? | Probability | Next Step |
|---------|-------------|---------------|-------------|-----------|
| **A: WORKS** | ≤-0.015 | YES (≥1 bench) | 35% | → RMFD with forward KL at 90M. Validated mechanism. |
| **B: HELPS/FADES** | -0.01 to -0.005 | Maybe | 25% | → Try AMiD (α=-5) 3K probe at 90M. If no lift → scale to 166M. |
| **C: NEUTRAL** | ±0.005 | No | 20% | → Scale to 166M directly (1:8.8 ratio). Forward KL may work at better ratio. |
| **D: FAILS/HURTS** | >+0.005 | No | 20% | → Scale to 166M + try AMiD. If both fail → abandon KD at 1:10, focus on overtraining. |

**Decision tree execution:**
1. Finish both arms (control + KD) to step 15K
2. Run lm-eval on both 15K checkpoints
3. Compute BPT gap (eval BPT, not training BPT)
4. Compare lm-eval scores across all 7 benchmarks
5. Route to outcome A/B/C/D based on data
6. Codex Tier 2 review (Scaling Expert + Architecture Theorist + Competitive Analyst) before ANY major decision
7. Execute next step

**For ALL outcomes:** The 90M checkpoint at 20K total steps is our most trained model. Use it as warm-start for whatever comes next (scaling, further KD, or overtraining).

**Key monitoring signals:**
- KD loss trajectory: should DECREASE during peak phase (student learning from teacher). If flat → no knowledge transfer.
- Confidence gating activation: how many tokens get ×1.5 vs ×0.3? If mostly ×0.3 → teacher is mostly uncertain → wrong teacher.
- Kurtosis during taper: should stay < 2× control (stable consolidation). Spike = basin instability.

### 15K Gate Control Arm Curve Analysis (2026-03-27)

**Actual control data (steps are relative to gate start, from 5K warm-start):**

| Step | BPT | ΔBPT/1K | Kurtosis | Max Act | Phase |
|------|-----|---------|----------|---------|-------|
| 1000 | 4.895 | — | 3.2 | 51.5 | Stable LR (3e-4) |
| 2000 | 4.824 | -0.071 | 4.0 | 56.2 | Stable LR |
| 3000 | 4.680 | -0.144 | 5.2 | 72.5 | Stable LR (fast learning phase) |
| 4000 | 4.632 | -0.048 | 7.2 | 87.9 | Decelerating |
| 5000 | 4.612 | -0.020 | 10.1 | 107.4 | Plateau onset |
| 6000 | 4.616 | +0.003 | 41.7 | 97.8 | Apparent plateau. Kurtosis spike TRANSIENT (see below). |
| 7000 | 4.538 | -0.077 | 11.0 | 98.7 | Plateau broken. Resumed learning. |
| 8000 | 4.498 | -0.040 | 17.9 | 105.9 | Decelerating after post-stall spike. Steady progress. |
| 9000 | 4.442 | -0.056 | 227.1* | 206.6* | Kurtosis/max_act SPIKE — confirmed TRANSIENT at step 10K. |
| 10000 | 4.420 | -0.022 | 51.9 | 133.7 | Decelerating. Named checkpoint saved. WSD decay in 2K steps. |
| 11000 | 4.512 | +0.092 | 25.7 | 140.2 | REGRESSION — eval noise at flat-LR plateau. Kurtosis healthy (down). |
| 12000 | 4.374 | -0.138* | 45.0 | 124.0 | WSD DECAY STARTS. Named checkpoint saved. Last flat-LR eval. |
| 13000 | 4.284 | -0.090 | 257.7* | 175.1* | 33% through WSD. BPT dropping fast (-0.090/1K). Kurtosis spike TRANSIENT. |

*Step 9K spikes confirmed TRANSIENT: kurtosis 227.1→51.9, max_act 206.6→133.7. Same pattern as step 6K (41.7→11.0).

**Learning rate profile:** Steps 1-10K: flat 3e-4. WSD decay starts at step 12K (1K steps away from step 11K).

**Observations (updated step 12K — WSD DECAY ONSET):**
1. **Step 11K regression CONFIRMED as eval noise.** BPT trajectory: 4.420 (10K) → 4.512 (11K, noise) → **4.374 (12K)**. Underlying trend: 4.420→4.374 over 2K steps = -0.023/1K, consistent with flat-LR deceleration. The -0.138 from 11K→12K is the noise bouncing back.
2. **ΔBPT/1K trajectory (ignoring 11K noise):** -0.071, -0.144, -0.048, -0.020, +0.003, -0.077, -0.040, -0.056, -0.022, (noise), **-0.046/2K≈-0.023/1K**. Steady deceleration in late flat LR.
3. **Kurtosis: 45.0.** Down from 51.9 at step 10K. Elevated but stable. No spike.
4. **Max_act: 124.0.** Down from 133.7 at step 10K. Mild. No concern.
5. **WSD DECAY DELIVERING.** Step 12K→13K: -0.090 BPT (vs -0.023/1K at flat LR). That's 4× improvement rate. LR at 13K: 2.03e-4 (33% through decay). WSD providing massive consolidation as expected.
6. **Step 13K kurtosis spike (257.7):** Same pattern as steps 6K (41.7) and 9K (227.1). Spikes every ~3K steps, BPT improves simultaneously. Kurtosis_max is pure noise at this scale.
7. **Step 14K: BPT=4.1314 (WSD S-CURVE CONFIRMED).** Drop 13K→14K: -0.152 (1.69× the 12K→13K rate of -0.090). Efficiency went from 357 to 981 BPT/LR. The constant-efficiency deceleration model was WRONG — WSD efficiency increases during decay because the optimizer settles into sharper basins at lower LR. Kurtosis 442.8 (periodic spike, 4th in the ~3K cadence).
8. **Revised control@15K: ~4.03-4.10.** If efficiency stays at 981: drop = -0.057 → 4.074. If efficiency increases: 4.03-4.05. Conservative: 4.09. **Central estimate ~4.07.**
9. **Critical: step 12K checkpoint saved.** This is the PIVOTAL reference point for basin compatibility analysis. Both arms receive identical treatment from 12K→15K (pure NTP, decaying LR). Any gap at 15K is determined by where the model sits at 12K.

**Implication for KD arm:** The control is stronger than we feared during the step 6K scare. The KD arm's job is harder — it needs to beat a control that is actively improving, not plateaued. But the stop rule at step 6K (non-positive) is relative to control at step 6K, so the bar adjusts dynamically.

**Implication:** The original predictions were too optimistic. Revised forward estimates now in prediction table above. This actually makes the KD arm's job EASIER — the control baseline is weaker than expected, so a smaller absolute improvement from KD represents a larger relative gain.

**LR schedule (confirmed from code):** WSD = Warmup-Stable-Decay.
- Steps 0-500: warmup (0 → 3e-4)
- Steps 500-12000: flat 3e-4 (80% of 15K)
- Steps 12000-15000: linear decay (3e-4 → 1e-5)

**This changes everything about the extrapolation.** The flat-LR phase runs until step 12K. The -0.020/1K deceleration at step 5K is pure diminishing returns on the loss landscape at constant LR. Steps 12K-15K will see a massive BPT drop during WSD decay (as we observed in the 3K ablation).

**Revised extrapolation (incorporating step 13K actual: 4.284, WSD delivering):**
WSD decay active and delivering. Step 12K→13K: -0.090 BPT in 1K steps (4× the flat-LR rate).

**S-curve model (efficiency INCREASES during WSD — confirmed at step 14K):**
- 12K→13K: avg LR = 2.52e-4 → drop = -0.090 → efficiency = 357 BPT/LR
- 13K→14K: avg LR = 1.55e-4 → drop = **-0.152 (ACTUAL)** → efficiency = **981 BPT/LR** (2.75× increase!)
- The constant-efficiency model predicted -0.055 at 14K; actual was -0.152. WSD is an S-curve, not linear.
- 14K→15K: avg LR = 5.85e-5. If efficiency stays at 981: drop = -0.057 → BPT@15K ≈ 4.074
- If efficiency continues increasing: drop ≈ -0.08 to -0.10 → BPT@15K ≈ 4.03-4.05
- Conservative (efficiency reverts toward mean ~670): drop = -0.039 → BPT@15K ≈ 4.09
- **BPT@15K range: 4.03-4.10.** Central estimate ~4.07.

**WSD S-curve dynamics:** The LR decay doesn't just reduce gradient noise — it enables the optimizer to "settle" into a sharper basin. Each subsequent LR reduction is MORE productive per unit LR because the loss landscape curvature is better aligned with the reduced step size. This is the well-known WSD effect: the last 20% of training produces outsized improvements.

**CONTROL FINAL: 4.0820. KD arm needs ≤4.067 for -0.015 persistence proof.**

## KD Arm Tracking (2026-03-27, started after control completion)

**Setup:** Same 5K warm-start, 15K steps. Qwen3-1.7B teacher (92.6% vocab overlap, 3.4GB VRAM). Inverted-U alpha (0.10→0.60→0.10→0.0), rising tau (1.5→3.0), confidence gating ON. VRAM: ~18GB total.

**Stop Rules (from config):**
- Non-positive by 6K: KD BPT must be < control BPT at step 6K (4.616). **First decisive check.**
- Evidence by 10K: KD gap ≤ -0.02 vs control (need KD ≤ 4.400)
- Proof by 15K: KD gap ≤ -0.015 vs control (need KD ≤ 4.067) + lm-eval lift
- Abort: kurtosis > 2× control or max_act > 1.15× control

**KD Arm Data:**
| Step | KD BPT | Control BPT | Gap | KD Loss | Alpha | Notes |
|------|--------|-------------|-----|---------|-------|-------|
| 50 | 5.4496 | — | — | 0.3349 | 0.19 | First data. Alpha warmup active. |

## Basin Compatibility Theory: Why Logit > Rep During WSD (2026-03-27)

**Status: HYPOTHESIS — derived from ablation data, not yet tested independently**

**The asymmetry:** Rep KD advantage collapses during WSD (-0.130 → +0.018), while logit KD penalty SHRINKS during WSD (+0.327 → +0.285, -12.8%). Why?

**Surface compatibility hypothesis:** KD surfaces that operate on the SAME mathematical object as NTP are "basin-compatible" — they push the model toward minima that also look good under NTP. Surfaces that operate on DIFFERENT objects create basins that may be locally attractive but globally misaligned with NTP's landscape.

| Surface | Mathematical object | Same as NTP? | Basin compatible? |
|---------|-------------------|-------------|-------------------|
| NTP (cross-entropy) | Output logit distribution | — | Reference |
| Logit KD (forward KL) | Output logit distribution | YES | YES — both optimize output distribution |
| CKA (rep KD) | Pairwise representation similarity | NO | NO — optimizes internal geometry, not output |
| Semantic relational | Teacher-student correlation | NO | NO — optimizes latent relationships |

**Prediction from theory:**
- Basin-compatible surfaces (logit KD) → advantages survive LR decay
- Basin-incompatible surfaces (rep KD) → advantages evaporate during LR decay
- Mixed surfaces → interference during stable LR, instability during WSD

**Evidence:**
1. Rep KD peaks at -0.130 during stable LR, collapses to +0.018 after WSD ✓
2. Logit KD shows 23% MORE deep WSD recovery than control ✓
3. Combined (rep+logit) kurtosis spikes to 12.4 during WSD (2.5× control) ✓
4. Combined IF returns to 1.01 at very low LR (surfaces decouple when gradients vanish) ✓

**Implication for RMFD:** The future multi-teacher system should use logit-based surfaces for all teachers (basin-compatible), with rep surfaces only as auxiliary signals during early training (stabilization phase, then off). This is exactly what the Ekalavya curriculum prescribes — rep early, logit sustained.

**Connection to information geometry:** The NTP loss lives on the simplex of output distributions. Logit KD also lives on this simplex. Rep KD lives on a Grassmannian (the space of subspaces/representations). The WSD landscape changes on the simplex preserve logit KD's knowledge because they're moving within the same manifold. Rep KD's gains lie on a different manifold entirely — LR decay on the simplex has no obligation to preserve structure on the Grassmannian.

### RMFD Design Revision Implications (2026-03-27)

**The RMFD 4-surface design needs revision based on basin compatibility findings:**

| Original Surface | Type | Basin-compatible? | Revision |
|-----------------|------|------------------|----------|
| Token surface (logit KD) | Output distribution | YES | **KEEP — primary surface** |
| State surface (CKA at depth 8,16) | Grassmannian | NO | **DROP — or early-phase only** |
| Semantic surface (pooled rep) | Grassmannian | NO | **DROP — or early-phase only** |
| Exit surface (self-distillation) | Output distribution | YES | **KEEP — simplex-compatible** |

**Committee member impact:**
- **Qwen3-1.7B**: Logit KD ✓ (has LM head)
- **LFM2.5-1.2B**: Logit KD ✓ (has LM head)
- **Mamba2-780M**: Logit KD ✓ (has LM head)
- **EmbeddingGemma-300M**: Logit KD ✗ (encoder, no LM head) — **CANNOT do basin-compatible KD**

**Options for EmbeddingGemma:**
1. Drop entirely from committee (simplest, saves 0.6GB VRAM)
2. Use for early-phase state KD only (Phase 2), disable during WSD
3. Replace with a small decoder model (e.g., SmolLM-135M)

**Design question: should α be nonzero during WSD?** Current inverted-U zeros α at step 12K (when WSD starts). But basin compatibility says logit KD HELPS during WSD (same manifold). A small residual α (0.05-0.10) during steps 12K-15K might improve consolidation. **FLAG FOR CODEX REVIEW after 15K gate.**

**Caveat:** All of this is derived from ONE experiment at ONE scale. Question all conclusions before acting.

### Step 12K = Pivotal Checkpoint (2026-03-27)

**Key realization:** Alpha zeros out at step 12K. WSD decay also starts at step 12K. From 12K→15K, BOTH arms receive identical treatment (pure NTP, declining LR 3e-4→1e-5). Any gap at step 15K is entirely determined by the BASIN the model occupies at step 12K.

**This means:**
1. **Step 12K BPT comparison is the raw signal.** If KD arm is worse at 12K → very unlikely to win at 15K.
2. **Step 12K→15K BPT drop measures basin quality.** A deeper WSD drop = better basin. Basin compatibility theory predicts logit KD basin consolidates BETTER under NTP-only WSD.
3. **The 15K gate is actually a 12K gate + 3K verification.** The KD mechanism's effect ends at 12K. Steps 12K-15K just reveal whether the effect was durable.

**Pre-registered step 12K predictions:**
- Control@12K: ~4.38-4.42 (continued deceleration from flat LR)
- KD@12K: If mechanism works → 4.33-4.40 (≤ -0.02 gap). If fails → ≥ 4.38 (no gap)
- Both should then drop ~0.20-0.30 during WSD, but KD basin should drop MORE (basin compatibility)

**LR at each step during decay:**
| Step | LR | Decay progress |
|------|-----|---------------|
| 12000 | 3.00e-4 | 0% |
| 12500 | 2.52e-4 | 17% |
| 13000 | 2.03e-4 | 33% |
| 13500 | 1.55e-4 | 50% |
| 14000 | 1.07e-4 | 67% |
| 14500 | 5.83e-5 | 83% |
| 15000 | 1.00e-5 | 100% |

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

## Pre-Computed: 166M Scaling Config (for fallback path)

**Status: CONFIG ONLY — implementation pending 15K gate results + Codex decision**

If we need to scale from 90M → 166M, the best config is:

| Param | 90M (current) | 166M (target) | Notes |
|-------|--------------|---------------|-------|
| dim | 512 | 704 | √(166/90) × 512 ≈ 695, rounded to 704 (divisible by 64) |
| n_layers | 24 | 24 | Same depth (enables warm-start widening) |
| n_heads | 8 | 11 | 11 heads × 64 = 704. head_dim=64 is GPU-optimal (not 8×88). |
| head_dim | 64 | 64 | Standard. 88 (704/8) works but non-pow2 hurts GPU throughput. |
| ff_dim | 1536 | 2112 | 3 × dim |
| params | 90.2M | ~166M | 1.84× |
| KD ratio | 1:19 (5.3%) | 1:10 (9.8%) | At threshold for effective KD |
| Train VRAM | ~7GB | ~10GB | Plenty of room for teacher (~2GB) |

**Warm-start widening path:**
1. Copy 512-dim weights into first 512 dims of 704-dim matrices
2. Zero-pad remaining 192 dimensions
3. Scale output projections by 512/704 to preserve function initially
4. Random init new attention heads (if adding heads)
5. Need Net2Wider implementation — check if current code supports this

**Alternative: d=768, 197M** — cleaner config (12 heads × 64) but exceeds 166M target by 30M. Ratio 1:9 (11.6%) — well above KD threshold. VRAM ~10.5GB.

---

## PREPARED: Post-15K-Gate Codex Evidence Template (2026-03-27)

**Status: TEMPLATE — fill in blanks when gate completes, then send to Codex Tier 2**

```
[FILL] Control BPT@15K: ___
[FILL] KD BPT@15K: ___
[FILL] BPT Gap: ___
[FILL] Control kurtosis@15K: ___
[FILL] KD kurtosis@15K: ___
[FILL] lm-eval control: ARC-E=___, ARC-C=___, HS=___, WG=___, PIQA=___, SciQ=___, LAMBADA=___
[FILL] lm-eval KD: ARC-E=___, ARC-C=___, HS=___, WG=___, PIQA=___, SciQ=___, LAMBADA=___
```

**Codex prompt (Tier 2: Scaling Expert + Architecture Theorist + Competitive Analyst):**

```
MANDATORY FIRST STEP: Read CLAUDE.md in this repository root. [standard preamble...]

ACTUAL TASK: Review the 15K benchmark gate results as a panel of three concurrent reviewers.

## DATA

### Training Curves (BPT at eval checkpoints)
Control: [FILL full curve 1K→15K]
KD arm: [FILL full curve 1K→15K]
Gap: [FILL gap at each checkpoint]

### Stability Metrics
Control kurtosis: [FILL trajectory]
KD kurtosis: [FILL trajectory]
Max activations: [FILL]

### lm-eval Benchmarks (step 15K)
[FILL table: benchmark × (control, KD, delta)]

### Key Findings from Surface Ablation (prior experiment)
- Rep-only KD: head-start only, basin-incompatible (Grassmannian vs simplex)
- Logit-only KD: harmful at flat α=1.0, but basin-compatible
- Multi-surface: interference factor peaked at 2.14 during plateau
- Basin compatibility theory: logit surfaces survive WSD, rep surfaces don't

### 15K Gate Mechanism
- Inverted-U alpha: 0.10→0.60→0.10→0.0 (warmup 2K, peak 2K-10K, taper 10K-12K, zero 12K-15K)
- Rising tau: 1.5→3.0 over 10K steps
- Confidence gating: scale by teacher p_max (×1.5 if >0.5, ×0.3 if <0.1)
- Cross-tokenizer: DSKDv2-style ETA on 14,822 shared tokens (92.6%)

## QUESTIONS FOR EACH REVIEWER

### Scaling Expert (Persona 3)
1. Does the BPT gap SCALE or SHRINK with more training steps? What's the trend?
2. If we scale from 90M→166M (1:19→1:10 ratio), how much should we expect KD benefit to improve?
3. What's the compute-optimal training schedule for the 166M model with KD?
4. Is the kurtosis trajectory concerning for scaling?

### Architecture Theorist (Persona 6)
1. Does the basin compatibility theory hold under the 15K data?
2. Should α be nonzero during WSD (logit KD helps consolidation)?
3. Is the inverted-U schedule optimal, or should we explore other shapes?
4. What does the training curve shape tell us about the loss landscape?

### Competitive Analyst (Persona 8)
1. Where do these 15K lm-eval scores place us vs baselines (SmolLM-135M, Pythia-160M)?
2. How many training steps would we need WITHOUT KD to match the KD arm?
3. Is the KD efficiency gain meaningful (saves N% of training compute)?
4. What should we build next based on competitive positioning?

## DECISIONS NEEDED
1. Proceed with RMFD at 90M? Or scale to 166M first?
2. Which divergence: forward KL, AMiD, or something else?
3. RMFD surface selection: logit-only or logit+exit self-distillation?
4. EmbeddingGemma: drop from committee or use for early-phase state KD?
5. Training budget: 120K steps at 90M or 60K steps at 166M?
```

---

