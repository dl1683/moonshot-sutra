Context checked in [VISION.md](/C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-sutra/research/VISION.md), [ARCHITECTURE.md](/C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-sutra/research/ARCHITECTURE.md), [RESEARCH.md](/C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-sutra/research/RESEARCH.md), [dense_baseline.py](/C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-sutra/code/dense_baseline.py), and the latest probe artifact [kd_anchor_probe_soft6k_results.json](/C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-sutra/results/kd_anchor_probe_soft6k_results.json). The newest checkpoint is [lfm_soft6k_step6000.pt](/C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-sutra/results/checkpoints_anchor_probe_soft6k/lfm_soft6k_step6000.pt). I could not execute `nvidia-smi` in this shell because the command was blocked by policy, so GPU availability below uses the prompt’s stated “RTX 5090 fully free” status. I did not find a committed plaintext training log for the soft probe; the JSON result file is the authoritative trajectory.

1. **Assumption challenges with both-sides analysis**
- `Backbone lock`: Obvious interpretation: the 24x768 pure transformer should stay locked because the local hybrid campaign only found small, unstable gains and [ARCHITECTURE.md] shows backbone spread on the order of `0.07-0.18 BPT`. Alternative interpretations: the hybrid family may have failed only at `90M`, or a different adaptive-depth mechanism like MoR/LoopFormer could matter later. Distinguishing evidence: a matched `197M` hybrid would need to beat the locked backbone by `>=0.05 BPT` without worse stability. Universal or specific: this lock is specific to this codebase and current budget, not a universal “hybrids are dead.” Lean: keep the lock, `9/10`.
- `197M is the right student size`: Obvious interpretation: `24 x 768` is the best compromise because `197M@15K` already beats `90M@15K` on ARC-E/PIQA/WG and still leaves teacher headroom. Alternative interpretations: a smaller `125M` student might absorb KD more cleanly, or a `300M+` student might be needed for multi-teacher returns. Distinguishing evidence: matched warm-start continuations at `125M`, `197M`, and `~300M` under the same KD schedule. Universal or specific: very specific to the 24GB laptop budget. Lean: keep `197M`, `8/10`.
- `The soft LFM result proves KD works`: Obvious interpretation: yes, because `lfm_soft6k` ends at `3.5686`, below the warm-start checkpoint `3.5726`, and all exits improve. Alternative interpretations: the gain may come mostly from the longer WSD/consolidation schedule, or LFM may be acting as a regularizer rather than transferring durable teacher knowledge. Distinguishing evidence: run a matched `ce_soft6k` with the identical 6K schedule and compare the post-tail endpoint. Universal or specific: specific to this schedule, this teacher, and this checkpoint. Lean: positive but not decisive, `7/10`.
- `LFM should be the primary anchor`: Obvious interpretation: yes, because CKA compatibility is best for LFM (`0.865` mean vs `0.736` for Q0.6 and `0.674` for Q1.7) and LFM was the first teacher to produce a positive endpoint. Alternative interpretations: Q0.6 might still outperform LFM under the softened 6K schedule, or LFM’s win may mostly reflect softer logits rather than better knowledge content. Distinguishing evidence: run `q06_soft6k`, and run a temperature-matched control where Q0.6 is softened to the same effective entropy. Universal or specific: specific to this student and continuation regime. Lean: LFM first, `8/10`.
- `Implement the full 4-teacher router now`: Obvious interpretation: yes, because the backbone is locked and O4 is the mission. Alternative interpretations: full 4-teacher routing would hide the source of gains; the real next blocker is still “best single teacher vs matched CE,” not “all teachers at once.” Distinguishing evidence: a ladder `CE -> LFM -> Q0.6 -> static LFM+Q0.6 -> routed LFM+Q0.6 -> add Q1.7 -> add EmbGemma`. Universal or specific: specific to this project stage. Lean: do a 2-teacher proof first, `9/10`.
- `Q1.7 belongs in the first routed run`: Obvious interpretation: no, because committee-map and CKA data say it is a sparse hard-span specialist and the old scoring collapses toward it when ungated. Alternative interpretations: once routing is fixed, Q1.7 may be the source of the largest hard-example gains; excluding it too long could cap O1/O4. Distinguishing evidence: a late-phase A/B where Q1.7 is added only after a positive 2-teacher route, with a hard `<=10-12%` route-share cap. Universal or specific: specific to the current `197M` student. Lean: delay Q1.7 until after a positive 2-teacher result, `8/10`.
- `top_k=64 should be reduced immediately`: Obvious interpretation: yes, because the CTI scratch analysis says effective competitors per token are small. Alternative interpretations: reducing `k` now may lose useful tail mass while saving little, because the current code still materializes the full shared-vocab tensor before top-k selection. Distinguishing evidence: a matched `K_loss=16` vs `64` ablation after routing is live, while keeping routing metrics on `64`. Universal or specific: specific to the current implementation in [dense_baseline.py](/C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-sutra/code/dense_baseline.py). Lean: do not cut it immediately in the current codepath, `6/10`.
- `Teacher KD should be applied directly to exits 7 and 15`: Obvious interpretation: yes, because O5 needs shallow exits. Alternative interpretations: direct teacher pressure on shallow exits may overconstrain weak layers; the soft probe already improved all exits without shallow teacher losses, so late internal self-distillation may be the cleaner mechanism. Distinguishing evidence: add only final-to-exit KL in the CE tail and compare against teacher-to-exit KD. Universal or specific: specific to fixed-exit training at this scale. Lean: no teacher KD at shallow exits yet; use late self-distillation instead, `7/10`.
- `The 16K tokenizer should remain fixed`: Obvious interpretation: yes, because tokenizer efficiency is one of the strongest validated wins in the repo and overlap with Qwen/LFM is still `>92%`. Alternative interpretations: a shared teacher tokenizer or full DSKDv2 stack could make multi-source KD easier later. Distinguishing evidence: only revisit if low-overlap teachers or non-text teachers become primary. Universal or specific: specific to this corpus and parameter budget. Lean: keep `16K`, `8/10`.

2. **Research requests for Claude to execute**
- Research whether post-teacher consolidation tails are a known pattern in KD or self-distillation, especially “teacher off, student continues” schedules; I want exact recipes and failure modes, not generic annealing advice.
- Research multi-teacher DSKDv2/CDM implementation details for separate teachers with different tokenizers; the key question is whether student-side projector structure can be shared across teachers or must stay teacher-specific.
- Research best-layer selection for cross-architecture feature matching; CTI suggests final-layer alignment is often wrong, so I want concrete layer-pairing heuristics for LFM, Q0.6, Q1.7, and EmbeddingGemma.
- Research encoder-teacher integration for decoder students at sub-500M scale; specifically whether low-weight semantic alignment improves downstream commonsense benchmarks or mostly regularizes representation geometry.
- Cross-domain request: look for ambiguity-aware consensus and cross-inhibition mechanisms from neuroscience/swarm decision systems that can be implemented with negligible compute on a 16-span routing grid.

3. **Experiment/probe requests with methodology**
- `Probe 1: ce_soft6k`. Clone [probe_anchor_soft6k.json](/C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-sutra/results/probe_anchor_soft6k.json), remove teachers, keep the identical 6K LR/warmup/WSD schedule. This resolves the central ambiguity. Promote KD reality only if `lfm_soft6k` beats this control by `>=0.02 BPT` at step `6000`.
- `Probe 2: q06_soft6k`. The config already exists in the same JSON. Run it. If Q0.6 beats LFM by `>=0.01 BPT` at the endpoint, LFM is not the token anchor and the schedule must flip.
- `Probe 3: lfm_q06_static6k`. Use the existing multi-teacher path in `train_kd()` with both teachers active and equal total alpha split across them. This is the zero-new-code committee baseline. If static mixing cannot beat the best single teacher, routing must earn its complexity before full implementation.
- `Probe 4: routed_lfm_q06_12k`. Implement the minimal 2-teacher span router only. Success threshold: endpoint `committee <= best_single - 0.02 BPT` after consolidation, plus `HS(n)` or `PIQA` gain `>= +1.0` with no material `ARC-E/WG` regression.
- `Probe 5: exit_tail_sd_6k`. Add only final-to-exit KL during the last `20-30%` of training on the best single-teacher or best committee run. Success threshold: `bpt_exit_7` and/or `bpt_exit_15` improve by `>=0.02` with final `BPT` degradation `<=0.005`.
- `Probe 6: k_ablation`. After a routed run is positive, compare `K_metrics=64, K_loss=64` against `K_metrics=64, K_loss=16`. Adopt the smaller loss-K only if endpoint delta is within `0.005 BPT` and route shares stay within `±5%`.
- `Probe 7: q17_late_sidecar`. Start from a positive 2-teacher routed checkpoint, add Q1.7 only in a late window with share cap `<=10-12%`. Keep it only if it improves endpoint by `>=0.01 BPT` and does not change LFM/Q0.6 route shares by more than `10%` relative.
- `Probe 8: config_only_swap`. Use the existing config-driven teacher loader in `train_kd()` / `train_kd_phased()` to add or remove EmbeddingGemma or swap to another encoder with no trainer-code change. This is the O3 composability proof.

4. **Per-outcome confidence with justification**
- Outcome 1 (Intelligence): `8/10` — up from `7/10`. New empirical data justifying the increase: `lfm_soft6k` reaches `3.5686` at step `6000`, below the warm-start checkpoint `3.5726`, and exit BPTs improve from `4.3315 / 3.7642 / 3.6460` at step `500` to `4.2864 / 3.7045 / 3.5686` at step `6000`. I am not giving `9/10` because there is no matched `ce_soft6k`, no new lm-eval, and the committed generation samples are still weak and repetitive.
- Outcome 2 (Improvability): `8/10` — unchanged. The project now has sharper diagnostics than before: committee-map collapse, CKA compatibility, and the soft-probe schedule all isolate specific failure modes. There is still no empirical “surgical fix” demo where one teacher-specific rule is changed and the model recovers without collateral damage.
- Outcome 3 (Democratization): `7/10` — unchanged. The code already supports config-driven multi-teacher loading and phased hot-swaps, which is real infrastructure evidence, not just philosophy. It is still not `8` or `9` because there is no demonstrated config-only extension result in the artifacts.
- Outcome 4 (Data Efficiency): `8/10` — unchanged. New data proves that single-teacher logit KD can be non-harmful and plausibly positive at `197M`, which is important. I am not raising to `9/10` because O4 is explicitly multi-source, and there is still no matched 6K CE control, no `q06_soft6k`, and no `committee > best single > CE` result.
- Outcome 5 (Inference Efficiency): `7/10` — unchanged. The new exit-level numbers are encouraging because all exits improved in the positive LFM run. I am not raising the score because there is still no `deterministic_eval_dense` exit policy result, no latency benchmark, and no proof that early-exit ordering stays good after KD.

5. **What would raise confidence**
- Outcome 1: run `ce_soft6k`, then a routed `lfm_q06_12k` or `24k`; raise to `9/10` if the best KD run beats the matched CE control by `>=0.03 BPT` after consolidation and improves `HS(n)` or `PIQA` by `>=1.0` with no meaningful `ARC-E/WG` regression.
- Outcome 2: run the Q1.7 late-sidecar recovery test; raise to `9/10` if a regression can be reversed within `500-1000` steps by changing only `Q1.7` routing gates or caps while LFM/Q0.6 behavior stays stable.
- Outcome 3: run the config-only swap proof; raise to `9/10` if a teacher can be added, removed, or replaced via JSON only, with no trainer-code edit, and the resulting gain composes with an existing positive KD setup.
- Outcome 4: raise to `9/10` only if the hierarchy `committee > best single > CE` is shown at matched continuation budget, with the committee beating the best single by `>=0.02 BPT`. A second path to `9/10` is a positive encoder add-on: `LFM+Q0.6+EmbGemma` must beat `LFM+Q0.6` by `>=0.01 BPT` or a benchmark delta.
- Outcome 5: raise to `9/10` only if a KD-trained checkpoint keeps exit ordering healthy under `deterministic_eval_dense` and `early_exit_inference`, with `>=20-25%` average compute or latency reduction at `<=0.02 BPT` cost.

6. **What would lower confidence**
- Outcome 1: a matched `ce_soft6k` that equals or beats `lfm_soft6k`, or a routed committee that improves BPT but not `HS/PIQA`.
- Outcome 2: any routed system where changing one teacher rule unpredictably moves other teachers’ shares or causes unrecoverable instability.
- Outcome 3: if config-driven swaps require trainer-code edits, optimizer surgery, or incompatible checkpoints, the democratization story falls sharply.
- Outcome 4: if `q06_soft6k` and the routed 2-teacher committee both fail to beat matched CE after consolidation, the current O4 strategy is overfitted to one LFM schedule and confidence should drop hard.
- Outcome 5: if KD improves final BPT but hurts shallow exits, collapses exit calibration, or yields no real latency win, O5 should drop.

7. **INTUITIONS**
- LFM may be helping mainly by reshaping the basin, not by improving online token loss. Trigger: active-KD steps are flat/worse, but the CE tail harvests the gain. Conviction: high. Validate with `ce_soft6k` plus a sharpness or loss-landscape proxy on the final checkpoint.
- Q0.6 may become much stronger if its target is softened or interpolated, not simply routed later. Trigger: LFM’s advantages look like “capacity-calibrated softness” as much as architecture. Conviction: medium. Validate with `q06_soft6k` and a TAID-style or higher-tau Q0.6 probe.
- Teacher-specific ZPD centers are probably better than centering every teacher at `gap/student_entropy = 1`. Trigger: committee-map ratios are already different in the repo: LFM is naturally gentler, Q0.6 harder, Q1.7 hardest. Conviction: medium-high. Validate by comparing universal-ZPD vs teacher-centered-ZPD routing on the same 2-teacher run.
- Direct teacher KD at shallow exits is probably the wrong O5 move; late student-final-to-exit KL is probably the right one. Trigger: the soft LFM probe improved all exits without any exit teacher loss. Conviction: medium-high. Validate with `exit_tail_sd_6k`.
- The right sparse setting is likely `K_metrics=64`, `K_loss=16`, not `16` everywhere. Trigger: CTI suggests tiny effective competitor sets, but the current router still needs stable JS/gap estimates. Conviction: medium. Validate with the `k_ablation`.

8. **Design proposal**
Need more data before a full 4-teacher production run. The recommended next mainline is a **gated 2-teacher routed proof**. Direct answers to the brief: Q1 is answered by the confidence section (`O1=8`, `O4=8`); Q2: do **not** implement the full 4-teacher system yet, do the 2-teacher proof first; Q3: do **not** cut `top_k` globally in the next run, keep `64` until routing is positive, then split `K_metrics=64` and `K_loss=16`; Q4: keep the budget at `24K` for the first routed proof, but make roughly half the run consolidation; Q5: do **not** add teacher KD directly to exits yet, use late internal self-distillation; Q6: yes, redesign around a long consolidation tail.

```text
Backbone:
- Student = locked Sutra-24A-197M
- 24 layers, d_model=768, GQA attention, FFN=2304 SwiGLU
- exits at layers 7, 15, 23
- real-valued hidden states
- SS-RMSNorm
- warm-start from control_197m_60k_step60000.pt

Teacher set for the next proof only:
- t in {LFM2.5-1.2B, Qwen3-0.6B}
- token surface only
- Q1.7 off
- EmbeddingGemma off

Routing variables per span m in {1..16}:
H_m        = mean token entropy of student on span m
gap_{m,t}  = mean KL(p_t || p_s) on aligned shared-vocab logits, K_metrics=64
conf_{m,t} = mean max_v p_t(v) on full teacher distribution
r_{m,t}    = log((gap_{m,t}+1e-4)/(H_m+1e-4))

Teacher-specific ZPD centers from committee-map ratios:
mu_lfm = log(0.84)
mu_q06 = log(1.13)

zpd_{m,t} = exp(- (r_{m,t} - mu_t)^2 / (2 * 0.35^2))

score_{m,t} =
  1[H_m >= 2.35]
  1[cka_{m,t} >= kappa_t]
  1[zpd_{m,t} >= 0.35]
  * cka_{m,t}^1.5
  * zpd_{m,t}^1.5
  * conf_{m,t}^0.25

Use:
kappa_lfm = 0.72
kappa_q06 = 0.65

Routing:
- if no teacher eligible: CE only on that span
- if JS(p_lfm, p_q06) <= 0.10 and |score_lfm - score_q06| < 0.05:
    consensus
- else:
    single-winner argmax_t score_{m,t}

Consensus:
log p_cons(v) = w_lfm log p_lfm(v) + w_q06 log p_q06(v)
w_t = score_{m,t} / (score_{m,lfm} + score_{m,q06})
```

```text
Loss:
L = CE_23 + 0.35 * CE_15 + 0.20 * CE_7
  + mean_m alpha_tok(m,u) * L_tok(m)
  + lambda_sd(u) * L_exit_sd

L_tok(single,t) = tau_t(u)^2 * KL(p_t^{tau_t(u)} || p_s^{tau_t(u)})
L_tok(cons)     = tau_c^2   * KL(p_cons^{tau_c} || p_s^{tau_c})

Set tau_c = 1.8

Late O5-only self-distillation:
L_exit_sd =
    0.05 * KL(stopgrad p_23^{1.5} || p_15^{1.5})
  + 0.03 * KL(stopgrad p_23^{1.5} || p_7^{1.5})
```

```text
24K schedule:
0-1K     : alpha_lfm 0 -> 0.08, alpha_q06 = 0
1K-3K    : LFM only, alpha_lfm = 0.08
3K-8K    : routed LFM+Q0.6, alpha_lfm = 0.08, alpha_q06 ramps 0 -> 0.06 then holds
8K-12K   : decay both alphas linearly to 0
12K-24K  : CE-only consolidation
14K-24K  : turn on L_exit_sd only

Temperature:
tau_lfm : 1.4 -> 2.2 over first 6K
tau_q06 : 1.5 -> 2.2 over 3K-8K

Controller audit every 256 steps:
- update cka EMA
- if one teacher share > 0.75 and routed-loss win <= 0, multiply its score by 0.9
- if gradient cosine with CE < -0.10 at two consecutive audits, reduce that teacher's alpha_max by 20%
```

- Optimizer: AdamW. Continuation LR `1e-4`, `min_lr=1e-5`, warmup `500`, WSD through the 24K run. Betas `(0.9, 0.95)`, weight decay `0.1`, clip `1.0`. I am not switching optimizers because local evidence in [RESEARCH.md](/C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-sutra/research/RESEARCH.md) favors AdamW/SS-RMSNorm over Muon for stability.
- Number system and precision: real-valued bf16 student, fp32 optimizer states, fp16/Q8 teachers as already used. I am not introducing complex or quaternion states here because O4 proof is the bottleneck and a new number system would confound attribution.
- Parameters: backbone `196.7M`; router/controller params are effectively zero in the analytic formulation above. If EmbeddingGemma is added later, one `768 -> 768` semantic projector costs `590,592` params. Token-only Q1.7 later adds no projector params.
- VRAM: this 2-teacher proof should sit comfortably under the prompt’s 4-teacher estimate. Practical expectation is roughly `student+activations 10-12GB`, LFM `~2.4GB`, Q0.6 `~1.2GB`, misc/controller/cache `<1GB`, so about `14-16GB`.
- Promotion rule to the real 4-teacher run: only after `ce_soft6k`, `q06_soft6k`, and the 2-teacher routed proof are all positive. Then add EmbeddingGemma first at low constant semantic weight, and only then test Q1.7 as a late, capped specialist.