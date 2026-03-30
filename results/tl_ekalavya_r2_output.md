The Round 2 evidence says Ekalavya should stop being prior-led and become hard-gated, bottlenecked, and difficulty-conditioned. Evidence: [R2 prompt](C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/tl_ekalavya_r2_prompt.md#L38), [committee map JSON](C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/committee_map_audit.json#L8), [scratchpad synthesis](C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/research/SCRATCHPAD.md#L13), [locked student spec](C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/research/ARCHITECTURE.md#L38), [60K config](C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/kd_197m_60k_gate.json#L7).

**Phase A**
- `Warm-start from 60K`: confirmed harder. The 197M control is real now, not speculative: ARC-E `45.4`, ARC-C(n) `25.3`, HS(n) `29.0`, PIQA `59.4`, WG `51.3`. Throwing away that basin would be irrational.
- `Per-span routing`: confirmed. Span entropy drops from `3.44` at span `0` to `2.29` at span `15`, so routing by whole sample would erase the strongest structure in the new data.
- `Temporal surface separation`: confirmed. EmbeddingGemma has no logit surface, rep losses were already fragile, and cross-domain research converges on staging.
- `Q0.6 as the single anchor`: challenged. LFM has the lowest KL gap (`2.08` vs `2.80` Q0.6, `3.15` Q1.7), so it is the better bootstrap teacher. Q0.6 should be the fallback/default decoder, not the unquestioned anchor.
- `Exclusive one-teacher-per-span`: challenged, not killed. With only `14.1%` low-margin disagreement, most spans are not “pick one master or die.” The right rule is `hard shortlist first, then either single winner or top-2 log-pool when the top pair is close`.
- `Routing prior in the score`: dead. `pi=0.5` pushed Q0.6 to `87.5%` of spans. Remove fixed priors from the score entirely. Use controller caps outside the score if collapse appears.
- `Compatibility = truncated cosine`: dead. Replace it with `64`-dim shared-bottleneck projector similarity plus window-level CKA regularization. GW is a probe, not the Round 2 mainline.
- `EmbeddingGemma in routing`: confirmed dead for token routing. Keep it semantic-only.
- `Complex cross-tokenizer token KD`: mostly unnecessary now. With `92.6%-94.0%` vocab overlap, token KD should use exact shared-vocab alignment with renormalization. Keep byte spans only for state/semantic alignment and routing.
- `Cross-domain inverse effectiveness + hard gating`: yes, this changes the curriculum. Easy spans should often get `null` KD. Q1.7 should only touch the hardest spans, late.

**Phase B**
Research requests for Round 3:
- Run the same lm-eval slice on `Q0.6`, `LFM`, and `Q1.7` so anchor choice is tied to actual `HS/PIQA/ARC/WG` strengths, not proxy metrics.
- Work out exact top-`K` log-pooling on partial-vocab overlap so consensus spans are mathematically clean.
- Compare `32/48/64` bottlenecks; the cross-domain prior says `32-64`, but the repo has no evidence yet for the smallest safe size.

Probe requests for Round 3:
- `Single vs committee`: `LFM-only`, `Q0.6-only`, `LFM→Q0.6 dual-anchor`, `dual-anchor + late Q1.7`. If committee does not beat best-single-teacher, stop paying routing tax.
- `Compatibility bakeoff`: truncated cosine vs `64d projector cosine + CKA` vs GW. The winning metric must predict which teacher gives the best `500-1000` step continuation on the same spans.
- `Hard gate vs top-2 consensus vs soft weighting`: same alpha envelope, same cache, same batch budget.
- `EmbeddingGemma ablation`: semantic tail on/off; if it does not move HS/PIQA or retrieval-like internal metrics, remove it from v1.
- `Offline-cache pilot`: rolling cached teacher logits/states vs online one-teacher forward. Measure throughput and peak VRAM, not just loss.

**Phase C**
Use a simpler v2 system: no learned router, no prior term, no novelty term until fixed.

Core roles:
- `LFM`: bootstrap/state anchor.
- `Q0.6`: default decoder teacher.
- `Q1.7`: late hard-span specialist.
- `EmbeddingGemma`: semantic-only tail.

Exact shapes:
- Student stays locked: `24 x 768`, `SwiGLU 2304`, exits `7/15/23`, `16` spans, `512` context.
- Teacher hidden sizes from local configs: `Q0.6=1024`, `LFM=2048`, `Q1.7=2048`, `EmbGemma=768`.
- Shared bottleneck: `64`.
- Projectors:
```text
P_s^r:   768 -> 64, r in {7,15,23}
P_q06^r: 1024 -> 64
P_lfm^r: 2048 -> 64
P_q17^r: 2048 -> 64
P_s^sem: 768 -> 64
P_emb:   768 -> 64
```
- Parameter overhead with per-depth state projectors plus semantic pair: `1,229,696` trainable params. Much better than the R1 `~7.2M` side-system.

Routing v2:
```text
H_m = mean token entropy of student on span m
d_m = clip((H_m - 2.29) / (3.44 - 2.29), 0, 1)

u_s^{r,m} = norm(RMS(P_s^r h_s^{r,m}))
u_t^{r,m} = norm(RMS(P_t^r h_t^{phi_t(r),m}))

kappa_{m,t} = 0.75 * 0.5*(1 + cos(u_s^{15,m}, u_t^{15,m}))
            + 0.25 * CKA(U_s^{15}, U_t^{15})
```

Hard gates:
```text
theta_null = 0.20
theta_lfm  = 0.20
theta_q06  = 0.50
theta_q17  = 0.80 for steps < 12K, then 0.70

kappa_min = 0.55 for steps < 12K, then 0.60
gate_{m,t} = 1[d_m >= theta_t] * 1[kappa_{m,t} >= kappa_min] * 1[EMA grad-cos_t > -0.10]
```

Teacher score, with committee-map numbers baked in:
```text
g_t in {2.08, 2.80, 3.15}          # mean KL gaps
c_t in {0.321, 0.332, 0.373}       # confidences
v_t in {0.940, 0.926, 0.926}       # vocab overlap fractions

g'_t = (g_t - 2.08) / 1.07
c'_t = (c_t - 0.321) / 0.052
v'_t = (v_t - 0.926) / 0.014

score_{m,t} = gate_{m,t} * (0.55 g'_t + 0.20 c'_t + 0.15 kappa_{m,t} + 0.10 v'_t)
```

Selection rule:
- If `d_m < 0.20`, use `null`.
- Else choose the top eligible teacher.
- If the top two eligible teachers differ by `< 0.08` and their span-level JS on mapped shared-vocab top-`64` logits is `< 0.15`, use top-2 log-pooling instead of winner-take-all.
- No teacher is allowed to exceed `45%` route share on a `2K`-step EMA before `12K` unless it is also best on routed loss. This is the anti-`87.5%` collapse rule.

Loss:
```text
L = L_ntp + 0.20 L_exit + alpha_state L_state + alpha_sem L_sem + mean_m alpha_eff(m,t,u) L_tok(m,t)

alpha_eff(m,t,u) = alpha_t(u) * d_m^{gamma_t}
gamma_lfm = 1.25
gamma_q06 = 1.50
gamma_q17 = 2.00
```

Surface losses:
```text
L_state = sum_r lambda_r * [0.75 * (1 - mean_m cos(u_s^{r,m}, u_lfm^{r,m}))
                          + 0.25 * (1 - CKA(U_s^r, U_lfm^r))]
lambda = {0.2, 0.3, 0.5}

L_sem = 1 - cos(P_s^sem h_s^23, P_emb h_emb)

L_tok = tau_t^2 * KL(p_t^{tau_t} || p_s^{tau_t})
```
`L_tok` is on exact shared-vocab mapped ids with `top_k=64`; no AMiD, no OT on the token surface in v2.

Schedule:
- Optimizer: `AdamW`. Keep it. The new blocker is KD control, not optimizer uncertainty.
- LRs: backbone `1e-4`, projector stack `2.5e-4`, `min_lr=1e-5`.
- Betas / wd: `(0.9, 0.95)`, `wd=0.1` backbone, `0.05` projectors.
- Batch: `8`, grad accum `4`, seq `512`, spans `16`, warmup `500`, total continuation `24K`, WSD over final `4.8K`.
- Precision: student/projectors `bf16`, optimizer `fp32`, cached teacher logits/states `fp16` on disk/CPU.
- Number system: real-valued only. There is no new evidence that complex/quaternion math attacks the actual blocker.

Alpha / tau curriculum:
- `0-500`: no external KD.
- `500-3K`: `alpha_state 0→0.03`, `alpha_sem 0.01→0.02`, `alpha_lfm 0→0.18`, `alpha_q06 0→0.10`, `alpha_q17 0`.
- `3K-12K`: `alpha_state 0.03→0 by 6K`, `alpha_sem 0.02`, `alpha_lfm 0.18→0.28`, `alpha_q06 0.10→0.22`, `alpha_q17 0`.
- `12K-20K`: `alpha_sem 0.02→0.015`, `alpha_lfm 0.24`, `alpha_q06 0.20`, `alpha_q17 0→0.10 by 16K then hold`.
- `20K-24K`: `alpha_lfm 0.08`, `alpha_q06 0.06`, `alpha_q17 0.02`, `alpha_sem 0.01`, `alpha_state 0`.
- `tau_lfm 1.2→1.8`, `tau_q06 1.4→2.0`, `tau_q17 1.8→2.4`.

Feasibility:
- Best practical path is a rolling offline KD cache: cache top-`64` logits for all `3` generative teachers plus `64d` span states, refresh every `1-2K` steps.
- Step-time VRAM stays near the control run: about `10.5-11GB`, because teachers are off the critical path.
- The online audit path with one teacher loaded at a time should still stay safely under `24GB`.

**Confidence**
- `O1 = 7/10`. `+1` from R1 is justified by new real benchmarks: `45.4 ARC-E`, `25.3 ARC-C(n)`, `51.3 WG`, `29.0 HS(n)`, `59.4 PIQA` at `197M/60K`. The backbone is now strong enough that KD can matter. Still capped at `7` because there is no positive committee win yet.
- `O2 = 8/10`. Unchanged. New data improves diagnosis, not proof: we now know the exact failures are prior-dominated routing, broken compatibility, and EmbeddingGemma surface mismatch. That is strong improvability evidence, but not yet module-swap evidence.
- `O3 = 7/10`. Unchanged. The analytic router and `1.23M` projector interface are more composable than the R1 learned side-system, but there is still no contributor workflow evidence.
- `O4 = 8/10`. `+1` from R1 is justified by new empirical blocker removal: shared vocab is `92.6%-94.0%`, so cross-tokenizer token KD is no longer the hard part, and the teacher gap ordering `2.08 / 2.80 / 3.15` gives a concrete curriculum instead of guesswork.
- `O5 = 7/10`. Unchanged. New span-entropy structure (`3.44 -> 2.29`) supports difficulty-aware allocation, but there is no new exit-quality or latency result in this round.