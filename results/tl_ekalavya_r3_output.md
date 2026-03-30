**Phase A**
- LFM should move from “co-anchor” to primary anchor. The new CKA probe is decisive: mean CKA `0.865` for LFM vs `0.736` for Q0.6 and `0.674` for Q1.7, and LFM rises to `0.919` on span 15 while Q1.7 falls to `0.671` ([cka_compatibility_probe.json:49](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/cka_compatibility_probe.json#L49), [cka_compatibility_probe.json:71](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/cka_compatibility_probe.json#L71), [tl_ekalavya_r3_prompt.md:21](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/tl_ekalavya_r3_prompt.md#L21)).
- Q1.7B should be more restricted than in R2. The new evidence says it is not a general teacher; it is a sparse high-difficulty specialist. Gap×conf gives it `100%` of spans, which is exactly the failure mode we must avoid ([tl_ekalavya_r3_prompt.md:35](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/tl_ekalavya_r3_prompt.md#L35)).
- LFM should get more relative share on late spans, but not more total KD mass. Total KD should still follow inverse effectiveness, because span 0 is about `3.38x` heavier than span 15; within the late-span bucket, LFM should dominate more strongly than R2 assumed ([tl_ekalavya_r3_prompt.md:55](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/tl_ekalavya_r3_prompt.md#L55)).
- CKA should replace the broken compatibility term in the routing score. Do not use the old truncated cosine again. If you keep local projected cosine at all, keep it as telemetry or a low-weight tiebreaker after projector warmup, not as a primary routing signal ([tl_ekalavya_r2_output.md:10](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/tl_ekalavya_r2_output.md#L10), [tl_ekalavya_r3_prompt.md:51](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/tl_ekalavya_r3_prompt.md#L51)).
- ZPD is the correct core principle. Monotonic gap×conf is structurally wrong for a `197M` student because it always rewards the largest teacher. The right score is `compatibility × ZPD × weak confidence`, not `gap-heavy weighted sum`. The committee map already showed prior-driven collapse to Q0.6 (`87.5%` share), and removing the prior just flips collapse to Q1.7 ([committee_map_audit.json:8](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/committee_map_audit.json#L8), [tl_ekalavya_r3_prompt.md:38](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/tl_ekalavya_r3_prompt.md#L38)).

**Phase B**
- O1 to `9/10`: finish the 3K anchor probe and require `LFM-only` or `LFM+Q0.6 consensus` to beat the fresh-optimizer control by at least `0.03 BPT` at `1.5K` and `0.05 BPT` at `3K`, then show `+1.0` or better on `HS(n)` or `PIQA` with no `ARC-E/WG` regression ([tl_ekalavya_r3_prompt.md:68](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/tl_ekalavya_r3_prompt.md#L68)).
- O2 to `9/10`: run one surgical-fix proof. Example: enable Q1.7 late, detect regression, then recover by only tightening `CKA_q17` and `share_cap_q17`, with BPT recovering within `500-1000` steps and LFM/Q0.6 unchanged.
- O3 to `9/10`: prove config-level composability. Add or remove one teacher via config only, with no trainer code change, and show clean route-share and loss changes. Best test: swap `EmbeddingGemma` for another encoder or add/remove `Q1.7`.
- O4 to `9/10`: require `committee > best single teacher > CE control` at matched continuation budget. If the committee does not beat the best single teacher by at least `0.02 BPT`, multi-teacher routing is still not earning its complexity.
- O5 to `9/10`: run `deterministic_eval_dense` before and after KD and require no inference slowdown, no exit-order collapse, and better or flat shallow-exit quality ([dense_baseline.py:2252](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/dense_baseline.py#L2252)).

**Phase C**
- Backbone stays locked: `24 x 768`, `12` heads with GQA, `FFN 2304`, context `512`, exits `7/15/23`, `AdamW`, bf16 student/fp32 optimizer ([ARCHITECTURE.md:38](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/research/ARCHITECTURE.md#L38), [ARCHITECTURE.md:49](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/research/ARCHITECTURE.md#L49), [ARCHITECTURE.md:52](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/research/ARCHITECTURE.md#L52)).
- Warm-start from `control_197m_60k_step60000.pt`. Do not restart ([probe_anchor_config.json:3](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/probe_anchor_config.json#L3)).
- Teacher roles: `LFM = primary token/state anchor`, `Q0.6 = mid-band workhorse decoder`, `Q1.7 = sparse hard-span specialist`, `EmbeddingGemma = semantic-only`.
- Teacher dims: `Q0.6=1024`, `LFM=2048`, `Q1.7=2048`, `EmbGemma=768` ([RESEARCH.md:777](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/research/RESEARCH.md#L777), [RESEARCH.md:805](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/research/RESEARCH.md#L805), [RESEARCH.md:810](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/research/RESEARCH.md#L810), [RESEARCH.md:872](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/research/RESEARCH.md#L872)).
- Keep the `64d` bottleneck and rolling offline cache from R2: top-`64` logits for generative teachers plus `64d` span states, refresh every `2048` steps ([tl_ekalavya_r2_output.md:133](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/tl_ekalavya_r2_output.md#L133)).

```text
Per-span quantities:
H_m      = mean token entropy of student on span m
d_m      = clip((H_m - 2.20) / 1.25, 0, 1)          # committee-map init
w_inv(m) = clip(exp((H_m - 2.525) / 1.0), 0.75, 2.5)

gap_{m,t}  = mean KL(p_t || p_s) on mapped shared-vocab top-64 logits in span m
conf_{m,t} = mean max-prob of teacher over tokens in span m, from full teacher distro
cka_{m,t}  = EMA span-index CKA, initialized from cka_compatibility_probe.json

zpd_{m,t}  = exp(-abs(log((gap_{m,t}+1e-4)/(H_m+1e-4))) / 0.45)

gate_{m,t} = 1[d_m >= theta_t]
           * 1[cka_{m,t} >= kappa_t]
           * 1[zpd_{m,t} >= zeta_t]
           * 1[gcos_ema_t > -0.10]

score_{m,t} = gate_{m,t} * cka_{m,t}^1.5 * zpd_{m,t}^1.5 * conf_{m,t}^0.25
```

- Use `theta_lfm=0.10`, `theta_q06=0.20`, `theta_q17=0.60`.
- Use `kappa_lfm=0.70`, `kappa_q06=0.62`, `kappa_q17=0.78` before `12K`, then `0.75`.
- Use `zeta_lfm=0.25`, `zeta_q06=0.30`, `zeta_q17=0.40`.
- If `d_m < 0.15`, route `null` and apply CE only.
- If max pairwise JS among eligible teachers on mapped top-64 logits is `<= 0.10`, use consensus.
- Else if top-1 minus top-2 score is `< 0.05` and their pairwise JS is `<= 0.15`, use top-2 log-pooling.
- Else use single-winner routing.
- Consensus distribution: `log p_cons(v) = sum_t w_t log p_t(v)`, `w_t = score_{m,t} / sum_j score_{m,j}`.

```text
Loss:
L = CE_23 + 0.35 * CE_15 + 0.20 * CE_7
  + alpha_state(u) * L_state
  + alpha_sem(u)   * L_sem
  + mean_m alpha_tok(m,u) * L_tok(m)

alpha_tok(m,u) = w_inv(m) * alpha_mode(m,u)

L_state = sum_{r in {7,15,23}} lambda_r *
          [0.9 * mean_m (1 - cos(z_s^{r,m}, z_lfm^{r,m}))
         + 0.1 * (1 - CKA(Z_s^r, Z_lfm^r))]
lambda = {0.2, 0.3, 0.5}

L_sem = 1 - cos(e_s, e_emb)

L_tok(single)    = tau_t(u)^2 * KL(p_t^tau || p_s^tau)
L_tok(consensus) = 1.8^2 * KL(p_cons^1.8 || p_s^1.8)
```

- Surfaces: `LFM` supplies `state + token`; `Q0.6` supplies `token`; `Q1.7` supplies `token`; `EmbeddingGemma` supplies `semantic` only.
- Training schedule:
  - `0-500`: CE only.
  - `500-6K`: LFM state primer + LFM/Q0.6 token KD; Q1.7 off.
  - `6K-12K`: LFM/Q0.6 consensus-first token KD; state decays off by `6K`.
  - `12K-20K`: admit Q1.7 under strict gates and `10%` route-share cap.
  - `20K-24K`: decay all KD to consolidation tail.
- Exact alpha schedule:
  - `alpha_state`: `0→0.04 [500,1500]`, hold to `3K`, `→0 [3K,6K]`
  - `alpha_lfm`: `0→0.20 [500,1500]`, `→0.28 [1500,6K]`, hold to `12K`, `→0.18 [12K,20K]`, `→0.06 [20K,24K]`
  - `alpha_q06`: `0→0.08 [500,1500]`, `→0.18 [1500,6K]`, `→0.22 [6K,12K]`, `→0.16 [12K,20K]`, `→0.05 [20K,24K]`
  - `alpha_q17`: `0 [<12K]`, `0→0.06 [12K,14K]`, hold to `18K`, `→0.01 [18K,24K]`
  - `alpha_sem`: `0.01→0.02 [500,2K]`, hold to `16K`, `→0.01 [16K,24K]`
- Exact tau schedule:
  - `tau_lfm: 1.2→1.8 [500,8K]`
  - `tau_q06: 1.4→2.0 [500,8K]`
  - `tau_q17: 2.0→2.6 [12K,18K]`
- Optimizer/systems:
  - backbone lr `1e-4`, projector lr `2.5e-4`, warmup `500`, `min_lr=1e-5`, WSD final `4.8K`
  - AdamW betas `(0.9, 0.95)`, wd `0.1` backbone, `0.05` projectors
  - batch `8`, grad-accum `4`, seq `512`, precision `bf16/bf16/fp32`
- Controller:
  - audit every `256` steps; update `cka_ema` and gradient-cos EMA
  - if `gcos_ema_t < -0.10` on two audits, `alpha_t_max *= 0.8`
  - if teacher share exceeds cap and routed-loss win is not positive, multiply its score by `0.85` until next cache refresh
  - caps on KD-active spans: `LFM 0.60`, `Q0.6 0.45`, `Q1.7 0.10` before `18K`, then `0.15`

- Implementation hooks already exist around full-distribution confidence, shared-vocab logit extraction, tau scheduling, and teacher-dim caching in [dense_baseline.py:1781](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/dense_baseline.py#L1781), [dense_baseline.py:1790](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/dense_baseline.py#L1790), [dense_baseline.py:4642](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/dense_baseline.py#L4642), [dense_baseline.py:5064](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/dense_baseline.py#L5064).

- One allowed branchpoint: if the pending anchor probe finishes with `Q0.6-only` beating `LFM-only` by `>=0.02 BPT` at both `1.5K` and `3K`, swap only the token alpha schedules for LFM and Q0.6. Do not change the state surface or the Q1.7 gate.

**Confidence**
- No score increases from R2. This round is diagnostic, not yet performance-positive.
- O1 `7/10`: stronger because CKA now cleanly identifies the right anchor and the right restriction rule, but there is still no positive 197M KD result; the anchor probe is still pending ([tl_ekalavya_r3_prompt.md:68](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/tl_ekalavya_r3_prompt.md#L68)).
- O2 `8/10`: new empirical diagnosis is much better. We now know exactly why R2-style scoring fails and exactly where Q1.7 becomes unsafe ([tl_ekalavya_r3_prompt.md:35](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/tl_ekalavya_r3_prompt.md#L35), [cka_compatibility_probe.json:71](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/cka_compatibility_probe.json#L71)).
- O3 `7/10`: teacher roles are clearer and cleaner than in R2, but there is still no config-only hot-swap proof.
- O4 `8/10`: this round removed the two main blockers from R2, broken compatibility and monopoly routing, but it still did not show a committee win over control or over the best single teacher ([cka_compatibility_probe.json:49](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/cka_compatibility_probe.json#L49), [tl_ekalavya_r3_prompt.md:38](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/tl_ekalavya_r3_prompt.md#L38)).
- O5 `7/10`: the new span-difficulty structure still supports adaptive KD concentration, but this round produced no new exit-quality or latency evidence ([committee_map_audit.json:22](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/committee_map_audit.json#L22), [tl_ekalavya_r3_prompt.md:55](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/tl_ekalavya_r3_prompt.md#L55)).