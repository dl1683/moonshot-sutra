**Phase A**
1. `Warm-start from the March 27, 2026 control checkpoint?`
For: `197M@60K` already reached `BPT=3.5726`; throwing that basin away is irrational on one GPU. Against: KD changes the objective, so a restart could co-adapt router + ports more cleanly. Resolve with a `1K` continue-control vs continue-KD stability audit. Lean: warm-start from `control_197m_60k_step60000.pt`. Confidence `9/10`.

2. `Should Qwen3-1.7B be the anchor teacher?`
For: strongest reasoning signal. Against: local failures were ratio-driven at `1:19`; literature and repo synthesis say teacher size past the comfort zone hurts, and `197M:0.6B` is `~1:3`, while `197M:1.7B` is `~1:8.6`. Resolve with equal-FLOP `Qwen3-0.6B` vs `Qwen3-1.7B` anchor continuation. Lean: `Qwen3-0.6B` anchor, `Qwen3-1.7B` delayed specialist. Confidence `9/10`.

3. `Should all teachers supervise the same sample/span at once?`
For: maximum information per step. Against: knowledge purification papers and local rep+logit ablations both say naive combination creates conflict; your own 90M run showed simultaneous surfaces become toxic when NTP weakens. Resolve with routed vs averaged committee ablation. Lean: all teachers audit every span, but only one generative teacher supervises a given span. Confidence `9/10`.

4. `Should routing be per-sample?`
For: simpler and matches most MTKD papers. Against: your bridge is already `16` byte spans; samples contain mixed content; per-sample routing wastes local specialists. Resolve with per-sample vs per-span routing at matched teacher FLOPs. Lean: per-span routing on the byte-span scaffold. Confidence `8/10`.

5. `Should representation and logit KD run together?`
For: DSKD-style dual-space transfer can work in literature. Against: local evidence is explicit: rep-only is head-start, logit-only at flat alpha is harmful, rep+logit is interference. Resolve with temporal-separation vs simultaneous-surface ablation. Lean: temporal separation only; state early, logit main, semantic thin tail. Confidence `9/10`.

6. `Is final-layer-only KD enough?`
For: layer-selection literature says not to over-engineer. Against: your student has natural exits at `7/15/23`, current code only uses final teacher hidden states, and multi-depth transfer is one of the explicit missing pieces. Resolve with tiny multi-depth primer vs final-only. Lean: multi-depth state primer at relative depths `1/3, 2/3, 1.0`, but low-weight and early-only. Confidence `7/10`.

7. `Should Byte-Span Bridge be replaced by OT/ALM?`
For: OT/ALM are more principled. Against: byte-span is already implemented, tokenizer-agnostic, and uniquely supports span routing. Resolve with GW-for-routing while keeping byte spans fixed. Lean: keep byte spans as the universal scaffold; add GW only to score compatibility. Confidence `8/10`.

8. `Do we need full PCGrad/GCond every step?`
For: explicit conflict resolution is literature-supported. Against: repeated extra backwards are expensive on one 5090, and routing should remove most conflicts before optimizer-level surgery. Resolve with periodic conflict-audit vs no-controller ablation. Lean: periodic gradient-conflict audit that adjusts alphas; only escalate to full GCond if needed. Confidence `7/10`.

**Phase B**
Research requests:
1. Work out the cheapest entropic `GW/Sinkhorn` implementation for `16x16` span graphs in PyTorch and whether `8-10` iterations are enough.
2. Pull exact `TinyLLM`, `TAID`, and `TCS` mechanics for extreme-ratio multi-teacher and intermediate-teacher scaffolding.
3. Study neuroscience analogies for routing: thalamic gating, basal-ganglia action selection, dendritic compartmentalization.
4. Study biology/ecology analogies: clonal selection, immune suppression, niche partitioning, quorum sensing.
5. Study physics analogies: renormalization/successive refinement, spin-glass disagreement, barycenter consensus under noisy experts.

Probe requests:
1. `Audit-only committee map`: run the `60K` student and all four teachers on `2K` windows, log span-level agreement, novelty, compatibility, and target route shares.
2. `Routing granularity ablation`: `3K` continuation from `60K` with `control`, `Q0.6 anchor-only`, `committee per-sample`, `committee per-span`.
3. `Surface separation ablation`: `3K` continuation comparing `logit-only main + tiny early state primer` against `simultaneous state+logit`.
4. `Conflict controller ablation`: routed committee with and without periodic cosine-audit alpha control.
5. `Q1.7 admission gate`: after `Q0.6+LFM` is non-negative, run a `2K` sidecar where `Q1.7` enters only on hard spans, comparing `forward KL` vs `AMiD`.

**Confidence**
`O1 6/10` — the backbone is now strong enough to matter (`197M@60K = 3.5726 BPT`), and the design keeps token-level supervision dominant, but there is still no positive 197M routed-KD result.  
`O2 8/10` — ports, span router, teacher-share telemetry, and conflict audits make failure localization much cleaner than the current equal-average KD loop.  
`O3 7/10` — a universal teacher interface plus fixed teacher roles is composable, but packaging and contributor tooling do not exist yet.  
`O4 7/10` — the tokenizer blocker is solved, `Q0.6` sits in the safe ratio zone, and multi-teacher purification is well-supported, but the repo still lacks a committee win.  
`O5 7/10` — inference stays unchanged because the protocol is train-time only, and exit self-distillation remains intact, but no new exit-quality data exists yet.

**Phase C**
Ekalavya should be `simultaneous across the window, exclusive at the span`.

Core system:
- Teachers: `Qwen3-0.6B` = default generative anchor, `LFM2.5-1.2B` = structural specialist, `Qwen3-1.7B` = delayed hard-span specialist, `EmbeddingGemma-300M` = always-on semantic anchor.
- Granularity: split each window into `M=16` byte spans and route at span level.
- Student exits: `7/15/23`. Teacher depth maps: `Qwen {9,18,28}`, `LFM {5,11,16}`, `EmbeddingGemma {24}`.
- New trainable modules: state hub `d=384`, semantic hub `d=256`, analytic router with learned teacher priors, conflict controller. New params are about `7.2M`, well under `4%` overhead.

Math:
```text
z_s^(r,m) = LN(W_s^r S_s^(r,m)),   W_s^r: 768 -> 384
z_t^(r,m) = LN(W_t^r T_t^(phi_t(r),m)),   W_t^r: D_t -> 384

compat_t = exp(-4 * GW(G_s^15, G_t^15))
conf_(m,t) = mean_i max p_t(i), over hardest 25% student-entropy tokens in span m
gap_(m,t)  = mean_i KL(stopgrad(p_t(i)) || p_s(i))
novel_(m,t)= mean_u!=t JS(p_t(i), p_u(i))

score_(m,t)= pi_t * compat_t * conf_(m,t) * gap_(m,t)^0.75 * (0.25 + novel_(m,t))^0.5
```

Routing rule:
- If span disagreement is below the rolling `P30`, use `consensus`.
- Else if top score margin is above rolling `P60` and compatibility is above rolling `P50`, route to that specialist.
- Else if no specialist wins, default to `Qwen3-0.6B`.
- Else use `null` and train CE-only on that span.
- Maintain target span shares by controller: `Q0.6 40-60%`, `LFM 15-25%`, `Q1.7 10-20%`, `consensus 15-25%`, `null <15%`.

Losses:
```text
L = L_ntp + 0.20 L_exit
  + alpha_state(u) * L_state
  + alpha_sem(u)   * L_sem
  + alpha_06(u)    * L_tok^q06
  + alpha_lfm(u)   * L_tok^lfm
  + alpha_17(u)    * L_tok^q17
```

```text
L_state = sum_{r in {7,15,23}} sum_{m} rho_(m,t) * lambda_r * (1 - cos(z_s^(r,m), z_t^(r,m)))
lambda = {0.2, 0.3, 0.5}

L_sem = 1 - cos(u_s, u_eg) + 0.1 * ||K_s - K_eg||_F^2 / B^2
```

```text
Qwen3-0.6B, LFM: forward KL on routed spans
Qwen3-1.7B: AMiD on routed hard spans only, alpha=-3, lambda=0.2
Consensus spans: weighted teacher mixture on aligned support, weights proportional to score_(m,t)
```

Schedule for Round 1 continuation from `control_197m_60k_step60000.pt`:
- Optimizer: `AdamW`.
- LR: student `1e-4`, ports/router `3e-4`, `min_lr=1e-5`, `warmup=500`, WSD in last `20%`.
- Precision: student/ports bf16, optimizer fp32, teachers fp16 no-grad.
- Duration: `24K` continuation, not another blind `60K`.

Alpha/temperature curriculum:
- `alpha_state`: `0 -> 0.04` by `500`, hold to `2K`, off by `4K`.
- `alpha_sem`: `0.01 -> 0.03` by `2K`, hold to `18K`, tail `0.01`.
- `alpha_q06`: `0 -> 0.20` by `500`, `0.45` by `2K`, hold to `16K`, `0.10` by `20K`, tail `0.05`.
- `alpha_lfm`: off until `1K`, `0.18` by `4K`, hold to `16K`, tail `0.02`.
- `alpha_q17`: off until `4K`, and only enabled if `Q0.6+LFM` is non-negative on the audit stream; peak `0.12`, off by `20K`.
- `tau_q06: 1.2 -> 2.2`, `tau_lfm: 1.4 -> 2.4`, `tau_q17: 1.8 -> 3.0`.

Conflict resolution:
- Every `250` steps, run one audit microbatch and compute gradient cosine EMA on `{base, token, state, semantic}` buckets.
- If `cos(base,bucket) < -0.10`, multiply that bucket’s max alpha by `0.8`.
- If `cos(base,bucket) > 0.05` and the bucket improved its routed audit loss, multiply by `1.05` up to cap.
- If noisy-span share exceeds `25%` twice in a row, freeze `Q1.7` and halve `LFM`.
- This is the thalamus/immune-system hybrid: select first, suppress conflict second, never average contradictions by default.

Net: the protocol should not be “all teachers, all losses, all the time.” It should be `anchor-led, span-routed, depth-matched, surface-separated, and conflict-audited`. That is the version most consistent with both the literature and your own failures.