**1. Assumption Challenges**
1. `Projected branches are still the right 100M mainline.` Obvious interpretation: the early `+0.54` BPT gain means the family is close. Alternative interpretations: the projected subfamily is dead at 100M; only full-dim/no-post-branch-projection hybrids remain viable; or only sparse hybrid placement is viable. Distinguishing evidence: one all-hybrid `P-GQA` run at 100M plus one last projected `R7-P` sidecar. Lean: projected branches are dead as the 100M mainline. Confidence: `9/10`.
2. `Full-dim branches blow the 100M budget.` Obvious interpretation: removing `256->512` expansion makes the block too large. Alternative interpretation: GQA recovers most of that cost. Param math says a `24x512` transformer block is about `3.41M`, current projected R6 blocks are about `3.61M`, and a full-dim `P-GQA` block is about `3.54M`; total model lands around `93-94M`, which is basically the same budget class as R6-F/R6-S. Lean: full-dim is viable if GQA is kept; full-dim without GQA is not. Confidence: `9/10`.
3. `Post-fusion normalization should be the next mainline fix.` Obvious interpretation: it directly attacks fused-RMS collapse. Alternative interpretations: it only normalizes the cancellation residue while leaving branch anti-alignment intact; it may rescue short-horizon training but not long-horizon geometry. Distinguishing evidence: one `R7-P` sidecar with raw-fused RMS, post-norm RMS, and branch cosine telemetry. Lean: yes as one last projected sidecar, no as the mainline. Confidence: `7/10`.
4. `Weight norm or spectral norm on the projections is worth a round.` Obvious interpretation: cap projection growth, cap instability. Alternative interpretations: the core issue is directional divergence, not just operator norm; this is another patch on a falsified subfamily. Distinguishing evidence: only worth testing if `R7-P` almost works. Lean: do not spend a round on projection regularization. Confidence: `8/10`.
5. `Mixed schedules should come before an all-hybrid full-dim rerun.` Obvious interpretation: fewer hybrid layers reduce accumulation. Alternative interpretations: if all-hybrid full-dim is stable, mixed schedules are unnecessary; if all-hybrid full-dim still fails, mixed schedules may only hide the problem. Distinguishing evidence: run mixed schedules only after one all-hybrid `P-GQA` discriminant. Lean: mixed schedule is the fallback, not the first move. Confidence: `7/10`.
6. `100M is too small, so either ship transformer now or jump to 200M.` Obvious interpretation: three 100M failures mean the idea is dead here. Alternative interpretations: projected hybrids are dead, but full-dim hybrids are still untested at 100M; the current failure is structural, not undercapacity, so 200M is the wrong next spend. Distinguishing evidence: one 100M `P-GQA` run. Lean: do not jump to 200M with projected branches; if `P-GQA` fails, ship the transformer and freeze hybrid work at 100M. Confidence: `9/10`.

**2. Research Requests**
1. Pull primary-source evidence on sub-1B hybrids that avoid post-branch expansion entirely: full-dim hybrid heads, sparse hybrid layer placement, and whether any successful small hybrid uses post-fusion norm instead of branch norm.
2. Pull primary-source evidence on whether post-fusion norm plus direct branch addition has been enough to stabilize anti-aligned branches below 1B, or whether successful models instead avoid this geometry by design.
3. For O4, pull primary-source evidence on MiniPLM reference-size sensitivity and choose the first pooled-state teacher among `nomic-embed-text`, `EmbeddingGemma-300M`, and `Qwen3-Embedding-0.6B`.

**3. Experiment / Probe Requests**
1. Add one telemetry field before any rerun: per-layer cosine similarity between the two branch outputs. RMS growth plus fused-RMS shrink already implies divergence; cosine makes it explicit.
2. Mainline probe: `P-GQA-100M`. `24x512`, `SS-RMSNorm`, full-dim GQA attention branch, full-dim gated-conv branch, `k=4`, direct `0.5*(a+c)` fusion, no post-branch projections, exits `8/16/24`, `5K` steps. Success: `>= +0.05` BPT vs transformer with `kurtosis<=2.1`, `max_act<=52`, and no depthwise fused-RMS collapse.
3. Sidecar probe: `R7-P` using the existing post-fusion-norm block. Run `2.5K` first; extend to `5K` only if it is still ahead and telemetry shows fused contribution is not dying. If it fails, kill projected branches completely.
4. Mixed-schedule fallback only if `P-GQA` is stable but not clearly better: `8P + 16A`, with hybrids in lower layers and pure attention in upper layers. Goal: keep early hybrid benefit while preventing deep accumulation.
5. Kill rule: if `P-GQA` and `R7-P` both fail, stop hybrid work at 100M and promote the pure transformer to production.
6. O4 in parallel: run `MiniPLM top-50% vs raw` on the stable carrier, starting with the pure transformer if backbone selection is still unresolved. This advances O4 regardless of the hybrid decision.

**4. Per-Outcome Confidence**
1. Outcome 1 (Intelligence): `5/10`, down from `6/10`. New empirical evidence lowered it: `R6-F` and `R6-S` both finish behind the transformer at 5K, so the projected hybrid path lost its strongest claim.
2. Outcome 2 (Improvability): `6/10`, unchanged. New telemetry isolated the failure to the projection path, which is good evidence for diagnosability, but there is still no demonstrated surgical fix on a saved checkpoint.
3. Outcome 3 (Democratization): `5/10`, unchanged. No new composability evidence exists.
4. Outcome 4 (Data Efficiency): `4/10`, unchanged. The 500-window MiniPLM result is better evidence than the pilot, but there is still no downstream training win.
5. Outcome 5 (Inference Efficiency): `5/10`, unchanged. No new exit-ordering, PTQ, or latency evidence exists.

**5. What Would Raise Confidence**
1. O1: a stable `P-GQA` win at 100M, or a stable mixed schedule if all-hybrid is only near parity.
2. O2: a module-local repair that improves behavior without a full retrain.
3. O3: a branch or teacher plug-in that composes cleanly with an existing checkpoint.
4. O4: MiniPLM-filtered training beating raw, plus one successful representation-teacher pilot.
5. O5: exit ordering surviving PTQ with measured latency savings.

**6. What Would Lower Confidence**
1. O1: `P-GQA` and mixed schedules both fail against the transformer.
2. O2: the only way to fix hybrid instability is still full-model retraining.
3. O3: branch or teacher swaps break behavior instead of composing.
4. O4: filtered data is neutral or negative.
5. O5: exits collapse after quantization or yield no real speedup.

**7. Intuitions**
1. The real signal is `full-dim branch interaction`, not `projected hybrid cleverness`. Trigger: the only stable positive hybrid evidence is the P-block, while every projected 100M variant fails. Conviction: `high`. Validation: `P-GQA-100M`.
2. Upper layers probably want to be attention-only at 100M. Trigger: the hybrid advantage is early, while fused RMS shrinks with depth. Conviction: `medium`. Validation: `8P + 16A`.
3. `R7-P` may rescue projected branches for short training horizons but not for production-length stability. Trigger: it fixes dying fused magnitude but not the anti-alignment mechanism. Conviction: `medium`. Validation: `2.5K -> 5K` sidecar.
4. O4 will likely move the frontier more than another round of projection patches once a stable carrier exists. Trigger: MiniPLM rankings are already consistent, and the data gap versus SmolLM2 remains enormous. Conviction: `high`. Validation: filtered-vs-raw A/B.

**8. Design Proposal**
1. Abandon projected branches as the 100M mainline.
2. Keep the hybrid direction alive for exactly one discriminant: a full-dim `P-GQA` block.
3. Treat post-fusion norm as one last projected sidecar only.
4. Do not spend a round on projection weight normalization.
5. Use mixed schedules only if `P-GQA` is stable but not clearly better.
6. If `P-GQA` fails, ship the pure transformer and stop 100M hybrid iteration.

Proposed `P-GQA` block:
```text
u  = g1 * h / sqrt(mean(h^2) + eps)

a  = GQAttn(u)                                        # q: 512->256, k/v: 512->128, out: 256->512
c  = Wco(DWConv1D_k=4(Wcv u) * sigmoid(Wcg u))       # Wcv,Wcg: 512->512, depthwise conv, out: 512

r  = h + 0.5 * (a + c)

n  = g2 * r / sqrt(mean(r^2) + eps)
h' = r + Wdown(silu(Wgate n) * Wup n)
```

Use real-valued activations, `SwiGLU`, `SS-RMSNorm` everywhere, `AdamW` at `3e-4`, betas `(0.9, 0.95)`, weight decay `0.1`, `WSD`, warmup `200`, min LR `1e-5`, clip `1.0`, BF16 forward/backward, FP32 optimizer state, `24x512`, FFN `1536`, tied `16K` embeddings, exits `8/16/24`. Param budget is viable: attention `~0.393M` per block, conv `~0.788M`, FFN `~2.359M`, total `~3.541M` per block, or `~93-94M` model-wide with exits, roughly the same class as R6-F/R6-S. VRAM should stay in the same `~13-16GB` seq-512 regime as the current 100M runs.

This is the hard Round 7 recommendation: `projected hybrids are empirically dead at 100M; hybrid as a whole is not dead until full-dim GQA is tested once.` If that one test fails, the correct move is the pure transformer, not a fourth projected patch and not a 200M escalation.

Evidence anchors: [ARCHITECTURE.md:1985](C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-sutra/research/ARCHITECTURE.md#L1985), [ARCHITECTURE.md:2118](C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-sutra/research/ARCHITECTURE.md#L2118), [ARCHITECTURE.md:2130](C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-sutra/research/ARCHITECTURE.md#L2130), [RESEARCH.md:1272](C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-sutra/research/RESEARCH.md#L1272), [RESEARCH.md:1862](C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-sutra/research/RESEARCH.md#L1862), [dense_baseline.py:325](C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-sutra/code/dense_baseline.py#L325), [dense_baseline.py:377](C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-sutra/code/dense_baseline.py#L377), [dense_baseline.py:791](C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-sutra/code/dense_baseline.py#L791).