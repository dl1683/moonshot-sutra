**1. Assumption Challenges**
1. `The 100M gate means HEMD-R5-G is ready for production.` Obvious interpretation: it beat the transformer by `+0.11` BPT at 5K, so promote it. Alternative interpretations: the result selects the hybrid family, not the exact fusion block; the `+3.3M` param gap helped somewhat; the late instability means the win may not survive long training. Distinguishing evidence: one stabilized 100M hybrid rerun plus a short 8-10K continuation. Lean: promote the hybrid direction, not R5-G as-is. Confidence: `9/10`.

2. `The per-channel β vectors are the main instability source.` Obvious interpretation: the branches are normalized, then β reintroduces unbounded scale across 24 layers. Alternative interpretations: the branch `RMSNorm` weights are a second hidden per-channel gain vector; the `256→512` projections are the real source; the issue is every-layer hybrid compounding rather than β specifically. Distinguishing evidence: log β max/mean by layer, log branch norm weight growth, compare `N-noβ` against `N-softmax` and a projected mean-fusion P-block. Lean: the real problem is stacked per-channel scale freedom, with β the first thing to remove. Confidence: `7/10`.

3. `The best first fix is clamp or softmax, not deletion.` Obvious interpretation: bounded learned mixing preserves flexibility. Alternative interpretations: the only locally stable family was fixed-scale mean fusion; clamp is an arbitrary patch; softmax bounds relative mixing but not all absolute scale sources. Distinguishing evidence: `N-noβ` vs `N-softmax` at 100M. Lean: first rerun should remove β entirely; bounded softmax is the backup, not the first move. Confidence: `8/10`.

4. `Projection itself is the stability problem.` Obvious interpretation: the stable 42M P-block used full-dim branches, while the 100M N-block projects `256→512`. Alternative interpretations: projection is fine if scale is controlled; the real failure is projected branches plus unconstrained gains; depth, not projection, is the amplifier. Distinguishing evidence: a projected mean-fusion `P-GQA` probe with no β. Lean: test `P-GQA`, but do not abandon asymmetric branches yet. Confidence: `6/10`.

5. `A mixed N/A schedule should be the next move.` Obvious interpretation: fewer hybrid layers means less accumulation of scale drift. Alternative interpretations: this hides the root cause; it may throw away the very mechanism producing the BPT win; interleaving is a second-order optimization after fusion is fixed. Distinguishing evidence: only relevant if a fixed-scale all-N rerun is still unstable. Lean: defer mixed schedules until after one clean stabilization rerun. Confidence: `7/10`.

6. `O4 should wait until architecture is locked.` Obvious interpretation: architecture first, data later. Alternative interpretations: MiniPLM scoring is already operational and CPU-parallel; delaying it wastes idle throughput; a filtered corpus can be ready by the time the stabilized hybrid is promoted. Distinguishing evidence: none needed beyond the completed pilot. Lean: start full corpus scoring now in parallel; do not train a custom 100M Qwen reference first. Confidence: `9/10`.

7. `We should keep iterating the gate until confidence is 9/10.` Obvious interpretation: production should wait for near-certainty. Alternative interpretations: that is too expensive for a single-GPU workflow; the repo’s own rule is that one targeted rerun is cheaper than one failed production restart; 7/10 is the right production threshold for O1. Distinguishing evidence: stable 100M rerun plus short continuation. Lean: stop gate iteration at `O1 = 7/10`, not `9/10`. Confidence: `8/10`.

**2. Research Requests**
1. Pull primary-source evidence on sub-1B hybrid fusion stabilization: branch-normalized addition, convex branch mixing, scalar-only branch norms, and post-fusion norm. The question is not “what is common,” but “what is actually stable below 1B.”
2. Pull primary-source evidence on MiniPLM reference-size sensitivity. The key unresolved question is whether the current `Qwen3-0.6B` same-tokenizer reference is good enough for first-pass corpus scoring, or whether a custom ~100M Qwen-tokenizer reference is worth the detour.

**3. Experiment / Probe Requests**
1. Add telemetry to the hybrid block before any rerun: per-layer `beta_max`, `beta_mean`, branch norm weight max, branch RMS pre/post norm, and fused pre-`W_out` RMS at every eval. This directly tests the β hypothesis.
2. Run a 100M stabilization mini-gate with the same `24×512`, `SS-RMSNorm`, plain NTP, exits `8/16/24`, seq `512`, `5K` steps:
   `R6-F`: remove β entirely, use branch `SS-RMSNorm`, keep projected branches and `0.5*(a+c)`.
   `R6-S`: same block, but replace β with per-channel softmax mixing so `α_a + α_c = 1`.
   Success criterion: retain at least half the current BPT gain (`>= +0.05` vs transformer) while ending with `kurtosis_max <= 2.1` and `max_act <= 52`.
3. Run a `P-GQA` discriminant probe to answer the projection question: GQA attention `256`, conv `256`, both projected to `512`, mean fusion, no β, branch `SS-RMSNorm`. If this is stable, projection is not the main problem.
4. Continue the winning stabilized 100M arm to `8-10K` steps. The current failure window peaked at step `4500`; the next promotion test must prove that the late spike does not recur.
5. Start full-corpus MiniPLM scoring now, in parallel with the stabilization work. Output shard/window weights and a top-50% manifest for a later filtered-vs-raw A/B.

**4. Per-Outcome Confidence**
1. Outcome 1 (Intelligence): `6/10` — up from `5/10`. New empirical evidence exists: at 100M scale the hybrid beats the pure transformer by `+0.11` BPT. It does not go higher because stability fails and generation quality is still nondiagnostic.
2. Outcome 2 (Improvability): `6/10` — unchanged. The architecture still has clear branch/exit boundaries, but there is still no frozen-trunk module-isolation proof.
3. Outcome 3 (Democratization): `5/10` — unchanged. The modular story remains architectural intent, not demonstrated composability.
4. Outcome 4 (Data Efficiency): `4/10` — up from `3/10`. New empirical evidence exists: the MiniPLM pilot ran end-to-end on real corpus windows and produced sensible spread and source ranking. It stays low because no downstream training win exists yet.
5. Outcome 5 (Inference Efficiency): `5/10` — unchanged. Fixed exits remain the only live path, and Round 6 produced no new exit-ordering, PTQ, or latency evidence.

**5. What Would Raise Confidence**
1. O1: a stabilized 100M hybrid wins at 5K and stays healthy through 8-10K.
2. O2: a branch-local change improves behavior without retraining the whole trunk.
3. O3: a new branch or teacher port composes cleanly with an existing checkpoint.
4. O4: filtered-data training beats raw-data training at equal compute.
5. O5: exit ordering survives PTQ and yields measured latency savings.

**6. What Would Lower Confidence**
1. O1: every stabilized hybrid loses most of the BPT gain or still blows up late.
2. O2: useful fixes still require full-model retraining.
3. O3: branch swaps break the model instead of composing.
4. O4: MiniPLM scoring produces noisy weights and filtered training is neutral or worse.
5. O5: exits collapse under quantization or provide no real speedup.

**7. Intuitions**
1. The hybrid family is correct and the failure is optimization-level, not architectural. Trigger: the 100M hybrid learns much faster early, then degrades late. Conviction: `high`. Validation: one stabilized rerun.
2. The BPT gain is coming from local+global complementarity, not from β. Trigger: the gain appears before the instability explodes. Conviction: `medium-high`. Validation: `R6-F` should keep most of the gain.
3. Gutenberg/Wikipedia-heavy filtered data will help the first production run more than WildChat-heavy data. Trigger: the MiniPLM pilot ranked literature and factual text highest, chat lowest. Conviction: `medium`. Validation: filtered-vs-raw A/B.

**8. Design Proposal**
Direct answers to your five questions:

1. `Q1`: choose `c`, but narrowly. Do not start production on `R5-G` as-is. Do not rerun an open-ended transformer-vs-hybrid gate either. Treat Round 6 as having selected the hybrid family, then run one stabilization mini-gate inside that family.
2. `Q2`: best first fix is `d`: remove β entirely. If allowed one companion change, also do `e`: use branch `SS-RMSNorm` instead of full `RMSNorm`. Backup choice is `b` softmax mixing. Do not start with clamp.
3. `Q3`: projection could be part of the problem, but it is not yet proven. Yes, test a `P-block with GQA` variant; that is the clean discriminator.
4. `Q4`: stop iterating and commit to production when `O1 = 7/10`, defined concretely as:
   a stabilized 100M hybrid win at 5K;
   stability within roughly `1.25x` transformer on kurtosis/max-act;
   no late divergence in an 8-10K continuation.
5. `Q5`: choose `b` with a caveat. Start full corpus scoring now in parallel with the stabilization rerun. Do not train a custom 100M Qwen-tokenizer reference first.

Production candidate block, pending the mini-gate:

```text
u  = g1 * h / sqrt(mean(h^2) + eps)

a0 = GQAttn(u)                                        # d_attn=256, 4Q/2KV, head_dim=64
c0 = DWConv1D_k=4(Wv u) * sigmoid(Wg u)              # d_conv=256

a  = sa * AttnProj(a0) / rms(AttnProj(a0))           # scalar SS-RMSNorm only
c  = sc * ConvProj(c0) / rms(ConvProj(c0))           # scalar SS-RMSNorm only

m  = Wout(0.5 * (a + c))
r  = h + m

n  = g2 * r / sqrt(mean(r^2) + eps)
h' = r + Wdown(silu(Wgate n) * Wup n)
```

Use real-valued activations, `SwiGLU`, `AdamW`, lr `3e-4`, betas `(0.9, 0.95)`, wd `0.1`, `WSD`, warmup `200`, min lr `1e-5`, clip `1.0`, BF16 forward/backward, FP32 optimizer state. Keep `24×512`, FFN `1536`, tied `16K` embeddings, exits `8/16/24`. Parameter count and VRAM stay effectively the same as the current 100M gate.

This is the right next move because it preserves the only thing Round 6 proved strongly — the hybrid family learns more efficiently — while removing the most likely source of late-training scale drift before you spend production-time compute.