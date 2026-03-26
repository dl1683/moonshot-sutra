**1. Assumption Challenges**
1. `k=16 won the concat sweep, so the 100M gate should use k=16.` Obvious reading: `concat_1to1_k16` beat `concat_1to1_k4` at 42M. Alternative reading: kernel preference is fusion-dependent; the healthier fusion family is mean/additive, and there `k=4` beat `k=64` by `0.15` BPT with much better kurtosis (`0.82` vs `1.45`). Distinguishing test: branch-normalized additive `k=4` vs `k=16`. Lean: use `k=4` for the gate, because `k=16` is only favored inside the raw concat family, and raw concat is not ready to promote. Confidence: `8/10`.

2. `Falcon-H1 says attention should be small, so use 2:3.` Obvious reading: conv/SSM-heavy ratios are the right prior. Alternative reading: Falcon-H1’s local branch is Mamba-2, not our weaker GatedConv; at 42M, `2:3` beat `1:1` only by adding params (`50.9M` vs `47.6M`), so the ratio effect is confounded with capacity. Distinguishing test: matched-total-param `1:1` vs `2:3`, or rerun after upgrading the local branch. Lean: `1:1` for the gate. Falcon-H1 does not override local evidence while the non-attention branch is still GatedConv. Confidence: `7/10`.

3. `42M evidence is not enough to greenlight the 100M gate.` Obvious reading: no 100M result exists yet. Alternative reading: eight 42M probes have already done their job; they isolated the live trunk family, killed several bad auxiliaries, and exposed the fusion failure mode. Distinguishing test: the 100M gate itself. Lean: greenlight the 100M gate now, but do not greenlight raw concat as currently implemented. Confidence: `8/10`.

4. `Concat just needs more steps, so the gate should be 10K instead of 5K.` Obvious reading: concat improved late and `concat_2to3_k16` nearly matched the inter-layer winner. Alternative reading: mean/additive also improved late, and concat’s defining issue is the instability window around steps `3000-4000`, not just slow learning. Distinguishing test: only extend a patched concat arm after fusion is fixed. Lean: keep the gate at `5K`; continue only the winner to `10K`. Confidence: `7/10`.

5. `Pre-fusion normalization is optional.` Obvious reading: concat is already close enough. Alternative reading: every concat arm showed the same instability signature, and Hymba’s exact recipe normalizes each branch before combining because branch magnitudes diverge. Distinguishing test: one patched concat probe. Lean: if concat stays alive at all, pre-fusion norm is mandatory. Confidence: `9/10`.

6. `The simplest parallel O4 move is to add online teacher loss during the gate.` Obvious reading: O4 is weakest, so add supervision now. Alternative reading: local evidence says extra online losses hurt before the base trunk is proven; the lowest-risk O4 work is still offline data shaping. Distinguishing test: MiniPLM pipeline pilot on a corpus slice. Lean: do offline MiniPLM in parallel, not online KD. Confidence: `9/10`.

**2. Research Requests**
1. Find any sub-500M evidence for branch-normalized additive fusion with projected/GQA branches versus concat, to validate that the R5 gate block is the right translation of Hymba-style balancing.
2. Verify the fastest CPU or quantized local path for `Qwen3-1.7B` and `Qwen3-0.6B` scoring on a `1-10%` corpus slice so the O4 pilot can run without touching the gate GPU.

**3. Experiment / Probe Requests**
1. Run the 100M promotion gate: `24x512` pure transformer vs `24x512` branch-normalized additive hybrid, `SS-RMSNorm`, plain NTP, exits `8/16/24`, `5K` steps, seq `512`. Success: hybrid wins by at least `0.10` BPT with `max_act <=` transformer.
2. If concat should remain alive, run one short sidecar: patched concat with pre-fusion branch norm, `k=16`, `1:1` vs `2:3`, `2-3K` steps. Purpose: test whether concat was failing because of missing normalization or because concat itself is the wrong fusion family.
3. Run the O4 parallel pilot: MiniPLM on `10%` of the corpus with `Qwen3-1.7B` teacher and `Qwen3-0.6B` reference, top-50% selection, score histogram, and shard-weight plumbing check. This is infrastructure validation, not a final O4 verdict.

**4. Per-Outcome Confidence**
1. Outcome 1 (Intelligence): `5/10` — unchanged. New March 26, 2026 probe data made the hybrid story tighter, but all wins are still `42M/5K`, and the gate fusion choice changed from raw concat to normalized additive.
2. Outcome 2 (Improvability): `6/10` — unchanged. The architecture remains modular by branch and by exit, but there is still no frozen-trunk proof that a module-local change improves behavior cleanly.
3. Outcome 3 (Democratization): `5/10` — unchanged. The branch/module interface is getting clearer, but contributor-style composition is still architectural intent, not evidence.
4. Outcome 4 (Data Efficiency): `3/10` — unchanged. MiniPLM is concrete, but still unvalidated locally as a real win.
5. Outcome 5 (Inference Efficiency): `5/10` — unchanged. Fixed exits remain the only live path; the new data improved trunk stability, not exit calibration or PTQ evidence.

**5. What Would Raise Confidence**
1. O1: the 100M hybrid beats the matched 100M transformer with better generations.
2. O2: a frozen-trunk branch or teacher-port change improves behavior without full retraining.
3. O3: a contributor-style branch or teacher addition composes cleanly.
4. O4: MiniPLM-filtered training beats raw-data training at equal compute.
5. O5: exit ordering survives PTQ and yields real latency savings.

**6. What Would Lower Confidence**
1. O1: the 100M hybrid loses, or only “wins” with worse activation health.
2. O2: every useful change still requires touching the whole model.
3. O3: adding or swapping a branch breaks the model badly.
4. O4: MiniPLM weights are noisy or downstream-neutral.
5. O5: exit ordering collapses after quantization.

**7. Intuitions**
1. The best 100M gate block is neither raw mean nor raw concat; it is normalized additive fusion with projected branches. Trigger: mean gave the stability signal, concat exposed the balancing failure mode. Conviction: `high`. Validation: the 100M gate plus one short 42M patched-concat sidecar.
2. `2:3` will only become clearly right after the local branch is stronger than GatedConv or after fusion is fixed. Trigger: the current `2:3` win is capacity-confounded. Conviction: `medium`. Validation: normalized `1:1` vs `2:3`.
3. The first real O4 win will come from data shaping, not online KD. Trigger: every local online auxiliary has hurt so far, while MiniPLM remains the cleanest low-risk lever. Conviction: `high`. Validation: the `10%` MiniPLM pilot.

**8. Design Proposal**
Use a new gate block now. Do not promote raw concat.

**Round 5 answers**
1. Kernel for the 100M gate: `k=4`.
2. Ratio for the 100M gate: `1:1`.
3. Is 42M evidence enough: `yes`, for a 100M gate now.
4. Should the gate be `10K`: `no`; run `5K`, then extend only the winner.
5. Add pre-fusion normalization to concat: `yes`, if concat remains alive at all.
6. Simplest O4 action in parallel: `10%` MiniPLM pipeline pilot with same-tokenizer Qwen teacher/reference.

**HEMD-R5-G**
- `24` blocks, hidden dim `512`, FFN dim `1536`, tied `16K` embeddings, exits after blocks `8/16/24`, seq `512` for the gate.
- Real-valued activations, `SS-RMSNorm`, `SwiGLU`, BF16 forward/backward, FP32 optimizer state.
- Optimizer: `AdamW`, lr `3e-4`, betas `(0.9, 0.95)`, wd `0.1`, `WSD`, warmup `200`, min lr `1e-5`, clip `1.0`.

Block equations:
```text
u  = g1 * h / sqrt(mean(h^2) + eps)

a0 = GQAttn(u)                                          # 4 Q heads, 2 KV heads, head_dim=64, raw dim 256
c0 = DWConv1D_k=4(Wv u) * sigmoid(Wg u)                # raw dim 256

a  = beta_a ⊙ Norm_a(Wao a0)                           # 256 -> 512
c  = beta_c ⊙ Norm_c(Wco c0)                           # 256 -> 512

m  = Wo(0.5 * (a + c))
r  = h + m

n  = g2 * r / sqrt(mean(r^2) + eps)
h' = r + Wdown(silu(Wgate n) * Wup n)
```

- `Norm_a` and `Norm_c` are independent branch norms.
- `beta_a, beta_c ∈ R^512`, initialized to `1`.
- Residual-output projections use scaled init `0.02 / sqrt(2L)`.

Approximate parameter count for the gate:
- Embeddings: `~8.2M`
- Per block: attention path `~0.46M`, conv path `~0.33M`, output projection `~0.26M`, FFN `~2.36M`, norms/betas negligible
- Total: `~90M`

VRAM expectation:
- seq `512`, microbatch `16`, BF16: roughly `12-14 GB`
- seq `1024`: defer until after the gate

Why this is the right move: it keeps the hybrid direction that now has the strongest local evidence, keeps the `k=4` local branch that won in the stable fusion family, rejects the raw concat failure mode instead of hoping longer training fixes it, and advances O4 in parallel through offline data shaping rather than risky online losses.