**Verdict**

1. `High`: Round 3 still does not *solve* convergence. The failed probe says easy and hard tokens are still indistinguishable at step 4, `5.1%` vs `5.4%` done, with the real jump only at `7 -> 8` [experiment_convergence.json#L4](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/experiment_convergence.json#L4) [experiment_convergence.json#L7](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/experiment_convergence.json#L7). Raising the ceiling to 12 and adding residual supervision is a *falsification harness*, not a guarantee of separation [codex_unified_v1_round3.md#L193](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/codex_unified_v1_round3.md#L193) [codex_unified_v1_round3.md#L264](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/codex_unified_v1_round3.md#L264). Without headroom after pass 8 plus budget pressure plus forced-freeze curriculum, the model can still converge uniformly, just over 12 passes.

2. `High`: The differentiable frozen prefix cache is plausible, but the spec is still missing the only implementation detail that matters: it must be `O(B*T*d_kv)`, not scratchpad-style `O(B*T*T*d_kv)`. Current memory paths are detached today [launch_v054.py#L168](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/launch_v054.py#L168) [spectrum_memory.py#L220](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/spectrum_memory.py#L220). If you store compressed `192`-dim K/V per frozen token, K+V is only about `3 MB` bf16 at `B=8,T=512`. If you materialize a causal per-position cache like `(B,T,T,192)`, K+V is about `1.6 GB` before autograd, which is not viable. The spec needs to forbid the `T^2` version explicitly.

3. `High`: Warm-start parity and the Stage-7 graph change currently conflict. Round 3 wants `7 -> {4,5,6,7}` [codex_unified_v2_fixes.md#L7](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/codex_unified_v2_fixes.md#L7), but the transition kernel already produces all `7x7` logits and relies on the graph mask to kill illegal edges [sutra_v05_ssm.py#L27](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/sutra_v05_ssm.py#L27) [sutra_v05_ssm.py#L95](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/sutra_v05_ssm.py#L95). Unmasking `7 -> 5/6` will expose learned nonzero logits from the warm-start checkpoint. That is not “zero-initialized and dormant” unless you add an explicit clamp/bias/gate for those edges.

4. `Medium`: “Matched compute” is still overstated. Dense masking is fine for semantics [codex_unified_v1_round3.md#L193](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/codex_unified_v1_round3.md#L193), but the current loop remains fully dense matmuls over all tokens each pass [launch_v054.py#L124](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/launch_v054.py#L124). Token-pass accounting is not real FLOP reduction unless inactive tokens are actually compacted or skipped. So `v0.6.0` can test *quality allocation*, not yet true efficiency.

5. `Medium`: The sampled contrastive target is a decent ranking proxy, not a faithful CE surrogate in general. It is acceptable only if the final-pass confuser set is stable across passes and most denominator mass sits in those negatives [codex_unified_v1_round3.md#L15](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/codex_unified_v1_round3.md#L15) [codex_unified_v1_round3.md#L280](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/codex_unified_v1_round3.md#L280). It breaks when early-pass confusion is diffuse, the top wrong tokens change across passes, or many medium-probability negatives dominate CE.

**Direct Answers**

1. No. `12` passes plus inter-step loss does not by itself solve convergence. It only creates the possibility of separation. The base-case is still uniform convergence unless hard tokens show real `8 -> 12` headroom and easy tokens can be frozen early without harming later-token loss.

2. The frozen cache can work. Gradient path is: later-token loss -> frozen-cache read attention -> frozen `k^F,v^F` -> freeze-pass `mu`. That gives frozen tokens memory utility without further state updates. But it is only buildable if the cache is streamed/prefix-scanned, not materialized as a `T^2` causal tensor.

3. The `K=32` sampled target is good enough for stop/continue *ranking* if the fidelity probe passes. It is not a faithful residual CE estimator in the unbiased sense. Expect failure on distributional shifts across passes and on broad uncertainty states.

4. The minimum is smaller, but not yet small enough to build confidently as one monolithic `v0.6.0`. It is small enough to build as `v0.6.0a = probes + parity scaffold`. The missing pieces are: parity-safe graph migration, exact frozen-cache implementation, and honest compute accounting.

5. First thing to build and test: a controller-off dense-12 instrumentation branch on the current core. Measure real per-pass CE trajectories on a validation slice, `8 -> 12` headroom by difficulty bucket, negative-set drift across passes, and a forced-freeze cache ablation. If those fail, stop before adding `g_resid`.

6. Remaining holes:
- Graph change vs warm-start parity.
- Token-pass budget vs actual compute.
- Frozen-cache implementation complexity is not specified tightly enough.
- The spec still assumes inter-step supervision will *create* separation, when it may only *measure* uniformly slow convergence.
- Current production evidence says late recurrence is globally important [RESEARCH.md#L1767](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/research/RESEARCH.md#L1767), so premature freezing remains the most likely failure mode.

If you want a buildable cut, I would redefine the next milestone as: `probe-only v0.6.0a`, not “minimum elastic controller.”