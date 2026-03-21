**Verdict**

Round 1 is better than the earlier outer-loop sketch, but it is still not implementation-ready. It fixes the wrong stop target and the literal `pass_logits` OOM issue from [codex_v060_design_review.md:1](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/codex_v060_design_review.md#L1), but it still does not solve active-mask mechanics, frozen-token training signal, or true hard-token extra-thinking. As written, it tests “better allocation inside 8 steps,” not the stronger elasticity claim from [codex_budget_tension.md:27](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/codex_budget_tension.md#L27).

1. **Holes**
- The biggest missing cost is compute, not memory. Round 1 needs draft CE from `logits_p` at every recurrent step [codex_unified_v1_proposal.md:36](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/codex_unified_v1_proposal.md#L36), but current `v0.5.4` computes full-vocab logits once after the loop [launch_v054.py:177](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/launch_v054.py#L177). Online discard fixes OOM, not the 8x draft-readout cost.
- The stage target is not identifiable. It assigns one-step CE improvement to `argmax(pi_{p+1})` [codex_unified_v1_proposal.md:44](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/codex_unified_v1_proposal.md#L44), but each step mixes kernel evolution, stage bank, routing, writing, scratchpad, and pheromone together [launch_v054.py:128](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/launch_v054.py#L128). That is not a clean action-value label.
- Freeze mechanics are still underspecified. The live model has no `active_mask` path; it updates all tokens each step [launch_v054.py:124](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/launch_v054.py#L124), and stage ops only know top-2 sparsity, not done-state sparsity [sutra_v05_ssm.py:122](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/sutra_v05_ssm.py#L122).
- Frozen tokens still are not trained to be useful context. Scratchpad writes detach now [launch_v054.py:168](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/launch_v054.py#L168), and spectrum memory detaches too [spectrum_memory.py:220](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/spectrum_memory.py#L220).
- The reroute rule conflicts with the live graph. Proposal says reroute failures to `{4,5,6}` [codex_unified_v1_proposal.md:111](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/codex_unified_v1_proposal.md#L111), but Stage 7 currently only transitions to `{4,7}` [sutra_v05_ssm.py:35](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/sutra_v05_ssm.py#L35).
- It still overweights `lam` as a feature even though production evidence says lambda is anti-calibrated for correctness [RESEARCH.md:1783](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/research/RESEARCH.md#L1783).

2. **Contradictions**
- It does address prior hole #1: stop target is now residual future gain, not present confidence.
- It does address prior hole #6: it stops storing full `pass_logits`.
- It partially addresses hole #3 by deferring spectrum memory until `write_scratch()` is made causal [codex_unified_v1_proposal.md:169](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/codex_unified_v1_proposal.md#L169).
- It does **not** address hole #2, #4, or #5 from [codex_v060_design_review.md:1](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/codex_v060_design_review.md#L1).
- It directly conflicts with the hard-token conclusion in [codex_budget_tension.md:57](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/codex_budget_tension.md#L57): Round 1 explicitly refuses a real outer continuation loop [codex_unified_v1_proposal.md:97](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/codex_unified_v1_proposal.md#L97), so it cannot test “some tokens need 12-16 serial refinements.”

3. **Feasibility**
- `v0.6.0` is feasible on one RTX 5090 if kept to residual-gain estimation, diagnostics, and no-op-compatible freezing. Current live run is on an RTX 5090 Laptop GPU with 24.46 GiB VRAM; current training is around `22.5k tok/s`, which implies about `40.5h` for `100k` optimizer steps.
- The extra `238k` params are irrelevant versus the current `67.6M` model. VRAM is not the blocker.
- The blocker is wall-clock. Per-pass logits at `B=8,T=512,V=50257` are about `205.9M` logits per step, about `0.38 GiB` bf16 transient. Doing that 8 times will hurt throughput badly.
- `v0.6.2` and `v0.6.3` are not the immediate feasibility problem. The likely failure comes earlier: noisy gain labels plus confounded action credit will cause bad freezing/routing before memory or teacher logic matter.
- The most likely thing to go wrong is premature freezing of tokens that still need the late recurrent gains that are currently strongest at steps 6-7 [RESEARCH.md:1767](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/research/RESEARCH.md#L1767).

4. **Experiments Needed Before Implementation**
- `Draft-readout cost benchmark`: add per-step draft logits to the current checkpoint and measure throughput/VRAM. Pass only if throughput drop is `<=25%` and peak VRAM stays `<18 GiB`; otherwise redesign supervision.
- `Residual-gain learnability`: offline AUROC/AUPRC for `target_resid>0.05` on a fixed 100k-token slice. Pass only if learned predictor beats simple `entropy+margin` baseline by `>=0.03 AUROC`.
- `Frozen-context regression`: force easy tokens to freeze early and measure later-token CE. Pass only if later-token CE worsens by `<0.02`; otherwise detached frozen memory is unusable.
- `12-step dense baseline`: before any controller, raise unroll to `12` with controller off. Pass only if no NaNs, `lam` stays in-range, and hard-token CE still improves materially after step 8. If not, extra-pass elasticity is dead on arrival.
- `Action attribution sanity`: on synthetic local/gist/exact-copy/extra-serial tasks, compare executed-stage labels against counterfactual best action. Pass only if agreement is `>=70%`; otherwise `g_stage` supervision is too noisy.
- `Warm-start parity`: exact no-op parity and resume parity, as Round 1 proposes. This one is mandatory.

5. **Simplification**
- Cut `g_zoom`, spectrum memory, teacher absorption, occupancy regularizers, forced exploration, and the shared/local budget split for the first pass.
- Absolute minimum test of the unified-control idea: keep the current kernel and scratchpad, add only per-step draft CE, `g_resid`, a real `done` mask, and compute accounting.
- Then test one question only: can predicted residual gain freeze easy tokens at 8 steps without hurting quality?
- If yes, the next minimal test is not stage control. It is raising the ceiling to `12/16` while keeping average compute near 8. That is the first real test of the elasticity hypothesis.

6. **Ordering**
- First: instrument current `v0.5.4` for per-step CE, throughput, and gain diagnostics with behavior unchanged.
- Second: run offline learnability and frozen-context probes on the existing checkpoint.
- Third: implement a no-op-compatible `v0.6.0` scaffold with exact parity and resume tests.
- Fourth: train `g_resid` in shadow mode only.
- Fifth: enable freeze/continue at fixed 8-step ceiling.
- Sixth: only if that works, raise the ceiling above 8 and test matched-average-compute continuation.
- Seventh: only after continuation works, consider gain-driven stage routing.
- Eighth: spectrum memory after causal fix.
- Ninth: teacher absorption last.
- Parallelizable: offline diagnostics, throughput probe, and spectrum-memory causality tests. Not parallelizable: controller implementation before those results.

7. **What Would Make Me Say No**
- If easy and hard tokens are not separable by residual-gain trajectories, stop.
- If `g_resid` cannot beat trivial entropy/margin baselines, stop.
- If freezing causes later-token regression because frozen states are bad context, stop.
- If `8 -> 12/16` continuation at matched average compute gives no hard-token win, stop. That falsifies the “extra thinking for hard tokens” thesis.
- If per-step draft supervision costs `>2x` wall-clock for negligible active-step savings, stop.
- If spectrum memory cannot be made prefix-causal, stop that branch entirely.
- If after all this the best result is just “same 8 steps, better heuristics,” then the unified-control-law story is overstated. That is not a new architecture regime; it is a controller regularizer.