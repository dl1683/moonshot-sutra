**Critical Findings**

1. The proposed verifier is learning the wrong target. In [codex_outer_loop_budget.md](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/codex_outer_loop_budget.md#L41), `q_p` is trained against `exp(-CE_p)`. That predicts current confidence, not stop-worthiness. The stop decision needs “will another pass help enough to justify cost?”, i.e. future regret or marginal gain (`CE_p - CE_final`, or `CE_p - CE_{p+1..P}`). Otherwise you freeze confidently wrong tokens and you never teach “continue because improvement is still available.”

2. The design ignores that the current memory path is mostly stop-gradient already. [launch_v054.py](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/launch_v054.py#L168), [spectrum_memory.py](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/spectrum_memory.py#L220), and [spectrum_memory.py](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/spectrum_memory.py#L229) all detach writes. Then the v0.6 sketch detaches frozen states again in [codex_outer_loop_budget.md](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/codex_outer_loop_budget.md#L171). That means later-token loss does not train earlier frozen tokens to remain useful as context. The design assumes frozen tokens become good read-only memory, but the current training path barely gives that signal.

3. `spectrum_memory` is not a drop-in causal replacement for the current scratchpad. Current scratch is prefix-causal per position in [scratchpad.py](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/scratchpad.py#L77). `ContinuousSpectrumMemory.write_scratch()` in [spectrum_memory.py](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/spectrum_memory.py#L227) writes one global sequence summary, not a prefix-causal `(B,T,S,D)` state. If v0.6 uses that scratch across passes, it leaks future tokens.

4. The current codebase cannot implement “frozen tokens stop updating” by wrapping the existing core. The live modules do not accept an `active_mask`; they still run on all tokens in [launch_v054.py](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/launch_v054.py#L124) and [sutra_v05_ssm.py](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/sutra_v05_ssm.py#L282). Also Stage 7 is not terminal in the graph: `7 -> {4,7}` in [sutra_v05_ssm.py](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/sutra_v05_ssm.py#L35). Pinning `pi` to Stage 7 is not enough unless done tokens are hard-masked before transition/router/writer.

5. Variable-pass v0.6 is out-of-distribution for multiple learned mechanisms. `lam` only grows in [sutra_v05_ssm.py](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/sutra_v05_ssm.py#L171), so tokens that exceed the trained 8-step regime become increasingly rigid and push `log_var` into unseen ranges for [spectrum_memory.py](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/spectrum_memory.py#L164). Pheromone is tied to a fixed inner-step schedule in [launch_v054.py](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/launch_v054.py#L125) and [launch_v054.py](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/launch_v054.py#L170). The switching kernel has no pass input in [sutra_v05_ssm.py](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/sutra_v05_ssm.py#L78). The design only adds pass info to verifier/scale; the core dynamics still do not know which pass they are on.

6. The proposed training interface will blow up memory if implemented literally. Storing `pass_logits` for every pass in [codex_outer_loop_budget.md](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/codex_outer_loop_budget.md#L131) and [codex_outer_loop_budget.md](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/codex_outer_loop_budget.md#L193) is too expensive at `B=8, T=512, V=50k`. You need online accumulation of per-pass CE/verify losses, not a list of full logits.

**What’s Missing / What Breaks**

- Checkpoint resume will break immediately. Training resume is strict in [train_v054.py](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/train_v054.py#L173). Optimizer resume will also break in [train_v054.py](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/train_v054.py#L206). There is no `warmstart_v060`; only `warmstart_v054` exists in [launch_v054.py](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/launch_v054.py#L197).
- Budget fairness is underspecified. Pure sequence-level budget lets one pathological token monopolize compute. The hybrid local-stipend-plus-shared-pool idea in [codex_compute_budget.md](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/codex_compute_budget.md#L4) is safer.
- The design says “budget exhausted => accept” in [codex_outer_loop_budget.md](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/codex_outer_loop_budget.md#L11). That directly reintroduces the hard-token failure mode flagged in [codex_elastic_compute.md](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/codex_elastic_compute.md#L70).
- Pass identity is missing from embeddings and transition dynamics. Current position encoding only knows token index in [launch_v054.py](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/launch_v054.py#L111).

**Simplest Viable v0.6**

Do not start with outer passes plus spectrum memory plus fail types. Minimal v0.6 should be:

1. Keep the current v0.5.4 recurrent loop as the only loop.
2. Add per-step draft logits and a verifier head inside that loop.
3. Add a hard `done` mask and stop updating done tokens.
4. Use a ceiling above 8 only after the verifier is calibrated.
5. Match compute with a simple token-step accountant first.
6. Defer spectrum memory, fail types, route-width pricing, active-set compaction, and any refill/service-rate mechanism.

So:
- `v0.6`: verifier + freezing + token-step/shared budget, same scratchpad.
- `v0.6.1`: raise ceiling above 8 at matched average compute.
- `v0.6.2`: budget-aware spectrum memory, after causality is fixed.
- `v0.6.3`: active-set compaction and true elastic wall-clock savings.

**Quick Tests Before Full Implementation**

1. Instrument current v0.5.4 to emit per-step CE on the existing 8 recurrent steps. If easy tokens do not converge materially earlier than hard tokens, freezing is not worth building.
2. Calibrate a verifier offline on existing `v054` checkpoints using current-step entropy, margin, `lam`, and correctness. If entropy/margin beat everything else and `lam` is anti-calibrated, do not center the controller on `lam`.
3. Measure stop-worthiness, not just correctness: for each token and step, compute whether future steps improve CE. That gives the right supervision target for “freeze now?”.
4. Add a causality test for `spectrum_memory` scratch. As written, it should fail because the scratch summary is global.
5. Run a warm-start compatibility probe: instantiate v0.6, partially load [step_15000.pt](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/checkpoints_v054/step_15000.pt), and verify no NaNs and unchanged outputs when verifier/freezing are disabled.

**Implementation Order**

1. Patch [launch_v054.py](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/launch_v054.py) to emit step-wise diagnostics and optional per-step logits, with freezing disabled by default.
Test: forward/backward, causality, exact parity with current outputs when diagnostics are off.

2. Patch [train_v054.py](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/train_v054.py) to train online inter-step CE plus verifier calibration loss, and log verifier AUROC / marginal-gain AUROC.
Test: 500-step smoke, resume from checkpoint, no OOM, no NaNs.

3. Patch [launch_v054.py](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/launch_v054.py) again to add `done` masking and a simple shared token-step budget. Keep scratchpad, disable spectrum memory.
Test: when verifier threshold is impossible, behavior matches baseline; when enabled, average active steps drop.

4. Add a v0.6 warm-start loader and new checkpoint namespace in [train_v054.py](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/train_v054.py).
Test: save at step 10, resume, verify identical loss trajectory for a few steps.

5. Only then touch [spectrum_memory.py](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/spectrum_memory.py). First make scratch prefix-causal; then add budget/pass inputs to `scale_net`.
Test: causality unit test, then matched-compute ablation against scratchpad baseline.

Bottom line: the v0.6 idea is directionally right, but the current design still skips the hardest parts: stop target definition, gradient path for useful freezing, causal cross-pass memory, and OOD behavior beyond 8 steps. If you build the full outer-loop version first, you will discover these mid-training. If you stage it through verifier calibration and in-loop freezing first, you can falsify the core hypothesis cheaply.