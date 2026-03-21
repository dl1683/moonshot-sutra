Round 3 is written in [results/codex_unified_v1_round3.md](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/codex_unified_v1_round3.md).

The concrete fixes are:
- Replace per-pass full logits with a sampled contrastive residual-gain target: true token vs `K=32` hard negatives taken from the single final-pass readout.
- Remove `g_stage` from `v0.6.0`; stage credit is too confounded. Defer stage control to `v0.6.1` and train it only with counterfactual one-step interventions.
- Solve frozen context by freezing token dynamics but keeping a differentiable read-only prefix cache alive, so later-token loss still trains frozen states to be useful memory.
- Change the graph contract to `7 -> {4,5,6,7}` now, but keep `7 -> 5/6` zero-initialized and dormant in `v0.6.0`.
- Remove raw `lam` / `log_var` from the controller entirely. Use `mu`, `pi`, sampled margin, margin slope, `delta_mu`, pass fraction, budget fraction, and frozen-read mass instead.
- Add a real outer continuation loop by raising the ceiling to `12` passes with dense masking and a sequence budget constrained to average `8*T`.

The minimum `v0.6.0` is deliberately small:
- `12`-pass ceiling
- one `g_resid` head
- sampled-margin supervision
- `done_mask` / `active_mask`
- differentiable frozen prefix cache
- sequence-level budget penalty
- exact warm-start parity

The must-pass gates before rollout are:
- Dense-12 headroom probe: hard tokens improve materially after pass 8, easy tokens do not.
- Sampled-margin fidelity probe: cheap surrogate tracks residual CE well enough.
- Frozen-context utility probe: differentiable frozen cache does not break later-token CE.
- Warm-start parity probe: closed-gate `v0.6.0` matches `v0.5.4` exactly.

I only added the Round 3 artifact. I did not implement model code or run training/tests yet.