The final buildable spec is in [results/codex_v060a_buildable_spec.md](C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/codex_v060a_buildable_spec.md).

The key decision is: `v0.6.0a` is **probe-only**, not a controller. It builds a new `12`-pass from-scratch training path on top of the `v0.5.4` core, with:
- `code/launch_v060a.py`
- `code/train_v060a.py`
- one shadow `ResidualGainProbe`
- one optional `FrozenPrefixCache` used only for forced-freeze ablations
- sampled inter-step supervision using `K=32` negatives from the final-pass readout

It explicitly excludes stage control, budget control, graph changes, spectrum memory, teacher routing, and weight warm-start. The proceed gates are also concrete: only build the full controller if dense-12 training is stable after pass 8, easy/hard tokens separate by residual-after-8, the sampled target tracks full CE well enough, the probe beats margin-only baselines, and forced freeze remains safe with the cache. Estimated implementation time is `8-10` hours. Warm-start answer: **reuse the `v0.5.4` code, but train `v0.6.0a` weights from scratch**.