Grounded in [master_research_synthesis.md](C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-sutra/results/master_research_synthesis.md), [codex_warmstart_fix.md](C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-sutra/results/codex_warmstart_fix.md), [chrome_gated_warmstart.json](C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-sutra/results/chrome_gated_warmstart.json), [launch_v054.py](C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-sutra/code/launch_v054.py), [train_v054.py](C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-sutra/code/train_v054.py), and [RESEARCH.md](C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-sutra/research/RESEARCH.md).

1. Next Chrome should focus on mechanisms that preserve the current topology and strengthen late-step routing/state, because that is what already survives scale.
- First choice: `spatial coupling` as a stronger version of scratchpad-to-router coupling. The math matches what is already working: weak local processors become strong when boundary/global signals propagate inward.
- Second choice: `BP/LDPC-style reliability-weighted message passing`. That means damping, confidence-weighted routing, or packetized sparse messages inside the existing router, not a new controller.
- Third choice: `contraction-aware recurrence`. Use contraction as a design constraint or regularizer on the recurrent core, not as a new module.
- Defer to `dim=1024 scratch-only`: `nGPT`, `wave-PDE routing`, `tropical routing`, `unitary/complex transitions`, `NCA pre-pretraining`. Those are still good bets, but they are geometry swaps, not safe warm-start refinements.

2. Chrome methodology needs a `dim=768 mini-Chrome`. I would not trust a scaling law from `128 -> 768` yet.
- The repo already shows three broken transfers from small scale: LR boundary, recurrence-depth importance, and Grokfast.
- Keep `dim=128` for bug-finding and cheap ranking only.
- Add a `dim=768 canary` with the exact production trainer, exact `max_steps=8`, bf16, warm-start path if relevant, and `200-1000` optimizer steps.
- Gate every candidate on:
  - first-forward continuity for warm-start,
  - no-NaN / no-loss-jump stability,
  - matched eval improvement over the live `v0.5.3` baseline,
  - preserved value of late recurrent steps.
- If you want a middle tier, use `dim=384/512`, but `768` has to be the final decision gate.

3. The critical path to beating Pythia-70M is mostly eval hygiene + data + one more routing/state win, not more speculative architecture.
- Important metric clarification: on `2026-03-20`, [v053_metrics.json](C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-sutra/results/v053_metrics.json) shows `5.9579` BPT at step `5000`, while [clean_benchmark_pythia.json](C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-sutra/results/clean_benchmark_pythia.json) shows Pythia-70M at `3.5592` BPT on a different clean benchmark. Those are not the same eval.
- So the first job is to lock a single matched benchmark. Otherwise you can cross an internal line without having a publishable claim.
- On the current production track, the fastest path is:
  - keep `v0.5.3` behavior continuous,
  - exploit delayed pheromone / late-bias refinements,
  - finish FineWeb and move to better data,
  - avoid any geometry-changing trick until scratch training at larger scale.
- My base-rate view: better data and longer stable training will move you more than another fancy `dim=128` Chrome win.

4. Scale to `dim=1024` on FineWeb once infrastructure is ready, not once theory is exhausted.
- Required before the jump:
  - FineWeb tokenization/streaming is restart-safe and complete,
  - a `dim=768` canary pipeline exists,
  - the `v0.5.4` line is stable for at least one real continuation block,
  - matched eval vs baseline is locked,
  - NaN/stability gates are automated.
- I would run `dim=1024` as a `scratch` program on FineWeb, not as another warm-start mutation of `768`.
- At `1024`, test only one new major idea per run. Best order: baseline `v0.5.4-scratch`, then `+NCA pre-pretraining` or `+nGPT`, not both together.

5. Architectural changes that can be added to a running model without a warm-start are a very small class.
- Safe: parameter-free or state-only changes inside existing routing/state dynamics.
- Examples:
  - better pheromone deposit/read rules,
  - local-excitation/global-inhibition bias on routing scores,
  - message norm caps or contraction-style damping,
  - retrieval `k` / temperature / damping schedule changes.
- Not safe mid-run:
  - new norms,
  - new projections,
  - new memory banks,
  - geometry changes,
  - controller changes.
- In practice: if it adds learned parameters or changes the function class, treat it as requiring warm-start or scratch.

My blunt recommendation: stop asking `dim=128` Chrome to predict production winners. Use it to find ideas; use `dim=768` mini-Chrome to make decisions. The immediate research frontier is not “more novelty,” it is “late-step communication that stays low-entropy at scale.”