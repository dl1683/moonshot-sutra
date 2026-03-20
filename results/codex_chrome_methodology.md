Add a `dim=768` canary. If you add only one new tier, do not make it `dim=384`.

Your repo already shows three bad transfers from tiny Chrome to production-like scale: LR boundary broke, recurrence-depth importance reversed, and Grokfast flipped from `+11%` at `dim=128` to divergence at `dim=768` ([RESEARCH.md:1705](C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-sutra/research/RESEARCH.md#L1705), [RESEARCH.md:1777](C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-sutra/research/RESEARCH.md#L1777), [train_v054.py:51](C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-sutra/code/train_v054.py#L51)). That means `128 -> 384 -> 768` is not the core problem. The problem is using a non-production regime as the decision-maker.

**Better Chrome methodology**
- `dim=128` stays, but only as idea triage: bug-finding, obvious loser killing, rough mechanism shaping.
- Optional `dim=384` becomes a sweep tier, not a shipping tier: use it only to narrow LR/lambda/ramp ranges before the expensive run.
- `dim=768` becomes the decision tier: exact production trainer, exact `max_steps=8`, bf16, exact warm-start path, same optimizer/schedule/guards, `200-1000` optimizer steps, always against a matched baseline.
- Any change that alters optimization dynamics, recurrence usage, or adds learned control must clear the `768` canary before you believe it.
- Any change that alters function class materially should be treated as “needs 768 canary or scratch,” not “Chrome-pass means ship.”

**Decision rule**
- Promote on `768` only if it is stable and beats matched baseline on held-out BPT/loss over the canary window.
- Use per-token win/loss, not just mean BPT, to detect “helpful on a few tokens, harmful overall.”
- For recurrence/halting changes, require extra-step utility at `768`; tiny-scale depth conclusions are not trustworthy.
- For optimizer tricks, require a stability-margin readout, not just early improvement.

**What transfers across scales**
- Implementation invariants: causality, first-forward continuity, warm-start compatibility, no obvious numerical pathologies.
- Broad mechanism class signals: simple shared state / low-entropy bias looks more transferable than learned high-entropy control. Your scratchpad result is the positive example.
- Strong negative evidence can transfer: if something is already unstable or incoherent at tiny scale, it is usually not worth promoting.

**What does not transfer reliably**
- Absolute BPT uplift from `dim=128`.
- Hyperparameter optima or stability boundaries: LR, lambda, ramp, delay.
- Recurrence-depth importance and halting thresholds.
- Single-seed 300-step rankings.
- Internal proxy metrics like entropy/lambda unless they are recalibrated at production scale.

So the answer is: add a `dim=768` canary as the required gate. Add `dim=384` only if you want a cheaper hyperparameter contouring tier, not as the predictor of production behavior.