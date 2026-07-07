# Sutra & Eklavya

Sutra is a small-model intelligence research program built around one question:
can a cheap byte-native system reason well by separating reasoning geometry from
factual storage?

The current answer is sharper than where the project started. We proved that
transplanting teacher coordinates into a newborn byte model does not work as a
moonshot mechanism. That negative result forced the project toward
evidence-native judgment. The first evidence-native prototype has now failed its
own gates, so the live direction is under pressure rather than proven.

## Current Thesis

**Intelligence = reasoning geometry + factual knowledge search.**

Reasoning geometry should be compact, cheap, transferable, and trainable on
limited hardware. Factual knowledge should not be crammed into a 121M parameter
model from scratch. It should live in retrievable evidence, with the model
trained to judge, compare, and use that evidence better than dumb retrieval
baselines.

That makes the live moonshot candidate:

**Evidence-Native Retrieval-Born Sutra** - a byte-native evidence-conditioned
judge that learns how to decide from retrieved public evidence.

But the v0 implementation did not prove the claim. It lost the first gate:
retrieved evidence did not beat no-evidence, shuffled evidence, or the best
same-retriever dumb baseline. Evidence-native remains alive only as a stronger,
more demanding next test, not as a result.

## The Pivot

Brainseed v0 is dead as the mainline.

After 50 work-loop iterations and 42 question-loop iterations, every Brainseed
scorer tested worse than raw codec-only scoring. Ridge, MLP, bilinear, and
learned-cosine variants all failed to beat the codec baseline. Zero-cost chart
rescues did not change the conclusion.

That is not a project failure. It is a discovery:

> We proved that transplanting coordinates fails, but the search for reasoning
> geometry led us to evidence-native judgment - a fundamentally different
> approach to small-model intelligence.

The first evidence-native v0 run then gave the next hard lesson: adding evidence
is not enough. The model must learn judgment geometry that survives controls.

## What Still Matters

The codec is infrastructure, not the moonshot. It remains useful as a
byte-to-token addressability layer and diagnostic bridge, but it is not itself
the breakthrough claim.

Chain-init is the baseline, not the moonshot. It showed a weak positive
compatibility signal and remains the pragmatic fallback number to beat. If a
future Evidence-Native Sutra cannot beat chain-init and same-retriever controls,
it is not the mainline either.

The active moonshot claim is narrower and stronger:

**Can a 121M byte-native judge use retrieved evidence to produce a large,
control-resistant benchmark lift that cannot be explained by retrieval alone?**

## Key Results So Far

| Result | Status | Meaning |
|--------|--------|---------|
| Byte-marginal KD improved byte prediction but not downstream judgment | Failed as mainline | Better compression did not become better reasoning. |
| Brainseed v0 scorers all lost to codec-only | Dead | Coordinate extraction did not produce useful frozen judgment. |
| Chain-init compatibility probe showed weak positive signal | Baseline | Coordinate inheritance may help, but it is not the moonshot. |
| Evidence-native v0 prototype failed all gates | On life support | The first implementation did not beat no-evidence, shuffled evidence, or dumb retrieval baselines. |
| Strong evidence-native test | Open | Needs a real judge architecture, external evidence controls, geometry probes, and repeated seeds. |

## Active Tracks

| Track | Status | Role |
|-------|--------|------|
| Evidence-Native Retrieval-Born Sutra | Alive but on life support | Moonshot candidate only if the next stronger test shows control-resistant learned judgment. |
| Chain-init | Strong baseline/fallback | Establish inherited-coordinate performance to beat. |
| Codec | Infrastructure | Provide byte-native addressability and diagnostics. |
| Brainseed v0 | Dead | Preserved as negative-result science and diagnostic history. |
| S0/E1/E2 byte-KD stack | Historical infrastructure | Useful code and baselines, no longer the mainline claim. |

## Current Research Map

- [Vision](research/VISION.md) - mission and stakes.
- [Status](research/STATUS.md) - live source of truth for active/dead tracks.
- [Deep Rethink](research/DEEP_RETHINK.md) - full research history and pivot trail.
- [Supervisor Check-in 4](research/dual_loop_supervisor_checkin_4.md) - formal Brainseed death and evidence-native pivot.
- [Supervisor Check-in 5](research/dual_loop_supervisor_checkin_5.md) - Evidence-Native v0 post-mortem.
- [Experiments](experiments/EXPERIMENTS.md) - experiment ledger index.

## Success Standard

Evidence-native Sutra survives only if it beats the boring explanations:

- same-retriever dumb baselines,
- shuffled evidence,
- wrong-topic evidence,
- nearest-neighbor label shortcuts,
- no-evidence-trained controls,
- corpus leakage,
- length and formatting artifacts,
- chain-init when that baseline is ready.

The project is still aiming for a stop-scrolling result. The claim is now
strictly evidence-bounded: small models do not need to memorize the world only if
they can learn a cheap geometry for judging what they find.