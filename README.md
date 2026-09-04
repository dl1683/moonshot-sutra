# Moonshot Sutra

> There are billions of dollars of accumulated knowledge locked in pretrained
> models — language understanding, visual reasoning, semantic structure — that
> gets thrown away every time someone builds a new system. What if new, smaller
> models could *inherit* that knowledge instead of starting from scratch?

**Eklavya** is the governing idea: a framework for knowledge inheritance from
pretrained models. Not a specific technique — the thesis that accumulated
knowledge in existing models can be unlocked, transferred, and reused so that
AI development becomes *building on what exists* rather than *rebuilding from
nothing*. If this works, a person with one GPU benefits from knowledge inside
models they could never afford to train.

**Sutra** is the current exploration ground, testing Eklavya across text,
vision, and audio embedding models.

## Five Sacred Outcomes (fixed points)

1. **Genuine Intelligence** — actually capable, not merely large or benchmark-shaped
2. **Improvability** — failures found, understood, repaired surgically
3. **Democratized Development** — reproducible, modifiable, extendable by anyone
4. **Data Efficiency** — learns more from less through better structure
5. **Inference Efficiency** — cheap to run, deployable widely

Everything else — including specific mechanisms — is replaceable.

## Current Status

**Embedding reboot** (September 2026): Building compact text embedding models
(149M student, 384-dim) that inherit from larger teachers (BGE-large, 335M).
Training on real retrieval data with contrastive and KD losses.

The research question is not *whether* knowledge transfer works (it does —
distillation is well-established) but *what form of transfer unlocks the most
value*. Teacher tomography (15 mechanism kills) and per-teacher indexed heads
have been ruled out. The open question is what mechanism best captures the
knowledge that makes large models good — their output distributions, their
sense of what's hard, their internal representations, or something else.

A shipping pipeline trains, exports, and evaluates models independent of the
mechanism research — the artifact outranks the measurement (§2.7).

The durable methodology from the earlier 14-kill search remains: precommitted
terminal tokens, equal-information baselines, absorption ladders, and strict
claim ceilings.

## Substrate-Open Search

Neural networks are a candidate substrate, not the assumed answer. The right
system may be neural, symbolic, algebraic, verifier-first, program-synthesis
based, energy/search based, physics-inspired, hybrid, or something not yet
named.

"Intelligence = Geometry, not Scale" means the mathematical structure of
intelligence itself: invariants, interfaces, transformations, error signals,
repair operations, memory, action, verification, and compute flow. It does not
mean geometric deep learning or a new neural architecture by default.

## What The History Says

The project has killed a sequence of neural-training-era and supplied-frame
approaches. The important pattern is proxy-function divergence and absorption:
losses, byte prediction, NLL compatibility, retrieval probes, smooth compute
laws, compact proof/program artifacts, compact frame packets, and hand-authored
teacher-tomography packets can look strong while real capability or novelty does
not survive the strongest boring explanation.

That kill history is evidence that the default frame may be wrong. Future work
must ask what function is being created and why the measurement is aligned with
that function before any experiment earns compute.

## Pipeline

- `code/experiment_e1.py --mode ship` — standard KD training at scale
- `code/experiment_e1.py --mode e1.5` — corrected adjudication experiment
- `code/export_model.py` — sentence-transformers export
- `code/eval_mteb.py` — MTEB benchmark evaluation

## Current Research Map

- [State](STATE.md) - live state, direction, and experimental verdicts.
- [Vision](research/VISION.md) - governing first-principles doctrine.
- [Status](research/STATUS.md) - kill record and artifact index.
- [Experiments](experiments/EXPERIMENTS.md) - experiment ledger index.
- [Eklavya Doctrine](research/EKLAVYA_DOCTRINE.md) - surviving doctrine context.
- [Deep Rethink](research/DEEP_RETHINK.md) - historical kill log and paradigm interpretation.

Historical work-loop, question-loop, and supervisor files are preserved as the
research record.

## Operating Standard

Think first. Formalize first. Test small. Build the smallest thing that can be
wrong, then measure it honestly.
