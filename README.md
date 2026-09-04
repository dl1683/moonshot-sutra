# Moonshot Sutra

Moonshot Sutra is a first-principles research lab for one question:

> What structure makes intelligence cheap?

The project is no longer defined by Sutra as a specific byte-native neural
network, Eklavya as multi-teacher KD, coordinate inheritance, evidence-native
retrieval, CTI, FrameSeed, WGD, E3, or any fixed build sequence. Those are
historical approaches unless re-admitted through a fresh precommit gate. They
remain useful as evidence, not doctrine.

## Current Vision

The active canon is `research/VISION.md`.

The five sacred outcomes are the only fixed points:

1. **Genuine Intelligence** - the system is actually capable, not merely large
   or benchmark-shaped.
2. **Improvability** - failures can be found, understood, and repaired
   surgically.
3. **Democratized Development** - independent researchers and communities can
   understand, reproduce, modify, and extend the system.
4. **Data Efficiency** - the system learns more from less because it captures
   the right structure.
5. **Inference Efficiency** - the system is cheap to run and deploy widely.

Everything else is replaceable.

## Current Status

**Embedding reboot** (September 2026): Building compact text embedding models
via knowledge distillation from larger teachers. A 149M-parameter student
(384-dim) trains on real retrieval data with contrastive and KD losses.

The open research question is whether teacher-indexed distillation (per-teacher
auxiliary heads preserving teacher identity in the loss) adds signal beyond
standard single-teacher KD. Corrected adjudication (E1.5) with paired bootstrap
CI across multiple seeds is running. See `STATE.md` for live numbers and verdicts.

A parallel shipping pipeline trains, exports, and evaluates standard KD models
independent of the research question — the artifact outranks the measurement.

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
