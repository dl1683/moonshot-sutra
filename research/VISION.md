# Vision

## What Sutra Is

Sutra is the world's most efficient learning mechanism — a from-scratch language
model designed to absorb knowledge from multiple pretrained teachers of diverse
architectures, using their pretrained reach as signal. It currently uses a
byte-level patch-global architecture, but the architecture is a means to an end.
If a better substrate for cross-teacher learning emerges, the architecture pivots.

## What Eklavya Is

Eklavya is the multi-teacher learning protocol that trains Sutra. It extracts
transferable invariants from diverse teacher models (different architectures,
different tokenizers, different training data) and routes them to the student
based on measured disagreement.

The key insight: teachers are instruments, not masters. The student learns from
their disagreements, not their consensus. A lesson is only admitted when the
student has a measured gap that the lesson addresses.

## The Core Thesis

Sutra wins by having a better learning geometry, not by being a smaller copy of
a larger model.

Intelligence = Geometry, not Scale. Mathematical structure beats brute-force
parameters. If the theory is right, you don't need a data center.

## Five Sacred Outcomes

The outcomes are load-bearing. The mechanisms that achieve them are replaceable.

1. **Genuine Intelligence** — The model is actually smart, not just large.
2. **Improvability** — Failures can be found and fixed surgically, not by
   retraining everything.
3. **Democratized Development** — Community can build on it like Linux. Byte
   substrate + open protocol = no proprietary moat.
4. **Data Efficiency** — Learn more from less. Eklavya's multi-teacher
   extraction is how we get there.
5. **Inference Efficiency** — Cheap to run. Easy tokens exit early. Hard
   tokens get full compute.

## What We Claim

- Byte-level substrate eliminates tokenizer lock-in, providing a simpler and
  more universal interface for cross-architecture knowledge transfer than
  token-projection or chunk-matching alternatives.
- Multi-teacher KD from architecturally diverse sources (transformer, SSM,
  hybrid, embedding) produces retained gains that single-teacher KD cannot.
- Gap-driven lesson selection (teach only where the student is weak) is more
  efficient than uniform distillation.

## What We Do Not Claim

- We do not claim frontier-competitive quality at this scale (121.7M params).
- We do not claim novel architecture — the byte/patch design builds on MEGABYTE
  and related work.
- We do not claim our theory is proven. S0 is a scout build. Results from the
  first training run will falsify or validate the approach.

## Current State

- **S0** (121.7M param scout): implementation complete, training imminent.
- **E1** (single-teacher KD): designed and tested, pending S0 checkpoint.
- **E2** (multi-teacher KD): fully wired with mmap-backed cache, integration
  tests, and GPU launch checklist. Ready for GPU.
- **771 unit tests passing** (17 S0 + 46 E1 + 452 E2 + 256 tooling), all CPU-only.

## Build Order

```
D0 → S0 → E1 → E2 → T0 → G0 → P0 → G1 → O0
data   scout  single  multi  teacher  gap   packet  runtime  ownership
       build  teacher teacher profile  map   compile          gates
```

Everything before G1 runs on a single RTX 5090 (24 GB VRAM). G1 is the
integrated build that proves whether the full runtime thesis holds.

## Hardware

Single NVIDIA RTX 5090 Laptop, 24 GB VRAM. S0 fits at 5.5 GB peak with
activation checkpointing. If we need a cluster, the theory is wrong.
