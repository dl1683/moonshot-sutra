# Sutra Architecture Reference

**Status: FRESH START (2026-03-25)**

This file is the single source of truth for Sutra's architecture. It is populated and updated by Tesla+Leibniz design sessions. Every T+L round reads this file and questions every decision in it.

---

## Current Architecture

**No architecture has been selected yet.** The T+L design loop will evaluate options from first principles, informed by the research in `research/RESEARCH.md` and the 5 non-negotiable outcomes in `research/VISION.md`.

The search space is unlimited: transformers, SSMs, hybrids, gated convolutions, hyperbolic networks, sheaf-theoretic models, reservoir computing, evolutionary systems, or something entirely novel. The architecture must be DERIVED from mathematics, not copied from convention.

---

## Design Decision Audit

Every design choice will be classified:
- **DERIVED** — mathematically justified from first principles for our specific constraints
- **INHERITED** — carried from convention or prior work, needs re-evaluation
- **VALIDATED** — empirically tested and confirmed beneficial
- **QUESTIONED** — under active investigation
- **FALSIFIED** — empirically tested and found wanting

*This section will be populated as T+L sessions make and validate decisions.*

---

## Parameter Budget

*To be determined by T+L design sessions.*

Constraints:
- Trainable from scratch on a single RTX 5090 (24GB VRAM)
- Target: competitive in the 100M-4B parameter class
- 16K custom BPE tokenizer (validated as biggest single win)
- Must be deployable on edge hardware (phones, laptops, embedded)

---

## Benchmark Tables

*To be populated as models are trained and evaluated.*

Baselines for comparison:
- Pythia-160M (300B tokens)
- SmolLM2-135M (2T tokens)
- Gemma-3-1B (~6T tokens)
- Qwen3-4B (~18T tokens)
- Phi-4 5.6B (~10T tokens)

---

## Prior Attempts (from git history)

Previous architectures were tried and evolved through v0.5.x → v0.6.x → EDSR-98M. These are documented in git history. **Important caveat:** Many "falsification" results were specific to our particular implementation at a specific scale (42-68M params, specific hyperparameters, specific training setup). They should be interpreted as "this specific implementation didn't work" rather than "this concept doesn't work." The T+L process must question whether alternative implementations of the same concepts might succeed.

Key learnings that appear robust (survived multiple implementations):
- **16K tokenizer** is a massive win over 50K GPT-2 tokenizer (recovered 56.5% of dead embedding parameters)
- **Elastic compute** (shallower depth competitive with full depth) replicated across ALL experiments
- **Warm-starting** consistently outperforms from-scratch at equivalent wall-clock time

Learnings that are implementation-specific (question before generalizing):
- Recurrence results depend heavily on implementation details (shared vs unshared weights, number of passes, etc.)
- Online KD results are specific to our teacher selection and scale
- Stage bank results are specific to our 7-stage decomposition

See `research/RESEARCH.md` for field research on alternative approaches.
