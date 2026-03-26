# Sutra (सूत्र)

### Intelligence is not scale. Intelligence is geometry.

> *Panini compressed all of Sanskrit into ~4,000 sutras. We're compressing intelligence into minimal parameters — not by shrinking existing architectures, but by starting from better mathematics.*

**Sutra is a from-scratch language model built on a single laptop GPU that challenges the fundamental economics of AI.** The architecture is derived from first principles across multiple domains — information theory, geometric deep learning, statistical mechanics, coding theory, neuroscience, and more. Nothing is copied from existing architectures without a mathematical argument for why it's optimal.

The question we're answering: **Can mathematical insight close the gap that brute-force scale creates?**

---

## The Mission

AI should be cheap, ubiquitous, and useful to the poorest person on the street — not just the richest corporation in the cloud.

If intelligence requires a data center, it will always be controlled by those who can afford data centers. If intelligence is geometry, it's as free as mathematics itself.

---

## The 5 Non-Negotiable Outcomes

Everything in Sutra serves exactly 5 outcomes. The outcomes are sacred. The mechanisms used to achieve them are not — they evolve through rigorous empirical testing.

1. **Genuine Intelligence** — The model must be genuinely smart: benchmarks, generation quality, competitive with best-in-class
2. **Improvability** — When it fails, you can identify WHAT failed and fix it surgically
3. **Democratized Development** — Anyone can improve it for their domain. Improvements compose like Linux subsystems
4. **Data Efficiency** — Learn from everything: other models, mathematical structure, symbolic systems — not just raw text
5. **Inference Efficiency** — Cheap to run. Easy tokens exit early, hard tokens get full compute

---

## The Research Process

Sutra uses a rigorous **Tesla+Leibniz design workflow** — deep theoretical reasoning combined with empirical probes as first-class reasoning tools. Architecture is discovered through iterative cycles of assumption-questioning, cross-domain research, and falsification testing.

### Key Research Areas (2025-2026 Literature)

| Area | Key Innovations | Relevance |
|------|----------------|-----------|
| **DeepSeek Engram** | O(1) hash-indexed n-gram memory, separates knowledge from reasoning | Frees neural capacity for reasoning, memory in CPU RAM |
| **Multi-Token Prediction** | Predict D+1 tokens, discard at inference or use for 1.8x speedup | Direct data efficiency multiplier, zero inference cost |
| **Hyperbolic Neural Networks** | Exponential volume growth matches hierarchical data, 4% gains | Fewer dimensions needed, structural priors reduce data needs |
| **Sheaf Neural Networks** | Relationship-specific geometries per token pair | Richer per-parameter expressiveness |
| **Hyena Edge** | Gated convolutions replace 2/3 of attention, 30% faster on edge | O(N log N), better quality AND speed |
| **Elastic Depth** | LoopFormer, MoR — token-level adaptive compute | Matched inference cost reduction |

Full details: `research/RESEARCH.md`

---

## Hardware

**Single NVIDIA RTX 5090 laptop (24GB VRAM).** If the theory is right, that's enough.

---

## Data

- 22.9B tokens available across 246 shards, 18 sources
- 16K custom BPE tokenizer (validated as biggest single efficiency win)
- No restrictions on data sources

---

## Competitive Targets

| Model | Params | Training Data | Key Benchmarks |
|-------|--------|--------------|----------------|
| Pythia-160M | 160M | 300B tokens | HellaSwag ~30%, PIQA ~62% |
| SmolLM2-135M | 135M | 2T tokens | HellaSwag 42.1%, PIQA 68.4% |
| Gemma-3-1B | 1B | ~6T tokens | strong mid-range |
| Qwen3-4B | 4B | ~18T tokens | frontier-class |

**The gap:** SmolLM2-135M trains on 2T tokens — a 100x+ data advantage. The architecture must be dramatically more data-efficient to compensate. That's the thesis.

---

## Repository Structure

```
sutra/
├── code/
│   ├── dense_baseline.py      # Current model + trainer
│   ├── data_loader.py          # Sharded streaming data pipeline
│   └── lm_eval_wrapper.py      # lm-eval harness integration
├── research/
│   ├── VISION.md               # Full infrastructure vision + 5 outcomes
│   ├── RESEARCH.md             # Field research informing design
│   ├── ARCHITECTURE.md         # Architecture reference (populated by T+L sessions)
│   ├── TESLA_LEIBNIZ_CODEX_PROMPT.md  # Design session template
│   └── TESLA_LEIBNIZ_WORKFLOW.md      # Design loop specification
├── results/                    # Structured JSON: benchmarks, metrics
├── experiments/
│   └── ledger.jsonl            # Every experiment logged
├── eval/                       # Custom evaluation suite
└── CLAUDE.md                   # Project constitution
```

---

## Where We Are — Honest Status

This is a **fresh start** (2026-03-25). Previous architectural explorations (v0.5.x recurrent SSM, v0.6.x 12-pass recurrence, EDSR dense with early exits) taught us valuable lessons but were products of path-dependent assumptions. Rather than continue iterating on designs born from reactive pivots, we're returning to first principles.

**What we're keeping:**
- The 5 non-negotiable outcomes (thoroughly validated through 20+ design rounds)
- The Tesla+Leibniz design workflow (proven methodology for rigorous architecture discovery)
- Field research on cutting-edge approaches (DeepSeek innovations, geometric representations, alternative architectures)
- Hardware constraints and data pipeline
- The 16K tokenizer (empirically validated across all experiments)

**What we're resetting:**
- Architecture (to be discovered by T+L from first principles)
- Implementation-specific conclusions (our specific failures ≠ universal truths)
- Benchmark claims (no claims until new architecture is trained and evaluated)

**Next:** Launch fresh T+L design sessions where the architecture is discovered dynamically, informed by the 5 outcomes + field research + questioning of every assumption.

---

## The Manifesto

*AI should be like electricity. It should be like vaccines. It should be cheap, ubiquitous, and useful to the poorest person on the street, not just the richest corporation in the cloud.*

*To be a tool that uplifts everyone, instead of concentrating value in the tech oligarchy and their investment bankers.*

**Everything here is open. Follow along, challenge our assumptions, or build with us.**

## Part of [AI Moonshots](https://github.com/dl1683/ai-moonshots)

*By Devansh — building intelligence from first principles.*

## License

MIT
