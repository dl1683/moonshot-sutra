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

Full field research: `research/RESEARCH.md`

---

## Hardware

**Single NVIDIA RTX 5090 laptop (24GB VRAM).** If the theory is right, that's enough.

---

## Data

- ~22.9B tokens available across 246 shards, 18 sources
- 16K custom BPE tokenizer
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
│   ├── dense_baseline.py      # Model + trainer (to be replaced by T+L output)
│   ├── data_loader.py          # Sharded streaming data pipeline
│   └── lm_eval_wrapper.py      # lm-eval harness integration
├── research/
│   ├── VISION.md               # Full infrastructure vision + 5 outcomes
│   ├── RESEARCH.md             # Field research informing design
│   ├── ARCHITECTURE.md         # Architecture reference (populated by T+L sessions)
│   ├── TESLA_LEIBNIZ_CODEX_PROMPT.md  # Design session template
│   └── TESLA_LEIBNIZ_WORKFLOW.md      # Design loop specification
├── results/                    # Structured JSON: benchmarks, metrics (empty — fresh start)
├── experiments/
│   └── ledger.jsonl            # Every experiment logged
├── eval/                       # Custom evaluation suite
└── CLAUDE.md                   # Project constitution
```

---

## Where We Are

Architecture is an open question. The T+L design workflow will discover it from first principles, informed by the 5 outcomes and extensive field research on cutting-edge approaches (DeepSeek innovations, geometric representations, alternative architectures, quantization-native design).

What we're keeping: the process (T+L workflow, Codex reviewer suites, Chrome experimentation cycle), the data pipeline, the 16K tokenizer, and the field research.

---

## The Manifesto

*AI should be like electricity. It should be like vaccines. It should be cheap, ubiquitous, and useful to the poorest person on the street, not just the richest corporation in the cloud.*

*To be a tool that uplifts everyone, instead of concentrating value in the tech oligarchy and their investment bankers.*

**Everything here is open. Follow along, challenge our assumptions, or build with us.**

## Part of [AI Moonshots](https://github.com/dl1683/ai-moonshots)

*By Devansh — building intelligence from first principles.*

## License

MIT
