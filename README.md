# Sutra (सूत्र)

**What if intelligence isn't about scale — it's about geometry?**

> *"You don't need a data center to be intelligent. You just need better mathematics."*

Sutra is a from-scratch language model that questions the fundamental assumptions of modern AI. Named after the Sanskrit tradition where Panini compressed all of Sanskrit grammar into ~4,000 sutras, we're compressing intelligence into minimal parameters by starting from better math — not by shrinking existing architectures.

**Everything here is open. Follow along, challenge our assumptions, or build with us.**

---

## Why This Exists

Every frontier AI model today is built the same way: take a transformer, make it bigger, train it longer, spend more money. The implicit axiom is **Intelligence = Scale**.

We believe **Intelligence = Geometry**. A system designed from first principles — where every component has a mathematical reason to exist — can achieve the same capabilities at a fraction of the cost. Not by being clever about engineering, but by asking different questions about what computation actually needs to happen.

If we're right, powerful AI becomes accessible to everyone with a laptop. If we're wrong, we'll have learned exactly why scale is necessary — which is also valuable.

---

## The Architecture: Stage-Superposition State Machine

Sutra isn't a transformer. It's a **Stage-Superposition State Machine** — a system where each token exists in a probability distribution over 7 processing stages simultaneously, with content-dependent transitions governing how computation flows.

### The 7 Stages

| Stage | Function | Analogy |
|-------|----------|---------|
| 1. Segmentation | Compress input into meaningful units | The eye parsing a scene |
| 2. Addressing | Initialize state, locate in context | Opening the right file |
| 3. Local Construction | Build local representations | Understanding a sentence |
| 4. Communication/Routing | Route information between positions | Neurons talking to each other |
| 5. State Update/Memory | Write to persistent shared memory | Taking notes |
| 6. Compute Control | Decide how much processing is needed | Knowing when to think harder |
| 7. Readout/Verify | Generate output, check correctness | Proofreading your answer |

### What Makes This Different

- **Per-position stage probabilities**: Each token is in a superposition of stages. Easy tokens emit early; hard tokens loop through more processing. The model allocates compute where it's needed.
- **Content-dependent transitions**: A learned Markov kernel decides how each position moves through stages based on what it's processing. Not a fixed pipeline.
- **Shared discourse memory (scratchpad)**: 8 shared memory slots that all positions can read/write. This gives the model a "working memory" for maintaining context across the sequence — like a human keeping track of the topic while reading.
- **Recurrent processing**: 8 iterations of the stage graph per forward pass. Each iteration refines the representation. Late iterations are the most valuable — this is where reasoning happens.
- **Bayesian state updates**: Precision-weighted evidence accumulation, not simple residual addition. The model tracks confidence.

### Key Innovations

- **Switching Kernel**: 2-mode content-dependent transition (+4.1% over baseline)
- **Scratchpad Memory**: Shared discourse state validated at 2-4x training speedup
- **Gated Peri-LN**: LayerNorm with learned bypass gates for safe warm-starting between versions
- **Delayed Pheromone Routing**: Stigmergic position traces that improve late-step recurrence by 2.6x

---

## Current Status

**v0.5.4** — Training on a single RTX 5090

| Metric | Value |
|--------|-------|
| Parameters | 69.4M |
| Architecture | Stage-Superposition + Switching Kernel + Scratchpad + Gated Peri-LN |
| Training data | 1.7B tokens (MiniPile) — expanding to 25B tokens (9 diverse sources) |
| Best eval | 5.96 BPT at step 5K (v0.5.3), v0.5.4 improving |
| Hardware | Single NVIDIA RTX 5090 (24GB VRAM) |

### Architecture Evolution

```
v0.5.0  Base Stage-Superposition SSM
  |
v0.5.2  + Switching Kernel (+4.1%) + Gain Clamp (eliminates NaN)
  |
v0.5.3  + Scratchpad Memory (+10.2%) — 2-4x training speedup at production scale
  |
v0.5.4  + Gated Peri-LN + Delayed Pheromone (current, training)
```

Each version warm-starts from the previous — knowledge accumulates, nothing is wasted.

---

## The Research Process

We use a **Chrome workflow** — theory and experiments feed each other in alternating cycles, inspired by Chrome from Dr. Stone. No pure armchair derivation. No blind trial-and-error. Theory proposes, experiment disposes, theory refines.

### What We've Tested (and What Failed)

We believe in radical transparency about what doesn't work:

| Experiment | Result | Lesson |
|-----------|--------|--------|
| Switching kernel | **+4.1%** | Coarse mode selection works at small scale |
| Scratchpad memory | **+10.2%** | Simple shared state is powerful |
| Gated Peri-LN | **+5.9%** | Normalization stabilizes recurrence |
| Grokfast gradient filter | +11% at dim=128, **diverges at dim=768** | Small-scale Chrome doesn't predict production |
| Error scratchpad | -0.4% (late steps 2.2x better) | Novel mechanisms need delayed activation |
| Pheromone router | -0.1% (late steps 2.6x better) | Same delayed-start pattern |
| Depth-drop bootstrap | NaN | KL to teacher unstable at small scale |
| Complex embeddings | -36% | Too complex for 67M params |
| CfC time constants | -14% | Needs 200M+ to show benefit |
| Surprise memory bank | -1.7% to -2.1% | Hurts every combination it touches |

**Pattern discovered**: At 69M parameters, only simple shared state and coarse bias work. Complex learned control mechanisms fail — they need more model capacity to learn both the base task AND the control mechanism simultaneously. We call this the **delayed-start principle**: novel mechanisms that depend on accumulated state must activate only after the model has formed a provisional hypothesis (~recurrent step 3).

### 15-Domain Cross-Disciplinary Research

We searched far beyond ML for inspiration:

- **Quantum physics**: Information lives in geometry (amplitudes, phases, subspaces)
- **Biology**: Predictive coding, dendritic computation, neuromodulation, cerebellar learning
- **Category theory**: Objects are defined by relationships (Yoneda), adjunctions are universal
- **Thermodynamics**: Free energy minimization, fluctuation theorems, Fisher information geometry
- **Coding theory**: BP on LDPC proves sparse local iterative computation is near-optimal
- **Collective intelligence**: Stigmergy validates scratchpad, Physarum routing is provably optimal
- **Dynamical systems**: Contraction analysis, reservoir computing, edge of chaos
- **Signal processing**: Wavelets/MRA, compressed sensing, rate-distortion theory
- **NCA/SOMs**: Shared local rules + iterations = arbitrary complexity from minimal parameters

The full synthesis is in `results/master_research_synthesis.md`.

---

## Data Strategy

Training on academic papers alone teaches the model to write like a journal, not think like a human. We're building a diverse corpus:

| Source | Tokens | What It Adds |
|--------|--------|-------------|
| FineWeb-Edu | 8.7B | Educational content across all subjects |
| Wikipedia | 4.3B | Encyclopedic, factual, well-structured |
| OpenWebMath | 3B | Math, LaTeX, proofs, reasoning |
| Project Gutenberg | 3B | Fiction, literature, narrative style |
| StackExchange | 2B | Q&A across every domain imaginable |
| MiniPile | 1.7B | Academic papers (current training data) |
| TinyStories | 500M | Narrative coherence |
| MetaMathQA | 200M | Step-by-step math solutions |
| OpenAssistant | 50M | Real human conversation |
| **Total** | **~25B** | **9 domains, 15x current data** |

---

## Scaling Plan

| Phase | Params | Data | Goal |
|-------|--------|------|------|
| **Current** | 69M (dim=768) | 1.7B tokens (MiniPile) | Validate architecture |
| **Next** | 105M (dim=1024) | 25B tokens (diverse) | First reasoning emergence |
| **Target** | 148M (dim=1280) | 25B+ tokens | Competitive with SmolLM2-135M |
| **Stretch** | <4B | Full corpus | The David story |

We warm-start between scales using Net2Net widening — no knowledge is thrown away.

---

## Philosophy

### What We're NOT Doing
- Incremental improvements to transformers
- "GPT but slightly better at X"
- Copying architectures and hoping they work
- Hiding our failures

### What We ARE Doing
- Questioning every assumption from first principles
- Deriving components from math, not convention
- Testing ruthlessly (Chrome workflow)
- Being honest about what fails and why
- Building in public so others can learn from our journey

### The Manifesto

*AI should be like electricity. It should be like vaccines. It should be cheap, ubiquitous, and useful to the poorest person on the street, not just the richest corporation in the cloud. To be a tool that uplifts everyone, instead of concentrating value in the tech oligarchy and their investment bankers.*

If intelligence requires a data center, it will always be controlled by those who can afford data centers. If intelligence is geometry, it's as free as mathematics itself.

---

## Repository Structure

```
sutra/
├── code/
│   ├── sutra_v05_ssm.py          # Core architecture
│   ├── launch_v054.py             # Current model (Gated Peri-LN + Pheromone)
│   ├── train_v054.py              # Production trainer
│   ├── grokfast.py                # Gradient filter module (disabled at dim=768)
│   ├── scratchpad.py              # Shared discourse memory
│   ├── download_diverse_data.py   # Multi-source data pipeline
│   └── eval_checkpoint_review.sh  # Generation-based evaluation
├── research/
│   ├── RESEARCH.md                # All findings (~5000 lines)
│   ├── VISION.md                  # The full infrastructure vision
│   └── STAGE_ANALYSIS.md          # Deep theoretical design for 7 stages
├── results/
│   ├── master_research_synthesis.md  # 15-domain research synthesis
│   ├── codex_v054_master_design.md   # Architecture design document
│   └── *.json                        # Chrome probe results
├── eval/
│   ├── sutra_eval_500.jsonl       # 500 reasoning questions
│   └── clean_eval.py              # Evaluation scripts
├── experiments/
│   └── ledger.jsonl               # Every experiment logged
└── CLAUDE.md                      # Project constitution
```

---

## Get Involved

This is open research. Everything — the successes, the failures, the dead ends — is here for you to learn from, challenge, or build on.

- **Read the research**: `research/RESEARCH.md` has every finding from every Chrome cycle
- **Challenge our assumptions**: Open an issue if you think we're wrong about something
- **Try the architecture**: The code is self-contained and trains on a single GPU
- **Suggest experiments**: What should we test next? What did we miss?

## Hardware

Single NVIDIA RTX 5090 laptop (24GB VRAM). If the theory is right, that's enough.

## Part of [AI Moonshots](https://github.com/dl1683/ai-moonshots)

## License

MIT
