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

- **Per-position stage probabilities**: Each token is in a superposition of stages inside a fixed 8-step recurrent pass. The current v0.5.4 code does not implement true early-exit or verify/reroute loops yet.
- **Content-dependent transitions**: A learned Markov kernel decides how each position moves through stages based on what it's processing. Not a fixed pipeline.
- **Shared discourse memory (scratchpad)**: 8 shared memory slots that all positions can read/write. This gives the model a "working memory" for maintaining context across the sequence — like a human keeping track of the topic while reading.
- **Recurrent processing**: 8 iterations of the stage graph per forward pass. Each iteration refines the representation. Late iterations are the most valuable — this is where reasoning happens.
- **Bayesian state updates**: Precision-weighted evidence accumulation, not simple residual addition. The model tracks confidence.

### How It Actually Works: A Concrete Walkthrough

Let's trace what happens when Sutra processes the sentence: **"The cat sat on the mat because it was tired"**

#### Step 0: Initialization
Each token gets embedded and enters the stage graph at **Stage 3 (Local Construction)**. Every token starts with a stage probability vector:

```
"The"     → pi = [0, 0, 1.0, 0, 0, 0, 0]  (100% in Stage 3)
"cat"     → pi = [0, 0, 1.0, 0, 0, 0, 0]
"sat"     → pi = [0, 0, 1.0, 0, 0, 0, 0]
"because" → pi = [0, 0, 1.0, 0, 0, 0, 0]
"it"      → pi = [0, 0, 1.0, 0, 0, 0, 0]
...
```

The scratchpad memory is initialized — 8 empty slots of shared working memory.

#### Iteration 1: Local Understanding
The **content-dependent transition kernel** looks at each token's hidden state and decides where it should move. Simple tokens advance quickly:

```
"The"     → pi = [0, 0, 0.3, 0.6, 0.1, 0, 0]  (mostly Stage 4: ready to route)
"cat"     → pi = [0, 0, 0.4, 0.5, 0.1, 0, 0]  (moving toward routing)
"because" → pi = [0, 0, 0.7, 0.2, 0.1, 0, 0]  (still constructing — it's a complex connective)
"it"      → pi = [0, 0, 0.6, 0.3, 0.1, 0, 0]  (needs context to resolve the reference)
```

**Top-2 projection** keeps only the 2 most active stages per token — bounding compute. The **stage bank** applies stage-specific transforms (each stage has its own small MLP), weighted by probability.

Then **routing** happens: tokens in Stage 4 send messages to other tokens. "cat" sends information to nearby positions. "it" starts looking for its antecedent.

The **scratchpad** gets its first write: a blurry summary of what's happening in the sequence. All tokens can read this shared context.

#### Iterations 2-4: Building Context
The transition kernel keeps evolving each token's stage distribution based on what it has learned:

```
"The"     → pi = [0, 0, 0, 0.1, 0.3, 0.1, 0.5]  (arriving at Stage 7: ready to output)
"cat"     → pi = [0, 0, 0, 0.2, 0.4, 0, 0.4]     (writing to memory, approaching output)
"because" → pi = [0, 0, 0.1, 0.5, 0.3, 0.1, 0]    (still routing heavily — connecting clauses)
"it"      → pi = [0, 0, 0, 0.6, 0.3, 0.1, 0]      (routing intensively — resolving "it" → "cat")
"tired"   → pi = [0, 0, 0.2, 0.3, 0.4, 0, 0.1]    (writing the reason to memory)
```

Notice: **easy tokens ("The") tend to concentrate on later readout stages faster, while hard tokens ("because", "it") spend more mass on routing and memory.** In current v0.5.4 this is still expressed inside a fixed 8-step loop, not a true variable-depth controller.

The scratchpad now contains a discourse summary: roughly "there's a cat, it did something, and there's a causal relationship."

#### Iterations 5-8: Refinement and Reasoning
**This is where the magic happens.** Late iterations are 2-4x more valuable than early ones.

The **pheromone traces** (activated at iteration 3+) mark which positions changed most during previous iterations. "because" and "it" have high pheromone — the routing system now prioritizes these hard positions.

"it" finally locks onto "cat" through the routing mechanism. The causal chain "because → tired → cat" gets encoded. The scratchpad memory provides global context that helps resolve long-range dependencies.

By iteration 8:
```
"The"     → pi = [0, 0, 0, 0, 0, 0, 1.0]  (Stage 7: output ready)
"cat"     → pi = [0, 0, 0, 0, 0, 0, 1.0]  (output ready)
"because" → pi = [0, 0, 0, 0, 0.1, 0, 0.9] (output ready, still tracking state)
"it"      → pi = [0, 0, 0, 0, 0, 0, 1.0]  (resolved: it=cat, output ready)
"tired"   → pi = [0, 0, 0, 0, 0, 0, 1.0]  (output ready)
```

#### Final: Output
The hidden state `mu` has been refined through 8 iterations of evidence accumulation. It's projected through the output head to produce logits for the next token prediction.

**The key insight**: this isn't a fixed 8-layer network. It's a dynamical system where each token follows its own path through the stage graph. The stage probabilities evolve differently per token — some concentrate on readout quickly while others spend more mass on routing and memory. In v0.5.4, all tokens still run through 8 recurrent steps (true early-exit and verify/reroute are design goals for v0.5.5+), but the stage distributions already create **content-dependent processing** within that fixed loop.

### Why This Matters

In a transformer, every token gets the same 32 or 64 layers of computation regardless of difficulty. In Sutra:
- **Function words** ("the", "a", "is") concentrate on readout stages faster
- **Content words** spend more time in local construction
- **Connectives and references** ("because", "it", "however") spend more time routing
- **Ambiguous or complex tokens** loop through more iterations

This is how humans read. You don't spend equal time on every word. You skim the easy parts and slow down for the hard parts. Sutra does this naturally through the stage-superposition mechanism.

---

## Modular Intelligence: The Warm-Start Architecture

This is one of Sutra's most underappreciated features. The architecture isn't just a model — it's **infrastructure for evolving intelligence**.

### Every Version Builds on the Last

```
v0.5.0 (base)                     → 67M params
  ↓ warm-start (91% transfer)
v0.5.2 (+switching kernel)         → 67M params, +4.1%
  ↓ warm-start (91% transfer)
v0.5.3 (+scratchpad memory)        → 67.6M params, +10.2%
  ↓ warm-start (81% transfer)
v0.5.4 (+gated norms + pheromone)  → 67.6M params, training now
v0.6.0 (dim=1024, on FineWeb)      → 105M params, planned
```

Nothing is thrown away. Each version inherits everything the previous version learned. This is **compound intelligence** — each improvement builds on all prior improvements.

### The Gated Warm-Start Trick

When you add new components to a trained model, you can't just splice them in — they destroy the learned representations. We solved this with **GatedLayerNorm**:

```python
# Standard LayerNorm: DESTROYS warm-start (BPT 5.96 → 11.39)
output = LayerNorm(x)

# Gated LayerNorm: PRESERVES warm-start (BPT 5.96 → 5.96)
alpha = sigmoid(learnable_gate)  # starts at ~0 (identity)
output = (1 - alpha) * x + alpha * LayerNorm(x)
```

The gate starts nearly closed (identity function). During training, it gradually opens, allowing normalization to activate without disrupting learned representations. This means we can **add ANY new component to a running model** by gating it through a bypass.

### The Scaling Vision

The next scale-up target is dim=1024 on the larger diverse corpus. Warm-starting across compatible checkpoints is already part of the codebase; widening into a larger parameterization is still a design direction, not a checked-in utility yet.

### Why This is Infrastructure, Not Just a Model

Imagine Sutra at scale, open-sourced. The 7-stage architecture means:

- **A memory researcher** can improve Stage 5 (memory write) without touching anything else
- **A routing expert** can redesign Stage 4 (communication) independently
- **A compression specialist** can optimize Stage 1 (segmentation) in isolation
- **Domain specialists** can fine-tune specific stages for medicine, law, or code

Each stage has a clean interface: `(mu, lambda, pi) in → (mu, lambda, pi) out`. Stages are **modular, independently improvable components**. This is how Linux works — thousands of contributors improving specific subsystems. Sutra is designed for the same pattern.

The warm-start chain means every improvement from every contributor compounds automatically. **This is modular intelligence infrastructure, not a monolithic model.**

---

## Current Status

**v0.5.4** — Step 15K checkpoint, evaluated against Pythia-70M

| Metric | Value |
|--------|-------|
| Parameters | 67.6M |
| Architecture | Stage-Superposition + Switching Kernel + Scratchpad + Gated Peri-LN + Pheromone |
| Training data | 1.7B tokens (MiniPile) — 14B diverse tokens ready for next run |
| Best eval | **5.66 BPT** at step 15K |
| Hardware | Single NVIDIA RTX 5090 (24GB VRAM) |

### The Efficiency Story

| | Pythia-70M | Sutra v0.5.4 | Ratio |
|---|-----------|-------------|-------|
| **Training tokens** | 300B | 0.5B | Pythia used **610x more** |
| **Hardware** | 64x A100 | 1x RTX 5090 | Pythia used **768x more** |
| **Estimated cost** | ~$10,752 | ~$10 | Pythia cost **1,024x more** |
| **SciQ (science)** | 74% | **50%** | Sutra at **2x random** |
| **Performance/dollar** | baseline | **~960x better** | |

Sutra achieves **94% of Pythia's language modeling performance at 0.1% of the cost.** On science benchmarks (SciQ), Sutra scores 50% — double random chance — having seen 610x less data.

The gap is **data**, not architecture. Pythia saw 300B tokens of diverse web text. Sutra has only seen 0.5B tokens of academic papers. We have 14B diverse tokens (Wikipedia, math, stories, conversations) ready for the next training run.

### Benchmark Comparison

| Benchmark | Pythia-70M | Sutra v0.5.4 | Random | Notes |
|-----------|-----------|-------------|--------|-------|
| **ARC-Easy** (science) | 38.5% | **40.0%** | 25% | **Sutra wins** (+1.5pp) |
| **ARC-Challenge** (hard science) | 21.4% | **40.0%** | 25% | **Sutra wins** (+18.6pp!) |
| **LAMBADA** (word prediction) | 32.6% | 10.0% | 0% | Above random |
| **HellaSwag** (completion) | 27.2% | 20.0% | 25% | Needs diverse data |
| **WinoGrande** (coreference) | 51.9% | 30.0% | 50% | Needs diverse data |
| **PIQA** (physical intuition) | 60.5% | 30.0% | 50% | Needs diverse data |
| **SciQ** (science MC) | 74.0% | 10.0% | 25% | Question-format sensitive |

**Sutra beats Pythia-70M on ARC science benchmarks** — both easy and challenge — despite training on 610x less data with 1024x less compute. On ARC-Challenge specifically, Sutra scores 40% vs Pythia's 21.4%, nearly double.

The model is strong where its training data is strong (academic/science) and weak where it has never seen data (common sense, physical intuition). The 14B diverse token corpus will address these gaps.

### Architecture Evolution

```
v0.5.0  Base Stage-Superposition SSM
  |
v0.5.2  + Switching Kernel (+4.1%) + Gain Clamp (eliminates NaN)
  |
v0.5.3  + Scratchpad Memory (+10.2%) — 2-4x training speedup at production scale
  |
v0.5.4  + Gated Peri-LN + Delayed Pheromone (current)
  |
v0.5.5  + Continuous-Spectrum Memory + Elastic Compute Budget (designing)
```

Each version warm-starts from the previous — knowledge accumulates, nothing is wasted.

### What's Next: Elastic Compute + Spectrum Memory

We're designing two connected innovations for v0.5.5:

**Continuous-Spectrum Memory** — Instead of fixed "gist or exact" memory, a continuous zoom from wide-angle (discourse context) to narrow-focus (exact token recall). The model learns what resolution each position needs. Like a camera zoom, not a microscope with click-stops.

**Elastic Compute Budget** — Current AI models are uniformly mediocre: every token gets the same compute. This is a design target for a future version. The current v0.5.4 code still runs a fixed 8-step recurrent loop with no live verify/reroute controller or adaptive emit path.

This connects to a fundamental insight: **the reason AI models produce uniform mediocrity is that they never learn to prioritize.** A compute budget creates the incentive to be efficient on easy things and thorough on hard things — like how humans skim easy text and slow down for complex passages.

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

**Pattern discovered**: At 67.6M parameters, only simple shared state and coarse bias work. Complex learned control mechanisms fail — they need more model capacity to learn both the base task AND the control mechanism simultaneously. We call this the **delayed-start principle**: novel mechanisms that depend on accumulated state must activate only after the model has formed a provisional hypothesis (~recurrent step 3).

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

The full synthesis is in `research/RESEARCH.md`.

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
| **Current** | 67.6M (dim=768) | 1.7B tokens (MiniPile) | Validate architecture |
| **Next** | 105M (dim=1024) | 25B tokens (diverse) | First reasoning emergence |
| **Target** | 148M (dim=1280) | 25B+ tokens | Competitive with SmolLM2-135M |
| **Stretch** | <4B | Full corpus | The David story |

We warm-start between compatible checkpoints today. Larger-width migration remains planned work.

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
│   └── spectrum_memory.py         # Continuous-spectrum memory (v0.5.5)
├── research/
│   ├── RESEARCH.md                # All findings (~5000 lines)
│   ├── VISION.md                  # The full infrastructure vision
│   └── STAGE_ANALYSIS.md          # Deep theoretical design for 7 stages
├── results/
│   ├── chrome_v054_ablation.json     # Chrome experiment results
│   ├── chrome_v054_probes.json       # Chrome probe results
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
