# CLAUDE.md — Moonshot Sutra (सूत्र)

**This is a sub-project of [AI Moonshots](../CLAUDE.md). Always read the parent CLAUDE.md and the global `~/.claude/CLAUDE.md` for universal workflow rules, Codex review gates, and engineering discipline.**

---

## Project Identity

Sutra is a from-scratch edge AI model. The thesis: **compression = intelligence**. Just as Panini compressed all of Sanskrit into ~4,000 sutras, we compress intelligence into minimal parameters by starting from better mathematics — not by shrinking existing architectures.

Every existing small model inherits the assumptions of the large model paradigm. Sutra questions those assumptions from the ground up.

---

## Hard Rules

### No Inherited Paradigms
- **No pre-trained weights.** Everything trains from scratch.
- **No copying existing tokenizers or architectures** without a mathematical argument for why that specific design is optimal for our constraints. "Everyone uses it" is not a justification.
- **Every architecture decision requires a theoretical argument before implementation.** No "let's try it and see" without a hypothesis for why it should work.
- **No assumption that the answer is a neural network.** The search space includes ALL of ML (kernel methods, probabilistic models, Bayesian networks, reservoir computing, evolutionary systems, etc.), ALL biological intelligence (brains, immune systems, collective intelligence, gene networks), ALL mathematical frameworks (information theory, category theory, algebraic topology, statistical mechanics, optimal transport, dynamical systems), and any other system that processes information efficiently. Modern deep learning is one option, not the default.

### Mathematical First
- Design choices flow from math, not intuition or convention.
- If a component exists in the model, there must be a reason grounded in information theory, geometry, optimization theory, or physics — not "transformers do it this way."
- When in doubt, derive. Don't copy.

### Benchmarking is Mandatory
- Every capability claim must be benchmarked against existing edge models in the same parameter class.
- Primary baselines: Phi-4 (5.6B), Qwen3-4B, Gemma-3-1B, Molmo-1B, Ministral-3B, SmolLM series.
- We don't get to claim "best" without numbers against these.

---

## Definition of Done

Sutra is "done" when ALL of the following are true:
1. **Benchmark sweep**: Competitive or superior on major intelligence benchmarks (MMLU, ARC, HellaSwag, GSM8K, HumanEval, WinoGrande, TruthfulQA, etc.) against models in the same parameter class
2. **Demo-able**: A live, callable model that can be tested interactively
3. **Custom hard eval**: A suite of supremely difficult queries we design ourselves, tested head-to-head against local LLMs
4. **The David story**: The narrative is "this should be much weaker but it actually outperforms" — efficiency through better math, not more scale

---

## Design Constraints

### Parameter / Capacity Budget
- Target: **4B parameters or below** — but parameter count may not even be the relevant metric if the architecture is not a neural network
- The real constraint is: trainable from scratch on a single RTX 5090 (24GB VRAM), deployable on edge hardware
- If the design transcends neural networks, define the appropriate capacity metric and justify the budget

### Hardware
- **Training**: single NVIDIA RTX 5090 (24GB VRAM)
- **Inference target**: phones, laptops, embedded devices
- Design for the constraint, don't fight it

### Compute / Time
- Not a constraint. Run as much as needed. The architecture itself should make training fast — that's the point.

### Quantization-Native
- Designed for low precision from day one — not quantized after the fact
- Architecture choices must be robust to INT4/INT8 from the start

### Text Only
- **Text only.** No multimodal. No images, audio, video, or structured data.
- Multimodal is a future project when we have the resources. Not in scope for Sutra.

---

## Design Process

This moonshot uses the **Chrome workflow aggressively** — deep theoretical reasoning with empirical probes as a first-class reasoning tool. Theory and experiments feed each other in alternating cycles. We don't just think then build — we think, probe, learn, revise, probe again. Every theoretical claim gets tested against reality early and often.

Inspired by Chrome from Dr. Stone: the self-taught scientist who builds understanding through disciplined experimentation grounded in theory. No pure armchair derivation. No blind trial-and-error. Theory proposes, experiment disposes, theory refines.

### Pre-Training Gate (MANDATORY — learned from 7 mid-training bugs)
Before ANY production training run (>1000 steps), ALL must pass:
1. **Codex architecture audit** — causality, leakage, numerical stability
2. **Codex pipeline audit** — data loading, loss, checkpointing, eval
3. **Checkpoint resume test** — save step 10, kill, resume, verify
4. **500-step smoke test** — loss drops, no NaN/Inf
5. **Generation sanity** — greedy decode produces real words (ASCII-safe!). Run 3-5 diverse prompts, visually inspect output for coherence. BPT can be deceptive — low BPT does NOT guarantee coherent generation. This check is non-negotiable after ANY model change.
6. **Causality formal test** — changing token N has zero effect on logits 0..N-1
7. **LR stability test** — validate at production dim, not just small-scale
8. **NaN guard active** — training loop catches and skips NaN loss/gradients

### Multi-Codex Review Panel (MANDATORY at every eval checkpoint + major releases)

Run parallel Codex reviews using these 8 specialized personas. Not all are needed every time — use judgment, but ALL must review before any public-facing release (README update, paper, benchmark claim).

**ALWAYS run at eval checkpoints (every 5K steps):** 1, 2, 3, 8
**ALWAYS run before public claims:** 4, 5, 8
**Run at architecture changes:** 6, 7
**Run at major milestones:** ALL 8

#### Priority Tiers

**TIER 1 — LOOP UNTIL CLEAN (every code change, every commit):**
- **1. Correctness Engineer** — Run after every code change. Loop: fix → re-review → fix until clean.
- **2. Performance Engineer** — Run after every code change. Loop: profile → fix → re-profile until clean. Non-negotiable after OOM disasters.

**TIER 2 — EVERY CHECKPOINT (5K training steps):**
- **3. Scaling Skeptic** — At every eval, check if results transfer across scales.
- **8. Competitive Analyst** — At every eval, update positioning vs baselines.

**TIER 3 — BEFORE PUBLIC CLAIMS (README, paper, benchmark):**
- **4. Research Integrity Auditor** — Before any public number. Non-negotiable.
- **5. Novelty Challenger** — Before any novelty claim.

**TIER 4 — AT DESIGN GATES ONLY:**
- **6. Architecture Theorist** — At Chrome round design decisions.
- **7. Edge Deployment Engineer** — At major milestones, before deployment claims.

**THE LOOP RULE:** Tier 1 reviewers (Correctness + Performance) run in a loop. If they find issues, fix them, then re-run the SAME reviewer. Repeat until the reviewer returns CLEAN. Do not assume one pass catches everything — fixing one issue often introduces another.

#### The 8 Reviewer Personas

1. **Correctness Engineer** — Bugs, edge cases, pipeline integrity, data leakage, off-by-one errors. "Does the code do what you think it does?" Reviews: training loop, eval pipeline, data loading, checkpoint save/load.

2. **Performance Engineer** — Memory profiling, GPU/CPU utilization, throughput bottlenecks, batch sizing, framework overhead. "Will this OOM? What's the peak memory? Where's the bottleneck?" Must profile before any long-running job. Would have caught: bootstrap_iters=100K OOM, zombie process contention.

3. **Scaling Skeptic** — Does this result at dim=128 transfer to dim=768? At 200M params? Challenges whether Chrome probe results generalize across scales. Demands dim=768 canary runs before production decisions. Would have caught: Grokfast divergence at production scale, Peri-LN warm-start failure.

4. **Research Integrity Auditor** — Statistical methodology, benchmark validity, sample sizes, fair comparisons, overclaims. "Can you defend this number if someone adversarial reads your README?" Checks: sample sizes, answer-position bias, proper evaluation harness, confidence intervals, comparison fairness. Would have caught: 10-question ARC claim, answer-always-at-index-0 bias.

5. **Novelty Challenger** — Searches for prior art aggressively. "Who else has done this? What's ACTUALLY new vs. rediscovered?" Compares every mechanism against existing work: AdaPonderLM, Universal Transformer, PonderNet, Latent Reasoning, etc. Forces honest novelty claims. Would have caught: "elastic compute is novel" overclaim.

6. **Architecture Theorist** — The Chrome round reviewer. Derives from first principles, validates mathematical arguments, ensures system coherence between stages. Challenges whether components are justified by theory or just intuition. **Also forward-looking:** proposes new mechanisms derived from math/physics/biology that could improve the architecture, identifies theoretical gaps in the current design, and connects findings from one Chrome cycle to hypotheses for the next. **Cross-domain pattern recognition is key**: the same mathematical structures appear in neural networks, biological brains, physics, coding theory, and dynamical systems. This reviewer should actively seek analogies across domains — not just "what can we copy from ML" but "what principles from thermodynamics, information geometry, category theory, collective intelligence, etc. could inform our next design?" The 15-domain research sweep that led to v0.5.4 is the template.

7. **Edge Deployment Engineer** — Quantization readiness, inference latency, memory footprint on target hardware (phones, laptops, embedded). "Can this actually run on edge? What's the latency of 8 recurrent steps vs. a single-pass transformer?" Validates that architecture choices serve the deployment goal, not just training convenience.

8. **Competitive Analyst** — Where does Sutra actually stand vs. Pythia, SmolLM2, Phi, Gemma, Qwen at the same param class? Real numbers on real benchmarks. Tracks the moving target — what baselines have improved since we last checked? Forces honest positioning. **Also research-forward:** identifies what the field is doing that we should consider, flags gaps in our architecture that new research could fill, and recommends what to build next based on competitive landscape. Not just "where are we" but "where should we go."

### Sutra = Infrastructure, Not a Model
**Full vision: research/VISION.md** — read this document every session.
Design for modularity: each stage is a swappable, independently improvable component.
Stage interfaces: (mu, lam, pi) in → (mu, lam, pi) out. Always.
Future: stages become sub-graphs (e.g., Memory → {short_term, long_term, domain}).
The transition kernel routes to sub-stages automatically.
Community members improve specific stages for their domains.

### Exploration = Building
There are no separate "research" and "implementation" phases. Research drives data, data drives research. They are coupled. Mix long shots with close-to-home approaches. Codex decides when the design is sound.

### Pivot Criteria
The data tells us. Codex reviews artifacts and decides whether to continue or pivot. No approach is sacred — if the probes say it's dead, it's dead.

### Autonomy — ELITE SCIENTIST MODE
Fully autonomous operation. User is monitoring and will provide feedback directly.

**NEVER STOP WORKING. NOT FOR ONE SECOND. ZERO IDLE TIME.**
Every message must contain productive work — experiments, code, theory, research.
If you catch yourself writing status updates without doing work, START DOING WORK.
When experiments run in background:
- Develop alternative theories, challenge assumptions, sketch new architectures
- Research cross-domain insights (biology, physics, math — NOT just ML)
- Discuss ideas with Codex, iterate on theory
- Prototype components, run analysis, generate data
- Update task list with next steps discovered during theory work
- Log ALL discussions, theorems, insights in research/RESEARCH.md

**Think like an elite scientist:** when the lab equipment is running, you're at the whiteboard
deriving the next experiment, reading papers, challenging your own assumptions, arguing with
colleagues (Codex). Compute time = theory time. ZERO idle time.

**Task list is LIVE:** Automatically create tasks for new ideas, new probes, new theories as
they emerge during research. The task list drives autonomous momentum — never run out of
things to do.

### THE SUTRA VISION: Stage-Superposition State Machine

The 7 processing stages are NOT a linear pipeline. They form a STATE GRAPH:
- Multiple stages ACTIVE SIMULTANEOUSLY on different positions
- Stages LOOP BACK (verify → reroute → retry)
- Each position in SUPERPOSITION of stages (like quantum states)
- Each patch progresses at its OWN RATE (like spiking neural networks)
- Stage 7 (verify) can loop to Stage 4 (reroute) — decode-verify is iterative

The 7 stages (converged through 4 Codex debate rounds):
1. Segmentation/Compression  2. State Init/Addressing
3. Local Construction  4. Communication/Routing
5. State Update/Memory Write  6. Compute Control  7. Readout/Decode/Verify

For EACH stage: derive the optimal mechanism FROM MATH, not from existing models.
DO NOT copy (adding attention = copying Jamba). DERIVE something better.

### Data Strategy
- Open datasets + any files available on this laptop
- Complete freedom — no constraints on data sources
- Data pipeline is part of the architecture design, not an afterthought

---

## File Management (HARD RULES — ZERO EXCEPTIONS)

These rules prevent the repo from exploding during autonomous work.

### Maximum File Inventory
| Purpose | File(s) | Location |
|---------|---------|----------|
| Constitution | `CLAUDE.md` | root |
| Public face | `README.md` | root |
| All research, theory, Chrome cycle results, literature | `research/RESEARCH.md` | single rolling doc |
| Experiment ledger (machine-readable) | `experiments/ledger.jsonl` | one JSONL file |
| Experiment summary (human-readable) | `experiments/EXPERIMENTS.md` | one summary |
| Source code | `code/*.py` (minimal files) | one canonical runner + modules |
| Results | `results/*.json` | structured JSON only |
| Figures | `results/figures/*.png` | publication-quality only |
| Custom eval suite | `eval/` | eval queries + scoring |

### Anti-Entropy Rules
- **Never create a new file when you can edit an existing one.**
- **Every commit must leave the repo cleaner or equal — never dirtier.**
- **Delete outdated content aggressively** — old probe results, superseded theory sections, dead code.
- **No duplicate information across files** — one canonical location per fact.
- **Consolidation pass after every major Chrome cycle** — merge, prune, compress.
- **No `v2.py`, `new.py`, `temp.py`, `old_*.py`** — if it's superseded, delete it.
- **No script sprawl** — one canonical execution path, config-driven variation.
- **Results are structured JSON** — not loose text files, not notebooks, not CSVs.
- **Figures are pruned** — only keep what's publication-quality or actively needed for Codex review.

### What Gets Deleted
- Any file not referenced by CLAUDE.md, RESEARCH.md, EXPERIMENTS.md, or active code
- Superseded theory sections (replaced in RESEARCH.md, not archived)
- Probe results older than the current design iteration (summarized in ledger, raw deleted)
- Dead code paths, commented-out blocks, TODO placeholders

---

## Artifact Strategy

### What We Store and Why
| Artifact | Format | Retention | Purpose |
|----------|--------|-----------|---------|
| Chrome cycle findings | Section in `research/RESEARCH.md` | Consolidated after each cycle | Theory evolution |
| Probe results (success) | `results/*.json` | Keep if validates current theory | Positive evidence |
| Probe results (failure) | `results/*.json` | Keep if invalidates a hypothesis | Negative evidence (equally valuable) |
| Failed approaches | Section in `research/RESEARCH.md` | Permanent — never delete | Prevents revisiting dead ends, informs pivots |
| Experiment runs | Entry in `experiments/ledger.jsonl` | Permanent (append-only) | Reproducibility — includes failures with status="FAIL" |
| Code | `code/*.py` | Keep only current working version | Implementation |
| Codex review output | Summarized into `research/RESEARCH.md` | Output files deleted after ingestion | Design decisions |
| Figures | `results/figures/*.png` | Only publication-quality | Communication |

### Failure Tracking (MANDATORY)
Negative results are first-class data in the Chrome workflow. Every failed probe must be:
1. **Logged in `experiments/ledger.jsonl`** with `status: "FAIL"` and `notes` explaining why it failed
2. **Summarized in `research/RESEARCH.md`** under a "Dead Ends" section with: what was tried, what happened, why it failed, what we learned
3. **Referenced in Sutra memory** (`Failed Approaches` section) if it's a major directional dead end
4. **Never deleted** — failed approaches prevent future sessions from repeating mistakes
5. **Reviewed by Codex** — failures often reveal more than successes about the underlying structure

### Codex Review Artifacts
Every Codex review produces an output file. After Claude reads it:
1. Key decisions/findings get added to `research/RESEARCH.md`
2. Action items get executed
3. **The output file is deleted** — RESEARCH.md is the single source of truth

---

## Codex Prompt Template (MANDATORY)

**Every single Codex invocation MUST begin with this preamble. No exceptions.**

```
MANDATORY FIRST STEP: Read CLAUDE.md in this repository root. Every rule there is binding.

CONTEXT YOU MUST KNOW:
- We are in a CHROME WORKFLOW: theory + experiments alternate.
  v0.5.4 (69.4M params, Gated Peri-LN + Delayed Pheromone) is training on GPU.
  Building diverse 25B-token corpus in parallel. Next milestone: dim=1024 scale-up.
- The architecture vision is a STAGE-SUPERPOSITION STATE MACHINE with 7 stages that
  form a state graph (not a pipeline). See research/STAGE_ANALYSIS.md for full details.
- DO NOT recommend copying existing architectures (e.g., "just add attention layers").
  Every mechanism must be DERIVED from first principles (math, info theory, biology).
- CRITICAL: This is an INTEGRATED SYSTEM. All stages must COMPLEMENT each other,
  not compete. Each stage's output must be exactly what the next stage needs.
  Components must lift each other exponentially, not drag each other down with
  conflicting biases. Evaluate proposals for SYSTEM COHERENCE, not just per-stage quality.
- File management: NEVER create new files unless necessary. Edit existing. Delete outdated.
- All findings → research/RESEARCH.md. Raw ideas → research/SCRATCHPAD.md.

After reading CLAUDE.md, explore the repo structure to understand current state.

ACTUAL TASK:
[task goes here]
```

This template is non-negotiable. If a Codex prompt doesn't start with this, it's a bug.

---

## Resources Available

### From AI Moonshots Parent Project
- **CTI Universal Law**: `../moonshot-cti-universal-law/` — universal law of representation quality, could directly inform architecture
- **Fractal Embeddings**: `../moonshot-fractal-embeddings/` — multi-scale geometric structure in embeddings
- **Self-Constructing Intelligence**: findings on evolutionary emergence from random init
- **Latent Space Reasoning**: random tokens unlocking hidden capabilities
- **Model Directory**: `../models/MODEL_DIRECTORY.md` — all available models for comparison

### Hardware
- NVIDIA RTX 5090 (24GB VRAM)
- Full laptop compute available

### Data
- Open datasets (HuggingFace, academic, etc.)
- Any files on this machine
- No restrictions
