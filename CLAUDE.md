# CLAUDE.md — Moonshot Sutra (सूत्र)

**This is a sub-project of [AI Moonshots](../CLAUDE.md). Always read the parent CLAUDE.md and the global `~/.claude/CLAUDE.md` for universal workflow rules, Codex review gates, and engineering discipline.**

---

## Project Identity

Sutra is a from-scratch edge AI model. The thesis: **compression = intelligence**. Just as Panini compressed all of Sanskrit into ~4,000 sutras, we compress intelligence into minimal parameters by starting from better mathematics — not by shrinking existing architectures.

Every existing small model inherits the assumptions of the large model paradigm. Sutra questions those assumptions from the ground up.

---

## Design Philosophy: Zero Ego, Full Commitment to Mission

**Nothing is sacred except outcomes.** Every design choice exists to enforce a specific outcome. If that outcome can be achieved better another way, the implementation MUST change. The full philosophy with deep explanations is in `research/VISION.md` — READ IT EVERY SESSION.

**The 5 Sacred Outcomes** (these are the GOALS — the mechanisms we use to pursue them are negotiable):

1. **Genuine Intelligence** — The model must be genuinely smart: understand context, reason about complex problems, generate high-quality text. This is the #1 gating criterion. Our current mechanism is *state superposition* (different content follows different compute paths, modeled as probability distributions over processing stages with 12 recurrent passes). We chose superposition because it mirrors how thinking works — you don't spend equal effort on every word. **But superposition is the mechanism, not the goal. If a better mechanism produces higher intelligence from the same resources, replace it.**

2. **Improvability** — When the model fails at something, you can identify WHAT is failing and fix it surgically without breaking everything else. The cost of improving intelligence must be low. Our current mechanism is *modular stage decomposition* (7 named stages with uniform interfaces). Stages make the system debuggable and composable. **The goal is improvability, not stages. If a different decomposition gives better improvability, use it.**

3. **Democratized Development** — No single team is the bottleneck. Domain experts improve their domain. Improvements compose. Intelligence is a public utility built by a community, not a proprietary moat. Our current mechanism is the *infrastructure vision* (swappable stage modules, like Linux subsystems). **The goal is democratization, not any specific architecture pattern.**

4. **Data Efficiency** — Extract maximum intelligence from minimum data by absorbing knowledge from every available source — other models, mathematical structure, symbolic systems, code patterns, not just raw text. Our current mechanism is *multi-teacher learning* (LEAST developed pillar — currently training from scratch due to warm-start technical issues). **Any approach that improves knowledge absorption per training token is welcome.**

5. **Inference Efficiency** — Cheap to run. Easy tokens exit early, hard tokens get full compute. ~40% cost reduction if 60% of tokens can stop after 3 passes instead of 12. Our current mechanism is *elastic compute via BayesianWrite + lambda* (precision-weighted energy budget that creates natural halting pressure). The halting pressure constraint is LOAD-BEARING (elastic compute needs it), but the specific formula is negotiable. **If a better halting mechanism exists, propose it.**

**Load-bearing choices** (solve real architectural constraints — keep unless a better solution to the SAME constraint exists):
- **Scratchpad** — External memory for precise retrieval. SSMs/recurrence lose signal over long contexts. Without this, the model degrades on any task requiring recall of earlier information.
- **Lambda/BayesianWrite** — Energy budget that creates natural stopping pressure for elastic compute. Without budget pressure, elastic compute degenerates to "always use max passes."

**Negotiable choices** (historical or experimental — replace freely if something better achieves the same outcome):
- Shared params across passes (could be partially shared or adaptive)
- Pheromone routing (any cross-position info flow mechanism works)
- 7-stage count (could be 5 or 10 — emerged from iteration, not theory)
- From-scratch training (want warm-start/multi-teacher, hit technical issues)

**The gating criterion is PERFORMANCE.** Benchmarks (MMLU, ARC, HellaSwag, GSM8K, etc.), generation quality, competitive positioning against the BEST models in our parameter class (Phi-4, Qwen3-4B, Gemma-3-1B, SmolLM). What "winning" means: same or fewer params, trained on a fraction of the data, with a fraction of the compute, matching or approaching the performance of models that used 100-1000x more resources to train. Not a specific number — the demonstration that mathematical insight closes the resource gap.

**For every Codex review, every heartbeat, every design decision**: Ask "what outcome does this enforce?" and "is there a better way?"

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
5. **Generation sanity** — greedy decode produces real words (ASCII-safe!)
6. **Causality formal test** — changing token N has zero effect on logits 0..N-1
7. **LR stability test** — validate at production dim, not just small-scale
8. **NaN guard active** — training loop catches and skips NaN loss/gradients

### Codex Pre-Training Audit (MANDATORY — learned from detached-history bug)
**Training is the most expensive operation. Never start without deep review.**

**THE LOOP:**
```
REPEAT:
  1. Run Correctness Engineer on model + trainer + data loader
  2. Run Performance Engineer — memory profile, throughput, OOM risk
  3. Run Scaling Expert — LR scaling for depth, VRAM budget, warm-start
  4. Collect ALL findings from all three
  5. Fix every HIGH and MEDIUM issue
  6. GOTO 1 (re-run ALL three reviewers on the fixed code)
UNTIL: ALL three reviewers return CLEAN (zero HIGH, zero MEDIUM)
THEN: training may begin
```

**This is NOT optional. This is NOT one-pass.** Fixing one issue often introduces another. The loop continues until convergence — all reviewers simultaneously agree the code is clean. Only then does GPU training start.

**Why this matters:**
- Detached-history bug: L_step didn't train recurrent core → entire training run wasted
- NaN at step 762: LR too high for 12-pass depth → 762 steps wasted
- Data loader OOM: loaded all shards into RAM → training couldn't start
- Each bug cost hours of GPU time. Each would have been caught by one loop iteration.

**Rule: the cost of N review loops is ALWAYS less than the cost of 1 failed training restart.**

### Multi-Codex Review Panel (MANDATORY at every eval checkpoint + major releases)

Run parallel Codex reviews using these 8 specialized personas. Not all are needed every time — use judgment, but ALL must review before any public-facing release (README update, paper, benchmark claim).

**ALWAYS run at eval checkpoints (every 5K steps):** 1, 2, 3, 6, 8
**ALWAYS run before public claims:** 4, 5, 8
**Run at architecture changes:** 6, 7
**Run at major milestones:** ALL 8

#### Priority Tiers

**TIER 1 — LOOP UNTIL CLEAN (every code change, every commit):**
- **1. Correctness Engineer** — Run after every code change. Loop: fix → re-review → fix until clean.
- **2. Performance Engineer** — Run after every code change. Loop: profile → fix → re-profile until clean. Non-negotiable after OOM disasters.

**TIER 2 — EVERY CHECKPOINT (5K training steps):**
- **3. Scaling Expert** — At every eval, deep analysis of scaling implications: what breaks, what improves, what emerges, what to retest.
- **6. Architecture Theorist** — At every eval, review current design + propose next mechanisms from cross-domain research. Ensures we're always thinking ahead, not just maintaining.
- **8. Competitive Analyst** — At every eval, update positioning vs baselines + research-forward recommendations.

**TIER 3 — BEFORE PUBLIC CLAIMS (README, paper, benchmark):**
- **4. Research Integrity Auditor** — Before any public number. Non-negotiable.
- **5. Novelty Challenger** — Before any novelty claim.

**TIER 4 — AT DESIGN GATES ONLY:**
- **7. Edge Deployment Engineer** — At major milestones, before deployment claims.

**THE LOOP RULE:** Tier 1 reviewers (Correctness + Performance) run in a loop. If they find issues, fix them, then re-run the SAME reviewer. Repeat until the reviewer returns CLEAN. Do not assume one pass catches everything — fixing one issue often introduces another.

#### The 8 Reviewer Personas

1. **Correctness Engineer** — Bugs, edge cases, pipeline integrity, data leakage, off-by-one errors. "Does the code do what you think it does?" Reviews: training loop, eval pipeline, data loading, checkpoint save/load. **ALSO: Aggressive Entropy Auditor.** Every file in the repo starts at NEGATIVE valence — it must justify its existence through active utility or unique insight. The Correctness Engineer MUST systematically challenge every file's right to exist: code files, research docs, result JSONs, Codex output files, configs — everything. Flags: stale files, outdated artifacts, Codex outputs that should be ingested into RESEARCH.md and deleted, dead code, unused imports, temporary scripts, superseded research docs, old version files, completed probe code (results live in JSON), any file whose insights have been consolidated elsewhere. Files that can be consolidated with other positive-valence files should be merged. The repo must shrink or stay flat — never grow without justification. **Code is a liability. Every file is a liability. Git preserves history — deletion is safe and mandatory.**

2. **Performance Engineer** — Memory profiling, GPU/CPU utilization, throughput bottlenecks, batch sizing, framework overhead. "Will this OOM? What's the peak memory? Where's the bottleneck?" Must profile before any long-running job. Would have caught: bootstrap_iters=100K OOM, zombie process contention.

3. **Scaling Expert** — Deep analysis of what happens at different scales. Not just "does this transfer?" but thinking from every angle: What mechanisms will **break** at scale (e.g., Grokfast diverged at dim=768)? What mechanisms will **improve** at scale (e.g., attention patterns need more capacity)? What **emergent behaviors** might appear at 200M+ params that don't exist at 69M? What architectural choices **prevent** scaling (e.g., O(T²) operations, fixed memory sizes)? What choices **enable** scaling (e.g., modular stages, warm-start compatibility)? Thinks about: parameter efficiency curves, compute-optimal training (Chinchilla scaling), memory footprint at target scales, warm-start vs scratch tradeoffs at each scale jump, which killed mechanisms from Chrome should be **retested** at larger scale. Constructive, not just critical — flags both risks AND opportunities that emerge with scale. **Critical focus: asymmetric scaling** — what changes increase effective representational capacity WITHOUT adding parameters? More recurrent passes, better routing, richer stage interactions, improved memory utilization — these are "free" capacity gains from better geometry. The Scaling Expert should actively identify which architectural properties give asymmetric returns: "this mechanism costs 0 extra params but doubles effective depth" or "this routing change lets the same parameters express 3x more functions." This is the core Sutra thesis — Intelligence = Geometry means scaling the math, not the parameter count.

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

### Resource Management: GPU Priority + Secondary Experiments
**GPU is for production training. Everything else yields to it.**

When training is running:
- Chrome probes and secondary experiments run on **CPU only** (`CUDA_VISIBLE_DEVICES=""`)
- If CPU experiments interfere with training throughput, pause them immediately
- Never load giant tensors (>4GB) on CPU while training uses >20GB VRAM

When training is stopped (audit loops, checkpoints, shard operations):
- GPU is free for Chrome probes, quick experiments, benchmarks
- **Pause/resume**: all secondary experiments must be designed to save state and resume. Use checkpoint files, not in-memory state. When training needs to start, kill experiments → start training → restart experiments on CPU when training is stable.
- Run multiple CPU experiments in parallel when RAM allows (we have 68GB total)

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

### Mission Alignment Heartbeat (MANDATORY — Session Start + Every 60 Minutes)

At the start of every session and recurring every 60 minutes:

1. Read `research/VISION.md`, this `CLAUDE.md`, the global `~/.claude/CLAUDE.md` (additive — global rules ADD to project rules, never replaced by them), and all memory files
2. Schedule `/loop 60m` heartbeat for self-reflection alignment check
3. Each heartbeat checks:
   - Are active tasks serving the manifesto (Intelligence = Geometry, democratize AI)?
   - Are we repeating dead ends already logged in RESEARCH.md?
   - Should any Codex reviews be re-run (Tier 1 every code change, Tier 2 every 5K steps)?
   - Are there idle resources (GPU/CPU) we should be using?
   - Is the task list healthy — nothing stale, nothing missing, no gaps?
4. Output a brief alignment report and take corrective actions immediately
5. If Chrome isn't running experiments, START ONE. Zero idle time.

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

## Tesla+Leibniz Design Workflow (STRATEGIC ARCHITECTURE SESSIONS)

**For strategic architecture design — NOT for code review or correctness checks.**

Full workflow spec: `research/TESLA_LEIBNIZ_WORKFLOW.md`

Use this workflow when designing or redesigning architecture. It runs Codex as a committed senior architect (not a reviewer) through an iterative loop: internalize mission → question assumptions → identify gaps → research → design → iterate until >=9/10 confidence on all 5 outcomes.

**Key rules:**
- Every Codex session is FRESH (no resume — other sessions may run between rounds)
- Full persona + vision + outcomes injected EVERY round
- Claude autonomously executes research/probes between rounds
- Hardware availability is DYNAMIC (filled by Claude from live nvidia-smi each round)
- Codex's technical insights are valuable. Its strategic surrender is NOT.
- NEVER use the generic reviewer personas for strategic direction — they default to safety

---

## Codex Prompt Template (MANDATORY for non-Tesla/Leibniz sessions)

**Every non-Tesla/Leibniz Codex invocation (correctness, performance, etc.) MUST begin with this preamble. No exceptions.**

```
MANDATORY FIRST STEP: Read CLAUDE.md in this repository root. Every rule there is binding.
Then read research/VISION.md — especially the "Design Philosophy" section at the top.

CONTEXT YOU MUST KNOW:
- DESIGN PHILOSOPHY: Nothing is sacred except outcomes. Every design choice exists to
  enforce a specific outcome. If achievable better another way, CHANGE IT. Zero ego.
  The 5 sacred OUTCOMES (the mechanisms used to pursue them are negotiable):
    1. Genuine Intelligence — the model must be smart. Mechanism: state superposition.
    2. Improvability — find and fix failures surgically. Mechanism: modular stages.
    3. Democratized Development — community builds it like Linux. Mechanism: infrastructure vision.
    4. Data Efficiency — learn more from less. Mechanism: multi-teacher learning.
    5. Inference Efficiency — cheap to run, easy tokens exit early. Mechanism: elastic compute.
  If you can propose a BETTER mechanism for any outcome, propose it. Mechanisms are not sacred.
  Load-bearing CONSTRAINTS (the problem is real, solution negotiable): Scratchpad (precise
  retrieval bypassing recurrence signal loss), Lambda (halting pressure for elastic compute).
  See VISION.md for deep explanations of each outcome and mechanism.
- We are in a CHROME WORKFLOW: theory + experiments alternate.
  v0.6.0a (68.3M params, 12 recurrent passes, attached history) is training on GPU.
  Step 8700+, BPT monotonically improving. Critical finding: late passes (7-11) are MOST
  valuable (63% of BPT improvement). Sampled CE metric was MISLEADING — build_negative_set()
  creates selection bias from final-pass logits that INVERTS the quality signal.
- The architecture vision is a STAGE-SUPERPOSITION STATE MACHINE with 7 stages that
  form a state graph (not a pipeline). See research/VISION.md for full details.
- DO NOT recommend copying existing architectures (e.g., "just add attention layers").
  Every mechanism must be DERIVED from first principles (math, info theory, biology).
- CRITICAL: This is an INTEGRATED SYSTEM. All stages must COMPLEMENT each other,
  not compete. Each stage's output must be exactly what the next stage needs.
  Components must lift each other exponentially, not drag each other down with
  conflicting biases. Evaluate proposals for SYSTEM COHERENCE, not just per-stage quality.
- PERFORMANCE IS THE GATING CRITERION. Evaluate against best-in-class (Phi-4, Qwen3-4B,
  Gemma-3-1B), not strawmen. The goal: same or fewer params, fraction of the training data,
  fraction of the compute, matching or approaching the best in our parameter class.
- File management: NEVER create new files unless necessary. Edit existing. Delete outdated.
- All findings → research/RESEARCH.md. Raw ideas → research/SCRATCHPAD.md.

After reading CLAUDE.md and VISION.md, explore the repo structure to understand current state.

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
