# Sutra Vision: Modular Intelligence Infrastructure

## The Mission

**AI should be cheap, ubiquitous, and useful to the poorest person on the street — not just the richest corporation in the cloud.**

The current AI paradigm assumes Intelligence = Scale: more parameters, more data, more compute, more money. We believe Intelligence = Geometry: better mathematical structure, better information allocation, better representations. You don't need a data center to be intelligent. You just need better math.

Sutra exists to prove this. It is a from-scratch language model designed to match or beat models trained with vastly more resources — not by being clever about shrinking big models, but by starting from fundamentally better mathematics.

**What "winning" looks like concretely:** A model with the same or fewer parameters (e.g., 68M-4B), trained on a fraction of the data (e.g., 20B tokens vs 1-15T tokens), requiring a fraction of the compute to train (a single RTX 5090 vs clusters of thousands of GPUs), with similar or lower inference cost — that matches or comes close to the benchmark performance and generation quality of the best models in its parameter class. The narrative is NOT about a specific number beating another number. It is about demonstrating that mathematical insight can close a 100x-1000x resource gap, proving that powerful intelligence does not require massive scale.

**The baselines we measure against:** The best publicly available models in our parameter class — Phi-4 (5.6B), Qwen3-4B, Gemma-3-1B, SmolLM series, Ministral-3B. Not weak baselines, not old models. The current best. If we can't compete with them despite using orders of magnitude fewer resources to train, we haven't proven our thesis.

---

## Design Philosophy: Outcomes Over Implementations

**Nothing in this architecture is sacred except the outcomes it serves.**

Every design choice in Sutra exists because it enforces a specific outcome. If that outcome can be achieved better another way, the implementation MUST change. There is zero ego attached to any mechanism, any stage count, any routing strategy, any loss function. The creator of this project has explicitly stated: "I have no ego about this. Whatever we have to replace, however the system has to evolve, let it evolve."

The question is never "should we keep X?" — it is always **"what outcome does X enforce, and is there a better way to achieve it?"**

### The Gating Criterion: Performance

Everything is gated by performance. Benchmarks (MMLU, ARC, HellaSwag, GSM8K, HumanEval, WinoGrande, TruthfulQA). Generation quality (does it produce coherent, useful text?). Competitive positioning against the best models in our parameter class.

"Ultimately, no matter how good this is, if people can't use it, if people don't like it, it makes no difference."

If a mechanism is theoretically beautiful but doesn't improve performance, it goes. If an ugly hack improves performance, we study WHY it works and then derive the clean version. Theory serves performance, not the other way around.

---

## The 5 Outcomes: What We Actually Care About

These are the 5 real outcomes that define success for Sutra. They are the GOALS — the things we actually want to achieve. Below each outcome, we describe our current mechanism for achieving it. **The outcomes are sacred. The mechanisms are not.** If you can think of a better mechanism to achieve any of these outcomes, propose it immediately.

### Outcome 1: Genuine Intelligence

**What we actually want:** The model must be genuinely smart. It must understand context, reason about complex problems, generate high-quality coherent text, and handle diverse tasks competently. Without this, nothing else matters — no amount of efficiency or modularity helps if the model produces garbage. This is the gating criterion above all else: benchmarks (MMLU, ARC, HellaSwag, GSM8K, HumanEval, WinoGrande, TruthfulQA), generation quality, and competitive performance against the best models in our parameter class (Phi-4, Qwen3-4B, Gemma-3-1B).

**Why it's hard with limited resources:** Current frontier models achieve intelligence by throwing massive scale at the problem — trillions of tokens, billions of parameters, thousands of GPUs. We have a single GPU and a fraction of the data. To match their intelligence with our resources, the model must process information more efficiently. It needs to allocate its limited compute where it matters most — spending deep thought on hard problems and breezing past easy content. It needs to learn in a way that mirrors how thinking actually works: you don't apply the same amount of mental effort to every word you read.

**Our current mechanism — State Superposition:** We chose state superposition because we believe it models how intelligence actually operates. When you read a sentence, you don't process every word identically — "the" barely registers while "therefore" triggers deep reasoning that connects multiple premises. State superposition lets different tokens be at different processing stages simultaneously: one token might be gathering context while another is already verifying its output. Each position carries a probability distribution over stages (pi), and the model learns from data which processing path each type of content needs. This is like a factory assembly line where different products are at different stations simultaneously, versus a transformer where every token marches through the same layers in lockstep.

Currently implemented as 12 recurrent passes through shared parameters with stage probability vectors. The number 12, the parameter sharing, and the specific superposition mechanism are ALL negotiable. **If you can propose a mechanism that produces higher intelligence from the same parameters and data — whether it looks like superposition or not — propose it.**

**How to evaluate:** Full-vocabulary benchmarks, generation quality tests, and per-pass contribution analysis (using full-vocab metrics, NOT sampled metrics — we learned the hard way that sampled metrics can give inverted signals).

### Outcome 2: Improvability

**What we actually want:** When the model fails at something — when it hallucinates, when it can't do math, when its code generation is weak — we need to be able to identify WHAT is failing and fix it surgically, without breaking everything else. When we want to make the model better at reasoning, we should be able to improve reasoning without degrading language fluency. The model must be debuggable and incrementally improvable, not a black box where fixing one thing breaks three others.

**Why this matters for the mission:** Monolithic architectures (standard transformers, SSMs) are black boxes. If your model hallucinates, is it the attention mechanism? The feed-forward network? Layer 7? Layer 23? You can't tell. You retrain the whole thing and hope. This is enormously wasteful — it costs the same to fix one problem as to build the model from scratch. If we want intelligence to be cheap and accessible, the cost of IMPROVING intelligence must also be cheap.

Improvability is the difference between a system that requires billions to improve and one that a grad student with a good idea can make meaningfully better. Consider the current reality: if you want to improve GPT-4's reasoning, you need OpenAI's full training infrastructure, data, and budget. Nobody else CAN improve it. But if intelligence is decomposed into identifiable components with clear inputs and outputs, then a researcher who understands memory can improve the memory component, a researcher who understands verification can improve the verification component, and a researcher who has a brilliant new routing algorithm can swap it in and test it — all without needing access to the full system's training pipeline. Each improvement is scoped, testable, and composable.

This also applies to debugging. Right now when we discover a problem in Sutra (like the sampled CE metric inversion), we can trace it to a specific component (`build_negative_set()` in the loss computation) and fix it there. In a monolithic transformer, the equivalent problem would be buried somewhere in the interaction between 24 identical layers, and you'd have no idea which layer is responsible.

**Our current mechanism — Modular Stage Decomposition:** We break processing into named stages with clean interfaces. Each stage has a defined role (local construction, routing, memory, verification, etc.), and each stage can be studied, measured, replaced, or specialized independently. If memory retrieval is failing, we look at the Memory stage. If routing between positions is weak, we look at the Routing stage. The stages communicate through a uniform interface: (mu, lambda, pi) in → (mu_new, lambda_new) out. This makes stages independently swappable — like software modules with defined APIs.

Currently 7 stages. The number 7 emerged from design iteration, NOT from deep theory, and has no mathematical significance. Could be 5 or 10. The specific stage boundaries, inner loop structure, and naming are all negotiable. **What matters is that we CAN isolate and improve individual capabilities. If a different decomposition achieves better improvability, use it.**

### Outcome 3: Democratized Development

**What we actually want:** No single company or team should be the bottleneck for making the model better. Anyone — a university lab, a domain expert, a solo researcher, a startup in a developing country — should be able to take Sutra and make it better at what they care about, without needing to understand or retrain the whole system. Their improvements should compose with everyone else's improvements, so the collective intelligence of the community accumulates over time. Intelligence should be a public utility that a community builds together, not a proprietary moat controlled by one entity.

**Why this matters for the mission — the compounding effect:** This is the manifesto's most powerful lever, and it's worth explaining in full:

Right now, improving a language model is a centralized activity. OpenAI improves GPT. Google improves Gemini. Meta improves Llama. Each company independently builds the entire stack — data, architecture, training, evaluation — from scratch. The world's collective intelligence about how to build AI is fragmented across competing silos. If a brilliant researcher in Lagos discovers a better memory mechanism, they can't contribute it to any existing model. They'd have to build their own model from scratch just to test their idea.

Sutra's vision is fundamentally different. Imagine thousands of contributors worldwide, each improving the piece they understand best:
- A university NLP lab publishes a better Routing stage optimized for long-range dependencies in legal documents
- A medical AI startup publishes a specialized Memory stage enriched with clinical knowledge
- A student in India publishes a Verification stage tuned for mathematical proof checking
- A group in Brazil publishes a Segmentation stage optimized for Portuguese morphology
- A hardware lab publishes a quantization-friendly rewrite of the Compute Control stage

Because all stages follow the same interface contract, these improvements COMPOSE. You can download the Brazilian segmentation + the Indian verification + the medical memory and assemble a model that's better at medical text in Portuguese — and nobody had to build that specific combination from scratch. The improvements accumulate, like packages in a package manager or subsystems in Linux.

This is how Linux went from Linus Torvalds' hobby project to running 90%+ of the world's servers: thousands of contributors improving the pieces they understood, with a kernel that let improvements compose. The filesystem team didn't need to understand the network stack. The GPU driver team didn't need to understand the scheduler. Each contribution was scoped, testable, and independently valuable. We want the same for intelligence.

**The piggyback principle:** A solo researcher can't train a model on 15 trillion tokens — that requires Google-scale resources. But they CAN take a foundation Sutra model that the community has already trained, and improve ONE stage for their domain. They piggyback off everyone else's work (the base model, other contributors' stage improvements) and contribute their own piece back. The barrier to entry drops from "build the whole thing" to "improve one component." This is how you democratize intelligence development itself, not just intelligence access.

**Our current mechanism — Infrastructure Vision:** We design Sutra as a platform, not a model. Stage modules have uniform interfaces, making them independently downloadable and swappable. A medical company could replace the Memory stage with a medical-knowledge-enriched version, add domain verification, fine-tune only those modules on medical data, and contribute them back. This is impossible with transformers — you can't swap out "layer 12's memory" because transformers don't have named, interfaced stages.

The Linux analogy in detail:
- Linux kernel = Sutra's state graph + transition kernel (the core routing logic)
- Filesystem = Sutra's Memory stage (how information is stored and retrieved)
- Network stack = Sutra's Routing stage (how information flows between positions)
- Scheduler = Sutra's Compute Control (how processing time is allocated)
- Device drivers = Domain-specific stage specializations (medical, legal, code, etc.)

Currently partially realized — interfaces exist but community tooling does not. This is a long-term goal. **If a different architecture pattern enables better community extensibility, propose it.**

### Outcome 4: Data Efficiency

**What we actually want:** The model should extract maximum intelligence from minimum data. Current frontier models train on 1-15 trillion tokens because they learn only from raw text via next-token prediction. We want to learn from EVERYTHING — not just text, but also the knowledge already embedded in existing trained models, structural regularities from mathematics, insights from information theory, patterns from code execution, signals from symbolic reasoning systems, even patterns from biological neural connectivity. By absorbing knowledge from every available source, we should achieve the same intelligence with a fraction of the raw data.

**Why this matters for the mission:** One person with a laptop can't generate 15 trillion tokens of training data. But they CAN leverage the knowledge already embedded in publicly available models, mathematical structure, and open datasets. If the model can efficiently absorb knowledge from diverse sources instead of requiring massive raw text, the training cost drops dramatically. This is another lever for closing the resource gap — better learning, not just better architecture.

**What "teacher" means broadly:** Not just model distillation. A "teacher" is any source of structured knowledge: a pre-trained model's attention patterns, a symbolic reasoning engine's proof traces, information-theoretic bounds on optimal representations, code execution traces, human feedback signals, mathematical structure. Multi-teacher means learning from ALL of these simultaneously.

**Our current mechanism — Multi-Teacher Learning:** This pillar is currently the LEAST developed. We're training from scratch because warm-starting from v0.5.4 had technical issues (architecture changes made weight transfer non-trivial). We would prefer warm-start and multi-teacher learning. This is one of the highest-priority improvements once the current architecture stabilizes. **Any approach that improves knowledge absorption per training token is welcome — distillation, structured priors, curriculum design, synthetic data from existing models, whatever works.**

### Outcome 5: Inference Efficiency

**What we actually want:** The model should be cheap to run. Not just "fits on a phone" cheap — genuinely fast and energy-efficient. The key insight is that not all tokens are equally hard. The word "the" in a simple sentence is utterly predictable and needs almost zero thought. The word "therefore" connecting a complex logical argument needs deep multi-step reasoning. Right now, transformers spend the same compute on both. We want the model to autonomously decide, for each token, how much computation is needed — and stop early when the answer is obvious. If 60% of tokens are easy and can exit after 3 passes instead of 12, average inference cost drops ~40% with zero quality loss on the hard tokens that matter.

**Why this matters for the mission:** Inference cost determines who can use the model. If it costs $0.01 per query, it's accessible. If it costs $1.00, only corporations use it. Adaptive computation is the most direct lever for reducing cost: the same model becomes cheaper to run without any degradation where it matters. This enables deployment on edge devices (phones, laptops, embedded systems) and makes the model practical for resource-constrained settings — exactly the people the manifesto is about.

**What this requires:** A mechanism that (a) measures how "done" each token is, (b) decides when to stop computing, and (c) has natural pressure to stop early rather than always using maximum passes. The decision must be learned from data, not hardcoded, because "difficulty" is contextual — "bank" is easy in "river bank" but hard in a finance-law document.

**Our current mechanism — Elastic Compute via BayesianWrite + Lambda:** Each position carries a precision scalar (lambda) that tracks accumulated confidence. The BayesianWrite update rule blends new information proportionally: `mu_new = (lam*mu + alpha*m) / (lam + alpha)`. As lambda increases, new information has less impact — the representation converges. Lambda acts as a natural energy budget: it creates stopping pressure without arbitrary thresholds. This is LOAD-BEARING in the sense that elastic compute NEEDS some form of halting pressure — without it, the system would always use maximum passes. But the specific BayesianWrite formula is not sacred. **Any mechanism that creates learned, content-dependent halting pressure would work. If you can propose a better halting mechanism, propose it.**

---

## Design Choice Origins: Why Things Are the Way They Are

Every choice has a history. Some solve real problems (load-bearing). Others were experiments that happened to stick (negotiable). **Only load-bearing choices have earned their place. Everything else is on trial.**

### Load-Bearing Choices (solve real constraints — keep unless a better solution to the SAME constraint is found)

**Scratchpad (External Memory)**
- **What it is:** An external working memory buffer that positions can read from and write to, separate from the recurrent hidden state.
- **Why it exists:** State-space models (SSMs) and recurrent architectures lose signal over long contexts. Information written into the hidden state at position 10 may be degraded or lost by position 1000 due to the compressive nature of recurrence. The scratchpad provides a bypass — a position can write a precise value to memory and another position can retrieve it exactly, without it passing through hundreds of recurrent steps that could corrupt it.
- **The constraint it solves:** Precise retrieval over long distances in a recurrent architecture. This is a fundamental limitation of recurrence that attention solves (attention can look back directly), but attention is O(T²) in sequence length. The scratchpad gives attention-like retrieval at lower cost.
- **Load-bearing because:** Without this or something equivalent, the model would degrade on any task requiring precise recall of earlier information (copying, reference resolution, multi-step reasoning that refers back to earlier premises).

**Lambda / BayesianWrite (Energy Budget)**
- **What it is:** Each position carries a "precision" scalar (lambda) that tracks accumulated confidence. The BayesianWrite update blends new information proportionally: high lambda = the model is already confident, so new info has less impact. Low lambda = the model is uncertain, so new info has more impact.
- **Why it exists:** Without some form of budget pressure, an elastic compute system has no reason to ever STOP computing. If more passes always help, why not run 100 passes? 1000? Lambda acts as a convergence signal — as precision increases, the marginal value of additional computation decreases, creating natural stopping pressure. It also serves as an information-theoretic zoom: high-lambda positions are "zoomed in" and precisely specified, while low-lambda positions are still coarse and need more work.
- **The constraint it solves:** Elastic compute requires a halting mechanism. Something must tell the compute controller "this token is done, stop wasting cycles on it." Lambda provides this signal through the math of precision-weighted averaging — it's not an arbitrary threshold but an emergent property of how information accumulates.
- **Load-bearing because:** Without budget pressure, elastic compute degenerates to "always use maximum passes" (wasting compute on easy tokens) or requires an arbitrary external threshold (fragile, not learned).

### Negotiable Choices (historical, on trial — replace if something better achieves the same outcome)

**Shared Parameters Across Passes**
- **What it is:** All 12 recurrent passes use the same weight matrices.
- **Why it was chosen:** Seemed natural for superposition — if stages are just probability distributions over processing, the same parameters can implement all stages. Also keeps parameter count low (68M instead of 68M × 12).
- **Why it's negotiable:** Partially shared parameters (e.g., shared core + per-pass adapters) or adaptive parameter routing could give each pass specialized behavior while keeping parameter count manageable. The outcome (parameter efficiency + stage flexibility) might be better served by a hybrid approach.

**Pheromone Routing**
- **What it is:** A bio-inspired mechanism for cross-position information flow, modeled after how ants leave chemical trails that influence other ants' paths.
- **Why it was chosen:** Collective intelligence experiment — an exploration of whether swarm-like coordination could be more efficient than attention for routing information between positions.
- **Why it's negotiable:** It's an experiment, not a proven winner. Any mechanism that achieves efficient cross-position information flow (sparse attention, linear attention, state-space convolutions, locality-sensitive hashing, etc.) would serve the same outcome.

**7 Stages**
- **What it is:** The current decomposition of processing into 7 named stages.
- **Why it was chosen:** Emerged from multiple rounds of design iteration and Codex debates, not from deep theory. The number 7 has no mathematical significance.
- **Why it's negotiable:** Could be 5 (merge related stages) or 10 (split complex stages). The outcome (modular decomposition) is independent of the exact count. What matters is that stages have clean interfaces and meaningful semantic boundaries, not that there are exactly 7 of them.

**From-Scratch Training**
- **What it is:** The model trains from randomly initialized weights, not from a pre-trained checkpoint.
- **Why it was chosen:** NOT by choice. Warm-starting from the previous version (v0.5.4) had technical issues (architecture changes made weight transfer non-trivial). We would prefer warm-start.
- **Why it's negotiable:** Multi-teacher learning (Pillar 4) explicitly calls for absorbing knowledge from existing sources. From-scratch training is a temporary constraint, not a design decision.

---

## The Core Thesis

Sutra is NOT a language model. It is a **modular intelligence infrastructure** where computation flows through a state graph of independently improvable processing stages.

Every existing AI architecture (transformers, SSMs, hybrids) is monolithic: you can't improve the memory system without retraining the whole model. You can't swap out the routing mechanism. You can't let domain experts improve just the verification stage for their field.

Sutra changes this. Each position in a sequence carries a **probability distribution over processing stages**, and the model's computation is driven by **content-dependent transitions** on a state graph. The stages are independent modules with clean interfaces. Anyone can replace, subdivide, or specialize any stage without touching the rest.

## The Stage-Superposition State Machine

### What It Is

At every position in the sequence, the model maintains three quantities:
- **mu** (features): A vector representing what the model currently "knows" about this position — its semantic content, context, and role in the sequence. This is analogous to a hidden state in an RNN, but it evolves through named stages rather than generic layers.
- **lambda** (precision): A scalar measuring how confident/converged the model's representation of this position is. High lambda means "I've processed this enough, the representation is stable." Low lambda means "this position still needs work." This drives elastic compute decisions.
- **pi** (stage probabilities): A probability distribution over the 7 processing stages. This tells you WHAT the model is currently doing at this position — is it routing information? Writing to memory? Verifying output? Different positions have different pi distributions, which is what makes this a superposition.

The key insight: **different positions can be at different stages simultaneously**. One position might be 80% in the routing stage (gathering context from elsewhere in the sequence) while another position is 90% in the verify stage (checking its output against expectations). This is fundamentally different from a transformer, where every token goes through the same layers in lockstep.

Think of it like a factory assembly line where different products are at different stations simultaneously, versus a batch process where every product goes through every station in the same order. The factory is more efficient because each product gets exactly the processing it needs.

### Content-Dependent Transitions

The transition between stages is NOT fixed. A **learned transition kernel** (Markov matrix) depends on:
- The current hidden state (what the position contains)
- The precision (how confident the position is)
- The verification score (did the last readout attempt succeed?)

This means different types of content naturally follow different processing paths:
- **Simple prose** follows a fast-write-early-verify path (simple local structure, few passes needed)
- **Complex code** follows a more-routing-late-verify path (complex long-range dependencies, needs many passes)
- **Mathematical reasoning** would follow a heavy-compute-control path (needs multiple reasoning steps, iterative refinement)

The model discovers these strategies FROM DATA. We don't hardcode them. This is the superposition pillar in action.

## The Infrastructure Vision in Detail

### Why Infrastructure, Not a Model

A model is a fixed artifact you deploy. Infrastructure is something people build on.

**Linux analogy (in detail):**
- Linux kernel = Sutra's state graph + transition kernel (the core routing logic that decides what processing happens when)
- Filesystem = Sutra's Memory stage (Stage 5) (how information is stored, organized, and retrieved)
- Network stack = Sutra's Routing stage (Stage 4) (how information flows between different parts of the system)
- Scheduler = Sutra's Compute Control (Stage 6) (how processing time and resources are allocated)

Just as different teams improve different parts of Linux independently (the filesystem team doesn't need to understand the network stack, the scheduler team doesn't need to understand the GPU driver), different teams can improve different stages of Sutra independently.

### How Modularity Works

Every stage module follows the same interface contract:
```
Input:  (mu, lambda, pi, context)  →  Output: (mu_new, lambda_new)
```

This uniform interface means:
1. **Replace** any stage module with a better one — swap in a new routing mechanism without touching memory or verification
2. **Subdivide** any stage into a sub-graph of specialists — split Memory into short-term, long-term, episodic, domain-specific
3. **Specialize** any stage for a specific domain — medical verification, legal reasoning, code analysis
4. **Compose** improvements from different contributors — one team's better routing + another team's better memory = better overall model

### Future: Hierarchical Stages

Each stage evolves from a single module into a sub-graph of specialists:

```
Stage 4 (Route) → {
    local_route:   handles within-paragraph dependencies
    global_route:  handles cross-paragraph dependencies
    domain_route:  domain-specific routing patterns
}

Stage 5 (Memory) → {
    short_term:    working memory for current context
    long_term:     consolidated knowledge
    episodic:      specific event memory
    domain:        domain-specific memory (legal, medical, code...)
}

Stage 7 (Verify) → {
    syntax_verify:   grammatical correctness
    semantic_verify:  meaning consistency
    domain_verify:    domain-specific validation
    factual_verify:   factual accuracy checking
}
```

The transition kernel routes to sub-stages WITHIN each stage group. The interface stays the same. Everything composes.

### How Someone Improves Sutra for Their Domain

Example: A medical AI company wants better medical reasoning.

1. They download the base Sutra model
2. They replace `Stage 5: Memory` with a medical-knowledge-enriched version
3. They add `domain_verify` to Stage 7 that checks medical accuracy
4. They fine-tune ONLY these new modules on medical data
5. Everything else (routing, compute control, base language) stays from the foundation model
6. They contribute their medical modules back to the ecosystem

This is **impossible** with transformers. You can't swap out "layer 12's memory mechanism" because transformers don't have named, interfaced stages.

## Why This Matters

### For Efficiency (The Manifesto)
Modular architecture means you only train what you need. A medical company doesn't retrain the whole 4B-parameter model — just the medical modules. Elastic compute means you don't waste cycles on easy tokens. Multi-teacher learning means you don't need 15 trillion tokens of training data. Everything compounds to make powerful AI accessible on less compute.

### For Science
By separating processing into named stages with measurable behaviors, we can STUDY how intelligence works. Which stages activate for math? For creative writing? For code? How does the transition kernel route different types of content? This is interpretable by design, not as an afterthought.

### For the Community
Open-source Sutra with clear "how to add a stage" documentation. The community improves specific stages. Improvements compose. The system grows organically, like Linux. Intelligence becomes a public utility, not a proprietary moat.

### For the Paradigm Shift
Every other architecture asks: "how many layers?" We ask: "what stages of processing does this content need?" That's a fundamentally different question, and it leads to fundamentally different capabilities.

---

## What This Means for Every Decision

These rules govern every architectural choice, every experiment design, every Codex review, every heartbeat self-check:

1. **NEVER assume a design choice is sacred.** Question everything. The only sacred things are the 5 outcomes (intelligence, improvability, democratization, data efficiency, inference efficiency) — not the mechanisms we currently use to pursue them (superposition, stages, infrastructure pattern, multi-teacher, elastic compute). If you can propose a better mechanism for any outcome, do it.
2. **Always ask: "What OUTCOME does this enforce?"** If the outcome can be achieved better another way, change the implementation immediately.
3. **Run eval tasks constantly to challenge assumptions, not reinforce bias.** The sampled CE metric inversion (where our per-pass metric was giving the OPPOSITE signal from reality) is a permanent reminder: metrics can lie. Always validate with multiple independent measurements.
4. **Evaluate against the best-in-class, not convenient baselines.** We compete against Phi-4, Qwen3-4B, Gemma-3-1B — the current state of the art in our parameter class. Winning against weak baselines proves nothing.
5. **The manifesto IS the mission.** Intelligence = Geometry. Democratize AI. Every choice serves this or it goes.
6. **Performance gates everything.** Beautiful theory that doesn't improve benchmarks or generation quality is not worth keeping. Performance is not a secondary concern — it is THE concern.

---

## Current Status (2026-03-22)

### v0.6.0a Training (COMPLETE — 20K steps)
- 68.3M params, 12 recurrent passes, attached inter-step history
- **Best BPT: 6.7946 at step 20K** (new best, beat 17K's 6.8284)
- Training STOPPED at 20K: 655M tokens (48% of Chinchilla-optimal 1.36B for 68M)
- Benchmarks: SciQ 48.1%, PIQA 54.5%, LAMBADA 11.2%, ARC-Easy 31.3%
- Throughput ~6927 tok/s on a single RTX 5090
- **Next: v0.6.0b-rd12** (random-depth warm-start to fix pass collapse)

### Critical Discovery: PASS COLLAPSE (structural)
- Pass 11 alone = 61.6% of all BPT improvement. Passes 0-7 = 6.4%.
- Logit entropy flat 10.1-10.4 for passes 1-11, crashes to 5.84 at pass 12
- cos(pass 11, pass 12) = 0.293 (near-orthogonal) — final pass applies massive transformation
- This pattern is STABLE from 14K to 20K — structural, not training dynamics
- The model is effectively a 1-pass system with 11 wasted passes
- **Fix: v0.6.0b-rd12** (random-depth training forces useful output at every pass)
- **Historical note**: Earlier "late-pass value" finding was CORRECT but understated — the issue isn't that late passes are valuable (they are) but that ONLY the final pass matters (pass collapse).

### What Works (Validated)
- Stage-superposition: positions flow through stages at their own rate ✓
- Content-dependent transitions: prose vs code follow different paths ✓
- 12-pass recurrence: ALL passes contribute, late passes most valuable ✓
- Attached history: +29% better than detached (validated at dim=128) ✓
- 3-part loss structure: L_final + L_step + L_probe (L_step provides gradient flow) ✓
- Competitive learning efficiency vs Pythia at equivalent steps ✓

### Active Chrome Investigations
- **L_step_exact**: Replace sampled CE loss with full-vocab CE on late passes (correct gradient + gradient flow)
- **Dense baseline**: Matched comparison (same tokenizer, data, params, plain decoder) — needed to honestly assess whether the architecture itself adds value beyond the training setup
- **Elastic compute**: Freeze at optimal pass per token (gated behind L_step fix)

### Dead Ends (Valuable Negative Results)
| Probe | Result | Lesson |
|-------|--------|--------|
| L_regret | KILLED — solved non-existent problem | Sampled CE artifact, not real degradation |
| L_step=0 | KILLED — lost gradient flow | L_step helps via gradient flow, not gradient direction |
| Grokfast | KILLED — overfits at dim=768 | Only training tricks work at 67M params, architecture changes need 200M+ |
| Syndrome scratchpad | KILLED — no signal (rho=-0.002) | Architecture-scale mechanisms need larger models |
| Resonant write dither | KILLED — 0% effect | Noise injection negligible at this scale |
| NCA warm-start | KILLED — 0% BPT improvement | NCA is an init-only benefit, not an ongoing one |

### The Bet
This paradigm either works — positions naturally specialize their computation based on content, stages become independently improvable modules, and the system outperforms monolithic architectures at equivalent scale — or it doesn't.

Blue Lock: all or nothing. We sink this paradigm or die trying.
