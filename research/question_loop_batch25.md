# Q-Loop B25: PCCP Adversarial Stress Test

**Date:** 2026-07-07
**Role:** Question Loop worker
**Iterations:** I169-I175
**Status:** analysis-only adversarial stress test; CPU-only constraint; no model, dataset, GPU, or implementation runs; prior-art reading used only for conceptual comparison.

---

## Grounding

Read in the requested order:

1. `research/VISION.md`
2. `research/question_loop_batch24.md`
3. `research/dual_loop_supervisor_checkin_15.md`
4. `research/DEEP_RETHINK.md`
5. `research/STATUS.md`

Additional process grounding:

- `C:\Users\devan\.claude\projects\C--Users-devan-OneDrive-Desktop-Projects-AI-Moonshots-moonshot-sutra\memory\MEMORY.md`

Binding facts:

- The five sacred outcomes are fixed: genuine intelligence, improvability, democratized development, data efficiency, and inference efficiency.
- The substrate is open. Neural, symbolic, programmatic, proof-based, analog, active-inference, reservoir, category-theoretic, and hybrid systems must be evaluated on equal footing.
- The current repo source of truth says there is no live moonshot mechanism. B24 proposed PCCP; supervisor #15 accepted it as live candidate but ordered hostile attack.
- The kill history's central result is proxy/function divergence: repeated systems improved visible metrics while failing the actual function.
- The B24 anti-NN bias must be corrected. Criterion (f) is not "non-neural." It is whether the direction serves the five outcomes better than alternatives.
- PCCP-A means synthesis under a given verifier. PCCP-B means verifier/spec discovery or refinement. PCCP-A is clean but risks being old synthesis. PCCP-B is moonshot-shaped but may be circular or too hard.
- CPU-only means theory, formalization, precommit design, hostile decomposition, source comparison, and cheap falsifier design. No training.

Prior-art anchors checked for this batch:

| Anchor | Why it matters for PCCP |
|---|---|
| CEGIS / OGIS | Counterexample-guided synthesis already has candidate generation, formal verifier, counterexamples, and finite-space guarantees. See Jha and Seshia's CEGIS analysis and formal-synthesis framework: <https://arxiv.org/abs/1407.5397>, <https://arxiv.org/abs/1505.03953>. |
| ILP | Learns logic programs from positive/negative examples and background knowledge; modern ILP has predicate invention, recursion, compression, and neuro-symbolic variants. See Cropper and Dumancic: <https://arxiv.org/abs/2008.07912>, <https://arxiv.org/abs/2102.10556>. |
| DreamCoder | Learns programs and reusable abstractions with wake-sleep library learning and neural-guided search. See Ellis et al.: <https://arxiv.org/abs/2006.08381>. |
| Symbolic regression / AI Feynman | Searches compact executable expressions, exploits symmetry, separability, modularity, complexity/accuracy tradeoffs. See Udrescu and Tegmark: <https://arxiv.org/abs/1905.11481>, <https://arxiv.org/abs/2006.10782>. |
| Proof-carrying code | Executable code carrying a proof is an old formal-methods idea. PCCP cannot pretend the proof-carrying part is novel. See Necula/PCC references summarized at <https://en.wikipedia.org/wiki/Proof-carrying_code>. |
| Causal discovery / abstraction | Causal graphs, structural causal models, interventions, and abstraction already cover much of "causal program" vocabulary. See Pearl and causal abstraction work: <http://bayes.cs.ucla.edu/BOOK-2K/>, <https://arxiv.org/abs/1812.03789>, <https://arxiv.org/abs/1906.11583>. |
| Invariant/specification mining | Verifier discovery overlaps with dynamic invariant detection, spec mining, and ICE/Horn-ICE invariant learning. See Daikon and Horn-ICE: <https://plse.cs.washington.edu/daikon/>, <https://arxiv.org/abs/1712.09418>. |
| Exact query learning | Verifier discovery also overlaps with Angluin-style membership/equivalence query learning and counterexample-driven concept learning. See Angluin's L* tradition summarized at <https://en.wikipedia.org/wiki/Dana_Angluin>. |
| Computational mechanics | Minimal predictive causal states and epsilon-machines are directly relevant to CWC-E. See Shalizi and Crutchfield: <https://arxiv.org/abs/cond-mat/9907176>. |
| AIXI / algorithmic information | The shortest executable explanation plus decision theory is not new in principle. See Hutter and AIXI: <https://arxiv.org/abs/1202.6153>, <https://arxiv.org/abs/0909.0801>. |
| Active inference / free energy | A competing broad intelligence frame centered on generative models, action, uncertainty, and self-evidencing. See Friston-oriented summaries: <https://arxiv.org/abs/2201.06387>, <https://arxiv.org/abs/2207.06415>. |
| Category-theoretic AI | Composition/invariance/logical architecture language that may support, not replace, an engine. See categorical deep learning and CT/ML surveys: <https://arxiv.org/abs/2402.15332>, <https://arxiv.org/abs/2106.07032>. |
| Reservoir / analog / cellular automata | Competing substrate ideas for cheap dynamics, physical computation, and simple rules. See reservoir and analog anchors: <https://arxiv.org/abs/1706.00280>, <https://arxiv.org/abs/2302.06417>, and computational irreducibility/coarse-graining: <https://arxiv.org/abs/nlin/0309047>. |

Current strongest PCCP version to attack:

```text
PCCP-H: a proof-carrying causal program stack where neural or non-neural
front-ends may propose symbols, features, candidates, or perceptual parses;
the core intelligence claim is that durable knowledge becomes compact
executable structure with public proof/test obligations, counterexample-driven
repair, and eventually learned or refined verifiers.
```

This is stronger than B24's pure framing because it removes anti-NN bias:

```text
Goal is sacred. Method is not.
```

---

## I169: Prior Art Absorption

### Steelman

The strongest PCCP claim after supervisor correction is not:

```text
We invented counterexample-guided program synthesis.
```

That would be false.

The strongest claim is:

```text
The unit of cheap intelligence should be an executable causal hypothesis whose
correctness obligation travels with it, whose failures are converted into
counterexamples, and whose repair is local.
```

In this steelman, PCCP is not a single algorithm. It is a research program that tries to fuse five commitments into one artifact:

| Commitment | PCCP interpretation |
|---|---|
| Functional target | The system is judged against the function, not BPB, NLL, reconstruction, hidden cosine, or other proxies. |
| Executable knowledge | Knowledge is a program/rule/procedure, not only weights, embeddings, text, or a score. |
| Causal robustness | The artifact must survive interventions and transformations, not just iid examples. |
| Proof/test obligations | The artifact carries verifiable reasons to trust it within a stated domain. |
| Local repair | Failures produce counterexamples that identify the smallest patchable unit. |

The best defense against prior-art absorption is that PCCP is a synthesis-level principle:

```text
Cheap intelligence is function-preserving executable compression with carried
verification and local repair.
```

CEGIS gives the loop. ILP gives logic-rule induction. DreamCoder gives abstraction/library learning. Symbolic regression gives compact executable equations. Causal discovery gives intervention semantics. Proof-carrying code gives proof obligations. Spec mining gives a path toward verifier discovery. PCCP tries to put these in one outcome-first frame, with explicit smuggling controls and public artifact discipline.

If this is allowed as a paradigm, the novelty is not a primitive algorithm. It is a demand about what learned knowledge must be:

```text
not a latent, not a loss curve, not a benchmark score, but an executable,
causally stress-tested, proof/test-carrying object.
```

### Attack

The hostile expert response is brutal:

```text
This is rebranding.
```

CEGIS already has a candidate program space, a verifier, a loop that returns counterexamples, iterative repair through accumulated examples, and formal analysis of finite-space termination and oracle/counterexample variants. PCCP-A, under a given verifier, is therefore almost exactly CEGIS with different marketing language.

SyGuS already has logical specifications, grammar-constrained search spaces, SMT-style verification, and explicit syntax restrictions that are basically the PCCP DSL. PCCP's "DSL/search space with smuggling controls" is just SyGuS honesty.

ILP already has induction of logic programs from examples, background knowledge, positive and negative constraints, predicate invention, recursion, compression-guided search, and interpretable hypotheses. PCCP's causal rules are not meaningfully beyond ILP unless it adds intervention semantics, proof-carrying outputs, and repair metrics in a way ILP baselines cannot already imitate.

DreamCoder already has program induction, learned libraries, reusable abstractions, neural guidance, transfer across tasks, and interpretable learned concepts. The PCCP "shortest executable structure" story strongly overlaps with DreamCoder's language-learning and library-compression story. If DreamCoder is extended with property tests and causal task families, it may become PCCP without changing its core.

Symbolic regression already has compact expressions, complexity-accuracy tradeoff, symmetry and separability tests, recursive decomposition, and executable formulas that generalize from few data points. For numeric causal laws, PCCP risks being symbolic regression plus a verifier.

Causal discovery already has causal graphs, structural equations, interventions, counterfactuals, identifiability conditions, and abstraction across levels. For causal worlds, PCCP risks being structural causal model discovery plus program synthesis.

Proof-carrying code already has code paired with a proof, cheap proof checking by a consumer, explicit safety policies, and separation of producer and checker. The "proof-carrying" part is not a new intelligence principle. It is a known formal-methods packaging idea.

Verifier discovery already has invariant inference, spec mining, model learning from membership/equivalence queries, ICE/Horn-ICE invariant learning, property-based testing, and metamorphic relation discovery.

The hostile conclusion:

```text
Every technical organ in PCCP has already been grown elsewhere. PCCP is a
collage, and a collage is not a paradigm shift.
```

The one possible escape is to name a principle that the prior art does not already own.

Candidate principle:

```text
Function-preserving executable causal compression with proof-carrying local
repair is the right unit of learned intelligence.
```

But that is still not obviously new. It sounds like:

```text
MDL + CEGIS + causal invariance + proof-carrying code + ILP.
```

The prior-art attack kills three weaker claims:

1. **PCCP is a new algorithm.** Killed.
2. **PCCP-A is a moonshot by itself.** Killed.
3. **Proof-carrying executable knowledge is new because LLMs do not do it.** Killed. Formal methods do it.

What remains possible:

```text
PCCP may be a useful research doctrine: evaluate intelligence artifacts by
function-aligned executable compression, causal intervention survival,
verifier transparency, and repair locality.
```

That is less glamorous than a paradigm.

### New Hardest Objection

The hardest new objection is:

```text
If an existing CEGIS/ILP/DreamCoder/symbolic-regression system can be made
"PCCP" by adding causal test families and proof logs, then PCCP is not a new
direction. It is a benchmark/evaluation wrapper around prior art.
```

This is not fatal if the repo's goal is to build a winning artifact, not coin an acronym. But it is fatal to the narrative "we found a new principle" unless the precommit spec forces a separation from those baselines.

The separation must not be philosophical. It must be operational:

- PCCP must beat strong prior-art baselines in their own favorable zones.
- Or PCCP must prove a theorem those baselines do not target.
- Or PCCP must produce a distinct artifact: a causal program plus check obligations plus counterexample-localized repair trace plus hidden-intervention survival, under a human-labor accounting that prior systems fail.

### Verdict + Next-Gate Ranking

Verdict:

```text
PCCP-A is absorbed by prior art unless framed as an outcome-first benchmark
discipline. PCCP-B/verifier discovery is where novelty could live, but it also
overlaps with spec mining, invariant learning, and exact query learning.
```

Next-gate ranking after I169:

| Rank | Direction | Update |
|---:|---|---|
| 1 | PCCP-H: hybrid proof-carrying causal core | Survives only as integrated artifact discipline, not as new primitive algorithm. |
| 2 | Verifier discovery under restricted classes | Most moonshot-shaped, but must be compared to spec mining and invariant learning. |
| 3 | Existing CEGIS/ILP/DreamCoder + causal/proof stress suite | Becomes a serious replacement if it matches PCCP with less novelty theater. |
| 4 | CWC-E / computational mechanics support | Useful theory of causal compression; not enough as synthesis engine. |
| 5 | Pure PCCP-A | Demoted: likely "just synthesis under a given verifier." |

---

## I170: Narrowness Attack

### Steelman

The fair steelman is:

```text
PCCP should not claim to cover all human intelligence first. It should claim
that many high-value intelligent acts have a verifier-rich core, and that
cheap intelligence begins where function can be checked, repaired, and
compiled.
```

Many real workflows do have verifier-rich substructure:

- code can be type-checked, unit-tested, fuzzed, statically analyzed, and reviewed against specs;
- math can be proof-checked;
- data transformations can be schema-checked;
- planning can be simulator-checked under constraints;
- scientific hypotheses can be tested against experiments;
- causal claims can be attacked by interventions and counterexamples;
- legal or policy work can be checked against citations, jurisdiction, contradiction, and rule applicability, even if not fully automated;
- medical triage can be checked against guidelines, contraindications, lab ranges, and risk calculators, even if final judgment stays human.

The strongest decomposition story:

```text
Open-world intelligence is not one verifier. It is a lattice of partial
verifiers, evidence obligations, causal checks, uncertainty flags, and residual
judgment.
```

PCCP does not need a total verifier for "be wise." It can still be useful if it turns a messy task into perception, evidence gathering, candidate generation, local formal checks, causal consistency checks, explicit uncertainty, human-readable residual assumptions, and compiled procedures where correctness is crisp.

Under this interpretation, PCCP is not a replacement for all cognition. It is the verifier/checkable spine inside a broader intelligence stack.

### Attack

Now list 20 tasks humans consider intelligence and ask what PCCP can actually do.

| # | Task humans call intelligence | Pure PCCP-A fit | PCCP-H fit | Hard failure mode |
|---:|---|---|---|---|
| 1 | Recognize objects in raw images | Low | Medium with neural perception | Symbols are not given; verifier rarely labels every visual nuance. |
| 2 | Understand speech in noise | Low | Medium with neural/audio front-end | Perception is continuous, ambiguous, and data-heavy. |
| 3 | Hold open-ended conversation | Low | Medium | Correctness is contextual, social, and under-specified. |
| 4 | Translate idiomatic language | Low-medium | Medium-high | Verifiers catch grammar/meaning only partially; cultural nuance leaks. |
| 5 | Prove a theorem | High | High | Excellent fit if formal statement and proof checker exist. |
| 6 | Write/debug code | High | High | Strong fit for verifier-rich parts; weak for product taste and unclear requirements. |
| 7 | Diagnose a software production incident | Medium | High | Logs/tests help; causal hypothesis search is good; missing context remains hard. |
| 8 | Plan a route or schedule | Medium-high | High | Works if constraints are explicit; preferences and disruptions complicate. |
| 9 | Navigate a robot in a kitchen | Low | Medium | Perception, physics, affordances, and safety are open-world. |
| 10 | Make a medical diagnosis | Low-medium | Medium | Guidelines/labs help; causal uncertainty and liability prevent total verification. |
| 11 | Judge legal relevance | Medium | Medium-high | Citations and rules help; interpretation and jurisdictional ambiguity remain. |
| 12 | Discover a physics law | Medium-high | High in controlled settings | Symbolic regression already covers much; experimental design is hard. |
| 13 | Design a new experiment | Medium | Medium-high | Can optimize constraints; scientific taste and unknown unknowns remain. |
| 14 | Negotiate with a human | Low | Low-medium | Objectives, deception, emotion, and norms resist crisp verifiers. |
| 15 | Comfort a grieving person | Very low | Low | This is social attunement, not checkable rule execution. |
| 16 | Write a moving novel | Very low | Low-medium | Aesthetic value has no clean verifier; local grammar checks miss the function. |
| 17 | Compose music | Low | Medium | Formal constraints help; artistic success is subjective. |
| 18 | Infer someone's hidden motive | Low | Medium | Evidence is partial, adversarial, and unverifiable. |
| 19 | Learn a new board game from rules | High | High | Excellent if rules are formal and game states enumerable. |
| 20 | Form a new concept from few examples | Medium | Medium-high | Good in typed domains; weak for fuzzy natural categories. |

The table is not flattering.

Pure PCCP-A is strong on theorem proving, code under tests/specs, formal games, exact data transformations, constrained planning, and controlled scientific-law worlds. It is weak on raw perception, aesthetics, social reasoning, emotional intelligence, ambiguous language, open-world robotics, moral judgment, and commonsense in underspecified situations.

By unweighted task count, pure PCCP-A is strong on perhaps 5-7 of the 20. PCCP-H is plausible on more, but only because neural perception, retrieval, simulation, and human feedback absorb the messy parts.

The decomposition story can succeed in concrete cases:

| Open task | Decomposition that works | Why it works |
|---|---|---|
| Fix a failing PR | Natural language issue -> code search -> candidate patch -> tests/types/fuzz/static analysis -> review trace | Verifiers already exist and failures are local. |
| Transform messy CSV to target schema | Infer columns -> propose transform -> schema checks -> row-level counterexamples -> compiled script | Target function is explicit enough. |
| Generate SQL from a dashboard request | Parse intent -> candidate SQL -> run on sandbox -> schema/type/value checks -> compare expected slices | Execution exposes errors. |
| Solve a puzzle/game | Formal state -> rules -> search/program -> verifier -> counterexample states | Closed world and exact rules. |
| Fit a physics equation | Data -> symbolic expression -> dimensional/symmetry checks -> held-out experiments | Strong when the true law is compact. |

But the decomposition fails or becomes hand-waving in equally concrete cases:

| Open task | Why decomposition fails |
|---|---|
| "Is this essay persuasive?" | The target function is audience-dependent and aesthetic; local grammar/fact checks are not the function. |
| "Should we launch this product?" | Market, timing, ethics, distribution, and taste do not collapse into stable verifiers. |
| "Does this patient have rare disease X?" | Partial tests exist, but the causal graph is incomplete and errors carry high human cost. |
| "What did this person mean?" | Ambiguity and social context are not bugs; they are the task. |
| "Make a child feel safe" | Success is relational and dynamic; a proof artifact may be irrelevant or harmful. |

The hostile expert says:

```text
PCCP is a formal-intelligence engine. Human intelligence is mostly not formal.
```

The strongest version of this attack is not that formal tasks are useless. They are very useful. The attack is that the moonshot is "democratize intelligence," not "build an excellent symbolic synthesis workbench." If PCCP covers only domains with crisp validators, it may be a powerful tool but not a paradigm for intelligence.

The narrowness attack also exposes a hidden labor problem:

```text
Decomposition into verifier-rich subproblems may itself require intelligence.
```

If a human must decide the right subproblems, write the verifier, choose the DSL, determine which residual judgment is acceptable, and define the intervention families, then PCCP has not democratized intelligence. It has made expert-designed pipelines easier to run.

### New Hardest Objection

The hardest new objection is:

```text
"Decompose open-world problems into verifier-rich subproblems" is not yet an
algorithm. It is a hope that the hard part of intelligence can be manually
factored into easy parts.
```

This objection is stronger than prior-art absorption because it attacks the scope of the vision. Even if PCCP is technically excellent, it may cover the wrong fraction of intelligence.

To survive, PCCP needs a decomposition gate:

```text
Given a messy task, the system must itself propose partial verifiers,
assumption boundaries, residual uncertainty, and repairable subclaims, then
show that this improves function better than a neural-only or human-written
pipeline.
```

### Verdict + Next-Gate Ranking

Verdict:

```text
Pure PCCP is too narrow for the full vision. PCCP-H survives only as a
verifier-centered core inside a broader architecture, and only if task
decomposition becomes an explicit research target instead of a slogan.
```

Next-gate ranking after I170:

| Rank | Direction | Update |
|---:|---|---|
| 1 | PCCP-H with explicit decomposition/verifier proposal | Best chance to handle non-formal workflows without pretending they are fully formal. |
| 2 | Neural + tools + verifiers | Strong baseline; may already solve much of the decomposition problem pragmatically. |
| 3 | PCCP-A formal core | Valuable but too narrow to carry the moonshot alone. |
| 4 | Verifier discovery | Needed for scope expansion, but still underspecified. |
| 5 | Pure neural open-world model | Broad but weak on repair, proof, data efficiency, and democratized internals. |

---

## I171: Balanced Hybrid Evaluation

### Steelman

The strongest fair version is no longer pure PCCP. It is:

```text
neural perception/proposal front-end
-> PCCP reasoning and verifier core
-> compiled proof-carrying output
-> counterexample repair loop
```

This hybrid respects the supervisor correction:

```text
Neural components are not penalized for being neural. They are penalized only
if they weaken the five sacred outcomes relative to alternatives.
```

A hybrid can assign substrates by strength:

| Layer | Best substrate candidate | Why |
|---|---|---|
| Raw perception | Neural / reservoir / active-inference front-end | Handles high-dimensional noisy data. |
| Candidate generation | Neural, symbolic, search, retrieval | Broad proposal diversity matters. |
| Core reasoning | PCCP / CEGIS / ILP / theorem proving / symbolic search | Checkability and repair matter. |
| Evidence grounding | Retrieval + citations + data provenance | World contact. |
| Verification | Tests, proofs, constraints, simulators, learned verifiers | Function alignment. |
| Deployment | Compiled program/rule/proof/check trace | Cheap and inspectable inference. |

This architecture may serve the five outcomes better than pure PCCP:

- Genuine intelligence: neural front-end handles messy contact; PCCP core handles explicit function.
- Improvability: proof/test core localizes failures even when perception remains probabilistic.
- Democratized development: public checks and programs are editable; neural parts can be swapped.
- Data efficiency: symbolic constraints reduce data needs where applicable; neural priors help when data is messy.
- Inference efficiency: compiled artifacts can replace repeated large-model calls in stable domains.

The hybrid also gives a better public story:

```text
The neural model guesses. The proof-carrying core checks, explains, and compiles.
```

That is more credible than:

```text
No neural networks needed.
```

### Attack

Now apply criterion (f) honestly.

Question:

```text
If you remove the neural parts, does the core claim survive?
```

Answer:

```text
Only in formal or symbolically parsed domains.
```

Pure PCCP can still handle code, math, formal games, schema transforms, and constrained scientific-law worlds. But the claim that it handles broad intelligence collapses when neural perception/proposal is removed.

Question:

```text
If you remove the PCCP parts, does the neural-only system already satisfy the
five outcomes?
```

Answer:

```text
No, not fully.
```

Neural-only systems have broad competence, but they are weak on surgical repair, public proof obligations, exact hidden-intervention guarantees, reproducibility for independent builders, inference cost at scale, separation of target function from proxy objective, and user-editable internal knowledge.

But the hostile expert makes a sharper point:

```text
If the hybrid wins, the story is not "PCCP is the new substrate." The story is
"PCCP is a formal-methods module inside a mostly neural AI system."
```

That story may be useful. It may even be commercially strong. But is it paradigm-shifting?

It depends on the marginal contribution. The hybrid must prove:

```text
neural + PCCP > neural + ordinary tools/tests
```

and:

```text
neural + PCCP > DreamCoder/CEGIS/ILP with neural proposal
```

Otherwise PCCP is just the familiar modern pattern:

```text
LLM proposes code, tests check it, loop repairs it.
```

Current coding agents already do a weak version of this. They generate, run tests, inspect failures, patch, and repeat. The PCCP version must be more than "make that more formal."

Balanced score:

| Direction | Genuine intelligence | Improvability | Democratized development | Data efficiency | Inference efficiency | Total | Diagnosis |
|---|---:|---:|---:|---:|---:|---:|---|
| Pure PCCP-A | 3 | 5 | 5 | 4 | 5 | 22 | Excellent where specs exist; narrow. |
| PCCP-B verifier discovery | 4 | 5 | 4 | 4 | 4 | 21 | Potentially huge; currently unclear/circular. |
| PCCP-H hybrid | 4 | 4 | 4 | 4 | 4 | 20 | Most realistic breadth/repair balance; neural parts weaken transparency. |
| Pure neural open model | 4 | 2 | 3 | 2 | 2 | 13 | Broad competence; weak repair and cost. |
| Tool-using neural agent with tests | 4 | 3 | 3 | 3 | 2 | 15 | Strong practical baseline; PCCP must beat this. |
| Existing CEGIS/ILP/DreamCoder hybrid | 4 | 4 | 4 | 4 | 4 | 20 | May match PCCP-H unless PCCP defines a sharper artifact. |

The table is uncomfortable. PCCP-H scores high, but an existing "neural proposal + synthesis/verifier" stack may score similarly. The distinct PCCP claim is not secure.

The anti-NN correction also changes the ranking:

```text
Neural systems are not disqualified. They are weak where the five outcomes
require explicit repair, public structure, data efficiency, and cheap compiled
deployment. But they remain the strongest broad perception and proposal
substrate.
```

Therefore:

```text
The right question is not "is the core non-neural?"
The right question is "does the PCCP core produce outcome gains that neural +
ordinary tools cannot?"
```

### New Hardest Objection

The hardest new objection is:

```text
PCCP's contribution may be marginal once a capable neural agent is allowed to
propose programs and use existing tests, solvers, and proof assistants.
```

This is a serious kill risk. The precommit spec must include a neural-tool baseline:

```text
LLM/proposal system + ordinary tests/CEGIS/ILP/DreamCoder baseline + repair loop
```

PCCP only survives if it adds a measurable structural advantage: better hidden-intervention transfer, stronger local repair, shorter compiled artifacts, better proof obligation coverage, lower inference cost after compilation, fewer examples to reach function, or less human-written verifier/DSL labor.

### Verdict + Next-Gate Ranking

Verdict:

```text
The hybrid scores highest as an architecture, but it weakens any pure
non-neural PCCP narrative. PCCP survives only if its proof-carrying causal core
contributes measurable outcome gains over neural agents using ordinary tools.
```

Next-gate ranking after I171:

| Rank | Direction | Update |
|---:|---|---|
| 1 | PCCP-H with neural/tool baselines included | Best balanced direction if marginal contribution is proven. |
| 2 | Tool-using neural agent + existing synthesis/verifiers | Must be treated as strongest boring baseline, not strawman. |
| 3 | Pure PCCP-A formal benchmark | Good clean gate, but not enough for full vision. |
| 4 | PCCP-B verifier discovery | High potential; needs tractable restricted version. |
| 5 | Pure neural-only | Broad but still weak on the five outcomes that motivated the reset. |

---

## I172: Scale and Complexity

### Steelman

The strongest PCCP scaling story is not brute-force program synthesis. It is:

```text
Search only over typed, compositional, counterexample-constrained, causally
factored hypothesis spaces; reuse learned libraries; verify locally; compile
surviving structure.
```

PCCP can scale if the world has exploitable structure:

- sparse causal graphs;
- modular subprograms;
- reusable abstractions;
- typed interfaces;
- low executable/description complexity;
- intervention families that isolate failure;
- counterexamples with high teaching value;
- proof obligations that decompose;
- learned proposal/search guidance;
- cached libraries of prior verified components.

In such worlds, the worst-case NP-hardness of synthesis may not dominate practical performance. Many useful exact methods are worst-case hard but practically valuable under structure: SAT, SMT, ILP, type inference variants, planning, theorem proving, and constraint solving.

PCCP also has a potential inference scaling advantage:

```text
Search is expensive during learning, but the output can be a small executable
artifact. Neural models pay a large forward-pass cost every time unless
distilled, cached, or compiled.
```

If the same task family is run many times, PCCP amortizes:

```text
expensive synthesis once -> cheap verified execution many times
```

The strongest long-run architecture therefore includes neural or heuristic proposal to reduce search breadth, CEGIS/SMT/ILP-style verification to eliminate wrong candidates, DreamCoder-style library learning to reduce search depth, causal abstraction to reduce state dimension, and proof-carrying code to make deployment cheap.

### Attack

The worst-case scaling attack is still devastating.

Let:

- `n` = number of causal variables;
- `e` = possible causal edges;
- `k` = DSL primitive count;
- `d` = max program depth/length;
- `m` = intervention families;
- `r` = rule complexity;
- `b` = branching factor of candidate repairs;
- `v` = verifier cost per candidate;
- `h` = hidden-family diversity.

Naive search scales badly:

```text
program candidates ~= O(k^d)
```

Causal graph search scales badly:

```text
possible DAGs over n variables is super-exponential in n
```

Intervention coverage scales badly:

```text
single-variable interventions: O(n)
pairwise interventions: O(n^2)
higher-order interventions: O(2^n)
```

Rule interaction scales badly:

```text
local patching can break nonlocal invariants
```

Verification can also be hard:

- SMT queries can be expensive;
- theorem proving can diverge;
- equivalence checking can be undecidable;
- probabilistic causal verification may need many samples;
- hidden-family testing can explode combinatorially;
- learned verifiers reintroduce statistical uncertainty.

PCCP's core cost is not just synthesis:

```text
cost = design DSL + generate candidates + verify + counterexample search +
repair + prove + hidden-family evaluation + human audit
```

The human-labor term may dominate.

Neural scaling is ugly but real:

- gradient descent is massively parallel;
- representation learning avoids explicit DSL design;
- pretraining amortizes across many tasks;
- perception and language competence improve with data/compute;
- tool use can add verifiers without fully symbolic synthesis;
- distillation and quantization can reduce inference cost.

If neural scaling is better for broad competence, PCCP cannot win by saying "program synthesis is elegant." It must win where exactness, repair, and compilation matter enough to offset search.

The most hostile comparison:

```text
Neural systems scale by spending compute to learn a flexible prior.
PCCP scales by asking humans to define the hypothesis space and verifier.
```

That is not obviously more democratic.

The local-repair claim also has a hidden assumption:

```text
The true system must be modular in the same way the PCCP artifact is modular.
```

If causal factors interact densely, a local counterexample may force global restructuring. This is common in real code, biology, economics, and social systems. A patch to one rule can invalidate downstream proof obligations.

PCCP's strongest promised advantage, local repair, may vanish as interaction density rises.

Scaling comparison:

| Scaling axis | PCCP advantage | PCCP failure mode | Neural advantage | Neural failure mode |
|---|---|---|---|---|
| Causal graph grows | Sparse modular graphs can be explicit | Dense graphs explode | Learns distributed approximations | Opaque and data-hungry |
| DSL grows | Expressivity increases | Search blows up, smuggling rises | Flexible latent features | Hard to guarantee semantics |
| Interventions grow | Better causal guarantees | Coverage is combinatorial | Can generalize statistically | May learn shortcuts |
| Rule complexity grows | Proofs expose structure | Proof/search may diverge | Continuous optimization handles rough fit | Poor exactness and repair |
| Deployment repeats | Compiled program cheap | Only after expensive synthesis | Forward pass works broadly | Recurring inference cost |
| Distribution shifts | Verifier catches some failures | Unknown shifts need new verifiers | Robustness possible with scale/data | Confident failures common |

The scale attack forces a narrower claim:

```text
PCCP is not a general scaling replacement for neural learning. It is a method
for extracting compact, checkable, repeatedly deployable structure when the
task has low executable complexity and useful verifiers.
```

That may still be valuable, but it is not enough for "democratize intelligence" unless many important tasks fall into that class or can be made to fall into it.

### New Hardest Objection

The hardest new objection is:

```text
PCCP's scaling story depends on the world being modular, sparse, and
verifier-decomposable in ways the system can discover. If those assumptions
fail, PCCP has worse scaling than neural methods and worse breadth than
ordinary software engineering.
```

The next gate must therefore require scaling curves over DSL size, hidden causal-variable count, intervention families, and repair locality under increasing rule interaction density. It must also compare against neural proposal/search guidance and measure human-authored structure.

### Verdict + Next-Gate Ranking

Verdict:

```text
PCCP does not have a general scaling story yet. It has a structured-world
scaling story. That is enough for a first gate, but not enough for the
moonshot claim.
```

Next-gate ranking after I172:

| Rank | Direction | Update |
|---:|---|---|
| 1 | PCCP-H with scaling precommit | Must measure scaling, not just pass a tiny toy. |
| 2 | Neural-guided synthesis / DreamCoder-like search | May be necessary for scale; cannot be treated as a penalty. |
| 3 | Pure neural + tool/verifier baseline | Stronger scaling for breadth; weaker repair. |
| 4 | PCCP-A tiny finite worlds | Useful only as theorem/falsifier, not scaling proof. |
| 5 | Pure symbolic brute-force PCCP | Demoted: likely dies by combinatorial explosion. |

---

## I173: Verifier Discovery Deep Dive

### Steelman

The strongest version of PCCP-B is:

```text
The system does not merely satisfy human-given verifiers. It proposes,
refines, ranks, and composes correctness obligations from examples,
counterexamples, interventions, traces, and failures.
```

This is where the moonshot could live.

A verifier-discovery loop might look like:

1. Observe successful and failed behavior.
2. Propose candidate invariants, metamorphic relations, preconditions, postconditions, causal independences, or tests.
3. Use active interventions to distinguish candidate verifiers.
4. Synthesize a program under the current verifier set.
5. Search for counterexamples where program behavior violates intended function.
6. Revise both program and verifier.
7. Prefer verifier sets that predict hidden failures, localize repairs, and compress across tasks.

This reframes intelligence as:

```text
learning what must stay true.
```

The strongest formal footholds:

| Existing framework | What it gives PCCP-B |
|---|---|
| Dynamic invariant detection / Daikon | Candidate invariants from traces. |
| Specification mining | Behavioral automata and temporal/protocol properties from observed executions. |
| ICE/Horn-ICE learning | Counterexample formats for learning inductive invariants/contracts. |
| Angluin-style exact learning | Membership/equivalence query loops with counterexamples. |
| Property-based testing | Generates counterexamples from general properties. |
| Metamorphic testing | Finds relations that should hold when exact oracle is missing. |
| Causal discovery | Distinguishes observational regularity from intervention-stable structure. |
| Active learning / optimal experiment design | Chooses interventions that maximally distinguish hypotheses. |
| Reward/preference learning | Learns evaluators when target behavior is not formally specified. |

Verifier discovery is tractable when the verifier class is restricted, the domain has observable traces, counterexamples are informative, there is a membership/equivalence oracle or approximation, interventions can be run cheaply, the target property has low description complexity, false positives are penalized by hidden tests, and the verifier is treated as uncertain rather than sacred.

That gives a possible first PCCP-B gate:

```text
Given traces and counterexamples from hidden finite worlds, infer a minimal
property set that predicts held-out intervention failures and improves program
synthesis beyond direct example fitting.
```

This is still CPU-native.

### Attack

The circularity attack is severe:

```text
How does the system know what to verify without already knowing what matters?
```

A verifier is a theory of correctness. Learning it from data risks learning another proxy.

Example:

- The system observes many correct sorting outputs.
- It infers "output length equals input length" and "all outputs are integers."
- Those are true invariants but not sufficient.
- It may miss permutation preservation or ordering unless counterexamples expose them.

In open-world tasks, it is worse:

- "The advice is legally safe."
- "The answer is compassionate."
- "The diagnosis is clinically useful."
- "The scientific hypothesis is meaningful."
- "The design is elegant."

These are not simple invariants waiting to be mined. They are contested functions with hidden value judgments, domain expertise, and downstream consequences.

Verifier discovery can reduce to known hard problems:

- invariant inference;
- specification mining;
- active automata learning;
- causal discovery;
- reward learning;
- meta-learning;
- program synthesis for tests;
- scientific theory formation;
- preference aggregation.

If PCCP-B is "learn the verifier distribution across tasks," then it is meta-learning. If it is "infer the hidden reward," it is inverse reinforcement learning or preference learning. If it is "infer invariants from traces," it is spec mining. If it is "infer causal structure," it is causal discovery. If it is "learn a formal language by counterexamples," it is exact learning.

The hostile expert says:

```text
PCCP-B is a bag of hard problems renamed "verifier discovery."
```

There is also a Goodhart problem:

```text
Once a learned verifier becomes the target, the system can satisfy the learned
verifier while missing the real function.
```

This recreates the kill history at a higher level:

```text
BPB proxy -> killed.
Hidden-coordinate proxy -> killed.
CTI proxy -> killed.
Learned-verifier proxy -> likely killed unless grounded.
```

Verifier discovery only escapes proxy/function divergence if candidate verifiers are grounded in exact execution consequences, physical interventions, human-auditable obligations, held-out counterexamples, independent oracles, multiple adversarial decompositions, and explicit residual uncertainty.

Otherwise PCCP-B becomes:

```text
learn a better-looking proxy for correctness.
```

The hardest practical issue is false confidence. A bad learned verifier is worse than no verifier because it certifies the wrong artifact.

### New Hardest Objection

The hardest new objection is:

```text
A learned verifier is itself a proxy. PCCP-B only solves the kill history if it
can prove or empirically demonstrate that verifier learning is more aligned
with the target function than the proxies that already failed.
```

Therefore PCCP-B cannot be deferred vaguely. It must be formalized with kill gates:

- verifier class declared before learning;
- source of counterexamples declared;
- hidden properties not inferable from superficial traces;
- false-positive and false-negative costs measured;
- adversarial verifier gaming test;
- comparison to Daikon/spec-mining/ICE/exact-learning baselines;
- clear line between exact verifier, learned verifier, and human judgment.

### Verdict + Next-Gate Ranking

Verdict:

```text
Verifier discovery is the real moonshot, but it is not magic. In restricted
classes it is known prior art; in open worlds it risks becoming another proxy.
PCCP-B survives only as grounded, counterexample-rich verifier refinement under
declared hypothesis classes.
```

Next-gate ranking after I173:

| Rank | Direction | Update |
|---:|---|---|
| 1 | Restricted PCCP-B verifier refinement | Most important next theory gate; must avoid learned-proxy trap. |
| 2 | PCCP-H with explicit exact vs learned verifier boundary | Realistic architecture if uncertainty is honest. |
| 3 | Spec mining / invariant learning baselines | Must be treated as direct competitors. |
| 4 | Pure PCCP-A | Clean but no moonshot unless followed by B. |
| 5 | Learned reward/verifier without grounding | Demoted: repeats proxy/function divergence. |

---

## I174: Competing Directions B24 Missed

### Steelman

The fair question is:

```text
If goal is sacred and method is not, does something outside PCCP serve the five
outcomes better?
```

Evaluate the missed directions on equal footing.

#### A. Algorithmic Information Theory / Kolmogorov / MDL / Solomonoff

Steelman:

```text
Intelligence is compression of experience into the shortest predictive program.
```

This is deeply aligned with the repo's "structure that makes intelligence cheap" framing. It directly supports data efficiency and inference efficiency. AIXI and Solomonoff induction are more foundational than PCCP. PCCP may simply be a computable, verifier-constrained fragment of this older idea.

Attack:

```text
Kolmogorov complexity is incomputable. Solomonoff/AIXI are not practical
democratized systems. MDL needs a model class, which reintroduces DSL smuggling.
```

Verdict:

```text
AIT is stronger as theory support than as mainline implementation. It can
discipline PCCP's compression objective, but it does not replace the need for
tractable search, verification, and repair.
```

#### B. Active Inference / Free Energy Principle

Steelman:

```text
Intelligence is action-perception under uncertainty: a system maintains and
updates a generative model, acts to reduce uncertainty, and chooses epistemic
actions that improve future control.
```

This is broader than PCCP. It naturally includes perception, action, uncertainty, embodiment, and exploration. It may handle open-world intelligence better than formal synthesis.

Attack:

```text
The framework risks explaining everything and therefore predicting too little.
It often becomes another variational inference/generative-model stack, with
opaque internals and weak local repair.
```

It does not automatically give democratized development, proof-carrying artifacts, or cheap compiled inference.

Verdict:

```text
Active inference is a strong competitor for embodied/open-world intelligence,
but weaker on public verification and surgical repair. It may be a better
outer-loop theory for task selection and exploration than a replacement for the
PCCP core.
```

#### C. Computational Mechanics / Epsilon-Machines

Steelman:

```text
Intelligence begins with minimal predictive causal states: group histories by
the futures they imply.
```

This is extremely close to CWC-E. It gives a principled, non-neural account of causal compression and predictive state. It has real theorems about minimality and uniqueness in its setting.

Attack:

```text
Prediction is not action, proof, judgment, language, repair, or open-world
understanding. Epsilon-machines are strongest for stochastic processes, not
arbitrary intelligent tasks.
```

Verdict:

```text
Computational mechanics may beat B24's vague CWC-E as theory support. It does
not beat PCCP-H as an engine because it lacks proof-carrying synthesis and
task-level repair.
```

#### D. Hutter's AIXI and Tractable Approximations

Steelman:

```text
AIXI is closer to a formal AGI theory than PCCP. It combines universal
induction with sequential decision-making and reward maximization.
```

This directly targets genuine intelligence and not just formal tasks.

Attack:

```text
AIXI is incomputable; approximations are domain-limited; reward specification
is a verifier problem in disguise; inference efficiency is poor; local repair
is not natural.
```

Verdict:

```text
AIXI beats PCCP in philosophical ambition but loses on CPU-first artifact,
democratized repair, and near-term falsifiability.
```

#### E. Category-Theoretic / Topos / Compositional AI

Steelman:

```text
Intelligence is composition-preserving structure. Category theory can state
interfaces, invariants, transformations, abstraction, and compositionality in a
substrate-open way.
```

This aligns with "Intelligence = Geometry" better than almost anything. It may provide the language for composing verifiers, programs, causal abstractions, and learned modules.

Attack:

```text
Category theory is often a description layer, not an engine. It can make
existing ideas sound unified while doing no search, no perception, no verifier
discovery, and no repair.
```

Verdict:

```text
Category theory is powerful support language. It does not beat PCCP unless it
produces a concrete synthesis/verification loop.
```

#### F. Reservoir Computing / Echo State / Liquid State / Physical Reservoirs

Steelman:

```text
Use fixed nonlinear dynamics as cheap computation; train only a readout.
Leverage physical substrates for efficient inference.
```

This is attractive for inference efficiency and democratized cheap hardware. It can handle time series and continuous dynamics with low training cost.

Attack:

```text
Reservoirs are usually opaque, weakly repairable, and limited by reservoir
quality/size. They do not naturally produce proof-carrying, editable knowledge.
```

Verdict:

```text
Reservoir computing is a substrate candidate for perception/dynamics, not a
mainline theory of repairable intelligence.
```

#### G. Cellular Automata / Computational Irreducibility

Steelman:

```text
Simple rules can generate rich complexity. Intelligence may emerge from
searching the computational universe rather than training a giant model.
```

This passes the weirdness and gossip tests. "Tiny rules generate mind-like complexity" is a powerful story.

Attack:

```text
Complexity is not intelligence. Computational irreducibility often makes
prediction and control harder, not easier. Emergence without steering is not a
democratized intelligence artifact.
```

Verdict:

```text
Cellular-automata exploration is inspirational but currently weaker than PCCP
on all five outcomes except narrative weirdness and possible inference cheapness.
```

#### H. Analog / In-Memory / Neuromorphic / Physical Computing

Steelman:

```text
Use physics to compute cheaply. If the substrate's dynamics match the problem,
inference can be orders of magnitude more efficient than digital simulation.
```

This directly attacks inference efficiency and could democratize deployment if hardware becomes accessible.

Attack:

```text
Analog substrates are hard to program, inspect, reproduce, repair, and verify.
They often accelerate neural workloads rather than define a new intelligence
principle.
```

Verdict:

```text
Analog computing is a deployment substrate, not the intelligence claim. It can
support PCCP or neural systems later, but it does not replace the need for a
function-aligned theory.
```

### Attack

The competitor table:

Scoring scale:

```text
5 = strong, 1 = weak.
Narrative means gossip-magazine survivability.
Attack survival means how well the direction withstands hostile expert review.
```

| Rank | Direction | Genuine intelligence | Improvability | Democratized development | Data efficiency | Inference efficiency | Narrative | Attack survival | Total | Verdict |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | PCCP-H: hybrid verifier-centered executable intelligence | 4 | 4 | 4 | 4 | 4 | 5 | 3 | 28 | Best balanced mainline if prior-art and scale gates are fixed. |
| 2 | AIT/MDL + computable verifier/search fragment | 4 | 3 | 3 | 5 | 4 | 4 | 3 | 26 | Strong theory; becomes PCCP-like when made tractable. |
| 3 | Computational mechanics / CWC-E | 4 | 3 | 4 | 5 | 4 | 3 | 3 | 26 | Strong causal-compression theory support; weak action/proof engine. |
| 4 | Existing CEGIS/ILP/DreamCoder stack with causal stress suite | 3 | 4 | 4 | 4 | 4 | 3 | 3 | 25 | High score but less moonshot unless extended to causal/verifier discovery. |
| 5 | Tool-using neural agent + verifiers | 4 | 3 | 3 | 3 | 2 | 4 | 4 | 23 | Strong boring baseline; may beat PCCP in breadth. |
| 6 | Active inference / FEP | 4 | 3 | 3 | 3 | 3 | 4 | 2 | 22 | Broad open-world theory; weak concrete repair/check artifact. |
| 7 | Category/topos AI | 3 | 3 | 4 | 3 | 3 | 2 | 2 | 20 | Excellent language; not an engine yet. |
| 8 | Reservoir computing | 3 | 2 | 3 | 3 | 4 | 3 | 2 | 20 | Useful substrate; weak inspectable intelligence. |
| 9 | AIXI / universal RL approximations | 5 | 2 | 2 | 3 | 1 | 4 | 2 | 19 | Strong formal AGI ambition; poor tractability and repair. |
| 10 | Analog/physical computing | 2 | 1 | 2 | 2 | 5 | 4 | 2 | 18 | Hardware efficiency, not intelligence by itself. |
| 11 | Cellular automata / computational irreducibility | 2 | 1 | 3 | 2 | 4 | 5 | 1 | 18 | Fascinating narrative; poor control and repair. |

Note the uncomfortable result:

```text
Existing CEGIS/ILP/DreamCoder stack scores almost as well as PCCP-H.
```

This reinforces I169: PCCP must either absorb prior art explicitly or be absorbed by it.

No missed direction cleanly beats PCCP-H as mainline. But several directions should be reclassified as support:

| Support lane | Role |
|---|---|
| AIT/MDL | Compression objective and theorem language. |
| Computational mechanics | Minimal causal-state theory for CWC-E. |
| Category theory | Composition/interface language. |
| Active inference | Exploration and action-under-uncertainty outer loop. |
| Reservoir/analog | Possible efficient perception/dynamics substrate. |
| AIXI | Philosophical upper bound and warning about incomputability. |

### New Hardest Objection

The hardest new objection is:

```text
PCCP may not be the paradigm. It may be the practical intersection of several
older paradigms: MDL/AIT for compression, computational mechanics for causal
states, CEGIS/ILP/DreamCoder for synthesis, formal methods for proof, and
neural systems for perception.
```

That can still be the right engineering direction. But the story must change from:

```text
We discovered a new substrate.
```

to:

```text
We found the missing integration point: intelligence artifacts should be
executable, causally stress-tested, verifier-bearing, and locally repairable.
```

### Verdict + Next-Gate Ranking

Verdict:

```text
No missed direction beats PCCP-H as a CPU-first mainline, but AIT,
computational mechanics, and existing synthesis systems substantially absorb
its theoretical novelty. PCCP survives as integration discipline, not pure
conceptual invention.
```

Next-gate ranking after I174:

| Rank | Direction | Update |
|---:|---|---|
| 1 | PCCP-H with explicit prior-art absorption | Mainline if it names its ancestors and beats them. |
| 2 | AIT/MDL + computational mechanics theory spine | Best support framework for "cheap intelligence = executable causal compression." |
| 3 | Existing synthesis stack as baseline/reuse path | Must be used, not ignored. |
| 4 | Active inference outer loop | Worth keeping for exploration and verifier discovery. |
| 5 | Category/topos language | Useful only if tied to concrete PCCP artifacts. |

---

## I175: Final Verdict

### Steelman

After I169-I174, the strongest surviving version is:

```text
PCCP-H: a verifier-centered hybrid architecture that turns stable knowledge
into executable, proof/test-carrying causal programs; uses neural or other
substrates where they best serve perception/proposal; treats CEGIS, ILP,
DreamCoder, symbolic regression, causal discovery, and spec mining as ancestors
or baselines; and makes verifier discovery an explicit staged moonshot rather
than a vague future.
```

This version serves the five outcomes:

1. **Genuine intelligence:** function is checked under interventions and transformations, not merely optimized as a proxy.
2. **Improvability:** counterexamples point to repairable rules, tests, proofs, or assumptions.
3. **Democratized development:** artifacts are inspectable and public; baselines and smuggling controls are explicit.
4. **Data efficiency:** examples are constraints, not just gradient fuel.
5. **Inference efficiency:** verified artifacts compile into cheap execution.

This version is honest about neural systems:

```text
Use neural components when they serve perception, proposal, search guidance,
or open-world contact. Do not grant them default status, and do not punish them
for being neural.
```

This version is honest about prior art:

```text
PCCP is not allowed to claim novelty for CEGIS loops, ILP rule induction,
DreamCoder library learning, symbolic regression, causal discovery,
proof-carrying code, or invariant/spec mining.
```

The possible new contribution is:

```text
an outcome-first artifact contract for learned intelligence:
compact executable causal structure + proof/test obligations +
hidden-intervention survival + local repair + human-labor accounting.
```

### Attack

What was killed:

1. **Pure PCCP-A as moonshot.**

```text
Given-verifier synthesis is too close to CEGIS/SyGuS/ILP/program synthesis.
It can be a clean gate, not the paradigm claim.
```

2. **PCCP as a new algorithm.**

```text
The algorithmic pieces are prior art. Novelty must be in the artifact contract,
benchmark discipline, or verifier-discovery integration.
```

3. **Anti-neural scoring.**

```text
Neural perception/proposal may be necessary. Balanced criterion (f) favors
whatever serves the five outcomes.
```

4. **Broad "all intelligence decomposes into verifiers" rhetoric.**

```text
This is unproven and likely false in strong form. Decomposition must be tested.
```

5. **Toy formal puzzle as sufficient narrative.**

```text
A hand-authored DSL and verifier beating a neural net proves almost nothing.
```

What survived:

1. **Function-first discipline.**

```text
The kill history still points to proxy/function divergence as the enemy.
```

2. **Executable, checkable, repairable artifacts.**

```text
This remains the strongest answer to improvability, democratized development,
data efficiency, and inference efficiency.
```

3. **PCCP-H as mainline candidate.**

```text
Hybrid verifier-centered executable intelligence is stronger than pure PCCP.
```

4. **Verifier discovery as moonshot extension.**

```text
It must be restricted, grounded, and adversarially tested, not hand-waved.
```

5. **Precommit spec as next artifact.**

```text
Still correct, but must be upgraded to include prior-art baselines, hybrid
baselines, scaling gates, and verifier-discovery gates.
```

Balanced final ranking:

Scoring scale:

```text
5 = strong, 1 = weak.
Criterion (f) now means "balanced substrate fit": does the substrate choice
serve the five outcomes better than alternatives, without anti- or pro-neural
bias?
```

| Rank | Direction | (a) Manifesto alignment | (b) Narrative strength | (c) CPU-only feasibility | (d) Paradigm-shift potential | (e) Attack survival | (f) Balanced substrate fit | Total | Verdict |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | PCCP-H: hybrid proof-carrying causal core | 5 | 5 | 5 | 4 | 3 | 5 | 27 | MAINLINE, but only as hybrid/integration discipline. |
| 2 | Restricted verifier discovery / PCCP-B | 5 | 5 | 4 | 5 | 2 | 4 | 25 | Moonshot extension; must be grounded to avoid learned-proxy trap. |
| 3 | AIT/MDL + computational mechanics theory spine | 5 | 4 | 4 | 5 | 3 | 4 | 25 | Theory support; sharpens compression and causal-state claims. |
| 4 | Existing CEGIS/ILP/DreamCoder/symbolic stack with causal stress suite | 4 | 3 | 5 | 3 | 4 | 5 | 24 | Must be baseline and possible implementation substrate. |
| 5 | Tool-using neural agents + verifiers/tests | 4 | 4 | 3 | 3 | 4 | 4 | 22 | Strong boring baseline; may beat PCCP in breadth. |
| 6 | Active inference outer-loop architecture | 4 | 4 | 3 | 4 | 2 | 4 | 21 | Useful for exploration/embodiment; weak artifact repair story. |
| 7 | Category/topos compositional AI | 4 | 2 | 5 | 4 | 2 | 3 | 20 | Formal language lane; not an engine. |
| 8 | Reservoir/analog physical substrates | 3 | 4 | 2 | 3 | 2 | 3 | 17 | Deployment/perception substrate, not mainline intelligence theory. |
| 9 | Cellular automata / computational irreducibility | 3 | 5 | 3 | 4 | 1 | 2 | 18 | Strong weirdness, weak control/repair. |
| 10 | Pure neural scaling | 3 | 3 | 1 | 2 | 3 | 3 | 15 | Broad competence but fails the reset's repair/democratization/cost pressure unless heavily augmented. |
| 11 | Pure PCCP-A | 4 | 3 | 5 | 2 | 2 | 3 | 19 | Useful formal gate; no longer mainline by itself. |

Decision token:

```text
PCCP_SURVIVES_AS_HYBRID_VERIFIER_CORE_NOT_AS_NOVEL_STANDALONE_PARADIGM
```

Updated mainline statement:

```text
Cheap intelligence is not "program synthesis" and not "neural scaling."
The live candidate is a verifier-centered hybrid system that converts stable
knowledge into compact executable causal artifacts with public obligations,
hidden-intervention tests, and local repair, while using neural or other
substrates wherever they best serve the five outcomes.
```

### New Hardest Objection

The final hardest objection is:

```text
PCCP may be the correct engineering discipline but not the paradigm shift.
The real paradigm shift may require discovering verifiers and decompositions
automatically; until then, PCCP is formal-methods engineering plus good taste.
```

This objection survives the whole batch.

The only acceptable response is to make it a gate:

```text
The precommit spec must include at least one verifier/decomposition discovery
subgate, even if restricted, so PCCP does not get trapped forever in
human-given formal worlds.
```

### Verdict + Final Gate Requirements

Final verdict:

```text
PCCP survives, but B24's pure version does not. PCCP-A is demoted to a clean
formal gate. PCCP-H becomes the mainline. PCCP-B becomes the moonshot risk that
must be addressed early, not postponed indefinitely.
```

The next precommit spec must address these risks:

1. **Prior-art baseline absorption.** Include CEGIS, ILP, DreamCoder/library-learning, symbolic regression, causal discovery, and spec-mining baselines where applicable.
2. **Neural-tool baseline.** Include a tool-using neural proposal/repair baseline with ordinary tests/verifiers.
3. **Hybrid evaluation.** Score pure PCCP, hybrid PCCP, and neural-tool baselines by the five outcomes.
4. **Human-labor accounting.** Track every human-supplied DSL primitive, verifier, transformation, hidden family, and decomposition.
5. **DSL smuggling control.** Ensure primitives do not directly encode the hidden target family.
6. **Verifier smuggling control.** Ensure verifier is not merely an answer key.
7. **Scaling gates.** Vary causal variables, DSL size, intervention families, and rule interaction density.
8. **Repair locality metric.** Measure whether counterexamples cause bounded patching or global rewrite.
9. **Verifier discovery mini-gate.** Include a restricted task where the system induces or refines at least one useful verifier/property from traces/counterexamples.
10. **Decomposition gate.** Include a messy task where the system proposes partial verifiers and residual uncertainty, not just solves a given formal puzzle.
11. **Narrative limits.** Public claim cannot be "we solved intelligence." It can only be "we found a precommitted domain where proof-carrying executable knowledge beats proxy learning and prior synthesis baselines under hidden interventions."
12. **Outcome-first criterion (f).** Neural components are allowed if they improve the five outcomes. They must be measured, not ideologically discounted.

Kill rule:

```text
If PCCP only beats weak neural baselines on hand-authored formal worlds, kill
the moonshot claim and demote it to formal-tools support.

If existing CEGIS/ILP/DreamCoder/symbolic-regression baselines match PCCP under
the same causal/verifier stress suite, kill the acronym and adopt the prior-art
system as the implementation substrate.

If learned verifiers become ungrounded proxies that pass internal checks while
failing hidden function tests, kill PCCP-B until a stronger grounding theory is
specified.

If hybrid PCCP gains no measurable outcome advantage over neural-tool agents
with ordinary tests and repair loops, demote PCCP-H and search for a stronger
mainline.
```

Positive token discipline:

```text
PCCP_SIGNAL now requires beating strong prior-art synthesis baselines and
neural-tool baselines on a precommitted hidden-intervention task, while
producing a compact executable artifact with public proof/test obligations.

STRONG_PCCP requires local counterexample repair, scaling curves, and a theorem
or exact characterization explaining why proxy/reconstruction or baseline
synthesis fails.

MOONSHOT_PCCP requires verifier or decomposition discovery: the system must
infer or refine correctness obligations that predict hidden failures better
than spec-mining/invariant-learning baselines.
```

---

## Recommendation

**Verdict: KEEP PCCP, BUT REWRITE THE CLAIM.**

Kill as mainline:

```text
Pure PCCP-A as a standalone paradigm shift.
```

Kill as rhetoric:

```text
"Non-neural" as a virtue by itself.
```

Retain:

```text
Function-first executable, checkable, repairable artifacts as the best answer
to the kill history.
```

Mainline:

```text
PCCP-H: hybrid proof-carrying causal core with explicit prior-art baselines,
neural-tool baselines, scaling gates, and verifier-discovery roadmap.
```

Theory support:

```text
AIT/MDL for compression; computational mechanics for causal-state minimality;
category theory for composition/interface language; active inference for
exploration and uncertainty; existing synthesis systems as implementation
substrates or baselines.
```

W-loop implication:

```text
Writing `research/PCCP_PRECOMMIT_SPEC.md` remains correct, but the spec must be
more hostile than B24 suggested. It must not merely specify a toy verifier-rich
world. It must defend against prior-art absorption, narrowness, hybrid
baseline loss, learned-verifier proxy failure, and scaling collapse.
```

---

## What Must Change Before The Next Work Loop

Minimum spec upgrades:

1. **Rename the strongest candidate internally to PCCP-H or "verifier-centered executable intelligence."** This prevents pure PCCP-A overclaim.
2. **Declare novelty honestly.** No claim that CEGIS, ILP, DreamCoder, symbolic regression, causal discovery, proof-carrying code, or invariant mining are new.
3. **Use prior art as baselines or components.** A PCCP implementation may be CEGIS/ILP/DreamCoder-based if the artifact contract is new.
4. **Add a neural-tool baseline.** Balanced substrate evaluation requires it.
5. **Add a verifier-discovery subgate.** Even a restricted one prevents indefinite deferral of the real moonshot.
6. **Add a decomposition subgate.** The system must propose partial checks or uncertainty boundaries for at least one messy task.
7. **Scale beyond one toy.** Vary graph size, DSL size, intervention count, and rule interaction density.
8. **Track human-supplied structure.** DSL/verifier/decomposition labor must be counted as cost.
9. **Predeclare public claim boundaries.** The first result is a verifier-rich separation, not "AGI without neural nets."
10. **Let neural parts win where they genuinely win.** The sacred outcomes decide.

---

## NARRATIVE ATTACK

### 1. Strongest "that's obvious" dismissal

```text
Of course programs, proofs, tests, and counterexamples are easier to inspect
than neural weights. Of course a small program can beat a neural net on a
formal puzzle if you give it the right DSL and verifier. This is CEGIS, ILP,
DreamCoder, symbolic regression, causal discovery, and proof-carrying code
stitched together after the neural experiments failed.
```

This dismissal lands unless the result is explicitly stronger than prior art.

The defense cannot be:

```text
But our acronym combines them.
```

The defense must be:

```text
We precommitted a hostile task where prior synthesis systems, symbolic
regression, neural-tool agents, and proxy learners all get the same information.
The proof-carrying causal artifact wins on hidden interventions, local repair,
artifact length, verifier coverage, and inference cost, with human-supplied
structure accounted for.
```

### 2. Strongest "that's trivial" dismissal

```text
You made a tiny formal world, wrote the verifier, chose the primitives, hid the
real difficulty in benchmark design, and then announced that the program
searcher learned intelligence. It did not. It solved a puzzle whose ontology
you already supplied.
```

This kills PCCP if:

- the DSL contains the answer;
- the verifier is an answer key;
- the world is too small for search explosion to matter;
- the causal variables are hand-exposed;
- existing CEGIS/ILP/DreamCoder/symbolic baselines are absent or weak;
- the neural baseline is a strawman;
- the system never discovers a verifier or decomposition;
- local repair is just rerunning synthesis;
- hidden interventions are cosmetic;
- the public claim says more than the evidence.

The result is nontrivial only if:

- the hidden families require real transfer;
- strong prior-art systems are included;
- neural-tool baselines are included;
- human-authored structure is counted;
- scaling curves are reported;
- a learned/refined verifier predicts hidden failures;
- the artifact is shorter, cheaper, checkable, and locally repairable;
- the theorem or exact analysis explains why proxy learning or ordinary synthesis fails.

### 3. What the result needs to BE for the narrative to be unkillable

The unkillable result is not:

```text
A non-neural system beats a neural net on a toy.
```

The unkillable result is:

```text
A CPU-only system, using a precommitted public task suite, converts messy or
partially specified problems into compact executable causal artifacts with
proof/test obligations. Against CEGIS, ILP, DreamCoder-style library learning,
symbolic regression, causal discovery, spec-mining, and neural-tool baselines,
it needs fewer examples, survives hidden interventions, repairs locally from
counterexamples, and compiles to cheaper inference. At least one correctness
obligation is discovered or refined by the system rather than hand-written.
The result includes smuggling audits, scaling curves, and an exact explanation
of why the baselines fail.
```

Normal-person headline target:

```text
A laptop AI learned the rulebook, checked its own work, and fixed the broken
rule without retraining. The surprising part was not that it used programs.
The surprising part was that it beat the usual AI and the usual program
synthesis tools under tests it had never seen.
```

Final narrative verdict:

```text
PCCP is still alive, but only after losing its easy story. The surviving story
is not "symbolic beats neural." It is "intelligence should leave behind
checkable, executable, repairable knowledge." To make that unkillable, PCCP
must beat the old symbolic tools, not merely borrow them; it must use neural
parts where they help, not posture against them; and it must start discovering
what to verify, not only satisfying verifiers humans already wrote.
```
