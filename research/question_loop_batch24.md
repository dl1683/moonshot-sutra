# Q-Loop B24: Paradigm-Level Direction Search

**Date:** 2026-07-07
**Role:** Question Loop worker
**Iterations:** I162-I168
**Status:** analysis-only adversarial direction search; CPU-only constraint; no model, dataset, GPU, or web runs.

---

## Grounding

Read in the requested order:

1. `research/VISION.md`
2. `research/STATUS.md`
3. `research/DEEP_RETHINK.md`
4. `research/question_loop_batch23.md`
5. `research/dual_loop_supervisor_checkin_14.md`

Additional process grounding:

- `C:\Users\devan\.claude\projects\C--Users-devan-OneDrive-Desktop-Projects-AI-Moonshots-moonshot-sutra\memory\MEMORY.md`

Binding facts:

- The live mission is substrate-open. Neural networks are candidates, not doctrine.
- The five sacred outcomes are fixed: genuine intelligence, improvability, democratized development, data efficiency, and inference efficiency.
- There is no live mechanism. CWC is a B23 recommendation, not a binding answer.
- The kill history's central lesson is proxy/function divergence: optimizing a visible proxy repeatedly failed to produce the actual function.
- CTI as smooth compute law is dead; its measurement discipline survives only as hygiene.
- CPU-only means theory, proofs, formalization, hostile toy worlds, exact validators, and cheap falsifiers first.
- The new mandatory criterion is whether a direction actually escapes the neural-network paradigm or merely changes the training target.

Current strongest position to attack:

```text
The next direction should not be "learn causal representations with a smaller
model." It should ask whether intelligence can be made cheap by representing
knowledge as executable, verifiable, locally repairable structure: programs,
proofs, typed transformations, causal rules, search procedures, or algebraic
objects whose correctness is checked against the target function.
```

Working name for the strongest candidate:

```text
PCCP: Proof-Carrying Causal Programs.

Intelligence is the shortest executable structure that preserves the target
function under admissible transformations, interventions, and counterexamples,
and that carries enough verifier/proof/test machinery to localize and repair
its own failures.
```

This absorbs the useful part of CWC but changes the center of gravity:

```text
CWC says: compress causal state.
PCCP says: make the compressed causal state executable, checkable, and patchable.
```

---

## I162: What Does "Intelligence = Geometry" Mean Beyond Neural Networks?

### Steelman

The old trap was reading "geometry" as hidden-vector geometry. That keeps the answer inside neural representation learning. The substrate-open reading is larger:

```text
Geometry = the structure of equivalence, transformation, composition, evidence,
verification, memory, action, and repair.
```

The right mathematical object is not necessarily a manifold of embeddings. It may be a quotient:

```text
Two observations are equivalent when every functionally relevant action,
prediction, proof obligation, or intervention response is preserved.
```

That points to several mathematical structures:

| Structure | What it captures | Why it matters |
|---|---|---|
| Quotient spaces | Ignore nuisance variation while preserving function | Data efficiency |
| Categories | Composition of transformations, tasks, tools, and abstractions | Modular intelligence |
| Type theory | Specifications, proofs, executable terms, repairable errors | Verifier-first correctness |
| Causal states / bisimulation | Decision-equivalent histories under intervention | Causal compression |
| Program semantics | Meaning as behavior of executable artifacts | Function over proxy |
| Rewriting systems | Local transformations that preserve or change meaning | Improvability |
| Rate-distortion | Minimum bits for bounded functional loss | Cheapness as theorem target |
| Algebraic invariants | What survives transformations | Generalization |

The strongest non-neural synthesis:

```text
Intelligence is not a vector. It is an executable quotient of experience by
function-preserving transformations.
```

Cheapness appears when the system stores the quotient, not the surface. It does not memorize every example. It stores the rule, proof, causal program, invariant, or search strategy that makes many examples equivalent.

This reframes CWC:

```text
Bad CWC: train a model to learn causal latents.
Good CWC: derive or synthesize the minimal executable causal state that preserves
decisions, with a verifier that can reject wrong compressions.
```

The top candidate is therefore PCCP: proof-carrying causal programs. A PCCP system has typed observations, candidate executable causal programs, invariants or proof obligations, a verifier/test oracle that measures the actual function, counterexample-guided repair, and a compression preference for shorter, more compositional programs.

This is CPU-native. It can produce the cheapest possible artifact first: a definition, theorem, and finite verifier.

### Attack

The hostile expert says:

```text
This is a collage of old fields: program synthesis, formal methods, MDL,
causal abstraction, bisimulation, category theory, proof search, and symbolic AI.
Where is the new principle?
```

The attack is strong. Category language is especially dangerous. It can make the project sound profound while doing no work. A functor from observations to actions is not an algorithm. A quotient by functional equivalence is not useful unless the equivalence can be discovered cheaply. Type theory verifies what is specified; it does not tell us what to specify. Program synthesis explodes combinatorially. Causal states are exact only in worlds whose causal structure is already constrained.

The hardest attack:

```text
You have replaced "train a model" with "search a program," but the hard part
just moved into the DSL, verifier, and transformation set. If humans hand-write
those, the intelligence is in the humans, not the system.
```

That kills any version where "geometry" is mostly human-authored ontology.

### New Hardest Objection

The hardest new objection is:

```text
Geometry may define cheap intelligence only after the right ontology exists,
but discovering the ontology may be the expensive part.
```

Neural networks are attractive because they avoid explicit ontology design. A non-neural direction must show that typed structure, invariants, programs, or verifiers can be discovered or assembled cheaply enough to beat that advantage.

### Verdict + Next-Gate Ranking

Verdict:

```text
"Intelligence = Geometry" should mean executable functional equivalence, not
neural latent geometry. The strongest candidate is PCCP: a verifier-first
programmatic form of causal compression.
```

Next-gate ranking after I162:

| Rank | Direction | Reason |
|---:|---|---|
| 1 | PCCP | Turns geometry into executable, verifiable, locally repairable structure. |
| 2 | CWC-E | CWC survives only as executable causal compression, not learned latent CWC. |
| 3 | Algebraic/type-theoretic geometry | Strong theory lane; weak unless tied to an engine. |
| 4 | Retrieval + verifier | Strong support substrate; not enough alone. |
| 5 | Neural CWC | Demoted; too close to "train a causal model." |

---

## I163: What Can Non-Neural Systems Do That Neural Networks Fundamentally Cannot?

### Steelman

Neural networks are powerful approximators, but they are structurally bad at some things the five outcomes demand.

| Capability | Non-neural advantage | Outcome served |
|---|---|---|
| Verification | A proof, type check, SAT result, unit test, or exhaustive finite check can certify a function | Genuine intelligence |
| Surgical repair | A counterexample can point to a rule, branch, lemma, type, or subprogram | Improvability |
| Democratized modification | Humans can read and edit programs, proofs, rewrite rules, and evidence graphs | Democratized development |
| Data efficiency | Each example can become a constraint, not just a gradient contribution | Data efficiency |
| Inference efficiency | A compiled rule/program can be cheaper than a large forward pass | Inference efficiency |
| Exact composition | Programs, proofs, typed terms, and algebraic morphisms compose by rules | Scaling through modularity |
| Explanatory state | The system can say which rule/proof/evidence path produced the answer | Trust and repair |

This is not just "symbolic AI again." The key is the kill history:

```text
The project repeatedly optimized proxies. Non-neural verifier-first systems can
sometimes make the measured object the function itself.
```

If the task is "solve this equation," "prove this property," "construct a program satisfying this spec," "choose the action that satisfies these constraints," or "repair this causal model against this counterexample," the verifier can directly measure function. No BPB proxy, no hidden-coordinate proxy, no smooth compute proxy.

The strongest version of non-neural intelligence is not a brittle hand-coded expert system. It is:

```text
counterexample-guided executable compression.
```

The system searches for the smallest program/rule/proof structure that passes the real verifier, then patches only the failing piece when the verifier finds a counterexample.

This is structurally beyond neural networks because the core learning event is:

```text
specification + search + proof/check + repair
```

not:

```text
examples + loss + gradient + weights
```

### Attack

The hostile expert says:

```text
Non-neural systems have had decades to win and did not. They are brittle,
hand-engineered, domain-specific, and useless where perception, ambiguity, and
commonsense dominate.
```

The critique is technical. Program synthesis has search explosion. Symbolic systems require symbols. Proof systems require formal specs. Type systems catch only what was typed. Causal models need interventions. Retrieval systems retrieve false or irrelevant evidence. Exact composition works only when interfaces are exact. Commonsense is often under-specified, contextual, and non-formal.

Neural systems are weakly grounded but broadly adaptable. Non-neural systems are strongly grounded but narrow. The five outcomes require both capability and breadth. A system that proves theorems but cannot parse the world is not general intelligence.

The strongest attack:

```text
Verifier-first systems avoid proxy/function divergence only by retreating to
tasks where the function is already formalized. That may produce useful tools,
not a theory of intelligence.
```

### New Hardest Objection

The hardest new objection is:

```text
Non-neural systems may win exactly where intelligence is least mysterious:
math, code, puzzles, games, finite worlds. They may fail where the moonshot
matters most: open-world judgment under incomplete specification.
```

This forces a narrower but cleaner claim:

```text
First prove cheap intelligence in verifier-rich worlds, then ask how much of
open-world intelligence can be converted into verifier-rich subproblems.
```

### Verdict + Next-Gate Ranking

Verdict:

```text
Non-neural systems have a real structural advantage when the function can be
checked. The moonshot lane should exploit that advantage first, but must not
pretend formal domains solve open-world intelligence.
```

Next-gate ranking after I163:

| Rank | Direction | Update |
|---:|---|---|
| 1 | PCCP | Best use of non-neural strengths: verification, patching, compact execution. |
| 2 | Retrieval + verifier | Externalizes world knowledge; needs PCCP-like checking to avoid hallucinated evidence. |
| 3 | CWC-E | Useful if causal states become executable programs, not latent variables. |
| 4 | Algebraic/type-theoretic geometry | Becomes the specification language behind PCCP. |
| 5 | Neural CWC | Only acceptable as optional perception front-end, not main intelligence claim. |

---

## I164: What Is The Cheapest Possible System That Satisfies All Five Outcomes?

### Steelman

Start from the outcomes and work backward.

**Genuine intelligence** requires function, not surface imitation:

```text
The system must choose, construct, prove, or explain the right thing under
changes that defeat memorization.
```

**Improvability** requires localized failure:

```text
When wrong, the system must expose which rule, assumption, evidence edge, type,
or subprogram failed.
```

**Democratized development** requires inspectable artifacts:

```text
Independent people must be able to read, run, modify, and extend the system
without renting a proprietary weight stack.
```

**Data efficiency** requires examples to do more work:

```text
Each example should become a constraint, counterexample, invariant, or proof
obligation, not only another token in a loss.
```

**Inference efficiency** requires compiled structure:

```text
At inference time, the system should execute the learned structure or run a
small bounded search, not replay a huge parametric memory.
```

The cheapest system satisfying all five is not a small neural model. It is a small loop:

```text
1. Parse observations into typed terms or candidate facts.
2. Search for a compact executable hypothesis.
3. Verify the hypothesis against the actual task function.
4. Use counterexamples to patch the smallest failing part.
5. Compile the surviving hypothesis into a cheap program/rule/proof artifact.
```

Call this:

```text
PCCP-0: proof-carrying causal program induction.
```

The first gate can be CPU-only:

- finite generated worlds with hidden causal rules;
- nuisance variables and spurious shortcuts;
- held-out intervention families;
- exact verifier for the target function;
- strong baselines: memorization table, decision tree, next-step predictor, reconstruction compressor, generic SAT/SMT/program-synthesis solver, and a tiny neural baseline only as a comparison, not as doctrine;
- metrics: program length, examples needed, verification passes, hidden-family accuracy, repair locality, and inference cost.

The moonshot claim would be:

```text
A tiny executable causal program can beat larger proxy learners because it
stores the rule, not the surface.
```

### Attack

The hostile expert says:

```text
This is just inductive program synthesis with property-based tests. The cheap
part is true only because the benchmark is small and the verifier is given.
```

That attack is fair. The first gate cannot be a toy where the DSL contains the exact hidden rule primitives, the verifier is an answer key, all baselines are intentionally mismatched, the causal variables are directly exposed, or the held-out families differ only cosmetically.

Also, the five outcomes may not all be simultaneously satisfiable in one small artifact. A proof-carrying program can be efficient and repairable but narrow. A retrieval system can be broad but verification-limited. A neural model can be broad but opaque. The cheap system may need a hybrid stack:

```text
perception/retrieval for contact with the world;
program/proof/search for the core function;
verifiers for alignment;
compiled artifacts for inference.
```

The danger is that "hybrid" becomes a loophole that hides neural dependence.

### New Hardest Objection

The hardest new objection is:

```text
The cheapest possible system may be cheap only because humans supply the
expensive pieces: the DSL, verifier, examples, transformations, and task
decomposition.
```

The first serious benchmark must account for human-supplied structure. It must separate structure given to every baseline, structure discovered by the system, structure smuggled into the PCCP search space, and structure needed only for evaluation.

### Verdict + Next-Gate Ranking

Verdict:

```text
The cheapest plausible substrate is not a network but a counterexample-guided
loop that induces compact executable structure and verifies it against the
actual function. The moonshot depends on whether the system discovers enough
structure rather than receiving it from humans.
```

Next-gate ranking after I164:

| Rank | Direction | Update |
|---:|---|---|
| 1 | PCCP | Best outcome-first design. |
| 2 | CWC-E | Defines the compression target PCCP should make executable. |
| 3 | Retrieval + verifier | Necessary for external knowledge, but not the first clean theorem gate. |
| 4 | Algebraic/type-theoretic geometry | Support layer for composition and proof obligations. |
| 5 | Physics/energy/search dynamics | Interesting, but no cheap concrete first artifact yet. |

---

## I165: Is There A Mathematical Framework Where Proxy/Function Divergence Does Not Arise?

### Steelman

Proxy/function divergence arises when the measured objective is not the target function:

```text
BPB is not judgment.
Hidden cosine is not capability.
Compute is not function.
Training loss is not understanding.
Early proxy curves are not held-out intervention behavior.
```

The clean escape is not a better proxy. It is:

```text
Make the target function executable.
```

Mathematical frameworks where proxy/function divergence can be removed or sharply bounded:

| Framework | Alignment mechanism | Failure mode |
|---|---|---|
| Formal verification | The property checked is the property claimed | Spec may be wrong |
| Dependent types | Correctness becomes part of the program type | Only typed properties count |
| Hoare logic | Programs carry pre/postcondition proofs | Specs incomplete |
| SAT/SMT solving | Candidate either satisfies constraints or not | Constraint encoding may be wrong |
| CEGIS | Counterexamples directly refine the program | Search space may exclude truth |
| Proof search | Derivation is the artifact | Premises may not match reality |
| Exact finite-world verification | Exhaustive check is possible | Finite world may be unrepresentative |
| Property-based testing | Counterexamples reveal function failure | Sampling may miss rare cases |
| MDL with functional distortion | Shortest artifact under task distortion | Distortion must be function-aligned |

In these settings, the central quantity can be:

```text
D_func(P) = 0 if executable artifact P satisfies the verifier/spec on the
admissible domain; otherwise D_func(P) is the actual counterexample or
functional error.
```

This is different from training a model to lower a proxy loss. The score is the function, or a formal/exhaustive approximation to it.

The strongest PCCP statement:

```text
An intelligent artifact is one whose correctness obligation travels with it.
```

That is what "proof-carrying" buys:

- the artifact is executable;
- the claim is explicit;
- the verifier is public;
- a failure produces a counterexample;
- repair targets the failing condition.

### Attack

The hostile expert says:

```text
You did not solve proxy/function divergence. You moved it into the spec.
```

This is the most serious attack in the entire batch. If the spec is incomplete, the system can satisfy it while violating the real goal. Formal verification prevents implementation bugs relative to a spec; it does not guarantee that the spec is the right one.

Goodhart reappears as verifier gaming, incomplete property sets, narrow test generators, wrong causal interventions, DSL primitives that force the wrong abstraction, and hidden assumptions in the environment generator.

In open-world intelligence, the actual function may not be formalizable in advance. "Be helpful," "judge evidence," "understand context," and "act wisely" do not come with total verifiers.

The strongest attack:

```text
Verifier-first intelligence is only aligned by construction in worlds where
humans already know how to construct the verifier. That is not intelligence;
that is automation under a solved specification.
```

### New Hardest Objection

The hardest new objection is:

```text
The real moonshot may be verifier discovery, not verified search.
```

If a system can only operate after humans give it the verifier, it is narrow. If it can propose, test, refine, and compose verifiers from counterexamples, evidence, and invariants, then it starts to look like intelligence.

The next gate must distinguish:

```text
PCCP-A: search under given verifier.
PCCP-B: learn or refine the verifier/spec itself.
```

PCCP-A is the clean first CPU gate. PCCP-B is the moonshot.

### Verdict + Next-Gate Ranking

Verdict:

```text
Proxy/function divergence can be eliminated only where the function is made
executable or formally checkable. PCCP is strongest precisely there, but its
moonshot extension requires verifier/spec discovery rather than verifier use
alone.
```

Next-gate ranking after I165:

| Rank | Direction | Update |
|---:|---|---|
| 1 | PCCP | Top, but must track spec/verifier smuggling as primary risk. |
| 2 | Verifier discovery | Emerging subdirection; harder and more moonshot than verifier use. |
| 3 | CWC-E | Useful if causal compression defines the admissible verifier domain. |
| 4 | Retrieval + verifier | Needed for evidence-grounded specs, vulnerable to evidence quality. |
| 5 | Neural CWC | Still fails criterion (f) unless neural part is peripheral. |

---

## I166: What Would Make Someone Stop Scrolling?

### Steelman

"We trained a small model better" does not pass the gossip-magazine test. "We learned causal representations" probably does not either.

The strongest stop-scrolling narrative is:

```text
Intelligence does not have to be a neural network.
```

But that line alone is cheap. The result must make it undeniable.

Possible headline targets:

```text
A laptop learned the rule instead of the answer key.
```

```text
A 50-line non-neural system beat a trained model because it could prove what it
understood.
```

```text
The tiny AI did not get smarter by training longer. It got smarter by finding
the counterexample and rewriting the rule.
```

```text
We made an AI whose knowledge is not hidden in weights. It is a program you can
read, verify, and repair.
```

The best public artifact would be visual and exact:

- a toy world where surface patterns lie;
- a neural/proxy learner that gets high training score and fails hidden interventions;
- a tiny PCCP system that induces an executable rule;
- a verifier that catches the proxy failure;
- a counterexample that patches the PCCP artifact locally;
- a one-page proof explaining why reconstruction/proxy compression fails.

The normal-person story:

```text
Most AI studies for the test. This one learns the rulebook.
```

The moonshot version:

```text
Cheap intelligence is not smaller prediction. It is proof-carrying
understanding: knowledge that can be executed, checked, and fixed.
```

### Attack

The hostile expert says:

```text
This is a toy demo wearing a revolution costume.
```

That will be the default reaction unless the artifact has a sharp contrast:

- strong boring baselines;
- hidden-family tests;
- no hand-exposed causal variable;
- no DSL that trivially contains the answer;
- precommitted metrics;
- exact counterexamples;
- a nontrivial proof or exhaustive check;
- visible data/compute advantage.

"AI without neural networks" is not automatically impressive. Classical algorithms solve many things without neural networks. The claim becomes interesting only if the system does something people associate with intelligence: discovers a rule from few examples, transfers under intervention, explains the failure of a larger proxy learner, repairs itself from a counterexample, and produces an artifact humans can inspect.

Also, "beats GPT-4 at X" is dangerous. If X is a contrived formal puzzle, the headline can be attacked as cherry-picked. If X is broad language judgment, a CPU non-neural system likely loses.

### New Hardest Objection

The hardest new objection is:

```text
The narrative must be simple, but the honest claim may be narrow. Overstating
it will kill credibility faster than a modest result.
```

The first public claim should not be "we solved intelligence." It should be:

```text
We found a domain where the neural-training habit provably compresses the
wrong thing, and a tiny proof-carrying program wins because it stores the
function.
```

That is narrower but much harder to knock down.

### Verdict + Next-Gate Ranking

Verdict:

```text
The strongest story is "AI whose knowledge is executable, verifiable, and
repairable instead of hidden in weights." It becomes viral only if a tiny
CPU artifact visibly beats proxy-trained baselines under hidden interventions.
```

Next-gate ranking after I166:

| Rank | Direction | Narrative status |
|---:|---|---|
| 1 | PCCP | Best story: proof-carrying understanding beats memorized prediction. |
| 2 | Verifier discovery | Strongest deeper story, but harder first gate. |
| 3 | CWC-E | Good support story if made executable and visual. |
| 4 | Retrieval + verifier | Familiar "open-book AI" story; less paradigm-shifting alone. |
| 5 | Algebraic/type theory | Powerful internally, weak gossip story unless embodied in PCCP. |

---

## I167: Attack The Top Candidate - What Kills PCCP?

### Steelman

The strongest PCCP proposal:

```text
Proof-Carrying Causal Programs: intelligence is compact executable structure
that preserves the target function under transformations and interventions,
with a public verifier and counterexample-guided repair loop.
```

First CPU gate:

```text
PCCP-0: exact finite-world suite. Generate hidden rule families with nuisance
variables, spurious shortcuts, and intervention shifts. The system must infer a
compact executable program/rule set, attach verifier obligations, and repair
from counterexamples. It must beat reconstruction, memorization, decision-tree,
generic synthesis, and small neural baselines on hidden families with fewer
examples and lower inference cost.
```

The theorem/proof target:

```text
There exist finite world families where reconstruction-optimal compression and
proxy-optimal prediction provably discard the decision-preserving causal
program, while PCCP-style functional verification identifies the smaller
correct artifact from fewer examples.
```

The artifact target:

```text
A readable program/proof produced by the system, plus the counterexample trace
that repaired it.
```

### Attack

PCCP can die in at least ten ways.

1. **DSL smuggling.** The primitive set contains the answer.
2. **Verifier smuggling.** The verifier is effectively a hidden oracle for the exact target rule.
3. **Toy triviality.** The domain is so small that exhaustive search or a decision tree solves it.
4. **Prior-art absorption.** It is just CEGIS, ILP, DreamCoder-like library learning, causal discovery, or symbolic regression with new branding.
5. **Search explosion.** The method works only because the first world is tiny.
6. **No perception bridge.** The system requires clean symbols and cannot handle raw messy observations.
7. **Spec incompleteness.** It proves the wrong thing because the verifier is incomplete.
8. **Human labor hidden cost.** Humans design the world, DSL, invariants, transformations, and validators.
9. **No broad intelligence.** The result is a solver, not an intelligent system.
10. **Narrative overreach.** "No neural networks needed" is attacked as misleading because the result covers narrow formal tasks.

The hardest kill condition:

```text
If PCCP only wins where the answer language and verifier are hand-authored, it
does not explain cheap intelligence. It explains automation in formal domains.
```

The second hardest:

```text
If the best baseline is another existing program-synthesis or symbolic
regression system, and PCCP does not beat it, the direction has no novelty.
```

### New Hardest Objection

The hardest new objection is:

```text
Open-world intelligence may not be reducible to verifier-rich subproblems
without losing the very judgment we care about.
```

This objection is deeper than "program synthesis is hard." It says the PCCP frame may select for domains where correctness is crisp, while ordinary intelligence lives in ambiguous domains where goals, evidence, and values are contested.

The only honest response:

```text
Do not claim universality first. Prove the verifier-rich core. Then investigate
whether open-world tasks can be decomposed into evidence retrieval, local
formal checks, causal consistency checks, and human-auditable residual judgment.
```

### Verdict + Next-Gate Ranking

Verdict:

```text
PCCP survives as top candidate only under hostile precommit. It dies if it
becomes hand-authored symbolic AI, benchmark theater, or old program synthesis
with a new acronym.
```

Next-gate ranking after I167:

| Rank | Direction | Attack-adjusted status |
|---:|---|---|
| 1 | PCCP | Still #1, but only with DSL/verifier smuggling controls and strong synthesis baselines. |
| 2 | Verifier discovery | The real moonshot extension; too hard for first gate but must be planned. |
| 3 | CWC-E | Theory support; should define functional equivalence classes for PCCP. |
| 4 | Retrieval + verifier | Bridge to open world after formal core works. |
| 5 | Neural CWC | Not the post-reset answer unless used as a nonessential adapter. |

---

## I168: Final Direction Ranking

### Steelman

The final ranking should reward directions that:

- directly answer "What is intelligence, and what structure makes it cheap?";
- preserve all five sacred outcomes;
- can produce CPU-only theory or proof artifacts;
- have a gossip-magazine story;
- can be attacked with precommitted gates;
- actually escape the neural-network paradigm.

The strongest candidate is PCCP:

```text
Intelligence = executable, verifiable, locally repairable compression of the
target function.
```

Why this beats B23-style CWC:

```text
CWC as "learn causal representations" is still ML-centric.
CWC as "minimal causal state" is a theory target.
PCCP as "proof-carrying executable causal state" is a substrate-level break.
```

PCCP is not guaranteed to become general intelligence. It is the best first direction because it attacks the deepest kill-history lesson:

```text
Do not optimize proxies. Build artifacts whose correctness is checked against
the function itself.
```

### Attack

This ranking may be overcorrecting away from neural networks.

Neural systems are currently the only substrate with broad, fuzzy, open-world competence. A pure symbolic/proof/program line may win clean gates and still fail to matter. The user asked whether neural networks might need to be abandoned, not whether they must be banned.

The most realistic long-term architecture may be hybrid:

```text
neural/perceptual adapters + retrieval + verifier/search/program core +
proof-carrying compiled artifacts.
```

But the first post-reset batch should not collapse into "hybrid" too early. Hybrid is a destination if the non-neural core earns it, not a loophole for staying in the NN paradigm.

The top direction must therefore make criterion (f) explicit:

```text
If the core intelligence claim requires gradient-trained neural representations,
it fails the new reset. Neural parts may be adapters, baselines, or optional
perception modules, but not the core mechanism.
```

### New Hardest Objection

The hardest final objection is:

```text
PCCP may pass criterion (f) by leaving the neural paradigm, but fail the broader
vision by being too narrow to count as intelligence.
```

The first gate must therefore test not just correctness, but intelligence-like properties:

- few-shot rule induction;
- transfer under hidden interventions;
- counterexample-driven repair;
- explanation as executable artifact;
- composition of learned subrules;
- lower inference cost than proxy learners;
- failure localization.

### Verdict + Final Ranking

Scoring scale:

```text
5 = strong, 1 = weak.
Criterion (e) means attack survival: higher is better.
Criterion (f) means actually beyond NN paradigm: higher is more genuinely
non-neural or substrate-open at the core.
```

Final score:

| Rank | Direction | (a) Manifesto alignment | (b) Narrative strength | (c) CPU-only feasibility | (d) Paradigm-shift potential | (e) Attack survival | (f) Beyond NN paradigm | Total | Verdict |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | PCCP: proof-carrying causal programs | 5 | 5 | 5 | 5 | 3 | 5 | 28 | MAINLINE candidate; write precommit spec before implementation. |
| 2 | Verifier discovery / spec induction | 5 | 5 | 3 | 5 | 2 | 5 | 25 | Moonshot extension; too hard as first gate, but must be in roadmap. |
| 3 | CWC-E: executable causal world compression | 5 | 4 | 5 | 4 | 3 | 4 | 25 | THEORY SUPPORT; CWC survives only in executable/checkable form. |
| 4 | Algebraic/type-theoretic intelligence geometry | 5 | 3 | 5 | 5 | 2 | 5 | 25 | FORMAL LANGUAGE lane; not enough without PCCP engine. |
| 5 | Retrieval + verifier intelligence | 4 | 4 | 4 | 4 | 3 | 4 | 23 | OPEN-WORLD SUPPORT; avoids weight-memory but needs robust verifiers. |
| 6 | Energy/search over explicit candidates | 4 | 4 | 5 | 4 | 2 | 4 | 23 | Useful mechanism; not a full direction unless tied to verification. |
| 7 | Physics-based computation | 3 | 5 | 2 | 5 | 1 | 5 | 21 | Big story, weak immediate CPU path and high vagueness. |
| 8 | Neural CWC / learned causal representations | 3 | 2 | 2 | 2 | 2 | 1 | 12 | DEMOTE; likely NN paradigm with extra steps. |

Decision token:

```text
PIVOT_TO_PCCP_WITH_CWC_E_AS_THEORY_SUPPORT
```

Hard next gate:

```text
Write `research/PCCP_PRECOMMIT_SPEC.md` before any implementation. The spec
must define the function, verifier, admissible worlds, DSL/search space,
compression metric, hidden-family tests, baselines, smuggling controls, theorem
target, positive tokens, and kill tokens.
```

---

## Recommendation

**Verdict: PIVOT BEYOND CWC-AS-ML.**

Kill as mainline:

```text
CWC interpreted as training a small model to learn causal representations.
```

Retain:

```text
CWC's core insight: intelligence requires preserving functional/causal state,
not reconstructing surface data.
```

Mainline:

```text
Proof-Carrying Causal Programs.
```

Precise formulation:

```text
Cheap intelligence is compact executable structure whose correctness can be
checked, whose failures produce counterexamples, and whose repairs are local.
```

CWC-E role:

```text
Define the minimal causal/functional equivalence class that the executable
program must preserve.
```

Algebra/type-theory role:

```text
Provide the composition, specification, and proof language. Do not let it become
decorative math.
```

Retrieval role:

```text
Later bridge to open-world knowledge by retrieving evidence that can be checked
or converted into local verifier obligations.
```

W-Loop implication:

```text
Do not run neural training. Do not implement CWC as learned representation.
Next work should write a PCCP precommit spec and theorem/finite-world gate.
```

---

## What Must Change Before The Next Work Loop

Minimum requirements:

1. **Write `research/PCCP_PRECOMMIT_SPEC.md`.** Define the object before experiments.
2. **Declare the target function as executable.** No proxy loss as primary success.
3. **Define smuggling controls.** Track what is given in the DSL, verifier, generator, transformations, and human design.
4. **Use hidden world families.** The generated rule/intervention families must include held-out structures unknown during fitting.
5. **Include strong boring baselines.** Decision trees, memorization tables, symbolic regression, generic program synthesis/CEGIS, information bottleneck or reconstruction compressor, and a tiny neural baseline only as comparison.
6. **Require an inspectable artifact.** The output must be a readable program/rule/proof/evidence trace, not only a score.
7. **Require local repair.** A counterexample must identify and patch a bounded part of the artifact.
8. **Include a theorem target.** Even a finite-world separation theorem is better than another plot.
9. **Predeclare narrative limits.** No public "we solved intelligence" claim. The claim is a verifier-rich separation until broader evidence exists.
10. **Evaluate criterion (f) every time.** If the core mechanism becomes gradient-trained representation learning, the direction has drifted back into the killed frame.

Positive token discipline:

```text
PCCP_SIGNAL requires a CPU-only hidden-family result where a compact executable
artifact beats proxy/reconstruction/memorization baselines and survives verifier
and DSL smuggling controls.

STRONG_PCCP requires local counterexample repair plus a proof or exact
characterization of why proxy compression fails.

MOONSHOT_PCCP requires verifier/spec discovery: the system helps construct or
refine the correctness obligations, not merely satisfy human-given ones.
```

Kill rule:

```text
If PCCP-0 only wins because the DSL/verifier exposes the answer, do not count
it. Redesign once with hidden families and stronger synthesis baselines. If the
second gate repeats the same weakness, demote PCCP to formal-tools support and
move to verifier discovery, CDMD-style math discovery, or a stricter CWC-E
theory theorem.
```

---

## NARRATIVE ATTACK

### 1. Strongest "that's obvious" dismissal

```text
Of course programs and proofs are more verifiable than neural networks. Of
course a system should learn rules instead of memorizing examples. This is just
symbolic AI, program synthesis, and formal verification rediscovered after
neural experiments failed.
```

This attack is fair unless PCCP produces a sharper object than "symbolic is interpretable." The defense must be:

```text
We are not claiming symbolic AI as a slogan. We are building a hostile test
where proxy learning provably compresses the wrong thing, while a compact
proof-carrying executable artifact preserves the target function under hidden
interventions and repairs itself from counterexamples.
```

### 2. Strongest "that's trivial" dismissal

```text
You made a toy formal world, gave the system the right DSL and verifier, and
then showed that a program searcher beats a neural net on the formal world.
That is not intelligence. That is benchmark design.
```

This kills the direction if:

- the DSL contains the hidden rule directly;
- the verifier is an answer key;
- the causal variables are exposed;
- the baselines are weak by construction;
- the hidden interventions are cosmetic;
- the final artifact is not more than ordinary program synthesis.

The result is nontrivial only if:

- the rule/program is inferred, not handed over;
- hidden world families require real transfer;
- strong synthesis and symbolic baselines are included;
- proxy learners look good on ordinary metrics before failing;
- the PCCP artifact is shorter, cheaper, checkable, and locally repairable;
- the theorem explains why the separation must occur.

### 3. What the result needs to BE for the narrative to be unkillable

The unkillable result:

```text
A CPU-only, non-neural system infers a tiny executable rule from a few examples,
survives hidden interventions that fool larger proxy-trained baselines, produces
a readable proof/check trace, and repairs itself by changing a localized piece
when shown a counterexample. The benchmark was precommitted, the DSL/verifier
smuggling controls passed, and a finite theorem explains why memorization or
reconstruction had to fail.
```

Normal-person headline target:

```text
A laptop built an AI that learned the rulebook instead of memorizing the answer
key. When it was wrong, it found the broken rule and fixed it without retraining.
```

Final narrative verdict:

```text
PCCP is the strongest first post-reset direction because it actually leaves the
neural-network paradigm. It survives only if it becomes proof-carrying,
counterexample-repairable, and hostile to its own DSL/verifier smuggling. Anything
softer is CWC-as-ML or symbolic AI nostalgia in a new coat.
```
