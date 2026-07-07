# Q-Loop B28: Attack Frame Formation

**Date:** 2026-07-07
**Role:** Question Loop worker
**Iterations:** I190-I196
**Status:** analysis-only frame-formation attack; CPU-only constraint; no implementation, no training, no experiments. Web/source checks used only for prior-art mechanism grounding.

---

## Grounding

Read in the requested order:

1. `research/VISION.md`
2. `research/PCCP_PRECOMMIT_SPEC.md`
3. `research/PCCP_THEOREM_DRAFT.md`
4. `research/question_loop_batch24.md`
5. `research/question_loop_batch25.md`
6. `research/question_loop_batch26.md`
7. `research/question_loop_batch27.md`
8. `research/dual_loop_supervisor_checkin_15.md`
9. `research/dual_loop_supervisor_checkin_16.md`
10. `research/dual_loop_supervisor_checkin_17.md`
11. `research/dual_loop_supervisor_checkin_18.md`
12. `research/DEEP_RETHINK.md`
13. `research/STATUS.md`

Additional process grounding:

- `C:\Users\devan\.claude\projects\C--Users-devan-OneDrive-Desktop-Projects-AI-Moonshots-moonshot-sutra\memory\MEMORY.md`

Binding facts:

- The invariants are fixed: swing for the home run, and stop only when an adversary cannot knock it down.
- The five sacred outcomes remain genuine intelligence, improvability, democratized development, data efficiency, and inference efficiency.
- The substrate is open. Neural, symbolic, programmatic, proof-based, causal, search, hybrid, and unknown substrates are evaluated by outcome.
- The kill history's central lesson is proxy/function divergence.
- PCCP-H is an after-frame artifact contract: compact executable causal structure, public obligations, hidden-intervention survival, local repair, human-labor accounting, strong baselines, and neural-tool comparison.
- B27 identified the blind spot as frame formation: how a system discovers the function, verifier, intervention grammar, decomposition boundary, and uncertainty boundary before PCCP compression begins.
- Supervisor #18 sharpened this: the discovery move is the moonshot. The theorem and spec are foundation, not finish line.

Prior-art anchors checked for this batch:

| Anchor | Mechanism relevance |
|---|---|
| Daikon: <https://plse.cs.washington.edu/daikon/> | Reports likely invariants from observed executions. Direct baseline for trace-to-obligation discovery. |
| QuickCheck: <https://hackage.haskell.org/package/QuickCheck> | Randomly tests human-written properties. Strong counterexample generator, not property discovery. |
| Hypothesis: <https://hypothesis.readthedocs.io/en/latest/> | Python property-based testing over generated inputs and edge cases. Same baseline class as QuickCheck. |
| Metamorphic testing | Relations among related executions when exact oracles are unavailable. Direct baseline for intervention-style obligations. |
| PC, FCI, GES causal discovery | Graph/equivalence-class discovery under assumptions. Baseline for role and intervention-structure discovery. |
| Echidna: <https://github.com/crytic/echidna> | Smart-contract fuzzer for user-defined invariants/assertions. Finds failures once properties exist. |
| JML / Spec# / contract inference | Pre/postconditions, class invariants, assertions, and design-by-contract specifications. Baseline for contract-level verifier discovery. |

Current strongest position to attack:

```text
PCCP-H is credible after the frame exists. It is not yet credible as a paradigm
until a cheap system can discover useful verifier clauses, intervention-stability
claims, decomposition boundaries, or uncertainty flags without humans hiding the
key frame in the DSL, perturbation grammar, examples, or neural proposer.
```

---

## I190: How Do Existing Systems Discover Specifications?

### Steelman

Specification discovery is not empty. There are strong prior-art mechanisms.

| Mechanism | What it discovers | Cheapness | Limits for PCCP-B |
|---|---|---:|---|
| Daikon / dynamic invariant detection | Likely pointwise invariants: equalities, inequalities, sortedness, nullness, array/list properties | Cheap on small traces | Trace-biased, template-limited, likely not proved, weak at intent and intervention semantics. |
| General invariant mining | Numeric, shape, temporal, and relational invariants | Cheap to expensive | Often needs templates or a known proof goal. |
| ICE / Horn-ICE | Inductive invariants for transition systems | Tractable in restricted classes | Learns invariants for a known safety property, not the property itself. |
| QuickCheck / Hypothesis | Counterexamples to supplied properties | Very cheap | Human writes property and generator. Discovery is off-ledger. |
| Specification mining | API protocols, temporal automata, call-sequence rules | Moderate | Mines observed behavior, including bugs; misses rare or normative obligations. |
| Metamorphic testing | Input-output relations across transformed executions | Cheap once transformations exist | Relation discovery is domain-specific; human-chosen transforms can smuggle the answer. |
| Contract inference | Preconditions, postconditions, class invariants | Practical for bounded code | Needs code/interface structure and a contract language. |
| PC / FCI / GES causal discovery | Causal graph or equivalence class | CPU-feasible for small sparse graphs | Requires strong assumptions; observational data cannot identify all interventions. |
| Fuzzing-guided testing | Failure witnesses, coverage hot spots, minimized counterexamples | Very cheap | Mostly tests existing oracles. Crashes are weak specs. |
| Echidna-style property fuzzing | Falsifying call sequences for user invariants | Cheap | User supplies invariants. It finds violations, not missing rules. |

The optimistic reading is that PCCP-B can be built from these parts: dynamic invariant candidates, metamorphic relation search, causal effect screening, ICE/Horn invariant learning, and fuzzing-guided counterexamples.

### Frame Question

Do these systems already solve PCCP-B for finite worlds?

Answer: they solve parameter discovery inside a frame. They do not solve full frame formation.

If the missing clause is `output invariant under toggling S` and the human supplies `toggle_S`, then Daikon plus metamorphic testing or a tiny relation miner can solve it. That is not moonshot evidence. The key assumptions were already supplied: what to perturb, which output relation matters, which labels are trusted, and which trace pairs are in-domain.

### What We Might Be Missing

Prior art may be enough for the first PCCP-0 discovery mini-gate. That should not be hidden. If Daikon-style invariants plus metamorphic testing catch the missing finite-world failure, then the mini-gate is absorbed by prior art and the novelty boundary moves upward to transformation-grammar discovery or decomposition discovery.

### Verdict

```text
PARTIAL_PRIOR_ART_ABSORPTION.
```

Existing systems are sufficient for weak verifier discovery when the grammar of observations, perturbations, and candidate properties is human-given. They are not sufficient for the moonshot version: discovering which distinctions, transformations, and uncertainty boundaries matter.

---
## I191: What Would Cheap Verifier Discovery Actually Look Like?

### Steelman

A concrete PCCP-0 discovery mechanism can start as active effect screening over a finite typed world.

Minimal inputs:

```text
- typed observation fields o_1...o_n
- queries q and interventions i
- labels y for allowed seen cases
- partial verifier V0 known to be incomplete
- generic perturbation grammar T over fields
- bounded clause grammar G over paired traces
- candidate programs that may exploit shortcuts
```

Cheap algorithm:

```text
1. Generate paired traces (x, tau(x)) using generic perturbations tau.
2. Query the seen target oracle or public verifier for labels y and y_tau where allowed.
3. Estimate effect signatures:
      invariant: y_tau = y
      covariant: y_tau = phi_tau(y)
      unstable: no stable relation
4. Search a small grammar of candidate clauses:
      invariant_to(field_j)
      invariant_to(cluster_k)
      covariant_under(field_j := v)
      monotone_under(field_j increase)
      equivariant_under(permutation pi)
      precondition(boundary B)
5. Reject clauses already implied by V0.
6. Freeze top clauses by MDL score: short, supported, high failure-catch value, nonredundant.
7. Compile accepted clauses into verifier obligations.
```

This is not useful if it merely runs a candidate program on perturbed inputs and asks whether the candidate output changes. That learns the candidate's behavior. The useful version compares target behavior or counterexample behavior under controlled perturbations.

Example:

```text
Seen data has S highly correlated with target Y.
P_bad uses S and passes V0.
The discovery module toggles every observed field with no role labels.
It observes that changing S does not change target labels, while changing C1/C2 does.
It proposes O_new: decision invariant under do(S := s').
V0 + O_new rejects P_bad.
```

### Frame Question

Is this trivial?

It is trivial if the decisive perturbation and exact oracle are handed over. It is nontrivial only if the perturbation grammar is generic, causal roles are hidden, many false clauses are plausible, and the learned obligation transfers to held-out worlds or transformations.

The smallest nontriviality is role ambiguity: S predicts Y observationally but has zero causal effect under intervention. The system must prefer intervention-stability over correlation.

### What We Might Be Missing

The oracle is the hidden cost. Cheap discovery needs one of: exact labels on perturbed pairs, trusted simulator behavior, public counterexamples, differential comparison to a reference, human feedback, or a neural proposer. Without one of these, the system mines regularities, and regularities are proxies until grounded.

### Verdict

```text
CHEAP_DISCOVERY_EXISTS_AS_ACTIVE_EFFECT_SCREENING_UNDER_A_GIVEN_INTERVENTION_SPACE.
```

The W-Loop can implement this on CPU. It should not call it the moonshot unless the system also discovers useful transformations or obligation grammars not made obvious by the setup.

---

## I192: The Smuggling Problem For Discovery

### Steelman

If a verifier clause is discovered, the audit question is:

```text
Where did the intelligence come from?
```

| Source | Example | Diagnosis |
|---|---|---|
| DSL smuggling | Primitive `is_spurious(field)` | KILL if claimed as discovery. |
| Predicate-template smuggling | Candidate grammar contains exactly `invariant_to_S` | Downgrade or kill depending on specificity. |
| Transformation smuggling | Human supplies `toggle_spurious_feature` | Human chose the answer-shaped experiment. |
| Human labor | Human picks which field or relation to test | Count in labor ledger. |
| Neural smuggling | GPT proposes the useful clause | Strong baseline, not a cheap formal mechanism. |
| Generator smuggling | Public generator exposes role structure | Valid only if all baselines get it; weak discovery claim. |
| Genuine structure discovery | Generic perturbations reveal stable effects across role-randomized worlds | Candidate PCCP-B signal. |

### Frame Question

How do we distinguish real discovery from smuggling?

Controls:

1. Generic perturbation ledger: classify every transformation as role-neutral, domain-generic, or target-specific.
2. Name randomization: field and predicate names must not matter.
3. Role permutation: causal, nuisance, and spurious roles change across worlds.
4. Hidden transformation families: freeze clauses before structurally different evaluation.
5. Negative-control transformations: include plausible false relations.
6. Clause ablation: show V0 misses a failure, V0 + O_new catches it.
7. Prior-art parity: run Daikon, metamorphic miners, ICE/Horn where applicable, causal discovery, and random clause search.
8. Neural-tool parity: same traces, DSL, perturbation budget, and counterexample interface.
9. Human-labor accounting: count grammar, transformation, oracle, and decomposition design.
10. Cross-encoding check: change surface encodings and require behavioral rediscovery.

A clause is `DISCOVERED_BY_SYSTEM` only if it is selected from a generic predeclared candidate space by allowed evidence, survives role/name controls, and beats at least the relevant simple baselines.

### What We Might Be Missing

The repo needs two ledgers: an information ledger and a labor ledger. Even if hidden information did not leak, a hand-crafted perturbation grammar can still be expensive expert intelligence disguised as a cheap algorithm.

### Verdict

```text
SMUGGLING_IS_CONTROLLED_ONLY_BY_BEHAVIORAL_ROLE_RECOVERY_AND_EQUAL_INFORMATION_BASELINES.
```

A clause is not discovered because it appears in output. It is discovered only if the path to it survives name randomization, role permutation, negative controls, prior-art comparison, neural-tool comparison, and labor accounting.

---
## I193: Is Frame Formation Even Solvable In General?

### Steelman

No. Universal frame formation is impossible without assumptions.

Reasons:

- No-free-lunch: arbitrary target functions are unlearnable.
- Rice-style limits: nontrivial semantic properties of arbitrary programs are undecidable.
- Kolmogorov limits: shortest descriptions are not computable in general.
- Causal identifiability: observationally equivalent worlds can disagree under intervention.
- Query complexity: unrestricted Boolean verifier discovery can require exponential queries.
- Open-world ambiguity: many real tasks are normative, contested, or context-dependent.

But this defines an assumption ladder rather than killing the line.

| Level | Problem | Assumptions | Status |
|---|---|---|---|
| B0 | Learn parameters inside a human-given verifier template | Finite domain, fixed grammar, exact labels | Prior art. |
| B1 | Discover roles under generic perturbations | Hidden roles, typed fields, intervention access, sparse effects | Feasible CPU target. |
| B2 | Discover metamorphic relations from finite grammar | Paired oracle labels, MDL scoring, held-out relation tests | Hard but plausible. |
| B3 | Discover decomposition boundaries | Modular failures, local dependency cones, compositional tasks | Open but testable. |
| B4 | Discover messy open-world goals | Human feedback, social grounding, uncertain residuals | Not formally solved. |
| B5 | Universal frame formation | No assumptions | Impossible. |

The B26 exact-learning theorem is right for B0. It does not solve B2-B4 because it assumes the verifier class.

### Frame Question

What assumptions make frame formation tractable without making it trivial?

Minimum viable assumptions:

1. Finite typed worlds.
2. Low-complexity target frames.
3. Sparse causal/effect roles.
4. Intervention access.
5. A grounding oracle: labels, counterexamples, simulator, or trusted differential behavior.
6. Compositional transfer across worlds.
7. Negative controls.
8. Hidden-family holdout.

This creates a bounded scientific-discovery game: infer which transformations preserve or change the target function.

### What We Might Be Missing

The right theorem may be narrower than `PCCP-B discovers verifiers`:

```text
Under finite typed intervention access and low-complexity metamorphic relation
classes, active effect screening identifies the functional equivalence classes
needed for a verifier with query complexity polynomial in fields, values, and
candidate relation templates.
```

That would be less glamorous than open-world spec discovery, but it would be provable and implementable.

### Verdict

```text
FRAME_FORMATION_IS_IMPOSSIBLE_IN_GENERAL_BUT_TRACTABLE_AS_ACTIVE_DISCOVERY_IN_RESTRICTED_INTERVENTION_RICH_WORLDS.
```

PCCP-B should not pretend the first win will be open-world value discovery. The first real win is bounded frame discovery: find intervention-stable equivalence classes and compile them into obligations under hostile smuggling audit.

---

## I194: The Neural-Tool Baseline For Discovery

### Steelman

The strongest boring baseline is:

```text
Give a capable neural tool user the observations, DSL, partial verifier,
candidate traces, counterexamples, perturbation interface, and clause grammar.
Ask it to propose verifier clauses. Compile those clauses into the same PCCP
verifier and evaluate hidden families.
```

This baseline is existential. It can use prose reasoning, code execution, Daikon/spec-mining outputs, causal-discovery outputs, tests, counterexample feedback, solvers, and repair attempts. It may beat pure formal discovery because it carries broad priors about what invariants, tests, and decompositions tend to matter.

The harsh possibility:

```text
GPT-5 + tools may already be the best frame-formation engine.
```

If so, PCCP-H is not dead as an artifact contract, but the discovery mechanism is neural-tool use.

### Frame Question

If the neural-tool agent discovers verifier clauses as well as or better than a formal PCCP-B module, what does that mean?

It means:

```text
PCCP-H becomes the compiler, verifier, and audit layer.
The neural agent becomes the proposal and frame-formation layer.
The paradigm claim shifts from cheap formal discovery to neural proposal
compiled into cheap proof-carrying artifacts.
```

This may still serve some sacred outcomes. It can improve hidden-intervention correctness, repair, and amortized inference. But democratized development weakens if the proposer is proprietary or expensive, and data efficiency becomes ambiguous because the prior came from massive pretraining.

Required baseline controls:

| Condition | Control |
|---|---|
| Same information | Same traces, DSL, public verifier, perturbation budget, and hidden ignorance. |
| Same tools | If PCCP-B can run Daikon/CEGIS/solvers, the neural agent can too. |
| Same freeze | Proposed clauses freeze before hidden evaluation. |
| Same compilation | Clauses must compile into machine-checkable obligations, not prose. |
| Same labor accounting | Prompt engineering and human examples count. |
| Same cost accounting | Tokens, wall-clock, API/model cost, and calls reported. |

The neural-tool baseline wins if it gets equal or better hidden failure catch rate, false-positive control, artifact length, repair localization, human-labor cost, total discovery cost, and hidden-family transfer.

### What We Might Be Missing

Neural smuggling is not a reason to ban neural tools. It is an accounting category. The real question is whether the discovered frame compiles into public, inspectable, locally repairable obligations whose repeated use is cheap and whose hidden failures are lower than alternatives.

### Verdict

```text
NEURAL_TOOL_DISCOVERY_IS_THE_STRONGEST_BASELINE_AND_MAY_ABSORB_PCCP_B.
```

PCCP-H is unnecessary as a discovery paradigm if a neural-tool agent with ordinary spec-mining, tests, and solvers discovers the same verifier clauses. PCCP-H remains valuable only if its compiled artifacts, hidden-intervention discipline, repair locality, and human-labor accounting add measurable value beyond that baseline.

---
## I195: What Is The Smallest Discovery Demo?

### Steelman

The smallest useful demo should prove that a cheap automated mechanism can add a missing obligation that matters.

Concrete PCCP-0 demo:

```text
World family:
- latent bits C1, C2 are causal
- nuisance bits N1...Nk are high-entropy surface noise
- spurious bit S is highly correlated with Y in seen environments
- observation fields O are randomly permuted/encoded versions of C, N, S
- target Y = C1 xor C2

Partial verifier V0:
- checks seen labeled examples
- checks one known nuisance-like formatting perturbation
- does not check invariance to S

Bad candidate P_bad:
- uses S as shortcut
- passes V0 because S correlates with Y in seen examples
- fails hidden spurious-break interventions

Discovery module:
- receives no causal role labels
- receives generic single-field perturbations and finite value replacements
- generates paired traces over seen worlds
- queries allowed labels for perturbed examples
- mines paired metamorphic relations
- proposes O_new: decision invariant under perturbations of the field behaving like S

Verifier V1 = V0 + O_new:
- rejects P_bad
- accepts the true C1 xor C2 program
- catches a hidden failure V0 missed
```

Discovery algorithm:

```text
Pair-Difference Metamorphic Miner

For every observed field o_j:
  For every sampled input x:
    For every alternate value v in domain(o_j):
      x_prime = replace(o_j, v, x)
      record target labels y, y_prime

For each field o_j:
  stable_score(j) = Pr[y_prime = y]
  covariant_score(j, phi) = Pr[y_prime = phi_v(y)]
  shortcut_score(j) = mutual_info(o_j, y) on observational data

Propose:
  invariant_to(o_j) if stable_score high and shortcut_score high
  nuisance_invariant(o_j) if stable_score high and shortcut_score low
  causal_covariant(o_j, phi) if covariant_score high
```

CPU feasibility: 8-12 fields, binary or small finite domains, 256-4096 base cases, exact enumeration over single-field perturbations. Laptop-cheap.

### Frame Question

Does this satisfy the requirement that no prior-art system trivially achieves the same thing?

Honest answer: not by itself.

If the missing obligation is a simple single-field invariance and the perturbation grammar is given, prior art can probably solve it. A metamorphic-relation miner can discover it directly. Causal effect screening can identify the non-effect of S. A neural-tool agent may propose it from traces. Daikon may discover supporting pointwise invariants, though it will not by itself understand paired target invariance unless the paired traces are encoded.

To make the demo less trivial, add:

1. Role permutation across worlds.
2. Plausible false invariances.
3. Composite field clusters rather than single fields.
4. Covariance clauses, not only invariance clauses.
5. Hidden transformation transfer.
6. Baseline parity against Daikon, MR miners, causal discovery, random clauses, and neural-tool agents.

### What We Might Be Missing

Condition (d) may be the wrong first requirement. The right first requirement is not `no prior art can solve this`; it is `no prior art is ignored or handicapped`.

If prior art solves the first demo, that is data. It means frame formation inside a fixed finite perturbation grammar is solved enough, and the moonshot boundary moves to grammar discovery or decomposition discovery.

### Verdict

```text
SMALLEST_DEMO_FEASIBLE_BUT_NOT_MOONSHOT_BY_ITSELF.
```

Build it because it validates the pipeline: evidence to clause, clause to verifier, verifier to caught failure, failure to repair trace. Do not claim paradigm signal unless it transfers and beats prior-art and neural-tool baselines under equal information.

---

## I196: Final Frame-Formation Verdict

### Steelman

Frame formation is tractable enough to be part of the moonshot, but only in bounded form:

```text
Discover intervention-stable obligations in finite typed worlds by active
perturbation, counterexample mining, clause search, and hidden-family validation.
```

Proposed W-Loop mechanism:

```text
FDM-0: Frame Discovery Module v0
```

Components:

1. Trace collector: examples, candidate traces, verifier failures, interventions, labels.
2. Generic perturbation lattice: replacements, toggles, value swaps, masks, permutations, cluster swaps, order changes, duplication/removal, and compositions over typed fields.
3. Effect-signature estimator: invariant, covariant, monotone, equivariant, unstable, or under-sampled.
4. Clause grammar: invariance, covariance, permutation equivariance, monotonicity, conservation, precondition boundaries, uncertainty boundaries.
5. MDL clause scorer: length, support, coverage, failure-catch value, nonredundancy with V0, negative-control robustness, held-out seen transfer.
6. Compiler to obligations: candidate clauses become machine-checkable verifier checks.
7. Repair hook: failed obligations map to fields, transformations, and candidate-program dependencies.
8. Absorption suite: Daikon-style invariants, metamorphic mining, ICE/Horn where applicable, PC/FCI/GES-style causal discovery, random clauses, neural-tool proposal.
9. Smuggling ledger: human-authored transformations, clause grammar, oracle calls, neural calls, baseline access, hidden-family ignorance.

Minimum token:

```text
FRAME_SIGNAL:
FDM-0 proposes at least one new obligation not in V0; the obligation is frozen;
V0 + obligation catches a concrete hidden or held-out failure missed by V0; the
clause is not a name/template leak; and at least one simple baseline does worse.
```

Stronger token:

```text
DISCOVERY_SIGNAL:
The discovered obligation transfers across role permutations or hidden
transformation families and improves repair locality or hidden-family accuracy
against prior-art and neural-tool baselines.
```

Kill token:

```text
DISCOVERY_ABSORBED:
Daikon/metamorphic/causal/spec-mining/neural-tool baselines find equal or better
obligations under equal information, or the winning obligation depends on a
human-authored target-specific transformation.
```

### Frame Question

Is PCCP-H incomplete without frame formation?

Yes.

```text
PCCP-H is incomplete as a paradigm until frame formation is a demonstrated
mechanism rather than a roadmap item.
```

But this is not a reason to stop. It means the next work should attack frame formation directly with a bounded finite mechanism.

Honest hierarchy:

| Claim | Current status |
|---|---|
| Functional compression beats surface compression under supplied verifier | The theorem draft supports this in finite worlds. |
| Existing synthesis systems can produce PCCP-style artifacts | Likely; must test. |
| Cheap systems can discover verifier parameters inside a fixed grammar | Plausible and prior-art-heavy. |
| Cheap systems can discover useful perturbation/intervention roles | Plausible first moonshot subgate. |
| Cheap systems can discover open-world goals and values | Not solved. |

### What We Might Be Missing

In real tasks, the function can be socially constructed. Medical, legal, artistic, and interpersonal tasks do not reveal the true verifier from traces alone. The open-world extension should be:

```text
not "the system discovers the true verifier"
but "the system proposes partial obligations and explicit uncertainty boundaries
that humans can audit, contest, and revise."
```

### Verdict

```text
FRAME_FORMATION_IS_THE_MOONSHOT_AND_FDM_0_IS_THE_NEXT_CONCRETE_MECHANISM.
```

PCCP-H remains valid as a mainline artifact contract, but it is not paradigm-complete until FDM-style discovery produces useful obligations under hostile baselines. The W-Loop should implement the smallest finite witness where a discovered obligation catches a missed failure, then test absorption by Daikon/metamorphic/causal/neural-tool baselines.

Honest gossip-magazine sentence:

```text
The project is no longer asking whether a laptop can obey a rulebook; it is
asking whether a laptop can notice the missing rule before the human writes it.
```

---
## Recommendation

**Verdict: ATTACK FRAME FORMATION DIRECTLY.**

Keep:

```text
PCCP-H as the artifact contract: executable, checkable, intervention-robust,
locally repairable, and hostile to smuggling.
```

Add immediately:

```text
FDM-0: a bounded frame-discovery module that mines metamorphic/intervention
obligations from generic perturbations, target effect signatures, and
counterexamples.
```

Do not claim:

```text
PCCP-B solves open-world specification discovery.
```

Claim only:

```text
In finite typed worlds, we can test whether cheap active perturbation plus
clause mining discovers obligations that catch failures human-written partial
verifiers missed.
```

Required first experiment design:

1. Partial verifier V0 with at least one deliberately missing obligation.
2. Candidate program P_bad that passes V0 by exploiting a spurious shortcut.
3. Generic perturbation grammar with no role names.
4. FDM-0 proposes and freezes O_new.
5. V0 + O_new catches P_bad or a hidden failure missed by V0.
6. Compare against Daikon-style invariant mining, metamorphic-relation mining, causal discovery, random clause search, and neural-tool proposal.
7. Report human-labor and information ledgers.

Positive token discipline:

```text
FRAME_SIGNAL requires a discovered obligation that catches a missed failure.
DISCOVERY_SIGNAL requires hidden-family transfer and baseline wins.
MOONSHOT_PCCP still requires discovery or decomposition that is not absorbed by
prior art or neural-tool agents.
```

Kill rule:

```text
If FDM-0 only works because the human supplied the decisive transformation,
record transformation smuggling.

If prior art solves the same discovery under equal information, adopt the prior
art and move the novelty boundary upward.

If the neural-tool baseline wins, demote PCCP-H to verifier/compiler/audit layer
and stop claiming cheap non-neural frame formation.

If no bounded discovery mechanism catches a missed failure, PCCP-H remains an
after-frame discipline, not a paradigm.
```

---

## NARRATIVE ATTACK

### 1. Strongest "discovery is trivial" dismissal

```text
Perturbation testing already does this. Toggle fields, run the oracle, see what
changes, and add invariance tests. Daikon mines invariants. Metamorphic testing
mines relations. Causal discovery finds parents. Fuzzers find counterexamples.
There is no moonshot here; there is just ordinary testing and spec mining.
```

This dismissal is correct against weak PCCP-B.

If the system is given the right field perturbations and an exact target oracle, then discovering `S is spurious` is not a paradigm shift. It is a small active-testing routine.

The defense must be operational:

```text
We do not count discovery inside a hand-picked transformation as moonshot
evidence. We count it only if the system recovers hidden roles from generic
perturbations, selects non-obvious obligations among plausible false clauses,
transfers them to hidden worlds or transformations, and beats Daikon,
metamorphic miners, causal discovery, random clauses, and neural-tool agents.
```

If that defense fails, the honest verdict is:

```text
Prior art solved this layer. Use it.
```

### 2. Strongest "discovery is impossible" dismissal

```text
Without assumptions, no learner can infer the right verifier. No-free-lunch says
arbitrary functions are unlearnable. Rice's theorem blocks general semantic
verification. Observational equivalence blocks causal guarantees. Open-world
goals are normative and contested. Any learned verifier is just another proxy.
```

This dismissal is correct against universal PCCP-B.

The response is to narrow the claim:

```text
PCCP-B is not universal verifier discovery. It is bounded frame discovery under
finite typed domains, low-complexity clause grammars, intervention access,
grounded counterexamples, and hidden-family validation.
```

The project survives only if those assumptions still cover valuable intelligence-like work and if the system discovers enough structure inside them to reduce human labor.

### 3. What would the discovery mechanism need to BE for the narrative to be unkillable?

It must be an active experimental scientist over finite formal worlds, not a passive invariant miner.

Required properties:

1. Active interventions: chooses perturbations to distinguish candidate frames.
2. Generic operation space: starts with role-neutral field/value/structure transformations.
3. Clause induction: proposes invariance, covariance, equivariance, precondition, and uncertainty-boundary clauses.
4. Grounding: tests clauses against labels, counterexamples, simulators, or trusted oracles.
5. Compilation: clauses become machine-checkable obligations, not prose.
6. Repair value: adding the clause catches a real missed failure or localizes repair.
7. Hidden transfer: clause works on held-out worlds or transformations.
8. Baseline victory: beats or extends Daikon, metamorphic testing, causal discovery, ICE/Horn learning, fuzzing, random search, and neural-tool proposal under equal information.
9. Smuggling audit: human labor, DSL priors, perturbation priors, oracle calls, and neural priors are counted.
10. Uncertainty honesty: when evidence is insufficient, it outputs a boundary, not a fake verifier.

Unkillable version:

```text
A CPU-only system is given a partial verifier and messy finite traces. It
actively invents the missing intervention test, compiles it into a verifier,
catches a hidden shortcut that the human spec missed, and does so under a
role-randomized benchmark where the standard spec-mining tools and neural-tool
agent do not find an equally good clause with the same information.
```

Final narrative verdict:

```text
PCCP-H is alive, but the home run is not proof-carrying programs. The home run
is frame discovery: a cheap system that learns what must stay true before the
human writes the rulebook. Until that exists, PCCP-H is an excellent after-frame
discipline, not the paradigm shift.
```
