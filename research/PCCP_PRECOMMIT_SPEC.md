# PCCP-H Precommit Specification v1

**Date: 2026-07-07**
**Status: DRAFT - design-gate document; no implementation, no training, no experiments**
**Origin: Q-Loop B24 + Supervisor Check-in #15 + W-Loop B18 + Q-Loop B25 + Supervisor Check-in #16 + W-Loop B19**

---

## 0. Precommit Scope

PCCP means **Proof-Carrying Causal Programs**.

PCCP-H means **hybrid verifier-centered executable intelligence**: a stack in
which neural, symbolic, search, retrieval, or other front-ends may propose
features, symbols, parses, programs, obligations, or repairs, while the durable
knowledge claim must compile into an executable PCCP artifact with public
proof/test obligations, hidden-intervention evaluation, and local repair traces.

PCCP-H is now the mainline candidate. Pure PCCP/PCCP-A, meaning synthesis under
a human-given verifier with no learned or neural proposal layer, remains a
clean formal gate and an important ablation. It is not the standalone moonshot
claim.

Working thesis:

```text
Cheap intelligence should leave behind compact executable structure that
preserves the target function under admissible transformations, interventions,
and counterexamples; carries verifier/proof/test obligations; exposes local
repair handles; and uses whatever substrate best serves those outcomes.
```

This document defines the object and benchmark before any PCCP-H experiment is
allowed to begin. It is deliberately hostile to the direction. PCCP-H is killed
as a moonshot candidate if it wins because the humans hid the answer in the DSL,
verifier, generator, decomposition, or baseline setup, or if a prior-art or
neural-tool baseline achieves the same outcomes under equal information.

PCCP-H is not CTI, not Eklavya, not byte-level modeling, not a new neural loss,
and not a claim that non-neural systems are virtuous by default. Those are
historical unless re-derived from `research/VISION.md`.

The five fixed outcomes are genuine intelligence, improvability, democratized
development, data efficiency, and inference efficiency. The substrate is open:
neural, symbolic, program-synthesis, verifier-first, hybrid, and unknown
mechanisms are evaluated by how well they serve those outcomes, not by whether
they match a preferred ideology.

---

## 1. Definition: What Is A PCCP-H Artifact?

### 1.1 Formal Object

A PCCP-H artifact keeps the PCCP artifact contract. The hybrid part describes
how the artifact may be proposed or refined; it does not weaken the requirement
that the final knowledge object be executable, inspectable, and independently
checkable.

A PCCP artifact is a tuple:

```text
A = (L, P, I, O, C, V, R, M)
```

where:

- `L` is the declared typed language/search space.
- `P` is a finite executable program in `L`.
- `I` is the public interpreter or operational semantics for `L`.
- `O` is the set of proof obligations, invariants, and test obligations claimed
  by `P`.
- `C` is the set of certificates, proof traces, counterexample traces, type
  derivations, or exhaustive-check summaries attached to the artifact.
- `V` is an independent verifier that checks `(P, O, C)` against the benchmark
  contract and returns either `PASS` or a concrete counterexample.
- `R` is a repair map or localization trace explaining which subprogram,
  assumption, branch, lemma, or rule is implicated by each counterexample.
- `M` is metadata: DSL version, generator family id, verifier version, training
  sample ids, hidden-family holdout manifest hash, and human-labor accounting.

`A` is not just a score. The artifact must be inspectable and executable without
the synthesizer that produced it.

### 1.2 Executable

`P` is executable if all of the following are true:

- It has a deterministic or explicitly randomized interpreter with public
  semantics.
- Its inputs and outputs are typed.
- It terminates within a predeclared finite resource bound on every admissible
  input.
- It can be run by a fresh evaluator using only `(L, P, I)` and the public input.
- It does not call the world generator, hidden rule, target oracle, verifier, or
  training set at inference time.
- It produces a decision plus a trace sufficient to identify the branches,
  subroutines, and obligations used.

For PCCP-0, the executable is a finite AST interpreted by a small typed DSL.
Later phases may admit lambda terms, rewrite systems, relational programs,
finite-state machines, Hoare triples, proof terms, or library programs, but only
if they compile to a common finite semantics and length metric.

### 1.3 Proof-Carrying

`P` is proof-carrying if the artifact includes checkable evidence for its own
correctness claims. Evidence may include:

- Proof obligations: preconditions, postconditions, invariants, admissible
  transformations, and causal intervention obligations.
- Type-level invariants: totality, domain bounds, variable-role constraints,
  no-hidden-oracle constraints, and resource bounds.
- Exhaustive finite certificates: exact enumeration over admitted finite
  subdomains where exhaustive checking is feasible.
- SMT/SAT/proof certificates: solver outputs with independently checkable
  witnesses when such solvers are used.
- Property test suites: generated tests bound to a public test generator and seed
  manifest.
- Counterexample traces: concrete failing input, intervention, expected target
  decision, actual program decision, and localized rule/subprogram implicated.

The verifier must be independent of the synthesizer. Passing the verifier means
the public obligations are satisfied on the declared domain. It does not mean the
open-world problem is solved.

Proof-carrying does **not** mean the verifier contains the hidden rule in a form
the synthesizer can inspect, the artifact can query the verifier at inference
time, human prose explanations count as proof without machine-checkable
obligations, or a neural confidence score/training loss/reconstruction score is
a proof.

### 1.4 Causal

`P` is causal if it preserves the target decision under interventions and
counterfactual changes, not merely under the observational distribution.

For a finite generated world `w`, let:

```text
SCM_w = (U, Z, E)
```

where `U` are exogenous variables, `Z` are endogenous variables, and `E` are
structural equations. Observations are generated by an encoder:

```text
x = Obs_w(z, n, s)
```

where `n` are nuisance features and `s` are spurious features. A query `q` and
intervention `i` define the target:

```text
F_w(x, q, i) = target decision after applying intervention i to SCM_w.
```

`P` is causally correct on a world family if:

```text
for every admissible w, x, q, i:
    I(P, x, q, i) = F_w(x, q, i)
```

Required causal cases are nuisance interventions that should not change the
decision, causal interventions that should change the decision as specified,
spurious-break interventions that defeat observational correlations, and
counterfactual swaps that hold selected variables fixed while changing a causal
parent or structural equation.

Correlation is insufficient. A program that uses a feature correlated with the
target in the training environment but fails under `do(...)` intervention is not
a PCCP success.

### 1.5 Program

For PCCP-0, a program is a finite typed AST with variables over finite domains,
constants, bounded conditionals, bounded quantifiers over finite sets, equality,
finite-domain comparisons, Boolean connectives, bounded integer/modular
arithmetic, tuple/set/map/relation operations, `let` bindings, named subroutines,
composition of previously synthesized subroutines, and static bounded recursion
when the bound is counted in length.

The artifact may be represented as DSL AST, rewrite rules, relational rules,
lambda terms, or decision procedures, but it must compile to this finite
operational form for measurement.

The hidden target rule is never a primitive. If the hidden rule is parity, the
DSL may contain Boolean operations, but it may not contain a primitive named
`target_parity`, `causal_parent_selector`, `true_rule`, or anything equivalent.

---

## 2. Target Function Definition

### 2.1 Function To Preserve

The target function is **decision preservation under intervention**:

```text
F: (observation, query, intervention) -> decision
```

For each finite world instance `w`, `F_w` is generated by the hidden structural
equations and benchmark target declaration. The artifact must preserve `F_w` on
admissible worlds, including held-out intervention families.

Primary correctness is:

```text
D_func(P) = 0 if P agrees with F on the exact admissible evaluation domain.
D_func(P) = counterexample set otherwise.
```

There is no primary loss curve. There is no BPB, NLL, reconstruction loss,
embedding score, training accuracy, or proxy metric that can substitute for
`D_func`.

### 2.2 Why This Cannot Be A Proxy

The kill history's central lesson is proxy/function divergence: byte prediction
improved while downstream judgment barely moved; coordinate/readout methods
produced surface or compatibility signal without reliable function; evidence
variants exposed retrieval or corpus shortcuts; CTI-style smooth quantities
tracked proxy behavior while missing functional prediction.

Therefore PCCP starts by making the claimed function executable and checkable.
The benchmark may track proxies as diagnostics, but a proxy win is never a PCCP
win.

### 2.3 Exact Verifier

The verifier for PCCP-0 is exact because the worlds are finite. For a frozen
candidate artifact, the verifier enumerates or otherwise exactly covers all
public training worlds required by the seen-domain obligations, all public
intervention families required by the seen-domain obligations, all hidden
evaluation worlds in the precommitted holdout manifest, all hidden intervention
families in the holdout manifest, and all admissible queries in each checked
world.

The verifier returns:

```text
PASS
```

or:

```text
FAIL(counterexample = (world_id, x, q, i, expected, actual, obligation_id))
```

The final hidden verifier is called only after the artifact is frozen. During
synthesis, the engine may use a public seen verifier and a bounded
counterexample interface. It may not inspect hidden family definitions or hidden
evaluation examples.

### 2.4 Function Alignment By Construction

Function alignment by construction means:

1. The primary objective is to find the shortest executable artifact satisfying
   the exact verifier.
2. The verifier checks the claimed target function, not a proxy for it.
3. Proof obligations are attached to the artifact and checked independently.
4. Counterexamples identify functional failures, not only loss increases.
5. Repair is evaluated by whether the artifact regains the function with local
   changes.

This does not solve specification error. If the target function is the wrong
function, PCCP-H can be exactly wrong. That risk is tracked under verifier
smuggling, decomposition smuggling, and the verifier-discovery gates.

---

## 3. World Families: The Benchmark Design

### 3.1 Admissible World Family

PCCP-0 uses finite generated worlds. A world family is:

```text
W = (Domain, SCM grammar, observation encoder, query grammar,
     intervention grammar, target declaration, split manifest)
```

Every domain is finite. Every generated world has hidden causal structure. The
generator class is public; the sampled family parameters, structural equations,
surface encodings, and hidden split seeds are not visible to synthesis.

### 3.2 Hidden Structure

Hidden means the synthesis engine does not receive the causal graph, parent set
of the target decision, structural equations, nuisance/spurious role labels,
world seed, held-out intervention families, target program used by the generator,
or hidden evaluation examples.

The engine may receive typed observations, typed intervention descriptors for
seen intervention families, decisions on allowed training examples, public DSL
and interpreter, public seen verifier, counterexamples from allowed seen-domain
checks, and the generator class description if baselines receive it too.

### 3.3 Variables

Each world contains at least:

- causal variables `C`: variables that affect the target under intervention;
- nuisance variables `N`: surface variables that do not affect the target;
- spurious variables `S`: variables correlated with the target in some
  environments but not causally responsible;
- observation variables `O`: possibly permuted, encoded, duplicated, noised, or
  bundled surface features;
- query variables `Q`: what the program is asked to decide;
- intervention descriptors `I`: what was changed or should be counterfactually
  evaluated.

The benchmark must include worlds where `C`, `N`, and `S` are not directly
role-labeled in the observation.

### 3.4 Nuisance Variables

Nuisance variables may include random identifiers, color/name/order encodings,
redundant copies, irrelevant high-entropy fields, variable permutations,
observation formatting artifacts, reversible surface encodings, and environment
tags that are not target causes.

Nuisance interventions must change these features while preserving the target
decision. A PCCP artifact that depends on nuisance variables fails hidden-family
accuracy.

### 3.5 Spurious Shortcuts

Spurious variables are deliberately dangerous. They may be perfectly correlated
with the target in training, high mutual-information features in the
observational distribution, easier to express in the DSL than the true rule, and
stable across seen examples but broken by hidden interventions.

The holdout set must include interventions where spurious variables are
randomized, inverted, re-correlated with a different target, or made independent.

### 3.6 Intervention Families

Required intervention families:

1. `do(N := n')`: nuisance replacement; target invariant.
2. `permute_surface`: rename, reorder, or re-encode observations; target
   invariant after decoding.
3. `do(S := s')`: spurious replacement; target invariant if `S` is not causal.
4. `do(C_j := c')`: causal replacement; target changes according to structural
   equation.
5. `counterfactual_hold`: change one causal parent while holding specified
   non-descendants fixed.
6. `environment_shift`: change correlations between `S`, `N`, and `C` without
   changing the target structural equation.
7. `composition_shift`: combine two or more valid interventions.

Hidden intervention families must be structurally different, not only cosmetic
renamings of seen transformations.

### 3.7 Generator Specification

PCCP-0 generator class:

- Choose finite domain sizes for causal, nuisance, and spurious variables.
- Sample a causal graph from a bounded grammar.
- Sample target structural equations from a bounded grammar.
- Sample nuisance variables independently or from non-target parents.
- Sample spurious variables from environment-dependent correlations.
- Encode latent variables into observations through a sampled surface encoder.
- Generate queries and interventions from public grammars.
- Split worlds and intervention families into seen and hidden partitions before
  synthesis begins.

Allowed structural-equation grammar for PCCP-0: Boolean connectives, bounded
integer comparisons, modular arithmetic over declared small moduli, small
threshold rules, finite relation membership, and composition of two or more
subrules in PCCP-1.

The grammar provides parts, not answers. If a hidden rule is `xor(c1, c2)`, then
`xor` may be a generic Boolean primitive, but the DSL/generator must not expose
which variables are the causal parents.

### 3.8 Minimum Complexity Requirements

A benchmark instance is invalid if memorization or trivial exhaustive lookup is
competitive under the predeclared metrics.

Minimum PCCP-0 requirements:

- At least 8 latent variables total, with at least 2 causal variables, 2 nuisance
  variables, and 1 spurious variable.
- At least 4096 admissible observation-query-intervention cases per benchmark
  suite after expansion.
- Training examples cover no more than 20 percent of exact evaluation cases.
- At least 3 held-out world families and 3 held-out intervention families.
- Spurious features have at least 0.9 correlation with target on at least one
  seen environment and no more than 0.55 correlation, or inverted correlation,
  on at least one hidden environment.
- The shortest lookup table for the exact evaluated function is at least 20x
  longer than the shortest known generator-side target rule.
- A decision tree on raw surface features must not exceed the predeclared
  hidden-family threshold on a dry-run benchmark design audit. If it does, the
  benchmark is too easy or the surface encoder leaks causal roles.

The toy theorem in Section 9 may use only 2-3 variables. The benchmark must be
harder than the theorem construction.

### 3.9 Decomposition Gate: Messy Partial Specifications

The benchmark must include at least one decomposition gate that is not a clean
formal puzzle.

Setup:

- The system receives a partially specified problem with some known verifier
  properties, some examples, some counterexample traces, and explicit uncertainty
  about which properties are complete.
- The system must propose partial verifiers, candidate invariants, uncertainty
  boundaries, residual assumptions, and confidence flags before solving.
- The proposed decomposition is frozen before hidden-family evaluation.
- Human-written decompositions, if supplied, are logged as baselines and counted
  in the human-labor ledger.

Success condition:

```text
The system-proposed decomposition catches hidden failures, localizes repairs, or
improves hidden-family accuracy better than direct solving with no decomposition.
```

Failure condition:

```text
The decomposition is trivial, merely restates the given verifier, hides the
answer in human-authored boundaries, or adds no value over direct solving or a
neural-tool agent.
```

This gate exists because "decompose open-world problems into verifier-rich
subproblems" is not an algorithm until the system demonstrates it on a messy
case.

### 3.10 Scaling Gates

A PCCP-H benchmark report must vary the following axes rather than reporting a
single toy point:

| Axis | Required levels |
|---|---|
| Causal variables | `2 -> 8 -> 16` |
| DSL primitive count | `10 -> 30 -> 100` |
| Intervention family count | `3 -> 10 -> 30` |
| Rule interaction density | independent -> pairwise -> higher-order |

For each axis, report hidden-family accuracy, artifact length, synthesis cost,
verifier calls, repair locality, human-authored structure, and inference cost.
The report must answer:

```text
How does PCCP-H performance degrade, and is there a qualitative transition where
local repair, search, or verifier coverage collapses?
```

No moonshot claim is allowed from the easiest scale tier alone.

---

## 4. DSL / Search Space

### 4.1 PCCP-0 DSL Primitives

The PCCP synthesis engine may use finite-domain variables and constants,
equality, inequality, finite comparisons, Boolean operations, bounded integer and
modular arithmetic, finite tuple projection/construction, finite set membership,
bounded `forall` and `exists`, `if/then/else`, `let` bindings, named subroutines
with explicit argument and return types, static bounded recursion,
proof-obligation annotations, and trace labels for repair localization.

The DSL may not include:

- a primitive that names the hidden target rule;
- a primitive that selects causal parents by oracle;
- a primitive that asks whether a feature is nuisance, spurious, or causal;
- access to the generator seed;
- access to hidden verifier results during synthesis;
- learned embeddings or neural modules inside the PCCP-0 core program;
- calls to external solvers at inference time.

External solvers may be used during synthesis only if their use is logged and
baselines are given comparable access where appropriate.

### 4.2 Structure Accounting

Every primitive in `L` must be classified before experiments:

| Structure | Allowed? | Accounting rule |
|---|---:|---|
| Generic Boolean/arithmetic operations | Yes | Counted as DSL prior shared by all synthesis baselines |
| Type declarations | Yes | Public to all baselines |
| Observation schema | Yes | Public to all baselines |
| Intervention grammar | Yes | Public to all baselines |
| Causal role labels | No for hidden variables | If exposed, benchmark is invalid |
| Hidden target parent set | No | Exposure is KILL_PCCP |
| Hidden structural equation | No | Exposure is KILL_PCCP |
| Verifier source for hidden families | No during synthesis | Public only after final freeze if released for audit |
| Generator class | Conditional | Allowed only if all baselines get it |
| Generator sampled parameters | No | Exposure is generator smuggling |

### 4.3 DSL Must Not Contain The Hidden Rule

The DSL fails the smuggling gate if a one-token primitive implements the target
rule, a primitive names the correct latent parent selector, the observation
schema labels the causal parents, the intervention descriptor reveals target
causes in a way not available to baselines, or proof obligations are prewritten
so specifically that filling in variable names is the only remaining task.

The DSL may contain generic operators from which the rule can be built. The
distinction is whether the system discovers the composition and variable roles.

### 4.4 Overpowered DSL Audit

Before any run, perform a design audit:

1. Write the shortest known target program using the DSL.
2. Write the shortest spurious shortcut program using the DSL.
3. Write the shortest lookup/memorization program using the DSL.
4. Estimate or enumerate how many candidate programs of length <= target length
   exist.
5. Check whether generic CEGIS, ILP, symbolic regression, SAT/SMT, or DreamCoder
   could find the same program with the same information.
6. If the target program is only a variable rename away from a DSL primitive, the
   benchmark is invalid.

PCCP novelty is weak if a generic solver finds an equal or shorter artifact with
the same verifier access and no PCCP-specific machinery.

### 4.5 Baseline Search Space

Each baseline receives the same information class as the PCCP engine unless the
baseline's method cannot consume it. Any difference must be logged.

Default rule:

```text
If PCCP gets typed observations, intervention descriptors, training labels,
public DSL, public seen verifier, and counterexamples, then synthesis baselines
get the same package.
```

Neural and tree baselines do not receive the DSL unless adapted versions can use
it, but they receive equivalent raw examples, labels, intervention descriptors,
and train/validation splits.

---

## 5. Compression Metric

### 5.1 Primary Metric

Primary compression metric:

```text
L(A) = L(P) + alpha * L(O) + beta * L(C) + gamma * L(lib)
```

where `L(P)` is program AST length, `L(O)` is proof-obligation length, `L(C)` is
certificate length, and `L(lib)` is learned library/subroutine length.

For PCCP-0:

```text
alpha = 0.25
beta = 0.10
gamma = 1.00
```

The executable program is the primary artifact. Obligations and certificates
count because they can smuggle information, but they are discounted because a
proof trace can be longer than the rule it certifies.

All length units are AST tokens under a public grammar. String names, comments,
and formatting do not affect length. Numeric constants are charged by encoded
bit length.

### 5.2 MDL Objective

The PCCP objective is:

```text
minimize L(A)
subject to V(A) = PASS
```

If no candidate passes, the result is not a PCCP signal, regardless of proxy
score.

### 5.3 Composition And Subroutine Reuse

Subroutines are paid once and charged per call:

```text
L(lib) = sum(length(subroutine_j)) + call_cost * number_of_calls
```

Default:

```text
call_cost = 1 AST token per call
```

A library is credited only if the same subroutine is reused across at least two
distinct worlds, queries, or intervention families; replacing it with inlined
code increases total length; and the subroutine is not just the hidden target
rule under another name.

### 5.4 Why Shorter Programs Should Generalize

The formal claim is not "shorter is always smarter." The precommitted claim is:

In a fixed DSL with a fixed verifier and no smuggled answer, if the causal rule
has shorter description length than memorization and spurious shortcuts, then
MDL over exact functional constraints biases search toward the invariant rule.

Finite hypothesis bound:

Let `H_l` be the set of programs with length at most `l`. If a wrong program has
hidden-family error at least `epsilon`, and training examples are sampled from a
distribution that exposes that error independently with probability `epsilon`,
then:

```text
Pr[there exists wrong P in H_l that passes m examples]
    <= |H_l| * (1 - epsilon)^m
```

Since `|H_l` grows with program length, shorter hypothesis classes need fewer
examples to rule out wrong programs. This argument only supports PCCP if the
hidden-family test contains interventions that distinguish causal rules from
spurious or reconstructive shortcuts.

---

## 6. Baselines: Strong, Not Strawmen

Every baseline receives the same train/validation/test splits and the same
information class as PCCP unless explicitly impossible. Any baseline that beats
PCCP on the precommitted metrics becomes the relevant state of the art, not an
inconvenient footnote.

### 6.1 Memorization Table

Stores observed `(x, q, i) -> y` mappings.

Required variants:

- exact lookup with majority fallback;
- nearest-neighbor over typed surface features;
- intervention-aware table keyed by seen intervention descriptors.

Expected strength: high seen accuracy. Hidden-family failure is meaningful only
if the train/test split makes exact lookup impossible.

### 6.2 Decision Tree

ID3/C4.5-style tree on surface features and intervention descriptors.

Required variants:

- raw surface features;
- one-hot typed features;
- simple engineered finite-domain features available to all systems.

If the tree solves hidden families, the benchmark is too easy or the causal
variables were leaked.

### 6.3 Generic CEGIS

Counterexample-guided inductive synthesis over the same DSL and public seen
verifier.

This is the most important prior-art baseline for pure PCCP/PCCP-A. If generic
CEGIS matches PCCP-H in length, sample efficiency, hidden accuracy, proof
coverage, and repair locality under equal information, then the PCCP-H artifact
contract has no demonstrated novelty beyond CEGIS plus packaging.

### 6.4 ILP

Inductive logic programming, e.g. Metagol or Popper when feasible on CPU.

Inputs:

- relational encoding of observations, queries, interventions, and labels;
- background predicates corresponding only to allowed DSL primitives;
- positive and negative examples from the same split.

ILP must not be denied background predicates that PCCP effectively receives.

### 6.5 Symbolic Regression

Symbolic regression, e.g. PySR-style search, over finite encodings of the typed
features.

It receives the same operators as the numeric/Boolean subset of the DSL. If
symbolic regression finds the compact causal rule, PCCP must beat it on proof
obligations, repair, or composition to claim novelty.

### 6.6 Generic SAT/SMT Solver

Encodes the synthesis problem as constraints over the same finite DSL or a
bounded equivalent. Receives the same examples and seen verifier constraints.

This baseline tests whether PCCP is merely a worse front end for existing
solvers.

### 6.7 DreamCoder-Style Library Learning

If feasible on CPU, use a library-learning baseline that induces reusable
subroutines from multiple tasks/worlds.

If not feasible, document why and include a smaller enumerative library-learning
control. Do not silently omit it.

### 6.8 Reconstruction Compressor

Learns or searches for a compact representation that predicts/reconstructs the
next observation or full observation state, then derives decisions through the
best allowed decoder/probe.

This baseline embodies the killed proxy habit. It is strong only if allowed to
optimize reconstruction well. Its failure is meaningful only if it had enough
capacity to reconstruct surface regularities and still failed the function.

### 6.9 Tiny Neural Baseline

A small MLP or tiny transformer trained on the same examples and intervention
descriptors.

Rules:

- CPU-only.
- Same train/validation/test split.
- Same examples.
- No hidden-family examples.
- Comparison only, not doctrine.
- Do not penalize it for being neural or favor PCCP for being non-neural.

### 6.10 Random Program Baseline

Samples programs from the DSL by length prior and checks the public seen
verifier.

This estimates how much of PCCP's success is simply the DSL prior plus brute
force. If random sampling finds passing compact programs at nontrivial rate, the
benchmark is too easy.

### 6.11 Neural-Tool Agent Baseline

A strong boring baseline is a tool-using neural agent. It may use a neural core
or neural proposal model and may invoke ordinary tests, public verifiers,
debuggers, trace analyzers, repair tools, CEGIS/ILP/DreamCoder-style helpers,
SAT/SMT solvers, and proof assistants under the same CPU and interface budget as
PCCP-H.

Information parity rule:

```text
The neural-tool agent receives the same task, same examples, same public DSL or
an equivalent tool-facing representation, same public verifier interface, same
counterexample interface, same visible traces, and same hidden-family ignorance
as PCCP-H.
```

The neural-tool agent wins the head-to-head if it achieves equal or better:

- hidden-family accuracy;
- repair locality;
- inference cost after any allowed compilation or caching.

If it also requires less human-authored DSL/verifier/decomposition labor, the
PCCP-H result is demoted even if the compiled artifact is prettier. PCCP-H is
not allowed to win by giving the neural agent worse tools, weaker verifier
access, fewer traces, or no repair loop.

---

## 6A. Hybrid Evaluation: Substrate Shootout

Every report must score three configurations on the five sacred outcomes:

1. **Pure PCCP/PCCP-A:** non-neural synthesis core under a given verifier.
2. **PCCP-H:** neural or other perception/proposal front-end plus PCCP reasoning,
   verifier, proof/test obligations, executable artifact, and repair core.
3. **Neural-tool agent:** neural core with access to tools, tests, verifiers,
   debuggers, solvers, and repair loops.

Score each configuration from 1 to 5 on genuine intelligence, improvability,
democratized development, data efficiency, and inference efficiency. The winner
is whichever configuration best serves the outcomes, regardless of substrate.

Required table:

| Configuration | Genuine intelligence | Improvability | Democratized development | Data efficiency | Inference efficiency | Outcome winner? | Failure diagnosis |
|---|---:|---:|---:|---:|---:|---|---|
| Pure PCCP/PCCP-A | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| PCCP-H | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Neural-tool agent | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

A PCCP-H claim is only meaningful if the PCCP core contributes measurable value
over neural + ordinary tools and over existing synthesis systems with comparable
proposal guidance. If the neural-tool agent or prior-art hybrid wins, report that
as the result.

---

## 7. Smuggling Controls Checklist

Every PCCP report must include this checklist.

| Risk | Question | Required control | Fail consequence |
|---|---|---|---|
| DSL smuggling | Does the DSL contain the target rule or causal selector? | Shortest-program audit; primitive inventory; prior-art solver comparison | KILL_PCCP if yes |
| Verifier smuggling | Is the verifier an answer key available during synthesis? | Hidden verifier inaccessible until freeze; seen verifier logs all calls | KILL_PCCP if hidden answers leak |
| Generator smuggling | Does the generator expose causal structure directly? | Public/private generator split; no role labels; seed holdout | Invalidate benchmark |
| Transformation smuggling | Are interventions only cosmetic? | At least 3 structurally different hidden intervention families | VOID or redesign |
| Baseline handicapping | Do baselines get worse information than PCCP? | Information parity table for every baseline | KILL_PCCP if uncorrected |
| Human labor | Did humans design the rule, DSL, invariants, decomposition, or patch? | Human-labor ledger; separate given/discovered structure | Downgrade or kill depending on severity |
| Prior-art absorption | Is the best system just CEGIS/ILP/DreamCoder? | Strong synthesis baselines with same access | KILL_PCCP if PCCP does not beat or add something real |
| Neural-tool absorption | Does a neural-tool agent match or beat PCCP-H with the same interfaces? | Section 6.11 head-to-head; same verifier/tool access | KILL_PCCP if uncorrected |
| Decomposition smuggling | Did humans choose the useful subproblem boundaries? | Decomposition gate; labor ledger; human baseline | Downgrade or kill depending on severity |
| Toy triviality | Is the domain too small or linearly separable? | Minimum complexity gates; random-program baseline | VOID |
| Scaling collapse | Does performance disappear as variables, DSL, interventions, or interaction density grow? | Scaling gates and degradation report | VOID or demote to toy formal tool |
| Spec incompleteness | Does the verifier miss important failures? | Hidden interventions and adversarial counterexamples | VOID or KILL depending on claim |
| Neural loophole | Did the core claim become gradient representation learning with a program wrapper? | Criterion (f) substrate-balance score and neural-tool baseline | KILL_PCCP for core drift |

Human-labor accounting must classify every design choice:

```text
GIVEN_TO_ALL: public schema, DSL primitives, examples, verifier interface
DESIGNED_BY_HUMANS: generator class, target-function family, intervention grammar
DISCOVERED_BY_SYSTEM: selected variables, composed rule, subroutines, repairs
HIDDEN_FOR_EVAL: held-out worlds, interventions, seeds, exact labels
```

---

## 8. Metrics

### 8.1 Hidden-Family Accuracy

Primary accuracy:

```text
HFA = correct decisions on hidden worlds and hidden intervention families
      / total hidden evaluation cases
```

For PCCP_SIGNAL, hidden-family accuracy must be at least 0.95 and all verifier
obligations required for the claimed domain must pass. For finite exact domains,
zero verifier failures is the preferred bar.

### 8.2 Program Length

Report `L(P)`, `L(O)`, `L(C)`, `L(A)`, lookup-table length, and best baseline
program length where applicable.

PCCP_SIGNAL requires the PCCP artifact to be at least 10x shorter than the exact
lookup table on the evaluated finite domain.

### 8.3 Examples Needed

Measure the smallest number of labeled examples or counterexamples needed to
reach first seen-verifier pass, first hidden-family threshold pass, and stable
final artifact. Report active counterexample queries separately from passive
labeled examples.

### 8.4 Inference Cost

Measure AST interpreter steps, wall-clock CPU time for a fixed batch, memory
footprint, and subroutine calls. Inference cost excludes synthesis time. At
inference, the artifact may not call the verifier or solver.

### 8.5 Repair Locality

Given a counterexample `ce`, repair locality is:

```text
repair_edit_ratio = AST edit distance(P_before, P_after) / max(1, L(P_before))
behavior_delta = fraction of previously correct checked cases whose output changed
```

STRONG_PCCP requires at least one real counterexample repair with:

```text
repair_edit_ratio <= 0.20
behavior_delta <= 0.10
```

unless the artifact proves that the failure required a larger global change.

### 8.6 Verification Passes

Report type-check pass/fail, resource-bound pass/fail, seen-domain verifier
pass/fail, hidden-domain verifier pass/fail, proof-certificate check pass/fail,
property-test pass/fail, and counterexample localization pass/fail.

No artifact with failed proof obligations can receive PCCP_SIGNAL.

### 8.7 Explanation Quality

Explanation quality is secondary but required for democratized development.

Measure with a blinded human audit:

- Can a technically competent reader identify the decision rule from the
  artifact within 10 minutes?
- Can they predict the artifact's output on 5 new cases without running it?
- Can they identify which subroutine would need repair for a provided
  counterexample?
- Is the proof/check trace linked to concrete program parts?

Score:

```text
0 = unreadable
1 = executable but opaque
2 = readable rule, weak proof trace
3 = readable rule and localized proof/counterexample trace
```

PCCP_SIGNAL requires explanation quality >= 2. STRONG_PCCP requires >= 3.

---

## 9. Theorem Target

### 9.1 Required Theorem

PCCP-0 must include a finite-world theorem:

```text
There exist finite world families where reconstruction-optimal compression
provably discards the decision-preserving causal program, while a compact
causal program preserves the target function under intervention.
```

This theorem must be proved before empirical claims are made. A plot is not a
proof.

### 9.2 Minimal Construction

A sufficient 3-variable construction:

- `A`: causal binary variable, `A ~ Bernoulli(epsilon)`.
- `N`: nuisance vector, `N ~ Uniform({0,1}^m)`, independent of `A`.
- `S`: spurious binary variable, `S = A` in the observational training
  environment.
- Observation: `X = (A, N, S)` after an arbitrary public surface encoding.
- Target decision: `Y = A`.
- Hidden interventions:
  - `do(N := n')`: target unchanged.
  - `do(S := 1 - S)` or `do(S := random)`: target unchanged.
  - `do(A := a')`: target becomes `a'`.

Reconstruction baseline:

- Code budget: `m` bits.
- Objective: minimize expected Hamming reconstruction error on `(A, N, S)` under
  the observational distribution.

Decision-preserving program:

```text
P(X, q, i) = value of A after applying intervention i
```

with obligations:

```text
invariant to N
invariant to S
covariant with do(A)
```

### 9.3 Proof Sketch To Be Formalized

Let `epsilon < 1/4`.

1. A compressor that stores `N` exactly and decodes `A_hat = 0`, `S_hat = 0` has
   expected reconstruction error `2 * epsilon`, from the rare cases where
   `A = S = 1`.
2. Any code that preserves `A` for all `N` under the `m`-bit budget must spend at
   least one bit distinguishing `A`, so it cannot store all `m` nuisance bits.
   It therefore incurs at least `1/2` expected Hamming error on one nuisance bit.
3. Since `2 * epsilon < 1/2`, reconstruction-optimal compression stores `N` and
   discards `A`.
4. Any decision rule derived only from that reconstruction code has no
   information about `A` under the balanced intervention `do(A := a')`, so it
   fails the causal decision function.
5. The executable program that selects `A` is constant-size and passes the
   intervention verifier.

The final theorem artifact must state the assumptions precisely and either prove
the lower bound analytically or exhaustively enumerate the finite encoder class
for the chosen `m`.

### 9.4 Why This Matters

This construction separates surface reconstruction, observational correlation,
decision preservation, and intervention robustness. It is the smallest proof
target for the kill-history lesson: a system can compress the visible world well
while discarding the small causal bit that matters for the function.

---

## 10. Verdict Tokens (Precommitted)

Token names are retained for continuity. Unless a run is explicitly labeled
pure PCCP/PCCP-A, the verdict tokens evaluate PCCP-H against prior-art synthesis
and neural-tool baselines under equal information.

### 10.1 PCCP_SIGNAL

Award only if all conditions hold:

- CPU-only result.
- Hidden-family accuracy >= 0.95, preferably exact verifier pass.
- Compact executable artifact beats memorization, reconstruction, decision tree,
  tiny neural, neural-tool agent, random program, and at least one strong
  synthesis baseline.
- Artifact is at least 10x shorter than exact lookup table.
- DSL/verifier/generator/decomposition/baseline smuggling controls pass.
- Explanation quality >= 2.
- Any gradient-trained component is isolated, measured, and shown not to be the
  whole source of the claimed proof-carrying executable advantage.

### 10.2 STRONG_PCCP

Award only if PCCP_SIGNAL holds plus:

- Local counterexample repair succeeds under the repair-locality thresholds, or
  the artifact gives an exact characterization of why the needed repair is
  global.
- A finite theorem or exact characterization explains why proxy or
  reconstruction compression fails.
- Scaling curves report degradation across causal-variable count, DSL primitive
  count, intervention family count, and rule interaction density.
- PCCP-H beats or materially extends generic CEGIS/ILP/SAT/SMT/symbolic
  regression, DreamCoder-style systems, and the neural-tool agent under equal
  information.

### 10.3 MOONSHOT_PCCP

Award only if STRONG_PCCP holds plus:

- The system helps construct, refine, or select correctness obligations rather
  than merely satisfying human-given obligations.
- Verifier/spec discovery is tested on held-out families and compared against
  invariant/spec-mining baselines.
- The system proposes useful partial verifiers, uncertainty boundaries, or
  decompositions for at least one messy partially specified task.
- Human-labor accounting shows the system discovered nontrivial obligations,
  invariants, or decompositions.

This is the PCCP-B extension inside PCCP-H. It is no longer allowed to remain a
vague future promise; PCCP-0 includes a restricted mini-gate.

### 10.4 KILL_PCCP

Assign KILL_PCCP if any of the following occur:

- The artifact wins only because the DSL contains the hidden rule, target
  selector, or causal-role oracle.
- The verifier exposes hidden answers during synthesis.
- The generator exposes causal structure directly.
- The best baseline is an existing synthesis system and PCCP-H does not beat it
  or add proof/repair/composition/decomposition value.
- The neural-tool agent achieves equal or better hidden-family accuracy, repair
  locality, and inference cost under equal information.
- PCCP-H collapses back into gradient-trained representation learning as the
  core mechanism with a proof/program wrapper.
- Baselines are materially handicapped relative to PCCP-H.
- Human-authored invariants, decompositions, or patches do the work claimed as
  system discovery.

### 10.5 VOID

Assign VOID if results are inconclusive, hidden interventions are too weak or
cosmetic, the benchmark is too small or too easy, a bug/leak/evaluator issue
invalidates the run, or no baseline comparison is trustworthy.

VOID means redesign the benchmark. It is not evidence for PCCP.

---

## 11. Criterion (f): Substrate Balance

Criterion (f) is not "non-neural good, neural bad." It is:

```text
Does the core intelligence claim require gradient-trained representations?
```

Scoring:

| Score | Meaning |
|---:|---|
| 0 | Core claim is gradient-trained representation learning; PCCP framing is cosmetic |
| 1 | Neural components supply the essential rule or verifier; program layer is mostly wrapper |
| 2 | Neural components serve perception/adapter roles; typed PCCP core still carries the claim |
| 3 | Core result survives with neural parts removed; neural parts are optional or improve interfaces |

Rules:

- Do not penalize neural components that genuinely serve the five sacred
  outcomes.
- Do not favor non-neural components merely for being non-neural.
- Always report: "If neural parts are removed, what claim remains?"
- A tiny neural baseline may beat PCCP. If it does, report that honestly.
- PCCP-H may use neural perception or proposal adapters, but PCCP-0 and PCCP-1
  must still establish what the executable/verifiable core contributes when the
  neural parts are removed or replaced.

---

## 12. Roadmap

### 12.1 PCCP-0: Finite-World Theorem + Toy Benchmark

Goal:

- Prove the finite reconstruction/function separation theorem.
- Build the exact benchmark specification.
- Run only CPU finite-world tests after this spec passes design review.

Required before running:

- final theorem statement;
- generator contract;
- DSL primitive inventory;
- verifier contract;
- baseline information-parity table including the neural-tool agent;
- smuggling checklist;
- hidden split manifest;
- verdict-token thresholds.

Restricted verifier-discovery mini-gate required in PCCP-0:

- The system is given examples, counterexample traces, and a partial verifier
  that is missing at least one important property.
- The system must propose at least one additional property, invariant,
  metamorphic relation, or obligation from the traces.
- The proposed property is frozen and tested on hidden families.
- Compare against Daikon-style dynamic invariant detection, ICE/Horn-ICE
  invariant learning where applicable, and random property generation.
- Success means the proposed property catches hidden failures or localizes repair
  better than the partial verifier alone and better than the listed baselines.
- Failure means the property is superficial, overfits seen traces, duplicates
  the given verifier, or becomes another proxy that hidden tests defeat.

Gate to PCCP-1:

- PCCP_SIGNAL or a clear VOID with redesign path.
- Verifier-discovery mini-gate result reported, even if negative.
- No KILL_PCCP condition.

### 12.2 PCCP-1: Richer Worlds

Goal:

- More variables.
- Composition.
- Subroutine reuse.
- Multiple query types.
- Multi-step interventions.
- Stronger prior-art baselines.

New required evidence:

- library/subroutine reuse reduces length;
- local repair works across more than one world family;
- generic synthesis baselines do not fully absorb PCCP.

Gate to PCCP-2:

- STRONG_PCCP on at least one nontrivial finite suite.
- Explanation quality >= 3.
- Prior-art absorption risk addressed directly.

### 12.3 PCCP-2 / PCCP-H: Perception And Proposal Bridge

Goal:

- Convert raw observations into typed terms.
- Allow neural or other adapters if they improve perception, proposal diversity,
  compression, or usability.
- Keep the PCCP core verifier-first and executable.
- Compare PCCP-H directly against the neural-tool baseline.

Neural adapters may be used for perception, embedding-to-symbol proposals,
candidate feature extraction, noisy observation parsing, program proposals, and
repair suggestions. They may not replace the target verifier, proof obligations,
executable causal program, hidden-family intervention test, or human-labor
ledger.

Gate to PCCP-B:

- Removing the neural adapter degrades perception or proposal quality but leaves
  a measurable typed PCCP core claim intact.
- Adapter errors are separately measured and not confused with core reasoning
  failures.
- PCCP-H beats neural + ordinary tools/tests on at least one precommitted
  outcome-relevant axis, or the result is reported as neural-tool absorption.

### 12.4 PCCP-B: Verifier Discovery

Goal:

- The system helps discover or refine correctness obligations.
- It proposes invariants, counterexample classes, intervention families, or
  verifier clauses.

This is the moonshot extension. PCCP-A with a given verifier may be useful, but
PCCP-B is where the direction becomes more than program synthesis under a human
spec.

Gate for MOONSHOT_PCCP:

- system-proposed obligations catch failures humans did not explicitly encode;
- obligations transfer to held-out families;
- human-labor accounting shows nontrivial verifier work was system-discovered.

### 12.5 Cross-Cutting Gates: Decomposition And Scaling

The decomposition gate in Section 3.9 and scaling gates in Section 3.10 are not
optional narrative decorations. They are required evidence for any claim broader
than a verifier-rich toy world.

- A pure formal-core result may proceed without passing the decomposition gate,
  but it must be labeled narrow PCCP/PCCP-A.
- A PCCP-H mainline claim requires the decomposition gate to add value over no
  decomposition and over the neural-tool agent.
- A scaling claim requires reported degradation curves and a stated transition
  point if search, repair, verifier coverage, or human labor collapses.

---

## 13. Prior-Art Absorption And Novelty Declaration

PCCP-H is close to existing work. The spec explicitly does **not** claim novelty
for:

- CEGIS or OGIS counterexample loops;
- ILP rule induction, predicate invention, or relational background knowledge;
- DreamCoder-style abstraction and library learning;
- symbolic regression or compact expression search;
- SAT/SMT/SyGuS-style bounded synthesis and formal verification;
- causal discovery, structural causal models, interventions, or causal
  abstraction;
- proof-carrying code;
- property-based testing, metamorphic testing, dynamic invariant detection,
  specification mining, Daikon-style invariant mining, or ICE/Horn-ICE invariant
  learning;
- neural tool use, neural proposal guidance, or learned search heuristics.

The only possible novelty claim is the combined artifact contract applied to
intelligence:

```text
function-aligned executable compression + proof/test obligations +
hidden-intervention survival + local repair + human-labor accounting
```

This contract must be evaluated as a discipline, not as a claim that each organ
is new. Existing prior-art systems may be used as implementation substrates if
they produce the required artifact and win the outcome tests.

What remains to be proven:

1. The combined contract beats each relevant prior-art baseline under equal
   information, not merely weak neural baselines.
2. PCCP-H adds measurable value over a neural-tool agent with the same verifier,
   tests, repair tools, and traces.
3. The verifier-discovery and decomposition gates discover useful structure
   rather than smuggling human judgment or learning another proxy.
4. Scaling degradation is acceptable in the structured-world regimes where the
   claim is made.

If none of these is true, PCCP-H is good formal-tools engineering discipline,
not a moonshot direction. In that case the honest move is to adopt whichever
prior-art or neural-tool substrate wins.

---

## 14. Narrative Gate

### 14.1 Gossip-Magazine One-Sentence Story

Given only this specification:

```text
A laptop-scale hybrid AI has to learn the rulebook, prove what it knows, survive
hidden interventions, and fix broken rules without retraining; the surprise only
counts if it beats both ordinary neural tool use and ordinary program synthesis.
```

### 14.2 Does It Survive "That's Just CEGIS With A Verifier"?

Conditional, and the burden is on PCCP-H.

It does **not** survive if the result is merely a candidate generator, a verifier,
and counterexamples over a hand-authored DSL. That is CEGIS/SyGuS-shaped prior
art and should be named as such.

It survives only if the final artifact contract adds measured value that generic
CEGIS, ILP, DreamCoder-style search, symbolic regression, SAT/SMT synthesis, and
spec-mining baselines do not match under equal information: hidden-intervention
survival, proof/test obligations, shorter executable artifacts, local repair,
useful decomposition, verifier discovery, or lower deployed inference cost.

### 14.3 Does It Survive "The Neural-Tool Agent Already Does This"?

Conditional, and this is the strongest boring objection.

It does **not** survive if a neural agent with the same verifier interface,
tests, debuggers, solvers, traces, and repair loop reaches equal or better
hidden-family accuracy, repair locality, and inference cost. In that case the
PCCP-H mainline is absorbed by neural-tool engineering.

It survives only if the PCCP core contributes something the neural-tool agent
lacks: a more compact compiled artifact, cleaner proof/test obligations, better
counterexample-localized repair, better hidden-intervention transfer, lower
repeat inference cost, or less human-authored structure for the same outcome.

### 14.4 Honest Narrative Verdict

The specification alone is not a result. It is not yet a moonshot.

The honest story is alive but harder:

```text
If PCCP-H beats weak neural baselines on a tiny formal puzzle, demote it. If it
beats strong prior-art synthesis and neural-tool agents under hidden
interventions, with smuggling audits, scaling curves, local repair, and at least
one useful discovered verifier or decomposition, then the direction has a real
paradigm-level signal. If not, kill the acronym and keep the winning tools.
```
