# Q-Loop B29: Test the Existential Threats to PCCP-H

**Date:** 2026-07-07
**Role:** Question Loop worker
**Iterations:** I197-I203
**Status:** analysis-only existential-threat test; CPU-only constraint; no implementation, no training, no experiments, no web runs.

---

## Grounding

Read in the requested order:

1. `research/VISION.md`
2. `research/PCCP_PRECOMMIT_SPEC.md`
3. `research/PCCP_THEOREM_DRAFT.md`
4. `research/question_loop_batch27.md`
5. `research/question_loop_batch28.md`
6. `research/dual_loop_supervisor_checkin_18.md`
7. `research/dual_loop_supervisor_checkin_19.md`
8. `code/pccp0_witness.py`
9. `research/DEEP_RETHINK.md`

Additional process grounding:

- `C:\Users\devan\.claude\projects\C--Users-devan-OneDrive-Desktop-Projects-AI-Moonshots-moonshot-sutra\memory\MEMORY.md`

Binding facts:

- The two invariants remain fixed: swing for the home run, and the loop stops only when an adversary cannot knock it down.
- The five sacred outcomes remain genuine intelligence, improvability, democratized development, data efficiency, and inference efficiency.
- The substrate is open. Neural, symbolic, programmatic, proof-based, causal, search, hybrid, and unknown substrates are evaluated by outcome.
- The kill history's central lesson is proxy/function divergence: repeated mechanisms improved BPB, reconstruction, coordinate compatibility, readout, or smooth-law proxies while task function did not move.
- PCCP-H is now an after-frame artifact contract plus a discovery ambition. The after-frame story is complete enough for the current phase: theorem Parts 1-2 are clean, Part 3 is restricted exact learning, and `code/pccp0_witness.py` demonstrates the narrow STRONG_PCCP finite witness.
- The current moonshot question is no longer "can a compact proof-carrying causal program beat surface reconstruction after the verifier is supplied?" The answer is yes in the finite witness.
- The current moonshot question is: can a cheap system discover useful obligations, metamorphic relations, intervention clauses, or decomposition boundaries that humans did not already smuggle into the frame?
- B28 designed FDM-0 as active effect screening plus clause mining. It did not yet test whether FDM-0 is just exhaustive perturbation testing, or whether a neural-tool agent absorbs the discovery role.
- Supervisor #19 makes this batch's burden explicit: the future depends on whether FDM-0 discovers useful obligations that prior art and neural tools cannot.

Current strongest position to stress-test:

```text
FDM-0 is the first concrete PCCP-B mechanism: in finite typed worlds, it uses
generic perturbations, target effect signatures, clause scoring, and verifier
compilation to discover missing obligations before the human writes them.
```

Existential warning:

```text
If FDM-0 only finds single-field invariances under a supplied perturbation API,
it is absorbed by exhaustive perturbation testing. If a neural-tool agent finds
the same clauses under equal information, PCCP-H is a compiler/audit discipline,
not a discovery paradigm.
```

---

## I197: The Trivial-Discovery Stress Test

### Steelman

The strongest attack is not vague. It is an exact algorithm.

Call it **Exhaustive Single-Field Effect Screening**:

```text
Input:
  examples E = {(x_i, y_i)}
  fields f_1...f_n with finite domains D_1...D_n
  allowed perturbation operator replace(x, f_j, v)
  target membership oracle or public seen verifier label oracle

For each field f_j:
  For each example x_i:
    For each value v in D_j \ {x_i[j]}:
      x' = replace(x_i, f_j, v)
      query y' = F(x')
      record relation between y' and y_i

Classify f_j as:
  invariant if y' = y_i for all queried pairs
  covariant under phi_v if y' = phi_v(y_i) for all queried pairs
  unstable otherwise
```

Query complexity:

```text
Q_B1 = sum_j |E| * (|D_j| - 1)
     = O(|E| * sum_j |D_j|)
```

For `n` binary fields this is `O(n * |E|)`. If `E` is the full finite admissible domain, the result is an exact certificate for all single-field perturbations in the supplied perturbation grammar. If `E` is a sampled seen set, it is a statistical effect screen with ordinary finite-sample failure risk.

On the current PCCP-0 witness family, this is devastatingly strong. The witness has role-permuted observed fields, two causal bits, nuisance bits, and one spurious bit. If the screening routine can make valid interventions or target-label queries on perturbed cases, it discovers:

| Role | Single-field screen result |
|---|---|
| `N_j` nuisance bit | invariant: changing it leaves output fixed |
| `S` spurious bit | invariant under intervention despite observational correlation |
| `C0` causal bit | covariant: setting/flipping it changes parity according to output XOR |
| `C1` causal bit | covariant: same |

This discovers the B1 obligations FDM-0 was expected to find:

```text
invariant_to(N_j)
invariant_to(S)
covariant_with(C0 := v)
covariant_with(C1 := v)
```

It also rejects the bad shortcut `P_bad = S` once `invariant_to(S)` is compiled into the verifier.

This is not a toy objection. It means the first FDM-0 demo from B28 is probably absorbed if it is limited to single-field invariance/covariance over a supplied finite perturbation API.

### Threat Question

Is exhaustive single-field perturbation sufficient to discover **all** obligations FDM-0 would find in PCCP-0 finite worlds?

Answer:

```text
Yes for B1 obligations that are expressible as total unary effect signatures
over supplied fields, supplied valid perturbations, and supplied target labels.
No for obligations whose truth depends on relations, conditions, compositions,
latent encodings, unknown transformations, or decomposition boundaries.
```

More precise boundary:

| Discovery target | Exhaustive single-field screen finds it? | Reason |
|---|---:|---|
| Total field invariance | Yes | Directly checks `F(x) = F(replace_j(x,v))`. |
| Total unary covariance | Yes, if output transform grammar is supplied | Checks `F(replace_j(x,v)) = phi_v(F(x))`. |
| Spurious role in PCCP-0 | Yes | Spurious is intervention-invariant. |
| Causal role in Boolean parity PCCP-0 | Yes | Causal bit has stable unary covariance. |
| Nuisance role in PCCP-0 | Yes | Nuisance bit has stable unary invariance. |
| Pair-only relation | No | No single field may reveal the relation alone. |
| Conditional invariance | Not reliably | Need precondition discovery. |
| Equivariance under permutations | Only if each permutation is supplied | Field replacement is not group-action discovery. |
| Monotonicity | Only if ordered adjacent changes are tested | Needs order structure and relation grammar. |
| Precondition boundary | No, except by brute boundary enumeration | Needs candidate boundary predicates. |
| Decomposition boundary | No | Needs dependency graph or higher-order interaction analysis. |
| Hidden latent cluster | No | Individual observed fields may be uninformative. |
| Unknown valid transformation | No | Perturbation grammar is the frame. |

### What We Might Be Missing

The biggest hidden cost is not the loop over fields. It is the validity and grounding of the perturbation.

If `replace(x, f_j, v)` creates an impossible observation off the world manifold, the target oracle may be undefined or meaningless. The trivial algorithm is valid only when the benchmark supplies an admissible intervention semantics:

```text
do(field_j := v) is a valid intervention and F(x') is queryable.
```

That is already a major piece of the frame. If humans supplied the valid intervention grammar, they supplied much of the discovery substrate.

### Verdict

```text
B1_DISCOVERY_IS_ABSORBED_BY_EXHAUSTIVE_SINGLE_FIELD_EFFECT_SCREENING.
```

If FDM-0 only implements this layer, it adds useful engineering around logging, scoring, compilation, and smuggling audits, but it does not add a discovery paradigm. The honest move is to concede B1 absorption and move the live claim to B2 relation discovery and B3 decomposition discovery.

---

## I198: What Discovery Is NOT Trivial?

### Steelman

Single-field invariance being trivial does not make frame formation trivial. It just locates the first nontrivial boundary.

The smallest nontrivial problems are those where the missing obligation is not:

```text
changing field f_j preserves output
```

but instead:

```text
changing a structured part of the input induces a structured transformation of
the output, under a condition, inside a component, or across composed actions.
```

The useful taxonomy:

| Class | Example obligation | Query complexity if grammar is supplied | Why it matters |
|---|---|---:|---|
| Unary covariance | `flip(C0) -> flip(y)` | `O(|E| * n * d * |Phi|)` | Causal effect laws, not nuisance invariance. |
| Pairwise relation | `flip(C0,C1) -> identity(y)` | `O(|E| * n^2 * d^2 * |Phi|)` | Interactions, parity, compensation, conservation. |
| Conditional invariance | `field_j invariant only when mode=m` | `O(|E| * |T| * |G_pre|)` | Medical/financial/access rules are mostly conditional. |
| Compositional rule | `tau_a then tau_b -> phi_b(phi_a(y))` | `O(|E| * |T|^2 * |Phi|)` | Multi-step interventions, programs, protocols. |
| Permutation equivariance | `rename users -> rename outputs` | `O(|E| * |G|)` or `O(|E| * generators)` | Names, IDs, ordering, graph isomorphism. |
| Monotonicity | `increase dose/risk/permission -> nondecrease score` | `O(|E| * n * (d-1))` for adjacent checks | Triage, thresholds, ranking, resource allocation. |
| Precondition boundary | `rule valid iff amount <= limit` | `O(|E| * n log d)` if monotone, otherwise `O(|E| * d)` | Most real specs have validity domains. |
| Decomposition boundary | `fields A affect suboutput A only` | first-order `O(|E| * n * d)`, interactions `O(|E| * n^2 * d^2)` | Scales synthesis and local repair. |

The first real jump is not from invariance to "anything." It is from **unary effect signatures** to **metamorphic relation discovery**:

```text
Find tau over inputs and phi over outputs such that F(tau(x)) = phi(F(x)).
```

This is still finite and CPU-testable, but it is qualitatively different from B1 because the system must discover both sides of the relation.

### Threat Question

Which of these matters for real intelligence tasks, and which is cheaply testable?

Real tasks rarely hinge on naked field invariance. They hinge on:

1. Conditional invariances: an irrelevant variable becomes relevant under a mode, exception, or subgroup.
2. Precondition boundaries: a rule is valid inside a finite region and invalid outside it.
3. Decomposition boundaries: different components, patients, accounts, users, or modules obey different local rules.
4. Equivariance: renaming, reordering, or duplicating entities should transform the answer predictably.
5. Monotonicity: increasing a risk, permission, amount, or dependency should move a decision in a constrained direction.
6. Compositional rules: two safe transformations can interact unsafely.

Cheaply testable:

```text
Unary covariance, monotonicity over declared orders, small pairwise relations,
and finite permutation-generator checks.
```

Not cheaply testable without strong assumptions:

```text
High-order interactions, arbitrary precondition grammars, dense decompositions,
and unknown transformation grammars.
```

### What We Might Be Missing

There is a danger of moving the goalpost too far. B2 should not be "discover any relation." It should be a finite, predeclared class:

```text
input transforms T from a typed generic grammar
output transforms Phi from a small relation grammar
score relation tau, phi by exact paired-label agreement, MDL length,
negative controls, and hidden-family transfer
```

If the relation class is not predeclared, the system can overfit. If the class is too specific, humans smuggled the answer.

### Verdict

```text
THE LIVE DISCOVERY CLAIM STARTS AT B2, NOT B1.
```

B1 is testing hygiene. B2 is the smallest real discovery target. B3 is the first possible paradigm-level target, because decomposition discovery is where search complexity, local repair, and human frame labor actually change.

---
## I199: The Neural-Tool Baseline Protocol

### Steelman

The neural-tool baseline should be treated as the strongest boring explanation, not as a nuisance.

Protocol name:

```text
NTB-0: Neural-Tool Baseline for PCCP-B Clause Discovery
```

The baseline gets a sealed public task bundle:

```text
public_bundle/
  README_TASK.md
  dsl.py
  interpreter.py
  partial_verifier.py
  seen_cases.jsonl
  seen_counterexamples.jsonl
  perturbation_api.py
  query_oracle.py
  clause_schema.json
  budget.json
  submit_clauses.py
```

The hidden evaluator is absent:

```text
hidden_worlds.jsonl          # not present
hidden_verifier.py           # not present
role_to_obs.json             # not present
target_rule.py               # not present
heldout_labels.jsonl         # not present
```

Allowed tools:

- code execution;
- ordinary Python analysis;
- SAT/SMT/CEGIS helpers if installed;
- Daikon-style invariant mining if available;
- custom perturbation scripts;
- solver calls within the same CPU budget;
- counterexample feedback through the public seen verifier;
- the same `query_oracle.py` calls that FDM-0 receives.

Forbidden:

- hidden evaluator access;
- role labels;
- generator sampled parameters;
- hidden family labels;
- inspecting private seeds;
- human hints after the run begins;
- prose-only clauses that do not compile.

The agent's required output:

```text
clauses.json:
  [
    {
      "id": "...",
      "kind": "invariance|covariance|equivariance|monotonicity|precondition|decomposition",
      "input_transform": {...},
      "output_relation": {...},
      "precondition": {...} | null,
      "support_evidence": [...],
      "expected_failure_caught": "...",
      "uncertainty": "..."
    }
  ]

optional verifier_patch.py:
  compile clauses into executable checks under the public clause schema
```

### Threat Question

What does "equal information" mean precisely?

It means:

| Information or resource | FDM-0 | Neural-tool baseline |
|---|---:|---:|
| Public DSL/interpreter | Same | Same |
| Seen examples/traces | Same | Same |
| Partial verifier V0 | Same | Same |
| Perturbation grammar | Same | Same |
| Query budget | Same maximum calls | Same maximum calls |
| Counterexample interface | Same | Same |
| Solver/tool access | If FDM-0 uses it, NTB-0 may use it | If NTB-0 uses it, logged |
| Hidden worlds/families | No | No |
| Role labels | No | No |
| Clause schema | Same | Same |
| Freeze rule | Clauses frozen before hidden evaluation | Same |
| Human labor after start | None | None |

Equal information does **not** mean equal implementation style. The neural agent may reason in prose, write scripts, run exhaustive perturbation tests, call solvers, inspect traces, and decide which clauses to submit. That is the threat.

The baseline wins if it achieves equal or better:

1. hidden failure catch rate;
2. hidden false-positive rate;
3. hidden-family transfer;
4. verifier compilation pass;
5. repair localization;
6. clause length / MDL score;
7. total oracle calls;
8. total wall-clock CPU time;
9. human-labor cost;
10. reproducibility across role/name randomization.

FDM-0 wins only if it is better on at least one meaningful axis while not losing the core correctness axes:

```text
same or better hidden transfer, fewer oracle calls, lower cost, more stable
under randomization, cleaner compiled obligations, or better repair localization.
```

### What We Might Be Missing

The neural-tool baseline can absorb both trivial and nontrivial discovery because it can run the same algorithms as subroutines.

A fair GPT-5-style tool user can write:

```text
for each candidate transform tau:
  query paired labels
  test identity/not/affine/permutation output relations
  rank clauses by support and simplicity
  compile the winning relation into the verifier schema
```

That is already FDM-0 unless FDM-0 has a better search rule, better active-query policy, stronger guarantees, or lower cost.

### Verdict

```text
NEURAL_TOOL_BASELINE_IS_EXISTENTIAL_AND_MUST_RUN BEFORE DISCOVERY_SIGNAL.
```

If NTB-0 matches FDM-0, PCCP-H is not dead as an artifact contract. It becomes:

```text
neural/tool proposal -> PCCP verifier/compiler/audit layer -> compact executable artifact
```

That may still be valuable for proof, repair, audit, and cheap repeated inference. But the claim "cheap formal discovery mechanism" is dead unless FDM-0 beats the neural-tool baseline under equal information.

---

## I200: Real-World Finite Domains Where Frame Formation Matters

### Steelman

There are real finite or effectively bounded domains where the correct function is hidden, intervention-like checks are possible, existing tools are insufficient, and cheap verifier discovery would be valuable.

Concrete candidates:

| Domain | Finite or bounded? | Hidden but checkable function | Why existing tools are insufficient | Useful discovery |
|---|---:|---|---|---|
| API protocol verification | Finite state machines over calls/flags | Tests, reference implementation, or maintainer oracle | Fuzzers find crashes; specs are missing or stale | Temporal preconditions, state decompositions, invalid call sequences. |
| Smart contracts | Bounded transaction sequences and state variables | EVM execution plus economic/security oracle | Tools test supplied invariants; missing invariant is the failure | Conservation laws, role equivariance, reentrancy preconditions. |
| Access control / IAM | Finite users, roles, resources, actions | Admin decisions or policy simulator | Logs encode historical bugs; prose policies ambiguous | Role equivalence, deny/allow monotonicity, separation-of-duty clauses. |
| Medical protocol checklists | Bounded patient fields, labs, meds, thresholds | Guideline adjudication or clinician review | Rule interactions and exceptions are under-specified | Contraindication boundaries, conditional invariances, monotone risk rules. |
| Financial compliance | Finite transaction/account features | Regulation, audit decisions, backtests | Static rules miss interaction clauses and exceptions | Threshold preconditions, entity decomposition, invariant audit totals. |
| Data pipelines / ETL | Finite schemas, transforms, records in test fixtures | Golden output, downstream acceptance, differential old/new runs | Unit tests assert examples, not metamorphic guarantees | Row permutation invariance, duplicate handling, null boundary clauses. |
| Config/security policy | Finite config keys and connectivity outcomes | Simulator, model checker, or penetration test | Human intent is not fully encoded in tests | Monotone permission effects, forbidden reachability, component boundaries. |
| Robotics/safety interlocks in small controllers | Finite modes/sensors/actions | Simulator or hardware-in-loop oracle | Verification depends on unknown mode interactions | Mode preconditions, fail-safe invariants, decomposition of subsystems. |

These domains satisfy the four requested criteria in a limited but real sense:

1. The domain can be bounded for a verification artifact.
2. The correct function is not fully known as a spec but can be queried by intervention, simulator, reviewer, or differential run.
3. Existing specification tools need human-written properties or mine observed behavior that may include bugs.
4. A cheap discovered obligation can prevent expensive failures or localize repairs.

### Threat Question

Do these domains make PCCP-H a paradigm shift?

Not automatically.

They make PCCP-H valuable if it discovers B2/B3 structure:

```text
conditional obligations, metamorphic relations, precondition boundaries,
equivariances, conservation laws, and decompositions.
```

They do not make PCCP-H valuable if it only discovers:

```text
field X does not affect output Y
```

because that layer is already cheap perturbation testing.

### What We Might Be Missing

Many real-world "correct functions" are not hidden natural laws. They are institutional choices.

For medicine, finance, access control, and law-adjacent rules, the right output may be contested, updated, or context-dependent. PCCP-H should not claim to discover "the true verifier" there. The honest claim is narrower:

```text
Given an adjudication source, simulator, review process, or trusted differential
oracle, discover candidate partial obligations and uncertainty boundaries that
humans can audit before compilation.
```

### Verdict

```text
REAL_FINITE_DOMAINS_EXIST, BUT THEY REQUIRE B2/B3 DISCOVERY TO MATTER.
```

The best early applied targets are not broad medical or legal judgment. They are bounded software, smart-contract, IAM, data-pipeline, and protocol domains where intervention queries are cheap and obligations can compile into tests.

---

## I201: The B2 Experiment - Metamorphic Relation Discovery

### Steelman

B2 should test relation discovery, not just invariance.

Minimal finite world:

```text
Latents:
  C0, C1 in {0,1}
  N0...Nk nuisance bits
  S spurious bit

Observation:
  role-permuted and optionally encoded fields x0...xn

Target:
  y = C0 XOR C1

Partial verifier V0:
  checks seen examples
  checks one easy nuisance invariance
  does not include covariance or pair-composition obligations

Missing obligations:
  flip(C0) -> flip(y)
  flip(C1) -> flip(y)
  flip(C0, C1) -> identity(y)
  change(S) -> identity(y)
```

The relation `flip(C0) -> flip(y)` is not field invariance. It is covariance:

```text
F(tau_C0(x)) = NOT(F(x))
```

FDM-0 can be extended to B2 as **Relation Miner v0**:

```text
Inputs:
  candidate input transformations T
  candidate output transformations Phi
  examples E
  target query oracle for paired cases

For tau in T:
  for x in E:
    y = F(x)
    y_tau = F(tau(x))
  for phi in Phi:
    score(tau, phi) = Pr_x[y_tau = phi(y)]
  accept (tau, phi) if:
    score = 1 on exact seen domain, or above threshold with confidence
    relation is short under MDL
    relation is nonredundant with V0
    negative controls do not also pass
    relation transfers to held-out seen worlds before hidden freeze
```

For binary output:

```text
Phi = {identity, NOT}
```

For modular output:

```text
Phi = {y -> y + b mod r}
```

For categorical output:

```text
Phi = small learned label permutations or a predeclared permutation grammar
```

Query complexity:

```text
Q_B2 = |E| * |T|

Score cost = O(|E| * |T| * |Phi|)
```

If `T` contains all single-field toggles and pair toggles over `n` binary fields:

```text
|T| = n + n(n-1)/2
Q_B2 = O(|E| * n^2)
```

If relation arity is bounded by `k` and each field has domain size at most `d`:

```text
|T_k| = O(n^k * d^k)
Q_B2 = O(|E| * n^k * d^k)
```

This is CPU-feasible for `k <= 2` and small finite worlds. It becomes exponential when `k` is unbounded.

### Threat Question

Can FDM-0 be extended to find these, and what would count as non-absorbed?

Yes, FDM-0 can be extended mechanically. But the absorption baseline is now:

```text
exhaustive metamorphic relation mining over the same T and Phi.
```

FDM-0 is non-absorbed only if it adds at least one of:

1. active query selection that reduces `Q_B2` compared with exhaustive mining;
2. MDL scoring that selects the transferable relation among many spurious relations;
3. composition closure checks, e.g. `tau_a ; tau_b -> phi_b o phi_a`;
4. hidden-family transfer under role/name randomization;
5. better compiled verifier clauses and repair localization;
6. lower human-labor cost for the same relation class.

Otherwise B2 is also absorbed, just by a stronger baseline:

```text
metamorphic-relation mining with exact paired labels.
```

### What We Might Be Missing

B2 relation discovery can be fooled by finite coincidences. In a small world, many false relations hold accidentally.

Required controls:

1. negative-control fields and transforms;
2. role randomization across worlds;
3. held-out seen worlds before hidden freeze;
4. exact support counts, not just top scores;
5. minimality preference: reject long relations if a shorter one explains the same effect;
6. relation ablation: show V0 misses a failure and V0 plus relation catches it.

### Verdict

```text
B2_IS_THE_RIGHT_NEXT_DISCOVERY_EXPERIMENT, BUT ITS BASELINE IS EXHAUSTIVE_METAMORPHIC_RELATION_MINING.
```

The next experiment should not stop at "FDM-0 found invariant_to(S)." It should force FDM-0 to find at least one covariance or composition relation and compare it against exhaustive relation mining and NTB-0.

---
## I202: The Decomposition Question

### Steelman

Decomposition discovery is where PCCP-H could stop being a clause miner and become a search-space transformer.

Minimal B3 finite demo:

```text
Latents:
  Component A: A0, A1, AN0...ANk, AS
  Component B: B0, B1, BN0...BNk, BS

Targets:
  y_A = f_A(A0, A1)
  y_B = f_B(B0, B1)
  y_combo = compose(y_A, y_B)

Queries:
  q_A, q_B, q_combo

Observation:
  all fields shuffled and encoded into one flat vector

Partial verifier V0:
  seen labels and some generic interventions
  no component boundary
```

The system must propose:

```text
boundary_A = {fields that affect q_A and A-side combo behavior}
boundary_B = {fields that affect q_B and B-side combo behavior}
composition_rule = y_combo = compose(y_A, y_B)
```

A concrete algorithm:

```text
1. Estimate first-order sensitivity matrix M[field, query/output].
2. Estimate pairwise interaction tensor I[field_i, field_j, query/output].
3. Build a graph with fields as nodes.
4. Add edge i-j when fields share output effects or have nonzero interaction.
5. Cluster graph into components.
6. Search small composition rules over component outputs.
7. Compile component-local obligations:
     perturb outside component A leaves y_A invariant
     perturb inside A affects only y_A and combo through compose
     same for B
```

First-order query complexity:

```text
Q_first = O(|E| * n * d * r)
```

where `r` is the number of query/output channels.

Pairwise interaction complexity:

```text
Q_pair = O(|E| * n^2 * d^2 * r)
```

For interaction width `k`:

```text
Q_k = O(|E| * n^k * d^k * r)
```

This is tractable only under sparsity and bounded interaction width. Arbitrary decomposition discovery is exponential and underdetermined.

### Threat Question

If a task has two independent subproblems, can a system discover the decomposition boundary?

In finite worlds:

```text
Yes, if independence leaves a detectable effect signature: block-diagonal
sensitivity, sparse interaction graph, or reusable subprogram structure.
```

No, or not uniquely, if:

```text
the output entangles components through a dense function;
only one scalar output is observed;
multiple decompositions have equal MDL;
the component boundary is semantic rather than behavioral;
or the needed interventions are not supplied.
```

A good demo should measure:

1. boundary recovery, e.g. adjusted Rand index against hidden component labels after freeze;
2. verifier value, e.g. hidden failures caught by component-local obligations;
3. synthesis value, e.g. search time or candidate count reduction from decomposed synthesis;
4. repair locality, e.g. counterexample in component A edits only subprogram A;
5. baseline comparison against direct synthesis, exhaustive sensitivity clustering, and NTB-0.

### What We Might Be Missing

The "true" decomposition is not always identifiable from behavior. If two components are composed by XOR, many invertible reparameterizations produce equivalent decompositions:

```text
y = f_A(A) XOR f_B(B)
```

can be rewritten by flipping one component and compensating in the other. The benchmark should not demand recovery of metaphysical components. It should demand recovery of a useful decomposition:

```text
shorter verifier, shorter program, lower search cost, or more local repair.
```

### Verdict

```text
B3_IS_OPEN_BUT_TESTABLE; IT IS THE FIRST PLACE PCCP-H COULD BECOME A SEARCH-SPACE PARADIGM.
```

Do not start with B3 implementation before B2 absorption is measured. But design B2 so it naturally extends into B3: collect sensitivity matrices, interaction tensors, and relation clauses in a form that can support component discovery.

---

## I203: Final Existential Verdict

### Steelman

The favorable reading remains alive:

```text
PCCP-H has proved the after-frame separation and now has a concrete path to
bounded frame discovery: active perturbations, metamorphic relation mining,
clause compilation, hidden transfer, and repair localization.
```

The hostile reading is also stronger after this batch:

```text
FDM-0 at B1 is just exhaustive perturbation testing. FDM-0 at B2 may be
metamorphic-relation mining. FDM-0 at B3 may be sensitivity clustering and
program decomposition. A neural-tool agent can run all of these.
```

### Threat Questions

#### (a) Is FDM-0 absorbed by exhaustive perturbation testing for B1-level problems?

Yes.

```text
B1-level single-field invariance/covariance is absorbed under a supplied valid
perturbation grammar and target-label oracle.
```

FDM-0 should still implement B1 as a baseline and pipeline check, but it should expect:

```text
DISCOVERY_ABSORBED_B1
```

if the exhaustive screen finds the same obligations.

#### (b) Does FDM-0 have a non-absorbed edge at B2 level?

Conditional.

FDM-0 has a possible edge if it does more than exhaustive relation enumeration:

1. active query selection;
2. MDL relation selection under many false positives;
3. composition closure;
4. hidden transfer under role/name randomization;
5. verifier compilation and repair localization;
6. lower cost than exhaustive metamorphic mining and NTB-0.

If it does not, B2 is absorbed by:

```text
exhaustive metamorphic relation mining over supplied T and Phi.
```

#### (c) Does the neural-tool baseline kill or complement PCCP-H?

It kills PCCP-H as a cheap formal discovery paradigm if it matches FDM-0 under equal information.

It complements PCCP-H as an artifact discipline if the neural agent proposes clauses but PCCP-H compiles them into:

```text
machine-checkable obligations, hidden-intervention tests, compact executable
programs, repair traces, and human-labor/accounting ledgers.
```

This is a demotion, not necessarily a total kill:

```text
discovery substrate = neural/tool agent
durable knowledge substrate = PCCP artifact
```

The moonshot version survives only if the compiled PCCP loop adds measurable value beyond the neural agent's ordinary test-writing and code-repair workflow.

#### (d) What is the honest next experiment?

Build the absorption suite, not just FDM-0.

Minimum next W-loop design:

```text
Experiment B1:
  V0 misses spurious invariance.
  P_bad uses S and passes V0.
  Run FDM-0 B1.
  Run Exhaustive Single-Field Effect Screening.
  Expected verdict: DISCOVERY_ABSORBED_B1 unless FDM-0 reduces query cost or transfers better.

Experiment B2:
  V0 misses covariance/composition relations.
  Required clauses include flip(C0) -> NOT(y), flip(C1) -> NOT(y),
  flip(C0,C1) -> identity(y).
  Run FDM-0 Relation Miner.
  Run Exhaustive Metamorphic Relation Miner.
  Run NTB-0 protocol.
  Report hidden transfer, false positives, clause length, oracle calls, repair localization.

Experiment B3 design-only:
  Construct two-component finite world.
  Do not claim success until B2 absorption is measured.
```

Precommit tokens:

```text
FRAME_SIGNAL:
  discovered clause catches a missed failure.

DISCOVERY_ABSORBED_B1:
  exhaustive single-field screen finds equal or better B1 clauses.

B2_DISCOVERY_SIGNAL:
  FDM-0 finds relation clauses that transfer and beat exhaustive MR mining or NTB-0
  on a meaningful axis.

NEURAL_TOOL_ABSORBED:
  NTB-0 finds equal or better clauses under equal information.

VOID:
  task too small, transform grammar answer-shaped, hidden transfer absent,
  or baselines materially handicapped.
```

#### (e) What is the gossip-magazine sentence?

```text
The laptop already learned to obey the rulebook; now it has to notice the
missing rule before both the tester script and the tool-using genius do.
```

### What We Might Be Missing

There is a third threat behind the two named threats:

```text
The human-written perturbation grammar may be the real intelligence.
```

Even if FDM-0 beats a neural-tool baseline inside a supplied transformation space, the victory may not transfer to real frame formation if humans supplied exactly the transformations that matter.

The information ledger must therefore separate:

```text
field discovery
relation discovery
transformation-grammar discovery
output-relation discovery
decomposition discovery
goal/verifier discovery
```

Only the latter three start to look paradigm-level.

### Verdict

```text
B1: ABSORBED.
B2: LIVE BUT PRIOR-ART-HEAVY.
B3: OPEN AND POTENTIALLY PARADIGM-LEVEL.
NEURAL-TOOL BASELINE: EXISTENTIAL.
NEXT MOVE: BUILD THE ABSORPTION SUITE, EXPECT B1 TO LOSE, AND TEST B2 HONESTLY.
```

PCCP-H survives this batch only by becoming more honest:

```text
Do not defend FDM-0 as novel perturbation testing. Use B1 as a sanity check,
then test whether relation and decomposition discovery survive exhaustive
metamorphic miners and neural-tool agents under equal information.
```

---

## Recommendation

**Verdict: CONCEDE B1 ABSORPTION; MOVE DISCOVERY CLAIM TO B2/B3.**

Keep:

```text
PCCP-H as executable, checkable, intervention-robust, locally repairable
artifact discipline.
```

Demote:

```text
single-field invariance discovery under a supplied perturbation grammar.
```

Build next:

```text
FDM-0 absorption suite:
  B1 exhaustive perturbation baseline
  B2 exhaustive metamorphic relation baseline
  NTB-0 neural-tool baseline protocol
  hidden transfer and smuggling ledgers
```

Do not claim:

```text
MOONSHOT_PCCP from finding invariant_to(S) in the current witness.
```

Claim only:

```text
We are testing whether bounded frame discovery survives the obvious absorption
baselines. If it does not, PCCP-H becomes the compiler/audit layer around the
winning discovery substrate.
```

---

## NARRATIVE ATTACK

### 1. Strongest "FDM-0 is just exhaustive perturbation testing with a name" dismissal

```text
FDM-0 sounds like discovery, but for the actual PCCP-0 witness it reduces to a
for-loop. Toggle every finite field, query the label, and record whether the
answer changed. Complexity is O(fields * domain_size * examples). This finds
nuisance fields, spurious fields, and causal fields in one pass. It discovers
the missing invariant_to(S) clause, rejects the shortcut program, and compiles
the obvious verifier check.

The hard part was not FDM-0. The hard part was that the human supplied a valid
perturbation API and a target oracle. Once those exist, discovery is ordinary
testing. Calling the result "frame discovery" is relabeling a perturbation
screen as a paradigm.
```

This dismissal is correct for B1.

The only defense is to stop claiming B1 is the moonshot. B1 is a calibration target and an absorption baseline.

### 2. Strongest "neural-tool agents already do this better" dismissal

```text
Give GPT-5 the same DSL, examples, partial verifier, perturbation API,
counterexample feedback, solver access, and code execution. It will write the
same perturbation loops, run Daikon-style invariant mining if useful, test
metamorphic relations, inspect failures, and submit verifier clauses. It may
also infer better preconditions and decompositions from broad prior knowledge.

If its clauses compile and pass hidden evaluation as well as FDM-0, then FDM-0
is not the discovery engine. The neural-tool agent is. PCCP-H is merely the
format that turns the agent's discoveries into checkable artifacts.
```

This dismissal remains live until NTB-0 is run.

It is not enough to say the neural agent is expensive or proprietary. The PCCP-H spec is substrate-open. If neural-tool discovery best serves the outcomes, the honest move is to use it and measure what the PCCP compiler/audit layer adds.

### 3. What would PCCP-H need to demonstrate to survive both attacks?

PCCP-H needs to demonstrate all of the following:

1. B1 absorption is reported honestly, not hidden.
2. A B2 relation is discovered that is not merely a supplied single-field invariance.
3. The relation transfers to role-randomized or transformation-held-out worlds.
4. Exhaustive metamorphic relation mining does not find an equal or better clause at comparable cost, or FDM-0 has a measured cost/repair/transfer advantage.
5. NTB-0 does not find equal or better clauses under equal information, or PCCP-H shows a separate compiled-artifact advantage that matters.
6. Human labor is accounted for: perturbation grammar, output relation grammar, clause templates, and decomposition hints are not counted as system discovery.
7. At least one B3 decomposition demo shows reduced synthesis cost or improved repair locality over direct solving.
8. The discovered obligations are machine-checkable and catch real hidden failures, not just produce plausible prose.
9. The result survives negative controls, name randomization, role permutation, and hidden-family transfer.
10. The application target is a finite domain where discovered obligations matter: software protocols, smart contracts, IAM, data pipelines, bounded safety controllers, or similar.

Unkillable version:

```text
A CPU-only system receives a partial verifier, finite traces, and generic
role-neutral intervention operations. It discovers a metamorphic relation or
decomposition boundary that catches a hidden shortcut, transfers across
role-randomized worlds, improves local repair, and beats both exhaustive
relation mining and a neural-tool agent under equal information.
```

Final narrative verdict:

```text
PCCP-H survived the proxy-compression fight. It has not survived the discovery
fight. The next honest result is allowed to be negative: if perturbation testing
or GPT-5 does the discovery, PCCP-H should become the public verifier/compiler
for that winner. The home run requires a cheap system that discovers relations
or decompositions the obvious tester and the tool-using model both miss.
```
