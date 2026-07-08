# Q-Loop Batch 46: Final Hard-Domain Attack And Methodology Paper Prep

**Date:** 2026-07-08  
**Role:** Question-Loop worker  
**Iterations:** I393-I406  
**Status:** adversarial attack on the final >=64-rule WGD escape route, with conditional methodology-paper outline if the final hard-domain attempt is absorbed.

Two invariants held fixed:

1. Swing for the home run.
2. The loop only stops on a won-over adversary.

## Grounding

Read the required context in the current checkout:

1. `research/dual_loop_supervisor_checkin_35.md`
2. `research/wgd_0_precommit_spec.md`
3. `research/work_loop_batch37.md`
4. `research/frameseed_milestone_report.md`
5. `research/METHODOLOGY_TEMPLATE.md`
6. `research/VISION.md`

Additional checkout evidence used for continuity:

- `research/work_loop_batch36.md`
- `research/question_loop_batch42.md`
- `research/question_loop_batch43.md`
- `research/question_loop_batch44.md`
- `research/question_loop_batch45.md`
- `experiments/wgd0_b37_hidden_measurement.json`

Live checkout boundary:

```text
research/work_loop_batch38.md is not present in this checkout.
No B38 hidden-result artifact is present.
B37 is the last measured WGD artifact present here.
```

Therefore this batch cannot honestly adjudicate an actual B38 hidden terminal
token. It does the assigned Q-Loop job in the only defensible form available:
attack the proposed final >=64-rule domain before or during construction, find
the absorption routes that remain alive even when flat enumeration is made
expensive, and prepare the methodology-paper outline that should be used if the
final hard-domain run is absorbed.

## Executive Verdict

The final hard-domain attempt cannot win by saying:

```text
The grammar has >=64 compositional rules, so brute-force enumeration is too
large.
```

That only kills the weakest straw baseline: flat enumeration over all complete
rule subsets. A hostile adversary will not enumerate `2^64` worlds. The
adversary will factor the grammar, exploit locality, learn the dependency graph,
use active membership/equivalence queries, synthesize constraints, infer
bindings from richer signatures, compress repeated rule schemas into a library,
or charge the hand-authored generator/substrate that made the rule family
learnable.

The key adversarial statement is:

```text
Compositionality is not automatically a defense against absorption.
Compositionality is exactly what the strongest absorbers exploit.
```

The final hard-domain attempt wins only if all of these are false under equal
information and all-in cost:

- schema/binding discovery identifies the latent rule roles;
- PBE/CEGIS synthesizes the transition, guard, repair, and abstention programs;
- active querying identifies the live rule set cheaply;
- constraint/SAT/SMT/ILP/ASP-style solving learns the admissible language;
- MDL/library learning compresses the 64 rules into reusable macros;
- generator-family classification predicts the hidden branch or rule templates;
- operation ontology, typed primitives, verifier labels, or substrate design
  already carve the answer space;
- repair/abstention/composition performance is explained by native domain tools;
- the claimed grammar is post-hoc compression rather than a causal ingredient.

If the final hard-domain run is absorbed, the correct deliverable is not a
weaker WGD narrative. It is the methodology paper:

```text
The Absorption Ladder: How to Honestly Test AI Discovery Claims
```

## Absorption Routes At >=64 Rules

| Hard-domain feature | Why it looks strong | Absorption route that remains alive |
|---|---|---|
| 64+ rules | Flat enumeration explodes. | Factorized learning, group testing, SAT/SMT, CEGIS, active queries. |
| Compositional rules | Transfer seems grammar-like. | MDL/library learning and program synthesis are native compositional tools. |
| Opaque names | Surface semantics removed. | Distributional signatures, type/arity patterns, dependency fingerprints. |
| Hidden dependencies | Requires structure. | Causal/invariant discovery and constraint mining recover local neighborhoods. |
| Repair and abstention | Seems beyond program synthesis. | Nearest-valid search, constraint repair, active retry, anomaly abstention. |
| Held-out composition | Tests reuse. | Library composition, e-graphs, synthesized pipelines, generator-template reuse. |
| Large search space | Blocks toy brute force. | Domain tools exploit the generator's structure rather than enumerating worlds. |

## I393: Attack The Missing B38 Premise

### Attack

The supervisor says W-Loop B38 should be the final hard-domain attempt. This
checkout does not contain a B38 work-loop report, hidden measurement JSON, or
terminal token. The first adversarial move is to refuse result inflation.

If this batch wrote "B38 was absorbed" without an artifact, it would violate the
same methodology it is supposed to defend.

### What This Adds

This is not a deflection. It preserves hidden-open discipline. The B46 product
must be a pre-result attack and conditional paper prep, not a fabricated
post-result report.

### Verdict

```text
NO B38 ARTIFACT, NO B38 RESULT CLAIM.
```

## I394: Attack Rule Count As Enumeration Theater

### Attack

The phrase ">=64 compositional rules" can become theater. It proves only that a
flat search over all complete rule subsets is expensive. But no competent
absorber is obligated to use flat search.

If each rule has observable local effects, a baseline learns rules one at a
time. If rules share templates, a library learner compresses them. If rules are
constraints, a solver infers clauses. If rules can be queried, active learning
identifies the live boundary.

### Harder Absorption

A 64-rule domain may be easier than a 16-rule domain if the larger rule set
creates more public signatures. Richer behavior can give schema matchers,
causal learners, and constraint miners more handles.

### Required Defense

B38 must report the absorber it actually defeats:

```text
not flat brute force
not naive enumeration
not intentionally weak PBE
but factorized, symmetry-aware, query-aware, constraint-aware native absorbers
```

### Verdict

```text
LARGE RULE COUNT KILLS ONLY THE BASELINE NOBODY SHOULD USE.
```

## I395: Attack Factorization And Locality

### Attack

Compositional rules usually imply a factor graph. If rule `r_i` affects only a
small neighborhood of fields, objects, states, or event types, the absorber does
not learn a 64-rule monolith. It learns local Markov blankets and composes them.

That route is native to:

- causal graph discovery;
- invariant mining;
- sparse dependency testing;
- relational feature selection;
- local constraint learning;
- graphical model structure learning.

### Harder Absorption

If B38 blocks local factorization by making every rule interact globally with
every other rule, it may create a domain that is not learnable by WGD either. A
domain that only defeats absorbers by destroying identifiability is void, not
signal.

### Required Defense

B38 needs a factorized absorber:

```text
estimate dependency neighborhoods
learn local transition/guard/repair clauses
compose local clauses
score held-out global behavior
charge graph-learning and clause bits
```

If that absorber reaches the gates under <4x all-in, the terminal token should
be constraint, causal/invariant, CEGIS, or library absorption.

### Verdict

```text
COMPOSITIONAL STRUCTURE IS AN ABSORBER ENTRY POINT, NOT A SHIELD.
```

## I396: Attack Through Constraint Solvers

### Attack

A 64-rule typed world is likely a constraint system in disguise:

```text
preconditions
postconditions
invariants
unsafe predicates
repair clauses
composition constraints
abstention conditions
```

That is the native territory of SAT, SMT, ILP, ASP, CSP, table constraints,
version spaces, and typed program synthesis. The baseline should not enumerate
rules. It should encode the public transcript as constraints and solve for the
smallest consistent executable theory.

### Harder Absorption

If WGD emits an executable grammar but a solver emits an equivalent constraint
theory, interpretability does not rescue WGD. The solver has recovered the
behavioral grammar under a known substrate.

### Required Defense

B38 must include a constraint-solving absorber with:

```text
same public observations
same feedback bits
same typed primitives
same repair and abstention output contract
same composition queries
same timeout and cost ledger
```

Missing this absorber makes the maximum token inconclusive, not signal.

### Verdict

```text
RULE DISCOVERY OVER TYPED WORLDS IS PRESUMPTIVELY CONSTRAINT LEARNING.
```

## I397: Attack Through Active Queries

### Attack

At >=64 rules, passive observation may be hard. But WGD's act, repair, and
abstain interface creates queries. Each proposed action, repair attempt, or
abstention judgement can reveal bits about the hidden grammar.

A competent active learner will not sample randomly. It will use group tests:
queries whose outcomes split many candidate rule states at once.

### Harder Absorption

The query complexity of a 64-rule family may be closer to `O(64 log k)` or
lower if rules have shared templates, not `2^64`. If feedback labels include
`ACCEPTED`, `REJECTED`, `UNSAFE`, `AMBIGUOUS`, and `WRONG`, each answer can
carry more than one bit.

### Required Defense

B38 must separate:

```text
passive final-domain condition
active final-domain condition
repair-without-feedback condition
repair-with-feedback-charged condition
```

Then it must run active CEGIS and active learning with the same answer channel.
If active learning wins, the honest result may still be valuable, but the token
is active-learning absorption.

### Verdict

```text
A LARGE RULE SET CAN STILL BE CHEAPLY IDENTIFIED BY GOOD QUESTIONS.
```

## I398: Attack Through Grammar Induction And Program Synthesis

### Attack

The final domain is called grammar discovery. That gives grammar induction and
program synthesis first refusal. If the hidden rules are generated from a
small meta-grammar, a synthesizer can search the meta-grammar rather than the
expanded rule set.

Native absorbers include:

- enumerative typed DSL synthesis with pruning;
- CEGIS over predicates and transitions;
- e-graph equality saturation for composed rewrites;
- automata or transducer induction;
- ILP-style relational rule learning;
- anti-unification over successful traces.

### Harder Absorption

If B38 hides the meta-grammar from baselines but WGD's substrate embodies it in
the grammar IR, parser, canonicalizer, or public primitive inventory, the result
is substrate asymmetry.

### Required Defense

The strongest PBE/CEGIS baseline must receive the same hypothesis vocabulary or
WGD must charge its vocabulary as `H` or `O`. If WGD's grammar IR is a better
search space, the win may belong to human-authored representation design.

### Verdict

```text
THE BASELINE SHOULD SEARCH THE GENERATOR'S COMPRESSED LANGUAGE, NOT ITS EXPANDED OUTCOMES.
```

## I399: Attack Through MDL And Library Learning

### Attack

If 64 rules share motifs, MDL wins. If they do not share motifs, WGD transfer is
unlikely to win. This is the central fork:

```text
shared structure -> library learning absorption
no shared structure -> no reusable grammar claim
```

A grammar that transfers across siblings is evidence of reusable structure, but
MDL library learning is the ordinary explanation for reusable structure.

### Harder Absorption

Composition strengthens MDL. Repeated subclauses, guards, repair operators,
abstention triggers, and dependency patterns are exactly what a library learner
compresses.

### Required Defense

B38 needs an MDL/sibling-library learner that can emit:

```text
macro library
per-sibling bindings
per-task residual programs
repair macros
abstention macros
composition macros
all-in cost
```

If `L + B_i + P_i + R_i + A_i` matches WGD under <4x, WGD is absorbed by
library learning.

### Verdict

```text
TRANSFER IS PRESUMPTIVELY LIBRARY LEARNING UNTIL THE LIBRARY LEARNER LOSES.
```

## I400: Attack Schema And Binding At Scale

### Attack

B37 died because role binding and feedback-program induction solved the hidden
cases cheaply. B38 may try to fix this by making more rules and more fields.
That can backfire.

More rules can create richer fingerprints:

- a field participating in many safety failures;
- a relation always touched before repairs;
- a status value correlated with ambiguity;
- an object type appearing in every invalidity path;
- a dependency node whose edit changes many downstream outcomes.

These are binding signatures.

### Harder Absorption

Opaque names do not remove signatures. If the public transcript has enough
interventions, each latent role can become more identifiable, not less. A
schema/binding baseline can use bipartite graph matching, spectral signatures,
constraint participation vectors, or intervention-response fingerprints.

### Required Defense

B38 must include binding attacks that improve with scale:

```text
role participation fingerprints
intervention-response vectors
dependency graph matching
schema isomorphism holdouts
value-distribution decoys
key/cardinality decoys
unit/scale decoys
binding-only ablation
schema-plus-PBE pipeline
```

If binding-only or binding-plus-PBE passes, B38 is absorbed even if flat
enumeration fails.

### Verdict

```text
MORE RULES CAN MAKE BINDINGS EASIER, NOT HARDER.
```

## I401: Attack Generator-Family Identification

### Attack

Synthetic hard domains usually come from a generator. If the generator creates
64 rules by templates, public statistics may identify the template. If the
template is identified, the solver only needs residual bindings and parameters.

This is not hypothetical. B42 already made latent family ID separate from
behavioral grammar. B38 must prove that distinction at larger scale.

### Harder Absorption

If public feature classifiers predict:

```text
rule family
composition form
repair location
obligation class
abstention bucket
hidden query bucket
sibling template
```

then WGD has not discovered geometry. It has decoded generator fingerprints.

### Required Defense

B38 needs adversarial public-feature classifiers that are stronger at scale,
not weaker:

```text
random forest / boosted trees
compression classifier
program-feature search
graph-kernel classifier
serializer-offset probe
feedback-sequence predictor
split-reconstruction attack
```

If these classifiers predict hidden structure above thresholds and help solve,
the token is generator leakage or generator-family absorption.

### Verdict

```text
A LARGE SYNTHETIC GENERATOR CAN LEAK A LARGER FINGERPRINT.
```

## I402: Attack Operation Ontology And Domain Tools

### Attack

A hard domain needs an action interface. That interface can supply the ontology:
what can be edited, what counts as a delta, how repairs are represented, what a
composition is, what an unsafe result looks like, and what abstention reasons
exist.

At >=64 rules, the public DSL can become the real discovery mechanism. The
learner only searches inside a human-authored operation space.

### Harder Absorption

If B38 uses a recognizable domain such as tables, workflows, programs, graphs,
plans, types, theorem states, spreadsheets, databases, or finite automata, then
domain tools get first refusal. A domain-native solver may beat WGD without
looking like the original B37 schema baseline.

### Required Defense

B38 must report both regimes:

```text
substrate_free
substrate_charged
```

If the signal survives only because the typed parser, operation DSL, verifier,
canonicalizer, or domain toolkit is free, the claim ceiling must say that the
substrate was supplied, not learned.

### Verdict

```text
THE HAND-AUTHORED ACTION SPACE MAY BE THE GRAMMAR.
```

## I403: Attack Repair, Abstention, And Composition As Separate Claims

### Attack

B38 may pass action HFA while still failing the real WGD claim. WGD is not only
action. It claims discovered transformations, invalidity, unsafe conditions,
obligations, local repairs, abstention, siblings, and held-out composition.

Each surface has its own absorber:

- repair: nearest-valid search, constraint repair, CEGIS repair, active retry;
- abstention: anomaly detection, calibrated uncertainty, always-abstain under
  bad utility;
- composition: synthesized pipelines, e-graphs, library composition, saved
  macros.

### Harder Absorption

Repair and abstention can conceal failures. A system can abstain on hard
compositions and repair by querying until accepted. That looks safe and
improvable while dodging the intended burden.

### Required Defense

B38 must report per-surface terminal evidence:

```text
action HFA
invalidity F1
unsafe F1
obligation F1 only where behaviorally identifiable
repair without feedback
repair with feedback charged
repair locality and preservation
abstention risk/coverage and utility
composition HFA with interference and noncommutation
component erasure drops
```

A single aggregate HFA cannot carry WGD signal.

### Verdict

```text
ACTION SUCCESS IS NOT WORLD-GEOMETRY DISCOVERY.
```

## I404: Attack All-In Cost At 64 Rules

### Attack

WGD's own artifact may become expensive at 64 rules. A grammar object, node
provenance, interpreter, repair library, abstention policy, composition rules,
bindings, and substrate ledger can cost more than a boring solver's compact
decision diagram or constraint theory.

The final domain can therefore produce a strange result:

```text
WGD reaches high HFA, but the boring absorber is cheaper because it represents
the same behavior more compactly.
```

That is absorption.

### Harder Absorption

The all-in ledger must include:

- human design of the hard generator;
- parser and DSL design;
- feature extractors and canonicalizers;
- absorber adapters;
- WGD grammar interpreter;
- public smoke iteration;
- query and repair feedback;
- residual bindings and per-task programs;
- library bits and solver code.

If those are assigned after seeing the result, the ledger is not evidence.

### Required Defense

B38 must emit machine-recomputable costs for every system:

```text
G, B_i, P_i, E_i, C_i, Q_i, V_i, R_i, A_i, L, H, O, N
runtime
query count
adapter bits
substrate_free_total
substrate_charged_total
cost to threshold
ratio against WGD
```

If a solver reaches the gates at <=4x WGD cost, the token is the solver's
absorber, not signal.

### Verdict

```text
AT 64 RULES, WGD MAY BE THE BLOATED EXPLANATION.
```

## I405: Conditional Methodology Paper Outline

### Trigger

Use this outline if the final hard-domain attempt is absorbed, void, or
inconclusive because a required absorber remains unimplemented. Do not cite B38
as a result until `research/work_loop_batch38.md` or an equivalent hidden
measurement artifact exists.

### Working Title

```text
The Absorption Ladder: How to Honestly Test AI Discovery Claims
```

### Thesis

Most AI discovery claims fail because the apparent discovery is absorbed by a
boring explanation: representation priors, teaching sets, active queries,
program synthesis, schema binding, library learning, domain tools, nuisance
oracles, or human-authored substrate. A valid methodology should make those
absorbers executable before celebrating any positive result.

### Abstract Skeleton

```text
We introduce the absorption ladder, a methodology for evaluating claims that an
AI system has discovered reusable structure rather than received, searched,
synthesized, compressed, or bound it through ordinary mechanisms. The method
precommits terminal tokens, gives strongest boring explanations equal
information, opens hidden seeds only after a manifest freeze, requires all-in
cost ledgers, and treats negative or absorbed outcomes as first-class results.
Across a sequence of toy discovery programs, the ladder repeatedly prevented
false positive narratives: perfect hidden accuracy was insufficient when
teaching dimension, schema binding, PBE/CEGIS, or library-style explanations
matched under lower cost. The contribution is not a new discovery mechanism but
a reproducible discipline for deciding when a discovery claim has survived its
best alternatives.
```

### Paper Structure

1. **Introduction: The Discovery Claim Problem**  
   Explain why high benchmark accuracy, compact artifacts, and interpretability
   are not enough. State the enemy: confusing supplied structure with discovered
   structure.

2. **Definitions: Signal, Absorption, Void, Trap, Negative**  
   Define the terminal-token vocabulary. Emphasize that mixed evidence cannot be
   narrated upward.

3. **The Absorption Ladder**  
   Present the ladder: representation prior, parser/substrate prior, teaching
   dimension, active learning, PBE/CEGIS, library learning, schema/binding,
   domain tools, nuisance oracles, human substrate, post-hoc compression.

4. **Equal-Information Baseline Contract**  
   Same bytes are not enough. Define equal affordance, adapter parity,
   canonicalizer parity, query parity, and hidden-open parity.

5. **All-In Accounting**  
   Formalize `F/G`, `B_i`, `P_i`, `E_i`, `C_i`, `Q_i`, `V_i`, `R_i`, `A_i`,
   `L`, `H`, `O`, and `N`. Show how uncharged substrate creates fake cheapness.

6. **Hidden-Open Discipline**  
   Manifest freeze, public smoke seeds, hidden seed once, post-hidden mutation
   voiding, and report skeletons written before results.

7. **Causal Artifact Tests**  
   Packet/grammar erasure, component erasure, randomized labels, no-language
   controls, role/schema/unit/order permutations, leakage classifiers, and
   identifiability alternatives.

8. **AFTD And Transfer Without Clone Accounting**  
   Explain all-in amortized frame/grammar teaching distance and why siblings
   must be nonduplicate and reduced after residual bindings, programs, queries,
   and libraries are charged.

9. **Case Studies From The Project Record**  
   - PCCP-H: useful after-frame/audit discipline, discovery absorbed by
     exhaustive baselines.
   - FrameSeed Boolean: perfect HFA absorbed by teaching dimension.
   - FrameSeed SHEETS-0: typed packet absorbed by schema binding and typed
     pipelines.
   - WGD B37: grammar candidate reached 1.0 HFA, but schema/binding and
     PBE/CEGIS reached 1.0 HFA at roughly 8.8% and 10.7% of WGD cost.
   - B38 final hard domain, if measured: report the exact terminal token and
     absorber. If no artifact exists, omit as result and keep it as future work.

10. **Why Negative Results Are The Result**  
    Argue that the repeated kills are not failure to publish. They are evidence
    that the methodology prevents self-deception and makes future positive
    claims harder to fake.

11. **What The Method Does Not Prove**  
    It does not prove discovery is impossible. It does not prove scale is
    useless. It does not prove the tested mechanisms are worthless. It only
    proves which claims did not survive their strongest boring alternatives.

12. **Practical Checklist For New Claims**  
    Include a one-page checklist adapted from `research/METHODOLOGY_TEMPLATE.md`.

13. **Conclusion: Methodology As Moonshot**  
    The contribution is an immune system for discovery claims: before asking the
    world to believe a new mechanism, make every ordinary explanation dangerous.

### Required Figures And Tables

- Figure 1: Terminal-token precedence flow.
- Figure 2: The absorption ladder as a first-refusal stack.
- Figure 3: Hidden-open lifecycle from direction charter to report.
- Table 1: Cost categories and examples.
- Table 2: Equal-information failure modes.
- Table 3: Case-study outcomes and terminal tokens.
- Table 4: Claim ceilings: what can and cannot be said after signal,
  absorption, void, trap, negative, or inconclusive.

### Claim Ceiling For The Paper

Allowed:

```text
The absorption ladder is a methodology for preventing premature AI discovery
claims by forcing executable boring explanations to compete under equal
information, hidden-open discipline, and all-in accounting.
```

Banned:

```text
We proved AI discovery is impossible.
We proved our preferred mechanism works.
We proved toy negative results settle real-world intelligence.
We proved all future claims must use this exact harness.
```

### Paper Token

```text
METHODOLOGY_PAPER_OUTLINE_READY_IF_FINAL_HARD_DOMAIN_ABSORBED
```

## I406: Final Adversarial Synthesis

### Attack

The final hard-domain route remains scientifically alive only under a much
harder condition than "64 rules makes enumeration fail." It must defeat the
absorbers that thrive on rule structure:

```text
factorized dependency learning
constraint solving
active learning
PBE/CEGIS
grammar induction
MDL/library learning
schema/binding at scale
generator-family classification
operation-ontology/domain-tool absorption
repair/abstention/composition baselines
all-in cost compression
```

The most likely B38 absorption is not naive enumeration. The most likely B38
absorption is one of:

```text
WGD_ABSORBED_BY_SCHEMA_OR_BINDING_DISCOVERY
WGD_ABSORBED_BY_PBE_OR_CEGIS
WGD_ABSORBED_BY_ACTIVE_LEARNING
WGD_ABSORBED_BY_CONSTRAINT_LEARNING
WGD_ABSORBED_BY_MDL_LIBRARY_LEARNING
WGD_ABSORBED_BY_OPERATION_ONTOLOGY_SUPPLY
WGD_ABSORBED_BY_GENERATOR_FAMILY_IDENTIFICATION
WGD_ABSORBED_BY_HAND_AUTHORED_SUBSTRATE
```

### Minimum Conditions To Win Over The Adversary

B38 can win only if the final report shows:

1. The hidden family has >=64 behaviorally identifiable rules without public
   generator fingerprints or ontology supply.
2. The public substrate does not make the operation grammar cheap by design, or
   that substrate is charged and the claim ceiling is limited.
3. Native factorized, constraint, PBE/CEGIS, active, schema/binding, MDL/library,
   domain-tool, repair, abstention, composition, leakage, and post-hoc
   compression absorbers all run with equal information.
4. Those absorbers fail or pay >=4x all-in cost under both functional and
   transfer gates.
5. The WGD grammar is frozen before held-out scoring and component erasure drops
   the intended metrics.
6. Repair works without uncharged interactive search.
7. Abstention preserves utility and does not hide hard composition cases.
8. Siblings are nonduplicate and reduced after residual bindings, programs,
   queries, and library bits are charged.
9. The all-in ledger is recomputable and includes human substrate.
10. The terminal token is assigned by precommitted precedence, not narrative.

### Final Token

```text
Q_LOOP_B46_FINAL_DOMAIN_ATTACK_COMPLETE_METHODOLOGY_OUTLINE_READY_CONDITIONAL_ON_B38_ABSORPTION
```

## Final Position

The final hard-domain attempt should proceed only if it is willing to be killed
by the strongest non-flat absorbers. A 64-rule compositional world is not a
home run by itself. It is a more serious arena for the same question:

```text
Did a cheap system discover executable world geometry, or did ordinary methods
exploit the structure we supplied?
```

If ordinary methods still win, the project should not dilute the result. The
paper is the honest moonshot deliverable: a reusable absorption methodology that
turns "we got a good hidden score" into "we know what explanation survived."