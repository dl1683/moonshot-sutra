# Q-Loop Batch 42: Attack WGD Pre-Design

**Date:** 2026-07-07  
**Role:** Question-Loop worker  
**Iterations:** I337-I350  
**Status:** adversarial pre-design attack before `research/wgd_0_precommit_spec.md`.

Two invariants held fixed:

1. Swing for the home run.
2. The loop only stops on a won-over adversary.

## Grounding

Read the requested context in the current checkout:

1. `research/dual_loop_supervisor_checkin_33.md`
2. `research/question_loop_batch41.md`
3. `research/METHODOLOGY_TEMPLATE.md`
4. `research/frameseed_milestone_report.md`
5. `research/VISION.md`

Binding starting point:

```text
WGD is approved as a direction.
Approval is not evidence.
FrameSeed died because supplied compact packets transmitted supplied geometry.
WGD dies the same way if discovery is hidden in the public substrate.
```

The home-run version is still worth attacking:

```text
Can a cheap system discover the transformation and obligation geometry of a new
world well enough to act, repair, abstain, and transfer, under hostile native
absorbers and clean all-in accounting?
```

The adversarial presumption for B42:

```text
WGD IS ABSORBED UNTIL PROVEN OTHERWISE.
```

## Executive Verdict

WGD is not yet safe to spec. It is the right next question only if the precommit
spec refuses every route by which "discovery" can be silently supplied.

The likely absorption routes are stronger than B41 stated:

| Route | How WGD gets absorbed | Required pre-design defense |
|---|---|---|
| PBE | Before/after traces define a transformation program. | Native PBE over the exact public transcript, not a roster name. |
| CEGIS | Black-box failure feedback becomes counterexample-guided spec learning. | Charge counterexamples as information and run native CEGIS with the same oracle. |
| Library learning | Reusable grammar is just an MDL macro library. | Native library learner across siblings, same grammar hypothesis class, all-in cost. |
| Active learning | Query access isolates the hidden grammar cheaply. | Active-query baseline with identical budget and answer channel. |
| Schema matching | Names, types, units, distributions, keys, and constraints reveal roles. | Symbol-shuffled, role-permuted, name/unit-erased, schema-matching baselines. |
| Verifier templates | Obligations are exposed by labels, failure reasons, or scorer semantics. | No explanatory feedback unless charged and given identically to absorbers. |
| Operation ontology | Public action DSL already carves the world into valid transforms. | Minimal proposal interface plus ontology-erasure controls. |
| Generator leakage | Public statistics predict hidden family IDs or latent obligations. | Predictive leakage audits against grammar, binding, and obligation labels. |
| Human substrate | Parser, typed DSL, harness constructor, or natural language names do the work. | Human/substrate ledger with substrate-charged result and symbolic no-language condition. |

Minimum pre-design demand:

```text
WGD_SIGNAL cannot be assigned unless native PBE, native CEGIS, native MDL
library learning, native active learning, native schema/binding discovery,
predictive leakage classifiers, operation/verifier oracle controls, and
geometry-erasure ablations all fail or pay >=4x all-in.
```

If any of those are only proxy or capability-mode scored, the token is not
signal. The token is `WGD_INCONCLUSIVE_BASELINES_NOT_NATIVE`.

## I337: Attack The Word "Grammar"

### Attack

"World grammar" sounds like a discovered object, but the benchmark generator may
already define the object. If the hidden family is hand-authored as a small
grammar over latent operations, WGD can become generator reverse engineering.
The learner is not discovering world geometry; it is inferring which constructor
branch produced the episode.

This is not a weak attack. It hits the first design move. The hidden grammar
family itself can be a frame packet, moved from the learner input into the
harness.

### Absorption Route

```text
WGD_ABSORBED_BY_GENERATOR_FAMILY_IDENTIFICATION
```

If a classifier can predict hidden family ID, obligation class, or transform
family from public transcript statistics, then WGD has not discovered geometry.
It has decoded generator fingerprints.

### Pre-Design Demand

The spec must separate:

- latent generator family ID;
- behavioral equivalence class;
- discoverable executable grammar;
- nuisance surface statistics.

Signal requires the discovered grammar to work on held-out worlds whose nuisance
statistics do not reveal the same family branch. If two behaviorally equivalent
hidden grammars produce the same public evidence, the spec must forbid claims
about which grammar was discovered.

### Verdict

```text
DO NOT LET "HIDDEN GRAMMAR FAMILY" BECOME THE NEW PACKET.
```

## I338: Attack The Public Substrate

### Attack

B41 says WGD needs a bounded public substrate. True, but every bounded substrate
is a smuggling surface. Typed equality, ordering, containment, records, events,
state transitions, and action serialization can imply joins, filters, unit
normalization, dedupe, constraint checking, temporal ordering, repair, and
unsafe-action detection.

The FrameSeed SHEETS failure was not "typed domains are impossible." It was
that public typed machinery plus exact bindings made the packet unnecessary.
WGD can repeat this with a thinner disguise: no operation names, but a substrate
whose primitives make the operation ontology obvious.

### Absorption Route

```text
WGD_ABSORBED_BY_OPERATION_ONTOLOGY_SUPPLY
WGD_ABSORBED_BY_HAND_AUTHORED_SUBSTRATE
```

The token fires if the public proposal interface, serializer, type system,
primitive predicates, or parser exposes the transformation ontology at a level
where ordinary search only selects among supplied moves.

### Pre-Design Demand

The spec must run two accounting regimes:

```text
substrate_free: all systems receive the public substrate at zero cost.
substrate_charged: parser, type predicates, proposal DSL, validators, adapters,
and any human-authored representational convenience are charged as H.
```

WGD may only claim discovered geometry in the substrate-free regime. If the
claim disappears in substrate-charged accounting, the report must say:

```text
No claim is made that the hand-authored substrate was learned.
```

### Verdict

```text
THE SUBSTRATE IS NOT NEUTRAL. IT IS THE FIRST SUSPECT.
```

## I339: Attack Through PBE

### Attack

If training evidence includes before/after examples, successful action traces,
or input/output records, then PBE owns the first refusal. The WGD learner can
infer transformations by synthesizing programs consistent with examples, then
rename the resulting program set "grammar."

This is especially dangerous in typed worlds. Once records, fields, and values
are public, a PBE engine can enumerate projections, filters, joins, edits,
normalizations, aggregations, constraints, and guards. The grammar is then a
compressed program library over typed examples.

### Harder Attack

Obligations do not automatically escape PBE. A PBE system can synthesize:

- a transformer;
- a precondition;
- a postcondition;
- a rejection predicate;
- a repair program;
- a witness query.

If WGD's obligation object is behaviorally equivalent to synthesized
pre/postconditions, PBE has absorbed it.

### Absorption Route

```text
WGD_ABSORBED_BY_PBE
```

or, if the same engine is run as counterexample-guided synthesis:

```text
WGD_ABSORBED_BY_PBE_CEGIS
```

### Pre-Design Demand

The precommit spec must implement a native PBE baseline before hidden open.
Roster names are not enough. The baseline must receive the exact same public
examples, typed primitives, action traces, feedback labels, and cost budget as
the WGD system.

Required PBE outputs:

```text
transform_program
validity_guard
invalidity_guard
unsafe_guard
repair_program
abstention_rule
composition_program
```

If PBE reaches the functional thresholds at matched or lower-than-4x all-in
cost, WGD is absorbed even if the WGD artifact looks more interpretable.

### Verdict

```text
PBE IS NOT A BASELINE. IT IS WGD'S DEFAULT BORING EXPLANATION.
```

## I340: Attack Through CEGIS

### Attack

WGD wants black-box outcome feedback. That feedback is exactly the fuel of
CEGIS. The learner proposes a grammar/action; the world returns failure; the
failure becomes a counterexample; the hypothesis is refined.

If the benchmark offers counterexamples, failed traces, invalid-action labels,
or repair hints, a CEGIS loop can isolate the admissible transformation class.
Then "discovery" is just spec learning by counterexample.

### Harder Attack

Even binary success/failure can be enough. In a finite typed world, membership
queries over candidate actions can identify a validity language. If the action
proposal interface is small, CEGIS can enumerate it. If the interface is rich
but structured, CEGIS can synthesize within that structure.

Therefore the dangerous object is not only explanatory feedback. The dangerous
object is any feedback channel whose query complexity is small relative to the
claimed discovery.

### Absorption Route

```text
WGD_ABSORBED_BY_CEGIS
WGD_ABSORBED_BY_ACTIVE_CEGIS
```

### Pre-Design Demand

The spec must define:

```text
C_i = counterexample and feedback bits
Q_i = active query/action proposal count
feedback_channel = exact content returned on success, failure, invalidity,
unsafe action, abstention, and repair attempt
```

Native CEGIS must run with the same channel. If WGD receives failure traces,
CEGIS receives failure traces. If WGD receives only scalar reward, CEGIS receives
only scalar reward. If the scalar reward still isolates the grammar cheaply,
that is active-learning absorption, not signal.

### Verdict

```text
BLACK-BOX FEEDBACK IS NOT FREE MYSTERY. IT IS A SPECIFICATION ORACLE.
```

## I341: Attack Through MDL Library Learning

### Attack

WGD says it discovers reusable transformation and obligation geometry. MDL
library learning says it discovers reusable macros that compress solutions
across tasks. Those may be the same object under different branding.

The more WGD succeeds at transfer, the more it risks becoming library learning:
extract common subprograms, compose them on new tasks, amortize cost over
siblings. That is not a defect in library learning. It is the strongest boring
explanation for any reusable grammar result.

### Harder Attack

Adding invalidity, unsafe conditions, repair, and abstention may still not
escape. A library learner can compress a richer library:

```text
macro_transform
macro_guard
macro_repair
macro_counterexample_request
macro_abstain
macro_compose
```

If the same MDL objective learns these macros and transfers at threshold, WGD is
absorbed by learned library formation.

### Absorption Route

```text
WGD_ABSORBED_BY_MDL_LIBRARY_LEARNING
WGD_ABSORBED_BY_SIBLING_LIBRARY_LEARNING
```

### Pre-Design Demand

The spec must implement a native MDL library learner or a formal lower bound
against one. It must not compare WGD against a no-library synthesizer and then
claim reusable structure.

The library baseline must receive:

- all public transcripts;
- the same typed substrate;
- the same sibling set;
- the same held-out split policy;
- the same cost categories;
- the same output requirements, including guards and repairs.

Cost accounting must separate:

```text
L = learned library bits
F = claimed WGD grammar bits
P_i = task-specific programs
B_i = bindings
R_i = residual teaching/search bits
```

If `L + sum(P_i + B_i + R_i)` matches WGD within 4x all-in, no signal.

### Verdict

```text
REUSABLE GRAMMAR IS PRESUMPTIVELY A LEARNED LIBRARY.
```

## I342: Attack Through Active Learning

### Attack

The approved WGD story says the system can act, repair, abstain, and transfer.
Acting in a hidden world creates queries. Repair attempts create queries.
Abstention can create implicit queries if the system is told whether abstention
was right. Every interactive affordance is an active-learning channel.

If the system can choose informative candidate actions and receive validity,
safety, or reward answers, it may identify the grammar with far fewer examples
than passive learners. That would be an active-learning result, not WGD.

### Harder Attack

The phrase "cheap system discovers" can hide a query-complexity theorem:

```text
Given membership/equivalence queries, this concept class is efficiently
learnable.
```

That is valuable, but it belongs to exact/active learning unless WGD shows a
distinct structure or defeats the native active learner.

### Absorption Route

```text
WGD_ABSORBED_BY_ACTIVE_LEARNING
```

### Pre-Design Demand

The spec needs two modes:

```text
passive_discovery: no chosen queries, only frozen public transcripts.
active_discovery: chosen action/query attempts are allowed and charged.
```

Active WGD cannot claim a passive discovery result. In active mode, a native
active learner with the same hypothesis class, query budget, and answer channel
must be run. If it isolates the grammar, assign the active-learning token.

The report must include a query ledger:

```text
query_count
query_type
answer_bits
whether_query_was_adaptive
whether_query_touched_hidden_evaluation_distribution
```

### Verdict

```text
EVERY REPAIR ATTEMPT IS POTENTIALLY A QUERY. CHARGE IT.
```

## I343: Attack Through Schema Matching And Binding Discovery

### Attack

Typed practical worlds leak structure through names, units, distributions,
cardinalities, null patterns, key uniqueness, temporal monotonicity, value
ranges, row order, foreign-key-like co-occurrence, and error messages.

Schema matching, entity resolution, type inference, constraint discovery, and
record linkage can convert those leaks into bindings. Once bindings are known,
the remaining transformations may be ordinary typed pipelines. This is exactly
where SHEETS-0 died.

### Harder Attack

Removing names is not enough. A column with unique stable IDs, a near-monotone
timestamp, a low-cardinality status field, or a unit-scaled quantity can reveal
its role without a name. Distributional fingerprints are bindings.

### Absorption Route

```text
WGD_ABSORBED_BY_SCHEMA_OR_BINDING_DISCOVERY
WGD_ABSORBED_BY_ENTITY_RESOLUTION
WGD_ABSORBED_BY_CONSTRAINT_DISCOVERY
```

### Pre-Design Demand

The spec must precommit schema/binding baselines:

- name/description matcher;
- type/shape matcher;
- distributional matcher;
- key/entity resolver;
- unit/scale detector;
- constraint miner;
- combined schema+PBE pipeline.

It must also include erasures:

```text
role_name_permutation
unit_name_erasure
row_order_permutation
field_order_permutation
value_distribution_decoy
key_cardinality_decoy
schema_isomorphism_holdout
```

If binding discovery explains the result, WGD is absorbed even if the final
artifact contains a beautiful grammar.

### Verdict

```text
BINDINGS ARE NOT PRELUDE. THEY MAY BE THE WHOLE GAME.
```

## I344: Attack Verifier And Obligation Smuggling

### Attack

WGD wants discovered obligations. That is the most important upgrade over
plain transformation synthesis, and the most obvious smuggling route.

Obligations can be supplied through:

- public scorer semantics;
- labels such as valid, invalid, unsafe, ambiguous, or repairable;
- failure messages;
- counterexample shape;
- test-case naming;
- evaluator APIs;
- hidden-family-specific verifier templates;
- report fields that the learner is trained to fill.

If the system learns obligations by reading how the harness complains, then the
world did not reveal geometry. The experimenter did.

### Harder Attack

Even a binary invalid label can transmit an obligation boundary. If the label is
behaviorally necessary, it must be charged and shared. If the label is not
behaviorally necessary, obligation claims should not use it.

### Absorption Route

```text
WGD_ABSORBED_BY_VERIFIER_TEMPLATE_SUPPLY
WGD_ABSORBED_BY_OBLIGATION_LABEL_SUPPLY
```

### Pre-Design Demand

The spec must distinguish:

```text
training_feedback_labels
evaluation_only_labels
learner_output_obligation_predicates
hidden_gold_obligation_predicates
scorer_explanation_bits
```

Obligation F1 is allowed only if obligations are behaviorally identifiable from
permitted evidence. If multiple obligation sets imply the same action behavior,
the claim ceiling must fall to action validity only.

Native absorbers must include an obligation-template library and a verifier
oracle control. If either matches, WGD is not signal.

### Verdict

```text
DISCOVERED OBLIGATIONS ARE THE HOME RUN. SUPPLIED OBLIGATIONS ARE FRAMESEED.
```

## I345: Attack Identifiability

### Attack

There are two ways WGD can be void before it starts:

1. The hidden geometry is inferable from surface leakage, so discovery is cheap
   for boring reasons.
2. The hidden geometry is not identifiable from permitted evidence, so any
   claimed discovered grammar is arbitrary.

The spec must live between those failures. That is harder than saying "opaque
typed worlds."

### Harder Attack

Obligation geometry may be underdetermined even when action behavior is
identified. Example: two different hidden obligations can reject the same set of
training actions but imply different repair explanations or future unsafe
conditions. If held-out tasks do not separate them, the metric is rewarding a
chosen explanation, not discovered truth.

### Absorption Or Void Route

```text
WGD_ABSORBED_BY_GENERATOR_LEAKAGE
WGD_VOID_UNIDENTIFIABLE_GRAMMAR
WGD_VOID_SUBJECTIVE_HIDDEN_SEMANTICS
```

### Pre-Design Demand

Before hidden open, the spec must define an identifiability audit:

```text
For each claimed grammar component, construct or search for an alternative
component that is public-evidence-equivalent but hidden-evaluation-distinct.
```

If such alternatives exist and the evaluation does not separate them, no claim
for that component. If no alternatives exist because the public substrate
already pins the component, run the representation-prior and generator-leakage
absorbers first.

### Verdict

```text
WGD MUST PROVE THE TARGET IS NEITHER LEAKED NOR METAPHYSICAL.
```

## I346: Attack LLM Priors And Human Substrate

### Attack

If WGD uses an LLM or natural-language descriptions, the system may import a
large prior over spreadsheets, databases, workflows, safety, units, dates,
entities, and repairs. That prior was paid for elsewhere and cannot be handed
only to the claimed system.

If WGD uses a symbolic learner, the human may import the same prior through a
parser, DSL, hand-authored predicate set, benchmark constructor, or feature
extractor.

### Harder Attack

"Cheap" can be fake. A tiny runtime system with a massive pretrained prior, or
a small symbolic engine sitting on a hand-authored domain DSL, is not cheap in
the sense the Vision cares about unless the prior/substrate is charged and
democratizable.

### Absorption Route

```text
WGD_ABSORBED_BY_REPRESENTATION_PRIOR
WGD_ABSORBED_BY_HUMAN_SUBSTRATE
WGD_VOID_BASELINE_ASYMMETRY
```

### Pre-Design Demand

The spec must include:

```text
language_condition: natural names/descriptions allowed
symbolic_erasure_condition: names/descriptions replaced with opaque symbols
no_language_condition: no NL strings, no pretrained semantic model
substrate_charged_condition: parser/DSL/features charged as H
```

If only the language condition works, the honest claim is that a language prior
helps solve the world. That may be useful, but it is not a clean WGD signal
unless baselines receive and charge an equivalent prior.

### Verdict

```text
DO NOT LET PRETRAINED SEMANTICS OR HUMAN DSL DESIGN DO THE DISCOVERY.
```

## I347: Attack The Output Artifact

### Attack

An inspectable executable grammar can be post-hoc compression. A system can
solve tasks by search, synthesize per-task programs, then compress them into a
grammar-looking artifact for the report. That is not discovery unless the
grammar causally improves held-out action, repair, abstention, and transfer.

### Harder Attack

Interpretability is not evidence. A neat grammar is not signal if:

- removing it does not hurt;
- replacing it with an MDL library performs the same;
- it is created after seeing held-out failures;
- it cannot predict new invalid/unsafe cases;
- it does not reduce all-in cost on fresh siblings.

This is the SHEETS packet-erasure lesson in WGD form.

### Absorption Route

```text
WGD_ABSORBED_BY_POST_HOC_COMPRESSION
WGD_ABSORBED_BY_MDL_LIBRARY_LEARNING
```

### Pre-Design Demand

The learned grammar must be frozen before held-out evaluation. Required
ablations:

```text
grammar_erasure
obligation_erasure
repair_erasure
abstention_erasure
replace_with_mdl_library
replace_with_per_task_programs
randomized_obligation_labels
held_out_composition_forms
```

If erasing the grammar does not reduce functional performance, the grammar was
not the active ingredient.

### Verdict

```text
AN INSPECTABLE GRAMMAR IS NOT ENOUGH. IT MUST DO CAUSAL WORK.
```

## I348: Attack Repair And Abstention Metrics

### Attack

Repair and abstention are supposed to protect WGD from pure synthesis
absorption. They can also be gamed.

Abstention can look good if invalid/unsafe cases are common and coverage is
not charged. Repair can look good if the nearest valid output is obvious from
the scorer or if local repair means "try common edits until accepted."

### Harder Attack

Repair is just another active query channel unless attempts are charged.
Abstention is just a classifier unless calibrated against ambiguity and
opportunity cost. Invalidity detection is just anomaly detection if the data
distribution makes invalid cases outliers.

### Absorption Route

```text
WGD_ABSORBED_BY_ACTIVE_LEARNING
WGD_ABSORBED_BY_ANOMALY_OR_CONSTRAINT_SOLVER
WGD_NEGATIVE_LOW_COVERAGE
```

### Pre-Design Demand

The spec must require:

```text
risk_coverage_curve
abstention_cost
false_abstention_cost
unsafe_false_negative_cost
repair_attempt_budget
repair_distance_metric
locality_of_repair
repair_without_feedback condition
repair_with_feedback charged condition
```

Repair success must be compared against constraint solvers, nearest-valid
search, and active retry baselines. Abstention must be compared against
calibrated uncertainty and anomaly-detection baselines.

### Verdict

```text
REPAIR AND ABSTENTION ARE NOT MAGIC. THEY ARE CLASSIFIERS, SEARCH, OR QUERIES UNTIL PROVEN OTHERWISE.
```

## I349: Attack Sibling And Composition Gates

### Attack

Sibling transfer is the proposed home-run evidence. It is also where generator
design can smuggle the answer. If siblings share the same latent primitives,
library learning should win. If siblings are too different, nobody can transfer
and the test becomes arbitrary. If siblings share names, types, or distribution
fingerprints, schema matching wins.

Composition gates are similarly fragile. If the generator composes operations
from a known finite set, CEGIS and library learning can synthesize the
composition. If composition forms are supplied by the action interface, the
operation ontology has already been transmitted.

### Harder Attack

The AFTD denominator can be gamed by counting many near-duplicate siblings. A
grammar that reduces ten clones is not a paradigm shift. It is amortized
overfitting to a generator.

### Absorption Route

```text
WGD_ABSORBED_BY_SIBLING_LIBRARY_LEARNING
WGD_ABSORBED_BY_GENERATOR_REUSE
WGD_TRAP_NEAR_DUPLICATE_SIBLINGS
```

### Pre-Design Demand

The spec must define sibling independence:

```text
no shared role names
no shared schema positions
no shared nuisance fingerprints
held-out composition forms
minimum behavioral distance between siblings
maximum generator-template overlap
```

AFTD must be all-in and clone-resistant:

```text
AFTD_all_in = all_counted_cost / count_nonduplicate_reduced_siblings
```

The `count_nonduplicate_reduced_siblings` function must be frozen before hidden
open. If the count is hand-adjusted after results, void.

### Verdict

```text
TRANSFER IS EVIDENCE ONLY IF THE SIBLINGS ARE NOT A DISGUISED TRAINING SET.
```

## I350: Final Pre-Design Attack Synthesis

### Attack

WGD survives as a question, not as a default belief. The clean adversarial
position is:

```text
Every WGD success is presumed to be PBE, CEGIS, active learning, library
learning, schema matching, verifier-template supply, operation-ontology supply,
generator leakage, or human substrate until the spec defeats those explanations
with native absorbers and clean accounting.
```

The precommit spec must not merely list absorbers. It must make them terminal.

### Required Terminal Tokens

Minimum B42 additions to B41's token list:

```text
WGD_SIGNAL
WGD_NEGATIVE
WGD_INCONCLUSIVE_BASELINES_NOT_NATIVE
WGD_VOID_PROTOCOL_OR_LEAKAGE
WGD_VOID_SUBSTRATE_ASYMMETRY
WGD_VOID_UNIDENTIFIABLE_GRAMMAR
WGD_VOID_SUBJECTIVE_HIDDEN_SEMANTICS
WGD_TRAP_NEAR_DUPLICATE_SIBLINGS
WGD_ABSORBED_BY_PBE
WGD_ABSORBED_BY_CEGIS
WGD_ABSORBED_BY_ACTIVE_CEGIS
WGD_ABSORBED_BY_ACTIVE_LEARNING
WGD_ABSORBED_BY_MDL_LIBRARY_LEARNING
WGD_ABSORBED_BY_SIBLING_LIBRARY_LEARNING
WGD_ABSORBED_BY_SCHEMA_OR_BINDING_DISCOVERY
WGD_ABSORBED_BY_ENTITY_RESOLUTION
WGD_ABSORBED_BY_CONSTRAINT_DISCOVERY
WGD_ABSORBED_BY_OPERATION_ONTOLOGY_SUPPLY
WGD_ABSORBED_BY_VERIFIER_TEMPLATE_SUPPLY
WGD_ABSORBED_BY_OBLIGATION_LABEL_SUPPLY
WGD_ABSORBED_BY_GENERATOR_LEAKAGE
WGD_ABSORBED_BY_GENERATOR_FAMILY_IDENTIFICATION
WGD_ABSORBED_BY_HAND_AUTHORED_SUBSTRATE
WGD_ABSORBED_BY_REPRESENTATION_PRIOR
WGD_ABSORBED_BY_POST_HOC_COMPRESSION
```

### Required Precommit Rule

```text
If the strongest boring explanation is not implemented natively, proven as a
formal lower bound, or explicitly labeled as untested before hidden open, then
the result cannot be WGD_SIGNAL.
```

### Hostile Reviewer Sentence

```text
Show me that the system discovered executable obligation geometry, not that your
typed substrate, generator, query oracle, schema fingerprints, or library learner
already contained it.
```

### Final Token

```text
WGD_PRE_DESIGN_ATTACK_COMPLETE_SPEC_MUST_START_FROM_ABSORPTION
```

### Final Position

WGD is still the best next moonshot direction because it asks the right
post-FrameSeed question: discovery, not transmission. But the pre-design is not
allowed to inherit optimism from that fact. It must begin with the assumption
that WGD is ordinary program synthesis, active learning, schema matching, or
library learning in costume, and make those absorbers lose under equal
information before any signal is claimed.

The adversary is not yet won over by WGD. The adversary may be won over by a
spec that can kill WGD cleanly.