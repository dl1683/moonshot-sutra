# Q-Loop Batch 38: Monitor SHEETS-0 Hardening

**Date:** 2026-07-07  
**Role:** Question-Loop worker  
**Iterations:** I281-I294  
**Status:** hardening monitor; no implementation; no hidden result.

---

## Grounding

Read for this batch:

1. `research/dual_loop_supervisor_checkin_29.md`
2. `research/frameseed_sheets_0_spec.md`
3. `research/question_loop_batch37.md`
4. `code/frameseed0_harness.py`
5. `research/VISION.md`

Binding invariants:

1. Swing for the home run.
2. The loop only stops on a won-over adversary.

Working observation: no separate W-Loop B30 hardening artifact is present in the
requested read set. This batch therefore monitors the current
`frameseed_sheets_0_spec.md` as the hardened target under review. The spec has
absorbed many B37 labels and gates in prose, but prose hardening is not yet an
executable win over the typed absorbers.

## Summary Verdict

```text
SHEETS-0 HARDENING IS STRONGER BUT NOT YET ADVERSARY-WINNING.
TYPED REPRESENTATION-NONCONTAINMENT IS NOT ENFORCEABLE AS WRITTEN.
TYPED BASELINE FAIRNESS IS DECLARED BUT NOT YET OPERATIONAL.
THE MOST LIKELY POST-HARDENING ABSORBER IS TYPED MDL LIBRARY LEARNING /
PBE-CEGIS PIPELINE SYNTHESIS OVER THE SAME SIR-0 OPERATORS AND BINDINGS.
```

The spec is now honest about the danger. It has typed outputs, same-type decoys,
opaque names, cost splits, AFTD_all_in, role permutation audits, parser parity,
generator MI audits, baselines, ablations, and terminal precedence. That is real
hardening.

But the hostile reviewer is not won over. The current positive token still
depends on a certificate whose hardest predicate is semantic:

```text
no low-cost primitive, parser, verifier, transform, binding rule, tie-break, or
search metric is isomorphic to stable-key identity, unit normalization, or
guarded typed action
```

That is not an enforceable condition unless the representation language, parser
inventory, search algorithm, and cost model are finite enough to exhaust. If
they are finite enough to exhaust, the corresponding typed CEGIS/library
baseline gets the same finite search space and likely absorbs.

The home-run version remains alive only if SHEETS-0 tests reusable typed
obligation structure that lowers future binding, transformation-choice,
verification, and local-repair cost after strong table-program, unit, entity,
schema, PBE, wrangling, solver, and MDL-library baselines receive native
executable access.

---

## I281: Hardening By Checklist Is Not Hardening By Token

### Single Most Dangerous Question

Did the spec make B37's absorbers executable, or did it mostly copy their names
into a checklist?

### Attack

The SHEETS spec now contains many B37 demands:

```text
representation-noncontainment
parser and human-labor ledger
frame/binding/program cost split
typed generator leakage audit
goal/obligation semantics
composition and local repair
typed CEGIS and library baselines
```

That is better than the pre-design. But the terminal token ladder still keeps
the generic Boolean-era absorbers:

```text
TEACHING_DIMENSION
REPRESENTATION_PRIOR
NUISANCE_ORACLE
LIBRARY_LEARNING
ACTIVE_LEARNING
CEGIS
RAG
BOOLEAN_TRAP
VOID
NEGATIVE
```

B37 asked for domain-specific absorbers because a positive result can be
demolished differently by SQL, UCUM, entity resolution, schema matching, PBE,
OpenRefine-like action history, constraint solving, and data repair. The
current spec folds several of those into broad L2/library/nuisance labels. That
loses diagnostic precision.

If a SQL-style key-discovery baseline solves H1/H4, the token should say
relational absorption, not generic CEGIS. If a unit-aware dimensional baseline
solves H3/H4, the token should say unit-system absorption, not generic library
learning. If a saved wrangling script solves H6, the token should say
data-wrangling or PBE absorption. The name matters because the next reframe
depends on the absorber.

### Required Hardening

Add executable domain-specific absorber slots, even if they map internally to
generic scorer code:

```text
ABSORBED_BY_RELATIONAL_ALGEBRA
ABSORBED_BY_UNIT_SYSTEM
ABSORBED_BY_EXACT_KEY_OR_ENTITY_RESOLUTION
ABSORBED_BY_SCHEMA_MATCHING
ABSORBED_BY_PBE_OR_WRANGLING_SCRIPT
ABSORBED_BY_CONSTRAINT_SOLVER_OR_DATA_REPAIR
ABSORBED_BY_TYPED_CEGIS
ABSORBED_BY_TYPED_MDL_LIBRARY
```

The result report should name the first domain absorber and the strongest
all-in absorber. Generic token names are not enough for the milestone gate.

### Verdict

```text
THE SPEC IS HARDER, BUT ITS ABSORPTION VOCABULARY IS STILL TOO COARSE.
```

---

## I282: Representation-Noncontainment Is Not Enforceable As A Semantic Claim

### Single Most Dangerous Question

Can the typed representation-noncontainment certificate actually be checked?

### Attack

The certificate requires:

```text
no h in Reach(L0, public_schema, public_data, B0) reaches typed threshold
no low-cost named primitive is isomorphic to the frames
TD-H0 cannot induce transfer at matched or <4x budget
public type tags do not isolate target roles
```

This is only enforceable under a fully finite and frozen universe:

```text
finite R0 primitives
finite parser inventory
finite H0 grammar
finite update/search algorithm
finite budget semantics
finite role-isomorphism checker
finite hidden threshold predicate
```

The spec gestures at freezing those objects, but it does not define a concrete
noncontainment algorithm. "Isomorphic to stable-key identity" is not a syntactic
property. A primitive can contain the frame without being named like the frame:

```text
argmax_unique_overlap_pair
minimal_loss_join_candidate
dimension_compatible_aggregate
same_entity_by_low_edit_distance_and_transitive_closure
deny_action_if_foreign_key_not_total
```

These are not banned terms. They are ordinary typed heuristics. If they are in
R0, representation-prior absorption. If they are not in R0 but are allowed in
L2/library search, they likely absorb there. If they are denied to baselines but
available to L3 through packet patches, void.

### Required Hardening

Replace the semantic certificate with two explicit artifacts:

```text
1. A syntactic reachability enumerator over a frozen finite R0/H0/SIR-0 subset.
2. A red-team primitive-equivalence suite that tests disguised role predicates,
   not just banned names.
```

The report must say which part was exact, which part was approximate, and which
part remains an assumption. If the certificate is approximate, the claim cannot
be "noncontainment passes"; it must be "no absorber found under this bounded
search."

### Verdict

```text
NONCONTAINMENT IS OPERATIONAL ONLY AS A BOUNDED SEARCH RESULT, NOT AS A PROOF.
```

---

## I283: If Noncontainment Becomes Exact, CEGIS Gets Stronger Too

### Single Most Dangerous Question

Does making representation-noncontainment enforceable hand the same machinery
to a typed synthesis absorber?

### Attack

There is a trap in hardening I282. To make noncontainment exact, SHEETS-0 must
define the reachable hypothesis space precisely enough to enumerate or solve.
But once the typed world is finite and enumerable, the baseline can search it.

The spec already names a searchable typed substrate:

```text
canonicalize_id
same_entity
join_on_key
normalize_unit
group_by_key
aggregate_normalized
validate_unique
validate_foreign_key
validate_range
validate_unit_dimension
guard_action
canonical_row_multiset
```

These are exactly the operators a table-program CEGIS solver or MDL library
learner wants. If L3 can install them as packet fields, baselines must receive
lossless executable translations. If baselines receive them, the likely
solution is a compact synthesized program or macro library:

```text
bind candidate key by uniqueness/overlap
canonicalize ID
join
normalize compatible unit fields
group/aggregate
validate constraints
accept/reject action with reason code
```

That is the post-hardening absorber. Stronger noncontainment does not rescue the
frame claim unless it separates "the target frame is not in L0" from "no
searcher with the same typed DSL can cheaply synthesize the target."

### Required Hardening

For each noncontainment check, record the induced synthesis problem:

```text
reachable_program_count
minimum consistent program bits
minimum counterexamples to isolate target
typed pruning factor
best CEGIS cost
best MDL library cost
```

If the exact certificate was obtained by making the world small enough to
search, CEGIS and library learning must get first refusal under the same search.

### Verdict

```text
AN EXACT CERTIFICATE CAN BECOME THE ABSORBER'S BLUEPRINT.
```

---

## I284: Same-Type Decoys Do Not Defeat Role Predictors

### Single Most Dangerous Question

Do same-type decoys prevent leakage, or do typed statistics still identify the
roles?

### Attack

The spec requires multiple stable-id-like, display-name-like, numeric, unit, and
constraint-like decoys. That blocks the weakest leak:

```text
the only ID-looking column is the ID
the only unit-looking column is the unit
```

But typed baselines do not stop at type tags. They use role predicates:

```text
uniqueness
near-uniqueness
cross-table overlap
foreign-key coverage
missingness
duplicate pattern
stability under public examples
unit compatibility with requested aggregation
value distribution plausibility
constraint violation localization
action-label correlation
```

The stable key must have some public affordance or the task is ungrounded. The
unit column must be compatible with the value column or unit normalization is
undefined. The constraint column must correlate with accept/reject evidence or
guard learning is impossible. Those affordances are precisely what schema
matching, entity-resolution, and constraint-learning baselines exploit.

Same-type decoys make the problem less trivial. They do not make role inference
noncontained.

### Required Hardening

Run role predictors as absorbers, not just leakage audits:

```text
P(role | public schema statistics)
P(binding | public examples + schema statistics)
P(frame family | public operation request + public examples)
```

If a predictor can recover bindings or frame family cheaply, emit
representation-prior, schema-matching, exact-key, or unit-system absorption
depending on the recovered role.

### Verdict

```text
TYPE AMBIGUITY IS NECESSARY, BUT ROLE-STATISTIC AMBIGUITY IS THE REAL TEST.
```
---

## I285: Baseline Fairness Requires Native Baselines, Not Just Translations

### Single Most Dangerous Question

Are the typed baselines genuinely fair if they are implemented as generic SIR-0
adapters rather than native prior-art tools?

### Attack

The spec says every baseline receives:

```text
same public typed schema
typed parsers
unit registry
operation grammar
packet entries
target tasks
sibling tasks
hidden families
role/schema permutations
or a lossless executable translation
```

That is a good parity sentence. It is not yet a fair baseline suite.

Fairness in SHEETS-0 means the boring systems get to be good at their native
strengths:

```text
relational baseline: key discovery, join planning, grouping, aggregation
unit baseline: dimensional compatibility, exact rational conversion, aliases
entity baseline: blocking, similarity, transitive closure, active labels
schema baseline: field matching under renames and value distributions
PBE baseline: example-driven typed program synthesis
wrangling baseline: reusable action history and script transfer
solver baseline: uniqueness, FK, range, denial constraints, repair
library baseline: macro invention across target and siblings
```

A "lossless executable translation" into a weak common interface can still
strawman these systems if it removes their search biases, native objective, or
library structure. Conversely, giving L3 a hand-designed SIR-0 interface while
baselines receive only translated packet bytes can privilege L3.

### Required Hardening

For each domain baseline, include:

```text
native hypothesis language
native scoring objective
adapter from SHEETS task to native form
adapter from native output to canonical typed output
budget accounting for adapter, examples, queries, final program, and library
```

The audit must check output equivalence, not just packet-byte parity.

### Verdict

```text
BASELINE PARITY IS NOT BASELINE COMPETENCE.
```

---

## I286: Prior Accounting Can Decide The Result Before The Run

### Single Most Dangerous Question

How are mature typed priors charged?

### Attack

SHEETS-0 sits in a domain where prior art is large. The fairness problem has two
bad extremes:

```text
1. Count the entire SQL/unit/PBE/OpenRefine/library stack against baselines.
   Then L3 wins by using a custom benchmark substrate with hidden human prior.

2. Give all mature typed libraries free prior status.
   Then library and PBE baselines probably absorb immediately.
```

The current spec mostly chooses a shared substrate model: public primitives are
free, packet-installable/searchable primitives are counted or searched. That is
reasonable, but it is incomplete. Human-authored choices remain outside the
ledger:

```text
which SIR-0 operators exist
which unit registry exists
which parser grammar exists
which hidden families exist
which verifier obligations are expressible
which baselines are given native operators
which timeouts are tolerated
```

Those choices can encode the real intelligence. The Vision forbids proxy wins:
if the measurement only shows "the benchmark designer provided the right
geometry," it is not yet a cheap-intelligence principle.

### Required Hardening

Report two regimes:

```text
substrate-free regime:
  all systems get SIR-0, typed parsers, unit registry, and table primitives free.

substrate-charged regime:
  human-authored parsers, registries, operators, and adapters are amortized and
  charged as public infrastructure.
```

Signal should survive the substrate-free regime. The substrate-charged regime is
for honesty about democratized development, not for rescuing the token.

### Verdict

```text
WITHOUT PRIOR ACCOUNTING, BASELINE FAIRNESS IS A POLICY CHOICE.
```

---

## I287: Binding Search Is The Hidden Center Of SHEETS-0

### Single Most Dangerous Question

After hardening, is the hardest part still discovering bindings?

### Attack

The spec correctly separates:

```text
F_frame
B_task
examples
counterexamples
verifiers
program bits
library bits
residual teaching bits
```

But the task family itself makes binding central. The frame "stable IDs beat
names" is cheap. The expensive question is which fields, entity types,
relationships, dimensions, constraints, and actions instantiate it.

The B37 frame/binding attack gets sharper after hardening because the spec adds
same-type decoys and opaque names. Those changes make bindings harder. If L3
gets bindings in B_task and baselines must infer them from scratch, parity fails.
If baselines also get B_task, many will solve. If nobody gets B_task, L3 may
fail or reduce to examples/program synthesis.

The decisive metric is not just AFTD. It is:

```text
binding discovery cost after F_frame versus binding discovery cost without F_frame
```

If F_frame does not reduce future binding discovery under fresh schemas, the
packet is an explanation label, not a transferable capability.

### Required Hardening

Add binding-specific baselines and gates:

```text
schema_matcher_binding_cost
entity_resolution_binding_cost
unit_dimension_binding_cost
constraint_binding_cost
L3_binding_cost_after_frame
binding_error_repair_cost
```

Emit schema-binding absorption if binding bits or binding search dominate the
success cost, even when the frame bytes are small.

### Verdict

```text
THE FRAME CAN BE SMALL BECAUSE THE BINDING IS DOING THE WORK.
```

---

## I288: Generator Leakage Is A Predictability Problem, Not An MI Checkbox

### Single Most Dangerous Question

Does the typed generator leak through public statistics even if MI metrics pass?

### Attack

The spec requires MI audits over surfaces such as column name/index, row order,
unit alias, packet order, and latent role. That inherits the Boolean harness
spirit. It is not enough for typed tables.

Typed leakage is often conditional and relational:

```text
the column with maximum cross-table overlap is the key
the unit column whose symbols dimension-match the requested aggregate is target
the numeric column whose raw values make public examples agree but hidden fail
the constraint whose violations align with action labels is the guard
the duplicate pattern distinguishes display names from stable IDs
the missingness pattern identifies optional versus invalid relationships
```

Marginal MI can be low while a decision-tree or MDL role predictor wins. The
leak is not "unit alias has high MI with role"; it is "unit alias plus value
distribution plus operation request identifies the role."

The Boolean harness checked simple slot/name/orientation leakage because the
Boolean world was simple. SHEETS-0 needs adversarial prediction audits over
feature sets that real baselines use.

### Required Hardening

Replace or supplement MI with predictor audits:

```text
role_predictor_name_only
role_predictor_type_only
role_predictor_stats_only
role_predictor_stats_plus_public_examples
role_predictor_stats_plus_operation_request
binding_predictor_all_public
family_predictor_all_public
```

If the best predictor reaches useful binding accuracy at low cost, it is an
absorber, not just a leakage warning.

### Verdict

```text
LOW MI DOES NOT MEAN THE GENERATOR IS BLIND.
```

---

## I289: Goal Semantics Can Be Smuggled By The Operation Request

### Single Most Dangerous Question

Does `operation_request` tell the learner the hard part?

### Attack

The spec defines a query as:

```text
q = (input_tables, operation_request, optional_action)
```

with operation requests like:

```text
lookup
merge_update
aggregate_by_key
compare_threshold
validate_and_apply
canonical_join
```

This is a major hardening risk. If the operation request says
`aggregate_by_key`, then the learner already knows to look for a key and an
aggregate. If it says `validate_and_apply`, then the learner already knows a
guarded action is intended. If it says `canonical_join`, relational search is
being pointed at the right family.

The spec tries to define goal/obligation semantics, but the positive result can
still be absorbed if the operation request supplies the goal and the packet only
supplies execution details. Conversely, if the operation request is too vague,
hidden scoring becomes subjective or requires hidden goal labels.

### Required Hardening

Separate three task regimes:

```text
operation-given:
  request names the operation family; claim is only execution/binding transfer.

obligation-given:
  request gives finite verifier obligations; solver and CEGIS get same clauses.

goal-ambiguous:
  request leaves operation choice ambiguous; active goal-disambiguation and
  abstention-aware baselines get first refusal.
```

Do not let a result in operation-given mode claim goal/obligation discovery.

### Verdict

```text
IF THE REQUEST NAMES THE OPERATION, THE FRAME MAY ALREADY BE CHOSEN.
```
---

## I290: Composition Does Not Escape Pipeline Synthesis

### Single Most Dangerous Question

Does composing stable key, unit normalization, and guard checks create a new
signal, or just a pipeline?

### Attack

The spec requires composition through H4/H5/H6 and adds `F_guard`. That is the
right move against single-operation absorption. It still may not escape:

```text
canonicalize ID
join
normalize units
group or compare
validate constraints
apply or reject action
```

This is a standard ETL/dataflow/wrangling/program-synthesis pipeline. The more
the spec makes the pipeline explicit in SIR-0, the easier it is for a typed
library learner to discover reusable macros:

```text
normalize_join_aggregate
guarded_update
safe_unit_compare
canonical_entity_merge
```

The composition gate demands sublinear packet growth and ablation drops, but
sublinearity can be a property of macro learning too. A library learner is
supposed to compress repeated pipeline structure.

### Required Hardening

For composition, require direct comparison against:

```text
best single saved pipeline
best library of reusable table macros
best CEGIS-composed program
best action-history transfer script
```

Also report local repair:

```text
repair bits after a wrong key binding
repair bits after a unit alias error
repair bits after a false constraint rejection
regression rate on previously solved families
```

If repair is a full replacement pipeline, absorb into synthesis/library
learning.

### Verdict

```text
COMPOSITION HELPS ONLY IF IT BEATS MACRO PIPELINES ON COST AND REPAIR.
```

---

## I291: Non-Boolean Outputs Can Still Hide A Boolean Core

### Single Most Dangerous Question

Does the typed-output floor actually escape the Boolean trap?

### Attack

The spec requires at least 50% non-Boolean outputs:

```text
StableID
UnitValue(Rational, Unit)
CanonicalRecord
CanonicalRowMultiset
ActionAccepted(canonical_effect)
ActionRejected(canonical_reason_code)
```

This is necessary. It is not sufficient. Many typed outputs decompose into:

```text
Boolean role selection + deterministic renderer
```

Examples:

```text
choose the key column, then copy canonical ID
choose the value/unit columns, then apply registry conversion
choose duplicate/missing-key rule, then emit reason code
choose join relation, then serialize canonical row multiset
```

The hard part may still be a finite selection problem over roles and operators.
Exact rational output and canonical serialization make scoring cleaner; they do
not prove a non-Boolean capability.

### Required Hardening

Report a Boolean-core decomposition audit:

```text
role_selection_bits
operator_selection_bits
deterministic_render_bits
typed_value_computation_bits
hidden output entropy after roles/operators are known
```

If hidden success reduces to selecting a small role/operator set followed by
public deterministic postprocessing, emit typed Boolean-trap or CEGIS/library
absorption.

### Verdict

```text
TYPED OUTPUTS ARE NOT ENOUGH IF THE LEARNING PROBLEM IS STILL FINITE SELECTION.
```

---

## I292: Siblings May Reward Generator-Template Learning, Not Frame Transfer

### Single Most Dangerous Question

Do sibling tasks test reusable frames, or do they share generator templates that
make library learning easy?

### Attack

SHEETS-0 uses siblings for AFTD:

```text
s_key
s_unit
s_composed
```

They share the reusable frame but differ in schema, names, order, row count,
drift, units, constraints, and hidden queries. That is correct in spirit.

But siblings are also where MDL library learning is strongest. If the target and
siblings are generated from the same finite family, the library baseline can
learn the family:

```text
all tasks use stable ID columns with high uniqueness
all unit tasks use one value column plus one unit column
all guard tasks use a small fixed set of constraint templates
all composed tasks use the same pipeline skeleton
```

If sibling structure is too shared, library learning absorbs. If sibling
structure is too different, the frame may not transfer without task-specific
binding help. AFTD can be gamed on either side.

### Required Hardening

Make sibling diversity measurable:

```text
template_id independence
operator skeleton edit distance
binding role distribution shift
unit-dimension distribution shift
constraint-family distribution shift
public-example version-space overlap
```

Then run a template learner:

```text
learn generator family from target/public siblings
predict sibling bindings and programs
charge learned library + residual bits
```

If it matches under <4x, emit typed MDL library absorption.

### Verdict

```text
AFTD IS VALID ONLY IF SIBLINGS DO NOT HAND THE GENERATOR FAMILY TO A LIBRARY.
```

---

## I293: The Single Most Likely Absorption Route After Hardening

### Single Most Dangerous Question

After all current hardening, which absorber most likely moves the token?

### Attack

The most likely route is not the simplest one-operation absorber. The current
spec has already made simple absorption harder:

```text
opaque column names
same-type decoys
unit aliases
hidden row shuffles
display-name drift
constraints
composition
typed outputs
AFTD_all_in
ablation controls
role/schema permutation stability
```

Those changes weaken naive SQL, naive unit conversion, and naive exact-key
matching. They do not weaken a typed MDL library learner or PBE-CEGIS pipeline
solver. They strengthen it by making the domain a finite, repeated, typed
program family with reusable operators.

The most likely post-hardening absorber is:

```text
TYPED_MDL_LIBRARY_PBE_CEGIS
```

Operationally:

```text
1. Receive the same SIR-0 public primitives, unit registry, parsers, examples,
   counterexamples, verifiers, bindings, and operation grammar.
2. Search typed table programs for target tasks.
3. Compress repeated subprograms into macros:
   canonicalize_id, join_on_key, normalize_unit, validate_constraints,
   guarded_action, canonical_row_multiset.
4. Transfer macros to siblings with schema-binding search.
5. Pay library + per-task binding/program/query bits.
6. Match L3 hidden HFA and AFTD_all_in under <4x.
```

This route absorbs because it explains exactly what SHEETS-0 is building:
reusable typed transformation programs over spreadsheet worlds. If FrameSeed's
packet is better, it must beat this absorber on all-in cost, local repair, and
fresh-schema binding discovery.

### Required Hardening

Promote `TYPED_MDL_LIBRARY_PBE_CEGIS` to the primary hostile baseline, not one
line inside generic L2/library. Give it native status, multiple search regimes,
and first refusal before signal.

### Verdict

```text
THE POST-HARDENING ABSORBER IS A REUSABLE TABLE-PROGRAM LIBRARY.
```

---

## I294: Final B38 Verdict And Conditions To Win Over The Adversary

### Single Most Dangerous Question

What would make this Q-Loop stop moving the token?

### Attack Synthesis

SHEETS-0 has moved in the right direction. It no longer looks like a casual
"joins and units" demo. It is a precommit with audits, costs, hidden families,
ablation controls, sibling transfer, token precedence, and a claim ceiling.

But the adversary is not won over for three reasons:

```text
1. Typed representation-noncontainment is not enforceable as written.
2. Typed baseline fairness is declared in prose but not proven by native,
   competent, budgeted baseline implementations.
3. The strongest absorber after hardening is the same shape as the benchmark:
   typed table-program/library synthesis over repeated spreadsheet families.
```

### Direct Answers

**Is typed representation-noncontainment enforceable?**

```text
Not as written. It is enforceable only as a bounded finite search over a frozen
R0/H0/SIR-0/parser/budget universe plus adversarial primitive-equivalence tests.
If it remains semantic, it is an assumption. If it becomes exact, typed
CEGIS/library search must get the same exact universe and likely absorbs.
```

**Are typed baselines fair?**

```text
Not yet. The spec states parity, but fair typed baselines require native
relational, unit, entity-resolution, schema-matching, PBE/wrangling,
constraint/data-repair, typed CEGIS, and MDL-library implementations with
matched executable information, matched bindings or binding-search budgets,
adapter costs, timeouts, and output canonicalization.
```

**What is the single most likely absorption route after hardening?**

```text
Typed MDL library learning / PBE-CEGIS pipeline synthesis. It can use the same
typed operators, examples, counterexamples, verifiers, unit registry, and
bindings to learn reusable table-program macros and transfer them to siblings.
```

### Minimum B38 Conditions Before Signal

Before `FRAMESEED_SHEETS_T3R_SIGNAL`, require:

1. Domain-specific absorber tokens or report slots, not only generic CEGIS and
   library labels.
2. Exact finite reachability where possible; bounded-search language where not.
3. A primitive-equivalence red team for disguised key/unit/guard predicates.
4. Native relational, unit, entity, schema, PBE/wrangling, solver, CEGIS, and
   MDL-library baselines.
5. Two prior-accounting regimes: substrate-free and substrate-charged.
6. Binding-specific cost and repair ledgers.
7. Predictor-based leakage audits over public schema statistics, not only MI.
8. Operation-request regimes that separate operation-given, obligation-given,
   and goal-ambiguous tasks.
9. Pipeline/library composition baselines with local-repair accounting.
10. Boolean-core decomposition for typed outputs.
11. Sibling template-independence and generator-family learning audits.
12. `TYPED_MDL_LIBRARY_PBE_CEGIS` treated as the primary expected absorber.

### Final Kill Records

```text
KR-B38-1: If representation-noncontainment is not implemented as a bounded
search/certificate with explicit assumptions, no signal token is interpretable.

KR-B38-2: If native typed baselines are replaced by weak generic adapters, void
for baseline competence failure even when packet-byte parity passes.

KR-B38-3: If public schema/value statistics predict bindings or frame family at
low cost, emit representation-prior, schema-matching, exact-key, or unit-system
absorption.

KR-B38-4: If operation_request names the operation family and the result only
executes that operation, do not claim goal/obligation discovery.

KR-B38-5: If typed outputs reduce to small role/operator selection plus public
deterministic rendering, emit typed Boolean-trap or synthesis absorption.

KR-B38-6: If a typed MDL library / PBE-CEGIS pipeline solver matches target and
siblings under matched or <4x all-in budget, emit typed library/CEGIS absorption
and reframe toward self-discovered transformation grammars.
```

### Final Recommendation

```text
CONDITIONAL ALIVE, BUT THE NEXT HARDENING MUST CENTER THE TABLE-PROGRAM
LIBRARY ABSORBER.
```

SHEETS-0 is still the right next filter because it is practical, typed, and
repair-oriented. It is not yet a won-over-adversary design. The milestone gate
should not ask whether the packet beats straw spreadsheet heuristics. It should
ask whether an inspectable packet transmits reusable typed obligation structure
more cheaply than the best native table-program synthesis and library-learning
systems that get the same typed substrate.

If that answer is no, the honest token is absorption, and the honest reframe is:

```text
self-discovered typed transformation grammars and libraries are the live
substrate, not FrameSeed packets as a separate explanation.
```