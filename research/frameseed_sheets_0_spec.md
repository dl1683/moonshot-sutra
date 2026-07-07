# FRAMESEED-SHEETS-0 Precommit Specification

**Date:** 2026-07-07  
**Status:** PRECOMMIT SPEC ONLY. No implementation, no hidden run, no result.  
**Role:** W-Loop B29 worker  
**Purpose:** typed-domain escape from the Boolean absorption result.

## 0. Scope

FRAMESEED-0 Boolean emitted:

```text
FRAMESEED_T3_ABSORBED_BY_TEACHING_DIMENSION
```

That is an honest negative/absorption result, not a harness failure. Boolean
worlds are small enough that exact finite teaching/search solves them. The next
arena is:

```text
Can a compact, inspectable packet transmit reusable typed frames to a bounded
cheap learner in spreadsheet/data-cleaning worlds where brute search over
schemas, joins, units, records, constraints, and actions is combinatorially
expensive?
```

Maximum positive claim:

```text
controlled evidence for typed amortized frame-teaching separation
```

Banned claims: cheap general intelligence, intelligence vaccines generally work,
scale has been defeated, or FrameSeed is proven outside this typed synthetic
protocol.

Binding invariants:

1. Swing for the home run: cheap, useful, improvable intelligence for ordinary
   people, not another toy benchmark.
2. The loop stops only when a hostile fresh-eyes adversary cannot move the token.

## 1. Terminal Tokens

SHEETS-0 uses the same absorption ladder as the hardened Boolean spec, with
SHEETS-specific names. If an implementation keeps the original token namespace,
these map one-to-one to `FRAMESEED_T3_*`.

```text
FRAMESEED_SHEETS_T3R_SIGNAL
FRAMESEED_SHEETS_ABSORBED_BY_TEACHING_DIMENSION
FRAMESEED_SHEETS_ABSORBED_BY_REPRESENTATION_PRIOR
FRAMESEED_SHEETS_ABSORBED_BY_NUISANCE_ORACLE
FRAMESEED_SHEETS_ABSORBED_BY_LIBRARY_LEARNING
FRAMESEED_SHEETS_ABSORBED_BY_ACTIVE_LEARNING
FRAMESEED_SHEETS_ABSORBED_BY_CEGIS
FRAMESEED_SHEETS_ABSORBED_BY_RAG
FRAMESEED_SHEETS_BOOLEAN_TRAP
FRAMESEED_SHEETS_VOID_SMUGGLED_FRAME
FRAMESEED_SHEETS_NEGATIVE
```

Precedence:

1. Smuggling, hidden leakage, baseline parity failure, constructor
   noninterference failure, or post-hidden protocol mutation -> VOID.
2. Typed task degenerates into renamed Boolean masks, one-hot lookup, binary-only
   labels, or type/name leakage -> BOOLEAN_TRAP.
3. L0 already contains the frame as a low-cost primitive, parser, macro,
   verifier, feature mask, tie-break, or reachable hypothesis ->
   REPRESENTATION_PRIOR.
4. L3 full packet misses typed hidden thresholds -> NEGATIVE unless a stronger
   absorber applies.
5. Absorbing baseline at matched or less-than-4x total budget -> the most
   specific absorber, in this order: TEACHING_DIMENSION, LIBRARY_LEARNING,
   NUISANCE_ORACLE, CEGIS, ACTIVE_LEARNING, RAG.
6. All signal gates pass -> T3R_SIGNAL.
7. Any remaining non-smuggling failure -> NEGATIVE.

Multi-absorption rule:

```text
If at least two of teaching dimension, representation prior, active learning,
CEGIS, RAG, nuisance oracle, or library learning absorb in SHEETS-0, kill
FrameSeed as the main moonshot direction or radically reframe it.
```

Allowed reframes: typed CEGIS/proof compilation, MDL library learning, active
counterexample selection, or self-discovered transformation grammars. Disallowed
rescues: weakening baselines, hiding typed semantics from baselines, larger L3,
natural-language packet priors, or threshold changes after hidden opening.

## 2. 20-Iteration Design Ledger

### W29-I1: Directive Grounding

The supervisor directive is to specify the typed escape, not to rescue the
Boolean result. SHEETS-0 must be a precommit, not a posthoc positive story. It
inherits the same harness integrity, absorption ladder, AFTD requirement,
role/schema permutation stability, packet-erasure tests, and claim ceiling.

### W29-I2: Domain Selection

The domain is small spreadsheet/data-cleaning automation with typed objects:

```text
Table, Row, Column, Cell, StableID, DisplayName, UnitValue(value, unit), Date,
ForeignKey, Constraint, Action, TypedProgram, FramePatch, Binding
```

Practical mistakes must look like real automation failures: joining by row
order, trusting display names, adding inches to centimeters, accepting invalid
records, or acting before checking constraints.

Typed-output floor:

```text
At least 50% of hidden queries must require non-Boolean typed outputs:
canonical ID, rational UnitValue, canonical record, canonical row multiset, or
normalized aggregate. Boolean accept/reject is allowed only for guarded action
families.
```

### W29-I3: Candidate Frames

Primary frames:

```text
Frame A: stable-key identity.
Display names, row positions, and formatting are not identity. Stable IDs
survive display-name drift, duplicate names, row shuffles, and formatting.

Frame B: unit normalization.
Numbers with units must be converted to a common dimension/unit before
comparison, aggregation, thresholding, or action.
```

Composition frame:

```text
Frame C: guarded typed composition.
Before applying a join, update, aggregate, or action, validate uniqueness,
referential integrity, type/range checks, missing keys, duplicate keys, and unit
dimension compatibility.
```

Signal requires A and B separately plus at least one hidden family requiring
composition. A result that solves only key matching is not SHEETS-0 signal.

### W29-I4: Frame/Binding Separation

Reusable frame packet:

```text
F_key: stable identifiers are row-order and display-name invariant; key joins
align records by canonicalized ID equality.

F_unit: UnitValue values are normalized through a shared unit registry before
math.

F_guard: typed actions execute only after finite constraint obligations pass.
```

Task binding packet:

```text
B_task: table/column ids mapped to stable-id candidate, display-name candidate,
value column, unit column, foreign-key relation, output field, and constraint.
```

Signal requires costs to be separated:

```text
|V_full| = |F_frame| + sum |B_task_i| + examples + counterexamples + verifiers
           + residual teaching bits + optional final program bits
```

If the packet wins by naming target columns, it is a binding/teaching result, not
frame transfer. If the frame alone solves target tasks without task bindings,
void or representation-prior absorption applies.

### W29-I5: World Generator

For nuisance size:

```text
m in {4, 16, 64, 256}
rows_per_table r in {16, 32, 64, 128}
tables_per_world in {2, 3}
```

Each world contains entity, event/update, and optional auxiliary tables. Latent
roles are:

```text
K = stable key columns
D = display-name or alias columns
P = row-position/order trap
V = numeric value columns
U = unit columns
C = constraint columns or obligations
N = nuisance columns
```

Column names are random or adversarial opaque strings. Scientific runs must not
use semantic names such as key, id, unit, amount, name, valid, foreign, join,
customer, target, or hidden. Any semantic parser from column names to roles voids
the run.

Stable IDs are formatted variants of canonical entity identifiers. Display names
collide, drift, alias, and go missing. Unit values use exact rational conversion
factors. The public unit registry is shared with every system if L3 can use it;
the frame is normalize-before-math, not memorizing conversion factors.

### W29-I6: Typed Ambiguity

Every world must contain public slices where wrong heuristics agree with the
correct rule and hidden slices where they fail.

Stable-key requirements:

```text
public row-position join agrees at least once
hidden row shuffle breaks row-position join
display-name duplicate breaks name matching
stable-ID formatting variant requires canonicalization
```

Unit requirements:

```text
public raw aggregation agrees at least once
hidden raw aggregation fails
hidden unit alias appears in public registry but not packet examples
nuisance numeric column is public-correlated and hidden-broken
```

Constraint/action requirements:

```text
valid-looking invalid row
missing key
duplicate key
unit-dimension mismatch or out-of-range typed value
```

Type tags may not isolate the answer. Each target role must have same-type
decoys:

```text
stable-id-like columns >= max(3, ceil(log2(m + 4)))
display-name-like columns >= max(3, ceil(log2(m + 4)))
numeric value columns >= max(4, ceil(log2(m + 4)))
unit columns >= max(3, ceil(log2(m + 4)))
constraint-like fields >= 2 when constraints are active
```

If public types identify the target columns, emit representation-prior
absorption or void if the leak is accidental metadata.

### W29-I7: Hidden Families

Freeze hidden families before any run:

```text
H1_KEY_RENAME:
  key merge under renamed columns, shuffled rows, and ID formatting changes.

H2_KEY_ADVERSARIAL_NAME:
  duplicate/missing/display-name drift where display-name matching fails.

H3_UNIT_NORMALIZE:
  mixed units, new unit aliases, nuisance numeric columns, exact rational output.

H4_KEY_UNIT_COMPOSED:
  join by stable key, then normalize units before aggregation or comparison.

H5_CONSTRAINT_ACTION:
  validate uniqueness, foreign keys, range/type, and unit dimension before typed
  action or canonical rejection reason.

H6_FULL_STRESS:
  high m, row shuffle, misleading names, missing aliases, duplicate keys,
  nuisance columns, new units, and composed join-normalize-guard tasks.
```

Tuning may use seen families only. Hidden evaluation must include held-out
schemas, units, constraints, row counts, nuisance columns, display-name drift
patterns, and output forms.

### W29-I8: Query And Output Semantics

A query is:

```text
q = (input_tables, operation_request, optional_action)
```

Operation requests include lookup, merge_update, aggregate_by_key,
compare_threshold, validate_and_apply, and canonical_join. Outputs include
StableID, UnitValue(Rational, Unit), CanonicalRecord, CanonicalRowMultiset,
ActionAccepted(canonical_effect), and ActionRejected(canonical_reason_code).

Hidden query mix per world:

```text
20% row-order traps
20% display-name/adversarial-alias traps
20% unit-normalization traps
20% nuisance-column traps
20% composed join-unit-constraint traps
```

All numeric outputs use exact rationals. Row outputs canonicalize by stable IDs
and typed fields, never row order.

### W29-I9: Learner States

Pre-packet learner:

```text
L0 = (R0, H0, A0, B0)
```

R0 includes typed storage, row/column lookup by explicit id, same-column
equality, raw rational arithmetic, literal snippets, finite examples/labels, and
opaque typed records.

R0 excludes low-cost versions of canonicalize_stable_id, join_on_key,
display-name invariance, row-order invariance, normalize_unit,
unit_dimension_check, group_by_key, validate_constraints, foreign_key_repair,
missing/duplicate-key policy, and search over all schema bindings.

L3 after packet:

```text
L3(packet V, public_schema Sigma, query q) -> typed output
```

L3 has no active hidden queries. It can execute only public typed schema,
packet-declared transforms/verifiers/macros, and serialized task bindings.
Natural-language column interpretation is banned.

### W29-I10: SIR-0 Representation Language

SIR-0 is the Sheet Intermediate Representation.

Always-public primitives, shared with all systems:

```text
cell(table,row,column), type_of(column), rows(table), columns(table),
literal(value), eq_same_type(a,b), raw_add, raw_compare, lookup_literal,
unit_registry_lookup(unit), serialize_record(record)
```

Packet-installable or baseline-searchable primitives:

```text
canonicalize_id, same_entity, join_on_key, normalize_unit, group_by_key,
aggregate_normalized, validate_unique, validate_foreign_key, validate_range,
validate_unit_dimension, guard_action, canonical_row_multiset
```

Banned in executable packet fields:

```text
target_key, true_key, stable_id_role, display_name_role, unit_role, nuisance,
hidden_family, answer_column, generator_seed, row_trap, oracle_label,
target_program, solution_schema
```

L2 and library baselines may search typed primitives as a strong absorber. L0 and
L3 pre-packet may not get them for free.

### W29-I11: SHEETP-0 Packet

Packet object:

```text
V = (header, TE, TC, TH, TV, FP, B, CP)
```

Fields:

```text
header: version, schema hash, L0/H0/SIR-0 hash, seed-manifest hash
TE: typed examples with operation request and typed output
TC: typed counterexamples to row-position, display-name, raw-unit, missing-key,
    and duplicate-key rules
TH: transform hints for ID canonicalization, unit normalization, row-order
    invariance, typed grouping, and schema-binding invariance
TV: verifier clauses over keys, joins, units, constraints, and actions
FP: frame patches, operators, verifiers, transforms, search metrics, or macros
B: charged task-specific bindings
CP: optional compact SIR-0 programs, counted and baseline-visible
```

No prose is executable. Prose bytes may be present only as counted inert data.
Every executable field must compile to SIR-0.

### W29-I12: Cost Model

Canonical binary serialization defines packet length:

```text
|V| = bit length of canonical SHEETP-0 serialization
```

Costs include table ids, row ids, column ids, opcodes, typed literals, rational
payloads, string bytes, unit ids, table snippets, bindings, AST nodes, frame
patch schemas, verifier obligations, oracle queries/answers, final programs,
learned libraries/macros, and residual sibling teaching bits.

Input tables at inference are task input, not packet bits. Any task-specific
selection, binding, macro, learned library, verifier expansion, final executable
program, or residual teaching data is counted.

Report:

```text
P_full(m), P_frame(m), P_binding_per_task(m), P_examples(m), P_verifier(m),
P_final_program(m), P_library(m)
```

Sublinear growth requirements:

```text
alpha_hat(P_frame versus m) <= 0.50
alpha_hat(P_full versus m) <= 0.65
P_frame(m)/m decreases from 16 to 64 to 256
no individual seed has P_full slope > 0.85
```

### W29-I13: Representation-Noncontainment

Before hidden evaluation, freeze:

```text
R0 primitive list
H0 hypothesis grammar
A0 update/search algorithm
B0 packet/query/search/runtime/description budgets
SIR-0 public primitive list
packet-installable primitive list
teaching-dimension solver or approximation
role/schema-isomorphism test
unit/parser parity statement
```

The certificate passes only if:

```text
1. Bounded non-reachability: no h in Reach(L0, public_schema, public_data, B0)
   reaches typed threshold on target plus counted siblings.
2. No low-cost named primitive: no R0 primitive, parser, feature, macro,
   verifier, transform, binding rule, tie-break, or search metric is isomorphic
   to stable-key identity, unit normalization, or guarded typed action.
3. No equivalent teaching set: TD-H0 cannot induce target plus sibling transfer
   at matched or less-than-4x total packet/query budget.
4. Type ambiguity: public type tags do not isolate target roles.
```

Failure of 1, 2, or 4 emits representation-prior absorption. Failure of 3 emits
teaching-dimension absorption.

### W29-I14: Baselines

All baselines receive the same public typed schema, typed parsers, unit registry,
operation grammar, packet entries, target tasks, sibling tasks, hidden families,
and role/schema permutations, or a lossless executable translation.

A baseline absorbs if it reaches typed hidden threshold at matched budget or at
less-than-4x total information budget. Exactly 4x or more is reported but does
not absorb.

Baselines:

```text
TD-H0:
  shortest typed teaching set over original H0. No representation additions.

L0 rote/retrieval:
  stores snippets as opaque typed records. Does not execute transforms,
  verifiers, patches, or compact programs.

L1 active typed learner:
  can query row shuffles, duplicate names, unit swaps, missing keys, duplicate
  keys, and invalid constraints under budget. Cannot use representation patches
  outside R0.

L2 typed CEGIS:
  receives SIR-0 with typed joins, filters, unit conversion, grouping,
  validation, and bounded macros as searchable DSL components. All failed
  queries and final programs are counted.

RAG:
  indexes canonical typed fields and text. Variants: RAG-NN, RAG-CLAUSE,
  RAG-PROG, RAG-PATCH.

Nuisance-oracle suite:
  O0 relevant-column oracle, O1 function-only MDL, O2 invariant active typed
  learner, O3 nuisance-ignoring CEGIS, O4 randomized nuisance relabeling.

Library learner:
  may invent reusable typed macros such as canonicalize_id, join_on_key,
  normalize_unit, validate_constraints, and guarded_action across target and
  siblings. Total library + per-task program + query bits are counted.
```

If L2 or the library learner matches under less-than-4x budget, accept the
absorption. They are serious expected absorbers, not strawmen.

### W29-I15: Siblings And AFTD

Every target world has at least three siblings:

```text
S = {s_key, s_unit, s_composed}
```

Siblings must share the reusable frame but differ in schema, surface names,
column order, row count or arity, display-name drift, units, constraints, and
hidden query set. Target labels are absent from sibling evaluation except for
residual teaching bits explicitly charged.

Definitions:

```text
TD_H0(t) = minimum counted packet length that makes L0 choose h in H0 reaching
           typed threshold on task t with no representation additions.

reduced(sibling) iff TD_after(F_frame, sibling) <= 0.50 * TD_H0(sibling)
                   and HFA_typed >= 0.95.

AFTD(F_frame,S) = |F_frame| / count_reduced_siblings(S)

AFTD_all_in = (|F_frame| + sum residual sibling teaching bits + sum binding bits
               + final program/library bits) / count_reduced_siblings(S)
```

Signal requires:

```text
count_reduced_siblings(S) >= 3
AFTD(F_frame,S) < 0.25 * mean_i TD_H0(sibling_i)
AFTD_all_in < 0.50 * mean_i TD_H0(sibling_i)
```

### W29-I16: Measurement Protocol

Evaluation size:

```text
for each m in {4,16,64,256}:
  hidden_worlds_per_m >= 64
  role_schema_permutations_per_world >= 10
  hidden_queries_per_world >= 256
  sibling_tasks_per_world >= 3
```

Typed hidden functional accuracy:

```text
HFA_typed(system,m,hidden_family,task) = exact canonical typed outputs /
                                         total hidden queries
```

Signal thresholds:

```text
HFA_typed(L3_full,m,family,target) >= 0.95 for every m and family
HFA_typed(L3_full,m,family,sibling_i) >= 0.95 for at least 3 siblings
mean HFA_typed(L3_full) >= 0.97 overall
non-Boolean typed-output floor passes
```

Report total information bits, bits_to_0.95, queries_to_0.95, baseline ratios,
packet construction CPU time, inference CPU time, final AST cost, learned macro
cost, and every baseline's best ratio. Diagnostics cannot override tokens.

### W29-I17: Ablations And Controls

Ablations:

```text
V_examples_only
V_no_key_frame
V_no_unit_frame
V_no_constraint_guard
V_no_bindings
V_bindings_only
V_no_counterexamples
V_no_verifier
V_shuffled_packet
```

Signal requires:

```text
HFA_typed(L3_full) - HFA_typed(L3_ablation) >= 0.20 aggregate
and >= 0.20 for at least three of four m values
```

Examples-only and bindings-only must fail AFTD. Frame-only may not solve target
tasks without task bindings, but it must reduce residual sibling teaching cost.
Packet shuffling must preserve the token unless order is explicitly serialized,
counted, and frozen before hidden opening.

Golden controls:

```text
randomized labels: no system may exceed 0.60 HFA_typed
oracle frame packet: L3 must pass
bad unit registry: verifier must reject incompatible dimensions
swapped key/display roles: audit must detect mismatch
row-order-only world: key frame must not be credited
```

### W29-I18: Noninterference And Smuggling Audit

Before hidden performance, freeze commit, spec hash, public seed, hidden seed
rule, generator, constructor, serializer, SIR-0, SHEETP-0, learners, baselines,
budget policy, timeout policy, unit registry, schema generator, scorer, and token
precedence.

Seed rule:

```text
seen_seed   = sha256(public_seed | "sheets0-seen" | manifest_hash)
hidden_seed = sha256(public_seed | "sheets0-hidden" | manifest_hash |
                     "unopened-until-freeze")
```

Split RNG streams by world structure, schema names, row order, display-name
drift, unit choices, nuisance columns, constraints, packet construction, learner
tie-breaks, baseline tie-breaks, ablation replacements, and hidden queries.

Constructor blind mode:

```text
The packet constructor sees only public typed transcripts, public schemas,
allowed oracle answers, and prior packet entries. It cannot read latent role
maps, hidden labels, hidden family ids, target solution programs, or hidden query
labels.
```

Every packet entry logs provenance: source public fact, constructor rule, bits
charged, and whether it is frame, binding, example, verifier, or program.

Required audits:

```text
1. Generator MI over >= 10,000 dry-run worlds: column name/index, row order, unit
   alias, packet order, and latent role must show no uncounted leakage.
2. Serializer audit: independently parse packet bytes, recompute costs, reject
   banned strings and hidden metadata.
3. Baseline parity audit: every field L3 executes is provided to baselines as
   executable typed data or declared lossless translation.
4. Type-system parity audit: ID/date/rational/unit/table/action parsers are
   shared wherever L3 can use them.
5. Role/schema permutation audit: resample names, columns, tables, rows, units,
   and binding ids; HFA, packet length, AFTD, and token must be stable.
6. Human-labor ledger: human-designed, packet-selected, frozen-before-hidden,
   and hidden-for-eval surfaces are listed.
```

Any constructor, scorer, timeout, baseline adapter, unit parser, or token-policy
change after hidden opening voids the run unless hidden seeds rotate.

### W29-I19: Signal And Absorption Conditions

Emit `FRAMESEED_SHEETS_T3R_SIGNAL` only if all hold:

1. Smuggling and noninterference audits pass.
2. Representation-noncontainment passes.
3. Typed-output floor passes.
4. L3 full packet passes every m, every hidden family, target, and three siblings.
5. Mean HFA is at least 0.97.
6. Packet growth is sublinear.
7. AFTD and AFTD_all_in pass.
8. TD-H0 does not absorb.
9. L0, L1, L2, RAG, nuisance-oracle, and library baselines do not absorb.
10. Ablations drop by at least 20 points where required.
11. Role/schema permutation stability passes:

```text
same terminal token for >= 95% of permutation bundles
HFA_typed std <= 0.02
packet length coefficient of variation <= 0.10
AFTD coefficient of variation <= 0.10
```

12. Randomized-label control stays <= 0.60 HFA.
13. Every packet, query, binding, verifier, frame patch, program, library, macro,
    and residual teaching bit is counted.
14. The result report honors the claim ceiling.

Token-specific absorbers follow Section 1. Mixed results cannot be narrated into
signal.

### W29-I20: B30 Run Protocol

B30 may implement and run SHEETS-0 only after this spec and the Q-Loop B37 attack
are incorporated or explicitly rejected in a dated hardening note.

Allowed B30 sequence:

```text
1. Implement generator, serializer, SIR-0/SHEETP-0, constructor provenance,
   scorer, audits, golden controls, and baselines.
2. Run static audits and golden controls on public/smoke seeds.
3. Freeze manifest, hashes, hyperparameters, baselines, timeouts, budgets, and
   token precedence.
4. Run public smoke measurement on a separate smoke seed.
5. Open hidden seed exactly once.
6. Assign exactly one terminal token.
7. Make no code changes under the same hidden seed after hidden opening.
```

If audit infrastructure cannot pass before hidden opening, B30 emits a blocking
ledger, not a partial performance result. If the hidden result is absorbed, void,
Boolean-trapped, or negative, report it directly.

Narrative gate:

```text
SHEETS-0 matters only if a cheap local learner gains reusable typed automation
frames from a small inspectable packet, and hostile typed baselines with the same
information cannot cheaply reproduce the transfer.
```

## Final Precommit Checklist

- [ ] Domain uses typed tables, records, IDs, display names, units, dates,
      constraints, actions, and non-Boolean outputs.
- [ ] At least 50% of hidden queries require non-Boolean typed outputs.
- [ ] Primary frames are stable-key identity and unit normalization.
- [ ] At least one hidden family composes key join and unit normalization.
- [ ] Frame and binding costs are separated.
- [ ] Every target role has same-type decoys; type tags do not reveal answers.
- [ ] Column names are random/adversarial and not semantically parsed.
- [ ] Hidden families cover renamed columns, nuisance columns, shuffled rows,
      display-name drift, duplicate names, formatted IDs, new units, missing
      aliases, duplicate keys, missing keys, and adversarial valid-looking rows.
- [ ] Query mix includes row-order, display-name, unit, nuisance, and composed
      traps.
- [ ] L0/H0/R0/A0/B0 are frozen.
- [ ] SIR-0 public primitives and packet-installable primitives are separated.
- [ ] Packet fields compile to SIR-0; prose is not executable.
- [ ] Canonical packet serialization counts every used bit.
- [ ] Sublinear growth is required for frame and full packet bits.
- [ ] Representation-noncontainment checks reachability, prior primitives,
      equivalent teaching sets, and type ambiguity.
- [ ] TD-H0, L0, L1, L2, RAG, nuisance-oracle, and library-learning baselines
      receive equal typed information and matched budgets.
- [ ] Baselines optimize functional typed accuracy, not table reconstruction.
- [ ] Unit/date/ID parsers and typed schemas are shared wherever L3 can use them.
- [ ] At least three sibling tasks are required.
- [ ] AFTD and AFTD_all_in are measured.
- [ ] Evaluation uses m = {4,16,64,256}, at least 64 hidden worlds per m, at
      least 10 schema/role permutations, and at least 256 hidden queries/world.
- [ ] Every hidden family and every m must pass.
- [ ] Ablations include examples-only, no key frame, no unit frame, no guard, no
      bindings, bindings-only, no counterexamples, no verifier, and shuffled
      packet.
- [ ] Full-vs-ablation gaps are at least 20 percentage points where required.
- [ ] Constructor provenance exists for every packet entry.
- [ ] Generator MI, serializer, baseline parity, type-system parity,
      role/schema permutation, randomized-label, and human-labor audits run.
- [ ] Hidden seed opens only after manifest freeze.
- [ ] Post-hidden constructor/scorer/baseline/timeout changes void the run.
- [ ] Verdict tokens have exact precedence.
- [ ] Multi-absorption kills or radically reframes FrameSeed.
- [ ] Claim ceiling is binding.

## Final Narrative Section

One-sentence story:

```text
Can a cheap local agent learn the spreadsheet frames humans rely on every day -
stable IDs beat names, units normalize before math, and actions need typed
checks - from a compact public packet rather than from scale?
```

Does it survive "isn't that obvious?":

```text
Only if fair teaching, active learning, synthesis, retrieval, nuisance-oracle,
representation-prior, and library-learning baselines all get the same typed
information and still cannot cheaply match the amortized transfer.
```

Does it survive "so what?":

```text
Yes if it works: spreadsheets are ordinary-user automation, and inspectable frame
packets would point toward cheap, repairable local systems. No if it is absorbed:
then ordinary typed program synthesis or library learning is the honest substrate.
```

If the honest narrative is boring:

```text
Say it. SHEETS-0 is designed to either produce a hostile-baseline-resistant typed
AFTD signal or remove FrameSeed as the exciting explanation.
```