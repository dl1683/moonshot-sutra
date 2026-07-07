# Q-Loop Batch 37: Attack SHEETS-0 Pre-Design

**Date:** 2026-07-07  
**Role:** Question-Loop worker  
**Iterations:** I267-I280  
**Status:** pre-design adversarial attack; no implementation.

---

## Grounding

Read in required order: `research/dual_loop_supervisor_checkin_28.md`, `research/work_loop_batch28.md`, `research/question_loop_batch33.md`, `research/question_loop_batch34.md`, `research/frameseed_0_precommit_spec.md`, and `research/VISION.md`.

Binding invariants:

1. Swing for the home run.
2. The loop only stops on a won-over adversary.

Binding result from B28:

```text
FRAMESEED_T3_ABSORBED_BY_TEACHING_DIMENSION
```

The Boolean world did not almost work. It was solved by every baseline. The next design cannot merely add strings, dates, units, rows, and columns. It must avoid building the same small enumerable grammar with prettier nouns.

## Summary Verdict

```text
SHEETS-0 IS ALIVE ONLY AS A HARDER ABSORPTION FILTER.
THE CURRENT PRE-DESIGN IS LIKELY ABSORBED BY RELATIONAL ALGEBRA, UNIT
SYSTEMS, ENTITY RESOLUTION, PROGRAMMING BY EXAMPLE, CONSTRAINT SOLVING,
DATA CLEANING, SCHEMA MATCHING, AND LIBRARY LEARNING.
```

Typed tables are a better narrative arena than Boolean slots, but they sit on top of mature prior art. The dangerous fact is not merely that the tasks are useful in industry. The dangerous fact is that the exact frames named by B34 are canonical operations: join by key, normalize units, match stable identifiers, validate constraints, ignore row order, canonicalize dates, canonicalize IDs, and reconcile names.

If SHEETS-0 tests whether a packet can teach these operations to a learner whose prior already contains parsers, relational operators, unit conversion, fuzzy matching, and constraint validation, it is representation-prior absorption. If the packet supplies those operations, it is programming by example, data wrangling, CEGIS, or library learning. If baselines are denied those operations, it is a parity void.

SHEETS-0 can still be worth running, but only if W29 makes the typed-domain absorbers executable and terminal before any hidden result is opened.

## Prior-Art Pressure Points

Limited primary/official prior-art check:

- Codd's relational model already treats ordering dependence, keys, relations, integrity, and joins as core data-model issues: https://www.seas.upenn.edu/~zives/03f/cis550/codd.pdf
- Microsoft PROSE synthesizes programs from input-output examples over DSLs: https://www.microsoft.com/en-us/research/project/prose-framework/
- Singh and Gulwani's semantic spreadsheet transformations learn typed column transformations from examples: https://arxiv.org/abs/1204.6079
- UCUM gives machine-interpretable unit semantics and dimensional analysis: https://unitsofmeasure.org/ucum
- OpenRefine is a local open-source tool for messy data cleaning, reconciliation, transformation, and augmentation: https://github.com/OpenRefine/OpenRefine
- HoloClean combines constraints, external data, statistics, and probabilistic inference for data repair: https://arxiv.org/abs/1702.00820
- Modern record-linkage work generalizes Fellegi-Sunter style entity matching for multi-file linkage: https://arxiv.org/abs/1205.3217

The point is not citation theater. The point is that a hostile reviewer can name boring systems that already own much of the proposed SHEETS-0 surface.

---

## I267: Relational-Algebra Absorption

### Single Most Dangerous Question

If the proposed frame is "join by key, not row position," why is this not just relational algebra and database normalization?

### Attack

B34's first recommended frame is:

```text
join by stable key, not row position or display name
```

That is not a new typed frame. It is the relational model's oldest lesson: tuples are not identified by display order, and relationships should be represented through domains, keys, and relations rather than storage paths. Codd's 1970 paper is directly about removing ordering, indexing, and access-path dependence from user logic.

SHEETS-0 risks becoming this:

```text
L3 receives a tiny packet saying "foreign_key orders.customer_id -> customers.id"
and beats baselines that are forced to compare row positions or names.
```

That is void if baselines do not get relational operators. It is absorbed if baselines do get them. A SQL baseline, a relational algebra enumerator, or a pandas merge search over candidate keys can solve the toy.

The typed-domain escape can still be a Boolean trap if hidden variation is only renamed columns, shuffled rows, and nuisance columns. A relational solver is designed for exactly those invariances.

### Control Or Redirect

Add a terminal absorber:

```text
FRAMESEED_SHEETS0_ABSORBED_BY_RELATIONAL_ALGEBRA
```

Emit it if a frozen relational baseline reaches target plus siblings under matched or less-than-4x cost using candidate key discovery, primary/foreign-key constraints, joins, projections, selections, grouping, aggregation, row-order invariance, and name-blind schema roles inferred from public examples.

Do not let the L3 packet be the only object allowed to express `join_on_key`. If `join_on_key` is in L3's prior, it is in the baseline prior. If it is in the packet, relational and CEGIS baselines must execute it with the same counted bits.

### Verdict + Kill Records

```text
JOIN-BY-KEY IS ABSORBED UNTIL PROVEN OTHERWISE.
```

Kill records:

```text
KR-I267-1: If the key-based join frame is expressible as ordinary relational algebra over public types and constraints, emit relational-algebra absorption.
KR-I267-2: If row order is the main negative control, require a row-shuffle oracle baseline; if it matches, absorb.
KR-I267-3: If baselines receive flat rows while L3 receives executable relation types, void for baseline parity failure.
```

### Narrative Attack

Strongest dismissal: "You rediscovered SQL joins for spreadsheets."

What would survive: a packet that changes a weak learner's reusable ability to decide which relational structure is semantically valid under ambiguity, not a packet that names or executes a normal join.

---

## I268: Unit-System Absorption

### Single Most Dangerous Question

If the frame is "normalize units before comparison or aggregation," why is this not just dimensional analysis plus a unit conversion library?

### Attack

B34's second recommended frame is:

```text
normalize units before comparison or aggregation
```

That is even more exposed than joins. Unit systems are formal. UCUM exists to make units machine-interpretable and computationally comparable. Dimensional analysis already tells a system that inches and centimeters can be converted, that Celsius needs affine treatment, and that meters cannot be added to dollars.

SHEETS-0 can fail in two opposite ways:

```text
1. L3 receives a unit parser/converter denied to baselines -> void.
2. Baselines receive the same unit semantics -> ordinary unit-normalization baselines solve the task.
```

If hidden tasks only change unit labels, metric/imperial systems, nuisance numeric columns, or column names, then a unit-aware MDL baseline will discover:

```text
parse_unit(value, unit)
convert_to_common_dimension
aggregate
```

This is not a new frame. It is a library call plus a type check.

### Control Or Redirect

Add a terminal absorber:

```text
FRAMESEED_SHEETS0_ABSORBED_BY_UNIT_SYSTEM
```

Run unit baselines:

```text
U0: UCUM/dimensional oracle - parsed unit expressions and dimensions, no target answer.
U1: Unit-library search - conversions, compatibility checks, affine special cases, grouping, aggregation.
U2: Unit-PBE synthesizer - learns intended unit transformation from examples and counterexamples.
U3: Unit-error detector - rejects raw-unit aggregation and incompatible comparisons through constraints.
```

A signal requires L3 to beat all of these. It cannot win by being the only system allowed to understand units.

### Verdict + Kill Records

```text
UNIT NORMALIZATION IS PRIOR ART, NOT A HOME RUN.
```

Kill records:

```text
KR-I268-1: If a unit-library baseline matches L3 under matched or <4x cost, emit unit-system absorption.
KR-I268-2: If L3 gets unit dimensions or conversion factors not available to baselines, void.
KR-I268-3: If the packet carries the conversion table or target aggregation program, route to teaching-dimension, CEGIS, RAG, or library-learning absorption.
```

### Narrative Attack

Strongest dismissal: "You showed that converting inches to centimeters before adding works."

What would survive: a frame packet that helps choose and verify the correct quantity semantics across ambiguous, conflicting columns while unit libraries and PBE systems get the same semantic substrate and still cannot cheaply match.

---

## I269: Entity-Resolution Absorption

### Single Most Dangerous Question

If the frame is "stable identity beats display name," why is this not deterministic or probabilistic record linkage?

### Attack

The stable-ID frame has a trivial branch and a mature branch.

Trivial branch:

```text
There is a stable ID column.
```

Then the solution is exact-key matching. It is absorbed by a deterministic record-linkage baseline, a relational key baseline, or a schema matcher.

Mature branch:

```text
There is no clean stable ID; names, addresses, dates, aliases, and duplicates must be reconciled.
```

Then the task enters entity resolution and record linkage. That field already contains probabilistic matching, blocking, active labeling, clerical review, similarity functions, transitivity constraints, and linkage uncertainty. A FrameSeed packet that gives a few examples of "same entity despite name change" looks exactly like an active-learning or machine-teaching episode for an entity matcher.

The phrase "stable ID survives display-name drift" is not enough. Either the ID is visible, or the match has to be inferred by known record-linkage methods.

### Control Or Redirect

Add terminal absorbers:

```text
FRAMESEED_SHEETS0_ABSORBED_BY_EXACT_KEY_MATCHING
FRAMESEED_SHEETS0_ABSORBED_BY_ENTITY_RESOLUTION
```

Run entity baselines:

```text
E0: exact-key detector - uniqueness, stability, and cross-table agreement.
E1: deterministic linkage - normalized IDs, names, dates, aliases, blocking rules.
E2: probabilistic linkage - match/non-match weights from public examples and constraints.
E3: active entity matcher - buys counterexamples to disambiguate uncertain matches.
E4: transitive clustering - enforces entity-level consistency across more than two tables.
```

If SHEETS-0 includes stable IDs, exact-key detection gets first refusal. If it removes stable IDs, record linkage gets first refusal. The positive token must survive both branches.

### Verdict + Kill Records

```text
ENTITY MATCHING IS NOT EMPTY SPACE.
```

Kill records:

```text
KR-I269-1: If exact stable-key discovery solves the target plus siblings, emit exact-key absorption.
KR-I269-2: If a probabilistic or active record-linkage baseline matches under matched or <4x cost, emit entity-resolution absorption.
KR-I269-3: If the packet says which columns are stable identifiers, charge that as binding; if AFTD depends mostly on that binding, no frame signal.
```

### Narrative Attack

Strongest dismissal: "You taught a toy deduper to prefer IDs over names."

What would survive: a reusable frame that lowers future identity-binding cost across changing schemas and partial evidence after exact-key, probabilistic, and active linkage baselines fail fairly.

---

## I270: Constraint-Solver And Data-Repair Absorption

### Single Most Dangerous Question

If the frame is "validate constraints before action," why is this not database integrity checking, constraint solving, or data repair?

### Attack

B34 names type, range, uniqueness, referential constraints, and action preconditions. Those are standard database and solver objects.

The dangerous version of SHEETS-0:

```text
L3 gets verifier clauses and validates records before acting.
Baselines see examples of accepted/rejected rows but cannot execute the constraints.
```

That is a parity void. If baselines can execute the same constraints, they will likely match. If constraints are hidden and must be inferred, then CEGIS, rule learning, data repair, and active learning become the boring absorbers.

Data-cleaning systems such as HoloClean explicitly combine integrity constraints, external data, statistics, and probabilistic inference to repair inconsistent data. SHEETS-0 cannot call a constraint verifier a "frame" unless it beats constraint-aware baselines.

### Control Or Redirect

Add terminal absorbers:

```text
FRAMESEED_SHEETS0_ABSORBED_BY_CONSTRAINT_SOLVING
FRAMESEED_SHEETS0_ABSORBED_BY_DATA_REPAIR
```

Run these baselines:

```text
C0: declared-constraint executor - executes every verifier/precondition L3 can execute.
C1: constraint learner - searches uniqueness, foreign-key, range, type, and denial constraints.
C2: SMT/finite-domain solver - synthesizes minimal constraints consistent with public examples.
C3: data-repair baseline - uses constraints plus statistics to reject or repair invalid records.
C4: action-guard baseline - learns precondition/action policies from public valid/invalid traces.
```

If any reaches hidden action success under matched or <4x cost, no T3-R signal.

### Verdict + Kill Records

```text
VALIDATION IS A SOLVER BASELINE BEFORE IT IS A FRAME.
```

Kill records:

```text
KR-I270-1: If a finite-domain constraint solver learns or executes the guard, emit constraint-solver absorption.
KR-I270-2: If data repair reaches the same post-cleaning functional success, emit data-repair absorption.
KR-I270-3: If L3 receives verifier clauses unavailable to baselines, void.
```

### Narrative Attack

Strongest dismissal: "You gave one system a validator and asked the others to guess."

What would survive: a packet that changes how a cheap learner composes and audits constraints across new schemas, while solver/data-repair baselines get the same executable obligations and still cannot amortize the cost.

---

## I271: Programming-By-Example And Data-Wrangling Absorption

### Single Most Dangerous Question

Is a SHEETS-0 packet anything more than a data-wrangling recipe or a programming-by-example task?

### Attack

Spreadsheet transformation is one of the strongest prior-art traps in the entire project. FlashFill/PROSE-style systems synthesize programs from examples over DSLs. Singh and Gulwani specifically target semantic string transformations for columns that may represent dates, currencies, and other typed data. Wrangler and OpenRefine-style systems turn interactive cleaning and transformation into reusable scripts or histories.

A SHEETS-0 packet with:

```text
examples
counterexamples
transforms
verifiers
bindings
macros
```

is nearly a PBE/data-wrangling interface. If the hidden tasks are table cleanup, unit conversion, joins, ID canonicalization, or constraint repair, a PBE solver or data-wrangling action-history learner is the first boring explanation.

The positive story cannot be "a small packet taught a table transformation." That is already the pitch of programming by example.

### Control Or Redirect

Add terminal absorbers:

```text
FRAMESEED_SHEETS0_ABSORBED_BY_PBE
FRAMESEED_SHEETS0_ABSORBED_BY_DATA_WRANGLING
```

Run:

```text
P0: PROSE-like typed PBE - synthesizes table/string/unit/date/join transformations from the same input-output examples.
P1: CEGIS table-transform solver - enumerates table programs with counterexample-guided refinement.
P2: OpenRefine-like action-history learner - searches reusable action sequences: split, trim, cluster, reconcile, canonicalize, derive column, join, filter, validate.
P3: saved-script transfer baseline - learns a reusable wrangling script on target tasks and applies it to sibling schemas with binding search.
```

If these match AFTD or hidden success, absorb. Do not weaken them because they are "too close"; closeness is the reason they are mandatory.

### Verdict + Kill Records

```text
SPREADSHEETS MAKE PBE THE DEFAULT ABSORBER.
```

Kill records:

```text
KR-I271-1: If typed PBE synthesizes the same transformation under matched or <4x cost, emit PBE absorption.
KR-I271-2: If a reusable data-wrangling script/action history matches AFTD, emit data-wrangling absorption.
KR-I271-3: If the packet is mostly examples plus transform DSL entries, do not claim T3-R until PBE and wrangling baselines fail fairly.
```

### Narrative Attack

Strongest dismissal: "This is FlashFill plus joins and units."

What would survive: a packet that is not merely a synthesized table program or saved wrangling recipe, but a reusable operational interface for deciding which transformations are valid across unseen schemas.

---

## I272: Typed-Parser Prior Trap

### Single Most Dangerous Question

Where does the type system come from, and why is it not the actual frame?

### Attack

B34 says all systems share public typed schema: type grammar, unit syntax, row/table grammar, operations, and labels. That sounds fair, but it may move the whole answer into the public substrate.

Typed parsing can smuggle the frame:

```text
parse_customer_id
parse_date
parse_currency
parse_unit
canonicalize_id
normalize_name
infer_foreign_key
detect_unique_key
detect_measure_dimension
```

If those exist before the packet, L0 may already contain the frame. If they do not exist, SHEETS-0 becomes parser induction, not frame transfer. If L3 gets them but baselines do not, void.

The typed-domain equivalent of the Boolean trap is:

```text
Replace bits with typed cells, then give the learner a type system whose primitives already encode the intended invariants.
```

The Boolean spec banned `causal`, `alias`, and `select_causal_pair`. SHEETS-0 must likewise ban or charge `stable_id`, `foreign_key`, `unit_dimension`, `canonical_entity`, `valid_record`, `row_identity`, and any parser that exposes those roles at low cost.

### Control Or Redirect

Require a `Typed Representation-Noncontainment Certificate`:

```text
R0_typed primitive list
parser inventory
unit grammar
date grammar
ID grammar
string similarity functions
key discovery functions
constraint language
schema matching features
action semantics
```

For each primitive, run a role-isomorphism audit:

```text
Can this primitive identify the target frame under schema permutation at cost <= B0?
```

If yes, emit representation-prior absorption.

### Verdict + Kill Records

```text
THE TYPE SYSTEM MAY BE THE HIDDEN ANSWER.
```

Kill records:

```text
KR-I272-1: If public typed parsers expose stable IDs, unit dimensions, foreign keys, or constraint roles cheaply, emit representation-prior absorption.
KR-I272-2: If parser invention is required and only L3 can invent parsers, void for asymmetric substrate.
KR-I272-3: If parser cost is outside the packet/baseline budget, cap the claim at "uses a hand-authored typed benchmark," not frame transfer.
```

### Narrative Attack

Strongest dismissal: "The intelligence was in the parser and schema designer."

What would survive: an explicit accounting split between public substrate, charged packet content, discovered bindings, executable transforms, and human-authored type semantics.

---

## I273: Frame/Binding Collapse In Typed Tables

### Single Most Dangerous Question

In SHEETS-0, is the reusable frame actually small while the task-specific binding does all the work?

### Attack

B34 correctly says:

```text
"Column C is the stable ID" is a binding.
"Stable IDs survive display-name drift and row shuffle" is a frame.
```

The problem is that in spreadsheets, the generic frame is often almost free and the binding is almost everything.

Generic frame:

```text
Use stable IDs.
Normalize units.
Join on keys.
Validate constraints.
```

Binding:

```text
Which columns are IDs?
Which IDs refer to the same entity type?
Which units share dimensions?
Which date format is intended?
Which rows are headers?
Which duplicates are legitimate?
Which constraint is policy versus data error?
Which action should be blocked?
```

AFTD can be faked if the reusable frame packet is tiny because it says an obvious general rule, while each sibling gets hidden unreported binding help. Or the reverse can happen: the packet includes bindings, but the report calls them frame cost.

The Boolean result was absorbed because exact finite teaching found the support. The typed version will be absorbed if exact or approximate binding search finds the columns and operations.

### Control Or Redirect

Split all costs:

```text
F = reusable typed frame packet
B_i = task-specific schema/entity/unit/constraint binding packet for sibling i
P_i = executable per-task program or action policy
H = human-authored type-system and parser cost
```

Report:

```text
AFTD_frame_only = |F| / count_reduced_siblings
AFTD_all_in = (|F| + sum_i |B_i| + sum_i |P_i| + charged H share) / count_reduced_siblings
binding_ratio = sum_i |B_i| / (|F| + sum_i |B_i|)
```

If `binding_ratio` dominates, the result is schema binding, not frame transfer.

### Verdict + Kill Records

```text
WITHOUT FRAME/BINDING ACCOUNTING, SHEETS-0 IS UNINTERPRETABLE.
```

Kill records:

```text
KR-I273-1: If task-specific binding bits dominate total success cost, emit FRAMESEED_SHEETS0_ABSORBED_BY_SCHEMA_BINDING.
KR-I273-2: If sibling transfer depends on uncharged binding oracle outputs, void.
KR-I273-3: If F is a generic truism and B_i does the hidden work, no T3-R signal even when AFTD_frame_only looks small.
```

### Narrative Attack

Strongest dismissal: "The frame said 'use the ID'; the hard part was knowing which field was the ID."

What would survive: the same frame packet reduces the cost of discovering new bindings under adversarial schemas, and schema-matching/entity/unit baselines cannot cheaply reproduce that reduction.

---

## I274: Finite Typed Enumeration Trap

### Single Most Dangerous Question

Is SHEETS-0 just a larger finite DSL where CEGIS/search still wins?

### Attack

The supervisor says typed domains should be "combinatorially expensive." That is a hope, not a proof.

A small table world with:

```text
2-4 tables
5-20 columns
small rows
finite units
finite date formats
finite join candidates
finite constraints
finite actions
```

is still enumerable. The combinatorics can be cut by types, uniqueness tests, foreign-key tests, unit dimensions, value overlap, and public examples. A CEGIS/table-program learner does not need to brute-force all strings. It can use the same typed features L3 uses and search likely joins, conversions, filters, and constraints.

The typed Boolean trap is:

```text
The domain looks rich, but the hidden family is generated by a tiny grammar and the answer is a low-cost term in that grammar.
```

If the generator emits only canonical spreadsheet mistakes, a library learner will learn the generator's DSL faster than FrameSeed learns a new frame.

### Control Or Redirect

Before W29 run, compute an enumerability audit:

```text
N_join_candidates
N_unit_transform_candidates
N_schema_bindings
N_constraint_sets
N_action_policies
typed_pruning_factor
public_example_version_space
minimum_distinguishing_counterexamples
```

Then run `TYPED_CEGIS_EXACT` where feasible, `TYPED_CEGIS_BEAM` where exact is infeasible, `TYPED_MDL_LIBRARY` over the generator operator family, and an `ACTIVE_COUNTEREXAMPLE_TABLE` learner.

If these solve, absorb. The fact that the search space is bigger than Boolean does not matter if typed pruning makes it small.

### Verdict + Kill Records

```text
COMBINATORIAL EXPENSE MUST BE MEASURED, NOT ASSUMED.
```

Kill records:

```text
KR-I274-1: If typed CEGIS reaches target plus siblings under matched or <4x budget, emit CEGIS absorption.
KR-I274-2: If version-space size after public examples is small enough for exact teaching/search, emit teaching-dimension absorption.
KR-I274-3: If all hidden tasks are generated by a compact known table DSL, emit typed Boolean-trap unless prior-art baselines fail anyway.
```

### Narrative Attack

Strongest dismissal: "You made a finite spreadsheet puzzle and synthesis found the intended script."

What would survive: measured evidence that typed search remains expensive under the same public substrate while the packet changes reusable representation and not merely the candidate ranking.

---

## I275: Generator And Constructor Smuggling In Typed Worlds

### Single Most Dangerous Question

Can the SHEETS-0 generator or packet constructor launder hidden semantics into a clean-looking typed packet?

### Attack

B34's implementation noninterference warning gets worse in SHEETS-0. Typed worlds have many more leakage channels:

```text
column names
header order
value distributions
ID formats
unit symbol choices
currency conventions
date formats
row counts
duplicate patterns
missingness patterns
foreign-key overlap
constraint violation locations
action labels
generated comments
serialized object reprs
schema ids
```

A packet constructor can look blind while using generator artifacts: choose the only column with UUID-like values, choose the only pair with high overlap, choose the only numeric column with unit suffixes, choose examples where raw-unit aggregation fails maximally, choose constraints that reveal the target action precondition, or choose siblings that reuse the same binding template.

This is not cheating by role names. It is smuggling through distribution design. The Boolean MI audit checked surface names and slot roles. SHEETS-0 needs MI and predictability audits over typed statistics.

### Control Or Redirect

Add a `Typed Constructor Noninterference Contract`:

```text
Allowed public observations:
  explicit public rows, public labels, public schema tokens, public type annotations declared before generation.

Forbidden constructor access:
  latent entity ids, true foreign-key map, unit dimension labels, intended constraint family, hidden sibling family, generator template id, hidden action policy, post-clean truth table.
```

Run leakage probes:

```text
MI(column_name_features, latent_role)
MI(value_distribution_features, latent_role)
MI(format_features, latent_role)
MI(row_order, latent_role)
MI(missingness_pattern, latent_role)
MI(packet_example_order, latent_role)
predict(latent_role | public_schema_statistics)
predict(frame_family | public_schema_statistics)
```

A role predictor over public schema statistics is a baseline. If it solves, the frame was in the generator surface.

### Verdict + Kill Records

```text
TYPED GENERATORS CAN LEAK SEMANTICS WITHOUT BANNED WORDS.
```

Kill records:

```text
KR-I275-1: If latent roles are predictable from uncharged schema/value statistics above threshold, void.
KR-I275-2: If the constructor uses latent generator fields or hidden semantic labels to choose packet examples, void.
KR-I275-3: If sibling tasks share hidden templates that let the constructor reuse bindings cheaply, no AFTD signal.
```

### Narrative Attack

Strongest dismissal: "The generator put a neon sign on the ID column, and the packet just charged for pointing at it."

What would survive: a typed generator where public statistics alone do not identify the frame, constructor provenance is public, and strengthened baselines still cannot move the token.

---

## I276: Prior-Art Absorption Ladder For SHEETS-0

### Single Most Dangerous Question

What boring systems get first refusal before SHEETS-0 can claim signal?

### Attack

The Boolean spec had a strong absorption ladder. SHEETS-0 needs a typed ladder, not a copy-paste. The absorbers are not merely TD-H0, L1, L2, RAG, nuisance oracle, and library learner. Typed tables require domain-specific absorbers:

| Candidate SHEETS frame | First absorber | Why |
|---|---|---|
| Join by key | relational algebra / SQL / pandas merge search | This is the canonical table operation. |
| Stable ID over display name | exact key matching / entity resolution | Identity matching is prior art. |
| Unit normalization | UCUM / dimensional analysis / unit libraries | Unit semantics are formal and reusable. |
| Date canonicalization | parser and calendar libraries | Parsing, normalization, and time zones are standard. |
| Constraint validation | database integrity / SMT / data repair | Constraints and validators are solver-native. |
| Data cleaning actions | OpenRefine/Wrangler-style scripts | Reusable cleaning recipes are existing tools. |
| Example-driven transforms | PBE/PROSE/FlashFill lineage | Spreadsheets are the home turf of PBE. |
| Schema rename transfer | schema matching / ontology alignment | Matching fields across schemas is prior art. |
| Frame reuse across tasks | MDL/library learning | Reusable macro discovery is the baseline. |

Without these, a positive SHEETS token will be demolished immediately.

### Control Or Redirect

Add SHEETS-specific terminal tokens:

```text
FRAMESEED_SHEETS0_SIGNAL
FRAMESEED_SHEETS0_ABSORBED_BY_RELATIONAL_ALGEBRA
FRAMESEED_SHEETS0_ABSORBED_BY_UNIT_SYSTEM
FRAMESEED_SHEETS0_ABSORBED_BY_ENTITY_RESOLUTION
FRAMESEED_SHEETS0_ABSORBED_BY_SCHEMA_MATCHING
FRAMESEED_SHEETS0_ABSORBED_BY_PBE
FRAMESEED_SHEETS0_ABSORBED_BY_DATA_WRANGLING
FRAMESEED_SHEETS0_ABSORBED_BY_CONSTRAINT_SOLVING
FRAMESEED_SHEETS0_ABSORBED_BY_DATA_REPAIR
FRAMESEED_SHEETS0_ABSORBED_BY_TYPED_CEGIS
FRAMESEED_SHEETS0_ABSORBED_BY_LIBRARY_LEARNING
FRAMESEED_SHEETS0_ABSORBED_BY_PARSER_PRIOR
FRAMESEED_SHEETS0_TYPED_BOOLEAN_TRAP
FRAMESEED_SHEETS0_VOID_SMUGGLED_SCHEMA
FRAMESEED_SHEETS0_NEGATIVE
```

Precedence should mirror the Boolean spec:

```text
void > parser/representation prior > exact prior-art domain absorber > L3 negative > generic baseline absorption > SHEETS0 signal
```

### Verdict + Kill Records

```text
NO SHEETS SIGNAL WITHOUT DOMAIN-SPECIFIC ABSORBERS.
```

Kill records:

```text
KR-I276-1: If W29 ships only generic Boolean-era baselines, do not interpret a positive SHEETS run.
KR-I276-2: If any domain-specific absorber matches under matched or <4x cost, emit the specific absorption token.
KR-I276-3: If multiple typed absorbers match, kill or radically reframe FrameSeed toward the strongest honest substrate.
```

### Narrative Attack

Strongest dismissal: "Every operation in the benchmark has a 20-year-old tool."

What would survive: a typed-domain run that gives those tools fair executable access and still leaves a measured, reproducible amortized frame-teaching gap.

---

## I277: Practicality Can Hide Solvedness

### Single Most Dangerous Question

Does making the task useful to ordinary users make it scientifically harder, or does it just move into solved automation territory?

### Attack

SHEETS-0 is attractive because it speaks to the Vision: cheap local automation, spreadsheet cleanup, data integration, form validation. But user-legibility can be a trap. The more the tasks resemble everyday spreadsheet mistakes, the more they resemble existing tool workflows:

```text
OpenRefine cleaning
spreadsheet formulas
Power Query
SQL joins
pandas scripts
unit libraries
deduplication tools
data validation rules
```

The adversary does not need to deny utility. They can say:

```text
Useful, yes. New intelligence principle, no.
```

The home run is not "a local agent cleans a toy sheet." That is a product demo. The home run is evidence that compact public frames create capability that ordinary scripts, PBE, data repair, and library learning cannot cheaply create.

### Control Or Redirect

Require a `Solvedness Audit` before finalizing SHEETS-0:

```text
For each task family:
  1. Name the closest existing tool or literature family.
  2. Implement or approximate its baseline.
  3. State what part of the frame remains unabsorbed.
  4. Declare whether the task is a utility demo, a science test, or both.
```

If a task cannot name a non-absorbed core after existing tools get first refusal, remove it from the signal path. It can remain as a demonstration only after the science token exists elsewhere.

### Verdict + Kill Records

```text
PRACTICAL DOES NOT MEAN PARADIGM-SHIFTING.
```

Kill records:

```text
KR-I277-1: If the best explanation is "this is useful data cleaning," demote to demo and do not claim FrameSeed signal.
KR-I277-2: If ordinary local tools can solve the benchmark with saved scripts or recipes, emit data-wrangling or library-learning absorption.
KR-I277-3: If the benchmark is selected because it markets well rather than because it isolates an unabsorbed principle, reject the spec.
```

### Narrative Attack

Strongest dismissal: "You built a nice spreadsheet assistant, not a new route to cheap intelligence."

What would survive: a measurable principle of reusable, inspectable frame transmission that happens to power a spreadsheet demo, not a spreadsheet demo standing in for the principle.

---

## I278: The Hidden Hard Problem Is Goal Semantics, Not Operations

### Single Most Dangerous Question

What is the part of SHEETS-0 that existing operations do not already solve?

### Attack

Joins, conversions, constraints, cleaning, and matching are operations. The harder problem is deciding which operation is valid for the user's goal under ambiguous evidence.

Examples:

```text
Two columns both look like IDs; one is an account ID, one is a shipment ID.
Two unit columns share dimensions; one is package weight, one is payload weight.
Duplicate names may indicate duplicate records or legitimate family members.
Missing keys may mean invalid data or a one-to-many relationship.
A constraint violation may be a data error or a policy exception.
A row order may be meaningful in a time series but meaningless in a lookup table.
```

If SHEETS-0 avoids these cases, it is too easy. If it includes them but the generator labels the intended goal, the goal is smuggled. If it includes them and leaves the goal ambiguous, accuracy may become ill-defined.

This is the typed version of intervention semantics in Boolean FRAMESEED-0: the meaning of the action must be operationally defined without leaking the answer.

### Control Or Redirect

Define task goals as verifier obligations, not prose:

```text
Goal = finite set of public and hidden obligations:
  preservation obligations
  transformation obligations
  rejection obligations
  uncertainty/abstention obligations
  repair obligations
  action-safety obligations
```

Then test whether the frame packet teaches how to generate or apply the right obligation structure, not merely how to run a named operation.

Add baselines:

```text
G0: operation enumerator with verifier search
G1: goal-conditioned CEGIS
G2: active goal-disambiguation learner
G3: library learner over obligation templates
G4: abstention-aware validator
```

If these match, absorb.

### Verdict + Kill Records

```text
THE ONLY LIVE CORE MAY BE GOAL/OBLIGATION DISCOVERY.
```

Kill records:

```text
KR-I278-1: If SHEETS-0 tests execution of known operations but not goal semantics, absorb into prior art.
KR-I278-2: If hidden goal labels are exposed to the constructor or packet, void.
KR-I278-3: If goal ambiguity makes hidden scoring subjective, redesign before hidden run.
```

### Narrative Attack

Strongest dismissal: "The system knew which operation to apply; applying it was routine."

What would survive: a packet that teaches a cheap learner to derive and verify the right obligation structure under ambiguity, while operation-search baselines get fair access and still cannot cheaply match.

---

## I279: Frame Composition Can Be Absorbed Too

### Single Most Dangerous Question

If SHEETS-0 composes key-join plus unit-normalization plus constraint checking, does composition rescue it from prior art?

### Attack

B34 says typed replication becomes more serious when frames compose:

```text
stable-ID matching + unit normalization + constraint validation
```

But composition is also prior art:

```text
ETL pipelines
relational query plans
dataflow systems
wrangling scripts
program synthesis with library macros
workflow automation
constraint pipelines
```

A pipeline like:

```text
canonicalize IDs -> join -> normalize units -> group -> validate -> act
```

is a standard data transformation program. Calling each step a frame does not make the pipeline a theory of intelligence. A library learner or PBE system can invent reusable subroutines and compose them under MDL.

Composition only helps if FrameSeed shows subadditive, repairable, audited composition that prior-art pipeline/library learners cannot match.

### Control Or Redirect

For every composed SHEETS task, report:

```text
cost(F_join)
cost(F_unit)
cost(F_constraint)
cost(F_composed)
cost(bindings per component)
interference failures
repair packet size after failure
library learner composed macro cost
PBE/CEGIS pipeline cost
OpenRefine-like action history cost
```

Require:

```text
cost(F_composed) < cost(F_join) + cost(F_unit) + cost(F_constraint)
and repairs preserve earlier verified behavior
and library/pipeline baselines fail or pay >=4x.
```

### Verdict + Kill Records

```text
COMPOSITION IS NOT A RESCUE UNLESS IT BEATS PIPELINE AND LIBRARY BASELINES.
```

Kill records:

```text
KR-I279-1: If composed frames are just a saved ETL/query/wrangling pipeline, emit data-wrangling, CEGIS, or library-learning absorption.
KR-I279-2: If frame composition breaks earlier behavior and requires broad retuning, no improvability claim.
KR-I279-3: If repair packets are full replacement programs, absorb into program synthesis.
```

### Narrative Attack

Strongest dismissal: "You built a reusable data pipeline."

What would survive: composable packets with subadditive cost and local repair that remain cheaper than the strongest pipeline, synthesis, and library-learning baselines.

---

## I280: Final Pre-Design Verdict

### Single Most Dangerous Question

What must W29 change before SHEETS-0 is worth finalizing?

### Attack Synthesis

SHEETS-0 is the right battlefield only if it is designed as an absorption trap, not as a nicer demonstration. The named frames are all close to existing technical objects:

```text
join by key              -> relational algebra
stable ID matching       -> entity resolution / record linkage
unit normalization       -> dimensional analysis / unit libraries
constraint validation    -> database constraints / solvers / data repair
schema rename transfer   -> schema matching
examples and transforms  -> PBE / PROSE / data wrangling
frame reuse              -> MDL/library learning
frame composition        -> ETL/dataflow/pipeline synthesis
```

The Boolean trap was "finite truth tables are too easy." The SHEETS trap is "typed semantics are preinstalled or already solved."

The honest live claim is narrower and harder:

```text
Can a compact packet teach a bounded cheap learner a reusable typed obligation interface that reduces future schema-binding, transformation-choice, verification, and repair cost across hidden spreadsheet families, after relational, unit, entity-resolution, PBE, data-wrangling, constraint-solving, schema-matching, CEGIS, RAG, and library-learning baselines receive equal executable information?
```

If W29 does not want that harder claim, SHEETS-0 should be demoted to a utility demo or killed as a science test.

### Required W29 Modifications

W29 should add:

1. `SHEETS-0 Typed Representation-Noncontainment Certificate`
2. `Typed Parser And Human-Labor Ledger`
3. `Frame/Binding/Program Cost Split`
4. `Domain-Specific Absorption Tokens`
5. `Relational Algebra Baseline`
6. `Unit-System Baseline`
7. `Entity-Resolution Baseline`
8. `Schema-Matching Baseline`
9. `PBE/PROSE-Style Baseline`
10. `OpenRefine/Wrangling-Script Baseline`
11. `Constraint-Solver/Data-Repair Baseline`
12. `Typed CEGIS And Typed Library-Learning Baselines`
13. `Typed Generator Leakage Audit`
14. `Goal/Obligation Semantics Contract`
15. `Composition And Local-Repair Gate`

### Minimum Signal Conditions

```text
FRAMESEED_SHEETS0_SIGNAL
```

may be emitted only if:

1. typed smuggling audit passes;
2. parser/type-system representation-prior audit passes;
3. L3 reaches hidden target and sibling thresholds;
4. frame/binding/program/human-labor costs are separated;
5. AFTD_all_in beats independent typed teaching sets;
6. domain-specific absorbers fail or pay >=4x;
7. generic Boolean-era absorbers fail or pay >=4x;
8. packet-erasure drops sibling transfer;
9. role/name/schema/unit/row permutations preserve the token;
10. composed frames show subadditive cost or local repair;
11. no ordinary local tool/script baseline matches the result;
12. claim ceiling remains "controlled evidence for typed amortized frame-teaching separation."

### Final Kill Records

```text
KR-FINAL-B37-1: If SHEETS-0 gives L3 typed semantics denied to baselines, void.
KR-FINAL-B37-2: If parsers, unit systems, key discovery, schema matching, or constraint languages already contain the frame at low cost, emit representation-prior or parser-prior absorption.
KR-FINAL-B37-3: If relational algebra, unit libraries, entity-resolution, PBE/data-wrangling, constraint solving, typed CEGIS, or library learning matches under matched or <4x cost, emit the corresponding absorption token.
KR-FINAL-B37-4: If AFTD ignores task-specific binding and program costs, no signal token is valid.
KR-FINAL-B37-5: If practical local automation is achieved only by using ordinary scripts, formulas, SQL, or OpenRefine-like recipes, demote to demo.
KR-FINAL-B37-6: If the spec cannot define hidden goal/obligation semantics without leaking the answer, redesign before hidden run.
```

### Final Recommendation

```text
CONDITIONAL GO, BUT THE PRE-DESIGN MUST GET MUCH HARDER.
```

Do not finalize SHEETS-0 as "typed joins and units." That will be absorbed. The home-run version tests whether compact, inspectable packets transmit reusable typed obligation structure that survives the strongest boring table tools and program-synthesis baselines.

If W29 cannot make that executable, the right answer is not to soften the baselines. The right answer is to report:

```text
FRAMESEED_SHEETS0_PRE_DESIGN_ABSORBED_BY_TYPED_PRIOR_ART
```

and redirect toward the deeper problem B33/B34 already named:

```text
Self-discovered transformation grammars in typed practical domains.
```

### Final Narrative Attack

Strongest "that's obvious" dismissal:

```text
Useful spreadsheet work is made of joins, unit conversions, constraints, deduping, and saved scripts. You put them in a packet.
```

Strongest "that's trivial" dismissal:

```text
The typed benchmark was still a finite generator with a small DSL, and existing table tools found the DSL.
```

What the result needs to be:

```text
A hostile reviewer should be able to run SQL/search, unit libraries, entity resolution, schema matching, PBE, OpenRefine-like action histories, constraint solvers, typed CEGIS, retrieval, and MDL library learning against the same hidden tasks and still fail to reassign the token away from signal.
```

Until then, SHEETS-0 is not the home run. It is only the next honest filter.