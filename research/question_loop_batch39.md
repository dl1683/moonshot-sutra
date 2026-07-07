# Q-Loop Batch 39: SHEETS-0 Measurement Oversight And Adversarial Prep

**Date:** 2026-07-07  
**Role:** Question-Loop worker  
**Iterations:** I295-I308  
**Status:** measurement oversight; adversarial milestone-gate prep; no hidden SHEETS-0 result found in current checkout.

---

## Grounding

Read for this batch:

1. `research/dual_loop_supervisor_checkin_30.md`
2. `code/frameseed_sheets0_harness.py`
3. `research/frameseed_sheets_0_spec.md`
4. `research/question_loop_batch38.md`
5. `research/VISION.md`

Additional checkout evidence inspected:

- `research/work_loop_batch30.md`
- `code/test_frameseed_sheets0_harness.py`
- `experiments/frameseed_sheets0_b30_public_audit.json`
- `experiments/frameseed_sheets0_b31_reaudit.json`

Binding invariants:

1. Swing for the home run.
2. The loop only stops on a won-over adversary.

## Current Measurement State

There is no SHEETS-0 hidden measurement artifact in the current checkout.

The only B31-named artifact found is:

```text
experiments/frameseed_sheets0_b31_reaudit.json
```

It is a repeated public preimplementation audit, not a hidden HFA measurement:

```text
name: frameseed_sheets0_preimplementation_audit
passed: true
no_performance_runs: true
hidden_hfa_reported: false
hidden_results_opened: false
public_seed: FRAMESEED_SHEETS0_B30_PUBLIC_SEED
```

Therefore B39 cannot monitor hidden scores yet. It can monitor the measurement
boundary and prepare the adversarial gate. The present result is:

```text
SHEETS-0 HIDDEN MEASUREMENT NOT YET PRESENT.
THE B31-NAMED ARTIFACT IS A RE-AUDIT, NOT A MEASUREMENT.
NO SIGNAL, ABSORPTION, OR NEGATIVE TOKEN IS INTERPRETABLE FROM IT.
```

## Summary Verdict

```text
CURRENT SHEETS-0 IS AUDIT-CLEAN BUT MEASUREMENT-INCOMPLETE.

THE SPEC DEMANDS HIDDEN WORLDS, HIDDEN QUERY MIX, TARGET AND SIBLING HFA,
AFTD, ABLATIONS, ROLE/SCHEMA PERMUTATIONS, NATIVE DOMAIN BASELINES,
PARSER/TYPE PARITY, 10,000-WORLD LEAKAGE AUDITS, AND ALL-IN COST RATIOS.

THE IMPLEMENTATION CURRENTLY PROVIDES A PUBLIC AUDIT HARNESS:
GENERATOR SCAFFOLDING, PACKET SERIALIZATION, PROVENANCE, COST CATEGORIES,
BASELINE ROSTER PARITY, MARGINAL MI LEAKAGE, ENUMERABILITY METRICS, AND
TOKEN PRECEDENCE CONTROLS.

THAT IS NOT ENOUGH FOR A MILESTONE SIGNAL.
```

B38's strongest absorber was typed MDL library learning / PBE-CEGIS pipeline
synthesis. B39 sharpens the attack: before that absorber even runs, the
measurement may already be confounded by artifact identity, hidden-run absence,
declarative baselines, cost-ledger ambiguity, public operation semantics,
role-statistic leakage, and an audit harness that checks roster parity rather
than native competence.

If a hidden result appears later, the fresh-eyes reviewer should ask first:

```text
Is this an actual frozen hidden measurement, or another audit artifact wearing
a measurement name?
```

If it is an actual measurement, the reviewer should ask second:

```text
Did the hostile native baselines receive the same typed substrate and still fail
or pay >=4x under all-in accounting?
```

If either answer is no, no home-run claim is alive.

---

## I295: Artifact Census Is The First Measurement Audit

### Attack

The directive says W-Loop B31 should run a hidden SHEETS-0 measurement. The
checkout does not contain that measurement. It contains a B31-named re-audit
whose own fields say:

```text
no_performance_runs = true
hidden_hfa_reported = false
hidden_results_opened = false
```

This matters because a milestone review can be poisoned by file naming. A file
named `b31_reaudit` is not a hidden run. A public audit passing again is not
evidence that L3 reached typed HFA, that baselines failed, or that any token
should move.

### Measurement Confound

The milestone could accidentally treat "B31 happened" as "SHEETS-0 was
measured." That would be a process failure, not a negative result.

### Gate Criterion

Before any adversarial review starts, require a measurement artifact with:

```text
hidden_hfa_reported = true
hidden_results_opened = true
terminal_token present
hidden_seed_hash present
measurement_version distinct from preimplementation audit
target and sibling HFA tables present
all baseline HFA and all-in budget ratios present
```

If absent, the gate verdict is:

```text
NO HIDDEN SHEETS-0 MEASUREMENT PRESENT.
```

### Verdict

```text
THE CURRENT CHECKOUT HAS AUDIT EVIDENCE, NOT MEASUREMENT EVIDENCE.
```

---

## I296: The Harness Is Audit-Only Unless A Hidden Runner Exists

### Attack

`code/frameseed_sheets0_harness.py` declares itself a public audit harness. Its
top-level `run_preimplementation_audit()` explicitly emits:

```text
no_performance_runs = true
hidden_hfa_reported = false
```

It has token precedence controls, but no hidden-world scorer that evaluates:

```text
m in {4,16,64,256}
64 hidden worlds per m
10 role/schema permutations per world
256 hidden queries per world
3 sibling tasks per target
H1-H6 hidden families
ablation HFA drops
baseline HFA ratios
```

### Measurement Confound

If B31 opens hidden using an unreviewed new runner, the hidden runner itself
becomes the highest-risk artifact. If B31 does not open hidden, no measurement
exists. There is no middle path where the public audit harness alone supports
signal.

### Gate Criterion

The fresh-eyes packet must include the hidden runner source, its hash before
hidden opening, and an explicit diff showing no scorer, constructor, baseline,
timeout, token-policy, parser, or unit-registry mutation after hidden opening.

### Verdict

```text
A TOKEN ASSIGNER WITHOUT A HIDDEN SCORER IS NOT A MEASUREMENT.
```

---

## I297: Baseline Roster Parity Is Not Baseline Execution

### Attack

The harness has a broad baseline list:

```text
relational_algebra
unit_system
exact_key_matching
entity_resolution
schema_matching
pbe_prose
data_wrangling
constraint_solver
data_repair
typed_cegis_exact
typed_cegis_beam
typed_mdl_library
...
```

But the current audit only checks that these names exist and receive the same
packet bytes. The manifest versions are adapter labels. The timeouts are all
zero. There is no native relational search, unit library, linkage learner,
schema matcher, PROSE-style synthesizer, wrangling script learner, constraint
solver, data repair engine, exact typed CEGIS, or MDL library learner producing
HFA and cost ratios.

### Measurement Confound

A measurement can look fair because every baseline is listed, while no boring
baseline is competent enough to move the token. The adversary will call that a
strawman even if packet-byte parity passes.

### Gate Criterion

For each domain absorber, require:

```text
native hypothesis language
native objective
native search budget
adapter to canonical typed output
all-in bits charged
hidden HFA by m and family
target plus sibling transfer result
absorber ratio versus L3_full
```

No native execution means no signal. At most it means:

```text
MEASUREMENT BLOCKED BY UNEXECUTED DOMAIN BASELINES.
```

### Verdict

```text
BASELINE NAMES DO NOT WIN OVER A HOSTILE REVIEWER.
```

---

## I298: The Cost Ledger Has A Double-Counting Ambiguity

### Attack

The B31 re-audit reports:

```text
packet_bits = 20800
frame_bits = 5816
binding_bits = 2360
examples_bits = 4048
verifier_bits = 3128
program_bits = 2328
total_bits = 38480
```

The category sum is bounded by packet bits, but `total_bits` includes
`packet_bits` plus the categories. That makes the reported total larger than
the packet serialization by construction.

This is not a harmless cosmetic issue. SHEETS-0's decisive metrics are
AFTD_all_in, baseline ratios, binding dominance, program dominance, and the
4x absorption rule. If the same bytes are counted both as packet bits and as
category bits, the all-in denominator can move tokens incorrectly.

### Measurement Confound

Double counting can either over-penalize L3 or create ambiguous ratios that a
reviewer cannot reproduce. Either way, the measurement cannot be adversarially
accepted until the ledger semantics are unambiguous.

### Gate Criterion

The measurement report must define exactly one canonical all-in formula:

```text
all_in_bits = frame_bits + binding_bits + examples_bits + verifier_bits
              + program_bits + final_program_bits + learned_library_bits
              + residual_sibling_teaching_bits + parser_bits
              + human_labor_bits + baseline_adapter_bits
              + failed_query_bits
```

Then separately report:

```text
serialized_packet_bits
classified_packet_bits
unclassified_packet_bits
all_in_bits
```

If `serialized_packet_bits` is the source of truth, categories must partition
it. If categories are the source of truth, `packet_bits` cannot be summed again.

### Verdict

```text
ALL-IN COST MUST BE RECOMPUTABLE WITHOUT DOUBLE COUNTING.
```

---

## I299: Parser And Human Labor Are Declared But Not Charged

### Attack

The parser/human ledger declares:

```text
opaque_table_schema
rational_parser
date_parser_stub
unit_registry
typed_action_api
frame_patch_templates
verifier_obligation_templates
binding_cost_rules
```

It also reports:

```text
charged_parser_bits = 0
charged_human_bits = 0
```

The spec allows parser and human-labor costs to be charged or declared outside
the claim, but a hostile reviewer will not let this disappear. The typed action
API, unit registry, obligation grammar, packet templates, and public operation
grammar are a large part of the intelligence geometry. If all systems receive
them free, the claim is substrate-free only. If only L3 benefits from how they
are shaped, void for substrate asymmetry.

### Measurement Confound

The measured benefit may come from the benchmark designer's hand-authored
typed substrate rather than from FrameSeed transfer.

### Gate Criterion

Report two regimes:

```text
substrate-free: all systems get the public typed substrate free
substrate-charged: parser, registry, API, verifier grammar, and adapters are
                   amortized and charged
```

Signal must be argued only in the substrate-free regime unless the
substrate-charged result also survives. The report must explicitly say:

```text
No claim is made that the hand-authored typed substrate was learned.
```

### Verdict

```text
ZERO-CHARGED HUMAN GEOMETRY CAN EXPLAIN THE RESULT.
```

---

## I300: Operation Requests May Hand Over The Goal

### Attack

The public schema and transcript expose operation families:

```text
canonical_join
aggregate_by_key
validate_and_apply
abstain_on_ambiguous_binding
```

The packet entries also use executable-looking semantic operators:

```text
same_entity_by_canonical_equality_after_binding
normalize_quantity_before_math
charged_same_type_binding_search
join_normalize_guard
```

B38 warned that `operation_request` may choose the frame. B39's stronger point:
the current public artifacts already hand the learner a typed task ontology. If
the hidden query says "aggregate_by_key", and the packet says
"normalize_quantity_before_math", then the remaining problem is binding and
program execution.

### Measurement Confound

The result may measure "execute the named operation under typed API" rather
than "learn reusable obligation structure."

### Gate Criterion

Every hidden result must be stratified by request regime:

```text
operation-given
obligation-given
goal-ambiguous
```

A signal in operation-given mode can only claim execution and binding transfer.
It cannot claim goal or obligation discovery. If goal-conditioned CEGIS or
operation-enumerator baselines match, emit the appropriate absorber.

### Verdict

```text
IF THE REQUEST NAMES THE OPERATION, THE FRAME MAY ALREADY BE SELECTED.
```

---

## I301: Cross-Table Role Statistics May Identify The Key

### Attack

Same-type decoys are present, but the generator creates an event foreign-key
column by copying entity stable keys. That creates a powerful public statistic:
cross-table overlap. The exact-key and entity-resolution baselines in the spec
are supposed to exploit:

```text
uniqueness
stability
cross-table agreement
overlap
blocking
transitive consistency
```

The current leakage audit does not test a role predictor using cross-table
overlap, uniqueness, duplicate structure, unit/value compatibility, constraint
localization, or operation request features. It tests marginal normalized MI
over simpler features such as index bucket, name prefix, missingness, and
value shape.

### Measurement Confound

The hidden result can be absorbed by exact-key matching or schema matching even
when marginal MI is low.

### Gate Criterion

Before hidden opening, run predictor baselines:

```text
role_predictor_type_only
role_predictor_stats_only
role_predictor_overlap_uniqueness
role_predictor_stats_plus_public_examples
role_predictor_stats_plus_operation_request
binding_predictor_all_public
```

If these recover target bindings cheaply, emit representation-prior,
exact-key, entity-resolution, schema-matching, or schema-binding absorption.

### Verdict

```text
SAME-TYPE DECOYS DO NOT DEFEAT RELATIONAL ROLE PREDICTORS.
```

---

## I302: The Non-Boolean Floor Is Currently A Schema Constant

### Attack

The harness reports:

```text
non_boolean_output_fraction = 4 / 6
```

This is derived from output forms listed in the public schema, not from the
actual hidden query mix. The spec requires at least 50% of hidden queries to
require non-Boolean typed outputs. A schema with four non-Boolean output forms
does not prove the hidden queries used them.

### Measurement Confound

The task can pass the audit while the hidden measurement is dominated by
accept/reject or small finite selection problems. That reopens the typed
Boolean-trap.

### Gate Criterion

The hidden measurement must report:

```text
hidden_query_count_by_output_form
hidden_query_count_by_family
hidden_non_boolean_fraction_by_m
hidden_non_boolean_fraction_by_family
HFA by output form
Boolean-core decomposition bits
```

If non-Boolean typed outputs reduce to role/operator selection plus public
deterministic rendering, emit typed Boolean-trap or synthesis absorption.

### Verdict

```text
OUTPUT FORMS IN A SCHEMA ARE NOT A HIDDEN QUERY MIX.
```

---

## I303: Siblings Are Generated, Not Yet AFTD-Tested

### Attack

The harness creates three sibling worlds, but they are not the spec's required
sibling tasks:

```text
s_key
s_unit
s_composed
```

The current siblings are generated by the same generic world generator with a
namespace derived from the target world id. The audit reports `sibling_count =
3`, but it does not report:

```text
TD_H0(sibling)
TD_after(F_frame,sibling)
count_reduced_siblings
AFTD
AFTD_all_in
sibling HFA
sibling diversity
template independence
```

### Measurement Confound

Sibling transfer can be faked in two opposite ways:

1. Siblings share too much generator template structure, so MDL library learning
   absorbs.
2. Siblings are nominally present but not scored for AFTD, so the packet only
   solves a target task.

### Gate Criterion

The hidden result must include a sibling table:

```text
sibling_id
sibling_family
schema/template distance from target
HFA_full
TD_H0
TD_after_frame
binding_bits
program_bits
reduced = true/false
```

If `count_reduced_siblings < 3`, no signal. If a generator-template learner
predicts sibling bindings/programs under <4x, emit library-learning absorption.

### Verdict

```text
THREE SIBLING IDS ARE NOT THREE REDUCED SIBLINGS.
```

---

## I304: The Leakage Audit Is Narrower Than The Spec Claims

### Attack

The spec requires typed generator leakage audits over:

```text
names, indices, row order, value formats, ID formats, unit symbols, date
formats, row counts, duplicate patterns, missingness, foreign-key overlap,
constraint violation locations, action labels, serialized representations,
schema IDs, packet order, sibling templates
```

The current implementation measures a smaller set:

```text
index_bucket
name_prefix
missingness = none
value_shape
sibling_index_target_signature_nmi
```

The B30/B31 audit uses `sample_count = 1000`, while the spec requires generator
MI over at least 10,000 dry-run worlds before hidden performance.

### Measurement Confound

The audit can pass with `worst_metric = 0.014158...` while the actual public
features used by competent baselines still predict roles or frame family.

### Gate Criterion

Do not accept a hidden signal unless the leakage report includes:

```text
>= 10000 dry-run worlds
conditional/predictive leakage models, not only marginal MI
foreign-key overlap features
unit/value compatibility features
duplicate and missingness features
constraint violation localization features
operation-request features
packet-order features
sibling-template predictors
```

If a predictor is useful, classify it as a baseline absorber, not just an audit
warning.

### Verdict

```text
LOW MARGINAL MI IS NOT GENERATOR BLINDNESS.
```

---

## I305: Enumerability Metrics Are Not CEGIS Cost

### Attack

The audit reports for one `m = 16` audit world:

```text
N_join_candidates = 132
N_unit_transform_candidates = 66
N_schema_bindings = 792
N_constraint_sets = 4096
N_action_policies = 24
public_example_version_space = 209088
minimum_distinguishing_counterexamples = 18
```

These are useful descriptive metrics. They are not a CEGIS run, a library
learning run, an active query run, or a proof that synthesis costs >=4x.

### Measurement Confound

The report can imply combinatorial hardness while the actual typed search is
small enough for a competent PBE/CEGIS/library solver to absorb. Conversely, it
can overstate hardness because typed pruning and public examples may collapse
the version space quickly.

### Gate Criterion

For every m and hidden family, report:

```text
exact_cegis_feasible = true/false
exact_cegis_best_bits
beam_cegis_best_bits
mdl_library_bits
failed_query_bits
counterexamples_to_isolate_target
best_program_ast_bits
best_macro_library_bits
HFA of each synthesis baseline
```

If exact search is infeasible, the beam/approximation quality must be stated as
an assumption, not narrated as failure of CEGIS.

### Verdict

```text
VERSION-SPACE SIZE IS NOT AN ABSORBER RESULT.
```

---

## I306: Domain Absorber Before Negative Can Hide L3 Failure Unless Reported

### Attack

The token precedence lets domain-specific absorption fire before negative. The
golden control explicitly checks relational absorption even when L3 threshold
is false.

This may be correct token policy: if a boring domain system solves the task, it
is the more informative explanation. But a milestone reviewer still needs to
know whether L3 itself failed.

### Measurement Confound

A public token such as `ABSORBED_BY_RELATIONAL_ALGEBRA` can hide two different
worlds:

```text
L3 succeeded, but relational algebra matched under <4x.
L3 failed, and relational algebra succeeded.
```

Both are not signal, but they imply different project diagnoses.

### Gate Criterion

Every non-signal token must report:

```text
l3_full_threshold_passed
l3_mean_hfa
best_absorber_name
best_absorber_hfa
best_absorber_all_in_ratio
negative_failures
absorption_failures
void_failures
```

Do not let a single token replace the diagnostic ledger.

### Verdict

```text
THE TOKEN MUST BE UNIQUE; THE DIAGNOSIS MUST NOT BE COLLAPSED.
```

---

## I307: Manifest Freeze Must Bind Actual Files, Not Declarations

### Attack

The manifest audit checks booleans:

```text
frozen_before_hidden = true
hidden_results_opened = false
post_hidden_changes = []
```

But the default manifest does not itself prove file hashes, git status, hidden
runner hash, baseline implementation hashes, scorer hash, or post-hidden diff.
A function can recreate a clean default manifest after the fact. That is fine
for a public audit scaffold. It is not enough for a hidden measurement.

### Measurement Confound

Post-hidden changes can be hidden behind regenerated manifest fields or
untracked artifacts. The current `b31_reaudit` is untracked in this checkout,
which is not a problem by itself, but it shows why the review must treat git
state and artifact provenance as first-class evidence.

### Gate Criterion

The hidden measurement packet must include:

```text
git commit hash before hidden opening
git status before hidden opening
hashes of spec, harness, hidden runner, tests, baseline adapters, scorer
manifest hash used inside hidden seed derivation
hidden seed hash
post-hidden git status
post-hidden diff summary
artifact hash
rerun command
```

If any code/scorer/baseline/timeout/parser/token-policy file changed after
hidden opening under the same hidden seed, void.

### Verdict

```text
FREEZE IS A REPRODUCIBLE ARTIFACT CHAIN, NOT A BOOLEAN FIELD.
```

---

## I308: Fresh-Eyes Milestone Gate Criteria

### Attack Synthesis

The adversary should not be asked, "Does this look promising?" The adversary
should be asked to move the token if any of the following are true:

```text
the artifact is not a hidden measurement
the hidden runner was not frozen
the hidden query mix does not satisfy the spec
the typed-output floor is schema-declared rather than query-measured
the cost ledger cannot reproduce all-in ratios
the public substrate or operation request supplies the hard part
role/statistic predictors recover bindings cheaply
native baselines were not run
PBE/CEGIS/library baselines match under <4x
sibling AFTD is absent or template-learned
ablation drops are absent
role/schema permutations are absent or unstable
composition is a saved pipeline
local repair is full program replacement
claim ceiling is exceeded
```

### Milestone Review Packet

The gate packet must contain:

```text
1. Artifact census
   - exact measurement file
   - exact terminal token
   - hidden seed hash
   - code/spec/runner/baseline/scorer hashes

2. Hidden measurement tables
   - HFA by system, m, hidden family, output form, target/sibling
   - hidden query counts by family and output form
   - ablation HFA and drops
   - role/schema permutation stability
   - randomized-label and oracle controls

3. Cost tables
   - serialized packet bits
   - frame, binding, example, verifier, program, parser, human, library bits
   - AFTD_frame_only and AFTD_all_in
   - baseline all-in ratios and 4x decisions

4. Absorber tables
   - relational algebra
   - unit system
   - exact-key/entity resolution
   - schema matching/binding
   - PBE/PROSE
   - wrangling/script
   - constraint/data repair
   - typed CEGIS exact/beam
   - typed MDL library
   - active learning, RAG, nuisance oracle, TD-H0

5. Leakage and prior tables
   - 10000-world leakage audit
   - predictive role/binding/family models
   - parser/type-system noncontainment search
   - substrate-free and substrate-charged accounting
```

### Decision Rules

Emit or accept `FRAMESEED_SHEETS0_SIGNAL` only if:

```text
actual hidden measurement exists
all spec thresholds pass
L3 passes every m/family target and >=3 reduced siblings
mean L3 HFA >= 0.97
query-measured non-Boolean floor passes
AFTD and AFTD_all_in pass under non-double-counted costs
every native domain absorber fails or pays >=4x
PBE/CEGIS/library fails or pays >=4x
role/statistic predictors do not cheaply recover bindings
parser/representation noncontainment is bounded and explicit
ablation drops and randomized-label controls pass
role/schema permutations preserve token
composition beats pipeline/library synthesis on cost and repair
claim ceiling is honored
```

Emit or accept absorption if:

```text
any native domain baseline reaches threshold at matched or <4x all-in budget
typed CEGIS or MDL library reaches threshold at matched or <4x all-in budget
role predictors recover bindings cheaply
operation/obligation baselines match once the public request supplies the goal
```

Emit or accept void if:

```text
hidden seed opened before freeze
post-hidden scorer/constructor/baseline/parser/token changes occurred
L3 receives typed semantics denied to baselines
hidden goals are subjective or changed after opening
leakage predictors exploit uncounted generator statistics
artifact provenance is unreproducible
```

Emit or accept negative if:

```text
L3 misses hidden thresholds and no stronger void or absorber applies
```

### Final B39 Verdict

```text
DO NOT LET THE MILESTONE GATE TREAT THE CURRENT B31 RE-AUDIT AS A HIDDEN
MEASUREMENT.

DO NOT LET A FUTURE HIDDEN MEASUREMENT CLAIM SIGNAL UNLESS IT BEATS NATIVE
DOMAIN BASELINES AND THE TYPED MDL/PBE-CEGIS LIBRARY ABSORBER UNDER CLEAN
ALL-IN ACCOUNTING.

THE ADVERSARY IS WON OVER ONLY BY A FROZEN, REPRODUCIBLE HIDDEN RESULT WITH
QUERY-MEASURED TYPED OUTPUTS, REDUCED SIBLINGS, AFTD_ALL_IN, ABLATIONS,
ROLE/PERMUTATION STABILITY, PREDICTOR-BASED LEAKAGE CLEARANCE, AND REAL
ABSORBER FAILURES.
```

SHEETS-0 still points at the right home-run question: can cheap local systems
acquire reusable, inspectable typed automation structure instead of brute-force
program synthesis? The current checkout has not answered it. It has built the
audit shell. The next honest step is either a real hidden measurement with
native absorbers or an explicit blocked ledger saying the baselines are not yet
real enough to open hidden.