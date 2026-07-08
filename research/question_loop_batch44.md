# Q-Loop Batch 44: WGD Harness Oversight

**Date:** 2026-07-07  
**Role:** Question-Loop worker  
**Iterations:** I365-I378  
**Status:** WGD-0 harness oversight against W-Loop B36 directive.

Two invariants held fixed:

1. Swing for the home run.
2. The loop only stops on a won-over adversary.

## Grounding

Read the requested context in the current checkout:

1. `research/dual_loop_supervisor_checkin_34.md`
2. `research/wgd_0_precommit_spec.md`
3. `research/question_loop_batch43.md`
4. `research/VISION.md`

The supervisor directive is binding:

```text
W-Loop B36 builds the WGD-0 audit harness.
Q-Loop B44 monitors W36's implementation.
Main risks: weakened native absorbers, indecisive geometry erasure, and dishonest
cost accounting.
```

## Current Checkout Evidence

The current checkout does not contain a W36 WGD harness implementation artifact.

Evidence from the live tree:

```text
git log -n 1:
  a9b3b72 supervisor check-in #34: WGD-0 spec hardened, harness implementation next

rg --files -g "*wgd*" -g "*WGD*" -g "*batch36*" -g "*harness*":
  research/wgd_0_precommit_spec.md
  research/question_loop_batch36.md
  code/test_frameseed_sheets0_harness.py
  code/frameseed_sheets0_harness.py
  code/frameseed0_harness.py
  code/benchmark_harness.py
  code/test_benchmark_harness.py
  code/test_frameseed0_harness.py

git status --short --untracked-files=all:
  clean
```

There is no `research/work_loop_batch36.md`, no `code/wgd0_harness.py`, no
`code/test_wgd0_harness.py`, no WGD manifest, no WGD absorber handles, no WGD
cost extractor, and no WGD smoke audit output in the current checkout.

This means the oversight answer is not "the harness is weak." It is sharper:

```text
There is no W36 WGD harness implementation present to approve.
```

If a hidden seed were opened from this checkout, the maximum defensible token
would be:

```text
WGD_INCONCLUSIVE_BASELINES_NOT_NATIVE
```

If any result narrative claimed signal anyway, higher-precedence voids would
also become live:

```text
WGD_VOID_BASELINE_PARITY_FAILURE
WGD_VOID_COST_LEDGER_FAILURE
WGD_VOID_SUBSTRATE_ASYMMETRY
```

No hidden result exists here, so B44 assigns no measurement terminal token. B44
assigns an implementation oversight verdict only.

## Executive Verdict

W36 is not in the current checkout. Therefore the three monitoring questions
receive hard answers:

| Question | Current answer | Why |
|---|---|---|
| Are the native absorbers genuinely trying? | No executable evidence. | No WGD native absorber implementations, capability witnesses, status labels, budget curves, or parity tests exist. |
| Is the geometry-erasure ablation decisive? | No executable evidence. | No frozen `G_hat` IR, component-level erasure harness, scoring comparison, or erasure smoke output exists. |
| Is the cost ledger honest? | No ledger exists. | No deterministic cost extractor, human-substrate ledger, query ledger, adapter ledger, or recomputation test exists. |

B43 warned that `NATIVE_ABSORBER_THEATER` was the most likely failure mode.
The current checkout has not reached theater. It has not reached implementation.

The adversarial stance for W36 is now:

```text
Do not let the next worker count a missing harness as progress, and do not let
a newly added harness pass merely because it has the right filenames. It must
make the boring explanations dangerous before WGD is allowed to look good.
```

## I365: Attack Checkout Reality

### Attack

The first oversight move is not philosophical. It is filesystem reality. The
supervisor says W36 should build the WGD-0 audit harness. The current checkout
does not contain the W36 report or any WGD harness code.

This matters because WGD-0's spec makes implementation state part of the
scientific object. A native absorber is not a name in a document. A geometry
erasure is not a table in a spec. A cost ledger is not a category list. Each has
to be runnable before hidden opening.

### What This Refuses

This refuses a soft interpretation where B36 is "conceptually next" and B44 can
pretend to monitor intent. The adversary cannot be won over by intent. The loop
only stops on artifacts.

### Implementation Demand

Before any later Q-loop can approve W36, the tree needs at minimum:

```text
research/work_loop_batch36.md
code/wgd0_harness.py
code/test_wgd0_harness.py
experiments/wgd0_public_smoke_audit.json or equivalent
frozen WGD manifest and report skeleton
```

### Verdict

```text
ABSENCE IS NOT A WEAK PASS. ABSENCE IS A BLOCKED HARNESS.
```

## I366: Attack Harness Integrity As More Than File Presence

### Attack

Even when W36 appears, the first trap will be a harness-shaped file that audits
documents but cannot execute the WGD scientific contract. The FrameSeed harness
pattern is useful only if the WGD harness inherits the hard parts: token
precedence, manifest freeze, baseline parity, ledger extraction, ablation
execution, leakage audits, and reproducible report output.

Copying the shape of `frameseed0_harness.py` or `frameseed_sheets0_harness.py`
would not be enough. Those harnesses belonged to packet/transmission
experiments. WGD is about discovered executable grammar under native absorbers.

### What This Adds Beyond I365

I365 says the harness is absent. I366 says file creation alone cannot fix the
problem. W36 must implement the WGD-specific contract, not merely instantiate a
new namespace.

### Implementation Demand

The WGD harness must expose a one-command pre-hidden audit that verifies:

```text
manifest_hashes
blind_boundary
native_absorber_roster
equal_information_contract
affordance_parity_matrix
grammar_ir_schema
component_erasure_roster
cost_extraction_rules
query_ledger_rules
leakage_audit_roster
sibling_independence_function
token_precedence_table
measurement_report_skeleton
```

### Verdict

```text
A HARNESS THAT ONLY CHECKS THAT THE SPEC EXISTS IS NOT WGD HARNESS INTEGRITY.
```

## I367: Attack Native Absorber Handles

### Attack

The WGD spec requires a large absorber roster: schema/binding, entity
resolution, PBE, PBE+CEGIS, CEGIS, active CEGIS, MDL library learning, sibling
library learning, active learning, causal/invariant discovery, constraint
learning and repair, anomaly/uncertainty abstention, ontology oracle, verifier
template oracle, obligation-label oracle, generator leakage classifier,
nuisance/leakage oracle, representation/parser/substrate prior, language prior
if used, and post-hoc compression.

The current checkout has none of these WGD absorber handles. Therefore native
absorbers are not genuinely trying. They are not trying at all.

### What This Adds Beyond I366

I366 attacks generic harness presence. I367 attacks the specific scientific
opponents WGD must defeat. Without absorber handles, there is no first-refusal
ladder.

### Implementation Demand

Every absorber must declare:

```text
absorber_name
status
input_artifacts
lossless_translation_hash
hypothesis_class
search_or_learning_strategy
query_budget
runtime_budget
cost_categories
functional_metrics
capability_witness_result
absorption_token
failure_mode
```

Any `proxy_absorber`, `capability_mode_scored`, or `untested_roster_entry` must
lower the maximum token before hidden opening.

### Verdict

```text
AN ABSORBER WITHOUT AN EXECUTABLE HANDLE IS A MISSING ABSORBER.
```

## I368: Attack Absorber Capability Witnesses

### Attack

The next loophole is worse than missing absorbers: absorbers that run but never
prove competence. A weak PBE baseline can always lose. A weak schema matcher can
always fail. A weak CEGIS enumerator can always time out. Their loss would mean
nothing.

B43 already demanded public capability witnesses. B44 makes that demand a
hidden-open blocker.

### What This Adds Beyond I367

I367 requires absorber handles. I368 requires each handle to win somewhere
before its loss can count.

### Implementation Demand

W36 must include absorber-owned calibration worlds:

```text
schema/binding world where schema/binding wins
entity-resolution world where entity resolution wins
PBE world where examples synthesize the policy
CEGIS world where counterexamples drive convergence
active-learning world where adaptive queries isolate the rule
MDL/library world where reusable macros win
constraint world where validators/nearest-valid search win
causal/invariant world where invariant discovery wins
anomaly world where uncertainty abstention wins
oracle controls where supplied ontology/templates/labels win
post-hoc compression world where compression is detected as non-causal
```

If W36 cannot make a boring explanation win on its home turf, W36 has not built
a hostile absorber.

### Verdict

```text
BASELINES THAT NEVER WIN IN PUBLIC CANNOT LOSE AS EVIDENCE IN HIDDEN.
```

## I369: Attack Equal Information At The Adapter Boundary

### Attack

The spec's equal-information contract is not satisfied by giving every system
the same JSON file. The WGD learner may receive convenient typed records,
canonical accessors, and direct grammar-construction affordances while a
baseline receives the same bytes through a clumsy adapter.

That is not equal information. It is equal storage with unequal search geometry.

### What This Adds Beyond I368

I368 asks whether absorbers can win in isolation. I369 asks whether they are
given the same affordances when competing with WGD.

### Implementation Demand

W36 must emit an affordance parity matrix:

```text
learner_public_field
WGD_access_path
absorber_access_path
translation_function
round_trip_test_hash
adapter_bits
empirical_access_cost
semantic_loss_allowed: false
known_disadvantage
charged_category
```

Any WGD-only convenience around parser support, typed semantics, action slots,
feedback labels, query answers, repair hints, or canonicalization should trigger
`WGD_VOID_SUBSTRATE_ASYMMETRY` or `WGD_VOID_BASELINE_PARITY_FAILURE`.

### Verdict

```text
EQUAL BYTES PLUS UNEQUAL ADAPTERS IS A PARITY FAILURE.
```

## I370: Attack Implementation Order Bias

### Attack

If W36 builds the WGD learner first and the absorbers second, the substrate will
almost certainly become WGD-shaped. The generator, serializer, grammar IR,
scorer, canonicalizer, and public primitives will be tuned around the learner's
needs. Then absorber parity becomes retrofit theater.

The current checkout has not yet made this mistake, because W36 is absent. But
the next implementation can make it quickly.

### What This Adds Beyond I369

I369 attacks adapter equality after systems exist. I370 attacks the build order
that decides whether equality will be real before adapters are written.

### Implementation Demand

W36 should implement absorber-first calibration before any WGD performance
path:

```text
1. freeze learner-public substrate skeleton
2. implement absorber capability worlds
3. prove each absorber can win on its home turf
4. implement parity/adapters/ledger
5. implement WGD learner or grammar constructor only after absorber witnesses
```

The WGD learner may be a stub during harness integrity. That is preferable to a
polished learner surrounded by weak late baselines.

### Verdict

```text
THE FIRST IMPLEMENTED SYSTEM BENDS THE BENCHMARK AROUND IT.
```

## I371: Attack The Grammar IR Smuggling Surface

### Attack

WGD requires a frozen executable `G_hat`. That creates a payload-laundering
surface. A grammar object can secretly be a per-task program set, a cache, an
interpreter-specific policy, a search controller, or a saved pipeline.

The current checkout has no WGD grammar IR schema, which means there is not yet
any protection against this failure.

### What This Adds Beyond I370

I370 attacks substrate bias before implementation. I371 attacks the primary
artifact WGD would use to claim discovery.

### Implementation Demand

Before any signal run, W36 must freeze:

```text
allowed_node_types
allowed_predicate_forms
allowed_repair_operators
allowed_composition_operators
allowed_public_fact_references
forbidden_code_blobs
forbidden_trace_caches
forbidden_hidden_labels
forbidden_per_task_program_slots
interpreter_hash
node_level_cost_attribution
execution_trace_logging
```

The grammar IR must be incapable of hiding a solver without counting that solver
as `P_i`, `L`, `H`, or `O`.

### Verdict

```text
WGD CANNOT DISCOVER A GRAMMAR UNTIL THE HARNESS CAN SAY WHAT A GRAMMAR IS NOT.
```

## I372: Attack Geometry-Erasure Decisiveness

### Attack

The required geometry-erasure ablations are decisive only if they remove the
declared component while leaving unrelated machinery intact. Otherwise an
erasure can be fake in either direction:

- too weak, because the active solver survives elsewhere;
- too strong, because it breaks plumbing instead of testing grammar causality.

The current checkout has no erasure executor, no component dependency map, and
no smoke result demonstrating that erasure changes the intended metric.

### What This Adds Beyond I371

I371 freezes the grammar container. I372 demands causal surgery on that
container.

### Implementation Demand

W36 must run public smoke erasures for:

```text
full_G_hat
T_hat transformations
O_hat obligations
R_hat repairs
A_hat abstention
C_hat composition
bindings_only
examples_counterexamples_only
active_query_only
MDL_library_replacement
per_task_program_replacement
verifier_template_oracle
generator_family_classifier
randomized_labels_obligations
role_name_unit_order_permutation
schema_isomorphism_holdout
repair_without_feedback
no_language_symbolic_condition
substrate_charged_accounting
sibling_clone_audit
```

Each erasure must report raw and canonicalized scores, affected component,
unaffected component checks, metric drop, and whether a boring replacement
matched under <4x all-in.

### Verdict

```text
ERASURE IS DECISIVE ONLY WHEN IT CAN KILL THE CLAIM WITHOUT KILLING THE HARNESS.
```

## I373: Attack Cost Ledger Honesty

### Attack

The current checkout has no deterministic WGD cost extractor. That means the
cost ledger is not honest or dishonest yet. It is nonexistent.

This is a blocker because WGD's positive claim depends on all-in cost: native
absorbers must fail or pay >=4x, and WGD's transfer must survive after grammar,
bindings, programs, examples, counterexamples, queries, verifier clauses,
repairs, abstention evidence, learned libraries, human substrate, ontology, and
nuisance-oracle bits are counted.

### What This Adds Beyond I372

I372 asks whether the grammar is causal. I373 asks whether the causal artifact
is cheap after every hidden subsidy is charged.

### Implementation Demand

W36 must provide a recomputable ledger:

```text
artifact_path
artifact_hash
system_or_absorber
role_in_execution
category in {G,B_i,P_i,E_i,C_i,Q_i,V_i,R_i,A_i,L,H,O,N}
bit_count_rule
runtime_count_rule
query_count_rule
human_design_minutes_or_commit_rule
adapter_bits
charged_to_wgd
charged_to_absorber
substrate_free_total
substrate_charged_total
reviewer_override_allowed: false
```

The ledger must be generated by code and tested by recomputation, not written
as prose after seeing outcomes.

### Verdict

```text
NO COST EXTRACTOR, NO ALL-IN CLAIM.
```

## I374: Attack Query And Feedback Accounting

### Attack

WGD can smuggle discovery through feedback. `ACCEPTED`, `REJECTED`, `UNSAFE`,
`AMBIGUOUS`, `WRONG`, repair failures, counterexamples, scalar rewards, diffs,
and failure messages are all information. If WGD receives them adaptively while
absorbers do not, the run is a query-oracle experiment.

The current checkout has no WGD query ledger or active/passive discovery mode
freeze.

### What This Adds Beyond I373

I373 counts static artifact cost. I374 attacks dynamic information acquired
during interaction.

### Implementation Demand

W36 must freeze and log:

```text
discovery_mode
query_count
query_type
answer_bits
adaptive_or_frozen
feedback_channel_payload
whether_query_touched_hidden_evaluation_distribution
which_system_received_answer
counterexample_payload_bits
repair_attempt_count
abstention_probe_count
charged_category
```

Active WGD cannot claim passive discovery. If active queries explain the
result, active learning and active CEGIS get first refusal.

### Verdict

```text
UNCOUNTED FEEDBACK IS A SPECIFICATION ORACLE.
```

## I375: Attack Leakage And Identifiability

### Attack

The WGD spec requires leakage and identifiability audits, but none are
implemented in the current checkout. This matters because synthetic typed worlds
can leak through high-order features: serializer offsets, row order, missingness
patterns, key cardinalities, hash lengths, split artifacts, feedback timing,
query ordering, and generator retry artifacts.

Low-order MI and simple classifiers would not be enough even if present.

### What This Adds Beyond I374

I374 attacks information intentionally returned through feedback. I375 attacks
information accidentally encoded in public artifacts.

### Implementation Demand

W36 must include hostile leakage audits:

```text
normalized_MI_by_target
linear_tree_forest_knn_predictors
boosted_or_adversarial_tree_predictors
compression_classifier
program_feature_search
serializer_offset_probe
split_reconstruction_attack
generator_retry_artifact_probe
public_feedback_sequence_predictor
banned_metadata_scan
permutation_stability_test
no_language_symbolic_condition
identifiability_alternative_search
```

Targets must include family id, role map, binding labels, transform class,
obligation class, repair location, abstention requirement, composition form,
sibling template, hidden query bucket, and scorer explanation category.

### Verdict

```text
PUBLIC FEATURES THAT PREDICT HIDDEN GEOMETRY ARE NOT BENIGN FEATURES.
```

## I376: Attack Siblings And Composition

### Attack

WGD's transfer claim depends on at least three nonduplicate reduced siblings and
held-out composition. The current checkout has no sibling generator, no
behavioral-distance function, no clone audit, no composition-form holdout, and
no interference tests.

Without those pieces, AFTD can be won by near clones and composition can be won
by saved pipelines.

### What This Adds Beyond I375

I375 attacks leakage of hidden geometry. I376 attacks the possibility that the
"held-out" geometry is not actually new.

### Implementation Demand

W36 must freeze:

```text
count_nonduplicate_reduced_siblings
minimum_behavioral_distance
maximum_template_overlap
nuisance_fingerprint_similarity_threshold
binding_reuse_fraction
schema_position_reuse_check
generator_family_classifier_shortcut_check
composition_noncommutation_tests
guard_conflict_tests
interference_tests
preserved_component_behavior_tests
composition_pipeline_baseline
MDL_library_composition_baseline
PBE_CEGIS_composition_baseline
```

Sibling count is not evidence unless each sibling imposes real new burden after
binding, program, library, query, and teaching bits are charged.

### Verdict

```text
TRANSFER WITHOUT DISTANCE IS CLONE ACCOUNTING.
```

## I377: Attack Repair And Abstention Utility

### Attack

Repair and abstention are sacred-outcome surfaces: improvability and usefulness
under uncertainty. They are also easy to game. Repair can become active retry.
Abstention can become refusal on hard cases.

The current checkout has no WGD repair harness, no no-feedback repair condition,
no charged-feedback repair condition, no nearest-valid/constraint/active-retry
baselines, no risk-coverage curves, and no utility accounting.

### What This Adds Beyond I376

I376 attacks transfer and composition. I377 attacks whether WGD remains useful
when it fails or does not know enough.

### Implementation Demand

W36 must report:

```text
repair_without_feedback
repair_with_single_failure_case
repair_with_interactive_feedback
changed_grammar_nodes
repair_patch_bits
repair_attempt_budget
preserved_behavior_HFA
nearest_valid_search_baseline
constraint_repair_baseline
CEGIS_repair_baseline
active_retry_baseline
patch_library_baseline
risk_coverage_curve
coverage_by_query_bucket
coverage_by_hidden_family
coverage_by_composition_form
false_abstention_opportunity_cost
unsafe_false_negative_cost
calibrated_uncertainty_baseline
anomaly_abstention_baseline
```

If only interactive repair works, the result is active learning unless active
absorbers fail under identical query information. If abstention hides hard
composition cases, it is not useful intelligence.

### Verdict

```text
REPAIR AND ABSTENTION ARE FUNCTIONAL CLAIMS, NOT ERROR-HANDLING DECORATION.
```

## I378: Attack Hidden-Open Governance And Final Synthesis

### Attack

The final implementation risk is governance. Once a hidden seed is opened,
there will be pressure to fix crashes, adjust timeouts, patch scorer edge cases,
repair malformed families, or reinterpret a missing ledger as harmless. The WGD
spec says mutations void the seed. The implementation must make that rule
automatic before the real hidden opening.

The current checkout has no WGD token decision table, no fake hidden-open fault
drill, no post-hidden mutation detector, and no reviewer countersignature
surface.

### What This Adds Beyond I377

I377 attacks functional behavior. I378 attacks whether the project can obey its
own terminal-token law when the result becomes emotionally expensive.

### Implementation Demand

W36 must precompute token assignment for:

```text
baseline_crash_after_hidden_open
scorer_bug_after_hidden_open
serializer_bug_after_hidden_open
timeout_mismatch_after_hidden_open
malformed_hidden_family_after_hidden_open
unexpected_leak_after_hidden_open
missing_absorber_after_hidden_open
missing_cost_category_after_hidden_open
post_hidden_threshold_change
post_hidden_query_mix_change
post_hidden_parser_or_canonicalizer_change
```

The decision table must put voids, parity failures, leakage, traps, and absorber
wins before negative or signal. Mixed evidence cannot be narrated upward.

### Verdict

```text
A TOKEN POLICY THAT IS NOT EXECUTED UNDER STRESS IS A WISH.
```

## Required W36 Gate Before Any Hidden Opening

The next W-loop cannot move directly to hidden signal measurement. It must first
produce a pre-hidden harness integrity bundle with these artifacts:

1. WGD harness code and tests.
2. W36 work-loop report with implementation evidence.
3. Manifest freeze and blind-boundary proof.
4. Native absorber handles for the full required roster.
5. Public capability witnesses where each absorber wins on its own home turf.
6. Affordance parity matrix with round-trip adapter tests.
7. Frozen grammar IR schema with solver-smuggling bans.
8. Public smoke geometry-erasure output.
9. Deterministic cost extractor and human-substrate ledger.
10. Query and feedback ledger.
11. Leakage, permutation, no-language, and identifiability audits.
12. Sibling independence and composition hostility checks.
13. Repair and abstention utility reports with active/constraint baselines.
14. Token decision table exercised on fake hidden-open faults.

If any item is missing, WGD may continue public implementation, but it cannot
open a hidden seed for signal.

## Final Token

```text
WGD_HARNESS_OVERSIGHT_BLOCKED_NO_W36_IMPLEMENTATION_IN_CURRENT_CHECKOUT
```

## Final Position

WGD remains the right kind of post-FrameSeed question because it asks for
discovered executable geometry rather than transmitted packets. But the current
checkout has not built the B36 harness the supervisor requested.

The strongest adversarial answer to the monitoring questions is therefore:

```text
Native absorbers are not genuinely trying because they do not exist here.
Geometry erasure is not decisive because no erasure harness exists here.
The cost ledger is not honest because no WGD ledger exists here.
```

The next real progress is not a hidden run. It is an implementation that gives
the boring explanations enough machinery, parity, and accounting power to kill
WGD if WGD is only schema matching, PBE, CEGIS, active querying, constraint
learning, library learning, leakage, language prior, or hand-authored substrate.

Until then, the adversary is not merely unconvinced. The adversary has no
implementation object to judge.
