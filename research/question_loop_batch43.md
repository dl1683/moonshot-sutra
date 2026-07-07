# Q-Loop Batch 43: WGD Implementation Readiness

**Date:** 2026-07-07  
**Role:** Question-Loop worker  
**Iterations:** I351-I364  
**Status:** post-hardening implementation-readiness attack on WGD-0.

Two invariants held fixed:

1. Swing for the home run.
2. The loop only stops on a won-over adversary.

## Grounding

Read the requested context in the current checkout:

1. `research/wgd_0_precommit_spec.md`
2. `research/question_loop_batch42.md`
3. `research/dual_loop_supervisor_checkin_33.md`
4. `research/frameseed_milestone_report.md`
5. `research/VISION.md`

Binding starting point:

```text
WGD is approved.
Approval is not evidence.
The WGD-0 spec is much harder than the pre-design object attacked in B42.
The remaining danger is no longer mostly missing language in the spec.
The danger is implementation theater: weak absorbers, asymmetric adapters,
subjective cost accounting, generator-shaped tasks, and post-hidden judgment.
```

The home-run question remains worth protecting:

```text
Can a cheap system discover the transformation and obligation geometry of a new
world well enough to act, repair, abstain, and transfer, under hostile native
absorbers and clean all-in accounting?
```

The B43 adversarial presumption:

```text
WGD-0 IS NOT READY FOR HIDDEN OPENING UNTIL ITS IMPLEMENTATION CAN MAKE THE
BORING EXPLANATIONS STRONGER THAN THE WGD LEARNER.
```

## Executive Verdict

WGD-0 is conceptually worth implementing, but it is not implementation-ready
merely because the precommit spec lists the right absorbers and void tokens.
FrameSeed already taught the harder lesson: an apparently clean hidden run can
still be scientifically dead if the active ingredient was supplied by the
substrate, bindings, search, typed pipeline, or accounting convention.

The most dangerous remaining failure is not leakage in the crude sense. It is
baseline theater:

```text
The repo implements a WGD learner carefully, implements native absorbers just
enough to name them, gives those absorbers lossy or awkward adapters, counts
WGD engineering as free substrate, and then treats absorber failure as evidence.
```

If that happens, the honest terminal token is:

```text
WGD_INCONCLUSIVE_BASELINES_NOT_NATIVE
```

or, if the asymmetry is material:

```text
WGD_VOID_BASELINE_PARITY_FAILURE
WGD_VOID_SUBSTRATE_ASYMMETRY
WGD_VOID_COST_LEDGER_FAILURE
```

The implementation readiness bar is therefore:

```text
Before hidden opening, every required absorber must be able to win on public
calibration worlds designed for its boring explanation, using the same bytes,
same interface affordances, same canonicalizer, same budgets, and a cost ledger
that a hostile reviewer can recompute.
```

## Single Most Likely Failure Mode

The single most likely WGD-0 failure mode is:

```text
NATIVE_ABSORBER_THEATER
```

Definition:

```text
Each required absorber exists as code, but at least one is not the strongest
reasonable native implementation of the boring explanation it represents; or
it receives a worse representation than WGD; or its adapter/canonicalizer
charges cost differently; or it is not tuned and validated with the same care.
```

Why this is the most likely failure:

- It is easier to write a WGD learner than to write twelve hostile native
  absorbers well.
- It is easier to make parity true at the byte level than at the affordance
  level.
- It is easier to undercount human substrate than to quantify it.
- It is easier to declare a baseline "native executable" than to prove it is
  adversarially competent.
- It is emotionally easier to let a weak absorber lose than to let the boring
  explanation win.

Kill criterion:

```text
If any required absorber cannot demonstrate competence on pre-hidden public
calibration tasks that instantiate its absorption route, WGD-0 cannot open the
hidden seed for signal. Maximum token: WGD_INCONCLUSIVE_BASELINES_NOT_NATIVE.
```

## Implementation Confound Catalog

| ID | Confound | How it survives the hardened spec | Required adversarial control |
|---|---|---|---|
| C01 | Native absorber theater | The spec names native absorbers, but implementation quality is weak. | Each absorber must pass public capability witnesses before hidden open. |
| C02 | Equal bytes, unequal affordances | Baselines receive the same serialized data but need harder parsing, search, or canonicalization. | Publish an affordance parity matrix, not only a file parity matrix. |
| C03 | Adapter tax | WGD gets a direct learner-public adapter while absorbers get awkward translations. | Charge adapter work and verify lossless, ergonomic translations by round trip. |
| C04 | Grammar container smuggling | `GRAMMAR(grammar_object)` can hide a solver, interpreter, cache, or per-task programs. | Freeze a grammar schema with banned executable payloads and size attribution. |
| C05 | Cost-bit fiction | `G`, `B_i`, `P_i`, `H`, `O`, and `N` are listed but not mechanically countable. | Provide deterministic cost extraction scripts before public smoke. |
| C06 | Human substrate laundering | Generator design, feature selection, parser code, and baseline adapters become uncharged background. | Maintain a human-substrate ledger with commit-linked source and charge class. |
| C07 | Public-dev overfitting | Repeated public/smoke iteration sculpts the generator toward WGD without post-hidden mutation. | Limit dev-family iteration, log every generator change, and report dev overfit risk. |
| C08 | WGD-shaped generator | Worlds are designed around the intended WGD hypothesis class rather than around neutral discovery pressure. | Require absorber-first generator review and calibration worlds where absorbers win. |
| C09 | Hidden distribution hand tuning | Query mix or family balance makes WGD's strengths count more than boring baselines' strengths. | Freeze stratified query buckets and report per-bucket absorber outcomes. |
| C10 | Canonicalizer doing semantics | Output canonicalization normalizes away errors or reveals target structure. | Score raw and canonicalized outputs; charge canonicalizer semantics as `H` or `O`. |
| C11 | Feedback channel oracle | `ACCEPTED`, `REJECTED`, `UNSAFE`, `AMBIGUOUS`, and `WRONG` frequencies reveal hidden predicates. | Treat every feedback bit as query information and run feedback-only absorbers. |
| C12 | Repair as active search | Local repair becomes try-until-accepted under a small patch budget. | Separate no-feedback repair from charged-feedback repair and compare active baselines. |
| C13 | Abstention threshold gaming | High abstention recall hides low coverage or composition failure. | Require risk-coverage curves, utility accounting, and per-bucket over-abstention. |
| C14 | Sibling clone inflation | Three siblings are near duplicates under nuisance changes, inflating AFTD. | Precompute behavioral distance and clone-resistance before hidden open. |
| C15 | Composition as saved pipeline | Held-out composition is just known component programs glued in an obvious order. | Include interference, noncommutation, guard conflict, and order-error probes. |
| C16 | Low-order leakage audit | MI, linear, tree, and nearest-neighbor tests miss high-order serializer or generator side channels. | Add program-feature, compression, adversarial classifier, and split-reconstruction audits. |
| C17 | Formal lower-bound laundering | A weak formal bound is used where a native executable absorber is needed. | Lower bounds must cover the exact public interface and hypothesis class. |
| C18 | Timeout favoritism | WGD gets optimized runtime while absorbers time out due to unoptimized search. | Run budget curves and engineering-effort ledgers for WGD and absorbers. |
| C19 | Post-hidden bug exception | A "harmless bug fix" after hidden open changes scorer, parser, query mix, or baseline behavior. | Any material post-hidden code change voids the seed; dry-run the void rule. |
| C20 | Token adjudication drift | Mixed evidence is narrated toward signal despite a higher-precedence void or absorber. | Precompute a token decision table and require reviewer countersignature. |

## Adversarial Review Criteria

No hidden seed should open until a hostile reviewer can answer every item below
with "yes" from runnable artifacts, not intent.

1. **Manifest reproducibility:** Are all generator, serializer, scorer,
   canonicalizer, baseline, learner, timeout, budget, query-mix, audit, and
   report files hashed and runnable from one manifest?
2. **Blind boundary:** Can the reviewer prove that learners, constructors, and
   baselines cannot read hidden role maps, labels, seeds, scorer internals, or
   hidden answers?
3. **Absorber capability:** Does each required absorber win on public
   calibration worlds where its boring explanation is deliberately true?
4. **Affordance parity:** For every WGD field, does the matching absorber field
   have identical bytes or a verified lossless translation with comparable
   ergonomics and counted adapter cost?
5. **Cost recomputation:** Can a fresh reviewer recompute `G`, `B_i`, `P_i`,
   `E_i`, `C_i`, `V_i`, `R_i`, `A_i`, `L`, `H`, `O`, and `N` from artifacts?
6. **Grammar causality:** On public smoke, does erasing each grammar component
   cause the expected functional drop without changing unrelated machinery?
7. **No solver smuggling:** Is the grammar object schema unable to contain
   per-task programs, caches, opaque executable code, hidden labels, or
   interpreter-specific shortcuts without being charged as `P_i`, `L`, or `H`?
8. **Leakage adversary:** Do frozen leakage audits include high-order,
   compression, serializer-offset, split-reconstruction, and program-feature
   attacks, not only simple MI and classifier checks?
9. **Sibling independence:** Are siblings behaviorally separated enough that
   near-duplicate clone reduction cannot pass AFTD?
10. **Composition hostility:** Are held-out compositions constructed to expose
    order, guard, interference, and preservation failures, not only sequential
    pipeline reuse?
11. **Repair accounting:** Are repair attempts, failed patches, feedback bits,
    and changed grammar nodes all charged and compared against active retry,
    nearest-valid, and constraint-repair baselines?
12. **Abstention utility:** Does the run report risk-coverage, false-abstention
    opportunity cost, unsafe false negatives, and per-bucket abstention behavior?
13. **Budget curves:** Do WGD and absorbers have performance-vs-cost curves, so
    a single unlucky timeout cannot decide novelty?
14. **Token table:** Is there a deterministic precedence table that maps every
    observed failure, missing absorber, asymmetry, leak, mutation, and metric
    miss to exactly one terminal token?
15. **Negative-result discipline:** Has the report template already committed
    to publishing inconclusive, void, absorbed, or negative outcomes without
    narrative rescue?

If any item fails, implementation may continue on public seeds, but hidden
opening for signal is premature.

## I351: Attack Baseline Theater

### Attack

B42 attacked missing absorbers. The hardened spec adds them. That does not solve
the implementation problem. A named absorber is not a hostile absorber.

The likely implementation path is asymmetric engineering:

```text
WGD learner: carefully designed, iterated, profiled, and debugged.
PBE/CEGIS/MDL/active/schema baselines: implemented to satisfy the roster.
```

Then the report says the absorbers failed. A hostile reviewer will not be moved.
They will ask whether the boring explanations were represented by strong native
systems or by placeholder code.

### Remaining Danger After Hardening

The spec uses status labels such as `native_executable`, but the label can be
wrong. Native means more than runnable. It means the absorber's hypothesis
class, search strategy, query policy, repair machinery, and cost accounting are
strong enough that its failure is evidence.

### Implementation Demand

Every required absorber must pass a public capability witness:

```text
schema/binding absorber wins on binding-shaped worlds.
PBE absorber wins on example-synthesis worlds.
CEGIS absorber wins on counterexample-shaped worlds.
MDL absorber wins on macro-library worlds.
active absorber wins on query-identifiable worlds.
constraint absorber wins on validator/repair worlds.
causal/invariant absorber wins on invariant-shaped worlds.
```

If an absorber cannot win when its own explanation is true, it cannot lose as
evidence when WGD wins.

### Verdict

```text
A WEAK ABSORBER IS NOT A CONTROL. IT IS A DECORATION.
```

## I352: Attack Equal Information At The Affordance Level

### Attack

The spec requires identical bytes or lossless translations. That is necessary
and insufficient. Equal bytes can still be unequal information if one system's
adapter exposes the right affordances and another must rediscover them through
awkward encoding.

Example:

```text
WGD receives structured objects already arranged for grammar induction.
PBE receives the same bytes but through a wrapper that makes enumeration,
unification, or counterexample reuse expensive.
```

The bytes are equal. The experiment is not.

### Remaining Danger After Hardening

The public substrate has primitives, serializers, canonicalizers, and action
interfaces. Those are not neutral. They define a search geometry. If WGD's
search geometry aligns with the interface and the absorbers' geometry does not,
absorber failure is not evidence.

### Implementation Demand

Add an affordance parity matrix:

```text
field_or_operation
WGD_access_path
absorber_access_path
lossless_translation_hash
adapter_bits_charged
round_trip_test
asymptotic_or_empirical_access_cost
known_disadvantage
```

Any material disadvantage must be charged or the run is void.

### Verdict

```text
EQUAL FILES ARE NOT EQUAL AFFORDANCES.
```

## I353: Attack The Grammar Object Itself

### Attack

The spec says WGD must output an executable grammar. That is the right causal
pressure, but it creates a smuggling surface. A "grammar" can secretly be:

- an interpreter plus hidden policy;
- a compressed per-task program set;
- a cache of public traces;
- a solver with a prettier API;
- a search strategy whose real cost appears only at execution time;
- a library learner renamed as geometry.

The danger is not post-hoc interpretability alone. The danger is executable
payload laundering.

### Remaining Danger After Hardening

The grammar tuple `G_hat` includes predicates, repair operators, composition
rules, uncertainty, provenance, and executable cost. Without a frozen schema,
those fields can contain arbitrary machinery. The cost ledger may then count
the wrapper as `G` while the active solver lives in `P_i`, `L`, `H`, or hidden
runtime state.

### Implementation Demand

Freeze a grammar intermediate representation before public smoke:

```text
allowed node types
allowed predicate forms
allowed references to public facts
allowed repair operators
allowed composition operators
runtime interpreter hash
forbidden opaque code blobs
forbidden caches
forbidden hidden labels
mandatory node-level cost attribution
```

Then run erasure at node granularity. If deleting the declared grammar nodes
does not hurt the relevant function, the grammar was not causal.

### Verdict

```text
A GRAMMAR THAT CAN CONTAIN A SOLVER IS NOT A GRAMMAR CONTROL.
```

## I354: Attack Cost Accounting As A Scientific Instrument

### Attack

The spec lists the right cost categories. Implementation can still fail because
the categories are not mechanically measurable. Bits are easy to name and hard
to count.

The hardest categories are:

```text
H = human-authored substrate and design work.
O = supplied ontology or verifier-template bits.
N = nuisance or leakage feature selection.
L = learned library bits.
P_i = per-task policy hidden inside grammar or repair.
```

If these are assigned by narrative after results, the ledger is not evidence.

### Remaining Danger After Hardening

FrameSeed SHEETS-0 died because binding-only and typed pipeline machinery did
the work. WGD can die the same way if generator design, parser design, feature
inventory, or canonicalizer design is counted as zero-cost infrastructure.

### Implementation Demand

Before hidden opening, create a deterministic cost extraction tool and a human
substrate ledger:

```text
artifact_path
artifact_hash
author_or_generator
role_in_execution
cost_category
bit_count_rule
human_design_minutes_or_commit_count
charged_to_wgd
charged_to_absorbers
reviewer_override_allowed: false after hidden open
```

The exact bit model may be crude, but it must be frozen and symmetric.

### Verdict

```text
UNMECHANIZED COST ACCOUNTING IS A NARRATIVE, NOT A LEDGER.
```

## I355: Attack Public-Seed Sculpting

### Attack

The post-hidden mutation rule is strong. It does not prevent pre-hidden
overfitting. A team can iterate on public dev and smoke seeds until the
generator, query mix, and learner fit each other. Then hidden opening is clean
but the benchmark is already WGD-shaped.

This is not cheating in the crude sense. It is research drift.

### Remaining Danger After Hardening

The spec allows public dev and public smoke. If every failure on public seeds
causes generator revisions, metric revisions, threshold revisions, or
representation revisions, the hidden family becomes a selected artifact of the
WGD learner's strengths.

### Implementation Demand

Log pre-hidden adaptation:

```text
generator_change_count
threshold_change_count
baseline_change_count
learner_change_count
which public failure motivated the change
whether change helped WGD, absorbers, or both
```

Add a public "absorber-first" calibration set where each boring explanation is
supposed to win. If WGD wins every public calibration condition too, the
generator may be encoding WGD's inductive bias as the target.

### Verdict

```text
HIDDEN-SEED DISCIPLINE DOES NOT CANCEL PUBLIC-SEED OVERFITTING.
```

## I356: Attack Generator-Shaped Identifiability

### Attack

The spec voids unidentifiable semantics and leakage. Good. The remaining danger
is a generator that makes exactly the WGD hypothesis class identifiable while
making boring classes look unnatural.

That is subtler than leakage. The hidden grammar is not leaked. It is selected
from a family where WGD's representation is the natural one.

### Remaining Danger After Hardening

Every synthetic benchmark privileges some hypothesis class. WGD-0 can become a
test of whether WGD matches the generator, not whether cheap systems can
discover world grammar in a substrate-neutral way.

### Implementation Demand

For each hidden pressure, define the boring rival it is meant to stress:

```text
H1 surface invariance -> schema/binding and nuisance absorbers
H2 typed measurement -> constraint/PBE/CEGIS absorbers
H3 dependency closure -> causal/invariant/constraint absorbers
H4 safety and invalidity -> verifier/constraint absorbers
H5 underidentification -> active learning and abstention baselines
H6 local repair -> constraint repair and nearest-valid baselines
H7 composition -> MDL/library/CEGIS absorbers
```

Then construct public calibration worlds where each rival wins. If the harness
cannot make the rival win on its own home turf, the rival's hidden loss is not
meaningful.

### Verdict

```text
A FAIR GENERATOR MUST BE ABLE TO EMBARRASS WGD.
```

## I357: Attack Scorer And Canonicalizer Semantics

### Attack

The scorer and canonicalizer are not passive. They define equivalence classes.
They can silently normalize away wrong behavior, reveal target structure, or
make some representations cheaper than others.

A learner can exploit:

- canonical ordering;
- accepted output normal forms;
- failure bucket boundaries;
- repair-location hints implicit in diffs;
- serialization offsets;
- action-delta normalization.

### Remaining Danger After Hardening

The spec freezes scorer and canonicalizer, but freeze is not neutrality. A
frozen semantic leak is still a leak. A frozen representation advantage is still
an advantage.

### Implementation Demand

Score both raw and canonicalized outputs where possible:

```text
raw_output_validity
canonicalized_output_validity
canonicalizer_edit_distance
canonicalizer_changed_semantics
absorber_delta_after_canonicalization
```

Run a canonicalizer-oracle absorber. If canonicalization supplies the decisive
structure, emit representation, parser, verifier, or hand-substrate absorption.

### Verdict

```text
THE CANONICALIZER CAN BE THE TEACHER.
```

## I358: Attack Repair As Membership Query Learning

### Attack

Local repair is supposed to show improvability. It can also become a membership
query loop:

```text
try patch -> receive failure -> adjust patch -> receive failure -> converge.
```

The grammar looks repairable because the protocol let it interrogate the world.

### Remaining Danger After Hardening

The spec charges feedback and patch bits, but implementation can still blur
repair discovery, patch search, and active querying. The threshold
`changed_grammar_nodes <= max(2, ceil(0.10 * total_grammar_nodes))` is not
enough if the changed nodes are powerful or if attempts are numerous.

### Implementation Demand

Split repair into three measured regimes:

```text
repair_without_feedback
repair_with_single_failure_case
repair_with_interactive_feedback
```

For each regime, compare against:

```text
nearest_valid_search
constraint_repair
CEGIS_repair
active_retry
patch_library_baseline
```

If only interactive repair works, the claim is active repair learning unless
the native active absorber fails under equal query information.

### Verdict

```text
REPAIR IS IMPROVABILITY ONLY AFTER QUERY LEARNING FAILS.
```

## I359: Attack Abstention Without Utility

### Attack

Abstention metrics can look principled while hiding weak action. The spec sets
recall, precision, and over-abstention thresholds. That is a start, not a
utility model.

A learner can optimize:

```text
abstain on composition-hard cases
act on easy surface-invariant cases
avoid unsafe false negatives
miss the home-run behavior
```

The metrics can pass while the system is too timid to be useful.

### Remaining Danger After Hardening

The Vision cares about ordinary usefulness, not only classifier balance.
Abstention must preserve usefulness under uncertainty. Otherwise the system is
safe by retreat, not intelligent.

### Implementation Demand

Report risk-coverage and utility:

```text
coverage_by_query_bucket
coverage_by_hidden_family
coverage_by_composition_form
utility_at_risk_thresholds
false_abstention_opportunity_cost
unsafe_false_negative_cost
low_confidence_calibration
```

Require a baseline that abstains by calibrated uncertainty and anomaly score.
If that baseline matches, WGD's abstention is not special.

### Verdict

```text
ABSTENTION THAT DESTROYS USEFULNESS IS NOT A SACRED OUTCOME.
```

## I360: Attack Sibling And Composition Accounting

### Attack

The spec requires three reduced siblings and held-out composition. The remaining
danger is that siblings are different enough to look legitimate but not
different enough to defeat amortized generator reuse.

Similarly, held-out composition can be a saved pipeline:

```text
component A then component B
```

with no genuine interference, noncommutation, guard conflict, or repair
preservation challenge.

### Remaining Danger After Hardening

AFTD can be won by counting near duplicates. Composition can be won by pipeline
assembly. Both would look like transfer while staying inside library learning.

### Implementation Demand

Before hidden open, define:

```text
behavioral_distance_between_siblings
shared_template_overlap
nuisance_fingerprint_similarity
binding_reuse_fraction
component_noncommutation_tests
guard_conflict_tests
interference_tests
preserved_component_behavior_tests
```

Only nonduplicate reduced siblings count in `AFTD_all_in`.

### Verdict

```text
TRANSFER IS NOT A COUNT. IT IS A DISTANCE-WEIGHTED BURDEN.
```

## I361: Attack Leakage Audits That Are Too Simple

### Attack

The spec lists MI, predictors, permutation, banned metadata, and no-language
audits. That is strong against obvious leakage. It may miss the leakage that
actually kills synthetic worlds:

- high-order feature interactions;
- serializer offset patterns;
- ordering conventions;
- hash or length side channels;
- split construction artifacts;
- generator retry artifacts;
- compression fingerprints;
- public feedback timing;
- decoy imbalance.

### Remaining Danger After Hardening

Low-order leakage tests can pass while a small program-feature classifier
predicts hidden family, role map, query bucket, or obligation class.

### Implementation Demand

Add hostile leakage audits:

```text
compression_classifier
program_feature_search
serializer_offset_probe
split_reconstruction_attack
generator_retry_artifact_probe
adversarial_random_forest_or_boosted_tree
small_synthesis_predictor
public_feedback_sequence_predictor
```

The leakage target set must include not only family ID but also repair location,
abstention requirement, composition form, and reduced-sibling membership.

### Verdict

```text
LEAKAGE THAT NEEDS A CLEVER CLASSIFIER IS STILL LEAKAGE.
```

## I362: Attack Formal Lower Bound And Proxy Escape Hatches

### Attack

The spec permits some absorbers to be native executable or formal lower bound.
That is dangerous. A lower bound over the wrong class can become a permission
slip to skip the real absorber.

Example:

```text
We prove a simple CEGIS enumeration bound is too expensive.
But a typed, symmetry-aware, constraint-pruned CEGIS would be cheap.
```

The lower bound did not cover the actual boring explanation.

### Remaining Danger After Hardening

Formal lower bounds are valuable only if they cover the exact public interface,
the exact hypothesis class, and the strongest known pruning or representation
tricks. Otherwise they launder an implementation gap into rigor.

### Implementation Demand

Any formal lower bound must state:

```text
covered hypothesis class
covered primitives
covered feedback channel
covered symmetry reductions
covered type constraints
covered query policy
gap_to_best_known_native_algorithm
```

If the gap is nontrivial, the maximum token is inconclusive, not signal.

### Verdict

```text
A LOWER BOUND ON A STRAW ABSORBER IS NOT A DEFENSE.
```

## I363: Attack Hidden-Open Governance

### Attack

The post-hidden mutation rule is clear, but implementation reality creates
pressure:

- a baseline crashes;
- a scorer has an edge-case bug;
- a timeout was misconfigured;
- a serializer emits invalid records;
- the canonicalizer rejects a harmless output;
- a hidden family is malformed.

Each case invites a "reasonable" fix. Enough reasonable fixes destroy the
hidden seed.

### Remaining Danger After Hardening

The spec says mutation voids the run. The team may still treat certain fixes as
non-material because otherwise the run is lost. That is where adversarial trust
dies.

### Implementation Demand

Before hidden open, dry-run a fake hidden opening with injected faults and
practice token assignment:

```text
baseline_crash_after_hidden_open
scorer_bug_after_hidden_open
serializer_bug_after_hidden_open
timeout_mismatch_after_hidden_open
malformed_hidden_family_after_hidden_open
unexpected_leak_after_hidden_open
```

For each, precommit whether the result is void, inconclusive, negative, or
rerun with a rotated seed. No case-by-case rescue after real opening.

### Verdict

```text
HIDDEN-SEED DISCIPLINE IS GOVERNANCE, NOT A SENTENCE IN THE SPEC.
```

## I364: Final Implementation Readiness Attack Synthesis

### Attack

The WGD-0 spec is no longer obviously underhardened. That is precisely why the
next adversarial layer matters. A strong spec can fail by weak implementation.

The remaining adversarial question is:

```text
Can a hostile reviewer run the repo and see that the boring explanations were
implemented with enough strength, parity, and cost discipline that their loss
would actually mean something?
```

If the answer is no, WGD-0 has not reached hidden-opening readiness.

### Required Pre-Hidden Implementation Gates

Before hidden opening, WGD-0 needs:

```text
G1 runnable manifest with hashes and one-command reproduction.
G2 blind boundary proof for learner, constructor, baselines, and scorer.
G3 absorber capability witnesses for every required native absorber.
G4 affordance parity matrix and round-trip translation tests.
G5 deterministic cost extraction and human-substrate ledger.
G6 frozen grammar IR with no solver/caching smuggling.
G7 public smoke erasures proving grammar components are causal when they should be.
G8 high-order leakage audits beyond MI and simple predictors.
G9 clone-resistant sibling distance and hostile composition probes.
G10 repair/abstention utility reports with active and constraint baselines.
G11 budget curves for WGD and absorbers.
G12 token decision table exercised on fake hidden failures.
```

### Hostile Reviewer Sentence

```text
Do not show me that WGD beat your baselines. Show me that your baselines were
the strongest boring explanations your own spec could imagine, that they had
equal affordances, that they were allowed to win on calibration worlds, and
that every cost and post-hidden decision was locked before the hidden seed.
```

### Final Token

```text
WGD_IMPLEMENTATION_READINESS_ATTACK_COMPLETE_NATIVE_ABSORBER_THEATER_IS_THE_MAIN_RISK
```

### Final Position

WGD is still the right post-FrameSeed direction because it asks for discovery
rather than transmission. The hardened spec is a serious start. But the
implementation can still fail in the exact way FrameSeed failed: the apparent
home-run object can be supplied by surrounding machinery, and the absorbers can
be too weak to expose it.

The adversary is not yet won over by WGD-0 implementation readiness. The
adversary can be won over only by a pre-hidden harness where absorber strength,
affordance parity, cost accounting, grammar causality, leakage resistance,
sibling independence, repair/abstention utility, and hidden-open governance are
all executable before the first hidden seed opens.
