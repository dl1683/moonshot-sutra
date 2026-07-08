# Q-Loop Batch 45: WGD-0 Harness Code Review And Measurement Oversight

**Date:** 2026-07-07  
**Role:** Question-Loop worker  
**Iterations:** I379-I392  
**Status:** code-visible WGD-0 harness oversight after B44's blind/absent-harness review.

Two invariants held fixed:

1. Swing for the home run.
2. The loop only stops on a won-over adversary.

## Grounding

Read the requested context in the current checkout:

1. `code/wgd0_harness.py`
2. `code/test_wgd0_harness.py`
3. `research/wgd_0_precommit_spec.md`
4. `research/question_loop_batch43.md`
5. `research/question_loop_batch44.md`
6. `research/VISION.md`

B44 is now stale in one important way: the harness is no longer absent. The checkout contains a real WGD-0 audit harness and tests. That changes the review from "nothing exists" to a sharper question:

```text
Does the existing harness merely instantiate the B43/B44 checklist, or does it make native boring explanations dangerous enough that their future loss would mean something?
```

## Validation Run

Local validation performed:

```text
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'; python -m pytest code/test_wgd0_harness.py -q
```

Result:

```text
8 passed, 1 warning in 0.19s
```

The warning was pytest cache creation denied under `.pytest_cache`; it did not affect test results.

Direct smoke audit:

```text
python -c "import sys; sys.path.insert(0, 'code'); import wgd0_harness as h; r=h.run_preimplementation_audit(dry_run_worlds=120, leakage_threshold=0.5); ..."
```

Result:

```text
passed=True
finding_count=74
bad=[]
absorber_count=20
hidden_seed_opened=False
wgd_signal_measured=False
baseline_calibration_only=True
```

This proves the scaffold is internally green. It does not prove hidden-opening readiness.

## Executive Verdict

The harness is a real improvement over B44's missing-artifact state. It now has world generation, blind packet construction, a grammar IR smoke object, baseline packet parity checks, a human-substrate ledger, 20 absorber capability witnesses, leakage NMI probes, sibling and composition checks, repair/abstention controls, token precedence controls, hidden-open governance drills, and tests.

But the adversary is not won over.

The harness currently certifies a **pre-hidden audit scaffold**, not a measurement system that can support `WGD_SIGNAL`. Its strongest honest claim is:

```text
The WGD-0 project now has a runnable public integrity harness that checks the presence and internal consistency of several precommit surfaces.
```

It cannot yet claim:

```text
The 20 native absorbers are genuinely hostile on the WGD target distribution.
Geometry erasure proves discovered grammar is causal.
NATIVE_ABSORBER_THEATER is avoided.
```

The current maximum scientific token before hidden measurement remains:

```text
WGD_INCONCLUSIVE_BASELINES_NOT_NATIVE
```

If a hidden signal narrative were built directly on this harness without the missing measurement layers, the live higher-precedence risks would be:

```text
WGD_VOID_BASELINE_PARITY_FAILURE
WGD_VOID_SUBSTRATE_ASYMMETRY
WGD_VOID_COST_LEDGER_FAILURE
WGD_TRAP_LOOKUP_OR_TINY_DSL
WGD_ABSORBED_BY_HAND_AUTHORED_SUBSTRATE
WGD_ABSORBED_BY_REPRESENTATION_PRIOR
WGD_ABSORBED_BY_PBE
WGD_ABSORBED_BY_CEGIS
WGD_ABSORBED_BY_ACTIVE_LEARNING
WGD_ABSORBED_BY_MDL_LIBRARY_LEARNING
WGD_ABSORBED_BY_POST_HOC_COMPRESSION
```

## Direct Answers

| Question | B45 answer | Reason |
|---|---|---|
| Are the 20 native absorbers genuinely trying? | Partly as toy capability witnesses; not yet as hostile native absorbers. | Every required absorber has a runnable witness, but the witnesses are tiny self-contained calibration exercises, not full competitors on the WGD substrate, query mix, hidden-style worlds, budgets, and output gates. |
| Is geometry-erasure decisive? | No. | The harness freezes the 22-ablation roster but `audit_ablation_roster()` explicitly reports no hidden erasure HFA, and there is no component-erasure executor measuring causal drops. |
| Is `NATIVE_ABSORBER_THEATER` avoided? | Not yet. | The theater risk has moved from absent absorbers to self-certified witness absorbers, identity parity declarations, and declared-but-unrun high-order audits. |

## I379: Attack The New Reality, Not The Old Absence

### Attack

B44 said no WGD harness existed. That is no longer true. The new harness is 1,352 lines, has 8 tests, and the direct preimplementation audit returns 74 passing findings.

The adversarial mistake would be to keep repeating B44's absence verdict after the artifact appears. That would be lazy oversight.

### What This Adds

The live question is not file presence. The live question is whether the files create enough adversarial pressure to make future hidden evidence meaningful.

### Verdict

```text
THE HARNESS EXISTS. EXISTENCE IS NOW THE FLOOR, NOT THE BAR.
```

## I380: Attack Self-Consistency As A Substitute For Scientific Force

### Attack

The tests and smoke audit pass. That matters. But most checks are contract checks over objects the harness itself constructs:

```text
packet bits recompute
grammar node bits recompute
absorber witness outputs contain expected keys
baseline views receive identical packet hashes
ablation roster has length 22
token controls map to expected strings
```

Those are useful integrity checks. They are not evidence that WGD discovered geometry or that native absorbers lost fairly.

### What This Adds Beyond I379

I379 acknowledges the new artifact. I380 separates runnable green checks from adversarial measurement.

### Required Hardening

Every green check should be labeled by evidence class:

```text
presence_check
self_consistency_check
toy_capability_witness
target_distribution_smoke
full_measurement_gate
```

Only the last two can move hidden-opening readiness.

### Verdict

```text
A GREEN AUDIT SCAFFOLD IS NOT A WON-OVER ADVERSARY.
```

## I381: Attack The 20 Absorbers As Toy Witnesses

### Attack

All 20 required absorbers are present and pass. That is a meaningful improvement. But most witnesses are deliberately tiny and hand-shaped:

```text
PBE chooses among 7 candidate programs.
PBE+CEGIS learns a one-dimensional threshold.
CEGIS learns a modular predicate over a small candidate list.
Active CEGIS binary-searches a threshold.
MDL library compresses identical task tuples.
Entity resolution uses exact keys and high-overlap aliases.
Generator leakage uses a deliberate serializer-offset side channel.
Post-hoc compression detects an artifact explicitly marked created_after_solutions.
```

These prove that each absorber can win a toy case. They do not prove each absorber is the strongest reasonable native implementation for WGD-0.

### What This Adds Beyond I380

I380 attacks green checks generally. I381 attacks the absorber witness content.

### Required Hardening

Each absorber needs a second witness tier:

```text
same WGD public substrate
same packet/transcript format
same hidden-style role opacity
same output contract
same query and repair ledger
same budget curves
calibration worlds generated by the same generator family or by declared rivals
```

### Verdict

```text
THE ABSORBERS ARE PRESENT. THEY ARE NOT YET DANGEROUS.
```

## I382: Attack The Same-Bytes Contract

### Attack

`same_bytes_contract` defaults to `True` in `_cap()`. The baseline parity machinery constructs identity JSON translations for every absorber. The affordance matrix then declares every field reachable through `canonical_json_identity_translation`.

That is not a proof of equal affordance. It is a declaration over a shared packet object.

The B43 danger was not only denied bytes. It was equal bytes with unequal search geometry.

### What This Adds Beyond I381

I381 says the absorbers are toy-capable. I382 says even those absorbers are not shown to operate through real absorber-specific adapters with real ergonomic parity.

### Required Hardening

Replace identity declarations with absorber adapters that actually run:

```text
adapter input bytes
adapter output structure
round-trip proof
feature access calls
access-cost trace
adapter bit charge
semantic-loss assertion
failure case where denied fields break the absorber
```

### Verdict

```text
SAME-BYTES TRUE BY DEFAULT IS NOT PARITY.
```

## I383: Attack The World Generator As A Tiny DSL Trap

### Attack

The current generated world has six fixed latent roles:

```text
source_ref, target_ref, quantity, unit, guard, status
```

Public traces edit only:

```text
quantity, guard, status
```

The hidden behavior is correspondingly small:

```text
quantity above unsafe_threshold -> UNSAFE
locked non-guard edit -> REJECTED
locked and guard true after edit -> AMBIGUOUS
otherwise ACCEPTED
```

This is a reasonable smoke world. It is not yet a home-run world grammar discovery challenge.

### What This Adds Beyond I382

I382 attacks parity mechanics. I383 attacks the object being protected: the world itself may be too small and too regular to avoid trap tokens later.

### Required Hardening

Before hidden opening, hidden-style public smoke worlds need multiple grammar families with:

```text
nontrivial dependency closure
behaviorally identifiable obligations
repair locality pressures
composition interference
underidentification cases
role ambiguity after type decoys
absorber-first rival worlds
```

### Verdict

```text
A SIX-ROLE SMOKE WORLD CAN TEST THE HARNESS. IT CANNOT CARRY WGD_SIGNAL.
```

## I384: Attack The Grammar IR As Smoke, Not Causal Grammar

### Attack

`make_smoke_grammar_ir()` creates seven fixed nodes:

```text
typed_projection
equality_guard
range_guard
dependency_closure
repair_patch
abstention_rule
composition_rule
```

The IR bans obvious solver-smuggling strings and recomputes bits. Good. But the IR is not executed against tasks. It is not the output of a learner. It is not used to act, repair, abstain, compose, or transfer.

### What This Adds Beyond I383

I383 attacks world complexity. I384 attacks the declared grammar artifact.

### Required Hardening

Add an executable grammar interpreter and score:

```text
raw action HFA from G_hat
invalidity F1
unsafe F1
obligation F1 for identifiable obligations
repair success
abstention precision/recall and coverage
composition HFA
sibling transfer HFA
```

Then freeze the learner-produced `G_hat` before held-out scoring.

### Verdict

```text
A GRAMMAR IR THAT IS NEVER EXECUTED IS A SCHEMA, NOT EVIDENCE.
```

## I385: Attack Geometry Erasure Directly

### Attack

The harness includes `REQUIRED_ABLATIONS` and verifies the roster length is 22. But `audit_ablation_roster()` also says:

```text
hidden_hfa_reported: False
```

There is no erasure executor, no component dependency map, no before/after functional metrics, no raw-vs-canonical comparison, and no proof that deleting `T_hat`, `O_hat`, `R_hat`, `A_hat`, or `C_hat` changes the right behavior.

### What This Adds Beyond I384

I384 says the grammar is not causal yet. I385 says the advertised causality test is not run.

### Required Hardening

Implement public smoke erasure before hidden opening:

```text
score(G_hat)
score(remove_T_hat)
score(remove_O_hat)
score(remove_R_hat)
score(remove_A_hat)
score(remove_C_hat)
score(bindings_only)
score(examples_counterexamples_only)
score(per_task_program_replacement)
score(MDL_library_replacement)
```

Each erasure must report the expected component drop and unaffected-component checks.

### Verdict

```text
AN ABLATION ROSTER IS NOT GEOMETRY ERASURE.
```

## I386: Attack Cost Ledger Honesty At Outcome Level

### Attack

The ledger now exists and charges the harness file, spec file, packet, and grammar nodes. That is progress over B44.

But the ledger does not yet recompute all-in costs for actual competing systems on the same tasks. Absorber witnesses return small local cost dictionaries, but there is no unified outcome ledger comparing WGD, PBE, CEGIS, active learning, MDL, schema binding, repair baselines, language prior, representation prior, and post-hoc compression under one target distribution.

The current ledger proves categories can be populated. It does not prove the 4x all-in absorber condition.

### What This Adds Beyond I385

I385 attacks causal erasure. I386 attacks cheapness and all-in comparison.

### Required Hardening

For every measured system and absorber, emit:

```text
G, B_i, P_i, E_i, C_i, Q_i, V_i, R_i, A_i, L, H, O, N
runtime
query count
adapter bits
human substrate charge
substrate_free_total
substrate_charged_total
metric achieved
cost to target threshold
ratio against WGD
```

### Verdict

```text
CATEGORY COVERAGE IS NOT 4X ALL-IN EVIDENCE.
```

## I387: Attack Query Accounting

### Attack

The top-level preimplementation audit sets `query_budget=0` for baseline views. Meanwhile several absorber witnesses internally use counterexamples, active queries, or oracle bits in their toy worlds.

That is acceptable for witness calibration, but it is not yet the frozen query ledger B43/B44 demanded. There is no target-run discovery mode, no per-system feedback log, no repair attempt accounting, and no distinction between passive public transcript learning and active query learning in a measured WGD run.

### What This Adds Beyond I386

I386 attacks static cost. I387 attacks dynamic information.

### Required Hardening

Create a query ledger object and make every learner/absorber use it:

```text
query_type
answer_bits
counterexample_bits
adaptive_or_frozen
received_by
hidden_distribution_touch
repair_attempt_count
abstention_probe_count
charged_category
```

### Verdict

```text
UNTIL QUERY INFORMATION IS CENTRALIZED, ACTIVE LEARNING CAN STILL HIDE IN THE WALLS.
```

## I388: Attack Leakage Audits That Are Declared But Not Run

### Attack

`run_predictive_leakage_audit()` computes several normalized-MI metrics and declares that high-order attacks are in the frozen roster:

```text
compression_classifier
program_feature_search
serializer_offset_probe
split_reconstruction_attack
feedback_sequence_predictor
```

But those high-order attacks are not implemented. The harness records the roster as a passing finding.

That is a direct B43/B44 failure mode: low-order leakage tests can pass while a clever public-feature classifier wins.

### What This Adds Beyond I387

I387 attacks intentional feedback. I388 attacks accidental public side channels.

### Required Hardening

Run the declared attacks, not only name them:

```text
boosted/tree adversarial predictor
program-feature synthesis predictor
compression classifier
serializer-offset probe
split-reconstruction attack
feedback-sequence predictor
permutation stability score
identifiability alternative search
```

Targets must include role map, binding labels, obligation class, repair location, abstention requirement, composition form, sibling template, and hidden query bucket.

### Verdict

```text
A DECLARED LEAKAGE ROSTER DOES NOT CATCH LEAKAGE.
```

## I389: Attack Sibling Independence As Behavior-Signature Accounting

### Attack

Sibling generation now exists, and the count function requires zero shared field IDs and feedback-signature Hamming distance at least 0.20. This is progress.

But the behavior signature is only the sequence of feedback labels from up to eight public transcript edits. It does not measure:

```text
template overlap
nuisance fingerprint similarity
binding reuse fraction
schema-position reuse
generator-family classifier shortcuts
reduced-sibling HFA
TD_after(G_hat, sibling)
AFTD_all_in
```

The function can count siblings as nonduplicate before proving they are reduced siblings in the spec's sense.

### What This Adds Beyond I388

I388 attacks leakage. I389 attacks transfer accounting.

### Required Hardening

Split sibling checks into:

```text
nonduplicate_sibling_count
reduced_sibling_count
behavioral_distance
template_overlap
nuisance_similarity
binding_reuse
TD_after_G_hat
TD_H0
AFTD_all_in
```

Only reduced nonduplicates should count toward signal.

### Verdict

```text
NON-DUPLICATE IS NOT THE SAME AS REDUCED.
```

## I390: Attack Composition Hostility As A Toy Probe

### Attack

`audit_composition_probes()` correctly includes noncommutation, guard conflict, interference, and preservation declarations. But the probe is a fixed integer toy:

```text
inc_then_clip
clip_then_inc
```

It is not integrated with the generated WGD worlds, hidden-style tasks, learner-produced grammar, PBE/CEGIS/MDL composition baselines, or repair preservation suites.

### What This Adds Beyond I389

I389 attacks sibling transfer. I390 attacks composition transfer.

### Required Hardening

Create held-out composition tasks in the WGD substrate and compare:

```text
G_composed_cost
sum(G_component_costs)
composition_HFA
order-error rate
guard-conflict rate
interference-failure rate
preserved component HFA
PBE/CEGIS/active/MDL composition costs
```

### Verdict

```text
A COMPOSITION TOY CAN WARN THE HARNESS. IT CANNOT PROVE COMPOSITION TRANSFER.
```

## I391: Attack Repair And Abstention As Reported Controls

### Attack

The harness has repair and abstention witnesses. The repair baseline mines a sum/range constraint and fixes one invalid case. The abstention baseline scores five cases using a distance/gap anomaly rule.

This proves the report fields can exist. It does not prove WGD repair is local, preserves prior behavior, or beats nearest-valid, CEGIS repair, active retry, and patch-library baselines under the same failures. It also does not prove abstention preserves usefulness across hidden family, query bucket, and composition form.

### What This Adds Beyond I390

I390 attacks composition. I391 attacks improvability and useful uncertainty.

### Required Hardening

Run actual repair and abstention suites:

```text
repair_without_feedback
repair_with_single_failure_case
repair_with_interactive_feedback_charged
changed_grammar_nodes
repair_patch_bits
preserved_behavior_HFA
nearest_valid_search
constraint_repair
CEGIS_repair
active_retry
risk_coverage_curve
coverage_by_bucket
false_abstention_cost
unsafe_false_negative_cost
calibrated_uncertainty_baseline
```

### Verdict

```text
REPAIR AND ABSTENTION WITNESSES ARE NOT REPAIR AND ABSTENTION MEASUREMENT.
```

## I392: Final B45 Adversarial Synthesis

### Attack

The harness now has many of the right nouns. It has them in runnable form. That matters. But the B43/B44 risk was never only missing nouns. It was a system that can pass a checklist while still failing to make boring explanations strong, equal, and expensive enough to lose meaningfully.

The current harness reduces one class of risk:

```text
The project no longer has zero WGD harness.
```

It leaves the central risk alive:

```text
The native absorbers are not yet hostile competitors on the WGD measurement distribution; geometry erasure is not yet causal; all-in accounting is not yet attached to full system outcomes.
```

### Required Next Gate

Before any hidden opening for signal, add a `wgd0_measurement_smoke` layer with:

1. A learner or baseline-produced executable `G_hat`.
2. An executable grammar interpreter and raw functional scoring.
3. Public smoke component erasures with expected metric drops.
4. Absorber adapters that actually run on WGD packets.
5. Absorber budget curves on the WGD substrate.
6. Unified cost/query ledgers per system outcome.
7. Implemented high-order leakage and permutation attacks.
8. Reduced sibling and `AFTD_all_in` accounting.
9. Held-out WGD composition tasks and composition baselines.
10. Repair/abstention suites with utility and preservation metrics.
11. Token assignment that treats any missing measurement layer as inconclusive or void, not as signal.

### Final Token

```text
WGD_HARNESS_CODE_REVIEW_COMPLETE_PREHIDDEN_SCAFFOLD_GREEN_NATIVE_ABSORBER_THEATER_NOT_YET_AVOIDED
```

### Final Position

WGD still serves the Vision better than FrameSeed because it asks for discovered geometry, improvability, data efficiency, inference efficiency, and cheap publicly inspectable intelligence. The current harness is a useful step toward that standard.

But a useful step is not the home run. The adversary is not asking for an audit object that can say all required absorbers exist. The adversary is asking for a measurement object where those absorbers are strong enough to kill WGD.

The present harness is allowed to continue as a pre-hidden integrity scaffold. It is not allowed to certify signal, and it is not yet enough to open a hidden seed with a won-over adversary.