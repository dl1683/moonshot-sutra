# Q-Loop Batch 36: Harness Review Against B35 Evidence Gate

**Date:** 2026-07-07
**Role:** Question-Loop worker
**Iterations:** I253-I266
**Status:** actual harness reviewed; B35 evidence gate not satisfied.

Two invariants held fixed:

1. Swing for the home run.
2. The loop only stops on a won-over adversary.

## Grounding

Required files read:

1. `research/dual_loop_supervisor_checkin_27.md`
2. `code/frameseed0_harness.py`
3. `code/test_frameseed0_harness.py`
4. `experiments/frameseed0_b27_audit.json`
5. `research/question_loop_batch35.md`
6. `research/frameseed_0_precommit_spec.md`

Verification rerun on this checkout:

```text
git rev-parse --short HEAD
2001664

$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'; python -m pytest code/test_frameseed0_harness.py
10 passed

python -c "import sys,json; sys.path.insert(0,'code'); import frameseed0_harness as h; ..."
{"failed": [], "finding_count": 29, "hidden_hfa_reported": false, "no_performance_runs": true, "packet_bits": 20496, "passed": true, "sibling_count": 2, "worst_mi": 0.004915049666568067}
```

The existing audit artifact `experiments/frameseed0_b27_audit.json` also reports
`passed: true`, `hidden_hfa_reported: false`, `no_performance_runs: true`, and
worst normalized MI `0.004915049666568067`.

## Summary Verdict

```text
THE B27 HARNESS PASSES ITS OWN PREIMPLEMENTATION AUDIT.
IT DOES NOT SATISFY THE B35 EVIDENCE REQUIREMENTS FOR HIDDEN SIGNAL MEASUREMENT.
```

This is not a current void token, because no hidden HFA was opened and no learner
signal was reported. It is a no-go for W28 hidden signal measurement under the
current evidence bundle.

The delivered harness is a useful scaffold: public-transcript constructor input,
canonical serialization, simple provenance checks, a support-swap sabotage
control, byte-level baseline packet parity, marginal MI audits, and token
precedence helpers. But B35 demanded a hostile-review evidence package. The
actual harness mostly proves that its audit booleans can go green; it does not
yet prove replayability, all-in cost accounting, semantic baseline fairness,
hidden-open sealing, representation noncontainment, AFTD transfer, or strongest
absorber readiness.

## B35 Checklist Result

| # | B35 evidence requirement | B36 result |
|---|---|---|
| 1 | Manifest-bound implementation commit and file list | Fail: commit exists in checkout, but manifest does not bind commit, file hashes, command lines, or raw artifacts. |
| 2 | Capability-isolated constructor input type | Partial pass: constructor accepts `PublicTranscript`, not `World`, but transcript production and sabotage evidence are too weak. |
| 3 | Replayable provenance with rejected candidate logging | Fail: provenance IDs exist, replay and rejected candidate logs do not. |
| 4 | Domain-separated RNG streams and rejection-safe seed handling | Partial: SHA domain separation exists, but stream use and seed metadata audits are incomplete. |
| 5 | End-to-end golden controls | Partial: token switchboard controls pass; end-to-end control bundles do not exist. |
| 6 | Field-by-field baseline executable parity matrix | Fail: same bytes are checked, executable semantics are not. |
| 7 | Independent report-budget and all-in-budget recomputation | Fail: packet bits are recomputed; all-in teacher/search/query costs are not. |
| 8 | Conditional MI, per-family leakage tests, adversarial classifiers | Fail: marginal MI only. |
| 9 | Accepted/rejected world distribution audit | Partial: no rejection path is visible, but decisive evidence selection and transcript curriculum are unaudited. |
| 10 | Packet and set order shuffle invariance with replay | Fail: reverse/rotate syntax checks only. |
| 11 | One-way hidden-open ledger with raw artifact hashes | Partial: manifest declares hidden unopened; no sealed ledger exists. |
| 12 | Frame/binding AFTD decomposition and fresh-sibling transfer | Fail: sibling IDs exist, no AFTD evidence exists. |
| 13 | Representation-prior cost geometry and primitive ablations | Fail: token state exists, no noncontainment audit exists. |
| 14 | Strongest-absorber posture | Fail: baseline names exist, absorber implementations do not. |

---

## I253: A Green Manifest Is Not A Bound Artifact Chain

### Most Dangerous Question

Can the harness report "frozen before hidden" without cryptographically or
procedurally binding what was frozen?

### Actual Evidence

`default_manifest()` declares a public seed, hidden seed rule, constructor ID,
serializer ID, scorer ID, baseline version strings, zero timeouts, and booleans
for `frozen_before_hidden=True` and `hidden_results_opened=False`. The audit
checks those booleans and checks that every baseline name has a version string.

The live checkout has a commit (`2001664`) and the audit JSON exists, but the
manifest itself does not include:

```text
implementation_commit
dirty_tree_status
file_hashes
files_under_review
audit_command_lines
stdout_stderr_hashes
raw_output_paths
dependency_versions
hidden_seed_commitment_hash
```

### Miss

B35 asked for manifest-bound implementation evidence. The harness has a manifest
object, not a manifest-bound artifact chain.

### Harder Attack

A future run could change `code/frameseed0_harness.py`, rerun the audit, and
still produce the same style of manifest declaration. A reviewer could not tell
which code, dependency state, or exact command produced the green JSON without
trusting the operator.

### Verdict

Fails B35 requirement #1. This blocks hidden signal measurement, but does not
void the current state because hidden HFA was not opened.

### Required Fix Before W28 Hidden Open

Write a manifest artifact that binds commit, dirty state, file hashes,
dependency versions, commands, outputs, audit JSON hash, and hidden-open status.
The manifest must be generated by code, not manually narrated.

---

## I254: Constructor Blindness Is Realer Than A Flag, But The Transcript Is Loaded

### Most Dangerous Question

Does the constructor avoid latent objects while still receiving a public
transcript that was selected by a teacher with the answer key?

### Actual Evidence

The constructor entry point is:

```text
BlindPacketConstructor.construct(transcript: PublicTranscript, rng: AuditedRandom)
```

That is materially better than `construct(world, blind=True)`. The constructor
does not accept `World`. `_assert_transcript_is_public()` rejects obvious latent
field names such as `slot_to_role`, `kernel_id`, `rho`, `beta`, `pi`, and
`seed_namespace`.

But `make_public_transcript(world)` is generated directly from `World`. It
enumerates every base row and every single-slot edit for two rounds per cell.
The constructor then selects the two slots with the highest label-changing edit
counts. This is public evidence, but it is also a complete causal-support
teaching transcript.

### Miss

B35 asked for capability isolation plus sabotage evidence:

```text
swap latent roles after PublicTranscript creation
constructor output must not change
scorer expected token must change or audit must detect mismatch
```

The test suite checks `constructor_mode == "blind"` and checks provenance. It
does not run the latent-role-swap noninterference test.

### Harder Attack

The constructor can be clean while the public transcript is an uncharged oracle
mask in slow motion. If the transcript already contains all single-slot
intervention labels, the support list is no longer a discovered frame. It is the
answer extracted from exhaustive public oracle access.

### Verdict

Partial pass for object shape. Fail for B35-level blindness evidence.

### Required Fix Before W28 Hidden Open

Persist the public transcript as an artifact with its query policy, oracle query
cost, and raw fact hashes. Add the latent-role-swap sabotage. Charge transcript
construction as teacher work, not just packet bytes.

---

## I255: Provenance Is Present, But It Is Still A Receipt

### Most Dangerous Question

Can the audit replay why each packet entry exists, or only check that each entry
points to some public fact?

### Actual Evidence

`audit_constructor_provenance()` verifies:

```text
CONSTRUCTOR_BLIND_MODE
CONSTRUCTOR_PROVENANCE_PRESENT
CONSTRUCTOR_SUPPORT_FROM_PROVENANCE
```

The support-swap sabotage rewrites `representation_patch.ast_or_schema.support_slots`
to inert slots and confirms the provenance audit rejects it. That is a useful
negative control.

### Miss

B35 required per-entry:

```text
entry_id
entry_hash
constructor_rule_id
allowed_input_hashes
constructor_rng_draws
rejected_candidate_count
tie_break_rule
declared_bit_cost
replay_status
```

The current audit does not replay packet construction from the transcript and
packet RNG. It does not record rejected candidates, tie-breaks, or rule IDs. It
does not show that provenance was emitted by the construction rule rather than
attached after selection.

### Harder Attack

The constructor scans many facts, selects support slots by label-change counts,
sorts ties by slot ID, chooses first examples by sorted support-pair values, and
then shuffles entries. None of the selection search is logged as teaching cost.
The provenance says which facts justify the final support, not how much search
was spent finding that support.

### Verdict

Fails B35 requirement #3.

### Required Fix Before W28 Hidden Open

Add a replay log and a replay auditor. The replay auditor must reconstruct the
packet byte-identically from transcript hash, constructor version, packet RNG
seed hash, and rule logs. It must also charge rejected candidate search.

---

## I256: Split RNG Streams Are Declared Even When They Are Unused

### Most Dangerous Question

Does the generator use independent streams, or does the audit count stream names
that were merely instantiated?

### Actual Evidence

`derive_seed(public_seed, purpose, namespace)` uses SHA-256 over
`public_seed|purpose|namespace`, which is a domain-separated seed derivation
shape. `RNG_PURPOSES` includes world structure, names, orientations, hidden
queries, packet construction, learner tie-breaks, baseline tie-breaks, and
ablations.

But the test and MI audit count purposes by record presence. A generated world
records all purposes even when several draw counts are zero. On this checkout:

```json
{"ablations":0,"baseline_tie_breaks":0,"hidden_queries":0,"learner_tie_breaks":0,"names":20,"orientations":20,"packet_construction":0,"world_structure":23}
```

### Miss

B35 asked for a seed contract that includes role permutation, surface names,
public queries, hidden queries, packet construction, learner/baseline
tie-breaks, ablations, golden controls, and MI dry-run streams. It also asked
for resampling isolation and seed-derived MI probes.

The harness has no hidden query generator, no role-permutation stream, no public
query stream, no golden-control stream, no resample counts, and no audit that
draw counts match actual use.

### Harder Attack

The finding `GENERATOR_RNG_STREAMS_SPLIT` currently proves "the record objects
exist," not "the stochastic decisions were isolated." That is a false comfort
line in the audit output.

### Verdict

Partial pass for seed derivation. Fail for B35 requirement #4 as evidence.

### Required Fix Before W28 Hidden Open

Make RNG stream records usage-sensitive: required streams must either have
expected draw-count ranges or be explicitly declared unused for this pre-hidden
phase. Add seed metadata MI probes and resampling isolation logs.

---

## I257: Golden Controls Exercise The Token Switchboard, Not The Experiment

### Most Dangerous Question

Do the golden controls prove the full harness can assign the right token, or
only that `assign_terminal_token()` branches in the expected order?

### Actual Evidence

`run_golden_token_controls()` covers:

```text
smuggling -> VOID
boolean escape failure -> BOOLEAN_TRAP
representation noncontainment failure -> REPRESENTATION_PRIOR
low L3 -> NEGATIVE
teaching dimension -> teaching absorption
library learning -> library absorption
nuisance oracle -> nuisance absorption
CEGIS before active and RAG -> CEGIS absorption
clean synthetic gates -> T3R_SIGNAL
```

The tests also check the helper directly.

### Miss

B35 required end-to-end controls:

```text
control_bundle -> audits -> baselines -> ablations -> token
```

The current controls construct `TokenEvidence` by hand. No golden world is run.
No baseline adapter is exercised. No broken budget ledger is converted to a
void token. No explicit oracle mask control proves absorption. No randomized
label world proves negative. Active-learning and RAG absorptions are not tested
as pure precedence cases.

### Harder Attack

The most dangerous line is the clean signal control. It proves that if every
signal gate is manually set to true, the switchboard emits `FRAMESEED_T3R_SIGNAL`.
It does not prove the harness can earn those gates without smuggling, prior
containment, or baseline handicapping.

### Verdict

Partial pass for token precedence. Fail for B35 requirement #5 as end-to-end
evidence.

### Required Fix Before W28 Hidden Open

Build golden artifacts that run through the same audit, baseline, budget,
ablation, and token assignment path as the hidden run. Include expected VOID,
absorption, negative, and a narrowly justified signal-shaped synthetic case.

---

## I258: Baseline Parity Is Byte Equality, Not Operational Fairness

### Most Dangerous Question

Can every baseline execute the packet semantics, or do they only receive the
same JSON?

### Actual Evidence

`make_baseline_views()` gives each named baseline a canonical JSON packet,
packet bit length, task bundle hash, and query budget. The audit checks:

```text
BASELINE_ALL_PRESENT
BASELINE_PACKET_HASH_PARITY
BASELINE_BUDGET_PARITY
```

The test suite confirms that removing `entries` for `l2_cegis` is detected.

### Miss

B35 required a field-by-field matrix:

```text
field_type x system -> {parsed, costed, executable, constraining, audited}
```

No adapter executes `example`, `counterexample`, `invariant`,
`representation_patch`, or `verifier_clause` semantics. No synthetic candidate
is evaluated across systems. No baseline inability is explicitly charged or
converted to non-parity.

### Harder Attack

The packet includes executable-looking fields such as `truth_table2`,
`set_complement`, `paired_edit_effect_support_v1`, and verifier clauses. Equal
bytes do not prove that TD-H0, CEGIS, RAG, nuisance-oracle, or library learning
can use those fields with the same force that L3 would.

### Verdict

Fails B35 requirement #6.

### Required Fix Before W28 Hidden Open

Implement baseline adapters or explicitly mark the run pre-adapter. Add
semantic equivalence tests for each packet field type and each baseline. Same
packet plus same synthetic query must produce the same constraint truth value or
a counted non-parity void.

---

## I259: The Budget Ledger Counts The Envelope, Not The Teacher

### Most Dangerous Question

Does the audit charge the work that produced the compact packet?

### Actual Evidence

`BudgetLedger` has fields for packet bits, oracle query bits, oracle answer
bits, final program bits, learned library bits, residual sibling teaching bits,
failed query bits, and verifier expansion bits. The audit checks only that
fields exist and that `packet_bits` matches canonical serialization.

In the top-level audit, the ledger is:

```text
BudgetLedger(packet_bits=packet_bit_length(packet))
```

Everything else is zero. Meanwhile the public transcript for `m=16` contains
168 facts and is generated by enumerating base rows plus every single-slot edit
for two rounds per cell.

### Miss

B35 required report budget and all-in budget, including candidate packet count,
rejected candidates, oracle queries requested/used, verifier expansion,
representation-patch bits, final program bits, macro bits, per-task binding
bits, runtime, search node count, cache events, and human intervention events.

### Harder Attack

This is the largest scientific gap. The harness can produce a 20,496-bit packet
only after an exhaustive public oracle transcript has already exposed which
slots change labels under intervention. If those 168 public facts and the
constructor's selection search are not charged, the packet length is an
underbill.

### Verdict

Fails B35 requirement #7. It also threatens teaching-dimension and
active-learning absorption in any future performance run.

### Required Fix Before W28 Hidden Open

Charge transcript construction, public oracle access, selection search,
candidate rejection, verifier expansion, adapter compilation, and runtime. Report
both precommitted budget and adversarial all-in budget.

---

## I260: The MI Audit Is Clean, But It Only Attacks Marginals

### Most Dangerous Question

Can a learner predict roles from public artifacts even if the reported MI probes
are below threshold?

### Actual Evidence

`run_generator_mi_audit()` reports low marginal normalized MI for:

```text
role_category vs slot_index
role_category vs name_prefix
role_category vs orientation
role_category vs kernel_id
sibling_index vs target_role_bucket
```

The audit passed at 10,000 worlds with worst metric
`0.004915049666568067`.

### Miss

B35 required conditional MI, per-family reports, adversarial role classifiers,
query distribution parity, packet-entry position probes, seed draw count probes,
and corrections across multiple tests.

The current MI audit does not test exact causal-pair membership, hidden family,
role-permutation variants, name rank/hash features beyond one hex character,
packet order, transcript query order, resample count, or classifier accuracy.

### Harder Attack

`role_kernel_id_nmi` is weak evidence because all slots in a given world share
the same kernel. `sibling_id_target_role_map_nmi` tests only sibling index
against a target bucket, not whether sibling artifacts leak target or sibling
roles. The audit can be numerically clean while missing the artifact features a
cheap learner would exploit.

### Verdict

Fails B35 requirement #8 as a leakage proof. Passes only the narrower B27
marginal MI check.

### Required Fix Before W28 Hidden Open

Add conditional leakage tables, adversarial role classifiers, packet/transcript
order probes, seed metadata probes, and per-family/per-`m` reports. The input to
the classifier must be every non-label artifact the constructor or learner can
see.

---

## I261: There Is No Rejection Filter, But There Is Still A Curriculum Filter

### Most Dangerous Question

Does the generator avoid accepted/rejected-world bias while the transcript
constructor silently curates decisive evidence?

### Actual Evidence

`generate_world()` chooses an admitted two-input kernel, shuffles `rho`, shuffles
roles, samples orientations and names, and returns the world. There is no
visible rejection loop. Because admitted kernels depend on both causal inputs,
`decisive_intervention_exists()` should usually be structurally guaranteed.

### Miss

B35 asked for accepted/rejected counts, rejection reasons, decisive row counts,
minimum decisive intervention length, decisive row position, role slot IDs,
kernel/rho/orientation summaries, and sibling generation attempts.

The harness has no accepted/rejected audit. More importantly, it has no audit of
the public transcript curriculum. The transcript gives every single-slot edit
label, so the decisive rows are not merely present; they are guaranteed to be in
front of the constructor.

### Harder Attack

The absence of rejection does not clear the evidence requirement. It moves the
confound from world acceptance into transcript policy. The teacher gives the
constructor the full single-edit intervention table, then the packet extracts
support slots from it. That can make a later signal look like frame transfer
when it is really active intervention labeling.

### Verdict

Partial pass for "no rejection loop found." Fail for B35 requirement #9 as a
distribution and curriculum audit.

### Required Fix Before W28 Hidden Open

Log transcript query policy, decisive evidence positions, query counts by type,
and sibling generation attempts. If exhaustive public intervention tables are
allowed, charge them and give every absorber the same advantage.

---

## I262: Packet Order Control Is Syntactic

### Most Dangerous Question

Does the order audit prove order cannot carry the frame, or only that reversing
entries preserves a multiset hash?

### Actual Evidence

`audit_packet_order_control()` checks:

```text
reversed packet has the same entry multiset hash
rotated packet has the same bit length
```

The constructor also shuffles packet entries with the packet RNG.

### Miss

B35 required semantic/nonsemantic order classification, multiple independent
post-construction shuffles, packet hash/bit/provenance replay, learner output,
baseline output, set-order shuffles, and checks that IDs or hashes do not
preserve pre-shuffle positions.

The current audit does not shuffle `support_slots`, set literals,
`obs_mask` ordering, examples inside a support pair, candidate listings, or
clause IDs. It does not run L3 or baselines under shuffles.

### Harder Attack

The payload still carries semantic order: `support_slots` is sorted by
label-change count and then slot ID; examples are selected by first occurrence
and emitted by sorted support-pair value before entry shuffle. Entry shuffle
does not erase those deeper order channels.

### Verdict

Fails B35 requirement #10.

### Required Fix Before W28 Hidden Open

Define semantic and nonsemantic order. For nonsemantic order, run multiple
shuffles through provenance replay, L3, and every baseline adapter. For semantic
order, charge it and test whether it predicts roles.

---

## I263: Hidden-Open Discipline Is A Boolean Declaration, Not A Seal

### Most Dangerous Question

Can hidden results be opened, debugged, or rerun without a ledger proving the
one-way event?

### Actual Evidence

The harness manifest declares:

```text
hidden_seed_rule = "sha256(public_seed|hidden|unopened-until-freeze)"
frozen_before_hidden = true
hidden_results_opened = false
post_hidden_changes = ()
```

The top-level audit reports:

```text
no_performance_runs = true
hidden_hfa_reported = false
```

### Miss

B35 required a hidden-open ledger with timestamp, commit, manifest hash, command
line, stdout/stderr hash, raw artifact hash, token result, and seed rotation
rules after bug fixes.

The current harness does not generate or enforce such a ledger. It does not
commit to dependency versions, raw hidden artifact storage, temp-file policy, or
post-open immutability.

### Harder Attack

The current state is safe only because hidden performance has not been run. The
moment W28 opens hidden HFA, a boolean `hidden_results_opened=False` in a default
manifest is no longer evidence. It is just a field someone can set.

### Verdict

Partial pass for no hidden results so far. Fail for B35 requirement #11 before
any hidden open.

### Required Fix Before W28 Hidden Open

Create a one-way hidden-open command that writes an append-only ledger with
commit, manifest hash, raw artifact hashes, dependency versions, command line,
and token result. Any post-open implementation change under the same hidden
seeds must force a void or seed rotation.

---

## I264: Sibling Transfer Exists As IDs, Not As AFTD Evidence

### Most Dangerous Question

Does the harness prove reusable frame transfer, or only generate two sibling
world IDs?

### Actual Evidence

`generate_sibling_worlds()` creates two sibling worlds and forces a different
kernel if the sampled kernel matches the target. `TaskBundle` contains target
and sibling IDs. The top-level audit reports `sibling_count = 2`.

The packet is constructed only from the target transcript, not from sibling
labels. That avoids one obvious "packed every sibling answer" failure.

### Miss

B35 required:

```text
F_reusable_frame_bits
B_target_binding_bits
B_sibling_1_binding_bits
B_sibling_2_binding_bits
per_task_kernel_or_program_bits
per_task_public_label_bits
shared_vs_task_specific_provenance
fresh sibling withheld from packet construction
erasure and swap tests
```

None of that exists. There is no L3, no sibling HFA, no AFTD, no residual
teaching budget, and no fresh-sibling transfer test.

### Harder Attack

Sibling IDs in a task bundle are not transfer evidence. They are placeholders.
Without frame/binding separation, any future improvement could be target-specific
support selection, teaching dimension, library learning, or active intervention
search.

### Verdict

Fails B35 requirement #12.

### Required Fix Before W28 Hidden Open

Implement AFTD decomposition before performance reporting. Include erasure,
binding swap, and fresh-sibling tests. Report both report-budget AFTD and all-in
AFTD.

---

## I265: Representation Noncontainment Is A Token Flag, Not A Certificate

### Most Dangerous Question

Does the harness prove L0 lacks the frame, or only provide a token branch for
when someone later says it lacks the frame?

### Actual Evidence

`TokenEvidence` includes:

```text
representation_noncontainment_passed
representation_prior_absorbed
```

The golden token controls check representation-prior precedence. The packet
serialization audit scans for banned role words in executable payloads.

### Miss

B35 and the hardened spec required a cost-prior audit:

```text
prior-only search over R0/H0
rank of true causal-support family
cost gap to nearest alias/support alternative
tie-break dependency on slot permutation
randomized-label success
role-permutation success
primitive ablations for truth_table2, set_complement, edited_value, etc.
```

No L0, H0, A0, B0, prior-only search, or primitive ablation exists.

### Harder Attack

The packet uses exactly the dangerous generic-looking primitives: `truth_table2`,
`set_complement`, `paired_edit_effect_support_v1`, single-set interventions, and
finite verifier clauses. If those are free or native in L0/L3, the frame may
already be installed by the representation language rather than the packet.

### Verdict

Fails B35 requirement #13.

### Required Fix Before W28 Hidden Open

Define L0/R0/H0/A0/B0 in executable code. Run prior-only searches and primitive
ablations before hidden HFA. If the true frame is cheap before the packet, emit
representation-prior absorption.

---

## I266: The Absorbers Are Named But Not Armed

### Most Dangerous Question

Does the harness try to make teaching dimension, CEGIS, active learning,
nuisance-oracle, RAG, and library learning win if they can?

### Actual Evidence

The harness names the baselines:

```text
l3_full
td_h0
l0_rotenn
l1_active
l2_cegis
rag
nuisance_oracle
library_learning
```

It also defines terminal-token precedence for teaching dimension, library
learning, nuisance oracle, CEGIS, active learning, and RAG. Tests confirm some
names and precedence cases.

### Miss

There are no absorber implementations. There are no strongest-known variants,
budget curves at 1x/2x/4x, timeout parity tests, query distribution parity
tests, near-miss diagnostics, or golden absorption cases that run real baselines.
Manifest timeouts are all zero, which is acceptable only because no performance
run occurs.

### Harder Attack

The current harness can prevent a few kinds of smuggling, but it cannot assign
the most important non-signal tokens. If CEGIS, teaching dimension, active
learning, nuisance-oracle, RAG, or library learning would absorb the effect, this
harness has no machinery to let them do it.

### Verdict

Fails B35 requirement #14. This is the final adversary's veto.

### Required Fix Before W28 Hidden Open

Implement the absorbers before hidden signal measurement. For each absorber,
document the strongest known variant, budget ratios, executable packet parity,
query parity, and near-miss diagnostics. The harness must welcome absorption.

---

## Final Recommendation

```text
NO-GO FOR HIDDEN SIGNAL MEASUREMENT UNDER THE CURRENT HARNESS.
```

What B36 approves:

```text
The B27 code is a clean preimplementation audit scaffold.
Its 10 tests pass when pytest plugin autoload is disabled.
Its in-memory 10,000-world audit passes with worst normalized MI 0.004915049666568067.
It correctly reports no hidden HFA and no performance run.
```

What B36 does not approve:

```text
Using this evidence bundle to open hidden HFA or claim readiness for
FRAMESEED_T3R_SIGNAL measurement.
```

Before W28 opens hidden results, the project must add:

1. Manifest-bound artifact chain.
2. Transcript/query-policy charging and latent-role-swap noninterference.
3. Byte-identical constructor replay with rejected candidate logging.
4. Usage-sensitive RNG and seed metadata audits.
5. End-to-end golden control bundles.
6. Semantic baseline adapter parity.
7. Report and all-in budget ledgers.
8. Conditional MI and leakage classifiers.
9. Transcript curriculum and distribution audits.
10. Semantic/nonsemantic order invariance tests.
11. One-way hidden-open ledger.
12. AFTD frame/binding decomposition and fresh-sibling tests.
13. Representation-prior cost-geometry audit.
14. Strong absorber implementations.

If W28 runs hidden HFA before those are in place, B36's adversary is not won
over. The correct response would be:

```text
FRAMESEED_T3_VOID_SMUGGLED_FRAME
```

Current token recommendation:

```text
NO TOKEN YET: PRE-HIDDEN HARNESS SCAFFOLD PASSED, B35 HIDDEN-RUN GATE FAILED.
```