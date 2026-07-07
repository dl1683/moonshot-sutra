# Q-Loop Batch 35: Implementation Oversight

**Date:** 2026-07-07
**Role:** Question Loop worker
**Iterations:** I239-I252
**Status:** implementation oversight gate; live checkout has no W27 implementation artifact yet; this batch attacks the harness shape W27 must satisfy before any hidden result can be interpreted.

---

## Grounding

Read in required order:

1. `research/dual_loop_supervisor_checkin_26.md`
2. `research/question_loop_batch34.md`
3. `research/frameseed_0_precommit_spec.md`
4. `research/VISION.md`

Binding invariants:

1. Swing for the home run. Paradigm-shifting or nothing.
2. The loop only stops on a won-over adversary.

Live checkout check:

```text
latest commit observed: 0d38631 supervisor check-in #26: conditional go to implementation, harness first
git status observed: master ahead of origin; no unstaged W27 implementation file
research/ contains no work_loop_batch27.md at this checkpoint
code/ contains no FrameSeed-specific harness file at this checkpoint
```

Therefore this batch cannot honestly certify W27 behavior. The first oversight finding is that W27 is not yet visible in the live repo. The correct move is not to invent a review target. The correct move is to turn B34's implementation warnings into sharper executable traps for the actual W27 harness.

## Summary Verdict

```text
NO W27 IMPLEMENTATION ARTIFACT WAS PRESENT IN THE LIVE CHECKOUT.
Q35 THEREFORE ISSUES A HARD PRE-RUN OVERSIGHT SPEC, NOT A PASS.
```

The harness must prove four things before any learner result can matter:

1. The constructor cannot see, infer, cache, or be parameterized by latent roles, hidden labels, hidden family ids, target kernels, or sibling role maps.
2. The generator, serializer, scorer, and audits are independently reconstructible from raw artifacts.
3. Every baseline receives executable parity, not just nominal access to the same packet.
4. Golden controls are decisive enough to fail in the right ways, not just produce a comforting green audit line.

Conditional gate:

```text
W27 may proceed only as a harness and audit implementation.
If W27 reports hidden HFA, tunes packet templates, or edits constructor/scorer/
baseline adapters after opening hidden results, emit FRAMESEED_T3_VOID_SMUGGLED_FRAME.
```

---

## I239: Phantom Implementation Is Itself A Confound

### Single Most Dangerous Question

Can the project create the social impression that W27 has been monitored before there is a reviewable W27 artifact?

### Attack

The live repo currently has the hardened spec, B34 implementation review, and supervisor #26. It does not show a W27 harness implementation file or a FrameSeed code module. That matters. Oversight can smuggle confidence just as easily as code can smuggle a frame.

The failure mode:

```text
Supervisor says "W27 begins implementation."
Q-loop writes as if it reviewed W27 implementation.
Later W27 code appears and inherits a false sense of clearance.
```

That is not just clerical. It corrupts provenance. The first scientific artifact is supposed to be proof that the harness cannot cheat. If the oversight artifact overclaims what it has seen, the review chain itself has already cheated.

### What The Harness Might Miss

The harness can pass local tests while the review record fails at the process layer:

1. no commit hash for the W27 code under review;
2. no manifest hash for generator, constructor, scorer, baseline adapters, and audit scripts;
3. no raw artifact bundle path;
4. no distinction between planned tests and executed tests;
5. no one-way hidden-open event logged.

### Required Evidence Before Q-Loop Can Say "Monitored"

W27 must provide:

```text
implementation_commit
manifest_hash
files_under_review
audit_command_lines
raw_output_paths
hidden_seed_status = unopened | opened_once | rotated_after_void
golden_control_results
baseline_parity_results
constructor_provenance_sample
```

No evidence, no clearance.

### Verdict + Kill Records

```text
OVERSIGHT CANNOT BE RETROACTIVE.
```

Kill records:

```text
KR-I239-1: If W27 lacks a manifest-bound implementation commit, Q-loop may not
assign any implementation pass.

KR-I239-2: If a later report treats this B35 file as validation of code that did
not exist in the checkout, void the review chain and restart oversight from the
actual W27 commit.

KR-I239-3: If hidden results are opened before this manifest exists, emit
FRAMESEED_T3_VOID_SMUGGLED_FRAME.
```

### NARRATIVE ATTACK

**Strongest process-rigging dismissal:** You claimed an adversarial harness review before there was a harness to review.

**What the result needs to be:** every oversight claim must bind to a commit, command, artifact, and hidden-open state.

---

## I240: Constructor Blindness Must Be Capability Isolation, Not A Flag

### Single Most Dangerous Question

Does "blind mode" physically prevent latent-role access, or is it just a boolean argument passed to code that still holds the world object?

### Attack

The obvious bad implementation is:

```text
construct_packet(world, blind=True)
```

where `world` still contains `pi`, `rho`, `beta`, causal roles, alias roles, hidden family id, sibling maps, and labels. A flag does not create noninterference. It creates a promise.

True constructor blindness requires an object-capability boundary. The constructor should receive only a public transcript type that cannot represent latent metadata. If the public transcript is built from a world object, the boundary must be tested by sabotage and serialization checks.

### Harness-Missing Path

Static source scans for banned strings are insufficient. The constructor can learn roles through:

1. precomputed public examples selected by an answer-aware generator;
2. row order;
3. sibling ordering;
4. object ids or repr strings;
5. cached helper fields;
6. "public" decisive interventions chosen using the latent map;
7. schema fields that include role-correlated names or slots;
8. exception messages or debug metadata.

The constructor can be formally blind to `world.roles` while still being fed an answer-aware public transcript.

### Required Implementation Shape

W27 must split types:

```text
LatentWorld: generator/scorer only
PublicTranscript: constructor and learners only
ScoringBundle: scorer only
AuditBundle: independent recomputation only
```

The constructor entrypoint must accept `PublicTranscript` and a packet-construction RNG stream only. It must not import or close over the generator module's latent map.

Minimum sabotage:

```text
1. swap causal and alias roles inside LatentWorld after PublicTranscript creation;
2. packet constructor output must not change;
3. scorer expected token must change or audit must detect mismatch;
4. if constructor output tracks the latent swap, blindness failed.
```

### Verdict + Kill Records

```text
BLINDNESS IS AN ACCESS-CONTROL PROPERTY, NOT AN INTENTION.
```

Kill records:

```text
KR-I240-1: If constructor accepts LatentWorld or any object containing role maps,
void, even if `blind=True` is set.

KR-I240-2: If public transcripts are answer-selected without charging the
selection rule and query cost, void or absorb by teaching dimension.

KR-I240-3: If sabotage role swaps change constructor output before scoring,
emit FRAMESEED_T3_VOID_SMUGGLED_FRAME.
```

### NARRATIVE ATTACK

**Strongest constructor-laundering dismissal:** The code wore a blindfold while holding the answer key in its hand.

**What the result needs to be:** capability isolation, sabotage evidence, and provenance for every public fact that reaches the constructor.

---

## I241: Provenance Logs Can Become Decorative Receipts

### Single Most Dangerous Question

Do provenance logs explain why each packet entry exists, or do they merely label entries after the constructor has already chosen the answer?

### Attack

A provenance field like:

```json
{"source": "public_transcript", "entry": 7}
```

is not enough. It proves an entry came from something public. It does not prove the public fact caused the entry by a frozen rule. The smuggling channel is the unlogged selection function.

The constructor may choose:

1. the decisive intervention among many;
2. the specific counterexample that kills alias support;
3. the invariant that names the complement of the causal pair;
4. the representation patch that makes the intended search cheap;
5. the order that puts causal slots first.

A provenance log that points to public rows after these choices is a receipt, not an audit.

### Harness-Missing Path

Unit tests can check "every packet entry has provenance" while missing whether provenance is complete. The missing object is a minimal sufficient cause:

```text
entry = f(frozen_public_inputs, frozen_constructor_state, constructor_rng)
```

If the log cannot replay `f`, the provenance is decorative.

### Required Evidence

For every packet entry:

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

The audit must replay packet construction from the public transcript and the packet-construction RNG stream. Replayed packet hashes must match.

### New Confound

Rejected candidates matter. If the constructor scans many public interventions and emits only the one that kills the alias hypothesis, the search cost is part of teaching. If rejected candidates are not logged and charged, the packet receives free active learning.

### Verdict + Kill Records

```text
PROVENANCE WITHOUT REPLAY IS STORYTELLING.
```

Kill records:

```text
KR-I241-1: If packet construction cannot be replayed byte-identically from
logged public inputs and RNG draws, no signal token is valid.

KR-I241-2: If rejected candidate search is omitted from provenance and budget
ledgers, absorb by active learning or void for hidden constructor selection.

KR-I241-3: If provenance is attached after construction rather than emitted by
the construction rule, treat it as non-evidence.
```

### NARRATIVE ATTACK

**Strongest audit-theater dismissal:** Every answer-key line had a receipt, but the receipt did not say who picked the line or why.

**What the result needs to be:** replayable, minimal-cause provenance with charged selection search.

---

## I242: RNG Stream Separation Can Still Leak Through Seed Derivation

### Single Most Dangerous Question

Are RNG streams truly independent, or are they deterministic siblings of a seed schedule that encodes roles, family ids, or iteration order?

### Attack

B34 required split RNG streams. That is necessary but weak. A naive implementation will do:

```text
seed_world = base_seed + i
seed_names = base_seed + i + 1
seed_orientation = base_seed + i + 2
seed_packet = base_seed + i + 3
```

This creates deterministic coupling across world index, hidden family, role permutation, sibling id, and packet order. Even if the PRNG is good, the seed schedule can leak structure because implementation code often assigns `i` by family or role condition.

### Harness-Missing Path

MI tests over generated outcomes may pass at small sample size while seed metadata remains predictive. A learner or constructor need not see the seed directly if seed-derived artifacts correlate with:

1. slot index;
2. name length or lexical bucket;
3. orientation parity;
4. packet example order;
5. sibling ordering;
6. kernel split order;
7. intervention row order.

The generator might also resample invalid worlds. If resampling consumes variable RNG draws, downstream streams can reveal whether earlier hidden structure was hard or easy.

### Required Seed Contract

W27 must freeze a domain-separated derivation:

```text
stream_seed = HMAC(precommit_seed, purpose || world_id || sibling_id || role_perm_id)
```

Purposes must include at least:

```text
world_structure
role_permutation
surface_names
orientations
public_queries
hidden_queries
packet_construction
learner_tiebreaks
baseline_tiebreaks
ablations
golden_controls
mi_dry_run
```

Invalid-world resampling must not share RNG state with any public or packet-facing stream. Resampling count must be logged and tested for correlation with latent role patterns.

### Required MI Expansion

B34's MI tests must include seed-derived fields:

```text
MI(resample_count, latent_role_layout)
MI(world_id_mod_bucket, kernel_id)
MI(sibling_order, target_role_map)
MI(packet_rng_draw_count, emitted_support)
MI(name_hash_bucket, latent_role)
MI(orientation_vector, causal_slot_index)
```

### Verdict + Kill Records

```text
SPLIT STREAMS DO NOT HELP IF THE SEED SCHEDULE IS A ROLE CHANNEL.
```

Kill records:

```text
KR-I242-1: If RNG streams are derived by arithmetic offsets from structured ids,
reject the harness until domain-separated seed derivation is implemented.

KR-I242-2: If invalid-world resampling affects packet, name, hidden-query, or
tie-break streams, void the run.

KR-I242-3: If resample counts, seed buckets, or RNG draw counts predict latent
roles or hidden family above threshold, emit FRAMESEED_T3_VOID_SMUGGLED_FRAME.
```

### NARRATIVE ATTACK

**Strongest seed-leak dismissal:** The answer was not in the names; it was in the calendar by which the names were drawn.

**What the result needs to be:** domain-separated seeds, independent streams after rejection sampling, and MI tests over seed metadata, not only visible names.

---

## I243: Golden Controls Must Assign Tokens, Not Just Pass Assertions

### Single Most Dangerous Question

Do golden controls force the harness to emit the correct terminal token under known worlds, or do they only test local helper functions?

### Attack

The directive says "Scorer and terminal-token assignment on golden controls." The dangerous implementation is a suite of assertions like:

```text
negative_control_accuracy_low == true
positive_control_accuracy_high == true
```

That does not test the claim machinery. The real object is token assignment under full audit:

```text
control_bundle -> audits -> baselines -> ablations -> token
```

Golden controls must prove the token logic can say signal, absorbed, void, and negative in the right circumstances.

### Required Golden Control Set

Minimum controls:

| Control | Construction | Required token |
|---|---|---|
| No causal/spurious distinction | aliases and causal slots never diverge under interventions | `FRAMESEED_T3_NEGATIVE` or no-decisive-world rejection before hidden use |
| Explicit oracle causal mask | packet contains counted oracle mask and baselines also receive it | nuisance-oracle or teaching-dimension absorption unless all costs justify T3-R |
| Constructor role access | constructor intentionally receives latent role map | `FRAMESEED_T3_VOID_SMUGGLED_FRAME` |
| Baseline denied executable field | L3 executes patch but baseline gets inert text | `FRAMESEED_T3_VOID_SMUGGLED_FRAME` |
| Randomized labels | labels independent fair bits | no system above 0.60 HFA; otherwise void |
| Full program shortcut | packet contains target program | CEGIS/RAG/library/teaching absorption, not T3-R signal |
| Broken budget ledger | final program bits omitted | void |
| Role-order leakage | causal role always low slots | void by MI or role permutation |

### Harness-Missing Path

Golden positives are especially dangerous. A positive control with an explicit oracle causal mask can train the team to expect `FRAMESEED_T3R_SIGNAL`, but by the spec the oracle mask is often an absorption or counted-answer condition. The control should verify humility, not reward a rigged success.

### Verdict + Kill Records

```text
GOLDEN CONTROLS MUST PROVE THE HARNESS CAN KILL ITS FAVORITE RESULT.
```

Kill records:

```text
KR-I243-1: If golden controls do not exercise terminal-token assignment end to
end, they do not satisfy W27.

KR-I243-2: If an explicit oracle-mask control emits T3R signal without charging
and baseline parity, the scorer is invalid.

KR-I243-3: If no golden control is expected to emit VOID, the smuggling audit has
not been tested.
```

### NARRATIVE ATTACK

**Strongest fake-control dismissal:** The controls proved arithmetic helpers worked, not that the experiment could tell signal from cheating.

**What the result needs to be:** adversarial golden worlds whose correct outputs include void, absorption, negative, and only a narrowly justified signal.

---

## I244: Baseline Adapter Parity Must Be Semantic, Not Serialization Equality

### Single Most Dangerous Question

Do baselines receive an executable equivalent of every field L3 uses, or do they receive the same bytes in a weaker interface?

### Attack

"Same packet entries" can still be unfair. L3 may receive FIR-0 ASTs and executable verifier clauses while a baseline receives canonical text. The report can truthfully say the field was provided, but the baseline's adapter makes it inert.

The parity condition is not:

```text
baseline_packet_bytes == l3_packet_bytes
```

It is:

```text
for every packet fact used by L3, baseline can either execute the same semantics
inside its declared model class or the missing capability is charged as part of
the baseline's model limitation and reported as non-parity.
```

Non-parity is a void path, not a footnote.

### Harness-Missing Path

Adapter tests often compare round-trip parse success. They miss whether the parsed object is operationally usable:

1. invariant clauses parse but cannot constrain baseline search;
2. representation patches are visible but opaque;
3. verifier clauses are checked for syntax but not available to CEGIS;
4. counterexamples include `actual_label` fields that only L3 reads;
5. compact programs are executable in L3 but untrusted in RAG/CEGIS;
6. macros count for L3 but not library learner;
7. hidden sibling tasks are exposed to L3 constructor but not to TD-H0/library.

### Required Parity Tests

For each packet field type:

```text
example
counterexample
invariant
transform
verifier_clause
representation_patch
program
```

W27 must provide a parity matrix:

```text
field_type x system -> {parsed, costed, executable, constraining, audited}
```

Then run executable equivalence tests:

```text
same public packet + same query -> same constraint truth value
same verifier clause -> same pass/fail on synthetic candidate
same compact program -> same output or explicit counted inability
same patch -> same transformed candidate set or explicit non-parity void
```

### Verdict + Kill Records

```text
FAIR BASELINES NEED EXECUTABLE PARITY, NOT ACCESS THEATER.
```

Kill records:

```text
KR-I244-1: If L3 executes any packet field that a baseline only stores or sees
as inert text, emit FRAMESEED_T3_VOID_SMUGGLED_FRAME.

KR-I244-2: If adapter tests do not include semantic equivalence on synthetic
candidates and queries, baseline parity is unproven.

KR-I244-3: If sibling-task access differs across L3, TD-H0, CEGIS, nuisance
oracle, RAG, or library learning, no T3-R token is valid.
```

### NARRATIVE ATTACK

**Strongest baseline-handicap dismissal:** Everyone got the same envelope, but only the favored learner got a key to open it.

**What the result needs to be:** a field-by-field operational parity matrix and adversarial equivalence tests.

---

## I245: Budget Accounting Must Charge Search, Rejections, And Compilation

### Single Most Dangerous Question

Does the ledger count only emitted packet bits, or does it count the hidden work used to choose and operationalize those bits?

### Attack

The spec counts packet bits, query bits, final program bits, library/macro bits, verifier-clause bits, and frame-patch bits. That is still incomplete if implementation lets free work happen off-ledger.

Off-ledger work can include:

1. constructor search over candidate packets;
2. invalid-world rejection and decisive-row mining;
3. public query selection;
4. verifier expansion into many finite obligations;
5. AST normalization and simplification;
6. baseline adapter compilation;
7. failed CEGIS candidates;
8. cached candidate pools from seen-family tuning;
9. manual threshold changes after seeing runs;
10. library wake/sleep pretraining across siblings.

If L3 receives a polished compact patch after the constructor searched thousands of candidates, the packet length is not the teaching cost. It is the receipt after free optimization.

### Harness-Missing Path

An independent bit recomputation from serialized packet bytes will pass while free constructor search remains invisible. The ledger must include raw logs from the process that produced the bytes, not only the bytes.

### Required Ledger

W27 must log and recompute:

```text
emitted_packet_bits
candidate_packet_count
rejected_candidate_packet_bits
oracle_queries_requested
oracle_queries_used
failed_oracle_queries
verifier_clause_bits
verifier_expanded_obligation_bits
representation_patch_bits
final_program_bits
library_macro_bits
per_task_binding_bits
adapter_compilation_bits_or_declared_free_public_code
runtime_budget
search_node_count
cache_hits
cache_misses
human_intervention_events
```

There must be two views:

```text
report_budget = precommitted metric
all_in_budget = everything the adversary will charge
```

AFTD must be reported under both. A signal that survives only under report budget is not persuasive.

### Verdict + Kill Records

```text
BIT COUNTING WITHOUT SEARCH COUNTING IS A FREE-TEACHER LOOPHOLE.
```

Kill records:

```text
KR-I245-1: If constructor candidate search, rejected interventions, verifier
expansion, final programs, and macro invention are not logged, no AFTD claim is
valid.

KR-I245-2: If all-in AFTD collapses into teaching dimension or library learning,
emit the corresponding absorption token.

KR-I245-3: If the budget ledger cannot be recomputed from raw logs by an
independent script, the token is not assignable.
```

### NARRATIVE ATTACK

**Strongest accounting dismissal:** The packet was short only because the expensive teacher thinking happened off the invoice.

**What the result needs to be:** report budget, all-in budget, independent recomputation, and absorption when the all-in cost destroys the claimed separation.

---

## I246: MI Tests Must Attack Conditional Leakage, Not Just Marginals

### Single Most Dangerous Question

Do generator MI tests prove no role leakage, or only no obvious marginal leakage?

### Attack

B34 asked for MI over at least 10,000 dry-run worlds. That is a good start, but marginal tests can miss conditional leakage. The leak may be:

```text
role independent of name overall
role predictable from name given m, kernel class, sibling id, or orientation bucket
```

or:

```text
slot index independent of role overall
causal pair has lower ordered pair rank within hidden family H3
```

The adversary will not stop at one-dimensional MI.

### Harness-Missing Path

MI audit can be underpowered or misdirected:

1. 10,000 worlds is too small for rare high-confidence leaks in `m=256`;
2. aggregate MI hides family-specific leaks;
3. permutation variants generated by the same biased process mask bias;
4. p-values are not corrected across many probes;
5. name hashes are tested but name statistics used by serializers are not;
6. packet order is tested before ablation shuffling, not after final construction;
7. hidden query distribution is not tested against constructor query distribution.

### Required Conditional Audit

For each role-sensitive target:

```text
latent_role
causal_pair_membership
alias_pair_membership
kernel_id
hidden_family
sibling_task_id
```

test predictors over:

```text
slot_index
slot_index_rank_features
name_hash_bucket
name_length
name_lexicographic_rank
orientation_bit
packet_entry_position
example_order
intervention_order
resample_count
seed_draw_count
query_type_distribution
```

Use both MI and adversarial classifiers. Report per-family and per-`m` results, not only aggregate means.

### Stronger Test

Train a cheap leakage classifier on dry-run artifacts:

```text
input: all non-label public artifacts the constructor or learner could see
target: latent role class
require: no classifier above precommitted chance tolerance after correction
```

If a classifier predicts roles from artifacts, void even if pairwise MI looks small.

### Verdict + Kill Records

```text
MARGINAL MI CAN MISS THE EXACT LEAK A LEARNER WOULD USE.
```

Kill records:

```text
KR-I246-1: If leakage audits are aggregate-only rather than per-family and
conditional, generator cleanliness is unproven.

KR-I246-2: If an adversarial classifier predicts role or family metadata from
public artifacts above threshold, void.

KR-I246-3: If hidden query distributions overlap constructor-selected query
templates more than baseline query templates, flag transfer/familiarity
confounding before any HFA interpretation.
```

### NARRATIVE ATTACK

**Strongest statistics dismissal:** The leak was invisible to your summary statistic and obvious to a classifier.

**What the result needs to be:** conditional MI, role-prediction probes, query-distribution parity, and family-specific leakage reports.

---

## I247: Decisive-Intervention Generation Can Make Hidden Worlds Too Easy

### Single Most Dangerous Question

Does the generator condition hidden worlds on having clean decisive interventions in a way that makes the target easier than the declared distribution?

### Attack

The spec requires every generated world to contain at least one decisive intervention row where causal and alias hypotheses diverge. That requirement is scientifically necessary. It is also a selection mechanism.

If W27 repeatedly samples worlds until decisive interventions are easy, short, or early in the public transcript, the hidden distribution is no longer the declared Boolean world. It becomes:

```text
worlds whose decisive evidence is easy for the constructor to find and package
```

That can create a false T3-R gap because the packet constructor sees a curated world family while baselines are judged as if the distribution were generic.

### Harness-Missing Path

A generator test may assert "decisive intervention exists" and miss:

1. how many worlds were rejected;
2. whether rejection rate depends on kernel, alias map, role placement, or `m`;
3. whether accepted decisive rows are shorter or more canonical;
4. whether accepted worlds favor causal slots with lower ids;
5. whether public transcript examples overrepresent decisive rows;
6. whether sibling tasks are accepted only when they share target-easy structure.

### Required Audit

W27 must log:

```text
raw_candidate_world_count
accepted_world_count
rejection_reason
decisive_row_count_per_world
minimum_decisive_intervention_length
decisive_row_position_in_public_transcript
decisive_role_slot_ids
kernel_id
rho_class
orientation_vector_summary
m
sibling_generation_attempts
```

Then compare accepted vs rejected candidates on all role-correlated fields. If accepted worlds are easier or role-biased, either redefine the distribution in the manifest and rerun all baselines on it, or void.

### New Requirement

Baselines must get the same distributional advantage. If the packet constructor benefits from curated decisive rows, TD-H0, L1, L2, RAG, nuisance-oracle, and library baselines must receive access to the same curated public transcript and the same candidate-selection cost.

### Verdict + Kill Records

```text
DECISIVE-WORLD FILTERING CAN BECOME ANSWER-AWARE CURRICULUM.
```

Kill records:

```text
KR-I247-1: If invalid-world rejection rates or decisive-row positions correlate
with roles, kernels, or hidden families, void until distribution is redefined and
audited.

KR-I247-2: If constructor receives curated decisive rows while baselines pay to
find them, emit baseline parity void or active-learning absorption.

KR-I247-3: If sibling tasks are filtered into target clones under a different
surface name, no AFTD claim is valid.
```

### NARRATIVE ATTACK

**Strongest distribution-rigging dismissal:** The world generator quietly selected the worlds where your lesson would look wise.

**What the result needs to be:** rejection logs, accepted/rejected distribution audits, and baseline-equivalent access to curated evidence.

---

## I248: Packet Order Shuffling Must Preserve Semantics And Costs

### Single Most Dangerous Question

Does packet shuffling remove covert order channels without changing the semantic packet or the budget ledger?

### Attack

B34 said to shuffle packet order, set literal order, examples, counterexamples, invariants, and verifier clauses. That is necessary, but shuffling can create two opposite failures:

1. if performance changes, order was a covert channel or learner prior;
2. if performance does not change, the shuffle may not have touched semantic order channels.

Examples:

```text
set_literal([13,2]) shuffled to [2,13] but canonical AST later sorts it back
examples shuffled after constructor already selected first-example role
counterexample ids preserve original order
clause_id hashes preserve semantic sequence
packet hash names encode pre-shuffle order
budget ledger uses original order for tie-breaks
```

### Harness-Missing Path

An audit can report "shuffled packets pass" while all useful order has been canonicalized elsewhere. Conversely, if the learner's tie-break depends on canonical AST serialization, a shuffle may not test tie-break bias.

### Required Tests

W27 must define which order is semantic and which is not:

```text
semantic order: explicit sequence operators, charged
nonsemantic order: examples, clauses, set literals, candidate listings, logs
```

For nonsemantic order:

```text
1. generate at least N independent shuffles after construction;
2. recompute packet hash, bit length, provenance replay, and learner output;
3. require invariant token and bounded HFA variance;
4. run same shuffles for every baseline adapter;
5. confirm no hidden ids, clause ids, or hashes preserve pre-shuffle positions.
```

For semantic order:

```text
charge it as data and test whether order alone predicts roles.
```

### Verdict + Kill Records

```text
ORDER IS EITHER NONSEMANTIC AND INVARIANT, OR SEMANTIC AND CHARGED.
```

Kill records:

```text
KR-I248-1: If nonsemantic shuffles change token assignment, void for order
channel or learner tie-break dependence.

KR-I248-2: If supposedly shuffled packets retain pre-shuffle ids, hashes, or
canonical ranks that predict roles, void.

KR-I248-3: If an order feature is needed for success, charge it explicitly and
rerun teaching-dimension, active-learning, CEGIS, RAG, and library baselines.
```

### NARRATIVE ATTACK

**Strongest covert-channel dismissal:** You did not teach a frame; you taught readers to look at the first thing.

**What the result needs to be:** semantic/nonsemantic order contract, post-shuffle provenance replay, and parity shuffles for all systems.

---

## I249: Hidden-Open Discipline Must Be Cryptographic Enough To Be Boring

### Single Most Dangerous Question

Can W27 change constructor, scorer, timeout, seed, or baseline code after seeing hidden behavior while preserving plausible deniability?

### Attack

B34 correctly says the first hidden result is one-way. The implementation risk is that "hidden" is a social convention rather than a sealed event.

Potential leak paths:

1. hidden seed rule exists in code before harness freeze;
2. developer can run hidden dry-runs during debugging;
3. failing hidden logs are left in temp directories;
4. tests import hidden fixtures;
5. golden controls share hidden family structure;
6. hidden hashes are generated after code changes;
7. timeouts are adjusted after observing which baseline is close;
8. packet template changes are justified as bug fixes after hidden failure.

### Required Hidden-Open Protocol

Before any hidden run:

```text
freeze implementation commit
freeze dependency versions
freeze seed derivation rule
freeze public/seen split
write manifest hash
write hidden-seed commitment hash
run golden controls
run dry-run MI tests without hidden labels
record no-hidden-open statement
```

At hidden open:

```text
timestamp
commit
manifest hash
command line
stdout/stderr hash
raw artifact hash
token result
```

After hidden open:

```text
allowed: analyze, report, assign token
not allowed under same seeds: edit constructor, scorer, baseline adapters,
timeouts, packet grammar, seed rules, hidden query distribution, thresholds
```

Bug fixes require fresh hidden seeds and a new manifest.

### Harness-Missing Path

Version control alone is not enough if ignored temp files contain hidden labels or if a developer can run hidden commands without logging them. W27 needs a hidden-open ledger and temp-artifact policy.

### Verdict + Kill Records

```text
HIDDEN EVALUATION IS A ONE-WAY DOOR, NOT A DEBUG MODE.
```

Kill records:

```text
KR-I249-1: If hidden labels or hidden role maps are available during ordinary
unit tests, void.

KR-I249-2: If constructor, scorer, baseline adapters, packet grammar, timeouts,
or thresholds change after hidden open under the same seeds, void.

KR-I249-3: If raw hidden-run artifacts are not hashed and preserved, a hostile
reviewer cannot assign the token.
```

### NARRATIVE ATTACK

**Strongest hidden-tuning dismissal:** The experiment had a hidden set only until it became inconvenient.

**What the result needs to be:** manifest-bound one-way hidden open, artifact hashes, and seed rotation after any post-open implementation change.

---

## I250: Sibling Transfer Can Be Faked By Shared Constructor Choices

### Single Most Dangerous Question

Does AFTD measure reusable frame transfer, or repeated task-specific binding selected by the same answer-aware constructor?

### Attack

The spec requires at least two sibling tasks. That blocks single-task teaching dimension, but it introduces a new constructor confound:

```text
constructor sees target + siblings and emits one packet optimized jointly
```

If the constructor is allowed to see sibling labels, sibling role maps, or sibling hidden queries, AFTD becomes multi-task answer compression. That may be useful, but it is not necessarily frame transmission.

Even if sibling tasks are public to the constructor under the spec, the cost split must be explicit:

```text
frame cost F
target binding cost B_target
sibling binding costs B_s1, B_s2
```

Without the split, a "reusable" packet may just contain three support lists and three compact kernels.

### Harness-Missing Path

AFTD can look good if:

1. siblings share exact causal slots instead of only frame type;
2. siblings reuse target public examples;
3. sibling kernels are different but support binding is identical;
4. constructor emits a patch plus hidden per-sibling bindings;
5. per-task binding bits are counted as frame bits and amortized incorrectly;
6. sibling failure cases are excluded by generator filtering.

### Required AFTD Decomposition

W27 must report:

```text
F_reusable_frame_bits
B_target_binding_bits
B_sibling_1_binding_bits
B_sibling_2_binding_bits
per_task_kernel_or_program_bits
per_task_public_label_bits
shared_vs_task_specific_provenance
```

Run erasure tests:

```text
keep F, remove B_i -> should not solve task i
keep B_i, remove F -> should fail or lose AFTD gap
swap B_i across siblings -> behavior should follow binding, not target identity
reuse F on fresh sibling not seen by constructor -> should retain reduced cost
```

The fresh-sibling test is the hard one. AFTD is weak if all siblings were visible during packet construction.

### Verdict + Kill Records

```text
SIBLING SUCCESS IS NOT REUSE IF THE CONSTRUCTOR PACKED EVERY SIBLING ANSWER.
```

Kill records:

```text
KR-I250-1: If frame bits and per-task binding bits are not separated, AFTD is
uninterpretable.

KR-I250-2: If the frame fails on a fresh sibling withheld from packet
construction, demote to teaching-dimension or library-learning absorption.

KR-I250-3: If sibling tasks share exact support, query order, or generator
filtering with the target beyond the declared frame type, no T3-R signal is
valid.
```

### NARRATIVE ATTACK

**Strongest amortization dismissal:** You amortized three answer keys and called the common header a frame.

**What the result needs to be:** frame/binding cost split, erasure/swap tests, and at least one fresh sibling not visible to the constructor.

---

## I251: Representation-Noncontainment Can Fail Through Tie-Breaks And Canonicalization

### Single Most Dangerous Question

Does L0 or L3 already prefer the intended frame because of canonical costs, tie-breaks, or AST normalization?

### Attack

B34 named learner prior risk. The implementation-specific version is sharper: even if FIR-0 has no banned names, its cost model can make the causal-support program uniquely cheap.

Risky details:

1. `set_complement([causal_pair])` is much cheaper than enumerating nuisances;
2. sorted slot pairs make low-index causal pairs favored under some seed schedules;
3. truth-table programs over two slots are exactly the target family;
4. canonical AST serialization breaks ties by slot order;
5. verifier normalization removes distinctions that baselines must search;
6. `edited_value` is native, so intervention semantics are already installed;
7. role-isomorphism test ignores compound primitives.

The frame can be "preinstalled" as the cheapest basis vector even without role words.

### Harness-Missing Path

A noncontainment certificate that checks only primitive names misses cost geometry. The right question is not "does R0 contain `select_causal_pair`?" It is:

```text
Does R0 make the intended support/intervention frame much cheaper than the
alternatives before packet content is applied?
```

### Required Cost-Prior Audit

Run a prior-only search:

```text
input: public schema, no packet, no labels beyond allowed public observations
search: R0/H0 with canonical costs and tie-breaks
output: ranked candidate frames/supports/programs
```

Then measure:

```text
rank of true causal-support family
cost gap between true support and nearest alias/support alternative
tie-break dependency on slot permutation
success under randomized labels
success under role permutations
success after removing truth_table_2, set_complement, forall_slot, edited_value one at a time
```

If removing a supposedly generic primitive destroys the signal, the primitive belongs in the packet cost or the result is representation-prior absorption.

### Verdict + Kill Records

```text
THE PRIOR CAN SMUGGLE THE FRAME THROUGH COST, NOT NAMES.
```

Kill records:

```text
KR-I251-1: If true support is consistently low-rank under prior-only canonical
costs, emit representation-prior absorption or redesign the cost model.

KR-I251-2: If success depends on `edited_value`, `truth_table_2`,
`set_complement`, or universal slot quantification being free in R0, count those
as packet-installed frame components or absorb.

KR-I251-3: If canonical tie-breaks change HFA/token under role permutation, void
or negative according to leakage evidence.
```

### NARRATIVE ATTACK

**Strongest prior dismissal:** The learner did not learn the frame; its cost model had already made the frame the path of least resistance.

**What the result needs to be:** cost-prior audits, primitive ablations, and tie-break invariance before any T3-R interpretation.

---

## I252: The Harness Must Hunt Its Own Strongest Absorber

### Single Most Dangerous Question

Does W27 merely implement the audits, or does it actively try to make CEGIS, teaching dimension, nuisance oracle, and library learning win?

### Attack

The sacred outcome is not "make FrameSeed pass." It is find structure that makes intelligence cheap. If ordinary synthesis or library learning explains the effect, that is valuable. The harness must be designed so the strongest absorber can win.

The W27 danger is a polite baseline suite:

1. TD-H0 approximate solver underpowered "for speed";
2. CEGIS capped before it reaches the natural two-slot truth-table solution;
3. library learner receives siblings but not enough wake/sleep or enumeration;
4. nuisance-oracle gets masks but not functional search;
5. RAG cannot execute compact programs that L3 executes;
6. active learner uses weak random query proposals;
7. baseline timeouts are shorter than L3 constructor/search time;
8. baselines run on harder query distributions than L3.

This would create a green harness and a fake moonshot.

### Required Adversarial Baseline Posture

For each baseline, W27 must include:

```text
strongest_known_variant
why this variant should be able to absorb if the effect is boring
budget curve at 1x, 2x, 4x
timeout parity
query distribution parity
executable packet parity
near-miss diagnostics
expected absorption cases from golden controls
```

The report should welcome near-wins. A baseline that never gets close is suspicious in this Boolean world.

### Final Oversight Gate

Before hidden performance:

```text
1. W27 implementation commit exists and is manifest-bound.
2. Constructor is capability-blind and replayably provenance-logged.
3. Seed streams are domain-separated and rejection-safe.
4. Golden controls assign expected tokens end to end.
5. Baseline adapters pass executable parity.
6. Budget ledgers recompute report and all-in costs.
7. MI/leakage audits include conditional and classifier probes.
8. Decisive-world filtering is logged and distribution-audited.
9. Packet shuffles and role permutations preserve token or void.
10. Hidden-open protocol is one-way and artifact-hashed.
11. AFTD splits reusable frame cost from per-task binding cost.
12. Representation-prior cost geometry is audited.
13. Strong absorbers are implemented to win if they can.
14. No learner optimization, hidden HFA reporting, or packet template tuning has occurred.
```

### Verdict + Kill Records

```text
THE HARNESS IS NOT TRUSTWORTHY UNTIL IT TRIES HARD TO DISPROVE FRAMESEED.
```

Kill records:

```text
KR-I252-1: If W27 implements audits but weakens the absorbing baselines, no
positive token can persuade a hostile reviewer.

KR-I252-2: If no absorber nearly wins on any golden or seen-family slice, suspect
baseline handicapping and require stronger baselines before hidden open.

KR-I252-3: If W27 starts learner optimization before these fourteen oversight
conditions pass, revert to B34 no-go: implementation has skipped the harness
gate.
```

### NARRATIVE ATTACK

**Strongest moonshot-self-deception dismissal:** The harness made cheating hard but made boring explanations harder.

**What the result needs to be:** an audit harness whose favorite outcome is the truth, including absorption.

---

## Batch 35 Oversight Checklist For W27

W27 must attach evidence for each item:

1. Manifest-bound implementation commit and file list.
2. Capability-isolated constructor input type.
3. Replayable constructor provenance with rejected candidate logging.
4. Domain-separated RNG streams and rejection-safe seed handling.
5. End-to-end golden controls that emit signal, absorption, negative, and void where expected.
6. Field-by-field baseline executable parity matrix.
7. Independent report-budget and all-in-budget recomputation.
8. Conditional MI, per-family leakage tests, and adversarial role classifiers.
9. Accepted/rejected world distribution audit.
10. Packet-order and set-order shuffle invariance with post-shuffle provenance replay.
11. One-way hidden-open ledger with raw artifact hashes.
12. Frame/binding AFTD decomposition and fresh-sibling transfer test.
13. Representation-prior cost geometry audit and primitive ablations.
14. Strongest-absorber posture for TD-H0, CEGIS, nuisance-oracle, RAG, active learning, and library learning.

## Final Recommendation

```text
NO IMPLEMENTATION PASS YET.
```

Proceed only when W27 produces a reviewable harness commit and artifact bundle satisfying the checklist above. If W27 instead reports hidden learner performance before these gates pass, the Q-loop should recommend:

```text
FRAMESEED_T3_VOID_SMUGGLED_FRAME
```

The home run is not a positive Boolean number. The home run is a sealed, replayable, parity-respecting harness that lets a hostile reviewer try to reassign the token and fail.