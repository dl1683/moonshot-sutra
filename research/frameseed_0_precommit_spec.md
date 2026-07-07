# FRAMESEED-0 Precommit Specification

**Date:** 2026-07-07  
**Status:** PRECOMMIT SPEC ONLY. No implementation, no exploration, no results.  
**Origin:** Q-Loop B32, Dual-Loop Supervisor Check-in #24, W-Loop B25.  
**Terminal token required:** one of `FRAMESEED_T3_SIGNAL`, `FRAMESEED_T3_ABSORBED_BY_ACTIVE_LEARNING`, `FRAMESEED_T3_ABSORBED_BY_CEGIS`, `FRAMESEED_T3_ABSORBED_BY_RAG`, `FRAMESEED_T3_VOID_SMUGGLED_FRAME`, `FRAMESEED_T3_NEGATIVE`.

## 0. Scope

FRAMESEED-0 tests one claim:

```text
A compact, inspectable, frame-bearing packet can teach a weak generic learner to
use the right intervention/invariant/verifier frame on hidden families, while
rote examples, active learning, CEGIS, and RAG get equal information and fail or
need at least 4x more packet/query budget.
```

This is not a claim that FrameSeed solves intelligence. It is a precommitted T3
filter. A positive toy demo without absorption baselines is not evidence.

Binding givens:

1. Swing for the home run: cheap, ubiquitous, useful AI for people without a
   data center.
2. Stop only when a hostile fresh-eyes reviewer cannot tear the repo down.

All mechanisms are replaceable. The five sacred outcomes from `research/VISION.md`
are fixed: genuine intelligence, improvability, democratized development, data
efficiency, and inference efficiency.

Terminal-token precedence:

1. Any smuggling or parity failure -> `FRAMESEED_T3_VOID_SMUGGLED_FRAME`.
2. L3 full packet below threshold -> `FRAMESEED_T3_NEGATIVE`.
3. Any absorbing baseline at matched or less-than-4x budget -> corresponding
   absorption token, with CEGIS precedence over active learning over RAG.
4. All signal gates pass -> `FRAMESEED_T3_SIGNAL`.
5. Any remaining non-smuggling gate failure -> `FRAMESEED_T3_NEGATIVE`.

---

## I225: World Design

### Steelman

The world must make observational surface learning ambiguous while making a
small interventional frame decisive. Nuisance entropy should punish
reconstruction and rote packets, but not a compact frame packet.

### Formal World

For each nuisance size:

```text
m in {4, 16, 64, 256}
d = m + 4
```

A world instance is:

```text
W = (m, K, rho, beta, pi, names, seed)
```

Latent roles:

```text
C = (c0, c1)        causal bits
S = (s0, s1)        spurious alias bits
N = (n1, ..., nm)   nuisance bits
```

Sampling before intervention:

```text
c0, c1 ~ Bernoulli(0.5)
N_i    ~ Bernoulli(0.5) independently
S      = rho(C)
```

`K` is a 2-input Boolean truth table that depends on both inputs. Constants and
single-variable projections are excluded. `rho` is a bijection on bit pairs.
`beta` is an orientation bit per surface slot. `pi` randomly maps latent roles
to surface slots. Learners see only surface slots and random names.

Surface value:

```text
x[j] = latent_value(pi^{-1}(j)) xor beta[j]
```

Surface names are random 96-bit identifiers. They must not contain role words or
stable role-correlated patterns.

### Query And Target

A query is:

```text
q = (x, tau)
tau = [set(slot_id, observed_bit), ...]
```

The target is:

```text
y = K(C_after)
```

Only edits to causal-role surface slots update `C_after`. Edits to spurious or
nuisance slots change the visible query but not the target cause. Surgical edits
do not refresh aliases.

### Observational Ambiguity

On no-intervention observations:

```text
H_C(x, none) = K(C)
H_S(x, none) = K(rho^{-1}(S))
```

Since `S = rho(C)`, `H_C` and `H_S` are exactly observationally equivalent.
They diverge under targeted intervention:

```text
set(causal_slot, v): H_C may change while H_S holds fixed
set(alias_slot, v):  H_C holds fixed while H_S may change
```

Every generated world must contain at least one decisive intervention row where
`H_C != H_S`.

### Hidden Families

Freeze a kernel split from a public precommit seed:

```text
K_seen = 4 admitted kernels
K_hidden = 6 admitted kernels
```

Tune only on `K_seen`. Hidden evaluation must include at least:

```text
H1: hidden kernel, identity alias map, random roles/names
H2: hidden kernel, non-identity alias map, random roles/names
H3: hidden kernel, non-identity alias map, random orientations
H4: H3 plus high nuisance m and composed interventions
```

### Role Permutation And Name Randomization

For each world, generate at least 10 role-permutation variants. Each variant
preserves the latent structure and labels but resamples surface slots, names,
and, unless isolated, orientation bits. Passing requires:

```text
same terminal token for >= 95% of permutation bundles
within-bundle HFA std <= 0.02
```

### Data-Reading Check

This implements Batch 32's two-rule observational ambiguity and Check-in #24's
requirements for nuisance growth, role permutation, and name randomization.

### Alternative Interpretation

The first experiment could start outside Boolean worlds. The approved path is a
CPU-first synthetic absorption filter, not a public victory claim.

### Right Experiment Check

This is the right experiment for this iteration only if the confirm/kill/void criteria make at least one hostile dismissal executable. The token path remains locked to the six terminal verdicts; narrative appeal cannot override them.

### Attack
```text
You made a puzzle whose answer is just "use the causal slots."
```

Correct risk. Therefore slot ids, interventions, clauses, and programs are
counted, baselines receive equal information, and role/name leakage voids the
run.

### Confirm / Kill / Void

Confirm if observational equivalence is exact, decisive interventions exist for
every hidden world, hidden families are genuinely held out, and nuisance bits
are independent. Kill/redesign if raw decision trees from no-intervention data
reach 0.95 HFA or decisive interventions are too rare. Void if names, order,
metadata, or hidden splits leak role labels.

### NARRATIVE SECTION

Gossip story: the tiny learner sees two worlds that look identical, and the
vaccine must teach the one surgical question that reveals which world is real.

It survives "isn't that obvious?" and "so what?" only if the intervention frame
stays compact as nuisance bits grow and the boring baselines cannot get the same
transfer under equal budget.

If boring: the Boolean world is boring by itself. It matters only as a clean trap
for absorption.

---

## I226: Learner Architecture

### Steelman

L3 must be able to ingest rich teaching packets, but it must not already contain
the target frame. It should be generic, deterministic, and weak.

### L3: Packet-Conditioned Generic Finite Learner

```text
L3(packet V, public_schema Sigma) -> predictor P
P(query q) -> bit
```

L3 cannot make active queries. It learns only from the packet and public schema.

`Sigma` contains surface slot count, bit slot type, intervention grammar,
query grammar, label type, packet grammar, canonical decoder, and the generic
representation language. It does not contain causal/spurious/nuisance labels,
alias maps, kernel identity, hidden family id, generator seed, or target-slot
selectors.

### Required Inputs

L3 must ingest:

```text
example(masked_observation, intervention, label)
counterexample(candidate_program, query, expected_label, actual_label)
invariant(transform_schema, context_schema, output_relation)
transform(name_token, operation_schema, admissibility_scope)
verifier_clause(clause_id, finite_obligation_schema)
program(ast, declared_cost, inputs, outputs) optional
```

`name_token` is a random identifier, not semantic prose.

### Generic Representation: FIR-0

Sorts:

```text
Bit, Slot, Observation, Edit, Query, Set[Slot], Program
```

Allowed primitives:

```text
value(obs, slot)
edited_value(obs, edits, slot)
eq, not, and, or, if
member(slot, set)
forall_slot(set, predicate), exists_slot(set, predicate)
set_literal(slot_list), set_complement(slot_list, universe_slots)
truth_table_2(bit, bit, table4)
let(binding, body)
```

`truth_table_2` is a generic 4-bit table interpreter, not named XOR/AND/OR or a
target primitive.

Banned primitives include:

```text
causal, spurious, nuisance, alias, true_role, target_kernel,
select_causal_pair, hidden_family, generator_seed, rho, beta, pi, oracle_label
```

Natural-language rule interpretation is banned. Executable packet entries must
compile to FIR-0.

### L3 Algorithm

1. Decode packet canonically.
2. Reject any opcode outside FIR-0.
3. Build constraints from examples, counterexamples, invariant clauses, verifier
   clauses, and transformation hints.
4. Candidate pool = supplied compact programs plus all FIR-0 programs with cost
   `<= B_L3`.
5. Default bound: `B_L3 = min(128, packet_bit_length)`.
6. Filter candidates by all packet constraints.
7. Choose minimum-cost consistent candidate.
8. Break ties by public role-blind canonical AST serialization after slot
   renaming.
9. If no candidate survives, output packet majority label.

L3 is weak because it has no active queries, hidden labels, natural language,
unbounded synthesis, generator access, or named target roles.

### Data-Reading Check

The supervisor required a learner with no named target primitives that can ingest
examples, counterexamples, invariant clauses, transformation hints, verifier
clauses, and compact programs. FIR-0 is the precommitted interface.

### Alternative Interpretation

A small neural learner might be more realistic, but it would blur whether the
packet or the learned prior supplied the frame. FRAMESEED-0 starts with a finite
learner.

### Right Experiment Check

This is the right experiment for this iteration only if the confirm/kill/void criteria make at least one hostile dismissal executable. The token path remains locked to the six terminal verdicts; narrative appeal cannot override them.

### Attack
```text
L3 is just CEGIS with a better packet interface.
```

Maybe. Therefore L2 CEGIS gets the same facts and a strong generic DSL. If L2
matches under budget, report CEGIS absorption.

### Confirm / Kill / Void

Confirm if every accepted packet compiles to FIR-0, no primitive names latent
roles, and behavior is role/name invariant. Kill/redesign if L3 solves hidden
worlds from observational examples alone or succeeds after frame-component
ablations. Void if FIR-0 contains answer-shaped primitives or hidden metadata.

### NARRATIVE SECTION

Gossip story: the learner is deliberately ignorant, and the packet has to teach
it the frame without whispering the answer in a secret language.

It survives only if the same ignorant learner uses compact formal lessons that
ordinary search does not absorb.

If boring: if L3 is merely CEGIS with branding, the run is CEGIS absorption.
---

## I227: Vaccine Packet Format

### Steelman

The packet is the experimental object. It may teach through examples,
targeted interventions, counterexamples, invariant hints, verifier clauses, and
optional compact programs. No channel gets free bits.

### Packet Object

```text
V = (header, E, I, CE, H, VC, CP)
```

Where:

- `header`: version, schema hash, surface slot count.
- `E`: observational or masked examples.
- `I`: targeted intervention examples.
- `CE`: counterexamples to candidate rules.
- `H`: invariant and transformation hints.
- `VC`: packet-level verifier clauses.
- `CP`: optional compact FIR-0 programs or fragments.

No human prose is executable. If prose enters an executable channel, its UTF-8
bytes are counted and no semantic parser may use it.

### Entry Types

Example:

```text
example(obs_mask=[(slot_id, bit), ...], intervention=[...], label=bit)
```

Targeted intervention:

```text
intervention_example(base_mask, edit=set(slot_id, bit), label_before?, label_after)
```

Counterexample:

```text
counterexample(candidate_program_ast, query, expected_label, candidate_label)
```

Invariant hint:

```text
invariant(transform_schema, context_schema,
          relation=output_unchanged | output_changes_as(program_ast))
```

Allowed transform schemas:

```text
set_one(slot_id, bit)
set_any(slot_set, bit)
permute_slots(permutation)
rename_slots(permutation)
compose(transform_schema, transform_schema)
```

Verifier clause:

```text
verifier_clause(clause_id, finite_scope, required_relation)
```

Compact program:

```text
program(ast, declared_cost)
```

If a full program solves the task and CEGIS/RAG can use it under budget, the run
is absorbed. If it contains hidden metadata, the run is void.

### Packet Length Metric

All systems use the same canonical binary serialization:

```text
|V| = bit length of canonical packet serialization
```

Costs:

```text
slot_bits      = ceil(log2(d))
opcode_bits    = ceil(log2(number_of_packet_opcodes))
bit_value      = 1 bit
truth_table_2  = opcode + 4 bits
edit           = opcode + slot_bits + 1
label          = 1 bit
set_literal    = opcode + count_bits + k * slot_bits
set_complement = opcode + count_bits + k * slot_bits
program_ast    = preorder opcode serialization plus literal costs
```

Full example cost:

```text
opcode + d bits + intervention_cost + label
```

Masked example cost:

```text
opcode + count_bits + k * (slot_bits + bit_value) + intervention_cost + label
```

The packet may use `set_complement([slot_a, slot_b], all_slots)` to avoid listing
all nuisance slots, but the mentioned slot ids and opcode are counted.

### Sublinear Growth Requirement

For each `m`, let `P(m)` be median full-vaccine packet length across hidden
worlds and role permutations. Passing requires:

```text
1. P(m)/m decreases from m=16 to m=64 to m=256.
2. log-log slope alpha_hat of P(m) versus m over {4,16,64,256} is <= 0.50.
3. No individual seed has slope > 0.80.
4. Full-observation packets are reported separately and must meet the same test.
```

This rejects "five full examples" when each carries all nuisance bits.

### Data-Reading Check

Batch 32 required packet length to stay sublinear in nuisance bits and to count
examples, interventions, counterexamples, invariant hints, verifier clauses, and
optional compact programs.

### Alternative Interpretation

Tokenizing natural-language lessons would be more human-like, but it would
import uncontrolled priors. FRAMESEED-0 uses formal packets only.

### Right Experiment Check

This is the right experiment for this iteration only if the confirm/kill/void criteria make at least one hostile dismissal executable. The token path remains locked to the six terminal verdicts; narrative appeal cannot override them.

### Attack
```text
The packet wins because the answer is hidden in an uncounted clause or prose.
```

Canonical serialization is the answer: every slot id, set, clause, and AST node
is on the bill.

### Confirm / Kill / Void

Confirm if the same serializer is used for vaccines and baselines, all used
fields are counted, and `P(m)` passes sublinear growth. Kill/redesign if success
requires full observations or a full program that CEGIS absorbs. Void if any
learner uses uncounted information.

### NARRATIVE SECTION

Gossip story: the vaccine is a tiny formal lesson, and every bit it whispers to
the learner goes on the bill.

It survives only if the bill stays small while nuisance entropy grows and
ordinary packets cannot buy the same transfer.

If boring: if the packet is just a compressed answer key, the result is absorbed
or void.

---

## I228: Baseline Specifications

### Steelman

Every hostile dismissal must become executable. The baselines get equal
information, matched budgets, and the same hidden evaluation.

### Shared Information And Budgets

For a vaccine packet `V`:

```text
B_bits = |V|
B_queries = number of oracle labels used to construct V
```

Every baseline receives the same public schema and the same packet entries, or a
lossless canonical translation. If a baseline ignores a field by design, the
report must say so; the field is still provided and counted.

Matched baselines satisfy:

```text
total_information_bits <= B_bits
oracle_query_count <= B_queries
```

Also run 2x and 4x curves. A baseline absorbs if it reaches threshold at matched
budget or at less than 4x budget. If it needs exactly 4x or more, report the
ratio but do not emit absorption.

### L0: Rote Example / Nearest Neighbor

Input: all packet examples, intervention examples, and counterexamples as
labeled query records. Invariant and transformation hints are converted only
into explicitly generated labeled records if they enumerate finite records
within budget. Compact programs are stored as opaque records and not executed.

Distance:

```text
dist(q, q') = Hamming(shared mentioned surface bits)
            + Hamming(intervention slot ids and values)
            + 0.5 * missing_bit_count
```

Prediction: majority label among `k in {1,3,5}` nearest records, with `k` chosen
on seen-family validation and frozen. Tie-break: packet majority, then 0.

Absorbs if `HFA >= 0.95` under matched or less-than-4x budget. If L0 absorbs,
the packet is rote/retrieval.

### L1: Active Learner Over Supplied Feature Hypotheses

L1 tests "this is just active learning."

Hypothesis class:

```text
H_active =
  all 1-slot predictors;
  all 2-slot truth_table_2 predictors over surface slots;
  all predictors constrained by packet invariant clauses when expressible;
  all candidates induced by packet counterexamples;
  no latent role predicates.
```

A 2-slot predictor:

```text
h_{a,b,t}(x,tau) = truth_table_2(
  edited_value(x,tau,a), edited_value(x,tau,b), table=t)
```

L1 may ask public training-oracle queries under budget. It chooses queries by
expected version-space reduction. It may use examples, counterexamples,
invariants, verifier clauses, and compact programs as constraints if expressible
inside `H_active`. It may not inspect hidden labels or latent metadata.

Prediction: minimum-description-length consistent hypothesis; ties by role-blind
canonical serialization.

Absorbs if `HFA >= 0.95` under matched or less-than-4x budget. If L1 absorbs,
FrameSeed is active curriculum inside a supplied hypothesis class.

### L2: CEGIS / Program Learner With Supplied DSL

L2 tests "this is just synthesis."

DSL:

```text
FIR-0 plus bounded enumeration over all surface slots, all 2-slot supports,
generic truth tables, generic invariance constraints, counterexample-guided
refinement, and optional use of compact programs as seeds.
```

Banned: causal/spurious/nuisance/alias predicates, target kernel primitives,
generator seed, hidden family id, `rho`, `beta`, `pi`.

Loop:

1. Initialize constraints from packet entries.
2. Enumerate candidate programs by increasing canonical cost.
3. Test on public examples and packet verifier clauses.
4. Query public counterexample oracle only within budget.
5. Return minimum-cost surviving program or majority fallback.

Every oracle query, answer, and final program bit is counted.

Absorbs if `HFA >= 0.95` under matched or less-than-4x budget. If L2 absorbs,
the vaccine is ordinary CEGIS/program synthesis under a supplied generic DSL.

### RAG Baseline: Few-Shot Retrieval

Corpus: every packet entry serialized as canonical text and canonical fields.
Index by mentioned slots, values, intervention slots, opcode ids, and AST nodes.

Retrieve top `k in {1,3,5,8}`, selected on seen-family validation and frozen, by
weighted Jaccard overlap.

Variants:

1. `RAG-NN`: majority label from retrieved labeled queries.
2. `RAG-CLAUSE`: applies retrieved invariant clauses only when they map the test
   query to a retrieved labeled query.
3. `RAG-PROG`: executes a retrieved compact FIR-0 program if present.

The strongest variant is the RAG baseline. Absorbs if `HFA >= 0.95` under
matched or less-than-4x budget. If `RAG-PROG` absorbs because the packet
contains the full executable solution, report RAG absorption unless CEGIS also
absorbs.

### Data-Reading Check

Batch 32 named active learning, CEGIS, and RAG as the key absorption risks. This
section gives all three equal information and clean win conditions.

### Alternative Interpretation

A neural or LLM baseline will matter later. FRAMESEED-0 is CPU-first and uses
formal baselines; an added neural baseline must receive the same packet access
and be reported separately.

### Right Experiment Check

This is the right experiment for this iteration only if the confirm/kill/void criteria make at least one hostile dismissal executable. The token path remains locked to the six terminal verdicts; narrative appeal cannot override them.

### Attack
```text
Your baselines are handicapped because they cannot read the same packet.
```

They receive the same packet or lossless translations. If their architecture
ignores a field, that limitation is explicit.

### Confirm / Kill / Void

Confirm if L0, L1, L2, and RAG run on the same hidden cases, budgets, packets,
and role permutations as L3. Kill or radically reframe if any two of L1, L2, and
RAG absorb. Void if a baseline lacks information L3 used or budget accounting
is inconsistent.

### NARRATIVE SECTION

Gossip story: the vaccine has to beat the boring explanations after the boring
explanations are handed the same evidence.

It survives only if active learning, synthesis, and retrieval all get fair shots
and still cannot buy the same hidden-family transfer cheaply.

If boring: if two baselines absorb, FrameSeed is boring as a moonshot and should
be killed or radically reframed.
---

## I229: Measurement Protocol

### Steelman

The measurement must force a real T3 claim: hidden-family transfer, sublinear
packet growth, equal-budget baselines, ablations, role permutation stability,
and smuggling audit before signal.

### Evaluation Size

For each `m in {4,16,64,256}`:

```text
hidden_worlds_per_m >= 64
role_permutations_per_world >= 10
hidden_eval_queries_per_world >= 512
```

Hidden queries are balanced:

```text
20% no intervention
20% causal-slot single edit
20% spurious-alias single edit
20% nuisance single edit
20% composed edits
```

Invalid worlds are resampled before learners see them.

### Primary Metric

```text
HFA(system, m, hidden_family) = correct hidden predictions / total hidden queries
```

`FRAMESEED_T3_SIGNAL` requires:

```text
HFA(L3_full, m, hidden_family) >= 0.95 for every m and hidden-family class
mean HFA(L3_full) >= 0.97 overall
```

### Budget Matching

For every system:

```text
total_information_bits = packet bits
                       + serialized oracle query bits
                       + serialized oracle answer bits
                       + final executable program bits if used at inference
```

Matched baselines must not exceed L3 full vaccine bits or query count.
Report 1x, 2x, and 4x curves:

```text
bits_to_0.95(system)
queries_to_0.95(system)
ratio_bits = bits_to_0.95(system) / bits_to_0.95(L3_full)
ratio_queries = queries_to_0.95(system) / queries_to_0.95(L3_full)
```

A baseline absorbs if either ratio is less than 4 and it reaches 0.95.

### Nuisance Growth Curve

For each `m`, report:

- vaccine packet length `P(m)`;
- L3 HFA;
- L0, L1, L2, RAG HFA at 1x, 2x, 4x;
- baseline bits/queries needed for 0.95, if reached;
- ablation HFA;
- role permutation stability.

Sublinear packet growth from I227 is mandatory.

### Ablations

For every full packet `V`, construct:

```text
V_no_intervention:
  remove targeted intervention examples; replace their bit budget with
  no-intervention examples selected by the same constructor.

V_no_counterexample:
  remove counterexamples; replace their bit budget with ordinary labeled examples.

V_no_invariant:
  remove invariant and transformation hints; replace their bit budget with
  ordinary labeled examples.
```

Optional:

```text
V_no_verifier:
  remove verifier clauses; replace their bit budget with examples.
```

Signal requires for each primary ablation:

```text
HFA(L3_full) - HFA(L3_ablation) >= 0.20 aggregate
and >= 0.20 for at least three of four m values
```

If an ablation remains at or above 0.95, no T3 signal.

### Role Permutation Stability

For each permutation bundle:

```text
terminal token identical
HFA std <= 0.02
packet length coefficient of variation <= 0.10
```

If instability is caused by leakage, void. If no leakage is found, negative.

### Secondary Metrics

Report packet construction CPU time, L3 inference CPU time, final AST cost,
number of slot ids mentioned, counts of packet entry types, randomized-label
control accuracy, and each baseline's best budget ratio. These diagnostics
cannot override terminal tokens.

### Data-Reading Check

This implements the requested threshold, matched budgets, four nuisance values,
20pp ablations, and role permutation stability.

### Alternative Interpretation

Averaging over hidden cases would be easier. FRAMESEED-0 requires every nuisance
size and hidden-family class to pass because hostile review will attack weak
subfamilies.

### Right Experiment Check

This is the right experiment for this iteration only if the confirm/kill/void criteria make at least one hostile dismissal executable. The token path remains locked to the six terminal verdicts; narrative appeal cannot override them.

### Attack
```text
The effect is a fragile average that disappears under nuisance growth or when a
packet component is removed.
```

The curve, ablations, and per-family thresholds make that executable.

### Confirm / Kill / Void

Confirm if L3 reaches all thresholds, packet growth is sublinear, baselines fail
absorption, ablations drop by 20pp, and role permutation is stable. Kill/reframe
if signal exists only at small `m`, ablations do not matter, or baselines need
only slightly more budget. Void if hidden eval is contaminated or budgets differ.

### NARRATIVE SECTION

Gossip story: the vaccine has to keep working when the world gets noisy,
renamed, rearranged, and stripped of its favorite lesson parts.

It survives only if the measured gap is robust across nuisance growth, ablation,
and fair baseline budgets.

If boring: a single accuracy number is boring; the curve and ablations are the
actual experiment.

---

## I230: Smuggling Audit

### Steelman

A T3 signal dies if the frame is already inside names, metadata, DSL primitives,
verifier clauses, learner priors, compact programs, or baseline handicaps.

### Smuggling Channels To Enumerate

1. Named roles: surface names or name statistics reveal causal/spurious/nuisance
   roles.
2. Hidden labels: labels embedded in packet order, ids, hashes, seeds, or family
   ids.
3. DSL primitives: causal/spurious/nuisance/alias/target selectors or kernel
   primitives.
4. Representation priors: low slot ids, first-mentioned slots, or tie-breaks
   correlate with causal roles.
5. Verifier clauses: public clauses equivalent to the hidden target through
   uncounted information.
6. Compact programs: hardcoded hidden family ids, seeds, role maps, or target
   program with uncounted slot map.
7. Baseline handicaps: baselines denied packet fields, query access, budgets, or
   hidden cases used by L3.
8. Split leakage: hidden seeds inspected before freezing hyperparameters.
9. Evaluation leakage: hidden eval queries or labels included in packet.
10. Human-labor leakage: manual packet edits or threshold changes after hidden
    inspection.

### Audit Procedure

Run before declaring any signal.

#### A. Static Generator Audit

Over at least 10,000 generated worlds, test:

```text
MI(surface_name, latent_role) ~= 0
MI(slot_index, latent_role) ~= 0
MI(packet_order, latent_role) only through counted slot ids
```

Any detectable uncounted role leak voids.

#### B. Packet Serialization Audit

For every packet:

- parse canonical binary serialization;
- list every slot id, opcode, AST node, name token, and clause;
- recompute packet length independently;
- confirm no executable field contains banned strings or hidden metadata;
- confirm ablation replacements respect equal or lower bit budgets.

#### C. DSL And Learner Audit

For L3, L1, L2, and RAG:

- enumerate allowed primitives;
- search source/config for banned role terms;
- run randomized-role worlds;
- run randomized-label worlds.

Randomized-label control:

```text
Replace labels with independent fair bits.
No system may exceed 0.60 HFA.
```

#### D. Baseline Parity Audit

Verify same packet facts, budgets, hidden cases, role permutations, and query
oracle restrictions for every system.

#### E. Role Permutation Audit

Remap slots, resample names, rerun packet construction and all learners, compare
HFA and terminal tokens.

#### F. Human-Labor Ledger

Record:

```text
DESIGNED_BY_HUMANS: generator, packet grammar, learner grammar, baseline specs
FROZEN_BEFORE_HIDDEN: kernel split, hyperparameters, constructor, audits
DISCOVERED_OR_SELECTED_BY_PACKET: slot ids, examples, interventions, clauses
HIDDEN_FOR_EVAL: hidden worlds, labels, role maps, query labels
```

Manual change after hidden inspection voids unless the run restarts with new
hidden seeds.

### Data-Reading Check

PCCP-H showed that DSLs, verifiers, roles, and decomposition choices can smuggle
answers. FRAMESEED-0 voids before interpreting positives.

### Alternative Interpretation

A packet may mention slot ids; that is not automatically smuggling. The audit
bans uncounted or semantically named role information.

### Right Experiment Check

This is the right experiment for this iteration only if the confirm/kill/void criteria make at least one hostile dismissal executable. The token path remains locked to the six terminal verdicts; narrative appeal cannot override them.

### Attack
```text
The frame was not transmitted; it was preinstalled.
```

The answer is not trust. It is static audit, randomized labels, role
permutation, and baseline parity.

### Confirm / Kill / Void

Confirm only if all leakage tests pass, randomized labels fail, role permutation
is stable, and parity is documented. Kill/redesign if L3 succeeds without frame
components. Void if hidden labels, seeds, roles, or baseline parity leak.

### NARRATIVE SECTION

Gossip story: before the vaccine gets credit, it is searched for hidden needles
in the labels, names, grammar, and rules.

It survives only if an enemy can see that the frame was not preinstalled.

If boring: audits are boring until they fail; if they fail, the positive story
was fake.

---

## I231: Verdict Token Specification

### Steelman

The run must end in a token, not a debate.

### `FRAMESEED_T3_SIGNAL`

Emit only if all hold:

1. Smuggling audit passes.
2. L3 full packet has `HFA >= 0.95` for every `m` and hidden-family class.
3. L3 mean HFA is `>= 0.97` overall.
4. Packet growth is sublinear by I227.
5. L0 does not reach 0.95 under matched budget.
6. L1 does not reach 0.95 under matched or less-than-4x budget.
7. L2 does not reach 0.95 under matched or less-than-4x budget.
8. RAG does not reach 0.95 under matched or less-than-4x budget.
9. Each primary ablation drops at least 20pp as specified.
10. Role permutation stability passes.
11. All packet, query, final-program, and verifier-clause bits are counted.

Allowed claim: controlled evidence for a T3 frame-bearing lesson in this
synthetic world. Banned claim: FrameSeed proves cheap general intelligence.

### `FRAMESEED_T3_ABSORBED_BY_ACTIVE_LEARNING`

Emit if the audit passes and L1 reaches `HFA >= 0.95` under matched or
less-than-4x bit/query budget with role stability. Interpretation: active
curriculum inside a supplied hypothesis class.

### `FRAMESEED_T3_ABSORBED_BY_CEGIS`

Emit if the audit passes and L2 reaches `HFA >= 0.95` under matched or
less-than-4x budget using only the predeclared generic DSL with role stability.
Interpretation: ordinary program synthesis with supplied constraints. This token
has precedence over other absorption tokens.

### `FRAMESEED_T3_ABSORBED_BY_RAG`

Emit if the audit passes and any RAG variant reaches `HFA >= 0.95` under matched
or less-than-4x budget with role stability. Interpretation: few-shot retrieval,
not frame transfer.

### `FRAMESEED_T3_VOID_SMUGGLED_FRAME`

Emit if any of these occur:

- role names, hidden labels, seeds, family ids, or target primitives enter any
  learner or packet;
- names, slot order, packet order, or orientation leak roles;
- L3 contains banned primitives;
- public verifier clauses expose hidden target through uncounted information;
- compact programs contain hidden metadata;
- baselines are denied information used by L3;
- hidden split is inspected before freezing hyperparameters;
- randomized-label control exceeds 0.60 HFA;
- role permutation exposes uncounted dependency.

Interpretation: no scientific claim; redesign and rerun with new hidden seeds.

### `FRAMESEED_T3_NEGATIVE`

Emit if the audit passes but L3 fails threshold, packet growth is not sublinear,
ablations do not drop enough, role stability fails without leakage, or any other
non-smuggling T3 gate fails while no absorption token condition holds.

Interpretation: the proposed packet did not transmit a robust T3 frame under
this spec.

### Data-Reading Check

These are exactly the requested tokens, with active learning, CEGIS, RAG,
smuggling, and negative outcomes made terminal.

### Alternative Interpretation

Absorption can be scientifically useful, but it is not a FrameSeed moonshot win.

### Right Experiment Check

This is the right experiment for this iteration only if the confirm/kill/void criteria make at least one hostile dismissal executable. The token path remains locked to the six terminal verdicts; narrative appeal cannot override them.

### Attack
```text
You will call a mixed result a signal.
```

No: a mixed result is absorption, void, or negative by precedence.

### Confirm / Kill / Void

Confirm if every run emits exactly one terminal token and logs secondary ratios.
Kill/reframe if two of L1, L2, and RAG absorb within the hard clock. Void if no
token can be assigned because protocol was violated.

### NARRATIVE SECTION

Gossip story: the experiment is not allowed to spin; it has to leave a verdict
token on the table.

It survives because a hostile reader can see exactly when the boring explanation
wins.

If boring: most tokens are boring by design. The direction earns drama only by
surviving them.
---

## I232: Teaching Ladder Formalization

### Steelman

"Teaching" is too vague. FRAMESEED-0 targets T3 and must not claim T4 or T5.

### T0: Rote Examples

Operational boundary:

```text
The learner stores examples and predicts by lookup, interpolation, nearest
neighbor, or majority over retrieved cases.
```

Evidence: packet length grows with covered cases or nuisance surface; L0/RAG-NN
matches; hidden-family transfer fails under role/name/nuisance shifts.

### T1: Active Curriculum

Operational boundary:

```text
The lesson is a sequence of labels or queries that identifies a hypothesis from
a predeclared hypothesis class already containing the target frame.
```

Evidence: L1 reaches threshold under matched or less-than-4x budget. Verdict:
`FRAMESEED_T3_ABSORBED_BY_ACTIVE_LEARNING`.

### T2: Counterexample-Guided Program Patch

Operational boundary:

```text
The lesson repairs or selects a program in a supplied DSL or patch grammar.
```

Evidence: L2 CEGIS reaches threshold under matched or less-than-4x budget using
the same examples, counterexamples, clauses, and programs. Verdict:
`FRAMESEED_T3_ABSORBED_BY_CEGIS`.

### T3: Frame-Bearing Lesson

Operational boundary:

```text
The packet transmits a compact intervention, invariant, verifier, or
transformation frame through a learner that did not contain named target
primitives; T0-T2 and RAG fail or need at least 4x more budget; hidden-family
transfer, ablations, packet growth, and role permutation all pass.
```

Evidence: exactly the `FRAMESEED_T3_SIGNAL` conditions.

### T4: Self-Generating Frame Teacher

Operational boundary:

```text
The teacher discovers which frame-bearing packet to construct under query and
packet budgets across unfamiliar world families.
```

Additional requirements: teacher algorithm frozen before new families; teacher
chooses interventions/invariants/counterexamples without human packet design;
teacher beats active query selection and synthesis baselines in choosing what
evidence to buy. FRAMESEED-0 does not claim T4.

### T5: Open-World Frame Formation

Operational boundary:

```text
The system forms useful frames in messy real domains without a synthetic role
taxonomy, hidden generator, or hand-authored verifier family.
```

Requires real or semi-real tasks, prior-art and neural/tool baselines,
human-labor accounting, public adversarial review, and a useful artifact or
repair loop. FRAMESEED-0 does not claim T5.

### Why T3

T0-T2 are the boring explanations. T4-T5 are too broad for the first post-PCCP
test. T3 is the first level where the FrameSeed claim is measurable:

```text
Can a compact lesson transmit a frame, not just labels or a supplied program?
```

### Data-Reading Check

Batch 32 proposed T0-T5; Check-in #24 required concrete boundaries. This section
makes each rung operational.

### Alternative Interpretation

T3 may still be machine teaching. FRAMESEED-0 does not deny overlap; it tests
whether the T3 object survives the strongest machine-teaching and synthesis
baselines.

### Right Experiment Check

This is the right experiment for this iteration only if the confirm/kill/void criteria make at least one hostile dismissal executable. The token path remains locked to the six terminal verdicts; narrative appeal cannot override them.

### Attack
```text
T3 is a word game between curriculum and program synthesis.
```

Operational answer: if active learning or CEGIS matches, T3 is absorbed.

### Confirm / Kill / Void

Confirm if every result can be assigned by operational evidence and T3 requires
all gates. Kill/reframe if T3 cannot be separated from T1/T2 or if two-of-three
absorption holds. Void if the ladder changes after hidden results.

### NARRATIVE SECTION

Gossip story: FRAMESEED-0 is trying to climb above flashcards and program
search, but it is not claiming open-world teaching.

It survives only if T3 is operationally separated from the lower rungs.

If boring: if the result lands on T0, T1, or T2, report it that way.

---

## I233: Risk Register And Kill Conditions

### Steelman

The direction deserves a hard clock. If FrameSeed cannot produce a T3 signal or
a clean absorption/void/negative token quickly, the narrative should not inflate.

### Risk Register

| Risk | Failure mode | Required response |
|---|---|---|
| Active-learning absorption | L1 matches under matched or <4x budget | Emit absorption; repeated absorption kills/reframes. |
| CEGIS absorption | L2 synthesizes same predictor from same packet | Emit absorption; strongest kill risk. |
| RAG absorption | Retrieval/few-shot packet matches | Emit absorption; demote to retrieval curriculum. |
| Smuggled frame | Names, DSL, verifier, learner prior, or metadata contains target | Void and rerun with new hidden seeds. |
| Linear packet growth | Packet carries nuisance surface data | Negative unless redesigned before hidden run. |
| Ablation no-op | Removing intervention/counterexample/invariant does not hurt | Negative for T3. |
| Role instability | Result depends on surface order or names | Void if leakage, negative if no leakage. |
| Synthetic triviality | Raw decision trees or L0 exceed threshold | Redesign; no signal. |
| Hidden-family overfit | Seen tuning fails hidden kernels/maps/orientations | Negative. |
| Full-program shortcut | Compact program is the whole answer | Absorbed by CEGIS/RAG or void if smuggled. |
| Human packet authorship | Manual edits after hidden inspection | Void. |
| Metric gaming | Accuracy passes but budget/ablation fails | Negative. |
| Narrative overclaim | T3 reported as cheap general intelligence | Documentation failure; supervisor escalation. |
| Prior-art collapse | Teaching-dimension/CEGIS prior art fully explains result | Treat as absorption unless measured gap remains. |

### Two-of-Three Absorption Rule

Let:

```text
A = active learning absorbs
C = CEGIS absorbs
R = RAG absorbs
```

If at least two of `{A, C, R}` occur in FRAMESEED-0 or its first honest variant:

```text
kill FrameSeed as the main moonshot direction or radically reframe it
```

Allowed reframes: make active query selection the honest substrate; make
CEGIS/proof compilation the honest direction; move to T4 only if teacher choice
itself beats lower baselines. Disallowed rescues: larger L3, hidden prose,
removing baselines, or changing thresholds after hidden results.

### Hard Clock

Starting from W-Loop B25:

| Batch | Required state |
|---|---|
| W25 | Precommit spec written and audited. |
| W26 | First implementation may begin only after spec review. |
| W27 | First full FRAMESEED-0 run emits a token or blocking bug ledger. |
| W28 | One honest variant may address void/bug issues, not relax absorption. |
| W29 | If signal exists, test harsher hidden families or a less Boolean variant. |
| W30 | Supervisor assessment: signal, absorption, void with redesign, negative, or kill. |

No indefinite "almost there" state is allowed.

### Data-Reading Check

The supervisor set a five W-Loop hard clock and the prompt required a
two-of-three absorption rule. Both are binding.

### Alternative Interpretation

Absorption may still be scientifically useful, as PCCP-H was. It is not a
FrameSeed moonshot win.

### Right Experiment Check

This is the right experiment for this iteration only if the confirm/kill/void criteria make at least one hostile dismissal executable. The token path remains locked to the six terminal verdicts; narrative appeal cannot override them.

### Attack
```text
You will keep the direction alive by redefining the goal after every baseline
wins.
```

The hard clock and two-of-three rule prevent that.

### Confirm / Kill / Void

Confirm if every run emits a token, two-of-three absorption triggers kill/reframe,
and W30 cannot pass without supervisor assessment. Kill if two baselines absorb,
no token emerges by W30, or signal requires relaxed thresholds. Void only for
protocol failures requiring new hidden seeds.

### NARRATIVE SECTION

Gossip story: FrameSeed gets five batches to prove it is more than a slogan
before the boring baselines take the wheel.

It survives because the project is willing to kill the exciting phrase if
ordinary methods explain it.

If boring: if the hard clock ends in absorption, the immune system worked.

---

## I234: Integration Test

### Steelman

A different implementer should be able to build FRAMESEED-0 from this spec. This
walk-through checks that the components fit.

### Hypothetical World

```text
m = 16
d = 20
K = xor truth table [0,1,1,0]
rho = identity
c0 -> surface slot 13
c1 -> surface slot 2
s0 -> surface slot 7
s1 -> surface slot 18
n1..n16 -> all other slots
```

Names are random identifiers. No name encodes a role.

Observational row:

```text
c0 = 0, c1 = 1, s0 = 0, s1 = 1, N arbitrary, y = 1
```

No intervention:

```text
xor(c0,c1) = 1
xor(s0,s1) = 1
```

Decisive interventions:

```text
set(slot 13, 1): causal c0 becomes 1, y becomes xor(1,1)=0
set(slot 7, 1): alias s0 changes, causal C stays (0,1), y remains 1
```

### Full Vaccine Packet Seen By L3

Shown in prose here, but implemented as canonical FIR-0:

```text
header(d=20)

example([(13,0),(2,0)], none, 0)
example([(13,0),(2,1)], none, 1)
example([(13,1),(2,0)], none, 1)
example([(13,1),(2,1)], none, 0)

intervention_example([(13,0),(2,1),(7,0),(18,1)], set(13,1), before=1, after=0)
intervention_example([(13,0),(2,1),(7,0),(18,1)], set(7,1), before=1, after=1)

counterexample(
  candidate_program = truth_table_2(slot7, slot18, [0,1,1,0]),
  query = ([(13,0),(2,1),(7,0),(18,1)], set(7,1)),
  expected = 1,
  actual = 0)

invariant(
  transform_schema = set_any(set_complement([13,2]), bit),
  context_schema = listed_masks,
  relation = output_unchanged)

verifier_clause(
  finite_scope = listed masks and listed single-slot edits,
  required_relation = agree with examples and invariant)
```

Slot id cost here is:

```text
slot_bits = ceil(log2(20)) = 5
```

The complement set `[13,2]` costs two slot ids plus opcode/count overhead; it
does not list every nuisance slot.

### What L3 Does

L3 decodes the packet, builds constraints, and selects the minimum-cost
consistent candidate:

```text
truth_table_2(
  edited_value(obs, edits, slot13),
  edited_value(obs, edits, slot2),
  [0,1,1,0])
```

It rejects the alias candidate because of the counterexample and invariant.

### What Each Baseline Sees

L0 receives the same packet, extracts labeled query records, and uses nearest
neighbor. It cannot execute universal invariants unless they enumerate finite
labeled records within budget. If it reaches 0.95, no T3 signal.

L1 receives the same packet and query budget, searches all 1-slot and 2-slot
truth-table predictors, and may use expressible packet constraints. If it finds
slots 13 and 2 under matched or <4x budget, emit active-learning absorption.

L2 receives the same packet, same constraints, and the FIR-0-plus-enumeration
DSL. If it synthesizes the same program under matched or <4x budget, emit CEGIS
absorption. This is a serious expected risk, not a footnote.

RAG receives every packet entry serialized and indexed. If RAG-NN, RAG-CLAUSE,
or RAG-PROG reaches 0.95 under matched or <4x budget, emit RAG absorption unless
CEGIS also absorbs.

### What Gets Measured

For every world, role permutation, and `m`:

```text
HFA per system
packet bits
oracle query bits
final program bits
baseline budget ratios
ablation HFA
role-permutation HFA variance
smuggling audit result
terminal token
```

Across `m`:

```text
P(4), P(16), P(64), P(256)
alpha_hat
baseline bits_to_0.95 and queries_to_0.95
ablation drops
```

### Internal Consistency Check

The run has causal, spurious, and nuisance bits; observational equivalence;
divergent interventions; hidden nuisance/composed edits; generic L3 operation;
equal-information baselines; countable packet length; sublinear possible packet
encoding; and explicit absorption paths.

### Data-Reading Check

This is the exact first experiment proposed by Batch 32 and tightened by
Check-in #24: minimal frame vaccine under nuisance growth with equal-information
baselines and verdict tokens.

### Alternative Interpretation

The packet mentions the causal support slots. A hostile reviewer may call that
the answer. FRAMESEED-0 accepts the objection and asks whether that counted,
compact support/intervention frame beats active search, synthesis, and retrieval
under hidden-family transfer. If not, absorb it.

### Right Experiment Check

This is the right experiment for this iteration only if the confirm/kill/void criteria make at least one hostile dismissal executable. The token path remains locked to the six terminal verdicts; narrative appeal cannot override them.

### Attack
```text
This example shows CEGIS can probably find the same truth table.
```

Correct. That is why `FRAMESEED_T3_ABSORBED_BY_CEGIS` is a primary terminal
token.

### Confirm / Kill / Void

Confirm if an implementer can build generator, serializer, L3, baselines,
metrics, audits, and token logic from this spec alone. Kill/reframe if every
plausible path is CEGIS or active-learning absorption. Void if implementation
needs undeclared task-specific primitives or packet semantics remain ambiguous.

### NARRATIVE SECTION

Gossip story: in the demo run, the vaccine points the cheap learner to the two
slots that still matter after surgery, then the boring baselines get to prove
whether that was special.

It survives only if that pointing is a compact transferable frame under nuisance
growth, not just a support list that active learning or CEGIS can find.

If boring: this integration test may be boring because CEGIS may absorb it. That
would be an honest kill signal, not a failed spec.

---

## Final Precommit Checklist

- [ ] World has causal bits, nuisance bits, spurious bits, and observational
      ambiguity.
- [ ] Two observationally equivalent rules diverge under targeted intervention.
- [ ] Hidden families include held-out kernels, alias maps, orientations, role
      permutations, names, and nuisance growth.
- [ ] L3 has no named target primitives.
- [ ] L3 ingests examples, counterexamples, invariant clauses, transformation
      hints, verifier clauses, and compact programs through FIR-0.
- [ ] Packet format counts examples, interventions, counterexamples, invariant
      hints, verifier clauses, and optional programs.
- [ ] Packet growth criterion is sublinear in `m`.
- [ ] L0, L1, L2, and RAG are exactly specified.
- [ ] Each baseline gets equal information and matched budgets.
- [ ] Hidden-family accuracy threshold is at least 0.95 per family and per `m`.
- [ ] Nuisance curve uses `m = {4, 16, 64, 256}`.
- [ ] Ablations remove intervention, counterexample, and invariant components
      and require at least 20 percentage point drops.
- [ ] Role permutation stability test is required.
- [ ] Smuggling audit runs before any signal.
- [ ] Verdict tokens have exact conditions and precedence.
- [ ] Teaching ladder T0-T5 has operational boundaries.
- [ ] Two-of-three absorption rule is binding.
- [ ] Five W-Loop batch hard clock is binding.
- [ ] Integration test is internally consistent.

## Final NARRATIVE SECTION

Gossip-magazine one-sentence story:

```text
What if a tiny AI does not need the whole internet, just the one compact lesson
that teaches it which parts of the world still matter when you poke them?
```

Does that survive "isn't that obvious?":

```text
Only conditionally. Good teaching is obvious; a counted, compact,
intervention-bearing frame that survives nuisance growth and hidden-family
transfer after active learning, CEGIS, and RAG get equal information is not
obvious.
```

Does that survive "so what?":

```text
Yes, if it works. It would point toward shareable intelligence vaccines:
inspectable lessons that let cheap local systems acquire reliable frames
without renting frontier-scale training or inference.
```

If the honest narrative is boring, say so:

```text
The honest current narrative is still boring until evidence exists. This spec
does not prove FrameSeed. It defines a fair trap: either a T3 frame-transfer gap
survives, or the direction is absorbed by the ordinary methods it resembles.
```