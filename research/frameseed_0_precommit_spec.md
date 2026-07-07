# FRAMESEED-0 Precommit Specification

**Date:** 2026-07-07  
**Status:** PRECOMMIT SPEC ONLY. No implementation, no exploration, no results.  
**Origin:** Q-Loop B32, Q-Loop B33, Dual-Loop Supervisor Check-in #25, W-Loop B25/B26.
**Terminal token required:** one of `FRAMESEED_T3R_SIGNAL`, `FRAMESEED_T3_ABSORBED_BY_TEACHING_DIMENSION`, `FRAMESEED_T3_ABSORBED_BY_REPRESENTATION_PRIOR`, `FRAMESEED_T3_ABSORBED_BY_NUISANCE_ORACLE`, `FRAMESEED_T3_ABSORBED_BY_LIBRARY_LEARNING`, `FRAMESEED_T3_ABSORBED_BY_ACTIVE_LEARNING`, `FRAMESEED_T3_ABSORBED_BY_CEGIS`, `FRAMESEED_T3_ABSORBED_BY_RAG`, `FRAMESEED_T3_BOOLEAN_TRAP`, `FRAMESEED_T3_VOID_SMUGGLED_FRAME`, `FRAMESEED_T3_NEGATIVE`.

## 0. Scope

FRAMESEED-0 now tests one hardened claim:

```text
A compact, inspectable packet can transmit a representation-changing frame to a
bounded cheap learner: not merely choosing a hypothesis already inside the
learner, but installing or exposing a reusable operator, verifier,
transformation, decomposition, intervention generator, or search metric that
reduces teaching cost on a hidden task family.
```

This is a precommitted T3-R filter. A positive Boolean result is not evidence for
cheap general intelligence. Even the strongest positive token claims only:

```text
controlled evidence for amortized frame-teaching separation.
```

Binding givens:

1. Swing for the home run: cheap, ubiquitous, useful AI for people without a
   data center.
2. Stop only when a hostile fresh-eyes reviewer cannot tear the repo down.

All mechanisms are replaceable. The five sacred outcomes from `research/VISION.md`
are fixed: genuine intelligence, improvability, democratized development, data
efficiency, and inference efficiency.

Terminal-token precedence:

1. Any smuggling, parity failure, hidden leakage, or baseline information denial
   -> `FRAMESEED_T3_VOID_SMUGGLED_FRAME`.
2. A missed Boolean escape deadline after a positive Boolean-only result ->
   `FRAMESEED_T3_BOOLEAN_TRAP`.
3. Representation-noncontainment failure ->
   `FRAMESEED_T3_ABSORBED_BY_REPRESENTATION_PRIOR`.
4. L3 full packet below threshold -> `FRAMESEED_T3_NEGATIVE`.
5. Any absorbing baseline at matched or less-than-4x budget -> the most specific
   absorption token, with precedence:
   `FRAMESEED_T3_ABSORBED_BY_TEACHING_DIMENSION`,
   `FRAMESEED_T3_ABSORBED_BY_LIBRARY_LEARNING`,
   `FRAMESEED_T3_ABSORBED_BY_NUISANCE_ORACLE`,
   `FRAMESEED_T3_ABSORBED_BY_CEGIS`,
   `FRAMESEED_T3_ABSORBED_BY_ACTIVE_LEARNING`,
   `FRAMESEED_T3_ABSORBED_BY_RAG`.
6. All T3-R signal gates pass -> `FRAMESEED_T3R_SIGNAL`.
7. Any remaining non-smuggling gate failure -> `FRAMESEED_T3_NEGATIVE`.

## B26 Hardening Iteration Ledger

Each B26 correction below is a binding change to the executable spec, not a
commentary appendix. The detailed operational clauses appear in the sections
that follow.

### B26-I1: Replace Legacy Frame Claim With T3-R

Q-Loop source read: B33 I225, I226, and I231.

Spec modification: the positive object is T3-R, representation-changing frame
transfer. Selection of a hypothesis already reachable inside L0 is absorption,
not signal.

Steelman: a packet that only points at the right existing hypothesis can still
look impressive under nuisance growth. Requiring representation change forces the
claim to be about operational reuse.

Attack: any finite DSL technically contains the target if the search bound is
large enough.

Hostile dismissal executable: the representation-noncontainment certificate must
show bounded non-reachability, no low-cost target-isomorphic primitive, and no
equivalent teaching set over H0.

Alternative interpretation: T3-R may still be a specialized machine-teaching
object; the spec accepts that and measures the amortized gap rather than claiming
new field ownership.

#### NARRATIVE SECTION

The story changes from "a good lesson chose the right rule" to "a tiny lesson
changed what the cheap learner could reuse."

### B26-I2: Teaching Dimension Baseline

Q-Loop source read: B33 I225 and I229.

Spec modification: add an optimal teaching-set baseline over L0's original
hypothesis class H0 with the same packet channel, oracle budget, and hidden eval.

Steelman: if the vaccine is just the shortest set of examples or counterexamples
that identifies the concept in H0, teaching dimension already explains it.

Attack: exact teaching dimension may be expensive.

Hostile dismissal executable: the implementation must run the exact solver where
finite and the frozen approximation otherwise; if either reaches threshold under
matched or less-than-4x budget, emit teaching-dimension absorption.

Alternative interpretation: a teaching set can be a useful engineering artifact,
but it is not the FrameSeed moonshot unless it changes amortized representation
cost.

#### NARRATIVE SECTION

The old literature gets first refusal: if optimal teaching explains the packet,
the exciting label is stripped off.

### B26-I3: Library-Learning Baseline

Q-Loop source read: B33 I225 and I229.

Spec modification: add a DreamCoder-style MDL macro learner that receives the
same target and sibling tasks and may invent reusable frames.

Steelman: reusable abstractions that reduce future sample complexity are exactly
what program-synthesis library learning is built to do.

Attack: a weak library learner could be a strawman.

Hostile dismissal executable: its search budget, task corpus, oracle labels, and
macro-description bits are logged; if it matches AFTD under matched or
less-than-4x total cost, emit library-learning absorption.

Alternative interpretation: library learning may become the honest substrate if
it repeatedly absorbs FrameSeed.

#### NARRATIVE SECTION

If an MDL macro learner invents the frame just as cheaply, the packet was not a
new moonshot object.

### B26-I4: Nuisance-Oracle Fairness Baselines

Q-Loop source read: B33 I227.

Spec modification: add oracle causal-mask, function-only MDL, invariant active
learner, and nuisance-ignoring CEGIS baselines. All optimize functional accuracy,
not reconstruction.

Steelman: the original compactness story is rigged if vaccines can ignore
nuisance while baselines are punished for reconstructing it.

Attack: an oracle causal mask may be too strong.

Hostile dismissal executable: if the edge disappears when baselines can ignore
nuisance by construction, emit nuisance-oracle absorption.

Alternative interpretation: nuisance fairness can make FRAMESEED-0 harder than
the intended final use case, but that is correct for a kill filter.

#### NARRATIVE SECTION

The packet must beat fair functional learners, not lookup tables forced to carry
noise.

### B26-I5: AFTD Metric

Q-Loop source read: B33 I229.

Spec modification: define Amortized Frame Teaching Dimension as frame-install
packet cost divided by the number of sibling tasks whose teaching cost is
reduced after the frame is installed.

Steelman: a single-task win is almost always teaching dimension in disguise.
Amortization across siblings is the first measurable sign of reusable structure.

Attack: sibling tasks can be too similar and inflate amortization.

Hostile dismissal executable: siblings must require the same frame but different
surface functions, and independent teaching-set and library baselines are
measured on the same task bundle.

Alternative interpretation: AFTD is an experimental metric, not a completed
theory of intelligence.

#### NARRATIVE SECTION

The frame earns its name only if one lesson keeps paying rent on new tasks.

### B26-I6: Sibling-Task Transfer Requirement

Q-Loop source read: B33 I226 and I229.

Spec modification: every hidden evaluation includes at least two sibling tasks
requiring the same frame but different surface functions. Post-packet L3 must
improve on those siblings, not only the target task.

Steelman: frame transmission should survive after the original target labels stop
being the whole story.

Attack: siblings may leak the target if generated too close to it.

Hostile dismissal executable: sibling kernels, names, orientations, schemas, and
surface functions are held out and role-permuted independently.

Alternative interpretation: sibling transfer is still synthetic, so it is a
filter before FRAMESEED-SHEETS-0 rather than a public victory.

#### NARRATIVE SECTION

A real frame travels; a target-specific hint stays home.

### B26-I7: Representation-Noncontainment Contract

Q-Loop source read: B33 I226 and I231.

Spec modification: define exactly what it means for L0 not to already know the
frame: bounded non-reachability, no low-cost named primitive, and no equivalent
teaching set.

Steelman: without this contract, every result collapses into selecting the right
primitive, mask, verifier, or DSL expression.

Attack: noncontainment can be impossible to prove absolutely.

Hostile dismissal executable: the certificate is resource-bounded and frozen; if
it fails under the declared budgets, the representation-prior token wins.

Alternative interpretation: a stronger learner may contain the frame; this spec
only tests the bounded cheap learner promised by the moonshot.

#### NARRATIVE SECTION

The learner is not allowed to arrive already vaccinated.

### B26-I8: Boolean Escape Clause

Q-Loop source read: B33 I228 and I230.

Spec modification: FRAMESEED-SHEETS-0 must be specified by W28 and run by W29. No
positive Boolean FRAMESEED-0 result may be used as more than a filter.

Steelman: Boolean worlds invite finite enumeration and PCCP-H repetition.

Attack: forcing a second domain quickly could waste effort after a weak signal.

Hostile dismissal executable: a positive Boolean-only path with no typed second
domain emits `FRAMESEED_T3_BOOLEAN_TRAP`.

Alternative interpretation: if FRAMESEED-0 is negative or absorbed, the second
domain may be redesigned or cancelled; the escape clause binds positive Boolean
claims.

#### NARRATIVE SECTION

The Boolean toy may open the door, but it cannot be the house.

### B26-I9: Hardened Verdict Tokens

Q-Loop source read: B33 I225-I231 and Supervisor Check-in #25.

Spec modification: add `FRAMESEED_T3R_SIGNAL`, teaching-dimension,
representation-prior, nuisance-oracle, library-learning, and Boolean-trap tokens
with precedence.

Steelman: tokens keep the loop from turning mixed evidence into narrative.

Attack: too many tokens can make the outcome look bureaucratic.

Hostile dismissal executable: each boring explanation has a named terminal path
with exact conditions.

Alternative interpretation: secondary diagnostics may be scientifically useful,
but they cannot override the token.

#### NARRATIVE SECTION

The run leaves a verdict, not a mood.

### B26-I10: Claim Ceiling

Q-Loop source read: B33 I230 and I231.

Spec modification: even on `FRAMESEED_T3R_SIGNAL`, the maximum claim is
"controlled evidence for amortized frame-teaching separation."

Steelman: the moonshot ethos needs ambition, but public overclaim would recreate
the failure mode the loop exists to prevent.

Attack: a low claim ceiling may make the result sound small.

Hostile dismissal executable: any stronger claim is a documentation failure and
cannot be supported by FRAMESEED-0 alone.

Alternative interpretation: the public story can stay vivid, but the scientific
claim remains bounded until FRAMESEED-SHEETS-0 and later domains pass.

#### NARRATIVE SECTION

If the toy works, it earns the next test, not a manifesto victory lap.

---

## I225: World Design

### Steelman

The world must make observational surface learning ambiguous while making a
small interventional frame decisive. Nuisance entropy must not create a fake gap:
every baseline is allowed to ignore nuisance and optimize functional accuracy.
The Boolean domain is a kill filter for T3-R, not a publication claim.

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

### Sibling Task Family

For every hidden target task, generate at least two sibling tasks:

```text
Siblings(W) = {W_s1, W_s2, ...}, count >= 2
```

Sibling tasks share the same reusable frame type: intervention distinguishes
causal coordinates from aliases and nuisance coordinates. They must differ from
the target by at least one surface function:

```text
- different admitted kernel K_s, or
- different output relation over the same causal roles, or
- different composed-intervention obligation, or
- different verifier/invariant surface clause.
```

They also resample names, role permutations, orientations, and nuisance slots.
The target packet may install the frame, but sibling success must be measured
with separate sibling labels absent or restricted to the declared residual
teaching budget. A target-only improvement is not a T3-R signal.

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
requirements, then incorporates Batch 33 I226/I229 by adding sibling transfer as
a first-class world requirement.

### Alternative Interpretation

The first experiment could start outside Boolean worlds. The approved path is a
CPU-first synthetic absorption filter, with the Boolean escape clause binding any
positive result.

### Right Experiment Check

This is the right experiment for this iteration only if the confirm/kill/void
criteria make at least one hostile dismissal executable. The token path remains
locked to the hardened terminal verdicts; narrative appeal cannot override them.

### Attack
```text
You made a puzzle whose answer is just "use the causal slots."
```

Correct risk. Therefore slot ids, interventions, clauses, programs, sibling
transfer, teaching-dimension baselines, and nuisance-oracle baselines are all
counted or run. If the answer is just a support list, the run is absorbed.

### Confirm / Kill / Void

Confirm if observational equivalence is exact, decisive interventions exist for
every hidden world, sibling tasks require the same frame but different surface
functions, hidden families are held out, and nuisance bits are independent.
Kill/redesign if raw decision trees from no-intervention data reach 0.95 HFA or
sibling tasks are target clones. Void if names, order, metadata, or hidden splits
leak role labels.

### NARRATIVE SECTION

Gossip story: the tiny learner sees worlds that look identical, and the vaccine
must teach a reusable surgical question that keeps working on sibling tasks.

It survives "isn't that obvious?" and "so what?" only if the intervention frame
stays compact as nuisance bits grow, travels to siblings, and survives the fair
boring baselines.

If boring: the Boolean world is boring by itself. It matters only as a clean trap
for absorption before typed practical domains.

---

## I226: Learner Architecture

### Steelman

L3 must be able to ingest rich teaching packets, but the pre-packet learner L0
must not already contain the target frame as a cheap primitive, feature mask,
verifier, transformation, or reachable hypothesis. The experiment tests
representation-changing frame transfer, not hypothesis selection.

### L0: Pre-Packet Learner State

Define the bounded learner before the packet:

```text
L0 = (R0, H0, A0, B0)
```

Where:

- `R0`: representation language and primitive operations available before the
  packet.
- `H0`: hypotheses reachable by `A0` inside the declared budgets.
- `A0`: update/search procedure before representation patching.
- `B0`: packet length, query count, search depth, runtime, and description
  budget.

`H0` is the hypothesis class used by the teaching-dimension baseline. Any
post-packet operation not in `R0` must be encoded, counted, and audited.

### Representation-Noncontainment Contract

A frame `F` is not already known by L0 only if all three conditions hold:

```text
1. Bounded non-reachability:
   No h in Reach(L0, public_schema, public_data, B0) reaches the target and
   sibling hidden-family threshold.

2. No low-cost named primitive:
   R0 contains no primitive, feature, role name, verifier, intervention
   generator, transformation, decomposition, mask, or tie-break rule whose
   semantics are isomorphic to F under role permutation at cost <= B0.

3. No equivalent teaching set:
   The optimal teaching set over H0, using the same packet channel and budget,
   cannot induce target plus sibling transfer at matched or less-than-4x cost.
```

If any condition fails, emit
`FRAMESEED_T3_ABSORBED_BY_REPRESENTATION_PRIOR` or
`FRAMESEED_T3_ABSORBED_BY_TEACHING_DIMENSION`, according to the first passing
absorber.

### L3: Packet-Conditioned Generic Finite Learner

```text
L3(packet V, public_schema Sigma) -> predictor P
P(query q) -> bit
```

L3 cannot make active queries. It learns only from the packet and public schema.

`Sigma` contains surface slot count, bit slot type, intervention grammar, query
grammar, label type, packet grammar, canonical decoder, and the generic
representation language. It does not contain causal/spurious/nuisance labels,
alias maps, kernel identity, hidden family id, generator seed, target-slot
selectors, or sibling family ids.

### Required Inputs

L3 must ingest:

```text
example(masked_observation, intervention, label)
counterexample(candidate_program, query, expected_label, actual_label)
invariant(transform_schema, context_schema, output_relation)
transform(name_token, operation_schema, admissibility_scope)
verifier_clause(clause_id, finite_obligation_schema)
representation_patch(kind, ast_or_schema, declared_cost, scope)
program(ast, declared_cost, inputs, outputs) optional
```

`name_token` is a random identifier, not semantic prose.

### Generic Representation: FIR-0

Sorts:

```text
Bit, Slot, Observation, Edit, Query, Set[Slot], Program, FramePatch
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
apply_patch(frame_patch, program_or_query)
```

`truth_table_2` is a generic 4-bit table interpreter, not named XOR/AND/OR or a
target primitive. `apply_patch` only applies an explicitly serialized and counted
representation patch; it has no hidden semantics.

Banned primitives include:

```text
causal, spurious, nuisance, alias, true_role, target_kernel,
select_causal_pair, hidden_family, sibling_family, generator_seed,
rho, beta, pi, oracle_label, causal_mask_oracle
```

Natural-language rule interpretation is banned. Executable packet entries must
compile to FIR-0.

### L3 Algorithm

1. Decode packet canonically.
2. Reject any opcode outside FIR-0.
3. Build constraints from examples, counterexamples, invariant clauses, verifier
   clauses, transformation hints, and representation patches.
4. Candidate pool = supplied compact programs plus all FIR-0 programs and
   patch-augmented programs with total cost `<= B_L3`.
5. Default bound: `B_L3 = min(128, packet_bit_length)` unless overridden before
   hidden evaluation.
6. Filter candidates by all packet constraints.
7. Choose minimum-cost consistent candidate or minimum-cost patched search metric.
8. Break ties by public role-blind canonical AST serialization after slot
   renaming.
9. If no candidate survives, output packet majority label.

L3 is weak because it has no active queries, hidden labels, natural language,
unbounded synthesis, generator access, named target roles, or uncounted frame
library.

### T3-R Operational Definition

A packet `V` is T3-R only if applying it to L0 creates L3 with at least one
counted reusable change:

```text
- a new operator;
- a verifier;
- a transformation;
- a decomposition;
- an intervention generator;
- a search metric;
- an MDL macro usable across sibling tasks.
```

Selection is when L0 chooses `h in H0` without changing representation,
intervention language, verifier set, search metric, or future task teaching
cost. Transmission is when the counted packet reduces future sibling-task
teaching cost after the target task is removed.

### Data-Reading Check

This incorporates Q-Loop B33 I226/I231: L0 is explicit, representation
noncontainment is resource-bounded, and T3-R is defined as representation-changing
frame transfer rather than hypothesis selection.

### Alternative Interpretation

A small neural learner might be more realistic, but it would blur whether the
packet or the learned prior supplied the frame. FRAMESEED-0 starts with a finite
learner and moves to typed domains only after the Boolean filter.

### Right Experiment Check

This is the right experiment for this iteration only if the confirm/kill/void
criteria make at least one hostile dismissal executable. The token path remains
locked to the hardened terminal verdicts; narrative appeal cannot override them.

### Attack
```text
L3 already knows the frame; the packet just points at it.
```

That is now a terminal failure. The representation-noncontainment certificate,
optimal teaching baseline, library-learning baseline, and packet-erasure
ablations decide whether the frame was transmitted or merely selected.

### Confirm / Kill / Void

Confirm if every accepted packet compiles to FIR-0, no primitive names latent
roles, the noncontainment contract passes, and behavior is role/name invariant.
Kill/redesign if L3 solves hidden worlds from observational examples alone or
succeeds after frame-component erasure. Void if FIR-0 contains answer-shaped
primitives or hidden metadata.

### NARRATIVE SECTION

Gossip story: the learner is deliberately ignorant, and the packet must install a
reusable way to think without whispering the answer in a secret language.

It survives only if the same ignorant learner gets a counted operational
affordance that ordinary teaching, search, and library learning do not cheaply
reproduce.

If boring: if L3 is merely CEGIS, teaching dimension, or a preinstalled primitive
with branding, the run is absorbed.

---

## I227: Vaccine Packet Format

### Steelman

The packet is the experimental object. It may teach through examples, targeted
interventions, counterexamples, invariant hints, verifier clauses, representation
patches, and optional compact programs. No channel gets free bits, and no patch
may smuggle target roles.

### Packet Object

```text
V = (header, E, I, CE, H, VC, RP, CP)
```

Where:

- `header`: version, schema hash, surface slot count, L0 hash, H0 hash.
- `E`: observational or masked examples.
- `I`: targeted intervention examples.
- `CE`: counterexamples to candidate rules.
- `H`: invariant and transformation hints.
- `VC`: packet-level verifier clauses.
- `RP`: representation patches, macro schemas, intervention generators, or
  search-metric updates.
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

Representation patch:

```text
representation_patch(
  kind = operator | verifier | transform | decomposition |
         intervention_generator | search_metric | macro,
  ast_or_schema,
  declared_cost,
  admissibility_scope)
```

Verifier clause:

```text
verifier_clause(clause_id, finite_scope, required_relation)
```

Compact program:

```text
program(ast, declared_cost)
```

If a full program solves only the target and CEGIS/RAG can use it under budget,
the run is absorbed. If a representation patch contains hidden metadata, target
roles, or target-isomorphic names, the run is void or representation-prior
absorption depending on whether the leak is in the packet or L0.

### Packet Length Metric

All systems use the same canonical binary serialization:

```text
|V| = bit length of canonical packet serialization
```

Costs:

```text
slot_bits       = ceil(log2(d))
opcode_bits     = ceil(log2(number_of_packet_opcodes))
bit_value       = 1 bit
truth_table_2   = opcode + 4 bits
edit            = opcode + slot_bits + 1
label           = 1 bit
set_literal     = opcode + count_bits + k * slot_bits
set_complement  = opcode + count_bits + k * slot_bits
program_ast     = preorder opcode serialization plus literal costs
frame_patch_ast = kind opcode + serialized schema/AST + declared scope bits
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

### Packet-Erasure Requirement

For T3-R, the frame component must be separable from examples:

```text
V_examples_only: remove RP, verifier, and invariant components; spend the same
                 bits on examples.
V_patch_only:    keep RP/verifier/transform components; remove target examples
                 except the minimum declared install checks.
V_no_patch:      remove RP; keep all ordinary examples/counterexamples.
```

Signal requires the full packet to beat examples-only and no-patch variants on
sibling transfer. Patch-only may be imperfect, but it must improve sibling
teaching cost relative to L0.

### Data-Reading Check

Batch 32 required counted packet length. Batch 33 I226/I229 require the packet to
make representation change explicit and auditable rather than hide it in prose or
a full target program.

### Alternative Interpretation

Tokenizing natural-language lessons would be more human-like, but it would
import uncontrolled priors. FRAMESEED-0 uses formal packets only.

### Right Experiment Check

This is the right experiment for this iteration only if the confirm/kill/void
criteria make at least one hostile dismissal executable. The token path remains
locked to the hardened terminal verdicts; narrative appeal cannot override them.

### Attack
```text
The packet wins because the answer is hidden in an uncounted clause, prose, or
macro.
```

Canonical serialization is the answer: every slot id, set, clause, AST node,
patch schema, and scope bit is on the bill.

### Confirm / Kill / Void

Confirm if the same serializer is used for vaccines and baselines, all used
fields are counted, `P(m)` passes sublinear growth, and packet-erasure separates
frame transmission from example selection. Kill/redesign if success requires full
observations or a full program that CEGIS absorbs. Void if any learner uses
uncounted information.

### NARRATIVE SECTION

Gossip story: the vaccine is a tiny formal lesson, and every bit it uses to
change the learner goes on the bill.

It survives only if the bill stays small while nuisance entropy grows and the
representation patch helps on siblings, not just the target.

If boring: if the packet is just a compressed answer key, the result is absorbed
or void.

---

## I228: Baseline Specifications

### Steelman

Every hostile dismissal must become executable. The baselines get equal
information, matched budgets, the same hidden evaluation, the same sibling-task
bundle, and functional objectives. No baseline is forced to reconstruct nuisance.

### Shared Information And Budgets

For a vaccine packet `V`:

```text
B_bits = |V|
B_queries = number of oracle labels used to construct V
B_tasks = target task plus sibling tasks exposed to the packet constructor
```

Every baseline receives the same public schema and the same packet entries, or a
lossless canonical translation. If a baseline ignores a field by design, the
report must say so; the field is still provided and counted.

Matched baselines satisfy:

```text
total_information_bits <= B_bits
oracle_query_count <= B_queries
task_access <= B_tasks
```

Also run 2x and 4x curves. A baseline absorbs if it reaches threshold at matched
budget or at less than 4x budget. If it needs exactly 4x or more, report the
ratio but do not emit absorption.

### Functional Objective Clause

All baselines optimize:

```text
hidden-family functional accuracy + description/query cost
```

No baseline may be scored on reconstructing nuisance bits, aliases, names, slot
order, or full observations unless that reconstruction is also required of L3.

### TD-H0: Optimal Teaching Dimension Baseline

TD-H0 tests "this is just an optimal teaching set."

Inputs: L0's original representation `R0`, original reachable hypothesis class
`H0`, update procedure `A0`, packet grammar, public schema, target task, and the
same sibling-task bundle.

It searches for the shortest packet over the original channel that identifies a
hypothesis in `H0` and reaches hidden threshold. Allowed packet fields are
examples, counterexamples, and queries expressible without representation
patches. It may not add operators, verifiers, transformations, decompositions,
intervention generators, search metrics, or macros outside `R0`.

Run exact finite search where feasible. Otherwise run the frozen approximate
solver declared before hidden evaluation and report its approximation bound.

Absorbs if target plus sibling HFA reaches threshold under matched or
less-than-4x budget, or if its independent teaching-set total beats or matches
AFTD within the absorption threshold. If TD-H0 absorbs, the packet is teaching
dimension, not T3-R.

### L0: Rote Example / Nearest Neighbor

Input: all packet examples, intervention examples, and counterexamples as
labeled query records. Invariant and transformation hints are converted only
into explicitly generated labeled records if they enumerate finite records within
budget. Compact programs and representation patches are stored as opaque records
and not executed.

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
  no latent role predicates;
  no representation patches outside H0.
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

### Nuisance-Oracle Fairness Suite

The suite tests "your compactness gap is a nuisance-format trick."

`O0: Oracle causal-mask baseline`

Gets the true non-nuisance coordinate set `{c0,c1,s0,s1}` and the true causal
coordinate set `{c0,c1}` as anonymous slot ids, but not `K`, labels beyond the
matched budget, hidden family id, or target program. It optimizes the functional
target over the masked coordinates.

`O1: Function-only MDL feature-subset baseline`

Searches feature subsets, intervention dependencies, and truth-table functions
under the same description budget. It is rewarded only for hidden functional
accuracy and short description length, never for reconstructing nuisance.

`O2: Invariant active learner`

May spend its matched query budget on interventions designed to reject nuisance
or alias dependence. It uses expected functional version-space reduction, not
observation reconstruction.

`O3: CEGIS with nuisance-ignoring objective`

Runs CEGIS over functional error and program length only. Counterexamples are
chosen to distinguish target behavior, not to reconstruct full records.

`O4: Randomized nuisance relabeling control`

Nuisance bits are resampled across equivalent cases so every method can detect
statistical non-use of nuisance. A method that depends on nuisance should fail
this control.

If any O-baseline reaches the target plus sibling threshold under matched or
less-than-4x budget, emit `FRAMESEED_T3_ABSORBED_BY_NUISANCE_ORACLE` unless a
higher-precedence prior-art token also applies.

### RAG Baseline: Few-Shot Retrieval

Corpus: every packet entry serialized as canonical text and canonical fields.
Index by mentioned slots, values, intervention slots, opcode ids, AST nodes, and
representation-patch ids.

Retrieve top `k in {1,3,5,8}`, selected on seen-family validation and frozen, by
weighted Jaccard overlap.

Variants:

1. `RAG-NN`: majority label from retrieved labeled queries.
2. `RAG-CLAUSE`: applies retrieved invariant clauses only when they map the test
   query to a retrieved labeled query.
3. `RAG-PROG`: executes a retrieved compact FIR-0 program if present.
4. `RAG-PATCH`: reuses a retrieved representation patch only by exact canonical
   match and with its bits counted.

The strongest variant is the RAG baseline. Absorbs if `HFA >= 0.95` under
matched or less-than-4x budget. If `RAG-PROG` or `RAG-PATCH` absorbs because the
packet contains the full executable solution, report RAG absorption unless CEGIS
or teaching dimension also absorbs.

### Library-Learning Baseline: MDL Macro Learner

The library learner tests "this is just reusable macro discovery."

Inputs: the same target task, at least two sibling tasks, packet entries, public
schema, oracle query budget, and hidden evaluation split. It may invent reusable
FIR-0 macros, transformation schemas, verifier fragments, or intervention
policies if their description length is counted.

Objective:

```text
minimize total_description_length(library + per_task_programs + queries)
subject to hidden-family functional threshold.
```

A DreamCoder-style wake/sleep or enumerative MDL learner is acceptable if frozen
before hidden evaluation. It must not receive target role names, generator seeds,
or hidden labels.

Absorbs if it reaches target plus sibling threshold or matches AFTD under matched
or less-than-4x total description/query cost. If it absorbs, FrameSeed is library
learning, not a separate T3-R signal.

### Data-Reading Check

Batch 32 named active learning, CEGIS, and RAG. Batch 33 I225/I227/I229 add
teaching dimension, nuisance-oracle fairness, and library learning. This section
gives all of them equal information and clean win conditions.

### Alternative Interpretation

A neural or LLM baseline will matter later. FRAMESEED-0 is CPU-first and uses
formal baselines; an added neural baseline must receive the same packet access
and be reported separately.

### Right Experiment Check

This is the right experiment for this iteration only if the confirm/kill/void
criteria make at least one hostile dismissal executable. The token path remains
locked to the hardened terminal verdicts; narrative appeal cannot override them.

### Attack
```text
Your baselines are handicapped because they cannot read the same packet or ignore
nuisance the way the vaccine can.
```

They receive the same packet or lossless translations, and the nuisance suite is
explicitly allowed to ignore nuisance and optimize function.

### Confirm / Kill / Void

Confirm if TD-H0, L0, L1, L2, RAG, nuisance-oracle baselines, and library
learning run on the same hidden target/sibling cases, budgets, packets, and role
permutations as L3. Kill or radically reframe if multiple prior-art baselines
absorb. Void if a baseline lacks information L3 used or budget accounting is
inconsistent.

### NARRATIVE SECTION

Gossip story: the vaccine has to beat the boring explanations after the boring
explanations are handed the same evidence and fair functional objectives.

It survives only if optimal teaching, active learning, synthesis, retrieval,
nuisance-oracle methods, and library learning all get fair shots and still cannot
buy the same amortized frame transfer cheaply.

If boring: if the prior-art baselines absorb, FrameSeed is boring as a moonshot
and should be killed or radically reframed.
---

## I229: Measurement Protocol

### Steelman

The measurement must force a real T3-R claim: hidden-family transfer, sibling
transfer, amortized frame-teaching cost, sublinear packet growth, equal-budget
baselines, ablations, role permutation stability, representation-noncontainment,
and smuggling audit before signal.

### Evaluation Size

For each `m in {4,16,64,256}`:

```text
hidden_worlds_per_m >= 64
role_permutations_per_world >= 10
hidden_eval_queries_per_world >= 512
sibling_tasks_per_world >= 2
```

Hidden queries are balanced:

```text
20% no intervention
20% causal-slot single edit
20% spurious-alias single edit
20% nuisance single edit
20% composed edits
```

Invalid worlds or sibling tasks are resampled before learners see them.

### Primary Accuracy Metric

```text
HFA(system, m, hidden_family, task) = correct hidden predictions / total hidden queries
```

`FRAMESEED_T3R_SIGNAL` requires:

```text
HFA(L3_full, m, hidden_family, target) >= 0.95 for every m and hidden-family class
HFA(L3_full, m, hidden_family, sibling_i) >= 0.95 for at least two siblings
mean HFA(L3_full) >= 0.97 overall across target and counted siblings
```

### Frame Teaching Dimension Metrics

For a task `t`, define:

```text
TD_H0(t) = minimum counted packet length that makes L0 choose a hypothesis in H0
           reaching threshold on t, with no representation additions.
```

For a frame packet `V_frame` and sibling set `S`, define a sibling as reduced if:

```text
TD_after(V_frame, sibling) <= 0.50 * TD_H0(sibling)
and HFA(L3_after_frame, sibling) >= 0.95
```

Amortized Frame Teaching Dimension:

```text
AFTD(V_frame, S) = |V_frame| / count_reduced_siblings(S)
```

If `count_reduced_siblings(S) < 2`, no T3-R signal. Also report all-in cost:

```text
AFTD_all_in = (|V_frame| + sum residual sibling teaching bits) /
              count_reduced_siblings(S)
```

Signal requires the frame packet to beat independent teaching sets and library
learning on the same bundle:

```text
AFTD(V_frame, S) < 0.25 * mean_i TD_H0(sibling_i)
```

or the equivalent query-cost ratio, unless a baseline cannot reach threshold at
4x budget. If teaching dimension or library learning matches under matched or
less-than-4x cost, emit the corresponding absorption token.

### Budget Matching

For every system:

```text
total_information_bits = packet bits
                       + serialized oracle query bits
                       + serialized oracle answer bits
                       + final executable program bits if used at inference
                       + learned library/macro bits
                       + residual sibling teaching bits
```

Matched baselines must not exceed L3 full vaccine bits or query count. Report 1x,
2x, and 4x curves:

```text
bits_to_0.95(system, target_and_siblings)
queries_to_0.95(system, target_and_siblings)
ratio_bits = bits_to_0.95(system) / bits_to_0.95(L3_full)
ratio_queries = queries_to_0.95(system) / queries_to_0.95(L3_full)
```

A baseline absorbs if either ratio is less than 4 and it reaches 0.95.

### Nuisance Growth Curve

For each `m`, report:

- vaccine packet length `P(m)`;
- frame-install packet length `P_frame(m)`;
- L3 HFA on target and siblings;
- AFTD and AFTD_all_in;
- TD-H0, L0, L1, L2, RAG, nuisance-oracle suite, and library-learning HFA at
  1x, 2x, 4x;
- baseline bits/queries needed for 0.95, if reached;
- ablation HFA;
- role permutation stability.

Sublinear packet growth from I227 is mandatory. A constant vaccine-vs-reconstruction
gap is not sufficient; function-aligned nuisance baselines must also fail or pay
at least 4x.

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

V_no_representation_patch:
  remove RP entries; replace their bit budget with ordinary labeled examples.

V_examples_only:
  keep only examples/intervention examples within the same bit budget.
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

For sibling tasks, examples-only and no-representation-patch ablations must fail
the AFTD reduction requirement. If an ablation remains at or above 0.95 on target
and siblings, no T3-R signal.

### Role Permutation Stability

For each permutation bundle:

```text
terminal token identical
HFA std <= 0.02
packet length coefficient of variation <= 0.10
AFTD coefficient of variation <= 0.10
```

If instability is caused by leakage, void. If no leakage is found, negative.

### Secondary Metrics

Report packet construction CPU time, L3 inference CPU time, final AST cost,
learned macro cost, number of slot ids mentioned, counts of packet entry types,
randomized-label control accuracy, noncontainment certificate result, and each
baseline's best budget ratio. These diagnostics cannot override terminal tokens.

### Data-Reading Check

This implements the original threshold, matched budgets, four nuisance values,
20pp ablations, and role permutation stability, then adds Batch 33 I226/I229's
AFTD and sibling-task transfer requirements.

### Alternative Interpretation

Averaging over hidden cases would be easier. FRAMESEED-0 requires every nuisance
size and hidden-family class to pass because hostile review will attack weak
subfamilies.

### Right Experiment Check

This is the right experiment for this iteration only if the confirm/kill/void
criteria make at least one hostile dismissal executable. The token path remains
locked to the hardened terminal verdicts; narrative appeal cannot override them.

### Attack
```text
The effect is a one-task hint that disappears under nuisance growth, sibling
transfer, or frame erasure.
```

The curve, AFTD, siblings, ablations, and per-family thresholds make that
executable.

### Confirm / Kill / Void

Confirm if L3 reaches all thresholds, AFTD passes, packet growth is sublinear,
baselines fail absorption, ablations drop by 20pp, and role permutation is
stable. Kill/reframe if signal exists only at small `m`, only on the target,
without AFTD, or with baselines needing only slightly more budget. Void if hidden
eval is contaminated or budgets differ.

### NARRATIVE SECTION

Gossip story: the vaccine has to keep working when the world gets noisy,
renamed, rearranged, stripped of its favorite lesson parts, and asked to solve
siblings.

It survives only if the measured gap is amortized, robust across nuisance growth,
and tested against fair baseline budgets.

If boring: a single accuracy number is boring; AFTD, sibling transfer, and
ablations are the actual experiment.

---

## I230: Representation-Noncontainment And Smuggling Audit

### Steelman

A T3-R signal dies if the frame is already inside names, metadata, DSL primitives,
verifier clauses, learner priors, compact programs, baseline handicaps, or the
original hypothesis class H0. The audit must certify both no smuggling and no
low-cost representation prior.

### Representation-Noncontainment Certificate

Before hidden evaluation, freeze and publish:

```text
R0 primitive list
H0 hypothesis grammar
A0 update/search algorithm
B0 packet/query/search/runtime/description budgets
Reach(L0, public_schema, public_data, B0) procedure
teaching-dimension solver or approximation
role-isomorphism test for low-cost primitives
```

The certificate passes only if:

```text
1. Bounded non-reachability passes:
   no reachable h in H0 reaches 0.95 on target plus siblings under B0.

2. No low-cost named primitive passes:
   no primitive, feature, mask, verifier, transformation, intervention generator,
   decomposition, macro, or tie-break in R0 is isomorphic to the frame under role
   permutation at cost <= B0.

3. No equivalent teaching set passes:
   TD-H0 cannot induce target plus sibling transfer at matched or less-than-4x
   packet/query budget.
```

If condition 1 or 2 fails, emit
`FRAMESEED_T3_ABSORBED_BY_REPRESENTATION_PRIOR`. If condition 3 fails, emit
`FRAMESEED_T3_ABSORBED_BY_TEACHING_DIMENSION`.

### Smuggling Channels To Enumerate

1. Named roles: surface names or name statistics reveal causal/spurious/nuisance
   roles.
2. Hidden labels: labels embedded in packet order, ids, hashes, seeds, family ids,
   or sibling ids.
3. DSL primitives: causal/spurious/nuisance/alias/target selectors or kernel
   primitives.
4. Representation priors: low slot ids, first-mentioned slots, tie-breaks,
   masks, verifiers, or library entries correlate with causal roles.
5. Verifier clauses: public clauses equivalent to the hidden target through
   uncounted information.
6. Compact programs or patches: hardcoded hidden family ids, seeds, role maps,
   target program, or target-isomorphic macros with uncounted slot maps.
7. Baseline handicaps: baselines denied packet fields, query access, budgets,
   sibling tasks, or hidden cases used by L3.
8. Split leakage: hidden seeds inspected before freezing hyperparameters.
9. Evaluation leakage: hidden eval queries or labels included in packet.
10. Human-labor leakage: manual packet edits or threshold changes after hidden
    inspection.

### Audit Procedure

Run before declaring any signal.

#### A. Static Generator Audit

Over at least 10,000 generated worlds and siblings, test:

```text
MI(surface_name, latent_role) ~= 0
MI(slot_index, latent_role) ~= 0
MI(packet_order, latent_role) only through counted slot ids
MI(sibling_id, target_role_map) ~= 0
```

Any detectable uncounted role leak voids.

#### B. Packet Serialization Audit

For every packet:

- parse canonical binary serialization;
- list every slot id, opcode, AST node, name token, clause, patch schema, and
  declared scope;
- recompute packet length independently;
- confirm no executable field contains banned strings or hidden metadata;
- confirm ablation replacements respect equal or lower bit budgets.

#### C. DSL, Learner, And Representation-Prior Audit

For L0, L3, L1, L2, RAG, nuisance-oracle baselines, and library learning:

- enumerate allowed primitives and macros;
- search source/config for banned role terms;
- run randomized-role worlds;
- run randomized-label worlds;
- run the role-isomorphism test against each low-cost primitive;
- run Reach(L0, public data, B0) and log failures.

Randomized-label control:

```text
Replace labels with independent fair bits.
No system may exceed 0.60 HFA.
```

#### D. Baseline Parity Audit

Verify same packet facts, budgets, hidden target cases, sibling cases, role
permutations, and query oracle restrictions for every system.

#### E. Role Permutation Audit

Remap slots, resample names, rerun packet construction and all learners, compare
HFA, AFTD, and terminal tokens.

#### F. Human-Labor Ledger

Record:

```text
DESIGNED_BY_HUMANS: generator, packet grammar, learner grammar, baseline specs
FROZEN_BEFORE_HIDDEN: kernel split, hyperparameters, constructor, audits,
                      H0, B0, TD-H0 solver, library learner
DISCOVERED_OR_SELECTED_BY_PACKET: slot ids, examples, interventions, clauses,
                                  representation patches
HIDDEN_FOR_EVAL: hidden worlds, labels, role maps, query labels, sibling tasks
```

Manual change after hidden inspection voids unless the run restarts with new
hidden seeds.

### Data-Reading Check

PCCP-H showed that DSLs, verifiers, roles, and decomposition choices can smuggle
answers. Batch 33 I226 requires the stronger noncontainment certificate before a
T3-R interpretation.

### Alternative Interpretation

A packet may mention slot ids; that is not automatically smuggling. The audit
bans uncounted or semantically named role information and separately checks
whether the cheap learner already had the frame.

### Right Experiment Check

This is the right experiment for this iteration only if the confirm/kill/void
criteria make at least one hostile dismissal executable. The token path remains
locked to the hardened terminal verdicts; narrative appeal cannot override them.

### Attack
```text
The frame was not transmitted; it was preinstalled.
```

The answer is not trust. It is bounded reachability, primitive-isomorphism tests,
teaching-dimension comparison, static audit, randomized labels, role permutation,
and baseline parity.

### Confirm / Kill / Void

Confirm only if all leakage tests pass, the noncontainment certificate passes,
randomized labels fail, role permutation is stable, and parity is documented.
Kill/redesign if L3 succeeds without frame components. Void if hidden labels,
seeds, roles, or baseline parity leak.

### NARRATIVE SECTION

Gossip story: before the vaccine gets credit, it is searched for hidden needles
in the labels, names, grammar, rules, and the learner's own prior.

It survives only if an enemy can see that the frame was neither smuggled nor
preinstalled.

If boring: audits are boring until they fail; if they fail, the positive story
was fake.

---

## I231: Verdict Token Specification

### Steelman

The run must end in a token, not a debate. The token must identify whether the
T3-R claim survived or which boring explanation absorbed it.

### `FRAMESEED_T3R_SIGNAL`

Emit only if all hold:

1. Smuggling audit passes.
2. Representation-noncontainment certificate passes.
3. L3 full packet has `HFA >= 0.95` for every `m`, hidden-family class, target,
   and at least two sibling tasks.
4. L3 mean HFA is `>= 0.97` overall across target and counted siblings.
5. Packet growth is sublinear by I227.
6. AFTD passes by I229.
7. TD-H0 does not reach threshold or match AFTD under matched or less-than-4x
   budget.
8. L0 does not reach 0.95 under matched budget.
9. L1 does not reach 0.95 under matched or less-than-4x budget.
10. L2 does not reach 0.95 under matched or less-than-4x budget.
11. RAG does not reach 0.95 under matched or less-than-4x budget.
12. Nuisance-oracle suite does not reach 0.95 under matched or less-than-4x
    budget.
13. Library-learning baseline does not reach 0.95 or match AFTD under matched or
    less-than-4x cost.
14. Each primary ablation drops at least 20pp as specified and no-patch/examples-only
    ablations fail sibling AFTD.
15. Role permutation stability passes.
16. All packet, query, final-program, library, macro, frame-patch, and
    verifier-clause bits are counted.
17. Boolean escape clause is satisfied for any positive public use beyond an
    internal filter.

Allowed claim: controlled evidence for amortized frame-teaching separation.
Banned claim: FrameSeed proves cheap general intelligence, intelligence vaccines
work generally, or scale has been defeated.

### `FRAMESEED_T3_ABSORBED_BY_TEACHING_DIMENSION`

Emit if the audit passes and TD-H0 reaches target plus sibling threshold, or
matches AFTD, under matched or less-than-4x packet/query budget. Interpretation:
the packet is an optimal teaching set over the original hypothesis class.

### `FRAMESEED_T3_ABSORBED_BY_REPRESENTATION_PRIOR`

Emit if bounded reachability succeeds before the packet, or R0 contains a
low-cost primitive, mask, verifier, transformation, intervention generator,
decomposition, macro, or tie-break isomorphic to the target frame under role
permutation. Interpretation: the learner already knew the frame.

### `FRAMESEED_T3_ABSORBED_BY_NUISANCE_ORACLE`

Emit if any nuisance-oracle fairness baseline reaches target plus sibling
threshold under matched or less-than-4x budget. Interpretation: the gap came from
allowing the vaccine to ignore nuisance while weaker baselines were not equally
function-aligned.

### `FRAMESEED_T3_ABSORBED_BY_LIBRARY_LEARNING`

Emit if the MDL macro/library learner reaches target plus sibling threshold or
matches AFTD under matched or less-than-4x total description/query cost.
Interpretation: reusable abstraction learning absorbs the frame packet.

### `FRAMESEED_T3_ABSORBED_BY_ACTIVE_LEARNING`

Emit if the audit passes and L1 reaches `HFA >= 0.95` under matched or
less-than-4x bit/query budget with role stability. Interpretation: active
curriculum inside a supplied hypothesis class.

### `FRAMESEED_T3_ABSORBED_BY_CEGIS`

Emit if the audit passes and L2 reaches `HFA >= 0.95` under matched or
less-than-4x budget using only the predeclared generic DSL with role stability.
Interpretation: ordinary program synthesis with supplied constraints.

### `FRAMESEED_T3_ABSORBED_BY_RAG`

Emit if the audit passes and any RAG variant reaches `HFA >= 0.95` under matched
or less-than-4x budget with role stability. Interpretation: few-shot retrieval,
not frame transfer.

### `FRAMESEED_T3_BOOLEAN_TRAP`

Emit if a positive Boolean-only result is used as more than an internal filter,
or if `FRAMESEED_T3R_SIGNAL` occurs in FRAMESEED-0 and the repo does not specify
FRAMESEED-SHEETS-0 by W28 and run it by W29. Interpretation: PCCP-H's Boolean toy
failure mode has repeated.

### `FRAMESEED_T3_VOID_SMUGGLED_FRAME`

Emit if any of these occur:

- role names, hidden labels, seeds, family ids, sibling ids, or target primitives
  enter any learner or packet;
- names, slot order, packet order, orientation, or sibling construction leaks
  roles;
- L3 or a baseline contains banned primitives;
- public verifier clauses expose hidden target through uncounted information;
- compact programs or representation patches contain hidden metadata;
- baselines are denied information used by L3;
- hidden split is inspected before freezing hyperparameters;
- randomized-label control exceeds 0.60 HFA;
- role permutation exposes uncounted dependency.

Interpretation: no scientific claim; redesign and rerun with new hidden seeds.

### `FRAMESEED_T3_NEGATIVE`

Emit if the audit passes but L3 fails threshold, packet growth is not sublinear,
AFTD fails, sibling transfer fails, ablations do not drop enough, role stability
fails without leakage, or any other non-smuggling T3-R gate fails while no
absorption token condition holds.

Interpretation: the proposed packet did not transmit a robust representation-changing
frame under this spec.

### Data-Reading Check

This adds every Supervisor #25 verdict token and makes Q-Loop B33's absorption
routes terminal: teaching dimension, representation prior, nuisance oracle,
library learning, Boolean trap, and T3-R signal.

### Alternative Interpretation

Absorption can be scientifically useful, but it is not a FrameSeed moonshot win.
Secondary ratios remain logged for diagnosis.

### Right Experiment Check

This is the right experiment for this iteration only if the confirm/kill/void
criteria make at least one hostile dismissal executable. The token path remains
locked to the hardened terminal verdicts; narrative appeal cannot override them.

### Attack
```text
You will call a mixed result a signal.
```

No: a mixed result is absorption, Boolean trap, void, or negative by precedence.

### Confirm / Kill / Void

Confirm if every run emits exactly one terminal token and logs secondary ratios.
Kill/reframe if multiple prior-art baselines absorb within the hard clock. Void
if no token can be assigned because protocol was violated.

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

"Teaching" is too vague. FRAMESEED-0 targets T3-R and must not claim T4 or T5.
T3-R is explicitly separated from teaching dimension, active learning, CEGIS,
RAG, nuisance-oracle shortcuts, representation priors, and library learning.

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

### T2.5: Prior-Art Teaching And Library Absorption

Operational boundary:

```text
The lesson is an optimal teaching set over H0, a representation already in L0, a
nuisance-ignoring functional shortcut, or an MDL/library macro learned from the
same task bundle.
```

Evidence: TD-H0, representation-prior certificate, nuisance-oracle suite, or
library-learning baseline reaches the corresponding token condition.

### T3-R: Representation-Changing Frame Transfer

Operational boundary:

```text
A counted packet transforms L0 into a post-packet learner by installing or
exposing a reusable operator, verifier, transformation, decomposition,
intervention generator, search metric, or macro that was not low-cost reachable
inside L0; target plus at least two sibling tasks pass; AFTD beats independent
teaching sets and library learning; all lower and nuisance-fair baselines fail or
need at least 4x cost.
```

Evidence: exactly the `FRAMESEED_T3R_SIGNAL` conditions.

Non-evidence: selecting a support mask from H0, picking an existing truth-table
hypothesis, teaching one concept only, beating reconstruction baselines while
function-only baselines match, or transmitting a full target program.

### T4: Self-Generating Frame Teacher

Operational boundary:

```text
The teacher discovers which frame-bearing packet to construct under query and
packet budgets across unfamiliar world families.
```

Additional requirements: teacher algorithm frozen before new families; teacher
chooses interventions/invariants/counterexamples/representation patches without
human packet design; teacher beats active query selection, synthesis, teaching
dimension, and library-learning baselines in choosing what evidence to buy.
FRAMESEED-0 does not claim T4.

### T5: Open-World Frame Formation

Operational boundary:

```text
The system forms useful frames in messy real domains without a synthetic role
taxonomy, hidden generator, or hand-authored verifier family.
```

Requires real or semi-real tasks, prior-art and neural/tool baselines,
human-labor accounting, public adversarial review, and a useful artifact or
repair loop. FRAMESEED-0 does not claim T5.

### Why T3-R

T0-T2 and T2.5 are the boring explanations. T4-T5 are too broad for the first
post-PCCP test. T3-R is the first level where the FrameSeed claim is measurable:

```text
Can a compact lesson change a bounded learner's reusable representation, not
just labels, examples, or a supplied program?
```

### Data-Reading Check

Batch 32 proposed T0-T5; Batch 33 I225/I226/I229 kills the vague frame-bearing rung and preserves
only T3-R with representation noncontainment and AFTD.

### Alternative Interpretation

T3-R may still overlap with machine teaching. FRAMESEED-0 does not deny overlap;
it tests whether the object survives the strongest machine-teaching, nuisance,
and library-learning absorbers.

### Right Experiment Check

This is the right experiment for this iteration only if the confirm/kill/void
criteria make at least one hostile dismissal executable. The token path remains
locked to the hardened terminal verdicts; narrative appeal cannot override them.

### Attack
```text
T3-R is a word game between curriculum, program synthesis, and library learning.
```

Operational answer: if active learning, CEGIS, teaching dimension, nuisance
oracle, representation prior, or library learning matches, T3-R is absorbed.

### Confirm / Kill / Void

Confirm if every result can be assigned by operational evidence and T3-R requires
all gates. Kill/reframe if T3-R cannot be separated from T1/T2/T2.5 or if
multiple absorptions hold. Void if the ladder changes after hidden results.

### NARRATIVE SECTION

Gossip story: FRAMESEED-0 is trying to climb above flashcards, program search,
and macro learning, but it is not claiming open-world teaching.

It survives only if T3-R is operationally separated from the lower rungs and
measured as amortized reusable frame transfer.

If boring: if the result lands on T0, T1, T2, or T2.5, report it that way.

---

## I233: Risk Register And Kill Conditions

### Steelman

The direction deserves a hard clock. If FrameSeed cannot produce a T3-R signal or
a clean absorption/void/negative/Boolean-trap token quickly, the narrative should
not inflate.

### Risk Register

| Risk | Failure mode | Required response |
|---|---|---|
| Teaching-dimension absorption | TD-H0 matches target plus siblings or AFTD | Emit teaching-dimension absorption. |
| Representation-prior absorption | L0 already reaches frame or contains target-isomorphic primitive | Emit representation-prior absorption. |
| Active-learning absorption | L1 matches under matched or <4x budget | Emit absorption; repeated absorption kills/reframes. |
| CEGIS absorption | L2 synthesizes same predictor or patch from same packet | Emit absorption; strongest synthesis kill risk. |
| RAG absorption | Retrieval/few-shot packet matches | Emit absorption; demote to retrieval curriculum. |
| Nuisance-oracle absorption | Oracle mask/function-only/invariant/CEGIS fairness suite matches | Emit nuisance-oracle absorption. |
| Library-learning absorption | MDL macro learner matches AFTD or HFA | Emit library-learning absorption. |
| Smuggled frame | Names, DSL, verifier, learner prior, patch, or metadata contains target | Void and rerun with new hidden seeds. |
| Linear packet growth | Packet carries nuisance surface data | Negative unless redesigned before hidden run. |
| AFTD failure | Frame packet helps one task but not siblings | Negative or teaching-dimension absorption. |
| Sibling clone | Siblings are target copies under renaming | Redesign; no signal. |
| Ablation no-op | Removing intervention/counterexample/invariant/patch does not hurt | Negative for T3-R. |
| Role instability | Result depends on surface order or names | Void if leakage, negative if no leakage. |
| Synthetic triviality | Raw decision trees, L0, or TD-H0 exceed threshold | Absorb or redesign; no signal. |
| Hidden-family overfit | Seen tuning fails hidden kernels/maps/orientations/siblings | Negative. |
| Full-program shortcut | Compact program is the whole answer | Absorbed by CEGIS/RAG/library or void if smuggled. |
| Human packet authorship | Manual edits after hidden inspection | Void. |
| Metric gaming | Accuracy passes but budget/AFTD/ablation fails | Negative. |
| Boolean trap | Positive Boolean result not followed by typed domain spec/run | Emit Boolean trap. |
| Narrative overclaim | T3-R reported as cheap general intelligence | Documentation failure; supervisor escalation. |
| Prior-art collapse | Teaching dimension, CEGIS, or library learning fully explains result | Treat as absorption unless measured AFTD gap remains. |

### Multi-Absorption Rule

Let:

```text
A = active learning absorbs
C = CEGIS absorbs
R = RAG absorbs
T = teaching dimension absorbs
P = representation prior absorbs
N = nuisance oracle absorbs
L = library learning absorbs
```

If at least two of `{A, C, R, T, P, N, L}` occur in FRAMESEED-0 or its first
honest variant:

```text
kill FrameSeed as the main moonshot direction or radically reframe it
```

Allowed reframes: make active query selection the honest substrate; make
CEGIS/proof compilation the honest direction; make MDL library learning the
honest direction; move to self-discovered transformation grammars only if teacher
choice itself beats lower baselines. Disallowed rescues: larger L3, hidden prose,
removing baselines, weakening sibling tasks, or changing thresholds after hidden
results.

### Boolean Escape Clause

FRAMESEED-0 is Boolean and therefore cannot carry the public claim. The escape
clause is binding only after a positive Boolean result:

```text
W28: FRAMESEED-SHEETS-0 spec must exist.
W29: FRAMESEED-SHEETS-0 must run or emit FRAMESEED_T3_BOOLEAN_TRAP.
```

FRAMESEED-SHEETS-0 minimum requirements:

```text
- non-Boolean typed objects: strings, numbers, lists, records, IDs, dates, units,
  rows, columns, or actions;
- practical frame: stable ID over display name, unit normalization, key-based
  join, row-order non-identity, constraint validation, or equivalent;
- hidden transfer across renamed columns, nuisance columns, shuffled rows, new
  units, missing aliases, and adversarial display names;
- the same absorption baselines: active learning, CEGIS, RAG, teaching dimension,
  nuisance oracle, representation-prior audit, and library learning;
- useful local-automation narrative for cheap systems.
```

No positive FRAMESEED-0 Boolean result may be used as more than a filter unless
this clause is satisfied.

### Hard Clock

Starting from W-Loop B26 hardening:

| Batch | Required state |
|---|---|
| W25 | Original precommit spec written and audited. |
| W26 | Spec hardened with Q-Loop B33 corrections; no implementation. |
| Q34 | Hardened spec review and adversarial implementation criteria. |
| W27 | First implementation may begin; run emits token or blocking bug ledger. |
| W28 | FRAMESEED-SHEETS-0 spec if Boolean signal exists. |
| W29 | FRAMESEED-SHEETS-0 run or `FRAMESEED_T3_BOOLEAN_TRAP`. |
| W30 | Supervisor assessment: T3-R signal, absorption, void with redesign, negative, Boolean trap, or kill. |

No indefinite "almost there" state is allowed.

### Claim Ceiling

Even on `FRAMESEED_T3R_SIGNAL`, the maximum claim is:

```text
controlled evidence for amortized frame-teaching separation
```

Any stronger claim is barred until a typed non-Boolean domain passes with the
same absorption controls.

### Data-Reading Check

Supervisor #25 updates the hard clock: B26 is spec hardening, implementation
starts no earlier than W27, and FRAMESEED-SHEETS-0 is due by W28/W29 after a
Boolean signal.

### Alternative Interpretation

Absorption may still be scientifically useful, as PCCP-H was. It is not a
FrameSeed moonshot win.

### Right Experiment Check

This is the right experiment for this iteration only if the confirm/kill/void
criteria make at least one hostile dismissal executable. The token path remains
locked to the hardened terminal verdicts; narrative appeal cannot override them.

### Attack
```text
You will keep the direction alive by redefining the goal after every baseline
wins or by celebrating a Boolean toy.
```

The multi-absorption rule, Boolean escape clause, and claim ceiling prevent that.

### Confirm / Kill / Void

Confirm if every run emits a token, multiple absorption triggers kill/reframe,
FRAMESEED-SHEETS-0 follows any Boolean signal, and W30 cannot pass without
supervisor assessment. Kill if multiple baselines absorb, no token emerges by
W30, or signal requires relaxed thresholds. Void only for protocol failures
requiring new hidden seeds.

### NARRATIVE SECTION

Gossip story: FrameSeed gets a short, hard window to prove it is more than a
slogan before the boring baselines or the Boolean trap take the wheel.

It survives because the project is willing to kill the exciting phrase if
ordinary methods explain it.

If boring: if the hard clock ends in absorption or Boolean trap, the immune
system worked.

---

## I234: Integration Test

### Steelman

A different implementer should be able to build FRAMESEED-0 from this spec. This
walk-through checks that the components fit after the T3-R hardening.

### Hypothetical World

```text
m = 16
d = 20
K_target = xor truth table [0,1,1,0]
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

Sibling tasks share the intervention frame but differ in surface function:

```text
Sibling 1: K_s1 = and-like admitted table over the causal pair.
Sibling 2: K_s2 = implication-like admitted table over the causal pair.
```

They resample names, role order, orientations, nuisance values, and hidden query
sets.

### Full Vaccine Packet Seen By L3

Shown in prose here, but implemented as canonical FIR-0:

```text
header(d=20, L0_hash, H0_hash)

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

representation_patch(
  kind = intervention_generator,
  ast_or_schema = test candidate support by paired causal-vs-alias edits,
  declared_cost = serialized FIR-0 schema cost,
  admissibility_scope = bit slots and listed intervention grammar)

verifier_clause(
  finite_scope = listed masks and listed single-slot edits,
  required_relation = agree with examples and invariant)
```

Slot id cost here is:

```text
slot_bits = ceil(log2(20)) = 5
```

The complement set `[13,2]` costs two slot ids plus opcode/count overhead; it
does not list every nuisance slot. The representation patch must serialize every
operation it adds and may not contain the words causal, alias, nuisance, or the
hidden role map.

### What L3 Does

L3 decodes the packet, verifies representation noncontainment against L0, builds
constraints, applies the counted patch, and selects the minimum-cost consistent
post-patch candidate:

```text
truth_table_2(
  edited_value(obs, edits, slot13),
  edited_value(obs, edits, slot2),
  [0,1,1,0])
```

It rejects the alias candidate because of the counterexample and invariant. For
siblings, the post-packet learner can reuse the intervention generator or verifier
to identify the causal support with reduced residual teaching cost, then learn a
different kernel for each sibling.

### What Each Baseline Sees

TD-H0 receives L0, H0, the same packet channel, target, siblings, and budgets. If
an optimal teaching set over H0 reaches target plus siblings or matches AFTD,
emit teaching-dimension absorption.

L0 receives the same packet, extracts labeled query records, and uses nearest
neighbor. It cannot execute universal invariants or representation patches unless
they enumerate finite labeled records within budget. If it reaches 0.95, no T3-R
signal.

L1 receives the same packet and query budget, searches all 1-slot and 2-slot
truth-table predictors, and may use expressible packet constraints. If it finds
slots 13 and 2 under matched or <4x budget, emit active-learning absorption.

L2 receives the same packet, same constraints, and the FIR-0-plus-enumeration
DSL. If it synthesizes the same target or sibling solution under matched or <4x
budget, emit CEGIS absorption. This is a serious expected risk, not a footnote.

Nuisance-oracle baselines may ignore nuisance and optimize functional accuracy.
If oracle causal mask, function-only MDL, invariant active learning, or
nuisance-ignoring CEGIS matches under matched or <4x budget, emit nuisance-oracle
absorption.

RAG receives every packet entry serialized and indexed. If RAG-NN, RAG-CLAUSE,
RAG-PROG, or RAG-PATCH reaches 0.95 under matched or <4x budget, emit RAG
absorption unless a higher-precedence absorber also applies.

The library-learning baseline receives the target and sibling task bundle and
may invent a macro. If its total description length or AFTD matches under matched
or <4x cost, emit library-learning absorption.

### What Gets Measured

For every world, sibling, role permutation, and `m`:

```text
HFA per system
packet bits
frame-patch bits
oracle query bits
final program bits
library/macro bits
TD_H0
AFTD and AFTD_all_in
baseline budget ratios
ablation HFA
role-permutation HFA variance
representation-noncontainment result
smuggling audit result
terminal token
```

Across `m`:

```text
P(4), P(16), P(64), P(256)
P_frame(4), P_frame(16), P_frame(64), P_frame(256)
alpha_hat
baseline bits_to_0.95 and queries_to_0.95
AFTD curve
ablation drops
```

### Internal Consistency Check

The run has causal, spurious, and nuisance bits; observational equivalence;
divergent interventions; sibling tasks; hidden nuisance/composed edits; generic
L0/L3 operation; representation-noncontainment; equal-information baselines;
function-only nuisance baselines; teaching-dimension and library-learning
absorbers; countable packet length; sublinear possible packet encoding; AFTD;
Boolean escape; and explicit token paths.

### Data-Reading Check

This keeps the Batch 32 minimal frame-vaccine world but incorporates every
Supervisor #25 correction from Q-Loop B33 before any implementation.

### Alternative Interpretation

The packet mentions the causal support slots. A hostile reviewer may call that
the answer. FRAMESEED-0 accepts the objection and asks whether that counted
support/intervention frame changes representation and amortizes across siblings
better than optimal teaching, active search, synthesis, retrieval, nuisance
oracle, and library learning. If not, absorb it.

### Right Experiment Check

This is the right experiment for this iteration only if the confirm/kill/void
criteria make at least one hostile dismissal executable. The token path remains
locked to the hardened terminal verdicts; narrative appeal cannot override them.

### Attack
```text
This example shows CEGIS, teaching dimension, or library learning can probably
find the same truth table or macro.
```

Correct. That is why those are primary terminal tokens and why the positive claim
is only controlled evidence for amortized frame-teaching separation.

### Confirm / Kill / Void

Confirm if an implementer can build generator, serializer, L0/L3, baselines,
metrics, audits, AFTD, sibling tasks, Boolean escape logic, and token logic from
this spec alone. Kill/reframe if every plausible path is teaching dimension,
CEGIS, nuisance oracle, or library-learning absorption. Void if implementation
needs undeclared task-specific primitives or packet semantics remain ambiguous.

### NARRATIVE SECTION

Gossip story: in the demo run, the vaccine tries to give the cheap learner a
reusable way to find the slots that still matter after surgery, then the boring
baselines get to prove whether that was special.

It survives only if that pointing becomes a counted transferable frame under
nuisance growth and sibling tasks, not just a support list that active learning,
teaching dimension, CEGIS, or library learning can find.

If boring: this integration test may be boring because a prior-art baseline may
absorb it. That would be an honest kill signal, not a failed spec.

---

## Final Precommit Checklist

- [ ] World has causal bits, nuisance bits, spurious bits, and observational
      ambiguity.
- [ ] Two observationally equivalent rules diverge under targeted intervention.
- [ ] Hidden families include held-out kernels, alias maps, orientations, role
      permutations, names, nuisance growth, and composed interventions.
- [ ] At least two sibling tasks require the same frame but different surface
      functions.
- [ ] L0 is defined as `(R0, H0, A0, B0)`.
- [ ] Representation-noncontainment certificate checks bounded non-reachability,
      no low-cost named primitive, and no equivalent teaching set.
- [ ] L3 has no named target primitives.
- [ ] L3 ingests examples, counterexamples, invariant clauses, transformation
      hints, verifier clauses, representation patches, and compact programs
      through FIR-0.
- [ ] Packet format counts examples, interventions, counterexamples, invariant
      hints, verifier clauses, representation patches, and optional programs.
- [ ] Packet growth criterion is sublinear in `m`.
- [ ] AFTD and AFTD_all_in are measured.
- [ ] TD-H0 optimal teaching-set baseline over H0 is specified.
- [ ] L0, L1, L2, RAG, nuisance-oracle baselines, and library-learning baseline
      are exactly specified.
- [ ] Nuisance-oracle suite includes oracle causal mask, function-only MDL,
      invariant active learner, nuisance-ignoring CEGIS, and randomized nuisance
      relabeling.
- [ ] All baselines optimize functional accuracy, not reconstruction.
- [ ] Each baseline gets equal information, matched budgets, and the same target
      plus sibling task bundle.
- [ ] Hidden-family accuracy threshold is at least 0.95 per family, per `m`, and
      per counted target/sibling task.
- [ ] Nuisance curve uses `m = {4, 16, 64, 256}`.
- [ ] Ablations remove intervention, counterexample, invariant, verifier, and
      representation-patch components and require at least 20 percentage point
      drops where specified.
- [ ] Role permutation stability test is required.
- [ ] Smuggling audit runs before any signal.
- [ ] Verdict tokens have exact conditions and precedence.
- [ ] Token list includes `FRAMESEED_T3R_SIGNAL`,
      `FRAMESEED_T3_ABSORBED_BY_TEACHING_DIMENSION`,
      `FRAMESEED_T3_ABSORBED_BY_REPRESENTATION_PRIOR`,
      `FRAMESEED_T3_ABSORBED_BY_NUISANCE_ORACLE`,
      `FRAMESEED_T3_ABSORBED_BY_LIBRARY_LEARNING`,
      `FRAMESEED_T3_BOOLEAN_TRAP`, active-learning absorption, CEGIS absorption,
      RAG absorption, smuggling void, and negative.
- [ ] Teaching ladder makes T3-R operational and treats the vague frame-bearing rung as
      absorbed.
- [ ] Multi-absorption rule is binding.
- [ ] Boolean escape clause requires FRAMESEED-SHEETS-0 spec by W28 and run by
      W29 after any Boolean signal.
- [ ] Claim ceiling is binding: even `FRAMESEED_T3R_SIGNAL` means only
      controlled evidence for amortized frame-teaching separation.
- [ ] Hard clock is binding from W26/W27 as specified.
- [ ] Integration test is internally consistent.

## Final NARRATIVE SECTION

Gossip-magazine one-sentence story:

```text
What if a tiny AI does not need the whole internet, just one compact lesson that
installs a reusable way to tell which parts of the world still matter when you
poke them?
```

Does that survive "isn't that obvious?":

```text
Only conditionally. Good teaching is obvious; a counted representation-changing
packet that passes noncontainment, beats optimal teaching sets and library
learning, ignores nuisance fairly, and amortizes across sibling tasks is not
obvious.
```

Does that survive "so what?":

```text
Yes, if it works beyond the Boolean filter. It would point toward shareable,
inspectable lessons that let cheap local systems acquire reusable frames without
renting frontier-scale training or inference.
```

If the honest narrative is boring, say so:

```text
The honest current narrative is still boring until evidence exists. This spec
does not prove FrameSeed. It defines a fair trap: either a T3-R amortized
frame-teaching gap survives, or the direction is absorbed by the ordinary methods
it resembles, trapped in Boolean toys, voided, or killed.
```
