# Q-Loop Batch 34: Implementation Review Preparation

**Date:** 2026-07-07
**Role:** Question Loop worker
**Iterations:** I232-I238
**Status:** pre-implementation adversarial review; assumes W-Loop B26 incorporates all eight B33 corrections; no implementation; no web runs.

---

## Grounding

Read in required order: `research/frameseed_0_precommit_spec.md`, `research/question_loop_batch33.md`, `research/dual_loop_supervisor_checkin_25.md`, `research/VISION.md`, and `../CLAUDE.md`.

Binding invariants:

1. Swing for the home run. Paradigm-shifting or nothing. Narrative is first-class.
2. The loop only stops on a won-over adversary.

Assumption: the hardened spec includes T3-R, teaching-dimension and library-learning baselines, nuisance-oracle fairness, AFTD, sibling transfer, representation-noncontainment, Boolean escape, expanded verdict tokens, and the claim ceiling. Therefore this batch attacks what remains dangerous after those corrections.

## Summary Verdict

```text
THE HARDENED SPEC IS NECESSARY BUT NOT SUFFICIENT.
THE NEXT FATAL SURFACE IS IMPLEMENTATION NONINTERFERENCE.
```

The remaining danger is not the packet grammar alone. A formally clean packet can still be produced by an answer-aware constructor, an asymmetric baseline adapter, a seed schedule that leaks roles, a tie-breaker that selects the intended support, or a hidden-eval workflow that quietly tunes the result.

One extra hardening before implementation:

```text
Constructor Noninterference Contract:
  Freeze the packet constructor, seed manifest, audit harness, baseline
  adapters, scorer, and review prompt before hidden results are opened.
  Every packet entry must have a provenance log showing which allowed public
  transcript fact caused it to be emitted.
```

Recommendation:

```text
CONDITIONAL GO, BUT ONLY IF W27 STARTS WITH THE AUDITABLE HARNESS.
NO-GO IF W27 STARTS WITH A PERFORMANCE RUN.
```

---

## I232: Implementation Confound Catalog

### Single Most Dangerous Question

Can the code make the packet look like it transmitted a frame when the real teacher was the constructor, seed schedule, or baseline adapter?

### Attack

After B33 hardening, the most dangerous confounds move below the spec line.

Implementation choices that can create false positives:

| Surface | False-positive route | Why dangerous |
|---|---|---|
| Data generation | `pi`, `rho`, `beta`, kernel, role order, and names share PRNG correlations. | Names and slot indices can leak roles without banned strings. |
| Invalid-world resampling | Worlds without decisive interventions are resampled toward easy supports. | The hidden distribution silently becomes easier than declared. |
| Query construction | Hidden eval overweights query types used by the packet constructor. | HFA measures familiarity, not transfer. |
| Intervention semantics | Alias refresh, orientation, or composed edits are implemented inconsistently. | All systems can share the same bug and still agree. |
| Learner initialization | Candidate order, caches, or canonical tie-breaks favor low slots or first-mentioned slots. | Representation-noncontainment passes while selection bias remains. |
| Packet construction | Constructor reads latent role maps, hidden sibling labels, or human-selected examples. | The packet is counted, but answer selection is uncharged. |
| Packet order | First example, first counterexample, or first slot in a set names the causal support. | Order becomes a covert role channel. |
| Baseline translation | L3 executes constraints while baselines receive flattened records or inert text. | Equal information is false at the API level. |
| Budget accounting | Failed queries, verifier expansion, final programs, or library macros are omitted. | Baselines can be made to exceed budget while L3 gets free search. |
| Timeouts | CEGIS/library learners are capped by implementation limits rather than theory. | Absorption resistance becomes an engineering artifact. |
| Logs and caches | Hidden labels or role maps leak through debug files, pickles, paths, or object reprs. | Static source audits miss runtime artifacts. |

The worst version:

```text
The teacher knows the latent roles, emits a counted formal packet, and the
report credits the packet rather than the answer-aware constructor.
```

### Pre-Implementation Smuggling Audit Checklist

Before any hidden performance run:

1. Freeze a manifest: commit, configs, public seed, hidden seed rule, constructor, serializer, learner versions, baseline versions, scorer, timeouts, and oracle API.
2. Split RNG streams by purpose: world structure, names, orientations, hidden queries, packet construction, learner tie-breaks, baseline tie-breaks, and ablations.
3. Run generator MI tests over at least 10,000 dry-run worlds: role versus slot index, name statistics, packet order, kernel id, and orientation.
4. Require constructor provenance for every packet entry: allowed observation, label, schema field, or prior entry that caused emission.
5. Add constructor-blind mode: packet builder sees only the allowed public transcript, not latent role maps, hidden labels, or hidden family ids.
6. Shuffle packet order, set literal order, examples, counterexamples, invariants, and verifier clauses after construction. Verdict must be invariant unless order is explicitly counted.
7. Run sabotage controls: swap causal and alias roles before serialization; the audit must detect the mismatch.
8. Independently recompute packet bits, query bits, final-program bits, macro bits, failed-query bits, and verifier-expansion bits from raw logs.
9. Verify every baseline receives a lossless executable equivalent of every field L3 can execute.
10. Run randomized labels, randomized roles, randomized names, reversed packet order, permuted slot ids, and shuffled hidden query order.
11. Run golden negative controls with no causal/spurious distinction and golden positive controls with an explicit oracle causal mask.
12. Treat the first hidden result as one-way: any constructor, scorer, timeout, or baseline-adapter change after opening hidden results requires new hidden seeds.

### Feed Into Work Loop

Add a mandatory section: `Implementation Noninterference And Constructor Provenance`. It must gate signal, not merely diagnose it after the fact.

### Verdict + Kill Records

```text
PRIMARY IMPLEMENTATION RISK: CONSTRUCTOR SMUGGLING.
```

Kill records:

```text
KR-I232-1: If the constructor can read latent role maps, hidden labels, or
hidden-family ids through an uncharged channel, void.

KR-I232-2: If role, name, seed, or packet-order MI is detectable above the
precommitted threshold, void.

KR-I232-3: If a baseline adapter loses executable packet information used by
L3, void.

KR-I232-4: If timeout or cache asymmetry changes the token, discard and restart
with fresh hidden seeds.
```

### NARRATIVE ATTACK

**Strongest rigging dismissal:** The answer was not smuggled through packet syntax. It was smuggled through the constructor and laundered into a clean packet.

**Strongest triviality dismissal:** A teacher with the answer can write a short lesson. That is not a theory of intelligence.

**What the result needs to be:** a frozen, provenance-audited constructor plus executable parity for baselines, so the adversary cannot point to an uncharged decision that selected the frame.
---

## I233: Adversarial Review Criteria for FRAMESEED-0 Results

### Single Most Dangerous Question

What would a fresh hostile reviewer attack first after a positive FRAMESEED result?

### Attack

The reviewer will look for the cheapest demolition path:

1. The packet names the causal support, so the frame is an answer key.
2. The learner already contains the intervention/invariance language, so the packet selected a prior hypothesis.
3. The constructor saw latent roles or hidden labels, so the experiment measured answer-aware teaching.
4. Baselines received lossy translations or insufficient optimization.
5. Teaching dimension, CEGIS, nuisance-oracle, representation-prior, or library learning should have absorbed but was underpowered.
6. Hidden split information leaked through logs, failed runs, notebooks, timeouts, or cache files.
7. AFTD is a posthoc ratio, not a precommitted metric.
8. Role-permutation stability hides selection bias because every permutation is generated by the same biased constructor.
9. The effect is Boolean enumeration with no typed-domain escape.
10. The positive token depends on one seed family, threshold edge, or intervention-semantics implementation.

Hostile review question:

```text
If I reimplement only the generator, packet serializer, scorer, and strongest
CEGIS/library baseline from the spec, do I get the same token? If not, why
should I believe the reported signal?
```

### Exact Adversarial Review Prompt For Milestone Gate

```text
You are a hostile fresh-eyes reviewer. Decide whether FRAMESEED-0 produced
representation-changing frame transfer or whether the result is absorbed, void,
or negative.

Assume the authors are competent but self-deceived. Take nothing as given
except the two project invariants: swing for a paradigm shift, and stop only
when a won-over adversary cannot knock the result down.

Read, in order:
1. Final hardened FRAMESEED-0 spec.
2. Implementation manifest and constructor provenance logs.
3. Generator, serializer, scorer, L3, and every baseline adapter.
4. Raw run logs, seed manifests, hidden split hashes, ablations, role
   permutations, randomized-label controls, budget ledgers, and cache policy.
5. Result report and claimed terminal token.

Return:
1. Strongest absorption explanation: teaching dimension, active learning,
   CEGIS, RAG, nuisance oracle, representation prior, or library learning.
2. Strongest smuggling explanation: constructor access, seed leakage, packet
   order, baseline translation, hidden tuning, runtime caches, or scorer bugs.
3. Whether the AFTD gap was precommitted, reproducible, and meaningful after
   sibling-task transfer.
4. Whether any Boolean-only result is allowed to support a public claim before
   FRAMESEED-SHEETS-0.
5. Your token: SIGNAL, ABSORBED, VOID, NEGATIVE, or REVIEWER CAN NOT ASSIGN
   TOKEN.
6. Minimum change that would make you believe the result, or a kill
   recommendation if no small change can do so.

Do not reward novelty language. Reward only auditability, fair baselines,
precommitted metrics, independent reproducibility, and survival against the
strongest boring explanation.
```

### What Wins Over The Reviewer

A reviewer can say "I can't knock this down" only if:

1. A clean checkout with the manifest reproduces the token.
2. Constructor provenance proves no hidden labels, hidden role maps, or uncharged human choices entered the packet.
3. Every baseline receives a lossless executable equivalent of L3's packet fields.
4. The strongest baselines are credible enough that at least one nearly wins on some slice.
5. L3 passes every hidden family and `m`; ablations drop hard; randomized labels fail; role permutations preserve both HFA and token.
6. AFTD is computed from raw logs and improves sibling tasks, not only the target task.
7. Failure cases and near-absorptions are reported.
8. The claim ceiling is honored: controlled evidence for amortized frame-teaching separation, not cheap general intelligence.

### What Makes It Obviously Rigged

Immediate rejection if:

1. Source/config exposes role metadata to learner, constructor, or baseline adapters outside allowed generator/scorer boundaries.
2. Constructor is edited after hidden inspection.
3. Packet order, file names, seed names, or slot indices predict latent roles.
4. L3 executes fields baselines receive as inert text.
5. CEGIS/library/teaching baselines use weaker DSLs than L3 or arbitrary timeouts.
6. Signal disappears under packet shuffling, role permutation, or sibling transfer.
7. Boolean-only results are marketed as manifesto-level evidence.

### Verdict + Kill Records

```text
THE RESULT GATE IS NOT "DID IT PASS?" BUT "CAN A HOSTILE REVIEWER REASSIGN THE
TOKEN FROM RAW ARTIFACTS?"
```

Kill records:

```text
KR-I233-1: If the reviewer cannot reconstruct the token from raw logs, the
result is not publishable.

KR-I233-2: If the strongest baseline was not close enough to be credible, rerun
with a stronger baseline before interpreting signal.

KR-I233-3: If the report relies on narrative not backed by artifact-level
evidence, demote to internal note.
```

### NARRATIVE ATTACK

**Strongest marketing dismissal:** The paper asks me to admire the word "frame" while ordinary teaching, synthesis, or constructor access did the work.

**Strongest reproducibility dismissal:** I cannot assign the token from artifacts without trusting the authors.

**What the result needs to be:** a reviewer should be able to hate the claim, rerun the manifest, inspect provenance, strengthen a baseline, and still fail to move the token away from signal.

---

## I234: Single Most Likely Failure Mode

### Single Most Dangerous Question

Given the hardened spec, what is the most likely honest failure?

### Attack

The most likely failure is:

```text
FRAMESEED_T3_ABSORBED_BY_CEGIS or FRAMESEED_T3_ABSORBED_BY_LIBRARY_LEARNING.
```

The minimal world is two causal bits, two alias bits, finite truth tables, finite interventions, and formal verifier clauses. If L3 can use the packet to identify a support, reject the alias support, and choose a two-slot truth-table program, then a strong CEGIS or MDL library learner with the same executable facts should usually synthesize the same object under less than 4x budget.

B33 hardening makes this more likely, not less. Teaching-dimension, AFTD, and sibling-task gates remove easy false positives. Once baselines are fair, the remaining positive gap must be real. In a tiny Boolean world, real gaps are rare.

### What The Failure Teaches

If CEGIS or library learning absorbs, the lesson is:

```text
FRAMESEED-0 formalizes a strong answer-aware machine-teaching/program-synthesis
setting, but has not found a new representation-changing object.
```

Specific lessons:

1. FIR-0 plus intervention semantics may already be the decisive representation.
2. The frame may live in the human-chosen DSL, not in the packet.
3. AFTD may collapse to ordinary MDL: pay once for a macro, reuse it on siblings.
4. Boolean nuisance growth may be too easy for feature-subset search and synthesis.
5. The hard problem may be frame discovery or typed semantics, not frame transmission.

### Is That Failure Publishable?

Yes, but not as victory.

Publishable scoped claim:

```text
An absorption-first protocol for compact teaching claims shows that a formal
frame-packet result collapses into CEGIS/library learning under fair executable
baselines.
```

That is useful because it prevents a false manifesto claim, preserves an adversarial methodology, and points to the next harder target: self-discovered transformation grammars in typed domains.

### Verdict + Kill Records

```text
MOST LIKELY TOKEN: CEGIS OR LIBRARY ABSORPTION.
```

Kill records:

```text
KR-I234-1: If CEGIS or library learning absorbs, do not rerun with weaker DSLs
or tighter timeouts.

KR-I234-2: If absorption repeats in the first honest variant, trigger the kill
or radical-reframe rule.

KR-I234-3: If the team cannot explain the absorption without claiming signal,
stop and write the negative result first.
```

### NARRATIVE ATTACK

**Strongest expected-result dismissal:** A finite symbolic packet in a finite symbolic world was absorbed by finite symbolic synthesis. Nothing surprising happened.

**Strongest useful-negative defense:** The experiment honestly found the boundary: compact formal lessons are not yet a new intelligence primitive when synthesis gets the same information.

**What the result needs to be:** survive synthesis/library absorption, or pivot without ego to where frames come from.
---

## I235: Learner Prior Risk Deep Dive

### Single Most Dangerous Question

Even with representation-noncontainment, how might the learner's prior language smuggle the answer?

### Attack

Banned names are not enough. The dangerous prior is structural.

FIR-0 can smuggle the frame through affordances that are generic in name but answer-shaped in this world:

| Prior affordance | Risk |
|---|---|
| `edited_value(obs, edits, slot)` | Intervention semantics are native before the packet. |
| `truth_table_2(bit, bit, table4)` | The target is pre-shaped as a two-input Boolean function. |
| `set_complement` | "Ignore all other slots" is exactly nuisance invariance and is cheap. |
| `forall_slot` / `exists_slot` | Universal invariance language exists before teaching. |
| Minimum-cost selection | The intended two-slot explanation may be uniquely cheap. |
| Canonical tie-breaks | Role-blind serialization can still be distribution-biased. |
| Verifier clauses | Public clauses can be target-equivalent while looking generic. |
| Optional compact programs | Teaching can collapse into program delivery. |

Representation-noncontainment must separate three claims:

```text
expressibility: Can L0 express the target at any cost?
reachability: Can L0 find it under budget before packet?
prior fit: Does L0 make the target family uniquely cheap before packet?
```

A packet that tells a built-in two-slot intervention DSL where to look has selected a prior hypothesis. It has not necessarily transmitted a frame.

### Weakest Learner That Can Still Demonstrate Frame Transfer

The weakest credible learner is a role-blind packet interpreter, not a strong enumerator.

```text
L_min before packet:
  - stores typed observations and labels;
  - compares values for equality/inequality;
  - executes only packet-declared transforms and verifiers;
  - cannot enumerate arbitrary slot subsets;
  - has no built-in truth_table_2 over all slot pairs;
  - has no set_complement or universal slot quantifier;
  - has no active query access;
  - chooses among packet-declared candidates by public consistency checks.

Allowed packet effect:
  - install a bounded transform, verifier, macro, or intervention test;
  - bind it to public observations through charged slot ids;
  - reuse it on sibling tasks.
```

The cleanest design separates frame cost from binding cost:

```text
Frame packet F:
  teaches reusable operation or verifier and may not mention target support.

Binding packet B_i:
  gives task-specific public bindings and is charged per task.

Transmission claim:
  cost(F + B_1 + ... + B_k) << cost(TD_1 + ... + TD_k)
  while F alone solves no target.
```

If the frame packet names the causal slots, it is partly a binding packet. That may be allowed only if AFTD separately charges reusable frame cost and per-task binding cost.

### Proving Transmitted, Not Selected

Required proof pattern:

1. Pre-packet nonreachability: L_min fails hidden and sibling tasks under public data, optimal teaching, active learning, CEGIS, and library budgets.
2. Frame/binding separation: reusable frame packet contains no target slot ids, hidden labels, hidden family ids, or kernel identity.
3. Sibling reuse: the same frame packet lowers binding cost for at least three sibling tasks with different supports, kernels, schemas, or units.
4. Erasure test: remove frame but keep examples/bindings, and success drops. Keep frame but swap bindings, and behavior follows the binding.
5. Prior ablation: move suspicious affordances such as `truth_table_2` or `set_complement` from prior into the packet. If success disappears unless prior owns them, report representation-prior absorption.
6. Lower bound: show independent optimal teaching or synthesis cost without the frame is larger by the precommitted factor.
7. Counterfactual packet: give the same examples without executable verifier/transform semantics. If it works, the result is selection or teaching dimension.

### Verdict + Kill Records

```text
LEARNER PRIOR RISK REMAINS LIVE AFTER B33.
```

Kill records:

```text
KR-I235-1: If pre-packet L3 reaches target because FIR-0 makes the right frame
uniquely cheap, emit representation-prior absorption.

KR-I235-2: If success depends on `truth_table_2`, `set_complement`, or universal
slot quantification living in the prior, the packet did not transmit the frame.

KR-I235-3: If reusable frame cost and task-specific binding cost are not
separated, AFTD is uninterpretable.
```

### NARRATIVE ATTACK

**Strongest preinstallation dismissal:** The learner already knew interventions, two-slot kernels, and nuisance complements. The packet only pointed.

**Strongest DSL-trick dismissal:** You called the answer a generic representation language and banned only the embarrassing names.

**What the result needs to be:** the frame object must be a paid, reusable operational addition, not a cheap built-in affordance waiting for slot ids.

---

## I236: FRAMESEED-SHEETS-0 Pre-Design

### Single Most Dangerous Question

What typed domain would make FrameSeed look like useful intelligence rather than Boolean enumeration?

### Attack

The second domain must not be bits with prettier names. It needs typed objects where practical failures look like real automation mistakes: joining by row order, trusting display names, adding inches to centimeters, accepting invalid records, or acting before checking constraints.

Candidate domain:

```text
FRAMESEED-SHEETS-0:
  Small typed table worlds with records, columns, units, IDs, dates, strings,
  numeric fields, row order, missing values, aliases, and validation rules.
```

Core objects:

```text
Table, Row, Column, Cell, StableID, DisplayName, UnitValue(value, unit),
Date, ForeignKey, Constraint, Action
```

Hidden variation:

1. Column names randomized, adversarial, or misleading.
2. Row order shuffled.
3. Display names duplicated or changed.
4. Stable IDs preserved but formatted differently.
5. Units vary across metric, imperial, currency, time, and rates.
6. Nuisance columns added.
7. Missing aliases and duplicate records introduced.
8. Constraints differ across schemas but share a frame.
9. Join cardinality varies: one-to-one, one-to-many, missing-key, duplicate-key.
10. Actions require precondition checks before execution.

### Candidate Frames

| Frame | Example task | Hidden transfer |
|---|---|---|
| Entity matching by stable ID | Merge updates using `customer_id`, not display name. | Renamed columns, duplicate names, shuffled rows, formatted IDs. |
| Unit normalization | Aggregate inventory after converting units. | New unit systems, mixed units, nuisance numeric columns. |
| Join by key | Combine orders and shipments by foreign key, not row position. | New arities, reordered rows, missing keys, duplicate keys. |
| Constraint validation | Reject actions violating type, range, uniqueness, or referential constraints. | New schemas, new constraints, adversarial valid-looking rows. |

### Packet Shape

A typed frame packet should contain typed examples, counterexamples, transforms, verifiers, task-specific bindings, and optional macros:

```text
examples: before/after table snippets with typed cells and labels
counterexamples: display-name match fails, row-position join fails, raw-unit sum fails
transforms: normalize(unit_value), canonicalize_id, join_on_key, validate_constraint
verifiers: finite obligations over rows, keys, units, and constraints
bindings: schema-column to typed-role mappings, charged separately
macros: optional executable typed programs, charged and baseline-visible
```

Frame/binding separation is crucial. "Column C is the stable ID" is a binding. "Stable IDs survive display-name drift and row shuffle" is a frame.

### Fair Baselines In A Typed World

Baselines must not parse raw strings while L3 receives types.

Fairness rules:

1. Public typed schema is shared: type grammar, unit syntax, row/table grammar, operations, and labels.
2. Baselines get canonical parsers for IDs, numbers, dates, and units if L3 gets them.
3. Teaching-dimension baseline searches shortest typed teaching set.
4. Active learner can query typed counterexamples: shuffled rows, duplicate names, unit swaps, missing keys, invalid constraints.
5. CEGIS gets typed joins, filters, unit conversion, equality, grouping, validation, and bounded macros.
6. RAG gets serialized typed records and retrieves by schema, value, operation, and counterexample type.
7. Nuisance-oracle may receive true non-nuisance columns but not the correct frame or operation.
8. Library learner may invent typed macros and amortize them across siblings.
9. Scoring is functional task success, not table reconstruction.
10. Human type-system design is charged or explicitly outside the claim.

### Teaching Ladder For Typed Domains

```text
T0 typed rote: memorize snippets; fails under schema rename and row shuffle.
T1 typed active: query examples identifying a current schema rule inside an existing class.
T2 typed CEGIS: synthesize join, normalization, or validation programs.
T3-R typed frame: install reusable verifier/transform lowering sibling binding cost while lower baselines fail or pay >=4x.
T4 typed teacher: choose counterexamples that reveal stable ID, unit, key, or constraint structure without human packet design.
T5 practical frame formation: repair messy real user sheets with inspectable packets and cheap local inference.
```

Minimum viable SHEETS-0 should start with two frames:

```text
Frame A: join by stable key, not row position or display name.
Frame B: normalize units before comparison or aggregation.
```

They are practical, user-legible, strongly baselined, typed beyond Boolean bits, and composable.

### Verdict + Kill Records

```text
FRAMESEED-SHEETS-0 SHOULD BE THE FIRST BOOLEAN ESCAPE.
```

Kill records:

```text
KR-I236-1: If SHEETS-0 gives L3 typed parsers or unit semantics denied to
baselines, void.

KR-I236-2: If typed frames reduce to Boolean feature masks with renamed columns,
emit BOOLEAN_TRAP.

KR-I236-3: If the result cannot be explained as cheap local automation useful
to ordinary users, do not use it for manifesto narrative.
```

### NARRATIVE ATTACK

**Strongest still-a-toy dismissal:** You replaced `c0` with `customer_id`, but the task is still a hand-authored puzzle.

**Strongest data-cleaning dismissal:** Joins and unit normalization are solved engineering operations, not new intelligence.

**What the result needs to be:** a cheap learner acquiring a reusable typed frame from a tiny inspectable packet and applying it across renamed, shuffled, adversarial schemas while fair synthesis, retrieval, and teaching baselines cannot cheaply match the amortized transfer.
---

## I237: What Would Make FrameSeed Nobel-Track?

### Single Most Dangerous Question

If FrameSeed works, what turns it from controlled methodology into a paradigm shift?

### Attack

A Boolean T3-R signal is not Nobel-track. A typed replication is not enough either. FrameSeed becomes serious only if it reveals a general law:

```text
Some useful intelligence is better modeled as transmission and composition of
compact frames than as scale, memorization, or task-by-task training.
```

That needs theorem plus artifact.

### Path From Toy Evidence To Paradigm Shift

1. Controlled separation: a precommitted AFTD gap in FRAMESEED-0 under hostile baselines.
2. Typed replication: the same gap in SHEETS-0 with joins, units, IDs, and constraints.
3. Frame composition: independently taught frames compose without retraining, for example stable-ID matching plus unit normalization plus constraint validation.
4. Repairability: when the learner fails, a small corrective packet fixes the failure without breaking earlier frames.
5. Public frame library: inspectable frame packets ordinary users can audit, modify, and reuse on cheap hardware.
6. Real messy task: a local agent uses the library for spreadsheet, workflow, or tool-use tasks at far lower data and inference cost than a cloud LLM baseline.

### Theoretical Result That Would Elevate It

The theory must be stronger than "good hints help."

Candidate theorem:

```text
Frame Teaching Dimension Separation:
  There exists a typed task family E_k and bounded learner L0 such that any
  independent teaching sets, active queries, or synthesis patches inside L0's
  original representation require total cost Omega(k * g(n)) across k sibling
  tasks, while one frame packet F of cost O(g(n)) plus per-task bindings O(1)
  reaches threshold. No nuisance-oracle or library learner in the original
  representation matches the amortized cost under the same description rules.
```

The theorem must specify learner prior, packet language, task distribution, lower bound for original representation, upper bound after packet extension, noncontainment, and composition rules.

A stronger target:

```text
Frame Composition Theorem:
  Under stated interface conditions, frame packets compose subadditively while
  preserving verifier obligations and allowing local repair.
```

That would connect directly to improvability and cheap public infrastructure.

### Real-World Demonstration That Makes The Manifesto Headline True

The manifesto headline becomes credible only if:

```text
A cheap local system starts weak, receives a small public library of inspectable
frame packets, and solves useful messy tasks with low data, low inference cost,
and surgical repair after failures.
```

Minimum real demo:

1. Runs on commodity hardware.
2. Uses no frontier-cloud model at inference.
3. Solves practical tasks: spreadsheet cleanup, local data integration, form validation, document/table reconciliation, or safe tool-use workflows.
4. Reports cost per successful task against a cloud LLM/tool baseline.
5. Logs failures and fixes some with small corrective packets.
6. Lets a third party inspect and modify packets.
7. Shows packet additions do not require full retraining.

The headline is not "Boolean learner solves parity." It is:

```text
Useful local intelligence can be upgraded by sharing compact, inspectable frames
instead of retraining or renting scale.
```

### Verdict + Kill Records

```text
NOBEL-TRACK REQUIRES AFTD SEPARATION + FRAME COMPOSITION + PRACTICAL LOCAL UTILITY.
```

Kill records:

```text
KR-I237-1: If FrameSeed cannot produce a formal separation beyond teaching
dimension or library learning, do not call it theory.

KR-I237-2: If frames do not compose or repair failures locally, do not claim
improvability.

KR-I237-3: If the public artifact requires frontier-scale inference, it misses
the manifesto even if the benchmark passes.
```

### NARRATIVE ATTACK

**Strongest not-a-paradigm-shift dismissal:** You found a neat pedagogy for toy learners, not a new account of intelligence.

**Strongest not-useful dismissal:** Ordinary people do not need a Boolean theorem. They need cheap systems that handle messy data and can be repaired when wrong.

**What the result needs to be:** a theorem-backed and artifact-backed route where compact public frames make cheap systems measurably more capable, reusable, and repairable than scale-first systems on practical tasks.

---

## I238: Pre-Implementation Verdict

### Single Most Dangerous Question

Is the hardened spec ready for implementation?

### Attack

The hardened spec is close enough to implement only if implementation starts with audit infrastructure. It is not ready for a performance run.

```text
The remaining risk is not undefined T3. B33 fixed that.
The remaining risk is that code will silently define the real experiment through
constructor access, baseline adapters, seed handling, timeouts, caches, and
scorer behavior.
```

If those are casual, the repo can get a beautiful terminal token that means nothing.

### ONE Remaining Risk W-Loop Should Address

```text
Constructor noninterference.
```

Definition:

```text
The packet constructor, baseline adapters, scorer, and run harness must be
unable to use uncharged information or asymmetric execution semantics to
produce the claimed gap.
```

Required addition:

```text
Pre-Implementation Harness Gate:
  Before L3 hidden performance is interpreted, pass generator audit, serializer
  audit, constructor provenance audit, baseline parity audit, budget
  recomputation, role/name/seed MI audit, randomized-label control, packet-order
  shuffler, role-permutation control, and golden token controls.
```

This is the narrow fix that covers the broadest post-spec danger.

### Final Go / No-Go Recommendation

```text
CONDITIONAL GO.
```

Allowed first implementation scope:

1. Generator with audited RNG streams.
2. Canonical serializer and independent bit recomputation.
3. Packet constructor in provenance-logged blind mode.
4. Baseline adapter parity tests.
5. Scorer and terminal-token assignment on golden controls.
6. Smuggling audit suite.

Not allowed as first implementation scope:

1. Optimizing L3 performance.
2. Reporting hidden HFA.
3. Tuning packet templates against hidden failure.
4. Weakening CEGIS/library/teaching baselines for speed.
5. Writing result narrative before the audit harness emits expected tokens.

If W26 adds constructor noninterference and W27 begins with the harness, proceed. If W26 hardens only the conceptual spec and W27 jumps to learner performance, no-go.

### Final Kill Records

```text
KR-FINAL-B34-1: If constructor provenance is missing, no signal token is valid.

KR-FINAL-B34-2: If the first hidden run requires code changes to constructor,
baseline adapters, scorer, or timeout policy, discard and rotate hidden seeds.

KR-FINAL-B34-3: If CEGIS/library absorption occurs, accept it and redirect
toward typed frame discovery rather than weakening baselines.

KR-FINAL-B34-4: If a Boolean signal occurs without SHEETS-0 spec work, keep the
claim internal and bounded.
```

### NARRATIVE ATTACK

**Strongest fooled-yourself dismissal:** The spec was adversarial, but the code path quietly became the real experiment.

**Strongest go-slower dismissal:** A performance run before the audit harness is not bold. It is how a moonshot generates fake evidence.

**What the result needs to be:** the first implementation should make cheating hard before making success possible. Only then can a positive token mean the adversary has fewer places left to hide.

---

## Final Recommendation

Proceed only under this precommit:

```text
FRAMESEED-0 implementation begins with noninterference, provenance, parity, and
token-audit infrastructure. The first scientific artifact is not HFA. It is
proof that the harness emits the right tokens on controls and cannot smuggle the
frame through the constructor.
```

The home run is not a clean demo. The home run is a result a hostile reviewer cannot knock down.