# Q-Loop Batch 52: Attack the E3 Toy Signal

**Date:** 2026-07-08
**Role:** adversarial question loop
**Iterations:** I463-I469
**Status:** E3 toy signal downgraded to positive-control evidence; hostile supplied-geometry gate required

## Grounding

Read before writing: `research/VISION.md`, `research/STATUS.md`, `research/question_loop_batch51.md`, `research/work_loop_batch43.md`, `research/dual_loop_supervisor_checkin_41.md`, `code/e3_teacher_tomography.py`, and `experiments/e3_teacher_tomography_result_50seed.json`.

The fixed mission is unchanged: swing for the home run, and stop only when a hostile adversary is won over. The current evidence is a 50-seed friendly toy signal: E3 source-specific lesson packets score 0.8588 hidden accuracy while the best ordinary absorber is 0.5034. The terminal token in the result file is `E3_TOY_SIGNAL_SOURCE_SPECIFIC_COUNTERFACTUAL_LESSON`.

This batch attacks whether that signal is teacher tomography or a supplied geometry artifact.

External competitive check used for this batch:

- Query-by-committee active learning and disagreement acquisition: https://arxiv.org/abs/2211.10013
- Data programming and weak-supervision source modeling: https://arxiv.org/abs/1605.07723
- Snorkel weak supervision: https://arxiv.org/abs/1711.10160
- Dataset cartography and example-level training dynamics: https://arxiv.org/abs/2009.10795
- Reinforced multi-teacher selection for KD: https://arxiv.org/abs/2012.06048
- Adaptive multi-teacher KD with meta-learning: https://arxiv.org/abs/2306.06634
- Disagreement-modulated policy self-distillation: https://arxiv.org/abs/2607.02502
- Multi-view learning surveys: https://arxiv.org/abs/1304.5634 and https://arxiv.org/abs/1610.01206

## Executive Verdict

E3 is not killed by the toy. It is also not meaningfully validated by the toy.

Final token:

```text
Q_LOOP_B52_E3_TOY_SIGNAL_DOWNGRADED_TO_POSITIVE_CONTROL_HOSTILE_GEOMETRY_GATE_REQUIRED
```

What survived:

- Source identity matters in the friendly toy: shuffled measurements and shuffled identity collapse near chance.
- The student is teacher-free at inference.
- E3 beats the implemented CE-only, single-teacher, averaging, weighted-vote, active-hard-example, shuffled, and weak augmentation baselines by large margins.

What did not survive:

- The 35pp gap cannot be interpreted as a paradigm signal yet.
- The implementation gives E3 explicit sensor roles, hand-authored transformations, and a compositional packet rule.
- The exact domain tool diagnostic reaches 1.0 and is merely not admitted.
- The implemented augmentation baseline is too weak to answer the FrameSeed/WGD supplied-geometry objection.

The current toy is a positive control proving that the intended mechanism can work when the world is friendly. The next test must ask whether any residual E3 value remains after geometry parity.

## I463: Attack The Size Of The Gap

### Pre-committed tokens

```text
B52_I463_SURVIVE_AS_POSITIVE_CONTROL_IF_SHUFFLES_FAIL_AND_E3_WINS
B52_I463_KILL_AS_LABEL_ORACLE_ADVANTAGE_IF_PACKET_RULE_IS_GRANTED_TRUE_FUNCTION
B52_I463_VOID_IF_RESULT_SUMMARY_MISMATCHES_JSON
```

### Steelman

The result is not a small noisy bump. Across 50 seeds, E3 gets 0.8588 hidden accuracy while CE-only, best single teacher, teacher averaging, weighted vote, active hard-example mining, and both shuffled controls sit around chance. The packet-value forecast is positive. If the only thing happening were extra labels, teacher average, or hard-example selection, those baselines should have moved.

The strongest pro-E3 reading is: the useful signal is not teacher correctness; it is the source-specific pattern of complementary teacher measurements.

### Attack

The gap is suspiciously large because the toy is binary, closed-form, and hand-authored. `code/e3_teacher_tomography.py` names `semantic_z0` and `verifier_z1`, then uses those roles to construct packet labels. E3 is not merely reading disagreement; it is allowed to compose two privileged latent sensors.

The ordinary baselines are intentionally bad in this world. A single teacher sees one latent bit. Averaging a latent-bit teacher with another latent-bit teacher does not compute XOR. Active disagreement over teacher probabilities does not discover the XOR composition. Shuffled controls break the only good composition. Therefore the 35pp gap may say only: XOR composition beats non-XOR baselines on an XOR world.

That is a positive control, not tomography.

### New Hardest Objection

E3 may be the only method in the experiment that is handed the true sufficient factorization: semantic source gives z0, verifier source gives z1, compose them with XOR.

If so, the large gap is not impressive. It is exactly what should happen when one method receives the correct latent interface and every baseline receives misaligned labels.

### Reconstruct What Survived

The signal survives only as source-specific composition can matter when teachers expose complementary factors and the student needs the composed factor at inference.

It does not yet survive as teacher tomography discovers functional geometry.

### Iteration Verdict

Chosen token:

```text
B52_I463_SURVIVE_AS_POSITIVE_CONTROL_IF_SHUFFLES_FAIL_AND_E3_WINS
```

But the claim ceiling is strict: this is a positive control with label-oracle risk.

## I464: Attack The Supplied Geometry

### Pre-committed tokens

```text
B52_I464_SURVIVE_IF_E3_BEATS_GEOMETRY_PARITY_ABSORBERS
B52_I464_KILL_IF_NUISANCE_ORACLE_OR_ENHANCED_AUGMENTATION_MATCHES
B52_I464_VOID_IF_HIDDEN_CONSTRUCTOR_LEAKS_UNFAIRLY
```

### Steelman

The toy already includes several anti-triviality checks. Shuffling teacher measurements destroys the result. Shuffling teacher identity destroys the result. E3 is not using the exact hidden labels at inference. It trains a small student and removes teachers. The fact that teacher identity matters is exactly what B51 demanded.

### Attack

The FrameSeed/WGD pattern is not "signals disappear when anything is shuffled." The pattern is: signals disappear when boring baselines get the same geometry the proposed method was quietly granted.

The E3 toy has at least five granted geometry objects:

1. The teacher roster is explicitly role-labeled.
2. The useful teacher pair is hard-coded.
3. The composition family is hard-coded as the packet rule.
4. The transform variants are hand-authored.
5. The hidden split is a specific nuisance regime.

The implemented B10 is not the decisive augmentation absorber. It augments with the same transformed examples but uses teacher-average labels. A hostile baseline should instead get the same transformation generator and either labels propagated from calibration under the known transform law, a small rule search over teacher-output Boolean compositions, a nuisance oracle naming the invariance/counterfactual axes, an exact symbolic solver over the public input bits, or the same role map without teacher-specific packet machinery.

### New Hardest Objection

The right B15 is not shuffled identity. The right B15 is: same transformations, same role ontology, same candidate pool, no source-specific tomography.

If that baseline wins, E3 adds no value beyond authored geometry.

### Reconstruct What Survived

The next hostile toy must separate three quantities:

```text
value(authored_transform_geometry)
value(source_specific_teacher_measurements)
value(student_training_with_extra_pseudolabels)
```

E3 survives only in the residual:

```text
E3_residual = hidden_transfer(E3) - max(hidden_transfer(geometry_parity_absorbers))
```

### Iteration Verdict

Chosen token:

```text
B52_I464_SURVIVE_IF_E3_BEATS_GEOMETRY_PARITY_ABSORBERS
```

The token is conditional. The current result does not yet meet it.

## I465: Attack The Natural-Domain Path

### Pre-committed tokens

```text
B52_I465_SURVIVE_IF_CHEAP_NATURAL_TEACHERS_HAVE_COMPLEMENTARY_FAILURES
B52_I465_KILL_IF_NATURAL_TEACHERS_COLLAPSE_TO_CORRELATED_NOISE_OR_SNORKEL
B52_I465_VOID_IF_NATURAL_TEST_REINTRODUCES_RULE_BASED_SENSOR_TOY
```

### Steelman

E3 does not need literal rule-based teachers forever. Natural domains already have heterogeneous cheap signals: weak labeling functions, small models, retrievers, unit tests, static analyzers, OCR systems, parsers, heuristic extractors, and verifier-like tools. If these sources fail differently, their disagreement patterns might reveal portable lessons that a small student can keep after teacher removal.

A natural E3 artifact could be: when the lexical source and entity-linker disagree under negation or temporal shift, teach a relation-extraction student the counterfactual feature that must flip the candidate ranking.

### Attack

Natural teachers are not clean sensors. They are correlated models, shallow heuristics, and tools with their own ontologies. They do not naturally expose `z0` and `z1`. If they are cheap enough for the democratization mission, they may be too brittle to carry functional geometry. If they are strong enough to be useful, they may be proprietary, expensive, and non-democratic.

There is also an obvious absorber: weak supervision. Data programming and Snorkel already model noisy, conflicting labeling sources and denoise them into training labels. If E3's natural-domain version is "combine heuristic teachers better," it will be absorbed.

### New Hardest Objection

Cheap natural teacher ecologies may have no discoverable complementary structure. They may only expose correlated artifacts of the same pretraining distribution, the same benchmark labels, or the same retrieval surface.

Then teacher tomography becomes expensive agreement engineering over redundant errors.

### Reconstruct What Survived

The cheapest credible natural domain should not start with free-form natural language generation. It should start where hidden transformations are natural but checkable, teacher sources are cheap and independently authored, exact tools get first refusal, lesson packets are reusable beyond one student, and source-specific identity plausibly carries more information than labels.

Candidate low-cost domains:

1. Weak-supervision relation extraction with negation, temporal, and entity-role flips.
2. Code diagnostics where teachers are tests, type checkers, linters, small repair models, and static analyzers.
3. Unit/numeric reasoning where sources include a parser, unit checker, small model, and retrieval, with exact solvers admitted as absorbers.

### Iteration Verdict

Chosen token:

```text
B52_I465_SURVIVE_IF_CHEAP_NATURAL_TEACHERS_HAVE_COMPLEMENTARY_FAILURES
```

The natural path is plausible only if E3 beats Snorkel-style source denoising, active learning, and exact tools on hidden transformations.

## I466: Attack Novelty Against The Competitive Landscape

### Pre-committed tokens

```text
B52_I466_SURVIVE_IF_E3_NOVELTY_IS_SHAREABLE_COUNTERFACTUAL_LESSON_PACKETS
B52_I466_KILL_IF_E3_IS_ONLY_QBC_PLUS_MULTI_TEACHER_KD
B52_I466_VOID_IF_LITERATURE_CHECK_IS_TOO_SHALLOW
```

### Steelman

The exact phrase "functional teacher tomography" is not the important claim. The interesting claim is that teacher disagreement can be inverted into student-owned lessons that forecast hidden-transform value. That is more specific than active learning and more functional than ordinary KD.

### Attack

The adjacent literature is crowded:

- Query-by-committee already uses model disagreement to choose high-value measurements.
- Dataset cartography already treats model behavior as a diagnostic map of example difficulty, ambiguity, and possible label error.
- Data programming and Snorkel already model multiple noisy weak sources.
- Multi-teacher KD already weights, selects, or meta-weights teachers by instance and student capability.
- Multi-view learning already studies complementary views and shared latent structure.
- Disagreement-modulated self-distillation already uses teacher/student distribution discrepancy to decide when to adopt guidance.

Therefore the dismissive expert can say: you renamed committee disagreement, weak supervision, and dynamic teacher selection.

### New Hardest Objection

E3 has no novelty in using disagreement or selecting teachers. Its only possible novelty is discovering an explicit counterfactual lesson object that is reusable, editable, teacher-free, and predictive of hidden transformation transfer.

If the packet is just a pseudolabeled dataset, the novelty evaporates.

### Reconstruct What Survived

The competitive claim must be written negatively first:

```text
E3 is not query-by-committee because the endpoint is not query selection.
E3 is not Snorkel because the endpoint is not denoised labels.
E3 is not multi-teacher KD because the endpoint is not a weighted teacher distribution.
E3 is not dataset cartography because the endpoint is not example diagnosis.
E3 is not multi-view learning unless it exports a functional lesson object.
```

Then the positive claim:

```text
E3 compiles source-specific disagreement into a counterfactual ranking lesson whose value is forecast before training and retained after teacher removal.
```

### Iteration Verdict

Chosen token:

```text
B52_I466_SURVIVE_IF_E3_NOVELTY_IS_SHAREABLE_COUNTERFACTUAL_LESSON_PACKETS
```

The novelty lives or dies on the packet object, not on disagreement.

## I467: Attack The 13-Kill Pattern Directly

### Pre-committed tokens

```text
B52_I467_SURVIVE_IF_E3_MEASURES_FUNCTION_NOT_TEACHER_SURFACE
B52_I467_KILL_IF_PACKET_GAIN_IS_ONLY_PSEUDOLABEL_SURFACE
B52_I467_VOID_IF_FUNCTIONAL_GEOMETRY_IS_NOT_OPERATIONALIZED
```

### Steelman

E3 tries to escape the old failure pattern. It does not treat teacher disagreement as truth. It evaluates hidden counterfactual transfer, requires teacher-free retained gain, and includes shuffled-source controls. That is closer to function than byte KL, proxy loss, hidden activation maps, or compact FrameSeed/WGD packets.

### Attack

At current toy level, E3 has not escaped the pattern. It has demonstrated a possible escape route under authored conditions.

The old pattern was: measurement surface looks structured; functional geometry is elsewhere.

The E3 risk is: teacher surface looks triangulable; functional geometry was authored into teacher names, transforms, and packet compiler.

Even hidden accuracy can be a proxy if the hidden split is simply the one regime the authored transform generator was designed to cover. A packet of examples is not necessarily geometry. A pseudolabel vector can be compact, effective, and still non-explanatory.

### New Hardest Objection

The current packet may not be an inspectable lesson. It may be a generated training set whose labels happen to be correct because the harness gave E3 the right teacher composition.

That is exactly the kind of proxy/function confusion the 13-kill synthesis warns against.

### Reconstruct What Survived

To escape the kill pattern, E3 must add tests that pseudolabel surfaces fail:

1. **Role-discovery test:** remove semantic role names and require E3 to infer the useful teacher axes from calibration/probe traces.
2. **Compression test:** the packet must be representable as a small explicit invariant/counterfactual rule, not only as examples.
3. **Edit test:** changing the named invariant should predictably change hidden behavior.
4. **Reuse test:** the same packet must improve multiple students or cohorts without re-querying teachers.
5. **Removal test:** teacher artifacts, teacher calls, and hidden constructors must be absent at inference.
6. **Geometry-parity test:** baselines receiving the authored geometry must still fail, or the E3 residual is zero.

### Iteration Verdict

Chosen token:

```text
B52_I467_SURVIVE_IF_E3_MEASURES_FUNCTION_NOT_TEACHER_SURFACE
```

Again this is conditional. The current toy is not enough.

## I468: Attack Mission Alignment

### Pre-committed tokens

```text
B52_I468_SURVIVE_IF_PACKETS_AMORTIZE_TEACHER_COST_AND_USE_OPEN_CHEAP_SOURCES
B52_I468_KILL_IF_E3_DEMOCRATIZES_DEPENDENCE_ON_EXPENSIVE_TEACHERS
B52_I468_VOID_IF_COST_LEDGER_OMITS_HUMAN_COMPILER_BITS
```

### Steelman

E3 has a mission-shaped story: instead of renting big models forever, use them or other sources to produce compact lessons that cheap local students retain. If the packet can be shared, audited, edited, and reused, it serves democratized development, data efficiency, and inference efficiency.

### Attack

That story can invert into a trap. If every useful packet requires expensive teacher calls, hand-authored probes, hidden ontologies, and researcher-written compilers, then E3 does not make intelligence cheap. It turns scarce expert infrastructure into a new bottleneck.

The poorest person on the street does not benefit from paying three proprietary teachers to generate bespoke lessons for every task. They benefit only if open cheap sources generate reusable packets that many small models can absorb.

### New Hardest Objection

E3 may democratize a dependency chain, not intelligence. A shareable lesson packet is mission-aligned only if the packet's all-in creation cost amortizes over many users and models.

### Reconstruct What Survived

Mission-aligned E3 must report:

```text
teacher_calls_per_packet
human_compiler_bits
probe_authoring_bits
student_training_cost
teacher_free_inference_cost
reuse_count_needed_to_break_even
packet_editability
open_source_reproducibility
```

The first natural-domain E3 result should be rejected if a cheap exact tool, retrieval pipeline, or weak-supervision label model gives the same utility at lower all-in cost.

### Iteration Verdict

Chosen token:

```text
B52_I468_SURVIVE_IF_PACKETS_AMORTIZE_TEACHER_COST_AND_USE_OPEN_CHEAP_SOURCES
```

E3 is mission-aligned only as amortized public lesson infrastructure.

## I469: Attack The Decision Not To Kill E3 Now

### Pre-committed tokens

```text
B52_I469_CONTINUE_IF_TOY_IS_POSITIVE_CONTROL_AND_HOSTILE_GATE_IS_CLEAR
B52_I469_KILL_IF_CURRENT_TOY_ALREADY_IMPLIES_SUPPLIED_GEOMETRY_ONLY
B52_I469_PIVOT_IF_HOSTILE_GEOMETRY_PARITY_ABSORBS_E3
```

### Steelman

The project has 13 kills. The first positive signal should not be murdered by purity reflex. The toy did exactly what a positive control should do: prove that the proposed mechanism can win in a world where its assumptions are true. The right move is not immediate kill. The right move is hostile escalation.

### Attack

Positive control can become a loophole. FrameSeed and WGD also had friendly worlds where the intended object looked real. The project only learned when it gave baselines the same structure and watched the signal collapse.

If B44 gives ordinary baselines the same transformations, role ontology, and composition search space, and those baselines match E3, then E3 should be killed in current form immediately.

The forbidden continuation would be: try a more natural domain because the hostile toy is embarrassing.

If the hostile toy kills E3, the lesson is not scale it. The lesson is that teacher tomography was still supplied geometry.

### New Hardest Objection

If E3 only works when the researcher names the teacher roles and the packet compiler, then the true missing invention is role and geometry discovery before teacher distillation.

Not better KD. Not more teachers. Not larger toys.

### Reconstruct What Survived

E3 survives this batch as a constrained live hypothesis:

```text
Source-specific teacher measurements may add residual value after authored geometry, exact tools, weak-supervision denoising, active learning, and teacher-selection KD get first refusal.
```

If that residual is zero, kill the current E3 mechanism and pivot to hypotheses that learn from the failure:

1. **Role-discovery hypothesis:** useful structure is not teacher disagreement but unsupervised discovery of teacher axes and refusal regions.
2. **Invariant-compression hypothesis:** the valuable object is a minimal explicit invariant/counterfactual rule, regardless of whether it comes from teachers.
3. **Source-ecology audit hypothesis:** teacher disagreement is useful for deciding which exact tool, verifier, or data source to use, not for training a student directly.

### Iteration Verdict

Chosen token:

```text
B52_I469_CONTINUE_IF_TOY_IS_POSITIVE_CONTROL_AND_HOSTILE_GATE_IS_CLEAR
```

Final batch token:

```text
Q_LOOP_B52_E3_TOY_SIGNAL_DOWNGRADED_TO_POSITIVE_CONTROL_HOSTILE_GEOMETRY_GATE_REQUIRED
```

## Failure Synthesis Check

The 13-kill pattern says every killed arc confused a measurement surface with functional geometry.

At current toy level, E3 does **not** yet escape that pattern. It shows a possible escape route:

```text
teacher measurements -> inferred counterfactual lesson -> teacher-free hidden transfer
```

But the current implementation makes the hostile objection too strong:

```text
teacher measurements + named roles + authored transforms + hard-coded composition -> correct pseudolabels
```

E3 escapes only if the next test proves all of the following:

1. Teacher identity remains necessary after baselines receive the same transform generator and candidate pool.
2. A rule-search or nuisance-oracle baseline cannot match the packet.
3. Exact tools either solve cheaper, killing the claim for that domain, or fail under the same all-in cost ledger.
4. The packet is an explicit reusable invariant/counterfactual object, not just a pseudolabeled example set.
5. Packet value is forecast before training and transfers after teacher removal.

Otherwise E3 falls into the 13-kill pattern as measurement-surface #14.

## NARRATIVE ATTACK

### 1. Strongest "that's obvious" dismissal

```text
Of course E3 wins. You built a binary XOR world, named the two latent sensors, and let E3 compose them. The other baselines were never given the operation that solves the task.
```

Answer:

```text
Correct as an attack on the current claim ceiling. The toy is only a positive control. It becomes non-obvious only if a geometry-parity absorber with the same transform generator, role ontology, and composition search space still fails.
```

### 2. Strongest "that's trivial" dismissal

```text
This is weak supervision or multi-teacher KD with a more dramatic name.
```

Answer:

```text
Correct unless E3 produces a reusable counterfactual lesson object whose hidden transfer is forecast before training and retained after teacher removal.
```

The anti-triviality clause is:

```text
source identity changes the inferred lesson type, shuffled identity kills the gain, active learning and weak-supervision denoising fail at matched cost, and the packet remains useful across students.
```

### 3. Mission test: does E3 serve making intelligence cheap/accessible?

Not yet. The friendly toy serves the mission only as a cheap falsifier and positive control.

E3 becomes mission-aligned if packets are cheap to create from open or low-cost teacher sources, inspectable and shareable, amortized across many students and users, and teacher-free at inference. E3 fails the mission if it requires bespoke proprietary teacher calls and hand-authored compilers for every domain.

### 4. What would the result need to BE for the narrative to be unkillable?

The unkillable result is:

```text
Given a failing small student and heterogeneous cheap teachers with no supplied role names, E3 infers the teacher axes, discovers a compact counterfactual ranking lesson, forecasts its value before training, trains the student, removes all teacher artifacts, and wins on hidden transformations while CE-only, extra data, best single teacher, teacher averages, Dawid-Skene/Snorkel-style source denoising, active learning, enhanced augmentation, role-oracle baselines, exact tools, retrieval, and chain-init fail at comparable all-in cost.
```

That result would make the public story legitimate:

```text
Instead of copying big AI answers, a tiny AI learns why the big AIs disagree.
```

## NEXT DIRECTIONS

### 1. Hostile supplied-geometry absorber test

Implement W-Loop B44 as the decisive toy escalation.

Required absorbers:

- **B13 exact domain tool:** admit the hidden constructor and record 1.0 as an absorber where applicable.
- **B15 nuisance oracle:** give baselines the transform axes and hidden nuisance structure without teacher tomography.
- **B10+ enhanced augmentation:** same transformed examples, same transform generator, and labels propagated by the granted transform law.
- **Boolean composition search:** search small formulas over teacher hard labels on the calibration set, then train on generated packet examples.
- **Role-map oracle:** give baselines the useful teacher pair without the E3 packet compiler.

Continuation token:

```text
E3_HOSTILE_TOY_SIGNAL_RESIDUAL_SOURCE_ECOLOGY
```

Kill token:

```text
E3_ABSORBED_BY_SUPPLIED_GEOMETRY_PARITY
```

### 2. Role-discovery ablation

Remove teacher role names. Randomly permute teacher identities, add decoy teachers, and require E3 to infer which teachers measure latent factors, which teachers are surface shortcuts, which teachers are redundant, which teacher pair composes into the target relation, and which signals should be refused.

If role discovery fails, the missing invention is not distillation; it is sensor calibration and source-axis discovery.

### 3. First natural weak-supervision domain

Use a small relation-extraction or slot-filling slice where cheap teachers are not rule-perfect:

- regex or dictionary source;
- entity-linker source;
- small local classifier;
- retrieval source;
- negation/temporal heuristic;
- optional verifier where available.

Hidden transformations should include entity swaps, negation flips, temporal shifts, and distractor-preserving paraphrases. Snorkel/data-programming source denoising must be a baseline, not an inspiration-only citation.

### 4. Code-diagnostics domain

Test a tiny code repair or bug-classification slice where teachers are unit tests, type checker, linter, static analyzer, small repair model, and retrieval over examples.

Exact tools get first refusal. E3 only survives if it produces a reusable lesson about failure mode and repair conditions that a small model retains after tool removal.

### 5. Packet reuse and amortization test

A lesson packet is not mission-aligned unless it amortizes.

Train at least three small students or student initializations from the same packet. Compare against extra labels, KD, active learning, exact tools, and retrieval. Report:

```text
reuse_count_needed_to_break_even
teacher_calls_per_successful_student
packet_edit_success_rate
hidden_transfer_after_teacher_removal
```

### Kill-contingent pivots

If hostile B44 kills E3, do not publish the kill as the deliverable. Pivot to:

1. **Unsupervised source-axis discovery:** infer teacher roles and refusal regions before any lesson compilation.
2. **Invariant-first packet discovery:** search directly for minimal counterfactual invariants, using teachers only as noisy probes.
3. **Source ecology as routing infrastructure:** use disagreement to choose between exact tools, retrieval, verifiers, and human data collection, rather than to train a student.

Kills are data. A supplied-geometry kill would teach that the missing home-run object is geometry discovery, not teacher disagreement.
