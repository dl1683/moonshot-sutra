# Q-Loop Batch 51: E3 Functional Teacher Tomography Sharpening

**Date:** 2026-07-08
**Role:** adversarial question loop
**Iterations:** I456-I462
**Status:** E3 claim sharpening under supervisor check-in #40

## Grounding

Read before writing: `research/VISION.md`, `research/STATUS.md`,
`research/question_loop_batch50.md`, `research/work_loop_batch42.md`,
`research/dual_loop_supervisor_checkin_40.md`, `research/EKLAVYA_DOCTRINE.md`,
`research/DEEP_RETHINK.md`, and `research/METHODOLOGY_TEMPLATE.md`.

The controlling constraint is still the substrate-open vision: mechanisms are
replaceable; the five sacred outcomes are not. The 13-kill synthesis says the
old arcs repeatedly confused a measurement surface with functional geometry.
E3 therefore has no right to treat teacher disagreement as truth. Teacher
disagreement is only an instrument reading.

External analogy search used for this batch:

- Geophysical joint inversion with structural regularization:
  https://arxiv.org/abs/1808.05441
- Gravity/magnetic joint inversion with cross-gradient constraints:
  https://arxiv.org/abs/2001.03579
- Crowdsourcing/rater measurement-error model with item heterogeneity:
  https://arxiv.org/abs/2405.19521
- Integrative multi-omics matrix factorization example:
  https://arxiv.org/abs/2103.03184

## Executive Verdict

E3 survives only if its claim is narrowed to this:

```text
E3 infers counterfactual ranking geometry from heterogeneous teacher
measurements and compiles it into student-owned lessons that transfer across
heldout transformations better than ordinary baselines at matched all-in cost.
```

Not allowed:

```text
teachers_as_sensors_general
teacher_disagreement_as_signal
multi_teacher_KD_with_better_words
hard_example_mining_with_extra_steps
```

Specific geometry object:

```text
counterfactual_ranking_geometry:
  C: context family
  Y: candidate/action set
  T_inv: transformations that must preserve the candidate ranking
  T_cf: transformations with known or inferred candidate-rank permutation phi
  R: pairwise ranking relation over Y under each probe
  S: sensitivity map naming which latent feature flips which ranking edge
  E: error/localization obligations explaining violated constraints
  G: student-gap state: absent_representation | unreadable_representation | surface_noise
  L: landing zone and lesson type predicted before training
```

Measurement object:

```text
teacher_signature_tensor:
  example_id
  probe_id
  teacher_id
  teacher_family
  candidate_pair
  signed_margin
  entropy_or_confidence
  declared_sensor_role
  measurement_cost
```

E3's claim is not that teacher margins are correct. The claim is that the
source-specific pattern of margin signs, stability, flips, and localization
across probes lets Eklavya infer the hidden transformation law cheaply enough
to teach the student.

Decision token:

```text
Q_LOOP_B51_E3_SPECIFIED_BUT_UNPROVEN_ONE_SHOT_KILL_REQUIRED
```

## I456: Attack B50 For Leaving "Geometry" Too Soft

### Previous Conclusion Under Attack

B50 redirected Eklavya from E2 byte-KL to E3 functional teacher tomography.

### Attack

"Functional geometry" is still too easy to worship. The old project already
made this mistake with hidden activations, byte marginals, FrameSeed packets,
and WGD grammars. A teacher disagreement map can become the next proxy surface.

If E3 says only:

```text
teachers are sensors for task structure
```

then it is not a claim. It is a metaphor. A hostile reviewer can dismiss it as
ensemble uncertainty, active learning, or curriculum generation.

The specific claim must be:

```text
Given a context family C, candidate/action set Y, and probe transformations,
heterogeneous teacher measurements identify a counterfactual ranking geometry:
which candidate rankings are invariant, which rankings flip under which
interventions, and which latent feature caused the flip.
```

The geometry is measured by:

1. `ranking_sign_accuracy`: pairwise order over candidates on hidden probes.
2. `invariance_consistency`: rankings remain stable under answer-preserving
   transformations.
3. `counterfactual_flip_accuracy`: rankings change according to the predicted
   transformation map `phi`.
4. `sensitivity_localization`: the inferred feature responsible for the flip
   matches a verifier, exact tool, or hidden construction.
5. `student_gap_prediction`: before training, E3 predicts whether the failure is
   absent representation, unreadable representation, or teacher-surface noise.
6. `teacher_free_retained_gain`: after training, the student improves without
   teacher calls or teacher artifacts at inference.
7. `packet_value_forecast_error`: predicted packet value tracks realized
   control-adjusted retained gain.

### Generative Countermove

Define E3 as an inverse problem:

```text
observed: teacher_signature_tensor over probes
unknown: counterfactual_ranking_geometry
compiled_output: lesson_packet with landing zone and removal test
```

No hidden activations. No byte marginals as terminal evidence. No aggregate
teacher vote. Only action-relevant ranking, invariance, counterfactual, and
localization structure.

### Narrative Attack

Obvious dismissal: "You renamed robust classification."

Trivial dismissal: "You generated paraphrases and counterfactuals."

Mission test: It serves the mission only if the inferred geometry gives a
smaller, reusable, inspectable lesson than extra labels, extra data, retrieval,
or a domain tool.

Unkillable version: E3 predicts which lesson type will transfer before training,
the prediction works on hidden transformations, teacher-axis erasure kills the
gain, and ordinary absorbers fail at matched cost.

### Iteration Verdict

E3's specific functional geometry is counterfactual ranking geometry. Anything
softer is dead.

## I457: Attack I456 For Smuggling The Answer Into The Probe Generator

### Previous Conclusion Under Attack

I456 says E3 should infer counterfactual ranking geometry.

### Attack

This can still cheat. If the researcher writes the transformations, candidate
set, and verifier obligations, then the geometry may already be supplied. That
was exactly the FrameSeed/WGD failure pattern: the packet looked compact because
the real structure had been moved into the constructor.

The hard question:

```text
Who paid for T_inv, T_cf, phi, candidate construction, and verifier semantics?
```

If E3 gets those for free, it is absorbed by human/substrate prior or domain
tool. If the exact checker already knows the transformation law, E3 is not
discovering geometry; it is formatting labels.

### Absorber Roster

E3 must lose unless it beats the strongest boring explanation in the applicable
domain. Required roster:

```text
B0_CE_only_same_student:
  same student, same base data, same steps, no teacher packets

B1_extra_data_same_cost:
  spend the teacher/probe budget on ordinary labeled or unlabeled data

B2_raw_byte_KL:
  old E2-style byte-marginal KD where available

B3_token_or_representation_KD:
  token-space, embedding, codec, or representation transfer where available

B4_best_single_teacher:
  strongest teacher alone, using the same probe/candidate budget

B5_naive_teacher_average:
  arithmetic or logit/margin average across teachers

B6_weighted_vote_or_Dawid_Skene_style:
  estimate teacher reliability/confusion and aggregate labels without E3 geometry

B7_shuffled_teacher_measurements:
  preserve marginals, destroy example/probe/teacher correspondence

B8_shuffled_teacher_identity:
  preserve measurements, destroy source role and sensor-family identity

B9_active_learning_or_hard_example_mining:
  same query budget, choose examples from student uncertainty or error gradients

B10_counterfactual_data_augmentation:
  same transformations and labels, no teacher-tomography inference

B11_curriculum_only:
  staged examples and difficulty labels without teacher-specific geometry

B12_readout_adapter:
  train an energy/ranking/readout head if the representation already contains
  the answer

B13_exact_domain_tool:
  solver, verifier, parser, compiler, SQL engine, unit checker, symbolic
  algebra, CEGIS, or constraint solver where applicable

B14_retrieval_or_chain_init:
  retrieve needed facts at inference, or initialize/compress through an ordinary
  anchor chain where applicable

B15_nuisance_oracle:
  give baselines the discovered invariant, role, parser, or binding to test
  whether E3 only hid a supplied prior
```

Baseline labels must be declared before hidden opening:

```text
native_executable | proxy_absorber | capability_mode_scored |
formal_lower_bound | untested_roster_entry
```

### Generative Countermove

Make E3 pay for every ounce of geometry:

```text
geometry_cost =
  candidate_generation_bits
  + transformation_generation_bits
  + verifier_semantics_bits
  + teacher_measurement_cost
  + packet_compiler_human_rules
  + student_training_cost
  + validation_cost
```

If a hand-authored transformation law is needed, it must be either charged as
human/substrate prior or given identically to baselines.

### Narrative Attack

Obvious dismissal: "Your clever probe generator did the work."

Trivial dismissal: "A domain checker plus augmentation solves this."

Mission test: E3 is useful only if it reduces the cost of discovering and
teaching the relevant geometry, not if it hides the geometry in authored probes.

Unkillable version: E3 infers which transformations matter from teacher response
surfaces, while baselines receiving the same candidate/probe budget cannot match
hidden transformation transfer.

### Iteration Verdict

The absorber roster must be brutal. The first E3 precommit must explicitly
charge probe and verifier structure, or E3 dies as another supplied-geometry arc.

## I458: Attack I457 For Designing A Giant Trial Instead Of A Fast Kill

### Previous Conclusion Under Attack

I457 builds a full absorber roster and cost ledger.

### Attack

This is how the project turns into another methodology museum. A giant harness
can be correct and still miss the mission. E3 needs one fast falsifier, not a
cathedral.

The cheapest kill question:

```text
Can source-specific teacher measurements predict a lesson's hidden-transform
value better than active learning, best single teacher, and shuffled sensors
before any score is seen?
```

If not, teacher tomography buys nothing.

### One-Shot Kill Test

Domain:

```text
controlled_candidate_ranking_world_v0
```

Requirements:

- CPU-runnable.
- Candidate/action set is explicit.
- Hidden transformations include paraphrase, distractor-preserving,
  irrelevant-slot change, and true counterfactual flips.
- The constructor hides which latent feature controls the flip until opening.
- Teachers are heterogeneous sensors, not clones:
  - one lexical/surface-biased teacher;
  - one semantic/paraphrase-biased teacher;
  - one verifier/localization teacher;
  - optional weak noisy teacher.
- Student starts with measured failures.

Protocol:

```text
1. Freeze public manifest, seeds, hidden constructors, teacher roles, costs,
   terminal tokens, and baselines.
2. Generate a small public scout set and a hidden transform set.
3. Query teachers on local probe neighborhoods.
4. E3 infers lesson packets and ranks them by predicted packet_value_prior.
5. Train only the top-k E3 packets under a tiny fixed budget.
6. Train B0-B15 absorbers under matched cost where applicable.
7. Open hidden transformations once.
8. Assign exactly one terminal token.
```

Continuation gate:

```text
E3_SIGNAL only if:
  E3 hidden counterfactual ranking accuracy beats best(B0..B12) by >= 5pp
  and E3 beats B9 active learning by >= 3pp
  and E3 beats B4 best single teacher by >= 3pp
  and E3 beats B7/B8 shuffled controls by >= 6pp
  and packet_value_prior ranks successful packets above failed packets
  and teacher-free retained gain is positive
  and no exact domain tool B13 solves the task at lower all-in cost
```

Kill tokens:

```text
E3_ABSORBED_BY_CE_OR_EXTRA_DATA
E3_ABSORBED_BY_SINGLE_TEACHER
E3_ABSORBED_BY_TEACHER_AVERAGE_OR_WEIGHTED_VOTE
E3_ABSORBED_BY_ACTIVE_LEARNING
E3_ABSORBED_BY_AUGMENTATION
E3_ABSORBED_BY_READOUT_ADAPTER
E3_ABSORBED_BY_DOMAIN_TOOL
E3_ABSORBED_BY_CHAIN_INIT_OR_RETRIEVAL
E3_SHUFFLED_SENSORS_MATCH_REAL
E3_PROXY_ONLY_TEACHER_SURFACE
E3_VOID_SUPPLIED_GEOMETRY_OR_LEAKAGE
E3_NEGATIVE
```

### Generative Countermove

The first experiment should not try to prove E3 generally. It should ask whether
the central object exists:

```text
teacher_ecology_residual =
  value(E3 source-specific geometry)
  - max(value(active learner),
        value(best single teacher),
        value(weighted vote),
        value(augmentation),
        value(exact tool))
```

If this residual is not positive on a friendly controlled world, stop.

### Narrative Attack

Obvious dismissal: "You designed an artificial world where E3 wins."

Trivial dismissal: "Toy worlds do not prove cheap useful AI."

Mission test: A toy win does not satisfy the mission, but a toy loss kills the
mechanism cheaply. That is the correct asymmetry.

Unkillable version: E3 wins first in the toy, then survives the same absorber
logic on one exact-tool domain and one natural-language candidate-ranking slice.

### Iteration Verdict

The one-shot kill test is a packet-value forecast test on hidden transformations.
If teacher-specific measurements cannot forecast and teach hidden-transform
value, E3 is dead.

## I459: Attack I458 For Inventing Methodology From Scratch

### Previous Conclusion Under Attack

I458 designs a one-shot kill test.

### Attack

The test still sounds homegrown. If E3 is real, there should be a mature outside
analogy where multiple imperfect instruments infer one hidden structure. If
there is no such analogy, "teacher tomography" may be aesthetic invention.

### Cross-Domain Analogy Search

Candidate analogies:

```text
statistics_inter_rater_reliability:
  observers give noisy labels
  infer latent true label plus observer error rates
  useful method stolen: model teacher confusion, item difficulty, and
  heterogeneity; never trust majority vote blindly
  limitation: usually infers labels, not transformation geometry

biology_multi_omics:
  transcriptome, epigenome, proteome, metabolome measure different views of a
  biological state
  useful method stolen: shared latent factors plus modality-specific factors;
  do not force every modality into one early concatenation
  limitation: often correlation-heavy and validation is hard

physics_and_geophysics_joint_inversion:
  gravity, magnetic, seismic, resistivity, or PDE-governed measurements observe
  different physical properties of one hidden medium
  useful method stolen: independent forward models, single-instrument inversions,
  joint structural regularization, noise weighting, synthetic phantoms, and
  comparison against independent inversion
  limitation: E3 teachers do not have clean physical forward equations yet
```

Strongest analogy:

```text
geophysical_joint_inversion_with_structural_coupling
```

Why it wins:

- It distinguishes measurement from hidden structure.
- Each sensor has its own forward model and noise.
- Independent inversion is a required baseline.
- Joint inversion is allowed only through a structural coupling term, not by
  naively averaging sensor values.
- The coupling can assert shared boundaries/topology without asserting that the
  physical values are equal.
- It is evaluated on synthetic phantoms and real cases against independent
  inversions.

Methodology to steal:

```text
E3_joint_inversion_protocol:
  1. Define a teacher-specific forward model:
       teacher_measurement = F_teacher(counterfactual_ranking_geometry) + noise

  2. Run independent inversion per teacher:
       infer geometry from each teacher alone
       score against hidden transformations

  3. Run naive fusion:
       average or vote teacher measurements
       score against hidden transformations

  4. Run structural joint inversion:
       fit all teacher measurements while penalizing disagreement in the
       boundaries of ranking flips, invariance regions, and localization spans

  5. Sweep coupling strength:
       too weak = independent sensors
       too strong = forced consensus and erased minority expertise
       useful = improves hidden reconstruction without erasing source roles

  6. Randomize sensors:
       shuffled teacher identity and shuffled example/probe pairing must break
       the joint advantage

  7. Open hidden transformations once:
       score geometry reconstruction and student-owned retained gain
```

E3 structural coupling term:

```text
minimize:
  sum_t data_misfit(F_t(G), M_t)
  + lambda_structure * disagreement_between_flip_boundaries(G_t)
  + lambda_source * penalty_for_erasing_valid_teacher_specific_expertise
  + lambda_cost * all_in_measurement_and_training_cost
```

The E3 version of cross-gradient is not "make teacher logits similar." It is:

```text
ranking flip boundaries and invariant regions should align across sensors when
they measure the same latent distinction, while sensor-specific biases remain
explicit.
```

### Narrative Attack

Obvious dismissal: "Physics has real equations; your teachers are opaque."

Trivial dismissal: "Without forward models, joint inversion is just a metaphor."

Mission test: E3 must earn forward models empirically by calibration records:
what hidden state each teacher measures, when it fails, and what signal is
refused.

Unkillable version: E3 creates teacher calibration curves and source-specific
error models, then joint inversion beats independent teacher inversions and
naive fusion on hidden transformations.

### Iteration Verdict

Steal geophysical joint inversion, not the word tomography. E3 must implement
single-sensor inversion, naive fusion, structural joint inversion, coupling
sweep, shuffled-sensor controls, and hidden phantoms.

## I460: Attack I459 For Dodging The Active Learning Objection

### Previous Conclusion Under Attack

I459 says E3 should borrow joint inversion methodology.

### Attack

Even with joint inversion language, the core objection remains:

```text
Is teacher tomography just active learning with extra steps?
```

If all E3 does is find examples where models disagree, then yes. Active learning
already asks for labels on uncertain or high-value examples. Hard-example mining
already focuses training on failures. Query-by-committee already uses ensemble
disagreement.

E3 must name what teacher ecology buys that a single strong model plus hard
example mining does not.

### Concrete Answer

Teacher ecology buys exactly four things. If any one is absent, E3 weakens. If
all four are absent, E3 is dead.

```text
1. Error decomposition, not error discovery:
   active learning says "this example is hard."
   E3 must say "this example is hard because invariance is missing, a
   counterfactual feature is mislocalized, representation is absent, or readout
   is unreadable."

2. Sensor complementarity, not uncertainty:
   a verifier, encoder, decoder, and symbolic tool fail differently. E3 must
   use the pattern of disagreement to infer which latent distinction matters.
   A single model cannot expose cross-family measurement biases.

3. Lesson-type prediction:
   before training, E3 must predict whether to use a ranking packet, invariance
   packet, counterfactual packet, readout adapter, verifier-localization packet,
   exact tool, retrieval, or no packet.

4. Source-specific refusal:
   E3 must learn when not to use a teacher. It should reject style priors,
   tokenizer artifacts, correlated teacher echoes, and high-confidence nonsense.
```

The direct test:

```text
Give active learning the same budget and the same candidate pool.
Give E3 the same examples plus source-specific teacher measurements.
If E3 cannot beat active learning on hidden transformations, or if destroying
teacher identity does not remove the E3 advantage, tomography adds nothing.
```

### Generative Countermove

Make "teacher ecology residual" a first-class metric:

```text
teacher_ecology_residual =
  hidden_transfer(E3_with_source_identity)
  - hidden_transfer(best_active_learning_baseline_same_budget)
```

And require:

```text
identity_dependence =
  hidden_transfer(E3_with_source_identity)
  - hidden_transfer(E3_with_shuffled_teacher_identity)
```

E3 survives only if both are positive.

### Narrative Attack

Obvious dismissal: "Query-by-committee did this decades ago."

Trivial dismissal: "You just choose hard examples from ensemble disagreement."

Mission test: Hard examples alone do not democratize intelligence. Inspectable
failure decompositions and reusable lesson packets might.

Unkillable version: The same hard examples are available to all baselines, but
only E3's source-specific measurement model predicts the correct lesson type
and transfers on hidden transformations.

### Iteration Verdict

E3 is not active learning only if teacher identity changes the inferred lesson
type and the hidden retained gain. That must be tested directly.

## I461: Attack I460 For Being Too Dry To Recruit Belief

### Previous Conclusion Under Attack

I460 answers the active-learning objection technically.

### Attack

The mission is not only to satisfy internal auditors. A paradigm-shifting idea
needs a public handle. B50's headline was alive but unevidenced. E3 still risks
sounding like a grant abstract:

```text
counterfactual ranking geometry from heterogeneous teacher measurements
```

That will not make a celebrity-magazine reader care.

### Gossip-Magazine Headline Search

Candidate 1:

```text
Big AIs argue, tiny AI learns the secret.
```

Attack: cute, but too vague. It sounds like ensemble learning.

Candidate 2:

```text
Scientists teach a laptop AI by making expert AIs fight and stealing the rule
behind the fight.
```

Attack: stronger. "Stealing the rule" carries the geometry. "Laptop AI" carries
cheap and ubiquitous. "Expert AIs fight" carries teacher disagreement.

Candidate 3:

```text
Instead of copying big AI answers, a tiny AI learns why the big AIs disagree.
```

Attack: clearest. Less sensational, more accurate. It separates E3 from KD and
active learning.

Chosen headline:

```text
Instead of copying big AI answers, a tiny AI learns why the big AIs disagree.
```

Subheadline:

```text
If the trick works, communities could share compact lessons instead of renting
larger models forever.
```

### Headline Test

"Isn't that obvious?"

Answer:

```text
No, because ordinary distillation copies outputs and active learning collects
hard labels. E3 claims the disagreement pattern reveals the hidden rule that
makes the answer stable or flip under intervention.
```

"Isn't that trivial?"

Answer:

```text
It is trivial unless source identity matters, shuffled sensors fail, and active
learning loses at matched cost.
```

"Why should an ordinary person care?"

Answer:

```text
Because the useful artifact is not a private checkpoint. It is a small lesson
packet that can be inspected, shared, corrected, and used to improve cheap local
models.
```

### Generative Countermove

The public story should not mention tomography first. It should say:

```text
We are trying to turn expert disagreement into portable lessons for small AI.
```

Then the technical paper earns:

```text
counterfactual ranking geometry via structural joint inversion of teacher
measurement surfaces.
```

### Narrative Attack

Obvious dismissal: "This is PR."

Trivial dismissal: "A nice headline cannot rescue a weak method."

Mission test: The headline is useful only if it forces the method to produce
shareable lessons, not just a benchmark bump.

Unkillable version: A nontechnical reader can state the claim accurately:
"the little AI learns the reason experts disagree, not just their answers."

### Iteration Verdict

The gossip headline survives only with the anti-triviality clause attached:
source identity and lesson-type prediction must beat active learning and shuffled
sensors.

## I462: Attack I461 For Letting Narrative Hide The 13-Kill Pattern

### Previous Conclusion Under Attack

I461 gives E3 a public handle.

### Attack

Narrative is dangerous. The 13 kills did not fail because their slogans were
bad. They failed because the measured thing was not the function. E3 could
become kill #14:

```text
teacher disagreement improved
lesson packets looked compact
hidden transformations barely moved
active learning or exact tools absorbed it
```

The failure synthesis predicts exactly where E3 should look:

```text
where teacher measurement surfaces diverge but the hidden functional geometry
is stable enough to infer cheaply.
```

But it also predicts E3's death mode:

```text
teacher disagreement is just another measurement surface.
```

### Does E3 Avoid The Pattern?

Only if it obeys these laws:

```text
1. Teacher disagreement is never the target.
   It is an observation to invert.

2. The inferred object is action geometry.
   Ranking, invariance, counterfactual flips, localization, and student-gap
   state are the measurable functional variables.

3. Every teacher has a sensor record.
   what_hidden_state_it_measures
   how_it_differs_from_existing_sensors
   what_student_gap_it_exposes
   where_the_lesson_can_land
   what_it_costs_to_measure
   what_signal_is_refused
   what_removal_test_it_must_survive

4. Independent teacher inversions are baselines.
   If one teacher alone gives the geometry, multi-teacher E3 is absorbed.

5. Active learning gets equal budget.
   If uncertainty sampling or hard-example mining matches hidden transfer,
   E3 is absorbed.

6. Exact tools get first refusal.
   If a solver, verifier, or retrieval system solves the slice cheaper, use it
   and do not call E3 a breakthrough.

7. Teacher-axis erasure must hurt.
   If shuffled identity or shuffled measurements keep the gain, source ecology
   was fake.

8. Packet value must be forecast, not post-rationalized.
   E3 must predict which packets will work before training.

9. Teacher-free retained gain is binding.
   Inference cannot depend on teacher calls or hidden artifact lookup.
```

### Final E3 Direction Charter

```text
Direction name:
  E3 Functional Teacher Tomography

Core claim:
  Source-specific measurements from heterogeneous teachers can be structurally
  inverted into compact counterfactual ranking geometry, and that geometry can
  be compiled into student-owned lessons that transfer across hidden
  transformations better than ordinary baselines at matched all-in cost.

One-sentence home-run story:
  Instead of copying big AI answers, a tiny AI learns why the big AIs disagree.

Which sacred outcomes it serves:
  genuine_intelligence: tests action-relevant transformations, not logits
  improvability: failures map to lesson types and landing zones
  democratized_development: lessons are inspectable and shareable
  data_efficiency: one inferred law should replace many labels
  inference_efficiency: no teacher calls at inference

What ordinary explanation would make it boring:
  active learning, best single teacher, weighted vote, augmentation, readout
  adapter, exact tool, retrieval, or chain-init gives the same hidden transfer
  cheaper.

What evidence would kill it:
  shuffled sensors match real, active learning matches E3, best single teacher
  matches E3, packet value forecasts fail, exact tools solve cheaper, or hidden
  transformation transfer is absent.

What evidence would make a hostile reviewer pause:
  E3 predicts lesson value before training, teacher identity is necessary, hidden
  counterfactual transfer survives teacher removal, and all ordinary absorbers
  fail at matched cost.
```

### Final Narrative Attack

Obvious dismissal:

```text
Teacher tomography is just active learning with more ceremony.
```

Answer:

```text
Then active learning will absorb it. The first E3 experiment must give active
learning equal budget and stop if it matches E3.
```

Trivial dismissal:

```text
Teacher disagreement is just another proxy.
```

Answer:

```text
Correct. E3 survives only by reconstructing hidden counterfactual ranking
geometry and proving student-owned transfer on hidden transformations.
```

Mission test:

```text
Does E3 make intelligence cheaper, more accessible, and more improvable?
```

Answer:

```text
Not yet. It becomes mission-aligned only if lesson packets are reusable public
infrastructure and beat extra labels, active learning, exact tools, retrieval,
and chain-init on all-in cost for at least one meaningful slice.
```

Unkillable version:

```text
Given a small failing student, E3 calibrates heterogeneous teachers as sensors,
infers a compact counterfactual ranking geometry, forecasts which lesson packet
will transfer, trains the student, removes all teacher artifacts, and wins on
hidden transformations while CE-only, extra data, best single teacher, teacher
averages, weighted vote, shuffled sensors, active learning, augmentation,
readout adapters, exact tools, retrieval, and chain-init cannot match the
control-adjusted retained gain at comparable all-in cost.
```

### Iteration Verdict

E3 avoids the 13-kill pattern only by treating teacher disagreement as an
instrument reading to be inverted, not as the geometry itself. The next work-loop
must implement the one-shot kill test before any larger E3 claim is allowed.

## Bottom Line

E3 is alive but on probation.

The narrowed live bet:

```text
Teacher ecology can reveal the counterfactual ranking geometry of a task, not
because teachers are wise, but because their different failure surfaces make the
hidden functional structure triangulable.
```

The immediate kill gate:

```text
If source-specific teacher measurements cannot beat active learning, best single
teacher, weighted vote, augmentation, shuffled sensors, exact tools, and
retrieval/chain-init where applicable on hidden transformation transfer, E3 is
dead.
```

The next artifact should be runnable, not another doctrine file.
