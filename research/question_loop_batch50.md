# Q-Loop Batch 50: Eklavya Restart After 13 Kills

**Date:** 2026-07-08
**Role:** adversarial question loop
**Iterations:** I449-I455
**Status:** kill-or-redirect review of Eklavya E2

## Grounding

This batch starts from the substrate-open reset in `research/VISION.md`, the full kill record in `research/STATUS.md`, the current Eklavya doctrine, E2 protocol, deep rethink log, field survey, methodology template, and the live E2/S0 code.

The local status matters: `STATUS.md` says the methodology paper passed the terminal gate. This batch treats the user's prompt as an intentional restart of Eklavya under the higher substrate-open standard. Eklavya therefore gets no mainline privilege. It must re-earn admission against the five sacred outcomes.

External field check on 2026-07-08: I checked current primary arXiv-facing surfaces for post-survey movement in cross-tokenizer KD, byte-level distillation, multi-teacher KD, and on-policy distillation. I did not find a clear post-2026-06-27 primary result that changes the decision here. That is not proof of absence. The existing 2026 papers already make the competitive window tight enough: BLD validates byte interfaces but leaves CTD inconsistent; X-Token already shows two-teacher cross-tokenizer gains; Knowledge Purification validates router-style multi-teacher conflict handling; Token-to-Byte distillation validates representation-level byte conversion; CBD proves strong small-model performance through chain initialization; OPD work is moving distillation toward interactive functional correction.

Primary source links used for current-field check:

- BLD: https://arxiv.org/abs/2604.07466
- X-Token: https://arxiv.org/abs/2605.21699
- Knowledge Purification: https://arxiv.org/abs/2602.01064
- Token-to-Byte Distillation: https://arxiv.org/abs/2602.01007
- Chain-Based Distillation: https://arxiv.org/abs/2605.07783
- OPD Survey: https://arxiv.org/abs/2604.00626

## Executive Verdict

Eklavya is not dead. **E2 as designed is not the right mainline.**

The surviving moonshot is not "better multi-teacher KD." It is:

```text
Eklavya = gauge-invariant teacher tomography and lesson compilation.
```

Teachers are not masters to imitate. They are sensors. The object to discover is not a byte distribution, hidden vector, router weight, or KL target. The object is the operational geometry of a task: which distinctions matter, which transformations preserve action, which counterfactuals change action, which failure modes each teacher measures, and what smallest lesson causes the student to own that structure.

Decision token:

```text
Q_LOOP_B50_REDIRECT_EKLAVYA_FROM_E2_KD_TO_FUNCTIONAL_TOMOGRAPHY
```

Operational consequence:

- Do **not** push E2 byte-marginal multi-teacher KD as the home-run claim.
- Keep E2 code as instrumentation, negative-control machinery, and an absorber baseline.
- Reframe the next Eklavya arc around functional retained gain from teacher tomography: rankings, invariance probes, counterfactual probes, verifier localization, semantic addressability, and lesson value-of-information.

## I449: Is Eklavya Even The Right Direction?

### Starting Position

E2 says the old routing mechanism died, but the answer is to redesign routing: preserve teacher identity, build per-teacher ports, route/purify disagreement, budget gradients, and prove retained gain after teacher removal.

### Attack

This may already be the wrong game. `VISION.md` explicitly says the substrate is open and that neural networks get no sacred status. `DEEP_RETHINK.md` says KD improved byte prediction while downstream judgment stayed flat. The E2 protocol still centers byte KL, alignment/cosine ports, router weights, purified byte targets, and gradient caps. That is a better-engineered proxy machine, not yet a function machine.

The fatal question:

```text
What function does a purified byte target measure?
```

If the answer is "teacher next-byte belief," then E2 is still optimizing the same class of proxy that killed E1, Option C, S0 capacity, FMD, and routing. The teacher axis survives, but the transferred object may still be a shadow.

The E2 code confirms the risk. `eklavya_e2_router.py` routes distributions using gold likelihood, entropy, agreement, and student JSD. `eklavya_e2_losses.py` turns the routed result into sparse byte KL. `eklavya_e2_training.py` logs BPB, teacher losses, route stats, gradient caps, and CE. Those are necessary engineering controls, but none of them by themselves prove that a functional distinction entered the student.

### Generative Countermove

Kill "Eklavya = KD." Preserve "Eklavya = learning from external competence" only if the transferred object becomes a functional invariant:

```text
student failure
  -> local probe neighborhood
  -> teacher/verifier measurements
  -> inferred invariant/counterfactual structure
  -> compiled lesson
  -> student-native training
  -> heldout transformation test
  -> teacher-free retained gain
```

This is not output-distribution distillation. It is experimental science on teachers.

### Narrative Attack

Obvious dismissal: "You rediscovered that KD should use better data and probes."

Trivial dismissal: "This is just active learning plus data augmentation."

Mission test: E2-as-KD risks drifting into documenting losses and ablations rather than making intelligence cheaper. Tomography serves the mission only if one teacher-measured invariant generates many student-owned examples or repairs.

Unkillable version: a tiny student gains function from a small number of teacher-derived lessons, beats CE-only, raw KD, best single teacher, naive average, shuffled-teacher controls, and a cheap active learner on heldout transformations, with teacher-free inference and lower all-in cost.

### Iteration Verdict

Eklavya survives only after KD is demoted. E2 as designed is not the home-run direction.

## I450: Attack I449 For Overkilling KD

### Previous Conclusion Under Attack

I449 says KD is the wrong frame and E2 should be demoted.

### Attack

This could be an overcorrection. The data problem is brutal. The deep rethink log estimates that the 121M model sees orders of magnitude less data than standard small models. CBD's 138M result shows that small models can inherit much more capability when the transfer path preserves coordinate continuity. Token-to-byte distillation shows that byte-level students can retain token-model capability if the transfer happens at the representation level, not merely through byte marginals.

So the statement "KD is dead" is false. What died is:

```text
cross-architecture output/coordinate matching without proof of functional transfer
```

E2 may still have value because it has three hard-won pieces:

1. shared position manifest;
2. preserved teacher identity;
3. ablation machinery to test single-teacher, average, shuffled, no-router, and no-gradient-budget explanations.

That infrastructure can be the scaffold for tomography. Throwing it away would repeat a different failure mode: abandoning useful instruments because the first hypothesis attached to them was wrong.

### Generative Countermove

Rename E2's role:

```text
E2 is not the claim.
E2 is a measurement instrument and absorber suite.
```

Keep cache discipline, teacher registry, teacher identity axis, route/weight telemetry, shuffled-target falsification, single-teacher and no-router ablations, and cost/gradient ledgers.

Replace the central training object from byte KL targets to functional probe surfaces, candidate rankings, invariance/counterfactual maps, verifier spans, and lesson packets.

### Narrative Attack

Obvious dismissal: "This is just E2 with a new name."

Trivial dismissal: "All you changed is the data format."

Mission test: If the rewrite still optimizes BPB and calls it intelligence, it fails. If it turns a few teacher probes into reusable function-preserving lessons, it serves data efficiency and democratized development.

Unkillable version: the same E2 cache/router machinery fails as byte KL but succeeds when the target is an operational lesson object, proving the difference is the transferred geometry, not the engineering wrapper.

### Iteration Verdict

Do not kill all KD. Kill KD as imitation. Keep E2 as an instrument. The new claim must be functional transfer, not distribution transfer.

## I451: Attack I450 For Saving A Weak Narrative

### Previous Conclusion Under Attack

I450 says E2 can be kept as instrument and maybe salvaged by replacing the target.

### Attack

The public story "a 121M model learned from 5 teachers and retained the knowledge after removal" is not enough. In 2026 that sounds like a decent KD paper, not a paradigm shift. The field already has byte-interface KD, cross-tokenizer projection KD, token-to-byte representation distillation, multi-teacher purification, and chain-based initialization at the same parameter scale.

Even if E2 gets a positive result, the obvious reviewer response is:

```text
You combined known ingredients: byte interface, teacher routing, gradient caps, and ablations.
```

The stronger hostile response:

```text
Your five teachers are just correlated expensive labels. Show me the new principle of intelligence.
```

E2's current home-run story is too easy to reduce to engineering. It does not force a brilliant outsider to care about the question.

### Generative Countermove

The story must become:

```text
We discovered how to extract architecture-independent lessons from teacher disagreement, and those lessons taught a small student function that byte-KD, single teachers, averages, and ordinary baselines could not teach.
```

The unkillable object is not the 121M model. It is the method for discovering which distinctions matter.

A result worth chasing:

```text
Given a failing student and a small teacher roster, Eklavya constructs a lesson graph whose nodes are invariances, counterfactual transformations, verifier obligations, and candidate-ranking relations. Training on that graph produces teacher-free retained gain on heldout transformations at lower cost than raw teacher distillation or extra data.
```

This would make "Intelligence = Geometry" concrete. It would say intelligence is not stored in teacher logits. It is recoverable as a small set of task distinctions and transformations.

### Narrative Attack

Obvious dismissal: "Program synthesis and active learning already do this."

Trivial dismissal: "You made a curriculum generator."

Mission test: A curriculum generator is not enough unless it lowers the cost of useful capability for ordinary builders. The lesson graph must be inspectable, reusable, and cheaper than buying more model calls or training tokens.

Unkillable version: release a small lesson compiler where community members can add teachers/verifiers, generate lesson packets, and improve a local student without retraining from scratch. That serves democratized development more than one private checkpoint.

### Iteration Verdict

The E2 narrative is too weak. The paradigm-shift narrative is lesson discovery from disagreement, not multi-teacher KD.

## I452: Attack I451 For Drifting Into Process Instead Of Useful AI

### Previous Conclusion Under Attack

I451 reframes Eklavya as lesson graph discovery.

### Attack

This risks repeating the methodology-paper pivot: beautiful anti-self-deception machinery, no cheap useful AI. The mission is not "build a philosophy of lessons." The mission is cheap, ubiquitous, useful intelligence.

If the next artifact is another doctrine document, it fails the mission test. If the next experiment is a toy lesson graph that gets absorbed by active learning, it fails. If the result requires expensive teacher probing on every new task, it fails inference and development efficiency.

The question loop must not turn the project into a museum of disciplined kills.

### Counterattack

E2-as-designed also risks this failure. A 0.5-2 point benchmark gain after a complex multi-teacher cache and ablation suite would not be mission-significant. It might be publishable engineering, but it would not democratize intelligence.

The right way to avoid process drift is not to return to E2. It is to require every proposed lesson system to pass a utility gate:

```text
net_retained_gain =
  control_adjusted_teacher_free_function_gain
  - teacher_query_cost
  - lesson_construction_cost
  - training_cost
  - validation_cost
  - collateral_damage
```

If this number is not positive against CE-only, raw KD, active learning, and chain-init or retrieval controls, the direction dies.

### Competitive Window Check

The field survey's June 2026 novelty estimate was already only moderate. The window has not reopened. If anything, the external direction is worse for E2:

- BLD makes the byte interface less novel.
- X-Token makes multi-teacher cross-tokenizer KD less novel.
- Knowledge Purification makes routing/purification less novel.
- Token-to-Byte distillation makes representation-level byte conversion the more credible transfer path.
- CBD makes 121M/138M performance claims answerable by chain initialization, not clever post-hoc byte KL.
- OPD reframes distillation around student-generated failures and iterative correction, closer to function than cached imitation.

The absence of an obvious post-June-27 kill paper does not help E2. Existing work is enough to force a higher bar.

### Narrative Attack

Obvious dismissal: "This is an overcomplicated evaluation harness for curriculum learning."

Trivial dismissal: "Run CBD or use retrieval if you want a useful small model."

Mission test: A direction serves the moonshot only if it gives independent builders a cheaper route to capability than renting larger models or copying standard distillation recipes.

Unkillable version: the lesson compiler wins on all-in cost against CBD-like compression for a slice where chain-init is overkill or unavailable, and against retrieval for a slice where local teacher-free competence matters.

### Iteration Verdict

The redirect is only acceptable if it becomes a utility-positive functional system. Do not let "lesson graph" become another process artifact.

## I453: Attack I452 For Letting Competition Define The Goal

### Previous Conclusion Under Attack

I452 says competitive pressure and utility gates should drive the redirect.

### Attack

Competition is the wrong center. The vision says "Intelligence = Geometry, Not Scale." If the project merely asks how to beat X-Token, BLD, CBD, or OPD, it is still letting the field define the question. The bigger question is what the 13 kills say about intelligence itself.

The pattern across the kills is deeper than "KD failed":

1. proxy metrics moved while function did not;
2. supplied structure kept getting absorbed by ordinary baselines;
3. hidden artifacts looked impressive until a better boring explanation got equal information;
4. outputs and representations were not stable evidence of capability;
5. the real missing object was a function-aligned measurement.

This is the core synthesis:

```text
Every killed arc confused a measurement surface with the functional geometry behind it.
```

For KD, logits are the measurement surface. For representation alignment, hidden vectors are the measurement surface. For FrameSeed, packets were the measurement surface. For WGD, supplied grammars were the measurement surface. For CTI, smooth proxy laws were the measurement surface.

Eklavya becomes worth pursuing only if it attacks the hidden axiom:

```text
Can we infer the functional geometry behind multiple measurement surfaces?
```

### Generative Countermove

Make teacher disagreement the microscope, not the target.

Teachers are sensors with different biases. The object of study is the latent task state they imperfectly measure:

- decoder teacher: candidate behavior and language priors;
- encoder teacher: semantic neighborhood structure;
- verifier teacher: constraint violation and repair locality;
- symbolic checker: exact function on a narrow domain;
- curriculum teacher: prerequisite and example density;
- byte-boundary source: abstraction loss and surface exactness.

The experiment is to triangulate the hidden state:

```text
hidden_state_measured_by_teacher
  -> operational invariant
  -> minimal lesson
  -> student-owned function
```

This directly serves the geometry thesis. The geometry is not teacher geometry. It is the structure invariant across teacher measurement systems.

### Narrative Attack

Obvious dismissal: "This is just ensemble disagreement."

Trivial dismissal: "You are mining hard examples."

Mission test: Hard-example mining alone does not democratize intelligence. A teacher-disagreement atlas does, if it exposes reusable maps of failure modes, invariances, and repairs that many small students can use.

Unkillable version: show that disagreement patterns predict which lesson type will transfer, before training, and that the forecast improves over batches. That turns teacher choice from art into a public, auditable science.

### Iteration Verdict

Do not make Eklavya about beating the current KD field. Make it about discovering the functional geometry behind teacher behavior.

## I454: Attack I453 For Being Too Abstract

### Previous Conclusion Under Attack

I453 says Eklavya should infer functional geometry from teacher measurement surfaces.

### Attack

This is still too easy to admire and too hard to falsify. "Functional geometry" can become a slogan unless it compiles into a harness with terminal tokens, absorbers, and cheap gates. The methodology template exists exactly because attractive concepts kept surviving without enough ordinary baselines.

The next Eklavya arc must be killable before any GPU run.

### Concrete Redirect: Eklavya E3

Direction name:

```text
E3: Functional Teacher Tomography
```

Core claim:

```text
Multiple heterogeneous teachers can be used as sensors to infer compact, student-ownable functional lessons that transfer across heldout transformations better and cheaper than raw KD, CE-only, active learning, single-teacher distillation, teacher averaging, and chain-init/retrieval controls on the same slice.
```

Functional lesson object:

```text
lesson_packet:
  context_family:
  candidate_or_action_space:
  teacher_measurements:
  invariant_transformations:
  counterfactual_transformations:
  ranking_relations:
  verifier_obligations:
  student_gap_evidence:
  landing_zone:
  predicted_value:
  corruption_risk:
  training_recipe:
  heldout_transformation_tests:
  teacher_free_retained_gain:
```

Required absorbers:

- CE-only same student;
- extra-data same cost;
- raw byte KL;
- token/representation KD where available;
- best single teacher;
- naive teacher average;
- shuffled teacher measurements;
- active learner / hard-example miner;
- exact domain tool or verifier where applicable;
- CBD-like chain-init or retrieval when the slice permits it.

Terminal tokens:

```text
E3_SIGNAL
E3_ABSORBED_BY_CE_OR_EXTRA_DATA
E3_ABSORBED_BY_SINGLE_TEACHER
E3_ABSORBED_BY_TEACHER_AVERAGE
E3_ABSORBED_BY_ACTIVE_LEARNING
E3_ABSORBED_BY_DOMAIN_TOOL
E3_ABSORBED_BY_CHAIN_INIT_OR_RETRIEVAL
E3_PROXY_ONLY
E3_VOID_PROTOCOL_OR_LEAKAGE
E3_NEGATIVE
```

First cheap test:

```text
Build a controlled candidate-ranking domain with paraphrase, distractor, counterfactual, and verifier slices. Generate teacher measurements over local neighborhoods. Compile lesson packets. Train a small student in its own gauge. Open hidden transformations once. Assign one terminal token.
```

E2 code reuse:

- teacher registry and cache discipline;
- per-teacher source identity;
- shuffled-target controls;
- ablation config validation;
- gradient conflict diagnostics;
- route telemetry as a feature, not as a sufficient training target.

### Narrative Attack

Obvious dismissal: "This is just a benchmark for data augmentation."

Trivial dismissal: "A hand-written lesson schema smuggles the answer."

Mission test: The lesson schema must not provide the solution grammar. It must only define the audit envelope. The content must be inferred from teacher measurements and charged in the cost ledger.

Unkillable version: the lesson compiler forecast says which packet should work, the packet works on hidden transformations, shuffled/corrupted packets fail, and ordinary absorbers cannot match it at comparable all-in cost.

### Iteration Verdict

E3 is the clean redirect. It makes the abstraction falsifiable and turns E2 from mainline into reusable instrumentation.

## I455: Kill-Or-Redirect Decision

### Previous Conclusion Under Attack

I454 proposes E3 as the redirect.

### Attack

Maybe even E3 is not the highest-value move. The mission is cheap useful AI, and three alternatives compete:

1. **E3 teacher tomography**: highest novelty, medium uncertainty.
2. **Semantic addressability codec / Option G**: medium-high novelty, closer to current architecture and evidence.
3. **CBD-like chain-init / byteified compression**: lower novelty, highest chance of a useful small model quickly.

If the goal is "paradigm shift or failure," E3 is best. If the goal is "ship a cheap useful small model," chain-init may be best. If the goal is "prove Intelligence = Geometry in the current Sutra substrate," the semantic codec is the bridge.

The adversarial decision must not pretend these are the same goal.

### Final Decision

Do **not** push E2 as designed.

Radically reframe Eklavya around E3. Keep E2 as:

- an absorber baseline for "better KD";
- a source of teacher measurement infrastructure;
- a way to build negative evidence if byte-KL still fails.

The three strongest next directions are below.

## Direction 1: E3 Functional Teacher Tomography

Home-run story:

```text
Small models can become useful not by copying teacher outputs, but by learning compact operational lessons inferred from how diverse teachers disagree under intervention.
```

Why it could be paradigm-shifting:

- attacks the proxy/function failure pattern directly;
- makes teacher disagreement a discovery instrument;
- yields inspectable lesson packets, not private weight magic;
- supports surgical improvement and community-contributed lessons;
- directly instantiates "Intelligence = Geometry" as distinctions, transformations, rankings, and verifier obligations.

Kill gate:

```text
If E3 cannot beat CE-only, active learning, best single teacher, naive average, raw KD, shuffled packets, and domain-tool absorbers on hidden transformations at matched all-in cost, kill it.
```

## Direction 2: Semantic Addressability Codec Plus Functional Margins

Home-run story:

```text
The byte interface is not the thinking substrate. A small codec learns to turn raw bytes into semantic addresses where a compact reasoner can retrieve and use knowledge efficiently.
```

Why it could be paradigm-shifting:

- keeps byte-native I/O while attacking the byte-to-semantics bottleneck;
- builds a coordinate system for small models rather than imitating logits;
- bridges Eklavya teacher measurements with Sutra architecture;
- could compound with E3 lesson packets.

Risks:

- token-identity retrieval masquerading as semantics;
- fixed-permutation shuffled control is insufficient;
- patch-boundary supervision mismatch;
- still may top out below mission-grade usefulness without chain-init, retrieval, or larger anchors.

Kill gate:

```text
Per-occurrence random controls, frozen random-codec controls, token-frequency slices, matched Wide7 baseline, and Phase-2 functional gains must all clear. If the codec gives no control-adjusted functional lift, kill it as another proxy.
```

## Direction 3: Chain-Init / Retrieval As Ruthless Absorber And Utility Path

Home-run story:

```text
If ordinary compression or retrieval gives cheap useful intelligence faster, use it as the baseline that Eklavya must beat or join.
```

Why it matters:

- CBD is the strongest ordinary explanation for strong 121M-138M performance;
- retrieval may be the honest route to democratized usefulness when world knowledge cannot fit cheaply in weights;
- any Eklavya claim that cannot beat these at all-in cost is not a moonshot, it is vanity.

Kill gate:

```text
If CBD-like chain-init or retrieval reaches the same utility cheaper and with less fragility, Eklavya's current neural-transfer variant is absorbed. The surviving Eklavya contribution would need to be lesson discovery or local improvability, not benchmark score.
```

### Final Narrative Attack

Obvious dismissal:

```text
Eklavya is just KD with more controls.
```

Answer:

```text
Only if we keep E2 as the claim. E3 changes the object from output imitation to functional lesson discovery.
```

Trivial dismissal:

```text
E3 is active learning plus data augmentation.
```

Answer:

```text
Then active learning should absorb it under the ladder. The burden is on E3 to show teacher-ecology structure that active learning does not capture.
```

Mission test:

```text
Does this make intelligence cheaper, more accessible, and more improvable?
```

Answer:

```text
E2 alone probably does not. E3 might, because inspectable lessons, verifier obligations, and source-specific retained gain are closer to public infrastructure than one private distilled checkpoint.
```

What the result must be to become unkillable:

```text
Given a small failing student, Eklavya identifies a compact set of teacher-derived operational lessons; those lessons predictably repair function on hidden transformations; the repair survives teacher removal; ordinary absorbers fail at matched all-in cost; and the lesson packets are inspectable enough for outsiders to extend.
```

## Bottom Line

Eklavya should continue, but not as E2.

The live bet is:

```text
Teacher disagreement is not a routing problem.
Teacher disagreement is a measurement apparatus for discovering the geometry of useful distinctions.
```

If the next Eklavya arc cannot make that concrete, kill it. If it can, the project has a real moonshot again.
