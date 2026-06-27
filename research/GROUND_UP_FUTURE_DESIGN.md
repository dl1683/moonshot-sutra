# Sutra And Eklavya Ground-Up Future Design

> **Canonical reconciliation**: This document has been reconciled with
> `research/EKLAVYA_DOCTRINE.md` through R1-R6 adversarial deliberation.
> The unified build spec lives in **`research/SE1_CANONICAL_SPEC.md`**. For any
> conflict between this document and the canonical spec, the canonical spec wins.

This document deliberately ignores inherited design commitments.

It asks one question:

```text
If Sutra and Eklavya were designed from scratch to create the world's most
efficient generally useful AI system, what should they be?
```

The answer is not "a smaller transformer trained by many teachers." That is too
weak. It inherits the shape of current models and then tries to compress them.

The ground-up answer is:

```text
Eklavya is a protocol for extracting causal lessons from many competent systems.
Sutra is a runtime learner that turns those lessons into a compact, adaptive,
teacher-free cognition engine.
```

Eklavya is not distillation. Sutra is not a student copy. The unit of transfer is
not an output distribution, a hidden state, a trace, or a label. The unit of
transfer is a **lesson**:

```text
lesson = minimal intervention that changes the learner's future behavior
         in the intended way
         under teacher removal
         with bounded cost and bounded collateral damage
```

Everything follows from that.

## What I Would Have Told You

If I were designing this from zero, I would have told you this:

```text
Do not build a model first.
Build a learner whose failures are observable.

Do not collect teachers first.
Build a protocol that admits a teacher only when the student has a named gap.

Do not distill outputs.
Interrogate teachers until they reveal the smallest lesson that changes the
student's future behavior.

Do not make Sutra a checkpoint.
Make Sutra a compact runtime with modules, memory, verification, adaptive
compute, and local update ports.

Do not claim Eklavya because many teachers were used.
Claim Eklavya only when the student retains a lesson after teachers are removed.

Do not chase benchmark score first.
Measure owned gain per total lifecycle cost.
```

The first real system should therefore be a loop, not a run:

```text
observe_student_failure
  -> choose_teacher_instrument
  -> interrogate_with_controlled_probes
  -> compile_minimal_lesson_packet
  -> update_the_smallest_adequate_sutra_module
  -> remove_teacher
  -> measure_owned_gain_and_damage
  -> keep_or_delete_lesson
```

That loop is the product. The model checkpoint is only one state of the product.

## The North Star

Sutra should be efficient in the only definition that matters:

```text
useful_capability_over_lifetime_cost
```

Lifetime cost includes:

- training compute;
- inference compute;
- teacher interrogation cost;
- data construction cost;
- memory and retrieval cost;
- update cost;
- validation cost;
- repair cost;
- human supervision cost;
- deployment complexity;
- risk from opaque behavior.

The target is not maximum benchmark score per parameter. The target is an AI
system that can keep improving, specialize cheaply, run cheaply, be repaired
locally, and retain competence without permanent dependence on larger systems.

## First-Principles Theory

### 1. Intelligence Is Not Weight Count

Large models are not intelligent because they have many parameters. They are
intelligent because their parameters encode reusable structure about the world,
language, tasks, and interaction.

The question is therefore:

```text
What is the cheapest way to acquire, store, invoke, update, and verify reusable
structure?
```

Raw scale is one answer. It is not the only answer, and it is probably not the
best answer for Sutra.

### 2. Teachers Are Instruments

A teacher is not a master to imitate. A teacher is an instrument that reveals
some structure the learner cannot yet see cheaply.

Different instruments reveal different things:

- a large decoder reveals candidate behavior and linguistic priors;
- an encoder reveals semantic neighborhoods;
- a solver reveals constraint structure;
- a verifier reveals failure boundaries;
- a search process reveals decision paths;
- a simulator reveals transition dynamics;
- a human curriculum reveals what to study next;
- a memory system reveals retrieval keys and durable facts;
- an adversary reveals brittleness.

If two teachers reveal the same structure with the same errors, they are one
teacher for Eklavya purposes. If a small exact checker reveals the target
structure better than a giant model, the checker is the superior teacher.

### 3. The Student Must Have Places For Lessons To Land

Most distillation fails because it asks a student to absorb teacher behavior
without giving the student a structural place to store the lesson.

Sutra should not be a homogeneous block. It should be a set of specialized,
cheaply callable organs:

```text
Sutra:
  perception_surface:
    turns raw bytes, pixels, tables, traces, and tool outputs into compact
    typed states

  world_state:
    stores compressed task-relevant latent state

  program_core:
    performs reasoning, planning, transformation, and abstraction

  verifier_core:
    checks constraints, uncertainty, and local correctness

  memory_core:
    reads and writes durable externalized knowledge

  action_surface:
    emits text, code, tool calls, structured outputs, and byte-exact spans

  compute_governor:
    decides how much computation each case deserves

  update_ports:
    allow localized change without full-model retraining
```

A teacher lesson is admissible only if it names its landing site.

### 4. Efficiency Comes From Routing Structure, Not Only Tokens

The dominant waste in current models is not merely that they use too many tokens.
It is that every case flows through nearly the same expensive computation path.

Sutra must make computation conditional along several axes:

- easy versus hard examples;
- local versus global dependencies;
- exact surface reproduction versus semantic reasoning;
- remembered fact versus novel inference;
- verified requirement versus open-ended generation;
- stable skill versus recently updated skill;
- high-risk output versus low-risk output.

The runtime should behave like a cost-based planner:

```text
for each state:
  estimate uncertainty
  estimate value of more computation
  estimate cost of each available action
  choose the cheapest action that preserves required quality
```

This makes adaptive compute a core architectural principle, not a later
inference optimization.

## What Sutra Is

Sutra is a **compiled adaptive cognition runtime**.

It has three planes:

```text
Sutra_runtime:
  substrate_plane:
    compact neural machinery that transforms inputs into internal states and
    outputs

  control_plane:
    decides which machinery runs, how deep it runs, when memory is consulted,
    when verification is needed, and when to stop

  update_plane:
    admits new lessons, tests whether they were retained, localizes the update,
    and rolls back damaging changes
```

Current models mostly emphasize the substrate plane. Sutra's advantage should
come from making the control and update planes first-class.

### Sutra's Core Modules

```text
Sutra_core_v1:
  byte_and_symbol_surface:
    purpose:
      - preserve exact input/output identity
      - handle names, code, paths, numbers, formatting, schema details
      - prevent tokenizer artifacts from becoming semantic truth
    implementation_direction:
      - byte-preserving local encoder and decoder
      - learned patches or typed spans for global computation
      - residual channel for exact reconstruction

  semantic_core:
    purpose:
      - maintain meaning across paraphrase, abstraction, and composition
      - hold reusable task states
      - form compact representations independent of surface form
    implementation_direction:
      - compact attention backbone
      - optional recurrent or state-space lane for cheap long context
      - nested representations usable at multiple compute budgets

  program_core:
    purpose:
      - execute multi-step reasoning, transformation, and planning
      - represent latent programs rather than only next-token flow
    implementation_direction:
      - small recurrent scratch state
      - candidate plan heads
      - step verifier hooks
      - tool/action schema awareness

  verifier_core:
    purpose:
      - detect local errors before they become outputs
      - score constraints, schemas, arithmetic, citations, and task contracts
      - decide whether more computation is worth buying
    implementation_direction:
      - cheap local verifier heads
      - exact-oracle interfaces where available
      - disagreement and uncertainty sensors

  memory_core:
    purpose:
      - store externalized lessons, facts, retrieval keys, and update history
      - avoid forcing every fact into weights
    implementation_direction:
      - learned retrieval keys
      - typed memories with provenance
      - write gates and decay policies

  compute_governor:
    purpose:
      - select depth, width, verification, memory, and decoding strategy
      - optimize expected quality minus expected cost
    implementation_direction:
      - action policy trained from teacher disagreement, verifier feedback,
        uncertainty, and post-hoc cost accounting

  update_ports:
    purpose:
      - accept localized lessons from Eklavya
      - change target behavior without damaging unrelated behavior
    implementation_direction:
      - constrained adapters
      - module-specific packet losses
      - rollback and retained-gain checks
```

### Sutra's Inference Loop

Sutra should not always run "the model" once. It should run a small decision
process:

```text
Sutra_inference_loop:
  1. parse input into typed spans and compact state
  2. estimate task class, risk, uncertainty, memory need, and exactness need
  3. run cheap path if confidence and risk allow
  4. escalate selected spans or states through deeper computation when needed
  5. verify constraints locally or with exact tools when available
  6. repair only the failing region instead of regenerating everything
  7. emit answer with cost and uncertainty trace available internally
```

This is how Sutra becomes more efficient than a monolithic model: not by being
weaker, but by refusing to spend uniform compute.

## What Eklavya Is

Eklavya is a **lesson discovery and compilation protocol**.

It has five jobs:

```text
Eklavya_protocol:
  interrogate_teachers:
    ask controlled questions that expose what each teacher knows and where it
    fails

  infer_lessons:
    convert teacher behavior into causal claims about what the student should
    learn

  compile_packets:
    turn lessons into small, typed training/update objects with landing zones

  teach_student:
    apply packets through the cheapest update path that can retain the lesson

  prove_ownership:
    remove teachers and test whether Sutra now owns the competence
```

Eklavya's job is not to train forever. Its job is to decide what is worth
teaching and to verify that teaching worked.

## The Core Eklavya Object: A Lesson Packet

Every Eklavya object should reduce to a lesson packet:

```text
lesson_packet:
  lesson_id:
  target_behavior:
  target_invariant:
  teacher_sources:
  probes_that_exposed_it:
  student_gap_evidence:
  landing_zone:
  minimal_training_object:
  expected_retained_gain:
  expected_cost:
  corruption_risks:
  collateral_damage_tests:
  teacher_removal_test:
  rollback_plan:
```

Examples:

```text
decoder_margin_lesson:
  target_invariant: choose robust answer under paraphrase and distractor probes
  landing_zone: semantic_core + program_core
  minimal_training_object: candidate margins across controlled probes

verifier_span_lesson:
  target_invariant: localize the violated constraint before answering
  landing_zone: verifier_core + compute_governor
  minimal_training_object: error span, repair class, failing constraint

boundary_lesson:
  target_invariant: preserve byte-level distinctions that tokens erase
  landing_zone: byte_and_symbol_surface
  minimal_training_object: span boundary, residual flag, exact reconstruction

retrieval_lesson:
  target_invariant: map question state to the correct memory key
  landing_zone: memory_core
  minimal_training_object: query, positive memory, hard negatives

compute_lesson:
  target_invariant: know when cheap reasoning is enough and when to escalate
  landing_zone: compute_governor
  minimal_training_object: uncertainty state, chosen action, cost, outcome
```

This replaces distillation records. A logit vector is allowed only if it is part
of a lesson packet with a named target behavior and removal test.

## Teacher Interrogation

Eklavya should not ask teachers for generic answers. It should interrogate them
as instruments.

For each target skill, construct probe families:

```text
probe_families:
  invariance_probe:
    changes surface while preserving correct behavior

  counterfactual_probe:
    changes the hidden condition that should change behavior

  distractor_probe:
    adds irrelevant but tempting evidence

  corruption_probe:
    damages formatting, spelling, ordering, or schema

  decomposition_probe:
    asks for intermediate structure, not final answer

  verifier_probe:
    asks what would make the answer invalid

  compression_probe:
    removes words and tests whether the teacher preserves meaning

  adversarial_probe:
    targets known student failure modes
```

Teacher outputs are not the lesson. The response surface is evidence for a
lesson.

```text
response_surface:
  what stays stable
  what changes
  when confidence changes
  which distractors matter
  where teachers disagree
  which teacher is right under exact checks
  which student module would need to change
```

Eklavya should prefer probes that change decisions, not probes that merely
produce interesting disagreement.

## How Eklavya Avoids Distillation

Eklavya must reject five common traps:

1. **Output imitation.** If the student learns teacher style without the target
   invariant, the lesson failed.
2. **Teacher averaging.** If teacher identity disappears before the lesson is
   understood, the protocol cannot know what worked.
3. **Permanent teacher dependence.** If inference needs teacher calls, the
   student did not own the competence.
4. **Benchmark leakage.** If the lesson only improves the probe set, the lesson
   is not general enough.
5. **Capacity mismatch.** If the student lacks a landing zone, stronger teacher
   signal becomes optimization noise.

The positive rule:

```text
teach only what the student can store, invoke, and retain cheaply
```

## Training Protocol

The ground-up protocol is not pretrain then distill. It is cyclical:

```text
Eklavya_Sutra_training_cycle:
  0. initialize Sutra with modular landing zones
  1. train base substrate on clean, compact, diverse data
  2. map current student failures by module and task type
  3. select teachers as instruments for those failures
  4. interrogate teachers with controlled probes
  5. compile lesson packets
  6. apply lesson packets through localized update ports
  7. remove teachers and artifacts from inference
  8. test retained gain, collateral damage, and cost
  9. keep, revise, or delete the lesson
  10. repeat on the next most valuable gap
```

This cycle makes Eklavya an active teaching protocol rather than a one-time data
preparation step.

### Phase 0: Build The Learner Before The Teachers

Before teacher lessons matter, Sutra needs a body that can absorb them:

```text
phase_0_student_body:
  objectives:
    - byte-exact local surface
    - compact semantic state
    - basic generation and instruction behavior
    - cheap verifier hooks
    - memory key formation
    - compute governor observability
  forbidden:
    - teacher packet training
    - permanent external teacher dependence
    - untyped hidden compute
```

The point is not to maximize baseline scores. The point is to create a learner
with measurable gaps and real landing zones.

### Phase 1: Discover Student Gaps

Eklavya begins with the student, not the teacher.

```text
student_gap_map:
  gap_id:
  task_slice:
  failing_module_hypothesis:
  failure_examples:
  exact_or_human_judgment:
  expected_value_if_fixed:
  candidate_teacher_instruments:
```

No teacher is admitted until a student gap exists.

### Phase 2: Interrogate Teachers

Teachers are selected by gap, not prestige.

```text
teacher_admission_rule:
  admit_teacher_if:
    - it measures a hidden state relevant to the gap
    - it differs from existing teachers
    - it can be queried cheaply enough
    - it can produce a typed lesson
    - the lesson can land in a Sutra module
```

### Phase 3: Compile Lessons

Raw teacher artifacts are compressed into lesson packets.

```text
lesson_compilation:
  input:
    - probes
    - teacher response surfaces
    - student failures
    - exact checks or human adjudication
  output:
    - minimal lesson packet
    - target module
    - expected gain and cost
    - corruption tests
```

Compression is mandatory. If a lesson requires carrying the whole teacher
surface forever, it is probably not a lesson.

### Phase 4: Local Teaching

Apply the lesson to the smallest adequate part of Sutra:

```text
local_teaching_policy:
  prefer:
    - verifier head update over global model update
    - memory-key update over weight update
    - adapter update over full finetune
    - targeted curriculum over full corpus replay
    - compute-governor policy update over deeper always-on compute
  reject:
    - global update for local failure
    - loss term with no landing zone
    - improvement that breaks unrelated slices
```

### Phase 5: Ownership Test

A lesson counts only if it survives teacher removal:

```text
ownership_test:
  before_score: student before lesson
  during_score: student with teacher or artifact available during teaching
  after_score: student after teacher/artifact removal
  retained_gain: after_score - before_score
  dependence_gap: during_score - after_score
  collateral_damage: unrelated_score_before - unrelated_score_after
  utility_vector:
    - retained_gain
    - dependence_gap
    - collateral_damage
    - training_cost
    - inference_cost_delta
    - validation_cost
  scalar_decision_value:
    - computed_only_after_units_and_weights_are_predeclared
```

The lesson is kept only when its predeclared utility vector passes every hard
floor and its scalar decision value is positive under the predeclared utility
scale. Raw score deltas and costs must not be subtracted directly.

## The Exact Function Of Each System

### Eklavya's Function

Eklavya is the teacher economy.

It decides:

- what the student lacks;
- which teacher can reveal the missing structure;
- which probes expose that structure;
- what minimal packet should be taught;
- where the packet lands;
- whether the student retained the lesson;
- whether the lesson was worth its total cost.

Eklavya ends when a lesson is owned.

### Sutra's Function

Sutra is the owned runtime.

It decides:

- what representation to use;
- how much computation to spend;
- whether to consult memory;
- whether to verify;
- how to repair local failures;
- how to update without global damage;
- when to refuse cheap confidence and escalate.

Sutra should not need teachers at inference. It may use tools, memory, and exact
checkers only when they are admitted as versioned runtime capabilities rather
than hidden teaching artifacts. The teacher/runtime boundary is formalized below.

## The World-Class Efficiency Thesis

Sutra can become world-class efficient only by compounding six efficiencies:

```text
efficiency_stack:
  representational_efficiency:
    preserve exact surfaces locally and reason over compact global states

  teaching_efficiency:
    use teachers only to reveal high-value lessons

  compute_efficiency:
    run cheap paths for easy cases and deep paths only when needed

  update_efficiency:
    repair or extend one module without full retraining

  validation_efficiency:
    use targeted retained-gain tests instead of broad benchmark chasing

  memory_efficiency:
    store facts, traces, and lessons externally when weights are the wrong place
```

If any one of these is missing, Sutra may still be useful, but the "world's most
efficient" claim becomes fragile.

## What I Would Build First

The first real build should be small enough to fail cheaply and rich enough to
test the full theory.

```text
Sutra_Eklavya_G1:
  student_size:
    small_enough_for_fast_iteration
    large_enough_for modular_landing_zones

  substrate:
    byte-preserving local surface
    patch or span global semantic core
    shallow/deep compute paths
    verifier head
    memory key head
    update adapters

  first_teacher_instruments:
    decoder_teacher:
      use: candidate behavior surfaces and reasoning contrast
    encoder_teacher:
      use: semantic neighborhood and retrieval key structure
    exact_oracle_teacher:
      use: arithmetic, schema, code, or formal constraint truth
    curriculum_teacher:
      use: minimal contrast sets and learning order
    adversary_teacher:
      use: discover brittle regions and false confidence

  first_lessons:
    byte_exactness
    candidate_margin_under_probe
    verifier_error_localization
    retrieval_key_stability
    compute_escalation_policy

  first_success:
    not high benchmark score
    but positive net retained gain across multiple lesson types
    with teacher removal and bounded cost
```

This is the smallest complete system: it tests Sutra's body, Eklavya's lesson
protocol, and the ownership claim.

## The Build Sequence I Would Follow

```text
build_sequence:
  stage_1_observable_student:
    build the modular student with instrumentation first
    output: per-module failure map and cost map

  stage_2_gap_dataset:
    create small typed slices that expose exact known failure classes
    output: gap map, not benchmark leaderboard

  stage_3_teacher_interrogator:
    implement probe generation and teacher response-surface capture
    output: teacher surfaces with source identity

  stage_4_lesson_compiler:
    convert surfaces into lesson packets
    output: typed packets with landing zones and ownership tests

  stage_5_local_update_engine:
    apply packets through adapters, heads, memory, or curriculum
    output: localized updates with rollback

  stage_6_ownership_evaluator:
    remove teachers and score retained gain, dependence, damage, and cost
    output: lesson ledger

  stage_7_compute_governor:
    train runtime planning from uncertainty, verifier signals, and cost traces
    output: cheaper inference with hard-case floors

  stage_8_lesson_market:
    prioritize the next lessons by expected net value
    output: continuous improvement loop
```

This sequence is different from ordinary model development because it makes
learning units auditable before scale.

## The Central Metrics

Do not start with benchmark averages. Start with ownership economics:

```text
owned_gain:
  score_after_teacher_removal - score_before_teacher_lesson

teacher_dependence_gap:
  score_with_teacher_or_artifact - score_after_teacher_removal

lesson_decision_value:
  computed_from_predeclared_normalized_utility_terms
  after_hard_floors_pass

adaptive_compute_value:
  quality_static_cost_matched - quality_adaptive
  and
  cost_static_quality_matched - cost_adaptive

update_locality:
  target_gain / unrelated_damage

memory_weight_tradeoff:
  value_of_externalized_memory - value_of_weight_update_for_same_fact_or_skill
```

A system that improves benchmark score while failing these metrics is not Sutra.

## What The Architecture Must Refuse

Sutra/Eklavya should explicitly refuse:

- full-vocabulary distillation as the default;
- chain-of-thought copying as proof of reasoning transfer;
- teacher ensembles at inference;
- adding teachers without a student gap;
- adding modules without a landing-zone lesson;
- hidden local compute that makes global compression look free;
- benchmark claims without teacher removal;
- global finetunes for local failures;
- memory writes without provenance;
- adaptive compute claims without hard-case reporting.

Refusal is part of the design. It keeps the system from becoming another vague
small-model training recipe.

## The Future Form

If this works, Sutra is not one model checkpoint. It is a living, compact AI
runtime:

```text
future_Sutra:
  base_engine:
    compact modular neural substrate

  lesson_ledger:
    accepted lessons with evidence, cost, and rollback path

  adaptive_runtime:
    cost-based compute and verification planner

  memory_system:
    typed external knowledge with provenance and decay

  update_system:
    local lesson admission, rollback, and retained-gain evaluation

  deployment_profile:
    runs at multiple cost tiers by changing compute policy, not identity
```

Eklavya becomes the protocol that keeps this runtime improving without turning
it into a dependent shadow of larger models.

## Research Refresh For The Future Design

The ground-up doctrine should absorb current research only as design pressure,
not as a binding template.

Primary sources that matter:

```text
source_refresh_2026_06_23:
  byte_patch_models:
    BLT:
      source: https://arxiv.org/abs/2412.09871
      useful_pressure:
        - raw-byte modeling can be made competitive when bytes are grouped into
          dynamic patches
        - entropy can choose where representation should become finer
        - robustness and inference efficiency can improve when tokenization is
          not fixed
      design_consequence:
        - Sutra should treat bytes as truth and patches as compute units

    MEGABYTE:
      source: https://arxiv.org/abs/2305.07185
      useful_pressure:
        - local-within-patch and global-between-patch modeling is a real
          architectural family
        - long byte streams can be handled by multiscale decomposition
      design_consequence:
        - Sutra should make local/global decomposition a first-class substrate

    Charformer:
      source: https://arxiv.org/abs/2106.12672
      useful_pressure:
        - subword structure can be learned end-to-end rather than inherited
        - character/byte systems can be fast if block formation is learned
      design_consequence:
        - Sutra should not make a frozen tokenizer a constitutional dependency

  adaptive_compute:
    LayerSkip_self_speculative:
      source: https://arxiv.org/html/2404.16710v1
      useful_pressure:
        - early layers can draft and later layers can verify inside one model
        - self-speculation avoids permanent dependence on a second model
      design_consequence:
        - Sutra should use shallow/deep internal paths before external teacher
          dependence

    test_time_compute_scaling:
      source: https://arxiv.org/abs/2408.03314
      useful_pressure:
        - inference-time compute can improve hard-prompt performance
        - the training/inference compute tradeoff is itself an optimization
      design_consequence:
        - Sutra's compute governor must price extra thinking case-by-case

    sleep_time_compute:
      source: https://arxiv.org/html/2504.13171v1
      useful_pressure:
        - useful computation can happen before a query when context is known
      design_consequence:
        - Sutra should have a precompute mode for memory, profiles, and likely
          future queries rather than forcing all cognition into live inference

  small_model_future:
    slm_agentic_ai:
      source: https://arxiv.org/pdf/2506.02153
      useful_pressure:
        - small models can be more appropriate when modularity, latency, cost,
          privacy, and specialization matter
      design_consequence:
        - Sutra should be designed as a small capable engine inside a system,
          not as a shrunken monolith competing only by benchmark average

  teaching_and_updates:
    machine_teaching:
      source: https://arxiv.org/pdf/1801.05927
      useful_pressure:
        - the teacher can optimize the lesson for a known learner and target
        - lesson construction is an inverse problem to ordinary learning
      design_consequence:
        - Eklavya should search for minimal lessons given Sutra's actual
          learning dynamics, not collect teacher outputs generically

    sequential_machine_teaching:
      source: https://arxiv.org/abs/1810.06175
      useful_pressure:
        - teaching can be posed as a shortest-sequence or control problem
      design_consequence:
        - Eklavya should treat lesson ordering as part of the protocol, not as
          a dataloader detail

    LoRA:
      source: https://arxiv.org/abs/2106.09685
      useful_pressure:
        - localized low-rank deltas can adapt large models with far fewer
          trainable parameters than full finetuning
      design_consequence:
        - Sutra update ports should start with constrained deltas and only
          escalate to broader training when local deltas fail ownership tests

  control_and_information:
    active_inference:
      source: https://pmc.ncbi.nlm.nih.gov/articles/PMC13059084/
      useful_pressure:
        - action and perception can be understood as uncertainty-reducing
          control under a model of the world
      design_consequence:
        - Sutra's compute governor should select actions that reduce task
          uncertainty per unit cost, not just maximize raw answer probability

    know_when_enough:
      source: https://arxiv.org/html/2510.08517v1
      useful_pressure:
        - agents can waste compute by failing to terminate when they have enough
          information
      design_consequence:
        - Eklavya should teach stopping, retrieval, and escalation as explicit
          lessons rather than hoping they emerge
```

The common lesson is this:

```text
future_efficiency = compact_representation
                  + conditional_compute
                  + modular_specialization
                  + owned_lessons
                  + externalized_memory_when_weights_are_wrong
```

No single paper gives Sutra. They point toward a system shape.

## Formal Ontology

The design needs a small vocabulary that can survive implementation.

```text
world:
  the external task environment, data distribution, tool ecosystem, and user
  objective space

learner:
  Sutra at a specific revision, including weights, memory, adapters, runtime
  policies, and validators

teacher:
  any source that can reduce uncertainty about a learner gap

probe:
  controlled intervention used to reveal teacher behavior, student behavior, or
  hidden structure

gap:
  measured difference between current learner behavior and desired behavior,
  assigned to a suspected module, skill, or policy

lesson:
  minimal intervention expected to close a gap with bounded cost

packet:
  portable representation of a lesson with provenance, landing zone, tests, and
  rollback path

ownership:
  retained behavior after teacher and temporary teaching artifacts are removed

runtime_policy:
  decision rule for compute, memory, verification, tool use, and output repair

revision:
  versioned Sutra state after accepting or rejecting lessons
```

The central relation:

```text
teacher + probe + student_gap -> lesson_hypothesis
lesson_hypothesis + landing_zone -> packet
packet + local_update -> revised_sutra
revised_sutra - teacher -> ownership_test
ownership_test -> accept_or_reject_lesson
```

This relation is the protocol spine.

## Lesson Calculus

Eklavya should rank lessons before creating them and audit them after training.

Before teaching:

```text
expected_lesson_value =
  scalar_decision_value_from_utility_scale_v1_priors
  after_predeclared_expected_hard_floors

expected_utility_terms:
  gap_reality_probability:
  teacher_causal_relevance_probability:
  landing_probability:
  retention_probability:
  expected_retained_gain_u:
  expected_dependence_gap_u:
  expected_collateral_damage_u:
  expected_training_cost_u:
  expected_inference_cost_u:
  expected_validation_cost_u:
```

After teaching:

```text
realized_lesson_value =
  scalar_decision_value_from_utility_scale_v1
  after_all_hard_floors_pass
```

Forecast error matters:

```text
lesson_forecast_error = normalized_expected_lesson_value - realized_lesson_value
```

Large forecast error means Eklavya does not understand its own teacher economy.
That should update teacher admission and probe selection, not be hidden in an
aggregate score.

## Sutra State Model

Sutra should be treated as a stateful runtime, not as weights alone.

```text
Sutra_state:
  weights:
    base neural substrate

  adapters:
    localized learned deltas accepted through update ports

  memory:
    external typed facts, patterns, examples, traces, and lesson summaries

  policies:
    compute, verification, memory-read, memory-write, and repair policies

  validators:
    exact or learned checks used during inference and update

  lesson_ledger:
    accepted, rejected, rolled-back, and pending lesson packets

  profile_cache:
    runtime cost, uncertainty, and failure statistics used by the compute
    governor
```

A Sutra revision is only valid if all these pieces are versioned together:

```text
Sutra_revision_id = hash(weights, adapters, memory_schema, policies, validators,
                         accepted_lesson_ledger)
```

This prevents false claims where the "model" improves only because an
unversioned memory, prompt, tool, or validator changed.

The revision identity must include every runtime dependency that can affect a
trace:

```text
Sutra_revision_manifest_v2:
  required_hash_fields:
    - weights
    - adapters
    - memory_schema
    - memory_contents_or_snapshot_ref
    - runtime_policies
    - validators_code_and_config
    - validator_runtime_versions
    - tools_code_and_config
    - tool_runtime_versions
    - accepted_lesson_ledger
    - rejected_and_rolled_back_lesson_refs
    - teacher_registry_snapshot
    - eval_slice_versions
    - profile_cache_or_cache_invalidation_policy
    - prompt_and_probe_templates
    - random_seeds_or_nondeterminism_policy
```

If a dependency can change the output, cost, trace, lesson decision, or
validator result, it is part of the revision. Otherwise H8 reproducibility is a
fiction.

## Teacher, Tool, Oracle, And Runtime Boundary

The phrase "teacher removal" is meaningless unless the boundary is exact.

```text
source_role_boundary_v1:
  teacher:
    definition:
      - source used to produce teaching signal, lesson packet, probe response,
        target margin, example, trace, or label before ownership evaluation
    inference_status:
      - forbidden_at_inference_for_claimed_owned_lesson

  runtime_tool:
    definition:
      - capability intentionally available to Sutra during deployment for the
        task class being evaluated
    requirements:
      - versioned in Sutra_revision_manifest
      - costed in efficiency report
      - available to all compared controls unless the claim is tool-specific

  exact_oracle_teacher:
    definition:
      - validator or solver used to teach, score probes, label packets, or
        construct lessons
    inference_status:
      - removed during ownership evaluation unless explicitly admitted as a
        runtime_tool

  exact_runtime_validator:
    definition:
      - checker supplied by the task contract or deployment environment, such as
        a JSON schema validator for a schema-constrained output
    inference_status:
      - allowed only if included in revision manifest and charged as runtime
        cost

  hidden_teacher_dependency:
    definition:
      - any teacher-derived artifact, validator label, teacher cache, privileged
        trace, or prompt-side answer key used at inference without being
        declared as runtime_tool
    status:
      - invalidates ownership claim
```

Rule:

```text
teacher_removal_rule:
  for each accepted lesson:
    list every source used during teaching
    classify each as teacher, runtime_tool, or both
    remove all teacher-only sources during ownership evaluation
    keep only runtime tools that are versioned, costed, and control-matched
```

This closes the oracle loophole: exact validators are excellent teachers, but
they are not proof of owned competence unless Sutra either internalizes the
behavior or the validator is explicitly part of the runtime claim.

## Sutra Runtime Architecture V2

The first fully serious Sutra should have seven typed interfaces.

```text
Sutra_runtime_interfaces_v2:
  I0_input_surface:
    accepts:
      - raw bytes
      - structured records
      - tool outputs
      - retrieved memories
    emits:
      - typed spans
      - byte residual map
      - surface risk flags
      - initial uncertainty

  I1_compact_state:
    accepts:
      - typed spans
      - local byte states
    emits:
      - semantic state
      - task state
      - memory query state
      - verifier query state

  I2_reasoning_step:
    accepts:
      - compact state
      - scratch state
      - optional retrieved memory
    emits:
      - candidate next state
      - candidate action
      - uncertainty delta
      - trace summary

  I3_verification:
    accepts:
      - candidate answer
      - candidate action
      - constraints
      - exact oracle results when available
    emits:
      - pass_fail_or_unknown
      - error span
      - repair class
      - escalation recommendation

  I4_memory:
    accepts:
      - query state
      - write candidate
      - provenance
    emits:
      - retrieved records
      - confidence
      - conflict set
      - write decision

  I5_compute_governor:
    accepts:
      - uncertainty
      - risk
      - task class
      - verification state
      - cost profile
    emits:
      - stop
      - run_deeper
      - run_wider
      - retrieve
      - verify
      - repair
      - precompute

  I6_output_surface:
    accepts:
      - final compact state
      - byte residual map
      - verification state
    emits:
      - text
      - structured output
      - tool call
      - byte-exact span
```

Each interface has its own lesson types. This is how Sutra avoids becoming a
single undifferentiated parameter mass.

## Sutra Compute Modes

Sutra should not have one inference mode.

```text
Sutra_compute_modes:
  reflex_mode:
    use_when:
      - low risk
      - high confidence
      - known pattern
    path:
      - shallow substrate
      - no retrieval unless cached
      - cheap verifier only

  deliberation_mode:
    use_when:
      - reasoning uncertainty
      - multi-step dependency
      - high-value answer
    path:
      - deeper semantic and program core
      - verifier loop
      - possible memory read

  exactness_mode:
    use_when:
      - code
      - math
      - paths
      - schemas
      - citations
      - byte-sensitive output
    path:
      - byte/symbol surface emphasized
      - exact validators preferred
      - repair before final output

  sleep_mode:
    use_when:
      - stable context exists before future queries
      - repeated domain or user profile is known
    path:
      - precompute likely memory keys
      - summarize context
      - build verifier constraints
      - cache task profiles

  update_mode:
    use_when:
      - Eklavya accepts a lesson packet
    path:
      - select update port
      - apply local update
      - run ownership and damage tests
      - commit or roll back
```

The compute governor decides among these modes. Efficiency comes from choosing
correctly, not from making every path cheap.

## Eklavya Protocol State Machine

Eklavya should be implemented as a state machine so claims cannot skip proof.

```text
Eklavya_lesson_state_machine:
  observed_gap:
    entry:
      - student failure exists
      - baseline behavior captured
      - suspected landing zone named
    exit_to: teacher_search

  teacher_search:
    entry:
      - gap has value estimate
      - candidate teachers listed as instruments
    exit_to: probe_design
    reject_if:
      - no teacher measures hidden state relevant to gap

  probe_design:
    entry:
      - probes distinguish invariance, counterfactual, distractor, and
        corruption cases where relevant
    exit_to: teacher_interrogation
    reject_if:
      - probe cannot distinguish knowledge from style

  teacher_interrogation:
    entry:
      - teacher surfaces captured with source identity
      - student surface captured on same probes
    exit_to: lesson_inference
    reject_if:
      - teacher surface is redundant or too costly

  lesson_inference:
    entry:
      - target invariant named
      - causal story for student gap proposed
    exit_to: packet_compilation
    reject_if:
      - no landing zone or no minimal training object

  packet_compilation:
    entry:
      - packet has tests, cost, corruption risks, and rollback
    exit_to: local_teaching
    reject_if:
      - packet is just raw teacher output without lesson structure

  local_teaching:
    entry:
      - smallest adequate update port selected
      - baseline frozen for comparison
    exit_to: ownership_eval
    reject_if:
      - update requires global retraining for local gap without justification

  ownership_eval:
    entry:
      - teacher removed
      - retained gain measured
      - dependence gap measured
      - collateral damage measured
      - cost measured
    exit_to: accepted_lesson_or_rejected_lesson

  accepted_lesson:
    entry:
      - realized lesson value positive
      - damage bounded
      - ledger updated

  rejected_lesson:
    entry:
      - lesson failed retention, cost, or damage test
      - reason recorded
      - teacher/probe priors updated
```

This is the anti-distillation mechanism. It forces every teacher signal to pass
through gap, lesson, landing, ownership, and value.

## Teacher Ecology

Ground-up Eklavya needs teacher diversity by **measurement function**, not by
brand or parameter count.

```text
teacher_ecology:
  behavior_teacher:
    measures:
      - candidate preferences
      - uncertainty
      - response stability
    examples:
      - decoder LMs

  geometry_teacher:
    measures:
      - semantic neighborhoods
      - paraphrase invariance
      - retrieval structure
    examples:
      - embedding models
      - contrastive encoders

  constraint_teacher:
    measures:
      - correctness under rules
      - invalid states
      - repair target
    examples:
      - symbolic solvers
      - type checkers
      - unit tests
      - schema validators

  process_teacher:
    measures:
      - useful intermediate states
      - planning path
      - search frontier
    examples:
      - theorem prover traces
      - program search
      - planner rollouts

  curriculum_teacher:
    measures:
      - example order
      - minimal contrast
      - prerequisite relation
    examples:
      - human-authored lessons
      - synthetic curricula

  adversarial_teacher:
    measures:
      - brittleness
      - false confidence
      - distribution edge
    examples:
      - red-team generators
      - perturbation systems
```

The first Eklavya roster should include at least one behavior teacher, one
geometry teacher, one constraint teacher, one curriculum teacher, and one
adversarial teacher. The actual model names matter less than whether the roster
covers these measurement functions.

## Lesson Types

The first complete Eklavya protocol should support exactly these lesson types:

```text
lesson_type_registry_v1:
  surface_fidelity_lesson:
    lands_in: I0_input_surface + I6_output_surface
    teaches:
      - exact byte identity
      - symbol preservation
      - formatting constraints

  semantic_invariance_lesson:
    lands_in: I1_compact_state
    teaches:
      - paraphrase stability
      - abstraction
      - hard negative separation

  reasoning_transition_lesson:
    lands_in: I2_reasoning_step
    teaches:
      - valid next state
      - decomposition
      - plan repair

  verification_lesson:
    lands_in: I3_verification
    teaches:
      - invalidity detection
      - constraint localization
      - repair class

  memory_lesson:
    lands_in: I4_memory
    teaches:
      - retrieval key
      - write boundary
      - conflict handling

  compute_policy_lesson:
    lands_in: I5_compute_governor
    teaches:
      - when to stop
      - when to verify
      - when to retrieve
      - when to precompute

  update_locality_lesson:
    lands_in: update_ports
    teaches:
      - smallest safe change
      - rollback condition
      - collateral damage guard
```

If a proposed lesson does not fit the registry, either the registry is missing a
real module or the lesson is not ready.

## The First Complete Build

The first complete build should not try to be the final world's best model. It
should prove the full loop in miniature.

```text
Sutra_Eklavya_G1_complete_build:
  scale:
    target: small_enough_for_daily_iteration
    reason: protocol truth beats scale illusion in the first build

  substrate:
    input_surface:
      - byte preserving
      - learned or entropy patch candidate
      - typed spans

    compact_state:
      - small attention core
      - recurrent scratch lane
      - nested state heads for cheap/deep operation

    verifier:
      - learned local verifier
      - exact checker bridge for arithmetic, schema, code, and byte identity

    memory:
      - retrieval key head
      - small typed external store
      - write gate disabled until retrieval works

    compute_governor:
      - rule-based governor first
      - learned governor only after enough traces exist

    update_ports:
      - adapter slots per interface
      - verifier-head update
      - memory-key update
      - compute-policy update

  first_domains:
    - byte exact strings
    - multiple-choice reasoning with controlled candidates
    - arithmetic or schema tasks with exact validation
    - retrieval/paraphrase pairs
    - short generation guardrails

  first_success_standard:
    - at least three lesson types accepted
    - at least three teacher measurement functions used
    - retained gain positive after teacher removal
    - no accepted lesson has unbounded collateral damage
    - adaptive compute reduces cost on easy cases without hard-case collapse
    - every improvement is attached to a lesson packet and landing zone
```

This build is the first moment where "Sutra" and "Eklavya" are both real:
Sutra owns lessons, and Eklavya proves they were worth teaching.

## What To Build After G1

If G1 works, the next versions should extend one axis at a time.

```text
post_G1_roadmap:
  G2_better_substrate:
    question:
      - can learned dynamic patching beat fixed spans under the same ownership
        protocol?
    do_not_change:
      - lesson state machine
      - ownership metrics

  G3_better_compute:
    question:
      - can learned compute governance beat rule-based governance on p95 cost
        and hard-case preservation?
    do_not_change:
      - teacher admission policy

  G4_better_memory:
    question:
      - when should Sutra store knowledge externally instead of in weights?
    do_not_change:
      - versioning and provenance requirements

  G5_better_updates:
    question:
      - can new skills be admitted through local update ports with less damage
        than full finetuning?
    do_not_change:
      - rollback and retained-gain requirements

  G6_broader_teachers:
    question:
      - which new measurement function adds real lesson value?
    do_not_change:
      - teacher-as-instrument doctrine
```

Each generation changes one bottleneck. If many bottlenecks move at once, the
protocol loses causal attribution.

## G1 Execution Blueprint

This is not implementation code. It is the exact buildable design of the first
complete system.

```text
G1_blueprint:
  objective:
    prove_that_Eklavya_can_create_owned_lessons_inside_Sutra

  non_objective:
    maximize_general_benchmark_score

  invariant:
    every improvement must be attributable to:
      - a gap
      - a teacher measurement function
      - a probe set
      - a lesson packet
      - a landing zone
      - an ownership test

  student_revision_0:
    modules:
      - I0_input_surface
      - I1_compact_state
      - I2_reasoning_step
      - I3_verification
      - I4_memory_read_only
      - I5_rule_based_compute_governor
      - I6_output_surface
      - update_ports_per_interface

    required_observability:
      - per_interface_activation_trace
      - per_interface_uncertainty
      - per_interface_cost
      - verifier_decision_trace
      - memory_query_trace
      - compute_governor_action_trace

  eklavya_revision_0:
    services:
      - gap_mapper
      - teacher_registry
      - probe_factory
      - response_surface_recorder
      - lesson_inferencer
      - packet_compiler
      - local_update_selector
      - ownership_evaluator
      - lesson_ledger

  first_gap_classes:
    surface_gap:
      examples:
        - byte exact copy
        - code symbol preservation
        - path and schema fidelity
      target_interfaces:
        - I0_input_surface
        - I6_output_surface

    semantic_gap:
      examples:
        - paraphrase with hard negative
        - entailment under wording shift
      target_interfaces:
        - I1_compact_state

    reasoning_gap:
      examples:
        - constrained multiple choice
        - arithmetic with exact answer
        - small program trace
      target_interfaces:
        - I2_reasoning_step
        - I3_verification

    memory_gap:
      examples:
        - delayed recall
        - retrieval key selection
        - conflict between two records
      target_interfaces:
        - I4_memory

    compute_gap:
      examples:
        - easy case should stop early
        - hard case should escalate
        - answerable case should retrieve
        - uncertain case should verify
      target_interfaces:
        - I5_compute_governor
```

The first build is complete only when every service and every interface above
has at least one exercised path. A missing path means the protocol is still a
diagram.

## Algorithm 1: Gap Mapping

Gap mapping starts from Sutra, not from teachers.

```text
gap_mapping_algorithm:
  input:
    - Sutra_revision
    - typed_eval_slices
    - validators
    - cost_trace

  for each example:
    run Sutra in trace mode
    record:
      - output
      - correctness_or_quality
      - interface traces
      - uncertainty
      - compute cost
      - verifier state
      - memory state

    classify failure:
      - surface_failure
      - semantic_failure
      - reasoning_failure
      - verification_failure
      - memory_failure
      - compute_policy_failure
      - update_locality_failure
      - unknown_mixed_failure

    assign suspected landing zones:
      - one primary interface
      - optional secondary interface

    estimate gap value:
      - usage_frequency
      - severity
      - cost_if_unfixed
      - expected_teachability

  output:
    ranked_gap_map
```

Rules:

- unknown mixed failures are allowed, but they cannot admit teacher packets yet;
- every teacher search must point to a ranked gap;
- if no validator can score the gap, the first lesson is to build or admit a
  validator.

## Algorithm 2: Teacher Instrument Selection

Teacher selection is an experimental-design problem.

```text
teacher_selection_algorithm:
  input:
    - ranked_gap
    - teacher_registry
    - existing_teacher_surfaces
    - budget

  for each candidate_teacher:
    score:
      hidden_state_relevance
      measurement_distinctness
      expected_probe_cost
      expected_packet_compressibility
      expected_landing_zone_match
      corruption_risk
      legal_or_operational_risk

    compute:
      expected_teacher_value =
        hidden_state_relevance
        * measurement_distinctness
        * expected_packet_compressibility
        * expected_landing_zone_match
        - expected_probe_cost
        - corruption_risk
        - operational_risk

  admit:
    - highest expected_teacher_value teachers
    - at least one non-behavior teacher when gap class allows
    - exact validators before learned verifier teachers when exact validation
      exists

  reject:
    - prestige-only teachers
    - same-family echoes without distinct surface evidence
    - teachers whose signal cannot compress into a lesson packet
```

This prevents Eklavya from becoming teacher collection.

## Algorithm 3: Probe Construction

Probes are designed to identify lessons, not to make teachers look smart.

```text
probe_construction_algorithm:
  input:
    - gap
    - candidate_teacher
    - suspected_landing_zone

  construct_probe_set:
    base_case:
      original failure

    invariance_cases:
      surface changes that should preserve target behavior

    counterfactual_cases:
      hidden condition changes that should change target behavior

    distractor_cases:
      irrelevant but tempting changes

    corruption_cases:
      noise, formatting, schema, or byte perturbation

    minimal_pair_cases:
      smallest contrast that distinguishes the target invariant

    validator_cases:
      exact checks if available

  accept_probe_set_if:
    - it can distinguish style imitation from invariant acquisition
    - it includes at least one expected failure of current Sutra
    - it includes at least one control where no change should happen
    - it maps each probe to a landing-zone hypothesis
```

The best probe set is the smallest one that can change a teaching decision.

## Algorithm 4: Lesson Inference

Lesson inference turns response surfaces into a causal teaching hypothesis.

```text
lesson_inference_algorithm:
  input:
    - student_surface
    - teacher_surfaces
    - validators
    - gap_metadata

  identify:
    stable_teacher_behavior:
      what competent teachers keep invariant

    corrective_difference:
      how competent teacher behavior differs from Sutra

    source_disagreement:
      where teachers differ and why

    validator_anchor:
      which behavior is objectively correct when exact checks exist

    minimal_contrast:
      smallest example pair that exposes the desired change

  propose_lesson:
    target_invariant:
    landing_zone:
    minimal_training_object:
    expected_update_port:
    corruption_modes:
    ownership_test:

  reject_if:
    - proposed target invariant is just teacher output
    - landing zone is unknown
    - validator or adjudication cannot separate correct from plausible
    - minimal contrast is absent
```

Eklavya's hardest intellectual move is here: infer a lesson, not a label.

### Causal Lesson Inference Doctrine

Teacher surfaces are correlational until Eklavya proves that a specific
intervention changes Sutra. Lesson inference must therefore use an intervention
ladder.

```text
causal_lesson_inference_v1:
  stage_0_observation:
    evidence:
      - Sutra fails a validated gap
      - teacher surfaces differ from Sutra surfaces
    allowed_claim:
      - possible lesson
    forbidden_claim:
      - causal lesson

  stage_1_contrast:
    evidence:
      - minimal pairs separate target invariant from style or template
      - invariance and counterfactual probes disagree in expected directions
    allowed_claim:
      - candidate invariant
    forbidden_claim:
      - landing zone proven

  stage_2_micro_intervention:
    evidence:
      - applying the smallest packet to the suspected landing zone changes
        target behavior on ownership examples
      - applying a null packet does not
      - applying the same packet to the wrong landing zone does not match the
        target improvement
    allowed_claim:
      - plausible causal lesson

  stage_3_ablation:
    evidence:
      - freezing the claimed landing zone prevents the lesson
      - freezing unrelated interfaces does not fully prevent the lesson
      - removing the packet removes the gain
    allowed_claim:
      - landing-zone-supported causal lesson

  stage_4_generalization:
    evidence:
      - held-out probes preserve the intended invariant
      - adjacent probes reveal bounded scope
      - collateral slices remain within damage limit
    allowed_claim:
      - accepted owned lesson
```

Required controls:

```text
causal_lesson_controls:
  - null_packet_same_update_budget
  - shuffled_teacher_surface
  - wrong_landing_zone_packet
  - teacher_style_only_packet
  - best_single_teacher_output_imitation_control
  - no_teacher_curriculum_control
```

Eklavya may accept a lesson only after stage 3, and may call it general only
after stage 4. Anything earlier is a hypothesis.

## Algorithm 5: Packet Compilation

A packet is a contract between Eklavya and Sutra.

```text
packet_schema_v1:
  identity:
    packet_id:
    parent_gap_id:
    sutra_base_revision:
    teacher_sources:
    probe_set_id:

  lesson:
    lesson_type:
    target_invariant:
    target_behavior:
    landing_zone:
    expected_update_port:

  training_object:
    format:
      allowed:
        - candidate_margin_table
        - minimal_pair_set
        - verifier_span_record
        - retrieval_triplet
        - compute_action_trace
        - byte_boundary_record
    payload:
    masks:
    weights:

  tests:
    ownership_test:
    dependence_test:
    collateral_damage_test:
    hard_case_test:
    cost_test:

  economics:
    expected_lesson_value:
    maximum_allowed_training_cost:
    maximum_allowed_inference_cost_delta:
    rollback_threshold:

  provenance:
    teacher_revisions:
    validator_revisions:
    data_refs:
    generation_params:
```

Packet compilation fails if any required field is missing. A missing field is
not a TODO; it means the lesson is not yet teachable.

## Algorithm 6: Local Update Selection

Sutra should update the smallest adequate mechanism.

```text
local_update_selection_order:
  1_memory_record:
    use_if:
      - failure is factual or retrieval-key related
      - no change in general reasoning is required

  2_runtime_policy:
    use_if:
      - failure is stop/retrieve/verify/escalate decision
      - underlying competence already exists

  3_verifier_head:
    use_if:
      - failure is invalidity detection or repair localization

  4_interface_adapter:
    use_if:
      - failure is localized to one interface
      - low-rank or constrained delta can express the lesson

  5_targeted_curriculum_update:
    use_if:
      - lesson needs repeated practice across related examples

  6_substrate_update:
    use_if:
      - failure cannot be localized
      - multiple lesson attempts show same substrate bottleneck

  forbidden_default:
    - full_model_finetune_for_first_local_gap
```

Escalation is allowed, but only after cheaper update ports fail the ownership
test or cannot represent the lesson.

### Update Port Doctrine

Localized lessons are not real unless the update port is constrained and tested.

```text
update_port_contract_v1:
  port_id:
  owner_interface:
  allowed_parameters_or_state:
  frozen_interfaces:
  allowed_loss_terms:
  forbidden_loss_terms:
  max_trainable_parameter_delta:
  max_runtime_cost_delta:
  rollback_unit:
  interaction_tests:
  credit_assignment_tests:
```

Port placement rules:

```text
update_port_placement_rules:
  memory_record:
    place_if:
      - target change is factual, contextual, or provenance-bound
    freeze:
      - all neural weights

  runtime_policy:
    place_if:
      - target change is action selection among existing capabilities
    freeze:
      - substrate and output heads

  verifier_head:
    place_if:
      - target change is invalidity detection, span localization, or repair
        class prediction
    freeze:
      - compact state and reasoning core unless verifier inputs are insufficient

  interface_adapter:
    place_if:
      - target change belongs to one interface representation
    freeze:
      - all unrelated interfaces

  targeted_curriculum:
    place_if:
      - repeated examples are needed but no single interface bottleneck is known
    requirement:
      - must be followed by credit assignment before acceptance

  substrate_update:
    place_if:
      - at least two cheaper ports fail for the same gap class
      - ablations show the bottleneck is shared substrate capacity
```

Loss construction rules:

```text
update_loss_rules:
  - every loss term must name one packet field and one landing-zone state
  - packet losses must be masked away from examples where the invariant is not
    claimed
  - loss weights start below base or control loss influence and can increase
    only after small-budget retained gain
  - style losses are forbidden unless style is the explicit target behavior
  - teacher-output losses are allowed only as contrastive evidence, not as the
    whole lesson
```

Rollback rules:

```text
rollback_rules:
  memory_record:
    rollback_unit: record_or_index_delta
  runtime_policy:
    rollback_unit: policy_version
  verifier_head:
    rollback_unit: head_delta
  interface_adapter:
    rollback_unit: adapter_delta
  targeted_curriculum:
    rollback_unit: data_mixture_and_checkpoint_delta
  substrate_update:
    rollback_unit: full_revision_branch
```

No update port can be considered safe until its rollback unit has been tested.

### Module Credit Assignment Doctrine

Retained gain is not enough. Eklavya must prove the claimed module carried the
lesson.

```text
landing_zone_attribution_report_v1:
  packet_id:
  claimed_landing_zone:
  update_port:
  freeze_tests:
    - claimed_zone_frozen
    - unrelated_zones_frozen
  ablation_tests:
    - remove_packet
    - remove_adapter_or_policy_delta
    - wrong_landing_zone_update
    - null_packet_same_budget
  attribution_result:
    allowed:
      - attributed
      - mixed_attribution
      - unattributed
      - inconclusive
```

Rules:

```text
module_credit_assignment_rules:
  - accepted Eklavya lessons must be attributed, not merely useful
  - mixed attribution can remain as engineering value but cannot prove the
    claimed lesson mechanism
  - unattributed gains must be reclassified as ordinary training improvements
  - inconclusive attribution blocks promotion until better ablations exist
```

## Algorithm 7: Ownership Evaluation

Ownership evaluation is the proof of Eklavya.

```text
ownership_evaluation_algorithm:
  input:
    - packet
    - base_sutra_revision
    - taught_sutra_revision
    - teacher_removed_runtime

  run:
    baseline_suite:
      base_sutra on target and control examples

    taught_suite:
      taught_sutra with no teacher calls and no temporary teaching artifacts

    dependence_suite:
      compare taught_sutra against teacher-assisted or artifact-assisted mode

    damage_suite:
      unrelated and adjacent capabilities

    cost_suite:
      training cost, inference delta, memory delta, validation cost

  compute:
    retained_gain
    dependence_gap
    collateral_damage
    cost_delta
    realized_lesson_value

  accept_if:
    - retained_gain > 0
    - realized_lesson_value > 0
    - dependence_gap within packet limit
    - collateral_damage within packet limit
    - cost_delta within packet limit

  reject_or_roll_back_if:
    - any hard limit is violated
    - improvement requires teacher at inference
    - improvement cannot be attributed to the packet
```

No ownership test, no Eklavya claim.

## Algorithm 8: Runtime Compute Governance

The compute governor is Sutra's efficiency engine.

```text
compute_governor_objective:
  choose action a that maximizes:
    expected_quality_gain(a)
    + expected_uncertainty_reduction(a)
    + expected_future_cost_reduction(a)
    - immediate_compute_cost(a)
    - latency_cost(a)
    - risk_of_wrong_stop(a)
```

Estimator doctrine:

```text
compute_governor_estimators_v1:
  expected_quality_gain:
    labels:
      - historical improvement from run_deeper, retrieve, verify, or repair on
        matched states
      - validator or human outcome delta
    calibration:
      - reliability curve by task class
      - separate easy and hard case bins

  expected_uncertainty_reduction:
    labels:
      - entropy drop
      - verifier uncertainty drop
      - disagreement drop
      - retrieval conflict reduction
    anti_gaming:
      - uncertainty drop without quality gain is not reward

  expected_future_cost_reduction:
    labels:
      - avoided retries
      - avoided repair loops
      - successful sleep/precompute reuse

  risk_of_wrong_stop:
    labels:
      - early exit failure
      - hard-case miss
      - unsupported answer
      - invalid structured output
```

Actions:

```text
compute_actions:
  stop_and_emit:
    risk:
      - wrong early answer

  run_deeper:
    risk:
      - wasted compute

  run_wider:
    risk:
      - expensive branch search

  retrieve_memory:
    risk:
      - stale or conflicting memory

  verify:
    risk:
      - verifier false confidence

  repair_span:
    risk:
      - local repair breaks global coherence

  precompute_sleep:
    risk:
      - spend offline compute on unused future states
```

Training signals:

```text
compute_governor_training_signals:
  - teacher disagreement
  - verifier failure
  - entropy or uncertainty
  - successful early exit
  - failed early exit
  - retrieval usefulness
  - repair success
  - p95 cost
  - hard-case loss
```

The governor must be evaluated on both average cost and tail failures. A cheap
average with damaged hard cases is not Sutra efficiency.

Off-policy and anti-gaming rules:

```text
compute_governor_evaluation_rules:
  - compare learned governor to rule governor and static-depth baseline
  - use logged counterfactual actions where available
  - require calibration before optimizing policy from its own predictions
  - report action distribution shift after training
  - fail if stop action increases unsupported or invalid answers
  - fail if retrieval or verification is suppressed mainly to reduce cost
  - fail if hard-case p95 cost or quality violates predeclared bounds
```

## Protocol Acceptance Gates

These are the gates that make the design executable.

```text
G1_acceptance_gates:
  gate_A_observability:
    pass_if:
      - every Sutra interface emits trace, uncertainty, and cost fields
      - every failure can be mapped to known or unknown_mixed

  gate_B_teacher_instruments:
    pass_if:
      - at least three measurement functions produce non-redundant surfaces
      - at least one exact or constraint teacher is used where available

  gate_C_packet_validity:
    pass_if:
      - every packet has gap, teacher, probes, landing zone, tests, economics,
        and rollback

  gate_D_local_update:
    pass_if:
      - at least three update ports are exercised by fully evaluated packet
        attempts
      - full-model finetuning is not the default update route

  gate_E_ownership:
    pass_if:
      - every accepted lesson retains positive value after teacher removal
      - every accepted lesson passes attribution
      - rejected lessons have clear failure semantics
      - no accepted lesson violates damage or cost limits

  gate_E2_credit_assignment:
    pass_if:
      - every accepted lesson has landing-zone attribution report
      - claimed landing zone passes freeze_or_ablation test
      - wrong_landing_zone control fails to reproduce the same gain
      - no untracked interface carries the majority of measured improvement

  gate_F_compute:
    pass_if:
      - compute governor reduces easy-case cost
      - hard-case quality floor is preserved
      - p95 cost is reported

  gate_G_revision_integrity:
    pass_if:
      - Sutra revision includes weights, adapters, memory schema, policies,
        validators, and accepted lesson ledger
```

G1 has two valid terminal outcomes:

```text
G1_terminal_outcomes:
  protocol_proven:
    requires:
      - every gate passes
      - at least three accepted attributed lessons
      - at least four lesson types attempted
      - at least three update ports exercised

  protocol_falsified_informatively:
    requires:
      - every observability, packet, credit-assignment, revision, and failure
        semantics gate passes
      - lesson attempts fail with clear attributed reasons
      - failures identify which doctrine assumption broke

  inconclusive:
    triggered_by:
      - weak traces
      - missing attribution
      - missing controls
      - failure reasons unclear
      - revision cannot reproduce decisions
```

Only `protocol_proven` supports an Eklavya success claim. `protocol_falsified`
is still a useful research result, but it is not a finished Sutra/Eklavya
protocol.

## Failure Semantics

Failures should teach the next design move.

```text
failure_semantics:
  teacher_surface_redundant:
    interpretation:
      - teacher roster is measuring the same hidden state
    response:
      - search for a different measurement function

  lesson_will_not_compress:
    interpretation:
      - teacher knows something but not in a form Sutra can own cheaply
    response:
      - improve probe design or reject teacher for this gap

  packet_lands_but_not_retained:
    interpretation:
      - update path creates temporary performance without ownership
    response:
      - change update port or strengthen landing-zone capacity

  retained_but_damaging:
    interpretation:
      - lesson is too broad or update is poorly localized
    response:
      - narrow packet, add guards, use smaller adapter or memory route

  cheap_but_hard_cases_fail:
    interpretation:
      - compute governor learned avoidance, not efficiency
    response:
      - raise hard-case floor and train explicit escalation lessons

  protocol_cost_too_high:
    interpretation:
      - Eklavya overhead exceeds lesson value
    response:
      - restrict Eklavya to high-value gaps and simplify packet machinery
```

This is how the design avoids self-deception: every failure updates doctrine,
not just hyperparameters.

## Concrete State And Trace Schemas

The future builder needs one canonical trace vocabulary. Otherwise every
experiment will invent its own bookkeeping and Eklavya will lose attribution.

```text
Sutra_trace_schema_v1:
  trace_id:
  sutra_revision_id:
  example_id:
  task_class:
  input_ref:
  output_ref:

  interface_records:
    I0_input_surface:
      typed_spans:
      byte_residual_flags:
      surface_risk_flags:
      uncertainty:
      cost:

    I1_compact_state:
      semantic_state_ref:
      task_state_ref:
      memory_query_state_ref:
      verifier_query_state_ref:
      uncertainty:
      cost:

    I2_reasoning_step:
      step_count:
      scratch_state_refs:
      candidate_action_refs:
      uncertainty_by_step:
      cost:

    I3_verification:
      verifier_inputs:
      verdict:
      error_spans:
      repair_classes:
      escalation_recommendation:
      cost:

    I4_memory:
      query_refs:
      retrieved_record_refs:
      conflicts:
      write_candidates:
      write_decision:
      cost:

    I5_compute_governor:
      actions_taken:
      actions_rejected:
      stop_reason:
      expected_quality_gain:
      expected_cost:
      realized_cost:

    I6_output_surface:
      output_type:
      byte_exact_regions:
      structured_output_schema:
      final_repair_actions:
      cost:

  validators:
    exact_validator_results:
    learned_validator_results:
    human_or_adjudicator_result:

  outcome:
    score_vector:
    failure_class:
    suspected_primary_interface:
    suspected_secondary_interface:
```

The trace is not optional instrumentation. It is the raw material from which
Eklavya discovers gaps.

## Teacher Surface Schema

A teacher surface is a measurement object, not a pile of completions.

```text
teacher_surface_schema_v1:
  surface_id:
  teacher_id:
  teacher_revision:
  teacher_measurement_function:
    allowed:
      - behavior
      - geometry
      - constraint
      - process
      - curriculum
      - adversarial

  gap_id:
  probe_set_id:
  examples:
    - example_id

  measurements:
    candidate_scores:
      shape: example_by_probe_by_candidate_by_field
      fields:
        - score
        - rank
        - margin
        - confidence
        - entropy
        - abstain_or_invalid

    geometry_scores:
      shape: example_pair_or_triplet_by_field
      fields:
        - similarity
        - margin
        - neighborhood_rank
        - hard_negative_flag

    constraint_scores:
      shape: example_by_constraint_by_field
      fields:
        - pass_fail_unknown
        - violated_constraint
        - error_span
        - repair_class

    process_scores:
      shape: example_by_step_by_field
      fields:
        - intermediate_state
        - action
        - branch_score
        - termination_reason

    curriculum_scores:
      shape: example_by_field
      fields:
        - prerequisite_tags
        - difficulty
        - contrast_set_id
        - recommended_order

  cost:
    wall_time:
    compute:
    memory:
    money:
    human_time:

  corruption_flags:
    - style_bias
    - refusal_bias
    - tokenizer_artifact
    - benchmark_artifact
    - unsafe_shortcut
    - unknown
```

Every teacher surface must be comparable to a Sutra trace on the same probe set.
If it cannot be compared, it is not yet a usable Eklavya measurement.

## Gap Record Schema

The gap record is the first object in the Eklavya lifecycle.

```text
gap_record_schema_v1:
  gap_id:
  discovered_from_trace_ids:
  sutra_revision_id:
  gap_class:
    allowed:
      - surface
      - semantic
      - reasoning
      - verification
      - memory
      - compute_policy
      - update_locality
      - unknown_mixed

  target_behavior:
  current_behavior:
  validator_or_adjudicator:
  suspected_landing_zone:
  expected_value_if_fixed:
  failure_examples:
  control_examples:
  required_teacher_measurement_functions:
  status:
    allowed:
      - observed
      - teacher_search
      - probe_design
      - lesson_candidate
      - packet_compiled
      - taught
      - accepted
      - rejected
```

Gaps with status `unknown_mixed` can trigger analysis but not packet teaching.
The protocol must first split them into teachable gaps.

## Lesson Ledger Schema

The lesson ledger is the memory of Eklavya. It is how Sutra can improve without
lying about why it improved.

```text
lesson_ledger_row_v1:
  lesson_id:
  packet_id:
  gap_id:
  sutra_base_revision_id:
  sutra_taught_revision_id:
  sutra_committed_revision_id:

  lesson_type:
  landing_zone:
  update_port:
  teacher_measurement_functions:
  teacher_surface_ids:
  probe_set_id:

  expected:
    expected_retained_gain:
    expected_dependence_gap:
    expected_collateral_damage:
    expected_cost:
    expected_lesson_value:

  realized:
    retained_gain:
    dependence_gap:
    collateral_damage:
    training_cost:
    inference_cost_delta:
    validation_cost:
    realized_lesson_value:

  decision:
    allowed:
      - accepted
      - rejected
      - rolled_back
      - needs_more_evidence

  decision_reason:
  rollback_ref:
  followup_gap_ids:
```

The ledger is part of Sutra's revision identity. A checkpoint without the ledger
is not a Sutra revision; it is just weights.

## Tensor And State Shape Commitments

These shapes are deliberately abstract enough to survive implementation changes
but concrete enough to prevent hand-waving.

```text
G1_state_shapes:
  raw_bytes:
    shape: uint8[B, T_byte]
    owner: I0_input_surface

  typed_spans:
    shape: int[B, S, 3]
    fields:
      - byte_start
      - byte_end
      - span_type_id
    owner: I0_input_surface

  byte_residual_flags:
    shape: bool[B, T_byte]
    owner: I0_input_surface

  patch_or_span_states:
    shape: float[B, S, D]
    owner: I0_input_surface
    consumer: I1_compact_state

  compact_state:
    shape: float[B, S_compact, D]
    owner: I1_compact_state
    consumers:
      - I2_reasoning_step
      - I3_verification
      - I4_memory
      - I5_compute_governor

  scratch_state:
    shape: float[B, R, D]
    owner: I2_reasoning_step
    meaning: recurrent_or_stepwise_reasoning_workspace

  verifier_state:
    shape: float[B, V, D]
    owner: I3_verification
    emits:
      - verdict_logits
      - error_span_logits
      - repair_class_logits

  memory_query_state:
    shape: float[B, Q, D]
    owner: I4_memory

  compute_policy_state:
    shape: float[B, A]
    owner: I5_compute_governor
    fields:
      - action_logits
      - expected_quality_delta
      - expected_cost_delta
      - risk_score
      - stop_score

  output_state:
    shape: float[B, T_out_or_S_out, D]
    owner: I6_output_surface
```

The exact values of `D`, `S`, `R`, `V`, `Q`, and `A` are implementation choices.
The ownership of each state is not. Eklavya packets must target the owner
interface, not an anonymous hidden layer.

## End-To-End Lesson Lifecycle Example

The abstract protocol becomes clear only when one lesson is followed all the way
through. This is the reference lifecycle for G1.

```text
example_lesson_lifecycle:
  name: schema_repair_verification_lesson

  observed_gap:
    gap_class: verification
    failure:
      - Sutra emits JSON-like output that looks plausible but violates a
        required schema
    trace_evidence:
      - I6_output_surface produced malformed field type
      - I3_verification returned pass_or_unknown instead of fail
      - I5_compute_governor stopped without repair
    suspected_landing_zone:
      - I3_verification
      - I5_compute_governor

  teacher_instruments:
    constraint_teacher:
      type: exact_json_schema_validator
      measures:
        - pass_fail
        - violated_constraint
        - error_span
        - repair_class

    behavior_teacher:
      type: strong_decoder_or_code_model
      measures:
        - candidate repair preferences
        - plausible corrected outputs

    adversarial_teacher:
      type: schema_mutation_generator
      measures:
        - brittle malformed variants

  probe_set:
    base_case:
      - original malformed output prompt

    invariance_cases:
      - same schema with field order changed
      - same content with whitespace changed

    counterfactual_cases:
      - schema changes required type
      - required field removed

    distractor_cases:
      - natural-language instruction says output is acceptable though schema
        disagrees

    corruption_cases:
      - near-valid JSON
      - wrong enum casing
      - nested object type mismatch

  teacher_surface:
    constraint_surface:
      - exact pass_fail
      - violated field
      - error span
      - repair class

    behavior_surface:
      - candidate repair ranking
      - confidence margin between valid and invalid candidates

    adversarial_surface:
      - generated malformed variants grouped by failure class

  lesson_inference:
    target_invariant:
      - when a structured-output constraint exists, schema validity dominates
        surface plausibility

    minimal_training_object:
      - invalid output
      - violated constraint
      - error span
      - valid repair candidate
      - distractor invalid candidate

    lesson_type:
      - verification_lesson
      - compute_policy_lesson

    landing_zone:
      - I3_verification
      - I5_compute_governor

  packet:
    training_object_format:
      - verifier_span_record
      - compute_action_trace

    update_port:
      - verifier_head_update
      - runtime_policy_update

    ownership_test:
      - teacher and validator-derived labels unavailable at inference except
        built-in exact validator when the task explicitly supplies a schema
      - Sutra must detect invalidity and choose repair on held-out schema
        variants

    collateral_damage_test:
      - no degradation on valid JSON cases
      - no over-repair of natural-language outputs
      - no increased refusal on ordinary structured tasks

    cost_test:
      - verifier cost added only when schema-risk flag is active
      - average reflex-mode tasks do not pay schema verification cost

  local_update:
    first_try:
      - update I3 verifier head for error span and repair class
      - update I5 rule to escalate when schema-risk and verifier uncertainty are
        both high

    forbidden_first_try:
      - full model finetune on JSON outputs

  ownership_eval:
    accept_if:
      - held-out schema violation detection improves
      - repair selection improves
      - valid output over-repair remains below limit
      - cost is paid only on schema-risk tasks
      - teacher decoder is not called at inference

    reject_if:
      - Sutra only memorizes seen schemas
      - verifier flags too many valid outputs
      - compute governor verifies every structured output regardless of risk
      - improvement disappears when behavior teacher examples are removed

  accepted_lesson_effect:
    - Sutra now owns a local schema-repair behavior
    - Eklavya records exact validator as a high-value constraint teacher for
      this gap class
    - future similar gaps prioritize exact constraint teachers before decoder
      teachers
```

This is what "learning from many teachers without distillation" means in
practice. The decoder can help rank repairs, the exact validator supplies truth,
the adversary supplies brittle cases, but Sutra only accepts the lesson if it
owns the verification and repair behavior afterward.

## Second End-To-End Lifecycle: Compute Escalation

The first example teaches a verifier. The second teaches Sutra when to spend
more compute.

```text
compute_escalation_lesson_lifecycle:
  observed_gap:
    gap_class: compute_policy
    failure:
      - Sutra exits early on arithmetic word problems whose surface looks easy
        but contain a hidden multi-step dependency
    trace_evidence:
      - I5_compute_governor chose stop_and_emit
      - I2_reasoning_step ran only shallow path
      - exact arithmetic validator failed

  teacher_instruments:
    constraint_teacher:
      type: exact_answer_checker
      measures:
        - final correctness

    process_teacher:
      type: solver_or_search_trace
      measures:
        - number of required operations
        - dependency graph
        - intermediate state sequence

    behavior_teacher:
      type: decoder_teacher
      measures:
        - confidence under paraphrase and distractor probes

  probe_set:
    minimal_pairs:
      - one-step arithmetic problem
      - visually similar two-step arithmetic problem

    distractors:
      - irrelevant large number
      - familiar template with changed relation

    counterfactuals:
      - operation order changed
      - hidden dependency removed

  lesson_inference:
    target_invariant:
      - surface simplicity is not sufficient evidence for reflex mode when
        dependency markers are present

    lesson_type:
      - compute_policy_lesson
      - reasoning_transition_lesson

    landing_zone:
      - I5_compute_governor
      - I2_reasoning_step

    minimal_training_object:
      - input
      - dependency markers
      - shallow wrong answer
      - deeper correct trace summary
      - validator result
      - correct action: run_deeper

  local_update:
    first_try:
      - update compute policy threshold for dependency markers
      - add small reasoning-transition adapter for two-step state update

  ownership_eval:
    accept_if:
      - hard two-step cases improve
      - easy one-step cases still exit cheaply
      - p95 cost does not blow up
      - improvement survives without process teacher

    reject_if:
      - governor escalates all arithmetic
      - reasoning improves only on seen templates
      - cost increase exceeds retained gain
```

This is Sutra's efficiency story in miniature: it does not become efficient by
thinking less; it becomes efficient by knowing when thinking less is safe.

## Canonical Builder Handoff

The future builder should not begin by asking "which model do we train?" The
first question is:

```text
which artifact proves that the next design claim is no longer ambiguous?
```

Build order:

```text
builder_handoff_sequence:
  H0_doctrine_freeze:
    purpose:
      - freeze the definitions of Sutra, Eklavya, lesson, packet, ownership,
        and G1 success
    required_artifacts:
      - ground_up_future_design_doc
      - doctrine_level_interface_registry
      - doctrine_level_lesson_type_registry
      - doctrine_level_acceptance_gate_table
    completion_condition:
      - future builder can describe the system without referring to old SE1 or
        ordinary distillation

  H1_observable_sutra_body:
    purpose:
      - create a learner whose failures can be seen and localized
    required_artifacts:
      - Sutra_state_schema
      - Sutra_trace_schema
      - interface_trace_examples
      - cost_trace_examples
      - validator_registry
    completion_condition:
      - every first-pass example produces a trace with interface, uncertainty,
        cost, and outcome fields

  H2_gap_map:
    purpose:
      - identify teachable failures before admitting teachers
    required_artifacts:
      - gap_records
      - typed_eval_slices
      - validator_results
      - ranked_gap_map
    completion_condition:
      - at least one gap exists for surface, reasoning or verification, memory,
        and compute policy

  H3_teacher_instrumentation:
    purpose:
      - admit teachers as measurement instruments
    required_artifacts:
      - teacher_registry
      - teacher_surface_schema
      - probe_set_registry
      - response_surface_records
      - redundancy_report
    completion_condition:
      - at least three measurement functions produce comparable non-redundant
        surfaces on the same probe sets

  H4_lesson_compilation:
    purpose:
      - turn teacher surfaces into minimal packets
    required_artifacts:
      - packet_records
      - expected_lesson_value_estimates
      - landing_zone_map
      - rollback_plan
    completion_condition:
      - at least five packets exist across at least three lesson types, each
        with tests and economics complete

  H5_local_teaching:
    purpose:
      - apply lessons without defaulting to global finetuning
    required_artifacts:
      - update_port_selection_records
      - taught_sutra_revisions
      - training_cost_records
      - rejected_update_records
    completion_condition:
      - at least two update ports have been exercised and compared

  H6_ownership_evaluation:
    purpose:
      - prove which lessons are owned after teacher removal
    required_artifacts:
      - ownership_reports
      - dependence_gap_reports
      - collateral_damage_reports
      - realized_lesson_value_reports
      - lesson_ledger
    completion_condition:
      - every accepted lesson has positive realized value, attribution, and
        bounded damage after teacher removal
      - rejected lessons have clear failure semantics
      - the run is classified as protocol_proven, protocol_falsified_informatively,
        or inconclusive

  H7_compute_efficiency:
    purpose:
      - prove Sutra spends compute conditionally without hard-case collapse
    required_artifacts:
      - compute_governor_policy
      - action_trace_report
      - easy_case_cost_report
      - hard_case_quality_report
      - p95_cost_report
    completion_condition:
      - easy cases become cheaper while hard cases preserve quality

  H8_revision_integrity:
    purpose:
      - make Sutra a versioned runtime, not a loose checkpoint
    required_artifacts:
      - revision_manifest
      - weights_or_substrate_ref
      - adapter_refs
      - memory_schema_ref
      - policy_refs
      - validator_refs
      - accepted_lesson_ledger_ref
    completion_condition:
      - a reproduced revision yields the same traces, costs, and accepted
        lesson decisions
```

This sequence is the build plan. Skipping a stage means losing the ability to
prove whether Eklavya taught Sutra or whether something merely improved.

H0/W0 boundary:

```text
handoff_boundary:
  H0_doctrine_freeze:
    artifact_kind:
      - human-readable doctrine embedded in this document
    includes:
      - interface registry definitions
      - lesson type registry definitions
      - acceptance gate definitions
    excludes:
      - machine-readable manifests
      - executable schemas
      - implementation tickets

  W0_doctrine_and_registry_materialization:
    artifact_kind:
      - machine-readable and implementation-facing materialization of H0
    includes:
      - interface_registry.json_or_equivalent
      - lesson_type_registry.json_or_equivalent
      - teacher_measurement_function_registry.json_or_equivalent
      - G1_threshold_table.json_or_equivalent

  start_building_rule:
    - engineering may begin at W0 after H0 definitions are stable in doctrine
    - engineering may not begin at W1 until W0 materialized registries exist
      and match H0 definitions
```

## Start-Building Gates

The design phase can hand off to engineering only when these gates are written
and agreed.

```text
start_building_gates:
  doctrine_gate:
    required:
      - Sutra defined as runtime, not checkpoint
      - Eklavya defined as lesson protocol, not distillation
      - ownership defined as teacher-free retained gain
      - lifetime cost defined

  interface_gate:
    required:
      - I0_to_I6 interfaces named
      - each interface has trace fields
      - each interface has at least one possible lesson type
      - each interface has known failure classes

  packet_gate:
    required:
      - packet schema complete
      - gap schema complete
      - teacher surface schema complete
      - lesson ledger schema complete

  first_lifecycle_gate:
    required:
      - one verification lesson lifecycle specified
      - one compute-policy lesson lifecycle specified
      - each lifecycle has accept and reject conditions

  artifact_gate:
    required:
      - every H0_to_H8 stage has required artifacts
      - every artifact has a completion condition
      - no stage depends on a benchmark average alone

  pivot_gate:
    required:
      - failure semantics written
      - reset conditions written
      - scale-up conditions written
```

If these gates pass, engineering can begin. If they do not pass, more design is
required.

## Pivot And Reset Criteria

The system should not blindly continue if the ground-up thesis fails.

```text
pivot_criteria:
  pivot_teacher_ecology_if:
    - teacher surfaces are redundant across two probe revisions
    - exact or constraint teachers outperform learned teachers for the same gap
    - behavior teachers add style but no retained value

  pivot_student_substrate_if:
    - multiple lesson types fail because there is no stable landing zone
    - surface fidelity and compact reasoning cannot coexist under current
      representation
    - compute governor cannot observe the uncertainty signals it needs

  pivot_update_ports_if:
    - accepted lessons repeatedly require broad substrate updates
    - local adapters retain lessons but create high collateral damage
    - memory updates outperform weight updates for the same class of gap

  pivot_probe_design_if:
    - probes create disagreement but not teachable contrasts
    - minimal pairs do not predict held-out ownership
    - corruption probes expose teacher artifacts more than student gaps

  pivot_compute_governor_if:
    - cheap modes damage hard cases
    - p95 cost rises faster than retained gain
    - learned policy cannot beat a simple rule policy

  reset_G1_if:
    - fewer than three lessons are accepted and failures are not informative
      after two full packet cycles
    - no non-behavior teacher contributes positive realized value and the cause
      is not explained by clear failure semantics
    - revision identity cannot reproduce lesson decisions
    - protocol overhead exceeds realized lesson value or falsification value on
      most fully evaluated lessons
```

Reset is not failure. Reset is how the protocol avoids becoming a prestige
project defended by sunk cost.

## Scale-Up Criteria

Sutra should scale only after protocol truth is visible.

```text
scale_up_criteria:
  scale_student_if:
    - multiple packets fail because landing-zone capacity is insufficient
    - the same lessons succeed in small form but saturate quickly
    - added capacity improves retained gain per cost, not just raw score

  scale_teacher_roster_if:
    - current teachers leave high-value gaps unexplained
    - new measurement function is available
    - expected teacher value beats interrogation and packet cost

  scale_dataset_if:
    - current slices no longer expose new failure modes
    - accepted lessons generalize within slice but fail adjacent distributions
    - added data is attached to a gap or validator need

  scale_compute_if:
    - hard cases are bottlenecked by reasoning depth
    - compute governor can predict which cases deserve more cost
    - added test-time compute improves realized value after cost

  scale_memory_if:
    - repeated failures are factual, contextual, or retrieval-key failures
    - external memory beats weight update on reliability per cost
    - provenance and conflict handling are working
```

The default is not scale. The default is to ask what bottleneck scaling would
actually remove.

## Third End-To-End Lifecycle: Memory Retrieval

This lifecycle closes the memory lesson gap. It defines when Sutra should store
or retrieve knowledge externally instead of forcing the lesson into weights.

```text
memory_retrieval_lesson_lifecycle:
  observed_gap:
    gap_class: memory
    failure:
      - Sutra answers a delayed-recall question from stale context or parametric
        prior instead of retrieving the relevant provided record
    trace_evidence:
      - I4_memory query either absent or low-rank wrong record
      - I1_compact_state represents the topic but not the disambiguating key
      - I5_compute_governor chose answer_without_retrieval
    suspected_landing_zone:
      - I4_memory
      - I5_compute_governor
      - I1_compact_state

  teacher_instruments:
    geometry_teacher:
      type: embedding_or_retrieval_model
      measures:
        - query-positive similarity
        - hard-negative margin
        - neighborhood stability

    constraint_teacher:
      type: exact_record_match_or_citation_checker
      measures:
        - whether answer is supported by retrieved record
        - whether cited record contains required fact

    adversarial_teacher:
      type: hard_negative_generator
      measures:
        - confusable records
        - stale variants
        - same-entity different-time conflicts

  probe_set:
    base_case:
      - question answerable only from one provided record

    invariance_cases:
      - paraphrased question
      - record order changed
      - irrelevant context added

    counterfactual_cases:
      - target fact changed in new record
      - same entity with different date or jurisdiction

    distractor_cases:
      - semantically similar but unsupported record
      - parametric common answer that conflicts with provided record

  lesson_inference:
    target_invariant:
      - when task facts are supplied externally, answer authority comes from
        retrieved support, not parametric familiarity

    lesson_type:
      - memory_lesson
      - compute_policy_lesson
      - semantic_invariance_lesson

    landing_zone:
      - I4_memory query formation
      - I5 retrieve_or_answer policy
      - I1 compact state disambiguation keys

    minimal_training_object:
      - query
      - positive record
      - hard negative records
      - required support span
      - unsupported parametric answer
      - correct action: retrieve_then_answer

  local_update:
    first_try:
      - update memory query head with retrieval triplets
      - update compute policy to retrieve when support_required flag is active
      - do not change base semantic weights unless query formation cannot be
        learned locally

  ownership_eval:
    accept_if:
      - retrieval hit rate improves on held-out paraphrases
      - hard-negative confusion drops
      - answer support rate improves
      - ordinary no-memory questions do not trigger unnecessary retrieval
      - improvement survives without geometry teacher at inference

    reject_if:
      - Sutra memorizes training records in weights
      - retrieval works only for seen wording
      - memory is consulted for every answer
      - stale records are trusted over newer explicit records

  accepted_lesson_effect:
    - Sutra learns that some facts belong in external memory
    - Eklavya records geometry teachers as useful only when hard-negative and
      support validators prove value
    - future factual-update gaps try memory routes before weight updates
```

Memory is not a cheat code. It is an update-efficiency mechanism. It is admitted
only when retrieval is cheaper, more reliable, or more updateable than putting
the fact into weights.

## Fourth End-To-End Lifecycle: Semantic Invariance

This lifecycle closes the semantic invariance gap. It teaches Sutra to preserve
meaning across surface changes while still reacting to true counterfactuals.

```text
semantic_invariance_lesson_lifecycle:
  observed_gap:
    gap_class: semantic
    failure:
      - Sutra changes its answer under harmless paraphrase but fails to change
        under a subtle counterfactual
    trace_evidence:
      - I1_compact_state moves too far for paraphrase pair
      - I1_compact_state stays too close for counterfactual pair
      - I2_reasoning_step follows surface cue instead of relation change
    suspected_landing_zone:
      - I1_compact_state
      - I2_reasoning_step

  teacher_instruments:
    geometry_teacher:
      type: embedding_or_contrastive_encoder
      measures:
        - paraphrase neighborhood
        - hard-negative separation
        - counterfactual distance

    behavior_teacher:
      type: decoder_teacher_on_candidate_margins
      measures:
        - answer stability under paraphrase
        - margin flip under counterfactual

    curriculum_teacher:
      type: minimal_pair_generator
      measures:
        - smallest wording change preserving meaning
        - smallest relation change altering answer

  probe_set:
    invariance_cases:
      - lexical paraphrase
      - syntactic reorder
      - compression preserving relation

    counterfactual_cases:
      - subject-object relation swapped
      - quantifier changed
      - temporal condition changed

    distractor_cases:
      - emotionally salient but irrelevant phrase
      - familiar template with different relation

    minimal_pairs:
      - one pair where answer must remain stable
      - one pair where answer must change

  lesson_inference:
    target_invariant:
      - compact state should ignore surface variation that preserves task
        relation and amplify changes that alter the relation

    lesson_type:
      - semantic_invariance_lesson
      - reasoning_transition_lesson

    landing_zone:
      - I1_compact_state
      - I2_reasoning_step

    minimal_training_object:
      - anchor example
      - paraphrase positive
      - counterfactual negative
      - hard distractor
      - candidate answer margins
      - relation label or validator

  local_update:
    first_try:
      - compact-state contrastive adapter
      - reasoning transition adapter on relation-change probes
      - no broad language-model finetune

  ownership_eval:
    accept_if:
      - paraphrase stability improves
      - counterfactual sensitivity improves
      - unrelated stylistic diversity is preserved
      - behavior survives without geometry teacher
      - candidate margin gains beat naive teacher averaging

    reject_if:
      - state collapses paraphrase and counterfactual together
      - model learns teacher embedding geometry without task improvement
      - gains appear only on generated probe templates

  accepted_lesson_effect:
    - Sutra owns a relation-sensitive semantic state update
    - Eklavya learns whether geometry teacher signals are useful for this gap
      or merely diagnostic
```

This is the cleanest case where Eklavya must not become distillation. The lesson
is not the teacher's embedding. The lesson is a student-native relation geometry
that preserves the right invariances and breaks on the right counterfactuals.

## Normalized Efficiency Reporting Units

The phrase "world's most efficient model" is meaningless until the accounting
unit is fixed. Sutra should report efficiency in normalized units, not vibes.

```text
normalized_efficiency_report_v1:
  capability_units:
    primary:
      owned_task_score:
        definition:
          - task score after teacher removal
          - validated on target and adjacent slices
    secondary:
      hard_case_preservation:
      byte_exactness:
      support_rate:
      verifier_pass_rate:
      update_locality_score:

  cost_units:
    training_compute:
      unit: normalized_GPU_seconds_or_FLOP_estimate
      include:
        - base training
        - packet training
        - adapter training
        - failed lesson attempts

    teacher_cost:
      unit: normalized_teacher_query_cost
      include:
        - teacher inference
        - solver or validator execution
        - human adjudication
        - probe generation

    inference_cost:
      unit: expected_active_FLOPs_plus_latency
      report:
        - mean
        - p50
        - p95
        - hard_case_mean

    memory_cost:
      unit: storage_bytes_plus_retrieval_compute
      include:
        - index build
        - query cost
        - conflict checks
        - memory maintenance

    validation_cost:
      unit: validator_compute_plus_human_review
      include:
        - ownership eval
        - collateral damage eval
        - regression eval

    update_cost:
      unit: time_and_compute_to_accept_or_reject_lesson
      include:
        - packet construction
        - local update
        - rollback if failed

  derived_metrics:
    owned_gain_per_total_cost:
      formula:
        retained_gain / total_lifecycle_cost

    efficiency_frontier_delta:
      formula:
        Sutra_owned_gain_per_total_cost - best_control_owned_gain_per_total_cost

    teacher_return_on_cost:
      formula:
        realized_lesson_value / teacher_cost

    update_locality_ratio:
      formula:
        target_gain / max(collateral_damage, epsilon)

    adaptive_compute_return:
      formula:
        quality_preserved_per_cost_saved_on_easy_cases_with_hard_case_floor

    memory_weight_tradeoff:
      formula:
        memory_route_realized_value - weight_update_realized_value_for_same_gap
```

Utility scale:

```text
utility_scale_v1:
  rule:
    - never subtract raw capability points, dollars, FLOPs, latency, and damage
      without normalization

  normalized_terms:
    retained_gain_u:
      definition: retained_gain / predeclared_target_gain
    dependence_gap_u:
      definition: dependence_gap / max(teacher_assisted_gain, epsilon)
    collateral_damage_u:
      definition: collateral_damage / allowed_damage_limit
    training_cost_u:
      definition: training_cost / allocated_training_budget
    inference_cost_u:
      definition: inference_cost_delta / allowed_inference_delta
    validation_cost_u:
      definition: validation_cost / allocated_validation_budget

  scalar_decision_value:
    formula:
      retained_gain_u
      - dependence_gap_u
      - collateral_damage_u
      - training_cost_u
      - inference_cost_u
      - validation_cost_u

  hard_floors:
    - retained_gain_must_be_positive
    - collateral_damage_must_not_exceed_limit
    - dependence_gap_must_not_exceed_limit
    - hard_case_floor_must_hold
```

The scalar decision value is only a tie-breaker after hard floors pass. A lesson
with high utility but broken hard-case behavior is rejected.

Reporting table:

```text
efficiency_reporting_table_required_columns:
  - sutra_revision_id
  - lesson_or_eval_scope
  - owned_task_score
  - retained_gain
  - total_lifecycle_cost
  - training_compute
  - teacher_cost
  - inference_cost_mean
  - inference_cost_p95
  - memory_cost
  - validation_cost
  - update_cost
  - collateral_damage
  - owned_gain_per_total_cost
  - best_control_owned_gain_per_total_cost
  - efficiency_frontier_delta
```

Controls:

```text
efficiency_controls:
  - same_sutra_without_lesson
  - same_sutra_with_naive_output_distillation_if_applicable
  - same_sutra_with_best_single_teacher_packet
  - same_sutra_with_global_finetune
  - larger_baseline_model_when_available
  - cheaper_rule_or_exact_system_when_available
```

Sutra is more efficient only when it improves `owned_gain_per_total_cost`
against the relevant control while preserving hard-case quality and accounting
for teacher, memory, update, and validation cost.

## First Budget-Selection Doctrine

The first build must choose budgets by falsification value, not ambition.

```text
first_budget_selection_doctrine:
  purpose:
    - spend the smallest budget that can falsify the full Sutra/Eklavya loop

  budget_axes:
    student_capacity:
      choose_for:
        - enough modules to host I0_to_I6
        - enough capacity to show local updates can matter
      do_not_choose_for:
        - leaderboard score

    data_volume:
      choose_for:
        - enough examples per gap class to separate training, ownership, and
          damage checks
      do_not_choose_for:
        - broad corpus coverage before lesson value is proven

    teacher_queries:
      choose_for:
        - stable surfaces on selected probe sets
        - at least three measurement functions
      do_not_choose_for:
        - exhaustive teacher profiling

    packet_training_compute:
      choose_for:
        - testing local update ports
        - comparing accepted and rejected lessons
      do_not_choose_for:
        - rescuing weak packets with more optimization

    validation_budget:
      choose_for:
        - ownership tests
        - collateral damage tests
        - p95 compute checks
      do_not_choose_for:
        - broad benchmark theater
```

Budget rules:

```text
G1_budget_rules:
  rule_1:
    validation_budget_must_be_reserved_before_training

  rule_2:
    teacher_query_budget_must_be_capped_per_gap

  rule_3:
    failed_lesson_attempts_count_against_total_lifecycle_cost

  rule_4:
    no_packet_gets_more_budget_until_it_has_positive_small_budget_signal

  rule_5:
    scale_budget_only_after_failure_semantics_identify_capacity_as_the_bottleneck

  rule_6:
    if a cheaper exact system solves the gap, prefer it as teacher or runtime
    validator rather than forcing the lesson into neural weights
```

Initial G1 budget posture:

```text
G1_budget_posture:
  student:
    target:
      - smallest modular Sutra that can expose all interfaces
    reason:
      - protocol attribution is more valuable than raw capability in G1

  gap_classes:
    target:
      - surface
      - semantic
      - verification
      - memory
      - compute_policy

  lessons:
    target:
      - at least eight compiled packets
      - at least five fully evaluated lesson attempts
      - accepted lessons counted only if they pass ownership and attribution
      - rejected lessons count toward protocol_falsified_informatively only if
        failure semantics are clear

  teachers:
    target:
      - at least three measurement functions
      - exact validators wherever available
      - no teacher admitted by prestige alone

  compute:
    target:
      - enough to compare local update ports and ownership
      - not enough to hide bad lesson design behind brute force
```

The right first budget is the one that makes wrong theories fail quickly.

## Reference G1 Build Spec

This section gives a falsifiable reference prior, not a truth claim. The numbers
below are doctrine defaults chosen to exercise every interface and lesson route.
They must be replaced by measured hardware and pilot data before scale-up, but
they are specific enough to prevent the first build from drifting.

```text
G1_reference_build:
  name: Sutra_Eklavya_G1_reference
  purpose:
    - prove the full lesson protocol before scaling
    - produce a reusable runtime architecture and lesson ledger
    - reject ordinary distillation as the default learning path

  Sutra_reference_size:
    total_nominal_parameters: 485M
    active_in_reflex_mode_target: 160M_to_220M
    active_in_deliberation_mode_target: 320M_to_485M

  parameter_allocation:
    I0_input_surface:
      target_params: 45M
    I1_I2_compact_and_reasoning_core:
      target_params: 300M
    I3_verification:
      target_params: 35M
    I4_memory:
      target_params: 25M
    I5_compute_governor:
      target_params: 10M
    I6_output_surface:
      target_params: 50M
    update_port_reserve:
      target_params: 20M
```

Reference data and lesson budget:

```text
G1_reference_data_budget:
  gap_classes:
    surface:
      rows: 12000
    semantic:
      rows: 16000
    reasoning_verification:
      rows: 16000
    memory:
      rows: 12000
    compute_policy:
      rows: 12000
    collateral_damage_and_guardrails:
      rows: 20000

  splits:
    train: 60_percent
    ownership_eval: 20_percent
    collateral_damage_eval: 10_percent
    hidden_holdout: 10_percent

  teacher_query_budget:
    total_probe_examples: 60000
    max_teacher_queries_per_gap: 12000
    max_behavior_teacher_share: 40_percent
    minimum_non_behavior_teacher_share: 40_percent
    exact_validator_queries: uncapped_when_available_but_reported_as_cost

  packet_budget:
    candidate_packets: 12
    minimum_compiled_packets: 8
    minimum_fully_evaluated_packets: 5
    minimum_clear_failure_or_acceptance_decisions: 5
```

Reference training and evaluation budget:

```text
G1_reference_compute_budget:
  base_student_training:
    target_share: 45_percent
  gap_mapping_and_teacher_interrogation:
    target_share: 15_percent
  packet_training:
    target_share: 20_percent
  compute_governor_training:
    target_share: 10_percent
  ownership_and_damage_evaluation:
    target_share: 10_percent_reserved_before_training
```

Reference initial thresholds:

```text
G1_reference_thresholds:
  lesson_acceptance:
    retained_gain: > 0_on_hidden_holdout
    realized_lesson_value: > 0
    dependence_gap: <= 25_percent_of_teacher_assisted_gain
    collateral_damage: <= 2_percent_relative_on_guardrails
    inference_cost_delta: <= packet_declared_limit

  memory_lesson:
    retrieval_hit_improvement: >= 10_percent_relative
    hard_negative_confusion_drop: >= 10_percent_relative
    unnecessary_retrieval_increase: <= 3_percent_absolute

  semantic_lesson:
    paraphrase_stability_gain: >= 5_percent_absolute
    counterfactual_sensitivity_gain: >= 5_percent_absolute
    template_only_gain: reject_if_detected

  verification_lesson:
    violation_detection_gain: >= 10_percent_relative
    valid_output_overrepair: <= 2_percent_absolute_increase

  compute_lesson:
    easy_case_cost_reduction: >= 15_percent
    hard_case_quality_drop: <= 1_percent_absolute
    p95_cost_increase: <= 10_percent_unless_retained_gain_offsets_it

  G1_protocol_success:
    fully_evaluated_lessons: >= 5
    accepted_attributed_lessons_for_protocol_proven: >= 3
    accepted_lessons_for_protocol_falsified: no_minimum_if_failures_are_informative
    measurement_functions_used: >= 4
    update_ports_tested: >= 3
    lesson_types_attempted: >= 4
    protocol_result:
      allowed:
        - proven_if_multiple_attributed_lessons_succeed
        - falsified_if_lessons_fail_with_clear_failure_semantics
        - inconclusive_if_failures_are_unattributed_or_measurements_are_weak
```

These thresholds are not claims about nature. They are precommitments against
self-deception. They can be changed only by a doctrine update before seeing the
scores, or by a postmortem that explicitly marks the first threshold wrong.

## Implementation Work Packages

Still design-only, but exact enough that engineering can start without inventing
the plan.

```text
implementation_work_packages:
  W0_doctrine_and_registry_materialization:
    deliver:
      - interface_registry
      - lesson_type_registry
      - teacher_measurement_function_registry
      - G1_threshold_table

  W1_sutra_traceable_runtime:
    deliver:
      - Sutra_state_manifest
      - trace_schema_materialization
      - I0_to_I6_observable_paths
      - rule_based_compute_governor

  W2_gap_and_validator_suite:
    deliver:
      - five_gap_class_eval_slices
      - exact_validators_where_available
      - guardrail_and_collateral_damage_slices
      - ranked_gap_map

  W3_teacher_interrogation_layer:
    deliver:
      - teacher_registry
      - probe_factory
      - response_surface_recorder
      - redundancy_report

  W4_packet_compiler:
    deliver:
      - packet_records
      - expected_lesson_value_table
      - landing_zone_map
      - rollback_plans

  W5_local_update_engine:
    deliver:
      - memory_update_route
      - runtime_policy_update_route
      - verifier_head_update_route
      - interface_adapter_update_route
      - targeted_curriculum_update_route

  W6_ownership_evaluator:
    deliver:
      - retained_gain_report
      - dependence_gap_report
      - collateral_damage_report
      - cost_report
      - lesson_ledger

  W7_compute_efficiency_evaluator:
    deliver:
      - easy_hard_cost_report
      - p95_cost_report
      - hard_case_quality_report
      - adaptive_compute_return_report

  W8_revision_and_reproduction_package:
    deliver:
      - Sutra_revision_manifest
      - accepted_lesson_ledger
      - policy_refs
      - validator_refs
      - reproduction_trace_bundle
```

Engineering should proceed in this order. If a later work package becomes
tempting before an earlier one exists, the design is being bypassed.

## Path From G1 To General Usefulness

G1 proves the learning protocol, not general intelligence. The path from G1 to a
generally useful efficient system is a sequence of capability-frontier expansions
that preserve the same ownership doctrine.

```text
general_usefulness_ladder:
  G1_protocol_truth:
    domain:
      - byte exactness
      - semantic invariance
      - schema/arithmetic verification
      - memory retrieval
      - compute escalation
    claim:
      - Eklavya can create owned local lessons in Sutra

  G2_task_families:
    add:
      - code editing
      - factual QA with citations
      - tool-call planning
      - document transformation
    required_new_proof:
      - lessons transfer across adjacent task families without reintroducing
        teacher dependence

  G3_domain_specialization:
    add:
      - legal
      - scientific
      - software engineering
      - data analysis
    required_new_proof:
      - memory/update routes beat full retraining for domain changes

  G4_agentic_runtime:
    add:
      - multi-step tool use
      - long-horizon task state
      - sleep/precompute mode
      - self-repair loops
    required_new_proof:
      - compute governor improves cost and quality under long-horizon risk

  G5_deployment_profiles:
    add:
      - edge profile
      - workstation profile
      - server profile
      - offline/private profile
    required_new_proof:
      - same Sutra revision family can operate at different cost tiers by
        policy and adapter selection, not by becoming unrelated systems
```

General usefulness is not a single benchmark. It is the ability to keep
admitting owned lessons across task families while lifetime cost stays on the
efficient frontier.

## Final Design-Completion Audit

Before declaring the ground-up design complete, require evidence inside this
document or its successor for every line below.

```text
design_completion_audit:
  Sutra_identity:
    must_answer:
      - what Sutra is
      - what state it contains
      - how it runs inference
      - how it updates
      - how it proves efficiency
    current_status:
      - specified
    remaining_need:
      - cleared_by_final_adversarial_review

  Eklavya_identity:
    must_answer:
      - what Eklavya is
      - what teachers are
      - how teachers are admitted
      - how lessons are inferred
      - how ownership is proven
    current_status:
      - specified
    remaining_need:
      - cleared_by_final_adversarial_review

  non_distillation_proof:
    must_answer:
      - why this is not output imitation
      - why teacher averaging is not the default
      - how teacher removal is enforced
      - how lesson value is measured
    current_status:
      - specified
    remaining_need:
      - cleared_by_final_adversarial_review

  buildability:
    must_answer:
      - what to build first
      - what artifacts each stage emits
      - what gates must pass
      - when to pivot or reset
    current_status:
      - specified_in_builder_handoff
    remaining_need:
      - cleared_by_final_adversarial_review

  world_efficiency_claim:
    must_answer:
      - what efficiency means
      - which costs count
      - how adaptive compute is evaluated
      - how update and validation cost are counted
    current_status:
      - specified
    remaining_need:
      - cleared_by_final_adversarial_review
```

Current verdict:

```text
  ground_up_design_status:
  doctrine: strong
  protocol: strong
  schemas: adversarially_cleared_for_design_doctrine
  lifecycle_examples: sufficient_for_verification_compute_memory_and_semantics
  builder_handoff: now_specified
  budget_doctrine: specified
  efficiency_reporting: specified
  reference_build: specified
  implementation_work_packages: specified
  adversarial_review_status:
    latest_review: no_blocking_flaws_after_narrow_final_review
    current_patch_status: cleared_for_design_doctrine
  still_not_complete:
    - none_at_design_doctrine_level_after_final_adversarial_review
```

The design doctrine has survived the required adversarial review gate. Remaining
work is implementation and empirical calibration, not doctrine completion.

## The Hardest Open Problems

The original adversarial review correctly identified these as doctrine-level
risks, not mere implementation details. The patched doctrine above adds first
mechanisms for each, and the final adversarial pass found no remaining blocking
flaws. They remain high-risk implementation watchpoints.

```text
hard_open_problems:
  causal_lesson_inference:
    problem:
      - teacher response surfaces are correlational
      - Eklavya must infer what intervention will actually change Sutra
    required_solution:
      - probe families and ablations that distinguish cause from style

  module_credit_assignment:
    problem:
      - a lesson may appear to land in one module while another module carries
        the real change
    required_solution:
      - per-interface ablations and localized freeze tests

  compute_governor_alignment:
    problem:
      - a cheap-path policy may learn to avoid hard work rather than become
        efficient
    required_solution:
      - hard-case floors, risk-sensitive loss, and p95 reporting

  memory_weight_boundary:
    problem:
      - some knowledge belongs in external memory, some in weights, and some in
        runtime policy
    required_solution:
      - measured tradeoff between retrieval cost, update cost, and reliability

  teacher_corruption:
    problem:
      - teachers transfer biases, artifacts, refusal patterns, and shortcuts
    required_solution:
      - corruption probes and teacher-specific rejection rules

  protocol_overhead:
    problem:
      - lesson accounting can cost more than the lesson is worth
    required_solution:
      - use Eklavya only where expected value beats overhead
```

The point of writing these problems down is to keep the design honest. A future
Sutra cannot be declared world-class efficient if it only hides cost in protocol
overhead, memory, validators, or teacher preparation.

## The Short Doctrine

```text
Eklavya:
  learn from many teachers by extracting minimal causal lessons,
  not by copying their outputs.

Sutra:
  own those lessons in a compact adaptive runtime,
  not by becoming a smaller imitation of any teacher.

Proof:
  teacher removal, positive net retained gain, bounded cost,
  localized updates, and hard-case-preserving adaptive compute.
```

That is the ground-up system.
