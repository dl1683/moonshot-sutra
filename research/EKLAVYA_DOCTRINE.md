# Eklavya And Sutra Design Doctrine

> **Canonical reconciliation**: This document has been reconciled with
> `research/GROUND_UP_FUTURE_DESIGN.md` through R1-R6 adversarial deliberation.
> The unified build spec lives in **`research/SE1_CANONICAL_SPEC.md`**. For any
> conflict between this document and the canonical spec, the canonical spec wins.

This document is the active design doctrine for two things only:

1. **Eklavya**: a multiple-teacher learning system that extracts transferable
   invariants from diverse competent sources.
2. **Sutra**: the efficient student/system that uses Eklavya to become much
   more capable per parameter, token, joule, dollar, and human-hour than ordinary
   small models.

Everything else is subordinate. Meta-language is useful only when it helps choose
teachers, protect invariant transfer, reduce wasted compute, or prevent false
conclusions. It is not the project.

## Ruthless Reset

The previous meditation over-expanded into process machinery. That was a wrong
center of gravity.

The useful residue is small:

- do not let weak evidence steer architecture;
- do not mistake teacher count for teacher diversity;
- do not convert disagreement into an average;
- do not claim efficiency without measuring total cost;
- do not build mechanisms that cannot explain which outcome they improve.

All larger paper machinery is retired from the active design.

## The Core Thesis

Sutra should not win by being a smaller copy of a larger model. It should win by
having a better learning geometry.

The path is:

```text
many competent teachers
  -> teacher-specific invariant extraction
  -> structure-preserving compilation
  -> capacity-aware student ingestion
  -> retained competence after teacher removal
  -> adaptive inference that spends compute only where needed
```

Eklavya is the learning method. Sutra is the artifact that becomes possible if
the learning method works.

## What Eklavya Is

Eklavya is not ordinary knowledge distillation.

Ordinary KD asks: how can the student mimic a teacher's output distribution?

Eklavya asks: what hidden structure makes each teacher competent, and how can a
student re-express that structure in its own substrate without becoming dependent
on the teacher?

A teacher can be:

- a decoder language model;
- an encoder;
- a symbolic solver;
- a verifier;
- a search procedure;
- a proof trace;
- a memory system;
- a human-written curriculum;
- a simulator;
- a domain expert artifact;
- a strong model's reasoning trace;
- a negative example generator.

A teacher is useful only if it reveals something the student can own.

## What Transfers

The transferred object is not the teacher output. The transferred object is an
invariant.

Examples:

- token or byte probability structure;
- representation geometry;
- boundary or segmentation structure;
- difficulty maps;
- uncertainty maps;
- error localization;
- retrieval keys;
- proof obligations;
- latent planning steps;
- counterexample classes;
- domain-specific constraints;
- update rules that explain when not to spend compute.

The teacher output is evidence. The invariant is the lesson.

## What Must Not Transfer

Eklavya fails if it imports:

- teacher surface style without competence;
- teacher errors;
- teacher censorship or refusal quirks;
- tokenizer artifacts as semantic truth;
- correlated agreement between similar teachers;
- averaged distributions that erase minority expertise;
- benchmark-specific shortcuts;
- teacher dependence that disappears when the teacher is removed;
- extra inference cost disguised as student competence.

The student must end with owned structure, not a permanent teacher dependency.

## Why The Old Path Was Too Small

The legacy evidence points to several hard lessons:

1. **Two transformer teachers were not enough.** The observed KD ceiling was too
   small for a decisive "world's most efficient model" story.
2. **Arithmetic averaging was destructive.** It injected entropy and blurred
   teacher differences. Routing helped because it preserved conditional teacher
   usefulness.
3. **Student capacity was likely the bottleneck.** A 188M byte-global student
   paid a severe sequence tax and behaved more like a much smaller token-level
   model.
4. **Teacher diversity was only partially valid.** More teachers help only when
   they expose different invariants. Cross-tokenizer disagreement may be
   alignment noise rather than knowledge.
5. **Offline artifacts were useful but not magic.** Caches, profiles, and ports
   are tools for extracting structure, not proof that learning occurred.

The next design must not optimize the old KD path harder. It must change what
the student is and what counts as teacher signal.

## Sutra's Job

Sutra is the efficient learner produced by Eklavya.

Sutra should optimize for **lifetime intelligence per total cost**, not just
training loss or inference tokens.

Total cost includes:

- pretraining compute;
- teacher calls;
- teacher artifact storage;
- data curation;
- model parameters;
- active inference FLOPs;
- memory reads and writes;
- debugging cost;
- update cost;
- human supervision;
- benchmark and validation cost.

The target is not the lowest cost model. The target is the best capability per
full lifecycle cost under honest comparison.

## Sutra Architecture Direction

The strongest current direction is:

```text
byte/local boundary
  -> token/global reasoning substrate
  -> sparse or conditional capacity
  -> multi-resolution representations
  -> teacher-compiled curricula
  -> early-exit and self-verification
  -> retained teacher-free competence
```

### Token-Global, Byte-Local

Bytes are valuable at the boundary because they make cross-tokenizer teacher
alignment possible. But bytes are likely too expensive as the global reasoning
unit.

Sutra should separate:

- local byte encoder: spelling, morphology, boundary recovery, byte alignment;
- global token/patch reasoner: semantic and reasoning work;
- local byte decoder: exact surface realization.

This keeps byte precision without paying a byte-level global context tax.

### Conditional Capacity

Sutra should spend different compute on different content.

Candidate mechanisms:

- sparse FFN activation;
- mixture-of-depths token selection;
- early exit losses;
- self-speculative decoding;
- local/global routing;
- difficulty-conditioned teacher artifact use;
- matryoshka representations that allow coarse-to-fine use.

The rule: increase effective capacity without increasing average active cost.

### Teacher Ports

Every teacher family needs a port into student-native terms.

```text
teacher_port:
  teacher_family: decoder | encoder | verifier | symbolic | trace | human
  source_object: logits | hidden_state | proof | constraint | trace | example
  student_landing_zone: global_reasoner | local_encoder | local_decoder | memory | router
  invariant_claim: what structure this port exposes
  corruption_risk: what it may falsely transfer
  removal_test: how competence survives without teacher calls
```

Teacher ports should be small, typed, and inspectable. The port is not a data
dump. It is the interface through which an invariant can be learned.

### Eklavya Compiler

The learning pipeline should compile teachers into student-ingestible artifacts:

```text
raw_teacher_behavior
  -> teacher profile
  -> agreement/disagreement map
  -> invariant candidate
  -> compression target
  -> student training signal
  -> retained-gain test
```

Compiled artifacts may include:

- boundary maps;
- top-k distributions;
- contrast sets;
- hidden-state anchors;
- difficulty labels;
- counterexample clusters;
- retrieval keys;
- proof sketches;
- solution traces;
- verifier-localized error spans.

The compiler must preserve source identity. If a gain cannot be attributed to a
teacher family, it cannot prove Eklavya.

## Training Shape

A plausible Sutra/Eklavya training program:

1. Train a clean student base on high-quality compact data.
2. Profile teacher families on shared examples.
3. Identify where teachers disagree and where each teacher has unique edge.
4. Convert teacher signals into invariant packets.
5. Train the student with CE plus sparse, routed invariant objectives.
6. Remove teachers and test retained competence.
7. Add active compute pressure so easy cases exit cheaply.
8. Measure capability, cost, retained gain, and update locality.

The most important property is not initial teacher imitation. It is retained
student competence after the teacher is gone.

## Efficiency Definition

Sutra is more efficient only when it improves the frontier:

```text
efficiency =
  capability
  / (training_compute
     + inference_compute_over_expected_use
     + teacher_extraction_cost
     + data_cost
     + storage_cost
     + update_cost
     + validation_cost)
```

A technique that improves loss but requires constant teacher calls may be less
efficient. A technique that slightly underperforms but halves inference cost may
be more efficient. A technique that makes future improvement surgical has hidden
lifecycle value.

## The Design We Should Push

The current best design hypothesis is:

```text
Sutra-Eklavya v0:
  student:
    byte-local encoder/decoder
    token-or-patch global reasoner
    sparse FFN or conditional depth
    shared early-exit head
    compact memory/retrieval interface
  eklavya:
    teacher family registry
    teacher ports
    invariant compiler
    disagreement-driven curriculum
    source-specific contribution accounting
    teacher-removal retained-gain tests
  objective:
    CE base loss
    routed teacher-invariant losses
    representation geometry alignment only where low alignment is proven
    difficulty-aware compute pressure
    retained-gain regularization
```

The high-risk/high-upside part is not "add more teachers." It is building a
student that has places for different teacher invariants to land.

## Open Design Questions

1. What is the smallest student architecture with enough absorptive capacity for
   multi-family teacher structure?
2. Is token-global/byte-local the right split, or should the global unit be a
   learned patch rather than a token?
3. Which teacher families are genuinely complementary at small scale?
4. Can encoder teachers improve representations without wasting loss on already
   aligned regions?
5. Can verifier teachers teach error localization rather than final answers?
6. Can a symbolic teacher teach constraints that survive in neural weights?
7. Which active compute mechanism gives the cleanest efficiency gain with the
   least implementation complexity?
8. What is the minimum retained-gain test that proves teacher ownership?
9. How do we stop Eklavya from becoming ensemble compression?
10. What is the first design that could plausibly create a "David story" against
    larger models?

## Active Doctrine

Eklavya is the art of learning from many teachers without becoming an average of
them.

Sutra is the efficient student that becomes possible when the lessons are
compiled as invariants, placed into a capacity-aware architecture, and retained
after teacher removal.

## Concrete Design: Sutra-Eklavya SE1

The first design worth taking seriously is **SE1**: a byte-preserving,
patch-global student trained by teacher tomography rather than by averaged
teacher imitation.

The design principle:

```text
preserve bytes at the boundary
compress predictable spans into patches
reason globally over patches/tokens
keep teacher identities separate
train on invariant response surfaces
spend compute conditionally
test retained gain with teachers removed
```

### Student Substrate

```text
Sutra_SE1_student:
  input_channel:
    raw_bytes: always preserved
    learned_patches: primary global sequence
    residual_bytes: available when token/patch abstraction loses distinctions

  local_byte_encoder:
    responsibilities:
      - byte fidelity
      - morphology and spelling
      - cross-tokenizer alignment
      - entropy or boundary estimation
      - patch construction
    candidate_mechanisms:
      - BLT-style entropy patches
      - MEGABYTE-style local model inside patches
      - Charformer-style learned block scoring
      - n-gram hash features

  global_reasoner:
    responsibilities:
      - semantic composition
      - reasoning
      - long-range dependency handling
      - teacher-invariant absorption
    candidate_mechanisms:
      - compact transformer attention for content-based reasoning
      - selective SSM or recurrent lanes for cheap long context
      - sparse SwiGLU or conditional FFN for capacity without full activation
      - scale-separated states for hierarchy

  local_byte_decoder:
    responsibilities:
      - exact byte realization
      - surface repair
      - tokenizer-free output
      - byte-level teacher landing zone

  compute_controller:
    responsibilities:
      - early exit on easy positions
      - deeper processing on hard positions
      - self-speculative shallow draft plus deep verification
      - patch-level or token-level compute allocation
```

The key is that the global model should not be forced to pay for every byte.
Bytes remain the truth channel, but global reasoning happens on learned compact
units.

### Teacher Axis Must Survive

The earlier Eklavya mistake is to mix teachers too early. Once teachers are
averaged, the student can no longer know whether it learned a real invariant or
a blurred compromise.

SE1 therefore keeps a teacher axis:

```text
teacher_signature_tensor:
  example_id
  probe_id
  teacher_id
  teacher_family
  candidate_id
  probability_or_margin
  entropy
  disagreement_region
  landing_zone
  invariant_hypothesis
```

Training can route, weight, or contrast teachers, but the raw learning object is
not `p_mix`. It is a structured response surface.

### Teacher Families For SE1

The first serious teacher set should be small but heterogeneous:

```text
SE1_teacher_set:
  decoder_teacher:
    role: language distribution, reasoning traces, candidate margins
    landing_zone: global_reasoner
    signal: candidate-string logprob surfaces under probes

  encoder_teacher:
    role: semantic geometry, retrieval neighborhoods, paraphrase invariance
    landing_zone: global_reasoner and memory interface
    signal: pair/triplet ranking margins and representation anchors

  verifier_teacher:
    role: error localization, constraint satisfaction, answer checking
    landing_zone: verifier head and compute controller
    signal: pass/fail, localized error spans, constraint labels

  byte_boundary_teacher:
    role: segmentation, morphology, tokenizer-residual structure
    landing_zone: local_byte_encoder and local_byte_decoder
    signal: entropy boundaries, patch maps, byte residual targets

  curriculum_teacher:
    role: high-density examples and staged difficulty
    landing_zone: data scheduler
    signal: example quality, prerequisite graph, contrast sets
```

This is not "many teachers" for its own sake. It is a minimum set of different
information geometries.

### Teacher Tomography

The separate Eklavya repo contributes the right primitive: do not distill
decoded answers; distill behavior signatures under probes.

For a base input `x`, generate probes:

```text
probe_family:
  paraphrase_probe: should preserve answer
  distractor_probe: should preserve answer but change uncertainty
  counterfactual_probe: should change answer
  compression_probe: shorter equivalent wording
  corruption_probe: typo/noise/format disruption
  retrieval_probe: requires recalling earlier fact
  reasoning_probe: changes intermediate relation while preserving surface topic
```

Each teacher produces candidate margins or comparable measurements across those
probes. The invariant is the shape of the response surface:

```text
invariant = what remains stable, what changes, and how confidently it changes
```

This lets Eklavya learn:

- invariance, not only labels;
- sensitivity, not only final answers;
- calibration, not only correctness;
- disagreement regions, not only consensus.

### Invariant Packet Types

SE1 should compile teachers into a small number of packet types:

```text
invariant_packets_SE1:
  behavior_signature_packet:
    from: decoder or verifier teachers
    contains: candidate margins across probes
    trains: global decision geometry

  geometry_anchor_packet:
    from: encoder teachers
    contains: pair/triplet ranking, neighborhood stability, CKA target region
    trains: representation layout where student is not already aligned

  boundary_packet:
    from: byte/token/entropy teachers
    contains: patch boundaries, residual byte flags, morphology signals
    trains: local byte encoder and decoder

  error_localization_packet:
    from: verifier or symbolic teachers
    contains: wrong span, violated constraint, repair class
    trains: verifier head and compute controller

  curriculum_packet:
    from: human/model curriculum teachers
    contains: prerequisite, difficulty, contrastive examples
    trains: data order and active compute policy
```

Every packet must name a landing zone. If it cannot say where the invariant
lands in Sutra, it is not a training signal yet.

### Loss Shape

SE1 should not have one monolithic KD loss. It should have a small family of
landing-zone losses:

```text
SE1_loss:
  base_CE:
    purpose: language competence and grounding in real text

  behavior_signature_KL:
    purpose: match teacher response surfaces over candidate strings
    applies_to: selected probes and candidate sets
    teacher_axis: preserved until routing

  contrastive_invariance_loss:
    purpose: preserve stable answers under invariance probes and change under
             counterfactual probes

  representation_margin_loss:
    purpose: absorb encoder geometry only where student lacks it

  boundary_prediction_loss:
    purpose: learn patching and byte residual needs

  verifier_span_loss:
    purpose: locate errors and constraints

  compute_penalty:
    purpose: reward early exit or skipped depth when quality is preserved

  retained_gain_regularizer:
    purpose: penalize teacher-dependent gains that disappear after teacher removal
```

The loss is not "more objectives because more is better." Each objective maps to
one of Sutra's necessary interfaces.

### Training Phases

```text
SE1_training_phases:
  phase_0_base_student:
    train: byte-local / patch-global student on compact high-quality data
    goal: stable base competence and patch interface

  phase_1_teacher_tomography:
    freeze: student architecture decision
    collect: teacher signatures under probes
    output: teacher_signature_tensor and invariant packets

  phase_2_landing_zone_training:
    train: packet-specific losses by landing zone
    avoid: early teacher averaging
    measure: contribution by teacher family

  phase_3_active_compute_training:
    train: early exit, skip, or self-speculative controller
    signal: difficulty maps, verifier spans, entropy boundaries
    measure: quality per expected active FLOP

  phase_4_teacher_removal:
    remove: teacher calls and teacher artifacts from inference
    test: retained gain by teacher family and packet type

  phase_5_surgical_update:
    introduce: one new teacher family or packet type
    test: whether a localized update improves target skill without global damage
```

Phase 4 is the soul of Eklavya. If teacher removal collapses the gain, the
student did not learn.

### Retained Gain

Define:

```text
retained_gain =
  performance_after_teacher_removal
  - performance_before_teacher_exposure
```

But the raw difference is not enough. The record must be sliced by:

- teacher family;
- packet type;
- landing zone;
- task family;
- easy versus hard examples;
- inference compute tier;
- update locality.

The important question is not "did the model improve?" It is "which teacher
invariant became owned structure?"

### Efficiency Target

SE1 should report:

```text
lifecycle_efficiency_SE1 =
  retained_capability_gain
  / (base_training_cost
     + teacher_signature_cost
     + artifact_storage_cost
     + packet_training_cost
     + expected_inference_cost
     + validation_cost
     + update_cost)
```

This makes teacher cost visible. A teacher is good only if it buys retained
student competence at a better lifecycle price than alternatives.

### Why This Could Create The David Story

SE1 has a plausible path to beating larger ordinary small models because it
attacks several waste sources at once:

- byte fidelity without byte-global sequence tax;
- teacher diversity without teacher averaging;
- curriculum density instead of raw data volume;
- conditional compute instead of full-depth inference;
- invariant transfer instead of output mimicry;
- retained-gain testing instead of teacher-dependent demos;
- component landing zones instead of opaque whole-model improvement.

The story is not "small model imitates big model." The story is:

```text
small model receives better-shaped lessons,
stores them in better-shaped geometry,
and spends compute only where needed.
```

## SE1 Module Interface Record

SE1 needs a concrete object language before implementation. The goal is not to
freeze dimensions; it is to make every future module justify what it consumes,
what it emits, and which teacher invariant can land there.

Notation:

```text
B = batch size
Tb = byte length
P = patch length after local compression
D = global hidden width
L = global depth
M = teacher count
G = probe count
C = candidate count
S = scale count for optional multi-scale states
```

### Input And Patching Interface

```text
raw_byte_batch:
  bytes: uint8[B, Tb]
  attention_mask: bool[B, Tb]
  document_or_sequence_id: id[B]

local_byte_encoder_output:
  patch_states: float[B, P, D]
  patch_spans: int[B, P, 2]
  patch_entropy: float[B, P]
  residual_flags: bool[B, Tb]
  byte_side_state: float[B, Tb, d_local]
```

The local encoder has three duties:

1. preserve exact byte truth;
2. shorten the global sequence;
3. expose boundary uncertainty so active compute and teachers can target the
   hard places.

The patcher is allowed to be learned, entropy-based, or hybrid. It is not
allowed to erase byte identity.

### Global Reasoner Interface

```text
global_reasoner_output:
  layer_states: float[B, L, P, D]
  final_patch_states: float[B, P, D]
  optional_scale_states: float[B, S, P, d_scale]
  exit_logits_by_layer: float[B, L, P, V_or_C]
  difficulty_state: float[B, P, d_diff]
```

The global reasoner is where decoder teachers, encoder teachers, curriculum
teachers, and symbolic/verifier teachers should mostly land. It is the semantic
substrate.

The design should prefer attention for content-based reasoning and allow cheap
long-context lanes only where they reduce cost without losing reasoning quality.
SSM-like or recurrent lanes are support lanes, not replacements for semantic
attention unless proven otherwise.

### Decoder And Exactness Interface

```text
local_byte_decoder_output:
  byte_logits: float[B, Tb, 256]
  candidate_logits: float[B, G, C]
  reconstruction_loss_mask: bool[B, Tb]
  surface_repair_state: float[B, Tb, d_local]
```

The local decoder protects the project from the tokenization-limit failure:
global reasoning may compress, but exact output remains byte-grounded.

### Compute Controller Interface

```text
compute_controller_output:
  exit_layer: int[B, P]
  process_mask_by_layer: bool[B, L, P]
  draft_accept_probability: float[B, P]
  expected_active_flops: float[B]
  hard_case_flag: bool[B, P]
```

The compute controller should be trained from three sources:

- patch entropy from the byte side;
- teacher disagreement and uncertainty from tomography;
- verifier error localization.

The controller is not a speed trick after training. It is part of Sutra's
intelligence: knowing when not to think is a learned competence.

## SE1 Packet Format

Every Eklavya packet should have this common envelope:

```text
SE1_packet_envelope:
  packet_id:
  packet_type:
  source_teacher_id:
  source_teacher_family:
  example_id:
  probe_id:
  landing_zone:
  invariant_claim:
  corruption_risk:
  target_loss:
  cost_to_create:
  removal_test:
```

Specific payloads:

```text
behavior_signature_payload:
  candidates: string[C]
  teacher_margins: float[C]
  teacher_distribution: float[C]
  entropy: float
  probe_relation: invariant | sensitive | counterfactual

geometry_anchor_payload:
  positive_ids: id[]
  negative_ids: id[]
  teacher_rank_margins: float[]
  neighborhood_stability: float
  use_only_if_student_gap: bool

boundary_payload:
  byte_spans: int[P, 2]
  entropy_targets: float[P]
  residual_required: bool[Tb]
  tokenizer_disagreement_flag: bool[P]

error_localization_payload:
  violated_constraint:
  error_span: int[2]
  repair_class:
  verifier_confidence:
  compute_escalation_hint:

curriculum_payload:
  prerequisite_ids: id[]
  contrast_set_ids: id[]
  difficulty_level:
  should_repeat_after_gap:
  expected_landing_zone:
```

This is where Eklavya becomes more than "several teachers." It is a compiler
from teacher-specific behavior into landing-zone-specific learning objects.

## Retained Gain And Ownership

SE1 needs to distinguish four numbers:

```text
base_score = score(student_before_teacher_packets)
during_score = score(student_with_teacher_artifacts_available_if_any)
after_score = score(student_after_training_with_no_teacher_calls_at_inference)
control_score = score(matched_control)
```

The useful quantities are:

```text
owned_gain = after_score - base_score
teacher_dependence_gap = during_score - after_score
control_adjusted_gain = after_score - control_score
retention_ratio = owned_gain / max(during_score - base_score, epsilon)
```

A teacher packet is valuable only if:

- `owned_gain > 0`;
- `control_adjusted_gain > 0`;
- `teacher_dependence_gap` is small enough to show the student owns the skill;
- collateral damage on unrelated tasks is bounded;
- lifecycle cost is lower than alternative routes to the same gain.

This is the difference between Eklavya and a disguised ensemble.

## First Honest Comparison Suite

SE1 should be compared against:

```text
SE1_comparison_suite:
  B0_CE_only_same_student:
    purpose: prove teacher packets add value beyond data and architecture

  B1_best_single_decoder_teacher:
    purpose: prove multiple teachers beat the strongest single teacher

  B2_naive_teacher_average:
    purpose: prove teacher-axis preservation beats early averaging

  B3_byte_global_student:
    purpose: prove patch-global design beats byte-global sequence tax

  B4_no_active_compute_SE1:
    purpose: isolate inference-efficiency gain

  B5_no_boundary_packet:
    purpose: prove byte boundary teaching matters

  B6_no_encoder_packet:
    purpose: prove geometry anchors are not redundant

  B7_no_verifier_packet:
    purpose: prove error localization teaches more than answers

  B8_no_curriculum_packet:
    purpose: prove data ordering and contrast sets matter
```

If SE1 cannot beat these controls, it is not the right first Sutra.

## SE1 Doctrine Update

The concrete doctrine is now:

```text
Sutra_SE1_doctrine:
  bytes_are_truth
  patches_are_compute_units
  teacher_behavior_is_a_surface
  teacher_axis_must_survive_until_routing
  every_invariant_needs_a_landing_zone
  retained_gain_is_the_core_learning_signal
  efficiency_is_lifecycle_capability_per_cost
```

## SE1 First Build Target

The first Sutra should be small enough to expose architecture truth before scale
hides mistakes. Do not jump to a 3B student. The legacy attempts already showed
that a roughly 188M byte-global student can hit a capacity and sequence-tax
wall; SE1 should test whether a byte-preserving patch-global substrate and
teacher tomography change that frontier.

```text
SE1_first_scale_target:
  student_total_params: 180M_to_350M
  active_params_typical: 90M_to_180M
  global_sequence_unit: learned_patch_or_token_compatible_patch
  context_target: modest_first_do_not_chase_128k
  training_budget_posture: small_scout_then_scale
  teacher_calls: offline_signature_generation_only
  inference_teacher_calls: zero
```

The point of this target is not to win leaderboards. It is to learn whether the
Sutra/Eklavya shape gives retained capability per unit of training and inference
cost that a simpler small model cannot get.

## First Teacher Roster

The first roster should maximize teacher diversity, not teacher prestige.

```text
SE1_teacher_roster_v0:
  decoder_teacher_primary:
    candidates:
      - decoder-candidate-4B
      - anchor-decoder-1.7B
    purpose: candidate_behavior_surfaces_reasoning_style_calibration

  decoder_teacher_efficiency_reference:
    candidates:
      - HuggingFaceTB/SmolLM3-3B
      - HuggingFaceTB/SmolLM3-3B-Base
    purpose: small_efficient_decoder_peer_and_curriculum_reference

  decoder_teacher_small_control:
    candidates:
      - control-decoder-0.6B
    purpose: cheap_peer_teacher_and_scale_control

  encoder_teacher_primary:
    candidates:
      - embedding-candidate-0.6B
      - BAAI/bge-m3
    purpose: geometry_anchor_and_retrieval_paraphrase_surface

  encoder_teacher_small_control:
    candidates:
      - BAAI/bge-small-en-v1.5
    purpose: detect_when_encoder_teacher_scale_is_irrelevant

  verifier_teacher:
    candidates:
      - task_local_exact_checker
      - unit_test_oracle
      - constraint_validator
      - symbolic_algebra_checker
    purpose: error_localization_and_pass_fail_truth

  byte_boundary_teacher:
    candidates:
      - BLT_style_patch_entropy
      - MEGABYTE_style_local_global_split
      - Charformer_style_subword_learning
      - tokenization_limit_adversarial_cases
    purpose: byte_boundary_and_residual_byte_decisions

  curriculum_teacher:
    candidates:
      - textbook_quality_synthetic_examples
      - tinystories_style_minimal_worlds
      - rationale_and_contrast_set_examples
    purpose: high_density_data_ordering_and_intermediate_supervision
```

The decoder teachers teach behavior surfaces. The encoder teachers teach
geometry, retrieval, and paraphrase invariants. The verifier teachers teach
where answers break. The byte-boundary teachers teach what abstraction must not
destroy. The curriculum teachers teach order and density. Eklavya is the system
that makes those differences survive long enough for the student to own them.

## First Task Slices

SE1 should not begin with a vague benchmark basket. It needs slices that map to
the design claims.

```text
SE1_task_slices_v0:
  byte_exactness_slice:
    tests: unicode_paths_code_literals_tokenization_adversaries
    proves: bytes_are_truth

  candidate_reasoning_probe_slice:
    tests: controlled_multiple_choice_counterfactual_distractor_sets
    proves: behavior_signature_packets_add_retained_gain

  retrieval_paraphrase_geometry_slice:
    tests: paraphrase_retrieval_dense_sparse_multivector_agreement
    proves: encoder_geometry_packets_are_not_redundant

  verifier_error_span_slice:
    tests: code_unit_tests_math_constraints_structured_validation
    proves: verifiers_teach_localization_not_just_labels

  generation_coherence_slice:
    tests: short_instruction_generation_and_style_transfer
    proves: patch_global_student_keeps_language_quality

  active_compute_easy_hard_slice:
    tests: matched_easy_hard_pairs_by_entropy_disagreement_and_verifier_risk
    proves: compute_controller_saves_cost_without_hard_case_damage
```

Each slice must report retained gain and lifecycle cost, not just accuracy.

## First Loss Schedule

SE1 should not optimize every packet from step one. The schedule should create a
stable student substrate first, then teach cross-teacher invariants, then remove
teachers and measure ownership.

```text
SE1_loss_schedule_v0:
  stage_0_base_student:
    active_losses:
      - base_CE
      - byte_reconstruction
    rule: learn_raw_language_and_byte_fidelity_before_teacher_packets

  stage_1_boundary_stabilization:
    active_losses:
      - boundary_prediction_loss
      - residual_byte_loss
    rule: stabilize_patch_boundaries_before_global_packet_pressure

  stage_2_behavior_signature_training:
    active_losses:
      - behavior_signature_KL
      - margin_rank_loss
      - calibration_loss
    rule: preserve_teacher_axis_do_not_average_early

  stage_3_packet_landing_alternation:
    active_losses:
      - contrastive_invariance_loss
      - representation_margin_loss
      - verifier_span_loss
      - curriculum_order_loss
    rule: alternate_packet_families_and_measure_collision

  stage_4_gradient_conflict_control:
    active_methods:
      - uncertainty_weighting_candidate
      - GradNorm_candidate
      - PCGrad_candidate
    rule: use_only_if_gradient_conflict_metrics_require_it

  stage_5_active_compute_training:
    active_losses:
      - compute_penalty
      - early_exit_consistency
      - hard_case_recall_loss
    rule: make_easy_cases_cheaper_without_erasing_hard_cases

  stage_6_teacher_removal:
    active_metrics:
      - owned_gain
      - teacher_dependence_gap
      - control_adjusted_gain
      - retention_ratio
    rule: no_teacher_calls_at_inference_and_no_hidden_ensemble
```

This schedule is intentionally conservative. If the substrate is unstable,
teacher signals become noise. If teacher signals are averaged early, Eklavya
becomes ordinary distillation. If retained gain is not measured after teacher
removal, Sutra is not being built.

## Cost Accounting Doctrine

Efficiency must include the cost of acquiring teacher signal. The first SE1
paper design must price:

```text
SE1_cost_accounting_v0:
  student_train_tokens: required
  student_train_FLOPs: required
  teacher_signature_examples: required
  teacher_signature_teacher_seconds: required
  teacher_signature_token_count: required
  packet_storage_bytes: required
  active_inference_FLOPs_mean: required
  active_inference_FLOPs_p95: required
  validation_cost: required
  update_cost: required
  teacher_free_inference_required: true
```

Teacher signal is allowed only if it buys retained student capability more
cheaply than extra student scale, extra training tokens, retrieval at inference,
or a hand-written specialist.

## SE1 As Compiler, Student, And Runtime

The deeper shape is not "train a small model with many teachers." That is still
too close to old KD.

SE1 should be understood as three coupled systems:

```text
Eklavya_teacher_profiler:
  role: observe competent systems under controlled probes
  output: teacher_signature_tensor

Eklavya_invariant_compiler:
  role: convert teacher behavior into student-native packets
  output: invariant_packets_with_landing_zones

Sutra_runtime_planner:
  role: choose how much student compute to spend at inference
  output: teacher_free_answers_under_cost_budget
```

The teacher profiler is analogous to profile-guided optimization in compilers:
representative executions create profiles, and later builds use the profile to
make better static decisions. The invariant compiler is analogous to a database
optimizer: a high-level intent can have many equivalent physical plans, so the
system chooses the plan with the lowest estimated cost. The runtime planner is
the neural version of that idea: for each patch or decision, choose shallow
exit, deeper processing, verifier escalation, memory lookup, or byte-level
repair according to expected marginal utility.

This framing changes what SE1 must learn. It does not merely learn facts. It
learns:

- where information is fragile;
- which teacher is informative for which failure mode;
- which representation level should carry the lesson;
- which examples deserve expensive computation;
- which skills survive after teacher artifacts are removed.

## SE1 Module Diagram

```text
offline_teacher_side:
  raw_task_or_text_examples
    -> probe_generator
    -> teacher_family_ports
    -> teacher_signature_tensor
    -> disagreement_and_gap_analyzer
    -> invariant_packet_compiler
    -> packet_ledger

student_training_side:
  raw_byte_stream
    -> local_byte_profiler
    -> patch_builder
    -> patch_tape
    -> global_reasoner
    -> local_byte_decoder
    -> base_losses

packet_landing_side:
  packet_ledger
    -> packet_scheduler
    -> landing_zone_router
    -> packet_specific_losses
    -> retained_gain_evaluator

teacher_free_inference_side:
  raw_bytes
    -> local_byte_profiler
    -> patch_builder
    -> global_reasoner_with_early_exits
    -> cost_based_compute_controller
    -> optional_self_verification
    -> local_byte_decoder
    -> exact_bytes_or_structured_answer
```

The active inference path has no teacher calls and no teacher artifact lookup.
Teacher artifacts are training-time profiles. If inference depends on them, SE1
has become a disguised retrieval or ensemble system rather than Sutra.

## The Cost-Based Compute Planner

Sutra's compute controller should be treated as a planner with actions, state,
cost, and expected benefit.

```text
SE1_compute_planner:
  state_features:
    - patch_entropy
    - teacher_disagreement_score_seen_during_training
    - early_exit_margin
    - verifier_risk_score
    - byte_residual_required
    - retrieval_or_memory_need
    - current_layer
    - estimated_remaining_uncertainty

  actions:
    - exit_now
    - process_next_layer
    - process_sparse_ffn
    - invoke_self_speculative_head
    - escalate_to_verifier_head
    - request_local_byte_repair
    - read_internal_memory

  objective:
    maximize: expected_quality_gain_minus_lambda_cost
    subject_to:
      - hard_case_quality_floor
      - byte_exactness_floor
      - no_teacher_at_inference
```

The planner should inherit difficulty labels from Eklavya tomography. Teacher
disagreement is not only a training target; it is also a profile of where future
student computation is likely to be worth spending.

## First Dataset And Task Manifest

The first manifest should be small, typed, and designed to isolate the SE1
claims. It should not be a general benchmark suite.

```text
SE1_first_manifest_doctrine:
  base_text_slice:
    purpose: language and byte-grounded fluency
    size_posture: scout_scale_before_full_scale
    design_rule: high_quality_text_beats_raw_volume_for_first_design_truth

  byte_exactness_slice:
    purpose: prove residual bytes are necessary and useful
    examples:
      - unicode_confusables
      - code_literals
      - path_strings
      - rare_names
      - tokenizer_collision_cases

  candidate_reasoning_slice:
    purpose: exercise teacher tomography on constrained candidate spaces
    examples:
      - ARC_style_multiple_choice
      - counterfactual_variants
      - distractor_variants
      - paraphrase_variants

  retrieval_geometry_slice:
    purpose: test encoder teacher packets
    examples:
      - paraphrase_pairs
      - hard_negative_pairs
      - delayed_recall_pairs
      - domain_shifted_neighbors

  verifier_slice:
    purpose: train error localization and repair, not answer imitation
    examples:
      - unit_tested_code_fragments
      - arithmetic_or_algebra_constraints
      - JSON_schema_constraints
      - contradiction_detection

  generation_slice:
    purpose: protect language quality while optimizing efficiency
    examples:
      - short_instruction_following
      - style_preserving_rewrite
      - compact_explanation
      - constrained_format_generation

  active_compute_slice:
    purpose: train and audit easy/hard compute allocation
    examples:
      - paired_easy_hard_versions_from_each_slice
      - entropy_matched_controls
      - teacher_disagreement_matched_controls
```

The manifest is deliberately built around mechanisms. Every row should be able
to answer: which SE1 claim does this example test?

## Teacher Selection Gates

The first roster should be admitted through gates rather than preference.

```text
SE1_teacher_selection_gates:
  distinct_surface_gate:
    question: does this teacher behave differently under probes?
    reject_if: agreement_is_only_same_family_echo

  student_gap_gate:
    question: does the student lack the structure this teacher exposes?
    reject_if: current_student_already_has_high_alignment_or_skill

  landing_zone_gate:
    question: where in Sutra can this invariant live?
    reject_if: no_specific_landing_zone

  cost_gate:
    question: is the teacher signal cheaper than extra data_or_scale?
    reject_if: teacher_signature_cost_exceeds_expected_retained_gain

  removal_gate:
    question: can gain survive without teacher calls?
    reject_if: benefit_requires_teacher_or_artifact_at_inference
```

This is the defense against teacher collecting. A teacher that cannot pass these
gates is not an Eklavya teacher for SE1.

## Deeper Principle: Teachers Are Sensors, Not Masters

The right analogy is not school. It is instrumentation.

A decoder teacher is a sensor for candidate behavior. An encoder teacher is a
sensor for semantic geometry. A verifier is a sensor for constraint violation. A
byte-boundary source is a sensor for representational loss. A curriculum source
is a sensor for example order and density.

Eklavya is the system that turns sensor readings into owned structure.

That means the highest-leverage question is:

```text
what hidden state of the world_or_task does this teacher measure
that the student cannot cheaply infer yet?
```

If a teacher does not measure a hidden state the student needs, it is noise. If
many teachers measure the same hidden state with correlated errors, they are one
teacher for Eklavya purposes. If a small exact checker measures the hidden state
better than a giant language model, the checker is the better teacher.

## What Would Make Sutra World-Class Efficient

Sutra becomes the world's most efficient model only if it compounds several
orthogonal efficiencies:

```text
Sutra_efficiency_stack:
  data_efficiency:
    source: high_density_curricula_and_teacher_profiles
    proof: fewer_training_tokens_for_same_retained_gain

  representation_efficiency:
    source: byte_local_patch_global_substrate
    proof: shorter_global_sequence_without_byte_exactness_loss

  capacity_efficiency:
    source: deep_thin_small_model_design_sparse_or_shared_blocks
    proof: better_quality_per_active_parameter

  inference_efficiency:
    source: early_exit_dynamic_depth_self_verification
    proof: lower_expected_FLOPs_with_hard_case_quality_floor

  update_efficiency:
    source: landing_zone_packets_and_component_slices
    proof: new_teacher_or_skill_updates_are_local_and_low_damage

  validation_efficiency:
    source: retained_gain_slices_and_source_specific_attribution
    proof: failures_are_diagnosed_without_benchmark_sprawl
```

No single trick is enough. The world-class claim requires the stack.

## SE1 Utility Calculus

The next level of rigor is to make every candidate signal compete for budget.
SE1 needs an internal economics, not a pile of plausible mechanisms.

The central quantity is **net retained gain**:

```text
net_retained_gain =
  expected_owned_gain
  - collateral_damage_penalty
  - teacher_dependence_penalty
  - lifecycle_cost_penalty
  - optimization_instability_penalty
```

Where:

```text
expected_owned_gain:
  improvement_expected_after_teacher_removal

collateral_damage_penalty:
  loss_on_previous_or_unrelated_capabilities

teacher_dependence_penalty:
  apparent_gain_that_requires_teacher_or_packet_at_inference

lifecycle_cost_penalty:
  normalized_cost_of_teacher_calls_storage_training_inference_validation_update

optimization_instability_penalty:
  probability_weighted_cost_of_loss_conflict_or_training_collapse
```

Every teacher, packet, dataset slice, loss, module, and compute action should be
ranked by this quantity. If a component cannot be expressed in these terms, it
is not ready for SE1.

## Eklavya As Machine Teaching

Eklavya is closer to machine teaching than to ordinary distillation.

Ordinary training asks:

```text
given_data_and_loss -> what model is learned?
```

Machine teaching asks:

```text
given_learner_and_target_structure -> what smallest_lesson_teaches_it?
```

Eklavya's version is:

```text
given_student_substrate_and_teacher_sensors
  -> what packet_sequence_teaches_owned_invariants
     with_minimum_lifecycle_cost?
```

This matters because the first SE1 curriculum should not be iid. It should be a
constructed lesson. The best examples are not average examples; they are examples
that collapse the student's uncertainty about a target invariant.

This changes the job of the teacher:

- not to provide more labels;
- not to provide a bigger probability vector;
- not to demonstrate superiority;
- but to reveal the smallest set of probes and contrasts that causes the
  student to acquire a durable structure.

## Packet Value Of Information

A teacher packet is an information purchase. It should be priced before it is
created, and audited after training.

```text
packet_value_prior =
  P(student_gap_is_real)
  * P(packet_lands_in_target_zone)
  * P(gain_survives_removal)
  * estimated_capability_delta
  * complementarity_score
  - estimated_packet_cost
  - estimated_corruption_risk
```

After training:

```text
packet_value_realized =
  control_adjusted_gain
  * retention_ratio
  * hard_case_preservation
  * update_locality_score
  - measured_lifecycle_cost
  - measured_collateral_damage
```

The gap between prior and realized value is also useful:

```text
packet_forecast_error =
  packet_value_prior - packet_value_realized
```

High forecast error means the compiler does not yet know what teachers are worth.
This should update teacher selection, not be hidden behind aggregate scores.

## Probe Selection Rule

Teacher tomography should not use probes because they are interesting. It should
use probes because they maximize expected teaching value.

```text
probe_value =
  expected_information_gain_about_student_gap
  + expected_information_gain_about_teacher_complementarity
  + expected_information_gain_about_landing_zone
  - probe_generation_cost
  - teacher_scoring_cost
  - ambiguity_penalty
```

The best probe is often not the one that produces maximum disagreement. Maximum
disagreement can be alignment noise. The best probe is the one that tells us
which invariant is teachable, where it should land, and whether the student will
own it after teacher removal.

This creates a three-stage probe loop:

```text
SE1_probe_loop:
  scout:
    goal: find where student and teachers disagree

  isolate:
    goal: determine whether disagreement is knowledge, noise, style, or artifact

  teach:
    goal: construct the smallest packet sequence that transfers the invariant
```

## Capacity Allocation Rule

The student should allocate parameters and FLOPs by marginal net retained gain.

```text
capacity_allocation_rule:
  add_capacity_to_module_m_if:
    marginal_net_retained_gain_per_param(m)
      > marginal_net_retained_gain_per_param(all_alternatives)

  spend_inference_compute_on_action_a_if:
    expected_quality_delta(a)
      - lambda * expected_cost(a)
      > 0
```

This makes the parameter split a hypothesis, not an identity claim.

First prior:

```text
SE1_parameter_split_prior:
  total_student_params: 260M_nominal

  local_byte_encoder:
    target: 20M_to_35M
    reason: bytes_and_boundaries_need_precision_but_not_global_reasoning

  global_reasoner:
    target: 170M_to_220M
    reason: semantic_and_reasoning_capacity_are_the_main_bottleneck

  local_byte_decoder:
    target: 30M_to_50M
    reason: legacy_patch_size_evidence_says_decoder_burden_matters

  compute_controller_and_heads:
    target: 5M_to_15M
    reason: planner_must_be_cheap_or_it_erases_inference_efficiency

  optional_memory_or_adapter_space:
    target: 0M_to_20M
    reason: admit_only_after_retained_gain_or_update_locality_gap_is_measured
```

If the local decoder is too small, patches must shrink and global sequence cost
rises. If the local decoder is larger, patches can grow, but the decoder can
become a hidden second model. The first scout should therefore treat patch size
and local decoder capacity as a coupled design variable.

## Patch Size And Decoder Capacity

The legacy evidence strongly suggests patch size is not a cosmetic
hyperparameter. It is the exchange rate between local byte difficulty and global
reasoning cost.

The design equation is:

```text
total_cost(P, C_decoder) =
  global_reasoning_cost(P)
  + local_decoding_cost(P, C_decoder)
  + byte_exactness_penalty(P, C_decoder)
  + hard_case_penalty(P, C_decoder)
```

Where smaller `P` makes local decoding easier but increases global sequence
length, and larger `P` makes global reasoning cheaper but pushes more burden
onto the local decoder.

SE1 should therefore begin with a patch-size ladder:

```text
SE1_patch_ladder:
  P4:
    role: high_resolution_control
    expected: strong_byte_quality_high_global_cost

  P6:
    role: middle_default
    expected: balance_for_first_student

  P8_or_dynamic:
    role: strong_decoder_test
    expected: lower_global_cost_if_decoder_can_handle_local_burden

  entropy_dynamic:
    role: final_candidate_if_fixed_P_tradeoff_is_visible
    expected: spend_boundaries_where_uncertainty_is_high
```

Dynamic patching should not be first because it can hide the basic tradeoff. It
should enter after fixed-patch scouts show where the tradeoff sits.

## Loss Weight And Stage-Length Priors

Exact loss weights are an empirical question, but the first priors should obey
one rule: never let packet losses destabilize the base student.

```text
SE1_training_budget_prior:
  stage_0_base_student:
    share_of_initial_scout_compute: 50_to_65_percent
    packet_loss_weight: 0

  stage_1_boundary_stabilization:
    share_of_initial_scout_compute: 10_to_15_percent
    boundary_loss_weight: ramp_0_to_0_15_of_base_gradient_norm

  stage_2_behavior_signature:
    share_of_initial_scout_compute: 10_to_15_percent
    behavior_loss_weight: ramp_0_to_0_20_of_base_gradient_norm

  stage_3_packet_landing:
    share_of_initial_scout_compute: 10_to_20_percent
    total_packet_gradient_budget: at_most_0_30_of_base_gradient_norm

  stage_4_active_compute:
    share_of_initial_scout_compute: 5_to_10_percent
    compute_penalty: ramp_only_after_quality_floor_is_met

  stage_5_teacher_removal_eval:
    share_of_initial_scout_compute: evaluation_only
```

The first loss-balancing rule is gradient budgeting:

```text
packet_gradient_norm_sum <= 0.30 * base_gradient_norm
```

until a packet family has positive control-adjusted retained gain. Only proven
packet families earn more gradient budget.

## Promotion Gates

SE1 should advance only by gates that correspond to real claims.

```text
SE1_promotion_gates:
  gate_0_substrate:
    pass_if:
      - patch_global_beats_byte_global_under_matched_budget
      - byte_exactness_floor_met
      - generation_not_collapsed

  gate_1_teacher_profile:
    pass_if:
      - at_least_three_teacher_families_have_distinct_probe_surfaces
      - one_non_decoder_family_passes_student_gap_gate
      - teacher_signature_cost_is_reported

  gate_2_packet_landing:
    pass_if:
      - at_least_two_packet_families_show_positive_control_adjusted_gain
      - no_packet_family_causes_unbounded_collateral_damage
      - naive_average_loses_to_teacher_axis_preservation

  gate_3_retained_gain:
    pass_if:
      - owned_gain_positive_after_teacher_removal
      - teacher_dependence_gap_bounded
      - best_single_teacher_control_beaten

  gate_4_compute_efficiency:
    pass_if:
      - expected_active_FLOPs_lower_than_static_depth
      - hard_case_quality_floor_preserved
      - p95_cost_reported

  gate_5_update_locality:
    pass_if:
      - adding_one_teacher_or_packet_improves_target_slice
      - collateral_damage_below_threshold
      - update_cost_below_full_retrain_baseline
```

The strongest world-class claim requires all gates. A partial pass may still be
useful, but it is not "the most efficient model in the world."

## Readiness Doctrine

SE1 is now concrete enough to be dangerous. That means the project needs a
readiness doctrine, not more excitement.

The next phase is allowed to become engineering only when the following record
exists:

```text
SE1_readiness_record:
  dataset_manifest:
    status: exact_sources_license_size_and_slice_rules_written

  teacher_roster:
    status: model_ids_revisions_runtime_modes_and_costs_written

  substrate_scout:
    status: patch_choices_baselines_and_hidden_compute_controls_written

  packet_plan:
    status: every_packet_names_landing_zone_loss_corruption_risk_removal_test

  retained_gain_plan:
    status: teacher_removal_metrics_and_controls_written_before_training

  promotion_gates:
    status: numeric_thresholds_written_before_scores_are_seen
```

If one field is missing, the project is still doing design, not implementation.

## Dataset Doctrine

Datasets are not authority. They are instruments.

SE1 should admit a dataset only if it answers one of these questions:

- does the student substrate preserve bytes while shortening global sequence
  length?
- does a teacher family expose a student gap?
- does a packet land in a named module?
- does retained gain survive without teacher artifacts?
- does adaptive compute reduce expected cost without hard-case damage?
- does an update improve one target without global collateral damage?

A dataset row is useful when it changes a decision. A row that merely increases
benchmark coverage is a liability.

The first manifest should therefore be small, typed, and source-auditable. The
correct first dataset is not "the most representative internet sample." It is
the smallest set of slices that can falsify the SE1 design.

## Teacher Roster Doctrine

Teacher choice must be scored under actual hardware and cost.

The first mistake would be to write down impressive teacher IDs and then run a
different system because the impressive teachers do not fit. The second mistake
would be to use a small runnable teacher and pretend it represents a strong
teacher family.

SE1 avoids both mistakes by treating every teacher as an admitted sensor:

```text
teacher_as_sensor_record:
  what_hidden_state_it_measures: required
  how_it_differs_from_existing_sensors: required
  what_student_gap_it_exposes: required
  where_the_lesson_can_land: required
  what_it_costs_to_measure: required
  what_signal_is_refused: required
  what_removal_test_it_must_survive: required
```

The newest model in a family is not automatically the right teacher. A cheaper
or older model may be better if it creates a cleaner control, lower signature
cost, or more interpretable response surface.

## Threshold Doctrine

Numeric gates are not truth. They are precommitments against self-deception.

The first thresholds should be modest and written before training because their
job is to stop three common failures:

- claiming a substrate win that is actually hidden compute;
- claiming Eklavya when the gain is one teacher or an average;
- claiming efficiency while hard cases or update cost quietly degrade.

Thresholds can be wrong. But they must be wrong in public, before the score is
known. Post-hoc thresholds are not gates.

## Patch Scout Doctrine

Patch size is the exchange rate between local byte burden and global reasoning
cost.

If only two fixed patch choices can be run, SE1 should first run the extremes
that expose the tradeoff: small patches for global sequence tax, large patches
for local decoder burden. The middle patch is the likely engineering compromise,
but it is a weaker first falsifier.

Dynamic patching comes later. If it enters too early, it can hide whether the
design actually understands the fixed-patch tradeoff.

## Implementation Boundary

The next engineering act is not "train Sutra." It is to create a pre-harness
readiness package that can be reviewed without running a model.

That package must answer:

```text
pre_harness_questions:
  - which exact slices enter the scout and why?
  - which exact teachers are runnable on available hardware?
  - which signals are explicitly refused?
  - which two or three substrate variants are compared?
  - what hidden compute is forbidden?
  - what would falsify the architecture before teacher packets matter?
  - what would falsify Eklavya after teacher packets are added?
  - what result would justify continuing rather than resetting?
```

Until those questions have written answers, implementation would recreate the
old failure mode: running harder before knowing what the run is supposed to
prove.
