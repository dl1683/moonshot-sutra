# Eklavya And Sutra Design Scratchpad

This file is the working scratchpad and falsifier contract. It is not a
benchmark harness and not a launch plan. It exists to keep design claims honest
while we search for the right Eklavya/Sutra system.

## Active Scope

In scope:

- Eklavya as multi-teacher invariant learning;
- Sutra as the world's most efficient student/system;
- teacher ports;
- student architecture;
- efficiency definitions;
- falsifiers for fake multi-teacher learning;
- ruthless trimming of irrelevant artifacts.

Out of scope:

- named process expansion;
- paper ceremony that does not improve teacher learning or student efficiency;
- another ordinary KD run without a structural hypothesis;
- benchmark chasing before architecture falsifiers are clear.

## Retired Material

The previous process-heavy packets are retired from active design.

Useful ideas kept:

- every component needs an outcome;
- every claim needs a falsifier;
- teacher disagreement must not be averaged blindly;
- efficiency must include lifecycle cost.

Everything else is deleted from the active design surface.

## Design Requirements

```text
sutra_eklavya_requirements_v0:
  eklavya:
    must_use_multiple_teacher_families: true
    must_preserve_teacher_identity: true
    must_measure_source_specific_contribution: true
    must_include_teacher_removal_test: true
    must_not_reduce_to_arithmetic_mean_kd: true
  sutra:
    must_be_capacity_aware: true
    must_support_adaptive_compute: true
    must_have_local_global_decomposition: true
    must_be_improvable_by_component: true
    must_measure_lifecycle_efficiency: true
  proof_standard:
    compare_against_ce_only_base: required
    compare_against_best_single_teacher: required
    compare_against_naive_multi_teacher_average: required
    compare_against_same_active_compute_baseline: required
```

## Candidate Architecture A: Token-Global / Byte-Local Sutra

```text
architecture_A:
  local_encoder:
    input: bytes
    role: boundary recovery, morphology, cross-tokenizer alignment
    candidate_mechanisms:
      - byte CNN
      - mLSTM local encoder
      - n-gram hash embeddings
      - learned patching
  global_reasoner:
    input: tokens_or_learned_patches
    role: semantic and reasoning computation
    candidate_mechanisms:
      - compact transformer
      - sparse SwiGLU
      - shared blocks with stronger local decoder
      - mixture-of-depths token routing
  local_decoder:
    input: global states plus local byte context
    role: surface realization and exact byte output
    candidate_mechanisms:
      - mLSTM decoder
      - small transformer decoder
      - byteified-teacher KD
  active_compute:
    mechanisms:
      - early exit
      - self-speculative verification
      - difficulty-conditioned depth
```

Why this is currently favored:

- byte-global processing likely paid too much sequence tax;
- byte-local keeps tokenizer freedom and exact output;
- token/patch-global gives more effective context per FLOP;
- clean local/global interfaces improve debugging;
- teacher signals can land at the correct level.

Primary falsifier:

```text
architecture_A_falsifier:
  if token_global_byte_local does not beat byte_global under matched parameter,
  token, and compute budgets on both loss and generation quality, then the split
  is not the first Sutra architecture.
```

## Candidate Architecture B: Compiled Eklavya Teacher System

```text
architecture_B:
  teacher_registry:
    - decoder_lm
    - encoder
    - verifier
    - symbolic_solver
    - trace_model
    - human_curriculum
  compiler_outputs:
    - top_k_distribution_targets
    - representation_anchors
    - boundary_maps
    - difficulty_maps
    - counterexample_clusters
    - proof_or_solution_traces
    - verifier_error_spans
  student_ingestion:
    - routed_loss_terms
    - curriculum_sampling
    - local_encoder_targets
    - global_reasoner_targets
    - decoder_targets
    - active_compute_targets
```

Why this matters:

- unlimited teachers cannot be used online during training or inference;
- compiled artifacts turn teacher cost into reusable training signal;
- source-specific packets make teacher contribution auditable;
- disagreement becomes curriculum, not entropy.

Primary falsifier:

```text
architecture_B_falsifier:
  if compiled teacher artifacts do not beat best-single-teacher and CE-only
  controls under matched student architecture and training budget, then the
  compiler is not doing Eklavya work.
```

## Candidate Architecture C: Conditional Capacity Student

```text
architecture_C:
  mechanisms:
    sparse_ffn:
      purpose: increase effective subnetwork diversity without increasing active cost
    early_exit:
      purpose: cheap handling of easy tokens/tasks
    self_speculative:
      purpose: draft with shallow computation, verify with deeper computation
    mixture_of_depths:
      purpose: spend layers on important tokens, skip easy tokens
    matryoshka_states:
      purpose: make representations usable at multiple cost/quality levels
```

Primary falsifier:

```text
architecture_C_falsifier:
  if conditional capacity does not reduce expected inference cost at equal
  quality, or if routing overhead erases savings, it is not Sutra efficiency.
```

## Teacher Port Contract

```text
teacher_port_contract_v0:
  teacher_id:
  teacher_family:
  teacher_strength:
  source_rights_or_access_cost:
  signal_type:
  student_landing_zone:
  invariant_hypothesis:
  known_corruption_modes:
  disagreement_value:
  compression_format:
  training_use:
  removal_test:
  expected_sutra_efficiency_gain:
```

No teacher should enter Eklavya without a port contract.

## Eklavya Falsifiers

Eklavya is fake if:

- the multi-teacher result is explained by one teacher;
- arithmetic averaging matches or beats the proposed method;
- the gain disappears after teacher removal;
- the gain exists only on the teacher's own distribution;
- source identity cannot be traced;
- teacher disagreement is treated as noise instead of curriculum;
- the student lacks architecture capacity to absorb the signal;
- the method improves BPB but damages generation or reasoning;
- teacher calls are required at inference for the claimed capability.

## Sutra Falsifiers

Sutra is not efficient if:

- it wins only by using more hidden compute;
- it requires expensive teacher calls at inference;
- it is hard to update surgically;
- active compute always uses maximum depth;
- memory or retrieval costs exceed saved model compute;
- byte-level purity destroys effective context;
- the student cannot explain which component owns which competence;
- the comparison ignores training, validation, or update cost.

## Minimal Design Experiments

These are design experiments, not runs authorized by this file.

### E0: Legacy Evidence Reconstruction

Question: what did the old Sutra-Dyad/KD work actually prove?

Expected output:

- list of valid mechanisms;
- list of failed assumptions;
- list of inconclusive claims;
- requirements for a cleaner next attempt.

### E1: Token-Global / Byte-Local Scout

Question: does moving global computation away from bytes improve effective
capacity at the same parameter budget?

Required controls:

- old byte-global student;
- token-global/byte-local student;
- same training budget;
- same data;
- generation checks, not just BPB.

### E2: Teacher Port Contribution

Question: does each teacher family add a unique invariant?

Required controls:

- no teacher;
- best single teacher;
- naive average;
- routed multi-teacher;
- each teacher removed one at a time.

### E3: Compiled Artifacts Versus Online KD

Question: can offline teacher artifacts beat online KD at lower cost?

Required controls:

- online teacher calls;
- cached logits only;
- compiled invariant packets;
- same student and training budget.

### E4: Active Compute

Question: can Sutra reduce expected inference cost without quality loss?

Required controls:

- fixed-depth baseline;
- early exit;
- self-speculative;
- mixture-of-depths or sparse routing.

### E5: Retained Gain

Question: after teachers are removed, what competence remains?

Required measures:

- capability before teacher exposure;
- capability during teacher-guided training;
- capability after teacher removal;
- retained gain by domain and teacher family;
- cost per retained gain.

## Current Best Hypothesis

```text
current_best_hypothesis:
  build_direction:
    token_global_byte_local_sutra
    + compiled_multi_teacher_invariant_packets
    + sparse_or_depth_adaptive_compute
    + retained_gain_tests
  avoid:
    ordinary multi_teacher_KD
    arithmetic_teacher_averaging
    byte_global_reasoning_as_default
    teacher_count_as_diversity
    meta_or_benchmark_work_that_does_not_change_architecture
```

## What Would Change My Mind

The design should change if evidence shows:

- byte-global is actually more efficient after learned patching;
- MoE at this size is stable and beats sparse FFN;
- representation alignment from encoder teachers gives large retained gains;
- online teacher interaction is cheaper than artifact compilation;
- a symbolic/verifier teacher provides a stronger invariant than LLM teachers;
- active compute hurts reasoning more than it saves cost.

## Next Thinking Targets

1. Define the exact teacher families for the first real Eklavya design.
2. Decide whether the global unit is token, learned patch, or variable patch.
3. Design a teacher-port schema that is small enough to use.
4. Specify the retained-gain metric.
5. Decide the minimum Sutra architecture that can absorb multiple invariants.
6. Define the first "David story" comparison honestly.

## SE1 Paper Architecture Spec

This is the first concrete design target. It is still a paper design, not code.

```text
SE1:
  name: Sutra-Eklavya SE1
  goal: smallest credible efficient student with multiple teacher landing zones

  student:
    input: raw bytes
    local_encoder:
      produces:
        - patch embeddings
        - byte residual flags
        - boundary entropy estimates
      must_preserve: exact byte recoverability
    global_reasoner:
      sequence_unit: learned patch or token-compatible patch
      backbone: compact attention model with optional selective SSM lane
      capacity: sparse FFN or conditional depth before MoE
      representation: scale-separated states if cheap enough
    local_decoder:
      consumes:
        - global states
        - local byte context
        - residual flags
      produces: exact bytes
    compute_controller:
      controls:
        - early exit
        - skipped depth
        - self-speculative verification
      trained_by:
        - entropy boundaries
        - verifier spans
        - teacher difficulty maps

  eklavya:
    teacher_axis: preserved
    packets:
      - behavior_signature_packet
      - geometry_anchor_packet
      - boundary_packet
      - error_localization_packet
      - curriculum_packet
    required_teacher_families:
      - decoder_lm
      - encoder
      - verifier_or_symbolic
      - byte_boundary_or_tokenization_source
      - curriculum_source
```

## Teacher Port Schemas For SE1

### Decoder Teacher Port

```text
decoder_teacher_port_SE1:
  teacher_family: decoder_lm
  source_object: candidate_string_logprob_surface
  probes:
    - paraphrase
    - distractor
    - counterfactual
    - compression
    - reasoning_step
  landing_zone: global_reasoner
  packet_type: behavior_signature_packet
  target_loss:
    - behavior_signature_KL
    - contrastive_invariance_loss
  corruption_modes:
    - teacher_style_imitation
    - refusal_or_censorship_transfer
    - shortcut_from_candidate_format
  removal_test:
    candidate_margin_gain_survives_without_teacher
```

### Encoder Teacher Port

```text
encoder_teacher_port_SE1:
  teacher_family: encoder
  source_object: pair_triplet_similarity_surface
  probes:
    - paraphrase_neighborhood
    - hard_negative_swap
    - retrieval_key_stability
    - domain_shift
  landing_zone: global_reasoner_or_memory_interface
  packet_type: geometry_anchor_packet
  target_loss:
    - representation_margin_loss
  use_only_if:
    student_alignment_low_or_task_specific_gap_present
  corruption_modes:
    - wasting_loss_on_already_aligned_geometry
    - importing embedding_model_bias
  removal_test:
    retrieval_or_similarity_gain_survives_without_encoder
```

### Verifier Or Symbolic Teacher Port

```text
verifier_teacher_port_SE1:
  teacher_family: verifier_or_symbolic
  source_object: constraint_result_and_error_span
  probes:
    - wrong_step
    - missing_constraint
    - contradiction
    - repair_needed
  landing_zone: verifier_head_and_compute_controller
  packet_type: error_localization_packet
  target_loss:
    - verifier_span_loss
    - compute_controller_loss
  corruption_modes:
    - learning verifier_format_not_constraint
    - overfitting to synthetic error style
  removal_test:
    error_localization_and_repair_choice_survive_without_verifier
```

### Byte Boundary Teacher Port

```text
byte_boundary_teacher_port_SE1:
  teacher_family: byte_boundary_or_tokenization_source
  source_object: entropy_boundary_or_residual_map
  probes:
    - Unicode_collision
    - morphology_boundary
    - rare_string
    - tokenization_shift
  landing_zone: local_byte_encoder_and_decoder
  packet_type: boundary_packet
  target_loss:
    - boundary_prediction_loss
    - residual_reconstruction_loss
  corruption_modes:
    - overcommitting_to_one_tokenizer
    - losing_byte_truth_channel
  removal_test:
    byte_exactness_and_boundary_robustness_survive_without_teacher
```

### Curriculum Teacher Port

```text
curriculum_teacher_port_SE1:
  teacher_family: curriculum_source
  source_object: high_density_examples_and_prerequisite_graph
  probes:
    - prerequisite_order
    - contrast_set
    - minimal_pair
    - delayed_recall
  landing_zone: data_scheduler
  packet_type: curriculum_packet
  target_loss:
    - sampling_weight
    - difficulty_progression
  corruption_modes:
    - synthetic_style_overfit
    - too_narrow_curriculum
  removal_test:
    downstream_generalization_survives_without_curriculum_teacher
```

## SE1 Training Contract

```text
SE1_training_contract:
  phase_0:
    name: base_student
    allowed_signals:
      - raw_text_CE
      - byte_reconstruction
      - patch_boundary_self_supervision
    forbidden_shortcut:
      - teacher_artifacts_before_student_interface_stabilizes

  phase_1:
    name: teacher_tomography
    output:
      - teacher_signature_tensor
      - invariant_packets
    requirement:
      - teacher_axis_preserved

  phase_2:
    name: invariant_landing
    requirement:
      - every packet names landing_zone
      - every loss maps to landing_zone
      - no early arithmetic averaging

  phase_3:
    name: active_compute
    requirement:
      - compare quality at matched expected active FLOPs
      - report easy/hard split

  phase_4:
    name: retained_gain
    requirement:
      - remove teacher calls
      - measure retained gain by teacher family
      - compare against CE-only, best-single-teacher, and naive average

  phase_5:
    name: surgical_update
    requirement:
      - add one teacher family or packet type
      - measure target improvement and collateral damage
```

## SE1 Success Criteria

SE1 is worth implementing only if the paper design can defend these criteria:
`artifacts/accomplishment/accomplishment_contract.json` materializes the current
not-yet-accomplished evidence contract for the Eklavya and Sutra claims.
`artifacts/accomplishment/accomplishment_progress_report.json` reports the
current missing evidence and blockers for that contract.
`artifacts/accomplishment/accomplishment_work_order.json` orders those missing
evidence outputs by run-admission step without admitting execution.

```text
SE1_success_criteria:
  substrate:
    byte_truth_preserved: required
    global_sequence_shorter_than_byte_sequence: required
    local_global_interfaces_debuggable: required

  eklavya:
    teacher_axis_preserved_until_loss_routing: required
    each_teacher_family_has_unique_landing_zone: required
    retained_gain_measured_after_teacher_removal: required

  sutra:
    conditional_compute_saves_expected_FLOPs: required
    quality_not_lost_on_hard_cases: required
    update_locality_measured: required

  comparison:
    beats_CE_only: required
    beats_best_single_teacher: required
    beats_naive_teacher_average: required
    beats_same_student_without_active_compute_on_efficiency: required
```

## Design Risks

```text
SE1_risks:
  too_many_losses:
    symptom: optimization instability
    mitigation: phase losses and ablate packet types

  teacher_axis_complexity:
    symptom: packet bookkeeping dominates learning
    mitigation: only five packet types in SE1

  patcher_instability:
    symptom: global reasoner sees moving unit distribution
    mitigation: stabilize patcher before teacher packet training

  active_compute_false_savings:
    symptom: easy cases cheap but hard cases degraded
    mitigation: report hard-case quality separately

  encoder_signal_redundancy:
    symptom: alignment loss adds cost without gain
    mitigation: use encoder packet only where alignment gap is measured

  hidden_teacher_dependence:
    symptom: gain vanishes when teacher artifacts are removed
    mitigation: retained-gain phase is mandatory
```

## SE1 Interface Tables

These tables are still design contracts, not implementation.

### Core Tensor Interfaces

```text
SE1_core_tensors:
  raw_bytes:
    shape: uint8[B, Tb]
    owner: input_channel

  patch_states:
    shape: float[B, P, D]
    owner: local_byte_encoder
    consumers:
      - global_reasoner
      - compute_controller

  patch_spans:
    shape: int[B, P, 2]
    owner: local_byte_encoder
    consumers:
      - local_byte_decoder
      - boundary_packet_loss
      - verifier_span_alignment

  residual_flags:
    shape: bool[B, Tb]
    owner: local_byte_encoder
    consumers:
      - local_byte_decoder
      - tokenization_limit_eval

  layer_states:
    shape: float[B, L, P, D]
    owner: global_reasoner
    consumers:
      - early_exit_heads
      - self_speculative_heads
      - representation_margin_loss

  teacher_signature_tensor:
    shape: float[B, G, M, C, Ksig]
    owner: eklavya_teacher_tomography
    consumers:
      - behavior_signature_loss
      - disagreement_curriculum_builder
      - compute_controller_targets

  exit_layer:
    shape: int[B, P]
    owner: compute_controller
    consumers:
      - expected_FLOP_accounting
      - hard_case_analysis

  byte_logits:
    shape: float[B, Tb, 256]
    owner: local_byte_decoder
    consumers:
      - CE_loss
      - exact_byte_eval
```

`Ksig` is a small signature vector, not a full vocabulary. It is defined by
`SE1_Ksig_signature_vector_v0` below and is forbidden from smuggling in full
vocabulary KL.

### Packet To Landing-Zone Matrix

```text
SE1_packet_landing_matrix:
  behavior_signature_packet:
    primary_zone: global_reasoner
    secondary_zone: compute_controller
    never_zone: local_byte_decoder_only

  geometry_anchor_packet:
    primary_zone: global_reasoner
    secondary_zone: memory_interface
    never_zone: byte_decoder_surface_style

  boundary_packet:
    primary_zone: local_byte_encoder
    secondary_zone: local_byte_decoder
    never_zone: global_reasoner_only

  error_localization_packet:
    primary_zone: verifier_head
    secondary_zone: compute_controller
    never_zone: answer_imitation_only

  curriculum_packet:
    primary_zone: data_scheduler
    secondary_zone: compute_controller
    never_zone: hidden_teacher_dependency
```

## Retained-Gain Metrics

```text
SE1_metric_definitions:
  owned_gain:
    formula: after_score - base_score
    interpretation: competence retained after teacher removal

  control_adjusted_gain:
    formula: after_score - matched_control_score
    interpretation: gain not explained by architecture, data, or compute alone

  retention_ratio:
    formula: owned_gain / max(during_score - base_score, epsilon)
    interpretation: fraction of teacher-exposed gain retained without teacher

  teacher_dependence_gap:
    formula: during_score - after_score
    interpretation: how much apparent competence required teacher availability

  collateral_damage:
    formula: max(0, old_task_score_before - old_task_score_after)
    interpretation: update locality failure

  lifecycle_efficiency:
    formula: owned_gain / total_lifecycle_cost
    interpretation: retained capability per total cost
```

Required slices:

```text
SE1_required_metric_slices:
  by_teacher_family: required
  by_packet_type: required
  by_landing_zone: required
  by_task_family: required
  by_compute_tier: required
  by_easy_hard_split: required
  by_update_round: required
```

## First Honest Comparison Suite

```text
SE1_comparisons:
  B0_CE_only_same_student:
    isolates: teacher_packet_value

  B1_best_single_teacher:
    isolates: multi_teacher_value

  B2_naive_teacher_average:
    isolates: teacher_axis_preservation

  B3_byte_global_same_budget:
    isolates: patch_global_substrate_value

  B4_SE1_without_active_compute:
    isolates: active_compute_efficiency

  B5_SE1_without_boundary_packets:
    isolates: byte_boundary_teacher_value

  B6_SE1_without_encoder_packets:
    isolates: geometry_anchor_value

  B7_SE1_without_verifier_packets:
    isolates: error_localization_value

  B8_SE1_without_curriculum_packets:
    isolates: curriculum_value
```

Comparison rule:

```text
SE1_comparison_rule:
  no_single_metric_win_is_sufficient: true
  must_report:
    - quality
    - expected_active_FLOPs
    - teacher_signature_cost
    - retained_gain
    - collateral_damage
    - hard_case_quality
    - byte_exactness
```

## SE1 Falsification Ladder

```text
SE1_falsification_ladder:
  substrate_fails:
    condition: patch-global does not beat byte-global under matched budget
    consequence: rethink student substrate before teacher system

  teacher_axis_fails:
    condition: naive average matches teacher-axis routing
    consequence: Eklavya compiler not justified

  packet_landing_fails:
    condition: packet losses improve training metric but not retained gain
    consequence: packet type is diagnostic only, not learning signal

  active_compute_fails:
    condition: expected FLOP savings cause hard-case regression
    consequence: compute controller is premature

  ownership_fails:
    condition: teacher_dependence_gap is large
    consequence: student did not own teacher invariant

  efficiency_fails:
    condition: lifecycle_efficiency loses to simpler baseline
    consequence: SE1 is not Sutra
```

## SE1 Next Paper Deliverable

The next design document layer should specify:

```text
SE1_next_deliverable:
  module_diagram: drafted
  tensor_interfaces: drafted
  packet_formats: drafted
  loss_schedule: drafted
  retained_gain_math: drafted
  comparison_suite: drafted
  first_scale_target: specified_for_paper_design
  first_teacher_ids: candidate_roster_specified_not_final
  first_task_slices: drafted
  cost_accounting_constants: drafted
  first_dataset_task_manifest: drafted
  teacher_selection_gates: drafted
  compute_planner_contract: drafted
  remaining_holes:
    - exact_teacher_choice_after_hardware_constraints
    - exact_public_dataset_ids_after_license_and_size_review
    - exact_loss_weights_and_stage_lengths_after_scout_runs
```

## SE1 First Scale, Teacher Roster, Task Slices, And Loss Schedule

The first contract target is a sub-500M student. A larger first student makes it
too easy to confuse scale with design. A much smaller student may only reproduce
the old capacity bottleneck.

```text
SE1_first_scale_target_v0:
  student_parameter_window:
    lower: 180M
    upper: 350M

  active_parameter_window:
    lower: 90M
    upper: 180M

  patch_ratio_target:
    first_target: 3x_to_6x_shorter_than_bytes

  global_depth:
    candidate: 12_to_16_layers

  hidden_width:
    candidate: 768_to_1024

  local_encoder_decoder_budget:
    target: at_most_15_percent_of_params

  teacher_signature_budget:
    first_pass: at_most_5_teacher_families

  no_teacher_at_inference:
    required: true
```

The candidate teacher IDs are not final. They are the first roster to design
against.

```text
SE1_candidate_teacher_roster_v0:
  decoder_primary:
    candidate_ids:
      - decoder-candidate-4B
      - anchor-decoder-1.7B
    selection_rule: choose_largest_affordable_for_offline_signature_generation

  decoder_efficiency_peer:
    candidate_ids:
      - HuggingFaceTB/SmolLM3-3B
      - HuggingFaceTB/SmolLM3-3B-Base
    selection_rule: include_if_it_adds_distinct_behavior_not_family_redundancy

  decoder_small_control:
    candidate_ids:
      - control-decoder-0.6B
    selection_rule: use_to_price_teacher_scale_effects

  encoder_primary:
    candidate_ids:
      - embedding-candidate-0.6B
      - BAAI/bge-m3
    selection_rule: compare_instruction_aware_embedding_against_dense_sparse_multivector_embedding

  encoder_small_control:
    candidate_ids:
      - BAAI/bge-small-en-v1.5
    selection_rule: use_if_primary_encoder_gain_may_be_scale_or_noise

  verifier_symbolic:
    candidate_ids:
      - task_local_exact_checker
      - unit_test_oracle
      - constraint_validator
      - symbolic_algebra_checker
    selection_rule: prefer_exact_non_neural_oracles_when_available

  boundary_teacher:
    candidate_ids:
      - BLT_patch_entropy_objective
      - MEGABYTE_local_global_split_objective
      - Charformer_subword_learning_objective
      - tokenization_limit_adversarial_suite
    selection_rule: teacher_is_objective_or_cases_not_necessarily_a_model

  curriculum_teacher:
    candidate_ids:
      - high_density_textbook_examples
      - minimal_world_examples
      - rationale_examples
      - contrast_set_examples
    selection_rule: use_only_if_curriculum_packets_improve_retained_gain
```

```text
SE1_task_slices_v0:
  byte_exactness_slice:
    packet_targets:
      - boundary_packet
      - residual_byte_decision
    required_metrics:
      - byte_exactness
      - patch_ratio
      - adversarial_tokenization_accuracy

  candidate_reasoning_probe_slice:
    packet_targets:
      - behavior_signature_packet
      - curriculum_packet
    required_metrics:
      - retained_accuracy_gain
      - margin_calibration
      - distractor_robustness

  retrieval_paraphrase_slice:
    packet_targets:
      - geometry_anchor_packet
    required_metrics:
      - paraphrase_retrieval_recall
      - representation_alignment
      - downstream_transfer_after_teacher_removal

  verifier_error_span_slice:
    packet_targets:
      - error_localization_packet
    required_metrics:
      - verifier_pass_rate
      - span_localization_F1
      - repair_success

  generation_coherence_slice:
    packet_targets:
      - behavior_signature_packet
      - curriculum_packet
    required_metrics:
      - judged_quality
      - repetition_rate
      - instruction_following

  active_compute_easy_hard_slice:
    packet_targets:
      - compute_controller
    required_metrics:
      - expected_active_FLOPs
      - p95_active_FLOPs
      - hard_case_regression
      - easy_case_savings
```

```text
SE1_loss_schedule_v0:
  stage_0_base_student:
    losses:
      - base_CE
      - byte_reconstruction
    exit_condition: byte_fidelity_and_language_loss_stable

  stage_1_boundary_stabilization:
    losses:
      - boundary_prediction_loss
      - residual_byte_loss
    exit_condition: patch_ratio_target_met_without_byte_exactness_regression

  stage_2_behavior_signature_training:
    losses:
      - behavior_signature_KL
      - teacher_margin_rank_loss
      - calibration_loss
    exit_condition: teacher_axis_predictive_and_not_collapsed

  stage_3_packet_landing_alternation:
    losses:
      - contrastive_invariance_loss
      - representation_margin_loss
      - verifier_span_loss
      - curriculum_order_loss
    exit_condition: each_packet_family_has_positive_control_adjusted_gain

  stage_4_gradient_conflict_control:
    candidate_methods:
      - fixed_weights
      - uncertainty_weighting
      - GradNorm
      - PCGrad
    selection_rule: choose_the_simplest_method_that_reduces_measured_packet_conflict

  stage_5_active_compute_heads:
    losses:
      - compute_penalty
      - early_exit_consistency
      - hard_case_recall_loss
    exit_condition: easy_case_savings_with_bounded_hard_case_damage

  stage_6_teacher_removal_eval:
    losses:
      - retained_gain_regularizer_if_needed
    exit_condition: owned_gain_positive_and_teacher_dependence_gap_bounded
```

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

Contract rule: a packet that improves a training proxy but loses lifecycle
efficiency or retained gain is not a Sutra component. It is a diagnostic.

## SE1 Module Graph Contract

```text
SE1_module_graph_v0:
  offline_teacher_side:
    nodes:
      - task_or_text_examples
      - probe_generator
      - teacher_family_ports
      - teacher_signature_tensor
      - disagreement_gap_analyzer
      - invariant_packet_compiler
      - packet_ledger
    outputs:
      - invariant_packets
      - teacher_cost_log
      - source_specific_signature_stats

  student_training_side:
    nodes:
      - raw_byte_stream
      - local_byte_profiler
      - patch_builder
      - patch_tape
      - global_reasoner
      - local_byte_decoder
      - compute_controller
    outputs:
      - byte_logits
      - candidate_logits
      - exit_layer
      - expected_active_FLOPs

  packet_landing_side:
    nodes:
      - packet_scheduler
      - landing_zone_router
      - packet_specific_losses
      - retained_gain_evaluator
    invariant:
      - every_packet_names_landing_zone_target_loss_corruption_risk_removal_test

  inference_side:
    allowed_inputs:
      - raw_bytes
      - learned_student_weights
      - student_internal_memory_if_trained
    forbidden_inputs:
      - teacher_calls
      - teacher_logits
      - teacher_hidden_states
      - packet_lookup_required_for_answer
    outputs:
      - exact_bytes_or_structured_answer
      - inference_cost_record
```

The inference side may emit telemetry for future training. It may not consume
teacher artifacts to answer the current query.

## SE1 Compute Planner Contract

```text
SE1_compute_planner_contract_v0:
  state_features:
    local_byte:
      - patch_entropy
      - residual_byte_probability
      - boundary_uncertainty
    global_reasoner:
      - early_exit_margin
      - layerwise_candidate_stability
      - representation_drift
    eklavya_profile:
      - historical_teacher_disagreement_score
      - packet_family_difficulty_score
      - verifier_risk_score
    cost:
      - current_active_FLOPs
      - expected_remaining_FLOPs
      - memory_read_cost_if_any

  actions:
    - exit_now
    - run_next_layer
    - run_sparse_ffn
    - run_self_speculative_head
    - run_verifier_head
    - run_local_byte_repair
    - read_student_memory

  training_targets:
    - teacher_disagreement_maps
    - verifier_error_spans
    - entropy_boundary_maps
    - early_exit_correctness
    - hard_case_labels

  decision_rule:
    maximize: expected_quality_delta_minus_lambda_cost
    constraints:
      - no_teacher_at_inference
      - p95_cost_reported
      - hard_case_regression_bounded
      - byte_exactness_regression_bounded
```

The planner must be evaluated against static-depth, early-exit-only, and
self-speculative-only baselines. Otherwise its benefit is not isolated.

## SE1 First Dataset And Task Manifest

This is a design manifest, not a download manifest. Exact public dataset IDs
should be chosen after license, size, and hardware review.

```text
SE1_first_dataset_task_manifest_v0:
  D0_base_text_slice:
    purpose: base language_and_byte_fidelity
    candidate_sources:
      - high_quality_web_or_educational_text
      - textbook_like_synthetic_text
      - tiny_world_curriculum_text
    size_posture:
      scout: 1B_to_5B_bytes_or_tokens_equivalent
      scale_if_validated: 20B_to_50B_bytes_or_tokens_equivalent
    acceptance_metric:
      - stable_base_CE
      - exact_byte_reconstruction
      - no_generation_collapse

  D1_byte_exactness_slice:
    purpose: residual_byte_and_boundary_validation
    examples:
      - unicode_confusables
      - mixed_language_strings
      - code_literals
      - filesystem_paths
      - rare_names
      - tokenizer_collision_cases
    acceptance_metric:
      - byte_exactness
      - patch_ratio
      - adversarial_tokenization_accuracy

  D2_candidate_reasoning_slice:
    purpose: teacher_tomography_validation
    examples:
      - ARC_style_MCQ
      - factual_counterfactual_pairs
      - distractor_robustness_pairs
      - paraphrase_invariance_pairs
    acceptance_metric:
      - retained_accuracy_gain
      - calibration_gain
      - probe_stability

  D3_retrieval_geometry_slice:
    purpose: encoder_packet_validation
    examples:
      - paraphrase_positive_pairs
      - hard_negative_pairs
      - delayed_recall_pairs
      - domain_shifted_neighbors
    acceptance_metric:
      - retrieval_recall
      - rank_margin_gain
      - teacher_free_transfer

  D4_verifier_slice:
    purpose: exact_oracle_and_error_localization_validation
    examples:
      - unit_tested_python_fragments
      - arithmetic_constraints
      - JSON_schema_constraints
      - contradiction_detection
    acceptance_metric:
      - verifier_pass_rate
      - span_localization_F1
      - repair_success

  D5_generation_slice:
    purpose: protect_language_quality_under_efficiency_pressure
    examples:
      - short_instruction_following
      - compact_explanations
      - style_preserving_rewrites
      - constrained_format_generation
    acceptance_metric:
      - judged_quality
      - repetition_rate
      - format_adherence

  D6_active_compute_slice:
    purpose: easy_hard_cost_policy_validation
    construction:
      - paired_easy_hard_versions_from_D1_to_D5
      - entropy_matched_controls
      - teacher_disagreement_matched_controls
    acceptance_metric:
      - easy_case_savings
      - hard_case_regression
      - p95_active_FLOPs
```

Admission rule: a dataset slice enters SE1 only if it maps to a named claim,
packet type, landing zone, and falsifier.

## SE1 Teacher Selection Gates

```text
SE1_teacher_selection_gates_v0:
  distinct_surface_gate:
    measurement:
      - probe_response_correlation_with_existing_teachers
      - disagreement_localization
      - entropy_profile_difference
    admit_if: teacher_adds_nonredundant_probe_surface

  student_gap_gate:
    measurement:
      - student_teacher_gap_on_target_slice
      - representation_alignment_gap
      - verifier_or_oracle_gap
    admit_if: student_lacks_the_invariant

  landing_zone_gate:
    measurement:
      - named_student_module
      - target_loss
      - packet_type
    admit_if: invariant_has_a_place_to_land

  cost_gate:
    measurement:
      - teacher_seconds
      - teacher_tokens_or_examples
      - packet_storage
      - expected_owned_gain
    admit_if: expected_lifecycle_efficiency_beats_extra_data_or_scale

  removal_gate:
    measurement:
      - owned_gain
      - teacher_dependence_gap
      - control_adjusted_gain
    admit_if: gain_survives_teacher_free_inference
```

If a teacher fails a gate, its signal may still be logged as evidence, but it
does not become part of the first SE1 training design.

## SE1 First Teacher Decision Rules

```text
SE1_first_teacher_decision_rules:
  decoder_primary:
    choose_decoder_candidate_4B_if:
      - offline_signature_generation_fits_available_hardware
      - probe_surface_differs_from_small_control
    else_choose:
      - anchor_decoder_1_7B

  decoder_efficiency_peer:
    include_SmolLM3_3B_if:
      - distinct_surface_gate_passes_against_primary_family
      - curriculum_or_think_no_think_behavior_adds_signal

  encoder_primary:
    choose_embedding_candidate_0_6B_if:
      - instruction_aware_geometry_is_needed
      - storage_and_latency_are_strong_constraints
    choose_BGE_M3_if:
      - dense_sparse_multivector_diversity_is_more_important

  verifier_teacher:
    prefer_exact_checker_if_available:
      - unit_tests
      - schema_validation
      - arithmetic_symbolic_check
    use_LLM_verifier_only_if:
      - no_exact_oracle_exists
      - LLM_verifier_span_quality_beats_noise_floor

  boundary_teacher:
    start_with:
      - entropy_boundary_objective
      - tokenizer_collision_cases
    add_learned_boundary_teacher_only_if:
      - static_entropy_patching_fails_substrate_falsifier
```

The first teacher roster remains candidate-level until these gates are scored.

## SE1 Utility, Teaching, And Promotion Contracts

The next contract layer turns design choices into economic choices.

```text
SE1_net_retained_gain_contract_v0:
  net_retained_gain:
    formula:
      expected_owned_gain
      - collateral_damage_penalty
      - teacher_dependence_penalty
      - lifecycle_cost_penalty
      - optimization_instability_penalty

  required_before_admission:
    - forecast_expected_owned_gain
    - forecast_teacher_or_packet_cost
    - forecast_corruption_risk
    - named_control

  required_after_training:
    - owned_gain
    - control_adjusted_gain
    - retention_ratio
    - teacher_dependence_gap
    - collateral_damage
    - measured_lifecycle_cost
```

### Packet Value Contract

```text
SE1_packet_value_contract_v0:
  prior_value:
    formula:
      P_student_gap_real
      * P_packet_lands
      * P_gain_survives_removal
      * estimated_capability_delta
      * complementarity_score
      - estimated_packet_cost
      - estimated_corruption_risk

  realized_value:
    formula:
      control_adjusted_gain
      * retention_ratio
      * hard_case_preservation
      * update_locality_score
      - measured_lifecycle_cost
      - measured_collateral_damage

  forecast_error:
    formula: prior_value - realized_value
    use: update_teacher_and_packet_selection_rules
```

Packets with positive training loss impact but negative realized value are
diagnostics, not training primitives.

### Probe Value Contract

```text
SE1_probe_value_contract_v0:
  probe_value:
    formula:
      expected_information_gain_about_student_gap
      + expected_information_gain_about_teacher_complementarity
      + expected_information_gain_about_landing_zone
      - probe_generation_cost
      - teacher_scoring_cost
      - ambiguity_penalty

  probe_phases:
    scout:
      purpose: find_disagreement_regions
    isolate:
      purpose: separate_knowledge_from_noise_style_or_artifact
    teach:
      purpose: create_minimal_packet_sequence

  reject_probe_if:
    - disagreement_cannot_be_interpreted
    - teacher_cost_exceeds_expected_information_gain
    - no_landing_zone_is_identified
```

### Machine Teaching Contract

```text
SE1_machine_teaching_contract_v0:
  lesson_object:
    - packet_sequence
    - curriculum_order
    - contrast_set
    - probe_family

  target:
    - student_owned_invariant

  optimization_question:
    minimize:
      lifecycle_teaching_cost
    subject_to:
      - owned_gain_positive
      - retention_ratio_floor
      - collateral_damage_ceiling
      - teacher_free_inference
```

The dataset is allowed to be deliberately non-iid. That is not a flaw. It is the
point of teaching, as long as generalization and retained-gain controls pass.

## SE1 Parameter Split And Patch Ladder

```text
SE1_parameter_split_prior_v0:
  total_student_params_nominal: 260M
  acceptable_range: 180M_to_350M

  local_byte_encoder:
    params: 20M_to_35M
    first_job: boundary_entropy_residual_byte_flags

  global_reasoner:
    params: 170M_to_220M
    first_job: semantic_reasoning_and_packet_absorption

  local_byte_decoder:
    params: 30M_to_50M
    first_job: exact_byte_realization_and_surface_repair

  compute_controller_and_heads:
    params: 5M_to_15M
    first_job: cheap_action_selection_and_early_exit

  optional_memory_or_adapter_space:
    params: 0M_to_20M
    admission_rule: only_after_gap_is_measured
```

```text
SE1_patch_ladder_v0:
  fixed_P4:
    role: high_resolution_control
    expected_failure: global_sequence_cost_too_high

  fixed_P6:
    role: middle_default
    expected_failure: neither_local_nor_global_optimal

  fixed_P8:
    role: strong_decoder_test
    expected_failure: local_decoder_burden_too_high

  entropy_dynamic:
    role: uncertainty_sensitive_candidate
    admission_rule: only_after_fixed_patch_tradeoff_is_measured
```

Patch size and local decoder capacity must be evaluated as one coupled variable.
Do not compare patch sizes while silently changing decoder burden.

## SE1 Loss Weight And Stage-Length Priors

```text
SE1_loss_stage_prior_v0:
  stage_0_base_student:
    scout_compute_share: 0.50_to_0.65
    packet_loss_weight: 0
    exit_condition:
      - base_CE_stable
      - generation_not_collapsed
      - byte_exactness_floor_met

  stage_1_boundary_stabilization:
    scout_compute_share: 0.10_to_0.15
    boundary_loss_gradient_budget: 0_to_0.15_of_base_gradient_norm
    exit_condition:
      - patch_ratio_floor_met
      - byte_exactness_not_degraded

  stage_2_behavior_signature:
    scout_compute_share: 0.10_to_0.15
    behavior_loss_gradient_budget: 0_to_0.20_of_base_gradient_norm
    exit_condition:
      - teacher_axis_not_collapsed
      - calibration_gain_positive

  stage_3_packet_landing:
    scout_compute_share: 0.10_to_0.20
    total_packet_gradient_budget: at_most_0.30_of_base_gradient_norm
    exit_condition:
      - at_least_one_packet_family_positive_control_adjusted_gain

  stage_4_active_compute:
    scout_compute_share: 0.05_to_0.10
    compute_penalty_rule: ramp_only_after_quality_floor_met
    exit_condition:
      - expected_active_FLOPs_reduced
      - hard_case_quality_floor_preserved

  stage_5_teacher_removal_eval:
    scout_compute_share: evaluation_only
    exit_condition:
      - owned_gain_positive_or_design_falsified
```

Gradient budget rule:

```text
packet_gradient_norm_sum <= 0.30 * base_gradient_norm
```

until a packet family proves positive retained value. This prevents elegant
teacher losses from overpowering the student substrate.

## SE1 Promotion Gates

```text
SE1_promotion_gates_v0:
  G0_substrate:
    required_evidence:
      - patch_global_beats_byte_global_matched_budget
      - byte_exactness_floor_met
      - generation_not_collapsed

  G1_teacher_profile:
    required_evidence:
      - three_teacher_families_with_distinct_probe_surfaces
      - one_non_decoder_family_passes_student_gap_gate
      - teacher_signature_cost_reported

  G2_packet_landing:
    required_evidence:
      - two_packet_families_positive_control_adjusted_gain
      - teacher_axis_preservation_beats_naive_average
      - collateral_damage_bounded

  G3_retained_gain:
    required_evidence:
      - owned_gain_positive_after_teacher_removal
      - best_single_teacher_control_beaten
      - teacher_dependence_gap_bounded

  G4_compute_efficiency:
    required_evidence:
      - expected_active_FLOPs_below_static_depth
      - p95_active_FLOPs_reported
      - hard_case_quality_floor_preserved

  G5_update_locality:
    required_evidence:
      - one_new_teacher_or_packet_improves_target_slice
      - collateral_damage_below_threshold
      - update_cost_below_full_retrain_baseline
```

World-class Sutra requires the whole ladder. A subsystem pass is useful, but it
must be named as a subsystem pass.

## SE1 Readiness Lock: Concrete Scout Defaults

This section closes the current paper-design holes enough to make the next
review concrete. It is still not permission to build the harness. It is the
minimum decision surface a harness would have to satisfy.

### First Dataset Admission Lock

Exact dataset IDs matter only when they map to an SE1 claim. A large famous
dataset is worse than a small clean slice if it does not isolate a packet,
landing zone, falsifier, and cost.

```text
SE1_dataset_admission_lock_v0:
  D0_base_text:
    preferred_sources:
      - common-pile/arxiv_abstracts
      - common-pile/arxiv_papers
      - common-pile/caselaw_access_project
      - HuggingFaceFW/fineweb-edu:sample-10BT
    first_choice_rule:
      - prefer_public_domain_or_openly_licensed_common_pile_sources
      - verify_each_common_pile_shard_license_from_underlying_source
      - use_fineweb_edu_only_as_quality_or_scale_ablation
    rejects_if:
      - license_or_terms_are_unclear_for_training_use
      - slice_cannot_be_made_small_enough_for_scout
      - text_quality_requires_more_filtering_than_training

  D1_byte_exactness:
    public_sources:
      - openai/openai_humaneval
      - codeparrot/apps
      - google-research-datasets/paws:pawsx
    local_generated_sources:
      - unicode_confusables
      - filesystem_paths
      - rare_names
      - mixed_script_strings
      - tokenizer_collision_cases
    first_choice_rule:
      - local_generated_cases_are_the_canonical_byte_truth_tests
      - public_sources_supply_realistic_code_and_multilingual_surface_strings
    rejects_if:
      - example_does_not_require_exact_byte_or_span_recovery

  D2_candidate_reasoning:
    public_sources:
      - allenai/ai2_arc
      - openai/gsm8k
      - allenai/openbookqa
    first_choice_rule:
      - admit_ai2_arc_and_gsm8k_first
      - keep_openbookqa_as_question_set_only_until_distribution_license_is_resolved
    packet_targets:
      - behavior_signature_packet
      - curriculum_packet
    rejects_if:
      - teacher_scores_cannot_be_expressed_as_candidate_margin_surfaces

  D3_retrieval_geometry:
    public_sources:
      - google-research-datasets/paws:pawsx
      - allenai/scifact
      - allenai/scitail
    first_choice_rule:
      - use_pawsx_for_adversarial_paraphrase_and_structure_preservation
      - use_scifact_as_evidence_or_eval_only_unless_noncommercial_constraint_is_acceptable
      - keep_scitail_as_question_set_only_until_license_is_resolved
      - require_license_review_before_any_training_use
    packet_targets:
      - geometry_anchor_packet
    rejects_if:
      - encoder_teacher_alignment_is_already_high_on_the_student

  D4_verifier_and_constraints:
    public_sources:
      - openai/openai_humaneval
      - codeparrot/apps
      - openai/gsm8k
      - epfl-dlab/JSONSchemaBench
    local_oracles:
      - python_unit_tests
      - arithmetic_answer_checkers
      - json_schema_validators
      - contradiction_templates
    first_choice_rule:
      - prefer_exact_oracles_over_llm_verifiers
    packet_targets:
      - error_localization_packet
      - compute_controller
    rejects_if:
      - pass_fail_signal_cannot_be_localized_to_span_or_constraint

  D5_generation_quality:
    public_sources:
      - roneneldan/TinyStories
      - databricks/databricks-dolly-15k
    local_sources:
      - compact_explanation_prompts
      - constrained_format_generation_prompts
      - style_preserving_rewrite_pairs
    first_choice_rule:
      - use_tinystories_as_small_model_language_diagnostic_not_world_claim
      - use_instruction_data_only_to_guard_against_generation_collapse
    rejects_if:
      - benchmark_style_overfits_the_student_to_a_narrow_chat_format

  D6_active_compute:
    public_sources:
      - derived_from_D1_to_D5
    construction:
      - paired_easy_hard_versions
      - entropy_matched_controls
      - teacher_disagreement_matched_controls
      - verifier_risk_matched_controls
    first_choice_rule:
      - no_standalone_active_compute_dataset_until_base_slices_exist
    packet_targets:
      - compute_controller
    rejects_if:
      - easy_hard_labels_are_confounded_with_dataset_source
```

The first manifest should be small enough to inspect manually. If a slice cannot
be sampled, licensed, cached, and audited by source, it does not enter the scout.

### Teacher ID And Hardware Lock

Teacher choice is a cost decision, not a prestige decision. The roster must be
pinned by model ID, revision, quantization if any, tokenizer, prompt template,
and actual cost per signature example.

```text
SE1_teacher_hardware_lock_v0:
  no_or_tiny_gpu:
    decoder_primary:
      - control-decoder-0.6B
    decoder_control:
      - same_model_different_prompt_or_temperature_control
    encoder_primary:
      - BAAI/bge-small-en-v1.5
    verifier_primary:
      - exact_local_oracles
    rule:
      - do_not_fake_large_teacher_diversity_with_unrunnable_models

  single_24gb_gpu:
    decoder_primary:
      - anchor-decoder-1.7B
    decoder_small_control:
      - control-decoder-0.6B
    decoder_efficiency_peer:
      - HuggingFaceTB/SmolLM3-3B-Base
    encoder_primary:
      - embedding-candidate-0.6B
      - BAAI/bge-m3
    verifier_primary:
      - exact_local_oracles
    rule:
      - include_smolLM3_only_if_distinct_surface_gate_passes

  forty_eight_gb_or_multi_gpu:
    decoder_primary_candidates:
      - decoder-candidate-4B
      - decoder-candidate-4B-v3.5
    decoder_controls:
      - anchor-decoder-1.7B
      - control-decoder-0.6B
    encoder_candidates:
      - embedding-candidate-0.6B
      - embedding-candidate-4B
      - BAAI/bge-m3
    rule:
      - use_newer_primary_family_only_if_license_base_or_posttrain_semantics_and_cost_are_better_for_signatures
      - do_not_treat_latest_as_automatically_better_teacher
```

Admission record required for every teacher:

```text
SE1_teacher_admission_record_v0:
  teacher_id: required
  revision_or_commit: required
  license: required
  local_runtime_mode: fp16 | bf16 | int8 | int4 | api | exact_oracle
  cost_per_1000_signature_examples: required
  probe_surface_correlation_with_existing_teachers: required
  student_gap_measure: required
  landing_zone: required
  accepted_for_packet_types: required
  rejected_signals: required
```

Current local hardware audit:

```text
SE1_local_hardware_snapshot_2026_06_20:
  evidence_artifact: artifacts/hardware/local_hardware_snapshot.json
  tool: tools/local_hardware_snapshot.py
  gpu: NVIDIA_GeForce_RTX_5090_Laptop_GPU
  vram: 24463_MiB
  driver: 595.79
  default_teacher_tier: single_24gb_gpu
  model_download_allowed: false
  training_allowed: false

  default_decoder_order:
    - anchor-decoder-1.7B
    - control-decoder-0.6B
    - HuggingFaceTB/SmolLM3-3B-Base_if_distinct_surface_gate_passes

  default_encoder_order:
    - embedding-candidate-0.6B
    - BAAI/bge-m3_if_runtime_and_storage_fit
    - BAAI/bge-small-en-v1.5_as_small_control

  not_default_for_first_local_scout:
    - decoder-candidate-4B
    - decoder-candidate-4B-v3.5
    - embedding-candidate-4B

  reason:
    - first_scout_needs_many_signature_examples_and_controls_not_one_impressive_teacher
    - 4B_class_teachers_can_be_reconsidered_after_cost_per_signature_example_is_measured
```

### Two-Choice Patch Scout Lock

If hardware allows only two fixed patch choices, run the extremes first:

```text
SE1_two_choice_patch_scout_v0:
  first_two_fixed_choices:
    - P4
    - P8
  reason:
    - P4_exposes_global_sequence_tax
    - P8_exposes_local_decoder_burden
    - the_pair_estimates_the_tradeoff_slope_better_than_P4_plus_P6_or_P6_plus_P8

  hold_constant:
    - total_params
    - local_decoder_param_budget
    - global_reasoner_param_budget
    - train_tokens
    - data_order
    - byte_exactness_eval
    - generation_eval

  follow_up:
    choose_P6_if:
      - P4_is_too_expensive_and_P8_damages_byte_exactness
      - both_extremes_fail_for_opposite_reasons
    choose_entropy_dynamic_if:
      - fixed_patch_tradeoff_is_visible
      - boundary_uncertainty_predicts_error_or_cost
      - dynamic_patching_does_not_hide_decoder_burden
```

P6 is the likely engineering default, not the best first falsifier. The first
scout should maximize information about the design equation.

### Loss Conflict And Weight Lock

The first loss schedule is fixed and conservative. Adaptive multitask machinery
is admitted only after measurable conflict.

```text
SE1_loss_conflict_lock_v0:
  default:
    - fixed_stage_weights
    - packet_gradient_norm_sum_lte_0_30_base_gradient_norm
    - no_PCGrad_GradNorm_or_uncertainty_weighting_by_default

  measure_every:
    - 500_to_1000_optimizer_steps

  conflict_metrics:
    - pairwise_gradient_cosine_between_loss_families
    - per_loss_gradient_norm_ratio_to_base_CE
    - retained_gain_delta_by_packet_family
    - collateral_damage_delta

  trigger_loss_downweight_if:
    - packet_loss_gradient_norm_gt_0_30_base_gradient_norm
    - packet_family_has_no_positive_control_adjusted_gain
    - collateral_damage_exceeds_gate_floor

  trigger_PCGrad_candidate_if:
    - pairwise_gradient_cosine_lt_minus_0_20_for_3_windows
    - both_conflicting_losses_have_positive_prior_value
    - fixed_downweighting_loses_more_retained_gain_than_it_saves

  trigger_GradNorm_candidate_if:
    - useful_packet_loss_gradient_norm_is_gt_2x_or_lt_0_5x_target_budget_for_3_windows
    - imbalance_changes_retained_gain_not_only_training_loss

  reject_adaptive_balancing_if:
    - it_improves_proxy_loss_but_reduces_teacher_free_retained_gain
```

### Numeric Promotion Thresholds

These are scout thresholds, not publication thresholds. They are deliberately
modest because the first job is to falsify bad designs cheaply.

```text
SE1_numeric_promotion_thresholds_v0:
  G0_substrate:
    pass_if:
      - global_sequence_length_reduction_gte_25_percent_vs_byte_global
      - validation_loss_no_worse_than_byte_global_by_more_than_1_percent_relative
      - byte_exactness_floor_gte_99_9_percent_on_D1
      - generation_quality_no_detectable_collapse_on_D5
    fail_if:
      - patch_global_win_is_only_hidden_in_extra_decoder_compute

  G1_teacher_profile:
    pass_if:
      - at_least_3_teacher_families_profiled
      - at_least_1_non_decoder_family_has_student_gap_gte_5_percent_relative_or_2_absolute_points
      - at_least_2_teacher_pairs_have_probe_surface_correlation_lte_0_90
      - all_teacher_signature_costs_reported
    fail_if:
      - disagreement_is_unlocalized_or_explained_by_tokenization_noise

  G2_packet_landing:
    pass_if:
      - at_least_2_packet_families_have_control_adjusted_gain_gt_0
      - one_packet_family_improves_primary_slice_by_gte_2_absolute_points_or_10_percent_error_reduction
      - teacher_axis_preservation_beats_naive_average_by_gte_10_percent_of_available_gain
      - collateral_damage_lte_1_percent_relative_on_unrelated_slices
    fail_if:
      - packet_gain_disappears_when_source_identity_is_removed

  G3_retained_gain:
    pass_if:
      - owned_gain_gt_0_after_teacher_removal
      - retention_ratio_gte_0_50
      - teacher_dependence_gap_lte_0_25_of_teacher_exposed_gain
      - best_single_teacher_control_beaten_on_at_least_one_primary_slice
    fail_if:
      - apparent_gain_requires_teacher_artifacts_at_inference

  G4_compute_efficiency:
    pass_if:
      - mean_active_FLOPs_lte_0_80_static_depth_baseline
      - p95_active_FLOPs_lte_1_10_static_depth_baseline
      - hard_case_regression_lte_2_absolute_points
      - byte_exactness_regression_lte_0_1_absolute_points
    fail_if:
      - easy_case_savings_are_bought_by_hard_case_damage

  G5_update_locality:
    pass_if:
      - one_new_teacher_or_packet_improves_target_slice_by_gte_2_absolute_points_or_10_percent_error_reduction
      - collateral_damage_lte_1_percent_relative
      - update_cost_lte_0_30_full_retrain_baseline
      - rollback_or_demote_path_is_clear
    fail_if:
      - update_requires_global_retraining_to_avoid_damage
```

Any threshold can be revised after a scout only if the revision is written before
looking at the final score. Otherwise the gate has become post-hoc decoration.

## SE1 Pre-Harness Readiness Package

This is the current review package that must pass before any implementation
harness is built. Its purpose is to stop SE1 from quietly turning into another
run-first experiment.

Current verdict:

```text
SE1_pre_harness_readiness_verdict_2026_06_20:
  status: not_ready_for_harness
  reason:
    - real_common_pile_license_histogram_and_sample_manifest_missing
    - local_prepared_rows_and_real_D1_D2_D5_byte_profiles_missing
    - teacher_runtime_modes_and_signature_throughput_not_measured
    - accepted_teacher_bindings_and_real_signature_tensors_missing
    - baseline_gap_retained_gain_and_gate_metric_reports_not_run
  allowed_next_work:
    - complete_this_readiness_package
    - review_and_tighten_canonical_docs
    - measure_model_and_dataset_metadata_without_training
    - maintain_checked_readiness_claim_and_admission_artifacts
  forbidden_next_work:
    - launch_student_training
    - build_full_harness
    - download_large_base_text_corpora
    - claim_SE1_evidence
```

### Readiness Checklist

```text
SE1_pre_harness_checklist_v0:
  repo_surface:
    status: pass
    evidence:
      - active_repo_has_README_plus_canonical_research_docs
      - old_code_results_experiments_removed_from_active_surface

  local_hardware:
    status: pass
    evidence:
      - local_hardware_snapshot_artifact_created
      - single_24gb_gpu_tier_selected
      - RTX_5090_Laptop_GPU_24463_MiB_recorded
      - no_model_download_or_training_performed

  dataset_revision_pins:
    status: partial_license_review_written
    evidence:
      - first_candidate_dataset_revisions_recorded_below
      - first_candidate_dataset_size_and_split_metadata_recorded_below
      - first_planned_slice_policy_recorded_below
      - dataset_license_review_recorded_in_research/SE1_DATASET_LICENSE_REVIEW.md
      - D0_common_pile_row_filter_draft_created
      - d0_input_map_contract_created
      - D0_common_pile_filter_tool_fixture_passed
      - d0_source_admission_packet_created
      - sharealike_review_artifact_created_for_held_sources
      - scout_slice_row_preparation_manifest_created
    missing:
      - D0_common_pile_license_histogram
      - D0_common_pile_sample_manifest
      - local_prepared_rows_jsonl_for_admitted_slices
      - D5_distribution_policy_for_held_sources

  teacher_revision_pins:
    status: partial
    evidence:
      - first_candidate_model_revisions_recorded_below
      - safetensors_weight_footprints_recorded_below
      - Ksig_fixture_materialization_tool_and_reports_created
      - teacher_runtime_report_tool_fixture_passed
      - prompt_template_manifest_and_fixture_render_created
      - tokenizer_behavior_record_schema_fixture_passed
      - candidate_teacher_binding_manifest_created
      - teacher_port_contracts_created
      - teacher_runtime_scope_bound_to_ports_and_bindings
      - tokenizer_scope_bound_to_ports_and_bindings
      - ksig_signature_scope_bound_to_candidate_bindings
      - teacher_binding_admission_step_guard_created
      - source_ecology_record_created
      - teacher_selection_gate_record_created
    missing:
      - quantization_choice
      - accepted_teacher_bindings_with_real_records
      - real_tokenizer_behavior_records
      - examples_per_second
      - cost_per_1000_signature_examples
      - real_teacher_signature_tensor
      - real_surface_correlation_report

  patch_scout:
    status: pass_for_design_smoke_accounting_and_fixture_extraction_written
    evidence:
      - P4_and_P8_selected_as_first_two_fixed_patch_falsifiers
      - byte_global_P4_P8_shape_records_defined_below
      - B3_P4_P8_config_files_created_under_artifacts/shape_manifest/configs
      - synthetic_shape_smoke_reports_created_under_artifacts/shape_manifest
      - byte_length_profile_extractor_fixture_passed
    missing:
      - real_D1_D2_D5_byte_length_profiles
      - real_G0_shape_accounting_report

  loss_schedule:
    status: pass_for_design
    evidence:
      - fixed_stage_weights_required_before_adaptive_multitask_methods
      - gradient_conflict_triggers_written
      - metric_logger_spec_created_under_artifacts/metrics
      - gate_metric_binding_record_created_under_artifacts/gates
    missing:
      - real_metric_logs
      - real_metric_summary_report

  promotion_gates:
    status: pass_for_design
    evidence:
      - numeric_thresholds_written_for_G0_to_G5
      - fixture_preflight_audit_passed_with_harness_blocked
      - gate_report_template_file_created
      - gate_metric_binding_record_created
      - run_admission_manifest_created
      - external_condition_registry_created
      - teacher_admission_ordering_split_and_guarded
      - run_admission_graph_guard_created
      - remaining_blocker_gate_id_guard_created
      - run_admission_output_blocker_guard_created
      - run_admission_required_dependency_guard_created
      - artifact_index_real_evidence_boundary_guard_created
      - artifact_index_operator_input_boundary_guard_created
      - artifact_index_directory_status_guard_created
      - artifact_index_blocked_flag_guard_created
      - d0_dataset_license_boundary_guard_created
      - tokenizer_behavior_readme_boundary_guard_created
      - teacher_runtime_readme_boundary_guard_created
      - shape_manifest_readme_boundary_guard_created
      - tooling_docs_check_created
      - preflight_readme_guard_created
      - claim_boundary_created
      - public_claim_posture_created
      - accomplishment_contract_created
      - accomplishment_progress_report_created
      - accomplishment_work_order_created
    missing:
      - real_gate_report_rows_with_evidence_artifacts

  packet_ledger:
    status: template_file_created_rows_unfilled
    evidence:
      - packet_ledger_template_file_created
      - required_packet_rows_listed_before_harness
      - initial_packet_ledger_rows_created_as_templates
    missing:
      - packet_rows_filled_from_real_sources
      - cost_fields_from_real_runtime
      - retained_gain_fields_after_training

  baseline_gap:
    status: template_file_created_no_results
    evidence:
      - baseline_gap_template_file_created
      - baseline_gap_template_check_passed_no_results
    missing:
      - student_baseline_run
      - teacher_exposed_gap_measurements
      - teacher_free_retained_gain_measurements
      - best_single_teacher_and_no_teacher_controls

  exact_oracles:
    status: fixture_oracle_checks_created_no_real_eval
    evidence:
      - exact_oracle_manifest_created
      - gsm8k_numeric_answer_fixture_checked
      - json_schema_subset_fixture_checked
    missing:
      - real_D2_D4_prepared_rows
      - real_oracle_eval_report
      - student_baseline_before_gap_probe
```

### Candidate Dataset Revision Pins

These pins are metadata anchors, not admissions. A dataset enters the scout only
after the missing size, row count, split, and license checks are completed.

```text
SE1_candidate_dataset_revision_pins_2026_06_20:
  D0_candidate_after_row_filter_manifest:
    common-pile/arxiv_abstracts:
      revision: 828e35d1000f94579da8850f5f640c138279bdb5
      license_metadata: unspecified_on_card_row_license_CC0_public_domain_in_viewer
      first_config: default
      use:
        - D0_base_text_public_domain_scout
      hold:
        - complete_license_histogram_and_sample_manifest_before_download

    common-pile/caselaw_access_project:
      revision: 3c2cb5080b3a16a04d8d8d07b28eaec7c1ba7a90
      license_metadata: unspecified_on_card_row_license_public_domain_in_viewer
      first_config: default
      use:
        - D0_base_text_public_domain_scout
      hold:
        - complete_license_histogram_and_sample_manifest_before_download
        - sample_cap_before_large_download

    common-pile/biodiversity_heritage_library:
      revision: ad1bbb00d5df579c0ef0a9dfe46e50e2bb1715b8
      license_metadata: unspecified_on_card_row_license_public_domain_in_viewer
      first_config: default
      use:
        - D0_base_text_public_domain_scout
      hold:
        - complete_license_histogram_and_sample_manifest_before_download
        - sample_cap_before_large_download

    common-pile/arxiv_papers:
      revision: 963fe980c55b353980653f1a27c1dc0c8a2d7058
      license_metadata: unspecified_on_card_mixed_row_licenses_in_viewer
      first_config: default
      use:
        - D0_base_text_only_after_row_license_histogram
      hold:
        - held_until_explicit_row_license_filter_and_exclusion_report_exists

  default_admit_after_size_review:
    openai/gsm8k:
      revision: 740312add88f781978c0658806c59bc2815b9866
      license_metadata: mit
      first_config: main
      use:
        - D2_candidate_reasoning
        - D4_arithmetic_exact_oracle

    openai/openai_humaneval:
      revision: 7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544
      license_metadata: mit
      first_config: openai_humaneval
      use:
        - D1_code_byte_exactness
        - D4_unit_test_verifier

    codeparrot/apps:
      revision: 21e74ddf8de1a21436da12e3e653065c5213e9d1
      license_metadata: mit
      first_config: unresolved_by_parquet_api
      use:
        - D1_code_surface_strings
        - D4_unit_test_verifier
      hold:
        - resolve_config_and_download_size_before_use

    epfl-dlab/JSONSchemaBench:
      revision: 5bd0f4640badc6f3f02df796421d21cb0ca0b141
      license_metadata: mit
      first_config: default_or_Github_easy
      use:
        - D4_json_schema_verifier

  admit_with_attribution_or_sharealike_review:
    allenai/ai2_arc:
      revision: 210d026faf9955653af8916fad021475a3f00453
      license_metadata: cc-by-sa-4.0
      first_configs:
        - ARC-Challenge
        - ARC-Easy
      use:
        - D2_candidate_reasoning
      hold:
        - sharealike_implications_before_training_use
        - sharealike_review_keeps_training_use_blocked

    HuggingFaceFW/fineweb-edu:
      revision: 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9
      license_metadata: odc-by
      first_config: sample-10BT_only_if_base_text_ablation_is_needed
      use:
        - D0_base_text_quality_scale_ablation
      hold:
        - do_not_make_default_until_common_pile_shards_are_checked

    roneneldan/TinyStories:
      revision: f54c09fd23315a6f9c86f9dc80f725de7d8f9c64
      license_metadata: cdla-sharing-1.0
      first_config: default
      use:
        - D5_small_model_generation_guardrail
      hold:
        - not_a_world_capability_claim
        - sharing_terms_review_keeps_training_use_blocked_until_result_policy

    databricks/databricks-dolly-15k:
      revision: bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a
      license_metadata: cc-by-sa-3.0
      first_config: default
      use:
        - D5_instruction_generation_guardrail
      hold:
        - sharealike_implications_before_training_use
        - sharealike_review_keeps_training_use_blocked

  evaluation_or_question_only_until_license_review:
    google-research-datasets/paws:
      revision: 161ece9501cf0a11f3e48bd356eaa82de46d6a09
      license_metadata: other
      first_configs:
        - labeled_final
        - labeled_swap
      use:
        - D3_paraphrase_structure_eval
      hold:
        - inspect_repository_license_terms_before_training_use

    allenai/openbookqa:
      hold:
        - unknown_HF_license_metadata

    allenai/scitail:
      hold:
        - unspecified_HF_license_metadata

    allenai/scifact:
      hold:
        - noncommercial_metadata_blocks_default_training_claim
```

### Candidate Dataset Slice Metadata

This table records current API metadata for the first small sources. It is not
yet the final scout manifest because `planned_rows` and admission decisions must
still be written before download or training.

```text
SE1_candidate_dataset_slice_metadata_2026_06_20:
  D0_common_pile_arxiv_abstracts:
    dataset: common-pile/arxiv_abstracts
    config: default
    revision: 828e35d1000f94579da8850f5f640c138279bdb5
    license_metadata: unspecified_on_card_row_license_CC0_public_domain_in_viewer
    compressed_data_files: 4
    compressed_bytes_from_hf_tree_api: 1127871878
    rows_total: about_2540000
    splits:
      train: about_2540000
    planned_rows: held_until_license_histogram_and_sample_manifest
    admission_status: candidate_after_row_license_filter_manifest

  D0_common_pile_caselaw_access_project:
    dataset: common-pile/caselaw_access_project
    config: default
    revision: 3c2cb5080b3a16a04d8d8d07b28eaec7c1ba7a90
    license_metadata: unspecified_on_card_row_license_public_domain_in_viewer
    compressed_data_files: 48
    compressed_bytes_from_hf_tree_api: 6602477877
    rows_total: about_5520000
    splits:
      train: about_5520000
    planned_rows: held_until_license_histogram_sample_manifest_and_sample_cap
    admission_status: candidate_after_row_license_filter_manifest

  D0_common_pile_biodiversity_heritage_library:
    dataset: common-pile/biodiversity_heritage_library
    config: default
    revision: ad1bbb00d5df579c0ef0a9dfe46e50e2bb1715b8
    license_metadata: unspecified_on_card_row_license_public_domain_in_viewer
    compressed_data_files: 47
    compressed_bytes_from_hf_tree_api: 15830744336
    rows_total: about_45600000
    splits:
      train: about_45600000
    planned_rows: held_until_license_histogram_sample_manifest_and_sample_cap
    admission_status: candidate_after_row_license_filter_manifest

  D0_common_pile_arxiv_papers:
    dataset: common-pile/arxiv_papers
    config: default
    revision: 963fe980c55b353980653f1a27c1dc0c8a2d7058
    license_metadata: unspecified_on_card_mixed_row_licenses_in_viewer
    compressed_data_files: 22
    compressed_bytes_from_hf_tree_api: 6262555629
    rows_total: about_317000
    splits:
      train: about_317000
    planned_rows: 0
    admission_status: held_until_row_license_histogram_and_filter_report

  D2_gsm8k_main:
    dataset: openai/gsm8k
    config: main
    revision: 740312add88f781978c0658806c59bc2815b9866
    license_metadata: mit
    parquet_bytes: 2725633
    memory_bytes: 4709831
    rows_total: 8792
    splits:
      train: 7473
      test: 1319
    planned_rows: unset
    admission_status: candidate_admit_after_planned_subset_written

  D4_humaneval_verifier:
    dataset: openai/openai_humaneval
    config: openai_humaneval
    revision: 7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544
    license_metadata: mit
    parquet_bytes: 83920
    memory_bytes: 194394
    rows_total: 164
    splits:
      test: 164
    planned_rows: eval_only_all_164
    admission_status: candidate_admit_as_eval_or_oracle_source

  D4_jsonschema_github_easy:
    dataset: epfl-dlab/JSONSchemaBench
    config: Github_easy
    revision: 5bd0f4640badc6f3f02df796421d21cb0ca0b141
    license_metadata: mit
    parquet_bytes: 540610
    memory_bytes: 1933231
    rows_total: 1938
    splits:
      train: 1170
      val: 191
      test: 577
    planned_rows: unset
    admission_status: candidate_admit_after_planned_subset_written

  D2_ai2_arc_challenge:
    dataset: allenai/ai2_arc
    config: ARC-Challenge
    revision: 210d026faf9955653af8916fad021475a3f00453
    license_metadata: cc-by-sa-4.0
    parquet_bytes: 449460
    memory_bytes: 823135
    rows_total: 2590
    splits:
      train: 1119
      validation: 299
      test: 1172
    planned_rows: held_until_sharealike_review
    admission_status: held

  D2_ai2_arc_easy:
    dataset: allenai/ai2_arc
    config: ARC-Easy
    revision: 210d026faf9955653af8916fad021475a3f00453
    license_metadata: cc-by-sa-4.0
    parquet_bytes: 762935
    memory_bytes: 1444264
    rows_total: 5197
    splits:
      train: 2251
      validation: 570
      test: 2376
    planned_rows: held_until_sharealike_review
    admission_status: held

  D3_paws_labeled_final:
    dataset: google-research-datasets/paws
    config: labeled_final
    revision: 161ece9501cf0a11f3e48bd356eaa82de46d6a09
    license_metadata: other
    parquet_bytes: 10899391
    memory_bytes: 16125189
    rows_total: 65401
    splits:
      train: 49401
      validation: 8000
      test: 8000
    planned_rows: held_until_license_terms_review
    admission_status: question_or_eval_only

  D3_paws_labeled_swap:
    dataset: google-research-datasets/paws
    config: labeled_swap
    revision: 161ece9501cf0a11f3e48bd356eaa82de46d6a09
    license_metadata: other
    parquet_bytes: 5741756
    memory_bytes: 7911123
    rows_total: 30397
    splits:
      train: 30397
    planned_rows: held_until_license_terms_review
    admission_status: question_or_eval_only

  D5_tinystories_guardrail:
    dataset: roneneldan/TinyStories
    config: default
    revision: f54c09fd23315a6f9c86f9dc80f725de7d8f9c64
    license_metadata: cdla-sharing-1.0
    parquet_bytes: 1000775442
    memory_bytes: 2025789179
    rows_total: 2141709
    splits:
      train: 2119719
      validation: 21990
    planned_rows: unset_small_guardrail_sample_only
    admission_status: candidate_after_license_and_sample_cap_review

  D5_dolly_guardrail:
    dataset: databricks/databricks-dolly-15k
    config: default
    revision: bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a
    license_metadata: cc-by-sa-3.0
    parquet_bytes: 7747823
    memory_bytes: 12195589
    rows_total: 15011
    splits:
      train: 15011
    planned_rows: held_until_sharealike_review
    admission_status: held
```

### Planned Slice Policy

The first scout should fit in memory, be easy to inspect, and falsify SE1
mechanisms. It should not consume every available row just because the dataset
is small.

```text
SE1_planned_slice_policy_v0:
  D0_base_text:
    status: candidate_limited_common_pile_public_domain_scout
    reason:
      - dataset_license_review_written
      - common_pile_public_domain_shards_identified_for_row_filter_manifest
      - fineweb_edu_is_scale_ablation_not_default
    candidate_sources:
      - common-pile/arxiv_abstracts
      - common-pile/caselaw_access_project
      - common-pile/biodiversity_heritage_library
    held_sources:
      - common-pile/arxiv_papers_until_row_license_histogram_exists
      - HuggingFaceFW/fineweb-edu_as_default_base_text
    planned_rows: 0_until_license_histogram_and_sample_manifest_written

  D1_local_byte_exactness:
    source: local_generated
    planned_rows:
      unicode_confusables: 500
      filesystem_paths: 500
      rare_names: 500
      mixed_script_strings: 500
      tokenizer_collision_cases: 500
    split_policy:
      train: 60_percent
      validation: 20_percent
      test: 20_percent
    packet_targets:
      - boundary_packet
      - residual_byte_decision
    falsifier:
      - byte_exactness_or_boundary_robustness_does_not_improve

  D1_code_byte_exactness:
    source: openai/openai_humaneval
    planned_rows:
      eval_only: 164
    split_policy:
      - no_training_on_canonical_humaneval_tests
    packet_targets:
      - boundary_packet
      - error_localization_packet
    falsifier:
      - code_surface_byte_exactness_regresses

  D2_arithmetic_candidate_reasoning:
    source: openai/gsm8k
    config: main
    planned_rows:
      train: 7473
      test: 1319
    split_policy:
      - use_existing_train_test_split
      - no_socratic_config_until_main_surface_is_understood
    packet_targets:
      - behavior_signature_packet
      - error_localization_packet
      - curriculum_packet
    falsifier:
      - teacher_margin_surfaces_fail_to_predict_retained_accuracy_or_calibration_gain

  D2_science_candidate_reasoning:
    source: allenai/ai2_arc
    status: held
    planned_rows: 0
    hold_until:
      - sharealike_implications_reviewed
      - exact_multiple_choice_prompt_format_written

  D3_paraphrase_geometry:
    source: google-research-datasets/paws
    status: question_or_eval_only
    planned_rows: 0
    hold_until:
      - license_terms_reviewed
      - geometry_packet_value_is_needed_after_encoder_gap_check

  D4_json_schema_verifier:
    source: epfl-dlab/JSONSchemaBench
    config: Github_easy
    planned_rows:
      train: 1170
      validation: 191
      test: 577
    split_policy:
      - use_existing_train_val_test_split
      - keep_harder_configs_out_until_easy_schema_packet_has_value
    packet_targets:
      - error_localization_packet
      - compute_controller
    falsifier:
      - schema_pass_rate_or_error_localization_does_not_improve_after_teacher_removal

  D5_tinystories_generation_guardrail:
    source: roneneldan/TinyStories
    status: sample_cap_only_after_license_review
    planned_rows:
      train_cap: 50000
      validation_cap: 5000
    split_policy:
      - deterministic_seeded_sample_from_existing_splits
      - use_only_as_generation_collapse_guardrail
    packet_targets:
      - curriculum_packet
    falsifier:
      - generation_quality_regresses_under_efficiency_pressure

  D5_instruction_guardrail:
    source: databricks/databricks-dolly-15k
    status: held
    planned_rows: 0
    hold_until:
      - sharealike_implications_reviewed
      - instruction_guardrail_need_is_reconfirmed
```

Admission rule:

```text
SE1_slice_admission_rule_v0:
  admit_if:
    - license_status_allows_planned_use
    - planned_rows_are_written
    - split_policy_is_written
    - packet_target_is_named
    - falsifier_is_named
  reject_or_hold_if:
    - source_is_only_available_because_it_is_convenient
    - planned_slice_does_not_change_G0_or_G1_decision
    - source_cost_or_license_status_is_ambiguous
```

### Candidate Teacher Revision Pins

These are the default local candidates for the recorded single-24GB machine.
They still require local runtime checks before admission.

```text
SE1_candidate_teacher_revision_pins_2026_06_20:
  decoder_primary:
    id: anchor-decoder-1.7B
    revision: 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
    license_metadata: apache-2.0
    runtime_status: not_measured
    admission_goal:
      - behavior_signature_packet
      - candidate_margin_surfaces

  decoder_small_control:
    id: control-decoder-0.6B
    revision: c1899de289a04d12100db370d81485cdf75e47ca
    license_metadata: apache-2.0
    runtime_status: not_measured
    admission_goal:
      - scale_control
      - cheap_signature_baseline

  decoder_efficiency_peer:
    id: HuggingFaceTB/SmolLM3-3B-Base
    revision: d78a42f79198603e614095753484a04c10c2b940
    license_metadata: apache-2.0
    runtime_status: not_measured
    admit_only_if:
      - distinct_surface_gate_passes_against_primary_family

  encoder_primary:
    id: embedding-candidate-0.6B
    revision: 97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3
    license_metadata: apache-2.0
    runtime_status: not_measured
    admission_goal:
      - geometry_anchor_packet
      - retrieval_or_paraphrase_margin_surface

  encoder_retrieval_candidate:
    id: BAAI/bge-m3
    revision: 5617a9f61b028005a4858fdac845db406aefb181
    license_metadata: mit
    runtime_status: not_measured
    admit_only_if:
      - runtime_and_storage_fit_local_scout

  encoder_small_control:
    id: BAAI/bge-small-en-v1.5
    revision: 5c38ec7c405ec4b44b94cc5a9bb96e735b38267a
    license_metadata: mit
    runtime_status: not_measured
    admission_goal:
      - encoder_scale_control
```

### Candidate Teacher Runtime Metadata

These values are repository metadata, not local runtime measurements. They are
enough to choose what to test first, but not enough to admit a teacher.

```text
SE1_candidate_teacher_runtime_metadata_2026_06_20:
  anchor-decoder-1.7B:
    revision: 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
    safetensors_weight_bytes: 4063515592
    repo_used_storage_bytes: 4074938246
    first_runtime_prior: bf16_or_fp16
    quantization: unset
    examples_per_second: unmeasured
    peak_vram: unmeasured
    cost_per_1000_signature_examples: unmeasured

  control-decoder-0.6B:
    revision: c1899de289a04d12100db370d81485cdf75e47ca
    safetensors_weight_bytes: 1503300328
    repo_used_storage_bytes: 4522815806
    first_runtime_prior: bf16_or_fp16
    quantization: unset
    examples_per_second: unmeasured
    peak_vram: unmeasured
    cost_per_1000_signature_examples: unmeasured

  HuggingFaceTB/SmolLM3-3B-Base:
    revision: d78a42f79198603e614095753484a04c10c2b940
    safetensors_weight_bytes: 6150235008
    repo_used_storage_bytes: 38523035903
    first_runtime_prior: bf16_or_fp16_if_weights_only_download_is_supported
    quantization: unset
    examples_per_second: unmeasured
    peak_vram: unmeasured
    cost_per_1000_signature_examples: unmeasured
    note:
      - repo_storage_is_large_because_it_contains_ONNX_variants
      - first_probe_should_fetch_only_safetensors_tokenizer_and_config

  embedding-candidate-0.6B:
    revision: 97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3
    safetensors_weight_bytes: 1191586416
    repo_used_storage_bytes: 10170768124
    first_runtime_prior: fp16_or_bf16
    quantization: unset
    examples_per_second: unmeasured
    peak_vram: unmeasured
    cost_per_1000_signature_examples: unmeasured

  BAAI/bge-m3:
    revision: 5617a9f61b028005a4858fdac845db406aefb181
    safetensors_weight_bytes: 2271145830
    repo_used_storage_bytes: 13704659878
    first_runtime_prior: fp16_or_bf16
    quantization: unset
    examples_per_second: unmeasured
    peak_vram: unmeasured
    cost_per_1000_signature_examples: unmeasured

  BAAI/bge-small-en-v1.5:
    revision: 5c38ec7c405ec4b44b94cc5a9bb96e735b38267a
    safetensors_weight_bytes: 266974701
    repo_used_storage_bytes: 834515761
    first_runtime_prior: fp16_or_cpu_control_if_needed
    quantization: unset
    examples_per_second: unmeasured
    peak_vram: unmeasured
    cost_per_1000_signature_examples: unmeasured
```

First runtime measurement rule:

```text
SE1_teacher_runtime_measurement_rule_v0:
  measure_order:
    - BAAI/bge-small-en-v1.5
    - control-decoder-0.6B
    - embedding-candidate-0.6B
    - anchor-decoder-1.7B
    - BAAI/bge-m3
    - HuggingFaceTB/SmolLM3-3B-Base_if_distinct_surface_still_needed

  record_for_each:
    - exact_downloaded_files
    - local_disk_bytes
    - runtime_mode
    - quantization
    - max_batch_size
    - examples_per_second_on_signature_prompts
    - peak_vram
    - cost_per_1000_signature_examples

  reject_if:
    - required_files_exceed_local_disk_or_cache_budget
    - peak_vram_blocks_student_or_other_teacher_controls
    - examples_per_second_makes_signature_generation_more_expensive_than_extra_student_training
```

### Required Tables Before Harness

The readiness package is not complete until these tables exist in this contract
or in another explicitly canonical file.

```text
SE1_required_pre_harness_tables_v0:
  dataset_slice_table:
    required_columns:
      - slice_id
      - dataset_id_or_local_generator
      - revision
      - config
      - split
      - planned_rows
      - estimated_download_size
      - license_status
      - packet_target
      - falsifier
      - admitted_or_held

  teacher_runtime_table:
    required_columns:
      - teacher_id
      - revision
      - runtime_mode
      - quantization
      - max_context
      - examples_per_second
      - memory_peak
      - cost_per_1000_signature_examples
      - packet_targets
      - accepted_or_rejected

  packet_ledger_table:
    required_columns:
      - packet_id
      - source_teacher_or_oracle
      - source_revision
      - dataset_slice
      - landing_zone
      - target_loss
      - corruption_risk
      - refused_signal
      - removal_test
      - expected_value_prior
      - realized_value_after_training
      - demotion_rule

  gate_report_table:
    required_columns:
      - gate_id
      - threshold
      - evidence_artifact
      - pass_fail
      - failure_interpretation
      - next_action
```

The first harness can be built only after every row needed for G0 and G1 is
filled. G2 through G5 may start as templates, but their thresholds must remain
precommitted before any score is seen.

### Packet Ledger Template

The packet ledger is the anti-mush table. It prevents teacher signal, data
curation, and verifier feedback from collapsing into an untraceable training
mixture.

```text
SE1_packet_ledger_template_v0:
  packet_id:
    rule: stable_unique_id
    example: pkt_behavior_gsm8k_anchor17_margin_v0

  packet_family:
    allowed:
      - behavior_signature_packet
      - geometry_anchor_packet
      - boundary_packet
      - error_localization_packet
      - curriculum_packet

  source_kind:
    allowed:
      - decoder_teacher
      - encoder_teacher
      - exact_oracle
      - local_generator
      - curriculum_source
      - dataset_slice

  source_id:
    required: true
    examples:
      - anchor-decoder-1.7B
      - openai/gsm8k
      - local_unicode_confusable_generator

  source_revision:
    required_for_external_sources: true

  dataset_slice_id:
    required: true

  probe_family:
    required_for_teacher_packets: true
    allowed:
      - paraphrase
      - distractor
      - counterfactual
      - compression
      - reasoning_step
      - boundary_stress
      - verifier_error
      - curriculum_order

  landing_zone:
    allowed:
      - local_byte_encoder
      - global_reasoner
      - local_byte_decoder
      - verifier_head
      - compute_controller
      - data_scheduler

  target_loss:
    required: true

  refused_signal:
    required: true
    examples:
      - full_vocab_KL
      - teacher_style
      - teacher_hidden_state_without_gap
      - answer_only_label
      - inference_time_teacher_lookup

  corruption_risk:
    required: true

  removal_test:
    required: true

  expected_value_prior:
    required_before_training: true

  realized_value:
    required_after_training: true

  demotion_rule:
    required: true
```

First ledger rows that must be filled before G1/G2 work:

```text
SE1_required_packet_rows_before_harness_v0:
  - pkt_boundary_local_generated_byte_exactness_v0
  - pkt_behavior_gsm8k_anchor17_margin_v0
  - pkt_behavior_gsm8k_control06_scale_control_v0
  - pkt_error_gsm8k_exact_oracle_v0
  - pkt_error_jsonschema_github_easy_oracle_v0
  - pkt_geometry_encoder_gap_probe_v0
```

No packet may be consumed by training unless its ledger row has `source_id`,
`source_revision`, `landing_zone`, `target_loss`, `refused_signal`,
`corruption_risk`, `removal_test`, and `demotion_rule`.

Initial packet ledger rows:

```text
SE1_initial_packet_ledger_rows_v0:
  pkt_boundary_local_generated_byte_exactness_v0:
    packet_family: boundary_packet
    source_kind: local_generator
    source_id: local_byte_exactness_generators_v0
    source_revision: repo_commit_required_before_harness
    dataset_slice_id: D1_local_byte_exactness
    probe_family: boundary_stress
    landing_zone:
      - local_byte_encoder
      - local_byte_decoder
    target_loss:
      - boundary_prediction_loss
      - residual_byte_loss
    refused_signal:
      - tokenizer_specific_boundary_as_ground_truth
      - surface_style
    corruption_risk:
      - synthetic_cases_are_too_clean
      - boundary_model_overfits_to_generator_templates
    removal_test:
      - byte_exactness_and_boundary_robustness_hold_without_generator_at_inference
    expected_value_prior:
      status: plausible_high_for_G0
      reason: directly_tests_byte_truth_and_patch_boundary_failure
    realized_value: unset_until_training
    demotion_rule:
      - demote_if_byte_exactness_gain_lte_0_or_generation_regresses

  pkt_behavior_gsm8k_anchor17_margin_v0:
    packet_family: behavior_signature_packet
    source_kind: decoder_teacher
    source_id: t0_anchor_decoder
    source_revision: (see private teacher config)
    dataset_slice_id: D2_arithmetic_candidate_reasoning
    probe_family:
      - reasoning_step
      - distractor
      - counterfactual
    landing_zone:
      - global_reasoner
      - compute_controller
    target_loss:
      - behavior_signature_KL
      - teacher_margin_rank_loss
      - calibration_loss
    refused_signal:
      - full_vocab_KL
      - teacher_chain_of_thought_style
      - inference_time_teacher_lookup
    corruption_risk:
      - teacher_math_shortcuts
      - prompt_format_overfit
      - primary_family_bias
    removal_test:
      - retained_accuracy_and_calibration_gain_on_gsm8k_test_without_teacher_calls
    expected_value_prior:
      status: plausible_medium
      reason: candidate_margin_surfaces_may_teach_uncertainty_better_than_answer_labels
    realized_value: unset_until_training
    demotion_rule:
      - demote_if_best_single_teacher_or_answer_only_control_matches_retained_gain
      - demote_if_teacher_dependence_gap_exceeds_G3_limit

  pkt_behavior_gsm8k_control06_scale_control_v0:
    packet_family: behavior_signature_packet
    source_kind: decoder_teacher
    source_id: t2_control_decoder
    source_revision: (see private teacher config)
    dataset_slice_id: D2_arithmetic_candidate_reasoning
    probe_family:
      - reasoning_step
      - distractor
      - counterfactual
    landing_zone:
      - global_reasoner
      - compute_controller
    target_loss:
      - behavior_signature_KL
      - teacher_margin_rank_loss
      - calibration_loss
    refused_signal:
      - full_vocab_KL
      - same_family_agreement_as_independent_evidence
      - inference_time_teacher_lookup
    corruption_risk:
      - weak_teacher_noise
      - same_family_correlation_with_anchor17
    removal_test:
      - scale_control_distinguishes_large_teacher_value_from_family_echo
    expected_value_prior:
      status: control_not_primary_gain_source
      reason: needed_to_price_teacher_scale_and_family_redundancy
    realized_value: unset_until_training
    demotion_rule:
      - demote_as_training_signal_if_surface_correlation_with_anchor17_gt_0_90_and_no_unique_student_gap

  pkt_error_gsm8k_exact_oracle_v0:
    packet_family: error_localization_packet
    source_kind: exact_oracle
    source_id: gsm8k_answer_parser_and_arithmetic_checker_v0
    source_revision: repo_commit_required_before_harness
    dataset_slice_id: D2_arithmetic_candidate_reasoning
    probe_family: verifier_error
    landing_zone:
      - verifier_head
      - compute_controller
    target_loss:
      - verifier_span_loss_if_span_available
      - answer_correctness_loss
      - hard_case_recall_loss
    refused_signal:
      - LLM_judge_as_oracle_when_exact_answer_check_suffices
      - answer_only_training_without_error_class
    corruption_risk:
      - parser_accepts_spurious_format
      - arithmetic_checker_misses_reasoning_error_type
    removal_test:
      - verifier_pass_rate_or_error_classification_gain_survives_without_oracle_at_inference
    expected_value_prior:
      status: plausible_high_for_D4
      reason: exact_oracle_is_cheaper_and_cleaner_than_LLM_verifier
    realized_value: unset_until_training
    demotion_rule:
      - demote_if_oracle_signal_only_improves_format_adherence_not_correctness_or_repair

  pkt_error_jsonschema_github_easy_oracle_v0:
    packet_family: error_localization_packet
    source_kind: exact_oracle
    source_id: json_schema_validator_v0
    source_revision: repo_commit_required_before_harness
    dataset_slice_id: D4_json_schema_verifier
    probe_family: verifier_error
    landing_zone:
      - verifier_head
      - compute_controller
      - local_byte_decoder
    target_loss:
      - verifier_span_loss_if_available
      - constraint_violation_class_loss
      - compute_controller_loss
    refused_signal:
      - LLM_judge
      - schema_text_memorization
      - inference_time_external_validator_dependency_for_claimed_student_answer
    corruption_risk:
      - model_learns_schema_surface_without_constraint_generalization
      - easy_schema_distribution_is_too_narrow
    removal_test:
      - schema_validity_and_repair_choice_survive_without_external_validator_at_inference
    expected_value_prior:
      status: plausible_high_for_verifier_packet
      reason: exact_constraint_signal_maps_cleanly_to_verifier_head_and_compute_controller
    realized_value: unset_until_training
    demotion_rule:
      - demote_if_github_easy_gain_does_not_transfer_to_held_schema_splits

  pkt_geometry_encoder_gap_probe_v0:
    packet_family: geometry_anchor_packet
    source_kind: encoder_teacher
    source_id:
      - embedding-teacher-0.6B
      - embedding-teacher-small
    source_revision:
      (see private teacher config)
    dataset_slice_id: D3_paraphrase_geometry_or_local_probe
    probe_family:
      - paraphrase
      - distractor
      - hard_negative
    landing_zone:
      - global_reasoner
      - memory_interface_if_admitted_later
    target_loss:
      - representation_margin_loss
    refused_signal:
      - hidden_state_alignment_without_student_gap
      - geometry_transfer_as_capability_proof
      - noncommercial_or_license_ambiguous_training_rows
    corruption_risk:
      - student_already_has_high_alignment
      - embedding_teacher_bias
      - paraphrase_dataset_license_ambiguity
    removal_test:
      - retrieval_or_paraphrase_margin_gain_survives_without_encoder_at_inference
    expected_value_prior:
      status: unknown_until_encoder_gap_check
      reason: legacy_evidence_warns_alignment_may_already_be_high
    realized_value: unset_until_training
    demotion_rule:
      - skip_training_packet_if_student_teacher_alignment_gap_is_small
      - demote_if_geometry_gain_does_not_change_downstream_retained_skill
```

### Gate Report Template

The gate report is the anti-retcon table. It makes each threshold visible before
scores are known and attaches every pass/fail claim to a concrete artifact.

```text
SE1_gate_report_template_v0:
  gate_id:
    allowed:
      - G0_substrate
      - G1_teacher_profile
      - G2_packet_landing
      - G3_retained_gain
      - G4_compute_efficiency
      - G5_update_locality

  claim:
    required: true

  threshold:
    required: true

  evidence_artifact:
    required: true
    examples:
      - metadata_table
      - eval_json
      - cost_log
      - retained_gain_report
      - generation_review

  coverage:
    required: true
    allowed:
      - full_gate
      - one_slice
      - one_teacher_family
      - smoke_only
      - missing

  result:
    allowed:
      - not_run
      - pass
      - fail
      - inconclusive

  failure_interpretation:
    required_if_result_fail_or_inconclusive: true

  next_action:
    required: true
```

Initial pre-harness gate rows:

```text
SE1_initial_gate_rows_v0:
  G0_substrate:
    current_result: not_run
    pre_harness_requirement:
      - byte_global_baseline_shape_written
      - P4_and_P8_patch_global_shapes_written
      - byte_exactness_eval_defined
      - generation_guardrail_defined

  G1_teacher_profile:
    current_result: not_run
    pre_harness_requirement:
      - teacher_runtime_table_filled
      - at_least_three_teacher_families_have_probe_plan
      - distinct_surface_metric_defined
      - teacher_signature_cost_log_defined

  G2_packet_landing:
    current_result: template_only
    pre_harness_requirement:
      - packet_ledger_rows_exist
      - naive_average_control_defined
      - source_identity_ablation_defined

  G3_retained_gain:
    current_result: template_only
    pre_harness_requirement:
      - teacher_removal_eval_path_defined
      - best_single_teacher_control_defined

  G4_compute_efficiency:
    current_result: template_only
    pre_harness_requirement:
      - static_depth_baseline_defined
      - mean_and_p95_active_FLOPs_logger_defined

  G5_update_locality:
    current_result: template_only
    pre_harness_requirement:
      - no_first_harness_dependency
      - define_after_G2_or_G3_has_real_packet_signal
```

The first harness may cover only G0 and G1. If it does, the report must say so
plainly and must not imply that SE1, Eklavya, or Sutra has passed.

Concrete pre-harness gate rows:

```text
SE1_concrete_pre_harness_gate_rows_2026_06_20:
  G0_substrate_global_sequence_reduction:
    gate_id: G0_substrate
    claim: patch_global_student_shortens_global_sequence_without_hiding_compute
    threshold: global_sequence_length_reduction_gte_25_percent_vs_byte_global
    evidence_artifact: not_created
    coverage: missing
    result: not_run
    failure_interpretation: unset
    next_action:
      - instantiate_B3_P4_P8_config_files_from_shape_manifest
      - run_shape_accounting_on_D1_D2_and_D5_scout_rows
      - report_hidden_decoder_compute_accounting

  G0_substrate_loss_floor:
    gate_id: G0_substrate
    claim: patch_global_student_does_not_lose_language_modeling_quality_under_matched_budget
    threshold: validation_loss_no_worse_than_byte_global_by_more_than_1_percent_relative
    evidence_artifact: not_created
    coverage: missing
    result: not_run
    failure_interpretation: unset
    next_action:
      - define_validation_loss_slice
      - define_matched_budget_training_steps

  G0_substrate_byte_exactness:
    gate_id: G0_substrate
    claim: byte_truth_channel_survives_patch_global_compression
    threshold: byte_exactness_floor_gte_99_9_percent_on_D1
    evidence_artifact: not_created
    coverage: missing
    result: not_run
    failure_interpretation: unset
    next_action:
      - materialize_D1_local_byte_exactness_generator_spec
      - define_exact_byte_eval

  G0_substrate_generation_guardrail:
    gate_id: G0_substrate
    claim: patch_global_student_does_not_collapse_generation_quality
    threshold: generation_quality_no_detectable_collapse_on_D5
    evidence_artifact: not_created
    coverage: missing
    result: not_run
    failure_interpretation: unset
    next_action:
      - define_small_generation_guardrail_prompts
      - decide_whether_TinyStories_license_review_is_needed_for_guardrail

  G1_teacher_profile_family_count:
    gate_id: G1_teacher_profile
    claim: at_least_three_teacher_families_are_profiled_with_source_identity
    threshold: at_least_3_teacher_families_profiled
    evidence_artifact: not_created
    coverage: missing
    result: not_run
    failure_interpretation: unset
    next_action:
      - measure_decoder_teacher_runtime
      - measure_encoder_teacher_runtime
      - define_exact_oracle_runtime_record

  G1_teacher_profile_non_decoder_gap:
    gate_id: G1_teacher_profile
    claim: at_least_one_non_decoder_family_exposes_student_gap
    threshold: non_decoder_family_student_gap_gte_5_percent_relative_or_2_absolute_points
    evidence_artifact: not_created
    coverage: missing
    result: not_run
    failure_interpretation: unset
    next_action:
      - run_encoder_gap_probe_after_student_baseline_exists
      - run_exact_oracle_gap_probe_after_student_baseline_exists

  G1_teacher_profile_distinct_surface:
    gate_id: G1_teacher_profile
    claim: teacher_surfaces_are_not_redundant_echoes
    threshold: at_least_2_teacher_pairs_have_probe_surface_correlation_lte_0_90
    evidence_artifact: not_created
    coverage: missing
    result: not_run
    failure_interpretation: unset
    next_action:
      - materialize_signature_tensor_on_first_probe_rows
      - report_pairwise_surface_correlation_with_bootstrap_CI
      - compare_anchor17_control06_and_exact_oracle_surfaces_first

  G1_teacher_profile_cost_report:
    gate_id: G1_teacher_profile
    claim: teacher_signature_cost_is_visible_before_teacher_packets_train_student
    threshold: all_teacher_signature_costs_reported
    evidence_artifact: not_created
    coverage: missing
    result: not_run
    failure_interpretation: unset
    next_action:
      - fill_teacher_runtime_table
      - record_examples_per_second_peak_vram_and_cost_per_1000_signature_examples
```

Current gate conclusion:

```text
SE1_gate_status_after_pre_harness_rows:
  G0_substrate: not_run_shape_smoke_written_real_G0_artifacts_missing
  G1_teacher_profile: not_run_Ksig_fixture_written_blocked_on_runtime_real_signatures_and_baseline_gap
  G2_to_G5: template_only_not_first_harness_claims
```

## D1 Local Byte Exactness Specification

D1 is the first byte-truth slice. It exists to falsify the idea that SE1 can
compress bytes into patches while still preserving exact surface identity.

It is deliberately local-generated before external datasets enter, because the
first byte exactness proof should not depend on dataset license ambiguity.

```text
SE1_D1_local_byte_exactness_spec_v0:
  total_rows: 2500
  generator_seed: fixed_before_generation
  split_policy:
    train:
      rows: 1500
      percent: 60
    validation:
      rows: 500
      percent: 20
    test:
      rows: 500
      percent: 20

  row_schema:
    id: stable_string
    family:
      allowed:
        - unicode_confusable
        - filesystem_path
        - rare_name
        - mixed_script_string
        - tokenizer_collision_case
    raw_text: utf8_string
    raw_bytes_hex: lowercase_hex
    byte_length: integer
    expected_patch_stress:
      allowed:
        - boundary_ambiguity
        - residual_byte_required
        - exact_copy_required
        - script_switch
        - rare_surface_form
    forbidden_normalization:
      - unicode_normalization
      - case_folding
      - slash_or_separator_rewrite
      - whitespace_collapse
      - quote_normalization
```

Generator families:

```text
SE1_D1_generator_families_v0:
  unicode_confusable:
    rows: 500
    construction:
      - visually_similar_latin_cyrillic_greek_pairs
      - combining_marks
      - zero_width_joiner_or_non_joiner_cases
      - fullwidth_ascii_variants
    target_failure:
      - model_collapses_visually_similar_but_byte_distinct_strings

  filesystem_path:
    rows: 500
    construction:
      - windows_paths_with_spaces
      - posix_paths
      - escaped_backslashes
      - relative_paths
      - extension_and_case_variants
    target_failure:
      - model_rewrites_separators_or_collapses_escape_sequences

  rare_name:
    rows: 500
    construction:
      - uncommon_person_names
      - hyphenated_names
      - apostrophe_names
      - transliterated_names
      - accent_mark_variants
    target_failure:
      - model_regularizes_rare_surface_forms_to_common_names

  mixed_script_string:
    rows: 500
    construction:
      - latin_plus_devanagari
      - latin_plus_arabic
      - latin_plus_cjk
      - code_identifier_with_non_ascii_suffix
      - natural_language_plus_symbolic_token
    target_failure:
      - patcher_places_boundaries_that_destroy_script_switch_information

  tokenizer_collision_case:
    rows: 500
    construction:
      - same_prefix_different_suffix
      - whitespace_sensitive_pairs
      - punctuation_sensitive_pairs
      - code_literal_pairs
      - number_format_pairs
    target_failure:
      - global_patch_representation_merges_distinct_byte_surfaces
```

Exact byte eval:

```text
SE1_D1_exact_byte_eval_v0:
  primary_metric:
    byte_exact_match:
      definition: generated_or_reconstructed_bytes_equal_raw_bytes
      gate_floor: 0.999

  secondary_metrics:
    byte_edit_distance:
      definition: levenshtein_distance_over_bytes
      report:
        - mean
        - p95
        - max

    family_exact_match:
      definition: byte_exact_match_grouped_by_family
      report:
        - unicode_confusable
        - filesystem_path
        - rare_name
        - mixed_script_string
        - tokenizer_collision_case

    forbidden_normalization_rate:
      definition: fraction_of_failures_explained_by_forbidden_normalization
      target: 0

    patch_boundary_error_rate:
      definition: fraction_of_required_boundary_or_residual_cases_without_expected_flag
      report_by_family: true

  hard_failures:
    - any_family_exact_match_below_0_995
    - forbidden_normalization_rate_gt_0
    - byte_exact_match_below_0_999_on_test
```

D1 evidence artifact contract:

```text
SE1_D1_evidence_artifacts_v0:
  generator_manifest:
    path: artifacts/d1_byte_exactness/generator_manifest.json
    status: created_manifest_only_rows_not_generated
    required_fields:
      - generator_seed
      - row_count_by_family
      - split_counts
      - schema_version
      - repo_commit
    check_report: artifacts/d1_byte_exactness/generator_manifest_check.json

  generated_rows:
    path: artifacts/d1_byte_exactness/rows.jsonl
    status: not_created
    rule:
      - may_be_generated_only_after_harness_phase_is_approved

  eval_report:
    path: artifacts/d1_byte_exactness/eval_report.json
    status: not_created
    required_fields:
      - byte_exact_match
      - byte_edit_distance_mean
      - byte_edit_distance_p95
      - byte_edit_distance_max
      - family_exact_match
      - forbidden_normalization_rate
      - patch_boundary_error_rate
```

G0 mapping:

```text
SE1_D1_to_G0_mapping_v0:
  G0_substrate_byte_exactness:
    uses:
      - SE1_D1_exact_byte_eval_v0.byte_exact_match
      - SE1_D1_exact_byte_eval_v0.family_exact_match
      - SE1_D1_exact_byte_eval_v0.forbidden_normalization_rate
    pass_if:
      - byte_exact_match_gte_0_999_on_test
      - every_family_exact_match_gte_0_995
      - forbidden_normalization_rate_eq_0
    fail_means:
      - patch_global_substrate_does_not_preserve_byte_truth_yet
      - do_not_proceed_to_teacher_packets
```

## SE1 Baseline Shape Records

These are shape records, not benchmark evidence. They exist so B3, P4, and P8
cannot hide parameter or FLOP advantages inside the local encoder or byte
decoder.

```text
SE1_baseline_shape_records_v0:
  purpose:
    - define_the_byte_global_control_shape_before_harness_code
    - define_the_first_two_patch_global_shapes_before_harness_code
    - force_hidden_local_compute_into_the_accounting

  manifest_artifact: research/SE1_STUDENT_SHAPE_MANIFEST.md

  shared_budget_invariants:
    nominal_student_total_parameters: 260M
    allowed_student_window: 180M_to_350M
    same_training_bytes: required
    same_training_order_seed: required
    same_validation_slices: required
    same_optimizer_family: required
    same_training_step_or_token_budget: required
    teacher_packets_during_G0: forbidden

  module_parameter_targets:
    local_byte_encoder: 20M_to_35M
    global_reasoner: 170M_to_220M
    local_byte_decoder: 30M_to_50M
    compute_and_eval_heads: 5M_to_15M
    note: unused_local_budget_in_byte_global_control_must_be_reported_not_hidden

  required_report_fields:
    - total_parameters
    - active_parameters_mean
    - active_parameters_p95
    - global_sequence_length_mean
    - global_sequence_length_p95
    - local_encoder_FLOPs
    - global_reasoner_FLOPs
    - local_decoder_FLOPs
    - patch_to_byte_expansion_FLOPs
    - total_active_FLOPs_mean
    - total_active_FLOPs_p95
    - validation_loss
    - D1_byte_exact_match
    - D5_generation_guardrail_score
```

Byte-global control:

```text
SE1_byte_global_control_shape_v0:
  baseline_id: B3_byte_global_same_budget
  patch_size_bytes: 1
  input_units: raw_bytes
  global_sequence_length:
    formula: Tb
    reduction_vs_byte_global: 0

  local_byte_encoder:
    allowed_role:
      - byte_embedding
      - position_or_boundary_features_if_also_counted_in_patch_models
    forbidden_role:
      - learned_patch_compression
      - lookahead_byte_grouping

  global_reasoner:
    input_shape: float[B, Tb, D]
    role: all_semantic_and_surface_sequence_modeling_over_byte_positions
    compute_rule:
      - every_byte_position_is_a_global_token
      - attention_or_state_update_cost_is_counted_over_Tb

  local_byte_decoder:
    allowed_role:
      - byte_logits_from_global_state
    forbidden_role:
      - unreported_auxiliary_language_model
      - teacher_packet_decoder

  accounting_risk:
    - byte_global_may_look_worse_only_because_sequence_tax_is_explicit
    - if_patch_models_use_more_local_compute_the_extra_compute_must_be_charged
```

Patch-global controls:

```text
SE1_patch_global_P4_shape_v0:
  patch_size_bytes: 4
  input_units: fixed_4_byte_spans_before_learned_patch_scouts
  patch_count:
    formula: ceil(Tb / 4)
  global_sequence_length:
    formula: P4
    nominal_reduction_vs_byte_global: about_75_percent_before_ceil_and_padding

  local_byte_encoder:
    input_shape: uint8[B, Tb]
    output_shape: float[B, P4, D]
    required_outputs:
      - patch_states
      - patch_spans
      - residual_flags
    forbidden_role:
      - full_sequence_reasoning_hidden_in_encoder
      - teacher_signature_lookup

  global_reasoner:
    input_shape: float[B, P4, D]
    compute_rule:
      - global_attention_or_state_update_cost_counted_over_P4
      - same_depth_window_as_byte_global_or_reported_depth_adjustment

  local_byte_decoder:
    input_shape:
      - global_patch_states
      - patch_spans
      - residual_flags
      - local_byte_context
    output_shape: float[B, Tb, 256]
    compute_rule:
      - byte_reconstruction_cost_counted_over_Tb
      - residual_copy_or_edit_heads_counted_over_Tb

SE1_patch_global_P8_shape_v0:
  patch_size_bytes: 8
  input_units: fixed_8_byte_spans_before_learned_patch_scouts
  patch_count:
    formula: ceil(Tb / 8)
  global_sequence_length:
    formula: P8
    nominal_reduction_vs_byte_global: about_87_5_percent_before_ceil_and_padding

  local_byte_encoder:
    input_shape: uint8[B, Tb]
    output_shape: float[B, P8, D]
    required_outputs:
      - patch_states
      - patch_spans
      - residual_flags
    extra_risk:
      - larger_patch_may_overcompress_boundary_and_surface_identity

  global_reasoner:
    input_shape: float[B, P8, D]
    compute_rule:
      - global_attention_or_state_update_cost_counted_over_P8
      - same_depth_window_as_byte_global_or_reported_depth_adjustment

  local_byte_decoder:
    input_shape:
      - global_patch_states
      - patch_spans
      - residual_flags
      - local_byte_context
    output_shape: float[B, Tb, 256]
    compute_rule:
      - byte_reconstruction_cost_counted_over_Tb
      - decoder_burden_must_not_be_reported_as_free_global_compression
```

G0 baseline mapping:

```text
SE1_baseline_shapes_to_G0_mapping_v0:
  G0_substrate_global_sequence_reduction:
    compares:
      - SE1_byte_global_control_shape_v0.global_sequence_length
      - SE1_patch_global_P4_shape_v0.global_sequence_length
      - SE1_patch_global_P8_shape_v0.global_sequence_length
    pass_if:
      - observed_global_sequence_reduction_gte_25_percent_vs_byte_global
      - total_active_FLOPs_report_includes_local_encoder_and_decoder
      - no_teacher_packets_used

  G0_substrate_loss_floor:
    compares:
      - byte_global_validation_loss
      - P4_validation_loss
      - P8_validation_loss
    pass_if:
      - best_patch_global_loss_no_worse_than_byte_global_by_more_than_1_percent_relative
      - D1_byte_exact_match_floor_still_passes
      - D5_generation_guardrail_has_no_detectable_collapse

  failure_interpretation:
    P4_fails_and_P8_fails:
      - patch_global_substrate_not_earned
      - do_not_start_teacher_packet_harness
    P4_passes_P8_fails:
      - compression_limit_may_be_between_4_and_8_bytes
      - keep_P4_as_first_student_shape
    P8_passes_but_D1_fails:
      - global_sequence_shortening_destroyed_byte_truth
      - reject_until_local_decoder_or_residual_channel_is_fixed
```

## SE1 Teacher Signature And Surface Correlation

Teacher signatures are candidate-set surfaces, not full vocabulary imitation.
The first goal is to discover whether teachers expose different useful
surfaces, not to average their logits.

```text
SE1_Ksig_signature_vector_v0:
  tensor_shape: float[B, G, M, C, Ksig]
  indexes:
    B: examples
    G: probe_groups
    M: teacher_or_source_id
    C: fixed_candidate_index
    Ksig: signature_fields

  candidate_set_rule:
    C_is_fixed_before_teacher_scoring: true
    allowed_candidates:
      - gold_or_reference_candidate
      - plausible_distractor
      - counterfactual_candidate
      - format_or_constraint_violation_candidate
      - abstain_or_invalid_candidate_when_relevant
    forbidden_candidates:
      - teacher_generated_unbounded_chain_of_thought
      - full_vocabulary_distribution
      - post_hoc_candidate_added_after_seeing_teacher_scores

  Ksig_schema_length: 9
  fields:
    0_candidate_score:
      decoder_teacher: normalized_candidate_logprob
      encoder_teacher: normalized_pair_or_triplet_similarity_score
      exact_oracle: correctness_or_constraint_score
    1_margin_to_best_alternative:
      definition: candidate_score_minus_best_other_candidate_score
    2_rank_within_candidate_set:
      definition: one_based_rank_after_descending_score
    3_probability_within_candidate_set:
      definition: softmax_over_fixed_candidate_scores_if_scores_are_comparable
      missing_value: masked_for_nonprobabilistic_sources
    4_entropy_over_candidate_set:
      definition: entropy_of_candidate_set_distribution
      missing_value: masked_for_nonprobabilistic_sources
    5_teacher_confidence_z:
      definition: confidence_standardized_within_teacher_and_probe_group
    6_abstain_or_invalid_flag:
      definition: source_marks_candidate_or_prompt_as_invalid_unanswerable_or_out_of_scope
    7_probe_delta_from_base:
      definition: candidate_score_change_from_base_prompt_to_probe_variant
    8_error_or_correctness_class:
      definition: exact_or_parser_error_class_when_available_else_masked

  masking_rule:
    - every_missing_field_requires_explicit_boolean_mask
    - masked_fields_are_excluded_pairwise_not_zero_filled
```

Surface vector construction:

```text
SE1_teacher_surface_vector_v0:
  primary_decoder_surface:
    source_fields:
      - margin_to_best_alternative
      - rank_within_candidate_set
      - entropy_over_candidate_set
      - probe_delta_from_base
    normalization:
      - z_score_within_teacher_and_probe_group
      - pairwise_complete_mask_only

  encoder_surface:
    source_fields:
      - candidate_score
      - margin_to_best_alternative
      - rank_within_candidate_set
      - probe_delta_from_base
    normalization:
      - z_score_within_teacher_and_probe_group
      - report_separately_from_decoder_primary_if_candidate_semantics_differ

  exact_oracle_surface:
    source_fields:
      - correctness_or_constraint_score
      - abstain_or_invalid_flag
      - error_or_correctness_class
    rule:
      - do_not_force_probability_correlation_against_decoder_teachers
      - compare_as_disagreement_overlap_and_error_localization_value
```

Surface correlation metric:

```text
SE1_surface_correlation_metric_v0:
  primary_metric:
    name: spearman_rank_correlation
    vector: teacher_surface_vector.primary_decoder_surface
    rationale: robust_to_score_scale_differences_between_teacher_families

  secondary_metrics:
    - pearson_correlation_on_z_scored_margin_vector
    - cosine_similarity_on_z_scored_surface_vector
    - disagreement_overlap_on_top_ranked_candidate
    - error_class_overlap_for_exact_oracle_sources

  pairwise_rules:
    minimum_pairwise_complete_rows: 500
    minimum_probe_groups: 3
    bootstrap_resamples: 1000
    confidence_interval: 95_percent
    inconclusive_if:
      - pairwise_complete_rows_lt_500
      - bootstrap_CI_crosses_0_90_for_primary_metric
      - candidate_sets_not_identical_for_compared_decoder_surfaces

  G1_distinct_surface_gate:
    threshold: at_least_2_teacher_pairs_have_primary_correlation_lte_0_90
    stricter_count_rule:
      - count_pair_only_if_bootstrap_CI_high_lte_0_90
      - count_pair_only_if_disagreement_examples_are_not_concentrated_in_one_prompt_template
    fail_means:
      - candidate_teachers_are_redundant_echoes_for_this_probe_set
      - do_not_treat_same_family_agreement_as_independent_evidence

  required_report_fields:
    - teacher_pair
    - probe_groups
    - pairwise_complete_rows
    - primary_spearman_rho
    - primary_spearman_bootstrap_CI
    - secondary_pearson
    - secondary_cosine
    - disagreement_overlap
    - top_disagreement_examples_manifest
    - correlation_failure_interpretation
```

G1 mapping:

```text
SE1_Ksig_to_G1_mapping_v0:
  G1_teacher_profile_distinct_surface:
    uses:
      - SE1_Ksig_signature_vector_v0
      - SE1_teacher_surface_vector_v0
      - SE1_surface_correlation_metric_v0
    pass_if:
      - at_least_2_teacher_pairs_have_primary_correlation_lte_0_90
      - counted_pairs_have_CI_high_lte_0_90
      - disagreement_examples_are_source_interpretable
    inconclusive_means:
      - increase_probe_rows_or_probe_diversity_before_training_packets
    fail_means:
      - teacher_roster_or_probe_set_is_not_diverse_enough_for_Eklavya
      - do_not_start_multi_teacher_packet_training
```
