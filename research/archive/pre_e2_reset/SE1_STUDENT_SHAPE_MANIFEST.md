# SE1 Student Shape Manifest

This manifest materializes the B3, P4, and P8 student shapes named in
`EKLAVYA_BENCHMARK_CONTRACT.md`.

It is not a harness result. It is the pre-code configuration contract that a
future shape-accounting script must instantiate and verify.

## Manifest Status

```text
SE1_student_shape_manifest_2026_06_20_v0:
  source_contract:
    - SE1_baseline_shape_records_v0
    - SE1_byte_global_control_shape_v0
    - SE1_patch_global_P4_shape_v0
    - SE1_patch_global_P8_shape_v0
    - SE1_baseline_shapes_to_G0_mapping_v0

  status: config_files_and_smoke_accounting_created_no_real_G0_evidence
  harness_status: not_run
  training_status: not_started
  teacher_packets_allowed: false
  purpose:
    - freeze_the_first_three_student_shapes_before_implementation
    - prevent_patch_models_from_hiding_compute_in_local_modules
    - give_G0_a_concrete_shape_accounting_target
```

## Shared Invariants

```text
SE1_shape_shared_invariants_v0:
  comparison_group:
    - B3_byte_global_same_budget
    - P4_patch_global_fixed_4_byte
    - P8_patch_global_fixed_8_byte

  input_truth:
    input_units: raw_utf8_bytes
    byte_vocabulary_size: 256
    normalization_allowed: false
    tokenizer_dependency_allowed: false

  train_eval_invariants:
    same_training_byte_stream: required
    same_training_order_seed: required
    same_validation_slices: required
    same_optimizer_family: required
    same_training_step_or_token_budget: required
    teacher_packets_during_G0: forbidden
    inference_teacher_calls: forbidden

  nominal_budget:
    total_parameter_target: 260M
    allowed_total_window: 180M_to_350M
    local_encoder_target: 20M_to_35M
    global_reasoner_target: 170M_to_220M
    local_decoder_target: 30M_to_50M
    heads_and_controllers_target: 5M_to_15M

  report_before_any_G0_claim:
    - actual_total_parameters
    - actual_trainable_parameters
    - actual_active_parameters_mean
    - actual_active_parameters_p95
    - actual_global_sequence_length_mean
    - actual_global_sequence_length_p95
    - actual_local_encoder_FLOPs
    - actual_global_reasoner_FLOPs
    - actual_local_decoder_FLOPs
    - actual_patch_to_byte_expansion_FLOPs
    - actual_total_active_FLOPs_mean
    - actual_total_active_FLOPs_p95
```

## Shared Module Defaults

These defaults define the first instantiation target. They are allowed to change
only by editing this manifest and the benchmark contract in the same commit.

```text
SE1_shared_module_defaults_v0:
  global_reasoner:
    family: compact_transformer
    layers: 14
    d_model: 896
    attention_heads: 14
    head_dim: 64
    kv_heads: 7
    mlp_family: swiglu
    mlp_intermediate_target: 4096
    parameter_intent: keep_global_reasoner_inside_170M_to_220M_target_band_after_counter_verification
    norm: rmsnorm
    position_encoding: rope_or_relative_position_equivalent
    dropout: 0_for_first_shape_accounting

  byte_output:
    output_distribution: byte_logits
    byte_classes: 256
    required_output_shape: float[B, Tb, 256]

  compute_controller:
    active_for_G0: false
    reason: G0_must_first_test_static_substrate_shape
    allowed_outputs_to_log:
      - exit_layer
      - expected_FLOPs

  generation_guardrail:
    required_for_claim: true
    current_status: prompt_set_not_materialized
```

## Shape B3: Byte-Global Control

```text
SE1_shape_B3_byte_global_same_budget_v0:
  manifest_id: B3_byte_global_same_budget
  patch_size_bytes: 1
  patching: none
  global_sequence_length_formula: Tb
  nominal_reduction_vs_byte_global: 0

  byte_input_adapter:
    input_shape: uint8[B, Tb]
    output_shape: float[B, Tb, 896]
    allowed_components:
      - byte_embedding
      - byte_position_or_boundary_feature_if_shared_with_patch_shapes
    forbidden_components:
      - learned_patch_compression
      - lookahead_grouping
      - local_sequence_reasoner_that_reduces_Tb

  global_reasoner:
    defaults_ref: SE1_shared_module_defaults_v0.global_reasoner
    input_shape: float[B, Tb, 896]
    output_shape: float[B, Tb, 896]
    cost_axis: Tb

  byte_decoder:
    input_shape: float[B, Tb, 896]
    output_shape: float[B, Tb, 256]
    allowed_components:
      - byte_lm_head
      - shallow_surface_adapter_if_also_counted_for_patch_shapes
    forbidden_components:
      - hidden_auxiliary_language_model
      - teacher_packet_decoder

  parameter_match_rule:
    - if_actual_params_are_more_than_1_percent_below_patch_shape_total_report_under_budget_delta
    - if_needed_add_only_global_reasoner_capacity_to_nearest_non_exceeding_budget_control
    - do_not_add_local_patch_compression_to_byte_global_control
```

## Shape P4: Fixed 4-Byte Patch-Global

```text
SE1_shape_P4_patch_global_fixed_4_byte_v0:
  manifest_id: P4_patch_global_fixed_4_byte
  patch_size_bytes: 4
  patch_count_formula: ceil(Tb / 4)
  global_sequence_length_formula: P4
  nominal_reduction_vs_byte_global: about_75_percent_before_ceil_and_padding

  local_byte_encoder:
    input_shape: uint8[B, Tb]
    span_shape: int[B, P4, 2]
    output_shape: float[B, P4, 896]
    required_outputs:
      - patch_states
      - patch_spans
      - residual_flags
    allowed_components:
      - byte_embedding
      - small_depthwise_or_causal_local_mixer
      - fixed_span_pooling_or_projection
      - residual_flag_head
    forbidden_components:
      - global_attention_over_Tb
      - teacher_signature_lookup
      - hidden_reasoning_stack

  global_reasoner:
    defaults_ref: SE1_shared_module_defaults_v0.global_reasoner
    input_shape: float[B, P4, 896]
    output_shape: float[B, P4, 896]
    cost_axis: P4

  local_byte_decoder:
    input_shape:
      - float[B, P4, 896]
      - int[B, P4, 2]
      - bool[B, Tb]
      - local_byte_context
    output_shape: float[B, Tb, 256]
    required_behavior:
      - reconstruct_or_generate_every_original_byte_position
      - preserve_residual_or_boundary_sensitive_bytes
      - expose_decoder_FLOPs_over_Tb
```

## Shape P8: Fixed 8-Byte Patch-Global

```text
SE1_shape_P8_patch_global_fixed_8_byte_v0:
  manifest_id: P8_patch_global_fixed_8_byte
  patch_size_bytes: 8
  patch_count_formula: ceil(Tb / 8)
  global_sequence_length_formula: P8
  nominal_reduction_vs_byte_global: about_87_5_percent_before_ceil_and_padding

  local_byte_encoder:
    input_shape: uint8[B, Tb]
    span_shape: int[B, P8, 2]
    output_shape: float[B, P8, 896]
    required_outputs:
      - patch_states
      - patch_spans
      - residual_flags
    allowed_components:
      - byte_embedding
      - small_depthwise_or_causal_local_mixer
      - fixed_span_pooling_or_projection
      - residual_flag_head
    forbidden_components:
      - global_attention_over_Tb
      - teacher_signature_lookup
      - hidden_reasoning_stack
    known_risk:
      - larger_patch_can_destroy_boundary_and_surface_identity

  global_reasoner:
    defaults_ref: SE1_shared_module_defaults_v0.global_reasoner
    input_shape: float[B, P8, 896]
    output_shape: float[B, P8, 896]
    cost_axis: P8

  local_byte_decoder:
    input_shape:
      - float[B, P8, 896]
      - int[B, P8, 2]
      - bool[B, Tb]
      - local_byte_context
    output_shape: float[B, Tb, 256]
    required_behavior:
      - reconstruct_or_generate_every_original_byte_position
      - preserve_residual_or_boundary_sensitive_bytes
      - expose_decoder_FLOPs_over_Tb
    extra_accounting_rule:
      - decoder_burden_must_not_be_reported_as_free_global_compression
```

## Expected Sequence Accounting

```text
SE1_shape_sequence_accounting_v0:
  formulas:
    B3_global_length: Tb
    P4_global_length: ceil(Tb / 4)
    P8_global_length: ceil(Tb / 8)

  reduction_formula:
    patch_reduction_vs_byte_global: 1 - patch_global_length / Tb

  report_by_slice:
    - D1_local_byte_exactness
    - D2_arithmetic_candidate_reasoning
    - D5_generation_slice

  required_statistics:
    - mean_Tb
    - p50_Tb
    - p95_Tb
    - mean_global_length_B3
    - mean_global_length_P4
    - mean_global_length_P8
    - p95_global_length_B3
    - p95_global_length_P4
    - p95_global_length_P8
    - mean_reduction_P4
    - mean_reduction_P8
```

## G0 Materialization Checklist

```text
SE1_shape_manifest_to_G0_checklist_v0:
  completed_smoke_artifacts:
    - instantiate_B3_P4_P8_config_files_from_this_manifest
    - write_parameter_counter
    - write_FLOP_counter_that_counts_local_encoder_global_reasoner_and_decoder
    - write_sequence_length_counter_for_synthetic_smoke_profile
    - confirm_teacher_packets_disabled

  created_artifacts:
    shape_config_dir: artifacts/shape_manifest/configs/
    parameter_report: artifacts/shape_manifest/parameter_report.json
    sequence_report: artifacts/shape_manifest/sequence_report.json
    flop_report: artifacts/shape_manifest/flop_report.json

  still_required_for_real_G0:
    - replace_synthetic_smoke_profile_with_real_D1_D2_D5_byte_lengths
    - rerun_parameter_sequence_and_FLOP_reports_against_real_profiles
    - attach_reports_to_G0_gate_rows

  still_forbidden:
    - training
    - harness_pass_claim
    - teacher_packet_consumption
    - reporting_patch_global_as_better_without_D1_and_D5_guardrails
```
