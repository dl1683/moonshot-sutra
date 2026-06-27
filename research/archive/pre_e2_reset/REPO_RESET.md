# Repo Reset Notes

## 2026-06-20 Ruthless Trim

The project drifted again. The work over-expanded into a named protocol and
paper governance system. That material is now removed from the active research
surface.

The active project is only:

```text
Eklavya = multiple-teacher invariant learning
Sutra = the efficient student/system built from Eklavya
```

## What Was Deleted Conceptually

Removed from active design:

- protocol constitution language;
- adjudication packets;
- warrant packets;
- quarantine/release machinery;
- paper-play permission layers;
- ceremony around claims that did not directly improve Eklavya or Sutra.

Kept as small design constraints:

- every component must justify its outcome;
- teacher disagreement cannot be averaged blindly;
- efficiency claims require total lifecycle cost;
- evidence must distinguish multi-teacher learning from ordinary KD.

## Current Four-File Discipline

- `research/EKLAVYA_DOCTRINE.md`: main Eklavya/Sutra design doctrine.
- `research/EKLAVYA_BENCHMARK_CONTRACT.md`: scratchpad, falsifiers, candidate
  designs, and experiment definitions.
- `research/EKLAVYA_SOURCE_EVIDENCE.md`: source map and live questions.
- `research/REPO_RESET.md`: reset ledger and working rules.

No new document should be created unless these four cannot absorb the work.

## Working Rules

1. Design Eklavya and Sutra only.
2. Treat legacy code/results as evidence, not as destiny.
3. Do not assume Sutra-Dyad is the right student.
4. Do not assume byte-global computation is right.
5. Do not use Eklavya as a synonym for multi-teacher KD.
6. Do not treat teacher count as teacher diversity.
7. Do not average teachers unless averaging is the thing being falsified.
8. Do not optimize a benchmark before the architecture hypothesis is clear.
9. Delete or compress anything that does not help the design.

## Current Best Direction

```text
Sutra:
  byte-local / token-or-patch-global
  sparse or conditional capacity
  early-exit or self-speculative inference
  component-level improvability

Eklavya:
  teacher family registry
  teacher ports
  invariant compiler
  disagreement-driven curriculum
  retained-gain tests after teacher removal
```

## Next Deep Thinking Target

Design the first concrete Sutra-Eklavya architecture on paper:

- exact student substrate;
- teacher families;
- teacher ports;
- invariant packet types;
- training phases;
- active compute mechanism;
- retained-gain metric;
- honest baselines.

## 2026-06-20 SE1 Design Pass

The active design now has a concrete name: **Sutra-Eklavya SE1**.

SE1 means:

```text
byte truth at the boundary
learned patch or token-compatible global sequence
compact global reasoner
conditional compute
teacher tomography
invariant packets
teacher axis preserved
retained gain after teacher removal
```

Current design judgment:

- Sutra should not be byte-global by default.
- Sutra should not abandon bytes either.
- Eklavya should not average teachers early.
- Eklavya should compile teacher behavior surfaces into landing-zone packets.
- Geometry is useful as a diagnostic, not as a promise of capability transfer.
- Fractal or scale-separated representations are allowed only as measured
  efficiency hypotheses.

Next target:

```text
write the full SE1 architecture record:
  exact modules
  tensor interfaces
  packet formats
  training phase order
  retained gain metric
  first honest comparison suite
```

## 2026-06-20 SE1 Interface Pass

This pass made SE1 mechanically sharper.

Added design objects:

```text
SE1_core_objects:
  raw_byte_batch
  local_byte_encoder_output
  global_reasoner_output
  local_byte_decoder_output
  compute_controller_output
  SE1_packet_envelope
  behavior_signature_payload
  geometry_anchor_payload
  boundary_payload
  error_localization_payload
  curriculum_payload
```

The most important current rule:

```text
every_teacher_signal_must_name:
  landing_zone
  target_loss
  corruption_risk
  removal_test
```

Open holes that remain:

- exact first scale target;
- first concrete teacher IDs;
- first task slices;
- loss schedule;
- module diagram;
- cost accounting constants.

## 2026-06-20 SE1 Scale And Roster Pass

This pass closed the first paper-design holes without returning to protocol
bureaucracy.

Decisions:

- first student target is 180M to 350M total parameters, with roughly 90M to
  180M active parameters on typical examples;
- first target is a byte-preserving patch-global student, not a byte-global
  transformer and not a 3B student;
- teacher calls are offline signature generation only;
- inference teacher calls are forbidden;
- decoder roster begins with primary-family-size controls and SmolLM3 as an efficiency
  peer;
- encoder roster begins with embedding candidate 0.6B or BGE-M3, with BGE-small as a
  scale/control anchor;
- verifier teachers should be exact task oracles whenever possible, not another
  language model by default;
- boundary teachers are objectives and adversarial cases from BLT, MEGABYTE,
  Charformer, and tokenization-limit work;
- curriculum teachers are high-density examples, rationales, and contrast sets,
  not a permission system.

Drafted contracts:

```text
SE1_added_contracts:
  first_scale_target: drafted
  candidate_teacher_roster: drafted
  first_task_slices: drafted
  loss_schedule: drafted
  cost_accounting_constants: drafted
```

Open holes that remain:

- module diagram;
- exact teacher choice after hardware constraints;
- exact first dataset/task manifest;
- exact loss weights and stage lengths after scout runs.

## 2026-06-20 SE1 Compiler And Runtime Pass

This pass reframed SE1 as three coupled systems:

```text
teacher_profiler -> invariant_compiler -> teacher_free_sutra_runtime
```

New design commitments:

- teacher signatures are profile data, not inference dependencies;
- invariant packets are compiled training objects with landing zones;
- Sutra's compute controller is a cost-based planner over actions, not a vague
  router;
- teacher disagreement is both curriculum signal and future compute-difficulty
  profile;
- first datasets are organized by claims and packet types, not by benchmark
  prestige;
- teachers enter only through distinct-surface, student-gap, landing-zone, cost,
  and removal gates.

External analogies imported:

- compiler profile-guided optimization: representative behavior guides later
  static optimization;
- database cost-based planning: equivalent plans compete by total cost;
- query-by-committee: disagreement identifies informative examples;
- curriculum learning: order and contrast can be teacher signal;
- early-exit/adaptive-compute systems: easy cases should be cheap and hard cases
  should retain quality.

Additional sibling lessons imported:

- DKS: packet ledgers need identity, provenance, and revision discipline;
- CTI: geometry can be a scoped state sensor but must carry scope gates;
- complex/fractal adapters: constrained update spaces may matter after the first
  retained-gain proof.

Drafted contracts:

```text
SE1_added_contracts_this_pass:
  module_graph: drafted
  compute_planner_contract: drafted
  first_dataset_task_manifest: drafted
  teacher_selection_gates: drafted
  first_teacher_decision_rules: drafted
```

Open holes that remain:

- exact public dataset IDs after license and size review;
- exact teacher choice after available-hardware review;
- exact loss weights and stage lengths after scout runs;
- exact parameter split between local encoder, global reasoner, local decoder,
  and compute controller.

## 2026-06-20 SE1 Utility Calculus Pass

This pass added the economics that were missing.

New central rule:

```text
every_SE1_component_competes_by_net_retained_gain
```

Meaning:

- teacher packets are information purchases;
- probes are bought only if they change a useful decision;
- dataset rows are teaching objects, not generic examples;
- losses earn gradient budget only after they show retained value;
- parameters and inference FLOPs go where marginal net retained gain is highest.

New design objects:

```text
SE1_added_objects_this_pass:
  net_retained_gain
  packet_value_prior
  packet_value_realized
  packet_forecast_error
  probe_value
  machine_teaching_contract
  parameter_split_prior
  patch_ladder
  loss_stage_prior
  promotion_gates
```

Research imported:

- information bottleneck: compress only while preserving relevant information;
- MDL: complexity counts against the model;
- value of information: information is valuable only relative to decisions and
  costs;
- robust expected information gain: probe ranking should penalize ambiguity;
- machine teaching and teaching dimension: construct the smallest lesson that
  teaches the target structure to the actual learner.

Legacy evidence re-emphasized:

- previous multi-teacher presence did not prove Eklavya;
- student capacity and byte-global sequence tax were likely bottlenecks;
- patch geometry and decoder burden are first-order, not tuning trivia;
- alignment packets should be skipped when the student already owns the
  geometry.

Open holes that remain:

- exact dataset IDs;
- exact teacher IDs after hardware/cost scoring;
- exact patch ladder subset if only two choices can be scouted;
- exact loss weights after measuring gradient conflict;
- exact thresholds for the six promotion gates.

## 2026-06-20 SE1 Readiness Lock Pass

This pass closed the previous paper-design holes as scout defaults, not as
permission to train.

New commitments:

- exact dataset candidates now exist for each SE1 slice;
- Common Pile style licensed/open-domain sources are preferred before broader
  web-crawl text for the first base-text scout;
- ARC and GSM8K are the first candidate-reasoning sources;
- HumanEval, APPS, GSM8K, and JSONSchemaBench are the first verifier/oracle
  sources;
- TinyStories and Dolly-style data are generation guardrails, not the core
  Eklavya claim;
- teacher choice is tiered by actual hardware and signature cost;
- primary family, SmolLM3, embedding candidate, and BGE candidates must be pinned by revision,
  license, runtime mode, and distinct probe surface;
- if only two fixed patch choices can be run, P4 and P8 are the first scout
  because they expose the sequence-tax/local-decoder tradeoff;
- loss-balancing tricks are rejected by default and admitted only after measured
  gradient conflict;
- numeric thresholds now exist for G0 through G5 as scout gates.

Current boundary:

```text
next_step_is_pre_harness_readiness_package_not_training
```

Open holes that remain:

- verify current licenses and download sizes for every candidate dataset before
  download;
- inspect available hardware before final teacher IDs are admitted;
- pin exact model revisions and quantization/runtime modes;
- write the pre-harness readiness package from the benchmark contract;
- do not start an implementation harness until that package passes review.

## 2026-06-20 Dataset Metadata Audit

Queried current Hugging Face dataset metadata for the first SE1 source
candidates.

Result:

- default admit after size review: `openai/gsm8k`,
  `openai/openai_humaneval`, `codeparrot/apps`,
  `epfl-dlab/JSONSchemaBench`;
- admit only with attribution/share-alike review:
  `allenai/ai2_arc`, `HuggingFaceFW/fineweb-edu`,
  `roneneldan/TinyStories`, `databricks/databricks-dolly-15k`;
- do not admit until license ambiguity is resolved: `allenai/openbookqa`,
  `allenai/scitail`;
- do not use as a default training claim source: `allenai/scifact`, because the
  current metadata is noncommercial;
- Common Pile remains preferred as a source family, but individual chosen shards
  need underlying-source license verification because the tested dataset cards
  did not expose license metadata directly.

Open holes that remain:

- exact planned subset row counts;
- exact train/eval split policy per admitted slice;
- model revisions, quantization, and runtime cost;
- pre-harness readiness package.

## 2026-06-20 Pre-Harness Readiness Package

The benchmark contract now contains the first explicit SE1 pre-harness
readiness package.

Current verdict:

```text
status: not_ready_for_harness
```

What is now pinned:

- candidate dataset revisions for the first admitted, share-alike-review, and
  question-only sources;
- current local teacher model revision pins for the single-24GB GPU tier;
- default local teacher tier based on the RTX 5090 Laptop GPU;
- P4 and P8 as first fixed-patch falsifiers;
- fixed loss schedule as default, with gradient-conflict triggers before
  adaptive balancing;
- numeric scout thresholds for G0 through G5.

What remains before any harness:

- final dataset slice table with planned row counts, split policy, license
  status, packet target, and falsifier;
- teacher runtime table with quantization, memory peak, throughput, and cost per
  1000 signature examples;
- packet ledger table with source identity, landing zone, target loss,
  corruption risk, refused signal, removal test, value, and demotion rule;
- gate report table with each threshold tied to an evidence artifact.

Current boundary:

```text
do_not_build_harness_until_G0_and_G1_readiness_rows_are_filled
```

## 2026-06-20 Dataset Size Metadata Pass

Queried Hugging Face dataset and datasets-server metadata for the first small
candidate sources.

Now recorded in the benchmark contract:

- `openai/gsm8k` main: 8,792 rows, 2,725,633 parquet bytes;
- `openai/openai_humaneval`: 164 rows, 83,920 parquet bytes;
- `epfl-dlab/JSONSchemaBench` `Github_easy`: 1,938 rows, 540,610 parquet bytes;
- `allenai/ai2_arc` challenge/easy: row counts and parquet bytes recorded, but
  still held for share-alike review;
- `google-research-datasets/paws` labeled configs: row counts and parquet bytes
  recorded, but still held for license terms review;
- `roneneldan/TinyStories`: 2,141,709 rows, 1,000,775,442 parquet bytes;
- `databricks/databricks-dolly-15k`: 15,011 rows, 7,747,823 parquet bytes, but
  still held for share-alike review.

Open holes that remain:

- final license decisions for held/share-alike sources;
- Common Pile shard-level source license and size review;
- teacher runtime table;
- packet ledger table;
- gate report table.

## 2026-06-20 Planned Slice Policy Pass

The benchmark contract now records planned row counts and split policies for the
first admitted or candidate-held slices.

Current planned admitted/core slices:

- local byte exactness: 2,500 generated rows across Unicode, paths, rare names,
  mixed scripts, and tokenizer collision cases, split 60/20/20;
- HumanEval: all 164 rows, eval/oracle only;
- GSM8K main: existing train/test split, 7,473 train and 1,319 test rows;
- JSONSchemaBench `Github_easy`: existing train/val/test split, 1,170 train,
  191 validation, 577 test rows.

Current held/capped slices:

- AI2 ARC remains held until share-alike implications and prompt format are
  reviewed;
- PAWS remains question/eval-only until license terms are reviewed;
- TinyStories has a 50,000 train / 5,000 validation cap after license review,
  and is only a generation-collapse guardrail;
- Dolly remains held until share-alike implications and instruction-guardrail
  need are reviewed;
- base text remains not admitted until Common Pile shard-level source license
  and size review is done.

Open holes that remain:

- Common Pile shard-level source license and size review;
- final license decisions for held/share-alike sources;
- teacher runtime measurement table;
- packet ledger rows populated from the new template;
- gate report rows populated from the new template.

## 2026-06-20 Teacher Metadata Audit

Queried current Hugging Face model metadata for the first teacher candidates.

Result:

- primary family decoder and embedding candidates are currently Apache 2.0 and ungated;
- `HuggingFaceTB/SmolLM3-3B-Base` is currently Apache 2.0 and ungated;
- `BAAI/bge-m3` and `BAAI/bge-small-en-v1.5` are currently MIT and ungated;
- the local default remains the single-24GB tier, so first scout candidates are
  `anchor-decoder-1.7B`, `control-decoder-0.6B`,
  `HuggingFaceTB/SmolLM3-3B-Base` if distinct, `embedding-candidate-0.6B`,
  `BAAI/bge-m3`, and `BAAI/bge-small-en-v1.5`.

Open holes that remain:

- local runtime fit and quantization choices;
- signature throughput;
- cost per 1000 signature examples;
- pre-harness readiness package.

## 2026-06-20 Local Hardware Audit

`nvidia-smi` reports:

```text
gpu: NVIDIA GeForce RTX 5090 Laptop GPU
vram: 24463 MiB
driver: 595.79
```

Current consequence:

```text
SE1_default_teacher_tier: single_24gb_gpu
```

Default local scout posture:

- decoder: start with `anchor-decoder-1.7B`, with `control-decoder-0.6B` as small
  control;
- decoder peer: include `HuggingFaceTB/SmolLM3-3B-Base` only if it passes the
  distinct-surface gate;
- encoder: start with `embedding-candidate-0.6B` or `BAAI/bge-m3`, with
  `BAAI/bge-small-en-v1.5` as a small/control anchor;
- do not make 4B-class teachers the first local default unless the measured
  cost per signature example justifies it.

Open holes that remain:

- exact planned subset row counts;
- exact train/eval split policy per admitted slice;
- quantization and runtime cost;
- pre-harness readiness package.

## 2026-06-20 Teacher Footprint Metadata Pass

Queried Hugging Face model file metadata for the default single-24GB teacher
candidates. The benchmark contract now records full revisions and safetensors
weight footprints.

Recorded weight footprints:

- `control-decoder-0.6B`: 1,503,300,328 safetensors bytes;
- `anchor-decoder-1.7B`: 4,063,515,592 safetensors bytes;
- `HuggingFaceTB/SmolLM3-3B-Base`: 6,150,235,008 safetensors bytes, with large
  repo storage due to ONNX variants;
- `embedding-candidate-0.6B`: 1,191,586,416 safetensors bytes;
- `BAAI/bge-m3`: 2,271,145,830 safetensors bytes;
- `BAAI/bge-small-en-v1.5`: 266,974,701 safetensors bytes.

Current measurement order:

1. `BAAI/bge-small-en-v1.5`
2. `control-decoder-0.6B`
3. `embedding-candidate-0.6B`
4. `anchor-decoder-1.7B`
5. `BAAI/bge-m3`
6. `HuggingFaceTB/SmolLM3-3B-Base` only if the distinct-surface need remains.

Open holes that remain:

- exact downloaded file allowlist per model;
- quantization and runtime mode;
- local peak VRAM;
- signature examples per second;
- cost per 1000 signature examples;
- packet ledger rows populated from template;
- gate report rows populated from template.

## 2026-06-20 Packet Ledger And Gate Template Pass

The benchmark contract now includes:

- `SE1_packet_ledger_template_v0`;
- required first packet rows before harness;
- `SE1_gate_report_template_v0`;
- initial G0 through G5 gate rows.

Current consequence:

```text
packet_and_gate_schema_status: template_ready_rows_unfilled
```

Open holes that remain:

- materialize evidence artifacts referenced by the first G0/G1 rows;
- teacher runtime measurement table;
- Common Pile shard-level source license and size review;
- final license decisions for held/share-alike sources.

## 2026-06-20 Packet And Gate Row Population Pass

The benchmark contract now fills the first required packet rows:

- `pkt_boundary_local_generated_byte_exactness_v0`;
- `pkt_behavior_gsm8k_anchor17_margin_v0`;
- `pkt_behavior_gsm8k_control06_scale_control_v0`;
- `pkt_error_gsm8k_exact_oracle_v0`;
- `pkt_error_jsonschema_github_easy_oracle_v0`;
- `pkt_geometry_encoder_gap_probe_v0`.

It also fills concrete G0/G1 pre-harness gate rows. All rows are explicitly
`not_run` and point to missing evidence artifacts rather than pretending
readiness evidence exists.

Current status:

```text
packet_rows: populated_prior_only_realized_value_unset
G0_G1_gate_rows: populated_not_run
G2_to_G5: template_only_not_first_harness_claims
```

Open holes that remain:

- generate D1 rows only after harness phase is approved;
- write byte-global, P4, and P8 baseline shape records;
- fill teacher runtime table with measured throughput, peak VRAM, runtime mode,
  and cost;
- define `Ksig` and teacher-surface correlation metric;
- Common Pile shard-level source license and size review;
- final license decisions for held/share-alike sources.

## 2026-06-20 D1 Byte Exactness Spec Pass

The benchmark contract now defines `SE1_D1_local_byte_exactness_spec_v0`.

New design commitments:

- D1 has 2,500 local-generated rows across five families:
  Unicode confusables, filesystem paths, rare names, mixed-script strings, and
  tokenizer collision cases;
- split is fixed at 60/20/20;
- row schema includes `raw_text`, `raw_bytes_hex`, `byte_length`, stress type,
  and forbidden normalizations;
- primary metric is byte exact match with a 0.999 test floor;
- every family must stay above 0.995 exact match;
- forbidden normalization rate must be zero;
- evidence artifact paths are specified under `artifacts/d1_byte_exactness/`,
  but artifacts are not created yet.

Current status:

```text
D1_spec_status: designed_not_generated
```

Open holes that remain:

- write byte-global, P4, and P8 baseline shape records;
- fill teacher runtime table with measured throughput, peak VRAM, runtime mode,
  and cost;
- define `Ksig` and teacher-surface correlation metric;
- Common Pile shard-level source license and size review;
- final license decisions for held/share-alike sources.

## 2026-06-20 Baseline Shape And Signature Metric Pass

The benchmark contract now defines the pre-harness shape records and teacher
surface metric that the G0/G1 rows were waiting on.

Added contract records:

- `SE1_baseline_shape_records_v0`;
- `SE1_byte_global_control_shape_v0`;
- `SE1_patch_global_P4_shape_v0`;
- `SE1_patch_global_P8_shape_v0`;
- `SE1_baseline_shapes_to_G0_mapping_v0`;
- `SE1_Ksig_signature_vector_v0`;
- `SE1_teacher_surface_vector_v0`;
- `SE1_surface_correlation_metric_v0`;
- `SE1_Ksig_to_G1_mapping_v0`.

Current status:

```text
baseline_shapes: defined_not_materialized
Ksig_surface_metric: defined_not_materialized
G0_G1_gate_rows: still_not_run_artifacts_missing
```

Open holes that remain:

- materialize shape manifest for byte-global, P4, and P8 from live configs;
- run G0 shape accounting on D1, D2, and D5 scout rows;
- fill teacher runtime table with measured throughput, peak VRAM, runtime mode,
  and cost;
- materialize first `Ksig` signature tensors and pairwise surface-correlation
  reports;
- Common Pile shard-level source license and size review;
- final license decisions for held/share-alike sources.

## 2026-06-20 Student Shape Manifest Pass

Added `research/SE1_STUDENT_SHAPE_MANIFEST.md` as the canonical B3/P4/P8
shape manifest.

The manifest fixes:

- shared G0 invariants across byte-global, P4, and P8;
- nominal module defaults for the first instantiation target;
- the byte-global B3 control shape;
- the fixed 4-byte patch-global P4 shape;
- the fixed 8-byte patch-global P8 shape;
- the sequence-accounting formulas and per-slice report fields;
- the first artifact paths for configs, parameter report, sequence report, and
  FLOP report.

Current status:

```text
shape_manifest: written_nominal_no_code_instantiated
G0_shape_configs: not_created
G0_shape_accounting_reports: not_created
```

Open holes that remain:

- instantiate B3, P4, and P8 config files from
  `research/SE1_STUDENT_SHAPE_MANIFEST.md`;
- write parameter, FLOP, and sequence-length counters;
- run G0 shape accounting on D1, D2, and D5 scout rows;
- fill teacher runtime table with measured throughput, peak VRAM, runtime mode,
  and cost;
- materialize first `Ksig` signature tensors and pairwise surface-correlation
  reports;
- Common Pile shard-level source license and size review;
- final license decisions for held/share-alike sources.

## 2026-06-20 Dataset License Review Pass

Added `research/SE1_DATASET_LICENSE_REVIEW.md` and wired it into the README,
benchmark contract, and source-evidence map.

Current dataset decisions:

- `common-pile/arxiv_abstracts`, `common-pile/caselaw_access_project`, and
  `common-pile/biodiversity_heritage_library` are D0 candidates only after a
  row-license filter manifest exists;
- `common-pile/arxiv_papers` remains held until a row-level license histogram
  and exclusion report exist;
- `openai/gsm8k`, `openai/openai_humaneval`, `codeparrot/apps`, and
  `epfl-dlab/JSONSchemaBench` remain default candidates after planned subsets
  are written;
- `allenai/ai2_arc`, `roneneldan/TinyStories`, and
  `databricks/databricks-dolly-15k` remain held for share-alike/sharing review;
- `HuggingFaceFW/fineweb-edu` remains a scale ablation, not default D0 base
  text;
- `allenai/scifact` is evidence/eval-only by default because of noncommercial
  metadata;
- `allenai/openbookqa` and `allenai/scitail` remain held until license metadata
  is resolved.

Current status:

```text
dataset_license_review: written
D0_common_pile_row_filter_manifest: not_created
D0_common_pile_license_histogram: not_created
sharealike_review_artifact: not_created
```

Open holes that remain:

- create D0 Common Pile row-filter manifest and license histogram before any
  D0 download;
- instantiate B3, P4, and P8 config files from
  `research/SE1_STUDENT_SHAPE_MANIFEST.md`;
- write parameter, FLOP, and sequence-length counters;
- run G0 shape accounting on D1, D2, and D5 scout rows;
- fill teacher runtime table with measured throughput, peak VRAM, runtime mode,
  and cost;
- materialize first `Ksig` signature tensors and pairwise surface-correlation
  reports;
- finish share-alike/sharing review for held sources.

## 2026-06-20 Shape Config And Smoke Accounting Pass

Added nominal B3/P4/P8 shape configs and a stdlib-only shape accounting tool.

New files:

- `artifacts/shape_manifest/configs/B3_byte_global_same_budget.json`;
- `artifacts/shape_manifest/configs/P4_patch_global_fixed_4_byte.json`;
- `artifacts/shape_manifest/configs/P8_patch_global_fixed_8_byte.json`;
- `artifacts/shape_manifest/sequence_profiles/smoke_byte_lengths.json`;
- `tools/shape_accounting.py`;
- `artifacts/shape_manifest/parameter_report.json`;
- `artifacts/shape_manifest/sequence_report.json`;
- `artifacts/shape_manifest/flop_report.json`.

Current smoke results:

```text
shape_config_status: created_nominal_no_training
parameter_report_status: nominal_config_count_not_G0_evidence
sequence_report_status: synthetic_smoke_profile_not_G0_evidence
flop_report_status: rough_formula_smoke_not_G0_evidence
teacher_packets_enabled: false
```

Open holes that remain:

- replace synthetic smoke byte lengths with real D1, D2, and D5 byte-length
  profiles before making any G0 claim;
- create D0 Common Pile row-filter manifest and license histogram before any
  D0 download;
- run real G0 shape accounting and attach the reports to the G0 rows;
- fill teacher runtime table with measured throughput, peak VRAM, runtime mode,
  and cost;
- materialize first `Ksig` signature tensors and pairwise surface-correlation
  reports;
- finish share-alike/sharing review for held sources.

## 2026-06-20 Common Pile Row Filter Draft Pass

Added `artifacts/dataset_license/common_pile_row_filter_manifest.json`.

The draft records:

- candidate D0 shards: `common-pile/arxiv_abstracts`,
  `common-pile/caselaw_access_project`, and
  `common-pile/biodiversity_heritage_library`;
- held shard: `common-pile/arxiv_papers`;
- allowed and rejected license values;
- per-shard revisions;
- sample caps;
- source and license metadata fields that must be preserved.

Current status:

```text
D0_common_pile_row_filter_manifest: draft_created_no_rows_counted
D0_common_pile_training_download_allowed: false
D0_common_pile_license_histogram: not_created
D0_common_pile_sample_manifest: not_created
```

Open holes that remain:

- run a no-training license histogram pass before any D0 download or use;
- create a D0 Common Pile sample manifest with row counts before and after the
  filter;
- replace synthetic shape smoke byte lengths with real D1, D2, and D5
  byte-length profiles before making any G0 claim;
- run real G0 shape accounting and attach the reports to the G0 rows;
- fill teacher runtime table with measured throughput, peak VRAM, runtime mode,
  and cost;
- materialize first `Ksig` signature tensors and pairwise surface-correlation
  reports;
- finish share-alike/sharing review for held sources.

## 2026-06-20 Common Pile License Filter Tool Pass

Added an executable local filter for the D0 Common Pile row-filter draft:

- `tools/common_pile_license_filter.py`;
- fixture input rows under `artifacts/dataset_license/fixtures/`;
- `artifacts/dataset_license/common_pile_fixture_license_histogram.json`;
- `artifacts/dataset_license/common_pile_fixture_sample_manifest.json`.

The fixture proves the filter keeps only allowed license rows and rejects
missing, unknown, noncommercial, and no-derivatives rows. It is not Common Pile
evidence and does not permit download or training.

Current status:

```text
D0_common_pile_filter_tool: fixture_passed
D0_common_pile_fixture_histogram: created_not_common_pile_evidence
D0_common_pile_fixture_sample_manifest: created_not_common_pile_evidence
D0_common_pile_training_download_allowed: false
```

Open holes that remain:

- run the filter on prepared real Common Pile files to create
  `artifacts/dataset_license/common_pile_license_histogram.json`;
- create `artifacts/dataset_license/common_pile_sample_manifest.json` with real
  row counts before and after filtering;
- replace synthetic shape smoke byte lengths with real D1, D2, and D5
  byte-length profiles before making any G0 claim;
- run real G0 shape accounting and attach the reports to the G0 rows;
- fill teacher runtime table with measured throughput, peak VRAM, runtime mode,
  and cost;
- materialize first `Ksig` signature tensors and pairwise surface-correlation
  reports;
- finish share-alike/sharing review for held sources.

## 2026-06-20 Ksig Fixture Materialization Pass

Added fixture-only teacher signature tooling:

- `tools/ksig_surface.py`;
- `artifacts/teacher_signatures/fixtures/ksig_fixture_scores.json`;
- `artifacts/teacher_signatures/ksig_fixture_signature_records.json`;
- `artifacts/teacher_signatures/ksig_fixture_surface_correlation.json`;
- `artifacts/teacher_signatures/README.md`.

The fixture materializes 27 Ksig records with the 9-field schema and computes
pairwise fixture surface correlations. It is not G1 evidence.

Current status:

```text
Ksig_fixture_materialization: created
Ksig_schema_length: 9
Ksig_fixture_records: 27
surface_correlation_fixture: created_not_G1_evidence
real_teacher_signature_tensor: not_created
real_surface_correlation_report: not_created
```

Open holes that remain:

- run real teacher signature generation after teacher runtime and prompt
  templates are set;
- collect at least 500 pairwise-complete rows before any real G1 distinct
  surface claim;
- add bootstrap confidence intervals for real G1 reports;
- run the filter on prepared real Common Pile files to create
  `artifacts/dataset_license/common_pile_license_histogram.json`;
- create `artifacts/dataset_license/common_pile_sample_manifest.json` with real
  row counts before and after filtering;
- replace synthetic shape smoke byte lengths with real D1, D2, and D5
  byte-length profiles before making any G0 claim;
- run real G0 shape accounting and attach the reports to the G0 rows;
- fill teacher runtime table with measured throughput, peak VRAM, runtime mode,
  and cost;
- get explicit distribution-policy approval before any training use of held
  share-alike/sharing sources.

## 2026-06-20 Share-Alike And Sharing Review Pass

Added `artifacts/dataset_license/sharealike_review.md`.

The review covers:

- `allenai/ai2_arc`;
- `roneneldan/TinyStories`;
- `databricks/databricks-dolly-15k`.

Current status:

```text
sharealike_review_artifact: created_repo_policy_hold_not_legal_clearance
training_download_allowed: false
admitted_training_sources_added: 0
D2_sharealike_sources: held_out_of_training
D5_sharealike_or_sharing_sources: held_out_of_training
```

Open holes that remain:

- get qualified license review or project-owner distribution-policy approval
  before any training use of ARC, TinyStories, or Dolly;
- create attribution manifests before any eval/reference use of held sources;
- define whether any adapted rows, model checkpoints, or model outputs would be
  redistributed before admitting share-alike/sharing sources;
- run the filter on prepared real Common Pile files to create
  `artifacts/dataset_license/common_pile_license_histogram.json`;
- create `artifacts/dataset_license/common_pile_sample_manifest.json` with real
  row counts before and after filtering;
- replace synthetic shape smoke byte lengths with real D1, D2, and D5
  byte-length profiles before making any G0 claim;
- run real G0 shape accounting and attach the reports to the G0 rows;
- fill teacher runtime table with measured throughput, peak VRAM, runtime mode,
  and cost;
- replace fixture `Ksig` signatures with real teacher signature tensors and
  surface-correlation reports.

## 2026-06-20 Teacher Runtime Report Scaffold Pass

Added no-download teacher runtime reporting scaffolding:

- `tools/teacher_runtime_report.py`;
- `artifacts/teacher_runtime/measurement_manifest.json`;
- `artifacts/teacher_runtime/fixtures/runtime_observations_fixture.json`;
- `artifacts/teacher_runtime/runtime_fixture_report.json`;
- `artifacts/teacher_runtime/README.md`.

The tool validates explicit runtime observation records and computes
examples-per-second plus cost-per-1000 signature examples. It does not download
models, run inference, or admit any real teacher. By default it refuses
non-fixture teacher IDs unless `--allow-real-teachers` is passed.

Current status:

```text
teacher_runtime_report_tool: fixture_passed
teacher_runtime_fixture_rows: 2
real_teacher_runtime_rows: not_created
model_download_allowed: false
training_download_allowed: false
missing_real_candidates: BAAI/bge-small-en-v1.5, control-decoder-0.6B, embedding-candidate-0.6B, anchor-decoder-1.7B, BAAI/bge-m3, HuggingFaceTB/SmolLM3-3B-Base_if_distinct_surface_still_needed
```

Open holes that remain:

- write real observation rows only after an explicit local measurement pass;
- measure exact downloaded files, disk bytes, runtime mode, quantization,
  max batch size, examples per second, peak VRAM, and cost per 1000 signature
  examples for each admitted teacher candidate;
- keep real teacher admission blocked until runtime rows, prompt templates,
  tokenizer behavior records, real Ksig tensors, and surface-correlation reports
  all exist;
- prepare real local D1, D2, and D5 rows after source admission and split policy
  are final;
- run `tools/byte_length_profile.py` against the real prepared rows;
- run `tools/shape_accounting.py` against the real extracted profile;
- attach real G0 shape reports to the G0 gate rows.

## 2026-06-20 Prompt Template Render Pass

Added deterministic prompt-template rendering for teacher-signature surfaces:

- `tools/prompt_template_render.py`;
- `artifacts/teacher_signatures/prompt_template_manifest.json`;
- `artifacts/teacher_signatures/fixtures/prompt_cases_fixture.json`;
- `artifacts/teacher_signatures/prompt_render_fixture_report.json`.

The renderer validates that each fixture case supplies every template
placeholder and records prompt SHA-256 hashes, byte lengths, and line counts.
It does not call tokenizers, download models, or score any candidate.

Current status:

```text
prompt_template_manifest: draft_created
prompt_render_fixture_report: created_not_teacher_evidence
rendered_prompt_cases: 4
model_download_allowed: false
training_download_allowed: false
real_teacher_prompt_template_binding: not_created
tokenizer_behavior_record: not_created
```

Open holes that remain:

- bind prompt templates to real teacher candidates after runtime mode and
  tokenizer behavior are known;
- write tokenizer behavior records for each admitted teacher candidate;
- generate real teacher scores with these frozen prompt/template hashes;
- replace fixture `Ksig` signatures with real teacher signature tensors and
  surface-correlation reports;
- keep G1 blocked until runtime rows, prompt bindings, tokenizer records,
  Ksig tensors, and baseline gap checks all exist.

## 2026-06-20 Tokenizer Behavior Scaffold Pass

Added tokenizer behavior record scaffolding:

- `tools/tokenizer_behavior_report.py`;
- `artifacts/tokenizer_behavior/record_manifest.json`;
- `artifacts/tokenizer_behavior/fixtures/tokenizer_cases_fixture.json`;
- `artifacts/tokenizer_behavior/tokenizer_fixture_report.json`;
- `artifacts/tokenizer_behavior/README.md`.

The fixture report validates a UTF-8 byte-identity tokenizer only. It records
roundtrip status, token previews, text hashes, and required fields for future
real tokenizer records. It does not download or inspect primary family, BGE, SmolLM, or
any other real tokenizer.

Current status:

```text
tokenizer_behavior_record_schema: created
tokenizer_fixture_report: created_not_real_tokenizer_evidence
real_tokenizer_behavior_records: 0
model_download_allowed: false
training_download_allowed: false
```

Open holes that remain:

- load exact tokenizer artifacts for each admitted teacher candidate only after
  a real measurement pass is approved;
- record normalization, special-token, BOS/EOS, byte-fallback, context-length,
  prompt-hash, roundtrip, and truncation behavior for each real tokenizer;
- bind real tokenizer records to the prompt-template hashes and runtime rows;
- keep teacher admission blocked until real tokenizer records and real Ksig
  signatures exist.

## 2026-06-20 Fixture Preflight Audit Pass

Added a conservative repo-level fixture preflight audit:

- `tools/se1_preflight_audit.py`;
- `artifacts/preflight/se1_fixture_preflight_report.json`;
- `artifacts/preflight/README.md`.

The audit checks that required fixture reports exist, all artifact JSON parses,
no artifact enables model or training download, and known blocked real evidence
files are still absent. It passes while explicitly keeping harness and training
blocked.

Current status:

```text
fixture_preflight_audit: passed
json_files_checked: 27
harness_allowed: false
training_allowed: false
remaining_real_blockers: real_common_pile_license_histogram_and_sample_manifest, real_D1_D2_D5_byte_length_profiles, real_G0_shape_accounting_report, real_teacher_runtime_rows, real_teacher_prompt_template_bindings, real_tokenizer_behavior_records, real_teacher_signature_tensors, real_surface_correlation_report, baseline_gap_checks
```

Open holes that remain:

- replace fixture reports with real source, shape, runtime, tokenizer, and
  teacher-signature evidence before any harness claim;
- keep the audit in the default verification path for future fixture changes;
- write actual gate reports only after real G0/G1 evidence exists.

## 2026-06-20 Gate And Packet Template Artifact Pass

Materialized the contract's gate-report and packet-ledger templates as artifact
files:

- `tools/gate_packet_template_check.py`;
- `artifacts/gates/gate_report_template.json`;
- `artifacts/gates/initial_gate_rows.json`;
- `artifacts/gates/packet_ledger_template.json`;
- `artifacts/gates/required_packet_rows.json`;
- `artifacts/gates/gate_packet_template_check.json`;
- `artifacts/gates/README.md`.

The checker verifies that all six gates have initial rows, every initial gate
row remains `not_run`, no fixture row points to evidence, and required packet
IDs are listed but unfilled.

Current status:

```text
gate_packet_template_check: passed
gate_rows_checked: 6
required_packet_rows_listed: 6
harness_allowed: false
training_allowed: false
packet_ledger_rows: listed_not_filled
```

Open holes that remain:

- fill packet ledger rows only after real sources, runtime records, tokenizer
  behavior records, and teacher signatures exist;
- attach real evidence artifacts to G0/G1 gate rows before any harness claim;
- keep G2 through G5 template-only until the first harness scope expands beyond
  G0/G1.

## 2026-06-20 D1 Generator Manifest Pass

Materialized the D1 byte-exactness generator plan without generating rows:

- `tools/d1_generator_manifest_check.py`;
- `artifacts/d1_byte_exactness/generator_manifest.json`;
- `artifacts/d1_byte_exactness/generator_manifest_check.json`;
- `artifacts/d1_byte_exactness/README.md`.

The manifest fixes the 2,500-row D1 plan across the five required families and
records the 1,500/500/500 train/validation/test split. The checker fails if
`artifacts/d1_byte_exactness/rows.jsonl` or `eval_report.json` appears before the
contract permits it.

Current status:

```text
D1_generator_manifest: created_manifest_only_rows_not_generated
D1_planned_rows: 2500
D1_family_rows: 500_each
D1_split_counts: train_1500_validation_500_test_500
D1_rows_jsonl: not_created_forbidden_until_harness_phase_approved
D1_eval_report: not_created
harness_allowed: false
training_allowed: false
```

Open holes that remain:

- keep D1 row generation blocked until the harness phase is explicitly approved;
- generate real D1 rows only after that approval and with the manifest seed;
- extract a real D1 byte-length profile from generated rows;
- run D1 byte-exactness eval only after a student baseline exists.

## 2026-06-20 Scout Slice Row Preparation Manifest Pass

Materialized the planned row-preparation manifest without downloading datasets
or creating prepared rows:

- `tools/row_preparation_manifest_check.py`;
- `artifacts/scout_slices/row_preparation_manifest.json`;
- `artifacts/scout_slices/row_preparation_manifest_check.json`;
- `artifacts/scout_slices/README.md`.

The manifest records planned row counts for D1, D2 GSM8K, and D4 JSONSchemaBench,
and keeps D5 TinyStories/Dolly held at zero rows pending distribution policy.
The checker fails if blocked prepared-row JSONL files appear before admission and
staging are approved.

Current status:

```text
row_preparation_manifest: created_no_prepared_rows
D1_local_byte_exactness_planned_rows: 2500
D2_arithmetic_candidate_reasoning_planned_rows: 8792
D4_json_schema_verifier_planned_rows: 1938
D5_generation_guardrail_planned_rows: 0_until_distribution_policy
dataset_download_allowed: false
training_allowed: false
```

Open holes that remain:

- keep external prepared rows blocked until source admission and staging are
  explicitly approved;
- prepare D2/D4 local JSONL rows only after that approval;
- keep D5 TinyStories/Dolly out of training until a distribution policy exists;
- extract real D2 and D5 byte-length profiles only from approved prepared rows.

## 2026-06-20 Baseline Gap Template Pass

Materialized the baseline-gap and retained-gain control template without running
baselines:

- `tools/baseline_gap_template_check.py`;
- `artifacts/baseline_gap/baseline_gap_template.json`;
- `artifacts/baseline_gap/baseline_gap_template_check.json`;
- `artifacts/baseline_gap/README.md`.

The template precommits controls for non-decoder student gap, packet
control-adjusted gain, teacher-free retained gain, teacher dependence, and
compute efficiency. Every row remains `not_run`, and the checker fails if
baseline result artifacts appear prematurely.

Current status:

```text
baseline_gap_template: created_no_results
baseline_gap_checks_count: 5
baseline_gap_report: not_created
retained_gain_report: not_created
harness_allowed: false
training_allowed: false
```

Open holes that remain:

- run an actual student baseline before measuring teacher gap;
- measure teacher-exposed gap only after real teacher signatures exist;
- measure retained gain only after packet training is approved and completed;
- compare against no-teacher, best-single-teacher, naive-average, answer-only,
  and compute baselines before any G2/G3 claim.

## 2026-06-20 Teacher Binding Manifest Pass

Materialized candidate teacher-to-template bindings without accepting any real
teacher:

- `tools/teacher_binding_manifest_check.py`;
- `artifacts/teacher_bindings/candidate_binding_manifest.json`;
- `artifacts/teacher_bindings/candidate_binding_manifest_check.json`;
- `artifacts/teacher_bindings/README.md`.

The manifest maps anchor/control decoder candidates, the embedding candidate, and the
GSM8K exact oracle to the prompt templates they would use. Every binding remains
blocked until real runtime records, tokenizer behavior records, and signature
outputs exist.

Current status:

```text
candidate_teacher_bindings: 4
accepted_teacher_bindings: 0
accepted_teacher_bindings_file: not_created
model_download_allowed: false
training_allowed: false
```

Open holes that remain:

- create real runtime rows for candidate teachers;
- create real tokenizer behavior records for candidate teachers;
- generate real teacher signature outputs;
- accept teacher bindings only after those dependencies exist.

## 2026-06-20 Exact Oracle Fixture Pass

Added fixture-only exact oracle checks:

- `tools/exact_oracle_fixture.py`;
- `artifacts/exact_oracles/oracle_manifest.json`;
- `artifacts/exact_oracles/fixtures/oracle_cases_fixture.json`;
- `artifacts/exact_oracles/oracle_fixture_report.json`;
- `artifacts/exact_oracles/README.md`.

The fixture covers GSM8K-style numeric answer equivalence and a small
JSON-schema subset. It validates oracle mechanics only and does not read
benchmark rows or create a real oracle eval report.

Current status:

```text
exact_oracle_fixture_checks: passed
oracle_fixture_cases: 4
oracle_eval_report: not_created
dataset_download_allowed: false
harness_allowed: false
training_allowed: false
```

Open holes that remain:

- prepare real D2/D4 rows only after source admission and staging approval;
- run real oracle eval only after prepared rows and baseline context exist;
- keep exact-oracle surface claims blocked until real teacher signatures and
  baseline gap checks exist.

## 2026-06-20 Byte-Length Profile Extractor Pass

Added a no-download local byte-profile extractor for the G0 shape path:

- `tools/byte_length_profile.py`;
- `artifacts/shape_manifest/byte_profile_manifest.json`;
- hand-written fixture rows under `artifacts/shape_manifest/fixtures/`;
- `artifacts/shape_manifest/sequence_profiles/fixture_extracted_byte_lengths.json`;
- `artifacts/shape_manifest/byte_length_fixture_report.json`;
- `artifacts/shape_manifest/README.md`.

The fixture report proves that local prepared JSONL rows can be converted into a
`tools/shape_accounting.py`-compatible byte-length profile. It is not G0
evidence and does not permit training or dataset download.

Current status:

```text
byte_length_profile_extractor: fixture_passed
fixture_profile_status: handwritten_fixture_not_G0_evidence
fixture_profiles_counted: D1_local_byte_exactness_fixture, D2_candidate_reasoning_fixture, D5_generation_guardrail_fixture
real_D1_D2_D5_byte_length_profiles: not_created
training_download_allowed: false
```

Open holes that remain:

- prepare real local D1, D2, and D5 rows after source admission and split policy
  are final;
- run `tools/byte_length_profile.py` against the real prepared rows;
- run `tools/shape_accounting.py` against the real extracted profile;
- attach real G0 shape reports to the G0 gate rows;
- run the filter on prepared real Common Pile files to create
  `artifacts/dataset_license/common_pile_license_histogram.json`;
- create `artifacts/dataset_license/common_pile_sample_manifest.json` with real
  row counts before and after filtering;
- fill teacher runtime table with measured throughput, peak VRAM, runtime mode,
  and cost;
- replace fixture `Ksig` signatures with real teacher signature tensors and
  surface-correlation reports.

## 2026-06-20 Local Hardware Snapshot Pass

Added a no-download local hardware snapshot for teacher-runtime planning:

- `tools/local_hardware_snapshot.py`;
- `artifacts/hardware/local_hardware_snapshot.json`;
- `artifacts/hardware/README.md`.

The snapshot records stable OS/Python/GPU metadata through `nvidia-smi`. It does
not download models, load tokenizers, allocate GPU memory, run inference, or
start training.

Current status:

```text
local_hardware_snapshot: created
gpu: NVIDIA_GeForce_RTX_5090_Laptop_GPU
vram: 24463_MiB
driver: 595.79
default_teacher_tier: single_24gb_gpu
model_download_allowed: false
training_allowed: false
```

Open holes that remain:

- fill teacher runtime rows with measured throughput, peak VRAM, runtime mode,
  and cost;
- accept real teacher bindings only after runtime, tokenizer, and signature
  evidence exists;
- keep the 4B-class teacher candidates blocked until cost per signature example
  is measured.

## 2026-06-20 Dataset License Artifact README Pass

Added a directory-level guide for dataset-license artifacts:

- `artifacts/dataset_license/README.md`.

The guide separates checked-in fixture reports from the still-forbidden real
Common Pile histogram and sample manifest. It documents the fixture command and
keeps the real `common_pile_license_histogram.json` and
`common_pile_sample_manifest.json` outputs blocked until local prepared shard
files exist and the real row-counting step is explicitly approved.

Current status:

```text
dataset_license_readme: created
fixture_filter_reports: documented
real_common_pile_license_histogram: not_created
real_common_pile_sample_manifest: not_created
training_download_allowed: false
```

Open holes that remain:

- create local prepared Common Pile shard files only after approval;
- run the row filter on real prepared shard files;
- record before/after row counts, license histograms, source identifiers, and
  sample caps before any Common Pile training use.

## 2026-06-20 Artifact Index Pass

Added a top-level artifact index:

- `artifacts/README.md`.

The index gives every artifact subdirectory a status class and README pointer,
records the default verification command, and lists the real evidence files that
must stay absent until admission is explicitly approved and integrated into the
contract.

Current status:

```text
artifact_index: created
artifact_directories_indexed: 13
default_verification: python tools/run_fixture_checks.py
real_evidence_boundary: documented
harness_allowed: false
training_allowed: false
```

Open holes that remain:

- replace fixture/template artifacts only through explicit admission steps;
- keep the preflight audit synchronized with every new real-evidence boundary;
- avoid creating prepared rows, accepted bindings, runtime measurements, or
  signature tensors without contract integration.

## 2026-06-20 Artifact Index Check Pass

Added a local drift check for the artifact index:

- `tools/artifact_index_check.py`;
- `artifacts/preflight/artifact_index_check_report.json`.

The checker verifies that every direct `artifacts/` subdirectory has a
`README.md`, every subdirectory is listed in `artifacts/README.md`, and the
index does not point at missing artifact directories. The default fixture check
now runs it before the preflight audit.

Current status:

```text
artifact_index_check: passed
artifact_directories_checked: 13
harness_allowed: false
model_download_allowed: false
training_allowed: false
training_download_allowed: false
```

Open holes that remain:

- keep every new artifact directory paired with a local README;
- update `artifacts/README.md` before adding a new artifact family;
- keep `tools/se1_preflight_audit.py` aligned with new fixture reports and
  blocked real-evidence paths.

## 2026-06-20 Blocked Flag Audit Tightening

Tightened the recursive preflight flag audit in
`tools/se1_preflight_audit.py`.

The audit now fails if any artifact JSON sets one of these flags to `true`:

- `harness_allowed`;
- `training_allowed`;
- `dataset_download_allowed`;
- `training_download_allowed`;
- `model_download_allowed`.

Current status:

```text
blocked_preflight_flag_audit: tightened
fixture_preflight_audit: passed
harness_allowed: false
training_allowed: false
dataset_download_allowed: false
training_download_allowed: false
model_download_allowed: false
```

Open holes that remain:

- keep future artifact schemas using the same blocked flag names when relevant;
- add any new permission-like boolean to the recursive audit before it appears
  in checked artifacts.

## 2026-06-20 Runtime And Tokenizer Real-Artifact Boundary Pass

Reserved explicit future real-evidence paths for teacher runtime and tokenizer
records, and added them to `tools/se1_preflight_audit.py`:

- `artifacts/teacher_runtime/runtime_observations.json`;
- `artifacts/teacher_runtime/teacher_runtime_report.json`;
- `artifacts/tokenizer_behavior/teacher_tokenizer_behavior_records.json`;
- `artifacts/tokenizer_behavior/tokenizer_behavior_report.json`.

The paths are documented in the artifact index and the local runtime/tokenizer
README files. They must remain absent until real measurements are approved and
integrated into the contract.

Current status:

```text
teacher_runtime_real_artifact_paths: reserved_absent
tokenizer_behavior_real_artifact_paths: reserved_absent
fixture_preflight_audit: passed
model_download_allowed: false
training_allowed: false
```

Open holes that remain:

- run actual teacher runtime measurements only after model access is approved;
- record real tokenizer behavior only after exact tokenizer artifacts are
  loaded under an approved measurement step;
- update accepted teacher bindings only after the real runtime, tokenizer, and
  signature records exist.

## 2026-06-20 Held Source Attribution Distribution Template Pass

Added a checked template for held source attribution and distribution policy:

- `artifacts/dataset_license/source_attribution_distribution_template.json`;
- `tools/source_attribution_distribution_check.py`;
- `artifacts/dataset_license/source_attribution_distribution_template_check.json`.

The template keeps ARC, TinyStories, and Dolly held with zero planned rows, no
attribution manifest, no redistribution policy, and no training admission. The
checker also reserves the future real policy path
`artifacts/dataset_license/source_attribution_distribution_policy.json` and
fails if it appears before admission.

Current status:

```text
source_attribution_distribution_template: created
held_sources_checked: allenai/ai2_arc, databricks/databricks-dolly-15k, roneneldan/TinyStories
held_sources_admitted_for_training: 0
real_policy_path: not_created
dataset_download_allowed: false
training_allowed: false
```

Open holes that remain:

- write a real attribution and distribution policy only after owner approval or
  qualified review;
- keep D5 TinyStories and Dolly rows at zero until that policy exists;
- keep ARC out of training rows until attribution and share-alike handling are
  explicit.

## 2026-06-20 Shape Real-Artifact Boundary Pass

Reserved explicit future real-evidence paths for D1/D2/D5 byte-length profiles
and G0 shape accounting:

- `artifacts/shape_manifest/sequence_profiles/real_D1_D2_D5_byte_lengths.json`;
- `artifacts/shape_manifest/g0_parameter_report.json`;
- `artifacts/shape_manifest/g0_sequence_report.json`;
- `artifacts/shape_manifest/g0_flop_report.json`;
- `artifacts/shape_manifest/g0_shape_accounting_report.json`.

The paths are now listed in `tools/se1_preflight_audit.py`,
`artifacts/README.md`, and `artifacts/shape_manifest/README.md`. They must
remain absent until real prepared rows exist and G0 is explicitly approved.

Current status:

```text
real_D1_D2_D5_byte_length_profile_path: reserved_absent
real_G0_shape_accounting_paths: reserved_absent
fixture_preflight_audit: passed
training_allowed: false
```

Open holes that remain:

- prepare real D1/D2/D5 rows only after source admission and staging approval;
- extract the real byte-length profile from those prepared rows;
- run shape accounting against the real profile and connect the resulting G0
  reports to the gate ledger.

## 2026-06-20 Remaining Blocker Map Pass

Added a checked map for the preflight remaining blockers:

- `artifacts/preflight/remaining_blocker_map.json`;
- `tools/remaining_blocker_map_check.py`;
- `artifacts/preflight/remaining_blocker_map_check_report.json`.

The map is now the source for `remaining_blockers` in
`artifacts/preflight/se1_fixture_preflight_report.json`. Each blocker records
its gate or phase, required absent artifacts, next action, and retirement
condition. The checker fails if a mapped required artifact is not in the
preflight absent list or if any blocked artifact already exists.

Current status:

```text
remaining_blocker_map: created
remaining_blockers_checked: 9
fixture_preflight_audit: passed
harness_allowed: false
training_allowed: false
```

Open holes that remain:

- keep every new preflight blocker represented in the blocker map;
- retire blockers only by replacing mapped absent artifacts with integrated
  real evidence and updating gate rows;
- avoid adding loose blocker strings directly to the preflight report.

## 2026-06-20 Ksig Schema Artifact Pass

Materialized the Ksig schema as a checked artifact and made the fixture
signature generator consume it:

- `artifacts/teacher_signatures/ksig_schema.json`;
- `tools/ksig_surface.py`.

The schema now owns the 9-field Ksig order, exact-oracle error-class encoding,
surface-correlation fields, and the minimum 500 pairwise-complete rows required
for real G1 evidence. Fixture reports record the schema path they used.

Current status:

```text
ksig_schema_artifact: created
ksig_fixture_generation: passed_from_schema
real_teacher_signature_records: not_created
real_surface_correlation_report: not_created
training_allowed: false
```

Open holes that remain:

- generate real teacher signatures only after runtime, tokenizer, and accepted
  binding records exist;
- compute real surface correlations from real signature records;
- attach real G1 evidence to gate rows only after the pairwise-row threshold and
  baseline gap requirements are met.

## 2026-06-20 Prompt Template Schema And Binding Validation Pass

Added schema validation for teacher prompt templates and tightened candidate
teacher binding checks:

- `artifacts/teacher_signatures/prompt_template_schema.json`;
- `tools/prompt_template_render.py`;
- `tools/teacher_binding_manifest_check.py`.

Prompt rendering now validates template required fields, accepted teacher
families, score kinds, and duplicate template IDs against the schema. Candidate
binding checks now fail if a teacher family is bound to a prompt template that
does not accept that family.

Current status:

```text
prompt_template_schema: created
prompt_render_fixture_report: passed_from_schema
candidate_binding_family_compatibility_check: passed
accepted_teacher_bindings: not_created
model_download_allowed: false
training_allowed: false
```

Open holes that remain:

- keep accepted bindings blocked until runtime, tokenizer, and signature records
  exist;
- bind real teachers only to templates whose accepted family and score kind
  match the teacher surface;
- attach accepted bindings to real signature evidence before any G1 claim.

## 2026-06-20 Initial Packet Ledger Rows Pass

Materialized the initial packet ledger rows from the benchmark contract:

- `artifacts/gates/initial_packet_ledger_rows.json`;
- `artifacts/gates/packet_ledger_template.json`;
- `tools/gate_packet_template_check.py`.

The checker now verifies that every required packet ID has a template row with
landing zone, target loss, refused signal, corruption risk, removal test,
demotion rule, source identity, and source revision. Rows remain
`template_row_not_training_evidence` with `realized_value:
unset_until_training`.

Current status:

```text
initial_packet_ledger_rows: created
initial_packet_rows_checked: 6
packet_rows_training_evidence: false
gate_packet_template_check: passed
training_allowed: false
```

Open holes that remain:

- fill packet rows only after real source, signature, and baseline artifacts
  exist;
- attach packet rows to accepted teacher bindings and gate evidence before G2;
- demote any packet whose retained gain fails matched controls.

## 2026-06-20 Teacher Port Contract Artifact Pass

Materialized candidate teacher-port contracts as checked artifacts:

- `artifacts/teacher_ports/teacher_port_contracts.json`;
- `artifacts/teacher_ports/README.md`;
- `tools/teacher_port_contracts_check.py`;
- `artifacts/teacher_ports/teacher_port_contracts_check.json`.

The checker verifies that every candidate teacher binding and every source in
the initial packet ledger has a port contract. It checks teacher family, source
kind, revision, accepted packet family, landing zone, removal test, blocked
training-use boundary, corruption risks, signal type, and compression format.

Current status:

```text
teacher_port_contracts: created
teacher_ports_checked: 7
accepted_teacher_ports: 0
teacher_port_contracts_check: passed
harness_allowed: false
training_allowed: false
```

Open holes that remain:

- accept no teacher port until runtime, tokenizer, signature, and baseline
  evidence exists;
- keep packet rows and teacher bindings aligned with their port contracts;
- retire a port if its removal test or retained-gain evidence fails.

## 2026-06-20 Metric Logger Spec Pass

Materialized the missing implementation metric logger spec:

- `artifacts/metrics/metric_logger_spec.json`;
- `artifacts/metrics/README.md`;
- `tools/metric_logger_spec_check.py`;
- `artifacts/metrics/metric_logger_spec_check.json`.

The spec precommits metric formulas, required slices, comparison baselines,
training stages, minimum log-record fields, and reserved metric output paths.
It creates no metric logs and does not authorize harness or training.

Current status:

```text
metric_logger_spec: created
metric_logger_spec_check: passed
metric_logs_created: false
harness_allowed: false
training_allowed: false
```

Open holes that remain:

- create metric logs only inside an approved run;
- attach metric summary reports to baseline-gap and retained-gain evidence;
- keep every pass/fail gate claim tied to logged metrics and artifact refs.

## 2026-06-20 Gate Metric Binding Pass

Materialized a checked gate-to-metric binding layer:

- `artifacts/gates/gate_metric_bindings.json`;
- `tools/gate_metric_binding_check.py`;
- `artifacts/gates/gate_metric_binding_check.json`.

The binding file forces every promotion gate to name the metric IDs, metric
slices, controls, comparison IDs where applicable, and future artifact refs that
must exist before any pass/fail gate claim can be accepted. It also extends the
metric logger spec with gate-specific metrics such as `patch_ratio`,
`surface_correlation`, `p95_active_FLOPs`, `target_slice_gain`, and
`update_cost_ratio`.

Current status:

```text
gate_metric_bindings: created
gate_metric_binding_check: passed
gate_metric_evidence_report_created: false
harness_allowed: false
training_allowed: false
```

Open holes that remain:

- create `gate_metric_evidence_report.json` only inside an approved run;
- keep G0/G1 gate rows not-run until real shape, runtime, tokenizer, and
  signature evidence exists;
- keep G2 through G5 template-only until packet training and retained-gain
  evidence exist.

## 2026-06-20 Run Admission Manifest Pass

Materialized the checked approval boundary that the repo had been referring to
informally:

- `artifacts/preflight/run_admission_manifest.json`;
- `tools/run_admission_manifest_check.py`;
- `artifacts/preflight/run_admission_manifest_check_report.json`.

The manifest keeps every admission step false: source admission, row
preparation, G0 shape accounting, G1 teacher measurement, teacher binding,
teacher signature generation, exact-oracle eval, first harness scope, and gate
claim acceptance. The checker imports the preflight absent-real-artifact list
and fails if any forbidden output appears before its admission step changes. It
also requires every preflight absent-real-artifact path to be covered by exactly
the admission manifest's forbidden path set, so future blocked outputs cannot
float outside the approval boundary.

Current status:

```text
run_admission_manifest: created
run_admission_manifest_check: passed
dataset_download_allowed: false
model_download_allowed: false
harness_allowed: false
training_allowed: false
```

Open holes that remain:

- admit no run step without changing the manifest and preflight contract in the
  same commit;
- keep approval artifacts unset until the repo has real source, runtime,
  tokenizer, signature, metric, and gate evidence paths ready;
- keep future run outputs in the preflight absent-real-artifact list before
  any approved run creates them.

## 2026-06-20 Readiness Record Pass

Materialized the doctrine's six-field readiness record as checked artifacts:

- `artifacts/readiness/readiness_record.json`;
- `artifacts/readiness/README.md`;
- `tools/readiness_record_check.py`;
- `artifacts/readiness/readiness_record_check.json`.

The readiness record consolidates `dataset_manifest`, `teacher_roster`,
`substrate_scout`, `packet_plan`, `retained_gain_plan`, and `promotion_gates`
into one canonical artifact. Every field points to existing evidence refs,
names blockers from either the remaining-blocker map or the run-admission
manifest, and keeps the verdict `not_ready_for_harness`.

Current status:

```text
readiness_record: created
readiness_record_check: passed
verdict: not_ready_for_harness
harness_allowed: false
training_allowed: false
```

Open holes that remain:

- fill the missing real-evidence paths named by the readiness record only after
  their admission steps are approved;
- keep the readiness verdict blocked until G0 and G1 evidence rows are real;
- do not treat design-smoke artifacts as harness readiness evidence.

## 2026-06-20 Source Ecology Record Pass

Materialized the candidate source-ecology ledger:

- `artifacts/source_ecology/source_ecology_record.json`;
- `artifacts/source_ecology/README.md`;
- `tools/source_ecology_check.py`;
- `artifacts/source_ecology/source_ecology_check.json`.

The record covers every candidate teacher port and forces each source to name
its independence assumption, correlation risk, disagreement role,
aggregation-forbidden condition, and source-specific credit rule. This keeps
multi-teacher evidence from collapsing into averaging, routing, or correlated
same-family echoes.

Current status:

```text
source_ecology_record: created
source_ecology_check: passed
records_checked: 7
harness_allowed: false
training_allowed: false
```

Open holes that remain:

- replace candidate source-ecology rows with accepted rows only after real
  teacher runtime, tokenizer, and surface-correlation evidence exists;
- keep same-family teacher disagreement treated as a control, not independent
  evidence, until the distinct-surface gate passes;
- credit teacher gains only through source-specific retained-gain evidence.

## 2026-06-20 Teacher Selection Gate Pass

Materialized the teacher-selection gate layer:

- `artifacts/teacher_selection/teacher_selection_gates.json`;
- `artifacts/teacher_selection/README.md`;
- `tools/teacher_selection_check.py`;
- `artifacts/teacher_selection/teacher_selection_check.json`.

The selection record covers every candidate teacher port and keeps every
candidate blocked behind the five Eklavya admission gates: distinct surface,
student gap, landing zone, cost, and removal. The checker cross-checks teacher
ports and source-ecology rows so no teacher can be counted as admitted by
preference.

Current status:

```text
teacher_selection_gates: created
teacher_selection_check: passed
candidate_count: 7
accepted_teacher_count: 0
harness_allowed: false
training_allowed: false
```

Open holes that remain:

- admit no teacher until all five selection gates have real evidence;
- keep same-family decoder comparison as a scale control until distinct-surface
  evidence exists;
- attach any accepted teacher decision to runtime, tokenizer, source-ecology,
  surface-correlation, and retained-gain evidence.

## 2026-06-20 Claim Boundary Pass

Materialized the claim-boundary ledger:

- `artifacts/claims/claim_boundary.json`;
- `artifacts/claims/README.md`;
- `tools/claim_boundary_check.py`;
- `artifacts/claims/claim_boundary_check.json`.

The ledger explicitly blocks claims that Eklavya is accomplished, Sutra is
accomplished, the first harness is ready or passed, a teacher is admitted,
retained gain has been measured, SE1 is efficient/world-class, or a training
dataset has been admitted. Every blocked claim names required evidence refs and
blockers from the run-admission manifest or remaining-blocker map.

Current status:

```text
claim_boundary: created
claim_boundary_check: passed
claims_checked: 7
harness_allowed: false
training_allowed: false
```

Open holes that remain:

- admit no high-level claim until its real evidence refs exist through approved
  run-admission steps;
- keep public-facing language tied to claim-boundary status;
- update the claim boundary in the same commit as any future evidence promotion.

## 2026-06-20 Public Claim Posture Pass

Materialized a public claim posture record that binds outward-facing repository
language to the checked claim boundary:

- `artifacts/claims/public_claim_posture.json`;
- `tools/public_claim_posture_check.py`;
- `artifacts/claims/public_claim_posture_check.json`.

The posture checker requires `README.md`, the benchmark contract, the artifact
index, the claims README, and this reset ledger to keep SE1 in the
pre-harness, no-public-claims state while every high-level claim remains blocked
by the claim-boundary ledger.

Current status:

```text
public_claim_posture: created
public_claim_posture_check: passed
docs_checked: 5
harness_allowed: false
training_allowed: false
```

Open holes that remain:

- update the public posture record in the same commit as any future claim
  promotion;
- keep accomplishment, harness-pass, efficiency, retained-gain, and admission
  claims out of public docs until the claim boundary admits them.

## 2026-06-20 Canonical Contract Status Drift Pass

Updated `research/EKLAVYA_BENCHMARK_CONTRACT.md` so the pre-harness readiness
section no longer says the packet ledger schema or implementation metric logger
spec are missing after those checked artifacts were created.

Added:

- `tools/canonical_contract_status_check.py`;
- `artifacts/readiness/canonical_contract_status_check.json`.

The checker fails if stale status phrases return, and it requires the benchmark
contract to mention the current metric logger, gate metric binding,
source-ecology, teacher-selection, claim-boundary, run-admission, and initial
packet-ledger artifacts.

Current status:

```text
canonical_contract_status_check: passed
stale_packet_ledger_schema_missing_phrase: absent
stale_metric_logger_missing_phrase: absent
harness_allowed: false
training_allowed: false
```

Open holes that remain:

- keep canonical prose synchronized with checked artifacts whenever an artifact
  is added or promoted;
- keep stale “missing” language only for real evidence that is genuinely still
  absent;
- do not turn contract-status freshness into readiness for harness execution.

## 2026-06-20 Remaining Blocker Coverage Pass

Tightened the remaining-blocker map so every required absent-real artifact from
the preflight list has a named blocker row, not only an admission step.

Updated:

- `artifacts/preflight/remaining_blocker_map.json`;
- `tools/remaining_blocker_map_check.py`;
- `artifacts/preflight/remaining_blocker_map_check_report.json`.

The checker now fails if a preflight absent artifact is missing from the
blocker map. The map added explicit blockers for the source-attribution policy,
D1 byte-exactness eval report, D4 JSON-schema prepared rows, exact-oracle eval
report, and gate metric evidence report.

Current status:

```text
remaining_blocker_map_check: passed
blockers_checked: 14
harness_allowed: false
training_allowed: false
```

Open holes that remain:

- keep blocker rows synchronized with every new absent-real artifact;
- retire a blocker only in the same commit that admits and verifies its real
  evidence artifact.

## 2026-06-20 Accomplishment Contract Pass

Materialized an accomplishment exit contract for the two high-level claims the
repo must eventually earn:

- `artifacts/accomplishment/accomplishment_contract.json`;
- `artifacts/accomplishment/README.md`;
- `tools/accomplishment_contract_check.py`;
- `artifacts/accomplishment/accomplishment_contract_check.json`.

The contract decomposes `C0_eklavya_accomplished` and
`C1_sutra_accomplished` into concrete evidence requirements tied to the claim
boundary, readiness record, run-admission manifest, remaining-blocker map, and
G0-G5 gate metric bindings. It deliberately keeps both claims blocked until
real evidence exists.

Current status:

```text
accomplishment_contract: created
accomplishment_contract_check: passed
claims_checked: 2
harness_allowed: false
training_allowed: false
```

Open holes that remain:

- fill the real evidence artifacts named by the contract through approved
  run-admission steps;
- do not treat Eklavya or Sutra as accomplished until the contract, claim
  boundary, and gate evidence are promoted together.

## 2026-06-20 Accomplishment Progress Report Pass

Added a generated progress report for the accomplishment contract:

- `tools/accomplishment_progress_report.py`;
- `artifacts/accomplishment/accomplishment_progress_report.json`.

The report expands every Eklavya and Sutra accomplishment requirement into
current design refs, missing real evidence refs, admission steps, remaining
blockers, and blocker next actions. It keeps the verdict `not_accomplished`
while the required evidence artifacts are absent.

Current status:

```text
accomplishment_progress_report: passed_not_accomplished
claims_checked: 2
requirements_checked: 10
harness_allowed: false
training_allowed: false
```

Open holes that remain:

- use the progress report as the current checklist for real evidence work;
- regenerate it after every admission, blocker, gate, or accomplishment
  contract change.

## 2026-06-20 Accomplishment Work Order Pass

Added a generated work order that converts the accomplishment progress report
into ordered run-admission packages:

- `tools/accomplishment_work_order.py`;
- `artifacts/accomplishment/accomplishment_work_order.json`.

The work order keeps all packages `not_admitted`, but it now records the
frontier package, missing outputs, admission dependencies, blocker dependencies,
external conditions, and blocker next actions for every missing accomplishment
evidence path.

Current status:

```text
accomplishment_work_order: passed_not_admitted
work_packages: 9
frontier_package: D0_source_admission
harness_allowed: false
training_allowed: false
```

Open holes that remain:

- do not execute a work package until its admission step has an approval
  artifact and the benchmark contract is updated in the same commit;
- regenerate the work order after every progress-report or run-admission
  change.

## 2026-06-20 D0 Source Admission Packet Pass

Materialized the frontier D0 approval packet without admitting source data:

- `artifacts/dataset_license/d0_source_admission_packet.json`;
- `tools/d0_source_admission_packet_check.py`;
- `artifacts/dataset_license/d0_source_admission_packet_check.json`.

The packet records the Common Pile row-filter command shape, required blocked
outputs, fixture evidence, source-attribution policy dependency, candidate
shards, and approval conditions. The checker keeps `D0_source_admission`
unapproved and fails if any real D0 output appears early.

Current status:

```text
d0_source_admission_packet: not_approved
d0_source_admission_packet_check: passed
frontier_package: D0_source_admission
harness_allowed: false
training_allowed: false
```

Open holes that remain:

- obtain explicit approval before running the real Common Pile row filter;
- provide a project-owner-supplied local input map only after approval;
- keep real D0 outputs absent until the approval artifact and benchmark
  contract are updated together.

## 2026-06-20 D0 Input Map Contract Pass

Added a checked D0 input-map contract:

- `artifacts/dataset_license/d0_input_map_contract.json`;
- `tools/d0_input_map_contract_check.py`;
- `artifacts/dataset_license/d0_input_map_contract_check.json`.

The checker validates that the fixture input map exactly covers the current
Common Pile candidate shards, excludes held shards, uses checked local JSONL
fixtures, and keeps the future real local input map absent until D0 approval.

Current status:

```text
d0_input_map_contract: fixture_only_no_real_inputs
d0_input_map_contract_check: passed
frontier_package: D0_source_admission
harness_allowed: false
training_allowed: false
```

Open holes that remain:

- do not create the real local Common Pile input map until D0 is approved;
- keep fixture input-map validation separate from real source admission;
- require complete candidate-shard coverage before any real D0 row filter run.

## 2026-06-20 Work Order Blocker Dependency Pass

Tightened the accomplishment work-order generator so package-level blocker
dependencies include every blocker inferred from the package's forbidden output
paths, not only blockers named directly in the run-admission manifest.

Updated:

- `tools/accomplishment_work_order.py`;
- `artifacts/accomplishment/accomplishment_work_order.json`;
- `artifacts/accomplishment/README.md`.

This closes a D0 drift hole where
`source_attribution_distribution_policy` appeared as an output blocker for
`artifacts/dataset_license/source_attribution_distribution_policy.json` but was
not listed in the D0 package's `blocker_dependencies` or next-action summary.

Current status:

```text
accomplishment_work_order: passed_not_admitted
frontier_package: D0_source_admission
D0_blocker_dependencies: real_common_pile_license_histogram_and_sample_manifest, source_attribution_distribution_policy
harness_allowed: false
training_allowed: false
```

Open holes that remain:

- do not promote a work package until its admission step is explicitly approved;
- keep output-level blocker rows synchronized with package-level dependency
  summaries;
- keep all real evidence outputs absent until their admission packages are
  approved and regenerated through the work order.

## 2026-06-20 D0 Packet Work-Order Blocker Check Pass

Extended the D0 source-admission packet checker so it verifies the
`D0_source_admission` work-order package against the packet's blocked outputs.

Updated:

- `tools/d0_source_admission_packet_check.py`;
- `artifacts/dataset_license/d0_source_admission_packet_check.json`;
- `artifacts/dataset_license/README.md`.

The checker now records the D0 work-order output blockers and package-level
blocker dependencies, then fails if any packet blocked output lacks a
work-order output blocker or if any output blocker is missing from
`blocker_dependencies`.

Current status:

```text
d0_source_admission_packet_check: passed_not_approved
work_order_blocker_dependencies_checked: real_common_pile_license_histogram_and_sample_manifest, source_attribution_distribution_policy
frontier_package: D0_source_admission
harness_allowed: false
training_allowed: false
```

Open holes that remain:

- keep D0 unapproved until project-owner approval and local input-map evidence
  exist;
- preserve the source-attribution policy blocker alongside the Common Pile
  histogram/sample blocker;
- keep real D0 outputs absent until the packet, work order, and benchmark
  contract are promoted together.

## 2026-06-20 Accomplishment Evidence Blocker Coverage Pass

Tightened the accomplishment progress-report generator so every required real
evidence ref must have both a run-admission step and a blocker row.

Updated:

- `tools/accomplishment_progress_report.py`;
- `artifacts/accomplishment/accomplishment_progress_report.json`;
- `artifacts/accomplishment/README.md`.

The progress report now records `summary.evidence_blockers`, which is the set
of blocker-map rows attached to real evidence refs. It fails if any required
real evidence ref has no blocker coverage, preventing accomplishment
requirements from depending on unowned missing evidence.

Current status:

```text
accomplishment_progress_report: passed_not_accomplished
evidence_blockers_checked: 14
harness_allowed: false
training_allowed: false
```

Open holes that remain:

- keep requirement-level blockers and evidence-level blockers synchronized;
- do not remove a blocker row until the corresponding real evidence exists and
  the accomplishment claim boundary is updated;
- keep Eklavya and Sutra accomplishment claims blocked until all required real
  evidence refs are present and promoted together.

## 2026-06-20 Claim Boundary Evidence Blocker Coverage Pass

Tightened the public claim boundary so each absent real evidence ref must be
covered by a claim-level blocker or admission step.

Updated:

- `tools/claim_boundary_check.py`;
- `artifacts/claims/claim_boundary.json`;
- `artifacts/claims/claim_boundary_check.json`;
- `artifacts/claims/README.md`.

The stricter checker found that `C3_teacher_admitted` required
`artifacts/teacher_runtime/teacher_runtime_report.json` but did not directly
name the runtime blocker. The claim now lists
`G1_teacher_runtime_measurement` and `real_teacher_runtime_rows`.

The dataset-admission claim also now requires
`artifacts/dataset_license/source_attribution_distribution_policy.json` and
lists `source_attribution_distribution_policy` as a blocker, keeping the public
claim boundary aligned with the D0 packet's full blocked-output set.

Current status:

```text
claim_boundary_check: passed_no_claims_admitted
claims_checked: 7
evidence_blockers_checked: 17
harness_allowed: false
training_allowed: false
```

Open holes that remain:

- keep each claim's `blocked_by` list synchronized with its required absent
  evidence refs;
- keep dataset-admission claims blocked on source-attribution policy, not only
  Common Pile histogram/sample evidence;
- do not promote any public claim until its blockers are retired by approved
  real evidence artifacts.

## 2026-06-20 D0 Run Admission Blocker Coverage Pass

Tightened the run-admission manifest around the current frontier package.

Updated:

- `artifacts/preflight/run_admission_manifest.json`;
- `tools/run_admission_manifest_check.py`;
- `artifacts/preflight/run_admission_manifest_check_report.json`;
- `artifacts/preflight/README.md`.

`D0_source_admission` now lists both blockers for its forbidden outputs:
`real_common_pile_license_histogram_and_sample_manifest` and
`source_attribution_distribution_policy`. The checker now reads
`artifacts/preflight/remaining_blocker_map.json`, records
`output_blockers_checked`, and fails if a D0 forbidden output's blocker-map row
is missing from the D0 step's `blocked_by` list.

Current status:

```text
run_admission_manifest_check: passed_no_run_admitted
D0_blocked_by: real_common_pile_license_histogram_and_sample_manifest, source_attribution_distribution_policy
output_blockers_checked: 14
harness_allowed: false
training_allowed: false
```

Open holes that remain:

- keep D0 admission blocked until the Common Pile histogram/sample and
  source-attribution policy evidence can be produced under approval;
- do not generalize frontier admission into later package execution without a
  new approval artifact;
- keep the run-admission manifest synchronized with the remaining-blocker map
  whenever frontier outputs change.

## 2026-06-20 External Condition Registry Pass

Added a checked registry for run-admission blockers that are neither
run-admission steps nor remaining-blocker-map rows.

Added:

- `artifacts/preflight/external_condition_registry.json`;
- `tools/external_condition_registry_check.py`;
- `artifacts/preflight/external_condition_registry_check_report.json`.

Updated:

- `tools/run_fixture_checks.py`;
- `tools/se1_preflight_audit.py`;
- `tools/canonical_contract_status_check.py`;
- `artifacts/preflight/README.md`;
- `README.md`;
- `research/EKLAVYA_BENCHMARK_CONTRACT.md`.

The registry owns:

- `model_access_not_approved`;
- `student_baseline_without_teacher_packets`;
- `gate_metric_bindings_not_real_evidence`.

The checker verifies that those IDs exactly match the external dependencies in
`artifacts/preflight/run_admission_manifest.json`, remain uncleared, and keep
download, harness, and training flags false.

Current status:

```text
external_condition_registry_check: passed_not_cleared
external_conditions_checked: 3
harness_allowed: false
training_allowed: false
```

Open holes that remain:

- do not clear model access without explicit project-owner approval and a
  run-admission update;
- do not run exact-oracle evaluation until a teacher-free student baseline
  exists;
- do not treat gate metric bindings as real gate evidence.

## 2026-06-20 Accomplishment Requirement Blocker Coverage Pass

Tightened the accomplishment contract checker so every required real evidence
ref must be covered by the requirement's own `blocked_by` list through either a
run-admission step or a remaining-blocker-map row.

Updated:

- `tools/accomplishment_contract_check.py`;
- `artifacts/accomplishment/accomplishment_contract.json`;
- `artifacts/accomplishment/accomplishment_contract_check.json`;
- `artifacts/accomplishment/accomplishment_progress_report.json`;
- `artifacts/accomplishment/README.md`.

The stricter checker exposed missing requirement-level blockers for:

- E1 metrics evidence;
- E2 accepted teacher bindings;
- S0 D1 byte-exactness evidence;
- S2 D1 byte-exactness evidence.

Those requirement rows now name the relevant admission steps and blocker-map
rows, while keeping both Eklavya and Sutra claims blocked.

Current status:

```text
accomplishment_contract_check: passed_not_accomplished
claims_checked: 2
requirements_checked: 10
evidence_blockers_checked: 23
harness_allowed: false
training_allowed: false
```

Open holes that remain:

- keep requirement-level blockers synchronized with every required real
  evidence ref;
- do not treat a requirement as complete until every blocker is retired by
  approved real evidence;
- keep Eklavya and Sutra accomplishment claims blocked until the contract,
  progress report, work order, claim boundary, and gate evidence all promote
  together.

## 2026-06-20 Gate Metric Artifact Blocker Coverage Pass

Tightened the gate metric binding checker so every future gate-evidence
artifact ref must be owned by the run-admission manifest and blocker map.

Updated:

- `tools/gate_metric_binding_check.py`;
- `artifacts/gates/gate_metric_binding_check.json`;
- `artifacts/gates/README.md`.

The checker now reads `artifacts/preflight/run_admission_manifest.json` and
`artifacts/preflight/remaining_blocker_map.json`, then fails if any gate
binding's `required_artifact_refs` path is not assigned to an admission step or
has no blocker coverage.

Current status:

```text
gate_metric_binding_check: passed_no_gate_results
gates_checked: 6
artifact_blockers_checked: 17
harness_allowed: false
training_allowed: false
```

Open holes that remain:

- keep gate artifact refs synchronized with run-admission and blocker-map
  ownership;
- do not accept gate pass/fail rows until every required artifact ref is real
  evidence, not a template;
- keep metric logs and summary reports absent until a run is approved.

## 2026-06-21 Teacher Selection Evidence Coverage Pass

Tightened the teacher-selection checker so every selection-gate evidence ref is
classified as either an existing design ref or a missing real ref with
run-admission and blocker-map coverage.

Updated:

- `tools/teacher_selection_check.py`;
- `artifacts/teacher_selection/teacher_selection_check.json`;
- `artifacts/teacher_selection/README.md`.

The checker now reads `artifacts/preflight/run_admission_manifest.json` and
`artifacts/preflight/remaining_blocker_map.json`. It fails if a missing
selection-gate evidence ref is not assigned to an admission step or lacks
blocker coverage.

Current status:

```text
teacher_selection_check: passed_no_teacher_admitted
candidate_count: 7
missing_real_refs_checked: 6
evidence_blockers_checked: 8
harness_allowed: false
training_allowed: false
```

Open holes that remain:

- keep teacher-selection gate refs synchronized with run-admission and blocker
  ownership;
- do not accept any teacher until all five selection gates have real evidence;
- keep source ecology, teacher ports, and selection gates aligned before any
  teacher binding is accepted.

## 2026-06-21 Teacher Binding Dependency Coverage Pass

Cleaned the worktree before proceeding: the tracked tree was clean, fixture
checks passed, `tools/__pycache__/` was removed, and the only remaining
untracked path in `git clean -ndx` was `CLAUDE.md`, which was left untouched.

Tightened the teacher-binding checker so missing binding dependencies are not
only missing in the candidate manifest, but also owned by the run-admission
manifest and remaining-blocker map.

Updated:

- `tools/teacher_binding_manifest_check.py`;
- `artifacts/teacher_bindings/candidate_binding_manifest_check.json`;
- `artifacts/teacher_bindings/README.md`.

The checker now reads `artifacts/preflight/run_admission_manifest.json` and
`artifacts/preflight/remaining_blocker_map.json`. It fails if runtime rows,
tokenizer behavior records, signature tensors, or accepted teacher bindings
exist prematurely, are missing from run-admission `forbidden_until_approval`,
or lack blocker-map coverage.

Current status:

```text
teacher_binding_manifest_check: binding_manifest_check_passed_no_real_bindings
candidate_binding_count: 4
accepted_binding_count: 0
dependency_blockers_checked: 4
dependency_paths_checked: 6
harness_allowed: false
training_allowed: false
```

Open holes that remain:

- keep teacher-binding dependency paths synchronized with run-admission and
  blocker-map ownership;
- do not create `artifacts/teacher_bindings/accepted_teacher_bindings.json`
  until runtime, tokenizer, and signature evidence exists;
- keep teacher bindings, teacher ports, source ecology, and selection gates
  aligned before any teacher can be admitted.

## 2026-06-21 Claim Boundary Derived Readiness Pass

Normalized the claim-boundary treatment of
`readiness_record_not_ready_for_harness`. It remains a readable blocker on
Eklavya/Sutra accomplishment claims, but is now a checked derived condition
rather than a free-form report-local blocker name.

Updated:

- `tools/claim_boundary_check.py`;
- `artifacts/claims/claim_boundary.json`;
- `artifacts/claims/claim_boundary_check.json`;
- `artifacts/claims/README.md`.

The checker now defines the readiness-derived condition explicitly. Any claim
using it must cite `artifacts/readiness/readiness_record.json`, and the checker
verifies that the readiness verdict remains `not_ready_for_harness`. The Sutra
claim now cites the readiness record directly because it already depended on
that derived blocker.

Current status:

```text
claim_boundary_check: claim_boundary_check_passed_no_claims_admitted
claims_checked: 7
derived_conditions_checked: readiness_record_not_ready_for_harness
readiness_verdict_checked: not_ready_for_harness
harness_allowed: false
training_allowed: false
```

Open holes that remain:

- keep derived claim conditions explicit in the checker rather than hidden in
  claim `blocked_by` lists;
- do not admit Eklavya or Sutra claims while the readiness record verdict is
  `not_ready_for_harness`;
- keep claim evidence refs synchronized with run-admission, blocker-map, and
  readiness-record ownership.

## 2026-06-21 Row Preparation Blocker Coverage Pass

Replaced row-preparation report-local blocker labels with canonical
run-admission and remaining-blocker coverage.

Updated:

- `tools/row_preparation_manifest_check.py`;
- `artifacts/scout_slices/row_preparation_manifest_check.json`;
- `artifacts/scout_slices/README.md`.

The checker now reads `artifacts/preflight/run_admission_manifest.json` and
`artifacts/preflight/remaining_blocker_map.json`. Every blocked prepared-row
path must remain absent, must be forbidden by `row_generation_and_preparation`,
and must have blocker-map coverage such as
`real_D1_D2_D5_byte_length_profiles` or
`real_D4_jsonschema_prepared_rows`. Held D5 slices also keep explicit
`source_attribution_distribution_policy` coverage.

Current status:

```text
row_preparation_manifest_check: planned_manifest_check_passed_no_prepared_rows
blocked_prepared_row_paths_checked: 5
held_slice_ids: D5_instruction_guardrail, D5_tinystories_generation_guardrail
remaining_blockers: real_D1_D2_D5_byte_length_profiles, real_D4_jsonschema_prepared_rows, row_generation_and_preparation, source_attribution_distribution_policy
harness_allowed: false
training_allowed: false
```

Open holes that remain:

- keep planned prepared-row paths synchronized with run-admission and
  blocker-map ownership;
- do not create prepared D1/D2/D4/D5 row files before source admission and
  row-generation approval;
- keep held-source distribution policy blockers in place before any D5 slice
  can leave held status.

## 2026-06-21 D1 Generator Blocker Coverage Pass

Replaced D1 generator report-local blocker labels with canonical
run-admission and remaining-blocker coverage for forbidden D1 outputs.

Updated:

- `tools/d1_generator_manifest_check.py`;
- `artifacts/d1_byte_exactness/generator_manifest_check.json`;
- `artifacts/d1_byte_exactness/README.md`.

The checker now reads `artifacts/preflight/run_admission_manifest.json` and
`artifacts/preflight/remaining_blocker_map.json`. `rows.jsonl` and
`eval_report.json` must remain absent, must be forbidden by
`row_generation_and_preparation`, and must map to
`real_D1_D2_D5_byte_length_profiles` and `real_D1_byte_exactness_eval`.

Current status:

```text
D1_generator_manifest_check: manifest_check_passed_rows_not_generated
forbidden_outputs_checked: 2
remaining_blockers: real_D1_D2_D5_byte_length_profiles, real_D1_byte_exactness_eval, row_generation_and_preparation
harness_allowed: false
training_allowed: false
```

Open holes that remain:

- do not create D1 generated rows before row-generation approval;
- do not create D1 eval evidence before the exact D1 eval blocker is retired;
- keep D1 generator outputs synchronized with row-preparation, shape-profile,
  and readiness blockers.

## 2026-06-21 Exact Oracle Blocker Coverage Pass

Replaced exact-oracle fixture report-local blocker labels with canonical
run-admission, remaining-blocker, and external-condition coverage.

Updated:

- `tools/exact_oracle_fixture.py`;
- `artifacts/exact_oracles/oracle_fixture_report.json`;
- `artifacts/exact_oracles/README.md`.

The checker now reads `artifacts/preflight/run_admission_manifest.json`,
`artifacts/preflight/remaining_blocker_map.json`, and
`artifacts/preflight/external_condition_registry.json`. D2/D4 prepared rows,
the real oracle eval report, and teacher signature records must remain absent
with admission-step and blocker-map coverage. The student-baseline external
condition must remain active and uncleared for `exact_oracle_eval`.

Current status:

```text
exact_oracle_fixture_report: fixture_oracle_checks_passed_not_real_eval
case_count: 4
blocked_evidence_paths_checked: 4
external_conditions_checked: student_baseline_without_teacher_packets
harness_allowed: false
training_allowed: false
```

Open holes that remain:

- do not run real exact-oracle evaluation before D2/D4 prepared rows and the
  teacher-free student baseline exist;
- keep oracle fixture mechanics separated from real oracle eval evidence;
- do not use exact-oracle fixture results as teacher-surface or retained-gain
  evidence before real signatures and baseline controls exist.

## 2026-06-21 Gate Packet Blocker Coverage Pass

Replaced gate-packet template report-local blocker labels with canonical
run-admission and remaining-blocker coverage for the future evidence needed to
turn packet rows into actual evidence.

Updated:

- `tools/gate_packet_template_check.py`;
- `artifacts/gates/gate_packet_template_check.json`;
- `artifacts/gates/README.md`.

The checker now reads `artifacts/preflight/run_admission_manifest.json` and
`artifacts/preflight/remaining_blocker_map.json`. Prepared rows, G0/G1
evidence, teacher bindings, teacher signatures, exact-oracle eval evidence,
baseline reports, metric logs, and gate evidence must remain absent and must
map to admission steps and blocker-map rows before packet rows can promote.

Current status:

```text
gate_packet_template_check: template_check_passed_rows_unfilled
gate_rows_checked: 6
initial_packet_rows_checked: 6
blocked_evidence_paths_checked: 13
harness_allowed: false
training_allowed: false
```

Open holes that remain:

- keep packet rows as templates until every source, runtime, G0/G1, oracle,
  baseline, and gate evidence path is real and admitted;
- do not treat packet rows or initial gate rows as pass/fail evidence;
- keep later G2-G5 packet claims blocked by first-harness and gate-claim
  admission steps.

## 2026-06-21 Baseline Gap Blocker Coverage Pass

Replaced baseline-gap template report-local blocker labels with canonical
run-admission, remaining-blocker, and external-condition coverage.

Updated:

- `tools/baseline_gap_template_check.py`;
- `artifacts/baseline_gap/baseline_gap_template_check.json`;
- `artifacts/baseline_gap/README.md`.

The checker now reads `artifacts/preflight/run_admission_manifest.json`,
`artifacts/preflight/remaining_blocker_map.json`, and
`artifacts/preflight/external_condition_registry.json`. Baseline-gap reports,
retained-gain reports, metric logs, and metric summaries must remain absent
with `first_harness_scope` and `baseline_gap_measurements` coverage. The
student-baseline external condition must remain active and uncleared.

Current status:

```text
baseline_gap_template_check: template_check_passed_no_baseline_results
checks_count: 5
forbidden_result_artifacts_checked: 2
blocked_metric_artifacts_checked: 2
external_conditions_checked: student_baseline_without_teacher_packets
harness_allowed: false
training_allowed: false
```

Open holes that remain:

- do not create baseline-gap or retained-gain reports before first-harness
  admission;
- keep metric logs and summaries absent until real first-harness measurements
  exist;
- do not retire the baseline blockers until teacher-free student baseline,
  teacher-exposed gap, retained-gain, and best-single-teacher/no-teacher
  controls are real evidence.

## 2026-06-21 Blocker Reference Check Pass

Codified the blocker-reference scan that previously existed only as an ad hoc
validation command.

Updated:

- `tools/blocker_reference_check.py`;
- `tools/run_fixture_checks.py`;
- `tools/se1_preflight_audit.py`;
- `artifacts/preflight/blocker_reference_check_report.json`;
- `artifacts/preflight/se1_fixture_preflight_report.json`;
- `artifacts/preflight/README.md`;
- `artifacts/README.md`.

The new checker scans generated artifact JSON for blocker-like fields such as
`blocked_by`, `remaining_blockers`, `blockers_checked`,
`evidence_blockers_checked`, and `external_conditions_checked`. It fails if a
referenced ID is not a run-admission step, remaining-blocker-map row, external
condition, or checked derived condition. The default fixture runner executes it
both before and after the aggregate preflight audit so the preflight audit can
require the report while the final blocker-reference report still sees the
freshly regenerated preflight report.

Current status:

```text
blocker_reference_check: blocker_reference_check_passed
unknown_reference_count: 0
references_checked_count: 280
harness_allowed: false
training_allowed: false
```

Open holes that remain:

- keep new blocker-like report fields registered in
  `tools/blocker_reference_check.py`;
- do not introduce local blocker aliases that are not owned by run admission,
  remaining blocker map, external conditions, or derived-condition checks;
- keep the blocker-reference report required by the aggregate preflight audit.

## 2026-06-21 Accomplishment External Conditions Work Order Pass

Expanded the accomplishment work order so external admission conditions are
first-class executable blockers rather than bare string IDs.

Updated:

- `tools/accomplishment_work_order.py`;
- `tools/blocker_reference_check.py`;
- `artifacts/accomplishment/accomplishment_work_order.json`;
- `artifacts/accomplishment/README.md`;
- `artifacts/preflight/blocker_reference_check_report.json`.

The work-order generator now reads
`artifacts/preflight/external_condition_registry.json`, validates that each
external condition is present, uncleared, active, and used by the relevant
admission step, then emits its owner, type, blocking reason, clearance
requirements, and clearance artifact. The blocker-reference checker now scans
`external_conditions` fields as canonical references too.

Current status:

```text
accomplishment_work_order: accomplishment_work_order_passed_not_admitted
external_conditions_checked: gate_metric_bindings_not_real_evidence, model_access_not_approved, student_baseline_without_teacher_packets
harness_allowed: false
training_allowed: false
```

Open holes that remain:

- do not clear model access without explicit owner approval and a run-admission
  update;
- do not run exact-oracle evaluation until a teacher-free student baseline is
  real and integrated into the first harness scope;
- do not accept gate claims while gate metric bindings remain template-only.

## 2026-06-21 D0 Frontier Execution Runbook Pass

Materialized the D0 source-admission frontier as a machine-checked operator
runbook instead of leaving the next real step split between prose, the packet,
and ad hoc commands.

Updated:

- `artifacts/dataset_license/d0_frontier_execution_runbook.json`;
- `tools/d0_frontier_execution_runbook_check.py`;
- `artifacts/dataset_license/d0_source_admission_packet.json`;
- `tools/d0_source_admission_packet_check.py`;
- `tools/run_fixture_checks.py`;
- `tools/se1_preflight_audit.py`;
- `artifacts/dataset_license/README.md`.

The new runbook records the exact owner-approval boundary, local input-map
requirements, pre-approval fixture commands, planned real row-filter command,
blocked D0 outputs, and source-policy requirements. Its checker verifies that
D0 remains not admitted, all blocked outputs and the real input map remain
absent, the packet still points to the runbook, pre-approval commands cannot
create real Common Pile reports, and the runbook mirrors the input-map contract,
run-admission manifest, and accomplishment work order.

Current status:

```text
d0_frontier_execution_runbook: d0_frontier_execution_runbook_not_admitted
d0_frontier_execution_runbook_check: d0_frontier_execution_runbook_check_passed_not_admitted
blocked_outputs_checked: 3
required_dataset_ids_checked: 3
harness_allowed: false
training_allowed: false
```

Open holes that remain:

- do not create the local Common Pile input map before explicit owner approval;
- do not run the real Common Pile row filter while D0 remains not admitted;
- do not treat the D0 runbook as source evidence or downstream row-generation
  approval.

## 2026-06-21 D0 Source Admission Approval Template Pass

Added a checked template for the future D0 project-owner approval artifact
without creating or admitting the real approval artifact.

Updated:

- `artifacts/dataset_license/d0_source_admission_approval_template.json`;
- `tools/d0_source_admission_approval_template_check.py`;
- `artifacts/dataset_license/d0_source_admission_packet.json`;
- `artifacts/dataset_license/d0_frontier_execution_runbook.json`;
- `tools/d0_source_admission_packet_check.py`;
- `tools/d0_frontier_execution_runbook_check.py`;
- `tools/run_fixture_checks.py`;
- `tools/se1_preflight_audit.py`;
- `artifacts/dataset_license/README.md`;
- `artifacts/preflight/README.md`.

The approval template names the absent real approval artifact
`artifacts/dataset_license/d0_source_admission_approval.json`, the required
decision fields, the permitted D0 outputs after approval, explicit
non-authorizations, validation commands, and downstream release conditions. Its
checker verifies the template remains unapproved, the real approval artifact is
absent, the packet and runbook point back to the template, and no training,
harness, dataset download, model download, held-source use, or downstream row
generation is authorized by the template.

Current status:

```text
d0_source_admission_approval_template: d0_source_admission_approval_template_not_approved
d0_source_admission_approval_template_check: d0_source_admission_approval_template_check_passed_not_approved
approval_artifact_path: artifacts/dataset_license/d0_source_admission_approval.json
harness_allowed: false
training_allowed: false
```

Open holes that remain:

- do not create `artifacts/dataset_license/d0_source_admission_approval.json`
  until the project owner actually approves real D0 execution;
- do not use the template as permission to create blocked D0 outputs;
- keep downstream row generation blocked until D0 is admitted in the run
  admission manifest after the real outputs are reviewed.

## 2026-06-21 D0 Local Input Map Template Pass

Added a checked template for the future project-owner-supplied local Common
Pile input map without creating the real input map or admitting D0.

Updated:

- `artifacts/dataset_license/d0_local_input_map_template.json`;
- `tools/d0_local_input_map_template_check.py`;
- `artifacts/dataset_license/d0_input_map_contract.json`;
- `tools/d0_input_map_contract_check.py`;
- `artifacts/dataset_license/d0_source_admission_packet.json`;
- `tools/d0_source_admission_packet_check.py`;
- `artifacts/dataset_license/d0_frontier_execution_runbook.json`;
- `tools/d0_frontier_execution_runbook_check.py`;
- `artifacts/dataset_license/d0_source_admission_approval_template.json`;
- `tools/d0_source_admission_approval_template_check.py`;
- `tools/run_fixture_checks.py`;
- `tools/se1_preflight_audit.py`;
- `artifacts/dataset_license/README.md`;
- `artifacts/preflight/README.md`.

The local input-map template records the required JSON object shape, candidate
dataset IDs, manifest revisions, accepted local path suffixes, held-shard
exclusion, remote-prefix rejection, and local-file existence checks for the
future `artifacts/dataset_license/local_common_pile_input_map.json`. Its
checker defaults to template-only mode and fails if the real map appears before
explicit D0 approval; it also has an explicit future validation mode for an
approved candidate map.

Current status:

```text
d0_local_input_map_template: d0_local_input_map_template_no_real_inputs
d0_local_input_map_template_check: d0_local_input_map_template_check_passed_no_real_inputs
real_input_map_path: artifacts/dataset_license/local_common_pile_input_map.json
candidate_entries_checked: 3
harness_allowed: false
training_allowed: false
```

Open holes that remain:

- do not create or commit `local_common_pile_input_map.json` before project
  owner D0 approval;
- do not use placeholder paths as evidence of available Common Pile data;
- validate any future real map with explicit approval before running the real
  Common Pile row filter.

## 2026-06-21 D0 Row Filter Preservation Guard Pass

Hardened the Common Pile row filter so license admission is not enough by
itself: kept rows must also preserve the manifest-required metadata fields.

Updated:

- `tools/common_pile_license_filter.py`;
- `tools/d0_source_admission_packet_check.py`;
- `artifacts/dataset_license/fixtures/arxiv_abstracts_fixture.jsonl`;
- `artifacts/dataset_license/common_pile_fixture_license_histogram.json`;
- `artifacts/dataset_license/common_pile_fixture_sample_manifest.json`;
- `artifacts/dataset_license/d0_input_map_contract_check.json`;
- `artifacts/dataset_license/d0_source_admission_packet_check.json`;
- `artifacts/dataset_license/README.md`.

The row filter now rejects allowed-license rows that are missing the configured
text field or any `required_preserved_fields` from
`common_pile_row_filter_manifest.json`. Fixture reports now record
`text_field`, `license_field`, `required_preserved_fields`, and
`missing_required_field_counts` per shard. The arXiv fixture includes a local
allowed-license row with no `metadata.url`, proving the guard produces
`missing_required_preserved_field` without changing D0 admission status.

Current status:

```text
common_pile_fixture_license_histogram: local_fixture_not_common_pile_evidence
arxiv_fixture_row_count_before_filter: 4
arxiv_fixture_row_count_after_filter: 1
arxiv_missing_required_field_counts: metadata.url=1
harness_allowed: false
training_allowed: false
```

Open holes that remain:

- do not treat fixture row-filter results as Common Pile evidence;
- run the same preserved-field guard on approved local Common Pile files only
  after D0 approval;
- do not prepare downstream rows unless admitted source rows carry the required
  source URL, source identity, license, and revision metadata.

## 2026-06-21 Source Attribution Boundary Cross-Link Pass

Hardened the held-source attribution/distribution template so it is not just a
standalone held-source list. It now proves its links to the D0 packet, frontier
runbook, approval template, row-preparation manifest, run-admission manifest,
remaining-blocker map, and share-alike review.

Updated:

- `artifacts/dataset_license/source_attribution_distribution_template.json`;
- `tools/source_attribution_distribution_check.py`;
- `artifacts/dataset_license/source_attribution_distribution_template_check.json`;
- `artifacts/dataset_license/README.md`.

The checker now verifies that the real policy path remains absent, the D0
packet and runbook point at the template/check pair, the approval template lists
the real source policy as a permitted D0 output only after approval, held-source
training and downstream row generation remain explicitly unauthorized, the
source-attribution blocker and run-admission manifest cover the real policy
path, and held TinyStories/Dolly row-preparation slices remain zero-row,
preparation-forbidden, and tied to blocked D5 prepared-row paths.

Current status:

```text
source_attribution_distribution_template_check: template_check_passed_no_held_source_admitted
held_sources_checked: allenai/ai2_arc, databricks/databricks-dolly-15k, roneneldan/TinyStories
held_slice_ids_checked: D5_instruction_guardrail, D5_tinystories_generation_guardrail
downstream_blocked_paths_checked: D5_dolly_instruction_guardrail, D5_tinystories_guardrail
harness_allowed: false
training_allowed: false
```

Open holes that remain:

- do not create `source_attribution_distribution_policy.json` before D0
  approval and real source review;
- do not admit held ARC, TinyStories, or Dolly rows without the full attribution
  and distribution policy fields;
- keep D5 prepared-row paths absent until the held-source policy blocker is
  retired through real evidence, not fixture reports.

## 2026-06-21 Row Preparation Contract Guard Pass

Closed an interrupted worktree checkpoint by making the planned row-preparation
manifest self-enforcing. The manifest now records the prepared-row field
contract and every planned slice records the source revision, license status,
required prepared-row fields, and source-policy requirement needed before any
future JSONL rows may be created.

Updated:

- `artifacts/scout_slices/row_preparation_manifest.json`;
- `tools/row_preparation_manifest_check.py`;
- `artifacts/scout_slices/row_preparation_manifest_check.json`;
- `artifacts/scout_slices/README.md`.

The checker now validates that the row contract remains contract-only, the
required prepared-row fields are present on every slice, local generated rows
stay tied to a future harness commit, external slices carry real source
revisions, and held TinyStories/Dolly slices point to
`artifacts/dataset_license/source_attribution_distribution_policy.json`. It also
cross-checks `source_attribution_distribution_template.json` so the held D5
row-preparation slice IDs and blocked prepared-row paths stay aligned with the
source-attribution boundary.

Current status:

```text
row_preparation_manifest_check: planned_manifest_check_passed_no_prepared_rows
prepared_row_contract_fields_checked: row_id, slice_id, source, source_revision, license_status, split, input_payload, target_payload, provenance
source_policy_held_slices_checked: D5_instruction_guardrail, D5_tinystories_generation_guardrail
harness_allowed: false
training_allowed: false
```

Open holes that remain:

- do not treat the row contract as prepared training data;
- do not create any blocked prepared-row JSONL before the run-admission and
  blocker maps are retired through real evidence;
- keep D5 held-source rows blocked until the real source attribution and
  distribution policy exists after D0 approval.

## 2026-06-21 D1 Prepared-Row Envelope Guard Pass

Aligned the local D1 byte-exactness generator manifest with the central
prepared-row contract. Future D1 rows must now use the same top-level
`row_id`, `slice_id`, `source`, `source_revision`, `license_status`, `split`,
`input_payload`, `target_payload`, and `provenance` envelope as the scout-slice
row-preparation manifest.

Updated:

- `artifacts/d1_byte_exactness/generator_manifest.json`;
- `tools/d1_generator_manifest_check.py`;
- `artifacts/d1_byte_exactness/generator_manifest_check.json`;
- `artifacts/d1_byte_exactness/README.md`.

The checker now reads `artifacts/scout_slices/row_preparation_manifest.json`
and verifies that D1 total rows, split counts, prepared-row path, local source
status, source revision requirement, license status, and source-policy
requirement match the central row-preparation slice. It also verifies the D1
payload shape: raw text and raw bytes in `input_payload`, byte-length family
targets in `target_payload`, and generator manifest, seed, and git commit in
`provenance`.

Current status:

```text
d1_generator_manifest_check: manifest_check_passed_rows_not_generated
row_preparation_slice_id_checked: D1_local_byte_exactness
prepared_row_contract_fields_checked: row_id, slice_id, source, source_revision, license_status, split, input_payload, target_payload, provenance
generated_rows_allowed: false
harness_allowed: false
training_allowed: false
```

Open holes that remain:

- do not create `artifacts/d1_byte_exactness/rows.jsonl` before
  `row_generation_and_preparation` is admitted;
- do not run or claim the D1 byte-exactness eval before real D1 rows exist;
- keep the future D1 row `source_revision` tied to the git commit that actually
  generated the rows, not to this fixture-only contract.

## 2026-06-21 Exact Oracle Prepared-Row Contract Pass

Linked the exact-oracle fixture manifest to the central row-preparation
contract for D2 and D4. The oracle checker now proves that its accepted slice
IDs, prepared-row paths, source-policy requirements, and prepared-row field
contract still match `artifacts/scout_slices/row_preparation_manifest.json`.

Updated:

- `artifacts/exact_oracles/oracle_manifest.json`;
- `tools/exact_oracle_fixture.py`;
- `artifacts/exact_oracles/oracle_fixture_report.json`;
- `artifacts/exact_oracles/README.md`.

The checker still validates only hand-authored oracle fixtures. It now also
rejects any oracle manifest that accepts held slices, points at unblocked
prepared-row paths, omits the prepared-row envelope, or lets D2/D4 drift from
their external dataset-revision/license/attribution requirement.

Current status:

```text
exact_oracle_fixture: fixture_oracle_checks_passed_not_real_eval
oracle_slice_contracts_checked: D2_arithmetic_candidate_reasoning, D4_json_schema_verifier
oracle_prepared_rows_paths_checked: D2_gsm8k_main.jsonl, D4_jsonschema_github_easy.jsonl
harness_allowed: false
training_allowed: false
```

Open holes that remain:

- do not create `artifacts/exact_oracles/oracle_eval_report.json` before real
  prepared D2/D4 rows and student-baseline context exist;
- do not treat fixture oracle cases as benchmark evidence;
- keep exact-oracle accomplishment claims blocked until real oracle reports
  reference admitted prepared rows and baseline context.

## 2026-06-21 Shape Byte-Profile Prepared-Row Contract Pass

Aligned the byte-length profile manifest with the central row-preparation
contract so future G0 profiles cannot silently use stale slice names or
held-source rows. The fixture extraction still uses hand-written fixture fields,
but the real-profile contract now names the prepared-row envelope and the exact
row-preparation slice IDs behind each profile.

Updated:

- `artifacts/shape_manifest/byte_profile_manifest.json`;
- `tools/byte_length_profile.py`;
- `artifacts/shape_manifest/byte_length_fixture_report.json`;
- `artifacts/shape_manifest/README.md`.

The checker now verifies that D1 maps to `D1_local_byte_exactness`, D2 maps to
`D2_arithmetic_candidate_reasoning`, and D5 maps to the two held D5
row-preparation slices. D5 real profile rows are explicitly held at
`minimum_real_rows_required: 0` until source attribution and distribution policy
approval, rather than pretending a D5 real profile can be filled from fixture
rows.

Current status:

```text
byte_length_profile: handwritten_fixture_not_G0_evidence
row_preparation_slices_checked: D1_local_byte_exactness, D2_arithmetic_candidate_reasoning, D5_tinystories_generation_guardrail, D5_instruction_guardrail
D5_profile_status: held_no_real_rows_until_source_policy_admission
harness_allowed: false
training_allowed: false
```

Open holes that remain:

- do not treat fixture byte-length profiles as G0 shape evidence;
- do not create `real_D1_D2_D5_byte_lengths.json` until admitted prepared rows
  exist, with D5 either admitted or explicitly recorded as held;
- keep G0 shape reports absent until real byte profiles replace the fixture
  profile.

## 2026-06-21 Held Source Canonical Slice-ID Pass

Canonicalized the held-source intended slice IDs in the source-attribution
template and share-alike review. ARC now points to
`D2_arithmetic_candidate_reasoning`, TinyStories points to
`D5_tinystories_generation_guardrail`, and Dolly points to
`D5_instruction_guardrail`.

Updated:

- `artifacts/dataset_license/source_attribution_distribution_template.json`;
- `artifacts/dataset_license/sharealike_review.md`;
- `tools/source_attribution_distribution_check.py`;
- `artifacts/dataset_license/source_attribution_distribution_template_check.json`;
- `artifacts/dataset_license/README.md`.

The source-attribution checker now fails if those canonical intended slices
drift, or if the share-alike review reintroduces stale labels such as
`D2_candidate_reasoning`, `D5_small_model_generation_guardrail`, or
`D5_instruction_generation_guardrail`.

Current status:

```text
source_attribution_distribution_template_check: template_check_passed_no_held_source_admitted
intended_slices_checked: D2_arithmetic_candidate_reasoning, D5_tinystories_generation_guardrail, D5_instruction_guardrail
harness_allowed: false
training_allowed: false
```

Open holes that remain:

- do not admit ARC, TinyStories, or Dolly without the real source attribution
  and distribution policy;
- keep the canonical slice IDs synchronized with row-preparation slice IDs if
  any held source is later admitted;
- keep the share-alike review as repo policy readiness only, not legal
  clearance.

## 2026-06-21 Teacher Runtime Scope Binding Pass

Tied future teacher-runtime measurement scope to teacher ports and candidate
bindings. The runtime manifest now points to the teacher-port contract and
candidate-binding manifest, enumerates required external model teacher IDs, and
keeps optional runtime candidates separate from admitted teachers.

Updated:

- `artifacts/teacher_runtime/measurement_manifest.json`;
- `tools/teacher_runtime_report.py`;
- `artifacts/teacher_runtime/runtime_fixture_report.json`;
- `artifacts/teacher_bindings/candidate_binding_manifest.json`;
- `artifacts/teacher_bindings/candidate_binding_manifest_check.json`;
- `artifacts/teacher_runtime/README.md`;
- `artifacts/teacher_bindings/README.md`.

The runtime checker now verifies that every external model teacher port is in
the future measurement order. It also exposed and closed a scope gap by adding
the missing blocked BGE small geometry-control binding
`bind_bge_small_geometry_control_v0`. The binding is not accepted and still has
missing runtime, tokenizer, and signature records.

Current status:

```text
teacher_runtime_report: fixture_not_teacher_runtime_evidence
external_model_teacher_ids_checked: BAAI/bge-small-en-v1.5, control-decoder-0.6B, embedding-candidate-0.6B, anchor-decoder-1.7B
missing_candidate_binding_scope: none
candidate_binding_count: 5
harness_allowed: false
training_allowed: false
```

Open holes that remain:

- do not run or download any teacher model before model access and runtime
  measurement are explicitly approved;
- keep all candidate bindings unaccepted until real runtime, tokenizer, and
  signature records exist;
- keep optional runtime candidates as planning entries unless they receive
  teacher-port and binding coverage.

## 2026-06-21 Tokenizer Scope Binding Pass

Tied future teacher-tokenizer behavior scope to teacher ports and candidate
bindings. The tokenizer manifest now points to the teacher-port contract and
candidate-binding manifest, and enumerates the required external model teacher
IDs that would need real tokenizer records before admission.

Updated:

- `artifacts/tokenizer_behavior/record_manifest.json`;
- `tools/tokenizer_behavior_report.py`;
- `artifacts/tokenizer_behavior/tokenizer_fixture_report.json`;
- `artifacts/tokenizer_behavior/README.md`.

The tokenizer checker now verifies that every external model teacher port has a
blocked candidate-binding row, that no such binding is accepted, and that all
external binding rows still have `tokenizer_behavior_record: missing`. The
checked report remains a UTF-8 byte-identity fixture only.

Current status:

```text
tokenizer_fixture_report: fixture_not_real_tokenizer_evidence
external_model_teacher_ids_checked: BAAI/bge-small-en-v1.5, control-decoder-0.6B, anchor-decoder-1.7B, embedding-candidate-0.6B
missing_candidate_binding_scope: none
real_tokenizer_behavior_records: none
harness_allowed: false
training_allowed: false
```

Open holes that remain:

- do not download or load any teacher tokenizer before model access is
  explicitly approved;
- keep all candidate bindings unaccepted until runtime, tokenizer behavior, and
  signature records exist;
- keep the byte-identity fixture separate from real primary family, BGE, or any other
  teacher-tokenizer evidence.

## 2026-06-21 Ksig Signature Scope Binding Pass

Tied fixture Ksig signature generation and surface-correlation reports to the
candidate teacher-binding scope. The Ksig schema now points to teacher ports,
candidate bindings, and prompt templates, and enumerates the required candidate
binding IDs whose future real signature outputs would be needed before
admission.

Updated:

- `artifacts/teacher_signatures/ksig_schema.json`;
- `tools/ksig_surface.py`;
- `artifacts/teacher_signatures/ksig_fixture_signature_records.json`;
- `artifacts/teacher_signatures/ksig_fixture_surface_correlation.json`;
- `artifacts/teacher_signatures/README.md`;
- `README.md`.

The Ksig generator now verifies that every candidate binding remains blocked,
unaccepted, connected to a teacher port, attached to compatible prompt
templates, and still has `signature_output: missing`. Fixture generation also
refuses non-`fixture/` teachers.

Current status:

```text
ksig_fixture_signature_records: fixture_not_teacher_evidence
ksig_fixture_surface_correlation: fixture_not_G1_evidence
required_candidate_binding_ids_checked: 5
accepted_binding_count: 0
real_teacher_signature_records: none
harness_allowed: false
training_allowed: false
```

Open holes that remain:

- do not generate real teacher signatures until runtime and tokenizer behavior
  measurement are admitted through real evidence;
- do not treat fixture surface correlations as G1 diversity evidence;
- keep accepted teacher bindings absent until real runtime, tokenizer behavior,
  and signature artifacts all exist.

## 2026-06-21 Teacher Admission Ordering Pass

Fixed the G1 teacher-admission dependency order. The previous run-admission
graph made real signature generation depend on final teacher-binding
acceptance, while the binding contract also required signature outputs before a
binding could be accepted. The corrected graph separates the phases:

```text
G1_teacher_runtime_measurement
teacher_tokenizer_behavior_measurement
teacher_signature_generation
teacher_binding_acceptance
```

Updated:

- `artifacts/preflight/run_admission_manifest.json`;
- `artifacts/preflight/remaining_blocker_map.json`;
- `artifacts/preflight/external_condition_registry.json`;
- `artifacts/readiness/readiness_record.json`;
- `artifacts/claims/claim_boundary.json`;
- `artifacts/accomplishment/accomplishment_contract.json`;
- `artifacts/preflight/README.md`;
- `artifacts/teacher_bindings/README.md`;
- `artifacts/tokenizer_behavior/README.md`;
- `artifacts/teacher_signatures/README.md`.

Current interpretation:

```text
candidate_bindings_define_signature_scope: true
accepted_bindings_require_real_signatures: true
model_access_not_approved_applies_to_runtime_tokenizer_and_signature_measurement: true
harness_allowed: false
training_allowed: false
```

Open holes that remain:

- do not run runtime, tokenizer, or signature measurement without explicit
  model-access approval;
- do not publish `accepted_teacher_bindings.json` until real runtime,
  tokenizer, signature, and surface-correlation artifacts all exist;
- keep signature generation tied to candidate-binding scope until final
  accepted bindings can be justified by real evidence.

## 2026-06-21 Teacher Binding Admission-Step Guard Pass

Hardened the teacher-binding checker so the corrected G1 admission order is
machine-enforced, not just documented.

Updated:

- `tools/teacher_binding_manifest_check.py`;
- `artifacts/teacher_bindings/candidate_binding_manifest_check.json`;
- `artifacts/teacher_bindings/README.md`.

The checker now requires blocked dependency paths to live under their exact
admission steps:

```text
runtime rows -> G1_teacher_runtime_measurement
tokenizer behavior records -> teacher_tokenizer_behavior_measurement
teacher signatures -> teacher_signature_generation
accepted bindings -> teacher_binding_acceptance
```

Open holes that remain:

- keep these admission-step assignments fixed until a real run-admission update
  changes the graph and checker together;
- do not accept bindings while runtime, tokenizer, signature, or
  surface-correlation evidence is absent.

## 2026-06-21 Run Admission Graph Guard Pass

Hardened the run-admission manifest checker so admission-step ordering is
explicitly validated. The checker now fails if an admission step depends on a
later step, or if a forbidden real-artifact path is assigned to multiple
admission steps.

Updated:

- `tools/run_admission_manifest_check.py`;
- `artifacts/preflight/run_admission_manifest_check_report.json`;
- `artifacts/preflight/README.md`.

Current status:

```text
run_admission_manifest_check: run_admission_manifest_check_passed_no_run_admitted
forbidden_path_ownership: unique
admission_step_dependencies: earlier_steps_only
harness_allowed: false
training_allowed: false
```

Open holes that remain:

- keep the graph acyclic if future approval artifacts are added;
- keep every absent real-artifact path owned by exactly one admission step.

## 2026-06-21 Canonical Status Gate Refresh

Updated the benchmark-contract status gate so the canonical readiness checklist
must mention the recent teacher-scope and admission-order guards.

Updated:

- `research/EKLAVYA_BENCHMARK_CONTRACT.md`;
- `tools/canonical_contract_status_check.py`;
- `artifacts/readiness/canonical_contract_status_check.json`.

New required status phrases:

```text
teacher_runtime_scope_bound_to_ports_and_bindings
tokenizer_scope_bound_to_ports_and_bindings
ksig_signature_scope_bound_to_candidate_bindings
teacher_admission_ordering_split_and_guarded
teacher_binding_admission_step_guard_created
run_admission_graph_guard_created
```

Open holes that remain:

- keep the canonical benchmark contract synchronized with checked artifacts
  whenever the control plane changes;
- do not use status phrases as evidence of real runtime, tokenizer, signature,
  gate, or training results.

## 2026-06-21 Remaining Blocker Gate-ID Guard Pass

Hardened the remaining-blocker map checker so blocker rows cannot reference
invented or stale gate IDs. A blocker `gate_id` must now be either a gate in
`artifacts/gates/gate_metric_bindings.json` or an admission step in
`artifacts/preflight/run_admission_manifest.json`.

Updated:

- `tools/remaining_blocker_map_check.py`;
- `artifacts/preflight/remaining_blocker_map_check_report.json`;
- `artifacts/preflight/README.md`.

Current status:

```text
remaining_blocker_map_check: remaining_blocker_map_check_passed
blocker_count: 14
gate_ids_checked: canonical_gate_or_admission_ids_only
harness_allowed: false
training_allowed: false
```

Open holes that remain:

- keep blocker gate IDs synchronized with gate bindings and run-admission steps;
- do not retire blockers through renamed or unrecognized gate labels.

## 2026-06-21 Canonical Blocker Guard Status Pass

Updated the canonical benchmark-contract status gate so it must reflect the
remaining-blocker gate-ID guard and its checked report artifact.

Updated:

- `research/EKLAVYA_BENCHMARK_CONTRACT.md`;
- `tools/canonical_contract_status_check.py`;
- `artifacts/readiness/canonical_contract_status_check.json`.

New required status phrase:

```text
remaining_blocker_gate_id_guard_created
```

New required artifact:

```text
artifacts/preflight/remaining_blocker_map_check_report.json
```

Open holes that remain:

- keep canonical status phrases synchronized with blocker-map guard changes;
- do not treat blocker-map consistency as proof that any blocked real artifact
  has been produced.

## 2026-06-21 Universal Run-Admission Output Blocker Guard Pass

Hardened the run-admission manifest checker so every forbidden output path's
remaining-blocker row must be listed in that admission step's `blocked_by`
list. This replaces the earlier D0-only coverage rule.

Updated:

- `artifacts/preflight/run_admission_manifest.json`;
- `tools/run_admission_manifest_check.py`;
- `artifacts/preflight/run_admission_manifest_check_report.json`;
- `artifacts/preflight/README.md`;
- `research/EKLAVYA_BENCHMARK_CONTRACT.md`;
- `tools/canonical_contract_status_check.py`;
- `artifacts/readiness/canonical_contract_status_check.json`.

New required status phrase:

```text
run_admission_output_blocker_guard_created
```

Current status:

```text
run_admission_manifest_check: run_admission_manifest_check_passed_no_run_admitted
output_blocker_coverage: every_forbidden_output_blocker_listed_by_step
harness_allowed: false
training_allowed: false
```

Open holes that remain:

- keep step-level `blocked_by` lists synchronized with blocker-map output
  ownership;
- do not turn output-blocker coverage into approval to create any forbidden
  real artifact.

## 2026-06-21 Required Run-Admission Dependency Guard Pass

Hardened the run-admission manifest checker with required admission-step
dependencies so downstream packages cannot become apparent roots while their
inputs remain blocked. The manifest now makes G0 shape accounting depend on row
generation, first harness scope depend on accepted teacher bindings and G0, and
gate claim acceptance depend on first harness scope.

Updated:

- `artifacts/preflight/run_admission_manifest.json`;
- `tools/run_admission_manifest_check.py`;
- `artifacts/preflight/run_admission_manifest_check_report.json`;
- `artifacts/accomplishment/accomplishment_work_order.json`;
- `artifacts/preflight/README.md`;
- `research/EKLAVYA_BENCHMARK_CONTRACT.md`;
- `tools/canonical_contract_status_check.py`;
- `artifacts/readiness/canonical_contract_status_check.json`.

New required status phrase:

```text
run_admission_required_dependency_guard_created
```

Current status:

```text
run_admission_manifest_check: run_admission_manifest_check_passed_no_run_admitted
required_admission_dependencies: checked
harness_allowed: false
training_allowed: false
```

Open holes that remain:

- keep required dependency policy aligned with the canonical first-harness
  contract;
- do not admit downstream gates just because their dependency edges are now
  explicit.

## 2026-06-21 Artifact Index Real-Evidence Boundary Guard Pass

Hardened the artifact index checker so the public real-evidence boundary in
`artifacts/README.md` must exactly match
`tools/se1_preflight_audit.py`'s `REQUIRED_ABSENT_REAL_ARTIFACTS` list.

Updated:

- `tools/artifact_index_check.py`;
- `artifacts/preflight/artifact_index_check_report.json`;
- `artifacts/README.md`;
- `artifacts/preflight/README.md`;
- `research/EKLAVYA_BENCHMARK_CONTRACT.md`;
- `tools/canonical_contract_status_check.py`;
- `artifacts/readiness/canonical_contract_status_check.json`.

New required status phrase:

```text
artifact_index_real_evidence_boundary_guard_created
```

Current status:

```text
artifact_index_check: artifact_index_check_passed
real_evidence_boundary_paths: synchronized_with_preflight_absent_list
harness_allowed: false
training_allowed: false
```

Open holes that remain:

- keep human-facing README boundary lists generated or checked against the
  executable preflight policy;
- do not remove absent-real-artifact paths from the public list without changing
  the preflight audit and run-admission manifest together.

## 2026-06-21 Artifact Index Operator-Input Boundary Guard Pass

Extended the artifact index checker so the public operator-input boundary in
`artifacts/README.md` must exactly match
`tools/se1_preflight_audit.py`'s `REQUIRED_ABSENT_OPERATOR_INPUTS` list.

Updated:

- `tools/artifact_index_check.py`;
- `artifacts/preflight/artifact_index_check_report.json`;
- `artifacts/README.md`;
- `artifacts/preflight/README.md`;
- `research/EKLAVYA_BENCHMARK_CONTRACT.md`;
- `tools/canonical_contract_status_check.py`;
- `artifacts/readiness/canonical_contract_status_check.json`.

New required status phrase:

```text
artifact_index_operator_input_boundary_guard_created
```

Current status:

```text
artifact_index_check: artifact_index_check_passed
operator_input_boundary_paths: synchronized_with_preflight_absent_list
harness_allowed: false
training_allowed: false
```

Open holes that remain:

- keep operator-provided input maps out of tracked artifacts until D0 approval
  explicitly changes;
- update the README, preflight audit, and D0 runbook together if the real input
  boundary changes.

## 2026-06-21 D0 Dataset License Boundary Guard Pass

Hardened the D0 frontier runbook checker so the public real-artifact boundary
in `artifacts/dataset_license/README.md` must match the runbook's real input
map path plus blocked D0 outputs.

Updated:

- `tools/d0_frontier_execution_runbook_check.py`;
- `artifacts/dataset_license/d0_frontier_execution_runbook_check.json`;
- `artifacts/dataset_license/README.md`;
- `research/EKLAVYA_BENCHMARK_CONTRACT.md`;
- `tools/canonical_contract_status_check.py`;
- `artifacts/readiness/canonical_contract_status_check.json`.

New required status phrase:

```text
d0_dataset_license_boundary_guard_created
```

Current status:

```text
d0_frontier_execution_runbook_check: d0_frontier_execution_runbook_check_passed_not_admitted
d0_dataset_license_boundary: synchronized_with_runbook_blocked_outputs
harness_allowed: false
training_allowed: false
```

Open holes that remain:

- keep the dataset-license README boundary tied to the D0 runbook and approval
  template;
- do not create real D0 inputs or outputs before project-owner approval.

## 2026-06-21 Tokenizer Behavior Boundary Guard Pass

Hardened the tokenizer behavior fixture reporter so the public real-artifact
boundary in `artifacts/tokenizer_behavior/README.md` must match both the
`teacher_tokenizer_behavior_measurement` admission step and the
`real_tokenizer_behavior_records` remaining-blocker row.

Updated:

- `tools/tokenizer_behavior_report.py`;
- `artifacts/tokenizer_behavior/tokenizer_fixture_report.json`;
- `artifacts/tokenizer_behavior/README.md`;
- `research/EKLAVYA_BENCHMARK_CONTRACT.md`;
- `tools/canonical_contract_status_check.py`;
- `artifacts/readiness/canonical_contract_status_check.json`.

New required status phrase:

```text
tokenizer_behavior_readme_boundary_guard_created
```

Current status:

```text
tokenizer_behavior_report: fixture_not_real_tokenizer_evidence
tokenizer_boundary: synchronized_with_admission_step_and_blocker
model_download_allowed: false
training_download_allowed: false
```

Open holes that remain:

- keep tokenizer behavior evidence absent until tokenizer loading is approved;
- do not publish accepted teacher bindings from missing tokenizer evidence.

## 2026-06-21 Teacher Runtime Boundary Guard Pass

Hardened the teacher runtime fixture reporter so the public real-artifact
boundary in `artifacts/teacher_runtime/README.md` must match both the
`G1_teacher_runtime_measurement` admission step and the
`real_teacher_runtime_rows` remaining-blocker row.

Updated:

- `tools/teacher_runtime_report.py`;
- `artifacts/teacher_runtime/runtime_fixture_report.json`;
- `artifacts/teacher_runtime/README.md`;
- `research/EKLAVYA_BENCHMARK_CONTRACT.md`;
- `tools/canonical_contract_status_check.py`;
- `artifacts/readiness/canonical_contract_status_check.json`.

New required status phrase:

```text
teacher_runtime_readme_boundary_guard_created
```

Current status:

```text
teacher_runtime_report: fixture_not_teacher_runtime_evidence
teacher_runtime_boundary: synchronized_with_admission_step_and_blocker
model_download_allowed: false
training_download_allowed: false
```

Open holes that remain:

- keep runtime observations absent until real teacher measurement is approved;
- do not use fixture runtime rows as evidence for accepted teacher bindings.

## 2026-06-21 Shape Manifest Boundary Guard Pass

Hardened the shape accounting tool so the public real-artifact boundary in
`artifacts/shape_manifest/README.md` must match the `G0_shape_accounting`
admission step and the shape-related remaining-blocker rows.

Updated:

- `tools/shape_accounting.py`;
- `artifacts/shape_manifest/parameter_report.json`;
- `artifacts/shape_manifest/README.md`;
- `research/EKLAVYA_BENCHMARK_CONTRACT.md`;
- `tools/canonical_contract_status_check.py`;
- `artifacts/readiness/canonical_contract_status_check.json`.

New required status phrase:

```text
shape_manifest_readme_boundary_guard_created
```

Current status:

```text
shape_accounting: nominal_config_count_not_G0_evidence
shape_manifest_boundary: synchronized_with_G0_admission_and_blockers
harness_allowed: false
training_allowed: false
```

Open holes that remain:

- keep real shape reports absent until D1/D2/D5 row preparation and G0 approval;
- do not use synthetic shape smoke reports as real G0 substrate evidence.

## 2026-06-21 Markdown Boundary Helper Refactor

Moved the repeated Markdown artifact-boundary extraction logic into
`tools/markdown_boundary.py` and refactored the boundary-aware checkers to use
the shared helper.

Updated:

- `tools/markdown_boundary.py`;
- `tools/artifact_index_check.py`;
- `tools/d0_frontier_execution_runbook_check.py`;
- `tools/tokenizer_behavior_report.py`;
- `tools/teacher_runtime_report.py`;
- `tools/shape_accounting.py`.

Current status:

```text
boundary_helper: shared
fixture_checks: passed
harness_allowed: false
training_allowed: false
```

Open holes that remain:

- route future README boundary checks through the shared helper instead of
  ad hoc Markdown parsing;
- keep boundary helper behavior narrow: it extracts checked artifact-path
  blocks only and does not authorize any real artifact creation.

## 2026-06-21 Tools README Pass

Added `tools/README.md` to document the default fixture-check command, the
shared Markdown boundary helper, and the no-real-execution boundary for checker
scripts.

Updated:

- `tools/README.md`;
- `research/REPO_RESET.md`.

Current status:

```text
tools_readme: created
default_command: python tools/run_fixture_checks.py
harness_allowed: false
training_allowed: false
```

Open holes that remain:

- keep tool-level documentation synchronized when new checker categories or
  shared helpers are added;
- do not let tool documentation imply that fixture checks admit real evidence
  creation.

## 2026-06-21 Tooling Docs Check Pass

Added a checked tooling-docs report so `tools/README.md` stays synchronized
with the default fixture command and the shared Markdown-boundary helper users.

Updated:

- `tools/tooling_docs_check.py`;
- `tools/README.md`;
- `tools/markdown_boundary.py`;
- `tools/run_fixture_checks.py`;
- `tools/se1_preflight_audit.py`;
- `artifacts/preflight/tooling_docs_check_report.json`;
- `artifacts/preflight/README.md`;
- `research/EKLAVYA_BENCHMARK_CONTRACT.md`;
- `tools/canonical_contract_status_check.py`;
- `artifacts/readiness/canonical_contract_status_check.json`.

New required status phrase:

```text
tooling_docs_check_created
```

Current status:

```text
tooling_docs_check: tooling_docs_check_passed
default_command: python tools/run_fixture_checks.py
harness_allowed: false
training_allowed: false
```

Open holes that remain:

- update `tools/README.md` whenever a checker starts depending on shared
  boundary parsing;
- keep tooling docs checked as part of the default fixture pipeline.

## 2026-06-21 Artifact Directory Status Guard Pass

Tightened the artifact index checker so each status published in
`artifacts/README.md` must also appear in that artifact family's own README.
This keeps the top-level artifact index from drifting away from the local
family documentation.

Updated:

- `tools/artifact_index_check.py`;
- artifact-family README files missing their indexed family status;
- `artifacts/preflight/README.md`;
- `research/EKLAVYA_BENCHMARK_CONTRACT.md`;
- `tools/canonical_contract_status_check.py`;
- `artifacts/preflight/artifact_index_check_report.json`;
- `artifacts/readiness/canonical_contract_status_check.json`.

New required status phrase:

```text
artifact_index_directory_status_guard_created
```

Current status:

```text
artifact_index_check: artifact_index_check_passed
directory_statuses_checked: indexed_family_statuses_reflected_in_family_readmes
harness_allowed: false
training_allowed: false
```

Open holes that remain:

- update both `artifacts/README.md` and the family README when an artifact
  family status changes;
- do not let directory-status text imply that fixture reports are real evidence.

## 2026-06-21 Artifact Index Blocked-Flag Guard Pass

Tightened the artifact index checker so the public default-verification section
in `artifacts/README.md` must mention every blocked flag enforced by the
preflight audit.

Updated:

- `tools/artifact_index_check.py`;
- `artifacts/README.md`;
- `artifacts/preflight/artifact_index_check_report.json`;
- `research/EKLAVYA_BENCHMARK_CONTRACT.md`;
- `tools/canonical_contract_status_check.py`;
- `artifacts/readiness/canonical_contract_status_check.json`.

New required status phrase:

```text
artifact_index_blocked_flag_guard_created
```

Current status:

```text
artifact_index_check: artifact_index_check_passed
blocked_true_flag_keys_checked: dataset/model/training/harness flags
harness_allowed: false
training_allowed: false
```

Open holes that remain:

- keep `artifacts/README.md` aligned with
  `tools/se1_preflight_audit.py::BLOCKED_TRUE_FLAG_KEYS`;
- do not remove dataset-download language from public verification docs while
  D0 remains unadmitted.
