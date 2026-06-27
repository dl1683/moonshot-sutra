# Eklavya/Sutra Meditations

This document keeps the question set explicit and decision-oriented. It is a
question-first record, not a replacement for the doctrine, benchmark contract, or
artifact index.

## Why this exists

`research/EKLAVYA_DOCTRINE.md` defines the target state.  
`research/EKLAVYA_BENCHMARK_CONTRACT.md` defines admissible claims and gates.  
`research/EKLAVYA_SOURCE_EVIDENCE.md` tracks supporting sources.

`EKLAVYA_MEDIATIONS.md` keeps the unresolved high-leverage questions grouped by
decision layer so the team does not jump to implementation before the gating answer
exists.

## Active Meditations

### 1) Teacher stack — PARTIALLY RESOLVED (design-level)

**Design answers (R2 deliberation, 2026-06-27):**

- Teacher profiling (T0) runs parallel to student training — cost measurement,
  surface schema testing, probe plumbing, hardware admission are all allowed
  before student gaps exist.
- Teacher lesson admission is gap-first: no lesson is claimed, no packet compiled,
  no training weight allocated until a student gap is empirically observed.
- Promotion from candidate to accepted binding requires: runtime profile within
  VRAM budget, non-redundant surface schema vs existing accepted teachers,
  measurable gap that the teacher's signal class can address.

**Still needs GPU experiments:**
- Which teacher family contributes non-redundant signal (requires T0 profiling)
- Which families are structurally redundant (requires surface correlation report)
- Whether a teacher with net-loss runtime can be admitted if retained gain is
  exceptional (requires G0 gap mapping + P0 packet trials)

## 2) Substrate design — PARTIALLY RESOLVED (R3-R4)

**Design answers (R3-R4 deliberation, 2026-06-27):**

- Patch-global with local byte path is the chosen first architecture. P4 is the
  primary S0 default; P8 is the falsifier. Comparison MUST use identical decoder.
- Byte encoder upgraded from single Conv1d to 2-layer local mixer (R4 correction).
- Student module that absorbs teacher signals: landing zones are I0 (encoder
  adapters), I3 (verifier heads), I5 (governor policy), I6 (decoder adapters).
  Global reasoner (I1/I2) gets teacher signal only through loss backprop, not
  direct adapter injection at S0.
- Adaptive patching deferred to post-S0.

**Still needs GPU experiments:**
- Whether patch-global materially improves vs byte-global at matched budget
  (requires S0 training + byte-global baseline)
- Which module absorbs teacher signals without masking cost increases elsewhere
  (requires P0 packet compilation + ablation)

## 3) Retained gain and ownership — PARTIALLY RESOLVED (design-level)

**Design answers (R1-R2 deliberation, 2026-06-27):**

- Evidence for ownership: teacher removed at inference (no logits, no packet
  lookup). Test on NEW unseen inputs not in packet training. Required: exact
  byte accuracy up, logp margin up, relevant interface heads activate correctly.
  Controls: plain CE fine-tune, naive KD on final strings, global-only update
  with local adapters frozen.
- Ownership passes only if targeted route beats controls AND ablating relevant
  local adapters removes the specific gain.
- Teacher-family removal order: remove the most expensive teacher first (highest
  VRAM / FLOPs cost). If retained gain holds, that teacher's signal was
  successfully internalized. If not, the student still depends on it.
- Minimum proof-in-principle: gain must be decisive (5-7pp over baseline per
  memory feedback_decisive_margins), not noise (≤0.1 BPT is noise per
  feedback_noise_threshold).

**Still needs GPU experiments:**
- Actual retained gain numbers (requires S0 + at least one teacher trial)
- Whether margin threshold is correct or needs calibration
- Teacher-family removal interaction effects

## 4) Efficient compute — RESOLVED (design-level)

**Resolution date**: 2026-06-27 (R1-R4 deliberation)

### Placement

- **S0 (180-250M)**: NO adaptive compute. Fixed compute path. Scout exists to
  test byte/patch tradeoffs and teacher-free competence — adaptive compute at
  this scale is noise. Clean baseline needed for measurement.
- **G1 (350-500M)**: Three mechanisms, layered:
  1. Early exit via confidence thresholding (BranchyNet-style). Exit classifiers
     every 4 layers. Gate: verifier confidence > threshold → skip remaining layers.
  2. Mixture-of-Depths layer skipping for global reasoner. Easy tokens skip deep
     layers, hard tokens use all.
  3. Self-verification as compute trigger — inverts early exit: low verifier
     confidence → allocate MORE compute (re-run full depth or invoke probe).
- **Excluded from G1**: Sparse FFN/MoE, learned routing policies, sleep/precompute.

### Hard-case quality floor

- p95 latency must be reported — if easy cases are faster but hard cases 10x
  slower, the system is worse than fixed compute.
- Hard cases = bottom-10% by verifier confidence. Quality on hard cases with
  adaptive compute must be ≥ quality with fixed full compute.
- Any compute savings claim requires ≥ 95% of fixed-compute quality on every
  evaluation slice.

### Required metric stack before compute-claim handoff

```text
p95_flops_per_token           # worst-case compute cost
mean_flops_per_token          # average compute cost
hard_case_quality_delta       # quality on bottom-10% confidence inputs
easy_case_speedup             # actual speedup on top-50% confidence inputs
total_quality_retention       # overall quality as % of fixed-compute baseline
update_locality_after_change  # does compute policy change degrade owned lessons?
```

### Remaining empirical questions (need GPU)

- Early-exit overhead (parameter count for exit classifiers)
- Optimal layer spacing for quality/compute tradeoff at 350-500M
- Whether Mixture-of-Depths helps at <500M or only at scale
- Verifier confidence threshold for easy/hard split

## 5) Source and policy — RESOLVED

**Resolution date**: 2026-06-27 (license research + TinyStories review)

### Source admissibility

| Source | License | Training? | Decision |
|---|---|---|---|
| common-pile/arxiv_abstracts | CC0 | Yes | D1 base text |
| common-pile/caselaw_access_project | Public Domain | Yes | D1 base text |
| common-pile/biodiversity_heritage_library | Public Domain | Yes | D1 base text (sample cap) |
| openai/gsm8k | MIT | Yes | D2 reasoning |
| epfl-dlab/JSONSchemaBench | MIT | Yes | D4 exact verifier |
| roneneldan/TinyStories | CDLA-Sharing-1.0 | Yes (Results exempt) | D5 guardrail |
| allenai/ai2_arc | CC-BY-SA-4.0 | Eval only | Held from training |
| databricks/databricks-dolly-15k | CC-BY-SA-3.0 | No | Held |
| allenai/scifact | CC-BY-NC-2.0 | No | Rejected (non-commercial) |

### First concrete blocker

Project owner approval of D0 source admission — not a license issue. Licenses
are clean for core shards (all Public Domain/CC0). The dependency is operational:
someone must download Common Pile shards locally and provide the input map.

### Held sources as design-only controls

All held sources are acceptable for design-only use (reading, format reference,
prompt design). License restrictions only apply when materializing training rows
or distributing adapted material.

## 6) Pre-harness question order

Current order is:

1. Source policy and D0 admissions.
2. Row manifest and row-generation gate.
3. Byte profile + G0 substrate evidence.
4. Teacher runtime/tokenization/signature stack.
5. Teacher-free baseline and harness gates.

No next phase may be claimed before all earlier outputs in this order are
admitted and the corresponding blocker rows are cleared.

## 7) Closure criteria for this file

Treat this document as done only when:

- each open meditation maps to a required artifact row in the benchmark contract;
- each mapping has an explicit completion/clear condition;
- every completion condition is visible in the blocker map or accomplishment
  package status.

For current closure checkpoints, use:

```powershell
python tools/run_fixture_checks.py
python tools/accomplishment_progress_report.py
python tools/accomplishment_work_order.py
python tools/readiness_record_check.py
```

If none of the corresponding blockers are reduced after an edit cycle, pause for
an ownership reset before any further claim-layer wording changes.

## 8) Active Closure Map (Current Frontiers)

Updated 2026-06-27 to be fully traceable to `remaining_blocker_map.json`.
All 14 blockers mapped. Status: all `blocked_real_evidence_absent`.

### Phase 1: Source/policy (D0_source_admission gate) — Med 5 RESOLVED

| Blocker ID | Required Absent Artifacts | Status |
|---|---|---|
| `real_common_pile_license_histogram_and_sample_manifest` | `common_pile_license_histogram.json`, `common_pile_sample_manifest.json` | Blocked (needs project owner D0 approval + local shard download) |
| `source_attribution_distribution_policy` | `source_attribution_distribution_policy.json` | Blocked (policy research done, artifact creation needs D0 approval) |

### Phase 2: Row preparation (G0_substrate gate)

| Blocker ID | Required Absent Artifacts | Status |
|---|---|---|
| `real_D1_D2_D5_byte_length_profiles` | `rows.jsonl`, `D2_gsm8k_main.jsonl`, `D5_tinystories_guardrail.jsonl`, `D5_dolly_instruction_guardrail.jsonl`, `real_D1_D2_D5_byte_lengths.json` | Blocked (needs Phase 1) |
| `real_D4_jsonschema_prepared_rows` | `D4_jsonschema_github_easy.jsonl` | Blocked (needs Phase 1) |

### Phase 3: Substrate evidence (G0_substrate gate)

| Blocker ID | Required Absent Artifacts | Status |
|---|---|---|
| `real_D1_byte_exactness_eval` | `eval_report.json` | Blocked (needs Phase 2 + student) |
| `real_G0_shape_accounting_report` | `g0_parameter_report.json`, `g0_sequence_report.json`, `g0_flop_report.json`, `g0_shape_accounting_report.json` | Blocked (needs Phase 2) |

### Phase 4: Teacher evidence (G1_teacher_profile gate) — Med 1 partially resolved

| Blocker ID | Required Absent Artifacts | Status |
|---|---|---|
| `real_teacher_runtime_rows` | `runtime_observations.json`, `teacher_runtime_report.json` | Blocked (needs GPU) |
| `real_tokenizer_behavior_records` | `teacher_tokenizer_behavior_records.json`, `tokenizer_behavior_report.json` | Blocked (needs model access) |
| `real_teacher_signature_tensors` | `teacher_signature_records.json` | Blocked (needs runtime + tokenizer) |
| `real_surface_correlation_report` | `teacher_surface_correlation.json` | Blocked (needs signatures) |
| `real_teacher_prompt_template_bindings` | `accepted_teacher_bindings.json` | Blocked (needs runtime + tokenizer + signatures) |

### Phase 5: Harness/ownership evidence (G2-G5 gates) — Med 3 partially resolved

| Blocker ID | Required Absent Artifacts | Status |
|---|---|---|
| `real_exact_oracle_eval_report` | `oracle_eval_report.json` | Blocked (needs Phase 2-3 + student) |
| `baseline_gap_measurements` | `baseline_gap_report.json`, `retained_gain_report.json`, `metric_log.jsonl`, `metric_summary_report.json` | Blocked (needs Phase 4 + student) |
| `real_gate_metric_evidence_report` | `gate_metric_evidence_report.json` | Blocked (needs all above) |

### Critical path

Phases 1-5 are strictly ordered. Phase 1 is the current frontier — blocked by
a single operational action (project owner D0 approval). Everything downstream
is transitively blocked by Phase 1.

Only after all 14 blockers move to clear and are reflected in the readiness record
can the claim layer move from `not_accomplished` or `not_ready_for_harness`.
