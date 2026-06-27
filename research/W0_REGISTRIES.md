# W0 Registries — Materialized from Ground-Up Doctrine

Status: Predeclared. These registries define the G1 implementation contract.
S0/E1/E2 operate on a subset; G1 implements the full set.

Source: `research/GROUND_UP_FUTURE_DESIGN.md` (H0 definitions)

## Interface Registry

Seven typed interfaces. Each accepts structured input and emits structured
output. S0 implements I0 and I6 only (byte encoder/decoder). G1 implements all.

| ID | Name | Accepts | Emits | S0 Status |
|----|------|---------|-------|-----------|
| I0 | input_surface | raw bytes, structured records, tool outputs, retrieved memories | typed spans, byte residual map, surface risk flags, initial uncertainty | Partial (bytes only) |
| I1 | compact_state | typed spans, local byte states | semantic state, task state, memory query state, verifier query state | Not implemented |
| I2 | reasoning_step | compact state, scratch state, optional retrieved memory | candidate next state, candidate action, uncertainty delta, trace summary | Not implemented |
| I3 | verification | candidate answer/action, constraints, exact oracle results | pass/fail/unknown, error span, repair class, escalation recommendation | Not implemented |
| I4 | memory | query state, write candidate, provenance | retrieved records, confidence, conflict set, write decision | Not implemented |
| I5 | compute_governor | uncertainty, risk, task class, verification state, cost profile | stop, run_deeper, run_wider, retrieve, verify, repair, precompute | Not implemented |
| I6 | output_surface | final compact state, byte residual map, verification state | text, structured output, tool call, byte-exact span | Partial (byte logits only) |

## Lesson Type Registry

Seven lesson types, each landing in a specific interface. A proposed lesson
that does not fit this registry either reveals a missing module or is not ready.

| Lesson Type | Landing Zone | Teaches |
|-------------|-------------|---------|
| surface_fidelity | I0 + I6 | exact byte identity, symbol preservation, formatting constraints |
| semantic_invariance | I1 | paraphrase stability, abstraction, hard negative separation |
| reasoning_transition | I2 | valid next state, decomposition, plan repair |
| verification | I3 | invalidity detection, constraint localization, repair class |
| memory | I4 | retrieval key, write boundary, conflict handling |
| compute_policy | I5 | when to stop/verify/retrieve/precompute |
| update_locality | update_ports | smallest safe change, rollback condition, collateral damage guard |

### S0/E1/E2 Mapping

S0 and E1/E2 operate on a strict subset of this registry:

| E2 Signal | Maps To | Landing |
|-----------|---------|---------|
| First-byte KL | surface_fidelity | I0/I6 (ByteDecoder) |
| Token-patch alignment | surface_fidelity | I0 (ByteEncoder→GlobalReasoner boundary) |
| Semantic cosine loss | semantic_invariance | I1 (GlobalReasoner hidden states) |

No reasoning, verification, memory, compute policy, or update locality lessons
exist at the S0/E2 stage. Those require G1's full interface set.

## Teacher Measurement Function Registry

Six measurement function types. A teacher surface must declare which measurement
functions it supports. Each function produces a structured score tensor.

| Function | Score Shape | Fields |
|----------|-----------|--------|
| behavior | example x probe x candidate x field | score, rank, margin, confidence, entropy, abstain_or_invalid |
| geometry | example_pair_or_triplet x field | similarity, margin, neighborhood_rank, hard_negative_flag |
| constraint | example x constraint x field | pass_fail_unknown, violated_constraint, error_span, repair_class |
| process | example x step x field | intermediate_state, action, branch_score, termination_reason |
| curriculum | example x field | prerequisite_tags, difficulty, contrast_set_id, recommended_order |
| adversarial | (same shape as primary function) | (same fields + attack_type, perturbation_budget) |

### S0/E2 Mapping

| E2 Teacher | Measurement Functions Used |
|------------|--------------------------|
| Anchor decoder 1.7B | behavior (first-byte logits), geometry (token alignment) |
| Diversity hybrid 1.2B | behavior (first-byte logits), geometry (token alignment) |
| Control decoder 0.6B | behavior (first-byte logits), geometry (token alignment) |
| Semantic embedding 300M | geometry (embedding cosine) |
| Diversity SSM 780M | behavior (first-byte logits) |

## G1 Threshold Table

Predeclared thresholds for G1 readiness decisions. Values are provisional —
actual thresholds will be set by S0/E2 results and adversarial review.

| Threshold | Default | Purpose | Set By |
|-----------|---------|---------|--------|
| gap_admission_nll | 3.5 | Student NLL above this triggers gap selection | S0 burn-in results |
| gap_admission_entropy | 4.0 | Student entropy above this triggers gap selection | S0 burn-in results |
| teacher_jsd_low | 0.05 | JSD below this = consensus zone | E2 pilot cache stats |
| teacher_jsd_high | 0.20 | JSD above this = disagreement zone | E2 pilot cache stats |
| retained_gain_floor | 0.01 BPB | Teacher must contribute at least this much | E2.5 ablation A2 vs A0 |
| collateral_damage_ceil | 0.005 BPB | Removing teacher must not improve any class by more | E2.5 LOO runs |
| per_teacher_grad_cap | 0.10 | Fraction of CE gradient allowed per teacher | Fixed (from E2 protocol) |
| total_teacher_grad_cap | 0.30 | Total fraction of CE gradient for all teachers | Fixed (from E2 protocol) |
| ownership_confidence | 0.95 | Minimum confidence that gain is real (not noise) | Statistical testing |

## Relationship to Build Stages

```
W0 (this doc) → W1 (traceable runtime) → W2 (gap/validator suite)
     ↓                    ↓                        ↓
  Registries         State manifest           Eval slices
  define what        defines how to           define what to
  G1 MUST have       observe it               test it against
```

Engineering may begin at W1 only after W0 registries exist and match H0
definitions. S0/E1/E2 operate on the subset marked above; they do not need
the full W0 contract.
