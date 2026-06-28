# E2 Monitoring Protocol — Phase-Specific Intervention Rules

Produced by Codex strategic review (R13 strategy session, June 2026).
Consult during GPU training at each phase boundary.

## Phase Boundary Intervention Triggers

### End PORT_WARMUP (step 750)

Student is frozen. Eval BPB should be unchanged from E1 baseline.

**Intervene if:**
- Eval BPB worsens by >0.02 from E1 baseline
- `align_t0_anchor_decoder` absent in >20% of log windows
- Anchor align loss has not fallen at least 10-15% from initial value
- `teacher_losses` dict is repeatedly empty

**Expected behavior:** This phase only proves anchor-port usability.

### End CONSENSUS (step 2750)

Active teachers: anchor + control.

**Intervene if:**
- `route_stats.n_routed < 4` (out of 16 sampled KL positions) for multiple consecutive log windows
- `mean_jsd` consistently near the 0.05 cutoff after filtering
- Control's average route weight stays <0.05 (anchor >0.95)
- Sum of `kl_purified_*` losses jumps >3x at phase entry
- Eval BPB regresses >0.05 from E1/port-warmup baseline across two evals

### End SEMANTIC_LANDING (step 5750)

Active teachers: anchor + control + semantic. Route stats only reflect KL-capable teachers; do NOT expect semantic weight in `avg_teacher_weights`.

**Intervene if:**
- `semantic_t3_semantic_embedding` loss absent/zero in >20% logs
- Semantic loss does not decline over first 500-1000 semantic steps
- Eval BPB worsens >0.07 from consensus boundary
- Train/eval BPB gap expands by >0.10 after full unfreeze

### During DISAGREEMENT (steps 5750-13750)

Active causal teachers: anchor, hybrid, control, SSM. Semantic remains separate.

**Intervene if:**
- `mean_route_entropy >1.30` for >500 steps (near-uniform routing = router not learning)
- `mean_route_entropy <0.20` for >500 steps (routing collapse = one teacher dominates)
- Anchor weight stays >0.85, non-anchor causal teachers combined <0.25
- `mean_jsd` remains <0.05 despite disagreement phase (teachers too similar)
- `n_routed` collapses below 4/16
- Eval BPB worsens >0.05 from semantic boundary
- Two consecutive evals show no recovery after a phase-entry spike

## Gradient Budget Monitoring

Log entries now include `grad_budget` dict with:
- `ce_grad_norm`: CE gradient magnitude (baseline)
- `total_teacher_before/after`: total teacher gradient norm before/after budget capping
- `total_scale`: scale factor applied to total teacher gradients
- `per_teacher_scales`: per-teacher scale factors

**Red flags:**
- `total_scale` consistently <0.1 = teachers being aggressively clipped, likely not contributing
- `per_teacher_scales` all near 0 = teacher gradients vastly larger than CE, possibly unstable
- `ce_grad_norm` near 0 = student not learning from CE, check freeze config

### Gradient Coherence (GCG) Telemetry

Logged every `gcg_log_interval` steps (default: 10) to avoid cloning full
gradient vectors every microstep. Fields in `grad_budget`:

- `ce_teacher_cosines`: cosine similarity between CE gradient and each teacher's
  gradient. Positive = teacher pulls in same direction as CE. Negative = teacher
  fights CE.
- `pairwise_coherence`: mean pairwise cosine similarity across all teacher pairs.
  Positive = teachers agree. Negative = teachers conflict. Near zero = orthogonal.

**Interpretation:**
- `pairwise_coherence > 0.3`: teachers largely agree — routing may not add much value
- `pairwise_coherence ~ 0`: teachers orthogonal — good, routing is doing meaningful work
- `pairwise_coherence < -0.1`: teachers actively conflict — gradient budgeting is critical
- Single teacher with negative `ce_teacher_cosines`: that teacher may be hurting training

## GPU-Specific Failure Modes (CPU Tests Cannot Catch)

1. **Sparse-cache starvation**: DataLoader samples arbitrary byte windows; if (shard_id, seq_start) lookup misses most cached positions, steps silently become CE-only
2. **Python/router overhead**: Routing and purification are per-position Python/Numpy ops with mmap unpacking — monitor GPU utilization, should be >70%
3. **Real-teacher distribution pathologies**: Correlated teachers, bad top-K tail mass, or teachers with systematically wrong gold-byte likelihoods
4. **Long-run memory behavior**: Phase-transition optimizer resets + gradient budgeting backward passes may fragment GPU memory over 13,750 steps

## Ablation-Specific Monitoring

### A7 (No Gradient Budget)

Without gradient budgeting, teacher gradients flow uncapped.
`grad_budget` in logs will show `{"enabled": false}`.

**Watch for:**
- Training instability (loss spikes, NaN) — teacher gradients may overwhelm CE
- BPB suddenly improving then crashing — overfitting to teacher signal
- If A7 matches A2: gradient budgeting is unnecessary complexity, remove it
- If A7 is clearly worse: gradient budgeting validated as load-bearing

### A8 (No Phased Admission)

All teachers active from the first student-updating phase. PORT_WARMUP
still anchor-only (port alignment needs a stable reference).

**Watch for:**
- Early training instability from conflicting teacher signals
- Route entropy near maximum from the start (no curriculum effect)
- If A8 matches A2: phased admission is unnecessary, simplify
- If A8 is clearly worse: phased admission validated

### BLD (Single-Teacher Byte KL Baseline)

No router, no purification, no alignment, no semantic, no calibration.
Just CE loss + weighted KL from anchor teacher's byte distributions.
`bld_kl_loss` and `bld_kl_bits` fields appear in logs.

**Watch for:**
- BLD should converge faster than A2 (simpler loss landscape)
- BLD's ceiling should be lower than A2 (single teacher limits richness)
- If BLD matches or beats A2: E2 machinery adds no value, fundamental problem
- If A2 clearly beats BLD: the machinery (routing, multi-teacher) is justified

### A9a/A9b/A9c (Gold-Free Router Variants)

A9 tests whether routing can work without conditioning on the gold byte.
Each variant adds more signal:
- A9a: entropy only (`--router-mode gold_free_entropy`)
- A9b: entropy + teacher agreement (`--router-mode gold_free_agreement`)
- A9c: entropy + agreement + student-teacher JSD (`--router-mode gold_free_student_jsd`)

**Watch for:**
- A9c matching A2 within 0.01 BPB = oracle routing was unnecessary, promote A9c
- A2 beating A9c by >0.02 BPB = oracle routing is material, cannot claim deployable
- A9a-b-c monotonically improving = each signal adds value, confirms design
- A9c ≈ A5 (no router) = gold-free signals too weak, routing concept unproven
- Route entropy patterns across A9 variants: more signals should reduce entropy

### A5a/A5b/A5c (Static Baseline Variants)

A5 variants provide fair static baselines (no routing). Critical for X-Token comparison.

- A5a: prior-weighted (`--static-weight-mode prior`)
- A5b: hand-tuned weights (`--static-weight-mode custom --static-weights "..."`)
- A5c: best-2 teachers, prior-weighted (`--teachers t0 t1 --static-weight-mode prior`)

**Watch for:**
- A9c must beat A5b by >0.02 BPB globally and >0.03 on high-disagreement bucket
- If A5b matches A9c: routing adds nothing over tuned static — simplify
- If A5c matches A9c: 2-teacher static is sufficient — X-Token approach wins
- Route entropy telemetry uses `-sum(w log w)` for non-uniform weights
- `bpb_high_disagreement` is the key differentiating metric for routing value

## Ablation Priority (Two-Phase Strategy)

### Phase 1: Feasibility (does multi-teacher help at all?)

1. **A2** (full E2, oracle router) — main system / oracle ceiling
2. **A0** (CE-only) — does E2 beat doing nothing?
3. **BLD** (byte KL baseline) — does E2 beat simple KL?
4. **A1** (anchor-only) — does multi-teacher beat single?

**Stop if A2 fails any Phase 1 comparison.** A2 is oracle-aided.

### Phase 2: Publishability (does deployable system beat static mixing?)

5. **A9c** (gold-free full router) — the publishable system
6. **A5b** (tuned static weights) — fairest static baseline
7. **A5a** (prior-weighted static) — does routing beat priors?
8. **A5c** (best-2 teachers, prior-weighted) — X-Token comparison
9. **A7** (no gradient budget) — novelty-critical
10. **A8** (no phased admission) — novelty-critical
11. **A6** (shuffled targets) — falsification

### 48-hour minimum: A2, A0, BLD, A1, A9c, A5b (all 8K steps). A5c if time.

Defer A3, A4 until Phase 2 baseline numbers exist.

## Automated Anomaly Detection (`monitor.py`)

The monitor automatically detects 5 anomaly types during live training:

| Anomaly | Trigger | Meaning |
|---------|---------|---------|
| Non-finite CE loss | Any step with NaN/Inf ce_loss | Model has diverged — training should have aborted via hard-fail |
| Route entropy collapse | Last 10 readings all <0.1 | Router locked to one teacher — diversity lost |
| Gradient budget suppression | Last 5 total_scale <0.01 | Teacher signal effectively zeroed — check CE grad norm |
| Zero teacher signal | >50% of steps with all-zero teacher losses | Cache coverage insufficient for this data distribution |
| No routed positions | >30% of disagreement steps with n_routed=0 | Router not activating — check JSD thresholds |

Use `python monitor.py --log logs/e2_train.jsonl --watch` for live dashboard with anomaly alerts.
