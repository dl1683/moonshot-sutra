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
- Weighted `kl_purified` loss jumps >3x at phase entry
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

## GPU-Specific Failure Modes (CPU Tests Cannot Catch)

1. **Sparse-cache starvation**: DataLoader samples arbitrary byte windows; if (shard_id, seq_start) lookup misses most cached positions, steps silently become CE-only
2. **Python/router overhead**: Routing and purification are per-position Python/Numpy ops with mmap unpacking — monitor GPU utilization, should be >70%
3. **Real-teacher distribution pathologies**: Correlated teachers, bad top-K tail mass, or teachers with systematically wrong gold-byte likelihoods
4. **Long-run memory behavior**: Phase-transition optimizer resets + gradient budgeting backward passes may fragment GPU memory over 13,750 steps

## Ablation Priority (Limited GPU Time)

If only 4 ablations can run, execute in this order:
1. **A2** (full E2) — main system
2. **A0** (CE-only) — does E2 beat doing nothing?
3. **A1** (anchor-only) — does multi-teacher beat single?
4. **A6** (shuffled targets) — falsification: are real signals necessary?

Defer A3, A4, A5 until baseline numbers exist. Use route telemetry from A2 to decide next.
