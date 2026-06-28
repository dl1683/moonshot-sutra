# E2.5 Ownership and Ablation Plan

Status: Predeclared. Execute after E2.4 training completes.

## Purpose

E2.5 answers one question: **did multi-teacher KD produce durable,
attributable gains, or just noise?**

## Key Design Principle

BPB is a property of frozen student weights. Ablation identity comes from
**which training config produced the checkpoint**, not from eval-time flags.
Each ablation requires a separate training run with its own configuration.
Evaluation (`eval_e2.py`) is a pure frozen-weight BPB evaluator.

## Ablation Matrix

Each ablation is a separate training run producing its own checkpoint:

| ID | Condition | Training Config | Question |
|----|-----------|----------------|----------|
| A0 | CE-only continuation | E1 best + 8K CE steps (no teacher losses) | Does E2 beat doing nothing? |
| A1 | Single-teacher (anchor) | E2 with anchor teacher only | Does multi-teacher beat single? |
| A2 | Full E2 system | E2 with all admitted teachers + full router | Full system performance |
| A3 | Leave-one-out (best non-anchor) | E2 minus strongest diversity teacher | Does best diversity teacher contribute? |
| A4 | No semantic teachers | E2 minus embedding teacher | Do embeddings help? |
| A5 | No router (uniform mean) | E2 with uniform arithmetic mean, no router | Does routing matter vs uniform? |
| A5a | No router (prior-weighted) | E2 with protocol-prior arithmetic mean | Does routing beat tuned priors? |
| A5b | No router (tuned static) | E2 with hand-tuned static weights | Does routing beat best static? |
| A5c | No router, best-2 teachers | 2 best teachers, prior-weighted, no router | Does E2 beat X-Token-style 2-teacher? |
| A6 | Shuffled teacher targets | E2 with position-shuffled teacher records | Are real signals necessary? |
| A7 | No gradient budget | E2 with gradient budget disabled | Does gradient budgeting help? |
| A8 | No phased admission | E2 with all teachers active from step 0 | Does phased admission help? |
| A9a | Gold-free entropy router | E2 with router_mode=gold_free_entropy | Does entropy-only routing work? |
| A9b | Gold-free agreement router | E2 with router_mode=gold_free_agreement | Does agreement penalty help? |
| A9c | Gold-free student JSD router | E2 with router_mode=gold_free_student_jsd | Full gold-free router |
| BLD | Single-teacher byte KL | Raw byte KL from anchor, no E2 machinery | BLD-style baseline comparison |

## Strategic Framing (Codex R18, June 2026)

**A2 is the oracle ceiling, NOT the publishable system.** A2 uses gold-byte
likelihood in the router (oracle_gold mode). A9c (gold-free) is the deployable
system. A9c must beat strong static baselines to prove routing adds value.

**A5 (uniform) is a strawman.** X-Token uses static weighting, not uniform.
A5a (prior-weighted) and A5b (tuned static) are the fair comparisons.

## Information Value Ranking (Updated June 2026)

Field survey, Codex R14, and Codex R18 competitive strategy assessment revised
the priority order. Two-phase approach: feasibility screen (Phase 1), then
publishability proof (Phase 2).

### Phase 1: Feasibility (does multi-teacher help at all?)

1. **A2 vs A0**: Does E2 beat doing nothing? If no, stop everything.
2. **A2 vs A1**: Does multi-teacher beat single? Core Eklavya claim.
3. **A2 vs BLD**: Does our machinery add value over simple byte KL?

If A2 fails any of these, stop. A2 is an oracle-aided upper bound.

### Phase 2: Publishability (does the deployable system beat static mixing?)

4. **A9c vs A5a**: Does gold-free routing beat prior-weighted static?
5. **A9c vs A5b**: Does gold-free routing beat best tuned static?
6. **A9c vs A5c**: Does 5-teacher routed beat 2-teacher static (X-Token bar)?
7. **A2 vs A9c**: Oracle gap. How much does gold-free router leave on the table?
8. **A7 vs A2**: Does gradient budgeting help? (Novelty claim.)
9. **A8 vs A2**: Does phased admission help? (Novelty claim.)
10. **A6 (short)**: Sanity check. If A6 matches A2, signals are noise.
11. **A3 vs A2**: Does the strongest non-anchor teacher contribute?
12. **A4 vs A2**: Do semantic embeddings help?

Only run Phase 2 if Phase 1 passes. Use `compare_ablations.py --phase1-gate`
to automate this check (exit 0 = proceed, exit 1 = stop).
A9a/A9b are diagnostics for A9c failure.

### Decisive Proof Thresholds

A9c must beat best of A1, BLD, and A5b by >0.02 BPB globally.
A9c must beat A5b by >0.03 BPB on high-NLL and high-disagreement buckets.
A9c must improve first-byte accuracy by >1.3pp over A1 (X-Token's bar).
A6 must be worse than A9c by >0.02 BPB.

## Metrics

Each ablation measures on the same held-out eval shards:

| Metric | What It Measures | Decisive Margin |
|--------|-----------------|-----------------|
| BPB (bytes per byte) | Raw prediction quality | >0.02 BPB gap |
| First-byte accuracy | Top-1 byte prediction | >1pp gap |
| Per-gap-class BPB | Performance on high-NLL / high-entropy / control | >0.03 BPB gap |
| Generation quality | 50 prompted samples, human-rated coherence | Qualitative (post-GPU) |
| Uncommon-token BPB | Performance on rare/numeric/Unicode bytes | >0.03 BPB gap (post-GPU) |
| Route entropy | Teacher selection diversity (E2 runs only) | Diagnostic |
| Uniform-JSD | Teacher disagreement independent of routing | Diagnostic |
| Retained gain per teacher | Per-teacher contribution to final BPB | >0.005 BPB |

## Training Commands

Each ablation config is passed to `eklavya_e2_training.py`:

```bash
# A0: CE-only continuation (no teacher losses)
python code/eklavya_e2_training.py \
  --student-checkpoint checkpoints/e1/e1_best.pt \
  --cache-dir eklavya_e2_cache \
  --output-dir checkpoints/e2_a0_ce_only \
  --ablation-id A0 --ce-only --steps 8000

# A1: Single-teacher (anchor only)
python code/eklavya_e2_training.py \
  --student-checkpoint checkpoints/e1/e1_best.pt \
  --cache-dir eklavya_e2_cache \
  --output-dir checkpoints/e2_a1_anchor_only \
  --ablation-id A1 --teachers t0_anchor_decoder

# A2: Full E2 system (baseline)
python code/eklavya_e2_training.py \
  --student-checkpoint checkpoints/e1/e1_best.pt \
  --cache-dir eklavya_e2_cache \
  --output-dir checkpoints/e2_a2_full \
  --ablation-id A2

# A5: No router (uniform arithmetic mean)
python code/eklavya_e2_training.py \
  --student-checkpoint checkpoints/e1/e1_best.pt \
  --cache-dir eklavya_e2_cache \
  --output-dir checkpoints/e2_a5_no_router \
  --ablation-id A5 --disable-router --static-weight-mode uniform

# A5a: No router (protocol prior weights — fair static baseline)
python code/eklavya_e2_training.py \
  --student-checkpoint checkpoints/e1/e1_best.pt \
  --cache-dir eklavya_e2_cache \
  --output-dir checkpoints/e2_a5a_prior \
  --ablation-id A5a --disable-router --static-weight-mode prior

# A5b: No router (tuned static weights — strongest static baseline)
# NOTE: t3_semantic_embedding has no KL, so only KL teachers get weights.
# Tune final weights from A5a validation telemetry. Starting point below.
python code/eklavya_e2_training.py \
  --student-checkpoint checkpoints/e1/e1_best.pt \
  --cache-dir eklavya_e2_cache \
  --output-dir checkpoints/e2_a5b_tuned \
  --ablation-id A5b --disable-router --static-weight-mode custom \
  --static-weights "t0_anchor_decoder:0.45,t1_diversity_hybrid:0.25,t2_control_decoder:0.15,t4_diversity_ssm:0.15"

# A5c: Best-2 teachers, prior-weighted, no router (X-Token 2-teacher comparison)
# Default: anchor + hybrid (most architecturally diverse pair).
# Override if cache scores show t0+t4 has better complementarity.
python code/eklavya_e2_training.py \
  --student-checkpoint checkpoints/e1/e1_best.pt \
  --cache-dir eklavya_e2_cache \
  --output-dir checkpoints/e2_a5c_best2 \
  --ablation-id A5c --disable-router --static-weight-mode prior \
  --teachers t0_anchor_decoder t1_diversity_hybrid

# A6: Shuffled teacher targets
python code/eklavya_e2_training.py \
  --student-checkpoint checkpoints/e1/e1_best.pt \
  --cache-dir eklavya_e2_cache \
  --output-dir checkpoints/e2_a6_shuffled \
  --ablation-id A6 --shuffle-teacher-targets

# A7: No gradient budget (uncapped teacher gradients)
python code/eklavya_e2_training.py \
  --student-checkpoint checkpoints/e1/e1_best.pt \
  --cache-dir eklavya_e2_cache \
  --output-dir checkpoints/e2_a7_no_grad_budget \
  --ablation-id A7 --disable-gradient-budget

# A8: No phased admission (all teachers from step 0)
python code/eklavya_e2_training.py \
  --student-checkpoint checkpoints/e1/e1_best.pt \
  --cache-dir eklavya_e2_cache \
  --output-dir checkpoints/e2_a8_no_phased \
  --ablation-id A8 --no-phased-admission

# A9a: Gold-free entropy-only router
python code/eklavya_e2_training.py \
  --student-checkpoint checkpoints/e1/e1_best.pt \
  --cache-dir eklavya_e2_cache \
  --output-dir checkpoints/e2_a9a_gf_entropy \
  --ablation-id A9a --router-mode gold_free_entropy

# A9b: Gold-free entropy + agreement router
python code/eklavya_e2_training.py \
  --student-checkpoint checkpoints/e1/e1_best.pt \
  --cache-dir eklavya_e2_cache \
  --output-dir checkpoints/e2_a9b_gf_agreement \
  --ablation-id A9b --router-mode gold_free_agreement

# A9c: Gold-free full router (entropy + agreement + student JSD)
python code/eklavya_e2_training.py \
  --student-checkpoint checkpoints/e1/e1_best.pt \
  --cache-dir eklavya_e2_cache \
  --output-dir checkpoints/e2_a9c_gf_student_jsd \
  --ablation-id A9c --router-mode gold_free_student_jsd

# BLD: Single-teacher byte KL baseline (no E2 machinery)
python code/eklavya_e2_training.py \
  --student-checkpoint checkpoints/e1/e1_best.pt \
  --cache-dir eklavya_e2_cache \
  --output-dir checkpoints/e2_bld_baseline \
  --ablation-id BLD --bld-mode --steps 8000
```

## Evaluation Commands

After each training run produces a checkpoint, evaluate with `eval_e2.py`:

```bash
# Evaluate any ablation checkpoint
python code/eval_e2.py \
  --checkpoint checkpoints/e2_a2_full/e2_best.pt \
  --eval-shards data/shards_bytes_full \
  --ablation-id A2 \
  --run-label full_e2 \
  --output ablations/a2_full.json
```

## Report Schema

Each evaluation produces a JSON file:

```json
{
  "ablation_id": "A2",
  "run_label": "full_e2",
  "checkpoint": "checkpoints/e2_a2_full/e2_best.pt",
  "step": 13750,
  "checkpoint_phase": "DISAGREEMENT",
  "training_config": {},
  "metrics": {
    "bpb": 1.05,
    "first_byte_acc": 0.42,
    "bpb_high_nll": 1.82,
    "bpb_high_entropy": 1.45,
    "bpb_control": 0.78,
    "n_eval_tokens": 50000
  }
}
```

## Decision Rules

| Comparison | If True | Action |
|-----------|---------|--------|
| A2 BPB ≤ A0 BPB | E2 doesn't help | Abandon multi-teacher KD |
| A2 BPB ≤ A1 BPB | Multi-teacher ≤ single | Drop to single-teacher (E1 sufficient) |
| A2 BPB ≤ BLD BPB | E2 machinery adds no value | Simplify to raw byte KL |
| A5 BPB ≈ A2 BPB (within 0.02) | Uniform mixing matches oracle router | Router effect is trivial |
| A5a BPB ≈ A2 BPB (within 0.02) | Prior-weighted matches oracle router | Router adds nothing over priors |
| A5b BPB ≈ A9c BPB (within 0.02) | Tuned static matches gold-free router | Routing not proven vs best static |
| A5c BPB ≈ A9c BPB (within 0.02) | 2-teacher static matches 5-teacher routed | X-Token approach sufficient |
| A7 BPB ≈ A2 BPB (within 0.02) | Gradient budget doesn't help | Remove gradient budgeting |
| A8 BPB ≈ A2 BPB (within 0.02) | Phased admission doesn't help | Remove phased admission |
| A6 BPB ≈ A2 BPB (within 0.02) | Signals are noise | Something is fundamentally wrong |
| A3 BPB ≈ A2 BPB (within 0.02) | Best diversity teacher expendable | Drop it, reduce cache cost |
| A4 BPB ≈ A2 BPB (within 0.02) | Semantic embeddings expendable | Drop embedding teacher |
| A9c within 0.01 BPB of A2 and beats A5 by >0.02 | Gold-free router works | Promote A9c, demote A2 to oracle ceiling |
| A2 beats A9c by >0.02 BPB | Oracle-aided routing material | Router was oracle-dependent, don't claim deployable |
| A9c ≈ A5 within 0.02 BPB | Routing concept unproven | Use arithmetic/prior mixture |
| A2 BPB < A0 by >0.02, A2 < A1 by >0.01 | Multi-teacher wins | Eklavya validated |

## Retained-Gain Per Teacher

After A2 vs A0 confirms E2 works:

```
contribution_t = BPB(A2) - BPB(E2_minus_t)
utility_t = contribution_t / (cache_cost_t + training_cost_delta_t)
```

Teachers with utility_t ≤ 0 across all gap classes are permanently dropped.
Teachers with utility_t > 0 only on specific gap classes keep their budget
proportional to that class's frequency.

## Collateral Damage Check

For each leave-one-out run (A3, and any per-teacher LOO runs):

- Does removing teacher T *improve* performance on any gap class?
- If yes, that teacher's signal interferes with another's domain.
- Reduce its weight or restrict its gap-class scope.

## Cost Note

Each ablation is a separate training run (12K+ steps). Priority order
minimizes wasted compute: if A2 vs A0 fails, skip all others. If A2 vs A1
fails, skip A3-A6.
