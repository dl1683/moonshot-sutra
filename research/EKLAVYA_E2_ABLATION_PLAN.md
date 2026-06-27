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
| A5 | No router (arithmetic mean) | E2 with arithmetic mean, no router | Does routing matter? |
| A6 | Shuffled teacher targets | E2 with position-shuffled teacher records | Are real signals necessary? |

## Information Value Ranking

Run ablations in this order (highest information first):

1. **A2 vs A0**: Does E2 beat doing nothing? If no, stop everything.
2. **A2 vs A1**: Does multi-teacher beat single? Core Eklavya claim.
3. **A5 vs A2**: Does routing matter? If A5 matches A2, delete router.
4. **A6 vs A2**: Sanity check. If A6 matches A2, signals are noise.
5. **A3 vs A2**: Does the strongest non-anchor teacher contribute?
6. **A4 vs A2**: Do semantic embeddings help?

Only run A3-A6 if A2 clearly beats A1.

## Metrics

Each ablation measures on the same held-out eval shards:

| Metric | What It Measures | Decisive Margin |
|--------|-----------------|-----------------|
| BPB (bytes per byte) | Raw prediction quality | >0.02 BPB gap |
| First-byte accuracy | Top-1 byte prediction | >1pp gap |
| Per-gap-class BPB | Performance on high-NLL / high-entropy / control | >0.03 BPB gap |
| Generation quality | 50 prompted samples, human-rated coherence | Qualitative |

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

# A5: No router (arithmetic mean)
python code/eklavya_e2_training.py \
  --student-checkpoint checkpoints/e1/e1_best.pt \
  --cache-dir eklavya_e2_cache \
  --output-dir checkpoints/e2_a5_no_router \
  --ablation-id A5 --disable-router

# A6: Shuffled teacher targets
python code/eklavya_e2_training.py \
  --student-checkpoint checkpoints/e1/e1_best.pt \
  --cache-dir eklavya_e2_cache \
  --output-dir checkpoints/e2_a6_shuffled \
  --ablation-id A6 --shuffle-teacher-targets
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
| A5 BPB ≈ A2 BPB (within 0.02) | Router doesn't help | Delete router, use arithmetic mean |
| A6 BPB ≈ A2 BPB (within 0.02) | Signals are noise | Something is fundamentally wrong |
| A3 BPB ≈ A2 BPB (within 0.02) | Best diversity teacher expendable | Drop it, reduce cache cost |
| A4 BPB ≈ A2 BPB (within 0.02) | Semantic embeddings expendable | Drop embedding teacher |
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
