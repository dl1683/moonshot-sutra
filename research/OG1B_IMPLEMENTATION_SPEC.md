# OG-1b Implementation Spec

**Date:** 2026-07-03
**Status:** Implement next
**Input:** Full 3-seed OG-1 results
**Primary question:** Does relational Operational Geometry beat matched counterfactual augmentation?

## Final OG-1 Verdict

The complete 3-seed results materially strengthen the positive signal, but they still do not validate full Operational Geometry.

Updated interpretation:

1. **D vs B transforms is now the standout result.** D beats B by +9.3pp avg_trans across 3 seeds. This is large, targets the right metric, and is no longer a one-seed curiosity.
2. **D vs A MCQ is a real counterfactual-supervision signal.** D beats A by +3.8pp mean MCQ, all three seeds are positive, and D has the lowest MCQ variance. That is enough to say the counterfactual channel helps this toy task.
3. **This is still not an OG proof.** D and F are tied because invariance is dead, and the current CF loss is still relabeled counterfactual augmentation.
4. **Ranking alone is net negative.** B is below A on MCQ and transformed accuracy.
5. **C is unstable.** Dead/weak invariance plus ranking creates high transformed variance.
6. **E is a valid adversarial-label sanity check.** It proves auxiliary labels carry gradient authority.

The correct claim boundary:

```text
OG-1 proves: counterfactual supervised structure improves transformed robustness.
OG-1 does not prove: relational Operational Geometry beats matched augmentation.
OG-1b must test: D_rel > A_cf_ce and D_rel > D_aug on held-out transform relations.
```

Do not rerun patched OG-1. Preserve it as the archival baseline. Implement OG-1b as a new script.

## Implement As New File

Create:

```text
code/toy_opgeom_og1b.py
```

Do not mutate `code/toy_opgeom_og1.py` except for non-behavioral archival comments if absolutely needed. OG-1 results depend on the exact old code, including its dead invariance and no-decay schedule.

Recommended archival code policy:

1. Keep `toy_opgeom_og1.py` behavior frozen.
2. Commit `research/OG1_RESULTS_ANALYSIS.md`, this spec, and any raw OG-1 logs if available.
3. Add a top-of-file comment to OG-1 only if desired:

```python
# ARCHIVAL: reproduces OG-1 as run. Known issues are documented in
# research/OG1_RESULTS_ANALYSIS.md and research/OG1B_IMPLEMENTATION_SPEC.md.
```

No behavior changes are required before committing OG-1 as archival.

## OG-1b Core Design

Keep the same teacher/student/task for continuity:

```python
Teacher = ToyTeacher(d_model=128, n_layers=4, n_heads=4)
Student = ToyByteStudent(d_model=64, n_layers=4, n_heads=4, patch_size=4)
SEEDS = [0, 1, 2, 3, 4]
N_STEPS = 12_000
WARMUP_STEPS = 500
CHECKPOINT_STEPS = [4_000, 6_000, 8_000, 12_000]
EVAL_EXAMPLES = 2_000
ANSWER_CE_WEIGHT = 5.0
```

A 3-seed pilot is acceptable for hyperparameter selection, but the reportable OG-1b result should be 5 seeds.

## Data And Transforms

### Preserving Transforms

Keep the three OG-1 preserving transforms:

```python
T_PRESERVE = ["swap", "change_irrelevant", "rename_other"]
```

For training, sample one preserving transform per step for normal variants. For diagnostics, support `all_preserve=True` to average over all three transforms.

### Counterfactual Transforms

Implement two single-edit counterfactual transforms:

```python
T_CF = ["query_other_entity", "change_query_slot"]
```

`query_other_entity` is OG-1's existing transform:

```text
same scene, same query_attr, query_person = 1 - original_query_person
```

`change_query_slot` is new:

```text
same scene, same query_person, query_attr sampled from the two non-original attrs
```

Reject no-op counterfactuals:

```python
if cf_correct == correct:
    resample cf transform up to 10 times
    if still no-op, skip CF loss for that step
```

This matters mostly for `actn`, where both entities can share the same action.

### Held-Out Composite Counterfactuals

Do not train on these. Use them only for evaluation:

```python
T_CF_HELDOUT = [
    "query_other_then_change_slot",
    "change_slot_then_query_other",
    "preserve_then_query_other",
    "preserve_then_change_slot",
]
```

These distinguish relational transfer from memorizing single edited examples.
## Candidate Sets

Replace fixed 4-choice candidates with mode-based candidate builders.

### Same-Attribute Candidate Set

Use for ranking and ordinary MCQ:

```python
def make_same_attr_candidates(correct, query_attr, rng, shuffle=True):
    if query_attr == "colr":
        pool = COLORS
    elif query_attr == "room":
        pool = ROOMS
    else:
        pool = ACTIONS

    candidates = list(pool)  # includes correct
    assert correct in candidates
    if shuffle:
        rng.shuffle(candidates)
    gold_idx = candidates.index(correct)
    return candidates, gold_idx
```

Effective candidate counts:

```text
color: 8
room: 8
action: 4
```

Do not force 16 candidates in this vocabulary. If a later expanded toy has larger pools, set `max_candidates=16`.

### Relational Candidate Set

Use for counterfactual relationship losses:

```python
def make_relational_candidates(orig_correct, cf_correct, orig_attr, cf_attr, rng):
    candidates = []
    candidates.extend(pool_for_attr(orig_attr))
    candidates.extend(pool_for_attr(cf_attr))
    candidates = unique_preserve_order(candidates)
    assert orig_correct in candidates
    assert cf_correct in candidates
    rng.shuffle(candidates)
    return candidates, candidates.index(orig_correct), candidates.index(cf_correct)
```

This allows direct score comparison between original and counterfactual answers even when the queried slot changes from color to room or action.

## Optimization

Use a real schedule:

```python
optimizer = torch.optim.AdamW(student.parameters(), lr=LR_PEAK, weight_decay=0.01)

LR_PEAK = 1e-3
LR_MIN = 1e-4
LR_WARMUP_STEPS = 500
N_STEPS = 12_000
GRAD_CLIP = 1.0
```

Schedule:

```python
def lr_at_step(step):
    if step < LR_WARMUP_STEPS:
        return LR_PEAK * (step + 1) / LR_WARMUP_STEPS
    progress = (step - LR_WARMUP_STEPS) / max(1, N_STEPS - LR_WARMUP_STEPS)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return LR_MIN + (LR_PEAK - LR_MIN) * cosine
```

Set LR every step before `optimizer.step()`.

Checkpoint selection:

```text
Report every checkpoint.
Primary test result = best validation avg of MCQ + avg_trans + cf_direction on a fixed validation split.
Also report final 12K result.
Never silently replace final with best; show both.
```

## Loss Functions

### Base CE

Same as OG-1:

```python
L_ce = compute_ce_loss(student, tokens + [correct])
```

### Ranking Loss

Use shuffled same-attribute candidates and explicit `gold_idx`:

```python
def compute_ranking_loss_v2(student, context_tokens, candidates, gold_idx, tau):
    scores = student.score_candidates_batch(context_tokens, candidates)
    target = torch.tensor([gold_idx], dtype=torch.long, device=DEVICE)
    return F.cross_entropy((scores / tau).unsqueeze(0), target), scores
```

Default:

```python
TAU_RANK = 0.5
```

Sweep:

```text
TAU_RANK in {1.0, 0.5, 0.25}
```

Use `0.5` for the main run unless the pilot shows instability.
### Fixed Invariance Loss

The old KL-only invariance is dead. Replace it with supervised preserving rank plus margin/geometry consistency.

Helpers:

```python
EPS = 1e-6

def center(scores):
    return scores - scores.mean()

def zcenter(scores):
    c = center(scores)
    return c / (c.std(unbiased=False) + EPS)

def gold_margins(scores, gold_idx):
    gold = scores[gold_idx]
    mask = torch.ones_like(scores, dtype=torch.bool)
    mask[gold_idx] = False
    return gold - scores[mask]
```

Loss:

```python
def compute_invariance_loss_v2(
    student,
    orig_tokens,
    trans_tokens,
    candidates,
    gold_idx,
    tau_rank=0.5,
):
    L_pres_rank, scores_trans = compute_ranking_loss_v2(
        student, trans_tokens, candidates, gold_idx, tau_rank
    )
    scores_orig = student.score_candidates_batch(orig_tokens, candidates)

    L_margin = F.smooth_l1_loss(
        gold_margins(scores_orig, gold_idx),
        gold_margins(scores_trans, gold_idx),
    )
    L_vec = F.smooth_l1_loss(zcenter(scores_orig), zcenter(scores_trans))

    L_inv = 0.50 * L_pres_rank + 0.25 * L_margin + 0.25 * L_vec
    return L_inv, {
        "inv_pres_rank": L_pres_rank.detach(),
        "inv_margin": L_margin.detach(),
        "inv_vec": L_vec.detach(),
        "inv_entropy_orig": entropy_from_scores(scores_orig.detach(), tau_rank),
        "inv_entropy_trans": entropy_from_scores(scores_trans.detach(), tau_rank),
    }
```

Defaults:

```python
LAMBDA_INV = 0.20
TAU_RANK = 0.5
```

Diagnostic sweep:

```text
LAMBDA_INV in {0.10, 0.20, 0.30}
TAU_RANK in {1.0, 0.5, 0.25}
```

Invariance is considered active only if:

```text
mean grad_norm(L_inv) >= 0.05 * mean grad_norm(L_ce)
and preserve_agreement improves over B_rank by >= 2pp in pilot.
```

### Counterfactual Augmentation Loss

This is the current OG-1 style CF, extended to both CF types. It remains an augmentation-style baseline.

```python
def compute_cf_rank_loss(student, cf_tokens, cf_candidates, cf_gold_idx, tau_rank):
    return compute_ranking_loss_v2(student, cf_tokens, cf_candidates, cf_gold_idx, tau_rank)
```

Default:

```python
LAMBDA_CF_AUG = 0.25
```

### Relational Counterfactual Loss

This is the key OG-1b addition. It trains the direction of score transport under a counterfactual, not just the edited example's label.

Inputs:

```text
orig_tokens, orig_correct, orig_attr
cf_tokens, cf_correct, cf_attr
rel_candidates containing both orig_correct and cf_correct
orig_idx, cf_idx in rel_candidates
```

Loss:

```python
def compute_relational_cf_loss(
    student,
    orig_tokens,
    cf_tokens,
    cf_candidates,
    cf_gold_idx,
    rel_candidates,
    orig_idx,
    cf_idx,
    tau_rank=0.5,
    margin=0.25,
):
    L_cf_rank, _ = compute_ranking_loss_v2(
        student, cf_tokens, cf_candidates, cf_gold_idx, tau_rank
    )

    s_orig = student.score_candidates_batch(orig_tokens, rel_candidates)
    s_cf = student.score_candidates_batch(cf_tokens, rel_candidates)

    orig_pref = s_orig[orig_idx] - s_orig[cf_idx]
    cf_pref = s_cf[cf_idx] - s_cf[orig_idx]

    L_reversal = (
        F.softplus(margin - orig_pref)
        + F.softplus(margin - cf_pref)
    ) / 2

    delta_cf_correct = s_cf[cf_idx] - s_orig[cf_idx]
    delta_orig_correct = s_cf[orig_idx] - s_orig[orig_idx]
    L_delta = F.softplus(margin - (delta_cf_correct - delta_orig_correct))

    L_cf_rel = 0.50 * L_cf_rank + 0.25 * L_reversal + 0.25 * L_delta
    return L_cf_rel, {
        "cf_rank": L_cf_rank.detach(),
        "cf_reversal": L_reversal.detach(),
        "cf_delta": L_delta.detach(),
        "orig_pref": orig_pref.detach(),
        "cf_pref": cf_pref.detach(),
    }
```

Defaults:

```python
LAMBDA_CF_REL = 0.35
CF_MARGIN = 0.25
```

Sweep:

```text
CF_MARGIN in {0.10, 0.25, 0.50}
LAMBDA_CF_REL in {0.25, 0.35, 0.50}
```

Main run default:

```text
CF_MARGIN = 0.25
LAMBDA_CF_REL = 0.35
```
## Variants To Implement

Implement these exact variants.

| Variant | Loss | Purpose |
|---------|------|---------|
| A_ce | `L_ce` | Original CE baseline |
| A_more_ce | `0.5 * L_ce(orig) + 0.5 * L_ce(extra_orig)` | Matched extra-data CE control |
| A_cf_ce | `0.5 * L_ce(orig) + 0.5 * L_ce(cf)` | Matched counterfactual data augmentation baseline |
| B_rank | `L_ce + 0.25 * L_rank` | Ranking-only check |
| C_inv_fixed | `L_ce + 0.25 * L_rank + 0.20 * L_inv_v2` | Does fixed invariance help without CF? |
| D_aug_cf | `L_ce + 0.25 * L_rank + 0.25 * L_cf_rank` | Current D's active mechanism, cleaned up |
| D_rel_full | `L_ce + 0.25 * L_rank + 0.20 * L_inv_v2 + 0.35 * L_cf_rel` | Main OG-1b candidate |
| E_adv | `L_ce + wrong 0.25 * L_rank + wrong 0.35 * L_cf_rel` | Adversarial-label sanity control |
| F_rand_inv | `L_ce + 0.25 * L_rank + 0.20 * L_inv_v2(unrelated) + 0.35 * L_cf_rel` | Active random-invariance control |

Notes:

1. For `A_cf_ce`, sample one counterfactual type per step from `T_CF`, reject no-ops, and train CE on the resulting full sequence. Do not use ranking, invariance, or relational losses.
2. For `A_more_ce`, generate a second unrelated original example and average CE over original plus extra. This tests whether any gain is just more CE data per step.
3. For `D_aug_cf`, use both CF types but only `L_cf_rank`. This is the direct competitor to `D_rel_full`.
4. For `E_adv`, wrong labels should be adversarial as in OG-1: choose fake gold from wrong candidates only. Record it as adversarial, not shuffled.
5. For `F_rand_inv`, the unrelated transformed context should use the same candidate set and gold index as the original. If invariance is active, this should hurt preserving metrics.

## Training Loop Pseudocode

```python
for step in range(N_STEPS):
    set_lr(optimizer, lr_at_step(step))

    tokens, correct, _, meta = generate_binding_example(rng)
    candidates, gold_idx = make_same_attr_candidates(correct, meta["query_attr"], rng)
    L_ce = compute_ce_loss(student, tokens + [correct])

    loss = build_variant_ce_part(...)

    if variant uses rank and step >= WARMUP_STEPS:
        L_rank, rank_scores = compute_ranking_loss_v2(
            student, tokens, candidates, gold_idx, TAU_RANK
        )
        loss += LAMBDA_RANK * L_rank

    if variant uses inv and step >= WARMUP_STEPS:
        t_type = rng.choice(T_PRESERVE)
        trans_tokens = make_preserve_or_unrelated(...)
        L_inv, inv_stats = compute_invariance_loss_v2(
            student, tokens, trans_tokens, candidates, gold_idx, TAU_RANK
        )
        loss += LAMBDA_INV * L_inv

    if variant uses cf and step >= WARMUP_STEPS:
        cf_type = rng.choice(T_CF)
        cf = apply_counterfactual_transform_v2(tokens, meta, rng, cf_type, reject_noop=True)

        if variant == "D_aug_cf":
            cf_candidates, cf_gold_idx = make_same_attr_candidates(cf.correct, cf.query_attr, rng)
            L_cf_rank, _ = compute_cf_rank_loss(
                student, cf.tokens, cf_candidates, cf_gold_idx, TAU_RANK
            )
            loss += LAMBDA_CF_AUG * L_cf_rank

        if variant in ("D_rel_full", "F_rand_inv"):
            cf_candidates, cf_gold_idx = make_same_attr_candidates(cf.correct, cf.query_attr, rng)
            rel_candidates, orig_idx, cf_idx = make_relational_candidates(
                correct, cf.correct, meta["query_attr"], cf.query_attr, rng
            )
            L_cf_rel, cf_stats = compute_relational_cf_loss(...)
            loss += LAMBDA_CF_REL * L_cf_rel

    optimizer.zero_grad()
    loss.backward()
    collect_component_grad_norms_if_due(...)
    torch.nn.utils.clip_grad_norm_(student.parameters(), GRAD_CLIP)
    optimizer.step()
```

For component gradient norms, compute them every 500 steps on a fixed mini-batch, not every step, to avoid excessive cost.

## Evaluation Metrics

Use `n_examples=2_000` for final evaluation. Save per-example records to JSONL.

### Standard Metrics

1. `mcq_accuracy`
2. `teacher_agreement`
3. `bpb`
4. `clean_accuracy`
5. `swap_accuracy`
6. `irrelevant_accuracy`
7. `rename_accuracy`
8. `avg_transformed`

Fix transformed eval candidate order:

```text
Shuffle candidates for every example.
Use random tie-breaking if max scores are exactly equal.
```

### Preserving Geometry Metrics

For each preserving transform:

```python
orig_pick = argmax(scores_orig)
trans_pick = argmax(scores_trans)
gold_margin_orig = scores_orig[gold_idx] - max(scores_orig[wrong_idxs])
gold_margin_trans = scores_trans[gold_idx] - max(scores_trans[wrong_idxs])
```

Metrics:

```text
preserve_agreement = P(orig_pick == trans_pick)
preserve_gold_agreement = P(orig_pick == gold and trans_pick == gold)
preserve_margin_delta = mean(abs(gold_margin_trans - gold_margin_orig))
preserve_margin_delta_median = median(abs(...))
preserve_score_vec_l2 = mean(||zcenter(scores_trans) - zcenter(scores_orig)||_2)
```

Interpretation:

```text
higher preserve_agreement is better
higher preserve_gold_agreement is better
lower preserve_margin_delta is better
lower preserve_score_vec_l2 is better
```

### Counterfactual Metrics

For each CF type and held-out composite CF type:

```python
orig_pref = s_orig[orig_idx] - s_orig[cf_idx]
cf_pref = s_cf[cf_idx] - s_cf[orig_idx]
delta_direction = (s_cf[cf_idx] - s_orig[cf_idx]) - (s_cf[orig_idx] - s_orig[orig_idx])
```

Metrics:

```text
cf_accuracy = P(argmax(cf_scores over same-attr candidates) == cf_correct)
cf_direction_accuracy = P(orig_pref > 0 and cf_pref > 0)
cf_delta_accuracy = P(delta_direction > 0)
cf_reversal_margin = mean(min(orig_pref, cf_pref))
cf_delta_margin = mean(delta_direction)
cf_noop_rate = P(cf_correct == orig_correct)  # should be 0 after rejection
```

Primary held-out metric:

```text
heldout_cf_direction_accuracy
```

This is the main relational-geometry metric.

### Statistical Reporting

For each pairwise comparison, report:

```text
mean over seeds
std over seeds
paired bootstrap 95% CI over eval examples, per seed and pooled
per-seed deltas
```

Minimum acceptable reporting:

```text
5 seeds, 2_000 eval examples, per-seed deltas shown
```
## Success Gates

### Debug Gates

All must pass:

1. Teacher MCQ >= 95%, preferred >= 98%.
2. `A_ce` MCQ >= 50%.
3. `E_adv` MCQ <= 35%, proving adversarial labels still have force.
4. `L_inv_v2` active: gradient norm >= 5% of CE gradient after warmup.
5. `F_rand_inv` is worse than `D_rel_full` on at least one preserving-geometry metric. If F ties D again, invariance is still not validated.
6. BPB degradation for `D_rel_full` vs `A_ce` <= 5%.
7. No final-checkpoint regression > 2pp from best validation checkpoint without being reported.

### Scientific Gates

Primary OG gate:

```text
D_rel_full beats A_cf_ce by >= +2pp on heldout_cf_direction_accuracy,
and the paired bootstrap 95% CI lower bound is > 0.
```

Secondary gates:

1. `D_rel_full` beats `D_aug_cf` by >= +2pp on heldout composite CF direction accuracy.
2. `D_rel_full` beats `A_cf_ce` by >= +2pp on avg_transformed.
3. `D_rel_full` beats `A_cf_ce` by >= +2pp on preserve_agreement or reduces preserve_margin_delta by >= 10%.
4. `D_aug_cf` and/or `A_cf_ce` beat `A_ce` by >= +3pp MCQ or avg_transformed, confirming CF remains a genuine signal.
5. `D_rel_full` beats `B_rank` by >= +5pp avg_transformed.
6. `D_rel_full` has BPB degradation <= 5% vs `A_ce`.

Interpretation:

```text
If D_aug_cf and A_cf_ce improve over A_ce, CF augmentation works.
If D_rel_full beats A_cf_ce and D_aug_cf on held-out relational metrics, OG-1b supports Operational Geometry.
If D_rel_full only ties A_cf_ce, the honest result is counterfactual augmentation, not OG.
```

## Hyperparameter Sweep Plan

Do not sweep everything across all variants. Use a pilot on `C_inv_fixed`, `D_aug_cf`, and `D_rel_full` with seeds `[0, 1, 2]`.

Full pilot grid:

```text
TAU_RANK:       {1.0, 0.5, 0.25}
LAMBDA_INV:     {0.10, 0.20, 0.30}
LAMBDA_CF_REL:  {0.25, 0.35}
CF_MARGIN:      {0.10, 0.25}
```

Practical reduced pilot:

```text
Config P0: tau=0.5,  lambda_inv=0.20, lambda_cf_rel=0.35, margin=0.25
Config P1: tau=0.25, lambda_inv=0.20, lambda_cf_rel=0.35, margin=0.25
Config P2: tau=0.5,  lambda_inv=0.30, lambda_cf_rel=0.35, margin=0.25
Config P3: tau=0.5,  lambda_inv=0.20, lambda_cf_rel=0.25, margin=0.10
```

Selection rule:

```text
validation_score =
    0.40 * heldout_cf_direction_accuracy
  + 0.30 * avg_transformed
  + 0.20 * preserve_agreement
  - 0.10 * normalized_bpb_degradation
```

Then run all variants on 5 seeds with the selected config. If no pilot is possible, use P0.

## Direct Answers To The Design Questions

### Does +9.3pp D-B transforms change the assessment?

Yes. It upgrades the CF result from interesting but preliminary to real toy signal. It does not upgrade OG-1 to a validation of Operational Geometry because the effect can still be explained by counterfactual augmentation and because F remains tied to D.

### Is +3.8pp D-A MCQ with low D variance sufficient?

It is sufficient to say counterfactual supervision provides genuine signal on this toy. It is not sufficient to say the signal is OG rather than augmentation.

### Rerun fixed OG-1 or go straight to OG-1b?

Go straight to OG-1b. A fixed OG-1 would be OG-1b in all meaningful ways: LR decay, live invariance, second CF transform, matched augmentation controls, and relational losses change the experiment identity. Preserve OG-1 as the archival baseline.

### Code changes before committing OG-1 archival?

No behavior changes. Optional non-behavioral archival comment only. The important code work belongs in `toy_opgeom_og1b.py`.

## Implementation Checklist

1. Copy OG-1 to `code/toy_opgeom_og1b.py`.
2. Add transform return objects with fields: `tokens`, `correct`, `query_attr`, `query_person`, `transform_type`, `is_noop`.
3. Add `change_query_slot` and held-out composite transforms.
4. Add same-attribute and relational candidate builders with shuffled candidates and explicit gold indices.
5. Replace old invariance with `compute_invariance_loss_v2`.
6. Add `compute_relational_cf_loss`.
7. Add variants exactly as listed above.
8. Add AdamW cosine LR schedule and checkpoint evaluation.
9. Add fixed validation/test splits by seed.
10. Add per-example JSONL output for MCQ, preserving, and CF metrics.
11. Add paired bootstrap CI helper.
12. Run 3-seed pilot, choose hyperparameters by validation score, then run 5-seed main.
