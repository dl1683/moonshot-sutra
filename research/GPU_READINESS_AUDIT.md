# GPU Readiness Audit - Eklavya E2

Date: 2026-06-28

## Verdict

Score: 7/10.

E2 is ready for a controlled GPU pilot where a human watches logs, validates cache coverage, and runs post-hoc comparisons deliberately. It is not yet ready for unattended ablation execution or for treating the full ablation matrix as automatically enforced.

The core GCG variance-decomposition change looks mathematically correct. The remaining risk is mostly operational: some documented decision rules are only partially encoded, some failure modes are warnings instead of hard stops, and the default CLI/logging path can produce misleading ablation records.

One requested source, `CLAUDE.md`, is absent from this checkout. I audited the remaining requested files plus `code/eval_e2.py` and `code/monitor.py`, because the task directly depends on the eval and monitor surfaces.

## 1. Completeness

Not all documented ablation decision rules are enforced in code.

Implemented:

- Training flag validation exists for A0, A1, A2, A3, A4, A5, A5a, A5b, A5c, A6, A7, A8, A9a, A9b, A9c, and BLD in `code/eklavya_e2_training.py`.
- `compare_ablations.py` has global BPB comparisons for the major pairs, excludes hard-failed runs from decision comparisons, detects regressions with `[REGRESS]`, and includes gap-class rules for A9c vs A5b on `bpb_high_disagreement` and `bpb_high_nll`.

Missing or partial:

- The ablation plan says A9c must beat the best of A1, BLD, and A5b by >0.02 BPB. Code checks A9c vs A5b, but not A9c vs A1 or A9c vs BLD.
- The ablation plan says A9c must improve first-byte accuracy by >1.3pp over A1. `eval_e2.py` emits `first_byte_acc`, and `compare_ablations.py` prints it, but no decision rule enforces it.
- The ablation plan says A6 must be worse than A9c by >0.02 BPB. Code checks A2 vs A6, not A9c vs A6.
- Retained-gain utility per teacher (`>0.005 BPB`) is not automated. A3/A4 approximate leave-one-out checks, but there is no general per-teacher utility calculation or gap-class proportional budget recommendation.
- Collateral damage checks are described but not implemented as a report or rule. There is no automatic flag for "removing teacher T improves another gap class."
- Phase-1 stop logic is advisory. There is no runner/orchestrator that refuses Phase 2 if A2 fails A0/A1/BLD; `compare_ablations.py` only reports results after the GPU time has already been spent.
- The documented qualitative generation-quality metric and uncommon-token BPB are not produced by `eval_e2.py` and are not consumed by `compare_ablations.py`.

## 2. Safety

Strong safety coverage:

- Cache validation is called with `deep=True` before training.
- Cache/train `seq_len` mismatch hard-fails.
- Zero cached positions in the training shard range hard-fails.
- Warmup teacher-signal coverage below `warmup_min_coverage` hard-fails.
- 200 consecutive non-warmup CE-only teacherless steps hard-fail.
- Non-finite CE loss and non-finite grad norm hard-fail and write `HARD_FAIL` entries.

Remaining safety issues:

1. Severity: high - BLD can silently become CE-only. In BLD mode, `compute_bld_kl_loss()` can return `None`. The training loop then falls through to CE-only backward, and `expects_teachers` is false for BLD, so the run does not trip the CE-only starvation guard. A broken anchor KL cache could produce a meaningless BLD baseline.

2. Severity: high - non-finite teacher losses are silently dropped. The loop filters `teacher_losses` with `if torch.isfinite(v)`. If a teacher loss becomes NaN/Inf, it disappears from the step instead of hard-failing or logging a teacher-specific corruption event. The monitor has non-finite teacher-loss detection, but it cannot catch values that were filtered before logging.

3. Severity: high - default ablation logs are easy to contaminate. The CLI exposes `--output-dir` but not `--log-file`; `E2Config.log_file` defaults to `logs/e2_train.jsonl`. The checklist launches separate ablations with separate checkpoint dirs, but the code will append all runs to the same default log unless called programmatically. `compare_ablations.py` expects one log per ablation; a mixed log can produce misleading summaries.

4. Severity: medium - stale cache/checkpoint mismatch is only a warning. The trainer warns when the cache provenance checkpoint basename does not contain the loaded checkpoint step. For a publishable ablation, stale gap positions should probably be a hard fail unless explicitly overridden.

5. Severity: medium - low density is only a warning. The trainer warns below 10 positions/shard and hard-fails only after 200 consecutive teacherless post-warmup steps. Sparse intermittent coverage can still waste hours by delivering mostly CE updates while occasionally avoiding the hard fail.

6. Severity: medium - eval split can have zero cached eval positions. The trainer warns and continues. That is acceptable for CE-only eval, but it means route/gap diagnostics may be absent exactly when ablation decisions need them.

7. Severity: medium - warm-start is optional in code. If `align_proj` is missing from the E1 checkpoint, E2 proceeds with a random anchor port. The protocol says warm-start from E1 is the path. This should be a hard fail for normal A2/A9/A5/BLD-family runs, with an explicit bypass only for tests or scratch experiments.

## 3. VRAM Budget

The documented peak of roughly 11-15 GB, or about 15.3 GB peak in the recent summary, is plausible but optimistic. I would not treat it as a tight upper bound.

The GCG change itself is sound from a memory perspective:

- The previous per-teacher raw-gradient clone set is gone.
- Pairwise coherence now keeps one running raw gradient sum.
- For a 121.7M-param student, a full fp32 gradient-size tensor is about 0.49 GB, matching the stated intermittent GCG cost.

The caveat is that the budget should count fp32 gradient bookkeeping, not only bf16 activation/runtime estimates. During a GCG microstep, the gradient budgeter can have CE grads, total capped teacher grads, an accumulated saved grad on the second accumulation microstep, live `p.grad`, and the fp32 `teacher_raw_sum`. That can approach roughly 2 GB of gradient bookkeeping before allocator fragmentation, even though the incremental GCG sum is only about 0.49 GB.

Recommendation: document a practical "abort/reduce batch" threshold at 18-19 GB reserved VRAM after warmup, and run the first pilot with `torch.cuda.max_memory_reserved()` logging at phase boundaries and during GCG steps. The RTX 5090 24 GB target still has margin, but the current doc should not imply 15.3 GB is a proven hard maximum.

## 4. Monitoring

The monitor catches the five automated anomaly types listed at the bottom of `E2_MONITORING_PROTOCOL.md`:

- non-finite CE loss
- route entropy collapse
- gradient budget near zero
- zero teacher signal
- no routed positions in disagreement

It does not automatically catch all phase-specific intervention rules in the protocol. Missing automated checks include:

- eval BPB regression thresholds at phase boundaries
- anchor align loss failing to fall 10-15 percent
- anchor/control route-weight dominance thresholds
- `mean_jsd` being too low or stuck near threshold by phase
- KL loss jumping >3x at phase entry
- semantic loss absence/zero rate and early decline
- train/eval gap expansion after unfreeze
- route entropy being too high for too long (`>1.30`)
- anchor weight staying >0.85 or non-anchor teachers combined too low
- GPU utilization below target from Python/router overhead
- long-run CUDA memory fragmentation

The telemetry exists for several of these, but "visible" is not the same as "caught." This is enough for supervised pilot monitoring, not enough for an unattended launch.

## 5. Warm-Start

The mechanism exists: `MultiTeacherProjectionPorts.warm_start_from_e1()` copies the E1 `AlignProjection` into the anchor port, and `train_e2()` invokes it when `align_proj` is present in the loaded checkpoint.

The warm-start path is tested at unit level, and the CPU end-to-end suite has resume/checkpoint/eval coverage in `test_eklavya_e2.py`. The safety gap is that the trainer does not require `align_proj` for normal E2 runs. A missing or wrong E1 checkpoint can silently turn "E1 -> E2 warm-start" into "E2 with random anchor port."

## 6. GCG Correctness

I do not see a math bug in the variance decomposition.

The implemented identity is:

```text
sum_{i<j} dot(g_i, g_j) = (||sum_i g_i||^2 - sum_i ||g_i||^2) / 2
```

The denominator is:

```text
sum_{i<j} ||g_i|| ||g_j||
```

So the reported value is a norm-weighted mean pairwise cosine. That matches the monitoring doc. It is exact for two teachers and well-defined for mismatched or disjoint parameter supports because missing gradients behave as zeros in the sum vector. The added tests for mismatched supports and partial overlap cover the right failure mode.

One naming caveat: this is not an unweighted average of pairwise cosines for three or more teachers. The docs correctly call it norm-weighted, so that is not a bug.

## 7. Eval Pipeline

For the metrics `compare_ablations.py` currently consumes, `eval_e2.py` is mostly sufficient:

- `bpb`
- `first_byte_acc`
- `bpb_high_nll`
- `bpb_high_entropy`
- `bpb_high_disagreement`
- `bpb_control`
- `n_eval_tokens`
- checkpoint step and training config provenance

However, the eval pipeline does not produce every metric documented in the ablation plan:

- no uncommon-token BPB
- no generation-quality artifact
- no retained-gain-per-teacher utility output
- no collateral-damage table
- no route entropy or uniform-JSD from frozen eval output

Some of these are available from training logs, not eval JSON. That is fine for diagnostics, but the ablation report schema should either include them or the plan should state they are log-derived and optional.

## Test Run

I attempted the focused validation suite:

```bash
python -m pytest code/test_compare_ablations.py code/test_eklavya_e2.py -q
```

The first run failed before collection because an unrelated globally installed pytest plugin attempted to write outside the workspace. With plugin autoload disabled, the suite collected and ran, but many tests using `tmp_path` errored because pytest could not create or inspect temp directories in this sandbox.

Observed signal from the isolated run:

- 405 tests passed before temp-fixture errors.
- 104 errors were temp-directory permission errors, not code assertions.
- 2 real assertion failures remain in `test_eklavya_e2.py`: old tests still expect `[FAIL]` when the updated comparison logic now emits `[REGRESS]` for an expected-better ablation that is actually worse. The dedicated `test_compare_ablations.py` coverage appears aligned with the new regression semantics, so these look like stale duplicate tests.

## Remaining Gaps

1. Severity: high - hard-fail BLD if anchor KL is absent or invalid. A BLD baseline must never silently degrade to CE-only.

2. Severity: high - hard-fail or explicitly log non-finite teacher losses before filtering. Silent teacher-loss removal can corrupt results.

3. Severity: high - expose `--log-file` or derive per-ablation logs from `--output-dir`. Current default logging can mix ablations in one JSONL file.

4. Severity: high - implement missing decisive rules: A9c vs best of A1/BLD/A5b, A9c first-byte accuracy vs A1, A6 vs A9c, retained-gain utility, and collateral-damage checks.

5. Severity: medium - make Phase 1 gating executable. A script should refuse Phase 2 unless A2 beats A0, A1, and BLD by the declared margins.

6. Severity: medium - require E1 anchor-port warm-start for normal E2 launches, with an explicit escape hatch for tests.

7. Severity: medium - convert cache/checkpoint provenance mismatch from warning to hard fail unless explicitly overridden.

8. Severity: medium - add automated monitor checks for the phase-specific intervention rules, especially eval regression, high route entropy, anchor dominance, semantic-loss absence, and JSD collapse.

9. Severity: medium - add GPU memory telemetry to logs, especially `max_memory_allocated` and `max_memory_reserved` around phase boundaries and GCG steps.

10. Severity: low - update stale duplicate decision-rule tests that expect `[FAIL]` instead of `[REGRESS]`.

11. Severity: low - either implement uncommon-token BPB and generation-quality outputs, or remove them from the required ablation metric table.

## Bottom Line

I would launch a 500-1000 step GPU smoke pilot for A2 only after fixing the high-severity safety items. I would not spend the full ablation budget until the missing decision rules and per-ablation logging are fixed, because those are exactly the places where GPU hours can produce numbers that look decisive but are not scientifically admissible.
