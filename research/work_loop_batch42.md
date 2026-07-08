# W-Loop Batch 42: Eklavya Fresh-Start CPU Audit

**Date:** 2026-07-08
**Mode:** CPU-only. No GPU experiments run.
**Role:** Work loop. Do the work, keep the claim killable.

## Executive Verdict

Eklavya E2 is not ready for a full GPU ablation suite in this checkout.

It is close enough in core Python wiring for a supervised GPU runtime smoke after CPU-closeable gaps are fixed, but it is not scientifically ready to spend the full E2 ablation budget. The live repository has a hard mismatch between the launch/audit documents and the actual file surface: `compare_ablations.py`, `test_compare_ablations.py`, `benchmark_harness.py`, `test_benchmark_harness.py`, `burnin_verdict.py`, `test_burnin_verdict.py`, `test_vram_profile.py`, and `research/E2_MONITORING_PROTOCOL.md` are referenced by docs or tests but are absent from this checkout. More importantly, `code/test_eklavya_e2.py` imports `compare_ablations` at module scope, so the claimed E2 test suite cannot collect.

The minimum next GPU experiment is not the full ablation matrix. It is an oracle-ceiling feasibility scout:

```text
A0  CE-only continuation
A1  anchor-only E2
BLD raw anchor byte-KL baseline
A2  full oracle-routed E2
```

Run those first on the same small cache, same seed discipline, same held-out eval, and stop unless A2 beats A0, A1, and BLD by a predeclared margin. A2 is an oracle-aided upper bound. If it cannot beat anchor-only and raw byte-KL, E2 has no student-learnable multi-teacher residual and Kill #9 remains closed.

Overall Batch 42 terminal token:

```text
B42_KILL_FULL_GPU_ABLATION_NOT_READY_CONFIRM_MINIMAL_SIGNAL_SCOUT_SPEC_READY
```

Secondary token:

```text
B42_VOID_TEMPFILE_HEAVY_TEST_VERDICT_IN_THIS_SANDBOX
```

Reason: Python file writes to sandbox temp directories failed with `PermissionError`, while PowerShell file writes worked. The tempfile-heavy tests therefore do not provide a clean code verdict here.

## Work Register

Files read or inspected:

- `research/VISION.md`
- `research/STATUS.md`
- `research/EKLAVYA_DOCTRINE.md`
- `research/EKLAVYA_E1_PROTOCOL.md`
- `research/EKLAVYA_E2_PROTOCOL.md`
- `research/EKLAVYA_E2_ABLATION_PLAN.md`
- `research/EKLAVYA_E2_TEACHER_FEASIBILITY.md`
- `research/GPU_READINESS_AUDIT.md`
- `research/GPU_LAUNCH_CHECKLIST.md`
- `research/METHODOLOGY_TEMPLATE.md`
- `research/work_loop_batch14.md`
- `research/dual_loop_supervisor_checkin_13.md`
- `README.md`
- `experiments/EXPERIMENTS.md`
- `experiments/ledger.jsonl`
- `code/s0_architecture.py`
- `code/s0_configs.py`
- `code/eklavya_training.py`
- `code/eklavya_e2_training.py`
- `code/eklavya_e2_router.py`
- `code/eklavya_e2_losses.py`
- `code/eklavya_e2_cache.py`
- `code/eval_e2.py`
- `code/preflight.py`
- `code/monitor.py`
- live test inventory under `code/test_*.py`

CPU gates run:

```text
git status --short
rg --files
python -m compileall -q code
python code/s0_configs.py
python -c "import ... s0_architecture, eklavya_training, eklavya_e2_* ..."
python -m pytest code/test_eklavya.py code/test_eklavya_e2.py -q
python -m pytest code/test_utilities.py code/test_monitor_inspect.py code/test_export_log_csv.py -q
python -m pytest code/test_option_c.py code/test_overfit.py -q
```

Key observed gate results:

- `compileall`: pass.
- import smoke for the core modules: pass.
- `s0_configs.py`: confirms live parameter counts: P4 121.7M, P8 124.1M, D640 145.3M, D768 156.2M, Wide7 121.7M.
- `test_eklavya_e2.py`: collection fails because `compare_ablations.py` is missing.
- tempfile-heavy tests: blocked by sandbox `PermissionError` on Python-created temp files, so they are not interpretable as code failures in this run.

## Iteration 1 - Re-ground In The Live Canon

**Attack target:** The prompt assumes Eklavya is a live GPU-ready direction.

**Precommitted tokens:**

```text
B42_I1_CONFIRM_EKLAVYA_REOPENED_IF_LIVE_CANON_SUPPORTS_IT
B42_I1_KILL_EKLAVYA_AS_CURRENT_DOCTRINE_IF_STATUS_SAYS_HISTORICAL
B42_I1_VOID_IF_STATUS_AND_PROMPT_CANNOT_BE_RECONCILED
```

**Gate chain:** read `VISION.md`, `STATUS.md`, and `README.md`; compare prompt against live status.

**Result:** `B42_I1_KILL_EKLAVYA_AS_CURRENT_DOCTRINE_IF_STATUS_SAYS_HISTORICAL`.

The live canon says the project is no longer defined by Eklavya as multi-teacher KD, and `STATUS.md` records Eklavya routing as Kill #9. This does not forbid a fresh-start audit, but it means E2 must be treated as a revival hypothesis, not active doctrine.

**Attack on conclusion:** Maybe `STATUS.md` is stale and the E2 docs/code were created after Kill #9 to fix the problem. Continue into the E2 design.

## Iteration 2 - Understand Kill #9 Before Trusting E2

**Attack target:** "The old routing mechanism failed, but E2 fixes routing."

**Precommitted tokens:**

```text
B42_I2_CONFIRM_E2_IF_KILL9_ONLY_ATTACKED_AN_OBSOLETE_IMPLEMENTATION
B42_I2_KILL_ROUTING_REVIVAL_IF_ORACLE_CEILING_FAILED
B42_I2_VOID_IF_KILL9_EVIDENCE_IS_TOO_NOISY_TO_USE
```

**Gate chain:** read `research/work_loop_batch14.md` and `research/dual_loop_supervisor_checkin_13.md`; identify whether the failure was learned-router-only or oracle-level.

**Result:** `B42_I2_KILL_ROUTING_REVIVAL_IF_ORACLE_CEILING_FAILED`.

Kill #9 was not just bad router. It reported that single-teacher KD beat multi-teacher routing on aggregate, oracle routing underperformed single-teacher KD, random routing beat learned routing, and the teacher oracle gap did not become student accuracy. E2's first burden is therefore not to show a more elegant router. It must show that a student can learn a residual from multi-teacher targets beyond the best single teacher.

**Attack on conclusion:** E2 changes more than the router: per-teacher ports, gradient budgeting, heterogeneous teachers, semantic loss, cache validation, and source identity. Maybe the old oracle failure does not transfer.

## Iteration 3 - Read E1/E2 Protocols As Claims

**Attack target:** "E2 is still just routing."

**Precommitted tokens:**

```text
B42_I3_CONFIRM_E2_HAS_A_DISTINCT_TESTABLE_MECHANISM
B42_I3_KILL_E2_IF_IT_IS_ONLY_OLD_ROUTING_WITH_MORE_NAMES
B42_I3_VOID_IF_PROTOCOL_HAS_NO_NUMERIC_GATES
```

**Gate chain:** read E1 protocol, E2 protocol, and E2 ablation plan; extract admission gates and decision rules.

**Result:** `B42_I3_CONFIRM_E2_HAS_A_DISTINCT_TESTABLE_MECHANISM`.

E2 is a real redesign: per-teacher projection ports, preserved teacher identity, binary cache with shared position manifest, oracle and gold-free router modes, arithmetic/log-pool/route purification, per-teacher and total gradient budgets, phased teacher admission, and CE-only/anchor-only/BLD/shuffled/static/gold-free controls. But the decisive claim still lives on the same axis as Kill #9: A2 must beat A1 and BLD. If it does not, there is no Eklavya-specific residual.

**Attack on conclusion:** A protocol can look good and still be absent or unrunnable. Audit the live code.

## Iteration 4 - Audit The Live Code Surface

**Attack target:** "The codebase is code-complete."

**Precommitted tokens:**

```text
B42_I4_CONFIRM_CODE_SURFACE_MATCHES_PROTOCOL
B42_I4_KILL_CODE_COMPLETE_CLAIM_IF_REFERENCED_GATE_FILES_ARE_MISSING
B42_I4_VOID_IF_FILE_SURFACE_IS_TOO_DIRTY_TO_AUDIT
```

**Gate chain:** run `rg --files`; read core E1/E2 code; search for referenced launch/gate files; compile all Python; import core modules.

**Result:** `B42_I4_KILL_CODE_COMPLETE_CLAIM_IF_REFERENCED_GATE_FILES_ARE_MISSING`.

Core modules compile and import. The student architecture, E1 trainer, E2 cache, E2 router, E2 losses, E2 trainer, evaluator, preflight, and monitor are present. The E2 trainer has real ablation flag validation and per-ablation log defaults. But the launch/checklist/test surface references missing files, especially `code/compare_ablations.py`. That missing comparator prevents `test_eklavya_e2.py` from collecting.

**Attack on conclusion:** Some audit gaps may have been fixed inside `eklavya_e2_training.py`; do not over-penalize stale docs.

## Iteration 5 - Re-score GPU Readiness Against Live Code

**Attack target:** "The old 7/10 readiness audit still holds."

**Precommitted tokens:**

```text
B42_I5_CONFIRM_7_OF_10_IF_HIGH_SEVERITY_AUDIT_GAPS_ARE_FIXED
B42_I5_KILL_7_OF_10_IF_LIVE_GATE_CHAIN_CANNOT_RUN
B42_I5_VOID_IF_SANDBOX_PREVENTS_ALL_READINESS_TESTING
```

**Gate chain:** read `GPU_READINESS_AUDIT.md`; compare high-severity items against current `eklavya_e2_training.py`; run compile/import; run available tests where possible.

**Result:** `B42_I5_KILL_7_OF_10_IF_LIVE_GATE_CHAIN_CANNOT_RUN`.

Some old audit concerns are improved: BLD no-teacher starvation now has a hard fail, non-finite teacher losses are logged before filtering, `--log-file` exists, default logs are per ablation, GPU memory telemetry exists, `eval_e2.py` emits byte-category first-byte BPB, and monitor has phase-boundary checks.

But the checkout cannot be scored as 7/10 because the decision gate chain is not executable. Current practical readiness:

```text
core wiring for supervised smoke: 6/10
scientific readiness for full ablation suite: 3/10
overall GPU readiness in this checkout: 5/10
```

**Attack on conclusion:** Test failures may be sandbox-only. Separate sandbox limits from real code gaps.

## Iteration 6 - CPU Gate Results And Sandbox Limits

**Attack target:** "Tests fail, therefore code is bad."

**Precommitted tokens:**

```text
B42_I6_CONFIRM_CPU_TESTS_PASS_IF_TEMP_WRITES_WORK
B42_I6_KILL_E2_TEST_SURFACE_IF_COLLECTION_FAILS_ON_MISSING_MODULE
B42_I6_VOID_TEMPFILE_RESULTS_IF_SANDBOX_BLOCKS_PYTHON_WRITES
```

**Gate chain:** run compileall, import smoke, E1/E2 tests, utility/monitor/export tests, and Python/PowerShell write probes.

**Result:** mixed:

```text
B42_I6_KILL_E2_TEST_SURFACE_IF_COLLECTION_FAILS_ON_MISSING_MODULE
B42_I6_VOID_TEMPFILE_RESULTS_IF_SANDBOX_BLOCKS_PYTHON_WRITES
```

The real code-surface failure is the missing comparator module. The tempfile failures are not clean code evidence in this sandbox: Python could create temp directories but then failed to write files inside them, while PowerShell could write a file in the workspace.

**Attack on conclusion:** If the missing comparator is the main hard gap, the minimal CPU closure plan may be short.

## Iteration 7 - Define CPU-Closable Gaps

**Attack target:** "We need GPU to make progress."

**Precommitted tokens:**

```text
B42_I7_CONFIRM_CPU_CAN_CLOSE_SCIENTIFIC_GATE_GAPS
B42_I7_KILL_CPU_CLOSURE_IF_ONLY_GPU_CAN_VALIDATE
B42_I7_VOID_IF_GAPS_REQUIRE_PRIVATE_TEACHER_ACCESS
```

**Gate chain:** partition gaps into code-surface, docs-surface, teacher/runtime, and empirical-signal categories.

**Result:** `B42_I7_CONFIRM_CPU_CAN_CLOSE_SCIENTIFIC_GATE_GAPS`.

CPU-closable before GPU:

1. Restore or implement `code/compare_ablations.py`.
2. Add/restore `code/test_compare_ablations.py`, or move the duplicate compare tests out of `test_eklavya_e2.py` cleanly.
3. Make `preflight.py` reference only existing tests, or restore the missing tests.
4. Add an executable Phase-1 gate that refuses Phase 2 unless A2 beats A0, A1, and BLD by predeclared margins.
5. Require E1 `align_proj` for normal E2 launches, with explicit bypass only for scratch tests.
6. Make strict provenance the default for real E2 launches.
7. Add a fresh E2 precommit spec that says A2 is oracle ceiling, not deployable evidence.
8. Align `GPU_LAUNCH_CHECKLIST.md` with the live roster and current files.
9. Produce a teacher feasibility JSON schema and require it before cache build.
10. Add a CPU synthetic E2 cache/trainer smoke once Python file writes are available outside this sandbox.

Not CPU-closable: teacher model load sanity, VRAM, throughput, real E2 cache build, real retained-gain measurement, and benchmark movement.

**Attack on conclusion:** Closing gates is not the same as having a meaningful experiment. Design the cheapest falsifier.

## Iteration 8 - Minimal Experiment For Any E2 Signal

**Attack target:** "Run the 16-ablation matrix."

**Precommitted tokens:**

```text
B42_I8_CONFIRM_MINIMAL_SCOUT_IF_A2_ORACLE_CAN_BOUND_SIGNAL
B42_I8_KILL_FULL_ABLATION_IF_PHASE1_HAS_NOT_PASSED
B42_I8_VOID_IF_METRIC_ONLY_MEASURES_PROXY_NOT_FUNCTION
```

**Gate chain:** use Kill #9 as the primary adversary; use E2 ablation plan but minimize it; require comparisons that distinguish E2 from CE continuation, anchor-only KD, and raw BLD.

**Result:** `B42_I8_CONFIRM_MINIMAL_SCOUT_IF_A2_ORACLE_CAN_BOUND_SIGNAL`.

Cheapest decisive GPU plan after CPU gaps close:

```text
Stage 0: runtime smoke, no claim
  A2 only, 200-500 microsteps, 5-10 shards
  pass only if no crashes, nonzero teacher coverage, finite losses, route stats

Stage 1: signal scout, claim-bearing
  A0, A1, BLD, A2
  same E1 checkpoint, same cache, same shard split, same step budget
  1000-2000 optimizer updates if possible, or 8000 microsteps if noise is high

Stage 2: noise falsifier only if Stage 1 passes
  A6 shuffled targets, shorter but same data/cache
```

Predeclared Stage 1 continuation gate:

```text
continue iff:
  BPB(A2) < BPB(A0) by at least 0.02
  BPB(A2) < BPB(A1) by at least 0.02
  BPB(A2) < BPB(BLD) by at least 0.02
  high_disagreement BPB(A2) < best(A1, BLD) by at least 0.03
  first_byte_acc(A2) > max(A1, BLD) by at least 1.0 percentage point
  no hard-case or byte-category regression larger than 0.03 BPB
```

If A2 fails any comparison, stop. Because A2 is oracle-aided, failure means the upper bound is below the boring baselines.

**Attack on conclusion:** This still measures byte prediction. Maybe even a positive result would not prove Eklavya.

## Iteration 9 - Deeper Attack: Is The Fundamental Question Wrong?

**Attack target:** "If A2 beats baselines, Eklavya is back."

**Precommitted tokens:**

```text
B42_I9_CONFIRM_E2_IS_MEANINGFUL_ONLY_AS_LEARNABLE_RESIDUAL_TEST
B42_I9_KILL_MOONSHOT_CLAIM_IF_E2_ONLY_IMPROVES_BYTE_KL
B42_I9_VOID_IF_TEACHER_OUTPUTS_ARE_USED_AS_FUNCTION
```

**Gate chain:** compare doctrine's invariant-transfer claim against implemented losses; ask whether teacher disagreement is structure or label noise; ask whether the student has landing zones for real invariants.

**Result:** `B42_I9_CONFIRM_E2_IS_MEANINGFUL_ONLY_AS_LEARNABLE_RESIDUAL_TEST`.

The deeper issue is not only routing. KD mostly transfers output distributions. Multi-teacher disagreement can be true complementary structure, teacher calibration noise, tokenizer artifact, architecture artifact, style prior conflict, a target too hard for the student, or a teacher-level oracle gap that does not map into student weights.

E2 improves the old design by preserving teacher identity and capping gradient damage, but its central train signal is still mostly byte KL plus projection alignment. That can test whether multi-teacher byte-native targets have a student-learnable residual. It cannot, by itself, prove the larger doctrine: teachers as sensors for hidden task structure and student-owned invariants after teacher removal.

Strict positive claim ceiling:

```text
E2 shows a learnable multi-teacher byte-prediction residual under this cache, student, teacher roster, and budget.
```

It does not yet show democratized intelligence, genuine reasoning, surgical improvability, or a paradigm shift.

**Attack on conclusion:** If the claim ceiling is this low, is the GPU scout worth doing at all?

## Iteration 10 - Final Adversarial Decision

**Attack target:** "Do not spend compute on methodology or nostalgia."

**Precommitted tokens:**

```text
B42_I10_CONFIRM_GPU_SCOUT_WORTH_IT_IF_IT_CAN_DECISIVELY_CLOSE_E2
B42_I10_KILL_GPU_SCOUT_IF_ONLY_A_NEGATIVE_RESULT_PAPER_RESULTS
B42_I10_VOID_IF_NO_HOSTILE_REVIEWER_WOULD_CARE
```

**Gate chain:** apply the mission test from `VISION.md`; apply Kill #9 as adversary; apply methodology token discipline; decide whether to prepare GPU or stop.

**Result:** `B42_I10_CONFIRM_GPU_SCOUT_WORTH_IT_IF_IT_CAN_DECISIVELY_CLOSE_E2`.

The scout is worth doing only because it is cheap relative to ambiguity and can close the E2 revival cleanly. It is not worth the full ablation matrix until the oracle upper bound passes Phase 1.

If A2 beats A0/A1/BLD under strict gates, E2 earns Phase 2 publishability tests against gold-free routing and static mixing. If A2 fails, E2 is killed again, more cleanly than Kill #9 because the redesigned code got its fair upper-bound test.

Final continuation posture:

```text
CPU first:
  fix comparator/gate/preflight/docs/provenance gaps

GPU next:
  runtime smoke only

claim-bearing GPU:
  A0/A1/BLD/A2 oracle-ceiling scout

full ablations:
  only after A2 Phase 1 pass
```

## Concrete Readiness Assessment

### What Exists

- S0 architecture is implemented and imports.
- E1 trainer is implemented.
- E2 cache formats and `E2CacheView` exist.
- E2 router supports oracle and gold-free modes.
- E2 losses and gradient budgeting exist.
- E2 trainer validates ablation flags.
- E2 trainer logs per-ablation JSONL by default.
- E2 evaluator emits BPB, first-byte accuracy, gap-class metrics, and byte-category first-byte BPB.
- Monitor contains E1/E2 anomaly checks.
- Full 565-shard byte data appears present locally.

### What Is Not Ready

- E2 tests cannot collect because `compare_ablations.py` is missing.
- The executable Phase-1/Phase-2 comparator is absent.
- Preflight references missing files.
- Launch checklist references missing files and stale test counts.
- The live docs are inconsistent about teacher roster and missing monitoring artifacts.
- E2 normal launch does not hard-require E1 `align_proj`.
- Strict cache/checkpoint provenance is optional rather than the default.
- Teacher feasibility has a runbook but no local results artifact.
- No CPU synthetic writer/trainer smoke could be verified in this sandbox because Python file writes are blocked.
- No GPU runtime, VRAM, or throughput has been measured.

## Minimal E2 Precommit Spec For The Next GPU Run

Direction name:

```text
Eklavya E2 learnable multi-teacher residual scout
```

Core claim:

```text
Given an E1 student checkpoint and a shared gap-position cache, full oracle-routed E2 produces a teacher-free student checkpoint whose held-out byte-prediction function beats CE-only continuation, anchor-only KD, and raw anchor byte-KL under the same budget.
```

Signal token:

```text
E2_SCOUT_ORACLE_RESIDUAL_SIGNAL
```

Kill tokens:

```text
E2_SCOUT_ABSORBED_BY_CE_CONTINUATION
E2_SCOUT_ABSORBED_BY_ANCHOR_ONLY_KD
E2_SCOUT_ABSORBED_BY_RAW_BYTE_KL
E2_SCOUT_ROUTER_ORACLE_UPPER_BOUND_FAILS
E2_SCOUT_SHUFFLED_TARGETS_MATCH_REAL
E2_SCOUT_NEGATIVE_NO_STUDENT_LEARNABLE_RESIDUAL
```

Void tokens:

```text
E2_SCOUT_VOID_CACHE_COVERAGE
E2_SCOUT_VOID_STALE_PROVENANCE
E2_SCOUT_VOID_MISSING_E1_WARMSTART
E2_SCOUT_VOID_LOG_CONTAMINATION
E2_SCOUT_VOID_UNMATCHED_STEPS_OR_SHARDS
E2_SCOUT_VOID_EVAL_SPLIT_OVERLAP
E2_SCOUT_VOID_NONFINITE_OR_HARD_FAIL
```

Token precedence:

1. Any void condition wins.
2. If A2 fails A0, emit CE absorption.
3. If A2 fails A1, emit anchor-only absorption.
4. If A2 fails BLD, emit raw byte-KL absorption.
5. If A6 matches A2 after A2 passes, emit shuffled-target absorption.
6. If A2 passes all Phase 1 gates, emit oracle residual signal.

Claim ceiling:

```text
This is not a moonshot result. It is only permission to run Phase 2.
```

## NARRATIVE SECTION

**Gossip-magazine one-sentence story:** The supposedly rebuilt Eklavya is not GPU-ready yet, and its first real test is whether its best possible oracle version can beat one good teacher.

**Does it survive "isn't that obvious?" and "so what?":** It survives only if we keep the oracle-ceiling framing. The non-obvious part is that multi-teacher disagreement can look valuable in teacher outputs while being unlearnable or harmful for the student. The "so what" is that a four-run scout can decide whether to spend or stop.

**Mission test:** A positive E2 scout could matter because compressing complementary teachers into one cheap teacher-free student would serve data and inference efficiency. The current CPU audit itself does not make intelligence cheaper; it prevents wasting compute on an unearned full run.

**Methodology warning:** A methodology, framework, preflight, or negative result is never the moonshot. E2 only re-enters the moonshot search if it creates owned, teacher-free capability that beats boring baselines under strict cost and claim ceilings.
