# Work-Loop Batch 31: SHEETS-0 Hidden HFA Measurement

**Date:** 2026-07-07  
**Role:** W-Loop B31 worker  
**Directive:** Re-verify the SHEETS-0 harness, smoke on a separate seed, open the hidden seed once, score all systems including typed baselines, emit terminal token, and write the batch log.  
**CPU only:** yes.  
**Code changes after hidden open:** no.  
**Hidden seed opened:** yes, once.

## Outcome

SHEETS-0 hidden HFA was measured at the full spec-sized grid. L3 reached perfect hidden typed HFA, but the result did not survive the adversary. Charged exact schema/task bindings alone were enough to solve the hidden tasks, and the same information let typed PBE, data-wrangling, CEGIS, and MDL-library baselines solve at threshold.

Terminal token:

```text
FRAMESEED_SHEETS0_ABSORBED_BY_SCHEMA_BINDING
```

Primary artifacts:

- `code/frameseed_sheets0_measurement.py`
- `experiments/frameseed_sheets0_b31_reaudit.json`
- `experiments/frameseed_sheets0_b31_smoke_measurement.json`
- `experiments/frameseed_sheets0_b31_hidden_hfa.json`
- `research/work_loop_batch31.md`

## Validation And Commands

Pre-hidden harness re-verification:

```text
python -m py_compile code/frameseed_sheets0_harness.py
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'; python -m pytest code/test_frameseed_sheets0_harness.py -q
python code/frameseed_sheets0_harness.py --dry-run-worlds 1000 --leakage-threshold 0.08 --output experiments/frameseed_sheets0_b31_reaudit.json
```

Results:

```text
py_compile harness: pass
SHEETS harness tests: 7 passed
B31 reaudit: pass
B31 reaudit worst leakage metric: 0.014158279193434313 <= 0.08
```

Measurement runner validation and separate-seed smoke:

```text
python -m py_compile code/frameseed_sheets0_measurement.py
python code/frameseed_sheets0_measurement.py --public-seed FRAMESEED_SHEETS0_B31_SMOKE_SEED --worlds-per-m 1 --role-permutations 1 --hidden-queries-per-world 20 --nuisance-sizes 4 --audit-worlds 100 --leakage-threshold 0.25 --output experiments/frameseed_sheets0_b31_smoke_measurement.json
```

Smoke result:

```text
terminal_token: FRAMESEED_SHEETS0_ABSORBED_BY_SCHEMA_BINDING
smoke audit failures: 0
smoke L3 mean/min HFA: 1.0 / 1.0
smoke binding-only HFA: 1.0
smoke packet-erasure drop: 0.0
```

Final hidden-open command:

```text
python code/frameseed_sheets0_measurement.py --public-seed FRAMESEED_SHEETS0_B31_PUBLIC_SEED --worlds-per-m 64 --role-permutations 10 --hidden-queries-per-world 256 --nuisance-sizes 4,16,64,256 --audit-worlds 1000 --leakage-threshold 0.08 --output experiments/frameseed_sheets0_b31_hidden_hfa.json
```

Final hidden run:

```text
elapsed_s: 845.158
hidden_seed_hash: 972aa9421e94ce8b4cb06b7171dce5031918394818d5dd6daaa8589971f7760e
passed: true
audit_failure_count: 0
target_bundles: 15360
hidden_queries_scored_per_system: 15728640
role_permutation_stability max_l3_hfa_std: 0.0
```

## Final Metrics

Primary HFA:

```text
L3 full mean HFA: 1.0
L3 full min HFA: 1.0
PBE/PROSE min HFA: 1.0
Typed CEGIS exact min HFA: 1.0
Typed MDL library min HFA: 1.0
Data wrangling min HFA: 1.0
Nuisance oracle min HFA: 1.0
Relational/key/schema-only mean HFA: 0.400390625
Unit-only mean HFA: 0.19986979166667657
Constraint/data-repair-only mean HFA: 0.19986979166666757
TD-H0, L0 rote, RAG mean HFA: 0.0
```

Typed output mix:

```text
non_boolean_fraction: 0.8001302083333334
StableID: 3153920
CanonicalRowMultiset: 3143680
UnitValue(Rational,Unit): 6287360
ActionAccepted(canonical_effect): 1391447
ActionRejected(canonical_reason_code): 1752233
```

Cost and gate facts:

```text
packet_growth_sublinear: true
binding_growth_sublinear: true
binding_only_ablation mean_hfa: 1.0
packet_erasure_drop_pp: 0.0
packet_erasure_drop_passed: false
aftd_all_in_passed: false
composition_gate_passed: false
bits_counted: true
claim_ceiling_honored: true
```

Absorptions:

```text
schema_binding: true
pbe: true
data_wrangling: true
typed_cegis: true
library_learning: true
nuisance_oracle: true
```

Terminal precedence selected `SCHEMA_BINDING` before PBE, typed CEGIS, library learning, and nuisance oracle.

## 20 Iterations

### B31-I1: Directive Grounding

Read the supervisor check-in, SHEETS harness, harness tests, hardened spec, Q38 warnings, Boolean measurement template, and Vision. The task was measurement, not more design meditation.

### B31-I2: Harness Boundary

Confirmed B30 had a pre-hidden audit harness but no SHEETS-specific hidden HFA runner. The Boolean B28 runner was the implementation template.

### B31-I3: Memory Boundary

A quick memory lookup found older moonshot-sutra docs/hygiene entries, not this measurement line. Live repository files controlled the run.

### B31-I4: Harness Reverification

Re-ran the typed harness compile, tests, and leakage audit before hidden work. The existing B30 harness passed.

### B31-I5: Hidden Runner Design

Added `code/frameseed_sheets0_measurement.py` before hidden open. It derives the hidden seed from `sha256(public_seed|sheets0-hidden|manifest_hash|unopened-until-freeze)` and keeps the B30 harness code frozen.

### B31-I6: Hidden Families

Implemented six hidden families: key rename, adversarial display-name/key, unit normalization, key-unit composition, constraint/action, and full stress. The final run covered all six.

### B31-I7: Typed Query Semantics

Queries used deterministic typed operations: stable-ID lookup, canonical join, unit normalization, aggregate by key, and validate/apply. Outputs were canonical typed JSON objects.

### B31-I8: Strong Baseline First Refusal

The runner scored every declared baseline name. Strong typed pipeline proxies included PBE/PROSE, data wrangling, typed CEGIS, typed MDL library, operation verifier search, goal-conditioned CEGIS, obligation-template library, active goal disambiguation, and nuisance oracle.

### B31-I9: Binding Accounting

The decisive choice was to expose charged exact task bindings equally to L3 and the strong typed baselines. This is conservative: if exact task bindings are enough, the packet has not shown frame transfer.

### B31-I10: Smoke Bug

The first smoke caught an unscored declared baseline: `active_goal_disambiguation`. This was fixed before hidden open.

### B31-I11: Smoke Scale Repair

The first clean smoke also exposed that a list-backed accumulator would be too memory-heavy for the full run. Replaced it with an online Welford accumulator before hidden open.

### B31-I12: Smoke Result

The separate smoke seed passed and emitted the same expected absorption token. No hidden seed had been opened yet.

### B31-I13: Hidden Open

Opened `FRAMESEED_SHEETS0_B31_PUBLIC_SEED` once with the full grid. No code edits were made after this point.

### B31-I14: Full Measurement

The full run scored 15,728,640 hidden queries per system across 15,360 target bundles. Runtime was 845.158 seconds on CPU.

### B31-I15: L3 Result

L3 full packet plus charged bindings reached 1.0 mean and 1.0 min HFA across all m, families, targets, siblings, and role permutations.

### B31-I16: Absorber Result

PBE/PROSE, data wrangling, typed CEGIS, typed MDL library, active goal disambiguation, and nuisance oracle also reached 1.0 min HFA under the same binding and typed-operator information.

### B31-I17: Packet Erasure

The binding-only ablation reached 1.0 HFA. Packet-erasure drop was 0.0, so the packet did not demonstrate necessary reusable frame content.

### B31-I18: Token Assignment

Token evidence had `schema_binding`, `pbe`, `data_wrangling`, `typed_cegis`, and `library_learning` true. Terminal precedence selected schema-binding absorption.

### B31-I19: Adversarial Interpretation

This is not signal. It says the current SHEETS-0 measurement is explained by charged schema/task bindings and ordinary typed table-program pipelines, exactly the class Q38 warned could absorb the result.

### B31-I20: Final State

B31 produced a hidden measurement artifact and terminal token. The loop is not won over; the next milestone gate must treat this as absorption unless the direction is radically reframed around binding discovery, self-discovered typed transformation grammars, or native typed synthesis/library learning.

## Verdict

```text
FRAMESEED_SHEETS0_ABSORBED_BY_SCHEMA_BINDING
```

The home-run interpretation fails here. The typed packet can be made to score perfectly only in a regime where charged bindings and standard typed pipeline synthesis also score perfectly. The honest result is absorption, not a FrameSeed signal.