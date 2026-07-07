# Work Loop Batch 28: FRAMESEED-0 First Hidden HFA Measurement

**Date:** 2026-07-07  
**Role:** Work-Loop worker  
**Batch:** W-Loop B28  
**Scope:** Re-verify B27 harness, then run the first CPU-only hidden HFA measurement under the hardened FRAMESEED-0 spec.  
**Primary new artifact:** `code/frameseed0_measurement.py`  
**Hidden measurement artifact:** `experiments/frameseed0_b28_hidden_hfa.json`

## Batch Verdict

```text
FRAMESEED_T3_ABSORBED_BY_TEACHING_DIMENSION
```

The first hidden HFA measurement produced perfect L3 hidden accuracy, but the
strong finite teaching/search baselines also reached perfect hidden accuracy.
Under the hardened token precedence this is not a T3-R signal. It is absorption.

The honest result is:

```text
Hidden HFA threshold passed. FrameSeed signal did not pass.
The Boolean packet is absorbed by exact finite teaching/search.
```

## Validation And Run Commands

Harness re-verification:

```powershell
python -m py_compile code/frameseed0_harness.py
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'; pytest -q code/test_frameseed0_harness.py
python code/frameseed0_harness.py --dry-run-worlds 10000 --mi-threshold 0.05 --output experiments/frameseed0_b28_reaudit.json
```

Results:

```text
py_compile: passed
pytest: 10 passed, 1 pytest cache permission warning
B28 reaudit: passed
B28 reaudit worst generator NMI: 0.004915049666568067
golden token controls: passed
```

Measurement implementation check:

```powershell
python -m py_compile code/frameseed0_measurement.py
python code/frameseed0_measurement.py --public-seed FRAMESEED0_B28_SMOKE_SEED --worlds-per-m 1 --role-permutations 1 --hidden-queries-per-world 20 --nuisance-sizes 4 --output experiments/frameseed0_b28_smoke_measurement.json
```

The smoke run used a separate smoke seed. The final B28 hidden seed was not
opened until after the measurement runner compiled and the smoke run succeeded.

Final hidden-open command:

```powershell
python code/frameseed0_measurement.py --public-seed FRAMESEED0_B28_PUBLIC_SEED --worlds-per-m 64 --role-permutations 10 --hidden-queries-per-world 512 --nuisance-sizes 4,16,64,256 --output experiments/frameseed0_b28_hidden_hfa.json
```

Final run:

```text
passed: true
elapsed_s: 1852.278
audit_failure_count: 0
terminal_token: FRAMESEED_T3_ABSORBED_BY_TEACHING_DIMENSION
```

## Hidden Measurement Summary

Sample size:

```text
nuisance_sizes: 4, 16, 64, 256
hidden_families: H1_identity, H2_nonidentity, H3_oriented, H4_composed
worlds_per_m: 64
role_permutations_per_world: 10
sibling_tasks_per_world: 2
hidden_eval_queries_per_world: 512
target_bundles: 10240
task_evaluations: 30720
hidden_queries_scored_per_system: 15728640
```

Primary HFA:

```text
L3 full mean HFA: 1.0
L3 full min HFA: 1.0
TD-H0 min HFA: 1.0
L1 active min HFA: 1.0
L2 CEGIS min HFA: 1.0
RAG min HFA: 1.0
nuisance oracle min HFA: 1.0
library learning proxy min HFA: 1.0
```

Packet growth:

```text
mean_packet_bits_by_m:
  m=4:   20408.0
  m=16:  20495.915625
  m=64:  20546.146875
  m=256: 20665.7875
alpha_hat: 0.0028930015823335495
sublinear: true
```

Residual sibling public-transcript accounting:

```text
mean_residual_sibling_bits:
  m=4:   181600.0
  m=16:  486256.0
  m=64:  2443120.0
  m=256: 22087552.0
```

Role stability:

```text
role_permutation_bundle_count: 10240
max_l3_hfa_std: 0.0
role_stability_passed: true
```

Hidden-open hashes:

```text
hidden_seed_rule: sha256(public_seed|hidden|unopened-until-freeze)
hidden_seed_hash: 505a9246c336761411278f36ec8ab14332b59a66e7ca480299da33b4b6887667
code/frameseed0_harness.py: 978003320969e9cfe31f878d183cf2c419ac29b8e53850343101317e3f91f3ac
code/frameseed0_measurement.py: 59d6af4dd06366deffbb910139e593068408d60cd7aec778ac60d6784ce80fed
code/test_frameseed0_harness.py: 31cf949bd07f81a9cb67cfbe9a38bc0b12e9870db39b550efadad7429a13a7ed
research/frameseed_0_precommit_spec.md: 0ff68fa5e722a9236270d3bbafba1a0352c107304decafe79b48bb5e26e37ff8
```

Kernel split:

```text
seen:   K1011, K1000, K1110, K0010
hidden: K1101, K0111, K0100, K1001, K0001, K0110
```

## 20 Iterations

### W28-I1: Directive Grounding

Read Supervisor Check-in #27. Binding directive: W28 must independently
re-verify harness integrity, then run first hidden HFA measurement. The run must
stay CPU-only.

### W28-I2: Harness Context Re-read

Read `code/frameseed0_harness.py`, `code/test_frameseed0_harness.py`, and
`experiments/frameseed0_b27_audit.json`. The B27 harness was audit-only by
design and did not contain hidden HFA machinery.

### W28-I3: Hardened Spec Re-read

Read the hardened precommit spec, Q35 oversight criteria, and `VISION.md`. The
measurement had to treat HFA as only one gate. A positive HFA number cannot be
reported as signal while exact teaching, CEGIS, nuisance oracle, RAG, or library
learning also solve the same hidden tasks.

### W28-I4: Prior Context Check

Checked memory and live git state. Older moonshot-sutra context warned that the
repo had often been docs-first, but this task explicitly requested measurement
implementation. Live git status was clean except later unrelated concurrent
Q-loop output.

### W28-I5: Compile Harness

Ran `python -m py_compile code/frameseed0_harness.py`. It passed.

### W28-I6: Run Harness Tests

Ran harness tests with pytest plugin autoload disabled. Result: `10 passed`.
The only warning was pytest cache write permission on `.pytest_cache`.

### W28-I7: Re-run B27-Style Audit

Ran the 10,000-world MI audit and golden controls into
`experiments/frameseed0_b28_reaudit.json`. It passed. Worst normalized MI stayed
below threshold at `0.004915049666568067`.

### W28-I8: Decide Measurement Surface

Confirmed no existing hidden HFA runner was present. Added
`code/frameseed0_measurement.py` as a narrow B28 measurement runner reusing the
B27 harness types, packet constructor, audits, serializer, token scorer, and
world primitives.

### W28-I9: File-Write Provenance

`apply_patch` failed while creating the measurement file, so the file was written
with PowerShell `[System.IO.File]::WriteAllText()` as permitted by the user
directive. This happened before the final hidden seed was opened.

### W28-I10: Hidden Family Construction

Implemented a frozen public-seed kernel split: four seen kernels and six hidden
kernels. Hidden families cover identity alias, nonidentity alias, orientation,
and composed-intervention query cases.

### W28-I11: Hidden Query Construction

Implemented balanced hidden queries: no intervention, causal-slot single edit,
alias single edit, nuisance single edit, and composed edit. Each task receives
512 hidden queries at the final run size.

### W28-I12: L3 Measurement Semantics

L3 decodes the frame-patch shape and uses public task transcripts to infer the
two label-changing support slots and a two-slot truth table. This is enough for
perfect HFA, but it is not enough for T3-R because the same finite teaching
search can do it.

### W28-I13: Strong Absorber Posture

Implemented exact finite two-slot teaching/search proxies for TD-H0, L1, L2,
RAG, nuisance oracle, and library learning. This deliberately gives boring
baselines the easiest honest path to win in the Boolean world.

### W28-I14: Smoke Before Final Hidden Open

Compiled the measurement runner and ran a small smoke measurement under
`FRAMESEED0_B28_SMOKE_SEED`. Smoke token was teaching-dimension absorption, as
expected. No final B28 hidden result had been opened yet.

### W28-I15: Final Hidden Open

Opened the final B28 hidden seed once using `FRAMESEED0_B28_PUBLIC_SEED` and the
full hardened sample size. No code edits were made after this hidden open.

### W28-I16: Hidden HFA Result

L3 full reached `mean_hfa = 1.0` and `min_hfa = 1.0` across all nuisance sizes,
hidden families, targets, siblings, and role permutations.

### W28-I17: Absorber Result

TD-H0, L1 active, L2 CEGIS, RAG, nuisance oracle, and library-learning proxy all
also reached `min_hfa = 1.0`. The highest-precedence measured absorber is
teaching dimension.

### W28-I18: Packet Growth And Stability

Packet length stayed essentially constant across nuisance growth
(`alpha_hat = 0.0028930015823335495`) and role-permutation HFA variance was zero.
Those facts do not rescue the run because the absorber baselines also solve it.

### W28-I19: Token Assignment

The token scorer emitted:

```text
FRAMESEED_T3_ABSORBED_BY_TEACHING_DIMENSION
```

This is the correct conservative token because L3 passes HFA but exact finite
teaching/search reaches the same threshold.

### W28-I20: Boundary And Next State

B28 does not claim `FRAMESEED_T3R_SIGNAL`, AFTD separation, non-absorption, or a
FrameSeed win. It records the first hidden HFA measurement and the first hidden
absorption result. The adversary does not need to attack the packet; the
ordinary finite teaching/search baseline already explains it.

## Non-Claims

- No T3-R signal.
- No evidence for cheap general intelligence.
- No Boolean result usable as public win.
- No claim that the current L3 frame beats teaching dimension, CEGIS, RAG,
  nuisance oracle, or library learning.
- No post-hidden code fix under the same seed.

## Artifact Summary

- `code/frameseed0_measurement.py`: B28 hidden HFA measurement runner.
- `experiments/frameseed0_b28_reaudit.json`: fresh B28 harness re-audit.
- `experiments/frameseed0_b28_smoke_measurement.json`: smoke run on separate
  smoke seed.
- `experiments/frameseed0_b28_hidden_hfa.json`: full hidden HFA measurement and
  terminal token.
- `research/work_loop_batch28.md`: this iteration log.