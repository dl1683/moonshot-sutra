# Work-Loop Batch 37: WGD-0 First Hidden Measurement

**Date:** 2026-07-07  
**Role:** W-Loop worker  
**Iterations:** 20  
**Status:** First WGD-0 hidden measurement completed. Hidden result absorbed by schema/binding discovery. No WGD signal claimed.

Two invariants held fixed:

1. Swing for the home run.
2. The loop only stops on a won-over adversary.

## Grounding

Read the required live context:

- `research/dual_loop_supervisor_checkin_34.md`
- `code/wgd0_harness.py`
- `code/test_wgd0_harness.py`
- `research/wgd_0_precommit_spec.md`
- `research/question_loop_batch43.md`
- `research/question_loop_batch44.md`
- `research/VISION.md`

Binding interpretation: B36 produced only a pre-hidden audit harness. B37 therefore needed a separate measurement runner before hidden opening. The B36 harness was not modified. The new runner was created before hidden opening, smoked on a separate seed, then frozen for the final hidden run.

## Artifacts

- Added pre-hidden measurement runner: `code/wgd0_measurement.py`
- Smoke artifact: `experiments/wgd0_b37_smoke_measurement.json`
- Hidden artifact: `experiments/wgd0_b37_hidden_measurement.json`
- Pre-hidden audit artifact: `.codex_tmp/wgd0_b37_prehidden_audit.json`

No code changes were made after the final hidden measurement command.

## Harness Re-Verification

Commands run before hidden opening:

```powershell
python -m py_compile code/wgd0_harness.py
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'; pytest -q -p no:cacheprovider code/test_wgd0_harness.py
python code/wgd0_harness.py --output .codex_tmp/wgd0_b37_prehidden_audit.json > .codex_tmp/wgd0_b37_prehidden_audit_stdout.json
```

Observed:

```text
pytest: 8 passed in 0.19s
pre-hidden audit passed: True
findings: 74
failed findings: []
absorbers: 20
hidden_seed_opened: False
hidden_hfa_reported: False
wgd_signal_measured: False
worst_leakage: 0.11569208856346087
sibling_nonduplicate_count: 3
manifest_hash: 5b846521564f774189ca4798af523829
```

## Measurement Protocol

The B37 runner measures hidden feedback HFA over opaque typed WGD worlds. It freezes measurement version, public and smoke seeds, implementation hashes, scorer identity, hidden families, case kinds, systems and absorbing systems, and token precedence.

Hidden seed rule:

```text
sha256(public_seed|public_smoke_seed|manifest_hash|hidden|unopened_until_freeze)
```

The final hidden command was run once:

```powershell
python code/wgd0_measurement.py --output experiments/wgd0_b37_hidden_measurement.json > .codex_tmp/wgd0_b37_hidden_stdout.json
```

Smoke was run first on separate seeds:

```powershell
python code/wgd0_measurement.py --public-seed WGD0_B37_SMOKE_MEASUREMENT_SEED --smoke-seed WGD0_B37_SMOKE_AUDIT_SEED --worlds-per-family 4 --cases-per-world 12 --dry-run-worlds 2000 --output experiments/wgd0_b37_smoke_measurement.json > .codex_tmp/wgd0_b37_smoke_stdout.json
```

Smoke result:

```text
passed: True
terminal_token: WGD_ABSORBED_BY_SCHEMA_OR_BINDING_DISCOVERY
hidden_cases: 192
hidden_worlds: 16
wgd_grammar HFA: 1.0
schema_binding HFA: 1.0
pbe_cegis HFA: 1.0
majority_feedback HFA: 0.18229166666666666
schema_binding cost ratio vs WGD: 0.08928746173094936
pbe_cegis cost ratio vs WGD: 0.10852333887855954
```

## Hidden Result

Final hidden measurement summary:

```text
passed: True
terminal_token: WGD_ABSORBED_BY_SCHEMA_OR_BINDING_DISCOVERY
interpretation: schema/binding baseline matches hidden feedback HFA at <=4x all-in cost
manifest_hash: 1e7fb9be866f72aba74f5be022afb6a9
hidden_seed_hash: 6661dae8b2ceff0caac1c80c25c8797a3a449915213e6e16a92ca651a769aa8b
hidden_worlds: 128
hidden_cases: 1536
scored_predictions: 6144
```

System HFA:

```text
wgd_grammar: 1.0
schema_binding: 1.0
pbe_cegis: 1.0
majority_feedback: 0.19401041666666666
```

Mean cost bits:

```text
wgd_grammar: 35690.125
schema_binding: 3147.1875
pbe_cegis: 3835.1875
majority_feedback: 191.125
```

Absorber cost ratios versus WGD:

```text
schema_binding: 0.08818090438181429
pbe_cegis: 0.10745794529999544
```

Pre-hidden audit embedded in final artifact:

```text
passed: True
findings: 74
failed_findings: []
absorber_count: 20
hidden_seed_opened before measurement: False
wgd_signal_measured before measurement: False
worst_leakage: 0.11868583898798497
sibling_nonduplicate_count: 3
```

## Iteration Log

### B37-I1: Directive Grounding

Read the supervisor directive, hardened spec, Q43 theater warning, Q44 oversight criteria, harness code, harness tests, and Vision. The request is measurement, but the adversarial rule remains stronger than the request to narrate a win.

### B37-I2: Harness Reality Check

Confirmed `code/wgd0_harness.py` is explicitly pre-hidden only: it does not open a hidden seed, report hidden HFA, or measure WGD signal. A separate runner was required before any hidden seed could be opened.

### B37-I3: Integrity Before Measurement

Re-ran compile, tests, and full pre-hidden audit. The harness passed 74 findings with no failed checks, 20 absorber witnesses, hidden flags false, and no signal measurement.

### B37-I4: Measurement Surface Choice

Chose a narrow first hidden measurement: feedback HFA over hidden opaque typed worlds using the B36 substrate. This is not enough for a positive WGD claim if a higher-precedence absorber fires.

### B37-I5: Code Isolation

Added `code/wgd0_measurement.py` as a separate B37 artifact. The B36 harness, tests, constructor, scorer controls, token policy, and pre-hidden audit were not mutated.

### B37-I6: Hidden Seed Rule

Bound the final hidden seed to the public seed, smoke seed, frozen measurement manifest hash, and hidden namespace:

```text
sha256(public_seed|public_smoke_seed|manifest_hash|hidden|unopened_until_freeze)
```

### B37-I7: Manifest Freeze

The measurement manifest records implementation hashes, systems, absorbing systems, hidden families, case kinds, scorer identity, token precedence, and configuration before deriving the hidden seed.

### B37-I8: Hidden Families

Used four hidden family labels matching WGD pressures: surface invariance, typed measurement, safety/invalidity, and repair/abstention/composition. These labels parameterize hidden generation; they are not learner-public operation labels.

### B37-I9: Hidden Case Mix

Scored accepted quantity edits, unsafe quantity edits, locked-record rejects, guard/status ambiguity, accepted guard/status edits, wrong field/object cases, decoy edits, and locked decoy rejects.

### B37-I10: WGD Candidate

The `wgd_grammar` candidate infers quantity, guard, and status fields from the public transcript and applies the public feedback rules through a frozen grammar container and packet cost.

### B37-I11: Native Absorbers In The Measurement

The matched schema/binding and PBE/CEGIS systems receive only the same public transcript structure. They infer the same role bindings and feedback program without hidden role-map access.

### B37-I12: Smoke Measurement

Ran the full protocol on separate smoke seeds. Smoke produced the same terminal shape expected from the public transcript geometry: WGD HFA 1.0, schema/binding HFA 1.0, PBE/CEGIS HFA 1.0, majority feedback low, and schema/PBE costs far below the 4x absorption boundary.

### B37-I13: Final Freeze Check

After smoke, recompiled `code/wgd0_harness.py` and `code/wgd0_measurement.py` and reran the harness tests. Both Python files compiled and `pytest` again reported 8 passing tests.

### B37-I14: Hidden Opening

Opened the final B37 hidden seed once by running the default B37 measurement command. No measurement code was edited afterward.

### B37-I15: Hidden WGD HFA

The WGD grammar candidate achieved 1.0 HFA over 1,536 hidden cases across 128 hidden worlds. This alone is not signal because positive WGD requires native absorbers to fail or pay at least 4x all-in.

### B37-I16: Schema/Binding Absorption

The schema/binding baseline also achieved 1.0 HFA. Its mean cost was 3,147.1875 bits versus 35,690.125 bits for WGD, a 0.08818 cost ratio. This is well under the 4x absorption boundary.

### B37-I17: PBE/CEGIS Absorption

The PBE/CEGIS baseline achieved 1.0 HFA with 3,835.1875 mean bits, a 0.10746 cost ratio versus WGD. It independently supports the same absorption conclusion, but schema/binding has higher token precedence.

### B37-I18: Token Precedence

Because schema/binding and PBE/CEGIS matched hidden HFA at far less than 4x WGD cost, `native_absorbers_fail_or_pay_4x` is false. Token precedence assigns:

```text
WGD_ABSORBED_BY_SCHEMA_OR_BINDING_DISCOVERY
```

### B37-I19: Claim Ceiling

No `WGD_SIGNAL` claim is allowed. The correct claim is that this first hidden measurement exposed a public-transcript role-binding route strong enough to solve the hidden feedback cases cheaply.

### B37-I20: Adversary Verdict

The adversary is not won over. The measurement is useful because it killed the first hidden WGD-0 shape cleanly: the active ingredient is not discovered world geometry beyond boring role binding and public-feedback program induction.

## Final Token

```text
WGD_ABSORBED_BY_SCHEMA_OR_BINDING_DISCOVERY
```

## Next Gate

The next W-loop should not rescue this hidden seed. The seed is spent and absorbed. If WGD continues, the generator and learner-public transcript must be hardened so edited-field exposure and public feedback programs no longer let schema/binding or PBE/CEGIS solve the hidden cases at a tiny fraction of WGD cost.