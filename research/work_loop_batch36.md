# Work-Loop Batch 36: WGD-0 Harness Implementation

**Date:** 2026-07-07  
**Role:** W-Loop worker  
**Iterations:** 20  
**Status:** WGD-0 pre-hidden audit harness implemented. Harness integrity only. No hidden seed opened. No WGD signal measured.

Two invariants held fixed:

1. Swing for the home run.
2. The loop only stops on a won-over adversary.

## Grounding

Read and used the live B36 directive surface:

- `research/dual_loop_supervisor_checkin_34.md`
- `research/wgd_0_precommit_spec.md`
- `research/question_loop_batch43.md`
- `research/METHODOLOGY_TEMPLATE.md`
- `code/frameseed0_harness.py`
- `research/VISION.md`

Binding interpretation: WGD-0 is not ready for signal measurement until the harness can prove that boring explanations are implemented as real native absorbers, receive equal affordances, have public calibration worlds where they can win, and are charged through a recomputable cost ledger.

## Implementation Artifacts

- Added `code/wgd0_harness.py`.
- Added `code/test_wgd0_harness.py`.
- Generated validation artifact: `.codex_tmp/wgd0_audit_full.json`.

## Iteration Log

### B36-I1: Directive Grounding
Read the supervisor directive. This batch is implementation, not further doctrine. The scope is harness integrity first, no hidden scoring and no WGD signal claim.

### B36-I2: Spec Extraction
Extracted the hard requirements from `wgd_0_precommit_spec.md`: native absorber roster, equal-information contract, cost categories, grammar IR constraints, erasure roster, leakage audits, sibling independence, composition hostility, repair/abstention regimes, and token precedence.

### B36-I3: Q43 Threat Model
Treated `NATIVE_ABSORBER_THEATER` as the main engineering risk. A named absorber is not enough; the harness must make each absorber win on a public calibration world where its boring explanation is true.

### B36-I4: FrameSeed Pattern Transfer
Reused the FrameSeed harness pattern: audited RNG streams, canonical JSON serialization, blind construction, provenance checks, baseline parity, cost recomputation, leakage probes, and golden token controls.

### B36-I5: Public Typed World Scaffold
Implemented opaque typed WGD worlds with hidden role maps, same-type decoys, typed records, literal deltas, public feedback symbols, and generic output forms. Learner-public schema avoids operation names, hidden roles, solution programs, and scorer internals.

### B36-I6: Blind Public Packet
Implemented `BlindWGDPacketConstructor`, which constructs only from learner-public transcript facts. The packet declares typed surface inventory, feedback channel, grammar IR schema, repair/abstention contract, sibling/composition gate, provenance, and cost categories.

### B36-I7: Grammar IR Freeze
Implemented a frozen `GrammarIR` with allowed node types, node-level provenance, cost attribution, and smuggling audit. The audit rejects code blobs, caches, hidden labels, solver payloads, and unapproved node types.

### B36-I8: Equal-Affordance Views
Implemented baseline packet views and an affordance parity matrix. Absorbers receive identical canonical packet bytes, matched task bundle hashes, matched query budgets, round-trip translations, and equal adapter access cost.

### B36-I9: Cost Ledger
Implemented WGD-0 cost categories: `G`, `B_i`, `P_i`, `E_i`, `C_i`, `Q_i`, `V_i`, `R_i`, `A_i`, `L`, `H`, `O`, `N`. The ledger reports substrate-free and substrate-charged totals plus mechanical ratios.

### B36-I10: Human-Substrate Ledger
Charged the harness file, precommit spec, and public packet as explicit artifacts. The claim ceiling states that no claim is made that the hand-authored substrate was learned.

### B36-I11: Native Absorber Roster
Declared 20 required native absorbers/controls: schema binding, entity resolution, PBE, PBE+CEGIS, CEGIS, active CEGIS, MDL library, sibling library, active learning, causal/invariant, constraint repair, anomaly abstention, ontology oracle, verifier-template oracle, obligation-label oracle, generator-leakage classifier, nuisance oracle, representation/substrate prior, language prior, and post-hoc compression.

### B36-I12: Absorber Capability Witnesses
Implemented public calibration witnesses where every absorber wins on its own home turf. These include exhaustive intervention scoring, key/alias linkage, typed DSL PBE, version-space CEGIS, binary active CEGIS, MDL macro induction, sibling library reuse, causal intervention scoring, nearest-valid repair, anomaly abstention, oracle controls, leakage classifier controls, and post-hoc compression audit.

### B36-I13: PBE/MDL Hardening
A first smoke run caught weak calibration: PBE was underconstrained and MDL compression was not strong enough. Added an unsafe disambiguating example and expanded the macro library witness to eight tasks so the absorber win is genuine.

### B36-I14: Packet Denylist Audit Fix
A first packet audit flagged the grammar denylist itself as banned public text. Fixed the scan so denylist terms are allowed only inside the frozen grammar-IR schema declaration, not executable learner-public payload.

### B36-I15: Sibling Clone Resistance
A first sibling audit caught zero behavior distance. Randomized target guard/status behavior and added explicit sibling behavior regimes. Final sibling audit counts three nonduplicate siblings with zero shared field IDs.

### B36-I16: Composition Hostility
Added public composition probes for noncommutation, guard conflict, interference, and preserved component behavior declarations. This prevents composition from being only a saved pipeline.

### B36-I17: Repair And Abstention Controls
Added three repair regimes and native repair baselines: nearest-valid search, constraint repair, CEGIS repair, active retry, and patch library. Added anomaly/uncertainty abstention with risk-coverage reporting.

### B36-I18: Leakage And Erasure Rosters
Added predictive leakage audit with MI metrics over type tags, positions, opaque IDs, value shapes, serializer length, and hash prefixes. Added frozen high-order attack roster and all 22 required geometry-erasure entries without running hidden erasure HFA.

### B36-I19: Token And Governance Controls
Implemented WGD token precedence and golden controls. Added fake hidden-open governance scenarios: baseline crash, scorer bug, serializer bug, timeout mismatch, malformed hidden family, and unexpected leak all map to void tokens.

### B36-I20: Validation And Final Check
Validated syntax, tests, and the full pre-hidden audit. The harness passes without opening hidden seed, without reporting hidden HFA, and without measuring WGD signal.

## Validation

Commands run:

```powershell
python -m py_compile code/wgd0_harness.py
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'; pytest -q -p no:cacheprovider code/test_wgd0_harness.py
python code/wgd0_harness.py --output .codex_tmp/wgd0_audit_full.json > .codex_tmp/wgd0_audit_stdout.json
python -c "import json; p=json.load(open('.codex_tmp/wgd0_audit_full.json')); print('passed', p['passed']); print('findings', len(p['findings'])); print('failed', [f['check_id'] for f in p['findings'] if not f['passed']]); m=p['metrics']; print('absorbers', m['absorber_count']); print('hidden_seed_opened', m['hidden_seed_opened']); print('hidden_hfa_reported', m['hidden_hfa_reported']); print('signal_measured', m['wgd_signal_measured']); print('worst_leakage', m['predictive_leakage_audit']['worst_metric']); print('sibling_nonduplicate_count', m['sibling_independence_audit']['nonduplicate_count'])"
```

Observed validation summary:

```text
pytest: 8 passed in 0.19s
harness passed: True
findings: 74
failed findings: []
absorbers: 20
hidden_seed_opened: False
hidden_hfa_reported: False
wgd_signal_measured: False
worst_leakage: 0.11569208856346087
sibling_nonduplicate_count: 3
```

## Final Token

```text
WGD_0_PRE_HIDDEN_AUDIT_HARNESS_IMPLEMENTED_NATIVE_ABSORBERS_CALIBRATED_NO_SIGNAL_MEASURED
```

## Next Gate

Q-Loop B44 should attack the implementation, especially:

- whether each calibration witness is strong enough to count as native competence;
- whether the affordance parity matrix is too identity-translation-heavy;
- whether leakage threshold `0.12` is acceptable for the first public harness or should be hardened toward `0.05`;
- whether sibling behavior distance should become richer than feedback-sequence distance;
- whether cost `H` should split harness code, spec labor, parser labor, and baseline adapter labor into finer ledger entries.