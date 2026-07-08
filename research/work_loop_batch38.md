# Work-Loop Batch 38: WGD-0 Final Hard Domain Attempt

**Date:** 2026-07-07  
**Role:** W-Loop worker  
**Iterations:** 20  
**Status:** Final hard-domain measurement completed. Enumeration did not absorb. Generic constraint discovery did absorb. No WGD signal claimed.

Two invariants held fixed:

1. Swing for the home run.
2. The loop only stops on a won-over adversary.

## Grounding

Read the required live context:

- `research/dual_loop_supervisor_checkin_35.md`
- `research/wgd_0_precommit_spec.md`
- `research/work_loop_batch37.md`
- `code/wgd0_harness.py`
- `research/METHODOLOGY_TEMPLATE.md`
- `research/VISION.md`

Also checked the immediate B45 harness critique and B36/B37 implementation history to avoid repeating the same toy-schema failure.

Binding interpretation: B38 needed one last serious hard-domain attempt. The specific directive was not to rescue the B37 hidden seed, but to create a domain with at least 64 compositional rules, exponential search, and baselines that genuinely enumerate rather than exploit edited-field binding or public feedback-program shortcuts.

## Artifacts

- Added B38 hard-domain runner: `code/wgd0_b38_hard_domain.py`
- Pre-hidden audit artifact: `.codex_tmp/wgd0_b38_prehidden_audit.json`
- Smoke artifact: `experiments/wgd0_b38_smoke_measurement.json`
- Hidden artifact: `experiments/wgd0_b38_hidden_measurement.json`
- Hidden stdout capture: `.codex_tmp/wgd0_b38_hidden_stdout.json`

`apply_patch` failed to write new files in this workspace, so the B38 code and this log were written with PowerShell `[System.IO.File]::WriteAllText()` as explicitly allowed by the B38 request.

No measurement code was changed after the hidden command.

## Domain

B38 uses an opaque XOR-basis composition world:

```text
rule_count = 64
state_bits = 128
candidate subset space = 2^64 = 18,446,744,073,709,551,616
held-out ordered composition space log2 = 144.0
```

Each world has 64 opaque atomic rules. Each rule maps the zero state to an opaque 128-bit vector. A composition is a subset XOR of those atomic vectors. Hidden cases require:

- action subset recovery;
- held-out composition recovery;
- single-rule local repair;
- abstention on out-of-span targets.

The hard part for a pure enumerator is explicit: to solve a valid target by search alone, it must enumerate candidate subsets of the 64 rules. The per-case hidden enumeration budget was frozen at 8,000 candidates, or about `4.33680868994202e-16` of the candidate subset space.

## Systems

Measured systems:

```text
wgd_basis_grammar
constraint_solver_absorber
lexicographic_enumerator
size_first_enumerator
random_enumerator
meet_in_middle_truncated
```

The four enumerators are deliberately genuine enumerators. They compose candidate subsets and compare them to the target. They do not call rank, solve, inverse, schema binding, Gaussian elimination, or any shortcut solver.

The constraint absorber is intentionally present. If the WGD-like basis grammar only wins because the world is linear over GF(2), a generic rank/solve method is the boring explanation and must get first refusal.

## Validation Commands

Pre-hidden syntax and audit:

```powershell
python -m py_compile code/wgd0_b38_hard_domain.py
python code/wgd0_b38_hard_domain.py --mode audit --output .codex_tmp/wgd0_b38_prehidden_audit.json > .codex_tmp/wgd0_b38_prehidden_audit_stdout.json
```

Pre-hidden audit summary:

```text
passed=True
token=B38_PREHIDDEN_AUDIT_ONLY_NO_HIDDEN_OPEN
manifest=f96e95b3684d2eea2c76b24de4d84345
audit_passed=True
failed=
rule_count=64
space_log2=64
budget=8000
hidden_open=False
```

Smoke command:

```powershell
python code/wgd0_b38_hard_domain.py --mode smoke --worlds 4 --cases-per-world 16 --enumeration-budget 2000 --output experiments/wgd0_b38_smoke_measurement.json > .codex_tmp/wgd0_b38_smoke_stdout.json
```

Smoke summary:

```text
passed=True
terminal_token=WGD_ABSORBED_BY_CONSTRAINT_DISCOVERY
hidden_open=False
smoke_open=True
manifest=88fc492ae1bce9fec58494a2fee28590
worlds=4
cases=64
scored_predictions=384
rules=64
space_log2=64
ordered_log2=144.0
enumeration_fraction_per_case=1.0842021724855e-16
baselines_genuinely_enumerate=True
wgd_hfa=1.0
constraint_hfa=1.0
lexicographic_hfa=0.25
size_first_hfa=0.25
random_hfa=0.25
meet_in_middle_truncated_hfa=0.25
constraint_absorbs=True
native_absorbers_fail_or_pay_4x=False
constraint_cost_ratio_vs_wgd=0.9673932788374205
```

Final compile check before hidden open:

```powershell
python -m py_compile code/wgd0_b38_hard_domain.py
```

Final hidden command, run once:

```powershell
python code/wgd0_b38_hard_domain.py --mode hidden --output experiments/wgd0_b38_hidden_measurement.json > .codex_tmp/wgd0_b38_hidden_stdout.json
```

## Hidden Result

Final hidden measurement summary:

```text
passed=True
terminal_token=WGD_ABSORBED_BY_CONSTRAINT_DISCOVERY
hidden_seed_opened=True
manifest_hash=f96e95b3684d2eea2c76b24de4d84345
secret_seed_hash=b520415b7915cbda33a18c3b415ded309c80890d045bf27d4b311c0ca86a24c7
worlds=8
cases=256
scored_predictions=1536
elapsed_s=8.413
```

Hardness:

```text
grammar_rule_count=64
candidate_space=18446744073709551616
candidate_space_log2=64
ordered_composition_space_log2=144.0
enumeration_budget_per_case=8000
enumeration_fraction_per_case=4.33680868994202e-16
baselines_genuinely_enumerate=True
```

Functional metrics:

```text
wgd_basis_grammar HFA=1.0
constraint_solver_absorber HFA=1.0
lexicographic_enumerator HFA=0.25
size_first_enumerator HFA=0.25
random_enumerator HFA=0.25
meet_in_middle_truncated HFA=0.25
composition_hfa=1.0
repair_success=1.0
abstention_recall=1.0
functional_gates_passed=True
```

Absorber result:

```text
constraint_solver_absorbs=True
constraint_cost_ratio_vs_wgd=0.9673932788374205
native_absorbers_fail_or_pay_4x=False
lexicographic_enumerator absorbs=False
size_first_enumerator absorbs=False
random_enumerator absorbs=False
meet_in_middle_truncated absorbs=False
```

## Iteration Log

### B38-I1: Directive Grounding

Read the B35 supervisor directive. B38 is the final hard-domain attempt before methodology-paper fallback. The domain must be hard enough that brute enumeration is not a cheap explanation.

### B38-I2: B37 Failure Extraction

B37 was absorbed because edited-field exposure and public feedback programs made schema/binding and PBE/CEGIS cheaper than WGD. B38 therefore cannot be another small typed role-binding task.

### B38-I3: Threat Model Selection

Selected the direct threat named by the supervisor: finite enumeration over supplied grammar. The domain must make that search exponential in rule count, not merely inconvenient.

### B38-I4: Rule Count Commitment

Committed to exactly 64 atomic compositional rules. This meets the minimum rule-count directive and makes the subset-composition search space exactly `2^64`.

### B38-I5: State Space Choice

Used 128-bit opaque states so 64 rule vectors can be independently embedded with room for out-of-span abstention cases. This avoids a degenerate full-basis world where every target is valid.

### B38-I6: Public Rule Atlas

The learner-public object is a rule atlas of opaque handles and observed atomic effects. It does not use semantic operation names. It is still a strong substrate, and that cost is counted.

### B38-I7: Composition Gate

Hidden composition cases require held-out 24-rule compositions. Ordered sequence space has `144.0` log2 size, while the subset target space remains `2^64`.

### B38-I8: Repair Gate

Repair cases present a corrupted proposed subset but score recovery of the correct subset from the target delta. The WGD basis grammar repairs by solving the target delta, not by retrying feedback.

### B38-I9: Abstention Gate

Out-of-span targets are generated outside the 64-rule span. A correct system must abstain rather than force a subset.

### B38-I10: Enumerative Baselines

Implemented four genuine enumerators: lexicographic, size-first, random, and truncated meet-in-the-middle. They only compose candidate subsets and compare to the target.

### B38-I11: Shortcut Ban

The enumerators do not call rank, solve, inverse, Gaussian elimination, schema binding, or target-specific field inference. Their artifact records `used_shortcuts=False`.

### B38-I12: Honest Constraint Absorber

Added `constraint_solver_absorber`, a generic GF(2) rank/solve baseline. This is a deliberate kill switch: if the WGD basis grammar is only linear algebra, the result must not be narrated as discovered intelligence.

### B38-I13: Cost Ledger

Recorded grammar bits, program bits, query bits, candidate attempts, candidate-attempt bits, total bits, elapsed time, and cost ratios versus WGD for every measured system.

### B38-I14: Pre-Hidden Audit

Ran audit mode before hidden opening. It passed rule-count, exponential-space, enumeration-budget, enumerative-baseline, and constraint-absorber checks without opening a hidden seed.

### B38-I15: Smoke Run

Ran the smaller public smoke measurement. WGD and the constraint absorber solved perfectly; the four enumerators each scored 0.25 HFA by abstaining correctly only on out-of-span cases.

### B38-I16: Smoke Token

Smoke assigned `WGD_ABSORBED_BY_CONSTRAINT_DISCOVERY`. This was the expected honest shape: enumeration was defeated, but the linear constraint solver was not.

### B38-I17: Final Freeze Check

Recompiled `code/wgd0_b38_hard_domain.py` after smoke and before hidden open. No code changes were made after this point.

### B38-I18: Hidden Open

Opened the final hidden seed once with the default B38 hidden configuration: 8 worlds, 32 cases per world, 8,000 enumerated candidates per case.

### B38-I19: Hidden Metrics

Hidden metrics matched smoke: WGD HFA 1.0, composition 1.0, repair 1.0, abstention 1.0. Enumerators remained at 0.25 HFA. The generic constraint solver also scored 1.0.

### B38-I20: Terminal Verdict

Enumeration was finally made genuinely expensive, but the hard-domain win was absorbed by ordinary GF(2) constraint discovery at `0.9673932788374205x` WGD cost. The adversary is not won over.

## Final Token

```text
WGD_ABSORBED_BY_CONSTRAINT_DISCOVERY
```

## Interpretation

B38 answers the supervisor's hard-domain question sharply:

```text
A 64-rule exponential domain can defeat genuine brute enumeration on CPU, but the moment the structure is made learnable and executable, an ordinary constraint solver can absorb it.
```

This is not a schema/binding repeat. It is a stronger negative result. The previous worlds were too small. This world is large enough to kill enumeration, but the only cheap positive route is an ordinary algebraic solver. Therefore no WGD signal is claimed.

## Next Gate

Per supervisor check-in #35, the correct next move if B38 fails is Option C: write the methodology result. The absorption ladder is the surviving contribution.
