# Work-Loop Batch 30: Harden SHEETS-0 Spec + Begin Typed Harness

**Date:** 2026-07-07  
**Role:** W-Loop B30 worker  
**Directive:** Incorporate Q-Loop B37 hardening into `research/frameseed_sheets_0_spec.md`, then begin typed-domain harness implementation in `code/`.  
**CPU only:** yes.  
**Hidden seeds opened:** no.  
**Hidden HFA reported:** no.

## Outcome

B30 hardened SHEETS-0 from a generic typed joins/units spec into a domain-specific absorption trap. The public implementation now covers typed generator scaffolding, packet serialization, provenance, cost ledgers, parser/human-labor ledgers, baseline parity, domain baseline roster, typed enumerability, generator leakage, and token precedence. It does not run hidden evaluation.

Primary artifacts:

- `research/frameseed_sheets_0_spec.md`
- `code/frameseed_sheets0_harness.py`
- `code/test_frameseed_sheets0_harness.py`
- `experiments/frameseed_sheets0_b30_public_audit_smoke.json`
- `experiments/frameseed_sheets0_b30_public_audit.json`

## 20 Iterations

### B30-I1: Grounding

Read the supervisor directive, current SHEETS-0 spec, Q-Loop B37 attacks, Boolean precommit spec, Boolean harness, and Vision. Bound the batch to public pre-hidden hardening plus implementation scaffolding.

### B30-I2: Token Surface

Replaced the Boolean-era SHEETS token list with Q37 domain tokens: relational algebra, unit system, exact-key matching, entity resolution, schema matching, schema binding, PBE, data wrangling, constraint solving, data repair, typed CEGIS, library learning, parser prior, typed Boolean trap, schema-smuggling void, negative, and signal.

### B30-I3: Precedence

Hardened token precedence so void and parser/representation prior fire first, domain-specific absorbers get first refusal before negative, generic Boolean-era absorbers fire only after L3 passes, and signal is last.

### B30-I4: Typed Representation Noncontainment

Added the typed representation-noncontainment certificate: parser inventory, unit grammar, date grammar, ID grammar, string similarity, key features, schema matching, constraint language, action semantics, H0/A0/B0, and role-isomorphism audit.

### B30-I5: Parser And Human-Labor Ledger

Added required separation of public substrate, packet-design labor, frozen-before-hidden surfaces, and hidden-eval-only surfaces. The harness implements `ParserHumanLedger` and audits the declaration.

### B30-I6: Frame/Binding/Program Cost Split

Added `F`, `B_i`, `P_i`, and human/parser cost accounting. The harness implements `BudgetLedger`, `make_budget_ledger`, `audit_budget_recomputation`, and `audit_cost_split`.

### B30-I7: Relational Absorber

Added relational algebra / SQL / pandas-merge first refusal to the spec and baseline roster. Token precedence includes `FRAMESEED_SHEETS0_ABSORBED_BY_RELATIONAL_ALGEBRA` before negative.

### B30-I8: Unit Absorber

Added UCUM-style dimensional oracle, unit-library search, unit-PBE, and unit-error detector requirements. The harness roster includes `unit_system`.

### B30-I9: Entity Absorbers

Added exact-key and entity-resolution baselines. The harness roster includes `exact_key_matching` and `entity_resolution`.

### B30-I10: Schema Matching

Added schema-matching / ontology-alignment first refusal and schema-binding absorption when binding cost explains success. The harness roster includes `schema_matching`; token precedence includes `schema_binding`.

### B30-I11: PBE And Wrangling

Added PROSE-style typed PBE and OpenRefine/Wrangler-style action-history baselines. The harness roster includes `pbe_prose` and `data_wrangling`.

### B30-I12: Constraint Solver And Data Repair

Added declared-constraint executor, constraint learner, finite-domain/SMT solver, data-repair baseline, and action-guard learner requirements. The harness roster includes `constraint_solver` and `data_repair`.

### B30-I13: Typed CEGIS And Library Learning

Added typed enumerability metrics: join candidates, unit-transform candidates, schema bindings, constraint sets, action policies, typed pruning factor, public example version space, and minimum distinguishing counterexamples. The harness implements `audit_enumerability`.

### B30-I14: Generator Leakage

Added typed generator leakage audit over names, indices, row order, value distributions, ID formats, unit symbols, missingness, duplicates, constraint locations, schema ids, packet order, and sibling templates. The harness implements `run_generator_leakage_audit`.

### B30-I15: Goal/Obligation Semantics

Added finite verifier-obligation goal semantics: preservation, transformation, rejection, uncertainty/abstention, repair, and action safety. The harness declares and audits the obligation set.

### B30-I16: Composition And Repair

Added the composition/local-repair gate and required comparison against pipeline, PBE, CEGIS, wrangling, and library baselines. The packet constructor emits a `composition_gate` entry and the ledger charges it as program/composition bits.

### B30-I17: Solvedness And Typed Boolean Trap

Added solvedness audit language and tightened the typed Boolean-trap guard: binary-only labels, one-hot lookups, tiny enumerable typed DSLs, and type-isolated targets cannot carry signal.

### B30-I18: Harness Implementation

Created `code/frameseed_sheets0_harness.py` with typed world generation, public transcript construction, blind packet constructor, canonical JSON serialization, budget/parser ledgers, baseline parity, domain roster, packet-order control, enumerability, generator leakage, manifest audit, token evidence, and golden token controls.

### B30-I19: Tests

Created `code/test_frameseed_sheets0_harness.py` covering generator decoys/RNG streams, blind constructor/provenance/serialization, budget split, baseline parity failure on denied fields, domain roster, leakage smoke, token precedence, golden controls, and top-level public audit.

### B30-I20: Validation

Validation run:

```text
python -m py_compile code/frameseed_sheets0_harness.py
python code/frameseed_sheets0_harness.py --dry-run-worlds 400 --leakage-threshold 0.25 --output experiments/frameseed_sheets0_b30_public_audit_smoke.json
python code/frameseed_sheets0_harness.py --dry-run-worlds 1000 --leakage-threshold 0.08 --output experiments/frameseed_sheets0_b30_public_audit.json
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'; python -m pytest code/test_frameseed_sheets0_harness.py -q
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'; python -m pytest code/test_frameseed0_harness.py -q
```

Results:

```text
py_compile: pass
public audit smoke: pass, no hidden HFA, no performance runs
public audit stricter sample: pass, no hidden HFA, no performance runs, worst leakage metric 0.014158279193434313 <= 0.08
new SHEETS tests: 7 passed
existing Boolean harness tests: 10 passed
```

Pytest needed `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` because a globally installed `dandi` pytest plugin attempted to write outside the workspace under `AppData` before test collection. With plugin autoload disabled, tests ran normally. Pytest also warned that `.pytest_cache` could not be written in this workspace; that did not affect test outcomes.

## Current Boundary

B30 did not implement actual L3 hidden performance, domain baselines, or hidden scoring. It implemented the pre-hidden audit/control surface required before any hidden run. The next batch must decide whether to deepen executable baselines first or proceed toward a freeze/hidden protocol only after the domain absorbers are real enough to move the token.