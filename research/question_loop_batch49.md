# Q-Loop Batch 49: Second Terminal Adversarial Gate Review

**Date:** 2026-07-08  
**Role:** fresh adversarial reviewer with zero prior project context  
**Iterations:** I435-I448  
**Status:** second terminal gate review of `research/methodology_paper.md` after W-Loop B41

Two invariants held fixed:

1. Swing for the home run: paradigm-shifting or nothing.
2. The loop only stops on a won-over adversary.

## Grounding

I started with the required review surface:

- `research/methodology_paper.md`
- `experiments/b40_positive_control_measurement.json`
- `experiments/frameseed0_b28_hidden_hfa.json`
- `experiments/frameseed_sheets0_b31_hidden_hfa.json`
- `experiments/wgd0_b37_hidden_measurement.json`
- `experiments/wgd0_b38_hidden_measurement.json`
- `code/b40_positive_control.py`
- `code/frameseed_sheets0_measurement.py`
- `code/wgd0_measurement.py`
- `research/STATUS.md`
- `experiments/EXPERIMENTS.md`
- `experiments/ledger.jsonl`
- `research/question_loop_batch48.md`
- `research/dual_loop_supervisor_checkin_38.md`

Then I checked the surrounding repo surface that bears on the claims: `README.md`, `research/VISION.md`, `research/work_loop_batch41.md`, the FrameSeed and WGD token-assignment harnesses, `code/wgd0_b38_hard_domain.py`, the relevant harness tests, repo-wide searches for stale `SIGNAL`/native-status language, and the frozen implementation hashes inside the artifacts. I excluded generated temp/cache trees from the scientific source surface.

Validation run in this review:

```text
python code\b40_positive_control.py --output .\tmp_b49\b40_b49_check.json
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'; python -m pytest code\test_wgd0_harness.py code\test_frameseed_sheets0_harness.py code\test_frameseed0_harness.py -q
python code\wgd0_b38_hard_domain.py --mode audit --output .\tmp_b49\wgd0_b38_b49_audit.json
```

Observed:

- B40 rerun emitted `B40_POSITIVE_CONTROL_ABSORBED_BY_FULL_TARGET_CLASS_PBE`.
- B40 full-target PBE hidden HFA was 1.0 with cost ratio 2.8376722817764164.
- B40 component-erasure drop was 25.0 percentage points.
- Harness tests: 25 passed.
- B38 prehidden audit passed.
- Frozen implementation hashes for C1-C4 matched the current checkout.
- Full SHEETS-0 hidden rerun was not attempted because the stored artifact represents 15,728,640 hidden queries scored per system; I used the artifact, code path, tests, and hash checks instead.

## Executive Hostile Verdict

The adversary is won over at the current claim ceiling.

This is not a broad acceptance of an independently validated field standard. It is acceptance of the revised paper as what it now claims to be: a methodology proposal with internal negative case evidence and an absorbed positive-control attempt used as a residual-risk demonstration.

B48 found three concrete defects:

1. C2 SHEETS-0 status inflation.
2. C3 WGD toy status inflation.
3. B40 circular positive-control signal.

B41 actually fixed those defects in the paper, code, and artifacts:

- C2 typed PBE/CEGIS/data-wrangling/library rows are now described as `capability_mode_scored`, not native typed-baseline execution.
- C3 `pbe_cegis` is now described as the same shared public-feedback role-inference family, not an independent native PBE/CEGIS implementation.
- B40 now includes `full_target_class_pbe`, charges all 1320 candidates, emits absorption, and no longer claims a positive-control signal.

The remaining attacks find release-hygiene and validation-boundary issues, not a terminal contradiction in the paper's own claim. A hostile reviewer can still ask for external validation, a real fair positive control, machine-generated status ledgers, and fresher repo indexes. The paper already admits those as missing or keeps its claim below them.

Final token:

```text
Q_LOOP_B49_ADVERSARY_WON_OVER_ACCEPT_AT_CURRENT_CLAIM_CEILING
```

## I435: Attack B41's Claimed Fix Directly

### Previous Position Under Attack

B41 says it fixed B48's three fatal defects. Treat that as self-serving until proven against the code and artifacts.

### Attack

The old failure mode was not subtle: B40 had called a target-class search a claimed learner, withheld the full version from the absorber roster, and emitted a roster-relative `SIGNAL`. C2 and C3 had promoted weaker evidence modes into native status. If any one of those defects remains, the paper still fails its own method.

Current checkout evidence:

- `research/methodology_paper.md` source map now includes `STATUS.md`, `EXPERIMENTS.md`, and `ledger.jsonl`.
- C2 ledger rows now use `capability_mode_scored` for typed CEGIS/PBE, data wrangling, library/active/nuisance rows.
- C3 ledger says `pbe_cegis` is the same shared `infer_role_model()` family with a different source string.
- `code/b40_positive_control.py` includes `full_target_class_pbe` in `SYSTEMS` and the manifest absorber roster.
- `experiments/b40_positive_control_measurement.json` emits `B40_POSITIVE_CONTROL_ABSORBED_BY_FULL_TARGET_CLASS_PBE`.

### Verdict

```text
B48_THREE_DEFECTS_APPEAR_MATERIALLY_FIXED
```

## I436: Attack I435 For Being Too Easily Satisfied On C2

### Previous Position Under Attack

I435 accepts that C2 was fixed because the paper relabeled typed rows. That may be too quick. The artifact itself still says typed PBE/CEGIS/library/data-wrangling solve and marks domain absorptions true.

### Attack

`experiments/frameseed_sheets0_b31_hidden_hfa.json` still contains:

- `domain_absorptions.pbe = true`
- `domain_absorptions.data_wrangling = true`
- `domain_absorptions.typed_cegis = true`
- `domain_absorptions.library_learning = true`
- token interpretation: typed PBE/CEGIS/library baselines solve under the same information

The code confirms this is mode scoring, not independent native typed learners:

```text
mode = system_mode(system)
hit = mode_solves(mode, query.operation)
score = 1.0 if hit else 0.0
```

So a machine reader of only the JSON can still overread the artifact.

### Counterattack

The paper no longer overreads it. It explicitly says those rows are `capability_mode_scored`, cites the code path and warning docs, and states that native typed-prior-art execution remains unimplemented. B48's fatal issue was paper-level status inflation. That specific overclaim is gone.

### Verdict

```text
C2_ARTIFACT_REMAINS_STATUS_AMBIGUOUS_BUT_PAPER_NO_LONGER_OVERCLAIMS_IT
```

## I437: Attack I436 For Missing The Actual C2 Terminal Basis

### Previous Position Under Attack

I436 says C2 remains dangerous because artifact token evidence has true typed absorptions. That is a residual packaging risk, but it does not address the terminal basis the paper now relies on.

### Attack

The revised paper's C2 claim is not that native typed PBE or native typed CEGIS solved the domain. The claim is narrower:

```text
binding-only HFA = 1.0
packet-erasure drop = 0.0 pp
```

Those fields are present in the artifact and arise from the code's binding-only accounting. The paper says the typed packet advantage was absorbed by schema/binding and granted typed pipeline substrate. It forbids upward narration into native typed baseline execution.

### Verdict

```text
C2_NEGATIVE_CONCLUSION_STABLE_AFTER_RELABELING
```

## I438: Attack I437 For Letting Schema Binding Smuggle An Oracle

### Previous Position Under Attack

I437 accepts schema/binding as the C2 terminal basis. A hostile reviewer should ask whether this is just an oracle in nicer words.

### Attack

In `code/frameseed_sheets0_measurement.py`, `WorldOracle` has exact role columns and implements the correct typed operations. The binding-only counter increments for every hidden query:

```text
binding_only_total += 1
binding_only_hits += 1
```

That is not a learned schema matcher. It is an upper-bound/granted-binding demolition: if exact bindings and typed operators are granted, the packet adds no value. Calling that `native_executable` could still be too strong.

### Counterattack

The paper's public wording now matches that limitation. It says the packet was unnecessary once exact bindings and typed pipeline substrate were granted, and it warns that native typed-prior-art execution remains unimplemented. Since no positive C2 claim survives, a granted-substrate upper bound is enough to kill the supplied-packet story.

### Verdict

```text
C2_SCHEMA_BINDING_IS_GRANTED_SUBSTRATE_NOT_DISCOVERY_SIGNAL_AND_THE_PAPER_SAYS_SO
```

## I439: Attack I438 For Moving Past C3 Too Fast

### Previous Position Under Attack

I438 says C2 is contained. C3 could still be fatal if the paper preserves the old independent-PBE story.

### Attack

`code/wgd0_measurement.py` still constructs three non-majority systems with the same function:

```text
"wgd_grammar": infer_role_model(...)
"schema_binding": infer_role_model(...)
"pbe_cegis": infer_role_model(...)
```

The artifact still has `absorptions.pbe = true` and `absorptions.cegis = true`. A naive table can still sell the same role-inference program under multiple absorber labels.

### Counterattack

The current paper does not sell it that way. It says the `pbe_cegis` label is the same shared `infer_role_model` family, not independent native PBE/CEGIS, and that cheaper public-feedback role inference is what absorbs. That is exactly the correction B48 demanded.

### Verdict

```text
C3_LABEL_REUSE_REMAINS_IN_ARTIFACT_BUT_NOT_IN_THE_REVISED_CLAIM
```

## I440: Attack I439 For Underplaying The B40 Circularity

### Previous Position Under Attack

I439 says C2 and C3 are fixed. B40 was more central: it was the attempted answer to the rejection-machine criticism.

### Attack

The B40 code still shows the circularity:

```text
claimed = learn_best_candidate(train, candidates)
full_pbe = learn_best_candidate(train, candidates)
```

The target is still moved out of the budgeted PBE prefix when necessary:

```text
if target_index < BUDGETED_PBE_CANDIDATES:
    target_index += BUDGETED_PBE_CANDIDATES
```

That means the old positive-control construction remains engineered against the budgeted absorber.

### Counterattack

B41 no longer uses that as signal. The manifest now includes `full_target_class_pbe`; the artifact records full PBE hidden HFA 1.0; cost ratio is 2.8376722817764164; token is absorption; residual risk is not high because the once-omitted absorber is now run. The paper calls B40 a residual-risk demonstration, not a fair positive control.

### Verdict

```text
B40_CIRCULARITY_NOW_KILLS_B40_RATHER_THAN_THE_PAPER
```

## I441: Attack I440 For Accepting A Paper With No Fair Positive Control

### Previous Position Under Attack

I440 says B40 is fixed because it is demoted. But maybe a methodology paper needs at least one fair `SIGNAL` case to prove it is not just a rejection machine.

### Attack

The revised evidence base is all absorptions:

- C1 absorbed.
- C2 absorbed.
- C3 absorbed.
- C4 absorbed.
- B40 attempted positive control absorbed.

No case demonstrates that the ladder can fairly emit signal.

### Counterattack

The paper now admits this. It does not claim a fair positive control. It claims a protocol with internal negative evidence and a residual-risk demonstration. The value proposition is not "we validated yes/no calibration." It is "we prevented upward narration, including our own attempted positive-control overclaim." That is a narrower paper, but it is internally consistent.

### Verdict

```text
NO_FAIR_POSITIVE_CONTROL_IS_A_SCOPE_LIMIT_NOT_A_TERMINAL_CONTRADICTION
```

## I442: Attack I441 For Letting Self-Audit Validate Anti-Self-Deception

### Previous Position Under Attack

I441 accepts the narrower scope. A hostile conference reviewer will say self-audited synthetic failures cannot validate an anti-self-deception methodology.

### Attack

The same project created the domains, absorbers, cost ledgers, adversarial loops, and paper. There is no third-party rerun, blinded absorber selection, external claim reanalysis, or independent reviewer before hidden opening. This can document internal discipline; it cannot validate a field-level standard.

### Counterattack

The paper says exactly that in Section 8. It explicitly bans the claim that the methodology has been independently validated as a field-level standard and lists the missing validation items. The abstract and submission position stay within protocol-proposal language.

### Verdict

```text
SELF_AUDIT_LIMIT_IS_DISCLOSED_AND_CLAIM_CEILING_HOLDS
```

## I443: Attack I442 With Repo-Level Staleness

### Previous Position Under Attack

I442 says the paper is honest. The surrounding repo still has stale public surfaces.

### Attack

`README.md` still describes FrameSeed as the current status and says there is no active mainline, while `research/STATUS.md` says WGD is killed and the methodology paper is the moonshot deliverable. `experiments/EXPERIMENTS.md` and `experiments/ledger.jsonl` do not include WGD B37/B38 or B40/B41 entries, even though the paper now uses those as central artifacts.

A hostile reviewer can use this to say the public artifact package is not fully synchronized with the paper's live claim surface.

### Counterattack

This is real release hygiene, but not a terminal paper contradiction. The paper's claims are grounded in the specific JSON artifacts, code, STATUS warning, prior review, and supervisor directive. The stale README/index do not assert that WGD or B40 had a different terminal token; they are incomplete public navigation surfaces.

### Verdict

```text
RELEASE_INDEX_STALENESS_SHOULD_BE_FIXED_BUT_DOES_NOT_MOVE_THE_TERMINAL_TOKEN
```

## I444: Attack I443 For Dismissing Machine-Ledger Weakness

### Previous Position Under Attack

I443 calls ledger staleness release hygiene. But the methodology itself emphasizes ledgers, so missing machine-ledger entries could be central.

### Attack

`experiments/ledger.jsonl` is described as the machine-readable append-only ledger. It contains the SHEETS warning B41 needed, but it does not contain WGD or B40/B41 terminal records. The paper's case ledgers are hand-written tables with artifact-bound citations, not machine-generated status rows.

A stricter version of the methodology would require every absorber status to be machine-readable inside the artifact or ledger.

### Counterattack

The current paper does not claim that such a machine-ledger system already exists. It defines the method and fills status ledgers in the paper with citations to artifacts, code paths, and warning docs. B48 asked for artifact-bound status citations; B41 supplies them. A future artifact schema should encode statuses directly, but the absence of that schema is not an overclaim in the current paper.

### Verdict

```text
MACHINE_LEDGER_SCHEMA_IS_FUTURE_HARDENING_NOT_CURRENT_REJECTION
```

## I445: Attack I444 With Provenance Drift

### Previous Position Under Attack

I444 says the cited artifacts are enough. That only holds if the artifacts still bind to the current code and can be checked.

### Attack

If implementation hashes drifted, the paper would be relying on stale artifacts. If B40 could not be reproduced, the core B41 fix would be suspect.

### Evidence

The frozen hashes in C1-C4 matched the current checkout for the cited code/spec/test files. B40 reran and emitted the same terminal token, HFA, cost ratio, and component-erasure drop. The relevant harness tests passed. The B38 audit rerun passed.

B40 is weaker than C1-C4 on provenance because it does not include an `implementation_hashes` block in the JSON, but it is a cheap residual-risk demonstration and was reproduced in this review.

### Verdict

```text
PROVENANCE_ATTACK_FAILS_FOR_TERMINAL_CASES_WITH_MINOR_B40_SCHEMA_HARDENING_REMAINING
```

## I446: Attack I445 On Cost Accounting

### Previous Position Under Attack

I445 says artifacts reproduce. Cost accounting can still be arbitrary enough to change absorption conclusions.

### Attack

The bit ledgers are homemade. The <=4x absorption boundary is a project convention. C4's constraint solver ratio is 0.9673932788374205, which is close enough that serialization changes could affect rhetorical strength. B40's full PBE ratio is inside the boundary but depends on candidate-attempt bit pricing.

### Counterattack

The paper already separates terminal stability from rhetorical strength. The important terminals do not depend on tiny perturbations:

- C1 has multiple perfect absorbers.
- C2 is killed by binding-only HFA 1.0 and packet-erasure drop 0.0, not by typed capability bit prices.
- C3 schema role inference is about 0.088x WGD cost.
- C4 constraint solving has HFA 1.0 and is comparable cost, not an expensive rescue.
- B40 full PBE is 2.837x under the recorded convention, inside the declared <=4x boundary.

A reviewer can dispute the boundary as a field standard, but the paper does not claim that boundary has external authority yet.

### Verdict

```text
COST_LEDGER_IS_LOCAL_BUT_CLAIM_LIMITS_AND_TERMINAL_STABILITY_ARE_DISCLOSED
```

## I447: Attack I446 On Novelty And Home-Run Ambition

### Previous Position Under Attack

I446 says the paper is internally consistent. A final hostile move is to call the whole thing obvious: preregistration, hidden tests, fair baselines, ablations, and red-team review are known.

### Attack

If the contribution is merely "try boring explanations first," it is not paradigm-shifting. The evidence base is self-authored and negative. No fair positive control remains. Why should a conference treat this as more than a lab notebook about being cautious?

### Counterattack

The paper does not claim primitive novelty for preregistration, hidden tests, or baselines. It claims procedural composition: absorber roster, equal-information affordance map, all-in cost rule, hidden-open manifest, terminal token, status ledger, and residual-risk ceiling as a single decision procedure. The B40/B41 episode is a concrete demonstration of the procedure's value: it caught the manuscript's own attempt to use roster-relative language to preserve a positive signal.

That is enough for a methodology proposal. It is not enough for field-level validation, and the paper does not claim field-level validation.

### Verdict

```text
NOVELTY_IS_PROCEDURAL_COMPOSITION_AND_THE_CURRENT_SCOPE_CAN_SUPPORT_THAT
```

## I448: Terminal Synthesis

### Previous Position Under Attack

I447 says the paper can stand as a methodology proposal. The terminal question is whether, after trying to tear it down, the adversary is genuinely won over.

### Attack

I tried the available rejection routes:

1. Reopen B48's C2 status-inflation defect.
2. Reopen B48's C3 status-inflation defect.
3. Reopen B48's B40 circular-positive-control defect.
4. Attack the absence of a fair positive control.
5. Attack self-audited synthetic validation.
6. Attack stale repo indexes and machine-ledger incompleteness.
7. Attack provenance drift.
8. Attack cost accounting.
9. Attack novelty.

The first three fail because B41 actually fixed them. The next six produce real caveats, but the paper now places those caveats inside its own claim ceiling instead of hiding them.

### Final Verdict

I am won over, but only at the exact claim ceiling the paper now states.

Accepted claim:

```text
The absorption ladder is a roster-relative methodology proposal, supported by this project's internal record of four absorbed synthetic case studies and one absorbed attempted positive control, showing how the protocol prevents this project from narrating attractive artifacts upward into AI discovery claims.
```

Rejected claim:

```text
The absorption ladder has been independently validated as a field-level standard, has demonstrated calibrated positive signal detection, or has exhausted all ordinary explanations for AI discovery claims in general.
```

The revised paper says the first thing and explicitly refuses the second. That is why the terminal adversary is won over.

Final token:

```text
Q_LOOP_B49_ADVERSARY_WON_OVER_ACCEPT_AT_CURRENT_CLAIM_CEILING
```

## Required Cleanup Before Public Release

These are not terminal blockers for the paper's current claim, but they are the remaining hostile-review footguns:

1. Update `README.md` so the top-level current status matches `research/STATUS.md` and the methodology-paper pivot.
2. Update `experiments/EXPERIMENTS.md` and `experiments/ledger.jsonl` with WGD B37/B38 and B40/B41 entries, or stop calling them the current experiment index / machine ledger.
3. Add an `implementation_hashes` block to the B40 artifact for consistency with the other hidden artifacts.
4. Consider changing "the ladder caught the paper overclaiming" to "the adversarial ladder workflow caught the paper overclaiming," because B40's first application did not self-correct until B48.
5. If this becomes a public submission, add a small explicit table separating: terminal evidence, supporting evidence, admitted missing validation, and forbidden claims.

## Final Adversarial Position

The paper stands because it finally obeys its own standard. The strongest hostile sentence after B41 is no longer fatal:

```text
This is not an independently validated field standard and has no fair positive control, but it is an internally consistent methodology proposal whose own evidence is bounded to absorbed cases and residual-risk demonstration.
```

That sentence is acceptable because it is now the paper's own position.