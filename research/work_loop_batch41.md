# W-Loop Batch 41: Ledger Truth And Positive-Control Demotion

Date: 2026-07-08
Role: W-Loop worker
Status: B48 defects fixed in the workspace
Target: `research/methodology_paper.md`, `code/b40_positive_control.py`, `experiments/b40_positive_control_measurement.json`

Two invariants held:

1. Swing for the home run: the method must change how AI discovery claims are reviewed, not merely advise caution.
2. The loop only stops on a won-over adversary: B41 is written to give Q-Loop B49 no easy status-inflation or circular-positive-control rejection.

## Supervisor Directive

Supervisor check-in #38 accepted Q-Loop B48's three defects:

1. C2 SHEETS-0 status inflation: typed PBE/CEGIS/data-wrangling/library rows were capability-mode scored, not native typed-baseline execution.
2. C3 WGD toy status inflation: `pbe_cegis` is cheaper public-feedback role inference using the shared `infer_role_model()` function, not independent native PBE/CEGIS.
3. B40 positive control circularity: full target-class PBE over all 1320 candidates is the claimed search procedure viewed as an absorber and must be run.

## Deliverables

| Artifact | Status | Purpose |
|---|---|---|
| `code/b40_positive_control.py` | revised | adds `full_target_class_pbe` as a declared absorber and charges all 1320 candidate attempts |
| `experiments/b40_positive_control_measurement.json` | regenerated | records absorption by full target-class PBE |
| `research/methodology_paper.md` | revised | relabels C2/C3 ledgers, demotes B40, adds artifact-bound status citations |
| `research/work_loop_batch41.md` | added | 20-iteration work log |

## B40/B41 Result

The revised runner includes `full_target_class_pbe` in the absorber roster.
The regenerated artifact reports:

```text
terminal_token = B40_POSITIVE_CONTROL_ABSORBED_BY_FULL_TARGET_CLASS_PBE
claimed_hidden_hfa = 1.0
full_target_class_pbe_hidden_hfa = 1.0
full_target_class_pbe_cost_ratio_vs_claimed = 2.8376722817764164
component_erasure_drop_pp = 25.0
```

The planted interaction remains real, but it is absorbed by ordinary full target-class PBE inside the <=4x boundary.
B40 is no longer a positive-control signal.
It is a residual-risk demonstration: the ladder catches even its own attempted positive control.

## Twenty Iterations

| Iteration | Attack handled | Revision action | Result |
|---:|---|---|---|
| 1 | B48 fatal defect 1: C2 overclaims native typed baselines | Re-read `STATUS.md`, `EXPERIMENTS.md`, `ledger.jsonl`, and `code/frameseed_sheets0_measurement.py`. | Confirmed B31 used capability-mode scoring for typed rows. |
| 2 | C2 paper row said `native executable` for typed PBE/CEGIS | Relabeled typed CEGIS/PBE as `capability_mode_scored`. | Paper no longer claims native learned typed PBE/CEGIS execution. |
| 3 | C2 paper row said data-wrangling/domain tools were native | Relabeled data-wrangling/domain tools as `capability_mode_scored`. | The absorption remains schema/binding plus capability-mode pipeline scoring. |
| 4 | C2 paper row said library/active/nuisance were native | Relabeled library/active/nuisance as `capability_mode_scored`. | No typed-frame signal is narrated from mode-scored rows. |
| 5 | C2 absorption might be weakened by relabeling | Preserved binding-only HFA 1.0 and packet-erasure drop 0.0 in the C2 ledger and abstract. | The negative result still holds on the native schema/binding absorber. |
| 6 | B48 fatal defect 2: C3 `pbe_cegis` overclaims independent PBE/CEGIS | Re-read `code/wgd0_measurement.py` and the B37 artifact. | Confirmed `wgd_grammar`, `schema_binding`, and `pbe_cegis` share `infer_role_model()`. |
| 7 | C3 ledger sold `pbe_cegis` as independent native PBE/CEGIS | Rewrote the C3 row as shared public-feedback role inference with a different source string. | The paper now claims cheaper role inference, not independent PBE/CEGIS. |
| 8 | C3 abstract and summaries still said PBE/CEGIS too strongly | Replaced broad PBE/CEGIS wording with shared role-inference wording. | The summary matches the code path. |
| 9 | B48 fatal defect 3: B40 omitted the exact full absorber | Revised `code/b40_positive_control.py` to add `full_target_class_pbe` to `SYSTEMS` and the manifest roster. | The absorber roster now contains the omitted dangerous absorber. |
| 10 | Full PBE needed all 1320 candidates charged | Added `candidate_attempt_bits = len(candidates) * 8` for `full_target_class_pbe`. | Full PBE cost is artifact-bound. |
| 11 | Token still allowed `SIGNAL` | Changed token precedence so full target-class PBE absorption wins before signal. | The regenerated token is `B40_POSITIVE_CONTROL_ABSORBED_BY_FULL_TARGET_CLASS_PBE`. |
| 12 | Artifact needed deterministic regeneration, not hand editing | Ran the revised B40 runner to regenerate JSON. | JSON records HFA 1.0 and 2.8376722817764164x cost ratio for full PBE. |
| 13 | Paper still framed B40 as positive control | Rewrote Section 4 as a residual-risk demonstration. | The paper admits the positive-control failure directly. |
| 14 | Abstract still needed the narrative gate | Rewrote the abstract around four absorptions plus one absorbed positive-control attempt. | The paper becomes stronger by showing the ladder catches its own overclaim. |
| 15 | Source map missed claim-ceiling warnings | Added `STATUS.md`, `EXPERIMENTS.md`, and `ledger.jsonl` to the source map. | The paper now binds itself to the repo's own warning surfaces. |
| 16 | Status ledgers were prose assertions | Added artifact-bound status-source citations to every row in the C1-C4 ledgers. | Each status now points to JSON fields, code functions, or warning docs. |
| 17 | B40 table also needed artifact-bound status | Added source citations for each B40 residual-risk row. | The residual demo is ledgered like the cases. |
| 18 | Paper growth risk | Rewrote the paper compactly instead of layering caveats over B40. | The paper stays short and removes old yes-signal padding. |
| 19 | Claim ceiling risk after no fair positive control | Updated Scope and Submission Position to say protocol with internal evidence and residual-risk demo. | No independent validation or fair-positive-control claim remains. |
| 20 | Narrative gate | Ended with the required one-sentence story: the ladder killed every claim, including its own positive-control attempt. | The fixed paper obeys its own standard. |

## Validation

Commands run:

```text
python code/b40_positive_control.py --output experiments/b40_positive_control_measurement.json
rg -n "SIGNAL|native executable|capability_mode_scored|full_target_class_pbe|infer_role_model|STATUS.md|EXPERIMENTS.md|ledger.jsonl" research/methodology_paper.md code/b40_positive_control.py experiments/b40_positive_control_measurement.json
```

Observed validation:

- B40/B41 token: `B40_POSITIVE_CONTROL_ABSORBED_BY_FULL_TARGET_CLASS_PBE`.
- Claimed hidden HFA: 1.0.
- Full target-class PBE hidden HFA: 1.0.
- Full target-class PBE cost: 14,824 bits.
- Claimed learner cost: 5,224 bits.
- Full target-class PBE cost ratio: 2.8376722817764164.
- Component-erasure drop: 25.0 percentage points.
- C2 typed rows in the paper now use `capability_mode_scored`.
- C3 `pbe_cegis` now says shared public-feedback role inference, not independent native PBE/CEGIS.
- Source map now includes `research/STATUS.md`, `experiments/EXPERIMENTS.md`, and `experiments/ledger.jsonl`.

## Residual Risks For B49

| Risk | Current handling |
|---|---|
| No fair positive control remains | Admitted directly; B40 is demoted to residual-risk demonstration. |
| C2 typed baselines are not native | Ledger says `capability_mode_scored` and cites the warning docs/code. |
| C3 PBE/CEGIS label is reused | Ledger says shared `infer_role_model()` public-feedback role inference. |
| Internal cases are self-authored | Scope section narrows to methodology proposal with internal evidence. |
| External validation absent | Listed as missing validation, not hidden. |

## Narrative Gate

We built a roster-relative protocol for testing AI discovery claims, killed every claim we tested including our own attempt at a positive control, and the methodology's value is that it prevents exactly this kind of overclaim.