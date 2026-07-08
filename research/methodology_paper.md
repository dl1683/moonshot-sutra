# The Absorption Ladder: Roster-Relative Tests for AI Discovery Claims

Draft date: 2026-07-08.
Status: W-Loop B41 revision after Q-Loop B48 and supervisor check-in #38.
Core claim: an AI discovery claim is credible only relative to a predeclared absorber roster, equal-information affordance map, all-in cost rule, hidden-open manifest, and explicit residual-risk statement.
Claim ceiling: this paper does not certify that all ordinary explanations have failed; it documents which ordinary explanations were made executable, which won, which failed, and which remain untested.

## Abstract

High hidden accuracy is not enough to claim that an AI system discovered a rule, frame, grammar, strategy, circuit, or world model.
A compact or interpretable artifact is not enough either.
The missing question is whether the discovery explanation survives first refusal by ordinary explanations under equal information and all-in accounting.
Those explanations include representation priors, parser and substrate priors, finite teaching, active queries, program synthesis, schema binding, domain tools, library learning, nuisance oracles, generator fingerprints, human-authored substrate, and post-hoc compression.

The absorption ladder is a roster-relative protocol for predeclaring those explanations, making them executable or labeling why they are not, assigning terminal tokens after one hidden opening, and bounding the public claim to the evidence actually tested.
The contribution is not a universal proof that the strongest possible boring explanations have failed.
The contribution is a decision procedure for saying exactly what survived: claim, roster, equal-information map, status ledger, cost rule, hidden-open manifest, result token, and residual absorber risk.

The project record contains four absorbed internal case studies and one absorbed positive-control attempt.
FrameSeed Boolean reached hidden HFA 1.0, but exact teaching/search also reached 1.0.
FrameSeed SHEETS-0 reached typed hidden HFA 1.0, but binding-only HFA was 1.0 and packet erasure dropped 0.0 points; the typed PBE/CEGIS/data-wrangling/library rows are capability-mode scored, not native typed-baseline executions.
WGD-0 toy grammar reached HFA 1.0, but cheaper public-feedback role inference absorbed it; the `pbe_cegis` label is the same shared `infer_role_model` family, not an independent native PBE/CEGIS implementation.
WGD-0 hard domain defeated flat enumerators on the measured slice, but a GF(2) constraint solver matched HFA 1.0 at comparable all-in cost.
B40 tried to add a positive control; once full target-class PBE over all 1320 candidates was restored to the absorber roster, it also reached HFA 1.0 at 2.8377x claimed cost and emitted `B40_POSITIVE_CONTROL_ABSORBED_BY_FULL_TARGET_CLASS_PBE`.

That failure strengthens the paper.
The ladder caught the paper overclaiming its own positive-control attempt.
The result is a stricter methodology proposal with internal negative evidence and a residual-risk demonstration, not a completed field validation.

## Source Map

| Source | Role |
|---|---|
| `research/dual_loop_supervisor_checkin_38.md` | B41 directive: fix C2, C3, and B40 defects |
| `research/question_loop_batch48.md` | terminal adversarial attacks addressed here |
| `research/STATUS.md` | public claim ceiling warning for SHEETS-0 typed baselines |
| `experiments/EXPERIMENTS.md` | human-readable SHEETS warning and experiment index |
| `experiments/ledger.jsonl` | machine ledger warning not to claim native typed-baseline execution |
| `research/METHODOLOGY_TEMPLATE.md` | reusable token, ladder, cost, and hidden-open vocabulary |
| `experiments/frameseed0_b28_hidden_hfa.json` | FrameSeed Boolean hidden artifact |
| `experiments/frameseed_sheets0_b31_hidden_hfa.json` | FrameSeed SHEETS-0 hidden artifact |
| `experiments/wgd0_b37_hidden_measurement.json` | WGD-0 toy grammar hidden artifact |
| `experiments/wgd0_b38_hidden_measurement.json` | WGD-0 hard-domain hidden artifact |
| `experiments/b40_positive_control_measurement.json` | B40/B41 residual-risk demonstration artifact |
| `code/frameseed_sheets0_measurement.py` | C2 capability-mode scoring implementation |
| `code/wgd0_measurement.py` | C3 shared `infer_role_model` implementation |
| `code/b40_positive_control.py` | B40/B41 full-PBE absorber runner |

## 1. The Claim Being Made

The old version implied that a finite paper could exhaust the strongest ordinary explanations.
That was too absolute.
The revised claim is roster-relative:

```text
An AI discovery claim is credible only relative to a predeclared absorber roster,
the equal-information affordance map for that roster, the all-in cost rule, the
hidden-open manifest, and an explicit residual-risk statement for omitted,
proxy, capability-mode, or untested absorbers.
```

A positive result does not mean discovery in the abstract.
It means no declared absorber matched the threshold inside the frozen information, status, and cost boundary.
A missing dangerous native absorber cannot be narrated upward into signal.
It emits `INCONCLUSIVE`, lowers the claim ceiling, or, if run and successful, absorbs.

## 2. Terminal Tokens And Stopping Rule

A hidden opening emits exactly one token.
The token is a decision rule, not a narrative mood.

| Token class | Meaning | Public claim ceiling |
|---|---|---|
| `VOID` | leakage, hidden mutation, subjective semantics, or baseline asymmetry broke the protocol | no scientific claim from that opening except that the protocol failed |
| `TRAP` | the domain was degenerate, tiny, leaked by representation, or not a real test of the target function | the domain did not test the intended claim |
| `ABSORBED_BY_X` | ordinary explanation X matched threshold within the cost rule | the apparent discovery is carried by X at the tested scale |
| `NEGATIVE` | the claimed system missed its functional gate | the claimed system failed its own test |
| `INCONCLUSIVE` | required metric, absorber, status, or cost field is missing | the claim remains unproven |
| `SIGNAL` | claimed system passed and declared absorbers failed under the manifest | narrow signal relative to roster, scale, status ledger, cost rule, and residual risk |

Token precedence is conservative: void, trap, representation or substrate absorption, any declared ordinary absorber inside the cost boundary, negative, then signal.
Mixed evidence cannot be narrated upward.
A result with a missing dangerous native absorber is not signal.

A run may stop only when four fields are complete: claim surface, roster rationale, status ledger, and residual-risk statement.
The status ledger uses these labels: `native_executable`, `proxy_absorber`, `capability_mode_scored`, `formal_lower_bound`, `non_native_omitted`, and `untested_roster_entry`.
High residual risk means a plausible native absorber was not run and could match the claimed system; no strong positive claim is allowed.

## 3. Method In One Page

1. Define the discovery claim as a function, not as vibes around an artifact.
2. Precommit terminal tokens and token precedence.
3. Build a roster of ordinary explanations native to the domain.
4. Map equal information: parser, type system, examples, counterexamples, query channel, verifier, canonicalizer, operation grammar, bindings, and human substrate.
5. Freeze all-in cost accounting before hidden opening.
6. Freeze manifest hashes, public/smoke seeds, hidden seed rule, thresholds, baselines, and scorer.
7. Open hidden once and forbid post-hidden constructor, scorer, baseline, timeout, parser, token, or audit changes under that seed.
8. Emit token, status ledger, cost sensitivity, and residual-risk statement.

The delta over standard rigor is composition: ordinary explanations receive first refusal under equal executable affordances and all-in cost, and every public claim is bounded by the roster and residual risk.

## 4. B40 Residual-Risk Demonstration

B40 was intended to be a positive control.
B48 showed the construction was circular: the claimed `interaction_learner` was full target-class best-candidate search, while the declared PBE absorber was the same search truncated to the first 128 candidates.
B41 restores the omitted absorber.

Artifact: `experiments/b40_positive_control_measurement.json`.
Runner: `code/b40_positive_control.py`.
Token: `B40_POSITIVE_CONTROL_ABSORBED_BY_FULL_TARGET_CLASS_PBE`.
Runtime: 0.089586 seconds on the recorded run.

Domain: 12-bit inputs, 192 public training examples, 512 hidden cases, 1320 three-feature interaction candidates, target form `(x_i AND x_j) XOR x_k XOR bias`, threshold 0.98 hidden HFA.

| System | Hidden HFA | Status | Source | Claim effect |
|---|---:|---|---|---|
| `claimed_interaction_learner` | 1.0 | claimed target-class search | `experiments/b40_positive_control_measurement.json:system_summary.claimed_interaction_learner` | passes but does not get signal |
| `full_target_class_pbe` | 1.0 | native_executable absorber over all 1320 candidates | `experiments/b40_positive_control_measurement.json:token_evidence.full_target_class_pbe_absorbs` | winning absorption |
| `budgeted_pbe_probe` | 0.7578125 | native_executable but budget-limited prefix probe | `experiments/b40_positive_control_measurement.json:system_summary.budgeted_pbe_probe` | failed prefix probe; not enough for signal |
| `single_bit_prior` | 0.75 | native_executable simple prior | `experiments/b40_positive_control_measurement.json:system_summary.single_bit_prior` | failed |
| `pair_conjunction_prior` | 0.583984375 | native_executable low-order prior | `experiments/b40_positive_control_measurement.json:system_summary.pair_conjunction_prior` | failed |
| `lookup_memorizer` | 0.498046875 | native_executable memorization control | `experiments/b40_positive_control_measurement.json:system_summary.lookup_memorizer` | failed despite train HFA 1.0 |
| `majority_label` | 0.498046875 | native_executable simple prior | `experiments/b40_positive_control_measurement.json:system_summary.majority_label` | failed |
| `random_interaction_probe` | 0.498046875 | negative control | `experiments/b40_positive_control_measurement.json:system_summary.random_interaction_probe` | failed |

The full PBE absorber costs 14,824 bits versus 5,224 bits for the claimed learner, a ratio of 2.8376722817764164.
That is inside the <=4x absorption boundary.
Component erasure still drops 25.0 points, so the planted interaction is real, but it is real in a target class that ordinary full PBE searches directly.
B40 is therefore a residual-risk demonstration: the ladder catches even its own positive-control attempt.

## 5. Case Study Summary

| Case | Artifact | Claimed success | Winning token | What the paper may claim |
|---|---|---|---|---|
| C1 FrameSeed Boolean | `experiments/frameseed0_b28_hidden_hfa.json` | L3 hidden HFA 1.0 | `FRAMESEED_T3_ABSORBED_BY_TEACHING_DIMENSION` | hidden success was absorbed by finite teaching/search |
| C2 FrameSeed SHEETS-0 | `experiments/frameseed_sheets0_b31_hidden_hfa.json` | L3 typed hidden HFA 1.0 | `FRAMESEED_SHEETS0_ABSORBED_BY_SCHEMA_BINDING` | typed packet advantage was absorbed by schema/binding; typed pipelines are capability-mode scored |
| C3 WGD-0 Toy Grammar | `experiments/wgd0_b37_hidden_measurement.json` | WGD HFA 1.0 | `WGD_ABSORBED_BY_SCHEMA_OR_BINDING_DISCOVERY` | grammar success was absorbed by cheaper shared public-feedback role inference |
| C4 WGD-0 Hard Domain | `experiments/wgd0_b38_hidden_measurement.json` | WGD HFA 1.0; flat enumerators 0.25 | `WGD_ABSORBED_BY_CONSTRAINT_DISCOVERY` | brute enumeration failed on the slice, but GF(2) constraint discovery absorbed |
| PC B40/B41 | `experiments/b40_positive_control_measurement.json` | claimed HFA 1.0 | `B40_POSITIVE_CONTROL_ABSORBED_BY_FULL_TARGET_CLASS_PBE` | the positive-control attempt was absorbed, so it is residual-risk evidence |

The project case studies are evidence that the method prevented this project from overclaiming its own mechanisms.
They are not independent validation of a field-level standard.

## 6. Filled Absorber-Status Ledgers

### C1. FrameSeed Boolean

| Rung | Status | Evidence | Status source | Claim-ceiling effect |
|---|---|---|---|---|
| Hidden-open discipline | native_executable | audit failures 0; manifest frozen; no post-hidden edits | `experiments/frameseed0_b28_hidden_hfa.json:harness_reports/implementation_hashes` | protocol opening usable |
| Representation prior | native_executable | `representation_prior_absorbed=false`; role max HFA std 0.0 | `experiments/frameseed0_b28_hidden_hfa.json:token_evidence.representation_prior_absorbed` | not the winning absorber |
| Teaching dimension/search | native_executable | TD-H0 min HFA 1.0 against L3 min HFA 1.0 | `experiments/frameseed0_b28_hidden_hfa.json:token_evidence.baseline_absorptions.teaching_dimension` | winning absorption |
| Active learning | native_executable | L1 active min HFA 1.0 | `experiments/frameseed0_b28_hidden_hfa.json:token_evidence.baseline_absorptions.active_learning` | additional absorption, no signal |
| CEGIS/synthesis | native_executable | L2 CEGIS min HFA 1.0 | `experiments/frameseed0_b28_hidden_hfa.json:token_evidence.baseline_absorptions.cegis` | additional absorption, no signal |
| Nuisance/RAG/library | native_executable controls | nuisance, RAG, and library-learning min HFA all 1.0 | `experiments/frameseed0_b28_hidden_hfa.json:token_evidence.baseline_absorptions` | no frame-transmission claim |
| Residual risk | low for positive claims because no positive claim is made | earlier absorbers already win | `experiments/frameseed0_b28_hidden_hfa.json:terminal_token` | absorption token is stable |

### C2. FrameSeed SHEETS-0

| Rung | Status | Evidence | Status source | Claim-ceiling effect |
|---|---|---|---|---|
| Hidden-open discipline | native_executable | audit failures 0; domain roster audit missing `[]`; role stability passed | `experiments/frameseed_sheets0_b31_hidden_hfa.json:audit_failure_count/pre_hidden_gate_reports` | protocol opening usable |
| Parser/representation prior | native_executable | parser and representation prior not absorbed in token evidence | `experiments/frameseed_sheets0_b31_hidden_hfa.json:token_evidence.parser_prior_absorbed` | not the winning absorber |
| Schema/binding | native_executable | binding-only HFA 1.0; packet-erasure drop 0.0 pp | `experiments/frameseed_sheets0_b31_hidden_hfa.json:binding_only_ablation` | winning absorption |
| Typed CEGIS/PBE | capability_mode_scored | typed CEGIS exact and PBE/PROSE score 1.0 through `system_mode`/`mode_solves`, not independent native typed learners | `code/frameseed_sheets0_measurement.py:FULL_PIPELINE_SYSTEMS/system_mode/mode_solves`; `research/STATUS.md`; `experiments/EXPERIMENTS.md`; `experiments/ledger.jsonl` | supports capability ceiling, not native typed-baseline execution |
| Data wrangling/domain tools | capability_mode_scored | data-wrangling score 1.0 through full-pipeline capability mode | `code/frameseed_sheets0_measurement.py:FULL_PIPELINE_SYSTEMS`; `experiments/ledger.jsonl` | capability-mode absorption route, not native tool execution |
| Library/active/nuisance | capability_mode_scored | typed MDL library, active goal disambiguation, library learning, and nuisance oracle score 1.0 through mode scoring | `code/frameseed_sheets0_measurement.py:FULL_PIPELINE_SYSTEMS`; `research/STATUS.md` | no typed-frame signal; do not overclaim native execution |
| Residual risk | low for the negative conclusion; high for any public claim that native typed baselines were run | schema/binding already wins; native typed-prior-art execution remains unimplemented | `research/STATUS.md:Public Claim Ceiling`; `experiments/EXPERIMENTS.md:frameseed_sheets_b31` | absorption stable, public native-baseline claim forbidden |

### C3. WGD-0 Toy Grammar

| Rung | Status | Evidence | Status source | Claim-ceiling effect |
|---|---|---|---|---|
| Hidden-open discipline | native_executable | pre-hidden audit passed; 74 findings; hidden seed opened once; post-hidden code changes false | `experiments/wgd0_b37_hidden_measurement.json:prehidden_audit_summary/code_changes_after_hidden_open` | protocol opening usable |
| Majority/simple prior | native_executable | majority HFA 0.19401041666666666 | `experiments/wgd0_b37_hidden_measurement.json:system_summary.majority_feedback` | failed; domain not solved by majority |
| Schema/binding | native_executable shared role inference | schema-binding HFA 1.0 at ratio 0.08818090438181429 | `experiments/wgd0_b37_hidden_measurement.json:absorber_summary.schema_binding_absorbs`; `code/wgd0_measurement.py:infer_role_model` | winning absorption |
| `pbe_cegis` label | native_executable shared role inference, not independent PBE/CEGIS | HFA 1.0 at ratio 0.10745794529999544, produced by the same `infer_role_model()` function with a different source string | `experiments/wgd0_b37_hidden_measurement.json:absorber_summary.pbe_cegis_absorbs`; `code/wgd0_measurement.py:models` | cheaper public-feedback role inference absorbs; no independent PBE/CEGIS claim |
| WGD grammar | claimed system | WGD HFA 1.0; min-family HFA 1.0 | `experiments/wgd0_b37_hidden_measurement.json:functional_gate_summary` | functional success but not discovery signal |
| Later rungs | stopped after earlier absorptions | constraint, active, and library risks remain relevant for future positive claims | `experiments/wgd0_b37_hidden_measurement.json:token_evidence.absorptions` | no upward claim allowed |
| Residual risk | low for absorption; medium for methodology-validation story | internal synthetic case only | `experiments/wgd0_b37_hidden_measurement.json:terminal_token` | supports case report, not universal method validation |

### C4. WGD-0 Hard Domain

| Rung | Status | Evidence | Status source | Claim-ceiling effect |
|---|---|---|---|---|
| Hidden-open discipline | native_executable | pre-hidden audit passed; hidden seed opened once; post-hidden code changes false | `experiments/wgd0_b38_hidden_measurement.json:prehidden_audit_summary/code_changes_after_hidden_open` | protocol opening usable |
| Flat enumerators | native_executable but weak | lexicographic, size-first, random, and meet-in-middle truncated HFA all 0.25 overall | `experiments/wgd0_b38_hidden_measurement.json:enumeration_summary/system_summary` | brute enumeration genuinely failed on measured slice |
| WGD basis grammar | claimed system | HFA 1.0; composition 1.0; repair 1.0; abstention recall 1.0 | `experiments/wgd0_b38_hidden_measurement.json:functional_gate_summary` | functional success |
| GF(2) constraint discovery | native_executable | constraint solver HFA 1.0; ratio 0.9673932788374205 | `experiments/wgd0_b38_hidden_measurement.json:absorber_summary.constraint_solver_absorbs` | winning absorption |
| PBE/active/CEGIS variants | not winning in this artifact | token evidence lists PBE, CEGIS, and active CEGIS as false absorptions | `experiments/wgd0_b38_hidden_measurement.json:token_evidence.absorptions` | no positive claim depends on them |
| Sampling-validity boundary | explicit limitation | 8 worlds, 256 cases, 1536 scored predictions | `experiments/wgd0_b38_hidden_measurement.json:counts/hardness_summary` | claim limited to measured slice |
| Residual risk | low for absorption; high for broad field generalization | synthetic CPU-only domain | `experiments/wgd0_b38_hidden_measurement.json:terminal_token` | do not title this as proof about all AI discovery |

## 7. Cost Robustness

All-in accounting is necessary, but local serialization choices can move a marginal ratio.
The paper therefore reports terminal stability separately from rhetorical strength.

| Case | Cost fact | Stability conclusion | Allowed wording |
|---|---|---|---|
| C1 Boolean | multiple absorbers match HFA 1.0 | terminal does not depend on small bit perturbations | absorbed by teaching/search and related controls |
| C2 SHEETS-0 | binding-only HFA 1.0; packet-erasure drop 0.0 | terminal does not depend on typed capability-mode bit prices | absorbed by schema/binding; typed pipeline rows are capability-mode scored |
| C3 WGD toy schema | schema ratio 0.08818 | robustly cheaper under broad perturbation | absorbed by shared role inference |
| C3 WGD toy `pbe_cegis` | ratio 0.10746 | robustly cheaper, but not independent PBE/CEGIS | cheaper public-feedback role inference also absorbs |
| C4 WGD hard constraint | ratio 0.96739 | cheapness is fragile but <=4x absorption is stable | comparable-cost native constraint absorption |
| B40/B41 | full PBE ratio 2.83767 | inside <=4x; signal is killed | positive-control attempt absorbed |

## 8. Scope And Self-Audit Limits

This paper is self-audited.
The same project built the claims, domains, absorbers, cost ledgers, and adversarial reviews.
That is not independent validation of an anti-self-deception method.
It is an internally documented attempt to avoid self-deception.

Allowed claim:

```text
In this project record, the ladder prevented this project from narrating four
internal synthetic successes and one attempted positive control upward into
AI discovery claims.
```

Banned claim:

```text
The methodology has been independently validated as a field-level standard for
all AI discovery claims.
```

Missing validation: third-party rerun, blinded absorber selection, external adversarial review before hidden opening, reanalysis of a public AI discovery claim, and deterministic token reproduction from frozen artifacts.
Until at least one of those exists, the paper is a protocol with internal evidence and a residual-risk demonstration.

## 9. Related-Work Delta And Onboarding

The ladder is built from familiar tools: preregistration, hidden tests, fair baselines, ablations, MDL, program synthesis, and red-team review.
Its claim to novelty is procedural composition.
Before discovery language is allowed, ordinary explanations most native to the claim surface must win, fail, or be recorded as residual risk that lowers the claim.

Use this workflow for a new evaluator:

1. Define the discovery surface: function, scale, metric, threshold, and artifact role.
2. Classify the substrate: representation, parser, type system, verifier, action grammar, query channel, and human-authored tools.
3. Choose mandatory rungs by claim type: teaching/search, schema/binding, PBE/CEGIS, constraint/domain tools, library learning, generator probes.
4. Justify omissions before hidden opening: non-native, dominated, out of scope, or high residual risk.
5. Predeclare cost and query budgets: bits, runtime, examples, counterexamples, adapters, bindings, substrate, and residual programs.
6. Assign each absorber a status: native executable, proxy, capability-mode, formal bound, omitted non-native, or untested roster entry.
7. Run the hidden-open protocol: freeze manifest, smoke on public seed, open hidden once, forbid post-hidden edits.
8. Emit token, filled ledger, sensitivity band, residual-risk statement, and public claim ceiling.

## 10. Submission Position

The paper should be pitched as a methodology proposal with internal case evidence and an absorbed positive-control attempt.
It should not be pitched as a completed empirical survey of AI discovery claims.
The field-facing home run is still real if the protocol is adopted: discovery claims become less about attractive artifacts and more about which ordinary explanation survived first refusal.
But the paper earns that ambition only by obeying its own claim ceiling.

## Narrative Gate

We built a roster-relative protocol for testing AI discovery claims, killed every claim we tested including our own attempt at a positive control, and the methodology's value is that it prevents exactly this kind of overclaim.
The paper is stronger because it admits the positive-control failure instead of engineering around it.