# The Absorption Ladder: Roster-Relative Tests for AI Discovery Claims

Draft date: 2026-07-08.
Status: W-Loop B40 revision after Q-Loop B47.
Core claim: an AI discovery claim is credible only relative to a predeclared absorber roster, equal-information affordance map, all-in cost rule, hidden-open manifest, and explicit residual-risk statement.
Claim ceiling: this paper does not certify that all ordinary explanations have failed; it gives a decision procedure for documenting which ordinary explanations were made executable, which won, which failed, and which remain untested.

## Abstract

High hidden accuracy is not enough to claim that an AI system discovered a rule, frame, grammar, strategy, circuit, or world model.
A compact artifact is not enough.
An interpretable artifact is not enough.
The missing question is whether the discovery explanation survives first refusal by ordinary executable explanations under equal information and all-in accounting.
Those ordinary explanations include representation priors, parser and substrate priors, finite teaching, active queries, program synthesis, schema binding, domain tools, library learning, nuisance oracles, generator fingerprints, human-authored substrate, and post-hoc compression.

This paper presents the absorption ladder: a roster-relative protocol for predeclaring those explanations, making them executable, assigning terminal tokens after one hidden opening, and bounding the public claim to the evidence actually tested.
The contribution is not proof that the strongest possible boring explanations have failed.
That universal quantifier is not operational.
The contribution is a procedure for turning discovery language into an auditable statement: claim, roster, equal-information map, cost rule, hidden-open manifest, result token, and residual absorber risk.

The project record contains four absorbed internal case studies and one small positive control.
FrameSeed Boolean reached perfect hidden HFA, but exact teaching/search also reached 1.0.
FrameSeed SHEETS-0 reached perfect typed HFA, but schema binding, typed CEGIS/PBE, data wrangling, active disambiguation, nuisance, and library routes solved the task and packet erasure had zero-point drop.
WGD-0 toy grammar reached perfect HFA, but schema/binding and PBE/CEGIS matched it at 8.8 percent and 10.7 percent of WGD cost.
WGD-0 hard domain defeated flat enumerators over a 2^64 candidate space on the measured slice, but a GF(2) constraint solver matched perfect HFA at 96.7 percent of WGD cost.
A B40 positive control then produced a narrow roster-relative signal: a planted three-feature interaction learner reached 1.0 hidden HFA, the best declared absorber reached 0.71875, component erasure dropped 28.125 points, and the token was `B40_POSITIVE_CONTROL_SIGNAL_ROSTER_RELATIVE`.
That control is deliberately claim-limited because a full target-class PBE absorber was omitted and recorded as residual risk.

The ladder is therefore not a rejection machine.
It can emit signal.
But the strongest version of the method is also the narrowest: signal always means signal relative to the declared roster and residual-risk statement, not universal discovery.

## Source Map

| Source | Role |
|---|---|
| `research/dual_loop_supervisor_checkin_37.md` | B40 revision directive and B47 priority list |
| `research/question_loop_batch47.md` | adversarial attacks addressed here |
| `research/dual_loop_supervisor_checkin_36.md` | prior methodology-paper framing and B38 context |
| `research/METHODOLOGY_TEMPLATE.md` | reusable token, ladder, cost, and hidden-open vocabulary |
| `experiments/frameseed0_b28_hidden_hfa.json` | FrameSeed Boolean hidden artifact |
| `experiments/frameseed_sheets0_b31_hidden_hfa.json` | FrameSeed SHEETS-0 hidden artifact |
| `experiments/wgd0_b37_hidden_measurement.json` | WGD-0 toy grammar hidden artifact |
| `experiments/wgd0_b38_hidden_measurement.json` | WGD-0 hard-domain hidden artifact |
| `experiments/b40_positive_control_measurement.json` | B40 CPU-only positive-control artifact |
| `code/b40_positive_control.py` | B40 positive-control runner |

## 1. The Claim Being Made

The old draft used an over-absolute standard: it implied that a finite paper could exhaust the strongest ordinary explanations.
That was too absolute.
A finite protocol cannot know every possible ordinary explanation, and a self-authored study cannot certify its own completeness by assertion.
The revised claim is roster-relative:

```text
An AI discovery claim is credible only relative to a predeclared absorber roster,
the equal-information affordance map for that roster, the all-in cost rule, the
hidden-open manifest, and an explicit residual-risk statement for omitted or
proxy absorbers.
```

This is a sharper claim, not a weaker method.
It removes the impossible universal quantifier and replaces it with an auditable burden.
A reviewer can now ask concrete questions:

1. What exact discovery surface was claimed?
2. Which ordinary explanations were declared before hidden opening?
3. Which were native executable, proxy, formal lower bound, capability-only, or untested?
4. Which executable affordances did the claimed system receive, and were those shared or charged?
5. What all-in cost rule decided absorption?
6. Which terminal token fired?
7. Which omitted absorber risks lower the claim ceiling?

A positive result does not mean discovery in the abstract.
It means no declared absorber matched the threshold inside the frozen information and cost boundary.
A negative or absorbed result is not a failed paper.
It is the method doing its job.

## 2. Terminal Tokens

A hidden opening emits exactly one token.
The token is a decision rule, not a narrative mood.

| Token class | Meaning | Public claim ceiling |
|---|---|---|
| `VOID` | leakage, hidden mutation, subjective semantics, or baseline asymmetry broke the protocol | no scientific claim from that opening except that the protocol failed |
| `TRAP` | the domain was degenerate, tiny, leaked by representation, or not a real test of the target function | the domain did not test the intended claim |
| `ABSORBED_BY_X` | ordinary explanation X matched threshold within the cost rule | the apparent discovery is carried by X at the tested scale |
| `NEGATIVE` | the claimed system missed its functional gate | the claimed system failed its own test |
| `INCONCLUSIVE` | required metric, absorber, or cost field is missing | the claim remains unproven |
| `SIGNAL` | claimed system passed and declared absorbers failed under the manifest | narrow signal relative to roster, scale, cost rule, and residual risk |

Token precedence is intentionally conservative:

1. Void conditions outrank everything.
2. Trap conditions outrank functional success.
3. Representation, parser, substrate, binding, and domain-tool priors outrank model success.
4. Any declared ordinary absorber that reaches threshold within the cost boundary outranks signal.
5. A missed functional gate emits negative.
6. Signal can fire only after the above have failed or been bounded.

Mixed evidence cannot be narrated upward.
A result with a missing dangerous native absorber is not signal; it is inconclusive or a lower-ceiling result.

## 3. Absorber Completeness Stopping Rule

The ladder is open-ended, so it needs an operational stopping rule.
A run may stop and emit a bounded token only when the report contains all four items below.

| Requirement | Stop condition | If missing |
|---|---|---|
| Claim surface | functional surface, scale, threshold, and artifact role are written before hidden opening | `INCONCLUSIVE` or `VOID` if changed after hidden opening |
| Roster rationale | each included absorber is tied to a domain-native ordinary explanation | lower claim ceiling; no broad signal language |
| Status ledger | every dangerous rung is `native_executable`, `proxy_absorber`, `capability_mode_scored`, `formal_lower_bound`, `non_native_omitted`, or `untested_roster_entry` | `INCONCLUSIVE` for positive claims; absorption still stands if an earlier native absorber wins |
| Residual-risk statement | omitted absorbers are named and assigned low, medium, or high residual risk | signal cannot be narrated beyond the tested roster |

Residual-risk bands:

| Band | Definition | Claim effect |
|---|---|---|
| Low | omitted absorber is non-native, dominated by an executed native absorber, or irrelevant to the claim surface | no extra demotion beyond roster-relative wording |
| Medium | plausible absorber exists but is proxy-only, budget-limited, or not fully native | signal becomes preliminary; absorption claims stay narrow |
| High | plausible native absorber was not run and could match the claimed system | no strong positive claim; at most a positive control or protocol proposal |

This rule also handles B47's infinite-regress objection.
A reviewer can always propose another absorber.
The paper does not pretend otherwise.
The protocol says exactly how that proposal affects the claim ceiling: add it to the roster next time, or mark residual risk now.

## 4. Method In One Page

1. Define the discovery claim as a function, not as vibes around an artifact.
2. Precommit terminal tokens and token precedence.
3. Build a roster of native ordinary explanations for the domain.
4. Map equal information: parser, type system, examples, counterexamples, query channel, verifier, canonicalizer, operation grammar, bindings, and human substrate.
5. Freeze the all-in cost rule before hidden opening.
6. Freeze manifest hashes, public/smoke seeds, hidden seed rule, thresholds, baselines, and scorer.
7. Open hidden once and forbid post-hidden constructor, scorer, baseline, timeout, parser, token, or audit changes under that seed.
8. Emit a token, a filled absorber ledger, cost sensitivity, and residual-risk statement.

The method's novelty is not one component in isolation.
Preregistration, hidden tests, ablations, MDL, program synthesis, fair baselines, causal interventions, and red-team review all already exist.
The delta is binding them into a terminal-token decision procedure where ordinary explanations receive first refusal under equal executable affordances and all-in cost, and where every public claim is bounded by the roster and residual risk.

## 5. Positive Control: A Small Domain Where Signal Fires

B47 correctly attacked the old paper for only showing absorptions.
A method that can only say no is a rejection machine.
B40 adds a CPU-only positive control, not as field evidence for AI discovery, but as a sanity check that the terminal-token machinery can emit a bounded yes.

Artifact: `experiments/b40_positive_control_measurement.json`.
Runner: `code/b40_positive_control.py`.
Token: `B40_POSITIVE_CONTROL_SIGNAL_ROSTER_RELATIVE`.
Runtime: 0.064 seconds on the recorded run.

Domain:

- 12-bit inputs.
- Public training examples: 192.
- Hidden cases: 512, disjoint from training rows.
- Hypothesis class: 1320 three-feature interaction candidates.
- Target form: `(x_i AND x_j) XOR x_k XOR bias`.
- Threshold: 0.98 hidden HFA.

Declared absorber roster:

| Absorber | Hidden HFA | Status | Claim effect |
|---|---:|---|---|
| `majority_label` | 0.51171875 | native simple prior | failed |
| `single_bit_prior` | 0.71875 | native representation prior | failed |
| `pair_conjunction_prior` | 0.646484375 | native low-order prior | failed |
| `lookup_memorizer` | 0.51171875 | native memorization control | failed despite 1.0 train HFA |
| `budgeted_pbe_probe` | 0.712890625 | native but budget-limited PBE probe over first 128 candidates | failed |
| `random_interaction_probe` | 0.541015625 | negative control | failed |
| `claimed_interaction_learner` | 1.0 | claimed causal artifact | passed |

Causal controls:

| Control | Value | Effect |
|---|---:|---|
| component erasure hidden HFA | 0.71875 | interaction term is causal |
| component-erasure drop | 28.125 percentage points | artifact removal damages performance |
| randomized-label hidden HFA | 0.505859375 | learned artifact does not survive label randomization |

Cost snapshot:

| System | Total bits | Hidden HFA |
|---|---:|---:|
| `claimed_interaction_learner` | 5224 | 1.0 |
| `budgeted_pbe_probe` | 5280 | 0.712890625 |
| `lookup_memorizer` | 5720 | 0.51171875 |
| `single_bit_prior` | 3384 | 0.71875 |
| `pair_conjunction_prior` | 3696 | 0.646484375 |

Claim ceiling:

```text
The control shows that the ladder can emit a narrow SIGNAL when a planted causal
interaction survives the declared roster. It does not show that all ordinary
explanations failed, because a full target-class PBE over all 1320 candidates
was deliberately omitted and recorded as high residual risk.
```

This is exactly the revised core claim in miniature.
The ladder can say yes, but only by saying what the yes is relative to.

## 6. Case Study Summary

| Case | Artifact | Claimed success | Winning token | What the paper may claim |
|---|---|---|---|---|
| C1 FrameSeed Boolean | `experiments/frameseed0_b28_hidden_hfa.json` | L3 hidden HFA 1.0 | `FRAMESEED_T3_ABSORBED_BY_TEACHING_DIMENSION` | hidden success was absorbed by finite teaching/search |
| C2 FrameSeed SHEETS-0 | `experiments/frameseed_sheets0_b31_hidden_hfa.json` | L3 typed hidden HFA 1.0 | `FRAMESEED_SHEETS0_ABSORBED_BY_SCHEMA_BINDING` | typed packet advantage was absorbed by schema/binding and typed pipelines |
| C3 WGD-0 Toy Grammar | `experiments/wgd0_b37_hidden_measurement.json` | WGD HFA 1.0 | `WGD_ABSORBED_BY_SCHEMA_OR_BINDING_DISCOVERY` | grammar success was absorbed by schema/binding and PBE/CEGIS |
| C4 WGD-0 Hard Domain | `experiments/wgd0_b38_hidden_measurement.json` | WGD HFA 1.0; flat enumerators 0.25 | `WGD_ABSORBED_BY_CONSTRAINT_DISCOVERY` | brute enumeration failed, but GF(2) constraint discovery absorbed |
| PC B40 Positive Control | `experiments/b40_positive_control_measurement.json` | claimed HFA 1.0 | `B40_POSITIVE_CONTROL_SIGNAL_ROSTER_RELATIVE` | signal only relative to a declared, incomplete roster |

The four project case studies are not methodology validation by themselves.
They are evidence that the method prevented this project from overclaiming its own mechanisms.
The positive control validates a different property: the token rule is not hard-coded to reject.
External validation remains future work.

## 7. Filled Absorber-Status Ledgers

### C1. FrameSeed Boolean

| Rung | Status | Evidence | Cost note | Claim-ceiling effect |
|---|---|---|---|---|
| Hidden-open discipline | native executable | audit failures 0; manifest frozen; no post-hidden edits in artifact | manifest and code hashes recorded | protocol opening usable |
| Representation prior | native executable | `representation_prior_absorbed=false`; role max HFA std 0.0 | charged in token evidence as counted bits | not the winning absorber |
| Teaching dimension/search | native executable | TD-H0 min HFA 1.0 against L3 min HFA 1.0 | finite teaching/search counted | winning absorption |
| Active learning | native executable | L1 active min HFA 1.0 | same hidden query surface | additional absorption, no signal |
| CEGIS/synthesis | native executable | L2 CEGIS min HFA 1.0 | same examples and scorer | additional absorption, no signal |
| Nuisance/RAG/library | native executable controls | nuisance, RAG, and library-learning min HFA all 1.0 | same public transcript and scorer | no frame-transmission claim |
| Residual risk | low for positive claims because no positive claim is made | earlier absorbers already win | later omitted stronger absorbers cannot rescue the original claim | absorption token is stable |

### C2. FrameSeed SHEETS-0

| Rung | Status | Evidence | Cost note | Claim-ceiling effect |
|---|---|---|---|---|
| Hidden-open discipline | native executable | audit failures 0; domain roster audit missing `[]`; role stability passed | cost split passed; bits counted | protocol opening usable |
| Parser/representation prior | native executable | parser and representation prior not absorbed in token evidence | parser/human labor ledger present | not the winning absorber |
| Schema/binding | native executable | binding-only HFA 1.0; packet-erasure drop 0.0 pp | charged task bindings | winning absorption |
| Typed CEGIS/PBE | native executable | typed CEGIS exact min HFA 1.0; PBE/PROSE min HFA 1.0 | same typed action API | independent absorption route |
| Data wrangling/domain tools | native executable | data-wrangling min HFA 1.0 | typed table substrate shared or charged | domain-tool absorption route |
| Library/active/nuisance | native executable | typed MDL library, active goal disambiguation, and nuisance oracle min HFA 1.0 | same examples and public substrate | no typed-frame signal |
| Residual risk | low for the negative conclusion | many native routes already solve | unrun stronger variants only make discovery claim harder | absorption token is stable |

### C3. WGD-0 Toy Grammar

| Rung | Status | Evidence | Cost note | Claim-ceiling effect |
|---|---|---|---|---|
| Hidden-open discipline | native executable | pre-hidden audit passed; 74 findings; hidden seed opened once; post-hidden code changes false | manifest hash recorded | protocol opening usable |
| Majority/simple prior | native executable | majority HFA 0.19401041666666666 | 191.125 mean bits | failed; domain not solved by majority |
| Schema/binding | native executable | schema-binding HFA 1.0 | 3147.1875 mean bits; ratio 0.08818090438181429 | winning absorption |
| PBE/CEGIS | native executable | PBE/CEGIS HFA 1.0 | 3835.1875 mean bits; ratio 0.10745794529999544 | independent winning absorption |
| WGD grammar | claimed system | WGD HFA 1.0; min-family HFA 1.0 | 35690.125 mean bits | functional success but not discovery signal |
| Later rungs | stopped after earlier native absorptions | constraint, active, and library risks remain relevant for future positive claims | not needed to kill this claim | no upward claim allowed |
| Residual risk | low for absorption; medium for any broader methodology-validation story | internal synthetic case only | no external-stakes validation | supports case report, not universal method validation |

### C4. WGD-0 Hard Domain

| Rung | Status | Evidence | Cost note | Claim-ceiling effect |
|---|---|---|---|---|
| Hidden-open discipline | native executable | pre-hidden audit passed; hidden seed opened once; post-hidden code changes false | manifest hash recorded | protocol opening usable |
| Flat enumerators | native executable but weak | lexicographic, size-first, random, and meet-in-middle truncated HFA all 0.25 overall | ratios 6.72 to 6.78 vs WGD; 8000 candidates per case | brute enumeration genuinely failed on measured slice |
| WGD basis grammar | claimed system | HFA 1.0; composition 1.0; repair 1.0; abstention recall 1.0 | 704640 total bits; 88080 mean bits per world | functional success |
| GF(2) constraint discovery | native executable | constraint solver HFA 1.0 | 681664 total bits; ratio 0.9673932788374205 | winning absorption |
| PBE/active/CEGIS variants | not winning in this artifact | token evidence lists PBE, CEGIS, active CEGIS as false absorptions | constraint solver already wins | no positive claim depends on them |
| Sampling-validity boundary | explicit limitation | 8 worlds, 256 cases, 1536 scored predictions | candidate-space log2 64 is search-space hardness, not evaluation-scale proof | claim limited to measured slice |
| Residual risk | low for absorption; high for broad field generalization | synthetic CPU-only domain | external cases absent | do not title this as proof about all AI discovery |

## 8. Cost Robustness

The old draft leaned too hard on exact bit ratios.
All-in accounting is necessary, but local serialization choices can move a marginal ratio.
The revised rule is to report sensitivity bands and separate terminal stability from rhetorical strength.

| Case | Nominal ratio or cost fact | Sensitivity conclusion | Allowed wording |
|---|---|---|---|
| C1 FrameSeed Boolean | multiple absorbers match HFA 1.0 | terminal does not depend on small bit perturbations because functional equality already kills the frame claim | absorbed by teaching/search and related controls |
| C2 SHEETS-0 | binding-only HFA 1.0; packet-erasure drop 0.0 | terminal does not depend on exact bit price of the packet because removing it causes no loss | absorbed by schema/binding and typed pipelines |
| C3 WGD toy schema | schema ratio 0.08818 | even absorber x2 and WGD x0.5 gives about 0.353 | robustly cheaper under broad perturbation |
| C3 WGD toy PBE | PBE ratio 0.10746 | even absorber x2 and WGD x0.5 gives about 0.430 | robustly cheaper under broad perturbation |
| C4 WGD hard constraint | constraint ratio 0.96739 | absorber x1.1 and WGD x0.9 gives about 1.18; exact cheapness is fragile | comparable-cost native constraint absorption, not a strong cheapness claim |
| B40 positive control | claimed 5224 bits; best declared passing absorber absent; budgeted PBE 5280 bits but HFA 0.71289 | token is functionally stable to cost perturbation among declared absorbers; omitted full PBE is high residual risk | narrow roster-relative signal only |

A cost ratio near 1.0 should not be sold as a dramatic efficiency result.
For C4, the honest claim is that the constraint solver is a native ordinary explanation at comparable all-in cost under the recorded ledger.
The result would remain an absorption under the template's <=4x rule, but the phrase "cheaper" should be avoided unless a sensitivity band supports it.

The minimum cost-reporting standard is now:

1. exact serialization rule;
2. shared substrate separated from system-specific structure;
3. examples, counterexamples, bindings, programs, library bits, human substrate, and runtime/query side metrics when material;
4. nominal ratio;
5. at least one punitive perturbation band;
6. terminal-token stability under the perturbation.

## 9. Scope And Self-Audit Limits

This paper is self-audited.
The same project built the claims, domains, absorbers, cost ledgers, and adversarial reviews.
That is not independent validation of an anti-self-deception method.
It is an internally documented attempt to avoid self-deception.

Allowed claim:

```text
In this project record, the ladder prevented this project from narrating four
internal synthetic successes upward into discovery claims, and the B40 positive
control shows the token rule can emit a bounded signal when the declared roster
fails.
```

Banned claim:

```text
The methodology has been independently validated as a field-level standard for
all AI discovery claims.
```

The missing validation steps are concrete:

- third-party rerun of at least one manifest;
- blinded absorber selection by someone outside the work loop;
- external adversarial review before hidden opening;
- reanalysis of a public AI discovery claim with before/after claim ceilings;
- deterministic reproduction of token assignment from frozen artifacts.

Until at least one of those exists, the paper is a protocol with internal evidence and a positive-control sanity check, not a completed community validation.

## 10. Related-Work Delta

The ladder is built from familiar scientific tools.
Its claim to novelty is procedural composition, not isolated invention.

| Existing practice | What it already gives | What the ladder adds |
|---|---|---|
| Preregistration | predeclared hypotheses and analysis plans | terminal-token precedence with absorber first refusal |
| Hidden-test benchmarks | holdout protection | hidden-open manifest plus post-hidden mutation voiding |
| Fair baseline design | comparison discipline | equal executable affordance map, not just equal bytes |
| Ablations | component importance | causal artifact tests tied to token ceilings |
| MDL and compression | cost-aware explanation | all-in accounting across frames, bindings, programs, libraries, human substrate, and residual teaching |
| Program synthesis and CEGIS | ordinary constructive alternatives | mandatory native absorbers when the claim is executable rule discovery |
| Red-team review | adversarial critique | roster ledger where critique changes the claim ceiling |
| No-free-lunch baseline skepticism | warns against weak baselines | operational stopping rule and residual-risk bands |

The paper's method is therefore not "use good baselines" in new terminology.
It is a refusal protocol: before discovery language is allowed, the ordinary explanations most native to the claim surface must either win, fail, or be recorded as residual risk that lowers the claim.

## 11. Domain Onboarding Recipe

Use this eight-step workflow for a new evaluator.

1. Define the discovery surface: function, scale, metric, threshold, and artifact role.
2. Classify the substrate: representation, parser, type system, verifier, action grammar, query channel, and human-authored tools.
3. Choose mandatory rungs by claim type: teaching/search for compact packets, schema/binding for typed worlds, PBE/CEGIS for executable rules, constraint/domain tools for algebraic tasks, library learning for transfer, generator probes for synthetic domains.
4. Justify omissions before hidden opening: non-native, dominated, out of scope, or high residual risk.
5. Predeclare cost and query budgets: bits, runtime, examples, counterexamples, adapters, bindings, substrate, and residual programs.
6. Assign each absorber a status: native executable, proxy, capability-mode, formal bound, omitted non-native, or untested roster entry.
7. Run the hidden-open protocol: freeze manifest, smoke on public seed, open hidden once, forbid post-hidden edits.
8. Emit token, filled ledger, sensitivity band, residual-risk statement, and public claim ceiling.

Mini-examples:

| Claim | Mandatory first rungs | Likely claim ceiling if missing |
|---|---|---|
| game agent discovered a world model | representation priors, simulator-state leakage, model-free policy baseline, planner/domain tool, intervention erasure, generator classifier | inconclusive for world-model claim |
| theorem prover discovered lemmas | premise-selection baseline, proof-search budget, library retrieval, statement-template generator, verifier-substrate accounting | at most search-engineering claim |
| LLM generated scientific hypotheses | literature retrieval, template recombination, citation leakage, domain-tool baseline, randomized-topic control, human-substrate accounting | protocol proposal or ideation claim, not discovery |

## 12. What The Absorptions Show

The absorptions show an escalation pattern inside this project.
Simple Boolean frame transmission was absorbed by finite teaching/search.
Typed frame transfer was absorbed by schema binding and typed pipelines.
Toy grammar discovery was absorbed by schema/binding and PBE/CEGIS.
Hard grammar discovery beat flat enumeration but was absorbed by native GF(2) constraint discovery.

That pattern matters because it changes what the next honest experiment must beat.
It does not prove that discovery is impossible.
It does not prove that larger systems cannot separate.
It does not prove the tested mechanisms are useless engineering.
It says only that the public discovery claim did not survive the declared ordinary explanation that won.

The hard-domain wording must be especially careful.
The `2^64` candidate space establishes why flat enumeration was a weak route on this domain.
It does not make 8 worlds and 256 hidden cases a field-scale evaluation.
The measured result is narrower: on this slice, flat enumerators failed, WGD succeeded, and the constraint solver also succeeded at comparable all-in cost.

## 13. Appendices Compressed Into Evidence

The previous Appendix F, G, and H material was mostly generated checklist volume.
It has been removed from the paper body.
The useful content is now compressed into three evidence objects:

1. the one-page onboarding recipe above;
2. the filled case ledgers in Section 7;
3. the compact rung and token cards below.

### A. Rung Cards

| Rung | Native question | Typical absorber | Failure consequence |
|---|---|---|---|
| Representation prior | do public features expose the answer? | feature parser, type tags, names, ontology | absorb, trap, or lower ceiling |
| Parser/substrate prior | does the parser, verifier, DSL, action space, or canonicalizer do the work? | shared-or-charged substrate baseline | absorb or void for asymmetry |
| Teaching dimension | is the packet just a small teaching set? | exact or bounded teaching/search | absorb |
| Active learning | do queries isolate the target cheaply? | query planner, group tests, active CEGIS | absorb or charge query bits |
| PBE/CEGIS | can a program be synthesized from the same evidence? | PBE, PROSE, SyGuS, ILP, CEGIS | absorb |
| Library learning | do reusable macros explain transfer? | MDL library, DreamCoder-style library, e-graphs | absorb or lower transfer claim |
| Schema/binding | do role bindings solve the task? | schema matcher, binding fingerprints, entity resolver | absorb |
| Domain tool | is this native to a known solver? | SQL, spreadsheet tools, SAT/SMT/ILP/ASP/CSP | absorb |
| Nuisance oracle | does difficulty vanish when nuisance is removed? | relevant-feature or invariant oracle | absorb or lower ceiling |
| Constraint discovery | is the artifact a constraint theory? | rank solver, SAT/SMT encoding, table constraints | absorb |
| Generator family | do public statistics identify the synthetic family? | template classifier, compression classifier, graph probe | trap, absorb, or lower ceiling |
| Post-hoc compression | does the artifact merely summarize behavior after success? | frozen-before-score compression control | no artifact-causality claim |

### B. Cost Fields

| Field | Meaning |
|---|---|
| `F` | reusable frame, rule, grammar, verifier, representation, or method bits |
| `G` | general system, solver, interpreter, or substrate code bits when charged separately |
| `B_i` | task-specific bindings |
| `P_i` | executable per-task program or policy |
| `E_i` | examples and labels |
| `C_i` / `Q_i` | counterexamples, active queries, or query answers |
| `V_i` | verifier obligations or proof/test clauses |
| `R_i` | residual teaching after reusable structure is installed |
| `A_i` | abstention policy and abstention labels |
| `L` | learned library or macro bits |
| `H` | human-authored parser, substrate, design, verifier, and adapter work |
| `O` | operation ontology or typed action-space supply |
| `N` | nuisance-removal, invariant, or oracle information |

### C. Claim-Ceiling Rules

| Result | Say | Do not say |
|---|---|---|
| Signal | survived the declared roster under this manifest and residual-risk statement | all ordinary explanations failed |
| Absorption | named ordinary route carries the measured success | the claimed mechanism is useless in every setting |
| Void | the hidden opening failed procedurally | the model almost had signal |
| Trap | the domain did not test the intended function | the model solved a meaningful discovery task |
| Negative | the claimed system missed its own threshold | an untested variant would have worked |
| Inconclusive | a required absorber, metric, or cost field is missing | mixed evidence supports discovery |

## 14. Submission Position

The paper should be pitched as a methodology proposal with internal case evidence and a positive-control sanity check.
It should not be pitched as a completed empirical survey of AI discovery claims.
The most honest title surface is roster-relative testing, not universal certification.

The field-facing home run is still real if the protocol is adopted: discovery claims become less about attractive artifacts and more about which explanation survived first refusal.
That changes review behavior.
But the paper earns that ambition only by obeying its own claim ceiling.

## NARRATIVE SECTION

We built a rigorous roster-relative methodology for testing AI discovery claims.
It killed four of our own discovery stories, then emitted one deliberately narrow positive-control signal when a planted causal interaction survived the declared roster.
The method is alive because it can say both no and yes; the paper is now honest because every yes is bounded by the roster, cost rule, hidden-open manifest, and residual absorber risk.