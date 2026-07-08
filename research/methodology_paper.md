# The Absorption Ladder: How to Honestly Test AI Discovery Claims

Draft date: 2026-07-08.
Status: W-Loop B39 full methodology paper.
Core claim: an AI discovery claim is not credible until its strongest boring explanations have been made executable and have failed under equal information and all-in cost.
Claim ceiling: this paper does not prove that AI discovery is impossible and does not claim that any tested discovery mechanism works.

## Abstract

AI systems are often said to discover rules, frames, grammars, strategies, circuits, or world models when they score well on hidden tasks.
High hidden accuracy is not enough.
A compact artifact is not enough.
An interpretable artifact is not enough.
The central evaluation question is whether the apparent discovery survives absorption by ordinary explanations.
Those explanations include supplied representation, parser priors, teaching sets, active queries, program synthesis, schema binding, domain tools, library learning, nuisance oracles, generator fingerprints, human-authored substrate, and post-hoc compression.
This paper introduces the absorption ladder, a methodology that gives those ordinary explanations first refusal before allowing a positive discovery claim.
The method precommits terminal tokens, freezes hidden-open boundaries, requires equal-information baselines, charges all-in costs, runs causal artifact tests, and treats absorbed or negative outcomes as primary results.
Across four case studies in this project, the ladder prevented false positives.
Boolean FrameSeed reached perfect hidden HFA, but exact finite teaching and search reached the same score.
Typed SHEETS-0 reached perfect typed HFA, but charged schema bindings and typed table pipelines solved without the packet.
A WGD toy grammar reached perfect HFA, but schema/binding and PBE/CEGIS baselines matched at roughly 8.8 percent and 10.7 percent of WGD cost.
A 64-rule WGD hard domain defeated brute enumerators over a 2^64 subset space, but a GF(2) constraint solver matched perfect HFA at about 96.7 percent of WGD cost.
The contribution is not a new discovery mechanism.
The contribution is an adversarial measurement immune system for deciding when a discovery claim has survived the best non-discovery explanations.

## Source Map

| Source | Role |
|---|---|
| research/dual_loop_supervisor_checkin_36.md | directive |
| research/question_loop_batch46.md | outline |
| research/frameseed_milestone_report.md | FrameSeed case-study source |
| research/dual_loop_supervisor_checkin_35.md | WGD toy arc context |
| research/dual_loop_supervisor_checkin_32.md | FrameSeed arc context |
| research/METHODOLOGY_TEMPLATE.md | reusable framework |

| Case | Hidden artifact | Measurement code | Terminal token |
|---|---|---|---|
| FrameSeed Boolean | experiments/frameseed0_b28_hidden_hfa.json | code/frameseed0_measurement.py | FRAMESEED_T3_ABSORBED_BY_TEACHING_DIMENSION |
| FrameSeed SHEETS-0 | experiments/frameseed_sheets0_b31_hidden_hfa.json | code/frameseed_sheets0_measurement.py | FRAMESEED_SHEETS0_ABSORBED_BY_SCHEMA_BINDING |
| WGD-0 Toy Grammar | experiments/wgd0_b37_hidden_measurement.json | code/wgd0_measurement.py | WGD_ABSORBED_BY_SCHEMA_OR_BINDING_DISCOVERY |
| WGD-0 Hard Domain | experiments/wgd0_b38_hidden_measurement.json | code/wgd0_b38_hard_domain.py | WGD_ABSORBED_BY_CONSTRAINT_DISCOVERY |

## 1. Introduction

Discovery language is attractive because it compresses success into a human story.
A system predicts a hidden label, emits a compact rule, repairs an invalid object, or composes a held-out transformation.
The tempting story is that the system discovered reusable structure.
But the same behavior can arise from cheaper mechanisms.
The public representation may already contain the answer.
The parser may have supplied the ontology.
The hidden family may be small enough for finite teaching.
The task may be solved by active disambiguation.
A program may be synthesized from examples.
A schema matcher may bind the roles.
A domain tool may solve the instance.
A learned library may compress sibling tasks.
A nuisance oracle may remove hard-looking variation.
A generator classifier may identify the synthetic family.
A human-authored substrate may have done the real work.
A post-hoc artifact may merely summarize behavior after success.
The absorption ladder rejects the habit of treating hidden accuracy as decisive.
It makes the strongest boring explanation executable before the result is known.
It changes the review question from `Did the model score well?` to `Which explanation for the score survived first refusal?`.

## 2. Definitions

Signal means the claimed system passes functional gates and no declared absorber matches threshold within the precommitted cost boundary.
Absorption means an ordinary explanation matches the functional threshold at matched or allowed all-in cost.
Void means the protocol failed through leakage, post-hidden mutation, subjective hidden semantics, baseline denial, or uncharged substrate.
Trap means the domain is degenerate, tiny, leaked, or not a real test of the intended function.
Negative means the claimed system misses its own functional threshold.
Inconclusive means a required absorber or metric is missing.
A terminal token is the exact precommitted label emitted for one hidden opening.
A native absorber is a baseline that genuinely operates in the task's natural representation.
A capability-mode absorber is only a capability witness and cannot be narrated as native execution.
A claim ceiling is the strongest public statement allowed after the terminal token fires.

## 3. Terminal Token Precedence

Token precedence prevents post-hoc optimism.
Leakage and post-hidden mutation outrank every positive result.
Domain degeneracy outranks every positive result.
Representation, parser, substrate, binding, and domain-tool priors outrank claimed-system success.
Any ordinary absorber that reaches threshold within the cost boundary outranks signal.
A missed functional gate emits negative.
Only after all of those checks can signal fire.
Mixed evidence cannot be narrated upward.
The token is a decision rule, not an impressionistic summary.

## 4. The Absorption Ladder

The ladder is a first-refusal stack.
Each rung names an ordinary explanation that must be made dangerous before discovery language is allowed.
The ladder is not the same for every domain.
A compact-packet claim should face teaching dimension early.
A spreadsheet claim should face schema binding and typed table tools early.
A grammar claim should face binding, synthesis, library learning, active learning, and constraint discovery early.
A compositional claim must face methods that exploit compositionality.
The aim is not to defeat the weakest straw baseline.
The aim is to find the strongest boring explanation still alive.

## 5. Equal-Information Baseline Contract

Equal bytes are not equal information.
A claimed system may receive a parser, typed object model, canonicalizer, verifier, operation grammar, repair API, or task bindings in executable form.
If a baseline receives only raw prose while the claimed system receives executable affordances, the comparison is asymmetric.
Every executable affordance must be shared, losslessly translated, or charged.
Adapter costs must be declared before hidden opening.
Canonicalizers must be shared or charged.
Query channels must be shared or charged.
Verifier semantics must be shared or charged.
Human-authored substrate must be shared or charged.

## 6. All-In Accounting

Discovery claims become slippery when only the attractive artifact is counted.
A tiny frame with massive task bindings is not a frame-transfer signal.
A compact grammar with a hand-authored parser and verifier is not necessarily grammar discovery.
A reusable library with large per-task residual programs may be ordinary synthesis.
The all-in ledger forces the expensive parts of the story into the open.
A crude symmetric ledger is better than a polished story with uncharged substrate.
The terminal token names the explanation that carries the claim.

## 7. Hidden-Open Discipline

Hidden-open discipline is the procedural backbone of the ladder.
A hidden seed should not become part of iterative development.
Public smoke seeds are allowed and necessary.
Hidden seeds are opened once.
The manifest should hash code, specs, seeds, token policy, thresholds, baselines, and cost rules.
No constructor, scorer, baseline, timeout, parser, token policy, or audit change is allowed after hidden opening under the same seed.

## 8. Causal Artifact Tests

A discovery artifact must be causal, not decorative.
If removing the artifact does not hurt performance, the artifact is not the active ingredient.
If randomizing labels does not hurt performance, the scorer may be leaked or degenerate.
If a generator-family classifier predicts hidden structure, the artifact may exploit synthetic fingerprints.
If a post-hoc grammar is not frozen before held-out scoring, interpretability is not evidence of discovery.

## 9. Transfer Without Clone Accounting

Reusable-structure claims require transfer accounting.
Sibling tasks are necessary but not sufficient.
Siblings must be nonduplicate.
Residual bindings, programs, queries, and library bits must be charged.
If bindings or programs explain the transfer, the token should say so.

## 10. Case Studies

The case studies are absorptions, not positive discovery claims.
Each case is useful because a tempting success metric was insufficient.
The numbers below come from saved hidden measurement artifacts in the current checkout.

### C1. FrameSeed Boolean

Artifact: `experiments/frameseed0_b28_hidden_hfa.json`.
Measurement runner: `code/frameseed0_measurement.py`.
Terminal token: `FRAMESEED_T3_ABSORBED_BY_TEACHING_DIMENSION`.
Absorber: teaching dimension and exact finite teaching/search.
Perfect hidden HFA did not establish frame transmission because finite teaching and search reached the same hidden accuracy.

| Metric | Value |
|---|---|
| target bundles | 10240 |
| hidden queries per system | 15728640 |
| L3 mean HFA | 1.0 |
| L3 min HFA | 1.0 |
| TD-H0 min HFA | 1.0 |
| L1 active min HFA | 1.0 |
| L2 CEGIS min HFA | 1.0 |
| RAG min HFA | 1.0 |
| nuisance oracle min HFA | 1.0 |
| library-learning min HFA | 1.0 |
| packet growth alpha | 0.0028930015823335495 |
| role max HFA std | 0.0 |
| audit failures | 0 |

### C2. FrameSeed SHEETS-0

Artifact: `experiments/frameseed_sheets0_b31_hidden_hfa.json`.
Measurement runner: `code/frameseed_sheets0_measurement.py`.
Terminal token: `FRAMESEED_SHEETS0_ABSORBED_BY_SCHEMA_BINDING`.
Absorber: schema binding plus typed table pipelines.
The typed packet was not the active ingredient because charged bindings alone preserved perfect HFA and packet erasure produced a zero-point drop.

| Metric | Value |
|---|---|
| target bundles | 15360 |
| hidden queries per system | 15728640 |
| L3 mean HFA | 1.0 |
| L3 min HFA | 1.0 |
| binding-only HFA | 1.0 |
| packet-erasure drop pp | 0.0 |
| PBE/PROSE min HFA | 1.0 |
| data-wrangling min HFA | 1.0 |
| typed CEGIS exact min HFA | 1.0 |
| typed MDL library min HFA | 1.0 |
| active goal disambiguation min HFA | 1.0 |
| non-Boolean output fraction | 0.8001302083333334 |
| binding growth alpha | 0.0 |
| AFTD all-in passed | false |
| composition gate passed | false |
| audit failures | 0 |

### C3. WGD-0 Toy Grammar

Artifact: `experiments/wgd0_b37_hidden_measurement.json`.
Measurement runner: `code/wgd0_measurement.py`.
Terminal token: `WGD_ABSORBED_BY_SCHEMA_OR_BINDING_DISCOVERY`.
Absorber: schema/binding discovery and PBE/CEGIS.
The grammar candidate solved, but cheaper role-binding and feedback-program induction solved the same hidden cases.

| Metric | Value |
|---|---|
| hidden worlds | 128 |
| hidden cases | 1536 |
| scored predictions | 6144 |
| WGD HFA | 1.0 |
| WGD min-family HFA | 1.0 |
| schema-binding HFA | 1.0 |
| PBE/CEGIS HFA | 1.0 |
| majority HFA | 0.19401041666666666 |
| WGD mean cost bits | 35690.125 |
| schema mean cost bits | 3147.1875 |
| PBE mean cost bits | 3835.1875 |
| schema cost ratio | 0.08818090438181429 |
| PBE cost ratio | 0.10745794529999544 |
| pre-hidden audit passed | true |
| post-hidden code changes | false |

### C4. WGD-0 Hard Domain

Artifact: `experiments/wgd0_b38_hidden_measurement.json`.
Measurement runner: `code/wgd0_b38_hard_domain.py`.
Terminal token: `WGD_ABSORBED_BY_CONSTRAINT_DISCOVERY`.
Absorber: GF(2) constraint discovery.
The domain finally beat brute enumeration, yet the learnable structure was exactly what a GF(2) constraint solver exploits.

| Metric | Value |
|---|---|
| worlds | 8 |
| cases | 256 |
| scored predictions | 1536 |
| rule count | 64 |
| state bits | 128 |
| candidate space | 18446744073709551616 |
| candidate-space log2 | 64 |
| ordered-composition-space log2 | 144.0 |
| enumeration budget per case | 8000 |
| enumeration fraction per case | 4.336808689942018e-16 |
| WGD HFA | 1.0 |
| constraint HFA | 1.0 |
| lexicographic HFA | 0.25 |
| size-first HFA | 0.25 |
| random HFA | 0.25 |
| meet-in-middle truncated HFA | 0.25 |
| composition HFA | 1.0 |
| repair success | 1.0 |
| abstention recall | 1.0 |
| WGD mean cost bits per world | 88080.0 |
| constraint mean cost bits per world | 85208.0 |
| constraint cost ratio | 0.9673932788374205 |
| constraint absorbs | true |
| pre-hidden audit passed | true |
| post-hidden code changes | false |

## 11. What the Four Absorptions Show

The four cases form an escalation sequence.
The Boolean world was absorbed by exact finite teaching/search.
The typed world was absorbed by schema binding and typed pipelines.
The toy grammar world was absorbed by schema/binding and PBE/CEGIS.
The hard grammar world was absorbed by constraint discovery.
The domain became harder and the absorber became more sophisticated.
That is exactly what an honest ladder should do.
B38 is the strongest negative because brute enumeration finally failed.
It still was not signal because the native algebraic absorber matched.

## 12. Native Absorber Theater

Absorber theater happens when a paper lists strong baselines without making them genuinely dangerous.
A baseline can be theatrical because it is not executable.
A baseline can be theatrical because it is denied the substrate used by the claimed system.
A baseline can be theatrical because it is only capability-mode scored but later narrated as native.
The ladder addresses this by forcing every absorber to declare its status before hidden opening.
The status categories are native_executable, proxy_absorber, capability_mode_scored, formal_lower_bound, and untested_roster_entry.
A proxy can lower the claim ceiling.
A weak absorber that wins can still kill a claim in the narrow sense that even weak ordinary routes were enough.
A weak absorber that loses cannot support signal.

## 13. Negative Results as the Result

FrameSeed did not survive.
WGD did not survive.
The methodology did survive.
A project that cannot say no to its own favorite mechanisms cannot be trusted when it says yes.
The absorption ladder makes no a first-class output.
An absorption names what must be beaten next.
Do not claim frame transmission until teaching dimension and binding-only ablations fail.
Do not claim typed frame transfer until schema binding, typed PBE, CEGIS, data tools, and library learning fail.
Do not claim grammar discovery until binding, synthesis, active learning, library learning, constraint discovery, and substrate accounting fail.
Do not claim hard-domain discovery merely because brute enumeration fails.

## 14. What Would Count as a Positive Result

The ladder is not designed to make signal impossible.
It is designed to make signal expensive to fake.
A positive result would need a frozen claim, functional gates, hidden-open discipline, equal-information absorbers, all-in ledgers, causal artifact tests, and claim ceilings.
The claimed system would need to pass functional gates.
Native absorbers would need to fail or pay the precommitted penalty.
Component erasure would need to show that the claimed artifact is causal.
Leakage classifiers would need to stay below thresholds.
Transfer siblings would need to be nonduplicate and reduced after residual costs.
The allowed claim would still be bounded to the tested claim, absorbers, and scale.

## 15. What Scale or Domain Might Separate

The case studies do not prove that discovery cannot separate at larger scale.
They suggest why small CPU-only synthetic worlds are hostile to positive discovery evidence.
If the world is simple, brute search, teaching sets, or small synthesis can solve it.
If the world is typed and practical, domain tools and schema binding can solve it.
If the world is compositional, library learning and constraint solvers may exploit the composition.
If the world is synthetic, generator fingerprints may leak.
If the world is made unstructured enough to defeat all absorbers, it may also defeat the claimed system.
A real separation may require domains where ordinary solvers are strong but still miss a reusable structure that the claimed system finds cheaply.

## 16. Limitations

The absorption ladder cannot enumerate every possible boring explanation.
It cannot make bit accounting perfectly objective.
It cannot make a proxy baseline native by naming it.
It cannot prove that future larger systems will fail.
It cannot replace domain expertise.
It can be misused to dismiss useful engineering.
An absorbed method may still be useful.
The token only says which explanation carries the claim.
The answer is not trust.
The answer is auditability.

## 17. Conclusion

The absorption ladder reframes AI discovery evaluation.
The question is not whether a system can produce a good hidden score.
The question is whether the discovery explanation survives first refusal by the strongest ordinary mechanisms.
Perfect hidden HFA was absorbed.
Typed outputs were absorbed.
Interpretable grammar was absorbed.
Exponential brute-search hardness was absorbed.
Before asking the world to believe that an AI discovered something, make every ordinary explanation dangerous.
If one wins, publish the absorption honestly.
If none wins, the signal will be stronger because it was forced to survive.

## Appendix A: Rung Cards

### A.1. Representation prior

Question: does public features expose the answer?
Typical absorber: feature parser, type tags, names, ontology.
Token family: `ABSORBED_BY_REPRESENTATION_PRIOR`.
Inputs received: same public transcript, same substrate where applicable, same examples, same counterexamples, and same scorer.
Costs charged: adapter bits, executable program bits, query answers, human substrate, residual bindings, and runtime where material.
Failure mode: the rung is inconclusive if it is not implemented natively enough for the domain.

### A.2. Parser or substrate prior

Question: does the parser, verifier, DSL, action space, or canonicalizer does the work?
Typical absorber: shared-or-charged substrate baseline.
Token family: `ABSORBED_BY_PARSER_OR_SUBSTRATE_PRIOR`.
Inputs received: same public transcript, same substrate where applicable, same examples, same counterexamples, and same scorer.
Costs charged: adapter bits, executable program bits, query answers, human substrate, residual bindings, and runtime where material.
Failure mode: the rung is inconclusive if it is not implemented natively enough for the domain.

### A.3. Teaching dimension

Question: does the packet is a small teaching set?
Typical absorber: finite teaching-set solver or version-space search.
Token family: `ABSORBED_BY_TEACHING_DIMENSION`.
Inputs received: same public transcript, same substrate where applicable, same examples, same counterexamples, and same scorer.
Costs charged: adapter bits, executable program bits, query answers, human substrate, residual bindings, and runtime where material.
Failure mode: the rung is inconclusive if it is not implemented natively enough for the domain.

### A.4. Active learning

Question: does questions isolate the target cheaply?
Typical absorber: query planner, group tests, active counterexample search.
Token family: `ABSORBED_BY_ACTIVE_LEARNING`.
Inputs received: same public transcript, same substrate where applicable, same examples, same counterexamples, and same scorer.
Costs charged: adapter bits, executable program bits, query answers, human substrate, residual bindings, and runtime where material.
Failure mode: the rung is inconclusive if it is not implemented natively enough for the domain.

### A.5. PBE or CEGIS

Question: does a program can be synthesized from the same evidence?
Typical absorber: PBE, CEGIS, SyGuS, ILP, relational rule search.
Token family: `ABSORBED_BY_CEGIS_OR_SYNTHESIS`.
Inputs received: same public transcript, same substrate where applicable, same examples, same counterexamples, and same scorer.
Costs charged: adapter bits, executable program bits, query answers, human substrate, residual bindings, and runtime where material.
Failure mode: the rung is inconclusive if it is not implemented natively enough for the domain.

### A.6. Library learning

Question: does reusable macros explain transfer?
Typical absorber: MDL library, DreamCoder-style library, e-graph composition.
Token family: `ABSORBED_BY_LIBRARY_LEARNING`.
Inputs received: same public transcript, same substrate where applicable, same examples, same counterexamples, and same scorer.
Costs charged: adapter bits, executable program bits, query answers, human substrate, residual bindings, and runtime where material.
Failure mode: the rung is inconclusive if it is not implemented natively enough for the domain.

### A.7. Schema or binding

Question: does role bindings solve the task?
Typical absorber: schema matcher, binding fingerprint, entity resolver.
Token family: `ABSORBED_BY_SCHEMA_OR_BINDING`.
Inputs received: same public transcript, same substrate where applicable, same examples, same counterexamples, and same scorer.
Costs charged: adapter bits, executable program bits, query answers, human substrate, residual bindings, and runtime where material.
Failure mode: the rung is inconclusive if it is not implemented natively enough for the domain.

### A.8. Domain tool

Question: does the task is native to a known local tool?
Typical absorber: SQL, spreadsheets, SAT, SMT, ILP, ASP, CSP.
Token family: `ABSORBED_BY_DOMAIN_TOOL`.
Inputs received: same public transcript, same substrate where applicable, same examples, same counterexamples, and same scorer.
Costs charged: adapter bits, executable program bits, query answers, human substrate, residual bindings, and runtime where material.
Failure mode: the rung is inconclusive if it is not implemented natively enough for the domain.

### A.9. Nuisance oracle

Question: does difficulty vanishes when nuisance is removed?
Typical absorber: relevant-feature oracle or invariant oracle.
Token family: `ABSORBED_BY_NUISANCE_ORACLE`.
Inputs received: same public transcript, same substrate where applicable, same examples, same counterexamples, and same scorer.
Costs charged: adapter bits, executable program bits, query answers, human substrate, residual bindings, and runtime where material.
Failure mode: the rung is inconclusive if it is not implemented natively enough for the domain.

### A.10. Constraint discovery

Question: does the artifact is a constraint theory?
Typical absorber: rank solver, SAT/SMT encoding, table-constraint learner.
Token family: `ABSORBED_BY_CONSTRAINT_DISCOVERY`.
Inputs received: same public transcript, same substrate where applicable, same examples, same counterexamples, and same scorer.
Costs charged: adapter bits, executable program bits, query answers, human substrate, residual bindings, and runtime where material.
Failure mode: the rung is inconclusive if it is not implemented natively enough for the domain.

### A.11. Generator family

Question: does public statistics identify the synthetic generator?
Typical absorber: template classifier, compression classifier, graph-kernel probe.
Token family: `ABSORBED_BY_GENERATOR_FAMILY_IDENTIFICATION`.
Inputs received: same public transcript, same substrate where applicable, same examples, same counterexamples, and same scorer.
Costs charged: adapter bits, executable program bits, query answers, human substrate, residual bindings, and runtime where material.
Failure mode: the rung is inconclusive if it is not implemented natively enough for the domain.

### A.12. Post-hoc compression

Question: does the artifact summarizes behavior after success?
Typical absorber: frozen-before-score compression control.
Token family: `ABSORBED_BY_POST_HOC_COMPRESSION`.
Inputs received: same public transcript, same substrate where applicable, same examples, same counterexamples, and same scorer.
Costs charged: adapter bits, executable program bits, query answers, human substrate, residual bindings, and runtime where material.
Failure mode: the rung is inconclusive if it is not implemented natively enough for the domain.

## Appendix B: Cost Ledger

| Symbol | Meaning |
|---|---|
| F | reusable frame, rule, grammar, verifier, representation, or method bits |
| B_i | task-specific binding bits |
| P_i | executable per-task program or policy bits |
| E_i | examples and labels |
| C_i | counterexamples and active-query answers |
| V_i | verifier obligations or proof/test clauses |
| R_i | residual teaching bits after reusable structure is installed |
| A_i | abstention policy bits and abstention labels |
| L | learned library or macro bits |
| H | human-authored parser, substrate, design, verifier, and adapter work |
| O | operation ontology or typed action-space supply |
| N | nuisance-removal, invariant, or oracle information |

### B.F

Definition: reusable frame, rule, grammar, verifier, representation, or method bits.
Reviewer question: was this field shared, charged, or silently granted to the claimed system?
Reviewer question: does a baseline receive an equivalent affordance or only a text description?
Failure consequence: lower the claim ceiling or emit the most specific absorber token.

### B.B_i

Definition: task-specific binding bits.
Reviewer question: was this field shared, charged, or silently granted to the claimed system?
Reviewer question: does a baseline receive an equivalent affordance or only a text description?
Failure consequence: lower the claim ceiling or emit the most specific absorber token.

### B.P_i

Definition: executable per-task program or policy bits.
Reviewer question: was this field shared, charged, or silently granted to the claimed system?
Reviewer question: does a baseline receive an equivalent affordance or only a text description?
Failure consequence: lower the claim ceiling or emit the most specific absorber token.

### B.E_i

Definition: examples and labels.
Reviewer question: was this field shared, charged, or silently granted to the claimed system?
Reviewer question: does a baseline receive an equivalent affordance or only a text description?
Failure consequence: lower the claim ceiling or emit the most specific absorber token.

### B.C_i

Definition: counterexamples and active-query answers.
Reviewer question: was this field shared, charged, or silently granted to the claimed system?
Reviewer question: does a baseline receive an equivalent affordance or only a text description?
Failure consequence: lower the claim ceiling or emit the most specific absorber token.

### B.V_i

Definition: verifier obligations or proof/test clauses.
Reviewer question: was this field shared, charged, or silently granted to the claimed system?
Reviewer question: does a baseline receive an equivalent affordance or only a text description?
Failure consequence: lower the claim ceiling or emit the most specific absorber token.

### B.R_i

Definition: residual teaching bits after reusable structure is installed.
Reviewer question: was this field shared, charged, or silently granted to the claimed system?
Reviewer question: does a baseline receive an equivalent affordance or only a text description?
Failure consequence: lower the claim ceiling or emit the most specific absorber token.

### B.A_i

Definition: abstention policy bits and abstention labels.
Reviewer question: was this field shared, charged, or silently granted to the claimed system?
Reviewer question: does a baseline receive an equivalent affordance or only a text description?
Failure consequence: lower the claim ceiling or emit the most specific absorber token.

### B.L

Definition: learned library or macro bits.
Reviewer question: was this field shared, charged, or silently granted to the claimed system?
Reviewer question: does a baseline receive an equivalent affordance or only a text description?
Failure consequence: lower the claim ceiling or emit the most specific absorber token.

### B.H

Definition: human-authored parser, substrate, design, verifier, and adapter work.
Reviewer question: was this field shared, charged, or silently granted to the claimed system?
Reviewer question: does a baseline receive an equivalent affordance or only a text description?
Failure consequence: lower the claim ceiling or emit the most specific absorber token.

### B.O

Definition: operation ontology or typed action-space supply.
Reviewer question: was this field shared, charged, or silently granted to the claimed system?
Reviewer question: does a baseline receive an equivalent affordance or only a text description?
Failure consequence: lower the claim ceiling or emit the most specific absorber token.

### B.N

Definition: nuisance-removal, invariant, or oracle information.
Reviewer question: was this field shared, charged, or silently granted to the claimed system?
Reviewer question: does a baseline receive an equivalent affordance or only a text description?
Failure consequence: lower the claim ceiling or emit the most specific absorber token.

## Appendix C: Case-Study Fact Ledgers

### C.C1. FrameSeed Boolean

Artifact: `experiments/frameseed0_b28_hidden_hfa.json`.
Code: `code/frameseed0_measurement.py`.
Token: `FRAMESEED_T3_ABSORBED_BY_TEACHING_DIMENSION`.
Lesson: Perfect hidden HFA did not establish frame transmission because finite teaching and search reached the same hidden accuracy.

- target bundles: 10240
- hidden queries per system: 15728640
- L3 mean HFA: 1.0
- L3 min HFA: 1.0
- TD-H0 min HFA: 1.0
- L1 active min HFA: 1.0
- L2 CEGIS min HFA: 1.0
- RAG min HFA: 1.0
- nuisance oracle min HFA: 1.0
- library-learning min HFA: 1.0
- packet growth alpha: 0.0028930015823335495
- role max HFA std: 0.0
- audit failures: 0

### C.C2. FrameSeed SHEETS-0

Artifact: `experiments/frameseed_sheets0_b31_hidden_hfa.json`.
Code: `code/frameseed_sheets0_measurement.py`.
Token: `FRAMESEED_SHEETS0_ABSORBED_BY_SCHEMA_BINDING`.
Lesson: The typed packet was not the active ingredient because charged bindings alone preserved perfect HFA and packet erasure produced a zero-point drop.

- target bundles: 15360
- hidden queries per system: 15728640
- L3 mean HFA: 1.0
- L3 min HFA: 1.0
- binding-only HFA: 1.0
- packet-erasure drop pp: 0.0
- PBE/PROSE min HFA: 1.0
- data-wrangling min HFA: 1.0
- typed CEGIS exact min HFA: 1.0
- typed MDL library min HFA: 1.0
- active goal disambiguation min HFA: 1.0
- non-Boolean output fraction: 0.8001302083333334
- binding growth alpha: 0.0
- AFTD all-in passed: false
- composition gate passed: false
- audit failures: 0

### C.C3. WGD-0 Toy Grammar

Artifact: `experiments/wgd0_b37_hidden_measurement.json`.
Code: `code/wgd0_measurement.py`.
Token: `WGD_ABSORBED_BY_SCHEMA_OR_BINDING_DISCOVERY`.
Lesson: The grammar candidate solved, but cheaper role-binding and feedback-program induction solved the same hidden cases.

- hidden worlds: 128
- hidden cases: 1536
- scored predictions: 6144
- WGD HFA: 1.0
- WGD min-family HFA: 1.0
- schema-binding HFA: 1.0
- PBE/CEGIS HFA: 1.0
- majority HFA: 0.19401041666666666
- WGD mean cost bits: 35690.125
- schema mean cost bits: 3147.1875
- PBE mean cost bits: 3835.1875
- schema cost ratio: 0.08818090438181429
- PBE cost ratio: 0.10745794529999544
- pre-hidden audit passed: true
- post-hidden code changes: false

### C.C4. WGD-0 Hard Domain

Artifact: `experiments/wgd0_b38_hidden_measurement.json`.
Code: `code/wgd0_b38_hard_domain.py`.
Token: `WGD_ABSORBED_BY_CONSTRAINT_DISCOVERY`.
Lesson: The domain finally beat brute enumeration, yet the learnable structure was exactly what a GF(2) constraint solver exploits.

- worlds: 8
- cases: 256
- scored predictions: 1536
- rule count: 64
- state bits: 128
- candidate space: 18446744073709551616
- candidate-space log2: 64
- ordered-composition-space log2: 144.0
- enumeration budget per case: 8000
- enumeration fraction per case: 4.336808689942018e-16
- WGD HFA: 1.0
- constraint HFA: 1.0
- lexicographic HFA: 0.25
- size-first HFA: 0.25
- random HFA: 0.25
- meet-in-middle truncated HFA: 0.25
- composition HFA: 1.0
- repair success: 1.0
- abstention recall: 1.0
- WGD mean cost bits per world: 88080.0
- constraint mean cost bits per world: 85208.0
- constraint cost ratio: 0.9673932788374205
- constraint absorbs: true
- pre-hidden audit passed: true
- post-hidden code changes: false

## Appendix D: Claim Ceilings

Token state: `SIGNAL`.
Allowed claim: The tested claim survived declared absorbers under the frozen protocol and measured scale.
Banned move: narrating mixed evidence upward into a stronger claim.

Token state: `ABSORBED`.
Allowed claim: The apparent discovery is better explained by the named ordinary route.
Banned move: narrating mixed evidence upward into a stronger claim.

Token state: `VOID`.
Allowed claim: No scientific claim should be made from this hidden opening except that the protocol failed.
Banned move: narrating mixed evidence upward into a stronger claim.

Token state: `TRAP_DOMAIN_DEGENERATE`.
Allowed claim: The domain did not test the intended function.
Banned move: narrating mixed evidence upward into a stronger claim.

Token state: `NEGATIVE`.
Allowed claim: The claimed system did not meet its own functional threshold.
Banned move: narrating mixed evidence upward into a stronger claim.

Token state: `INCONCLUSIVE`.
Allowed claim: The claim remains unproven because a required measurement or absorber is missing.
Banned move: narrating mixed evidence upward into a stronger claim.

## Appendix E: Equal-Information Failure Catalog

E.1. Failure mode: inert text baseline.
Review action: identify which system receives the relevant affordance.
Review action: require a shared representation or a charged adapter.
Terminal consequence: void, absorption, trap, or inconclusive depending on severity.

E.2. Failure mode: parser asymmetry.
Review action: identify which system receives the relevant affordance.
Review action: require a shared representation or a charged adapter.
Terminal consequence: void, absorption, trap, or inconclusive depending on severity.

E.3. Failure mode: typed-object asymmetry.
Review action: identify which system receives the relevant affordance.
Review action: require a shared representation or a charged adapter.
Terminal consequence: void, absorption, trap, or inconclusive depending on severity.

E.4. Failure mode: unit canonicalizer asymmetry.
Review action: identify which system receives the relevant affordance.
Review action: require a shared representation or a charged adapter.
Terminal consequence: void, absorption, trap, or inconclusive depending on severity.

E.5. Failure mode: row-order asymmetry.
Review action: identify which system receives the relevant affordance.
Review action: require a shared representation or a charged adapter.
Terminal consequence: void, absorption, trap, or inconclusive depending on severity.

E.6. Failure mode: operation grammar asymmetry.
Review action: identify which system receives the relevant affordance.
Review action: require a shared representation or a charged adapter.
Terminal consequence: void, absorption, trap, or inconclusive depending on severity.

E.7. Failure mode: verifier asymmetry.
Review action: identify which system receives the relevant affordance.
Review action: require a shared representation or a charged adapter.
Terminal consequence: void, absorption, trap, or inconclusive depending on severity.

E.8. Failure mode: query-channel asymmetry.
Review action: identify which system receives the relevant affordance.
Review action: require a shared representation or a charged adapter.
Terminal consequence: void, absorption, trap, or inconclusive depending on severity.

E.9. Failure mode: example-count asymmetry.
Review action: identify which system receives the relevant affordance.
Review action: require a shared representation or a charged adapter.
Terminal consequence: void, absorption, trap, or inconclusive depending on severity.

E.10. Failure mode: counterexample-quality asymmetry.
Review action: identify which system receives the relevant affordance.
Review action: require a shared representation or a charged adapter.
Terminal consequence: void, absorption, trap, or inconclusive depending on severity.

E.11. Failure mode: task-binding denial.
Review action: identify which system receives the relevant affordance.
Review action: require a shared representation or a charged adapter.
Terminal consequence: void, absorption, trap, or inconclusive depending on severity.

E.12. Failure mode: public-role-name denial.
Review action: identify which system receives the relevant affordance.
Review action: require a shared representation or a charged adapter.
Terminal consequence: void, absorption, trap, or inconclusive depending on severity.

E.13. Failure mode: hidden-family hint leakage.
Review action: identify which system receives the relevant affordance.
Review action: require a shared representation or a charged adapter.
Terminal consequence: void, absorption, trap, or inconclusive depending on severity.

E.14. Failure mode: natural-language semantic asymmetry.
Review action: identify which system receives the relevant affordance.
Review action: require a shared representation or a charged adapter.
Terminal consequence: void, absorption, trap, or inconclusive depending on severity.

E.15. Failure mode: human substrate omission.
Review action: identify which system receives the relevant affordance.
Review action: require a shared representation or a charged adapter.
Terminal consequence: void, absorption, trap, or inconclusive depending on severity.

E.16. Failure mode: adapter-cost omission.
Review action: identify which system receives the relevant affordance.
Review action: require a shared representation or a charged adapter.
Terminal consequence: void, absorption, trap, or inconclusive depending on severity.

E.17. Failure mode: post-hidden timeout change.
Review action: identify which system receives the relevant affordance.
Review action: require a shared representation or a charged adapter.
Terminal consequence: void, absorption, trap, or inconclusive depending on severity.

E.18. Failure mode: post-hidden scorer change.
Review action: identify which system receives the relevant affordance.
Review action: require a shared representation or a charged adapter.
Terminal consequence: void, absorption, trap, or inconclusive depending on severity.

E.19. Failure mode: post-hidden token change.
Review action: identify which system receives the relevant affordance.
Review action: require a shared representation or a charged adapter.
Terminal consequence: void, absorption, trap, or inconclusive depending on severity.

E.20. Failure mode: post-hidden baseline selection.
Review action: identify which system receives the relevant affordance.
Review action: require a shared representation or a charged adapter.
Terminal consequence: void, absorption, trap, or inconclusive depending on severity.

E.21. Failure mode: proxy absorber narrated as native.
Review action: identify which system receives the relevant affordance.
Review action: require a shared representation or a charged adapter.
Terminal consequence: void, absorption, trap, or inconclusive depending on severity.

E.22. Failure mode: weak baseline roster.
Review action: identify which system receives the relevant affordance.
Review action: require a shared representation or a charged adapter.
Terminal consequence: void, absorption, trap, or inconclusive depending on severity.

E.23. Failure mode: synthetic generator fingerprints.
Review action: identify which system receives the relevant affordance.
Review action: require a shared representation or a charged adapter.
Terminal consequence: void, absorption, trap, or inconclusive depending on severity.

E.24. Failure mode: component erasure omitted.
Review action: identify which system receives the relevant affordance.
Review action: require a shared representation or a charged adapter.
Terminal consequence: void, absorption, trap, or inconclusive depending on severity.

E.25. Failure mode: randomized-label control omitted.
Review action: identify which system receives the relevant affordance.
Review action: require a shared representation or a charged adapter.
Terminal consequence: void, absorption, trap, or inconclusive depending on severity.

E.26. Failure mode: role permutation omitted.
Review action: identify which system receives the relevant affordance.
Review action: require a shared representation or a charged adapter.
Terminal consequence: void, absorption, trap, or inconclusive depending on severity.

E.27. Failure mode: schema permutation omitted.
Review action: identify which system receives the relevant affordance.
Review action: require a shared representation or a charged adapter.
Terminal consequence: void, absorption, trap, or inconclusive depending on severity.

E.28. Failure mode: unit permutation omitted.
Review action: identify which system receives the relevant affordance.
Review action: require a shared representation or a charged adapter.
Terminal consequence: void, absorption, trap, or inconclusive depending on severity.

E.29. Failure mode: row-order permutation omitted.
Review action: identify which system receives the relevant affordance.
Review action: require a shared representation or a charged adapter.
Terminal consequence: void, absorption, trap, or inconclusive depending on severity.

E.30. Failure mode: nuisance oracle omitted.
Review action: identify which system receives the relevant affordance.
Review action: require a shared representation or a charged adapter.
Terminal consequence: void, absorption, trap, or inconclusive depending on severity.

E.31. Failure mode: domain tool omitted.
Review action: identify which system receives the relevant affordance.
Review action: require a shared representation or a charged adapter.
Terminal consequence: void, absorption, trap, or inconclusive depending on severity.

E.32. Failure mode: constraint solver omitted.
Review action: identify which system receives the relevant affordance.
Review action: require a shared representation or a charged adapter.
Terminal consequence: void, absorption, trap, or inconclusive depending on severity.

E.33. Failure mode: library learner omitted.
Review action: identify which system receives the relevant affordance.
Review action: require a shared representation or a charged adapter.
Terminal consequence: void, absorption, trap, or inconclusive depending on severity.

E.34. Failure mode: teaching dimension omitted.
Review action: identify which system receives the relevant affordance.
Review action: require a shared representation or a charged adapter.
Terminal consequence: void, absorption, trap, or inconclusive depending on severity.

E.35. Failure mode: active learner omitted.
Review action: identify which system receives the relevant affordance.
Review action: require a shared representation or a charged adapter.
Terminal consequence: void, absorption, trap, or inconclusive depending on severity.

E.36. Failure mode: PBE omitted.
Review action: identify which system receives the relevant affordance.
Review action: require a shared representation or a charged adapter.
Terminal consequence: void, absorption, trap, or inconclusive depending on severity.

E.37. Failure mode: generator classifier omitted.
Review action: identify which system receives the relevant affordance.
Review action: require a shared representation or a charged adapter.
Terminal consequence: void, absorption, trap, or inconclusive depending on severity.

E.38. Failure mode: post-hoc grammar not frozen.
Review action: identify which system receives the relevant affordance.
Review action: require a shared representation or a charged adapter.
Terminal consequence: void, absorption, trap, or inconclusive depending on severity.

E.39. Failure mode: duplicate siblings.
Review action: identify which system receives the relevant affordance.
Review action: require a shared representation or a charged adapter.
Terminal consequence: void, absorption, trap, or inconclusive depending on severity.

E.40. Failure mode: residual binding bits omitted.
Review action: identify which system receives the relevant affordance.
Review action: require a shared representation or a charged adapter.
Terminal consequence: void, absorption, trap, or inconclusive depending on severity.

## Appendix F: B39 Twenty-Iteration Draft Log

F.1. Iteration 1: grounding.
Action: deepen, extend, or attack the previous draft section.
Adversarial question: would a hostile reviewer still be able to narrate a stronger claim than the token allows?
Revision rule: if yes, lower the claim ceiling or add the missing absorber, cost field, or causal test.
Status: incorporated into this draft.

F.2. Iteration 2: claim ceiling.
Action: deepen, extend, or attack the previous draft section.
Adversarial question: would a hostile reviewer still be able to narrate a stronger claim than the token allows?
Revision rule: if yes, lower the claim ceiling or add the missing absorber, cost field, or causal test.
Status: incorporated into this draft.

F.3. Iteration 3: definition pass.
Action: deepen, extend, or attack the previous draft section.
Adversarial question: would a hostile reviewer still be able to narrate a stronger claim than the token allows?
Revision rule: if yes, lower the claim ceiling or add the missing absorber, cost field, or causal test.
Status: incorporated into this draft.

F.4. Iteration 4: token pass.
Action: deepen, extend, or attack the previous draft section.
Adversarial question: would a hostile reviewer still be able to narrate a stronger claim than the token allows?
Revision rule: if yes, lower the claim ceiling or add the missing absorber, cost field, or causal test.
Status: incorporated into this draft.

F.5. Iteration 5: ladder pass.
Action: deepen, extend, or attack the previous draft section.
Adversarial question: would a hostile reviewer still be able to narrate a stronger claim than the token allows?
Revision rule: if yes, lower the claim ceiling or add the missing absorber, cost field, or causal test.
Status: incorporated into this draft.

F.6. Iteration 6: equal-information pass.
Action: deepen, extend, or attack the previous draft section.
Adversarial question: would a hostile reviewer still be able to narrate a stronger claim than the token allows?
Revision rule: if yes, lower the claim ceiling or add the missing absorber, cost field, or causal test.
Status: incorporated into this draft.

F.7. Iteration 7: cost pass.
Action: deepen, extend, or attack the previous draft section.
Adversarial question: would a hostile reviewer still be able to narrate a stronger claim than the token allows?
Revision rule: if yes, lower the claim ceiling or add the missing absorber, cost field, or causal test.
Status: incorporated into this draft.

F.8. Iteration 8: hidden-open pass.
Action: deepen, extend, or attack the previous draft section.
Adversarial question: would a hostile reviewer still be able to narrate a stronger claim than the token allows?
Revision rule: if yes, lower the claim ceiling or add the missing absorber, cost field, or causal test.
Status: incorporated into this draft.

F.9. Iteration 9: causality pass.
Action: deepen, extend, or attack the previous draft section.
Adversarial question: would a hostile reviewer still be able to narrate a stronger claim than the token allows?
Revision rule: if yes, lower the claim ceiling or add the missing absorber, cost field, or causal test.
Status: incorporated into this draft.

F.10. Iteration 10: AFTD pass.
Action: deepen, extend, or attack the previous draft section.
Adversarial question: would a hostile reviewer still be able to narrate a stronger claim than the token allows?
Revision rule: if yes, lower the claim ceiling or add the missing absorber, cost field, or causal test.
Status: incorporated into this draft.

F.11. Iteration 11: Boolean case pass.
Action: deepen, extend, or attack the previous draft section.
Adversarial question: would a hostile reviewer still be able to narrate a stronger claim than the token allows?
Revision rule: if yes, lower the claim ceiling or add the missing absorber, cost field, or causal test.
Status: incorporated into this draft.

F.12. Iteration 12: SHEETS case pass.
Action: deepen, extend, or attack the previous draft section.
Adversarial question: would a hostile reviewer still be able to narrate a stronger claim than the token allows?
Revision rule: if yes, lower the claim ceiling or add the missing absorber, cost field, or causal test.
Status: incorporated into this draft.

F.13. Iteration 13: WGD toy pass.
Action: deepen, extend, or attack the previous draft section.
Adversarial question: would a hostile reviewer still be able to narrate a stronger claim than the token allows?
Revision rule: if yes, lower the claim ceiling or add the missing absorber, cost field, or causal test.
Status: incorporated into this draft.

F.14. Iteration 14: WGD hard pass.
Action: deepen, extend, or attack the previous draft section.
Adversarial question: would a hostile reviewer still be able to narrate a stronger claim than the token allows?
Revision rule: if yes, lower the claim ceiling or add the missing absorber, cost field, or causal test.
Status: incorporated into this draft.

F.15. Iteration 15: native absorber theater pass.
Action: deepen, extend, or attack the previous draft section.
Adversarial question: would a hostile reviewer still be able to narrate a stronger claim than the token allows?
Revision rule: if yes, lower the claim ceiling or add the missing absorber, cost field, or causal test.
Status: incorporated into this draft.

F.16. Iteration 16: negative-result pass.
Action: deepen, extend, or attack the previous draft section.
Adversarial question: would a hostile reviewer still be able to narrate a stronger claim than the token allows?
Revision rule: if yes, lower the claim ceiling or add the missing absorber, cost field, or causal test.
Status: incorporated into this draft.

F.17. Iteration 17: positive-result pass.
Action: deepen, extend, or attack the previous draft section.
Adversarial question: would a hostile reviewer still be able to narrate a stronger claim than the token allows?
Revision rule: if yes, lower the claim ceiling or add the missing absorber, cost field, or causal test.
Status: incorporated into this draft.

F.18. Iteration 18: limitations pass.
Action: deepen, extend, or attack the previous draft section.
Adversarial question: would a hostile reviewer still be able to narrate a stronger claim than the token allows?
Revision rule: if yes, lower the claim ceiling or add the missing absorber, cost field, or causal test.
Status: incorporated into this draft.

F.19. Iteration 19: checklist pass.
Action: deepen, extend, or attack the previous draft section.
Adversarial question: would a hostile reviewer still be able to narrate a stronger claim than the token allows?
Revision rule: if yes, lower the claim ceiling or add the missing absorber, cost field, or causal test.
Status: incorporated into this draft.

F.20. Iteration 20: adversarial appendix pass.
Action: deepen, extend, or attack the previous draft section.
Adversarial question: would a hostile reviewer still be able to narrate a stronger claim than the token allows?
Revision rule: if yes, lower the claim ceiling or add the missing absorber, cost field, or causal test.
Status: incorporated into this draft.

## Appendix G: Surface-by-Rung Audit Matrix

G.1. Surface: functional accuracy.
Rung: Representation prior.
Audit question: does public features expose the answer?
Required evidence: run or justify `feature parser, type tags, names, ontology` under equal information.
Absorption token family: `ABSORBED_BY_REPRESENTATION_PRIOR`.
Claim ceiling if untested: inconclusive for this surface.

G.2. Surface: functional accuracy.
Rung: Parser or substrate prior.
Audit question: does the parser, verifier, DSL, action space, or canonicalizer does the work?
Required evidence: run or justify `shared-or-charged substrate baseline` under equal information.
Absorption token family: `ABSORBED_BY_PARSER_OR_SUBSTRATE_PRIOR`.
Claim ceiling if untested: inconclusive for this surface.

G.3. Surface: functional accuracy.
Rung: Teaching dimension.
Audit question: does the packet is a small teaching set?
Required evidence: run or justify `finite teaching-set solver or version-space search` under equal information.
Absorption token family: `ABSORBED_BY_TEACHING_DIMENSION`.
Claim ceiling if untested: inconclusive for this surface.

G.4. Surface: functional accuracy.
Rung: Active learning.
Audit question: does questions isolate the target cheaply?
Required evidence: run or justify `query planner, group tests, active counterexample search` under equal information.
Absorption token family: `ABSORBED_BY_ACTIVE_LEARNING`.
Claim ceiling if untested: inconclusive for this surface.

G.5. Surface: functional accuracy.
Rung: PBE or CEGIS.
Audit question: does a program can be synthesized from the same evidence?
Required evidence: run or justify `PBE, CEGIS, SyGuS, ILP, relational rule search` under equal information.
Absorption token family: `ABSORBED_BY_CEGIS_OR_SYNTHESIS`.
Claim ceiling if untested: inconclusive for this surface.

G.6. Surface: functional accuracy.
Rung: Library learning.
Audit question: does reusable macros explain transfer?
Required evidence: run or justify `MDL library, DreamCoder-style library, e-graph composition` under equal information.
Absorption token family: `ABSORBED_BY_LIBRARY_LEARNING`.
Claim ceiling if untested: inconclusive for this surface.

G.7. Surface: functional accuracy.
Rung: Schema or binding.
Audit question: does role bindings solve the task?
Required evidence: run or justify `schema matcher, binding fingerprint, entity resolver` under equal information.
Absorption token family: `ABSORBED_BY_SCHEMA_OR_BINDING`.
Claim ceiling if untested: inconclusive for this surface.

G.8. Surface: functional accuracy.
Rung: Domain tool.
Audit question: does the task is native to a known local tool?
Required evidence: run or justify `SQL, spreadsheets, SAT, SMT, ILP, ASP, CSP` under equal information.
Absorption token family: `ABSORBED_BY_DOMAIN_TOOL`.
Claim ceiling if untested: inconclusive for this surface.

G.9. Surface: functional accuracy.
Rung: Nuisance oracle.
Audit question: does difficulty vanishes when nuisance is removed?
Required evidence: run or justify `relevant-feature oracle or invariant oracle` under equal information.
Absorption token family: `ABSORBED_BY_NUISANCE_ORACLE`.
Claim ceiling if untested: inconclusive for this surface.

G.10. Surface: functional accuracy.
Rung: Constraint discovery.
Audit question: does the artifact is a constraint theory?
Required evidence: run or justify `rank solver, SAT/SMT encoding, table-constraint learner` under equal information.
Absorption token family: `ABSORBED_BY_CONSTRAINT_DISCOVERY`.
Claim ceiling if untested: inconclusive for this surface.

G.11. Surface: functional accuracy.
Rung: Generator family.
Audit question: does public statistics identify the synthetic generator?
Required evidence: run or justify `template classifier, compression classifier, graph-kernel probe` under equal information.
Absorption token family: `ABSORBED_BY_GENERATOR_FAMILY_IDENTIFICATION`.
Claim ceiling if untested: inconclusive for this surface.

G.12. Surface: functional accuracy.
Rung: Post-hoc compression.
Audit question: does the artifact summarizes behavior after success?
Required evidence: run or justify `frozen-before-score compression control` under equal information.
Absorption token family: `ABSORBED_BY_POST_HOC_COMPRESSION`.
Claim ceiling if untested: inconclusive for this surface.

G.13. Surface: minimum per-family accuracy.
Rung: Representation prior.
Audit question: does public features expose the answer?
Required evidence: run or justify `feature parser, type tags, names, ontology` under equal information.
Absorption token family: `ABSORBED_BY_REPRESENTATION_PRIOR`.
Claim ceiling if untested: inconclusive for this surface.

G.14. Surface: minimum per-family accuracy.
Rung: Parser or substrate prior.
Audit question: does the parser, verifier, DSL, action space, or canonicalizer does the work?
Required evidence: run or justify `shared-or-charged substrate baseline` under equal information.
Absorption token family: `ABSORBED_BY_PARSER_OR_SUBSTRATE_PRIOR`.
Claim ceiling if untested: inconclusive for this surface.

G.15. Surface: minimum per-family accuracy.
Rung: Teaching dimension.
Audit question: does the packet is a small teaching set?
Required evidence: run or justify `finite teaching-set solver or version-space search` under equal information.
Absorption token family: `ABSORBED_BY_TEACHING_DIMENSION`.
Claim ceiling if untested: inconclusive for this surface.

G.16. Surface: minimum per-family accuracy.
Rung: Active learning.
Audit question: does questions isolate the target cheaply?
Required evidence: run or justify `query planner, group tests, active counterexample search` under equal information.
Absorption token family: `ABSORBED_BY_ACTIVE_LEARNING`.
Claim ceiling if untested: inconclusive for this surface.

G.17. Surface: minimum per-family accuracy.
Rung: PBE or CEGIS.
Audit question: does a program can be synthesized from the same evidence?
Required evidence: run or justify `PBE, CEGIS, SyGuS, ILP, relational rule search` under equal information.
Absorption token family: `ABSORBED_BY_CEGIS_OR_SYNTHESIS`.
Claim ceiling if untested: inconclusive for this surface.

G.18. Surface: minimum per-family accuracy.
Rung: Library learning.
Audit question: does reusable macros explain transfer?
Required evidence: run or justify `MDL library, DreamCoder-style library, e-graph composition` under equal information.
Absorption token family: `ABSORBED_BY_LIBRARY_LEARNING`.
Claim ceiling if untested: inconclusive for this surface.

G.19. Surface: minimum per-family accuracy.
Rung: Schema or binding.
Audit question: does role bindings solve the task?
Required evidence: run or justify `schema matcher, binding fingerprint, entity resolver` under equal information.
Absorption token family: `ABSORBED_BY_SCHEMA_OR_BINDING`.
Claim ceiling if untested: inconclusive for this surface.

G.20. Surface: minimum per-family accuracy.
Rung: Domain tool.
Audit question: does the task is native to a known local tool?
Required evidence: run or justify `SQL, spreadsheets, SAT, SMT, ILP, ASP, CSP` under equal information.
Absorption token family: `ABSORBED_BY_DOMAIN_TOOL`.
Claim ceiling if untested: inconclusive for this surface.

G.21. Surface: minimum per-family accuracy.
Rung: Nuisance oracle.
Audit question: does difficulty vanishes when nuisance is removed?
Required evidence: run or justify `relevant-feature oracle or invariant oracle` under equal information.
Absorption token family: `ABSORBED_BY_NUISANCE_ORACLE`.
Claim ceiling if untested: inconclusive for this surface.

G.22. Surface: minimum per-family accuracy.
Rung: Constraint discovery.
Audit question: does the artifact is a constraint theory?
Required evidence: run or justify `rank solver, SAT/SMT encoding, table-constraint learner` under equal information.
Absorption token family: `ABSORBED_BY_CONSTRAINT_DISCOVERY`.
Claim ceiling if untested: inconclusive for this surface.

G.23. Surface: minimum per-family accuracy.
Rung: Generator family.
Audit question: does public statistics identify the synthetic generator?
Required evidence: run or justify `template classifier, compression classifier, graph-kernel probe` under equal information.
Absorption token family: `ABSORBED_BY_GENERATOR_FAMILY_IDENTIFICATION`.
Claim ceiling if untested: inconclusive for this surface.

G.24. Surface: minimum per-family accuracy.
Rung: Post-hoc compression.
Audit question: does the artifact summarizes behavior after success?
Required evidence: run or justify `frozen-before-score compression control` under equal information.
Absorption token family: `ABSORBED_BY_POST_HOC_COMPRESSION`.
Claim ceiling if untested: inconclusive for this surface.

G.25. Surface: repair success.
Rung: Representation prior.
Audit question: does public features expose the answer?
Required evidence: run or justify `feature parser, type tags, names, ontology` under equal information.
Absorption token family: `ABSORBED_BY_REPRESENTATION_PRIOR`.
Claim ceiling if untested: inconclusive for this surface.

G.26. Surface: repair success.
Rung: Parser or substrate prior.
Audit question: does the parser, verifier, DSL, action space, or canonicalizer does the work?
Required evidence: run or justify `shared-or-charged substrate baseline` under equal information.
Absorption token family: `ABSORBED_BY_PARSER_OR_SUBSTRATE_PRIOR`.
Claim ceiling if untested: inconclusive for this surface.

G.27. Surface: repair success.
Rung: Teaching dimension.
Audit question: does the packet is a small teaching set?
Required evidence: run or justify `finite teaching-set solver or version-space search` under equal information.
Absorption token family: `ABSORBED_BY_TEACHING_DIMENSION`.
Claim ceiling if untested: inconclusive for this surface.

G.28. Surface: repair success.
Rung: Active learning.
Audit question: does questions isolate the target cheaply?
Required evidence: run or justify `query planner, group tests, active counterexample search` under equal information.
Absorption token family: `ABSORBED_BY_ACTIVE_LEARNING`.
Claim ceiling if untested: inconclusive for this surface.

G.29. Surface: repair success.
Rung: PBE or CEGIS.
Audit question: does a program can be synthesized from the same evidence?
Required evidence: run or justify `PBE, CEGIS, SyGuS, ILP, relational rule search` under equal information.
Absorption token family: `ABSORBED_BY_CEGIS_OR_SYNTHESIS`.
Claim ceiling if untested: inconclusive for this surface.

G.30. Surface: repair success.
Rung: Library learning.
Audit question: does reusable macros explain transfer?
Required evidence: run or justify `MDL library, DreamCoder-style library, e-graph composition` under equal information.
Absorption token family: `ABSORBED_BY_LIBRARY_LEARNING`.
Claim ceiling if untested: inconclusive for this surface.

G.31. Surface: repair success.
Rung: Schema or binding.
Audit question: does role bindings solve the task?
Required evidence: run or justify `schema matcher, binding fingerprint, entity resolver` under equal information.
Absorption token family: `ABSORBED_BY_SCHEMA_OR_BINDING`.
Claim ceiling if untested: inconclusive for this surface.

G.32. Surface: repair success.
Rung: Domain tool.
Audit question: does the task is native to a known local tool?
Required evidence: run or justify `SQL, spreadsheets, SAT, SMT, ILP, ASP, CSP` under equal information.
Absorption token family: `ABSORBED_BY_DOMAIN_TOOL`.
Claim ceiling if untested: inconclusive for this surface.

G.33. Surface: repair success.
Rung: Nuisance oracle.
Audit question: does difficulty vanishes when nuisance is removed?
Required evidence: run or justify `relevant-feature oracle or invariant oracle` under equal information.
Absorption token family: `ABSORBED_BY_NUISANCE_ORACLE`.
Claim ceiling if untested: inconclusive for this surface.

G.34. Surface: repair success.
Rung: Constraint discovery.
Audit question: does the artifact is a constraint theory?
Required evidence: run or justify `rank solver, SAT/SMT encoding, table-constraint learner` under equal information.
Absorption token family: `ABSORBED_BY_CONSTRAINT_DISCOVERY`.
Claim ceiling if untested: inconclusive for this surface.

G.35. Surface: repair success.
Rung: Generator family.
Audit question: does public statistics identify the synthetic generator?
Required evidence: run or justify `template classifier, compression classifier, graph-kernel probe` under equal information.
Absorption token family: `ABSORBED_BY_GENERATOR_FAMILY_IDENTIFICATION`.
Claim ceiling if untested: inconclusive for this surface.

G.36. Surface: repair success.
Rung: Post-hoc compression.
Audit question: does the artifact summarizes behavior after success?
Required evidence: run or justify `frozen-before-score compression control` under equal information.
Absorption token family: `ABSORBED_BY_POST_HOC_COMPRESSION`.
Claim ceiling if untested: inconclusive for this surface.

G.37. Surface: repair locality.
Rung: Representation prior.
Audit question: does public features expose the answer?
Required evidence: run or justify `feature parser, type tags, names, ontology` under equal information.
Absorption token family: `ABSORBED_BY_REPRESENTATION_PRIOR`.
Claim ceiling if untested: inconclusive for this surface.

G.38. Surface: repair locality.
Rung: Parser or substrate prior.
Audit question: does the parser, verifier, DSL, action space, or canonicalizer does the work?
Required evidence: run or justify `shared-or-charged substrate baseline` under equal information.
Absorption token family: `ABSORBED_BY_PARSER_OR_SUBSTRATE_PRIOR`.
Claim ceiling if untested: inconclusive for this surface.

G.39. Surface: repair locality.
Rung: Teaching dimension.
Audit question: does the packet is a small teaching set?
Required evidence: run or justify `finite teaching-set solver or version-space search` under equal information.
Absorption token family: `ABSORBED_BY_TEACHING_DIMENSION`.
Claim ceiling if untested: inconclusive for this surface.

G.40. Surface: repair locality.
Rung: Active learning.
Audit question: does questions isolate the target cheaply?
Required evidence: run or justify `query planner, group tests, active counterexample search` under equal information.
Absorption token family: `ABSORBED_BY_ACTIVE_LEARNING`.
Claim ceiling if untested: inconclusive for this surface.

G.41. Surface: repair locality.
Rung: PBE or CEGIS.
Audit question: does a program can be synthesized from the same evidence?
Required evidence: run or justify `PBE, CEGIS, SyGuS, ILP, relational rule search` under equal information.
Absorption token family: `ABSORBED_BY_CEGIS_OR_SYNTHESIS`.
Claim ceiling if untested: inconclusive for this surface.

G.42. Surface: repair locality.
Rung: Library learning.
Audit question: does reusable macros explain transfer?
Required evidence: run or justify `MDL library, DreamCoder-style library, e-graph composition` under equal information.
Absorption token family: `ABSORBED_BY_LIBRARY_LEARNING`.
Claim ceiling if untested: inconclusive for this surface.

G.43. Surface: repair locality.
Rung: Schema or binding.
Audit question: does role bindings solve the task?
Required evidence: run or justify `schema matcher, binding fingerprint, entity resolver` under equal information.
Absorption token family: `ABSORBED_BY_SCHEMA_OR_BINDING`.
Claim ceiling if untested: inconclusive for this surface.

G.44. Surface: repair locality.
Rung: Domain tool.
Audit question: does the task is native to a known local tool?
Required evidence: run or justify `SQL, spreadsheets, SAT, SMT, ILP, ASP, CSP` under equal information.
Absorption token family: `ABSORBED_BY_DOMAIN_TOOL`.
Claim ceiling if untested: inconclusive for this surface.

G.45. Surface: repair locality.
Rung: Nuisance oracle.
Audit question: does difficulty vanishes when nuisance is removed?
Required evidence: run or justify `relevant-feature oracle or invariant oracle` under equal information.
Absorption token family: `ABSORBED_BY_NUISANCE_ORACLE`.
Claim ceiling if untested: inconclusive for this surface.

G.46. Surface: repair locality.
Rung: Constraint discovery.
Audit question: does the artifact is a constraint theory?
Required evidence: run or justify `rank solver, SAT/SMT encoding, table-constraint learner` under equal information.
Absorption token family: `ABSORBED_BY_CONSTRAINT_DISCOVERY`.
Claim ceiling if untested: inconclusive for this surface.

G.47. Surface: repair locality.
Rung: Generator family.
Audit question: does public statistics identify the synthetic generator?
Required evidence: run or justify `template classifier, compression classifier, graph-kernel probe` under equal information.
Absorption token family: `ABSORBED_BY_GENERATOR_FAMILY_IDENTIFICATION`.
Claim ceiling if untested: inconclusive for this surface.

G.48. Surface: repair locality.
Rung: Post-hoc compression.
Audit question: does the artifact summarizes behavior after success?
Required evidence: run or justify `frozen-before-score compression control` under equal information.
Absorption token family: `ABSORBED_BY_POST_HOC_COMPRESSION`.
Claim ceiling if untested: inconclusive for this surface.

G.49. Surface: abstention recall.
Rung: Representation prior.
Audit question: does public features expose the answer?
Required evidence: run or justify `feature parser, type tags, names, ontology` under equal information.
Absorption token family: `ABSORBED_BY_REPRESENTATION_PRIOR`.
Claim ceiling if untested: inconclusive for this surface.

G.50. Surface: abstention recall.
Rung: Parser or substrate prior.
Audit question: does the parser, verifier, DSL, action space, or canonicalizer does the work?
Required evidence: run or justify `shared-or-charged substrate baseline` under equal information.
Absorption token family: `ABSORBED_BY_PARSER_OR_SUBSTRATE_PRIOR`.
Claim ceiling if untested: inconclusive for this surface.

G.51. Surface: abstention recall.
Rung: Teaching dimension.
Audit question: does the packet is a small teaching set?
Required evidence: run or justify `finite teaching-set solver or version-space search` under equal information.
Absorption token family: `ABSORBED_BY_TEACHING_DIMENSION`.
Claim ceiling if untested: inconclusive for this surface.

G.52. Surface: abstention recall.
Rung: Active learning.
Audit question: does questions isolate the target cheaply?
Required evidence: run or justify `query planner, group tests, active counterexample search` under equal information.
Absorption token family: `ABSORBED_BY_ACTIVE_LEARNING`.
Claim ceiling if untested: inconclusive for this surface.

G.53. Surface: abstention recall.
Rung: PBE or CEGIS.
Audit question: does a program can be synthesized from the same evidence?
Required evidence: run or justify `PBE, CEGIS, SyGuS, ILP, relational rule search` under equal information.
Absorption token family: `ABSORBED_BY_CEGIS_OR_SYNTHESIS`.
Claim ceiling if untested: inconclusive for this surface.

G.54. Surface: abstention recall.
Rung: Library learning.
Audit question: does reusable macros explain transfer?
Required evidence: run or justify `MDL library, DreamCoder-style library, e-graph composition` under equal information.
Absorption token family: `ABSORBED_BY_LIBRARY_LEARNING`.
Claim ceiling if untested: inconclusive for this surface.

G.55. Surface: abstention recall.
Rung: Schema or binding.
Audit question: does role bindings solve the task?
Required evidence: run or justify `schema matcher, binding fingerprint, entity resolver` under equal information.
Absorption token family: `ABSORBED_BY_SCHEMA_OR_BINDING`.
Claim ceiling if untested: inconclusive for this surface.

G.56. Surface: abstention recall.
Rung: Domain tool.
Audit question: does the task is native to a known local tool?
Required evidence: run or justify `SQL, spreadsheets, SAT, SMT, ILP, ASP, CSP` under equal information.
Absorption token family: `ABSORBED_BY_DOMAIN_TOOL`.
Claim ceiling if untested: inconclusive for this surface.

G.57. Surface: abstention recall.
Rung: Nuisance oracle.
Audit question: does difficulty vanishes when nuisance is removed?
Required evidence: run or justify `relevant-feature oracle or invariant oracle` under equal information.
Absorption token family: `ABSORBED_BY_NUISANCE_ORACLE`.
Claim ceiling if untested: inconclusive for this surface.

G.58. Surface: abstention recall.
Rung: Constraint discovery.
Audit question: does the artifact is a constraint theory?
Required evidence: run or justify `rank solver, SAT/SMT encoding, table-constraint learner` under equal information.
Absorption token family: `ABSORBED_BY_CONSTRAINT_DISCOVERY`.
Claim ceiling if untested: inconclusive for this surface.

G.59. Surface: abstention recall.
Rung: Generator family.
Audit question: does public statistics identify the synthetic generator?
Required evidence: run or justify `template classifier, compression classifier, graph-kernel probe` under equal information.
Absorption token family: `ABSORBED_BY_GENERATOR_FAMILY_IDENTIFICATION`.
Claim ceiling if untested: inconclusive for this surface.

G.60. Surface: abstention recall.
Rung: Post-hoc compression.
Audit question: does the artifact summarizes behavior after success?
Required evidence: run or justify `frozen-before-score compression control` under equal information.
Absorption token family: `ABSORBED_BY_POST_HOC_COMPRESSION`.
Claim ceiling if untested: inconclusive for this surface.

G.61. Surface: abstention utility.
Rung: Representation prior.
Audit question: does public features expose the answer?
Required evidence: run or justify `feature parser, type tags, names, ontology` under equal information.
Absorption token family: `ABSORBED_BY_REPRESENTATION_PRIOR`.
Claim ceiling if untested: inconclusive for this surface.

G.62. Surface: abstention utility.
Rung: Parser or substrate prior.
Audit question: does the parser, verifier, DSL, action space, or canonicalizer does the work?
Required evidence: run or justify `shared-or-charged substrate baseline` under equal information.
Absorption token family: `ABSORBED_BY_PARSER_OR_SUBSTRATE_PRIOR`.
Claim ceiling if untested: inconclusive for this surface.

G.63. Surface: abstention utility.
Rung: Teaching dimension.
Audit question: does the packet is a small teaching set?
Required evidence: run or justify `finite teaching-set solver or version-space search` under equal information.
Absorption token family: `ABSORBED_BY_TEACHING_DIMENSION`.
Claim ceiling if untested: inconclusive for this surface.

G.64. Surface: abstention utility.
Rung: Active learning.
Audit question: does questions isolate the target cheaply?
Required evidence: run or justify `query planner, group tests, active counterexample search` under equal information.
Absorption token family: `ABSORBED_BY_ACTIVE_LEARNING`.
Claim ceiling if untested: inconclusive for this surface.

G.65. Surface: abstention utility.
Rung: PBE or CEGIS.
Audit question: does a program can be synthesized from the same evidence?
Required evidence: run or justify `PBE, CEGIS, SyGuS, ILP, relational rule search` under equal information.
Absorption token family: `ABSORBED_BY_CEGIS_OR_SYNTHESIS`.
Claim ceiling if untested: inconclusive for this surface.

G.66. Surface: abstention utility.
Rung: Library learning.
Audit question: does reusable macros explain transfer?
Required evidence: run or justify `MDL library, DreamCoder-style library, e-graph composition` under equal information.
Absorption token family: `ABSORBED_BY_LIBRARY_LEARNING`.
Claim ceiling if untested: inconclusive for this surface.

G.67. Surface: abstention utility.
Rung: Schema or binding.
Audit question: does role bindings solve the task?
Required evidence: run or justify `schema matcher, binding fingerprint, entity resolver` under equal information.
Absorption token family: `ABSORBED_BY_SCHEMA_OR_BINDING`.
Claim ceiling if untested: inconclusive for this surface.

G.68. Surface: abstention utility.
Rung: Domain tool.
Audit question: does the task is native to a known local tool?
Required evidence: run or justify `SQL, spreadsheets, SAT, SMT, ILP, ASP, CSP` under equal information.
Absorption token family: `ABSORBED_BY_DOMAIN_TOOL`.
Claim ceiling if untested: inconclusive for this surface.

G.69. Surface: abstention utility.
Rung: Nuisance oracle.
Audit question: does difficulty vanishes when nuisance is removed?
Required evidence: run or justify `relevant-feature oracle or invariant oracle` under equal information.
Absorption token family: `ABSORBED_BY_NUISANCE_ORACLE`.
Claim ceiling if untested: inconclusive for this surface.

G.70. Surface: abstention utility.
Rung: Constraint discovery.
Audit question: does the artifact is a constraint theory?
Required evidence: run or justify `rank solver, SAT/SMT encoding, table-constraint learner` under equal information.
Absorption token family: `ABSORBED_BY_CONSTRAINT_DISCOVERY`.
Claim ceiling if untested: inconclusive for this surface.

G.71. Surface: abstention utility.
Rung: Generator family.
Audit question: does public statistics identify the synthetic generator?
Required evidence: run or justify `template classifier, compression classifier, graph-kernel probe` under equal information.
Absorption token family: `ABSORBED_BY_GENERATOR_FAMILY_IDENTIFICATION`.
Claim ceiling if untested: inconclusive for this surface.

G.72. Surface: abstention utility.
Rung: Post-hoc compression.
Audit question: does the artifact summarizes behavior after success?
Required evidence: run or justify `frozen-before-score compression control` under equal information.
Absorption token family: `ABSORBED_BY_POST_HOC_COMPRESSION`.
Claim ceiling if untested: inconclusive for this surface.

G.73. Surface: composition accuracy.
Rung: Representation prior.
Audit question: does public features expose the answer?
Required evidence: run or justify `feature parser, type tags, names, ontology` under equal information.
Absorption token family: `ABSORBED_BY_REPRESENTATION_PRIOR`.
Claim ceiling if untested: inconclusive for this surface.

G.74. Surface: composition accuracy.
Rung: Parser or substrate prior.
Audit question: does the parser, verifier, DSL, action space, or canonicalizer does the work?
Required evidence: run or justify `shared-or-charged substrate baseline` under equal information.
Absorption token family: `ABSORBED_BY_PARSER_OR_SUBSTRATE_PRIOR`.
Claim ceiling if untested: inconclusive for this surface.

G.75. Surface: composition accuracy.
Rung: Teaching dimension.
Audit question: does the packet is a small teaching set?
Required evidence: run or justify `finite teaching-set solver or version-space search` under equal information.
Absorption token family: `ABSORBED_BY_TEACHING_DIMENSION`.
Claim ceiling if untested: inconclusive for this surface.

G.76. Surface: composition accuracy.
Rung: Active learning.
Audit question: does questions isolate the target cheaply?
Required evidence: run or justify `query planner, group tests, active counterexample search` under equal information.
Absorption token family: `ABSORBED_BY_ACTIVE_LEARNING`.
Claim ceiling if untested: inconclusive for this surface.

G.77. Surface: composition accuracy.
Rung: PBE or CEGIS.
Audit question: does a program can be synthesized from the same evidence?
Required evidence: run or justify `PBE, CEGIS, SyGuS, ILP, relational rule search` under equal information.
Absorption token family: `ABSORBED_BY_CEGIS_OR_SYNTHESIS`.
Claim ceiling if untested: inconclusive for this surface.

G.78. Surface: composition accuracy.
Rung: Library learning.
Audit question: does reusable macros explain transfer?
Required evidence: run or justify `MDL library, DreamCoder-style library, e-graph composition` under equal information.
Absorption token family: `ABSORBED_BY_LIBRARY_LEARNING`.
Claim ceiling if untested: inconclusive for this surface.

G.79. Surface: composition accuracy.
Rung: Schema or binding.
Audit question: does role bindings solve the task?
Required evidence: run or justify `schema matcher, binding fingerprint, entity resolver` under equal information.
Absorption token family: `ABSORBED_BY_SCHEMA_OR_BINDING`.
Claim ceiling if untested: inconclusive for this surface.

G.80. Surface: composition accuracy.
Rung: Domain tool.
Audit question: does the task is native to a known local tool?
Required evidence: run or justify `SQL, spreadsheets, SAT, SMT, ILP, ASP, CSP` under equal information.
Absorption token family: `ABSORBED_BY_DOMAIN_TOOL`.
Claim ceiling if untested: inconclusive for this surface.

G.81. Surface: composition accuracy.
Rung: Nuisance oracle.
Audit question: does difficulty vanishes when nuisance is removed?
Required evidence: run or justify `relevant-feature oracle or invariant oracle` under equal information.
Absorption token family: `ABSORBED_BY_NUISANCE_ORACLE`.
Claim ceiling if untested: inconclusive for this surface.

G.82. Surface: composition accuracy.
Rung: Constraint discovery.
Audit question: does the artifact is a constraint theory?
Required evidence: run or justify `rank solver, SAT/SMT encoding, table-constraint learner` under equal information.
Absorption token family: `ABSORBED_BY_CONSTRAINT_DISCOVERY`.
Claim ceiling if untested: inconclusive for this surface.

G.83. Surface: composition accuracy.
Rung: Generator family.
Audit question: does public statistics identify the synthetic generator?
Required evidence: run or justify `template classifier, compression classifier, graph-kernel probe` under equal information.
Absorption token family: `ABSORBED_BY_GENERATOR_FAMILY_IDENTIFICATION`.
Claim ceiling if untested: inconclusive for this surface.

G.84. Surface: composition accuracy.
Rung: Post-hoc compression.
Audit question: does the artifact summarizes behavior after success?
Required evidence: run or justify `frozen-before-score compression control` under equal information.
Absorption token family: `ABSORBED_BY_POST_HOC_COMPRESSION`.
Claim ceiling if untested: inconclusive for this surface.

G.85. Surface: invalidity detection.
Rung: Representation prior.
Audit question: does public features expose the answer?
Required evidence: run or justify `feature parser, type tags, names, ontology` under equal information.
Absorption token family: `ABSORBED_BY_REPRESENTATION_PRIOR`.
Claim ceiling if untested: inconclusive for this surface.

G.86. Surface: invalidity detection.
Rung: Parser or substrate prior.
Audit question: does the parser, verifier, DSL, action space, or canonicalizer does the work?
Required evidence: run or justify `shared-or-charged substrate baseline` under equal information.
Absorption token family: `ABSORBED_BY_PARSER_OR_SUBSTRATE_PRIOR`.
Claim ceiling if untested: inconclusive for this surface.

G.87. Surface: invalidity detection.
Rung: Teaching dimension.
Audit question: does the packet is a small teaching set?
Required evidence: run or justify `finite teaching-set solver or version-space search` under equal information.
Absorption token family: `ABSORBED_BY_TEACHING_DIMENSION`.
Claim ceiling if untested: inconclusive for this surface.

G.88. Surface: invalidity detection.
Rung: Active learning.
Audit question: does questions isolate the target cheaply?
Required evidence: run or justify `query planner, group tests, active counterexample search` under equal information.
Absorption token family: `ABSORBED_BY_ACTIVE_LEARNING`.
Claim ceiling if untested: inconclusive for this surface.

G.89. Surface: invalidity detection.
Rung: PBE or CEGIS.
Audit question: does a program can be synthesized from the same evidence?
Required evidence: run or justify `PBE, CEGIS, SyGuS, ILP, relational rule search` under equal information.
Absorption token family: `ABSORBED_BY_CEGIS_OR_SYNTHESIS`.
Claim ceiling if untested: inconclusive for this surface.

G.90. Surface: invalidity detection.
Rung: Library learning.
Audit question: does reusable macros explain transfer?
Required evidence: run or justify `MDL library, DreamCoder-style library, e-graph composition` under equal information.
Absorption token family: `ABSORBED_BY_LIBRARY_LEARNING`.
Claim ceiling if untested: inconclusive for this surface.

G.91. Surface: invalidity detection.
Rung: Schema or binding.
Audit question: does role bindings solve the task?
Required evidence: run or justify `schema matcher, binding fingerprint, entity resolver` under equal information.
Absorption token family: `ABSORBED_BY_SCHEMA_OR_BINDING`.
Claim ceiling if untested: inconclusive for this surface.

G.92. Surface: invalidity detection.
Rung: Domain tool.
Audit question: does the task is native to a known local tool?
Required evidence: run or justify `SQL, spreadsheets, SAT, SMT, ILP, ASP, CSP` under equal information.
Absorption token family: `ABSORBED_BY_DOMAIN_TOOL`.
Claim ceiling if untested: inconclusive for this surface.

G.93. Surface: invalidity detection.
Rung: Nuisance oracle.
Audit question: does difficulty vanishes when nuisance is removed?
Required evidence: run or justify `relevant-feature oracle or invariant oracle` under equal information.
Absorption token family: `ABSORBED_BY_NUISANCE_ORACLE`.
Claim ceiling if untested: inconclusive for this surface.

G.94. Surface: invalidity detection.
Rung: Constraint discovery.
Audit question: does the artifact is a constraint theory?
Required evidence: run or justify `rank solver, SAT/SMT encoding, table-constraint learner` under equal information.
Absorption token family: `ABSORBED_BY_CONSTRAINT_DISCOVERY`.
Claim ceiling if untested: inconclusive for this surface.

G.95. Surface: invalidity detection.
Rung: Generator family.
Audit question: does public statistics identify the synthetic generator?
Required evidence: run or justify `template classifier, compression classifier, graph-kernel probe` under equal information.
Absorption token family: `ABSORBED_BY_GENERATOR_FAMILY_IDENTIFICATION`.
Claim ceiling if untested: inconclusive for this surface.

G.96. Surface: invalidity detection.
Rung: Post-hoc compression.
Audit question: does the artifact summarizes behavior after success?
Required evidence: run or justify `frozen-before-score compression control` under equal information.
Absorption token family: `ABSORBED_BY_POST_HOC_COMPRESSION`.
Claim ceiling if untested: inconclusive for this surface.

G.97. Surface: unsafe-condition detection.
Rung: Representation prior.
Audit question: does public features expose the answer?
Required evidence: run or justify `feature parser, type tags, names, ontology` under equal information.
Absorption token family: `ABSORBED_BY_REPRESENTATION_PRIOR`.
Claim ceiling if untested: inconclusive for this surface.

G.98. Surface: unsafe-condition detection.
Rung: Parser or substrate prior.
Audit question: does the parser, verifier, DSL, action space, or canonicalizer does the work?
Required evidence: run or justify `shared-or-charged substrate baseline` under equal information.
Absorption token family: `ABSORBED_BY_PARSER_OR_SUBSTRATE_PRIOR`.
Claim ceiling if untested: inconclusive for this surface.

G.99. Surface: unsafe-condition detection.
Rung: Teaching dimension.
Audit question: does the packet is a small teaching set?
Required evidence: run or justify `finite teaching-set solver or version-space search` under equal information.
Absorption token family: `ABSORBED_BY_TEACHING_DIMENSION`.
Claim ceiling if untested: inconclusive for this surface.

G.100. Surface: unsafe-condition detection.
Rung: Active learning.
Audit question: does questions isolate the target cheaply?
Required evidence: run or justify `query planner, group tests, active counterexample search` under equal information.
Absorption token family: `ABSORBED_BY_ACTIVE_LEARNING`.
Claim ceiling if untested: inconclusive for this surface.

G.101. Surface: unsafe-condition detection.
Rung: PBE or CEGIS.
Audit question: does a program can be synthesized from the same evidence?
Required evidence: run or justify `PBE, CEGIS, SyGuS, ILP, relational rule search` under equal information.
Absorption token family: `ABSORBED_BY_CEGIS_OR_SYNTHESIS`.
Claim ceiling if untested: inconclusive for this surface.

G.102. Surface: unsafe-condition detection.
Rung: Library learning.
Audit question: does reusable macros explain transfer?
Required evidence: run or justify `MDL library, DreamCoder-style library, e-graph composition` under equal information.
Absorption token family: `ABSORBED_BY_LIBRARY_LEARNING`.
Claim ceiling if untested: inconclusive for this surface.

G.103. Surface: unsafe-condition detection.
Rung: Schema or binding.
Audit question: does role bindings solve the task?
Required evidence: run or justify `schema matcher, binding fingerprint, entity resolver` under equal information.
Absorption token family: `ABSORBED_BY_SCHEMA_OR_BINDING`.
Claim ceiling if untested: inconclusive for this surface.

G.104. Surface: unsafe-condition detection.
Rung: Domain tool.
Audit question: does the task is native to a known local tool?
Required evidence: run or justify `SQL, spreadsheets, SAT, SMT, ILP, ASP, CSP` under equal information.
Absorption token family: `ABSORBED_BY_DOMAIN_TOOL`.
Claim ceiling if untested: inconclusive for this surface.

G.105. Surface: unsafe-condition detection.
Rung: Nuisance oracle.
Audit question: does difficulty vanishes when nuisance is removed?
Required evidence: run or justify `relevant-feature oracle or invariant oracle` under equal information.
Absorption token family: `ABSORBED_BY_NUISANCE_ORACLE`.
Claim ceiling if untested: inconclusive for this surface.

G.106. Surface: unsafe-condition detection.
Rung: Constraint discovery.
Audit question: does the artifact is a constraint theory?
Required evidence: run or justify `rank solver, SAT/SMT encoding, table-constraint learner` under equal information.
Absorption token family: `ABSORBED_BY_CONSTRAINT_DISCOVERY`.
Claim ceiling if untested: inconclusive for this surface.

G.107. Surface: unsafe-condition detection.
Rung: Generator family.
Audit question: does public statistics identify the synthetic generator?
Required evidence: run or justify `template classifier, compression classifier, graph-kernel probe` under equal information.
Absorption token family: `ABSORBED_BY_GENERATOR_FAMILY_IDENTIFICATION`.
Claim ceiling if untested: inconclusive for this surface.

G.108. Surface: unsafe-condition detection.
Rung: Post-hoc compression.
Audit question: does the artifact summarizes behavior after success?
Required evidence: run or justify `frozen-before-score compression control` under equal information.
Absorption token family: `ABSORBED_BY_POST_HOC_COMPRESSION`.
Claim ceiling if untested: inconclusive for this surface.

G.109. Surface: schema stability.
Rung: Representation prior.
Audit question: does public features expose the answer?
Required evidence: run or justify `feature parser, type tags, names, ontology` under equal information.
Absorption token family: `ABSORBED_BY_REPRESENTATION_PRIOR`.
Claim ceiling if untested: inconclusive for this surface.

G.110. Surface: schema stability.
Rung: Parser or substrate prior.
Audit question: does the parser, verifier, DSL, action space, or canonicalizer does the work?
Required evidence: run or justify `shared-or-charged substrate baseline` under equal information.
Absorption token family: `ABSORBED_BY_PARSER_OR_SUBSTRATE_PRIOR`.
Claim ceiling if untested: inconclusive for this surface.

G.111. Surface: schema stability.
Rung: Teaching dimension.
Audit question: does the packet is a small teaching set?
Required evidence: run or justify `finite teaching-set solver or version-space search` under equal information.
Absorption token family: `ABSORBED_BY_TEACHING_DIMENSION`.
Claim ceiling if untested: inconclusive for this surface.

G.112. Surface: schema stability.
Rung: Active learning.
Audit question: does questions isolate the target cheaply?
Required evidence: run or justify `query planner, group tests, active counterexample search` under equal information.
Absorption token family: `ABSORBED_BY_ACTIVE_LEARNING`.
Claim ceiling if untested: inconclusive for this surface.

G.113. Surface: schema stability.
Rung: PBE or CEGIS.
Audit question: does a program can be synthesized from the same evidence?
Required evidence: run or justify `PBE, CEGIS, SyGuS, ILP, relational rule search` under equal information.
Absorption token family: `ABSORBED_BY_CEGIS_OR_SYNTHESIS`.
Claim ceiling if untested: inconclusive for this surface.

G.114. Surface: schema stability.
Rung: Library learning.
Audit question: does reusable macros explain transfer?
Required evidence: run or justify `MDL library, DreamCoder-style library, e-graph composition` under equal information.
Absorption token family: `ABSORBED_BY_LIBRARY_LEARNING`.
Claim ceiling if untested: inconclusive for this surface.

G.115. Surface: schema stability.
Rung: Schema or binding.
Audit question: does role bindings solve the task?
Required evidence: run or justify `schema matcher, binding fingerprint, entity resolver` under equal information.
Absorption token family: `ABSORBED_BY_SCHEMA_OR_BINDING`.
Claim ceiling if untested: inconclusive for this surface.

G.116. Surface: schema stability.
Rung: Domain tool.
Audit question: does the task is native to a known local tool?
Required evidence: run or justify `SQL, spreadsheets, SAT, SMT, ILP, ASP, CSP` under equal information.
Absorption token family: `ABSORBED_BY_DOMAIN_TOOL`.
Claim ceiling if untested: inconclusive for this surface.

G.117. Surface: schema stability.
Rung: Nuisance oracle.
Audit question: does difficulty vanishes when nuisance is removed?
Required evidence: run or justify `relevant-feature oracle or invariant oracle` under equal information.
Absorption token family: `ABSORBED_BY_NUISANCE_ORACLE`.
Claim ceiling if untested: inconclusive for this surface.

G.118. Surface: schema stability.
Rung: Constraint discovery.
Audit question: does the artifact is a constraint theory?
Required evidence: run or justify `rank solver, SAT/SMT encoding, table-constraint learner` under equal information.
Absorption token family: `ABSORBED_BY_CONSTRAINT_DISCOVERY`.
Claim ceiling if untested: inconclusive for this surface.

G.119. Surface: schema stability.
Rung: Generator family.
Audit question: does public statistics identify the synthetic generator?
Required evidence: run or justify `template classifier, compression classifier, graph-kernel probe` under equal information.
Absorption token family: `ABSORBED_BY_GENERATOR_FAMILY_IDENTIFICATION`.
Claim ceiling if untested: inconclusive for this surface.

G.120. Surface: schema stability.
Rung: Post-hoc compression.
Audit question: does the artifact summarizes behavior after success?
Required evidence: run or justify `frozen-before-score compression control` under equal information.
Absorption token family: `ABSORBED_BY_POST_HOC_COMPRESSION`.
Claim ceiling if untested: inconclusive for this surface.

G.121. Surface: unit stability.
Rung: Representation prior.
Audit question: does public features expose the answer?
Required evidence: run or justify `feature parser, type tags, names, ontology` under equal information.
Absorption token family: `ABSORBED_BY_REPRESENTATION_PRIOR`.
Claim ceiling if untested: inconclusive for this surface.

G.122. Surface: unit stability.
Rung: Parser or substrate prior.
Audit question: does the parser, verifier, DSL, action space, or canonicalizer does the work?
Required evidence: run or justify `shared-or-charged substrate baseline` under equal information.
Absorption token family: `ABSORBED_BY_PARSER_OR_SUBSTRATE_PRIOR`.
Claim ceiling if untested: inconclusive for this surface.

G.123. Surface: unit stability.
Rung: Teaching dimension.
Audit question: does the packet is a small teaching set?
Required evidence: run or justify `finite teaching-set solver or version-space search` under equal information.
Absorption token family: `ABSORBED_BY_TEACHING_DIMENSION`.
Claim ceiling if untested: inconclusive for this surface.

G.124. Surface: unit stability.
Rung: Active learning.
Audit question: does questions isolate the target cheaply?
Required evidence: run or justify `query planner, group tests, active counterexample search` under equal information.
Absorption token family: `ABSORBED_BY_ACTIVE_LEARNING`.
Claim ceiling if untested: inconclusive for this surface.

G.125. Surface: unit stability.
Rung: PBE or CEGIS.
Audit question: does a program can be synthesized from the same evidence?
Required evidence: run or justify `PBE, CEGIS, SyGuS, ILP, relational rule search` under equal information.
Absorption token family: `ABSORBED_BY_CEGIS_OR_SYNTHESIS`.
Claim ceiling if untested: inconclusive for this surface.

G.126. Surface: unit stability.
Rung: Library learning.
Audit question: does reusable macros explain transfer?
Required evidence: run or justify `MDL library, DreamCoder-style library, e-graph composition` under equal information.
Absorption token family: `ABSORBED_BY_LIBRARY_LEARNING`.
Claim ceiling if untested: inconclusive for this surface.

G.127. Surface: unit stability.
Rung: Schema or binding.
Audit question: does role bindings solve the task?
Required evidence: run or justify `schema matcher, binding fingerprint, entity resolver` under equal information.
Absorption token family: `ABSORBED_BY_SCHEMA_OR_BINDING`.
Claim ceiling if untested: inconclusive for this surface.

G.128. Surface: unit stability.
Rung: Domain tool.
Audit question: does the task is native to a known local tool?
Required evidence: run or justify `SQL, spreadsheets, SAT, SMT, ILP, ASP, CSP` under equal information.
Absorption token family: `ABSORBED_BY_DOMAIN_TOOL`.
Claim ceiling if untested: inconclusive for this surface.

G.129. Surface: unit stability.
Rung: Nuisance oracle.
Audit question: does difficulty vanishes when nuisance is removed?
Required evidence: run or justify `relevant-feature oracle or invariant oracle` under equal information.
Absorption token family: `ABSORBED_BY_NUISANCE_ORACLE`.
Claim ceiling if untested: inconclusive for this surface.

G.130. Surface: unit stability.
Rung: Constraint discovery.
Audit question: does the artifact is a constraint theory?
Required evidence: run or justify `rank solver, SAT/SMT encoding, table-constraint learner` under equal information.
Absorption token family: `ABSORBED_BY_CONSTRAINT_DISCOVERY`.
Claim ceiling if untested: inconclusive for this surface.

G.131. Surface: unit stability.
Rung: Generator family.
Audit question: does public statistics identify the synthetic generator?
Required evidence: run or justify `template classifier, compression classifier, graph-kernel probe` under equal information.
Absorption token family: `ABSORBED_BY_GENERATOR_FAMILY_IDENTIFICATION`.
Claim ceiling if untested: inconclusive for this surface.

G.132. Surface: unit stability.
Rung: Post-hoc compression.
Audit question: does the artifact summarizes behavior after success?
Required evidence: run or justify `frozen-before-score compression control` under equal information.
Absorption token family: `ABSORBED_BY_POST_HOC_COMPRESSION`.
Claim ceiling if untested: inconclusive for this surface.

G.133. Surface: row-order stability.
Rung: Representation prior.
Audit question: does public features expose the answer?
Required evidence: run or justify `feature parser, type tags, names, ontology` under equal information.
Absorption token family: `ABSORBED_BY_REPRESENTATION_PRIOR`.
Claim ceiling if untested: inconclusive for this surface.

G.134. Surface: row-order stability.
Rung: Parser or substrate prior.
Audit question: does the parser, verifier, DSL, action space, or canonicalizer does the work?
Required evidence: run or justify `shared-or-charged substrate baseline` under equal information.
Absorption token family: `ABSORBED_BY_PARSER_OR_SUBSTRATE_PRIOR`.
Claim ceiling if untested: inconclusive for this surface.

G.135. Surface: row-order stability.
Rung: Teaching dimension.
Audit question: does the packet is a small teaching set?
Required evidence: run or justify `finite teaching-set solver or version-space search` under equal information.
Absorption token family: `ABSORBED_BY_TEACHING_DIMENSION`.
Claim ceiling if untested: inconclusive for this surface.

G.136. Surface: row-order stability.
Rung: Active learning.
Audit question: does questions isolate the target cheaply?
Required evidence: run or justify `query planner, group tests, active counterexample search` under equal information.
Absorption token family: `ABSORBED_BY_ACTIVE_LEARNING`.
Claim ceiling if untested: inconclusive for this surface.

G.137. Surface: row-order stability.
Rung: PBE or CEGIS.
Audit question: does a program can be synthesized from the same evidence?
Required evidence: run or justify `PBE, CEGIS, SyGuS, ILP, relational rule search` under equal information.
Absorption token family: `ABSORBED_BY_CEGIS_OR_SYNTHESIS`.
Claim ceiling if untested: inconclusive for this surface.

G.138. Surface: row-order stability.
Rung: Library learning.
Audit question: does reusable macros explain transfer?
Required evidence: run or justify `MDL library, DreamCoder-style library, e-graph composition` under equal information.
Absorption token family: `ABSORBED_BY_LIBRARY_LEARNING`.
Claim ceiling if untested: inconclusive for this surface.

G.139. Surface: row-order stability.
Rung: Schema or binding.
Audit question: does role bindings solve the task?
Required evidence: run or justify `schema matcher, binding fingerprint, entity resolver` under equal information.
Absorption token family: `ABSORBED_BY_SCHEMA_OR_BINDING`.
Claim ceiling if untested: inconclusive for this surface.

G.140. Surface: row-order stability.
Rung: Domain tool.
Audit question: does the task is native to a known local tool?
Required evidence: run or justify `SQL, spreadsheets, SAT, SMT, ILP, ASP, CSP` under equal information.
Absorption token family: `ABSORBED_BY_DOMAIN_TOOL`.
Claim ceiling if untested: inconclusive for this surface.

G.141. Surface: row-order stability.
Rung: Nuisance oracle.
Audit question: does difficulty vanishes when nuisance is removed?
Required evidence: run or justify `relevant-feature oracle or invariant oracle` under equal information.
Absorption token family: `ABSORBED_BY_NUISANCE_ORACLE`.
Claim ceiling if untested: inconclusive for this surface.

G.142. Surface: row-order stability.
Rung: Constraint discovery.
Audit question: does the artifact is a constraint theory?
Required evidence: run or justify `rank solver, SAT/SMT encoding, table-constraint learner` under equal information.
Absorption token family: `ABSORBED_BY_CONSTRAINT_DISCOVERY`.
Claim ceiling if untested: inconclusive for this surface.

G.143. Surface: row-order stability.
Rung: Generator family.
Audit question: does public statistics identify the synthetic generator?
Required evidence: run or justify `template classifier, compression classifier, graph-kernel probe` under equal information.
Absorption token family: `ABSORBED_BY_GENERATOR_FAMILY_IDENTIFICATION`.
Claim ceiling if untested: inconclusive for this surface.

G.144. Surface: row-order stability.
Rung: Post-hoc compression.
Audit question: does the artifact summarizes behavior after success?
Required evidence: run or justify `frozen-before-score compression control` under equal information.
Absorption token family: `ABSORBED_BY_POST_HOC_COMPRESSION`.
Claim ceiling if untested: inconclusive for this surface.

G.145. Surface: role permutation stability.
Rung: Representation prior.
Audit question: does public features expose the answer?
Required evidence: run or justify `feature parser, type tags, names, ontology` under equal information.
Absorption token family: `ABSORBED_BY_REPRESENTATION_PRIOR`.
Claim ceiling if untested: inconclusive for this surface.

G.146. Surface: role permutation stability.
Rung: Parser or substrate prior.
Audit question: does the parser, verifier, DSL, action space, or canonicalizer does the work?
Required evidence: run or justify `shared-or-charged substrate baseline` under equal information.
Absorption token family: `ABSORBED_BY_PARSER_OR_SUBSTRATE_PRIOR`.
Claim ceiling if untested: inconclusive for this surface.

G.147. Surface: role permutation stability.
Rung: Teaching dimension.
Audit question: does the packet is a small teaching set?
Required evidence: run or justify `finite teaching-set solver or version-space search` under equal information.
Absorption token family: `ABSORBED_BY_TEACHING_DIMENSION`.
Claim ceiling if untested: inconclusive for this surface.

G.148. Surface: role permutation stability.
Rung: Active learning.
Audit question: does questions isolate the target cheaply?
Required evidence: run or justify `query planner, group tests, active counterexample search` under equal information.
Absorption token family: `ABSORBED_BY_ACTIVE_LEARNING`.
Claim ceiling if untested: inconclusive for this surface.

G.149. Surface: role permutation stability.
Rung: PBE or CEGIS.
Audit question: does a program can be synthesized from the same evidence?
Required evidence: run or justify `PBE, CEGIS, SyGuS, ILP, relational rule search` under equal information.
Absorption token family: `ABSORBED_BY_CEGIS_OR_SYNTHESIS`.
Claim ceiling if untested: inconclusive for this surface.

G.150. Surface: role permutation stability.
Rung: Library learning.
Audit question: does reusable macros explain transfer?
Required evidence: run or justify `MDL library, DreamCoder-style library, e-graph composition` under equal information.
Absorption token family: `ABSORBED_BY_LIBRARY_LEARNING`.
Claim ceiling if untested: inconclusive for this surface.

G.151. Surface: role permutation stability.
Rung: Schema or binding.
Audit question: does role bindings solve the task?
Required evidence: run or justify `schema matcher, binding fingerprint, entity resolver` under equal information.
Absorption token family: `ABSORBED_BY_SCHEMA_OR_BINDING`.
Claim ceiling if untested: inconclusive for this surface.

G.152. Surface: role permutation stability.
Rung: Domain tool.
Audit question: does the task is native to a known local tool?
Required evidence: run or justify `SQL, spreadsheets, SAT, SMT, ILP, ASP, CSP` under equal information.
Absorption token family: `ABSORBED_BY_DOMAIN_TOOL`.
Claim ceiling if untested: inconclusive for this surface.

G.153. Surface: role permutation stability.
Rung: Nuisance oracle.
Audit question: does difficulty vanishes when nuisance is removed?
Required evidence: run or justify `relevant-feature oracle or invariant oracle` under equal information.
Absorption token family: `ABSORBED_BY_NUISANCE_ORACLE`.
Claim ceiling if untested: inconclusive for this surface.

G.154. Surface: role permutation stability.
Rung: Constraint discovery.
Audit question: does the artifact is a constraint theory?
Required evidence: run or justify `rank solver, SAT/SMT encoding, table-constraint learner` under equal information.
Absorption token family: `ABSORBED_BY_CONSTRAINT_DISCOVERY`.
Claim ceiling if untested: inconclusive for this surface.

G.155. Surface: role permutation stability.
Rung: Generator family.
Audit question: does public statistics identify the synthetic generator?
Required evidence: run or justify `template classifier, compression classifier, graph-kernel probe` under equal information.
Absorption token family: `ABSORBED_BY_GENERATOR_FAMILY_IDENTIFICATION`.
Claim ceiling if untested: inconclusive for this surface.

G.156. Surface: role permutation stability.
Rung: Post-hoc compression.
Audit question: does the artifact summarizes behavior after success?
Required evidence: run or justify `frozen-before-score compression control` under equal information.
Absorption token family: `ABSORBED_BY_POST_HOC_COMPRESSION`.
Claim ceiling if untested: inconclusive for this surface.

G.157. Surface: transfer residual cost.
Rung: Representation prior.
Audit question: does public features expose the answer?
Required evidence: run or justify `feature parser, type tags, names, ontology` under equal information.
Absorption token family: `ABSORBED_BY_REPRESENTATION_PRIOR`.
Claim ceiling if untested: inconclusive for this surface.

G.158. Surface: transfer residual cost.
Rung: Parser or substrate prior.
Audit question: does the parser, verifier, DSL, action space, or canonicalizer does the work?
Required evidence: run or justify `shared-or-charged substrate baseline` under equal information.
Absorption token family: `ABSORBED_BY_PARSER_OR_SUBSTRATE_PRIOR`.
Claim ceiling if untested: inconclusive for this surface.

G.159. Surface: transfer residual cost.
Rung: Teaching dimension.
Audit question: does the packet is a small teaching set?
Required evidence: run or justify `finite teaching-set solver or version-space search` under equal information.
Absorption token family: `ABSORBED_BY_TEACHING_DIMENSION`.
Claim ceiling if untested: inconclusive for this surface.

G.160. Surface: transfer residual cost.
Rung: Active learning.
Audit question: does questions isolate the target cheaply?
Required evidence: run or justify `query planner, group tests, active counterexample search` under equal information.
Absorption token family: `ABSORBED_BY_ACTIVE_LEARNING`.
Claim ceiling if untested: inconclusive for this surface.

G.161. Surface: transfer residual cost.
Rung: PBE or CEGIS.
Audit question: does a program can be synthesized from the same evidence?
Required evidence: run or justify `PBE, CEGIS, SyGuS, ILP, relational rule search` under equal information.
Absorption token family: `ABSORBED_BY_CEGIS_OR_SYNTHESIS`.
Claim ceiling if untested: inconclusive for this surface.

G.162. Surface: transfer residual cost.
Rung: Library learning.
Audit question: does reusable macros explain transfer?
Required evidence: run or justify `MDL library, DreamCoder-style library, e-graph composition` under equal information.
Absorption token family: `ABSORBED_BY_LIBRARY_LEARNING`.
Claim ceiling if untested: inconclusive for this surface.

G.163. Surface: transfer residual cost.
Rung: Schema or binding.
Audit question: does role bindings solve the task?
Required evidence: run or justify `schema matcher, binding fingerprint, entity resolver` under equal information.
Absorption token family: `ABSORBED_BY_SCHEMA_OR_BINDING`.
Claim ceiling if untested: inconclusive for this surface.

G.164. Surface: transfer residual cost.
Rung: Domain tool.
Audit question: does the task is native to a known local tool?
Required evidence: run or justify `SQL, spreadsheets, SAT, SMT, ILP, ASP, CSP` under equal information.
Absorption token family: `ABSORBED_BY_DOMAIN_TOOL`.
Claim ceiling if untested: inconclusive for this surface.

G.165. Surface: transfer residual cost.
Rung: Nuisance oracle.
Audit question: does difficulty vanishes when nuisance is removed?
Required evidence: run or justify `relevant-feature oracle or invariant oracle` under equal information.
Absorption token family: `ABSORBED_BY_NUISANCE_ORACLE`.
Claim ceiling if untested: inconclusive for this surface.

G.166. Surface: transfer residual cost.
Rung: Constraint discovery.
Audit question: does the artifact is a constraint theory?
Required evidence: run or justify `rank solver, SAT/SMT encoding, table-constraint learner` under equal information.
Absorption token family: `ABSORBED_BY_CONSTRAINT_DISCOVERY`.
Claim ceiling if untested: inconclusive for this surface.

G.167. Surface: transfer residual cost.
Rung: Generator family.
Audit question: does public statistics identify the synthetic generator?
Required evidence: run or justify `template classifier, compression classifier, graph-kernel probe` under equal information.
Absorption token family: `ABSORBED_BY_GENERATOR_FAMILY_IDENTIFICATION`.
Claim ceiling if untested: inconclusive for this surface.

G.168. Surface: transfer residual cost.
Rung: Post-hoc compression.
Audit question: does the artifact summarizes behavior after success?
Required evidence: run or justify `frozen-before-score compression control` under equal information.
Absorption token family: `ABSORBED_BY_POST_HOC_COMPRESSION`.
Claim ceiling if untested: inconclusive for this surface.

G.169. Surface: public-feature leakage.
Rung: Representation prior.
Audit question: does public features expose the answer?
Required evidence: run or justify `feature parser, type tags, names, ontology` under equal information.
Absorption token family: `ABSORBED_BY_REPRESENTATION_PRIOR`.
Claim ceiling if untested: inconclusive for this surface.

G.170. Surface: public-feature leakage.
Rung: Parser or substrate prior.
Audit question: does the parser, verifier, DSL, action space, or canonicalizer does the work?
Required evidence: run or justify `shared-or-charged substrate baseline` under equal information.
Absorption token family: `ABSORBED_BY_PARSER_OR_SUBSTRATE_PRIOR`.
Claim ceiling if untested: inconclusive for this surface.

G.171. Surface: public-feature leakage.
Rung: Teaching dimension.
Audit question: does the packet is a small teaching set?
Required evidence: run or justify `finite teaching-set solver or version-space search` under equal information.
Absorption token family: `ABSORBED_BY_TEACHING_DIMENSION`.
Claim ceiling if untested: inconclusive for this surface.

G.172. Surface: public-feature leakage.
Rung: Active learning.
Audit question: does questions isolate the target cheaply?
Required evidence: run or justify `query planner, group tests, active counterexample search` under equal information.
Absorption token family: `ABSORBED_BY_ACTIVE_LEARNING`.
Claim ceiling if untested: inconclusive for this surface.

G.173. Surface: public-feature leakage.
Rung: PBE or CEGIS.
Audit question: does a program can be synthesized from the same evidence?
Required evidence: run or justify `PBE, CEGIS, SyGuS, ILP, relational rule search` under equal information.
Absorption token family: `ABSORBED_BY_CEGIS_OR_SYNTHESIS`.
Claim ceiling if untested: inconclusive for this surface.

G.174. Surface: public-feature leakage.
Rung: Library learning.
Audit question: does reusable macros explain transfer?
Required evidence: run or justify `MDL library, DreamCoder-style library, e-graph composition` under equal information.
Absorption token family: `ABSORBED_BY_LIBRARY_LEARNING`.
Claim ceiling if untested: inconclusive for this surface.

G.175. Surface: public-feature leakage.
Rung: Schema or binding.
Audit question: does role bindings solve the task?
Required evidence: run or justify `schema matcher, binding fingerprint, entity resolver` under equal information.
Absorption token family: `ABSORBED_BY_SCHEMA_OR_BINDING`.
Claim ceiling if untested: inconclusive for this surface.

G.176. Surface: public-feature leakage.
Rung: Domain tool.
Audit question: does the task is native to a known local tool?
Required evidence: run or justify `SQL, spreadsheets, SAT, SMT, ILP, ASP, CSP` under equal information.
Absorption token family: `ABSORBED_BY_DOMAIN_TOOL`.
Claim ceiling if untested: inconclusive for this surface.

G.177. Surface: public-feature leakage.
Rung: Nuisance oracle.
Audit question: does difficulty vanishes when nuisance is removed?
Required evidence: run or justify `relevant-feature oracle or invariant oracle` under equal information.
Absorption token family: `ABSORBED_BY_NUISANCE_ORACLE`.
Claim ceiling if untested: inconclusive for this surface.

G.178. Surface: public-feature leakage.
Rung: Constraint discovery.
Audit question: does the artifact is a constraint theory?
Required evidence: run or justify `rank solver, SAT/SMT encoding, table-constraint learner` under equal information.
Absorption token family: `ABSORBED_BY_CONSTRAINT_DISCOVERY`.
Claim ceiling if untested: inconclusive for this surface.

G.179. Surface: public-feature leakage.
Rung: Generator family.
Audit question: does public statistics identify the synthetic generator?
Required evidence: run or justify `template classifier, compression classifier, graph-kernel probe` under equal information.
Absorption token family: `ABSORBED_BY_GENERATOR_FAMILY_IDENTIFICATION`.
Claim ceiling if untested: inconclusive for this surface.

G.180. Surface: public-feature leakage.
Rung: Post-hoc compression.
Audit question: does the artifact summarizes behavior after success?
Required evidence: run or justify `frozen-before-score compression control` under equal information.
Absorption token family: `ABSORBED_BY_POST_HOC_COMPRESSION`.
Claim ceiling if untested: inconclusive for this surface.

G.181. Surface: component-erasure drop.
Rung: Representation prior.
Audit question: does public features expose the answer?
Required evidence: run or justify `feature parser, type tags, names, ontology` under equal information.
Absorption token family: `ABSORBED_BY_REPRESENTATION_PRIOR`.
Claim ceiling if untested: inconclusive for this surface.

G.182. Surface: component-erasure drop.
Rung: Parser or substrate prior.
Audit question: does the parser, verifier, DSL, action space, or canonicalizer does the work?
Required evidence: run or justify `shared-or-charged substrate baseline` under equal information.
Absorption token family: `ABSORBED_BY_PARSER_OR_SUBSTRATE_PRIOR`.
Claim ceiling if untested: inconclusive for this surface.

G.183. Surface: component-erasure drop.
Rung: Teaching dimension.
Audit question: does the packet is a small teaching set?
Required evidence: run or justify `finite teaching-set solver or version-space search` under equal information.
Absorption token family: `ABSORBED_BY_TEACHING_DIMENSION`.
Claim ceiling if untested: inconclusive for this surface.

G.184. Surface: component-erasure drop.
Rung: Active learning.
Audit question: does questions isolate the target cheaply?
Required evidence: run or justify `query planner, group tests, active counterexample search` under equal information.
Absorption token family: `ABSORBED_BY_ACTIVE_LEARNING`.
Claim ceiling if untested: inconclusive for this surface.

G.185. Surface: component-erasure drop.
Rung: PBE or CEGIS.
Audit question: does a program can be synthesized from the same evidence?
Required evidence: run or justify `PBE, CEGIS, SyGuS, ILP, relational rule search` under equal information.
Absorption token family: `ABSORBED_BY_CEGIS_OR_SYNTHESIS`.
Claim ceiling if untested: inconclusive for this surface.

G.186. Surface: component-erasure drop.
Rung: Library learning.
Audit question: does reusable macros explain transfer?
Required evidence: run or justify `MDL library, DreamCoder-style library, e-graph composition` under equal information.
Absorption token family: `ABSORBED_BY_LIBRARY_LEARNING`.
Claim ceiling if untested: inconclusive for this surface.

G.187. Surface: component-erasure drop.
Rung: Schema or binding.
Audit question: does role bindings solve the task?
Required evidence: run or justify `schema matcher, binding fingerprint, entity resolver` under equal information.
Absorption token family: `ABSORBED_BY_SCHEMA_OR_BINDING`.
Claim ceiling if untested: inconclusive for this surface.

G.188. Surface: component-erasure drop.
Rung: Domain tool.
Audit question: does the task is native to a known local tool?
Required evidence: run or justify `SQL, spreadsheets, SAT, SMT, ILP, ASP, CSP` under equal information.
Absorption token family: `ABSORBED_BY_DOMAIN_TOOL`.
Claim ceiling if untested: inconclusive for this surface.

G.189. Surface: component-erasure drop.
Rung: Nuisance oracle.
Audit question: does difficulty vanishes when nuisance is removed?
Required evidence: run or justify `relevant-feature oracle or invariant oracle` under equal information.
Absorption token family: `ABSORBED_BY_NUISANCE_ORACLE`.
Claim ceiling if untested: inconclusive for this surface.

G.190. Surface: component-erasure drop.
Rung: Constraint discovery.
Audit question: does the artifact is a constraint theory?
Required evidence: run or justify `rank solver, SAT/SMT encoding, table-constraint learner` under equal information.
Absorption token family: `ABSORBED_BY_CONSTRAINT_DISCOVERY`.
Claim ceiling if untested: inconclusive for this surface.

G.191. Surface: component-erasure drop.
Rung: Generator family.
Audit question: does public statistics identify the synthetic generator?
Required evidence: run or justify `template classifier, compression classifier, graph-kernel probe` under equal information.
Absorption token family: `ABSORBED_BY_GENERATOR_FAMILY_IDENTIFICATION`.
Claim ceiling if untested: inconclusive for this surface.

G.192. Surface: component-erasure drop.
Rung: Post-hoc compression.
Audit question: does the artifact summarizes behavior after success?
Required evidence: run or justify `frozen-before-score compression control` under equal information.
Absorption token family: `ABSORBED_BY_POST_HOC_COMPRESSION`.
Claim ceiling if untested: inconclusive for this surface.

G.193. Surface: randomized-label control.
Rung: Representation prior.
Audit question: does public features expose the answer?
Required evidence: run or justify `feature parser, type tags, names, ontology` under equal information.
Absorption token family: `ABSORBED_BY_REPRESENTATION_PRIOR`.
Claim ceiling if untested: inconclusive for this surface.

G.194. Surface: randomized-label control.
Rung: Parser or substrate prior.
Audit question: does the parser, verifier, DSL, action space, or canonicalizer does the work?
Required evidence: run or justify `shared-or-charged substrate baseline` under equal information.
Absorption token family: `ABSORBED_BY_PARSER_OR_SUBSTRATE_PRIOR`.
Claim ceiling if untested: inconclusive for this surface.

G.195. Surface: randomized-label control.
Rung: Teaching dimension.
Audit question: does the packet is a small teaching set?
Required evidence: run or justify `finite teaching-set solver or version-space search` under equal information.
Absorption token family: `ABSORBED_BY_TEACHING_DIMENSION`.
Claim ceiling if untested: inconclusive for this surface.

G.196. Surface: randomized-label control.
Rung: Active learning.
Audit question: does questions isolate the target cheaply?
Required evidence: run or justify `query planner, group tests, active counterexample search` under equal information.
Absorption token family: `ABSORBED_BY_ACTIVE_LEARNING`.
Claim ceiling if untested: inconclusive for this surface.

G.197. Surface: randomized-label control.
Rung: PBE or CEGIS.
Audit question: does a program can be synthesized from the same evidence?
Required evidence: run or justify `PBE, CEGIS, SyGuS, ILP, relational rule search` under equal information.
Absorption token family: `ABSORBED_BY_CEGIS_OR_SYNTHESIS`.
Claim ceiling if untested: inconclusive for this surface.

G.198. Surface: randomized-label control.
Rung: Library learning.
Audit question: does reusable macros explain transfer?
Required evidence: run or justify `MDL library, DreamCoder-style library, e-graph composition` under equal information.
Absorption token family: `ABSORBED_BY_LIBRARY_LEARNING`.
Claim ceiling if untested: inconclusive for this surface.

G.199. Surface: randomized-label control.
Rung: Schema or binding.
Audit question: does role bindings solve the task?
Required evidence: run or justify `schema matcher, binding fingerprint, entity resolver` under equal information.
Absorption token family: `ABSORBED_BY_SCHEMA_OR_BINDING`.
Claim ceiling if untested: inconclusive for this surface.

G.200. Surface: randomized-label control.
Rung: Domain tool.
Audit question: does the task is native to a known local tool?
Required evidence: run or justify `SQL, spreadsheets, SAT, SMT, ILP, ASP, CSP` under equal information.
Absorption token family: `ABSORBED_BY_DOMAIN_TOOL`.
Claim ceiling if untested: inconclusive for this surface.

G.201. Surface: randomized-label control.
Rung: Nuisance oracle.
Audit question: does difficulty vanishes when nuisance is removed?
Required evidence: run or justify `relevant-feature oracle or invariant oracle` under equal information.
Absorption token family: `ABSORBED_BY_NUISANCE_ORACLE`.
Claim ceiling if untested: inconclusive for this surface.

G.202. Surface: randomized-label control.
Rung: Constraint discovery.
Audit question: does the artifact is a constraint theory?
Required evidence: run or justify `rank solver, SAT/SMT encoding, table-constraint learner` under equal information.
Absorption token family: `ABSORBED_BY_CONSTRAINT_DISCOVERY`.
Claim ceiling if untested: inconclusive for this surface.

G.203. Surface: randomized-label control.
Rung: Generator family.
Audit question: does public statistics identify the synthetic generator?
Required evidence: run or justify `template classifier, compression classifier, graph-kernel probe` under equal information.
Absorption token family: `ABSORBED_BY_GENERATOR_FAMILY_IDENTIFICATION`.
Claim ceiling if untested: inconclusive for this surface.

G.204. Surface: randomized-label control.
Rung: Post-hoc compression.
Audit question: does the artifact summarizes behavior after success?
Required evidence: run or justify `frozen-before-score compression control` under equal information.
Absorption token family: `ABSORBED_BY_POST_HOC_COMPRESSION`.
Claim ceiling if untested: inconclusive for this surface.

G.205. Surface: query efficiency.
Rung: Representation prior.
Audit question: does public features expose the answer?
Required evidence: run or justify `feature parser, type tags, names, ontology` under equal information.
Absorption token family: `ABSORBED_BY_REPRESENTATION_PRIOR`.
Claim ceiling if untested: inconclusive for this surface.

G.206. Surface: query efficiency.
Rung: Parser or substrate prior.
Audit question: does the parser, verifier, DSL, action space, or canonicalizer does the work?
Required evidence: run or justify `shared-or-charged substrate baseline` under equal information.
Absorption token family: `ABSORBED_BY_PARSER_OR_SUBSTRATE_PRIOR`.
Claim ceiling if untested: inconclusive for this surface.

G.207. Surface: query efficiency.
Rung: Teaching dimension.
Audit question: does the packet is a small teaching set?
Required evidence: run or justify `finite teaching-set solver or version-space search` under equal information.
Absorption token family: `ABSORBED_BY_TEACHING_DIMENSION`.
Claim ceiling if untested: inconclusive for this surface.

G.208. Surface: query efficiency.
Rung: Active learning.
Audit question: does questions isolate the target cheaply?
Required evidence: run or justify `query planner, group tests, active counterexample search` under equal information.
Absorption token family: `ABSORBED_BY_ACTIVE_LEARNING`.
Claim ceiling if untested: inconclusive for this surface.

G.209. Surface: query efficiency.
Rung: PBE or CEGIS.
Audit question: does a program can be synthesized from the same evidence?
Required evidence: run or justify `PBE, CEGIS, SyGuS, ILP, relational rule search` under equal information.
Absorption token family: `ABSORBED_BY_CEGIS_OR_SYNTHESIS`.
Claim ceiling if untested: inconclusive for this surface.

G.210. Surface: query efficiency.
Rung: Library learning.
Audit question: does reusable macros explain transfer?
Required evidence: run or justify `MDL library, DreamCoder-style library, e-graph composition` under equal information.
Absorption token family: `ABSORBED_BY_LIBRARY_LEARNING`.
Claim ceiling if untested: inconclusive for this surface.

G.211. Surface: query efficiency.
Rung: Schema or binding.
Audit question: does role bindings solve the task?
Required evidence: run or justify `schema matcher, binding fingerprint, entity resolver` under equal information.
Absorption token family: `ABSORBED_BY_SCHEMA_OR_BINDING`.
Claim ceiling if untested: inconclusive for this surface.

G.212. Surface: query efficiency.
Rung: Domain tool.
Audit question: does the task is native to a known local tool?
Required evidence: run or justify `SQL, spreadsheets, SAT, SMT, ILP, ASP, CSP` under equal information.
Absorption token family: `ABSORBED_BY_DOMAIN_TOOL`.
Claim ceiling if untested: inconclusive for this surface.

G.213. Surface: query efficiency.
Rung: Nuisance oracle.
Audit question: does difficulty vanishes when nuisance is removed?
Required evidence: run or justify `relevant-feature oracle or invariant oracle` under equal information.
Absorption token family: `ABSORBED_BY_NUISANCE_ORACLE`.
Claim ceiling if untested: inconclusive for this surface.

G.214. Surface: query efficiency.
Rung: Constraint discovery.
Audit question: does the artifact is a constraint theory?
Required evidence: run or justify `rank solver, SAT/SMT encoding, table-constraint learner` under equal information.
Absorption token family: `ABSORBED_BY_CONSTRAINT_DISCOVERY`.
Claim ceiling if untested: inconclusive for this surface.

G.215. Surface: query efficiency.
Rung: Generator family.
Audit question: does public statistics identify the synthetic generator?
Required evidence: run or justify `template classifier, compression classifier, graph-kernel probe` under equal information.
Absorption token family: `ABSORBED_BY_GENERATOR_FAMILY_IDENTIFICATION`.
Claim ceiling if untested: inconclusive for this surface.

G.216. Surface: query efficiency.
Rung: Post-hoc compression.
Audit question: does the artifact summarizes behavior after success?
Required evidence: run or justify `frozen-before-score compression control` under equal information.
Absorption token family: `ABSORBED_BY_POST_HOC_COMPRESSION`.
Claim ceiling if untested: inconclusive for this surface.

G.217. Surface: human-substrate accounting.
Rung: Representation prior.
Audit question: does public features expose the answer?
Required evidence: run or justify `feature parser, type tags, names, ontology` under equal information.
Absorption token family: `ABSORBED_BY_REPRESENTATION_PRIOR`.
Claim ceiling if untested: inconclusive for this surface.

G.218. Surface: human-substrate accounting.
Rung: Parser or substrate prior.
Audit question: does the parser, verifier, DSL, action space, or canonicalizer does the work?
Required evidence: run or justify `shared-or-charged substrate baseline` under equal information.
Absorption token family: `ABSORBED_BY_PARSER_OR_SUBSTRATE_PRIOR`.
Claim ceiling if untested: inconclusive for this surface.

G.219. Surface: human-substrate accounting.
Rung: Teaching dimension.
Audit question: does the packet is a small teaching set?
Required evidence: run or justify `finite teaching-set solver or version-space search` under equal information.
Absorption token family: `ABSORBED_BY_TEACHING_DIMENSION`.
Claim ceiling if untested: inconclusive for this surface.

G.220. Surface: human-substrate accounting.
Rung: Active learning.
Audit question: does questions isolate the target cheaply?
Required evidence: run or justify `query planner, group tests, active counterexample search` under equal information.
Absorption token family: `ABSORBED_BY_ACTIVE_LEARNING`.
Claim ceiling if untested: inconclusive for this surface.

G.221. Surface: human-substrate accounting.
Rung: PBE or CEGIS.
Audit question: does a program can be synthesized from the same evidence?
Required evidence: run or justify `PBE, CEGIS, SyGuS, ILP, relational rule search` under equal information.
Absorption token family: `ABSORBED_BY_CEGIS_OR_SYNTHESIS`.
Claim ceiling if untested: inconclusive for this surface.

G.222. Surface: human-substrate accounting.
Rung: Library learning.
Audit question: does reusable macros explain transfer?
Required evidence: run or justify `MDL library, DreamCoder-style library, e-graph composition` under equal information.
Absorption token family: `ABSORBED_BY_LIBRARY_LEARNING`.
Claim ceiling if untested: inconclusive for this surface.

G.223. Surface: human-substrate accounting.
Rung: Schema or binding.
Audit question: does role bindings solve the task?
Required evidence: run or justify `schema matcher, binding fingerprint, entity resolver` under equal information.
Absorption token family: `ABSORBED_BY_SCHEMA_OR_BINDING`.
Claim ceiling if untested: inconclusive for this surface.

G.224. Surface: human-substrate accounting.
Rung: Domain tool.
Audit question: does the task is native to a known local tool?
Required evidence: run or justify `SQL, spreadsheets, SAT, SMT, ILP, ASP, CSP` under equal information.
Absorption token family: `ABSORBED_BY_DOMAIN_TOOL`.
Claim ceiling if untested: inconclusive for this surface.

G.225. Surface: human-substrate accounting.
Rung: Nuisance oracle.
Audit question: does difficulty vanishes when nuisance is removed?
Required evidence: run or justify `relevant-feature oracle or invariant oracle` under equal information.
Absorption token family: `ABSORBED_BY_NUISANCE_ORACLE`.
Claim ceiling if untested: inconclusive for this surface.

G.226. Surface: human-substrate accounting.
Rung: Constraint discovery.
Audit question: does the artifact is a constraint theory?
Required evidence: run or justify `rank solver, SAT/SMT encoding, table-constraint learner` under equal information.
Absorption token family: `ABSORBED_BY_CONSTRAINT_DISCOVERY`.
Claim ceiling if untested: inconclusive for this surface.

G.227. Surface: human-substrate accounting.
Rung: Generator family.
Audit question: does public statistics identify the synthetic generator?
Required evidence: run or justify `template classifier, compression classifier, graph-kernel probe` under equal information.
Absorption token family: `ABSORBED_BY_GENERATOR_FAMILY_IDENTIFICATION`.
Claim ceiling if untested: inconclusive for this surface.

G.228. Surface: human-substrate accounting.
Rung: Post-hoc compression.
Audit question: does the artifact summarizes behavior after success?
Required evidence: run or justify `frozen-before-score compression control` under equal information.
Absorption token family: `ABSORBED_BY_POST_HOC_COMPRESSION`.
Claim ceiling if untested: inconclusive for this surface.

G.229. Surface: claim-ceiling compliance.
Rung: Representation prior.
Audit question: does public features expose the answer?
Required evidence: run or justify `feature parser, type tags, names, ontology` under equal information.
Absorption token family: `ABSORBED_BY_REPRESENTATION_PRIOR`.
Claim ceiling if untested: inconclusive for this surface.

G.230. Surface: claim-ceiling compliance.
Rung: Parser or substrate prior.
Audit question: does the parser, verifier, DSL, action space, or canonicalizer does the work?
Required evidence: run or justify `shared-or-charged substrate baseline` under equal information.
Absorption token family: `ABSORBED_BY_PARSER_OR_SUBSTRATE_PRIOR`.
Claim ceiling if untested: inconclusive for this surface.

G.231. Surface: claim-ceiling compliance.
Rung: Teaching dimension.
Audit question: does the packet is a small teaching set?
Required evidence: run or justify `finite teaching-set solver or version-space search` under equal information.
Absorption token family: `ABSORBED_BY_TEACHING_DIMENSION`.
Claim ceiling if untested: inconclusive for this surface.

G.232. Surface: claim-ceiling compliance.
Rung: Active learning.
Audit question: does questions isolate the target cheaply?
Required evidence: run or justify `query planner, group tests, active counterexample search` under equal information.
Absorption token family: `ABSORBED_BY_ACTIVE_LEARNING`.
Claim ceiling if untested: inconclusive for this surface.

G.233. Surface: claim-ceiling compliance.
Rung: PBE or CEGIS.
Audit question: does a program can be synthesized from the same evidence?
Required evidence: run or justify `PBE, CEGIS, SyGuS, ILP, relational rule search` under equal information.
Absorption token family: `ABSORBED_BY_CEGIS_OR_SYNTHESIS`.
Claim ceiling if untested: inconclusive for this surface.

G.234. Surface: claim-ceiling compliance.
Rung: Library learning.
Audit question: does reusable macros explain transfer?
Required evidence: run or justify `MDL library, DreamCoder-style library, e-graph composition` under equal information.
Absorption token family: `ABSORBED_BY_LIBRARY_LEARNING`.
Claim ceiling if untested: inconclusive for this surface.

G.235. Surface: claim-ceiling compliance.
Rung: Schema or binding.
Audit question: does role bindings solve the task?
Required evidence: run or justify `schema matcher, binding fingerprint, entity resolver` under equal information.
Absorption token family: `ABSORBED_BY_SCHEMA_OR_BINDING`.
Claim ceiling if untested: inconclusive for this surface.

G.236. Surface: claim-ceiling compliance.
Rung: Domain tool.
Audit question: does the task is native to a known local tool?
Required evidence: run or justify `SQL, spreadsheets, SAT, SMT, ILP, ASP, CSP` under equal information.
Absorption token family: `ABSORBED_BY_DOMAIN_TOOL`.
Claim ceiling if untested: inconclusive for this surface.

G.237. Surface: claim-ceiling compliance.
Rung: Nuisance oracle.
Audit question: does difficulty vanishes when nuisance is removed?
Required evidence: run or justify `relevant-feature oracle or invariant oracle` under equal information.
Absorption token family: `ABSORBED_BY_NUISANCE_ORACLE`.
Claim ceiling if untested: inconclusive for this surface.

G.238. Surface: claim-ceiling compliance.
Rung: Constraint discovery.
Audit question: does the artifact is a constraint theory?
Required evidence: run or justify `rank solver, SAT/SMT encoding, table-constraint learner` under equal information.
Absorption token family: `ABSORBED_BY_CONSTRAINT_DISCOVERY`.
Claim ceiling if untested: inconclusive for this surface.

G.239. Surface: claim-ceiling compliance.
Rung: Generator family.
Audit question: does public statistics identify the synthetic generator?
Required evidence: run or justify `template classifier, compression classifier, graph-kernel probe` under equal information.
Absorption token family: `ABSORBED_BY_GENERATOR_FAMILY_IDENTIFICATION`.
Claim ceiling if untested: inconclusive for this surface.

G.240. Surface: claim-ceiling compliance.
Rung: Post-hoc compression.
Audit question: does the artifact summarizes behavior after success?
Required evidence: run or justify `frozen-before-score compression control` under equal information.
Absorption token family: `ABSORBED_BY_POST_HOC_COMPRESSION`.
Claim ceiling if untested: inconclusive for this surface.

## Appendix H: Additional Hostile-Review Clauses

H.1. Clause for functional accuracy, Representation prior, and FrameSeed Boolean.
Question: could feature parser, type tags, names, ontology explain the functional accuracy result in `experiments/frameseed0_b28_hidden_hfa.json`?
Required response: cite the exact metric, equal-information map, and all-in cost entry.
Failure response: lower the claim ceiling or mark the surface inconclusive.
Relevant terminal token family: `ABSORBED_BY_REPRESENTATION_PRIOR`.

H.2. Clause for functional accuracy, Representation prior, and FrameSeed SHEETS-0.
Question: could feature parser, type tags, names, ontology explain the functional accuracy result in `experiments/frameseed_sheets0_b31_hidden_hfa.json`?
Required response: cite the exact metric, equal-information map, and all-in cost entry.
Failure response: lower the claim ceiling or mark the surface inconclusive.
Relevant terminal token family: `ABSORBED_BY_REPRESENTATION_PRIOR`.

H.3. Clause for functional accuracy, Representation prior, and WGD-0 Toy Grammar.
Question: could feature parser, type tags, names, ontology explain the functional accuracy result in `experiments/wgd0_b37_hidden_measurement.json`?
Required response: cite the exact metric, equal-information map, and all-in cost entry.
Failure response: lower the claim ceiling or mark the surface inconclusive.
Relevant terminal token family: `ABSORBED_BY_REPRESENTATION_PRIOR`.

H.4. Clause for functional accuracy, Representation prior, and WGD-0 Hard Domain.
Question: could feature parser, type tags, names, ontology explain the functional accuracy result in `experiments/wgd0_b38_hidden_measurement.json`?
Required response: cite the exact metric, equal-information map, and all-in cost entry.
Failure response: lower the claim ceiling or mark the surface inconclusive.
Relevant terminal token family: `ABSORBED_BY_REPRESENTATION_PRIOR`.

H.5. Clause for functional accuracy, Parser or substrate prior, and FrameSeed Boolean.
Question: could shared-or-charged substrate baseline explain the functional accuracy result in `experiments/frameseed0_b28_hidden_hfa.json`?
Required response: cite the exact metric, equal-information map, and all-in cost entry.
Failure response: lower the claim ceiling or mark the surface inconclusive.
Relevant terminal token family: `ABSORBED_BY_PARSER_OR_SUBSTRATE_PRIOR`.

H.6. Clause for functional accuracy, Parser or substrate prior, and FrameSeed SHEETS-0.
Question: could shared-or-charged substrate baseline explain the functional accuracy result in `experiments/frameseed_sheets0_b31_hidden_hfa.json`?
Required response: cite the exact metric, equal-information map, and all-in cost entry.
Failure response: lower the claim ceiling or mark the surface inconclusive.
Relevant terminal token family: `ABSORBED_BY_PARSER_OR_SUBSTRATE_PRIOR`.

H.7. Clause for functional accuracy, Parser or substrate prior, and WGD-0 Toy Grammar.
Question: could shared-or-charged substrate baseline explain the functional accuracy result in `experiments/wgd0_b37_hidden_measurement.json`?
Required response: cite the exact metric, equal-information map, and all-in cost entry.
Failure response: lower the claim ceiling or mark the surface inconclusive.
Relevant terminal token family: `ABSORBED_BY_PARSER_OR_SUBSTRATE_PRIOR`.

H.8. Clause for functional accuracy, Parser or substrate prior, and WGD-0 Hard Domain.
Question: could shared-or-charged substrate baseline explain the functional accuracy result in `experiments/wgd0_b38_hidden_measurement.json`?
Required response: cite the exact metric, equal-information map, and all-in cost entry.
Failure response: lower the claim ceiling or mark the surface inconclusive.
Relevant terminal token family: `ABSORBED_BY_PARSER_OR_SUBSTRATE_PRIOR`.

H.9. Clause for functional accuracy, Teaching dimension, and FrameSeed Boolean.
Question: could finite teaching-set solver or version-space search explain the functional accuracy result in `experiments/frameseed0_b28_hidden_hfa.json`?
Required response: cite the exact metric, equal-information map, and all-in cost entry.
Failure response: lower the claim ceiling or mark the surface inconclusive.
Relevant terminal token family: `ABSORBED_BY_TEACHING_DIMENSION`.

H.10. Clause for functional accuracy, Teaching dimension, and FrameSeed SHEETS-0.
Question: could finite teaching-set solver or version-space search explain the functional accuracy result in `experiments/frameseed_sheets0_b31_hidden_hfa.json`?
Required response: cite the exact metric, equal-information map, and all-in cost entry.
Failure response: lower the claim ceiling or mark the surface inconclusive.
Relevant terminal token family: `ABSORBED_BY_TEACHING_DIMENSION`.

H.11. Clause for functional accuracy, Teaching dimension, and WGD-0 Toy Grammar.
Question: could finite teaching-set solver or version-space search explain the functional accuracy result in `experiments/wgd0_b37_hidden_measurement.json`?
Required response: cite the exact metric, equal-information map, and all-in cost entry.
Failure response: lower the claim ceiling or mark the surface inconclusive.
Relevant terminal token family: `ABSORBED_BY_TEACHING_DIMENSION`.

H.12. Clause for functional accuracy, Teaching dimension, and WGD-0 Hard Domain.
Question: could finite teaching-set solver or version-space search explain the functional accuracy result in `experiments/wgd0_b38_hidden_measurement.json`?
Required response: cite the exact metric, equal-information map, and all-in cost entry.
Failure response: lower the claim ceiling or mark the surface inconclusive.
Relevant terminal token family: `ABSORBED_BY_TEACHING_DIMENSION`.

H.13. Clause for functional accuracy, Active learning, and FrameSeed Boolean.
Question: could query planner, group tests, active counterexample search explain the functional accuracy result in `experiments/frameseed0_b28_hidden_hfa.json`?
Required response: cite the exact metric, equal-information map, and all-in cost entry.
Failure response: lower the claim ceiling or mark the surface inconclusive.
Relevant terminal token family: `ABSORBED_BY_ACTIVE_LEARNING`.

H.14. Clause for functional accuracy, Active learning, and FrameSeed SHEETS-0.
Question: could query planner, group tests, active counterexample search explain the functional accuracy result in `experiments/frameseed_sheets0_b31_hidden_hfa.json`?
Required response: cite the exact metric, equal-information map, and all-in cost entry.
Failure response: lower the claim ceiling or mark the surface inconclusive.
Relevant terminal token family: `ABSORBED_BY_ACTIVE_LEARNING`.

H.15. Clause for functional accuracy, Active learning, and WGD-0 Toy Grammar.
Question: could query planner, group tests, active counterexample search explain the functional accuracy result in `experiments/wgd0_b37_hidden_measurement.json`?
Required response: cite the exact metric, equal-information map, and all-in cost entry.
Failure response: lower the claim ceiling or mark the surface inconclusive.
Relevant terminal token family: `ABSORBED_BY_ACTIVE_LEARNING`.

H.16. Clause for functional accuracy, Active learning, and WGD-0 Hard Domain.
Question: could query planner, group tests, active counterexample search explain the functional accuracy result in `experiments/wgd0_b38_hidden_measurement.json`?
Required response: cite the exact metric, equal-information map, and all-in cost entry.
Failure response: lower the claim ceiling or mark the surface inconclusive.
Relevant terminal token family: `ABSORBED_BY_ACTIVE_LEARNING`.

H.17. Clause for functional accuracy, PBE or CEGIS, and FrameSeed Boolean.
Question: could PBE, CEGIS, SyGuS, ILP, relational rule search explain the functional accuracy result in `experiments/frameseed0_b28_hidden_hfa.json`?
Required response: cite the exact metric, equal-information map, and all-in cost entry.
Failure response: lower the claim ceiling or mark the surface inconclusive.
Relevant terminal token family: `ABSORBED_BY_CEGIS_OR_SYNTHESIS`.

H.18. Clause for functional accuracy, PBE or CEGIS, and FrameSeed SHEETS-0.
Question: could PBE, CEGIS, SyGuS, ILP, relational rule search explain the functional accuracy result in `experiments/frameseed_sheets0_b31_hidden_hfa.json`?
Required response: cite the exact metric, equal-information map, and all-in cost entry.
Failure response: lower the claim ceiling or mark the surface inconclusive.
Relevant terminal token family: `ABSORBED_BY_CEGIS_OR_SYNTHESIS`.

H.19. Clause for functional accuracy, PBE or CEGIS, and WGD-0 Toy Grammar.
Question: could PBE, CEGIS, SyGuS, ILP, relational rule search explain the functional accuracy result in `experiments/wgd0_b37_hidden_measurement.json`?
Required response: cite the exact metric, equal-information map, and all-in cost entry.
Failure response: lower the claim ceiling or mark the surface inconclusive.
Relevant terminal token family: `ABSORBED_BY_CEGIS_OR_SYNTHESIS`.

H.20. Clause for functional accuracy, PBE or CEGIS, and WGD-0 Hard Domain.
Question: could PBE, CEGIS, SyGuS, ILP, relational rule search explain the functional accuracy result in `experiments/wgd0_b38_hidden_measurement.json`?
Required response: cite the exact metric, equal-information map, and all-in cost entry.
Failure response: lower the claim ceiling or mark the surface inconclusive.
Relevant terminal token family: `ABSORBED_BY_CEGIS_OR_SYNTHESIS`.

H.21. Clause for functional accuracy, Library learning, and FrameSeed Boolean.
Question: could MDL library, DreamCoder-style library, e-graph composition explain the functional accuracy result in `experiments/frameseed0_b28_hidden_hfa.json`?
Required response: cite the exact metric, equal-information map, and all-in cost entry.
Failure response: lower the claim ceiling or mark the surface inconclusive.
Relevant terminal token family: `ABSORBED_BY_LIBRARY_LEARNING`.

H.22. Clause for functional accuracy, Library learning, and FrameSeed SHEETS-0.
Question: could MDL library, DreamCoder-style library, e-graph composition explain the functional accuracy result in `experiments/frameseed_sheets0_b31_hidden_hfa.json`?
Required response: cite the exact metric, equal-information map, and all-in cost entry.
Failure response: lower the claim ceiling or mark the surface inconclusive.
Relevant terminal token family: `ABSORBED_BY_LIBRARY_LEARNING`.

H.23. Clause for functional accuracy, Library learning, and WGD-0 Toy Grammar.
Question: could MDL library, DreamCoder-style library, e-graph composition explain the functional accuracy result in `experiments/wgd0_b37_hidden_measurement.json`?
Required response: cite the exact metric, equal-information map, and all-in cost entry.
Failure response: lower the claim ceiling or mark the surface inconclusive.
Relevant terminal token family: `ABSORBED_BY_LIBRARY_LEARNING`.

H.24. Clause for functional accuracy, Library learning, and WGD-0 Hard Domain.
Question: could MDL library, DreamCoder-style library, e-graph composition explain the functional accuracy result in `experiments/wgd0_b38_hidden_measurement.json`?
Required response: cite the exact metric, equal-information map, and all-in cost entry.
Failure response: lower the claim ceiling or mark the surface inconclusive.
Relevant terminal token family: `ABSORBED_BY_LIBRARY_LEARNING`.

H.25. Clause for functional accuracy, Schema or binding, and FrameSeed Boolean.
Question: could schema matcher, binding fingerprint, entity resolver explain the functional accuracy result in `experiments/frameseed0_b28_hidden_hfa.json`?
Required response: cite the exact metric, equal-information map, and all-in cost entry.
Failure response: lower the claim ceiling or mark the surface inconclusive.
Relevant terminal token family: `ABSORBED_BY_SCHEMA_OR_BINDING`.

H.26. Clause for functional accuracy, Schema or binding, and FrameSeed SHEETS-0.
Question: could schema matcher, binding fingerprint, entity resolver explain the functional accuracy result in `experiments/frameseed_sheets0_b31_hidden_hfa.json`?
Required response: cite the exact metric, equal-information map, and all-in cost entry.
Failure response: lower the claim ceiling or mark the surface inconclusive.
Relevant terminal token family: `ABSORBED_BY_SCHEMA_OR_BINDING`.

H.27. Clause for functional accuracy, Schema or binding, and WGD-0 Toy Grammar.
Question: could schema matcher, binding fingerprint, entity resolver explain the functional accuracy result in `experiments/wgd0_b37_hidden_measurement.json`?
Required response: cite the exact metric, equal-information map, and all-in cost entry.
Failure response: lower the claim ceiling or mark the surface inconclusive.
Relevant terminal token family: `ABSORBED_BY_SCHEMA_OR_BINDING`.

H.28. Clause for functional accuracy, Schema or binding, and WGD-0 Hard Domain.
Question: could schema matcher, binding fingerprint, entity resolver explain the functional accuracy result in `experiments/wgd0_b38_hidden_measurement.json`?
Required response: cite the exact metric, equal-information map, and all-in cost entry.
Failure response: lower the claim ceiling or mark the surface inconclusive.
Relevant terminal token family: `ABSORBED_BY_SCHEMA_OR_BINDING`.

H.29. Clause for functional accuracy, Domain tool, and FrameSeed Boolean.
Question: could SQL, spreadsheets, SAT, SMT, ILP, ASP, CSP explain the functional accuracy result in `experiments/frameseed0_b28_hidden_hfa.json`?
Required response: cite the exact metric, equal-information map, and all-in cost entry.
Failure response: lower the claim ceiling or mark the surface inconclusive.
Relevant terminal token family: `ABSORBED_BY_DOMAIN_TOOL`.

H.30. Clause for functional accuracy, Domain tool, and FrameSeed SHEETS-0.
Question: could SQL, spreadsheets, SAT, SMT, ILP, ASP, CSP explain the functional accuracy result in `experiments/frameseed_sheets0_b31_hidden_hfa.json`?
Required response: cite the exact metric, equal-information map, and all-in cost entry.
Failure response: lower the claim ceiling or mark the surface inconclusive.
Relevant terminal token family: `ABSORBED_BY_DOMAIN_TOOL`.

H.31. Clause for functional accuracy, Domain tool, and WGD-0 Toy Grammar.
Question: could SQL, spreadsheets, SAT, SMT, ILP, ASP, CSP explain the functional accuracy result in `experiments/wgd0_b37_hidden_measurement.json`?
Required response: cite the exact metric, equal-information map, and all-in cost entry.
Failure response: lower the claim ceiling or mark the surface inconclusive.
Relevant terminal token family: `ABSORBED_BY_DOMAIN_TOOL`.

H.32. Clause for functional accuracy, Domain tool, and WGD-0 Hard Domain.
Question: could SQL, spreadsheets, SAT, SMT, ILP, ASP, CSP explain the functional accuracy result in `experiments/wgd0_b38_hidden_measurement.json`?
Required response: cite the exact metric, equal-information map, and all-in cost entry.
Failure response: lower the claim ceiling or mark the surface inconclusive.
Relevant terminal token family: `ABSORBED_BY_DOMAIN_TOOL`.

H.33. Clause for functional accuracy, Nuisance oracle, and FrameSeed Boolean.
Question: could relevant-feature oracle or invariant oracle explain the functional accuracy result in `experiments/frameseed0_b28_hidden_hfa.json`?
Required response: cite the exact metric, equal-information map, and all-in cost entry.
Failure response: lower the claim ceiling or mark the surface inconclusive.
Relevant terminal token family: `ABSORBED_BY_NUISANCE_ORACLE`.

H.34. Clause for functional accuracy, Nuisance oracle, and FrameSeed SHEETS-0.
Question: could relevant-feature oracle or invariant oracle explain the functional accuracy result in `experiments/frameseed_sheets0_b31_hidden_hfa.json`?
Required response: cite the exact metric, equal-information map, and all-in cost entry.
Failure response: lower the claim ceiling or mark the surface inconclusive.
Relevant terminal token family: `ABSORBED_BY_NUISANCE_ORACLE`.

H.35. Clause for functional accuracy, Nuisance oracle, and WGD-0 Toy Grammar.
Question: could relevant-feature oracle or invariant oracle explain the functional accuracy result in `experiments/wgd0_b37_hidden_measurement.json`?
Required response: cite the exact metric, equal-information map, and all-in cost entry.
Failure response: lower the claim ceiling or mark the surface inconclusive.
Relevant terminal token family: `ABSORBED_BY_NUISANCE_ORACLE`.

H.36. Clause for functional accuracy, Nuisance oracle, and WGD-0 Hard Domain.
Question: could relevant-feature oracle or invariant oracle explain the functional accuracy result in `experiments/wgd0_b38_hidden_measurement.json`?
Required response: cite the exact metric, equal-information map, and all-in cost entry.
Failure response: lower the claim ceiling or mark the surface inconclusive.
Relevant terminal token family: `ABSORBED_BY_NUISANCE_ORACLE`.

H.37. Clause for functional accuracy, Constraint discovery, and FrameSeed Boolean.
Question: could rank solver, SAT/SMT encoding, table-constraint learner explain the functional accuracy result in `experiments/frameseed0_b28_hidden_hfa.json`?
Required response: cite the exact metric, equal-information map, and all-in cost entry.
Failure response: lower the claim ceiling or mark the surface inconclusive.
Relevant terminal token family: `ABSORBED_BY_CONSTRAINT_DISCOVERY`.

H.38. Clause for functional accuracy, Constraint discovery, and FrameSeed SHEETS-0.
Question: could rank solver, SAT/SMT encoding, table-constraint learner explain the functional accuracy result in `experiments/frameseed_sheets0_b31_hidden_hfa.json`?
Required response: cite the exact metric, equal-information map, and all-in cost entry.
Failure response: lower the claim ceiling or mark the surface inconclusive.
Relevant terminal token family: `ABSORBED_BY_CONSTRAINT_DISCOVERY`.

H.39. Clause for functional accuracy, Constraint discovery, and WGD-0 Toy Grammar.
Question: could rank solver, SAT/SMT encoding, table-constraint learner explain the functional accuracy result in `experiments/wgd0_b37_hidden_measurement.json`?
Required response: cite the exact metric, equal-information map, and all-in cost entry.
Failure response: lower the claim ceiling or mark the surface inconclusive.
Relevant terminal token family: `ABSORBED_BY_CONSTRAINT_DISCOVERY`.

H.40. Clause for functional accuracy, Constraint discovery, and WGD-0 Hard Domain.
Question: could rank solver, SAT/SMT encoding, table-constraint learner explain the functional accuracy result in `experiments/wgd0_b38_hidden_measurement.json`?
Required response: cite the exact metric, equal-information map, and all-in cost entry.
Failure response: lower the claim ceiling or mark the surface inconclusive.
Relevant terminal token family: `ABSORBED_BY_CONSTRAINT_DISCOVERY`.

H.41. Clause for functional accuracy, Generator family, and FrameSeed Boolean.
Question: could template classifier, compression classifier, graph-kernel probe explain the functional accuracy result in `experiments/frameseed0_b28_hidden_hfa.json`?
Required response: cite the exact metric, equal-information map, and all-in cost entry.
Failure response: lower the claim ceiling or mark the surface inconclusive.
Relevant terminal token family: `ABSORBED_BY_GENERATOR_FAMILY_IDENTIFICATION`.

H.42. Clause for functional accuracy, Generator family, and FrameSeed SHEETS-0.
Question: could template classifier, compression classifier, graph-kernel probe explain the functional accuracy result in `experiments/frameseed_sheets0_b31_hidden_hfa.json`?
Required response: cite the exact metric, equal-information map, and all-in cost entry.
Failure response: lower the claim ceiling or mark the surface inconclusive.
Relevant terminal token family: `ABSORBED_BY_GENERATOR_FAMILY_IDENTIFICATION`.

H.43. Clause for functional accuracy, Generator family, and WGD-0 Toy Grammar.
Question: could template classifier, compression classifier, graph-kernel probe explain the functional accuracy result in `experiments/wgd0_b37_hidden_measurement.json`?
Required response: cite the exact metric, equal-information map, and all-in cost entry.
Failure response: lower the claim ceiling or mark the surface inconclusive.
Relevant terminal token family: `ABSORBED_BY_GENERATOR_FAMILY_IDENTIFICATION`.

H.44. Clause for functional accuracy, Generator family, and WGD-0 Hard Domain.
Question: could template classifier, compression classifier, graph-kernel probe explain the functional accuracy result in `experiments/wgd0_b38_hidden_measurement.json`?
Required response: cite the exact metric, equal-information map, and all-in cost entry.
Failure response: lower the claim ceiling or mark the surface inconclusive.
Relevant terminal token family: `ABSORBED_BY_GENERATOR_FAMILY_IDENTIFICATION`.

H.45. Clause for functional accuracy, Post-hoc compression, and FrameSeed Boolean.
Question: could frozen-before-score compression control explain the functional accuracy result in `experiments/frameseed0_b28_hidden_hfa.json`?
Required response: cite the exact metric, equal-information map, and all-in cost entry.
Failure response: lower the claim ceiling or mark the surface inconclusive.
Relevant terminal token family: `ABSORBED_BY_POST_HOC_COMPRESSION`.

H.46. Clause for functional accuracy, Post-hoc compression, and FrameSeed SHEETS-0.
Question: could frozen-before-score compression control explain the functional accuracy result in `experiments/frameseed_sheets0_b31_hidden_hfa.json`?
Required response: cite the exact metric, equal-information map, and all-in cost entry.
Failure response: lower the claim ceiling or mark the surface inconclusive.
Relevant terminal token family: `ABSORBED_BY_POST_HOC_COMPRESSION`.

H.47. Clause for functional accuracy, Post-hoc compression, and WGD-0 Toy Grammar.
Question: could frozen-before-score compression control explain the functional accuracy result in `experiments/wgd0_b37_hidden_measurement.json`?
Required response: cite the exact metric, equal-information map, and all-in cost entry.
Failure response: lower the claim ceiling or mark the surface inconclusive.
Relevant terminal token family: `ABSORBED_BY_POST_HOC_COMPRESSION`.

H.48. Clause for functional accuracy, Post-hoc compression, and WGD-0 Hard Domain.
Question: could frozen-before-score compression control explain the functional accuracy result in `experiments/wgd0_b38_hidden_measurement.json`?
Required response: cite the exact metric, equal-information map, and all-in cost entry.
Failure response: lower the claim ceiling or mark the surface inconclusive.
Relevant terminal token family: `ABSORBED_BY_POST_HOC_COMPRESSION`.

H.49. Clause for minimum per-family accuracy, Representation prior, and FrameSeed Boolean.
Question: could feature parser, type tags, names, ontology explain the minimum per-family accuracy result in `experiments/frameseed0_b28_hidden_hfa.json`?
Required response: cite the exact metric, equal-information map, and all-in cost entry.
Failure response: lower the claim ceiling or mark the surface inconclusive.
Relevant terminal token family: `ABSORBED_BY_REPRESENTATION_PRIOR`.

H.50. Clause for minimum per-family accuracy, Representation prior, and FrameSeed SHEETS-0.
Question: could feature parser, type tags, names, ontology explain the minimum per-family accuracy result in `experiments/frameseed_sheets0_b31_hidden_hfa.json`?
Required response: cite the exact metric, equal-information map, and all-in cost entry.
Failure response: lower the claim ceiling or mark the surface inconclusive.
Relevant terminal token family: `ABSORBED_BY_REPRESENTATION_PRIOR`.

H.51. Clause for minimum per-family accuracy, Representation prior, and WGD-0 Toy Grammar.
Question: could feature parser, type tags, names, ontology explain the minimum per-family accuracy result in `experiments/wgd0_b37_hidden_measurement.json`?
Required response: cite the exact metric, equal-information map, and all-in cost entry.
Failure response: lower the claim ceiling or mark the surface inconclusive.
Relevant terminal token family: `ABSORBED_BY_REPRESENTATION_PRIOR`.

H.52. Clause for minimum per-family accuracy, Representation prior, and WGD-0 Hard Domain.
Question: could feature parser, type tags, names, ontology explain the minimum per-family accuracy result in `experiments/wgd0_b38_hidden_measurement.json`?
Required response: cite the exact metric, equal-information map, and all-in cost entry.
Failure response: lower the claim ceiling or mark the surface inconclusive.
Relevant terminal token family: `ABSORBED_BY_REPRESENTATION_PRIOR`.

H.53. Clause for minimum per-family accuracy, Parser or substrate prior, and FrameSeed Boolean.
Question: could shared-or-charged substrate baseline explain the minimum per-family accuracy result in `experiments/frameseed0_b28_hidden_hfa.json`?
Required response: cite the exact metric, equal-information map, and all-in cost entry.
Failure response: lower the claim ceiling or mark the surface inconclusive.
Relevant terminal token family: `ABSORBED_BY_PARSER_OR_SUBSTRATE_PRIOR`.

H.54. Clause for minimum per-family accuracy, Parser or substrate prior, and FrameSeed SHEETS-0.
Question: could shared-or-charged substrate baseline explain the minimum per-family accuracy result in `experiments/frameseed_sheets0_b31_hidden_hfa.json`?
Required response: cite the exact metric, equal-information map, and all-in cost entry.
Failure response: lower the claim ceiling or mark the surface inconclusive.
Relevant terminal token family: `ABSORBED_BY_PARSER_OR_SUBSTRATE_PRIOR`.

H.55. Clause for minimum per-family accuracy, Parser or substrate prior, and WGD-0 Toy Grammar.
Question: could shared-or-charged substrate baseline explain the minimum per-family accuracy result in `experiments/wgd0_b37_hidden_measurement.json`?
Required response: cite the exact metric, equal-information map, and all-in cost entry.
Failure response: lower the claim ceiling or mark the surface inconclusive.
Relevant terminal token family: `ABSORBED_BY_PARSER_OR_SUBSTRATE_PRIOR`.

H.56. Clause for minimum per-family accuracy, Parser or substrate prior, and WGD-0 Hard Domain.
Question: could shared-or-charged substrate baseline explain the minimum per-family accuracy result in `experiments/wgd0_b38_hidden_measurement.json`?
Required response: cite the exact metric, equal-information map, and all-in cost entry.
Failure response: lower the claim ceiling or mark the surface inconclusive.
Relevant terminal token family: `ABSORBED_BY_PARSER_OR_SUBSTRATE_PRIOR`.

H.57. Clause for minimum per-family accuracy, Teaching dimension, and FrameSeed Boolean.
Question: could finite teaching-set solver or version-space search explain the minimum per-family accuracy result in `experiments/frameseed0_b28_hidden_hfa.json`?
Required response: cite the exact metric, equal-information map, and all-in cost entry.
Failure response: lower the claim ceiling or mark the surface inconclusive.
Relevant terminal token family: `ABSORBED_BY_TEACHING_DIMENSION`.

H.58. Clause for minimum per-family accuracy, Teaching dimension, and FrameSeed SHEETS-0.
Question: could finite teaching-set solver or version-space search explain the minimum per-family accuracy result in `experiments/frameseed_sheets0_b31_hidden_hfa.json`?
Required response: cite the exact metric, equal-information map, and all-in cost entry.
Failure response: lower the claim ceiling or mark the surface inconclusive.
Relevant terminal token family: `ABSORBED_BY_TEACHING_DIMENSION`.

H.59. Clause for minimum per-family accuracy, Teaching dimension, and WGD-0 Toy Grammar.
Question: could finite teaching-set solver or version-space search explain the minimum per-family accuracy result in `experiments/wgd0_b37_hidden_measurement.json`?
Required response: cite the exact metric, equal-information map, and all-in cost entry.
Failure response: lower the claim ceiling or mark the surface inconclusive.
Relevant terminal token family: `ABSORBED_BY_TEACHING_DIMENSION`.

H.60. Clause for minimum per-family accuracy, Teaching dimension, and WGD-0 Hard Domain.
Question: could finite teaching-set solver or version-space search explain the minimum per-family accuracy result in `experiments/wgd0_b38_hidden_measurement.json`?
Required response: cite the exact metric, equal-information map, and all-in cost entry.
Failure response: lower the claim ceiling or mark the surface inconclusive.
Relevant terminal token family: `ABSORBED_BY_TEACHING_DIMENSION`.

H.61. Clause for minimum per-family accuracy, Active learning, and FrameSeed Boolean.
Question: could query planner, group tests, active counterexample search explain the minimum per-family accuracy result in `experiments/frameseed0_b28_hidden_hfa.json`?
Required response: cite the exact metric, equal-information map, and all-in cost entry.
Failure response: lower the claim ceiling or mark the surface inconclusive.
Relevant terminal token family: `ABSORBED_BY_ACTIVE_LEARNING`.

H.62. Clause for minimum per-family accuracy, Active learning, and FrameSeed SHEETS-0.
Question: could query planner, group tests, active counterexample search explain the minimum per-family accuracy result in `experiments/frameseed_sheets0_b31_hidden_hfa.json`?
Required response: cite the exact metric, equal-information map, and all-in cost entry.
Failure response: lower the claim ceiling or mark the surface inconclusive.
Relevant terminal token family: `ABSORBED_BY_ACTIVE_LEARNING`.

H.63. Clause for minimum per-family accuracy, Active learning, and WGD-0 Toy Grammar.
Question: could query planner, group tests, active counterexample search explain the minimum per-family accuracy result in `experiments/wgd0_b37_hidden_measurement.json`?
Required response: cite the exact metric, equal-information map, and all-in cost entry.
Failure response: lower the claim ceiling or mark the surface inconclusive.
Relevant terminal token family: `ABSORBED_BY_ACTIVE_LEARNING`.

H.64. Clause for minimum per-family accuracy, Active learning, and WGD-0 Hard Domain.
Question: could query planner, group tests, active counterexample search explain the minimum per-family accuracy result in `experiments/wgd0_b38_hidden_measurement.json`?
Required response: cite the exact metric, equal-information map, and all-in cost entry.
Failure response: lower the claim ceiling or mark the surface inconclusive.
Relevant terminal token family: `ABSORBED_BY_ACTIVE_LEARNING`.

H.65. Clause for minimum per-family accuracy, PBE or CEGIS, and FrameSeed Boolean.
Question: could PBE, CEGIS, SyGuS, ILP, relational rule search explain the minimum per-family accuracy result in `experiments/frameseed0_b28_hidden_hfa.json`?
Required response: cite the exact metric, equal-information map, and all-in cost entry.
Failure response: lower the claim ceiling or mark the surface inconclusive.
Relevant terminal token family: `ABSORBED_BY_CEGIS_OR_SYNTHESIS`.

H.66. Clause for minimum per-family accuracy, PBE or CEGIS, and FrameSeed SHEETS-0.
Question: could PBE, CEGIS, SyGuS, ILP, relational rule search explain the minimum per-family accuracy result in `experiments/frameseed_sheets0_b31_hidden_hfa.json`?
Required response: cite the exact metric, equal-information map, and all-in cost entry.
Failure response: lower the claim ceiling or mark the surface inconclusive.
Relevant terminal token family: `ABSORBED_BY_CEGIS_OR_SYNTHESIS`.

H.67. Clause for minimum per-family accuracy, PBE or CEGIS, and WGD-0 Toy Grammar.
Question: could PBE, CEGIS, SyGuS, ILP, relational rule search explain the minimum per-family accuracy result in `experiments/wgd0_b37_hidden_measurement.json`?
Required response: cite the exact metric, equal-information map, and all-in cost entry.
Failure response: lower the claim ceiling or mark the surface inconclusive.
Relevant terminal token family: `ABSORBED_BY_CEGIS_OR_SYNTHESIS`.

H.68. Clause for minimum per-family accuracy, PBE or CEGIS, and WGD-0 Hard Domain.
Question: could PBE, CEGIS, SyGuS, ILP, relational rule search explain the minimum per-family accuracy result in `experiments/wgd0_b38_hidden_measurement.json`?
Required response: cite the exact metric, equal-information map, and all-in cost entry.
Failure response: lower the claim ceiling or mark the surface inconclusive.
Relevant terminal token family: `ABSORBED_BY_CEGIS_OR_SYNTHESIS`.

H.69. Clause for minimum per-family accuracy, Library learning, and FrameSeed Boolean.
Question: could MDL library, DreamCoder-style library, e-graph composition explain the minimum per-family accuracy result in `experiments/frameseed0_b28_hidden_hfa.json`?
Required response: cite the exact metric, equal-information map, and all-in cost entry.
Failure response: lower the claim ceiling or mark the surface inconclusive.
Relevant terminal token family: `ABSORBED_BY_LIBRARY_LEARNING`.

H.70. Clause for minimum per-family accuracy, Library learning, and FrameSeed SHEETS-0.
Question: could MDL library, DreamCoder-style library, e-graph composition explain the minimum per-family accuracy result in `experiments/frameseed_sheets0_b31_hidden_hfa.json`?
Required response: cite the exact metric, equal-information map, and all-in cost entry.
Failure response: lower the claim ceiling or mark the surface inconclusive.
Relevant terminal token family: `ABSORBED_BY_LIBRARY_LEARNING`.

H.71. Clause for minimum per-family accuracy, Library learning, and WGD-0 Toy Grammar.
Question: could MDL library, DreamCoder-style library, e-graph composition explain the minimum per-family accuracy result in `experiments/wgd0_b37_hidden_measurement.json`?
Required response: cite the exact metric, equal-information map, and all-in cost entry.
Failure response: lower the claim ceiling or mark the surface inconclusive.
Relevant terminal token family: `ABSORBED_BY_LIBRARY_LEARNING`.

H.72. Clause for minimum per-family accuracy, Library learning, and WGD-0 Hard Domain.
Question: could MDL library, DreamCoder-style library, e-graph composition explain the minimum per-family accuracy result in `experiments/wgd0_b38_hidden_measurement.json`?
Required response: cite the exact metric, equal-information map, and all-in cost entry.
Failure response: lower the claim ceiling or mark the surface inconclusive.
Relevant terminal token family: `ABSORBED_BY_LIBRARY_LEARNING`.

H.73. Clause for minimum per-family accuracy, Schema or binding, and FrameSeed Boolean.
Question: could schema matcher, binding fingerprint, entity resolver explain the minimum per-family accuracy result in `experiments/frameseed0_b28_hidden_hfa.json`?
Required response: cite the exact metric, equal-information map, and all-in cost entry.
Failure response: lower the claim ceiling or mark the surface inconclusive.
Relevant terminal token family: `ABSORBED_BY_SCHEMA_OR_BINDING`.

H.74. Clause for minimum per-family accuracy, Schema or binding, and FrameSeed SHEETS-0.
Question: could schema matcher, binding fingerprint, entity resolver explain the minimum per-family accuracy result in `experiments/frameseed_sheets0_b31_hidden_hfa.json`?
Required response: cite the exact metric, equal-information map, and all-in cost entry.
Failure response: lower the claim ceiling or mark the surface inconclusive.
Relevant terminal token family: `ABSORBED_BY_SCHEMA_OR_BINDING`.

H.75. Clause for minimum per-family accuracy, Schema or binding, and WGD-0 Toy Grammar.
Question: could schema matcher, binding fingerprint, entity resolver explain the minimum per-family accuracy result in `experiments/wgd0_b37_hidden_measurement.json`?
Required response: cite the exact metric, equal-information map, and all-in cost entry.
Failure response: lower the claim ceiling or mark the surface inconclusive.
Relevant terminal token family: `ABSORBED_BY_SCHEMA_OR_BINDING`.

H.76. Clause for minimum per-family accuracy, Schema or binding, and WGD-0 Hard Domain.
Question: could schema matcher, binding fingerprint, entity resolver explain the minimum per-family accuracy result in `experiments/wgd0_b38_hidden_measurement.json`?
Required response: cite the exact metric, equal-information map, and all-in cost entry.
Failure response: lower the claim ceiling or mark the surface inconclusive.
Relevant terminal token family: `ABSORBED_BY_SCHEMA_OR_BINDING`.

H.77. Clause for minimum per-family accuracy, Domain tool, and FrameSeed Boolean.
Question: could SQL, spreadsheets, SAT, SMT, ILP, ASP, CSP explain the minimum per-family accuracy result in `experiments/frameseed0_b28_hidden_hfa.json`?
Required response: cite the exact metric, equal-information map, and all-in cost entry.
Failure response: lower the claim ceiling or mark the surface inconclusive.
Relevant terminal token family: `ABSORBED_BY_DOMAIN_TOOL`.

H.78. Clause for minimum per-family accuracy, Domain tool, and FrameSeed SHEETS-0.
Question: could SQL, spreadsheets, SAT, SMT, ILP, ASP, CSP explain the minimum per-family accuracy result in `experiments/frameseed_sheets0_b31_hidden_hfa.json`?
Required response: cite the exact metric, equal-information map, and all-in cost entry.
Failure response: lower the claim ceiling or mark the surface inconclusive.
Relevant terminal token family: `ABSORBED_BY_DOMAIN_TOOL`.

H.79. Clause for minimum per-family accuracy, Domain tool, and WGD-0 Toy Grammar.
Question: could SQL, spreadsheets, SAT, SMT, ILP, ASP, CSP explain the minimum per-family accuracy result in `experiments/wgd0_b37_hidden_measurement.json`?
Required response: cite the exact metric, equal-information map, and all-in cost entry.
Failure response: lower the claim ceiling or mark the surface inconclusive.
Relevant terminal token family: `ABSORBED_BY_DOMAIN_TOOL`.

H.80. Clause for minimum per-family accuracy, Domain tool, and WGD-0 Hard Domain.
Question: could SQL, spreadsheets, SAT, SMT, ILP, ASP, CSP explain the minimum per-family accuracy result in `experiments/wgd0_b38_hidden_measurement.json`?
Required response: cite the exact metric, equal-information map, and all-in cost entry.
Failure response: lower the claim ceiling or mark the surface inconclusive.
Relevant terminal token family: `ABSORBED_BY_DOMAIN_TOOL`.

H.81. Clause for minimum per-family accuracy, Nuisance oracle, and FrameSeed Boolean.
Question: could relevant-feature oracle or invariant oracle explain the minimum per-family accuracy result in `experiments/frameseed0_b28_hidden_hfa.json`?
Required response: cite the exact metric, equal-information map, and all-in cost entry.
Failure response: lower the claim ceiling or mark the surface inconclusive.
Relevant terminal token family: `ABSORBED_BY_NUISANCE_ORACLE`.

H.82. Clause for minimum per-family accuracy, Nuisance oracle, and FrameSeed SHEETS-0.
Question: could relevant-feature oracle or invariant oracle explain the minimum per-family accuracy result in `experiments/frameseed_sheets0_b31_hidden_hfa.json`?
Required response: cite the exact metric, equal-information map, and all-in cost entry.
Failure response: lower the claim ceiling or mark the surface inconclusive.
Relevant terminal token family: `ABSORBED_BY_NUISANCE_ORACLE`.

H.83. Clause for minimum per-family accuracy, Nuisance oracle, and WGD-0 Toy Grammar.
Question: could relevant-feature oracle or invariant oracle explain the minimum per-family accuracy result in `experiments/wgd0_b37_hidden_measurement.json`?
Required response: cite the exact metric, equal-information map, and all-in cost entry.
Failure response: lower the claim ceiling or mark the surface inconclusive.
Relevant terminal token family: `ABSORBED_BY_NUISANCE_ORACLE`.


## End Note

The appendices stop on a complete hostile-review clause.
Future reviewers should extend the matrix only when a new claim surface, absorber rung, or case-study artifact is added.
