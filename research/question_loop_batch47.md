# Q-Loop Batch 47: Adversarial Review Of The Methodology Paper

**Date:** 2026-07-08  
**Role:** Question-Loop worker  
**Iterations:** I407-I420  
**Status:** hostile review of `research/methodology_paper.md`

Two invariants held fixed:

1. Swing for the home run: the paper must be paradigm-shifting for AI evaluation, not merely sensible.
2. The loop only stops on a won-over adversary.

## Grounding

Required context read:

- `research/methodology_paper.md`
- `research/dual_loop_supervisor_checkin_36.md`
- `research/question_loop_batch46.md`

Additional evidence checked because the paper rests on saved measurements:

- `experiments/frameseed0_b28_hidden_hfa.json`
- `experiments/frameseed_sheets0_b31_hidden_hfa.json`
- `experiments/wgd0_b37_hidden_measurement.json`
- `experiments/wgd0_b38_hidden_measurement.json`
- `code/wgd0_b38_hard_domain.py`

Important checkout update relative to B46: B46 correctly refused to claim a B38 result because the artifact was absent in that checkout. In the current checkout, `experiments/wgd0_b38_hidden_measurement.json` exists and reports `WGD_ABSORBED_BY_CONSTRAINT_DISCOVERY`.

## Executive Hostile Verdict

Reject as currently written.

The methodology is promising, but the paper does not yet clear the home-run bar. The most dangerous global weakness is that it sells an "adversarial measurement immune system" while its evidence is four internal, synthetic, CPU-only absorptions plus large generated appendices that mostly specify what a future reviewer should ask. A skeptical reviewer can dismiss the draft as a disciplined lab notebook, not a paradigm-shifting AI evaluation contribution.

The adversary is not won over.

## I407: Attack The Paper's Core Evidence Claim

### Previous Position Under Attack

B46 framed the methodology paper as the correct deliverable after repeated absorptions. The paper adopts that frame: the mechanisms died, therefore the methodology survived.

### Attack

That inference is too fast. Four negative internal case studies do not establish that the methodology works as a general AI evaluation method. They establish only that this project can construct toy domains, run baselines, and honestly report absorptions.

The paper's most dangerous sentence is the abstract-level claim that "Across four case studies in this project, the ladder prevented false positives." A hostile reviewer can say: prevented false positives relative to what? There is no external claim that would otherwise have been accepted, no blinded reviewer decision changed by the ladder, no comparison against a conventional evaluation process, and no positive control showing that the ladder does not merely crush every small domain it touches.

### Single Most Dangerous Weakness

The paper treats internal negative case studies as validation of the methodology, but it never validates the methodology against an independent evaluation failure.

### Dismissal Line

"You did not show a new evaluation paradigm. You showed that your own toy claims did not survive your own baselines."

### Required Fix

Add a methodology-validation section with at least one independent or comparative target: a reanalysis of a published discovery claim, a comparison against a conventional evaluation that would have overclaimed, a positive-control task where narrow signal survives, or an explicit retreat to protocol proposal rather than validated methodology.

### Verdict

```text
REJECT_UNTIL_METHOD_VALIDATION_IS_SEPARATED_FROM_INTERNAL_NEGATIVE_RESULTS
```

## I408: Attack I407 For Being Too Friendly To The Case Studies

### Previous Iteration Under Attack

I407 says the issue is missing independent validation. That is not hostile enough. Even before independence, the case-study set itself is structurally biased.

### Attack

The four case studies are all absorption stories. That makes the paper look calibrated only for saying no. The draft says a positive result is possible, but it provides no worked example of what the ladder would do when a claim survives. Section 14 lists conditions for a positive result, but there is no concrete positive-control artifact, no simulated signal, and no example of claim-ceiling wording after signal.

Without that, a reviewer can say the ladder is not an evaluation method. It is a rejection machine. The paper can insist that signal is possible, but the only empirical behavior demonstrated is absorption. A paradigm-shifting evaluation method must distinguish signal, absorption, void, trap, negative, and inconclusive in practice. This draft demonstrates only absorption.

### Single Most Dangerous Weakness

The paper has no positive control, so it cannot prove the ladder distinguishes real discovery from ordinary explanations rather than simply making every small discovery claim impossible.

### Dismissal Line

"This is a well-engineered no-machine. Where is the example showing it can say yes?"

### Required Fix

Add a deliberately constructed positive-control domain with declared absorbers, all-in ledger, hidden opening, a narrow signal token, component-erasure damage, and a modest claim ceiling. If no positive control exists, demote the paper from methodology validation to negative-results protocol proposal.

### Verdict

```text
REJECT_UNTIL_THE_LADDER_CAN_BE_SEEN_SAYING_YES_OR_THE_CLAIM_IS_DEMOTED
```

## I409: Attack I408 For Mistaking Positivity For The Core Problem

### Previous Iteration Under Attack

I408 demands a positive control. That is useful, but it is not the fatal weakness. A positive control would not save a ladder whose endpoint is under-defined.

### Attack

The paper repeatedly invokes the "strongest boring explanations," but it never makes "strongest" operational. The absorber roster is open-ended: representation priors, parser priors, teaching dimension, active learning, PBE, CEGIS, schema binding, library learning, domain tools, nuisance oracles, generator fingerprints, human substrate, post-hoc compression, and so on.

That openness is both the paper's power and its fatal vulnerability. If a claim survives the declared absorbers, a hostile reviewer can always say the real strongest absorber was omitted. If the paper says the ladder only covers declared absorbers, then the core claim "not credible until its strongest boring explanations have failed" is too strong. The draft has a rhetorical universal and a procedural finite checklist. Those are not the same thing.

### Single Most Dangerous Weakness

The ladder has no stopping rule for absorber completeness, so signal can always be challenged as "you missed the actual strongest boring explanation."

### Dismissal Line

"Your method cannot certify discovery because its critical quantifier, strongest, is unknowable."

### Required Fix

Replace the universal standard with an auditable roster standard:

```text
An AI discovery claim is credible only relative to a predeclared absorber roster, the domain rationale for that roster, and the residual untested absorber risk.
```

Add a required absorber-completeness argument for each domain: why included absorbers are native or relevant, why omitted absorbers are non-native or future work, and how residual risk lowers the claim ceiling.

### Verdict

```text
REJECT_UNTIL_ABSORBER_COMPLETENESS_HAS_A_STOPPING_RULE
```

## I410: Attack I409 For Ignoring The Cost Ledger's Fragility

### Previous Iteration Under Attack

I409 says the main issue is absorber completeness. That is too abstract. Even a complete absorber roster would fail if the all-in cost comparison is not defensible.

### Attack

The paper leans hard on all-in accounting, but the bit ledger is not operational enough for external review. The cost categories are named, but the conversion from implementation choices to bits can be disputed. The B38 artifact exposes the problem: WGD costs `704640` total bits and the constraint absorber costs `681664`, giving a 0.967 cost ratio. But both are dominated by huge `G` terms. A skeptical reviewer can ask whether those `G` charges are measuring discovery difficulty, source-code size, human-designed substrate, duplicated implementation scaffolding, or arbitrary serialization overhead.

If the cost ledger can swing results by representation decisions, then "absorbed at 96.7 percent of WGD cost" is not a robust scientific result. It is a property of a local accounting convention.

### Single Most Dangerous Weakness

The all-in cost metric is central to the claim but not standardized, sensitivity-tested, or externally reproducible.

### Dismissal Line

"Your conclusion depends on homemade bit accounting. I cannot tell whether the absorber is cheaper, equally expensive, or merely encoded differently."

### Required Fix

Add a cost robustness section: exact serialization rules, sensitivity bands under alternative fair encodings, separated shared substrate and system-specific structure, runtime and query-count side metrics, and terminal-token stability under plausible cost perturbations.

### Verdict

```text
REJECT_UNTIL_ALL_IN_COST_IS_ROBUST_AND_RECOMPUTABLE
```

## I411: Attack I410 For Letting Native Absorber Theater Survive

### Previous Iteration Under Attack

I410 makes the cost ledger the fatal issue. That is not enough. Perfect cost accounting still cannot rescue theatrical absorbers.

### Attack

Section 12 names `native_executable`, `proxy_absorber`, `capability_mode_scored`, `formal_lower_bound`, and `untested_roster_entry`. Naming those categories is not the same as proving the paper's absorbers are native. The appendices repeatedly say "run or justify" and "inconclusive if untested," but the case summaries often report only the winning absorber, not a full status ledger for the other dangerous rungs.

B38 is the clearest example. The paper says the hard domain was absorbed by GF(2) constraint discovery. Good. But the hostile question is broader: did the hard domain also face active learning, PBE/CEGIS over the true hypothesis language, MDL/library learning, generator-family classification, representation-prior probes, and substrate accounting in native executable form? Appendix H asks these questions, but the body does not answer them.

### Single Most Dangerous Weakness

The paper addresses `NATIVE_ABSORBER_THEATER` conceptually but does not prove, case by case, that its own absorber roster avoided theater.

### Dismissal Line

"You have an excellent term for bad baselines, but your evidence table still leaves most absorber statuses implicit."

### Required Fix

For each case study, add a compact absorber-status table with rung, status, executable artifact, equal-information note, cost note, result, and claim-ceiling effect. Any untested dangerous native absorber must lower the paper's case-study claim.

### Verdict

```text
REJECT_UNTIL_NATIVE_ABSORBER_STATUS_IS_EVIDENCE_NOT_TAXONOMY
```

## I412: Attack I411 For Staying Inside The Synthetic Sandbox

### Previous Iteration Under Attack

I411 focuses on whether the internal absorbers are native. That still grants too much. Even if every internal absorber is native, the paper's field-level ambition remains unsupported.

### Attack

The paper's title and abstract target AI discovery claims broadly: rules, frames, grammars, strategies, circuits, and world models. The evidence comes from four synthetic CPU-only cases. None involves a modern LLM, a real scientific discovery task, an embodied environment, a theorem-proving benchmark, a mechanistic interpretability claim, a robotics task, or a data-analysis workflow where "discovery" is actually contested.

Section 15 admits that the case studies do not prove discovery cannot separate at larger scale. But the paper still wants the contribution to be an "adversarial measurement immune system" for AI discovery. A hostile reviewer will say the methodology may be reasonable, but the demonstrated evidence is too narrow for the promised surface.

### Single Most Dangerous Weakness

The framework is claimed for AI discovery generally but demonstrated only on four internal synthetic CPU domains.

### Dismissal Line

"This is a toy-domain evaluation discipline with a general title."

### Required Fix

Either narrow the paper to synthetic discovery benchmarks or add at least one external, non-synthetic or semi-real case study where the claimed system is not authored by the same project, the task has a real stakeholder interpretation, the absorber roster is chosen with domain expertise, and the result changes the allowed public claim.

### Verdict

```text
REJECT_UNTIL_SCOPE_MATCHES_EVIDENCE
```

## I413: Attack I412 For Missing The Scale Illusion Inside The Case Studies

### Previous Iteration Under Attack

I412 says the problem is external generalization. That is not the sharpest attack. The case studies may not even carry their internal scale rhetoric cleanly.

### Attack

The B38 hard domain is presented as a serious escalation because it has 64 rules and a `2^64` candidate space. But the measurement artifact reports only 8 worlds, 256 cases, and 1536 scored predictions. Each case kind has 64 examples. The paper says the domain defeated brute enumeration, but the actual adversarial reader sees a tiny evaluation grid wrapped in large combinatorial language.

This creates a scale-illusion risk. The candidate space is huge because the generator defines it that way, but the empirical evaluation remains small. A reviewer can say the paper should not lean on `2^64` unless it also proves that the hidden cases sample that space in a way that matters.

### Single Most Dangerous Weakness

The hard-domain case uses large theoretical search-space numbers to imply seriousness while the actual hidden measurement is small.

### Dismissal Line

"You did not evaluate a hard domain; you evaluated 256 generated cases from a hard-sounding space."

### Required Fix

Add a sampling-validity section for B38: explain why 8 worlds and 256 cases are enough, report confidence intervals or exact coverage arguments, separate candidate-space hardness from evaluation-set strength, include larger-world stress tests if CPU budget permits, and state that the result defeats the enumerators only on this measured slice.

### Verdict

```text
REJECT_UNTIL_COMBINATORIAL_HARDNESS_IS_NOT_USED_AS_EVALUATION_SCALE_THEATER
```

## I414: Attack I413 For Ignoring The "Isn't This Just Good Practice?" Problem

### Previous Iteration Under Attack

I413 attacks scale rhetoric. That is valid but still too local. The paper can fix B38 wording and still fail as a contribution.

### Attack

The paper does not position itself against existing evaluation practice strongly enough. Precommitting tokens resembles preregistration. Equal-information baselines resemble fair baseline design. Component erasure resembles ablation. Teaching dimension, MDL, CEGIS, active learning, constraint solving, and schema matching are existing ideas. Claim ceilings resemble cautious statistical reporting.

The paper's contribution may be a useful packaging of known practices, but the home-run invariant demands paradigm-shifting AI evaluation. A hostile reviewer will ask what is technically new beyond insisting that researchers run strong baselines and report negative results honestly.

### Single Most Dangerous Weakness

The paper risks being dismissed as "good experimental hygiene with new terminology" rather than a methodology contribution.

### Dismissal Line

"I agree with the advice, but I do not see the new method."

### Required Fix

Add a related-work and delta section that explicitly distinguishes the ladder from preregistration, hidden-test benchmark discipline, ablations, causal intervention tests, MDL evaluation, program-synthesis methodology, no-free-lunch baseline arguments, and red-team evaluation. State the novel synthesis as terminal-token first refusal binding absorber execution, equal-information affordances, all-in cost, hidden-open discipline, causal artifact tests, and claim ceilings into one decision procedure.

### Verdict

```text
REJECT_UNTIL_THE_NOVEL_DELTA_OVER_EXISTING_EVAL_PRACTICE_IS_EXPLICIT
```

## I415: Attack I414 For Letting The Paper Be Self-Referential

### Previous Iteration Under Attack

I414 says the paper needs a related-work delta. That is necessary, but it still assumes the paper can argue its contribution from internal reasoning. The deeper problem is self-reference.

### Attack

The project builds the claims, builds the domains, builds the absorbers, assigns the tokens, and then declares that the methodology survived because it killed the claims. That may be honest, but it is not independent. The methodology has not been stress-tested by someone trying to use it against the authors, by a blinded evaluator, or by a case where the authors wanted the original claim to survive after public scrutiny.

This matters because the whole paper is about preventing self-deception. A methodology for preventing self-deception needs evidence that it works under incentives, not only evidence that its authors chose to be honest.

### Single Most Dangerous Weakness

The paper's central anti-self-deception claim is supported by a self-authored, self-adjudicated project history.

### Dismissal Line

"You built an honesty machine and then graded your own honesty."

### Required Fix

Add an independence plan or result: third-party rerun of one case study from the manifest, blinded absorber selection by someone outside the work loop, external adversarial review before hidden opening, deterministic public reproduction of token assignment, or an explicit statement that the current paper is self-audited and claim-limited.

### Verdict

```text
REJECT_UNTIL_SELF_AUDIT_LIMITS_ARE_CONFRONTED
```

## I416: Attack I415 For Missing The Draft-Quality Failure

### Previous Iteration Under Attack

I415 attacks self-audit. That is high-level. A reviewer may reject sooner for a simpler reason: the paper reads padded.

### Attack

The draft is 3205 lines, but the argument is mostly in the first 350. Appendix F is a templated twenty-iteration draft log where every entry repeats the same action, adversarial question, revision rule, and status. Appendix G is a generated surface-by-rung matrix with repeated "run or justify" clauses. Appendix H stops after hostile-review clause H.83, then says the appendices stop on a complete hostile-review clause.

This is dangerous because a methodology paper must project discipline. Generated-looking appendices make the paper look like it is substituting volume for evidence. The appendices may be useful internally, but a hostile reviewer will call them filler.

### Single Most Dangerous Weakness

The appendices damage credibility by looking generated, repetitive, and under-integrated with the actual case-study evidence.

### Dismissal Line

"The paper is 10 percent argument and 90 percent boilerplate checklist."

### Required Fix

Cut or restructure the appendices: remove Appendix F or replace it with a short provenance note, compress Appendix G into a one-page matrix, convert Appendix H into filled hostile-review examples, move only the executable checklist into the main method, and keep raw matrices in supplemental protocol files if needed.

### Verdict

```text
REJECT_UNTIL_APPENDICES_ARE_EVIDENCE_OR_CHECKLISTS_NOT_PADDING
```

## I417: Attack I416 For Treating The Appendices As Mere Style

### Previous Iteration Under Attack

I416 says the appendices are filler. That undersells the problem. The appendices are not just aesthetically bad; they reveal a missing evidence product.

### Attack

Appendix G lists surfaces and rungs. Appendix H asks case-by-case hostile questions. Those are exactly the tables the paper needs to answer, not merely list. The current version repeatedly says a failure to cite exact metrics, equal-information maps, and all-in cost entries should lower the claim ceiling. But the paper itself does not provide filled answers for most of those clauses.

That means the paper is vulnerable to its own audit standard. It says untested surfaces should become inconclusive, but the case-study sections do not show which surfaces were tested, which were proxy-only, and which remain untested.

### Single Most Dangerous Weakness

The paper's own hostile-review apparatus implies many missing case-by-rung evidence entries, which would force inconclusive claim ceilings unless filled.

### Dismissal Line

"Your appendix asks the right questions, but your paper does not answer them."

### Required Fix

Replace the generated appendices with completed evidence ledgers:

| Case | Surface | Rung | Executed? | Native? | Metric | Cost | Token effect |
|---|---|---|---|---|---|---|---|

The table can be compact, but it must show actual statuses. Any "not run" entry must explicitly lower the claim ceiling.

### Verdict

```text
REJECT_UNTIL_THE_PAPER_PASSES_ITS_OWN_APPENDIX_TESTS
```

## I418: Attack I417 For Missing The "So What Do I Do Monday?" Problem

### Previous Iteration Under Attack

I417 says the evidence matrix must be filled. Even a filled matrix may not make the paper useful.

### Attack

The paper tells researchers to make ordinary explanations dangerous, but it does not give a concrete workflow for choosing absorbers in a new domain. The ladder is described as domain-dependent, which is true, but the paper does not provide enough decision rules to prevent arbitrary cherry-picking.

A practitioner asks: I have a model claiming a world model in a game, a theorem prover claiming lemma discovery, or an LLM claiming scientific hypothesis generation. Which absorbers are mandatory? Which are optional? How do I decide when a proxy is acceptable? What budget do I allocate? What makes a domain tool native? When is an absorber too expensive to require? The paper gestures at these questions but does not operationalize them.

### Single Most Dangerous Weakness

The methodology is not yet actionable for a new external evaluator without substantial author judgment.

### Dismissal Line

"This is a philosophy of baselines, not a protocol I can reliably apply."

### Required Fix

Add a domain onboarding recipe:

1. define the discovery claim surface;
2. classify the domain representation and substrate;
3. choose mandatory rungs by claim type;
4. justify omitted rungs;
5. predeclare cost and query budgets;
6. assign native/proxy/capability status;
7. run hidden-open protocol;
8. emit token and claim ceiling.

Include three worked mini-examples outside the four case studies.

### Verdict

```text
REJECT_UNTIL_THE_LADDER_IS_A_REPRODUCIBLE_WORKFLOW_NOT_AUTHOR_JUDGMENT
```

## I419: Attack I418 For Understating The Paradigm-Shift Burden

### Previous Iteration Under Attack

I418 asks for more actionable workflow. That would make the paper useful, but useful is not enough. The invariant says home run.

### Attack

For the paper to be paradigm-shifting, it must do more than tell researchers how to be rigorous. It must show that a meaningful class of AI discovery claims changes status under this methodology. The current draft does not confront a live claim that the field cares about. It does not reclassify an accepted benchmark result. It does not expose a hidden flaw in a public "AI discovered X" story. It does not show a community-facing result that would make evaluators change their default review checklist.

The paper has a strong slogan: hidden accuracy is not enough. But the field already knows that in many forms. The paper needs a demonstration with stakes.

### Single Most Dangerous Weakness

The paper does not yet demonstrate field-level consequence; it only demonstrates internal project discipline.

### Dismissal Line

"Nothing in my reviewing practice changes because all your examples are your own synthetic failures."

### Required Fix

Add one field-facing case: re-evaluate an existing AI discovery benchmark or public claim, show the ordinary explanation that would have been missed, produce a before/after claim ceiling, and make the artifact reproducible enough that reviewers can inspect the token assignment. If this cannot be done before submission, narrow the pitch to a proposal and remove paradigm-shift language.

### Verdict

```text
REJECT_UNTIL_THE_METHOD_CHANGES_THE_STATUS_OF_A_CLAIM_WITH_EXTERNAL_STAKES
```

## I420: Attack I419 For Letting The Main Claim Remain Over-Absolute

### Previous Iteration Under Attack

I419 demands an external-stakes case. That would help, but even a public case would not save the current core claim if the claim ceiling remains over-absolute.

### Attack

The paper's first-page core claim says an AI discovery claim is "not credible until its strongest boring explanations have been made executable and have failed under equal information and all-in cost." This sounds decisive and universal. But the paper's own limitations admit that the ladder cannot enumerate every boring explanation and cannot make bit accounting perfectly objective.

That mismatch is the final fatal weakness. The paper is strongest when it says credibility is conditional, claim ceilings are bounded, and missing absorbers lower the result to inconclusive. It is weakest when it speaks as if "the strongest boring explanations" can be known and exhausted. A hostile reviewer can attack the whole paper as overclaiming its epistemic authority while accusing others of overclaiming.

### Single Most Dangerous Weakness

The paper violates its own claim-ceiling discipline by making an absolute credibility claim that its finite, domain-dependent ladder cannot justify.

### Dismissal Line

"Your paper is about not overclaiming, and its core claim overclaims."

### Required Fix

Rewrite the core claim:

```text
An AI discovery claim is credible only relative to a predeclared absorber roster, equal-information affordance map, all-in cost rule, hidden-open manifest, and explicit residual-risk statement.
```

Rewrite the contribution:

```text
The contribution is not proof that all ordinary explanations have failed. It is a protocol for making the most relevant ordinary explanations executable, documenting which ones were actually tested, and bounding the public claim to that evidence.
```

Then update the abstract, conclusion, Section 14, limitations, and Appendix D so they all enforce this bounded version.

### Verdict

```text
REJECT_UNTIL_THE_PAPER_OBEYS_ITS_OWN_CLAIM_CEILING
```

## Final Adversarial Synthesis

The adversary is not won over.

The paper has a real core:

```text
Do not accept AI discovery claims until ordinary explanations have executable first refusal under equal information, all-in accounting, hidden-open discipline, causal artifact tests, and bounded claim ceilings.
```

But the current draft is not yet the paradigm-shifting version of that core. The most dangerous weaknesses are:

1. internal absorptions are treated as methodology validation;
2. no positive control shows the ladder can say yes;
3. "strongest boring explanations" has no stopping rule;
4. all-in bit accounting is not robustness-tested;
5. `NATIVE_ABSORBER_THEATER` is named more convincingly than it is case-audited;
6. four synthetic CPU-only cases cannot support the broad AI-discovery framing;
7. B38's `2^64` rhetoric risks hiding a small 8-world, 256-case measurement;
8. the contribution is not clearly distinguished from existing good practice;
9. the project self-audits its own anti-self-deception method;
10. appendices look generated and under-integrated;
11. the appendix questions are not converted into filled evidence ledgers;
12. the workflow is not turnkey for external evaluators;
13. there is no field-facing case with external stakes;
14. the core claim overstates what a finite absorber roster can prove.

## Minimum Revision To Win Over This Adversary

The paper can become strong if it narrows and sharpens:

- Demote the universal credibility claim to a roster-relative claim.
- Add a filled absorber-status ledger for every case.
- Add cost sensitivity analysis.
- Cut or compress generated appendices.
- Add either a positive control or clearly state that none exists.
- Add one external or semi-real case, or narrow the title and scope.
- Add a related-work delta explaining why terminal-token first refusal is more than good practice.

## Final Token

```text
Q_LOOP_B47_ADVERSARY_NOT_WON_OVER_REJECT_UNTIL_CLAIM_CEILING_AND_EVIDENCE_GAPS_ARE_FIXED
```
