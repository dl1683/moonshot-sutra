# Q-Loop Batch 48: Terminal Adversarial Gate Review

**Date:** 2026-07-08  
**Role:** fresh adversarial reviewer with zero prior project context  
**Iterations:** I421-I434  
**Status:** terminal gate review of `research/methodology_paper.md` after B40 revision

Two invariants held fixed:

1. Swing for the home run: paradigm-shifting or nothing.
2. The loop only stops on a won-over adversary.

## Grounding

I read the required review surface first:

- `research/methodology_paper.md`
- `experiments/b40_positive_control_measurement.json`
- `experiments/frameseed0_b28_hidden_hfa.json`
- `experiments/frameseed_sheets0_b31_hidden_hfa.json`
- `experiments/wgd0_b37_hidden_measurement.json`
- `experiments/wgd0_b38_hidden_measurement.json`
- `code/b40_positive_control.py`
- `research/question_loop_batch47.md`
- `research/dual_loop_supervisor_checkin_37.md`

Then I checked the surrounding repo surface that bears directly on the claims: `README.md`, `research/STATUS.md`, `experiments/EXPERIMENTS.md`, `experiments/ledger.jsonl`, `research/question_loop_batch40.md`, `research/frameseed_milestone_report.md`, `research/dual_loop_supervisor_checkin_36.md`, `research/work_loop_batch40.md`, `code/frameseed_sheets0_measurement.py`, `code/frameseed_sheets0_harness.py`, `code/wgd0_measurement.py`, `code/wgd0_b38_hard_domain.py`, and the relevant harness tests.

A recursive raw filesystem walk also exposed ignored binary shards and temp/pytest directories, some unreadable. I treated those as outside the active scientific artifact surface; the active source/docs/artifact surface from `rg --files` was the review target.

## Executive Hostile Verdict

Reject as currently written.

The B40 revision fixed much of B47's prose-level overclaim. The paper is now roster-relative, self-audit-limited, cost-sensitive, and much less padded. That is real progress.

But the adversary is not won over because the revised paper now fails in a more concrete way: its evidence ledgers over-upgrade mode-scored or label-reused absorbers into `native executable` evidence, and its new positive control obtains `SIGNAL` by omitting the exact ordinary absorber that would solve the task. The paper's core idea survives. The paper's terminal gate does not.

The fatal defects are:

1. C2 SHEETS-0 contradicts the repo's own claim ceiling: the current paper says typed PBE/CEGIS/data-wrangling/library routes were `native executable`; `research/STATUS.md`, `experiments/EXPERIMENTS.md`, `experiments/ledger.jsonl`, and Q40 say B31 used capability-mode scoring and must not be claimed as native typed-baseline execution.
2. C3 WGD toy labels `schema_binding`, `pbe_cegis`, and `wgd_grammar` as separate native absorbers even though the runner constructs them with the same `infer_role_model()` function and different source strings.
3. B40's positive control is not a clean yes: the claimed learner is full target-class best-candidate search; the budgeted PBE absorber is the same search truncated to the first 128 candidates; the code deliberately moves the planted target out of that prefix; and the artifact admits full target-class PBE would absorb.
4. Under the cost convention used in B40, full target-class PBE would be about 14,816 bits versus the claimed system's 5,224 bits, a ratio of about 2.84x. That is inside the <=4x absorption boundary used elsewhere in the project.
5. The paper's own Section 3 says a missing dangerous native absorber is not signal or lowers the claim ceiling, but B40 emits `B40_POSITIVE_CONTROL_SIGNAL_ROSTER_RELATIVE` while recording `residual_risk_high=true` and `omitted_full_target_class_pbe_would_absorb=true`.

## I421: Attack The B40 Victory Frame

### Previous Position Under Attack

B40 presents the revision as a successful repair pass: roster-relative claim, positive control, self-audit limits, filled ledgers, cost robustness, related-work delta, and a domain onboarding recipe.

### Attack

This is checklist success, not adversarial success. The paper did address the B47 headings, but it did not consistently bind its own claims to the underlying artifacts. The new paper is much cleaner, but cleanliness makes the remaining contradictions sharper.

The paper's strongest move is to say: signal is always relative to a roster, status ledger, cost rule, hidden-open manifest, and residual risk. The problem is that the paper then misstates statuses in the ledgers and accepts a positive control whose residual risk is known to absorb.

### Verdict

```text
B40_REPAIR_IS_REAL_BUT_NOT_TERMINAL
```

## I422: Attack I421 For Being Too General - C2 Is A Direct Claim-Ceiling Violation

### Previous Position Under Attack

I421 says the revision remains broadly under-supported. That is too vague. The strongest rejection is a concrete contradiction in C2.

### Attack

The C2 ledger in `research/methodology_paper.md` says:

- `Typed CEGIS/PBE | native executable | typed CEGIS exact min HFA 1.0; PBE/PROSE min HFA 1.0`
- `Data wrangling/domain tools | native executable`
- `Library/active/nuisance | native executable`

But the repo's own current truth says the opposite. `research/STATUS.md` warns that B31 used capability-mode scoring for typed baselines, not native learned PBE, CEGIS, schema matching, or MDL-library execution. `experiments/EXPERIMENTS.md` says SHEETS is a granted-binding and typed pipeline-substrate absorption, not a public claim that every named typed prior-art baseline was natively executed. `experiments/ledger.jsonl` says: do not claim native typed-baseline execution unless a later artifact implements it.

The code confirms the warning. In `code/frameseed_sheets0_measurement.py`, `FULL_PIPELINE_SYSTEMS` names systems such as `l2_typed_cegis`, `pbe_prose`, `data_wrangling`, `typed_cegis_exact`, `typed_mdl_library`, and `library_learning`. The hidden loop then computes:

```text
mode = system_mode(system)
hit = mode_solves(mode, query.operation)
score = 1.0 if hit else 0.0
```

For `full_pipeline`, `mode_solves()` returns true. Binding-only HFA is also made 1.0 by incrementing `binding_only_hits` for every hidden query. That is a conservative packet-erasure/granted-substrate absorption. It is not native execution of independent typed PBE, CEGIS, data-wrangling, and MDL-library baselines.

### Verdict

```text
REJECT_UNTIL_C2_LEDGER_IS_RELABELED_CAPABILITY_MODE_OR_GRANTED_SUBSTRATE
```

## I423: Attack I422 For Localizing The Problem To SHEETS

### Previous Position Under Attack

I422 treats C2 as the decisive inconsistency. That is true but too local. The same status inflation appears in C3.

### Attack

`code/wgd0_measurement.py` declares `SYSTEMS = ("wgd_grammar", "schema_binding", "pbe_cegis", "majority_feedback")`. But all three non-majority systems are produced by the same function:

```text
"wgd_grammar": infer_role_model(...)
"schema_binding": infer_role_model(...)
"pbe_cegis": infer_role_model(...)
```

The difference is the `source` string and cost payload, not an independent PBE/CEGIS implementation. The paper's C3 ledger says `PBE/CEGIS | native executable | PBE/CEGIS HFA 1.0` and calls it an independent winning route. That overstates the artifact. A fair wording is: the WGD toy grammar is absorbed by a cheaper public-feedback role-inference model; the PBE/CEGIS label is a costed interpretation, not a separately implemented native synthesizer.

This still kills the WGD toy claim. It does not support the paper's stronger ledger claim that native PBE/CEGIS independently solved it.

### Verdict

```text
REJECT_UNTIL_C3_DISTINGUISHES_SHARED_ROLE_INFERENCE_FROM_NATIVE_PBE_CEGIS
```

## I424: Attack I423 For Treating The Problem As A Few Bad Labels

### Previous Position Under Attack

I423 says the problem is status-label inflation in C2 and C3. That is still too shallow. The deeper issue is that the paper's filled ledgers are not generated evidence ledgers; they are prose assertions.

### Attack

The paper's Section 3 makes status categories binding: `native_executable`, `proxy_absorber`, `capability_mode_scored`, `formal_lower_bound`, `non_native_omitted`, and `untested_roster_entry`. But the case ledgers do not point to a machine-readable status field per absorber. They summarize the artifact and choose a status. That choice is wrong in at least C2 and overstated in C3.

This is exactly the failure mode B47 called `NATIVE_ABSORBER_THEATER`: naming the status taxonomy is easier than proving each absorber has that status. B40 replaced a bloated appendix with compact tables, but compact tables can launder overclaim faster than bloated ones.

### Verdict

```text
FILLED_LEDGER_NOT_SUFFICIENT_UNLESS_STATUS_IS_ARTIFACT_BOUND
```

## I425: Attack I424 For Letting The Hard-Domain Case Do Too Much Work

### Previous Position Under Attack

I424 focuses on ledger evidence. That still lets C4 carry too much validation weight.

### Attack

C4 is a legitimate absorption, but not a strong methodology-validation case. In `code/wgd0_b38_hard_domain.py`, both `wgd_basis_grammar` and `constraint_solver_absorber` call `solve_with_basis(world, case)`. The public rule atlas lists each atomic rule's `before_state` and `after_state`, and the absorber uses GF(2) solve over that structure. This is a good ordinary explanation. It also means WGD and the constraint absorber are nearly the same executable solution family with different accounting wrappers.

The paper is careful not to overclaim B38 scale, and that improvement matters. Still, C4 shows that once the domain is linear algebra, linear algebra wins. It does not validate a general ladder beyond showing that a well-chosen domain-native solver can kill a generated grammar story.

### Verdict

```text
C4_IS_A_REAL_ABSORPTION_NOT_A_METHOD_VALIDATION_BREAKTHROUGH
```

## I426: Attack I425 For Missing The New Positive-Control Failure

### Previous Position Under Attack

I425 says C4 should be demoted as methodology validation. That is not the fatal issue. The fatal issue is the B40 positive control, because B40 was added specifically to answer B47's rejection-machine attack.

### Attack

B40 is presented as evidence that the ladder can say yes. It does not show that. It shows that the ladder can say yes when the winning ordinary explanation is placed on the claimed-system side and omitted from the absorber roster.

The positive-control artifact itself records:

```text
omitted_full_target_class_pbe_would_absorb = true
residual_risk_high = true
```

The methodology paper repeats that full target-class PBE over all 1320 candidates was deliberately omitted. But the token is still `B40_POSITIVE_CONTROL_SIGNAL_ROSTER_RELATIVE`, and the paper says this proves the ladder can emit a narrow `SIGNAL`.

A hostile reviewer will not accept that as a positive control. The missing absorber is not speculative. It is known, native, cheap enough to discuss directly, and exactly target-class aligned.

### Verdict

```text
B40_DOES_NOT_PROVE_THE_LADDER_CAN_SAY_YES_FAIRLY
```

## I427: Attack I426 For Being Too Abstract - The Claimed Learner Is The Omitted Absorber

### Previous Position Under Attack

I426 says B40 omitted a dangerous absorber. That is correct but not sharp enough.

### Attack

`code/b40_positive_control.py` makes the identity explicit. The claimed system is:

```text
claimed = learn_best_candidate(train, candidates)
```

The budgeted PBE probe is:

```text
budgeted_candidates = candidates[:BUDGETED_PBE_CANDIDATES]
budgeted = learn_best_candidate(train, budgeted_candidates)
```

So the claimed `interaction_learner` is full target-class best-candidate search over the same candidate class. The declared `budgeted_pbe_probe` is the same algorithm truncated to the first 128 candidates. The omitted full target-class PBE is not merely an omitted absorber. It is the claimed algorithm viewed as an ordinary absorber.

This makes the positive control circular. It does not show discovery surviving ordinary explanations. It calls the ordinary target-class search the claimed artifact and withholds the full version from the absorber side.

### Verdict

```text
B40_SIGNAL_IS_CIRCULAR_RELATIVE_TO_TARGET_CLASS_SEARCH
```

## I428: Attack I427 For Ignoring The Cost Rule

### Previous Position Under Attack

I427 says the claimed learner and omitted absorber are algorithmically identical. Suppose a defender says algorithm identity is acceptable because the point is roster-relative behavior. The cost rule still kills the token.

### Attack

B40's cost ledger charges the budgeted PBE probe:

```text
G = 1536
P_i = 224
E_i = 2496
candidate_attempt_bits = 128 * 8 = 1024
total = 5280
```

Under the same convention, full PBE over 1320 candidates is roughly:

```text
1536 + 224 + 2496 + (1320 * 8) = 14816 bits
```

The claimed system is 5224 bits. The estimated full-PBE ratio is:

```text
14816 / 5224 = 2.83614088820827
```

The WGD cases use <=4x all-in cost as an absorption boundary. Under that boundary, full target-class PBE would absorb B40. The paper cannot use cost robustness to rescue a signal when its own omitted absorber is inside the ordinary cost boundary.

### Verdict

```text
B40_FULL_PBE_WOULD_ABSORB_UNDER_THE_PROJECTS_OWN_COST_NORM
```

## I429: Attack I428 For Missing The Constructed-Failure Mechanism

### Previous Position Under Attack

I428 says full PBE would absorb by cost. That is strong, but it does not explain why the declared budgeted absorber fails.

### Attack

The code makes the budgeted PBE failure partly constructed. After choosing `target_index`, B40 does this:

```text
if target_index < BUDGETED_PBE_CANDIDATES:
    target_index += BUDGETED_PBE_CANDIDATES
```

Since `BUDGETED_PBE_CANDIDATES = 128`, the planted target is forced out of the first 128 candidates whenever it would otherwise fall inside the budgeted PBE prefix. Then the absorber roster includes only that first-prefix PBE probe.

That is not a neutral positive control. It is a designed roster boundary that keeps the omitted full ordinary explanation out of reach. The result may be a useful toy demonstration of residual-risk wording, but it is not evidence that a discovery claim survived its strongest relevant ordinary explanation.

### Verdict

```text
B40_BUDGETED_ABSORBER_FAILURE_IS_ENGINEERED_BY_PREFIX_EXCLUSION
```

## I430: Attack I429 For Letting Roster-Relative Wording Excuse Too Much

### Previous Position Under Attack

I429 says the positive control is engineered. A defender can answer: yes, and the paper says signal is only relative to the declared roster.

### Attack

If roster-relative means a researcher may omit a known native <=4x absorber and still emit `SIGNAL`, then the protocol is too easy to game. The paper's own Section 3 tries to prevent that by saying a missing dangerous native absorber is not signal, is inconclusive, or lowers the claim ceiling. B40 violates that rule.

This is not a philosophical nit. It is the central epistemic contract. The paper cannot both say:

```text
A result with a missing dangerous native absorber is not signal.
```

and also say:

```text
B40 is a SIGNAL while full target-class PBE would absorb and is high residual risk.
```

The fix is not wording. The token assignment is wrong under the paper's own standard.

### Verdict

```text
B40_SIGNAL_CONTRADICTS_THE_PAPERS_OWN_STOPPING_RULE
```

## I431: Attack I430 For Focusing Only On The Positive Control

### Previous Position Under Attack

I430 says the positive-control token is internally inconsistent. That is fatal, but it also reveals a repo-level discipline problem.

### Attack

The paper's Source Map includes B47, supervisor #37, the methodology template, the case artifacts, and B40 code. It does not include the repo files that directly warn against the paper's C2 wording: `research/STATUS.md`, `experiments/EXPERIMENTS.md`, `experiments/ledger.jsonl`, and `research/question_loop_batch40.md`.

This matters because the methodology claims that public claims must be bound to residual risks and status ledgers. The live repo already contains the status warning. The revised paper regressed past it by calling capability-mode SHEETS baselines native executable.

A terminal gate reviewer should not accept a paper about claim ceilings when the paper misses the current repo's own ceiling note.

### Verdict

```text
SOURCE_MAP_MISSES_THE_FILES_THAT_PREVENT_C2_OVERCLAIM
```

## I432: Attack I431 For Being Too Repo-Hygiene Focused

### Previous Position Under Attack

I431 says the paper failed to ingest repo warnings. That is true but still operational. The broader conference-review issue remains.

### Attack

Even after B40's narrowing, the paper is still trying to be a paradigm-shifting AI discovery methodology using self-authored synthetic examples and one invalid positive control. The self-audit section is honest, but acknowledging non-independence does not create independent validation. The title and abstract still address AI discovery claims broadly: rules, frames, grammars, strategies, circuits, and world models.

With C2/C3 status inflation and B40 invalidated, the evidence base becomes:

- several honest internal absorptions;
- one hard-domain linear algebra absorption;
- no fair positive control;
- no external claim reanalysis;
- no independent rerun;
- no third-party absorber selection.

That is enough for a strong protocol proposal. It is not enough for the terminal home-run claim that the adversary should be won over.

### Verdict

```text
SELF_AUDITED_SYNTHETIC_ABSORPTIONS_PLUS_INVALID_POSITIVE_CONTROL_DO_NOT_CLEAR_HOME_RUN_BAR
```

## I433: Attack I432 For Overrejecting The Core

### Previous Position Under Attack

I432 sounds like the whole paper should be dismissed. That is too harsh. The core is alive.

### Attack

The paper's central idea is strong after B40's rewrite:

```text
Discovery claims should be made relative to declared ordinary absorbers, equal information, all-in cost, hidden-open discipline, terminal tokens, and residual risk.
```

That is not trivial. The negative case studies do show unusual internal honesty. The revised paper also removed much of the old generated padding, demoted universal language, added cost sensitivity, and gave a useful onboarding recipe.

The adversarial rejection should be precise. The paper does not fail because the framework is bad. It fails because its own terminal evidence violates the framework:

1. status ledgers misclassify capability-mode or label-reused absorbers;
2. the positive control emits signal while omitting a known absorbing ordinary explanation that is effectively the claimed algorithm.

Fix those, and the paper could stand as a protocol proposal with internal negative case evidence. Without those fixes, it fails its own gate.

### Verdict

```text
CORE_METHOD_ALIVE_PAPER_REJECTED_FOR_EVIDENCE_DISCIPLINE_FAILURE
```

## I434: Terminal Synthesis

### Previous Position Under Attack

I433 says the core is alive but the paper fails. The final question is whether, after sustained attack, the adversary is won over anyway.

### Attack

No. I am not won over.

The paper improved enough that the old B47 rejection is no longer the right rejection. The current rejection is narrower and more damaging: the revised paper claims to enforce ledger truth, then violates ledger truth.

The most hostile conference-review sentence is now:

```text
This is a promising protocol for preventing discovery overclaims, but its own evidence table overclaims native absorber execution, and its positive control obtains signal by omitting the exact ordinary search procedure that would absorb the result.
```

That sentence is fatal because it attacks the paper by its own standard, not by an external taste preference.

### Final Token

```text
Q_LOOP_B48_ADVERSARY_NOT_WON_OVER_REJECT_UNTIL_LEDGER_TRUTH_AND_POSITIVE_CONTROL_ARE_FIXED
```

## Minimum Revision To Win Over This Adversary

1. Relabel C2 SHEETS-0 accurately. The valid claim is granted-binding / typed-pipeline substrate absorption with capability-mode typed baselines. Do not call typed PBE, typed CEGIS, data-wrangling, schema matching, or typed MDL library `native executable` unless new artifacts implement them.
2. Relabel C3 WGD toy accurately. Say cheaper public-feedback role inference absorbed WGD. Do not sell `pbe_cegis` as an independent native PBE/CEGIS implementation when it is produced by the same `infer_role_model()` function.
3. Add an artifact-bound absorber status table. Every row should cite whether its status comes from native code, capability-mode scoring, formal bound, proxy, or omission. The status should not be inferred only in prose.
4. Replace or demote B40. Either run full target-class PBE as an absorber and let it absorb, or design a positive control where the claimed system is not just full target-class search and no known <=4x native absorber is deliberately omitted. If full PBE remains omitted, the token should be inconclusive / residual-risk demonstration, not `SIGNAL`.
5. Add B40 tests or at least a deterministic verification script that recomputes the artifact and checks token, cost, and omitted-absorber accounting.
6. Update the paper's source map to include `research/STATUS.md`, `experiments/EXPERIMENTS.md`, `experiments/ledger.jsonl`, and `research/question_loop_batch40.md`, or otherwise prove those warnings have been superseded by new artifacts.
7. Keep the narrow submission framing: methodology proposal with internal case evidence, not independently validated field standard.

## Final Adversarial Position

The methodology is worth preserving. The paper is not terminally accepted.

The adversary is not rejecting because the paper is too cautious. The adversary is rejecting because the paper is not cautious enough where it matters most: absorber status truth and the positive-control token.