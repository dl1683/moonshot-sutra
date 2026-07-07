# Q-Loop Batch 40: Fresh-Eyes Adversarial Review of FrameSeed Arc

**Date:** 2026-07-07  
**Role:** Question-Loop worker  
**Iterations:** I309-I322  
**Status:** fresh-eyes milestone adversarial review after Boolean and SHEETS hidden measurements.

Two invariants held fixed:

1. Swing for the home run.
2. The loop only stops on a won-over adversary.

## Grounding

Read the requested FrameSeed arc in the current checkout: `VISION.md`, supervisor check-ins 24-31, Q-loop batches 32-39, W-loop batches 27, 28, 30, 31, both specs, both harnesses, both measurement runners, and both hidden HFA JSON files.

Checkout discrepancy:

```text
research/work_loop_batch29.md is absent.
```

Supervisor #29 and `research/frameseed_sheets_0_spec.md` identify W-Loop B29 as the SHEETS-0 spec, so this review treats the spec as the W29 substance. The missing separate W29 work log remains a provenance gap.

B39 discrepancy:

```text
Q-Loop B39 reported no SHEETS hidden measurement in its checkout.
The current checkout now contains work_loop_batch31.md and
experiments/frameseed_sheets0_b31_hidden_hfa.json.
```

Therefore B39 is historically useful but stale for the current measurement state.

## Executive Verdict

```text
FRAMESEED AS CURRENTLY CONCEIVED IS ABSORBED.
```

The adversary is won over on one point: there is no FrameSeed signal in the current arc. The adversary is not won over on a stronger point: the SHEETS-0 hidden measurement is not a full native-baseline benchmark. It is a conservative packet-erasure / schema-binding absorption demonstration whose scorer gives systems success by declared capability mode, not by executing independent learned programs against hidden outputs.

That distinction does not rescue FrameSeed. It lowers the public methodology claim from "native typed baselines rigorously solved the benchmark" to "when task bindings and typed pipeline substrate are granted, the packet is unnecessary."

## I309: Artifact Provenance Is Better Than Usual, But Not Clean Enough

### Attack

The arc is mostly reconstructible, but not perfectly bound. `work_loop_batch29.md` is missing; Q39's measurement boundary is stale relative to B31; and the measurement runners are scientifically decisive even though the user's explicit code list named only the harnesses.

### Verdict

```text
PROVENANCE IS GOOD ENOUGH FOR INTERNAL DIRECTION-KILLING, NOT PUBLIC-GRADE REPRODUCIBILITY.
```

### Kill Record

Any public report must explain the missing W29 work log and the B39/B31 checkout mismatch before claiming a fully sealed review chain.

## I310: The Boolean Absorption Is Honest And Strong

### Attack

B28 is not close. The JSON reports `FRAMESEED_T3_ABSORBED_BY_TEACHING_DIMENSION`, L3 min/mean HFA 1.0/1.0, every Boolean baseline min HFA 1.0, 15,728,640 hidden queries scored per system, and zero audit failures.

The code explains why: `make_public_transcript()` enumerates base rows and every single-slot edit; `support_slots()` selects the two slots whose edits change labels; most named systems then use the same exact two-slot program extraction or an oracle equivalent. That is fair for a kill test.

### Harder Attack

The labels `l0_rotenn`, `l1_active`, `l2_cegis`, and `library_learning` overstate distinct implementation. They are mostly wrappers around `exact_program(transcript, ...)`. Fine for an absorber, not fine as a claim of separate learner benchmarking.

### Verdict

```text
BOOLEAN FRAMESEED-0 IS CLEANLY ABSORBED. DO NOT REVIVE IT.
```

## I311: Boolean Did Not Actually Test T3-R Noncontainment

### Attack

The B28 token evidence says `representation_noncontainment_passed: true`, but the run never needed a strong noncontainment certificate. Lower-rung finite teaching/search solved first. The Boolean result kills the Boolean domain; it does not prove representation-changing packets are impossible in the abstract.

### Verdict

```text
B28 BYPASSES T3-R; IT DOES NOT FALSIFY T3-R DIRECTLY.
```

### Kill Record

Do not cite B28 as evidence that representation-changing packets cannot exist. Cite it as evidence that this supplied-frame Boolean packet is absorbed by finite teaching/search.
## I312: The SHEETS Hidden Measurement Is Mode-Scored, Not Executed

### Attack

This is the largest fresh-eyes finding. In `code/frameseed_sheets0_measurement.py`, the hidden loop computes `correct_output(query)`, but system success is assigned by declared mode:

```text
mode = system_mode(system)
hit = mode_solves(mode, query.operation)
score = 1.0 if hit else 0.0
```

`mode_solves("full_pipeline", operation)` always returns true. The full-pipeline set includes `l3_full`, `l2_typed_cegis`, `pbe_prose`, `data_wrangling`, `typed_cegis_exact`, `typed_cegis_beam`, `typed_mdl_library`, `library_learning`, `operation_verifier_search`, `goal_conditioned_cegis`, `active_goal_disambiguation`, `obligation_template_library`, and `nuisance_oracle`.

The measurement does not ask these systems to infer schema bindings, synthesize programs, execute candidates, or compare produced outputs to `correct_output`. It says systems with full-pipeline mode solve every operation.

The binding-only ablation is even more direct:

```text
binding_only_total += 1
binding_only_hits += 1
```

So binding-only HFA is 1.0 by construction.

### Verdict

```text
SHEETS-0 IS ABSORBED AS A FRAMESEED CLAIM, BUT THE MEASUREMENT IS NOT PUBLICATION-GRADE BASELINE EXECUTION.
```

### Kill Record

Do not say typed PBE, CEGIS, and MDL-library baselines were implemented and empirically solved the benchmark unless a later artifact implements native baseline execution.

## I313: The SHEETS Absorption Is Honest, But Its Name Is Too Clean

### Attack

The token says `FRAMESEED_SHEETS0_ABSORBED_BY_SCHEMA_BINDING`. The hidden data supports a conservative version of that: binding-only HFA is 1.0, packet-erasure drop is 0.0, and schema-binding/PBE/data-wrangling/typed-CEGIS/library absorptions are true.

But B31 proves only this:

```text
If exact charged bindings plus public typed operators are granted, packet frames are unnecessary.
```

It does not prove this:

```text
A schema-binding baseline discovered the bindings cheaply from public evidence.
```

The more precise diagnosis is `ABSORBED_BY_GRANTED_BINDING_AND_PIPELINE_SUBSTRATE`. The existing schema-binding token is acceptable as a conservative internal token because packet erasure completely failed, but the evidence claim must be lowered.

### Verdict

```text
ACCEPT SCHEMA-BINDING ABSORPTION; DO NOT OVERCLAIM WHAT WAS EXECUTED.
```

## I314: The Specs Hardened Faster Than The Executables

### Attack

The specs are much stronger than the code. They name AFTD, representation noncontainment, domain-specific absorbers, parser/human ledgers, typed leakage, local repair, operation/obligation semantics, and claim ceilings.

The executables instantiate only a subset:

- Boolean `aftd_passed` is set false in token evidence.
- SHEETS `aftd_all_in_passed` is set false.
- SHEETS `composition_gate_passed` is set false.
- Representation noncontainment is a token/audit field, not a bounded reachability proof.
- SHEETS native baselines are roster names and mode classes, not solvers.
- Leakage audits are mostly marginal MI, not predictive role/binding classifiers.

### Verdict

```text
THE METHODOLOGY IS INTELLECTUALLY RIGOROUS, BUT EXECUTABLE RIGOR LAGS SPEC RIGOR.
```

### Kill Record

Future arcs need each terminal gate to have an executable witness or an explicit "untested" label before hidden opening.

## I315: The Supervisor Sometimes Over-Trusted The Work Loop

### Attack

Supervisor #27 says the B27 harness proves its integrity. Q36 then reads the actual harness and says it passes its own audit but fails the stricter B35 evidence gate. Supervisor #30 says typed harness and baselines are built; W30 itself says actual hidden performance, domain baselines, and hidden scoring are not implemented. B31 then implements a hidden runner, but with capability-mode scoring.

The dual-loop worked because the Q-loop kept catching overclaim. The supervisor summaries sometimes rounded partial executable surfaces into completed rigor.

### Verdict

```text
THE LOOP IS HONEST ENOUGH TO KILL FRAMESEED, BUT NOT DISCIPLINED ENOUGH TO MAKE ITS OWN SUPERVISOR SUMMARIES EVIDENCE-BOUND.
```

### Kill Record

Future check-ins should distinguish declared baseline roster, proxy absorber, native executable absorber, and formal proof/lower bound as separate states.
## I316: The Home-Run Question Became A Benchmark Question

### Attack

The Vision asks what structure makes intelligence cheap. FrameSeed became a benchmark question: can compact packets beat optimal teaching/synthesis baselines in synthetic Boolean and spreadsheet worlds?

That was a reasonable filter. After two absorptions, continuing to design domains would be benchmark grinding. The evidence says the packet does not create the structure; the public transcript, schema, operation grammar, typed substrate, and task bindings do.

### Verdict

```text
THE CURRENT ARC HAS EXHAUSTED FRAME TRANSMISSION AS THE MAIN STORY.
```

### Kill Record

Do not spend the last batch on another FrameSeed domain. The next step should be milestone report and direction assessment.

## I317: The Deep Failure Is Supplied Geometry

### Attack

PCCP-H died because supplied frames were enumerable. FrameSeed repeats that pattern with teaching:

```text
Boolean: the public intervention table supplies the geometry.
SHEETS: the public typed operation grammar, exact bindings, and pipeline substrate supply the geometry.
```

The packet can be compact because the experiment designer already carved the world into the right primitives.

### Stronger Interpretation

Whenever the benchmark designer supplies the ontology, intervention grammar, operator grammar, verifier semantics, and binding schema, the remaining packet is usually a teaching set or program sketch.

### Verdict

```text
THE LIVE PROBLEM IS FRAME CREATION / GEOMETRY DISCOVERY, NOT PACKET DELIVERY.
```

## I318: Binding Discovery Is A Path Forward, But Not FrameSeed

### Attack

B31 shows exact bindings are decisive. That points to a live problem:

```text
Can a cheap system discover the right schema/entity/unit/constraint/action bindings under ambiguity, with calibrated abstention and local repair?
```

That is useful and close to ordinary-user automation. But it is not the current FrameSeed claim. It is a schema/goal/binding discovery problem with active queries, verifiers, and uncertainty. A packet might be one component, not the central explanation.

### Required Shape

A real binding-discovery direction would need hidden schemas with no granted bindings, predictive binding baselines, active disambiguation budgets, abstention scoring, repair after wrong binding, native SQL/PBE/entity/schema solvers, and cost per resolved binding or safe action.

### Verdict

```text
BINDING DISCOVERY IS A PATH FORWARD; FRAMESEED IS NOT THE RIGHT NAME FOR IT.
```

### Kill Record

Do not rescue FrameSeed by saying "the frame is binding discovery" unless the packet stops receiving the binding and the system actually discovers it under fair native baselines.

## I319: Self-Discovered Transformation Grammars Are The Cleanest Continuation

### Attack

Q33 and Q37 already named the reframe: self-discovered transformation grammars in typed practical domains. Fresh-eyes review agrees.

The hard object is not "teach join by key" or "teach normalize units." The hard object is to infer which transformations are admissible, meaning-preserving, meaning-changing, unsafe, or goal-relevant in a new domain.

### Required Shape

Start without giving the system a named operation family. Let it propose candidate transformations, counterexamples, verifier obligations, abstention conditions, and repair operations. Then compare against active learning, CEGIS, PBE, schema matching, and library learning.

### Verdict

```text
SELF-DISCOVERED TRANSFORMATION GRAMMARS IS THE BEST CONTINUATION OF THE NEGATIVE EVIDENCE.
```

### Kill Record

Do not begin this reframe with a hand-authored transformation grammar. That would recreate PCCP-H and FrameSeed with new labels.
## I320: A Formal Lower-Bound Program Is The Third Missed Path

### Attack

The specs repeatedly invoke AFTD and less-than-4x absorption. But no arc artifact proves a lower bound where independent teaching or synthesis must pay more than a frame packet. Without lower bounds, any positive benchmark remains vulnerable to "your baseline was not strong enough."

### Stronger Route

Try to prove a toy theorem:

```text
There exists a family where any learner restricted to representation R0 needs Omega(k * g(n)) teaching/search cost across k tasks, while a paid representation extension F plus small bindings solves at O(g(n) + k).
```

Then implement the theorem's exact absorber and exact signal case.

### Verdict

```text
A THEOREM-FIRST FRAME SEPARATION COULD REVIVE THE ABSTRACT QUESTION, BUT NOT THE CURRENT EMPIRICAL FRAMESEED ARC.
```

## I321: Would A Hostile Reviewer Be Convinced?

### Attack

A hostile reviewer would be convinced of the negative result:

```text
No signal was shown.
The Boolean packet is absorbed.
The SHEETS packet is unnecessary once exact bindings/pipeline substrate are granted.
```

They would not be convinced by stronger claims:

```text
The harnesses are fully clean.
Native typed baselines were actually run.
The methodology is publication-grade as-is.
FrameSeed was killed by concept failure rather than by the tested domains and proxy scorers.
```

The correct hostile-review sentence is:

```text
This repo is unusually honest about killing its own signal, but its executable baselines lag its adversarial prose.
```

### Verdict

```text
CONVINCING AS A DIRECTION-KILLING INTERNAL RECORD; NOT YET CONVINCING AS A PUBLIC CLAIM OF COMPLETE BASELINE RIGOR.
```

## I322: Final Fresh-Eyes Recommendation

### Attack Synthesis

FrameSeed asked whether compact packets can transmit reusable frames to cheap learners. The observed answer is no in the tested form. Boolean worlds are absorbed by exact teaching/search. SHEETS worlds are absorbed once exact bindings and typed pipeline substrate are granted.

The methodology's value is not "FrameSeed works." Its value is the adversarial discipline: precommit the exciting claim, arm the boring absorbers, let them win, and record the token without rescue.

### Final Token Recommendation

```text
FRAMESEED_CURRENT_FORM_ABSORBED
```

More specific status:

```text
BOOLEAN: FRAMESEED_T3_ABSORBED_BY_TEACHING_DIMENSION
SHEETS: FRAMESEED_SHEETS0_ABSORBED_BY_SCHEMA_BINDING
ARC: kill or radically reframe; do not run another FrameSeed domain as-is
```

### Next Direction Recommendation

Redirect to one of these, in order:

1. Self-discovered typed transformation grammars.
2. Binding/goal discovery under ambiguity with active disambiguation and abstention.
3. A theorem-first AFTD separation with exact lower-bound absorbers.

### Final Kill Records

```text
KR-B40-1: If a future report says SHEETS native baselines were fully executed, correct it; B31 used capability-mode scoring.
KR-B40-2: If a future direction keeps the public typed substrate, operation grammar, and exact bindings fixed, it has not escaped FrameSeed absorption.
KR-B40-3: If the next experiment weakens teaching, CEGIS, PBE, schema-binding, or library-learning baselines to manufacture a packet win, reject it before hidden opening.
KR-B40-4: If the project wants a home run, move from frame transmission to frame/geometry discovery or prove a formal separation.
```

## Final Answer To The Prompt

Are the absorptions honest?

```text
Yes for internal direction assessment. Boolean absorption is strong. SHEETS schema-binding absorption is directionally honest, but the evidence is a conservative packet-erasure/mode-scoring demolition rather than a native typed baseline benchmark.
```

Is there a path forward we missed?

```text
Yes, but not as current FrameSeed. The live paths are binding discovery, self-discovered transformation grammars, or theorem-first frame separation. Each is a new direction or radical reframe.
```

Is the methodology sound?

```text
The adversarial methodology is sound in spirit and unusually honest in negative token assignment. It is not yet sound enough to make strong public claims about implemented absorber competence unless future work closes the executable gap.
```

Would a hostile reviewer be convinced the work was rigorous?

```text
They would be convinced that no signal exists and that the team did not overclaim the two absorptions. They would not be convinced that SHEETS-0 fully implemented the typed prior-art baselines named in the spec.
```

Bottom line:

```text
Stop current FrameSeed. Preserve the absorption ladder. Redirect toward discovering the geometry rather than packaging geometry supplied by the experiment designer.
```