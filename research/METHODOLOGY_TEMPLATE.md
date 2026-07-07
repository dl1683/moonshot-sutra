# Reusable Absorption Methodology Template

**Status:** Carry-forward framework after the FrameSeed arc.
**Use when:** A new moonshot direction needs a precommit, absorber ladder,
hidden-open harness, and adversarial claim ceiling before implementation.

This template is the preserved artifact from PCCP-H and FrameSeed. It is not a
FrameSeed rescue. It is a way to make the next direction killable, auditable, and
hard to overclaim.

## 0. Direction Charter

Start every new arc with these fields:

```text
Direction name:
Core claim:
One-sentence home-run story:
Which sacred outcomes it serves:
What ordinary explanation would make it boring:
What evidence would kill it:
What evidence would make a hostile reviewer pause:
```

Rules:

1. The claim must be about function, not a proxy.
2. The mechanism is replaceable. The five sacred outcomes are not.
3. A direction cannot begin with a private ontology, verifier, grammar, parser,
   binding map, or operation family that already contains the answer.
4. If the strongest boring explanation wins, emit its token and stop.

## 1. Precommitted Terminal Tokens

Define tokens before implementation. Use one namespace per direction.

Minimum token classes:

```text
<DIR>_SIGNAL
<DIR>_ABSORBED_BY_REPRESENTATION_PRIOR
<DIR>_ABSORBED_BY_PARSER_OR_SUBSTRATE_PRIOR
<DIR>_ABSORBED_BY_TEACHING_DIMENSION
<DIR>_ABSORBED_BY_ACTIVE_LEARNING
<DIR>_ABSORBED_BY_CEGIS_OR_SYNTHESIS
<DIR>_ABSORBED_BY_LIBRARY_LEARNING
<DIR>_ABSORBED_BY_SCHEMA_OR_BINDING
<DIR>_ABSORBED_BY_DOMAIN_TOOL
<DIR>_ABSORBED_BY_NUISANCE_ORACLE
<DIR>_TRAP_DOMAIN_DEGENERATE
<DIR>_VOID_PROTOCOL_OR_LEAKAGE
<DIR>_NEGATIVE
```

Token precedence:

1. Leakage, smuggling, hidden mutation, baseline asymmetry, subjective hidden
   semantics, or uncounted human/parser substrate -> `VOID`.
2. Degenerate task, tiny lookup table, renamed Boolean mask, or type/name leak ->
   `TRAP_DOMAIN_DEGENERATE`.
3. Public representation, parser, substrate, task grammar, binding, or prior
   already contains the result -> the relevant representation/prior token.
4. Any ordinary baseline reaches threshold at matched or less-than-4x total
   information/cost -> the most specific absorber token.
5. Claimed system misses the functional threshold -> `NEGATIVE`.
6. All signal gates pass and no absorber wins -> `SIGNAL`.

Mixed evidence cannot be narrated into signal. Exactly one terminal token is
assigned per hidden opening.

## 2. Absorption Ladder

The ladder is the center of the method. Implement the strongest boring
explanations first, not after a positive result appears.

Required rungs:

| Rung | Question | Typical absorber |
|---|---|---|
| Representation prior | Did public features already expose the answer? | Parser, feature, type tag, name, ontology |
| Teaching dimension | Is the packet just a shortest teaching set? | Exact or bounded teaching-set solver |
| Active disambiguation | Can counterexamples cheaply isolate the target? | Active learner, query planner |
| Synthesis / CEGIS | Can a program be synthesized from the same evidence? | CEGIS, PBE, PROSE, SyGuS, ILP |
| Library learning | Can reusable macros explain transfer? | DreamCoder-style library, MDL macros |
| Domain tool | Is this already a solved local tool problem? | SQL, schema matching, unit systems, constraint solving |
| Nuisance oracle | Does removing nuisance structure solve it? | Relevant-feature oracle, invariant oracle |
| Human/substrate prior | Did the researcher or substrate do the work? | Hand-authored parser, grammar, verifier, binding map |

Each rung must declare:

```text
Inputs received:
Information/cost charged:
Functional metric:
Threshold:
Native execution, proxy absorber, capability-mode scoring, or proof:
Failure mode:
Absorption token if it wins:
```

## 3. Equal-Information Baseline Contract

Any executable field the claimed system can use must be given to baselines as
identical data or as a declared lossless translation.

Baseline parity checklist:

- Same public training/transcript data.
- Same examples and counterexamples, charged identically.
- Same public parser, type system, ontology, unit registry, verifier semantics,
  operation grammar, tool API, and schema metadata.
- Same hidden-open boundary.
- Same time, query, packet, program, and human-labor budgets or a justified cost
  multiplier.
- Same canonical functional scorer.
- Same abstention and invalid-output rules.

If a baseline is only approximated, label it as such before hidden opening:

```text
native_executable
proxy_absorber
capability_mode_scored
formal_lower_bound
untested_roster_entry
```

Do not upgrade a proxy absorber into a native-baseline claim after the run.

## 4. Cost Ledger

Every result needs a cost split. Use the vocabulary below even when the direction
does not use packets.

```text
F = reusable frame, rule, grammar, verifier, representation, or method bits
B_i = task-specific binding bits
P_i = executable per-task program or policy bits
E_i = examples and labels
C_i = counterexamples and active-query answers
V_i = verifier obligations or proof/test clauses
L = learned library or macro bits
H = charged share of human-authored parser/substrate/design work
R_i = residual teaching bits after reusable structure is installed
```

Report:

```text
total_cost = F + sum(B_i + P_i + E_i + C_i + V_i + R_i) + L + H
frame_only_cost = F
binding_ratio = sum(B_i) / max(1, total_cost)
program_ratio = sum(P_i) / max(1, total_cost)
human_substrate_ratio = H / max(1, total_cost)
```

A tiny reusable claim plus large bindings or programs is not signal. It is a
binding, synthesis, or human-substrate result.

## 5. Hidden-Open Harness Pattern

A reusable harness has these surfaces:

1. Public spec and token policy.
2. Frozen manifest with hashes of code, specs, parameters, token precedence,
   baselines, thresholds, seeds, and cost rules.
3. Split RNG for public/smoke, hidden constructor, nuisance, role names,
   tie-breaks, and scorer queries.
4. Blind constructor that cannot read latent roles, hidden labels, solution
   programs, hidden family IDs, or hidden query answers.
5. Provenance log for every artifact entry: source fact, constructor rule, cost
   charged, category, and executable status.
6. Public smoke on a separate seed.
7. Hidden seed opened once after the manifest freeze.
8. No constructor, scorer, baseline, timeout, parser, token-policy, or audit
   changes after hidden opening under the same seed.

Minimum audits:

- Generator leakage audit over public features and latent roles.
- Serializer/cost audit.
- Banned-string and hidden-metadata audit.
- Baseline parity audit.
- Type/parser/tool parity audit.
- Role/schema/name/unit/order permutation audit.
- Packet-erasure or component-erasure ablation.
- Randomized-label or randomized-obligation control.
- Human-labor and parser-substrate ledger.

## 6. Functional Metrics

Define metrics from the function the direction claims to create.

Required fields:

```text
functional_accuracy:
minimum_per_family_accuracy:
hidden_family_coverage:
abstention_quality:
repair_success:
locality_of_repair:
cost_per_success:
amortized_transfer_gap:
permutation_stability:
claim_ceiling_honored:
```

Proxy metrics can be logged, but they cannot decide the terminal token unless
the direction explicitly claims the proxy and proves why it is aligned with
function.

## 7. AFTD-Style Transfer Test

When the claim is reusable structure, require siblings.

Definitions:

```text
TD_H0(t) = minimum counted teaching/search cost for baseline H0 on task t
reduced(sibling) = TD_after(reusable_structure, sibling) <= beta * TD_H0(sibling)
AFTD = reusable_structure_cost / count_reduced_siblings
AFTD_all_in = all_counted_cost / count_reduced_siblings
```

Use a strict `beta`, such as `0.50`, unless precommitted otherwise. The all-in
version is binding. If bindings, programs, or library bits explain the transfer,
emit the appropriate absorber.

## 8. Direction Report Skeleton

Every measurement report should include:

```text
# <Direction> Measurement Report

Date:
Hidden seed opened:
Code changes after hidden open:
Manifest hash:
Terminal token:

## Executive Verdict
## What Was Tested
## Token Precedence
## Harness Integrity
## Baselines And Absorbers
## Cost Ledger
## Functional Metrics
## Ablations And Erasures
## Permutation / Leakage / Randomization Audits
## What Absorbed
## What Did Not Absorb
## Claim Ceiling
## Kill Or Next Gate
```

If the result is negative, explain why the methodology still helped. If the
result is positive, explain which boring explanations were defeated and how.

## 9. Preflight Checklist

- [ ] The direction starts from `research/VISION.md`.
- [ ] Sacred outcomes served are explicit.
- [ ] Signal, absorption, void, trap, and negative tokens are precommitted.
- [ ] Token precedence is exact.
- [ ] Strongest boring baselines are listed before implementation.
- [ ] Native/proxy/capability-mode/proof status is declared for each baseline.
- [ ] Equal-information translation is specified.
- [ ] Cost ledger categories are frozen.
- [ ] Hidden-open boundary is frozen.
- [ ] Public smoke seed differs from hidden seed.
- [ ] Constructor is blind to latent roles and hidden labels.
- [ ] Generator leakage audit exists.
- [ ] Serializer and cost audit exists.
- [ ] Baseline parity audit exists.
- [ ] Role/schema/name/order permutation audit exists.
- [ ] Randomized control exists.
- [ ] Component-erasure ablations exist.
- [ ] Claim ceiling is written before results.
- [ ] A hostile reviewer sentence is written before results.
- [ ] The report template exists before hidden opening.

## 10. Specific Warning For The Next Arc

If the next direction is self-discovered typed transformation grammars, it must
not start by providing:

- a named transformation grammar;
- exact task bindings;
- a complete operation ontology;
- a hidden-family-specific verifier;
- a public parser that identifies the target roles;
- a small typed DSL whose programs are enumerable under the same budget.

The live problem is discovering admissible transformations, obligations,
bindings, and repair operations under ambiguity. If those are supplied, the next
arc will be absorbed for the same reason PCCP-H and FrameSeed were absorbed.
