# Q-Loop B30: Design the B2 World and Attack the Relation Miner

**Date:** 2026-07-07
**Role:** Question Loop worker
**Iterations:** I204-I210
**Status:** analysis-only B2 absorption attack; CPU-only constraint; no implementation, no training, no experiments, no web runs.

---

## Grounding

Read in the requested order:

1. `research/VISION.md`
2. `research/question_loop_batch29.md`
3. `research/dual_loop_supervisor_checkin_20.md`
4. `code/pccp0_witness.py`
5. `research/PCCP_THEOREM_DRAFT.md`
6. `research/DEEP_RETHINK.md`

Additional process grounding:

- `C:\Users\devan\.claude\projects\C--Users-devan-OneDrive-Desktop-Projects-AI-Moonshots-moonshot-sutra\memory\MEMORY.md`

Binding facts:

- The two invariants remain fixed: swing for the home run, and the loop stops only when an adversary cannot knock it down.
- The five sacred outcomes remain genuine intelligence, improvability, democratized development, data efficiency, and inference efficiency.
- The kill history's central lesson remains proxy/function divergence. A measured proxy can improve while the real function does not.
- PCCP-H's after-frame artifact story is still alive: finite proof-carrying causal programs can preserve the target function while surface reconstruction pays nuisance cost.
- PCCP-H's discovery story is not yet alive. B1 was absorbed by exhaustive single-field screening in B22.
- Supervisor #20 makes the current burden explicit: B2 relation discovery must be tested against exhaustive metamorphic relation mining over the same supplied `(T, Phi)` grammar.

Current absorption ladder:

```text
B0: prior art - absorbed
B1: single-field invariance - ABSORBED
B2: metamorphic relation discovery - live but under attack
B3: decomposition discovery - open, potentially paradigm-level
B4: open-world / transformation-grammar discovery - unsolved
B5: universal frame formation - impossible
```

This batch's burden:

```text
Design a concrete B2 world, then try to kill B2 by showing that a smart
exhaustive metamorphic miner gets the same result under equal information.
```

---

## I204: B2 World Design

### Steelman

The B2 world must not be another spurious-field invariance demo. The bad program must pass the partial verifier and the B1 invariance checks, then fail only when covariance or composition obligations are added.

Use a role-permuted parity world with a deliberately incomplete public support:

```text
Parameters:
  m = 12 nuisance bits
  observed field count n = 2 causal + 12 nuisance + 1 spurious = 15
  binary observed fields x0...x14
  role permutation pi sampled per world

Latents:
  C0, C1 in {0,1}
  N0...N11 in {0,1}
  S in {0,1}

Factual spurious construction:
  S := C0 XOR C1

Target:
  y = C0 XOR C1

Observation:
  x_pi(C0) = C0
  x_pi(C1) = C1
  x_pi(Nj) = Nj
  x_pi(S)  = S
```

The public V0 support deliberately fixes one causal bit:

```text
Public base examples E_pub:
  C0 = 0
  C1 in {0,1}
  N ranges over 16 fixed nuisance assignments
  S ranges over {0,1}

So:
  |E_pub| = 1 * 2 * 16 * 2 = 64
```

This avoids the all-label-zero degeneracy. V0 sees both target labels:

```text
when C0=0:
  y = C1
```

The true program is:

```text
P_true(x) = x_pi(C0) XOR x_pi(C1)
```

The B2 bad program is:

```text
P_bad_B2(x) = x_pi(C1)
```

This is the exact B2 analogue of the B1 shortcut. It is not a spurious-field shortcut; it is a support shortcut. On the public support `C0=0`, it is extensionally correct:

```text
P_bad_B2(x) = C1 = C0 XOR C1 = y.
```

It also passes B1 invariance checks:

```text
flip(Nj) -> id(y)   for all j
flip(S)  -> id(y)
```

because `P_bad_B2` ignores all `N_j` and `S`.

The B2 input transformation grammar is supplied as:

```text
T_single:
  flip(xj) for every observed binary field xj

T_pair:
  flip(xj, xk) for every unordered observed pair j < k

T = T_single union T_pair
```

For `n=15`:

```text
|T_single| = 15
|T_pair|   = C(15,2) = 105
|T|        = 120
```

The output relation grammar is supplied as:

```text
Phi = {id, NOT}
```

So exhaustive B2 mining has:

```text
paired target queries = |E_pub| * |T| = 64 * 120 = 7,680
relation score checks = |E_pub| * |T| * |Phi| = 15,360
```

This is CPU-trivial.

### Threat Question

What does `V0_B2` check, and what does it not check?

`V0_B2` checks:

```text
1. Public labels:
   P(x) = y for every x in E_pub.

2. B1 nuisance invariance:
   P(flip(Nj)(x)) = P(x) for every x in E_pub and every Nj.

3. B1 spurious invariance:
   P(flip(S)(x)) = P(x) for every x in E_pub.
```

`V0_B2` does not check:

```text
1. flip(C0) -> NOT(y)
2. flip(C1) -> NOT(y)
3. flip(C0,C1) -> id(y)
4. any relation involving C0 composed with nuisance or spurious flips
5. any held-out public support with C0=1 unless produced through the relation oracle
```

The hidden B2 tests are paired relation checks, not ordinary held-out accuracy checks:

```text
H_C0:
  for x in E_pub:
    P(flip(C0)(x)) == NOT(P(x))

H_C1:
  for x in E_pub:
    P(flip(C1)(x)) == NOT(P(x))

H_C0C1:
  for x in E_pub:
    P(flip(C0,C1)(x)) == P(x)
```

`P_bad_B2 = C1` behaves as follows:

| Test | `P_true` | `P_bad_B2` |
|---|---:|---:|
| Public labels, `C0=0` | pass | pass |
| `flip(Nj) -> id` | pass | pass |
| `flip(S) -> id` | pass | pass |
| `flip(C1) -> NOT` | pass | pass |
| `flip(C0) -> NOT` | pass | fail all 64 paired cases |
| `flip(C0,C1) -> id` | pass | fail all 64 paired cases |

This is the smallest honest B2 failure: B1-only invariance cannot reject `P_bad_B2`, because the bad program already ignores nuisance and spurious fields. The missing obligation is a metamorphic relation.

### What We Might Be Missing

This world is honest for testing relation compilation, but it is not enough to prove B2 novelty.

The exhaustive miner over the supplied grammar finds the decisive clauses immediately:

```text
flip(C0)    -> NOT
flip(C1)    -> NOT
flip(C0,C1) -> id
flip(Nj)    -> id
flip(S)     -> id
```

The exact B2 world is therefore good as an absorption suite, not as a moonshot victory. It will almost certainly produce:

```text
B2_DISCOVERY_ABSORBED
```

unless the relation miner shows a measured advantage in query count, false-positive rejection, transfer, or compiled repair value against a smart baseline.

### Verdict

```text
B2_WORLD_SPECIFIED.
P_bad_B2 = x_pi(C1).
V0_B2 catches B1 failures but deliberately omits covariance and pair composition.
The world is CPU-small and B1-only invariant mining cannot reject the bad program.
However, exhaustive metamorphic mining over supplied T and Phi should reject it cheaply.
```

---

## I205: When Does Exhaustive Metamorphic Mining Fail?

### Steelman

The strongest absorption baseline is:

```text
Input:
  examples E = {(x_i, y_i)}
  input transform grammar T
  output relation grammar Phi
  target oracle F for transformed inputs

For each tau in T:
  For each x in E:
    query y_tau = F(tau(x))
  For each phi in Phi:
    score(tau, phi) = count_x[y_tau == phi(y)]
  accept exact or high-confidence relations by support, MDL, and controls.
```

For binary fields with flips up to arity `k`:

```text
|T_<=k| = sum_{a=1..k} C(n,a)
Q       = |E| * |T_<=k|
Score   = |E| * |T_<=k| * |Phi|
```

For fields with domain size `d` and replacement transforms:

```text
|T_<=k| = sum_{a=1..k} C(n,a) * (d-1)^a
```

For the exact B2 world:

```text
n = 15
|E| = 64
k = 2
|Phi| = 2
|T| = 15 + 105 = 120
Q = 7,680 paired target queries
Score = 15,360 relation checks
```

That is not just feasible. It is negligible.

Now scale it:

| `n` | `|E|` | `k` | binary `|T_<=k|` | paired queries |
|---:|---:|---:|---:|---:|
| 15 | 64 | 2 | 120 | 7,680 |
| 64 | 128 | 2 | 2,080 | 266,240 |
| 100 | 1,000 | 2 | 5,050 | 5,050,000 |
| 500 | 1,000 | 2 | 125,250 | 125,250,000 |
| 100 | 1,000 | 3 | 166,750 | 166,750,000 |
| 100 | 1,000 | 4 | 4,087,975 | 4,087,975,000 |

For cheap Python with a trivial oracle, millions are fine. Hundreds of millions are annoying but still not conceptually prohibitive. Billions are where the CPU-only suite starts to hurt.

Large domain size hurts faster:

```text
n = 100, d = 16, k = 2
|T| = 100*15 + C(100,2)*15^2
    = 1,500 + 4,950*225
    = 1,115,250

If |E|=1,000:
  Q = 1.115 billion paired target queries
```

Large output grammar can also dominate. For binary output:

```text
|Phi| = 2
```

For categorical output with `r` labels and arbitrary label permutations:

```text
|Phi| = r!
```

That explodes immediately. But if `Phi` is restricted to a generated group, affine maps, monotone maps, or a small edit grammar, it becomes cheap again.

### Threat Question

Under what conditions does exhaustive B2 mining become too expensive?

It becomes expensive when at least one of these is true:

| Driver | Why it hurts | But what it means |
|---|---|---|
| large `n` | pair and higher-order transforms grow as `n^k` | still manageable for `k=2` unless `n` is hundreds/thousands |
| large `d` | replacement transforms grow as `(d-1)^k` | common in numeric/categorical fields |
| high arity `k` | `C(n,k)` becomes explosive | this is drifting toward B3 interaction/decomposition |
| large `|E|` | every candidate transform needs support | active sampling helps but does not change identifiability |
| large `|Phi|` | output relation search can dominate | arbitrary output grammar is a smuggling/grammar problem |
| expensive oracle | each query is costly | not true for the tiny CPU parity witness |
| unknown `T` | cannot enumerate what was not supplied | this is B4 transformation-grammar discovery |

For the finite worlds currently under discussion, with supplied single and pair flips and `Phi={id,NOT}`, exhaustive mining remains cheap.

### What We Might Be Missing

The real hard part is not `O(|E| n^2 |Phi|)` in a Boolean toy world. The real hard part is that `(T, Phi)` are supplied.

If the human gives:

```text
T = {single flips, pair flips}
Phi = {id, NOT}
```

then the miner's job is ordinary enumeration and scoring. If the human gives:

```text
T = all valid semantic transformations of the domain
Phi = all meaningful output transformations
```

the human has already supplied the core frame.

### Verdict

```text
B2_IS_ABSORBED_FOR_SMALL_SUPPLIED_GRAMMARS.
```

Exhaustive metamorphic mining fails only when the supplied grammar is large enough to be painful, the oracle is expensive, high-order interactions matter, `Phi` is large, or the transformation grammar is not supplied. Those conditions push the live claim toward B3 or B4, not the toy B2 parity world.

---

## I206: Active Query Selection

### Steelman

There is one plausible B2 edge: active query selection can avoid testing every pair.

A simple active strategy:

```text
Input:
  E, T_single, optional T_pair, Phi

Stage 1: unary screen
  For each field j:
    query flip(j) on all or a confidence sample of E.
    assign best output transform phi_j in Phi.
    mark field stable if score(j, phi_j) is exact or high-confidence.

Stage 2: composition prediction
  For each pair (j,k):
    if j and k are stable:
      predict phi_jk = phi_k o phi_j.
      do not query the pair, or query only a small audit sample.
    else:
      query flip(j,k) directly.

Stage 3: compile and audit
  Prefer shortest generating set:
    flip(C0)->NOT
    flip(C1)->NOT
    flip(Nj)->id
    flip(S)->id
  Derive pair relations from closure when algebraically sound.
```

In the exact B2 world:

```text
n = 15
|E| = 64
Exhaustive single+pair Q = 64 * (15 + 105) = 7,680

Unary-only active Q = 64 * 15 = 960
Reduction = 8.0x
```

If the active miner audits every pair on only `h=4` examples:

```text
Q_active_audited = |E|*n + h*C(n,2)
                 = 64*15 + 4*105
                 = 960 + 420
                 = 1,380

Reduction = 7,680 / 1,380 = 5.56x
```

For a larger CPU stress setting:

```text
n = 64
|E| = 128
|T_single+pair| = 64 + C(64,2) = 2,080

Q_exhaustive = 128 * 2,080 = 266,240
Q_unary_only = 128 * 64 = 8,192
reduction = 32.5x

With h=4 pair audits:
Q_active_audited = 8,192 + 4*C(64,2)
                 = 8,192 + 8,064
                 = 16,256
reduction = 16.4x
```

General sparse formula:

```text
n = total fields
u = unstable fields after unary screening
s = n-u stable fields

Direct pair tests needed:
  unstable-stable pairs: u*s
  unstable-unstable pairs: C(u,2)

Q_active = |E| * n + |E| * (u*s + C(u,2))
```

If `u << n`, this is much smaller than `O(|E| n^2)`. If `u = Theta(n)`, the advantage disappears.

### Threat Question

Is this a real edge over exhaustive mining?

It is a real edge over naive exhaustive mining. It is not a real edge over a smart exhaustive baseline.

The baseline can be upgraded to:

```text
Smart Exhaustive MR Miner:
  1. Test single flips first.
  2. Infer algebraic closure where exact.
  3. Test only unclosed or low-confidence pairs.
  4. Audit a random sample of derived pairs.
  5. Fall back to exhaustive enumeration when closure assumptions fail.
```

That is the natural version of exhaustive metamorphic mining. It is not special to FDM-0.

### What We Might Be Missing

Active selection becomes meaningful only under a strict budget:

```text
Q_budget < |E| * |T|
```

Then the question is not "can exhaustive eventually find it?" The question is:

```text
Which strategy finds the transferable clauses before the query budget ends?
```

That is a valid engineering contest, but the baseline must be allowed to use the same active strategy. Otherwise the experiment is rigged.

### Verdict

```text
ACTIVE_SELECTION_CAN_REDUCE_PAIR_QUERIES_FROM_O(|E|n^2)_TO_O(|E|n)
WHEN_UNARY_EFFECTS_ARE_STABLE_AND_COMPOSITION_IS_SOUND.
```

This is useful. It is not yet novel. A smart metamorphic miner should do it too.

---

## I207: MDL Scoring Under False Positives

### Steelman

In small finite worlds, exact relations can hold on public support by coincidence. B2 needs a false-positive stress test.

A concrete false-positive world keeps the same parity target but expands the transformation grammar with conditional transforms.

```text
Latents:
  C0, C1 in {0,1}
  G0...G19 guard bits in {0,1}
  N0...N7 nuisance bits
  S := C0 XOR C1

Target:
  y = C0 XOR C1

Public support:
  all public examples have G0=...=G19=0
  C0 and C1 vary enough to include y=0 and y=1
  N and S vary as controls

Hidden support:
  guard bits vary uniformly
```

Supply a transformation grammar containing:

```text
True short transforms:
  flip(C0)
  flip(C1)
  flip(C0,C1)
  flip(Nj)
  flip(S)

Conditional decoy transforms, for each guard Gj:
  tau_j_zero = if Gj == 0 then flip(C0) else id
  tau_j_one  = if Gj == 1 then flip(C0) else id
```

On public support, every `Gj=0`, so:

```text
tau_j_zero behaves exactly like flip(C0)
  apparent relation: tau_j_zero -> NOT
  hidden failure: when Gj=1, tau_j_zero becomes id, so NOT is false

tau_j_one behaves exactly like id
  apparent relation: tau_j_one -> id
  hidden failure: when Gj=1, tau_j_one becomes flip(C0), so id is false
```

Count the false positives.

Let the candidate transform set for this stress test be:

```text
T_true = {flip(C0), flip(C1), flip(C0,C1), flip(S)}
         plus 8 nuisance flips
       = 12 transforms

T_decoy = 2 * 20 = 40 conditional guard transforms

|T| = 52
|Phi| = 2
total (tau, phi) pairs = 104
```

For each conditional decoy transform, exactly one `phi` appears exact on public support and fails hidden:

```text
false exact seen pairs = 40
false exact seen rate = 40 / 104 = 38.46%
```

This meets the required 30% false-positive threshold.

### Threat Question

How should MDL scoring distinguish true from coincidental relations?

Use a score like:

```text
score(tau, phi) =
  support_log_likelihood(tau, phi)
  - lambda_T * description_length(tau)
  - lambda_Phi * description_length(phi)
  - lambda_cond * number_of_conditions(tau)
  - lambda_overlap * redundancy_with_shorter_relations
```

The short true relation:

```text
flip(C0) -> NOT
```

should beat:

```text
if Gj == 0 then flip(C0) else id -> NOT
```

because both have exact public support, but the conditional relation is longer and redundant with a shorter unconditional relation.

The rule should be:

```text
If a longer relation has no support advantage over a shorter relation that
explains the same transformed labels, reject the longer relation unless held-out
guard variation proves the condition is real.
```

That said, MDL alone is not magic. If the only relation in `T` is conditional, or if the accidental conditional is as short as the true one under the encoding, public-support MDL cannot identify truth. It is a prior, not a guarantee.

The required controls are:

```text
1. held-out seen validation with guard variation before hidden freeze
2. negative-control guards
3. role/name randomization
4. relation redundancy pruning
5. exact support counts
6. hidden-family transfer after clauses freeze
```

### What We Might Be Missing

The exhaustive miner can use the same MDL.

A fair baseline is:

```text
Exhaustive MR Miner + MDL:
  enumerate all supplied tau in T
  score all phi in Phi
  reject longer redundant exact relations
  validate on held-out seen support if available
  compile shortest nonredundant clauses
```

If Relation Miner v0 and exhaustive mining receive the same `T`, `Phi`, evidence, MDL code, and validation split, MDL is not a differentiator. It is part of the baseline.

### Verdict

```text
MDL_IS_NECESSARY_FOR_B2_FALSE_POSITIVES_BUT_NOT_A_NONABSORBED_EDGE.
```

The false-positive world is worth building because it prevents naive exact-support relation mining from overclaiming. But a smart exhaustive miner with the same MDL and validation controls gets the same protection.

---

## I208: The Composition Closure Edge

### Steelman

The strongest possible B2 edge is composition.

If the miner discovers:

```text
flip(C0) -> NOT
flip(C1) -> NOT
```

then it can derive:

```text
flip(C0,C1) -> id
```

without querying the pair.

Algebraically:

```text
F(tau_a(x)) = phi_a(F(x))
F(tau_b(x)) = phi_b(F(x))
```

If the second relation holds not only on original `x` but also on `tau_a(x)`, then:

```text
F(tau_b(tau_a(x)))
  = phi_b(F(tau_a(x)))
  = phi_b(phi_a(F(x)))
```

So:

```text
tau_b o tau_a -> phi_b o phi_a
```

For Boolean parity:

```text
NOT o NOT = id
```

therefore:

```text
flip(C1) o flip(C0) -> id
```

This is sound under strict conditions:

```text
1. tau_a and tau_b are total valid transformations on the checked domain.
2. tau_a(E) remains inside the domain where tau_b's relation holds.
3. relations are exact, not merely high-confidence.
4. phi_a and phi_b are total output transformations.
5. order is respected if input or output transforms do not commute.
6. no hidden precondition is attached to either relation.
```

For the exact B2 world, this is sound and useful.

### Threat Question

Is composition useful beyond Boolean parity?

Yes, when transformations form a known monoid or group and output relations are functions closed under composition.

| Output structure | Example `Phi` | Composition status | Usefulness |
|---|---|---|---|
| Boolean | `{id, NOT}` | closed group `Z2` | strong |
| finite labels | supplied permutations | closed if permutations compose | strong but `Phi` can be large |
| modular integers | `y -> y + b mod r` | closed group | strong |
| vectors | affine maps `Ay+b` | closed if dimensions fixed | strong but scoring harder |
| sets | element renamings | closed group | strong for equivariance |
| rankings | order-preserving maps | composition possible | relation may be partial |
| monotone scalar | `nondecrease` | weak transitive inequality | less precise |
| probabilities | calibration transforms | composition may drift | fragile |
| many-to-one summaries | lossy maps | composition loses information | weak |

The important distinction:

```text
functional output transforms compose cleanly;
relational or inequality obligations compose only into weaker obligations.
```

### What We Might Be Missing

Composition closure is a genuine query-count reducer, but it is not automatically a novelty claim.

A smart exhaustive baseline can also:

```text
1. enumerate unary generators;
2. learn their output transforms;
3. close the generated group;
4. query only relations not derivable from the generators;
5. audit derived relations.
```

This is standard algebraic hygiene. It becomes a discovery edge only if FDM-0 discovers the transformation algebra itself:

```text
which transforms are generators,
which compositions are valid,
which output maps represent them,
and which preconditions break closure.
```

If those are supplied, composition is an engineering optimization inside the exhaustive baseline.

### Verdict

```text
COMPOSITION_CLOSURE_IS_ALGEBRAICALLY_SOUND_AND_PRACTICALLY_USEFUL,
BUT_IT_IS_ABSORBED_IF_THE_BASELINE_IS_ALLOWED_THE_SAME_CLOSURE_RULES.
```

The closure edge becomes interesting only when the system infers the algebra rather than being handed it.

---

## I209: The Transformation Grammar Smuggling Problem

### Steelman

The deepest B2 attack is:

```text
The human-supplied transformation grammar is the real intelligence.
```

In the exact B2 world, the supplied grammar says:

```text
try flipping one field
try flipping two fields
compare output by id or NOT
```

That grammar already encodes the hypothesis that the world is Boolean, field-local, and parity-like. The relation miner merely measures which supplied transforms work.

A non-smuggled experiment would not supply `T={single flips, pair flips}`. It would supply only:

```text
1. a typed observation space X;
2. public examples and labels;
3. an oracle that can score proposed valid observations;
4. a budget;
5. a very weak edit substrate, or possibly only raw candidate x' proposals.
```

The system must infer a transformation grammar:

```text
T_hat = {tau_1, ..., tau_r}
```

and then infer output relations:

```text
Phi_hat = {phi_1, ..., phi_s}
```

A tractable version:

```text
Transformation Grammar Discovery v0

Input:
  typed finite examples E
  labels y
  admissibility oracle A(x') in {0,1}
  target oracle F(x') for admissible x'
  weak primitive library P:
    - choose a typed field
    - substitute another observed value of same type
    - swap two same-type entity records
    - rename IDs consistently
    - compose primitives up to length L

Search:
  1. infer schema automorphisms from type signatures and distribution symmetry;
  2. synthesize short edit programs tau over P with high admissibility;
  3. query F(tau(x)) for sampled x;
  4. score tau by target relation simplicity, support, MDL, and closure;
  5. promote short generators whose closure predicts additional valid relations.
```

This still supplies a primitive library `P`. That is less smuggled than supplying `T`, but it is not open-world discovery.

A more honest but harder version:

```text
No T and no edit DSL:
  Candidate transformations are arbitrary functions X -> X.
```

For finite `|X|=M`, the number of arbitrary transformations is:

```text
M^M
```

For even `M=2^15=32,768`, this is hopeless:

```text
(32768)^(32768)
```

No finite CPU system can discover arbitrary transformations without strong priors.

### Threat Question

What would "transformation grammar discovery" look like in the parity world?

A minimally honest parity experiment:

```text
World:
  same role-permuted parity world
  n=15 binary fields

Not supplied:
  single flips
  pair flips
  causal roles
  output NOT

Supplied:
  field values are binary
  observations are admissible bitstrings
  target oracle can label proposed bitstrings
  program-length prior favors small coordinate edits

System must discover:
  1. atomic coordinate flip edits are useful transformations;
  2. only two coordinates have NOT effect on y;
  3. other coordinates have id effect;
  4. composing two NOT-effect generators yields id;
  5. the concise grammar is "Boolean coordinate flips with Z2 output action."
```

But even here the phrase "binary fields" already gives a strong edit prior. The system naturally proposes bit flips because the schema says fields are bits.

A stronger structured-domain experiment:

```text
Entity world:
  input is a list of k records with arbitrary IDs
  target is invariant/equivariant under consistent ID renaming and record order

Not supplied:
  permutation grammar
  renaming grammar

System must infer:
  1. IDs are names, not quantities;
  2. record order is irrelevant;
  3. consistent renaming is valid;
  4. output should be renamed consistently.
```

This starts to look like real intelligence geometry. It is also B4-ish, because the hard step is discovering the valid semantic transformation family.

### What We Might Be Missing

There is no free lunch here.

Without a supplied transformation prior, "find all meaningful transformations" is underdetermined:

```text
Any finite labeled dataset admits many accidental automorphisms, many adversarial
transformations, and many equally short-but-wrong grammars under the wrong
encoding.
```

The only tractable path is to admit the priors explicitly:

```text
typed schemas,
locality,
entity structure,
conservation laws,
program-length bias,
admissibility oracles,
distribution support,
group closure,
and human-auditable candidate generators.
```

That is not a defeat. It is the honest ledger:

```text
system discovered relations over supplied T;
system discovered T over supplied primitive edit DSL;
system discovered primitive edit DSL from typed schema;
system discovered schema/goals from raw world.
```

These are different claims.

### Verdict

```text
TRANSFORMATION_GRAMMAR_DISCOVERY_IS_THE_REAL_FRAME_DISCOVERY_PROBLEM.
```

It is tractable only under strong typed, local, compositional priors. Without those priors, it is B4/open-world and essentially hopeless as a universal claim.

---

## I210: Final B2 Assessment

### Steelman

The favorable reading of B2:

```text
B2 upgrades FDM-0 from field invariance to executable metamorphic obligations.
It can discover covariance and composition relations that B1 cannot express,
compile them into verifier clauses, and reject bad support shortcuts like
P_bad_B2 = C1.
```

That is true but weak.

The hostile reading:

```text
Once the human supplies T and Phi, B2 is just metamorphic relation mining.
For small finite worlds and bounded arity, exhaustive or smart-active mining
finds the same clauses cheaply. MDL, pruning, and composition closure are
natural parts of the baseline.
```

That is also true, and stronger.

### Threat Questions

#### (a) Is B2 absorbed by exhaustive metamorphic mining for realistic finite worlds?

For supplied small grammars, yes.

```text
If T = single flips + pair flips
and Phi = {id, NOT}
and the oracle is cheap,
then B2 is absorbed.
```

The exact B2 world has:

```text
Q = 7,680 paired target queries
```

The stress world with `n=64`, `|E|=128`, pairs included has:

```text
Q = 266,240 paired target queries
```

Both are cheap.

Exhaustive starts to hurt only when:

```text
n is very large,
d is large,
arity k >= 3 or 4,
Phi is large,
the oracle is expensive,
or T is not supplied.
```

Those are not clean B2 wins. They are either scaling contests or shifts into B3/B4.

#### (b) If B2 is absorbed, what is the smallest non-absorbed discovery level?

The smallest plausible non-absorbed level is B3:

```text
decomposition discovery that reduces search, verification, or repair cost
by finding useful component boundaries not explicitly supplied in T.
```

B3 could matter if it demonstrates:

```text
1. recovered dependency blocks;
2. shorter verifier clauses;
3. fewer synthesis candidates;
4. local repair after counterexamples;
5. transfer across role-randomized component worlds;
6. advantage over sensitivity clustering and neural-tool baselines.
```

If the transformation grammar itself is not supplied, the claim moves to B4:

```text
transformation-grammar discovery from typed observations and weak edit priors.
```

#### (c) Does composition closure provide genuine novelty?

It provides genuine value but not genuine novelty under equal-information baselines.

```text
Value:
  reduces pair queries;
  derives relation families from generators;
  exposes algebraic structure;
  compiles concise verifier clauses.

Absorption:
  a smart exhaustive metamorphic miner can do the same closure.
```

Composition becomes novel only if the system discovers:

```text
the generators, the closure law, the output representation, and the precondition
boundaries without being handed them.
```

That is B3/B4 territory.

#### (d) Is transformation grammar discovery tractable or hopeless?

Both, depending on the claim.

Tractable:

```text
typed schemas;
local edits;
small generator programs;
entity renaming;
record permutation;
bounded AST rewrites;
admissibility oracle;
program-length prior;
closure checks.
```

Hopeless:

```text
arbitrary finite transformations X -> X with no schema, no edit prior,
no admissibility structure, and no goal prior.
```

For `|X|=M`, arbitrary transformations are `M^M`. Universal B4 is impossible. Typed, local B4 is a hard but legitimate research target.

#### (e) What is the honest narrative for PCCP-H right now?

The honest narrative is:

```text
PCCP-H has a credible proof-carrying artifact contract and a clean finite
after-frame separation. It does not yet have a non-absorbed discovery mechanism.
B1 is absorbed. B2 is likely absorbed when T and Phi are supplied. The live
moonshot has narrowed to decomposition discovery and transformation-grammar
discovery, with PCCP-H serving as the verifier/compiler/audit layer around
whatever discovery substrate survives.
```

### What We Might Be Missing

There may still be a practical B2 product:

```text
cheap metamorphic-clause mining + MDL + closure + verifier compilation
```

This could be valuable for APIs, smart contracts, IAM policies, data pipelines, and bounded controllers. But value is not the same as paradigm-level novelty.

The home-run standard requires:

```text
not merely "it finds a relation,"
but "it finds a relation or decomposition the obvious miner and tool-using
baseline do not find under equal information."
```

### Verdict

```text
B2: IMPLEMENT AS AN ABSORPTION TEST, NOT AS A VICTORY LAP.
B2 expected token: B2_DISCOVERY_ABSORBED unless query-budget or transfer evidence says otherwise.
B3: smallest plausible non-absorbed discovery level.
B4: real transformation grammar discovery, tractable only with explicit priors.
```

---

## Recommendation

Build W-Loop B23 exactly as an absorption suite:

```text
1. Exact B2 world:
   m=12, n=15, |E_pub|=64, C0 fixed to 0 on public support.

2. Programs:
   P_true = C0 XOR C1
   P_bad_B2 = C1

3. V0_B2:
   public labels plus B1 N/S invariance.

4. Missing clauses:
   flip(C0) -> NOT
   flip(C1) -> NOT
   flip(C0,C1) -> id

5. Baselines:
   no discovery;
   exhaustive MR mining;
   smart-active MR mining with unary-first pruning;
   MDL false-positive stress test;
   random relation search;
   NTB-0 later.

6. Verdict tokens:
   B2_DISCOVERY_ABSORBED if exhaustive/smart-active matches;
   B2_DISCOVERY_SIGNAL only if FDM-0 beats the smart baseline on query budget,
   false-positive control, transfer, repair locality, or compilation quality.
```

Do not claim:

```text
B2 proves frame discovery while T and Phi are supplied and exhaustive mining is cheap.
```

Claim only:

```text
B2 tests whether relation discovery survives the obvious metamorphic mining baseline.
```

Expected result:

```text
B2_DISCOVERY_ABSORBED.
```

This is not failure if reported honestly. It narrows the live moonshot to the next real boundary.

---

## NARRATIVE ATTACK

### 1. Strongest "B2 is also absorbed" dismissal

```text
B2 sounds like a leap from field invariance to relation discovery, but once the
human supplies the input transformations and output relation grammar, it is just
a nested loop. For each supplied tau, query F(tau(x)); for each supplied phi,
check whether y_tau = phi(y). In the proposed parity world, single flips and
pair flips over 15 binary fields require only 7,680 paired queries. The miner
finds flip(C0)->NOT, flip(C1)->NOT, and flip(C0,C1)->id immediately.

MDL does not save the novelty claim because the exhaustive miner can use MDL.
Active querying does not save it because the exhaustive miner can test singles
first and compose pairs. Composition closure does not save it because any smart
metamorphic miner can close a discovered group action.

B2 is not frame discovery. It is metamorphic testing over a frame the human
already wrote down.
```

This dismissal is correct for small finite worlds with supplied `T` and `Phi`.

### 2. Strongest "the transformation grammar is the real intelligence" dismissal

```text
The impressive part was not discovering that C0 flips the answer. The impressive
part was knowing to try field flips, pair flips, and NOT as the output action.
That hypothesis space already says the world is Boolean, local, compositional,
and parity-shaped. The relation miner only measures which of the human's
candidate symmetries hold.

In a real domain, the hard question is not whether a supplied rename operation
is equivariant. The hard question is noticing that IDs are names, row order is
irrelevant, a dosage increase should be monotone, two transactions conserve a
balance, or a program rewrite preserves semantics. That is transformation
grammar discovery. B2 does not solve it. It assumes it.
```

This dismissal remains live until the system discovers useful transformations rather than receiving them.

### 3. What would B2 need to demonstrate to avoid absorption?

B2 needs at least one of the following under equal information:

1. A strict query-budget win over smart-active exhaustive mining, not just naive exhaustive enumeration.
2. A false-positive win where FDM-0 rejects coincidental relations that exhaustive+MDL does not reject.
3. A transfer win across role-randomized or grammar-held-out worlds where exhaustive mining over the public grammar overfits.
4. A composition win where FDM-0 discovers the generator algebra or precondition boundaries, not merely applies supplied closure.
5. A verifier value win: shorter compiled obligations, better hidden failure catch rate, or better repair localization than the baseline clauses.
6. A transformation discovery win: the system proposes a useful `tau` family not listed in the supplied `T`, from a weaker typed edit substrate.
7. A neural-tool baseline win under the same sealed bundle and oracle budget.

Unkillable B2 sentence:

```text
A CPU-only system received examples, a partial verifier, and only weak typed
edit primitives; it discovered a metamorphic relation family not enumerated in
the supplied grammar, compiled it into verifier clauses, caught a hidden support
shortcut, and beat both smart-active metamorphic mining and a neural-tool agent
under equal information.
```

Current honest sentence:

```text
The B1 shortcut was caught by a for-loop. B2 will probably be caught by a
slightly bigger for-loop. The moonshot starts where the for-loop stops: useful
decompositions or transformation grammars that were not already handed to it.
```
