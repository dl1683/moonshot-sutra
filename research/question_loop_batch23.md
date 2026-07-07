# Q-Loop B23: Post-CTI Direction Search

**Date:** 2026-07-07
**Role:** Question Loop worker
**Iterations:** I155-I161
**Status:** analysis-only adversarial direction search; CPU-only constraint; no model, dataset, GPU, or web runs.

---

## Grounding

Read in the requested order:

1. `research/VISION.md`
2. `research/CTI_PRECOMMIT_SPEC.md`
3. `research/work_loop_batch16.md`
4. `research/work_loop_batch17.md`
5. `research/question_loop_batch22.md`
6. `research/dual_loop_supervisor_checkin_14.md`
7. `research/DEEP_RETHINK.md`

Additional local grounding:

- `research/question_loop_batch19.md` contains the prior five-direction scoring that selected CTI before CTI-1 failed.
- A root `CLAUDE.md` was not present in the checkout. The five-direction list is therefore taken from the current prompt and from B19's recovered list.
- `README.md` and the current addendum of `DEEP_RETHINK.md` record later doctrine pressure around evidence-native/Brainseed-style directions. For this batch, those are treated as falsification-pattern evidence, not as a replacement assignment.

Binding facts:

- The sacred mission survived: paradigm shift or failure, democratized intelligence, and "Intelligence = Geometry, not Scale."
- CTI as smooth law `D(C) = D_inf + k*C^(-alpha)` is dead across Board 1 and Board 2.
- Board 1 killed the smooth law on random modular arithmetic: CTI power-law MAE `0.226068`, worse than proxy-only `0.194245`; all non-random forecasters picked `quarter_data`, actual winner was `label_only`.
- Board 2 killed the rescue domain: SmolLM2 MCQ LoRA was not monotone; CTI MAE `0.086820`, worse than linear and proxy-only; CTI picked `label_only`, actual winner was `single_teacher`.
- The repeated live signal is proxy/function divergence: training loss, BPB, hidden-coordinate alignment, early held-out curves, and compute all repeatedly improved without producing robust function.
- The repeated dead pattern is treating an observable shadow as the thing itself.

Current strongest position to attack:

```text
The next direction should not fix CTI. It should pivot from scale/proxy/coordinate
observables to functional geometry: the minimal causal distinctions a small system
must preserve to act correctly under intervention and distribution shift.
```

---

## I155: What Did Ten Kills Actually Teach?

### Steelman

The ten kills are not random. They form one repeated theorem-by-failure:

```text
Every dead direction optimized or forecasted a proxy that was not functionally
sufficient for the task.
```

The Eklavya line tried to transfer teacher knowledge through byte marginals, hidden-coordinate losses, representation alignment, ranking losses, or energy readouts. The proxies moved. BPB improved. Some toy representations contained the answer. But downstream function did not survive the controls.

The CTI line then tried to abstract above mechanisms and ask whether compute reduces functional distortion by a smooth law. That also failed, because compute is not a sufficient state variable. The same compute can produce learning, grokking, memorization, or collapse depending on the regime.

The shared failure is not "KD is hard" or "power laws are wrong." It is deeper:

```text
We keep asking whether a visible scalar, coordinate system, or local training
objective is enough to stand in for intelligence.
```

The data says no:

| Dead proxy | What improved | What failed |
|---|---|---|
| Byte-marginal KD | BPB / byte prediction | HellaSwag-style judgment |
| Hidden cosine alignment | teacher-space statistics | conditional knowledge transfer |
| Width/depth changes | BPB and convergence speed | reasoning benchmark lift |
| Frozen readout toy result | synthetic binding readout | real S0 HellaSwag transfer |
| CTI smooth law | proxy or early curve fitting | held-out functional forecast |
| Label-only MCQ LoRA | train accuracy and loss | held-out function |

The strongest new direction is therefore not another output metric. It is a theory of functional sufficiency:

```text
What compressed internal state preserves exactly the distinctions that matter
for intervention, judgment, and generalization?
```

That points toward Causal World Compression (CWC), not as a video-world-model clone, but as an exact small-scale science of causal states, invariances, and decision-preserving compression.

### Attack

This steelman may be too convenient.

"Functional sufficiency" can become a new grand word for "validation accuracy." If the project simply says "we should preserve causal distinctions" and then trains another classifier on synthetic tasks, the hostile expert will say:

```text
You renamed supervised learning after ten failed losses.
```

There is also a danger of hindsight compression. The kills only look unified after the fact. Each one had a different technical cause: byte-token mismatch, gauge non-identifiability, insufficient world knowledge, underpowered data, phase transitions, overfitting. A single abstraction may erase useful distinctions.

The hardest attack:

```text
Causal sufficiency is obvious in toy worlds where the causal variables are
hand-labeled, and impossible in language where the causal variables are unknown.
```

If the project cannot define causal state without smuggling in the answer key, CWC becomes ordinary feature engineering.

### New Hardest Objection

The missing axiom may not be "we ignored causality." It may be:

```text
The repo keeps assuming intelligence must be discoverable from small closed
experiments. Maybe the phenomenon is intrinsically open-world and cannot be
made Nobel-grade on CPU toy worlds.
```

That objection attacks every CPU-only pivot, not just CWC.

### Verdict + Next-Gate Ranking

Verdict:

```text
The connecting pattern is proxy insufficiency. The next direction must define
and test functional sufficiency directly, or it repeats the kill loop.
```

Next-gate ranking after I155:

| Rank | Direction | Reason |
|---:|---|---|
| 1 | CWC | Directly attacks proxy insufficiency by defining decision-preserving compression. |
| 2 | Renormalization | Explains phase changes, but stays mostly about training dynamics rather than intelligence. |
| 3 | CDMD | Strong narrative, verifier-friendly, but less connected to the kill history. |
| 4 | CTI fragments | Useful diagnostics only: `D_gap`, regime labels, trap detection. |
| 5 | ENI | Important but not yet a theory of intelligence. |

---

## I156: Rescore The Five Nobel-Track Directions After CTI Death

### Steelman

B19's old ranking selected CTI because it salvaged Eklavya's failures into a one-GPU forecasting program:

| Old B19 rank | Direction | Old total |
|---:|---|---:|
| 1 | CTI | 21 |
| 2 | Renormalization | 19 |
| 3 | CDMD | 18 |
| 4 | CWC | 15 |
| 5 | ENI | 15 |

The new evidence changes that table.

CTI had the cleanest first artifact because it made precommitted predictions. That strength is also why it is dead. It did the right scientific thing and lost. Its fragments remain valuable:

- `D_gap` as a memorization-trap warning.
- regime labels as diagnostic vocabulary.
- locked forecasts and baseline discipline as process infrastructure.

But CTI no longer owns the moonshot.

Renormalization improves after CTI's death. Board 1 had grokking-like behavior, Board 2 had memorization-collapse behavior, and both violated monotone smooth curves. A phase-transition theory can say:

```text
The object is not the curve. The object is the phase boundary between
memorization, latent generalization, collapse, and robust function.
```

CWC also improves sharply. The repeated failure of reconstruction and proxy metrics says the small model must preserve causal/functional state, not surface form. CWC can absorb memorization-trap detection:

```text
Memorization is compression of sample identity.
Causal world compression is compression of intervention-stable structure.
```

CDMD remains tempting because verification can be CPU-only. If a small program discovers a shorter proof, construction, sequence rule, or theorem lemma, the story is immediate. But it is less naturally connected to the ten kills unless recast as "compressed causal/program structure."

ENI remains important but weak as a first pivot. Energy per correct decision matters, but without a new algorithmic substrate it becomes measurement and accounting.

### Attack

The rescore might overreact to the last failure.

Renormalization is the obvious post-grokking pivot, but "phase transition" is already a crowded metaphor. It only matters if the project produces a true coarse-graining operator, finite-size scaling, or predictive phase boundary. Otherwise it is CTI with a more flexible vocabulary.

CWC has the opposite problem: too broad. "Causal world compression" could mean JEPA, predictive-state representations, bisimulation, causal representation learning, world models, retrieval, or model-based RL. That breadth makes it easy to sound profound and hard to falsify.

CDMD may actually be the cleanest CPU-only moonshot because math has exact validators. A proof search that finds one new construction beats all conceptual elegance. The attack on CDMD is competition, not incoherence.

### New Hardest Objection

The five-direction list may be missing the real direction:

```text
Evidence-native judgment / retrieval-born intelligence may be the natural
continuation of "do not store the world in 121M weights," but it is not one of
the five listed directions unless folded into CWC.
```

If CWC is chosen, it must explicitly include factual knowledge search as external causal evidence, not just closed-world latent compression.

### Verdict + Next-Gate Ranking

Updated score, using the requested criteria:

| Direction | Manifesto alignment | Narrative strength | CPU-only feasibility | Paradigm-shift potential | Surviving attack surface | Total |
|---|---:|---:|---:|---:|---:|---:|
| CWC | 5 | 5 | 5 | 5 | 3 | 23 |
| Renormalization | 5 | 4 | 5 | 4 | 3 | 21 |
| CDMD | 4 | 5 | 5 | 4 | 2 | 20 |
| CTI fragments | 3 | 2 | 5 | 2 | 4 | 16 |
| ENI | 4 | 3 | 3 | 3 | 2 | 15 |

Verdict:

```text
CWC becomes the top candidate only if it is narrowed to exact causal-state
compression and evidence-conditioned judgment. Renormalization becomes the
theory lane. CTI becomes diagnostics. CDMD remains the optional headline lane.
```

Next-gate ranking after I156:

| Rank | Direction | Gate |
|---:|---|---|
| 1 | CWC | Define a CPU-exact causal compression benchmark and theorem obligation. |
| 2 | Renormalization | Define coarse-graining and order parameters; no metaphor-only phase talk. |
| 3 | CDMD | Identify a niche with cheap verification and real prior-best baselines. |
| 4 | CTI fragments | Build trap diagnostics only if serving CWC/RG. |
| 5 | ENI | Defer unless a new energy-native algorithm appears. |

---

## I157: Is Memorization-Trap Detection A Moonshot Seed?

### Steelman

The strongest positive case:

```text
Memorization-trap detection is the first thing the project has measured twice
that clearly matters.
```

Board 1:

- `quarter_data` reached `100%` train accuracy.
- held-out accuracy fell to `4.76%`.
- proxy loss stayed excellent.
- more compute made the wrong structure more confident.

Board 2:

- `label_only` reached `100%` train accuracy.
- held-out accuracy fell to `43.06%`.
- training loss approached zero.
- `single_teacher` preserved held-out accuracy.

This is a real small-lab problem. Most labs cannot afford to run every intervention to completion. A cheap early warning that says "this run is learning the sample, not the rule" has practical value.

The bigger idea it feeds:

```text
Memorization trap = non-causal compression.
Generalization = causal compression.
```

That converts a feature into a moonshot seed. The goal is not to detect overfitting. The goal is to distinguish two kinds of compression:

| Compression type | Preserves | Fails under |
|---|---|---|
| Sample compression | IDs, shortcuts, surface correlations | held-out interventions |
| Proxy compression | loss/BPB/teacher coordinates | task function |
| Causal compression | intervention-stable state | only when the causal model is wrong |

If a CPU program can expose, prove, and predict that distinction, it becomes a theory of cheap intelligence:

```text
Small models win when they compress causes, not tokens.
```

### Attack

"Memorization trap" is the most obvious thing in machine learning.

Every practitioner knows training loss can fall while validation accuracy falls. Every intro course teaches overfitting. Every early-stopping method monitors train/validation gap. The hostile expert says:

```text
You discovered overfitting and renamed it non-causal compression.
```

This attack kills the direction unless the result is stronger than gap monitoring. It must show one of:

- A trap detector that works before validation collapse is visible.
- A causal/compression invariant that separates memorization from generalization even when validation scores are equal.
- A theorem linking compression type to out-of-distribution failure.
- A construction where reconstruction-trained or scaling-law-trained systems necessarily lose to causal compression.

Otherwise it is a feature.

### New Hardest Objection

The trap detector may depend on held-out labels. If the detector needs the same validation signal it claims to protect, it is not a new intelligence principle:

```text
Without access to a trusted held-out distribution, D_gap is just supervised
early stopping with extra words.
```

The next direction must use transformations, counterfactuals, invariance tests, or evidence consistency checks that create held-out structure cheaply.

### Verdict + Next-Gate Ranking

Verdict:

```text
Memorization-trap detection is not the moonshot. It is an order parameter for
the real moonshot: causal compression versus sample compression.
```

Next-gate ranking after I157:

| Rank | Direction | Update |
|---:|---|---|
| 1 | CWC | Absorbs memorization trap as non-causal compression. |
| 2 | Renormalization | Uses `D_gap` and trap onset as order parameters. |
| 3 | CDMD | Can use compression/trap logic in proof-search heuristics, but indirect. |
| 4 | CTI fragments | Trap dashboard only; not a direction. |
| 5 | ENI | Energy waste story helps narrative but not core theory. |

---

## I158: We Said Geometry, But Tested Scaling Laws

### Steelman

The manifesto says:

```text
Intelligence = Geometry, not Scale.
```

But CTI tested:

```text
D_func as a function of compute.
```

That is a scaling-law question. Even if CTI had worked, it would have shown a compute-performance relation, not necessarily a geometry of intelligence. Its death is therefore clarifying:

```text
Compute is not the coordinate. Geometry is.
```

The correct question is:

```text
What geometry makes two states equivalent for action, and what compression
preserves that equivalence with the fewest bits/parameters/examples?
```

This can be made mathematical on CPU. Candidate objects:

- causal states / predictive-state representations;
- bisimulation metrics;
- sufficient statistics for decision utility;
- invariance groups and counterfactual transformations;
- rate-distortion with distortion defined by action error, not reconstruction error;
- finite-state environments where the true minimal causal partition is computable.

The decisive geometry test is not "does loss go down?" It is:

```text
Does a compressed state preserve the correct action under interventions and
distribution shifts while throwing away surface variation?
```

CWC can be stated as:

```text
Find the shortest state representation Z = f(X) such that for every admissible
intervention do(a), the decision distribution pi(Y | Z, do(a)) matches
pi(Y | X, do(a)) within epsilon.
```

That is a geometry claim, not a scale claim.

### Attack

This is elegant, but elegance has been a trap in this repo.

The hostile expert says:

```text
You are building a tiny causal-representation benchmark where you know the
answer. Of course a hand-designed causal compressor wins. That teaches nothing
about language or intelligence.
```

Also, exact causal-state methods have a long history: computational mechanics, predictive state representations, bisimulation, causal abstraction, information bottleneck, rate-distortion, and JEPA-like predictive representation learning. The project needs a new wedge, not a literature relabel.

The possible wedge is the kill history:

```text
Existing systems optimize reconstruction or prediction. This project can build
the benchmark that proves reconstruction compression and causal compression
separate in exactly the failure modes that killed Eklavya and CTI.
```

But that is a benchmark/theory contribution first, not a model training win.

### New Hardest Objection

If CWC is mostly mathematical, the gossip story may lag:

```text
A theorem about finite-state causal compression does not automatically
democratize intelligence or beat Arjun.
```

The theorem must point to a visible cheap-system advantage: fewer examples, fewer parameters, or less energy for the same functional robustness.

### Verdict + Next-Gate Ranking

Verdict:

```text
The project should stop asking scaling-law questions as the mainline. The
next mainline should test geometry directly: causal/functional equivalence
classes and compression under intervention.
```

Next-gate ranking after I158:

| Rank | Direction | Update |
|---:|---|---|
| 1 | CWC | Best literal interpretation of "Geometry, not Scale." |
| 2 | Renormalization | Useful if it defines multiscale maps between causal states. |
| 3 | CDMD | Math-discovery lane can help prove/search compression laws. |
| 4 | CTI fragments | Keep as negative controls for scale/proxy failure. |
| 5 | ENI | Energy becomes an evaluation axis after geometry exists. |

---

## I159: Strongest David vs Goliath Narrative From Existing Data

### Steelman

The strongest narrative cannot be "we found a power law." That died.

The strongest honest narrative from existing data is:

```text
More training made the model better at the test it was optimizing and worse at
the intelligence task. The small-lab lesson is not to buy more compute. It is
to stop compressing the wrong thing.
```

That is understandable outside ML:

```text
The model memorized flashcards instead of learning the subject.
```

But the moonshot version must go further:

```text
Small AI can beat wasteful AI by keeping only causes and throwing away surface
noise.
```

The David vs Goliath frame:

| Goliath habit | David counter |
|---|---|
| train larger models on more tokens | learn smaller causal states |
| optimize reconstruction/loss | optimize intervention-stable judgment |
| store the world in weights | retrieve/evaluate evidence when needed |
| trust smooth scaling curves | detect regime/trap/phase changes |
| measure surface fluency | measure causal action preservation |

The first stop-scrolling headline target:

```text
A laptop showed why more AI training can make a model dumber - then built a
tiny test where keeping causes beats memorizing data.
```

The stricter version:

```text
We can tell when an AI is learning the answer key instead of the rule, and we
can train a smaller system to keep the rule.
```

That narrative belongs to CWC with renormalization diagnostics, not to CTI.

### Attack

The story risks being too educational, not breathtaking.

"Memorizing flashcards is bad" is obvious. "Keep causes, not noise" is common causal-ML language. "Tiny synthetic test" sounds like a toy.

To survive gossip-magazine standards, the result must include surprise:

- A simple proxy-winning learner loses catastrophically under a hidden intervention.
- A much smaller causal compressor wins with less data/compute.
- The causal state is inspectable in a one-page diagram or short program.
- The test is generated so humans can see the rule and the trap.
- The method predicts or prevents the trap before the final held-out collapse.

The result cannot be "we built a benchmark where our metric wins." It must be "we built a benchmark where standard reconstruction learning is provably the wrong objective, and a tiny causal compressor wins for the reason the proof says."

### New Hardest Objection

The best David-vs-Goliath story may require a villain: scale. But if the benchmark is too synthetic, Goliath never entered the ring.

```text
To beat Goliath, the project must include a strong ordinary learner baseline
that looks good by conventional metrics and then loses under the exact
causal/intervention test.
```

Without that, there is no public contrast.

### Verdict + Next-Gate Ranking

Verdict:

```text
The live narrative is "causes beat memorization," not "compute follows a law."
It can be viral only if the first CWC gate visibly defeats a proxy-winning
baseline under intervention.
```

Next-gate ranking after I159:

| Rank | Direction | Narrative status |
|---:|---|---|
| 1 | CWC | Best story: small causal state beats larger memorizer. |
| 2 | CDMD | Best alternate story if it produces a real discovery. |
| 3 | Renormalization | Strong explanation story, weaker public artifact. |
| 4 | ENI | Energy-saving story possible, but needs algorithmic win. |
| 5 | CTI fragments | Negative-result story only. |

---

## I160: Attack The Top Candidate - What Kills CWC?

### Steelman

The strongest CWC proposal is:

```text
Causal World Compression: intelligence is the minimal compressed state that
preserves decisions under intervention. Small models win by learning causal
state, not by reconstructing surfaces or scaling compute.
```

First CPU gate:

```text
CWC-0: exact finite-world benchmark. Generate small environments with known
latent causal state, nuisance variables, spurious shortcuts, and intervention
shifts. Compare reconstruction/prediction learners against causal-state
compressors under equal CPU budget. Require precommitted compression,
intervention, and generalization metrics.
```

The mathematical target:

```text
Prove or empirically demonstrate a separation between reconstruction-optimal
compression and decision-optimal causal compression.
```

This is CPU-only, theory-friendly, and directly answers the kill history. It does not require GPU training. It can produce:

- a theorem or proposition;
- exact small environments;
- inspectable causal partitions;
- adversarial baselines;
- a trap detector that does not merely reuse held-out labels.

### Attack

CWC can die in at least seven ways.

1. **Toy-world triviality.** If the causal variables are explicit in the generator, the result is scripted.
2. **Prior-art absorption.** If the theorem is just information bottleneck, predictive states, or bisimulation under a new name, there is no paradigm shift.
3. **No language bridge.** If the method cannot explain byte/language failures, it becomes a separate toy project.
4. **Benchmark overfitting.** If the benchmark is designed so reconstruction must fail, the result is advocacy, not discovery.
5. **No automatic causal discovery.** If humans hand-author transformations and causal partitions, the method does not democratize intelligence.
6. **No stronger-than-validation signal.** If CWC needs a large held-out interventional set, it is just evaluation engineering.
7. **No public punch.** If the first artifact is a dense theorem without a visible demo, it will not pass the gossip-magazine test.

The kill condition should be predeclared:

```text
If CWC-0 cannot produce a separation that beats reconstruction/prediction
baselines and survives hidden-generator or held-out-family tests, CWC is not
the mainline. Demote it to theory notes and move to CDMD or renormalization.
```

### New Hardest Objection

CWC may require the thing it claims to produce:

```text
To identify causal state, the system needs interventions or invariances. But
in language and open-world intelligence, the relevant interventions are not
given. The method may become dependent on human-supplied transformation groups.
```

This is the hardest technical problem. The first gate must separate:

- supplied transformations as scaffolding;
- discovered transformations as the moonshot;
- external retrieved evidence as a source of quasi-interventions.

### Verdict + Next-Gate Ranking

Verdict:

```text
CWC survives only if narrowed, formalized, and made hostile from the first
artifact. It dies if it becomes JEPA-lite, causal-label engineering, or a toy
where the answer is embedded in the generator.
```

Next-gate ranking after I160:

| Rank | Direction | Attack-adjusted status |
|---:|---|---|
| 1 | CWC | Still #1, but only with hidden-generator/held-out-family controls. |
| 2 | Renormalization | Best fallback if CWC cannot define causal states without hand labels. |
| 3 | CDMD | Best escape if CWC/RG become toy theory with no headline artifact. |
| 4 | CTI fragments | Continue as measurement hygiene, not direction. |
| 5 | ENI | Still deferred. |

---

## I161: Final Direction Ranking

### Steelman

The final ranking should reward directions that:

- answer the pattern of ten kills;
- can run CPU-only;
- can produce mathematical artifacts;
- have a public story;
- are not just incremental ML tooling;
- can fail cleanly under hostile gates.

The strongest candidate is CWC, but only in a sharpened form:

```text
Causal World Compression = exact theory and CPU demonstrations of minimal
decision-preserving state under intervention, with memorization traps treated
as non-causal compression failures.
```

Renormalization should be a companion theory lane:

```text
Use phase/order-parameter language only when there is an actual coarse-graining
operator and predictive phase boundary.
```

CDMD should remain the optional headline lane:

```text
If the project needs a clean proof/search artifact, pick a narrow verifier-rich
math domain and search for compressed constructions.
```

CTI should not be revived as a law program:

```text
Keep `D_gap`, regime labels, locked forecasts, and baseline discipline. Retire
the CTI name from moonshot positioning.
```

ENI should wait:

```text
Measure joules after there is a geometry worth measuring. Do not start with
energy accounting.
```

### Attack

This ranking may be aesthetically coherent but strategically conservative.

The truly bold move might be CDMD: a math discovery can be verified, circulated, and remembered. CWC may spend months building a careful theory only to produce "causal representation learning but smaller."

The truly practical move might be evidence-native/retrieval-born judgment: stop trying to store world knowledge in weights and build a small judge that uses external evidence. That is not separately listed, but it should be folded into CWC as "causal compression over retrieved evidence." If CWC ignores evidence, it will repeat the closed-world trap.

The truly immediate move might be renormalization because CTI's failure directly revealed phases. But phase theory risks becoming meta-analysis of training runs rather than a new path to democratized intelligence.

### New Hardest Objection

The top direction is still not yet a concrete experiment.

```text
Until CWC has a precommit spec like CTI did, it is just the new favorite phrase.
```

The next batch must write a `CWC_PRECOMMIT_SPEC.md` or equivalent before any experiment. It must define:

- admissible worlds/tasks;
- causal-state target;
- compression metric;
- action/intervention distortion;
- baselines;
- hidden-family splits;
- theorem/proof obligations;
- verdict tokens;
- public-claim rules.

### Verdict + Final Ranking

Final score:

| Rank | Direction | Manifesto alignment | Narrative strength | CPU-only feasibility | Paradigm-shift potential | Surviving attack surface | Total | Verdict |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | CWC | 5 | 5 | 5 | 5 | 3 | 23 | MAINLINE, but only as exact causal/decision compression. |
| 2 | Renormalization | 5 | 4 | 5 | 4 | 3 | 21 | THEORY LANE; use for phase boundaries and coarse-graining. |
| 3 | CDMD | 4 | 5 | 5 | 4 | 2 | 20 | OPTIONAL HEADLINE LANE; needs a niche verifier target. |
| 4 | CTI fragments | 3 | 2 | 5 | 2 | 4 | 16 | DIAGNOSTIC INFRA; do not revive CTI branding. |
| 5 | ENI | 4 | 3 | 3 | 3 | 2 | 15 | DEFER; evaluate energy after algorithmic geometry exists. |

Decision token:

```text
PIVOT_TO_CWC_WITH_RENORMALIZATION_SUPPORT
```

Hard next gate:

```text
Write a CWC precommit spec before any implementation. The first artifact must
be CPU-only and must separate causal compression from reconstruction/proxy
compression under hidden interventions or held-out world families.
```

---

## Recommendation

**Verdict: PIVOT.**

Kill:

```text
CTI as a moonshot direction, not only CTI as smooth power law.
```

Retain:

```text
CTI's measurement discipline, `D_gap`, regime labels, blind locks, and baseline
hygiene as support tools.
```

Mainline:

```text
Causal World Compression.
```

Precise formulation:

```text
Intelligence is the shortest state that preserves correct action under
intervention. Memorization is compression of samples. Understanding is
compression of causes.
```

Renormalization role:

```text
Use RG language only to study multiscale causal-state formation and phase
boundaries between memorization and causal compression.
```

CDMD role:

```text
Keep as a parallel proof/search lane if CWC cannot produce a public artifact
fast enough.
```

W-Loop implication:

```text
Do not run new GPU training. Do not implement another CTI curve fitter. Next
work should create a CWC precommit spec and a CPU-only exact-world gate.
```

---

## What Must Change Before W-Loop B18

Minimum requirements:

1. **Write `research/CWC_PRECOMMIT_SPEC.md`.** Define the object before experiments.
2. **Use decision distortion, not reconstruction loss.** The primary metric must be action/intervention error.
3. **Require hidden-family tests.** The generator or transformation family must have held-out regimes the compressor did not see.
4. **Compare against strong boring baselines.** Reconstruction autoencoder, next-step predictor, validation early stopping, information bottleneck, simple causal oracle where allowed, and nearest-neighbor/memorization baselines.
5. **Make trap detection label-free where possible.** Use invariance/counterfactual/evidence consistency, not only held-out labels.
6. **Include a theorem/proposition target.** Even a finite-world separation theorem is better than another plot.
7. **Predeclare kill tokens.** If causal compression only wins where the answer is hand-labeled, kill or demote.
8. **Preserve narrative discipline.** No "world model" or "causal intelligence" public claims until a hidden-family CPU gate passes.

Positive token discipline:

```text
CWC_SIGNAL requires a CPU-only separation where causal compression beats
reconstruction/proxy baselines under held-out interventions.

STRONG_CWC requires the same result across at least two world families, with a
proof or exact characterization of the causal state.

MOONSHOT_CWC requires a visible small-system win: less data/compute/parameters
for equal or better intervention-robust function, plus an artifact a non-expert
can understand.
```

Kill rule:

```text
If the first CWC gate is only a hand-authored toy where causal labels are given
and all non-causal baselines are weak by construction, do not count it. Redesign
once. If the second gate repeats the same weakness, demote CWC and move to CDMD
or a strictly formal renormalization theorem.
```

---

## NARRATIVE ATTACK

### 1. Strongest "that's obvious" dismissal

```text
Of course models should learn causes instead of memorizing correlations. This
is just causal representation learning, information bottleneck, and overfitting
avoidance with new branding.
```

This attack is fair unless CWC produces a sharper object than "learn causes." The defense must be:

```text
We are not claiming causality as a slogan. We are building an exact hostile
test where reconstruction and proxy optimization provably compress the wrong
state, while a smaller decision-preserving causal state wins under hidden
intervention.
```

### 2. Strongest "that's trivial" dismissal

```text
You made a toy world where the causal variable is known, then showed that using
the causal variable beats memorizing noise.
```

This kills the direction if the first gate uses exposed causal labels or hand-tuned transformations as the answer key.

The result is nontrivial only if:

- the causal state is inferred or compressed, not handed over;
- the held-out interventions/world families are hidden during fitting;
- reconstruction/proxy learners look strong on ordinary metrics before failing;
- the causal compressor wins with fewer bits/examples/parameters;
- the theorem explains why the separation must occur.

### 3. What the result needs to BE for the narrative to be unkillable

The unkillable result:

```text
A CPU-only system learns a tiny causal state that throws away most surface data,
predicts which training runs are memorizing instead of understanding, and beats
larger reconstruction-trained baselines on hidden interventions. The causal
state is inspectable, the separation is precommitted, and the proof explains
why more proxy training makes the wrong model worse.
```

Normal-person headline target:

```text
A laptop proved why some AI gets dumber when trained harder: it memorizes the
answer key instead of the rule. The fix is a tiny model that keeps causes, not
noise.
```

Final narrative verdict:

```text
CWC is the best post-CTI direction, but only if it becomes exact causal
compression with hostile CPU gates. Anything softer is another proxy in a new
coat.
```
