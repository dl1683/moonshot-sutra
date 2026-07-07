# QUESTION LOOP - Batch 19: What If Eklavya Fails? The Pivot Playbook
Date: 2026-07-07
Iterations: 127-133
## Grounding
I read the requested local context first, in order:
1. `research/VISION.md`
2. `research/dual_loop_supervisor_checkin_12.md`
3. `research/question_loop_batch18.md`
4. `research/question_loop_batch17.md`
5. `research/work_loop_batch13.md`
6. `research/work_loop_batch14.md`
No GPU runs, training runs, benchmark runs, or experiments were performed. This batch is analysis only.
`research/work_loop_batch14.md` was read as-is before any overwrite. Its current verdict is:
```text
BLOCKED_EXACT_STUDENT_UNAVAILABLE
```
That is not a terminal Eklavya result. The exact base `HuggingFaceTB/SmolLM2-135M` checkpoint was missing locally, and the cached instruct sibling was correctly rejected as an invalid substitute. Therefore this batch prepares for both future branches rather than interpreting B14 as pass or fail.
I also attempted to read `CLAUDE.md` because the prompt says it lists the five pivot directions. No `CLAUDE.md` file is present in this checkout, and `rg --files -g 'CLAUDE.md' -g '*CLAUDE*'` returned no matches. The five directions below are therefore treated as prompt-provided, not file-verified.
External sources checked for the literature-dependent parts:
- Compute and scaling laws: https://arxiv.org/abs/1712.00409, https://arxiv.org/abs/2001.08361, https://arxiv.org/abs/2010.14701, https://arxiv.org/abs/2203.15556, https://arxiv.org/abs/2408.03314, https://arxiv.org/abs/1503.02406
- Renormalization, grokking, and phase transitions: https://arxiv.org/abs/1410.3831, https://arxiv.org/abs/2201.02177, https://arxiv.org/abs/2210.01117, https://arxiv.org/abs/2301.05217, https://arxiv.org/abs/2303.06173, https://arxiv.org/abs/2310.03789, https://arxiv.org/abs/2310.06110, https://arxiv.org/abs/2407.12332
- JEPA, causal states, and causal world models: https://arxiv.org/abs/2301.08243, https://arxiv.org/abs/2506.09985, https://arxiv.org/abs/2602.11389, https://arxiv.org/abs/2012.14228, https://arxiv.org/abs/2010.05451
- Algorithm and math discovery: https://www.nature.com/articles/s41586-022-05172-4, https://www.nature.com/articles/s41586-023-06004-9, https://www.nature.com/articles/s41586-023-06924-6, https://arxiv.org/abs/2506.13131, https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/, https://arxiv.org/abs/2307.14503
- Energy and efficiency: https://arxiv.org/abs/1907.10597, https://arxiv.org/abs/2509.20241, https://arxiv.org/abs/2511.17179
## Binding State
The invariants are not negotiable:
```text
Paradigm-shifting or nothing.
Serve the democratization moonshot.
The loop only stops when a hostile adversarial reviewer cannot knock it down.
```
The actual state entering this batch:
- `FAIL_S0_CAPACITY` is binding. Wide7 fit the train split but did not show broad held-out MCQ generalization.
- Byte-native is demoted from near-term mainline, not permanently dead.
- `PASS_DISAGREEMENT` remains fuel, not engine. It proves teacher complementarity under label-anchored audit, not a deployable router.
- SmolLM2-135M is only a terminal token-level Eklavya protocol control. It is not Sutra proof, byte proof, or compute-fair proof.
- B18's strict evidence hierarchy is binding: invalid -> fail capacity -> ordinary -> marginal -> pass -> strong -> moonshot candidate.
- B18's strict baseline board is binding: A0 through A9, with label-only, single-teacher, static teacher, uniform/entropy mix, random/shuffled routing, and learned Eklavya router all separated.
- Minimum continuation after a valid B14 pass is not moonshot evidence. It is only the right to continue.
The fork:
| Future B14 verdict | Meaning | Correct posture |
|---|---|---|
| `PASS_EKLAVYA_MECHANISM` | Eklavya has residual protocol value on a competent token-level student. | Continue as protocol, harden heavily, do not claim Sutra solved. |
| `STRONG_EKLAVYA` | The result is robust, disagreement-local, and survives stronger checks. | Treat as serious protocol line; byte-return or token-identity decision becomes urgent. |
| `MOONSHOT_CANDIDATE` | A 135M-class student approaches or beats stronger teacher policy with transfer/sample-efficiency evidence. | Public validation path opens. |
| `ORDINARY_FINE_TUNING`, `ORDINARY_KD`, `MARGINAL_EKLAVYA`, or `FAIL_EKLAVYA_MECHANISM` | No Eklavya residual worth mainline continuation. | Pivot moonshot. |
| `INVALID_EKLAVYA_TEST` or `BLOCKED_EXACT_STUDENT_UNAVAILABLE` | No legal terminal evidence. | Fix validity, rerun, do not pivot on this token alone. |
---
## Iteration 127: If B14 Passes - What Does Eklavya's Next 5 Steps Look Like?
### Steelman
Assume the strongest legal survival condition, not a vague bump:
```text
PASS_EKLAVYA_MECHANISM:
Eklavya learned/calibrated disagreement routing beats all strict non-Eklavya baselines
by >=3pp aggregate, on >=2/3 benchmarks, without held-out leakage, and with
disagreement-slice lift plus no consensus damage.
```
That would be real. It would answer the question left alive after `FAIL_SCAFFOLD` and `FAIL_S0_CAPACITY`:
```text
Does teacher disagreement contain trainable residual value when the student is already competent?
```
If yes, Eklavya survives as a protocol. The project should not immediately jump back into byte-native romance, nor should it declare victory. The next sequence has to convert a tiny terminal control into a hard-to-kill mechanism line.
Five concrete experiments, in order:
| Step | Experiment | Exact question | Pass bar | Why this order |
|---:|---|---|---|---|
| 1 | `B14R_REPLICATION_AND_AUDIT` | Does the residual survive exact rerun, seed/count audit, and full baseline board? | Same >=3pp aggregate over A0-A9, >=2/3 benchmarks, exact count deltas, disagreement-slice lift, no consensus damage, no held-out leakage. If LoRA was used, full fine-tune label-only spot-check cannot erase the residual. | A one-run pass on tiny held-out slices is continuation evidence, not a foundation. |
| 2 | `B15_TRANSFER_BOARD` | Does Eklavya transfer outside the tuned triad? | Preserve positive residual on at least one not-tuned benchmark/task family: WinoGrande, OpenBookQA, MMLU cloze, CommonsenseQA, or a held-out MCQ family not touched by router design. | If it only works on HellaSwag/PIQA/ARC-Easy, it can be benchmark engineering. |
| 3 | `B16_TEACHER_DIVERSITY_BOARD` | Is the gain teacher-disagreement structure, not SmolLM-family inheritance? | Add at least one non-shared-family teacher if locally available or cacheable: Pythia/Gemma/Phi/Mistral-small-class/Mamba once kernels work. Require teacher-ablation table and per-teacher contribution. | SmolLM2-360M -> SmolLM2-135M is too family-coupled for the original diverse-teacher story. |
| 4 | `B17_LABEL_EFFICIENCY_CURVE` | Does Eklavya learn more from less? | At 25%, 50%, and 100% label budgets, Eklavya must either match label-only with <=50% labels or beat label-only by >=5pp at matched labels on at least the best-supported slice. | Democratization is not "uses more teacher signal." It is retained gain per scarce label/compute unit. |
| 5 | `B18_BYTE_RETURN_OR_IDENTITY_DECISION` | Can the protocol return to byte-native, or must Eklavya become token-level? | Precommit one of two branches: byte-return via chain-init/cross-tokenizer bridge/functional choice-level compiler, or explicit token-level Eklavya identity with byte-native archived as future architecture research. | The manifesto cannot stay ambiguous forever. Protocol survival forces identity choice. |
The fifth step is not "train a bigger model" by default. Scaling teachers, data, and students is useful only after the mechanism survives replication, transfer, diversity, and sample-efficiency checks. Otherwise scale becomes a fog machine.
The byte-return experiment should start conservatively:
| Byte-return path | First legal test | Why |
|---|---|---|
| Functional choice-level return | Compile the Eklavya router's choice-level targets into Wide7/S0 only after a fresh label-capacity gate. | Smallest bridge from token control to byte student without pretending token logits align with bytes. |
| Cross-tokenizer bridge | Learn token-to-byte teacher distribution projections and test on frozen choice scoring before full training. | Directly attacks tokenizer mismatch, but prior art and engineering risk are high. |
| Chain-init byte rebirth | Initialize byte student from a competent token LM or aligned representation, then rerun label-only capacity. | Best chance to fix missing semantic birth, but it is a new architecture project. |
| Hybrid byte interface over token core | Keep pretrained token core, expose byte/patch ingress-egress. | Most practical but weakest original Sutra identity. |
Public posture after a plain pass:
```text
Allowed:
"A strict token-level control found residual value in disagreement-aware multi-teacher routing."
Forbidden:
"Sutra works."
"Byte-native works."
"One RTX 5090 beat SmolLM2."
"Eklavya is a moonshot."
```
### Attack
A B14 pass can still be small, fragile, and narratively weak.
If the held-out set is 48 examples per benchmark, two examples are already +4.17pp. A >=3pp criterion is count-granular. It is legal as a continuation bar, but it cannot carry public weight. The hostile reviewer will say:
```text
You got a two-example movement on tiny slices after adapting a public 2T-token BPE model.
```
They will also attack the family coupling:
```text
Your "teacher" and "student" are same-family Hugging Face checkpoints.
The router may be exploiting calibration quirks, benchmark identity, or inherited data overlap.
```
They will attack the training mode:
```text
LoRA, PEFT, KD, static teacher selection, and confidence weighting are ordinary tools.
Unless the Eklavya row beats all of them robustly, the protocol is not isolated.
```
They will attack the manifesto:
```text
The original vision was a from-scratch byte-level student trained on one local GPU.
The positive result uses an externally pretrained BPE checkpoint trained on a large GPU fleet.
```
All of that is fair. A pass is not the moonshot. It is an admission ticket to a tougher program.
### Decision
If B14 passes, Eklavya continues only as:
```text
Eklavya-as-protocol: disagreement-aware multi-teacher learning that can produce
residual retained gain in a competent student beyond ordinary adaptation baselines.
```
The next five experiments are, in order:
1. `B14R_REPLICATION_AND_AUDIT`
2. `B15_TRANSFER_BOARD`
3. `B16_TEACHER_DIVERSITY_BOARD`
4. `B17_LABEL_EFFICIENCY_CURVE`
5. `B18_BYTE_RETURN_OR_IDENTITY_DECISION`
Do not scale first. Harden first.
### NARRATIVE ATTACK
**Strongest "that's obvious" dismissal:** A competent pretrained model learned a small MCQ slice better than a weak byte model.
**Strongest "that's trivial" dismissal:** Multi-teacher KD with a learned weight heuristic beat a few underpowered baselines by a couple examples.
**What would make the narrative hard to kill:** The same router family beats all strict baselines under replication, transfers to unseen tasks, works with genuinely diverse teachers, shows label efficiency, and leaves one cheap student at inference.
### Attack On The Next Defense
The next defense will say a pass proves the engine works. It proves only that the engine fired once on a borrowed chassis. The proof of engine design starts with replication and transfer.
---
## Iteration 128: If B14 Passes - Is Token-Level Eklavya Actually A Moonshot?
### Steelman
Token-level Eklavya can become moonshot-relevant under one story:
```text
A small open student can internalize complementary teacher knowledge so efficiently
that it approaches or beats stronger teacher policy at inference cost far below
teacher ensembles, using fewer labels and no proprietary training cluster.
```
That can serve the manifesto even if the substrate is BPE rather than bytes. The Vision says the mechanisms are replaceable and the sacred outcomes are intelligence, improvability, democratized development, data efficiency, and inference efficiency. A token-level Eklavya that gives small models surgical learning from teacher disagreement could serve those outcomes better than a nonfunctional byte-native identity project.
A real David-vs-Goliath story is possible:
```text
The little student did not copy the big teacher.
It learned where the big teachers disagreed, kept the useful parts, and ran alone.
```
That line is repeatable by a normal person. It contrasts with ordinary fine-tuning because the key object is not more data or bigger parameters; it is disagreement geometry.
The moonshot version would require all of these:
| Requirement | Why it matters |
|---|---|
| Residual over all strict baselines | Separates Eklavya from fine-tuning, KD, ensembles, and route complexity. |
| Disagreement-local lift | Shows the thesis is teacher disagreement, not benchmark identity. |
| Sample-efficiency curve | Connects to democratization and learning more from less. |
| Teacher diversity | Prevents same-family checkpoint laundering. |
| Single student at inference | Converts ensemble teacher cost into cheap retained gain. |
| Transfer outside tuned benchmarks | Prevents MCQ overfit narrative. |
| Open reproducibility on a single high-end consumer GPU | Makes democratization visible. |
| Byte-return plan or explicit token identity | Ends the unresolved Sutra/Eklavya identity split. |
If a 135M-class student after Eklavya beats a static 360M/600M teacher policy on aggregate, with no teacher ensemble at inference, that is no longer a normal KD paper. It becomes a claim about compressing disagreement structure into a cheaper learner.
### Attack
The hostile reviewer from B18 is still right:
```text
This is a rigorous negative-results repo importing a pretrained BPE student.
```
A token-level B14 pass does not erase that. SmolLM2-135M is not an underdog in training history. It is a public model trained on massive external compute. The project cannot use its post-pretraining adaptation result as evidence that one RTX 5090 can create a champion from scratch.
The prior-art field is also crowded. Knowledge distillation, multi-teacher KD, instance-level teacher weighting, confidence-aware teacher selection, meta-routing, LoRA, QLoRA, and ensemble distillation all exist. The phrase "disagreement routing" is not automatically novel. If the result is a small residual on tiny splits, the paper title becomes:
```text
Careful Baselines for Multi-Teacher Distillation on SmolLM2-135M
```
That may be useful. It is not paradigm-shifting.
There is also a narrative trap: the more the project celebrates token-level success, the more it looks like it abandoned the distinctive byte substrate after failure. The reviewer will not care that the Vision allowed mechanism replacement unless the replacement is better in outcome space.
Token-level Eklavya becomes paradigm-shifting only if it changes the answer to a common question:
```text
How should small models learn from larger models?
```
It does not become paradigm-shifting by changing the answer to:
```text
Can SmolLM2 be fine-tuned?
```
### What Would Make It Paradigm-Shifting?
The public moonshot bar should be:
```text
One 135M-class open student, trained locally with a precommitted disagreement protocol,
beats every ordinary adaptation/KD/ensemble baseline and matches or beats a stronger
teacher policy on held-out tasks, while using materially fewer labels or less inference
compute than the alternatives.
```
Concrete thresholds:
| Tier | Public meaning | Required evidence |
|---|---|---|
| `PASS_EKLAVYA_MECHANISM` | Continue privately. | >=3pp aggregate over A0-A9 and >=2/3 benchmarks, disagreement-slice lift. |
| `STRONG_EKLAVYA` | Serious research claim. | >=5pp over strongest baseline, robust seed/bootstrap, teacher diversity, transfer check. |
| `MOONSHOT_CANDIDATE` | Public moonshot candidate. | 135M-class student matches/beats stronger teacher policy or >=3pp over it on >=2/3 benchmarks; sample efficiency; unseen task transfer; one-student inference. |
| `PARADIGM_SHIFT_CANDIDATE` | Gossip-magazine headline can survive experts. | Reproduced by an outside runner or clean public harness; cost accounting; no hidden teacher ensemble; result matters outside tiny MCQ slices. |
The viral version is:
```text
A small model learned not by copying its teacher, but by studying teacher disagreement.
```
The boring version is:
```text
We tuned SmolLM2 with soft labels.
```
The difference is the strict evidence board.
### Decision
Token-level Eklavya is not a moonshot at `PASS_EKLAVYA_MECHANISM`. It becomes a moonshot candidate only after it proves:
1. robust residual over all ordinary baselines;
2. disagreement-local mechanism;
3. teacher diversity;
4. sample efficiency;
5. transfer;
6. one-student inference;
7. either byte return or an explicit new identity.
Until then, the right token is:
```text
TOKEN_PROTOCOL_SURVIVES_NOT_MOONSHOT
```
### NARRATIVE ATTACK
**Strongest "that's obvious" dismissal:** You used a pretrained model that already had the benchmark priors your byte model lacked.
**Strongest "that's trivial" dismissal:** This is multi-teacher KD with better hygiene.
**What would make the narrative hard to kill:** A small local adaptation converts disagreement among bigger teachers into a cheaper single model that beats the teacher policy or reaches it with far fewer labels.
### Attack On The Next Defense
The next defense will say "the manifesto says mechanisms are replaceable." Correct. But replacement is allowed only if the new mechanism better serves the sacred outcomes. Token-level convenience is not enough.
---
## Iteration 129: If B14 Fails - Which Pivot Direction Is The Biggest Swing?
### Steelman
Assume B14 returns one of the terminal negative tokens:
```text
FAIL_EKLAVYA_MECHANISM
ORDINARY_FINE_TUNING
ORDINARY_KD
MARGINAL_EKLAVYA
```
Then the mainline KD moonshot is dead. The project should not spend the preserved FMD repair shot on another weak byte student. The honest question becomes:
```text
Which new direction best serves "Intelligence = Geometry, not Scale" on one RTX 5090?
```
The five prompt-provided pivots:
1. CTI - Compute Thermodynamics of Intelligence: `D(C) = D_inf + k C^(-alpha)`
2. Renormalization Theory of Phase Transitions - grokking/stat-mech phase diagrams
3. CWC - Causal World Compression: JEPA + causal states
4. CDMD - Compression-Driven Math Discovery
5. ENI - Energy-Native Intelligence
Honest ranking by combined moonshot fit, demonstrability, novelty wedge, and salvage from Eklavya:
| Rank | Direction | Why |
|---:|---|---|
| 1 | CTI | Best continuation of "geometry not scale" with a one-GPU empirical law program. It can absorb Eklavya's failures as data about compute, distortion, student capacity, and residual transfer. Prior art is heavy, but the target can be narrowed to a precommitted compute-distortion law across interventions rather than another LLM scaling curve. |
| 2 | Renormalization | Strongest pure theory fit. "Predict when memorization becomes understanding" is powerful. Single-GPU toy experiments are feasible. But the RG analogy already has prior art and becomes empty unless made mathematically precise. |
| 3 | CDMD | Best gossip headline if it works: "one laptop GPU discovered new math/code." Single-GPU verification loops are plausible. But DeepMind has AlphaTensor, AlphaDev, FunSearch, AlphaEvolve, and 2025-2026 large-scale math exploration, so competition is brutal. |
| 4 | CWC | Strong manifesto fit if intelligence is compression of causal state, not token prediction. But JEPA/world models are now a major Meta/DeepMind/World Labs battlefield. One RTX 5090 can do small causal environments, not a world-model headline. |
| 5 | ENI | Democratization and energy efficiency matter, but the direction is hardware/measurement heavy. Without new substrate, it risks becoming "we measured joules per token" rather than energy-native intelligence. |
### Attack
Each pivot has a hostile review.
#### CTI
Steelman:
```text
Find a universal relation between compute spent and distortion reduced.
If the law holds across model classes, tasks, and train/test-time regimes,
then intelligence has a geometry measurable on one GPU.
```
Hostile review:
```text
Scaling laws already exist. Kaplan, Hestness, Henighan, Chinchilla, broken scaling laws,
and test-time compute scaling cover this territory. A few small local curves do not prove
a universal thermodynamics of intelligence.
```
How CTI survives:
- Define distortion before running.
- Predict held-out compute points, not just fit seen curves.
- Compare against standard scaling-law baselines.
- Include negative cases and broken laws.
- Show that geometry-changing interventions shift `k` or `alpha` in reproducible ways.
#### Renormalization
Steelman:
```text
Grokking and delayed generalization look like phase transitions.
If we can map training runs to order parameters and predict phase boundaries,
we can explain when networks stop memorizing and start generalizing.
```
Hostile review:
```text
Calling grokking a phase transition is already common. Mehta-Schwab mapped variational RG
to RBMs. Several grokking papers already use first-order transitions, lazy-to-rich dynamics,
mechanistic progress measures, and singular learning theory. Where is your new theorem?
```
How it survives:
- Define an actual coarse-graining operator.
- Precommit order parameters.
- Predict phase boundaries before observing them.
- Show finite-size scaling or critical exponents across width/data/regularization.
- Transfer from algorithmic toy tasks to one non-toy learning failure.
#### CWC
Steelman:
```text
Intelligence is not next-token scale. It is compact causal state.
A small learner should compress observations into intervention-ready predictive states.
```
Hostile review:
```text
JEPA is Meta's field. Causal representation learning is active. World models are flooded
with capital and giant video datasets. Your one-GPU gridworld will look like a toy.
```
How it survives:
- Avoid "foundation world model" competition.
- Use tiny controlled environments where causal-state minimality can be measured exactly.
- Demonstrate compression that preserves intervention/planning performance under distribution shift.
- Build a benchmark where generative reconstruction loses to causal compression.
#### CDMD
Steelman:
```text
Mathematical discovery is naturally verifier-driven.
One GPU plus a strict evaluator can search compressed programs/constructions
and produce an artifact humans can inspect.
```
Hostile review:
```text
DeepMind already did AlphaTensor, AlphaDev, FunSearch, and AlphaEvolve.
If you do not find a new result, you are recreating a small open-source FunSearch clone.
```
How it survives:
- Pick a niche where verification is cheap and prior bests are clear.
- Use compression as the actual novelty: shorter generating rules, lower description length, better proof sketches.
- Produce a human-readable construction, not just score-chasing.
- Beat a known baseline or publish a clean negative benchmark.
#### ENI
Steelman:
```text
If intelligence is bounded by energy, democratization needs energy-native algorithms,
not just smaller models. Joules should be first-class in the objective.
```
Hostile review:
```text
Without neuromorphic hardware or a new physical substrate, you are doing green-AI accounting.
Landauer, neuromorphic computing, energy-based models, and AI power measurement are all mature enough
that a single-GPU software project is unlikely to be fundamental.
```
How it survives:
- Make energy a controlled optimization target, not a post-hoc metric.
- Compare accuracy-distortion-per-joule frontiers.
- Show a reproducible algorithmic gain over compute-equivalent baselines.
- Avoid hardware claims unless measured.
### Cross-Direction Scoreboard
Scores are 1-5, where 5 is strongest.
| Direction | Manifesto fit | Gossip headline | RTX 5090 demonstrability | Prior-art breathing room | Eklavya salvage | Total |
|---|---:|---:|---:|---:|---:|---:|
| CTI | 5 | 4 | 5 | 2 | 5 | 21 |
| Renormalization | 5 | 3 | 5 | 2 | 4 | 19 |
| CDMD | 4 | 5 | 4 | 2 | 3 | 18 |
| CWC | 5 | 4 | 3 | 1 | 2 | 15 |
| ENI | 4 | 3 | 3 | 2 | 3 | 15 |
The numerical total slightly underrates renormalization's intellectual upside and overrates CTI's novelty. But CTI remains the best default because it converts the whole Eklavya arc into evidence rather than abandoning it. It asks:
```text
What was the compute-distortion geometry of every failed intervention?
```
That is a disciplined pivot rather than a shiny-object reset.
### Decision
If B14 fails, default pivot:
```text
CTI_MAINLINE
```
with:
```text
RENORMALIZATION_THEORY_LANE
CDMD_OPTIONAL_HEADLINE_LANE
```
CWC and ENI should not be the first pivot unless the project explicitly chooses a clean-sheet world-model or energy-measurement program. They are too externally crowded and too likely to demand resources the repo does not have.
### NARRATIVE ATTACK
**Strongest "that's obvious" dismissal:** After KD failed, the repo picked a new grand theory name and started over.
**Strongest "that's trivial" dismissal:** CTI is just scaling laws with a thermodynamics coat of paint.
**What would make the narrative hard to kill:** The pivot reuses Eklavya's kill ledger as the first dataset, precommits a compute-distortion law, predicts held-out runs, and kills itself if the law does not forecast better than standard scaling baselines.
### Attack On The Next Defense
The next defense will say CTI has a bigger Nobel upside. Upside is not evidence. The first CTI artifact must be a falsifiable one-GPU law, not a manifesto.
---
## Iteration 130: If B14 Fails - What Can Be Salvaged From The Eklavya Arc?
### Steelman
The Eklavya arc is not worthless if it fails. It produced several durable assets:
| Salvage item | Keep? | Why |
|---|---|---|
| Kill discipline | Yes | The repo repeatedly killed favored directions instead of laundering proxy wins. This is rare and valuable. |
| Evidence hierarchy | Yes | Invalid/fail/ordinary/marginal/pass/strong/moonshot candidate should carry into any pivot. |
| Strict baseline board | Yes | A0-A9 is a reusable anti-self-deception pattern. |
| Dual-loop methodology | Yes | Q-loop/W-loop separation creates adversarial pressure and implementation feedback. |
| PASS_DISAGREEMENT dataset | Yes, bounded | It is evidence of teacher complementarity and oracle headroom, not a validated router. |
| Capacity-gate doctrine | Yes | "Can the student learn labels first?" should become a universal preflight rule. |
| MCQ forced-choice harnesses | Yes | Useful as small functional probes for CTI and renormalization, even if KD dies. |
| Negative-result ledger | Yes | Publishable internally, maybe externally after cleanup, as a case study in proxy failure. |
| FMD repair shot | Archive | Do not spend it if Eklavya fails. Preserve only as a historical method. |
| Byte-native student line | Archive, not mainline | It failed current capacity gates. It needs a new birth mechanism before re-entry. |
The publishable negative result is not:
```text
Byte models cannot work.
```
The publishable negative result is narrower:
```text
In this repo's 121M byte-student setting, multiple teacher/KD/proxy objectives failed
to produce held-out MCQ function, and the strongest byte checkpoint fit train labels
without broad generalization. Capacity gates and strict baselines prevented proxy
success from being mistaken for intelligence.
```
That can be useful to the field if written with humility and artifacts:
- exact student configs;
- exact data splits;
- train vs held-out behavior;
- objective kill table;
- what was not tested;
- why no broad impossibility claim follows.
### Attack
The salvage story can become self-flattery.
Process is not the moonshot. A hostile reviewer will say:
```text
You built an elaborate self-review machine around a line of work that never produced
a positive functional student result.
```
That is true. The dual-loop methodology is valuable only if it now accelerates better decisions. It is not valuable if it becomes a ritual that makes every pivot feel profound.
The negative results are also not automatically publishable. They are small, local, and entangled with implementation details:
- 288 train examples is tiny.
- 48 held-out examples per benchmark is count-granular.
- S0/Wide7 training history is local.
- Some failures used weak scaffolds later killed as invalid substrates.
- The teacher portfolio changed midstream.
- Exact external reproducibility may be hard if checkpoints/data/cache state are not packaged.
The repo can honestly publish a negative-results technical report. It cannot claim a general theorem about byte-native KD.
### Transfer To Each Pivot
| Eklavya asset | CTI use | Renormalization use | CDMD use | CWC use | ENI use |
|---|---|---|---|---|---|
| Train/held-out divergence | Compute-distortion curves and capacity failure points. | Memorization vs generalization phase markers. | Less direct. | Less direct. | Energy wasted on non-generalizing updates. |
| PASS_DISAGREEMENT | Distortion headroom and oracle residual. | Teacher disagreement as competing basin signal. | Evaluator diversity analogy. | Causal-state ambiguity signal. | Selective compute allocation. |
| Strict baselines | Standard for all future claims. | Standard for phase-diagram predictions. | Essential for search-agent claims. | Essential for world-model comparisons. | Essential for joule-frontier claims. |
| Dual-loop | Falsification harness for law fitting. | Theory/experiment split. | Generator/evaluator adversarial protocol. | Causal claim audit. | Measurement and claim audit. |
| Byte failure | Evidence against naive substrate loyalty. | Case study of memorization without transfer. | Not useful except process. | Not useful except process. | Energy cost of wrong representation. |
### Worth Keeping
Keep these as first-class repo artifacts after a failure:
1. `research/NEGATIVE_RESULTS_LEDGER.md` or equivalent index.
2. A terminal Eklavya postmortem that separates objective kills, scaffold kills, capacity kills, and protocol test.
3. B18 evidence hierarchy as the default claim taxonomy.
4. Capacity-gate rule: no advanced objective until label-only capacity passes.
5. Baseline-board rule: no mechanism claim until ordinary baselines are beaten.
6. Prompt discipline: every new moonshot has a precommitted hostile-review section.
### Sunk Cost
Stop carrying these as active assumptions if B14 fails:
- Eklavya is the main moonshot.
- Teacher disagreement is useful engine rather than only fuel.
- Byte-native is near-term mainline.
- FMD deserves a preserved "one more shot."
- SmolLM2 token success can be recovered by changing the router.
- The 288-example MCQ format is sufficient for public claims.
### Decision
If B14 fails, salvage the discipline and artifacts, not the identity.
Correct token:
```text
EKLAVYA_ARC_SALVAGED_AS_FALSIFICATION_INFRASTRUCTURE
```
Not:
```text
EKLAVYA_STILL_ALIVE_SOMEHOW
```
### NARRATIVE ATTACK
**Strongest "that's obvious" dismissal:** You learned that weak students and tiny data do not produce magic.
**Strongest "that's trivial" dismissal:** The only transferable result is "run baselines."
**What would make the narrative hard to kill:** The pivot starts by importing the exact Eklavya kill ledger as falsification data and uses the same strict baseline culture to kill or validate the next thesis quickly.
### Attack On The Next Defense
The next defense will say the dual-loop itself could be a methodology paper. Maybe later. First it must produce a successful pivot or at least a cleaner negative result in a second domain.
---
## Iteration 131: The CTI Direction Deep-Dive
### Steelman
CTI asks for a law:
```text
D(C) = D_inf + k C^(-alpha)
```
where `D` is distortion and `C` is compute. The phrase "thermodynamics" earns its keep only if the law is predictive, not decorative.

On one RTX 5090, the demonstration should not attempt frontier-LM scaling. It should run a controlled compute-distortion lab:

| Component | Concrete choice |
|---|---|
| Distortion `D` | Precommit a vector, then a scalar: held-out error, NLL/BPB, calibration error, train/held-out gap, and task-specific distortion such as MCQ margin error or causal-state prediction error. |
| Compute `C` | Report FLOPs estimate, wall-clock GPU seconds, peak VRAM, trainable parameters, and optionally joules if measured. Use FLOPs for law fitting; keep wall-clock/joules for democratization accounting. |
| Model families | Tiny byte transformers, tiny token transformers, Pythia-70M/160M or SmolLM-class adaptation if cached, shallow MLP/Transformer algorithmic tasks, and one non-language control. |
| Tasks | MCQ forced-choice slices, synthetic algorithmic tasks, modular arithmetic/grokking tasks, compression/reconstruction tasks, and one causal-state/world-model toy environment. |
| Compute schedules | Log-spaced budgets: 10, 30, 100, 300, 1k, 3k update-equivalents where feasible; label budgets 25/50/100%; test-time compute schedules such as best-of-N or verifier search. |
| Interventions | Architecture change, data quality change, label-only vs teacher signal, routing vs static policy, train-time vs test-time compute allocation. |
| Prediction test | Fit on early/small budgets, predict later/held-out budgets and one held-out intervention family. |

The minimum one-GPU CTI result:
```text
Given the first 20-30% of compute points, CTI predicts the later distortion curve
and identifies which intervention changes the exponent or only shifts the constant.
```

That would be useful because Eklavya's failure pattern already contains exactly the right kind of data:
```text
proxy loss moved, train accuracy moved, held-out functional distortion did not.
```

CTI can turn that into a law:
```text
Which compute spend reduces real held-out distortion, and which compute spend only reduces proxy distortion?
```

### Demonstration Design On One GPU
First CTI board:

| Axis | Levels |
|---|---|
| Student birth | random tiny, pretrained small token, byte-trained Wide7/S0 if available |
| Objective | label-only, proxy/KD, disagreement target, reconstruction/NLL |
| Compute | log-spaced step budgets |
| Data | 25/50/100% label budgets, plus shuffled-label control |
| Evaluation | held-out accuracy, margin, calibration, train/held gap |

Precommitted law forms:
```text
Power law:         D(C) = D_inf + k C^(-alpha)
Broken power law:  D(C) = D_inf + k C^(-alpha_1) * transition(C; tau, beta)
Null:              no stable extrapolation beats a naive monotone baseline
```

Baseline forecasters:
- naive last-point extrapolation;
- ordinary power-law fit on proxy loss only;
- Chinchilla/Kaplan-style loss-only fit where applicable;
- per-task independent fit with no shared structure;
- random intervention ranking.

Pass bar:
```text
PASS_CTI_LAW_0:
CTI predicts held-out distortion at unseen compute budgets better than all
baseline forecasters on >=2/3 task families, and correctly classifies at least
one intervention as "constant shift" vs "exponent shift" before full-budget
results are observed.
```

Moonshot candidate bar:
```text
CTI predicts which low-compute intervention will dominate before running it to
completion, across language and non-language tasks, with enough reliability that
the repo can save real compute by following the law.
```

### Headline
Normal-person headline:
```text
A laptop predicted which AI training ideas were worth the electricity before they finished training.
```

Expert headline:
```text
A precommitted compute-distortion law forecast held-out functional improvement,
not just training loss, across model births and objectives on one GPU.
```

This is more repeatable than "compute thermodynamics" and less overclaimed than "universal law of intelligence."

### Prior Art And Novelty
Prior art is heavy:
- Hestness et al. showed empirical power-law scaling across domains.
- Kaplan et al. fit language-model loss as a function of model size, data, and training compute.
- Henighan et al. extended autoregressive scaling to images, video, multimodal modeling, and math.
- Chinchilla corrected compute-optimal allocation between parameters and tokens.
- Test-time compute scaling shows inference compute can sometimes outperform model scaling when allocated adaptively by difficulty.
- Information bottleneck and rate-distortion theory already connect compression, relevant information, and distortion.

So CTI is not novel if it says:
```text
Loss follows a power law with compute.
```

CTI could be novel if it says:
```text
For small-resource intelligence research, functional distortion obeys a
precommitted compute law that distinguishes proxy improvement from real
generalization, and geometry-changing interventions alter the law's parameters
predictably.
```

The wedge is not scale. The wedge is:
```text
functional distortion + intervention taxonomy + held-out prediction + one-GPU decision value
```

### Hardest Part
Hardest part is distortion definition, not raw experimentation.

If `D` is just validation loss, CTI becomes scaling laws. If `D` is a basket of hand-picked metrics, the hostile reviewer says the law was tuned to the repo. If `D` includes accuracy, NLL, calibration, train/held gap, and margin, the law may be too noisy.

The second-hardest part is narrative. "Thermodynamics" invites ridicule unless the artifact uses thermodynamic language sparingly and only after the measured objects exist.

Theory is third. A theory can come after the empirical law. The first job is to prove that the law predicts anything useful.

### Attack
The hostile reviewer will say:
```text
This is scaling laws with worse data and a grander title.
```

They will be right if the CTI batch only fits curves after seeing them. They will also attack the sample size:
```text
One GPU means small models, tiny tasks, and high noise. Universal-law claims are absurd.
```

They will attack Eklavya salvage:
```text
You are retrofitting a theory to explain why the old project failed.
```

They will attack compute accounting:
```text
FLOPs, wall-clock, joules, data quality, and pretrained checkpoint history are
different currencies. Combining them into `C` is not thermodynamics.
```

These attacks kill a manifesto. They do not kill a precommitted forecasting program.

### Decision
If B14 fails, CTI should become the main pivot, but only under this first artifact:
```text
research/CTI_PRECOMMIT_SPEC.md
```

Required contents:
1. Distortion definition before any new run.
2. Compute accounting standard.
3. Model/task grid small enough for one GPU.
4. Baseline forecasters.
5. Prediction protocol.
6. Kill tokens: `INVALID_CTI`, `NO_PREDICTIVE_LAW`, `PROXY_ONLY_LAW`, `PASS_CTI_LAW_0`, `STRONG_CTI`.

No public CTI language until `PASS_CTI_LAW_0`.

### NARRATIVE ATTACK
**Strongest "that's obvious" dismissal:** More compute reduces loss with a power law. Everyone knows this.

**Strongest "that's trivial" dismissal:** You fitted curves after the fact and renamed validation error "distortion."

**What would make the narrative hard to kill:** CTI must predict held-out functional distortion and intervention value before full-budget runs complete, beating naive scaling-law baselines and saving real one-GPU compute.

### Attack On The Next Defense
The next defense will say universal laws start with small systems. True, but small systems only matter if the law predicts withheld data. Do not ask for belief. Show forecast error.

---
## Iteration 132: The Renormalization Direction Deep-Dive
### Steelman
Renormalization is attractive because the Eklavya failure was not "loss did not move." It was:
```text
train function moved; held-out function did not.
```

That is exactly the language of phases:
- memorization phase;
- proxy-improvement phase;
- representation-formation phase;
- generalization phase.

The demonstration on one GPU should not claim "RG explains deep learning." It should ask a narrower question:
```text
Can we predict the boundary between memorization and generalization from precommitted order parameters?
```

### Demonstration Design On One GPU
Start with small systems where phases are visible and cheap:

| Task | Why |
|---|---|
| Modular addition / group composition | Direct grokking lineage; Fourier/progress measures can be inspected. |
| Sparse parity / algorithmic rules | Clean memorization-vs-rule separation. |
| Small synthetic MCQ tasks with known rule | Bridge to Eklavya-style forced choice. |
| Tiny language/byte task | Tests whether the phase machinery survives a non-toy text surface. |
| Wide7/S0 train/held artifacts | Historical case study: train absorption without broad generalization. |

Models:
- two-layer MLPs;
- tiny transformers;
- one-layer transformer for algorithmic tasks;
- tiny byte transformer if cheap;
- frozen-representation probe vs full fine-tune to mimic Eklavya failure modes.

Control axes:
- width;
- depth;
- data fraction;
- weight decay;
- optimizer and learning rate;
- label noise;
- training steps;
- initialization scale.

Order parameters to precommit:

| Order parameter | Meaning |
|---|---|
| Train/test gap | Memorization vs generalization. |
| Feature movement from initialization | Lazy-to-rich transition signal. |
| Effective rank / participation ratio of representations | Compression/coarsening signal. |
| Weight norm and margin norm | Regularization and circuit cleanup signal. |
| Fourier/circuit progress measure on modular tasks | Mechanistic progress, not just accuracy. |
| Hessian or sharpness proxy | Basin change signal. |
| Mutual information or reconstruction entropy proxy | Information compression signal. |
| Student-teacher disagreement absorption | Eklavya-specific competing-basin signal. |

Precommitted target:
```text
Predict the critical compute/data/regularization boundary at which held-out
generalization begins, before seeing the full training trajectory.
```

Pass bar:
```text
PASS_RENORM_PHASE_0:
Across >=2 algorithmic task families and one Eklavya-derived functional probe,
the precommitted order parameters predict the generalization transition earlier
than held-out accuracy alone, and the predicted phase boundary transfers across
at least one width or data-scale change.
```

### Is The RG Analogy Real Or Metaphor?
It is metaphor until these objects exist:

| RG object | ML analogue that must be made explicit |
|---|---|
| Coarse-graining map | A deterministic or learned map from fine model/representation variables to coarser variables. |
| Relevant variables | Order parameters that predict generalization phase boundaries. |
| Irrelevant variables | Hyperparameters/features whose changes wash out under coarse-graining. |
| Fixed point | Stable representation/training behavior under repeated coarse-graining. |
| Critical exponents | Scaling of transition time/error around width/data/regularization boundary. |
| Universality class | Multiple tasks/models sharing phase behavior after rescaling. |

Mehta-Schwab gives legitimacy for mapping variational RG to RBM-like deep architectures, but it does not automatically justify every transformer training analogy. Grokking papers give real phase-transition language, mechanistic progress measures, lazy-to-rich transitions, and first-order-transition models. That raises the bar. The repo must add prediction or a new coarse-graining operator.

### Prior Art
The prior-art stack is strong:
- RG/deep-learning mapping exists in RBM form.
- Grokking was introduced as delayed generalization after overfitting.
- Omnigrok extends the phenomenon beyond purely algorithmic data.
- Mechanistic progress-measure work reverse-engineers modular addition and shows continuous progress under sudden accuracy change.
- Lazy-to-rich dynamics and first-order phase-transition papers already make the phase-transition analogy concrete.
- Theoretical modular-addition grokking work connects kernel-like early behavior to later feature-learning behavior.

Therefore novelty cannot be:
```text
Grokking is a phase transition.
```

Novelty must be:
```text
Here is a concrete coarse-graining/operator/order-parameter system that predicts
phase boundaries and explains a real small-model failure mode.
```

### Does This Build On Eklavya?
Partially.

It uses:
- the train/held-out divergence from Wide7;
- capacity-gate doctrine;
- MCQ forced-choice probes;
- teacher disagreement as competing signal;
- the evidence hierarchy and baseline discipline.

It does not use:
- most KD code;
- FMD as a live objective;
- the byte-native mission as mainline.

So this is a partial continuation, not a clean continuation. CTI absorbs more of the old arc. Renormalization explains the old arc better if the phase machinery works.

### Headline
Normal-person headline:
```text
A laptop predicted when an AI stops memorizing and starts understanding.
```

Expert headline:
```text
Precommitted order parameters predicted grokking/generalization phase boundaries
across small model families and a real failed MCQ adaptation case.
```

This headline is powerful, but only if it predicts. If it merely plots phases after the fact, it is not a headline.

### Hardest Part
The hardest part is making the RG analogy rigorous. The experiment is feasible. The narrative is clear. The theory is the bottleneck.

Minimum rigor:
1. Define the coarse-graining operator.
2. Show it can be applied repeatedly.
3. Show some variables become irrelevant under it.
4. Show at least one order parameter predicts phase boundary.
5. Show finite-size scaling or a transfer rule across width/data.

Without those, the project should call it:
```text
phase-diagram learning dynamics
```
not renormalization.

### Attack
The hostile reviewer will say:
```text
You replaced KD jargon with physics jargon. The field already knows grokking can
look like a phase transition.
```

They will attack toy tasks:
```text
Modular addition is not intelligence. You can make pretty phase diagrams while
learning nothing about real models.
```

They will attack post-hoc interpretation:
```text
Order parameters chosen after seeing transition curves are not predictions.
```

They will attack the Eklavya bridge:
```text
Wide7 did not grok; it overfit. Explaining that with phase language does not
create a new result.
```

All valid. The defense must be precommitted prediction.

### Decision
Renormalization should be the theory lane, not the default main pivot, unless the project wants a theory-first reset.

Token:
```text
RENORMALIZATION_THEORY_LANE_WITH_PREDICTIVE_PHASE_BAR
```

First artifact:
```text
research/RENORMALIZATION_PHASE_PRECOMMIT.md
```

It should define:
- tasks;
- model families;
- order parameters;
- coarse-graining operator;
- phase-boundary prediction protocol;
- prior-art boundary;
- kill tokens.

### NARRATIVE ATTACK
**Strongest "that's obvious" dismissal:** Grokking and phase transitions have already been studied.

**Strongest "that's trivial" dismissal:** You made toy phase diagrams and called them intelligence.

**What would make the narrative hard to kill:** A real coarse-graining/operator framework predicts generalization phase boundaries across task/model scales before full training completes, including one Eklavya-derived failure mode.

### Attack On The Next Defense
The next defense will say physics analogies often become rigorous later. Maybe, but this repo cannot spend another arc on metaphor. The first renormalization batch must define the operator or downgrade the name.

---
## Iteration 133: What Would The Adversarial Fresh-Eyes Reviewer Say About A Pivot?
### Steelman
A fresh reviewer could respect the repo if the pivot is presented as:
```text
Disciplined researchers killed a direction under precommitted gates, preserved
the evidence, and pivoted to the next hypothesis using the old failures as
calibration data.
```

They would respect:
- B13's honest `FAIL_S0_CAPACITY`;
- B14's refusal to substitute the instruct checkpoint;
- B18's baseline board and evidence hierarchy;
- a clear terminal postmortem;
- a pivot precommit that does not reuse old labels as new evidence;
- no public moonshot claims before a positive result.

The repo can survive scrutiny if the pivot looks like continuity of method, not continuity of branding.

Correct repo shape after a failed valid B14:

| Artifact | Purpose |
|---|---|
| `research/EKLAVYA_TERMINAL_POSTMORTEM.md` | Cleanly separate objective kills, scaffold kill, capacity kill, and protocol verdict. |
| `research/NEGATIVE_RESULTS_LEDGER.md` | Index every kill token, artifact, and what claim it does or does not falsify. |
| `research/PIVOT_DECISION_B14.md` | One-page decision: why B14 failed, why Eklavya mainline ended, why CTI is next. |
| `research/CTI_PRECOMMIT_SPEC.md` | Defines the next law, metrics, baselines, and kill tokens before work begins. |
| `research/CLAIM_TAXONOMY.md` | Imports B18 hierarchy for future directions. |
| `research/BASELINE_BOARD_STANDARD.md` | Generalizes A0-A9 into a reusable baseline pattern. |
| `research/archive/eklavya/` or index links | Archive, do not erase. |

If B14 passes instead, the repo shape changes:

| Artifact | Purpose |
|---|---|
| `research/EKLAVYA_PROTOCOL_SURVIVAL_PLAN.md` | The five-step hardening plan from I127. |
| `research/EKLAVYA_PUBLIC_CLAIM_RULES.md` | What can and cannot be said after token-level survival. |
| `research/BYTE_RETURN_OR_TOKEN_IDENTITY_DECISION.md` | Forces the identity decision after replication/transfer. |

If B14 remains blocked, the repo should not pivot:
```text
BLOCKED_EXACT_STUDENT_UNAVAILABLE is an operational status, not scientific evidence.
```

### Attack
The fresh reviewer may still see:
```text
A repo with no finished results in any direction.
```

That attack is dangerous because it is partly true. The work has many loops, many tokens, and many documents. It has not yet produced a public positive result. The pivot can look like:
```text
Researchers wasted months on KD and are now chasing a different shiny object.
```

They will especially attack if:
- B14 never actually ran and the repo pivots anyway;
- CTI starts with manifesto prose instead of precommitted metrics;
- Eklavya files remain scattered with no terminal index;
- old claims are quietly softened instead of killed;
- the new direction inherits "Nobel-track" language before evidence;
- there is no first falsifiable experiment;
- source links are broad but not connected to a specific novelty boundary.

They will say:
```text
The repo's skill is generating frameworks faster than results.
```

That is the sentence the pivot must be designed to defeat.

### What The Pivot Must Look Like
The pivot must be visually and logically boring:

1. Terminal verdict first.
2. Negative ledger second.
3. Salvage map third.
4. New precommit fourth.
5. Only then new experiments.

Do not open with:
```text
Introducing CTI, the universal thermodynamics of intelligence.
```

Open with:
```text
Eklavya failed to show residual protocol value under the terminal control. We
therefore stop KD as mainline. The next hypothesis is that the common structure
under the failures is compute-distortion geometry. Here is the precommitted law
and the exact way it can fail.
```

The pivot must preserve labels:

| Old object | New status |
|---|---|
| Eklavya | Archived mainline unless B14 passes. |
| Byte-native | Future architecture research, not current proof path. |
| FMD | Historical repair shot, not live option after B14 fail. |
| `PASS_DISAGREEMENT` | Oracle headroom dataset, not mechanism proof. |
| Dual-loop | Process carried forward, not success claim. |
| Strict baselines | Mandatory inherited standard. |

The fresh reviewer must be able to answer in five minutes:
```text
What died?
What survived?
Why this pivot?
What would kill the pivot?
What is the first test?
```

If any answer is fuzzy, the pivot is not ready.

### Decision
If B14 fails, the correct public-internal positioning is:
```text
Disciplined kill and falsification-informed pivot.
```

But only if the repo contains:
1. terminal postmortem;
2. negative ledger;
3. CTI precommit;
4. no live Eklavya repair loopholes;
5. exact first-pass kill criteria for CTI.

Without those, the reviewer will fairly call it:
```text
unfinished repo chasing a new shiny object.
```

### NARRATIVE ATTACK
**Strongest "that's obvious" dismissal:** The researchers failed at KD and renamed the project.

**Strongest "that's trivial" dismissal:** The only finished product is a trail of markdown files.

**What would make the narrative hard to kill:** The repo shows a clean terminal decision, a negative-results ledger, a salvage map, and a new precommitted CTI forecasting test that can fail quickly.

### Attack On The Next Defense
The next defense will say the process itself is rare. It is. But rarity is not impact. The process must now create either a positive result or a cleaner kill in less time than the Eklavya arc consumed.

---
## Batch 19 Final Playbook
Current B14 file status:
```text
BLOCKED_EXACT_STUDENT_UNAVAILABLE
```

That means no terminal scientific decision yet.

If B14 later returns `PASS_EKLAVYA_MECHANISM`:
```text
Eklavya survives as a protocol, not as Sutra proof.
Next: replicate, transfer, diversify teachers, prove label efficiency, then
decide byte-return versus token-level identity.
```

If B14 later returns `STRONG_EKLAVYA` or `MOONSHOT_CANDIDATE`:
```text
Open a public validation path, but still keep claims bounded by the exact tier.
The public story is retained disagreement gain in one cheap student, not
from-scratch byte victory unless byte-return evidence exists.
```

If B14 later returns `ORDINARY_FINE_TUNING`, `ORDINARY_KD`, `MARGINAL_EKLAVYA`, or `FAIL_EKLAVYA_MECHANISM`:
```text
End Eklavya as mainline. Archive FMD. Preserve the negative results and dual-loop
discipline. Pivot default to CTI with renormalization as theory lane and CDMD as
optional headline lane.
```

First pivot artifact after a valid fail:
```text
research/CTI_PRECOMMIT_SPEC.md
```

First fresh-reviewer requirement:
```text
The repo must look like a disciplined kill and precommitted pivot, not a new brand launch.
```

Final hostile statement:
```text
Eklavya gets no more narrative credit after B14. A pass earns replication and
transfer; a fail earns a clean archive. The next moonshot must inherit the
evidence discipline, not the excuses.
```



