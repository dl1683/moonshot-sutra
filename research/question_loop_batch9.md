# QUESTION LOOP - Batch 9: Evidence-Native v0 Post-Mortem + v1 Design Gate

Date: 2026-07-07

Iterations: 57-63

## Grounding

I read the requested context in order:

1. `research/VISION.md`
2. `research/DEEP_RETHINK.md`
3. `research/question_loop_batch8.md`
4. `research/dual_loop_supervisor_checkin_4.md`
5. `research/work_loop_batch6.md`
6. `C:/sutra_fast/evidence_native/metrics.json`
7. `C:/sutra_fast/evidence_native/closed_book_control/metrics.json`
8. `C:/sutra_fast/evidence_native/teacher_mixed/metrics.json`
9. `code/evidence_native_sutra.py`

Additional live context checked after the required reads:

- `research/dual_loop_supervisor_checkin_5.md`
- `research/STATUS.md`

Metric-source note: the prompt's `C:/sutra_fast/evidence_native/*` metrics and the later audited `research/work_loop_batch6.md` rerun differ in exact percentages because the work-loop report describes a control-repaired workspace-local run. The conclusion is invariant across both: retrieved evidence does not help the evidence-trained model, shuffled/wrong-topic controls do not collapse, and BM25 overlap beats the learned judge. This Batch uses the prompt-specified `C:/sutra_fast` metrics as the primary numbers and uses the work-loop rerun only as corroboration.

## Binding State Entering Batch 9

Evidence-Native v0 is dead as a prototype.

Primary requested metrics:

| Model | Train evidence | Eval evidence | Overall | HellaSwag | PIQA |
|---|---|---|---:|---:|---:|
| M_evidence | retrieved | retrieved | 38.18% | 25.98% | 50.39% |
| M_evidence | retrieved | none | 38.53% | 25.39% | 51.66% |
| M_evidence | retrieved | shuffled | 38.72% | 25.39% | 52.05% |
| M_evidence | retrieved | wrong_topic | 37.55% | 24.02% | 51.07% |
| M_evidence | retrieved | gold | 38.04% | 25.29% | 50.78% |
| M_none | none | retrieved | 40.28% | 27.44% | 53.12% |
| M_none | none | none | 38.38% | 26.95% | 49.80% |
| M_none | none | shuffled | 38.18% | 26.07% | 50.29% |
| M_none | none | wrong_topic | 38.67% | 26.27% | 51.07% |
| M_none | none | gold | 38.57% | 26.46% | 50.68% |
| M_teacher_mixed | retrieved + teacher KL | retrieved | 38.18% | 25.39% | 50.98% |
| M_teacher_mixed | retrieved + teacher KL | none | 37.89% | 25.29% | 50.49% |
| M_teacher_mixed | retrieved + teacher KL | shuffled | 38.62% | 26.56% | 50.68% |
| M_teacher_mixed | retrieved + teacher KL | gold | 38.87% | 25.68% | 52.05% |

Dumb baselines from the same primary metric files:

| Baseline | Overall | HellaSwag | PIQA |
|---|---:|---:|---:|
| BM25 evidence overlap ranker | 41.41% | 26.86% | 55.96% |
| Shortest candidate | 39.89% | 24.80% | 54.98% |
| Nearest-neighbor train label | 39.21% | 28.71% | 49.71% |
| Unigram frequency | 38.87% | 25.98% | 51.76% |
| Majority label | 36.57% | 23.54% | 49.61% |

The most damaging comparison remains:

```text
M_none(retrieved) - M_none(none)         = +1.90pp
M_evidence(retrieved) - M_evidence(none) = -0.34pp

M_none(retrieved) - M_none(shuffled)         = +2.10pp
M_evidence(retrieved) - M_evidence(shuffled) = -0.54pp
```

So the model trained without evidence is the better evidence user.

## Iteration 57: Anti-Signal Attack - Evidence Training Did Not Fail Quietly

### Current Strongest Position

The surviving defense after Batch 8 is:

```text
v0 was too small, too noisy, too weakly retrieved, and too architecturally simple
to judge the real evidence-native thesis.
```

That defense is partly true. v0 is a 10.07M parameter prototype with only 5.54M trainable parameters, a frozen codec, two 256-dim reasoner layers, independent candidate scoring, and 1024 training records. It is not a 121M evidence-native judge.

### Attack

The failure is not merely absence of lift. It is signed against the thesis.

The steelman says evidence-conditioned training should create a better evidence user. The result says the opposite:

```text
M_none(retrieved)     = 40.28%
M_evidence(retrieved) = 38.18%
gap                  = -2.10pp for evidence training
```

This is not explained by "evidence itself is useless", because the no-evidence trained model gets a clean retrieval lift:

```text
M_none(retrieved) - M_none(none) = +1.90pp
```

The evidence exists, and a model not trained on it can exploit it somewhat. The model trained on evidence cannot.

Teacher mixing does not rescue the story. The teacher-mixed run has:

```text
retrieved - none     = +0.29pp
retrieved - shuffled = -0.44pp
retrieved - BM25     = -3.22pp
```

So adding teacher-score KL softens the worst sign on retrieved-vs-none but leaves the shuffled reversal and dumb-baseline failure intact.

The audited work-loop rerun is even harsher: retrieved is essentially tied with none, shuffled is better, wrong-topic is better, and BM25 wins. The exact numbers move, but the direction does not.

### What Survived

Evidence as an inference input survived weakly, because `M_none(retrieved)` shows retrieved text can help a byte-native scorer by about 2pp in this setup.

The broad idea "small model plus external evidence" is not killed.

### What Died

The claim that this evidence-conditioned training procedure changes judgment geometry died. It trained the model into worse evidence sensitivity than the closed-book control.

The correct label for v0 is not:

```text
weak positive prototype
```

It is:

```text
counterproductive evidence-training prototype
```

### Attack On The Next Defense

The next defense will say the architecture was too weak. That is plausible, but it moves the burden. A v1 cannot merely be larger. It must explain why the same training signal would stop being counterproductive.

## Iteration 58: Architecture Excuse Attack - Scaling The Classifier Could Scale The Mistake

### Current Strongest Position

After Iteration 57, the defense becomes:

```text
Evidence training failed because the architecture was not actually evidence-native.
Use separate context/evidence/candidate encoders, cross-attention, and joint
candidate comparison.
```

This diagnosis is strong. The v0 architecture serializes:

```text
context + evidence + candidate -> scalar
```

for each candidate independently. It then applies listwise CE across scalar scores. The model has no explicit evidence factor, no shared candidate set representation, no span attribution, no counterfactual evidence objective, no support relation, and no mechanism that forces the retrieved passage to update the candidate margin.

Last+mean pooling is especially weak. It invites the model to treat evidence as extra texture rather than as a variable that should change the answer.

### Attack

Architecture is a diagnosis, not an exoneration.

The v0 training objective did not ask:

```text
How should the margin change when evidence changes?
```

It asked:

```text
Given this long string, which candidate label is correct?
```

That objective lets the model solve the task by candidate artifacts, context priors, dataset label quirks, passage length, repeated words, or noise. Evidence can become a nuisance variable. In v0 it apparently did.

A 121M cross-attention version trained on the same weak supervision could learn the same wrong thing more efficiently. It could become a better artifact learner, not a better evidence user.

The architecture must therefore be paired with an identifying training design. At minimum, the same `(context, candidates)` item must be presented under:

- no evidence;
- retrieved supporting evidence;
- shuffled evidence;
- wrong-topic evidence;
- counterfactual evidence supporting a wrong candidate;
- true gold evidence sufficient to choose the correct candidate.

The loss must directly supervise margin deltas:

```text
margin(correct, best_wrong | supporting evidence)
  > margin(correct, best_wrong | no evidence)
  > margin(correct, best_wrong | shuffled evidence)

margin(counterfactual_supported_wrong, correct | counterfactual evidence)
  moves in the predicted direction
```

Without these paired constraints, "evidence-native" remains an input formatting choice, not a geometry claim.

### What Survived

A real v1 architecture would need factorization:

- encode context separately;
- encode evidence separately;
- encode candidates separately;
- let evidence attend to candidate spans and candidate spans attend to evidence;
- score candidates jointly under the same evidence;
- expose evidence-candidate support states for probing.

### What Died

The idea that v1 is mainly "same idea, bigger model" died.

The next model must not be a scaled pooled classifier. If it is, the v0 failure is already diagnostic enough to reject it before training.

### Attack On The Next Defense

The next defense will say better architecture plus better evidence is enough. But the evidence itself was not just weak. The corpus and "gold" condition were not clean enough to make a positive result meaningful.

## Iteration 59: Evidence-Quality Attack - A Better Reader Cannot Learn From A Bad Library

### Current Strongest Position

After Iteration 58, the defense becomes:

```text
v1 should use real evidence: clean corpus, better retrieval, and true gold
rationales. The v0 evidence was too weak.
```

This is correct. BM25 over 2200 documents, including benchmark training examples, is not a serious knowledge system.

### Attack

The evidence problem is more severe than "BM25 is weak."

The v0 corpus includes training examples as evidence documents:

```text
context + all choices
```

for HellaSwag and PIQA train, plus a small diverse shard. This is circular. It is useful as a contaminated diagnostic, but it cannot support a positive moonshot claim.

The "gold" condition is also not gold. It is label-conditioned BM25:

```text
query = context + correct choice
```

That is an oracle retrieval query, not a decisive supporting passage. It can retrieve answer-like lexical overlap while still giving the judge no usable reason.

The leakage audit is nonzero:

```text
exact context hits: 13 / 2048
exact choice hits: 2
long 12-gram hits: 1
```

These leaks do not explain a negative result. They do, however, prove the current corpus would be disqualifying if the result had been positive.

Most importantly: if BM25 evidence is too weak for the learned judge, why does BM25 overlap beat the judge?

```text
BM25 overlap ranker = 41.41%
M_evidence retrieved = 38.18%
```

The same evidence has more usable signal for a word-overlap rule than for the trained model. That is a reader/training failure, not just a retriever failure.

### What Survived

A true gold-evidence ceiling remains the cleanest diagnostic:

```text
true_gold high, retrieved low -> retrieval/corpus bottleneck
true_gold low                 -> judge/training bottleneck
```

### What Died

No v1 should be trained before proving an evidence-quality ceiling. The project should not spend 121M-scale GPU time to discover that the library is bad.

### Required v1 Evidence Gate

Before a v1 judge is considered a moonshot experiment, run an oracle evidence preflight:

| Gate | Required |
|---|---:|
| Strong teacher or competent judge with true gold evidence | >= 35% HellaSwag or >= +10pp over no evidence |
| True gold vs shuffled/wrong-topic | >= +8pp on HellaSwag |
| External corpus after exact/ngram/MinHash/SimHash dedupe | documented manifest hashes |
| Contaminated train-as-corpus condition | reported only as upper-bound diagnostic |
| Label-conditioned BM25 "gold" | banned from being called gold |

If true gold evidence cannot move a strong reference judge, evidence-native has no training substrate.

### Attack On The Next Defense

The next defense will say 10M and 1024 examples are too small for any of this. That is true, but using scale as the next excuse weakens the core thesis.

## Iteration 60: Scale Attack - "10M Is Too Small" Is True But Not Enough

### Current Strongest Position

After Iteration 59, the defense becomes:

```text
v0 was a tiny overfit scout. A 121M model, more data, and real gold evidence could
learn the evidence-conditioned geometry.
```

The scale objection is legitimate. A 5.54M-trainable model trained for six epochs on 1024 examples is not a serious representation learner. The evidence-trained run reaches about 90% train accuracy. The no-evidence control reaches about 99%. This is memorization territory.

### Attack

The problem is that scale is now being used to rescue a negative sign.

If v0 had shown:

```text
retrieved - none = +1pp
retrieved - shuffled = +1pp
retrieved below BM25 but directionally positive
```

then scaling would be justified. But v0 shows:

```text
retrieved - none = -0.34pp
retrieved - shuffled = -0.54pp
retrieved - BM25 = -3.22pp
M_evidence(retrieved) - M_none(retrieved) = -2.10pp
```

Scaling a negative mechanism is not a moonshot plan. It is a bet that the sign will flip for reasons not yet demonstrated.

This also pressures the slogan:

```text
Intelligence = Geometry, not Scale.
```

If the next answer is "make it 121M, use much more data, and add a much better retriever", then the result may be useful engineering, but it is not yet evidence for cheap transferable geometry.

Scale can be used only after a smaller preflight has the right sign.

### What Survived

The v0 result is not fatal to every larger architecture. It is fatal to spending large-model effort without a signed preflight.

### What Died

The project cannot defend evidence-native by saying:

```text
The tiny run was too small to matter.
```

It mattered enough to detect the wrong sign, and the wrong sign is the result that must be explained.

### Required v1 Scale Gate

Before a 121M v1 is treated as mainline, run a cheaper 20M-50M preflight with the new evidence-factorized objective:

| Gate | Required |
|---|---:|
| M_evidence(retrieved) - M_none(retrieved) | >= +2pp aggregate |
| M_evidence(none) - M_none(none) | >= +1pp aggregate |
| M_evidence(retrieved) - M_evidence(shuffled) | >= +2pp aggregate |
| True gold - none | >= +8pp HellaSwag |
| Best same-retriever dumb baseline beaten | yes, on HellaSwag and aggregate |
| Three seeds | all same sign |

If this preflight fails, a 121M run is not a design gate. It is an expensive rerun of an unearned hope.

### Attack On The Next Defense

The next defense will say evidence-native remains the only novel moonshot, while chain-init is merely engineering. But the loop cannot ignore that chain-init is the only positive empirical signal.

## Iteration 61: Chain-Init Attack - The Only Positive Signal Should Own The Mainline

### Current Strongest Position

After Iteration 60, the defense becomes:

```text
Evidence-native still deserves one serious v1 because it is the real moonshot.
Chain-init is just CBD-with-bytes, useful as a baseline but not the main idea.
```

This was the supervisor position before v0: evidence-native as moonshot, chain-init as baseline/fallback.

### Attack

That split no longer matches the evidence.

The empirical ledger now says:

| Direction | Empirical sign |
|---|---|
| Brainseed learned scorers | negative, lose to codec-only |
| Byte-marginal KD / E1 / Option C | BPB improves, task judgment flat |
| Hidden alignment / operational geometry variants | toy signals fail to transfer or collapse under controls |
| Evidence-native v0 | wrong sign, worse than closed-book control at evidence use |
| Chain-init probe | weak but positive, copied Qwen layers beat random by about 1.7 nats/token |

The chain-init signal is not benchmark-capable yet. But it is in the right direction, and it attacks the actual bottleneck identified repeatedly in `DEEP_RETHINK.md`: small byte-native models do not discover useful semantic coordinates from scratch under cheap training.

Evidence-native asks a tiny/randomly initialized judge to learn evidence use from scarce supervised examples. Chain-init starts from inherited coordinates that already encode a world-model geometry, then asks the byte interface to adapt.

That is closer to the surviving interpretation of "Intelligence = Geometry":

```text
The valuable object is the coordinate system of useful distinctions.
Do not relearn it from scratch if it can be inherited, byteified, compressed,
and tested under coordinate-disruption controls.
```

The fact that this resembles CBD is not a reason to avoid it. The moonshot is not "be maximally weird." The moonshot is "produce a stop-scrolling result that changes assumptions about small byte-native models."

If inherited-coordinate Sutra beats same-size from-scratch Sutra, random-layer controls, layer-scrambled controls, and token-model baselines under matched active compute, that is not trivial engineering. That is a falsifiable geometry claim.

### What Survived

Evidence-native survives as a possible runtime/readout layer after the model has real semantic coordinates. It does not survive as the default mainline.

### What Died

The supervisor split from check-in #4 should be revised:

```text
old: evidence-native mainline, chain-init fallback
new: coordinate inheritance mainline, evidence-native v1 kill-gated support branch
```

### Attack On The Next Defense

The next defense will say chain-init proves scale or copying, not geometry. That is a serious attack. The geometry thesis must become falsifiable rather than a post-hoc slogan.

## Iteration 62: Geometry Falsifiability Attack - Stop Letting The Slogan Absorb Failures

### Current Strongest Position

After Iteration 61, the defense becomes:

```text
Inherited-coordinate chain-init is the best remaining route to prove Intelligence
= Geometry.
```

### Steelman Against It

A fresh adversary will say:

```text
That is not geometry. That is copying a bigger model.
```

They will be partly right. If the plan is merely "wrap Qwen in bytes and keep most of Qwen", then the project has become tokenizer conversion plus compression. That may be useful, but it does not prove the strong Sutra thesis.

### Attack And Reconstruction

The only way to make chain-init a geometry proof is to test the coordinate system itself.

The new falsifiable claim should be:

```text
Reasoning capability is concentrated in transferable coordinate geometry.
If that geometry is preserved, a small byte-native runtime learns and judges far
more efficiently than a same-size random model. If the geometry is disrupted, the
advantage collapses.
```

This is testable.

### Coordinate-Inheritance Geometry Program

Build a byteified inherited-coordinate model and compare against destructive coordinate controls.

Core variants:

| Variant | Purpose |
|---|---|
| Inherited layers + trained byte adapters | main coordinate-preserving model |
| Same architecture, random layers | random baseline |
| Inherited layers, shuffled layer order | tests depth/coordinate organization |
| Inherited layers, random orthogonal residual rotations without matching LM head | tests gauge disruption |
| Inherited layers, frozen bad codec | tests byte adapter quality |
| Token model teacher direct | upper bound |
| Wide7 from scratch | byte-native from-scratch baseline |
| Evidence-native v1 head on frozen inherited model | downstream evidence-use probe |

Required first gates:

| Gate | Required |
|---|---:|
| Token/patch NLL vs random at same inputs | >= 2.0 nats improvement |
| HellaSwag vs Wide7 from scratch after same byte adaptation budget | >= +5pp |
| PIQA/ARC aggregate vs Wide7 | >= +3pp |
| Coordinate-disrupted controls lose most of the gain | yes |
| Compression/pruning to <= 121M or <= 121M active params keeps at least half the lift | yes |
| Byte adapter robustness on typo/OCR/Unicode perturbations | beats tokenized baseline or clearly narrows gap |

If the inherited model wins but coordinate-disrupted controls also win, the gain is not geometry. It is some artifact of parameter count, training data, or scoring.

If the inherited model does not beat random and Wide7 after a fair adaptation budget, the geometry-transfer thesis is false.

### What This Does To The Original Thesis

The strong from-scratch version is now very weak:

```text
121M byte-native model discovers benchmark-grade reasoning geometry cheaply from
raw training or tiny evidence supervision.
```

The surviving version is narrower but still meaningful:

```text
Large-scale training discovers useful reasoning coordinates; Sutra's job is to
inherit, byteify, compress, and expose those coordinates with far cheaper active
compute and better updateability.
```

This is less romantic, but more testable.

### What Survived

The geometry thesis survives only if "geometry" means a transferable structure of distinctions that can be preserved, disrupted, compressed, and measured.

### What Died

The unfalsifiable slogan died:

```text
Every failure just means we did not find the right geometry yet.
```

Batch 9 must set death conditions. If inherited-coordinate controls fail, if evidence-internalization fails under true gold evidence, and if from-scratch byte-native baselines remain near chance, the project must stop claiming that geometry has been found.

### Attack On The Next Defense

The final defense is emotional inertia: after 63 question-loop iterations, maybe one more pivot will find the missing piece. A fresh reviewer will not grant that.

## Iteration 63: Fresh Reviewer Attack - After 63 Iterations The Burden Is No Longer Patience

### Current Strongest Position

After Iteration 62, the position is:

```text
Evidence-native should be demoted; inherited-coordinate chain-init should become
the mainline geometry proof; evidence-native v1 may run only as a hard-gated
support branch.
```

### Fresh Reviewer View

A fresh adversarial reviewer reading the whole arc sees:

- The Vision demands paradigm shift or failure.
- Byte-level KD improved BPB but not task judgment.
- Brainseed extraction failed every learned scorer control.
- Operational geometry produced toy signals that did not transfer cleanly.
- Real S0 energy probing did not reveal hidden HellaSwag knowledge.
- Width improved byte modeling and speed but not reasoning benchmarks.
- Evidence-native v0 trained on evidence and became worse at using evidence.
- Chain-init is weak, ugly, and incomplete, but it is the only positive sign.

Honest one-line summary:

```text
Sutra has repeatedly shown that small byte models can learn surface form cheaply,
but not benchmark-grade judgment from scratch; the only live moonshot is to
transfer an existing reasoning coordinate system into a cheap byte-native runtime.
```

### Diagnostic Or Fatal?

v0 is diagnostic of several concrete failures:

- independent candidate scoring cannot prove evidence binding;
- mean pooling is too weak for support attribution;
- the corpus is circular and not claim-safe;
- label-conditioned BM25 is fake gold;
- 1024 examples encourage artifact memorization;
- the frozen codec supplies byte/token addressability, not evidence judgment.

But v0 is fatal to the current evidence-native mainline because the sign is wrong. The training signal did not merely underperform. It made the model a worse evidence user than the no-evidence-trained control.

So the correct verdict is:

```text
v0 failure: diagnostic of mechanisms, fatal to v0.
evidence-native as application/runtime idea: alive.
evidence-native as moonshot mainline: demote now.
```

### What Exactly Went Wrong?

All major axes contributed:

| Axis | Failure |
|---|---|
| Architecture | independent `(context, evidence, candidate)` scoring, last+mean pooling, no joint candidate comparison, no evidence-span support state |
| Training | tiny 1024-record supervised set, high train accuracy, no paired margin-delta objective, no counterfactual evidence supervision |
| Evidence | circular train-as-corpus, BM25 lexical noise, fake gold, nonzero leakage, no true decisive passage ceiling |
| Codec | frozen byte/token-identity infrastructure, not trained for evidence support or candidate judgment |
| Scale | too small to adjudicate final architecture, but the wrong sign blocks scale-up as a mainline bet |
| Evaluation | good first controls, but missing paired stats, learned <=20M baselines, true gold, geometry probes, and open-book token judge |

The root failure is not any single line of code. The root failure is non-identifying supervision: the model was never forced to learn the operation "evidence changes the candidate margin for this reason."

### v1 Design Gate

Evidence-native v1 is allowed only as a kill-gated support branch, not as the project mainline. It must clear all of these before being promoted again:

| Category | Gate |
|---|---|
| True gold ceiling | `true_gold >= no_evidence + 10pp` on HellaSwag or `true_gold >= 35%` HellaSwag |
| Training effect | `M_evidence(retrieved) >= M_none(retrieved) + 3pp` aggregate |
| Internalization | `M_evidence(none) >= M_none(none) + 2pp` aggregate |
| Evidence use | `M_evidence(retrieved) >= M_evidence(none) + 3pp` aggregate |
| Negative controls | shuffled/wrong-topic gains <= 25% of retrieved gain |
| Counterfactual control | wrong-support evidence moves margin in predicted wrong direction |
| Baselines | beat BM25, kNN, TF-IDF/logistic, MLP features, and <=20M same-input cross-encoder |
| Corpus | external-only manifest with exact/ngram/MinHash/SimHash dedupe |
| Seeds/stats | at least 3 seeds, paired bootstrap/McNemar-style tests |
| Geometry probes | support-relation probe, evidence masking, paraphrase invariance, margin-update analysis |
| Transfer | immediate win on at least two task families, then three before any public claim |

If v1 fails the internalization gate, evidence-native should be permanently classified as a RAG/reranking application layer, not the moonshot.

### Final Mainline Decision

After 63 total Q-loop iterations:

```text
KILL_EVIDENCE_NATIVE_V0
DEMOTE_EVIDENCE_NATIVE_FROM_MOONSHOT_MAINLINE
PROMOTE_COORDINATE_INHERITANCE_CHAIN_INIT_AS_MAINLINE
ALLOW_EVIDENCE_NATIVE_V1_ONLY_AS_KILL_GATED_SUPPORTING_BRANCH
```

The single strongest remaining direction for proving `Intelligence = Geometry` is:

```text
Coordinate-Inheritance Sutra:
byteify an inherited pretrained coordinate system, preserve it through byte
adapters, compress/prune it toward 121M or 121M active params, and prove the gain
collapses under coordinate-disruption controls.
```

This direction has one weak positive signal already: copied Qwen layers are more compatible with codec inputs than random layers by about 1.7 nats/token. It also has a clean falsification framework. It can fail. That is a feature.

Evidence-native can return later as a judgment/readout layer on top of a model that already has usable semantic coordinates. But evidence-native should no longer be trusted to create those coordinates from scratch.

## Synthesis: After 63 Total Question Iterations

Evidence-native v0 is dead.

Evidence-native as the main moonshot direction should be demoted. It earned one serious redesign only as a constrained support branch, and only because the broad idea of external evidence remains important. It did not earn continued mainline status.

The project should stop asking:

```text
Can a tiny randomly initialized byte judge learn to use retrieved evidence from
weak supervised examples?
```

The answer so far is no, and worse than no: evidence-conditioned training made the model less evidence-sensitive than the closed-book control.

The project should now ask:

```text
Can Sutra inherit, preserve, compress, and expose a reasoning coordinate geometry
that a small byte-native model cannot afford to discover from scratch?
```

That is the only remaining path that is simultaneously empirical, falsifiable, aligned with the repeated failure history, and still capable of a home-run result.

### Final Gossip-Magazine Headline

The evidence-native judge read the book and got worse. The inheritance path only twitched, but at least it twitched in the right direction.
