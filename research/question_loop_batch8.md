# QUESTION LOOP - Batch 8: Post-Prototype Attack

Date: 2026-07-07

Grounding: I read the requested context in order:

1. `research/VISION.md`
2. `research/DEEP_RETHINK.md`
3. `research/question_loop_batch7.md`
4. `research/dual_loop_supervisor_checkin_4.md`
5. `research/work_loop_batch6.md` check: file does not exist in this checkout
6. `C:/sutra_fast/evidence_native/metrics.json`
7. `code/evidence_native_sutra.py`

Additional relevant artifact found during inspection:

- `C:/sutra_fast/evidence_native/closed_book_control/metrics.json`

Binding state entering Batch 8:

- The Work Loop markdown report is absent, but raw metrics exist.
- The prototype is not the promised 121M evidence-native judge. It is a fast falsification model: 10.07M total params, 5.54M trainable params, frozen 4.26M codec encoder, 2-layer 256-dim reasoner.
- Training data: 512 HellaSwag train + 512 PIQA train.
- Eval data: 1024 HellaSwag validation + 1024 PIQA validation.
- Corpus: 2200 docs from `benchmark_train:hellaswag:train`, `benchmark_train:piqa:train`, and `diverse_shard_public_text`.
- Teacher score cache attached to 0/1024 training records. This run is not teacher-margin training.

The result is not subtle:

| Model | Eval evidence | Overall | HellaSwag | PIQA |
|---|---:|---:|---:|---:|
| train retrieved | retrieved | 38.28% | 26.07% | 50.49% |
| train retrieved | none | 38.48% | 25.39% | 51.56% |
| train retrieved | shuffled | 38.72% | 25.10% | 52.34% |
| train retrieved | wrong topic | 37.65% | 24.02% | 51.27% |
| train retrieved | gold | 37.79% | 25.20% | 50.39% |
| train none | retrieved | 40.28% | 27.44% | 53.12% |
| train none | none | 38.38% | 26.95% | 49.80% |
| train none | shuffled | 38.18% | 26.07% | 50.29% |
| train none | wrong topic | 38.67% | 26.27% | 51.07% |
| train none | gold | 38.57% | 26.46% | 50.68% |

Dumb baselines on the same eval set:

| Baseline | Overall | HellaSwag | PIQA |
|---|---:|---:|---:|
| BM25 evidence overlap ranker | 41.41% | 26.86% | 55.96% |
| Shortest candidate | 39.89% | 24.80% | 54.98% |
| Nearest-neighbor train label | 39.21% | 28.71% | 49.71% |
| Unigram frequency | 38.87% | 25.98% | 51.76% |
| Majority label | 36.57% | 23.54% | 49.61% |

The raw gates all failed:

- retrieved minus no evidence: -0.20pp for the evidence-trained model;
- retrieved minus best dumb baseline: -3.12pp;
- retrieved minus shuffled: -0.44pp;
- shuffled evidence was better than retrieved evidence overall;
- gold evidence was worse than retrieved evidence overall;
- the no-evidence-trained control with retrieved evidence beat the evidence-trained model with retrieved evidence by +2.00pp.

This kills the first prototype. It does not yet kill the steelman, because the steelman is not "append evidence at inference." The steelman is:

```text
Evidence-conditioned training changes the internal judgment geometry.
```

The prototype did not show that. Worse: its control run points the other way.

## Batch 7 Prediction Scorecard

| B7 prediction | Score | Why |
|---|---|---|
| Retrieved evidence will lift HellaSwag above closed-book. | Mostly refuted | Evidence-trained HellaSwag retrieved is 26.07% vs its no-evidence 25.39%, a tiny +0.68pp, but it is below the no-evidence-trained control with retrieved evidence at 27.44% and below nearest-neighbor train-label at 28.71%. |
| Lift largest on HellaSwag, weaker on PIQA/ARC/WinoGrande. | Refuted by available tasks | Evidence-trained PIQA gets worse with retrieved evidence: 50.49% vs 51.56% no evidence. The train-none control's retrieval lift is larger on PIQA than HellaSwag. ARC/WinoGrande were not tested. |
| Dumb baselines will explain more lift than expected. | Confirmed harder than predicted | BM25 evidence overlap beats the evidence-trained judge by +3.12pp overall. Nearest-neighbor train label beats it on HellaSwag. |
| Shuffled controls will not reduce all the way to no-evidence baseline. | Confirmed catastrophically | For the evidence-trained model, shuffled is better than retrieved overall: 38.72% vs 38.28%. |
| Small learned classifiers will be competitive. | Unclear but likely | No <=20M learned baseline was run, but the 5.54M-trainable judge failed to beat dumb baselines and overfit training accuracy. The warning remains live. |
| Qwen3-0.6B-with-evidence will beat first Sutra judge by a wide margin. | Not tested | The run did not include fair open-book token model baselines. |
| First prototype will not prove byte-native advantage. | Confirmed | Retrieval is BM25 over normalized text. No byte-native retriever or tokenizer-hostile robustness test was run. |

## Iteration 50: Result Attack - The Prototype Failed Before Philosophy Started

### Current Strongest Position

The fair defense is:

```text
This was only a small fast falsification pass. It should not be treated as the
121M evidence-native moonshot.
```

That defense is correct as far as it goes. The prototype has only 5.54M trainable parameters, trains for six epochs over 1024 examples, and freezes the codec. It cannot adjudicate the final architecture.

### Steelman

The run still tested a useful first question:

```text
Can a frozen codec plus small reasoner learn any evidence-conditioned candidate
judgment signal above no-evidence and dumb baselines?
```

If this had produced even a modest clean signal, it would justify scaling the judge toward 121M. A cheap positive slope would have been meaningful because it would show that the codec/evidence serialization contains usable decision information.

### Attack

It did not produce a signal.

The evidence-trained model gets:

```text
retrieved: 38.28%
none:      38.48%
shuffled: 38.72%
gold:      37.79%
```

So retrieved evidence is worse than no evidence and worse than shuffled evidence. Gold evidence is not an upper bound. It is also worse than retrieved.

The worst result for the steelman is the no-evidence-trained control:

```text
train_none + retrieved = 40.28%
train_retrieved + retrieved = 38.28%
```

If evidence-conditioned training were changing the model into a better evidence user, the trained-with-evidence model should not lose by 2pp to a model trained without evidence when both are handed retrieved evidence at test time.

The run also fails the "retrieval is not the hero" gate:

```text
BM25 overlap ranker = 41.41%
evidence-trained judge = 38.28%
```

The dumb ranker wins by 3.12pp overall and by 5.47pp on PIQA.

### What Survived

Survived:

- The full evidence-native steelman is not killed by this specific model size.
- The code now provides a useful harness for retrieved/none/shuffled/wrong/gold comparisons.
- The result is a clean negative against the first cheap judge.

### What Died

Dead:

- This exact v0 architecture as evidence-native proof.
- Any claim that retrieved evidence helped the evidence-trained model.
- Any claim that evidence-conditioned training was beneficial in this run.
- Any headline based on the 38.28% number.

### Narrative Attack

1. "That's obvious" dismissal: The small classifier could not use the retrieved snippets.
2. "That's trivial" dismissal: The retrieval baseline did better without neural judgment.
3. What would beat the attack: Evidence-trained retrieved must beat no-evidence-trained retrieved, no-evidence evaluation, shuffled controls, and dumb same-retriever baselines.

### Gossip-Magazine Headline

The judge opened the book, forgot how to read, and lost to a word-count rule.

## Iteration 51: Sample-Size Attack - The Failure Is Real For Moonshot Effects, Not For Tiny Effects

### Current Strongest Position

After Iteration 50, the defense becomes:

```text
The result is negative, but 1024 train examples per mixed task and 1024 eval
examples per task are too small to conclude much.
```

### Steelman

There is a real sample-size objection.

The train set is tiny. A 5.54M-trainable model reaches 90.23% train accuracy in the evidence-trained run and 98.73% train accuracy in the no-evidence control. That is overfit territory, not representation-learning territory.

The evaluation set is also not enough for fine-grained effects. Around 25-50% accuracy with n=1024 per task gives a rough two-condition 95% difference band of about +/-4pp per task if treated conservatively. Overall n=2048 gives a rough band near +/-3pp. Without paired McNemar counts, the exact uncertainty is unknown.

So the run cannot kill a 1-2pp subtle geometry effect.

### Attack

But the Vision does not need a 1-2pp subtle effect. It needs a stop-scrolling effect.

Batch 7 set a +5pp evidence gate, a +3pp dumb-baseline gate, and a collapse under shuffled/wrong-topic evidence. This sample is adequate to reject those large first-pass gates directionally because the signs are wrong:

```text
retrieved - none      = -0.20pp
retrieved - shuffled  = -0.44pp
retrieved - best dumb = -3.12pp
```

Even if the true effect were hidden by noise, it is not large enough in this run to matter. A moonshot cannot retreat to "maybe there is a tiny positive effect underpowered by 2048 eval examples" after the first prototype loses to BM25 overlap.

The sample-size objection saves only the broad research direction, not the run.

### What Survived

Survived:

- Do not kill evidence-native solely from one seed, 1024 training records, and a 10M parameter judge.
- Future work needs at least three seeds and paired significance tests.
- The current eval is enough for gating large effects, not for measuring small representation deltas.

### What Died

Dead:

- "Maybe the +5pp lift is just hidden by noise."
- Any public claim from this result.
- Any next run that reports only one seed and no paired prediction overlap.

### Narrative Attack

1. "That's obvious" dismissal: A tiny overfit prototype is noisy.
2. "That's trivial" dismissal: Noise cannot explain the retrieved condition losing to shuffled and BM25.
3. What would beat the attack: Three seeds, paired bootstrap/McNemar tests, and confidence intervals whose lower bounds clear the survival gates.

### Gossip-Magazine Headline

The confidence interval was wide, but the corpse was facing the wrong way.

## Iteration 52: BM25 Attack - A Weak Retriever Is Not An Excuse When BM25 Wins

### Current Strongest Position

After Iteration 51, the defense becomes:

```text
The prototype may have failed because BM25 is a weak retriever. Evidence-native
cannot learn judgment from bad evidence.
```

### Steelman

This is plausible. BM25 over a tiny 2200-document corpus is not a serious evidence system. It retrieves lexical neighbors, not necessarily decisive commonsense facts. If evidence is irrelevant or misleading, the judge should not be expected to improve.

Better retrieval could produce a different result:

- denser semantic recall;
- more relevant factual or script evidence;
- lower noise in top-k passages;
- stronger training signal for evidence-conditioned judgment.

Failure with weak evidence does not prove evidence-native is wrong.

### Attack

But weak BM25 did not merely fail to help the neural judge. Weak BM25 beat it.

The BM25 evidence overlap ranker gets:

```text
overall:   41.41%
HellaSwag: 26.86%
PIQA:      55.96%
```

The evidence-trained judge gets:

```text
overall:   38.28%
HellaSwag: 26.07%
PIQA:      50.49%
```

So the retriever was strong enough for a dumb overlap rule to exploit, but the learned judge did not exploit it. That is not just a retriever problem. It is a judge/evidence-use problem.

The "gold" condition also fails to rescue the judge. In the code, gold is not true human gold evidence; it is BM25 retrieval using:

```text
context + correct choice
```

as the query. That is label-conditioned oracle retrieval, not deployed retrieval. Even this does not improve the evidence-trained model:

```text
gold:      37.79%
retrieved: 38.28%
none:      38.48%
```

If a better retriever fixes everything, then the intelligence may live in the retriever. The next run must not just swap in better retrieval. It must measure retriever contribution separately.

### What Survived

Survived:

- BM25 is too weak to be the final evidence source.
- Better retrieval is allowed as a next experimental branch.
- Retrieved-evidence quality must be bucketed and measured.

### What Died

Dead:

- Using BM25 weakness to excuse losing to BM25 overlap.
- Calling label-conditioned BM25 "gold evidence."
- Scaling the judge before proving it can use evidence that a dumb overlap rule already finds useful.

### Narrative Attack

1. "That's obvious" dismissal: Better search might help.
2. "That's trivial" dismissal: In this run, worse search plus a dumb rule already helped more than the model.
3. What would beat the attack: A true gold-evidence ceiling and a retrieved condition where the judge beats the same evidence under overlap, kNN, and small learned baselines.

### Gossip-Magazine Headline

The retriever was supposedly too weak until the dumb ranker used it better.

## Iteration 53: Corpus-Contamination Attack - Train-As-Corpus Makes The Library Circular

### Current Strongest Position

After Iteration 52, the defense becomes:

```text
The BM25 baseline winning is a corpus/retrieval artifact. The next prototype can
use a cleaner and stronger corpus.
```

### Steelman

Yes. The current corpus was designed for a fast pass, not a reviewer-proof claim. It includes benchmark training examples and a small diverse shard. A better corpus could be:

- source-manifested;
- benchmark-deduped;
- larger and more semantically relevant;
- split by task provenance;
- cleanly separated from eval records.

The current failure may reflect bad evidence inventory, not bad evidence-native learning.

### Attack

The current corpus is not merely weak. It is circular.

The implementation constructs corpus docs from training examples:

```text
context + all choices
```

for HellaSwag and PIQA train, then retrieves against that corpus. This means the system is trained and evaluated in a world where benchmark-shaped contexts and choices are themselves the evidence store.

The leakage audit is already nonzero:

```text
exact context hits: 13 / 2048 eval examples
exact choice hits: 2
long 12-gram hits: 1
```

Those numbers are not huge, and they do not explain the failed result. But they show that the current corpus cannot support a clean positive claim if a later run improves.

The nearest-neighbor train-label baseline is also instructive:

```text
nearest-neighbor HellaSwag: 28.71%
evidence-trained HellaSwag: 26.07%
```

On HellaSwag, train-neighbor label transfer beats the evidence-native judge. That is exactly the "benchmark-neighbor engine" failure mode Batch 7 warned about.

### What Survived

Survived:

- Train-as-corpus is acceptable as a contaminated upper-bound diagnostic.
- It is useful for finding whether benchmark-neighbor evidence has any signal.
- It is not acceptable as moonshot evidence.

### What Died

Dead:

- Any reviewer-facing claim from a corpus that includes benchmark training contexts plus choices as evidence.
- Any eval that does not separate "train-as-corpus", "train-context-only", "choices removed", and "external public evidence only."
- Any leakage audit limited to exact strings and one 12-gram pass.

### Narrative Attack

1. "That's obvious" dismissal: It retrieved benchmark neighbors.
2. "That's trivial" dismissal: It put the benchmark training set in the library, choices included.
3. What would beat the attack: Gains persist when the corpus excludes benchmark train choices, excludes near duplicates, and uses frozen external evidence with manifest hashes.

### Gossip-Magazine Headline

The open-book exam studied from last year's multiple-choice sheets and still missed the point.

## Iteration 54: Architecture Attack - The Model Was Not Built To Prove Internalized Evidence Geometry

### Current Strongest Position

After Iteration 53, the defense becomes:

```text
Clean up the corpus, improve retrieval, and the direction gets a fair test.
```

### Steelman

That is necessary. It is not sufficient.

Evidence-native is not just a data pipeline. It needs an architecture and loss that make the model learn how evidence changes judgment.

The current code is deliberately simple:

```text
context + separator + evidence + separator + candidate -> scalar score
```

Each candidate is serialized separately. The model sees all candidates only through grouped cross-entropy after independent scoring. It uses a frozen codec, a projection, a 2-layer reasoner, and last+mean pooling.

This is a reasonable fast classifier harness.

### Attack

It is not a strong test of internalized evidence geometry.

The architecture has no explicit mechanism for:

- comparing candidates against the same evidence jointly;
- identifying which evidence span supports which candidate;
- penalizing evidence-insensitive predictions;
- learning from counterfactual evidence that should flip the answer;
- preserving an evidence-conditioned representation when evidence is removed;
- separating context priors from evidence updates.

The strongest negative result is the control:

```text
train_none + retrieved > train_retrieved + retrieved
```

This attacks the steelman directly. Evidence-conditioned training did not make the model a better evidence user. It may have made it worse, or just added noisy irrelevant text during training.

If the next version simply scales this same independent-candidate, pooled classifier to 121M, it may learn stronger artifacts. That would make the result more dangerous, not more scientific.

### What Survived

Survived:

- The current code is a useful negative-control harness.
- The frozen codec may still be useful as byte/token infrastructure.
- A larger reasoner could be tested, but only with internalization controls.

### What Died

Dead:

- Treating this 10M model as a miniature proof of the 121M architecture.
- Scaling without architectural changes that force evidence sensitivity.
- Any run that lacks the train-with-evidence vs train-without-evidence comparison.

### Narrative Attack

1. "That's obvious" dismissal: The classifier pooled text and guessed.
2. "That's trivial" dismissal: Training with evidence did not improve evidence use compared to training without evidence.
3. What would beat the attack: The evidence-trained model must beat an identical no-evidence-trained model both with retrieved evidence and after evidence is removed.

### Gossip-Magazine Headline

The geometry lesson became a long input string and a mean pool.

## Iteration 55: Gold-Evidence And Geometry Attack - If Gold Does Not Move It, The Judge Cannot Read

### Current Strongest Position

After Iteration 54, the defense becomes:

```text
The current gold evidence is not true gold. Build better supervision and probe
the representations before judging the geometry claim.
```

### Steelman

Correct. The current "gold" condition is not a true upper bound. It is retrieved evidence from a label-conditioned query. It may return:

- a near duplicate;
- irrelevant lexical overlap;
- a passage that includes answer words but not the reason;
- nothing decisive.

The next run needs real gold evidence:

```text
Given this context and these candidates, this passage/sentence is sufficient
to prefer the correct answer.
```

If true gold evidence produces a large jump, the judge can read but retrieval is the bottleneck. If true gold evidence fails, the architecture or training loss is the bottleneck.

### Attack

This is exactly why the next run must not be allowed to hide behind "retrieval is hard."

The gold-evidence ceiling is the fastest way to disambiguate:

```text
true_gold - none large  -> retrieval/corpus bottleneck
true_gold - none small  -> judge/training bottleneck
```

The current run has:

```text
label-query gold - none = -0.69pp overall
```

That does not prove true gold will fail, but it means no evidence sensitivity has been observed yet.

Now attack the bigger claim:

```text
Intelligence = Geometry, not Scale.
```

A 4-way MCQ scalar scorer does not prove geometry. A dumb overlap rule also defines a geometry. A logistic regression over lexical features defines a geometry. A nearest-neighbor label transfer defines a geometry.

The project needs measurable content:

- Does evidence training create hidden states that cluster by support relation, not by answer length or lexical overlap?
- Does counterfactual evidence rotate the correct-candidate margin in the predicted direction?
- Does the evidence-trained representation retain a decision advantage when evidence is removed at test time?
- Can a linear probe extract evidence-candidate support from hidden states, and does the full model beat that probe on adversarial evidence?
- Does paraphrased evidence map to similar margin updates?
- Does masking decisive evidence spans causally erase the margin?

Without probes like these, "judgment geometry" is just a more flattering name for a classifier boundary.

### What Survived

Survived:

- The steelman still has one clean open test: true gold evidence.
- Geometry can be made concrete through representation and causal probes.
- The current prototype did not run those probes.

### What Died

Dead:

- Treating MCQ accuracy alone as proof of learned geometry.
- Treating label-conditioned BM25 as a gold-evidence ceiling.
- Treating evidence-at-input gains as evidence-conditioned training gains.

### Narrative Attack

1. "That's obvious" dismissal: Give the answer-like passage and accuracy may rise.
2. "That's trivial" dismissal: Rising with oracle evidence only proves the oracle helped unless representations and controls show the judge learned a reusable update rule.
3. What would beat the attack: True gold evidence creates a large ceiling, and internal probes show evidence-conditioned margin geometry that cheap baselines do not learn.

### Gossip-Magazine Headline

The geometry was beautiful until someone asked what coordinate changed.

## Iteration 56: Path-To-42.65 Attack - The Road From 26% To CBD Is Not This Road Yet

### Current Strongest Position

After Iteration 55, the defense becomes:

```text
The first prototype failed, but a real 121M evidence-trained judge with true
gold ceilings, clean corpus, better retrieval, and geometry probes could still
be the moonshot.
```

### Steelman

This is the surviving version.

Evidence-native should not be killed because a 10M fast pass failed. The direction still answers the failure history better than closed-book 121M training:

- do not force tiny weights to memorize all commonsense;
- use public evidence as repairable memory;
- train a compact model to judge evidence;
- evaluate updateability and robustness, not just closed-book recall.

The steelman is alive if and only if evidence-conditioned training changes the model's internal decision behavior.

### Attack

The path to 42.65% HellaSwag is not visible in the current run.

Current HellaSwag:

```text
evidence-trained retrieved: 26.07%
best dumb HellaSwag baseline: 28.71% nearest-neighbor train label
closed-book-trained retrieved: 27.44%
CBD target: 42.65%
```

The gap from the evidence-trained judge to CBD is 16.58pp. The gap from the best current HellaSwag baseline to CBD is still 13.94pp.

Even reaching 30% would not prove the Vision. It would only return the project to the low-30s zone already discussed in `DEEP_RETHINK.md`. A realistic path to 42%+ needs at least one of:

- a true evidence-use ceiling well above 35%;
- a 121M judge that beats no-evidence-trained controls by several points;
- transfer beyond HellaSwag and PIQA;
- evidence-native training that improves no-evidence internal judgment;
- chain-init or teacher-inherited coordinates combined with evidence use;
- a measured efficiency win over stronger open-book token models.

Otherwise, the direction becomes a worse reranker wrapped in a better story.

### What Survived

Survived:

- Evidence-native remains a research direction worth one more serious, adversarially designed iteration.
- The next iteration must test the steelman directly, not just retrieval input.
- Chain-init remains the fallback and fair baseline.

### What Died

Dead:

- The v0 evidence-native prototype.
- The idea that BM25+small frozen-codec judge is a promising first positive slope.
- Any plan to scale before proving true gold evidence, internalization, and baseline separation.

### Narrative Attack

1. "That's obvious" dismissal: A better evidence system might improve a multiple-choice scorer.
2. "That's trivial" dismissal: Unless evidence training changes the model, it is just RAG/reranking.
3. What would beat the attack: A fresh reviewer sees the evidence-trained model outperform all no-evidence-trained, dumb, small-learned, and open-book baselines under clean retrieval, plus probes showing reusable internal evidence geometry.

### Gossip-Magazine Headline

The moonshot found a map, but the first arrow pointed backward.

## Batch 9 Conditions For A Fresh Adversarial Reviewer

The next run must be designed around the steelman:

```text
Evidence-conditioned training changes internal judgment, not just inference
inputs.
```

Minimum Batch 9 survival bundle:

1. **Write the Work Loop report.** `research/work_loop_batch6.md` or its Batch 9 equivalent must exist and summarize commands, metrics, seeds, failures, and exact artifacts.

2. **Run at least three seeds.** Report paired prediction overlap, bootstrap CIs, and McNemar-style tests for retrieved vs none, retrieved vs shuffled, and train-evidence vs train-none.

3. **Separate training effect from inference effect.** Train two identical models:

```text
M_evidence: trained with retrieved or gold evidence
M_none:     trained with no evidence
```

Required gates:

```text
M_evidence(retrieved) >= M_none(retrieved) + 3pp aggregate
M_evidence(none)      >= M_none(none)      + 2pp aggregate
M_evidence(retrieved) >= M_evidence(none)  + 3pp aggregate
```

The second line is the internalization gate. If it fails, the model may use evidence at inference, but it has not shown evidence-conditioned learning.

4. **Run a true gold-evidence ceiling.** Gold evidence must be a decisive passage/sentence/rationale created without held-out label leakage into retrieval queries. A label-conditioned BM25 query is not enough.

Required gate:

```text
true_gold >= no_evidence + 10pp on HellaSwag
or true_gold >= 35% HellaSwag
```

If true gold fails, the judge architecture is dead. If true gold succeeds but retrieved fails, retrieval/corpus is the bottleneck.

5. **Clean the corpus into explicit conditions.** Report separately:

- contaminated train-as-corpus upper bound;
- train-context-only corpus with choices removed;
- external public evidence corpus;
- external corpus after exact, n-gram, MinHash/SimHash, and retrieval-neighbor dedupe;
- corpus manifest hashes and source categories.

No positive claim may rely on the contaminated condition.

6. **Beat same-retriever baselines.** Required gates:

```text
M_evidence(retrieved) >= best non-neural same-retriever baseline + 5pp on HellaSwag
M_evidence(retrieved) >= best <=20M same-input learned baseline + 3pp aggregate
```

At minimum, run logistic/TF-IDF, BM25 overlap, kNN label transfer, MLP feature baseline, and a <=20M cross-encoder/reranker.

7. **Negative controls must collapse.** Define:

```text
retrieved_gain = M_evidence(retrieved) - M_evidence(none)
```

Required gates:

```text
M_evidence(shuffled) - M_evidence(none) <= 25% of retrieved_gain
M_evidence(wrong_topic) - M_evidence(none) <= 25% of retrieved_gain
```

Also add counterfactual evidence that supports a wrong candidate and require the correct-candidate margin to move in the predicted direction.

8. **Probe geometry.** Report at least three:

- margin update under evidence vs no evidence;
- counterfactual evidence flip rate;
- hidden-state probe for candidate-support relation;
- paraphrased-evidence invariance;
- causal evidence-span masking;
- representation comparison between M_evidence and M_none.

The probes must compare against dumb and <=20M learned baselines.

9. **Transfer beyond the first format.** The next serious run must include at least HellaSwag + PIQA + one of ARC, WinoGrande, or generated/reranked continuations.

Required gate:

```text
Evidence-trained model beats best same-retriever baseline on at least two task
families immediately, and Batch 10 must extend this to three.
```

10. **Do not claim byte-native yet.** The current retrieval path is text-token BM25. The byte-native claim requires a separate test:

- byte-native retrieval or no external semantic retriever;
- typo/OCR/Unicode/code/mixed-format robustness;
- same-size token evidence judge comparison;
- raw-byte corpus update without tokenizer retraining.

Until then, the honest phrase is:

```text
small evidence-conditioned judge with byte I/O
```

not:

```text
byte-native intelligence
```

## Synthesis: After 56 Total Question Iterations

Evidence-Native Sutra is **on life support**.

The exact v0 prototype is dead. It failed the main gates, lost to BM25 overlap, lost to shuffled evidence, failed its label-query gold condition, and was outperformed by a no-evidence-trained control when both were evaluated with retrieved evidence.

The broader direction is not dead because the prototype did not test the strongest claim at full scale. It tested:

```text
Can a 10M frozen-codec pooled classifier use BM25 evidence after 1024 training
records?
```

It did not test:

```text
Can evidence-conditioned training produce internal judgment geometry in a 121M
model that transfers beyond the evidence seen at training time?
```

But the burden has shifted. Evidence-native no longer gets optimism by default. Batch 9 must show that evidence training changes the model, not just the input.

The single most dangerous unanswered question is:

```text
Does training with evidence make an otherwise identical model a better judge
after controlling for retrieval, corpus leakage, dumb baselines, model size,
and no-evidence-trained controls?
```

If the answer is no, kill evidence-native as the moonshot mainline and demote it to an application-layer RAG/reranking tool. The mainline should move back to chain-init, inherited coordinates, or a larger Sutra-family anchor.

If the answer is yes, the moonshot survives, because that would finally be evidence that the training signal changed the model's internal judgment geometry rather than merely giving it a searchable crutch.

### Final Gossip-Magazine Headline

The evidence-native dream survived the crash, but only because the black box was too small to be the real plane.
