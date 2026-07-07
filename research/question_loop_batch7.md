# QUESTION LOOP - Batch 7: Attack Evidence-Native Sutra

Date: 2026-07-07

Grounding: I read the requested files in order:

1. `research/VISION.md`
2. `research/DEEP_RETHINK.md`
3. `research/question_loop_batch6.md`
4. `research/dual_loop_supervisor_checkin_4.md`
5. `research/work_loop_batch5.md`
6. `research/work_loop_batch6.md` check: file does not exist as of this batch

Binding state after 42 Q-Loop iterations:

- Brainseed v0 is dead as a birth artifact.
- Codec/Brainseed remains infrastructure and diagnostic evidence, not downstream
  intelligence.
- Chain-init has a weak compatibility signal and is now the strong baseline /
  fallback, not the moonshot mainline.
- Evidence-Native Retrieval-Born Sutra is the accepted moonshot mainline, but
  no positive evidence-native result exists yet.
- The Work Loop is expected to build the first prototype around retrieval,
  evidence-conditioned training, teacher-generated curriculum, and controls.

This batch assumes the prototype will produce at least one attractive-looking
number. The job here is to make that number hard to overinterpret.

## Iteration 43: Classifier Attack - A Candidate Judge Is Not A Language Model

### Current Strongest Position

Evidence-Native Sutra is now the mainline:

```text
context + retrieved evidence + candidates -> learned judgment margin
```

The strongest defense is that this is the right object. HellaSwag, PIQA, ARC,
and many deployment decisions are already candidate-selection tasks. If small
models cannot store the world closed-book, then a 121M byte-native model should
learn the geometry of evidence-conditioned choice.

### Steelman

The old interface was wrong. Byte autoregressive likelihood asks:

```text
Which surface string is most likely byte by byte?
```

The benchmark asks:

```text
Which candidate is the best continuation or action?
```

Those are different random variables. The toy frozen-readout result proved this
can matter: a model may contain a useful distinction that the byte decoder
cannot express. A native judgment head is therefore not a cheap classifier add-on
by default. It may be the correct output channel for intelligence under small
model constraints.

Evidence-native training also aligns with the pivot:

- no private teacher coordinates;
- no byte-marginal KD;
- no claim that 121M weights memorize all commonsense;
- public evidence and inspectable decision margins;
- failures can be repaired by adding evidence or lessons.

If the prototype trains directly on evidence-conditioned judgments, it may
finally optimize the thing HellaSwag measures instead of another proxy.

### Attack

This can become a multiple-choice classifier with a noble backstory.

The mapping:

```text
(context, evidence, candidate_1..candidate_4) -> label
```

is exactly the supervised format of many benchmark hacks. A model can learn:

- candidate length artifacts;
- dataset-specific ending style;
- lexical overlap between context and candidate;
- answer-position priors;
- HellaSwag's adversarial-ending generator artifacts;
- teacher-rationale phrasing artifacts;
- PIQA's typical "more practical" option style.

That is useful classification. It is not yet language modeling, generation, or
general intelligence.

The Work Loop prototype will probably find that direct candidate scoring gives
an immediate lift over closed-book byte likelihood. That is the easiest lift to
get because the output target is now aligned with the benchmark. But the first
adversarial reviewer will ask:

```text
Can this system do anything except choose among four benchmark-shaped options?
```

If the answer is no, the moonshot has shrunk from "small intelligence" to
"small open-book multiple-choice solver."

### Prototype Prediction

Likely first result:

- HellaSwag rises above the 26-29% codec/Wide7 zone, perhaps into the low 30s,
  if training examples are formatted as evidence + four candidates.
- PIQA sees a smaller lift because its evidence is less script-like and the
  task often needs affordance judgment rather than retrieved text.
- ARC may move only when retrieval returns near textbook facts.
- WinoGrande/free-form generation will likely stay weak unless explicitly
  trained.

This would be a real engineering signal, but not a Vision-level signal.

### What Survived

Survived:

- A judgment head is a legitimate output interface for candidate decisions.
- HellaSwag-style evaluation should not be forced through byte likelihood if
  the research question is decision quality.
- Evidence-native training can be a better benchmark-aligned objective than
  byte CE alone.

### What Died

Dead:

- Claiming HellaSwag lift from candidate classification as language-model
  intelligence without additional evidence.
- Treating an evidence-conditioned MCQ head as equivalent to a generative model.
- Any headline that says "121M language model beats X" when the system is a
  task-shaped candidate scorer.

### Controls And Gates

The classifier attack only fails if the prototype passes all of these:

- Candidate-order randomization with no answer-position leakage.
- Candidate-length, unigram, and style-feature baselines reported.
- Evaluation on adversarial candidate sets where all candidates have matched
  length, overlap, tense, and syntactic compatibility.
- Leave-format-out testing: train on HellaSwag-style four-way records, test on
  binary PIQA, ARC answer choices, WinoGrande, and generated candidate pools.
- Generative bridge: use the judge to rerank continuations sampled from the
  model or a small generator, then show improved continuation quality over byte
  likelihood alone.
- Free-form audit: after selecting an answer, ask the system to emit a concise
  evidence-grounded explanation or continuation, and score whether the
  explanation cites the decisive evidence rather than answer artifacts.

Minimum survival gate:

```text
The judge must improve at least three different decision formats, and the
reranked/generated continuation test must show a real gain over byte likelihood.
```

### Narrative Attack

1. "That's obvious" dismissal: Training a model on the exact benchmark output
   format improves the benchmark.
2. "That's trivial" dismissal: This is supervised multiple-choice classification
   with retrieved text attached.
3. What would beat the attack: The same evidence-conditioned geometry improves
   candidate selection, reranking, and generation across task formats.

### Gossip-Magazine Headline

The newborn learned to circle A, B, C, or D and called it wisdom.

## Iteration 44: Retriever Attack - The Library May Be Doing The Thinking

### Current Strongest Position

After Iteration 43, the defense is:

```text
Yes, the first interface is candidate judgment, but the learned judge still has
to bind evidence to candidates. The intelligence is in evidence use, not in
being generative from day one.
```

### Steelman

This is a serious defense. Retrieval alone does not decide the answer unless
the system can:

- identify which evidence sentence is relevant;
- ignore plausible but irrelevant overlap;
- map evidence to candidate consequences;
- handle negation, temporal order, affordances, and causal scripts;
- maintain calibrated margins when evidence is partial.

The learned object can be:

```text
evidence-conditioned decision geometry
```

not mere task classification. If the same retriever feeds all methods and Sutra
beats overlap, BM25 scores, nearest-neighbor labels, and small rerankers, then
the judge is doing work.

### Attack

The retriever may still be the whole system.

BM25 or a dense retriever can surface a passage that contains a near-paraphrase
of the right continuation, an activity script, or a strongly correlated phrase.
Then a dumb rule can win:

```text
choose the candidate with highest lexical overlap with retrieved evidence
choose the candidate whose words are most common in retrieved docs
choose the candidate closest to the nearest retrieved training example's label
choose the candidate with the highest BM25/corpus co-occurrence score
```

The Work Loop prototype will be tempted to compare:

```text
closed-book Sutra vs evidence-native Sutra
```

That is the wrong comparison. The honest comparison is:

```text
same corpus + same retriever + no learned judge
same corpus + same retriever + tiny learned judge
same corpus + same retriever + 121M Sutra judge
```

If a dumb overlap baseline gets most of the lift, then the learned geometry is
decorative.

### Prototype Prediction

Likely first result:

- Gold/oracle evidence will make the task look very solvable.
- Retrieved evidence will produce a noisy lift.
- A BM25/overlap baseline will be embarrassingly competitive on examples where
  retrieval succeeds.
- The neural judge may beat dumb baselines on a subset, but the subset will be
  hard to characterize without evidence-quality buckets.

The first prototype may therefore show:

```text
large lift over closed-book, small lift over same-retriever baselines
```

That is not enough.

### What Survived

Survived:

- Retrieval is a valid way to externalize world knowledge.
- The learned judge matters only if it beats non-neural evidence-use rules under
  identical retrieval conditions.
- Evidence-quality slicing is mandatory.

### What Died

Dead:

- Any result that reports only closed-book vs open-book.
- Any claim that the judge is intelligent without same-retriever baselines.
- Any prototype where the retriever, corpus, or oracle evidence is allowed to
  own the gain while the 121M model gets the credit.

### Controls And Gates

Same-retriever baselines must include:

- Random candidate.
- Longest candidate.
- Candidate prior by label/position after randomization.
- Context-candidate lexical overlap.
- Evidence-candidate lexical overlap.
- BM25 score of context + candidate against retrieved docs.
- TF-IDF/logistic regression over context/evidence/candidate text.
- kNN over retrieved train examples with label transfer.
- Tiny MLP/ridge reranker over hand features.
- Tiny transformer classifier at 5M-20M parameters.

Required evidence ablations:

- No evidence.
- Retrieved evidence.
- Shuffled evidence from the same batch.
- Wrong-topic evidence with matched length and score.
- Top-k evidence with rank order scrambled.
- Oracle/gold evidence upper bound.
- Evidence with answer-like phrases masked.

Minimum survival gate:

```text
Sutra judge >= best same-retriever non-neural baseline + 5pp on HellaSwag
AND >= best <=20M same-input learned baseline + 3pp
AND shuffled/wrong-topic evidence removes most of the gain.
```

"Most" should be operationalized:

```text
retrieved_gain = acc(retrieved) - acc(no_evidence)
shuffled_gain <= 25% of retrieved_gain
wrong_topic_gain <= 25% of retrieved_gain
```

### Narrative Attack

1. "That's obvious" dismissal: Search found similar text.
2. "That's trivial" dismissal: A few overlap rules can use that text.
3. What would beat the attack: Same evidence, same retrieval, same corpus; the
   121M learned judge clearly beats all cheap ways of exploiting the retrieval.

### Gossip-Magazine Headline

The library whispered the answer while the judge took the trophy.

## Iteration 45: Leakage Attack - A Clean Baseline Can Still Be Contaminated

### Current Strongest Position

After Iteration 44, the defense is:

```text
We will use the same retriever for every method and require the learned judge to
beat dumb baselines. That isolates the model's contribution.
```

### Steelman

This is the right next move. Same-retriever controls remove the easiest
overclaim. If Sutra beats BM25 overlap, kNN label transfer, and tiny rerankers
using identical evidence, then at least the neural judge adds something.

The prototype can also report evidence sensitivity:

```text
correct evidence increases margin;
shuffled evidence collapses margin;
counterfactual evidence flips margin.
```

That begins to look like real evidence use.

### Attack

A contaminated corpus can make every method look scientific.

HellaSwag is derived from ActivityNet-style video caption contexts and
adversarial endings. If the retrieval corpus contains ActivityNet-adjacent
captions, YouTube descriptions, benchmark mirrors, scraped examples, or teacher
generated records that saw evaluation contexts, then the system is not using
commonsense evidence. It is using benchmark-neighbor evidence.

Leakage is not solved by saying:

```text
we did not put the eval split into the corpus
```

Near-duplicate leakage can arrive through:

- HellaSwag train records that are very similar to validation/test records;
- original ActivityNet captions;
- web pages that quote the dataset;
- HuggingFace dataset mirrors;
- benchmark solution blogs;
- teacher-generated evidence conditioned on eval contexts;
- paraphrases generated from contaminated teacher memory;
- retrieval corpora with unknown provenance and no date/source hashes.

The danger is worse for evidence-native than closed-book training because the
system is explicitly allowed to look things up at inference. If the corpus is
dirty, the moonshot becomes a benchmark lookup system.

### Prototype Prediction

Likely first result:

- A broad web/text corpus will retrieve surprisingly on-topic snippets for many
  HellaSwag contexts.
- Some snippets will contain exact or near-exact activity-script language.
- Removing obvious dataset strings will not remove all contamination.
- Teacher-generated evidence will look clean but may still encode teacher memory
  of benchmark examples or format artifacts.

The first prototype will not be leakage-proof unless leakage-proofing is treated
as part of the artifact, not an afterthought.

### What Survived

Survived:

- Same-retriever baselines are necessary.
- Evidence sensitivity remains meaningful only after corpus provenance is clean.
- Teacher-generated evidence can be used for training, but not as untrusted
  eval-time evidence without hard separation.

### What Died

Dead:

- Any evidence-native HellaSwag number from an unhashable or weakly documented
  corpus.
- Any use of eval/test contexts to generate evidence, rationales, hard negatives,
  or retrieval queries outside the strict evaluation protocol.
- Any claim of commonsense reasoning when near-duplicate corpus retrieval has not
  been audited.

### Controls And Gates

Corpus requirements:

- Every document has source, timestamp if available, license/provenance, and
  stable hash.
- Exclude HellaSwag, PIQA, ARC, WinoGrande, their mirrors, and known benchmark
  solution text from the corpus.
- Exclude ActivityNet-derived caption sources for HellaSwag evaluation unless
  they are used only in a separately labeled contaminated upper-bound condition.
- Freeze corpus before evaluation and publish corpus manifest hashes.
- Separate train-only evidence generation from validation/test evidence
  retrieval. No teacher generation conditioned on held-out labels.

Leakage audits:

- Exact substring search for contexts and candidates.
- 8-gram and 13-gram overlap against all eval contexts/candidates.
- MinHash/SimHash near-duplicate scan at document and sentence level.
- Retrieval-neighbor manual audit sample for top-k retrieved docs.
- Train/eval cross-neighbor audit: evaluate whether train examples retrieve
  validation/test analogs with shared scripts.
- Performance bucketed by maximum n-gram overlap and near-duplicate score.

Minimum survival gate:

```text
After removing all retrieved evidence above the precommitted overlap threshold,
Sutra retains at least 70% of its retrieved-evidence gain and still beats
same-retriever baselines.
```

A stricter moonshot gate:

```text
At least one benchmark result must come from a corpus whose provenance makes
benchmark-neighbor leakage structurally implausible, not merely undetected.
```

Examples: curated public manuals/textbooks for ARC, hand-built commonsense
evidence records from train-only sources, or fresh generated evaluation records
whose source material was not in the retrieval corpus.

### Narrative Attack

1. "That's obvious" dismissal: The answer was in the lookup table.
2. "That's trivial" dismissal: You built a nearest-neighbor benchmark engine.
3. What would beat the attack: A frozen, hashed, leakage-audited corpus and
   retained gains after all high-overlap evidence is removed.

### Gossip-Magazine Headline

The open-book exam went well because the answer key was hiding in the library.

## Iteration 46: Transfer Attack - A Leak-Proof HellaSwag Win Can Still Be A Hack

### Current Strongest Position

After Iteration 45, the defense is:

```text
We can make the corpus leakage-proof, run same-retriever baselines, and show
the judge's evidence sensitivity. Then HellaSwag is a real signal.
```

### Steelman

That would be much stronger than anything so far. A clean corpus, hard controls,
and a learned judge beating baselines would establish that evidence-native is
not merely Brainseed with retrieval attached.

It would also match the Vision better than closed-book memorization:

```text
public evidence + small learned judgment + repairable memory
```

This is a serious candidate for cheap, improvable intelligence.

### Attack

HellaSwag alone is still the wrong proof object.

Evidence-native training may learn a HellaSwag-specific decision surface:

- contexts are short activity descriptions;
- endings are four natural-language continuations;
- wrong endings are adversarial but stylistically odd;
- evidence often consists of similar event scripts;
- the right answer frequently preserves physical/social continuity.

That does not prove transfer to:

- PIQA affordance choice;
- ARC factual/scientific question answering;
- WinoGrande pronoun resolution;
- open-ended generation;
- multi-hop evidence use;
- unseen domains or user-provided evidence.

If the system requires one retriever, one corpus, one serialization, one
teacher curriculum, and one head per benchmark, then it is not a general
evidence-native geometry. It is a set of benchmark-specific solvers.

### Prototype Prediction

Likely first result:

- HellaSwag shows the largest lift because event-script retrieval maps naturally
  onto its format.
- PIQA improves less because evidence has to connect object affordances to
  practical outcomes.
- ARC is split: Easy may improve with factual snippets; Challenge remains hard
  because distractors require reasoning and grade-school science context.
- WinoGrande stays near chance unless the model learns coreference and
  discourse, not just evidence overlap.
- Free-form generation remains mostly unchanged unless the judgment head is
  integrated into decoding.

The prototype may therefore pass a HellaSwag gate and still fail the moonshot.

### What Survived

Survived:

- A clean HellaSwag win is a meaningful first signal.
- Evidence-native should be evaluated as a cross-task judgment method, not a
  single benchmark intervention.
- Transfer is the difference between a benchmark system and a research thesis.

### What Died

Dead:

- "HellaSwag improved, therefore evidence-native works."
- Per-benchmark retriever/corpus/head tuning as moonshot evidence.
- Treating event-script judgment as equivalent to general evidence-conditioned
  reasoning.

### Controls And Gates

Transfer must be built into the first serious artifact:

- One shared serialization for context, evidence, and candidates across
  HellaSwag, PIQA, ARC, and WinoGrande.
- One shared judge architecture; no per-benchmark head unless reported as a
  weaker tuned condition.
- Train-on-HellaSwag, evaluate zero-shot or few-shot on PIQA/ARC slices.
- Train mixed curriculum, evaluate leave-one-benchmark-out.
- Evidence-type slices: script, factual, affordance, coreference, causal,
  temporal, negation.
- Counterfactual evidence tests across all benchmarks.
- Calibration transfer: confidence should mean similar correctness across tasks.

Minimum survival gate:

```text
Evidence-native Sutra must beat the best same-retriever baselines on at least
three benchmark families, with no single family contributing more than half of
the aggregate lift.
```

Stronger gate:

```text
Leave-one-benchmark-out training preserves at least 50% of the mixed-training
lift on the held-out benchmark.
```

Free-form gate:

```text
The judgment head must improve reranking of generated continuations or answers
on held-out prompts, not only official candidate sets.
```

### Narrative Attack

1. "That's obvious" dismissal: HellaSwag has event-script artifacts.
2. "That's trivial" dismissal: You trained the exact benchmark format.
3. What would beat the attack: One evidence-conditioned judge transfers across
   task families and improves generated outputs.

### Gossip-Magazine Headline

The genius was brilliant at one exam because it studied the shape of that exam.

## Iteration 47: Overkill Attack - 121M Parameters May Be Wasted On A Reranker

### Current Strongest Position

After Iteration 46, the defense is:

```text
If the same 121M evidence-conditioned judge transfers across benchmarks, then
it is not just a HellaSwag hack.
```

### Steelman

That would be a major improvement. A shared small judge that uses retrieved
evidence across different decision formats starts to look like the public
geometry Batch 6 wanted:

```text
evidence close to supported candidates;
irrelevant evidence ignored;
counterfactual evidence flips margins;
decision rules survive task format.
```

At that point, the system may be worth the 121M core because the core is
learning reusable decision procedures.

### Attack

A much smaller classifier may do the same thing.

If the input already contains:

- the context;
- retrieved evidence;
- four candidate answers;
- teacher-generated rationales or margins;
- benchmark-specific labels;

then the neural task is no longer "small language model intelligence." It is
supervised reranking over rich features. A 5M to 20M cross-encoder, a logistic
regression over overlap features, or a frozen embedding + MLP classifier may
match the 121M judge.

If that happens, Sutra is not the mechanism. The dataset is.

The adversary's line will be:

```text
You used 121M parameters to learn what a tiny reranker learns.
```

This attack is especially dangerous because the previous Brainseed history
already showed learned scorers can be misleading. A larger judge can hide
spurious correlations longer than a tiny baseline.

### Prototype Prediction

Likely first result:

- A small transformer classifier trained on the same evidence records will get
  a large fraction of Sutra's gain.
- If evidence is strong, the small model may match Sutra.
- If evidence is weak, both will fail.
- Sutra's advantage, if any, will show in transfer, adversarial evidence, or
  low-data sample efficiency, not raw in-domain HellaSwag.

If Work Loop B6 does not include small learned baselines, it will not know
whether 121M mattered.

### What Survived

Survived:

- A 121M judge is justified only if it learns reusable structure that smaller
  rerankers cannot.
- The model-size claim must be measured by a scaling curve, not assumed from the
  architecture.
- Sample efficiency and robustness may matter more than peak in-domain score.

### What Died

Dead:

- "121M is impressive" if 5M gets the same result.
- Reporting only closed-book Wide7 vs evidence-native Wide7.
- Ignoring tiny supervised baselines because they are not philosophically
  interesting.

### Controls And Gates

Size baselines:

- Logistic regression over hand features.
- 1M-5M MLP over TF-IDF/BM25/evidence overlap features.
- 5M byte or token cross-encoder.
- 20M transformer cross-encoder.
- 50M transformer cross-encoder if feasible.
- 121M Sutra judge.
- 121M randomly initialized token-model judge if available.

All must use:

- same corpus;
- same retriever;
- same evidence strings;
- same training records;
- same train/validation/test splits;
- same label budget.

Report:

- accuracy;
- calibration;
- evidence sensitivity;
- transfer;
- adversarial evidence robustness;
- training examples needed to reach each accuracy threshold;
- inference latency and memory.

Minimum survival gate:

```text
121M Sutra must beat the best <=20M same-input classifier by >=3pp aggregate
AND by >=5pp on at least one transfer/adversarial slice where dumb evidence use
should fail.
```

Alternative survival gate:

```text
If peak accuracy is tied, Sutra must reach the same score with <=25% of the
training labels while preserving better transfer/calibration.
```

Kill gate:

```text
If a <=20M classifier is within 2pp on aggregate and matches the evidence
sensitivity/transfer profile, evidence-native Sutra is not the moonshot.
```

### Narrative Attack

1. "That's obvious" dismissal: Large classifiers beat smaller classifiers.
2. "That's trivial" dismissal: Actually, a tiny classifier did the same thing.
3. What would beat the attack: A scaling curve showing 121M learns something
   reusable that smaller same-input models do not.

### Gossip-Magazine Headline

The tiny model brought 121 million parameters to a reranker fight.

## Iteration 48: Open-Book Fairness Attack - The Honest Opponent Also Gets Evidence

### Current Strongest Position

After Iteration 47, the defense is:

```text
Maybe 121M really does beat tiny rerankers. That would prove the judge is not
just an overbuilt classifier.
```

### Steelman

That would be valuable. A 121M model beating both dumb baselines and tiny
classifiers under the same evidence would show that model capacity adds
meaningful judgment.

The system could then claim:

```text
small enough for laptop deployment;
knowledge outside weights;
judge inside a compact neural core.
```

### Attack

The honest opponent also gets evidence.

Comparing evidence-native Sutra to closed-book CBD, closed-book SmolLM2, or
closed-book Pythia is not a scientific win. It is an open-book system versus
closed-book models.

The honest comparison is:

```text
Evidence-Native Sutra
vs CBD-with-evidence
vs Qwen3-0.6B-with-evidence
vs SmolLM2-with-evidence
vs a standard open-book reranker
```

If Qwen3-0.6B with the same retrieved snippets and a simple prompt gets 65-75%
on HellaSwag/PIQA-style decisions, then the 121M judge may be useful but not
stop-scrolling. The story becomes:

```text
small open-book model is worse than ordinary small open-book model
```

The Vision does not require beating 0.6B models in every absolute condition, but
it does require an extraordinary result. Cost-efficiency can count only if it is
measured honestly.

### Prototype Prediction

Likely first result:

- Qwen3-0.6B or another competent token model with retrieved evidence will beat
  the first Sutra evidence-native judge by a wide margin.
- SmolLM2-with-evidence may also improve substantially.
- CBD-with-evidence, if available, becomes the brutally fair benchmark because
  CBD already wins closed-book.
- Sutra's best defense will be latency/VRAM/updateability, not raw accuracy.

If Work Loop B6 only compares against closed-book baselines, it will overstate
the result.

### What Survived

Survived:

- Open-book Sutra may still be valuable if it is much cheaper and reasonably
  close to stronger open-book models.
- A 121M evidence-native judge can be a deployment win even if it is not the
  strongest absolute model.
- The comparison must separate scientific novelty from engineering utility.

### What Died

Dead:

- "Beat CBD" unless CBD gets the same retrieval/evidence condition or the
  open-book advantage is stated in the headline.
- Any table that mixes closed-book and open-book numbers as if they are the
  same race.
- Any claim that evidence-native proves superior intelligence merely because
  it beats closed-book baselines.

### Controls And Gates

Open-book baselines:

- Qwen3-0.6B with same retrieved evidence, same candidate order randomization,
  and fixed prompt.
- SmolLM2-135M with same evidence if feasible.
- Pythia-160M with same evidence as weak token baseline.
- CBD-with-evidence if checkpoint or published-compatible reproduction exists.
- Standard sentence-transformer/cross-encoder reranker if allowed, clearly
  labeled as external model baseline.
- Chain-init Sutra baseline when available.

Report:

- accuracy;
- latency;
- peak memory;
- tokens/bytes processed;
- index size;
- retriever model size;
- total system footprint;
- whether external tokenizers/embedding models are used at inference.

Minimum survival gate:

```text
Sutra must either:
  (A) beat Qwen3-0.6B-with-evidence or CBD-with-evidence on at least one
      precommitted task family, OR
  (B) reach >=90% of their aggregate open-book accuracy at <=25% of their
      inference memory/latency footprint, while beating all dumb/small
      same-retriever baselines.
```

Moonshot gate:

```text
Evidence-native Sutra >=42.65% HellaSwag is not enough if Qwen3-0.6B with the
same evidence is far higher. The claim must be either absolute win or
measured efficiency-adjusted win.
```

Kill gate:

```text
If Qwen3-0.6B-with-evidence beats Sutra by >10pp aggregate and Sutra has no
large efficiency, updateability, or byte-native robustness advantage, the
mainline should be killed or demoted to product engineering.
```

### Narrative Attack

1. "That's obvious" dismissal: Open-book beats closed-book.
2. "That's trivial" dismissal: A better small model with the same book wins.
3. What would beat the attack: Sutra either wins under the same open-book
   condition or gives a clearly superior accuracy-per-resource tradeoff.

### Gossip-Magazine Headline

The open-book champion forgot that everyone else could open the book too.

## Iteration 49: Byte-Native Attack - If Search Uses Tokens, Where Is Sutra?

### Current Strongest Position

After Iteration 48, the defense is:

```text
Even if bigger open-book models are stronger, Sutra can still win as a compact,
byte-native, tokenizer-free evidence judge with better updateability and laptop
economics.
```

### Steelman

This is the last viable version of the story.

The evidence-native system can be important if it provides:

- cheap local inference;
- public editable corpus;
- no proprietary tokenizer dependency;
- byte-level robustness to spelling, OCR, code, mixed scripts, and arbitrary
  inputs;
- a small learned judge that can be updated by adding evidence rather than
  retraining weights.

The Vision never said the model must win by closed-book memory. It said cheap,
democratic, improvable intelligence. Byte-native evidence use can serve that
vision if the byte substrate matters.

### Attack

The byte-native advantage may be accounting fiction.

If the retriever is BM25 over tokenized text, or a standard embedding model, or
Qwen/SBERT-derived vectors, then the semantic addressability is not byte-native.
The evidence is plain text. The training labels come from teachers. The task is
multiple-choice classification. The byte model is only the final scorer.

An adversary can say:

```text
You built a normal text retrieval/classification pipeline and swapped in a byte
I/O model at the end.
```

The codec itself is also dangerous for the narrative. It maps bytes toward Qwen
embedding identity. That can be useful, but if the system's semantics depend on
Qwen's tokenizer geometry, then the project has not escaped tokenizer lock-in.
It has hidden the tokenizer behind a learned adapter.

The Work Loop prototype will probably use ordinary retrieval first because it
is the fastest way to test evidence-native judgment. That is fine for an
engineering prototype. It is not enough for the byte-native claim.

### Prototype Prediction

Likely first result:

- Retrieval will be BM25, text-tokenized, or use an off-the-shelf embedding
  retriever.
- The judge will consume byte strings, but the retrieval path will already have
  performed most of the semantic narrowing.
- Token-model baselines with the same evidence will be easier to run and likely
  strong.
- Byte-native robustness will not be measured in the first prototype.

So the first prototype will probably not prove the byte-native part of the
Vision even if it proves some evidence-use value.

### What Survived

Survived:

- The first prototype may use standard retrieval as scaffolding.
- The byte-native claim must be evaluated separately from evidence-native
  judgment.
- The codec can remain infrastructure only if its dependence on Qwen geometry is
  honestly labeled.

### What Died

Dead:

- Calling the whole system byte-native if retrieval, indexing, and evidence
  selection are token-model dependent.
- Claiming tokenizer freedom when the semantic bridge is just a soft imitation
  of one teacher tokenizer.
- Treating byte input/output as a sufficient differentiator.

### Controls And Gates

Byte-native proof must include:

- A tokenized-retrieval condition and a byte-native retrieval condition reported
  separately.
- A no-external-embedding inference condition where retrieval does not depend on
  a larger semantic model.
- Robustness tests on typos, OCR noise, Unicode, code-like strings, rare words,
  mixed formatting, and tokenizer-hostile inputs.
- Same-evidence comparison against a same-size token model.
- Codec ablation: raw bytes vs codec features vs tokenized baseline.
- Corpus-update test: add new evidence in raw byte form and show immediate
  behavior change without tokenizer retraining or model finetuning.

Minimum survival gate:

```text
If the headline says byte-native, Sutra must show at least one clear advantage
over a same-size token/evidence judge on tokenizer-hostile or updateability
tests while retaining comparable benchmark accuracy.
```

Kill gate:

```text
If the best system requires a standard token/embedding retriever, loses to a
same-size token classifier on normal text, and shows no robustness/updateability
advantage, byte-native should be demoted to implementation preference rather
than moonshot claim.
```

### Narrative Attack

1. "That's obvious" dismissal: Text retrieval plus a classifier is common.
2. "That's trivial" dismissal: The bytes are just an input encoding tax.
3. What would beat the attack: Byte-native evidence use gives robustness,
   updateability, or efficiency that a tokenized evidence judge cannot match.

### Gossip-Magazine Headline

The tokenizer-free revolution used a tokenizer-shaped ladder to climb onto a
regular retrieval system.

## SYNTHESIS: After 49 Total Question Iterations

Evidence-Native Sutra remains alive only as a hard, narrow claim:

```text
A 121M byte-native model can learn evidence-conditioned judgment that beats
dumb retrieval, small rerankers, and fair open-book baselines enough to justify
the architecture, while transferring beyond one benchmark and surviving leakage
controls.
```

It is not alive as a vibe. It is not alive as "RAG but small." It is not alive
as "HellaSwag went up."

### What The Work Loop Prototype Will Probably Find

Prediction:

- Retrieved/gold evidence will lift HellaSwag above closed-book Wide7/codec.
- The first lift will be largest on HellaSwag and weaker on PIQA/ARC/WinoGrande.
- Dumb overlap and kNN baselines will explain more of the lift than expected.
- Shuffled and wrong-topic controls will reduce performance, but may not reduce
  it all the way to no-evidence baseline because format and candidate artifacts
  remain.
- Small learned classifiers will be competitive unless the evidence requires
  genuine multi-sentence binding.
- Qwen3-0.6B-with-evidence will likely beat the first Sutra judge by a wide
  margin.
- The first prototype will not prove byte-native advantage because standard
  text retrieval will likely be used.

This is not a reason to stop. It is a reason to precommit the gates before the
first attractive number appears.

### Exact Conditions Under Which Evidence-Native Sutra Survives A Fresh Adversarial Reviewer

Evidence-native survives only if the project can present a reproducible bundle
with all of the following.

#### 1. Honest Evaluation Framing

- Closed-book and open-book numbers are reported separately.
- No table implies that open-book Sutra and closed-book CBD/SmolLM2/Pythia are
  the same race.
- System footprint includes model params, retriever params, index size, corpus
  size, memory, latency, and any teacher/embedding model used at inference.

#### 2. Leakage-Proof Evidence

- Corpus manifest is frozen and hashed before evaluation.
- Benchmark mirrors, known solution text, ActivityNet-adjacent HellaSwag sources,
  and near duplicates are excluded or reported as contaminated upper bounds.
- Exact substring, n-gram, MinHash/SimHash, and retrieval-neighbor audits are
  run.
- Performance is reported after removing high-overlap retrieved evidence.
- No teacher generates evidence or rationales from held-out labels or eval
  answers.

Survival threshold:

```text
After leakage filtering, Sutra retains >=70% of its retrieved-evidence gain.
```

#### 3. Retriever Contribution Is Separated From Judge Contribution

Same corpus and same retrieved evidence are used for:

- random/length/position baselines;
- lexical overlap;
- BM25/TF-IDF scoring;
- kNN label transfer;
- logistic/ridge/MLP feature baselines;
- <=20M learned rerankers;
- 121M Sutra judge.

Survival threshold:

```text
Sutra >= best non-neural same-retriever baseline + 5pp on HellaSwag
AND Sutra >= best <=20M same-input learned baseline + 3pp aggregate.
```

Negative-control threshold:

```text
shuffled_gain <= 25% of retrieved_gain
wrong_topic_gain <= 25% of retrieved_gain
```

#### 4. Transfer Beyond One Benchmark

Survival threshold:

```text
Sutra beats best same-retriever baselines on at least three benchmark families
among HellaSwag, PIQA, ARC, WinoGrande, and free-form reranking.
```

Additional requirement:

```text
No single benchmark may provide more than half of the aggregate improvement.
```

Stronger reviewer-proof condition:

```text
Leave-one-benchmark-out training preserves >=50% of the mixed-training lift on
the held-out benchmark.
```

#### 5. Open-Book Baselines Are Fair

Same retrieved evidence must be given to:

- Qwen3-0.6B or comparable small token LM;
- SmolLM2-135M if feasible;
- Pythia-160M as weak token control;
- CBD-with-evidence if available;
- chain-init Sutra fallback when ready.

Survival threshold:

```text
Sutra either beats a stronger open-book baseline on a precommitted task family,
or reaches >=90% of its aggregate open-book accuracy at <=25% of its
inference memory/latency footprint.
```

If this condition fails, evidence-native may still be a useful prototype, but
not the moonshot mainline.

#### 6. It Is Not Merely A Classifier

Survival threshold:

```text
The same judge improves at least one generated/reranked free-form task over byte
likelihood alone.
```

Acceptable first version:

- Generate N candidate continuations or answers from a fixed small generator.
- Rerank with evidence-native Sutra.
- Show improved correctness/quality versus generator likelihood, dumb overlap,
  and tiny reranker baselines.

#### 7. Byte-Native Claim Is Earned Separately

Survival threshold for the byte-native headline:

```text
Sutra shows a clear advantage over same-size token/evidence judges on at least
one tokenizer-hostile or updateability axis while retaining comparable normal
benchmark accuracy.
```

Tokenizer-hostile axes include typos, OCR, rare words, Unicode, code-like text,
mixed formatting, and raw-byte corpus updates without retraining a tokenizer.

If this fails, the correct wording is:

```text
evidence-native small judge with byte I/O
```

not:

```text
byte-native intelligence breakthrough
```

### Exact Conditions Under Which Evidence-Native Sutra Should Be Killed

Kill the moonshot mainline, or demote it to a utility product, if any of these
hold after a fair prototype:

1. **No real lift.**

```text
Retrieved evidence improves Sutra by <3pp over no-evidence baseline on the
primary task.
```

2. **Retriever explains the lift.**

```text
Sutra is within 2pp of the best same-retriever dumb/feature baseline.
```

3. **Tiny classifier explains the lift.**

```text
A <=20M same-input learned classifier is within 2pp aggregate and matches the
transfer/evidence-sensitivity profile.
```

4. **Negative controls do not collapse.**

```text
Shuffled or wrong-topic evidence preserves >50% of the retrieved-evidence gain.
```

5. **Leakage cannot be ruled out.**

```text
Corpus provenance is unhashable, near-duplicate audits fail, or high-overlap
evidence removal destroys the result.
```

6. **HellaSwag-only success.**

```text
The lift does not transfer to at least two other task families or free-form
reranking.
```

7. **Open-book baselines crush it.**

```text
Qwen3-0.6B-with-evidence or CBD-with-evidence beats Sutra by >10pp aggregate
and Sutra has no large measured cost/updateability/robustness advantage.
```

8. **No byte-native advantage.**

```text
The best system depends on token/embedding retrieval, loses to same-size token
judges, and shows no tokenizer-hostile robustness or raw-byte update advantage.
```

9. **Only benchmark-shaped classification works.**

```text
The model improves official MCQ formats but does not improve generated/reranked
answers or explanations under held-out evidence.
```

### Final Verdict

Evidence-Native Sutra is still the right thing to attack next because it is the
only path that honestly answers the failure history:

```text
do not force 121M weights to memorize the world;
do not worship teacher coordinates;
teach a small model to use public evidence.
```

But the adversarial bar is high. The prototype must prove:

```text
retrieval is not the hero;
leakage is not the source;
classification artifacts are not the trick;
tiny rerankers are not enough;
open-book baselines are not silently stronger;
bytes are not mere branding.
```

If it proves those, evidence-native survives a fresh reviewer.

If it does not, the moonshot mainline should move to chain-init or a larger
Sutra-family inherited-coordinate path, and evidence-native should be demoted
to an application layer.

### Final Gossip-Magazine Headline

The small model may yet learn to judge evidence, but first it has to prove the
evidence, the search engine, the benchmark, and the branding are not doing the
thinking for it.
