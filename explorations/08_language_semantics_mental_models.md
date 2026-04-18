# Linguistic & Semantic Mental Models for Knowledge Transfer

**Context.** Sutra-Dyad (188M byte-level). Classical KD (match teacher softmax) has plateaued. Neuroscience, information theory, physics, biology, and mathematics have each given us a lens. This file adds the lens that is, on paper, most embarrassingly missing from the KD literature: **philosophy of language and linguistics**. Language models are language systems. We distill them as if they were function approximators.

---

## TL;DR — What is classical KD missing, linguistically?

Classical KD treats the teacher's output as a **literal, context-free proposition** at every token: "the probability of byte b at position t is p". It then asks the student to parrot the proposition. Linguistics has known for 70 years that this is not how meaning works:

- **Meaning is compositional, not enumerative** (Frege) — KD hands over examples; the actual prize is the *composition rule*.
- **Meaning is use, not description** (Wittgenstein) — the teacher's distribution is a record of a *game being played*; matching the distribution without playing the game transfers the trace, not the competence.
- **Utterances are acts, not facts** (Austin/Searle) — a teacher's next-token prediction is a *speech act* (assertion, correction, continuation, prediction). KD collapses all illocutionary force to "probability".
- **What's said ≪ what's meant** (Grice, Sperber-Wilson) — a teacher's soft logits encode Gricean implicatures (contrast, salience, relevance) that KL-to-targets washes away.
- **Grammar comes from priors, not data** (Chomsky) — the student's inductive biases are doing more work than the examples; KD pours more examples into a bias vacuum.

Each of the 5 models below is one of those observations weaponised. At the bottom, cross-pollination maps how they reinforce each other.

---

## Model 1 — Poverty of the Stimulus / Universal Grammar (Chomsky)

**The model in one sentence.** Children acquire grammar from impoverished, noisy, negative-evidence-free data because they come pre-equipped with strong structural priors (Universal Grammar); the teacher's role is not to *provide examples* but to *select parameter settings* inside a pre-wired space of possible grammars.

**ML translation.** In KD, the student is the child and the teacher is the linguistic environment. Classical KD assumes the student is a *tabula rasa* and loads everything onto examples — which is *precisely* the stimulus that Chomsky argued is too poor to explain acquisition. The correction: give the student **structural priors** so the teacher's job shrinks from "teach the whole distribution" to "disambiguate which of a few pre-wired structures I should settle into". The student converges because its hypothesis space is shaped, not because the teacher is loud.

**Sutra thought experiment.** Hardwire a small, fixed set of **grammatical priors** into Sutra-Dyad at initialisation — not trained, not distilled, just *imposed*:
- **Hierarchical attention decay**: attention is biased to respect nested bracket structures (code, JSON, Markdown, parentheses, quotes) as an architectural constraint, not a learned pattern.
- **Locality prior**: a per-head power-law bias on distance — local syntax is cheap, global dependencies pay explicit cost.
- **Recursivity prior**: weight sharing across depth for "constituent" operations — the same small operator handles nested structure at every level.

Then distill from teachers. Hypothesis: the same teacher signal goes *much* further because it selects parameters inside a smaller, pre-shaped manifold, instead of trying to carve the manifold from scratch.

**What it reframes about KD.** The KD plateau is probably not a teacher problem. It is a **hypothesis-space problem**. Teachers are already rich. The student's manifold is too large and too flat, so teacher signal is a weak gradient on a featureless plain. The fix is not more/better teachers — it is making the student's initial hypothesis space *narrower along the right axes*. Distillation becomes disambiguation.

**Smallest testable prediction.** Add a hierarchical-locality architectural prior (e.g., ALiBi-style distance bias combined with a learned nesting-aware mask over brackets/quotes/indent) *before* KD. On Sutra-Dyad at 188M, this will either (a) reduce BPB at equal steps by ≥ 5% with zero change to the teacher loss, or (b) make multi-teacher KD finally show a decisive gap over single-teacher — *because the teachers are now selecting inside a constrained space instead of fighting over an unconstrained one*. Either outcome is an extraordinarily cheap one-line experiment.

---

## Model 2 — Distributional Semantics + Distribution-to-Distribution Maps

**The model in one sentence.** "You shall know a word by the company it keeps" (Firth 1957) — meaning is the *distribution of contexts*, not any single context; therefore the teacher's knowledge is a **distribution over distributions**, and the student should learn a **distribution-to-distribution map**, not a point-to-point map.

**ML translation.** Classical KD: `student(x) ≈ teacher(x)` — one input, one output vector, KL. This is a *point* match. But distributional semantics says the *meaning* of `x` is the neighbourhood `N(x)` — the set of contexts-and-continuations in which `x` is felicitous. The teacher implicitly encodes this via its output distribution across *perturbations* of `x`. A "good student" must match how teacher outputs **move** as input moves — the Jacobian, not the value; the local distribution, not the point.

**Sutra thought experiment.** For each training batch, draw a small cloud of **perturbations** of each input `x`: synonymy-preserving byte edits (casing, whitespace, paraphrase-by-teacher), truncations, context extensions. Collect the teacher's logit *responses* to the cloud. Instead of KL on each point, match the **second-order structure**:
- Match `E[T(x')]` (first moment — standard KD).
- Match `Cov[T(x')]` over the cloud (second moment — how does the teacher's distribution deform under input perturbation?).
- Match *paired contrasts* `T(x') − T(x'')` for matched perturbation pairs — this is a discrete Jacobian.

The student must reproduce not just what the teacher says *here*, but how what-it-says *responds to its neighbourhood*. Firth, in gradient form.

**What it reframes about KD.** KD thinks the teacher is a function. Distributional semantics says the teacher is a **differential operator** — what it knows is encoded in how its outputs *move*. A student that matches values but not local derivatives has copied the surface and missed the meaning. Every flat plateau in KD may be a place where values match but derivatives don't — a local isometry failure invisible to pointwise KL.

**Smallest testable prediction.** Add a **neighbourhood-consistency** term to Sutra-Dyad KD: for 10% of batches, draw 2-4 cheap perturbations (whitespace/case noise, context-length jitter) per sequence, and regularise `||(S(x₁)−S(x₂)) − (T(x₁)−T(x₂))||² `. At 188M, this closes a meaningful fraction of the plateau gap on *semantic* benchmarks (paraphrase robustness, cloze) while barely affecting BPB — because the teacher's *values* were already nearly matched; what was missing was the *gradient*.

---

## Model 3 — Wittgenstein's Language Games (Meaning as Use)

**The model in one sentence.** "The meaning of a word is its use in the language" (PI §43) — a teacher does not *describe* knowledge, it *plays* a language game; distillation must transfer the rules of the *game*, and the game is only visible when the student plays and is corrected, not when it reads the teacher's transcripts.

**ML translation.** Classical KD takes the teacher's static outputs (its transcripts) and fits to them. This is reading someone's chess games with the pieces greyed out — you see moves, not *why*. A Wittgensteinian KD makes the student **produce** utterances in the teacher's game and the teacher **respond to student outputs**, not just to fixed inputs. The transferred object is not the teacher's marginal distribution; it is the teacher's *reaction function* — `teacher(context, student_utterance) → correction`.

**Sutra thought experiment.** At training step `t`:
1. Student generates `k` next bytes autoregressively from context `c`.
2. Teacher(s) score the student's *completion* — not the student's per-byte distribution over the next byte, but the teacher's surprisal on the **student's own production**: `−log p_T(student_sample | c)`.
3. Student's loss is a mixture of (a) standard KD on teacher logits at `c`, and (b) reinforcement of its own samples weighted by how "in-game" the teacher judged them.

This is on-policy distillation / teacher-as-critic, but framed philosophically: the teacher is not a *label*, the teacher is **another player whose competence is revealed by how it reacts to your moves**. The game *is* the knowledge.

**What it reframes about KD.** A soft-label distribution is a **frozen transcript** of a game the teacher is no longer playing. Trying to reconstruct the game from the transcript is the same error as trying to reconstruct chess from a bag of isolated board positions without move histories. The living content is in the *reaction*, not the record. This is also the reason self-distillation, iterated amplification, and on-policy RLHF-distill loops keep beating classical KD: they restore the game.

**Smallest testable prediction.** Replace 20% of KD steps with a **teacher-reacts-to-student-sample** step: student samples 16 tokens; teacher returns logprob of that continuation; loss = `−logp_T(sample) − α · KL(S ∥ T) on prefix`. At equal compute on Sutra-Dyad 188M, this outperforms pure offline logit-KD on generation-quality metrics (distinct-n, chrF vs teacher, eval perplexity on held-out) — *especially* at long contexts where offline KD cannot anticipate the student's own distribution drift.

---

## Model 4 — Grice's Maxims / Relevance Theory (What's Meant vs What's Said)

**The model in one sentence.** Speakers are *cooperative pragmatic agents* — they obey maxims of Quantity, Quality, Relation, Manner (Grice 1975), and listeners infer the intended meaning via **implicature** that goes far beyond the literal utterance; teacher logits are *pragmatically shaped speech*, and student KD currently ignores pragmatics entirely.

**ML translation.** A teacher's softmax is not an honest confidence — it is a **pragmatically modulated signal**. When the teacher puts 0.6 on token A and 0.3 on token B, the *contrast* is the implicature: "A is preferred here, but B is salient and non-trivial — don't collapse to a delta." Classical KD with KL on the flat distribution matches the numbers and discards the implicature. A Gricean KD pays special attention to:
- **Quantity** — regions where the teacher is *deliberately non-peaked* (informative uncertainty) vs regions where it is sharp (committed).
- **Relation** — which non-top tokens are *relevant* contrastively vs *irrelevant* noise. The ranking of the runners-up matters; their exact probabilities do not.
- **Quality** — where the teacher is *honest about ignorance* (high entropy on low-frequency tokens) vs *confidently wrong* (overconfident on OOD).

Relevance theory (Sperber & Wilson) sharpens this: the teacher's distribution is the **maximally relevant signal** given its processing budget. KD should recover the *relevance structure*, not the raw numbers.

**Sutra thought experiment.** Decompose the teacher distribution at each position into:
1. **Top-k ranked set** (who's in the running) — a *discrete* implicature.
2. **Relative margins** inside the top-k (contrastive structure) — a *ranking* target.
3. **Tail mass** (total probability outside top-k) — a *confidence* signal.

Distill each component with its own loss, calibrated so that (1) and (2) dominate when the teacher is sharp, and (3) dominates when the teacher is flat. This is *pragmatic KD*: the student learns not just the marginal numbers but **what the teacher was trying to communicate by choosing that distribution shape**.

**What it reframes about KD.** KL divergence is a tone-deaf listener. It takes the teacher at face value — every probability is a literal claim — and misses the pragmatic structure that carries most of the usable information (what's contrastive, what's salient, what's tail noise). A human learning from a lecturer pays attention to *emphasis*, not decimal places. KD should too.

**Smallest testable prediction.** Replace plain KL-on-logits with a **three-component pragmatic loss**: (i) top-k set-overlap loss (Jaccard or listwise), (ii) pairwise margin loss inside top-k, (iii) MSE on `log(1 − Σ top-k)`. On Sutra-Dyad 188M, this matches or beats KL at equal steps *and* produces a student with better calibration (ECE ↓) because it was never asked to fetishise the exact tail values the teacher itself didn't care about.

---

## Model 5 — Frege's Compositionality (Distil the Composition Rule, Not the Outputs)

**The model in one sentence.** "The meaning of a complex expression is determined by the meanings of its constituents and the rule of combination" (Frege, compositionality principle) — teacher knowledge is a **combinator**: a small composition function applied recursively to primitives; distil the combinator, not the outputs.

**ML translation.** Classical KD transfers `teacher(x)` for whole sequences. But `teacher(x)` is itself the output of some **compositional process** the teacher runs over sub-expressions of `x`. If we could distil the *process* — how the teacher binds, applies, and recurses over parts — we would get a student that generalises to *new compositions* of primitives it was never distilled on. This is the **compositional generalisation gap** that every LM struggles with. KD currently makes it worse, because it distills sentences (complex wholes) and throws away the teacher's handling of *parts*.

**Sutra thought experiment.** Construct a distillation dataset that *explicitly varies compositional structure*:
- For each training context, generate systematic sub-expressions: `f(a)`, `f(b)`, `g(a)`, `g(b)`, and the composition `g(f(a))`.
- These can be concrete (arithmetic templates, logical templates, code snippets with renamed vars, sentence templates with substituted NPs/VPs).
- Distillation loss *includes a consistency term*: the teacher's (and student's) output on `g(f(a))` must be predictable from its outputs on `f(a)`, `g(b)`, `f(b)` via a **small learnable composition operator**. If the teacher has true compositional competence, this operator is nearly identity and low-rank.
- Student is penalised when its composition operator *differs* from the teacher's on held-out substitutions.

This is Frege at the level of the distillation loss. Instead of matching `teacher(sequence)`, match `(teacher's combinator over parts)`.

**What it reframes about KD.** The teacher's next-byte distribution on a long context is the *output* of a deep compositional process the teacher ran. Distilling the output without distilling the process is like distilling a factorial function by memorising `(n, n!)` pairs — you can match up to `n=10000`, but you have not copied the function. The KD plateau is most likely the point where the student has memorised the teacher's *outputs* on the training distribution and has no mechanism to acquire the teacher's *combinator*.

**Smallest testable prediction.** Augment Sutra-Dyad KD with a *compositional consistency loss*: for 5% of batches, construct minimal-pair substitutions in code / arithmetic / template text (these are trivially machine-generatable), and regularise `||teacher_combinator − student_combinator||` where both combinators are estimated from outputs on `(f(a), f(b), g(a), g(b), g(f(a)))`. At 188M, this will show a **decisive gap** (≥ 10 pp) specifically on compositional-generalisation benchmarks (SCAN, COGS, GSM8K's symbolic variants) while barely affecting raw BPB — revealing that BPB was blind to compositional skill all along.

---

## Cross-Pollination

- **Chomsky (1) ↔ Frege (5).** Universal Grammar is *the composition rule being innate*. Model 1 imposes it architecturally; Model 5 distils it from the teacher. They are the same object from opposite directions: a compositional operator that is either pre-wired (Chomsky) or acquired (Frege). Best system does both: use (1) to narrow the hypothesis space, use (5) to fix the operator inside it.

- **Wittgenstein (3) ↔ Grice (4).** Both reject the teacher-as-transcript view. Wittgenstein: meaning is in the *reaction*. Grice: meaning is in the *shape of the utterance given maxims*. Combined: on-policy distillation where the teacher scores the student's samples *and* the loss pays attention to pragmatic structure (ranking, margins, tail). This is RLHF-distill with a much richer critic signal.

- **Distributional (2) ↔ Compositional (5).** Historically adversarial ("vector meaning" vs "symbolic composition") but the synthesis is the whole point: distil the *distribution*'s *compositional structure*. Model 2 gives the local geometry of meaning; Model 5 gives the algebra that combines pieces of it. A student that passes both has learned the teacher's semantic space *and* its grammar.

- **Chomsky (1) ↔ Distributional (2).** Historically adversarial again (UG vs usage-based). The synthesis: architectural priors (1) shape *which directions in representation space* the distributional signal (2) can refine. Without (1), the distributional gradient is diffuse; with (1), it is concentrated on the parameters that actually matter.

- **Grice (4) ↔ Distributional (2).** Pragmatic structure is *exactly* the second-order structure of the teacher distribution — the contrastive rankings, the tail shapes — that Model 2 asks the student to match via neighbourhood derivatives. Model 4 gives names to what Model 2 targets.

- **Wittgenstein (3) ↔ Compositional (5).** Wittgenstein's later philosophy explicitly attacks naive compositionality. But the synthesis in ML terms is cleaner than the philosophical one: the *combinator* is learned *by playing the game*. Model 5 specifies the target (a composition operator); Model 3 specifies the training loop (on-policy). Together: *on-policy compositional distillation* — the student samples, the teacher reacts, and both sides' combinators are compared.

---

## If you pick ONE and run with it

**For the Ekalavya plateau specifically:** **Model 4 (Gricean pragmatic KD)** is the cheapest, highest-signal first experiment. It is a loss swap. No architecture change. No data pipeline change. It directly attacks the hypothesis that KL is a tone-deaf listener missing the pragmatic structure teachers already encode — and it is calibration-friendly, which the current student needs.

**For the deepest conceptual bet:** **Model 5 (compositional consistency)** is where real generalisation lives. It explains why every LM plateaus on compositional benchmarks even at scale, and why pure logit-KD cannot close the gap. Needs a templated minimal-pair data generator (~1 day of work), then a consistency term.

**For the highest-leverage prior-loading bet:** **Model 1 (poverty of stimulus)** is the one-line change with the largest potential multiplier. Hierarchical/locality priors *before* KD change the game board — teachers that were barely distinguishable suddenly differentiate because the student's hypothesis space is finally shaped to care.

**For the critique of all current KD:** **Model 3 (Wittgenstein)** is the hardest one to ignore. Every offline distillation run is, structurally, the error Wittgenstein diagnosed: mistaking the transcript for the game. Any on-policy signal from the teacher (even a cheap scalar) is probably worth more than another pass of offline logits.

**Model 2 (distributional derivatives)** is the workhorse companion to all four — it can be added as a neighbourhood-consistency term to any of them with near-zero incremental cost.

The linguistic insight ML-KD is missing, in one line: **teachers are speakers, not datasets — we need to distil them as speakers.**
