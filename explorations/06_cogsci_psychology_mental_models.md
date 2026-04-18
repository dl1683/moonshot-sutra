# Cognitive Science & Psychology Mental Models for Knowledge Transfer

*A portfolio of alternative frameworks from how humans actually learn — as opposed to how we happen to train neural networks.*

---

## TL;DR — The Common Thread

The neuroscience portfolio (01) looked at learning at the **mechanism** level — synapses, circuits, replay. This one looks one level up, at **behavior and cognition**: how a learner who already has a mind comes to know something new. And here, the single most glaring absence from our KD setup is this:

**ML's default KD has no model of the student.** It treats the student as a passive distribution-matcher. But every serious theory of human learning — Piaget, Vygotsky, Ericsson, Bjork, Gentner — starts from the opposite stance: *learning is what the learner does when the learner's existing model fails*. The teacher is useful only insofar as they can diagnose the student's current understanding, find its edges, and supply precisely the signal that would break or extend it. Our current KD pipeline doesn't diagnose the student at all. It just pours teacher logits over every token uniformly — as if you tried to teach a child calculus by reading Rudin aloud at full speed from page 1.

The cogsci reframing: **a teacher is a model of the student's wrong model, not a library of correct answers.** Every model below is a different way to give KD a theory-of-student.

This is the most absent primitive in ML KD: **the student's current belief-state as an explicit first-class object that the teacher conditions on.**

---

## Model 1: Zone of Proximal Development (Vygotsky) — Teach at the Edge of Competence

**Core idea (one sentence):** Learning happens in the narrow band between "what the student can already do alone" and "what the student cannot yet do even with help"; outside that band, instruction is wasted — too easy is rehearsal, too hard is noise.

**What we'd BORROW:**
Per-token — not per-batch, per-token — decide whether the teacher should intervene at all. Define three regions by student confidence on that token: (1) **already competent** (student top-1 correct, high confidence) → no teacher signal, skip; (2) **in the ZPD** (student top-5 contains truth, confidence moderate, teacher confidence high) → apply heavy KD; (3) **beyond reach** (student's top-10 doesn't include truth, teacher also uncertain, or the token depends on context the student hasn't integrated yet) → use only hard-label CE, skip soft KD to avoid fitting noise. The KD loss becomes a **masked, gated** objective where the mask is derived from the (student, teacher) joint state.

**Toy thought experiment — Sutra-Dyad:**
At each step, for each token, compute: student top-1 probability `p_s`, teacher top-1 probability `p_t`, and whether student's top-k includes teacher's top-1. Classify every token into {skip, ZPD, too-hard}. Hypothesis: on Sutra-Dyad at 153M bytes, probably **40-60% of tokens are already competent** (common English prefixes, whitespace, punctuation), **25-35% are ZPD** (the actual learning signal), and **15-25% are too-hard** (code, rare entities, long-range dependencies the model can't yet resolve). If we apply KD *only on the ZPD 30%*, we get roughly the same effective signal at 30% of the teacher-forward FLOPs — and more importantly, **we stop wasting gradient on the already-known and on the currently-unreachable.**

**What this reframes:**
The default KD assumption "teacher signal is always beneficial" is the pedagogical equivalent of "reading Rudin to a toddler never hurts." It does hurt — it burns compute, floods easy-token gradients, and forces the student to fit teacher noise on tokens beyond its reach. The ZPD reframe: **teacher signal is only beneficial in a narrow confidence band**; outside that band, the signal is either redundant or harmful. Our BPB plateau at ~1.40 may simply be that 70% of our KD loss is either rehearsal or noise.

**Testable prediction (smallest falsifier):**
Measure the per-token KD gradient contribution on a Sutra-Dyad checkpoint. Sort tokens by `gap = p_t_top1 - p_s_top1` (teacher-student confidence delta) and by `hit = student_topk_contains_teacher_top1`. Hypothesis: **most of the useful KD gradient concentrates in a narrow middle band of `gap`** (neither tiny nor extreme). If the gradient signal is roughly uniform across `gap` values, ZPD doesn't apply — but if it's peaked in the middle third, masking the outer thirds should match or beat uniform KD at substantially lower teacher FLOPs. 2 GPU-hours.

---

## Model 2: Schema Theory — Assimilation vs. Accommodation (Piaget)

**Core idea (one sentence):** Knowledge arrives in two regimes: **assimilation** (new info fits an existing schema, cheap, incremental) or **accommodation** (new info breaks the schema, forcing it to restructure, expensive, discrete); the learner's job is to route each new piece of information correctly.

**What we'd BORROW:**
Give the student an explicit, low-dim **"schema signature"** per layer — e.g., a running exponential moving average of the subspace spanned by recent activations. For each new (teacher, student) disagreement, classify: does this disagreement **fit** the student's current schema (teacher's answer lies within the activation subspace)? Or does it require a **dimension the student doesn't yet have** (teacher's answer points orthogonally out of the current subspace)? Route the two cases to **different losses and different learning rates**: assimilate with small LR on existing weights; accommodate by allocating new capacity (rank-1 LoRA expansion, or gating a previously-dead feature).

**Toy thought experiment — Sutra-Dyad:**
At step 5K, take the student's layer-8 representations on a 10K-token eval set. Compute the top-32 PCA directions (the "current schema"). For each teacher-student disagreement, project the gradient onto this subspace. Tokens where >95% of the gradient lies inside the subspace = assimilation-type: the student has the machinery, just needs to strengthen it. Tokens where >30% of the gradient lies *outside* the subspace = accommodation-type: the student is missing a representational axis. Hypothesis: Sutra's plateau at 1.40 BPB is dominated by **accommodation-type gradients that have no allocated capacity to absorb them** — the loss keeps pointing in a direction the weights can't rotate into because the rank is already used up. This predicts that **adding tiny capacity (a LoRA rank-4 adapter) precisely on the orthogonal residual** should unstick the plateau faster than any amount of extra data.

**What this reframes:**
The default KD assumption treats all loss signals as fungible — more gradient = more learning. Schema theory says **gradient that lies in an existing direction is 100x cheaper to absorb than gradient that requires a new direction**, and a stalled loss often means the learner is pinned on the accommodation wall, not that training has "converged." This is also **mechanistically aligned with Outcome 2 (surgical improvability)**: if we can *identify* the accommodation directions, we've identified exactly where new plug-in capacity needs to go. The teacher stops being a loss and starts being a **diagnostician of missing dimensions.**

**Testable prediction (smallest falsifier):**
At the plateau (step 6K, BPB ≈ 1.40), compute the fraction of gradient norm in layer 8 that lies outside the top-32 PCA directions of recent activations. If this fraction is small (<10%), accommodation isn't the bottleneck — the plateau is elsewhere (optimizer, LR, data). If this fraction is large (>30%), then adding a rank-4 LoRA aligned to the top residual direction should drop BPB by 0.05+ within 500 steps at negligible compute. 3 GPU-hours including the LoRA training.

---

## Model 3: Analogical Transfer / Structure-Mapping (Gentner) — Relations Over Surface Features

**Core idea (one sentence):** Transfer is governed by **relational structure** (who does what to whom, in what order) rather than surface features (the specific entities or tokens); a learner who has grasped "A is to B as C is to D" can apply a schema from a known domain to a novel one.

**What we'd BORROW:**
Stop aligning student logits to teacher logits. Align **structural relationships** — specifically, the pattern of which tokens attend to which, or the pattern of which context tokens most *change* the prediction when ablated. Define a "relational fingerprint" per position: e.g., the top-k influence graph (which previous bytes, when perturbed, most change the next-byte distribution). The KD objective becomes **"match the teacher's influence graph, not its logits."** Two models can have very different logits but identical relational structure — and the relational structure is what actually generalizes.

**Toy thought experiment — Sutra-Dyad:**
Pick 1000 tokens from held-out text. For each, compute the teacher's influence map: perturb each of the previous 256 bytes and record the KL shift in the next-byte distribution. This gives a sparse attention-like influence matrix. Compute the same for the student. Train Sutra-Dyad with a KD loss that penalizes divergence of the *influence graphs* (e.g., earth-mover distance on sparse influence vectors) rather than divergence of the logits themselves. Hypothesis: structural KD is **a much lower-bandwidth signal** than logit KD (64 bytes of influence structure vs. 256 floats of logit) but carries the **generalizing part** of what the teacher knows — the "who-depends-on-whom" shape — while being agnostic to the specific byte vocabulary match.

**What this reframes:**
The default KD paradigm is pure **surface mimicry** — we're asking the student to have the same output distribution, which is the shallowest possible form of "knowing what the teacher knows." Structure-mapping theory says the deep, portable form of knowledge is relational. If we align relational structure, the student can share structure with teachers of **entirely different vocabularies, tokenizers, and even architectures** — which is exactly the Ekalavya problem. Cross-tokenizer KD isn't solved by clever projection layers; it's solved by **giving up on matching surface outputs** and matching relational invariants instead.

**Testable prediction (smallest falsifier):**
Compute the Spearman correlation between teacher and student influence maps on a 1K-token eval set, for (a) the current Sutra-Dyad and (b) a version trained additionally with an influence-graph KL loss for 1K steps. If (b) has substantially higher influence-map correlation but *no BPB improvement*, structure-mapping has a nice theoretical property that doesn't cash out — KD needs surface alignment too. If (b) has higher influence correlation AND comparable or better BPB on held-out text that uses *different byte-level patterns* (e.g., code, non-English), structure-mapping generalizes better and we've found the Ekalavya substrate. 4 GPU-hours.

---

## Model 4: Desirable Difficulty (Bjork) — Friction Is Feature, Not Bug

**Core idea (one sentence):** Retention and transfer are *improved* by introducing processing difficulty at encoding time — spacing, interleaving, variability, generation effort — because easy learning produces brittle, context-bound traces while effortful learning produces robust, transferable ones.

**What we'd BORROW:**
Systematically **inject difficulty** into the KD signal. Three concrete levers: (1) **Interleaving** — instead of blocked batches from one data source, interleave conceptually adjacent-but-distinct sources (web text, code, math, dialogue) at the token level, forcing the model to constantly re-select context; (2) **Spacing** — when a rare pattern appears in training, deliberately *space out* subsequent occurrences rather than clustering them in the same batch, even if it means holding tokens in a buffer; (3) **Generation before reception** — for each teacher-supervised token, force the student to *first* produce its own prediction (hard sample from its own distribution), compare, and only *then* receive the teacher signal. The student pays an encoding cost that biology says produces deeper learning than passive mimicry.

**Toy thought experiment — Sutra-Dyad:**
Run two 8K-step Sutra-Dyad variants at matched data and FLOPs: (A) standard blocked batches, standard teacher forcing; (B) interleaved batches (4 domains mixed per sequence at chunk boundaries) + spacing buffer for rare byte n-grams (track 4-gram frequency in a sliding window; delay over-frequent 4-grams) + generation-before-reception on 20% of tokens (student samples, loss is on *both* CE against truth *and* KL against teacher given student sample). Hypothesis: (B) has **higher** train loss at any given step (difficulty slows memorization) but **lower** held-out BPB, especially on out-of-distribution segments, and **better generation quality** (the most important metric per your feedback that BPT ≤0.1 is noise).

**What this reframes:**
The default ML assumption is that **training loss should go down as fast as possible** — fewer steps to target is better. Bjork's decades of work says the opposite: **fast-dropping training loss is a warning sign**, often predictive of poor generalization and poor retention under distribution shift. The plateau at 1.40 BPB might not be a bug at all — it might be a model that has taken the *easy* route and is now stuck at a brittle local representation that low-difficulty training can't escape. Injecting desirable difficulty means accepting slower early progress for better late-training representation geometry. This is deeply aligned with Intelligence=Geometry: **geometric quality is traded for training ease by default**; difficulty injection buys back geometric quality.

**Testable prediction (smallest falsifier):**
Train (A) standard and (B) with interleaved 4-domain batches only (the cheapest lever). At step 4K, (B) should have **higher** train loss but **equal or lower** held-out BPB on domain-mixed text. More strongly: (B) should have **lower variance** of BPB across domains (the whole point of interleaving is domain-robust representations). If (B) has higher train loss AND higher held-out BPB AND equal cross-domain variance, desirable difficulty is wrong for this setup — the model simply needs more capacity, not harder training. 3 GPU-hours with our existing data-loader machinery.

---

## Model 5: Teaching as Model-Repair (Cognitive Tutoring) — The Teacher Owns a Model of the Student

**Core idea (one sentence):** A skilled human tutor doesn't recite content — they maintain an evolving model of the *student's wrong model*, predict where the student will err next, and supply interventions designed to repair specific misconceptions; the teacher's output is a **function of the student's state**, not a fixed curriculum.

**What we'd BORROW:**
Invert the KD flow. Today, teacher is stateless and the student queries it. Proposal: add a **student-model** — a small network (could be 1-10M params) whose job is to predict the student's output distribution *from a compressed representation of the student's internal state*. Train this student-model online as training proceeds. Then, at each training token, the teacher's supervision is *gated by the student-model's prediction*: when the student-model predicts the student will get it right, reduce KD weight; when it predicts the student will err in a specific way, up-weight KD *and* potentially pick from a **menu of teachers** the one whose output best counters that predicted error. The teacher is selected **per-token, conditioned on the student's predicted behavior.**

**Toy thought experiment — Sutra-Dyad + Ekalavya:**
Maintain 3 teachers with distinct strengths (a fluent-English teacher, a code-specialist teacher, a math-symbolic teacher). Train a 5M-param "student predictor" that maps `student_hidden_state → predicted_student_logits`. At each token: (i) student predictor forecasts student's output, (ii) compare forecast to ground truth to predict student's error type, (iii) pick the teacher whose strength matches the error type, (iv) apply KD only from that teacher, only on tokens where the student-predictor forecasts a non-trivial error. This is **conditional, student-aware, multi-teacher KD** — which is exactly what Ekalavya needs, and which no existing multi-teacher KD scheme does.

**What this reframes:**
Classical KD treats the student as a **sink** — signal flows teacher → student, student's current state is irrelevant to what the teacher emits. Cognitive tutoring says the student is the **primary context** for every pedagogical decision. This inverts the polarity of the whole pipeline: the teacher becomes a *function of the student*, not the other way around. It also **naturally solves the overfitting-to-one-teacher problem** — the teacher-selector routes different tokens to different teachers based on predicted student error, so no single teacher dominates. And the `student_predictor` itself is a valuable artifact: it's a **self-model**, usable for uncertainty estimation, for deciding inference-time depth (Outcome 5: easy tokens exit early = "the self-model says I'll get this right"), and for identifying failure modes surgically (Outcome 2).

**Testable prediction (smallest falsifier):**
Train a 5M-param student-predictor for 1K steps on a frozen Sutra-Dyad checkpoint, with targets = the student's own logits. Measure: can this predictor forecast the student's correctness with >70% AUC on held-out tokens? If yes, we have a cheap, trainable self-model — and gating KD on its predictions is immediately testable. If no (AUC ≤ 60%), student behavior is too high-dimensional to cheaply predict, and tutoring-style conditional KD is too expensive to implement. 2 GPU-hours.

---

## CROSS-POLLINATION

- **ZPD ↔ Information theory (Portfolio 02):** The ZPD is literally the region of maximum *information gain* per token — tokens outside the ZPD contribute near-zero mutual information between teacher and student update. Computing optimal teacher-signal allocation under a FLOPs budget is the rate-distortion problem applied to pedagogy. The Blahut-Arimoto iteration **is** a ZPD scheduler.

- **Schema theory ↔ Neuroscience critical periods (Portfolio 01, Model 4):** Accommodation = opening a new plasticity window; assimilation = operating within an existing window. The Piagetian framing gives the *cognitive content* that critical periods provide the *mechanism* for. Together they suggest: detect accommodation-type gradients, dynamically open a plasticity window (allocate LoRA capacity + raise LR locally), train briefly, close the window.

- **Structure-mapping ↔ Mathematics (Portfolio 05) and category theory:** Analogical transfer via relational correspondence is literally a **functor** between source and target categories, preserving the composition structure. The influence-graph KD loss is a functorial KD loss. This is also the natural home for **sheaf-based KD across tokenizers**: the sheaf condition requires agreement on overlaps, and relational structure is exactly the data that overlaps cleanly across different surface encodings.

- **Desirable difficulty ↔ Physics / statistical mechanics (Portfolio 03):** Fast-descending training loss = quenched annealing into a nearby local minimum; difficulty injection = keeping the temperature high enough to escape shallow minima and find deeper, flatter ones (wider basins = better generalization — the flat-minimum / SAM literature). Bjork's "desirable difficulty" is Boltzmann sampling with the right temperature schedule.

- **Teaching-as-model-repair ↔ Biology / collective intelligence (Portfolio 04):** The "student model" is the pedagogical analog of theory-of-mind in primates, and the mechanism that enables coordinated teaching in social learners. The teacher-selector routing tokens to specialists by predicted error matches the way eusocial colonies (ants, bees) route tasks to specialists by context — **a society of teachers, each with a model of the shared student.** This is the clean answer to Ekalavya: multi-teacher KD not as ensemble averaging, but as a **dispatch network over a self-model**.

- **All five ↔ The Ekalavya question:** Every one of these gives multi-teacher a natural shape that uniform logit-KD cannot: ZPD routes per-token to whichever teacher's signal falls in the student's learning band; schema theory routes per-direction to whichever teacher supplies the missing dimension; structure-mapping routes per-relational-motif across teachers regardless of vocabulary; desirable difficulty uses teacher diversity as the interleaving substrate; model-repair routes per-predicted-error to the matching specialist. **The current KD paradigm can't absorb multiple teachers because it has no conditioning variable on which to switch between them.** Cogsci's gift is the conditioning variable: **the student's state.** Multi-teacher KD isn't a loss-combining problem; it's a teacher-selection-given-student-state problem, and every model above is a different selection policy.

- **Meta-insight across the portfolio:** If neuroscience (01) says *change how signal enters the student*, cognitive science (06) says *change who decides what signal to send at all*. Both converge on the same move that is currently missing from ML KD — **closing the loop between student state and teacher signal**. Every model here is a different way to close that loop. The one most worth running with first, if I had to pick, is **Model 5 (Teaching-as-Model-Repair)**, because the `student_predictor` artifact it creates is independently useful for Outcomes 2 and 5 even if the conditional-KD story doesn't pan out — which makes it the lowest-regret bet.
