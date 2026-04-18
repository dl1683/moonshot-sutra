# Information-Theoretic Mental Models for Knowledge Transfer

**Purpose:** Alternative frameworks for thinking about teacher→student knowledge transfer when classical KD (match softmax outputs) plateaus. Each model is designed to be *grabbable*: read one, run with it.

---

## TL;DR

Classical KD treats the teacher as an **oracle of probabilities** and the student as a **function approximator** — a category mistake that ignores the teacher's role as a **compressor of the data-generating process**. The information-theoretic insight ML is most under-using: **knowledge is a short program, not a probability table**. The teacher has already paid the entropy cost of discovering structure — transfer should move *that structure* (the compression), not its shadow (the output distribution). Three mental shifts flow from this: (1) distill the *code*, not the *codeword*; (2) KD is a **channel coding problem** with the teacher as the encoder and training data as a noisy channel; (3) the right loss is often asymmetric (directional KL, rate-distortion front) because knowledge flows one way.

---

## Model 1: MDL / Kolmogorov — Distill the Program, Not the Output

**The model in one sentence.** The teacher is a short program that reproduces the data; the student's job is to compress that program further, using the teacher's outputs as *hints about program structure*, not as ground truth.

**What we'd BORROW.** Treat every teacher forward pass as a **partial execution trace** of the underlying generative program. The knowledge we want to transfer isn't `p_teacher(y|x)` — it's the *computational path* the teacher took. Concrete translations: (a) penalize student description length directly (explicit MDL loss: `L = NLL + λ · ||θ||_MDL`), (b) match **sparsity patterns** of teacher activations (which neurons fire together is the program structure), (c) distill into a *smaller hypothesis class* before training, not via loss alone — structural KD.

**Toy thought experiment.** We train Sutra-Dyad with a loss that is 70% next-byte prediction and 30% *teacher-activation-sparsity match* (matching which teacher MLP neurons are ≥0 vs <0, not their magnitudes). Training looks like the student developing the same *circuit topology* as the teacher even when the magnitudes differ. We'd expect: (i) student loss curve has a kink where it "discovers" the teacher's decomposition, (ii) pruning-robustness goes up (we inherited the compressed program), (iii) OOD generalization improves because we inherited *structure* not *memorization*.

**What this reframes.** KD-as-softmax-matching optimizes for *the same output for the same input*. MDL-KD optimizes for *the same reasoning for the same input*. The former is extensional (behavioral cloning), the latter is intensional (algorithmic cloning). Implication: teacher output on hard examples may be *less* informative than teacher internals on easy examples — reverse the usual intuition.

**Testable prediction.** Take a 50M student. Train one copy with standard KL-KD from a 1.5B teacher, train another copy matching only sign-patterns of teacher MLP activations (no logit matching). The sign-pattern student should (a) reach lower compression on held-out text despite never seeing teacher probabilities, and (b) have higher circuit-overlap with the teacher under mechanistic interpretability probes. If it matches within 5% BPB, the "program transfer" hypothesis is supported. 1-day experiment.

---

## Model 2: Rate-Distortion — The Optimal Curriculum Is On The Frontier

**The model in one sentence.** For any fixed student capacity (rate R), there is an *optimal teacher-to-distill-from* that sits on the rate-distortion curve — distilling from a teacher off this curve is strictly wasteful.

**What we'd BORROW.** Treat student capacity as channel capacity. For Sutra-Dyad at 188M, the information we can actually absorb per training token is bounded. A 70B teacher gives us signal we *cannot fit* — most of its entropy is wasted. The right teacher isn't "the biggest one" but the one at the **matched rate**. Concrete translations: (a) **capacity-matched ensembling** — many small diverse teachers whose *union* is at our rate, (b) **distortion-targeted KD** — distill only the coordinates of teacher output where student capacity is not yet saturated, (c) **rate schedules** — anneal which teacher features to match as student rate grows during training.

**Toy thought experiment.** Instead of "KD from Qwen-7B for all tokens," we maintain a running estimate of per-token student entropy `H(p_student(·|x))`. Tokens where student is already low-entropy → no KD signal (it's already on-manifold, teacher adds noise). Tokens where student is high-entropy → full teacher match. Training looks like **adaptive KD density** — 30% of tokens distilled at start, shrinking as student sharpens. Expect: faster convergence, less over-smoothing, better calibration (we stop pulling confident-correct predictions toward teacher's softer ones).

**What this reframes.** The default assumption "more teacher signal is better" is false — past a threshold, teacher signal *competes* with data signal for the same rate budget. KD and data training are adversarial resource consumers, not allies. Implication: **KD weight should anneal down**, not stay constant — once the student's rate is saturated by structure, additional teacher signal is pure distortion.

**Testable prediction.** Run 3 KD schedules at identical total compute: (a) constant α=0.5, (b) linear decay α=0.5→0.0, (c) entropy-gated (KD only when student entropy > threshold). Predict: (c) > (b) > (a) on held-out BPB by a margin that grows with training length. Cost: 3 × 10K step runs.

---

## Model 3: Information Bottleneck — Compress X, Preserve Y, Ignore the Teacher's Irrelevant Bits

**The model in one sentence.** The student's hidden state T should minimize I(X;T) while maximizing I(T;Y) — and the teacher's job is to tell us *which bits of X are irrelevant*, not what Y is.

**What we'd BORROW.** Teacher provides a **relevance mask** over input bits. The normal KD loss asks "what should T map to?" — IB asks "what of X should T *forget*?" Concrete translation for Sutra-Dyad: use teacher attention entropy / teacher gradient norm w.r.t. each input byte as a **relevance weight** for each input position. Bytes the teacher ignores should be *compressed away* by the student's early layers. This is KD as **input-side pruning signal**, not output-side matching.

**Toy thought experiment.** At each layer of Sutra-Dyad, we add a bottleneck regularizer `β · I(X; T_l)` estimated via a variational bound. The teacher's per-byte saliency (from integrated gradients or attention) modulates β per-position — high-salience bytes get low β (preserve them), low-salience bytes get high β (compress them). Training looks like the student developing a *learned bytes-to-keep mask* that matches the teacher's implicit one. Expect: sharper early layers, flatter later layers, improved long-context retrieval (we stopped wasting capacity on filler bytes).

**What this reframes.** KD as currently practiced transfers *output beliefs*. IB-KD transfers *input attention*. These are different knowledge: the former is "what is the answer?" and the latter is "what is the question actually about?" For byte-level models especially, the second is dominant — most bytes are linguistic glue. Implication: a teacher that's *wrong about Y* but *right about what X-bits matter* is still valuable — a use case classical KD can't express.

**Testable prediction.** Compute teacher per-byte saliency on a 1M-token corpus. Train two students: (a) standard KL on teacher output, (b) no output KD but byte-level L2 between student attention entropy and teacher saliency. Predict: (b) reaches within 0.1 BPB of (a) at half the teacher FLOPs, AND has better retrieval accuracy on needle-in-haystack. 2-day experiment, uses our existing teacher cache.

---

## Model 4: Channel Coding — Teacher as Encoder, Training Data as Noisy Channel

**The model in one sentence.** The teacher encoded knowledge into its parameters; the training corpus is a lossy channel re-transmitting that knowledge to the student; KD is the decoder design problem — and Shannon tells us we need *redundancy*, not fidelity.

**What we'd BORROW.** Reframe KD as error-correcting decoding. The student doesn't observe teacher knowledge directly — it observes *teacher outputs on training inputs*, which is a noisy encoding. The fundamental move: add **structured redundancy** to the transmission. Concrete translations: (a) query the teacher on **adversarially perturbed inputs** — repetition codes through the data distribution, (b) require student outputs to satisfy **consistency constraints** the teacher satisfies (e.g., logit differences across paraphrases — parity checks), (c) **soft-decision decoding** — weight teacher outputs by our estimate of their reliability (teacher confidence × teacher calibration).

**Toy thought experiment.** For each training example x, we query the teacher on x and on k=3 augmentations {x'_1, x'_2, x'_3} (paraphrases, typos, byte-drops). The student is trained to match the *consensus* teacher output, with high weight on points where the k+1 teacher outputs agree (low channel noise) and low weight where they disagree (high channel noise). Training looks like **teacher-confidence-weighted distillation** but derived from Shannon capacity, not heuristics. Expect: robust to teacher hallucinations, automatically down-weights teacher's weak points.

**What this reframes.** The assumption "teacher output is ground truth for KD" — it's not, it's the *output of a noisy channel* and should be treated as such. The moment you see teacher-KD as a decoding problem, *every* trick from coding theory becomes available: belief propagation, turbo decoding, iterative refinement, soft-decision vs hard-decision. We've been using the ML equivalent of the Hamming code when LDPC exists.

**Testable prediction.** Two KD setups on Sutra-Dyad: (a) teacher query per example, (b) k=3 augmented queries with variance-weighted loss. Predict: (b) closes 2x more of the teacher-student gap per FLOP of KD. Additionally: (b)'s residual errors will be *less correlated* with teacher errors — we've broken out of the teacher's failure modes. 3-day experiment.

---

## Model 5: Relative Entropy Geometry — KL Is Directional And We've Been Using It Wrong

**The model in one sentence.** KL divergence is an asymmetric statistical manifold distance; `KL(teacher || student)` and `KL(student || teacher)` are **different experiments** with different geometric consequences — and neither is "correct," they answer different questions.

**What we'd BORROW.** Explicit use of *forward* vs *reverse* KL as a design knob, plus natural-gradient thinking. Forward KL (`KL(T || S)`) is **mean-seeking** — student covers every mode of teacher, including teacher's noise. Reverse KL (`KL(S || T)`) is **mode-seeking** — student commits to one teacher mode, sharper but potentially over-confident. Classical KD uses forward KL by default because it falls out of softmax cross-entropy; this is an accident, not a choice. Concrete translations: (a) explicit reverse-KL or mixed (α·forward + (1-α)·reverse) loss, (b) **natural gradient** updates — move in Fisher metric on the output simplex, not in parameter space, so each step is matched in bits, (c) **Rényi-divergence KD** — tune α in D_α(S||T) to trade off mode coverage vs sharpness.

**Toy thought experiment.** On Sutra-Dyad, switch KD from forward KL to Jensen-Shannon (symmetric but non-geodesic) and separately to reverse KL. We'd see: forward-KL student is a "muddled average" teacher (high diversity outputs, lower accuracy), reverse-KL student is a "decisive minority" teacher (committed to specific answers, higher top-1 but worse top-k diversity), JS student interpolates. Natural-gradient runs would converge in fewer steps because each update is a fixed-information-size move.

**What this reframes.** The default KD loss is mean-seeking and makes the student a *confused average* of teacher modes — which is *exactly what we don't want* for a small model that can't afford to hedge. For a 188M student, committing to teacher modes (reverse KL) may outperform mean-seeking forward KL by wide margins on reasoning tasks where the teacher is confidently correct. This is a free hyperparameter we've been ignoring.

**Testable prediction.** Three KD losses at identical compute: forward KL (baseline), reverse KL, JS. Measure held-out BPB *and* top-1 accuracy on a structured-answer probe (multiple choice via PMI). Predict: reverse KL wins on top-1, forward KL wins on BPB, and the gap reveals whether our primary metric (BPB) is even measuring what we want. 1-day experiment — this should have been run a month ago.

---

## Cross-Pollination

- **MDL (Model 1) ↔ Evolutionary Biology.** Short programs = high-fitness genotypes. Teacher activation patterns as the student's "developmental program" mirrors Waddington's epigenetic landscape — canalization of phenotypes via shared ancestral structure. Connects to self-constructing intelligence work from the parent project.
- **Rate-Distortion (Model 2) ↔ Neuroscience.** Cortical feedback gain control looks like an adaptive rate allocator. The brain doesn't maximally attend to teachers — it attends to its own prediction error. Our "entropy-gated KD" is the computational analog of predictive coding.
- **Information Bottleneck (Model 3) ↔ Thermodynamics.** IB objective is formally equivalent to a free energy minimization — β is inverse temperature. Relates to Landauer-limit arguments about the energetic cost of preserving irrelevant bits. A student that ignores irrelevant bytes is literally more thermodynamically efficient.
- **Channel Coding (Model 4) ↔ Collective Intelligence.** k-augmentation teacher queries = ensemble voting = quorum sensing in bacteria. The noisy-channel frame connects directly to the Ekalavya multi-teacher protocol: multiple teachers are a repetition code *across architectures*, which dominates repetition across inputs.
- **Relative Entropy Geometry (Model 5) ↔ Optimal Transport & Differential Geometry.** Natural gradient is geodesic flow on the Fisher manifold; Wasserstein-based KD is geodesic flow on a different manifold. The choice of divergence *is* the choice of geometry — which is the choice of "what does small mean?" The Intelligence = Geometry thesis lives here most directly.
