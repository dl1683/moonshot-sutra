# Computer Science Mental Models for Knowledge Transfer

Alternative framings of knowledge distillation (KD) through the lens of classical CS: algorithms, data structures, complexity theory, PL theory, type theory, and formal methods. The goal is not analogy for its own sake but to surface design axes that the thermodynamic / information-theoretic framings have been silently collapsing.

---

## TL;DR

Five CS lenses, each reframing a different latent assumption in our current KD pipeline:

1. **Abstract Interpretation** — The student is a *sound abstract interpreter* of the teacher's concrete semantics. KD loss should measure soundness (no false certainties) before it measures precision. Reframes teacher-student KL as "is the student's abstraction a valid over-approximation?"
2. **Cache Coherence / Memory Hierarchy** — Student = fast local cache, teacher = slow coherent store. KD is a *coherence protocol*, not a single-shot copy. Predicts that write-invalidate vs write-update style KD schedules yield measurably different generalization.
3. **Program Synthesis / Inductive Logic Programming** — Teacher's distribution is a *specification*; student must synthesize the shortest program consistent with it. Reframes token-level KL as a weak, high-entropy spec; counterfactual/structural supervision is a strong spec and should converge faster per bit.
4. **CRDTs (Conflict-Free Replicated Data Types)** — Multi-teacher KD = replica merge. If teacher contributions are designed as CRDT-shaped updates (commutative, associative, idempotent), Ekalavya ordering becomes irrelevant and consensus emerges for free.
5. **Delta Debugging** — The failures of a KD run are *minimal failing inputs* for the teacher-student contract. Systematic bisection of the data/teacher/loss triple is a more principled debugging loop than ablation scatter-plots.

The cross-cutting insight: **KD is not a loss function; it is a protocol.** Protocols have invariants, consistency models, soundness properties, and failure modes. We have been optimizing a scalar when we should be designing a distributed system.

---

## 1. Abstract Interpretation — The Student as a Sound Over-Approximator

**Mechanism (one sentence).** Abstract interpretation computes sound over-approximations of concrete program semantics by executing the program in a simpler abstract domain that is guaranteed to never miss a real behavior — precision is sacrificed, soundness is not.

**ML translation.** The teacher induces a "concrete semantics" on language: a high-resolution probability over continuations. The student, with fewer parameters, must approximate this semantics. Current KD implicitly asks for *precision* (match the teacher's exact distribution). Abstract interpretation asks instead for *soundness*: the student's distribution must cover the teacher's support, even if it is wider. A sound student never rules out what the teacher would have allowed. An unsound student hallucinates certainty the teacher never had.

**Sutra thought experiment.** Reformulate the KD objective as a *Galois connection* between student and teacher distributions. Replace forward KL `KL(T || S)` with a soundness penalty that only punishes the student when it assigns <ε mass to a token the teacher considers plausible. Precision (matching *low* teacher probabilities) becomes a secondary objective activated only after soundness is satisfied. At 153M bytes vs multi-billion-param teachers, precision-first KD is asking the impossible; soundness-first KD is asking the student to be a valid, widened abstract interpreter of the teachers.

**What it reframes.** Plateau phenomena in KD may not be "the student has learned all it can"; they may be "the student has hit the precision wall but is still unsound on a measurable fraction of tokens." Soundness and precision are *orthogonal* axes. The KD literature has been optimizing their sum without noticing.

**Smallest testable prediction.** Define "soundness coverage" = fraction of tokens where `S(y|x) ≥ α · T(y|x)` for all `y` with `T(y|x) > τ`. Plot soundness coverage vs. training steps alongside BPB. Prediction: BPB plateaus *before* soundness coverage does, and a soundness-weighted KD loss continues to improve held-out generation quality after plain KD has stalled.

---

## 2. Cache Coherence / Memory Hierarchy — KD as a Coherence Protocol

**Mechanism (one sentence).** Memory hierarchies keep a fast-but-small cache synchronized with a slow-but-authoritative store through coherence protocols (MESI, MOESI, directory-based) that define when to invalidate, update, write-back, or broadcast.

**ML translation.** The student is L1 cache: small, fast, local, sometimes stale. The teacher is main memory: large, slow, authoritative. KD is the coherence traffic between them. Every KD paper picks an implicit protocol — one-shot offline soft labels is "cache cold-fill, no coherence," online KD is "write-through," TAID-style adaptive targets is "write-back with dirty bits," multi-teacher KD is "directory-based with multiple owners."

**Sutra thought experiment.** Model Ekalavya as a directory-based coherence system. Each token position has a "directory entry" tracking which teachers have an opinion and how confident they are. The student consults the directory only on *cache misses* — positions where its own prediction is low-confidence. On hits (student is already confident and correct per cheap heuristic), no teacher traffic is incurred. This turns global KD loss into an *event-driven* protocol with O(misses) teacher forward passes rather than O(tokens).

**What it reframes.** The question "how much teacher supervision?" is actually "what is the coherence policy?" Strict coherence (every token, every step) is MESI — maximally consistent, maximally expensive. Relaxed consistency (eventual, per-block, per-phase) is what modern CPUs actually use, and yields huge wins in bandwidth. The same should be true of KD. We have been running strict coherence without measuring the bandwidth cost.

**Smallest testable prediction.** Implement a miss-driven KD schedule: compute teacher logits only for the top-k% of student-uncertain tokens per batch. Prediction: at 20% teacher compute, BPB matches full-teacher KD within 0.02 BPB, because the 80% of tokens the student is already confident on contribute nearly zero gradient signal. If true, this is a decisive compute win and an Ekalavya unlock (can afford more teachers when each teacher is queried 5× less).

---

## 3. Program Synthesis / Inductive Logic Programming — The Spec-Bandwidth View

**Mechanism (one sentence).** Program synthesis recovers a program from a specification; the spec can range from weak (input-output examples) to strong (logical pre/post-conditions, types, or partial programs), and synthesis difficulty scales inversely with spec strength.

**ML translation.** Current KD is synthesis-from-examples on the weakest possible spec: "for each input `x`, produce a distribution close to `T(·|x)`." There is no structural spec, no invariant, no type constraint. The student must induce the entire program (its internal computation) from billions of noisy I/O pairs. Strong-spec synthesis in ILP converges exponentially faster than weak-spec synthesis. KD has been stuck on weak spec by default.

**Sutra thought experiment.** Add *structural* specifications the teacher can express about its own computation: (a) invariances the teacher satisfies (permutation, negation-scope, arithmetic closure), (b) counterfactual predictions — "if token N were replaced with `y'`, the distribution at N+1 would shift by `Δ`," (c) compositional guarantees — "the distribution on `ab` factors through the distributions on `a` and `b` like so." These are richer specs than soft labels. A 153M student learning to satisfy 100 structural specs over 1B tokens is doing strong-spec synthesis; it should converge in fewer examples than weak-spec KD on 10B tokens.

**What it reframes.** Data efficiency in KD is not about "smarter examples"; it is about *spec bandwidth*. Every byte of the teacher's output that the student consumes carries some number of bits about the true program. Soft labels are low-bandwidth; counterfactuals are high-bandwidth; invariants are infinite-bandwidth (one statement constrains uncountably many examples). Sutra's data efficiency mandate maps directly to: maximize bits-per-teacher-query.

**Smallest testable prediction.** Compare two KD runs matched on teacher compute: (A) standard soft-label KD on 2× data, (B) soft-label KD + 3 counterfactual probes per example on 1× data. Prediction: (B) reaches lower BPB and better generation quality despite fewer teacher forward passes, because counterfactuals are higher-bandwidth specs.

---

## 4. CRDTs — Order-Independent Multi-Teacher Merging

**Mechanism (one sentence).** Conflict-free replicated data types are data structures whose updates are mathematically designed (commutative, associative, idempotent) such that any two replicas that have seen the same set of updates converge to the same state regardless of ordering or delivery.

**ML translation.** In standard multi-teacher KD, the loss is a weighted sum of per-teacher KLs — which means ordering of gradient updates, teacher weighting, and data interleaving all change the final student. This is not a CRDT; it is a *non-commutative* merge. Ekalavya's current fragility is a symptom: different teacher schedules give different students.

**Sutra thought experiment.** Design teacher contributions as CRDT-shaped updates. Each teacher produces not a distribution but a *constraint* — e.g., a convex set of admissible distributions, or a ranked list of tokens that must be in the top-k. Merge constraints by intersection (commutative, associative, idempotent). The student projects onto the intersection. Now teacher order, batching, and dropout are provably irrelevant: every valid training trajectory converges to the same student.

**What it reframes.** Multi-teacher instability is not a tuning problem; it is a *data structure* problem. The field has been treating teacher outputs as soft vectors to be averaged, when the right abstraction is *constraint sets to be intersected*. Intersection is a CRDT; averaging is not. The Hindi-branded Ekalavya protocol could be literally defined as a CRDT over teacher constraint sets.

**Smallest testable prediction.** Convert three teachers' outputs to top-k token sets per position, merge by intersection (fallback: union when intersection is empty), train student to place mass on the merged set. Prediction: the student is insensitive (within 0.05 BPB) to teacher ordering, weighting, and dropout schedules — whereas standard averaged-KD shows >0.2 BPB swings under the same perturbations.

---

## 5. Delta Debugging — Finding the Minimal Failing Contract

**Mechanism (one sentence).** Delta debugging bisects a failing input to find the smallest subset that still triggers the failure, turning "what went wrong?" into a systematic search with `O(log n)` queries instead of `O(n)` ablations.

**ML translation.** When a KD run plateaus or underperforms, we currently respond with scattered ablations: change the teacher, change the loss, change the data mix, change the schedule. This is `O(n)` and mostly noise. Delta debugging treats the KD run as a (config, data, teacher, loss) triple whose *output* is a failure signal (e.g., BPB > target on held-out), and bisects the triple.

**Sutra thought experiment.** For the current plateau: take the successful config from Stage 0 and the failing config from the current stage. Treat their symmetric difference as a set of atomic changes. Bisect: apply half the changes, measure, recurse. In `log₂(Δ)` training runs you have identified the minimal subset of changes that caused the plateau. Compare to current practice of ablating one axis at a time with no systematic bisection.

**What it reframes.** "Why did this KD run fail?" is a *program debugging* question, not a research question. It has a mature algorithmic answer. The ML community runs full ablation grids because it hasn't imported the 1999 CS result. This is an embarrassment of riches on our side: the method is off-the-shelf.

**Smallest testable prediction.** On the next KD plateau, run delta debugging between last-known-good and current-failing. Prediction: the minimal failing subset is ≤3 changes, and reverting any single change in that subset recovers ≥50% of the regression — meaning we will not need to re-run the full ablation matrix to identify the culprit.

---

## Cross-Pollination

The five lenses are not independent; they interlock along three axes that together describe a better KD system than any one of them alone.

**Axis 1 — Soundness, not precision (lenses 1, 3, 4).** Abstract interpretation says the student must be a sound over-approximator. Program synthesis says the teacher's spec should constrain the student, not mimic it. CRDTs say multi-teacher constraints intersect (a fundamentally sound-approximation operation). All three converge on: *KD should produce the most constrained student that is consistent with what the teachers know, not the student closest to their exact outputs.* This is a strict upgrade over forward KL.

**Axis 2 — Event-driven protocols, not static losses (lenses 2, 5).** Cache coherence and delta debugging both say: query expensively only where it matters. Coherence says: teacher calls only on student cache misses. Delta debugging says: ablations only along the bisection path. Both reject the "run full configuration matrices" default of modern ML. Combined, they suggest an *adaptive* KD pipeline that spends teacher compute where the student is uncertain and spends human/research compute where the failure is provably localized.

**Axis 3 — Abstractions as first-class objects (lenses 3, 4).** Program synthesis and CRDTs both elevate *structure* above *values*. A spec is higher-bandwidth than an example; a constraint is more composable than an average. This is the deepest CS insight available to KD: **data structures determine algorithmic complexity**. We are using the wrong data structure (scalar distributions) for a problem that has natural algebraic structure (constraints, types, invariants).

**A unifying hypothesis for Ekalavya.** Model Ekalavya as: (i) a CRDT over teacher constraint sets (axis 3), (ii) queried via a cache-coherence miss-driven protocol (axis 2), (iii) optimized for soundness before precision (axis 1), (iv) with structural (counterfactual, invariance) specs augmenting soft-label specs (axis 3), and (v) debugged via systematic bisection when plateaus occur (axis 2). Each piece is an off-the-shelf CS technique. The novelty is the *composition* — which is exactly what Sutra's "Intelligence = Geometry" mandate demands: better mathematics, not more compute.

**Connection to other explorations.** Lens 1 (abstract interpretation) rhymes with the information-theory lens on *sufficient statistics* — a sound abstraction is a kind of coarsened sufficient statistic. Lens 2 (cache coherence) rhymes with the biology lens on *attention as resource allocation* — only spend metabolic/teacher cost where signal exceeds noise. Lens 4 (CRDTs) rhymes with the dynamical-systems lens on *consensus dynamics* — Paxos, gossip protocols, and CRDTs are all algorithmic cousins of the same limit theorem. The CS lens is not a new continent; it is a map of the roads the other continents already have.
