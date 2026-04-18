# Mathematical Mental Models for Knowledge Transfer

**Context:** Sutra-Dyad (188M byte-level). Classical KD (match teacher softmax) has plateaued. We need alternative *mental models* — not new loss functions bolted onto the same picture, but new pictures.

## TL;DR — What is classical KD missing, mathematically?

Classical KD treats knowledge as **pointwise probability vectors** and transfer as **KL minimization in output space**. This is a zero-dimensional picture: every input is a point, every output is a point, and we match points. What it misses:

- **Structure between points** (category theory, equivariance) — KD doesn't care whether "king→queen" and "man→woman" are the same kind of arrow. It only cares that the marginal distributions match.
- **Locality and gluing** (sheaves) — teachers know different things in different regions; KD averages them into a blurred consensus instead of patching them together.
- **Multi-scale invariants** (fractals, persistent homology) — teacher features are continuous over scales; KD collapses this to one level.
- **Geometry of the distribution space** (optimal transport) — KL is myopic near the teacher; Wasserstein gives a gradient even when supports barely overlap.
- **Connections and parallel transport** (fiber bundles) — teacher and student live on different tokenizations/spaces; a principled way to *move vectors between them* is missing.

Five lenses follow. Each reframes "what is knowledge" and suggests a different minimal experiment. They are not mutually exclusive — the closing section maps the cross-pollination.

---

## Model 1 — Category Theory / Functoriality

**The model in one sentence.** A teacher is a *functor* `T: Input → Output` that preserves the structure (composition, identities, morphisms) of the input category, and distillation is not matching objects but asking the student to be a **natural transformation** between the teacher functor and the student functor, i.e. making the diagram commute.

```
          T
  Input -----> Output_T
    |             |
  id|             | α   (natural transformation: student ≈ teacher up to α)
    v             v
  Input -----> Output_S
          S
```

**What we'd BORROW.** Classical KD enforces `S(x) ≈ T(x)` pointwise. The categorical view demands `S(f(x)) ≈ f'(S(x))` — student must preserve **relationships** the teacher does. Concretely for Sutra:

- For a pair of prompts `(x, x')` where the teacher induces some transformation (e.g., `x' = paraphrase(x)`, or `x' = x[:n-1]` and `T(x') = prefix_of T(x)`), the student must preserve the same relation.
- The loss becomes `||T(x) − T(x')||  vs  ||S(x) − S(x')||` — a *relational* distillation ("relational KD", Park et al. 2019, but with an explicit functorial framing).
- More powerfully: enforce functoriality over **composable morphisms** — e.g., if `x₁ →byte_append→ x₂ →byte_append→ x₃` in the input category, the student's hidden states must compose the same way the teacher's do.

**Toy thought experiment.** Take 3 teacher models with different tokenizers (the Ekalavya setup). Each defines a functor `T_k`. The student `S` factors through all three iff there exist natural transformations `α_k: T_k ⇒ S ∘ U_k` where `U_k` is tokenization. Training: sample *diagrams* (pairs/triples of related inputs), score how badly the diagram fails to commute in student-space, backprop. **Expect:** student embeddings acquire algebraic structure (translation invariance, composition) that pointwise KD cannot produce; downstream analogy/relation tasks should jump.

**What this reframes.** KD is not "copy the answers," it is "copy the *inference rules*." The teacher's knowledge lives in its **morphisms**, not its objects. Under this view, soft-label KD captures almost nothing — it samples the functor at isolated objects and throws away every arrow.

**Testable prediction.** Relational KD over matched-prefix pairs (an obvious morphism in byte-space: `x ↪ x ++ b`) will outperform token-matched logit KD on a fixed compute budget, and the gap will *grow* with sequence length because longer sequences compose more morphisms.

---

## Model 2 — Sheaf Theory

**The model in one sentence.** Each teacher is a **local section** — knowledge defined only over the open set of inputs it's competent on — and the student is the global section we obtain by **gluing** them on the overlaps where they agree.

**What we'd BORROW.** A *presheaf* of knowledge: to every region `U` of input space (a topic, a sequence-length regime, a language, a domain), assign a "restriction" `T|_U` of the teacher. A presheaf becomes a **sheaf** iff compatible local data glue uniquely to global data. In ML terms:

- **Restrictions.** Teachers `T_math`, `T_code`, `T_web` each dominate a subset of input space. Their "restrictions" are their outputs on those regions. Classical KD averages them into a mushy consensus. Sheaf KD insists they only speak where they are locally authoritative.
- **The Čech gluing condition.** On overlapping regions (e.g., "math-in-code"), the teachers must agree up to tolerance, and *their disagreement is itself data* — it tells the student which regions are genuinely multi-valued and require a richer representation.
- **Cohomology as distillation obstruction.** `H¹` of the cover measures how badly local teachers fail to glue. Nonzero cohomology is a *measure of irreducible disagreement* that no single student can satisfy. This is a formal notion of "the teachers genuinely conflict."

**Toy thought experiment.** Partition input distribution by a gating classifier into regions `{U_k}`. For each `U_k`, pick the most competent teacher (or weighted mixture). Train student with **per-region KD loss** plus an **overlap consistency term**: on inputs in `U_i ∩ U_j`, penalize `|T_i(x) − T_j(x)|` as "teacher-space conflict" and only distill the *agreed* component. **Expect:** faster training on clean regions (no averaging noise) and explicit cohomological signal on genuinely conflicted regions — which become the *frontier* the student explores on its own.

**What this reframes.** There is no single teacher distribution to match. Knowledge is **locally defined and locally true**. The question "what does the teacher say here?" has a different answer in every chart, and *where the teachers disagree is exactly where the student learns something none of them individually had*.

**Testable prediction.** A sheaf-gated KD loss (per-region teacher selection + overlap-consistency term) will beat uniform-mix multi-teacher KD by a decisive margin (≥ 3-5 pp BPB-equivalent on held-out genre mix), and the gap will be **largest on inputs that mix regions** (code with math, dialog with facts).

---

## Model 3 — Fractal Geometry / Self-Similarity

**The model in one sentence.** Teacher knowledge is **self-similar across scales** — the same mathematical structure appears at the character, word, phrase, and paragraph level — and distillation should transfer a *scale-invariant law*, not the raw outputs at one scale.

**What we'd BORROW.** The moonshot-fractal-embeddings result (adjacent project): embeddings with nested multi-scale structure capture hierarchical semantics better than flat ones. Translate into KD:

- **Multi-scale distillation targets.** For each input, the teacher produces not one distribution but a *family* indexed by context window / pooling scale: `T_k(x) = teacher's next-byte dist given last 2^k bytes`. Ask the student to match the **scale-relationship** — how the distribution sharpens as context grows — rather than matching any single `T_k`.
- **Scaling exponents as invariants.** For each position, the teacher induces an entropy-vs-context curve `H(2^k)`. This curve typically has a fractal character (power-law tails). Distill the **exponent** — the student must produce the same asymptotic scaling of uncertainty with context — a hugely compressed target compared to full distributions.
- **Self-similar architecture.** If knowledge is self-similar, the student should have a self-similar architecture: the same block applied at character, patch, and phrase scales (same law, different scale) — Sutra-Dyad's byte/patch duality is already a step toward this.

**Toy thought experiment.** For each training sequence, cut it at 10 random positions, pool into 2^k-byte windows for k∈{4..10}, and have the teacher emit entropy curves. Student is trained to match the *entropy curve shape* (low-parameter target: a few power-law coefficients) instead of per-token logits. **Expect:** dramatically lower KD target dimensionality, faster convergence, and student that generalizes across sequence lengths because it learned the law, not the instances.

**What this reframes.** Knowledge is not a function; it is a **renormalization-group flow**. The teacher's "true" signal is invariant under scale transformation, and matching a single scale is like photographing a coastline at one zoom — most of the information is in the self-similarity itself.

**Testable prediction.** A fractal-exponent KD loss (match teacher's entropy-vs-context power-law exponent at each position, via 3-5 sampled scales) will produce better long-context extrapolation than per-token logit KD at equal training compute — measured by BPB on 2× the training context length.

---

## Model 4 — Optimal Transport / Wasserstein Geometry

**The model in one sentence.** Teacher and student distributions are **points in a metric space of measures**, and distillation is a **gradient flow in Wasserstein geometry** — which, unlike KL, remains informative even when student and teacher barely overlap.

**What we'd BORROW.** KL divergence is brittle: when `S(x)` puts zero mass where `T(x)` has support, the gradient is infinite/undefined; when supports are disjoint, KL is the same (∞) regardless of *how far apart* the supports are. Wasserstein geometry fixes this:

- **Earth-Mover distance over the vocabulary.** Byte space has a natural metric (e.g., character similarity, semantic neighborhood from the teacher's own embedding). W_2 between teacher and student is a *geometric* distance that knows "predicting 'b' when truth is 'c' is closer than predicting 'z'".
- **Gradient flow picture.** Training becomes a flow in the space of probability distributions along the Wasserstein gradient of the distillation cost — a direct instance of the JKO scheme. This replaces the ill-conditioned KL geometry with one that has bounded curvature.
- **Multi-teacher as barycenter.** The Wasserstein **barycenter** of teacher distributions is the student target. Unlike KL-average (which produces a smeared mixture), the W-barycenter preserves *modes* — if teachers confidently assert different-but-plausible answers, the barycenter represents them as distinct modes, not a mush.

**Toy thought experiment.** Define a cost matrix `c(i,j)` over byte vocabulary using teacher embeddings. Replace KD loss with entropic OT (Sinkhorn) distance between `T(x)` and `S(x)`. For multi-teacher, compute Sinkhorn barycenter on-the-fly. **Expect:** (1) student learns a more peaked-yet-calibrated distribution; (2) in the multi-teacher Ekalavya setting, modes from each teacher are preserved rather than blurred — directly addressing the "0.003 BPT noise" plateau from averaging.

**What this reframes.** Distillation is *not* an information-theoretic problem, it is a **transportation problem**. The student doesn't minimize surprise under the teacher; it moves mass along the cheapest routes in a meaningful metric space. Supports don't need to overlap for the gradient to be informative.

**Testable prediction.** Sinkhorn-KD with a byte-embedding cost matrix will beat KL-KD by ≥ 1 BPB-point at the *early* stage of training (when student and teacher supports barely overlap), and this advantage will diminish late in training where KL and W agree — confirming the "ill-conditioned early geometry" hypothesis.

---

## Model 5 — Differential Geometry / Fiber Bundles (Parallel Transport)

**The model in one sentence.** Teacher and student live on *different fibers over the same base space*, and distillation is a **connection** — a rule for parallel-transporting vectors between fibers that respects the base-space geometry.

**What we'd BORROW.** The Ekalavya problem is: three teachers with three tokenizers produce logits in three incompatible spaces. The student has yet another space (bytes). There is no canonical `identity` map. Fiber-bundle geometry gives the right object:

- **Base space = inputs** (byte sequences). Shared across all models.
- **Fibers = per-model representation spaces** (logits, hidden states). Each teacher has its own fiber over each input point.
- **Connection = a learned linear map (per-input) that transports a vector from one fiber to another, constrained to be smooth over the base.** This is a cross-tokenizer alignment learned as a field, not a global matrix.
- **Curvature = obstruction to globally consistent transport.** Nonzero curvature means "you cannot translate teacher A's knowledge into teacher B's space without loss." Measuring it tells you where multi-teacher KD is lossless and where it is not.

**Toy thought experiment.** Instead of distilling teacher logits directly, train a per-input "gauge transformation" `g(x): Fiber_T → Fiber_S` as a small network (1-2 layers, input-conditioned). The distillation loss is `||S(x) − g(x) · T(x)||`, but with a smoothness prior `||∇_x g||` so the gauge varies *smoothly* over input space. **Expect:** with only a tiny extra parameter count, the student absorbs multi-teacher signal *without being forced to live in any one teacher's tokenization*. Byte-level Sutra-Dyad is a natural base space for this.

**What this reframes.** The assumption "teacher and student must share a vocabulary/tokenization" is wrong. They share the *base space* (text) and nothing else; distillation is *fundamentally* a gauge theory. Classical KD implicitly assumes trivial bundles and flat connections — a special case that fails the moment tokenizers differ.

**Testable prediction.** Adding a learned input-conditioned 2-layer gauge network between each teacher and the student will enable cross-tokenizer KD with ≥ 2x better sample efficiency than teacher-probability-interpolation baselines, and the learned gauge's *curvature* (measured as `||g(x) − g(x')||` for similar `x,x'`) will correlate with where multi-teacher KD stalls.

---

## Cross-Pollination

- **Category ↔ Sheaf.** A sheaf *is* a functor (from the opposite category of opens to sets/abelian groups). Sheaf KD is categorical KD with a locality restriction; combining them = enforce functoriality *only on local patches* and glue.
- **Sheaf ↔ Fiber Bundle.** Sheaves of sections are the algebraic avatar of bundles. A multi-teacher gauge (Model 5) **is** a sheaf of teacher-knowledge with non-trivial transition functions (Model 2); curvature ↔ cohomology as obstruction.
- **Fractal ↔ Category.** Self-similarity is a categorical statement: the same functor acts at every scale. Fractal KD is functorial KD where the morphisms are *scale transformations*.
- **Optimal Transport ↔ Sheaf.** Wasserstein barycenters of teacher distributions give the *right* way to glue local teacher data into a global student — the transportation cost quantifies the Čech obstruction quantitatively.
- **Fiber Bundle ↔ Optimal Transport.** Parallel transport between fibers is (morally) an optimal-transport map; the Levi-Civita connection is the OT map under a quadratic cost. So cross-tokenizer distillation = Wasserstein transport between teacher and student fibers over each base point.

---

## If you pick ONE and run with it

**For the Ekalavya plateau specifically:** Model 5 (fiber bundles / gauge) is the most directly actionable — it formalizes the cross-tokenizer problem that multi-teacher KD actually faces and is implementable as a small per-input network + smoothness prior, testable in a single training run at 188M scale.

**For the "knowledge is relational" hypothesis:** Model 1 (functoriality / relational KD) requires only a paired-input data loader and is the cheapest first test.

**For the plateau that says "averaging teachers blurs them":** Model 4 (Wasserstein barycenter) directly replaces KL averaging and is a drop-in loss swap with Sinkhorn, evaluable in hours.

The others — sheaves (Model 2) and fractals (Model 3) — are higher-structure but slower-to-implement. They are where the biggest conceptual wins live if the small wins of 1/4/5 confirm the direction.
