# WORK LOOP - Batch 1: Geometry Transplantation

Grounding completed: read `research/DEEP_RETHINK.md` fully, `research/VISION.md`, `research/STATUS.md`, `../CLAUDE.md`, and inspected the core code surfaces under `code/` (`s0_architecture.py`, `s0_configs.py`, `semantic_codec.py`, `codec_phase2_model.py`, `codec_phase1_train.py`, `sutra_energy_probe.py`).

Hard evidence inherited from the repo: byte-marginal KL improved BPB but not task accuracy; hidden cosine alignment failed against shuffled controls; multi-loss operational geometry failed its gates; real S0 frozen energy probing did not reveal a useful HellaSwag readout gap; Wide7 improved BPB/speed but not reasoning; the semantic codec learns byte-to-token-embedding retrieval but has not yet proven semantic addressability.

This batch asks whether trained weight geometry itself can be extracted and transplanted without gradient-based training or iterative loss optimization. Allowed operations: decomposition, projection, conjugation, spectral truncation, randomized sketches, finite probes, closed-form linear algebra, deterministic surgery. Disallowed as success evidence: fine-tuning, learned adapters, gradient KD, or supervised repair after transplant.

## Iteration 1: Raw SVD Transplant Is Not Geometry

### Register
- Exploring: top singular/eigen components of trained weights copied into a smaller model.
- Why: the tempting story is that intelligence lives in high-energy spectral modes.
- Confirm if: `CONFIRM_SVD_RAW` only if no-training top-k transplant beats random/top-k-energy controls by >=5pp and survives gauge-equivalent teachers.
- Kill if: `KILL_SVD_RAW` if a function-equivalent hidden-basis rotation changes the transplant result.
- Void if: `VOID_SVD_RAW` if basis identity leaks through same initialization or same-family checkpoints.

### Design
Minimum viable experiment: train/load a tiny binding-task teacher; create a gauge-equivalent teacher by `W1' = A W1`, `W2' = W2 A^-1`; transplant top-k SVD components from both teachers into the same smaller student; evaluate immediate function agreement and MCQ accuracy with no tuning.

Design-gate: pass. This is the cheapest way to test whether raw singular vectors are invariants or just coordinates. Alternative interpretation: success on an unrotated teacher may only mean teacher and student share a lucky basis.

### Derivation
For a layer `W`, SVD gives `W = U Sigma V^T` and `W_k = U_k Sigma_k V_k^T`. But internal hidden states have gauge freedom. If `h' = A h`, adjacent weights can change while the function remains identical: `W1' = A W1`, `W2' = W2 A^-1`, so `W2' W1' = W2 W1`. The singular vectors of `W1'` and `W2'` generally differ from those of `W1` and `W2`. Raw singular vectors of internal layers are coordinates, not architecture-independent intelligence.

### Dry-Run / Sanity Check
SVD of 576, 1024, or 1152 square-ish matrices is trivial on one RTX 5090; top-128 randomized SVD per layer is feasible. Compute is not the risk. False invariance is the risk.

### Evidence Gate
Prove it by surviving non-orthogonal gauge rotations and beating matched-spectrum random matrices. Kill it if the same teacher function under a different basis yields different transplanted behavior. Current verdict: killed as a primary transplant idea; keep raw SVD as a negative control.

### Attack
Hostile expert objection: This is low-rank compression with moonshot branding. It copies coordinates, not intelligence.

### What Survived
The negative lesson: any serious weight-transplant framework must be gauge-aware.

### NARRATIVE GATE
- Gossip-magazine headline: The obvious way to photocopy a brain copies the filing cabinet, not the thoughts.
- "Isn't that obvious?": To experts after the gauge argument, yes.
- "So what?": It prevents another attractive but fake spectral direction.
- Honest assessment: boring but necessary.

### Next Iteration
Construct a gauge-fixed transplant instead of raw spectral copying.

## Iteration 2: Gauge-Fixed Procrustes Transplant

### Register
- Exploring: estimate a teacher-to-student coordinate map from paired activations, then conjugate teacher operators into student space.
- Why: if coordinates are arbitrary, first align gauges, then transplant.
- Confirm if: `CONFIRM_PROCRUSTES` only if closed-form gauge mapping plus direct block surgery improves a frozen student by >=3pp and beats shuffled-pair controls by >=4pp.
- Kill if: `KILL_PROCRUSTES` if mapped transplants match shuffled-pair transplants or fail held-out prompt families.
- Void if: `VOID_PROCRUSTES` if any gradient update is used after surgery.

### Design
Collect paired activations `(H_T, H_S)` on shared public inputs, whiten both activation spaces, compute a CCA/Procrustes map `P`, then insert teacher operators as `W_S = P_out W_T P_in^+`. Replace one student sublayer at a time and evaluate immediately.

Design-gate: pass only with shuffled pairs and deliberate teacher gauge rotations. Alternative interpretation: CCA may align activation clouds but not computation.

### Derivation
Let centered activations be `H_T in R^{d_T x n}` and `H_S in R^{d_S x n}`. Compute:

```text
C_T = H_T H_T^T / n + eps I
C_S = H_S H_S^T / n + eps I
C_ST = H_S H_T^T / n
M = C_S^-1/2 C_ST C_T^-1/2
M = U Rho V^T
P = C_S^1/2 U_k V_k^T C_T^-1/2
```

A teacher local operator then moves by projection/conjugation: `J_S = P_out J_T P_in^+`. This is deterministic surgery, not learned alignment.

### Dry-Run / Sanity Check
For Qwen-like `d_T=1024` and Wide7 `d_S=1152`, `P` is `1152 x 1024`; for S0 it is `576 x 1024`. Activation caches at 8192 anchors are tens of MB per layer. Token/byte mismatch means first pass should use only clean token-end / patch-end anchors.

### Evidence Gate
Prove it if re-estimated `P` survives gauge rotations, shuffled pairs fail, and direct surgery improves behavior before training. Kill it if high CCA correlations produce no behavioral gain or random orthogonal maps perform similarly. Current verdict: promising infrastructure, not sufficient as a claim.

### Attack
Hostile expert objection: This is hidden-state alignment with fancier algebra; cloud alignment does not identify algorithms.

### What Survived
The gauge map survives only as infrastructure. The real claim must come from executable transplanted operators improving behavior.

### NARRATIVE GATE
- Gossip-magazine headline: Before copying a mind, line up the languages its neurons speak.
- "Isn't that obvious?": The need is obvious after failure; executable conjugation is not.
- "So what?": This is the first route to no-gradient block transplant.
- Honest assessment: promising but fragile.

### Next Iteration
Use function-level operator fingerprints, not just activation clouds.
## Iteration 3: Jacobian Fingerprint Transplant

### Register
- Exploring: extract teacher block action on meaningful perturbation directions and instantiate the matching low-rank student operator.
- Why: intelligence may live in response geometry: what changes when meaning changes.
- Confirm if: `CONFIRM_JACOBIAN` only if a closed-form sketched operator improves held-out behavior by >=3pp and beats random semantic-probe controls.
- Kill if: `KILL_JACOBIAN` if local operator matching does not improve behavior over random low-rank operators.
- Void if: `VOID_JACOBIAN` if the method trains student weights.

### Design
Choose meaningful probe directions: paraphrase deltas, counterfactual deltas, and candidate-choice deltas. Estimate teacher response `y_i = F_T(h_T + eps v_i) - F_T(h_T)`. Map directions and responses into student coordinates with the gauge map. Build the minimum-norm operator `J* = Y V^T (V V^T)^+`. Insert it as a residual low-rank operator `h' = h + F_S(h) + beta J*(h - mu)`.

Design-gate: pass for toy and microdomain tasks; do not claim whole-model knowledge. Alternative interpretation: the Jacobian may encode local smoothness rather than capability.

### Derivation
Let `V` contain `m` probe vectors and `Y` contain teacher responses. The closed-form operator that matches the probes is `J* V = Y`, hence `J* = Y V^+`. If `m << d`, store it as low-rank factors rather than a full matrix. Set `beta` by matching median residual norms on calibration data, not by optimizing a loss.

### Dry-Run / Sanity Check
For `d=1152`, `m=512`, `V` and `Y` are each about 0.6M floats. Teacher forward passes dominate but remain cheap compared with training. The hard part is constructing perturbations that are semantically causal rather than lexical noise.

### Evidence Gate
Prove it if meaningful probes outperform random probes, held-out transformations improve, and hidden norms/BPB stay stable. Kill it if gains occur only near cached examples or synthetic gains fail on real benchmark slices. Current verdict: good circuit-level transplant candidate, not full intelligence transplant.

### Attack
Hostile expert objection: A first-order Taylor sketch is not a mind. You are copying local reflexes.

### What Survived
The method survives as local skill transfer. It should be tested on controlled circuits before world knowledge.

### NARRATIVE GATE
- Gossip-magazine headline: We copied how the model reacts when meaning changes.
- "Isn't that obvious?": No; most distillation copies outputs.
- "So what?": It may transfer specific reasoning reflexes without training.
- Honest assessment: promising but limited.

### Next Iteration
Target where factual/world knowledge is likely stored: MLP associative memory.

## Iteration 4: MLP Memory-Slot Extraction

### Register
- Exploring: extract MLP key-value memory slots from a teacher and transplant the highest causal-effect slots.
- Why: dense models may store world knowledge in MLP associative memories; this attacks inherited knowledge directly.
- Confirm if: `CONFIRM_MLP_SLOTS` only if no-training transplanted slots improve factual/commonsense candidate margins by >=3pp and beat frequency-matched and shuffled key/value controls.
- Kill if: `KILL_MLP_SLOTS` if slots only encode token frequency or morphology.
- Void if: `VOID_MLP_SLOTS` if benchmark labels are used to select slots.

### Design
Decompose teacher SwiGLU MLPs into slot-like components; estimate each slot's causal effect on teacher candidate margins; map top keys/values into Sutra space via Procrustes or codec chart; add a deterministic external slot memory:

```text
query = A h_S
score_i = query dot k_i
retrieved = sum_i softmax(score_i) v_i
h_S' = h_S + B retrieved
```

No learned `A` or `B`; they come from gauge maps. Design-gate: pass only with shuffled key/value and frequency controls. Alternative interpretation: MLP neurons are not facts; knowledge may be distributed.

### Derivation
For SwiGLU:

```text
MLP(h) = W_down (silu(W_gate h) * W_up h)
       = sum_i silu(g_i^T h) (u_i^T h) v_i
```

Each intermediate dimension is a candidate slot `(g_i, u_i, v_i)`. Approximate causal effect for a candidate pair `(y+, y-)` by:

```text
effect_i ~= a_i(h_x) * (e_y+ - e_y-)^T v_i
```

Keep slots whose effect is large across many calibration contexts and not explained by token frequency.

### Dry-Run / Sanity Check
A table of 8192 teacher slots with 1024-d keys and values is about 32MB in fp16. Compute is not the blocker. The blocker is whether Sutra hidden states can query teacher slots without training.

### Evidence Gate
Prove it if the correct slot table improves held-out margins while shuffled key/value pairing and frequency-matched slots fail. Kill it if student queries retrieve wrong slots or improvements are lexical only. Current verdict: one of the strongest concrete paths, but dependent on gauge quality.

### Attack
Hostile expert objection: This is neuron folklore. Distributed representations will not become portable flashcards.

### What Survived
Do not call slots facts. Call them causal associative components. The method survives only if causal slot controls pass.

### NARRATIVE GATE
- Gossip-magazine headline: Steal the big model's flashcards.
- "Isn't that obvious?": The story is obvious; making the flashcards work in another model is not.
- "So what?": This directly attacks the knowledge-transfer bottleneck.
- Honest assessment: high value, high control burden.

### Next Iteration
Look for algebraic factorization that can resize trained structure, not just attach slots.

## Iteration 5: Kronecker/Tensor Factor Transplant

### Register
- Exploring: factor trained matrices into separable Kronecker/tensor components, resize factors, and reconstruct student weights.
- Why: if learned computation has separable head/feature structure, tensor factors may preserve relations better than rank truncation.
- Confirm if: `CONFIRM_KRON` only if tensor-factor transplant beats equal-parameter SVD and random-structured controls by >=2pp.
- Kill if: `KILL_KRON` if low reconstruction error does not yield behavioral benefit.
- Void if: `VOID_KRON` if factors are tuned by gradient descent.

### Design
Pick attention or MLP matrices with natural factorizations; rearrange `W` so `A kron B` becomes rank-1 in rearranged space; SVD the rearranged matrix; resize factors into student dimensions; compare to SVD, pruning, random Kronecker, and shuffled factors.

Design-gate: pass as a supporting method; not primary until behavioral superiority is shown. Alternative interpretation: pretty algebra may reconstruct weights without preserving computation.

### Derivation
For `W in R^{ab x cd}`:

```text
W ~= sum_r A_r kron B_r
R(W) ~= sum_r sigma_r u_r v_r^T
A_r = reshape(u_r)
B_r = reshape(v_r)
W_S = sum_r resize(A_r) kron resize(B_r)
```

Sutra dimensions make head-aware factorization natural: `576 = 9 x 64`, `1024 = 16 x 64`, `1152 = 18 x 64`.

### Dry-Run / Sanity Check
Randomized SVD of 1152x1152 matrices is feasible. The main failure risk is arbitrary reshape; factor dimensions must correspond to real head/block structure.

### Evidence Gate
Prove it if resized Kronecker factors outperform SVD at matched parameter count and head-shuffled factors fail. Kill it if behavior tracks reconstruction error only or gains appear only in same-architecture settings. Current verdict: plausible compression/transplant component, not a standalone moonshot.

### Attack
Hostile expert objection: You are imposing structure after the fact. A Kronecker fit is not cognition.

### What Survived
Only behavior-gated tensor resizing survives. Reconstruction metrics are insufficient.

### NARRATIVE GATE
- Gossip-magazine headline: Shrink the model by keeping the multiplication table of its thoughts.
- "Isn't that obvious?": Not if it preserves relations across size.
- "So what?": It may make chain-init compression more principled.
- Honest assessment: mathematically clean, medium narrative.

### Next Iteration
Try deterministic coarsening of trained features.
## Iteration 6: Renormalization Coarsening of Features

### Register
- Exploring: merge equivalent features/neurons/heads by response similarity to coarsen a trained model into a smaller one.
- Why: a smaller model may be a renormalized version of a larger one: same large-scale behavior, fewer microscopic degrees of freedom.
- Confirm if: `CONFIRM_RG` only if deterministic coarsening preserves margins better than magnitude pruning at equal size and beats random merge controls.
- Kill if: `KILL_RG` if response-equivalent merges do not outperform pruning.
- Void if: `VOID_RG` if fine-tuning is used to repair the model.

### Design
Compute response fingerprints `r_i = [activation_i(x_1), ..., activation_i(x_n)]`; cluster highly similar features with compatible output values; merge incoming keys/gates by weighted averaging and outgoing values by summing; evaluate no-training behavior after each compression step.

Design-gate: pass for same-family compression; cross-architecture use is secondary. Alternative interpretation: response similarity may be local to calibration data.

### Derivation
For an MLP:

```text
F(h) = sum_i a_i(h) v_i
```

If `a_i(h) ~= c a_j(h)`, then:

```text
a_i(h)v_i + a_j(h)v_j ~= a_j(h)(c v_i + v_j)
```

So two features can be replaced by one when their response functions are collinear and values are compatible.

### Dry-Run / Sanity Check
Layer-at-a-time activation collection over 1024-8192 sequences is feasible. Direct Qwen3-0.6B to 121M coarsening is too aggressive first. Start with toy and smaller same-family checkpoints.

### Evidence Gate
Prove it if merge error predicts behavior loss, coarsened models beat pruning/random merges, and rare-slice evaluation does not collapse. Kill it if similar features are not causally interchangeable or small merge errors amplify through layernorm/residual dynamics. Current verdict: useful for same-family chain compression, uncertain for wild transplant.

### Attack
Hostile expert objection: Renormalization is a metaphor. Neural networks do not guarantee clean scale separation.

### What Survived
Drop the metaphor. Keep deterministic response-equivalence coarsening with causal controls.

### NARRATIVE GATE
- Gossip-magazine headline: Boil the big brain down without boiling away the idea.
- "Isn't that obvious?": Compression is obvious; no-training causal coarsening is not.
- "So what?": It could make pretrained intelligence cheap without another training run.
- Honest assessment: feasible, less revolutionary unless it crosses architectures.

### Next Iteration
Transfer algorithms/circuits rather than weights.

## Iteration 7: Circuit Genome Transplant

### Register
- Exploring: extract a teacher circuit as a typed executable "genome" and instantiate it in Sutra without training.
- Why: if coordinates fail, transfer the learned trick: binding, comparison, negation, temporal order, affordance.
- Confirm if: `CONFIRM_CIRCUIT_GENOME` only if extracted circuit genomes solve held-out variants at >=80% of teacher accuracy and beat wrong-genome controls.
- Kill if: `KILL_CIRCUIT_GENOME` if the algorithm is manually supplied rather than extracted.
- Void if: `VOID_CIRCUIT_GENOME` if the genome is just generated training data.

### Design
Start with trained toy teachers for binding and comparison. Use causal interventions to infer roles: entity encoder, attribute binder, query matcher, answer selector. Express the inferred circuit as typed operations:

```text
types: entity, attribute, value
ops: bind(entity, attribute, value), query(entity, attribute), select(value)
```

Instantiate with fixed key-value memory and bilinear energy scoring. Evaluate held-out compositions without gradient updates.

Design-gate: pass only if extraction is automated and wrong-genome controls fail. Alternative interpretation: this may reinvent symbolic AI with parsing as the hidden hard part.

### Derivation
A binding circuit can be represented as:

```text
B[e,a] = v
query(e,a) -> argmax_v score(B[e,a], v)
```

Neural instantiation:

```text
k(e,a) = phi(e) kron psi(a)
memory += k(e,a) v(value)^T
score = <memory^T k(e,a), v(candidate)>
```

The transplant object is not weights. It is an inferred transition law plus an analytic instantiation.

### Dry-Run / Sanity Check
Toy binding is too easy. The ladder must include `toy binding -> synthetic physical affordances -> HellaSwag microdomain`. Compute is trivial. Automated extraction is the hard part.

### Evidence Gate
Prove it if extracted genomes predict teacher interventions not used for extraction, instantiated modules work on held-out variants, and wrong genomes fail. Kill it if human-coded algorithms do the real work or natural language parsing becomes the unsolved step. Current verdict: high narrative, high risk, best as a moonshot branch.

### Attack
Hostile expert objection: You did not extract intelligence; you wrote a program and called it a genome.

### What Survived
Only an adversarial extraction benchmark survives. The circuit must be inferred from teacher behavior/interventions.

### NARRATIVE GATE
- Gossip-magazine headline: We don't copy the brain; we copy the trick it learned.
- "Isn't that obvious?": The story is simple; automatic extraction is not.
- "So what?": A library of extracted tricks could make small models useful without storing all knowledge densely.
- Honest assessment: story-strong, technically risky.

### Next Iteration
Test the wilder biological analogy: can a compact rule generate trained weights?

## Iteration 8: Weight Genome by Spectral Development

### Register
- Exploring: infer a compact program that generates trained weights layer by layer, then resize the program to emit a smaller model.
- Why: biology transfers developmental rules, not adult synapses.
- Confirm if: `CONFIRM_WEIGHT_GENOME` only if generated weights beat random initialization before training and held-out layers are predictable from the inferred program.
- Kill if: `KILL_WEIGHT_GENOME` if generated weights are no better than random after matching spectra/norms.
- Void if: `VOID_WEIGHT_GENOME` if a neural generator is trained by backprop.

### Design
Extract deterministic descriptors per layer: spectra, head-block covariance, MLP slot moments, layer index. Fit a closed-form recurrence:

```text
z_l ~= z_0 + sum_m a_m phi_m(l/L)
```

Generate student weights from resized descriptors and orthogonal frames. Evaluate before any training.

Design-gate: only one harsh smoke test is justified. Alternative interpretation: descriptors may encode training thermodynamics, not knowledge.

### Derivation
For layer descriptor `z_l`:

```text
z_l = [log singular values, block covariance, slot moments]
A = Z Phi^T (Phi Phi^T)^+
W_l' = U_l diag(sigma_l') V_l^T
```

This yields trained-looking weights by construction. The issue is whether they compute anything.

### Dry-Run / Sanity Check
Generated weights may have stable norms but random function. First success cannot be HellaSwag. It must be: BPB at initialization at least 10% better than random, and continuation choice above random by >=2pp.

### Evidence Gate
Prove it if held-out layers/heads are predicted by the program, generated models have immediate functional advantage, and layer-order shuffling kills the advantage. Kill it if only activation stability improves while behavior stays random. Current verdict: highest moonshot story, lowest feasibility.

### Attack
Hostile expert objection: This is numerology. You can generate a healthy-looking corpse.

### What Survived
Only the harsh above-random smoke test. Kill quickly if it fails.

### NARRATIVE GATE
- Gossip-magazine headline: Find the DNA of a neural network and grow a smaller one.
- "Isn't that obvious?": No; it is likely false.
- "So what?": If true, it would be intelligence by construction.
- Honest assessment: breakthrough narrative, low probability.

### Next Iteration
Define the adversarial benchmark that decides which transplant ideas are real.
## Iteration 9: The Transplant Gauntlet

### Register
- Exploring: a benchmark suite every no-gradient transplant method must pass before production Sutra surgery.
- Why: the project has repeatedly produced plausible signals that controls later killed.
- Confirm if: `CONFIRM_GAUNTLET` only if it validates known-good transplants and kills fake controls across seeds.
- Kill if: `KILL_GAUNTLET` if random/fake methods pass or exact known-good cases fail.
- Void if: `VOID_GAUNTLET` if methods are allowed to train or repair after transplant.

### Design
Tier 1: known-gauge linear teacher. Exact transplant is mathematically known; raw SVD should fail under non-orthogonal gauge; gauge-aware transplant should pass.

Tier 2: nonlinear toy circuit. Binding/comparison teacher; test Procrustes, Jacobian, MLP slots, wrong controls.

Tier 3: same-family real compression. Test renormalization and Kronecker resizing on a smaller pretrained family if available.

Tier 4: Sutra bridge. Test only methods that cleared earlier tiers on S0/Wide7/codec surfaces.

Design-gate: pass. This is required before any claim of direct intelligence copying. Alternative interpretation: toy gauntlet success may not imply real model success; toy failure is still decisive.

### Derivation
A transplant method is:

```text
theta_S' = T(theta_T, theta_S, D_cal)
```

No gradients. Score by specificity:

```text
specificity =
  [score(theta_S') - score(theta_S)]
  - [score(control_transplant) - score(theta_S)]
```

Controls: shuffled calibration pairs, random gauge, same-spectra random vectors, shuffled key/value slots, wrong circuit genome, frequency-matched slots, layer-order shuffle.

### Dry-Run / Sanity Check
Can reuse existing toy infrastructure: `toy_rep_align_test.py`, `toy_opgeom_*`, and `toy_readout_r1.py`. Runtime is small because this is forward-pass and linear-algebra heavy, not training-heavy.

### Evidence Gate
Prove it if exact transplants pass, raw SVD dies under gauge rotation, and fake controls fail. Kill it if the gauntlet cannot separate true and fake mechanisms. Current verdict: build before trusting any transplant result.

### Attack
Hostile expert objection: You are building a benchmark instead of making the breakthrough.

### What Survived
Given the repo's false-positive history, the benchmark is the adversary required by the methodology. It is not the moonshot, but it prevents fake moonshots.

### NARRATIVE GATE
- Gossip-magazine headline: A lie detector for claims that you can copy intelligence.
- "Isn't that obvious?": It should be, but most distillation work lacks these controls.
- "So what?": It saves the project from another attractive dead end.
- Honest assessment: necessary infrastructure.

### Next Iteration
Rank the approaches and choose the first build.

## Iteration 10: Buildable Geometry Transplant Roadmap

### Register
- Exploring: final ranked roadmap and first build recommendation.
- Why: the batch must end with concrete, buildable work.
- Confirm if: `CONFIRM_ROADMAP` only if the recommendation has exact artifacts, gates, controls, and a path from toy proof to real Sutra.
- Kill if: `KILL_ROADMAP` if the first build cannot distinguish transplant from compression, calibration, or data augmentation.
- Void if: `VOID_ROADMAP` if it requires more than one RTX 5090.

### Design
Surviving approach families: gauge-fixed operator transplant; MLP memory-slot transplant; renormalization coarsening; circuit genome transplant; Kronecker/tensor factor support; weight genome smoke test.

Design-gate: build the gauntlet first, then run gauge-fixed operator transplant and MLP slot transplant through it. Alternative interpretation: this may look cautious. It is not safe; it is adversarial. The risk is in the mechanisms, not in skipping controls.

### Derivation
The surviving mathematical template is:

```text
theta_S' = Instantiate(Extract(theta_T), Gauge(theta_T, theta_S, D_cal), A_S)
```

Where `Extract` returns operators, slots, circuits, or factors; `Gauge` maps teacher coordinates to student coordinates; and `Instantiate` creates executable student weights/modules by closed-form surgery.

Best first concrete method:

```text
P_l = CCA(H_T^l, H_S^l)
J_T^l = Sketch(F_T^l, semantic probe directions)
J_S^l = P_{l+1} J_T^l P_l^+
h_S' = h_S + beta J_S^l(h_S - mu_S)
```

Best second concrete method:

```text
slot_i^T = (g_i, u_i, v_i)
slot_i^S = (P_key g_i, P_key u_i, P_val v_i)
retrieval(h_S) = sum_i softmax((A h_S)^T k_i^S) v_i^S
```

### Dry-Run / Sanity Check
Feasible on one RTX 5090: activation caches are MB to low-GB; SVD/CCA/Jacobian sketches are cheap relative to training; real checkpoint loading is the main operational cost; no training runs are required for Tier 1/Tier 2.

### Evidence Gate
Prove the roadmap if Tier 1 exact case passes, raw SVD fails under gauge, gauge-fixed transplant beats shuffled controls, and at least one method improves a real frozen Sutra/Wide7 probe by >=2pp while beating shuffled transplant. Kill it if no method clears Tier 2, real gains match controls, or byte/token gauge maps are too weak.

Current evidence-gated verdict: valid roadmap, not a positive transplant claim yet.

### Attack
Hostile expert objection: You still have not shown intelligence transfer. You have plausible compression and interpretability machinery.

### What Survived
The strongest honest claim is: direct transplantation is only meaningful after gauge-fixing. The first build should prove or kill that premise.

### NARRATIVE GATE
- Gossip-magazine headline: The first real test for copying intelligence without training.
- "Isn't that obvious?": No. The obvious tests copy coordinates or outputs.
- "So what?": If any method passes, Sutra has a credible no-gradient transplant path.
- Honest assessment: promising roadmap, not breakthrough evidence.

### Next Iteration
Implement `toy_weight_transplant_gauntlet.py` and run Tier 1/Tier 2 before touching production checkpoints.

## SYNTHESIS: After 10 Iterations

### Top Approaches (Ranked)
1. Gauge-Fixed Operator Transplant
   - Impact: very high.
   - Feasibility: medium-high on toy, medium on real Sutra.
   - Narrative: strong but technical.
   - Verdict: build first after gauntlet scaffold.

2. MLP Memory-Slot Transplant
   - Impact: very high because it targets inherited world knowledge.
   - Feasibility: medium.
   - Narrative: strongest simple story: "steal the big model's flashcards."
   - Verdict: build second once gauge maps exist.

3. Renormalization Coarsening
   - Impact: high for same-family compression and chain-init.
   - Feasibility: medium-high.
   - Narrative: medium.
   - Verdict: useful if the project returns to byteified chain-init or larger Sutra anchors.

4. Circuit Genome Transplant
   - Impact: paradigm-level if automated.
   - Feasibility: low-medium.
   - Narrative: excellent.
   - Verdict: moonshot branch with strict extraction controls.

5. Kronecker/Tensor Factor Transplant
   - Impact: medium-high.
   - Feasibility: medium.
   - Narrative: medium.
   - Verdict: supporting method, not mainline.

6. Weight Genome by Spectral Development
   - Impact: extreme if true.
   - Feasibility: low.
   - Narrative: extreme.
   - Verdict: one harsh above-random smoke test only.

Raw SVD is dead as a transplant method and should be retained only as a control.

### Build This First
Build:

```text
code/toy_weight_transplant_gauntlet.py
research/WEIGHT_TRANSPLANT_GAUNTLET_SPEC.md
```

First milestone:

```text
Tier 1: linear known-gauge transplant
- Construct teacher function F = W2 W1.
- Create gauge-equivalent teacher W1' = A W1, W2' = W2 A^-1.
- Show raw SVD transplant changes under A.
- Show gauge-aware transplant recovers the same student function.
Gate: gauge-aware specificity >= 0.95 cosine/function agreement; raw SVD fails under non-orthogonal A.
```

Second milestone:

```text
Tier 2: nonlinear toy binding transplant
- Compare raw SVD, Procrustes operator, Jacobian fingerprint, MLP slot table, wrong/shuffled controls.
Gate: any real method must beat its strongest control by >=4pp and hold on unseen transformations.
```

Third milestone:

```text
Tier 3: real frozen probe
- Use S0/Wide7 hidden states and a token teacher if available.
- No training.
- Test whether transplanted operators or slots improve MCQ candidate margins.
Gate: >=2pp over frozen baseline and >=2pp over shuffled transplant on held-out HellaSwag/PIQA slices.
```

Do not start with Qwen-to-Sutra production surgery. It would confound gauge, tokenizer, architecture, and benchmark noise before the method has passed exact cases.

### The Math That Matters
Gauge dependence:

```text
h' = A h
W1' = A W1
W2' = W2 A^-1
W2' W1' = W2 W1
```

Whitened CCA/Procrustes:

```text
M = C_S^-1/2 C_ST C_T^-1/2
M = U Rho V^T
P = C_S^1/2 U_k V_k^T C_T^-1/2
```

Operator transplant:

```text
J_S = P_out J_T P_in^+
```

Jacobian sketch:

```text
J* = Y V^T (V V^T)^+
```

MLP slot decomposition:

```text
MLP(h) = sum_i silu(g_i^T h) (u_i^T h) v_i
```

Coarsening rule:

```text
if a_i(h) ~= c a_j(h):
  a_i(h)v_i + a_j(h)v_j ~= a_j(h)(c v_i + v_j)
```

### What's Genuinely Novel
- Treating weight transplantation as a gauge-fixing problem before treating it as compression.
- A no-gradient transplant gauntlet with deliberately gauge-equivalent teachers.
- Combining CCA gauge maps with direct operator/Jacobian block surgery.
- Treating MLP slots as portable memory only after causal margin scoring and shuffled controls.
- Reframing the semantic codec as a possible coordinate chart for future transplant, not another KD objective.

### What's Known But Misapplied
- SVD is known, but usually used without adversarial gauge tests.
- Procrustes/CCA is known, but usually used for representation comparison rather than executable weight conjugation.
- Jacobian matching is known, but usually optimized as a loss; here it becomes closed-form surgery.
- MLP key-value memory is known, but rarely turned into a no-training transplant object.
- Renormalization analogies are common, but need deterministic merge equations and behavior gates.

### Gossip-Magazine Summary
If this all works:

"Scientists find a way to copy the useful tricks inside a giant AI into a tiny one without retraining it."

Honest headline today:

"The first step is building a lie detector, because most ways of copying intelligence only copy the model's private coordinate system."
