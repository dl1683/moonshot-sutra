# QUESTION LOOP - Batch 1

Grounding: I read the required local history before this pass: `research/DEEP_RETHINK.md`, `research/VISION.md`, `research/STATUS.md`, and `../CLAUDE.md`. The live constraint is severe: byte-marginal KD improved BPB by 33-42% while barely moving HellaSwag; naive hidden alignment failed shuffled controls; operational-geometry losses did not survive matched controls; readout-only was huge on toy binding but not on real Sutra; width-only made byte modeling much better but left reasoning benchmarks flat; Option G's codec proved byte/token-identity retrieval, not yet semantic intelligence. The manifesto constraint is also severe: "Intelligence = Geometry, not Scale" is only real if the hard part is not secretly done by a teacher, a human labeler, or a huge training run.

## Iteration 1: Weight Space Is Not the Geometry

### Current Strongest Position

The tempting first answer is: a trained model is a point, or perhaps a basin, in weight space. If intelligence is geometry, maybe the "knowledge" is encoded in spectral structure, low-rank directions, singular subspaces, Hessian/Fisher eigenspaces, mode-connectivity paths, and mergeable task vectors. Direct extraction would mean taking a large model's weight geometry and reconstructing an equivalent smaller model by SVD, pruning, tensor factorization, Fisher-aware compression, or basis change.

### Steelman

This position has real evidence behind it.

Model soups show that independently fine-tuned checkpoints from the same pretrained model can often be weight-averaged into a better single model without inference cost. Task arithmetic shows that fine-tuning deltas can behave like meaningful directions in weight space. TIES-Merging and DARE show that parameter deltas contain redundancy and sign/interference structure that can be manipulated without ordinary training. Git Re-Basin shows that once permutation symmetries are accounted for, independently trained networks can sometimes be moved into near-linear connectivity. Linear mode connectivity says many solutions are not isolated miracles but connected regions.

There is a story here: the model is not a bag of numbers; it is a geometric object with symmetries and low-dimensional structure. If model merging can combine abilities without data and DARE can drop most deltas while keeping ability, perhaps intelligence is already stored in sparse transferable directions. If those directions can be found, maybe a smaller Sutra can inherit the important geometry without gradient-descent imitation.

The 5090 version is feasible: take Qwen3-0.6B and local Sutra checkpoints, compute layerwise SVD/effective rank, delta sparsity, Fisher or diagonal Hessian salience on a small corpus, activation-aware low-rank approximations, and try direct compression/initialization. This is laptop-scale.

### Attack

This is probably the wrong geometry.

The brutal objection is gauge dependence. A neural network has many parameterizations for the same function: neuron permutations, attention-head permutations, layernorm scale shifts, MLP neuron rescalings, residual-stream rotations, embedding/LM-head transformations, and architecture-specific hidden bases. Weight-space geometry is not intrinsic unless it is quotiented by these symmetries. A singular vector of a weight matrix can be meaningful in one gauge and meaningless after an equivalent reparameterization.

The merge literature mostly works for homologous models: same architecture, same tokenizer, same pretrained parent, same coordinate system. CBD's advantage in the local history is exactly this: coordinate continuity. That is not a proof that knowledge is directly transferable across Qwen token transformers and Sutra byte-patch transformers. It is evidence for the opposite: direct weight geometry works when the basis is already shared.

Even worse, spectral compression is not "intelligence extraction." SVD preserves matrix action under a norm, not task-relevant causal distinctions. A low-rank approximation of an MLP matrix might preserve common activation energy while deleting rare but crucial factual or reasoning circuits. Fisher-aware pruning gets closer, but the Fisher is still computed relative to a data distribution and an output interface. It is not a free-standing essence of intelligence.

The "photocopy a brain" story fails because the hard part is hidden in a shared coordinate system. Model merging says: if two models are already siblings, their deltas can sometimes be spliced. It does not say: extract a mind and transplant it into a different body.

### What Survived

The useful survivor is not raw weight space. It is the demand for a quotient geometry: identify objects invariant under permutations, rotations, rescalings, tokenization, and architecture. Weight spectra, Fisher eigenspaces, task vectors, and mode-connectivity paths can be diagnostic instruments, but only if they are tied to functionally invariant behavior.

The strongest surviving idea: do not ask "which weights matter?" Ask "which distinctions in the teacher's behavior remain stable under gauge changes, and can those distinctions be represented in a smaller basis?"

### What Died

Dead for moonshot purposes:

- Raw SVD/low-rank compression as "direct intelligence extraction."
- Weight-space interpolation across unrelated architectures.
- Treating model merging results as evidence that cross-architecture transplant is solved.
- Any narrative where "geometry" means "beautiful plots of weight spectra" rather than capability-preserving invariants.

### New Leading Direction

Move from parameter geometry to quotient/function geometry. A trained model should be treated as an equivalence class of mechanisms implementing a conditional distribution and decision behavior. The extraction target must be invariant under reparameterization.

Candidate target: a compact "capability geometry" consisting of:

- contexts grouped by which actions remain good;
- candidate continuations ordered by teacher preference margins;
- transformations that preserve or change those orderings;
- sparse internal features only if causal interventions prove they control those margins;
- a metric not on weights but on task-relevant distinctions.

### NARRATIVE ATTACK

1. "That's obvious" dismissal: Everyone knows neural networks have symmetries, mode connectivity, and mergeable deltas. This is just model merging plus compression.
2. "That's trivial" dismissal: The only working cases require same architecture or pretrained parent, so the hard part is the shared coordinate system. The transplant is not real.
3. What the result needs to BE for unkillable narrative: A large token model and a small byte model with no shared tokenizer or architecture, where an automatic geometry extractor produces an initialization or module that improves held-out reasoning before any KD-style training.

### Gossip-Magazine Headline

Scientists find AI's "mind-shape" and move it into a smaller machine without copying its body.

### Next Iteration Starting Position

Attack the new claim that function-level or behavioral invariants are the true transferable geometry.

## Iteration 2: Behavior Alone Collapses Back Into Distillation

### Current Strongest Position

The previous round killed raw weight geometry and left a function/invariant view: knowledge is not theta, but the equivalence class of behavior implemented by theta. So perhaps the geometry of intelligence is the teacher's behavioral manifold: which inputs are close because they imply the same actions, which counterfactuals flip the answer, which candidate margins are stable, and which transformations preserve meaning.

### Steelman

This is much harder to dismiss than weight geometry.

Behavior is public. It survives tokenizer changes and architecture changes. A token transformer and a byte-patch model can disagree about hidden coordinates but still rank "the man lifts the barbell" above "the man eats the barbell." Operational geometry from the local history was exactly this insight: coordinate losses transfer coordinates; behavioral losses transfer behavior; invariant losses transfer knowledge.

This also connects to CKA/model-stitching style reasoning: compare representations by functional interchangeability rather than coordinate equality. It connects to the Neural Tangent Kernel view: the model's training dynamics can be studied in function space, not just parameter space. It connects to information geometry: the Fisher metric measures sensitivity of output distributions, making parameter directions meaningful only through changes in behavior.

The 5090 feasibility is decent. We can query Qwen3-0.6B on carefully generated local neighborhoods, build a graph of contexts/candidates/transformations/margins, then compile lessons into Sutra training or initialization. This does not need a cluster.

### Attack

This may be just distillation with better branding.

If the teacher produces labels, rankings, explanations, transformations, or margins, and the student is trained to match them, we are still in the paradigm the user explicitly wants to leave: "train a neural network to match another neural network's behavior using some loss function." The unit changed from token logits to candidate rankings, but the dependency structure is the same. The teacher does the hard semantic work; the student imitates.

The local repo already ran this movie. Operational Geometry OG-1 looked promising, then matched controls showed much of it was augmentation or fragile ranking behavior. OG-1b killed multi-loss operational geometry. The toy readout breakthrough did not transfer to real Sutra. TAD-1 teacher-as-data was predicted to be a formal kill-gated control, not the moonshot. The history warns that "behavioral lessons" can easily become glorified supervised fine-tuning.

The story fails the "that's trivial" test if the impressive result depends on millions of teacher-labeled examples. A gossip reader hears "tiny AI learned from big AI's examples," not "scientists discovered AI DNA." ML researchers hear "dataset distillation, preference distillation, data augmentation, or curriculum learning."

The deeper problem: the full function of a language model is astronomically large. You cannot extract it by black-box sampling. Any finite behavioral graph is an empirical training set unless it has a compressed generative law. Without that law, behavior geometry is not direct extraction; it is expensive observation.

### What Survived

Behavior is still the right invariant, but the transferable object cannot be a pile of teacher answers. It must be a compact rule, basis, or constructor that explains many teacher behaviors with less description length than the examples themselves.

This pushes toward MDL: intelligence geometry is not the graph of all teacher outputs; it is the minimal program that generates the teacher's stable distinctions over a domain. A "lesson" is only nontrivial if it compresses behavior.

### What Died

Dead:

- Ranking losses as the main moonshot.
- Teacher-as-data as a paradigm shift.
- Operational geometry if it is just more labeled neighborhoods.
- Any approach where the teacher remains the semantic engine and Sutra only memorizes outcomes.

### New Leading Direction

The target becomes a compressed behavioral law: extract "AI DNA," not answers. In biological terms, brains do not copy neuron states; organisms inherit developmental rules. The neural-net analog would be an automatically extracted developmental program that builds a compact semantic basis, circuits, or routing structure in the small model.

The leading question changes:

Can we infer a compact constructor from a teacher, such that the constructor generates a smaller model's semantic coordinate system without imitating every teacher output?

### NARRATIVE ATTACK

1. "That's obvious" dismissal: This is just distillation on rankings, counterfactuals, and augmented data.
2. "That's trivial" dismissal: The teacher still answers everything. The hard work is hidden in the teacher queries.
3. What the result needs to BE for unkillable narrative: The extracted artifact must be small, inspectable, and generative. It must produce many correct held-out distinctions that were not individually labeled by the teacher.

### Gossip-Magazine Headline

AI learns the rulebook inside a giant model, not just its answers.

### Next Iteration Starting Position

Attack the "AI DNA" or developmental-rule framing: is it real, or just a hypernetwork/meta-learning fantasy?

## Iteration 3: AI DNA Is a Beautiful Trap

### Current Strongest Position

The current leading direction is to extract a compact constructor: not weights, not logits, not examples, but the developmental rule that causes useful internal geometry to appear. The small model would not copy the teacher's body; it would grow a compatible semantic coordinate system from a compressed recipe.

### Steelman

This is the first direction with a genuinely viral story.

"AI DNA" is simple. A giant AI contains a code for intelligence; we read that code and grow a tiny AI with the same instincts. This is much stronger than "we trained a smaller model on better labels." It speaks directly to democratization: once the recipe is known, cheap machines can grow useful intelligence without giant compute.

There is also technical plausibility. Hypernetworks, CPPN/HyperNEAT-style generative encodings, lottery tickets, Net2Net, model growth, neural architecture search, tensor factorization, and low-rank adapters all suggest that weights may be generated from smaller descriptions. Transformers already reuse motifs: attention heads, MLP feature memories, residual-stream transport, induction-like circuits, token/position binding patterns. A developmental rule could encode repeated structure more efficiently than explicit weights.

The local history points the same way. The semantic codec thread says the bottleneck is semantic addressability. A codec is already a kind of developmental rule: first learn to read bytes into a teacher-anchored latent, then train the core over that coordinate system. The problem is that current Phase 1 proves token-identity retrieval, not semantic addressability. A true AI DNA program would go deeper: it would specify how to build the semantic address space itself.

5090 feasibility: a small experiment can fit. Use Qwen3-0.6B as specimen, extract repeated low-rank motifs from activation transitions and weight blocks, synthesize a small Sutra initialization or adapter from a generative rule, and test before ordinary training.

### Attack

This may be pure metaphor.

Biological DNA works because the organism has a rich developmental chemistry. A neural network has no comparable built-in morphogenesis unless we design one. If we build a hypernetwork that emits Sutra weights, what trains the hypernetwork? If gradient descent trains it to match teacher behavior, we are back to distillation. If evolution searches over constructors, the 5090 will not scale. If a human designs the constructor, the story fails: the hard part is hand-engineered.

The phrase "AI DNA" can conceal an impossibility: there may be no short program that maps a Qwen-like teacher into a 121M byte model with retained commonsense. Kolmogorov compression exists in principle but is uncomputable in general. "Find the minimal program" is not an algorithm.

The local data is hostile. Pythia-160M with enormous data remains near low HellaSwag; Sutra Wide7 improves BPB, speed, and LAMBADA but not commonsense; S0 hidden states do not contain HellaSwag knowledge. CBD succeeds because coordinate continuity preserves a learned basis. That is not a developmental law; it is inherited weights through a homologous chain.

The "AI DNA" story survives only if the constructor is automatically extracted from the teacher's mathematical structure and produces held-out ability. Otherwise, it is either hypernetwork distillation or myth.

### What Survived

The developmental framing survives as a constraint on the artifact: it must be compact and generative. But "DNA" is too loose. We need a mathematically specified object.

The surviving candidate is a circuit/feature atlas plus transition laws:

- a sparse dictionary of teacher features;
- causal tests identifying which features affect candidate margins;
- transition operators describing how features compose across tokens/layers;
- a compiler that maps the atlas into a smaller architecture's basis.

This is less romantic but more concrete. It says: extract the teacher's reusable computational primitives and their composition law, then reconstruct them in a smaller basis.

### What Died

Dead:

- Vague developmental metaphors.
- Hypernetworks trained end-to-end to imitate teacher outputs as a moonshot.
- Evolutionary search as the main path on a single laptop.
- "Find the Kolmogorov program" without an operational approximation.

### New Leading Direction

Mechanistic transplant: identify causal circuits or sparse features inside the teacher, then compile those circuits into Sutra. This is closer to "understanding the mathematical structure itself." It replaces behavior imitation with program analysis.

Possible tools:

- sparse autoencoders and crosscoders for feature dictionaries;
- causal activation patching/interventions for feature effects;
- model stitching/probes to test functional interchangeability;
- low-rank/tensor decompositions for repeated operators;
- Fisher/NTK sensitivity to identify which directions matter for margins.

### NARRATIVE ATTACK

1. "That's obvious" dismissal: Everyone wants mechanistic interpretability and circuit extraction. Calling it DNA adds no substance.
2. "That's trivial" dismissal: If the constructor is trained by gradient descent against teacher outputs, the hard part is still imitation.
3. What the result needs to BE for unkillable narrative: An extracted, compact feature/circuit atlas must initialize or assemble a small model that gains capability on tasks not used to build the atlas, with shuffled/rotated/control atlases failing.

### Gossip-Magazine Headline

Researchers turn a giant AI into a blueprint and grow a pocket-sized one from it.

### Next Iteration Starting Position

Attack mechanistic transplant: can circuits actually be found and moved, or are they too distributed, superposed, and architecture-bound?

## Iteration 4: Circuits Are Not Organs You Can Transplant

### Current Strongest Position

The current position is that the teacher contains causal computational primitives. If we can identify sparse features and circuits, we can compile them into a smaller model's basis. This would be closer to direct extraction than KD: program analysis, not output imitation.

### Steelman

This is the most literal interpretation of "what is the geometry inside a trained network?"

Mechanistic interpretability has real traction. Sparse autoencoders can decompose polysemantic activations into more interpretable features. Induction heads, truth directions, refusal directions, and task-specific causal circuits suggest that at least some capabilities have localized or low-dimensional structure. The Linear Representation Hypothesis, even with caveats, says concepts often behave like directions or low-rank subspaces under the right inner product. Model stitching asks whether one model's representation can be functionally connected to another's downstream computation.

If intelligence is in a sparse feature algebra, a small model does not need all teacher weights. It needs the right feature dictionary and the right transition/composition operators. This fits the geometric-limits warning: if width caps feature count, transfer the features that matter and ignore the rest.

The 5090 path is plausible for a 0.6B teacher:

- collect activations on a restricted corpus and benchmark-style candidate sets;
- train or fit SAEs on selected layers;
- identify features causally affecting candidate margins;
- build a feature-transition graph;
- synthesize a student energy/readout or initialization from that graph;
- test controls: random features, shuffled feature labels, random rotations, noncausal high-variance features.

### Attack

The phrase "transplant circuits" is misleading.

Circuits are not neatly boxed modules. Modern interpretability keeps finding superposition, distributed computation, duplicate features across layers, context-dependent feature semantics, and low modularity. Causal head gating finds heads whose roles depend on interactions with other heads. A feature that is causal in Qwen's residual stream may require the entire surrounding residual basis, layernorm statistics, MLP gates, attention routes, and tokenization to function.

Even if an SAE feature is interpretable, it is an activation-space coordinate, not a weight-space constructor. Turning a feature dictionary into a working small model is the hard step. If we train Sutra to activate those features, we are back to representation alignment. The local toy hidden-alignment failure is the warning: matching teacher coordinates, even contextual hidden states, learned marginal geometry and failed shuffled controls. A dictionary is still gauge-dependent unless its causal role is expressed in public variables.

Mechanistic interpretability is also expensive. A serious circuit analysis of HellaSwag-level commonsense in a 0.6B model may be larger than the original training problem. It may produce beautiful local facts with no scalable compiler.

The story risks becoming "we used a giant AI and interpretability tools to hand-pick circuits." That does not democratize intelligence. It creates a priesthood of circuit surgeons.

### What Survived

What survives is not transplanting whole circuits. It is extracting a gauge-invariant feature algebra.

A feature is admissible only if it passes three tests:

1. Decodability: it can be detected in teacher activations.
2. Causality: intervening on it changes teacher behavior on a defined distinction.
3. Public grounding: the same distinction can be described over inputs/candidates/transforms, not only in teacher coordinates.

The transferable object is not the SAE feature vector. It is the relation:

`public condition -> latent feature -> causal margin effect -> composition rule`

This becomes a small algebra of semantic operations, not a map of neurons.

### What Died

Dead:

- Naive circuit transplant.
- SAE feature dictionaries as sufficient by themselves.
- Hidden-state/feature alignment without causal and shuffled controls.
- Interpretability as a human-narrated explanation rather than an executable compiler input.

### New Leading Direction

Build a causal semantic atlas. It is a hybrid object:

- internal enough to exploit teacher geometry;
- behavioral enough to be gauge-invariant;
- compact enough to be generative;
- executable enough to initialize or constrain Sutra without endless teacher labels.

The atlas should be organized around candidate-margin effects, not around natural-language feature names. For Sutra, the first domain should be continuation discrimination, because BPB/task disconnect is the central wound.

### NARRATIVE ATTACK

1. "That's obvious" dismissal: This is just mechanistic interpretability plus sparse autoencoders.
2. "That's trivial" dismissal: Humans or giant interpretability pipelines do the circuit discovery; the small model only receives curated modules.
3. What the result needs to BE for unkillable narrative: An automatic atlas builder must discover causal features and compile them into a small model, with no manual circuit selection, and the compiled model must improve on held-out tasks before KD.

### Gossip-Magazine Headline

Scientists find the tiny switches inside a giant AI that make it choose the right answer.

### Next Iteration Starting Position

Attack the causal semantic atlas: does it still depend on teacher outputs and learned feature extractors, making it another disguised training signal?

## Iteration 5: The Atlas Still Smuggles the Teacher

### Current Strongest Position

The leading position is now a causal semantic atlas: automatically extract teacher features that are decodable, causally tied to margins, publicly grounded, and compositional. Then compile the atlas into Sutra.

### Steelman

This is the cleanest bridge between the manifesto and the failed repo history.

It respects gauge freedom: no raw hidden cosine worship. It respects the "that's trivial" test better than teacher-as-data: the output is not a dataset but an atlas of causal distinctions. It respects the local evidence that real Sutra lacks HellaSwag representation: the atlas attacks semantic addressability, not readout or BPB. It also refines Option G: instead of token-identity codec, build a margin-causal codec where context/candidate geometry is shaped by distinctions the teacher actually uses.

A practical formulation:

- Define contexts x and candidates y.
- Query teacher only to establish margin orderings and sensitivity.
- Collect teacher activations at selected layers.
- Learn sparse features on activations.
- For each feature z_i, estimate causal effect on margins by patching, ablation, or feature steering.
- Keep only features whose effect generalizes across paraphrases/counterfactuals.
- Fit a compact student basis whose Gram matrix and transition laws preserve these causal relations.

This has a clear metric: before any CE/KD continuation training, does the compiled basis improve HellaSwag/PIQA/ARC margins relative to random and shuffled atlases?

5090 feasibility: restricted version yes. Use Qwen3-0.6B, maybe 10k-50k contexts, top layers only, randomized SVD/SAE with modest feature count, candidate-margin probes. The full atlas is expensive, but the falsifier is not.

### Attack

The teacher is still doing the semantic work.

If the atlas asks "which candidate is better?" and Qwen answers, then Qwen supplies the target distinction. If it asks "which feature changes the margin?" and intervention confirms it, the teacher's output head still defines the effect. This may be a more surgical distillation signal, but it is a signal. The user asked for extraction of mathematical structure, not better labels.

The strongest hostile version: the atlas is just distillation with an interpretability bottleneck. It adds an SAE and causal filtering between teacher and student, but the final effect is still "student learns teacher's distinctions." It might improve science, but not paradigm.

There is also a compression danger. A causal atlas over HellaSwag margins might become task-specific. It would improve one benchmark but fail the sacred outcome of genuine intelligence. A cheap model with a HellaSwag atlas is not a democratized intelligence engine; it is a clever benchmark adapter.

And a mathematical danger: causal features in activation space may not compose. Feature A and feature B can each be causal alone, yet their combination can be nonlinear, context-dependent, or suppressed by another circuit. The atlas may be a bag of local effects, not a global geometry.

### What Survived

The atlas survives only if it becomes self-contained after extraction. The teacher may be used as a specimen once, but the extracted structure must generalize without further teacher queries and without per-example labels.

The key upgrade: model the atlas as a metric/transition system, not a list of features.

Objects:

- feature states s in a sparse semantic basis;
- input operators A_token or A_patch that update s;
- candidate operators B_y that map candidate text into comparable states;
- margin functional M(s_context, s_candidate);
- invariance group G of transformations that preserve M;
- counterfactual operators C_v that predict margin flips.

This is closer to a program. It can be tested on unseen examples by executing the transition system.

### What Died

Dead:

- Atlas as curated feature list.
- Benchmark-specific margin distillation.
- Teacher query graph without a compressed executable model.
- "Semantic codec" claims based only on token-embedding retrieval.

### New Leading Direction

Extract an executable semantic dynamics model: a small state machine / energy system / operator algebra derived from teacher structure. Sutra then receives this as its semantic address layer or initialization.

This is not full symbolic AI. It is a neural operator algebra:

`bytes -> sparse semantic state -> transition operators -> candidate energy`

The moonshot claim would be: the giant model's intelligence is compressible into a small operator algebra that a byte model can use as its semantic coordinate system.

### NARRATIVE ATTACK

1. "That's obvious" dismissal: This is just representation learning with sparse features and a classifier.
2. "That's trivial" dismissal: The teacher labels all distinctions, so the hard semantic work is outsourced.
3. What the result needs to BE for unkillable narrative: The extracted operator algebra must predict new teacher/student distinctions compositionally, not by stored examples, and it must improve multiple held-out reasoning tasks from the same compact artifact.

### Gossip-Magazine Headline

Tiny AI gets a pocket-sized map of meaning extracted from a giant AI.

### Next Iteration Starting Position

Attack the operator-algebra idea: is "semantic dynamics" real, or is it just another learned latent model needing gradient descent and data?

## Iteration 6: Semantic Dynamics May Be the Same Old Training in Disguise

### Current Strongest Position

The leading direction is an executable semantic dynamics model: extract a sparse semantic basis and transition/energy operators from a teacher, then compile them into Sutra. This is meant to be the true "geometry": not coordinates, but dynamics of meaningful distinctions.

### Steelman

This direction has the best mathematical shape so far.

A trained transformer is not just weights; it is a compositional dynamical system over residual streams. Each layer updates a state; attention transports information; MLPs implement feature memories; layernorm stabilizes scale; the LM head reads out energies over tokens. If we quotient away gauge, what remains may be transition structure: which semantic variables are created, preserved, combined, suppressed, or made decision-relevant.

Information geometry gives a principled metric: directions matter to the extent they change the model's predictive distribution or decision margins. NTK/Fisher-like objects describe sensitivity of outputs to parameters or features. Topology can diagnose whether training creates global structures, but likely as a diagnostic rather than a compiler. Spectral analysis can expose low-rank operators. Representation theory/equivariant thinking says known symmetries should be built into the representation so learning does not waste capacity rediscovering them.

The practical Sutra version:

- build a teacher activation corpus over normal text plus candidate-comparison neighborhoods;
- estimate a causal/Fisher metric over activations with respect to candidate margins;
- compute an activation-aware low-dimensional basis under that metric, not Euclidean PCA;
- fit sparse transition operators between basis states across token/patch steps;
- initialize Sutra's codec/reasoner/energy head to realize those operators;
- evaluate before and after minimal CE.

This is no longer "match Qwen logits." It is "construct the small model in the teacher's capability coordinates."

### Attack

The word "fit" is doing dangerous work.

If we fit basis vectors, transition operators, and energy functions from activation data using optimization, are we not still training a model? The manifesto does not ban all math or all optimization, but the user's target explicitly says not through training signals, not through gradient descent, not through loss functions. A semantic dynamics model trained by reconstruction loss could be seen as autoencoder distillation.

The honest distinction must be sharper:

- If we optimize a neural student to match teacher behavior, dead.
- If we solve a linear algebra/statistical estimation problem to identify an invariant object, possibly alive.
- If we then use that object to initialize/assemble a model, alive only if the capability appears before normal training can explain it.

There is also the "basis trap." A Fisher/activation basis extracted from Qwen may still be Qwen-specific. Even if it is low-dimensional, Sutra's byte-patch architecture may not implement the same transitions. The compiler could become as hard as the original problem.

Finally, semantic dynamics may not be low-rank or clean. LLM knowledge could be a massive associative memory with no small general operator algebra. CBD's success may reflect brute inherited memory compressed through homologous weights, not a neat semantic law.

### What Survived

The strongest survivor is a narrower, falsifiable claim:

There may exist a compact semantic address basis for a restricted but important slice of language understanding, and the way to test it is not to train a full student. It is to build a geometry compiler and demand immediate capability lift from the compiled initialization/module.

This reframes direct transplant as a three-stage non-KD pipeline:

1. Extract invariants by measurement: covariance, Fisher sensitivity, causal margin effects, topology/spectral diagnostics.
2. Solve for a compact basis/operator system with mostly closed-form or convex/linear-algebraic steps: randomized SVD, Procrustes, CCA/CKA, sparse coding with strict controls, low-rank regression.
3. Assemble or initialize Sutra modules from the solution, then freeze most of it and test capability.

Gradient descent can be allowed only as a later engineering refinement, not as evidence for the moonshot.

### What Died

Dead:

- A neural semantic-dynamics model trained end-to-end as the evidence.
- Any claim that "we used an SAE" automatically escapes distillation.
- Euclidean activation PCA as intelligence geometry.
- Topological/category-theory diagnostics as a mainline unless they produce a compiler.

### New Leading Direction

The leading direction becomes:

Gauge-Invariant Semantic Basis Extraction (GISBE).

GISBE is not a loss for Sutra. It is a compiler:

- Input: a trained teacher as a mathematical specimen.
- Measurement: causal/Fisher-weighted geometry over activations and candidate margins.
- Quotienting: remove gauge artifacts by using relational objects: Gram matrices, margin effects, transformation laws, and operator spectra.
- Compression: find a small basis that preserves those relational objects.
- Reconstruction: assemble Sutra's semantic codec and judgment energy in that basis.
- Test: capability must appear before KD or large-scale CE.

### NARRATIVE ATTACK

1. "That's obvious" dismissal: This is just representation compression, CCA/CKA, Fisher pruning, and low-rank regression.
2. "That's trivial" dismissal: Optimization still fits teacher data; the hard work is merely hidden in a fancy compiler.
3. What the result needs to BE for unkillable narrative: A mostly algebraic extraction from one teacher checkpoint must produce a small byte model module that improves several unseen tasks immediately, while random rotations, shuffled margins, and noncausal bases fail.

### Gossip-Magazine Headline

Scientists decode the hidden coordinate system of AI thought and install it in a tiny model.

### Next Iteration Starting Position

Attack GISBE itself: maybe no such compact, cross-architecture semantic basis exists, and the only honest path is chain-init or retrieval.

## Iteration 7: The Won-Over Adversary Test

### Current Strongest Position

After six attacks, the strongest position is GISBE: extract a gauge-invariant semantic basis and operator system from a trained teacher, then compile that geometry into Sutra so a small byte model gets semantic addressability without ordinary KD.

### Steelman

GISBE is the only surviving direction that is both bold and not obviously fake.

It avoids raw weight worship by quotienting gauge. It avoids teacher-as-data by requiring a compact executable artifact. It avoids naive hidden alignment by using causal margin effects and relational geometry. It avoids pure interpretability theater by demanding a compiler. It respects the local failures: byte KL, hidden cosine, ranking losses, width-only, and real readout-only all died because they did not create semantic addressability in the 121M model. GISBE directly targets semantic addressability.

It also has a story that survives better than the others:

"Big AI models are expensive because they learn their own private coordinates for meaning. We found a way to read those coordinates once, compress them into a tiny public map, and give that map to small AI."

This is democratization. The expensive teacher is used once as a microscope specimen, not as a runtime dependency. The artifact could be shared openly. If it works, future small models start with semantic coordinates instead of spending billions of tokens discovering them.

The 5090 feasibility is real for a decisive falsifier:

- teacher: Qwen3-0.6B or another canonical local teacher;
- corpus: 10k-50k contexts plus candidate sets, not internet-scale;
- compute: activation extraction, randomized SVD, sparse basis, causal patching on selected features/layers;
- student: small module/initialization for Wide7/Sutra, not full retraining;
- gate: frozen or minimally trained energy/codec lift on held-out HellaSwag/PIQA/ARC.

### Attack

The adversary has three kill shots.

First: no compact basis may exist. Commonsense in an LLM may be a huge distributed associative memory. There may be low-rank local structure, but no small global atlas that preserves useful margins. CBD might work because a 138M homologous model can store compressed inherited weights, not because there is a neat semantic coordinate system.

Second: cross-architecture compilation may be the real impossibility. Even if Qwen has a compact semantic basis, Sutra may not implement the same operations efficiently because byte-patch input, patch lag, decoder mechanics, layer count/depth, and residual geometry differ. Coordinate continuity may be not a convenience but the essence of transfer.

Third: evidence can be faked easily. If GISBE uses teacher labels, trained probes, learned SAEs, and a small fine-tune, any lift can be dismissed as distillation, probing leakage, or benchmark-specific adaptation. The bar must be brutal.

The adversary is not fully defeated by a plausible plan. The adversary is defeated only by a falsifiable design whose controls would kill it if the story is fake.

### What Survived

GISBE survives, but only as a strict research program with hard evidence rules.

The minimal acceptable claim is not:

"We can directly transplant intelligence."

The honest claim is:

"We can test whether a trained teacher contains a compact, gauge-invariant semantic address geometry that can be extracted once and used to initialize a smaller byte-native model."

The exact surviving object is:

- not weights;
- not logits;
- not hidden states;
- not human-named circuits;
- not teacher-generated examples;
- but a causal relational basis: a compressed set of semantic coordinates plus transition and margin operators that preserve teacher-relevant distinctions under transformations.

This is revolutionary if it works because it changes the unit of transfer from examples/gradients to geometry.

### What Died

Dead or demoted:

- "Directly copy the geometry of weights."
- "Train on better teacher outputs."
- "Align hidden states."
- "Use rankings/counterfactuals as another auxiliary loss."
- "Add a judgment head before representation exists."
- "Make the model wider and hope reasoning appears."
- "Call token-identity retrieval a semantic codec."
- "Use category theory/topology as rhetoric without a compiler."

### New Leading Direction

The leading direction is a two-track falsifier:

Track A: Geometry Existence Test.

Measure whether teacher capability margins over continuation tasks have a compact relational representation. Build candidate bases from:

- Fisher-weighted activation covariance;
- causal feature effects on candidate margins;
- CKA/model-stitching relations across layers;
- low-rank transition operators;
- transformation-consistency constraints.

The basis is valid only if it predicts held-out teacher margin structure much better than random, shuffled, Euclidean PCA, token-frequency, and lexical baselines.

Track B: Geometry Transplant Test.

Compile that basis into Sutra as a frozen or mostly frozen semantic codec/energy module. The model must improve on held-out tasks before ordinary KD or large-scale CE can explain the improvement.

Hard gate:

- >=5pp HellaSwag lift over matched Wide7/S0 from the compiled module or initialization, before KD;
- PIQA/ARC aggregate lift >=2pp;
- shuffled/rotated/noncausal basis lift <=1pp;
- lexical/token-identity codec control fails to match;
- no teacher calls at inference;
- artifact size small enough to publish and reuse.

If Track A fails, the geometry does not exist in compact form. If Track A passes but Track B fails, the geometry exists but is not transplantable cross-architecture. If both pass, this is the first real evidence for "Intelligence = Geometry" in this repo.

### NARRATIVE ATTACK

1. "That's obvious" dismissal: This is a mashup of Fisher geometry, CKA, sparse features, model stitching, and compression.
2. "That's trivial" dismissal: The teacher and extractor still do the work; the small model receives a precomputed crutch.
3. What the result needs to BE for unkillable narrative: A released extractor takes a teacher checkpoint and a small unlabeled/candidate corpus, emits a compact geometry artifact, and a byte-native small model immediately gains reasoning accuracy across held-out tasks with no teacher at inference and controls dead.

### Gossip-Magazine Headline

A laptop reads the secret map inside a giant AI and gives it to a tiny one.

### Next Iteration Starting Position

The next batch should attack the operational details: how exactly to estimate a causal relational basis without leaking labels, overfitting HellaSwag, or reintroducing hidden KD.

## SYNTHESIS: After 7 Rounds of Attack

### What Survived Everything

The survivor is not "extract weights." It is not "match outputs." It is not "semantic codec" in the current token-identity sense.

What survived is:

**Gauge-Invariant Semantic Basis Extraction (GISBE): extract a compact causal relational geometry from a trained teacher and compile it into a smaller byte-native model as a semantic address system.**

The core surviving thesis:

A trained model's intelligence is best understood as a geometry of task-relevant distinctions. The coordinates are private and architecture-bound; the distinctions, transformation laws, and causal margin effects may be public. If those public relational objects admit a compact basis, then a small model can inherit semantic addressability without imitating byte logits or hidden coordinates.

This is still risky. It may fail. But it is the first direction in this loop that:

- directly attacks the byte-to-semantics bottleneck;
- respects gauge dependence;
- avoids glorified output KD as the evidence;
- has a nontrivial viral story;
- can be falsified on a single RTX 5090;
- would be genuinely paradigm-level if it worked.

### The Honest Narrative

The honest one-sentence story is:

**We are testing whether the expensive part of AI is not the answers, but the hidden coordinate system for meaning, and whether that coordinate system can be extracted once and installed in cheap models.**

This is not yet:

"We can photocopy a brain."

It is:

"We may be able to copy the map that lets a small brain find meaning."

### What We Should Build

Build the smallest brutal falsifier, not a full system.

1. Define a candidate-margin probe set.
   Use held-out HellaSwag-style, PIQA-style, and synthetic counterfactual examples. Keep train/test separation strict. Include lexical confound checks so the geometry cannot be token-frequency or first-byte statistics.

2. Extract teacher relational measurements.
   For Qwen3-0.6B, collect selected layer activations, logits only for candidate margins, and perturbation/patching measurements for a small number of contexts. Do not train a student yet.

3. Build competing bases.
   Compare at least: random basis, Euclidean PCA, token-embedding/identity basis, activation SVD, Fisher-weighted basis, causal-margin basis, sparse feature basis, and combined causal relational basis.

4. Track A gate: geometry existence.
   Can a compact basis predict held-out teacher margin geometry and transformation laws better than controls? If not, stop. Do not build Sutra integration.

5. Compile only after Track A passes.
   Map the basis into a small Sutra module using Procrustes/CCA/low-rank regression/closed-form initialization wherever possible. Avoid end-to-end KD as the proof.

6. Track B gate: transplant.
   Freeze most of the compiled module and test immediate lift. Require >=5pp HellaSwag and cross-task movement before claiming anything. Controls must fail: shuffled margins, random rotations, lexical-only codec, noncausal high-variance features.

7. Only then consider Option G Phase 2.5.
   Current Option G should be reframed as an engineering cousin, not the proof. Its Phase 1 token-identity retrieval is useful infrastructure, but not semantic geometry until margin-causal controls pass.

### What We Should NOT Build

Do not build another byte-marginal KL variant. The repo already proved BPB can improve dramatically while task accuracy stays flat.

Do not build another naive hidden-state alignment. The toy experiment showed correct vs shuffled contextual alignment were not meaningfully separated.

Do not build ranking/invariance/counterfactual multi-loss soup as the main line. OG-1b showed the multi-loss recipe did not survive controls.

Do not treat readout as solved path for real Sutra. The toy +77pp result was real, but the real S0 energy probe did not transfer.

Do not run width/depth sweeps as a moonshot. Wide7 improved byte modeling, speed, and LAMBADA, not commonsense reasoning.

Do not call token-embedding retrieval a semantic codec. It may be a necessary alphabet bridge, but the current evidence says token identity, not semantic addressability.

Do not pursue category theory, topology, or representation theory as document-only elegance. They become relevant only if they specify the compiler or the invariant tests.

Do not let teacher-as-data become the main story. It is a useful control and perhaps a practical ingredient, but it fails the "that's trivial" attack as a paradigm.

### Open Questions

Does a compact semantic address basis actually exist for HellaSwag-level commonsense, or is the knowledge too distributed and memory-like?

Can causal-margin geometry be estimated cheaply enough from a 0.6B teacher on a 5090, or does intervention cost explode?

Which metric is the right one: Fisher over logits, Fisher over candidate margins, NTK-like sensitivity, CKA relational geometry, causal effect size, or a hybrid?

Can a basis extracted from token-model activations be compiled into a byte-patch architecture without reintroducing ordinary KD?

How do we prevent lexical/token-identity leakage from masquerading as semantics?

What is the smallest artifact that would count as "AI DNA": a basis, a transition graph, an initialization, a codec, an energy module, or a full generated checkpoint?

What held-out tests prove general semantic addressability rather than HellaSwag adaptation?

If GISBE fails, is the honest moonshot pivot byteified chain-init plus compression, retrieval-augmented small models, or accepting a 300M-500M minimum viable Sutra?
