# Sources, Evidence, And Questions For Eklavya/Sutra

This file tracks only sources and questions that help design Eklavya as a
multi-teacher learning system or Sutra as an efficient student.

## Local Evidence To Preserve

### Legacy Sutra Principles

Source: `../moonshot-sutra-legacy-pre-reset-2026-06-06/legacy_pre_eklavya_reset_2026-06-06/CLAUDE.pre_reset.md`

Useful principles:

- Sutra is an edge-oriented model/system whose thesis is compression as
  intelligence.
- Outcomes matter more than mechanisms.
- Data efficiency through multi-source learning and inference efficiency through
  adaptive compute are core outcomes.
- Multi-source knowledge absorption must be designed up front. "Add KD later"
  is not a plan.
- Every file and component has negative valence until it proves active utility.
- Scaling should seek asymmetric capacity gains, not only more parameters.

### Legacy Architecture Review

Source: `../moonshot-sutra-legacy-pre-reset-2026-06-06/legacy_pre_eklavya_reset_2026-06-06/tesla_session_design.md`

Useful evidence:

- Covering decomposition and routing worked better than arithmetic mean
  teacher mixing.
- Arithmetic mean teacher aggregation was harmful.
- KD with two transformer teachers had a low apparent ceiling.
- Student capacity, not teacher signal, was likely the bottleneck.
- Byte-global computation likely imposed too much sequence tax.
- Token-global/byte-local architecture was already identified as a stronger
  candidate.
- MoE below 500M total parameters was uncertain; sparse FFN was a cleaner first
  capacity multiplier.
- Compiled teacher artifacts were useful, but should not be treated as proof of
  learning by themselves.

### Legacy Eklavya Gate Result

Source: `../moonshot-sutra-legacy-pre-reset-2026-06-06/legacy_pre_eklavya_reset_2026-06-06/results/probe_eklavya_full_protocol_gate_r1.json`

Important result:

- The full Eklavya claim did **not** pass.
- Missing precise eval and trace audit artifacts blocked the claim.
- The candidate had three teachers across two families, including an encoder,
  but evidence was insufficient.
- The BGE state-channel probe was promising but not enough.

Lesson:

```text
multi_teacher_presence != Eklavya_success
```

### Legacy Multisource Encoder Probe

Source: `../moonshot-sutra-legacy-pre-reset-2026-06-06/legacy_pre_eklavya_reset_2026-06-06/results/probe_multisource_encoder.json`

Important result:

- Student/BGE linear CKA was high.
- Shuffled baseline was near zero.
- Decision was to skip alignment loss because alignment was already strong.

Lesson:

```text
teacher_signal_should_target_unowned_structure_not_already_aligned_structure
```

## External Research Sources

### Compute-Optimal Training

Source: <https://arxiv.org/abs/2203.15556>

Use:

- Sutra efficiency must balance model size and training tokens.
- A smaller model trained longer or better can beat a larger undertrained model.
- Efficiency cannot be judged by parameter count alone.

### Inference-Aware Scaling

Source: <https://arxiv.org/html/2401.00448v2>

Use:

- If inference demand is high, optimal training may favor smaller models trained
  more heavily.
- Sutra's target should be lifecycle cost, not pretraining cost alone.

### Multi-Teacher Knowledge Distillation

Source: <https://arxiv.org/abs/2302.07215>

Use:

- Multi-teacher compression can work.
- But Eklavya must go beyond ensemble compression by preserving teacher identity,
  disagreement structure, and retained teacher-free competence.

### DeepSeek-R1 Distillation

Source: <https://arxiv.org/abs/2501.12948>

Use:

- Reasoning patterns from larger models can transfer into smaller models.
- Eklavya should study whether reasoning traces, not just answer logits, are the
  transferable object.

### Sparsely Gated Mixture Of Experts

Source: <https://arxiv.org/abs/1701.06538>

Use:

- Conditional computation can increase capacity without proportional active
  compute.
- But MoE complexity and routing instability must be treated as risks at Sutra's
  likely scale.

### Mixture Of Depths

Source: <https://arxiv.org/abs/2404.02258>

Use:

- Dynamic token-level compute allocation directly supports Sutra's inference
  efficiency outcome.
- The important idea is predictable total budget with content-sensitive compute.

### LayerSkip And Self-Speculative Decoding

Source: <https://arxiv.org/abs/2404.16710>

Use:

- Early exit plus self-verification can make a single model act like a cheap
  draft and stronger verifier.
- This fits Sutra better than always using maximum depth.

### Big Little Decoder

Source: <https://arxiv.org/abs/2302.07863>

Use:

- A small model can draft and a larger process can correct occasionally.
- Sutra may internalize this pattern as shallow/deep self-speculation rather
  than permanent dependence on an external large model.

### Matryoshka Representation Learning

Source: <https://arxiv.org/abs/2205.13147>

Use:

- Nested representations support multiple cost/quality tradeoffs.
- Sutra should consider representations that remain usable at partial width,
  partial depth, or partial precision.

### Textbooks Are All You Need

Source: <https://arxiv.org/abs/2306.11644>

Use:

- High-quality, curated synthetic or textbook-like data can create strong small
  models.
- Eklavya should treat teachers as curriculum builders, not only output
  distributors.

### TinyStories

Source: <https://arxiv.org/abs/2305.07759>

Use:

- Small models can show surprising capability when the data distribution is
  carefully shaped.
- Sutra may need high-density developmental curricula rather than massive raw
  text.

### LIMA

Source: <https://arxiv.org/abs/2305.11206>

Use:

- Small, carefully curated instruction data can strongly shape behavior.
- Quantity is not the only lever; selection and exemplar quality matter.

### Byte Latent Transformer

Source: <https://arxiv.org/abs/2412.09871>

Use:

- Supports the idea that byte-level models can use dynamically sized patches as
  primary compute units.
- Important for Sutra because it preserves byte truth while avoiding byte-level
  global compute everywhere.
- Entropy-based patching is a direct candidate for the local byte encoder and
  compute controller.

### MEGABYTE

Source: <https://arxiv.org/abs/2305.07185>

Use:

- Supports a local/global multiscale sequence architecture.
- Important for Sutra because it separates local byte modeling inside patches
  from global modeling between patches.
- Gives architectural precedent for sub-quadratic byte-level modeling.

### Charformer

Source: <https://arxiv.org/abs/2106.12672>

Use:

- Supports learned subword/block formation from characters or bytes.
- Important for Sutra because the patch/token unit can be learned rather than
  inherited from a tokenizer.

### SSM Architecture

Source: <https://arxiv.org/abs/2312.00752>

Use:

- Selective state spaces are relevant as a cheap long-context lane.
- Sutra should not replace attention blindly, because content-based reasoning is
  still central, but an SSM lane may be useful for memory-like propagation at
  lower cost.

## Sibling Project Evidence

### Eklavya Teacher Tomography

Source: `../Eklavya/SYSTEM.md`

Useful idea:

- Distill behavior signatures under controlled probes rather than decoded
  answers, full-vocab KL, or architecture-coupled hidden states.
- Candidate-string distributions are cross-tokenizer and avoid full-vocab KL.
- The measured object is a response surface: invariance, sensitivity,
  calibration, and margins.

Design consequence:

```text
Sutra_Eklavya_should_keep_teacher_signature_tensors_not_early_teacher_averages
```

### Tokenization Limit

Source: `../moonshot-tokenization-limit/research/PROOF_SKETCH.md`

Useful idea:

- Tokenizers create quotient geometries: distinctions below tokenization can be
  unrecoverable from tokens alone.
- The escape is to change the representation, for example with hierarchical or
  residual byte channels.

Design consequence:

```text
Sutra_should_preserve_raw_bytes_or_residual_bytes_even_if_global_reasoning_uses_patches
```

### Fractal Embeddings

Source: `../moonshot-fractal-embeddings/research/FRACTAL_INTELLIGENCE_THEORY.md`

Useful idea:

- Hierarchical scale-separated representations may reduce compositional burden.
- The local claim should be treated as a hypothesis, not imported as proven law
  for Sutra.

Design consequence:

```text
Sutra_should_consider_scale_separated_states_as_a_representation_prior_and_measure_not_assume_their_value
```

### Neural Genome

Source: `../moonshot-llm-genome/README.md`

Useful idea:

- Sparse geometric transfer did not recover coherent capability; NLL recovery
  was decoupled from generation coherence.
- Geometry is more reliable as a diagnostic and health monitor than as a magic
  transplant method.

Design consequence:

```text
Eklavya_should_use_geometry_measurements_to_decide_where_to_teach_not_assume_geometry_transfer_is_capability_transfer
```

## Questions That Matter

### Eklavya Questions

1. What is the smallest useful definition of a teacher?
2. Which teacher families expose different invariants rather than correlated
   echoes?
3. How do we tell alignment noise from useful disagreement?
4. Which teacher signals should land in the local encoder, global reasoner,
   local decoder, memory, or router?
5. What makes a teacher contribution survive after teacher removal?
6. Can a symbolic or verifier teacher teach constraints better than a decoder
   teacher?
7. Can teacher disagreement become curriculum without becoming entropy?
8. What is the minimum teacher-port schema that preserves enough provenance
   without becoming bureaucracy?

### Sutra Questions

1. What is the correct global computation unit: token, learned patch, byte span,
   or variable patch?
2. Can byte-local/token-global preserve exactness while escaping byte sequence
   tax?
3. Which conditional compute mechanism is simplest and strongest at small scale?
4. Does sparse FFN provide enough effective capacity before MoE complexity is
   justified?
5. Can early exits be trained without damaging hard-case reasoning?
6. Can representations be nested so cheap modes remain useful?
7. What is Sutra's first honest "David story" baseline?
8. Which metric best captures lifecycle intelligence per cost?

### Joint Eklavya/Sutra Questions

1. What architecture gives teacher invariants places to land?
2. What evidence proves the student owns the invariant?
3. What evidence proves the gain is not just better data curation?
4. How much teacher cost is acceptable per retained gain?
5. Can teacher-compiled data reduce training tokens enough to matter?
6. Can the same teacher artifacts also train active compute decisions?
7. When should Eklavya choose not to teach because the student is too small?

### SE1 Architecture Questions

1. Should the patcher be entropy-based, learned block scoring, or hybrid?
2. Can patch boundaries themselves be teacher signals?
3. Should the global reasoner use attention only, or attention plus a selective
   SSM lane?
4. Is sparse FFN enough conditional capacity before MoE?
5. Can teacher tomography train early exits by identifying easy response
   surfaces?
6. Which invariant packets are worth keeping if optimization becomes unstable?
7. What is the cheapest retained-gain test that catches teacher dependence?
8. Can scale-separated states produce measurable gains on compositional tasks,
   or are they speculative overhead?

## Current Research Bias

The most promising direction is not "more teachers." It is:

```text
better student substrate
  + teacher-specific ports
  + invariant compilation
  + disagreement-driven curricula
  + retained-gain measurement
  + adaptive compute
```

This should guide every next research pass.

## Source Priority For Next Pass

Highest priority:

- BLT, MEGABYTE, and Charformer for student substrate.
- Eklavya teacher tomography for teacher signatures.
- Tokenization-limit proof for byte residual necessity.
- LayerSkip, Mixture-of-Depths, and SSM for conditional compute.

Lower priority:

- Fractal hierarchy as a representation prior.
- Neural Genome geometry as diagnostic tooling.

Do not let lower-priority geometry speculation replace the concrete
byte-preserving patch-global student design.

## Research Refresh: SE1 Interfaces

### EAGLE

Source: <https://arxiv.org/abs/2401.15077>

Use:

- Supports the idea that feature-level prediction can accelerate autoregressive
  decoding.
- Useful for Sutra's self-speculative path: shallow or intermediate states may
  draft future states, with deeper computation verifying hard cases.

### Medusa

Source: <https://arxiv.org/abs/2401.10774>

Use:

- Shows that extra decoding heads can predict multiple future tokens and reduce
  decoding steps.
- Useful for Sutra if multi-head prediction can be trained as an internal
  efficiency head rather than as an external draft model dependency.

### Distilling Step By Step

Source: <https://arxiv.org/abs/2305.02301>

Use:

- Supports richer teacher supervision than final labels.
- Relevant to Eklavya because reasoning rationales, verifier spans, and
  intermediate constraints may be higher-value teacher objects than answer
  distributions alone.

## SE1 Interface Questions

1. What should `Ksig` contain in the teacher signature tensor: probability,
   margin, entropy, rank, calibration, or all five?
2. Should patch entropy be learned only from bytes, or should teacher
   disagreement also influence patch boundaries?
3. Should self-speculative heads predict bytes, patches, candidates, or hidden
   states?
4. Should retained gain be measured on task accuracy, BPB, generation quality,
   verifier pass rate, or a vector of all four?
5. How do we price teacher signature cost fairly against extra training tokens?
6. Can curriculum packets improve data efficiency without narrowing the model?
7. What is the minimum comparison suite that catches fake Eklavya without
   becoming benchmark sprawl?

## Updated Source Priority

Highest priority:

- BLT, MEGABYTE, and Charformer for student substrate.
- Eklavya teacher tomography for teacher signatures.
- Tokenization-limit proof for byte residual necessity.
- LayerSkip, Mixture-of-Depths, SSM, EAGLE, and Medusa for conditional compute.
- Distilling Step By Step for rationale/intermediate-supervision teacher
  packets.

Lower priority:

- Fractal hierarchy as a measured representation prior.
- Neural Genome geometry as diagnostic tooling.

## Research Refresh: SE1 Roster And Loss Schedule

### Decoder Teacher Family

Sources:

- <https://arxiv.org/html/2505.09388v1>
- (HF model page for 8B embedding variant)
- <https://huggingface.co/papers/2506.05176>

Use:

- Candidate decoder-teacher family across practical sizes, including cheap
  small controls and stronger offline teachers.
- Candidate embedding-teacher family for instruction-aware geometry anchors.
- Useful for Eklavya because one family gives scale controls, but that also
  creates a redundancy risk. SE1 must not mistake same-family agreement for
  true cross-teacher invariance.

### SmolLM3

Sources:

- <https://huggingface.co/HuggingFaceTB/SmolLM3-3B>
- <https://huggingface.co/blog/smollm3>

Use:

- Candidate efficient decoder peer around the 3B class.
- Useful as a reference for staged curriculum, long-context engineering, and
  think/no-think behavior surfaces.
- Should be included only if its probe surface adds distinct invariants beyond
  the primary decoder family.

### BGE-M3 And BGE Small

Source: <https://huggingface.co/BAAI/bge-m3>

Use:

- Candidate encoder teacher for dense, sparse, and multi-vector geometry.
- Useful for retrieval/paraphrase anchors and for testing whether encoder
  teachers add retained gain beyond decoder-only distillation.
- BGE-small is a scale/control option when a large encoder gain might just be
  capacity or evaluator coupling.

### Loss Balancing Sources

Sources:

- GradNorm: <https://arxiv.org/abs/1711.02257>
- PCGrad: <https://arxiv.org/abs/2001.06782>
- Uncertainty weighting: <https://arxiv.org/abs/1705.07115>

Use:

- These are candidate tools for stage-four packet conflict control.
- They should not become default complexity. The default should be fixed
  scheduled weights until packet gradients demonstrably conflict.
- The design question is not "which multitask trick is clever"; it is whether
  teacher-packet families can coexist without destroying retained gain.

## SE1 Roster Questions

1. Is the 4B decoder candidate affordable enough for first offline signatures, or should the
   initial decoder teacher be the 1.7B anchor with the 0.6B control as the scale control?
2. Does SmolLM3-3B produce distinct probe behavior, or is it redundant with
   same-family decoder teachers for the first pass?
3. Should the first encoder primary be embedding-candidate-0.6B for current
   instruction-aware embedding behavior or BGE-M3 for dense/sparse/multi-vector
   diversity?
4. Which measured packet-gradient conflict should trigger PCGrad, GradNorm, or
   uncertainty weighting instead of fixed scheduled weights?
5. How small can the student be while still distinguishing substrate failure
   from ordinary capacity failure?

## Research Refresh: Outside-Domain Design Patterns

### Profile-Guided Optimization

Source: <https://go.dev/doc/pgo>

Use:

- PGO feeds runtime profiles from representative executions back into the
  compiler so later builds make better static optimization decisions.
- SE1 analogy: teacher signatures are profiles. They should guide student
  training and compute planning, but they should not be needed at inference.
- Design consequence: Eklavya should create reusable profile artifacts, and
  Sutra should internalize the optimization.

### Cost-Based Query Optimization

Source: <https://research.ibm.com/publications/access-path-selection-in-a-relational-database-management-system>

Use:

- System R separated high-level SQL intent from physical access paths and used
  estimated total access cost to choose plans.
- SE1 analogy: a prompt, patch, or subproblem can have multiple physical compute
  plans: exit now, run deeper, verify, repair bytes, or consult student memory.
- Design consequence: Sutra's compute controller should be a cost-based planner
  with explicit action costs and expected quality deltas.

### Query By Committee

Source: <https://collaborate.princeton.edu/en/publications/query-by-committee/>

Use:

- Query-by-committee chooses examples by maximal disagreement.
- SE1 analogy: teacher disagreement should select probes and curriculum
  examples, but only after separating useful disagreement from alignment noise.
- Design consequence: Eklavya should build disagreement-localized curricula, not
  average away disagreement.

### Curriculum Learning

Source: <https://ronan.collobert.com/pub/2009_curriculum_icml.pdf>

Use:

- Curriculum learning frames example order as an optimization aid for hard
  non-convex learning.
- SE1 analogy: a teacher can teach through ordering, contrast sets, and
  prerequisite graphs, not only outputs.
- Design consequence: curriculum packets are first-class Eklavya packets.

## Research Refresh: Efficient Small-Model Design

### MobileLLM

Source: <https://arxiv.org/abs/2402.14905>

Use:

- Sub-billion models benefit heavily from architecture choices such as deep-thin
  design, embedding sharing, and grouped-query attention.
- Design consequence: SE1 should not treat 180M to 350M as a shrunken large
  model. It should choose the shape for sub-billion efficiency.

### MiniCPM

Source: <https://arxiv.org/html/2404.06395v1>

Use:

- Small-language-model work emphasizes resource-efficient scaling, wind-tunnel
  experiments, and WSD-style continued training/domain adaptation.
- Design consequence: SE1 should use scout-scale design trials before full
  scale, and should treat WSD-style schedules as relevant for update efficiency.

### BitNet

Source: <https://www.jmlr.org/papers/v26/24-2050.html>

Use:

- Native low-bit training can change latency, memory, throughput, and energy
  economics, not just parameter count.
- Design consequence: BitNet-style low-bit substrate is not SE1's first
  architectural claim, but it should become a later Sutra efficiency axis if
  the SE1 learning geometry works.

### BranchyNet, ACT, And SkipNet

Sources:

- BranchyNet: <https://arxiv.org/abs/1709.01686>
- Adaptive Computation Time: <https://arxiv.org/abs/1603.08983>
- SkipNet: <https://openaccess.thecvf.com/content_ECCV_2018/html/Xin_Wang_SkipNet_Learning_Dynamic_ECCV_2018_paper.html>

Use:

- Early exits, adaptive halting, and dynamic routing all support the principle
  that easy inputs should use less compute than hard inputs.
- ACT's sequence observation is especially relevant: harder-to-predict
  transitions can reveal boundaries.
- Design consequence: SE1 should train the compute controller from patch entropy,
  teacher disagreement, and verifier risk, then report p95 cost and hard-case
  damage separately.

## Research Refresh: Additional Sibling Project Lessons

### Deterministic Knowledge Structure

Source: `../moonshot-deterministic-knowledge-structure/README.md`

Useful idea:

- Once information is committed, identity, provenance, time, merge semantics, and
  conflict classification matter.

Design consequence:

```text
Eklavya_packet_ledger_should_preserve_source_identity_cost_and_revision_history
```

This is not process overhead. It is how we prevent teacher artifacts from
becoming unauditable mush.

### CTI Universal Law

Source: `../moonshot-cti-universal-law/README.md`

Useful idea:

- Representation quality can sometimes be predicted from measurable geometry,
  but the project also records scope breaks and architecture-shift failures.

Design consequence:

```text
Sutra_should_use_geometry_as_a_state_sensor_not_as_universal_unchecked_truth
```

CTI-style sensors may help the SE1 compute planner detect local boundary safety
or representation risk, but any such sensor must carry scope gates.

### Complex Fractal Adapters

Source: `../moonshot-complex-fractal-adapters/README.md`

Useful idea:

- Constrained adapter families can define safe spaces of rewiring moves.

Design consequence:

```text
SE1_surgical_update_should_prefer_constrained_adapter_spaces_over_global_finetuning
```

This belongs after the first retained-gain proof. It is a candidate update
efficiency mechanism, not a first substrate assumption.

## Updated Source Priority After This Pass

Highest priority:

- BLT, MEGABYTE, Charformer for byte-local/patch-global substrate.
- Eklavya teacher tomography for behavior profiles.
- PGO and System R as the compiler/runtime planning analogy.
- Query-by-committee and curriculum learning for disagreement-driven example
  selection.
- BranchyNet, ACT, SkipNet, LayerSkip, Mixture-of-Depths, EAGLE, and Medusa for
  adaptive compute.
- MobileLLM and MiniCPM for sub-billion architecture and scout-scaling habits.

Use carefully:

- BitNet for later low-bit efficiency after the learning geometry works.
- CTI and fractal/geometric projects as sensors and priors, not magic transfer.
- DKS as packet-ledger discipline, not as a process layer.

## Research Refresh: Teaching And Information Economics

### Information Bottleneck

Source: <https://arxiv.org/abs/physics/0004057>

Use:

- The information bottleneck frames learning as finding a compact code for one
  signal that preserves information about another relevant signal.
- SE1 analogy: patches, packets, and teacher signatures should compress away
  irrelevant surface detail while preserving the information needed for retained
  competence.
- Design consequence: a packet should be judged by relevant information retained
  per cost, not by how much teacher data it contains.

### Minimum Description Length

Source: <https://homepages.cwi.nl/~paulv/course-kc/mdlintro.pdf>

Use:

- MDL treats regularity as what allows data to be described more compactly.
- SE1 analogy: Sutra should prefer a teacher packet or module only if it
  shortens the description of future errors, future updates, or future compute.
- Design consequence: packet storage and module complexity are part of the
  description length, not free metadata.

### Value Of Information

Sources:

- Howard summary: <https://www.semanticscholar.org/paper/Information-Value-Theory-Howard/a7b3c2a88ca459d50010a33db8c2f113f1323e0c>
- Graph-theoretic VOI analysis: <https://arxiv.org/pdf/1302.3596>

Use:

- Information has value only relative to a decision and a cost.
- SE1 analogy: a teacher call, probe, verifier run, or geometry measurement is
  worth buying only if it changes a training or inference decision enough to
  justify its cost.
- Design consequence: teacher profiles should be priced by expected net retained
  gain, not by prestige or raw information volume.

### Robust Expected Information Gain

Source: <https://proceedings.mlr.press/v180/go22a.html>

Use:

- Expected information gain can be sensitive to priors and sampling error; robust
  variants account for ambiguity.
- SE1 analogy: teacher disagreement and probe ranking are unstable if the prior
  about student gaps is wrong.
- Design consequence: Eklavya should penalize ambiguous probes and avoid
  over-trusting one noisy disagreement estimate.

### Machine Teaching And Teaching Dimension

Sources:

- Machine teaching overview: <https://arxiv.org/abs/1801.05927>
- Machine teaching as inverse ML: <https://ojs.aaai.org/index.php/AAAI/article/view/9761>
- Teaching dimension for linear learners: <https://www.jmlr.org/papers/volume17/15-630/15-630.pdf>

Use:

- Machine teaching asks for an optimal training set given a learner and a target
  model.
- Teaching dimension makes lesson size a formal object.
- SE1 analogy: Eklavya should construct minimal high-value lessons for Sutra's
  actual learning algorithm, not collect iid examples or teacher outputs.
- Design consequence: curriculum packets and contrast sets should be optimized
  as teaching sequences.

## Legacy Evidence Re-Read: Patch Geometry And Capacity

Sources:

- `../moonshot-sutra-legacy-pre-reset-2026-06-06/legacy_pre_eklavya_reset_2026-06-06/tesla_session_design.md`
- `../moonshot-sutra-legacy-pre-reset-2026-06-06/legacy_pre_eklavya_reset_2026-06-06/CLAUDE.pre_reset.md`
- `../moonshot-sutra-legacy-pre-reset-2026-06-06/legacy_pre_eklavya_reset_2026-06-06/results/probe_eklavya_full_protocol_gate_r1.json`
- `../moonshot-sutra-legacy-pre-reset-2026-06-06/legacy_pre_eklavya_reset_2026-06-06/results/probe_multisource_encoder.json`

Useful evidence:

- The old KD claim failed its evidence gate because precise eval and trace audit
  artifacts were missing.
- A candidate with three teachers and a non-decoder family was still not enough
  to prove Eklavya.
- The BGE state channel was promising, but a later multisource encoder probe
  found high student/BGE CKA and recommended skipping alignment loss.
- Prior architecture reasoning repeatedly identified student capacity and
  byte-global sequence tax as the likely bottleneck.
- Patch size and local decoder burden were treated as first-order design
  variables, not minor tuning.

Design consequence:

```text
SE1_should_price_teacher_signal_against_student_capacity_and_patch_geometry
```

If the student lacks landing capacity, better teachers will not help. If the
student already owns the geometry, alignment packets waste gradient budget.

## Source Priority For Next Pass After Utility Calculus

Highest priority:

- Teaching dimension and machine teaching for minimal lesson construction.
- Value-of-information and robust information gain for probe and teacher
  pricing.
- BLT/MEGABYTE/Charformer plus legacy patch-size evidence for patch ladder
  design.
- MobileLLM/MiniCPM for sub-billion shape and scout-scaling.

Key unresolved source questions:

1. Which exact public datasets best instantiate each SE1 manifest slice without
   creating benchmark sprawl?
2. What is the smallest practical teacher-signature sample that gives stable
   distinct-surface estimates?
3. Can packet value be estimated before training from student-gap and teacher
   complementarity measurements?
4. What loss-gradient budget keeps packet objectives useful without destabilizing
   base CE?
5. Which patch-size ladder should be run first if hardware allows only two fixed
   patch choices?

## Research Refresh: Concrete Scout Sources

This pass converts the source map into a first manifest candidate. These are not
automatic admissions. They are the sources that should be reviewed first because
they map cleanly to SE1 packet claims.

### Common Pile

Sources:

- <https://huggingface.co/common-pile>
- <https://arxiv.org/abs/2506.05209>

Use:

- Preferred base-text source family for SE1 scout slices when license clarity is
  more important than raw web scale.
- The collection is described as public domain and openly licensed text across
  many sources.
- Candidate exact IDs visible from the collection/search surface include
  `common-pile/arxiv_abstracts`, `common-pile/arxiv_papers`,
  `common-pile/caselaw_access_project`, and
  `common-pile/biodiversity_heritage_library`.

Design consequence:

```text
SE1_base_text_should_prefer_licensed_open_sources_before_generic_web_crawl
```

### FineWeb-Edu

Source: <https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu>

Use:

- Candidate educational web-text ablation.
- `sample-10BT` is a practical scout-scale config relative to the full corpus.
- Should not be the default if the Common Pile source family can supply enough
  clean base text for the first scout.

Design consequence:

```text
FineWeb_Edu_is_a_quality_scale_ablation_not_the_default_legal_clean_base
```

### ARC, GSM8K, And OpenBookQA

Sources:

- ARC: <https://huggingface.co/datasets/allenai/ai2_arc>
- ARC license context: <https://registry.opendata.aws/allenai-arc/>
- GSM8K: <https://huggingface.co/datasets/openai/gsm8k>
- OpenBookQA: <https://huggingface.co/datasets/allenai/openbookqa>
- OpenBookQA source license context: <https://github.com/allenai/OpenBookQA/blob/main/LICENSE>

Use:

- `allenai/ai2_arc` gives multiple-choice science candidate surfaces.
- `openai/gsm8k` gives arithmetic reasoning with exact answer checking and an
  MIT license in the dataset card.
- `allenai/openbookqa` is a plausible science QA candidate, but the Hugging Face
  dataset card reports unknown license even though the source repository has an
  Apache 2.0 license. It should not enter training until distribution license
  ambiguity is resolved.

Design consequence:

```text
candidate_reasoning_slice_should_start_with_AI2_ARC_and_GSM8K
```

### PAWS-X, SciFact, And SciTail

Sources:

- PAWS-X repository: <https://github.com/google-research-datasets/paws/tree/master/pawsx>
- PAWS-X paper: <https://aclanthology.org/D19-1382/>
- SciFact: <https://huggingface.co/datasets/allenai/scifact>
- SciTail: <https://huggingface.co/datasets/allenai/scitail>

Use:

- PAWS-X is the cleanest first adversarial paraphrase / structure-preservation
  source because word order changes can break shallow similarity.
- SciFact is useful if the geometry packet needs claim/evidence structure, but
  its current Hugging Face metadata is noncommercial, so it is evidence/eval-only
  by default.
- SciTail is useful if entailment structure is needed, but its current license
  metadata is unspecified, so it is not admitted until license review resolves.

Design consequence:

```text
retrieval_geometry_should_test_structure_preservation_not_only_easy_similarity
```

### HumanEval, APPS, And JSONSchemaBench

Sources:

- HumanEval: <https://huggingface.co/datasets/openai/openai_humaneval>
- HumanEval repository: <https://github.com/openai/human-eval>
- APPS: <https://huggingface.co/datasets/codeparrot/apps>
- JSONSchemaBench: <https://huggingface.co/datasets/epfl-dlab/JSONSchemaBench>
- JSONSchemaBench repository: <https://github.com/guidance-ai/jsonschemabench>

Use:

- `openai/openai_humaneval` and `codeparrot/apps` supply executable code tasks
  and realistic code strings for byte-exactness and verifier slices.
- `epfl-dlab/JSONSchemaBench` supplies real-world JSON schemas for structured
  output and schema-validation tasks.
- These sources should be used with exact local oracles whenever possible,
  rather than with LLM verifier judgment.

Design consequence:

```text
verifier_packets_should_prefer_exact_executable_or_schema_oracles
```

### TinyStories And Dolly

Sources:

- TinyStories: <https://huggingface.co/datasets/roneneldan/TinyStories>
- TinyStories paper: <https://arxiv.org/abs/2305.07759>
- Dolly 15k: <https://huggingface.co/datasets/databricks/databricks-dolly-15k>
- Dolly release context: <https://github.com/databrickslabs/dolly>

Use:

- TinyStories is a diagnostic for small-model coherent generation, not a
  real-world capability proof.
- Dolly-style instruction data can guard against generation collapse, but it
  should not turn SE1 into a chat-tuning project.

Design consequence:

```text
generation_slices_are_guardrails_not_the_core_SE1_claim
```

### Current Teacher Model Candidates

Sources:

- Decoder candidate 4B: (HF model page)
- Control decoder 0.6B: (HF model page)
- Decoder candidate 4B (v3.5): (HF model page)
- Embedding candidate 0.6B: (HF model page)
- Embedding candidate 4B: (HF model page)
- SmolLM3 3B Base: <https://huggingface.co/HuggingFaceTB/SmolLM3-3B-Base>
- BGE-M3: <https://huggingface.co/BAAI/bge-m3>
- BGE small: <https://huggingface.co/BAAI/bge-small-en-v1.5>

Use:

- Primary-family dense models are useful because they provide scale controls across
  0.6B, 1.7B, 4B, and newer 4B-class variants.
- SmolLM3 is useful only if it creates a distinct response surface from the primary family,
  not because another 3B-class decoder looks impressive.
- Primary-family embedding and BGE provide different encoder-teacher candidates, with BGE
  remaining useful as a small/control or multi-function retrieval source.

Design consequence:

```text
teacher_roster_must_be_pinned_by_revision_cost_and_distinct_surface_not_latest_model_name
```

Metadata snapshot from the Hugging Face model API on 2026-06-20:

```text
SE1_hf_model_metadata_snapshot_2026_06_20:
  control-decoder-0.6B:
    license_metadata: apache-2.0
    gated: false
    sha_prefix: c1899de289a0
    last_modified: 2025-07-26

  anchor-decoder-1.7B:
    license_metadata: apache-2.0
    gated: false
    sha_prefix: 70d244cc86cc
    last_modified: 2025-07-26

  decoder-candidate-4B:
    license_metadata: apache-2.0
    gated: false
    sha_prefix: 1cfa9a720891
    last_modified: 2025-07-26

  decoder-candidate-4B-v3.5:
    license_metadata: apache-2.0
    gated: false
    sha_prefix: 851bf6e806ef
    last_modified: 2026-03-02

  embedding-candidate-0.6B:
    license_metadata: apache-2.0
    gated: false
    sha_prefix: 97b0c614be4d
    last_modified: 2026-04-20

  embedding-candidate-4B:
    license_metadata: apache-2.0
    gated: false
    sha_prefix: 5cf2132abc99
    last_modified: 2025-06-20

  HuggingFaceTB/SmolLM3-3B-Base:
    license_metadata: apache-2.0
    gated: false
    sha_prefix: d78a42f79198
    last_modified: 2025-08-14

  BAAI/bge-m3:
    license_metadata: mit
    gated: false
    sha_prefix: 5617a9f61b02
    last_modified: 2024-07-03

  BAAI/bge-small-en-v1.5:
    license_metadata: mit
    gated: false
    sha_prefix: 5c38ec7c405e
    last_modified: 2024-02-22
```

Immediate consequence:

```text
SE1_teacher_metadata_readiness_update:
  local_default_tier:
    - single_24gb_gpu

  default_decoder_candidates:
    - anchor-decoder-1.7B
    - control-decoder-0.6B
    - SmolLM3-3B-Base_if_distinct_surface_gate_passes

  default_encoder_candidates:
    - embedding-candidate-0.6B
    - bge-m3_if_runtime_and_storage_fit
    - bge-small-en-v1.5_as_control

  defer_for_first_local_scout:
    - decoder-candidate-4B
    - decoder-candidate-4B-v3.5
    - embedding-candidate-4B

  still_required:
    - full_revision_pin_not_only_sha_prefix
    - local_memory_fit
    - tokens_per_second_or_examples_per_second
    - cost_per_1000_signature_examples
```

## Source Priority After Concrete Scout Pass

Highest priority before any harness:

- Verify exact dataset licenses and download sizes for the dataset admission
  lock.
- Pick a hardware tier and pin runnable teacher model revisions.
- Confirm whether the v3.5 variant is a better signature teacher than v3 for this
  design, rather than assuming newer is better.
- Build the first scout around Common Pile or other licensed/open sources before
  using broader web-crawl text.
- Use exact verifier oracles for code, arithmetic, and JSON before considering
  LLM verifier teachers.

The important unresolved empirical question is no longer "what could we use?"
It is:

```text
which_small_source_teacher_slice_changes_the_next_SE1_decision_enough_to_pay_for_it
```

## Hugging Face Metadata Snapshot

Checked via the Hugging Face dataset API on 2026-06-20. This is not legal
advice. The admission decisions are recorded in
`research/SE1_DATASET_LICENSE_REVIEW.md`.

```text
SE1_hf_dataset_metadata_snapshot_2026_06_20:
  HuggingFaceFW/fineweb-edu:
    license_metadata: odc-by
    gated: false
    last_modified: 2025-07-11

  openai/gsm8k:
    license_metadata: mit
    gated: false
    last_modified: 2026-03-23

  openai/openai_humaneval:
    license_metadata: mit
    gated: false
    last_modified: 2024-01-04

  codeparrot/apps:
    license_metadata: mit
    gated: false
    last_modified: 2022-10-20

  allenai/ai2_arc:
    license_metadata: cc-by-sa-4.0
    gated: false
    last_modified: 2023-12-21

  allenai/openbookqa:
    license_metadata: unknown
    gated: false
    last_modified: 2024-01-04
    readiness_status: do_not_admit_until_distribution_license_is_resolved

  allenai/scifact:
    license_metadata: cc-by-nc-2.0
    gated: false
    last_modified: 2023-12-21
    readiness_status: noncommercial_metadata_blocks_default_training_use

  allenai/scitail:
    license_metadata: unspecified
    gated: false
    last_modified: 2024-01-04
    readiness_status: do_not_admit_until_license_is_resolved

  epfl-dlab/JSONSchemaBench:
    license_metadata: mit
    gated: false
    last_modified: 2025-03-31

  roneneldan/TinyStories:
    license_metadata: cdla-sharing-1.0
    gated: false
    last_modified: 2024-08-12

  databricks/databricks-dolly-15k:
    license_metadata: cc-by-sa-3.0
    gated: false
    last_modified: 2023-06-30

  common-pile/arxiv_abstracts:
    license_metadata: unspecified_on_dataset_card
    gated: false
    last_modified: 2025-06-06

  common-pile/arxiv_papers:
    license_metadata: unspecified_on_dataset_card
    gated: false
    last_modified: 2025-06-06

  common-pile/caselaw_access_project:
    license_metadata: unspecified_on_dataset_card
    gated: false
    last_modified: 2025-06-06

  common-pile/biodiversity_heritage_library:
    license_metadata: unspecified_on_dataset_card
    gated: false
    last_modified: 2025-06-06
```

Immediate consequence:

```text
SE1_dataset_license_readiness_update:
  default_candidate_after_row_filter_manifest:
    - common-pile/arxiv_abstracts
    - common-pile/caselaw_access_project
    - common-pile/biodiversity_heritage_library

  default_admit_after_size_review:
    - openai/gsm8k
    - openai/openai_humaneval
    - codeparrot/apps
    - epfl-dlab/JSONSchemaBench

  admit_with_sharealike_or_attribution_review:
    - allenai/ai2_arc
    - HuggingFaceFW/fineweb-edu
    - roneneldan/TinyStories
    - databricks/databricks-dolly-15k

  do_not_admit_until_license_resolved:
    - allenai/openbookqa
    - allenai/scitail

  do_not_use_for_default_training_claim:
    - allenai/scifact

  common_pile_action:
    - use_collection_and_paper_as_source_family_evidence
    - create_D0_common_pile_license_histogram_and_sample_manifest_before_download
    - keep_common-pile/arxiv_papers_held_until_license_histogram_exists
```

The manifest should now treat SciFact as evidence/evaluation-only unless a
noncommercial constraint is acceptable for the specific run. OpenBookQA and
SciTail stay in the question set, not the admitted training set. FineWeb-Edu
stays a scale ablation, not the default D0 base text source.
