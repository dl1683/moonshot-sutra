# Sutra Architecture Reference

**Status: Round 1 architecture proposal (2026-03-26)**

This file is the architecture source of truth for Sutra. It is written so a fresh session can read it without prior conversation context.

---

## Ground Truth This Round

- Mandatory context read this round: `research/TESLA_LEIBNIZ_CODEX_PROMPT.md`, `research/VISION.md`, `research/RESEARCH.md`, `CLAUDE.md`, `code/dense_baseline.py`.
- Live hardware check run on **2026-03-26 00:38:58 America/New_York** via `nvidia-smi`:
  - GPU: `0 MiB / 24463 MiB`
  - GPU util: `0%`
  - Only visible process: idle `ollama.exe`
- `results/` contains no active checkpoint directories and no training logs relevant to a live run. This is operationally a clean slate.
- `code/dense_baseline.py` is a useful implementation artifact and probe harness, but it is **not** a binding architecture commitment. It contains a dense decoder baseline plus additive probes for MTP, halting, and n-gram memory.

---

## Round 1 Verdict

The best current path is **not** a pure transformer, pure SSM, or fully learned halting architecture. The strongest design under current evidence is:

1. Keep the validated `16K` tokenizer.
2. Build a **hybrid causal decoder**: mostly gated long-convolution blocks, with periodic attention blocks for exact retrieval and synchronization.
3. Use **fixed exits plus internal self-distillation** first. Do not make learned halting part of the first production run.
4. Make **multi-source learning a first-class training interface**, but start with **representation-first distillation and offline data reweighting**, not full logit KD across tokenizer boundaries.
5. Treat **external n-gram memory** and **TOP-style future-order supervision** as additive modules with delayed activation, not as day-0 requirements for the first scout run.
6. Exploit the repo's strongest process finding: **warm-starting beats scratch**. Build a width-compatible scout model first, then widen into the main model.

The proposed architecture family is:

- **Scout model:** `~101M` GPU-resident params
- **Main model:** `~166M` GPU-resident params
- **Optional CPU memory table:** `~67M` sparse params on system RAM

This is the first architecture in the repo that is explicitly designed around all five outcomes at once instead of optimizing only backbone quality.

---

## 1. Phase A - Assumption Challenges

### A1. The 16K tokenizer should remain fixed

**Obvious interpretation:** The 16K custom BPE was the biggest single efficiency win, so changing it would be self-sabotage.

**Alternative interpretations that also fit the data:**

- The gain may have come from escaping the particularly bad fit of the 50K GPT-2 tokenizer, not from `16K` being globally optimal.
- The real win may be corpus matching, not vocabulary size. A `12K`, `20K`, or unigram tokenizer could still beat `16K`.
- The tokenizer may now be a liability for Outcome 4 because it complicates logit-level distillation from external teachers.

**Strongest argument FOR keeping it:** This is the clearest positive empirical result in the repo. The project context states that the 16K tokenizer recovered **56.5% of dead embedding parameters** relative to the 50K GPT-2 tokenizer. At the 100M-200M scale, embedding waste is not a rounding error; it directly steals capacity from the reasoning trunk.

**Strongest argument AGAINST keeping it:** Every external teacher named in `research/RESEARCH.md` uses a different tokenizer. If Outcome 4 is the biggest gap, a tokenizer that blocks practical KD could become a larger cost than the embedding savings.

**Evidence that would resolve it:** Train matched tiny models with `12K`, `16K`, and `24K` vocabularies on the same shard subset and compare:

- bits-per-token
- tokens-per-character inflation
- logit-distillation bridge complexity
- embedding parameter share

**Current lean:** Keep `16K` for Round 1.

**Confidence:** `8/10`. This is one of the few repo findings that survived every prior architecture reset.

### A2. The student should remain decoder-only and autoregressive

**Obvious interpretation:** Sutra is a language model, so decoder-only causal LM is the correct default.

**Alternative interpretations that also fit the data:**

- A prefix-LM, UL2-style mixed objective, or encoder-decoder setup could be more data-efficient.
- Some of Outcome 4 might benefit from a bidirectional student, not just bidirectional teachers.
- The final deployment task is generation, but the pre-training task need not be purely next-token prediction.

**Strongest argument FOR keeping it:** The evaluation suite, deployment target, and codebase are all generation-centric. Decoder-only LM keeps the architecture compatible with standard benchmark harnesses, fixed exits, and teacher logit/representation transfer from decoder LMs.

**Strongest argument AGAINST keeping it:** Autoregressive-only pre-training may be one reason frontier models need huge data. If the thesis is "better geometry beats scale," it is fair to ask whether pure NTP is itself the inherited bottleneck.

**Evidence that would resolve it:** A matched scout probe comparing decoder-only NTP against a prefix-LM or span-corruption variant at equal compute and equal architecture.

**Current lean:** Keep decoder-only for Round 1.

**Confidence:** `7/10`. The deployment objective and existing harness strongly favor this, but the data-efficiency critique is real.

### A3. The trunk should be hybrid, not pure transformer or pure SSM

**Obvious interpretation:** A hybrid local/global backbone gives the best trade-off between quality and efficiency.

**Alternative interpretations that also fit the data:**

- At `512-1024` context, attention is cheap enough that a pure transformer may still win on simplicity.
- Pure SSM or pure gated-convolution may be more edge-efficient and easier to quantize.
- The benefit in Hyena Edge may depend on the specific search process and billion-scale training, not carry down to our scale.

**Strongest argument FOR hybrid:** `research/RESEARCH.md` cites Hyena Edge results where replacing roughly `2/3` of attention with gated convolutions made the model **better and faster**. Separating local pattern synthesis from global retrieval also serves Outcome 2 because failures can be localized to named subsystems.

**Strongest argument AGAINST hybrid:** There is **no project-local empirical result yet** showing that a hybrid trunk beats the dense control at our scale. A hybrid adds implementation complexity exactly where previous bugs already cost the project time.

**Evidence that would resolve it:** A matched three-way scout comparison:

- pure transformer
- pure gated-convolution
- hybrid `3:1` local/global schedule

same tokenizer, same params, same training tokens, same exits.

**Current lean:** Hybrid.

**Confidence:** `6/10`. Strong literature signal, weak repo-local signal.

### A4. The main model should live around 150M-170M GPU params

**Obvious interpretation:** We should compete near the SmolLM2/Pythia class while keeping room for teacher ports and exits.

**Alternative interpretations that also fit the data:**

- Smaller (`80M-120M`) may be the real sweet spot if memory and teacher signals substitute for raw width.
- Larger (`250M-500M`) may be necessary for multi-source learning to pay off.
- The right capacity metric may not be dense params at all if external memory carries real load.

**Strongest argument FOR this range:** It keeps us in the same rough class as Pythia-160M and SmolLM2-135M while leaving architectural room for exits, ports, and memory. `research/RESEARCH.md` also notes that a `~200M` student leaves enough VRAM headroom for several online teachers on a 24GB GPU.

**Strongest argument AGAINST this range:** SmolLM2-135M had a **2T token** training run. If architecture alone does not close the gap, a 166M model might end up neither cheap enough nor capable enough.

**Evidence that would resolve it:** Scaling scout runs at approximately `100M`, `166M`, and `250M` with the same trunk family, measuring:

- tokens/second
- BPT at matched wall-clock
- teacher-loss utilization
- exit quality

**Current lean:** Build a `~101M` scout and widen to `~166M` main.

**Confidence:** `5/10`. This is still an architectural bet, not a validated result.

### A5. Fixed exits should come before learned halting

**Obvious interpretation:** Learned halting failed before, so fixed exits are the right first mechanism.

**Alternative interpretations that also fit the data:**

- Learned halting may have failed because of gradient design, not because token-level routing is wrong.
- Fixed exits may be too blunt and could obscure the true O5 potential.
- A better learned gate might work only after the exits themselves become strong.

**Strongest argument FOR fixed exits first:** The project states that **elastic compute was replicated across all experiments** and that **post-hoc exit calibration on fixed exits showed compute savings without learned halting overhead**. This is the strongest repo-local Outcome 5 evidence.

**Strongest argument AGAINST fixed exits first:** Fixed exits still spend the same compute on all tokens up to each exit point. That is not the full vision of per-token adaptive computation.

**Evidence that would resolve it:** On the same trained trunk, compare:

- fixed exits only
- fixed exits plus self-distillation
- learned halting initialized from fixed-threshold behavior

**Current lean:** Fixed exits plus internal KD for the first production architecture.

**Confidence:** `8/10` for the first run, `4/10` as the final O5 mechanism.

### A6. External n-gram memory is worth another try

**Obvious interpretation:** Engram-style memory is too aligned with the mission to ignore.

**Alternative interpretations that also fit the data:**

- Small models may not have enough contextual maturity for the gate to open usefully.
- The memory may mostly memorize easy surface forms and add little beyond a good tokenizer.
- Prior failure may have been due to integration timing rather than the module itself.

**Strongest argument FOR retrying it:** `research/RESEARCH.md` reports strong external results for Engram, including CPU-offloadable memory with negligible throughput penalty and gains on ARC, BBH, HumanEval, and long-context retrieval. For Sutra specifically, moving memorization load out of the dense trunk is exactly how we should think about Outcome 4.

**Strongest argument AGAINST retrying it:** The repo already contains a probe implementation and the prior summary says the gate "barely opened." That is a concrete warning, even if not a final verdict.

**Evidence that would resolve it:** A delayed-activation probe:

- no memory
- memory active from step 0
- memory activated only after the base LM stabilizes

Track gate usage, BPT, and exit quality.

**Current lean:** Keep memory in the architecture as an additive, zero-init module gated behind a probe.

**Confidence:** `5/10`.

### A7. Multi-source learning should be representation-first, not logit-first

**Obvious interpretation:** Because of tokenizer mismatch, the cleanest O4 path is representation alignment and data reweighting.

**Alternative interpretations that also fit the data:**

- Logits are still the richest teacher signal, and avoiding them could throw away the most valuable supervision.
- A tokenizer bridge may be cheaper than assumed.
- The best outcome may be mixed: representation-first early, logits later on a hard subset.

**Strongest argument FOR representation-first:** `research/RESEARCH.md` explicitly flags vocabulary mismatch as a critical implementation detail. CKA, contrastive embedding alignment, and MiniPLM-style reweighting avoid that trap while remaining architecture-agnostic.

**Strongest argument AGAINST representation-first:** Representation losses can be too diffuse. If the student never sees teacher token uncertainty, it may miss the sharpest next-token supervision.

**Evidence that would resolve it:** A three-way subset study:

- representation-only KD
- tokenizer-bridged logit KD
- mixed strategy

at equal storage and compute.

**Current lean:** Representation-first, with sparse hard-doc logits as an optional second phase.

**Confidence:** `7/10`.

### A8. TOP-style future-order supervision is a better core auxiliary than classic MTP

**Obvious interpretation:** Small models should use TOP, not standard multi-token heads.

**Alternative interpretations that also fit the data:**

- DeepSeek-style sequential `D=1` MTP might still work if introduced later and trained carefully.
- The student's hybrid trunk may make standard MTP less harmful than in a pure transformer.
- TOP may improve reasoning-like benchmarks but do little for generative quality.

**Strongest argument FOR TOP:** `research/RESEARCH.md` says standard MTP hurts small models, while TOP improved even `340M` models with negligible overhead. Sutra's prior MTP `D=1` attempt also lost to control.

**Strongest argument AGAINST TOP:** TOP is still literature-only for this repo. It also does not directly support speculative decoding the way MTP can.

**Evidence that would resolve it:** Matched scout probe:

- NTP only
- NTP + TOP
- NTP + sequential `D=1` MTP

same trunk, same teacher schedule.

**Current lean:** TOP as the default planning auxiliary, MTP as a later O5 probe.

**Confidence:** `6/10`.

### A9. AdamW plus WSD should remain the default optimizer/schedule

**Obvious interpretation:** Use what the repo has already seen work.

**Alternative interpretations that also fit the data:**

- AdamW may be the main source of quantization-hostile outliers.
- Muon or another rotation-invariant optimizer could fit the mission better.
- The right answer may be optimizer splitting by parameter type.

**Strongest argument FOR AdamW plus WSD:** The project explicitly lists **WSD learning rate schedule worked well** and the dense baseline/trainer already uses `AdamW(betas=(0.9, 0.95), wd=0.1)` successfully enough to establish a process baseline.

**Strongest argument AGAINST AdamW plus WSD:** `research/RESEARCH.md` cites OSP results where optimizer choice was a major outlier source. If we care about NVFP4 deployment, optimizer-induced outliers matter.

**Evidence that would resolve it:** A small matched optimizer/norm probe measuring:

- BPT
- activation kurtosis
- INT4 or FP4 PTQ degradation

for AdamW, Muon, and split-optimizer setups.

**Current lean:** AdamW plus WSD for the first full run.

**Confidence:** `8/10` for trainability, `4/10` for quantization-optimality.

### A10. Quantization-native should mean architecture-friendly design plus late QAT, not BitNet from step 0

**Obvious interpretation:** Avoid overcommitting to ternary training before the student can even beat a dense baseline.

**Alternative interpretations that also fit the data:**

- Ternary training from the start may be the true deployment breakthrough.
- The capacity tax may be acceptable if memory/exits/teachers offset it.
- Late QAT may be too little, too late if the architecture learns outlier-heavy internal structure.

**Strongest argument FOR late QAT:** There is no project-local evidence yet that BitNet-style training is stable or worthwhile at our scale. Late QAT is additive, reversible, and compatible with the existing BF16 trainer path.

**Strongest argument AGAINST late QAT:** If the backbone forms outlier channels early, later QAT may only partially repair the problem.

**Evidence that would resolve it:** Compare:

- BF16 train then QAT
- BF16 with outlier-safe norms from step 0
- ternary or ultra-low-bit training from step 0

at scout scale.

**Current lean:** Design for NVFP4 friendliness now, postpone BitNet commitment.

**Confidence:** `7/10`.

### A11. The trunk should stay real-valued Euclidean, with geometry pushed into a semantic side-port

**Obvious interpretation:** Do not make the whole backbone hyperbolic on Round 1.

**Alternative interpretations that also fit the data:**

- The project thesis explicitly says better geometry beats scale, so a Euclidean trunk may be too conservative.
- A mixed-curvature trunk could outperform a Euclidean trunk with only modest overhead.
- Hyperbolic value may only appear if it is integrated deeply, not as a side-head.

**Strongest argument FOR a Euclidean trunk:** Stable kernels, simpler warm-start widening, easy integration with existing PyTorch LM code, and lower implementation risk.

**Strongest argument AGAINST a Euclidean trunk:** `research/RESEARCH.md` reports hyperbolic gains and dimension compression. If geometry is the mission, delaying manifold structure to a side-port may underuse the central thesis.

**Evidence that would resolve it:** Add a hyperbolic semantic head first. If that improves semantic alignment and downstream quality, then consider promoting geometry deeper into the trunk.

**Current lean:** Euclidean trunk, hyperbolic semantic port.

**Confidence:** `5/10`.

### A12. The architecture must expose named module interfaces, not anonymous repeated layers

**Obvious interpretation:** Outcome 2 and Outcome 3 demand modularity.

**Alternative interpretations that also fit the data:**

- Overly explicit module boundaries can hurt raw model quality.
- The real unit of improvement may be adapters or losses, not blocks.
- A simpler monolith with good tooling might beat an overly structured model.

**Strongest argument FOR named interfaces:** The repo's own philosophy is explicit: improve failures surgically and enable Linux-style composability. A model with local mixers, global mixers, memory ports, teacher ports, and exit heads gives contributors real handles.

**Strongest argument AGAINST named interfaces:** Interfaces alone do not create true composability. If everything co-adapts tightly during pre-training, later module swaps may still break behavior.

**Evidence that would resolve it:** After the scout run, attempt module-local improvement:

- freeze trunk, retrain only memory
- freeze trunk, retrain only teacher ports
- replace one local mixer block family

Measure whether improvements compose or destabilize.

**Current lean:** Keep strong interfaces.

**Confidence:** `7/10`.

### A13. Warm-start widening should be part of the architecture plan, not an afterthought

**Obvious interpretation:** The repo already learned that warm-starting beats from-scratch wall-clock, so design for widening from day 1.

**Alternative interpretations that also fit the data:**

- With no surviving checkpoint, warm-starting could become premature optimization.
- A scout model may learn the wrong representation basis and poison the widened model.
- The widening path may add engineering complexity before the core trunk is proven.

**Strongest argument FOR warm-starting:** The project context explicitly states that **warm-starting consistently outperforms from-scratch at equivalent wall-clock**. This is one of the few robust findings that survived the reset.

**Strongest argument AGAINST warm-starting:** If the scout architecture is wrong, warm-starting makes it easier to scale the wrong thing faster.

**Evidence that would resolve it:** Compare:

- `~101M` scout widened to `~166M`
- `~166M` from scratch

at matched wall-clock, not matched steps.

**Current lean:** Make width-compatible scout-to-main widening part of the plan.

**Confidence:** `7/10`.

---

## 2. Phase B - Research Requests for Claude

These are literature and implementation research asks, not training runs.

1. **Tokenizer-bridge research:** Find the best current methods for distilling logits across incompatible tokenizers, especially any method that works without storing full alignments or retraining teachers on the student tokenizer.
2. **Small-model Hyena implementation research:** Collect the most reliable sub-200M language-model implementations of Hyena-style or gated-convolution blocks, including exact filter parameterizations, FFT requirements, and failure modes.
3. **TOP research at small scale:** Pull exact loss formulations and implementation details for Token Order Prediction on `<=500M` models, including whether ranking real future token embeddings is sufficient.
4. **Outlier-safe optimizer/norm research:** Compare Muon, AdamW, DyT, and Single-Scale RMSNorm specifically for small causal LMs trained in BF16 then quantized to INT4 or FP4.
5. **Cross-architecture feature-matching research:** Find the best online CKA or SVCCA recipes for decoder student versus encoder, SSM, and hybrid teachers, with concrete advice on which student layers to align.
6. **Sparse hard-doc KD research:** Investigate whether caching top-K logits only on the hardest `1%-5%` of teacher-selected documents captures most of the benefit of full offline KD.
7. **Hyperbolic side-head research:** Find the lowest-risk way to add Lorentz or Poincare semantic heads to a standard causal LM without destabilizing the trunk.

---

## 3. Phase B - Experiment and Probe Requests for Claude

These are ordered by expected decision value. GPU is currently free, but the first four probes are the most important.

### Probe 1. Trunk Choice Probe

**Question:** Does the hybrid trunk actually beat pure transformer and pure gated-convolution at our scale?

**Method:**

- Build three scout models near `~100M` params with the same tokenizer, same exits, same optimizer, same data subset.
- Variants:
  - pure transformer
  - pure gated-convolution
  - hybrid `H,H,A,H` repeated three times
- Train each for the same wall-clock or same token budget.

**Measure:**

- bits-per-token
- tokens/second
- early-exit quality
- activation kurtosis

**What it tells us:** Whether the central architecture choice is real or wishful.

### Probe 2. Fixed Exit Plus Self-Distillation Probe

**Question:** Do early exits become materially better if they learn from the final exit directly?

**Method:**

- Use the best scout trunk from Probe 1.
- Compare:
  - exit CE only
  - exit CE plus KL from final logits to shallow exits
- Keep exit positions fixed.

**Measure:**

- `bpt_exit_4`, `bpt_exit_8`, final BPT
- calibrated compute savings at matched quality

**What it tells us:** Whether the first Outcome 5 mechanism should be fixed-exit KD rather than learned halting.

### Probe 3. Delayed Memory Activation Probe

**Question:** Was prior n-gram memory failure caused by integration timing?

**Method:**

- Same scout trunk.
- Three variants:
  - no memory
  - memory active from step 0
  - memory weights present but gates activated only after the base loss plateaus or after a fixed token threshold
- Keep tables on CPU RAM.

**Measure:**

- gate-open fraction
- BPT
- exit quality
- throughput hit

**What it tells us:** Whether external memory belongs in the first full run or in a later stage.

### Probe 4. TOP versus Sequential MTP Probe

**Question:** Which future-prediction auxiliary actually helps a small student?

**Method:**

- Best scout trunk from Probe 1.
- Compare:
  - NTP only
  - NTP + TOP
  - NTP + sequential `D=1` MTP
- Delay the auxiliary until the model reaches basic competence.

**Measure:**

- BPT
- custom hard eval deltas
- training throughput

**What it tells us:** Whether TOP deserves to be in the main production recipe.

### Probe 5. MiniPLM Data Reweighting Probe

**Question:** Can offline difficulty weighting deliver immediate Outcome 4 gains with almost no architectural risk?

**Method:**

- Use `Qwen3-1.7B` as teacher and `Pythia-160M` as reference on a representative shard subset.
- Compute `delta(doc) = NLL_ref(doc) - NLL_teacher(doc)`.
- Sample documents proportionally to `exp(delta / tau)`.
- Train identical scout models on weighted and unweighted data.

**Measure:**

- BPT at matched steps and matched wall-clock
- loss curve slope in the first `1k-5k` steps

**What it tells us:** Whether the easiest O4 lever is already enough to justify inclusion in the first run.

### Probe 6. Representation-First Multi-Teacher Probe

**Question:** Can the student absorb useful structure from diverse teachers without logit KD?

**Method:**

- Use online rotated teachers:
  - `Qwen3-0.6B`
  - `Mamba-790M`
  - `Granite-4.0-Micro`
  - `EmbeddingGemma-300M`
  - `CodeBERT` on code-only batches
- Compare NTP baseline versus NTP plus teacher ports with pooled-state alignment.

**Measure:**

- BPT
- custom eval deltas by category
- gradient conflict between teacher losses and NTP

**What it tells us:** Whether representation-first multi-source learning is practical on 24GB.

### Probe 7. Optimizer/Norm/Quantization Probe

**Question:** Are we leaving too much on the table by defaulting to AdamW plus standard RMS-like normalization?

**Method:**

- Tiny `20M-40M` models only.
- Compare:
  - AdamW + RMSNorm
  - AdamW + Single-Scale RMSNorm
  - AdamW + DyT
  - Muon + Single-Scale RMSNorm

**Measure:**

- BPT
- activation kurtosis
- PTQ degradation to INT4 or FP4

**What it tells us:** Whether the Round 1 optimizer/norm default should change before a long run.

### Probe 8. Warm-Start Widening Probe

**Question:** Does widening a good scout beat training the main model from scratch at equal wall-clock?

**Method:**

- Train scout to a modest checkpoint.
- Widen to main via Net2Wider-style channel duplication.
- Compare against main-from-scratch for equal total wall-clock.

**Measure:**

- BPT
- custom eval
- stability after widening

**What it tells us:** Whether warm-starting should be mandatory in the main pipeline or just optional.

---

## 4. Phase B.1 - Multi-Source Learning Proposals

Outcome 4 is the weakest pillar. The architecture below only makes sense if these items are treated as core, not future work.

### B1. Offline Difficulty Reweighting Is Mandatory

Use a MiniPLM-style sampler before any major training run.

- Teacher: `Qwen3-1.7B`
- Reference: `Pythia-160M`
- Raw-text difficulty score:

`delta(doc) = NLL_ref(doc) - NLL_teacher(doc)`

- Sampling weight:

`p(doc) propto exp(delta(doc) / tau)`

with `tau` tuned on a subset.

**Why this is the correct first O4 step:** It is architecture-independent, cheap in storage, avoids tokenizer mismatch, and directly targets the biggest gap: our limited raw-token budget.

### B2. Universal Teacher Interface (UTI)

The student should expose small, named latent ports instead of trying to copy whole teachers.

- `Port-decoder`: aligned to `Qwen3-0.6B`
- `Port-ssm`: aligned to `Mamba-790M`
- `Port-hybrid`: aligned to `Granite-4.0-Micro`
- `Port-semantic`: aligned to `EmbeddingGemma-300M`
- `Port-code`: aligned to `CodeBERT` on code shards only

Each port reads a pooled student state and projects it into a teacher-specific target space.

This is the mechanism that makes multi-source learning compatible with Outcome 2 and Outcome 3. A contributor can improve one port or add a new one without rewriting the trunk.

### B3. Teacher Rotation Curriculum, Not Full Teacher Parallelism

Do **not** run all teacher losses on every batch at first. That is the easiest way to create gradient conflict.

Use a rotation schedule:

- generic text batches rotate among decoder, SSM, hybrid, and semantic teachers
- code batches always include the code teacher
- teacher losses start after the student has basic next-token competence

This keeps VRAM and gradient interference manageable while still letting the student see diverse inductive biases.

### B4. Hard-Document Sparse Logit Bank

Do not commit to storing top-K logits for the entire `22.9B` token corpus. That likely costs too much storage for unclear marginal value.

Instead:

- select the hardest `1%-5%` of documents according to the difficulty score
- store top-32 or top-64 logits only for those documents
- use them as a second-phase sharpened KD signal

This is the most storage-efficient compromise between "no logits at all" and "12TB logit warehouse."

### B5. Geometry-First Semantic Distillation

Use the hyperbolic semantic port as the place where hierarchical teacher structure is distilled.

- Student trunk stays Euclidean for stability.
- Final pooled state is projected into a low-dimensional Lorentz manifold.
- Semantic teacher losses operate there using geodesic distance.

This is the cleanest way to make the "Intelligence = Geometry" thesis concrete without rewriting the full backbone.

---

## 5. Per-Outcome Confidence

**Round 1 note:** There is no previous T+L round. These are initial scores grounded in repo evidence only. They are not victory claims.

- **Outcome 1 (Intelligence): `3/10`** - There is no trained Round 1 model yet. Confidence is above zero only because the repo already established three relevant facts: the 16K tokenizer was a major win, warm-starting consistently helped, and elastic depth repeatedly preserved quality surprisingly well. That is enough to justify a concrete architecture, but not enough to trust it.
- **Outcome 2 (Improvability): `6/10`** - The proposed design has explicit module boundaries: local mixer, global mixer, memory port, teacher ports, and exit heads. That is directly aligned with the repo's constitution and better than a monolithic transformer stack. Confidence is not higher because modular interfaces are still a design claim, not an empirical composition result.
- **Outcome 3 (Democratization): `4/10`** - The architecture is intentionally structured so contributors can improve memory, teacher ports, exits, or one mixer family in isolation. That is better than the clean-slate `0/10` state, but there is no tooling, package format, or proof that independently improved modules compose.
- **Outcome 4 (Data Efficiency): `1/10`** - This remains the weakest pillar by far. The repo has a thorough literature survey and many available teacher candidates, but no multi-source learning has been implemented and no Outcome 4 win exists locally. The design only earns `1/10` because it finally has a concrete multi-teacher path instead of a placeholder.
- **Outcome 5 (Inference Efficiency): `5/10`** - This is the strongest pillar after modularity because the repo repeatedly observed that shallower compute remained competitive and fixed exits showed promise under calibration. Confidence is capped because there is still no trained Round 1 model and no final per-token exit policy.

---

## 6. What Would Raise or Lower Confidence

### Outcome 1

**Raise confidence:**

- Hybrid scout beats pure transformer and pure gated-conv at matched compute.
- Early exits preserve quality with self-distillation.
- Custom hard eval shows generation quality above the old dense control.

**Lower confidence:**

- Hybrid scout loses clearly to a simpler dense baseline.
- External memory and teacher ports consistently hurt the base LM.
- Main model cannot absorb auxiliary losses without regression.

### Outcome 2

**Raise confidence:**

- Memory-only or port-only fine-tuning improves targeted behavior without harming general LM metrics.
- Replacing one mixer family leaves the rest of the model stable.

**Lower confidence:**

- Every improvement requires retraining the whole model.
- Module-local changes produce global regressions unpredictably.

### Outcome 3

**Raise confidence:**

- A contributor-style experiment can swap or improve one named subsystem with a clear contract.
- Teacher ports become a real extension mechanism rather than one-off probe code.

**Lower confidence:**

- The named interfaces turn out to be fake boundaries and everything is still tightly entangled.
- Memory, exits, or ports cannot be versioned independently.

### Outcome 4

**Raise confidence:**

- MiniPLM-style reweighting improves convergence on a scout run.
- Representation-first teacher ports improve BPT or hard-eval performance.
- Sparse hard-doc logits add incremental value beyond reweighting.

**Lower confidence:**

- Teacher losses conflict with NTP and reduce final quality.
- Tokenizer mismatch kills practical KD routes.
- The student ignores teacher ports entirely.

### Outcome 5

**Raise confidence:**

- Exit self-distillation sharply improves shallow-exit BPT.
- Calibration yields real compute savings with negligible quality loss.
- QAT to NVFP4 or INT4 preserves most quality on exits and final layer.

**Lower confidence:**

- Exit heads remain weak even after distillation.
- The hybrid trunk provides no real throughput benefit on target hardware.
- Quantization breaks exit ordering or confidence calibration.

---

## 7. Intuitions Worth Testing

### I1. Delayed activation is the missing ingredient for failed additive modules

**Suspicion:** Memory and future-prediction auxiliaries likely failed before because they were forced to learn before the base LM had useful representations.

**Trigger:** The prior project history says MTP, halting, and memory all lost; the dense baseline code also zero-initialized new modules heavily, which suggests concern about cold-start interference.

**Conviction:** `medium`

**Validation path:** Run delayed-activation probes for memory and TOP/MTP versus step-0 activation.

### I2. The biggest O4 gain will come from data selection before it comes from teacher mimicry

**Suspicion:** MiniPLM-style reweighting may outperform more glamorous KD mechanisms in the first month because it attacks corpus inefficiency directly and avoids tokenizer mismatch.

**Trigger:** The repo has zero multi-source wins, storage for full offline logits is huge, and the literature summary makes data reweighting look unusually cheap relative to its upside.

**Conviction:** `medium`

**Validation path:** Weighted versus unweighted scout training on a representative shard subset.

### I3. A small hybrid trunk will benefit more from teacher diversity than from teacher size

**Suspicion:** One transformer, one SSM, and one hybrid teacher will teach more than three transformers of similar size.

**Trigger:** `research/RESEARCH.md` explicitly claims architecture diversity beats teacher size, and the repo's whole mission is about better inductive bias rather than brute scale.

**Conviction:** `medium`

**Validation path:** Compare one-family versus mixed-family teacher rotation on the same scout student.

### I4. Hyperbolic geometry belongs in a side-port first, not the trunk

**Suspicion:** Hierarchical semantic structure is real, but the least risky place to exploit it is the semantic readout, not the whole backbone.

**Trigger:** The geometry literature in the repo is promising, but the codebase has no manifold-aware backbone infrastructure and no project-local evidence yet.

**Conviction:** `medium`

**Validation path:** Add only the hyperbolic semantic head and check whether semantic alignment and hard eval improve.

### I5. Learned halting should return only after fixed exits are already strong

**Suspicion:** The previous halting failure probably came from trying to learn routing and representation quality at the same time.

**Trigger:** The repo simultaneously reports repeated elastic-depth success and specific learned-halting failure.

**Conviction:** `high`

**Validation path:** Train fixed exits first, calibrate them, then initialize a learned gate from the calibrated policy instead of random.

### I6. Sparse hard-doc logit caching may be the practical middle ground for tokenizer-mismatched KD

**Suspicion:** Most of the value of logit KD is concentrated in a small fraction of difficult documents.

**Trigger:** Full-corpus logit storage is huge, yet teacher/reference difficulty scores likely have a long tail.

**Conviction:** `low`

**Validation path:** Cache logits only for the hardest subset and compare against representation-only KD.

---

## 8. Phase C - Design Proposal

I am ready to propose a Round 1 architecture family. Confidence is moderate, not high. The correct next move is:

1. build the scout model first
2. run the trunk, exit, and O4 probes
3. widen into the main model if the scout survives

### 8.1 Architecture Name and Family

**Proposed family name:** `Sutra HEMD`

`HEMD` = **Hybrid Elastic Memory Decoder**

This is a causal decoder with:

- hybrid local/global blocks
- fixed exits
- additive external memory
- teacher ports
- training-only future-order head

### 8.2 Architecture Summary

**Scout (`HEMD-S`):**

- GPU params: `~101M`
- hidden size `d = 768`
- FFN size `ff = 2048`
- query heads `12`
- KV heads `3`
- head dim `64`
- block schedule: `H,H,A,H` repeated `3` times
- exits after blocks `4, 8, 12`

**Main (`HEMD-M`):**

- GPU params: `~166M`
- hidden size `d = 1024`
- FFN size `ff = 2560`
- query heads `16`
- KV heads `4`
- head dim `64`
- same block schedule
- same exits

**Common design choices:**

- tokenizer: custom `16K` BPE
- training context: start `512`, then `1024`
- hard architecture cap: `2048`
- position encoding: RoPE in attention blocks
- number system: real-valued Euclidean trunk, Lorentz hyperbolic semantic side-port
- normalization: `Single-Scale RMSNorm`
- activation: `SwiGLU` in FFN, sigmoid gates in local mixer and memory
- optimizer: AdamW plus WSD for the trunk, higher LR for newly introduced auxiliary modules, SparseAdam for memory tables
- precision: BF16 training, late-stage QAT toward INT8 activations plus NVFP4 or INT4 weights

### 8.3 Why Each Mechanism Exists

- **Hybrid trunk:** Outcome 1 and Outcome 5. Local mixers are cheaper; attention layers preserve exact retrieval.
- **Fixed exits:** Outcome 5. Repo-local evidence already points here.
- **Teacher ports:** Outcome 4, Outcome 2, Outcome 3. Data efficiency without collapsing into one giant KD objective.
- **External n-gram memory:** Outcome 4 and Outcome 5. Move memorization out of the dense trunk.
- **Warm-start width family:** Outcome 1 and Outcome 5. Reuse the repo's strongest process win.
- **Single-Scale RMSNorm:** Outcome 5. Keep RMS-like stability while reducing per-channel scaling.
- **Hyperbolic semantic port:** Outcome 1 and Outcome 4. Test geometry where hierarchy matters most with minimal risk.

### 8.4 Exact Mathematical Formulation

Let `x_1:T` be token IDs, `E in R^{V x d}` be the embedding matrix, and `h_t^0 = sqrt(d) * E[x_t]`.

#### 8.4.1 H-block: local gated-convolution block

For block `l` of type `H`:

`u_t = Norm(h_t^l)`

`a_t = W_a u_t`

`b_t = W_b u_t`

`c_t = W_c u_t`

`f_1:T = Hyena(a_1:T)`

`m_t = W_o (f_t * sigmoid(b_t) + c_t)`

`r_t = h_t^l + m_t`

`n_t = Norm(r_t)`

`ff_t = W_down (SiLU(W_gate n_t) * (W_up n_t))`

`h_t^{l+1} = r_t + ff_t`

`Hyena(.)` here means a causal long-convolution operator with an implicit filter. In the first implementation, it is acceptable to use an FFT-based causal filter with kernel length `64` and learnable per-channel filter generation.

#### 8.4.2 A-block: grouped-query attention block

For block `l` of type `A`:

`u_t = Norm(h_t^l)`

`Q = W_Q u`

`K = W_K u`

`V = W_V u`

Apply RoPE to `Q` and `K`.

Grouped-query attention:

`Attn(u) = W_O softmax(Q K^T / sqrt(64)) V`

with `16` query heads and `4` KV heads in `HEMD-M`, `12` query heads and `3` KV heads in `HEMD-S`.

Then:

`r_t = h_t^l + Attn(u)_t`

`n_t = Norm(r_t)`

`ff_t = W_down (SiLU(W_gate n_t) * (W_up n_t))`

`h_t^{l+1} = r_t + ff_t`

#### 8.4.3 Normalization

Use **Single-Scale RMSNorm**:

`Norm(x) = g * x / sqrt(mean(x^2) + eps)`

where `g` is one learned scalar per block, not one learned value per channel.

Reason for this choice:

- closer to the repo's working RMS-style trainer than DyT
- less outlier-prone than full per-channel RMSNorm
- simpler than introducing full DyT in the first run

#### 8.4.4 External memory

At selected layers `l in {2, 8}`:

`m2_t = T_bigram[hash(x_{t-1}, x_t)]`

`m3_t = T_trigram[hash(x_{t-2}, x_{t-1}, x_t)]`

`mcat_t = [m2_t ; m3_t]`

`mproj_t = W_mem mcat_t`

`gmem_t = sigmoid(W_gmem Norm(h_t^l) + b_mem)`

`h_t^l <- h_t^l + gmem_t * mproj_t`

Table design for Round 1:

- bigram buckets: `1,048,576`
- trigram buckets: `1,048,576`
- memory dim per table: `32`
- table placement: CPU RAM
- dense projection and gate: GPU

Initialization:

- table weights zero-mean small normal if trained from scratch, or zero for delayed-activation probes
- memory projection `W_mem` zero-init for additive safety
- memory gate bias `b_mem = -2.0` so the module starts mostly closed

#### 8.4.5 Exit heads

Exits after blocks `4`, `8`, and `12`.

For exit `i`:

`p_i(x_{t+1} | x_{<=t}) = softmax(E^T Norm(h_t^{l_i}))`

using tied embeddings.

#### 8.4.6 Pooled teacher states

For an exit state sequence `h_1:T`:

`alpha_t = softmax(w_p^T tanh(W_p h_t))`

`s = sum_t alpha_t h_t`

This pooled state feeds all teacher ports.

#### 8.4.7 Teacher ports

Each teacher family gets a small projection:

- `z_dec = P_dec s`
- `z_ssm = P_ssm s`
- `z_hyb = P_hyb s`
- `z_sem = P_sem s`
- `z_code = P_code s`

Losses:

- semantic port:
  `L_sem = 1 - cos(z_sem, e_sem_teacher)`
- decoder, SSM, hybrid ports:
  `L_f = 1 - CKA(Z_f, T_f)`
  where
  `CKA(A, B) = ||B^T A||_F^2 / (||A^T A||_F * ||B^T B||_F)`
- code port only on code batches

#### 8.4.8 Hyperbolic semantic head

Project the final pooled state into a Lorentz manifold:

`u = P_hyp s_final`

`z_hyp = exp_0^c(u)`

and match teacher semantic geometry using geodesic distance:

`L_hyp = d_L(z_hyp, z_teacher)`

This is an auxiliary semantic head, not part of next-token decoding.
