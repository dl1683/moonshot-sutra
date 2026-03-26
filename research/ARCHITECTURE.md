# Sutra Architecture Reference

**Status: ARCHITECTURE LOCKED (2026-03-26). Pure 24-layer Transformer (Sutra-24A-90M). Hybrid campaign complete — 7 rounds, all variants tested. Now pivoting to RMFD (Routed Multi-Family Distillation) as the main training system.**

This file is the architecture source of truth for Sutra. It is written so a fresh session can read it without prior conversation context.

---

## ARCHITECTURE LOCK DECISION (Round 8, 2026-03-26)

**Locked backbone: Sutra-24A-90M (pure transformer)**

Kill rule applied to all hybrid variants:

| Variant | BPT@5000 | Delta vs Transformer | kurtosis_max | max_act | Pass? |
|---------|----------|---------------------|-------------|---------|-------|
| V1: Transformer (24A) | 4.757 | — | 1.5 | 41.7 | BASELINE |
| V2: P-GQA (24Q) | 4.729 | +0.028 | 1.2 | 28.9 | FAIL (need +0.05) |
| V3: Mixed 8Q+16A | 4.744 | +0.013 | 1.0 | 35.7 | FAIL (need +0.05) |
| R7-P: 6G+18A | 4.686 | +0.071 | 2.0 | 58.6 | FAIL (max_act > 52) |

**Verdict:** All hybrid variants showed better stability but insufficient BPT advantage. P-GQA (V2) had the best stability profile (max_act 28.9 vs 41.7) but only +0.028 BPT. Per the pre-declared kill rule, we lock the simpler architecture.

**Codex R8 reasoning:** "The architecture campaign has found a real hybrid signal, but the entire spread between recent architectural variants is on the order of 0.07-0.18 BPT. That is too small to justify spending Round 8 on more backbone churn when every competitive small model we care about is distilled."

**No further hybrid variants after this.** The main lever is now KD.

### Locked Student Spec

```text
Student: Sutra-24A-90M
Params: ~90.2M
Depth: 24 layers
Width: 512
Heads: 8
FFN: 1536 (SwiGLU)
Tokenizer: custom 16K BPE
Context: 512
Exits: layers 7, 15, 23 (0-indexed)
Norm: RMSNorm
Activation: SwiGLU
Optimizer: AdamW (lr=3e-4, betas=(0.9,0.95), wd=0.1)
Precision: bf16 student, fp32 optimizer
```

### lm-eval Benchmarks at 5K Steps (near-random baseline)

| Benchmark | Transformer (24A) | P-GQA (24Q) | Random |
|-----------|-------------------|-------------|--------|
| ARC-Easy acc | 33.6% | 33.5% | 25% |
| ARC-Easy acc_norm | 32.4% | 31.8% | 25% |
| ARC-Challenge acc | 16.6% | 17.2% | 25% |
| ARC-Challenge acc_norm | 21.8% | 21.7% | 25% |

Both architectures produce identical near-random benchmarks at 5K steps, confirming BPT convergence → benchmark convergence. Architecture is not the bottleneck. KD is.

---

## RMFD: Routed Multi-Family Distillation System (Round 8 Design)

Key elements (from Codex R8 design, ingested into RESEARCH.md §6.4.21):

### 4 Distillation Surfaces
1. **Token surface:** final logits at depth 24 — for decoder-like teachers
2. **State surface:** span-pooled hidden states at depths 8 and 16 — for SSM/hybrid/recurrent teachers
3. **Semantic surface:** pooled final representation at depth 24 — for encoder/embedding teachers
4. **Exit surface:** shallow logits at depths 8 and 16 — for internal self-distillation

### Resident Committee (always loaded, ~5.4GB)
- Qwen3-1.7B-Base (Q8, ~2.0GB) — anchor decoder teacher
- LFM2.5-1.2B-Base (Q8, ~1.2GB) — hybrid architecture diversity
- Mamba2-780M (FP16, ~1.6GB) — SSM architecture diversity
- EmbeddingGemma-300M (FP16, ~0.6GB) — semantic geometry

### Specialist Bursts (loaded for dedicated miniblocks)
- Qwen2.5-Math-1.5B — math windows
- RoBERTa-base — discourse/syntax
- CodeGen-350M + CodeBERT — deferred (no dedicated code corpus)

### Cross-Tokenizer: Byte-Span Bridge
Split raw text into M=16 equal byte spans. For each model, pool hidden states over tokens that intersect each byte span. This gives a common scaffold regardless of tokenizer. Use CKA loss on projected span matrices.

### Gradient Conflict Prevention
PCGrad on three loss buckets: L_base, L_gen, L_aux. Auto-halve KD weights if gradient cosine < -0.15 for 3 consecutive evals.

### 5-Phase Training Curriculum (120K total steps from 5K warm-start)
- Phase 0: Fix checkpoint loading + exit supervision (DONE)
- Phase 1: Steps 5K-5.5K — stabilization, no KD
- Phase 2: Steps 5.5K-10K — single anchor (Qwen3-1.7B only)
- Phase 3: Steps 10K-30K — routed resident committee (4 teachers)
- Phase 4: Steps 30K-95K — + specialist bursts + hard-bank
- Phase 5: Steps 95K-120K — consolidation (decay KD weights)

### First KD Probe (COMPLETED — 3000 steps, head-start only)
1. Control: NTP + exits only → BPT 4.498
2. Single-teacher: + Qwen3-1.7B state + semantic KD → BPT 4.550 (no advantage)
*Original plan (arms 3-4) replaced by 4-arm surface ablation: control vs rep vs logit vs rep+logit*

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

#### 8.4.9 TOP-style future-order head

Use `K = 4`.

`q_t = P_top h_t^{12}`

For actual future tokens `x_{t+1}, ..., x_{t+4}`:

`r_{t,j} = q_t^T E[x_{t+j}] / sqrt(d)`

Target ranking distribution:

`y_j = exp(-j / tau) / sum_k exp(-k / tau)`, with `tau = 1`

Loss:

`L_top = - sum_t sum_{j=1}^4 y_j log softmax(r_t)_j`

This uses the true future tokens already in the batch and adds planning pressure without adding full MTP blocks.

### 8.5 Training Objective

Base LM objective:

`L_ntp = CE(p_12, x_{t+1})`

Exit losses:

`L_exit = 0.25 * CE(p_4, x_{t+1}) + 0.50 * CE(p_8, x_{t+1})`

Internal self-distillation:

`L_sd = 0.10 * KL(stopgrad(p_12 / T) || p_4 / T) + 0.10 * KL(stopgrad(p_12 / T) || p_8 / T)`

with `T = 2`.

Teacher loss:

`L_teacher = sum_f lambda_f * m_f(batch) * L_f`

where `m_f(batch)` is the rotation mask selecting which teacher families are active on the current batch.

Future-order loss:

`L_aux = lambda_top * L_top`

Total:

`L_total = L_ntp + L_exit + L_sd + L_teacher + L_aux + lambda_hyp * L_hyp`

Recommended Round 1 initial weights:

- `lambda_top = 0.05` after delayed activation
- `lambda_hyp = 0.02`
- teacher port weights:
  - decoder `0.05`
  - SSM `0.05`
  - hybrid `0.04`
  - semantic `0.03`
  - code `0.05` on code batches only

### 8.6 Training Curriculum

This curriculum is part of the architecture. It exists to protect the trunk from the same cold-start interference that likely hurt earlier probes.

1. **Stage 0: offline data shaping**
   - Run MiniPLM-style difficulty scoring.
   - Build weighted sampler before long training.

2. **Stage 1: scout trunk only**
   - Train `HEMD-S`.
   - Active losses: `L_ntp + L_exit + L_sd`
   - Context `512`
   - No teacher ports, no TOP, no memory

3. **Stage 2: teacher ports on scout**
   - Turn on teacher rotation after the scout has basic competence.
   - Keep context `512`

4. **Stage 3: delayed TOP and memory**
   - Activate `L_top`
   - Run memory-gated variant if Probe 3 is positive
   - Increase context to `1024` if throughput is still acceptable

5. **Stage 4: widen to main**
   - Net2Wider-style width expansion from scout to main
   - Resume training on weighted data with active ports

6. **Stage 5: late quantization-aware fine-tuning**
   - INT8 activations, NVFP4 or INT4 weights
   - recalibrate exits after quantization

### 8.7 Optimizer, Schedule, Initialization, and Precision

**Backbone optimizer:** AdamW

- LR `3e-4`
- betas `(0.9, 0.95)`
- weight decay `0.1`
- gradient clip `1.0`

**Schedule:** WSD

- warmup: first `2%` of total steps, or `500` steps minimum
- stable plateau until `80%` of total steps
- linear decay to `1e-5` in the final `20%`

**Auxiliary modules:** higher LR than backbone

- exits, teacher ports, TOP head: `6e-4`
- memory dense projection and gate: `6e-4`
- memory tables: SparseAdam at `6e-4`

**Initialization:**

- embeddings and main linear layers: normal with std `0.02`
- residual output projections: scale by `1 / sqrt(2L)` where `L = 12`
- memory projection: zero-init
- memory gate bias: `-2.0`
- teacher ports: small normal
- hyperbolic port: small normal

**Precision:**

- train in BF16
- keep optimizer states in FP32
- activation checkpointing on for long runs
- late QAT only after the backbone is already competent

### 8.8 Parameter Breakdown and VRAM Estimate

For `HEMD-M` (`d=1024`, `ff=2560`), approximate GPU-resident parameter counts are:

| Component | Params |
|----------|--------|
| Token embedding / tied unembedding | `16.38M` |
| 9 H-blocks | `109.12M` |
| 3 A-blocks | `31.46M` |
| Exit heads and norms | `0.53M` |
| Teacher ports | `6.29M` |
| TOP head | `1.05M` |
| Memory gate and projection | `1.11M` |
| **Total GPU params** | **`165.94M`** |
| CPU memory tables | **`67.11M`** |

Approximate BF16 / AdamW memory footprint for `HEMD-M`:

| Item | Estimate |
|------|----------|
| BF16 weights | `~0.31 GB` |
| BF16 gradients | `~0.31 GB` |
| FP32 Adam states | `~1.24 GB` |
| Activations at seq `512`, microbatch `8`, checkpointing on | `~3-5 GB` |
| Activations at seq `1024`, microbatch `4`, checkpointing on | `~5-8 GB` |
| CPU memory tables in BF16 | `~0.125 GB` |

Practical interpretation:

- The student should fit comfortably on the 24GB GPU.
- There is still room for rotated online teachers.
- This is consistent with the broader VRAM analysis in `research/RESEARCH.md`.

### 8.9 Module Contracts for Outcomes 2 and 3

These interfaces are not optional. They are the mechanism for improvability and democratization.

1. **Local mixer interface**
   - input: `B x T x d`
   - output: `B x T x d`
   - swappable family: gated convolution, Hyena variant, or future alternative

2. **Global mixer interface**
   - input: `B x T x d`
   - output: `B x T x d`
   - swappable family: GQA, sparse attention, or later state-space attention hybrid

3. **Memory port**
   - input: tokens and hidden states
   - output: residual delta
   - independently trainable and replaceable

4. **Teacher port**
   - input: pooled state
   - output: teacher-aligned latent
   - adding a new teacher should mean adding one new port, not rewriting the trunk

5. **Exit head**
   - input: hidden state at a fixed depth
   - output: logits and calibration metrics
   - independently calibratable

### 8.10 Current Design Decision Audit

| Decision | Status | Confidence | Evidence |
|----------|--------|------------|----------|
| 16K custom tokenizer | VALIDATED | `8/10` | Project reports it as the biggest single efficiency win |
| Warm-start widening path | VALIDATED AS PROCESS, PROPOSED AS ARCHITECTURE | `7/10` | Project reports warm-starting consistently beats scratch at equal wall-clock |
| Fixed exits before learned halting | VALIDATED FOR FIRST PASS | `8/10` | Elastic compute replicated across all prior experiments; calibrated fixed exits showed promise |
| Hybrid local/global trunk | PROPOSED | `6/10` | Strong literature signal, no project-local proof yet |
| Representation-first multi-source learning | PROPOSED | `7/10` | Architecture-agnostic and avoids tokenizer mismatch; still untested here |
| External n-gram memory | QUESTIONED BUT RETAINED AS MODULE | `5/10` | Strong external signal, prior local attempt weak |
| TOP auxiliary | PROPOSED | `6/10` | Better small-model literature story than classic MTP; untested here |
| AdamW plus WSD | VALIDATED FOR TRAINABILITY | `8/10` | Direct repo evidence |
| Single-Scale RMSNorm | PROPOSED | `5/10` | Quantization rationale strong, local proof absent |
| Late QAT to NVFP4 / INT4 | PROPOSED | `7/10` | Hardware and research fit, no local run yet |
| Hyperbolic semantic side-port | PROPOSED | `5/10` | Geometry thesis aligned, low-risk insertion point |

### 8.11 Training Priorities

Immediate priorities for the next session:

1. Implement `HEMD-S` first, not `HEMD-M`.
2. Run Probes 1, 2, 5, and 6 before any long production pre-train.
3. Only add memory and TOP if Probes 3 and 4 are non-negative.
4. Widen to `HEMD-M` only after the scout proves the trunk and exit recipe.
5. Do not start a learned halting run before fixed exits plus calibration are already strong.

---

## Final Round 1 Position

Round 1 should not end with "need more data and no design." There is enough evidence to choose a direction.

The direction is:

- **base architecture:** hybrid causal decoder
- **compute control:** fixed exits plus self-distillation
- **data efficiency core:** offline data reweighting plus representation-first multi-teacher ports
- **memory:** external n-gram table as delayed additive module
- **scaling path:** `~101M` scout to `~166M` main via warm-start widening

This is the first Sutra architecture that is explicitly optimized for all five outcomes at once, while still respecting the strongest local evidence from the repo.

---

## 9. Round 2 Evidence Integration

Round 2 does **not** overturn HEMD. The hybrid elastic family still looks like the best current system-level bet. What changed is the default training stack and the order of proof. The new evidence is strong enough to demote DyT-as-default, promote Muon plus Single-Scale RMSNorm, and make data shaping more concrete and mandatory.

### 9.1 Executive Delta From Round 1

1. The mainline optimizer stack is no longer pure AdamW. The default is now **Muon for dense matrix weights, AdamW for non-matrix/scalar parameters, and SparseAdam for CPU memory tables**.
2. **Single-Scale RMSNorm** is no longer merely proposed. It is the **default normalization** for the first production scout.
3. **DyT is removed from the mainline recipe.** It survives only as a modified salvage probe with residual scaling or residual gating.
4. Stage 0 is split into **0A quality filtering** and **0B MiniPLM difference sampling**. Data shaping is now mandatory before any long scout run.
5. Teacher supervision is no longer static rotation. It becomes **adaptive family rotation with diversity floors**.
6. **TOP stays in the architecture but remains conditional.** The literature signal is strong enough to keep the head; the local probe is not finished, so it is not yet a mandatory loss.
7. External n-gram memory remains additive, but it drops behind optimizer/norm validation and the O4 data pipeline in execution priority.

### 9.2 Evidence-By-Evidence Analysis

#### Evidence 1. DyT vs RMSNorm local probe (`42M`, `5000` steps)

**Observed result:**

- RMSNorm finished at **`5.08` BPT** versus **`5.83` BPT** for DyT.
- DyT had worse average and max kurtosis and worse generation quality.
- The convergence gap narrowed, but it did **not** close by step `5000`.
- The most useful interpretation is mechanistic: **DyT squashes internal activations, but it does not normalize the residual stream.** The residual path `x + block(DyT(x))` can still accumulate scale.

**Architectural consequence:**

- Round 1's "keep Single-Scale RMSNorm unless DyT proves itself" is now resolved. **DyT as a drop-in replacement is falsified for the mainline at this scale.**
- The first production scout should use **Single-Scale RMSNorm everywhere the Round 1 design used `Norm(.)`**: local blocks, attention blocks, exits, and teacher-port inputs.
- DyT moves to a **research branch only**. The next DyT probe must test one of:
  - `DyT + residual scaling`, e.g. `h^{l+1} = h^l + block(DyT(h^l)) / sqrt(L)`
  - `DyT + learned residual gate`
  - `DyT` only in internal sublayers while keeping residual-stream normalization

**Expected effect:**

- Preserve RMS-like scale control for Outcome 1.
- Remove per-channel gamma amplification as an outlier source for Outcome 5.
- Stop spending mainline budget on a currently losing normalization variant.

#### Evidence 2. TOP vs NTP probe is still in progress

**Observed result:**

- We have a partial trajectory for the NTP-only arm, but **no completed comparison yet**.
- This means Round 2 has **no local causal evidence** that TOP helps or hurts the scout at `42M`.

**Architectural consequence:**

- Do **not** promote TOP from "conditional additive module" to "mandatory Stage 1 loss" yet.
- Keep the TOP head in the architecture because Evidence 7 is strong and the overhead is tiny.
- Keep the delayed-activation rule from Round 1, but make the graduation criterion explicit:
  - TOP becomes default only if the `NTP+TOP` arm is **non-negative on final BPT** and **non-negative on generation quality / hard eval** at equal compute.
  - If the local result is mixed, TOP stays as an optional research branch rather than contaminating the mainline scout.

**Expected effect:**

- Preserve the upside of TOP without letting incomplete evidence steer the whole scout recipe.

#### Evidence 3. Muon optimizer research

**Observed result:**

- Muon reaches AdamW-level quality at roughly **`52%` of the FLOPs** and around **`2/3` of the steps**.
- The outlier story matters as much as the speed story: Muon prevents "privileged bases" from forming.
- The strongest quantization-relevant result is the combination **Muon + Single-Scale RMSNorm**, which reportedly yields near-Gaussian activations and dramatically lower kurtosis than Adam-based training.

**Architectural consequence:**

- Round 1's default `AdamW + WSD` is superseded. The Round 2 mainline stack is:
  - **Muon** for dense matrix parameters with `ndim >= 2` in the trunk, exits, teacher ports, and TOP head
  - **AdamW** for non-matrix or scalar/vector parameters: norm scales, biases, scalar gates, and any parameter where Muon is not naturally defined
  - **SparseAdam** for CPU-resident n-gram memory tables
- Keep the **WSD-style schedule** for the first local validation run. The point of the Round 2 change is optimizer geometry, not schedule churn.

**Expected effect:**

- Better scout convergence at fixed wall-clock for Outcome 1.
- Lower activation kurtosis and easier PTQ/QAT for Outcome 5.
- A cleaner first production run because optimizer choice now serves both intelligence and deployment.

#### Evidence 4. MiniPLM difference sampling

**Observed result:**

- MiniPLM gives a concrete offline KD recipe with **zero runtime training cost**.
- It is **cross-family**, so it fits Sutra's multi-source requirement better than architecture-specific KD.
- The repo already has the right insertion point: `code/data_loader.py` exposes `score_shards_teacher()` and `ShardedDataset(weight_file=...)`, and `_split_candidates()` already multiplies candidate sampling weights by `importance_weight`.

**Architectural consequence:**

- Round 1's B1 was directionally right but underspecified. Round 2 makes it concrete:
  1. Extend `score_shards_teacher()` so it accepts both a **teacher** and a **reference** model and operates over **`data/shards_16k`**.
  2. For each shard, sample `N` random decoded windows first. A good first pass is `8` windows of `256` tokens each.
  3. Score each raw-text window under the teacher and the reference model using their own tokenizers.
  4. Compute per-window difficulty gap:

     `delta(window) = NLL_ref(window) - NLL_teacher(window)`

  5. Average to shard level:

     `delta_shard = mean_window delta(window)`

  6. Convert to sampling weight:

     `w_shard = clip(exp((delta_shard - median(delta)) / tau), 0.5, 2.0)`

  7. Save the JSON weight file and pass it into `ShardedDataset(weight_file=...)`.
- The current prompt-based quality scoring in `score_shards_teacher()` is still useful, but **that is a quality filter scaffold, not actual MiniPLM difference sampling**.

**Expected effect:**

- A zero-runtime-cost Outcome 4 baseline.
- Better use of the `22.9B` token budget before we spend effort on heavier teacher losses.

#### Evidence 5. SmolLM2 quality-filtering result

**Observed result:**

- SmolLM2's training story says **quality filtering matters more than raw volume growth**.
- This directly attacks Sutra's biggest structural disadvantage: we do **not** have frontier-scale raw token volume.

**Architectural consequence:**

- Stage 0A is now mandatory: **quality filter before difference sampling**.
- The conservative first-pass policy should be:
  - use the existing `quality / informativeness / difficulty` scaffold as a bootstrap filter
  - compute a coarse shard or source score from sampled decoded windows
  - **drop only the bottom `10%`** after manual spot-checking
  - **downweight the next `20%`** to `0.5x`
  - keep the upper `70%` eligible for MiniPLM reweighting
- In other words: **filter first, reweight second**. Do not waste the MiniPLM budget on obviously low-value text.

**Expected effect:**

- More useful gradients per token for Outcome 4.
- Cleaner teacher signals during later adaptive rotation.

#### Evidence 6. Multi-teacher KD survey

**Observed result:**

- Static averaging is the wrong baseline.
- **Adaptive weighting** matters.
- **Teacher diversity matters more than teacher quality**.
- Multi-round aggregation matters more than "turn everything on at once."

**Architectural consequence:**

- Round 1's teacher-port idea survives, but Round 1's simple rotation needs refinement.
- The Round 2 teacher schedule is:
  1. **Teacher families are structural, not just model IDs**: `decoder`, `ssm`, `hybrid`, `semantic`, with `code` conditional on code batches.
  2. **Warmup phase:** uniform rotation while the teacher-enabled stage first comes online.
  3. **Adaptive phase:** every `1000` steps, evaluate each family on the same held-out hard slice and estimate utility from the held-out next-token delta.
  4. Convert utilities to rotation probabilities:

     `p_f propto softmax(u_f / tau)` with a floor `p_min = 0.15`

  5. On generic text batches, activate **exactly one structural family per batch**.
  6. Run the semantic port on a lower-frequency cadence, e.g. every fourth generic batch.
  7. On code batches, always include the code teacher plus one structural family.
  8. If a teacher family is negative for three consecutive evaluation intervals, freeze it and inspect rather than averaging it into the trunk.
- The teacher set itself should remain deliberately diverse: at least one transformer-family teacher, one SSM-family teacher, one hybrid-family teacher, and one semantic encoder.

**Expected effect:**

- Less gradient conflict for Outcome 4.
- A more real Outcome 3 infrastructure story because adding a new teacher family becomes a bounded extension point.

#### Evidence 7. TOP paper

**Observed result:**

- TOP has strong external evidence at `340M`, `1.8B`, and `7B`.
- It is specifically attractive because it avoids the small-model failure mode of classic MTP.
- The overhead is tiny: one extra unembedding-style head.

**Architectural consequence:**

- TOP stays ahead of sequential MTP in the roadmap.
- Sequential MTP is now a fallback research branch, not a peer baseline.
- TOP activation stays **delayed** and **low-weight** until the local probe finishes.

**Expected effect:**

- If the local probe clears the gate, TOP becomes the lowest-risk path to inject planning pressure into small models.

### 9.3 Superseding Design Decisions

This section supersedes the relevant parts of Round 1 Sections `8.6`, `8.7`, and `8.10`.

| Decision | Round 1 Status | Round 2 Status | Round 2 Rule |
|----------|----------------|----------------|--------------|
| `AdamW + WSD` default | Mainline default | **SUPERSEDED** | Use split `Muon + AdamW + SparseAdam`; keep WSD-style schedule until local evidence says otherwise |
| Single-Scale RMSNorm | Proposed | **PROMOTED TO MAINLINE** | Default norm for the first production scout |
| DyT as a mainline contender | Open contender | **DEMOTED** | Do not use as a drop-in norm in the mainline; only test with residual scaling/gating |
| MiniPLM reweighting | Mandatory but underspecified | **PROMOTED AND SPECIFIED** | Run actual teacher-reference difference sampling through the existing shard-weight path |
| Data quality filtering | Implied by literature, not explicit | **NEW MAINLINE REQUIREMENT** | Filter or downweight low-quality shards before MiniPLM |
| Teacher rotation | Static rotation curriculum | **REFINED** | Adaptive family rotation with diversity floors and held-out utility updates |
| TOP auxiliary | Proposed | **CONDITIONAL** | Keep the head; promote only if the running local probe is non-negative |
| External n-gram memory | Retained additive module | **RETAINED BUT DEPRIORITIZED** | Do not spend early scout budget here until the revised optimizer/norm/O4 stack is positive |

### 9.4 Round 2 Mainline Training Curriculum

This is the new first-production scout recipe. It supersedes the Round 1 execution order.

1. **Stage 0A: quality filtering**
   - Score `data/shards_16k`.
   - Drop the bottom decile after spot-checking.
   - Downweight the next-lowest band.

2. **Stage 0B: MiniPLM difference sampling**
   - Run teacher-reference scoring on the filtered pool.
   - Produce shard weights consumed by `ShardedDataset(weight_file=...)`.

3. **Stage 1: scout trunk with exits only**
   - Train `HEMD-S` with **Muon + AdamW + SparseAdam** and **Single-Scale RMSNorm**.
   - Active losses: `L_ntp + L_exit + L_sd`
   - No teachers, no TOP, no memory

4. **Stage 2: adaptive teacher rotation**
   - Turn on the teacher ports only after the scout has basic NTP competence.
   - Use diverse-family adaptive rotation, not naive averaging.

5. **Stage 3: conditional TOP**
   - Activate TOP only if the running local probe is non-negative.
   - Keep delayed activation and low weight.

6. **Stage 4: delayed memory**
   - Run the memory branch only after Stages `0A-3` are non-negative.
   - If memory still does not open its gate or help BPT, cut it quickly.

7. **Stage 5: widen to `HEMD-M`**
   - Widen only after the revised scout proves trunk quality and stable exits.

8. **Stage 6: quantization validation / late QAT**
   - Evaluate PTQ first on the revised outlier-safe stack.
   - Run late QAT only if PTQ leaves meaningful quality on the table.

### 9.5 Updated Per-Outcome Confidence

**Round 2 note:** These scores are still far from the `>=9/10` target. The point of Round 2 is not to pretend victory; it is to raise confidence only where new evidence is genuinely load-bearing.

| Outcome | Round 1 | Round 2 | Change | Evidence Driving Change | Why This Changed |
|---------|---------|---------|--------|-------------------------|------------------|
| Genuine Intelligence | `3/10` | `4/10` | `+1` | **1, 3** | Evidence 1 removes a locally failing norm path from the mainline. Evidence 3 gives a stronger training stack with better convergence and outlier behavior. This raises confidence that the scout can become smart, but it is still not direct benchmark evidence. |
| Improvability | `6/10` | `6/10` | `0` | none | Round 2 refines the interfaces, but we still have no local evidence that module-local repairs compose cleanly. |
| Democratized Development | `4/10` | `5/10` | `+1` | **4, 6** | Evidence 4 makes cross-family offline supervision concrete through an existing hook, and Evidence 6 shows that diverse teacher families are the right unit of extension. That makes the infrastructure story more real. |
| Data Efficiency | `1/10` | `3/10` | `+2` | **4, 5** | Evidence 4 gives a directly implementable MiniPLM path, and Evidence 5 says quality filtering is a first-order lever. The architecture now has a concrete O4 stack instead of a placeholder. |
| Inference Efficiency | `5/10` | `6/10` | `+1` | **1, 3** | Evidence 1 supports a norm choice with better residual-scale control than DyT, and Evidence 3 makes the quantization path stronger by preventing outliers during training. This helps O5, but exits and token-level stopping are still not newly validated. |

### 9.6 Updated Probe Priorities

The Round 1 probe list remains useful, but the order changes materially.

1. **Highest priority: revised optimizer/norm probe**
   - Compare:
     - `AdamW + RMSNorm`
     - `AdamW + Single-Scale RMSNorm`
     - `Muon + Single-Scale RMSNorm`
   - Measure `BPT`, activation kurtosis, max activation, generation quality, and `INT4 / NVFP4` PTQ degradation.
   - This is now the gating probe for both Outcome 1 and Outcome 5.

2. **Highest priority alongside it: data-shaping probe**
   - Split old Probe 5 into:
     - `quality filter only`
     - `MiniPLM only`
     - `quality filter + MiniPLM`
   - The combined variant is the real Round 2 mainline candidate.

3. **Finish the running TOP vs NTP probe**
   - Do not rerun sequential MTP unless TOP is clearly negative.
   - The local TOP result is now a gate, not a curiosity.

4. **Adaptive teacher-rotation probe**
   - Compare:
     - diverse static rotation
     - diverse adaptive rotation
     - naive parallel teacher losses
   - This is the key Round 2 probe for Outcome 4 and Outcome 3.

5. **Delayed memory activation probe**
   - Keep it, but move it later.
   - Memory should not consume early scout budget before the revised O4 pipeline is validated.

6. **Warm-start widening probe**
   - Still important, but only after the revised scout recipe proves stable.

### 9.7 New Probes Suggested By Round 2

#### Probe 9. DyT Salvage Probe

Compare:

- `Single-Scale RMSNorm`
- `DyT` drop-in
- `DyT + residual scaling`
- `DyT + learned residual gate`

**Purpose:** Determine whether DyT itself is dead for Sutra, or only the naive drop-in formulation is dead.

#### Probe 10. Quality-Filter Ablation

Compare:

- raw corpus
- quality filter only
- MiniPLM only
- quality filter plus MiniPLM

**Purpose:** Measure whether SmolLM2-style filtering is actually a first-order lever on our `22.9B` token pool.

#### Probe 11. Teacher Diversity / Adaptive Rotation Probe

Compare:

- one strongest decoder teacher only
- multiple teachers with static rotation
- multiple teachers with adaptive diverse-family rotation

**Purpose:** Test the claim from Evidence 6 that diversity plus adaptive weighting beats naive quality ranking.

#### Probe 12. Quantized Exit Fidelity Probe

Take the revised scout and measure:

- full-depth PTQ quality
- exit calibration after PTQ
- shallow-exit ordering after PTQ
- throughput on target quantization formats

**Purpose:** Close the biggest remaining O5 gap with local evidence instead of literature extrapolation.

### 9.8 Remaining Gaps To Reach `>=9/10`

Round 2 improves the architecture, but it does **not** yet close the real uncertainties.

- **Outcome 1 gap:** no integrated revised scout beats the dense control yet.
  - **What closes it:** the revised optimizer/norm probe plus a full `HEMD-S` scout run against the dense baseline and the trunk-choice probe.

- **Outcome 2 gap:** no proof that module-local changes improve behavior without collateral damage.
  - **What closes it:** after the scout exists, run module-isolation experiments: teacher-port-only improvement, memory-port-only improvement, and mixer-family replacement.

- **Outcome 3 gap:** the extension story is architectural, not operational.
  - **What closes it:** a contributor-style proof where one new teacher family or one new memory module is added without retraining the whole trunk.

- **Outcome 4 gap:** the new O4 stack is concrete, but it is still literature-backed rather than locally validated.
  - **What closes it:** Probe 10, Probe 11, and a scout run showing that `quality filter + MiniPLM + adaptive diverse teachers` beats unweighted NTP at equal compute.

- **Outcome 5 gap:** no revised-stack exit/quantization result exists yet, and TOP is still unfinished.
  - **What closes it:** finish the TOP probe, run Probe 12, and show calibrated exits remain ordered after quantization.

The main Round 2 conclusion is therefore:

- keep the HEMD family
- change the mainline training stack
- make data shaping concrete and mandatory
- stop treating DyT as a default
- spend the next scout budget on `Muon + Single-Scale RMSNorm + quality filter + MiniPLM` before touching more exotic modules

---

## 10. Round 3 Addendum (2026-03-26)

**Codex T+L session:** `019d28df-dc73-7f61-ae2a-ecb91f2d3d97`
**Full output:** `results/tl_round3_output.md`

Round 3 incorporates new probe evidence (DyT, TOP, Muon optimizer, MTP, halting) and field research (Falcon-H1, Hymba, NorMuon, MiniPLM). Several Round 2 decisions are **superseded**.

### 10.1 What Round 3 Supersedes From Round 2

| Round 2 Decision | Round 3 Verdict | Reason |
|---|---|---|
| Muon as default optimizer | **AdamW is default.** Muon demoted to side probe. | Muon at lr=0.02 lost by +0.10 BPT at 42M with 2.7x larger max activations and periodic instability at steps 2000, 3500-4000. |
| Inter-layer hybrid (alternating block types) | **Intra-layer parallel hybrid** (attention + conv heads WITHIN each block). | March 2026 evidence: Falcon-H1 and Hymba both use parallel attn+SSM inside every block. Systematic analysis says intra-layer > inter-layer. |
| 42M as the decision scale | **Two-scale protocol.** 42M = catastrophic screen. 100M = promotion gate. | TOP, MTP, halting all failed at 42M but literature validates them at 200M+. Capacity-threshold effects are real. |
| Day-1 memory, teacher ports, planning losses | **Brutally simple first scout.** No memory, no teacher ports, no online KD. | Every extra online loss hurt locally. Complexity before base stability = probe failure. |
| `Muon + SS-RMSNorm + quality filter + MiniPLM` | **AdamW + SS-RMSNorm + quality filter + MiniPLM.** Only optimizer changed. | SS-RMSNorm validated (+0.08 BPT), but Muon did not. |

### 10.2 Assumption Audit

1. **42M is the right probe scale.** Confirmed as FLOOR, not ceiling. Catastrophic failures (DyT +0.75, TOP +4.61 with kurtosis 99.3) stay dead. Mild negatives (Muon +0.10) get one 100M retry. Confidence: 9/10.

2. **All failed auxiliaries are universally dead.** Nuanced: catastrophic 42M failures = dead for scout. Mild negatives = retest at 100M. TOP, classic MTP, and learned halting are dead for scout mainline. Confidence: 8/10.

3. **Inter-layer hybrid is the right hybrid.** **Superseded.** Pivot to intra-layer parallel hybrid. The live inter-layer trunk probe serves only as a lower bound for "does mixing help at all." Confidence: 7/10.

4. **Muon should be the scout optimizer.** **Superseded.** AdamW + SS-RMSNorm is the default. Muon at lower LR (0.01, 0.005) and NorMuon are 100M side probes only. Confidence: 8/10.

5. **Day-1 extras (memory, teachers, planning).** **Deferred.** First scout = data shaping offline, plain NTP online, fixed exits, nothing else. Confidence: 9/10.

### 10.3 HEMD-R3-S Scout Specification

**Name:** HEMD-R3-S (Hybrid Elastic Memory Decoder, Round 3 Scout)
**NOT Round 2 HEMD.** It is an intra-layer parallel-hybrid decoder with fixed exits and no day-1 extras.

#### Architecture

| Parameter | Value | Rationale |
|---|---|---|
| Blocks | 14 | Slightly deeper than old 12-block scout, stays near 100M budget |
| Hidden dim | 768 | Multiples of 64, fits 100M budget |
| FFN dim | 2048 | ~2.67x hidden dim (SwiGLU) |
| Tied embeddings | Yes | Parameter savings at this scale |
| Fixed exits | After blocks 5, 10, 14 | Early/mid/final |
| Tokenizer | 16K (existing) | Already validated |
| Context | 512 first, then 1024 | Start short, extend after base loss stable |

#### Parallel Hybrid Block (per block)

```text
u = g * h / sqrt(mean(h^2) + eps)                    # SS-RMSNorm (pre-norm)
a = Attn(Wa u)                                        # d_att=256, GQA 4Q/2KV, head_dim=64
c = ConvOut(DepthwiseConv1D(Wv u) * sigmoid(Wg u))   # d_conv=512, kernel=64
r = h + Wmix[a ; c]                                   # concat then project
n = g' * r / sqrt(mean(r^2) + eps)                    # SS-RMSNorm (post-norm)
h_next = r + Wdown(silu(Wgate n) * Wup n)            # SwiGLU FFN
```

Keeps the Falcon-H1/Hymba principle but uses causal depthwise conv1d instead of Mamba for implementation realism (GatedConv code already exists in repo).

#### Number System

| Choice | Value |
|---|---|
| Activations | Real-valued (no complex, no hyperbolic) |
| FFN | SwiGLU |
| Norm | SS-RMSNorm everywhere |
| Training precision | BF16 |
| Optimizer states | FP32 |
| **Excluded** | Complex states, hyperbolic geometry, DyT |

#### Optimizer

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| LR | 3e-4 |
| Betas | (0.9, 0.95) |
| Weight decay | 0.1 |
| Schedule | WSD (warmup-stable-decay) |
| Warmup | 200 steps |
| Min LR | 1e-5 |
| Gradient clip | 1.0 |
| **Side probes (not default)** | Muon lr=0.01, Muon lr=0.005, NorMuon |

#### Loss

```
L = CE_14 + 0.2 * CE_5 + 0.35 * CE_10
```

No TOP, no MTP, no learned halting, no memory, no online KD in the first scout. Self-distillation only as a follow-up if exit quality is weak.

#### Data Pipeline

| Stage | Description |
|---|---|
| 0A: Quality filter | Drop bottom 10% of shards (spot-checked), half-weight next 20% |
| 0B: MiniPLM | Qwen3-1.7B teacher, Qwen3-0.6B reference, score raw text windows, feed weights into shard-weight path |
| **Excluded day-1** | Online teacher losses, multi-source learning |

Multi-source learning enters only AFTER plain scout is positive. First extension: one decoder teacher + one semantic embedding teacher, one family per batch.

### 10.4 Probe Scale Policy

```
42M / 5K steps  →  catastrophic screen (kills obviously bad ideas)
100M / 5-10K steps  →  promotion gate (validates capacity-sensitive mechanisms)
```

- Catastrophic 42M failures (TOP, DyT) stay dead permanently.
- Mild 42M negatives (Muon) get exactly one 100M retry.
- Capacity-sensitive mechanisms (MTP, halting) wait for 100M scale.

### 10.5 Remaining Probes (ordered by priority)

1. **Finish live trunk-choice probe** (running on GPU): pure transformer vs pure conv vs inter-layer hybrid at 42M. Does NOT decide final hybrid style — only tells us whether non-attention mixing helps at all.

2. **100M intra-layer parallel hybrid vs pure transformer** (next GPU job): Same tokenizer, AdamW, SS-RMSNorm, plain NTP, fixed exits, no memory, no TOP, no KD. THE architectural decision probe.

3. **100M optimizer probe**: AdamW+SS-RMSNorm vs Muon lr=0.01 vs Muon lr=0.005. Promote Muon only if it beats AdamW at final BPT AND keeps max activation under 2x the AdamW baseline.

4. **100M data-shaping probe**: raw data vs quality filter only vs MiniPLM only vs quality filter + MiniPLM. The real O4 gate.

5. **Multi-source probe** (only after plain scout is healthy): pooled-state alignment to one decoder teacher + one embedding teacher, one family per batch.

6. **Quantized exit fidelity probe** (from Round 2, still needed for O5).

### 10.6 Per-Outcome Confidence After Round 3

| Outcome | R2 | R3 | Δ | Key Evidence |
|---|---|---|---|---|
| O1: Intelligence | 4 | 4 | = | Simplified recipe (SS-RMSNorm validated, TOP/Muon rejected), but no integrated scout win yet. |
| O2: Improvability | 6 | 6 | = | Architecture still exposes named modules, but no module-isolation proof. |
| O3: Democratization | 5 | 5 | = | Modular ports + swappable blocks, still architectural intent. |
| O4: Data Efficiency | 3 | 3 | = | MiniPLM + quality filtering concrete but unvalidated locally. |
| O5: Inference Efficiency | 6 | 5 | -1 | TOP killed at 42M, halting negative, Muon not quantization-friendly. Fixed exits are the only live O5 path. |

### 10.7 What Would Change Confidence

**Raise:**
- O1: 100M scout beats best 42M control by ≥0.25 BPT on same eval cache.
- O2: Module-local change improves behavior without retraining trunk.
- O3: Contributor-style extension (new teacher port or block family) works.
- O4: Quality filter + MiniPLM wins at equal compute.
- O5: Fixed exits stay well-ordered after PTQ with real latency savings.

**Lower:**
- O1: 100M scout still can't beat plain transformer control.
- O2: Every improvement requires touching the whole model.
- O4: MiniPLM and quality filtering are neutral or negative locally.
- O5: Fixed exits don't calibrate well after training.

### 10.8 Remaining Gaps to Reach ≥9/10

Round 3 clarified the recipe but did not yet produce wins. The gaps remain structural:

- **O1 gap:** No scout beats a dense baseline. Closes when 100M HEMD-R3-S scout beats 42M control by ≥0.25 BPT.
- **O2 gap:** No module isolation proof. Closes after scout exists and one module is changed successfully.
- **O3 gap:** No operational ecosystem test. Closes with one contributor-style extension.
- **O4 gap:** MiniPLM + quality filter unvalidated locally. Closes with 100M data-shaping probe.
- **O5 gap:** Only fixed exits remain as live path. Closes with quantized exit fidelity probe.

### 10.9 Round 3 Design Intuitions (High Conviction)

1. **Data > losses.** The scout should get most early gains from better data, not clever losses. Every extra online loss has hurt locally; quality filtering and MiniPLM are low-risk and scale well.

2. **Muon's failure is LR/horizon, not conceptual.** It wins hard early, then destabilizes late. NorMuon or lower-LR Muon may fix this at 100M.

3. **Intra-layer hybrid is the right pivot,** but GatedConv is the right first SSM surrogate because it already exists in the repo.

4. **Stop asking the scout to solve O4 and O5 with the same mechanism.** TOP, MTP, and halting all tried to do both and all failed. Keep O4 offline first, O5 as fixed exits first.

---

## 11. Round 4 Addendum (2026-03-26)

**Full output:** `results/tl_round4_output.md`

Round 4 incorporates new evidence: trunk-choice probe V3 (hybrid wins by 0.19 BPT), plus detailed characterization of Falcon-H1, Hymba, NorMuon, MiniPLM, and depth-vs-width tradeoffs.

### 11.1 What Round 4 Supersedes From Round 3

| Round 3 Decision | Round 4 Verdict | Reason |
|---|---|---|
| 14 blocks × dim=768 | **18 blocks × dim=640** | Both Falcon-H1 and Hymba prefer depth. 18×640 is balanced; 14×768 too shallow for hybrids. |
| d_conv=512, kernel=64 | **d_conv=384, kernel=8** | Pure k=64 conv lost by +0.70 BPT. Falcon-H1's d_conv=4 shows short kernels work in successful hybrids. k=8 is the Mamba-2 surrogate. |
| 1:2 attention:conv split (256:512) | **2:3 attention:conv split (256:384)** | Pure conv's large loss shifts ratio toward more attention. 2:3 is a moderate correction. |
| Unspecified fusion method | **Concat-then-project with branch scales** | Mean (Hymba) is simpler but branch widths are asymmetric. Learned fusion is safer. Branch scales: s_a=1.0, s_c≈sqrt(d_a/d_c)=0.816. |
| No muP-style scaling | **Weak muP-inspired branch scales** | Not Falcon-H1's exact constants, but width-based initialization help for branch balance. |
| Muon lr=0.005 as optimizer arm | **NorMuon replaces Muon lr=0.005** | NorMuon directly addresses the observed per-neuron max_act spike. More informative than a second LR point. |
| FFN dim 2048 | **FFN dim 1792** | Adjusted for dim=640 (~2.8x ratio, SwiGLU-appropriate). |
| Exit layers 5/10/14 | **Exit layers 6/12/18** | Adjusted for 18 blocks (early/mid/final). |

### 11.2 HEMD-R4-S Scout Specification (Codex R4 Output — Authoritative)

**Name:** HEMD-R4-S (Hybrid Elastic Memory Decoder, Round 4 Scout)
**Supersedes HEMD-R3-S. Source: `results/tl_round4_output.md`.**

Key R4 pivot vs R3: **depth over width** (24-26×512 instead of 18×640). R4 proposes 1:1 branch ratio and k=16 (pending microprobe confirmation).

#### Architecture

| Parameter | Value | Rationale |
|---|---|---|
| Blocks | 26 (target) / 24 (promotion gate) | Deeper narrow. Falcon-H1, Hymba, and local 12×512 probe all point to depth wins. |
| Hidden dim | 512 | Narrower than R3's 640 — keeps proven width from 42M probes. |
| FFN dim | 1536 | 3x hidden (SwiGLU) |
| Tied embeddings | Yes | Parameter savings at this scale |
| Fixed exits | After blocks 9, 18, 26 (target) / 8, 16, 24 (gate) | Early/mid/final |
| Tokenizer | 16K (existing) | Already validated |
| Context | 512 first, then 1024 | Batch reduction or activation checkpointing for 1024 |

#### Parallel Hybrid Block (per block)

```text
u = g1 * h / sqrt(mean(h^2) + eps)                              # SS-RMSNorm
a = Attn(RoPE(Wq u), RoPE(Wk u), Wv u)                          # d_att=256, 4 Q heads, 2 KV heads, head_dim=64
c = Wco( DWConv1D_k=16(Wcv u) * sigmoid(Wcg u) )               # d_conv=256, depthwise causal gated conv
m = Wmix([ga * a ; gc * c])                                     # scaled concat-project, ga=gc=1.0 init
r = h + m
n = g2 * r / sqrt(mean(r^2) + eps)                              # SS-RMSNorm
h_next = r + Wdown( silu(Wgate n) * Wup n )                     # SwiGLU FFN
```

Key design choices (R4):
- **Concat-then-project** fusion (not mean, not head-interleaved)
- **Branch scales** ga=gc=1.0 init (R4 proposes equal init; 42M microprobes may refine)
- **k=16** (pending microprobe: k=4/k=16/k=64 sweep in progress)
- **1:1 branch ratio** d_attn=d_conv=256 (pending microprobe: 1:1 vs 2:3)
- **GQA 4Q/2KV** for KV-cache efficiency at inference
- **SS-RMSNorm** everywhere (validated in prior probe)

#### Parameter Estimate (~98.6M at 26×512)

Codex R4 estimate: ~98.6M params before tiny scalar/bias terms.
- Embeddings: ~8.2M
- Each hybrid block: ~3.48M
- Total trunk: ~90.4M

At 24×512 (promotion gate): ~91.9M.

Training memory at seq=512, microbatch=16, BF16: ~13-16 GB (confirmed: 10.04 GB at 98.5M with exits).
Seq=1024: requires batch reduction or more accumulation.

#### Init

- All linear/embedding weights: N(0, 0.02)
- Residual-output projections (W_mix, W_down): std = 0.02 / sqrt(2L) where L=26
- Branch scales: ga=gc=1.0 (R4 proposes uniform; current code uses ga=1.0, gc=sqrt(d_a/d_c))
- BF16 forward/backward, FP32 optimizer states

#### Optimizer (unchanged from R3 except NorMuon replaces Muon lr=0.005)

| Parameter | Value |
|---|---|
| Default optimizer | AdamW |
| LR | 3e-4 |
| Betas | (0.9, 0.95) |
| Weight decay | 0.1 |
| Schedule | WSD (warmup-stable-decay) |
| Warmup | 200 steps |
| Min LR | 1e-5 |
| Gradient clip | 1.0 |
| **Side probes** | Muon lr=0.01, NorMuon (β₁=0.95, β₂=0.95) |

#### Loss

```
L = CE_18 + 0.35 * CE_12 + 0.2 * CE_6
```

#### VRAM Budget

- seq=512, microbatch=16, grad_accum=2: ~12-14GB
- seq=1024: ~18-20GB (requires activation checkpointing)

### 11.3 Updated Probe Queue (Round 4 — from Codex R4 output)

1. **42M parallel hybrid probe** (DONE/RUNNING): k=64+mean (BPT=4.9536 ✓), k=4+mean (running).
2. **42M R4 microprobe** (QUEUED): concat-project k=4/k=16/k=64 (1:1 ratio) + concat 2:3 at k=16. Resolves kernel + fusion + ratio.
3. **100M promotion gate**: 24×512 pure transformer vs 24×512 intra-layer hybrid. Success: hybrid wins ≥0.10 BPT with max_act ≤ transformer.
4. **100M optimizer probe**: AdamW vs Muon lr=0.01 vs Muon lr=0.005 vs NorMuon lr=3.6e-4. NorMuon 500-step smoke test first.
5. **MiniPLM reference probe** (10% corpus): Qwen3-1.7B teacher / Qwen3-0.6B ref vs custom ~100M ref. Compare score spread, top-50% overlap, downstream pilot.
6. **O4 multi-source pilot** (after plain scout positive): one decoder teacher + one embedding teacher, one family per batch, pooled-state alignment at 0.05 after step 1000.
7. **Quantized exit fidelity probe** (after 100M scout): PTQ/NVFP4 full-depth quality, exit ordering after PTQ, real latency savings.

### 11.4 Per-Outcome Confidence After Round 4 (from Codex R4)

| Outcome | R3 | R4 | Δ | Key Evidence |
|---|---|---|---|---|
| O1: Intelligence | 4 | 5 | +1 | Trunk-choice probe: hybrid 4.7218 BPT vs transformer 4.9131, lower kurtosis/max_act. Still only 42M/5K steps, not yet intra-layer form. |
| O2: Improvability | 6 | 6 | = | Hybrid pivot helps (cleaner "swap the local branch" interface), but still design intent. |
| O3: Democratization | 5 | 5 | = | Modular story still architectural intent, not proven. |
| O4: Data Efficiency | 3 | 3 | = | MiniPLM/filtering concrete but unvalidated. Cross-tokenizer issue unresolved. |
| O5: Inference Efficiency | 5 | 5 | = | Fixed exits only live path. Healthier activations are good news, but no new exit/PTQ/latency evidence. |

### 11.5 What Would Raise/Lower Confidence (Codex R4)

**Raise:** O1: 100M hybrid beats matched pure transformer + better generations. O2: frozen-trunk module swap improves behavior. O3: new branch/teacher added without full retrain, gain composes. O4: filtering + MiniPLM + multi-source beats raw at equal compute. O5: exits ordered after PTQ + real latency savings.

**Lower:** O1: 100M hybrid loses to transformer. O2: every improvement still requires full model touch. O3: branch/teacher swap breaks model. O4: all O4 pilots neutral/negative. O5: exit collapse, PTQ breaks ordering, poor speculation acceptance.

### 11.5 Round 4 Design Intuitions (Codex R4)

1. **Hybrid win is conservative.** Achieved with plain RMSNorm; SS-RMSNorm (validated separately) should add more. Conviction: medium-high.
2. **Small-k conv will beat k=64 in intra-layer blocks** because inter-layer used k=64 to compensate for infrequent mixing. Conviction: medium.
3. **NorMuon will look mediocre early, separate late** — row-wise normalization targets the exact late-stage spike failure mode. Conviction: medium.
4. **First O4 win: offline MiniPLM + one-family-per-batch representation transfer**, not simultaneous multi-teacher averaging. Conviction: high.
5. **24×512 for gate, 26×512 for target** — spends full budget at proven width. Conviction: medium.

4. **The first O4 win will be offline corpus reshaping + alternating teacher families, not simultaneous multi-teacher online loss.** Local evidence punishes online complexity; purification literature says naive mixing hurts. High conviction.

---

## 12. Round 5 Addendum (2026-03-26)

**Full output:** `results/tl_round5_output.md`

Round 5 incorporates: complete R4 microprobe results (4 concat variants), parallel hybrid probe results (mean fusion k4/k64), deep Hymba/Falcon-H1 branch fusion research, and the finding that mean fusion strictly dominates concat on both quality and stability.

### 12.1 What Round 5 Supersedes From Round 4

| Round 4 Decision | Round 5 Verdict | Reason |
|---|---|---|
| Concat-then-project fusion (C-blocks) | **Normalized additive fusion** | ALL concat variants unstable (kurtosis 1.4-5.8). Mean k4 at 9L beats best concat at 12L. Hymba confirms: per-branch norm + per-channel β is the minimum requirement. |
| k=16 kernel | **k=4 kernel** | k=16 only wins in the unstable concat family. k=4 wins in mean fusion with dramatically better stability. |
| Scalar branch scales (s_a, s_c) | **Per-channel learnable β vectors + per-branch RMSNorm** | Hymba-style. Scalar scales insufficient for magnitude balancing. |
| 2:3 branch ratio option | **1:1 only for now** | 2:3 win was capacity-confounded (+7% params), not ratio-driven. Revisit only after stronger local branch. |
| Gate: 100M, plain concat | **Gate: 100M, normalized additive** | Concat not ready to promote. New block synthesizes mean stability + GQA efficiency. |

### 12.2 HEMD-R5-G Gate Specification (Codex R5 Output — Authoritative)

**Name:** HEMD-R5-G (Hybrid Elastic Memory Decoder, Round 5 Gate)
**Supersedes HEMD-R4-S for the 100M gate. R4-S remains the theoretical target spec pending gate results.**

#### Architecture

| Parameter | Value | Rationale |
|---|---|---|
| Blocks | 24 (gate) | Same as R4's gate proposal |
| Hidden dim | 512 | Proven width from all 42M probes |
| FFN dim | 1536 | 3x hidden (SwiGLU) |
| Tied embeddings | Yes | Parameter savings |
| Fixed exits | After blocks 8, 16, 24 | Early/mid/final |
| Tokenizer | 16K (existing) | Already validated |
| Context | 512 | Gate only; 1024 deferred |

#### Normalized Additive Hybrid Block (HEMD-R5-G)

```text
u  = g1 * h / sqrt(mean(h^2) + eps)                    # SS-RMSNorm

a0 = GQAttn(u)                                          # 4Q/2KV, head_dim=64, raw dim 256
c0 = DWConv1D_k=4(Wv u) * sigmoid(Wg u)                # raw dim 256

a  = beta_a ⊙ Norm_a(Wao a0)                           # project 256->512, per-branch RMSNorm, per-channel β
c  = beta_c ⊙ Norm_c(Wco c0)                           # project 256->512, per-branch RMSNorm, per-channel β

m  = Wo(0.5 * (a + c))                                 # mean fusion + output projection
r  = h + m

n  = g2 * r / sqrt(mean(r^2) + eps)                    # SS-RMSNorm
h' = r + Wdown(silu(Wgate n) * Wup n)                   # SwiGLU FFN
```

Key differences from R4:
- **Normalized additive** instead of concat-project: each branch independently normalized before combining
- **Per-channel β vectors** (not scalar scales): β_a, β_c ∈ R^512, initialized to 1.0
- **Mean fusion (0.5*(a+c))** instead of concat: provides implicit magnitude control
- **k=4** instead of k=16: stable-family winner
- **Each branch projects to full dim** (256→512) via learned projection before normalization

#### Parameter Estimate (~90M at 24×512)

- Embeddings: ~8.2M
- Per block: attention ~0.46M, conv ~0.33M, output proj ~0.26M, FFN ~2.36M, norms/betas negligible
- Total: ~90M

#### Optimizer (unchanged)

AdamW lr=3e-4, betas (0.9, 0.95), wd 0.1, WSD schedule, 200 warmup, 1e-5 min LR, clip 1.0.

#### Loss

```
L = CE_24 + 0.35 * CE_16 + 0.2 * CE_8
```

### 12.3 R5 Microprobe Results Summary

| Rank | Variant | Params | Layers | BPT | kurtosis_max | max_act |
|------|---------|--------|--------|-----|-------------|---------|
| 1 | inter-layer hybrid | 47.0M | 12 | **4.7218** | 1.03 | 19.21 |
| 2 | concat_2to3_k16 | 50.9M | 12 | 4.7713 | 1.9 | 33.1 |
| 3 | mean_k4 (P-block) | 46.0M | 9 | 4.8005 | **0.82** | **18.15** |
| 4 | concat_1to1_k16 | 47.6M | 12 | 4.8320 | 2.0 | 35.2 |
| 5 | concat_1to1_k64 | 47.7M | 12 | 4.8502 | 5.8 | 28.1 |
| 6 | pure_transformer | 49.1M | 12 | 4.9131 | 1.11 | 23.91 |
| 7 | concat_1to1_k4 | 47.5M | 12 | 4.9421 | 1.4 | 26.9 |

Key findings:
- Kernel preference is fusion-dependent: mean→k=4, concat→k=16
- ALL concat variants show instability (kurtosis 1.4-5.8) — fusion-method problem, not kernel
- Mean k4 at 9L matches concat k16 at 12L with 2.4x better stability
- 2:3 ratio helps BPT but is capacity-confounded (+7% params)

### 12.4 R5 Research Findings

**Hymba fusion (from arxiv, exact formula):** `Y = W_out_proj(β₁·norm(attn_out) + β₂·norm(ssm_out))`. Per-channel β vectors + independent per-branch normalization before combining. Element-wise addition, not concat. Motivation: SSM magnitudes consistently larger than attention.

**Falcon-H1 fusion:** SSM:Attention:MLP = 2:1:5. Attention is the smallest at 1/8 of channels. "More attention channels significantly degrades performance." Concat SSM+attention → output proj. No pre-fusion norm (they solve magnitude by making attention tiny).

### 12.5 Probe Queue (Round 5)

1. **100M promotion gate** (NEXT): 24×512 pure transformer vs 24×512 HEMD-R5-G hybrid. Success: hybrid ≥0.10 BPT better, max_act ≤ transformer.
2. **Optional concat sidecar**: patched concat with pre-fusion branch norm, k=16, 1:1 vs 2:3, 2-3K steps. Tests if concat was failing from missing norm or from being the wrong fusion family.
3. **O4 parallel pilot**: MiniPLM 10% corpus scoring with Qwen3-1.7B teacher, Qwen3-0.6B reference. Infrastructure validation.
4. **100M optimizer probe** (after gate): AdamW vs NorMuon on winning architecture.

### 12.6 Per-Outcome Confidence After Round 5 (from Codex R5)

| Outcome | R4 | R5 | Δ | Key Evidence |
|---|---|---|---|---|
| O1: Intelligence | 5 | 5 | = | Microprobe tightened block design but all wins still 42M/5K. Gate fusion changed from raw concat to normalized additive. |
| O2: Improvability | 6 | 6 | = | Branch/exit modularity intact. No frozen-trunk proof yet. |
| O3: Democratization | 5 | 5 | = | Interface clearer but still intent, not evidence. |
| O4: Data Efficiency | 3 | 3 | = | MiniPLM concrete but unvalidated locally. |
| O5: Inference Efficiency | 5 | 5 | = | Fixed exits remain only live path. Better trunk stability, not exit evidence. |

### 12.7 R5 Design Intuitions (Codex R5)

1. **Best gate block is normalized additive with projected branches.** Mean gave stability signal, concat exposed balancing failure. Conviction: high.
2. **2:3 only becomes clearly right after stronger local branch or fixed fusion.** Current 2:3 win is capacity-confounded. Conviction: medium.
3. **First real O4 win will come from data shaping, not online KD.** Every local online auxiliary has hurt. Conviction: high.

## 13. Round 6 Addendum (2026-03-26) — 100M Gate Results

**Full evidence:** `results/tl_r6_evidence.md`, `results/probe_100m_gate_results.json`

### 13.1 100M Gate Verdict: BPT WINS, STABILITY FAILS

| Model | Params | Final BPT | kurtosis_max | max_act |
|-------|--------|-----------|-------------|---------|
| Transformer 24×512 | 90.2M | 4.8087 | 1.66 | 44.8 |
| HEMD-R5-G 24×512 | 93.5M | **4.7019** | 3.94 | 63.6 |

- **BPT: +0.11** — passes ≥0.10 criterion
- **Stability: FAILS** — kurtosis 2.4x worse, max_act 1.4x worse

### 13.2 Trajectory Analysis

The hybrid learns faster early (step 1000-1500: up to +0.73 BPT advantage), but instability grows monotonically. Kurtosis peaks at 5.09 at step 4500 before partial recovery to 3.94 at step 5000. The transformer also shows instability (kurtosis 1.91 at step 4500) but recovers better (1.66 at step 5000).

**Critical observation**: The instability pattern is the same as concat (kurtosis growth in step 3000-4500 window) but less severe. Normalized additive fusion IMPROVED over concat but did NOT solve the fundamental stability problem.

### 13.3 Hypothesis

Per-channel β vectors amplify branch magnitude differences over training. Even with normalized branches, betas can learn extreme scales. This compounds across 24 layers. The P-block (simple 0.5 mean, no learnable betas) was stable at 42M — betas are likely the problem.

### 13.4 MiniPLM Pilot (O4)

200 windows scored on CPU (Qwen3-1.7B teacher, Qwen3-0.6B reference).
- Diff score: 0.264 ± 0.113
- Source ranking: gutenberg (0.37) > wikipedia (0.29) > fineweb (0.28) > math (0.23) > wildchat (0.21)
- Infrastructure validated end-to-end.

### 13.5 R6 Codex Output (2026-03-26) — Stabilization Design

**Full output:** `results/tl_round6_output.md`

**Key decisions:**

1. **Promote hybrid direction, NOT R5-G as-is.** The 100M gate selected the hybrid *family*, not the exact fusion block. The instability must be fixed before production.

2. **Remove β entirely first (R6-F).** The per-channel β vectors are the most likely instability source — they reintroduce unbounded per-channel scale after normalization. First fix: delete them. Backup: softmax mixing (R6-S). Do NOT start with clamp.

3. **Use SS-RMSNorm for branch norms** (not full per-channel RMSNorm). Reduces hidden per-channel gain vectors that compound across 24 layers.

4. **P-GQA discriminant probe.** Test whether projection (256→512) itself is a stability problem by running a projected mean-fusion block with no β and SS-RMSNorm.

5. **Stop gate iteration at O1=7/10**, defined as: stabilized 100M hybrid wins at 5K, stability within ~1.25x transformer on kurtosis/max_act, no late divergence in 8-10K continuation.

6. **Start full corpus MiniPLM scoring NOW** in parallel with stabilization work. Do not train a custom 100M Qwen-tokenizer reference first.

7. **Per-layer telemetry required** before any rerun: branch RMS pre/post norm, fused pre-W_out RMS, norm scale values.

**Confidence scores (R6):**
- O1 Intelligence: 6/10 (up from 5)
- O2 Improvability: 6/10 (unchanged)
- O3 Democratization: 5/10 (unchanged)
- O4 Data Efficiency: 4/10 (up from 3)
- O5 Inference Efficiency: 5/10 (unchanged)

### 13.6 R6-F Production Candidate Block

```text
u  = g1 * h / sqrt(mean(h^2) + eps)        # SS-RMSNorm

a0 = GQAttn(u)                              # d_attn=256, 4Q/2KV, head_dim=64
c0 = DWConv1D_k=4(Wv u) * sigmoid(Wg u)    # d_conv=256

a  = sa * AttnProj(a0) / rms(AttnProj(a0))  # scalar SS-RMSNorm only
c  = sc * ConvProj(c0) / rms(ConvProj(c0))  # scalar SS-RMSNorm only

m  = Wout(0.5 * (a + c))                    # mean fusion, no β
r  = h + m

n  = g2 * r / sqrt(mean(r^2) + eps)
h' = r + Wdown(silu(Wgate n) * Wup n)       # SwiGLU FFN
```

Config: 24×512, FFN 1536, tied 16K embeddings, exits 8/16/24, AdamW lr=3e-4, WSD, BF16.

### 13.7 R6 Stabilization Mini-Gate

**Config:** `code/q4_r6_stabilize.json`

Three-way comparison at 100M scale, 5K steps:
- **Transformer baseline** (24×512, all A blocks) — stability reference
- **R6-F** (24×512, all F blocks) — no β, SS-RMSNorm branches
- **R6-S** (24×512, all S blocks) — softmax mixing, SS-RMSNorm branches

**Success criteria:**
- BPT: ≥ +0.05 vs transformer (half of R5-G's +0.11)
- Stability: kurtosis_max ≤ 2.1, max_act ≤ 52
- No late divergence pattern

### 13.8 R6 Stabilization Gate RESULTS (2026-03-26)

**Full results:** `results/probe_r6_stabilization_gate_results.json`

| Model | Params | Final BPT | kurtosis_max | max_act | Pass? |
|-------|--------|-----------|-------------|---------|-------|
| Transformer | 90.2M | **4.6669** | **1.7** | **41.6** | baseline |
| R6-F (no β) | 93.4M | 4.7768 | 3.5 | 58.5 | **FAIL** |
| R6-S (softmax) | 93.4M | 4.7661 | 2.6 | 55.0 | **FAIL** |

**Both R6 variants fail ALL criteria.** Transformer wins on both BPT and stability.

**Root cause (telemetry):** Branch projections (256→512) accumulate magnitude through depth. Attention a_rms grows 6-8x from layer 0→23. Fused RMS shrinks from 0.36→0.08 as branches diverge in direction. This is structural to the projected dual-branch architecture, not a mixing mechanism problem.

**R6 Intuition updates:**
1. ~~Hybrid family is correct, failure is optimization-level.~~ **FALSIFIED at 100M.** The early advantage (+0.54 at step 2000) is real but unsustainable. Every projected variant collapses by step 5000.
2. ~~BPT gain comes from complementarity, not β.~~ **CONFIRMED but irrelevant.** Removing β didn't help — confirming complementarity is real, but projection amplification destroys it.
3. **Gutenberg/Wikipedia-heavy filtering validated** by 500-window MiniPLM (0.268 diff score, consistent ranking).

### 13.9 T+L Round 7 Decisions (2026-03-26)

**Codex verdict: Projected branches are empirically dead at 100M.**

Key decisions:
1. **Abandon projected branches as the 100M mainline.** Three consecutive variants (N, F, S) failed. The instability is structural — branch projections (256→512) amplify magnitude while branches diverge in direction.
2. **Full-dim P-GQA is the last hybrid test.** GQA attention with full-dim output (512), full-dim gated conv (512), direct 0.5*(a+c) fusion. No projections at all. Param count: 93.4M — same budget as projected variants.
3. **R7-P post-fusion norm is a sidecar only.** Run as a sidecar probe (2.5K first, extend only if ahead). Does not replace the mainline.
4. **Kill rule: if P-GQA AND R7-P both fail → ship pure transformer.** No more hybrid iterations at 100M.
5. **Mixed schedule (8Q+16A) is the fallback** only if P-GQA is stable but not clearly better.

**Confidence scores (R7):**
- O1 (Intelligence): **5/10** (DOWN from 6 — projected hybrid failures reduced evidence)
- O2 (Improvability): 6/10 (unchanged)
- O3 (Democratization): 5/10 (unchanged)
- O4 (Data Efficiency): 4/10 (unchanged)
- O5 (Inference Efficiency): 5/10 (unchanged)

**P-GQA block equation:**
```text
u  = g1 * h / sqrt(mean(h^2) + eps)
a  = GQAttn(u)                                        # q: 512->256, k/v: 512->128, out: 256->512
c  = Wco(DWConv1D_k=4(Wcv u) * sigmoid(Wcg u))       # Wcv,Wcg: 512->512, depthwise conv, out: 512
r  = h + 0.5 * (a + c)                                # direct mean fusion, no projection
h' = r + SwiGLU(g2 * r / rms(r))
```

Params per block: ~3,541K (attn ~393K + conv ~788K + FFN ~2,359K). Model total: ~93.4M.

**New telemetry field:** per-layer cosine similarity between branch outputs (confirms directional divergence).

### 13.10 R7-P Sidecar Results (2026-03-26)

**Config:** `code/q5b_r7p_hybrids_only.json` (block type G = post-fusion SS-RMSNorm after mean fusion of attention + conv branches)

| Variant | BPT@5000 | kurtosis_max | max_act | Params | Verdict |
|---------|----------|-------------|---------|--------|---------|
| Transformer 24A (baseline) | 4.757 | 1.9 | 39.0 | 90M | PASS |
| R7-P 24G (all hybrid) | 4.909 | 2.5 | 75.6 | 112M | **FAIL** |
| R7-P 6G+18A (mixed) | **4.686** | 2.0 | 58.5 | 96M | **PARTIAL** |

**Key findings:**

1. **24G all-hybrid: DEAD.** Post-fusion norm delayed instability by ~1000 steps (kurtosis exceeded 2.1 at step 3500 vs step 2500 for R6) but did NOT prevent it. BPT worse than transformer. Projected branches + conv branches in every layer is fundamentally unstable at depth 24.

2. **6G+18A mixed schedule: FIRST HYBRID TO BEAT TRANSFORMER ON BPT.** This is the most significant hybrid result in the entire campaign:
   - BPT 4.686 vs transformer 4.757 = **+0.071 improvement** (1.5% better)
   - kurtosis_max=2.0, within 2.1 threshold
   - **BUT max_act=58.5, exceeds 52 threshold** (1.12x over limit)
   - Stability trajectory: kurtosis was 1.9 at step 4000, IMPROVED to 1.7 at step 4500 (LR decay helped), then rose to 2.0 at step 5000
   - max_act trajectory: 48.2 → 50.4 → 58.5 (monotonically increasing, concerning)

3. **Architecture insight:** Limiting G-blocks to early layers (positions 1-6) + attention for rest is the WINNING hybrid schedule. Conv branches in early layers capture local patterns efficiently, then hand off to attention for global reasoning. The 6-layer limit is likely close to the stability frontier — 8+ G-blocks would probably blow up.

4. **Open question:** The max_act violation is in the conv branch of early G-blocks. Could be addressable via per-layer activation clipping, smaller conv init, or reducing the conv contribution ratio. The BPT win suggests this is worth investigating IF P-GQA fails.

### 13.11 P-GQA Mainline Probe (IN PROGRESS)

**Config:** `code/q6_pgqa.json` (block type Q = P-GQA full-dim parallel, GQA 4Q/2KV, full-dim gated conv, direct mean fusion, NO projections)

Probe launched 2026-03-26. Three variants:
- `transformer_24x512` (baseline, 24A)
- `pgqa_24x512` (all Q-blocks, 24Q)
- `pgqa_mixed_8q16a` (8Q early + 16A late)

**This is the PRIMARY R7 test.** Kill rule: if P-GQA AND R7-P both fail → ship pure transformer at 100M.
