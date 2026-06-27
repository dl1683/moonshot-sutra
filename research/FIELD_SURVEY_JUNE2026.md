# Field Survey: Cross-Tokenizer KD × Byte-Level Models × Multi-Teacher Distillation
## June 2026

This survey covers the three research threads relevant to Ekalavya E2's goal:
**efficient multi-teacher cross-architecture knowledge distillation.**

The byte-level substrate is our current mechanism for cross-tokenizer alignment,
not the goal itself. The goals are: (1) Eklavya succeeds as a multi-teacher KD
protocol, (2) Sutra becomes the world's most efficient learning mechanism. If a
surveyed approach offers a better path to those goals, we should consider pivoting.

---

## 1. Cross-Tokenizer Knowledge Distillation

The field has exploded in 2025-2026. Six months ago, cross-tokenizer KD was
a niche problem. Now there are 8+ competing approaches.

### BLD — Byte-Level Distillation (Apr 2026)
- **arXiv:** 2604.07466
- **Authors:** Singh, Wu, Cioba, Bernacchia, Buffelli
- **Core idea:** Convert teacher output distribution to byte-level probabilities,
  attach lightweight byte-level decoder head to student, distill through the
  shared byte interface.
- **Scale:** 1B to 8B parameter models.
- **Result:** Competitive with or surpasses more sophisticated CTD methods.
- **RELEVANCE TO E2:** HIGH. BLD proposes exactly our interface choice (bytes)
  for cross-tokenizer alignment. But it is single-teacher only and doesn't
  address multi-teacher routing, disagreement, or cross-architecture issues.
  BLD validates our byte-level interface choice.

### X-Token (May 2026, NVIDIA)
- **arXiv:** 2605.21699
- **Authors:** Sreenivas et al. (NVIDIA)
- **Core idea:** Projection-guided cross-tokenizer KD. Identifies two failure
  modes in existing methods: uncommon-token suppression (critical tokens fall
  into the unmatched subset) and over-conservative matching (strict 1-to-1
  matching excludes near-equivalent tokens).
- **Fix:** P-KL addresses uncommon-token suppression, H-KL relaxes matching.
  Projection matrix W is built rule-based from tokenizer strings.
- **Result:** +3.82 avg points over GOLD baseline on Llama-3.2-1B.
- **Multi-teacher result:** Two-teacher setup (Phi-4-Mini + Llama-3B) improves
  +1.3 points over single-teacher. Static weighting, no routing. Teachers show
  complementarity: Phi-4-Mini contributes math/reasoning, Llama-3B contributes
  commonsense. W projection matrix extends naturally to multiple teachers.
- **RELEVANCE TO E2:** MEDIUM-HIGH (upgraded from MEDIUM). Uses token-level
  projections, not byte-level. Our byte interface sidesteps both failure modes.
  But their multi-teacher extension is now a direct competitor claim-wise. Their
  +1.3 from static two-teacher weighting is our minimum bar — we must beat this
  with our five-teacher routed system to prove routing adds value over averaging.

### SimCT — Simple Cross-Tokenizer OPD (May 2026)
- **arXiv:** 2605.07711
- **Authors:** Sun, Zheng, Song et al.
- **Core idea:** Under heterogeneous tokenizers, exact shared-token matching
  discards teacher signal at vocabulary disagreement positions. SimCT compares
  teacher and student over short multi-token continuations.
- **RELEVANCE TO E2:** LOW-MEDIUM. On-policy distillation focus (student
  generates, teacher scores). Our cached distillation is off-policy.

### Breaking the Tokenizer Barrier (Jun 2026)
- **arXiv:** 2606.09456
- **Authors:** Niu, Xiao, Liu et al.
- **Core idea:** Token-mapping algorithm enabling standard on-policy distillation
  across model families with different tokenizers.
- **Result:** More compute-efficient than baselines on various benchmarks.
- **RELEVANCE TO E2:** LOW. On-policy method. We use cached teacher logits.

### Universal CT Distillation (NeurIPS 2025)
- **arXiv:** 2503.20083
- **Authors:** Minixhofer, Vulić, Ponti
- **Core idea:** Identifies comparable chunks of tokens across tokenizers,
  minimizes likelihood differences between chunks.
- **Result:** First to enable effective subword→byte transfer.
- **RELEVANCE TO E2:** MEDIUM-HIGH. Their chunk-based approach is an
  alternative to our byte-level interface. They explicitly demonstrate
  subword→byte transfer, validating the direction.

### Cross-Tokenizer Likelihood Scoring (Dec 2025)
- **arXiv:** 2512.14954
- **Core idea:** Exploits recursive structure in BPE to compute exact
  next-token likelihoods across different BPE tokenizers.
- **RELEVANCE TO E2:** LOW. Elegant math but BPE-specific. Doesn't apply
  to non-BPE tokenizers (SentencePiece, byte-level).

### DSKD — Dual-Space KD (EMNLP 2024 + extensions)
- **GitHub:** github.com/songmzhang/DSKD
- **Core idea:** KD in both logit space and hidden-state space, with
  key-query matching for vocabulary mismatch.
- **KQ extension (arXiv 2603.22056, Mar 2026):** DSKD-CMA-GA adds Generative
  Adversarial learning to address the mismatched key-query distributions between
  teacher and student cross-attention. Analysis reveals limitations of vanilla
  DSKD-CMA attention patterns.
- **DSKDv2 (arXiv 2504.11426):** Generalizes to a dual-space framework for
  arbitrary teacher-student pairs. GitHub: songmzhang/DSKDv2.
- **RELEVANCE TO E2:** MEDIUM. Our align loss (E1) operates in hidden-state
  space, similar to their dual-space concept. The GAN extension is interesting
  but adds complexity we don't need (our byte interface avoids the distribution
  mismatch problem entirely).

### DWA-KD — Dual-Space Weighting + Time-Warped Alignment (Feb 2026)
- **arXiv:** 2602.21669
- **Authors:** Vu, Chi, Van, Ngo, Dinh, Le
- **Core idea:** Maps teacher representations into student space and vice
  versa, performs dual-space KL. Key innovation: **entropy-based token
  weighting** that up-weights tokens where student is uncertain and teacher
  is confident. Sequence alignment via Soft-DTW.
- **RELEVANCE TO E2:** HIGH. Their entropy-based weighting principle is
  closely related to our gold-free router's student-uncertainty gating
  (U_s * z(D_t) term in A9c). Different granularity (token vs byte position)
  but same insight: uncertain student + confident teacher = high-value
  teaching moment. We should cite for validation of entropy-weighted routing.

---

## 2. Byte-Level Language Models

### Byte Latent Transformer — BLT (Meta, Dec 2024)
- **arXiv:** 2412.09871
- **Architecture:** Three-component: lightweight byte encoder → global
  latent transformer → byte decoder. Dynamic entropy-based patching.
- **Scale:** Up to 8B parameters, 4T training bytes.
- **Result:** Compute-parity with BPE LLMs at scale. Up to 50% inference
  FLOP savings in high-compression regimes.
- **RELEVANCE TO E2:** HIGH. BLT's patch-global architecture is structurally
  similar to our Sutra-Dyad (local encoder → global transformer). BLT uses
  entropy-based patching; we use fixed-size patches (simpler, but less
  adaptive). BLT validates the patch-global paradigm at scale.

### EvaByte (Jan 2025, HKU + SambaNova)
- **Architecture:** Improved byte-level model with multibyte prediction and
  EVA efficient attention mechanism.
- **Scale:** 6.5B parameters, 1.5T bytes.
- **Result:** First open-source byte-level model matching tokenizer-based LMs.
  Excels at coding tasks, 2x faster decoding.
- **RELEVANCE TO E2:** MEDIUM. Proves byte-level can match token-level at
  scale. Our model is much smaller (153M) and targets efficiency, not scale.

### Bolmo — Byteifying LLMs (Allen AI, Dec 2025)
- **arXiv:** 2512.15586
- **Authors:** Minixhofer et al. (same author as tokenkit/cross-tokenizer KD)
- **Core idea:** Two-stage conversion of existing subword models into byte-level.
  Transplants the transformer "brain" (global layers) into a byte-level "body"
  (local encoder/decoder). Avoids training from scratch.
- **Scale:** 1B and 7B parameters. Open source (allenai/bolmo-core).
- **Results:** +16.5% absolute over prior byte LLMs. Competitive with subword
  models — Bolmo 1B: -3.2% MMLU vs source, but +5.1% Lambada, +3.3% CoQA,
  +32.5% CUTE. Inference speed competitive with subword LMs at high
  compression ratios.
- **RELEVANCE TO E2:** HIGH. Bolmo validates the patch-global architecture at
  scale (same structure as BLT and our Sutra-Dyad). Key difference: Bolmo
  converts an EXISTING subword model — we train from scratch. Their approach
  preserves world knowledge cheaply; ours aims to LEARN more efficiently via
  multi-teacher KD. At 153M we're much smaller than their 1B. Their work shows
  the byte-level paradigm is production-ready; our novelty is the learning
  mechanism, not the architecture.

### HoloByte — Continuous Hyperspherical Byte Distillation (Mar 2026)
- **arXiv:** 2603.16917
- **Core idea:** Projects byte chunks into continuous hyperspherical manifold via
  invertible orthogonal rotation. Macro-transformer on compressed continuous
  representations + micro-decoder for byte recovery.
- **Result:** 1.484 nats/byte vs BPE's 1.954 nats/byte under identical compute.
  Reduces attention from O(N²D) to O(N²/W² D + ND²).
- **RELEVANCE TO E2:** LOW-MEDIUM. Very different approach (continuous manifold
  vs discrete patch-global). Interesting theoretically — their lower entropy
  bound suggests continuous representations may beat discrete patches. But
  untested at meaningful scale and no competitive benchmarks yet. Worth
  monitoring but not a direct competitor to our training approach.

### ByteFlow — Adaptive Byte Compression (ICLR 2026)
- **arXiv:** 2603.03583
- **Authors:** Deng et al. (Rice University + Amazon Science)
- **Core idea:** Removes tokenizers entirely. Uses coding-rate-based compression
  to learn adaptive byte boundaries. Top-K selection preserves static computation
  graph while yielding semantically meaningful chunks.
- **Result:** Outperforms both BPE Transformers AND prior byte-level architectures.
  Accepted at ICLR 2026.
- **RELEVANCE TO E2:** MEDIUM. Alternative to our fixed-patch approach. Their
  adaptive chunking is more elegant than fixed-size patches. Our Sutra-Dyad uses
  fixed 6-byte patches for simplicity; ByteFlow shows learned boundaries can help.
  Not a KD competitor — purely an architecture paper. Could inform Stage 3+ if
  we move to adaptive patching.

### MBLM — Multiscale Byte Language Models (Feb 2025)
- **arXiv:** 2502.14553
- **Authors:** Egli, Manica et al.
- **Core idea:** Extends MegaByte hierarchy to unlimited stages. Refines byte
  representations through a hierarchy of generic decoder models. Hybrid
  Transformer+Mamba blocks for long sequences.
- **Scale:** 5M byte context windows on single GPU in full precision.
- **Result:** Hybrid architectures handle extremely long byte sequences during
  training. First evaluation of BLMs on visual Q&A tasks — pure next-byte
  prediction matches CNN-LSTM architectures.
- **RELEVANCE TO E2:** LOW-MEDIUM. Focuses on extreme context length (millions
  of bytes), which is not our concern at 153M params. Validates hybrid
  Transformer+Mamba for byte-level models — we could explore this if Sutra
  moves to a hybrid backbone.

### AU-Net — Autoregressive U-Nets for Byte Modeling (Jun 2025)
- **arXiv:** 2506.14761
- **Authors:** Videau, Youbi Idrissi, Leite, Schoenauer, Teytaud, Lopez-Paz
- **Core idea:** Learned multi-scale pooling (bytes → words → pairs of words →
  4 words). Deeper stages predict further ahead. Multi-scale view without
  fixed tokenization.
- **Result:** Shallow hierarchies tie strong BPE baselines; deeper hierarchies
  show promising trend but not yet clearly superior.
- **RELEVANCE TO E2:** LOW. Academic exploration of learnable byte hierarchies.
  Not competitive with BPE yet, and doesn't address KD or multi-teacher.

### Token→Byte Distillation (Feb 2026)
- **arXiv:** 2602.01007
- **Authors:** Bao, Leng, Wang, Peng, Lu
- **Core idea:** Two-stage curriculum: (1) progressive KD aligning byte
  representations with token-teacher embeddings, (2) byte-level SFT.
- **Result:** Retains 90%+ teacher performance using only ~125B bytes.
  Validated across Llama, Qwen, OLMo families.
- **RELEVANCE TO E2:** VERY HIGH. This is the closest single-teacher
  analogue to what we're building. Their two-stage curriculum parallels our
  E1→E2 pipeline. Key difference: they use a SINGLE token-based teacher;
  we use FIVE teachers across different architectures.

---

## 3. Multi-Teacher Knowledge Distillation

### Knowledge Purification (ICLR 2026)
- **arXiv:** 2602.01064
- **Core finding:** Multi-teacher KD performance DECLINES as teacher count
  increases without purification. Proposes five purification methods.
- **Five methods ranked (FLAN-T5 large, avg accuracy):**
  1. RL-based selection: 67.55% (CMV +0.029) — adaptive but expensive
  2. Similarity-based router: 67.20% (CMV +0.032) — best OOD transfer
  3. PLM classifier router: 66.40% (CMV +0.021) — ms latency
  4. Plackett-Luce ranking: 64.50% (CMV +0.010) — no training needed
  5. GPT-4 aggregation: 63.32% (CMV -0.004) — AGGREGATION HURTS
  Baseline TinyLLM (no purification): 62.53%
- **Key insight:** Routing > aggregation. The similarity-based router uses
  learned teacher embeddings + cosine similarity, wins on OOD because it
  needs only the question (not pre-sampled rationales) for routing.
- **RELEVANCE TO E2:** HIGH. Validates the general principle that routing
  beats aggregation for multi-teacher KD — their aggregation baseline has
  NEGATIVE CMV. Routing methods generalize best to OOD data. However, their
  work is at the rationale/response level (LLM reasoning); our E2 routing
  operates at the distribution/logit level on byte sequences. The paper
  supports "routing > averaging" broadly, not byte-level disagreement routing
  or phased admission specifically. Their similarity-router architecture
  parallels our PL-style router concept.

### MST-Distill — Mixture of Specialized Teachers (ACM MM 2025)
- **arXiv:** 2507.07015
- **Core idea:** Diverse teacher ensemble with instance-level routing and
  a plug-in masking module to suppress modality-specific discrepancies.
- **RELEVANCE TO E2:** HIGH. Instance-level routing is analogous to our
  per-position JSD-based routing. Their masking module parallels our
  purification step. Different domain (vision/audio/text multimodal) but
  same structural insight: specialized teachers need intelligent routing.

### Unified Multi-Teacher Across Hybrid Architectures (ICLR 2026 under review)
- **OpenReview:** 1lHp49KdwW
- **Core idea:** Learnable model token that interacts with features across
  multiple representation spaces via alternating intra-space and inter-space
  modules.
- **RELEVANCE TO E2:** MEDIUM. Vision-focused. Interesting "model token"
  concept for bridging representation spaces. Our approach uses separate
  projection ports per teacher architecture instead.

### Multi-Teacher Ensemble Distillation (Jan 2026)
- **arXiv:** 2601.09165
- **Core idea:** Mathematical framework for probability-domain knowledge
  aggregation from multiple teachers.
- **RELEVANCE TO E2:** MEDIUM. Theoretical grounding for teacher aggregation.
  Our arithmetic mean purification is a special case of their framework.

### Reliability-Gated Multi-Teacher KD (Apr 2026)
- **arXiv:** 2604.03192
- **Authors:** BRAC University
- **Core idea:** EWAD (Entropy-Weighted Agreement-Aware Distillation) — a
  token-level mechanism that routes supervision between teacher distillation
  and gold supervision based on **inter-teacher agreement**. Also proposes
  CPDP (Capacity-Proportional Divergence Preservation) — a geometric
  constraint on the student's position relative to heterogeneous teachers.
- **Domain:** Low-resource abstractive summarization (Bangla).
- **Result:** Logit-level KD provides most reliable gains. More complex
  distillation improves ROUGE for short outputs, degrades longer ones.
- **RELEVANCE TO E2:** VERY HIGH. EWAD's agreement-aware routing is
  conceptually identical to our A9 gold-free router's agreement penalty
  term (-gamma * z(A_t)). Same signal (teacher agreement = reliability),
  different implementation (they gate teacher vs gold; we weight teachers
  against each other). Must cite and differentiate: our routing is across
  5 heterogeneous-architecture teachers at byte-distribution level, theirs
  is across homogeneous teachers at token level. Validates our design.

### CaMOPD — Counteraction-Aware Multi-Teacher OPD (May 2026)
- **arXiv:** 2605.27115
- **Core idea:** Decoupling alternating training and gap-based sample
  selection to prevent gradient counteraction between multiple teachers.
- **Key mechanisms:**
  1. **Decoupled alternating schedule:** n_g general steps, then 1 domain
     step. Updates only the active branch, avoiding gradient counteraction.
  2. **Gap-based sample scoring:** Score = mean |teacher - student| log-prob
     gap per sample. Select top-scoring subset, skip low-demand tail that
     flattens useful signal ("weak-signal flattening" problem).
  3. **Gradient Coherence Gain (GCG):** Measures gradient alignment of
     selected subset vs full batch. GCG = (Coh(selected) - Coh(full)) /
     Coh(full), where Coh(A) = ‖Σg_i‖ / Σ‖g_i‖.
- **Result:** +5.74 LiveCodeBench, +19.6 Creative Writing over baseline.
- **RELEVANCE TO E2:** HIGH. Directly addresses teacher gradient conflicts.
  Their alternating schedule maps to our phased curriculum. Their gap-based
  scoring maps to our NLL-threshold cache selection. **E2v2 candidate:**
  implement GCG as a diagnostic metric in monitor.py — negative GCG on a
  selected subset indicates teacher gradient conflicts (note: GCG measures
  subset-vs-full coherence, not a direct per-teacher help/hurt detector). Our approach
  (cap gradients) is simpler; theirs (alternate + select) is more adaptive.
  Different solutions to the same problem. We should cite and differentiate.

### MT-BKD — Bayesian Multi-Teacher KD (May 2026)
- **arXiv:** 2605.27967
- **Authors:** Fang, Chen, Cai, Ma, Zhong
- **Core idea:** Teacher-informed mixture prior that integrates guidance from
  multiple teachers into one probabilistically coherent model. Uses Bayesian
  inference to capture inherent uncertainty in the distillation process.
- **RELEVANCE TO E2:** MEDIUM. Principled statistical framework for multi-teacher
  aggregation. Their mixture prior is a Bayesian version of our
  arithmetic-mean purification. More theoretically elegant, but requires
  posterior inference (expensive). Our router + gradient budget achieves
  similar goals more cheaply. Worth citing as theoretical support that
  uncertainty-aware teacher aggregation is principled.

### GRACE — Principled Teacher Selection (ICLR 2026)
- **arXiv:** 2511.02833
- **Core idea:** Lightweight score (GRACE) to predict which teacher will be
  most effective for a given student-task pair, without requiring trial runs.
  Measures distributional properties of student gradients. Correlates 86%
  with final distillation performance.
- **RELEVANCE TO E2:** LOW-MEDIUM. Focuses on SELECTING teachers before
  training, not routing during training. Could inform our initial teacher
  priors (replace hand-tuned prior weights with GRACE scores). But our
  routing is dynamic (per-position), while GRACE is static (per-teacher).

### Axiomatic Framework for Adaptive Weighting (Jan 2026)
- **arXiv:** 2601.17910
- **Authors:** Flouro, Chadwick
- **Core idea:** Operator-agnostic axiomatic framework for adaptive weighting
  in multi-teacher KD across three scales: token, task, and context. Identifies
  four structural conditions any valid weighting scheme must satisfy:
  normalization, bounded influence, regularity, and ordinal safety monotonicity.
- **Key result:** Proves existence and non-uniqueness of conforming operators.
  Characterizes convergence and stability bounds for gradient-based optimization
  under the axioms.
- **RELEVANCE TO E2:** HIGH. Our gold-free router (A9c) is an adaptive weighting
  operator. We should verify it satisfies these four axioms — especially bounded
  influence (no single teacher dominates) and ordinal safety monotonicity (safer
  teacher gets more weight). Our gradient budget already enforces bounded influence.
  Normalization is satisfied by softmax. Regularity and safety monotonicity are
  worth formal verification. Could strengthen our theoretical claims.

### Entropy-Aware OPD (ICML 2026)
- **arXiv:** 2603.07079
- **Authors:** Jin, Min, Yang et al.
- **Core idea:** Augments standard reverse KL distillation with forward KL on
  tokens where the teacher distribution has high entropy. Addresses mode-seeking
  brittleness of reverse KL under teacher uncertainty.
- **Result:** +1.37 to +5.05 Pass@8 accuracy gains on math reasoning benchmarks
  across small (0.6B) to medium (4B) models. Accepted at ICML 2026.
- **RELEVANCE TO E2:** MEDIUM-HIGH. Validates teacher entropy as a routing signal
  (same core insight as our A9c router's teacher entropy term). Their solution
  (switch KL direction) is token-level and single-teacher; ours is multi-teacher
  routing at byte level. But the shared insight — high teacher entropy requires
  different treatment — is confirmed at a top venue. Cite for validation.

### Drive-KD — Asymmetric Gradient Projection (Jan 2026)
- **arXiv:** 2601.21288
- **Authors:** Drive-KD team
- **Core idea:** Asymmetric Gradient Projection (AGP) for multi-teacher
  gradient conflict resolution. Treats CE loss as anchor, teacher losses
  as followers. Only removes the conflicting gradient component from
  followers, preserving non-conflicting teacher signal.
- **AGP formula:** `g_f ← g_f − (g_f⊤g_a / ‖g_a‖²) g_a, if g_f⊤g_a < 0`
  where g_a is the CE (anchor) gradient and g_f is the teacher (follower)
  gradient. Non-conflicting components pass through unmodified.
- **Result:** +7.47 reasoning points, +4.48 planning points over naive
  multi-teacher distillation without AGP.
- **RELEVANCE TO E2:** HIGH. Our gradient budget uniformly scales teacher
  grads to 0.30 of CE norm. AGP is more surgical — only removes the
  conflicting component. **E2v2 candidate:** combine AGP (remove conflicts)
  + cap (safety bound on non-conflicting magnitude). Requires per-parameter
  inner products (compute cost), but preserves more teacher signal. Must
  benchmark against current cap-only approach after E2v1 baseline.

### QR-Distill — Routing Diverse Reasoning Paths (2025/2026)
- **arXiv:** 2508.16861
- **Core idea:** Quality filtering + conditional routing + cooperative peer
  teaching. Routes reasoning paths based on student's CURRENT learning
  state, not static teacher quality.
- **Key insight:** Conditional routing adapts which reasoning examples each
  student receives based on its learning progress. Students also mutually
  distill diverse insights (peer teaching).
- **RELEVANCE TO E2:** MEDIUM. Their student-adaptive routing partially
  maps to our phased admission (different teachers enter at different
  training phases based on student readiness). Their peer teaching doesn't
  apply (we have one student). Validates adaptive routing over static.

### PCGrad-Style Gradient Surgery for Multi-Teacher KD
- **Background:** PCGrad (Yu et al., 2020) projects conflicting gradients
  to eliminate negative transfer. AGP (above) is the multi-teacher KD
  specialization with asymmetric anchor-follower roles.
- **RELEVANCE TO E2:** Superseded by AGP analysis above.

---

## 4. Gap Analysis — What E2 Does That Nobody Else Does

| Capability | BLD | X-Token | Token→Byte | KPurify | MST-Distill | CaMOPD | DWA-KD | EWAD | MT-BKD | EA-OPD | Axiom | **E2** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Byte-level interface | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| Multi-teacher | ❌ | ✅† | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ |
| Cross-architecture | ❌ | ✅† | ❌ | ❌ | ✅* | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| Disagreement routing | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ |
| Entropy-weighted selection | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | ✅ | ❌ | ✅ |
| Gradient conflict handling | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| Gold-free routing | N/A | N/A | N/A | ❌ | ❌ | N/A | N/A | Partial | ❌ | N/A | N/A | ✅ |
| Phased teacher admission | ❌ | ❌ | Partial | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| Uncertainty-aware aggregation | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ |
| Formal weighting axioms | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | Partial |
| Small student (<200M) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |

*MST-Distill crosses modalities, not neural architectures.
†X-Token demonstrated 2-teacher multi-architecture (Phi-4-Mini + Llama-3B)
with static weighting (+1.3 over single-teacher). No routing, no gradient
budgeting. This is our minimum bar.

**E2's unique combination:** Multi-teacher × cross-architecture ×
disagreement routing × gradient budgeting × phased admission ×
compact student. The byte-level substrate is our current mechanism
for cross-tokenizer alignment, but the core novelty is the integrated
multi-teacher learning protocol, not the byte processing itself.
No published work combines more than three of these capabilities.
X-Token's two-teacher result narrows the gap on multi-teacher +
cross-architecture, but without routing or gradient management.

### Nearest Threats
1. **X-Token multi-teacher (NVIDIA, May 2026)** — Already demonstrated
   two-teacher cross-architecture KD with +1.3 improvement. Uses static
   weighting, no routing. If they add routing, they directly compete.
   Our advantage: five teachers, dynamic routing, gradient budgeting,
   phased admission. But they have NVIDIA resources and visibility.
2. **BLD + multi-teacher extension** — Someone could extend BLD to multiple
   teachers. Our head start: we already have the routing, purification,
   gradient budgeting, and phased admission built and tested.
3. **Token→Byte + multi-teacher** — Similar risk. The Feb 2026 paper's
   two-stage curriculum could be extended to multiple teachers.
4. **MT-BKD Bayesian framework (May 2026)** — Principled uncertainty-aware
   aggregation. If combined with cross-tokenizer methods, could be a
   theoretically cleaner alternative to our router. Our advantage: we're
   already built and tested; their framework needs cross-tokenizer support.

### What the Field Validates
1. **Bytes as cross-tokenizer interface** — BLD and Universal CT both confirm
   this is a sound approach. Not controversial anymore.
2. **Router-based purification** — ICLR 2026 Knowledge Purification paper
   confirms router-based methods generalize best.
3. **More teachers can hurt** — Knowledge Purification confirms that naive
   multi-teacher averaging hurts. Routing-based selection is the defense;
   our phased admission is an orthogonal mitigation (untested by this paper).
4. **Patch-global architectures work** — BLT proves this at 8B scale. ByteFlow
   (ICLR 2026) shows adaptive byte chunking can outperform both BPE and prior
   byte-level models — validates the byte-level paradigm broadly.
5. **Entropy-weighted routing** — DWA-KD (Feb 2026) and EWAD (Apr 2026)
   independently confirm that weighting by student uncertainty / teacher
   confidence is a principled approach. Our A9 gold-free router uses
   the same core insight across heterogeneous architectures.
6. **Agreement-based teacher reliability** — EWAD's inter-teacher agreement
   gating directly validates our A9 agreement penalty term (-gamma * z(A_t)).
7. **Teacher entropy as routing signal** — Entropy-Aware OPD (ICML 2026)
   confirms that high teacher entropy requires different treatment (they switch
   KL direction). Validates our A9c router's teacher entropy term.
8. **Formal axioms for adaptive weighting** — Axiomatic Framework (2601.17910)
   provides structural conditions (normalization, bounded influence, regularity,
   ordinal safety monotonicity) that our router should satisfy. Our gradient
   budget enforces bounded influence; softmax gives normalization.

### What the Field Has NOT Solved
1. Multi-teacher KD with >2 heterogeneous architectures (transformer + SSM +
   hybrid + embedding model) — X-Token showed 2-teacher but with static
   weighting; **5+ teachers with routing remains unpublished**.
2. Disagreement-based teacher routing at the byte-distribution level —
   **nobody has published this**.
3. Gradient budgeting for multi-teacher conflicts — CaMOPD uses alternating
   training; Drive-KD uses AGP projection; **per-teacher gradient capping
   with cap + projection remains unpublished**.
4. Sub-200M byte-level model with multi-teacher KD — **nobody has published this**.
5. Combined gold-free routing + gradient budgeting + phased admission —
   **nobody has published this combination**.

### E2v2 Upgrade Candidates (Post-Baseline)
1. **AGP (Drive-KD):** Replace uniform gradient cap with conflict-aware
   projection. Only remove conflicting gradient components, preserve
   non-conflicting teacher signal. Combine with existing cap as safety bound.
2. **Gradient Coherence Gain (CaMOPD):** Implement as diagnostic metric in
   monitor.py. If GCG is negative, teacher signal is counterproductive.
3. **Contrastive teacher embeddings (Knowledge Purification):** Replace
   static teacher priors with learned embeddings and cosine routing. Their
   similarity-based router is most robust on OOD.
4. **Gap-based sample scoring (CaMOPD):** Score training samples by
   teacher-student gap magnitude, train only on high-gap samples. Related
   to our NLL-threshold selection but applied during training, not caching.

---

## 5. Implications for E2 Experiments

### Strengthened claims
- Our byte-level interface choice is validated by multiple 2026 papers.
- Router-based teacher aggregation is confirmed as the right approach.
- Phased teacher admission is validated by knowledge purification findings.

### New baselines to compare against
- **BLD** should be our primary cross-tokenizer KD baseline (if code is
  available). It's the fairest comparison since it also uses bytes.
- **Token→Byte distillation** is the single-teacher byte baseline.
- We should cite Knowledge Purification as theoretical support for our
  routing design.

### Urgency — ELEVATED
The field is accelerating. Eight+ papers in 6 months (Jan-Jun 2026) on
cross-tokenizer KD. X-Token (NVIDIA) has already demonstrated multi-teacher
cross-architecture KD with measured gains. MT-BKD provides a Bayesian
framework for multi-teacher uncertainty. Our window of novelty is narrowing.
The unique value is the FULL COMBINATION: byte-level + 5 heterogeneous
teachers + disagreement routing + gradient budgeting + phased admission.
We need results NOW to claim this territory before someone combines
X-Token's multi-teacher with routing.

**Updated survey date:** 2026-06-27
