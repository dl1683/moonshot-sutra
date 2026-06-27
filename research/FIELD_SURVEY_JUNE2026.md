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
- **RELEVANCE TO E2:** MEDIUM. Uses token-level projections, not byte-level.
  Our byte interface sidesteps both failure modes entirely (every teacher
  produces byte distributions, so there's no vocabulary mismatch).

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

### DSKD — Dual-Space KD (EMNLP 2024 + KQ extension Mar 2026)
- **GitHub:** github.com/songmzhang/DSKD
- **Core idea:** KD in both logit space and hidden-state space, with
  key-query matching for vocabulary mismatch.
- **RELEVANCE TO E2:** MEDIUM. Our align loss (E1) operates in hidden-state
  space, similar to their dual-space concept.

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
- **Key result:** Router-based methods have the best generalization.
- **RELEVANCE TO E2:** VERY HIGH. Directly validates our design choices:
  (1) disagreement-based routing is the right approach, (2) naive teacher
  averaging hurts, (3) purification is necessary. Their work is at the
  rationale/response level for LLMs; ours is at the distribution/logit level.
  Complementary, not competing.

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

### CaMOPD — Counteraction-Aware Multi-Teacher OPD (May 2026)
- **arXiv:** 2605.27115
- **Core idea:** Decoupling alternating training and gap-based sample
  selection to prevent gradient counteraction between multiple teachers.
- **RELEVANCE TO E2:** HIGH. Directly addresses teacher gradient conflicts,
  which is the problem our gradient budgeting solves. Their approach
  alternates teachers; ours caps gradients. Different solutions to the
  same problem. We should cite them and differentiate.

### PCGrad-Style Gradient Surgery for Multi-Teacher KD
- **Background:** PCGrad (Yu et al., 2020) projects conflicting gradients
  to eliminate negative transfer. Applied to multi-teacher KD by projecting
  teacher gradients to the normal plane of student gradients.
- **RELEVANCE TO E2:** MEDIUM-HIGH. Our gradient budgeting is simpler
  (scale/cap) than projection. We should consider whether PCGrad-style
  projection would be a stronger baseline than our budget capping.

---

## 4. Gap Analysis — What E2 Does That Nobody Else Does

| Capability | BLD | X-Token | Token→Byte | KPurify | MST-Distill | CaMOPD | **E2** |
|---|---|---|---|---|---|---|---|
| Byte-level interface | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ |
| Multi-teacher | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ |
| Cross-architecture | ❌ | ❌ | ❌ | ❌ | ✅* | ❌ | ✅ |
| Disagreement routing | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | ✅ |
| Gradient conflict handling | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ |
| Phased teacher admission | ❌ | ❌ | Partial | ❌ | ❌ | ❌ | ✅ |
| Small student (<200M) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |

*MST-Distill crosses modalities, not neural architectures.

**E2's unique combination:** Multi-teacher × cross-architecture ×
disagreement routing × gradient budgeting × phased admission ×
compact student. The byte-level substrate is our current mechanism
for cross-tokenizer alignment, but the core novelty is the integrated
multi-teacher learning protocol, not the byte processing itself.
No published work combines more than two of these capabilities.

### Nearest Threats
1. **BLD + multi-teacher extension** — Someone could extend BLD to multiple
   teachers. Our head start: we already have the routing, purification,
   gradient budgeting, and phased admission built and tested.
2. **Token→Byte + multi-teacher** — Similar risk. The Feb 2026 paper's
   two-stage curriculum could be extended to multiple teachers.

### What the Field Validates
1. **Bytes as cross-tokenizer interface** — BLD and Universal CT both confirm
   this is a sound approach. Not controversial anymore.
2. **Router-based purification** — ICLR 2026 Knowledge Purification paper
   confirms router-based methods generalize best.
3. **More teachers can hurt** — Knowledge Purification confirms that naive
   multi-teacher hurts. Our phased admission is the right defense.
4. **Patch-global architectures work** — BLT proves this at 8B scale.

### What the Field Has NOT Solved
1. Multi-teacher KD with heterogeneous architectures (transformer + SSM +
   hybrid + embedding model) — **nobody has published this**.
2. Disagreement-based teacher routing at the byte-distribution level —
   **nobody has published this**.
3. Gradient budgeting for multi-teacher conflicts — **nobody has published this**.
4. Sub-200M byte-level model with multi-teacher KD — **nobody has published this**.

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

### Urgency
The field is moving fast. Six papers in 5 months (Jan-Jun 2026) on
cross-tokenizer KD alone. Our window of novelty is the COMBINATION of
multi-teacher + byte-level + cross-architecture. We need results to
claim this territory.
