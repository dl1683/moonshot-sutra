# Independent Analysis: What Eklavya Needs (First Principles)

**Author: Claude (independent of Codex, for cross-validation)**
**Date: 2026-07-03**

---

## The Question

What does it take for a 121M byte-level student to achieve ≥42.65% HellaSwag
(beating CBD at 138M)? Not "what should we try next" but "what does first-principles
reasoning tell us MUST be true for this to work?"

---

## Decomposition

HellaSwag requires: given 4 continuations, pick the correct one. Our scoring
method: compute byte-level perplexity for each continuation, choose lowest.

For the student to succeed, it must satisfy:

```
For each example:
  BPB(correct_continuation | context) < BPB(best_wrong | context)
```

This margin must be positive on >42.65% of examples.

### What Creates Margin?

The margin comes from the model assigning higher probability to bytes in the
correct continuation vs bytes in wrong ones. For a byte-level model:

```
margin = Σ_j [log P(b_j^wrong | ctx, b_{<j}^wrong) - log P(b_j^correct | ctx, b_{<j}^correct)]
```

For this to be positive, the model must "prefer" the byte sequence of the
correct continuation. This requires the model to have learned WHICH byte
sequences are plausible continuations — a semantic judgment encoded through
byte predictions.

### The Fundamental Tension

A byte-level model must simultaneously:
1. **Learn byte mechanics**: spelling, spacing, encoding, common byte patterns
2. **Learn semantics**: which concepts follow which in language
3. **Learn discrimination**: which continuations are more/less plausible

Standard training (CE on bytes) does #1 and #2. KD via byte marginals
supposedly helps #2. But neither directly addresses #3.

HellaSwag measures ONLY #3.

---

## Three Independent Arguments for Why We Fail

### Argument 1: Information Bottleneck (Information Theory)

The teacher produces a distribution over V tokens: P(t | context).
Each token maps to bytes: t → (b_0, b_1, ..., b_{n-1}).

The byte-0 marginal is: Q(b_0 = b) = Σ_{t: first_byte(t)=b} P(t)

**How much of the teacher's information does Q preserve?**

Teacher entropy: H(T) ≈ log₂(V) in the worst case ≈ 15 bits for V=32K.
But typical distributions are much more concentrated. Let's say H(T) ≈ 4-8 bits.

Byte-0 entropy: H(B_0) ≤ 8 bits.

Mutual information: I(T; B_0) = H(T) - H(T | B_0).

H(T | B_0) depends on how many tokens share each first byte. For BPE
tokenizers, common ASCII bytes (a-z, A-Z, space, etc.) have 100+ tokens
starting with them. So:

H(T | B_0 = b) ≈ log₂(count_tokens_starting_with_b) ≈ 3-7 bits

On average: H(T | B_0) ≈ 4-6 bits.

So I(T; B_0) ≈ H(T) - H(T|B_0) ≈ 2-4 bits.

**The byte-0 marginal captures maybe 30-60% of the teacher information.**
And that's for FIRST BYTE ONLY. We use first-byte-only caching
(byte_positions="first"), so we get one marginal per patch position,
covering only 1/4 of the bytes in a P=4 patch.

Over a full token, the byte marginals are NOT independent — they're
computed from the SAME token distribution. But we use only byte 0.
So the student sees byte_0 marginals at each patch position, which
capture maybe 30-60% of the teacher's information at that position.

**The remaining 40-70% contains the semantic structure that distinguishes
correct from incorrect continuations.**

### Argument 2: Capacity Allocation (Architecture Analysis)

The student has 121.7M parameters distributed across:

| Component | Layers | Width | Params (est.) | Purpose |
|-----------|--------|-------|---------------|---------|
| ByteEncoder | 2 + MLP | 576 | ~5M | Bytes → patch state |
| GlobalReasoner | 30 | 576 | ~100M | Semantic processing |
| ByteDecoder | 4 | 576 | ~15M | Patch state → bytes |

ByteEncoder + ByteDecoder: ~20M params (~16% of total)

These 20M params learn byte-level mechanics that a token-level model
gets for FREE from its tokenizer. A token-level model at 138M (CBD)
uses ALL 138M params for semantics. We effectively have ~100M for semantics.

But worse: the ByteDecoder generates bytes autoregressively within each
patch. This means the decoder's 15M params learn intra-token byte patterns
("if previous byte was 't', next is likely 'h'" for "the"). A token-level
model has NO such overhead.

**Our effective semantic capacity is ~73% of CBD's (100M vs 138M).**

### Argument 3: Objective Misalignment (Optimization Theory)

KL divergence on byte distributions minimizes:

```
D_KL(Q_teacher || P_student) = Σ_b Q(b) log(Q(b) / P(b))
```

This is a POINTWISE measure — it cares about matching the distribution
at each byte position independently. It has no concept of:
- Continuation ranking
- Multi-byte coherence
- Semantic plausibility

HellaSwag scoring computes:

```
score(continuation) = Σ_j log P_student(b_j | context, b_{<j})
```

and picks the continuation with highest score.

KL training improves EACH P_student(b_j | ...) toward Q_teacher(b_j | ...).
But this improvement is uniform across all continuations — correct AND
incorrect. The margin doesn't change because the model gets better at
ALL byte predictions, not specifically at discriminating correct from wrong.

**For margin to improve, the model needs to learn RELATIVE preferences,
not absolute byte probabilities.**

---

## What WOULD Work (From First Principles)

### Approach A: Direct Ranking Objective

Train the student to directly rank continuations:

```
L_rank = -log(exp(score(correct)) / Σ_i exp(score(choice_i)))
```

where score(choice) = -BPB(choice | context).

This is InfoNCE / contrastive loss applied at the continuation level.
It directly optimizes the HellaSwag metric.

**Problem**: Requires labeled continuation pairs during training. We have
HellaSwag data (10K examples), but that's tiny compared to our training data.

**Solution**: Use the TEACHER to generate rankings. For any context in our
training data, generate random continuations and have the teacher rank them.
The student learns the teacher's ranking, not the teacher's byte distribution.

### Approach B: Hidden State Injection

Bypass the byte interface entirely. Inject teacher knowledge into the
GlobalReasoner's hidden states:

```
L_hidden = ||h_student_layer_l - W_proj @ h_teacher_layer_k||²
```

where W_proj: R^{d_teacher} → R^{d_student} is a learned projection.

The student still learns bytes via CE. The teacher signal goes directly
into the reasoner, not through the byte head. This avoids the byte-marginal
information bottleneck.

**Problem**: Cross-architecture hidden state matching is hard. Teacher and
student process different granularities (tokens vs patches).

**Solution**: Align at SEMANTIC POSITIONS — after the teacher has processed
a complete sentence, compare the teacher's final hidden state at that
position with the student's hidden state at the corresponding patch position.

### Approach C: Functional Distillation

Don't match distributions. Match the INPUT-OUTPUT FUNCTION.

Given many (context, continuation) pairs, the teacher assigns a score
(log probability) to each. Train the student so that:

```
rank_student(choices | context) ≈ rank_teacher(choices | context)
```

This is model-agnostic. It doesn't matter what architecture the teacher uses,
what tokenizer it has, or what representation it builds internally. All that
matters is the FUNCTION mapping (context, choice) → score.

**This completely sidesteps the byte-marginal bottleneck, the architecture
mismatch, and the objective misalignment.** The student learns the teacher's
evaluation function directly.

**How to implement:**
1. Pre-compute teacher scores for many (context, continuation) pairs
2. Train student to match these scores via ranking loss
3. Use CE on bytes for basic language competence
4. The ranking loss and CE loss are INDEPENDENT objectives

### Approach D: Progressive Chain (Inspired by CBD)

The 5:1 compression ratio (620M → 121M) is too large for one step.

Build a chain: 620M → 300M → 121M (or even more steps).

Each step crosses a 2:1 ratio, which is within the geometric capacity limits.

**Problem**: Building intermediate models is expensive.
**Advantage**: Proven to work (CBD achieves 42.65%).

---

## Ranking the Approaches

| Approach | Expected Impact | Feasibility | Novel? | Risk |
|----------|----------------|-------------|--------|------|
| C: Functional | HIGH | HIGH | YES | Medium — untested |
| A: Ranking | HIGH | HIGH | No — established | Low |
| B: Hidden State | MEDIUM | MEDIUM | No | High — alignment hard |
| D: Chain | HIGH | LOW | No — CBD does this | Low |

**My recommendation: Approach C (Functional Distillation) is the most
promising because it completely sidesteps all three failure modes
simultaneously. It's also the most aligned with the manifesto — it's a
fundamentally different way of transferring knowledge.**

---

## Deep Dive: The Granularity Mismatch Problem (2026-07-03)

Both Codex and I converge on representation alignment (Approach B above, Codex
calls it Strategy C). The hardest technical problem is: HOW do you align
representations when teacher and student operate at completely different
granularities?

### The Setup

```
Teacher (Qwen3-0.6B):
  Input: "Hello world" → tokens [15339, 1917]
  Hidden states: h_teacher[0] (at "Hello"), h_teacher[1] (at " world")
  Each hidden state: R^1024

Student (Sutra S0):
  Input: "Hello world" → bytes [72,101,108,108,111,32,119,111,114,108,100]
  Patches: [72,101,108,108] [111,32,119,111] [114,108,100,PAD]
  Patch states after GlobalReasoner: h_student[0], h_student[1], h_student[2]
  Each hidden state: R^576
```

Token "Hello" spans bytes 0-4, which spans patches 0 AND partially patch 1.
Token " world" spans bytes 5-10, which spans patches 1 AND partially patch 2.

### Why Naive Position Matching Fails

If we try: align h_student[i] to h_teacher[j] where patch i overlaps with
token j — we get many-to-many mappings. Patch 1 overlaps with BOTH tokens.
Token " world" overlaps with BOTH patches 1 and 2.

### Five Possible Solutions

**Solution 1: Token-Start Alignment**
For each teacher token, find the student patch that contains its FIRST byte.
Align student patch state to teacher hidden state at that position.

- "Hello" starts at byte 0 → patch 0. Align h_student[0] ↔ h_teacher[0].
- " world" starts at byte 5 → patch 1. Align h_student[1] ↔ h_teacher[1].

Pro: Simple. One-to-one mapping (mostly). Already have `compute_token_byte_spans()`.
Con: Short tokens (single byte " ") might pile up — multiple tokens in one patch.

**Solution 2: Weighted Pool**
For each teacher token, compute what FRACTION of its bytes fall in each patch.
Weight the alignment loss accordingly.

- "Hello" (5 bytes): 4/5 in patch 0, 1/5 in patch 1.
  L_align += 0.8 * ||W·h_teacher[0] - h_student[0]||² + 0.2 * ||W·h_teacher[0] - h_student[1]||²

Pro: Handles boundary splits correctly.
Con: More complex. May blur the alignment signal.

**Solution 3: Cross-Attention (Implicit Alignment)**
Add a small cross-attention module that lets student patch states attend to
ALL teacher hidden states. No explicit position matching — the attention
learns the alignment.

L_align = ||CrossAttn(h_student, h_teacher) - h_student_target||²

Or simpler: add cross-attention as a new component, and let gradients flow
through it to teach the student.

Pro: Handles arbitrary granularity mismatches. Most flexible.
Con: Requires teacher hidden states at training time (not just cached logits).
This changes our caching strategy entirely.

**Solution 4: Aggregate to Sentence-Level**
Don't align at individual position level. Pool both teacher and student states
over full sequences, and align the pooled representations.

L_align = ||mean(h_student) - W·mean(h_teacher)||²

Pro: Granularity mismatch vanishes.
Con: Loses positional information. Too coarse — may not transfer enough.

**Solution 5: Align at Decoder Output (Post-Hoc)**
Let the student process bytes normally. At the ByteDecoder output, for each
byte position, compare what the student produces vs what the teacher's
representation IMPLIES at that byte.

This is closer to our current byte-marginal approach but uses teacher hidden
states instead of logits.

### My Ranking (Independent of Codex)

| Solution | Feasibility | Expected Impact | My Confidence |
|----------|------------|-----------------|---------------|
| 1: Token-Start | HIGH | MEDIUM | 70% — good starting point |
| 3: Cross-Attn | MEDIUM | HIGH | 65% — most flexible but expensive |
| 2: Weighted Pool | HIGH | MEDIUM | 50% — may blur signal |
| 5: Decoder Output | HIGH | LOW-MEDIUM | 40% — still at output level |
| 4: Sentence-Level | HIGH | LOW | 30% — too coarse |

**Recommendation: Start with Solution 1 (token-start alignment).**
It's simplest, uses existing `compute_token_byte_spans()`, and gives a clean
diagnostic: does ANY form of representation alignment help?

If Solution 1 shows signal, upgrade to Solution 3 (cross-attention) for
production. If Solution 1 shows nothing, the problem is deeper than alignment
granularity.

### The Caching Implication

**This changes what we cache from teachers.** Currently we cache byte-marginal
logit distributions. For representation alignment, we need to cache:

- Teacher hidden states at each token position (R^1024 per token)
- Token-to-byte position mapping (already have `compute_token_byte_spans`)

Storage: ~1024 floats * 2 bytes (fp16) = 2KB per token position.
For a 1024-token sequence: ~2MB per sequence.
For 1M training sequences: ~2TB. Too large.

**Compromise**: Cache only the PROJECTED teacher states (W·h_teacher → R^576).
This cuts storage to ~1.1KB per position. Or: cache at only selected layers.

**Alternative**: Don't cache at all. Run teacher in forward-pass mode during
training (no gradients through teacher). At 0.6B parameters, teacher forward
pass is ~2.5x student forward pass. Adds 150% compute overhead but avoids
the caching problem entirely.

### What The Toy Experiment Should Test

Before building any of this for real:

1. Create a tiny teacher (width 128, 4 layers) and tiny student (width 64, 4 layers)
2. Train teacher on a simple language task (e.g., next-character prediction on text)
3. Distill via three methods:
   a. Byte-marginal KL (our current approach)
   b. Token-start representation alignment (Solution 1)
   c. Combined CE + representation alignment
4. Evaluate on a discrimination task (pick correct continuation)
5. Measure: which method creates the largest margin between correct/incorrect?

This is the minimum experiment that validates whether representation alignment
beats byte-marginal KL at transferring discriminative ability.

Expected time: ~1-2 hours (toy models, short training).

---

## Unknown Unknowns I Can't Answer Yet

1. How much of the BPB-accuracy disconnect is due to the harness bug (missing space)?
2. What is CBD's actual architecture? (Token-level? Same tokenizer? Width?)
3. Does a wider student (D=1024) with byte-level architecture work?
4. Is the GlobalReasoner actually learning semantic features, or just
   byte transition statistics dressed up to look semantic?
5. Would a token-level student at 121M match CBD? (This would prove bytes are the problem.)
