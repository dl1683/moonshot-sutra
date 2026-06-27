# Eklavya E1 Protocol — Byte-Level Knowledge Distillation for S0

Status: DESIGNED. Produced through 4-round adversarial deliberation
(R1-R4, June 2026) plus review of April-June 2026 cross-tokenizer KD literature.

## Core Insight

S0 is byte-level. Every teacher's output can be projected to bytes regardless
of tokenizer. This eliminates the cross-tokenizer alignment problem that makes
multi-teacher KD hard for token-level students.

## Prior Art Integration

- BLD (arXiv 2604.07466): byte-level interface for cross-tokenizer KD.
  Simple first-byte marginals are competitive with sophisticated methods.
- H-Net (arXiv 2602.01007): embedding alignment is critical (MMLU 70.7→31.2
  without it). Progressive curriculum prevents instability.
- Breaking Tokenizer Barrier (arXiv 2606.09456): on-policy cross-tokenizer OPD.

## Protocol Overview

```
E0: Train S0 to stable BPB (50K steps, base CE only)
E1: Single-teacher distillation (anchor decoder teacher)
    E1.0: Projection warmup (500 steps)
    E1.1: Alignment landing (1500 steps)
    E1.2: Full Eklavya (10K+ steps)
E2: Multi-teacher extension (add diversity, semantic, control teachers)
```

Multi-teacher (E2) only proceeds if single-teacher (E1) shows decisive gains.

## Two Loss Signals

### 1. Embedding Alignment (L_align)

Match S0 ByteEncoder patch states (pre-reasoner) to anchor teacher static token
embeddings. This teaches the encoder to represent byte spans in a space
compatible with the teacher's semantic structure.

```
Target: ByteEncoder output → teacher input embeddings
NOT: GlobalReasoner output → teacher hidden states (that's stage 2)
```

For a teacher token spanning bytes [a, b), compute weighted overlap pooling
over S0 patches touching that span:

```python
student_span = overlap_pool(patch_states, byte_start, byte_end, P=4)
z_s = LayerNorm(Linear_576→2048(LayerNorm(student_span)))
z_t = LayerNorm(teacher_embedding[token_id])
L_align = MSE(z_s, z_t)
```

Projection head: single linear 576→2048, bias=False, with LayerNorm.

### 2. Byte-Level KL (L_kl) — Patch Position 0 Only

At each patch boundary (position 0 within each patch), compute KL divergence
between student's next-byte prediction and teacher's first-byte marginal.

Why position 0 only:
- Predicted from global hidden state alone (highest leverage)
- Positions 1-3 are decoder-local, CE is sufficient
- First-byte marginal is cheap to compute from teacher logits
- Top-K=16 + tail probability for storage efficiency

Teacher first-byte marginal:
```python
# For each teacher token, group next-token probabilities by first byte
q = zeros(256)
for tok_id in topk(teacher_next_token_probs, 4096):
    first_byte = token_to_bytes(tokenizer, tok_id)[0]
    q[first_byte] += teacher_prob[tok_id]
q = normalize(q)
```

KL loss with top-K + tail:
```python
L_kl = T² * (-sum(top_probs * log_softmax(student_logit/T)[top_bytes])
             - tail_prob * log(1 - sum(softmax(student_logit/T)[top_bytes])))
```

## Training Schedule

### E1.0 — Projection Warmup (500 steps)
- Freeze all S0 parameters
- Train only align_proj (LayerNorm + Linear)
- Loss: L_align only
- LR: 3e-4

### E1.1 — Alignment Landing (1500 steps)
- Unfreeze ByteEncoder + patch aggregator
- Freeze GlobalReasoner + decoder
- Loss: CE + λ_align * L_align
- Base LR: 3e-5, proj LR: 3e-4

### E1.2 — Full Eklavya (10K+ steps)
- Unfreeze all S0 parameters
- Loss: CE + λ_align * L_align + λ_kl * L_kl
- Base LR: 3e-5, proj LR: 3e-4
- Cache refresh every 2K steps

## Hyperparameters

```
lambda_align = 0.05
lambda_kl = 0.10
kl_temperature = 2.0
base_lr = 3e-5
align_proj_lr = 3e-4
nll_floor = 3.5 nats (~5.05 bits)
nll_threshold = max(nll_floor, p90_nll)  # adaptive per-sequence
top_k_bytes = 16
initial_kl_records = 10M
initial_align_records = 20M
refresh_every = 2000 steps
refresh_kl_records = 2M per refresh
```

Gradient cap: all teacher losses ≤ 0.30 * base CE gradient norm.

## Teacher Signal Cache

Offline pre-computation, not online teacher inference during training.

### Alignment Records (~500 MB for 20M records)
```
shard_id, seq_offset, byte_start, byte_len, token_id
```
Token embedding looked up from shared table (stored once, ~622 MB fp16).

### Byte KL Records (~1.6 GB for 20M records)
```
shard_id, seq_offset, patch_idx, top_bytes[16], top_probs[16], tail_prob, entropy
```

### Selection Policy
```
cache patch position 0 where:
  student NLL > max(3.5, p90_nll)
  OR patch in eval failure window
  OR random control sample (~10-20%)
```

### Refresh Protocol
Every 2K steps:
1. Run student on sample of training data
2. Identify current top-K gap windows
3. Re-run teacher on those windows only
4. Update KL records for refreshed positions

## Controls and Ablations

Priority order:
```
C0: S0 checkpoint, no continuation
C1: CE-only continuation (12K steps)
C2: CE + align only
C3: CE + patch0 KL only
C4: CE + align + patch0 KL       ← main E1
C5: CE + shuffled align targets  ← falsification
C6: CE + shuffled KL targets     ← falsification
C7: CE + align + KL, no refresh
C8: CE + align + KL, with refresh
```

Minimum success claim:
- C4 > C1, C2, C3 individually on eval BPB
- C5/C6 fail or underperform real targets
- C8 ≥ C7 after enough steps
- BPB does not regress materially
- Byte accuracy improves on high-NLL slices

## Multi-Teacher Extension (E2)

**E2 now has its own canonical protocol document:**
[research/EKLAVYA_E2_PROTOCOL.md](EKLAVYA_E2_PROTOCOL.md)

Produced through 3-round adversarial deliberation (R1-R3, June 2026). Covers:
5-teacher roster, binary cache architecture, PL-style router, arithmetic/log-pool
purifier, per-teacher gradient budget, 7-ablation suite, retained-gain tests.

Implementation: `code/eklavya_e2_cache.py`, `code/eklavya_e2_router.py`,
`code/eklavya_e2_losses.py`, `code/eklavya_e2_training.py` (397 E2 tests passing).

E2 only proceeds after E1 shows decisive gains (>2pp improvement over CE-only).

## Files

- `code/eklavya_cache.py` — E1 offline teacher signal cache builder
- `code/eklavya_training.py` — E1 distillation training loop
- `code/test_eklavya.py` — E1 unit tests (30 passing)
- `code/eklavya_e2_cache.py` — E2 teacher registry + binary cache
- `code/eklavya_e2_router.py` — E2 router + purifier
- `code/eklavya_e2_losses.py` — E2 losses + gradient budget
- `code/eklavya_e2_training.py` — E2 trainer with curriculum
- `code/test_eklavya_e2.py` — E2 unit tests (224 passing)
- `research/EKLAVYA_E1_PROTOCOL.md` — this document
- `research/EKLAVYA_E2_PROTOCOL.md` — E2 canonical protocol

## Decision Log

| Round | Key Decision | Rationale |
|-------|-------------|-----------|
| R1 | Sparse gap-local, not dense byte KL | Dense is expensive, teaches tokenizer artifacts |
| R2 | Embedding alignment → Stage 1 priority | H-Net ablation: critical (70.7→31.2 without it) |
| R2 | Hybrid cache: offline + periodic refresh | Static packets go stale; full online crushes throughput |
| R3 | Align ByteEncoder (pre-reasoner), not GlobalReasoner | Static teacher embeddings match pre-context student representations |
| R3 | Byte KL at patch position 0 only | Highest leverage, cheapest to compute, CE handles positions 1-3 |
| R3 | Token-level cache, not byte-level | Storage feasibility: 500MB vs 45TB |
| R4 | Single linear projection (576→2048) | Avoid hiding failure behind trainable teacher projection |
| R4 | 3-phase training: warmup → landing → full | Prevent noisy alignment gradients from damaging encoder |
