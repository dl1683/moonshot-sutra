# E2.0 Teacher Feasibility Profiling — Executable Checklist

Status: Pre-GPU runbook. Run this before any E2 training.

## Purpose

E2.0 validates each teacher before spending GPU-hours building cache. A teacher
that fails feasibility is dropped or replaced before cache construction, not
during training.

## Per-Teacher Checks

Run each check sequentially per teacher (one teacher at a time for VRAM safety).

### Check 1: Load Sanity

```
For each teacher in TEACHER_REGISTRY:
  1. Load model via load_teacher_for_spec(spec, device)
  2. Verify model loads without error
  3. Verify actual VRAM < spec.vram_gb * 1.2  (20% headroom)
  4. Load tokenizer via AutoTokenizer.from_pretrained(spec.hf_name)
  5. Verify tokenizer encodes and decodes a test string
  6. Record: actual_vram_gb, load_time_s, tokenizer_vocab_size
```

Pass: model loads, tokenizer works, VRAM within budget.
Drop: model fails to load, VRAM exceeds budget with no quantization path.

### Check 2: Byte-Table Health (causal/hybrid teachers only)

```
For each teacher with has_kl=True or has_align=True:
  1. Build byte_table = build_token_byte_table(tokenizer)
  2. Count tokens with valid first-byte mapping
  3. Compute coverage = len(byte_table) / tokenizer.vocab_size
  4. Run first_byte_marginal on 10 diverse prompts
  5. Verify output distributions sum to ~1.0
  6. Verify top_bytes dtype == uint8, top_probs dtype == float16
  7. Record: byte_table_coverage, marginal_quality
```

Pass: coverage > 50%, marginals are valid distributions.
Drop: coverage < 20% or marginals are degenerate.

### Check 3: Embedding Availability (semantic teachers only)

```
For each teacher with has_semantic=True:
  1. Load model via AutoModel (not AutoModelForCausalLM)
  2. Verify model.get_input_embeddings() returns valid weight tensor
  3. Verify embedding shape matches spec.hidden_dim
  4. Verify embeddings are not all-zero or NaN
  5. Record: embedding_shape, embedding_norm_stats
```

Pass: embeddings exist with correct dimension.
Drop: model has no accessible embeddings or wrong dimension.

### Check 4: Forward Pass Sanity

```
For each teacher:
  test_input = "The quick brown fox jumps over the lazy dog."
  
  If has_kl:
    1. Tokenize test_input, run model forward
    2. Verify logits shape [1, seq_len, vocab_size]
    3. Verify logits are finite (no NaN/Inf)
    4. Compute first-byte marginal from last position
    5. Verify marginal has non-trivial entropy (> 0.5 bits)
  
  If has_semantic and not has_kl:
    1. Tokenize test_input, run model forward
    2. Verify output has last_hidden_state or equivalent
    3. Verify hidden states are finite
    4. Verify hidden_dim matches spec
```

Pass: forward pass produces valid outputs matching spec.
Drop: forward pass crashes, produces degenerate outputs, or mismatches spec.

### Check 5: Throughput Estimate

```
For each teacher:
  1. Run 20 forward passes on seq_len=2048 inputs
  2. Measure tokens/second
  3. Estimate time_per_shard = (seq_len * n_seqs_per_shard) / throughput
  4. Estimate cache_build_time = time_per_shard * 50 (pilot shards)
  5. Record: throughput_tok_per_s, estimated_cache_hours
```

Pass: estimated cache build < 24 hours for pilot (50 shards).
Drop: throughput too low for practical cache construction.

### Check 6: Cache Yield Estimate

```
For each teacher:
  1. Run on 2 shards with student position manifest
  2. Count KL records generated vs positions available
  3. Count align records generated (if applicable)
  4. Compute yield = records_generated / positions_available
  5. Record: kl_yield, align_yield, empty_position_fraction
```

Pass: KL yield > 80% for causal teachers, align yield > 50%.
Drop: yield < 30% (teacher cannot produce useful records at most positions).

## Teacher Roster

| ID | Name | HF ID | Family | Checks | VRAM |
|----|------|-------|--------|--------|------|
| 0 | t0_anchor_decoder | anchor-decoder-1.7B | Decoder | 1-6 | 3.4 GB |
| 1 | t1_diversity_hybrid | diversity-hybrid-1.2B | Hybrid | 1-6 | 2.4 GB |
| 2 | t2_control_decoder | control-decoder-0.6B | Decoder | 1-6 | 1.2 GB |
| 3 | t3_semantic_embedding | semantic-embedding-300M | Embedding | 1,3,4,5 | 0.6 GB |
| 4 | t4_diversity_ssm | diversity-ssm-780M | SSM | 1,2,4,5,6 | 1.6 GB |

## Admission Rules

A teacher is admitted to E2 cache building if:
1. All applicable checks pass
2. Total roster VRAM (peak single-teacher) fits in GPU budget
3. At least 3 teachers pass (minimum for meaningful routing/purification)
4. At least 1 non-decoder teacher passes (architecture diversity requirement)

A teacher is dropped when:
1. Any check fails with no viable fix
2. Throughput makes cache construction impractical
3. Byte-table coverage is too low for reliable KL records

## Execution Order

1. Run checks 1-4 for all 5 teachers (fast, mostly validation)
2. Record pass/fail per teacher
3. Drop any teacher that fails checks 1-4
4. Run checks 5-6 only for teachers that passed 1-4
5. Produce final admit/drop table
6. If fewer than 3 teachers pass, investigate fixes or replacements

## Output Artifact

```
e2_teacher_feasibility.json
{
  "timestamp": "...",
  "teachers": {
    "t0_anchor_decoder": {
      "status": "admitted" | "dropped",
      "checks": { "1_load": true, "2_byte_table": true, ... },
      "metrics": { "vram_gb": 3.1, "throughput_tok_s": 850, ... },
      "notes": ""
    },
    ...
  },
  "roster_summary": {
    "admitted": 5,
    "dropped": 0,
    "diversity_check": true
  }
}
```
