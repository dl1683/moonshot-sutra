# GPU Launch Checklist — S0 + E1

Pre-flight for when GPU becomes available. Execute sequentially.

## Phase 1: S0 Burn-In (500 steps, ~15 min)

### Prerequisites
- [ ] Data shards present: `data/shards_bytes_full/*.bin` (500+ shards expected)
- [ ] No other GPU process running (`nvidia-smi` shows free VRAM)
- [ ] Checkpoint dir exists: `C:/sutra_fast/checkpoints/s0/`
- [ ] OneDrive NOT syncing checkpoint dir (use C:/sutra_fast, not repo path)

### Launch
```bash
cd code/
python s0_training.py \
    --data-dir ../data/shards_bytes_full \
    --checkpoint-dir C:/sutra_fast/checkpoints/s0 \
    --total-steps 500 \
    --warmup-steps 100 \
    --eval-every 50
```

### Success Criteria (from SE1_CANONICAL_SPEC.md R11)
- BPB 3.5-5.0 at step 500: **GO** — proceed to full S0
- BPB 5.0-6.0: **CONDITIONAL GO** — extend burn-in to 1500 steps
- BPB > 7.0 or NaN: **NO-GO** — diagnose (see spec Section 15)
- Per-position accuracy: position 0 must show > 0.03 accuracy

### Verdict
```bash
python burnin_verdict.py --log logs/s0_train.jsonl
```

## Phase 2: Full S0 Training (50K steps, ~12-18 hours)

### Prerequisites
- [ ] Burn-in verdict is GO
- [ ] Resume from burn-in checkpoint (do NOT restart)

### Launch
```bash
python s0_training.py \
    --data-dir ../data/shards_bytes_full \
    --checkpoint-dir C:/sutra_fast/checkpoints/s0 \
    --total-steps 50000 \
    --warmup-steps 1000 \
    --eval-every 500 \
    --resume-from C:/sutra_fast/checkpoints/s0/s0_step500.pt
```

### Milestones
| Steps | BPB Target | Action if Missed |
|-------|-----------|-----------------|
| 5K | <= 3.6 | Investigate LR, data |
| 10K | <= 3.1 | Check train/eval gap |
| 25K | <= 2.6 | Consider early stop |
| 50K | <= 2.3 | S0 complete |

### Stop Conditions
- Eval BPB worsens across two consecutive milestone windows
- No >= 0.1 BPB improvement over 5K steps after 25K
- Train/eval gap > 1.0 BPB

### Output
Best checkpoint: `C:/sutra_fast/checkpoints/s0/s0_best.pt`

## Phase 3: E1 Cache Building (~2-4 hours)

### Prerequisites
- [ ] S0 training complete, best checkpoint available
- [ ] Teacher model downloadable: anchor decoder teacher 1.7B (~3.4 GB VRAM)
- [ ] Combined VRAM: S0 (0.25 GB eval) + teacher (3.4 GB) + logits < 24 GB

### VRAM Budget
```
S0 student (eval mode, bf16):    ~0.25 GB
Anchor decoder teacher (bf16):   ~3.4 GB
Teacher logits + activation:     ~2-4 GB
Overhead:                        ~1 GB
Total:                           ~7-9 GB (fits easily)
```

### Launch
```bash
cd code/
python eklavya_cache.py \
    --teacher anchor-decoder-1.7B \
    --data-dir ../data/shards_bytes_full \
    --output-dir C:/sutra_fast/eklavya_cache \
    --student-checkpoint C:/sutra_fast/checkpoints/s0/s0_best.pt \
    --max-shards 50
```

### Output Files
```
C:/sutra_fast/eklavya_cache/
  align_records.bin       ~500 MB (20M records)
  kl_records.bin          ~1.6 GB (20M records)
  teacher_embeddings.pt   ~622 MB (151K tokens * 2048 * fp16)
  cache_manifest.json     metadata
```

### Validation
- [ ] cache_manifest.json shows n_align > 0 and n_kl > 0
- [ ] align_records.bin and kl_records.bin are non-empty
- [ ] teacher_embeddings.pt loads without error

## Phase 4: E1 Training (12K steps, ~6-10 hours)

### Prerequisites
- [ ] S0 best checkpoint exists
- [ ] Cache built and validated (Phase 3)
- [ ] Sufficient disk for E1 checkpoints (~2 GB per checkpoint)

### VRAM Budget
```
S0 student (train mode, bf16):   ~3.2 GB (params + optimizer + grads)
Activations (batch=4, accum=2):  ~4-6 GB
AlignProjection:                 ~0.01 GB
Embedding table (bf16):          ~0.6 GB
Cache record tensors:            ~0.5 GB
Total:                           ~8-10 GB (fits within 24 GB)
```

### Launch
```bash
cd code/
python eklavya_training.py \
    --student-checkpoint C:/sutra_fast/checkpoints/s0/s0_best.pt \
    --cache-dir C:/sutra_fast/eklavya_cache \
    --output-dir C:/sutra_fast/checkpoints/e1 \
    --data-dir ../data/shards_bytes_full \
    --steps 12000
```

### Phase Schedule
| Phase | Steps | What Trains | Losses |
|-------|-------|-------------|--------|
| E1.0 | 0-499 | align_proj only | L_align |
| E1.1 | 500-1999 | encoder + align_proj | CE + L_align |
| E1.2 | 2000-11999 | all params | CE + L_align + L_kl |

### Monitoring
- BPB should NOT regress during E1.0 (student frozen)
- BPB may wobble slightly during E1.1 (encoder unfreezing)
- L_align should decrease monotonically in E1.0
- L_kl should activate and decrease in E1.2
- Cache refreshes at steps 2000, 4000, 6000, 8000, 10000

### Success Criteria (from E1 Protocol)
- E1 (C4) > CE-only continuation (C1) on eval BPB
- Shuffled targets (C5/C6) fail or underperform real targets
- BPB does not regress materially from S0 baseline
- Byte accuracy improves on high-NLL slices

## Phase 5: Ablation Controls (C0-C8)

Run AFTER main E1 to validate the result is real.

| Control | Description | Purpose |
|---------|-------------|---------|
| C0 | S0 checkpoint, no continuation | Baseline |
| C1 | CE-only continuation (12K steps) | Shows KD adds value |
| C2 | CE + align only | Isolates alignment signal |
| C3 | CE + KL only | Isolates KL signal |
| C4 | CE + align + KL | Main E1 (already done) |
| C5 | CE + shuffled align | Falsification |
| C6 | CE + shuffled KL | Falsification |
| C7 | CE + align + KL, no refresh | Tests refresh value |
| C8 | CE + align + KL, with refresh | Tests refresh value |

Priority: C0, C1, C4 first. Then C5/C6 (falsification). Then C2/C3/C7/C8.

## Quick Reference: File Locations

| What | Where |
|------|-------|
| S0 architecture | `code/s0_architecture.py` |
| S0 training loop | `code/s0_training.py` |
| S0 configs (P4/P8/D640/D768) | `code/s0_configs.py` |
| Burn-in verdict | `code/burnin_verdict.py` |
| E1 cache builder | `code/eklavya_cache.py` |
| E1 training loop | `code/eklavya_training.py` |
| E1 unit tests | `code/test_eklavya.py` |
| E1 protocol spec | `research/EKLAVYA_E1_PROTOCOL.md` |
| Full build spec | `research/SE1_CANONICAL_SPEC.md` |
| Training logs | `logs/s0_train.jsonl`, `logs/e1_train.jsonl` |
| Checkpoints | `C:/sutra_fast/checkpoints/{s0,e1}/` |
| Cache | `C:/sutra_fast/eklavya_cache/` |
| Data shards | `data/shards_bytes_full/*.bin` |
