# GPU Launch Checklist — S0 → E1 → E2

Pre-flight for when GPU becomes available. Execute sequentially.

**During E2 training**, consult [E2 Monitoring Protocol](E2_MONITORING_PROTOCOL.md)
for phase-specific intervention rules, gradient budget red flags, and GPU failure modes.

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

### Success Criteria
- BPB 3.5-5.0 at step 500: **GO**
- BPB 5.0-6.0: **CONDITIONAL GO** — extend burn-in to 1500 steps
- BPB > 7.0 or NaN: **NO-GO** — diagnose

### Verdict
```bash
python burnin_verdict.py --log logs/s0_train.jsonl
```

## Phase 2: Full S0 Training (50K steps, ~12-18 hours)

### Prerequisites
- [ ] Burn-in verdict is GO
- [ ] Resume from burn-in checkpoint

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
- [ ] S0 best checkpoint exists
- [ ] Anchor decoder teacher downloadable (~3.4 GB VRAM)
- [ ] Combined VRAM: S0 (0.25 GB eval) + teacher (3.4 GB) + logits < 24 GB

### Launch
```bash
python eklavya_cache.py \
    --teacher anchor-decoder-1.7B \
    --data-dir ../data/shards_bytes_full \
    --output-dir C:/sutra_fast/eklavya_cache \
    --student-checkpoint C:/sutra_fast/checkpoints/s0/s0_best.pt \
    --max-shards 50
```

### Validation
- [ ] cache_manifest.json shows n_align > 0 and n_kl > 0
- [ ] align_records.bin and kl_records.bin are non-empty
- [ ] teacher_embeddings.pt loads without error

## Phase 4: E1 Training (12K steps, ~6-10 hours)

### Prerequisites
- [ ] S0 best checkpoint exists
- [ ] E1 cache built and validated (Phase 3)

### Launch
```bash
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

### Success Criteria
- E1 > CE-only continuation on eval BPB
- Shuffled targets fail or underperform
- BPB does not regress from S0 baseline

## Phase 5: E2 Cache Building (~4-8 hours)

### Prerequisites
- [ ] E1 best checkpoint exists with decisive KD gains (>2pp over CE-only)
- [ ] All 5 teachers downloadable (total ~9.2 GB VRAM during cache build)
- [ ] Output dir: `C:/sutra_fast/eklavya_e2_cache/`

### VRAM Budget (Cache Build — One Teacher at a Time)
```
S0 student (eval mode, bf16):    ~0.25 GB
Largest teacher (anchor, bf16):  ~3.4 GB
Logits + activation:             ~2-4 GB
Total peak:                      ~6-8 GB (fits easily)
```

### Launch — Two-Pass Build
```bash
# Pass 1: generate position manifest (student-only, no teacher needed)
python eklavya_e2_cache_builder.py \
    --student-checkpoint C:/sutra_fast/checkpoints/e1/e1_best.pt \
    --data-dir ../data/shards_bytes_full \
    --output-dir C:/sutra_fast/eklavya_e2_cache \
    --max-shards 50 \
    --positions-only

# Pass 2: fill teacher records (one teacher at a time)
python eklavya_e2_cache_builder.py \
    --student-checkpoint C:/sutra_fast/checkpoints/e1/e1_best.pt \
    --data-dir ../data/shards_bytes_full \
    --output-dir C:/sutra_fast/eklavya_e2_cache \
    --max-shards 50 \
    --teachers-only
```

### Expected Output
```
C:/sutra_fast/eklavya_e2_cache/
  manifest.json
  teacher_registry.json
  positions.bin                 # shared position manifest
  teachers/
    t0_anchor_decoder/
      kl_records.bin
      align_records.bin
      teacher_embeddings.pt
    t1_diversity_hybrid/
      ...
    t2_control_decoder/
      ...
    t3_semantic_embedding/
      ...
    t4_diversity_ssm/
      ...
```

### Validation
- [ ] manifest.json shows n_positions > 0 and teacher_count == 5
- [ ] Each teacher subdir has non-empty records
- [ ] `python -c "from eklavya_e2_cache import E2CacheView; v=E2CacheView('C:/sutra_fast/eklavya_e2_cache'); print(v.manifest); v.close()"`

## Phase 6: E2 Training (~12-24 hours)

### Prerequisites
- [ ] E1 best checkpoint exists
- [ ] E2 cache built and validated (Phase 5)
- [ ] All tests passing (`pytest code/ -x`)

### VRAM Budget (Training)
```
S0 student (train mode, bf16):   ~3.2 GB
Activations (batch=4, accum=2):  ~4-6 GB
MultiTeacherProjectionPorts:     ~0.1 GB
Embedding tables (all teachers): ~2.5 GB
Cache mmap (disk-backed):        ~0 GB RAM
Total:                           ~10-12 GB (fits within 24 GB)
```

### Launch — Main E2 (Ablation A2)
```bash
python eklavya_e2_training.py \
    --student-checkpoint C:/sutra_fast/checkpoints/e1/e1_best.pt \
    --cache-dir C:/sutra_fast/eklavya_e2_cache \
    --output-dir C:/sutra_fast/checkpoints/e2 \
    --ablation-id A2
```

### Phase Schedule (default config, 13750 total steps)
| Phase | Steps | Description | Active Teachers |
|-------|-------|-------------|----------------|
| E2.1 | 0-749 | Projection-port warmup (750 steps) | Anchor only |
| E2.2 | 750-2749 | Low-conflict consensus (2000 steps) | Anchor + Control |
| E2.3 | 2750-5749 | Semantic geometry landing (3000 steps) | Anchor + Control + Semantic |
| E2.4 | 5750-13749 | Full disagreement-routed KD (8000 steps) | All admitted teachers |

### Monitoring
- BPB should NOT regress during E2.1 (student frozen)
- `teacher_losses_bits` in JSONL log — all values in bits for comparison with BPB
- `teacher_losses_nats` in JSONL log — raw values for gradient analysis
- Console prints teacher losses in bits with `(bits)` label

### Resume from Checkpoint
```bash
python eklavya_e2_training.py \
    --student-checkpoint C:/sutra_fast/checkpoints/e1/e1_best.pt \
    --cache-dir C:/sutra_fast/eklavya_e2_cache \
    --output-dir C:/sutra_fast/checkpoints/e2 \
    --ablation-id A2 \
    --resume C:/sutra_fast/checkpoints/e2/e2_step10000.pt
```

## Phase 7: E2 Ablation Controls

Run AFTER main E2 to validate multi-teacher value. Two-phase strategy:
Phase 1 (feasibility) must pass before Phase 2 (publishability) runs.

### Phase 1: Feasibility (does multi-teacher help at all?)

| Priority | ID | Command Flags | Question |
|----------|-----|---------------|----------|
| 1 | A2 | (default) | Full system (oracle router) |
| 2 | A0 | `--ce-only --steps 8000` | Does E2 beat doing nothing? |
| 3 | BLD | `--bld-mode --steps 8000` | Does E2 beat raw byte KL? |
| 4 | A1 | `--teachers t0_anchor_decoder` | Does multi-teacher beat single? |

**Stop if A2 fails any Phase 1 comparison.** A2 is an oracle-aided upper bound.

### Phase 2: Publishability (does the deployable system beat static mixing?)

| Priority | ID | Command Flags | Question |
|----------|-----|---------------|----------|
| 5 | A9c | `--router-mode gold_free_student_jsd` | Full gold-free router |
| 6 | A5b | `--disable-router --static-weight-mode custom --static-weights "..."` | Does routing beat tuned static? |
| 7 | A5a | `--disable-router --static-weight-mode prior` | Does routing beat prior-weighted? |
| 8 | A5c | `--disable-router --static-weight-mode prior --teachers t0 t1` | Does 5-teacher routed beat 2-teacher? |
| 9 | A7 | `--disable-gradient-budget` | Does gradient budgeting help? |
| 10 | A8 | `--no-phased-admission` | Does phased admission help? |
| 11 | A6 | `--shuffle-teacher-targets` | Sanity/falsification |

### 48-Hour Minimum Plan (8K steps each)

```bash
# Phase 1: A2, A0, BLD, A1
python eklavya_e2_training.py \
    --student-checkpoint C:/sutra_fast/checkpoints/e1/e1_best.pt \
    --cache-dir C:/sutra_fast/eklavya_e2_cache \
    --output-dir C:/sutra_fast/checkpoints/e2_a2 \
    --ablation-id A2 --steps 8000

python eklavya_e2_training.py \
    --student-checkpoint C:/sutra_fast/checkpoints/e1/e1_best.pt \
    --cache-dir C:/sutra_fast/eklavya_e2_cache \
    --output-dir C:/sutra_fast/checkpoints/e2_a0 \
    --ablation-id A0 --ce-only --steps 8000

python eklavya_e2_training.py \
    --student-checkpoint C:/sutra_fast/checkpoints/e1/e1_best.pt \
    --cache-dir C:/sutra_fast/eklavya_e2_cache \
    --output-dir C:/sutra_fast/checkpoints/e2_bld \
    --ablation-id BLD --bld-mode --steps 8000

python eklavya_e2_training.py \
    --student-checkpoint C:/sutra_fast/checkpoints/e1/e1_best.pt \
    --cache-dir C:/sutra_fast/eklavya_e2_cache \
    --output-dir C:/sutra_fast/checkpoints/e2_a1 \
    --ablation-id A1 --teachers t0_anchor_decoder --steps 8000

# Phase 2 (only if Phase 1 passes): A9c, A5b
python eklavya_e2_training.py \
    --student-checkpoint C:/sutra_fast/checkpoints/e1/e1_best.pt \
    --cache-dir C:/sutra_fast/eklavya_e2_cache \
    --output-dir C:/sutra_fast/checkpoints/e2_a9c \
    --ablation-id A9c --router-mode gold_free_student_jsd --steps 8000

python eklavya_e2_training.py \
    --student-checkpoint C:/sutra_fast/checkpoints/e1/e1_best.pt \
    --cache-dir C:/sutra_fast/eklavya_e2_cache \
    --output-dir C:/sutra_fast/checkpoints/e2_a5b \
    --ablation-id A5b --disable-router --static-weight-mode custom \
    --static-weights "t0_anchor_decoder:0.45,t1_diversity_hybrid:0.25,t2_control_decoder:0.15,t4_diversity_ssm:0.15" \
    --steps 8000

# A5c if time permits
python eklavya_e2_training.py \
    --student-checkpoint C:/sutra_fast/checkpoints/e1/e1_best.pt \
    --cache-dir C:/sutra_fast/eklavya_e2_cache \
    --output-dir C:/sutra_fast/checkpoints/e2_a5c \
    --ablation-id A5c --disable-router --static-weight-mode prior \
    --teachers t0_anchor_decoder t1_diversity_hybrid --steps 8000
```

### Evaluation
```bash
python eval_e2.py \
    --checkpoint C:/sutra_fast/checkpoints/e2_a2/e2_best.pt \
    --eval-shards data/shards_bytes_full \
    --ablation-id A2 \
    --cache-dir C:/sutra_fast/eklavya_e2_cache \
    --output ablations/a2_full.json
```

## Quick Reference: File Locations

| What | Where |
|------|-------|
| S0 architecture | `code/s0_architecture.py` |
| S0 training loop | `code/s0_training.py` |
| S0 configs | `code/s0_configs.py` |
| Burn-in verdict | `code/burnin_verdict.py` |
| E1 cache builder | `code/eklavya_cache.py` |
| E1 training loop | `code/eklavya_training.py` |
| E1 unit tests | `code/test_eklavya.py` (28 tests) |
| E2 cache builder | `code/eklavya_e2_cache_builder.py` |
| E2 training loop | `code/eklavya_e2_training.py` |
| E2 router/purifier | `code/eklavya_e2_router.py` |
| E2 losses/ports | `code/eklavya_e2_losses.py` |
| E2 evaluator | `code/eval_e2.py` |
| E2 unit tests | `code/test_eklavya_e2.py` (302 tests) |
| S0 tests | `code/test_overfit.py` (16 tests) |
| E1 protocol | `research/EKLAVYA_E1_PROTOCOL.md` |
| E2 protocol | `research/EKLAVYA_E2_PROTOCOL.md` |
| E2 monitoring | `research/E2_MONITORING_PROTOCOL.md` |
| Opsec checker | `code/check_opsec.py` |
| Ablation comparison | `code/compare_ablations.py` |
| Log CSV export | `code/export_log_csv.py` |
| Checkpoint inspector | `code/inspect_checkpoint.py` |
| Training logs | `logs/*.jsonl` |
| Checkpoints | `C:/sutra_fast/checkpoints/{s0,e1,e2}/` |
| E1 cache | `C:/sutra_fast/eklavya_cache/` |
| E2 cache | `C:/sutra_fast/eklavya_e2_cache/` |
| Data shards | `data/shards_bytes_full/*.bin` |

## Critical Operational Notes

1. **OneDrive stall prevention**: ALL checkpoints go to `C:/sutra_fast/`, never to repo path under OneDrive
2. **Telemetry units**: JSONL logs emit `teacher_losses_bits` (for comparison with BPB) and `teacher_losses_nats` (raw gradient-scale values)
3. **Mmap cache**: E2 trainer uses `E2CacheView` (memory-mapped). Record data stays on disk; only accessed records unpack at runtime. **Index RAM**: pilot (50K positions, 5 teachers) ~28 MB; production (10M positions, 5 teachers) ~5.7 GB. Check with `estimate_index_memory()` before building full-scale cache
4. **GradScaler safety**: PORT_WARMUP phase may produce zero backward passes on some batches; the trainer handles this gracefully
5. **Checkpoint resume**: CPU, CUDA, Python, and NumPy RNG states are all restored on resume for full reproducibility
