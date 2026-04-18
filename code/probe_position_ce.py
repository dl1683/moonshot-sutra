"""Probe: CE by position-in-window. Does context help?

If CE drops as position increases (more context), model uses context.
If CE is flat across positions, model is near-unigram and context wasted.
Tests rbor_v1b best.pt only (true mean BPB ~1.411, representative).
"""
import sys
import math
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn.functional as F
import numpy as np
from sutra_dyad import SutraDyadS1, ByteShardedDataset, DEVICE, DTYPE

CKPT = "results/checkpoints_rbor_v1b/best.pt"
N_BATCHES = 80
SEQ_BYTES = 1536
N_BUCKETS = 16  # Bucket the 1536 positions into 16 buckets of 96 each

print(f"Loading {CKPT}...")
ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
cfg = ckpt.get("config", {})
seq_bytes = cfg.get("seq_bytes", SEQ_BYTES)

model = SutraDyadS1(max_seq_bytes=seq_bytes)
model.load_state_dict(ckpt["model"], strict=False)
model = model.to(DEVICE)
model.eval()

dataset = ByteShardedDataset()

# Bucket-summed CE and counts
bucket_ce = np.zeros(N_BUCKETS, dtype=np.float64)
bucket_cnt = np.zeros(N_BUCKETS, dtype=np.int64)

t0 = time.time()
with torch.no_grad():
    for batch_i in range(N_BATCHES):
        x, y = dataset.sample_batch(8, seq_bytes, device=DEVICE, split='test')
        with torch.amp.autocast('cuda', dtype=DTYPE):
            logits, _ = model(x, return_internals=True)
        log_probs = F.log_softmax(logits.float(), dim=-1)
        valid_mask = (y >= 0)
        y_safe = y.clone()
        y_safe[~valid_mask] = 0
        ce = -log_probs.gather(-1, y_safe.unsqueeze(-1)).squeeze(-1)  # (B, T)
        ce = ce.masked_fill(~valid_mask, 0.0)
        valid_f = valid_mask.float()

        # Bucket by position
        bucket_size = seq_bytes // N_BUCKETS
        for b in range(N_BUCKETS):
            start = b * bucket_size
            end = start + bucket_size if b < N_BUCKETS - 1 else seq_bytes
            slice_ce = ce[:, start:end]
            slice_mask = valid_f[:, start:end]
            bucket_ce[b] += float(slice_ce.sum().item())
            bucket_cnt[b] += int(slice_mask.sum().item())

elapsed = time.time() - t0
print(f"Done in {elapsed:.1f}s, {N_BATCHES} batches, seq_bytes={seq_bytes}, {N_BUCKETS} buckets")

print(f"\n{'bucket':>7s}  {'pos_range':>15s}  {'n_valid':>10s}  {'mean_CE':>9s}  {'mean_BPB':>9s}")
for b in range(N_BUCKETS):
    bucket_size = seq_bytes // N_BUCKETS
    start = b * bucket_size
    end = start + bucket_size if b < N_BUCKETS - 1 else seq_bytes
    if bucket_cnt[b] > 0:
        mean_ce = bucket_ce[b] / bucket_cnt[b]
        mean_bpb = mean_ce / math.log(2)
    else:
        mean_ce = mean_bpb = 0.0
    print(f"  {b:>5d}  {start:>5d}-{end:>5d}  {bucket_cnt[b]:>10d}  {mean_ce:>9.4f}  {mean_bpb:>9.4f}")

import json
out = {
    "checkpoint": CKPT,
    "n_batches": N_BATCHES,
    "seq_bytes": seq_bytes,
    "n_buckets": N_BUCKETS,
    "buckets": [
        {
            "bucket": b,
            "pos_start": b * (seq_bytes // N_BUCKETS),
            "pos_end": (b+1) * (seq_bytes // N_BUCKETS) if b < N_BUCKETS - 1 else seq_bytes,
            "n_valid": int(bucket_cnt[b]),
            "mean_ce": float(bucket_ce[b] / bucket_cnt[b]) if bucket_cnt[b] > 0 else 0.0,
            "mean_bpb": float((bucket_ce[b] / bucket_cnt[b]) / math.log(2)) if bucket_cnt[b] > 0 else 0.0,
        }
        for b in range(N_BUCKETS)
    ],
}
Path("results/probe_position_ce.json").write_text(json.dumps(out, indent=2))
print(f"\nSaved to results/probe_position_ce.json")

# Conclusion
first_bpb = out["buckets"][0]["mean_bpb"]
last_bpb = out["buckets"][-1]["mean_bpb"]
delta = first_bpb - last_bpb
print(f"\n=== INTERPRETATION ===")
print(f"  First bucket BPB: {first_bpb:.4f}")
print(f"  Last bucket BPB:  {last_bpb:.4f}")
print(f"  Delta (first - last):  {delta:+.4f} BPB")
if delta > 0.05:
    print(f"  CONTEXT HELPS — model meaningfully uses context (later positions {delta*1000:.0f} mBPB lower)")
elif abs(delta) < 0.02:
    print(f"  CONTEXT NEAR-FLAT — model is mostly position-independent (no big context use)")
else:
    print(f"  WEAK CONTEXT EFFECT — small delta {delta:+.4f}")
