"""Probe: measure eval-batch variance for rbor_v1b best.pt.

Runs 5 independent eval_loss() calls (same n_batches=50 as training's
in-loop eval). Reports mean, std, range. Answers: is the 'ceiling' at 1.41
a real signal or within eval-batch noise?
"""
import sys
import math
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
from sutra_dyad import (
    SutraDyadS1, ByteShardedDataset, eval_loss,
    DEVICE, DTYPE, SEQ_BYTES,
)

CKPT = "results/checkpoints_rbor_v1b/best.pt"
N_EVALS = 5
N_BATCHES = 50  # matches training's eval_every block

print(f"Loading {CKPT}...")
ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
cfg = ckpt.get("config", {})
seq_bytes = cfg.get("seq_bytes", 1536)

model = SutraDyadS1(max_seq_bytes=seq_bytes)
model.load_state_dict(ckpt["model"], strict=False)
model = model.to(DEVICE)
model.eval()
print(f"Loaded. Step {ckpt.get('step')}, stored eval_loss={ckpt.get('eval_loss'):.4f}")

dataset = ByteShardedDataset()

print(f"\nRunning {N_EVALS} independent evals (n_batches={N_BATCHES}, seq={seq_bytes})...")
results = []
for i in range(N_EVALS):
    t0 = time.time()
    el = eval_loss(model, dataset, n_batches=N_BATCHES, seq_len=seq_bytes)
    bpb = el / math.log(2)
    elapsed = time.time() - t0
    print(f"  eval {i+1}: loss={el:.4f}  BPB={bpb:.4f}  ({elapsed:.1f}s)")
    results.append(bpb)

import statistics
mean_bpb = statistics.mean(results)
std_bpb = statistics.stdev(results) if len(results) > 1 else 0.0
min_bpb = min(results)
max_bpb = max(results)
rng = max_bpb - min_bpb

print(f"\n=== Noise stats (BPB, 50-batch eval, n={N_EVALS}) ===")
print(f"  mean:  {mean_bpb:.4f}")
print(f"  std:   {std_bpb:.4f}")
print(f"  min:   {min_bpb:.4f}")
print(f"  max:   {max_bpb:.4f}")
print(f"  range: {rng:.4f}")
print(f"\nTraining eval reported: 1.4075 (stored best)")
print(f"r3 regression was: 1.407 -> 1.439 (delta 0.032)")
print(f"Is r3 regression within this noise range? {'YES' if rng >= 0.032 else 'NO'}")

# Save results
import json
out = {
    "checkpoint": CKPT,
    "n_evals": N_EVALS,
    "n_batches": N_BATCHES,
    "per_eval_bpb": results,
    "mean_bpb": mean_bpb,
    "std_bpb": std_bpb,
    "min_bpb": min_bpb,
    "max_bpb": max_bpb,
    "range_bpb": rng,
    "r3_delta": 0.032,
    "r3_within_noise": bool(rng >= 0.032),
}
Path("results/probe_eval_noise.json").write_text(json.dumps(out, indent=2))
print(f"\nSaved to results/probe_eval_noise.json")
