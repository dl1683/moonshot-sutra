"""Probe: precise eval BPB estimates for session's key checkpoints.

Runs eval_loss() with n_batches=200 (4x training eval) on each session
checkpoint, 3 independent runs each. Produces tight mean+std estimate per
checkpoint so we can rank reliably beyond single-eval noise.
"""
import sys
import math
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
from sutra_dyad import SutraDyadS1, ByteShardedDataset, eval_loss, DEVICE

CHECKPOINTS = [
    ("diagnostic_1.381", "results/checkpoints_diagnostic_classical_kd/best.pt"),
    ("r2_1.409", "results/checkpoints_classical_continue_r2/best.pt"),
    ("rbor_v1b_1.407", "results/checkpoints_rbor_v1b/best.pt"),
]
N_BATCHES = 200
N_RUNS = 3

dataset = ByteShardedDataset()
results = {}

for name, path in CHECKPOINTS:
    print(f"\n{'='*60}")
    print(f"Checkpoint: {name}  ({path})")
    if not Path(path).exists():
        print("  SKIPPED (not found)")
        continue

    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    cfg = ckpt.get("config", {})
    seq_bytes = cfg.get("seq_bytes", 1536)

    model = SutraDyadS1(max_seq_bytes=seq_bytes)
    model.load_state_dict(ckpt["model"], strict=False)
    model = model.to(DEVICE)
    model.eval()
    print(f"  Loaded. Stored eval_loss={ckpt.get('eval_loss', 0):.4f} nats "
          f"({ckpt.get('eval_loss', 0)/math.log(2):.4f} BPB)")

    run_bpbs = []
    for i in range(N_RUNS):
        t0 = time.time()
        el = eval_loss(model, dataset, n_batches=N_BATCHES, seq_len=seq_bytes)
        bpb = el / math.log(2)
        elapsed = time.time() - t0
        print(f"  run {i+1}: BPB={bpb:.4f}  ({elapsed:.1f}s)")
        run_bpbs.append(bpb)

    import statistics
    mean_bpb = statistics.mean(run_bpbs)
    std_bpb = statistics.stdev(run_bpbs) if len(run_bpbs) > 1 else 0.0
    results[name] = {
        "path": path,
        "per_run_bpb": run_bpbs,
        "mean_bpb": mean_bpb,
        "std_bpb": std_bpb,
        "stored_bpb": ckpt.get("eval_loss", 0) / math.log(2),
    }
    print(f"  SUMMARY: mean={mean_bpb:.4f}  std={std_bpb:.4f}  stored={results[name]['stored_bpb']:.4f}")

    del model
    torch.cuda.empty_cache()

print("\n" + "="*60)
print("FINAL RANKING (by precise mean BPB, lower is better):")
print("="*60)
for name, data in sorted(results.items(), key=lambda x: x[1]["mean_bpb"]):
    delta_stored = data["mean_bpb"] - data["stored_bpb"]
    print(f"  {name}:  precise={data['mean_bpb']:.4f} ± {data['std_bpb']:.4f}  "
          f"(stored={data['stored_bpb']:.4f}, Δ={delta_stored:+.4f})")

import json
Path("results/probe_precise_eval.json").write_text(json.dumps(results, indent=2))
print(f"\nSaved to results/probe_precise_eval.json")
