"""Probe: per-byte CE distribution across session checkpoints.

3 checkpoints have nearly-identical mean BPB (1.406-1.411). Question: do
they distribute their errors differently? Compute per-byte CE histogram
(by byte-class: ASCII alpha / digit / punct / whitespace / other) for each.

If distributions differ, mechanisms ARE doing different things even though
mean BPB is similar. If identical, mechanism interventions are truly null.
"""
import sys
import math
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn.functional as F
from sutra_dyad import SutraDyadS1, ByteShardedDataset, DEVICE, DTYPE

CHECKPOINTS = [
    ("diagnostic", "results/checkpoints_diagnostic_classical_kd/best.pt"),
    ("r2", "results/checkpoints_classical_continue_r2/best.pt"),
    ("rbor_v1b", "results/checkpoints_rbor_v1b/best.pt"),
]
N_BATCHES = 50
SEQ_BYTES = 1536


def classify_byte(b: int) -> str:
    if 65 <= b <= 90 or 97 <= b <= 122:
        return "alpha"
    if 48 <= b <= 57:
        return "digit"
    if b in (32, 9, 10, 13):
        return "ws"
    if 33 <= b <= 126:
        return "punct"
    return "other"


def per_byte_ce(model, dataset, n_batches=50, seq_len=1536):
    """Returns dict: class -> list of per-position CE values."""
    model.eval()
    device = next(model.parameters()).device
    by_class = {"alpha": [], "digit": [], "ws": [], "punct": [], "other": []}
    total_ce = 0.0
    total_count = 0
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = dataset.sample_batch(8, seq_len, device=device, split='test')
            with torch.amp.autocast('cuda', dtype=DTYPE):
                logits, _ = model(x, return_internals=True)
            log_probs = F.log_softmax(logits.float(), dim=-1)
            # CE per position = -log(p[true_byte])
            valid_mask = (y >= 0)
            y_safe = y.clone()
            y_safe[~valid_mask] = 0
            ce_per_pos = -log_probs.gather(-1, y_safe.unsqueeze(-1)).squeeze(-1)
            ce_per_pos = ce_per_pos.masked_fill(~valid_mask, float('nan'))
            # Aggregate per byte class
            y_cpu = y.cpu().numpy()
            ce_cpu = ce_per_pos.cpu().numpy()
            mask_cpu = valid_mask.cpu().numpy()
            for b_idx in range(y_cpu.shape[0]):
                for t_idx in range(y_cpu.shape[1]):
                    if not mask_cpu[b_idx, t_idx]:
                        continue
                    cls = classify_byte(int(y_cpu[b_idx, t_idx]))
                    ce = float(ce_cpu[b_idx, t_idx])
                    by_class[cls].append(ce)
                    total_ce += ce
                    total_count += 1
    model.train()
    return by_class, total_ce / total_count if total_count else 0.0


def summarize(by_class):
    import statistics
    summary = {}
    for cls, ces in by_class.items():
        if not ces:
            summary[cls] = {"n": 0, "mean_ce": 0, "mean_bpb": 0, "std_bpb": 0}
            continue
        mean_ce = statistics.mean(ces)
        std_ce = statistics.stdev(ces) if len(ces) > 1 else 0.0
        summary[cls] = {
            "n": len(ces),
            "mean_ce": mean_ce,
            "mean_bpb": mean_ce / math.log(2),
            "std_bpb": std_ce / math.log(2),
            "n_frac": 0,  # filled below
        }
    total_n = sum(s["n"] for s in summary.values())
    for s in summary.values():
        s["n_frac"] = s["n"] / total_n if total_n else 0.0
    return summary


dataset = ByteShardedDataset()
results = {}

for name, path in CHECKPOINTS:
    print(f"\n{'='*60}")
    print(f"Checkpoint: {name}  ({path})")
    if not Path(path).exists():
        print("  SKIPPED")
        continue

    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    cfg = ckpt.get("config", {})
    seq_bytes = cfg.get("seq_bytes", SEQ_BYTES)

    model = SutraDyadS1(max_seq_bytes=seq_bytes)
    model.load_state_dict(ckpt["model"], strict=False)
    model = model.to(DEVICE)

    t0 = time.time()
    by_class, mean_ce = per_byte_ce(model, dataset, n_batches=N_BATCHES, seq_len=seq_bytes)
    elapsed = time.time() - t0
    summary = summarize(by_class)
    overall_bpb = mean_ce / math.log(2)
    print(f"  Done in {elapsed:.1f}s. Overall BPB={overall_bpb:.4f}")
    print(f"  Per byte-class:")
    for cls in ["alpha", "digit", "ws", "punct", "other"]:
        s = summary[cls]
        print(f"    {cls:8s}: n={s['n']:>7d}  frac={s['n_frac']:.1%}  mean_BPB={s['mean_bpb']:.4f}  std_BPB={s['std_bpb']:.4f}")
    results[name] = {"summary": summary, "overall_bpb": overall_bpb}

    del model
    torch.cuda.empty_cache()

print("\n" + "="*60)
print("CROSS-CHECKPOINT PER-CLASS COMPARISON (mean_bpb)")
print("="*60)
classes = ["alpha", "digit", "ws", "punct", "other"]
print(f"  {'class':8s}  " + "  ".join(f"{n:>10s}" for n in results.keys()))
for cls in classes:
    row = [f"{cls:8s}"]
    for name in results:
        if cls in results[name]["summary"]:
            row.append(f"{results[name]['summary'][cls]['mean_bpb']:>10.4f}")
        else:
            row.append(f"{'N/A':>10s}")
    print("  " + "  ".join(row))

import json
Path("results/probe_perbyte_ce.json").write_text(json.dumps(results, indent=2, default=str))
print(f"\nSaved to results/probe_perbyte_ce.json")
