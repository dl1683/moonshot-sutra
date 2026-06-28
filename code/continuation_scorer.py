"""Continuation scoring benchmark for byte-level models.

Forced-choice scoring: for each eval sequence, the model scores the correct
continuation vs a shuffled wrong continuation. Reports accuracy (fraction where
correct continuation has lower BPB).

Usage:
    python continuation_scorer.py \
        --checkpoint C:/sutra_fast/checkpoints/s0/s0_best.pt \
        --eval-shards data/shards_bytes_full \
        --output ablations/continuation_score.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))

from s0_architecture import SutraS0
from eklavya_training import EklavyaDataset


@torch.no_grad()
def score_continuations(
    student: SutraS0,
    eval_loader: DataLoader,
    device: torch.device,
    split_frac: float = 0.75,
    max_batches: int = 250,
    seed: int = 42,
) -> dict:
    """Score correct vs wrong continuations.

    For each batch pair (A, B), creates paired sequences:
      - correct[k]: A[k] (original)
      - wrong[k]:   A[k][:split] + B[perm[k]][split:]
    All pairs are batched into a single forward pass for efficiency.

    Reports accuracy (correct < wrong BPB on continuation portion).
    """
    student.eval()
    P = student.cfg.patch_size
    rng = np.random.RandomState(seed)

    correct_wins = 0
    total_pairs = 0
    correct_bpbs = []
    wrong_bpbs = []

    prev_batch = None

    for i, batch in enumerate(eval_loader):
        if max_batches > 0 and i >= max_batches:
            break

        byte_ids, _, _ = batch
        byte_ids = byte_ids.to(device)
        B, T = byte_ids.shape
        N = T // P

        split_patch = int(N * split_frac)
        if split_patch < 2 or split_patch >= N - 1:
            continue

        if prev_batch is None:
            prev_batch = byte_ids
            continue

        seq_a = prev_batch
        seq_b = byte_ids
        prev_batch = byte_ids
        n_pairs = min(seq_a.shape[0], seq_b.shape[0])

        all_correct = []
        all_wrong = []
        for k in range(n_pairs):
            perm_idx = rng.randint(0, seq_b.shape[0])
            a_patches = seq_a[k].reshape(N, P)
            b_patches = seq_b[perm_idx].reshape(N, P)
            all_correct.append(a_patches.reshape(-1))
            wrong = torch.cat([a_patches[:split_patch],
                               b_patches[split_patch:]], dim=0)
            all_wrong.append(wrong.reshape(-1))

        stacked = torch.stack(all_correct + all_wrong, dim=0)

        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            out = student(stacked, return_aux=False)

        logits = out["logits"]
        targets = stacked.reshape(2 * n_pairs, N, P)[:, 1:]

        cont_start = split_patch - 1
        cont_end = N - 1
        n_cont = cont_end - cont_start

        cont_logits = logits[:, cont_start:cont_end].reshape(
            2 * n_pairs, -1, 256)
        cont_targets = targets[:, cont_start:cont_end].reshape(
            2 * n_pairs, -1)

        per_seq_loss = F.cross_entropy(
            cont_logits.reshape(-1, 256), cont_targets.reshape(-1),
            reduction="none").reshape(2 * n_pairs, -1).mean(dim=1)

        for k in range(n_pairs):
            bpb_c = per_seq_loss[k].item() / math.log(2)
            bpb_w = per_seq_loss[n_pairs + k].item() / math.log(2)
            correct_bpbs.append(bpb_c)
            wrong_bpbs.append(bpb_w)
            if bpb_c < bpb_w:
                correct_wins += 1
            total_pairs += 1

    accuracy = correct_wins / max(total_pairs, 1)
    mean_correct_bpb = sum(correct_bpbs) / max(len(correct_bpbs), 1)
    mean_wrong_bpb = sum(wrong_bpbs) / max(len(wrong_bpbs), 1)

    return {
        "continuation_accuracy": round(accuracy, 4),
        "n_pairs": total_pairs,
        "mean_correct_bpb": round(mean_correct_bpb, 4),
        "mean_wrong_bpb": round(mean_wrong_bpb, 4),
        "bpb_gap": round(mean_wrong_bpb - mean_correct_bpb, 4),
        "split_fraction": split_frac,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Continuation scoring benchmark for byte models")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval-shards", required=True)
    parser.add_argument("--output", default="continuation_score.json")
    parser.add_argument("--ablation-id", default="")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=250)
    parser.add_argument("--split-frac", type=float, default=0.75)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--shard-range", type=int, nargs=2, default=None,
                        metavar=("START", "END"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_cfg = ckpt["model_cfg"]
    student = SutraS0(model_cfg).to(device)
    student.load_state_dict(ckpt["model"])
    student.eval()
    print(f"Loaded from step {ckpt.get('step', '?')}")

    shard_range = tuple(args.shard_range) if args.shard_range else None
    eval_dataset = EklavyaDataset(
        args.eval_shards, args.seq_len, model_cfg.patch_size,
        shard_range=shard_range)
    eval_loader = DataLoader(
        eval_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=1, pin_memory=True, drop_last=True)

    print(f"Scoring continuations (split={args.split_frac}, "
          f"max_batches={args.max_batches})...")
    metrics = score_continuations(
        student, eval_loader, device,
        split_frac=args.split_frac,
        max_batches=args.max_batches,
        seed=args.seed,
    )

    report = {
        "ablation_id": args.ablation_id,
        "checkpoint": args.checkpoint,
        "step": ckpt.get("step", -1),
        "shard_range": list(shard_range) if shard_range else None,
        "metrics": metrics,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved: {args.output}")
    print(f"  Continuation accuracy: {metrics['continuation_accuracy']*100:.1f}%")
    print(f"  Correct BPB: {metrics['mean_correct_bpb']:.3f}")
    print(f"  Wrong BPB: {metrics['mean_wrong_bpb']:.3f}")
    print(f"  BPB gap: {metrics['bpb_gap']:.3f}")
    print(f"  Pairs scored: {metrics['n_pairs']}")


if __name__ == "__main__":
    main()
