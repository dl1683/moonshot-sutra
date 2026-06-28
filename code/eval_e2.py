"""Eklavya E2 — Frozen checkpoint evaluator.

Pure frozen-weight BPB evaluator. Ablation identity comes from checkpoint
provenance (which training config produced it), not from eval-time flags.

Usage:
    python eval_e2.py \
      --checkpoint checkpoints/e2/e2_best.pt \
      --eval-shards data/shards_bytes_full \
      --ablation-id A2 \
      --output ablations/a2_full.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))

from s0_architecture import SutraS0
from eklavya_training import EklavyaDataset
from eklavya_e2_cache import (
    PositionRecord, SelectionReason,
    read_position_manifest,
)

BYTE_CATEGORIES = {
    "digit": set(range(48, 58)),
    "lowercase": set(range(97, 123)),
    "uppercase": set(range(65, 91)),
    "whitespace": {9, 10, 13, 32},
    "punctuation": (set(range(33, 48)) | set(range(58, 65))
                    | set(range(91, 97)) | set(range(123, 127))),
    "utf8_cont": set(range(128, 192)),
    "utf8_lead": set(range(192, 256)),
    "control": set(range(0, 9)) | {11, 12} | set(range(14, 32)) | {127},
}

_BYTE_TO_CAT = {}
for _cat, _bytes in BYTE_CATEGORIES.items():
    for _b in _bytes:
        _BYTE_TO_CAT[_b] = _cat


@dataclass
class EvalConfig:
    ablation_id: str = "A2"
    run_label: str = ""
    eval_shards: str = "data/shards_bytes_full"
    seq_len: int = 4096
    batch_size: int = 4
    max_eval_batches: int = 0
    output: str = "ablation_result.json"
    cache_dir: str | None = None


@torch.no_grad()
def evaluate_bpb(
    student: SutraS0,
    eval_loader: DataLoader,
    device: torch.device,
    max_batches: int = 0,
    cache_positions: list[PositionRecord] | None = None,
) -> dict:
    """Compute BPB and per-gap-class metrics on held-out data."""
    student.eval()
    P = student.cfg.patch_size

    total_loss = 0.0
    total_tokens = 0
    first_byte_correct = 0
    first_byte_total = 0

    gap_losses = {"high_nll": 0.0, "high_entropy": 0.0,
                  "high_disagreement": 0.0, "control": 0.0}
    gap_counts = {"high_nll": 0, "high_entropy": 0,
                  "high_disagreement": 0, "control": 0}
    gap_fb_correct = {"high_nll": 0, "high_entropy": 0,
                      "high_disagreement": 0, "control": 0}
    gap_fb_total = {"high_nll": 0, "high_entropy": 0,
                    "high_disagreement": 0, "control": 0}

    byte_cat_losses = {cat: 0.0 for cat in BYTE_CATEGORIES}
    byte_cat_counts = {cat: 0 for cat in BYTE_CATEGORIES}

    pos_by_loc = {}
    if cache_positions:
        for p in cache_positions:
            key = (p.shard_id, p.seq_offset)
            pos_by_loc.setdefault(key, []).append(p)

    for i, batch in enumerate(eval_loader):
        if max_batches > 0 and i >= max_batches:
            break

        byte_ids, shard_ids, seq_starts = batch
        byte_ids = byte_ids.to(device)
        B, T = byte_ids.shape
        N = T // P

        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            out = student(byte_ids, return_aux=False)
            logits = out["logits"]
            targets = byte_ids.reshape(B, N, P)[:, 1:]
            per_pos_ce = F.cross_entropy(
                logits.reshape(-1, 256), targets.reshape(-1),
                reduction="none")

        predicted = B * (T - P)
        total_loss += per_pos_ce.sum().item()
        total_tokens += predicted

        preds = logits[:, :, 0, :].argmax(dim=-1)
        first_tgt = targets[:, :, 0]
        first_byte_correct += (preds == first_tgt).sum().item()
        first_byte_total += first_tgt.numel()

        fb_ce = per_pos_ce.reshape(B, -1, P)[:, :, 0]
        fb_tgt_np = first_tgt.cpu().numpy()
        fb_bpb = (fb_ce / math.log(2)).cpu().numpy()
        for cat, byte_set in BYTE_CATEGORIES.items():
            mask = np.isin(fb_tgt_np, list(byte_set))
            n = int(mask.sum())
            if n > 0:
                byte_cat_losses[cat] += float(fb_bpb[mask].sum())
                byte_cat_counts[cat] += n

        if cache_positions:
            for b in range(B):
                key = (int(shard_ids[b]), int(seq_starts[b]))
                for pos in pos_by_loc.get(key, []):
                    pi = pos.patch_idx - 1
                    if pi < 0 or pi >= logits.shape[1]:
                        continue
                    patch_logits = logits[b, pi, 0]
                    patch_target = targets[b, pi, 0]
                    pos_loss = F.cross_entropy(
                        patch_logits.unsqueeze(0),
                        patch_target.unsqueeze(0)).item()
                    pos_bpb = pos_loss / math.log(2)

                    pos_pred = patch_logits.argmax().item()
                    pos_correct = int(pos_pred == patch_target.item())

                    if pos.reason_mask & SelectionReason.HIGH_NLL:
                        gap_losses["high_nll"] += pos_bpb
                        gap_counts["high_nll"] += 1
                        gap_fb_correct["high_nll"] += pos_correct
                        gap_fb_total["high_nll"] += 1
                    if pos.reason_mask & SelectionReason.HIGH_ENTROPY:
                        gap_losses["high_entropy"] += pos_bpb
                        gap_counts["high_entropy"] += 1
                        gap_fb_correct["high_entropy"] += pos_correct
                        gap_fb_total["high_entropy"] += 1
                    if pos.reason_mask & SelectionReason.DISAGREEMENT:
                        gap_losses["high_disagreement"] += pos_bpb
                        gap_counts["high_disagreement"] += 1
                        gap_fb_correct["high_disagreement"] += pos_correct
                        gap_fb_total["high_disagreement"] += 1
                    if pos.reason_mask == SelectionReason.CONTROL:
                        gap_losses["control"] += pos_bpb
                        gap_counts["control"] += 1
                        gap_fb_correct["control"] += pos_correct
                        gap_fb_total["control"] += 1

    student.train()
    if total_tokens == 0:
        raise ValueError(
            "Eval produced 0 tokens — eval shards too small or batch_size "
            "too large (drop_last=True discards partial batches)")
    avg_loss = total_loss / total_tokens
    bpb = avg_loss / math.log(2)

    result = {
        "bpb": round(bpb, 4),
        "first_byte_acc": round(first_byte_correct / max(first_byte_total, 1), 4),
        "n_eval_tokens": total_tokens,
    }

    for key in ("high_nll", "high_entropy", "high_disagreement", "control"):
        if gap_counts[key] > 0:
            result[f"bpb_{key}"] = round(
                gap_losses[key] / gap_counts[key], 4)
        if gap_fb_total[key] > 0:
            result[f"first_byte_acc_{key}"] = round(
                gap_fb_correct[key] / gap_fb_total[key], 4)

    for cat in BYTE_CATEGORIES:
        if byte_cat_counts[cat] > 0:
            result[f"fb_bpb_byte_{cat}"] = round(
                byte_cat_losses[cat] / byte_cat_counts[cat], 4)
            result[f"n_fb_byte_{cat}"] = byte_cat_counts[cat]

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Frozen checkpoint BPB evaluator for E2 ablations")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval-shards", required=True)
    parser.add_argument("--ablation-id", required=True,
                        help="Ablation label from training config")
    parser.add_argument("--run-label", default="",
                        help="Optional human-readable run label")
    parser.add_argument("--cache-dir", default=None,
                        help="E2 cache dir for per-gap-class BPB breakdown")
    parser.add_argument("--output", default="ablation_result.json")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--shard-range", type=int, nargs=2, default=None,
                        metavar=("START", "END"),
                        help="Shard index range [START, END) to evaluate. "
                             "Use to exclude training shards.")
    args = parser.parse_args()

    cfg = EvalConfig(
        ablation_id=args.ablation_id,
        run_label=args.run_label,
        eval_shards=args.eval_shards,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        max_eval_batches=args.max_batches,
        output=args.output,
        cache_dir=args.cache_dir,
    )

    use_cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Device: {device}")
    print(f"Ablation: {cfg.ablation_id}")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_cfg = ckpt["model_cfg"]
    student = SutraS0(model_cfg).to(device)
    student.load_state_dict(ckpt["model"])
    student.eval()
    print(f"Loaded student from step {ckpt.get('step', '?')}")

    cache_positions = None
    if cfg.cache_dir:
        pos_path = os.path.join(cfg.cache_dir, "positions.bin")
        cache_positions = read_position_manifest(pos_path)
        print(f"Loaded {len(cache_positions)} cache positions for per-gap metrics")

    all_shards = sorted(Path(cfg.eval_shards).glob("*.bin"))
    if not all_shards:
        raise ValueError(f"No .bin shards found in {cfg.eval_shards}")

    shard_range = tuple(args.shard_range) if args.shard_range else None
    if shard_range is not None:
        s, e = shard_range
        if s < 0 or e <= s or e > len(all_shards):
            raise ValueError(
                f"--shard-range {s} {e} is invalid for {len(all_shards)} shards. "
                f"Need 0 <= start < end <= {len(all_shards)}."
            )
    eval_dataset = EklavyaDataset(
        cfg.eval_shards, cfg.seq_len, model_cfg.patch_size,
        shard_range=shard_range)
    eval_loader = DataLoader(
        eval_dataset, batch_size=cfg.batch_size,
        shuffle=False, num_workers=1, pin_memory=True, drop_last=True)

    n_eval_shards = (shard_range[1] - shard_range[0]) if shard_range else len(all_shards)
    print(f"Evaluating on {n_eval_shards} shards"
          f"{f' (range [{shard_range[0]}, {shard_range[1]}))' if shard_range else ''}...")
    metrics = evaluate_bpb(
        student, eval_loader, device,
        max_batches=cfg.max_eval_batches,
        cache_positions=cache_positions,
    )

    training_config = ckpt.get("config", {})

    report = {
        "ablation_id": cfg.ablation_id,
        "run_label": cfg.run_label,
        "checkpoint": args.checkpoint,
        "step": ckpt.get("step", -1),
        "checkpoint_phase": str(ckpt.get("phase", "")),
        "shard_range": list(shard_range) if shard_range else None,
        "training_config": training_config,
        "metrics": metrics,
    }

    os.makedirs(os.path.dirname(os.path.abspath(cfg.output)), exist_ok=True)
    with open(cfg.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved: {cfg.output}")
    print(f"  BPB: {metrics['bpb']}")
    print(f"  First-byte accuracy: {metrics['first_byte_acc']}")
    for key in ("high_nll", "high_entropy", "high_disagreement", "control"):
        parts = []
        if f"bpb_{key}" in metrics:
            parts.append(f"bpb={metrics[f'bpb_{key}']}")
        if f"first_byte_acc_{key}" in metrics:
            parts.append(f"fb_acc={metrics[f'first_byte_acc_{key}']}")
        if parts:
            print(f"  {key}: {', '.join(parts)}")
    for cat in BYTE_CATEGORIES:
        if f"fb_bpb_byte_{cat}" in metrics:
            print(f"  fb_byte_{cat}: bpb={metrics[f'fb_bpb_byte_{cat}']} "
                  f"(n={metrics[f'n_fb_byte_{cat}']})")


if __name__ == "__main__":
    main()
