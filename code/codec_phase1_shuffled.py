"""Phase 1 Shuffled Control: Validate that Phase 1 signal is discriminative.

Same training as codec_phase1_train.py but with teacher embeddings RANDOMLY
PERMUTED. If the codec can still learn to "retrieve" under shuffled conditions,
the signal is positional, not semantic. If accuracy stays at chance, the signal
is real.

Codex R63 requirement: "shuffled-label controls must fail"

Usage:
  python codec_phase1_shuffled.py --data-dir C:/sutra_fast/data/shards_diverse --steps 1000
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast

from semantic_codec import SemanticCodec, CodecConfig, infonce_loss
from codec_phase1_train import (
    load_teacher_embeddings,
    load_tokenizer,
    ByteShardLoader,
    find_token_anchors,
)


def train_shuffled_control(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("=" * 60)
    print("SHUFFLED CONTROL: Teacher embeddings randomly permuted")
    print("Expected result: accuracy stays near chance (~0.1%)")
    print("=" * 60)

    teacher_embs, _ = load_teacher_embeddings()

    # CRITICAL: Randomly permute the embedding table
    # This destroys the token_id → embedding correspondence
    # token_id 42 now maps to a random embedding, not its real one
    perm = torch.randperm(teacher_embs.shape[0])
    teacher_embs_shuffled = teacher_embs[perm]
    print(f"Shuffled {teacher_embs.shape[0]} embeddings with random permutation")

    teacher_embs_shuffled = teacher_embs_shuffled.to(device)

    tokenizer = load_tokenizer()

    cfg = CodecConfig(window_size=args.window_size)
    codec = SemanticCodec(cfg, d_model=args.d_model).to(device)
    params = codec.count_params()
    print(f"Codec params: {params['total']:,}")

    loader = ByteShardLoader(args.data_dir, seq_len=args.seq_len)
    rng = np.random.default_rng(args.seed + 999)  # different seed for independence

    codec_params = list(codec.encoder.parameters()) + list(codec.patch_projection.parameters())
    align_params = list(codec.alignment_head.parameters())
    optimizer = torch.optim.AdamW([
        {"params": codec_params, "lr": args.lr_codec},
        {"params": align_params, "lr": args.lr_align},
    ], weight_decay=0.01)

    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / args.warmup_steps
        decay = 1.0 - (step - args.warmup_steps) / (args.steps - args.warmup_steps)
        return max(0.1, decay)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler()

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "phase1_shuffled_log.jsonl")
    log_f = open(log_path, "w")

    t0 = time.time()

    for step in range(1, args.steps + 1):
        byte_ids = loader.get_batch(args.batch_size, rng)
        anchor_pos, anchor_tids = find_token_anchors(
            byte_ids, tokenizer, max_anchors=args.max_anchors
        )

        byte_ids = byte_ids.to(device)
        anchor_pos = anchor_pos.to(device)
        anchor_tids = anchor_tids.to(device)

        # Use SHUFFLED embeddings — token_id maps to wrong embedding
        target_embs = teacher_embs_shuffled[anchor_tids]

        with autocast(dtype=torch.bfloat16):
            projected = codec.forward_phase1(byte_ids, anchor_pos)
            loss, acc = infonce_loss(projected, target_embs, temperature=cfg.temperature)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(codec.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if step % args.log_every == 0:
            elapsed = time.time() - t0
            entry = {
                "step": step,
                "loss": round(loss.item(), 4),
                "top1_acc": round(acc, 4),
                "lr": round(optimizer.param_groups[0]["lr"], 6),
                "elapsed_s": round(elapsed, 1),
                "control": "shuffled",
            }
            log_f.write(json.dumps(entry) + "\n")
            log_f.flush()
            print(f"[SHUFFLED {step}/{args.steps}] loss={loss.item():.4f} acc={acc:.4f}")

    log_f.close()
    print(f"\nShuffled control complete.")
    print(f"If acc >> chance ({1/(args.batch_size * args.max_anchors):.4f}), "
          f"Phase 1 signal is NOT semantic — it's positional.")
    print(f"If acc ~ chance, Phase 1 signal is REAL semantic alignment.")


def main():
    parser = argparse.ArgumentParser(description="Phase 1 Shuffled Control")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="C:/sutra_fast/codec_phase1_shuffled")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--max-anchors", type=int, default=128)
    parser.add_argument("--lr-codec", type=float, default=3e-4)
    parser.add_argument("--lr-align", type=float, default=1e-3)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=1152)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()

    train_shuffled_control(args)


if __name__ == "__main__":
    main()
