"""Phase 2: Train Sutra with frozen semantic codec.

Loads Phase 1 codec checkpoint, freezes the encoder, and trains
PatchProjection + GlobalReasoner + ByteDecoder with byte CE.

This is the critical experiment: does semantic addressability from the
codec enable the global reasoner to develop world knowledge that the
independent-patch ByteEncoder couldn't support?

Usage:
  python codec_phase2_train.py \
    --codec-checkpoint C:/sutra_fast/codec_phase1/codec_final.pt \
    --data-dir C:/sutra_fast/data/shards_diverse \
    --steps 20000 \
    --checkpoint-dir C:/sutra_fast/codec_phase2 \
    --log-file C:/sutra_fast/codec_phase2/training.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint

from s0_architecture import S0Config
from s0_training import (
    TrainConfig,
    ByteShardDataset,
    compute_loss,
    get_lr,
)
from codec_phase2_model import SutraCodecModel, load_codec_for_phase2


def atomic_save(obj, path):
    d = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    os.close(fd)
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def setup_checkpointing(model: SutraCodecModel, every_n: int = 2):
    """Activation checkpointing for the reasoner layers."""
    if every_n < 1:
        return

    def checkpointed_forward(patch_states):
        x = patch_states
        layers = list(model.reasoner.layers)
        freqs = model.reasoner.rope_freqs
        for i, layer in enumerate(layers):
            if i % every_n == 0 and x.requires_grad:
                x = checkpoint(layer, x, freqs, use_reentrant=False)
            else:
                x = layer(x, freqs)
        return model.reasoner.norm(x)

    model.reasoner.forward = checkpointed_forward


@torch.no_grad()
def evaluate(model, eval_loader, device, train_cfg, patch_size):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    P = patch_size
    amp_dtype = getattr(torch, train_cfg.dtype)
    pos_correct = torch.zeros(P, device=device)
    pos_total = torch.zeros(P, device=device)

    for i, batch in enumerate(eval_loader):
        if train_cfg.eval_batches > 0 and i >= train_cfg.eval_batches:
            break
        byte_ids = batch.to(device)
        B, T = byte_ids.shape
        N = T // P
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
            out = model(byte_ids, return_aux=False)
            losses = compute_loss(out, byte_ids, P)
        predicted_bytes = B * (T - P)
        total_loss += losses["byte_ce"] * predicted_bytes
        total_tokens += predicted_bytes

        targets = byte_ids.reshape(B, N, P)[:, 1:]
        preds = out["logits"].argmax(dim=-1)
        for p in range(P):
            pos_correct[p] += (preds[:, :, p] == targets[:, :, p]).sum()
            pos_total[p] += targets[:, :, p].numel()

    model.train()
    avg_loss = total_loss / max(total_tokens, 1)
    pos_acc = (pos_correct / pos_total.clamp(min=1)).cpu().tolist()
    return {
        "eval_loss": avg_loss,
        "eval_bpb": avg_loss / math.log(2),
        "eval_byte_acc": sum(pos_correct.cpu().tolist()) / max(sum(pos_total.cpu().tolist()), 1),
        "eval_pos_acc": [round(a, 4) for a in pos_acc],
    }


def train_phase2(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16

    print(f"Device: {device}")
    print(f"Phase 2: Frozen codec + trainable reasoner/decoder")

    # Load codec from Phase 1
    from s0_configs import ALL_CONFIGS
    model_cfg = ALL_CONFIGS[args.config]()

    codec = load_codec_for_phase2(args.codec_checkpoint, d_model=model_cfg.d_model)
    model = SutraCodecModel(model_cfg, codec).to(device)

    counts = model.count_parameters()
    print(f"Total params: {counts['total']:,} ({counts['total']/1e6:.1f}M)")
    print(f"Trainable: {counts['trainable']:,} ({counts['trainable']/1e6:.1f}M)")
    print(f"Frozen encoder: {counts['codec_encoder_frozen']:,} ({counts['codec_encoder_frozen']/1e6:.1f}M)")

    # Activation checkpointing
    setup_checkpointing(model, every_n=2)

    # Optimizer — decay vs no-decay param groups
    decay_params = []
    nodecay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or "bias" in name or "norm" in name or "emb" in name:
            nodecay_params.append(param)
        else:
            decay_params.append(param)

    train_cfg = TrainConfig(
        data_dir=args.data_dir,
        total_steps=args.steps,
        warmup_steps=args.warmup_steps,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        checkpoint_dir=args.checkpoint_dir,
        log_file=args.log_file,
        checkpoint_every=args.checkpoint_every,
        eval_every=args.eval_every,
        log_every=args.log_every,
        eval_hold_shards=args.eval_hold_shards,
        eval_batches=args.eval_batches,
    )

    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": train_cfg.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ], lr=train_cfg.lr, betas=(train_cfg.beta1, train_cfg.beta2), eps=train_cfg.eps)

    scaler = torch.amp.GradScaler("cuda", enabled=False)  # bf16 doesn't need scaler

    # Data
    all_shards = sorted(Path(train_cfg.data_dir).glob("*.bin"))
    n_shards = len(all_shards)
    n_eval = min(train_cfg.eval_hold_shards, max(1, n_shards // 10))
    train_range = (0, n_shards - n_eval)
    eval_range = (n_shards - n_eval, n_shards)
    print(f"Data: {train_range[1]} train shards, {n_eval} eval shards")

    train_dataset = ByteShardDataset(
        train_cfg.data_dir, train_cfg.seq_len_bytes,
        model_cfg.patch_size, shard_range=train_range,
    )
    eval_dataset = ByteShardDataset(
        train_cfg.data_dir, train_cfg.seq_len_bytes,
        model_cfg.patch_size, shard_range=eval_range,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=train_cfg.batch_size,
        shuffle=True, num_workers=2, pin_memory=True, drop_last=True,
    )
    eval_loader = DataLoader(
        eval_dataset, batch_size=train_cfg.batch_size,
        shuffle=False, num_workers=1, pin_memory=True, drop_last=True,
    )

    # Logging
    os.makedirs(os.path.dirname(train_cfg.log_file), exist_ok=True)
    os.makedirs(train_cfg.checkpoint_dir, exist_ok=True)
    log_f = open(train_cfg.log_file, "a")

    # Resume
    start_step = 0
    best_eval_bpb = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        if "best_eval_bpb" in ckpt:
            best_eval_bpb = ckpt["best_eval_bpb"]
        print(f"Resumed from step {start_step}")

    # Training loop
    model.train()
    data_iter = iter(train_loader)
    step = start_step
    accum_loss = 0.0
    accum_steps = 0
    train_start = time.time()
    t0 = time.time()

    while step < train_cfg.total_steps:
        lr = get_lr(step, train_cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)

        for micro_step in range(train_cfg.grad_accum_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            byte_ids = batch.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
                out = model(byte_ids, return_aux=False)
                losses = compute_loss(out, byte_ids, model_cfg.patch_size)
                loss = losses["loss"] / train_cfg.grad_accum_steps

            loss.backward()
            accum_loss += losses["byte_ce"]
            accum_steps += 1

        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            train_cfg.grad_clip,
        )

        if not math.isfinite(grad_norm.item()):
            fail_entry = {"step": step + 1, "HARD_FAIL": "non-finite grad_norm"}
            log_f.write(json.dumps(fail_entry) + "\n")
            log_f.flush()
            raise RuntimeError(f"HARD FAIL at step {step + 1}: grad_norm={grad_norm.item()}")

        optimizer.step()

        step += 1

        # Logging
        if step % train_cfg.log_every == 0:
            avg_loss = accum_loss / accum_steps
            bpb = avg_loss / math.log(2)
            elapsed = time.time() - t0
            tokens_per_sec = (accum_steps * train_cfg.batch_size * train_cfg.seq_len_bytes) / elapsed

            log_entry = {
                "step": step,
                "loss": round(avg_loss, 4),
                "bpb": round(bpb, 4),
                "lr": lr,
                "grad_norm": round(grad_norm.item(), 4),
                "tok_per_sec": round(tokens_per_sec),
                "elapsed_s": round(elapsed, 1),
            }
            log_f.write(json.dumps(log_entry) + "\n")
            log_f.flush()
            print(f"step {step:>6d} | loss {avg_loss:.4f} | bpb {bpb:.3f} | "
                  f"lr {lr:.2e} | gnorm {grad_norm:.2f} | {tokens_per_sec:.0f} tok/s")

            accum_loss = 0.0
            accum_steps = 0
            t0 = time.time()

        # Eval
        if step % train_cfg.eval_every == 0:
            eval_metrics = evaluate(model, eval_loader, device, train_cfg, model_cfg.patch_size)
            eval_entry = {"step": step, **eval_metrics}
            log_f.write(json.dumps(eval_entry) + "\n")
            log_f.flush()
            pos_str = " ".join(
                f"p{i}={a:.3f}" for i, a in enumerate(eval_metrics.get("eval_pos_acc", []))
            )
            print(f"  EVAL step {step}: bpb {eval_metrics['eval_bpb']:.3f} "
                  f"acc {eval_metrics.get('eval_byte_acc', 0):.4f} | {pos_str}")

            if eval_metrics["eval_bpb"] < best_eval_bpb:
                best_eval_bpb = eval_metrics["eval_bpb"]
                best_path = os.path.join(train_cfg.checkpoint_dir, "phase2_best.pt")
                atomic_save({
                    "step": step,
                    "model": model.state_dict(),
                    "model_cfg": model_cfg,
                    "model_type": "codec",
                    "codec_checkpoint": args.codec_checkpoint,
                    "eval_bpb": best_eval_bpb,
                }, best_path)
                print(f"  New best eval BPB {best_eval_bpb:.3f} -- saved {best_path}")

        # Checkpoint
        if step % train_cfg.checkpoint_every == 0:
            ckpt_path = os.path.join(train_cfg.checkpoint_dir, f"phase2_step{step}.pt")
            atomic_save({
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "model_cfg": model_cfg,
                "model_type": "codec",
                "codec_checkpoint": args.codec_checkpoint,
                "best_eval_bpb": best_eval_bpb,
            }, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    log_f.close()
    total_time = time.time() - train_start
    hours = total_time / 3600
    steps_done = step - start_step
    print(f"\nPhase 2 complete. {steps_done} steps in {hours:.2f}h. Best eval BPB: {best_eval_bpb:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Codec + Reasoner Training")
    parser.add_argument("--codec-checkpoint", required=True,
                        help="Path to Phase 1 codec checkpoint")
    parser.add_argument("--data-dir", required=True,
                        help="Path to byte shard data directory")
    parser.add_argument("--config", choices=["p4", "p8", "d640", "d768", "wide7"],
                        default="wide7")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=16)
    parser.add_argument("--checkpoint-dir", default="C:/sutra_fast/codec_phase2")
    parser.add_argument("--log-file", default="C:/sutra_fast/codec_phase2/training.jsonl")
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-hold-shards", type=int, default=2)
    parser.add_argument("--eval-batches", type=int, default=50)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    train_phase2(args)


if __name__ == "__main__":
    main()
