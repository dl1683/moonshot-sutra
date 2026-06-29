"""Option C: Teacher-guided S0 pretraining with Qwen KL from step 0.

Unlike E1 (post-training KD), Option C applies KD loss during initial
pretraining. The core bet: early output-distribution guidance while CE
anchors byte modeling produces better representations than CE-only.

Imports primitives from s0_training and eklavya_training — no duplication.

Usage:
    python s0_option_c_training.py \
        --data-dir ../data/shards_bytes_full \
        --cache-dir C:/sutra_fast/option_c_qwen_cache \
        --checkpoint-dir C:/sutra_fast/checkpoints/option_c_pilot \
        --steps 10000
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from s0_architecture import S0Config, SutraS0
from s0_training import (
    atomic_save, get_lr, setup_activation_checkpointing, compute_loss,
    TrainConfig,
)
from eklavya_training import (
    topk_tail_kl, apply_gradient_budget, EklavyaDataset, _rng_state,
)
from eklavya_cache import MappedByteKLCache, ByteKLRecord


@dataclass
class OptionCConfig:
    total_steps: int = 50000
    lr: float = 2e-4
    min_lr: float = 2e-5
    warmup_steps: int = 1500
    grad_clip: float = 1.0

    batch_size: int = 4
    seq_len_bytes: int = 4096
    grad_accum_steps: int = 2
    dtype: str = "bfloat16"

    kl_temperature: float = 2.0
    max_kl_per_seq: int = 64

    checkpoint_dir: str = "checkpoints/option_c"
    log_file: str = "logs/option_c.jsonl"
    data_dir: str = "data/shards_bytes_full"
    cache_dir: str = "option_c_cache"

    checkpoint_every: int = 1000
    eval_every: int = 500
    eval_batches: int = 50
    log_every: int = 10
    eval_hold_shards: int = 5
    checkpoint_layers: int = 2
    compile_model: bool = False

    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8

    resume_from: Optional[str] = None

    ce_only_no_go_steps: int = 20


def get_lambda_kd(step: int) -> float:
    if step < 2000:
        return 0.05 + (0.20 - 0.05) * step / 2000
    elif step < 20000:
        return 0.25
    elif step < 40000:
        return 0.20
    else:
        return 0.10


def get_teacher_grad_budget(step: int) -> float:
    if step < 2000:
        return 0.30
    elif step < 40000:
        return 0.45
    else:
        return 0.35


def compute_batch_kl_loss(
    logits: torch.Tensor,
    shard_ids: torch.Tensor,
    seq_offsets: torch.Tensor,
    cache: MappedByteKLCache,
    device: torch.device,
    T: float = 2.0,
    max_per_seq: int = 64,
) -> tuple[torch.Tensor, int, int]:
    B, Nm1, P, V = logits.shape
    all_losses = []
    n_records_used = 0
    n_seqs_with_signal = 0

    for b in range(B):
        sid = int(shard_ids[b].item())
        soff = int(seq_offsets[b].item())

        records = cache.get_records(sid, soff)
        if not records:
            continue

        if len(records) > max_per_seq:
            rng = np.random.default_rng(sid * 10007 + soff)
            indices = rng.choice(len(records), size=max_per_seq, replace=False)
            records = [records[i] for i in sorted(indices)]

        n_seqs_with_signal += 1
        for r in records:
            logit_idx = r.patch_idx - 1
            if logit_idx < 0 or logit_idx >= Nm1:
                continue

            student_logit = logits[b, logit_idx, 0]

            top_b = torch.from_numpy(r.top_bytes).to(device)
            top_p = torch.from_numpy(r.top_probs.astype(np.float32)).to(device)
            tail_p = torch.tensor(r.tail_prob, device=device, dtype=torch.float32)

            loss = topk_tail_kl(student_logit, top_b, top_p, tail_p, T=T)
            all_losses.append(loss)
            n_records_used += 1

    if not all_losses:
        return torch.tensor(0.0, device=device), 0, 0
    return torch.stack(all_losses).mean(), n_records_used, n_seqs_with_signal


def train_option_c(cfg: OptionCConfig, model_cfg: Optional[S0Config] = None):
    model_cfg = model_cfg or S0Config()

    use_cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0
    device = torch.device("cuda" if use_cuda else "cpu")
    amp_dtype = getattr(torch, cfg.dtype)

    print(f"Device: {device}")
    print(f"AMP dtype: {amp_dtype}")
    print(f"Option C: teacher-guided pretraining")

    cache = MappedByteKLCache(cfg.cache_dir)
    print(f"Cache loaded: {len(cache)} indexed sequences, "
          f"{cache.n_records} total KL records")
    cache_manifest = cache.manifest
    if "shard_range" in cache_manifest:
        cache_shard_range = tuple(cache_manifest["shard_range"])
        print(f"Cache shard range: {cache_shard_range}")

    model = SutraS0(model_cfg).to(device)
    counts = model.count_parameters()
    print(f"Model parameters: {counts['total']:,} ({counts['total']/1e6:.1f}M)")

    setup_activation_checkpointing(model, cfg.checkpoint_layers)

    s0_dead_prefixes = ("encoder.entropy_head", "encoder.residual_head", "verifier")
    for name, param in model.named_parameters():
        if any(name.startswith(p) for p in s0_dead_prefixes):
            param.requires_grad_(False)

    if cfg.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)

    decay_params = []
    nodecay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or "bias" in name or "norm" in name or "emb" in name:
            nodecay_params.append(param)
        else:
            decay_params.append(param)

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": cfg.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ], lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), eps=cfg.eps)

    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))

    all_shards = sorted(Path(cfg.data_dir).glob("*.bin"))
    n_shards = len(all_shards)
    n_eval = min(cfg.eval_hold_shards, max(1, n_shards // 10))
    train_range = (0, n_shards - n_eval)
    eval_range = (n_shards - n_eval, n_shards)
    print(f"Data split: {train_range[1]} train shards, {n_eval} eval shards")

    if "shard_range" in cache_manifest:
        cr = cache_shard_range
        if cr[0] > train_range[0] or cr[1] < train_range[1]:
            print(f"WARNING: Cache shard range {cr} does not cover "
                  f"train range {train_range}")

    train_dataset = EklavyaDataset(cfg.data_dir, cfg.seq_len_bytes,
                                    model_cfg.patch_size, shard_range=train_range)
    eval_dataset = EklavyaDataset(cfg.data_dir, cfg.seq_len_bytes,
                                   model_cfg.patch_size, shard_range=eval_range)
    sampler_gen = torch.Generator()
    sampler_gen.manual_seed(42)
    sampler_gen_state = sampler_gen.get_state()
    batches_consumed_in_epoch = 0
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size,
        shuffle=True, num_workers=2, pin_memory=True, drop_last=True,
        generator=sampler_gen,
    )
    eval_loader = DataLoader(
        eval_dataset, batch_size=cfg.batch_size,
        shuffle=False, num_workers=1, pin_memory=True, drop_last=True,
    )

    os.makedirs(os.path.dirname(cfg.log_file) or ".", exist_ok=True)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    log_f = open(cfg.log_file, "a")

    start_step = 0
    best_eval_bpb = float("inf")
    if cfg.resume_from:
        ckpt = torch.load(cfg.resume_from, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        if "rng_state" in ckpt:
            torch.set_rng_state(ckpt["rng_state"])
            if device.type == "cuda" and "cuda_rng_state" in ckpt:
                torch.cuda.set_rng_state(ckpt["cuda_rng_state"])
        if "sampler_gen_state" in ckpt:
            sampler_gen.set_state(ckpt["sampler_gen_state"])
            sampler_gen_state = ckpt["sampler_gen_state"]
            batches_consumed_in_epoch = ckpt.get("batches_consumed_in_epoch", 0)
        start_step = ckpt["step"]
        if "best_eval_bpb" in ckpt:
            best_eval_bpb = ckpt["best_eval_bpb"]
        print(f"Resumed from step {start_step} (best eval BPB: {best_eval_bpb:.3f})")

    model.train()
    data_iter = iter(train_loader)
    if cfg.resume_from and batches_consumed_in_epoch > 0:
        print(f"  Fast-forwarding {batches_consumed_in_epoch} batches...")
        for _ in range(batches_consumed_in_epoch):
            try:
                next(data_iter)
            except StopIteration:
                break

    step = start_step
    accum_loss = 0.0
    accum_ce = 0.0
    accum_kl = 0.0
    accum_steps = 0
    accum_kl_records = 0
    accum_kl_seqs = 0
    consecutive_no_kl = 0
    train_start = time.time()
    t0 = time.time()

    # Cosine LR via TrainConfig for get_lr compatibility
    lr_cfg = TrainConfig(
        lr=cfg.lr, min_lr=cfg.min_lr,
        warmup_steps=cfg.warmup_steps, total_steps=cfg.total_steps,
    )

    while step < cfg.total_steps:
        lr = get_lr(step, lr_cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)

        for micro_step in range(cfg.grad_accum_steps):
            try:
                batch = next(data_iter)
                batches_consumed_in_epoch += 1
            except StopIteration:
                sampler_gen_state = sampler_gen.get_state()
                data_iter = iter(train_loader)
                batch = next(data_iter)
                batches_consumed_in_epoch = 1

            byte_ids, shard_ids, seq_offsets = batch
            byte_ids = byte_ids.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
                out = model(byte_ids, return_aux=False)
                ce_losses = compute_loss(out, byte_ids, model_cfg.patch_size)
                L_ce = ce_losses["loss"]

                lambda_kd = get_lambda_kd(step)
                L_kl, n_kl_used, n_kl_seqs = compute_batch_kl_loss(
                    out["logits"], shard_ids, seq_offsets, cache, device,
                    T=cfg.kl_temperature, max_per_seq=cfg.max_kl_per_seq,
                )
                L_teacher = lambda_kd * L_kl

            accum_ce += ce_losses["byte_ce"]
            accum_kl += L_kl.item()
            accum_kl_records += n_kl_used
            accum_kl_seqs += n_kl_seqs
            accum_steps += 1

            has_teacher = n_kl_used > 0
            if has_teacher:
                consecutive_no_kl = 0
            else:
                consecutive_no_kl += 1

            budget = get_teacher_grad_budget(step)

            if has_teacher and budget > 0:
                gb_ce_norm, gb_teacher_norm, gb_scale = apply_gradient_budget(
                    trainable_params, L_ce / cfg.grad_accum_steps,
                    L_teacher / cfg.grad_accum_steps, budget,
                    scaler=scaler,
                )
            else:
                gb_ce_norm, gb_teacher_norm, gb_scale = 0.0, 0.0, 1.0
                scaled_loss = L_ce / cfg.grad_accum_steps
                if scaler is not None:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

            loss = L_ce + L_teacher
            accum_loss += loss.item()

        scaler.unscale_(optimizer)
        grad_norm = nn.utils.clip_grad_norm_(trainable_params, cfg.grad_clip)

        if not math.isfinite(grad_norm.item()):
            fail_entry = {"step": step + 1, "HARD_FAIL": "non-finite grad_norm",
                          "grad_norm": grad_norm.item()}
            log_f.write(json.dumps(fail_entry) + "\n")
            log_f.flush()
            raise RuntimeError(f"HARD FAIL at step {step + 1}: grad_norm={grad_norm.item()}")

        scaler.step(optimizer)
        scaler.update()

        step += 1

        if accum_steps > 0 and not math.isfinite(accum_loss):
            fail_entry = {"step": step, "HARD_FAIL": "non-finite loss",
                          "accum_loss": accum_loss}
            log_f.write(json.dumps(fail_entry) + "\n")
            log_f.flush()
            raise RuntimeError(f"HARD FAIL at step {step}: loss is non-finite")

        if consecutive_no_kl >= cfg.ce_only_no_go_steps * cfg.grad_accum_steps:
            optimizer_steps = consecutive_no_kl // cfg.grad_accum_steps
            fail_entry = {"step": step, "HARD_FAIL": "no_kl_signal",
                          "consecutive_ce_only_optimizer_steps": optimizer_steps}
            log_f.write(json.dumps(fail_entry) + "\n")
            log_f.flush()
            raise RuntimeError(
                f"Option C HARD FAIL: {optimizer_steps} consecutive optimizer steps "
                f"with NO teacher signal. Cache/shard mismatch.")

        if step % cfg.log_every == 0 and accum_steps > 0:
            avg_ce = accum_ce / accum_steps
            avg_kl = accum_kl / accum_steps
            bpb = avg_ce / math.log(2)
            elapsed = time.time() - t0
            tokens_per_sec = (accum_steps * cfg.batch_size * cfg.seq_len_bytes) / elapsed

            entry = {
                "step": step,
                "loss": round(accum_loss / accum_steps, 4),
                "ce": round(avg_ce, 4),
                "bpb": round(bpb, 4),
                "kl": round(avg_kl, 4),
                "lambda_kd": round(get_lambda_kd(step), 4),
                "teacher_grad_budget": round(get_teacher_grad_budget(step), 2),
                "gb_ce_norm": round(gb_ce_norm, 4),
                "gb_teacher_norm": round(gb_teacher_norm, 4),
                "gb_scale": round(gb_scale, 4),
                "kl_records_used": accum_kl_records,
                "kl_seq_coverage": accum_kl_seqs,
                "lr": lr,
                "grad_norm": round(grad_norm.item(), 4),
                "tok_per_sec": round(tokens_per_sec),
                "elapsed_s": round(elapsed, 1),
            }
            log_f.write(json.dumps(entry) + "\n")
            log_f.flush()

            if step % (cfg.log_every * 10) == 0:
                gb_info = f" | gb={gb_scale:.2f}" if gb_scale < 1.0 else ""
                print(f"step {step:>6d} | bpb {bpb:.3f} | kl {avg_kl:.4f} | "
                      f"λ_kd {get_lambda_kd(step):.3f} | "
                      f"lr {lr:.2e} | gnorm {grad_norm:.2f} | "
                      f"{tokens_per_sec:.0f} tok/s{gb_info}")

            accum_loss = 0.0
            accum_ce = 0.0
            accum_kl = 0.0
            accum_steps = 0
            accum_kl_records = 0
            accum_kl_seqs = 0
            t0 = time.time()

        if step % cfg.eval_every == 0:
            model.eval()
            eval_loss = 0.0
            eval_tokens = 0
            P = model_cfg.patch_size
            pos_correct = torch.zeros(P, device=device)
            pos_total = torch.zeros(P, device=device)

            with torch.no_grad():
                for i, ebatch in enumerate(eval_loader):
                    if cfg.eval_batches > 0 and i >= cfg.eval_batches:
                        break
                    ebyte_ids, _, _ = ebatch
                    ebyte_ids = ebyte_ids.to(device)
                    B, T = ebyte_ids.shape

                    with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
                        eout = model(ebyte_ids, return_aux=False)
                        elosses = compute_loss(eout, ebyte_ids, P)

                    eval_loss += elosses["byte_ce"] * B
                    eval_tokens += B

                    elogits = eout["logits"]
                    N = T // P
                    etargets = ebyte_ids.reshape(B, N, P)[:, 1:]
                    for pos in range(P):
                        preds = elogits[:, :, pos, :].argmax(dim=-1)
                        correct = (preds == etargets[:, :, pos]).sum().item()
                        pos_correct[pos] += correct
                        pos_total[pos] += preds.numel()

            avg_eval = eval_loss / max(eval_tokens, 1)
            eval_bpb = avg_eval / math.log(2)
            pos_acc = [round((pos_correct[i] / max(pos_total[i], 1)).item(), 4)
                       for i in range(P)]

            eval_entry = {
                "step": step,
                "eval_loss": round(avg_eval, 4),
                "eval_bpb": round(eval_bpb, 4),
                "eval_byte_acc": round(sum(pos_correct.cpu().tolist()) /
                                        max(sum(pos_total.cpu().tolist()), 1), 4),
                "eval_pos_acc": pos_acc,
            }
            log_f.write(json.dumps(eval_entry) + "\n")
            log_f.flush()
            pos_str = " ".join(f"p{i}={a:.3f}" for i, a in enumerate(pos_acc))
            print(f"  EVAL step {step}: bpb {eval_bpb:.3f} acc "
                  f"{eval_entry['eval_byte_acc']:.4f} | {pos_str}")

            if eval_bpb < best_eval_bpb:
                best_eval_bpb = eval_bpb
                best_path = os.path.join(cfg.checkpoint_dir, "optc_best.pt")
                atomic_save({
                    "step": step,
                    "model": model.state_dict(),
                    "model_cfg": model_cfg,
                    "eval_bpb": eval_bpb,
                }, best_path)
                print(f"  New best: {eval_bpb:.3f} → {best_path}")

            model.train()

        if step % cfg.checkpoint_every == 0:
            ckpt_path = os.path.join(cfg.checkpoint_dir, f"optc_step{step}.pt")
            atomic_save({
                "step": step,
                "model": model.state_dict(),
                "model_cfg": model_cfg,
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "best_eval_bpb": best_eval_bpb,
                "sampler_gen_state": sampler_gen_state,
                "batches_consumed_in_epoch": batches_consumed_in_epoch,
                **_rng_state(device),
            }, ckpt_path)

    final_path = os.path.join(cfg.checkpoint_dir, "optc_final.pt")
    atomic_save({
        "step": step,
        "model": model.state_dict(),
        "model_cfg": model_cfg,
        "eval_bpb": best_eval_bpb,
    }, final_path)

    log_f.close()
    cache.close()

    elapsed = time.time() - train_start
    print(f"\nOption C training complete in {elapsed:.0f}s")
    print(f"Best eval BPB: {best_eval_bpb:.3f}")
    print(f"Final checkpoint: {final_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Option C: teacher-guided S0 pretraining")
    parser.add_argument("--data-dir", default="data/shards_bytes_full")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--checkpoint-dir",
                        default="C:/sutra_fast/checkpoints/option_c")
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--min-lr", type=float, default=2e-5)
    parser.add_argument("--warmup-steps", type=int, default=1500)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--log-file", default="logs/option_c.jsonl")
    args = parser.parse_args()

    cfg = OptionCConfig(
        total_steps=args.steps,
        lr=args.lr,
        min_lr=args.min_lr,
        warmup_steps=args.warmup_steps,
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        checkpoint_dir=args.checkpoint_dir,
        log_file=args.log_file,
        resume_from=args.resume,
    )

    train_option_c(cfg)


if __name__ == "__main__":
    main()
