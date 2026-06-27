"""S0 Training Loop — byte-level autoregressive pre-training.

Base S0 loss: byte-level cross-entropy over predicted bytes within each patch.
No teacher/packet losses at S0; those come at G1+.

Follows SE1_CANONICAL_SPEC.md R6 training stability recipe:
- pre-norm RMSNorm (built into architecture)
- warmup schedule (cosine with linear warmup)
- gradient clipping
- bf16 with fp32 optimizer states (via torch.amp)
- FlashAttention (via scaled_dot_product_attention)
- activation checkpointing by 2-4 layer blocks
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from s0_architecture import S0Config, SutraS0


@dataclass
class TrainConfig:
    # Data
    seq_len_bytes: int = 4096
    batch_size: int = 4
    grad_accum_steps: int = 16  # effective batch = 4 * 16 = 64

    # Optimizer
    lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0

    # Schedule
    warmup_steps: int = 1000
    total_steps: int = 50000
    lr_schedule: str = "cosine"

    # Training
    dtype: str = "bfloat16"
    compile_model: bool = False
    checkpoint_every: int = 1000
    log_every: int = 10
    eval_every: int = 500

    # Eval
    eval_hold_shards: int = 2  # hold out last N shards for validation
    eval_batches: int = 50  # limit eval to this many batches (0 = full dataset)

    # Activation checkpointing
    checkpoint_layers: int = 2  # checkpoint every N reasoner layers

    data_dir: str = "data/shards_bytes_full"
    checkpoint_dir: str = "checkpoints/s0"
    log_file: str = "logs/s0_train.jsonl"

    # Resume
    resume_from: Optional[str] = None


class ByteShardDataset(Dataset):
    """Memory-mapped byte shard dataset.

    Each shard is a raw binary file of uint8 bytes. The dataset serves
    fixed-length chunks of bytes for autoregressive training.
    """
    def __init__(self, data_dir: str, seq_len: int, patch_size: int = 4,
                 shard_range: tuple[int, int] | None = None):
        self.seq_len = seq_len
        if seq_len % patch_size != 0:
            raise ValueError(f"seq_len ({seq_len}) must be divisible by patch_size ({patch_size})")
        all_shards = sorted(Path(data_dir).glob("*.bin"))
        if not all_shards:
            raise FileNotFoundError(f"No .bin shards found in {data_dir}")
        if shard_range is not None:
            self.shards = all_shards[shard_range[0]:shard_range[1]]
        else:
            self.shards = all_shards
        if not self.shards:
            raise ValueError(f"shard_range {shard_range} yields 0 shards out of {len(all_shards)}")

        self.shard_sizes = []
        self.cumulative = [0]
        for shard in self.shards:
            size = shard.stat().st_size
            n_seqs = size // seq_len
            self.shard_sizes.append(n_seqs)
            self.cumulative.append(self.cumulative[-1] + n_seqs)

        self.total_seqs = self.cumulative[-1]
        if self.total_seqs == 0:
            raise ValueError(
                f"Zero usable sequences: {len(self.shards)} shards with "
                f"total {sum(s.stat().st_size for s in self.shards)} bytes, "
                f"seq_len={seq_len}")
        self._mmap_cache: dict[int, memoryview] = {}

    def __len__(self):
        return self.total_seqs

    def _get_mmap(self, shard_idx: int):
        if shard_idx not in self._mmap_cache:
            import mmap
            f = open(self.shards[shard_idx], "rb")
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            self._mmap_cache[shard_idx] = mm
        return self._mmap_cache[shard_idx]

    def __getitem__(self, idx):
        import bisect
        shard_idx = bisect.bisect_right(self.cumulative, idx) - 1
        local_idx = idx - self.cumulative[shard_idx]
        mm = self._get_mmap(shard_idx)
        start = local_idx * self.seq_len
        raw = mm[start:start + self.seq_len]
        return torch.frombuffer(bytearray(raw), dtype=torch.uint8).long()


def get_lr(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * max(step, 1) / cfg.warmup_steps
    if step >= cfg.total_steps:
        return cfg.min_lr
    decay_ratio = (step - cfg.warmup_steps) / (cfg.total_steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return cfg.min_lr + coeff * (cfg.lr - cfg.min_lr)


def setup_activation_checkpointing(model: SutraS0, every_n: int = 2):
    """Wrap reasoner layers with activation checkpointing."""
    from torch.utils.checkpoint import checkpoint

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


def compute_loss(model_out: dict, target_bytes: torch.Tensor, patch_size: int) -> dict:
    """Compute S0 base losses.

    Logits are shifted by one patch: logits[j] predicts bytes of patch j+1.
    Targets are aligned accordingly (patches 1..N-1).
    """
    B, T = target_bytes.shape
    N = T // patch_size

    logits = model_out["logits"]  # (B, N-1, P, 256)
    logits_flat = logits.reshape(-1, 256)
    targets_flat = target_bytes.reshape(B, N, patch_size)[:, 1:].reshape(-1)

    byte_ce = F.cross_entropy(logits_flat, targets_flat)

    bpb = byte_ce.item() / math.log(2)

    return {
        "loss": byte_ce,
        "byte_ce": byte_ce.item(),
        "bpb": bpb,
    }


@torch.no_grad()
def evaluate(model: SutraS0, eval_loader: DataLoader, device: torch.device, cfg: TrainConfig) -> dict:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    P = model.cfg.patch_size
    amp_dtype = getattr(torch, cfg.dtype)
    pos_correct = torch.zeros(P, device=device)
    pos_total = torch.zeros(P, device=device)

    for i, batch in enumerate(eval_loader):
        if cfg.eval_batches > 0 and i >= cfg.eval_batches:
            break
        byte_ids = batch.to(device)
        B, T = byte_ids.shape
        N = T // P
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            out = model(byte_ids)
            losses = compute_loss(out, byte_ids, P)
        predicted_bytes = B * (T - P)
        total_loss += losses["byte_ce"] * predicted_bytes
        total_tokens += predicted_bytes

        targets = byte_ids.reshape(B, N, P)[:, 1:]  # (B, N-1, P)
        preds = out["logits"].argmax(dim=-1)  # (B, N-1, P)
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


def train(model_cfg: Optional[S0Config] = None, train_cfg: Optional[TrainConfig] = None):
    model_cfg = model_cfg or S0Config()
    train_cfg = train_cfg or TrainConfig()

    use_cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0
    device = torch.device("cuda" if use_cuda else "cpu")
    amp_dtype = getattr(torch, train_cfg.dtype)

    print(f"Device: {device}")
    print(f"AMP dtype: {amp_dtype}")

    # Model
    model = SutraS0(model_cfg).to(device)
    counts = model.count_parameters()
    print(f"Model parameters: {counts['total']:,} ({counts['total']/1e6:.1f}M)")

    # Checkpoint BEFORE compile (R8 fix: compile wraps the monkey-patched forward)
    setup_activation_checkpointing(model, train_cfg.checkpoint_layers)

    # Freeze dead heads BEFORE compile (R12 fix: compile renames params to _orig_mod.*)
    s0_dead_prefixes = ("encoder.entropy_head", "encoder.residual_head", "verifier")
    for name, param in model.named_parameters():
        if any(name.startswith(p) for p in s0_dead_prefixes):
            param.requires_grad_(False)

    if train_cfg.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)

    # Build optimizer param groups (dead heads already frozen above)
    decay_params = []
    nodecay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or "bias" in name or "norm" in name or "emb" in name:
            nodecay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": train_cfg.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ], lr=train_cfg.lr, betas=(train_cfg.beta1, train_cfg.beta2), eps=train_cfg.eps)

    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))

    # Data — hold out last N shards for validation
    all_shards = sorted(Path(train_cfg.data_dir).glob("*.bin"))
    n_shards = len(all_shards)
    n_eval = min(train_cfg.eval_hold_shards, max(1, n_shards // 10))
    train_range = (0, n_shards - n_eval)
    eval_range = (n_shards - n_eval, n_shards)
    print(f"Data split: {train_range[1]} train shards, {n_eval} eval shards")

    train_dataset = ByteShardDataset(train_cfg.data_dir, train_cfg.seq_len_bytes,
                                     model_cfg.patch_size, shard_range=train_range)
    eval_dataset = ByteShardDataset(train_cfg.data_dir, train_cfg.seq_len_bytes,
                                    model_cfg.patch_size, shard_range=eval_range)
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
    if train_cfg.resume_from:
        ckpt = torch.load(train_cfg.resume_from, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        if "rng_state" in ckpt:
            torch.set_rng_state(ckpt["rng_state"])
            if device.type == "cuda" and "cuda_rng_state" in ckpt:
                torch.cuda.set_rng_state(ckpt["cuda_rng_state"])
        start_step = ckpt["step"] + 1
        print(f"Resumed from step {start_step}")

    # Training loop
    model.train()
    data_iter = iter(train_loader)
    step = start_step
    accum_loss = 0.0
    accum_steps = 0
    best_eval_bpb = float("inf")
    train_start = time.time()
    t0 = time.time()

    while step < train_cfg.total_steps:
        # LR schedule
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

            with torch.amp.autocast("cuda", dtype=amp_dtype):
                out = model(byte_ids)
                losses = compute_loss(out, byte_ids, model_cfg.patch_size)
                loss = losses["loss"] / train_cfg.grad_accum_steps

            scaler.scale(loss).backward()
            accum_loss += losses["byte_ce"]
            accum_steps += 1

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)

        # R11 hard-fail: NaN/Inf detection
        if not math.isfinite(grad_norm.item()):
            fail_entry = {"step": step + 1, "HARD_FAIL": "non-finite grad_norm", "grad_norm": grad_norm.item()}
            log_f.write(json.dumps(fail_entry) + "\n")
            log_f.flush()
            raise RuntimeError(f"HARD FAIL at step {step + 1}: grad_norm={grad_norm.item()}")

        scaler.step(optimizer)
        scaler.update()

        step += 1

        # R11 hard-fail: NaN loss
        if accum_steps > 0 and not math.isfinite(accum_loss):
            fail_entry = {"step": step, "HARD_FAIL": "non-finite loss", "accum_loss": accum_loss}
            log_f.write(json.dumps(fail_entry) + "\n")
            log_f.flush()
            raise RuntimeError(f"HARD FAIL at step {step}: loss is non-finite")

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
            print(f"step {step:>6d} | loss {avg_loss:.4f} | bpb {bpb:.3f} | lr {lr:.2e} | gnorm {grad_norm:.2f} | {tokens_per_sec:.0f} tok/s")

            accum_loss = 0.0
            accum_steps = 0
            t0 = time.time()

        # Eval
        if step % train_cfg.eval_every == 0:
            eval_metrics = evaluate(model, eval_loader, device, train_cfg)
            eval_entry = {"step": step, **eval_metrics}
            log_f.write(json.dumps(eval_entry) + "\n")
            log_f.flush()
            pos_str = " ".join(f"p{i}={a:.3f}" for i, a in enumerate(eval_metrics.get("eval_pos_acc", [])))
            print(f"  EVAL step {step}: loss {eval_metrics['eval_loss']:.4f} bpb {eval_metrics['eval_bpb']:.3f} acc {eval_metrics.get('eval_byte_acc', 0):.4f} | {pos_str}")

            if eval_metrics["eval_bpb"] < best_eval_bpb:
                best_eval_bpb = eval_metrics["eval_bpb"]
                best_path = os.path.join(train_cfg.checkpoint_dir, "s0_best.pt")
                best_ckpt = {
                    "step": step,
                    "model": model.state_dict(),
                    "model_cfg": model_cfg,
                    "eval_bpb": best_eval_bpb,
                }
                torch.save(best_ckpt, best_path)
                print(f"  New best eval BPB {best_eval_bpb:.3f} — saved {best_path}")

        # Checkpoint
        if step % train_cfg.checkpoint_every == 0:
            ckpt_path = os.path.join(train_cfg.checkpoint_dir, f"s0_step{step}.pt")
            ckpt_data = {
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "model_cfg": model_cfg,
                "train_cfg": train_cfg,
                "rng_state": torch.get_rng_state(),
            }
            if device.type == "cuda":
                ckpt_data["cuda_rng_state"] = torch.cuda.get_rng_state()
            torch.save(ckpt_data, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    log_f.close()
    total_time = time.time() - train_start
    hours = total_time / 3600
    steps_done = step - start_step
    print(f"Training complete. {steps_done} steps in {hours:.2f}h ({steps_done/total_time:.1f} steps/s). Best eval BPB: {best_eval_bpb:.3f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="S0 training")
    parser.add_argument("--burnin", action="store_true",
                        help="Short 500-step burn-in with frequent checkpoints")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--config", choices=["p4", "p8", "d640", "d768"], default="p4")
    args = parser.parse_args()

    from s0_configs import ALL_CONFIGS
    model_cfg = ALL_CONFIGS[args.config]()
    train_cfg = TrainConfig()

    if args.burnin:
        train_cfg.total_steps = 500
        train_cfg.warmup_steps = 100  # R11: reach target LR by step 100
        train_cfg.checkpoint_every = 100
        train_cfg.eval_every = 50
        train_cfg.log_every = 5
        train_cfg.checkpoint_dir = "checkpoints/s0_burnin"
        train_cfg.log_file = "logs/s0_burnin.jsonl"
    if args.data_dir:
        train_cfg.data_dir = args.data_dir
    if args.resume:
        train_cfg.resume_from = args.resume
    if args.steps:
        train_cfg.total_steps = args.steps

    train(model_cfg, train_cfg)
