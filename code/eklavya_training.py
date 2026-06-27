"""Eklavya E1 — Knowledge distillation training for S0.

Three-phase training schedule:
  E1.0: Projection warmup (500 steps, freeze S0, train align_proj only)
  E1.1: Alignment landing (1500 steps, unfreeze encoder, CE + align)
  E1.2: Full Eklavya (10K+ steps, unfreeze all, CE + align + byte KL)

Usage:
    python eklavya_training.py \
        --student-checkpoint checkpoints/s0/s0_best.pt \
        --cache-dir eklavya_cache \
        --output-dir checkpoints/e1
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from s0_architecture import SutraS0
from s0_training import ByteShardDataset
from eklavya_cache import load_cache, AlignRecord, ByteKLRecord


class EklavyaDataset(ByteShardDataset):
    def __init__(self, data_dir: str, seq_len: int, patch_size: int = 4,
                 shard_range: tuple[int, int] | None = None):
        super().__init__(data_dir, seq_len, patch_size, shard_range)
        self._shard_offset = shard_range[0] if shard_range is not None else 0

    def __getitem__(self, idx):
        import bisect
        shard_idx = bisect.bisect_right(self.cumulative, idx) - 1
        local_idx = idx - self.cumulative[shard_idx]
        mm = self._get_mmap(shard_idx)
        start = local_idx * self.seq_len
        raw = mm[start:start + self.seq_len]
        byte_ids = torch.frombuffer(bytearray(raw), dtype=torch.uint8).long()
        global_shard_id = shard_idx + self._shard_offset
        return byte_ids, global_shard_id, start


@dataclass
class EklavyaConfig:
    lambda_align: float = 0.05
    lambda_kl: float = 0.10
    kl_temperature: float = 2.0

    base_lr: float = 3e-5
    align_lr: float = 3e-4
    weight_decay_base: float = 0.05
    weight_decay_proj: float = 0.01

    projection_warmup_steps: int = 500
    alignment_landing_steps: int = 1500
    full_e1_steps: int = 10000

    batch_size: int = 4
    seq_len: int = 4096
    grad_accum: int = 2
    max_grad_norm: float = 1.0
    teacher_grad_budget: float = 0.30

    checkpoint_dir: str = "checkpoints/e1"
    log_file: str = "logs/e1_train.jsonl"
    checkpoint_every: int = 1000
    eval_every: int = 200
    log_every: int = 10

    cache_refresh_every: int = 2000
    cache_dir: str = "eklavya_cache"

    data_dir: str = "data/shards_bytes_full"

    eval_batches: int = 50
    resume_from: str | None = None
    consecutive_ce_only_threshold: int = 200


class AlignProjection(nn.Module):
    def __init__(self, student_dim: int = 576, teacher_dim: int = 2048):
        super().__init__()
        self.norm = nn.LayerNorm(student_dim)
        self.proj = nn.Linear(student_dim, teacher_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(x))


@torch.no_grad()
def evaluate_e1(student: SutraS0, align_proj: AlignProjection, eval_loader,
                device: torch.device, amp_dtype: torch.dtype,
                use_cuda: bool, max_batches: int = 50) -> dict:
    student.eval()
    align_proj.eval()
    total_loss = 0.0
    total_tokens = 0
    P = student.cfg.patch_size
    for i, batch in enumerate(eval_loader):
        if i >= max_batches:
            break
        byte_ids = batch[0].to(device)
        B, T = byte_ids.shape
        N = T // P
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_cuda):
            out = student(byte_ids)
            logits = out["logits"]
            targets = byte_ids.reshape(B, N, P)[:, 1:]
            loss = F.cross_entropy(logits.reshape(-1, 256), targets.reshape(-1))
        predicted = B * (T - P)
        total_loss += loss.item() * predicted
        total_tokens += predicted
    student.train()
    align_proj.train()
    if total_tokens == 0:
        return {"eval_loss": float("inf"), "eval_bpb": float("inf")}
    avg_loss = total_loss / total_tokens
    return {"eval_loss": avg_loss, "eval_bpb": avg_loss / math.log(2)}


def overlap_pool(patch_states: torch.Tensor, byte_start: int, byte_end: int,
                 P: int = 4) -> torch.Tensor:
    p0 = byte_start // P
    p1 = (byte_end - 1) // P
    N = patch_states.shape[0] if patch_states.dim() == 2 else patch_states.shape[1]

    pieces = []
    weights = []
    for p in range(p0, min(p1 + 1, N)):
        pa = p * P
        pb = pa + P
        overlap = max(0, min(byte_end, pb) - max(byte_start, pa))
        if overlap > 0:
            if patch_states.dim() == 2:
                pieces.append(patch_states[p])
            else:
                pieces.append(patch_states[0, p])
            weights.append(float(overlap))

    if not pieces:
        return None

    w = torch.tensor(weights, device=patch_states.device, dtype=patch_states.dtype)
    w = w / w.sum()
    return sum(w[i] * pieces[i] for i in range(len(pieces)))


def topk_tail_kl(student_logits: torch.Tensor, top_bytes: torch.Tensor,
                 top_probs: torch.Tensor, tail_prob: torch.Tensor,
                 T: float = 2.0) -> torch.Tensor:
    logp = F.log_softmax(student_logits / T, dim=-1)
    p = F.softmax(student_logits / T, dim=-1)

    idx = top_bytes.long()
    logp_top = logp[idx]
    p_top_sum = p[idx].sum()
    p_tail = (1.0 - p_top_sum).clamp(min=1e-8)

    loss_top = -(top_probs * logp_top).sum()
    loss_tail = -tail_prob * torch.log(p_tail)

    return (T * T) * (loss_top + loss_tail)


def _grad_norm(params):
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += p.grad.detach().float().norm().item() ** 2
    return total ** 0.5


def apply_gradient_budget(params, L_ce, L_teacher, budget: float, scaler=None):
    """Two-pass backward with teacher gradient budget.

    Ensures teacher gradient norm ≤ budget × CE gradient norm.
    Preserves previously accumulated gradients for grad_accum > 1.
    Uses scaler.scale() when scaler is provided for CUDA AMP compatibility.
    Returns (ce_grad_norm, teacher_grad_norm, scale_applied).
    """
    saved = {}
    for p in params:
        if p.grad is not None:
            saved[id(p)] = p.grad.detach().clone()
            p.grad = None

    if scaler is not None:
        scaler.scale(L_ce).backward(retain_graph=True)
    else:
        L_ce.backward(retain_graph=True)

    ce_grads = {}
    for p in params:
        if p.grad is not None:
            ce_grads[id(p)] = p.grad.clone()
            p.grad = None

    if scaler is not None:
        scaler.scale(L_teacher).backward()
    else:
        L_teacher.backward()

    ce_norm = sum(g.float().norm().item() ** 2 for g in ce_grads.values()) ** 0.5
    teacher_norm = _grad_norm(params)

    scale = 1.0
    if teacher_norm > budget * ce_norm and ce_norm > 0:
        scale = budget * ce_norm / teacher_norm
        for p in params:
            if p.grad is not None:
                p.grad.mul_(scale)

    for p in params:
        parts = []
        if id(p) in saved:
            parts.append(saved[id(p)])
        if id(p) in ce_grads:
            parts.append(ce_grads[id(p)])
        for g in parts:
            if p.grad is not None:
                p.grad.add_(g)
            else:
                p.grad = g.clone()

    return ce_norm, teacher_norm, scale


class EklavyaTrainer:
    def __init__(self, cfg: EklavyaConfig, student: SutraS0,
                 align_proj: AlignProjection, cache: dict,
                 device: torch.device):
        self.cfg = cfg
        self.student = student
        self.align_proj = align_proj
        self.cache = cache
        self.device = device

        self.embedding_table = cache["embedding_table"]
        if self.embedding_table is not None:
            self.embedding_table = self.embedding_table.to(dtype=torch.bfloat16)

        self.align_records = cache["align_records"]
        self.kl_records = cache["kl_records"]

        self._index_records()

    def _index_records(self):
        self.align_by_seq = {}
        for r in self.align_records:
            key = (r.shard_id, r.seq_offset)
            self.align_by_seq.setdefault(key, []).append(r)

        self.kl_by_seq = {}
        for r in self.kl_records:
            key = (r.shard_id, r.seq_offset)
            self.kl_by_seq.setdefault(key, []).append(r)

    def compute_align_loss(self, patch_states: torch.Tensor,
                           seq_align_records: list[AlignRecord]) -> torch.Tensor:
        if not seq_align_records or self.embedding_table is None:
            return torch.tensor(0.0, device=self.device)

        P = self.student.cfg.patch_size
        losses = []

        for r in seq_align_records:
            if r.byte_start + r.byte_len > patch_states.shape[1] * P:
                continue
            student_span = overlap_pool(patch_states, r.byte_start,
                                        r.byte_start + r.byte_len, P)
            if student_span is None:
                continue
            z_s = F.layer_norm(self.align_proj(student_span), (self.align_proj.proj.out_features,))

            teacher_emb = self.embedding_table[r.token_id].to(device=self.device, dtype=z_s.dtype)
            z_t = F.layer_norm(teacher_emb, (teacher_emb.shape[-1],))

            losses.append(F.mse_loss(z_s, z_t))

        if not losses:
            return torch.tensor(0.0, device=self.device)
        return torch.stack(losses).mean()

    def compute_kl_loss(self, logits: torch.Tensor,
                        seq_kl_records: list[ByteKLRecord]) -> torch.Tensor:
        if not seq_kl_records:
            return torch.tensor(0.0, device=self.device)

        losses = []
        N_minus_1 = logits.shape[1]

        for r in seq_kl_records:
            logit_idx = r.patch_idx - 1
            if logit_idx < 0 or logit_idx >= N_minus_1:
                continue

            student_logit = logits[0, logit_idx, 0]  # position 0 within patch

            top_b = torch.from_numpy(r.top_bytes).to(self.device)
            top_p = torch.from_numpy(r.top_probs.astype(np.float32)).to(self.device)
            tail_p = torch.tensor(r.tail_prob, device=self.device, dtype=torch.float32)

            loss = topk_tail_kl(student_logit, top_b, top_p, tail_p,
                                T=self.cfg.kl_temperature)
            losses.append(loss)

        if not losses:
            return torch.tensor(0.0, device=self.device)
        return torch.stack(losses).mean()

    def refresh_cache(self, cache_dir: str):
        """Reload cache from disk (after external refresh process updates it)."""
        updated = load_cache(cache_dir)
        self.align_records = updated["align_records"]
        self.kl_records = updated["kl_records"]
        if updated["embedding_table"] is not None:
            self.embedding_table = updated["embedding_table"].to(dtype=torch.bfloat16)
        else:
            self.embedding_table = None
        self._index_records()
        print(f"  Cache refreshed: {len(self.align_records)} align, "
              f"{len(self.kl_records)} kl records")

    def get_phase(self, step: int) -> str:
        if step < self.cfg.projection_warmup_steps:
            return "E1.0_warmup"
        elif step < self.cfg.projection_warmup_steps + self.cfg.alignment_landing_steps:
            return "E1.1_landing"
        else:
            return "E1.2_full"

    def configure_freeze(self, phase: str):
        if phase == "E1.0_warmup":
            for p in self.student.parameters():
                p.requires_grad = False
            for p in self.align_proj.parameters():
                p.requires_grad = True

        elif phase == "E1.1_landing":
            for p in self.student.parameters():
                p.requires_grad = False
            for p in self.student.encoder.parameters():
                p.requires_grad = True
            for p in self.align_proj.parameters():
                p.requires_grad = True

        elif phase == "E1.2_full":
            for p in self.student.parameters():
                p.requires_grad = True
            for p in self.align_proj.parameters():
                p.requires_grad = True

    def build_optimizer(self):
        proj_params = list(self.align_proj.parameters())
        proj_ids = {id(p) for p in proj_params}

        base_params = [p for p in self.student.parameters()
                       if p.requires_grad and id(p) not in proj_ids]

        groups = []
        if base_params:
            groups.append({
                "params": base_params,
                "lr": self.cfg.base_lr,
                "weight_decay": self.cfg.weight_decay_base,
            })
        groups.append({
            "params": proj_params,
            "lr": self.cfg.align_lr,
            "weight_decay": self.cfg.weight_decay_proj,
        })

        return torch.optim.AdamW(groups, betas=(0.9, 0.95))


def train_e1(cfg: EklavyaConfig, student_ckpt_path: str, cache_dir: str):
    use_cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(student_ckpt_path, map_location="cpu", weights_only=False)
    model_cfg = ckpt["model_cfg"]
    student = SutraS0(model_cfg).to(device)
    student.load_state_dict(ckpt["model"])
    print(f"Loaded S0 from step {ckpt.get('step', '?')}")

    teacher_dim = 2048
    cache = load_cache(cache_dir)
    if cache["embedding_table"] is not None:
        teacher_dim = cache["embedding_table"].shape[1]
    print(f"Cache: {cache['manifest']}")

    align_proj = AlignProjection(model_cfg.d_model, teacher_dim).to(device)

    manifest = cache["manifest"]
    cache_shard_range = manifest.get("shard_range")
    all_shards = sorted(Path(cfg.data_dir).glob("*.bin"))

    if cache_shard_range is not None:
        cache_start, cache_end = cache_shard_range
        n_cached = cache_end - cache_start
        if n_cached < 2:
            raise ValueError(
                f"Cache covers only {n_cached} shards ({cache_start}-{cache_end}). "
                f"Need at least 2 (1 train + 1 eval).")
        n_eval = min(2, max(1, n_cached // 10))
        train_range = (cache_start, cache_end - n_eval)
        eval_range = (cache_end - n_eval, cache_end)
        print(f"  Cache shard range: [{cache_start}, {cache_end}), "
              f"train [{train_range[0]}, {train_range[1]}), "
              f"eval [{eval_range[0]}, {eval_range[1]})")
    else:
        n_eval = min(2, max(1, len(all_shards) // 10))
        train_range = (0, len(all_shards) - n_eval)
        eval_range = (len(all_shards) - n_eval, len(all_shards))
        print("  WARNING: Cache has no shard_range — training on all shards. "
              "Uncached shards will get CE-only signal.")

    train_dataset = EklavyaDataset(cfg.data_dir, cfg.seq_len,
                                   model_cfg.patch_size, shard_range=train_range)
    eval_dataset = EklavyaDataset(cfg.data_dir, cfg.seq_len,
                                  model_cfg.patch_size, shard_range=eval_range)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size,
                              shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    eval_loader = DataLoader(eval_dataset, batch_size=cfg.batch_size,
                             shuffle=False, num_workers=1, pin_memory=True, drop_last=True)

    trainer = EklavyaTrainer(cfg, student, align_proj, cache, device)

    total_steps = (cfg.projection_warmup_steps + cfg.alignment_landing_steps
                   + cfg.full_e1_steps)

    os.makedirs(os.path.dirname(cfg.log_file), exist_ok=True)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    amp_dtype = torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16)) if use_cuda else None
    step = 0
    current_phase = None
    optimizer = None
    best_eval_bpb = float("inf")
    consecutive_ce_only = 0
    log_fh = open(cfg.log_file, "a")

    if cfg.resume_from and os.path.exists(cfg.resume_from):
        resume_ckpt = torch.load(cfg.resume_from, map_location="cpu",
                                 weights_only=False)
        student.load_state_dict(resume_ckpt["model"])
        align_proj.load_state_dict(resume_ckpt["align_proj"])
        step = resume_ckpt["step"] + 1
        if "best_eval_bpb" in resume_ckpt:
            best_eval_bpb = resume_ckpt["best_eval_bpb"]
        resumed_phase = resume_ckpt.get("phase")
        if resumed_phase is not None:
            current_phase = resumed_phase
            trainer.configure_freeze(current_phase)
            optimizer = trainer.build_optimizer()
            if "optimizer" in resume_ckpt and resume_ckpt["optimizer"] is not None:
                optimizer.load_state_dict(resume_ckpt["optimizer"])
        if scaler is not None and "scaler" in resume_ckpt and resume_ckpt["scaler"] is not None:
            scaler.load_state_dict(resume_ckpt["scaler"])
        print(f"Resumed from step {step} phase {current_phase} "
              f"(best eval BPB: {best_eval_bpb:.3f})")

    P = model_cfg.patch_size
    t0 = time.time()
    data_iter = iter(train_loader)

    print(f"Starting E1 training: {total_steps} total steps")
    print(f"  E1.0 warmup: {cfg.projection_warmup_steps} steps")
    print(f"  E1.1 landing: {cfg.alignment_landing_steps} steps")
    print(f"  E1.2 full: {cfg.full_e1_steps} steps")

    while step < total_steps:
        phase = trainer.get_phase(step)
        if phase != current_phase:
            print(f"\n=== Phase: {phase} (step {step}) ===")
            trainer.configure_freeze(phase)
            optimizer = trainer.build_optimizer()
            current_phase = phase

        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        byte_ids, shard_ids, seq_offsets = batch
        byte_ids = byte_ids.to(device)

        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_cuda):
            out = student(byte_ids)
            logits = out["logits"]
            B, Nm1, Pp, V = logits.shape
            N = Nm1 + 1

            targets = byte_ids.reshape(B, N, P)[:, 1:]
            L_ce = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1))

            L_align = torch.tensor(0.0, device=device)
            L_kl = torch.tensor(0.0, device=device)

            if phase in ("E1.0_warmup", "E1.1_landing", "E1.2_full"):
                align_losses = []
                for b in range(B):
                    key = (shard_ids[b].item(), seq_offsets[b].item())
                    records = trainer.align_by_seq.get(key, [])
                    if records:
                        sampled = random.sample(records, min(32, len(records)))
                        al = trainer.compute_align_loss(
                            out["patch_states"][b:b+1], sampled)
                        align_losses.append(al)
                if align_losses:
                    L_align = torch.stack(align_losses).mean()

            if phase == "E1.2_full":
                kl_losses = []
                for b in range(B):
                    key = (shard_ids[b].item(), seq_offsets[b].item())
                    records = trainer.kl_by_seq.get(key, [])
                    if records:
                        sampled = random.sample(records, min(16, len(records)))
                        kl = trainer.compute_kl_loss(logits[b:b+1], sampled)
                        kl_losses.append(kl)
                if kl_losses:
                    L_kl = torch.stack(kl_losses).mean()

            L_teacher = cfg.lambda_align * L_align + cfg.lambda_kl * L_kl

            if phase == "E1.0_warmup":
                loss = L_align
            elif phase == "E1.1_landing":
                loss = L_ce + cfg.lambda_align * L_align
            else:
                loss = L_ce + L_teacher

        if not math.isfinite(loss.item()):
            fail_entry = {"step": step, "phase": phase,
                          "HARD_FAIL": "non-finite loss",
                          "loss": loss.item()}
            log_fh.write(json.dumps(fail_entry) + "\n")
            log_fh.flush()
            log_fh.close()
            raise RuntimeError(
                f"E1 HARD FAIL: non-finite loss at step {step} "
                f"(phase={phase})")

        trainable_params = [
            p for p in list(student.parameters()) + list(align_proj.parameters())
            if p.requires_grad
        ]

        has_teacher_signal = (L_align.item() > 0 or L_kl.item() > 0) and phase != "E1.0_warmup"

        expects_teacher = phase in ("E1.1_landing", "E1.2_full")
        if expects_teacher and not has_teacher_signal:
            consecutive_ce_only += 1
            if consecutive_ce_only >= cfg.consecutive_ce_only_threshold:
                log_fh.close()
                raise RuntimeError(
                    f"E1 received NO teacher signal for {consecutive_ce_only} "
                    f"consecutive steps (phase={phase}, step={step}). "
                    f"Likely cache/shard mismatch — aborting.")
        else:
            consecutive_ce_only = 0

        if has_teacher_signal and cfg.teacher_grad_budget > 0:
            apply_gradient_budget(
                trainable_params, L_ce / cfg.grad_accum,
                L_teacher / cfg.grad_accum, cfg.teacher_grad_budget,
                scaler=scaler,
            )
        else:
            scaled_loss = loss / cfg.grad_accum
            if scaler is not None:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

        if (step + 1) % cfg.grad_accum == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(
                trainable_params, cfg.max_grad_norm)
            if not math.isfinite(grad_norm.item()):
                fail_entry = {"step": step, "phase": phase,
                              "HARD_FAIL": "non-finite grad_norm",
                              "grad_norm": grad_norm.item()}
                log_fh.write(json.dumps(fail_entry) + "\n")
                log_fh.flush()
                log_fh.close()
                raise RuntimeError(
                    f"E1 HARD FAIL: non-finite grad_norm at step {step} "
                    f"(phase={phase})")
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if step % cfg.log_every == 0:
            bpb = L_ce.item() / math.log(2)
            entry = {
                "step": step, "phase": phase,
                "loss": loss.item(), "ce": L_ce.item(), "bpb": bpb,
                "align": L_align.item(), "kl": L_kl.item(),
                "elapsed": time.time() - t0,
            }
            log_fh.write(json.dumps(entry) + "\n")
            log_fh.flush()

            if step % (cfg.log_every * 10) == 0:
                print(f"  step {step:5d} | phase={phase} | bpb={bpb:.3f} | "
                      f"align={L_align.item():.4f} | kl={L_kl.item():.4f}")

        if step > 0 and step % cfg.eval_every == 0:
            eval_metrics = evaluate_e1(student, align_proj, eval_loader,
                                       device, amp_dtype, use_cuda,
                                       max_batches=cfg.eval_batches)
            eval_entry = {"step": step, "phase": phase, **eval_metrics,
                          "elapsed": time.time() - t0}
            log_fh.write(json.dumps(eval_entry) + "\n")
            log_fh.flush()

            if eval_metrics["eval_bpb"] < best_eval_bpb:
                best_eval_bpb = eval_metrics["eval_bpb"]
                best_path = os.path.join(cfg.checkpoint_dir, "e1_best.pt")
                torch.save({
                    "step": step,
                    "phase": phase,
                    "model": student.state_dict(),
                    "model_cfg": model_cfg,
                    "align_proj": align_proj.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict() if scaler is not None else None,
                    "best_eval_bpb": best_eval_bpb,
                    "config": cfg,
                }, best_path)
                print(f"  eval bpb={eval_metrics['eval_bpb']:.3f} "
                      f"[NEW BEST] — saved {best_path}")
            else:
                print(f"  eval bpb={eval_metrics['eval_bpb']:.3f} "
                      f"(best={best_eval_bpb:.3f})")

        if (step > 0 and cfg.cache_refresh_every > 0
                and step % cfg.cache_refresh_every == 0
                and phase == "E1.2_full"):
            if os.path.exists(os.path.join(cfg.cache_dir, "cache_manifest.json")):
                trainer.refresh_cache(cfg.cache_dir)

        if step > 0 and step % cfg.checkpoint_every == 0:
            ckpt_path = os.path.join(cfg.checkpoint_dir, f"e1_step{step}.pt")
            torch.save({
                "step": step,
                "phase": phase,
                "model": student.state_dict(),
                "model_cfg": model_cfg,
                "align_proj": align_proj.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None,
                "best_eval_bpb": best_eval_bpb,
                "config": cfg,
            }, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

        step += 1

    log_fh.close()
    final_path = os.path.join(cfg.checkpoint_dir, "e1_final.pt")
    torch.save({
        "step": step,
        "phase": phase,
        "model": student.state_dict(),
        "model_cfg": model_cfg,
        "align_proj": align_proj.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "best_eval_bpb": best_eval_bpb,
        "config": cfg,
    }, final_path)
    print(f"\nE1 training complete. Final checkpoint: {final_path}")
    print(f"Best eval BPB: {best_eval_bpb:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Eklavya E1 distillation training")
    parser.add_argument("--student-checkpoint", required=True)
    parser.add_argument("--cache-dir", default="eklavya_cache")
    parser.add_argument("--output-dir", default="checkpoints/e1")
    parser.add_argument("--data-dir", default="data/shards_bytes_full")
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--resume-from", default=None,
                        help="Path to E1 checkpoint to resume from")
    args = parser.parse_args()

    cfg = EklavyaConfig(
        checkpoint_dir=args.output_dir,
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        full_e1_steps=args.steps - 2000,
        resume_from=args.resume_from,
    )

    train_e1(cfg, args.student_checkpoint, args.cache_dir)


if __name__ == "__main__":
    main()
