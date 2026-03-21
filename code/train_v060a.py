"""Sutra v0.6.0a Training: Probe-only dense-12 from scratch.

Tests: does 12-pass inter-step supervision create convergence separation?
Trains from scratch on diverse 17B+ token corpus.

Loss = L_final + 0.25 * L_step + 0.20 * L_probe

L_final: full-vocab CE on final pass
L_step: weighted sampled CE over all 12 passes
L_probe: SmoothL1 on residual-gain prediction (shadow, doesn't act)

Usage: python code/train_v060a.py
"""

import json, math, os, random, time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REPO = Path(__file__).parent.parent

DIM = 768
FF_DIM = 1536
MAX_STEPS = 12
WINDOW = 4
K_RETRIEVAL = 8
SEQ_LEN = 512
BATCH_SIZE = 4          # Reduced from 8: 12 passes + history = more VRAM
GRAD_ACCUM = 16         # Effective batch = 64 (same)
LR = 3.5e-4  # Reduced for attached-history (audit loop 1: 4.5e-4 not validated)
WARMUP_STEPS = 1500
MAX_TRAIN_STEPS = 100000
EVAL_EVERY = 1000
SAVE_EVERY = 5000
ROLLING_SAVE = 100
LOG_EVERY = 100
VOCAB_SIZE = 50257
STEP_LOSS_COEF = 0.25
PROBE_LOSS_COEF = 0.20
GRAD_CLIP_NORM = 0.5

# Step weights for sampled inter-step CE
STEP_WEIGHTS = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.00]

import sys
sys.path.insert(0, str(REPO / "code"))
from launch_v060a import create_v060a
from data_loader import ShardedDataset


def autocast_ctx():
    if DEVICE.type == "cuda":
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def atomic_torch_save(obj, path):
    path = Path(path)
    tmp = path.with_name(f"{path.stem}.tmp")
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def compute_v060a_losses(logits, aux, y):
    """Three-part loss: final CE + weighted step CE + probe calibration."""
    B, T, V = logits.shape

    # 1. L_final: full-vocab CE on final pass
    L_final = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))

    # 2. L_step: weighted sampled CE over all passes
    sampled_ce = aux["sampled_ce_hist"]  # (B, T, 12)
    weights = torch.tensor(STEP_WEIGHTS, device=sampled_ce.device, dtype=sampled_ce.dtype)
    L_step = (sampled_ce * weights.unsqueeze(0).unsqueeze(0)).mean()

    # 3. L_probe: residual-gain prediction
    probe_pred = aux["probe_pred"]  # (B, T, 12)
    # Target: max(0, ce_p - min(ce_{p+1..11}))
    with torch.no_grad():
        targets = torch.zeros_like(sampled_ce)
        for p in range(11):
            future_min = sampled_ce[:, :, p+1:].min(dim=2).values
            targets[:, :, p] = (sampled_ce[:, :, p] - future_min).clamp(min=0, max=4)
        # Final pass target = 0 (already initialized)
    L_probe = F.smooth_l1_loss(probe_pred, targets.detach())

    L_total = L_final + STEP_LOSS_COEF * L_step + PROBE_LOSS_COEF * L_probe

    return L_total, {
        "L_final": L_final.item(),
        "L_step": L_step.item(),
        "L_probe": L_probe.item(),
        "L_total": L_total.item(),
    }


def evaluate(model, dataset, n_batches=20):
    model.eval()
    total_loss = 0
    try:
        with torch.no_grad():
            for _ in range(n_batches):
                x, y = dataset.sample_batch(min(BATCH_SIZE, 4), SEQ_LEN, device=DEVICE, split="test")
                with autocast_ctx():
                    logits, _ = model(x)
                    Tc = min(logits.size(1), y.size(1))
                    loss = F.cross_entropy(logits[:, :Tc].reshape(-1, VOCAB_SIZE), y[:, :Tc].reshape(-1))
                total_loss += loss.item()
    finally:
        model.train()  # MUST restore train mode even on exception
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {"bpt": (total_loss / n_batches) / math.log(2)}


def main():
    print(f"SUTRA v0.6.0a: PROBE-ONLY DENSE-12 FROM SCRATCH")
    print(f"  12 passes, inter-step supervision, residual-gain probe")
    print(f"Device: {DEVICE}, bf16: True")
    print(f"Training: bs={BATCH_SIZE}x{GRAD_ACCUM}={BATCH_SIZE*GRAD_ACCUM}, seq={SEQ_LEN}")
    print(f"{'='*60}")

    dataset = ShardedDataset()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    ckpt_dir = REPO / "results" / "checkpoints_v060a"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_file = REPO / "results" / "v060a_log.txt"
    metrics_file = REPO / "results" / "v060a_metrics.json"

    # Check for resume
    rolling = ckpt_dir / "rolling_latest.pt"
    permanent = sorted(ckpt_dir.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))

    start_step = 0
    best_bpt = float("inf")
    metrics_history = []

    model = create_v060a(dim=DIM, ff_dim=FF_DIM, max_steps=MAX_STEPS,
                         window=WINDOW, k_retrieval=K_RETRIEVAL).to(DEVICE)

    # Pick best checkpoint: try rolling + ALL permanents in descending step order
    ckpt = None
    resume_candidates = []
    if rolling.exists():
        resume_candidates.append(rolling)
    # Add ALL permanent checkpoints (descending by step) for robustness
    for p in reversed(permanent):
        resume_candidates.append(p)

    for cand in resume_candidates:
        try:
            c = torch.load(cand, weights_only=False, map_location=DEVICE)
            if ckpt is None or c.get("step", 0) > ckpt.get("step", 0):
                ckpt = c
                ckpt["_path"] = cand.name
        except Exception as e:
            print(f"Skip corrupt checkpoint {cand.name}: {e}")

    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
        start_step = ckpt["step"]
        best_bpt = ckpt.get("best_bpt", float("inf"))
        metrics_history = ckpt.get("metrics", [])
        print(f"RESUMED from step {start_step} ({ckpt['_path']})")
    else:
        print("Training from SCRATCH (v0.6.0a)")

    print(f"Params: {model.count_params():,} ({model.count_params()/1e6:.1f}M)")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01, betas=(0.9, 0.95))
    if ckpt is not None and "optimizer" in ckpt:
        opt.load_state_dict(ckpt["optimizer"])

    model.train()
    step = start_step
    running_losses = {"L_final": 0, "L_step": 0, "L_probe": 0, "L_total": 0}
    loss_count = 0
    start = time.time()

    while step < MAX_TRAIN_STEPS:
        x, y = dataset.sample_batch(BATCH_SIZE, SEQ_LEN, device=DEVICE, split="train")

        lr = LR * min(1.0, (step + 1) / WARMUP_STEPS) * (
            0.5 * (1 + math.cos(math.pi * max(0, step - WARMUP_STEPS)
                                 / max(1, MAX_TRAIN_STEPS - WARMUP_STEPS))))
        for pg in opt.param_groups:
            pg["lr"] = lr

        with autocast_ctx():
            logits, aux = model(x, y=y)
            Tc = min(logits.size(1), y.size(1))
            loss, loss_parts = compute_v060a_losses(logits[:, :Tc], aux, y[:, :Tc])
            loss = loss / GRAD_ACCUM

        if not torch.isfinite(loss):
            print(f"WARNING: Non-finite loss at step {step}, skipping microbatch", flush=True)
            opt.zero_grad()
            loss_count = 0
            running_losses = {k: 0 for k in running_losses}
            continue

        loss.backward()
        for k, v in loss_parts.items():
            running_losses[k] += v
        loss_count += 1

        if loss_count % GRAD_ACCUM == 0:
            if not torch.isfinite(loss) or any(
                p.grad is not None and not torch.isfinite(p.grad).all()
                for p in model.parameters()
            ):
                print(f"WARNING: NaN/Inf at step {step}, skipping", flush=True)
                opt.zero_grad()
                loss_count = 0
                running_losses = {k: 0 for k in running_losses}
                continue

            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            opt.step()
            opt.zero_grad()
            step += 1

            if step % LOG_EVERY == 0:
                avgs = {k: v / loss_count for k, v in running_losses.items()}
                elapsed = time.time() - start
                tps = (step - start_step) * BATCH_SIZE * GRAD_ACCUM * SEQ_LEN / max(elapsed, 1)
                msg = (f"Step {step:>6d}: L={avgs['L_total']:.4f} "
                       f"(fin={avgs['L_final']:.3f} stp={avgs['L_step']:.3f} prb={avgs['L_probe']:.3f}) "
                       f"lr={lr:.2e} {tps:.0f}tok/s")
                print(msg, flush=True)
                with open(log_file, "a") as f:
                    f.write(msg + "\n")
                running_losses = {k: 0 for k in running_losses}
                loss_count = 0

            if step % ROLLING_SAVE == 0:
                atomic_torch_save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                                   "step": step, "best_bpt": best_bpt}, ckpt_dir / "rolling_latest.pt")

            if step % EVAL_EVERY == 0:
                try:
                    metrics = evaluate(model, dataset)
                    bpt = metrics["bpt"]
                    is_best = bpt < best_bpt
                    if is_best:
                        best_bpt = bpt
                        atomic_torch_save(model.state_dict(), REPO / "results" / "v060a_best.pt")
                    entry = {"step": step, "test_bpt": round(bpt, 4), "best_bpt": round(best_bpt, 4),
                             "is_best": is_best, "lr": lr, "timestamp": datetime.now().isoformat()}
                    metrics_history.append(entry)
                    json.dump(metrics_history, open(metrics_file, "w"), indent=2)
                    marker = " *BEST*" if is_best else ""
                    eval_msg = f"  EVAL Step {step}: BPT={bpt:.4f}{marker}"
                    print(eval_msg, flush=True)
                    with open(log_file, "a") as f:
                        f.write(eval_msg + "\n")
                except Exception as e:
                    print(f"EVAL FAILED: {e}", flush=True)

            if step % SAVE_EVERY == 0:
                atomic_torch_save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                                   "step": step, "best_bpt": best_bpt, "metrics": metrics_history},
                                  ckpt_dir / f"step_{step}.pt")

    print(f"\nDone. {step} steps, best BPT={best_bpt:.4f}")


if __name__ == "__main__":
    main()
